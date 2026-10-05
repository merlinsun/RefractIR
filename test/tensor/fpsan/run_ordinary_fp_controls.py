#!/usr/bin/env python3
"""Run IEEE-f32 controls separately from Triton's FpSan mode."""

import argparse
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from test.lib.tensor_fp32_reference import (
  fma_f32_rne_bits,
  mul_add_separate_f32_bits,
  mul_f32_rne_bits,
  reduction_f32_rne_outcomes,
  sub_f32_rne_bits,
)
from test.lib.tensor_fpsan import (
  f32_fmod_bits,
  rounding_reassociation_bits,
  verify_installed_fpsan_source,
)

TRITON_VERSION = "3.8.0"
WARMUP_LAUNCHES = 3
MEASURED_LAUNCHES = 30


@triton.jit
def rounding_trees_kernel(a_ptr, b_ptr, c_ptr, output_ptr):
  a = tl.load(a_ptr)
  b = tl.load(b_ptr)
  c = tl.load(c_ptr)
  tl.store(output_ptr, (a + b) + c)
  tl.store(output_ptr + 1, a + (b + c))


@triton.jit
def arithmetic_edge_kernel(
  a_ptr,
  b_ptr,
  c_ptr,
  product_ptr,
  expression_ptr,
):
  offsets = tl.arange(0, 4)
  a = tl.load(a_ptr + offsets)
  b = tl.load(b_ptr + offsets)
  c = tl.load(c_ptr + offsets)
  product = a * b
  tl.store(product_ptr + offsets, product)
  tl.store(expression_ptr + offsets, a * b + c)


@triton.jit
def reduction_order_kernel(input_ptr, output_ptr):
  offsets = tl.arange(0, 4)
  values = tl.load(input_ptr + offsets, offsets < 3, other=0.0)
  tl.store(output_ptr, tl.sum(values))


@triton.jit
def remainder_kernel(dividend_ptr, divisor_ptr, output_ptr):
  offsets = tl.arange(0, 2)
  dividend = tl.load(dividend_ptr + offsets)
  divisor = tl.load(divisor_ptr + offsets)
  tl.store(output_ptr + offsets, libdevice.fmod(dividend, divisor))


@triton.jit
def subtraction_edge_kernel(left_ptr, right_ptr, output_ptr):
  offsets = tl.arange(0, 8)
  left = tl.load(left_ptr + offsets)
  right = tl.load(right_ptr + offsets)
  tl.store(output_ptr + offsets, left - right)


def _raw_words(tensor):
  return [value & 0xFFFFFFFF for value in tensor.view(torch.int32).cpu().tolist()]


def _float_tensor_from_words(words):
  return torch.tensor(words, dtype=torch.uint32, device="cuda").view(torch.float32)


def _ptx_has_fma(ptx):
  return re.search(r"(?m)^\s*fma(?:\.[a-z0-9]+)*\.f32\b", ptx) is not None


def _ptx_has_ftz_arithmetic(ptx):
  return (
    re.search(r"(?m)^\s*(?:add|sub|mul|fma)\.[a-z0-9.]*ftz[a-z0-9.]*\.f32\b", ptx)
    is not None
  )


def _binary32_subtraction_reference(left_words, right_words):
  if (
    not isinstance(left_words, (tuple, list))
    or not isinstance(right_words, (tuple, list))
    or not left_words
    or len(left_words) != len(right_words)
  ):
    raise ValueError("subtraction operands must have the same nonzero length")
  if any(
    type(word) is not int or not 0 <= word <= 0xFFFFFFFF
    for word in (*left_words, *right_words)
  ):
    raise ValueError("subtraction operands must be unsigned binary32 words")
  return [
    sub_f32_rne_bits(left, right)
    for left, right in zip(left_words, right_words, strict=True)
  ]


def _run_binary32_edge_cases(ir_directory):
  input_words = {
    "a": (0x3F800001, 0x00000001, 0x80000000, 0x3F800001),
    "b": (0x3F7FFFFE, 0x40000000, 0x40000000, 0x3F7FFFFE),
    "c": (0x00000000, 0x00000000, 0x80000000, 0xBF800000),
  }
  a, b, c = (_float_tensor_from_words(input_words[name]) for name in ("a", "b", "c"))
  product_outputs = []
  expression_outputs = []
  compiled = {}
  measurements = {}
  for fusion_enabled in (False, True):
    product_output = torch.empty((4,), dtype=torch.float32, device="cuda")
    expression_output = torch.empty_like(product_output)
    compiled[fusion_enabled], measurements[fusion_enabled] = _measure_launches(
      lambda: arithmetic_edge_kernel[(1,)](
        a,
        b,
        c,
        product_output,
        expression_output,
        num_warps=1,
        enable_fp_fusion=fusion_enabled,
      )
    )
    product_outputs.append(_raw_words(product_output))
    expression_outputs.append(_raw_words(expression_output))

  expected_product = [
    mul_f32_rne_bits(left, right)
    for left, right in zip(input_words["a"], input_words["b"], strict=True)
  ]
  expected_separate = [
    mul_add_separate_f32_bits(left, right, addend)
    for left, right, addend in zip(
      input_words["a"], input_words["b"], input_words["c"], strict=True
    )
  ]
  expected_fused = [
    fma_f32_rne_bits(left, right, addend)
    for left, right, addend in zip(
      input_words["a"], input_words["b"], input_words["c"], strict=True
    )
  ]
  ptx_by_fusion = {
    str(enabled).lower(): compiled[enabled].asm.get("ptx", "")
    for enabled in (False, True)
  }
  fusion_observations = {
    str(enabled).lower(): {
      "requested": enabled,
      "ptx_contains_fma_f32": _ptx_has_fma(ptx_by_fusion[str(enabled).lower()]),
      "ptx_contains_ftz_arithmetic": _ptx_has_ftz_arithmetic(
        ptx_by_fusion[str(enabled).lower()]
      ),
      "measurement": measurements[enabled],
    }
    for enabled in (False, True)
  }

  subtraction_input_words = {
    "left": (
      0x3F800000,
      0x3F800001,
      0x80000000,
      0x00000001,
      0x00000000,
      0,
      0,
      0,
    ),
    "right": (
      0x33000000,
      0xB3800000,
      0x00000000,
      0x00000000,
      0x80000000,
      0,
      0,
      0,
    ),
  }
  subtraction_left = _float_tensor_from_words(subtraction_input_words["left"])
  subtraction_right = _float_tensor_from_words(subtraction_input_words["right"])
  subtraction_output = torch.empty((8,), dtype=torch.float32, device="cuda")
  compiled_subtraction, subtraction_measurement = _measure_launches(
    lambda: subtraction_edge_kernel[(1,)](
      subtraction_left,
      subtraction_right,
      subtraction_output,
      num_warps=1,
      enable_fp_fusion=False,
    )
  )
  subtraction_actual = _raw_words(subtraction_output)
  subtraction_expected = _binary32_subtraction_reference(
    subtraction_input_words["left"], subtraction_input_words["right"]
  )
  subtraction_ptx = compiled_subtraction.asm.get("ptx", "")

  reduction_input_words = (0x3F800000, 0x33800000, 0xBF800000)
  reduction_input = _float_tensor_from_words(reduction_input_words)
  reduction_output = torch.empty((1,), dtype=torch.float32, device="cuda")
  compiled_reduction, reduction_measurement = _measure_launches(
    lambda: reduction_order_kernel[(1,)](
      reduction_input, reduction_output, num_warps=1, enable_fp_fusion=False
    )
  )
  reduction_actual = _raw_words(reduction_output)
  reduction_allowed = reduction_f32_rne_outcomes(reduction_input_words)
  reduction_ptx = compiled_reduction.asm.get("ptx", "")
  ir_directory.mkdir(parents=True, exist_ok=True)
  compiler_ir = {
    "arithmetic_fusion_disabled": _save_ir(
      compiled[False], "arithmetic-fusion-disabled", ir_directory
    ),
    "arithmetic_fusion_enabled": _save_ir(
      compiled[True], "arithmetic-fusion-enabled", ir_directory
    ),
    "reduction_order_sensitive": _save_ir(
      compiled_reduction, "reduction-order-sensitive", ir_directory
    ),
    "binary32_subtraction_rne_edges": _save_ir(
      compiled_subtraction, "subtraction-rne-edges", ir_directory
    ),
  }
  return {
    "cases": {
      "binary32_multiply_rne_edges": _case(product_outputs[0], expected_product)
      | {
        "input_bits": {
          key: [f"{word:08x}" for word in values] for key, values in input_words.items()
        }
      },
      "binary32_subtraction_rne_edges": _case(subtraction_actual, subtraction_expected)
      | {
        "input_bits": {
          key: [f"{word:08x}" for word in values]
          for key, values in subtraction_input_words.items()
        },
        "measurement": subtraction_measurement,
      },
      "binary32_subnormal_multiplication": _case([product_outputs[0][1]], [0x00000002])
      | {"policy": "gradual underflow; arithmetic PTX must not use .ftz"},
      "multiply_add_fusion_disabled": _case(expression_outputs[0], expected_separate),
      "multiply_add_fusion_enabled": _case(expression_outputs[1], expected_fused),
      "reduction_order_sensitive_sum": {
        "status": "pass"
        if len(reduction_actual) == 1 and reduction_actual[0] in reduction_allowed
        else "candidate_mismatch",
        "actual_bits": [f"{word:08x}" for word in reduction_actual],
        "allowed_binary32_tree_bits": [
          f"{word:08x}" for word in sorted(reduction_allowed)
        ],
        "input_bits": [f"{word:08x}" for word in reduction_input_words],
        "comparison": "observed result must be one rounded binary32 tree over all inputs",
        "measurement": reduction_measurement,
      },
    },
    "contracts": {
      "rounding": "binary32 round-to-nearest, ties-to-even",
      "fusion": fusion_observations,
      "subnormal": {
        "policy": "gradual underflow for ordinary f32 arithmetic",
        "minimum_subnormal_times_two_expected_bits": "00000002",
        "observed_bits": f"{product_outputs[0][1]:08x}",
        "ptx_ftz_arithmetic_present": any(
          item["ptx_contains_ftz_arithmetic"] for item in fusion_observations.values()
        ),
      },
      "reduction_order": {
        "policy": "no single tree is assumed; every binary32 tree result is admitted",
        "ptx_contains_ftz_arithmetic": _ptx_has_ftz_arithmetic(reduction_ptx),
      },
      "subtraction": {
        "policy": "binary32 RNE, IEEE signed zero, and gradual underflow",
        "ptx_contains_ftz_arithmetic": _ptx_has_ftz_arithmetic(subtraction_ptx),
      },
    },
    "compiler_ir": compiler_ir,
  }


def _case(actual, expected):
  return {
    "status": "pass" if actual == expected else "candidate_mismatch",
    "actual_bits": [f"{value:08x}" for value in actual],
    "expected_bits": [f"{value:08x}" for value in expected],
  }


def _save_ir(compiled, prefix, output_directory):
  ir_files = {}
  for name in ("ttir", "ttgir", "ptx", "sass"):
    code = getattr(compiled, "asm", {}).get(name)
    if isinstance(code, str) and code:
      path = output_directory / f"{prefix}.{name}"
      path.write_text(code)
      ir_files[name] = str(path)
  return ir_files


def _measure_launches(launch):
  started = time.perf_counter()
  compiled = launch()
  torch.cuda.synchronize()
  first_launch_wall_s = time.perf_counter() - started
  for _ in range(WARMUP_LAUNCHES):
    launch()
  torch.cuda.synchronize()

  samples_ms = []
  launches_per_sample = MEASURED_LAUNCHES // 3
  for _ in range(3):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(launches_per_sample):
      launch()
    end.record()
    torch.cuda.synchronize()
    samples_ms.append(start.elapsed_time(end) / launches_per_sample)
  return compiled, {
    "warmup_launches": WARMUP_LAUNCHES,
    "measured_launches": MEASURED_LAUNCHES,
    "first_launch_wall_s_including_compilation": first_launch_wall_s,
    "median_gpu_event_ms_per_launch": statistics.median(samples_ms),
  }


def run(output_path, ir_directory):
  if triton.__version__ != TRITON_VERSION:
    raise RuntimeError(f"expected Triton {TRITON_VERSION}, found {triton.__version__}")
  source_identity = verify_installed_fpsan_source(triton)
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA device is unavailable")
  device_name = torch.cuda.get_device_name()
  if "V100" not in device_name:
    raise RuntimeError(
      f"expected a V100 for the pinned IEEE control run, found {device_name}"
    )
  triton.knobs.compilation.instrumentation_mode = ""
  ir_directory.mkdir(parents=True, exist_ok=True)

  a = torch.tensor([1.0], dtype=torch.float32, device="cuda")
  b = torch.tensor([2.0**-24], dtype=torch.float32, device="cuda")
  c = torch.tensor([-1.0], dtype=torch.float32, device="cuda")
  rounding_output = torch.empty((2,), dtype=torch.float32, device="cuda")
  compiled_rounding, rounding_measurement = _measure_launches(
    lambda: rounding_trees_kernel[(1,)](a, b, c, rounding_output, num_warps=1)
  )
  actual_rounding = _raw_words(rounding_output)
  report = {
    "contract": "ordinary-ieee-f32-controls-v3",
    "triton_version": triton.__version__,
    "triton_source_identity": source_identity,
    "torch_version": torch.__version__,
    "instrumentation_mode": getattr(
      triton.knobs.compilation, "instrumentation_mode", ""
    ),
    "device_name": device_name,
    "compute_capability": list(torch.cuda.get_device_capability()),
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "cases": {
      "reassociation_sensitive_addition": _case(
        actual_rounding, list(rounding_reassociation_bits())
      )
      | {"measurement": rounding_measurement},
    },
    "compiler_ir": {
      "reassociation_sensitive_addition": _save_ir(
        compiled_rounding, "rounding", ir_directory
      )
    },
  }
  edge_results = _run_binary32_edge_cases(ir_directory)
  report["cases"].update(edge_results["cases"])
  report["compiler_ir"].update(edge_results["compiler_ir"])
  report["edge_contracts"] = edge_results["contracts"]

  dividends = torch.tensor([-0.0, 1.0e10], dtype=torch.float32, device="cuda")
  divisors = torch.tensor([3.0, 3.0], dtype=torch.float32, device="cuda")
  remainder_output = torch.empty_like(dividends)
  compiled_remainder, remainder_measurement = _measure_launches(
    lambda: remainder_kernel[(1,)](dividends, divisors, remainder_output, num_warps=1)
  )
  actual_remainder = _raw_words(remainder_output)
  expected_remainder = [
    f32_fmod_bits(-0.0, 3.0),
    f32_fmod_bits(1.0e10, 3.0),
  ]
  report["cases"]["signed_zero_and_large_fmod"] = _case(
    actual_remainder, expected_remainder
  ) | {"measurement": remainder_measurement}
  report["compiler_ir"]["signed_zero_and_large_fmod"] = _save_ir(
    compiled_remainder, "remainder", ir_directory
  )
  report["measurement_protocol"] = {
    "warmup_launches_per_case": WARMUP_LAUNCHES,
    "measured_launches_per_case": MEASURED_LAUNCHES,
    "num_warps": 1,
    "input_domain": "fixed binary32 rounding and remainder controls",
  }
  torch.cuda.synchronize()
  Path(output_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  failed_cases = [
    name for name, result in report["cases"].items() if result["status"] != "pass"
  ]
  fusion = report["edge_contracts"]["fusion"]
  for enabled in (False, True):
    key = str(enabled).lower()
    observation = fusion[key]
    expected_fma = enabled
    if observation["ptx_contains_fma_f32"] != expected_fma:
      failed_cases.append(f"fusion_contract_{key}")
    if observation["ptx_contains_ftz_arithmetic"]:
      failed_cases.append(f"subnormal_contract_fusion_{key}")
  if report["edge_contracts"]["subnormal"]["observed_bits"] != "00000002":
    failed_cases.append("subnormal_contract_result")
  if report["edge_contracts"]["reduction_order"]["ptx_contains_ftz_arithmetic"]:
    failed_cases.append("subnormal_contract_reduction")
  if report["edge_contracts"]["subtraction"]["ptx_contains_ftz_arithmetic"]:
    failed_cases.append("subtraction_contract_ftz")
  if failed_cases:
    raise AssertionError(
      "ordinary IEEE-f32 controls failed: " + ", ".join(sorted(failed_cases))
    )
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", default="ordinary-fp-controls.json")
  parser.add_argument("--ir-dir", default="ordinary-fp-ir")
  args = parser.parse_args()
  report = run(args.output, Path(args.ir_dir))
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
