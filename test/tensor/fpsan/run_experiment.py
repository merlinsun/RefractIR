#!/usr/bin/env python3
"""Run a small Triton 3.8.0 FpSan payload-algebra experiment on CUDA."""

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

from test.lib.tensor_fpsan import (
  add32,
  broadcast_scalar,
  compare_words,
  embed_known_f32,
  mul32,
  prove_payload_laws,
  reduce_tree,
  sub32,
  transpose_2x2,
  unembed_known_payload,
)

TRITON_VERSION = "3.8.0"
WARMUP_LAUNCHES = 3
MEASURED_LAUNCHES = 30


@triton.jit
def add_left_kernel(a_ptr, b_ptr, c_ptr, d_ptr, out_ptr):
  a = tl.load(a_ptr)
  b = tl.load(b_ptr)
  c = tl.load(c_ptr)
  d = tl.load(d_ptr)
  tl.store(out_ptr, (a + b) + (c + d))


@triton.jit
def add_reassociated_kernel(a_ptr, b_ptr, c_ptr, d_ptr, out_ptr):
  a = tl.load(a_ptr)
  b = tl.load(b_ptr)
  c = tl.load(c_ptr)
  d = tl.load(d_ptr)
  tl.store(out_ptr, (a + c) + (b + d))


@triton.jit
def add_cancel_kernel(a_ptr, b_ptr, out_ptr):
  tl.store(out_ptr, tl.load(a_ptr) + tl.load(b_ptr))


@triton.jit
def multiply_kernel(a_ptr, b_ptr, out_ptr):
  tl.store(out_ptr, tl.load(a_ptr) * tl.load(b_ptr))


@triton.jit
def subtract_zero_kernel(a_ptr, zero_ptr, out_ptr):
  tl.store(out_ptr, tl.load(a_ptr) - tl.load(zero_ptr))


@triton.jit
def operand_routing_fault_kernel(input_ptr, output_ptr):
  offsets = tl.arange(0, 4)
  routed_offsets = (offsets + 1) % 4
  tl.store(output_ptr + offsets, tl.load(input_ptr + routed_offsets))


@triton.jit
def lost_update_fault_kernel(input_ptr, output_ptr):
  tl.store(output_ptr, tl.load(input_ptr))


@triton.jit
def reduce_sum_kernel(input_ptr, output_ptr):
  offsets = tl.arange(0, 4)
  values = tl.load(input_ptr + offsets)
  tl.store(output_ptr, tl.sum(values, 0))


@triton.jit
def transpose_copy_kernel(input_ptr, output_ptr):
  rows = tl.arange(0, 2)
  columns = tl.arange(0, 2)
  offsets = rows[:, None] * 2 + columns[None, :]
  values = tl.load(input_ptr + offsets)
  tl.store(output_ptr + offsets, tl.trans(values))


@triton.jit
def broadcast_add_zero_kernel(input_ptr, output_ptr):
  offsets = tl.arange(0, 4)
  value = tl.load(input_ptr)
  zeros = tl.full((4,), 0.0, tl.float32)
  tl.store(output_ptr + offsets, zeros + value)


def _tensor(values):
  return torch.tensor(values, dtype=torch.float32, device="cuda")


def _raw_words(tensor):
  return [value & 0xFFFFFFFF for value in tensor.view(torch.int32).cpu().tolist()]


def _expect_words(name, tensor, expected):
  actual = _raw_words(tensor)
  expected = [value & 0xFFFFFFFF for value in expected]
  if actual != expected:
    raise AssertionError(
      f"{name}: got {[hex(value) for value in actual]}, "
      f"expected {[hex(value) for value in expected]}"
    )
  return {
    "actual_bits": [f"{value:08x}" for value in actual],
    "expected_bits": [f"{value:08x}" for value in expected],
    "status": "pass",
  }


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


def _capture_ir(compiled, case_name, ir_directory):
  assembly = getattr(compiled, "asm", {})
  paths = {}
  for name in ("ttir", "ttgir", "llir", "ptx", "sass"):
    source = assembly.get(name)
    if isinstance(source, str) and source:
      path = ir_directory / f"{case_name}.{name}"
      path.write_text(source)
      paths[name] = str(path)
  ttgir = assembly.get("ttgir", "")
  ptx = assembly.get("ptx", "")
  return {
    "files": paths,
    "layout_conversion_operations_in_ttgir": len(
      re.findall(r"(?:ttg\.)?convert_layout|convert_layout", ttgir)
    ),
    "warp_shuffle_operations_in_ptx": len(re.findall(r"\bshfl\.", ptx)),
    "shared_memory_transfers_in_ptx": len(re.findall(r"\b(?:ld|st)\.shared", ptx)),
  }


def _run_measured_case(name, launch, tensor, expected, ir_directory):
  compiled, measurement = _measure_launches(launch)
  result = _expect_words(name, tensor, expected)
  result["measurement"] = measurement
  result["compiler_ir"] = _capture_ir(compiled, name, ir_directory)
  return result


def run(output_path, ir_directory):
  if triton.__version__ != TRITON_VERSION:
    raise RuntimeError(f"expected Triton {TRITON_VERSION}, found {triton.__version__}")
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA device is unavailable")
  device_name = torch.cuda.get_device_name()
  if "V100" not in device_name:
    raise RuntimeError(
      f"expected a V100 for the pinned FpSan pilot, found {device_name}"
    )
  if not hasattr(triton.knobs.compilation, "instrumentation_mode"):
    raise RuntimeError("installed Triton has no FpSan instrumentation knob")
  triton.knobs.compilation.instrumentation_mode = "fpsan"
  ir_directory.mkdir(parents=True, exist_ok=True)
  if not prove_payload_laws():
    raise AssertionError("independent bit-vector payload checks failed")

  plus_one = _tensor([1.0])
  minus_one = _tensor([-1.0])
  tree_inputs = [_tensor([value]) for value in (1.0, -1.0, 1.0, 0.0)]
  matrix = _tensor([1.0, 0.0, -1.0, 1.0])
  zero = _tensor([0.0])
  reduction_input = _tensor([1.0, -1.0, 1.0, 0.0])
  output = torch.empty((1,), dtype=torch.float32, device="cuda")
  report = {
    "contract": "triton-fpsan-fixed-payload-ring-v1",
    "triton_version": triton.__version__,
    "torch_version": torch.__version__,
    "instrumentation_mode": triton.knobs.compilation.instrumentation_mode,
    "device_name": device_name,
    "compute_capability": list(torch.cuda.get_device_capability()),
    "bitvector_laws": "unsat for negated QF_BV identities",
    "measurement_protocol": {
      "warmup_launches_per_case": WARMUP_LAUNCHES,
      "measured_launches_per_case": MEASURED_LAUNCHES,
      "num_warps": 1,
      "selection": "fixed documented embeddings; no payload-dependent control flow",
    },
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "cases": {},
  }

  fixed_payloads = [embed_known_f32(value) for value in (1.0, -1.0, 1.0, 0.0)]
  expected_sum = reduce_tree(fixed_payloads, "left")
  if expected_sum != reduce_tree(fixed_payloads, "balanced"):
    raise AssertionError("host evaluator found a reduction-order mismatch")
  report["cases"]["add_tree_left"] = _run_measured_case(
    "add_tree_left",
    lambda: add_left_kernel[(1,)](*tree_inputs, output, num_warps=1),
    output,
    [unembed_known_payload(expected_sum)],
    ir_directory,
  )
  report["cases"]["add_tree_reassociated"] = _run_measured_case(
    "add_tree_reassociated",
    lambda: add_reassociated_kernel[(1,)](*tree_inputs, output, num_warps=1),
    output,
    [unembed_known_payload(expected_sum)],
    ir_directory,
  )

  report["cases"]["add_cancellation"] = _run_measured_case(
    "add_cancellation",
    lambda: add_cancel_kernel[(1,)](plus_one, minus_one, output, num_warps=1),
    output,
    [unembed_known_payload(add32(1, 0xFFFFFFFF))],
    ir_directory,
  )
  report["cases"]["multiply_fixed_points"] = _run_measured_case(
    "multiply_fixed_points",
    lambda: multiply_kernel[(1,)](plus_one, minus_one, output, num_warps=1),
    output,
    [unembed_known_payload(mul32(1, 0xFFFFFFFF))],
    ir_directory,
  )
  report["cases"]["subtract_zero_identity"] = _run_measured_case(
    "subtract_zero_identity",
    lambda: subtract_zero_kernel[(1,)](plus_one, zero, output, num_warps=1),
    output,
    [unembed_known_payload(sub32(1, 0))],
    ir_directory,
  )

  reduction_output = torch.empty((1,), dtype=torch.float32, device="cuda")
  report["cases"]["sum_reduction"] = _run_measured_case(
    "sum_reduction",
    lambda: reduce_sum_kernel[(1,)](reduction_input, reduction_output, num_warps=1),
    reduction_output,
    [unembed_known_payload(expected_sum)],
    ir_directory,
  )

  transposed = torch.empty((4,), dtype=torch.float32, device="cuda")
  expected_transpose = transpose_2x2(
    [embed_known_f32(value) for value in [1.0, 0.0, -1.0, 1.0]]
  )
  report["cases"]["transpose_copy"] = _run_measured_case(
    "transpose_copy",
    lambda: transpose_copy_kernel[(1,)](matrix, transposed, num_warps=1),
    transposed,
    [unembed_known_payload(value) for value in expected_transpose],
    ir_directory,
  )

  broadcast = torch.empty((4,), dtype=torch.float32, device="cuda")
  expected_broadcast = broadcast_scalar(embed_known_f32(-1.0), 4)
  report["cases"]["broadcast_and_add_zero"] = _run_measured_case(
    "broadcast_and_add_zero",
    lambda: broadcast_add_zero_kernel[(1,)](minus_one, broadcast, num_warps=1),
    broadcast,
    [unembed_known_payload(value) for value in expected_broadcast],
    ir_directory,
  )

  route_input = _tensor([1.0, 0.0, -1.0, 1.0])
  route_output = torch.empty((4,), dtype=torch.float32, device="cuda")
  routed_payloads = [embed_known_f32(value) for value in (1.0, 0.0, -1.0, 1.0)]
  expected_route_words = [unembed_known_payload(value) for value in routed_payloads]
  compiled_route, route_measurement = _measure_launches(
    lambda: operand_routing_fault_kernel[(1,)](route_input, route_output, num_warps=1)
  )
  actual_route_words = _raw_words(route_output)
  route_mismatches = compare_words(actual_route_words, expected_route_words)
  if not route_mismatches:
    raise AssertionError("operand-routing calibration fault was not detected")

  update_input = _tensor([1.0, -1.0])
  update_output = torch.empty((1,), dtype=torch.float32, device="cuda")
  compiled_update, update_measurement = _measure_launches(
    lambda: lost_update_fault_kernel[(1,)](update_input, update_output, num_warps=1)
  )
  actual_update_words = _raw_words(update_output)
  expected_update_words = [unembed_known_payload(add32(1, 0xFFFFFFFF))]
  update_mismatches = compare_words(actual_update_words, expected_update_words)
  if not update_mismatches:
    raise AssertionError("lost-update calibration fault was not detected")

  report["calibration_mutants"] = {
    "operand_routing": {
      "status": "detected",
      "mismatch_lanes": route_mismatches,
      "actual_bits": [f"{value:08x}" for value in actual_route_words],
      "expected_bits": [f"{value:08x}" for value in expected_route_words],
      "measurement": route_measurement,
      "compiler_ir": _capture_ir(compiled_route, "operand_routing_fault", ir_directory),
    },
    "lost_update": {
      "status": "detected",
      "mismatch_lanes": update_mismatches,
      "actual_bits": [f"{value:08x}" for value in actual_update_words],
      "expected_bits": [f"{value:08x}" for value in expected_update_words],
      "measurement": update_measurement,
      "compiler_ir": _capture_ir(compiled_update, "lost_update_fault", ir_directory),
    },
  }
  torch.cuda.synchronize()

  report["host_payloads"] = {
    "fixed_float_embeddings": {"+0": 0, "+1": 1, "-1": "ffffffff"},
    "admission": {
      "allowed_input_float_bits": [
        "00000000",
        "3f800000",
        "bf800000",
      ],
      "rejected_values_are_not_reinterpreted": True,
      "fixed_input_element_count": 4,
    },
    "input_values": ["+1", "-1", "+1", "+0"],
    "reduction_payload_hex": f"{expected_sum:08x}",
    "reduction_payload_signed": (
      expected_sum if expected_sum < 0x80000000 else expected_sum - (1 << 32)
    ),
  }
  Path(output_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", default="fpsan-experiment.json")
  parser.add_argument("--ir-dir", default="fpsan-ir")
  args = parser.parse_args()
  report = run(args.output, Path(args.ir_dir))
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
