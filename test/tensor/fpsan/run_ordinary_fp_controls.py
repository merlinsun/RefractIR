#!/usr/bin/env python3
"""Run IEEE-f32 controls separately from Triton's FpSan mode."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

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
def remainder_kernel(dividend_ptr, divisor_ptr, output_ptr):
  offsets = tl.arange(0, 2)
  dividend = tl.load(dividend_ptr + offsets)
  divisor = tl.load(divisor_ptr + offsets)
  tl.store(output_ptr + offsets, libdevice.fmod(dividend, divisor))


def _raw_words(tensor):
  return [value & 0xFFFFFFFF for value in tensor.view(torch.int32).cpu().tolist()]


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
    "contract": "ordinary-ieee-f32-controls-v1",
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
