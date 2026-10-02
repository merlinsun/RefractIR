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
  compare_words,
  prove_payload_kernel_equations,
  prove_payload_laws,
  synthesize_payload_case,
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


def _tensor_from_words(words):
  signed_words = [word if word < 0x80000000 else word - (1 << 32) for word in words]
  return torch.tensor(signed_words, dtype=torch.int32, device="cuda").view(
    torch.float32
  )


def _case_inputs(witness):
  return [_tensor_from_words([word]) for word in witness["input_float_words"]]


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


def _run_measured_case(name, launch, tensor, expected, ir_directory, witness):
  compiled, measurement = _measure_launches(launch)
  result = _expect_words(name, tensor, expected)
  result["measurement"] = measurement
  result["compiler_ir"] = _capture_ir(compiled, name, ir_directory)
  result["smt_witness"] = {
    "case": witness["case"],
    "query_sha256": witness["query_sha256"],
  }
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
  equation_proofs = prove_payload_kernel_equations()
  witness_names = (
    "add_tree",
    "add_cancellation",
    "multiply_fixed_points",
    "subtract_zero_identity",
    "sum_reduction",
    "transpose_copy",
    "broadcast_and_add_zero",
    "operand_routing_fault",
    "lost_update_fault",
  )
  witnesses = {name: synthesize_payload_case(name) for name in witness_names}

  tree_inputs = _case_inputs(witnesses["add_tree"])
  cancellation_inputs = _case_inputs(witnesses["add_cancellation"])
  multiply_inputs = _case_inputs(witnesses["multiply_fixed_points"])
  subtract_inputs = _case_inputs(witnesses["subtract_zero_identity"])
  reduction_input = _tensor_from_words(witnesses["sum_reduction"]["input_float_words"])
  matrix = _tensor_from_words(witnesses["transpose_copy"]["input_float_words"])
  broadcast_input = _tensor_from_words(
    witnesses["broadcast_and_add_zero"]["input_float_words"]
  )
  output = torch.empty((1,), dtype=torch.float32, device="cuda")
  report = {
    "contract": "triton-fpsan-fixed-payload-ring-v2",
    "triton_version": triton.__version__,
    "torch_version": torch.__version__,
    "instrumentation_mode": triton.knobs.compilation.instrumentation_mode,
    "device_name": device_name,
    "compute_capability": list(torch.cuda.get_device_capability()),
    "bitvector_laws": "unsat for negated QF_BV identities",
    "bitvector_kernel_equations": equation_proofs,
    "measurement_protocol": {
      "warmup_launches_per_case": WARMUP_LAUNCHES,
      "measured_launches_per_case": MEASURED_LAUNCHES,
      "num_warps": 1,
      "selection": "fixed documented embeddings; no payload-dependent control flow",
    },
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "cases": {},
    "smt_witnesses": witnesses,
  }

  expected_sum = witnesses["sum_reduction"]["reference_payloads"][0]
  if expected_sum != witnesses["sum_reduction"]["candidate_payloads"][0]:
    raise AssertionError("host evaluator found a reduction-order mismatch")
  report["cases"]["add_tree_left"] = _run_measured_case(
    "add_tree_left",
    lambda: add_left_kernel[(1,)](*tree_inputs, output, num_warps=1),
    output,
    witnesses["add_tree"]["reference_float_words"],
    ir_directory,
    witnesses["add_tree"],
  )
  report["cases"]["add_tree_reassociated"] = _run_measured_case(
    "add_tree_reassociated",
    lambda: add_reassociated_kernel[(1,)](*tree_inputs, output, num_warps=1),
    output,
    witnesses["add_tree"]["candidate_float_words"],
    ir_directory,
    witnesses["add_tree"],
  )

  report["cases"]["add_cancellation"] = _run_measured_case(
    "add_cancellation",
    lambda: add_cancel_kernel[(1,)](*cancellation_inputs, output, num_warps=1),
    output,
    witnesses["add_cancellation"]["reference_float_words"],
    ir_directory,
    witnesses["add_cancellation"],
  )
  report["cases"]["multiply_fixed_points"] = _run_measured_case(
    "multiply_fixed_points",
    lambda: multiply_kernel[(1,)](*multiply_inputs, output, num_warps=1),
    output,
    witnesses["multiply_fixed_points"]["reference_float_words"],
    ir_directory,
    witnesses["multiply_fixed_points"],
  )
  report["cases"]["subtract_zero_identity"] = _run_measured_case(
    "subtract_zero_identity",
    lambda: subtract_zero_kernel[(1,)](*subtract_inputs, output, num_warps=1),
    output,
    witnesses["subtract_zero_identity"]["reference_float_words"],
    ir_directory,
    witnesses["subtract_zero_identity"],
  )

  reduction_output = torch.empty((1,), dtype=torch.float32, device="cuda")
  report["cases"]["sum_reduction"] = _run_measured_case(
    "sum_reduction",
    lambda: reduce_sum_kernel[(1,)](reduction_input, reduction_output, num_warps=1),
    reduction_output,
    witnesses["sum_reduction"]["reference_float_words"],
    ir_directory,
    witnesses["sum_reduction"],
  )

  transposed = torch.empty((4,), dtype=torch.float32, device="cuda")
  report["cases"]["transpose_copy"] = _run_measured_case(
    "transpose_copy",
    lambda: transpose_copy_kernel[(1,)](matrix, transposed, num_warps=1),
    transposed,
    witnesses["transpose_copy"]["reference_float_words"],
    ir_directory,
    witnesses["transpose_copy"],
  )

  broadcast = torch.empty((4,), dtype=torch.float32, device="cuda")
  report["cases"]["broadcast_and_add_zero"] = _run_measured_case(
    "broadcast_and_add_zero",
    lambda: broadcast_add_zero_kernel[(1,)](broadcast_input, broadcast, num_warps=1),
    broadcast,
    witnesses["broadcast_and_add_zero"]["reference_float_words"],
    ir_directory,
    witnesses["broadcast_and_add_zero"],
  )

  route_witness = witnesses["operand_routing_fault"]
  route_input = _tensor_from_words(route_witness["input_float_words"])
  route_output = torch.empty((4,), dtype=torch.float32, device="cuda")
  expected_route_words = route_witness["reference_float_words"]
  candidate_route_words = route_witness["candidate_float_words"]
  compiled_route, route_measurement = _measure_launches(
    lambda: operand_routing_fault_kernel[(1,)](route_input, route_output, num_warps=1)
  )
  actual_route_words = _raw_words(route_output)
  if actual_route_words != candidate_route_words:
    raise AssertionError(
      "operand-routing kernel differs from its synthesized BV equation"
    )
  route_mismatches = compare_words(actual_route_words, expected_route_words)
  if not route_mismatches:
    raise AssertionError("operand-routing calibration fault was not detected")

  update_witness = witnesses["lost_update_fault"]
  update_input = _tensor_from_words(update_witness["input_float_words"])
  update_output = torch.empty((1,), dtype=torch.float32, device="cuda")
  compiled_update, update_measurement = _measure_launches(
    lambda: lost_update_fault_kernel[(1,)](update_input, update_output, num_warps=1)
  )
  actual_update_words = _raw_words(update_output)
  expected_update_words = update_witness["reference_float_words"]
  candidate_update_words = update_witness["candidate_float_words"]
  if actual_update_words != candidate_update_words:
    raise AssertionError("lost-update kernel differs from its synthesized BV equation")
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
      "runtime_synthesis_domain": ["+0", "+1", "-1"],
      "input_words_are_transported_through_integer_views": True,
    },
    "input_float_bits_by_case": {
      name: witness["input_float_bits"] for name, witness in witnesses.items()
    },
    "input_payloads_by_case": {
      name: witness["input_payloads"] for name, witness in witnesses.items()
    },
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
