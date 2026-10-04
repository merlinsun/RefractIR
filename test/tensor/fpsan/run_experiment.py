#!/usr/bin/env python3
"""Run a small Triton 3.8.0 FpSan payload-algebra experiment on CUDA."""

import argparse
import hashlib
import json
import math
import re
import statistics
import struct
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

from test.lib import tensor_exact_f32
from test.lib.tensor_fpsan import (
  compare_words,
  flatten_word_values,
  prove_payload_kernel_equations,
  prove_payload_laws,
  render_generated_region_triton_kernel,
  synthesize_generated_region,
  synthesize_payload_case,
  validate_fpsan_payload_lowering,
  validate_instrumentation_sensitivity,
  verify_installed_fpsan_source,
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
def instrumentation_sensitivity_kernel(a_ptr, b_ptr, out_ptr):
  tl.store(out_ptr, tl.load(a_ptr) + tl.load(b_ptr))


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


def _round_f32_add(left, right):
  if not math.isfinite(left) or not math.isfinite(right):
    raise ValueError("ordinary generated reference requires finite f32 operands")
  try:
    result = struct.unpack("<f", struct.pack("<f", left + right))[0]
  except OverflowError as error:
    raise ValueError("ordinary generated reference overflowed finite f32") from error
  if not math.isfinite(result):
    raise ValueError("ordinary generated reference overflowed finite f32")
  return result


def _ordinary_generated_region_output_words(source, input_bytes):
  """Execute admitted add/sum/broadcast regions with one RNE f32 step per op."""
  shape, iterations, tensor_types, nodes, updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  count = math.prod(shape)
  if not isinstance(input_bytes, bytes) or len(input_bytes) != count * 4:
    raise ValueError("ordinary generated reference input does not match the tensor shape")
  input_values = struct.unpack(f"<{count}f", input_bytes)
  if any(not math.isfinite(value) for value in input_values):
    raise ValueError("ordinary generated reference requires finite f32 inputs")
  values = {
    "x": tuple(input_values),
    "acc_x": tuple(input_values),
    "acc_y": (0.0,) * count,
  }

  for _ in range(iterations):
    for destination, operation, operands, attribute in nodes:
      output_shape = tensor_types[destination]
      if operation == "add":
        left = tensor_exact_f32._broadcast_values(
          values[operands[0]], tensor_types[operands[0]], output_shape
        )
        right = tensor_exact_f32._broadcast_values(
          values[operands[1]], tensor_types[operands[1]], output_shape
        )
        result = tuple(_round_f32_add(a, b) for a, b in zip(left, right))
      elif operation == "sum":
        input_shape = tensor_types[operands[0]]
        axis = attribute if attribute >= 0 else attribute + len(input_shape)
        result_values = []
        for output_coordinates in tensor_exact_f32._coordinates(output_shape):
          total = 0.0
          for axis_coordinate in range(input_shape[axis]):
            coordinates = list(output_coordinates)
            coordinates[axis] = axis_coordinate
            index = tensor_exact_f32._flat_index(coordinates, input_shape)
            total = _round_f32_add(total, values[operands[0]][index])
          result_values.append(total)
        result = tuple(result_values)
      else:
        result = tensor_exact_f32._broadcast_values(
          values[operands[0]], tensor_types[operands[0]], output_shape
        )
      values[destination] = result

    for state, root in updates:
      values[state] = tuple(values[root])

  output_left = tensor_exact_f32._broadcast_values(values["acc_x"], shape, shape)
  output_right = tensor_exact_f32._broadcast_values(values["acc_y"], shape, shape)
  output_values = tuple(
    _round_f32_add(left, right) for left, right in zip(output_left, output_right)
  )
  output_bytes = struct.pack(f"<{count}f", *output_values)
  return list(struct.unpack(f"<{count}I", output_bytes))


def _generated_region_corpus_sources():
  program_dir = Path(__file__).resolve().parents[1] / "gluon" / "programs"
  manifest_path = program_dir / "native-generated-corpus.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  if manifest.get("schema") != "refractir.generated-region-corpus.v1":
    raise ValueError(f"unsupported generated-region corpus: {manifest_path}")
  sources = []
  for record in manifest["programs"]:
    source_path = program_dir / record["source_file"]
    source = source_path.read_text(encoding="utf-8")
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if source_hash != record["source_sha256"]:
      raise ValueError(f"frozen generated-region source hash mismatch: {source_path}")
    sources.append(
      {
        "program_id": record["program_id"],
        "source_path": source_path,
        "source_sha256": source_hash,
        "generation": record["generation"],
      }
    )
  return sources


def _run_generated_region_case(
  ir_directory,
  source_path=None,
  *,
  target=None,
  target_index=(0, 0),
  timeout_s=30.0,
):
  case_started = time.perf_counter()
  ir_directory = Path(ir_directory)
  ir_directory.mkdir(parents=True, exist_ok=True)
  source_path = (
    Path(__file__).with_name("generated_region.sir")
    if source_path is None
    else Path(source_path)
  )
  source = source_path.read_text(encoding="utf-8")
  witness = synthesize_generated_region(
    source,
    target=target,
    target_index=target_index,
    timeout_s=timeout_s,
  )
  if witness["status"] != "SAT":
    return {
      "status": witness["status"],
      "elapsed_s": time.perf_counter() - case_started,
      "property": witness.get("property"),
      "cost_s": witness.get("cost_s"),
      "transport_contract": witness.get("transport_contract"),
      "solver": witness.get("solver"),
      "query_sha256": witness.get("query_sha256"),
      "query_smt2": witness.get("query_smt2"),
      "solver_outcome": witness.get("solver_outcome"),
      "region_source": {
        "path": source_path.name,
        "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
      },
    }
  shape = tuple(witness["shape"])
  inputs = _tensor_from_words(witness["input_float_words"]).reshape(shape)

  # The source is rendered only from the admitted region parsed by the host model.
  kernel_source_path = ir_directory / "generated_region_kernel.py"
  kernel_source = render_generated_region_triton_kernel(source)
  kernel_source_path.write_text(kernel_source, encoding="utf-8")
  namespace = {}
  exec(compile(kernel_source, str(kernel_source_path), "exec"), namespace)
  kernel = namespace["fpsan_generated_region"]
  output, compiled, measurement = _run_generated_kernel_mode(
    kernel, inputs, shape, "fpsan"
  )
  result = _expect_words("generated_region", output, witness["output_float_words"])
  result["measurement"] = measurement
  result["smt_witness"] = {
    "transport_contract": witness["transport_contract"],
    "query_sha256": witness["query_sha256"],
    "input_payloads": witness["input_payloads"],
    "output_payloads": witness["output_payloads"],
    "input_float_bits": witness["input_float_bits"],
    "output_float_bits": witness["output_float_bits"],
    "calibration_mutants": witness["calibration_mutants"],
  }
  result["smt_witness"]["property"] = witness["property"]
  result["smt_witness"]["cost_s"] = witness["cost_s"]
  result["payload_lowering"] = validate_fpsan_payload_lowering(
    getattr(compiled, "asm", {}).get("llir", "")
  )
  fpsan_ir = _capture_ir(compiled, "generated_region_fpsan", ir_directory)
  result["compiler_ir"] = fpsan_ir
  result["compiler_modes"] = {"fpsan": fpsan_ir}

  input_bytes = struct.pack(
    f"<{len(witness['input_float_words'])}I", *witness["input_float_words"]
  )
  ordinary_expected_words = _ordinary_generated_region_output_words(source, input_bytes)
  ordinary_output, ordinary_compiled, ordinary_measurement = _run_generated_kernel_mode(
    kernel, inputs, shape, ""
  )
  ordinary_result = _expect_words(
    "generated_region_ordinary", ordinary_output, ordinary_expected_words
  )
  ordinary_result["measurement"] = ordinary_measurement
  ordinary_ir = _capture_ir(ordinary_compiled, "generated_region_ordinary", ir_directory)
  result["compiler_modes"]["ordinary"] = ordinary_ir
  result["ordinary_reference"] = {
    "status": ordinary_result["status"],
    "actual_bits": ordinary_result["actual_bits"],
    "expected_bits": ordinary_result["expected_bits"],
    "measurement": ordinary_measurement,
    "compiler_ir": ordinary_ir,
  }
  result["compiler_mode_comparison"] = _compare_generated_compiler_modes(
    fpsan_ir, ordinary_ir
  )
  result["region_source"] = {
    "path": source_path.name,
    "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
  }
  result["calibration_mutants"] = {}
  for mutant_name, render_options in (
    ("lost_iteration", {"iteration_count": witness["iterations"] - 1}),
    ("lost_update_acc_y", {"omitted_updates": ("acc_y",)}),
  ):
    mutant_kernel_source = render_generated_region_triton_kernel(
      source, **render_options
    )
    mutant_source_path = ir_directory / f"generated_region_{mutant_name}.py"
    mutant_source_path.write_text(mutant_kernel_source, encoding="utf-8")
    mutant_namespace = {}
    exec(
      compile(mutant_kernel_source, str(mutant_source_path), "exec"),
      mutant_namespace,
    )
    mutant_output = torch.empty(shape, dtype=torch.float32, device="cuda")
    mutant_kernel = mutant_namespace["fpsan_generated_region"]
    mutant_compiled, mutant_measurement = _measure_launches(
      lambda: mutant_kernel[(1,)](inputs, mutant_output, num_warps=1)
    )
    actual_mutant_words = _raw_words(mutant_output)
    expected_mutant_words = [
      int(value, 16)
      for value in witness["calibration_mutants"][mutant_name][
        "candidate_float_bits"
      ]
    ]
    if actual_mutant_words != expected_mutant_words:
      raise AssertionError(
        f"{mutant_name} GPU output differs from its synthesized payload model"
      )
    mismatch_lanes = compare_words(actual_mutant_words, witness["output_float_words"])
    if not mismatch_lanes:
      raise AssertionError(f"generated-region {mutant_name} mutant was not detected")
    result["calibration_mutants"][mutant_name] = {
      "status": "detected",
      "mismatch_lanes": mismatch_lanes,
      "actual_bits": [f"{value:08x}" for value in actual_mutant_words],
      "expected_bits": [f"{value:08x}" for value in expected_mutant_words],
      "measurement": mutant_measurement,
      "payload_lowering": validate_fpsan_payload_lowering(
        getattr(mutant_compiled, "asm", {}).get("llir", "")
      ),
      "compiler_ir": _capture_ir(
        mutant_compiled, f"generated_region_{mutant_name}", ir_directory
      ),
    }
  result["elapsed_s"] = time.perf_counter() - case_started
  return result


def _run_generated_kernel_mode(kernel, inputs, shape, instrumentation_mode):
  output = torch.empty(shape, dtype=torch.float32, device="cuda")
  previous_mode = triton.knobs.compilation.instrumentation_mode
  try:
    triton.knobs.compilation.instrumentation_mode = instrumentation_mode
    compiled, measurement = _measure_launches(
      lambda: kernel[(1,)](inputs, output, num_warps=1)
    )
    torch.cuda.synchronize()
  finally:
    triton.knobs.compilation.instrumentation_mode = previous_mode
  return output, compiled, measurement


def _compare_generated_compiler_modes(fpsan_ir, ordinary_ir):
  metrics = (
    "layout_conversion_operations_in_ttgir",
    "warp_shuffle_operations_in_ptx",
    "shared_memory_transfers_in_ptx",
  )
  return {
    "stage_hashes_equal": {
      stage: fpsan_ir["stage_sha256"].get(stage)
      == ordinary_ir["stage_sha256"].get(stage)
      for stage in sorted(
        set(fpsan_ir["stage_sha256"]) | set(ordinary_ir["stage_sha256"])
      )
    },
    "metric_delta_fpsan_minus_ordinary": {
      metric: fpsan_ir.get(metric, 0) - ordinary_ir.get(metric, 0)
      for metric in metrics
    },
    "interpretation_limit": (
      "paired compiler artifacts describe observed transformations; repeated IR "
      "expressions alone do not establish rematerialization"
    ),
  }


def _raw_words(tensor):
  values = tensor.view(torch.int32).cpu().tolist()
  return [value & 0xFFFFFFFF for value in flatten_word_values(values)]


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
  stage_sha256 = {}
  stage_bytes = {}
  for name in ("ttir", "ttgir", "llir", "ptx", "sass"):
    source = assembly.get(name)
    if isinstance(source, str) and source:
      path = ir_directory / f"{case_name}.{name}"
      path.write_text(source)
      paths[name] = str(path)
      encoded = source.encode("utf-8")
      stage_sha256[name] = hashlib.sha256(encoded).hexdigest()
      stage_bytes[name] = len(encoded)
  ttgir = assembly.get("ttgir", "")
  ptx = assembly.get("ptx", "")
  return {
    "files": paths,
    "stage_sha256": stage_sha256,
    "stage_bytes": stage_bytes,
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


def _run_instrumentation_sensitivity(ir_directory):
  left = torch.tensor([1.0], dtype=torch.float32, device="cuda")
  right = torch.tensor([1.0], dtype=torch.float32, device="cuda")
  instrumented_output = torch.empty((1,), dtype=torch.float32, device="cuda")
  ordinary_output = torch.empty_like(instrumented_output)
  previous_mode = triton.knobs.compilation.instrumentation_mode
  try:
    triton.knobs.compilation.instrumentation_mode = "fpsan"
    compiled_instrumented = instrumentation_sensitivity_kernel[(1,)](
      left, right, instrumented_output, num_warps=1
    )
    torch.cuda.synchronize()
    instrumented_words = _raw_words(instrumented_output)

    triton.knobs.compilation.instrumentation_mode = ""
    compiled_ordinary = instrumentation_sensitivity_kernel[(1,)](
      left, right, ordinary_output, num_warps=1
    )
    torch.cuda.synchronize()
    ordinary_words = _raw_words(ordinary_output)
  finally:
    triton.knobs.compilation.instrumentation_mode = previous_mode

  ordinary_expected = [0x40000000]
  acceptance = validate_instrumentation_sensitivity(
    instrumented_words, ordinary_words, ordinary_expected
  )
  payload_lowering = validate_fpsan_payload_lowering(
    getattr(compiled_instrumented, "asm", {}).get("llir", "")
  )
  return {
    **acceptance,
    "payload_lowering": payload_lowering,
    "input_bits": ["3f800000", "3f800000"],
    "instrumented_bits": [f"{word:08x}" for word in instrumented_words],
    "ordinary_bits": [f"{word:08x}" for word in ordinary_words],
    "ordinary_expected_bits": [f"{word:08x}" for word in ordinary_expected],
    "compiler_ir": {
      "fpsan": _capture_ir(
        compiled_instrumented, "instrumentation_sensitivity_fpsan", ir_directory
      ),
      "ordinary": _capture_ir(
        compiled_ordinary, "instrumentation_sensitivity_ordinary", ir_directory
      ),
    },
  }


def run(
  output_path,
  ir_directory,
  *,
  generated_target=None,
  generated_target_index=(0, 0),
  generated_timeout_s=30.0,
):
  if triton.__version__ != TRITON_VERSION:
    raise RuntimeError(f"expected Triton {TRITON_VERSION}, found {triton.__version__}")
  source_identity = verify_installed_fpsan_source(triton)
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
  instrumentation_sensitivity = _run_instrumentation_sensitivity(ir_directory)
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
  generated_region_result = _run_generated_region_case(ir_directory)
  generated_region_corpus = _generated_region_corpus_sources()
  generated_region_results = {}
  for record in generated_region_corpus:
    case_ir_directory = ir_directory / "generated_regions" / record["program_id"]
    result = _run_generated_region_case(
      case_ir_directory,
      source_path=record["source_path"],
      target=generated_target,
      target_index=generated_target_index,
      timeout_s=generated_timeout_s,
    )
    result["program_id"] = record["program_id"]
    generated_region_results[record["program_id"]] = result
  report = {
    "contract": "triton-fpsan-fixed-payload-ring-v2",
    "triton_version": triton.__version__,
    "triton_source_identity": source_identity,
    "torch_version": torch.__version__,
    "instrumentation_mode": triton.knobs.compilation.instrumentation_mode,
    "device_name": device_name,
    "generated_region_protocol": {
      "program_ids": [record["program_id"] for record in generated_region_corpus],
      "logical_input_domain": [0, 8],
      "semantic_contract": "generated-region-qf-bv-payload-ring-v1",
      "property_contract": "instrumented payload output element equals an integer payload",
      "input_transport": (
        "payload integers in [0,8] map through the pinned finite-f32 mixer"
      ),
      "property": None
      if generated_target is None
      else {
        "kind": "output-payload-equals",
        "target": generated_target,
        "index": list(generated_target_index),
      },
      "selection_budget_s_per_program": generated_timeout_s,
      "gpu_launch_protocol": {
        "warmup_launches": WARMUP_LAUNCHES,
        "measured_launches_per_kernel": MEASURED_LAUNCHES,
      },
      "selection_policy": "SMT payload model with mutation-sensitive loop constraints",
      "transport_contract": (
        "triton-fpsan-f32-mixer-e2e356723ce35f46b5144373b145c55c4d712237"
      ),
    },
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
    "instrumentation_sensitivity": instrumentation_sensitivity,
    "generated_region": generated_region_result,
    "generated_regions": {
      "corpus_schema": "refractir.generated-region-corpus.v1",
      "payload_transport_contract": (
        "triton-fpsan-f32-mixer-e2e356723ce35f46b5144373b145c55c4d712237"
      ),
      "input_payload_range": [0, 8],
      "output_payload_range": [0, 1 << 24],
      "programs": generated_region_results,
    },
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
  report["status"] = (
    "pass"
    if generated_region_result.get("status") == "pass"
    and all(
      result.get("status") == "pass" for result in generated_region_results.values()
    )
    else "incomplete"
  )
  Path(output_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", default="fpsan-experiment.json")
  parser.add_argument("--ir-dir", default="fpsan-ir")
  parser.add_argument("--generated-target", type=int, default=None)
  parser.add_argument("--generated-target-index", default="0,0")
  parser.add_argument("--generated-timeout-s", type=float, default=30.0)
  args = parser.parse_args()
  try:
    generated_target_index = tuple(
      int(value.strip()) for value in args.generated_target_index.split(",")
    )
  except ValueError as error:
    parser.error(f"invalid generated target index: {error}")
  report = run(
    args.output,
    Path(args.ir_dir),
    generated_target=args.generated_target,
    generated_target_index=generated_target_index,
    generated_timeout_s=args.generated_timeout_s,
  )
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
