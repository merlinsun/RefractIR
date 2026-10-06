#!/usr/bin/env python3
"""Run a small Triton 3.8.0 FpSan payload-algebra experiment on CUDA."""

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import struct
import time
import traceback
from pathlib import Path

import torch
import triton
import triton.language as tl

from test.lib import tensor_exact_f32, tensor_fp32_reference, tensor_route_protocol
from test.lib.tensor_fpsan import (
  compare_words,
  flatten_word_values,
  generated_region_payload_operation_lines,
  generated_region_payload_operations,
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
WARMUP_LAUNCHES = tensor_route_protocol.WARMUP_LAUNCHES
MEASURED_LAUNCHES = tensor_route_protocol.MEASURED_LAUNCHES
MEASUREMENT_BATCHES = tensor_route_protocol.MEASUREMENT_BATCHES
PAIRED_OVERHEAD_BLOCKS = MEASUREMENT_BATCHES * 2


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


def _round_f32_mul(left, right):
  if not math.isfinite(left) or not math.isfinite(right):
    raise ValueError("ordinary generated reference requires finite f32 operands")
  left_word = struct.unpack("<I", struct.pack("<f", left))[0]
  right_word = struct.unpack("<I", struct.pack("<f", right))[0]
  try:
    result_word = tensor_fp32_reference.mul_f32_rne_bits(left_word, right_word)
  except OverflowError as error:
    raise ValueError("ordinary generated reference overflowed finite f32") from error
  return struct.unpack("<f", struct.pack("<I", result_word))[0]


def _round_f32_sub(left, right):
  if not math.isfinite(left) or not math.isfinite(right):
    raise ValueError("ordinary generated reference requires finite f32 operands")
  left_word = struct.unpack("<I", struct.pack("<f", left))[0]
  right_word = struct.unpack("<I", struct.pack("<f", right))[0]
  try:
    result_word = tensor_fp32_reference.sub_f32_rne_bits(left_word, right_word)
  except OverflowError as error:
    raise ValueError("ordinary generated reference overflowed finite f32") from error
  return struct.unpack("<f", struct.pack("<I", result_word))[0]


def _ordinary_generated_region_output_words(source, input_bytes):
  """Execute admitted add/sub/mul/sum/broadcast regions with binary32 RNE."""
  shape, iterations, tensor_types, nodes, updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  count = math.prod(shape)
  if not isinstance(input_bytes, bytes) or len(input_bytes) != count * 4:
    raise ValueError(
      "ordinary generated reference input does not match the tensor shape"
    )
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
      elif operation == "sub":
        left = tensor_exact_f32._broadcast_values(
          values[operands[0]], tensor_types[operands[0]], output_shape
        )
        right = tensor_exact_f32._broadcast_values(
          values[operands[1]], tensor_types[operands[1]], output_shape
        )
        result = tuple(_round_f32_sub(a, b) for a, b in zip(left, right))
        result_words = struct.unpack(
          f"<{len(result)}I", struct.pack(f"<{len(result)}f", *result)
        )
        if any(word & 0x80000000 for word in result_words):
          raise ValueError("ordinary RNE subtraction produced a negative value")
      elif operation == "mul":
        left = tensor_exact_f32._broadcast_values(
          values[operands[0]], tensor_types[operands[0]], output_shape
        )
        right = tensor_exact_f32._broadcast_values(
          values[operands[1]], tensor_types[operands[1]], output_shape
        )
        result = tuple(_round_f32_mul(a, b) for a, b in zip(left, right))
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


def _required_fpsan_payload_operations(source, iteration_count, *, omitted_updates=()):
  if type(iteration_count) is not int or iteration_count < 0:
    raise ValueError("payload validation iteration count must be nonnegative")
  return generated_region_payload_operations(
    source,
    iteration_count=iteration_count,
    omitted_updates=omitted_updates,
  )


def _payload_operation_transition(
  source_operations, live_operations, observed_operations
):
  source_operations = set(source_operations)
  live_operations = set(live_operations)
  observed_operations = set(observed_operations)
  return {
    "live_payload_operations": sorted(live_operations),
    "eliminated_payload_operations": sorted(source_operations - live_operations),
    "unobserved_live_payload_operations": sorted(live_operations - observed_operations),
  }


def _ordinary_generated_region_admission(source, input_bytes):
  """Admit RNE arithmetic whose reduction order is fixed or provably exact."""
  shape, _, tensor_types, nodes, _ = tensor_exact_f32._generated_region_program(source)
  count = math.prod(shape)
  if not isinstance(input_bytes, bytes) or len(input_bytes) != count * 4:
    raise ValueError("ordinary reference input does not match the tensor shape")
  input_words = struct.unpack(f"<{count}I", input_bytes)
  input_values = struct.unpack(f"<{count}f", input_bytes)
  reductions = []
  operations = list(
    dict.fromkeys([operation for _, operation, _, _ in nodes] + ["add"])
  )
  for destination, operation, operands, attribute in nodes:
    if operation != "sum":
      continue
    input_shape = tensor_types[operands[0]]
    axis = attribute if attribute >= 0 else attribute + len(input_shape)
    width = input_shape[axis]
    reductions.append({"destination": destination, "axis": axis, "width": width})
  max_width = max((item["width"] for item in reductions), default=0)
  result = {
    "status": "admitted",
    "contract": ("nonnegative-finite-binary32-rne-ops-reduction-width-at-most-2-v3"),
    "operations": operations,
    "max_reduction_width": max_width,
    "reductions": reductions,
  }
  if any(not math.isfinite(value) for value in input_values):
    result.update(status="unclassified", reason="input transport is not finite f32")
  elif any(word & 0x80000000 for word in input_words):
    result.update(
      status="unclassified", reason="input transport contains a negative f32 value"
    )
  elif max_width > 2:
    unsupported = next(item for item in reductions if item["width"] > 2)
    if any(not value.is_integer() for value in input_values):
      result.update(
        status="unclassified",
        reason=(
          f"reduction width {unsupported['width']} at %{unsupported['destination']} "
          "requires integer-valued inputs for an order-independent exactness proof"
        ),
      )
    else:
      input_domain = (int(min(input_values)), int(max(input_values)))
      try:
        proof, exact_output = tensor_exact_f32.certify(
          source, input_bytes, input_domain
        )
        ordinary_output_words = _ordinary_generated_region_output_words(
          source, input_bytes
        )
        ordinary_output = struct.pack(
          f"<{len(ordinary_output_words)}I", *ordinary_output_words
        )
        if exact_output != ordinary_output:
          raise ValueError("ordinary RNE output differs from the exact integer proof")
      except ValueError as error:
        result.update(
          status="unclassified",
          reason=(
            f"reduction width {unsupported['width']} at %{unsupported['destination']} "
            f"failed the exact-integer partial-sum contract: {error}"
          ),
        )
      else:
        partial_sums = [
          bound["max_partial_sum"]
          for bound in proof["bounds"]
          if "max_partial_sum" in bound
        ]
        proof_json = json.dumps(
          proof, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        result.update(
          contract="nonnegative-integer-f32-all-partials-within-2^24-v1",
          exact_integer_proof={
            "input_domain": list(input_domain),
            "integer_exactness_bound": proof["integer_exactness_bound"],
            "max_reduction_partial_sum": max(partial_sums, default=0),
            "max_materialized_value": max(
              bound["max_value"] for bound in proof["bounds"]
            ),
            "proof_sha256": hashlib.sha256(proof_json).hexdigest(),
          },
        )
  if result["status"] == "admitted":
    try:
      _ordinary_generated_region_output_words(source, input_bytes)
    except ValueError as error:
      result.update(
        status="unclassified",
        reason=f"ordinary RNE reference is unavailable: {error}",
      )
  return result


_REFERENCE_NOT_PROVIDED = object()


def _ordinary_generated_region_comparison(
  source,
  input_bytes,
  actual_words,
  *,
  admission=None,
  reference_words=_REFERENCE_NOT_PROVIDED,
):
  admission = (
    _ordinary_generated_region_admission(source, input_bytes)
    if admission is None
    else admission
  )
  if reference_words is _REFERENCE_NOT_PROVIDED:
    try:
      reference_words = _ordinary_generated_region_output_words(source, input_bytes)
    except ValueError as error:
      reference_words = None
      reference_error = f"sequential RNE reference unavailable: {error}"
      reason = admission.get("reason")
      admission = {
        **admission,
        "status": "unclassified",
        "reason": f"{reason}; {reference_error}" if reason else reference_error,
      }
  actual_words = [int(value) & 0xFFFFFFFF for value in actual_words]
  mismatch_lanes = (
    None if reference_words is None else compare_words(actual_words, reference_words)
  )
  if admission["status"] != "admitted" or reference_words is None:
    status = "unclassified"
  elif mismatch_lanes:
    status = "FAIL"
  else:
    status = "pass"
  if mismatch_lanes is None:
    difference = "unavailable"
  elif mismatch_lanes:
    difference = "different"
  else:
    difference = "matches"
  return {
    "status": status,
    "difference": difference,
    "admission": admission,
    "mismatch_lanes": mismatch_lanes,
    "actual_words": actual_words,
    "reference_words": reference_words,
    "actual_bits": [f"{value:08x}" for value in actual_words],
    "reference_bits": None
    if reference_words is None
    else [f"{value:08x}" for value in reference_words],
    "expected_bits": None
    if reference_words is None
    else [f"{value:08x}" for value in reference_words],
    "interpretation": (
      "bitwise agreement with the sequential RNE reference is classified only "
      "inside the admitted subset; other observations are diagnostic"
    ),
  }


def _write_report_checkpoint(output_path, report):
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  temporary_path = output_path.with_name(output_path.name + ".tmp")
  temporary_path.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  os.replace(temporary_path, output_path)


def _case_artifact_inventory(case_directory):
  if not case_directory.exists():
    return []
  artifacts = []
  for path in sorted(item for item in case_directory.rglob("*") if item.is_file()):
    content = path.read_bytes()
    artifacts.append(
      {
        "path": path.relative_to(case_directory).as_posix(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
      }
    )
  return artifacts


def _generated_case_status(entries):
  statuses = [entry.get("status") for entry in entries]
  if any(status in {"ERROR", "FAIL"} for status in statuses):
    return "failed"
  if any(status in {"queued", "running"} for status in statuses):
    return "running"
  if all(status == "pass" for status in statuses):
    return "pass"
  return "incomplete"


def _refresh_report_status(report):
  generated_regions = report.get("generated_regions", {})
  programs = generated_regions.get("programs", {})
  if programs:
    generated_regions["status"] = _generated_case_status(programs.values())
  else:
    generated_regions.setdefault("status", "queued")
  entries = list(programs.values())
  authored = report.get("generated_region")
  if isinstance(authored, dict):
    entries.append(authored)
  report["status"] = _generated_case_status(entries)


def _run_generated_region_case_record(
  report,
  output_path,
  case_directory,
  entry,
  *,
  source_path,
  target,
  target_index,
  timeout_s,
):
  """Run one generated case while atomically checkpointing structured evidence."""
  case_directory = Path(case_directory)
  source_path = Path(source_path)
  started = time.perf_counter()
  entry.setdefault("source", {"path": source_path.name, "sha256": None})
  entry.setdefault(
    "property",
    None
    if target is None
    else {
      "kind": "output-payload-equals",
      "target": target,
      "index": list(target_index),
    },
  )
  entry.update(status="running", stage="start", elapsed_s=0.0, artifacts=[])
  entry["work_variants"] = {
    name: {"status": "not_started"}
    for _, name in tensor_route_protocol.GENERATED_REGION_EXECUTION_PLAN
  }
  entry["measurement_accounting"] = (
    tensor_route_protocol.generated_region_work_accounting(entry["work_variants"])
  )
  _refresh_report_status(report)
  _write_report_checkpoint(output_path, report)

  def checkpoint(stage, **fields):
    variant_updates = fields.pop("work_variant_updates", {})
    calibration_update = fields.pop("calibration_mutant_update", None)
    if calibration_update is not None:
      if (
        not isinstance(calibration_update, dict)
        or not isinstance(calibration_update.get("name"), str)
        or not isinstance(calibration_update.get("record"), dict)
      ):
        raise ValueError("calibration mutant checkpoint is malformed")
      name = calibration_update["name"]
      if entry["work_variants"].get(name, {}).get("status") != "complete":
        variant_updates.setdefault(name, {})["calibration_result"] = calibration_update[
          "record"
        ]
    for name, update in variant_updates.items():
      if not isinstance(update, dict):
        raise ValueError(f"GPU variant checkpoint for {name} must be a mapping")
    if variant_updates:
      entry["work_variants"] = (
        tensor_route_protocol.checkpoint_generated_region_variants(
          entry["work_variants"], variant_updates
        )
      )
      entry["measurement_accounting"] = (
        tensor_route_protocol.generated_region_work_accounting(entry["work_variants"])
      )
    if calibration_update is not None:
      entry.setdefault("calibration_mutants", {})[calibration_update["name"]] = (
        calibration_update["record"]
      )
    entry["stage"] = stage
    entry.update(fields)
    entry["elapsed_s"] = time.perf_counter() - started
    _refresh_report_status(report)
    _write_report_checkpoint(output_path, report)

  try:
    case_directory.mkdir(parents=True, exist_ok=True)
    if entry["source"].get("sha256") is None:
      source_bytes = source_path.read_bytes()
      entry["source"]["sha256"] = hashlib.sha256(source_bytes).hexdigest()
    result = _run_generated_region_case(
      case_directory,
      source_path=source_path,
      target=target,
      target_index=target_index,
      timeout_s=timeout_s,
      checkpoint=checkpoint,
    )
    entry.update(result)
    entry["measurement_accounting"] = (
      tensor_route_protocol.generated_region_work_accounting(entry["work_variants"])
    )
    entry.setdefault("status", "ERROR")
    entry["stage"] = "complete"
  except Exception as error:
    failure_stage = entry.get("stage", "case_execution")
    entry["status"] = "ERROR"
    entry["stage"] = "failed"
    entry["failure"] = {
      "type": type(error).__name__,
      "message": str(error),
      "stage": failure_stage,
      "traceback": traceback.format_exc()[-8000:],
    }
  entry["elapsed_s"] = time.perf_counter() - started
  try:
    entry["artifacts"] = _case_artifact_inventory(case_directory)
  except OSError as error:
    entry["artifact_inventory_error"] = f"{type(error).__name__}: {error}"
    entry["artifacts"] = []
  _refresh_report_status(report)
  _write_report_checkpoint(output_path, report)
  return entry


def _run_generated_region_corpus(
  report,
  output_path,
  ir_directory,
  sources,
  *,
  target,
  target_index,
  timeout_s,
):
  section = report["generated_regions"]
  programs = {}
  section["programs"] = programs
  section["status"] = "running"
  report["status"] = "running"

  pending = []
  for record in sources:
    program_id = record["program_id"]
    source_path = Path(record["source_path"])
    source_bytes = source_path.read_bytes()
    source_sha256 = (
      record.get("source_sha256") or hashlib.sha256(source_bytes).hexdigest()
    )
    property_record = (
      None
      if target is None
      else {
        "kind": "output-payload-equals",
        "target": target,
        "index": list(target_index),
      }
    )
    entry = {
      "program_id": program_id,
      "status": "queued",
      "stage": "queued",
      "source": {"path": source_path.name, "sha256": source_sha256},
      "generation": record.get("generation"),
      "property": property_record,
      "elapsed_s": 0.0,
      "artifacts": [],
    }
    programs[program_id] = entry
    pending.append((record, entry))
  _refresh_report_status(report)
  _write_report_checkpoint(output_path, report)
  for record, entry in pending:
    program_id = entry["program_id"]
    source_path = Path(record["source_path"])
    _run_generated_region_case_record(
      report,
      output_path,
      Path(ir_directory) / "generated_regions" / program_id,
      entry,
      source_path=source_path,
      target=target,
      target_index=target_index,
      timeout_s=timeout_s,
    )
    section["status"] = _generated_case_status(programs.values())
    _refresh_report_status(report)
    _write_report_checkpoint(output_path, report)
  section["status"] = _generated_case_status(programs.values())
  _write_report_checkpoint(output_path, report)
  return programs


def _generated_region_corpus_sources(program_ids=None):
  program_dir = Path(__file__).resolve().parents[1] / "gluon" / "programs"
  manifest_path = program_dir / "native-generated-corpus.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  if manifest.get("schema") != "refractir.generated-region-corpus.v1":
    raise ValueError(f"unsupported generated-region corpus: {manifest_path}")
  records = [
    *manifest["programs"],
    *manifest.get("supplemental_programs", []),
  ]
  requested_program_ids = None if program_ids is None else set(program_ids)
  known_program_ids = {record["program_id"] for record in records}
  if requested_program_ids is not None:
    if not requested_program_ids or requested_program_ids - known_program_ids:
      unknown = sorted(requested_program_ids - known_program_ids)
      raise ValueError(f"unknown or empty generated program selection: {unknown}")
  sources = []
  for record in records:
    if (
      requested_program_ids is not None
      and record["program_id"] not in requested_program_ids
    ):
      continue
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
        "generation": record.get("generation"),
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
  checkpoint=None,
):
  case_started = time.perf_counter()
  ir_directory = Path(ir_directory)
  ir_directory.mkdir(parents=True, exist_ok=True)
  source_path = (
    Path(__file__).with_name("generated_region.sir")
    if source_path is None
    else Path(source_path)
  )

  def note(stage, **fields):
    if checkpoint is not None:
      checkpoint(stage, **fields)

  note("load_source", source={"path": source_path.name})
  source = source_path.read_text(encoding="utf-8")
  source_identity = {
    "path": source_path.name,
    "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
  }
  requested_property = (
    None
    if target is None
    else {
      "kind": "output-payload-equals",
      "target": target,
      "index": list(target_index),
    }
  )
  note(
    "smt_synthesis",
    region_source=source_identity,
    requested_property=requested_property,
  )
  witness = synthesize_generated_region(
    source,
    target=target,
    target_index=target_index,
    timeout_s=timeout_s,
  )
  query_smt2 = witness.get("query_smt2")
  if isinstance(query_smt2, str) and query_smt2:
    (ir_directory / "selection-query.smt2").write_text(query_smt2, encoding="utf-8")
  solver_record = {
    "status": witness.get("status"),
    "property": witness.get("property"),
    "cost_s": witness.get("cost_s"),
    "solver": witness.get("solver"),
    "query_sha256": witness.get("query_sha256"),
    "solver_outcome": witness.get("solver_outcome"),
  }
  note("smt_terminal", selection=solver_record)
  if witness["status"] != "SAT":
    return {
      "status": witness["status"],
      "elapsed_s": time.perf_counter() - case_started,
      "property": witness.get("property"),
      "cost_s": witness.get("cost_s"),
      "transport_contract": witness.get("transport_contract"),
      "solver": witness.get("solver"),
      "query_sha256": witness.get("query_sha256"),
      "query_file": "selection-query.smt2" if query_smt2 else None,
      "solver_outcome": witness.get("solver_outcome"),
      "region_source": source_identity,
    }
  shape = tuple(witness["shape"])
  smt_witness = {
    "transport_contract": witness["transport_contract"],
    "query_sha256": witness["query_sha256"],
    "input_payloads": witness["input_payloads"],
    "output_payloads": witness["output_payloads"],
    "input_diversity": witness["input_diversity"],
    "input_float_words": witness["input_float_words"],
    "output_float_words": witness["output_float_words"],
    "input_float_bits": witness["input_float_bits"],
    "output_float_bits": witness["output_float_bits"],
    "calibration_mutants": witness["calibration_mutants"],
    "property": witness["property"],
    "cost_s": witness["cost_s"],
  }
  note("smt_witness", smt_witness=smt_witness)
  inputs = _tensor_from_words(witness["input_float_words"]).reshape(shape)

  # The source is rendered only from the admitted region parsed by the host model.
  kernel_source_path = ir_directory / "generated_region_kernel.py"
  kernel_source = render_generated_region_triton_kernel(source)
  kernel_source_sha256 = hashlib.sha256(kernel_source.encode("utf-8")).hexdigest()
  kernel_source_path.write_text(kernel_source, encoding="utf-8")
  namespace = {}
  exec(compile(kernel_source, str(kernel_source_path), "exec"), namespace)
  kernel = namespace["fpsan_generated_region"]
  note(
    "fpsan_execution",
    input_float_words=witness["input_float_words"],
    work_variant_updates={"fpsan": {"status": "running"}},
  )
  output, compiled, fpsan_launch, fpsan_compile_wall_s = _compile_generated_kernel_mode(
    kernel, inputs, shape, "fpsan"
  )
  fpsan_words = _raw_words(output)
  fpsan_ir = _capture_ir(compiled, "generated_region_fpsan", ir_directory)
  note(
    "fpsan_output_captured",
    fpsan_output_words=fpsan_words,
    fpsan_expected_words=witness["output_float_words"],
    fpsan_first_launch_wall_s_including_compilation=fpsan_compile_wall_s,
    fpsan_compiler_ir=fpsan_ir,
    work_variant_updates={
      "fpsan": {
        "status": "running",
        "compiler": fpsan_ir,
        "execution_observed": True,
      }
    },
  )
  fpsan_result = _expect_words(
    "generated_region", output, witness["output_float_words"]
  )
  result = {
    **fpsan_result,
    "smt_witness": smt_witness,
    "fpsan_output_words": fpsan_words,
  }
  result["payload_lowering"] = _validate_generated_region_payload_lowering(
    compiled,
    required_payload_operations=_required_fpsan_payload_operations(
      source, witness["iterations"]
    ),
    payload_source_lines=generated_region_payload_operation_lines(
      source, iteration_count=witness["iterations"]
    ),
    payload_source_path=kernel_source_path,
  )
  result["compiler_ir"] = fpsan_ir
  result["compiler_modes"] = {"fpsan": fpsan_ir}

  input_bytes = struct.pack(
    f"<{len(witness['input_float_words'])}I", *witness["input_float_words"]
  )
  ordinary_admission = _ordinary_generated_region_admission(source, input_bytes)
  try:
    ordinary_expected_words = _ordinary_generated_region_output_words(
      source, input_bytes
    )
  except ValueError as error:
    ordinary_expected_words = None
    reference_error = f"sequential RNE reference unavailable: {error}"
    reason = ordinary_admission.get("reason")
    ordinary_admission = {
      **ordinary_admission,
      "status": "unclassified",
      "reason": f"{reason}; {reference_error}" if reason else reference_error,
    }
  note(
    "ordinary_execution",
    ordinary_reference_admission=ordinary_admission,
    ordinary_reference_words=ordinary_expected_words,
    work_variant_updates={"ordinary": {"status": "running"}},
  )
  ordinary_output, ordinary_compiled, ordinary_launch, ordinary_compile_wall_s = (
    _compile_generated_kernel_mode(kernel, inputs, shape, "")
  )
  ordinary_words = _raw_words(ordinary_output)
  ordinary_ir = _capture_ir(
    ordinary_compiled, "generated_region_ordinary", ir_directory
  )
  ordinary_comparison = _ordinary_generated_region_comparison(
    source,
    input_bytes,
    ordinary_words,
    admission=ordinary_admission,
    reference_words=ordinary_expected_words,
  )
  ordinary_comparison["compiler_ir"] = ordinary_ir
  paired_measurements, paired_overhead = _measure_paired_generated_modes(
    {"fpsan": fpsan_launch, "ordinary": ordinary_launch}
  )
  paired_protocol = {
    "kernel_source_sha256": kernel_source_sha256,
    "loop_form": "tl.static_range from the same rendered kernel source",
    "layout_policy": (
      "Triton-selected layout from the same source, input shape, num_warps and fusion setting"
    ),
    "input_float_words_sha256": hashlib.sha256(input_bytes).hexdigest(),
    "selection_timeout_s_per_source": timeout_s,
    "measurement_scope": (
      "GPU event time only; synthesis and first-launch compilation are excluded "
      "and reported separately"
    ),
    "mode_contracts": {
      "fpsan": "pinned FpSan payload-ring oracle",
      "ordinary": "independent binary32 RNE reference where admitted",
    },
  }
  fpsan_measurement = {
    **paired_measurements["fpsan"],
    "first_launch_wall_s_including_compilation": fpsan_compile_wall_s,
  }
  ordinary_measurement = {
    **paired_measurements["ordinary"],
    "first_launch_wall_s_including_compilation": ordinary_compile_wall_s,
  }
  ordinary_comparison["measurement"] = ordinary_measurement
  note(
    "ordinary_output_captured",
    ordinary_reference=ordinary_comparison,
    ordinary_compiler_ir=ordinary_ir,
  )
  note(
    "paired_runtime_measured",
    fpsan_measurement=fpsan_measurement,
    ordinary_measurement=ordinary_measurement,
    paired_overhead={**paired_overhead, "protocol": paired_protocol},
    work_variant_updates={
      "fpsan": {
        "status": "complete",
        "compiler": fpsan_ir,
        "execution_observed": True,
        "measurement": fpsan_measurement,
      },
      "ordinary": {
        "status": "complete",
        "compiler": ordinary_ir,
        "execution_observed": True,
        "measurement": ordinary_measurement,
      },
    },
  )
  result["measurement"] = fpsan_measurement
  result["compiler_modes"]["ordinary"] = ordinary_ir
  result["ordinary_reference"] = ordinary_comparison
  result["paired_overhead"] = {**paired_overhead, "protocol": paired_protocol}
  if ordinary_comparison["status"] == "FAIL":
    result["status"] = "FAIL"
    result["failure"] = {
      "type": "AssertionError",
      "message": "ordinary GPU output differs from the admitted RNE reference",
      "stage": "ordinary_output_captured",
    }
  result["compiler_mode_comparison"] = _compare_generated_compiler_modes(
    fpsan_ir, ordinary_ir
  )
  result["region_source"] = source_identity
  result["calibration_mutants"] = {}
  source_payload_operations = set(
    _required_fpsan_payload_operations(source, witness["iterations"])
  )
  for group, mutant_name in tensor_route_protocol.GENERATED_REGION_EXECUTION_PLAN:
    if group != "calibration":
      continue
    if mutant_name == "lost_iteration":
      render_options = {"iteration_count": witness["iterations"] - 1}
    elif mutant_name == "lost_update_acc_y":
      render_options = {"omitted_updates": ("acc_y",)}
    elif mutant_name == "lane_permutation":
      render_options = {"lane_permutation": True}
    else:
      raise ValueError(f"no generated-region calibration renderer for {mutant_name}")
    note(
      f"{mutant_name}_execution",
      work_variant_updates={mutant_name: {"status": "running"}},
    )
    mutant_kernel_source = render_generated_region_triton_kernel(
      source, **render_options
    )
    mutant_iteration_count = render_options.get(
      "iteration_count", witness["iterations"]
    )
    mutant_payload_operations = set(
      _required_fpsan_payload_operations(
        source,
        mutant_iteration_count,
        omitted_updates=render_options.get("omitted_updates", ()),
      )
    )
    mutant_payload_source_lines = generated_region_payload_operation_lines(
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
      lambda: mutant_kernel[(1,)](
        inputs,
        mutant_output,
        num_warps=tensor_route_protocol.NUM_WARPS,
        enable_fp_fusion=tensor_route_protocol.ENABLE_FP_FUSION,
      )
    )
    actual_mutant_words = _raw_words(mutant_output)
    mutant_ir = _capture_ir(
      mutant_compiled, f"generated_region_{mutant_name}", ir_directory
    )
    expected_mutant_words = [
      int(value, 16)
      for value in witness["calibration_mutants"][mutant_name]["candidate_float_bits"]
    ]
    note(
      f"{mutant_name}_output_captured",
      calibration_mutant_update={
        "name": mutant_name,
        "record": {
          "status": "output_captured",
          "actual_words": actual_mutant_words,
          "expected_words": expected_mutant_words,
          "measurement": mutant_measurement,
          "compiler_ir": mutant_ir,
        },
      },
      work_variant_updates={
        mutant_name: {
          "status": "complete",
          "compiler": mutant_ir,
          "execution_observed": True,
          "measurement": mutant_measurement,
        }
      },
    )
    if actual_mutant_words != expected_mutant_words:
      raise AssertionError(
        f"{mutant_name} GPU output differs from its synthesized payload model"
      )
    mismatch_lanes = compare_words(actual_mutant_words, witness["output_float_words"])
    if not mismatch_lanes:
      raise AssertionError(f"generated-region {mutant_name} mutant was not detected")
    mutant_payload_lowering = _validate_generated_region_payload_lowering(
      mutant_compiled,
      required_payload_operations=(),
      payload_source_lines=mutant_payload_source_lines,
      payload_source_path=mutant_source_path,
    )
    observed_payload_operations = set(
      mutant_payload_lowering["ttgir"]["payload_operation_counts"]
    )
    operation_transition = _payload_operation_transition(
      source_payload_operations,
      mutant_payload_operations,
      observed_payload_operations,
    )
    result["calibration_mutants"][mutant_name] = {
      "status": "detected",
      "mismatch_lanes": mismatch_lanes,
      "actual_bits": [f"{value:08x}" for value in actual_mutant_words],
      "expected_bits": [f"{value:08x}" for value in expected_mutant_words],
      "measurement": mutant_measurement,
      **operation_transition,
      "payload_lowering": mutant_payload_lowering,
      "compiler_ir": mutant_ir,
    }
    note(
      f"{mutant_name}_validated",
      calibration_mutant_update={
        "name": mutant_name,
        "record": result["calibration_mutants"][mutant_name],
      },
    )
  result["elapsed_s"] = time.perf_counter() - case_started
  note("complete")
  return result


def _validate_generated_region_payload_lowering(
  compiled,
  *,
  required_payload_operations,
  payload_source_lines=None,
  payload_source_path=None,
):
  """Check FpSan integerization in TTGIR and the final LLVM float guard."""
  assembly = getattr(compiled, "asm", {})
  return {
    "ttgir": validate_fpsan_payload_lowering(
      assembly.get("ttgir", ""),
      stage="ttgir",
      required_payload_operations=required_payload_operations,
      payload_source_lines=payload_source_lines,
      payload_source_path=payload_source_path,
    ),
    "llir": validate_fpsan_payload_lowering(
      assembly.get("llir", ""),
      stage="llvm",
      required_payload_operations=(),
    ),
  }


def _compile_generated_kernel_mode(kernel, inputs, shape, instrumentation_mode):
  output = torch.empty(shape, dtype=torch.float32, device="cuda")

  def launch():
    previous_mode = triton.knobs.compilation.instrumentation_mode
    try:
      triton.knobs.compilation.instrumentation_mode = instrumentation_mode
      return kernel[(1,)](
        inputs,
        output,
        num_warps=tensor_route_protocol.NUM_WARPS,
        enable_fp_fusion=tensor_route_protocol.ENABLE_FP_FUSION,
      )
    finally:
      triton.knobs.compilation.instrumentation_mode = previous_mode

  started = time.perf_counter()
  compiled = launch()
  torch.cuda.synchronize()
  compilation_wall_s = time.perf_counter() - started
  return output, compiled, launch, compilation_wall_s


def _paired_measurement_order(block_index):
  if type(block_index) is not int or block_index < 0:
    raise ValueError("paired measurement block index must be a nonnegative integer")
  if block_index % 2 == 0:
    return ("fpsan", "ordinary")
  return ("ordinary", "fpsan")


def _summarize_paired_overhead(blocks):
  if not isinstance(blocks, (list, tuple)) or not blocks:
    raise ValueError("paired overhead blocks must be a nonempty sequence")

  ordinary_ms = []
  fpsan_ms = []
  paired_ratios = []
  order_counts = {"fpsan_outer": 0, "ordinary_outer": 0}
  for index, block in enumerate(blocks):
    if not isinstance(block, dict):
      raise ValueError("paired overhead block must be a mapping")
    order = block.get("order")
    if not isinstance(order, (list, tuple)) or tuple(
      order
    ) != _paired_measurement_order(index):
      raise ValueError("paired overhead blocks require counterbalanced mode order")
    values = {}
    for mode in ("ordinary", "fpsan"):
      value = block.get(f"{mode}_ms")
      if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("paired overhead latencies must be finite positive numbers")
      values[mode] = float(value)
    ordinary_ms.append(values["ordinary"])
    fpsan_ms.append(values["fpsan"])
    paired_ratios.append(values["fpsan"] / values["ordinary"])
    order_counts["fpsan_outer" if order[0] == "fpsan" else "ordinary_outer"] += 1

  ratio_median = statistics.median(paired_ratios)
  return {
    "block_count": len(blocks),
    "ordinary_median_ms": statistics.median(ordinary_ms),
    "fpsan_median_ms": statistics.median(fpsan_ms),
    "paired_ratio_median": ratio_median,
    "paired_overhead_median_percent": (ratio_median - 1.0) * 100.0,
    "paired_ratio_range": [min(paired_ratios), max(paired_ratios)],
    "order_counts": order_counts,
    "interpretation_limit": (
      "descriptive per-source GPU-event comparison; the small paired sample "
      "does not establish statistical significance or general sanitizer overhead"
    ),
  }


def _measure_paired_generated_modes(launchers):
  if not isinstance(launchers, dict) or set(launchers) != {"fpsan", "ordinary"}:
    raise ValueError("paired mode launchers must contain fpsan and ordinary")
  if MEASURED_LAUNCHES % PAIRED_OVERHEAD_BLOCKS != 0:
    raise RuntimeError(
      "paired measurements must divide evenly into counterbalanced blocks"
    )

  for warmup in range(WARMUP_LAUNCHES):
    order = ("fpsan", "ordinary") if warmup % 2 == 0 else ("ordinary", "fpsan")
    for mode in order:
      launchers[mode]()
  torch.cuda.synchronize()

  launches_per_subsample = MEASURED_LAUNCHES // PAIRED_OVERHEAD_BLOCKS
  blocks = []
  for block_index in range(PAIRED_OVERHEAD_BLOCKS):
    order = _paired_measurement_order(block_index)
    samples = {}
    for mode in order:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      start_event.record()
      for _ in range(launches_per_subsample):
        launchers[mode]()
      end_event.record()
      torch.cuda.synchronize()
      samples[mode] = start_event.elapsed_time(end_event) / launches_per_subsample
    blocks.append(
      {
        "order": list(order),
        "fpsan_ms": samples["fpsan"],
        "ordinary_ms": samples["ordinary"],
      }
    )

  summary = _summarize_paired_overhead(blocks)
  measurements = {
    mode: {
      "warmup_launches": WARMUP_LAUNCHES,
      "measured_launches": MEASURED_LAUNCHES,
      "measurement_batches": MEASUREMENT_BATCHES,
      "launches_per_batch": MEASURED_LAUNCHES // MEASUREMENT_BATCHES,
      "paired_measurement_blocks": PAIRED_OVERHEAD_BLOCKS,
      "paired_blocks_per_batch": PAIRED_OVERHEAD_BLOCKS // MEASUREMENT_BATCHES,
      "launches_per_subsample": launches_per_subsample,
      "median_gpu_event_ms_per_launch": summary[f"{mode}_median_ms"],
    }
    for mode in ("fpsan", "ordinary")
  }
  return measurements, {**summary, "blocks": blocks}


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
      metric: fpsan_ir.get(metric, 0) - ordinary_ir.get(metric, 0) for metric in metrics
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
  launches_per_sample = MEASURED_LAUNCHES // tensor_route_protocol.MEASUREMENT_BATCHES
  for _ in range(tensor_route_protocol.MEASUREMENT_BATCHES):
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
    "measurement_batches": tensor_route_protocol.MEASUREMENT_BATCHES,
    "launches_per_batch": launches_per_sample,
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


def _run_experiment(
  output_path,
  ir_directory,
  *,
  generated_target=None,
  generated_target_index=(0, 0),
  generated_timeout_s=30.0,
  generated_program_ids=None,
  cache_record,
):
  output_path = Path(output_path)
  ir_directory = Path(ir_directory)
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
  generated_region_corpus = _generated_region_corpus_sources(generated_program_ids)
  requested_property = (
    None
    if generated_target is None
    else {
      "kind": "output-payload-equals",
      "target": generated_target,
      "index": list(generated_target_index),
    }
  )
  generated_region_result = {
    "program_id": "authored-generated-region",
    "status": "queued",
    "stage": "queued",
    "property": None,
  }
  generated_region_results = {}
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
      "property": requested_property,
      "selection_budget_s_per_program": generated_timeout_s,
      "gpu_launch_protocol": {
        "warmup_launches": WARMUP_LAUNCHES,
        "measured_launches_per_kernel": MEASURED_LAUNCHES,
        "measurement_batches": tensor_route_protocol.MEASUREMENT_BATCHES,
        "num_warps": tensor_route_protocol.NUM_WARPS,
        "enable_fp_fusion": tensor_route_protocol.ENABLE_FP_FUSION,
      },
      "paired_overhead_protocol": {
        "comparison": "ordinary and FpSan compile the same rendered Triton kernel per source",
        "loop_form": "tl.static_range",
        "layout_policy": "Triton-selected from that same source and fixed launch configuration",
        "input_pairing": "same raw f32 solver witness for both modes per source",
        "selection_budget_s_per_source": generated_timeout_s,
        "measured_launches_per_mode": MEASURED_LAUNCHES,
        "measurement_batches": MEASUREMENT_BATCHES,
        "paired_measurement_blocks": PAIRED_OVERHEAD_BLOCKS,
        "paired_blocks_per_batch": PAIRED_OVERHEAD_BLOCKS // MEASUREMENT_BATCHES,
        "within_batch_order": "alternating FpSan-first and ordinary-first pairs",
        "timing_scope": "GPU event time; excludes selection and compilation",
        "compilation_wall_time": "recorded separately per mode",
      },
      "selection_policy": (
        "SMT payload model with nonuniform-input, loop-mutation, and "
        "lane-permutation sensitivity constraints"
      ),
      "transport_contract": (
        "triton-fpsan-f32-mixer-e2e356723ce35f46b5144373b145c55c4d712237"
      ),
      "ordinary_reference_contract": {
        "admission": (
          "nonnegative-finite-binary32-rne-ops-reduction-width-at-most-2-v3"
        ),
        "excluded_case_policy": (
          "run and retain actual/reference words as unclassified diagnostic evidence"
        ),
      },
    },
    "compute_capability": list(torch.cuda.get_device_capability()),
    "bitvector_laws": "unsat for negated QF_BV identities",
    "bitvector_kernel_equations": equation_proofs,
    "measurement_protocol": {
      "warmup_launches_per_case": WARMUP_LAUNCHES,
      "measured_launches_per_case": MEASURED_LAUNCHES,
      "measurement_batches": tensor_route_protocol.MEASUREMENT_BATCHES,
      "num_warps": tensor_route_protocol.NUM_WARPS,
      "enable_fp_fusion": tensor_route_protocol.ENABLE_FP_FUSION,
      "selection": "fixed documented embeddings; no payload-dependent control flow",
    },
    "cache": cache_record,
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
      "status": "queued",
      "programs": generated_region_results,
    },
    "status": "running",
  }

  _write_report_checkpoint(output_path, report)
  _run_generated_region_case_record(
    report,
    output_path,
    ir_directory / "generated_region",
    report["generated_region"],
    source_path=Path(__file__).with_name("generated_region.sir"),
    target=None,
    target_index=generated_target_index,
    timeout_s=generated_timeout_s,
  )
  _run_generated_region_corpus(
    report,
    output_path,
    ir_directory,
    generated_region_corpus,
    target=generated_target,
    target_index=generated_target_index,
    timeout_s=generated_timeout_s,
  )
  generated_region_result = report["generated_region"]
  generated_region_results = report["generated_regions"]["programs"]
  report["generated_regions"]["measurement_accounting"] = (
    tensor_route_protocol.aggregate_generated_region_work_accounting(
      entry["measurement_accounting"] for entry in generated_region_results.values()
    )
  )
  report["status"] = "running"
  report["stage"] = "fixed_payload_kernel_cases"
  _write_report_checkpoint(output_path, report)

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
  generated_status = _generated_case_status(
    [generated_region_result, *generated_region_results.values()]
  )
  report["generated_regions"]["status"] = generated_status
  report["status"] = generated_status
  report["stage"] = "complete"
  _write_report_checkpoint(output_path, report)
  return report


def run(
  output_path,
  ir_directory,
  *,
  generated_target=None,
  generated_target_index=(0, 0),
  generated_timeout_s=30.0,
  generated_program_ids=None,
  cache_dir=None,
):
  ir_directory = Path(ir_directory)
  cache_path = (
    Path(cache_dir) if cache_dir is not None else ir_directory / "triton-cache"
  )
  cache_record = tensor_route_protocol.prepare_fresh_cache(cache_path)
  previous_cache = os.environ.get("TRITON_CACHE_DIR")
  os.environ["TRITON_CACHE_DIR"] = cache_record["path"]
  try:
    return _run_experiment(
      output_path,
      ir_directory,
      generated_target=generated_target,
      generated_target_index=generated_target_index,
      generated_timeout_s=generated_timeout_s,
      generated_program_ids=generated_program_ids,
      cache_record=cache_record,
    )
  finally:
    if previous_cache is None:
      os.environ.pop("TRITON_CACHE_DIR", None)
    else:
      os.environ["TRITON_CACHE_DIR"] = previous_cache


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", default="fpsan-experiment.json")
  parser.add_argument("--ir-dir", default="fpsan-ir")
  parser.add_argument("--generated-target", type=int, default=None)
  parser.add_argument("--generated-target-index", default="0,0")
  parser.add_argument("--generated-timeout-s", type=float, default=30.0)
  parser.add_argument("--generated-program-ids", default=None)
  parser.add_argument("--cache-dir", default=None)
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
    generated_program_ids=(
      None
      if args.generated_program_ids is None
      else tuple(
        value.strip()
        for value in args.generated_program_ids.split(",")
        if value.strip()
      )
    ),
    cache_dir=args.cache_dir,
  )
  print(json.dumps(report, indent=2, sort_keys=True))
  if report.get("status") != "pass":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
