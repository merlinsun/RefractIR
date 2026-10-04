#!/usr/bin/env python3
"""Compare random and SMT inputs on exact generated regions and V100 layouts."""

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

from test.lib import (
  tensor_evaluation,
  tensor_exact_f32,
  tensor_fixed_study,
)

ROOT = Path(__file__).resolve().parents[3]
DOMAIN = (0, 8)
TARGET = 63
TARGET_INDEX = (0, 0)
ITERATIONS = 2
ROUTE = "exact-f32"
SEMANTIC_CONTRACT = "composed-add-sum-f32-exact-loop-v1"
PROPERTY_ID = "output-element-0-0-equals-63"
PROGRAM_ID = "authored-2x2-two-state-two-iteration-loop-v1"
DOMAIN_ID = "nonnegative-integer-valued-f32-0-through-8"
COMPILER_SETTINGS_ID = (
  "triton-3.8.0-v100-num-warps-1-fp-fusion-off-layout-seed-20261002-pass-dump-v1"
)
LAYOUT_SEED = 20261002
MATRIX_INPUT_VALUES = (1, 2, 4, 8)
DEFAULT_BUDGET_S = 120.0
SOLVER_TIMEOUT_S = 60.0
PROGRAM_DIR = Path(__file__).with_name("programs")
_NATIVE_CORPUS_PATH = PROGRAM_DIR / "native-generated-corpus.json"
_NATIVE_CORPUS = json.loads(_NATIVE_CORPUS_PATH.read_text(encoding="utf-8"))
if _NATIVE_CORPUS.get("schema") != "refractir.generated-region-corpus.v1":
  raise ValueError(f"unsupported native generated-region corpus: {_NATIVE_CORPUS_PATH}")
NATIVE_PROGRAM_RECORDS = {
  record["program_id"]: record for record in _NATIVE_CORPUS["programs"]
}
NATIVE_PROGRAM_SOURCES = {
  program_id: PROGRAM_DIR / record["source_file"]
  for program_id, record in NATIVE_PROGRAM_RECORDS.items()
}
PROGRAM_SOURCES = {
  "row-reduction-v1": PROGRAM_DIR / "row-reduction-v1.sir",
  "column-reduction-v1": PROGRAM_DIR / "column-reduction-v1.sir",
  "two-axis-reduction-v1": PROGRAM_DIR / "two-axis-reduction-v1.sir",
  **NATIVE_PROGRAM_SOURCES,
}
CALIBRATION_CONTROL_CONTRACT = [
  {"control_id": "correct-layout-baseline", "expected_detection": False},
  {"control_id": "add-one-output-mutant", "expected_detection": True},
]


def _write_json(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  temporary.replace(path)


def _empty_costs():
  return {
    "generation_s": 0.0,
    "selection_s": 0.0,
    "input_materialization_s": 0.0,
    "reference_validation_s": 0.0,
    "compilation_s": 0.0,
    "gpu_execution_s": 0.0,
  }


def _text_output(value):
  if isinstance(value, bytes):
    return value.decode("utf-8", errors="replace")
  return value or ""


def _trial_record(policy, seed, budget_s):
  return {
    "route": ROUTE,
    "semantic_contract": SEMANTIC_CONTRACT,
    "property_id": PROPERTY_ID,
    "selection_policy": policy,
    "trial": f"seed-{seed}",
    "program_id": PROGRAM_ID,
    "input_domain_id": DOMAIN_ID,
    "compiler_settings_id": COMPILER_SETTINGS_ID,
    "budget_s": budget_s,
    "elapsed_s": 0.0,
    "admission_status": "refused",
    "execution_status": "not_run",
    "cost_s": _empty_costs(),
    "calibration_controls": [],
    "compiler_observation": None,
  }


def render_selection_query(*, iterations, target, target_index, source=None):
  """Render the integer twin for a selected loop and full-output element."""
  if source is not None:
    return tensor_exact_f32.render_generated_region_query(
      source, DOMAIN, target, target_index=target_index
    )
  if type(iterations) is not int or not 1 <= iterations <= 8:
    raise ValueError("iterations must be in [1,8]")
  if type(target) is not int:
    raise ValueError("selection target must be an integer")
  if (
    not isinstance(target_index, (tuple, list))
    or len(target_index) != 2
    or any(type(value) is not int or not 0 <= value < 2 for value in target_index)
  ):
    raise ValueError("target_index must identify an element of the 2x2 output")
  source = tensor_exact_f32.render_program(iterations=iterations, dtype="i32")
  source = source.replace(
    "fun @kernel(%x: tensor [2, 2] i32) : tensor [2, 2] i32 {",
    "fun @kernel() : tensor [2, 2] i32 {\n"
    "  sym %?x : value tensor [2, 2] i32 in [0, 8];",
  )
  source = source.replace("%x", "%?x")
  source = source.replace("^entry:\n", "  let mut %selected: i32 = undef;\n^entry:\n")
  source = source.replace(
    "  ret %out;",
    f"  %selected = tensor extract %out, [{target_index[0]}, {target_index[1]}];\n"
    f"  require %selected == {target};\n  ret %out;",
  )
  return source


def _exact_loop_path(iterations):
  if type(iterations) is not int or not 1 <= iterations <= 8:
    raise ValueError("iterations must be in [1,8]")
  return (
    ["^entry", "^header"]
    + [block for _ in range(iterations) for block in ("^body", "^header")]
    + ["^exit"]
  )


def _matrix_program_source(program_id):
  try:
    source_path = PROGRAM_SOURCES[program_id]
  except KeyError as error:
    raise ValueError(f"unknown generated selection program: {program_id}") from error
  source = source_path.read_text(encoding="utf-8")
  record = NATIVE_PROGRAM_RECORDS.get(program_id)
  if record is not None and hashlib.sha256(source.encode("utf-8")).hexdigest() != record[
    "source_sha256"
  ]:
    raise ValueError(f"frozen native generated source hash mismatch: {program_id}")
  tensor_exact_f32._generated_region_program(source)
  return source


def _matrix_cell_specs(*, program_ids, target_indices, seeds, layout_seeds):
  try:
    axes = {
      "program_ids": tuple(program_ids),
      "target_indices": tuple(tuple(index) for index in target_indices),
      "seeds": tuple(seeds),
      "layout_seeds": tuple(layout_seeds),
    }
  except TypeError as error:
    raise ValueError("selection matrix axes must be iterable") from error
  if any(not values for values in axes.values()):
    raise ValueError("selection matrix axes must not be empty")
  if any(
    not isinstance(program_id, str) or program_id not in PROGRAM_SOURCES
    for program_id in axes["program_ids"]
  ) or len(set(axes["program_ids"])) != len(axes["program_ids"]):
    raise ValueError("matrix program ids must be known and unique")
  if any(
    len(index) != 2
    or any(type(value) is not int or not 0 <= value < 2 for value in index)
    for index in axes["target_indices"]
  ) or len(set(axes["target_indices"])) != len(axes["target_indices"]):
    raise ValueError("matrix target indices must be unique elements of the 2x2 output")
  for name in ("seeds", "layout_seeds"):
    if any(type(value) is not int or value < 0 for value in axes[name]) or len(
      set(axes[name])
    ) != len(axes[name]):
      raise ValueError(
        f"matrix {name.replace('_', ' ')} must be unique nonnegative integers"
      )
  return [
    {
      "cell_id": (
        f"{program_id}-output-{index[0]}-{index[1]}-seed-{seed}-layout-{layout_seed}"
      ),
      "program_id": program_id,
      "target_index": index,
      "seed": seed,
      "layout_seed": layout_seed,
    }
    for program_id in axes["program_ids"]
    for index in axes["target_indices"]
    for seed in axes["seeds"]
    for layout_seed in axes["layout_seeds"]
  ]


def _gpu_child_command(
  gpu_report_path,
  ir_dir,
  *,
  input_values,
  iterations,
  target,
  target_index,
  layout_seed,
  program_source=None,
):
  if program_source is not None:
    return [
      sys.executable,
      "-m",
      "test.tensor.gluon.run_generated_region",
      "--program",
      str(program_source),
      "--output",
      str(gpu_report_path),
      "--ir-dir",
      str(ir_dir),
      "--input-values",
      ",".join(str(value) for value in input_values),
      "--domain",
      f"{DOMAIN[0]},{DOMAIN[1]}",
      "--target",
      str(target),
      "--target-index",
      f"{target_index[0]},{target_index[1]}",
      "--layout-seed",
      str(layout_seed),
    ]
  return [
    sys.executable,
    "-m",
    "test.tensor.gluon.run_exact_loop",
    "--output",
    str(gpu_report_path),
    "--ir-dir",
    str(ir_dir),
    "--input-values",
    ",".join(str(value) for value in input_values),
    "--iterations",
    str(iterations),
    "--target",
    str(target),
    "--target-index",
    f"{target_index[0]},{target_index[1]}",
    "--layout-seed",
    str(layout_seed),
  ]


def _combined_compiler_observation(gpu_report):
  conversions = []
  stages = {}
  effective_operations = set()
  loop_statuses = []
  compiler_records = []
  pass_compilations = []
  for name, case in gpu_report["cases"].items():
    compiler_records.append(case["compiler"]["observation"])
    pass_record = case["compiler"].get("pass_observation")
    if isinstance(pass_record, dict):
      pass_compilations.append((name, pass_record))
  calibration = gpu_report.get("calibration_compiler")
  if isinstance(calibration, dict) and isinstance(calibration.get("observation"), dict):
    compiler_records.append(calibration["observation"])
    pass_record = calibration.get("pass_observation")
    if isinstance(pass_record, dict):
      pass_compilations.append(("calibration-add-one-output-mutant", pass_record))
  pass_transitions = []
  pass_snapshot_count = 0
  for compilation_name, compilation in pass_compilations:
    pass_snapshot_count += compilation.get("snapshot_count", 0)
    for transition in compilation.get("transitions", []):
      pass_transitions.append({"compilation": compilation_name, **transition})
  nonzero_layout_transitions = [
    {
      "compilation": transition["compilation"],
      "pass": transition["pass"],
      "delta": transition["effective_layout_conversion_delta"],
      "operation_deltas": transition["effective_conversion_operation_deltas"],
    }
    for transition in pass_transitions
    if transition["effective_layout_conversion_delta"] != 0
  ]
  pass_observation = {
    "compilation_count": len(pass_compilations),
    "snapshot_count": pass_snapshot_count,
    "transition_count": len(pass_transitions),
    "effective_layout_conversion_added": sum(
      max(0, transition["effective_layout_conversion_delta"])
      for transition in pass_transitions
    ),
    "effective_layout_conversion_removed": sum(
      max(0, -transition["effective_layout_conversion_delta"])
      for transition in pass_transitions
    ),
    "effective_layout_conversion_net_delta": sum(
      transition["effective_layout_conversion_delta"] for transition in pass_transitions
    ),
    "effective_conversion_operation_net_deltas": {},
    "nonzero_layout_conversion_transitions": nonzero_layout_transitions,
    "duplicate_expression_group_delta": sum(
      transition["duplicate_expression_group_delta"] for transition in pass_transitions
    ),
    "duplicate_repeated_operation_delta": sum(
      transition["duplicate_repeated_operation_delta"]
      for transition in pass_transitions
    ),
    "loop_construct_delta": sum(
      transition["loop_construct_delta"] for transition in pass_transitions
    ),
    "interpretation_limit": (
      "pass snapshots measure IR shape changes; duplicate-expression changes are "
      "not a proof of rematerialization"
    ),
  }
  for transition in pass_transitions:
    for operation, delta in transition["effective_conversion_operation_deltas"].items():
      operation_deltas = pass_observation["effective_conversion_operation_net_deltas"]
      operation_deltas[operation] = operation_deltas.get(operation, 0) + delta
  pass_observation["effective_conversion_operation_net_deltas"] = dict(
    sorted(
      (operation, delta)
      for operation, delta in pass_observation[
        "effective_conversion_operation_net_deltas"
      ].items()
      if delta
    )
  )
  for observation in compiler_records:
    conversions.extend(observation["layout_conversions"])
    effective_operations.update(observation["effective_conversion_operations"])
    loop_statuses.append(observation["loop_status"])
    for stage, record in observation["stages"].items():
      merged = stages.setdefault(
        stage,
        {
          "reused_values": [],
          "duplicate_expressions": [],
          "loop_construct_count": 0,
          "loop_constructs": [],
          "operation_counts": {},
          "warp_shuffle_count": 0,
          "shared_memory_transfer_count": 0,
        },
      )
      merged["reused_values"].extend(record.get("reused_values", []))
      merged["duplicate_expressions"].extend(record.get("duplicate_expressions", []))
      merged["loop_construct_count"] += record.get("loop_construct_count", 0)
      merged["loop_constructs"].extend(record.get("loop_constructs", []))
      for operation, count in record.get("operation_counts", {}).items():
        merged["operation_counts"][operation] = (
          merged["operation_counts"].get(operation, 0) + count
        )
      for counter in ("warp_shuffle_count", "shared_memory_transfer_count"):
        merged[counter] += record.get(counter, 0)
  loop_status = (
    "retained"
    if loop_statuses and all(value == "retained" for value in loop_statuses)
    else "unrolled_or_removed"
    if any(value == "unrolled_or_removed" for value in loop_statuses)
    else "not_applicable"
  )
  return {
    "compilation_count": len(compiler_records),
    "layout_conversion_count": sum(
      conversion["layout_changed"] for conversion in conversions
    ),
    "layout_conversions": conversions,
    "effective_conversion_operations": sorted(effective_operations),
    "loop_status": loop_status,
    "loop_status_counts": {
      name: loop_statuses.count(name)
      for name in ("retained", "unrolled_or_removed", "not_applicable")
    },
    "stages": stages,
    "pass_observation": pass_observation,
  }


def _gpu_costs(gpu_report):
  measurements = [case["measurement"] for case in gpu_report["cases"].values()]
  calibration = gpu_report.get("calibration_compiler")
  if isinstance(calibration, dict) and isinstance(calibration.get("measurement"), dict):
    measurements.append(calibration["measurement"])
  compilation_s = sum(
    item["first_launch_wall_s_including_compilation"] for item in measurements
  )
  execution_s = sum(
    item["median_gpu_event_ms_per_launch"] * item["measured_launches"] / 1000.0
    for item in measurements
  )
  return compilation_s, execution_s


def _solve_selection(
  solver,
  output_dir,
  *,
  source,
  seed,
  budget_s,
  iterations,
  target,
  target_index,
  generated_region=False,
):
  trial = _trial_record("solver", seed, budget_s)
  arm_dir = output_dir / "solver"
  arm_dir.mkdir(parents=True)
  started = time.perf_counter()
  generation_started = time.perf_counter()
  query = render_selection_query(
    iterations=iterations,
    target=target,
    target_index=target_index,
    source=source if generated_region else None,
  )
  query_path = arm_dir / "selection-query.sir"
  solved_path = arm_dir / "selection.solved.sir"
  model_path = arm_dir / "selection-model.json"
  query_path.write_text(query, encoding="utf-8")
  (arm_dir / "program.sir").write_text(source, encoding="utf-8")
  trial["cost_s"]["generation_s"] = time.perf_counter() - generation_started
  command = [
    str(solver),
    "--tensor-profile-version",
    "v2",
    "--main",
    "@kernel",
    "--path",
    ",".join(_exact_loop_path(iterations)),
    "--timeout-ms",
    str(int(SOLVER_TIMEOUT_S * 1000)),
    "--output",
    str(solved_path),
    "--emit-model",
    str(model_path),
    str(query_path),
  ]
  remaining = budget_s - (time.perf_counter() - started)
  if remaining <= 0:
    trial["admission_status"] = "timeout"
    trial["selection_record"] = {"reason": "fixed arm budget expired before solving"}
    trial["elapsed_s"] = time.perf_counter() - started
    return trial, None
  solve_started = time.perf_counter()
  try:
    result = subprocess.run(
      command,
      capture_output=True,
      text=True,
      check=False,
      timeout=min(SOLVER_TIMEOUT_S, remaining),
    )
  except subprocess.TimeoutExpired as error:
    trial["admission_status"] = "timeout"
    trial["selection_record"] = {"reason": "SMT selection process timed out"}
    trial["cost_s"]["selection_s"] = time.perf_counter() - solve_started
    trial["elapsed_s"] = time.perf_counter() - started
    (arm_dir / "selection.stdout").write_text(
      _text_output(error.stdout), encoding="utf-8"
    )
    (arm_dir / "selection.stderr").write_text(
      _text_output(error.stderr), encoding="utf-8"
    )
    return trial, None
  except OSError as error:
    trial["admission_status"] = "refused"
    trial["selection_record"] = {"reason": f"cannot start solver: {error}"}
    trial["cost_s"]["selection_s"] = time.perf_counter() - solve_started
    trial["elapsed_s"] = time.perf_counter() - started
    return trial, None

  trial["cost_s"]["selection_s"] = time.perf_counter() - solve_started
  (arm_dir / "selection.stdout").write_text(
    _text_output(result.stdout), encoding="utf-8"
  )
  (arm_dir / "selection.stderr").write_text(
    _text_output(result.stderr), encoding="utf-8"
  )
  if time.perf_counter() - started > budget_s:
    trial["admission_status"] = "timeout"
    trial["selection_record"] = {
      "reason": "SMT selection completed after the arm budget"
    }
    trial["elapsed_s"] = time.perf_counter() - started
    return trial, None
  if result.returncode != 0:
    trial["admission_status"] = "unsat" if "UNSAT" in result.stdout else "refused"
    trial["selection_record"] = {
      "status": "unsat" if "UNSAT" in result.stdout else "solver_error",
      "returncode": result.returncode,
    }
    trial["elapsed_s"] = time.perf_counter() - started
    return trial, None

  materialization_started = time.perf_counter()
  try:
    model = json.loads(model_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    trial["admission_status"] = "refused"
    trial["selection_record"] = {"reason": f"invalid solver model: {error}"}
    trial["elapsed_s"] = time.perf_counter() - started
    return trial, None
  trial["cost_s"]["input_materialization_s"] = (
    time.perf_counter() - materialization_started
  )
  validation_started = time.perf_counter()
  try:
    witness = tensor_fixed_study.validate_solver_model(
      model,
      source,
      domain=DOMAIN,
      target=target,
      target_index=target_index,
    )
  except ValueError as error:
    trial["admission_status"] = "refused"
    trial["selection_record"] = {"reason": str(error)}
    trial["cost_s"]["reference_validation_s"] = time.perf_counter() - validation_started
    trial["elapsed_s"] = time.perf_counter() - started
    return trial, None
  trial["cost_s"]["reference_validation_s"] = time.perf_counter() - validation_started
  trial["admission_status"] = "admitted"
  trial["selection_record"] = {
    "status": "SAT",
    "values": list(witness["values"]),
    "proposal_count": 1,
    "input_sha256": hashlib.sha256(witness["input_bytes"]).hexdigest(),
    "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
    "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    "proof_obligations": len(witness["proof"]["bounds"]),
  }
  trial["elapsed_s"] = time.perf_counter() - started
  return trial, witness


def _run_gpu_policy(
  trial,
  output_dir,
  witness,
  *,
  budget_s,
  started,
  iterations=ITERATIONS,
  target=TARGET,
  target_index=TARGET_INDEX,
  layout_seed=LAYOUT_SEED,
  program_source_path=None,
):
  if trial["admission_status"] != "admitted":
    return
  remaining = budget_s - (time.perf_counter() - started)
  if remaining <= 0:
    trial["execution_status"] = "execution_failure"
    trial["selection_record"]["execution_error"] = "fixed arm budget exhausted"
    trial["elapsed_s"] = time.perf_counter() - started
    return
  policy = trial["selection_policy"]
  arm_dir = output_dir / policy
  gpu_report_path = arm_dir / "gluon.json"
  ir_dir = arm_dir / "ir"
  cache_dir = arm_dir / "triton-cache"
  cache_dir.mkdir(parents=True, exist_ok=True)
  cache_empty_at_start = not any(cache_dir.iterdir())
  trial["cache"] = {
    "path": str(cache_dir),
    "empty_at_start": cache_empty_at_start,
  }
  env = os.environ.copy()
  existing_pythonpath = env.get("PYTHONPATH", "")
  env["PYTHONPATH"] = os.pathsep.join(
    item for item in (str(ROOT), existing_pythonpath) if item
  )
  env["TRITON_CACHE_DIR"] = str(cache_dir)
  command = _gpu_child_command(
    gpu_report_path,
    ir_dir,
    input_values=witness["values"],
    iterations=iterations,
    target=target,
    target_index=target_index,
    layout_seed=layout_seed,
    program_source=program_source_path,
  )
  remaining = budget_s - (time.perf_counter() - started)
  if remaining <= 0:
    trial["execution_status"] = "execution_failure"
    trial["selection_record"]["execution_error"] = (
      "fixed arm budget exhausted before GPU child"
    )
    trial["elapsed_s"] = time.perf_counter() - started
    return
  child_started = time.perf_counter()
  try:
    result = subprocess.run(
      command,
      cwd=ROOT,
      env=env,
      capture_output=True,
      text=True,
      check=False,
      timeout=remaining,
    )
  except subprocess.TimeoutExpired as error:
    (arm_dir / "gluon.stdout").write_text(_text_output(error.stdout), encoding="utf-8")
    (arm_dir / "gluon.stderr").write_text(_text_output(error.stderr), encoding="utf-8")
    trial["execution_status"] = "execution_failure"
    trial["selection_record"]["execution_error"] = (
      "Gluon child exceeded fixed arm budget"
    )
    trial["elapsed_s"] = time.perf_counter() - started
    return
  (arm_dir / "gluon.stdout").write_text(_text_output(result.stdout), encoding="utf-8")
  (arm_dir / "gluon.stderr").write_text(_text_output(result.stderr), encoding="utf-8")
  if time.perf_counter() - started > budget_s:
    trial["execution_status"] = "execution_failure"
    trial["selection_record"]["execution_error"] = (
      "Gluon child completed after the fixed arm budget"
    )
    trial["elapsed_s"] = time.perf_counter() - started
    return
  if result.returncode != 0:
    trial["execution_status"] = "execution_failure"
    trial["selection_record"]["execution_error"] = result.stderr[-4000:]
    trial["elapsed_s"] = time.perf_counter() - started
    return
  try:
    gpu_report = json.loads(gpu_report_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    trial["execution_status"] = "execution_failure"
    trial["selection_record"]["execution_error"] = f"invalid GPU report: {error}"
    trial["elapsed_s"] = time.perf_counter() - started
    return
  expected_hash = hashlib.sha256(witness["input_bytes"]).hexdigest()
  trial["calibration_controls"] = gpu_report.get("calibration_controls", [])
  if gpu_report.get("input_bits_sha256") != expected_hash:
    trial["execution_status"] = "mismatch"
    trial["selection_record"]["execution_error"] = (
      "GPU input bytes differ from selected witness"
    )
  elif any(
    case.get("status") != "pass" for case in gpu_report.get("cases", {}).values()
  ):
    trial["execution_status"] = "mismatch"
  else:
    trial["execution_status"] = "pass"
  trial["selection_record"]["gluon_report_sha256"] = hashlib.sha256(
    gpu_report_path.read_bytes()
  ).hexdigest()
  trial["selection_record"]["gpu_child_elapsed_s"] = time.perf_counter() - child_started
  trial["compiler_observation"] = _combined_compiler_observation(gpu_report)
  compile_cost, gpu_cost = _gpu_costs(gpu_report)
  trial["cost_s"]["compilation_s"] = compile_cost
  trial["cost_s"]["gpu_execution_s"] = gpu_cost
  trial["cost_s"]["input_materialization_s"] += gpu_report.get(
    "input_materialization_s", 0.0
  )
  trial["cost_s"]["reference_validation_s"] += gpu_report.get(
    "reference_validation_s", 0.0
  )
  trial["elapsed_s"] = time.perf_counter() - started


def run(
  output_dir,
  *,
  solver,
  seed=17,
  proposal_budget=1000,
  budget_s=DEFAULT_BUDGET_S,
  iterations=ITERATIONS,
  target=TARGET,
  target_index=TARGET_INDEX,
  layout_seed=LAYOUT_SEED,
  program_source=None,
  program_id=None,
):
  output_dir = Path(output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=False)
  if type(seed) is not int or seed < 0:
    raise ValueError("seed must be a nonnegative integer")
  if type(proposal_budget) is not int or proposal_budget <= 0:
    raise ValueError("proposal_budget must be a positive integer")
  if type(iterations) is not int or not 1 <= iterations <= 8:
    raise ValueError("iterations must be in [1,8]")
  if type(target) is not int:
    raise ValueError("target must be an integer")
  if (
    not isinstance(target_index, (tuple, list))
    or len(target_index) != 2
    or any(type(value) is not int or not 0 <= value < 2 for value in target_index)
  ):
    raise ValueError("target_index must identify an element of the 2x2 output")
  if type(layout_seed) is not int or layout_seed < 0:
    raise ValueError("layout_seed must be a nonnegative integer")
  target_index = tuple(target_index)
  if (
    isinstance(budget_s, bool)
    or not isinstance(budget_s, (int, float))
    or not math.isfinite(budget_s)
    or budget_s <= 0
  ):
    raise ValueError("budget_s must be a finite positive number")
  budget_s = float(budget_s)
  generated_region = program_source is not None
  if generated_region:
    if not isinstance(program_source, str):
      raise ValueError("program_source must be generated SIR text")
    source = program_source
    _, source_iterations, _, _, _ = tensor_exact_f32._generated_region_program(source)
    if source_iterations != iterations:
      raise ValueError("program iteration count differs from the matrix cell")
    if not isinstance(program_id, str) or program_id not in PROGRAM_SOURCES:
      raise ValueError("generated program requires a known program id")
    semantic_contract = "generated-multi-output-region-f32-exact-loop-v1"
  else:
    source = tensor_exact_f32.render_program(iterations=iterations)
    program_id = (
      PROGRAM_ID
      if iterations == ITERATIONS
      else f"authored-2x2-two-state-loop-iterations-{iterations}-v1"
    )
    semantic_contract = SEMANTIC_CONTRACT
  trials = []

  random_started = time.perf_counter()
  random_trial = _trial_record("random", seed, budget_s)
  random_dir = output_dir / "random"
  random_dir.mkdir()
  generation_started = time.perf_counter()
  (random_dir / "program.sir").write_text(source, encoding="utf-8")
  random_trial["cost_s"]["generation_s"] = time.perf_counter() - generation_started
  try:
    selected = tensor_fixed_study.select_random_witness(
      source,
      domain=DOMAIN,
      target=target,
      target_index=target_index,
      seed=seed,
      proposal_budget=proposal_budget,
      deadline=random_started + budget_s,
    )
  except Exception as error:
    selected = {
      "status": "selection_error",
      "reason": f"random input selection failed: {error}",
      "generation_s": 0.0,
      "selection_s": time.perf_counter() - random_started,
      "reference_validation_s": 0.0,
      "proposal_count": 0,
      "values": None,
      "input_bytes": None,
    }
  random_trial["cost_s"]["generation_s"] += selected["generation_s"]
  random_trial["cost_s"]["selection_s"] = selected["selection_s"]
  random_trial["cost_s"]["reference_validation_s"] = selected["reference_validation_s"]
  random_trial["selection_record"] = {
    "status": selected["status"],
    "seed": seed,
    "proposal_budget": proposal_budget,
    "proposal_count": selected["proposal_count"],
    "reason": selected.get("reason"),
    "values": list(selected["values"]) if selected["values"] is not None else None,
    "input_sha256": hashlib.sha256(selected["input_bytes"]).hexdigest()
    if selected["input_bytes"] is not None
    else None,
  }
  random_trial["admission_status"] = (
    "admitted"
    if selected["status"] == "accepted"
    else "timeout"
    if selected["status"] == "timeout"
    else "refused"
  )
  random_trial["elapsed_s"] = time.perf_counter() - random_started
  if selected["status"] == "accepted":
    _run_gpu_policy(
      random_trial,
      output_dir,
      selected,
      budget_s=budget_s,
      started=random_started,
      iterations=iterations,
      target=target,
      target_index=target_index,
      layout_seed=layout_seed,
      program_source_path=random_dir / "program.sir" if generated_region else None,
    )
  trials.append(random_trial)

  solver_started = time.perf_counter()
  solver_trial, solver_witness = _solve_selection(
    Path(solver).resolve(),
    output_dir,
    source=source,
    seed=seed,
    budget_s=budget_s,
    iterations=iterations,
    target=target,
    target_index=target_index,
    generated_region=generated_region,
  )
  if solver_witness is not None:
    _run_gpu_policy(
      solver_trial,
      output_dir,
      solver_witness,
      budget_s=budget_s,
      started=solver_started,
      iterations=iterations,
      target=target,
      target_index=target_index,
      layout_seed=layout_seed,
      program_source_path=output_dir / "solver" / "program.sir"
      if generated_region
      else None,
    )
  trials.append(solver_trial)

  property_id = f"output-element-{target_index[0]}-{target_index[1]}-equals-{target}"
  compiler_settings_id = (
    "triton-3.8.0-v100-num-warps-1-fp-fusion-off-layout-seed-"
    f"{layout_seed}-pass-dump-v1"
  )
  trial_id = f"seed-{seed}"
  for trial in trials:
    trial.update(
      {
        "program_id": program_id,
        "semantic_contract": semantic_contract,
        "property_id": property_id,
        "compiler_settings_id": compiler_settings_id,
        "trial": trial_id,
      }
    )
  ledger = {
    "schema": "refractir.tensor-fixed-contract-study.v1",
    "protocol": {
      "budget_s_per_arm": budget_s,
      "random_seed": seed,
      "random_proposal_budget": proposal_budget,
      "solver_timeout_s": SOLVER_TIMEOUT_S,
      "iterations": iterations,
      "program_id": program_id,
      "semantic_contract": semantic_contract,
      "property": {"target": target, "index": list(target_index)},
      "layout_seed": layout_seed,
      "cache_isolation": "fresh TRITON_CACHE_DIR per policy arm",
      "calibration_control_contract": CALIBRATION_CONTROL_CONTRACT,
    },
    "fixed": {
      "program_id": program_id,
      "program_sha256": hashlib.sha256(source.encode()).hexdigest(),
      "input_domain_id": DOMAIN_ID,
      "input_domain": list(DOMAIN),
      "compiler_settings_id": compiler_settings_id,
      "budget_s": budget_s,
    },
    "trials": trials,
  }
  _write_json(output_dir / "study.json", ledger)
  summary = tensor_evaluation.summarize_fixed_contract_study(ledger)
  _write_json(output_dir / "summary.json", summary)
  return {"ledger": ledger, "summary": summary}


def run_matrix(
  output_dir,
  *,
  solver,
  program_ids=tuple(PROGRAM_SOURCES),
  target_indices=((0, 0), (1, 1)),
  seeds=(17, 29),
  layout_seeds=(LAYOUT_SEED, LAYOUT_SEED + 1),
  proposal_budget=1000,
  budget_s=DEFAULT_BUDGET_S,
):
  """Run matched random/solver pairs over a fixed program/property/layout matrix."""
  cells = _matrix_cell_specs(
    program_ids=program_ids,
    target_indices=target_indices,
    seeds=seeds,
    layout_seeds=layout_seeds,
  )
  program_ids = tuple(program_ids)
  target_indices = tuple(tuple(index) for index in target_indices)
  seeds = tuple(seeds)
  layout_seeds = tuple(layout_seeds)
  output_dir = Path(output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=False)
  matrix = {
    "schema": "refractir.tensor-selection-matrix.v1",
    "status": "in_progress",
    "protocol": {
      "input_domain": list(DOMAIN),
      "anchor_input_values": list(MATRIX_INPUT_VALUES),
      "program_ids": list(program_ids),
      "target_indices": [list(index) for index in target_indices],
      "seeds": list(seeds),
      "layout_seeds": list(layout_seeds),
      "proposal_budget": proposal_budget,
      "budget_s_per_policy": budget_s,
      "pairing": "random and solver share one frozen program, property, seed and layout seed",
    },
    "cells": [],
  }
  matrix_path = output_dir / "matrix.json"
  _write_json(matrix_path, matrix)
  for cell in cells:
    source = _matrix_program_source(cell["program_id"])
    shape, iterations, _, _, _ = tensor_exact_f32._generated_region_program(source)
    anchor_bytes = struct.pack(f"<{len(MATRIX_INPUT_VALUES)}f", *MATRIX_INPUT_VALUES)
    _, anchor_output = tensor_exact_f32.certify(source, anchor_bytes, DOMAIN)
    index = tuple(cell["target_index"])
    target_value = struct.unpack(f"<{len(anchor_output) // 4}f", anchor_output)[
      index[0] * shape[1] + index[1]
    ]
    if not target_value.is_integer():
      raise ValueError("matrix anchor must produce integer-valued properties")
    target = int(target_value)
    cell_report = run(
      output_dir / cell["cell_id"],
      solver=solver,
      seed=cell["seed"],
      proposal_budget=proposal_budget,
      budget_s=budget_s,
      iterations=iterations,
      target=target,
      target_index=index,
      layout_seed=cell["layout_seed"],
      program_source=source,
      program_id=cell["program_id"],
    )
    matrix["cells"].append(
      {
        **cell,
        "target": target,
        "program_id": cell_report["ledger"]["fixed"]["program_id"],
        "program_sha256": cell_report["ledger"]["fixed"]["program_sha256"],
        "property_id": cell_report["ledger"]["trials"][0]["property_id"],
        "study_path": str(Path(cell["cell_id"]) / "study.json"),
        "summary": cell_report["summary"],
      }
    )
    _write_json(matrix_path, matrix)
  matrix["status"] = "complete"
  _write_json(matrix_path, matrix)
  return matrix


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--symirsolve", default="build/bin/symirsolve")
  parser.add_argument("--seed", type=int, default=17)
  parser.add_argument("--proposal-budget", type=int, default=1000)
  parser.add_argument("--budget-s", type=float, default=DEFAULT_BUDGET_S)
  parser.add_argument("--matrix", action="store_true")
  parser.add_argument("--seeds", default="17,29")
  parser.add_argument("--program-ids", default=",".join(PROGRAM_SOURCES))
  parser.add_argument("--target-indices", default="0,0;1,1")
  parser.add_argument("--layout-seeds", default=f"{LAYOUT_SEED},{LAYOUT_SEED + 1}")
  args = parser.parse_args(argv)
  if args.matrix:
    try:
      program_ids = tuple(value.strip() for value in args.program_ids.split(","))
      seeds = tuple(int(value.strip()) for value in args.seeds.split(","))
      layout_seeds = tuple(int(value.strip()) for value in args.layout_seeds.split(","))
      target_indices = tuple(
        tuple(int(value.strip()) for value in pair.split(","))
        for pair in args.target_indices.split(";")
      )
    except ValueError as error:
      parser.error(f"invalid matrix axis: {error}")
    report = run_matrix(
      args.output_dir,
      solver=args.symirsolve,
      program_ids=program_ids,
      target_indices=target_indices,
      seeds=seeds,
      layout_seeds=layout_seeds,
      proposal_budget=args.proposal_budget,
      budget_s=args.budget_s,
    )
  else:
    report = run(
      args.output_dir,
      solver=args.symirsolve,
      seed=args.seed,
      proposal_budget=args.proposal_budget,
      budget_s=args.budget_s,
    )
    report = report["summary"]
  print(json.dumps(report, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
