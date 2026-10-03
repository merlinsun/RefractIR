#!/usr/bin/env python3
"""Compare random and SMT inputs on one exact composed loop and V100 layouts."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from test.lib import (
  build_tensor_exact_loop_case,
  tensor_evaluation,
  tensor_exact_f32,
  tensor_fixed_study,
)

ROOT = Path(__file__).resolve().parents[3]
DOMAIN = (0, 8)
TARGET = 63
TARGET_INDEX = (0, 0)
ROUTE = "exact-f32"
SEMANTIC_CONTRACT = "composed-add-sum-f32-exact-loop-v1"
PROPERTY_ID = "output-element-0-0-equals-63"
PROGRAM_ID = "authored-2x2-two-state-two-iteration-loop-v1"
DOMAIN_ID = "nonnegative-integer-valued-f32-0-through-8"
COMPILER_SETTINGS_ID = (
  "triton-3.8.0-v100-num-warps-1-fp-fusion-off-layout-seed-20261002-v1"
)
DEFAULT_BUDGET_S = 120.0
SOLVER_TIMEOUT_S = 60.0
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


def _combined_compiler_observation(gpu_report):
  conversions = []
  stages = {}
  effective_operations = set()
  loop_statuses = []
  compiler_records = [
    case["compiler"]["observation"] for case in gpu_report["cases"].values()
  ]
  calibration = gpu_report.get("calibration_compiler")
  if isinstance(calibration, dict) and isinstance(calibration.get("observation"), dict):
    compiler_records.append(calibration["observation"])
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


def _solve_selection(solver, output_dir, *, source, seed, budget_s):
  trial = _trial_record("solver", seed, budget_s)
  arm_dir = output_dir / "solver"
  arm_dir.mkdir(parents=True)
  started = time.perf_counter()
  generation_started = time.perf_counter()
  query = build_tensor_exact_loop_case.render_query()
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
    ",".join(build_tensor_exact_loop_case.PATH),
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
      target=TARGET,
      target_index=TARGET_INDEX,
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


def _run_gpu_policy(trial, output_dir, witness, *, budget_s, started):
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
  command = [
    sys.executable,
    "-m",
    "test.tensor.gluon.run_exact_loop",
    "--output",
    str(gpu_report_path),
    "--ir-dir",
    str(ir_dir),
    "--input-values",
    ",".join(str(value) for value in witness["values"]),
  ]
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
):
  output_dir = Path(output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=False)
  if type(seed) is not int or seed < 0:
    raise ValueError("seed must be a nonnegative integer")
  if type(proposal_budget) is not int or proposal_budget <= 0:
    raise ValueError("proposal_budget must be a positive integer")
  if (
    isinstance(budget_s, bool)
    or not isinstance(budget_s, (int, float))
    or not math.isfinite(budget_s)
    or budget_s <= 0
  ):
    raise ValueError("budget_s must be a finite positive number")
  budget_s = float(budget_s)
  source = tensor_exact_f32.render_program()
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
      target=TARGET,
      target_index=TARGET_INDEX,
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
    )
  trials.append(random_trial)

  solver_started = time.perf_counter()
  solver_trial, solver_witness = _solve_selection(
    Path(solver).resolve(), output_dir, source=source, seed=seed, budget_s=budget_s
  )
  if solver_witness is not None:
    _run_gpu_policy(
      solver_trial,
      output_dir,
      solver_witness,
      budget_s=budget_s,
      started=solver_started,
    )
  trials.append(solver_trial)

  ledger = {
    "schema": "refractir.tensor-fixed-contract-study.v1",
    "protocol": {
      "budget_s_per_arm": budget_s,
      "random_seed": seed,
      "random_proposal_budget": proposal_budget,
      "solver_timeout_s": SOLVER_TIMEOUT_S,
      "cache_isolation": "fresh TRITON_CACHE_DIR per policy arm",
      "calibration_control_contract": CALIBRATION_CONTROL_CONTRACT,
    },
    "fixed": {
      "program_id": PROGRAM_ID,
      "program_sha256": hashlib.sha256(source.encode()).hexdigest(),
      "input_domain_id": DOMAIN_ID,
      "input_domain": list(DOMAIN),
      "compiler_settings_id": COMPILER_SETTINGS_ID,
      "budget_s": budget_s,
    },
    "trials": trials,
  }
  _write_json(output_dir / "study.json", ledger)
  summary = tensor_evaluation.summarize_fixed_contract_study(ledger)
  _write_json(output_dir / "summary.json", summary)
  return {"ledger": ledger, "summary": summary}


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--symirsolve", default="build/bin/symirsolve")
  parser.add_argument("--seed", type=int, default=17)
  parser.add_argument("--proposal-budget", type=int, default=1000)
  parser.add_argument("--budget-s", type=float, default=DEFAULT_BUDGET_S)
  args = parser.parse_args(argv)
  report = run(
    args.output_dir,
    solver=args.symirsolve,
    seed=args.seed,
    proposal_budget=args.proposal_budget,
    budget_s=args.budget_s,
  )
  print(json.dumps(report["summary"], sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
