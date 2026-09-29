"""Validate and summarize predeclared, matched tensor campaign ledgers.

This module accounts for campaign evidence. It does not infer Triton correctness
from compiler-version differences or from a generated case failing alone.
"""

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path, PurePosixPath

_COST_FIELDS = (
  "generation_s",
  "selection_s",
  "input_materialization_s",
  "reference_validation_s",
  "compilation_s",
  "gpu_execution_s",
)
_WRONG_RESULT_STATUSES = {"exact_mismatch", "numeric_mismatch"}


def _nonnegative_finite(value, where):
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"{where} must be a finite nonnegative number")
  if not math.isfinite(value) or value < 0:
    raise ValueError(f"{where} must be a finite nonnegative number")
  return float(value)


def _positive_integer(value, where):
  if type(value) is not int or value <= 0:
    raise ValueError(f"{where} must be a positive integer")
  return value


def fixed_count_summary(records, *, target_count):
  """Count completed target executions and withhold unequal comparisons."""
  target_count = _positive_integer(target_count, "target_count")
  arms = {}
  for index, record in enumerate(records):
    if not isinstance(record, dict) or not isinstance(record.get("arm"), str):
      raise ValueError(f"record {index} must name an arm")
    completed = record.get("completed_target_runs")
    if type(completed) is not int or completed < 0:
      raise ValueError(f"record {index} completed_target_runs must be nonnegative")
    arms[record["arm"]] = arms.get(record["arm"], 0) + completed
  if not arms:
    raise ValueError("fixed-count protocol requires at least one arm")
  results = {
    arm: {
      "completed_target_runs": count,
      "target_count": target_count,
      "shortfall": max(0, target_count - count),
      "surplus": max(0, count - target_count),
    }
    for arm, count in sorted(arms.items())
  }
  complete = all(
    item["shortfall"] == 0 and item["surplus"] == 0 for item in results.values()
  )
  return {
    "protocol": "fixed-count",
    "status": "complete" if complete else "count_shortfall",
    "equal_count_comparison_valid": complete,
    "arms": results,
  }


def fixed_time_summary(records, *, budget_s):
  """Keep unequal execution counts as throughput outcomes under equal budgets."""
  budget = _nonnegative_finite(budget_s, "budget_s")
  if budget == 0:
    raise ValueError("budget_s must be positive")
  arms = {}
  for index, record in enumerate(records):
    if not isinstance(record, dict) or not isinstance(record.get("arm"), str):
      raise ValueError(f"record {index} must name an arm")
    if (
      _nonnegative_finite(record.get("budget_s"), f"record {index} budget_s") != budget
    ):
      raise ValueError("every arm must receive the declared fixed-time budget")
    elapsed = _nonnegative_finite(record.get("elapsed_s"), f"record {index} elapsed_s")
    if elapsed > budget:
      raise ValueError(f"record {index} exceeded its fixed-time budget")
    charged = end_to_end_seconds(record)
    if charged > elapsed:
      raise ValueError(
        f"record {index} stage costs exceed measured elapsed_s; preparation is not fully charged"
      )
    completed = record.get("completed_target_runs")
    if type(completed) is not int or completed < 0:
      raise ValueError(f"record {index} completed_target_runs must be nonnegative")
    aggregate = arms.setdefault(
      record["arm"], {"elapsed_s": 0.0, "completed_target_runs": 0}
    )
    aggregate["elapsed_s"] += elapsed
    aggregate["completed_target_runs"] += completed
  if not arms:
    raise ValueError("fixed-time protocol requires at least one arm")
  for arm, record in arms.items():
    if record["elapsed_s"] > budget:
      raise ValueError(f"arm {arm} used more than its declared wall-clock budget")
    record["budget_s"] = budget
    record["runs_per_second"] = (
      record["completed_target_runs"] / record["elapsed_s"]
      if record["elapsed_s"] > 0
      else None
    )
  return {
    "protocol": "fixed-time",
    "status": "complete",
    "arms": dict(sorted(arms.items())),
  }


def validate_cache_isolation(records):
  """Require one fresh cache identity per arm/trial and no sharing across them."""
  groups = {}
  errors = []
  for index, record in enumerate(records):
    if not isinstance(record, dict):
      errors.append(f"record {index} is not an object")
      continue
    arm, trial, cache = record.get("arm"), record.get("trial"), record.get("cache_id")
    if not all(isinstance(value, str) and value for value in (arm, trial, cache)):
      errors.append(f"record {index} must name arm, trial, and cache_id")
      continue
    key = (arm, trial)
    if record.get("cache_empty_at_start") is not True:
      errors.append(f"cache for {arm}/{trial} did not start empty")
    groups.setdefault(key, set()).add(cache)
  cache_owners = {}
  for group, cache_ids in groups.items():
    if len(cache_ids) != 1:
      errors.append(f"arm/trial {group[0]}/{group[1]} changed cache identity")
    for cache_id in cache_ids:
      previous = cache_owners.setdefault(cache_id, group)
      if previous != group:
        errors.append(f"cache {cache_id} is shared across arm/trial groups")
  return {"valid": not errors, "errors": errors, "groups": len(groups)}


def end_to_end_seconds(record):
  """Sum all six charged stages; missing preparation costs are an error."""
  if not isinstance(record, dict):
    raise ValueError("stage cost record must be an object")
  if "solving_s" in record:
    raise ValueError("solving_s is not a selection cost; use selection_s")
  return sum(_nonnegative_finite(record.get(key), key) for key in _COST_FIELDS)


def archived_v1_end_to_end_eligible(preparation_cost):
  """Archived v1 cases qualify only when generation and reference costs exist."""
  if not isinstance(preparation_cost, dict):
    return False
  required = ("generation_s", "reference_validation_s")
  try:
    for key in required:
      _nonnegative_finite(preparation_cost.get(key), key)
  except ValueError:
    return False
  return True


def qualifies_rediscovery(attempt):
  """Apply the predeclared fresh-case, paired-build, and triage predicate."""
  if not isinstance(attempt, dict):
    return False
  return (
    attempt.get("fresh_case") is True
    and attempt.get("affected_status") in _WRONG_RESULT_STATUSES
    and attempt.get("corrected_status") == "pass"
    and attempt.get("same_source_sha256") is True
    and attempt.get("same_inputs_sha256") is True
    and attempt.get("same_settings") is True
    and attempt.get("triage_same_root_cause") is True
  )


def matched_graph_pairs(ordinary_cases, directed_cases):
  """Pair only cases sharing graph, domain, and execution configuration."""
  match_fields = (
    "graph_id",
    "input_domain_id",
    "shape",
    "layout_id",
    "num_warps",
    "compiler_revision",
    "proposal_mechanism_id",
    "seed_schedule_id",
    "comparison_policy_id",
    "proposal_budget",
  )
  directed_by_graph = {}
  for case in directed_cases:
    directed_by_graph.setdefault(case.get("graph_id"), []).append(case)
  pairs, unmatched = [], {"ordinary": [], "directed": []}
  used_directed = set()
  for ordinary in ordinary_cases:
    candidates = directed_by_graph.get(ordinary.get("graph_id"), [])
    match = next(
      (
        case
        for case in candidates
        if id(case) not in used_directed
        and all(
          field in case and field in ordinary and case[field] == ordinary[field]
          for field in match_fields
        )
        and ordinary.get("property_passed") is True
        and case.get("property_passed") is True
        and case.get("distinguished") is True
        and ordinary.get("D") == case.get("D")
        and ordinary.get("target_model") == case.get("target_model")
      ),
      None,
    )
    if match is None:
      unmatched["ordinary"].append(ordinary.get("graph_id"))
      continue
    used_directed.add(id(match))
    pairs.append(
      {"graph_id": ordinary.get("graph_id"), "ordinary": ordinary, "directed": match}
    )
  for case in directed_cases:
    if id(case) not in used_directed:
      unmatched["directed"].append(case.get("graph_id"))
  return {"pairs": pairs, "unmatched": unmatched}


def model_specific_selection_summary(pairs):
  """Summarize target-specific paired selection, including true later rescues."""
  if not isinstance(pairs, list):
    raise ValueError("model-specific selection pairs must be an array")
  models = {}
  for index, pair in enumerate(pairs):
    if not isinstance(pair, dict):
      raise ValueError(f"pair {index} must be an object")
    graph_id = pair.get("graph_id")
    arms = pair.get("arms")
    if not isinstance(graph_id, str) or not graph_id or not isinstance(arms, list):
      raise ValueError(f"pair {index} must include a graph_id and arms")
    by_arm = {arm.get("arm"): arm for arm in arms if isinstance(arm, dict)}
    if set(by_arm) != {"ordinary", "directed"}:
      raise ValueError(f"pair {index} must include ordinary and directed arms")
    ordinary, directed = by_arm["ordinary"], by_arm["directed"]
    model_name = ordinary.get("target_model")
    if not isinstance(model_name, str) or not model_name:
      raise ValueError(f"pair {index} must name its target_model")
    for field in ("target_model", "D"):
      if ordinary.get(field) != directed.get(field):
        raise ValueError(f"pair {index} has mismatched {field}")
    for field in (
      "P",
      "input_domain_id",
      "proposal_mechanism_id",
      "seed_schedule_id",
      "proposal_budget",
    ):
      if ordinary.get(field) != directed.get(field):
        raise ValueError(f"pair {index} has mismatched {field}")
    if not all(
      isinstance(ordinary.get(field), str) and ordinary[field]
      for field in (
        "P",
        "input_domain_id",
        "proposal_mechanism_id",
        "seed_schedule_id",
      )
    ):
      raise ValueError(f"pair {index} is missing its P/domain/proposal identities")
    _positive_integer(ordinary.get("proposal_budget"), f"pair {index} proposal_budget")
    if (
      ordinary.get("graph_id", graph_id) != graph_id
      or directed.get("graph_id", graph_id) != graph_id
    ):
      raise ValueError(f"pair {index} has mismatched graph identity")
    report = models.setdefault(
      model_name,
      {
        "graphs": 0,
        "unique_structure_ids": set(),
        "no_applicable_model": 0,
        "applicable_graphs": 0,
        "property_exhausted": 0,
        "ordinary_not_distinguished": 0,
        "directed_witnesses": 0,
        "directed_proposal_exhausted": 0,
        "model_invalid_for_proposals": 0,
        "rescues": 0,
        "rescue_cost": {
          "ordinary_selection_s": 0.0,
          "ordinary_selection_elapsed_s": 0.0,
          "ordinary_property_reference_evaluations": 0,
          "ordinary_property_reference_s": 0.0,
          "ordinary_model_reference_evaluations": 0,
          "ordinary_model_reference_s": 0.0,
          "directed_selection_s": 0.0,
          "directed_selection_elapsed_s": 0.0,
          "directed_property_reference_evaluations": 0,
          "directed_property_reference_s": 0.0,
          "directed_model_reference_evaluations": 0,
          "directed_model_reference_s": 0.0,
        },
      },
    )
    report["graphs"] += 1
    report["unique_structure_ids"].add(
      pair.get("source_signature") or pair.get("graph_id")
    )
    ordinary_status = ordinary.get("status")
    directed_status = directed.get("status")
    if (
      ordinary_status == "no_applicable_model"
      or directed_status == "no_applicable_model"
    ):
      if ordinary_status != directed_status:
        raise ValueError(f"pair {index} disagrees on model applicability")
      report["no_applicable_model"] += 1
      continue
    report["applicable_graphs"] += 1
    if (
      ordinary_status == "property_exhausted" or directed_status == "property_exhausted"
    ):
      report["property_exhausted"] += 1
    ordinary_failed_d = (
      ordinary_status == "accepted"
      and ordinary.get("property_passed") is True
      and ordinary.get("objective_applicable") is True
      and ordinary.get("distinguished") is False
    )
    directed_found_d = (
      directed_status == "accepted"
      and directed.get("property_passed") is True
      and directed.get("objective_applicable") is True
      and directed.get("distinguished") is True
    )
    report["ordinary_not_distinguished"] += int(ordinary_failed_d)
    report["directed_witnesses"] += int(directed_found_d)
    report["directed_proposal_exhausted"] += int(
      directed_status == "proposal_exhausted"
      and directed.get("objective_applicable") is True
    )
    report["model_invalid_for_proposals"] += int(
      directed_status == "model_invalid_for_proposals"
    )
    rescue = (
      ordinary_failed_d
      and directed_found_d
      and type(ordinary.get("proposal_count")) is int
      and type(directed.get("proposal_count")) is int
      and directed["proposal_count"] > ordinary["proposal_count"]
    )
    if rescue:
      report["rescues"] += 1
      for arm_name, arm_record in (("ordinary", ordinary), ("directed", directed)):
        cost = arm_record.get("selection_cost", {})
        for field in (
          "selection_s",
          "selection_elapsed_s",
          "property_reference_evaluations",
          "property_reference_s",
          "model_reference_evaluations",
          "model_reference_s",
        ):
          value = cost.get(field, 0)
          if field.endswith("_evaluations"):
            if type(value) is not int or value < 0:
              raise ValueError(f"pair {index} has invalid {arm_name} {field}")
          else:
            value = _nonnegative_finite(value, f"pair {index} {arm_name} {field}")
          report["rescue_cost"][f"{arm_name}_{field}"] += value
  for report in models.values():
    report["unique_structures"] = len(report.pop("unique_structure_ids"))
    report["rescue_cost_per_rescue"] = {
      field: value / report["rescues"] if report["rescues"] else None
      for field, value in report["rescue_cost"].items()
    }
  return {"models": dict(sorted(models.items()))}


def prepare_matched_pair(
  case_directory,
  output_directory,
  *,
  seed,
  proposal_budget,
  compiler_revision,
  symiri=None,
  target_model=None,
):
  """Clone one graph and select ordinary/directed inputs over the same P/Ω."""
  from test.lib import tensor_coverage, tensor_directed

  source_case = Path(case_directory).resolve()
  output = Path(output_directory).resolve()
  if output.exists() and any(output.iterdir()):
    raise FileExistsError(f"matched-pair output is not empty: {output}")
  if not isinstance(compiler_revision, str) or not compiler_revision:
    raise ValueError("compiler_revision must be pinned")
  if type(seed) is not int or seed < 0:
    raise ValueError("seed must be a nonnegative integer")
  _positive_integer(proposal_budget, "proposal_budget")
  if proposal_budget > tensor_directed.MAX_PROPOSALS:
    raise ValueError(f"proposal_budget must not exceed {tensor_directed.MAX_PROPOSALS}")
  manifest = json.loads((source_case / "case.json").read_text(encoding="utf-8"))
  if manifest.get("profile") != "tensor-v2":
    raise ValueError("matched P/D generation requires tensor-v2")
  if manifest.get("coverage", {}).get("primary_stratum") not in {
    "multi-input-composition",
    "loop-reduction",
  }:
    raise ValueError("matched P/D generation requires a composition or loop case")
  if target_model is not None and (
    not isinstance(target_model, str) or not target_model
  ):
    raise ValueError("target_model must be a nonempty fault-model name")
  file_records = manifest.get("files")
  if not isinstance(file_records, dict):
    raise ValueError("source case must declare its artifacts in files")
  source = (source_case / "program.sir").read_text(encoding="utf-8")
  graph_id = tensor_coverage.source_signature(source)
  input_domain_id = "finite-integer-valued-elements-in-[1,4]"
  proposal_mechanism_id = "python-mt19937-seed-plus-golden-ratio-step-v1"
  seed_schedule_id = f"single-seed-{seed}"
  comparison_policy_id = json.dumps(
    manifest.get("comparison"), sort_keys=True, separators=(",", ":")
  )
  layout_descriptor = {
    "inputs": [
      {
        key: record.get(key)
        for key in (
          "name",
          "shape",
          "element_strides",
          "element_offset",
          "storage_span",
          "axis_order",
        )
      }
      for record in manifest.get("inputs", [])
    ],
    "output": {
      key: manifest.get("output", {}).get(key)
      for key in (
        "shape",
        "element_strides",
        "element_offset",
        "storage_span",
        "axis_order",
      )
    },
  }
  layout_id = hashlib.sha256(
    json.dumps(layout_descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
  ).hexdigest()
  output.mkdir(parents=True, exist_ok=True)
  records = []
  for arm in ("ordinary", "directed"):
    arm_path = output / arm
    arm_path.mkdir()
    shutil.copy2(source_case / "case.json", arm_path / "case.json")
    for name in file_records:
      relative = PurePosixPath(name)
      if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"case artifact path escapes its directory: {name}")
      source_artifact = source_case.joinpath(*relative.parts)
      if not source_artifact.is_file():
        raise ValueError(f"case artifact is missing: {name}")
      destination = arm_path.joinpath(*relative.parts)
      destination.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(source_artifact, destination)
    selection = tensor_directed.prepare_case_arm(
      arm_path,
      arm=arm,
      seed=seed,
      proposal_limit=proposal_budget,
      symiri=symiri,
      target_model=target_model,
    )
    arm_manifest = json.loads((arm_path / "case.json").read_text(encoding="utf-8"))
    records.append(
      {
        "arm": arm,
        "graph_id": graph_id,
        "input_domain_id": input_domain_id,
        "P": "defined-stable-reference",
        "D": f"fault-model:{target_model}"
        if target_model is not None
        else "at-least-one-applicable-model-distinguished",
        "target_model": target_model,
        "proposal_mechanism_id": proposal_mechanism_id,
        "seed_schedule_id": seed_schedule_id,
        "comparison_policy_id": comparison_policy_id,
        "proposal_budget": proposal_budget,
        "proposal_count": selection.get("proposal_count", 0),
        "selection_strategy": arm,
        "status": selection["status"],
        "property_passed": selection.get(
          "property_passed", selection["status"] == "accepted"
        ),
        "distinguished": selection.get("distinguished", False),
        "objective_applicable": selection.get("objective_applicable"),
        "selection_cost": selection.get("cost", {}),
        "shape": arm_manifest.get("output", {}).get("shape"),
        "layout_id": layout_id,
        "num_warps": arm_manifest.get("launch", {}).get("num_warps"),
        "compiler_revision": compiler_revision,
        "case_directory": arm,
        "cache_id": None,
      }
    )
  pair_report = {
    "schema": "refractir.tensor-matched-pair.v1",
    "seed": seed,
    "graph_id": graph_id,
    "input_domain_id": input_domain_id,
    "candidate_proposal_budget": proposal_budget,
    "arms": records,
    "pairing": matched_graph_pairs(
      [record for record in records if record["arm"] == "ordinary"],
      [record for record in records if record["arm"] == "directed"],
    ),
  }
  (output / "pair.json").write_text(
    json.dumps(pair_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  return pair_report


def _save_pair_report(path, report):
  temporary = path.with_name(path.name + ".tmp")
  temporary.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  temporary.replace(path)


def execute_matched_pair(
  pair_directory, *, mode, case_runner=None, resume=True, device_id=0
):
  """Run matched arms serially, saving each outcome so interrupted pairs resume."""
  if mode not in {"compiled", "interpret"}:
    raise ValueError("mode must be compiled or interpret")
  if type(resume) is not bool:
    raise ValueError("resume must be boolean")
  if mode == "compiled" and (type(device_id) is not int or device_id < 0):
    raise ValueError("device_id must be a nonnegative integer")
  selected_device = device_id if mode == "compiled" else None
  root = Path(pair_directory).resolve()
  pair_path = root / "pair.json"
  pair_report = json.loads(pair_path.read_text(encoding="utf-8"))
  arms = pair_report.get("arms")
  if not isinstance(arms, list):
    raise ValueError("matched-pair report must contain an arms array")
  if case_runner is None:
    from test.lib import run_triton_tests

    def default_case_runner(path, selected_mode):
      return run_triton_tests.run_case(path, selected_mode, device_id=selected_device)

    case_runner = default_case_runner

  previous_mode = pair_report.get("runs", {}).get(mode, {})
  if (
    resume
    and "device_id" in previous_mode
    and previous_mode["device_id"] != selected_device
  ):
    raise ValueError("cannot resume a matched pair on a different selected device")
  previous_arms = {
    record.get("arm"): record
    for record in previous_mode.get("arms", [])
    if isinstance(record, dict) and isinstance(record.get("arm"), str)
  }
  records_by_arm = {}
  cache_records = []
  for arm in arms:
    name = arm.get("arm")
    if name not in {"ordinary", "directed"}:
      raise ValueError("matched-pair arm must be ordinary or directed")
    old = previous_arms.get(name) if resume else None
    if old and old.get("run_status") not in {"not_run", "running", "runner_failure"}:
      records_by_arm[name] = old
      cache = old.get("cache")
      if mode == "compiled" and isinstance(cache, dict):
        cache_records.append(
          {
            "arm": name,
            "trial": str(pair_report.get("seed", "default")),
            "cache_id": cache.get("cache_id"),
            "cache_empty_at_start": cache.get("empty_at_start"),
          }
        )

  def current_records():
    return [
      records_by_arm.get(
        arm["arm"],
        {
          "arm": arm["arm"],
          "run_status": "not_run",
          "completed_target_runs": 0,
          "result": None,
          "cache": None,
        },
      )
      for arm in arms
    ]

  def persist(*, final=False):
    mode_arms = current_records()
    completed = sum(record.get("completed_target_runs", 0) for record in mode_arms)
    unresolved = any(
      record.get("run_status") in {"not_run", "running", "runner_failure"}
      for record in mode_arms
    )
    mode_report = {
      "status": "partial" if unresolved else "complete" if final else "in_progress",
      "device_id": selected_device,
      "completed_target_runs": completed,
      "arms": mode_arms,
    }
    if mode == "compiled":
      mode_report["cache_records"] = list(cache_records)
      attempted = sum(
        record.get("run_status")
        not in {"not_run", "selection_rejected", "directed_shortfall"}
        for record in mode_arms
      )
      mode_report["cache_evidence_complete"] = len(cache_records) == attempted
    pair_report.setdefault("runs", {})[mode] = mode_report
    _save_pair_report(pair_path, pair_report)
    return mode_report

  persist()
  for arm in arms:
    name = arm["arm"]
    if name in records_by_arm:
      continue
    eligible = (
      arm.get("status") == "accepted"
      and arm.get("property_passed") is True
      and (name != "directed" or arm.get("distinguished") is True)
    )
    record = {
      "arm": name,
      "run_status": "not_run",
      "completed_target_runs": 0,
      "result": None,
      "cache": None,
    }
    if not eligible:
      record["run_status"] = (
        "directed_shortfall"
        if name == "directed" and arm.get("property_passed") is True
        else "selection_rejected"
      )
      records_by_arm[name] = record
      persist()
      continue

    relative = PurePosixPath(arm.get("case_directory", name))
    if relative.is_absolute() or ".." in relative.parts:
      raise ValueError(f"matched-pair case path escapes its directory: {relative}")
    case_path = root.joinpath(*relative.parts)
    if not case_path.is_dir():
      raise ValueError(f"matched-pair case directory is missing: {relative}")
    record["run_status"] = "running"
    records_by_arm[name] = record
    persist()
    try:
      result = case_runner(case_path, mode)
      if not isinstance(result, dict) or not isinstance(result.get("status"), str):
        raise ValueError(f"runner returned an invalid result for {name}")
    except Exception as error:
      record.update(
        {
          "run_status": "runner_failure",
          "reason": f"{type(error).__name__}: {error}",
        }
      )
      records_by_arm[name] = record
      persist()
      continue

    status = result["status"]
    record.update(
      {
        "run_status": status,
        "completed_target_runs": int(status in ({"pass"} | _WRONG_RESULT_STATUSES)),
        "result": result,
      }
    )
    run_path = case_path / "runs" / mode / "run.json"
    if run_path.is_file():
      try:
        run_metadata = json.loads(run_path.read_text(encoding="utf-8"))
      except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid runner metadata for {name}: {error}") from error
      cache = run_metadata.get("compiler_cache")
      if mode == "compiled" and isinstance(cache, dict):
        record["cache"] = cache
        arm["cache_id"] = cache.get("cache_id")
        cache_records.append(
          {
            "arm": name,
            "trial": str(pair_report.get("seed", "default")),
            "cache_id": cache.get("cache_id"),
            "cache_empty_at_start": cache.get("empty_at_start"),
          }
        )
    elif mode == "compiled":
      record["cache_evidence"] = "missing"
    records_by_arm[name] = record
    persist()

  mode_report = persist(final=True)
  return {
    "mode": mode,
    "status": mode_report["status"],
    "completed_target_runs": mode_report["completed_target_runs"],
    "arms": mode_report["arms"],
    "cache_records": cache_records,
  }


def campaign_summary(data):
  """Produce case and distinct-defect counts without conflating the two."""
  if not isinstance(data, dict):
    raise ValueError("campaign ledger must be an object")
  records = data.get("cases", [])
  if not isinstance(records, list):
    raise ValueError("cases must be an array")
  defect_ids = {
    record["root_cause_id"]
    for record in records
    if isinstance(record, dict)
    and record.get("status") in _WRONG_RESULT_STATUSES
    and isinstance(record.get("root_cause_id"), str)
    and record["root_cause_id"]
    and record.get("triaged") is True
  }
  authored = data.get("authored_holdout_replay", {})
  fresh_exposure = any(
    record.get("fresh_generated_case") is True
    and record.get("holdout_root_cause") is True
    and record.get("affected_status") in _WRONG_RESULT_STATUSES
    and record.get("corrected_status") == "pass"
    and record.get("same_source_sha256") is True
    and record.get("same_inputs_sha256") is True
    and record.get("same_settings") is True
    and record.get("triaged") is True
    for record in data.get("holdout_attempts", [])
    if isinstance(record, dict)
  )
  if fresh_exposure:
    generalization_status = "exposed"
  elif isinstance(data.get("holdout_attempts"), list) and data["holdout_attempts"]:
    generalization_status = "not_exposed_within_budget"
  else:
    generalization_status = "not_run"
  return {
    "case_count": len(records),
    "failing_case_count": sum(
      isinstance(record, dict) and record.get("status") in _WRONG_RESULT_STATUSES
      for record in records
    ),
    "investigated_defect_count": len(defect_ids),
    "investigated_defect_ids": sorted(defect_ids),
    "holdout_status": data.get("holdout_status", "not_selected"),
    "authored_holdout_replay_status": authored.get("status", "not_run")
    if isinstance(authored, dict)
    else "not_run",
    "generator_generalization_status": generalization_status,
  }


def _model_selection_pairs_from_cases(cases):
  grouped = {}
  for record in cases:
    if not isinstance(record, dict):
      continue
    model = record.get("target_model")
    signature = record.get("source_signature")
    if not isinstance(model, str) or not model or not isinstance(signature, str):
      continue
    arm_name = record.get("arm", "")
    arm = arm_name.rsplit("-", 1)[-1]
    if arm not in {"ordinary", "directed"}:
      continue
    key = (record.get("seed"), signature, model)
    grouped.setdefault(key, {})[arm] = {
      "arm": arm,
      "graph_id": signature,
      "target_model": model,
      "D": record.get("D"),
      "P": record.get("P"),
      "input_domain_id": record.get("input_domain_id"),
      "proposal_mechanism_id": record.get("proposal_mechanism_id"),
      "seed_schedule_id": record.get("seed_schedule_id"),
      "proposal_budget": record.get("proposal_budget"),
      "status": record.get("selection_status"),
      "property_passed": record.get("property_passed"),
      "distinguished": record.get("distinguished"),
      "objective_applicable": record.get("objective_applicable"),
      "proposal_count": record.get("proposal_count"),
      "selection_cost": record.get("selection_cost", {}),
    }
  pairs, unmatched = [], 0
  for (seed, signature, _model), arms in grouped.items():
    if set(arms) != {"ordinary", "directed"}:
      unmatched += 1
      continue
    pairs.append(
      {
        "graph_id": signature,
        "source_signature": signature,
        "seed": seed,
        "arms": [arms["ordinary"], arms["directed"]],
      }
    )
  return pairs, unmatched


def summarize_ledger(ledger):
  """Validate one protocol ledger and attach campaign-level outcome counts."""
  protocol = ledger.get("protocol")
  if protocol == "fixed-count":
    result = fixed_count_summary(
      ledger.get("records", []), target_count=ledger.get("target_count")
    )
  elif protocol == "fixed-time":
    result = fixed_time_summary(
      ledger.get("records", []), budget_s=ledger.get("budget_s")
    )
  elif protocol == "rediscovery":
    config = ledger.get("frozen_configuration", {})
    required = (
      "generator_version",
      "seed_schedule",
      "affected_revision",
      "corrected_revision",
      "candidate_attempt_cap",
      "wall_clock_budget_s",
    )
    missing = [key for key in required if key not in config]
    if missing:
      raise ValueError("rediscovery configuration is not frozen: " + ", ".join(missing))
    if config["affected_revision"] == config["corrected_revision"]:
      raise ValueError("affected and corrected revisions must be distinct")
    if (
      not isinstance(config["generator_version"], str)
      or not config["generator_version"]
    ):
      raise ValueError("rediscovery generator_version must be a nonempty pin")
    if (
      not isinstance(config["seed_schedule"], (list, dict))
      or not config["seed_schedule"]
    ):
      raise ValueError("rediscovery seed_schedule must be a nonempty frozen schedule")
    _positive_integer(config["candidate_attempt_cap"], "candidate_attempt_cap")
    _nonnegative_finite(config["wall_clock_budget_s"], "wall_clock_budget_s")
    if config["wall_clock_budget_s"] == 0:
      raise ValueError("wall_clock_budget_s must be positive")
    attempts = ledger.get("attempts", [])
    qualifying = [attempt for attempt in attempts if qualifies_rediscovery(attempt)]
    exhausted = (
      ledger.get("budget_exhausted") is True
      or len(attempts) >= config["candidate_attempt_cap"]
      or (
        ledger.get("elapsed_wall_clock_s") is not None
        and _nonnegative_finite(ledger["elapsed_wall_clock_s"], "elapsed_wall_clock_s")
        >= config["wall_clock_budget_s"]
      )
    )
    result = {
      "protocol": protocol,
      "status": "rediscovered"
      if qualifying
      else "budget_exhausted_without_rediscovery"
      if exhausted
      else "in_progress_without_rediscovery",
      "attempted_cases": len(attempts),
      "qualifying_rediscoveries": len(qualifying),
      "attempts_to_first_rediscovery": next(
        (
          index + 1
          for index, attempt in enumerate(attempts)
          if qualifies_rediscovery(attempt)
        ),
        None,
      ),
      "configuration": config,
    }
  else:
    raise ValueError("protocol must be fixed-count, fixed-time, or rediscovery")
  cache_records = ledger.get("cache_records", [])
  cache = validate_cache_isolation(cache_records)
  if not cache_records:
    cache["valid"] = False
    cache["errors"].append("campaign ledger has no cache records")
  result["cache_isolation"] = cache
  if not cache["valid"]:
    result["protocol_status"] = result["status"]
    result["status"] = "invalid_cache_policy"
  result["campaign"] = campaign_summary(ledger)
  model_pairs, unmatched_model_pairs = _model_selection_pairs_from_cases(
    ledger.get("cases", [])
  )
  if model_pairs or unmatched_model_pairs:
    result["model_selection"] = model_specific_selection_summary(model_pairs)
    result["model_selection"]["unmatched_pairs"] = unmatched_model_pairs
  return result


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--ledger", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args(argv)
  try:
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    report = summarize_ledger(ledger)
  except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
    print(f"tensor evaluation: {error}", file=sys.stderr)
    return 2
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(report, sort_keys=True))
  return 0


if __name__ == "__main__":
  sys.exit(main())
