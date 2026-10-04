"""Shared, auditable protocol for paired tensor FP routes."""

import argparse
import json
import math
import os
from pathlib import Path

WARMUP_LAUNCHES = 3
MEASURED_LAUNCHES = 30
MEASUREMENT_BATCHES = 3
NUM_WARPS = 1
ENABLE_FP_FUSION = False
INPUT_DOMAIN = [0, 8]
TARGET_INDEX = [0, 0]
_WORK_SPLIT_FIELDS = (
  "useful_compilations",
  "calibration_compilations",
  "useful_executions",
  "calibration_executions",
  "useful_warmup_launches",
  "calibration_warmup_launches",
  "useful_measured_launches",
  "calibration_measured_launches",
)


def prepare_fresh_cache(path):
  """Create a cache directory only if it contains no stale compiler entries."""
  path = Path(path).expanduser().resolve()
  path.mkdir(parents=True, exist_ok=True)
  if any(path.iterdir()):
    raise ValueError(f"compiler cache must be empty at start: {path}")
  return {
    "path": str(path),
    "empty_at_start": True,
    "policy": "fresh-empty-cache-per-route-arm",
  }


def cache_record_from_environment():
  path = os.environ.get("TRITON_CACHE_DIR")
  if not path:
    return {"path": None, "empty_at_start": None, "policy": "unspecified"}
  return {
    "path": str(Path(path).expanduser().resolve()),
    "empty_at_start": os.environ.get("REFRACTIR_CACHE_EMPTY_AT_START") == "1",
    "policy": "fresh-empty-cache-per-route-arm",
  }


def measurement_accounting(
  *,
  useful_compilations,
  calibration_compilations,
  useful_executions=None,
  calibration_executions=None,
):
  """Separate experiment work from calibration while sharing launch counts."""
  counts = {
    "useful_compilations": useful_compilations,
    "calibration_compilations": calibration_compilations,
    "useful_executions": (
      useful_compilations if useful_executions is None else useful_executions
    ),
    "calibration_executions": (
      calibration_compilations
      if calibration_executions is None
      else calibration_executions
    ),
  }
  if any(type(value) is not int or value < 0 for value in counts.values()):
    raise ValueError("measurement counts must be nonnegative integers")
  total_executions = counts["useful_executions"] + counts["calibration_executions"]
  return {
    **counts,
    "total_compilations": (
      counts["useful_compilations"] + counts["calibration_compilations"]
    ),
    "total_executions": total_executions,
    "warmup_launches_per_execution": WARMUP_LAUNCHES,
    "measured_launches_per_execution": MEASURED_LAUNCHES,
    "measurement_batches": MEASUREMENT_BATCHES,
    "total_warmup_launches": total_executions * WARMUP_LAUNCHES,
    "total_measured_launches": total_executions * MEASURED_LAUNCHES,
  }


def _work_accounting(status, split_counts, *, trial_status_counts=None):
  if status not in {"not_started", "complete", "partial", "unknown"}:
    raise ValueError("invalid GPU work-accounting status")
  if not isinstance(split_counts, dict) or any(
    type(split_counts.get(name)) is not int or split_counts[name] < 0
    for name in _WORK_SPLIT_FIELDS
  ):
    raise ValueError("GPU work counts must be nonnegative integers")
  observed = {name: split_counts[name] for name in _WORK_SPLIT_FIELDS}
  observed.update(
    {
      "total_compilations": (
        observed["useful_compilations"] + observed["calibration_compilations"]
      ),
      "total_executions": (
        observed["useful_executions"] + observed["calibration_executions"]
      ),
      "total_warmup_launches": (
        observed["useful_warmup_launches"] + observed["calibration_warmup_launches"]
      ),
      "total_measured_launches": (
        observed["useful_measured_launches"] + observed["calibration_measured_launches"]
      ),
    }
  )
  exact = status in {"not_started", "complete"}
  result = {
    "status": status,
    "observed": observed,
    "total": dict(observed) if exact else {name: None for name in observed},
    "measurement_protocol": {
      "warmup_launches_per_execution": WARMUP_LAUNCHES,
      "measured_launches_per_execution": MEASURED_LAUNCHES,
      "measurement_batches": MEASUREMENT_BATCHES,
    },
  }
  if trial_status_counts is not None:
    result["trial_status_counts"] = trial_status_counts
  return result


def no_gpu_work_accounting():
  """Represent an arm known to have stopped before starting GPU work."""
  return _work_accounting("not_started", {name: 0 for name in _WORK_SPLIT_FIELDS})


def unknown_gpu_work_accounting():
  """Represent attempted GPU work for which no usable report was retained."""
  return _work_accounting("unknown", {name: 0 for name in _WORK_SPLIT_FIELDS})


def gpu_report_work_accounting(report):
  """Derive lower bounds and completeness from retained per-kernel records."""
  if not isinstance(report, dict):
    return unknown_gpu_work_accounting()
  cases = report.get("cases")
  if not isinstance(cases, dict):
    return unknown_gpu_work_accounting()
  counts = {name: 0 for name in _WORK_SPLIT_FIELDS}
  incomplete = False

  def account_entry(compiler, measurement, prefix):
    nonlocal incomplete
    compiler_observed = isinstance(compiler, dict) and isinstance(
      compiler.get("observation"), dict
    )
    measurement_present = isinstance(measurement, dict)
    measured_counts_valid = measurement_present and all(
      type(measurement.get(field)) is int and measurement[field] >= 0
      for field in ("warmup_launches", "measured_launches")
    )
    first_launch_time = (
      measurement.get("first_launch_wall_s_including_compilation")
      if measurement_present
      else None
    )
    first_launch_observed = (
      type(first_launch_time) in (int, float)
      and math.isfinite(first_launch_time)
      and first_launch_time >= 0
    )
    measurement_observed = measured_counts_valid or first_launch_observed
    if compiler_observed or measurement_observed:
      counts[f"{prefix}_compilations"] += 1
    launch_observed = first_launch_observed or (
      measured_counts_valid
      and sum(measurement[field] for field in ("warmup_launches", "measured_launches"))
      > 0
    )
    if compiler_observed or launch_observed:
      counts[f"{prefix}_executions"] += 1
    if measured_counts_valid:
      counts[f"{prefix}_warmup_launches"] += measurement["warmup_launches"]
      counts[f"{prefix}_measured_launches"] += measurement["measured_launches"]
    if not compiler_observed or not measured_counts_valid:
      incomplete = True
    return compiler_observed and measured_counts_valid and first_launch_observed

  complete_cases = 0
  for case in cases.values():
    if not isinstance(case, dict):
      incomplete = True
      continue
    complete_cases += account_entry(
      case.get("compiler"), case.get("measurement"), "useful"
    )

  calibration = report.get("calibration_compiler")
  calibration_complete = account_entry(
    calibration if isinstance(calibration, dict) else None,
    calibration.get("measurement") if isinstance(calibration, dict) else None,
    "calibration",
  )
  protocol = report.get("measurement_protocol", {})
  declared = (
    protocol.get("measurement_accounting", {}) if isinstance(protocol, dict) else {}
  )
  expected_cases = (
    protocol.get("expected_layout_count") if isinstance(protocol, dict) else None
  )
  if type(expected_cases) is not int:
    expected_cases = (
      declared.get("useful_compilations") if isinstance(declared, dict) else None
    )
  expected_calibrations = (
    declared.get("calibration_compilations", 1) if isinstance(declared, dict) else 1
  )
  all_cases_present = (
    type(expected_cases) is int
    and expected_cases > 0
    and len(cases) == expected_cases
    and complete_cases == len(cases)
  )
  calibration_count_matches = type(expected_calibrations) is int and (
    expected_calibrations == counts["calibration_compilations"]
  )
  complete = all_cases_present and calibration_complete and calibration_count_matches
  has_observed_work = any(counts.values())
  status = "complete" if complete else "partial" if has_observed_work else "unknown"
  if incomplete and complete:
    raise AssertionError("incomplete GPU work cannot be marked complete")
  return _work_accounting(status, counts)


def aggregate_gpu_work_accounting(records):
  """Aggregate exact and censored trial records without filling unknown totals."""
  counts = {name: 0 for name in _WORK_SPLIT_FIELDS}
  status_counts = {
    name: 0 for name in ("not_started", "complete", "partial", "unknown")
  }
  for index, record in enumerate(records):
    if not isinstance(record, dict) or record.get("status") not in status_counts:
      raise ValueError(f"GPU work record {index} has an invalid status")
    observed = record.get("observed")
    if not isinstance(observed, dict):
      raise ValueError(f"GPU work record {index} has no observed counts")
    status = record["status"]
    status_counts[status] += 1
    for name in _WORK_SPLIT_FIELDS:
      value = observed.get(name)
      if type(value) is not int or value < 0:
        raise ValueError(f"GPU work record {index} has an invalid {name}")
      counts[name] += value
  if status_counts["unknown"]:
    status = "unknown"
  elif status_counts["partial"]:
    status = "partial"
  elif status_counts["complete"]:
    status = "complete"
  else:
    status = "not_started"
  return _work_accounting(status, counts, trial_status_counts=status_counts)


def _admitted_exact_cell(cell):
  exact_route = _exact_route_record(cell)
  policies = exact_route.get("policies", {})
  return all(
    isinstance(policies.get(name), dict)
    and policies[name].get("admitted", 0) > 0
    and policies[name].get("gpu_executions", 0) > 0
    and policies[name].get("mismatches", 0) == 0
    and policies[name].get("refused", 0) == 0
    and policies[name].get("timeouts", 0) == 0
    and policies[name].get("unsat", 0) == 0
    for name in ("random", "solver")
  )


def _exact_route_record(cell):
  routes = cell.get("summary", {}).get("routes", {})
  return next(
    (
      record
      for route_id, record in routes.items()
      if route_id.startswith("exact-f32:")
      and record.get("property_id", cell.get("property_id")) == cell.get("property_id")
    ),
    {},
  )


def _fpsan_route_metrics(entry):
  if not entry:
    return None
  mutants = entry.get("calibration_mutants", {})
  ordinary = entry.get("ordinary_reference", {})
  compiler = entry.get("compiler_ir", {})
  return {
    "selection_status": entry.get("selection", {}).get("status"),
    "selection_cost_s": entry.get("selection", {}).get("cost_s"),
    "measurement_accounting": entry.get("measurement_accounting"),
    "fpsan_gpu_measurement": entry.get("measurement"),
    "ordinary_reference": {
      "status": ordinary.get("status"),
      "admission": ordinary.get("admission"),
      "measurement": ordinary.get("measurement"),
    },
    "calibration_mutants_detected": sum(
      record.get("status") == "detected" for record in mutants.values()
    ),
    "calibration_mutant_count": len(mutants),
    "compiler": {
      "layout_conversions": compiler.get("layout_conversion_operations_in_ttgir"),
      "warp_shuffles": compiler.get("warp_shuffle_operations_in_ptx"),
      "shared_memory_transfers": compiler.get("shared_memory_transfers_in_ptx"),
    },
  }


def _exact_route_metrics(cell, exact_protocol):
  route = _exact_route_record(cell)
  policies = route.get("policies", {})
  result = {}
  for name in ("random", "solver"):
    policy = policies.get(name, {})
    compiler = policy.get("compiler", {})
    result[name] = {
      "attempts": policy.get("attempts"),
      "admitted": policy.get("admitted"),
      "admission_rate": policy.get("admission_rate"),
      "gpu_executions": policy.get("gpu_executions"),
      "useful_executions": policy.get("useful_executions"),
      "mismatches": policy.get("mismatches"),
      "refused": policy.get("refused"),
      "timeouts": policy.get("timeouts"),
      "unsat": policy.get("unsat"),
      "cost_s": policy.get("cost_s"),
      "end_to_end_cost_per_useful_execution_s": policy.get(
        "end_to_end_cost_per_useful_execution_s"
      ),
      "stage_accounted_cost_per_useful_execution_s": policy.get(
        "stage_accounted_cost_per_useful_execution_s"
      ),
      "calibration": policy.get("calibration"),
      "compiler": {
        key: compiler.get(key)
        for key in (
          "compilations",
          "retained_loop_compilations",
          "unrolled_or_removed_loop_compilations",
          "effective_layout_conversions",
          "ptx_warp_shuffles",
          "ptx_shared_memory_transfers",
          "duplicate_expression_groups",
          "duplicate_operations",
        )
      },
      "measurement_accounting": (
        policy["work_accounting"]
        if isinstance(policy.get("work_accounting"), dict)
        else unknown_gpu_work_accounting()
      ),
    }
  return {
    "budget_s_per_policy": exact_protocol.get("budget_s_per_policy"),
    "policies": result,
  }


def compare_route_reports(fpsan_report, exact_matrix):
  """Intersect route outcomes only when source, domain, and property match."""
  fpsan_protocol = fpsan_report.get("generated_region_protocol", {})
  exact_protocol = exact_matrix.get("protocol", {})
  fpsan_domain = fpsan_protocol.get("logical_input_domain")
  exact_domain = exact_protocol.get("input_domain")
  fpsan_property = fpsan_protocol.get("property") or {}
  fpsan_target = fpsan_property.get("target")
  fpsan_index = fpsan_property.get("index")
  exact_target = exact_protocol.get("property_target")
  fpsan_programs = fpsan_report.get("generated_regions", {}).get("programs", {})
  exact_cells = exact_matrix.get("cells", [])
  comparison = {
    "schema": "refractir.tensor-route-comparison.v1",
    "status": "complete",
    "routes": {
      "fpsan": {
        "semantic_contract": fpsan_protocol.get("semantic_contract"),
        "input_transport": fpsan_protocol.get("input_transport"),
        "budget_s_per_program": fpsan_protocol.get("selection_budget_s_per_program"),
      },
      "exact": {
        "semantic_contract": exact_protocol.get("semantic_contract"),
        "input_transport": exact_protocol.get("input_transport"),
        "budget_s_per_policy": exact_protocol.get("budget_s_per_policy"),
      },
    },
    "comparison_key": {
      "input_domain": fpsan_domain,
      "target": fpsan_target,
      "target_index": fpsan_index,
    },
    "semantic_equivalence_claim": False,
    "interpretation_limit": (
      "The routes use different semantic contracts and raw input words differ; "
      "the intersection compares admission and execution outcomes only, not "
      "semantic equivalence or identical GPU inputs."
    ),
    "programs": [],
    "common_admitted_program_ids": [],
  }
  for cell in exact_cells:
    program_id = cell.get("program_id")
    fpsan = fpsan_programs.get(program_id, {})
    exclusions = []
    fpsan_source_hash = fpsan.get("region_source", {}).get("sha256")
    exact_source_hash = cell.get("program_sha256")
    if not fpsan:
      exclusions.append("program missing from FpSan report")
    if fpsan_source_hash != exact_source_hash:
      exclusions.append("source hash differs")
    if fpsan_domain != exact_domain:
      exclusions.append("input domain differs")
    if fpsan_target != cell.get("target") or exact_target != cell.get("target"):
      exclusions.append("target differs")
    if fpsan_index != cell.get("target_index"):
      exclusions.append("target index differs")
    if fpsan.get("status") != "pass":
      exclusions.append("FpSan outcome is not pass")
    if not _admitted_exact_cell(cell):
      exclusions.append("exact random/solver pair is not admitted and executed")
    record = {
      "program_id": program_id,
      "source_hash_matches": bool(
        fpsan_source_hash and fpsan_source_hash == exact_source_hash
      ),
      "input_domain_matches": fpsan_domain == exact_domain,
      "target": cell.get("target"),
      "target_index": cell.get("target_index"),
      "fpsan_status": fpsan.get("status"),
      "exact_pair_admitted_and_executed": _admitted_exact_cell(cell),
      "route_metrics": {
        "fpsan": _fpsan_route_metrics(fpsan),
        "exact": _exact_route_metrics(cell, exact_protocol),
      },
      "exclusions": exclusions,
      "status": "common-admitted" if not exclusions else "excluded",
    }
    comparison["programs"].append(record)
    if not exclusions:
      comparison["common_admitted_program_ids"].append(program_id)
  comparison["common_admitted_program_ids"] = sorted(
    set(comparison["common_admitted_program_ids"])
  )
  return comparison


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--fpsan", required=True)
  parser.add_argument("--exact", required=True)
  parser.add_argument("--output", required=True)
  args = parser.parse_args(argv)
  fpsan = json.loads(Path(args.fpsan).read_text(encoding="utf-8"))
  exact = json.loads(Path(args.exact).read_text(encoding="utf-8"))
  comparison = compare_route_reports(fpsan, exact)
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
  print(json.dumps(comparison, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
