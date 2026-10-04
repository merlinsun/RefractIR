"""Shared, auditable protocol for paired tensor FP routes."""

import argparse
import json
import os
from pathlib import Path

WARMUP_LAUNCHES = 3
MEASURED_LAUNCHES = 30
MEASUREMENT_BATCHES = 3
NUM_WARPS = 1
ENABLE_FP_FUSION = False
INPUT_DOMAIN = [0, 8]
TARGET_INDEX = [0, 0]


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


def _admitted_exact_cell(cell):
  routes = cell.get("summary", {}).get("routes", {})
  exact_route = next(
    (
      record
      for route_id, record in routes.items()
      if route_id.startswith("exact-f32:")
      and record.get("property_id", cell.get("property_id")) == cell.get("property_id")
    ),
    {},
  )
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
      },
      "exact": {
        "semantic_contract": exact_protocol.get("semantic_contract"),
        "input_transport": exact_protocol.get("input_transport"),
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
