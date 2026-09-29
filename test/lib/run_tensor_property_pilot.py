"""Run the frozen, matched tensor input-selection pilot."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import random
import sys
import tarfile
import tempfile
import threading
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from test.lib.tensor_property_inputs import (
  SYMIR_SOLVE,
  SYMIRI,
  Attempt,
  select_input,
)
from test.lib.tensor_property_workload import PilotProblem, load_workload

METHODS = ("ordinary", "boundary", "construction", "smt")
TERMINAL_STATUSES = {
  "validated",
  "unsat",
  "unknown",
  "timeout",
  "budget_exhausted",
  "unsupported_baseline",
  "unsupported_theory",
  "invalid_problem",
  "validation_failure",
}
SCHEMA = "refractir.tensor-property-pilot-run.v1"
_STAGES = (
  "query_render",
  "solver",
  "input_decode",
  "reference_execution",
  "materialized_query_replay",
  "proposal_generation",
)


def _canonical(value: object) -> bytes:
  return (
    json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
  ).encode("utf-8")


def _sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str | None:
  try:
    return _sha256(path.read_bytes())
  except OSError:
    return None


def _write_atomic(path: Path, data: bytes) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{path.name}.", dir=path.parent
  )
  try:
    with os.fdopen(descriptor, "wb") as output:
      output.write(data)
      output.flush()
      os.fsync(output.fileno())
    os.replace(temporary_name, path)
  finally:
    if os.path.exists(temporary_name):
      os.unlink(temporary_name)


def _source_fingerprint() -> dict[str, str]:
  module_root = Path(__file__).resolve().parents[2]
  files = (
    Path(__file__).resolve(),
    module_root / "test/lib/tensor_property_inputs.py",
    module_root / "test/lib/tensor_property_workload.py",
  )
  return {path.name: _file_sha256(path) or "unavailable" for path in files}


def _environment_record() -> dict[str, object]:
  cpu_model = "unavailable"
  try:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
      if line.lower().startswith(("model name", "hardware")):
        cpu_model = line.split(":", 1)[-1].strip()
        break
  except OSError:
    pass
  return {
    "python": sys.version,
    "python_executable": str(Path(sys.executable).resolve()),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor() or "unavailable",
    "cpu_model": cpu_model,
    "cpu_count": os.cpu_count(),
    "solver_name": os.environ.get("SOLVER", "unavailable"),
    "symiri_path": str(SYMIRI),
    "symiri_sha256": _file_sha256(SYMIRI),
    "symirsolve_path": str(SYMIR_SOLVE),
    "symirsolve_sha256": _file_sha256(SYMIR_SOLVE),
    "thread_environment": {
      key: os.environ.get(key)
      for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    },
  }


def _seed_for(seed: int, problem_id: str, trial: int, method: str) -> int:
  data = f"{seed}\0{problem_id}\0{trial}\0{method}".encode("utf-8")
  return int.from_bytes(hashlib.sha256(data).digest()[:4], "little")


def _build_schedule(
  problems: list[PilotProblem], *, trials: int, seed: int
) -> list[dict[str, object]]:
  rng = random.Random(seed)
  schedule = []
  schedule_index = 0
  for problem in problems:
    for trial in range(1, trials + 1):
      order = list(METHODS)
      rng.shuffle(order)
      for method_order, method in enumerate(order):
        schedule.append(
          {
            "schedule_index": schedule_index,
            "problem_id": problem.problem_id,
            "graph_id": problem.graph_id,
            "trial": trial,
            "method": method,
            "method_order": method_order,
            "seed": _seed_for(seed, problem.problem_id, trial, method),
          }
        )
        schedule_index += 1
  return schedule


def _key(record: dict[str, object]) -> tuple[str, int, str]:
  return str(record["problem_id"]), int(record["trial"]), str(record["method"])


def _load_existing_attempts(
  attempts_path: Path,
  schedule: list[dict[str, object]],
  identity_sha256: str,
) -> dict[tuple[str, int, str], dict[str, object]]:
  by_key = {_key(item): item for item in schedule}
  records: dict[tuple[str, int, str], dict[str, object]] = {}
  if not attempts_path.exists():
    return records
  with attempts_path.open(encoding="utf-8") as source:
    for line_number, line in enumerate(source, start=1):
      if not line.strip():
        raise ValueError(f"attempts.jsonl contains a blank line at {line_number}")
      try:
        record = json.loads(line)
      except json.JSONDecodeError as error:
        raise ValueError(
          f"invalid attempts.jsonl line {line_number}: {error}"
        ) from error
      if not isinstance(record, dict):
        raise ValueError(f"attempts.jsonl line {line_number} must be an object")
      key = _key(record)
      if key not in by_key:
        raise ValueError(f"attempt record is outside the frozen schedule: {key}")
      if key in records:
        raise ValueError(f"duplicate terminal attempt in ledger: {key}")
      if record.get("status") not in TERMINAL_STATUSES:
        raise ValueError(f"attempt {key} has no recognized terminal status")
      if record.get("run_identity_sha256") != identity_sha256:
        raise ValueError(f"attempt {key} belongs to a different run identity")
      if record.get("schedule_index") != by_key[key]["schedule_index"]:
        raise ValueError(f"attempt {key} has a different schedule position")
      records[key] = record
  return records


def _attempt_record(
  item: dict[str, object],
  problem: PilotProblem,
  attempt: Attempt,
  *,
  identity_sha256: str,
  driver_elapsed_s: float,
) -> dict[str, object]:
  if attempt.status not in TERMINAL_STATUSES:
    raise ValueError(f"selector returned a nonterminal status: {attempt.status!r}")
  input_bytes = attempt.input_bytes
  output_bytes = attempt.output_bytes
  timings = dict(attempt.stage_timings_s)
  record = {
    "schema": "refractir.tensor-property-attempt.v1",
    "run_identity_sha256": identity_sha256,
    **item,
    "family": problem.family,
    "symbol_count": problem.symbol_count,
    "target_pattern": problem.target_pattern,
    "target": list(problem.target),
    "status": attempt.status,
    "reason": attempt.reason,
    "supported": attempt.supported,
    "witness_validated": attempt.witness_validated or attempt.status == "validated",
    "input_hex": input_bytes.hex() if input_bytes is not None else None,
    "output_hex": output_bytes.hex() if output_bytes is not None else None,
    "input_sha256": _sha256(input_bytes) if input_bytes is not None else None,
    "output_sha256": _sha256(output_bytes) if output_bytes is not None else None,
    "elapsed_s": float(attempt.elapsed_s),
    "driver_elapsed_s": driver_elapsed_s,
    "stage_timings_s": timings,
    "attempt_count": int(attempt.attempt_count),
    "artifact_paths": dict(attempt.artifact_paths),
  }
  return record


def _append_record(
  attempts_file, lock: threading.Lock, record: dict[str, object]
) -> None:
  encoded = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
  with lock:
    attempts_file.write(encoded)
    attempts_file.flush()


def _summarize(
  schedule: list[dict[str, object]],
  records: dict[tuple[str, int, str], dict[str, object]],
  problems: list[PilotProblem],
  *,
  trials: int,
  budget_seconds: float,
  seed: int,
) -> dict[str, object]:
  method_summary: dict[str, dict[str, object]] = {}
  for method in METHODS:
    rows = [row for row in records.values() if row["method"] == method]
    counts = Counter(str(row["status"]) for row in rows)
    method_summary[method] = {
      "scheduled": sum(item["method"] == method for item in schedule),
      "terminal": len(rows),
      "status_counts": {
        status: counts.get(status, 0) for status in sorted(TERMINAL_STATUSES)
      },
      "validated": counts.get("validated", 0),
      "elapsed_s": sum(float(row.get("elapsed_s", 0.0)) for row in rows),
      "constructor_supported_attempts": sum(
        row.get("supported") is True for row in rows
      )
      if method == "construction"
      else None,
      "constructor_unsupported_attempts": sum(
        row.get("supported") is False for row in rows
      )
      if method == "construction"
      else None,
      "constructor_validated_supported": sum(
        row.get("supported") is True and row.get("status") == "validated"
        for row in rows
      )
      if method == "construction"
      else None,
      "stage_timings_s": {
        stage: sum(
          float(row.get("stage_timings_s", {}).get(stage, 0.0)) for row in rows
        )
        for stage in _STAGES
        if any(stage in row.get("stage_timings_s", {}) for row in rows)
      },
    }

  rows_by_problem: dict[str, list[dict[str, object]]] = defaultdict(list)
  for row in records.values():
    rows_by_problem[str(row["problem_id"])].append(row)
  feasibility = {}
  contradictions = []
  for problem in problems:
    rows = rows_by_problem.get(problem.problem_id, [])
    has_witness = any(
      row["status"] == "validated" or row.get("witness_validated") is True
      for row in rows
    )
    has_unsat = any(row["status"] == "unsat" for row in rows)
    if has_witness and has_unsat:
      contradictions.append(problem.problem_id)
    if has_witness:
      status = "confirmed_feasible"
    elif has_unsat:
      status = "solver_proved_infeasible"
    else:
      status = "unresolved"
    feasibility[problem.problem_id] = status

  complete = len(records) == len(schedule)
  incident_statuses = {"validation_failure", "invalid_problem"}
  incidents = [
    {
      "kind": "contradictory_unsat_and_witness",
      "problem_id": problem_id,
    }
    for problem_id in contradictions
  ]
  incidents.extend(
    {
      "kind": "attempt_validation_incident",
      "problem_id": row["problem_id"],
      "method": row["method"],
      "trial": row["trial"],
      "status": row["status"],
      "reason": row.get("reason"),
    }
    for row in records.values()
    if row["status"] in incident_statuses
  )
  output = {
    "schema": "refractir.tensor-property-selection-summary.v1",
    "status": (
      "contradictory_evidence"
      if contradictions
      else "complete"
      if complete
      else "incomplete"
    ),
    "comparative_conclusion_valid": complete and not incidents,
    "correctness_incident_count": len(incidents),
    "incidents": incidents,
    "workload_problem_count": len(problems),
    "scheduled_attempts": len(schedule),
    "terminal_attempts": len(records),
    "trials": trials,
    "budget_seconds_per_attempt": budget_seconds,
    "seed": seed,
    "methods": method_summary,
    "feasibility_by_problem": feasibility,
    "all_attempts_reconciled": complete,
    "repeated_trials_are_paired_measurements": True,
  }
  return output


def _archive_manifest(output_dir: Path) -> dict[str, object]:
  files = {}
  for path in sorted(output_dir.rglob("*")):
    if not path.is_file() or path.name in {
      "archive-manifest.json",
      "archive-status.json",
    }:
      continue
    relative = path.relative_to(output_dir).as_posix()
    data = path.read_bytes()
    files[relative] = {"bytes": len(data), "sha256": _sha256(data)}
  manifest = {"schema": "refractir.tensor-property-archive.v1", "files": files}
  _write_atomic(output_dir / "archive-manifest.json", _canonical(manifest))
  return manifest


def _write_archive(output_dir: Path) -> dict[str, object]:
  output_dir = output_dir.resolve()
  archive_path = output_dir.with_suffix(".tar.gz")
  manifest = _archive_manifest(output_dir)
  descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{archive_path.name}.", suffix=".tmp", dir=archive_path.parent
  )
  os.close(descriptor)
  temporary_path = Path(temporary_name)
  try:
    with tarfile.open(temporary_path, "w:gz") as archive:
      for relative in sorted(manifest["files"]):
        archive.add(output_dir / relative, arcname=f"{output_dir.name}/{relative}")
      archive.add(
        output_dir / "archive-manifest.json",
        arcname=f"{output_dir.name}/archive-manifest.json",
      )
    _verify_archive(temporary_path, output_dir.name, manifest)
    os.replace(temporary_path, archive_path)
  finally:
    temporary_path.unlink(missing_ok=True)
  return {"status": "created_and_verified", "path": str(archive_path)}


def _verify_archive(
  archive_path: Path, root_name: str, manifest: dict[str, object]
) -> None:
  with tarfile.open(archive_path, "r:gz") as archive:
    members = {
      member.name: member for member in archive.getmembers() if member.isfile()
    }
    manifest_member = members.get(f"{root_name}/archive-manifest.json")
    if manifest_member is None:
      raise ValueError("archive is missing archive-manifest.json")
    manifest_handle = archive.extractfile(manifest_member)
    if manifest_handle is None or manifest_handle.read() != _canonical(manifest):
      raise ValueError("archive manifest does not match archived evidence")
    for relative, expected in manifest["files"].items():
      member = members.get(f"{root_name}/{relative}")
      if member is None:
        raise ValueError(f"archive is missing {relative}")
      handle = archive.extractfile(member)
      if handle is None:
        raise ValueError(f"archive cannot read {relative}")
      data = handle.read()
      if len(data) != expected["bytes"] or _sha256(data) != expected["sha256"]:
        raise ValueError(f"archive digest mismatch for {relative}")


def _jsonl_records(path: Path) -> list[dict[str, object]]:
  if not path.is_file():
    return []
  with path.open(encoding="utf-8") as source:
    return [json.loads(line) for line in source if line.strip()]


def run_campaign(
  workload_path: str | Path,
  output_dir: str | Path,
  *,
  trials: int,
  budget_seconds: float,
  seed: int,
  workers: int = 1,
  archive_writer=None,
) -> dict[str, object]:
  """Execute or resume the frozen four-method schedule without reusing witnesses."""
  workload_path = Path(workload_path).resolve()
  output_dir = Path(output_dir).resolve()
  if type(trials) is not int or trials <= 0:
    raise ValueError("trials must be a positive integer")
  if isinstance(budget_seconds, bool) or not isinstance(budget_seconds, (int, float)):
    raise ValueError("budget_seconds must be finite and positive")
  try:
    finite_budget = math.isfinite(float(budget_seconds))
  except OverflowError:
    finite_budget = False
  if not finite_budget or budget_seconds <= 0:
    raise ValueError("budget_seconds must be finite and positive")
  if type(seed) is not int:
    raise ValueError("seed must be an integer")
  if type(workers) is not int or workers <= 0:
    raise ValueError("workers must be a positive integer")
  if not workload_path.is_file():
    raise ValueError(f"workload file does not exist: {workload_path}")
  workload_bytes = workload_path.read_bytes()
  workload_sha256 = _sha256(workload_bytes)
  _, problems = load_workload(workload_path)
  schedule = _build_schedule(problems, trials=trials, seed=seed)
  schedule_sha256 = _sha256(_canonical(schedule))
  source_fingerprints = _source_fingerprint()
  method_code_sha256 = _sha256(_canonical(source_fingerprints))
  environment = _environment_record()
  environment_sha256 = _sha256(_canonical(environment))
  identity = {
    "schema": SCHEMA,
    "workload_sha256": workload_sha256,
    "method_code_sha256": method_code_sha256,
    "source_fingerprints": source_fingerprints,
    "environment_sha256": environment_sha256,
    "environment": environment,
    "schedule_sha256": schedule_sha256,
    "trials": trials,
    "budget_seconds": float(budget_seconds),
    "seed": seed,
    "workers": workers,
  }
  identity_sha256 = _sha256(_canonical(identity))
  output_dir.mkdir(parents=True, exist_ok=True)
  manifest_path = output_dir / "run-manifest.json"
  if manifest_path.exists():
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing_manifest.get("identity_sha256") != identity_sha256:
      old_identity = existing_manifest.get("identity", {})
      changed = []
      for key, label in (
        ("workload_sha256", "workload hash"),
        ("method_code_sha256", "method code hash"),
        ("environment_sha256", "environment hash"),
        ("schedule_sha256", "schedule hash"),
      ):
        if old_identity.get(key) != identity.get(key):
          changed.append(label)
      if not changed:
        changed.append("run settings")
      raise ValueError("run identity mismatch: " + ", ".join(changed) + " changed")
  else:
    if (output_dir / "attempts.jsonl").exists():
      raise ValueError("attempt ledger exists without its frozen run manifest")
    _write_atomic(
      manifest_path,
      _canonical(
        {
          "schema": SCHEMA,
          "identity": identity,
          "identity_sha256": identity_sha256,
          "created_at": datetime.now(timezone.utc).isoformat(),
        }
      ),
    )

  attempts_path = output_dir / "attempts.jsonl"
  existing_records = _load_existing_attempts(attempts_path, schedule, identity_sha256)
  records_by_key = dict(existing_records)
  problem_by_id = {problem.problem_id: problem for problem in problems}
  groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
  for item in schedule:
    key = _key(item)
    if key not in existing_records:
      groups[(str(item["problem_id"]), int(item["trial"]))].append(item)
  append_lock = threading.Lock()
  attempts_path.parent.mkdir(parents=True, exist_ok=True)
  with attempts_path.open("a", encoding="utf-8") as attempts_file:

    def run_group(group: list[dict[str, object]]) -> None:
      for item in sorted(group, key=lambda entry: int(entry["method_order"])):
        key = _key(item)
        problem = problem_by_id[str(item["problem_id"])]
        artifact_dir = (
          output_dir / "artifacts" / f"attempt-{int(item['schedule_index']):06d}"
        )
        started = time.monotonic()
        try:
          attempt = select_input(
            problem,
            method=str(item["method"]),
            seed=int(item["seed"]),
            budget_s=float(budget_seconds),
            artifact_dir=artifact_dir,
          )
          driver_elapsed = time.monotonic() - started
          if attempt.status == "validated" and driver_elapsed > budget_seconds:
            attempt = Attempt(
              status="budget_exhausted",
              input_bytes=attempt.input_bytes,
              output_bytes=attempt.output_bytes,
              elapsed_s=attempt.elapsed_s,
              stage_timings_s=attempt.stage_timings_s,
              attempt_count=attempt.attempt_count,
              artifact_paths=attempt.artifact_paths,
              reason="validated witness returned after its end-to-end deadline",
              supported=attempt.supported,
              witness_validated=True,
            )
          record = _attempt_record(
            item,
            problem,
            attempt,
            identity_sha256=identity_sha256,
            driver_elapsed_s=driver_elapsed,
          )
        except Exception as error:
          driver_elapsed = time.monotonic() - started
          error_dir = output_dir / "errors"
          error_dir.mkdir(parents=True, exist_ok=True)
          error_path = error_dir / f"attempt-{int(item['schedule_index']):06d}.txt"
          error_path.write_text(traceback.format_exc(), encoding="utf-8")
          record = _attempt_record(
            item,
            problem,
            Attempt(
              status="invalid_problem",
              elapsed_s=driver_elapsed,
              artifact_paths=(("selector_exception", str(error_path)),),
              reason=f"selector exception: {type(error).__name__}: {error}",
            ),
            identity_sha256=identity_sha256,
            driver_elapsed_s=driver_elapsed,
          )
        _append_record(attempts_file, append_lock, record)
        records_by_key[key] = record

    pending_groups = [group for group in groups.values()]
    if pending_groups:
      with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_group, group) for group in pending_groups]
        for future in concurrent.futures.as_completed(futures):
          future.result()

  summary = _summarize(
    schedule,
    records_by_key,
    problems,
    trials=trials,
    budget_seconds=float(budget_seconds),
    seed=seed,
  )
  summary["run_identity_sha256"] = identity_sha256
  summary["workload_sha256"] = workload_sha256
  summary["method_code_sha256"] = method_code_sha256
  summary["environment_sha256"] = environment_sha256
  summary["schedule_sha256"] = schedule_sha256
  summary["worker_count"] = workers
  summary_path = output_dir / "selection-summary.json"
  _write_atomic(summary_path, _canonical(summary))
  if archive_writer is None:
    archive_writer = _write_archive
  try:
    archive_result = archive_writer(output_dir)
    archive_status = {"status": "created", "result": str(archive_result)}
  except Exception as error:
    archive_status = {
      "status": "failed",
      "error": f"{type(error).__name__}: {error}",
    }
  _write_atomic(output_dir / "archive-status.json", _canonical(archive_status))
  summary["archive_status"] = archive_status["status"]
  return summary


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--workload", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--trials", type=int, default=3)
  parser.add_argument("--budget-seconds", type=float, default=5.0)
  parser.add_argument("--seed", type=int, default=281)
  parser.add_argument("--workers", type=int, default=1)
  args = parser.parse_args(argv)
  summary = run_campaign(
    args.workload,
    args.output,
    trials=args.trials,
    budget_seconds=args.budget_seconds,
    seed=args.seed,
    workers=args.workers,
  )
  print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
  return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
  raise SystemExit(main())
