"""Reduce reproducible tensor failures and preserve a standalone replay bundle."""

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

MAX_ATTEMPTS = 1000
MAX_TIMEOUT_S = 3600
_MISMATCHES = {"exact_mismatch", "numeric_mismatch"}


def _positive_number(value, where):
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"{where} must be a positive finite number")
  if not math.isfinite(value) or value <= 0:
    raise ValueError(f"{where} must be a positive finite number")
  return float(value)


def _failure_key(result):
  if not isinstance(result, dict) or result.get("campaign_valid") is not True:
    return None
  status = result.get("status")
  fingerprint = result.get("failure_fingerprint")
  if status not in _MISMATCHES or not isinstance(fingerprint, str) or not fingerprint:
    return None
  actual = result.get("actual_bytes")
  if isinstance(actual, bytes):
    actual = hashlib.sha256(actual).hexdigest()
  elif actual is None:
    actual = result.get("actual_sha256")
  elif not isinstance(actual, str):
    return None
  return status, fingerprint, actual


def _json_safe(value):
  if isinstance(value, bytes):
    return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
  if isinstance(value, dict):
    return {
      str(key): _json_safe(item) for key, item in value.items() if key != "actual_bytes"
    }
  if isinstance(value, (list, tuple)):
    return [_json_safe(item) for item in value]
  if value is None or isinstance(value, (str, int, float, bool)):
    return value
  return repr(value)


def _write_json(path, value):
  Path(path).write_text(
    json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
  )


def _copy_tree(source, destination):
  shutil.copytree(source, destination, symlinks=False)


def _write_actual(case_directory, result, filename):
  value = result.get("actual_bytes") if isinstance(result, dict) else None
  if isinstance(value, bytes):
    (case_directory / filename).write_bytes(value)


def _write_replay_files(bundle, *, source_case, mode):
  replay = {
    "schema": "refractir.tensor-replay-bundle.v1",
    "source_case": str(source_case),
    "mode": mode,
    "command": [
      "python3",
      "-m",
      "test.lib.run_triton_tests",
      "--case",
      "./reduced",
      "--mode",
      mode,
    ],
    "environment_artifacts": [
      "original/runs/compiled/environment.json",
      "original/runs/compiled/package-lock.json",
      "reduced/runs/compiled/environment.json",
      "reduced/runs/compiled/package-lock.json",
    ],
  }
  _write_json(bundle / "replay.json", replay)
  script = (
    "#!/bin/sh\n"
    "set -eu\n"
    'BUNDLE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
    'python3 -m test.lib.run_triton_tests --case "$BUNDLE_DIR/reduced" --mode '
    + mode
    + "\n"
  )
  (bundle / "replay.sh").write_text(script, encoding="utf-8")
  os.chmod(bundle / "replay.sh", 0o755)


def _same_observed_failure(first, second):
  first_key, second_key = _failure_key(first), _failure_key(second)
  if first_key is None or second_key is None:
    return False
  if first_key[:2] != second_key[:2]:
    return False
  return first_key[2] is None or second_key[2] is None or first_key[2] == second_key[2]


def reduce_case(
  case_directory,
  *,
  output_dir,
  run_target,
  validate_candidate,
  transformations,
  mode="compiled",
  max_attempts=MAX_ATTEMPTS,
  timeout_s=MAX_TIMEOUT_S,
  monotonic=time.monotonic,
):
  """Try bounded edits, accepting only candidates with the same stable mismatch.

  ``validate_candidate`` owns the SIR/profile, reference definedness, property,
  and stability checks. ``run_target`` must return the comparison status and a
  stable failure fingerprint for a confirmed wrong result.
  """
  source_case = Path(case_directory).resolve()
  output = Path(output_dir).resolve()
  if not source_case.is_dir():
    raise ValueError("case_directory must exist")
  for required in ("case.json", "program.sir", "inputs.bin", "expected.bin"):
    if not (source_case / required).is_file():
      raise ValueError(f"case is missing required artifact {required}")
  if output.exists() and any(output.iterdir()):
    raise FileExistsError(f"replay bundle output is not empty: {output}")
  if type(max_attempts) is not int or not 0 <= max_attempts <= MAX_ATTEMPTS:
    raise ValueError(f"max_attempts must be in [0, {MAX_ATTEMPTS}]")
  timeout = _positive_number(timeout_s, "timeout_s")
  if timeout > MAX_TIMEOUT_S:
    raise ValueError(f"timeout_s must be at most {MAX_TIMEOUT_S}")
  if mode not in {"interpret", "compiled"}:
    raise ValueError("mode must be interpret or compiled")

  started = monotonic()
  first = run_target(source_case)
  second = run_target(source_case)
  if not _same_observed_failure(first, second):
    return {
      "status": "not_reproducible",
      "attempted_transformations": 0,
      "initial_runs": [_json_safe(first), _json_safe(second)],
    }

  output.mkdir(parents=True, exist_ok=True)
  original_bundle = output / "original"
  _copy_tree(source_case, original_bundle)
  _write_actual(original_bundle, first, "actual-replay-1.bin")
  _write_actual(original_bundle, second, "actual-replay-2.bin")
  current = output / "reduced"
  _copy_tree(source_case, current)
  _write_actual(current, second, "actual.bin")
  _write_replay_files(output, source_case=source_case, mode=mode)
  _write_json(output / "initial-runs.json", {"first": first, "second": second})

  trace = []
  accepted = 0
  attempt_root = output / "attempts"
  attempt_root.mkdir()
  with tempfile.TemporaryDirectory(prefix="tensor-reducer-") as temporary:
    current_work = Path(temporary) / "current"
    _copy_tree(current, current_work)
    while len(trace) < max_attempts and monotonic() - started < timeout:
      candidates = (
        transformations(current_work) if callable(transformations) else transformations
      )
      candidate = next(iter(candidates), None)
      if candidate is None:
        break
      index = len(trace) + 1
      name = candidate.get("name") if isinstance(candidate, dict) else None
      apply = candidate.get("apply") if isinstance(candidate, dict) else None
      if not isinstance(name, str) or not callable(apply):
        raise ValueError("each transformation must have a name and callable apply")
      candidate_dir = Path(temporary) / f"candidate-{index:04d}"
      _copy_tree(current_work, candidate_dir)
      item = {"index": index, "name": name, "status": "rejected"}
      try:
        apply(candidate_dir)
        validation = validate_candidate(candidate_dir)
      except Exception as error:  # Keep failed transformations as reviewable evidence.
        validation = {"valid": False, "reason": f"{type(error).__name__}: {error}"}
      item["validation"] = (
        "accepted"
        if isinstance(validation, dict) and validation.get("valid") is True
        else "rejected"
      )
      if item["validation"] == "rejected":
        item["reason"] = (
          validation.get("reason", "candidate validation failed")
          if isinstance(validation, dict)
          else "candidate validator returned an invalid record"
        )
        target_result = None
      elif monotonic() - started >= timeout:
        item["reason"] = "time_budget_exhausted_before_target_run"
        item["validation"] = "not_run"
        target_result = None
      else:
        target_result = run_target(candidate_dir)
        item["failure"] = _json_safe(target_result)
        if _same_observed_failure(second, target_result):
          item["status"] = "preserving_simplification"
          accepted += 1
          shutil.rmtree(current_work)
          _copy_tree(candidate_dir, current_work)
          shutil.rmtree(current)
          _copy_tree(candidate_dir, current)
          _write_actual(current, target_result, "actual.bin")
      trace.append(item)
      attempt_dir = attempt_root / f"{index:04d}"
      attempt_dir.mkdir()
      _copy_tree(candidate_dir, attempt_dir / "case")
      _write_json(attempt_dir / "attempt.json", item)
      if monotonic() - started >= timeout:
        break
      # A callable proposer may yield successive edits from the new current state.
      if not callable(transformations):
        try:
          transformations = iter(transformations)
          next(transformations)
        except StopIteration:
          break

  result = {
    "status": "preserving_simplification"
    if accepted
    else "no_preserving_simplification_found",
    "accepted_transformations": accepted,
    "attempted_transformations": len(trace),
    "limit": {"max_attempts": max_attempts, "timeout_s": timeout},
    "elapsed_s": max(0.0, monotonic() - started),
    "trace": trace,
    "initial_failure_fingerprint": _failure_key(second)[1],
  }
  _write_json(output / "reduction.json", result)
  return result
