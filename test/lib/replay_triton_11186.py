"""Replay the PR #11186 fixture on isolated complete Triton builds."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYMIRI = ROOT / "symiri"
DEFAULT_SYMIRC = ROOT / "symirc"
_REVISION = re.compile(r"^[A-Za-z0-9._-]+$")
_WRONG_RESULT = {"exact_mismatch", "numeric_mismatch"}


def revision_artifact_dir(output_root, revision):
  """Return a deterministic, traversal-safe artifact root for one build."""
  if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
    raise ValueError(
      "compiler revision label must use letters, digits, dot, underscore, dash"
    )
  return Path(output_root) / revision


def _relative_to(base, path):
  try:
    return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()
  except ValueError:
    return str(Path(path).resolve())


def calibration_status(affected, corrected, *, triage_confirmed, source_overlay=False):
  """Apply the full-build affected-mismatch/corrected-pass calibration gate."""
  if source_overlay:
    return "source_overlay"
  if not isinstance(affected, dict) or not isinstance(corrected, dict):
    return "not_calibrated"
  if affected.get("status") not in _WRONG_RESULT or corrected.get("status") != "pass":
    return "not_calibrated"
  if not affected.get("campaign_valid") or not corrected.get("campaign_valid"):
    return "not_calibrated"
  if not affected.get("revision") or not corrected.get("revision"):
    return "not_calibrated"
  if affected["revision"] == corrected["revision"] or not triage_confirmed:
    return "not_calibrated"
  return "calibrated"


def compare_pinned_runs(development, release):
  """Report each pinned build against the reference before comparing versions."""
  development_status = (
    development.get("status", "missing") if isinstance(development, dict) else "missing"
  )
  release_status = (
    release.get("status", "missing") if isinstance(release, dict) else "missing"
  )
  return {
    "development_status": development_status,
    "release_status": release_status,
    "version_difference": development_status != release_status,
  }


def _run_build(case_dir, output_dir, build, *, symiri, symirc, timeout_s):
  revision = build["revision"]
  build_root = revision_artifact_dir(output_dir, revision)
  replay_case = build_root / "case"
  if build_root.exists():
    shutil.rmtree(build_root)
  replay_case.parent.mkdir(parents=True, exist_ok=True)
  shutil.copytree(case_dir, replay_case)

  env = os.environ.copy()
  python_path = build.get("pythonpath")
  env["PYTHONPATH"] = os.pathsep.join(
    str(path) for path in (ROOT, python_path, env.get("PYTHONPATH")) if path
  )
  env["REFRACTIR_SYMIRI"] = str(symiri)
  env["REFRACTIR_SYMIRC"] = str(symirc)
  env["REFRACTIR_REVISION"] = os.environ.get("REFRACTIR_REVISION", "working-tree")
  env["REFRACTIR_COMPILER_FLAGS"] = os.environ.get(
    "REFRACTIR_COMPILER_FLAGS", "recorded by replay invocation"
  )
  command = [
    sys.executable,
    "-m",
    "test.lib.run_triton_tests",
    "--case",
    str(replay_case),
    "--mode",
    "compiled",
    "--timeout",
    str(timeout_s),
  ]
  try:
    process = subprocess.run(
      command,
      cwd=ROOT,
      env=env,
      capture_output=True,
      text=True,
      timeout=timeout_s + 30,
      check=False,
    )
  except (OSError, subprocess.TimeoutExpired) as error:
    return {
      "status": "runtime_timeout"
      if isinstance(error, subprocess.TimeoutExpired)
      else "environment_unavailable",
      "revision": revision,
      "campaign_valid": False,
      "reason": str(error),
      "artifact_dir": _relative_to(output_dir, build_root),
    }

  run_report = {}
  result_path = replay_case / "result-compiled.json"
  try:
    run_report = json.loads(result_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    try:
      run_report = json.loads(process.stdout)
    except json.JSONDecodeError:
      run_report = {}
  status = run_report.get("status", "runner_failure")
  campaign_valid = run_report.get("campaign_valid") is True
  environment = {}
  environment_path = replay_case / "runs" / "compiled" / "environment.json"
  try:
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    pass
  actual_revision = environment.get("triton_source_revision")
  expected_revision = build.get("expected_triton_revision")
  if expected_revision and actual_revision != expected_revision:
    status = "environment_mismatch"
    campaign_valid = False
  comparison = run_report.get("comparison", {})
  comparison_summary = None
  if isinstance(comparison, dict):
    comparison_summary = {
      "status": comparison.get("status", status),
      "mismatch_count": len(
        comparison.get("mismatching_indices", comparison.get("mismatches", []))
      ),
      "max_absolute_error": comparison.get("max_absolute_error"),
      "max_relative_error": comparison.get("max_relative_error"),
    }
  return {
    "status": status,
    "revision": actual_revision or revision,
    "revision_label": revision,
    "expected_triton_revision": expected_revision,
    "campaign_valid": campaign_valid,
    "comparison": comparison_summary,
    "exit_code": process.returncode,
    "artifact_dir": _relative_to(output_dir, build_root),
    "artifacts": run_report.get("artifacts", []),
    "artifact_digests": run_report.get("artifact_digests", {}),
  }


def replay_pair(
  case_dir,
  output_dir,
  affected_build,
  corrected_build,
  *,
  symiri=DEFAULT_SYMIRI,
  symirc=DEFAULT_SYMIRC,
  timeout_s=60,
  triage_confirmed=False,
):
  """Replay byte-identical case copies on separate pinned compiler builds."""
  case_dir = Path(case_dir).resolve()
  output_dir = Path(output_dir).resolve()
  if affected_build.get("revision") == corrected_build.get("revision"):
    raise ValueError("affected and corrected builds need distinct revision labels")
  output_dir.mkdir(parents=True, exist_ok=True)
  affected = _run_build(
    case_dir,
    output_dir,
    affected_build,
    symiri=symiri,
    symirc=symirc,
    timeout_s=timeout_s,
  )
  corrected = _run_build(
    case_dir,
    output_dir,
    corrected_build,
    symiri=symiri,
    symirc=symirc,
    timeout_s=timeout_s,
  )
  status = calibration_status(affected, corrected, triage_confirmed=triage_confirmed)
  case_hashes = {}
  for name in ("program.sir", "inputs.bin", "expected.bin", "case.json"):
    path = case_dir / name
    if path.is_file():
      case_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
  report = {
    "schema": "refractir.tensor-historical-replay.v1",
    "fixture": _relative_to(ROOT, case_dir),
    "case_sha256": case_hashes,
    "source_overlay": False,
    "affected": affected,
    "corrected": corrected,
    "triage_confirmed": triage_confirmed,
    "calibration_status": status,
  }
  (output_dir / "replay.json").write_text(
    json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
  )
  return report


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--case", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--affected-pythonpath", required=True)
  parser.add_argument("--affected-revision", required=True)
  parser.add_argument("--affected-expected-triton-revision", required=True)
  parser.add_argument("--corrected-pythonpath", required=True)
  parser.add_argument("--corrected-revision", required=True)
  parser.add_argument("--corrected-expected-triton-revision", required=True)
  parser.add_argument("--symiri", default=str(DEFAULT_SYMIRI))
  parser.add_argument("--symirc", default=str(DEFAULT_SYMIRC))
  parser.add_argument("--timeout", type=float, default=60)
  parser.add_argument("--triage-confirmed", action="store_true")
  args = parser.parse_args(argv)
  report = replay_pair(
    args.case,
    args.output,
    {
      "revision": args.affected_revision,
      "pythonpath": args.affected_pythonpath,
      "expected_triton_revision": args.affected_expected_triton_revision,
    },
    {
      "revision": args.corrected_revision,
      "pythonpath": args.corrected_pythonpath,
      "expected_triton_revision": args.corrected_expected_triton_revision,
    },
    symiri=args.symiri,
    symirc=args.symirc,
    timeout_s=args.timeout,
    triage_confirmed=args.triage_confirmed,
  )
  print(json.dumps(report, ensure_ascii=False, sort_keys=True))
  return 0 if report["calibration_status"] == "calibrated" else 1


if __name__ == "__main__":
  sys.exit(main())
