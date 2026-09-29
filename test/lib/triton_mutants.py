"""Generate and execute isolated test-only Triton fault variants."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from test.lib.run_triton_tests import compare_outputs, run_child

ROOT = Path(__file__).resolve().parents[2]
_MUTANT_CASES = {
  "wrong_softmax_axis": "softmax_2x4_axis1_pos",
  "missing_axis_restoration": "composition_axis-restoration_sum-expand-add",
  "wrong_transpose_permutation": "transpose_2x2_perm0_1",
  "wrong_output_stride": "softmax_2x4_axis1_pos",
  "uniform_distribution": "softmax_2x4_axis1_pos",
}


def mutate_kernel(name, source, shape):
  """Return a kernel source with one named fault and its edit record."""
  shape = tuple(shape)
  if not shape or any(type(size) is not int or size <= 0 for size in shape):
    raise ValueError("shape must contain positive integer dimensions")

  if name == "wrong_softmax_axis":
    match = re.search(r"tl\.softmax\(([^,]+),\s*dim=(\d+)\)", source)
    if not match:
      raise ValueError("cannot apply wrong_softmax_axis mutant: softmax not found")
    axis = int(match.group(2))
    wrong_axis = (axis + 1) % len(shape)
    replacement = f"tl.softmax({match.group(1)}, dim={wrong_axis})"
    mutated = source[: match.start()] + replacement + source[match.end() :]
    return mutated, {"name": name, "before": match.group(0), "after": replacement}

  if name == "missing_axis_restoration":
    pattern = r"^([ \t]*)(\w+) = tl\.expand_dims\(([^,]+),[ \t]*(?:axis=)?\d+\)[ \t]*$"
    match = re.search(pattern, source, flags=re.MULTILINE)
    if not match:
      raise ValueError(
        "cannot apply missing_axis_restoration mutant: expand_dims not found"
      )
    replacement = f"{match.group(1)}{match.group(2)} = {match.group(3)}"
    mutated = source[: match.start()] + replacement + source[match.end() :]
    return mutated, {"name": name, "before": match.group(0), "after": replacement}

  if name == "wrong_transpose_permutation":
    match = re.search(r"tl\.permute\(([^,]+),\s*\(([\d, ]+)\)\)", source)
    if not match:
      raise ValueError(
        "cannot apply wrong_transpose_permutation mutant: permute not found"
      )
    permutation = tuple(int(value) for value in match.group(2).split(","))
    wrong_permutation = tuple(reversed(permutation))
    replacement = (
      f"tl.permute({match.group(1)}, "
      f"({', '.join(str(value) for value in wrong_permutation)}))"
    )
    mutated = source[: match.start()] + replacement + source[match.end() :]
    return mutated, {"name": name, "before": match.group(0), "after": replacement}

  if name == "wrong_output_stride":
    if len(shape) != 2:
      raise ValueError("wrong_output_stride currently requires a rank-two case")
    pattern = r"^(\s*_sir_out_offsets = .*?_sir_out_d0 \* )(\d+)( \+ _sir_out_d1.*)$"
    match = re.search(pattern, source, flags=re.MULTILINE)
    if not match:
      raise ValueError(
        "cannot apply wrong_output_stride mutant: output offset formula not found"
      )
    wrong_stride = max(1, int(match.group(2)) - 1)
    replacement = f"{match.group(1)}{wrong_stride}{match.group(3)}"
    mutated = source[: match.start()] + replacement + source[match.end() :]
    return mutated, {"name": name, "before": match.group(0), "after": replacement}

  if name == "uniform_distribution":
    match = re.search(r"tl\.softmax\(([^,]+),\s*dim=(\d+)\)", source)
    if not match:
      raise ValueError("cannot apply uniform_distribution mutant: softmax not found")
    axis = int(match.group(2))
    if not 0 <= axis < len(shape):
      raise ValueError("softmax axis is outside the input shape")
    value = 1.0 / shape[axis]
    literal = format(value, ".17g")
    replacement = f"tl.full({shape!r}, {literal}, tl.float32)"
    mutated = source[: match.start()] + replacement + source[match.end() :]
    return mutated, {"name": name, "before": match.group(0), "after": replacement}

  raise ValueError(f"unknown Triton mutant: {name}")


def _canonical_json(value):
  return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _case_artifact(source, name, mode):
  run_artifact = source / "runs" / mode / name
  if run_artifact.is_file():
    return run_artifact
  return source / name


def _copy_case(source, target, *, mode="compiled"):
  target.mkdir(parents=True)
  for name in ("case.json", "program.sir", "inputs.bin", "expected.bin"):
    shutil.copy2(source / name, target / name)
  shutil.copy2(_case_artifact(source, "kernel.py", mode), target / "kernel.py")
  environment = json.loads(
    _case_artifact(source, "environment.json", mode).read_text(encoding="utf-8")
  )
  environment["package_lock_file"] = (
    "../../environments/v100-triton-3.8.0/package-lock.json"
  )
  environment_path = target / "environment.json"
  environment_path.write_text(_canonical_json(environment), encoding="utf-8")
  manifest_path = target / "case.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  manifest["environment"] = {
    "file": "environment.json",
    "package_lock": environment["package_lock_file"],
    "package_lock_sha256": environment["package_lock_sha256"],
  }
  manifest_path.write_text(_canonical_json(manifest), encoding="utf-8")
  return manifest, environment


def _sha256(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify_mutant_result(result):
  """Classify a mutant outcome without treating rejection as detection."""
  status = result.get("status")
  compiled = result.get("compiled") is True
  executed = result.get("executed") is True

  if status == "compile_failure" and not compiled and not executed:
    return "compile_rejection"
  if status in {"runtime_timeout", "compile_timeout", "timeout"}:
    return "timeout"
  if status in {"runtime_failure", "runtime_error"} and compiled and not executed:
    return "runtime_failure"
  if status in {"invalid", "invalid_mutant", "invalid_case"}:
    return "invalid"
  if not compiled or not executed:
    return "invalid"
  if status == "exact_mismatch":
    return "exact_kill"
  if status == "numeric_mismatch":
    return "numeric_kill"
  if status == "pass":
    return "survivor"
  return "invalid"


def summarize_mutant_results(results):
  """Summarize valid compiler detections and keep all outcome denominators."""
  classifications = [classify_mutant_result(result) for result in results]
  counts = {
    name: classifications.count(name)
    for name in (
      "exact_kill",
      "numeric_kill",
      "survivor",
      "compile_rejection",
      "runtime_failure",
      "invalid",
      "timeout",
    )
  }
  counts.update(
    {
      "attempted": len(results),
      "compiled": sum(result.get("compiled") is True for result in results),
      "executed": sum(result.get("executed") is True for result in results),
      "killed": counts["exact_kill"] + counts["numeric_kill"],
    }
  )
  return counts


def is_fault_detected(status):
  return status in {"exact_mismatch", "numeric_mismatch"}


def run_mutants(suite, output_dir, *, mode="compiled", timeout_s=60):
  """Copy five baseline cases, apply one fault each, and compare full outputs."""
  if mode not in {"compiled", "interpret"}:
    raise ValueError("mode must be compiled or interpret")
  suite = Path(suite).resolve()
  output_dir = Path(output_dir).resolve()
  if output_dir.exists() and any(output_dir.iterdir()):
    raise FileExistsError(f"mutant output directory is not empty: {output_dir}")
  output_dir.mkdir(parents=True, exist_ok=True)

  results = []
  for name, case_name in _MUTANT_CASES.items():
    source = suite / case_name
    case_dir = output_dir / name
    manifest, environment = _copy_case(source, case_dir, mode=mode)
    kernel_path = case_dir / "kernel.py"
    kernel, edit = mutate_kernel(
      name,
      kernel_path.read_text(encoding="utf-8"),
      manifest["inputs"][0]["shape"],
    )
    kernel_path.write_text(kernel, encoding="utf-8")
    manifest["test_only_mutant"] = edit
    (case_dir / "case.json").write_text(_canonical_json(manifest), encoding="utf-8")

    inputs = (case_dir / "inputs.bin").read_bytes()
    expected = (case_dir / "expected.bin").read_bytes()
    run_metadata = {
      "schema": "refractir.tensor-mutant-run.v1",
      "mode": mode,
      "mutant": edit,
      "source_sha256": _sha256(case_dir / "program.sir"),
      "inputs_sha256": hashlib.sha256(inputs).hexdigest(),
      "expected_sha256": hashlib.sha256(expected).hexdigest(),
      "kernel_sha256": _sha256(kernel_path),
      "triton_source_revision": environment.get("triton_source_revision"),
      "refractir_revision": environment.get("refractir_revision"),
      "launch": manifest["launch"],
      "grid": manifest["grid"],
      "package_lock_sha256": environment["package_lock_sha256"],
    }
    (case_dir / "run.json").write_text(_canonical_json(run_metadata), encoding="utf-8")

    command = [
      sys.executable,
      "-m",
      "test.lib.triton_case_worker",
      "--case",
      str(case_dir),
      "--mode",
      mode,
    ]
    try:
      child = run_child(command, mode, timeout_s)
    except subprocess.TimeoutExpired as error:
      child = None
      status = "runtime_timeout"
      stdout = error.stdout or b""
      stderr = error.stderr or b""
      try:
        phase = json.loads((case_dir / "worker-phase.json").read_text()).get(
          "phase", "compilation"
        )
      except (OSError, json.JSONDecodeError):
        phase = "compilation"
    else:
      stdout = child.stdout
      stderr = child.stderr
      if child.returncode:
        try:
          phase = json.loads((case_dir / "worker-phase.json").read_text()).get(
            "phase", "runtime"
          )
        except (OSError, json.JSONDecodeError):
          phase = "runtime"
        status = "compile_failure" if phase == "compilation" else "runtime_failure"
      elif not (case_dir / "actual.bin").is_file():
        phase = "runtime"
        status = "runtime_failure"
      else:
        phase = "runtime"
        actual = (case_dir / "actual.bin").read_bytes()
        output = manifest["output"]
        status = compare_outputs(
          actual,
          expected,
          output["dtype"],
          output["shape"],
          manifest["comparison"],
        )["status"]

    (case_dir / "stdout.log").write_bytes(stdout)
    (case_dir / "stderr.log").write_bytes(stderr)
    compiled = mode == "compiled" and phase != "compilation"
    executed = (
      child is not None
      and child.returncode == 0
      and (case_dir / "actual.bin").is_file()
    )
    result = {
      "mutant": name,
      "source_case": case_name,
      "mode": mode,
      "status": status,
      "compiled": compiled,
      "executed": executed,
    }
    classification = classify_mutant_result(result)
    detected = classification in {"exact_kill", "numeric_kill"}
    result.update(
      {
        "classification": classification,
        "detected": detected,
        "campaign_valid": bool(environment.get("campaign_available")),
        "input_sha256": hashlib.sha256(inputs).hexdigest(),
        "expected_sha256": hashlib.sha256(expected).hexdigest(),
        "actual_sha256": _sha256(case_dir / "actual.bin")
        if (case_dir / "actual.bin").is_file()
        else None,
        "actual_bytes": (case_dir / "actual.bin").stat().st_size
        if (case_dir / "actual.bin").is_file()
        else None,
        "artifacts": sorted(
          [path.name for path in case_dir.iterdir() if path.is_file()] + ["result.json"]
        ),
      }
    )
    (case_dir / "result.json").write_text(_canonical_json(result), encoding="utf-8")
    results.append(result)

  campaign = {
    "schema": "refractir.tensor-mutant-campaign.v1",
    "mode": mode,
    "mutant_count": len(results),
    **summarize_mutant_results(results),
    "all_detected": bool(results) and all(item["detected"] for item in results),
    "results": results,
  }
  (output_dir / "campaign.json").write_text(_canonical_json(campaign), encoding="utf-8")
  return campaign


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--suite", required=True)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--mode", choices=("compiled", "interpret"), default="compiled")
  parser.add_argument("--timeout", type=float, default=60)
  args = parser.parse_args(argv)
  try:
    report = run_mutants(
      args.suite, args.output_dir, mode=args.mode, timeout_s=args.timeout
    )
  except (OSError, ValueError, json.JSONDecodeError) as error:
    print(f"error: {error}", file=sys.stderr)
    return 2
  print(_canonical_json(report), end="")
  return 0 if report["all_detected"] else 1


if __name__ == "__main__":
  sys.exit(main())
