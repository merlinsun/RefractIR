"""Run one locked tensor case in an isolated Triton process."""

import argparse
import ast
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from test.lib import probe_triton_env, tensor_case
from test.lib.tensor_compare import compare

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYMIRC = Path(os.environ.get("REFRACTIR_SYMIRC", ROOT / "symirc")).resolve()
DEFAULT_TIMEOUT_S = 60
V2_WARP_COUNTS = (1, 2, 4, 8)
_I32_SENTINEL = 0x5A5A5A5A
_F32_SENTINEL = float("nan")


def _canonical_json(value):
  return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _write_atomic(path, data):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
  try:
    with os.fdopen(descriptor, "wb") as output:
      output.write(data)
    os.replace(temporary, path)
  finally:
    if os.path.exists(temporary):
      os.unlink(temporary)


def _coordinates(index, shape):
  if not shape:
    return []
  coordinate = [0] * len(shape)
  for axis in range(len(shape) - 1, -1, -1):
    coordinate[axis] = index % shape[axis]
    index //= shape[axis]
  return coordinate


def compare_outputs(actual, expected, dtype, shape, comparison):
  """Compare complete output buffers and retain coordinates and raw words."""
  if not isinstance(shape, list) or any(
    type(size) is not int or size <= 0 for size in shape
  ):
    return {
      "status": "invalid_case",
      "mismatches": [],
      "reason": "invalid output shape",
    }
  result = compare(
    actual,
    expected,
    dtype,
    comparison["mode"] == "exact",
    comparison["atol"],
    comparison["rtol"],
    zero_sign_exact=comparison.get("zero_sign_exact", False),
  )
  if result["status"] in {"invalid_case", "pass"}:
    result["mismatches"] = []
    return result

  count = len(expected) // 4
  if dtype == "f32":
    expected_words = struct.unpack(f"<{count}I", expected)
    actual_words = struct.unpack(f"<{count}I", actual)
  else:
    expected_words = struct.unpack(f"<{count}i", expected)
    actual_words = struct.unpack(f"<{count}i", actual)
  mismatches = []
  for index in result.get("mismatching_indices", []):
    mismatches.append(
      {
        "index": index,
        "coordinate": _coordinates(index, shape),
        "expected_bits": f"0x{expected_words[index] & 0xFFFFFFFF:08x}",
        "actual_bits": f"0x{actual_words[index] & 0xFFFFFFFF:08x}",
      }
    )
  result["mismatches"] = mismatches
  return result


def run_child(
  command,
  mode,
  timeout_s=DEFAULT_TIMEOUT_S,
  *,
  cwd=ROOT,
  cache_dir=None,
  device_id=None,
):
  """Run one mode in a fresh process with an explicit interpreter setting."""
  if mode not in {"compiled", "interpret"}:
    raise ValueError("mode must be compiled or interpret")
  if device_id is not None and (type(device_id) is not int or device_id < 0):
    raise ValueError("device_id must be a nonnegative integer")
  if mode == "interpret" and device_id is not None:
    raise ValueError("device_id is available only in compiled mode")
  env = dict(os.environ)
  env.pop("TRITON_INTERPRET", None)
  env.pop("TRITON_CACHE_DIR", None)
  if mode == "interpret":
    env["TRITON_INTERPRET"] = "1"
  elif cache_dir is not None:
    cache_path = Path(cache_dir).resolve()
    if cache_path.exists():
      shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True)
    env["TRITON_CACHE_DIR"] = str(cache_path)
  if mode == "compiled" and device_id is not None:
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
  existing_pythonpath = env.get("PYTHONPATH")
  env["PYTHONPATH"] = str(ROOT) + (
    os.pathsep + existing_pythonpath if existing_pythonpath else ""
  )
  return subprocess.run(
    command,
    cwd=cwd,
    env=env,
    capture_output=True,
    timeout=timeout_s,
    check=False,
  )


def _gpu_available(device_id=0):
  try:
    import torch

    return bool(torch.cuda.is_available() and device_id < torch.cuda.device_count())
  except Exception:
    return False


def _emit_kernel(case, symirc, artifact_dir):
  artifact_dir.mkdir(parents=True, exist_ok=True)
  target = artifact_dir / "kernel.py"
  descriptor, temporary_name = tempfile.mkstemp(
    prefix=".kernel.", suffix=".py", dir=artifact_dir
  )
  os.close(descriptor)
  temporary = Path(temporary_name)
  temporary.unlink()
  command = [
    str(symirc),
    "--target",
    "triton",
    "--entry",
    case.manifest["entry"],
    "--tensor-softmax-lowering",
    case.manifest["softmax_lowering"],
    "-o",
    str(temporary),
    str(case.path / "program.sir"),
  ]
  if case.manifest["profile"] in {"tensor-v2", "tensor-v3"}:
    buffers = []
    for role, records in (
      ("input", case.manifest["inputs"]),
      ("output", (case.manifest["output"],)),
    ):
      for record in records:
        if not record["shape"]:
          continue
        buffers.append(
          {
            "name": record["name"],
            "role": role,
            "element_strides": record["element_strides"],
            "element_offset": record["element_offset"],
            "storage_span": record["storage_span"],
            "axis_order": record["axis_order"],
          }
        )
    layout_manifest = {"buffers": buffers}
    layout_path = artifact_dir / "tensor-layout.json"
    layout_path.write_text(
      _canonical_json(layout_manifest) + "\n",
      encoding="utf-8",
    )
    version = "v2" if case.manifest["profile"] == "tensor-v2" else "v3"
    command[1:1] = ["--tensor-profile-version", version]
    command[1:1] = ["--tensor-layout-manifest", str(layout_path)]
  elif case.manifest["profile"] == "tensor-v4":
    command[1:1] = ["--tensor-profile-version", "v4"]
  try:
    result = subprocess.run(
      command,
      cwd=ROOT,
      capture_output=True,
      text=True,
      timeout=10,
      check=False,
    )
  except (OSError, subprocess.TimeoutExpired) as error:
    return None, str(error), ""
  if result.returncode != 0:
    temporary.unlink(missing_ok=True)
    return None, result.stderr, result.stdout
  try:
    text = temporary.read_text(encoding="utf-8")
    ast.parse(text, filename=str(target))
    os.replace(temporary, target)
  except (OSError, SyntaxError, UnicodeError) as error:
    temporary.unlink(missing_ok=True)
    return None, str(error), result.stdout
  return target, result.stderr, result.stdout


def _save_environment(artifact_dir, environment_record):
  probe_triton_env.write_environment(artifact_dir, environment_record)


def _report(case_path, mode, status, **fields):
  return {"case": str(case_path), "mode": mode, "status": status, **fields}


def _save_report(case_path, report):
  digests = {}
  for name in report.get("artifacts", []):
    path = Path(case_path) / name
    if path.is_file():
      data = path.read_bytes()
      digests[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
  report.setdefault("artifact_digests", digests)
  encoded = _canonical_json(report).encode("utf-8")
  mode = report.get("mode")
  mode_name = mode if mode in {"compiled", "interpret"} else "invalid"
  _write_atomic(Path(case_path) / f"result-{mode_name}.json", encoded)
  _write_atomic(Path(case_path) / "result.json", encoded)


def _classify_warp_probe(case_dir, report):
  """Keep legal target failures as candidates; classify only explicit limits."""
  status = report.get("status")
  if status == "pass" and report.get("campaign_valid") is not True:
    return "environment_unavailable"
  if status in {
    "pass",
    "unsupported_configuration",
    "resource_limit",
    "environment_unavailable",
    "invalid_case",
  }:
    return status

  diagnostic = [str(report.get("reason", ""))]
  for artifact in report.get("artifacts", []):
    try:
      path = (Path(case_dir) / artifact).resolve()
      path.relative_to(Path(case_dir).resolve())
      diagnostic.append(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError, TypeError):
      continue
  text = "\n".join(diagnostic).lower()
  resource_markers = (
    "outofresources",
    "out of resources",
    "out of memory",
    "resource exhausted",
    "too many resources requested",
  )
  if any(marker in text for marker in resource_markers):
    return "resource_limit"
  if status == "pass":
    return "pass"
  return "candidate"


def _copy_case_for_warp_probe(source_root, destination, manifest, num_warps):
  destination.mkdir(parents=True, exist_ok=False)
  source_root = Path(source_root).resolve()
  for name in manifest["files"]:
    source = (source_root / name).resolve()
    source.relative_to(source_root)
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

  probe_manifest = json.loads(json.dumps(manifest))
  probe_manifest["launch"]["num_warps"] = num_warps
  probe_manifest["execution"]["warp_probe"] = {
    "status": "not_probed",
    "num_warps": num_warps,
  }
  _write_atomic(
    destination / "case.json",
    _canonical_json(probe_manifest).encode("utf-8"),
  )


def probe_warp_settings(
  case_path,
  *,
  warp_counts=V2_WARP_COUNTS,
  timeout_s=DEFAULT_TIMEOUT_S,
  run_case_fn=None,
  gpu_probe=None,
  environment_probe=None,
  symiri=tensor_case.DEFAULT_SYMIRI,
  symirc=DEFAULT_SYMIRC,
  device_id=0,
):
  """Probe legal tensor-profile warp counts in isolated case trees."""
  root = Path(case_path).resolve()
  try:
    case = tensor_case.load_case(root, symiri=symiri)
  except (tensor_case.CaseValidationError, OSError) as error:
    return _report(root, "preflight", "invalid_case", reason=str(error), results=[])
  if case.manifest.get("profile") not in {"tensor-v2", "tensor-v3"}:
    return _report(
      root,
      "preflight",
      "invalid_case",
      reason="warp preflight is defined only for tensor-v2/v3 cases",
      results=[],
    )

  run_case_fn = run_case_fn or run_case
  run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"-{os.getpid()}"
  preflight_root = root / "preflight" / run_id
  preflight_root.mkdir(parents=True, exist_ok=False)
  results = []
  for raw_count in warp_counts:
    if type(raw_count) is not int or raw_count <= 0:
      results.append(
        {
          "num_warps": raw_count,
          "classification": "unsupported_configuration",
          "status": "unsupported_configuration",
          "reason": "num_warps must be a positive integer",
          "case_dir": None,
        }
      )
      continue
    if raw_count not in V2_WARP_COUNTS:
      results.append(
        {
          "num_warps": raw_count,
          "classification": "unsupported_configuration",
          "status": "unsupported_configuration",
          "reason": f"num_warps={raw_count} is outside the tensor profile set",
          "case_dir": None,
        }
      )
      continue

    probe_dir = preflight_root / f"warps-{raw_count}"
    _copy_case_for_warp_probe(root, probe_dir, case.manifest, raw_count)
    report = run_case_fn(
      probe_dir,
      "compiled",
      timeout_s=timeout_s,
      gpu_probe=gpu_probe,
      environment_probe=environment_probe,
      symiri=symiri,
      symirc=symirc,
      device_id=device_id,
    )
    classification = _classify_warp_probe(probe_dir, report)
    probe_manifest_path = probe_dir / "case.json"
    probe_manifest = json.loads(probe_manifest_path.read_text(encoding="utf-8"))
    probe_manifest["execution"]["warp_probe"] = {
      "status": "probed",
      "num_warps": raw_count,
      "device_id": device_id,
      "run_id": run_id,
      "results": [
        {
          "num_warps": raw_count,
          "classification": classification,
          "status": report.get("status", "unknown"),
          "case_dir": None,
        }
      ],
    }
    _write_atomic(
      probe_manifest_path,
      _canonical_json(probe_manifest).encode("utf-8"),
    )
    result = {
      "num_warps": raw_count,
      "classification": classification,
      "status": report.get("status", "unknown"),
      "case_dir": probe_dir.relative_to(root).as_posix(),
      "artifacts": report.get("artifacts", []),
    }
    if report.get("reason"):
      result["reason"] = report["reason"]
    if report.get("artifact_digests"):
      result["artifact_digests"] = report["artifact_digests"]
    results.append(result)

  summary = {
    "schema": "refractir.tensor-warp-preflight.v1",
    "status": "probed",
    "run_id": run_id,
    "source_case": ".",
    "selected_num_warps": case.manifest["launch"]["num_warps"],
    "results": results,
  }
  summary_path = preflight_root / "summary.json"
  _write_atomic(summary_path, _canonical_json(summary).encode("utf-8"))

  manifest_path = root / "case.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  manifest["execution"]["warp_probe"] = {
    "status": "probed",
    "num_warps": manifest["launch"]["num_warps"],
    "run_id": run_id,
    "summary": summary_path.relative_to(root).as_posix(),
    "results": results,
  }
  _write_atomic(manifest_path, _canonical_json(manifest).encode("utf-8"))
  return summary


def _write_log(path, value):
  if value is None:
    data = b""
  elif isinstance(value, bytes):
    data = value
  else:
    data = str(value).encode("utf-8", errors="replace")
  _write_atomic(path, data)


def _default_worker_command(case, mode, artifact_dir):
  return [
    sys.executable,
    "-m",
    "test.lib.triton_case_worker",
    "--case",
    str(case.path),
    "--artifacts-dir",
    str(artifact_dir),
    "--mode",
    mode,
  ]


def _available_ir(artifact_dir, case_root):
  return sorted(
    path.relative_to(case_root).as_posix()
    for path in Path(artifact_dir).glob("compiler-*.ir")
    if path.is_file()
  )


def run_case(
  case_path,
  mode,
  *,
  timeout_s=DEFAULT_TIMEOUT_S,
  worker_command=None,
  gpu_probe=None,
  environment_probe=None,
  symiri=tensor_case.DEFAULT_SYMIRI,
  symirc=DEFAULT_SYMIRC,
  device_id=0,
):
  """Validate, emit, execute, compare, and persist one complete case result."""
  root = Path(case_path).resolve()
  if mode not in {"compiled", "interpret"}:
    report = _report(
      root, mode, "invalid_case", reason="mode must be compiled or interpret"
    )
    _save_report(root, report)
    return report
  if mode == "compiled" and (type(device_id) is not int or device_id < 0):
    report = _report(
      root,
      mode,
      "invalid_case",
      reason="device_id must be a nonnegative integer",
      artifacts=[],
    )
    _save_report(root, report)
    return report
  try:
    case = tensor_case.load_case(root, symiri=symiri)
  except tensor_case.CaseValidationError as error:
    report = _report(root, mode, error.status, reason=str(error), artifacts=[])
    _save_report(root, report)
    return report
  except Exception as error:
    report = _report(
      root,
      mode,
      "invalid_case",
      reason=f"case validation failed: {error}",
      artifacts=[],
    )
    _save_report(root, report)
    return report

  if mode == "compiled":
    available_gpu = gpu_probe() if gpu_probe is not None else _gpu_available(device_id)
    if not available_gpu:
      report = _report(
        root,
        mode,
        "environment_unavailable",
        reason=f"CUDA device {device_id} unavailable",
        exit_code=2,
        artifacts=["program.sir", "inputs.bin", "expected.bin"],
      )
      _save_report(root, report)
      return report

  artifact_dir = root / "runs" / mode
  artifact_dir.mkdir(parents=True, exist_ok=True)
  for existing in artifact_dir.iterdir():
    if existing.is_dir():
      shutil.rmtree(existing)
    elif existing.is_file():
      existing.unlink()
  artifact_prefix = artifact_dir.relative_to(root).as_posix()

  def artifact(name):
    return f"{artifact_prefix}/{name}"

  kernel_path, emit_stderr, emit_stdout = _emit_kernel(case, symirc, artifact_dir)
  _write_log(artifact_dir / "compiler.log", emit_stdout + emit_stderr)
  if kernel_path is None:
    report = _report(
      root,
      mode,
      "compile_failure",
      reason=emit_stderr,
      artifacts=["program.sir", "inputs.bin", "expected.bin", artifact("compiler.log")],
    )
    _save_report(root, report)
    return report

  environment_probe = environment_probe or probe_triton_env.collect_environment
  try:
    environment_record = environment_probe()
  except Exception as error:
    report = _report(
      root,
      mode,
      "environment_unavailable",
      reason=f"environment probe failed: {error}",
      exit_code=2,
      artifacts=[
        "program.sir",
        "inputs.bin",
        "expected.bin",
        artifact("kernel.py"),
        artifact("compiler.log"),
      ],
    )
    _save_report(root, report)
    return report
  try:
    _save_environment(artifact_dir, environment_record)
  except Exception as error:
    report = _report(
      root,
      mode,
      "environment_unavailable",
      reason=f"environment artifact could not be saved: {error}",
      exit_code=2,
      artifacts=[artifact("kernel.py"), artifact("compiler.log")],
    )
    _save_report(root, report)
    return report
  if mode == "compiled" and not environment_record.get("campaign_available"):
    report = _report(
      root,
      mode,
      "environment_unavailable",
      reason=", ".join(environment_record.get("failure_reasons", [])),
      exit_code=2,
      artifacts=[
        "program.sir",
        "inputs.bin",
        "expected.bin",
        artifact("kernel.py"),
        artifact("compiler.log"),
        artifact("environment.json"),
        artifact("package-lock.json"),
      ],
    )
    _save_report(root, report)
    return report

  source_revision = os.environ.get(
    "REFRACTIR_REVISION", environment_record.get("refractir_revision") or "unavailable"
  )
  compiler_flags = os.environ.get("REFRACTIR_COMPILER_FLAGS", "unavailable")
  compiler_digest = None
  try:
    compiler_digest = hashlib.sha256(Path(symirc).read_bytes()).hexdigest()
  except OSError:
    pass
  campaign_valid = bool(
    environment_record.get("campaign_available")
    and source_revision != "unavailable"
    and compiler_flags != "unavailable"
    and compiler_digest
    and environment_record.get("triton_source_revision") not in (None, "unavailable")
    and environment_record.get("package_lock_sha256")
  )
  cache_dir = artifact_dir / "triton-cache" if mode == "compiled" else None
  compiler_cache = None
  if cache_dir is not None:
    if cache_dir.exists():
      shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    compiler_cache = {
      "cache_id": hashlib.sha256(str(cache_dir.resolve()).encode("utf-8")).hexdigest(),
      "path": f"{artifact_prefix}/triton-cache",
      "empty_at_start": not any(cache_dir.iterdir()),
    }
  run_metadata = {
    "schema": "refractir.tensor-run.v1",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "mode": mode,
    "entry": case.manifest["entry"],
    "grid": case.manifest["grid"],
    "launch": case.manifest["launch"],
    "buffer_layouts": {
      "inputs": [
        {
          "name": record["name"],
          "shape": record["shape"],
          "element_strides": record["element_strides"],
          "element_offset": record["element_offset"],
          "storage_span": record["storage_span"],
          "axis_order": record["axis_order"],
        }
        for record in case.manifest["inputs"]
      ],
      "output": {
        "shape": case.manifest["output"]["shape"],
        "element_strides": case.manifest["output"]["element_strides"],
        "element_offset": case.manifest["output"]["element_offset"],
        "storage_span": case.manifest["output"]["storage_span"],
        "axis_order": case.manifest["output"]["axis_order"],
      },
    }
    if case.manifest["profile"] in {"tensor-v2", "tensor-v3"}
    else None,
    "layout_sentinels": case.manifest.get("execution"),
    "source_sha256": hashlib.sha256(case.program).hexdigest(),
    "inputs_sha256": hashlib.sha256(case.inputs).hexdigest(),
    "expected_sha256": hashlib.sha256(case.expected).hexdigest(),
    "kernel_sha256": hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
    "refractir_revision": source_revision,
    "working_tree_dirty": os.environ.get("REFRACTIR_WORKTREE_DIRTY", "unknown"),
    "compiler": {
      "executable": str(symirc),
      "sha256": compiler_digest,
      "flags": compiler_flags,
    },
    "environment": artifact("environment.json"),
    "package_lock": artifact("package-lock.json"),
    "selected_device": device_id if mode == "compiled" else None,
    "compiler_cache": compiler_cache,
  }
  _write_atomic(
    artifact_dir / "run.json", _canonical_json(run_metadata).encode("utf-8")
  )
  actual_path = artifact_dir / "actual.bin"
  actual_path.unlink(missing_ok=True)
  (artifact_dir / "worker-phase.json").unlink(missing_ok=True)
  command = worker_command or _default_worker_command(case, mode, artifact_dir)
  try:
    child = run_child(
      command,
      mode,
      timeout_s,
      cwd=ROOT,
      cache_dir=cache_dir,
      device_id=device_id if mode == "compiled" else None,
    )
  except subprocess.TimeoutExpired as error:
    _write_log(artifact_dir / "stdout.log", error.stdout)
    _write_log(artifact_dir / "stderr.log", error.stderr)
    report = _report(
      root,
      mode,
      "runtime_timeout",
      timeout_s=timeout_s,
      exit_code=124,
      stdout=artifact("stdout.log"),
      stderr=artifact("stderr.log"),
      artifacts=[
        "program.sir",
        "inputs.bin",
        "expected.bin",
        artifact("kernel.py"),
        artifact("run.json"),
        artifact("compiler.log"),
        artifact("stdout.log"),
        artifact("stderr.log"),
        artifact("environment.json"),
        artifact("package-lock.json"),
        artifact("worker-phase.json"),
        artifact("worker-timing.json"),
        artifact("actual.bin"),
      ]
      + _available_ir(artifact_dir, root),
    )
    _save_report(root, report)
    return report
  except OSError as error:
    _write_log(artifact_dir / "stdout.log", b"")
    _write_log(artifact_dir / "stderr.log", str(error))
    report = _report(
      root,
      mode,
      "runtime_failure",
      reason=str(error),
      exit_code=127,
      artifacts=[
        "program.sir",
        "inputs.bin",
        "expected.bin",
        artifact("kernel.py"),
        artifact("stdout.log"),
        artifact("stderr.log"),
      ],
    )
    _save_report(root, report)
    return report

  _write_log(artifact_dir / "stdout.log", child.stdout)
  _write_log(artifact_dir / "stderr.log", child.stderr)
  common_artifacts = [
    "program.sir",
    "inputs.bin",
    "expected.bin",
    artifact("kernel.py"),
    artifact("run.json"),
    artifact("compiler.log"),
    artifact("stdout.log"),
    artifact("stderr.log"),
    artifact("environment.json"),
    artifact("package-lock.json"),
    artifact("worker-phase.json"),
    artifact("worker-timing.json"),
    artifact("actual.bin"),
  ]
  if (artifact_dir / "tensor-layout.json").is_file():
    common_artifacts.append(artifact("tensor-layout.json"))
  if (artifact_dir / "worker-staging.json").is_file():
    common_artifacts.append(artifact("worker-staging.json"))
  if (artifact_dir / "actual-physical.bin").is_file():
    common_artifacts.append(artifact("actual-physical.bin"))
  if child.returncode != 0:
    phase = "runtime"
    try:
      phase = json.loads((artifact_dir / "worker-phase.json").read_text()).get(
        "phase", phase
      )
    except (OSError, json.JSONDecodeError):
      pass
    status = "compile_failure" if phase == "compilation" else "runtime_failure"
    report = _report(
      root,
      mode,
      status,
      exit_code=child.returncode,
      stdout=artifact("stdout.log"),
      stderr=artifact("stderr.log"),
      artifacts=common_artifacts + _available_ir(artifact_dir, root),
    )
    _save_report(root, report)
    return report

  if not actual_path.is_file():
    report = _report(
      root,
      mode,
      "runtime_failure",
      exit_code=child.returncode,
      reason="worker did not produce actual.bin",
      artifacts=common_artifacts,
    )
    _save_report(root, report)
    return report
  actual = actual_path.read_bytes()
  if len(actual) != case.manifest["output"]["byte_length"]:
    report = _report(
      root,
      mode,
      "runtime_failure",
      exit_code=child.returncode,
      reason="worker output is partial or has trailing bytes",
      actual_bytes=len(actual),
      expected_bytes=case.manifest["output"]["byte_length"],
      artifacts=common_artifacts + _available_ir(artifact_dir, root),
    )
    _save_report(root, report)
    return report
  output = case.manifest["output"]
  comparison = compare_outputs(
    actual,
    case.expected,
    output["dtype"],
    output["shape"],
    case.manifest["comparison"],
  )
  report = _report(
    root,
    mode,
    comparison["status"],
    comparison=comparison,
    campaign_valid=campaign_valid,
    actual_bytes=len(actual),
    actual_sha256=hashlib.sha256(actual).hexdigest(),
    exit_code=0 if comparison["status"] == "pass" else 1,
    stdout=artifact("stdout.log"),
    stderr=artifact("stderr.log"),
    artifacts=common_artifacts + _available_ir(artifact_dir, root),
  )
  _save_report(root, report)
  return report


def main(argv=None, *, gpu_probe=None, environment_probe=None):
  parser = argparse.ArgumentParser(description=__doc__)
  selection = parser.add_mutually_exclusive_group(required=True)
  selection.add_argument("--case")
  selection.add_argument("--suite")
  parser.add_argument("--mode", choices=("compiled", "interpret"), required=True)
  parser.add_argument(
    "--probe-warps",
    action="store_true",
    help="run isolated tensor-profile preflight probes for num_warps 1, 2, 4, and 8",
  )
  parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
  parser.add_argument("--device", type=int, default=0, dest="device_id")
  args = parser.parse_args(argv)
  if args.probe_warps:
    if args.mode != "compiled":
      report = {
        "mode": "preflight",
        "status": "invalid_case",
        "reason": "warp preflight requires compiled mode",
      }
      print(json.dumps(report, ensure_ascii=False, sort_keys=True))
      return 2
    if args.case:
      report = probe_warp_settings(
        args.case,
        timeout_s=args.timeout,
        gpu_probe=gpu_probe,
        environment_probe=environment_probe,
        device_id=args.device_id,
      )
      print(json.dumps(report, ensure_ascii=False, sort_keys=True))
      if report["status"] != "probed":
        return 1
      if any(
        item["classification"] == "environment_unavailable"
        for item in report["results"]
      ):
        return 2
      return int(
        any(
          item["classification"] not in {"pass", "unsupported_configuration"}
          for item in report["results"]
        )
      )
    suite_path = Path(args.suite)
    case_paths = (
      sorted(
        child
        for child in suite_path.iterdir()
        if child.is_dir() and (child / "case.json").is_file()
      )
      if suite_path.is_dir()
      else []
    )
    if not case_paths:
      report = {
        "suite": args.suite,
        "mode": "preflight",
        "status": "invalid_case",
        "reason": "no case directories found",
      }
      print(json.dumps(report, ensure_ascii=False, sort_keys=True))
      return 2
    reports = [
      probe_warp_settings(
        case_path,
        timeout_s=args.timeout,
        gpu_probe=gpu_probe,
        environment_probe=environment_probe,
        device_id=args.device_id,
      )
      for case_path in case_paths
    ]
    summary = {
      "mode": "preflight",
      "status": "probed"
      if all(report["status"] == "probed" for report in reports)
      else "failure",
      "cases": reports,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if summary["status"] != "probed":
      return 1
    if any(
      item["classification"] == "environment_unavailable"
      for report in reports
      for item in report["results"]
    ):
      return 2
    return int(
      any(
        item["classification"] not in {"pass", "unsupported_configuration"}
        for report in reports
        for item in report["results"]
      )
    )
  if args.case:
    report = run_case(
      args.case,
      args.mode,
      timeout_s=args.timeout,
      gpu_probe=gpu_probe,
      environment_probe=environment_probe,
      device_id=args.device_id,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if report["status"] == "pass":
      return 0 if report.get("campaign_valid", False) else 1
    return int(report.get("exit_code", 1)) or 1
  else:
    suite_path = Path(args.suite)
    case_paths = (
      sorted(
        child
        for child in suite_path.iterdir()
        if child.is_dir() and (child / "case.json").is_file()
      )
      if suite_path.is_dir()
      else []
    )
  if not case_paths:
    report = {
      "suite": args.suite,
      "mode": args.mode,
      "status": "invalid_case",
      "reason": "no case directories found",
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 2
  reports = [
    run_case(
      case_path,
      args.mode,
      timeout_s=args.timeout,
      gpu_probe=gpu_probe,
      environment_probe=environment_probe,
      device_id=args.device_id,
    )
    for case_path in case_paths
  ]
  all_pass = all(report["status"] == "pass" for report in reports)
  campaign_valid = all(report.get("campaign_valid", False) for report in reports)
  summary = {
    "mode": args.mode,
    "status": "pass" if all_pass else "failure",
    "campaign_valid": campaign_valid,
    "cases": reports,
  }
  print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
  if summary["status"] == "pass" and campaign_valid:
    return 0
  return max(
    (
      int(report.get("exit_code", 1)) or 1
      for report in reports
      if report["status"] != "pass"
    ),
    default=1,
  )


if __name__ == "__main__":
  sys.exit(main())
