"""Capture the exact host, Python, package, and NVIDIA environment for cases."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

_LOCK_SCHEMA = "refractir.tensor-environment-lock.v1"
_ENV_SCHEMA = "refractir.tensor-environment.v1"
_UNAVAILABLE = "unavailable"


def _canonical_json(value):
  return json.dumps(
    value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
  ).encode("utf-8")


def _package_records():
  packages = []
  for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name")
    version = distribution.version
    if not name or not version:
      continue
    artifacts = []
    for file in distribution.files or ():
      file_hash = getattr(file, "hash", None)
      if file_hash is None or not file_hash.mode or not file_hash.value:
        continue
      artifacts.append(
        {
          "path": str(file),
          "algorithm": file_hash.mode,
          "digest": file_hash.value,
        }
      )
    packages.append(
      {
        "name": name,
        "version": version,
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
      }
    )
  return sorted(packages, key=lambda item: item["name"].casefold())


def _package_version(name):
  try:
    return importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:
    return None


def _nvidia_rows():
  try:
    result = subprocess.run(
      [
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
      ],
      capture_output=True,
      text=True,
      timeout=5,
      check=False,
    )
  except (OSError, subprocess.TimeoutExpired):
    return []
  if result.returncode != 0:
    return []
  return [row for row in csv.reader(result.stdout.splitlines()) if len(row) == 2]


def _gpu_records(torch_module):
  if torch_module is None:
    return []
  try:
    if not torch_module.cuda.is_available():
      return []
    count = torch_module.cuda.device_count()
    rows = _nvidia_rows()
    gpus = []
    for index in range(count):
      properties = torch_module.cuda.get_device_properties(index)
      name = str(properties.name)
      driver = rows[index][1].strip() if index < len(rows) else None
      gpus.append(
        {
          "name": name,
          "compute_capability": f"{properties.major}.{properties.minor}",
          "driver_version": driver,
        }
      )
    return gpus
  except Exception:  # torch can be installed without a working CUDA runtime
    return []


def _triton_source_revision():
  distribution = None
  try:
    distribution = importlib.metadata.distribution("triton")
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
      direct = json.loads(direct_url)
      info = direct.get("vcs_info", {})
      revision = info.get("commit_id")
      if revision:
        return revision
      parsed = urlparse(direct.get("url", ""))
      if parsed.scheme == "file" and parsed.netloc in ("", "localhost"):
        source_path = Path(unquote(parsed.path))
        if source_path.is_file():
          source_path = source_path.parent
        for directory in (source_path, *source_path.parents):
          if not (directory / ".git").exists():
            continue
          result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
          )
          if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
  except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError, OSError):
    pass

  try:
    import triton

    package_path = Path(triton.__file__).resolve()
    for directory in package_path.parents:
      if not (directory / ".git").exists():
        continue
      result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
      )
      if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
  except Exception:
    pass
  return _UNAVAILABLE


def _refractir_revision():
  try:
    result = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      capture_output=True,
      text=True,
      timeout=2,
      check=False,
    )
  except (OSError, subprocess.TimeoutExpired):
    return _UNAVAILABLE
  if result.returncode != 0:
    return _UNAVAILABLE
  return result.stdout.strip() or _UNAVAILABLE


def _system_snapshot():
  try:
    import torch
  except Exception:
    torch = None

  cuda_version = None
  if torch is not None:
    try:
      cuda_version = torch.version.cuda
    except Exception:
      cuda_version = None

  return {
    "gpus": _gpu_records(torch),
    "cuda_version": cuda_version,
    "python_version": platform.python_version(),
    "torch_version": getattr(torch, "__version__", None)
    if torch is not None
    else _package_version("torch"),
    "triton_version": _package_version("triton"),
    "triton_source_revision": _triton_source_revision(),
    "refractir_revision": _refractir_revision(),
    "packages": _package_records(),
  }


def _make_package_lock(packages):
  normalized = []
  for package in packages:
    artifacts = package.get("artifacts", [])
    normalized.append(
      {
        "name": str(package["name"]),
        "version": str(package["version"]),
        "artifacts": sorted(
          [
            {
              "path": str(artifact["path"]),
              "algorithm": str(artifact["algorithm"]),
              "digest": str(artifact["digest"]),
            }
            for artifact in artifacts
          ],
          key=lambda item: item["path"],
        ),
      }
    )
  normalized.sort(key=lambda item: item["name"].casefold())
  return {"schema": _LOCK_SCHEMA, "packages": normalized}


def collect_environment(probe=None):
  """Return a JSON-compatible environment record.

  `probe` is an optional zero-argument provider used by controlled probes and
  tests. The default provider inspects the current Python and CUDA runtime.
  """
  snapshot = (probe or _system_snapshot)()
  gpus = snapshot.get("gpus") or []
  source_revision = snapshot.get("triton_source_revision") or _UNAVAILABLE
  packages = snapshot.get("packages") or []
  package_names = {str(item.get("name", "")).casefold() for item in packages}

  failures = []
  if not gpus:
    failures.append("gpu_unavailable")
  else:
    for index, gpu in enumerate(gpus):
      for field in ("name", "compute_capability", "driver_version"):
        if not gpu.get(field):
          failures.append(f"gpu_{index}_{field}_unavailable")
  required = (
    "cuda_version",
    "python_version",
    "torch_version",
    "triton_version",
  )
  for field in required:
    if not snapshot.get(field):
      failures.append(f"{field}_unavailable")
  if "torch" not in package_names:
    failures.append("torch_package_lock_missing")
  if "triton" not in package_names:
    failures.append("triton_package_lock_missing")

  lock = _make_package_lock(packages)
  lock_digest = hashlib.sha256(_canonical_json(lock)).hexdigest()
  status = "environment_available" if not failures else "environment_unavailable"
  record = {
    "schema": _ENV_SCHEMA,
    "status": status,
    "campaign_available": not failures,
    "failure_reasons": failures,
    "gpus": gpus,
    "driver_version": gpus[0].get("driver_version") if gpus else None,
    "cuda_version": snapshot.get("cuda_version"),
    "python_version": snapshot.get("python_version"),
    "torch_version": snapshot.get("torch_version"),
    "triton_version": snapshot.get("triton_version"),
    "triton_source_revision": source_revision,
    "refractir_revision": snapshot.get("refractir_revision") or _UNAVAILABLE,
    "package_lock_file": "package-lock.json",
    "package_lock_sha256": lock_digest,
    "package_lock": lock,
  }
  return record


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


def write_environment(output_dir, record):
  output_dir = Path(output_dir)
  lock = record["package_lock"]
  environment = {key: value for key, value in record.items() if key != "package_lock"}
  _write_atomic(output_dir / "package-lock.json", _canonical_json(lock))
  _write_atomic(
    output_dir / "environment.json",
    (
      json.dumps(environment, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8"),
  )


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", required=True)
  args = parser.parse_args(argv)

  record = collect_environment()
  write_environment(args.output_dir, record)
  printable = {key: value for key, value in record.items() if key != "package_lock"}
  print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
  return 0 if record["campaign_available"] else 2


if __name__ == "__main__":
  sys.exit(main())
