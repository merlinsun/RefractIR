#!/usr/bin/env python3
"""Run one prospective tensor-v2 pilot with one worker per GPU."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _write_json(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(
    json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  os.replace(temporary, path)


def _load_case(path):
  manifest = json.loads((path / "case.json").read_text(encoding="utf-8"))
  source = (path / "program.sir").read_text(encoding="utf-8")
  loop_bounds = manifest.get("execution", {}).get("loop_bounds", [])
  is_single_loop_f32 = (
    manifest.get("coverage", {}).get("primary_stratum") == "loop-reduction"
    and len(loop_bounds) == 1
    and manifest.get("inputs", [{}])[0].get("dtype") == "f32"
    and "^outer_header:" not in source
  )
  return manifest, is_single_loop_f32


def _sha256(path):
  digest = hashlib.sha256()
  with Path(path).open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _deduplicate_package_locks(pilot_root):
  locks = sorted(Path(pilot_root).rglob("package-lock.json"))
  if not locks:
    return {"files": 0, "unique_sha256": []}
  shared = Path(pilot_root) / "orchestration" / "shared" / "package-lock.json"
  shared.parent.mkdir(parents=True, exist_ok=True)
  source = next((path for path in locks if path != shared), None)
  if source is None:
    return {"files": 0, "unique_sha256": []}
  shutil.copy2(source, shared)
  shared_digest = _sha256(shared)
  linked = 0
  digests = {shared_digest}
  for path in locks:
    if path == shared:
      continue
    digest = _sha256(path)
    digests.add(digest)
    if digest == shared_digest:
      path.unlink()
      os.link(shared, path)
      linked += 1
  return {
    "files": len(locks),
    "hardlinks_created": linked,
    "unique_sha256": sorted(digests),
    "shared_sha256": shared_digest,
  }


def _run_suite(source_root, suite_dir, mode, device, *, probe, timeout_s, env, log_dir):
  command = [
    sys.executable,
    "-m",
    "test.lib.run_triton_tests",
    "--suite",
    str(suite_dir),
    "--mode",
    mode,
    "--timeout",
    str(timeout_s),
    "--device",
    str(device),
  ]
  if probe:
    command.append("--probe-warps")
  started = time.monotonic()
  try:
    result = subprocess.run(
      command,
      cwd=source_root,
      env=env,
      capture_output=True,
      text=True,
      timeout=max(120, timeout_s * 4 * len(list(suite_dir.iterdir()))),
      check=False,
    )
    stdout = result.stdout
    stderr = result.stderr
    exit_code = result.returncode
  except subprocess.TimeoutExpired as error:
    stdout = error.stdout or ""
    stderr = error.stderr or ""
    if isinstance(stdout, bytes):
      stdout = stdout.decode("utf-8", errors="replace")
    if isinstance(stderr, bytes):
      stderr = stderr.decode("utf-8", errors="replace")
    exit_code = 124
  except OSError as error:
    stdout = ""
    stderr = str(error)
    exit_code = 127
  log_dir.mkdir(parents=True, exist_ok=True)
  (log_dir / f"{mode}{'-warps' if probe else ''}.stdout.log").write_text(
    stdout, encoding="utf-8", errors="replace"
  )
  (log_dir / f"{mode}{'-warps' if probe else ''}.stderr.log").write_text(
    stderr, encoding="utf-8", errors="replace"
  )
  report = None
  try:
    report = json.loads(stdout.splitlines()[-1])
  except (IndexError, json.JSONDecodeError):
    pass
  return {
    "mode": "preflight" if probe else mode,
    "probe_warps": probe,
    "device_id": device,
    "exit_code": exit_code,
    "elapsed_seconds": round(time.monotonic() - started, 6),
    "report": report,
    "stdout_log": str((log_dir / f"{mode}{'-warps' if probe else ''}.stdout.log")),
    "stderr_log": str((log_dir / f"{mode}{'-warps' if probe else ''}.stderr.log")),
  }


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--suite", required=True, type=Path)
  parser.add_argument("--source-root", required=True, type=Path)
  parser.add_argument("--triton-python", required=True)
  parser.add_argument("--ptxas", required=True)
  parser.add_argument("--source-snapshot-sha256", required=True)
  parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
  parser.add_argument("--timeout", type=float, default=60.0)
  args = parser.parse_args(argv)

  suite = args.suite.resolve()
  source_root = args.source_root.resolve()
  devices = [int(value) for value in args.devices.split(",") if value]
  if (
    not devices
    or len(set(devices)) != len(devices)
    or any(value < 0 for value in devices)
  ):
    parser.error("--devices must be a nonempty comma-separated list of unique GPU ids")
  if not suite.is_dir() or not source_root.is_dir():
    parser.error("suite and source-root must be existing directories")
  cases = sorted(
    (
      path
      for path in suite.iterdir()
      if path.is_dir() and (path / "case.json").is_file()
    ),
    key=lambda path: path.name,
  )
  if not cases:
    parser.error("suite contains no tensor case directories")

  orchestration = suite / "orchestration"
  if orchestration.exists():
    parser.error(f"orchestration directory already exists: {orchestration}")
  all_shards = [
    orchestration / "shards" / "all" / f"gpu-{device}" for device in devices
  ]
  probe_shards = [
    orchestration / "shards" / "single-loop-f32" / f"gpu-{device}" for device in devices
  ]
  for directory in all_shards + probe_shards:
    directory.mkdir(parents=True)
  assigned = {device: [] for device in devices}
  probe_counts = {device: 0 for device in devices}
  for index, case_path in enumerate(cases):
    device = devices[index % len(devices)]
    assigned[device].append(case_path)
    (all_shards[devices.index(device)] / case_path.name).symlink_to(case_path)
    _, should_probe = _load_case(case_path)
    if should_probe:
      (probe_shards[devices.index(device)] / case_path.name).symlink_to(case_path)
      probe_counts[device] += 1

  env = dict(os.environ)
  env.pop("CUDA_VISIBLE_DEVICES", None)
  env.pop("TRITON_INTERPRET", None)
  env.pop("TRITON_CACHE_DIR", None)
  env["PYTHONPATH"] = args.triton_python
  env["TRITON_PTXAS_PATH"] = args.ptxas
  env["REFRACTIR_REVISION"] = f"source-snapshot-sha256:{args.source_snapshot_sha256}"
  env["REFRACTIR_COMPILER_FLAGS"] = (
    "g++ -std=c++20 -Iinclude -Wall -Wextra -O2 "
    "-DREFRACTIR_TENSOR_REF -DUSE_BITWUZLA -I/tmp/boost_1_89"
  )
  env["REFRACTIR_WORKTREE_DIRTY"] = "dirty-source-snapshot"
  env["OMP_NUM_THREADS"] = "1"
  env["PYTHONHASHSEED"] = "0"
  nvidia_smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
    capture_output=True,
    text=True,
    check=False,
  )
  metadata = {
    "schema": "refractir.generator-refresh-pilot-run.v1",
    "case_count": len(cases),
    "source_snapshot_sha256": args.source_snapshot_sha256,
    "triton_python": args.triton_python,
    "ptxas": args.ptxas,
    "ptxas_sha256": _sha256(args.ptxas),
    "devices": devices,
    "gpu_inventory": nvidia_smi.stdout.splitlines(),
    "cases_per_device": {str(device): len(items) for device, items in assigned.items()},
    "single_loop_f32_cases_per_device": {
      str(device): count for device, count in probe_counts.items()
    },
    "commands": {
      "interpret": "python -m test.lib.run_triton_tests --suite SHARD --mode interpret",
      "compiled": "python -m test.lib.run_triton_tests --suite SHARD --mode compiled --device GPU",
      "warp_probe": "python -m test.lib.run_triton_tests --suite SHARD --mode compiled --probe-warps --device GPU",
    },
  }
  _write_json(orchestration / "plan.json", metadata)

  def run_device(device):
    slot = devices.index(device)
    log_dir = orchestration / "logs" / f"gpu-{device}"
    records = []
    for phase, probe in (("interpret", False), ("compiled", False)):
      report = _run_suite(
        source_root,
        all_shards[slot],
        phase,
        device,
        probe=False,
        timeout_s=args.timeout,
        env=env,
        log_dir=log_dir,
      )
      records.append(report)
      _write_json(orchestration / "workers" / f"gpu-{device}.json", records)
    if probe_counts[device]:
      report = _run_suite(
        source_root,
        probe_shards[slot],
        "compiled",
        device,
        probe=True,
        timeout_s=args.timeout,
        env=env,
        log_dir=log_dir,
      )
      records.append(report)
      _write_json(orchestration / "workers" / f"gpu-{device}.json", records)
    return device, records

  with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
    futures = [pool.submit(run_device, device) for device in devices]
    worker_records = dict(future.result() for future in futures)
  deduplication = _deduplicate_package_locks(suite)
  final = {
    **metadata,
    "worker_records": {
      str(device): records for device, records in worker_records.items()
    },
    "package_lock_deduplication": deduplication,
  }
  _write_json(orchestration / "summary.json", final)
  print(
    json.dumps(
      {
        "case_count": len(cases),
        "workers": len(devices),
        "single_loop_f32_probes": sum(probe_counts.values()),
        "package_lock_deduplication": deduplication,
        "summary": str(orchestration / "summary.json"),
      },
      sort_keys=True,
    )
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
