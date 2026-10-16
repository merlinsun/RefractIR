#!/usr/bin/env python3
"""Run a resumable, bounded-storage tensor campaign against Triton."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(
  os.environ.get("REFRACTIR_CAMPAIGN_BASE", "/tmp/refractir-bughunt-20261007-867e505")
)
SOURCE = Path(os.environ.get("REFRACTIR_SOURCE_ROOT", str(BASE / "source")))
CAMPAIGN = Path(
  os.environ.get("REFRACTIR_CAMPAIGN_DIR", str(BASE / "continuous-v2-configurable"))
)
WORK = CAMPAIGN / "work"
LOGS = CAMPAIGN / "logs"
PYTHON = Path(
  os.environ.get("REFRACTIR_CAMPAIGN_PYTHON", "/root/triton-latest/.venv/bin/python")
)
REVISION = os.environ.get("REFRACTIR_REVISION", "unavailable")
TRITON = os.environ.get("REFRACTIR_TRITON_BUILD", "3.9.0+git32f06df4")
TRITON_PYTHON = Path(
  os.environ.get("REFRACTIR_TRITON_PYTHON", "/root/triton-latest/python")
)
MAX_WORKERS = int(os.environ.get("REFRACTIR_CAMPAIGN_WORKERS", "4"))
GPU_IDS = tuple(
  int(value)
  for value in os.environ.get("REFRACTIR_CAMPAIGN_GPU_IDS", "0,1,2,3").split(",")
  if value.strip()
)
START_SEED = int(os.environ.get("REFRACTIR_CAMPAIGN_START_SEED", "2027"))
PROFILE_VERSION = os.environ.get("REFRACTIR_TENSOR_PROFILE_VERSION", "v2")
TENSOR_DTYPE = os.environ.get("REFRACTIR_TENSOR_DTYPE", "auto")
TENSOR_GENERATOR = os.environ.get("REFRACTIR_TENSOR_GENERATOR", "constructor")
TENSOR_INPUT_POLICY = os.environ.get("REFRACTIR_TENSOR_INPUT_POLICY", "ordinary")
TENSOR_PROPERTY_TARGET = os.environ.get("REFRACTIR_TENSOR_PROPERTY_TARGET")
TENSOR_OUTPUT_ELEMENTS = tuple(
  value
  for value in os.environ.get("REFRACTIR_TENSOR_OUTPUT_ELEMENTS", "").split(";")
  if value
)
CASES_PER_SEED = int(os.environ.get("REFRACTIR_CAMPAIGN_CASES_PER_SEED", "32"))
MIN_FREE_BYTES = 2 * 1024**3
NEXT_SEED = CAMPAIGN / "next_seed"
STATUS = CAMPAIGN / "status.json"
EVENTS = CAMPAIGN / "passes.jsonl"
STOP = CAMPAIGN / "STOP.json"
FOUND = CAMPAIGN / "FOUND.json"
CASE_SUMMARY_SCHEMA = "refractir.tensor-case-compiler-summary.v1"
SUMMARY_STAGES = ("ttir", "ttgir", "llvm", "ptx")
SUMMARY_STAGE_ALIASES = {
  "ttir": "ttir",
  "ttgir": "ttgir",
  "llir": "llvm",
  "llvm": "llvm",
  "ptx": "ptx",
}
REPRESENTATIVE_QUOTA = 64


def validate_route(
  *,
  generator,
  input_policy,
  dtype,
  profile_version="v2",
  property_target=None,
  output_elements=(),
):
  """Reject combinations that the tensor generator cannot reify."""
  output_elements = tuple(output_elements)
  if generator not in {"constructor", "compositional"}:
    raise ValueError("generator must be constructor or compositional")
  if input_policy not in {"ordinary", "boundary", "exact"}:
    raise ValueError("input policy must be ordinary, boundary, or exact")
  if generator == "constructor" and input_policy != "ordinary":
    raise ValueError("constructor generator supports only the ordinary input policy")
  if generator == "compositional" and profile_version not in {"v2", "v3"}:
    raise ValueError("compositional generator supports only tensor profile v2 or v3")
  if input_policy == "exact":
    if generator != "compositional" or dtype != "i32":
      raise ValueError("exact input selection requires compositional i32 generation")
    if property_target is not None and not str(property_target).strip():
      raise ValueError("exact input selection target must not be empty")
    if any(not str(target).strip() for target in output_elements):
      raise ValueError("exact output-element targets must not be empty")
    if bool(property_target is not None) == bool(output_elements):
      raise ValueError("exact input selection requires exactly one target form")
  elif property_target is not None or output_elements:
    raise ValueError("targets are available only with the exact input policy")


def build_generation_command(
  rysmith_path,
  output_dir,
  seed,
  *,
  profile_version="v2",
  dtype="auto",
  generator="constructor",
  input_policy="ordinary",
  property_target=None,
  output_elements=(),
  count=32,
):
  """Build one explicit, reproducible tensor generation invocation."""
  output_elements = tuple(output_elements)
  validate_route(
    generator=generator,
    input_policy=input_policy,
    dtype=dtype,
    profile_version=profile_version,
    property_target=property_target,
    output_elements=output_elements,
  )
  command = [
    str(rysmith_path),
    "--profile",
    "tensor",
    "--target",
    "sir",
    "--tensor-profile-version",
    profile_version,
    "--tensor-dtype",
    dtype,
    "--tensor-synthesis",
    "construction",
    "--tensor-generator",
    generator,
    "--tensor-input-policy",
    input_policy,
    "--seed",
    str(seed),
    "-n",
    str(count),
    "--output-dir",
    str(output_dir),
  ]
  if property_target is not None:
    command.extend(["--tensor-property-target", str(property_target)])
  for target in output_elements:
    command.extend(["--tensor-output-element", str(target)])
  return command


def route_metadata():
  return {
    "profile_version": PROFILE_VERSION,
    "tensor_dtype": TENSOR_DTYPE,
    "tensor_generator": TENSOR_GENERATOR,
    "tensor_input_policy": TENSOR_INPUT_POLICY,
    "property_target": TENSOR_PROPERTY_TARGET,
    "output_elements": list(TENSOR_OUTPUT_ELEMENTS),
  }


def campaign_metadata():
  return {
    "schema": "refractir.continuous-triton-campaign-config.v1",
    "refractir_revision": REVISION,
    "triton_build": TRITON,
    "source_root": str(SOURCE),
    "python": str(PYTHON),
    "triton_python": str(TRITON_PYTHON),
    "gpu_ids": list(GPU_IDS),
    "worker_count": MAX_WORKERS,
    "start_seed": START_SEED,
    "cases_per_seed": CASES_PER_SEED,
    **route_metadata(),
  }


def parse_arguments(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-root", type=Path, default=SOURCE)
  parser.add_argument("--campaign-dir", type=Path, default=CAMPAIGN)
  parser.add_argument("--python", type=Path, default=PYTHON)
  parser.add_argument("--triton-python", type=Path, default=TRITON_PYTHON)
  parser.add_argument("--revision", default=REVISION)
  parser.add_argument("--triton-build", default=TRITON)
  parser.add_argument(
    "--profile-version", choices=("v2", "v3"), default=PROFILE_VERSION
  )
  parser.add_argument("--dtype", choices=("auto", "i32", "f32"), default=TENSOR_DTYPE)
  parser.add_argument(
    "--generator", choices=("constructor", "compositional"), default=TENSOR_GENERATOR
  )
  parser.add_argument(
    "--input-policy",
    choices=("ordinary", "boundary", "exact"),
    default=TENSOR_INPUT_POLICY,
  )
  parser.add_argument("--property-target", default=TENSOR_PROPERTY_TARGET)
  parser.add_argument(
    "--output-element", action="append", default=list(TENSOR_OUTPUT_ELEMENTS)
  )
  parser.add_argument("--gpu-ids", default=",".join(str(i) for i in GPU_IDS))
  parser.add_argument("--workers", type=int, default=MAX_WORKERS)
  parser.add_argument("--start-seed", type=int, default=START_SEED)
  parser.add_argument("--cases-per-seed", type=int, default=CASES_PER_SEED)
  args = parser.parse_args(argv)
  try:
    args.gpu_ids = tuple(
      int(value.strip()) for value in args.gpu_ids.split(",") if value.strip()
    )
  except ValueError:
    parser.error("--gpu-ids must be comma-separated nonnegative integers")
  if (
    not args.gpu_ids
    or len(set(args.gpu_ids)) != len(args.gpu_ids)
    or any(device < 0 for device in args.gpu_ids)
  ):
    parser.error("--gpu-ids must contain unique nonnegative integers")
  if not 1 <= args.workers <= min(8, len(args.gpu_ids)):
    parser.error("--workers must be between 1 and the number of GPU ids (maximum 8)")
  if args.start_seed < 0 or args.cases_per_seed <= 0:
    parser.error(
      "--start-seed must be nonnegative and --cases-per-seed must be positive"
    )
  try:
    validate_route(
      generator=args.generator,
      input_policy=args.input_policy,
      dtype=args.dtype,
      profile_version=args.profile_version,
      property_target=args.property_target,
      output_elements=args.output_element,
    )
  except ValueError as error:
    parser.error(str(error))
  return args


def configure_campaign(args):
  global BASE, SOURCE, CAMPAIGN, WORK, LOGS, PYTHON, TRITON_PYTHON, REVISION, TRITON
  global MAX_WORKERS, GPU_IDS, START_SEED, CASES_PER_SEED, PROFILE_VERSION
  global TENSOR_DTYPE, TENSOR_GENERATOR, TENSOR_INPUT_POLICY
  global TENSOR_PROPERTY_TARGET, TENSOR_OUTPUT_ELEMENTS
  global NEXT_SEED, STATUS, EVENTS, STOP, FOUND

  SOURCE = args.source_root.resolve()
  BASE = SOURCE.parent
  CAMPAIGN = args.campaign_dir.resolve()
  WORK = CAMPAIGN / "work"
  LOGS = CAMPAIGN / "logs"
  PYTHON = args.python.resolve()
  TRITON_PYTHON = args.triton_python.resolve()
  REVISION = args.revision
  TRITON = args.triton_build
  GPU_IDS = args.gpu_ids
  MAX_WORKERS = args.workers
  START_SEED = args.start_seed
  CASES_PER_SEED = args.cases_per_seed
  PROFILE_VERSION = args.profile_version
  TENSOR_DTYPE = args.dtype
  TENSOR_GENERATOR = args.generator
  TENSOR_INPUT_POLICY = args.input_policy
  TENSOR_PROPERTY_TARGET = args.property_target
  TENSOR_OUTPUT_ELEMENTS = tuple(args.output_element)
  NEXT_SEED = CAMPAIGN / "next_seed"
  STATUS = CAMPAIGN / "status.json"
  EVENTS = CAMPAIGN / "passes.jsonl"
  STOP = CAMPAIGN / "STOP.json"
  FOUND = CAMPAIGN / "FOUND.json"


def now() -> str:
  return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str | None:
  try:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
      for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()
  except OSError:
    return None


def _fsync_directory(path: Path) -> None:
  descriptor = os.open(path, os.O_RDONLY)
  try:
    os.fsync(descriptor)
  finally:
    os.close(descriptor)


def _atomic_write_text(path: Path, value: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
  try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
      stream.write(value)
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)
  finally:
    if os.path.exists(temporary):
      os.unlink(temporary)


def _atomic_write_json(path: Path, record: dict) -> bytes:
  payload = (
    json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
    + "\n"
  ).encode("utf-8")
  _atomic_write_text(path, payload.decode("utf-8"))
  return payload


def write_json(path: Path, record: dict) -> None:
  _atomic_write_json(path, record)


def append_jsonl(path: Path, record: dict) -> None:
  existing = path.read_text(encoding="utf-8") if path.is_file() else ""
  _atomic_write_text(
    path,
    existing + json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n",
  )


def update_status(state: str, **fields) -> None:
  current = {}
  if STATUS.is_file():
    try:
      current = json.loads(STATUS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      pass
  current.update(
    {
      "schema": "refractir.continuous-triton-campaign.v1",
      "state": state,
      "pid": os.getpid(),
      "updated_at": now(),
      "refractir_revision": REVISION,
      "triton_build": TRITON,
      "profile": f"tensor-{PROFILE_VERSION}",
      **route_metadata(),
      "gpu_ids": list(GPU_IDS),
      "cases_per_seed": CASES_PER_SEED,
      "worker_count": MAX_WORKERS,
    }
  )
  current.update(fields)
  write_json(STATUS, current)


def prepare_environment() -> dict[str, str]:
  env = os.environ.copy()
  python_paths = [
    str(BASE / "torch312"),
    "/usr/local/lib/python3.10/dist-packages",
    str(TRITON_PYTHON),
  ]
  if env.get("PYTHONPATH"):
    python_paths.append(env["PYTHONPATH"])
  env["PYTHONPATH"] = ":".join(python_paths)
  cuda_libraries = [
    "/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib",
    "/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib",
    "/usr/local/lib/python3.10/dist-packages/nvidia/cuda_cupti/lib",
    "/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib",
    "/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib",
    "/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib",
    "/usr/local/lib/python3.10/dist-packages/nvidia/curand/lib",
    "/usr/local/lib/python3.10/dist-packages/nvidia/cusolver/lib",
    "/usr/local/lib/python3.10/dist-packages/nvidia/cusparse/lib",
    "/usr/local/lib/python3.10/dist-packages/nvidia/nvjitlink/lib",
    "/usr/local/lib/python3.10/dist-packages/nvidia/nvtx/lib",
    "/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib",
    "/usr/local/lib/python3.10/dist-packages/nvidia/cufile/lib",
    "/usr/local/nvidia/lib",
    "/usr/local/nvidia/lib64",
  ]
  if env.get("LD_LIBRARY_PATH"):
    cuda_libraries.append(env["LD_LIBRARY_PATH"])
  env["LD_LIBRARY_PATH"] = ":".join(cuda_libraries)
  env.pop("CUDA_VISIBLE_DEVICES", None)
  env["REFRACTIR_SYMIRI"] = str(SOURCE / "symiri")
  env["REFRACTIR_SYMIRC"] = str(SOURCE / "symirc")
  env["REFRACTIR_CAMPAIGN_SOURCE"] = str(SOURCE)
  env["REFRACTIR_CAMPAIGN_PYTHON"] = str(PYTHON)
  env["REFRACTIR_REVISION"] = REVISION
  env.setdefault("REFRACTIR_WORKTREE_DIRTY", "unknown")
  env.setdefault("REFRACTIR_COMPILER_FLAGS", "unavailable")
  return env


def stop_campaign(record: dict, candidate: bool = False) -> None:
  record = {
    "schema": "refractir.continuous-triton-stop.v1",
    "stopped_at": now(),
    "candidate": candidate,
    "refractir_revision": REVISION,
    "triton_build": TRITON,
    **route_metadata(),
    **record,
  }
  write_json(STOP, record)
  if candidate:
    write_json(FOUND, record)
  update_status(
    "candidate_found" if candidate else "stopped",
    stop_reason=record.get("classification"),
    stop_file=str(FOUND if candidate else STOP),
    current_seed=record.get("seed"),
    current_case=record.get("case_id"),
  )
  logging.error("campaign stopped: %s", json.dumps(record, sort_keys=True))


def load_report(case_dir: Path, mode: str) -> dict:
  report_path = case_dir / f"result-{mode}.json"
  try:
    return json.loads(report_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    return {"status": "missing_report", "reason": str(error)}


def run_mode(
  case_dir: Path, mode: str, env: dict[str, str], device_id: int
) -> tuple[dict, int, float]:
  driver_dir = case_dir / "campaign-driver"
  driver_dir.mkdir(parents=True, exist_ok=True)
  stdout_path = driver_dir / f"{mode}.stdout.log"
  stderr_path = driver_dir / f"{mode}.stderr.log"
  command = [
    env.get("REFRACTIR_CAMPAIGN_PYTHON", str(PYTHON)),
    "-m",
    "test.lib.run_triton_tests",
    "--case",
    str(case_dir),
    "--mode",
    mode,
    "--device",
    str(device_id),
  ]
  started = time.monotonic()
  with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
    try:
      result = subprocess.run(
        command,
        cwd=Path(env.get("REFRACTIR_CAMPAIGN_SOURCE", str(SOURCE))),
        env=env,
        stdout=stdout,
        stderr=stderr,
        check=False,
      )
      returncode = result.returncode
    except OSError as error:
      stderr.write(str(error).encode("utf-8", errors="replace"))
      returncode = 127
  elapsed = time.monotonic() - started
  return load_report(case_dir, mode), returncode, elapsed


def stop_for_run(
  seed: int,
  case_id: str,
  case_dir: Path,
  mode: str,
  report: dict,
  returncode: int,
  elapsed: float,
) -> None:
  status = report.get("status", "unknown")
  environment_failure = status in {"environment_unavailable", "invalid_case"} or (
    status == "pass" and report.get("campaign_valid") is not True
  )
  if environment_failure:
    classification = "environment_or_case_setup"
    candidate = False
  elif mode == "interpret":
    classification = "refractir_generator_or_interpreter_candidate"
    candidate = status != "pass"
  elif status == "compile_failure":
    classification = "triton_compile_failure_candidate"
    candidate = True
  elif status in {"runtime_failure", "runtime_timeout"}:
    classification = "compiled_execution_failure_candidate"
    candidate = True
  elif status != "pass":
    classification = "compiled_output_mismatch_candidate"
    candidate = True
  else:
    classification = "runner_exit_without_valid_pass"
    candidate = False
  stop_campaign(
    {
      "classification": classification,
      "seed": seed,
      "case_id": case_id,
      "case_dir": str(case_dir),
      "mode": mode,
      "runner_exit_code": returncode,
      "elapsed_seconds": round(elapsed, 3),
      "report": report,
      "artifacts_retained": True,
      "source_sha256": sha256(case_dir / "program.sir"),
      "manifest_sha256": sha256(case_dir / "case.json"),
      "inputs_sha256": sha256(case_dir / "inputs.bin"),
      "expected_sha256": sha256(case_dir / "expected.bin"),
    },
    candidate=candidate,
  )


def run_case_job(job: tuple[int, str, str, dict[str, str], int]) -> dict:
  seed, case_id, case_path, env, device_id = job
  case_dir = Path(case_path)
  mode_elapsed = {}
  for mode in ("interpret", "compiled"):
    report, returncode, elapsed = run_mode(case_dir, mode, env, device_id)
    mode_elapsed[mode] = elapsed
    if (
      returncode != 0
      or report.get("status") != "pass"
      or report.get("campaign_valid") is not True
    ):
      return {
        "passed": False,
        "seed": seed,
        "case_id": case_id,
        "case_dir": str(case_dir),
        "mode": mode,
        "report": report,
        "returncode": returncode,
        "elapsed": elapsed,
      }
  event = {
    "schema": "refractir.continuous-triton-pass.v1",
    "passed_at": now(),
    "seed": seed,
    "case_id": case_id,
    **route_metadata(),
    "interpret_seconds": round(mode_elapsed["interpret"], 3),
    "compiled_seconds": round(mode_elapsed["compiled"], 3),
    "source_sha256": sha256(case_dir / "program.sir"),
    "manifest_sha256": sha256(case_dir / "case.json"),
    "inputs_sha256": sha256(case_dir / "inputs.bin"),
    "expected_sha256": sha256(case_dir / "expected.bin"),
    "compiled_actual_sha256": load_report(case_dir, "compiled").get("actual_sha256"),
    "compiled_status": load_report(case_dir, "compiled").get("status"),
    "campaign_valid": load_report(case_dir, "compiled").get("campaign_valid"),
  }
  return {
    "passed": True,
    "seed": seed,
    "case_id": case_id,
    "case_dir": str(case_dir),
    "event": event,
  }


def generate_seed(seed: int, env: dict[str, str]) -> tuple[Path | None, list[str]]:
  seed_dir = CAMPAIGN / f"seed-{seed}"
  if seed_dir.exists():
    metadata_path = seed_dir / ".campaign-seed.json"
    try:
      metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
      case_ids = metadata["case_ids"]
      if len(case_ids) != CASES_PER_SEED:
        raise ValueError("seed metadata has unexpected case count")
      expected_route = route_metadata()
      if metadata.get("route") != expected_route:
        raise ValueError("seed metadata belongs to a different generator route")
      return seed_dir, case_ids
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
      stop_campaign(
        {
          "classification": "resume_state_corrupt",
          "seed": seed,
          "seed_dir": str(seed_dir),
          "reason": str(error),
          "artifacts_retained": True,
        }
      )
      return None, []

  temporary = WORK / f"seed-{seed}.tmp"
  if temporary.exists():
    shutil.rmtree(temporary)
  temporary.mkdir(parents=True)
  stdout_path = LOGS / f"generate-seed-{seed}.stdout.log"
  stderr_path = LOGS / f"generate-seed-{seed}.stderr.log"
  command = build_generation_command(
    SOURCE / "rysmith",
    temporary,
    seed,
    profile_version=PROFILE_VERSION,
    dtype=TENSOR_DTYPE,
    generator=TENSOR_GENERATOR,
    input_policy=TENSOR_INPUT_POLICY,
    property_target=TENSOR_PROPERTY_TARGET,
    output_elements=TENSOR_OUTPUT_ELEMENTS,
    count=CASES_PER_SEED,
  )
  logging.info("generating seed %d", seed)
  with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
    result = subprocess.run(
      command,
      cwd=Path(env.get("REFRACTIR_CAMPAIGN_SOURCE", str(SOURCE))),
      env=env,
      stdout=stdout,
      stderr=stderr,
      check=False,
    )
  case_ids = sorted(
    child.name
    for child in temporary.iterdir()
    if child.is_dir() and (child / "case.json").is_file()
  )
  if result.returncode != 0 or len(case_ids) != CASES_PER_SEED:
    stop_campaign(
      {
        "classification": "generator_failure",
        "seed": seed,
        "generator_exit_code": result.returncode,
        "generated_case_count": len(case_ids),
        "expected_case_count": CASES_PER_SEED,
        "partial_generation_dir": str(temporary),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "artifacts_retained": True,
      }
    )
    return None, []
  write_json(
    temporary / ".campaign-seed.json",
    {
      "seed": seed,
      "case_ids": case_ids,
      "generated_at": now(),
      "route": route_metadata(),
      "generation_command": command,
    },
  )
  os.replace(temporary, seed_dir)
  stdout_path.unlink(missing_ok=True)
  stderr_path.unlink(missing_ok=True)
  return seed_dir, case_ids


def passed_case_ids(seed_dir: Path) -> set[str]:
  path = seed_dir / ".passed-cases"
  if not path.exists():
    return set()
  return {
    line.strip()
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
  }


def _read_pass_events(path: Path) -> list[dict]:
  if not path.is_file():
    return []
  events = []
  for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
      continue
    try:
      record = json.loads(line)
    except json.JSONDecodeError as error:
      raise ValueError(f"pass journal line {line_number} is invalid JSON") from error
    if not isinstance(record, dict):
      raise ValueError(f"pass journal line {line_number} is not an object")
    events.append(record)
  return events


def mark_passed(
  seed_dir: Path,
  case_id: str,
  event: dict,
  *,
  events_path: Path | None = None,
) -> None:
  """Durably upsert a pass event and marker after its summary is published."""
  events_path = Path(events_path or EVENTS)
  events = _read_pass_events(events_path)
  matching = [
    index
    for index, previous in enumerate(events)
    if previous.get("seed") == event.get("seed") and previous.get("case_id") == case_id
  ]
  if len(matching) > 1:
    raise ValueError("pass journal contains duplicate case identities")
  changed = False
  if matching:
    index = matching[0]
    previous = events[index]
    for key in ("source_sha256", "manifest_sha256", "inputs_sha256", "expected_sha256"):
      if previous.get(key) and event.get(key) and previous[key] != event[key]:
        raise ValueError(f"pass event identity changed for {key}")
    merged = {**previous, **event}
    if merged != previous:
      events[index] = merged
      changed = True
  else:
    events.append(event)
    changed = True
  if changed:
    payload = "".join(
      json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in events
    )
    _atomic_write_text(events_path, payload)

  marker = Path(seed_dir) / ".passed-cases"
  completed = set()
  if marker.is_file():
    completed = {
      line.strip()
      for line in marker.read_text(encoding="utf-8").splitlines()
      if line.strip()
    }
  if case_id not in completed:
    completed.add(case_id)
    _atomic_write_text(marker, "".join(f"{value}\n" for value in sorted(completed)))


def _canonical_digest(value: dict) -> str:
  encoded = json.dumps(
    value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _optional_json(path: Path) -> dict:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return {}
  return value if isinstance(value, dict) else {}


def _physical_layout(record: dict | None) -> dict | None:
  if not isinstance(record, dict):
    return None
  fields = (
    "shape",
    "element_strides",
    "element_offset",
    "storage_span",
    "axis_order",
  )
  return {name: record[name] for name in fields if name in record}


def build_compiler_summary(
  case_dir: Path | None,
  event: dict,
  *,
  source_revision: str,
  triton_build: str,
) -> dict:
  """Build a versioned observation record without inferring missing IR."""
  from test.lib import tensor_compiler_observation, tensor_coverage

  case_path = Path(case_dir) if case_dir is not None else None
  manifest = {}
  if case_path is not None and case_path.is_dir():
    manifest = _optional_json(case_path / "case.json")
  case_id = str(event.get("case_id") or (case_path.name if case_path else "unknown"))
  seed = event.get("seed")
  source_digest = sha256(case_path / "program.sir") if case_path else None
  manifest_digest = sha256(case_path / "case.json") if case_path else None
  input_digest = sha256(case_path / "inputs.bin") if case_path else None
  expected_digest = sha256(case_path / "expected.bin") if case_path else None
  source_digest = source_digest or event.get("source_sha256")
  manifest_digest = manifest_digest or event.get("manifest_sha256")
  input_digest = input_digest or event.get("inputs_sha256")
  expected_digest = expected_digest or event.get("expected_sha256")
  if case_path is not None and case_path.is_dir() and case_path.name != case_id:
    raise ValueError("case directory name differs from the pass event identity")
  for field, actual in (
    ("source_sha256", source_digest),
    ("manifest_sha256", manifest_digest),
    ("inputs_sha256", input_digest),
    ("expected_sha256", expected_digest),
  ):
    recorded = event.get(field)
    if recorded is not None and actual is not None and recorded != actual:
      raise ValueError(f"case artifacts differ from pass event {field}")
  stable_identity = {
    "seed": seed,
    "case_id": case_id,
    "source_sha256": source_digest,
    "manifest_sha256": manifest_digest,
    "inputs_sha256": input_digest,
    "expected_sha256": expected_digest,
    "refractir_revision": source_revision,
    "triton_build": triton_build,
  }
  identity_digest = _canonical_digest(stable_identity)

  grouped_paths: dict[str, dict[str, list[Path]]] = {}
  run_metadata: dict[str, dict] = {}
  if case_path is not None and case_path.is_dir():
    for run_file in sorted(case_path.rglob("run.json")):
      run_id = run_file.parent.relative_to(case_path).as_posix()
      grouped_paths.setdefault(run_id, {})
      run_metadata[run_id] = {
        "run": _optional_json(run_file),
        "environment": _optional_json(run_file.parent / "environment.json"),
      }
    for artifact in sorted(case_path.rglob("compiler-*.ir")):
      raw_stage = artifact.stem.removeprefix("compiler-").lower()
      stage = SUMMARY_STAGE_ALIASES.get(raw_stage)
      if stage is None:
        continue
      run_id = (
        "case-artifacts"
        if artifact.parent == case_path
        else artifact.parent.relative_to(case_path).as_posix()
      )
      grouped_paths.setdefault(run_id, {}).setdefault(stage, []).append(artifact)
  if not grouped_paths:
    grouped_paths["case-artifacts"] = {}

  run_observations = {}
  unavailable_stages = {stage: [] for stage in SUMMARY_STAGES}
  parser_limitations = []
  for run_id, paths_by_stage in sorted(grouped_paths.items()):
    stage_records = {}
    artifact_texts = {}
    raw_texts = {}
    for stage in SUMMARY_STAGES:
      paths = paths_by_stage.get(stage, [])
      if not paths:
        stage_records[stage] = {
          "status": "unavailable",
          "reason": "compiler stage artifact was not emitted",
        }
        unavailable_stages[stage].append(run_id)
        continue
      artifacts = []
      for path in paths:
        try:
          data = path.read_bytes()
          text = data.decode("utf-8")
        except (OSError, UnicodeDecodeError) as error:
          artifacts.append(
            {
              "path": path.relative_to(case_path).as_posix(),
              "raw_sha256": sha256(path),
              "read_error": str(error),
            }
          )
          continue
        artifacts.append(
          {
            "path": path.relative_to(case_path).as_posix(),
            "raw_sha256": hashlib.sha256(data).hexdigest(),
          }
        )
        if stage not in artifact_texts:
          artifact_texts[stage] = text
          raw_texts[stage] = text
      if not artifact_texts.get(stage):
        stage_records[stage] = {
          "status": "unavailable",
          "reason": "compiler stage artifact could not be decoded as UTF-8",
          "artifacts": artifacts,
        }
        unavailable_stages[stage].append(run_id)
        continue
      from test.lib.tensor_coverage import normalize_triton_ir

      try:
        normalized = normalize_triton_ir(stage, artifact_texts[stage])
      except (TypeError, ValueError) as error:
        normalized = None
        parser_limitations.append(f"{run_id}/{stage}: normalization failed: {error}")
      stage_records[stage] = {
        "status": "available",
        "raw_sha256": artifacts[0].get("raw_sha256"),
        "normalized_sha256": (
          hashlib.sha256(normalized.encode("utf-8")).hexdigest()
          if normalized is not None
          else None
        ),
        "artifacts": artifacts,
        "parser_status": "pending",
      }

    observations = None
    if artifact_texts:
      try:
        observations = tensor_compiler_observation.observe_compiler_ir(
          artifact_texts,
        )
      except (TypeError, ValueError, re.error) as error:
        parser_limitations.append(
          f"{run_id}: compiler observation parser failed: {error}"
        )
    if observations is None:
      for stage in artifact_texts:
        stage_records[stage]["parser_status"] = "unavailable"
        stage_records[stage]["parser_reason"] = (
          parser_limitations[-1]
          if parser_limitations
          else "compiler parser did not run"
        )
      observation_record = {
        "status": "unavailable" if not artifact_texts else "parser_error",
        "reason": "no compiler IR artifacts were emitted"
        if not artifact_texts
        else "compiler IR could not be parsed",
      }
    else:
      for stage, stage_observation in observations["stages"].items():
        stage_records[stage]["parser_status"] = "available"
        stage_records[stage]["observations"] = stage_observation
      observation_record = {"status": "available", **observations}
      parser_limitations.append(observations["interpretation_limit"])

    ttgir = raw_texts.get("ttgir")
    layout_encodings = []
    if ttgir:
      layout_encodings = [
        line.strip()
        for line in ttgir.splitlines()
        if re.match(r"^\s*#[A-Za-z0-9_.-]+\s*=\s*#(?:ttg|triton)\.", line)
      ]
    metadata = run_metadata.get(run_id, {})
    run_record = metadata.get("run", {})
    environment = metadata.get("environment", {})
    run_observations[run_id] = {
      "stages": stage_records,
      "observation": observation_record,
      "ttgir_layout_encodings": layout_encodings,
      "compiler_identity": {
        "compiler": run_record.get("compiler"),
        "target_arch": environment.get("target_arch"),
        "triton_source_revision": environment.get("triton_source_revision"),
        "selected_device": run_record.get("selected_device"),
        "launch": run_record.get("launch"),
        "grid": run_record.get("grid"),
      },
    }

  pass_dump_paths = []
  if case_path is not None and case_path.is_dir():
    pass_dump_paths = [
      path
      for path in sorted(case_path.rglob("*"))
      if path.is_file() and "pass" in path.name.lower() and "dump" in path.name.lower()
    ]
  pass_files = {}
  for path in pass_dump_paths:
    relative = path.relative_to(case_path).as_posix()
    record = {"status": "unavailable", "raw_sha256": None}
    try:
      data = path.read_bytes()
      record["raw_sha256"] = hashlib.sha256(data).hexdigest()
      # Each file is one capture boundary. Joining partial captures can invent
      # a transition between different compilations or different kernel runs.
      observed = tensor_compiler_observation.observe_pass_dumps(data.decode("utf-8"))
      record.update(status="available", **observed)
    except (OSError, UnicodeDecodeError, TypeError, ValueError) as error:
      record["reason"] = f"pass dump parser failed: {error}"
      parser_limitations.append(f"{relative}: {record['reason']}")
    pass_files[relative] = record
  available = sum(item["status"] == "available" for item in pass_files.values())
  pass_deltas = {
    "status": (
      "unavailable"
      if not available
      else "available"
      if available == len(pass_files)
      else "partial"
    ),
    "artifacts": list(pass_files),
    "files": pass_files,
    "compilation_count": sum(
      item.get("compilation_count", 0) for item in pass_files.values()
    ),
  }
  if not pass_files:
    pass_deltas["reason"] = "compiler before/after pass dumps were not saved"
  elif not available:
    pass_deltas["reason"] = "no pass dump could be parsed independently"

  coverage = None
  if case_path is not None and case_path.is_dir():
    try:
      coverage = tensor_coverage.case_coverage(case_path)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
      parser_limitations.append(f"tensor coverage parser failed: {error}")

  inputs = manifest.get("inputs", []) if isinstance(manifest, dict) else []
  output = manifest.get("output", {}) if isinstance(manifest, dict) else {}
  settings = {
    "grid": manifest.get("grid") if isinstance(manifest, dict) else None,
    "launch": manifest.get("launch") if isinstance(manifest, dict) else None,
    "physical_layouts": {
      "inputs": [
        {"name": record.get("name"), **(_physical_layout(record) or {})}
        for record in inputs
        if isinstance(record, dict)
      ],
      "output": _physical_layout(output),
    },
  }
  features = {}
  for run_id, run in sorted(run_observations.items()):
    observed_stages = run["observation"].get("stages", {})
    features[run_id] = {
      "available_stages": sorted(
        stage for stage, item in run["stages"].items() if item["status"] == "available"
      ),
      "operations": {
        stage: item.get("operation_counts", {})
        for stage, item in sorted(observed_stages.items())
      },
      "loop_counts": {
        stage: item.get("loop_construct_count")
        for stage, item in sorted(observed_stages.items())
      },
      "conversion_shapes": sorted(
        json.dumps(
          [
            conversion.get("source_layout"),
            conversion.get("target_layout"),
            conversion.get("layout_changed"),
          ],
          sort_keys=True,
        )
        for conversion in run["observation"].get("layout_conversions", [])
      ),
      "ptx_communication": {
        stage: {
          key: item.get(key)
          for key in ("warp_shuffle_count", "shared_memory_transfer_count")
          if key in item
        }
        for stage, item in sorted(observed_stages.items())
        if "warp_shuffle_count" in item or "shared_memory_transfer_count" in item
      },
    }
  behavior_bucket = _canonical_digest(features)
  missing_case_artifacts = []
  if case_path is None or not case_path.is_dir():
    missing_case_artifacts.extend(
      ("case.json", "program.sir", "inputs.bin", "expected.bin")
    )
  else:
    missing_case_artifacts.extend(
      name
      for name in ("case.json", "program.sir", "inputs.bin", "expected.bin")
      if not (case_path / name).is_file()
    )
  return {
    "schema": CASE_SUMMARY_SCHEMA,
    "status": "available"
    if not missing_case_artifacts
    else "historical_artifacts_unavailable",
    "missing_case_artifacts": missing_case_artifacts,
    "case_identity": {
      **stable_identity,
      "identity_sha256": identity_digest,
    },
    "source_identity": {
      "sha256": source_digest,
      "signature": coverage.get("source_signature") if coverage else None,
    },
    "input_identity": {"sha256": input_digest},
    "expected_identity": {"sha256": expected_digest},
    "compiler_identity": {
      "refractir_revision": source_revision,
      "triton_build": triton_build,
      "runs": {
        run_id: value["compiler_identity"]
        for run_id, value in sorted(run_observations.items())
      },
    },
    "settings": settings,
    "coverage_signatures": {
      "compiled_signatures": coverage.get("compiled_signatures", {})
      if coverage
      else {},
      "compiled_runs": coverage.get("compiled_runs", {}) if coverage else {},
      "source_signature": coverage.get("source_signature") if coverage else None,
    },
    "compiler_observations": {
      "source_attribution": {
        "status": "unavailable",
        "reason": "compiler locations name emitted Python; no SIR source map is recorded",
      },
      "runs": run_observations,
      "unavailable_stages": {
        stage: sorted(run_ids)
        for stage, run_ids in unavailable_stages.items()
        if run_ids
      },
      "parser_limitations": sorted(set(parser_limitations)),
      "pass_deltas": pass_deltas,
    },
    "behavior_bucket": behavior_bucket,
  }


def _read_archive_index(path: Path) -> dict:
  if not path.is_file():
    return {
      "schema": "refractir.tensor-case-representatives.v1",
      "quota": REPRESENTATIVE_QUOTA,
      "buckets": {},
    }
  index = _optional_json(path)
  if (
    index.get("schema") != "refractir.tensor-case-representatives.v1"
    or index.get("quota") != REPRESENTATIVE_QUOTA
    or not isinstance(index.get("buckets"), dict)
  ):
    raise ValueError("representative archive index is invalid")
  return index


def _verify_archived_case(path: Path, summary: dict) -> bool:
  identity = summary["case_identity"]
  return (
    sha256(path / "program.sir") == identity.get("source_sha256")
    and sha256(path / "case.json") == identity.get("manifest_sha256")
    and sha256(path / "inputs.bin") == identity.get("inputs_sha256")
  )


def _archive_representative(
  case_dir: Path,
  campaign_dir: Path,
  summary: dict,
  archive_index: dict,
) -> dict:
  bucket = summary["behavior_bucket"]
  existing = archive_index["buckets"].get(bucket)
  if existing is not None:
    existing_path = Path(existing.get("path", ""))
    destination = campaign_dir / existing_path
    if destination.is_dir():
      if sha256(destination / "program.sir") != existing.get("source_sha256") or sha256(
        destination / "case.json"
      ) != existing.get("manifest_sha256"):
        raise ValueError("representative archive case differs from its index")
      return {
        "status": "already_represented",
        "bucket": bucket,
        "path": existing_path.as_posix(),
      }
    if not case_dir.is_dir():
      return {"status": "missing_representative", "bucket": bucket, "path": None}
    relative = (
      Path("representatives")
      / bucket[:2]
      / f"{summary['case_identity']['seed']}-{summary['case_identity']['case_id']}"
    )
    destination = campaign_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
      shutil.rmtree(temporary)
    shutil.copytree(case_dir, temporary)
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)
    archive_index["buckets"][bucket] = {
      **existing,
      "case_id": summary["case_identity"]["case_id"],
      "seed": summary["case_identity"]["seed"],
      "path": relative.as_posix(),
      "source_sha256": summary["case_identity"]["source_sha256"],
      "manifest_sha256": summary["case_identity"]["manifest_sha256"],
      "case_summary_sha256": sha256(
        campaign_dir
        / "summaries"
        / str(summary["case_identity"]["seed"])
        / f"{summary['case_identity']['case_id']}.json"
      ),
    }
    _atomic_write_json(campaign_dir / "representatives" / "index.json", archive_index)
    return {"status": "archived", "bucket": bucket, "path": relative.as_posix()}
  archive_status = summary["representative_archive"]["status"]
  if archive_status != "selected":
    return {"status": archive_status, "bucket": bucket, "path": None}
  if len(archive_index["buckets"]) >= REPRESENTATIVE_QUOTA:
    return {"status": "quota_exhausted", "bucket": bucket, "path": None}
  relative = (
    Path("representatives")
    / bucket[:2]
    / f"{summary['case_identity']['seed']}-{summary['case_identity']['case_id']}"
  )
  destination = campaign_dir / relative
  destination.parent.mkdir(parents=True, exist_ok=True)
  if destination.exists():
    if not _verify_archived_case(destination, summary):
      raise ValueError(
        "existing representative directory has a different case identity"
      )
  else:
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
      shutil.rmtree(temporary)
    shutil.copytree(case_dir, temporary)
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)
  archive_index["buckets"][bucket] = {
    "case_id": summary["case_identity"]["case_id"],
    "seed": summary["case_identity"]["seed"],
    "path": relative.as_posix(),
    "source_sha256": summary["case_identity"]["source_sha256"],
    "manifest_sha256": summary["case_identity"]["manifest_sha256"],
    "case_summary_sha256": sha256(
      campaign_dir
      / "summaries"
      / str(summary["case_identity"]["seed"])
      / f"{summary['case_identity']['case_id']}.json"
    ),
  }
  _atomic_write_json(campaign_dir / "representatives" / "index.json", archive_index)
  return {"status": "archived", "bucket": bucket, "path": relative.as_posix()}


def persist_case_summary(
  case_dir: Path | None,
  campaign_dir: Path,
  event: dict,
  *,
  source_revision: str,
  triton_build: str,
) -> dict:
  """Publish a durable summary and a bounded representative before success."""
  campaign_dir = Path(campaign_dir)
  seed = event.get("seed")
  case_id = str(
    event.get("case_id") or (Path(case_dir).name if case_dir else "unknown")
  )
  summary_path = campaign_dir / "summaries" / str(seed) / f"{case_id}.json"
  archive_index_path = campaign_dir / "representatives" / "index.json"
  archive_index = _read_archive_index(archive_index_path)
  if summary_path.is_file():
    summary = _optional_json(summary_path)
    current_identity = build_compiler_summary(
      case_dir,
      event,
      source_revision=source_revision,
      triton_build=triton_build,
    )["case_identity"]
    if (
      summary.get("schema") != CASE_SUMMARY_SCHEMA
      or summary.get("case_identity", {}).get("case_id") != case_id
      or summary.get("case_identity", {}).get("seed") != seed
      or summary.get("case_identity", {}).get("identity_sha256")
      != current_identity.get("identity_sha256")
    ):
      raise ValueError("existing case summary has a different identity")
  else:
    summary = build_compiler_summary(
      case_dir,
      event,
      source_revision=source_revision,
      triton_build=triton_build,
    )
    bucket = summary["behavior_bucket"]
    if bucket in archive_index["buckets"]:
      archive_status = "already_represented"
    elif case_dir is None or not Path(case_dir).is_dir():
      archive_status = "unavailable"
    elif len(archive_index["buckets"]) >= REPRESENTATIVE_QUOTA:
      archive_status = "quota_exhausted"
    else:
      archive_status = "selected"
    summary["representative_archive"] = {
      "status": archive_status,
      "quota": REPRESENTATIVE_QUOTA,
    }
    _atomic_write_json(summary_path, summary)
  summary_bytes = summary_path.read_bytes()
  summary_digest = hashlib.sha256(summary_bytes).hexdigest()
  expected_digest = event.get("case_summary_sha256")
  if expected_digest is not None and expected_digest != summary_digest:
    raise ValueError("persisted case summary differs from the pass event digest")
  if case_dir is None or not Path(case_dir).is_dir():
    existing = archive_index["buckets"].get(summary["behavior_bucket"])
    archive = (
      {
        "status": "already_represented",
        "bucket": summary["behavior_bucket"],
        "path": existing.get("path"),
      }
      if isinstance(existing, dict)
      else {
        "status": "unavailable",
        "bucket": summary["behavior_bucket"],
        "path": None,
      }
    )
  else:
    archive = _archive_representative(
      Path(case_dir), campaign_dir, summary, archive_index
    )
  return {
    "case_summary_path": str(summary_path),
    "case_summary_sha256": summary_digest,
    "behavior_bucket": summary["behavior_bucket"],
    "representative_archive": archive,
  }


def finalize_passed_case(
  seed_dir: Path,
  case_id: str,
  case_dir: Path,
  event: dict,
  *,
  campaign_dir: Path,
  source_revision: str,
  triton_build: str,
) -> dict:
  """Persist observations, record the pass, then remove disposable artifacts."""
  if not Path(case_dir).is_dir():
    raise ValueError("successful case artifacts are missing before summary publication")
  missing = [
    name
    for name in ("case.json", "program.sir", "inputs.bin", "expected.bin")
    if not (Path(case_dir) / name).is_file()
  ]
  if missing:
    raise ValueError(
      "successful case is missing required artifacts: " + ", ".join(missing)
    )
  persisted = persist_case_summary(
    case_dir,
    campaign_dir,
    event,
    source_revision=source_revision,
    triton_build=triton_build,
  )
  recorded = {**event, **persisted}
  mark_passed(
    seed_dir,
    case_id,
    recorded,
    events_path=Path(campaign_dir) / "passes.jsonl",
  )
  if Path(case_dir).exists():
    shutil.rmtree(case_dir)
  return recorded


def recover_passed_case(
  seed_dir: Path,
  case_id: str,
  case_dir: Path | None,
  *,
  campaign_dir: Path,
  source_revision: str,
  triton_build: str,
) -> dict:
  """Repair a legacy pass record only after its summary is durable."""
  campaign_dir = Path(campaign_dir)
  events_path = campaign_dir / "passes.jsonl"
  matching = [
    event
    for event in _read_pass_events(events_path)
    if event.get("seed") == int(Path(seed_dir).name.removeprefix("seed-"))
    and event.get("case_id") == case_id
  ]
  if len(matching) != 1:
    raise ValueError("passed-case marker does not have one recoverable pass event")
  event = matching[0]
  persisted = persist_case_summary(
    case_dir if case_dir is not None and Path(case_dir).is_dir() else None,
    campaign_dir,
    event,
    source_revision=source_revision,
    triton_build=triton_build,
  )
  if (
    event.get("case_summary_sha256")
    and event["case_summary_sha256"] != persisted["case_summary_sha256"]
  ):
    raise ValueError("pass event references a different case summary")
  recorded = {**event, **persisted}
  mark_passed(seed_dir, case_id, recorded, events_path=events_path)
  if case_dir is not None and Path(case_dir).exists():
    shutil.rmtree(case_dir)
  return recorded


def finalize_parent_result(
  seed_dir: Path,
  result: dict,
  *,
  campaign_dir: Path,
  source_revision: str,
  triton_build: str,
) -> dict:
  """Serialize worker completion evidence in the campaign's parent process."""
  if result.get("passed") is not True:
    return result
  event = finalize_passed_case(
    seed_dir,
    result["case_id"],
    Path(result["case_dir"]),
    result["event"],
    campaign_dir=campaign_dir,
    source_revision=source_revision,
    triton_build=triton_build,
  )
  return {**result, "event": event}


def main(argv=None) -> int:
  args = parse_arguments(argv)
  configure_campaign(args)
  CAMPAIGN.mkdir(parents=True, exist_ok=True)
  WORK.mkdir(parents=True, exist_ok=True)
  LOGS.mkdir(parents=True, exist_ok=True)
  logging.basicConfig(
    filename=CAMPAIGN / "campaign.log",
    level=logging.INFO,
    format="%(asctime)sZ %(levelname)s %(message)s",
  )
  lock = (CAMPAIGN / "driver.lock").open("a+")
  try:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
  except BlockingIOError:
    logging.error("another campaign driver holds the lock")
    return 2
  if FOUND.exists() or STOP.exists():
    logging.error("stop manifest already exists; refusing to resume automatically")
    return 0
  config_path = CAMPAIGN / "campaign.json"
  config = campaign_metadata()
  if config_path.is_file():
    try:
      saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
      logging.error("campaign config is unreadable: %s", error)
      return 2
    if saved_config != config:
      logging.error("campaign configuration changed; refusing to mix route artifacts")
      return 2
  else:
    write_json(config_path, config)
  if not PYTHON.is_file() or not (SOURCE / "rysmith").is_file():
    stop_campaign(
      {
        "classification": "runtime_setup_missing",
        "python": str(PYTHON),
        "source": str(SOURCE),
      }
    )
    return 2

  env = prepare_environment()
  next_seed = START_SEED
  if NEXT_SEED.is_file():
    try:
      next_seed = int(NEXT_SEED.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
      stop_campaign({"classification": "invalid_next_seed_checkpoint"})
      return 2
  old_status = {}
  if STATUS.is_file():
    try:
      old_status = json.loads(STATUS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      pass
  passed_count = int(old_status.get("passed_cases", 0))
  passed_seed_count = int(old_status.get("passed_seeds", 0))
  _atomic_write_text(NEXT_SEED, f"{next_seed}\n")
  for temporary in WORK.glob("seed-*.tmp"):
    shutil.rmtree(temporary, ignore_errors=True)
  for stale in CAMPAIGN.glob("seed-*"):
    if stale.is_dir() and stale.name != f"seed-{next_seed}":
      try:
        stale_seed = int(stale.name.removeprefix("seed-"))
      except ValueError:
        continue
      if stale_seed < next_seed:
        try:
          seed_metadata = json.loads(
            (stale / ".campaign-seed.json").read_text(encoding="utf-8")
          )
          stale_case_ids = seed_metadata["case_ids"]
          if (
            not isinstance(stale_case_ids, list)
            or seed_metadata.get("route") != route_metadata()
          ):
            raise ValueError(
              "stale seed metadata is incomplete or belongs to another route"
            )
        except (
          OSError,
          KeyError,
          TypeError,
          json.JSONDecodeError,
          ValueError,
        ) as error:
          stop_campaign(
            {
              "classification": "stale_seed_summary_recovery_failed",
              "seed": stale_seed,
              "seed_dir": str(stale),
              "reason": str(error),
              "artifacts_retained": True,
            }
          )
          return 2
        stale_completed = passed_case_ids(stale)
        stale_event_case_ids = {
          event.get("case_id")
          for event in _read_pass_events(EVENTS)
          if event.get("seed") == stale_seed
        }
        missing_passes = sorted(
          set(stale_case_ids) - stale_completed - stale_event_case_ids
        )
        if missing_passes:
          stop_campaign(
            {
              "classification": "stale_seed_has_unrecorded_cases",
              "seed": stale_seed,
              "case_ids": missing_passes,
              "seed_dir": str(stale),
              "artifacts_retained": True,
            }
          )
          return 2
        for stale_case_id in stale_case_ids:
          try:
            recover_passed_case(
              stale,
              stale_case_id,
              stale / stale_case_id,
              campaign_dir=CAMPAIGN,
              source_revision=REVISION,
              triton_build=TRITON,
            )
          except (OSError, ValueError, TypeError) as error:
            stop_campaign(
              {
                "classification": "stale_seed_summary_recovery_failed",
                "seed": stale_seed,
                "case_id": stale_case_id,
                "seed_dir": str(stale),
                "reason": str(error),
                "artifacts_retained": True,
              }
            )
            return 2
        shutil.rmtree(stale)

  update_status(
    "running",
    current_seed=next_seed,
    current_case=None,
    passed_cases=passed_count,
    passed_seeds=passed_seed_count,
    started_at=old_status.get("started_at", now()),
    next_seed=next_seed,
    worker_count=MAX_WORKERS,
  )
  logging.info("campaign started at seed %d with %d workers", next_seed, MAX_WORKERS)

  while True:
    if shutil.disk_usage(CAMPAIGN).free < MIN_FREE_BYTES:
      stop_campaign(
        {
          "classification": "storage_guard",
          "free_bytes": shutil.disk_usage(CAMPAIGN).free,
          "minimum_free_bytes": MIN_FREE_BYTES,
          "next_seed": next_seed,
        }
      )
      return 2

    seed_dir, case_ids = generate_seed(next_seed, env)
    if seed_dir is None:
      return 1
    completed = passed_case_ids(seed_dir)
    completed_events = {
      event.get("case_id")
      for event in _read_pass_events(EVENTS)
      if event.get("seed") == next_seed
    }
    pending_case_ids = []
    for case_id in case_ids:
      case_dir = seed_dir / case_id
      if case_id in completed or case_id in completed_events:
        try:
          recover_passed_case(
            seed_dir,
            case_id,
            case_dir,
            campaign_dir=CAMPAIGN,
            source_revision=REVISION,
            triton_build=TRITON,
          )
        except (OSError, ValueError, TypeError) as error:
          stop_campaign(
            {
              "classification": "resume_case_summary_recovery_failed",
              "seed": next_seed,
              "case_id": case_id,
              "seed_dir": str(seed_dir),
              "reason": str(error),
              "artifacts_retained": True,
            }
          )
          return 2
        continue
      if not case_dir.is_dir():
        stop_campaign(
          {
            "classification": "resume_case_missing_without_pass_record",
            "seed": next_seed,
            "case_id": case_id,
            "seed_dir": str(seed_dir),
          }
        )
        return 2
      pending_case_ids.append(case_id)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
      for batch_start in range(0, len(pending_case_ids), MAX_WORKERS):
        batch_case_ids = pending_case_ids[batch_start : batch_start + MAX_WORKERS]
        update_status(
          "running",
          current_seed=next_seed,
          current_case=batch_case_ids[0],
          active_case_ids=batch_case_ids,
          passed_cases=passed_count,
          passed_seeds=passed_seed_count,
          next_seed=next_seed,
          worker_count=MAX_WORKERS,
        )
        logging.info(
          "running seed %d cases %s with %d workers",
          next_seed,
          ",".join(batch_case_ids),
          MAX_WORKERS,
        )
        jobs = [
          (
            next_seed,
            case_id,
            str(seed_dir / case_id),
            env,
            GPU_IDS[slot % MAX_WORKERS],
          )
          for slot, case_id in enumerate(batch_case_ids)
        ]
        results = list(executor.map(run_case_job, jobs))
        failures = []
        for result in results:
          if not result["passed"]:
            failures.append(result)
            continue
          try:
            finalized = finalize_parent_result(
              seed_dir,
              result,
              campaign_dir=CAMPAIGN,
              source_revision=REVISION,
              triton_build=TRITON,
            )
            event = finalized["event"]
          except (OSError, ValueError, TypeError) as error:
            stop_campaign(
              {
                "classification": "case_summary_persistence_failed",
                "seed": result["seed"],
                "case_id": result["case_id"],
                "case_dir": result["case_dir"],
                "reason": str(error),
                "artifacts_retained": Path(result["case_dir"]).exists(),
              }
            )
            return 2
          passed_count += 1
          update_status(
            "running",
            current_seed=next_seed,
            current_case=result["case_id"],
            active_case_ids=batch_case_ids,
            passed_cases=passed_count,
            passed_seeds=passed_seed_count,
            next_seed=next_seed,
            last_passed_case=result["case_id"],
            last_passed_at=event["passed_at"],
            worker_count=MAX_WORKERS,
          )
          logging.info(
            "PASS seed %d case %s total=%d interpret=%.3fs compiled=%.3fs",
            next_seed,
            result["case_id"],
            passed_count,
            event["interpret_seconds"],
            event["compiled_seconds"],
          )
          if shutil.disk_usage(CAMPAIGN).free < MIN_FREE_BYTES:
            stop_campaign(
              {
                "classification": "storage_guard",
                "free_bytes": shutil.disk_usage(CAMPAIGN).free,
                "minimum_free_bytes": MIN_FREE_BYTES,
                "next_seed": next_seed,
              }
            )
            return 2

        if failures:
          failure = failures[0]
          stop_for_run(
            failure["seed"],
            failure["case_id"],
            Path(failure["case_dir"]),
            failure["mode"],
            failure["report"],
            failure["returncode"],
            failure["elapsed"],
          )
          return 1

    passed_seed_count += 1
    next_seed += 1
    _atomic_write_text(NEXT_SEED, f"{next_seed}\n")
    update_status(
      "running",
      current_seed=next_seed,
      current_case=None,
      passed_cases=passed_count,
      passed_seeds=passed_seed_count,
      next_seed=next_seed,
      last_completed_seed=next_seed - 1,
    )
    shutil.rmtree(seed_dir, ignore_errors=True)
    logging.info("completed seed %d; next seed %d", next_seed - 1, next_seed)


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except BaseException as error:
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
      raise
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    write_json(
      STOP,
      {
        "schema": "refractir.continuous-triton-stop.v1",
        "stopped_at": now(),
        "candidate": False,
        "classification": "driver_exception",
        "error": repr(error),
        "refractir_revision": REVISION,
        "triton_build": TRITON,
      },
    )
    raise
