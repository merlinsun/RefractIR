"""Bounded runner for the frozen boundary-input sensitivity study."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
import re
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from test.lib import tensor_boundary_sensitivity as study

DEFAULT_TARGET_COUNT = 256
DEFAULT_START_SEED = 2027
DEFAULT_WORKERS = 4
DEFAULT_PROPOSAL_TIMEOUT_S = 10
DEFAULT_REFERENCE_TIMEOUT_S = 10
DEFAULT_GPU_TIMEOUT_S = 120
DEFAULT_COMPILER_FLAGS = "unavailable"
CONTROL_CATALOGUE_SCHEMA = "refractir.tensor-boundary-control-catalogue.v1"
SOURCE_ROOT = Path(__file__).resolve().parents[2]
RUN_SCHEMA = "refractir.tensor-boundary-sensitivity-run.v1"
_WORKER_ENVIRONMENT = None


def load_or_create_corpus(
  path,
  *,
  config,
  base_revision,
  target_count,
  candidates_factory,
):
  """Load the immutable corpus, or create it exactly once from the schedule."""
  corpus_path = Path(path)
  sidecar_path = Path(f"{corpus_path}.sha256")
  if corpus_path.exists() or sidecar_path.exists():
    corpus = study.load_corpus(corpus_path)
    if corpus.get("config") != config:
      raise ValueError("frozen corpus configuration differs from requested config")
    if corpus.get("base_revision") != base_revision:
      raise ValueError("frozen corpus base revision differs from requested revision")
    if corpus.get("target_count") != target_count:
      raise ValueError("frozen corpus target count differs from requested count")
    return corpus

  candidates = candidates_factory()
  corpus = study.freeze_corpus(
    candidates,
    config=config,
    base_revision=base_revision,
    target_count=target_count,
  )
  study.save_corpus(corpus_path, corpus)
  return corpus


def _canonical_bytes(value):
  return json.dumps(
    value,
    ensure_ascii=False,
    allow_nan=False,
    sort_keys=True,
    separators=(",", ":"),
  ).encode("utf-8")


def _sha256(data):
  return hashlib.sha256(data).hexdigest()


def _atomic_json(path, value):
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  payload = (
    json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
    + "\n"
  ).encode("utf-8")
  temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
  try:
    with temporary.open("xb") as output:
      output.write(payload)
      output.flush()
      os.fsync(output.fileno())
    os.replace(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
  finally:
    temporary.unlink(missing_ok=True)


def load_or_create_source_snapshot(path, source_root):
  """Bind output to all source files and refuse a changed resume snapshot."""
  snapshot_path = Path(path)
  current = study.source_snapshot_manifest(source_root)
  if snapshot_path.exists():
    try:
      saved = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
      raise ValueError(f"cannot read source snapshot: {error}") from error
    saved_payload = {
      key: value for key, value in saved.items() if key != "snapshot_sha256"
    }
    if saved.get("snapshot_sha256") != _sha256(_canonical_bytes(saved_payload)):
      raise ValueError("source snapshot digest is invalid")
    if saved != current:
      raise ValueError("source snapshot changed since this study was frozen")
    return saved
  _atomic_json(snapshot_path, current)
  return current


def _control_catalogue_payload(corpus):
  kernels = []
  for kernel in corpus["kernels"]:
    kernels.append(
      {
        "kernel_id": kernel["kernel_id"],
        "source_sha256": kernel["source_sha256"],
        "controls": study.build_control_catalogue(kernel["source"]),
      }
    )
  return {
    "schema": CONTROL_CATALOGUE_SCHEMA,
    "corpus_sha256": corpus["corpus_sha256"],
    "kernels": kernels,
  }


def load_or_create_control_catalogue(path, corpus):
  """Persist the complete control set before either input arm is evaluated."""
  study.validate_frozen_corpus(corpus)
  catalogue_path = Path(path)
  if catalogue_path.exists():
    try:
      catalogue = json.loads(catalogue_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
      raise ValueError(f"cannot read frozen control catalogue: {error}") from error
    payload = {
      key: value for key, value in catalogue.items() if key != "catalogue_sha256"
    }
    digest = catalogue.get("catalogue_sha256")
    if (
      catalogue.get("schema") != CONTROL_CATALOGUE_SCHEMA
      or catalogue.get("corpus_sha256") != corpus["corpus_sha256"]
      or digest != _sha256(_canonical_bytes(payload))
    ):
      raise ValueError(
        "frozen control catalogue is invalid or belongs to another corpus"
      )
    kernels = catalogue.get("kernels")
    expected = {
      kernel["kernel_id"]: kernel["source_sha256"] for kernel in corpus["kernels"]
    }
    if not isinstance(kernels, list) or len(kernels) != len(expected):
      raise ValueError("frozen control catalogue kernel set is incomplete")
    observed = {}
    for item in kernels:
      if not isinstance(item, dict):
        raise ValueError("frozen control catalogue has an invalid kernel record")
      kernel_id = item.get("kernel_id")
      if (
        kernel_id not in expected
        or item.get("source_sha256") != expected[kernel_id]
        or not isinstance(item.get("controls"), dict)
        or set(item["controls"]) != set(study.CONTROL_IDS)
        or kernel_id in observed
      ):
        raise ValueError("frozen control catalogue differs from the corpus")
      observed[kernel_id] = item["source_sha256"]
    if observed != expected:
      raise ValueError("frozen control catalogue kernel identities changed")
    return catalogue

  payload = _control_catalogue_payload(corpus)
  catalogue = {
    **payload,
    "catalogue_sha256": _sha256(_canonical_bytes(payload)),
  }
  _atomic_json(catalogue_path, catalogue)
  return catalogue


def frozen_config(source_snapshot_sha256):
  return {
    "profile": "tensor-v2",
    "dtype": "f32",
    "input_domain": [-1, 2],
    "proposals_per_arm": 4,
    "comparison": {"mode": "numeric", "atol": [2, 100000], "rtol": [2, 1000]},
    "launch": {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False},
    "physical_layouts": {
      "policy": "contiguous-row-major-v1",
      "buffer_offsets": 0,
      "grid": [1, 1, 1],
    },
    "source_snapshot_sha256": source_snapshot_sha256,
  }


def command_manifest(
  *,
  source_root,
  generator_api,
  rysmith,
  symiri,
  symirc,
  compiler_flags,
  candidate_timeout_s,
  template_timeout_s,
  reference_timeout_s,
  gpu_timeout_s,
):
  """Describe the fixed invocations and command templates for every study arm."""
  source_root = str(Path(source_root).resolve())
  generator_api = str(Path(generator_api).resolve())
  rysmith = str(Path(rysmith).resolve())
  symiri = str(Path(symiri).resolve())
  symirc = str(Path(symirc).resolve())

  from test.tensor.campaigns.continuous_triton_campaign import build_generation_command

  template_argv = build_generation_command(
    rysmith,
    "<generation-root>",
    "<candidate-seed>",
    profile_version="v2",
    dtype="f32",
    generator="compositional",
    input_policy="ordinary",
    count=1,
  )
  template_argv.extend(["--max-retries", "1"])
  reference_argv = [
    "--tensor-profile-version",
    "v2",
    "--tensor-reference-stability",
    "--entry",
    "kernel",
    "--tensor-input",
    "<inputs.bin>",
    "--tensor-output",
    "<result.bin>",
    "<program.sir>",
  ]
  worker_argv = [
    sys.executable,
    "-m",
    "test.lib.triton_case_worker",
    "--case",
    "<case-directory>",
    "--artifacts-dir",
    "<case-directory>/runs/compiled",
    "--mode",
    "compiled",
  ]
  payload = {
    "schema": "refractir.tensor-boundary-sensitivity-commands.v1",
    "candidate_generation": {
      "argv": [generator_api, "--generate", "<seed>"],
      "cwd": source_root,
      "timeout_s": candidate_timeout_s,
    },
    "frontend_check": {
      "argv": [symiri, "--check", "--tensor-profile-version", "v2", "<program.sir>"],
      "cwd": source_root,
      "timeout_s": 10,
    },
    "template_generation": {
      "argv": template_argv,
      "cwd": source_root,
      "timeout_s": template_timeout_s,
      "frozen_launch_override": frozen_config("<source-snapshot-sha256>")["launch"],
      "warp_probe_metadata": "not_probed.num_warps follows the frozen launch",
    },
    "legacy_sampling": {
      "implementation": "test.lib.tensor_boundary_sensitivity.legacy_cycle_buffers",
      "input_domain": [-1, 2],
      "phase_argument": "proposal_index",
    },
    "boundary_sampling": {
      "argv": [generator_api, "<program.sir>", "<sampler-seed>"],
      "timeout_s": DEFAULT_PROPOSAL_TIMEOUT_S,
      "sampler_version": study.BOUNDARY_SAMPLER_VERSION,
    },
    "reference_replay": {
      "executable": symiri,
      "argv": [symiri, *reference_argv],
      "cwd": source_root,
      "timeout_s": reference_timeout_s,
      "reference_stability": True,
    },
    "control_evaluation": {
      "checker_argv": [
        symiri,
        "--check",
        "--tensor-profile-version",
        "v2",
        "<control.sir>",
      ],
      "checker_timeout_s": 10,
      "reference_argv": [symiri, *reference_argv],
      "comparison": frozen_config("<source-snapshot-sha256>")["comparison"],
    },
    "symir_emission": {
      "argv": [
        symirc,
        "--target",
        "triton",
        "--entry",
        "<entry>",
        "--tensor-softmax-lowering",
        "<softmax-lowering>",
        "-o",
        "<temporary-kernel.py>",
        "<program.sir>",
      ],
      "cwd": source_root,
      "compiler_flags": compiler_flags,
    },
    "compiled_execution": {
      "runner": "test.lib.run_triton_tests.run_case",
      "mode": "compiled",
      "worker_argv": worker_argv,
      "cwd": source_root,
      "symiri": symiri,
      "symirc": symirc,
      "compiler_flags": compiler_flags,
      "device_visibility": "CUDA_VISIBLE_DEVICES=<physical-device-id>",
      "cache_policy": "isolated-per-case",
      "timeout_s": gpu_timeout_s,
    },
  }
  payload["manifest_sha256"] = _sha256(_canonical_bytes(payload))
  return payload


def _load_or_create_command_manifest(path, manifest):
  manifest_path = Path(path)
  if manifest_path.exists():
    try:
      saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
      raise ValueError(f"cannot read frozen command manifest: {error}") from error
    if saved != manifest:
      raise ValueError("frozen command manifest changed on study resume")
    return saved
  _atomic_json(manifest_path, manifest)
  return manifest


def control_execution_provenance(
  control,
  *,
  input_bytes,
  comparison,
  symiri,
):
  """Bind one control result to its checked source, input, and commands."""
  source = control.get("source")
  if not isinstance(source, str):
    raise ValueError("control provenance requires a control source")
  executable = str(Path(symiri).resolve())
  return {
    "control_source_sha256": _sha256(source.encode("utf-8")),
    "input_sha256": _sha256(input_bytes),
    "comparison": copy.deepcopy(comparison),
    "checker_command": {
      "executable": executable,
      "argv": [
        executable,
        "--check",
        "--tensor-profile-version",
        "v2",
        "<control.sir>",
      ],
      "timeout_s": 10,
    },
    "reference_command": {
      "executable": executable,
      "argv": [
        executable,
        "--tensor-profile-version",
        "v2",
        "--tensor-reference-stability",
        "--entry",
        "kernel",
        "--tensor-input",
        "<inputs.bin>",
        "--tensor-output",
        "<result.bin>",
        "<control.sir>",
      ],
      "timeout_s": DEFAULT_REFERENCE_TIMEOUT_S,
    },
  }


def gpu_command_record(
  *,
  case_directory,
  source_root,
  symiri,
  symirc,
  device_id,
  timeout_s,
):
  """Return the persisted runner and child-process invocation for one GPU case."""
  case_directory = str(Path(case_directory).resolve())
  source_root = str(Path(source_root).resolve())
  symiri = str(Path(symiri).resolve())
  symirc = str(Path(symirc).resolve())
  if type(device_id) is not int or device_id < 0:
    raise ValueError("GPU command device id must be nonnegative")
  return {
    "runner": "test.lib.run_triton_tests.run_case",
    "mode": "compiled",
    "case_directory": case_directory,
    "source_root": source_root,
    "device_id": device_id,
    "timeout_s": timeout_s,
    "worker_argv": [
      sys.executable,
      "-m",
      "test.lib.triton_case_worker",
      "--case",
      case_directory,
      "--artifacts-dir",
      str(Path(case_directory) / "runs" / "compiled"),
      "--mode",
      "compiled",
    ],
    "symirc_argv_template": [
      symirc,
      "--target",
      "triton",
      "--entry",
      "<entry>",
      "--tensor-softmax-lowering",
      "<softmax-lowering>",
      "-o",
      "<temporary-kernel.py>",
      str(Path(case_directory) / "program.sir"),
    ],
    "environment": {
      "CUDA_VISIBLE_DEVICES": str(device_id),
      "REFRACTIR_SYMIRI": symiri,
      "REFRACTIR_SYMIRC": symirc,
      "REFRACTIR_COMPILER_FLAGS": os.environ.get(
        "REFRACTIR_COMPILER_FLAGS", "unavailable"
      ),
    },
  }


def _verify_import_root(root):
  """A recorded source snapshot must contain the helpers this process uses."""
  if Path(__file__).resolve().parents[2] != root:
    raise ValueError("study import root differs from the requested source snapshot")
  for name, module in tuple(sys.modules.items()):
    if name != "test" and not name.startswith("test."):
      continue
    filename = getattr(module, "__file__", None)
    if filename is not None and not Path(filename).resolve().is_relative_to(
      root / "test"
    ):
      raise ValueError(
        f"study import root differs for loaded helper {name}: {filename}"
      )


def configure_runtime(
  source_root,
  *,
  symiri,
  symirc,
  revision,
  compiler_flags,
):
  """Set paths before importing modules that cache the RefractIR executables."""
  root = Path(source_root).resolve()
  _verify_import_root(root)
  interpreter = Path(symiri).resolve()
  compiler = Path(symirc).resolve()
  for executable in (interpreter, compiler):
    if not executable.is_file() or not os.access(executable, os.X_OK):
      raise FileNotFoundError(
        f"required RefractIR executable is unavailable: {executable}"
      )
  os.environ["REFRACTIR_SYMIRI"] = str(interpreter)
  os.environ["REFRACTIR_SYMIRC"] = str(compiler)
  os.environ["REFRACTIR_SOURCE_ROOT"] = str(root)
  os.environ["REFRACTIR_REVISION"] = revision
  os.environ.setdefault("REFRACTIR_WORKTREE_DIRTY", "unknown")
  os.environ["REFRACTIR_COMPILER_FLAGS"] = compiler_flags
  os.environ.pop("CUDA_VISIBLE_DEVICES", None)
  python_path = [str(root)]
  if os.environ.get("PYTHONPATH"):
    python_path.append(os.environ["PYTHONPATH"])
  os.environ["PYTHONPATH"] = os.pathsep.join(python_path)
  return {"symiri": str(interpreter), "symirc": str(compiler)}


def _resolve_triton_revision(environment, requested=None):
  existing = environment.get("triton_source_revision")
  if existing not in {None, "", "unavailable"}:
    return existing
  if requested:
    return requested
  version = str(environment.get("triton_version") or "")
  match = re.search(r"\+git([0-9a-f]{7,40})", version, re.IGNORECASE)
  if match:
    return match.group(1)
  raise RuntimeError(
    "Triton source revision is unavailable; pass --triton-revision explicitly"
  )


def validate_v100_environment(environment, device_ids):
  """Require every selected physical device to be an available V100."""
  if (
    not isinstance(environment, dict)
    or environment.get("campaign_available") is not True
  ):
    reasons = (
      environment.get("failure_reasons", []) if isinstance(environment, dict) else []
    )
    raise RuntimeError(f"Triton environment is unavailable: {reasons}")
  gpus = environment.get("gpus")
  if not isinstance(gpus, list):
    raise RuntimeError("environment probe did not report GPU devices")
  selected = []
  for device_id in device_ids:
    if type(device_id) is not int or device_id < 0 or device_id >= len(gpus):
      raise ValueError(f"selected GPU id {device_id!r} is unavailable")
    gpu = gpus[device_id]
    name = str(gpu.get("name", ""))
    if "v100" not in name.casefold() or gpu.get("compute_capability") != "7.0":
      raise RuntimeError(
        f"GPU {device_id} is not a V100 (name={name!r}, "
        f"compute_capability={gpu.get('compute_capability')!r})"
      )
    selected.append({"device_id": device_id, **gpu})
  return selected


def _timestamp():
  return datetime.now(timezone.utc).isoformat()


@contextmanager
def _study_lock(output_dir):
  lock_path = Path(output_dir) / "study.lock"
  lock_path.parent.mkdir(parents=True, exist_ok=True)
  with lock_path.open("a+b") as lock_file:
    try:
      fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
      raise RuntimeError(f"another study driver holds {lock_path}") from error
    try:
      yield
    finally:
      fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_json_lines(path, value):
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  with destination.open("a", encoding="utf-8") as output:
    output.write(
      json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
    )
    output.flush()
    os.fsync(output.fileno())


def _load_or_create_ledger(path, corpus):
  ledger_path = Path(path)
  if ledger_path.exists():
    ledger = study.load_ledger(ledger_path, corpus)
    study.save_ledger(ledger_path, ledger)
    return ledger
  ledger = study.new_study_ledger(corpus)
  study.save_ledger(ledger_path, ledger)
  # Sidecars may have reached disk immediately before a prior process ended.
  if Path(f"{ledger_path}.proposals").is_dir():
    ledger = study.load_ledger(ledger_path, corpus)
    study.save_ledger(ledger_path, ledger)
  return ledger


def _record_run_status(output_dir, status, **fields):
  _atomic_json(
    Path(output_dir) / "status.json",
    {"schema": RUN_SCHEMA, "status": status, "updated_at": _timestamp(), **fields},
  )


def _binary_identity(path):
  executable = Path(path).resolve()
  if not executable.is_file():
    raise FileNotFoundError(f"required study executable is missing: {executable}")
  return {
    "path": str(executable),
    "sha256": _sha256(executable.read_bytes()),
    "bytes": executable.stat().st_size,
  }


def _revision_for(source_root, requested):
  if requested:
    return requested
  result = subprocess.run(
    ["git", "-C", str(source_root), "rev-parse", "HEAD"],
    capture_output=True,
    text=True,
    timeout=5,
    check=False,
  )
  revision = result.stdout.strip()
  if result.returncode != 0 or not revision:
    raise ValueError("--revision is required when source_root has no Git HEAD")
  return revision


def _load_or_create_run_configuration(path, configuration):
  config_path = Path(path)
  if config_path.exists():
    try:
      saved = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
      raise ValueError(f"cannot read frozen study configuration: {error}") from error
    if saved != configuration:
      raise ValueError("frozen study configuration changed on resume")
    return saved
  _atomic_json(config_path, configuration)
  return configuration


def _persist_environment(output_dir, environment):
  directory = Path(output_dir)
  environment_path = directory / "environment.json"
  lock_path = directory / "package-lock.json"
  printable = {
    key: value for key, value in environment.items() if key != "package_lock"
  }
  if environment_path.exists() or lock_path.exists():
    try:
      saved_environment = json.loads(environment_path.read_text(encoding="utf-8"))
      saved_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
      raise ValueError(f"cannot read frozen environment record: {error}") from error
    if saved_environment != printable or saved_lock != environment.get("package_lock"):
      raise ValueError("Triton environment changed on study resume")
    return
  _atomic_json(environment_path, printable)
  _atomic_json(lock_path, environment["package_lock"])


def _row_major_strides(shape):
  strides = []
  stride = 1
  for extent in reversed(shape):
    strides.append(stride)
    stride *= extent
  return list(reversed(strides))


def normalize_case_template_launch(case_directory, launch):
  """Apply and persist the study launch contract on an ordinary template."""
  if (
    not isinstance(launch, dict)
    or set(launch) != {"num_warps", "num_stages", "enable_fp_fusion"}
    or type(launch.get("num_warps")) is not int
    or launch["num_warps"] not in {1, 2, 4, 8}
    or launch.get("num_stages") != 1
    or launch.get("enable_fp_fusion") is not False
  ):
    raise ValueError("frozen template launch settings are invalid")

  manifest_path = Path(case_directory) / "case.json"
  try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(f"cannot read generated case template: {error}") from error
  if not isinstance(manifest, dict) or manifest.get("profile") != "tensor-v2":
    raise ValueError("launch normalization requires a tensor-v2 case template")
  execution = manifest.get("execution")
  warp_probe = execution.get("warp_probe") if isinstance(execution, dict) else None
  if (
    not isinstance(warp_probe, dict)
    or warp_probe.get("status") != "not_probed"
    or "results" in warp_probe
  ):
    raise ValueError("launch normalization requires an unprobed warp record")

  manifest["launch"] = dict(launch)
  warp_probe["num_warps"] = launch["num_warps"]
  _atomic_json(manifest_path, manifest)
  return manifest


def deduplicate_package_lock_artifact(case_directory, output_dir):
  """Hard-link identical per-run package locks to one durable study copy."""
  package_lock = Path(case_directory) / "runs" / "compiled" / "package-lock.json"
  if not package_lock.is_file():
    return False

  shared_lock = Path(output_dir) / "shared" / "package-lock.json"
  shared_lock.parent.mkdir(parents=True, exist_ok=True)
  package_digest = _sha256(package_lock.read_bytes())
  if shared_lock.exists():
    if _sha256(shared_lock.read_bytes()) != package_digest:
      raise ValueError("compiled cases produced different package-lock artifacts")
  else:
    try:
      os.link(package_lock, shared_lock)
    except FileExistsError:
      if _sha256(shared_lock.read_bytes()) != package_digest:
        raise ValueError("compiled cases produced different package-lock artifacts")

  if os.path.samefile(package_lock, shared_lock):
    return True
  temporary = package_lock.with_name(f".{package_lock.name}.{os.getpid()}.link")
  temporary.unlink(missing_ok=True)
  try:
    os.link(shared_lock, temporary)
    os.replace(temporary, package_lock)
    directory_fd = os.open(package_lock.parent, os.O_RDONLY)
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
  finally:
    temporary.unlink(missing_ok=True)
  return True


def validate_case_template(
  case_directory,
  *,
  source,
  symiri,
  config,
):
  """Validate frozen source, input/output signatures, comparator, and layouts."""
  from test.lib import tensor_case

  case = tensor_case.load_case(case_directory, symiri=Path(symiri))
  manifest = case.manifest
  if case.program.decode("utf-8") != source:
    raise ValueError("generated case template source differs from the frozen program")
  if manifest.get("entry") != "kernel" or manifest.get("profile") != "tensor-v2":
    raise ValueError("generated case template entry or profile changed")
  if manifest.get("grid") != config["physical_layouts"]["grid"]:
    raise ValueError("generated case template grid differs from frozen configuration")
  if manifest.get("launch") != config["launch"]:
    raise ValueError("generated case template launch differs from frozen configuration")
  if manifest.get("comparison") != config["comparison"]:
    raise ValueError(
      "generated case template comparator differs from frozen configuration"
    )

  expected_inputs = study.input_signature_from_source(source)
  manifest_inputs = manifest.get("inputs")
  if not isinstance(manifest_inputs, list) or len(manifest_inputs) != len(
    expected_inputs
  ):
    raise ValueError("generated case template input count differs from frozen source")
  for expected, actual in zip(expected_inputs, manifest_inputs, strict=True):
    for field in ("name", "dtype", "shape", "byte_length"):
      if actual.get(field) != expected[field]:
        raise ValueError(f"generated case template input {field} differs from source")
    if actual["dtype"] != "f32":
      raise ValueError("frozen study requires f32 runtime inputs")
  for record in [*manifest_inputs, manifest.get("output")]:
    shape = record.get("shape")
    if record.get("element_strides") != _row_major_strides(shape):
      raise ValueError(
        "generated case template does not use contiguous row-major layout"
      )
    if record.get("element_offset") != 0:
      raise ValueError("generated case template buffer offset is not zero")
    expected_span = math.prod(shape) if shape else 1
    if record.get("storage_span") != expected_span:
      raise ValueError("generated case template storage span is not contiguous")
    expected_order = sorted(
      range(len(shape)), key=lambda axis: (record["element_strides"][axis], axis)
    )
    if record.get("axis_order") != expected_order:
      raise ValueError("generated case template axis order is not row-major")
  if manifest.get("output", {}).get("dtype") != "f32":
    raise ValueError("frozen study requires an f32 output")
  return manifest


def materialize_or_reuse_case(
  template_directory,
  destination,
  *,
  input_bytes,
  expected_bytes,
  policy,
  sampler_version,
  proposal_id,
  corpus_sha256,
  input_seed=None,
):
  """Create a proposal case once and verify identity before reusing it."""
  template = Path(template_directory)
  target = Path(destination)
  identity = {"corpus_sha256": corpus_sha256, "proposal_id": proposal_id}
  if target.exists():
    try:
      manifest = json.loads((target / "case.json").read_text(encoding="utf-8"))
      program = (target / "program.sir").read_bytes()
      saved_inputs = (target / "inputs.bin").read_bytes()
      saved_expected = (target / "expected.bin").read_bytes()
    except (OSError, json.JSONDecodeError) as error:
      raise ValueError(f"existing proposal case is incomplete: {error}") from error
    if (
      manifest.get("boundary_sensitivity") != identity
      or program != (template / "program.sir").read_bytes()
      or saved_inputs != input_bytes
      or saved_expected != expected_bytes
    ):
      raise ValueError("existing proposal case identity or bytes changed")
    for name, payload in (
      ("inputs.bin", input_bytes),
      ("expected.bin", expected_bytes),
    ):
      record = manifest.get("files", {}).get(name)
      if (
        not isinstance(record, dict)
        or record.get("bytes") != len(payload)
        or record.get("sha256") != _sha256(payload)
      ):
        raise ValueError("existing proposal case identity has invalid file digests")
    generator = manifest.get("generator", {})
    if (
      generator.get("input_policy") != policy
      or generator.get("input_sampler_version") != sampler_version
      or (input_seed is not None and generator.get("input_seed") != input_seed)
    ):
      raise ValueError("existing proposal case identity has changed provenance")
    return target, True

  target.parent.mkdir(parents=True, exist_ok=True)
  case_path = study.materialize_proposal_case(
    template,
    target,
    input_bytes=input_bytes,
    expected_bytes=expected_bytes,
    policy=policy,
    sampler_version=sampler_version,
    proposal_id=proposal_id,
    corpus_sha256=corpus_sha256,
    input_seed=input_seed,
  )
  return case_path, False


def output_shape_from_source(source):
  match = re.search(
    r"fun\s+@kernel\(.*?\)\s*:\s*(?:tensor\s*\[([^]]*)\]\s*)?f32\b",
    source,
    re.DOTALL,
  )
  if match is None:
    raise ValueError("frozen source has an unsupported output type")
  shape_text = match.group(1)
  return list(study._parse_shape(shape_text)) if shape_text is not None else []


def _run_gpu_job(job):
  started = time.monotonic()
  case_directory = Path(job["case_directory"])
  try:
    outcome = execute_gpu_case(
      case_directory,
      device_id=job["device_id"],
      timeout_s=job["timeout_s"],
    )
  except Exception as error:
    saved_report = _read_json_if_present(case_directory / "result-compiled.json")
    outcome = {
      "report": saved_report
      if isinstance(saved_report, dict)
      else {"status": "worker_exception", "reason": str(error)},
      "recovered": isinstance(saved_report, dict),
      "worker_exception": str(error),
    }
  return {
    "proposal_id": job["proposal_id"],
    "case_directory": str(case_directory),
    "device_id": job["device_id"],
    "elapsed_s": max(0.0, time.monotonic() - started),
    **outcome,
  }


def run_compiled_jobs(jobs, *, workers, environment, on_result):
  """Run at most one case per V100 and keep all ledger writes in the parent."""
  if not jobs:
    return
  lanes = {}
  for job in jobs:
    lanes.setdefault(job["device_id"], []).append(job)
  if workers < 1 or workers > len(lanes):
    raise ValueError("GPU worker count must be between one and the selected devices")
  active_devices = list(lanes)[:workers]
  if any(lanes[device_id] for device_id in list(lanes)[workers:]):
    raise ValueError("jobs were assigned to more devices than the worker limit")
  positions = {device_id: 0 for device_id in active_devices}

  with ProcessPoolExecutor(
    max_workers=workers,
    mp_context=multiprocessing.get_context("spawn"),
    initializer=_initialize_gpu_worker,
    initargs=(environment,),
  ) as executor:
    pending = {}
    for device_id in active_devices:
      job = lanes[device_id][0]
      positions[device_id] = 1
      pending[executor.submit(_run_gpu_job, job)] = device_id
    while pending:
      completed, _ = wait(pending, return_when=FIRST_COMPLETED)
      for future in completed:
        device_id = pending.pop(future)
        try:
          outcome = future.result()
        except Exception as error:
          job = lanes[device_id][positions[device_id] - 1]
          outcome = {
            "proposal_id": job["proposal_id"],
            "case_directory": job["case_directory"],
            "device_id": device_id,
            "elapsed_s": None,
            "report": {"status": "worker_exception", "reason": str(error)},
            "recovered": False,
          }
        on_result(outcome)
        if positions[device_id] < len(lanes[device_id]):
          job = lanes[device_id][positions[device_id]]
          positions[device_id] += 1
          pending[executor.submit(_run_gpu_job, job)] = device_id


def _write_template_status(path, status):
  _atomic_json(path, status)


def ensure_case_template(
  kernel,
  *,
  template_root,
  rysmith,
  source_root,
  symiri,
  config,
  generation_timeout_s=120,
  environment=None,
):
  """Generate exactly one deterministic ordinary case for a frozen graph."""
  root = Path(template_root) / kernel["kernel_id"]
  generation_root = root / "generation"
  case_directory = generation_root / "case_000000"
  status_path = root / "template-status.json"
  if status_path.exists():
    try:
      status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
      raise ValueError(f"case template status is corrupt: {error}") from error
    if status.get("status") == "accepted":
      normalize_case_template_launch(case_directory, config["launch"])
      validate_case_template(
        case_directory, source=kernel["source"], symiri=symiri, config=config
      )
    return status

  command_path = root / "generation-command.json"
  if command_path.exists():
    if case_directory.is_dir():
      try:
        normalize_case_template_launch(case_directory, config["launch"])
        validate_case_template(
          case_directory, source=kernel["source"], symiri=symiri, config=config
        )
      except Exception as error:
        interrupted = {
          "status": "template_invalid",
          "reason": str(error),
          "case_directory": str(case_directory),
        }
      else:
        interrupted = {
          "status": "accepted",
          "case_directory": str(case_directory),
          "recovered": True,
        }
      _write_template_status(status_path, interrupted)
      return interrupted
    interrupted = {
      "status": "generation_interrupted",
      "reason": "a template generation command was checkpointed without a terminal status",
      "case_directory": str(case_directory),
    }
    _write_template_status(status_path, interrupted)
    return interrupted

  root.mkdir(parents=True, exist_ok=True)
  from test.tensor.campaigns.continuous_triton_campaign import build_generation_command

  command = build_generation_command(
    rysmith,
    generation_root,
    kernel["candidate_seed"],
    profile_version="v2",
    dtype="f32",
    generator="compositional",
    input_policy="ordinary",
    count=1,
  )
  command.extend(["--max-retries", "1"])
  _atomic_json(command_path, {"command": command, "cwd": str(source_root)})
  generation_root.mkdir(parents=True, exist_ok=True)
  try:
    result = subprocess.run(
      command,
      cwd=source_root,
      env=environment,
      capture_output=True,
      timeout=generation_timeout_s,
      check=False,
    )
    (root / "generation.stdout.log").write_bytes(result.stdout)
    (root / "generation.stderr.log").write_bytes(result.stderr)
    if result.returncode != 0 or not case_directory.is_dir():
      status = {
        "status": "generation_failure",
        "returncode": result.returncode,
        "reason": (result.stderr or result.stdout).decode("utf-8", errors="replace")[
          -4000:
        ],
        "case_directory": str(case_directory),
      }
      _write_template_status(status_path, status)
      return status
  except subprocess.TimeoutExpired as error:
    status = {
      "status": "generation_timeout",
      "reason": str(error),
      "case_directory": str(case_directory),
    }
    _write_template_status(status_path, status)
    return status
  except OSError as error:
    status = {
      "status": "generation_failure",
      "reason": str(error),
      "case_directory": str(case_directory),
    }
    _write_template_status(status_path, status)
    return status

  try:
    normalize_case_template_launch(case_directory, config["launch"])
    validate_case_template(
      case_directory, source=kernel["source"], symiri=symiri, config=config
    )
  except Exception as error:
    status = {
      "status": "template_invalid",
      "reason": str(error),
      "case_directory": str(case_directory),
    }
    _write_template_status(status_path, status)
    return status
  status = {"status": "accepted", "case_directory": str(case_directory)}
  _write_template_status(status_path, status)
  return status


def _decode_staged_buffers(staged, input_records):
  buffers = staged.get("input_buffers")
  if not isinstance(buffers, list):
    raise ValueError("staged accepted proposal omitted its input buffers")
  decoded = []
  for buffer in buffers:
    if not isinstance(buffer, dict) or not isinstance(buffer.get("bytes_hex"), str):
      raise ValueError("staged proposal contains a malformed input buffer")
    try:
      data = bytes.fromhex(buffer["bytes_hex"])
    except ValueError as error:
      raise ValueError("staged proposal contains invalid input bytes") from error
    decoded.append({**buffer, "bytes": data})
  combined = study.combine_input_buffers(decoded, input_records)
  try:
    declared = bytes.fromhex(staged["input_bytes_hex"])
  except (KeyError, TypeError, ValueError) as error:
    raise ValueError("staged proposal omitted valid combined input bytes") from error
  if combined != declared:
    raise ValueError("staged proposal buffers differ from combined input bytes")
  return decoded, combined


def prepare_proposal(slot, source, input_records, generator_api, *, staged=None):
  """Return one allocated proposal without retrying or replacing its input."""
  expected_version = slot.get("sampler_version")
  if expected_version not in {
    study.LEGACY_SAMPLER_VERSION,
    study.BOUNDARY_SAMPLER_VERSION,
  }:
    raise ValueError("allocated proposal has an unknown sampler version")
  if staged is not None:
    if not isinstance(staged, dict):
      raise ValueError("staged proposal must be an object")
    if staged.get("sampler_version") != expected_version:
      raise ValueError("staged proposal sampler version differs from allocation")
    proposal = copy.deepcopy(staged)
    if proposal.get("proposal_status") == "accepted":
      _buffers, input_bytes = _decode_staged_buffers(proposal, input_records)
      return proposal, input_bytes
    if proposal.get("proposal_status") not in study.PROPOSAL_STATUSES:
      raise ValueError("staged proposal status is invalid")
    return proposal, None

  if slot.get("arm") == "legacy_cycle":
    buffers = study.legacy_cycle_buffers(input_records, slot.get("legacy_phase"))
    status = "accepted"
    reason = None
    sampling_command = {
      "implementation": "legacy_cycle_buffers",
      "phase": slot.get("legacy_phase"),
      "input_domain": [-1, 2],
    }
  elif slot.get("arm") == "boundary":
    sampled = study.propose_boundary_inputs(
      generator_api,
      source,
      slot.get("sampler_seed"),
      timeout_s=DEFAULT_PROPOSAL_TIMEOUT_S,
    )
    status = sampled.get("status")
    if status not in {"accepted", "refused", "timeout", "error"}:
      status = "error"
    buffers = sampled.get("buffers", []) if status == "accepted" else []
    reason = sampled.get("reason")
    sampling_command = {
      "executable": str(Path(generator_api).resolve()),
      "argv": [
        str(Path(generator_api).resolve()),
        "<program.sir>",
        str(slot.get("sampler_seed")),
      ],
      "timeout_s": DEFAULT_PROPOSAL_TIMEOUT_S,
      "source_sha256": _sha256(source.encode("utf-8")),
    }
    if sampled.get("sampler_version") not in {None, expected_version}:
      raise ValueError("native sampler version differs from frozen allocation")
  else:
    raise ValueError("allocated proposal has an unknown input-policy arm")

  proposal = {
    "proposal_status": status,
    "sampler_version": expected_version,
    "source_sha256": _sha256(source.encode("utf-8")),
    "sampling_command": sampling_command,
  }
  if reason:
    proposal["reason"] = reason
  if status != "accepted":
    return proposal, None

  input_bytes = study.combine_input_buffers(buffers, input_records)
  proposal["input_bytes_hex"] = input_bytes.hex()
  proposal["input_sha256"] = _sha256(input_bytes)
  proposal["input_buffers"] = [
    {
      "name": record["name"],
      "dtype": record["dtype"],
      "shape": list(record["shape"]),
      "bytes_hex": buffer["bytes"].hex(),
    }
    for record in input_records
    for buffer in buffers
    if buffer["name"] == record["name"]
  ]
  return proposal, input_bytes


def replay_or_reuse_reference(slot, source, input_bytes, *, reference_runner=None):
  """Recover a checkpointed reference or run the full reference once."""
  checkpoint = slot.get("reference")
  if checkpoint is not None:
    if not isinstance(checkpoint, dict):
      raise ValueError("reference checkpoint must be an object")
    if checkpoint.get("source_sha256") != _sha256(source.encode("utf-8")):
      raise ValueError("reference checkpoint belongs to a different program")
    if not isinstance(input_bytes, bytes) or checkpoint.get("input_sha256") != _sha256(
      input_bytes
    ):
      raise ValueError("reference checkpoint belongs to different input bytes")
    reference = copy.deepcopy(checkpoint)
    if reference.get("status") == "accepted":
      try:
        output = bytes.fromhex(reference["output_bytes_hex"])
      except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
          "accepted reference checkpoint has invalid output bytes"
        ) from error
      if len(output) != 4 * math.prod(output_shape_from_source(source)):
        raise ValueError("reference checkpoint does not contain the complete output")
      return reference, output
    if reference.get("status") not in {"invalid_on_input", "timeout"}:
      raise ValueError("reference checkpoint has an invalid status")
    return reference, None

  if not isinstance(input_bytes, bytes):
    raise ValueError("reference replay requires accepted proposal bytes")
  started = time.monotonic()
  outcome = study.replay_reference(
    source, input_bytes, reference_runner=reference_runner
  )
  elapsed = max(0.0, time.monotonic() - started)
  reference = {key: value for key, value in outcome.items() if key != "output_bytes"}
  reference["elapsed_s"] = elapsed
  reference["source_sha256"] = _sha256(source.encode("utf-8"))
  reference["input_sha256"] = _sha256(input_bytes)
  reference["command"] = {
    "executable": os.environ.get("REFRACTIR_SYMIRI", "symiri"),
    "arguments": [
      "--tensor-profile-version",
      "v2",
      "--tensor-reference-stability",
      "--entry",
      "kernel",
      "--tensor-input",
      "<inputs.bin>",
      "--tensor-output",
      "<result.bin>",
      "<program.sir>",
    ],
    "timeout_s": DEFAULT_REFERENCE_TIMEOUT_S,
    "cwd": os.environ.get("REFRACTIR_SOURCE_ROOT"),
  }
  output = outcome.get("output_bytes") if outcome.get("status") == "accepted" else None
  if output is not None and not isinstance(output, bytes):
    raise ValueError("reference runner returned malformed output bytes")
  return reference, output


def _read_json_if_present(path):
  try:
    return json.loads(Path(path).read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError, TypeError):
    return None


def classify_gpu_run(case_dir, report, *, elapsed_s):
  """Count only compilation/execution phases confirmed by runner artifacts."""
  root = Path(case_dir)
  run_dir = root / "runs" / "compiled"
  phase_record = _read_json_if_present(run_dir / "worker-phase.json")
  timing_record = _read_json_if_present(run_dir / "worker-timing.json")
  phase = phase_record.get("phase") if isinstance(phase_record, dict) else None
  timing = timing_record if isinstance(timing_record, dict) else {}
  compile_s = timing.get("compilation_s")
  execution_s = timing.get("gpu_execution_s")
  compile_confirmed = phase in {"compilation", "runtime"} or (
    type(compile_s) in (int, float) and compile_s > 0
  )
  execution_confirmed = phase == "runtime" or (
    type(execution_s) in (int, float) and execution_s > 0
  )
  status = report.get("status") if isinstance(report, dict) else "unknown"
  confirmed_skipped = status in {
    "invalid_case",
    "environment_unavailable",
    "invalid_on_input",
  } or (status == "compile_failure" and phase is None)

  if compile_confirmed:
    compilation_count = 1
    compilation_state = "attempted"
  elif confirmed_skipped:
    compilation_count = 0
    compilation_state = "skipped"
  else:
    compilation_count = None
    compilation_state = "unknown"

  if execution_confirmed:
    execution_count = 1
    execution_state = "attempted"
  elif confirmed_skipped or phase == "compilation" or status == "compile_failure":
    execution_count = 0
    execution_state = "skipped"
  else:
    execution_count = None
    execution_state = "unknown"

  return {
    "compilation_count": compilation_count,
    "execution_count": execution_count,
    "compilation_state": compilation_state,
    "execution_state": execution_state,
    "worker_phase": phase,
    "worker_timing": timing,
    "runner_elapsed_s": (None if elapsed_s is None else max(0.0, float(elapsed_s))),
  }


def run_compiled_case(case_dir, *, device_id, timeout_s=DEFAULT_GPU_TIMEOUT_S):
  """Run the repository's compiled Triton case path using the frozen binaries."""
  from test.lib import run_triton_tests

  return run_triton_tests.run_case(
    case_dir,
    "compiled",
    timeout_s=timeout_s,
    symiri=Path(os.environ["REFRACTIR_SYMIRI"]),
    symirc=Path(os.environ["REFRACTIR_SYMIRC"]),
    device_id=device_id,
    environment_probe=(
      (lambda: _WORKER_ENVIRONMENT) if _WORKER_ENVIRONMENT is not None else None
    ),
  )


def execute_gpu_case(
  case_dir,
  *,
  device_id,
  timeout_s=DEFAULT_GPU_TIMEOUT_S,
  run_fn=None,
):
  """Execute once, recovering terminal reports and refusing hidden retries."""
  root = Path(case_dir)
  report_path = root / "result-compiled.json"
  if report_path.exists():
    report = _read_json_if_present(report_path)
    if not isinstance(report, dict) or not isinstance(report.get("status"), str):
      return {
        "report": {"status": "corrupt_terminal_report"},
        "recovered": True,
      }
    return {"report": report, "recovered": True}

  compiled_dir = root / "runs" / "compiled"
  if compiled_dir.exists() and any(compiled_dir.iterdir()):
    return {
      "report": {
        "status": "interrupted",
        "reason": "compiled run artifacts exist without a terminal report",
      },
      "recovered": False,
      "interrupted": True,
    }

  runner = run_fn or run_compiled_case
  report = runner(root, device_id=device_id, timeout_s=timeout_s)
  if not isinstance(report, dict) or not isinstance(report.get("status"), str):
    raise ValueError("compiled case runner returned an invalid report")
  return {"report": report, "recovered": False}


def recover_submitted_gpu_case(case_dir):
  """Recover a checkpointed submission without ever resubmitting it."""
  root = Path(case_dir)
  report_path = root / "result-compiled.json"
  if report_path.exists():
    return execute_gpu_case(root, device_id=0)
  return {
    "report": {
      "status": "interrupted",
      "reason": "GPU submission was checkpointed but has no terminal report",
    },
    "recovered": True,
    "interrupted": True,
  }


def _initialize_gpu_worker(environment):
  global _WORKER_ENVIRONMENT
  _WORKER_ENVIRONMENT = environment


def _skipped_gpu_work(reason):
  return {
    "compilation_count": 0,
    "execution_count": 0,
    "compilation_state": "skipped",
    "execution_state": "skipped",
    "skip_reason": reason,
  }


def _record_control(ledger, ledger_path, slot, control):
  study.record_control_result(ledger, slot["proposal_id"], control)
  study.save_proposal_record(ledger_path, ledger, slot["proposal_id"])


def _record_proposal(ledger, ledger_path, slot, proposal):
  study.record_proposal_input(ledger, slot["proposal_id"], proposal)
  study.save_proposal_record(ledger_path, ledger, slot["proposal_id"])


def _record_reference(ledger, ledger_path, slot, reference):
  study.record_reference_result(ledger, slot["proposal_id"], reference)
  study.save_proposal_record(ledger_path, ledger, slot["proposal_id"])


def _write_slot_result(
  ledger,
  ledger_path,
  slot,
  *,
  proposal,
  input_bytes,
  reference,
  gpu,
):
  compiled_elapsed = gpu.get("runner_elapsed_s", 0.0)
  result = {
    "proposal_status": proposal["proposal_status"],
    "sampler_version": slot["sampler_version"],
    "gpu": gpu,
    "cost_s": {
      "reference": float(reference.get("elapsed_s", 0.0)) if reference else 0.0,
      "compiled": (None if compiled_elapsed is None else float(compiled_elapsed)),
    },
  }
  if proposal.get("reason"):
    result["proposal_reason"] = proposal["reason"]
  if input_bytes is not None:
    result["input_bytes_hex"] = input_bytes.hex()
  if reference is not None:
    result["reference_status"] = reference["status"]
    if reference.get("reason"):
      result["reference_reason"] = reference["reason"]
    if reference.get("output_bytes_hex") is not None:
      result["output_bytes_hex"] = reference["output_bytes_hex"]
  study.record_proposal_result(ledger, slot["proposal_id"], result)
  study.save_proposal_record(ledger_path, ledger, slot["proposal_id"])


def _gpu_result_record(outcome, *, output_dir):
  case_directory = Path(outcome["case_directory"])
  report = outcome.get("report", {})
  gpu = classify_gpu_run(case_directory, report, elapsed_s=outcome["elapsed_s"])
  report_path = case_directory / "result-compiled.json"
  if report_path.is_file():
    gpu["report_path"] = str(report_path.relative_to(output_dir))
    gpu["report_sha256"] = _sha256(report_path.read_bytes())
  gpu["report_status"] = report.get("status", "unknown")
  gpu["report"] = report
  gpu["device_id"] = outcome["device_id"]
  gpu["case_directory"] = str(case_directory.relative_to(output_dir))
  gpu["recovered"] = outcome.get("recovered", False)
  if outcome.get("interrupted"):
    gpu["interrupted"] = True
  if outcome.get("worker_exception"):
    gpu["worker_exception"] = outcome["worker_exception"]
  actual_path = case_directory / "runs" / "compiled" / "actual.bin"
  if actual_path.is_file():
    actual = actual_path.read_bytes()
    gpu["actual_output_bytes"] = len(actual)
    gpu["actual_output_sha256"] = _sha256(actual)
    gpu["actual_output_bytes_hex"] = actual.hex()
  return gpu


def _report_summary(ledger, *, output_dir, corpus, run_configuration):
  summary = study.summarize_study(ledger)
  summary.update(
    {
      "schema": "refractir.tensor-boundary-sensitivity-summary.v1",
      "status": "complete" if summary["pending_proposals"] == 0 else "incomplete",
      "base_revision": corpus["base_revision"],
      "source_snapshot_sha256": corpus["config"]["source_snapshot_sha256"],
      "corpus_sha256": corpus["corpus_sha256"],
      "corpus_path": "corpus.json",
      "ledger_path": "study-ledger.json",
      "control_catalogue_path": "control-catalogue.json",
      "command_manifest_path": "commands.json",
      "command_manifest_sha256": run_configuration.get("command_manifest_sha256"),
      "configuration": run_configuration,
      "output_directory": str(output_dir),
    }
  )
  return summary


def _persist_summary(output_dir, ledger, corpus, run_configuration):
  summary = _report_summary(
    ledger,
    output_dir=output_dir,
    corpus=corpus,
    run_configuration=run_configuration,
  )
  _atomic_json(Path(output_dir) / "summary.json", summary)
  return summary


def _device_for_proposal(proposal_id, device_ids):
  value = int(_sha256(proposal_id.encode("utf-8"))[:16], 16)
  return device_ids[value % len(device_ids)]


def _proposal_control_ids(slot):
  return {
    item.get("control_id")
    for item in slot.get("control_results", [])
    if isinstance(item, dict)
  }


def _run_proposals(
  *,
  corpus,
  ledger,
  catalogue,
  templates,
  generator_api,
  symiri,
  output_dir,
  ledger_path,
  config,
  active_gpu_ids,
  gpu_timeout_s,
  environment,
  run_configuration,
):
  controls = {item["kernel_id"]: item["controls"] for item in catalogue["kernels"]}
  gpu_jobs = []

  for kernel in corpus["kernels"]:
    kernel_id = kernel["kernel_id"]
    input_records = study.input_signature_from_source(kernel["source"])
    output_shape = output_shape_from_source(kernel["source"])
    template_status = templates[kernel_id]
    template_directory = Path(template_status.get("case_directory", ""))
    template_available = template_status.get("status") == "accepted"
    for slot in ledger["proposals"]:
      if slot["kernel_id"] != kernel_id or slot.get("result") is not None:
        continue

      staged = slot.get("proposal")
      if staged is None:
        try:
          proposal, input_bytes = prepare_proposal(
            slot,
            kernel["source"],
            input_records,
            generator_api,
          )
        except Exception as error:
          proposal = {
            "proposal_status": "error",
            "sampler_version": slot["sampler_version"],
            "reason": str(error),
          }
          input_bytes = None
        _record_proposal(ledger, ledger_path, slot, proposal)
      else:
        proposal, input_bytes = prepare_proposal(
          slot,
          kernel["source"],
          input_records,
          generator_api,
          staged=staged,
        )

      reference = slot.get("reference")
      reference_output = None
      if proposal["proposal_status"] == "accepted":
        if reference is None:
          reference, reference_output = replay_or_reuse_reference(
            {}, kernel["source"], input_bytes
          )
          _record_reference(ledger, ledger_path, slot, reference)
        else:
          reference, reference_output = replay_or_reuse_reference(
            {"reference": reference}, kernel["source"], input_bytes
          )

      control_records = controls[kernel_id]
      existing_control_ids = _proposal_control_ids(slot)
      for control_id in study.CONTROL_IDS:
        if control_id in existing_control_ids:
          continue
        control = control_records[control_id]
        if proposal["proposal_status"] != "accepted":
          result = (
            {
              "control_id": control_id,
              "status": "inapplicable",
              "reason": control.get("reason"),
            }
            if control.get("status") == "inapplicable"
            else {
              "control_id": control_id,
              "status": "not_run",
              "reason": "input proposal was not accepted",
            }
          )
        elif reference is None or reference.get("status") != "accepted":
          result = (
            {
              "control_id": control_id,
              "status": "inapplicable",
              "reason": control.get("reason"),
            }
            if control.get("status") == "inapplicable"
            else {
              "control_id": control_id,
              "status": "not_run",
              "reason": f"original reference status was {reference.get('status') if reference else 'missing'}",
            }
          )
        else:
          started = time.monotonic()
          result = study.evaluate_control(
            control,
            original_source=kernel["source"],
            input_bytes=input_bytes,
            expected_bytes=reference_output,
            output_shape=output_shape,
            comparison=config["comparison"],
          )
          result["elapsed_s"] = max(0.0, time.monotonic() - started)
          if control.get("status") == "available":
            result.update(
              control_execution_provenance(
                control,
                input_bytes=input_bytes,
                comparison=config["comparison"],
                symiri=symiri,
              )
            )
        _record_control(ledger, ledger_path, slot, result)

      if proposal["proposal_status"] != "accepted":
        _write_slot_result(
          ledger,
          ledger_path,
          slot,
          proposal=proposal,
          input_bytes=None,
          reference=None,
          gpu=_skipped_gpu_work("input proposal was not accepted"),
        )
        continue

      if reference is None or reference.get("status") != "accepted":
        _write_slot_result(
          ledger,
          ledger_path,
          slot,
          proposal=proposal,
          input_bytes=input_bytes,
          reference=reference,
          gpu=_skipped_gpu_work("original reference was not accepted"),
        )
        continue

      if not template_available:
        _write_slot_result(
          ledger,
          ledger_path,
          slot,
          proposal=proposal,
          input_bytes=input_bytes,
          reference=reference,
          gpu=_skipped_gpu_work(
            f"case template status was {template_status.get('status', 'missing')}"
          ),
        )
        continue

      case_directory = (
        Path(output_dir)
        / "cases"
        / kernel_id
        / slot["arm"]
        / f"proposal-{slot['proposal_index']:02d}"
      )
      input_seed = slot.get("sampler_seed", slot.get("legacy_phase"))
      materialize_or_reuse_case(
        template_directory,
        case_directory,
        input_bytes=input_bytes,
        expected_bytes=reference_output,
        policy=slot["arm"],
        sampler_version=slot["sampler_version"],
        proposal_id=slot["proposal_id"],
        corpus_sha256=corpus["corpus_sha256"],
        input_seed=input_seed,
      )

      if slot.get("gpu_submitted") is True:
        outcome = recover_submitted_gpu_case(case_directory)
        outcome.update(
          {
            "proposal_id": slot["proposal_id"],
            "case_directory": str(case_directory),
            "device_id": slot.get("gpu_device_id", active_gpu_ids[0]),
            "elapsed_s": None,
          }
        )
        deduplicate_package_lock_artifact(case_directory, output_dir)
        _write_slot_result(
          ledger,
          ledger_path,
          slot,
          proposal=proposal,
          input_bytes=input_bytes,
          reference=reference,
          gpu=_gpu_result_record(outcome, output_dir=Path(output_dir)),
        )
        continue

      device_id = _device_for_proposal(slot["proposal_id"], active_gpu_ids)
      slot["gpu_submitted"] = True
      slot["gpu_device_id"] = device_id
      slot["gpu_case_directory"] = str(case_directory.relative_to(output_dir))
      slot["gpu_submitted_at"] = _timestamp()
      slot["gpu_command"] = gpu_command_record(
        case_directory=case_directory,
        source_root=run_configuration["source_root"],
        symiri=symiri,
        symirc=os.environ["REFRACTIR_SYMIRC"],
        device_id=device_id,
        timeout_s=gpu_timeout_s,
      )
      study.save_proposal_record(ledger_path, ledger, slot["proposal_id"])
      gpu_jobs.append(
        {
          "proposal_id": slot["proposal_id"],
          "case_directory": str(case_directory),
          "device_id": device_id,
          "timeout_s": gpu_timeout_s,
        }
      )

  study.save_ledger(ledger_path, ledger)
  _persist_summary(output_dir, ledger, corpus, run_configuration)

  slots_by_id = {slot["proposal_id"]: slot for slot in ledger["proposals"]}
  completed_gpu_jobs = 0

  def store_gpu_outcome(outcome):
    nonlocal completed_gpu_jobs
    slot = slots_by_id[outcome["proposal_id"]]
    proposal = slot["proposal"]
    input_bytes = bytes.fromhex(proposal["input_bytes_hex"])
    reference = slot["reference"]
    deduplicate_package_lock_artifact(outcome["case_directory"], output_dir)
    _write_slot_result(
      ledger,
      ledger_path,
      slot,
      proposal=proposal,
      input_bytes=input_bytes,
      reference=reference,
      gpu=_gpu_result_record(outcome, output_dir=Path(output_dir)),
    )
    completed_gpu_jobs += 1
    if completed_gpu_jobs % max(1, len(active_gpu_ids)) == 0:
      study.save_ledger(ledger_path, ledger)
      _persist_summary(output_dir, ledger, corpus, run_configuration)
      _record_run_status(
        output_dir,
        "running",
        phase="compiled_execution",
        completed_gpu_jobs=completed_gpu_jobs,
        submitted_gpu_jobs=len(gpu_jobs),
      )

  run_compiled_jobs(
    gpu_jobs,
    workers=min(len(active_gpu_ids), len({job["device_id"] for job in gpu_jobs}))
    if gpu_jobs
    else 0,
    environment=environment,
    on_result=store_gpu_outcome,
  )
  study.save_ledger(ledger_path, ledger)
  return _persist_summary(output_dir, ledger, corpus, run_configuration)


def run_study(
  *,
  source_root,
  output_dir,
  revision,
  generator_api,
  rysmith,
  symiri,
  symirc,
  gpu_ids,
  workers=DEFAULT_WORKERS,
  target_count=DEFAULT_TARGET_COUNT,
  start_seed=DEFAULT_START_SEED,
  max_candidates=None,
  candidate_timeout_s=30,
  template_timeout_s=120,
  gpu_timeout_s=DEFAULT_GPU_TIMEOUT_S,
  triton_revision=None,
  compiler_flags=DEFAULT_COMPILER_FLAGS,
):
  """Freeze inputs and run the fixed legacy-cycle/boundary comparison."""
  source_root = Path(source_root).resolve()
  output_dir = Path(output_dir).resolve()
  generator_api = Path(generator_api).resolve()
  rysmith = Path(rysmith).resolve()
  symiri = Path(symiri).resolve()
  symirc = Path(symirc).resolve()
  if output_dir == source_root or source_root in output_dir.parents:
    raise ValueError("study output directory must be outside the source snapshot")
  if type(target_count) is not int or target_count < 1:
    raise ValueError("target_count must be a positive integer")
  if type(start_seed) is not int or start_seed < 0 or start_seed > 0xFFFFFFFF:
    raise ValueError("start_seed must fit an unsigned 32-bit integer")
  if max_candidates is None:
    max_candidates = target_count * 50
  if type(max_candidates) is not int or max_candidates < target_count:
    raise ValueError("max_candidates must be at least target_count")
  if start_seed + max_candidates - 1 > 0xFFFFFFFF:
    raise ValueError("candidate seed schedule exceeds uint32")
  if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
    raise ValueError("gpu_ids must contain unique device ids")
  if type(workers) is not int or workers < 1 or workers > len(gpu_ids):
    raise ValueError("workers must be between one and the number of GPU ids")
  for name, value in (
    ("candidate timeout", candidate_timeout_s),
    ("template timeout", template_timeout_s),
    ("GPU timeout", gpu_timeout_s),
  ):
    if type(value) not in (int, float) or value <= 0:
      raise ValueError(f"{name} must be positive")

  output_dir.mkdir(parents=True, exist_ok=True)
  with _study_lock(output_dir):
    executable_paths = {
      "generator_api": generator_api,
      "rysmith": rysmith,
      "symiri": symiri,
      "symirc": symirc,
    }
    base_revision = _revision_for(source_root, revision)
    configure_runtime(
      source_root,
      symiri=symiri,
      symirc=symirc,
      revision=base_revision,
      compiler_flags=compiler_flags,
    )
    executable_identities = {
      name: _binary_identity(path) for name, path in executable_paths.items()
    }
    source_snapshot = load_or_create_source_snapshot(
      output_dir / "source-snapshot.json", source_root
    )
    config = frozen_config(source_snapshot["snapshot_sha256"])

    from test.lib import probe_triton_env

    environment = probe_triton_env.collect_environment()
    environment["triton_source_revision"] = _resolve_triton_revision(
      environment, triton_revision
    )
    environment["refractir_revision"] = base_revision
    active_gpu_ids = list(gpu_ids[:workers])
    selected_gpus = validate_v100_environment(environment, active_gpu_ids)
    environment["selected_devices"] = selected_gpus
    commands = command_manifest(
      source_root=source_root,
      generator_api=generator_api,
      rysmith=rysmith,
      symiri=symiri,
      symirc=symirc,
      compiler_flags=compiler_flags,
      candidate_timeout_s=candidate_timeout_s,
      template_timeout_s=template_timeout_s,
      reference_timeout_s=DEFAULT_REFERENCE_TIMEOUT_S,
      gpu_timeout_s=gpu_timeout_s,
    )
    _load_or_create_command_manifest(output_dir / "commands.json", commands)
    run_configuration = {
      "schema": RUN_SCHEMA,
      "base_revision": base_revision,
      "source_root": str(source_root),
      "source_snapshot_sha256": source_snapshot["snapshot_sha256"],
      "corpus_config": config,
      "target_count": target_count,
      "start_seed": start_seed,
      "max_candidates": max_candidates,
      "candidate_timeout_s": candidate_timeout_s,
      "template_timeout_s": template_timeout_s,
      "gpu_timeout_s": gpu_timeout_s,
      "command_manifest_sha256": commands["manifest_sha256"],
      "compiler_flags": compiler_flags,
      "proposals_per_arm": 4,
      "planned_original_evaluations": target_count * 2 * 4,
      "gpu_ids": list(gpu_ids),
      "active_gpu_ids": active_gpu_ids,
      "workers": workers,
      "selected_gpus": selected_gpus,
      "triton": {
        "version": environment.get("triton_version"),
        "source_revision": environment.get("triton_source_revision"),
        "cuda_version": environment.get("cuda_version"),
        "python_version": environment.get("python_version"),
        "torch_version": environment.get("torch_version"),
        "package_lock_sha256": environment.get("package_lock_sha256"),
      },
      "executables": executable_identities,
    }
    _load_or_create_run_configuration(
      output_dir / "study-configuration.json", run_configuration
    )
    _persist_environment(output_dir, environment)
    _record_run_status(
      output_dir,
      "running",
      phase="freeze_corpus",
      base_revision=base_revision,
      source_snapshot_sha256=source_snapshot["snapshot_sha256"],
    )

    corpus_path = output_dir / "corpus.json"
    corpus_was_present = corpus_path.exists()

    def candidates_factory():
      _write_json_lines(
        output_dir / "commands.jsonl",
        {
          "kind": "native_candidate_schedule",
          "command": [str(generator_api), "--generate", "<seed>"],
          "start_seed": start_seed,
          "target_count": target_count,
          "max_candidates": max_candidates,
          "timeout_s": candidate_timeout_s,
          "frontend": "symiri --check --tensor-profile-version v2",
        },
      )
      return study.generate_candidate_schedule(
        generator_api,
        target_count=target_count,
        start_seed=start_seed,
        max_candidates=max_candidates,
        timeout_s=candidate_timeout_s,
      )

    corpus = load_or_create_corpus(
      corpus_path,
      config=config,
      base_revision=base_revision,
      target_count=target_count,
      candidates_factory=candidates_factory,
    )
    if not corpus_was_present:
      _write_json_lines(
        output_dir / "commands.jsonl",
        {
          "kind": "frozen_corpus",
          "candidate_enumeration_count": len(corpus["candidate_enumeration"]),
          "selected_kernel_count": len(corpus["kernels"]),
          "corpus_sha256": corpus["corpus_sha256"],
        },
      )

    catalogue = load_or_create_control_catalogue(
      output_dir / "control-catalogue.json", corpus
    )
    ledger_path = output_dir / "study-ledger.json"
    ledger = _load_or_create_ledger(ledger_path, corpus)
    _persist_summary(output_dir, ledger, corpus, run_configuration)

    _record_run_status(
      output_dir,
      "running",
      phase="case_templates",
      corpus_sha256=corpus["corpus_sha256"],
      proposal_count_planned=ledger["proposal_count_planned"],
    )
    templates = {}
    for kernel in corpus["kernels"]:
      templates[kernel["kernel_id"]] = ensure_case_template(
        kernel,
        template_root=output_dir / "templates",
        rysmith=rysmith,
        source_root=source_root,
        symiri=symiri,
        config=config,
        generation_timeout_s=template_timeout_s,
        environment=os.environ.copy(),
      )
    _atomic_json(
      output_dir / "template-index.json",
      {
        "schema": "refractir.tensor-boundary-template-index.v1",
        "corpus_sha256": corpus["corpus_sha256"],
        "templates": templates,
      },
    )

    _record_run_status(
      output_dir,
      "running",
      phase="proposals_and_controls",
      corpus_sha256=corpus["corpus_sha256"],
      proposal_count_completed=sum(
        slot.get("result") is not None for slot in ledger["proposals"]
      ),
    )
    summary = _run_proposals(
      corpus=corpus,
      ledger=ledger,
      catalogue=catalogue,
      templates=templates,
      generator_api=generator_api,
      symiri=symiri,
      output_dir=output_dir,
      ledger_path=ledger_path,
      config=config,
      active_gpu_ids=active_gpu_ids,
      gpu_timeout_s=gpu_timeout_s,
      environment=environment,
      run_configuration=run_configuration,
    )
    complete = summary["pending_proposals"] == 0
    _record_run_status(
      output_dir,
      "complete" if complete else "incomplete",
      phase="finished",
      corpus_sha256=corpus["corpus_sha256"],
      completed_proposals=summary["completed_proposals"],
      planned_proposals=summary["planned_proposals"],
      gpu_work=summary["gpu_work"],
    )
    return summary


def parse_arguments(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--revision", default=os.environ.get("REFRACTIR_STUDY_REVISION"))
  parser.add_argument(
    "--generator-api",
    type=Path,
    default=SOURCE_ROOT / "build" / "bin" / "tensor_input_policy_api",
  )
  parser.add_argument("--rysmith", type=Path, default=SOURCE_ROOT / "rysmith")
  parser.add_argument("--symiri", type=Path, default=SOURCE_ROOT / "symiri")
  parser.add_argument("--symirc", type=Path, default=SOURCE_ROOT / "symirc")
  parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
  parser.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED)
  parser.add_argument("--max-candidates", type=int)
  parser.add_argument(
    "--gpu-ids", default=os.environ.get("REFRACTIR_STUDY_GPU_IDS", "0,1,2,3")
  )
  parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
  parser.add_argument("--candidate-timeout", type=float, default=30)
  parser.add_argument("--template-timeout", type=float, default=120)
  parser.add_argument("--gpu-timeout", type=float, default=DEFAULT_GPU_TIMEOUT_S)
  parser.add_argument(
    "--triton-revision", default=os.environ.get("REFRACTIR_STUDY_TRITON_REVISION")
  )
  parser.add_argument(
    "--compiler-flags",
    default=os.environ.get(
      "REFRACTIR_STUDY_COMPILER_FLAGS",
      os.environ.get("REFRACTIR_COMPILER_FLAGS", DEFAULT_COMPILER_FLAGS),
    ),
  )
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
    parser.error("--gpu-ids must contain unique nonnegative device ids")
  if args.target_count < 1 or not 0 <= args.start_seed <= 0xFFFFFFFF:
    parser.error("--target-count must be positive and --start-seed must fit uint32")
  if args.workers < 1 or args.workers > len(args.gpu_ids):
    parser.error("--workers must not exceed the number of selected GPU ids")
  if any(
    timeout <= 0
    for timeout in (
      args.candidate_timeout,
      args.template_timeout,
      args.gpu_timeout,
    )
  ):
    parser.error("timeouts must be positive")
  if args.max_candidates is not None and args.max_candidates < args.target_count:
    parser.error("--max-candidates must be at least --target-count")
  effective_candidates = args.max_candidates or args.target_count * 50
  if args.start_seed + effective_candidates - 1 > 0xFFFFFFFF:
    parser.error("candidate seed schedule exceeds uint32")
  return args


def main(argv=None):
  args = parse_arguments(argv)
  try:
    summary = run_study(
      source_root=args.source_root,
      output_dir=args.output_dir,
      revision=args.revision,
      generator_api=args.generator_api,
      rysmith=args.rysmith,
      symiri=args.symiri,
      symirc=args.symirc,
      gpu_ids=args.gpu_ids,
      workers=args.workers,
      target_count=args.target_count,
      start_seed=args.start_seed,
      max_candidates=args.max_candidates,
      candidate_timeout_s=args.candidate_timeout,
      template_timeout_s=args.template_timeout,
      gpu_timeout_s=args.gpu_timeout,
      triton_revision=args.triton_revision,
      compiler_flags=args.compiler_flags,
    )
  except KeyboardInterrupt:
    try:
      with _study_lock(args.output_dir):
        _record_run_status(
          args.output_dir,
          "interrupted",
          reason="user interruption; proposal journals and GPU outputs are retained",
        )
    except RuntimeError:
      pass
    return 130
  except Exception as error:
    if args.output_dir.exists():
      try:
        with _study_lock(args.output_dir):
          _record_run_status(args.output_dir, "failed", reason=str(error))
      except RuntimeError:
        pass
    raise
  print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
  return 0 if summary["pending_proposals"] == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
