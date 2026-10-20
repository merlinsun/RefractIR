"""Run one target-only bundle through the interpreter and the GPU.

Both modes consume the one frozen Triton source emitted from the bundle's SIR,
which is why the emission record is built once and staged into each mode's
artifact directory. There is no reference and no tolerance: the record states
whether the two paths produced identical bytes, preserves the bytes and the
first differing words when they did not, and classifies a failure by where it
happened rather than treating it as a comparison outcome.
"""

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from test.lib import tensor_emission, tensor_execution, tensor_target_case

SCHEMA = "refractir.tensor-target-run.v1"

# Where a run ended. These are separate facts: a compile failure is not a
# runtime failure, and an unavailable comparison is not an agreement.
COMPLETED = "completed"
COMPILE_FAILURE = "compile_failure"
RUNTIME_FAILURE = "runtime_failure"
TIMED_OUT = "timed_out"
STORAGE_VIOLATION = "storage_violation"
INTERPRETER_UNSUPPORTED = "interpreter_unsupported"
ENVIRONMENT_UNAVAILABLE = "environment_unavailable"
TARGET_AGREEMENT = "target_agreement"
TARGET_DISAGREEMENT = "target_disagreement"
COMPARISON_UNAVAILABLE = "comparison_unavailable"

STORAGE_MARKERS = (
  "wrote outside its logical output mapping",
  "modified output canary slots",
  "modified input allocation or gap canary",
)
UNSUPPORTED_MARKER = "unsupported"
MAX_REPORTED_MISMATCHES = 16


@dataclass(frozen=True)
class ModeRun:
  """What one fresh worker process did."""

  mode: str
  status: str
  exit_code: int | None
  phase: str
  actual: bytes | None
  stderr: str
  log: str

  def as_record(self):
    return {
      "mode": self.mode,
      "status": self.status,
      "exit_code": self.exit_code,
      "phase": self.phase,
      "actual_sha256": (
        hashlib.sha256(self.actual).hexdigest() if self.actual is not None else None
      ),
      "actual_byte_length": len(self.actual) if self.actual is not None else None,
      "log": self.log,
    }


@dataclass(frozen=True)
class TargetRun:
  """One complete interpreter/GPU run of a bundle, with no reference claim."""

  bundle: Path
  status: str
  emission: tensor_emission.EmissionRecord | None
  interpreter: ModeRun
  compiled: ModeRun
  mismatches: tuple = ()
  case: tensor_target_case.TargetCase | None = field(default=None, compare=False)

  @property
  def candidate(self):
    """Whether the run is evidence that needs diagnosis."""
    return self.status in {
      TARGET_DISAGREEMENT,
      COMPILE_FAILURE,
      RUNTIME_FAILURE,
      STORAGE_VIOLATION,
    }

  def as_record(self):
    return {
      "schema": SCHEMA,
      "bundle": str(self.bundle),
      "status": self.status,
      "candidate": self.candidate,
      "emission": self.emission.as_record() if self.emission is not None else None,
      "interpreter": self.interpreter.as_record(),
      "compiled": self.compiled.as_record(),
      "agreement": {
        "kind": _agreement_kind(self.interpreter, self.compiled),
        "mismatches": [
          {"index": index, "interpreter_bits": first, "compiled_bits": second}
          for index, first, second in self.mismatches
        ],
      },
      "reference": {"kind": "none"},
    }


def run(
  bundle_dir,
  *,
  symirc,
  symiri=None,
  timeout_s=tensor_execution.DEFAULT_TIMEOUT_S,
  device_id=0,
  worker_command=None,
  gpu_probe=None,
  artifact_dir=None,
):
  """Emit once, run both modes, and record what each produced.

  Raises `tensor_target_case.TargetCaseError` when the bundle itself is not
  executable, before any process starts. A GPU that is not visible yields
  `environment_unavailable`, not a comparison result.
  """
  case = tensor_target_case.load(bundle_dir, symiri=symiri)
  root = case.path
  artifacts = Path(artifact_dir) if artifact_dir is not None else root / "target-runs"
  emission_dir = artifacts / "emission"
  _reset_dir(emission_dir)
  kernel_path, emit_stderr, emit_stdout = tensor_execution.emit_kernel(
    case, symirc, emission_dir
  )
  (emission_dir / "compiler.log").write_text(
    emit_stdout + emit_stderr, encoding="utf-8"
  )
  if kernel_path is None:
    mode = ModeRun(
      mode="emission",
      status=COMPILE_FAILURE,
      exit_code=None,
      phase="emission",
      actual=None,
      stderr=emit_stderr,
      log="emission/compiler.log",
    )
    target = TargetRun(
      bundle=root,
      status=COMPILE_FAILURE,
      emission=None,
      interpreter=mode,
      compiled=mode,
      case=case,
    )
    _save(artifacts, target)
    return target

  layout_manifest = _layout_manifest(emission_dir)
  options = {
    "target": "triton",
    "profile": case.record["profile"],
    "softmax_lowering": case.record["softmax_lowering"],
    "layout_manifest": layout_manifest,
  }
  emission = tensor_emission.build_ordinary(
    sir=case.source,
    executable=symirc,
    layout=layout_manifest,
    layout_manifest=layout_manifest,
    profile=case.record["profile"],
    entry=case.record["entry"],
    launch=case.record["launch"],
    softmax_lowering=case.record["softmax_lowering"],
    options=options,
    kernel=kernel_path,
  )

  interpreter = _run_mode(
    case,
    "interpret",
    artifacts=artifacts,
    emission_dir=emission_dir,
    emission=emission,
    timeout_s=timeout_s,
    worker_command=worker_command,
  )
  visible = gpu_probe() if gpu_probe is not None else tensor_execution.gpu_available(device_id)
  if not visible:
    compiled = ModeRun(
      mode="compiled",
      status=ENVIRONMENT_UNAVAILABLE,
      exit_code=None,
      phase="unavailable",
      actual=None,
      stderr=f"CUDA device {device_id} unavailable",
      log="",
    )
  else:
    compiled = _run_mode(
      case,
      "compiled",
      artifacts=artifacts,
      emission_dir=emission_dir,
      emission=emission,
      timeout_s=timeout_s,
      worker_command=worker_command,
      device_id=device_id,
    )

  status, mismatches = _classify(interpreter, compiled)
  target = TargetRun(
    bundle=root,
    status=status,
    emission=emission,
    interpreter=interpreter,
    compiled=compiled,
    mismatches=mismatches,
    case=case,
  )
  _save(artifacts, target)
  return target


def _save(artifacts, target):
  tensor_execution.write_atomic(
    artifacts / "target-run.json",
    tensor_execution.canonical_json(target.as_record()).encode("utf-8"),
  )


def _run_mode(
  case,
  mode,
  *,
  artifacts,
  emission_dir,
  emission,
  timeout_s,
  worker_command,
  device_id=None,
):
  """Stage the recorded kernel and run one mode in a fresh process."""
  mode_dir = artifacts / mode
  _reset_dir(mode_dir)
  kernel_source = emission.kernel.source
  (mode_dir / "kernel.py").write_text(kernel_source, encoding="utf-8")
  if hashlib.sha256(kernel_source.encode("utf-8")).hexdigest() != emission.kernel_sha256:
    raise tensor_emission.EmissionRefused("the staged kernel differs from the emission record")
  layout_path = emission_dir / "tensor-layout.json"
  if layout_path.is_file():
    shutil.copy2(layout_path, mode_dir / "tensor-layout.json")

  command = (
    worker_command(case, mode, mode_dir)
    if worker_command is not None
    else [
      sys.executable,
      "-m",
      "test.lib.triton_case_worker",
      "--case",
      str(case.path),
      "--artifacts-dir",
      str(mode_dir),
      "--mode",
      mode,
    ]
  )
  log_path = mode_dir / "worker.log"
  try:
    completed = tensor_execution.run_child(
      command,
      mode,
      timeout_s,
      device_id=device_id,
      cache_dir=(mode_dir / "triton-cache") if mode == "compiled" else None,
    )
  except subprocess.TimeoutExpired as error:
    stderr = _text(error.stderr)
    log_path.write_text(stderr, encoding="utf-8")
    return ModeRun(
      mode=mode,
      status=TIMED_OUT,
      exit_code=None,
      phase=_phase(mode_dir),
      actual=_actual(mode_dir),
      stderr=stderr,
      log=_relative(log_path, artifacts),
    )
  except OSError as error:
    raise tensor_emission.EmissionRefused(f"cannot start the {mode} worker: {error}") from error
  stderr = _text(completed.stderr)
  log_path.write_text(stderr, encoding="utf-8")
  phase = _phase(mode_dir)
  actual = _actual(mode_dir)
  return ModeRun(
    mode=mode,
    status=_mode_status(
      mode, exit_code=completed.returncode, phase=phase, stderr=stderr, actual=actual
    ),
    exit_code=completed.returncode,
    phase=phase,
    actual=actual,
    stderr=stderr,
    log=_relative(log_path, artifacts),
  )


def _text(value):
  if value is None:
    return ""
  return value if isinstance(value, str) else value.decode("utf-8", errors="replace")


def _phase(mode_dir):
  """The worker's last written phase, or an empty string when it wrote none."""
  try:
    return json.loads((mode_dir / "worker-phase.json").read_text(encoding="utf-8"))["phase"]
  except (OSError, ValueError, KeyError, TypeError):
    return ""


def _actual(mode_dir):
  """The raw output bytes a successful run left behind."""
  try:
    return (mode_dir / "actual.bin").read_bytes()
  except OSError:
    return None


def _mode_status(mode, *, exit_code, phase, stderr, actual):
  lowered = stderr.lower()
  if any(marker in lowered for marker in STORAGE_MARKERS):
    return STORAGE_VIOLATION
  if exit_code == 0:
    return COMPLETED if actual is not None else RUNTIME_FAILURE
  if mode == "interpret" and UNSUPPORTED_MARKER in lowered:
    return INTERPRETER_UNSUPPORTED
  if phase == "compilation":
    return COMPILE_FAILURE
  return RUNTIME_FAILURE


def _classify(interpreter, compiled):
  """The overall status, with the first differing words when both completed."""
  if interpreter.status == TIMED_OUT or compiled.status == TIMED_OUT:
    return TIMED_OUT, ()
  if compiled.status == ENVIRONMENT_UNAVAILABLE:
    return ENVIRONMENT_UNAVAILABLE, ()
  for mode in (compiled, interpreter):
    for status in (COMPILE_FAILURE, STORAGE_VIOLATION, RUNTIME_FAILURE):
      if mode.status == status:
        return status, ()
  if interpreter.status == INTERPRETER_UNSUPPORTED:
    return COMPARISON_UNAVAILABLE, ()
  if interpreter.status != COMPLETED or compiled.status != COMPLETED:
    return COMPARISON_UNAVAILABLE, ()
  mismatches = _differences(interpreter.actual, compiled.actual)
  if mismatches:
    return TARGET_DISAGREEMENT, mismatches
  return TARGET_AGREEMENT, ()


def _differences(left, right):
  """The first differing words of two output buffers, as raw bits."""
  if len(left) != len(right):
    return ((0, f"length={len(left)}", f"length={len(right)}"),)
  differences = []
  for index in range(0, min(len(left), len(right)) - 3, 4):
    if left[index : index + 4] == right[index : index + 4]:
      continue
    first = int.from_bytes(left[index : index + 4], "little")
    second = int.from_bytes(right[index : index + 4], "little")
    differences.append((index // 4, f"0x{first:08x}", f"0x{second:08x}"))
    if len(differences) == MAX_REPORTED_MISMATCHES:
      break
  return tuple(differences)


def _agreement_kind(interpreter, compiled):
  if interpreter.status == INTERPRETER_UNSUPPORTED:
    return "unavailable"
  if interpreter.status != COMPLETED or compiled.status != COMPLETED:
    return "unavailable"
  return "identical-bytes" if interpreter.actual == compiled.actual else "differ"


def _layout_manifest(emission_dir):
  """The layout manifest `symirc` consumed, as the emission record names it."""
  try:
    return json.loads((emission_dir / "tensor-layout.json").read_text(encoding="utf-8"))
  except (OSError, ValueError):
    return {}


def _reset_dir(path):
  if path.exists():
    for child in path.iterdir():
      if child.is_dir():
        shutil.rmtree(child)
      else:
        child.unlink()
  else:
    path.mkdir(parents=True, exist_ok=True)


def _relative(path, root):
  try:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
  except ValueError:
    return str(path)
