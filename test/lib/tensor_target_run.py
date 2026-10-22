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

from test.lib import (
  tensor_emission,
  tensor_execution,
  tensor_target_case,
  tensor_target_reference,
  tensor_three_way,
)

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

# Diagnoses for a run that also has a diagnostic SIR observation. A target
# failure is more specific than an unavailable reference, so it is decided
# first; the reference's own availability is recorded beside the diagnosis.
INVALID_EXECUTION = "invalid_execution"
REFERENCE_UNAVAILABLE = "reference_unavailable"
BOTH_PATHS_FAILED = "both_paths_failed"
COMPILED_PATH_CANDIDATE = "compiled_path_candidate"
INTERPRETER_CANDIDATE = "interpreter_candidate"

# Diagnoses that name triage-worthy evidence even when the two target paths
# happened to agree with each other.
_TRIAGE_DIAGNOSES = frozenset(
  {
    INVALID_EXECUTION,
    BOTH_PATHS_FAILED,
    COMPILED_PATH_CANDIDATE,
    INTERPRETER_CANDIDATE,
    "emission suspect",
    "unresolved numerical difference",
    "reference drift",
  }
)

STORAGE_MARKERS = (
  "wrote outside its logical output mapping",
  "modified output canary slots",
  "modified input allocation or gap canary",
)
UNSUPPORTED_MARKERS = ("unsupported", "not supported")
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
  reason: str = ""
  # Logical output slots that still hold the declared canary. `None` means the
  # case declares no physical layout, so the question does not apply.
  unwritten: list[int] | None = None

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
      "reason": self.reason,
      "unwritten_logical_slots": self.unwritten,
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
  reference: tensor_target_reference.ReferenceObservation | None = field(
    default=None, compare=False
  )
  diagnosis: dict | None = field(default=None, compare=False)
  compiler_identity: dict | None = field(default=None, compare=False)

  @property
  def candidate(self):
    """Whether the run is evidence that needs diagnosis.

    A timeout is an incomplete observation rather than a proved defect, but a
    completed campaign must not lose it: the run did not establish agreement
    and the reason it stopped needs an answer. A diagnosis can also require
    triage when both target paths agreed, for example undefined behaviour in
    the SIR observation or an emission-suspect difference.
    """
    if self.status in {
      TARGET_DISAGREEMENT,
      COMPILE_FAILURE,
      RUNTIME_FAILURE,
      STORAGE_VIOLATION,
      TIMED_OUT,
    }:
      return True
    diagnosis = (self.diagnosis or {}).get("diagnosis")
    return diagnosis in _TRIAGE_DIAGNOSES

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
      "legality": {
        # Three separate facts. The frontend and the input/layout checks were
        # established before the first process started; concrete legality is
        # what the run itself established, and a missing comparison leaves it
        # unresolved rather than accepted.
        "frontend": {
          "status": "accepted",
          "entry": self.case.record["entry"] if self.case is not None else None,
          "signature": self.case.signature if self.case is not None else None,
        },
        "input_and_layout": {
          "status": "valid",
          "guard_envelope": (
            self.case.record["execution"]["guard_envelope"]
            if self.case is not None
            else None
          ),
        },
        "concrete_execution": {
          "status": _concrete_legality(self.status, self.reference)
        },
      },
      "reference": (
        self.reference.as_record() if self.reference is not None else {"kind": "none"}
      ),
      "diagnosis": self.diagnosis,
      "compiler_identity": self.compiler_identity,
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
  cache_dir=None,
  reference=None,
  compiler_identity=None,
):
  """Emit once, run both modes, and record what each produced.

  Raises `tensor_target_case.TargetCaseError` when the bundle itself is not
  executable, before any process starts. A GPU that is not visible yields
  `environment_unavailable`, not a comparison result. `reference` is an
  optional diagnostic SIR observation; when given, the record carries it and a
  path-specific diagnosis, but the bundle's own reference claim stays `none`.
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
      reference=reference,
      compiler_identity=compiler_identity,
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

  expected_output_bytes = case.record["output"]["byte_length"]
  interpreter = _run_mode(
    case,
    "interpret",
    artifacts=artifacts,
    emission_dir=emission_dir,
    emission=emission,
    timeout_s=timeout_s,
    worker_command=worker_command,
    expected_output_bytes=expected_output_bytes,
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
      cache_dir=cache_dir,
      expected_output_bytes=expected_output_bytes,
    )

  status, mismatches = _classify(interpreter, compiled)
  diagnosis = (
    diagnose_with_optional_reference(
      reference=reference, interpreter=interpreter, compiled=compiled
    )
    if reference is not None
    else None
  )
  target = TargetRun(
    bundle=root,
    status=status,
    emission=emission,
    interpreter=interpreter,
    compiled=compiled,
    mismatches=mismatches,
    case=case,
    reference=reference,
    diagnosis=diagnosis,
    compiler_identity=compiler_identity,
  )
  _save(artifacts, target)
  return target


_FAILURE_KINDS = {
  COMPILE_FAILURE: "compile_failure",
  RUNTIME_FAILURE: "runtime_failure",
  STORAGE_VIOLATION: "storage_violation",
  TIMED_OUT: "timeout",
}


def _path_candidates(interpreter, compiled):
  """Target failures with the path that produced each one."""
  found = []
  for path, run in (("P1", interpreter), ("P2", compiled)):
    if run is None:
      continue
    kind = _FAILURE_KINDS.get(run.status)
    if kind is not None:
      found.append({"path": path, "kind": kind, "status": run.status})
  return found


def _comparison(first, second):
  if first is None or second is None:
    return "unavailable"
  return "equal" if first == second else "different"


def diagnose_with_optional_reference(*, reference, interpreter, compiled):
  """Diagnose a run against an optional diagnostic SIR observation.

  Precedence: undefined behaviour first; then any target failure with its
  path; then an unavailable/failed/timed-out observation; only a completed
  observation delegates to the shared three-way comparison. The reference's
  availability is always recorded beside the diagnosis so a candidate verdict
  never hides an unavailable reference.
  """
  p0_availability = reference.status if reference is not None else "absent"
  candidates = _path_candidates(interpreter, compiled)
  if reference is not None and reference.status == tensor_target_reference.UB:
    return {
      "diagnosis": INVALID_EXECUTION,
      "p0_availability": p0_availability,
      "candidates": candidates,
      "observations": {},
    }
  if candidates:
    paths = {candidate["path"] for candidate in candidates}
    if paths == {"P2"}:
      diagnosis = COMPILED_PATH_CANDIDATE
    elif paths == {"P1"}:
      diagnosis = INTERPRETER_CANDIDATE
    else:
      diagnosis = BOTH_PATHS_FAILED
    return {
      "diagnosis": diagnosis,
      "p0_availability": p0_availability,
      "candidates": candidates,
      "observations": {},
    }
  if reference is None or reference.status in {
    tensor_target_reference.UNAVAILABLE,
    tensor_target_reference.FAILED,
    tensor_target_reference.TIMED_OUT,
  }:
    return {
      "diagnosis": REFERENCE_UNAVAILABLE,
      "p0_availability": p0_availability,
      "candidates": candidates,
      "observations": {},
    }
  observations = {
    "R-T": _comparison(reference.output, interpreter.actual),
    "R-G": _comparison(reference.output, compiled.actual),
    "T-G": _comparison(interpreter.actual, compiled.actual),
  }
  contract = {
    "R-T": "accepted" if observations["R-T"] == "equal" else "rejected",
    "R-G": "accepted" if observations["R-G"] == "equal" else "rejected",
    "T-G": "not_applicable",
  }
  diagnosis, _ = tensor_three_way._diagnose(
    reference_ok=True,
    execution={"P1": interpreter.status, "P2": compiled.status},
    observations=observations,
    contract=contract,
    candidates=(),
    exact=False,
    declared_contract=None,
  )
  return {
    "diagnosis": diagnosis,
    "p0_availability": p0_availability,
    "candidates": candidates,
    "observations": observations,
  }


def run_native(
  bundle_dir,
  *,
  kernel,
  launch,
  triton,
  symiri=None,
  timeout_s=tensor_execution.DEFAULT_TIMEOUT_S,
  device_id=0,
  worker_command=None,
  gpu_probe=None,
  artifact_dir=None,
  cache_dir=None,
  record=None,
  reference=None,
  compiler_identity=None,
):
  """Run one frozen Triton kernel through both modes, without an emitter.

  The bundle supplies the signature, inputs, and output layout; the kernel is
  frozen text, so `symirc` is never called. Both modes stage the same bytes and
  the record states the native path, its kernel digest, the launch, and the
  Triton identity. `record` reuses a verified record instead of building one.
  """
  case = tensor_target_case.load(bundle_dir, symiri=symiri)
  root = case.path
  artifacts = Path(artifact_dir) if artifact_dir is not None else root / "native-runs"
  emission_dir = artifacts / "emission"
  _reset_dir(emission_dir)
  staged_layout = tensor_execution.layout_manifest(case)
  (emission_dir / "tensor-layout.json").write_text(
    tensor_execution.canonical_json(staged_layout) + "\n", encoding="utf-8"
  )
  actual_launch = {
    key: case.record["launch"][key]
    for key in ("num_warps", "num_stages", "enable_fp_fusion")
  }
  if dict(launch) != actual_launch:
    raise tensor_emission.EmissionRefused(
      "the native launch must match the bundle's launch"
    )
  if record is None:
    emission = tensor_emission.build_native(
      sir=case.source,
      kernel=kernel,
      layout=staged_layout,
      profile=case.record["profile"],
      entry=case.record["entry"],
      launch=launch,
      triton=triton,
    )
  else:
    emission = tensor_emission.verify(
      record,
      sir=case.source,
      layout=staged_layout,
      profile=case.record["profile"],
      entry=case.record["entry"],
      launch=launch,
      options=dict(record.emitter.options),
      native=True,
      triton=triton,
    )
  expected_output_bytes = case.record["output"]["byte_length"]
  interpreter = _run_mode(
    case,
    "interpret",
    artifacts=artifacts,
    emission_dir=emission_dir,
    emission=emission,
    timeout_s=timeout_s,
    worker_command=worker_command,
    expected_output_bytes=expected_output_bytes,
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
      cache_dir=cache_dir,
      expected_output_bytes=expected_output_bytes,
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
    reference=reference,
    compiler_identity=compiler_identity,
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
  cache_dir=None,
  expected_output_bytes=None,
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
      cache_dir=(
        Path(cache_dir) / "compiled" if mode == "compiled" and cache_dir is not None else None
      ),
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
      reason="the worker exceeded its time budget",
      unwritten=_unwritten_logical_slots(_actual(mode_dir), case.record),
    )
  except OSError as error:
    raise tensor_emission.EmissionRefused(f"cannot start the {mode} worker: {error}") from error
  stderr = _text(completed.stderr)
  log_path.write_text(stderr, encoding="utf-8")
  phase = _phase(mode_dir)
  actual = _actual(mode_dir)
  status, reason = _mode_status(
    mode,
    exit_code=completed.returncode,
    phase=phase,
    stderr=stderr,
    actual=actual,
    expected_output_bytes=expected_output_bytes,
  )
  return ModeRun(
    mode=mode,
    status=status,
    exit_code=completed.returncode,
    phase=phase,
    actual=actual,
    stderr=stderr,
    log=_relative(log_path, artifacts),
    reason=reason,
    unwritten=_unwritten_logical_slots(actual, case.record),
  )


def _unwritten_logical_slots(actual, record):
  """Logical output slots that still hold the declared canary, or None.

  A slot equal to the canary was not written by the kernel, because the worker
  fills the allocation with that pattern before the run. A value that happens
  to equal the canary is therefore reported here rather than failed: the check
  states ownership, and the numerical comparison stays the only status.
  """
  if actual is None:
    return None
  profile = record.get("profile")
  if profile not in ("tensor-v2", "tensor-v3"):
    return None
  canaries = (record.get("execution") or {}).get("output_canary_by_dtype") or {}
  dtype = (record.get("output") or {}).get("dtype")
  canary = canaries.get(dtype)
  if not isinstance(canary, str):
    return None
  try:
    pattern = bytes.fromhex(canary)
  except ValueError:
    return None
  if len(pattern) != 4:
    return None
  return [
    index // 4
    for index in range(0, len(actual) - 3, 4)
    if actual[index : index + 4] == pattern
  ]


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


def _mode_status(mode, *, exit_code, phase, stderr, actual, expected_output_bytes=None):
  """One mode's status and, for a refusal, why it has that status."""
  lowered = stderr.lower()
  if any(marker in lowered for marker in STORAGE_MARKERS):
    return STORAGE_VIOLATION, ""
  if exit_code == 0:
    if actual is None:
      return RUNTIME_FAILURE, "the run left no output"
    if expected_output_bytes is not None and len(actual) != expected_output_bytes:
      return (
        RUNTIME_FAILURE,
        f"output is {len(actual)} bytes, expected {expected_output_bytes}",
      )
    return COMPLETED, ""
  if mode == "interpret" and any(marker in lowered for marker in UNSUPPORTED_MARKERS):
    return INTERPRETER_UNSUPPORTED, ""
  if phase == "compilation":
    return COMPILE_FAILURE, ""
  return RUNTIME_FAILURE, ""


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
  """The first differing words of two output buffers, as raw bits.

  Every byte of the buffer is compared, including a trailing partial word, so
  two buffers of one length cannot agree on bytes the word loop never read.
  """
  if len(left) != len(right):
    return ((0, f"length={len(left)}", f"length={len(right)}"),)
  differences = []
  for index in range(0, len(left), 4):
    first_bytes = left[index : index + 4]
    second_bytes = right[index : index + 4]
    if first_bytes == second_bytes:
      continue
    first = int.from_bytes(first_bytes.ljust(4, b"\0"), "little")
    second = int.from_bytes(second_bytes.ljust(4, b"\0"), "little")
    differences.append((index // 4, f"0x{first:08x}", f"0x{second:08x}"))
    if len(differences) == MAX_REPORTED_MISMATCHES:
      break
  return tuple(differences)


def _concrete_legality(status, reference=None):
  """What the run established about concrete execution.

  Agreement and disagreement both came from completed executions, so either is
  an established concrete result unless the SIR observation reports undefined
  behaviour: target agreement remains a factual observation, but it cannot
  establish legality for a program whose execution is undefined. A storage
  violation is a violated one. Every incomplete path leaves it unresolved;
  frontend acceptance alone never establishes it.
  """
  if reference is not None and reference.status == tensor_target_reference.UB:
    return "invalid_execution"
  if status in (TARGET_AGREEMENT, TARGET_DISAGREEMENT):
    return "established"
  if status == STORAGE_VIOLATION:
    return "violated"
  return "unresolved"


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
