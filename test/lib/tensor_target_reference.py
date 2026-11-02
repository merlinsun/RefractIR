"""One diagnostic SIR-interpreter observation for a target-only bundle.

The target-only route carries no expected bytes and makes no reference claim.
This observer still runs the RefractIR interpreter over the bundle's own
`inputs.bin`, so a completed run can be described against a third
implementation; every record it produces is marked `diagnostic` so no reader
mistakes the observation for an acceptance or a tolerance.
"""

from __future__ import annotations

import dataclasses
import hashlib
import subprocess
from pathlib import Path

from test.lib.tensor_execution import DispatchRefused

COMPLETED = "completed"
NOT_RUN = "not_run"
UB = "ub"
UNAVAILABLE = "unavailable"
FAILED = "failed"
TIMED_OUT = "timed_out"
STATUSES = frozenset({COMPLETED, UB, UNAVAILABLE, FAILED, TIMED_OUT, NOT_RUN})

PROFILE_VERSIONS = {
  "tensor-v2": "v2",
  "tensor-v3": "v3",
  "tensor-v4": "v4",
  "tensor-v5": "v5",
}
UB_EXIT_CODE = 5


@dataclasses.dataclass(frozen=True)
class ReferenceObservation:
  """One P0 observation; only `completed` carries output bytes."""

  status: str
  output: bytes | None = None
  exit_code: int | None = None
  stderr: str = ""
  reason: str = ""
  command: tuple[str, ...] = ()
  path: str = "refractir_interpreter"

  @property
  def completed(self):
    return self.status == COMPLETED

  @property
  def available(self):
    return self.status not in (UNAVAILABLE, NOT_RUN)

  def as_record(self):
    return {
      "kind": "diagnostic",
      "path": self.path,
      "status": self.status,
      "exit_code": self.exit_code,
      "output_sha256": (
        None if self.output is None else hashlib.sha256(self.output).hexdigest()
      ),
      "reason": self.reason,
    }


def observe(case, *, symiri, timeout_s=120.0, output_path=None, on_process=None):
  """Run `symiri` over the bundle's inputs and classify the observation.

  `case` only needs `path` and `record`; a loaded `TargetCase` qualifies, and a
  lightweight stand-in is enough for a unit test. The bundle is never modified.
  """
  profile = case.record.get("profile")
  version = PROFILE_VERSIONS.get(profile)
  if version is None:
    return ReferenceObservation(
      UNAVAILABLE, reason=f"unsupported tensor profile {profile!r}"
    )
  bundle = Path(case.path)
  inputs = bundle / "inputs.bin"
  if not inputs.is_file():
    return ReferenceObservation(UNAVAILABLE, reason="the bundle has no inputs.bin")
  output = Path(output_path) if output_path is not None else bundle / "runs" / "reference" / "reference.bin"
  output.parent.mkdir(parents=True, exist_ok=True)
  output.unlink(missing_ok=True)
  entry = case.record.get("entry", "kernel")
  command = [
    str(symiri),
    "--tensor-profile-version",
    version,
    "--entry",
    str(entry),
    "--tensor-input",
    str(inputs),
    "--tensor-output",
    str(output),
    str(bundle / "program.sir"),
  ]
  if on_process is not None:
    try:
      on_process({"event": "reserved", "mode": "reference", "command": list(command)})
    except DispatchRefused as error:
      return ReferenceObservation(NOT_RUN, reason=str(error), command=tuple(command))
  try:
    process = subprocess.Popen(
      command,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
    )
  except OSError as error:
    if on_process is not None:
      on_process(
        {"event": "finish", "mode": "reference", "status": "spawn_failed", "reason": str(error)}
      )
    return ReferenceObservation(
      UNAVAILABLE, reason=f"cannot start symiri: {error}", command=tuple(command)
    )
  try:
    if on_process is not None:
      on_process({"event": "start", "mode": "reference", "pid": process.pid})
    stdout, stderr_text = process.communicate(timeout=timeout_s)
  except subprocess.TimeoutExpired:
    process.kill()
    _, stderr_text = process.communicate()
    try:
      if on_process is not None:
        on_process({"event": "finish", "mode": "reference", "status": "timed_out"})
    except Exception:
      pass
    return ReferenceObservation(
      TIMED_OUT,
      stderr=_captured(stderr_text),
      reason=f"timed out after {timeout_s:g}s",
      command=tuple(command),
    )
  except BaseException:
    if process.poll() is None:
      process.kill()
    try:
      process.communicate(timeout=5)
    except Exception:
      pass
    try:
      if on_process is not None:
        on_process({"event": "finish", "mode": "reference", "status": "interrupted"})
    except Exception:
      pass
    raise
  completed = subprocess.CompletedProcess(
    command, process.returncode, stdout, stderr_text
  )
  if on_process is not None:
    on_process(
      {
        "event": "finish",
        "mode": "reference",
        "status": "completed" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
      }
    )
  stderr = completed.stderr or ""
  if completed.returncode == 0:
    data = output.read_bytes() if output.is_file() else None
    if data is None:
      return ReferenceObservation(
        FAILED,
        exit_code=0,
        stderr=stderr,
        reason="symiri exited without writing an output",
        command=tuple(command),
      )
    return ReferenceObservation(
      COMPLETED, output=data, exit_code=0, stderr=stderr, command=tuple(command)
    )
  status = UB if _reports_ub(completed.returncode, stderr) else FAILED
  return ReferenceObservation(
    status,
    exit_code=completed.returncode,
    stderr=stderr,
    reason=_first_line(stderr),
    command=tuple(command),
  )


def _reports_ub(exit_code, stderr):
  return exit_code == UB_EXIT_CODE or bool(_ub_line(stderr))


def _ub_line(stderr):
  for line in stderr.splitlines():
    stripped = line.strip()
    if stripped.startswith("UB:"):
      return stripped
  return ""


def _first_line(text):
  for line in text.splitlines():
    if line.strip():
      return line.strip()
  return ""


def _captured(value):
  if value is None:
    return ""
  if isinstance(value, (bytes, bytearray)):
    return value.decode("utf-8", errors="replace")
  return value
