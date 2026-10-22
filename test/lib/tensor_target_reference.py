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

COMPLETED = "completed"
UB = "ub"
UNAVAILABLE = "unavailable"
FAILED = "failed"
TIMED_OUT = "timed_out"
STATUSES = frozenset({COMPLETED, UB, UNAVAILABLE, FAILED, TIMED_OUT})

PROFILE_VERSIONS = {"tensor-v2": "v2", "tensor-v3": "v3", "tensor-v4": "v4"}
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
    return self.status != UNAVAILABLE

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


def observe(case, *, symiri, timeout_s=120.0, output_path=None):
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
  try:
    completed = subprocess.run(
      command,
      capture_output=True,
      text=True,
      check=False,
      timeout=timeout_s,
    )
  except subprocess.TimeoutExpired as error:
    return ReferenceObservation(
      TIMED_OUT,
      stderr=_captured(error.stderr),
      reason=f"timed out after {timeout_s:g}s",
      command=tuple(command),
    )
  except OSError as error:
    return ReferenceObservation(
      UNAVAILABLE, reason=f"cannot start symiri: {error}", command=tuple(command)
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
