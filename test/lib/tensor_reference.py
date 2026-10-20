"""Fresh reference replay and classification for a loaded tensor case.

A case records the reference its expected bytes came from, but a recorded
reference is evidence of an earlier run. This module executes the case's own
`program.sir` over its `inputs.bin` again and classifies what that run produced.
Profile and entry are read from the manifest, so a v3 case with a non-`kernel`
entry replays as that case instead of as a fixed v2 kernel.

The caller names one directory. Every replay writes its complete output there
and nowhere else:

* an exact case runs once, without a reference precision, to `reference.bin`;
* a numerical case runs once per recorded precision, to `reference-<bits>.bin`.

Each replay records its command, exit status, stdout, stderr, output bytes, and
the SHA-256 of those bytes, so a report can cite the run instead of the
directory it came from.

Exactly one classification is returned:

* `reference_ok` -- every replay produced the bytes stored in `expected.bin`. A
  numerical case also needs its precisions to agree with each other.
* `reference_mismatch` -- the produced bytes differ from `expected.bin`.
* `reference_unstable` -- the numerical replays produced different bytes.
* `sir_ub` -- the interpreter reported undefined behaviour. Execution is not
  defined, so no produced bytes can be an accepted reference.
* `reference_unavailable` -- the reference interpreter could not supply a
  replay: it could not be started, the profile is not one this module maps, a
  replay timed out, or a replay exited without writing an output.

The checks apply in that order of precedence: undefined behaviour, then an
incomplete replay set, then disagreement between the numerical replays, then
disagreement with `expected.bin`. A numerical case with one precision available
and one missing is therefore `reference_unavailable`: instability is an
observed disagreement between two outputs, and a replay that never completed
has no output to disagree with.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from test.lib import tensor_case

REFERENCE_OK = "reference_ok"
REFERENCE_UNSTABLE = "reference_unstable"
REFERENCE_UNAVAILABLE = "reference_unavailable"
SIR_UB = "sir_ub"
REFERENCE_MISMATCH = "reference_mismatch"
CLASSIFICATIONS = frozenset(
  {
    REFERENCE_OK,
    REFERENCE_UNSTABLE,
    REFERENCE_UNAVAILABLE,
    SIR_UB,
    REFERENCE_MISMATCH,
  }
)

# The CLI version each recorded profile selects. A profile outside this map is
# a case this module cannot replay, not a reason to fall back to another one.
PROFILE_VERSIONS = {
  tensor_case.PROFILE_V1: "v1",
  tensor_case.PROFILE_V2: "v2",
  tensor_case.PROFILE_V3: "v3",
  tensor_case.PROFILE_V4: "v4",
}

# `ExitCode::UndefinedBehavior` in include/error.hpp. The interpreter also
# writes a diagnostic beginning with "UB:"; either signal identifies the run.
UB_EXIT_CODE = 5
DEFAULT_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class Replay:
  """One interpreter run for a reference, accepted or not."""

  reference_bits: int | None
  command: tuple[str, ...]
  exit_code: int | None
  timed_out: bool
  stdout: str
  stderr: str
  output: bytes | None
  file: str
  error: str | None

  @property
  def sha256(self) -> str | None:
    """The hash of the bytes this run wrote, when it wrote any."""
    if self.output is None:
      return None
    return hashlib.sha256(self.output).hexdigest()

  def record(self) -> dict[str, object]:
    return {
      "reference_bits": self.reference_bits,
      "command": list(self.command),
      "exit_code": self.exit_code,
      "timed_out": self.timed_out,
      "stdout": self.stdout,
      "stderr": self.stderr,
      "file": self.file,
      "output_bytes": None if self.output is None else len(self.output),
      "output_sha256": self.sha256,
      "error": self.error,
    }


@dataclass(frozen=True)
class ReferenceResult:
  """The classification of one fresh replay of a case."""

  case: Path
  profile: str
  entry: str
  mode: str
  directory: Path
  classification: str
  reason: str
  replays: tuple[Replay, ...]
  expected_sha256: str
  reference: bytes | None

  @property
  def accepted(self) -> bool:
    """Whether the replay established the reference targets may use."""
    return self.classification == REFERENCE_OK

  def record(self) -> dict[str, object]:
    reference_sha256 = None
    if self.reference is not None:
      reference_sha256 = hashlib.sha256(self.reference).hexdigest()
    return {
      "case": str(self.case),
      "profile": self.profile,
      "entry": self.entry,
      "comparison_mode": self.mode,
      "directory": str(self.directory),
      "classification": self.classification,
      "reason": self.reason,
      "expected_sha256": self.expected_sha256,
      "reference_bytes": None if self.reference is None else len(self.reference),
      "reference_sha256": reference_sha256,
      "replays": [replay.record() for replay in self.replays],
    }


def replay(
  case,
  *,
  directory,
  symiri=tensor_case.DEFAULT_SYMIRI,
  timeout=DEFAULT_TIMEOUT_S,
):
  """Replay `case` with `symiri` and classify the produced reference."""
  directory = Path(directory)
  directory.mkdir(parents=True, exist_ok=True)
  manifest = case.manifest
  profile = manifest.get("profile")
  entry = manifest.get("entry")
  mode = manifest.get("comparison", {}).get("mode")
  expected_sha256 = hashlib.sha256(case.expected).hexdigest()

  version = PROFILE_VERSIONS.get(profile)
  if version is None:
    return ReferenceResult(
      case=case.path,
      profile=str(profile),
      entry=str(entry),
      mode=str(mode),
      directory=directory,
      classification=REFERENCE_UNAVAILABLE,
      reason=f"unsupported tensor profile {profile!r}",
      replays=(),
      expected_sha256=expected_sha256,
      reference=None,
    )
  if not isinstance(entry, str) or not entry:
    raise ValueError("a loaded case manifest must name its entry")
  precisions = _precisions(case, mode)

  replays = tuple(
    _replay(
      case,
      directory=directory,
      version=version,
      entry=entry,
      bits=bits,
      symiri=symiri,
      timeout=timeout,
    )
    for bits in precisions
  )
  classification, reason = _classify(mode, replays, case.expected, expected_sha256)
  return ReferenceResult(
    case=case.path,
    profile=version,
    entry=entry,
    mode=mode,
    directory=directory,
    classification=classification,
    reason=reason,
    replays=replays,
    expected_sha256=expected_sha256,
    reference=replays[-1].output if classification == REFERENCE_OK else None,
  )


def _precisions(case, mode):
  """The reference precisions this comparison mode declares."""
  if mode == "exact":
    return (None,)
  if mode == "numeric":
    precisions = case.manifest.get("reference", {}).get("precisions")
    if not isinstance(precisions, list) or not precisions:
      raise ValueError("a numerical case must record its reference precisions")
    return tuple(precisions)
  raise ValueError(f"unknown comparison mode {mode!r}")


def _replay(case, *, directory, version, entry, bits, symiri, timeout):
  """Run one reference precision and record it, successful or not."""
  name = "reference.bin" if bits is None else f"reference-{bits}.bin"
  output_path = directory / name
  output_path.unlink(missing_ok=True)
  command = [str(symiri), "--tensor-profile-version", version]
  if bits is not None:
    command.extend(["--tensor-reference-bits", str(bits)])
  command.extend(
    [
      "--entry",
      entry,
      "--tensor-input",
      str(case.path / "inputs.bin"),
      "--tensor-output",
      str(output_path),
      str(case.path / "program.sir"),
    ]
  )
  try:
    result = subprocess.run(
      command,
      capture_output=True,
      text=True,
      check=False,
      timeout=timeout,
    )
  except subprocess.TimeoutExpired as error:
    return Replay(
      reference_bits=bits,
      command=tuple(command),
      exit_code=None,
      timed_out=True,
      stdout=_captured(error.stdout),
      stderr=_captured(error.stderr),
      output=_read(output_path),
      file=name,
      error=f"timed out after {timeout:g}s",
    )
  except OSError as error:
    return Replay(
      reference_bits=bits,
      command=tuple(command),
      exit_code=None,
      timed_out=False,
      stdout="",
      stderr="",
      output=None,
      file=name,
      error=f"could not run {symiri}: {error}",
    )

  output = _read(output_path)
  error_text = None
  if result.returncode != 0:
    detail = _one_line(result.stderr)
    error_text = f"exited with status {result.returncode}"
    if detail:
      error_text += f": {detail}"
  elif output is None:
    error_text = "wrote no output"
  return Replay(
    reference_bits=bits,
    command=tuple(command),
    exit_code=result.returncode,
    timed_out=False,
    stdout=result.stdout,
    stderr=result.stderr,
    output=output,
    file=name,
    error=error_text,
  )


def _classify(mode, replays, expected, expected_sha256):
  """Return the one classification the replays support, with its reason."""
  for replay in replays:
    if _reports_undefined_behavior(replay):
      detail = _ub_diagnostic(replay.stderr) or "no diagnostic text"
      return (
        SIR_UB,
        f"{_replay_label(replay.reference_bits)} reported undefined behaviour: {detail}",
      )

  for replay in replays:
    if replay.timed_out or replay.exit_code != 0 or replay.output is None:
      return (
        REFERENCE_UNAVAILABLE,
        f"{_replay_label(replay.reference_bits)} produced no reference "
        f"({replay.error})",
      )

  if mode == "numeric":
    outputs = {replay.output for replay in replays}
    if len(outputs) != 1:
      detail = ", ".join(
        f"{replay.reference_bits}-bit {replay.sha256}" for replay in replays
      )
      return REFERENCE_UNSTABLE, f"reference replays disagree: {detail}"

  for replay in replays:
    if replay.output != expected:
      return (
        REFERENCE_MISMATCH,
        f"{_replay_label(replay.reference_bits)} produced {replay.sha256}, "
        f"not the stored expected.bin {expected_sha256}",
      )

  if mode == "exact":
    return REFERENCE_OK, "exact reference replay matches expected.bin"
  bits = " and ".join(str(replay.reference_bits) for replay in replays)
  return REFERENCE_OK, f"reference replays at {bits} bits match expected.bin"


def _reports_undefined_behavior(replay):
  """Whether this run states that the interpreter saw undefined behaviour."""
  if replay.exit_code is None:
    return False
  if replay.exit_code == UB_EXIT_CODE:
    return True
  if replay.exit_code == 0:
    return False
  return bool(_ub_diagnostic(replay.stderr))


def _ub_diagnostic(stderr):
  """The interpreter's own UB line, which every UB diagnostic starts with."""
  for line in stderr.splitlines():
    stripped = line.strip()
    if stripped.startswith("UB:"):
      return stripped
  return ""


def _replay_label(bits):
  return "exact reference replay" if bits is None else f"{bits}-bit reference replay"


def _read(path):
  return path.read_bytes() if path.is_file() else None


def _captured(value):
  """Normalize the partial output a timeout carries."""
  if value is None:
    return ""
  if isinstance(value, bytes):
    return value.decode("utf-8", errors="replace")
  return value


def _one_line(text):
  for line in text.splitlines():
    stripped = line.strip()
    if stripped:
      return stripped
  return ""
