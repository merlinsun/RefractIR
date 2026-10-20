"""Three-path reporting: one reference, one emission, two targets.

The three execution paths are named in the plan:

* P0, the case's SIR executed again by the RefractIR interpreter, classified by
  `test.lib.tensor_reference`;
* P1, the frozen emitted Triton source executed by Triton's interpreter;
* P2, that same source compiled and executed on the GPU.

P1 and P2 are callables this module calls with one shared emission record, so the
two targets cannot drift apart through a second emission. They return a
`PathRun`; what they do inside (compile, launch, guard, canary) is their
business, and the details they report are carried through unchanged.

Execution status, comparison observation and provisional diagnosis are separate
fields. A target that could not run is `not_run` and does not become a failure; a
target that ran and disagreed is an observation, not a verdict. The comparison
between two targets is reported for its raw equality and its screening
acceptance, and it is never turned into `permitted`: two individually acceptable
errors can point in opposite directions, and the configured tolerance is not a
semantic contract.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Callable, Mapping

from test.lib import tensor_compare, tensor_reference

STATUSES = ("completed", "unsupported", "failed", "timed_out", "not_run")

# The candidate outcomes the plan names. A path reports one of these instead of
# a plain failure when the detail says which it is.
CANDIDATE_KINDS = (
  "storage_violation",
  "compile_failure",
  "runtime_failure",
  "timeout",
  "input_mutation",
)

AGREEMENT = "agreement"
REFERENCE_DRIFT = "reference drift"
EMISSION_SUSPECT = "emission suspect"
COMPILED_PATH_CANDIDATE = "compiled-path candidate"
INTERPRETER_CANDIDATE = "interpreter candidate"
PERMITTED_NUMERICAL = "permitted numerical difference"
UNRESOLVED_NUMERICAL = "unresolved numerical difference"
TWO_WAY_PASSED_INTERPRETER_UNAVAILABLE = "two_way_passed_interpreter_unavailable"

DIAGNOSES = (
  AGREEMENT,
  REFERENCE_DRIFT,
  EMISSION_SUSPECT,
  COMPILED_PATH_CANDIDATE,
  INTERPRETER_CANDIDATE,
  PERMITTED_NUMERICAL,
  UNRESOLVED_NUMERICAL,
  TWO_WAY_PASSED_INTERPRETER_UNAVAILABLE,
)


def _digest(data):
  return None if data is None else hashlib.sha256(data).hexdigest()


@dataclasses.dataclass(frozen=True)
class PathRun:
  """One target's outcome, as the target reports it."""

  status: str
  output: bytes | None = None
  detail: Mapping[str, Any] = dataclasses.field(default_factory=dict)

  def __post_init__(self):
    if self.status not in STATUSES:
      raise ValueError(f"path status must be one of {', '.join(STATUSES)}")
    if self.output is not None and not isinstance(self.output, (bytes, bytearray)):
      raise TypeError("a path output must be bytes or None")

  def record(self):
    return {
      "status": self.status,
      "output_bytes": None if self.output is None else len(self.output),
      "output_sha256": _digest(self.output),
      "detail": dict(self.detail),
    }


@dataclasses.dataclass(frozen=True)
class ThreeWayResult:
  """Execution statuses, comparison observations and one provisional diagnosis."""

  execution: Mapping[str, str]
  observations: Mapping[str, str]
  contract: Mapping[str, str]
  diagnosis: str
  candidates: tuple[Mapping[str, Any], ...]
  permitted: bool
  detail: Mapping[str, Any]

  @property
  def interpretation_available(self):
    """Whether P1 ran, which distinguishes a two-way from a three-way result."""
    return self.execution.get("P1") == "completed"

  def record(self):
    return {
      "execution": dict(self.execution),
      "observations": dict(self.observations),
      "contract": dict(self.contract),
      "diagnosis": self.diagnosis,
      "candidates": [dict(candidate) for candidate in self.candidates],
      "permitted": self.permitted,
      "detail": dict(self.detail),
    }


def _observation(first, second):
  """Raw equality of two path outputs, or `unavailable` when one is missing."""
  if first is None or second is None:
    return "unavailable"
  return "equal" if bytes(first) == bytes(second) else "different"


def _acceptance(actual, reference, *, dtype, exact, atol, rtol, zero_sign_exact):
  """The configured comparator's verdict for one target against R."""
  if actual is None or reference is None:
    return "unavailable"
  verdict = tensor_compare.compare(
    bytes(actual),
    bytes(reference),
    dtype,
    exact,
    atol,
    rtol,
    zero_sign_exact=zero_sign_exact,
  )
  return "accepted" if verdict.get("status") == "pass" else "rejected"


def _candidate(kind, path, run):
  candidate = {"kind": kind, "path": path}
  for key in ("slots", "reason", "diagnostic"):
    if key in run.detail:
      candidate[key] = run.detail[key]
  if "slots" not in candidate and "storage_violation" in run.detail:
    candidate["slots"] = run.detail["storage_violation"]
  return candidate


def _candidates(compiled, interpreter):
  """Candidate outcomes a target retained, in the plan's vocabulary."""
  found = []
  for path, run in (("P1", interpreter), ("P2", compiled)):
    if run is None:
      continue
    kind = run.detail.get("candidate_kind")
    if kind in CANDIDATE_KINDS:
      found.append(_candidate(kind, path, run))
      continue
    if run.detail.get("storage_violation"):
      found.append(_candidate("storage_violation", path, run))
    elif run.status == "timed_out":
      found.append(_candidate("timeout", path, run))
    elif run.status == "failed" and run.detail.get("phase") == "compile":
      found.append(_candidate("compile_failure", path, run))
    elif run.status == "failed" and run.detail.get("input_mutation"):
      found.append(_candidate("input_mutation", path, run))
  return tuple(found)


def _diagnose(
  *,
  reference_ok,
  execution,
  observations,
  contract,
  candidates,
  exact,
  declared_contract,
):
  """One provisional diagnosis, following the plan's precedence."""
  if not reference_ok:
    return REFERENCE_DRIFT, False
  if candidates:
    return COMPILED_PATH_CANDIDATE, False
  if execution.get("P2") != "completed":
    return UNRESOLVED_NUMERICAL, False
  r_t = observations["R-T"]
  r_g = observations["R-G"]
  t_g = observations["T-G"]
  if execution.get("P1") != "completed":
    if r_g == "equal" or contract["R-G"] == "accepted":
      return TWO_WAY_PASSED_INTERPRETER_UNAVAILABLE, False
    return UNRESOLVED_NUMERICAL, False
  if r_t == "equal" and r_g == "equal":
    return AGREEMENT, False
  if r_t == "equal" and r_g == "different":
    return COMPILED_PATH_CANDIDATE, False
  if r_t == "different" and r_g == "equal":
    return INTERPRETER_CANDIDATE, False
  if t_g == "equal":
    # Both targets differ from R in the same way: the difference enters before
    # the paths separate, which points at the shared emission.
    return EMISSION_SUSPECT, False
  if declared_contract is not None and contract["R-G"] == "accepted":
    # Only a declared contract that states why a difference is acceptable can
    # make it permitted; the configured screening tolerance cannot.
    return PERMITTED_NUMERICAL, True
  return UNRESOLVED_NUMERICAL, False


def bytes_agree(contract):
  """Whether both targets were rejected by the same configured contract."""
  return contract.get("R-T") == "rejected" and contract.get("R-G") == "rejected"


def run(
  *,
  case,
  reference,
  interpreter: Callable[[Any], PathRun],
  compiled: Callable[[Any], PathRun],
  emission: Any = None,
  dtype: str = "f32",
  exact: bool = False,
  atol=(2, 100000),
  rtol=(2, 1000),
  zero_sign_exact: bool = False,
  declared_contract: str | None = None,
):
  """Run P1 and P2 against an accepted P0 reference and report the matrix.

  Both target callables receive the same `emission` object. When the reference
  was not accepted, neither is called: the plan makes them `not_run`, which is
  not `failed` and not `unsupported`.
  """
  reference_ok = reference.classification == tensor_reference.REFERENCE_OK
  if not reference_ok:
    return ThreeWayResult(
      execution={
        "P0": reference.classification,
        "P1": "not_run",
        "P2": "not_run",
      },
      observations={"R-T": "unavailable", "R-G": "unavailable", "T-G": "unavailable"},
      contract={"R-T": "unavailable", "R-G": "unavailable", "T-G": "not_applicable"},
      diagnosis=REFERENCE_DRIFT,
      candidates=(),
      permitted=False,
      detail={
        "P0": reference.record(),
        "emission": _emission_record(emission),
        "reason": f"the reference replay was {reference.classification}",
      },
    )

  reference_bytes = reference.reference
  interpreter_run = interpreter(emission)
  compiled_run = compiled(emission)

  observations = {
    "R-T": _observation(reference_bytes, interpreter_run.output),
    "R-G": _observation(reference_bytes, compiled_run.output),
    "T-G": _observation(interpreter_run.output, compiled_run.output),
  }
  contract = {
    "R-T": _acceptance(
      interpreter_run.output,
      reference_bytes,
      dtype=dtype,
      exact=exact,
      atol=atol,
      rtol=rtol,
      zero_sign_exact=zero_sign_exact,
    ),
    "R-G": _acceptance(
      compiled_run.output,
      reference_bytes,
      dtype=dtype,
      exact=exact,
      atol=atol,
      rtol=rtol,
      zero_sign_exact=zero_sign_exact,
    ),
    # The configured tolerance is stated against R. Applying it to T-G would
    # assume the two acceptable errors are interchangeable, which the plan
    # explicitly refuses.
    "T-G": "not_applicable",
  }
  candidates = _candidates(compiled_run, interpreter_run)
  execution = {
    "P0": "completed",
    "P1": interpreter_run.status,
    "P2": compiled_run.status,
  }
  diagnosis, permitted = _diagnose(
    reference_ok=reference_ok,
    execution=execution,
    observations=observations,
    contract=contract,
    candidates=candidates,
    exact=exact,
    declared_contract=declared_contract,
  )
  return ThreeWayResult(
    execution=execution,
    observations=observations,
    contract=contract,
    diagnosis=diagnosis,
    candidates=candidates,
    permitted=permitted,
    detail={
      "P0": reference.record(),
      "P1": interpreter_run.record(),
      "P2": compiled_run.record(),
      "declared_contract": declared_contract,
      "emission": _emission_record(emission),
    },
  )


def _emission_record(emission):
  """The shared emission's identity, when the caller supplied one."""
  if emission is None:
    return None
  record = getattr(emission, "record", None)
  if callable(record):
    return record()
  if isinstance(emission, dict):
    return dict(emission)
  return repr(emission)
