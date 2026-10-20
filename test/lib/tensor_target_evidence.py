"""Assemble one case's study record from the compiler observer's own results.

The observer in `test.lib.tensor_compiler_observation` measures loop form,
conversion placement, and rematerialization. This module is the study boundary:
it takes those completed observations and copies the evidence a case record has
to keep, instead of measuring again or summarizing the evidence into a verdict.
The PTX backedge counts, the named-pass conversion attribution, and every
confirmed rematerialization entry are carried as the observer reported them; a
section whose observation is missing says `unavailable` and why, so an absent
measurement is never read as a zero.

Research objectives and candidate validity are kept apart. An objective such as
a retained PTX loop or confirmed rematerialization guides the next decision, but
the plan does not let it decide whether an observed defect counts. Candidate
validity is therefore a caller-supplied value this module copies unchanged; no
objective status is read into it, and no validity value is read into an
objective status.

The record's `interpretation_limit` states the same limit the observer states:
these are compiler observations, not proof of semantic correctness.
"""

import copy
import dataclasses
from typing import Mapping

SCHEMA = "refractir.tensor-target-evidence.v1"

OBSERVED = "observed"
REACHED = "reached"
NOT_REACHED = "not_reached"
UNAVAILABLE = "unavailable"
OBJECTIVE_STATUSES = (REACHED, NOT_REACHED, UNAVAILABLE)

# The plan's research objectives. The first four are read from the observations
# below; the last two concern evidence the compiler observer does not produce and
# are supplied by the caller.
RETAINED_PTX_LOOP = "retained_ptx_loop"
IN_LOOP_CONVERSION_CHANGE = "in_loop_conversion_change"
SHARED_MEMORY_OR_SHUFFLE = "shared_memory_or_shuffle_communication"
MASK_AND_STRIDE = "mask_and_stride_evidence"
CONFIRMED_REMATERIALIZATION = "confirmed_rematerialization"
GLUON_MAPPING = "gluon_effective_mapping_and_conversion_survival"
DERIVED_OBJECTIVES = (
  RETAINED_PTX_LOOP,
  IN_LOOP_CONVERSION_CHANGE,
  SHARED_MEMORY_OR_SHUFFLE,
  CONFIRMED_REMATERIALIZATION,
)
SUPPLIED_OBJECTIVES = (MASK_AND_STRIDE, GLUON_MAPPING)
OBJECTIVES = DERIVED_OBJECTIVES + SUPPLIED_OBJECTIVES

_SUPPLIED_LABELS = {
  MASK_AND_STRIDE: "mask and stride",
  GLUON_MAPPING: "Gluon mapping",
}

_INTERPRETATION_LIMIT = (
  "PTX loop form, pass-dump conversions, and rematerialization are compiler "
  "observations, not semantic proof; an unavailable measurement is not a zero "
  "count, and research objectives do not decide candidate validity"
)


@dataclasses.dataclass(frozen=True)
class CandidateValidity:
  """Whether the case is a valid candidate, as the correctness gates decided.

  `valid` is `None` when the gates have not decided. A non-valid candidate needs
  a reason so the record can say which gate refused it. Research objectives are
  deliberately absent from this value: reaching one cannot promote a case, and
  missing one cannot demote it.
  """

  valid: bool | None
  reason: str | None = None

  def __post_init__(self):
    if type(self.valid) is not bool and self.valid is not None:
      raise TypeError("candidate validity must be a bool or None")
    if self.reason is not None and (
      not isinstance(self.reason, str) or not self.reason.strip()
    ):
      raise ValueError("a candidate validity reason must be a nonempty string")
    if self.valid is not True and not self.reason:
      raise ValueError("a non-valid candidate needs a reason")

  def record(self):
    return {"valid": self.valid, "reason": self.reason}


def _unavailable(reason, **fields):
  """An explicit unavailable section: the reason is kept beside absent values."""
  record = {"status": UNAVAILABLE, "reason": reason}
  record.update(fields)
  return record


def _nested_verdict(mapping, section, name):
  """The verdict for `name`, and whether the observation states it at all.

  A stated `None` is not a missing field: it means the observer declined to call
  the stage loop-free, which is why the caller keeps the difference.
  """
  value = mapping.get(section)
  if isinstance(value, Mapping) and name in value:
    return True, value[name]
  return False, None


def _ptx_observation(compiler_observation):
  """The observer's PTX loop and communication counts, copied not re-detected.

  The counts, the loop evidence kind, and the loop presence come from
  `observe_compiler_ir`. A stage the observer did not measure, or a loop verdict
  it declined to state because a branch was unreadable, stays unavailable with
  the reason; neither is reported as a zero backedge count.
  """
  fields = {
    "stage": "ptx",
    "loop_construct_count": None,
    "backedge_count": None,
    "unparsed_branch_count": None,
    "loop_evidence": None,
    "loop_presence": None,
    "loop_status": None,
    "warp_shuffle_count": None,
    "shared_memory_transfer_count": None,
  }
  if compiler_observation is None:
    return _unavailable("no compiler IR observation was supplied", **fields)
  if not isinstance(compiler_observation, Mapping):
    raise TypeError("the compiler IR observation must be a mapping")
  if compiler_observation.get("status") == UNAVAILABLE:
    reason = (
      compiler_observation.get("reason") or "the compiler observation is unavailable"
    )
    return _unavailable(reason, **fields)
  stages = compiler_observation.get("stages")
  if not isinstance(stages, Mapping):
    return _unavailable("the compiler IR observation carries no stages", **fields)
  ptx = stages.get("ptx")
  if ptx is None:
    return _unavailable("the compiler IR observation has no ptx stage", **fields)
  if not isinstance(ptx, Mapping):
    return _unavailable("the ptx stage observation is not a record", **fields)

  counts = {
    name: ptx.get(name)
    for name in (
      "loop_construct_count",
      "backedge_count",
      "unparsed_branch_count",
      "warp_shuffle_count",
      "shared_memory_transfer_count",
    )
  }
  if any(type(value) is not int or value < 0 for value in counts.values()):
    return _unavailable(
      "the ptx stage observation does not report its loop and communication counts",
      **fields,
    )
  found_evidence, evidence = _nested_verdict(
    compiler_observation, "loop_evidence", "ptx"
  )
  found_presence, presence = _nested_verdict(
    compiler_observation, "loop_presence", "ptx"
  )
  if not found_evidence and not found_presence:
    # An archived projection states the verdict on the stage record itself.
    found_evidence = "loop_evidence" in ptx
    found_presence = "loop_presence" in ptx
    evidence = ptx.get("loop_evidence")
    presence = ptx.get("loop_presence")
  if not found_evidence and not found_presence:
    return _unavailable(
      "the compiler observation does not state the ptx loop presence", **fields
    )

  fields.update(
    {
      "loop_construct_count": counts["loop_construct_count"],
      "backedge_count": counts["backedge_count"],
      "unparsed_branch_count": counts["unparsed_branch_count"],
      "loop_evidence": evidence,
      "loop_presence": presence,
      "loop_status": compiler_observation.get("loop_status"),
      "warp_shuffle_count": counts["warp_shuffle_count"],
      "shared_memory_transfer_count": counts["shared_memory_transfer_count"],
    }
  )
  if presence is None:
    reason = (
      "PTX contains branches the loop observer could not classify"
      if counts["unparsed_branch_count"]
      else "the loop observer stated no ptx loop presence"
    )
    return _unavailable(
      f"{reason}; loop presence is not established by this observation",
      **fields,
    )
  return {"status": OBSERVED, "reason": None, **fields}


def _pass_transitions(pass_observation):
  """The measured transitions of the pass dump, or an unavailable section.

  Returns `(reason, transitions)`; `reason` is None when at least one transition
  was measured. A pass dump with no transition gives a reason rather than an
  empty list, because no transition means the conversion and rematerialization
  readings have no before/after pair to stand on.
  """
  if pass_observation is None:
    return "no pass dump observation was supplied", ()
  if not isinstance(pass_observation, Mapping):
    raise TypeError("the pass dump observation must be a mapping")
  if pass_observation.get("status") == UNAVAILABLE:
    return (
      pass_observation.get("reason") or "the pass dump observation is unavailable",
      (),
    )
  compilations = pass_observation.get("compilations")
  if not isinstance(compilations, (list, tuple)) or not compilations:
    return "the pass dump observation carries no compilation", ()
  transitions = []
  for compilation_index, compilation in enumerate(compilations):
    if not isinstance(compilation, Mapping):
      raise TypeError("a pass dump compilation must be a mapping")
    measured = compilation.get("transitions")
    if not isinstance(measured, (list, tuple)):
      raise TypeError("a pass dump compilation's transitions must be a sequence")
    for transition_index, transition in enumerate(measured):
      if not isinstance(transition, Mapping):
        raise TypeError("a pass transition must be a mapping")
      if not isinstance(transition.get("pass"), str) or not transition["pass"]:
        raise ValueError("a pass transition needs a nonempty pass name")
      transitions.append((compilation_index, transition_index, transition))
  if not transitions:
    return "the pass dump has no measurable transition", ()
  return None, tuple(transitions)


def _in_loop_conversions(pass_observation):
  """The named-pass conversion records, each with the loops that enclosed it.

  A removal is attributed to the loop region its own fragment sat in, never to a
  loop that merely exists elsewhere in the module. A move that crosses a loop
  boundary is carried as the observer's full fragment, with the enclosing loop on
  either side. Both count as in-loop conversion work; a pass that saw no loop
  change stays not reached rather than unavailable.
  """
  fields = {
    "passes": [],
    "named_passes": [],
    "in_loop_removed_count": None,
    "in_loop_removed_regions": {},
  }
  reason, transitions = _pass_transitions(pass_observation)
  if reason is not None:
    return _unavailable(reason, **fields)

  passes = []
  removed_regions = {}
  in_loop_removed_count = 0
  for compilation_index, transition_index, transition in transitions:
    removed = transition.get("removed_in_loop_regions") or {}
    if not isinstance(removed, Mapping) or any(
      not isinstance(region, str) or type(count) is not int
      for region, count in removed.items()
    ):
      raise ValueError("in-loop removals must map loop regions to integer counts")
    before = transition.get("before")
    after = transition.get("after")
    before = before if isinstance(before, Mapping) else {}
    after = after if isinstance(after, Mapping) else {}
    movement = transition.get("conversion_movement") or []
    if not isinstance(movement, (list, tuple)) or any(
      not isinstance(fragment, Mapping) for fragment in movement
    ):
      raise ValueError("conversion movement must be a sequence of records")
    moved = [copy.deepcopy(dict(fragment)) for fragment in movement]
    removed_in_loop = sum(removed.values())
    moves_across_a_loop = any(
      fragment.get("enclosing_loop_before") is not None
      or fragment.get("enclosing_loop_after") is not None
      for fragment in moved
    )
    in_loop_removed_count += removed_in_loop
    for region, count in removed.items():
      removed_regions[region] = removed_regions.get(region, 0) + count
    passes.append(
      {
        "compilation_index": compilation_index,
        "transition_index": transition_index,
        "pass": transition["pass"],
        "pipeline": transition.get("pipeline"),
        "next_snapshot_pass": transition.get("next_snapshot_pass"),
        "loop_constructs_before": before.get("loop_construct_count"),
        "loop_constructs_after": after.get("loop_construct_count"),
        "conversion_match": dict(transition.get("conversion_match") or {}),
        "effective_conversion_operation_deltas": dict(
          transition.get("effective_conversion_operation_deltas") or {}
        ),
        "removed_in_loop_regions": dict(removed),
        "removed_in_loop_count": removed_in_loop,
        "removed_outside_a_loop": transition.get("removed_outside_a_loop"),
        "conversion_movement": moved,
        "in_loop": bool(removed_in_loop or moves_across_a_loop),
      }
    )
  return {
    "status": OBSERVED,
    "reason": None,
    "passes": passes,
    "named_passes": sorted({entry["pass"] for entry in passes}),
    "in_loop_removed_count": in_loop_removed_count,
    "in_loop_removed_regions": dict(sorted(removed_regions.items())),
  }


def _rematerialization(pass_observation):
  """The matcher's confirmed entries, carried field for field per pass.

  The confirmed entry holds the newly introduced computation, the layout each
  side resolved to, the redirected consumers, and the surviving original. It is
  copied whole: a summary string would drop exactly the conditions that separate
  a recomputation from a copy, so no entry is reduced here.
  """
  fields = {"observations": [], "confirmed_count": None, "unattributed_count": None}
  reason, transitions = _pass_transitions(pass_observation)
  if reason is not None:
    return _unavailable(reason, **fields)

  observations = []
  confirmed_count = 0
  unattributed_count = 0
  for compilation_index, transition_index, transition in transitions:
    matcher = transition.get("rematerialization")
    if not isinstance(matcher, Mapping):
      continue
    confirmed = matcher.get("confirmed")
    unattributed = matcher.get("unattributed")
    if (
      not isinstance(confirmed, (list, tuple))
      or not isinstance(unattributed, (list, tuple))
      or any(not isinstance(entry, Mapping) for entry in (*confirmed, *unattributed))
    ):
      raise ValueError(
        "a rematerialization observation needs confirmed and unattributed records"
      )
    entry_count = matcher.get("confirmed_count", len(confirmed))
    if type(entry_count) is not int:
      raise ValueError("a rematerialization confirmed count must be an integer")
    observations.append(
      {
        "compilation_index": compilation_index,
        "transition_index": transition_index,
        "pass": transition["pass"],
        "next_snapshot_pass": transition.get("next_snapshot_pass"),
        "status": matcher.get("status"),
        "confirmed": [copy.deepcopy(dict(entry)) for entry in confirmed],
        "confirmed_count": entry_count,
        "unattributed": [copy.deepcopy(dict(entry)) for entry in unattributed],
        "unattributed_count": matcher.get("unattributed_count"),
        "interpretation_limit": matcher.get("interpretation_limit"),
      }
    )
    confirmed_count += entry_count
    unattributed_count += matcher.get("unattributed_count") or 0
  if not observations:
    return _unavailable(
      "the pass dump observation did not run the rematerialization matcher",
      **fields,
    )
  return {
    "status": OBSERVED,
    "reason": None,
    "observations": observations,
    "confirmed_count": confirmed_count,
    "unattributed_count": unattributed_count,
  }


def _objective(status, *, reason=None, evidence=None):
  return {"status": status, "reason": reason, "evidence": evidence}


def _unavailable_objective(section):
  return _objective(UNAVAILABLE, reason=section["reason"])


def _derived_objectives(ptx, in_loop, rematerialization):
  if ptx["status"] != OBSERVED:
    loop_objective = _unavailable_objective(ptx)
    communication_objective = _unavailable_objective(ptx)
  else:
    loop_evidence = {
      "backedge_count": ptx["backedge_count"],
      "loop_construct_count": ptx["loop_construct_count"],
      "loop_evidence": ptx["loop_evidence"],
    }
    loop_objective = _objective(
      REACHED if ptx["loop_presence"] else NOT_REACHED,
      evidence=loop_evidence,
    )
    communication = {
      "warp_shuffle_count": ptx["warp_shuffle_count"],
      "shared_memory_transfer_count": ptx["shared_memory_transfer_count"],
    }
    communication_objective = _objective(
      REACHED
      if ptx["warp_shuffle_count"] or ptx["shared_memory_transfer_count"]
      else NOT_REACHED,
      evidence=communication,
    )

  if in_loop["status"] != OBSERVED:
    in_loop_objective = _unavailable_objective(in_loop)
  else:
    changed = [entry for entry in in_loop["passes"] if entry["in_loop"]]
    in_loop_objective = _objective(
      REACHED if changed else NOT_REACHED,
      evidence=[
        {
          "pass": entry["pass"],
          "removed_in_loop_regions": entry["removed_in_loop_regions"],
          "conversion_movement": entry["conversion_movement"],
        }
        for entry in changed
      ],
    )

  if rematerialization["status"] != OBSERVED:
    rematerialization_objective = _unavailable_objective(rematerialization)
  else:
    rematerialization_objective = _objective(
      REACHED if rematerialization["confirmed_count"] else NOT_REACHED,
      evidence={
        "confirmed_count": rematerialization["confirmed_count"],
        "unattributed_count": rematerialization["unattributed_count"],
        "confirmed_passes": [
          entry["pass"]
          for entry in rematerialization["observations"]
          if entry["confirmed_count"]
        ],
      },
    )
  return {
    RETAINED_PTX_LOOP: loop_objective,
    IN_LOOP_CONVERSION_CHANGE: in_loop_objective,
    SHARED_MEMORY_OR_SHUFFLE: communication_objective,
    CONFIRMED_REMATERIALIZATION: rematerialization_objective,
  }


def _supplied_objectives(objective_evidence):
  if objective_evidence is None:
    objective_evidence = {}
  if not isinstance(objective_evidence, Mapping):
    raise TypeError("objective evidence must be a mapping")
  unknown = sorted(set(objective_evidence) - set(SUPPLIED_OBJECTIVES))
  if unknown:
    raise ValueError(
      "objective evidence is read from the observations, not supplied: "
      + ", ".join(unknown)
    )
  objectives = {}
  for name in SUPPLIED_OBJECTIVES:
    supplied = objective_evidence.get(name)
    if supplied is None:
      objectives[name] = _objective(
        UNAVAILABLE,
        reason=f"no {_SUPPLIED_LABELS[name]} evidence was supplied",
      )
      continue
    if not isinstance(supplied, Mapping):
      raise TypeError(f"{name} evidence must be a mapping")
    status = supplied.get("status")
    if status not in OBJECTIVE_STATUSES:
      raise ValueError(f"{name} status must be one of " + ", ".join(OBJECTIVE_STATUSES))
    reason = supplied.get("reason")
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
      raise ValueError(f"{name} reason must be a nonempty string")
    if status == UNAVAILABLE and reason is None:
      raise ValueError(f"an unavailable {name} objective needs a reason")
    objectives[name] = _objective(
      status, reason=reason, evidence=copy.deepcopy(supplied.get("evidence"))
    )
  return objectives


def assemble_study_record(
  case_id,
  *,
  compiler_observation=None,
  pass_observation=None,
  candidate_validity,
  objective_evidence=None,
):
  """One case's study record from completed compiler observations.

  `compiler_observation` is an `observe_compiler_ir` result and
  `pass_observation` an `observe_pass_dumps` result, each optionally replaced by
  `{"status": "unavailable", "reason": ...}` when the case archive has none.
  Neither observation is measured again here.

  `candidate_validity` is copied to the record unchanged. `objective_evidence`
  supplies only the two objectives the observer does not measure (mask and
  stride evidence, Gluon mapping); asking it for a derived objective is refused,
  because those must come from the observation itself.
  """
  if not isinstance(case_id, str) or not case_id.strip():
    raise ValueError("a study record needs a nonempty case id")
  if not isinstance(candidate_validity, CandidateValidity):
    raise TypeError("candidate validity must be a CandidateValidity")
  ptx = _ptx_observation(compiler_observation)
  in_loop = _in_loop_conversions(pass_observation)
  rematerialization = _rematerialization(pass_observation)
  objectives = _derived_objectives(ptx, in_loop, rematerialization)
  objectives.update(_supplied_objectives(objective_evidence))
  return {
    "schema": SCHEMA,
    "case_id": case_id,
    "candidate_validity": candidate_validity.record(),
    "ptx_loop": ptx,
    "in_loop_conversions": in_loop,
    "rematerialization": rematerialization,
    "research_objectives": {name: objectives[name] for name in OBJECTIVES},
    "interpretation_limit": _INTERPRETATION_LIMIT,
  }
