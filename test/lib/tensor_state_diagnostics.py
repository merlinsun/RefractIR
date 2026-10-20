"""Interpreter-only collapse diagnostics for one free-form region.

A combined output can hide what a state contributes: if a loop body's result is
the same for every input, or it does not change when the trip count does, the
kernel is exercising less of its own computation than the program states. These
checks observe the interpreter only. They report observed behaviour on the
tested inputs and trip counts, not mathematical constancy over the domain.
"""

from dataclasses import dataclass

TRIP_COUNTS = (1, 2, 4)
INPUT_COUNT = 3

CONSTANT_ON_TESTED_INPUTS = "constant_on_tested_inputs"
UNCHANGED_ON_TESTED_TRIP_COUNTS = "unchanged_on_tested_trip_counts"
STATE_ZERO_ON_TESTED_INPUTS = "state_zero_on_tested_inputs"
COMBINATION_HIDDEN_ON_TESTED_INPUTS = "combination_hidden_on_tested_inputs"

COMPLETED = "completed"


class DiagnosticsError(ValueError):
  """The diagnostic request cannot be run as stated."""


@dataclass(frozen=True)
class Observation:
  """One interpreter result for one program variant."""

  variant: str
  trip_count: int
  input_index: int
  status: str
  actual: bytes | None

  def as_record(self):
    return {
      "variant": self.variant,
      "trip_count": self.trip_count,
      "input_index": self.input_index,
      "status": self.status,
      "actual_length": len(self.actual) if self.actual is not None else None,
      "actual_sha256": _sha256(self.actual),
    }


def variants(state_count):
  """The combined output first, then one variant returning each state."""
  if type(state_count) is not int or state_count <= 0:
    raise DiagnosticsError("diagnostics need at least one state")
  result = [{"name": "combined", "return": "combined"}]
  result.extend(
    {"name": f"state-{index}", "return": f"state:{index}"}
    for index in range(state_count)
  )
  return result


def observations(state_count):
  """Every variant under every frozen trip count and input index."""
  return [
    (variant, trip_count, input_index)
    for variant in variants(state_count)
    for trip_count in TRIP_COUNTS
    for input_index in range(INPUT_COUNT)
  ]


def diagnose(state_count, run_program):
  """Run every observation, then report the collapse labels it supports.

  `run_program(variant, trip_count, input_index)` returns `(status, actual)`,
  where `actual` is the raw interpreter output. A variant that did not complete
  is preserved as a candidate; it is never used as evidence for a label.
  """
  runs = []
  for variant, trip_count, input_index in observations(state_count):
    status, actual = run_program(variant, trip_count, input_index)
    if status == COMPLETED and actual is None:
      status = "runtime_failure"
    runs.append(Observation(variant["name"], trip_count, input_index, status, actual))
  return {
    "schema": "refractir.free-form-diagnostics.v1",
    "state_count": state_count,
    "trip_counts": list(TRIP_COUNTS),
    "input_count": INPUT_COUNT,
    "runs": [run.as_record() for run in runs],
    "candidates": [run.as_record() for run in runs if run.status != COMPLETED],
    "labels": collapse_labels(runs, state_count=state_count),
    "note": (
      "These labels describe observed behaviour on the tested inputs and trip "
      "counts. They do not establish constancy over the input domain."
    ),
  }


def collapse_labels(runs, *, state_count):
  """The four observed-behaviour labels the completed runs support."""
  completed = [run for run in runs if run.status == COMPLETED]
  by_variant_trip = {}
  by_variant_input = {}
  for run in completed:
    by_variant_trip.setdefault((run.variant, run.trip_count), []).append(run)
    by_variant_input.setdefault((run.variant, run.input_index), []).append(run)

  constant = {}
  for (variant, trip_count), group in by_variant_trip.items():
    if len(group) == INPUT_COUNT and len({run.actual for run in group}) == 1:
      constant.setdefault(variant, []).append(trip_count)
  unchanged = {}
  for (variant, input_index), group in by_variant_input.items():
    if len(group) == len(TRIP_COUNTS) and len({run.actual for run in group}) == 1:
      unchanged.setdefault(variant, []).append(input_index)

  state_variants = [variant["name"] for variant in variants(state_count)[1:]]
  zero_states = []
  for name in state_variants:
    group = [run for run in completed if run.variant == name]
    if group and all(_all_zero_float(run.actual) for run in group):
      zero_states.append(name)

  hidden = {}
  for name in state_variants:
    state_runs = {
      (run.trip_count, run.input_index): run.actual
      for run in completed
      if run.variant == name
    }
    combined_runs = {
      (run.trip_count, run.input_index): run.actual
      for run in completed
      if run.variant == "combined"
    }
    if combined_runs and combined_runs == state_runs:
      hidden[name] = True

  return {
    CONSTANT_ON_TESTED_INPUTS: {
      "observed": bool(constant),
      "variants": {name: sorted(set(trips)) for name, trips in sorted(constant.items())},
    },
    UNCHANGED_ON_TESTED_TRIP_COUNTS: {
      "observed": bool(unchanged),
      "variants": {
        name: sorted(set(inputs)) for name, inputs in sorted(unchanged.items())
      },
    },
    STATE_ZERO_ON_TESTED_INPUTS: {
      "observed": bool(zero_states),
      "states": sorted(zero_states),
    },
    COMBINATION_HIDDEN_ON_TESTED_INPUTS: {
      "observed": bool(hidden),
      "states": sorted(hidden),
    },
  }


def _all_zero_float(actual):
  if actual is None or len(actual) % 4 != 0:
    return False
  return all(actual[index : index + 4] == b"\0\0\0\0" for index in range(0, len(actual), 4))


def _sha256(actual):
  if actual is None:
    return None
  import hashlib

  return hashlib.sha256(actual).hexdigest()
