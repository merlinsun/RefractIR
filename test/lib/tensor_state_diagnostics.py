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


WINDOW_PROBE_SEQUENCE = (
  ("i0-base", 0, "combined", "base"),
  ("i1-read-plus-one", 0, "combined", "read-sensitivity"),
  ("i2-late-row", 0, "combined", "late-row-sensitivity"),
  ("i3-unused-rows", 0, "combined", "negative-control"),
  ("i0-base", -1, "combined", "trip-comparison"),
  ("i2-late-row", -1, "combined", "trip-comparison"),
)
MAX_WINDOW_PROBE_RUNS = 10
WINDOW_SCREEN_SCHEMA = "refractir.free-form-window-screen.v1"
NOT_SELECTED_INVARIANT = "invariant_to_read_probes"
NOT_SELECTED_UNUSED = "unused_rows_changed"
NOT_SELECTED_TRIP = "trip_invariant"
NOT_SELECTED_INCOMPLETE = "incomplete_observation"


def window_probe_schedule(trip_count):
  """The six base interpreter runs: four probes at T, I0 and I2 at T-1."""
  if type(trip_count) is not int or trip_count < 2:
    raise DiagnosticsError("window screening needs a trip count of at least two")
  runs = []
  for probe, delta, variant, purpose in WINDOW_PROBE_SEQUENCE:
    runs.append(
      {
        "probe": probe,
        "trip_count": trip_count + delta,
        "variant": variant,
        "purpose": purpose,
      }
    )
  return runs


def window_screen(*, trip_count, run_probe):
  """Run the frozen window schedule and report empirical selection facts.

  `run_probe(probe, trip_count, variant)` returns `(status, actual)`. A failed
  or non-finite run is a candidate, never evidence of collapse. At most four
  conditional state-variant runs follow when the base observations do not
  select the body, so the schedule never exceeds ten interpreter launches.
  """
  base_schedule = window_probe_schedule(trip_count)
  runs = []
  for item in base_schedule:
    status, actual = run_probe(item["probe"], item["trip_count"], item["variant"])
    finite = status == COMPLETED and actual is not None and _finite_actual(actual)
    if status == COMPLETED and actual is None:
      status = "runtime_failure"
    if status == COMPLETED and not finite:
      status = "non_finite"
    runs.append(
      {
        **item,
        "status": status,
        "finite": finite,
        "actual": actual,
        "actual_sha256": _sha256(actual),
      }
    )

  def find(probe, delta):
    return next(
      run for run in runs
      if run["probe"] == probe and run["trip_count"] == trip_count + delta
    )

  usable = {
    key: run["status"] == COMPLETED and run["finite"]
    for key, run in {
      "base": find("i0-base", 0),
      "broad": find("i1-read-plus-one", 0),
      "late": find("i2-late-row", 0),
      "unused": find("i3-unused-rows", 0),
      "trip_base": find("i0-base", -1),
      "trip_late": find("i2-late-row", -1),
    }.items()
  }
  base = find("i0-base", 0)
  broad = find("i1-read-plus-one", 0)
  late = find("i2-late-row", 0)
  unused = find("i3-unused-rows", 0)
  trip_base = find("i0-base", -1)
  sensitive_broad = usable["base"] and usable["broad"] and base["actual"] != broad["actual"]
  sensitive_late = usable["base"] and usable["late"] and base["actual"] != late["actual"]
  unused_agrees = usable["base"] and usable["unused"] and base["actual"] == unused["actual"]
  trip_changes = (
    usable["base"] and usable["trip_base"] and base["actual"] != trip_base["actual"]
  )
  selected = (
    sensitive_broad and sensitive_late and unused_agrees and trip_changes
  )

  conditional = []
  if not selected:
    reason = None
    if not (usable["base"] and usable["broad"] and usable["late"]):
      reason = NOT_SELECTED_INCOMPLETE
    elif not unused_agrees:
      reason = NOT_SELECTED_UNUSED
    elif not (sensitive_broad and sensitive_late):
      reason = NOT_SELECTED_INVARIANT
    elif not trip_changes:
      reason = NOT_SELECTED_TRIP
    # Identify a hidden state contribution: compare per-state outputs for the
    # base and the probe that failed (or the unused control, or the trip).
    if reason == NOT_SELECTED_UNUSED:
      probes = [("i0-base", 0), ("i3-unused-rows", 0)]
    elif not (sensitive_broad and sensitive_late):
      failing = (
        "i1-read-plus-one" if not sensitive_broad else "i2-late-row"
      )
      probes = [("i0-base", 0), (failing, 0)]
    elif reason == NOT_SELECTED_TRIP:
      probes = [("i0-base", 0), ("i0-base", -1)]
    else:
      probes = []
    for probe, delta in probes:
      for state in range(2):
        variant = f"state-{state}"
        status, actual = run_probe(probe, trip_count + delta, variant)
        finite = status == COMPLETED and actual is not None and _finite_actual(actual)
        if status == COMPLETED and actual is None:
          status = "runtime_failure"
        if status == COMPLETED and not finite:
          status = "non_finite"
        conditional.append(
          {
            "probe": probe,
            "trip_count": trip_count + delta,
            "variant": variant,
            "purpose": "hidden-contribution",
            "status": status,
            "finite": finite,
            "actual": actual,
            "actual_sha256": _sha256(actual),
          }
        )
  else:
    reason = None

  hidden = []
  for probe in ("i1-read-plus-one", "i2-late-row", "i3-unused-rows"):
    base_state = {
      run["variant"]: run["actual_sha256"]
      for run in conditional
      if run["probe"] == "i0-base" and run["trip_count"] == trip_count
    }
    probe_state = {
      run["variant"]: run["actual_sha256"]
      for run in conditional
      if run["probe"] == probe and run["trip_count"] == trip_count
    }
    for variant in sorted(set(base_state) & set(probe_state)):
      if base_state[variant] != probe_state[variant]:
        hidden.append({"probe": probe, "variant": variant})
  all_runs = runs + conditional
  if len(all_runs) > MAX_WINDOW_PROBE_RUNS:
    raise DiagnosticsError("window screening exceeded its interpreter cap")
  return {
    "schema": WINDOW_SCREEN_SCHEMA,
    "trip_count": trip_count,
    "runs": [
      {key: run[key] for key in
       ("probe", "trip_count", "variant", "purpose", "status", "finite", "actual_sha256")}
      for run in all_runs
    ],
    "selection": {
      "selected": selected,
      "sensitive_to_read_probes": sensitive_broad,
      "sensitive_to_late_row": sensitive_late,
      "unused_rows_unchanged": unused_agrees,
      "changed_with_trips": trip_changes,
      "reason": reason,
    },
    "candidates": [
      {key: run[key] for key in
       ("probe", "trip_count", "variant", "status", "finite", "actual_sha256")}
      for run in all_runs
      if run["status"] != COMPLETED
    ],
    "labels": {
      "hidden_contribution": hidden,
      "unused_rows_changed": bool(usable["base"] and usable["unused"] and
                                  base["actual"] != unused["actual"]),
    },
    "conditional_runs": len(conditional),
    "note": (
      "Empirical selection facts, not a correctness claim: a failed or "
      "non-finite run is never evidence of collapse."
    ),
  }


def _finite_actual(actual):
  if actual is None or len(actual) % 4 != 0:
    return False
  import struct

  for index in range(0, len(actual), 4):
    (value,) = struct.unpack("<f", actual[index : index + 4])
    if not (value == value and abs(value) != float("inf")):
      return False
  return True
