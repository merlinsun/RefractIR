"""Contract tests for the interpreter-only collapse diagnostics."""

import struct
import unittest

from test.lib import tensor_state_diagnostics as diagnostics

ZERO = b"\0\0\0\0" * 4


def payload(value, count=4):
  return struct.pack("<f", value) * count


ONE = payload(1.0)


class VariantTests(unittest.TestCase):
  def test_variants_are_the_combined_output_then_each_state(self):
    names = [variant["name"] for variant in diagnostics.variants(3)]
    self.assertEqual(names, ["combined", "state-0", "state-1", "state-2"])
    self.assertEqual(
      [variant["return"] for variant in diagnostics.variants(2)],
      ["combined", "state:0", "state:1"],
    )

  def test_every_variant_meets_every_trip_count_and_input(self):
    observations = diagnostics.observations(3)
    self.assertEqual(len(observations), 4 * 3 * 3)
    self.assertEqual(
      len({(variant["name"], trip, index) for variant, trip, index in observations}),
      len(observations),
    )


def run_table(state_count, values):
  """A runner whose output is stated per variant, trip count, and input."""
  calls = []

  def run(variant, trip_count, input_index):
    calls.append((variant["name"], trip_count, input_index))
    entry = values.get((variant["name"], trip_count, input_index))
    if entry is None:
      return "runtime_failure", None
    return "completed", entry

  return run, calls


class CollapseLabelTests(unittest.TestCase):
  def test_constant_across_tested_inputs_is_observed(self):
    constant = {
      (name, trip, index): payload(2.0)
      for name in ("combined", "state-0")
      for trip in diagnostics.TRIP_COUNTS
      for index in range(diagnostics.INPUT_COUNT)
    }
    run, _ = run_table(1, constant)
    record = diagnostics.diagnose(1, run)
    self.assertTrue(record["labels"][diagnostics.CONSTANT_ON_TESTED_INPUTS]["observed"])
    self.assertEqual(
      record["labels"][diagnostics.CONSTANT_ON_TESTED_INPUTS]["variants"]["combined"],
      list(diagnostics.TRIP_COUNTS),
    )
    varying = dict(constant)
    varying[("combined", 1, 1)] = payload(5.0)
    run, _ = run_table(1, varying)
    record = diagnostics.diagnose(1, run)
    self.assertEqual(
      record["labels"][diagnostics.CONSTANT_ON_TESTED_INPUTS]["variants"]["combined"], [2, 4]
    )

  def test_unchanged_across_tested_trip_counts_is_observed(self):
    constant = {
      (name, trip, index): payload(3.0)
      for name in ("combined", "state-0")
      for trip in diagnostics.TRIP_COUNTS
      for index in range(diagnostics.INPUT_COUNT)
    }
    run, _ = run_table(1, constant)
    record = diagnostics.diagnose(1, run)
    self.assertTrue(record["labels"][diagnostics.UNCHANGED_ON_TESTED_TRIP_COUNTS]["observed"])
    changed = dict(constant)
    changed[("state-0", 4, 0)] = payload(9.0)
    run, _ = run_table(1, changed)
    record = diagnostics.diagnose(1, run)
    self.assertEqual(
      record["labels"][diagnostics.UNCHANGED_ON_TESTED_TRIP_COUNTS]["variants"]["state-0"],
      [1, 2],
    )

  def test_a_state_zero_on_the_tested_inputs_is_observed(self):
    values = {
      ("combined", trip, index): ONE
      for trip in diagnostics.TRIP_COUNTS
      for index in range(diagnostics.INPUT_COUNT)
    }
    values.update(
      {
        ("state-0", trip, index): ZERO
        for trip in diagnostics.TRIP_COUNTS
        for index in range(diagnostics.INPUT_COUNT)
      }
    )
    run, _ = run_table(1, values)
    record = diagnostics.diagnose(1, run)
    self.assertTrue(record["labels"][diagnostics.STATE_ZERO_ON_TESTED_INPUTS]["observed"])
    self.assertEqual(record["labels"][diagnostics.STATE_ZERO_ON_TESTED_INPUTS]["states"], ["state-0"])
    nonzero = dict(values)
    nonzero[("state-0", 2, 2)] = ONE
    run, _ = run_table(1, nonzero)
    record = diagnostics.diagnose(1, run)
    self.assertFalse(record["labels"][diagnostics.STATE_ZERO_ON_TESTED_INPUTS]["observed"])

  def test_hidden_combination_is_observed_only_when_every_observation_matches(self):
    values = {
      ("combined", trip, index): payload(index + 1.0)
      for trip in diagnostics.TRIP_COUNTS
      for index in range(diagnostics.INPUT_COUNT)
    }
    values.update(
      {
        ("state-0", trip, index): payload(index + 1.0)
        for trip in diagnostics.TRIP_COUNTS
        for index in range(diagnostics.INPUT_COUNT)
      }
    )
    values.update(
      {
        ("state-1", trip, index): payload(7.0, count=3)
        for trip in diagnostics.TRIP_COUNTS
        for index in range(diagnostics.INPUT_COUNT)
      }
    )
    run, _ = run_table(2, values)
    record = diagnostics.diagnose(2, run)
    self.assertTrue(record["labels"][diagnostics.COMBINATION_HIDDEN_ON_TESTED_INPUTS]["observed"])
    self.assertEqual(
      record["labels"][diagnostics.COMBINATION_HIDDEN_ON_TESTED_INPUTS]["states"], ["state-0"]
    )
    partial = dict(values)
    partial[("combined", 4, 2)] = payload(11.0)
    run, _ = run_table(2, partial)
    record = diagnostics.diagnose(2, run)
    self.assertFalse(record["labels"][diagnostics.COMBINATION_HIDDEN_ON_TESTED_INPUTS]["observed"])

  def test_a_failed_variant_is_a_candidate_and_not_label_evidence(self):
    values = {
      (name, trip, index): payload(2.0)
      for name in ("combined", "state-0")
      for trip in diagnostics.TRIP_COUNTS
      for index in range(diagnostics.INPUT_COUNT)
    }
    del values[("combined", 2, 0)]
    run, calls = run_table(1, values)
    record = diagnostics.diagnose(1, run)
    self.assertEqual(len(record["candidates"]), 1)
    self.assertEqual(record["candidates"][0]["variant"], "combined")
    self.assertEqual(
      record["labels"][diagnostics.CONSTANT_ON_TESTED_INPUTS]["variants"]["combined"], [1, 4]
    )
    self.assertEqual(len(calls), len(diagnostics.observations(1)))


def window_runner(values=None, failures=None, states=None):
  """A callback for window_screen with per-run status and per-state bytes."""
  values = dict(values or {})
  failures = dict(failures or {})
  states = dict(states or {})
  calls = []

  def run(probe, trip_count, variant):
    calls.append((probe, trip_count, variant))
    if (probe, trip_count, variant) in failures:
      return failures[(probe, trip_count, variant)]
    if variant == "combined":
      return "completed", values[(probe, trip_count)]
    return "completed", states.get((probe, trip_count, variant), payload(1.0))

  return run, calls


class WindowScreenTests(unittest.TestCase):
  def test_the_schedule_has_six_base_runs_and_a_ten_run_cap(self):
    schedule = diagnostics.window_probe_schedule(4)
    self.assertEqual(len(schedule), 6)
    self.assertEqual(
      [(item["probe"], item["trip_count"]) for item in schedule],
      [
        ("i0-base", 4), ("i1-read-plus-one", 4), ("i2-late-row", 4),
        ("i3-unused-rows", 4), ("i0-base", 3), ("i2-late-row", 3),
      ],
    )
    self.assertEqual(diagnostics.MAX_WINDOW_PROBE_RUNS, 10)
    with self.assertRaises(diagnostics.DiagnosticsError):
      diagnostics.window_probe_schedule(1)

  def test_a_sensitive_body_is_selected_without_conditional_runs(self):
    run, calls = window_runner(
      values={
        ("i0-base", 4): payload(1.0),
        ("i1-read-plus-one", 4): payload(2.0),
        ("i2-late-row", 4): payload(3.0),
        ("i3-unused-rows", 4): payload(1.0),
        ("i0-base", 3): payload(0.0),
        ("i2-late-row", 3): payload(3.0),
      }
    )
    record = diagnostics.window_screen(trip_count=4, run_probe=run)
    self.assertTrue(record["selection"]["selected"])
    self.assertIsNone(record["selection"]["reason"])
    self.assertEqual(record["conditional_runs"], 0)
    self.assertEqual(len(calls), 6)
    self.assertEqual(record["candidates"], [])
    self.assertFalse(record["labels"]["unused_rows_changed"])

  def test_an_insensitive_body_is_a_selection_failure(self):
    run, calls = window_runner(
      values={
        ("i0-base", 4): payload(1.0),
        ("i1-read-plus-one", 4): payload(1.0),
        ("i2-late-row", 4): payload(3.0),
        ("i3-unused-rows", 4): payload(1.0),
        ("i0-base", 3): payload(0.0),
        ("i2-late-row", 3): payload(3.0),
      },
      states={
        ("i0-base", 4, "state-0"): payload(1.0),
        ("i1-read-plus-one", 4, "state-0"): payload(2.0),
      },
    )
    record = diagnostics.window_screen(trip_count=4, run_probe=run)
    self.assertFalse(record["selection"]["selected"])
    self.assertEqual(record["selection"]["reason"], diagnostics.NOT_SELECTED_INVARIANT)
    self.assertEqual(record["conditional_runs"], 4)
    self.assertLessEqual(len(calls), 10)
    self.assertIn(
      {"probe": "i1-read-plus-one", "variant": "state-0"},
      record["labels"]["hidden_contribution"],
    )

  def test_a_failed_run_is_never_collapse_evidence(self):
    run, calls = window_runner(
      values={
        ("i0-base", 4): payload(1.0),
        ("i1-read-plus-one", 4): payload(2.0),
        ("i2-late-row", 4): payload(3.0),
        ("i3-unused-rows", 4): payload(1.0),
        ("i0-base", 3): payload(0.0),
        ("i2-late-row", 3): payload(3.0),
      },
      failures={("i1-read-plus-one", 4, "combined"): ("runtime_failure", None)},
    )
    record = diagnostics.window_screen(trip_count=4, run_probe=run)
    self.assertFalse(record["selection"]["selected"])
    self.assertEqual(record["selection"]["reason"], diagnostics.NOT_SELECTED_INCOMPLETE)
    self.assertEqual(len(record["candidates"]), 1)
    self.assertEqual(record["candidates"][0]["status"], "runtime_failure")

  def test_unused_rows_changing_is_a_candidate(self):
    run, _ = window_runner(
      values={
        ("i0-base", 4): payload(1.0),
        ("i1-read-plus-one", 4): payload(2.0),
        ("i2-late-row", 4): payload(3.0),
        ("i3-unused-rows", 4): payload(4.0),
        ("i0-base", 3): payload(0.0),
        ("i2-late-row", 3): payload(3.0),
      }
    )
    record = diagnostics.window_screen(trip_count=4, run_probe=run)
    self.assertFalse(record["selection"]["selected"])
    self.assertEqual(record["selection"]["reason"], diagnostics.NOT_SELECTED_UNUSED)
    self.assertTrue(record["labels"]["unused_rows_changed"])
    candidates = [c for c in record["candidates"] if c.get("kind") == "unused_rows_changed"]
    self.assertEqual(len(candidates), 1)
    self.assertIn("base_sha256", candidates[0])
    self.assertIn("unused_sha256", candidates[0])

  def test_a_non_finite_result_is_not_a_completed_observation(self):
    infinite = struct.pack("<4f", float("inf"), 1.0, 2.0, 3.0)
    run, _ = window_runner(
      values={
        ("i0-base", 4): payload(1.0),
        ("i1-read-plus-one", 4): infinite,
        ("i2-late-row", 4): payload(3.0),
        ("i3-unused-rows", 4): payload(1.0),
        ("i0-base", 3): payload(0.0),
        ("i2-late-row", 3): payload(3.0),
      }
    )
    record = diagnostics.window_screen(trip_count=4, run_probe=run)
    self.assertEqual(record["candidates"][0]["status"], "non_finite")
    self.assertFalse(record["selection"]["selected"])

  def test_the_old_default_schedule_is_unchanged(self):
    self.assertEqual(len(diagnostics.observations(3)), 4 * 3 * 3)
    self.assertEqual(diagnostics.TRIP_COUNTS, (1, 2, 4))
    self.assertEqual(diagnostics.INPUT_COUNT, 3)


if __name__ == "__main__":
  unittest.main()
