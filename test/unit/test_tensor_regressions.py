"""Combined tensor test coverage: test_tensor_regressions.py, test_tensor_historical_replay.py."""

import unittest

# Consolidated from test_tensor_regressions.py.
try:
  from test.lib import build_tensor_regression_cases as regression_cases
except ImportError:
  regression_cases = None


class TensorRegressionMatrixTests(unittest.TestCase):
  def setUp(self):
    self.assertIsNotNone(regression_cases, "missing tensor regression matrix builder")

  def test_required_shape_matrix(self):
    expected = {
      (2, 2),
      (2, 4),
      (4, 2),
      (1, 4),
      (4, 1),
      (2, 2, 4),
    }
    self.assertEqual(set(regression_cases.required_shapes()), expected)

  def test_softmax_covers_every_axis_and_negative_spelling(self):
    cases = regression_cases.softmax_sweep()
    for shape in regression_cases.required_shapes():
      for axis in range(len(shape)):
        spellings = {
          case["axis_spelling"]
          for case in cases
          if tuple(case["shape"]) == tuple(shape) and case["axis"] == axis
        }
        self.assertEqual(spellings, {str(axis), str(axis - len(shape))})

  def test_transpose_sweep_covers_every_permutation_per_rank(self):
    cases = regression_cases.transpose_sweep()
    for rank in (1, 2, 3):
      expected = set(regression_cases.permutations_for_rank(rank))
      actual = {
        tuple(case["permutation"]) for case in cases if len(case["shape"]) == rank
      }
      self.assertEqual(actual, expected)

  def test_shape_changing_compositions_have_both_orders(self):
    pairs = regression_cases.composition_pairs()
    required = {
      "transpose-softmax",
      "sum-expand",
      "broadcast-softmax",
      "transpose-sum-expand",
    }
    self.assertEqual({pair["id"] for pair in pairs}, required)
    for pair in pairs:
      self.assertEqual(len(pair["orders"]), 2, pair["id"])

  def test_historical_issue_case_records_legacy_invocation(self):
    case = regression_cases.historical_case()
    self.assertEqual(case["input_values"], [0.0, 2.0, 3.0, 1.0])
    self.assertEqual(case["shape"], [2, 2])
    self.assertEqual(case["axis"], 1)
    self.assertEqual(case["legacy_keep_dims"], False)
    self.assertEqual(case["corrected_keep_dims"], None)


if __name__ == "__main__":
  unittest.main()

# Consolidated from test_tensor_historical_replay.py.
try:
  from test.lib import replay_triton_11186 as replay
except ImportError:
  replay = None


class HistoricalReplayTests(unittest.TestCase):
  def setUp(self):
    self.assertIsNotNone(replay, "missing PR #11186 replay support")

  def test_affected_mismatch_and_corrected_pass_satisfy_calibration(self):
    self.assertEqual(
      replay.calibration_status(
        {"status": "numeric_mismatch", "revision": "affected", "campaign_valid": True},
        {"status": "pass", "revision": "corrected", "campaign_valid": True},
        triage_confirmed=True,
      ),
      "calibrated",
    )

  def test_two_passing_builds_do_not_satisfy_calibration(self):
    self.assertEqual(
      replay.calibration_status(
        {"status": "pass", "revision": "affected", "campaign_valid": True},
        {"status": "pass", "revision": "corrected", "campaign_valid": True},
        triage_confirmed=True,
      ),
      "not_calibrated",
    )

  def test_revision_runs_use_separate_artifact_directories(self):
    affected = replay.revision_artifact_dir("/tmp/replay", "affected")
    corrected = replay.revision_artifact_dir("/tmp/replay", "corrected")
    self.assertNotEqual(affected, corrected)
    self.assertIn("affected", affected.parts)
    self.assertIn("corrected", corrected.parts)

  def test_source_overlay_cannot_satisfy_full_build_calibration(self):
    self.assertEqual(
      replay.calibration_status(
        {"status": "numeric_mismatch", "revision": "overlay"},
        {"status": "pass", "revision": "corrected"},
        triage_confirmed=True,
        source_overlay=True,
      ),
      "source_overlay",
    )

  def test_development_and_release_are_checked_independently(self):
    report = replay.compare_pinned_runs(
      {"status": "pass", "revision": "development"},
      {"status": "numeric_mismatch", "revision": "release"},
    )
    self.assertEqual(report["development_status"], "pass")
    self.assertEqual(report["release_status"], "numeric_mismatch")
    self.assertTrue(report["version_difference"])

  def test_affected_and_corrected_labels_must_be_distinct(self):
    with self.assertRaisesRegex(ValueError, "distinct revision labels"):
      replay.replay_pair(
        "/unused/case",
        "/unused/output",
        {"revision": "same"},
        {"revision": "same"},
      )


if __name__ == "__main__":
  unittest.main()
