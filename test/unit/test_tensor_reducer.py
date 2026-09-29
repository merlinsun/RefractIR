"""Behavioral tests for bounded, property-preserving failure reduction."""

import json
import tempfile
import unittest
from pathlib import Path

try:
  from test.lib import tensor_reducer
except ImportError:
  tensor_reducer = None


def make_case(root):
  root = Path(root)
  root.mkdir(parents=True)
  (root / "case.json").write_text(json.dumps({"schema": "test-case"}))
  (root / "program.sir").write_text(
    "fun @kernel(%x: i32) : i32 {\n  let mut %dead: i32 = undef;\n^entry:\n  %dead = 4;\n  ret %x;\n}\n"
  )
  (root / "inputs.bin").write_bytes(b"input")
  (root / "expected.bin").write_bytes(b"expected")
  return root


def mismatch():
  return {
    "campaign_valid": True,
    "status": "exact_mismatch",
    "failure_fingerprint": "backend-bug-1",
    "actual_bytes": b"actual",
  }


class TensorReducerTests(unittest.TestCase):
  def setUp(self):
    self.assertIsNotNone(tensor_reducer, "missing tensor failure reducer")

  def test_reducer_requires_reproducible_failure(self):
    with tempfile.TemporaryDirectory() as directory:
      case = make_case(Path(directory) / "case")
      results = iter((mismatch(), {"campaign_valid": True, "status": "pass"}))
      report = tensor_reducer.reduce_case(
        case,
        output_dir=Path(directory) / "bundle",
        run_target=lambda _path: next(results),
        validate_candidate=lambda _path: {"valid": True},
        transformations=[],
      )

    self.assertEqual(report["status"], "not_reproducible")
    self.assertEqual(report["attempted_transformations"], 0)

  def test_node_removal_is_kept_only_when_failure_persists(self):
    with tempfile.TemporaryDirectory() as directory:
      case = make_case(Path(directory) / "case")

      def remove_dead(candidate):
        source = (candidate / "program.sir").read_text()
        (candidate / "program.sir").write_text(source.replace("  %dead = 4;\n", ""))

      report = tensor_reducer.reduce_case(
        case,
        output_dir=Path(directory) / "bundle",
        run_target=lambda _path: mismatch(),
        validate_candidate=lambda _path: {"valid": True},
        transformations=[{"name": "remove-dead-store", "apply": remove_dead}],
      )

      reduced = Path(directory) / "bundle" / "reduced" / "program.sir"
      self.assertEqual(report["status"], "preserving_simplification")
      self.assertNotIn("%dead = 4", reduced.read_text())
      self.assertTrue((Path(directory) / "bundle" / "replay.sh").is_file())

  def test_undefined_or_invalid_shape_candidate_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      case = make_case(Path(directory) / "case")

      def invalidate(candidate):
        (candidate / "program.sir").write_text("invalid shape axis")

      report = tensor_reducer.reduce_case(
        case,
        output_dir=Path(directory) / "bundle",
        run_target=lambda _path: mismatch(),
        validate_candidate=lambda _path: {
          "valid": False,
          "reason": "invalid tensor axis",
        },
        transformations=[{"name": "bad-axis", "apply": invalidate}],
      )

      self.assertEqual(report["status"], "no_preserving_simplification_found")
      self.assertEqual(report["trace"][0]["validation"], "rejected")
      self.assertEqual(
        (Path(directory) / "bundle" / "reduced" / "program.sir").read_text(),
        (case / "program.sir").read_text(),
      )

  def test_property_or_tolerance_failure_is_rejected_by_validation(self):
    with tempfile.TemporaryDirectory() as directory:
      case = make_case(Path(directory) / "case")

      def shrink_input(candidate):
        (candidate / "inputs.bin").write_bytes(b"shrunk")

      report = tensor_reducer.reduce_case(
        case,
        output_dir=Path(directory) / "bundle",
        run_target=lambda _path: mismatch(),
        validate_candidate=lambda _path: {"valid": False, "reason": "property failed"},
        transformations=[{"name": "shrink-input", "apply": shrink_input}],
      )

      self.assertEqual(report["status"], "no_preserving_simplification_found")
      self.assertEqual(report["trace"][0]["reason"], "property failed")

  def test_no_smaller_case_is_a_valid_bundled_outcome(self):
    with tempfile.TemporaryDirectory() as directory:
      case = make_case(Path(directory) / "case")
      report = tensor_reducer.reduce_case(
        case,
        output_dir=Path(directory) / "bundle",
        run_target=lambda _path: mismatch(),
        validate_candidate=lambda _path: {"valid": True},
        transformations=[],
      )

      bundle = Path(directory) / "bundle"
      self.assertEqual(report["status"], "no_preserving_simplification_found")
      self.assertTrue((bundle / "original" / "inputs.bin").is_file())
      self.assertTrue((bundle / "original" / "expected.bin").is_file())
      self.assertTrue((bundle / "replay.json").is_file())
      self.assertTrue((bundle / "reduced" / "program.sir").is_file())

  def test_counted_loop_bound_reduction_keeps_mismatch_and_clean_bundle(self):
    source = (
      "fun @kernel(%x: i32) : i32 {\n"
      "  let mut %i: i32 = undef;\n"
      "^entry:\n  %i = 0;\n  br ^head;\n"
      "^head:\n  br %i < 2, ^body, ^done;\n"
      "^body:\n  %i = %i + 1;\n  br ^head;\n"
      "^done:\n  ret %x;\n}\n"
    )
    with tempfile.TemporaryDirectory() as directory:
      case = make_case(Path(directory) / "case")
      (case / "program.sir").write_text(source)

      def shorten_loop(candidate):
        program = candidate / "program.sir"
        program.write_text(program.read_text().replace("%i < 2", "%i < 1"))
        manifest_path = candidate / "case.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["execution"] = {"loop_bounds": [1], "total_body_executions": 1}
        manifest_path.write_text(json.dumps(manifest))

      bundle = Path(directory) / "fresh-bundle"
      report = tensor_reducer.reduce_case(
        case,
        output_dir=bundle,
        run_target=lambda _path: mismatch(),
        validate_candidate=lambda _path: {"valid": True},
        transformations=[{"name": "reduce-loop-bound", "apply": shorten_loop}],
      )

      self.assertEqual(report["status"], "preserving_simplification")
      self.assertIn("%i < 1", (bundle / "reduced" / "program.sir").read_text())
      self.assertTrue((bundle / "original" / "case.json").is_file())
      self.assertTrue((bundle / "reduced" / "case.json").is_file())
      self.assertTrue((bundle / "replay.sh").is_file())


if __name__ == "__main__":
  unittest.main()
