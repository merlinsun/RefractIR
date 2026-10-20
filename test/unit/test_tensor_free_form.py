"""Executable contract tests for free-form regions and their counted-loop assembly."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from test.unit.tensor_support import SYMIRI, check_source, describe_entry

FREE_FORM_API = Path(
  os.environ.get("REFRACTIR_TENSOR_FREE_FORM_API", "build/bin/tensor_free_form_api")
)
FREE_FORM_EXPORT = Path(
  os.environ.get("REFRACTIR_TENSOR_FREE_FORM_EXPORT", "build/bin/tensor_free_form_export")
)


class FreeFormContractTests(unittest.TestCase):
  """The builder and the assembler each keep their own stated contract."""

  def run_contract(self, name):
    self.assertTrue(
      FREE_FORM_API.is_file(), f"free-form API binary not found: {FREE_FORM_API}"
    )
    result = subprocess.run(
      [str(FREE_FORM_API), name], text=True, capture_output=True, check=False, timeout=30
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def render(self, name):
    self.assertTrue(
      FREE_FORM_API.is_file(), f"free-form API binary not found: {FREE_FORM_API}"
    )
    result = subprocess.run(
      [str(FREE_FORM_API), name], text=True, capture_output=True, check=False, timeout=30
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return result.stdout

  def test_two_typed_results_of_requested_shapes(self):
    self.run_contract("two-typed-results")

  def test_inputs_may_be_invariants_not_only_states(self):
    self.run_contract("inputs-not-only-states")

  def test_one_consumer_counts_a_value_named_twice_once(self):
    self.run_contract("add-of-one-value-is-one-consumer")

  def test_pruning_omits_dead_values_and_counts_live_consumers(self):
    self.run_contract("pruning-omits-dead-values")

  def test_same_seed_replays_and_impossible_limits_refuse(self):
    self.run_contract("deterministic-and-impossible")

  def test_three_state_region_assembles_into_valid_sir(self):
    self.run_contract("three-state-assembly")

  def test_state_and_invariant_initialization_is_respected(self):
    self.run_contract("invariant-initialization")

  def test_simultaneous_updates_read_saved_values(self):
    self.run_contract("simultaneous-updates")

  def test_incompatible_updates_and_unknown_terms_are_rejected(self):
    self.run_contract("rejected-updates")

  def test_canonical_adapter_emits_the_canonical_loop_byte_for_byte(self):
    self.run_contract("canonical-bytes")

  def test_policy_names_round_trip(self):
    self.run_contract("policy-names")

  def test_selection_scores_bias_without_excluding_actions(self):
    self.run_contract("selection-bounds")

  def test_unguided_can_select_outside_a_biased_rank(self):
    self.run_contract("policy-unguided-ignores-rank")

  def test_a_depth_bias_changes_the_selected_action(self):
    self.run_contract("policy-depth-changes-choice")

  def test_a_sharing_bias_selects_a_second_consumer(self):
    self.run_contract("policy-sharing-selects-a-consumer")

  def test_a_reduction_bias_selects_a_reduction(self):
    self.run_contract("policy-reduction-selects-a-reduction")

  def test_replay_is_separate_from_the_distribution(self):
    self.run_contract("replay-vs-distribution")

  def test_features_stay_off_unless_the_request_enables_them(self):
    self.run_contract("features-off-keep-the-core")

  def test_enabled_features_appear_in_generated_bodies(self):
    self.run_contract("features-enable-transpose-and-expand")

  def test_features_respect_the_rank_limit(self):
    self.run_contract("features-respect-the-rank-limit")

  def test_a_feature_request_replays_exactly(self):
    self.run_contract("features-replay-exactly")



@unittest.skipUnless(SYMIRI.is_file(), "native tensor tools unavailable")
class FreeFormFrontendTests(unittest.TestCase):
  """The assembled source is what the native frontend accepts and describes."""

  def render(self, name):
    self.assertTrue(
      FREE_FORM_API.is_file(), f"free-form API binary not found: {FREE_FORM_API}"
    )
    result = subprocess.run(
      [str(FREE_FORM_API), name], text=True, capture_output=True, check=False, timeout=30
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return result.stdout

  def test_three_state_source_passes_the_native_frontend(self):
    source = self.render("three-state-source")
    result = check_source(source, tensor_profile="v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_three_state_source_describes_its_typed_signature(self):
    source = self.render("three-state-source")
    result, description = describe_entry(source, tensor_profile="v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIsNotNone(description, result.stdout)
    self.assertEqual(description["entry"], "kernel")
    self.assertEqual(description["output"]["shape"], [2, 2])
    self.assertEqual(description["output"]["dtype"], "f32")
    self.assertEqual(
      [parameter["name"] for parameter in description["inputs"]], ["x", "y"]
    )

  def test_canonical_source_passes_the_native_frontend(self):
    source = self.render("canonical-source")
    result = check_source(source, tensor_profile="v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    _, description = describe_entry(source, tensor_profile="v2")
    self.assertEqual([parameter["name"] for parameter in description["inputs"]], ["x"])



@unittest.skipUnless(SYMIRI.is_file(), "native tensor tools unavailable")
class FreeFormExportTests(unittest.TestCase):
  """The exporter writes one case directory whose record states what it built."""

  def export(self, *arguments):
    self.assertTrue(
      FREE_FORM_EXPORT.is_file(), f"free-form export binary not found: {FREE_FORM_EXPORT}"
    )
    return subprocess.run(
      [str(FREE_FORM_EXPORT), *arguments],
      text=True,
      capture_output=True,
      check=False,
      timeout=30,
    )

  def test_exporter_writes_a_program_and_its_structure_record(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.export(
        "--policy", "deep", "--seed", "11", "--shape", "2,2",
        "--trip-count", "2", "--states", "3", "--out", directory,
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      source = (Path(directory) / "program.sir").read_text(encoding="utf-8")
      record = json.loads((Path(directory) / "region.json").read_text(encoding="utf-8"))
      self.assertEqual(record["schema"], "refractir.free-form-region.v1")
      self.assertEqual(record["policy"], "deep")
      self.assertEqual(record["seed"], 11)
      self.assertEqual(record["state_count"], 3)
      self.assertEqual(record["trip_count"], 2)
      self.assertEqual(record["route"], "target-only")
      self.assertEqual([parameter["name"] for parameter in record["parameters"]], ["%x", "%y"])
      self.assertEqual(record["results"], ["%next_x", "%next_y", "%next_z"])
      self.assertGreater(record["structure"]["nodes"], 0)
      self.assertEqual(record["structure"]["policy"], "deep")
      self.assertEqual(len(record["structure"]["result_dependencies"]), 3)
      self.assertNotIn("program_sha256", record, "the bundle writer computes the digests")
      checked = check_source(source, tensor_profile="v2")
      self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)


  def test_a_feature_request_records_its_features(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.export(
        "--policy", "wide", "--seed", "3", "--shape", "2,2", "--max-rank", "3",
        "--features", "transpose,expand", "--out", directory,
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      record = json.loads((Path(directory) / "region.json").read_text(encoding="utf-8"))
      self.assertEqual(record["features"], {"transpose": True, "expand": True})
      self.assertEqual(record["max_rank"], 3)
      checked = check_source(
        (Path(directory) / "program.sir").read_text(encoding="utf-8"),
        tensor_profile="v2",
      )
      self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

  def test_features_are_off_unless_asked_for(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.export(
        "--policy", "unguided", "--seed", "7", "--shape", "2,2", "--out", directory
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      record = json.loads((Path(directory) / "region.json").read_text(encoding="utf-8"))
      self.assertEqual(record["features"], {"transpose": False, "expand": False})
      self.assertNotIn("tensor transpose", (Path(directory) / "program.sir").read_text(encoding="utf-8"))
      self.assertNotIn("tensor expand", (Path(directory) / "program.sir").read_text(encoding="utf-8"))

  def test_a_feature_body_uses_the_operation_it_was_given(self):
    seen = {"transpose": False, "expand": False}
    with tempfile.TemporaryDirectory() as directory:
      for seed in range(1, 21):
        if all(seen.values()):
          break
        output = Path(directory) / f"seed-{seed}"
        result = self.export(
          "--policy", "unguided", "--seed", str(seed), "--shape", "2,2",
          "--max-rank", "3", "--features", "transpose,expand", "--out", str(output),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record = json.loads((output / "region.json").read_text(encoding="utf-8"))
        source = (output / "program.sir").read_text(encoding="utf-8")
        for operation in ("transpose", "expand"):
          if operation in record["structure"]["operator_counts"]:
            self.assertIn(f"tensor {operation} ", source)
            seen[operation] = True
      self.assertTrue(seen["transpose"], "a transposable body must appear within 20 seeds")
      self.assertTrue(seen["expand"], "an expandable body must appear within 20 seeds")

  def test_an_unsupported_feature_is_refused(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.export("--features", "rotate", "--out", directory)
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("unknown free-form feature", result.stderr)
      result = self.export("--max-rank", "4", "--out", directory)
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("max rank", result.stderr)

  def test_exporter_refuses_an_impossible_configuration(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.export("--states", "5", "--out", directory)
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("two to four states", result.stderr)
      self.assertFalse((Path(directory) / "program.sir").exists())

if __name__ == "__main__":
  unittest.main()
