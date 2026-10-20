"""Executable contract tests for free-form regions and their counted-loop assembly."""

import json
import os
import subprocess
import unittest
from pathlib import Path

from test.unit.tensor_support import SYMIRI, check_source, describe_entry

FREE_FORM_API = Path(
  os.environ.get("REFRACTIR_TENSOR_FREE_FORM_API", "build/bin/tensor_free_form_api")
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


if __name__ == "__main__":
  unittest.main()
