"""Selection helpers for a matched exact-domain policy study."""

import unittest

from test.lib import tensor_exact_f32, tensor_fixed_study


class TensorFixedSelectionTests(unittest.TestCase):
  def setUp(self):
    self.source = tensor_exact_f32.render_program()

  def test_seeded_random_policy_finds_a_checked_property_witness(self):
    result = tensor_fixed_study.select_random_witness(
      self.source,
      domain=(0, 8),
      target=63,
      target_index=(0, 0),
      seed=17,
      proposal_budget=1000,
    )

    self.assertEqual(result["status"], "accepted")
    self.assertTrue(result["proof"]["complete_output"])
    self.assertEqual(
      tensor_fixed_study.output_element(result["output_bytes"], (2, 2), (0, 0)), 63
    )
    self.assertLessEqual(result["proposal_count"], 1000)

  def test_seeded_random_policy_repeats_values_and_proposal_count(self):
    first = tensor_fixed_study.select_random_witness(
      self.source,
      domain=(0, 8),
      target=63,
      target_index=(0, 0),
      seed=23,
      proposal_budget=1000,
    )
    second = tensor_fixed_study.select_random_witness(
      self.source,
      domain=(0, 8),
      target=63,
      target_index=(0, 0),
      seed=23,
      proposal_budget=1000,
    )

    self.assertEqual(first["values"], second["values"])
    self.assertEqual(first["proposal_count"], second["proposal_count"])

  def test_random_policy_records_exhaustion_for_an_impossible_property(self):
    result = tensor_fixed_study.select_random_witness(
      self.source,
      domain=(0, 8),
      target=10000,
      target_index=(0, 0),
      seed=29,
      proposal_budget=5,
    )

    self.assertEqual(result["status"], "proposal_exhausted")
    self.assertEqual(result["proposal_count"], 5)
    self.assertEqual(result["values"], None)

  def test_solver_model_is_replayed_through_the_complete_reference(self):
    witness = tensor_fixed_study.validate_solver_model(
      {"@kernel": {"%?x": [1, 2, 4, 8]}},
      self.source,
      domain=(0, 8),
      target=63,
      target_index=(0, 0),
    )

    self.assertEqual(witness["values"], (1, 2, 4, 8))
    self.assertEqual(len(witness["output_bytes"]), 16)
    self.assertTrue(witness["proof"]["complete_output"])

  def test_solver_model_with_a_wrong_property_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "property"):
      tensor_fixed_study.validate_solver_model(
        {"@kernel": {"%?x": [0, 0, 0, 0]}},
        self.source,
        domain=(0, 8),
        target=63,
        target_index=(0, 0),
      )


if __name__ == "__main__":
  unittest.main()
