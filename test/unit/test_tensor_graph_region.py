import os
import subprocess
import unittest

EXECUTABLE = os.environ.get("REFRACTIR_TENSOR_GRAPH_REGION_API")


class TensorGraphRegionApiTests(unittest.TestCase):
  def run_contract(self, name):
    self.assertTrue(EXECUTABLE, "REFRACTIR_TENSOR_GRAPH_REGION_API is required")
    result = subprocess.run(
      [EXECUTABLE, name], capture_output=True, text=True, check=False
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_supplied_signature_and_named_results(self):
    self.run_contract("supplied-signature")

  def test_result_union_keeps_both_input_dependencies(self):
    self.run_contract("result-union")

  def test_operator_budget_includes_all_roots(self):
    self.run_contract("operator-budget")

  def test_same_seed_reproduces_region(self):
    self.run_contract("same-seed")

  def test_different_seeds_vary_the_region(self):
    self.run_contract("different-seeds")

  def test_mismatched_state_shapes_are_rejected(self):
    self.run_contract("mismatched-signature")

  def test_first_temporary_name_can_be_a_requested_result(self):
    self.run_contract("result-name-t0-minimal")

  def test_requested_name_is_reserved_across_generated_folds(self):
    self.run_contract("result-name-t0-folds")

  def test_adjacent_temporary_names_are_reserved(self):
    self.run_contract("adjacent-temporary-names")

  def test_requested_temporary_name_survives_three_output_roots(self):
    self.run_contract("result-name-t0-three-roots")

  def test_requested_temporary_name_reservation_is_seed_independent(self):
    self.run_contract("result-name-t0-seed-variation")

  def test_subtraction_mode_places_subtraction_on_a_live_root(self):
    self.run_contract("region-subtraction")

  def test_multiplication_mode_places_multiplication_on_a_live_root(self):
    self.run_contract("region-multiplication")

  def test_mixed_mode_keeps_both_arithmetic_roots_before_writeback(self):
    self.run_contract("region-mixed-arithmetic")


if __name__ == "__main__":
  unittest.main()
