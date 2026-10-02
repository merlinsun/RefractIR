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


if __name__ == "__main__":
  unittest.main()
