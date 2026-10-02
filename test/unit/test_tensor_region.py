"""Executable contract tests for reusable multi-output tensor regions."""

import os
import subprocess
import unittest
from pathlib import Path

REGION_API = Path(
  os.environ.get("REFRACTIR_TENSOR_REGION_API", "build/bin/tensor_region_api")
)


class TensorRegionTests(unittest.TestCase):
  def run_contract(self, name):
    self.assertTrue(
      REGION_API.is_file(), f"tensor region API binary not found: {REGION_API}"
    )
    result = subprocess.run(
      [str(REGION_API), name], text=True, capture_output=True, check=False
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_multiple_typed_results(self):
    self.run_contract("multi-results")

  def test_live_declarations_and_dead_node_pruning(self):
    self.run_contract("declarations-and-dead-nodes")

  def test_state_updates_are_separate_from_body_computation(self):
    self.run_contract("simultaneous-state-updates")

  def test_shared_dependencies_are_emitted_once_for_all_roots(self):
    self.run_contract("shared-dependencies")

  def test_invalid_shapes_axes_operands_and_arities_are_rejected(self):
    self.run_contract("invalid-graphs")

  def test_singleton_reduction_axis_can_expand(self):
    self.run_contract("singleton-broadcast")

  def test_scalar_reduction_can_be_broadcast_back_to_tensor(self):
    self.run_contract("scalar-reduction")

  def test_input_roots_can_be_swapped(self):
    self.run_contract("state-input-swap")

  def test_input_roots_can_form_long_cycles(self):
    self.run_contract("state-input-cycle")

  def test_identity_root_can_share_a_parallel_update(self):
    self.run_contract("state-input-identity")

  def test_pass_through_survives_computed_writeback(self):
    self.run_contract("state-input-computed-mix")

  def test_repeated_pass_through_sources_keep_old_values(self):
    self.run_contract("state-input-repeated")


if __name__ == "__main__":
  unittest.main()
