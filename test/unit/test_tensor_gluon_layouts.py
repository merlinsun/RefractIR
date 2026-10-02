"""Host checks for the small Gluon blocked-layout pilot contract."""

import unittest

from test.lib.tensor_gluon import BlockedLayoutPlan


class TensorGluonLayoutTests(unittest.TestCase):
  def test_row_sharded_blocked_layout_is_admitted(self):
    layout = BlockedLayoutPlan("rows", (1, 1), (32, 1), (1, 1), (0, 1))
    layout.validate((2, 2), num_warps=1)

  def test_column_sharded_blocked_layout_is_admitted(self):
    layout = BlockedLayoutPlan("columns", (1, 1), (1, 32), (1, 1), (1, 0))
    layout.validate((2, 2), num_warps=1)

  def test_invalid_thread_product_is_rejected(self):
    layout = BlockedLayoutPlan("invalid", (1, 1), (4, 4), (1, 1), (1, 0))
    with self.assertRaisesRegex(ValueError, "threads_per_warp"):
      layout.validate((2, 2), num_warps=1)

  def test_warp_allocation_must_match_launch(self):
    layout = BlockedLayoutPlan("invalid", (1, 1), (4, 8), (2, 1), (1, 0))
    with self.assertRaisesRegex(ValueError, "warps_per_cta"):
      layout.validate((2, 2), num_warps=1)

  def test_dimension_order_must_be_a_permutation(self):
    layout = BlockedLayoutPlan("invalid", (1, 1), (4, 8), (1, 1), (0, 0))
    with self.assertRaisesRegex(ValueError, "order"):
      layout.validate((2, 2), num_warps=1)

  def test_problem_tile_dimensions_are_powers_of_two(self):
    layout = BlockedLayoutPlan("row", (1, 1), (32, 1), (1, 1), (0, 1))
    with self.assertRaisesRegex(ValueError, "shape dimensions"):
      layout.validate((3, 2), num_warps=1)


if __name__ == "__main__":
  unittest.main()
