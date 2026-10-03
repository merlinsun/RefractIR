"""Host checks for the small Gluon blocked-layout pilot contract."""

import unittest

from test.lib.tensor_gluon import BlockedLayoutPlan, generate_layout_plans


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

  def test_layout_generation_is_seeded_and_reproducible(self):
    first = generate_layout_plans((8, 4), num_warps=4, seed=17, count=8)
    second = generate_layout_plans((8, 4), num_warps=4, seed=17, count=8)

    self.assertEqual(
      [item.to_dict() for item in first], [item.to_dict() for item in second]
    )

  def test_layout_generation_returns_distinct_distributions(self):
    plans = generate_layout_plans((8, 4), num_warps=4, seed=23, count=12)

    distributions = {
      (plan.size_per_thread, plan.threads_per_warp, plan.warps_per_cta, plan.order)
      for plan in plans
    }
    self.assertEqual(len(distributions), len(plans))

  def test_every_generated_layout_matches_the_launch(self):
    plans = generate_layout_plans((8, 4), num_warps=4, seed=31, count=20)

    for plan in plans:
      with self.subTest(layout=plan.name):
        plan.validate((8, 4), num_warps=4)

  def test_generation_rejects_an_excessive_layout_count(self):
    with self.assertRaisesRegex(ValueError, "available legal layouts"):
      generate_layout_plans((2, 2), num_warps=1, seed=0, count=100000)

  def test_generation_rejects_invalid_seed(self):
    with self.assertRaisesRegex(ValueError, "seed"):
      generate_layout_plans((2, 2), num_warps=1, seed=-1, count=2)


if __name__ == "__main__":
  unittest.main()
