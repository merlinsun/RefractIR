"""Host checks for the small Gluon blocked-layout pilot contract."""

import unittest

from test.lib import tensor_gluon
from test.lib.tensor_gluon import BlockedLayoutPlan, generate_layout_plans
from test.lib.tensor_gluon_lowering import reduction_blocked_layout


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

  def test_four_warp_row_reduction_shards_warps_over_rows(self):
    layout = reduction_blocked_layout((8, 8), {1}, num_warps=4)

    self.assertEqual(layout.threads_per_warp, (1, 32))
    self.assertEqual(layout.warps_per_cta, (4, 1))
    layout.validate((8, 8), num_warps=4)

  def test_four_warp_column_reduction_shards_warps_over_columns(self):
    layout = reduction_blocked_layout((8, 8), {0}, num_warps=4)

    self.assertEqual(layout.threads_per_warp, (32, 1))
    self.assertEqual(layout.warps_per_cta, (1, 4))
    layout.validate((8, 8), num_warps=4)

  def test_four_warp_orthogonal_reductions_pick_a_stable_primary_axis(self):
    layout = reduction_blocked_layout((8, 8), {0, 1}, num_warps=4)

    self.assertEqual(layout.threads_per_warp, (1, 32))
    self.assertEqual(layout.warps_per_cta, (4, 1))
    layout.validate((8, 8), num_warps=4)

  def test_four_warps_fit_small_nonreduction_extent_across_both_axes(self):
    layout = reduction_blocked_layout((2, 8), {1}, num_warps=4)

    self.assertEqual(layout.warps_per_cta, (2, 2))
    layout.validate((2, 8), num_warps=4)

  def test_reduction_layout_rejects_unsupported_warp_count(self):
    with self.assertRaisesRegex(ValueError, "num_warps"):
      reduction_blocked_layout((8, 8), {1}, num_warps=3)

  def test_one_warp_reduction_layout_preserves_the_existing_lane_mapping(self):
    layout = reduction_blocked_layout((2, 2), {1}, num_warps=1)

    self.assertEqual(layout.threads_per_warp, (1, 32))
    self.assertEqual(layout.warps_per_cta, (1, 1))
    self.assertEqual(layout.order, (1, 0))


class RequiredLayoutSelectionTests(unittest.TestCase):
  @staticmethod
  def _physical_key(layout):
    return (
      layout.size_per_thread,
      layout.threads_per_warp,
      layout.warps_per_cta,
      layout.order,
    )

  def test_seed_zero_keeps_required_and_sampled_layouts_physically_unique(self):
    required = reduction_blocked_layout((2, 2), {0, 1}, num_warps=1)
    sampled = generate_layout_plans((2, 2), num_warps=1, seed=0, count=5)

    selected = tensor_gluon.select_distinct_layouts(
      required, sampled, shape=(2, 2), num_warps=1, count=5
    )

    physical_layouts = {self._physical_key(layout) for layout in selected}
    self.assertEqual(len(selected), 5)
    self.assertEqual(len(physical_layouts), 5)
    self.assertEqual(selected[0], required)

  def test_overlapping_sample_is_replaced_without_losing_required_layout(self):
    required = reduction_blocked_layout((2, 2), {0, 1}, num_warps=1)
    sampled = generate_layout_plans((2, 2), num_warps=1, seed=0, count=5)
    self.assertIn(self._physical_key(required), map(self._physical_key, sampled))

    selected = tensor_gluon.select_distinct_layouts(
      required, sampled, shape=(2, 2), num_warps=1, count=5
    )

    self.assertEqual(selected[0].name, "reduction-blocked-1-warps")
    self.assertEqual(
      len({self._physical_key(layout) for layout in selected}), len(selected)
    )

  def test_nonoverlapping_sample_trims_back_to_the_requested_count(self):
    required = reduction_blocked_layout((2, 2), {0, 1}, num_warps=1)
    all_plans = generate_layout_plans((2, 2), num_warps=1, seed=0, count=48)
    required_key = self._physical_key(required)
    sampled = [plan for plan in all_plans if self._physical_key(plan) != required_key][
      :5
    ]

    selected = tensor_gluon.select_distinct_layouts(
      required, sampled, shape=(2, 2), num_warps=1, count=5
    )

    self.assertEqual(len(selected), 5)
    self.assertEqual(
      len({self._physical_key(layout) for layout in selected}), len(selected)
    )

  def test_selection_near_available_layout_limit_remains_unique(self):
    required = reduction_blocked_layout((2, 2), {0, 1}, num_warps=1)
    sampled = generate_layout_plans((2, 2), num_warps=1, seed=0, count=48)

    selected = tensor_gluon.select_distinct_layouts(
      required, sampled, shape=(2, 2), num_warps=1, count=47
    )

    self.assertEqual(len(selected), 47)
    self.assertEqual(
      len({self._physical_key(layout) for layout in selected}), len(selected)
    )

  def test_count_one_returns_only_the_required_layout(self):
    required = reduction_blocked_layout((2, 2), {0, 1}, num_warps=1)

    selected = tensor_gluon.select_distinct_layouts(
      required, (), shape=(2, 2), num_warps=1, count=1
    )

    self.assertEqual(selected, [required])

  def test_selection_rejects_an_insufficient_distinct_sample(self):
    required = reduction_blocked_layout((2, 2), {0, 1}, num_warps=1)

    with self.assertRaisesRegex(ValueError, "contain only 1"):
      tensor_gluon.select_distinct_layouts(
        required, (), shape=(2, 2), num_warps=1, count=2
      )


if __name__ == "__main__":
  unittest.main()
