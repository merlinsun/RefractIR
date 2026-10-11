"""Consumer-specific layout plans for the Gluon generated-region emitter."""

from __future__ import annotations

import ast
import unittest
from dataclasses import replace
from pathlib import Path

from test.lib.tensor_gluon import (
  BlockedLayoutPlan,
  classify_layout_pair,
  effective_mapping,
  region_consumers,
  region_plan_families,
)
from test.lib.tensor_gluon_lowering import render_gluon_kernel
from test.unit.test_tensor_exact_composed import generated_region_source

ORTHOGONAL_PROGRAM_PATH = (
  Path(__file__).resolve().parents[1]
  / "tensor/gluon/programs/orthogonal-reductions-region-v1.sir"
)
SHAPE = (8, 8)


def plan(name, size_per_thread, threads_per_warp, order=(1, 0)):
  return BlockedLayoutPlan(name, size_per_thread, threads_per_warp, (1, 1), order)


def identifier(layout):
  """The kernel-body identifier the emitter derives from a layout name."""
  return str(layout.name).replace("-", "_")


def owner_lanes(layout, shape, coordinate):
  """The physical lanes that own one logical coordinate."""
  index = coordinate[0] * shape[1] + coordinate[1]
  return {lane for _warp, lane, _register in effective_mapping(layout, shape)[index]}


class TensorGluonPlanMappingTests(unittest.TestCase):
  """Effective mappings decide whether two plans ask for the same work."""

  def test_distinct_parameters_with_one_effective_mapping_are_equivalent(self):
    narrow = plan("narrow", (2, 1), (1, 32))
    wide = plan("wide", (4, 1), (1, 32))

    self.assertEqual(effective_mapping(narrow, (2, 1)), effective_mapping(wide, (2, 1)))
    self.assertEqual(classify_layout_pair(narrow, wide, shape=(2, 1)), "equivalent")

  def test_declared_lane_bits_keep_physical_lane_identity(self):
    narrow = plan("narrow-lanes", (1, 1), (4, 8))
    wide = plan("wide-lanes", (1, 1), (16, 2))

    self.assertNotEqual(classify_layout_pair(narrow, wide, shape=(2, 2)), "equivalent")
    self.assertEqual(min(owner_lanes(narrow, (2, 2), (1, 0))), 8)
    self.assertEqual(min(owner_lanes(wide, (2, 2), (1, 0))), 2)

  def test_a_value_smaller_than_the_tile_is_replicated_on_every_lane(self):
    layout = plan("replicated", (1, 1), (1, 32))

    owners = effective_mapping(layout, (2, 1))

    self.assertEqual({lane for _warp, lane, _register in owners[0]}, set(range(32)))
    self.assertEqual({register for _warp, _lane, register in owners[0]}, {0})

  def test_a_size_per_thread_wider_than_the_tensor_keeps_lane_identity(self):
    layout = plan("wide-tile", (1, 2), (1, 32))

    owners = effective_mapping(layout, (2, 2))

    self.assertEqual({lane for _warp, lane, _register in owners[1]}, set(range(32)))
    self.assertEqual({register for _warp, _lane, register in owners[1]}, {1})
    self.assertEqual({register for _warp, _lane, register in owners[0]}, {0})

  def test_a_declared_warp_beyond_the_extent_repeats_the_tile_on_that_warp(self):
    layout = BlockedLayoutPlan("wide-warps", (1, 1), (1, 32), (4, 1), (1, 0))

    owners = effective_mapping(layout, (2, 2))

    self.assertEqual({warp for warp, _lane, _register in owners[0]}, {0, 2})
    self.assertEqual({warp for warp, _lane, _register in owners[2]}, {1, 3})

  def test_classification_follows_owner_sets_not_declared_extents(self):
    single = plan("single-warp", (1, 1), (32, 1))
    doubled = BlockedLayoutPlan("two-warps", (1, 1), (32, 1), (2, 1), (1, 0))

    self.assertNotEqual(
      effective_mapping(single, (2, 2)), effective_mapping(doubled, (2, 2))
    )
    self.assertEqual(classify_layout_pair(single, doubled, shape=(2, 2)), "cross_warp")

  def test_a_lane_change_is_cross_lane_and_a_warp_change_is_cross_warp(self):
    single = plan("single", (1, 1), (1, 32))
    lane = plan("lane", (1, 1), (32, 1))
    warp = BlockedLayoutPlan("warp", (1, 1), (1, 32), (2, 1), (1, 0))

    self.assertEqual(classify_layout_pair(single, lane, shape=(2, 4)), "cross_lane")
    self.assertEqual(classify_layout_pair(single, warp, shape=(2, 4)), "cross_warp")
    self.assertEqual(classify_layout_pair(single, single, shape=(2, 4)), "identical")


class TensorGluonRegionPlanTests(unittest.TestCase):
  """Each family states the roles and the per-consumer views it needs."""

  def setUp(self):
    self.layouts = [
      plan("blocked-a", (1, 1), (1, 32)),
      plan("blocked-b", (2, 1), (1, 32)),
      plan("blocked-c", (1, 2), (1, 32), order=(0, 1)),
      plan("blocked-d", (2, 2), (1, 32), order=(0, 1)),
    ]
    self.source = ORTHOGONAL_PROGRAM_PATH.read_text(encoding="utf-8")
    self.families = region_plan_families(
      SHAPE,
      num_warps=1,
      layouts=self.layouts,
      consumers=region_consumers(self.source),
    )

  def test_plan_families_cover_the_six_roles(self):
    self.assertEqual(
      sorted(self.families),
      [
        "all-same-layout",
        "load-to-compute",
        "loop-state-layout",
        "reduction-result-expand",
        "shared-value-reused",
        "split-reduction-consumers",
      ],
    )
    for name, region_plan in self.families.items():
      with self.subTest(family=name):
        self.assertEqual(region_plan.shape, SHAPE)
        self.assertIsNotNone(region_plan.load)
        self.assertIsNotNone(region_plan.compute)
        self.assertIsNotNone(region_plan.store)

  def test_consumer_views_never_reassign_the_producer(self):
    region_plan = self.families["split-reduction-consumers"]
    generated = render_gluon_kernel(self.source, plan=region_plan)

    ast.parse(generated.source)
    views = region_plan.consumer_views
    self.assertTrue(views, "the split family must ask for consumer-specific views")
    for (value, _node), layout in views.items():
      self.assertNotIn(
        f"{value} = gl.convert_layout({value}, {identifier(layout)})", generated.source
      )
      self.assertIn(f"_view_{identifier(layout)}", generated.source)

  def test_one_conversion_serves_compatible_consumers(self):
    region_plan = self.families["shared-value-reused"]
    generated = render_gluon_kernel(self.source, plan=region_plan)

    shared = [
      line
      for line in generated.source.splitlines()
      if "acc_x_view_" in line and "gl.convert_layout(acc_x" in line
    ]
    self.assertEqual(
      len(shared), 1, "compatible consumers must share one converted view"
    )
    self.assertIn("gl.sum(acc_x_view_blocked_b, axis=1)", generated.source)
    self.assertIn("gl.sum(acc_x_view_blocked_b, axis=0)", generated.source)

  def test_slice_layout_axes_follow_the_role_parent(self):
    region_plan = self.families["reduction-result-expand"]
    generated = render_gluon_kernel(self.source, plan=region_plan)

    self.assertIn(
      f"gl.SliceLayout(1, parent={identifier(region_plan.load)})", generated.source
    )
    self.assertIn(
      f"gl.SliceLayout(0, parent={identifier(region_plan.load)})", generated.source
    )
    self.assertIn(
      f"gl.SliceLayout(1, parent={identifier(region_plan.store)})", generated.source
    )

  def test_loop_state_layout_change_keeps_both_roots(self):
    region_plan = self.families["loop-state-layout"]
    generated = render_gluon_kernel(self.source, plan=region_plan)

    self.assertIn("acc_x = ", generated.source)
    self.assertIn("acc_y = ", generated.source)
    self.assertTrue(region_plan.state_layout is not None)
    self.assertIn(
      f"gl.convert_layout(acc_x, {identifier(region_plan.state_layout)})",
      generated.source,
    )

  def test_every_family_actually_requests_a_conversion(self):
    generated = {
      name: render_gluon_kernel(self.source, plan=family).source
      for name, family in self.families.items()
    }

    baseline = self.families["all-same-layout"]
    self.assertNotIn(
      "_view_", generated["all-same-layout"], "the baseline asks for no view"
    )
    for name, family in self.families.items():
      if name == "all-same-layout":
        continue
      with self.subTest(family=name):
        # A family differs from the baseline through its roles, its views, or
        # both, and the difference has to reach the emitted kernel.
        self.assertNotEqual(family.as_record(), baseline.as_record())
        self.assertNotEqual(
          generated[name],
          generated["all-same-layout"],
          f"{name} emitted the baseline kernel",
        )

  def test_a_view_the_region_never_queries_is_refused(self):
    family = self.families["shared-value-reused"]
    broken = replace(
      family,
      consumer_views={**family.consumer_views, ("v99", 7): family.store},
    )

    with self.assertRaisesRegex(ValueError, "never queries"):
      render_gluon_kernel(self.source, plan=broken)

  def test_unqualified_compiler_identity_is_refused(self):
    from test.lib import probe_triton_env

    with self.assertRaisesRegex(RuntimeError, "source revision"):
      probe_triton_env.require_qualified_triton(
        version=probe_triton_env.QUALIFIED_TRITON_VERSION,
        source_revision="0" * 40,
      )


class TensorGluonPlanEdgeTests(unittest.TestCase):
  def test_warp_factors_beyond_the_extent_replicate_and_stay_equivalent(self):
    rows = BlockedLayoutPlan("warp-rows", (1, 1), (1, 32), (2, 1), (1, 0))
    columns = BlockedLayoutPlan("warp-columns", (1, 1), (1, 32), (1, 2), (1, 0))

    owners = effective_mapping(rows, (1, 1))

    self.assertEqual(len(owners[0]), 64)
    self.assertEqual(classify_layout_pair(rows, columns, shape=(1, 1)), "equivalent")

  def test_equivalent_mapping_is_not_classified_as_cross_lane_work(self):
    layouts = [plan("a", (2, 1), (1, 32)), plan("b", (4, 1), (1, 32))]
    consumers = region_consumers(generated_region_source())
    families = region_plan_families(
      (2, 1), num_warps=1, layouts=layouts, consumers=consumers
    )

    baseline = families["all-same-layout"]
    self.assertEqual(
      classify_layout_pair(baseline.compute, baseline.load, shape=(2, 1)),
      "identical",
    )
    self.assertEqual(
      classify_layout_pair(layouts[0], layouts[1], shape=(2, 1)), "equivalent"
    )
    self.assertEqual(families["load-to-compute"].load.name, "b")
    self.assertEqual(families["load-to-compute"].compute.name, "a")
    generated = render_gluon_kernel(
      generated_region_source(),
      plan=region_plan_families(
        (2, 2),
        num_warps=1,
        layouts=[plan("a", (2, 1), (1, 32)), plan("b", (4, 1), (1, 32))],
        consumers=consumers,
      )["load-to-compute"],
    )

    self.assertIn(
      "gl.convert_layout(gl.load(input_ptr + load_offsets), a)", generated.source
    )
    self.assertIn("parent=b)", generated.source)
    self.assertNotIn("acc_x = gl.convert_layout(acc_x", generated.source)


if __name__ == "__main__":
  unittest.main()
