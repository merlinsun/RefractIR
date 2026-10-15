"""Consumer-specific layout plans for the Gluon generated-region emitter."""

from __future__ import annotations

import ast
import re
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
# The repro program of the Track C discovery batch's section 5.1: one fold
# feeds both reductions, and each reduction's broadcast feeds one state add.
FOLDED_ORTHOGONAL_PROGRAM_PATH = (
  Path(__file__).resolve().parents[1]
  / "tensor/gluon/programs/folded-orthogonal-reductions-8x8-region-v1.sir"
)
SHAPE = (8, 8)
LAYOUTS = (
  BlockedLayoutPlan("blocked-a", (1, 1), (1, 32), (1, 1), (1, 0)),
  BlockedLayoutPlan("blocked-b", (2, 1), (1, 32), (1, 1), (1, 0)),
  BlockedLayoutPlan("blocked-c", (1, 2), (1, 32), (1, 1), (0, 1)),
  BlockedLayoutPlan("blocked-d", (2, 2), (1, 32), (1, 1), (0, 1)),
)
# The same family on a graph the family cannot carry: both reduction-fed
# broadcasts write a loop state directly, so no consumer node reads an
# expansion and the plan has no use to state its view on.
ROOT_ONLY_EXPANSION_SOURCE = """fun @kernel(%x: tensor [8, 8] f32) : tensor [8, 8] f32 {
  let mut %i: i32 = undef;
  let mut %acc_x: tensor [8, 8] f32 = undef;
  let mut %acc_y: tensor [8, 8] f32 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  let mut %out: tensor [8, 8] f32 = undef;
  let mut %t0: tensor [8, 8] f32 = undef;
  let mut %t1: tensor [1, 8] f32 = undef;
  let mut %t2: tensor [8, 1] f32 = undef;
  let mut %t3: tensor [8, 8] f32 = undef;
  let mut %t4: tensor [8, 8] f32 = undef;
^entry:
  %acc_x = %x;
  %i = 0;
  br ^header;
^header:
  br %i < 1, ^body, ^exit;
^body:
  %t0 = tensor add %acc_y, %acc_x;
  %t1 = tensor sum %t0, 0, keep;
  %t2 = tensor sum %t0, 1, keep;
  %t3 = tensor broadcast %t1, [8, 8];
  %t4 = tensor broadcast %t2, [8, 8];
  %acc_x = %t3;
  %acc_y = %t4;
  %i = %i + 1;
  br ^header;
^exit:
  %out = tensor add %acc_x, %acc_y;
  ret %out;
}
"""


def plan(name, size_per_thread, threads_per_warp, order=(1, 0)):
  return BlockedLayoutPlan(name, size_per_thread, threads_per_warp, (1, 1), order)


def identifier(layout):
  """The kernel-body identifier the emitter derives from a layout name."""
  return str(layout.name).replace("-", "_")


def families_for(source, *, layouts=LAYOUTS):
  """The plan families one region graph and one layout pool can express."""
  return region_plan_families(
    SHAPE, num_warps=1, layouts=list(layouts), consumers=region_consumers(source)
  )


_CONVERT_LINE = re.compile(r"^(\w+) = gl\.convert_layout\(.*, (\w+)\)$")
_ZEROS_LINE = re.compile(r"^(\w+) = gl\.zeros\(.*layout=(\w+)\)$")
_PAIR_BROADCAST_LINE = re.compile(r"^(\w+), (\w+) = gl\.broadcast\((\w+), (\w+)\)$")
_RESULT_BROADCAST_LINE = re.compile(r"^(\w+), _ = gl\.broadcast\((\w+), (\w+)\)$")
_BINARY_LINE = re.compile(r"^(\w+) = (\w+) [-+*] (\w+)$")
_ALIAS_LINE = re.compile(r"^(\w+) = (\w+)$")


def view_layout(name):
  """The layout a converted view local is named after, or None."""
  _head, marker, layout = str(name).rpartition("_view_")
  return layout if marker else None


def local_layout(name, layouts):
  """The layout one emitted local carries."""
  layout = view_layout(name)
  if layout is not None:
    return layout
  if name not in layouts:
    raise AssertionError(f"the emitted kernel states no layout for {name!r}")
  return layouts[name]


def emitted_broadcasts(source):
  """The operand layouts of every gl.broadcast the emitted kernel calls.

  The Gluon frontend reads both operands of a broadcast in one layout, so a
  kernel that gives them two is refused before any IR exists. The emitter
  names every converted local after its layout and defines every other local
  with a layout-carrying operation, so that property is checkable on the
  generated source alone.
  """
  layouts = {}
  pairs = []
  for line in source.splitlines():
    text = line.strip()
    match = _CONVERT_LINE.match(text) or _ZEROS_LINE.match(text)
    if match:
      layouts[match[1]] = match[2]
      continue
    match = _PAIR_BROADCAST_LINE.match(text)
    if match:
      left, right = local_layout(match[3], layouts), local_layout(match[4], layouts)
      pairs.append((match[3], left, match[4], right))
      layouts[match[1]] = layouts[match[2]] = left
      continue
    match = _RESULT_BROADCAST_LINE.match(text)
    if match:
      left, right = local_layout(match[2], layouts), local_layout(match[3], layouts)
      pairs.append((match[2], left, match[3], right))
      layouts[match[1]] = left
      continue
    match = _BINARY_LINE.match(text)
    if match:
      layouts[match[1]] = local_layout(match[2], layouts)
      continue
    match = _ALIAS_LINE.match(text)
    if match and (match[2] in layouts or view_layout(match[2]) is not None):
      layouts[match[1]] = local_layout(match[2], layouts)
  return tuple(pairs)


def assert_broadcasts_pair_one_layout(case, source):
  """Every emitted broadcast reads both of its operands in one layout."""
  pairs = emitted_broadcasts(source)
  case.assertTrue(pairs, "the emitted kernel calls no gl.broadcast")
  for left, left_layout, right, right_layout in pairs:
    case.assertEqual(left_layout, right_layout, f"gl.broadcast({left}, {right})")


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


class TensorGluonPlanOwnershipTests(unittest.TestCase):
  """What a layout change requires of ownership, replicated copies included.

  A destination holder that already holds the element needs no transfer: the
  register that holds it can serve the new view. Only a destination holder the
  source does not have at all asks for communication, and the kind of movement
  follows the warp or lane the element has to reach.
  """

  def test_dropped_warp_replication_needs_no_movement(self):
    replicated = BlockedLayoutPlan("replicated", (1, 1), (1, 32), (1, 4), (1, 0))
    narrow = BlockedLayoutPlan("narrow", (1, 1), (1, 32), (4, 1), (1, 0))
    source = effective_mapping(replicated, (2, 1))
    destination = effective_mapping(narrow, (2, 1))

    self.assertTrue(
      all(
        any(source_owner[:2] == destination_owner[:2] for source_owner in source_owners)
        for source_owners, destination_owners in zip(source, destination)
        for destination_owner in destination_owners
      ),
      "every destination holder must already hold the element it must own",
    )
    self.assertEqual(classify_layout_pair(replicated, narrow, shape=(2, 1)), "subsumed")

  def test_dropped_replication_is_directional(self):
    replicated = BlockedLayoutPlan("replicated", (1, 1), (1, 32), (1, 4), (1, 0))
    narrow = BlockedLayoutPlan("narrow", (1, 1), (1, 32), (4, 1), (1, 0))

    self.assertEqual(classify_layout_pair(replicated, narrow, shape=(2, 1)), "subsumed")
    self.assertEqual(
      classify_layout_pair(narrow, replicated, shape=(2, 1)), "cross_warp"
    )

  def test_dropped_lane_replication_needs_no_movement(self):
    every_lane = plan("every-lane", (1, 1), (1, 32))
    split_lanes = plan("split-lanes", (1, 1), (32, 1))

    self.assertEqual(
      classify_layout_pair(every_lane, split_lanes, shape=(2, 1)), "subsumed"
    )

  def test_a_lane_change_that_needs_a_new_lane_stays_cross_lane(self):
    every_lane = plan("every-lane", (1, 1), (1, 32))
    split_lanes = plan("split-lanes", (1, 1), (32, 1))

    self.assertEqual(
      classify_layout_pair(split_lanes, every_lane, shape=(2, 4)), "cross_lane"
    )

  def test_a_missing_lane_in_a_warp_that_holds_the_element_is_cross_lane(self):
    split_lanes = BlockedLayoutPlan("split-lanes", (1, 1), (32, 1), (4, 1), (0, 1))
    every_lane = BlockedLayoutPlan("every-lane", (1, 1), (1, 32), (4, 1), (0, 1))

    # Every destination warp already holds the element on another lane, so the
    # new lanes are a shuffle inside a warp rather than inter-warp movement,
    # even though the source spreads the element over more warps.
    self.assertEqual(
      classify_layout_pair(split_lanes, every_lane, shape=(2, 1)), "cross_lane"
    )

  def test_a_warp_change_that_needs_a_new_warp_stays_cross_warp(self):
    replicated = BlockedLayoutPlan("replicated", (1, 1), (1, 32), (1, 4), (1, 0))
    narrow = BlockedLayoutPlan("narrow", (1, 1), (1, 32), (4, 1), (1, 0))

    # A shape wide enough to keep the declared warp bits spreads the tile over
    # the warps, so the destination asks warps the source does not use to hold
    # elements they never held.
    self.assertEqual(
      classify_layout_pair(replicated, narrow, shape=(2, 64)), "cross_warp"
    )

  def test_matching_parameters_are_identical(self):
    left = plan("left", (2, 1), (1, 32))
    right = plan("right", (2, 1), (1, 32))

    self.assertEqual(classify_layout_pair(left, right, shape=(2, 2)), "identical")

  def test_equal_owner_sets_are_equivalent(self):
    rows = plan("rows", (1, 1), (32, 1), order=(1, 0))
    columns = plan("columns", (1, 1), (32, 1), order=(0, 1))

    self.assertEqual(
      effective_mapping(rows, (2, 1)), effective_mapping(columns, (2, 1))
    )
    self.assertEqual(classify_layout_pair(rows, columns, shape=(2, 1)), "equivalent")

  def test_the_same_holders_in_another_register_stay_register_only(self):
    left = BlockedLayoutPlan("left", (1, 2), (1, 32), (1, 1), (0, 1))
    right = BlockedLayoutPlan("right", (2, 1), (32, 1), (1, 1), (0, 1))

    self.assertEqual(classify_layout_pair(left, right, shape=(2, 2)), "register_only")

  def test_a_shape_the_source_cannot_place_is_not_movement_free(self):
    gapped = plan("gapped", (1, 1), (1, 32))
    placed = plan("placed", (1, 1), (32, 1))

    self.assertEqual(effective_mapping(gapped, (3, 1))[2], ())
    self.assertEqual(classify_layout_pair(gapped, placed, shape=(3, 1)), "cross_warp")

  def test_a_shape_the_destination_cannot_place_is_not_movement_free(self):
    gapped = plan("gapped", (1, 1), (1, 32))
    placed = plan("placed", (1, 1), (32, 1))

    self.assertEqual(effective_mapping(gapped, (3, 1))[2], ())
    self.assertEqual(classify_layout_pair(placed, gapped, shape=(3, 1)), "cross_warp")

  def test_equal_mappings_that_both_leave_an_element_unowned_are_not_free(self):
    first = plan("first", (1, 1), (1, 32), order=(1, 0))
    second = plan("second", (1, 1), (1, 32), order=(0, 1))

    self.assertEqual(
      effective_mapping(first, (3, 1)), effective_mapping(second, (3, 1))
    )
    self.assertEqual(classify_layout_pair(first, second, shape=(3, 1)), "cross_warp")


class TensorGluonRegionPlanTests(unittest.TestCase):
  """Each family states the roles and the per-consumer views it needs."""

  def setUp(self):
    self.layouts = list(LAYOUTS)
    self.source = ORTHOGONAL_PROGRAM_PATH.read_text(encoding="utf-8")
    self.families = families_for(self.source)

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


class TensorGluonReductionExpansionTests(unittest.TestCase):
  """The expand family views the expansion, never the broadcast's operand.

  Gluon reads both operands of a broadcast in one layout, so a view stated on
  the operand the emitter pairs with its own reference makes the kernel
  invalid before any IR exists. The family therefore states its view on the
  value each consumer reads after the expansion -- and on every other value
  those consumers read, because a consumer that pairs its operands reads all
  of them in one layout.
  """

  def setUp(self):
    self.source = FOLDED_ORTHOGONAL_PROGRAM_PATH.read_text(encoding="utf-8")
    self.families = families_for(self.source)
    self.plan = self.families["reduction-result-expand"]
    self.kernel = render_gluon_kernel(self.source, plan=self.plan).source

  def test_the_repro_graph_expands_each_reduction_result_in_one_layout(self):
    ast.parse(self.kernel)
    assert_broadcasts_pair_one_layout(self, self.kernel)
    lines = self.kernel.splitlines()
    for result in ("v3", "v4"):
      with self.subTest(expansion=result):
        broadcast = next(
          index
          for index, line in enumerate(lines)
          if f"{result}, _ = gl.broadcast(" in line
        )
        expansion = next(
          index
          for index, line in enumerate(lines)
          if f"{result}_view_" in line and "gl.convert_layout" in line
        )
        self.assertGreater(
          expansion,
          broadcast,
          "the expansion is converted after the broadcast that produces it",
        )

  def test_a_second_graph_of_the_family_expands_in_one_layout(self):
    source = ORTHOGONAL_PROGRAM_PATH.read_text(encoding="utf-8")
    plan = families_for(source)["reduction-result-expand"]
    kernel = render_gluon_kernel(source, plan=plan).source

    ast.parse(kernel)
    assert_broadcasts_pair_one_layout(self, kernel)
    for (value, _consumer), layout in plan.consumer_views.items():
      self.assertIn(
        f"{value}_view_{identifier(layout)} = "
        f"gl.convert_layout({value}, {identifier(layout)})",
        kernel,
      )

  def test_a_broadcast_whose_other_operand_shares_the_layout_is_not_refused(self):
    self.assertIn("reduction-result-expand", self.families)
    self.assertNotIn("reduction-result-expand", self.families.omitted)
    self.assertTrue(self.plan.consumer_views)
    self.assertIn("v3, _ = gl.broadcast(v1, acc_x)", self.kernel)
    self.assertIn("v4, _ = gl.broadcast(v2, acc_x)", self.kernel)
    self.assertNotIn("v1_view_", self.kernel)
    self.assertNotIn("v2_view_", self.kernel)

  def test_a_graph_whose_expansion_reaches_no_consumer_is_refused(self):
    families = families_for(ROOT_ONLY_EXPANSION_SOURCE)

    self.assertNotIn("reduction-result-expand", families)
    self.assertRegex(
      families.omitted["reduction-result-expand"],
      "reduction-fed broadcast result reaches no consumer",
    )

  def test_the_family_never_keys_a_view_on_a_broadcast_operand(self):
    consumers = region_consumers(self.source)
    operands = consumers.reduction_result_broadcasts()

    self.assertEqual(
      sorted(self.plan.consumer_views),
      [("acc_x", 5), ("acc_y", 6), ("v3", 6), ("v4", 5)],
    )
    for operand in operands:
      with self.subTest(operand=operand):
        self.assertNotIn(operand, self.plan.consumer_views)

  def test_a_broadcast_with_both_operands_in_the_target_layout_converts_none(self):
    target = identifier(self.plan.compute)
    pairs = {
      left: (left_layout, right_layout)
      for left, left_layout, _right, right_layout in emitted_broadcasts(self.kernel)
    }

    for operand, broadcast in (("v1", "v3"), ("v2", "v4")):
      with self.subTest(operand=operand):
        self.assertIn(operand, pairs, f"the broadcast of {broadcast} reads {operand}")
        self.assertEqual(pairs[operand], (target, target))
        self.assertNotIn(f"{operand}_view_", self.kernel)
        self.assertNotIn(f"{operand} = gl.convert_layout({operand}", self.kernel)
        self.assertIn(f"{broadcast}, _ = gl.broadcast({operand}, acc_x)", self.kernel)


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
