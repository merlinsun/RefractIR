"""Plans are built from the consumers a generated region actually has."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from test.lib import tensor_region_generation
from test.lib.tensor_gluon import (
  PLAN_FAMILY_MINIMUM_LAYOUTS,
  BlockedLayoutPlan,
  region_consumers,
  region_plan_families,
)
from test.lib.tensor_gluon_lowering import render_gluon_kernel
from test.lib.tensor_region_generation import RegionConfiguration, generate_source

SHAPE = (8, 8)
EXPORTER_AVAILABLE = tensor_region_generation.EXPORTER.is_file()
ORTHOGONAL_FIXTURE_PATH = (
  Path(__file__).resolve().parents[1]
  / "tensor/gluon/programs/orthogonal-reductions-region-v1.sir"
)
LAYOUTS = (
  BlockedLayoutPlan("blocked-a", (1, 1), (1, 32), (1, 1), (1, 0)),
  BlockedLayoutPlan("blocked-b", (2, 1), (1, 32), (1, 1), (1, 0)),
  BlockedLayoutPlan("blocked-c", (1, 2), (1, 32), (1, 1), (0, 1)),
  BlockedLayoutPlan("blocked-d", (2, 2), (1, 32), (1, 1), (0, 1)),
)
# The `fold` family reduces two different values, so the region contains
# reductions but two of them never read the same value.
FOLD_SOURCE = """fun @kernel(%x: tensor [8, 8] f32) : tensor [8, 8] f32 {
  let mut %i: i32 = undef;
  let mut %acc_x: tensor [8, 8] f32 = undef;
  let mut %acc_y: tensor [8, 8] f32 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  let mut %out: tensor [8, 8] f32 = undef;
  let mut %t0: tensor [8, 8] f32 = undef;
  let mut %t1: tensor [8, 1] f32 = undef;
  let mut %t2: tensor [8, 8] f32 = undef;
  let mut %t3: tensor [8, 8] f32 = undef;
  let mut %t4: tensor [8, 1] f32 = undef;
  let mut %t5: tensor [8, 8] f32 = undef;
  let mut %t6: tensor [8, 8] f32 = undef;
  let mut %next_x: tensor [8, 8] f32 = undef;
  let mut %next_y: tensor [8, 8] f32 = undef;
^entry:
  %acc_x = %x;
  %i = 0;
  br ^header;
^header:
  br %i < 2, ^body, ^exit;
^body:
  %t0 = tensor add %acc_x, %acc_y;
  %t1 = tensor sum %t0, 1, keep;
  %t2 = tensor broadcast %t1, [8, 8];
  %t3 = tensor add %t0, %t2;
  %t4 = tensor sum %t3, 1, keep;
  %t5 = tensor broadcast %t4, [8, 8];
  %t6 = tensor add %t3, %t5;
  %next_x = tensor add %acc_x, %t6;
  %next_y = tensor add %acc_y, %t6;
  %acc_x = %next_x;
  %acc_y = %next_y;
  %i = %i + 1;
  br ^header;
^exit:
  %out = tensor add %acc_x, %acc_y;
  ret %out;
}
"""


def identifier(layout):
  """The kernel-body identifier the emitter derives from a layout name."""
  return str(layout.name).replace("-", "_")


def body_of(source):
  """The emitted body, i.e. everything after the kernel signature."""
  return source.split("\n):\n", 1)[1]


def families_for(source, *, layouts=LAYOUTS):
  """The plan families one region graph and one layout pool can express."""
  return region_plan_families(
    SHAPE, num_warps=1, layouts=list(layouts), consumers=region_consumers(source)
  )


class TensorGluonRegionConsumerTests(unittest.TestCase):
  """Consumer identities come from the graph, in the emitter's own naming."""

  def test_consumers_name_the_operations_and_the_values_they_read(self):
    consumers = region_consumers(FOLD_SOURCE)

    self.assertEqual(consumers.shape, SHAPE)
    self.assertEqual(consumers.roots, ("acc_x", "acc_y"))
    self.assertEqual(consumers.uses_of("sum"), (("v0", 1), ("v3", 4)))
    self.assertEqual(consumers.uses_of("add")[0], ("acc_x", 0))
    self.assertIsNone(consumers.shared_reduction_value())
    self.assertEqual(consumers.reduction_result_broadcasts(), (("v1", 2), ("v4", 5)))

  def test_a_shared_value_may_be_a_loop_state_root(self):
    consumers = region_consumers(ORTHOGONAL_FIXTURE_PATH.read_text(encoding="utf-8"))

    self.assertEqual(consumers.shared_reduction_value(), ("acc_x", (0, 2)))
    self.assertEqual(consumers.reduction_result_broadcasts(), (("v0", 1), ("v2", 3)))

  def test_a_consumer_names_the_producing_node_not_the_graph_temporary(self):
    consumers = region_consumers(FOLD_SOURCE)

    # The graph calls these `t1` and `t4`; the emitter calls them `v1` and `v4`.
    self.assertEqual(consumers.uses_of("broadcast"), (("v1", 2), ("v4", 5)))


class TensorGluonGraphDerivedFamilyTests(unittest.TestCase):
  """Every family states the consumers the graph really offers, or is omitted."""

  def test_a_graph_without_a_shared_reduction_value_omits_those_families(self):
    families = families_for(FOLD_SOURCE)

    for name in ("shared-value-reused", "split-reduction-consumers"):
      with self.subTest(family=name):
        self.assertNotIn(name, families)
        self.assertEqual(
          families.omitted[name], "no value is read by two reduction nodes"
        )
    self.assertIn("reduction-result-expand", families)

  def test_every_family_the_pool_allows_is_either_built_or_explained(self):
    families = families_for(FOLD_SOURCE)

    self.assertEqual(
      sorted(families) + sorted(families.omitted),
      sorted(PLAN_FAMILY_MINIMUM_LAYOUTS),
    )
    for name, plan in families.items():
      with self.subTest(family=name):
        self.assertEqual(plan.name, name)
        self.assertEqual(plan.shape, SHAPE)
        self.assertIsNotNone(plan.load)
        self.assertIsNotNone(plan.compute)
        self.assertIsNotNone(plan.store)

  def test_every_built_family_renders_the_emitter_without_collapsing(self):
    baseline = None
    for name, plan in families_for(FOLD_SOURCE).items():
      with self.subTest(family=name):
        generated = render_gluon_kernel(FOLD_SOURCE, plan=plan)

        ast.parse(generated.source)
        body = body_of(generated.source)
        for layout_name in generated.layout_parameters:
          self.assertRegex(body, rf"\b{re.escape(layout_name)}\b")
        if name == "all-same-layout":
          baseline = generated.source
        elif baseline is not None:
          self.assertNotEqual(generated.source, baseline)


class TensorGluonGeneratedRegionPathTests(unittest.TestCase):
  """The native exporter feeds the plan families end to end."""

  def setUp(self):
    if not EXPORTER_AVAILABLE:
      self.skipTest("the native graph exporter is supplied by the managed target")
    self.orthogonal = generate_source(
      RegionConfiguration(
        seed=1, shape=SHAPE, iterations=2, family="orthogonal-reductions"
      )
    )

  def test_consumers_name_the_native_graph_identities(self):
    consumers = region_consumers(self.orthogonal)

    self.assertEqual(consumers.shape, SHAPE)
    self.assertEqual(consumers.roots, ("acc_x", "acc_y"))
    self.assertEqual(consumers.shared_reduction_value(), ("v0", (1, 2)))
    self.assertEqual(consumers.reduction_result_broadcasts(), (("v1", 3), ("v2", 4)))
    self.assertEqual(consumers.uses_of("sum"), (("v0", 1), ("v0", 2)))

  def test_shared_and_split_families_view_the_value_the_graph_shares(self):
    families = families_for(self.orthogonal)
    shared = families["shared-value-reused"].consumer_views
    split = families["split-reduction-consumers"].consumer_views

    self.assertEqual(sorted(shared), [("v0", 1), ("v0", 2)])
    self.assertEqual(sorted(split), [("v0", 1), ("v0", 2)])
    self.assertEqual({layout.name for layout in shared.values()}, {LAYOUTS[1].name})
    self.assertNotEqual(split[("v0", 1)], split[("v0", 2)])
    self.assertEqual(
      sorted(families["reduction-result-expand"].consumer_views),
      [("v1", 3), ("v2", 4)],
    )

  def test_the_view_families_state_a_consumer_view_and_the_rest_do_not(self):
    families = families_for(self.orthogonal)

    for name in (
      "shared-value-reused",
      "split-reduction-consumers",
      "reduction-result-expand",
    ):
      with self.subTest(family=name):
        self.assertTrue(families[name].consumer_views)
    for name in ("all-same-layout", "load-to-compute", "loop-state-layout"):
      with self.subTest(family=name):
        self.assertFalse(families[name].consumer_views)

  def test_a_region_without_reductions_declares_no_reduction_role(self):
    source = generate_source(
      RegionConfiguration(seed=1, shape=SHAPE, iterations=2, family="cross-state")
    )
    families = families_for(source)

    for name, plan in families.items():
      with self.subTest(family=name):
        self.assertIsNone(plan.reduction)
    self.assertIsNotNone(families["loop-state-layout"].state_layout)

  def test_orthogonal_reductions_use_the_declared_reduction_layout(self):
    plan = families_for(self.orthogonal)["split-reduction-consumers"]
    generated = render_gluon_kernel(self.orthogonal, plan=plan)
    reduction = identifier(plan.reduction)
    other = identifier(LAYOUTS[2])

    self.assertIn(f"= gl.convert_layout(v0, {reduction})", generated.source)
    self.assertIn(f"= gl.convert_layout(v0, {other})", generated.source)
    self.assertEqual(generated.source.count(f"gl.sum(v0_view_{reduction},"), 1)
    self.assertEqual(generated.source.count(f"gl.sum(v0_view_{other},"), 1)

  def test_every_exported_family_renders_every_applicable_plan(self):
    for family in (
      "fold",
      "orthogonal-reductions",
      "reduction-arithmetic",
      "cross-state",
      "mixed-depth",
    ):
      source = generate_source(
        RegionConfiguration(seed=1, shape=SHAPE, iterations=2, family=family)
      )
      families = families_for(source)
      with self.subTest(family=family):
        self.assertIn("all-same-layout", families)
        self.assertIn("load-to-compute", families)
      for name, plan in families.items():
        with self.subTest(family=family, plan=name):
          generated = render_gluon_kernel(source, plan=plan)

          ast.parse(generated.source)
          body = body_of(generated.source)
          for layout_name in generated.layout_parameters:
            self.assertRegex(body, rf"\b{re.escape(layout_name)}\b")
          for (value, _index), layout in plan.consumer_views.items():
            view = f"{value}_view_{identifier(layout)}"
            self.assertIn(view, body)
            self.assertIn(
              f"{view} = gl.convert_layout({value}, {identifier(layout)})", body
            )


if __name__ == "__main__":
  unittest.main()
