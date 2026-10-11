"""Shared generated-region graphs lower to inspectable Gluon source."""

import ast
import re
import unittest
from pathlib import Path

from test.lib.tensor_gluon import BlockedLayoutPlan, GluonRegionPlan
from test.lib.tensor_gluon_lowering import (
  cross_lane_threads_per_warp,
  render_gluon_kernel,
)
from test.unit.test_tensor_exact_composed import (
  generated_region_source,
  generated_sequential_region_source,
)

ORTHOGONAL_PROGRAM_PATH = (
  Path(__file__).resolve().parents[1]
  / "tensor/gluon/programs/orthogonal-reductions-region-v1.sir"
)

LOAD = BlockedLayoutPlan("load-a", (1, 1), (1, 32), (1, 1), (1, 0))
COMPUTE = BlockedLayoutPlan("compute-b", (2, 1), (1, 32), (1, 1), (1, 0))
REDUCTION = BlockedLayoutPlan("reduction-c", (1, 2), (1, 32), (1, 1), (0, 1))
STORE = BlockedLayoutPlan("store-d", (2, 2), (1, 32), (1, 1), (0, 1))
STATE = BlockedLayoutPlan("state-e", (4, 2), (1, 32), (1, 1), (0, 1))
CONSUMER = BlockedLayoutPlan("consumer-f", (4, 1), (1, 32), (1, 1), (1, 0))


def identifier(layout):
  """The kernel-body identifier the emitter derives from a layout name."""
  return str(layout.name).replace("-", "_")


def role_plan(**overrides):
  """One plan whose roles are all distinct, so no role hides behind another."""
  fields = {
    "name": "distinct-roles",
    "shape": (8, 8),
    "load": LOAD,
    "compute": COMPUTE,
    "reduction": REDUCTION,
    "store": STORE,
  }
  fields.update(overrides)
  return GluonRegionPlan(**fields)


def kernel_body(source):
  """The emitted body, i.e. everything after the kernel signature."""
  return source.split("\n):\n", 1)[1]


class TensorGluonLoweringTests(unittest.TestCase):
  def test_rendered_kernel_contains_the_shared_region_operations(self):
    generated = render_gluon_kernel(generated_region_source())

    self.assertEqual(generated.shape, (2, 2))
    self.assertEqual(generated.iterations, 2)
    self.assertIn("v0 =", generated.source)
    self.assertIn(
      "gl.expand_dims(gl.sum(v0_reduction_input_1, axis=1)", generated.source
    )
    self.assertIn("v2, _ = gl.broadcast", generated.source)
    self.assertIn("v3 =", generated.source)
    self.assertIn("for _ in range(2):", generated.source)

  def test_rendered_kernel_preserves_sequential_state_update_order(self):
    source = generated_sequential_region_source("  %acc_x = %t0;\n  %acc_y = %acc_x;")
    generated = render_gluon_kernel(source)

    self.assertLess(
      generated.source.index("acc_x = v0"), generated.source.index("acc_y = acc_x")
    )

  def test_rendered_kernel_places_reduction_and_broadcast_axes(self):
    generated = render_gluon_kernel(generated_region_source(axis=0, iterations=1))

    self.assertIn("gl.sum(v0_reduction_input_1, axis=0)", generated.source)
    self.assertIn(
      "gl.expand_dims(gl.sum(v0_reduction_input_1, axis=0), 0), compute_layout",
      generated.source,
    )
    self.assertIn("gl.convert_layout(v1, compute_layout)", generated.source)
    self.assertIn("gl.broadcast(v1, acc_x)", generated.source)

  def test_rendered_kernel_is_python_and_attributes_layout_conversions(self):
    generated = render_gluon_kernel(generated_region_source())

    ast.parse(generated.source)
    self.assertTrue(generated.conversion_source_lines)
    self.assertIn("load_to_compute", generated.conversion_source_lines.values())
    self.assertIn("compute_to_store", generated.conversion_source_lines.values())

  def test_rejects_unsupported_region_operations_before_emission(self):
    source = generated_region_source().replace("tensor sum", "tensor transpose", 1)

    with self.assertRaisesRegex(ValueError, "unsupported tensor operation"):
      render_gluon_kernel(source)

  def test_normalizes_negative_reduction_axes_for_gluon(self):
    source = generated_region_source(axis=1).replace(
      "tensor sum %t0, 1, keep;", "tensor sum %t0, -1, keep;"
    )

    generated = render_gluon_kernel(source)

    self.assertIn("gl.sum(v0_reduction_input_1, axis=1)", generated.source)

  def test_selects_warp_lanes_along_a_nontrivial_reduction_axis(self):
    self.assertEqual(cross_lane_threads_per_warp((2, 2), (0, 1)), (1, 32))
    self.assertEqual(cross_lane_threads_per_warp((2, 2), (0,)), (32, 1))
    with self.assertRaisesRegex(ValueError, "no nontrivial reduction axis"):
      cross_lane_threads_per_warp((1, 1), (0, 1))

  def test_reduction_uses_a_temporary_layout_view(self):
    generated = render_gluon_kernel(generated_region_source())

    self.assertIn(
      "v0_reduction_input_1 = gl.convert_layout(v0, reduction_layout)",
      generated.source,
    )
    self.assertIn("gl.sum(v0_reduction_input_1, axis=1)", generated.source)

  def test_reduction_conversion_does_not_reassign_the_graph_value(self):
    generated = render_gluon_kernel(generated_region_source())

    self.assertNotIn("v0 = gl.convert_layout(v0, reduction_layout)", generated.source)

  def test_orthogonal_reductions_create_independent_input_views(self):
    source = ORTHOGONAL_PROGRAM_PATH.read_text(encoding="utf-8")
    generated = render_gluon_kernel(source)

    self.assertIn(
      "acc_x_reduction_input_0 = gl.convert_layout(acc_x, reduction_layout)",
      generated.source,
    )
    self.assertIn(
      "acc_x_reduction_input_2 = gl.convert_layout(acc_x, reduction_layout)",
      generated.source,
    )
    self.assertIn("gl.sum(acc_x_reduction_input_0, axis=1)", generated.source)
    self.assertIn("gl.sum(acc_x_reduction_input_2, axis=0)", generated.source)

  def test_orthogonal_reductions_preserve_compute_layout_for_broadcast(self):
    source = ORTHOGONAL_PROGRAM_PATH.read_text(encoding="utf-8")
    generated = render_gluon_kernel(source)

    self.assertNotIn(
      "acc_x = gl.convert_layout(acc_x, reduction_layout)", generated.source
    )
    self.assertIn("v1, _ = gl.broadcast(v0, acc_x)", generated.source)
    self.assertIn("v3, _ = gl.broadcast(v2, acc_x)", generated.source)

  def test_reduction_conversion_source_map_keeps_each_operation_identity(self):
    source = ORTHOGONAL_PROGRAM_PATH.read_text(encoding="utf-8")
    generated = render_gluon_kernel(source)

    mapped_lines = [
      generated.source.splitlines()[line_number - 1]
      for line_number, conversion in generated.conversion_source_lines.items()
      if conversion.endswith("_to_reduction")
    ]

    self.assertEqual(len(mapped_lines), 2)
    self.assertTrue(
      all("_reduction_input_" in line for line in mapped_lines), mapped_lines
    )


class TensorGluonDeclaredRoleTests(unittest.TestCase):
  """A plan states each role; the emitter honors the role without an override."""

  def setUp(self):
    self.source = ORTHOGONAL_PROGRAM_PATH.read_text(encoding="utf-8")

  def test_declared_reduction_layout_reaches_the_sum_operand(self):
    generated = render_gluon_kernel(self.source, plan=role_plan())
    view = f"acc_x_view_{identifier(REDUCTION)}"

    self.assertIn(
      f"{view} = gl.convert_layout(acc_x, {identifier(REDUCTION)})",
      generated.source,
    )
    self.assertIn(f"gl.sum({view}, axis=1)", generated.source)
    self.assertIn(f"gl.sum({view}, axis=0)", generated.source)

  def test_two_reduction_consumers_share_one_role_view(self):
    generated = render_gluon_kernel(self.source, plan=role_plan())
    view = f"acc_x_view_{identifier(REDUCTION)}"
    conversions = [
      line
      for line in generated.source.splitlines()
      if line.strip() == f"{view} = gl.convert_layout(acc_x, {identifier(REDUCTION)})"
    ]

    self.assertEqual(len(conversions), 1)
    self.assertEqual(generated.source.count(f"gl.sum({view},"), 2)

  def test_consumer_override_takes_precedence_over_the_declared_role(self):
    plan = role_plan(consumer_views={("acc_x", 0): CONSUMER})
    generated = render_gluon_kernel(self.source, plan=plan)

    self.assertIn(
      f"gl.sum(acc_x_view_{identifier(CONSUMER)}, axis=1)", generated.source
    )
    self.assertIn(
      f"gl.sum(acc_x_view_{identifier(REDUCTION)}, axis=0)", generated.source
    )

  def test_every_declared_role_layout_is_used_by_the_body(self):
    generated = render_gluon_kernel(self.source, plan=role_plan(state_layout=STATE))
    body = kernel_body(generated.source)

    for name in generated.layout_parameters:
      with self.subTest(layout=name):
        self.assertRegex(body, rf"\b{re.escape(name)}\b")

  def test_view_for_states_only_explicit_consumer_overrides(self):
    plan = role_plan(consumer_views={("acc_x", 0): CONSUMER})

    self.assertIs(plan.view_for("acc_x", 0), CONSUMER)
    self.assertIsNone(plan.view_for("acc_x", 2))
    self.assertIsNone(plan.view_for("v0", 1))

  def test_a_plan_without_a_reduction_role_keeps_the_sums_in_compute(self):
    generated = render_gluon_kernel(self.source, plan=role_plan(reduction=None))

    self.assertIn("gl.sum(acc_x, axis=1)", generated.source)
    self.assertIn("gl.sum(acc_x, axis=0)", generated.source)
    self.assertNotIn(f"acc_x_view_{identifier(COMPUTE)}", generated.source)


if __name__ == "__main__":
  unittest.main()
