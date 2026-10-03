"""Shared generated-region graphs lower to inspectable Gluon source."""

import ast
import unittest

from test.lib.tensor_gluon_lowering import (
  cross_lane_threads_per_warp,
  render_gluon_kernel,
)
from test.unit.test_tensor_exact_composed import (
  generated_region_source,
  generated_sequential_region_source,
)


class TensorGluonLoweringTests(unittest.TestCase):
  def test_rendered_kernel_contains_the_shared_region_operations(self):
    generated = render_gluon_kernel(generated_region_source())

    self.assertEqual(generated.shape, (2, 2))
    self.assertEqual(generated.iterations, 2)
    self.assertIn("v0 =", generated.source)
    self.assertIn("gl.expand_dims(gl.sum(v0, axis=1)", generated.source)
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

    self.assertIn("gl.sum(v0, axis=0)", generated.source)
    self.assertIn(
      "gl.expand_dims(gl.sum(v0, axis=0), 0), compute_layout", generated.source
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

    self.assertIn("gl.sum(v0, axis=1)", generated.source)

  def test_selects_warp_lanes_along_a_nontrivial_reduction_axis(self):
    self.assertEqual(cross_lane_threads_per_warp((2, 2), (0, 1)), (1, 32))
    self.assertEqual(cross_lane_threads_per_warp((2, 2), (0,)), (32, 1))
    with self.assertRaisesRegex(ValueError, "no nontrivial reduction axis"):
      cross_lane_threads_per_warp((1, 1), (0, 1))


if __name__ == "__main__":
  unittest.main()
