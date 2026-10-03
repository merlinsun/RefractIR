"""Host tests for measured Triton compiler transformations."""

import unittest

from test.lib import tensor_compiler_observation


class TensorCompilerObservationTests(unittest.TestCase):
  def test_attributes_layout_conversion_to_source_operation(self):
    ir = """\
#loc1 = loc("kernel.py":41:12)
#loc2 = loc("converted_value"(#loc1))
%4 = ttg.convert_layout %3 : tensor<2x2xf32, #blocked> -> tensor<2x2xf32, #slice> loc(#loc2)
"""
    report = tensor_compiler_observation.observe_compiler_ir(
      {"ttgir": ir}, source_lines={41: "input_to_compute"}
    )

    self.assertEqual(
      report["layout_conversions"][0]["source_operation"], "input_to_compute"
    )
    self.assertEqual(report["layout_conversions"][0]["source_line"], 41)

  def test_reports_ssa_values_reused_by_multiple_consumers(self):
    ir = """\
%1 = tt.addptr %arg0, %arg1 : tensor<4xi32>
%2 = tt.load %1 : tensor<4xi32>
%3 = tt.add %2, %2 : tensor<4xi32>
%4 = tt.mul %2, %3 : tensor<4xi32>
"""
    report = tensor_compiler_observation.observe_compiler_ir({"ttir": ir})

    self.assertIn(
      {"value": "%2", "use_count": 3}, report["stages"]["ttir"]["reused_values"]
    )

  def test_reports_identical_recomputed_expressions(self):
    ir = """\
%1 = tt.add %arg0, %arg1 : tensor<4xi32>
%2 = tt.add %arg0, %arg1 : tensor<4xi32>
%3 = tt.mul %1, %2 : tensor<4xi32>
"""
    report = tensor_compiler_observation.observe_compiler_ir({"ttir": ir})

    self.assertEqual(report["stages"]["ttir"]["duplicate_expressions"][0]["count"], 2)

  def test_reports_loop_retention_for_a_loop_kernel(self):
    report = tensor_compiler_observation.observe_compiler_ir(
      {
        "ttir": "scf.for %i = %c0 to %c2 step %c1 {\n  tt.add %a, %b\n}\n",
        "ttgir": "scf.for %i = %c0 to %c2 step %c1 {\n  tt.add %a, %b\n}\n",
      },
      expected_loop=True,
    )

    self.assertEqual(report["loop_status"], "retained")
    self.assertEqual(report["stages"]["ttgir"]["loop_construct_count"], 1)

  def test_refuses_a_layout_conversion_that_did_not_survive(self):
    with self.assertRaisesRegex(ValueError, "compute_to_store"):
      tensor_compiler_observation.observe_compiler_ir(
        {"ttgir": "%0 = tt.load %arg0\n"},
        expected_source_operations=("compute_to_store",),
      )

  def test_refuses_a_source_conversion_that_kept_the_same_layout(self):
    ir = """\
#loc1 = loc("kernel.py":61:12)
%4 = ttg.convert_layout %3 : tensor<2x2xf32, #blocked> -> tensor<2x2xf32, #blocked> loc(#loc1)
"""
    with self.assertRaisesRegex(ValueError, "did not change layout"):
      tensor_compiler_observation.observe_compiler_ir(
        {"ttgir": ir},
        source_lines={61: "compute_to_store"},
        expected_source_operations=("compute_to_store",),
      )

  def test_refuses_unplanned_effective_layout_conversions(self):
    ir = """\
#loc1 = loc("kernel.py":41:12)
#loc2 = loc("kernel.py":42:12)
%4 = ttg.convert_layout %3 : tensor<2x2xf32, #blocked> -> tensor<2x2xf32, #slice> loc(#loc1)
%5 = ttg.convert_layout %4 : tensor<2x2xf32, #slice> -> tensor<2x2xf32, #blocked> loc(#loc2)
"""
    with self.assertRaisesRegex(ValueError, "unplanned effective layout conversion"):
      tensor_compiler_observation.observe_compiler_ir(
        {"ttgir": ir},
        source_lines={41: "load_to_compute", 42: "compute_to_store"},
        expected_source_operations=("load_to_compute",),
        exact_source_operations=True,
      )


if __name__ == "__main__":
  unittest.main()
