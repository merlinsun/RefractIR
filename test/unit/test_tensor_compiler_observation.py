"""Host tests for measured Triton compiler transformations."""

import unittest

from test.lib import tensor_compiler_observation


def _pass_snapshot(name, ir):
  return (
    f"// -----// IR Dump Before {name}: {name.lower()} "
    "('builtin.module' operation) //----- //\n"
    f"{ir}\n"
  )


def _nested_snapshot(name, symbol, ir):
  return (
    f"// -----// IR Dump Before {name}: {name.lower()} "
    f"('tt.func' operation: @{symbol}) //----- //\n"
    f"{ir}\n"
  )


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

  def test_pass_dump_measures_layout_conversion_removal(self):
    before = """\
#loc1 = loc("kernel.py":41:12)
%4 = ttg.convert_layout %3 : tensor<2x2xf32, #blocked> -> tensor<2x2xf32, #slice> loc(#loc1)
"""
    after = "%3 = tt.load %arg0 : tensor<2x2xf32, #slice>\n"
    dump = "".join(
      (
        _pass_snapshot("GluonInline", before),
        _pass_snapshot("RemoveLayoutConversionsPass", after),
        _pass_snapshot("LLVMDIScope", after),
      )
    )

    report = tensor_compiler_observation.observe_pass_dumps(
      dump, source_lines={41: "load_to_compute"}
    )
    transition = report["compilations"][0]["transitions"][0]
    self.assertEqual(transition["effective_layout_conversion_delta"], -1)
    self.assertEqual(
      transition["effective_conversion_operation_deltas"], {"load_to_compute": -1}
    )

  def test_pass_dump_measures_layout_conversion_addition(self):
    before = "%3 = tt.load %arg0 : tensor<2x2xf32, #blocked>\n"
    after = """\
#loc1 = loc("kernel.py":42:12)
%4 = ttg.convert_layout %3 : tensor<2x2xf32, #blocked> -> tensor<2x2xf32, #slice> loc(#loc1)
"""
    dump = "".join(
      (
        _pass_snapshot("GluonInline", before),
        _pass_snapshot("ConvertLayoutPass", after),
        _pass_snapshot("LLVMDIScope", after),
      )
    )

    report = tensor_compiler_observation.observe_pass_dumps(
      dump, source_lines={42: "compute_to_reduction"}
    )
    self.assertEqual(
      report["compilations"][0]["transitions"][0]["effective_layout_conversion_delta"],
      1,
    )

  def test_pass_dump_measures_duplicate_expression_changes(self):
    before = "%1 = tt.add %a, %b : tensor<4xi32>\n"
    after = """\
%1 = tt.add %a, %b : tensor<4xi32>
%2 = tt.add %a, %b : tensor<4xi32>
"""
    dump = "".join(
      (
        _pass_snapshot("GluonInline", before),
        _pass_snapshot("RematerializePass", after),
        _pass_snapshot("LLVMDIScope", after),
      )
    )

    report = tensor_compiler_observation.observe_pass_dumps(dump)
    transition = report["compilations"][0]["transitions"][0]
    self.assertEqual(transition["duplicate_expression_group_delta"], 1)
    self.assertIn("not a proof of rematerialization", report["interpretation_limit"])

  def test_pass_dump_does_not_pair_snapshots_across_compilations(self):
    empty = "module {}\n"
    dump = "".join(
      (
        _pass_snapshot("GluonInline", empty),
        _pass_snapshot("CSEPass", empty),
        _pass_snapshot("LLVMDIScope", empty),
        _pass_snapshot("GluonInline", empty),
        _pass_snapshot("CSEPass", empty),
        _pass_snapshot("LLVMDIScope", empty),
      )
    )

    report = tensor_compiler_observation.observe_pass_dumps(dump)
    self.assertEqual(report["compilation_count"], 2)
    self.assertEqual(
      [len(compilation["transitions"]) for compilation in report["compilations"]],
      [2, 2],
    )

  def test_pass_dump_rejects_an_incomplete_compilation(self):
    dump = _pass_snapshot("GluonInline", "module {}\n")

    with self.assertRaisesRegex(ValueError, "does not end at the final pass"):
      tensor_compiler_observation.observe_pass_dumps(dump)


_LOOP_BODY = """\
module {
  scf.for %i = %c0 to %c2 step %c1 {
    %9 = tt.add %7, %8 : tensor<4xf32>
    scf.yield
  }
}
"""

_STRAIGHT_LINE_BODY = """\
module {
  %9 = tt.add %7, %8 : tensor<4xf32>
}
"""


class TensorCompilerLoopStageTests(unittest.TestCase):
  """A loop in one stage cannot stand in for retention in a later stage."""

  def test_loop_in_ttir_does_not_satisfy_final_ttgir_retention(self):
    report = tensor_compiler_observation.observe_compiler_ir(
      {"ttir": _LOOP_BODY, "ttgir": _STRAIGHT_LINE_BODY},
      expected_loop=True,
      loop_required_stages=("ttgir",),
    )

    self.assertEqual(report["loop_status"], "unrolled_or_removed")
    self.assertEqual(report["loop_presence"], {"ttir": True, "ttgir": False})

  def test_missing_required_loop_stage_stays_unavailable(self):
    report = tensor_compiler_observation.observe_compiler_ir(
      {"ttir": _LOOP_BODY},
      expected_loop=True,
      loop_required_stages=("ttgir",),
    )

    self.assertEqual(report["loop_status"], "unavailable")
    self.assertEqual(report["loop_presence"], {"ttir": True, "ttgir": None})


class TensorCompilerConversionIdentityTests(unittest.TestCase):
  """Encodings are compared by definition, not by alias name."""

  _RENAMED = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32]}>
#slice = #ttg.slice<{dim = 0, parent = #blocked}>
#blocked_alt = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32]}>
#slice_alt = #ttg.slice<{dim = 0, parent = #blocked_alt}>
#loc1 = loc("kernel.py":41:12)
module {
  %9 = ttg.convert_layout %8 : tensor<2x4xf32, #blocked_alt> -> tensor<2x4xf32, #slice_alt> loc(#loc1)
}
"""

  def test_renamed_aliases_and_ssa_names_match_the_same_conversion(self):
    before = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32]}>
#slice = #ttg.slice<{dim = 0, parent = #blocked}>
#loc1 = loc("kernel.py":41:12)
module {
  %4 = ttg.convert_layout %3 : tensor<2x4xf32, #blocked> -> tensor<2x4xf32, #slice> loc(#loc1)
}
"""
    report = tensor_compiler_observation.match_conversion_fragments(
      before, self._RENAMED, source_lines={41: "load_to_compute"}
    )

    self.assertEqual(report["preserved"], 1)
    self.assertEqual(report["removed"], 0)
    self.assertEqual(report["added"], 0)
    self.assertEqual(report["moved"], 0)
    self.assertEqual(report["fragments"][0]["status"], "preserved")
    self.assertEqual(report["fragments"][0]["source_operation"], "load_to_compute")

  def test_conversion_moving_into_a_loop_is_seen_at_equal_counts(self):
    before = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32]}>
#slice = #ttg.slice<{dim = 0, parent = #blocked}>
#loc1 = loc("kernel.py":41:12)
module {
  %4 = ttg.convert_layout %3 : tensor<2x4xf32, #blocked> -> tensor<2x4xf32, #slice> loc(#loc1)
}
"""
    after = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32]}>
#slice = #ttg.slice<{dim = 0, parent = #blocked}>
#loc1 = loc("kernel.py":41:12)
module {
  scf.for %i = %c0 to %c2 step %c1 {
    %4 = ttg.convert_layout %3 : tensor<2x4xf32, #blocked> -> tensor<2x4xf32, #slice> loc(#loc1)
    scf.yield
  }
}
"""
    report = tensor_compiler_observation.match_conversion_fragments(before, after)

    self.assertEqual(report["matched"], 1)
    self.assertEqual(report["preserved"], 0)
    self.assertEqual(report["moved"], 1)
    self.assertEqual(report["fragments"][0]["status"], "moved")
    self.assertEqual(report["fragments"][0]["enclosing_region_before"], "module")
    self.assertEqual(report["fragments"][0]["enclosing_region_after"], "scf.for")

  def test_same_encoding_conversion_is_a_no_op(self):
    ir = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32]}>
#loc1 = loc("kernel.py":41:12)
%4 = ttg.convert_layout %3 : tensor<2x4xf32, #blocked> -> tensor<2x4xf32, #blocked_alt> loc(#loc1)
"""
    report = tensor_compiler_observation.observe_compiler_ir(
      {"ttgir": ir}, source_lines={41: "load_to_compute"}
    )

    self.assertFalse(report["layout_conversions"][0]["layout_changed"])
    with self.assertRaisesRegex(ValueError, "did not change layout"):
      tensor_compiler_observation.observe_compiler_ir(
        {"ttgir": ir},
        source_lines={41: "load_to_compute"},
        expected_source_operations=("load_to_compute",),
      )


class TensorCompilerRematerializationTests(unittest.TestCase):
  """Redirected uses, not duplicate expressions, establish rematerialization."""

  _BEFORE = """\
module {
  %1 = arith.addf %a, %b : tensor<4xf32>
  %2 = arith.mulf %1, %c : tensor<4xf32>
}
"""

  def test_duplicate_with_redirected_consumers_confirms_rematerialization(self):
    after = """\
module {
  %1 = arith.addf %a, %b : tensor<4xf32>
  %7 = arith.addf %a, %b : tensor<4xf32>
  %2 = arith.mulf %7, %c : tensor<4xf32>
}
"""
    report = tensor_compiler_observation.observe_rematerialization(self._BEFORE, after)

    self.assertEqual(report["confirmed_count"], 1)
    candidate = report["confirmed"][0]
    self.assertEqual(candidate["original_value"], "%1")
    self.assertEqual(candidate["recomputed_values"], ["%7"])
    self.assertEqual(candidate["redirected_consumers"], ["%2"])
    self.assertEqual(candidate["remaining_consumers"], [])

  def test_duplicate_that_keeps_every_original_use_is_not_rematerialization(self):
    # The copy feeds a new consumer while the original keeps all of its uses:
    # that is duplicated work, not a use redirected to a recomputed value.
    after = """\
module {
  %1 = arith.addf %a, %b : tensor<4xf32>
  %2 = arith.mulf %1, %c : tensor<4xf32>
  %7 = arith.addf %a, %b : tensor<4xf32>
  %8 = arith.mulf %7, %c : tensor<4xf32>
  %9 = arith.addf %2, %8 : tensor<4xf32>
}
"""
    report = tensor_compiler_observation.observe_rematerialization(self._BEFORE, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["unattributed_count"], 1)
    self.assertEqual(report["status"], "unattributed_only")


class TensorCompilerNestedSnapshotTests(unittest.TestCase):
  """Nested pipelines dump their own snapshots inside a module-level one."""

  def test_nested_function_snapshots_do_not_contaminate_module_snapshot(self):
    module_body = "%1 = arith.addf %a, %b : tensor<4xf32>\n"
    nested_body = """\
%2 = ttg.convert_layout %1 : tensor<4xf32, #blocked> -> tensor<4xf32, #slice>
"""
    dump = "".join(
      (
        _pass_snapshot("InlinerPass", module_body),
        _nested_snapshot("CanonicalizerPass", "k", nested_body),
        _nested_snapshot("CanonicalizerPass", "k", nested_body),
        _pass_snapshot("RemoveLayoutConversionsPass", module_body),
        _pass_snapshot("LLVMDIScope", module_body),
      )
    )

    report = tensor_compiler_observation.observe_pass_dumps(dump)
    compilation = report["compilations"][0]
    self.assertEqual(compilation["snapshot_count"], 3)
    self.assertEqual(compilation["nested_snapshot_counts"], [2, 0, 0])
    for transition in compilation["transitions"]:
      self.assertEqual(transition["before"]["layout_conversion_count"], 0)
      self.assertEqual(transition["after"]["layout_conversion_count"], 0)


class TensorCompilerTargetPassTests(unittest.TestCase):
  """Evidence for a named pass is per-pass and never aggregated away."""

  def test_missing_target_pass_evidence_stays_unavailable(self):
    body = "%1 = arith.addf %a, %b : tensor<4xf32>\n"
    dump = "".join(
      (
        _pass_snapshot("GluonInline", body),
        _pass_snapshot("LLVMDIScope", body),
      )
    )

    report = tensor_compiler_observation.observe_pass_dumps(
      dump, target_passes=("RematerializePass",)
    )
    evidence = report["compilations"][0]["target_pass_evidence"]["RematerializePass"]
    self.assertEqual(evidence["status"], "unavailable")
    self.assertIsNone(evidence["transition_index"])
    self.assertEqual(evidence["transition_indices"], [])

  def test_target_pass_reports_loop_presence_around_it(self):
    loop_body = "scf.for %i = %c0 to %c2 step %c1 {\n  scf.yield\n}\n"
    flat_body = "%1 = arith.addf %a, %b : tensor<4xf32>\n"
    dump = "".join(
      (
        _pass_snapshot("GluonInline", loop_body),
        _pass_snapshot("RematerializePass", loop_body),
        _pass_snapshot("LLVMDIScope", flat_body),
      )
    )

    report = tensor_compiler_observation.observe_pass_dumps(
      dump, target_passes=("RematerializePass",)
    )
    compilation = report["compilations"][0]
    evidence = compilation["target_pass_evidence"]["RematerializePass"]
    self.assertEqual(evidence["status"], "observed")
    self.assertEqual(evidence["transition_index"], 1)
    self.assertEqual(evidence["transition_indices"], [1])
    self.assertEqual(
      evidence["occurrences"],
      [{"transition_index": 1, "loop_before": True, "loop_after": False}],
    )
    self.assertEqual(compilation["final_loop_construct_count"], 0)


if __name__ == "__main__":
  unittest.main()
