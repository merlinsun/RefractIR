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


class TensorCompilerBackedgeTests(unittest.TestCase):
  """A lowered loop is still a loop: later stages encode it as a branch cycle."""

  _PTX = """\
//
.version 8.8
.target sm_70
$L__BB0_1:
@%p1 bra $L__BB0_1;
bra.uni $L__BB0_1;
bra $L__BB0_3;
$L__BB0_3:
ret;
"""

  def test_ptx_backedge_counts_as_a_retained_device_loop(self):
    report = tensor_compiler_observation.observe_compiler_ir(
      {"ttgir": _STRAIGHT_LINE_BODY, "ptx": self._PTX},
      expected_loop=True,
      loop_required_stages=("ptx",),
    )

    self.assertEqual(report["loop_status"], "retained")
    self.assertEqual(report["loop_presence"]["ptx"], True)
    self.assertEqual(report["stages"]["ptx"]["backedge_count"], 2)
    self.assertEqual(report["loop_evidence"]["ptx"], "branch-cycle")

  def test_forward_branch_is_not_a_loop(self):
    forward = """\
$L__BB0_1:
bra $L__BB0_3;
$L__BB0_3:
ret;
"""
    report = tensor_compiler_observation.observe_compiler_ir(
      {"ptx": forward}, expected_loop=True, loop_required_stages=("ptx",)
    )

    self.assertEqual(report["stages"]["ptx"]["backedge_count"], 0)
    self.assertEqual(report["loop_status"], "unrolled_or_removed")

  def test_llvm_ir_backedge_counts_as_a_retained_loop(self):
    llir = """\
define void @k() {
entry:
  br label %loop
loop:
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
"""
    report = tensor_compiler_observation.observe_compiler_ir(
      {"llir": llir}, expected_loop=True, loop_required_stages=("llir",)
    )

    self.assertEqual(report["stages"]["llir"]["backedge_count"], 1)
    self.assertEqual(report["loop_status"], "retained")

  def test_lowered_mlir_loop_counts_as_present(self):
    lowered = """\
module {
  cf.br ^bb1
^bb1:
  cf.br ^bb1
}
"""
    report = tensor_compiler_observation.observe_compiler_ir(
      {"ttgir": lowered}, expected_loop=True, loop_required_stages=("ttgir",)
    )

    self.assertEqual(report["stages"]["ttgir"]["loop_construct_count"], 0)
    self.assertEqual(report["stages"]["ttgir"]["backedge_count"], 1)
    self.assertEqual(report["loop_status"], "retained")
    self.assertEqual(report["loop_evidence"]["ttgir"], "branch-cycle")

  def test_llvm_numeric_labels_count_as_a_backedge(self):
    # Optimized LLVM IR prints numeric labels, and a bottom-tested loop lists
    # its exit block first, so the backedge is not the first label operand.
    llir = """\
define void @k() {
entry:
  br label %56
56:
  %c = icmp slt i32 %i, 4
  br i1 %c, label %119, label %56
119:
  ret void
}
"""
    report = tensor_compiler_observation.observe_compiler_ir(
      {"llir": llir}, expected_loop=True, loop_required_stages=("llir",)
    )

    self.assertEqual(report["stages"]["llir"]["backedge_count"], 1)
    self.assertEqual(report["loop_status"], "retained")
    self.assertEqual(report["loop_evidence"]["llir"], "branch-cycle")

  def test_mlir_construct_reports_its_own_evidence(self):
    report = tensor_compiler_observation.observe_compiler_ir(
      {"ttir": _LOOP_BODY}, expected_loop=True, loop_required_stages=("ttir",)
    )

    self.assertEqual(report["loop_evidence"]["ttir"], "mlir-construct")


class TensorCompilerBranchCycleTests(unittest.TestCase):
  """Print position is not a CFG cycle: cycles are found in a branch graph."""

  def test_acyclic_blocks_printed_out_of_execution_order_are_not_a_loop(self):
    # Review counterexample: entry branches to body, body to exit, but exit is
    # printed before body, so the branch that reaches it looks backward.
    llir = """\
define void @f() {
entry:
  br label %body
exit:
  ret void
body:
  br label %exit
}
"""
    report = tensor_compiler_observation.observe_compiler_ir(
      {"llir": llir}, expected_loop=True, loop_required_stages=("llir",)
    )

    self.assertEqual(report["stages"]["llir"]["backedge_count"], 0)
    self.assertEqual(report["loop_evidence"], {"llir": None})
    self.assertEqual(report["loop_presence"], {"llir": False})
    self.assertEqual(report["loop_status"], "unrolled_or_removed")

  def test_acyclic_diamond_with_a_conditional_exit_is_not_a_loop(self):
    llir = """\
define void @f() {
entry:
  br i1 %c, label %exit, label %then
exit:
  ret void
then:
  br label %exit
}
"""
    report = tensor_compiler_observation.observe_compiler_ir(
      {"llir": llir}, expected_loop=True, loop_required_stages=("llir",)
    )

    self.assertEqual(report["stages"]["llir"]["backedge_count"], 0)
    self.assertEqual(report["loop_status"], "unrolled_or_removed")

  def test_same_label_in_two_functions_does_not_merge(self):
    llir = """\
define void @first() {
entry:
  br label %body
body:
  br label %exit
exit:
  ret void
}
define void @second() {
entry:
  br label %body
body:
  br label %exit
exit:
  ret void
}
"""
    report = tensor_compiler_observation.observe_compiler_ir(
      {"llir": llir}, expected_loop=True, loop_required_stages=("llir",)
    )

    self.assertEqual(report["stages"]["llir"]["backedge_count"], 0)
    self.assertEqual(report["loop_status"], "unrolled_or_removed")

  def test_loop_in_one_function_survives_reused_label_names(self):
    # Scoping must keep a real cycle without importing edges from the sibling
    # function that reuses the same block names.
    llir = """\
define void @plain() {
entry:
  br label %body
body:
  br label %exit
exit:
  ret void
}
define void @looping() {
entry:
  br label %body
body:
  br label %body
exit:
  ret void
}
"""
    report = tensor_compiler_observation.observe_compiler_ir(
      {"llir": llir}, expected_loop=True, loop_required_stages=("llir",)
    )

    self.assertEqual(report["stages"]["llir"]["backedge_count"], 1)
    self.assertEqual(report["stages"]["llir"]["unparsed_branch_count"], 0)
    self.assertEqual(report["loop_status"], "retained")

  def test_two_block_loop_counts_every_cyclic_branch(self):
    llir = """\
define void @k() {
entry:
  br label %header
header:
  br i1 %c, label %body, label %exit
body:
  br label %header
exit:
  ret void
}
"""
    report = tensor_compiler_observation.observe_compiler_ir(
      {"llir": llir}, expected_loop=True, loop_required_stages=("llir",)
    )

    self.assertEqual(report["stages"]["llir"]["backedge_count"], 2)
    self.assertEqual(report["loop_evidence"]["llir"], "branch-cycle")
    self.assertEqual(report["loop_status"], "retained")

  def test_single_block_self_loop_is_a_branch_cycle(self):
    llir = """\
define void @k() {
entry:
  br label %loop
loop:
  br label %loop
}
"""
    report = tensor_compiler_observation.observe_compiler_ir(
      {"llir": llir}, expected_loop=True, loop_required_stages=("llir",)
    )

    self.assertEqual(report["stages"]["llir"]["backedge_count"], 1)
    self.assertEqual(report["loop_evidence"]["llir"], "branch-cycle")
    self.assertEqual(report["loop_status"], "retained")

  def test_lowered_scf_for_is_a_branch_cycle(self):
    lowered = """\
module {
  tt.func public @k(%n: i32) {
    cf.br ^bb1(%n : i32)
  ^bb1(%i: i32):
    %0 = arith.addi %i, %c1 : i32
    cf.cond_br %1, ^bb1(%0 : i32), ^bb2
  ^bb2:
    tt.return
  }
}
"""
    report = tensor_compiler_observation.observe_compiler_ir(
      {"ttgir": lowered}, expected_loop=True, loop_required_stages=("ttgir",)
    )

    self.assertEqual(report["stages"]["ttgir"]["loop_construct_count"], 0)
    self.assertEqual(report["stages"]["ttgir"]["backedge_count"], 1)
    self.assertEqual(report["loop_evidence"]["ttgir"], "branch-cycle")
    self.assertEqual(report["loop_status"], "retained")

  def test_generic_form_mlir_branch_is_a_branch_cycle(self):
    # Generic-form syntax prints the same branch op with its block operands in
    # a successor list instead of the custom form.
    lowered = """\
module {
  func.func @k() {
    "cf.br"(%c)[^bb1] : (i32) -> ()
  ^bb1:
    "cf.br"(%c)[^bb1] : (i32) -> ()
  }
}
"""
    report = tensor_compiler_observation.observe_compiler_ir(
      {"ttgir": lowered}, expected_loop=True, loop_required_stages=("ttgir",)
    )

    self.assertEqual(report["stages"]["ttgir"]["backedge_count"], 1)
    self.assertEqual(report["stages"]["ttgir"]["unparsed_branch_count"], 0)
    self.assertEqual(report["loop_status"], "retained")

  def test_unparsed_ptx_branch_leaves_the_stage_unavailable(self):
    ptx = """\
.visible .entry k() {
$L__BB0_1:
  bra $L__BB0_2;
$L__BB0_2:
  brx.idx.uni %r1, %tbl;
  ret;
}
"""
    report = tensor_compiler_observation.observe_compiler_ir(
      {"ptx": ptx}, expected_loop=True, loop_required_stages=("ptx",)
    )

    self.assertEqual(report["stages"]["ptx"]["unparsed_branch_count"], 1)
    self.assertEqual(report["loop_evidence"], {"ptx": None})
    self.assertIsNone(report["loop_presence"]["ptx"])
    self.assertEqual(report["loop_status"], "unavailable")

  def test_unparsed_llvm_branch_leaves_the_stage_unavailable(self):
    llir = """\
define void @k() {
entry:
  indirectbr ptr %p, [label %a, label %b]
a:
  ret void
b:
  ret void
}
"""
    report = tensor_compiler_observation.observe_compiler_ir(
      {"llir": llir}, expected_loop=True, loop_required_stages=("llir",)
    )

    self.assertEqual(report["stages"]["llir"]["unparsed_branch_count"], 1)
    self.assertIsNone(report["loop_presence"]["llir"])
    self.assertEqual(report["loop_status"], "unavailable")

  def test_ptx_fall_through_closes_a_loop_through_the_next_block(self):
    # PTX blocks need no terminator: the first block reaches the second one by
    # falling into it, and the branch of the second one closes the cycle.
    ptx = """\
.visible .entry k() {
$L__BB0_1:
  add.f32 %f1, %f2, %f3;
$L__BB0_2:
  @%p1 bra $L__BB0_1;
  ret;
}
"""
    report = tensor_compiler_observation.observe_compiler_ir(
      {"ptx": ptx}, expected_loop=True, loop_required_stages=("ptx",)
    )

    self.assertEqual(report["stages"]["ptx"]["backedge_count"], 1)
    self.assertEqual(report["loop_evidence"]["ptx"], "branch-cycle")
    self.assertEqual(report["loop_status"], "retained")

  def test_ptx_return_ends_a_block_without_falling_through(self):
    # A backward branch into a block that returns closes no cycle, because the
    # returning block does not fall into the block printed after it.
    ptx = """\
.visible .entry k() {
$L__BB0_1:
  ret;
$L__BB0_2:
  bra $L__BB0_1;
}
"""
    report = tensor_compiler_observation.observe_compiler_ir(
      {"ptx": ptx}, expected_loop=True, loop_required_stages=("ptx",)
    )

    self.assertEqual(report["stages"]["ptx"]["backedge_count"], 0)
    self.assertEqual(report["loop_presence"], {"ptx": False})
    self.assertEqual(report["loop_status"], "unrolled_or_removed")


_INLINE_LAYOUTS = """\
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>
#loc1 = loc("kernel.py":43:12)
"""


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

  def test_inline_slice_against_a_blocked_alias_is_effective(self):
    """A conversion into an inline parameterized slice changes the layout."""
    ir = (
      _INLINE_LAYOUTS
      + """\
%0 = ttg.convert_layout %a : tensor<4xi32, #blocked2> -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> loc(#loc1)
"""
    )
    report = tensor_compiler_observation.observe_compiler_ir({"ttgir": ir})
    conversion = report["layout_conversions"][0]

    self.assertTrue(conversion["layout_changed"])
    self.assertEqual(conversion["source_layout"], "#blocked2")
    self.assertEqual(
      conversion["target_layout"], "#ttg.slice<{dim = 1, parent = #blocked3}>"
    )
    self.assertEqual(report["layout_conversion_count"], 1)

  def test_equal_parameters_under_different_names_are_not_a_change(self):
    """Two aliases, and an alias against its inline spelling, hold one layout."""
    ir = (
      _INLINE_LAYOUTS
      + """\
%0 = ttg.convert_layout %a : tensor<4xi32, #blocked2> -> tensor<4xi32, #blocked3> loc(#loc1)
%1 = ttg.convert_layout %b : tensor<4xi32, #blocked2> -> tensor<4xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>> loc(#loc1)
"""
    )
    report = tensor_compiler_observation.observe_compiler_ir({"ttgir": ir})
    inline = report["layout_conversions"][1]

    self.assertEqual(
      [entry["layout_changed"] for entry in report["layout_conversions"]],
      [False, False],
    )
    self.assertEqual(report["layout_conversion_count"], 0)
    self.assertEqual(
      inline["target_layout"],
      "#ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], "
      "warpsPerCTA = [1, 1], order = [0, 1]}>",
    )
    self.assertEqual(
      inline["source_layout_definition"], inline["target_layout_definition"]
    )

  def test_nested_parameters_resolve_before_comparison(self):
    """An alias nested inside an inline encoding is compared by its parameters."""
    ir = (
      _INLINE_LAYOUTS
      + """\
#slice2 = #ttg.slice<{dim = 1, parent = #blocked2}>
#slice4 = #ttg.slice<{dim = 1, parent = #blocked4}>
%0 = ttg.convert_layout %a : tensor<4xi32, #slice2> -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> loc(#loc1)
%1 = ttg.convert_layout %b : tensor<4xi32, #slice2> -> tensor<4xi32, #slice4> loc(#loc1)
"""
    )
    report = tensor_compiler_observation.observe_compiler_ir({"ttgir": ir})
    equal, different = report["layout_conversions"]

    self.assertFalse(equal["layout_changed"])
    self.assertEqual(
      equal["target_layout"], "#ttg.slice<{dim = 1, parent = #blocked3}>"
    )
    self.assertEqual(
      equal["target_layout_definition"],
      "#ttg.slice<{dim=1,parent=#ttg.blocked<{sizePerThread=[1,4],"
      "threadsPerWarp=[1,32],warpsPerCTA=[1,1],order=[0,1]}>}>",
    )
    self.assertTrue(different["layout_changed"])

  def test_generic_quoted_encoding_resolves(self):
    """A quoted dialect name denotes the same encoding as its custom form."""
    ir = (
      _INLINE_LAYOUTS
      + """\
#slice3 = #ttg.slice<{dim = 1, parent = #blocked3}>
%0 = ttg.convert_layout %a : tensor<4xi32, #blocked2> -> tensor<4xi32, "ttg.slice"<{dim = 1, parent = #blocked3}>> loc(#loc1)
%1 = ttg.convert_layout %b : tensor<4xi32, #slice3> -> tensor<4xi32, "ttg.slice"<{dim = 1, parent = #blocked3}>> loc(#loc1)
"""
    )
    report = tensor_compiler_observation.observe_compiler_ir({"ttgir": ir})
    quoted, equal = report["layout_conversions"]

    self.assertTrue(quoted["layout_changed"])
    self.assertEqual(
      quoted["target_layout"], '"ttg.slice"<{dim = 1, parent = #blocked3}>'
    )
    self.assertEqual(
      quoted["target_layout_definition"],
      "#ttg.slice<{dim=1,parent=#ttg.blocked<{sizePerThread=[1,4],"
      "threadsPerWarp=[1,32],warpsPerCTA=[1,1],order=[0,1]}>}>",
    )
    self.assertFalse(equal["layout_changed"])
    self.assertEqual(
      equal["source_layout_definition"], equal["target_layout_definition"]
    )

  def test_a_conversion_that_prints_the_same_inline_encoding_is_not_effective(self):
    """An inline encoding equal to the other side's alias changes no layout."""
    ir = (
      _INLINE_LAYOUTS
      + """\
%0 = ttg.convert_layout %a : tensor<4xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>> -> tensor<4xi32, #blocked2> loc(#loc1)
"""
    )
    report = tensor_compiler_observation.observe_compiler_ir({"ttgir": ir})
    conversion = report["layout_conversions"][0]

    self.assertFalse(conversion["layout_changed"])
    self.assertEqual(
      conversion["source_layout"],
      "#ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], "
      "warpsPerCTA = [1, 1], order = [0, 1]}>",
    )
    self.assertEqual(conversion["target_layout"], "#blocked2")
    self.assertEqual(
      conversion["source_layout_definition"], conversion["target_layout_definition"]
    )
    self.assertEqual(report["layout_conversion_count"], 0)
    self.assertEqual(report["effective_conversion_operations"], [])

  def test_an_alias_defined_twice_with_the_same_parameters_is_not_a_change(self):
    """Repeating one alias definition leaves one layout for its every use."""
    ir = (
      _INLINE_LAYOUTS
      + """\
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>
%0 = ttg.convert_layout %a : tensor<4xi32, #blocked2> -> tensor<4xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>> loc(#loc1)
"""
    )
    report = tensor_compiler_observation.observe_compiler_ir({"ttgir": ir})
    conversion = report["layout_conversions"][0]

    self.assertFalse(conversion["layout_changed"])
    self.assertEqual(
      conversion["target_layout"],
      "#ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>",
    )
    self.assertEqual(
      conversion["source_layout_definition"], conversion["target_layout_definition"]
    )
    self.assertEqual(report["layout_conversion_count"], 0)


_REGION_LAYOUTS = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32]}>
#slice = #ttg.slice<{dim = 0, parent = #blocked}>
#loc1 = loc("kernel.py":41:12)
"""


class TensorCompilerRegionMembershipTests(unittest.TestCase):
  """A counted result list opens a region as the single-result form does.

  MLIR prints repeated results as a counted list (`%name:2 = scf.for ...`)
  instead of repeating the name, and prints an operation in generic form with a
  quoted name (`"tt.reduce"(...) ({`). Both open a region, so a conversion
  inside the loop must resolve to the loop rather than to its enclosing
  function, and a preserved fragment must not be reported as moved.
  """

  def test_counted_result_header_opens_a_region(self):
    text = (
      _REGION_LAYOUTS
      + """\
module {
  tt.func public @k(%x: tensor<4xf32, #blocked>) {
    %single = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %x) -> (tensor<4xf32>) {
      %0 = ttg.convert_layout %a : tensor<4xf32, #blocked> -> tensor<4xf32, #slice> loc(#loc1)
      scf.yield %a : tensor<4xf32>
    }
    %counted:2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%b = %x, %c = %x) -> (tensor<4xf32>, tensor<4xf32>) {
      %1 = ttg.convert_layout %b : tensor<4xf32, #blocked> -> tensor<4xf32, #slice> loc(#loc1)
      scf.yield %b, %c : tensor<4xf32>, tensor<4xf32>
    }
  }
}
"""
    )
    report = tensor_compiler_observation.observe_compiler_ir({"ttgir": text})
    regions = {
      conversion["result"]: conversion["enclosing_region"]
      for conversion in report["layout_conversions"]
    }

    # Control: the single-result header the recogniser already accepted.
    self.assertEqual(regions["%0"], "scf.for")
    self.assertEqual(regions["%1"], "scf.for")

  def test_conversion_inside_a_counted_loop_names_the_loop_in_both_snapshots(self):
    before = (
      _REGION_LAYOUTS
      + """\
module {
  tt.func public @k(%x: tensor<4xf32, #blocked>) {
    %acc:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %x, %b = %x) -> (tensor<4xf32>, tensor<4xf32>) {
      %0 = ttg.convert_layout %a : tensor<4xf32, #blocked> -> tensor<4xf32, #slice> loc(#loc1)
      scf.yield %a, %b : tensor<4xf32>, tensor<4xf32>
    }
  }
}
"""
    )
    after = (
      _REGION_LAYOUTS
      + """\
module {
  tt.func public @k(%x: tensor<4xf32, #blocked>) {
    %acc_1:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%a_1 = %x, %b_1 = %x) -> (tensor<4xf32>, tensor<4xf32>) {
      %7 = ttg.convert_layout %a_1 : tensor<4xf32, #blocked> -> tensor<4xf32, #slice> loc(#loc1)
      scf.yield %a_1, %b_1 : tensor<4xf32>, tensor<4xf32>
    }
  }
}
"""
    )
    report = tensor_compiler_observation.match_conversion_fragments(before, after)
    fragment = report["fragments"][0]

    self.assertEqual(report["matched"], 1)
    self.assertEqual(fragment["enclosing_region_before"], "scf.for")
    self.assertEqual(fragment["enclosing_region_after"], "scf.for")
    self.assertEqual(fragment["status"], "preserved")

  def test_conversion_that_enters_a_counted_loop_is_moved(self):
    before = (
      _REGION_LAYOUTS
      + """\
module {
  tt.func public @k(%x: tensor<4xf32, #blocked>) {
    %0 = ttg.convert_layout %x : tensor<4xf32, #blocked> -> tensor<4xf32, #slice> loc(#loc1)
  }
}
"""
    )
    after = (
      _REGION_LAYOUTS
      + """\
module {
  tt.func public @k(%x: tensor<4xf32, #blocked>) {
    %acc:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %x, %b = %x) -> (tensor<4xf32>, tensor<4xf32>) {
      %7 = ttg.convert_layout %x : tensor<4xf32, #blocked> -> tensor<4xf32, #slice> loc(#loc1)
      scf.yield %a, %b : tensor<4xf32>, tensor<4xf32>
    }
  }
}
"""
    )
    report = tensor_compiler_observation.match_conversion_fragments(before, after)
    fragment = report["fragments"][0]

    self.assertEqual(report["moved"], 1)
    self.assertEqual(fragment["status"], "moved")
    self.assertEqual(fragment["enclosing_region_before"], "tt.func")
    self.assertEqual(fragment["enclosing_region_after"], "scf.for")

  def test_pass_that_counts_a_loop_result_does_not_move_the_body(self):
    before = (
      _REGION_LAYOUTS
      + """\
module {
  tt.func public @k(%x: tensor<4xf32, #blocked>) {
    %acc = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %x) -> (tensor<4xf32>) {
      %0 = ttg.convert_layout %a : tensor<4xf32, #blocked> -> tensor<4xf32, #slice> loc(#loc1)
      scf.yield %a : tensor<4xf32>
    }
  }
}
"""
    )
    after = (
      _REGION_LAYOUTS
      + """\
module {
  tt.func public @k(%x: tensor<4xf32, #blocked>) {
    %acc:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %x, %b = %x) -> (tensor<4xf32>, tensor<4xf32>) {
      %7 = ttg.convert_layout %a : tensor<4xf32, #blocked> -> tensor<4xf32, #slice> loc(#loc1)
      scf.yield %a, %b : tensor<4xf32>, tensor<4xf32>
    }
  }
}
"""
    )
    report = tensor_compiler_observation.match_conversion_fragments(before, after)
    fragment = report["fragments"][0]

    self.assertEqual(report["moved"], 0)
    self.assertEqual(fragment["status"], "preserved")
    self.assertEqual(fragment["enclosing_region_before"], "scf.for")
    self.assertEqual(fragment["enclosing_region_after"], "scf.for")

  def test_nested_region_inside_a_counted_loop_resolves_to_the_innermost(self):
    text = (
      _REGION_LAYOUTS
      + """\
module {
  tt.func public @k(%x: tensor<4xf32, #blocked>) {
    %acc:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %x, %b = %x) -> (tensor<4xf32>, tensor<4xf32>) {
      %r = "tt.reduce"(%a) <{axis = 0 : i32}> ({
      ^bb0(%left: f32, %right: f32):
        %inner = ttg.convert_layout %a : tensor<4xf32, #blocked> -> tensor<4xf32, #slice> loc(#loc1)
        tt.reduce.return %left : f32
      }) : (tensor<4xf32, #blocked>) -> tensor<f32, #slice> loc(#loc1)
      %1 = ttg.convert_layout %a : tensor<4xf32, #blocked> -> tensor<4xf32, #slice> loc(#loc1)
      scf.yield %a, %b : tensor<4xf32>, tensor<4xf32>
    }
  }
}
"""
    )
    report = tensor_compiler_observation.observe_compiler_ir({"ttgir": text})
    regions = {
      conversion["result"]: conversion["enclosing_region"]
      for conversion in report["layout_conversions"]
    }

    self.assertEqual(regions["%inner"], "tt.reduce")
    self.assertEqual(regions["%1"], "scf.for")

  def test_counted_result_header_needs_no_spaces(self):
    text = (
      _REGION_LAYOUTS
      + """\
module {
  tt.func public @k(%x: tensor<4xf32, #blocked>) {
    %acc_y:12=scf.for %i=%c0 to %c4 step %c1 iter_args(%a=%x, %b=%x) -> (tensor<4xf32>, tensor<4xf32>) {
      %1 = ttg.convert_layout %a : tensor<4xf32, #blocked> -> tensor<4xf32, #slice> loc(#loc1)
      scf.yield %a, %b : tensor<4xf32>, tensor<4xf32>
    }
  }
}
"""
    )
    report = tensor_compiler_observation.observe_compiler_ir({"ttgir": text})

    self.assertEqual(report["layout_conversions"][0]["enclosing_region"], "scf.for")


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

  def test_renamed_counterexample_is_not_rematerialization(self):
    # Review counterexample: the same two operations are printed with new SSA
    # numbers. Nothing was recomputed, and the old name did not lose a use.
    before = "%0 = arith.addf %a, %b : f32\n%1 = arith.mulf %0, %c : f32\n"
    after = "%10 = arith.addf %a, %b : f32\n%11 = arith.mulf %10, %c : f32\n"

    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["status"], "none")
    self.assertEqual(report["confirmed"], [])
    self.assertEqual(report["confirmed_count"], 0)

  def test_renumbered_snapshot_with_an_extra_copy_stays_unattributed(self):
    # An extra definition exists, but the original name is gone: which copy
    # continues which producer is ambiguous, so no use is attributed.
    after = """\
module {
  %7 = arith.addf %a, %b : tensor<4xf32>
  %8 = arith.addf %a, %b : tensor<4xf32>
  %9 = arith.mulf %8, %c : tensor<4xf32>
}
"""
    report = tensor_compiler_observation.observe_rematerialization(self._BEFORE, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["unattributed_count"], 1)
    self.assertEqual(report["status"], "unattributed_only")

  def test_two_recomputed_candidates_stay_unattributed(self):
    # Two equivalent copies took over consumers of one original: the producer
    # correspondence is ambiguous, so neither copy is confirmed by name.
    after = """\
module {
  %1 = arith.addf %a, %b : tensor<4xf32>
  %7 = arith.addf %a, %b : tensor<4xf32>
  %2 = arith.mulf %7, %c : tensor<4xf32>
  %8 = arith.addf %a, %b : tensor<4xf32>
  %9 = arith.mulf %8, %c : tensor<4xf32>
  %10 = arith.addf %2, %9 : tensor<4xf32>
}
"""
    report = tensor_compiler_observation.observe_rematerialization(self._BEFORE, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["unattributed_count"], 1)
    self.assertEqual(report["status"], "unattributed_only")

  def test_reused_ssa_name_for_a_different_definition_is_not_correspondence(self):
    # %1 survives the pass but now denotes a load. A name that changed meaning
    # is not the original producer, and the addf definition was not duplicated.
    after = """\
module {
  %1 = tt.load %p : tensor<4xf32>
  %10 = arith.addf %a, %b : tensor<4xf32>
  %11 = arith.mulf %10, %c : tensor<4xf32>
}
"""
    report = tensor_compiler_observation.observe_rematerialization(self._BEFORE, after)

    self.assertEqual(report["status"], "none")
    self.assertEqual(report["confirmed_count"], 0)

  def test_renamed_duplicate_without_an_extra_definition_is_not_confirmed(self):
    # The snapshot holds the same number of addf definitions before and after,
    # so a redirected-looking use is a rename of one duplicate, not extra work.
    before = """\
module {
  %1 = arith.addf %a, %b : tensor<4xf32>
  %4 = arith.addf %a, %b : tensor<4xf32>
  %2 = arith.mulf %1, %c : tensor<4xf32>
  %5 = arith.mulf %4, %d : tensor<4xf32>
}
"""
    after = """\
module {
  %10 = arith.addf %a, %b : tensor<4xf32>
  %4 = arith.addf %a, %b : tensor<4xf32>
  %11 = arith.mulf %10, %c : tensor<4xf32>
  %5 = arith.mulf %4, %d : tensor<4xf32>
}
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["unattributed_count"], 1)
    self.assertEqual(report["status"], "unattributed_only")

  def test_review_counterexample_with_a_rebound_name_stays_unattributed(self):
    # Review counterexample: the add and its consumer are renumbered, %0 now
    # holds a load, and an unrelated but equal add feeds a different consumer.
    # The reused name does not denote the original producer, so the duplicate is
    # unattributed work rather than a confirmed recomputation.
    before = "%0 = arith.addf %a, %b : f32\n%1 = arith.mulf %0, %c : f32\n"
    after = """\
%0 = tt.load %p : f32
%10 = arith.addf %a, %b : f32
%11 = arith.mulf %10, %c : f32
%12 = arith.addf %a, %b : f32
%13 = arith.mulf %12, %d : f32
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["status"], "unattributed_only")
    self.assertEqual(report["unattributed_count"], 1)
    self.assertEqual(report["unattributed"][0]["values"], ["%10", "%12"])

  def test_reused_name_holding_a_different_pure_expression_is_not_the_producer(self):
    # %0 survives the pass but now computes a subtraction. A definition that
    # differs from the original canonical expression is not a correspondence,
    # even though the name is unchanged.
    before = "%0 = arith.addf %a, %b : f32\n%1 = arith.mulf %0, %c : f32\n"
    after = """\
%0 = arith.subf %a, %b : f32
%10 = arith.addf %a, %b : f32
%11 = arith.mulf %10, %c : f32
%12 = arith.addf %a, %b : f32
%13 = arith.mulf %12, %d : f32
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["status"], "unattributed_only")

  def test_reused_name_holding_an_impure_producer_is_not_the_producer(self):
    # %0 survives the pass but now holds a call. An impure definition has no
    # canonical expression to compare, so it cannot be the original producer.
    before = "%0 = arith.addf %a, %b : f32\n%1 = arith.mulf %0, %c : f32\n"
    after = """\
%0 = tt.call @helper(%a) : (f32) -> f32
%10 = arith.addf %a, %b : f32
%11 = arith.mulf %10, %c : f32
%12 = arith.addf %a, %b : f32
%13 = arith.mulf %12, %d : f32
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["status"], "unattributed_only")

  def test_equivalent_surviving_producer_still_confirms_rematerialization(self):
    # The review counterexample with %0 still computing the add: the producer
    # survived with the same definition, so the copy and its redirected
    # consumer are a genuine recomputation.
    before = "%0 = arith.addf %a, %b : f32\n%1 = arith.mulf %0, %c : f32\n"
    after = """\
%0 = arith.addf %a, %b : f32
%10 = arith.addf %a, %b : f32
%11 = arith.mulf %10, %c : f32
%12 = arith.addf %a, %b : f32
%13 = arith.mulf %12, %d : f32
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 1)
    candidate = report["confirmed"][0]
    self.assertEqual(candidate["original_value"], "%0")
    self.assertEqual(candidate["recomputed_values"], ["%10"])
    self.assertEqual(candidate["redirected_consumers"], ["%11"])

  def test_renumbered_graph_with_a_surviving_name_is_not_rematerialization(self):
    # Only the consumer is renumbered. No definition was added, so the graph is
    # unchanged and nothing was recomputed, even though the surviving name
    # %0 appears to have lost its use.
    before = "%0 = arith.addf %a, %b : f32\n%1 = arith.mulf %0, %c : f32\n"
    after = "%0 = arith.addf %a, %b : f32\n%11 = arith.mulf %0, %c : f32\n"
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["unattributed_count"], 0)
    self.assertEqual(report["status"], "none")

  def test_reused_consumer_name_is_not_a_matching_consumer(self):
    # The former consumer is a call that %0 fed; after the pass the printed name
    # %1 holds a load of the copy. Names are rebound freely across snapshots, so
    # an impure definition, which canonicalizes to its own name, cannot witness
    # a redirected use by name coincidence.
    before = """\
%0 = arith.addf %a, %b : f32
%1 = tt.call @consume(%0) : (f32) -> f32
"""
    after = """\
%0 = arith.addf %a, %b : f32
%10 = arith.addf %a, %b : f32
%1 = tt.load %10 : f32
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["unattributed_count"], 1)
    self.assertEqual(report["status"], "unattributed_only")

  def test_one_copy_claimed_by_two_producers_stays_unattributed(self):
    # A single new copy serves a former consumer of each of two equivalent
    # producers: which producer it recomputes is ambiguous, so neither is named.
    before = """\
module {
  %1 = arith.addf %a, %b : tensor<4xf32>
  %4 = arith.addf %a, %b : tensor<4xf32>
  %2 = arith.mulf %1, %c : tensor<4xf32>
  %5 = arith.mulf %4, %d : tensor<4xf32>
}
"""
    after = """\
module {
  %1 = arith.addf %a, %b : tensor<4xf32>
  %4 = arith.addf %a, %b : tensor<4xf32>
  %7 = arith.addf %a, %b : tensor<4xf32>
  %2 = arith.mulf %7, %c : tensor<4xf32>
  %5 = arith.mulf %7, %d : tensor<4xf32>
}
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["unattributed_count"], 1)
    self.assertEqual(report["status"], "unattributed_only")

  def test_reviewer_alpha_renaming_pair_is_not_rematerialization(self):
    # Review counterexample: one simultaneous bijective renaming of the printed
    # names changes neither the operations nor the def-use edges, but it does
    # swap which pointer each load result reads. The addition that appears to
    # have survived no longer computes what the original computed, so no use was
    # redirected to a recomputation.
    before = """\
%lx = tt.load %p : f32
%ly = tt.load %q : f32
%0 = arith.addf %lx, %c : f32
%1 = arith.mulf %0, %d : f32
%8 = arith.mulf %0, %g : f32
%2 = arith.addf %ly, %c : f32
%3 = arith.mulf %2, %d : f32
%4 = arith.addf %ly, %c : f32
%5 = arith.mulf %4, %e : f32
%6 = arith.addf %ly, %c : f32
%7 = arith.mulf %6, %f : f32
tt.return %1, %8, %3, %5, %7 : f32, f32, f32, f32, f32
"""
    after = """\
%ly = tt.load %p : f32
%lx = tt.load %q : f32
%20 = arith.addf %ly, %c : f32
%21 = arith.mulf %20, %d : f32
%28 = arith.mulf %20, %g : f32
%10 = arith.addf %lx, %c : f32
%11 = arith.mulf %10, %d : f32
%12 = arith.addf %lx, %c : f32
%13 = arith.mulf %12, %e : f32
%0 = arith.addf %lx, %c : f32
%14 = arith.mulf %0, %f : f32
tt.return %21, %28, %11, %13, %14 : f32, f32, f32, f32, f32
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["confirmed"], [])
    self.assertEqual(report["unattributed_count"], 1)
    self.assertEqual(report["status"], "unattributed_only")

  def test_swapped_loads_of_different_pointers_are_not_interchangeable(self):
    # The two load results exchange their pointers and nothing else changes, so
    # the addition that looks like a surviving producer now reads %q: neither it
    # nor its printed twin computes the value the original computed.
    before = """\
%lx = tt.load %p : f32
%ly = tt.load %q : f32
%0 = arith.addf %lx, %c : f32
%1 = arith.mulf %0, %d : f32
%2 = arith.addf %ly, %c : f32
%3 = arith.mulf %2, %d : f32
"""
    after = """\
%lx = tt.load %q : f32
%ly = tt.load %p : f32
%0 = arith.addf %lx, %c : f32
%10 = arith.addf %lx, %c : f32
%11 = arith.mulf %10, %d : f32
%2 = arith.addf %ly, %c : f32
%3 = arith.mulf %2, %d : f32
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["confirmed"], [])
    self.assertEqual(report["unattributed_count"], 1)
    self.assertEqual(report["status"], "unattributed_only")

  def test_leaf_defined_in_only_one_snapshot_is_unproven(self):
    # A leaf that a definition names on one side and that has no definition on
    # the other (an argument, or a value defined outside the snapshot) cannot be
    # resolved the same way twice, so neither direction confirms a copy.
    defined_before = """\
%a = tt.load %p : f32
%0 = arith.addf %a, %c : f32
%1 = arith.mulf %0, %d : f32
"""
    undefined_before = """\
%0 = arith.addf %a, %c : f32
%1 = arith.mulf %0, %d : f32
"""
    redirected = """\
%0 = arith.addf %a, %c : f32
%10 = arith.addf %a, %c : f32
%11 = arith.mulf %10, %d : f32
"""
    defined_after = "%a = tt.load %p : f32\n" + redirected

    for label, before, after in (
      ("defined before only", defined_before, redirected),
      ("defined after only", undefined_before, defined_after),
    ):
      with self.subTest(case=label):
        report = tensor_compiler_observation.observe_rematerialization(before, after)

        self.assertEqual(report["confirmed_count"], 0)
        self.assertEqual(report["unattributed_count"], 1)
        self.assertEqual(report["status"], "unattributed_only")

  def test_changed_leaf_definition_is_not_the_same_computation(self):
    # The load now reads a different pointer: the surviving addition and its
    # copy still print the same names, but they no longer compute the value the
    # original computed, so the redirected consumer is not a recomputation.
    before = """\
%a = tt.load %p : f32
%0 = arith.addf %a, %c : f32
%1 = arith.mulf %0, %d : f32
"""
    after = """\
%a = tt.load %q : f32
%0 = arith.addf %a, %c : f32
%10 = arith.addf %a, %c : f32
%11 = arith.mulf %10, %d : f32
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["confirmed"], [])
    self.assertEqual(report["unattributed_count"], 1)
    self.assertEqual(report["status"], "unattributed_only")

  def test_renamed_pure_chain_with_a_renamed_leaf_still_confirms(self):
    # The whole chain feeding the surviving producer was renumbered, including
    # the load result at its leaf. The chain computes the same value, so the
    # copy that took over its consumer is a confirmed recomputation.
    before = """\
%lx = tt.load %p : f32
%1 = arith.addf %lx, %c : f32
%2 = arith.mulf %1, %d : f32
%3 = arith.addf %2, %e : f32
%4 = arith.mulf %3, %f : f32
"""
    after = """\
%lz = tt.load %p : f32
%11 = arith.addf %lz, %c : f32
%12 = arith.mulf %11, %d : f32
%3 = arith.addf %12, %e : f32
%13 = arith.addf %12, %e : f32
%14 = arith.mulf %13, %f : f32
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 1)
    candidate = report["confirmed"][0]
    self.assertEqual(candidate["original_value"], "%3")
    self.assertEqual(candidate["recomputed_values"], ["%13"])
    self.assertEqual(candidate["redirected_consumers"], ["%14"])
    self.assertEqual(candidate["remaining_consumers"], [])

  def test_leaf_resolution_differs_between_snapshots_is_unproven(self):
    # The operand is named %y on both sides, but a definition identifies it on
    # one side only: before the pass %x is an unparsed definition whose text is
    # %y, after it %y is an argument. A defined leaf is substituted inside
    # parentheses and an undefined one is not, so the two are never confused.
    before = """\
%x = %y
%0 = arith.addf %x, %c : f32
%1 = arith.mulf %0, %d : f32
"""
    after = """\
%0 = arith.addf %y, %c : f32
%10 = arith.addf %y, %c : f32
%11 = arith.mulf %10, %d : f32
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["unattributed_count"], 1)
    self.assertEqual(report["status"], "unattributed_only")

  def test_two_loads_of_the_same_pointer_swapped_are_not_a_difference(self):
    # Both load results read %p, so exchanging their names leaves every
    # computation unchanged: the copy that took over the consumer is still a
    # recomputation of the addition that survived, exactly as without the swap.
    before = """\
%lx = tt.load %p : f32
%ly = tt.load %p : f32
%0 = arith.addf %lx, %c : f32
%1 = arith.mulf %0, %d : f32
%2 = arith.mulf %ly, %e : f32
"""
    after = """\
%ly = tt.load %p : f32
%lx = tt.load %p : f32
%0 = arith.addf %ly, %c : f32
%10 = arith.addf %ly, %c : f32
%11 = arith.mulf %10, %d : f32
%2 = arith.mulf %lx, %e : f32
"""
    unswapped = """\
%lx = tt.load %p : f32
%ly = tt.load %p : f32
%0 = arith.addf %lx, %c : f32
%10 = arith.addf %lx, %c : f32
%11 = arith.mulf %10, %d : f32
%2 = arith.mulf %ly, %e : f32
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)
    control = tensor_compiler_observation.observe_rematerialization(before, unswapped)

    self.assertEqual(control["confirmed_count"], 1)
    self.assertEqual(report["confirmed_count"], 1)
    for observation in (control, report):
      candidate = observation["confirmed"][0]
      self.assertEqual(candidate["original_value"], "%0")
      self.assertEqual(candidate["recomputed_values"], ["%10"])
      self.assertEqual(candidate["redirected_consumers"], ["%11"])
      self.assertEqual(candidate["remaining_consumers"], [])

  def test_a_pointer_whose_computation_changed_is_not_the_same_pointer(self):
    """A load denotes its pointer's computation, not the pointer's name.

    The pointer keeps its printed name while its definition changes, so a load
    of it computes something else. The two additions below are textually equal
    and are not the same computation, and the observer has to say so.
    """
    before = """\
%p = arith.addi %a, %b : i32
%lx = tt.load %p : f32
%c = arith.addf %lx, %d : f32
%1 = arith.mulf %c, %e : f32
%2 = arith.mulf %c, %f : f32
"""
    after = """\
%p = arith.subi %a, %b : i32
%lx = tt.load %p : f32
%c = arith.addf %lx, %d : f32
%1 = arith.mulf %c, %e : f32
%c2 = arith.addf %lx, %d : f32
%2 = arith.mulf %c2, %f : f32
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["confirmed"], [])

  def test_a_renamed_pointer_keeps_a_genuine_recomputation_confirmable(self):
    """Resolution follows the computation, so a renamed pointer still matches."""
    before = """\
%p = arith.addi %a, %b : i32
%lx = tt.load %p : f32
%c = arith.addf %lx, %d : f32
%1 = arith.mulf %c, %e : f32
%2 = arith.mulf %c, %f : f32
"""
    after = """\
%p9 = arith.addi %a, %b : i32
%lx = tt.load %p9 : f32
%c = arith.addf %lx, %d : f32
%1 = arith.mulf %c, %e : f32
%c2 = arith.addf %lx, %d : f32
%2 = arith.mulf %c2, %f : f32
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 1)
    self.assertEqual(report["confirmed"][0]["original_value"], "%c")
    self.assertEqual(report["confirmed"][0]["recomputed_values"], ["%c2"])

  def test_reviewer_chain_pair_is_not_rematerialization(self):
    """The reviewed counterexample: four pointer steps and one bijective renaming.

    The graph keeps every operation and every def-use edge; only the printed
    names change, and the renaming swaps which pointer each load reads. The
    resolution budget cannot complete a four-step chain here, so the leftover
    definition text prints a name the snapshot defines, and that name is not an
    identity: neither addition is a recomputation of the other.
    """
    before = """\
%p0 = tt.addptr %ptrp, %off : !tt.ptr<f32>, i32
%p1 = tt.addptr %p0, %off : !tt.ptr<f32>, i32
%p2 = tt.addptr %p1, %off : !tt.ptr<f32>, i32
%p = tt.addptr %p2, %off : !tt.ptr<f32>, i32
%q0 = tt.addptr %ptrq, %off : !tt.ptr<f32>, i32
%q1 = tt.addptr %q0, %off : !tt.ptr<f32>, i32
%q2 = tt.addptr %q1, %off : !tt.ptr<f32>, i32
%q = tt.addptr %q2, %off : !tt.ptr<f32>, i32
%lx = tt.load %p : f32
%ly = tt.load %q : f32
%0 = arith.addf %lx, %c : f32
%1 = arith.mulf %0, %d : f32
%8 = arith.mulf %0, %g : f32
%2 = arith.addf %ly, %c : f32
%3 = arith.mulf %2, %d : f32
%4 = arith.addf %ly, %c : f32
%5 = arith.mulf %4, %e : f32
%6 = arith.addf %ly, %c : f32
%7 = arith.mulf %6, %f : f32
tt.return %1, %8, %3, %5, %7 : f32, f32, f32, f32, f32
"""
    after = """\
%q0 = tt.addptr %ptrp, %off : !tt.ptr<f32>, i32
%q1 = tt.addptr %q0, %off : !tt.ptr<f32>, i32
%q2 = tt.addptr %q1, %off : !tt.ptr<f32>, i32
%q = tt.addptr %q2, %off : !tt.ptr<f32>, i32
%p0 = tt.addptr %ptrq, %off : !tt.ptr<f32>, i32
%p1 = tt.addptr %p0, %off : !tt.ptr<f32>, i32
%p2 = tt.addptr %p1, %off : !tt.ptr<f32>, i32
%p = tt.addptr %p2, %off : !tt.ptr<f32>, i32
%ly = tt.load %q : f32
%lx = tt.load %p : f32
%20 = arith.addf %ly, %c : f32
%21 = arith.mulf %20, %d : f32
%28 = arith.mulf %20, %g : f32
%10 = arith.addf %lx, %c : f32
%11 = arith.mulf %10, %d : f32
%12 = arith.addf %lx, %c : f32
%13 = arith.mulf %12, %e : f32
%0 = arith.addf %lx, %c : f32
%14 = arith.mulf %0, %f : f32
tt.return %21, %28, %11, %13, %14 : f32, f32, f32, f32, f32
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["confirmed"], [])
    self.assertEqual(report["unattributed_count"], 1)
    self.assertEqual(report["status"], "unattributed_only")

  def test_a_deeper_reviewer_chain_pair_is_not_rematerialization(self):
    """The same unchanged graph with one more pointer step per chain.

    The reviewed reproduction also confirmed at five steps. Refusal must not
    depend on how far past the budget the chain reaches, so the same bijective
    renaming over a five-step chain stays unconfirmed as well.
    """
    before = """\
%p0 = tt.addptr %ptrp, %off : !tt.ptr<f32>, i32
%p1 = tt.addptr %p0, %off : !tt.ptr<f32>, i32
%p2 = tt.addptr %p1, %off : !tt.ptr<f32>, i32
%p3 = tt.addptr %p2, %off : !tt.ptr<f32>, i32
%p = tt.addptr %p3, %off : !tt.ptr<f32>, i32
%q0 = tt.addptr %ptrq, %off : !tt.ptr<f32>, i32
%q1 = tt.addptr %q0, %off : !tt.ptr<f32>, i32
%q2 = tt.addptr %q1, %off : !tt.ptr<f32>, i32
%q3 = tt.addptr %q2, %off : !tt.ptr<f32>, i32
%q = tt.addptr %q3, %off : !tt.ptr<f32>, i32
%lx = tt.load %p : f32
%ly = tt.load %q : f32
%0 = arith.addf %lx, %c : f32
%1 = arith.mulf %0, %d : f32
%8 = arith.mulf %0, %g : f32
%2 = arith.addf %ly, %c : f32
%3 = arith.mulf %2, %d : f32
%4 = arith.addf %ly, %c : f32
%5 = arith.mulf %4, %e : f32
%6 = arith.addf %ly, %c : f32
%7 = arith.mulf %6, %f : f32
tt.return %1, %8, %3, %5, %7 : f32, f32, f32, f32, f32
"""
    after = """\
%q0 = tt.addptr %ptrp, %off : !tt.ptr<f32>, i32
%q1 = tt.addptr %q0, %off : !tt.ptr<f32>, i32
%q2 = tt.addptr %q1, %off : !tt.ptr<f32>, i32
%q3 = tt.addptr %q2, %off : !tt.ptr<f32>, i32
%q = tt.addptr %q3, %off : !tt.ptr<f32>, i32
%p0 = tt.addptr %ptrq, %off : !tt.ptr<f32>, i32
%p1 = tt.addptr %p0, %off : !tt.ptr<f32>, i32
%p2 = tt.addptr %p1, %off : !tt.ptr<f32>, i32
%p3 = tt.addptr %p2, %off : !tt.ptr<f32>, i32
%p = tt.addptr %p3, %off : !tt.ptr<f32>, i32
%ly = tt.load %q : f32
%lx = tt.load %p : f32
%20 = arith.addf %ly, %c : f32
%21 = arith.mulf %20, %d : f32
%28 = arith.mulf %20, %g : f32
%10 = arith.addf %lx, %c : f32
%11 = arith.mulf %10, %d : f32
%12 = arith.addf %lx, %c : f32
%13 = arith.mulf %12, %e : f32
%0 = arith.addf %lx, %c : f32
%14 = arith.mulf %0, %f : f32
tt.return %21, %28, %11, %13, %14 : f32, f32, f32, f32, f32
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["confirmed"], [])
    self.assertEqual(report["unattributed_count"], 1)
    self.assertEqual(report["status"], "unattributed_only")

  def test_a_chain_one_step_past_the_budget_stays_unconfirmed(self):
    """A copy behind a chain the budget cannot resolve is never confirmed.

    In this shape the budget completes two pointer steps from the consumer that
    has to be matched, so this three-step chain runs out at a definition the
    snapshot holds. The pointer computations are then not compared at all, and
    the copy stays unconfirmed even though it looks like a recomputation.
    """
    before = """\
%p0 = tt.addptr %ptrp, %off : !tt.ptr<f32>, i32
%p1 = tt.addptr %p0, %off : !tt.ptr<f32>, i32
%p = tt.addptr %p1, %off : !tt.ptr<f32>, i32
%lx = tt.load %p : f32
%c = arith.addf %lx, %d : f32
%1 = arith.mulf %c, %e : f32
%2 = arith.mulf %c, %f : f32
"""
    after = """\
%p0 = tt.addptr %ptrp, %off : !tt.ptr<f32>, i32
%p1 = tt.addptr %p0, %off : !tt.ptr<f32>, i32
%p = tt.addptr %p1, %off : !tt.ptr<f32>, i32
%lx = tt.load %p : f32
%c = arith.addf %lx, %d : f32
%1 = arith.mulf %c, %e : f32
%c2 = arith.addf %lx, %d : f32
%2 = arith.mulf %c2, %f : f32
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["confirmed"], [])
    self.assertEqual(report["unattributed_count"], 1)
    self.assertEqual(report["status"], "unattributed_only")

  def test_a_chain_at_the_budget_still_confirms_a_genuine_copy(self):
    """A chain the budget completes still resolves, including a renamed one.

    Two pointer steps are the deepest this shape resolves end to end. The whole
    chain and the load result are renumbered, and the copy that took over a
    consumer is confirmed: resolution still follows the computation rather than
    the printed names wherever the budget reaches it.
    """
    before = """\
%p0 = tt.addptr %ptrp, %off : !tt.ptr<f32>, i32
%p = tt.addptr %p0, %off : !tt.ptr<f32>, i32
%lx = tt.load %p : f32
%c = arith.addf %lx, %d : f32
%1 = arith.mulf %c, %e : f32
%2 = arith.mulf %c, %f : f32
"""
    after = """\
%p9 = tt.addptr %ptrp, %off : !tt.ptr<f32>, i32
%p = tt.addptr %p9, %off : !tt.ptr<f32>, i32
%lz = tt.load %p : f32
%c = arith.addf %lz, %d : f32
%1 = arith.mulf %c, %e : f32
%c2 = arith.addf %lz, %d : f32
%2 = arith.mulf %c2, %f : f32
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 1)
    candidate = report["confirmed"][0]
    self.assertEqual(candidate["original_value"], "%c")
    self.assertEqual(candidate["recomputed_values"], ["%c2"])
    self.assertEqual(candidate["redirected_consumers"], ["%2"])
    self.assertEqual(candidate["remaining_consumers"], ["%1"])

  def test_an_impure_chain_past_the_budget_stays_unconfirmed(self):
    """A load whose pointer chain runs past the budget is not a resolved leaf.

    Only the deepest pointer step differs between the snapshots, and the budget
    cannot reach it: the load therefore computes a different address while its
    remaining text still matches, so no copy of it is confirmed.
    """
    before = """\
%p0 = tt.addptr %ptrp, %off : !tt.ptr<f32>, i32
%p1 = tt.addptr %p0, %off : !tt.ptr<f32>, i32
%p2 = tt.addptr %p1, %off : !tt.ptr<f32>, i32
%p = tt.addptr %p2, %off : !tt.ptr<f32>, i32
%lx = tt.load %p : f32
%c = arith.addf %lx, %d : f32
%1 = arith.mulf %c, %e : f32
%2 = arith.mulf %c, %f : f32
"""
    after = """\
%p0 = tt.addptr %ptrp, %off2 : !tt.ptr<f32>, i32
%p1 = tt.addptr %p0, %off : !tt.ptr<f32>, i32
%p2 = tt.addptr %p1, %off : !tt.ptr<f32>, i32
%p = tt.addptr %p2, %off : !tt.ptr<f32>, i32
%lx = tt.load %p : f32
%c = arith.addf %lx, %d : f32
%1 = arith.mulf %c, %e : f32
%c2 = arith.addf %lx, %d : f32
%2 = arith.mulf %c2, %f : f32
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["confirmed"], [])
    self.assertEqual(report["unattributed_count"], 1)
    self.assertEqual(report["status"], "unattributed_only")

  def test_a_rematerialization_behind_a_short_chain_still_confirms(self):
    """A short chain stays inside the budget, so a resolved copy confirms."""
    before = """\
%p = tt.addptr %ptrp, %off : !tt.ptr<f32>, i32
%lx = tt.load %p : f32
%c = arith.addf %lx, %d : f32
%1 = arith.mulf %c, %e : f32
%2 = arith.mulf %c, %f : f32
"""
    after = """\
%p = tt.addptr %ptrp, %off : !tt.ptr<f32>, i32
%lx = tt.load %p : f32
%c = arith.addf %lx, %d : f32
%1 = arith.mulf %c, %e : f32
%c2 = arith.addf %lx, %d : f32
%2 = arith.mulf %c2, %f : f32
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 1)
    candidate = report["confirmed"][0]
    self.assertEqual(candidate["original_value"], "%c")
    self.assertEqual(candidate["recomputed_values"], ["%c2"])
    self.assertEqual(candidate["redirected_consumers"], ["%2"])
    self.assertEqual(candidate["remaining_consumers"], ["%1"])

  def test_two_equal_chains_past_the_budget_stay_unconfirmed(self):
    """Equal but unresolvable pointer chains confirm neither copy.

    Both pointer chains are printed identically before and after, and the copy
    over %ly looks exactly like a genuine recomputation. The budget cannot
    establish what either load reads, so equality of the remaining text is not
    evidence and nothing is confirmed.
    """
    before = """\
%p0 = tt.addptr %ptrp, %off : !tt.ptr<f32>, i32
%p1 = tt.addptr %p0, %off : !tt.ptr<f32>, i32
%p2 = tt.addptr %p1, %off : !tt.ptr<f32>, i32
%p = tt.addptr %p2, %off : !tt.ptr<f32>, i32
%q0 = tt.addptr %ptrq, %off : !tt.ptr<f32>, i32
%q1 = tt.addptr %q0, %off : !tt.ptr<f32>, i32
%q2 = tt.addptr %q1, %off : !tt.ptr<f32>, i32
%q = tt.addptr %q2, %off : !tt.ptr<f32>, i32
%lx = tt.load %p : f32
%ly = tt.load %q : f32
%cx = arith.addf %lx, %d : f32
%1 = arith.mulf %cx, %e : f32
%2 = arith.mulf %cx, %f : f32
%cy = arith.addf %ly, %d : f32
%3 = arith.mulf %cy, %e : f32
%4 = arith.mulf %cy, %f : f32
"""
    after = """\
%p0 = tt.addptr %ptrp, %off : !tt.ptr<f32>, i32
%p1 = tt.addptr %p0, %off : !tt.ptr<f32>, i32
%p2 = tt.addptr %p1, %off : !tt.ptr<f32>, i32
%p = tt.addptr %p2, %off : !tt.ptr<f32>, i32
%q0 = tt.addptr %ptrq, %off : !tt.ptr<f32>, i32
%q1 = tt.addptr %q0, %off : !tt.ptr<f32>, i32
%q2 = tt.addptr %q1, %off : !tt.ptr<f32>, i32
%q = tt.addptr %q2, %off : !tt.ptr<f32>, i32
%lx = tt.load %p : f32
%ly = tt.load %q : f32
%cx = arith.addf %lx, %d : f32
%1 = arith.mulf %cx, %e : f32
%2 = arith.mulf %cx, %f : f32
%cy = arith.addf %ly, %d : f32
%3 = arith.mulf %cy, %e : f32
%cy2 = arith.addf %ly, %d : f32
%4 = arith.mulf %cy2, %f : f32
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["confirmed"], [])
    self.assertEqual(report["status"], "unattributed_only")


class TensorCompilerLayoutRematerializationTests(unittest.TestCase):
  """A recomputation in another tensor encoding is the same computation."""

  _LAYOUTS = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked_a = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
"""
  # The record keeps the encoding as the definition resolves it, so a reader
  # compares layouts rather than alias names.
  _BLOCKED = (
    "#ttg.blocked<{sizePerThread=[1,1],threadsPerWarp=[2,16],"
    "warpsPerCTA=[1,1],order=[1,0]}>"
  )
  _BLOCKED_ALT = (
    "#ttg.blocked<{sizePerThread=[1,4],threadsPerWarp=[8,4],"
    "warpsPerCTA=[1,1],order=[1,0]}>"
  )

  def _snapshot(self, body):
    return self._LAYOUTS + "module {\n" + body + "}\n"

  def test_copy_computed_in_another_encoding_confirms_rematerialization(self):
    # The copy computes the same broadcast from the same source for the consumer
    # that asked for another layout. The printed encodings differ; the
    # computation does not, so the redirected consumer confirms it.
    before = self._snapshot("""\
  %b = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>
  %d = arith.addf %b, %b : tensor<32x16xf32, #blocked>
  %e = arith.addf %b, %b : tensor<32x16xf32, #blocked>
""")
    after = self._snapshot("""\
  %b = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>
  %b2 = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked_alt>
  %d = arith.addf %b2, %b2 : tensor<32x16xf32, #blocked_alt>
  %e = arith.addf %b, %b : tensor<32x16xf32, #blocked>
""")
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["status"], "confirmed_only")
    self.assertEqual(report["confirmed_count"], 1)
    candidate = report["confirmed"][0]
    self.assertEqual(candidate["original_value"], "%b")
    self.assertEqual(candidate["recomputed_values"], ["%b2"])
    self.assertEqual(candidate["redirected_consumers"], ["%d"])
    self.assertEqual(candidate["remaining_consumers"], ["%e"])

  def test_rename_in_the_same_encoding_is_still_not_rematerialization(self):
    # The copy keeps the encoding the original had, so this is the recorded
    # rename: no additional computation exists and nothing was recomputed.
    before = self._snapshot("""\
  %b = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>
  %d = arith.addf %b, %b : tensor<32x16xf32, #blocked>
  %e = arith.addf %b, %b : tensor<32x16xf32, #blocked>
""")
    after = self._snapshot("""\
  %b = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>
  %b2 = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>
  %d = arith.addf %b2, %b2 : tensor<32x16xf32, #blocked>
  %e = arith.addf %b, %b : tensor<32x16xf32, #blocked>
""")
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["status"], "unattributed_only")
    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["unattributed_count"], 1)
    self.assertEqual(report["unattributed"][0]["values"], ["%b", "%b2"])

  def test_unrelated_duplicate_in_another_encoding_is_still_not_confirmed(self):
    # The copy in the other encoding serves a new consumer while the original
    # keeps every use it had: duplicated work, not a redirected recomputation.
    before = self._snapshot("""\
  %b = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>
  %d = arith.addf %b, %b : tensor<32x16xf32, #blocked>
""")
    after = self._snapshot("""\
  %b = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>
  %d = arith.addf %b, %b : tensor<32x16xf32, #blocked>
  %b2 = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked_alt>
  %f = arith.addf %b2, %b2 : tensor<32x16xf32, #blocked_alt>
""")
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["confirmed"], [])

  def test_the_record_states_the_encoding_change(self):
    # The equality ignores which layout each side holds, so the record says what
    # they were: a reader sees that a confirmed recomputation moved a layout.
    before = self._snapshot("""\
  %b = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>
  %d = arith.addf %b, %b : tensor<32x16xf32, #blocked>
""")
    after = self._snapshot("""\
  %b = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>
  %b2 = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked_alt>
  %d = arith.addf %b2, %b2 : tensor<32x16xf32, #blocked_alt>
""")
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 1)
    candidate = report["confirmed"][0]
    self.assertEqual(
      candidate["original_expression"],
      "tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>",
    )
    self.assertEqual(
      candidate["encoding_change"],
      {
        "changed": True,
        "original": [self._BLOCKED, self._BLOCKED],
        "recomputed": [self._BLOCKED, self._BLOCKED_ALT],
      },
    )

  def test_a_same_encoding_copy_that_only_renames_a_use_is_not_confirmed(self):
    # The copy is printed exactly as the original was, and the consumer already
    # read the original: nothing about the computation changed, so the pass only
    # renumbered a use. A layout rematerialization differs in the layout it
    # prints, which is what criterion (b) counts.
    before = self._snapshot("""\
  %b = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>
  %d = arith.addf %b, %b : tensor<32x16xf32, #blocked>
""")
    after = self._snapshot("""\
  %b = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>
  %b2 = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>
  %d = arith.addf %b2, %b2 : tensor<32x16xf32, #blocked>
""")
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["status"], "unattributed_only")

  def test_a_same_encoding_copy_that_replaces_a_conversion_still_confirms(self):
    # The pass removed the conversion and rewired its consumer to the copy, so
    # the consumer reads a recomputation even though the copy keeps the layout
    # the original had. This is the confirmation the matcher already made before
    # the layout reading existed, and it is unchanged.
    before = self._snapshot("""\
  %b = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>
  %c = ttg.convert_layout %b : tensor<32x16xf32, #blocked> -> tensor<32x16xf32, #blocked_alt>
  %d = arith.addf %c, %c : tensor<32x16xf32, #blocked_alt>
  %e = arith.addf %b, %b : tensor<32x16xf32, #blocked>
""")
    after = self._snapshot("""\
  %b = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>
  %b2 = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>
  %d = arith.addf %b2, %b2 : tensor<32x16xf32, #blocked>
  %e = arith.addf %b, %b : tensor<32x16xf32, #blocked>
""")
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 1)
    candidate = report["confirmed"][0]
    self.assertEqual(candidate["original_value"], "%b")
    self.assertEqual(candidate["redirected_consumers"], ["%d"])
    self.assertEqual(candidate["remaining_consumers"], ["%e"])
    self.assertEqual(candidate["encoding_change"]["changed"], False)

  def test_two_alias_names_for_one_layout_are_not_an_encoding_change(self):
    # The copy is printed through another alias, but both aliases define the same
    # layout, so no layout moved: the copy is a duplicate of the original under
    # one layout and the transition adds no recomputation. Were the aliases two
    # different layouts, the same pair of definitions would confirm (see
    # test_copy_computed_in_another_encoding_confirms_rematerialization).
    before = self._snapshot("""\
  %b = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>
  %d = arith.addf %b, %b : tensor<32x16xf32, #blocked>
""")
    after = self._snapshot("""\
  %b = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>
  %b2 = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked_a>
  %d = arith.addf %b2, %b2 : tensor<32x16xf32, #blocked_a>
""")
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["confirmed"], [])


# The review's alias-renaming probe, verbatim. An alias name is a snapshot-local
# binding: the after snapshot reuses `#blocked` and `#blocked_alt` for the other
# layout, and defines a third name for the layout `#blocked` used to denote.
_ALIAS_PROBE_BEFORE = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked_a = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module {
  %b = tt.splat %x : f32 -> tensor<32x16xf32, #blocked>
  %d = arith.addf %b, %b : tensor<32x16xf32, #blocked>
  %e = arith.addf %b, %b : tensor<32x16xf32, #blocked>
}
"""

_ALIAS_PROBE_AFTER = """\
#blocked_alt = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked_alt_a = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module {
  %b = tt.splat %x : f32 -> tensor<32x16xf32, #blocked_alt>
  %b2 = tt.splat %x : f32 -> tensor<32x16xf32, #blocked_alt>
  %d = arith.addf %b2, %b2 : tensor<32x16xf32, #blocked_alt>
  %e = arith.addf %b, %b : tensor<32x16xf32, #blocked_alt>
}
"""


class TensorCompilerAliasScopeTests(unittest.TestCase):
  """An encoding alias is resolved in the snapshot that printed it.

  Every snapshot binds its own alias names, so a comparison that resolves a
  before expression through the after table (or the reverse) reads a definition
  the snapshot never printed. The ``_BLOCKED``/``_BLOCKED_ALT`` texts are the
  two layouts the probes below use.
  """

  _BLOCKED = (
    "#ttg.blocked<{sizePerThread=[1,1],threadsPerWarp=[2,16],"
    "warpsPerCTA=[1,1],order=[1,0]}>"
  )
  _BLOCKED_ALT = (
    "#ttg.blocked<{sizePerThread=[1,4],threadsPerWarp=[8,4],"
    "warpsPerCTA=[1,1],order=[1,0]}>"
  )

  def test_reviewer_alias_renaming_probe_confirms_nothing(self):
    # The pass renamed the aliases in the after snapshot only. Both snapshots
    # compute the splat in the same layout, so nothing was recomputed: the copy
    # is printed through the name `#blocked_alt`, which in the after snapshot
    # denotes the layout the before snapshot called `#blocked`.
    report = tensor_compiler_observation.observe_rematerialization(
      _ALIAS_PROBE_BEFORE, _ALIAS_PROBE_AFTER
    )

    self.assertEqual(report["status"], "unattributed_only")
    self.assertEqual(report["confirmed"], [])
    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["unattributed"][0]["values"], ["%b", "%b2"])

  def test_a_genuine_encoding_change_still_reports_one(self):
    # The copy moved from `#blocked` to `#blocked_alt` while every alias keeps
    # its meaning, so the record states the layout the copy holds: the two
    # layouts are the before snapshot's definitions of those two names.
    before = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module {
  %b = tt.splat %x : f32 -> tensor<32x16xf32, #blocked>
  %d = arith.addf %b, %b : tensor<32x16xf32, #blocked>
  %e = arith.addf %b, %b : tensor<32x16xf32, #blocked>
}
"""
    after = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module {
  %b = tt.splat %x : f32 -> tensor<32x16xf32, #blocked>
  %b2 = tt.splat %x : f32 -> tensor<32x16xf32, #blocked_alt>
  %d = arith.addf %b2, %b2 : tensor<32x16xf32, #blocked_alt>
  %e = arith.addf %b, %b : tensor<32x16xf32, #blocked>
}
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 1)
    candidate = report["confirmed"][0]
    self.assertEqual(candidate["recomputed_values"], ["%b2"])
    self.assertEqual(
      candidate["encoding_change"],
      {
        "changed": True,
        "original": [self._BLOCKED],
        "recomputed": [self._BLOCKED_ALT],
      },
    )

  def test_a_retained_producer_is_not_a_new_recomputation(self):
    # `%b2` exists before the pass in the layout it holds after it, and `%d` is
    # merely rewired to it while `%b3` adds an unrelated definition in the
    # layout `%b` already holds. The extra definition is not a recomputation of
    # `%b`, because nothing in `%b`'s layout was created: the attribution is per
    # definition and layout, never a larger module-wide total.
    before = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module {
  %b = tt.splat %x : f32 -> tensor<32x16xf32, #blocked>
  %b2 = tt.splat %x : f32 -> tensor<32x16xf32, #blocked_alt>
  %d = arith.addf %b, %b : tensor<32x16xf32, #blocked>
  %e = arith.addf %b, %b : tensor<32x16xf32, #blocked>
  %f = arith.mulf %b2, %b2 : tensor<32x16xf32, #blocked_alt>
}
"""
    after = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module {
  %b = tt.splat %x : f32 -> tensor<32x16xf32, #blocked>
  %b2 = tt.splat %x : f32 -> tensor<32x16xf32, #blocked_alt>
  %b3 = tt.splat %x : f32 -> tensor<32x16xf32, #blocked>
  %b4 = tt.splat %x : f32 -> tensor<32x16xf32, #blocked_alt>
  %d = arith.addf %b2, %b2 : tensor<32x16xf32, #blocked_alt>
  %e = arith.addf %b, %b : tensor<32x16xf32, #blocked>
  %f = arith.mulf %b2, %b2 : tensor<32x16xf32, #blocked_alt>
  %g = arith.divf %b3, %b3 : tensor<32x16xf32, #blocked>
}
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed"], [])
    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["status"], "unattributed_only")

  def test_a_genuinely_new_copy_in_another_layout_confirms(self):
    # Only one producer of the computation exists before the pass, in `#blocked`.
    # The pass rewires `%d` to a copy in `#blocked_alt` that no definition held
    # before it, so that copy is new work for the consumer.
    before = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module {
  %b = tt.splat %x : f32 -> tensor<32x16xf32, #blocked>
  %d = arith.addf %b, %b : tensor<32x16xf32, #blocked>
  %e = arith.addf %b, %b : tensor<32x16xf32, #blocked>
}
"""
    after = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module {
  %b = tt.splat %x : f32 -> tensor<32x16xf32, #blocked>
  %b2 = tt.splat %x : f32 -> tensor<32x16xf32, #blocked_alt>
  %d = arith.addf %b2, %b2 : tensor<32x16xf32, #blocked_alt>
  %e = arith.addf %b, %b : tensor<32x16xf32, #blocked>
  %f = arith.mulf %b2, %b2 : tensor<32x16xf32, #blocked_alt>
}
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed_count"], 1)
    candidate = report["confirmed"][0]
    self.assertEqual(candidate["original_value"], "%b")
    self.assertEqual(candidate["recomputed_values"], ["%b2"])
    self.assertEqual(candidate["redirected_consumers"], ["%d"])
    self.assertEqual(candidate["encoding_change"]["changed"], True)

  def test_two_snapshots_with_different_alias_names_throughout_agree(self):
    # The smallest edge case: the after snapshot renamed every alias, and each
    # name still denotes the parameters its counterpart denoted. No encoding
    # moved, so the record says so and the duplicate stays unattributed work.
    before = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module {
  %b = tt.splat %x : f32 -> tensor<32x16xf32, #blocked>
  %d = arith.addf %b, %b : tensor<32x16xf32, #blocked>
  %e = arith.addf %b, %b : tensor<32x16xf32, #blocked>
}
"""
    after = """\
#renamed = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#renamed_alt = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module {
  %b = tt.splat %x : f32 -> tensor<32x16xf32, #renamed>
  %b2 = tt.splat %x : f32 -> tensor<32x16xf32, #renamed>
  %d = arith.addf %b2, %b2 : tensor<32x16xf32, #renamed>
  %e = arith.addf %b, %b : tensor<32x16xf32, #renamed>
}
"""
    report = tensor_compiler_observation.observe_rematerialization(before, after)

    self.assertEqual(report["confirmed"], [])
    self.assertEqual(report["confirmed_count"], 0)
    self.assertEqual(report["status"], "unattributed_only")
    self.assertEqual(report["unattributed"][0]["values"], ["%b", "%b2"])


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
