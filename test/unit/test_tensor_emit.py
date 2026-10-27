"""Triton source emission tests; no GPU is required."""

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test.unit.tensor_support import EMIT_API, SYMIRC, check_source


def emit_cli(source, extra_args=()):
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source_path = root / "case.sir"
    output_path = root / "kernel.py"
    source_path.write_text(source, encoding="utf-8")
    result = subprocess.run(
      [
        str(SYMIRC),
        "--target",
        "triton",
        "--entry",
        "kernel",
        *extra_args,
        "-o",
        str(output_path),
        str(source_path),
      ],
      capture_output=True,
      text=True,
      check=False,
      timeout=10,
    )
    output = output_path.read_text(encoding="utf-8") if output_path.exists() else None
    return result, output


def emit_api(source, sentinel=None):
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source_path = root / "case.sir"
    output_path = root / "kernel.py"
    source_path.write_text(source, encoding="utf-8")
    if sentinel is not None:
      output_path.write_bytes(sentinel)
    result = subprocess.run(
      [str(EMIT_API), "triton", str(source_path), str(output_path)],
      capture_output=True,
      text=True,
      check=False,
      timeout=10,
    )
    output = output_path.read_bytes() if output_path.exists() else None
    return result, output


def source_signature(source):
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source_path = root / "case.sir"
    case_path = root / "case"
    source_path.write_text(source, encoding="utf-8")
    result = subprocess.run(
      [str(EMIT_API), "signature", str(source_path), str(case_path)],
      capture_output=True,
      text=True,
      check=False,
      timeout=10,
    )
    if result.returncode != 0:
      return result, None
    manifest = json.loads((case_path / "case.json").read_text(encoding="utf-8"))
    return result, manifest["coverage"]["source_signature"]


SOFTMAX_SOURCE = """\
fun @kernel(%x: tensor [2, 4] f32) : tensor [2, 4] f32 {
  let mut %out: tensor [2, 4] f32 = undef;
^entry:
  %out = tensor softmax %x, 1;
  ret %out;
}
"""


DECOMPOSED_SOURCE = """\
fun @kernel(%x: tensor [2, 4] f32) : tensor [2, 4] f32 {
  let mut %maximum: tensor [2, 1] f32 = undef;
  let mut %shifted: tensor [2, 4] f32 = undef;
  let mut %exps: tensor [2, 4] f32 = undef;
  let mut %denom: tensor [2, 1] f32 = undef;
  let mut %out: tensor [2, 4] f32 = undef;
^entry:
  %maximum = tensor rmax %x, 1, keep;
  %shifted = tensor sub %x, %maximum;
  %exps = tensor exp %shifted;
  %denom = tensor sum %exps, 1, keep;
  %out = tensor div %exps, %denom;
  ret %out;
}
"""


class TensorEmitTests(unittest.TestCase):
  def setUp(self):
    self.assertTrue(SYMIRC.is_file(), f"symirc executable not found: {SYMIRC}")
    self.assertTrue(
      EMIT_API.is_file(), f"emission API executable not found: {EMIT_API}"
    )

  def test_direct_softmax_call_survives(self):
    result, output = emit_cli(SOFTMAX_SOURCE)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("tl.softmax(", output)
    self.assertIn("dim=1", output)
    self.assertNotIn("keep_dims", output)

  def test_v4_emits_native_triton_remainder(self):
    source = """\
fun @kernel(%x: tensor [2] f32, %y: tensor [2] f32) : tensor [2] f32 {
  let mut %out: tensor [2] f32 = undef;
^entry:
  %out = tensor rem %x, %y;
  ret %out;
}
"""
    result, output = emit_cli(source, ("--tensor-profile-version", "v4"))

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    tree = ast.parse(output)
    modulo_nodes = [
      node
      for node in ast.walk(tree)
      if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod)
    ]
    self.assertEqual(len(modulo_nodes), 1)
    self.assertNotIn("copysign", output)
    self.assertNotIn("trunc", output)

  def test_decomposed_keeps_reduction_axes(self):
    result, output = emit_cli(
      DECOMPOSED_SOURCE, ("--tensor-softmax-lowering", "decomposed")
    )

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("tl.max(", output)
    self.assertIn("tl.sum(", output)
    # Every reduction keeps its axis; a float extrema reduction emits a second
    # reduction for its signed-zero tie-break.
    reductions = (
      output.count("tl.max(") + output.count("tl.min(") + output.count("tl.sum(")
    )
    self.assertEqual(output.count("keep_dims=True"), reductions)
    self.assertNotIn("keep_dims=False", output)

  def test_transposed_output_offsets(self):
    source = """\
fun @kernel(%x: tensor [2, 4] i32) : tensor [4, 2] i32 {
  let mut %out: tensor [4, 2] i32 = undef;
^entry:
  %out = tensor transpose %x, [1, 0];
  ret %out;
}
"""

    result, output = emit_cli(source)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("tl.permute(", output)
    self.assertIn("_sir_out_d0 * 2 + _sir_out_d1", output)
    self.assertIn("tl.store(out_ptr + _sir_out_offsets", output)

  def test_runtime_inputs_are_loaded(self):
    source = """\
fun @kernel(%x: tensor [2, 4] f32, %bias: f32) : tensor [2, 4] f32 {
  let mut %out: tensor [2, 4] f32 = undef;
^entry:
  %out = tensor add %x, %bias;
  ret %out;
}
"""

    result, output = emit_cli(source)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("def kernel(x_ptr, bias_ptr, out_ptr):", output)
    self.assertIn("tl.load(x_ptr +", output)
    self.assertIn("tl.load(bias_ptr)", output)
    self.assertNotIn("tl.full((), 0.0", output)

  def test_output_offset_name_does_not_clobber_source_local(self):
    source = """\
fun @kernel(%out_offsets: tensor [2] i32) : tensor [2] i32 {
^entry:
  ret %out_offsets;
}
"""

    result, output = emit_cli(source)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertRegex(
      output,
      r"_sir_input_load_tmp\d+ = tl\.load\(out_offsets_ptr \+ _sir_out_offsets_offsets\)",
    )
    self.assertRegex(
      output,
      r"tl\.store\(out_ptr \+ _sir_out_offsets_\d+, _sir_input_load_tmp\d+\)",
    )

  def test_parameter_named_out_has_unique_pointer_argument(self):
    source = """\
fun @kernel(%out: tensor [2] i32) : tensor [2] i32 {
^entry:
  ret %out;
}
"""

    result, output = emit_cli(source)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    module = ast.parse(output)
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef))
    names = [argument.arg for argument in function.args.args]
    self.assertEqual(len(names), len(set(names)))

  def test_cpp_source_signature_preserves_operator_and_ignores_whitespace(self):
    left_minus_right = """\
fun @kernel(%a: i32, %b: i32) : i32 {
^entry:
  ret %a- %b;
}
"""
    right_minus_right = """\
fun @kernel(%a: i32, %b: i32): i32 { // comment
^entry: ret %b- %b;
}
"""
    spaced_left_minus_right = (
      "fun @kernel ( %a : i32 , %b : i32 ) : i32 { ^entry : ret %a - %b ; }"
    )

    signatures = [
      source_signature(program)
      for program in (left_minus_right, right_minus_right, spaced_left_minus_right)
    ]

    for result, signature in signatures:
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      self.assertIsNotNone(signature)
    self.assertNotEqual(signatures[0][1], signatures[1][1])
    self.assertEqual(signatures[0][1], signatures[2][1])

  def test_scalar_integer_division_uses_integer_operator(self):
    source = """\
fun @kernel(%b: i32, %x: tensor [2] i32) : tensor [2] i32 {
  let mut %quotient: i32 = undef;
  let mut %out: tensor [2] i32 = undef;
^entry:
  %quotient = 7 / %b;
  %out = tensor add %x, %quotient;
  ret %out;
}
"""

    result, output = emit_cli(source)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertRegex(output, r"_sir_quotient = \(7 // _sir_input_load_tmp\d+\)")
    self.assertNotIn("tl.full((), 7, tl.int32)", output)

  def test_negative_integer_division_constant_is_emitted_directly(self):
    source = """\
fun @kernel(%b: i32, %x: tensor [2] i32) : tensor [2] i32 {
  let mut %quotient: i32 = undef;
  let mut %out: tensor [2] i32 = undef;
^entry:
  %quotient = -7 / %b;
  %out = tensor add %x, %quotient;
  ret %out;
}
"""

    result, output = emit_cli(source)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertRegex(output, r"_sir_quotient = \(-7 // _sir_input_load_tmp\d+\)")
    self.assertNotIn("tl.full((), -7, tl.int32)", output)

  def test_logical_shift_uses_unsigned_right_shift(self):
    source = """\
fun @kernel(%shift: i32, %x: tensor [2] i32) : tensor [2] i32 {
  let mut %arithmetic: i32 = undef;
  let mut %shifted: i32 = undef;
  let mut %out: tensor [2] i32 = undef;
^entry:
  %arithmetic = -1 >> %shift;
  %shifted = -1 >>> %shift;
  %out = tensor add %x, %shifted;
  ret %out;
}
"""

    result, output = emit_cli(source)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertRegex(output, r"_sir_arithmetic = \(-1 >> _sir_input_load_tmp\d+\)")
    self.assertRegex(
      output,
      r"tl\.cast\(\(tl\.cast\(-1, tl\.uint32\) >> _sir_input_load_tmp\d+\), tl\.int32\)",
    )

  def test_profile_rejection_precedes_write(self):
    source = """\
fun @kernel(%x: tensor [2, 2] i32) : tensor [2, 2] i32 {
^entry:
  br ^next;
^next:
  ret %x;
}
"""
    sentinel = b"existing output must survive\n"
    cli_result, cli_output = emit_cli(source)
    api_result, api_output = emit_api(source, sentinel)

    self.assertNotEqual(cli_result.returncode, 0)
    self.assertIn("tensor-profile", cli_result.stderr)
    self.assertIsNone(cli_output)
    self.assertNotEqual(api_result.returncode, 0)
    self.assertIn("tensor-profile", api_result.stderr)
    self.assertEqual(api_output, sentinel)

  def test_scalar_reduction_stores_one_element(self):
    source = """\
fun @kernel(%x: tensor [2] i32) : i32 {
  let mut %out: i32 = undef;
^entry:
  %out = tensor sum %x, 0, drop;
  ret %out;
}
"""

    result, output = emit_cli(source)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("tl.store(out_ptr,", output)
    self.assertNotIn("_sir_out_offsets", output)

  def test_v3_transpose_recomputes_validity_in_output_coordinates(self):
    source = """\
fun @kernel(%x: tensor [17, 16] i32) : tensor [16, 17] i32 {
  let mut %out: tensor [16, 17] i32 = undef;
^entry:
  %out = tensor transpose %x, [1, 0];
  ret %out;
}
"""

    result, output = emit_cli(source, ("--tensor-profile-version", "v3"))

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("tl.permute(", output)
    self.assertRegex(output, r"_sir_x_valid = \(_sir_x_d0 < 17\) & \(_sir_x_d1 < 16\)")
    self.assertRegex(
      output, r"_sir_out_valid = \(_sir_out_d0 < 16\) & \(_sir_out_d1 < 17\)"
    )
    self.assertIn("mask=_sir_out_valid", output)

  def test_v3_sum_after_transpose_uses_transposed_validity(self):
    source = """\
fun @kernel(%x: tensor [17, 16] i32) : tensor [17] i32 {
  let mut %transposed: tensor [16, 17] i32 = undef;
  let mut %out: tensor [17] i32 = undef;
^entry:
  %transposed = tensor transpose %x, [1, 0];
  %out = tensor sum %transposed, 0, drop;
  ret %out;
}
"""

    result, output = emit_cli(source, ("--tensor-profile-version", "v3"))

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertRegex(
      output,
      r"tl\.where\(_sir_transposed_result_valid, _sir_transposed, 0\)",
    )
    self.assertIn("tl.sum(", output)
    self.assertIn("mask=_sir_out_valid", output)

  def test_v3_expand_keeps_tail_mask_on_inserted_axis(self):
    source = """\
fun @kernel(%x: tensor [17] i32) : tensor [17, 1] i32 {
  let mut %out: tensor [17, 1] i32 = undef;
^entry:
  %out = tensor expand %x, 1;
  ret %out;
}
"""

    result, output = emit_cli(source, ("--tensor-profile-version", "v3"))

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("tl.expand_dims(", output)
    self.assertRegex(
      output, r"_sir_out_valid = \(_sir_out_d0 < 17\) & \(_sir_out_d1 < 1\)"
    )
    self.assertIn("mask=_sir_out_valid", output)


  def test_rank_increasing_broadcast_aligns_the_operand_rank(self):
    source = """\
fun @kernel(%x: tensor [2] f32) : tensor [2, 2] f32 {
  let mut %out: tensor [2, 2] f32 = undef;
^entry:
  %out = tensor broadcast %x, [2, 2];
  ret %out;
}
"""

    result, output = emit_cli(source)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("tl.reshape(", output)
    self.assertIn("(1, 2)", output)
    self.assertLess(output.index("tl.reshape("), output.index("tl.broadcast_to("))

  def test_rank_zero_operand_broadcast_splats_a_full_block(self):
    source = """\
fun @kernel(%x: tensor [2] f32) : tensor [2, 2] f32 {
  let mut %total: f32 = undef;
  let mut %out: tensor [2, 2] f32 = undef;
^entry:
  %total = tensor sum %x, 0, drop;
  %out = tensor broadcast %total, [2, 2];
  ret %out;
}
"""

    result, output = emit_cli(source)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("tl.full((1, 1)", output)
    self.assertLess(output.index("tl.full("), output.index("tl.broadcast_to("))

  def test_equal_rank_broadcast_still_calls_broadcast_directly(self):
    source = """\
fun @kernel(%x: tensor [2, 1] f32) : tensor [2, 2] f32 {
  let mut %out: tensor [2, 2] f32 = undef;
^entry:
  %out = tensor broadcast %x, [2, 2];
  ret %out;
}
"""

    result, output = emit_cli(source)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertNotIn("tl.reshape(", output)
    self.assertIn("tl.broadcast_to(", output)

  def test_v3_rank_increasing_broadcast_keeps_target_validity(self):
    source = """\
fun @kernel(%x: tensor [16] i32) : tensor [17, 16] i32 {
  let mut %out: tensor [17, 16] i32 = undef;
^entry:
  %out = tensor broadcast %x, [17, 16];
  ret %out;
}
"""

    result, output = emit_cli(source, ("--tensor-profile-version", "v3"))

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("(1, 16)", output)
    self.assertLess(output.index("tl.reshape("), output.index("tl.broadcast_to("))
    self.assertRegex(
      output, r"_sir_out_valid = \(_sir_out_d0 < 17\) & \(_sir_out_d1 < 16\)"
    )

  def test_v3_broadcast_uses_target_shape_validity(self):
    source = """\
fun @kernel(%x: tensor [1, 16] i32) : tensor [17, 16] i32 {
  let mut %out: tensor [17, 16] i32 = undef;
^entry:
  %out = tensor broadcast %x, [17, 16];
  ret %out;
}
"""

    result, output = emit_cli(source, ("--tensor-profile-version", "v3"))

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("tl.broadcast_to(", output)
    self.assertIn("(32, 16)", output)
    self.assertRegex(
      output, r"_sir_out_valid = \(_sir_out_d0 < 17\) & \(_sir_out_d1 < 16\)"
    )
    self.assertIn("mask=_sir_out_valid", output)

  def test_v3_expand_then_broadcast_preserves_validity_between_stages(self):
    source = """\
fun @kernel(%x: tensor [17] i32) : tensor [17, 16] i32 {
  let mut %expanded: tensor [17, 1] i32 = undef;
  let mut %out: tensor [17, 16] i32 = undef;
^entry:
  %expanded = tensor expand %x, 1;
  %out = tensor broadcast %expanded, [17, 16];
  ret %out;
}
"""

    result, output = emit_cli(source, ("--tensor-profile-version", "v3"))

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("tl.expand_dims(", output)
    self.assertIn("tl.broadcast_to(", output)
    self.assertRegex(
      output,
      r"_sir_expanded_result_valid = \(_sir_expanded_result_d0 < 17\) & \(_sir_expanded_result_d1 < 1\)",
    )
    self.assertIn("mask=_sir_out_valid", output)

  def test_v3_loop_carried_tensor_keeps_validity_and_masks_each_update(self):
    source = """\
fun @kernel(%x: tensor [17] i32) : tensor [17] i32 {
  let mut %i: i32 = undef;
  let mut %acc: tensor [17] i32 = undef;
^entry:
  %acc = tensor mul %x, 0;
  %i = 0;
  br ^header;
^header:
  br %i < 3, ^body, ^exit;
^body:
  %acc = tensor add %acc, %x;
  %i = %i + 1;
  br ^header;
^exit:
  ret %acc;
}
"""

    result, output = emit_cli(source, ("--tensor-profile-version", "v3"))

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("for _sir_i in tl.range(0, 3):", output)
    self.assertRegex(output, r"tl\.where\(_sir_acc_result_valid, .*?, 0\)")
    self.assertRegex(output, r"tl\.load\(x_ptr \+ .*mask=_sir_x_valid, other=0\)")
    self.assertIn("mask=_sir_out_valid", output)

  def test_v3_loop_reduction_uses_masked_logical_values_each_iteration(self):
    source = """\
fun @kernel(%x: tensor [17] f32) : f32 {
  let mut %i: i32 = undef;
  let mut %row_sum: f32 = undef;
  let mut %total: f32 = undef;
^entry:
  %total = 0.0;
  %i = 0;
  br ^header;
^header:
  br %i < 2, ^body, ^exit;
^body:
  %row_sum = tensor sum %x, 0, drop;
  %total = %total + %row_sum;
  %i = %i + 1;
  br ^header;
^exit:
  ret %total;
}
"""

    result, output = emit_cli(source, ("--tensor-profile-version", "v3"))

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("for _sir_i in tl.range(0, 2):", output)
    self.assertRegex(
      output,
      r"tl\.sum\(tl\.where\(_sir_x_valid, .*?, 0\.0\), axis=0, keep_dims=False\)",
    )
    self.assertEqual(output.count("tl.load(x_ptr +"), 1)

  def test_v3_nested_loop_resets_inner_counter_inside_outer_loop(self):
    source = """\
fun @kernel(%x: tensor [17] i32) : tensor [17] i32 {
  let mut %outer: i32 = undef;
  let mut %inner: i32 = undef;
  let mut %acc: tensor [17] i32 = undef;
^entry:
  %acc = tensor mul %x, 0;
  %outer = 0;
  br ^outer_header;
^outer_header:
  br %outer < 2, ^outer_body, ^exit;
^outer_body:
  %inner = 0;
  br ^inner_header;
^inner_header:
  br %inner < 3, ^inner_body, ^outer_latch;
^inner_body:
  %acc = tensor add %acc, %x;
  %inner = %inner + 1;
  br ^inner_header;
^outer_latch:
  %outer = %outer + 1;
  br ^outer_header;
^exit:
  ret %acc;
}
"""

    result, output = emit_cli(source, ("--tensor-profile-version", "v3"))

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("for _sir_outer in tl.range(0, 2):", output)
    self.assertIn("for _sir_inner in tl.range(0, 3):", output)
    self.assertRegex(
      output, r"(?m)^        _sir_inner = tl\.full\(\(\), 0, tl\.int32\)"
    )
    self.assertIn("_sir_outer = tl.full((), 2, tl.int32)", output)
    self.assertIn("_sir_inner = tl.full((), 3, tl.int32)", output)

  def test_v3_loop_restores_induction_variable_to_bound_after_range(self):
    source = """\
fun @kernel(%x: tensor [17] i32) : tensor [17] i32 {
  let mut %i: i32 = undef;
  let mut %out: tensor [17] i32 = undef;
^entry:
  %i = 0;
  br ^header;
^header:
  br %i < 4, ^body, ^exit;
^body:
  %i = %i + 1;
  br ^header;
^exit:
  %out = tensor mul %x, %i;
  ret %out;
}
"""

    result, output = emit_cli(source, ("--tensor-profile-version", "v3"))

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertRegex(
      output,
      r"(?s)for _sir_i in tl\.range\(0, 4\):.*?_sir_i = tl\.full\(\(\), 4, tl\.int32\).*?_sir_out =",
    )
    bound_restore = output.index("_sir_i = tl.full((), 4, tl.int32)")
    output_expression = output.index("_sir_out = tl.where(_sir_out_result_valid")
    self.assertLess(bound_restore, output_expression)
    self.assertRegex(
      output, r"_sir_out = tl\.where\(_sir_out_result_valid, .*_sir_i.*\)"
    )

  def test_v3_loop_preserves_transposed_accumulator_mask_mapping(self):
    source = """\
fun @kernel(%x: tensor [17, 16] i32) : tensor [16, 17] i32 {
  let mut %i: i32 = undef;
  let mut %transposed: tensor [16, 17] i32 = undef;
  let mut %acc: tensor [16, 17] i32 = undef;
^entry:
  %transposed = tensor transpose %x, [1, 0];
  %acc = tensor mul %transposed, 0;
  %i = 0;
  br ^header;
^header:
  br %i < 2, ^body, ^exit;
^body:
  %transposed = tensor transpose %x, [1, 0];
  %acc = tensor add %acc, %transposed;
  %i = %i + 1;
  br ^header;
^exit:
  ret %acc;
}
"""

    result, output = emit_cli(source, ("--tensor-profile-version", "v3"))

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertGreaterEqual(output.count("tl.permute("), 2)
    self.assertRegex(output, r"_sir_transposed_result_valid = .*< 16.*< 17")
    self.assertRegex(output, r"_sir_acc_result_valid = .*< 16.*< 17")
    self.assertIn("mask=_sir_out_valid", output)

  def test_v1_scalar_parameter_splats_before_reduction(self):
    """A scalar initializer carries the declared tensor rank into the sum."""
    source = """\
fun @kernel(%s: i32) : i32 {
  let %x: tensor [4] i32 = %s;
  let mut %out: i32 = undef;
^entry:
  %out = tensor sum %x, 0, drop;
  ret %out;
}
"""

    result, output = emit_cli(source)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("_sir_x = tl.full((4,), _sir_input_load_tmp0, tl.int32)", output)
    reduced = self.reduction_operands(output)
    self.assertEqual(len(reduced), 1)
    self.assertIn("_sir_x", reduced[0])

  def test_v2_scalar_local_splats_before_transpose(self):
    """A scalar local splat preserves rank through a shape-sensitive op."""
    source = """\
fun @kernel(%s: i32) : tensor [4, 2] i32 {
  let %t: i32 = %s;
  let %x: tensor [2, 4] i32 = %t;
  let mut %out: tensor [4, 2] i32 = undef;
^entry:
  %out = tensor transpose %x, [1, 0];
  ret %out;
}
"""

    result, output = emit_cli(source, ("--tensor-profile-version", "v2"))

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("_sir_t = _sir_input_load_tmp0", output)
    self.assertIn("_sir_x = tl.full((2, 4), _sir_t, tl.int32)", output)
    self.assertIn("tl.permute(_sir_x", output)

  def test_v3_tensor_parameter_copy_is_not_a_scalar_fill(self):
    source = """\
fun @kernel(%x: tensor [4] i32) : tensor [4] i32 {
  let %y: tensor [4] i32 = %x;
^entry:
  ret %y;
}
"""

    result, output = emit_cli(source, ("--tensor-profile-version", "v3"))

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertRegex(output, r"_sir_y = _sir_input_load_tmp\d+")
    self.assertEqual(self.full_fill_values(output), [])

  def test_v3_tensor_local_copy_is_not_a_scalar_fill(self):
    source = """\
fun @kernel(%x: tensor [2, 2] i32) : tensor [2, 2] i32 {
  let %t: tensor [2, 2] i32 = %x;
  let %y: tensor [2, 2] i32 = %t;
^entry:
  ret %y;
}
"""

    result, output = emit_cli(source, ("--tensor-profile-version", "v3"))

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertRegex(output, r"_sir_t = _sir_input_load_tmp\d+")
    self.assertIn("_sir_y = _sir_t", output)
    self.assertEqual(self.full_fill_values(output), [])

  def test_v3_non_power_of_two_tensor_copy_feeds_sum(self):
    source = """\
fun @kernel(%x: tensor [3] i32) : i32 {
  let %y: tensor [3] i32 = %x;
  let mut %out: i32 = undef;
^entry:
  %out = tensor sum %y, 0, drop;
  ret %out;
}
"""

    result, output = emit_cli(source, ("--tensor-profile-version", "v3"))

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertRegex(output, r"_sir_y = _sir_input_load_tmp\d+")
    self.assertNotIn("tl.full((4,), _sir_x", output)
    self.assertEqual(self.full_fill_values(output), [])
    reduced = self.reduction_operands(output)
    self.assertEqual(len(reduced), 1)
    self.assertIn("_sir_y", reduced[0])

  def test_singleton_dimension_splat_keeps_declared_shape(self):
    """The smallest edge case: a singleton axis is still a tensor axis."""
    cases = (
      (
        """\
fun @kernel(%s: i32) : tensor [1, 4] i32 {
  let %x: tensor [1, 4] i32 = %s;
^entry:
  ret %x;
}
""",
        "v1",
        "tl.full((1, 4), _sir_input_load_tmp0, tl.int32)",
      ),
      (
        """\
fun @kernel(%s: f32) : tensor [2, 4] f32 {
  let %x: tensor [2, 4] f32 = %s;
^entry:
  ret %x;
}
""",
        "v2",
        "tl.full((2, 4), _sir_input_load_tmp0, tl.float32)",
      ),
    )
    for source, profile, expected in cases:
      with self.subTest(profile=profile, expected=expected):
        result, output = emit_cli(source, ("--tensor-profile-version", profile))

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        ast.parse(output)
        self.assertIn(expected, output)

  @staticmethod
  def full_fill_values(output):
    """Return each ``tl.full`` fill-value expression as source text."""
    return [
      ast.unparse(node.args[1])
      for node in ast.walk(ast.parse(output))
      if isinstance(node, ast.Call)
      and isinstance(node.func, ast.Attribute)
      and node.func.attr == "full"
      and len(node.args) >= 2
    ]

  @staticmethod
  def reduction_operands(output):
    """Return the reduced value expression of each ``tl.sum``/``min``/``max``."""
    return [
      ast.unparse(node.args[0])
      for node in ast.walk(ast.parse(output))
      if isinstance(node, ast.Call)
      and isinstance(node.func, ast.Attribute)
      and node.func.attr in ("sum", "min", "max")
      and node.args
    ]


def _triton_interpreter_available():
  try:
    import torch  # noqa: F401
    import triton  # noqa: F401
  except Exception:
    return False
  return True


@unittest.skipUnless(_triton_interpreter_available(), "Triton interpreter unavailable")
class TensorBroadcastExecutionTests(unittest.TestCase):
  """The aligned broadcast computes the right-aligned values the SIR states."""

  def run_kernel(self, source, arguments):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source_path = root / "case.sir"
      kernel_path = root / "kernel.py"
      source_path.write_text(source, encoding="utf-8")
      emitted = subprocess.run(
        [
          str(SYMIRC),
          "--target",
          "triton",
          "--entry",
          "kernel",
          "-o",
          str(kernel_path),
          str(source_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
      )
      self.assertEqual(emitted.returncode, 0, emitted.stdout + emitted.stderr)
      runner = root / "runner.py"
      runner.write_text(
        """\
import importlib.util
import sys

import torch

spec = importlib.util.spec_from_file_location("kernel", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
args = [torch.tensor(value, dtype=torch.float32) for value in eval(sys.argv[2])]
out = torch.full((2, 2), float("nan"), dtype=torch.float32)
module.kernel[(1, 1, 1)](*args, out)
print(out.tolist())
""",
        encoding="utf-8",
      )
      environment = dict(**__import__("os").environ)
      environment["TRITON_INTERPRET"] = "1"
      completed = subprocess.run(
        [__import__("sys").executable, str(runner), str(kernel_path), repr(arguments)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env=environment,
      )
      self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
      return completed.stdout.strip()

  def test_rank_one_operand_broadcasts_across_the_new_leading_axis(self):
    source = """\
fun @kernel(%x: tensor [2] f32) : tensor [2, 2] f32 {
  let mut %out: tensor [2, 2] f32 = undef;
^entry:
  %out = tensor broadcast %x, [2, 2];
  ret %out;
}
"""

    self.assertEqual(self.run_kernel(source, ([1.0, 2.0],)), "[[1.0, 2.0], [1.0, 2.0]]")

  def test_rank_zero_operand_broadcasts_its_sum_to_every_element(self):
    source = """\
fun @kernel(%x: tensor [2] f32) : tensor [2, 2] f32 {
  let mut %total: f32 = undef;
  let mut %out: tensor [2, 2] f32 = undef;
^entry:
  %total = tensor sum %x, 0, drop;
  %out = tensor broadcast %total, [2, 2];
  ret %out;
}
"""

    self.assertEqual(self.run_kernel(source, ([1.5, 2.5],)), "[[4.0, 4.0], [4.0, 4.0]]")


def emit_with_layout(source, buffers, profile):
  """Emit one case with an explicit per-buffer layout manifest."""
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source_path = root / "case.sir"
    output_path = root / "kernel.py"
    layout_path = root / "tensor-layout.json"
    source_path.write_text(source, encoding="utf-8")
    layout_path.write_text(json.dumps({"buffers": buffers}), encoding="utf-8")
    result = subprocess.run(
      [
        str(SYMIRC),
        "--target",
        "triton",
        "--entry",
        "kernel",
        "--tensor-profile-version",
        profile,
        "--tensor-layout-manifest",
        str(layout_path),
        "-o",
        str(output_path),
        str(source_path),
      ],
      capture_output=True,
      text=True,
      check=False,
      timeout=10,
    )
    output = output_path.read_text(encoding="utf-8") if output_path.exists() else None
    return result, output


def source_layout(name, shape, strides, offset, span, order):
  return {
    "name": name,
    "role": "input",
    "element_strides": strides,
    "element_offset": offset,
    "storage_span": span,
    "axis_order": order,
  }


WINDOW_V2_SOURCE = """\
fun @kernel(%src: tensor [8] f32, %i: i32) : tensor [4] f32 {
  let mut %w: tensor [4] f32 = undef;
^entry:
  %w = tensor window %src, %i, 0.0, [4, 1, 8];
  ret %w;
}
"""

WINDOW_LITERAL_SOURCE = """\
fun @kernel(%src: tensor [8] f32) : tensor [4] f32 {
  let mut %k: i32 = 3;
  let mut %w: tensor [4] f32 = undef;
^entry:
  %w = tensor window %src, %k, 0.5, [4, 1, 8];
  ret %w;
}
"""

WINDOW_V3_REDUCTION_SOURCE = """\
fun @kernel(%src: tensor [8] f32, %i: i32) : f32 {
  let mut %w: tensor [3] f32 = undef;
  let mut %total: f32 = undef;
^entry:
  %w = tensor window %src, %i, 0.0, [3, 1, 8];
  %total = tensor sum %w, 0, drop;
  ret %total;
}
"""


@unittest.skipUnless(SYMIRC.is_file(), "symirc executable not found")
class WindowEmissionTests(unittest.TestCase):
  def test_window_emits_a_masked_load_with_fill(self):
    result, output = emit_cli(WINDOW_V2_SOURCE, ("--tensor-profile-version", "v2"))
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("tl.load(", output)
    self.assertRegex(output, r"tl\.load\([^\n]*mask=[^\n]*other=0\.0?\)")

  def test_window_normalizes_a_literal_index(self):
    result, output = emit_cli(WINDOW_LITERAL_SOURCE, ("--tensor-profile-version", "v2"))
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("+ tl.full((), 0, tl.int32)", output)
    self.assertIn(".to(tl.int64)", output)

  def test_window_address_widens_before_multiplying(self):
    result, output = emit_cli(WINDOW_LITERAL_SOURCE, ("--tensor-profile-version", "v2"))
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertRegex(
      output,
      r"_sir_window_index64_tmp\d+ = _sir_window_index_tmp\d+\.to\(tl\.int64\)",
    )
    self.assertRegex(
      output,
      r"_sir_window_lin_tmp\d+ = _sir_window_index64_tmp\d+ \* 1 "
      r"\+ _sir_window_row_tmp\d+\.to\(tl\.int64\)",
    )

  def test_window_mask_conjoins_shape_validity_and_active(self):
    result, output = emit_cli(WINDOW_V3_REDUCTION_SOURCE, ("--tensor-profile-version", "v3"))
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertRegex(
      output,
      r"_sir_window_mask_tmp\d+ = \(_sir_window_row_tmp\d+ < 3\) "
      r"& \(_sir_window_lin_tmp\d+ < 8\)",
    )

  def test_window_v3_reduction_counts_only_logical_rows(self):
    result, output = emit_cli(WINDOW_V3_REDUCTION_SOURCE, ("--tensor-profile-version", "v3"))
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("tl.sum(", output)

  def test_window_result_carries_the_shape_validity(self):
    result, output = emit_cli(WINDOW_V3_REDUCTION_SOURCE, ("--tensor-profile-version", "v3"))
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("tl.where(", output)


WINDOW_STRIDED_SOURCE = """\
fun @kernel(%src: tensor [8] f32) : tensor [4] f32 {
  let mut %w: tensor [4] f32 = undef;
^entry:
  %w = tensor window %src, 0, 0.0, [4, 1, 8];
  ret %w;
}
"""

WINDOW_STRIDED_V3_SOURCE = """\
fun @kernel(%src: tensor [5] f32) : tensor [3] f32 {
  let mut %w: tensor [3] f32 = undef;
^entry:
  %w = tensor window %src, 0, 0.0, [3, 1, 5];
  ret %w;
}
"""

WINDOW_STRIDED_RANK_TWO_SOURCE = """\
fun @kernel(%src: tensor [4, 3] f32) : tensor [2, 3] f32 {
  let mut %w: tensor [2, 3] f32 = undef;
^entry:
  %w = tensor window %src, 0, 0.0, [2, 4, 4];
  ret %w;
}
"""


@unittest.skipUnless(SYMIRC.is_file(), "symirc executable not found")
class WindowStridedSourceTests(unittest.TestCase):
  def test_strided_source_stride_in_address(self):
    result, output = emit_with_layout(
      WINDOW_STRIDED_SOURCE,
      [source_layout("src", [8], [2], 0, 15, [0])],
      "v2",
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertRegex(output, r"src_ptr \+ _sir_window_lin_tmp\d+ \* 2")

  def test_strided_source_element_offset(self):
    result, output = emit_with_layout(
      WINDOW_STRIDED_SOURCE,
      [source_layout("src", [8], [2], 3, 18, [0])],
      "v2",
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertRegex(output, r"src_ptr \+ 3 \+ _sir_window_lin_tmp\d+ \* 2")

  def test_strided_source_logical_extent_in_mask(self):
    result, output = emit_with_layout(
      WINDOW_STRIDED_V3_SOURCE,
      [source_layout("src", [5], [3], 0, 13, [0])],
      "v3",
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("_sir_src_d0 < 5", output)
    self.assertRegex(
      output,
      r"_sir_window_mask_tmp\d+ = \(_sir_window_row_tmp\d+ < 3\) "
      r"& \(_sir_window_lin_tmp\d+ < 5\)",
    )

  def test_strided_source_widened_arithmetic(self):
    result, output = emit_with_layout(
      WINDOW_STRIDED_SOURCE,
      [source_layout("src", [8], [3], 0, 22, [0])],
      "v2",
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertRegex(
      output,
      r"_sir_window_lin_tmp\d+ = _sir_window_index64_tmp\d+ \* 1 "
      r"\+ _sir_window_row_tmp\d+\.to\(tl\.int64\)",
    )
    self.assertRegex(output, r"src_ptr \+ _sir_window_lin_tmp\d+ \* 3")

  def test_axis_order_mismatch_refused(self):
    result, output = emit_with_layout(
      WINDOW_STRIDED_RANK_TWO_SOURCE,
      [source_layout("src", [4, 3], [3, 1], 0, 12, [0, 1])],
      "v3",
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("axis order", result.stderr)

  def test_rank_two_strided_source_trailing_axis(self):
    result, output = emit_with_layout(
      WINDOW_STRIDED_RANK_TWO_SOURCE,
      [source_layout("src", [4, 3], [5, 1], 2, 20, [1, 0])],
      "v3",
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertRegex(
      output,
      r"src_ptr \+ 2 \+ _sir_window_lin_tmp\d+ \* 5 "
      r"\+ _sir_window_d1_tmp\d+ \* 1",
    )
    self.assertRegex(output, r"_sir_window_d1_tmp\d+ < 3")


WINDOW_PLAIN_SOURCE = """\
fun @kernel(%src: tensor [8] f32) : tensor [4] f32 {
  let mut %k: i32 = 1;
  let mut %w: tensor [4] f32 = undef;
^entry:
  %w = tensor window %src, %k, 0.0, [4, 1, 8];
  ret %w;
}
"""

WINDOW_LOOP_SOURCE = """\
fun @kernel(%src: tensor [8] f32) : tensor [4] f32 {
  let mut %i: i32 = undef;
  let mut %w: tensor [4] f32 = undef;
^entry:
  %w = tensor window %src, 0, -1.0, [4, 1, 8];
  %i = 2147483640;
  br ^header;
^header:
  br %i < 2147483641, ^body, ^exit;
^body:
  %w = tensor window %src, %i, -1.0, [4, 4, 8];
  %i = %i + 1;
  br ^header;
^exit:
  ret %w;
}
"""

WINDOW_RANK_TWO_SOURCE = """\
fun @kernel(%src: tensor [4, 3] f32) : tensor [2] f32 {
  let mut %w: tensor [2, 3] f32 = undef;
  let mut %total: tensor [2] f32 = undef;
^entry:
  %w = tensor window %src, 0, 0.0, [2, 4, 4];
  %total = tensor sum %w, 1, drop;
  ret %total;
}
"""


def _window_gpu_available():
  try:
    import torch
  except Exception:
    return False
  return torch.cuda.is_available()


def _window_triton_available():
  try:
    import torch  # noqa: F401
    import triton  # noqa: F401
  except Exception:
    return False
  return True


@unittest.skipUnless(
  SYMIRC.is_file() and _window_triton_available(), "symirc or Triton unavailable"
)
class WindowExecutionTests(unittest.TestCase):
  """The emitted window load runs on both the interpreter and the GPU."""

  def run_kernel(self, source, values, shape, output_length, profile, mode):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source_path = root / "case.sir"
      kernel_path = root / "kernel.py"
      source_path.write_text(source, encoding="utf-8")
      emitted = subprocess.run(
        [
          str(SYMIRC),
          "--target",
          "triton",
          "--entry",
          "kernel",
          "--tensor-profile-version",
          profile,
          "-o",
          str(kernel_path),
          str(source_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
      )
      self.assertEqual(emitted.returncode, 0, emitted.stdout + emitted.stderr)
      runner = root / "runner.py"
      runner.write_text(
        """\
import importlib.util
import sys

import torch

spec = importlib.util.spec_from_file_location("kernel", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
values = [float(value) for value in sys.argv[2].split(",")]
shape = [int(value) for value in sys.argv[3].split(",")]
output_length = int(sys.argv[4])
mode = sys.argv[5]
device = "cuda" if mode == "compiled" else "cpu"
dtype = torch.float32
x = torch.tensor(values, dtype=dtype, device=device).reshape(*shape)
out = torch.zeros(output_length, dtype=dtype, device=device)
module.kernel[(1, 1, 1)](x, out, num_warps=1, num_stages=1, enable_fp_fusion=False)
if mode == "compiled":
    torch.cuda.synchronize()
print(out.tolist())
""",
        encoding="utf-8",
      )
      environment = dict(os.environ)
      if mode == "interpret":
        environment["TRITON_INTERPRET"] = "1"
      else:
        environment.pop("TRITON_INTERPRET", None)
      completed = subprocess.run(
        [
          sys.executable,
          str(runner),
          str(kernel_path),
          ",".join(str(value) for value in values),
          ",".join(str(extent) for extent in shape),
          str(output_length),
          mode,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
        env=environment,
      )
      self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
      return completed.stdout.strip()

  def both_arms(self, source, values, shape, output_length, profile, expected):
    self.assertEqual(
      self.run_kernel(source, values, shape, output_length, profile, "interpret"),
      expected,
    )
    with self.subTest(arm="gpu"):
      if not _window_gpu_available():
        self.skipTest("gpu_arm: skipped (no CUDA device)")
      self.assertEqual(
        self.run_kernel(source, values, shape, output_length, profile, "compiled"),
        expected,
      )

  def test_window_literal_index_executes_in_both_modes(self):
    self.both_arms(
      WINDOW_PLAIN_SOURCE,
      [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
      [8],
      4,
      "v2",
      "[20.0, 30.0, 40.0, 50.0]",
    )

  def test_window_large_loop_index_emits_only_fill(self):
    self.both_arms(
      WINDOW_LOOP_SOURCE,
      [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
      [8],
      4,
      "v2",
      "[-1.0, -1.0, -1.0, -1.0]",
    )

  def test_window_rank_two_trailing_axis_is_masked(self):
    self.both_arms(
      WINDOW_RANK_TWO_SOURCE,
      [1.0] * 12,
      [4, 3],
      2,
      "v3",
      "[3.0, 3.0]",
    )


V3_RMIN_SOURCE = """\
fun @kernel(%x: tensor [3] f32) : tensor [1] f32 {
  let mut %out: tensor [1] f32 = undef;
^entry:
  %out = tensor rmin %x, 0, keep;
  ret %out;
}
"""

V3_RMAX_SOURCE = """\
fun @kernel(%x: tensor [3] f32) : tensor [1] f32 {
  let mut %out: tensor [1] f32 = undef;
^entry:
  %out = tensor rmax %x, 0, keep;
  ret %out;
}
"""

V3_RMIN_I32_SOURCE = """\
fun @kernel(%x: tensor [3] i32) : tensor [1] i32 {
  let mut %out: tensor [1] i32 = undef;
^entry:
  %out = tensor rmin %x, 0, keep;
  ret %out;
}
"""

V3_RMIN_SINGLE_SOURCE = """\
fun @kernel(%x: tensor [1] f32) : tensor [1] f32 {
  let mut %out: tensor [1] f32 = undef;
^entry:
  %out = tensor rmin %x, 0, keep;
  ret %out;
}
"""

V2_RMIN_SOURCE = """\
fun @kernel(%x: tensor [4] f32) : tensor [1] f32 {
  let mut %out: tensor [1] f32 = undef;
^entry:
  %out = tensor rmin %x, 0, keep;
  ret %out;
}
"""


@unittest.skipUnless(SYMIRC.is_file(), "symirc executable not found")
class V3ExtremaEmissionTests(unittest.TestCase):
  """A v3 extrema reduction excludes inactive lanes with a neutral value."""

  def test_v3_admits_an_extrema_reduction(self):
    result = check_source(V3_RMIN_SOURCE, tensor_profile="v3")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_v3_rmin_masks_with_positive_infinity(self):
    result, output = emit_cli(V3_RMIN_SOURCE, ("--tensor-profile-version", "v3"))
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn('float("inf")', output)
    self.assertLess(output.index("tl.where("), output.index("tl.min("))

  def test_v3_rmax_masks_with_negative_infinity(self):
    result, output = emit_cli(V3_RMAX_SOURCE, ("--tensor-profile-version", "v3"))
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn('-float("inf")', output)
    self.assertLess(output.index("tl.where("), output.index("tl.max("))

  def test_v3_i32_rmin_masks_with_int_max(self):
    result, output = emit_cli(V3_RMIN_I32_SOURCE, ("--tensor-profile-version", "v3"))
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("2147483647", output)
    self.assertLess(output.index("tl.where("), output.index("tl.min("))

  def test_v2_rmin_emits_no_mask(self):
    result, output = emit_cli(V2_RMIN_SOURCE)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("tl.min(", output)
    # v2 has no inactive lanes, so no neutral value masks the operand; the
    # remaining `tl.where` calls are the signed-zero tie-break.
    self.assertNotIn("inf", output)
    self.assertNotIn("tl.min(tl.where(", output)


@unittest.skipUnless(_triton_interpreter_available(), "Triton interpreter unavailable")
class V3ExtremaExecutionTests(unittest.TestCase):
  """The masked reduction returns the active values, not the neutral one."""

  def run_kernel(self, source, values, output_length):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source_path = root / "case.sir"
      kernel_path = root / "kernel.py"
      source_path.write_text(source, encoding="utf-8")
      emitted = subprocess.run(
        [
          str(SYMIRC),
          "--target",
          "triton",
          "--entry",
          "kernel",
          "--tensor-profile-version",
          "v3",
          "-o",
          str(kernel_path),
          str(source_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
      )
      self.assertEqual(emitted.returncode, 0, emitted.stdout + emitted.stderr)
      runner = root / "runner.py"
      runner.write_text(
        """\
import importlib.util
import sys

import torch

spec = importlib.util.spec_from_file_location("kernel", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
values = [float(value) for value in sys.argv[2].split(",")]
output_length = int(sys.argv[3])
dtype = torch.int32 if sys.argv[4] == "i32" else torch.float32
x = torch.tensor(values, dtype=dtype)
out = torch.full((output_length,), 0, dtype=dtype)
module.kernel[(1, 1, 1)](x, out)
print(out.tolist())
""",
        encoding="utf-8",
      )
      environment = dict(os.environ)
      environment["TRITON_INTERPRET"] = "1"
      completed = subprocess.run(
        [
          sys.executable,
          str(runner),
          str(kernel_path),
          ",".join(str(value) for value in values),
          str(output_length),
          "i32" if "i32" in source else "f32",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env=environment,
      )
      self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
      return completed.stdout.strip()

  def test_v3_extrema_reduces_a_padded_tile_end_to_end(self):
    self.assertEqual(self.run_kernel(V3_RMIN_SOURCE, [5.0, 7.0, 9.0], 1), "[5.0]")

  def test_v3_rmin_all_lanes_inactive(self):
    self.assertEqual(self.run_kernel(V3_RMIN_SINGLE_SOURCE, [5.0], 1), "[5.0]")


if __name__ == "__main__":
  unittest.main()
