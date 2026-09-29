"""Behavioral checks for tensor syntax, shape rules, and target gates."""

import tempfile
import unittest
from pathlib import Path

from test.unit.tensor_support import (
  EMIT_API,
  SYMIRC,
  SYMIRI,
  check_source,
  compile_source,
  compile_via_api,
  sir,
)


def tensor_program(param_type, result_type, expression, result="%out"):
  declarations = f"  let mut %out: {result_type} = undef;"
  return sir(
    f"%x: {param_type}",
    result_type,
    declarations,
    f"  %out = {expression};",
    result,
  )


def remainder_program(lhs_type="tensor [2] f32", rhs_type="tensor [2] f32"):
  return f"""\
fun @kernel(%x: {lhs_type}, %y: {rhs_type}) : {lhs_type} {{
  let mut %out: {lhs_type} = undef;
^entry:
  %out = tensor rem %x, %y;
  ret %out;
}}
"""


class TensorFrontendTests(unittest.TestCase):
  def setUp(self):
    self.assertTrue(SYMIRC.is_file(), f"symirc executable not found: {SYMIRC}")
    self.assertTrue(SYMIRI.is_file(), f"symiri executable not found: {SYMIRI}")
    self.assertTrue(
      EMIT_API.is_file(), f"shared emission API helper not found: {EMIT_API}"
    )

  def test_shape_identity(self):
    good = tensor_program("tensor [2, 4] f32", "tensor [2, 4] f32", "%x", result="%x")
    bad = tensor_program("tensor [2, 4] f32", "tensor [4, 2] f32", "%x", result="%x")

    accepted = check_source(good)
    rejected = check_source(bad)

    self.assertEqual(accepted.returncode, 0, accepted.stderr)
    self.assertNotEqual(rejected.returncode, 0)
    self.assertIn("tensor-type", rejected.stderr)

  def test_tensor_rem_requires_v4(self):
    source = remainder_program()

    accepted = check_source(source, tensor_profile="v4")
    rejected = check_source(source, tensor_profile="v3")

    self.assertEqual(accepted.returncode, 0, accepted.stderr)
    self.assertNotEqual(rejected.returncode, 0)

  def test_tensor_rem_requires_equal_shape_f32_vector(self):
    accepted = check_source(remainder_program(), tensor_profile="v4")
    invalid_sources = (
      remainder_program("tensor [2] i32", "tensor [2] i32"),
      remainder_program("tensor [2, 1] f32", "tensor [2, 1] f32"),
      remainder_program("tensor [2] f32", "tensor [4] f32"),
      remainder_program("tensor [1024] f32", "tensor [1024] f32"),
    )
    rejected = [check_source(source, tensor_profile="v4") for source in invalid_sources]

    self.assertEqual(accepted.returncode, 0, accepted.stderr)
    for result in rejected:
      self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_tensor_rem_must_be_the_complete_returned_computation(self):
    source = """\
fun @kernel(%x: tensor [2] f32, %y: tensor [2] f32) : tensor [2] f32 {
  let mut %remainder: tensor [2] f32 = undef;
  let mut %out: tensor [2] f32 = undef;
^entry:
  %remainder = tensor rem %x, %y;
  %out = tensor add %remainder, %x;
  ret %out;
}
"""

    result = check_source(source, tensor_profile="v4")

    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("tensor-v4", result.stderr)

  def test_rectangular_broadcast(self):
    good = tensor_program(
      "tensor [2, 1] i32",
      "tensor [2, 4] i32",
      "tensor broadcast %x, [2, 4]",
    )
    bad = tensor_program(
      "tensor [2] i32",
      "tensor [2, 4] i32",
      "tensor broadcast %x, [2, 4]",
    )

    accepted = check_source(good)
    rejected = check_source(bad)

    self.assertEqual(accepted.returncode, 0, accepted.stderr)
    self.assertNotEqual(rejected.returncode, 0)
    self.assertIn("tensor-broadcast", rejected.stderr)

  def test_reduction_shape_modes(self):
    source = """\
fun @keep(%x: tensor [2, 4] i32) : tensor [2, 1] i32 {
  let mut %out: tensor [2, 1] i32 = undef;
^entry:
  %out = tensor sum %x, -1, keep;
  ret %out;
}
fun @drop(%x: tensor [2, 4] i32) : tensor [2] i32 {
  let mut %out: tensor [2] i32 = undef;
^entry:
  %out = tensor sum %x, -1, drop;
  ret %out;
}
fun @rank_one_drop(%x: tensor [4] i32) : i32 {
  let mut %out: i32 = undef;
^entry:
  %out = tensor sum %x, 0, drop;
  ret %out;
}
"""
    result = check_source(source)

    self.assertEqual(result.returncode, 0, result.stderr)

  def test_axis_and_permutation_validation(self):
    invalid_axis = tensor_program(
      "tensor [2, 4] i32",
      "tensor [2] i32",
      "tensor sum %x, 2, drop",
    )
    invalid_permutation = tensor_program(
      "tensor [2, 4] i32",
      "tensor [2, 2] i32",
      "tensor transpose %x, [0, 0]",
    )

    axis_result = check_source(invalid_axis)
    permutation_result = check_source(invalid_permutation)

    self.assertNotEqual(axis_result.returncode, 0)
    self.assertIn("tensor-axis", axis_result.stderr)
    self.assertNotEqual(permutation_result.returncode, 0)
    self.assertIn("tensor-permutation", permutation_result.stderr)

  def test_unsupported_target_preserves_output(self):
    source = tensor_program("tensor [2] f32", "tensor [2] f32", "%x", result="%x")
    sentinel = b"keep-existing-output\x00exactly"

    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for target in ("c", "wasm", "python"):
        output = root / f"output.{target}"
        output.write_bytes(sentinel)

        result = compile_source(source, target, output)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tensor-target", result.stderr)
        self.assertEqual(output.read_bytes(), sentinel)

        api_output = root / f"api-output.{target}"
        api_output.write_bytes(sentinel)
        api_result = compile_via_api(source, target, api_output)

        self.assertNotEqual(api_result.returncode, 0)
        self.assertIn("tensor-target", api_result.stderr)
        self.assertEqual(api_output.read_bytes(), sentinel)

  def test_singleton_negative_expand_axis(self):
    source = tensor_program(
      "tensor [1] i32", "tensor [1, 1] i32", "tensor expand %x, -1"
    )

    result = check_source(source)

    self.assertEqual(result.returncode, 0, result.stderr)

  def test_recursive_pointer_struct_does_not_recurse_forever(self):
    source = """\
struct @Node { next: ptr @Node; }
fun @kernel() : i32 {
^entry:
  ret 0;
}
"""
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / "kernel.c"
      result = compile_source(source, "c", output)
      self.assertEqual(result.returncode, 0, result.stderr)
      self.assertTrue(output.is_file())

  def test_partially_undefined_tensor_initializer_is_rejected(self):
    source = """\
fun @kernel() : i32 {
  let %x: tensor [2] i32 = {undef, 1};
^entry:
  ret 0;
}
"""
    result = check_source(source)

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("tensor-init", result.stderr)

  def test_branch_condition_tensor_atom_is_seen_by_target_gate(self):
    source = """\
fun @kernel(%x: f32) : i32 {
^entry:
  br tensor exp %x < 2.0, ^yes, ^no;
^yes:
  ret 1;
^no:
  ret 0;
}
"""
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / "kernel.c"
      result = compile_source(source, "c", output)

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("tensor-target", result.stderr)
    self.assertFalse(output.exists())

  def test_branch_condition_tensor_atom_gets_reference_profile_checks(self):
    source = """\
fun @kernel(%x: f32) : i32 {
^entry:
  br tensor exp %x < 2.0, ^yes, ^no;
^yes:
  ret 1;
^no:
  ret 0;
}
"""
    result = check_source(source)

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("tensor-profile", result.stderr)
    self.assertIn("one basic block", result.stderr)

  def test_select_condition_tensor_atom_gets_reference_profile_checks(self):
    source = """\
fun @kernel(%x: f32) : i32 {
  let mut %r: i32 = undef;
^entry:
  %r = select tensor exp %x < 2.0, 1, 0;
  br ^done;
^done:
  ret %r;
}
"""
    result = check_source(source)

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("tensor-profile", result.stderr)
    self.assertIn("one basic block", result.stderr)

  def test_tensor_atom_on_select_condition_rhs_is_detected(self):
    source = """\
fun @kernel(%x: f32) : i32 {
  let mut %r: i32 = undef;
^entry:
  %r = select 2.0 < tensor exp %x, 1, 0;
  br ^done;
^done:
  ret %r;
}
"""
    result = check_source(source)

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("tensor-profile", result.stderr)
    self.assertIn("one basic block", result.stderr)

  def test_call_argument_tensor_atom_gets_no_call_profile_check(self):
    source = """\
fun @helper(%value: f32) : i32 {
^entry:
  ret 1;
}
fun @kernel(%x: f32) : i32 {
^entry:
  ret call @helper(tensor exp %x);
}
"""
    result = check_source(source)

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("tensor-profile", result.stderr)
    self.assertIn("cannot contain calls", result.stderr)

  def test_select_condition_tensor_atom_gets_legacy_target_rejection(self):
    source = """\
fun @kernel(%x: f32) : i32 {
  let mut %r: i32 = undef;
^entry:
  %r = select tensor exp %x < 2.0, 1, 0;
  ret %r;
}
"""
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / "kernel.c"
      result = compile_source(source, "c", output)

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("tensor-target", result.stderr)
    self.assertFalse(output.exists())

  def test_call_argument_tensor_atom_gets_legacy_target_rejection(self):
    source = """\
fun @helper(%value: f32) : i32 {
^entry:
  ret 1;
}
fun @kernel(%x: f32) : i32 {
^entry:
  ret call @helper(tensor exp %x);
}
"""
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / "kernel.c"
      result = compile_source(source, "c", output)

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("tensor-target", result.stderr)
    self.assertFalse(output.exists())

  def test_call_nested_in_select_condition_is_rejected_by_tensor_profile(self):
    source = """\
fun @helper(%value: i32) : i32 {
^entry:
  ret %value;
}
fun @kernel(%x: f32) : i32 {
  let mut %t: f32 = undef;
  let mut %r: i32 = undef;
^entry:
  %t = tensor exp %x;
  %r = select call @helper(1) < 2, 1, 0;
  ret %r;
}
"""
    result = check_source(source)

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("tensor-profile", result.stderr)
    self.assertIn("cannot contain calls", result.stderr)

  def test_scalar_function_with_tensor_atom_gets_no_call_profile_check(self):
    source = """\
fun @helper() : i32 {
^entry:
  ret 1;
}
fun @kernel(%x: f32) : i32 {
  let mut %y: f32 = undef;
  let mut %z: i32 = undef;
^entry:
  %y = tensor exp %x;
  %z = call @helper();
  ret %z;
}
"""
    result = check_source(source)

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("tensor-profile", result.stderr)
    self.assertIn("cannot contain calls", result.stderr)

  def test_profile_rejection_matrix(self):
    def scalar_return(type_):
      return f"fun @kernel(%x: {type_}) : i32 {{\n^entry:\n  ret 0;\n}}\n"

    cases = {
      "rank zero": (scalar_return("tensor [] i32"), "tensor-profile"),
      "rank four": (scalar_return("tensor [1, 1, 1, 1] i32"), "tensor-profile"),
      "zero dimension": (scalar_return("tensor [0] i32"), "tensor-shape"),
      "non-power dimension": (scalar_return("tensor [3] i32"), "tensor-profile"),
      "too many elements": (scalar_return("tensor [512] i32"), "tensor-profile"),
      "f64 element": (scalar_return("tensor [2] f64"), "tensor-type"),
      "nested tensor": (scalar_return("tensor [2] tensor [2] i32"), "tensor-type"),
      "array of tensor": (scalar_return("[2] tensor [2] i32"), "tensor-type"),
      "pointer to tensor": (scalar_return("ptr tensor [2] i32"), "tensor-type"),
      "struct field tensor": (
        "struct @WithTensor { value: tensor [2] i32; }\n" + scalar_return("i32"),
        "tensor-type",
      ),
      "symbolic axis": (
        "fun @kernel(%x: tensor [4] i32, %axis: i32) : i32 {\n"
        "  let mut %out: i32 = undef;\n^entry:\n"
        "  %out = tensor sum %x, %axis, drop;\n  ret %out;\n}\n",
        "tensor-axis",
      ),
      "duplicate permutation": (
        tensor_program(
          "tensor [2, 4] i32", "tensor [2, 2] i32", "tensor transpose %x, [0, 0]"
        ),
        "tensor-permutation",
      ),
      "tensor element write": (
        "fun @kernel(%x: tensor [2] i32) : i32 {\n^entry:\n  %x[0] = 1;\n  ret 0;\n}\n",
        "tensor-type",
      ),
      "multiple blocks": (
        "fun @kernel(%x: tensor [2] i32) : tensor [2] i32 {\n"
        "  let mut %out: tensor [2] i32 = undef;\n"
        "^entry:\n  br ^done;\n^done:\n  %out = %x;\n  ret %out;\n}\n",
        "tensor-profile",
      ),
      "tensor call": (
        "fun @copy(%x: tensor [2] i32) : tensor [2] i32 {\n^entry:\n  ret %x;\n}\n"
        "fun @kernel(%x: tensor [2] i32) : tensor [2] i32 {\n"
        "  let mut %out: tensor [2] i32 = undef;\n^entry:\n"
        "  %out = call @copy(%x);\n  ret %out;\n}\n",
        "tensor-profile",
      ),
    }

    for description, (source, diagnostic) in cases.items():
      with self.subTest(description=description):
        result = check_source(source)
        self.assertNotEqual(result.returncode, 0, f"{description} unexpectedly passed")
        self.assertIn(diagnostic, result.stderr, result.stderr)


if __name__ == "__main__":
  unittest.main()
