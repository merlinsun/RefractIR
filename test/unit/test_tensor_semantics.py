"""Combined tensor test coverage: test_tensor_exact.py, test_tensor_numeric.py."""

import struct
import unittest

from test.unit.tensor_support import SYMIRI, describe_entry, run_typed, sir


# Consolidated from test_tensor_exact.py.
def declarations(types):
  return "\n".join(f"  let mut %{name}: {ty} = undef;" for name, ty in types)


class TensorExactTests(unittest.TestCase):
  def setUp(self):
    self.assertTrue(SYMIRI.is_file(), f"symiri executable not found: {SYMIRI}")

  def test_coordinate_formula(self):
    shape = "tensor [2, 4] i32"
    locals_ = declarations(
      (name, shape) for name in ("row", "col", "r7", "c3", "sum", "out")
    )
    source = sir(
      "",
      shape,
      locals_,
      "\n".join(
        (
          "  %row = tensor iota [2, 4], 0;",
          "  %col = tensor iota [2, 4], 1;",
          "  %r7 = tensor mul %row, 7;",
          "  %c3 = tensor mul %col, 3;",
          "  %sum = tensor add %r7, %c3;",
          "  %out = tensor add %sum, 1;",
        )
      ),
    )

    result, output = run_typed(source, b"")

    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(output, struct.pack("<8i", 1, 4, 7, 10, 8, 11, 14, 17))

  def test_rectangular_transpose(self):
    source = sir(
      "%x: tensor [2, 4] i32",
      "tensor [4, 2] i32",
      "  let mut %out: tensor [4, 2] i32 = undef;",
      "  %out = tensor transpose %x, [1, 0];",
      result="%out",
    )
    values = struct.pack("<8i", *range(8))

    result, output = run_typed(source, values)

    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(output, struct.pack("<8i", 0, 4, 1, 5, 2, 6, 3, 7))

  def test_broadcast_correspondence(self):
    source = sir(
      "%a: tensor [2, 1] i32, %b: tensor [1, 4] i32",
      "tensor [2, 4] i32",
      "  let mut %out: tensor [2, 4] i32 = undef;",
      "  %out = tensor add %a, %b;",
      result="%out",
    )
    values = struct.pack("<2i", 10, 20) + struct.pack("<4i", 1, 2, 3, 4)

    result, output = run_typed(source, values)

    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(output, struct.pack("<8i", 11, 12, 13, 14, 21, 22, 23, 24))

  def test_reassociation_overflow_rejected(self):
    source = sir(
      "%x: tensor [4] i32",
      "i32",
      "  let mut %out: i32 = undef;",
      "  %out = tensor sum %x, 0, drop;",
      result="%out",
    )

    result, _ = run_typed(source, struct.pack("<4i", 2**31 - 1, 1, -1, 0))

    self.assertEqual(result.returncode, 5, result.stderr)
    self.assertIn("UB: tensor sum partial overflow", result.stderr)

  def test_binary_input_validation(self):
    source = sir(
      "%x: tensor [2] f32",
      "tensor [2] f32",
      "  let mut %out: tensor [2] f32 = undef;",
      "  %out = %x;",
      result="%out",
    )
    valid = struct.pack("<2f", -0.0, 1.25)

    result, output = run_typed(source, valid)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(output, valid)

    for malformed in (valid[:-1], valid + b"x", struct.pack("<2I", 0, 0x7F800000)):
      with self.subTest(malformed=malformed):
        result, output = run_typed(source, malformed)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(output)

  def test_layout_preserves_negative_zero_and_subnormal(self):
    source = sir(
      "%x: tensor [2, 2] f32",
      "tensor [2, 2] f32",
      "  let mut %out: tensor [2, 2] f32 = undef;",
      "  %out = tensor transpose %x, [1, 0];",
      result="%out",
    )
    source_words = (0x80000000, 0x00000001, 0x3F800000, 0x80000000)
    expected_words = (0x80000000, 0x3F800000, 0x00000001, 0x80000000)

    result, output = run_typed(source, struct.pack("<4I", *source_words))

    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(output, struct.pack("<4I", *expected_words))

  def test_tensor_describe_signature(self):
    source = sir(
      "%x: tensor [2, 4] i32, %bias: i32",
      "tensor [2, 4] i32",
      "  let mut %out: tensor [2, 4] i32 = undef;",
      "  %out = tensor add %x, %bias;",
      result="%out",
    )

    result, description = describe_entry(source)

    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(description["entry"], "kernel")
    self.assertEqual(
      description["inputs"][0],
      {
        "name": "x",
        "dtype": "i32",
        "shape": [2, 4],
        "byte_length": 32,
      },
    )
    self.assertEqual(
      description["inputs"][1],
      {
        "name": "bias",
        "dtype": "i32",
        "shape": [],
        "byte_length": 4,
      },
    )
    self.assertEqual(
      description["output"],
      {
        "name": "return",
        "dtype": "i32",
        "shape": [2, 4],
        "byte_length": 32,
      },
    )


if __name__ == "__main__":
  unittest.main()


# Consolidated from test_tensor_numeric.py.
class TensorNumericTests(unittest.TestCase):
  def test_f32_tensor_rem_reference_uses_fmod(self):
    source = """\
fun @kernel(%x: tensor [2] f32, %y: tensor [2] f32) : tensor [2] f32 {
  let mut %out: tensor [2] f32 = undef;
^entry:
  %out = tensor rem %x, %y;
  ret %out;
}
"""
    inputs = struct.pack("<4f", -0.0, 1.0, 3.0, 0.1)
    result, output = run_typed(source, inputs, tensor_profile="v4", reference_bits=128)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(output, struct.pack("<2I", 0x80000000, 0x3DCCCCCB))

  def test_f32_tensor_rem_reference_signs_and_zero_divisor(self):
    source = """\
fun @kernel(%x: tensor [8] f32, %y: tensor [8] f32) : tensor [8] f32 {
  let mut %out: tensor [8] f32 = undef;
^entry:
  %out = tensor rem %x, %y;
  ret %out;
}
"""
    values = (-5.0, 5.0, -6.0, 6.0, 1.0, -1.0, 5.0, -5.0)
    divisors = (3.0, 3.0, 3.0, 3.0, 0.1, 0.1, -3.0, -3.0)
    result, output = run_typed(
      source,
      struct.pack("<16f", *(values + divisors)),
      tensor_profile="v4",
      reference_bits=128,
    )

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(
      output,
      struct.pack(
        "<8I",
        0xC0000000,
        0x40000000,
        0x80000000,
        0x00000000,
        0x3DCCCCCB,
        0xBDCCCCCB,
        0x40000000,
        0xC0000000,
      ),
    )

    zero_divisors = run_typed(
      source,
      struct.pack("<16f", *(values + (0.0, 3.0, 3.0, 3.0, 0.1, 0.1, -3.0, -3.0))),
      tensor_profile="v4",
      reference_bits=128,
    )[0]
    self.assertNotEqual(zero_divisors.returncode, 0)
    self.assertIn("remainder by zero", zero_divisors.stderr)

  def test_softmax_is_row_wise(self):
    source = sir(
      "%x: tensor [2, 2] f32",
      "tensor [2, 2] f32",
      "  let mut %out: tensor [2, 2] f32 = undef;",
      "  %out = tensor softmax %x, 1;",
      result="%out",
    )

    result, output = run_typed(
      source, struct.pack("<4f", 0.0, 2.0, 3.0, 1.0), reference_bits=128
    )

    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(len(output), 16)
    values = struct.unpack("<4f", output)
    for actual, expected in zip(
      values, (0.11920292, 0.88079708, 0.88079708, 0.11920292)
    ):
      self.assertAlmostEqual(actual, expected, delta=1e-6)
    self.assertAlmostEqual(values[0] + values[1], 1.0, delta=1e-6)
    self.assertAlmostEqual(values[2] + values[3], 1.0, delta=1e-6)

  def test_equal_logits_and_rank_three_axis(self):
    shape = "tensor [2, 1, 4] f32"
    source = sir(
      f"%x: {shape}",
      shape,
      f"  let mut %out: {shape} = undef;",
      "  %out = tensor softmax %x, -1;",
      result="%out",
    )

    result, output = run_typed(
      source, struct.pack("<8f", *([7.0] * 8)), reference_stability=True
    )

    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(output, struct.pack("<8f", *([0.25] * 8)))

  def test_numerical_nodes_round_at_boundaries(self):
    source = sir(
      "%x: tensor [2] f32",
      "tensor [1] f32",
      "  let mut %sum: tensor [1] f32 = undef;\n  let mut %out: tensor [1] f32 = undef;",
      "\n".join(
        (
          "  %sum = tensor sum %x, 0, keep;",
          "  %out = tensor sub %sum, 16777216.0;",
        )
      ),
      result="%out",
    )

    result, output = run_typed(
      source, struct.pack("<2f", 16777216.0, 1.0), reference_bits=128
    )

    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(output, struct.pack("<f", 0.0))

  def test_wrong_normalized_output_is_rejected(self):
    from test.lib.tensor_compare import compare

    expected = struct.pack("<4f", 0.11920292, 0.88079708, 0.88079708, 0.11920292)
    uniform = struct.pack("<4f", 0.5, 0.5, 0.5, 0.5)

    report = compare(uniform, expected, "f32", False, (2, 100000), (0, 1))

    self.assertEqual(report["status"], "numeric_mismatch")
    self.assertEqual(report["mismatching_indices"], [0, 1, 2, 3])

  def test_unstable_reference_is_withheld(self):
    from test.lib.tensor_compare import run_if_reference_stable

    launched = []
    report = run_if_reference_stable(
      [b"first-node", b"second-node-128"],
      [b"first-node", b"second-node-256"],
      lambda: launched.append(True) or {"status": "pass"},
    )

    self.assertEqual(report["status"], "oracle_unstable")
    self.assertFalse(report["target_eligible"])
    self.assertFalse(launched)

  def test_f32_tensor_splat_is_rounded_before_numerical_reference(self):
    source = """\
fun @kernel() : tensor [1] f32 {
  let mut %source: tensor [1] f32 = 1.00000005;
  let mut %out: tensor [1] f32 = undef;
^entry:
  %out = tensor mul %source, 16777215.0;
  ret %out;
}
"""
    result, output = run_typed(source, b"", reference_bits=128)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(output, struct.pack("<f", 16777215.0))

  def test_explicit_tensor_broadcast_rounds_before_numerical_reference(self):
    source = """\
fun @kernel() : tensor [1] f32 {
  let mut %source: tensor [1] f32 = undef;
  let mut %out: tensor [1] f32 = undef;
^entry:
  %source = tensor broadcast 1.00000005, [1];
  %out = tensor mul %source, 16777215.0;
  ret %out;
}
"""
    result, output = run_typed(source, b"", reference_bits=128)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(output, struct.pack("<f", 16777215.0))

  def test_zero_reference_uses_absolute_tolerance(self):
    from test.lib.tensor_compare import compare

    reference = struct.pack("<f", 0.0)
    inside = struct.pack("<f", 0.000019)
    outside = struct.pack("<f", 0.000021)

    self.assertEqual(
      compare(inside, reference, "f32", False, (2, 100000), (0, 1))["status"],
      "pass",
    )
    self.assertEqual(
      compare(outside, reference, "f32", False, (2, 100000), (0, 1))["status"],
      "numeric_mismatch",
    )

  def test_numeric_zero_reference_requires_matching_zero_bits(self):
    from test.lib.tensor_compare import compare

    reference = struct.pack("<I", 0x80000000)
    wrong_sign = struct.pack("<f", 0.0)
    nonzero_within_tolerance = struct.pack("<f", -(2.0**-80))

    def compare_v4(actual):
      try:
        return compare(
          actual,
          reference,
          "f32",
          False,
          (2, 100000),
          (0, 1),
          zero_sign_exact=True,
        )
      except TypeError as error:
        self.fail(f"the v4 zero-sign comparison policy is missing: {error}")

    self.assertEqual(compare_v4(wrong_sign)["status"], "numeric_mismatch")
    self.assertEqual(compare_v4(nonzero_within_tolerance)["status"], "numeric_mismatch")


if __name__ == "__main__":
  unittest.main()
