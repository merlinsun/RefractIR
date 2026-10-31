"""`tensor narrow`: rounding f32 lanes to f16 or bf16 (feature-ranked plan, L2)."""

import struct
import unittest

import numpy as np

from test.unit.tensor_support import SYMIRC, SYMIRI, check_source, run_typed
from test.unit.test_tensor_emit import emit_cli


def narrow_program(fmt, count=8, element="f32"):
  return f"""\
fun @kernel(%x: tensor [{count}] {element}) : tensor [{count}] {element} {{
  let mut %out: tensor [{count}] {element} = undef;
^entry:
  %out = tensor narrow {fmt} %x;
  ret %out;
}}
"""


def bits(value):
  return struct.unpack("<I", struct.pack("<f", value))[0]


def bf16_rne(value):
  """Round an f32 to bf16 by its bits: round to nearest, ties to even."""
  word = bits(value)
  word = (word + 0x7FFF + ((word >> 16) & 1)) & 0xFFFF0000
  return struct.unpack("<f", struct.pack("<I", word))[0]


def run(fmt, values):
  return run_typed(
    narrow_program(fmt, len(values)),
    struct.pack(f"<{len(values)}f", *values),
    tensor_profile="v3",
    reference_bits=128,
  )


@unittest.skipUnless(SYMIRI.is_file(), "symiri is unavailable")
class NarrowTypingTests(unittest.TestCase):
  def test_narrow_keeps_the_f32_type_under_v2_and_v3(self):
    for fmt in ("f16", "bf16"):
      for profile in ("v2", "v3"):
        with self.subTest(fmt=fmt, profile=profile):
          result = check_source(narrow_program(fmt), tensor_profile=profile)
          self.assertEqual(result.returncode, 0, result.stderr)

  def test_narrow_rejects_integers_and_unknown_formats(self):
    result = check_source(narrow_program("f16", element="i32"), tensor_profile="v3")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("tensor-narrow", result.stderr)
    result = check_source(narrow_program("e4m3"), tensor_profile="v3")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("narrow format", result.stderr)


@unittest.skipUnless(SYMIRI.is_file(), "symiri is unavailable")
class NarrowReferenceTests(unittest.TestCase):
  def test_f16_rounds_to_nearest_even_with_subnormals_and_signed_zero(self):
    values = [
      1.0,
      1 + 2**-11,  # a tie: stays 1
      1 + 3 * 2**-11,  # a tie: rounds up to the even neighbour
      65504.0,
      2.0**-25,  # half the least subnormal: a tie to zero
      3 * 2.0**-26,  # rounds to the least subnormal
      -0.0,
      0.1,
    ]
    result, output = run("f16", values)
    self.assertEqual(result.returncode, 0, result.stderr)
    expected = [float(np.float32(np.float16(value))) for value in values]
    self.assertEqual(
      struct.unpack("<8I", output), tuple(bits(value) for value in expected)
    )

  def test_a_tiny_negative_lane_keeps_its_sign_in_f16(self):
    result, output = run("f16", [-(2.0**-30)] + [0.0] * 7)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(struct.unpack("<I", output[:4])[0], 0x80000000)

  def test_f16_overflow_is_undefined(self):
    result, output = run("f16", [65519.0] + [0.0] * 7)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(struct.unpack("<f", output[:4])[0], 65504.0)
    result, _ = run("f16", [-65520.0] + [0.0] * 7)
    self.assertEqual(result.returncode, 5, result.stderr)
    self.assertIn("UB: tensor narrow overflows f16", result.stderr)

  def test_bf16_rounds_to_nearest_even_and_overflow_is_undefined(self):
    values = [
      1.0,
      1 + 2**-8,  # a tie: stays 1
      1 + 3 * 2**-8,  # a tie: rounds up
      -(2.0**-130),  # an f32 subnormal survives
      3.0e38,
      -0.0,
      0.1,
      12345.678,
    ]
    result, output = run("bf16", values)
    self.assertEqual(result.returncode, 0, result.stderr)
    expected = [bf16_rne(float(np.float32(value))) for value in values]
    self.assertEqual(
      struct.unpack("<8I", output), tuple(bits(value) for value in expected)
    )
    result, _ = run("bf16", [3.4e38] + [0.0] * 7)
    self.assertEqual(result.returncode, 5, result.stderr)
    self.assertIn("UB: tensor narrow overflows bf16", result.stderr)


@unittest.skipUnless(SYMIRC.is_file(), "symirc is unavailable")
class NarrowEmissionTests(unittest.TestCase):
  def test_narrow_rounds_through_the_format_and_widens_back(self):
    for fmt, dtype in (("f16", "float16"), ("bf16", "bfloat16")):
      with self.subTest(fmt=fmt):
        result, output = emit_cli(
          narrow_program(fmt), ("--tensor-profile-version", "v3")
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(
          output,
          rf"\w+\.to\(tl\.{dtype}, fp_downcast_rounding=\"rtne\"\)\.to\(tl\.float32\)",
        )


if __name__ == "__main__":
  unittest.main()
