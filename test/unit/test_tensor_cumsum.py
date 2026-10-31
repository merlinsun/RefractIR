"""`tensor cumsum`: inclusive prefix sums along one axis (feature-ranked plan, L6)."""

import struct
import unittest

from test.unit.tensor_support import SYMIRC, SYMIRI, check_source, run_typed
from test.unit.test_tensor_emit import emit_cli


def cumsum_program(axis, element="i32", shape=(2, 4)):
  dims = ", ".join(str(d) for d in shape)
  return f"""\
fun @kernel(%x: tensor [{dims}] {element}) : tensor [{dims}] {element} {{
  let mut %out: tensor [{dims}] {element} = undef;
^entry:
  %out = tensor cumsum %x, {axis};
  ret %out;
}}
"""


@unittest.skipUnless(SYMIRI.is_file(), "symiri is unavailable")
class CumsumTypingTests(unittest.TestCase):
  def test_cumsum_keeps_its_shape_under_v2_and_v3(self):
    for profile in ("v2", "v3"):
      for axis in (0, 1, -1):
        with self.subTest(profile=profile, axis=axis):
          result = check_source(cumsum_program(axis), tensor_profile=profile)
          self.assertEqual(result.returncode, 0, result.stderr)

  def test_an_axis_outside_the_rank_is_refused(self):
    result = check_source(cumsum_program(2), tensor_profile="v3")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("tensor-axis", result.stderr)


@unittest.skipUnless(SYMIRI.is_file(), "symiri is unavailable")
class CumsumReferenceTests(unittest.TestCase):
  def test_i32_prefixes_along_each_axis(self):
    values = [1, 2, 3, 4, 10, 20, 30, 40]
    result, output = run_typed(
      cumsum_program(1), struct.pack("<8i", *values), tensor_profile="v3"
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(struct.unpack("<8i", output), (1, 3, 6, 10, 10, 30, 60, 100))
    result, output = run_typed(
      cumsum_program(0), struct.pack("<8i", *values), tensor_profile="v3"
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(struct.unpack("<8i", output), (1, 2, 3, 4, 11, 22, 33, 44))

  def test_a_tailed_axis_prefixes_only_its_logical_lanes(self):
    values = list(range(1, 10))
    result, output = run_typed(
      cumsum_program(0, shape=(3, 3)), struct.pack("<9i", *values), tensor_profile="v3"
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(struct.unpack("<9i", output), (1, 2, 3, 5, 7, 9, 12, 15, 18))

  def test_i32_overflow_in_a_prefix_is_undefined(self):
    values = [2**31 - 1, 1, 0, 0, 0, 0, 0, 0]
    result, _ = run_typed(
      cumsum_program(1), struct.pack("<8i", *values), tensor_profile="v3"
    )
    self.assertEqual(result.returncode, 5, result.stderr)
    self.assertIn("UB: tensor cumsum", result.stderr)

  def test_f32_prefixes_are_exact_sums_rounded_once(self):
    # 2^24 + 1 + 1: in order f32 loses both ones; each exact prefix rounds once.
    values = [float(2**24), 1.0, 1.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    result, output = run_typed(
      cumsum_program(1, "f32"),
      struct.pack("<8f", *values),
      tensor_profile="v3",
      reference_bits=128,
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(
      struct.unpack("<8f", output),
      (2.0**24, 2.0**24, 2.0**24 + 2, 2.0**24 + 4, 0.5, 0.75, 0.875, 0.9375),
    )


@unittest.skipUnless(SYMIRC.is_file(), "symirc is unavailable")
class CumsumEmissionTests(unittest.TestCase):
  def test_cumsum_lowers_to_tl_cumsum_on_its_axis(self):
    result, output = emit_cli(
      cumsum_program(-1, "f32"), ("--tensor-profile-version", "v3")
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertRegex(output, r"tl\.cumsum\(\w+, axis=1\)")


if __name__ == "__main__":
  unittest.main()
