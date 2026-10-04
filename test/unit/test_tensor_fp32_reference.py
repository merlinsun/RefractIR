"""Raw-word checks for elementary binary32 RNE reference operations."""

import unittest
from fractions import Fraction

from test.lib.tensor_fp32_reference import add_f32_rne_bits, round_fraction_to_f32_bits


class TensorFp32ReferenceTests(unittest.TestCase):
  def test_half_ulp_tie_rounds_to_even_lower_significand(self):
    self.assertEqual(add_f32_rne_bits(0x3F800000, 0x33800000), 0x3F800000)

  def test_half_ulp_tie_rounds_up_from_odd_significand(self):
    self.assertEqual(add_f32_rne_bits(0x3F800001, 0x33800000), 0x3F800002)

  def test_zero_sign_follows_rne_addition_rules(self):
    self.assertEqual(add_f32_rne_bits(0x80000000, 0x80000000), 0x80000000)
    self.assertEqual(add_f32_rne_bits(0x3F800000, 0xBF800000), 0x00000000)

  def test_subnormal_rounding_uses_even_minimum_subnormal_units(self):
    self.assertEqual(add_f32_rne_bits(0x00000001, 0x00000001), 0x00000002)
    self.assertEqual(round_fraction_to_f32_bits(Fraction(1, 1 << 150)), 0x00000000)
    self.assertEqual(round_fraction_to_f32_bits(Fraction(3, 1 << 150)), 0x00000002)
    self.assertEqual(round_fraction_to_f32_bits(Fraction(-1, 1 << 150)), 0x80000000)

  def test_finite_addition_overflow_is_reported_as_undefined(self):
    with self.assertRaisesRegex(OverflowError, "binary32 overflow"):
      add_f32_rne_bits(0x7F7FFFFF, 0x7F7FFFFF)


if __name__ == "__main__":
  unittest.main()
