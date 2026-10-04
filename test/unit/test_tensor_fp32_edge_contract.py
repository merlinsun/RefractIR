"""Host contracts for IEEE binary32 GPU edge controls."""

import importlib
import unittest


class TensorFp32EdgeContractTests(unittest.TestCase):
  def reference(self):
    return importlib.import_module("test.lib.tensor_fp32_reference")

  def test_multiply_rounds_exact_product_to_nearest_even(self):
    self.assertEqual(
      self.reference().mul_f32_rne_bits(0x3F800001, 0x3F7FFFFE), 0x3F800000
    )

  def test_fused_multiply_add_keeps_the_product_residual(self):
    self.assertEqual(
      self.reference().fma_f32_rne_bits(0x3F800001, 0x3F7FFFFE, 0xBF800000),
      0xA8800000,
    )

  def test_disabled_fusion_rounds_product_before_addition(self):
    reference = self.reference()
    separate = reference.mul_add_separate_f32_bits(0x3F800001, 0x3F7FFFFE, 0xBF800000)
    fused = reference.fma_f32_rne_bits(0x3F800001, 0x3F7FFFFE, 0xBF800000)
    self.assertEqual(separate, 0x00000000)
    self.assertNotEqual(separate, fused)

  def test_subnormal_multiplication_uses_gradual_underflow(self):
    self.assertEqual(self.reference().mul_f32_rne_bits(0x00000001, 0x40000000), 2)

  def test_multiplication_preserves_signed_zero_from_operand_signs(self):
    multiply = self.reference().mul_f32_rne_bits

    self.assertEqual(multiply(0x80000000, 0x3F800000), 0x80000000)
    self.assertEqual(multiply(0x80000000, 0xBF800000), 0x00000000)

  def test_reduction_oracle_allows_each_binary32_tree_result(self):
    outcomes = self.reference().reduction_f32_rne_outcomes(
      (0x3F800000, 0x33800000, 0xBF800000)
    )
    self.assertEqual(outcomes, frozenset((0x00000000, 0x33800000)))

  def test_singleton_reduction_preserves_negative_zero_bits(self):
    outcomes = self.reference().reduction_f32_rne_outcomes((0x80000000,))
    self.assertEqual(outcomes, frozenset((0x80000000,)))


if __name__ == "__main__":
  unittest.main()
