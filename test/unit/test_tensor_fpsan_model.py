"""Host-side contracts for the pinned FpSan payload subset."""

import unittest
from subprocess import CompletedProcess
from unittest import mock

from test.lib.tensor_fpsan import (
  add32,
  compare_words,
  embed_known_f32,
  f32_fmod_bits,
  prove_payload_laws,
  reduce_tree,
  rounding_reassociation_bits,
  sub32,
  transpose_2x2,
  unembed_known_payload,
)


class TensorFpSanModelTests(unittest.TestCase):
  def test_raw_word_comparison_accepts_equal_payloads(self):
    self.assertEqual(compare_words([0, 1, 0xFFFFFFFF], [0, 1, 0xFFFFFFFF]), [])

  def test_raw_word_comparison_finds_operand_routing_mutation(self):
    self.assertEqual(compare_words([1, 0, 0xFFFFFFFF], [0, 1, 0xFFFFFFFF]), [0, 1])

  def test_raw_word_comparison_requires_complete_equal_length_outputs(self):
    with self.assertRaisesRegex(ValueError, "length"):
      compare_words([1], [1, 1])

  def test_ieee_f32_control_distinguishes_reassociated_addition(self):
    self.assertEqual(rounding_reassociation_bits(), (0x00000000, 0x33800000))

  def test_ieee_fmod_control_preserves_large_precision_and_negative_zero(self):
    self.assertEqual(f32_fmod_bits(1.0e10, 3.0), 0x3F800000)
    self.assertEqual(f32_fmod_bits(-0.0, 3.0), 0x80000000)

  def test_fixed_embedding_values(self):
    self.assertEqual(embed_known_f32(0.0), 0)
    self.assertEqual(embed_known_f32(1.0), 1)
    self.assertEqual(embed_known_f32(-1.0), 0xFFFFFFFF)
    self.assertEqual(unembed_known_payload(0), 0x00000000)
    self.assertEqual(unembed_known_payload(1), 0x3F800000)
    self.assertEqual(unembed_known_payload(0xFFFFFFFF), 0xBF800000)

  def test_payload_addition_wraps_at_f32_width(self):
    self.assertEqual(add32(0xFFFFFFFF, 1), 0)

  def test_payload_subtraction_wraps_at_f32_width(self):
    self.assertEqual(sub32(0, 1), 0xFFFFFFFF)

  def test_reduction_tree_is_associative_in_the_payload_ring(self):
    values = [1, 0xFFFFFFFF, 1, 0]
    self.assertEqual(reduce_tree(values, "left"), 1)
    self.assertEqual(reduce_tree(values, "balanced"), 1)

  def test_transpose_moves_payloads_without_changing_them(self):
    self.assertEqual(transpose_2x2([1, 0, 0xFFFFFFFF, 1]), [1, 0xFFFFFFFF, 0, 1])

  def test_bitvector_solver_proves_ring_rewrites(self):
    self.assertTrue(prove_payload_laws())

  @mock.patch("test.lib.tensor_fpsan.subprocess.run")
  @mock.patch("test.lib.tensor_fpsan.shutil.which")
  def test_bitvector_proof_uses_bitwuzla_when_available(self, which, run):
    which.side_effect = lambda name: "/usr/bin/bitwuzla" if name == "bitwuzla" else None
    run.return_value = CompletedProcess(["bitwuzla"], 0, "unsat\n", "")

    self.assertTrue(prove_payload_laws())
    args = run.call_args.args[0]
    self.assertEqual(args, ["/usr/bin/bitwuzla"])
    self.assertIn("(set-logic QF_BV)", run.call_args.kwargs["input"])

  @mock.patch("test.lib.tensor_fpsan.subprocess.run")
  @mock.patch("test.lib.tensor_fpsan.shutil.which")
  def test_bitvector_proof_falls_back_to_z3_stdin_mode(self, which, run):
    which.side_effect = lambda name: "/usr/bin/z3" if name == "z3" else None
    run.return_value = CompletedProcess(["z3"], 0, "unsat\n", "")

    self.assertTrue(prove_payload_laws())
    self.assertEqual(run.call_args.args[0], ["/usr/bin/z3", "-in"])

  @mock.patch("test.lib.tensor_fpsan.subprocess.run")
  @mock.patch(
    "test.lib.tensor_fpsan.shutil.which",
    side_effect=lambda name: "/usr/bin/bitwuzla" if name == "bitwuzla" else None,
  )
  def test_bitvector_proof_rejects_sat_results(self, _which, run):
    run.return_value = CompletedProcess(["bitwuzla"], 0, "sat\n", "")
    with self.assertRaisesRegex(AssertionError, "not all UNSAT"):
      prove_payload_laws()

  @mock.patch("test.lib.tensor_fpsan.subprocess.run")
  @mock.patch(
    "test.lib.tensor_fpsan.shutil.which",
    side_effect=lambda name: "/usr/bin/bitwuzla" if name == "bitwuzla" else None,
  )
  def test_bitvector_proof_reports_process_failures(self, _which, run):
    run.return_value = CompletedProcess(["bitwuzla"], 2, "", "bad SMT-LIB")
    with self.assertRaisesRegex(RuntimeError, "bitwuzla failed \(2\)"):
      prove_payload_laws()

  @mock.patch("test.lib.tensor_fpsan.shutil.which", return_value=None)
  def test_bitvector_proof_explains_missing_solver_cli(self, _which):
    with self.assertRaisesRegex(RuntimeError, "bitwuzla or z3 executable"):
      prove_payload_laws()

  def test_host_model_rejects_an_unpinned_float_embedding(self):
    with self.assertRaises(ValueError):
      embed_known_f32(2.0)


if __name__ == "__main__":
  unittest.main()
