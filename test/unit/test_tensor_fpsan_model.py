"""Host-side contracts for the pinned FpSan payload subset."""

import hashlib
import importlib
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
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
  def fpsan(self):
    return importlib.import_module("test.lib.tensor_fpsan")

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

  def test_kernel_query_contains_its_per_kernel_bitvector_equations(self):
    api = self.fpsan()
    query = api.build_payload_query("sum_reduction")
    self.assertEqual(query["variables"], ("x0", "x1", "x2", "x3"))
    self.assertEqual(
      query["reference_expressions"],
      ("(bvadd (bvadd (bvadd x0 x1) x2) x3)",),
    )
    self.assertEqual(
      query["candidate_expressions"],
      ("(bvadd (bvadd x0 x1) (bvadd x2 x3))",),
    )
    self.assertIn("(set-logic QF_BV)", query["script"])
    self.assertIn("(set-option :produce-models true)", query["script"])
    self.assertIn("#x00000001", query["script"])

  def test_solver_model_parser_preserves_full_width_payloads(self):
    model = "sat\n((a #x00000001) (b #b11111111111111111111111111111111))\n"
    self.assertEqual(
      self.fpsan().parse_payload_model(model, ("a", "b")),
      {"a": 1, "b": 0xFFFFFFFF},
    )

  def test_independent_evaluator_covers_arithmetic_sums_and_shape_ops(self):
    api = self.fpsan()
    inputs = {"a": 1, "b": 0xFFFFFFFF, "c": 1, "d": 0}
    self.assertEqual(
      api.evaluate_payload_case("add_tree", inputs),
      {"reference": [1], "candidate": [1]},
    )
    reduction = {"x0": 1, "x1": 0xFFFFFFFF, "x2": 1, "x3": 0}
    self.assertEqual(
      api.evaluate_payload_case("sum_reduction", reduction),
      {"reference": [1], "candidate": [1]},
    )
    matrix = {"x0": 1, "x1": 0, "x2": 0xFFFFFFFF, "x3": 1}
    self.assertEqual(
      api.evaluate_payload_case("transpose_copy", matrix),
      {"reference": [1, 0xFFFFFFFF, 0, 1], "candidate": [1, 0xFFFFFFFF, 0, 1]},
    )

  def test_runtime_synthesis_records_models_and_exact_input_words(self):
    witness = self.fpsan().synthesize_payload_case("add_tree")
    self.assertEqual(set(witness["input_payloads"]), {"a", "b", "c", "d"})
    self.assertEqual(len(witness["input_float_bits"]), 4)
    self.assertEqual(witness["reference_payloads"], witness["candidate_payloads"])
    self.assertEqual(
      witness["input_float_bits"],
      [
        f"{unembed_known_payload(witness['input_payloads'][name]):08x}"
        for name in ("a", "b", "c", "d")
      ],
    )

  def test_synthesized_mutants_have_machine_checkable_counterexamples(self):
    api = self.fpsan()
    for name in ("operand_routing_fault", "lost_update_fault"):
      witness = api.synthesize_payload_case(name)
      self.assertNotEqual(witness["reference_payloads"], witness["candidate_payloads"])
      self.assertEqual(len(witness["input_float_bits"]), len(witness["input_payloads"]))

  def test_bitvector_solver_proves_named_kernel_equations(self):
    proofs = self.fpsan().prove_payload_kernel_equations()
    self.assertGreaterEqual(len(proofs), 7)
    self.assertTrue(all(item["status"] == "unsat" for item in proofs))

  def test_instrumentation_sensitivity_accepts_a_detected_difference(self):
    result = self.fpsan().validate_instrumentation_sensitivity(
      [0x2AAEB900], [0x40000000], [0x40000000]
    )
    self.assertEqual(result["status"], "pass")
    self.assertEqual(result["mismatch_lanes"], [0])

  def test_instrumentation_sensitivity_rejects_an_inactive_mode(self):
    with self.assertRaisesRegex(AssertionError, "matched ordinary IEEE"):
      self.fpsan().validate_instrumentation_sensitivity(
        [0x40000000], [0x40000000], [0x40000000]
      )

  def test_instrumentation_sensitivity_checks_the_disabled_reference(self):
    with self.assertRaisesRegex(AssertionError, "disabled-mode control"):
      self.fpsan().validate_instrumentation_sensitivity(
        [0x2AAEB900], [0x3F800000], [0x40000000]
      )

  def test_instrumentation_sensitivity_rejects_empty_outputs(self):
    with self.assertRaisesRegex(ValueError, "non-empty"):
      self.fpsan().validate_instrumentation_sensitivity([], [], [])

  def test_instrumentation_sensitivity_validates_raw_word_width(self):
    with self.assertRaisesRegex(ValueError, "unsigned 32-bit"):
      self.fpsan().validate_instrumentation_sensitivity(
        [0x1_0000_0000], [0x40000000], [0x40000000]
      )

  def test_payload_lowering_accepts_integer_reduction_without_float_arithmetic(self):
    result = self.fpsan().validate_fpsan_payload_lowering(
      "call i32 @llvm.vector.reduce.add.v2i32(<2 x i32> %payload)"
    )
    self.assertEqual(result["status"], "pass")
    self.assertEqual(len(result["integer_operation_lines"]), 1)

  def test_payload_lowering_rejects_floating_add_even_with_integer_add(self):
    with self.assertRaisesRegex(AssertionError, "floating-point arithmetic"):
      self.fpsan().validate_fpsan_payload_lowering(
        "%sum = add i32 %a, %b\n%float = fadd float %x, %y"
      )

  def test_payload_lowering_rejects_ir_without_integer_addition(self):
    with self.assertRaisesRegex(AssertionError, "integer payload addition"):
      self.fpsan().validate_fpsan_payload_lowering("ret void")

  def test_payload_lowering_rejects_empty_ir(self):
    with self.assertRaisesRegex(ValueError, "non-empty LLVM IR"):
      self.fpsan().validate_fpsan_payload_lowering("")

  def test_payload_lowering_rejects_floating_multiply(self):
    with self.assertRaisesRegex(AssertionError, "floating-point arithmetic"):
      self.fpsan().validate_fpsan_payload_lowering(
        "%sum = call i32 @llvm.vector.reduce.add.v2i32(%payload)\n"
        "%product = fmul float %x, %y"
      )

  def test_source_identity_accepts_the_pinned_revision_and_hashes(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      source = root / "lib/FpSan.cpp"
      source.parent.mkdir(parents=True)
      source.write_bytes(b"pinned implementation")
      digest = hashlib.sha256(source.read_bytes()).hexdigest()
      identity = self.fpsan().verify_fpsan_source_identity(
        root,
        "a" * 40,
        expected_revision="a" * 40,
        expected_sources={"lib/FpSan.cpp": digest},
      )
    self.assertEqual(identity["git_commit"], "a" * 40)
    self.assertEqual(identity["source_sha256"], {"lib/FpSan.cpp": digest})

  def test_source_identity_rejects_a_different_git_revision(self):
    with tempfile.TemporaryDirectory() as temporary:
      with self.assertRaisesRegex(RuntimeError, "pinned Triton revision"):
        self.fpsan().verify_fpsan_source_identity(
          Path(temporary),
          "b" * 40,
          expected_revision="a" * 40,
          expected_sources={},
        )

  def test_source_identity_rejects_changed_instrumentation_source(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      source = root / "lib/FpSan.cpp"
      source.parent.mkdir(parents=True)
      source.write_bytes(b"modified implementation")
      with self.assertRaisesRegex(RuntimeError, "source hash mismatch"):
        self.fpsan().verify_fpsan_source_identity(
          root,
          "a" * 40,
          expected_revision="a" * 40,
          expected_sources={"lib/FpSan.cpp": hashlib.sha256(b"pinned").hexdigest()},
        )

  def test_source_identity_rejects_missing_instrumentation_source(self):
    with tempfile.TemporaryDirectory() as temporary:
      with self.assertRaisesRegex(RuntimeError, "source file is missing"):
        self.fpsan().verify_fpsan_source_identity(
          Path(temporary),
          "a" * 40,
          expected_revision="a" * 40,
          expected_sources={"lib/FpSan.cpp": "0" * 64},
        )

  def test_source_identity_requires_a_nonempty_source_manifest(self):
    with tempfile.TemporaryDirectory() as temporary:
      with self.assertRaisesRegex(ValueError, "source manifest"):
        self.fpsan().verify_fpsan_source_identity(
          Path(temporary),
          "a" * 40,
          expected_revision="a" * 40,
          expected_sources={},
        )

  def test_source_identity_rejects_manifest_paths_outside_checkout(self):
    with tempfile.TemporaryDirectory() as temporary:
      with self.assertRaisesRegex(ValueError, "stay inside the source root"):
        self.fpsan().verify_fpsan_source_identity(
          Path(temporary),
          "a" * 40,
          expected_revision="a" * 40,
          expected_sources={"../outside.cpp": "0" * 64},
        )

  def test_installed_source_verification_records_the_checkout_revision(self):
    api = self.fpsan()
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      (root / ".git").mkdir()
      module_path = root / "python/triton/__init__.py"
      module_path.parent.mkdir(parents=True)
      module_path.write_text("", encoding="utf-8")
      module = SimpleNamespace(__file__=str(module_path))
      with mock.patch.object(
        api.subprocess,
        "run",
        return_value=CompletedProcess(["git"], 0, "a" * 40 + "\n", ""),
      ) as run:
        with mock.patch.object(
          api,
          "verify_fpsan_source_identity",
          return_value={"git_commit": "a" * 40, "source_sha256": {}},
        ):
          identity = api.verify_installed_fpsan_source(module)
    self.assertEqual(identity["git_commit"], "a" * 40)
    self.assertEqual(Path(identity["source_root"]), root.resolve())
    self.assertEqual(
      run.call_args.args[0], ["git", "-C", str(root.resolve()), "rev-parse", "HEAD"]
    )


if __name__ == "__main__":
  unittest.main()
