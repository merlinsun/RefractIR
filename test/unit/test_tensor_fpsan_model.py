"""Host-side contracts for the pinned FpSan payload subset."""

import ast
import hashlib
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from types import SimpleNamespace
from unittest import mock

from test.lib.tensor_fpsan import (
  add32,
  compare_words,
  embed_known_f32,
  f32_fmod_bits,
  flatten_word_values,
  prove_payload_laws,
  reduce_tree,
  rounding_reassociation_bits,
  sub32,
  transpose_2x2,
  unembed_known_payload,
)


def fpsan_generated_region_source(iterations=2):
  path = Path(__file__).parents[1] / "tensor/fpsan/generated_region.sir"
  source = path.read_text(encoding="utf-8")
  return source.replace(
    "br %i < 2, ^body, ^exit;", f"br %i < {iterations}, ^body, ^exit;"
  )


SINGLE_LANE_REGION_SOURCE = """fun @kernel(%x: tensor [1, 1] f32) : tensor [1, 1] f32 {
  let mut %i: i32 = undef;
  let mut %acc_x: tensor [1, 1] f32 = undef;
  let mut %acc_y: tensor [1, 1] f32 = {0.0};
  let mut %value: tensor [1, 1] f32 = undef;
  let mut %next_x: tensor [1, 1] f32 = undef;
  let mut %next_y: tensor [1, 1] f32 = undef;
  let mut %out: tensor [1, 1] f32 = undef;
^entry:
  %acc_x = %x;
  %i = 0;
  br ^header;
^header:
  br %i < 1, ^body, ^exit;
^body:
  %value = tensor add %acc_x, %acc_y;
  %next_x = tensor add %acc_x, %value;
  %next_y = tensor add %acc_y, %value;
  %acc_x = %next_x;
  %acc_y = %next_y;
  %i = %i + 1;
  br ^header;
^exit:
  %out = tensor add %acc_x, %acc_y;
  ret %out;
}
"""


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

  def test_raw_word_flattening_preserves_all_matrix_elements(self):
    self.assertEqual(flatten_word_values([[1, 2], [3, 4]]), [1, 2, 3, 4])

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

  def test_generated_region_payload_evaluator_executes_loop_and_reductions(self):
    result = self.fpsan().evaluate_generated_payload_region(
      fpsan_generated_region_source(iterations=2),
      [1, 0xFFFFFFFF, 0xFFFFFFFF, 1],
    )
    self.assertEqual(result, [1, 0xFFFFFFFF, 0xFFFFFFFF, 1])

  def test_generated_payload_evaluator_normalizes_negative_reduction_axis(self):
    source = fpsan_generated_region_source(iterations=1).replace(
      "tensor sum %t0, 1, keep", "tensor sum %t0, -1, keep"
    )
    api = self.fpsan()
    result = api.evaluate_generated_payload_region(
      source, [1, 0xFFFFFFFF, 0xFFFFFFFF, 1]
    )
    kernel = api.render_generated_region_triton_kernel(source)
    self.assertEqual(result, [1, 0xFFFFFFFF, 0xFFFFFFFF, 1])
    self.assertIn("tl.sum(value_0, axis=1)", kernel)

  def test_generated_region_query_encodes_tensor_graph_as_qf_bv(self):
    query = self.fpsan().build_generated_payload_query(
      fpsan_generated_region_source(iterations=2)
    )
    self.assertEqual(query["shape"], (2, 2))
    self.assertEqual(query["iterations"], 2)
    self.assertEqual(query["variables"], ("x0", "x1", "x2", "x3"))
    self.assertEqual(
      query["output_names"], ("output_0", "output_1", "output_2", "output_3")
    )
    self.assertIn("(set-logic QF_BV)", query["script"])
    self.assertIn("(bvadd", query["script"])
    self.assertNotIn("FloatingPoint", query["script"])

  def test_generated_region_query_constrains_output_to_exact_integer_f32_range(
    self,
  ):
    query = self.fpsan().build_generated_payload_query(
      fpsan_generated_region_source(iterations=1)
    )
    for output_name in query["output_names"]:
      self.assertIn(f"(assert (bvule {output_name} #x01000000))", query["script"])

  def test_generated_region_query_bounds_transport_and_requires_mutant_sensitivity(
    self,
  ):
    query = self.fpsan().build_generated_payload_query(
      fpsan_generated_region_source(iterations=2)
    )

    self.assertIn("(assert (bvule x0 #x00000008))", query["script"])
    self.assertIn("(assert (bvule output_0 #x01000000))", query["script"])
    self.assertIn("lost_iteration_output_0", query["script"])
    self.assertIn("lost_update_acc_y_output_0", query["script"])
    self.assertNotIn("(= output_0 #xFFFFFFFF)", query["script"])

  def test_generated_region_query_requires_nonuniform_input_payloads(self):
    query = self.fpsan().build_generated_payload_query(
      fpsan_generated_region_source(iterations=2)
    )

    self.assertIn(
      "(assert (or (not (= x0 x1)) (not (= x1 x2))",
      query["script"],
    )

  def test_generated_region_query_requires_lane_rotation_sensitivity(self):
    query = self.fpsan().build_generated_payload_query(
      fpsan_generated_region_source(iterations=2)
    )

    self.assertEqual(
      query["mutant_output_names"]["lane_permutation"],
      ("output_1", "output_2", "output_3", "output_0"),
    )
    self.assertIn(
      "(assert (or (not (= output_0 output_1)) (not (= output_1 output_2))",
      query["script"],
    )

  def test_lane_permutation_mutant_rotates_complete_payload_output(self):
    api = self.fpsan()
    source = fpsan_generated_region_source(iterations=1)
    expected = api.evaluate_generated_payload_region(source, [1, 2, 3, 4])

    actual = api.evaluate_generated_payload_mutant(
      source, [1, 2, 3, 4], "lane_permutation"
    )

    self.assertEqual(actual, expected[1:] + expected[:1])

  def test_generated_region_synthesis_finds_a_lane_sensitive_witness(self):
    witness = self.fpsan().synthesize_generated_region(
      fpsan_generated_region_source(iterations=2), timeout_s=10
    )
    inputs = list(witness["input_payloads"].values())

    self.assertEqual(witness["status"], "SAT")
    self.assertGreater(len(set(inputs)), 1)
    self.assertEqual(
      witness["input_diversity"]["distinct_payload_count"], len(set(inputs))
    )
    self.assertEqual(witness["input_diversity"]["lane_count"], len(inputs))
    self.assertGreater(len(set(witness["output_payloads"])), 1)
    self.assertEqual(
      witness["calibration_mutants"]["lane_permutation"]["status"], "detected"
    )

  def test_generated_region_query_can_require_a_specific_output_payload(self):
    query = self.fpsan().build_generated_payload_query(
      fpsan_generated_region_source(iterations=2),
      target=1268,
      target_index=(0, 0),
    )

    self.assertEqual(
      query["property"],
      {"kind": "output-payload-equals", "target": 1268, "index": [0, 0]},
    )
    self.assertIn("(assert (= output_0 #x000004F4))", query["script"])

  def test_generated_region_query_rejects_target_outside_output_shape(self):
    with self.assertRaisesRegex(ValueError, "outside the output shape"):
      self.fpsan().build_generated_payload_query(
        fpsan_generated_region_source(iterations=2),
        target=1268,
        target_index=(2, 0),
      )

  def test_generated_region_query_rejects_payload_above_bounded_output_range(self):
    with self.assertRaisesRegex(ValueError, "outside the bounded range"):
      self.fpsan().build_generated_payload_query(
        fpsan_generated_region_source(iterations=2),
        target=16_777_217,
      )

  def test_generated_region_query_rejects_a_single_lane_rotation_domain(self):
    with self.assertRaisesRegex(ValueError, "at least two output lanes"):
      self.fpsan().build_generated_payload_query(SINGLE_LANE_REGION_SOURCE)

  def test_generated_payload_transport_preserves_the_pinned_fixed_points(self):
    api = self.fpsan()
    for payload in (0, 1):
      word = api.unembed_generated_payload(payload)
      self.assertEqual(api.embed_generated_float_word(word), payload)
    self.assertEqual(api.unembed_generated_payload(1), 0x3F800000)

  def test_generated_payload_unembedding_uses_the_pinned_bit_mixer(self):
    api = self.fpsan()

    word = api.unembed_generated_payload(2)

    self.assertEqual(word, 0x2AAEB900)
    self.assertNotEqual(word, 0x40000000)
    self.assertEqual(api.embed_generated_float_word(word), 2)

  def test_generated_payload_input_domain_roundtrips_through_f32_words(self):
    api = self.fpsan()

    words = [api.unembed_generated_payload(value) for value in range(9)]

    self.assertEqual(len(set(words)), 9)
    self.assertEqual(
      [api.embed_generated_float_word(word) for word in words], list(range(9))
    )

  def test_generated_payload_transport_rejects_nonfinite_f32_preimages(self):
    with self.assertRaisesRegex(ValueError, "finite f32"):
      self.fpsan().unembed_generated_payload(45)

  def test_generated_payload_query_excludes_nonfinite_transport_words(self):
    query = self.fpsan().build_generated_payload_query(
      fpsan_generated_region_source(iterations=2)
    )

    self.assertIn("(define-fun fpsan_unmix_payload", query["script"])
    self.assertIn("#x7F800000", query["script"])
    self.assertIn("lost_iteration_output_0", query["script"])
    self.assertEqual(
      query["transport_contract"],
      "triton-fpsan-f32-mixer-e2e356723ce35f46b5144373b145c55c4d712237",
    )

  def test_generated_region_synthesis_finds_a_nontrivial_loop_witness(self):
    witness = self.fpsan().synthesize_generated_region(
      fpsan_generated_region_source(iterations=2), timeout_s=10
    )

    self.assertEqual(witness["status"], "SAT")
    inputs = [witness["input_payloads"][f"x{index}"] for index in range(4)]
    self.assertNotEqual(witness["output_payloads"], inputs)
    self.assertTrue(any(value > 1 for value in witness["output_payloads"]))
    self.assertEqual(
      {
        name: mutant["status"]
        for name, mutant in witness["calibration_mutants"].items()
      },
      {
        "lost_iteration": "detected",
        "lost_update_acc_y": "detected",
        "lane_permutation": "detected",
      },
    )

  def test_generated_region_synthesis_satisfies_a_fixed_output_payload(self):
    source_path = Path(__file__).parents[1] / "tensor/gluon/programs/native-seed-7.sir"
    witness = self.fpsan().synthesize_generated_region(
      source_path.read_text(encoding="utf-8"),
      target=1268,
      target_index=(0, 0),
      timeout_s=10,
    )

    self.assertEqual(witness["status"], "SAT")
    self.assertEqual(witness["property"]["target"], 1268)
    self.assertEqual(witness["property"]["index"], [0, 0])
    self.assertEqual(witness["output_payloads"][0], 1268)
    self.assertEqual(
      {
        name: mutant["status"]
        for name, mutant in witness["calibration_mutants"].items()
      },
      {
        "lost_iteration": "detected",
        "lost_update_acc_y": "detected",
        "lane_permutation": "detected",
      },
    )

  @mock.patch("test.lib.tensor_fpsan.subprocess.run")
  @mock.patch("test.lib.tensor_fpsan._solver_command", return_value=["z3", "-in"])
  def test_generated_region_synthesis_preserves_unsat_without_model_access(
    self, _command, run
  ):
    run.return_value = CompletedProcess(["z3", "-in"], 0, "unsat\n", "")

    result = self.fpsan().synthesize_generated_region(
      fpsan_generated_region_source(iterations=2)
    )

    self.assertEqual(result["status"], "UNSAT")
    self.assertEqual(
      result["transport_contract"],
      "triton-fpsan-f32-mixer-e2e356723ce35f46b5144373b145c55c4d712237",
    )
    self.assertEqual(run.call_count, 1)

  @mock.patch("test.lib.tensor_fpsan.subprocess.run")
  @mock.patch("test.lib.tensor_fpsan._solver_command", return_value=["z3", "-in"])
  def test_generated_region_synthesis_retains_unsat_for_a_fixed_property(
    self, _command, run
  ):
    run.return_value = CompletedProcess(["z3", "-in"], 0, "unsat\n", "")

    result = self.fpsan().synthesize_generated_region(
      fpsan_generated_region_source(iterations=2),
      target=16_777_216,
      target_index=(0, 0),
    )

    self.assertEqual(result["status"], "UNSAT")
    self.assertEqual(result["property"]["target"], 16_777_216)
    self.assertEqual(result["property"]["index"], [0, 0])
    self.assertEqual(run.call_count, 1)

  @mock.patch("test.lib.tensor_fpsan.subprocess.run")
  @mock.patch("test.lib.tensor_fpsan._solver_command", return_value=["z3", "-in"])
  def test_generated_region_synthesis_preserves_unknown(self, _command, run):
    run.return_value = CompletedProcess(["z3", "-in"], 0, "unknown\n", "")

    result = self.fpsan().synthesize_generated_region(
      fpsan_generated_region_source(iterations=2)
    )

    self.assertEqual(result["status"], "UNKNOWN")
    self.assertEqual(run.call_count, 1)

  @mock.patch(
    "test.lib.tensor_fpsan.subprocess.run", side_effect=TimeoutExpired(["z3"], 1)
  )
  @mock.patch("test.lib.tensor_fpsan._solver_command", return_value=["z3", "-in"])
  def test_generated_region_synthesis_preserves_timeout(self, _command, _run):
    result = self.fpsan().synthesize_generated_region(
      fpsan_generated_region_source(iterations=2), timeout_s=1
    )

    self.assertEqual(result["status"], "TIMEOUT")
    self.assertIsInstance(json.dumps(result), str)

  @mock.patch("test.lib.tensor_fpsan.subprocess.run")
  @mock.patch("test.lib.tensor_fpsan._solver_command", return_value=["bitwuzla"])
  def test_generated_region_synthesis_cross_checks_model_with_host_execution(
    self, _command, run
  ):
    words = {
      "x0": 1,
      "x1": 1,
      "x2": 2,
      "x3": 3,
      "output_0": 25,
      "output_1": 25,
      "output_2": 62,
      "output_3": 63,
    }
    model = (
      "sat\n("
      + " ".join(f"({name} #x{value:08X})" for name, value in words.items())
      + ")\n"
    )
    run.side_effect = [
      CompletedProcess(["bitwuzla"], 0, "sat\n", ""),
      CompletedProcess(["bitwuzla"], 0, model, ""),
    ]

    witness = self.fpsan().synthesize_generated_region(
      fpsan_generated_region_source(iterations=2)
    )

    self.assertEqual(
      witness["input_payloads"],
      {"x0": 1, "x1": 1, "x2": 2, "x3": 3},
    )
    self.assertEqual(witness["status"], "SAT")
    self.assertEqual(witness["output_payloads"], [25, 25, 62, 63])
    self.assertEqual(
      witness["output_float_bits"],
      ["4be15800", "4be15800", "7244a300", "5d735c00"],
    )
    self.assertEqual(run.call_count, 2)
    self.assertEqual(len(witness["query_sha256"]), 64)

  @mock.patch("test.lib.tensor_fpsan.subprocess.run")
  @mock.patch("test.lib.tensor_fpsan._solver_command", return_value=["bitwuzla"])
  def test_generated_region_synthesis_rejects_inconsistent_solver_output(
    self, _command, run
  ):
    words = {
      "x0": 1,
      "x1": 0xFFFFFFFF,
      "x2": 0xFFFFFFFF,
      "x3": 1,
      "output_0": 0,
      "output_1": 0,
      "output_2": 0,
      "output_3": 0,
    }
    model = (
      "sat\n("
      + " ".join(f"({name} #x{value:08X})" for name, value in words.items())
      + ")\n"
    )
    run.return_value = CompletedProcess(["bitwuzla"], 0, model, "")

    with self.assertRaisesRegex(AssertionError, "host payload evaluator"):
      self.fpsan().synthesize_generated_region(
        fpsan_generated_region_source(iterations=2)
      )

  def test_generated_region_triton_kernel_emits_compilable_loop_and_reductions(self):
    source = self.fpsan().render_generated_region_triton_kernel(
      fpsan_generated_region_source(iterations=2)
    )
    ast.parse(source)
    self.assertIn("def fpsan_generated_region(input_ptr, output_ptr):", source)
    self.assertIn("tl.static_range(2)", source)
    self.assertIn("tl.sum(", source)
    self.assertIn("tl.broadcast_to(", source)
    self.assertIn("tl.store(output_ptr + offsets, output)", source)

  def test_generated_region_triton_kernel_emits_lane_rotation(self):
    source = self.fpsan().render_generated_region_triton_kernel(
      fpsan_generated_region_source(iterations=2), lane_permutation=True
    )

    ast.parse(source)
    self.assertIn("tl.gather(flat_output, permuted_lanes, axis=0)", source)
    self.assertIn("permuted_lanes = (lanes + 1) % 4", source)

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

  def test_payload_lowering_accepts_dynamic_integer_multiplication(self):
    result = self.fpsan().validate_fpsan_payload_lowering(
      "%product = mul i32 %left, %right",
      required_payload_operations=("mul",),
    )

    self.assertEqual(result["status"], "pass")
    self.assertEqual(result["payload_operation_counts"], {"mul": 1})

  def test_payload_lowering_recognizes_ttgir_addition_and_subtraction(self):
    result = self.fpsan().validate_fpsan_payload_lowering(
      '#loc1 = loc("generated_region_kernel.py":10:1)\n'
      '#loc2 = loc("generated_region_kernel.py":11:1)\n'
      "%sum = arith.addi %left, %right : tensor<16xi32> loc(#loc1)\n"
      "%difference = arith.subi %sum, %left : tensor<16xi32> loc(#loc2)",
      stage="ttgir",
      required_payload_operations=("add", "sub"),
      payload_source_lines={10: ("add",), 11: ("sub",)},
    )

    self.assertEqual(result["ir_stage"], "ttgir")
    self.assertEqual(result["payload_operation_counts"], {"add": 1, "sub": 1})

  def test_payload_lowering_recognizes_ttgir_multiplication(self):
    result = self.fpsan().validate_fpsan_payload_lowering(
      '#loc1 = loc("generated_region_kernel.py":12:1)\n'
      "%product = arith.muli %left, %right : tensor<16xi32> loc(#loc1)",
      stage="ttgir",
      required_payload_operations=("mul",),
      payload_source_lines={12: ("mul",)},
    )

    self.assertEqual(result["payload_operation_counts"], {"mul": 1})

  def test_payload_lowering_does_not_certify_address_multiplication(self):
    ir = (
      '#loc8 = loc("generated_region_kernel.py":8:15)\n'
      "%offsets_2 = arith.muli %offsets_1, %cst : tensor<2x1xi32> loc(#loc8)"
    )

    with self.assertRaisesRegex(AssertionError, "integer payload multiplication"):
      self.fpsan().validate_fpsan_payload_lowering(
        ir,
        stage="ttgir",
        required_payload_operations=("mul",),
        payload_source_lines={12: ("mul",)},
      )

  def test_payload_lowering_attributes_only_generated_payload_source_lines(self):
    ir = (
      '#loc8 = loc("generated_region_kernel.py":8:15)\n'
      '#loc12 = loc("generated_region_kernel.py":12:19)\n'
      "%offsets_2 = arith.muli %offsets_1, %cst : tensor<2x1xi32> loc(#loc8)\n"
      "%value_0 = arith.addi %left, %right : tensor<2x2xi32> loc(#loc12)"
    )

    result = self.fpsan().validate_fpsan_payload_lowering(
      ir,
      stage="ttgir",
      required_payload_operations=("add",),
      payload_source_lines={12: ("add",)},
    )

    self.assertEqual(result["payload_operation_counts"], {"add": 1})
    self.assertEqual(len(result["non_payload_integer_operation_lines"]), 1)
    self.assertIn("arith.muli", result["non_payload_integer_operation_lines"][0])

  def test_payload_lowering_ignores_unlocated_ttgir_integer_arithmetic(self):
    with self.assertRaisesRegex(AssertionError, "integer payload addition"):
      self.fpsan().validate_fpsan_payload_lowering(
        "%sum = arith.addi %left, %right : tensor<16xi32>",
        stage="ttgir",
        required_payload_operations=("add",),
        payload_source_lines={12: ("add",)},
      )

  def test_payload_lowering_resolves_reduction_callsite_and_source_path(self):
    ir = (
      "#loc1 = loc(unknown)\n"
      '#loc2 = loc("/build/generated_region_kernel.py":13:27)\n'
      '#loc3 = loc("value_1_reduced"(#loc2))\n'
      "#loc4 = loc(callsite(#loc1 at #loc3))\n"
      '#loc5 = loc("/build/triton_standard.py":13:27)\n'
      "%payload_sum = arith.addi %left, %right : i32 loc(#loc4)\n"
      "%unrelated_sum = arith.addi %a, %b : i32 loc(#loc5)"
    )

    result = self.fpsan().validate_fpsan_payload_lowering(
      ir,
      stage="ttgir",
      required_payload_operations=("add",),
      payload_source_lines={13: ("add",)},
      payload_source_path="/build/generated_region_kernel.py",
    )

    self.assertEqual(result["payload_operation_counts"], {"add": 1})
    self.assertEqual(len(result["non_payload_integer_operation_lines"]), 1)

  def test_payload_lowering_rejects_float_arithmetic_in_ttgir(self):
    with self.assertRaisesRegex(AssertionError, "floating-point arithmetic"):
      self.fpsan().validate_fpsan_payload_lowering(
        '#loc1 = loc("generated_region_kernel.py":10:1)\n'
        '#loc2 = loc("generated_region_kernel.py":11:1)\n'
        "%sum = arith.addi %left, %right : tensor<16xi32> loc(#loc1)\n"
        "%difference = arith.subf %sum, %left : tensor<16xf32> loc(#loc2)",
        stage="ttgir",
        required_payload_operations=("add",),
        payload_source_lines={10: ("add",)},
      )

  def test_payload_lowering_allows_llvm_optimization_to_remove_subtraction(self):
    result = self.fpsan().validate_fpsan_payload_lowering(
      "%sum = add <4 x i32> %left, %right",
      stage="llvm",
      required_payload_operations=(),
    )

    self.assertEqual(result["ir_stage"], "llvm")
    self.assertNotIn("sub", result["payload_operation_counts"])

  def test_payload_lowering_requires_dynamic_integer_multiplication(self):
    with self.assertRaisesRegex(AssertionError, "integer payload multiplication"):
      self.fpsan().validate_fpsan_payload_lowering(
        "%mixed = mul i32 %payload, 14940041",
        required_payload_operations=("mul",),
      )

  def test_payload_lowering_requires_each_declared_operation_family(self):
    with self.assertRaisesRegex(AssertionError, "integer payload addition"):
      self.fpsan().validate_fpsan_payload_lowering(
        "%product = mul i32 %left, %right",
        required_payload_operations=("add", "mul"),
      )

  def test_payload_lowering_allows_an_operation_free_zero_iteration_mutant(self):
    result = self.fpsan().validate_fpsan_payload_lowering(
      "%mixed = mul i32 %payload, 14940041",
      required_payload_operations=(),
    )

    self.assertEqual(result["status"], "pass")
    self.assertEqual(result["payload_operation_counts"], {})

  def test_payload_lowering_still_rejects_float_arithmetic_without_payload_ops(self):
    with self.assertRaisesRegex(AssertionError, "floating-point arithmetic"):
      self.fpsan().validate_fpsan_payload_lowering(
        "%value = fmul float %left, %right",
        required_payload_operations=(),
      )

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
