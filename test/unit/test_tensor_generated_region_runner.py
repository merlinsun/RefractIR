"""Generated Gluon compiler summaries must classify counted loops."""

import hashlib
import importlib.util
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "test/tensor/gluon/run_generated_region.py"
MULTIPLY_REGION_SOURCE = (
  ROOT / "test/tensor/gluon/programs/elementwise-multiply-region-v1.sir"
).read_text(encoding="utf-8")
SUBTRACTION_REGION_SOURCE = (
  ROOT / "test/tensor/gluon/programs/elementwise-subtraction-region-v1.sir"
).read_text(encoding="utf-8")
STATE_ONLY_SUBTRACTION_SOURCE = """fun @kernel(%x: tensor [2, 2] f32) : tensor [2, 2] f32 {
  let mut %i: i32 = undef;
  let mut %acc_x: tensor [2, 2] f32 = undef;
  let mut %acc_y: tensor [2, 2] f32 = {0.0, 0.0, 0.0, 0.0};
  let mut %out: tensor [2, 2] f32 = undef;
  let mut %twice: tensor [2, 2] f32 = undef;
  let mut %squares: tensor [2, 2] f32 = undef;
  let mut %row_sum: tensor [2, 1] f32 = undef;
  let mut %row_broadcast: tensor [2, 2] f32 = undef;
  let mut %next_x: tensor [2, 2] f32 = undef;
  let mut %next_y: tensor [2, 2] f32 = undef;
^entry:
  %acc_x = %x;
  %i = 0;
  br ^header;
^header:
  br %i < 1, ^body, ^exit;
^body:
  %twice = tensor add %acc_x, %acc_x;
  %squares = tensor sub %twice, %acc_x;
  %row_sum = tensor sum %squares, 1, keep;
  %row_broadcast = tensor broadcast %row_sum, [2, 2];
  %next_x = tensor add %acc_x, %acc_y;
  %next_y = tensor add %acc_y, %row_broadcast;
  %acc_x = %next_x;
  %acc_y = %next_y;
  %i = %i + 1;
  br ^header;
^exit:
  %out = tensor add %acc_x, %acc_y;
  ret %out;
}
"""


def _load_runner():
  specification = importlib.util.spec_from_file_location(
    "test_tensor_generated_region_runner_target", RUNNER_PATH
  )
  if specification is None or specification.loader is None:
    raise RuntimeError("cannot load generated-region Gluon runner")
  module = importlib.util.module_from_spec(specification)
  with mock.patch.dict(
    sys.modules,
    {"torch": SimpleNamespace(), "triton": SimpleNamespace()},
  ):
    specification.loader.exec_module(module)
  return module


def _load_fpsan_experiment():
  specification = importlib.util.spec_from_file_location(
    "test_tensor_fpsan_experiment_target",
    ROOT / "test/tensor/fpsan/run_experiment.py",
  )
  if specification is None or specification.loader is None:
    raise RuntimeError("cannot load FpSan experiment runner")
  module = importlib.util.module_from_spec(specification)
  torch = ModuleType("torch")
  triton = ModuleType("triton")
  triton.__path__ = []
  triton.jit = lambda function: function
  triton_language = ModuleType("triton.language")
  with mock.patch.dict(
    sys.modules,
    {"torch": torch, "triton": triton, "triton.language": triton_language},
  ):
    specification.loader.exec_module(module)
  return module


class GeneratedRegionRunnerTests(unittest.TestCase):
  def test_region_parser_admits_elementwise_subtraction(self):
    from test.lib import tensor_exact_f32

    _shape, iterations, _types, nodes, updates = (
      tensor_exact_f32._generated_region_program(SUBTRACTION_REGION_SOURCE)
    )

    self.assertEqual(iterations, 1)
    self.assertEqual(nodes[0], ("twice", "add", ("acc_x", "acc_x"), None))
    self.assertEqual(nodes[1], ("difference", "sub", ("twice", "acc_x"), None))
    self.assertEqual(updates, [("acc_x", "next_x"), ("acc_y", "next_y")])

  def test_exact_certificate_bounds_subtraction_and_complete_output(self):
    from test.lib import tensor_exact_f32

    values = tuple(index % 9 for index in range(256))
    proof, output_bytes = tensor_exact_f32.certify(
      SUBTRACTION_REGION_SOURCE, struct.pack("<256f", *values), (0, 8)
    )

    self.assertEqual(
      struct.unpack("<256f", output_bytes), tuple(3.0 * value for value in values)
    )
    difference_bound = next(
      item for item in proof["bounds"] if item["destination"] == "%difference"
    )
    self.assertEqual(difference_bound["min_value"], 0)
    self.assertEqual(difference_bound["max_value"], 8)
    self.assertIn("sub", proof["allowed_operations"])

  def test_payload_evaluator_and_query_model_tensor_subtraction(self):
    from test.lib import tensor_fpsan

    payloads = tuple(index % 9 for index in range(256))
    query = tensor_fpsan.build_generated_payload_query(SUBTRACTION_REGION_SOURCE)
    output = tensor_fpsan.evaluate_generated_payload_region(
      SUBTRACTION_REGION_SOURCE, payloads
    )

    self.assertIn("(bvsub g_v0_n0_e0 x0)", query["script"])
    self.assertIn("(assert (bvuge g_v0_n0_e0 x0))", query["script"])
    self.assertEqual(query["script"].count("(assert (bvuge "), 512)
    self.assertEqual(output, [3 * value for value in payloads])

  def test_gluon_and_fpsan_lowerers_emit_subtraction(self):
    from test.lib.tensor_fpsan import (
      render_generated_region_triton_kernel,
      validate_fpsan_payload_lowering,
    )
    from test.lib.tensor_gluon_lowering import render_gluon_kernel

    gluon_source = render_gluon_kernel(SUBTRACTION_REGION_SOURCE).source
    fpsan_source = render_generated_region_triton_kernel(SUBTRACTION_REGION_SOURCE)
    validation = validate_fpsan_payload_lowering(
      "define i32 @kernel(i32 %left, i32 %right) {\n"
      "  %difference = sub i32 %left, %right\n"
      "  ret i32 %difference\n}\n",
      required_payload_operations=("sub",),
    )
    experiment = _load_fpsan_experiment()

    self.assertIn("v1 = v1_left - v1_right", gluon_source)
    self.assertIn("value_0 = acc_x + acc_x", fpsan_source)
    self.assertIn("value_1 = value_0 - acc_x", fpsan_source)
    self.assertIn("sub", validation["required_payload_operations"])
    self.assertEqual(
      experiment._required_fpsan_payload_operations(SUBTRACTION_REGION_SOURCE, 1),
      ("add", "sub"),
    )

  def test_fpsan_payload_contract_checks_ttgir_and_final_llvm(self):
    experiment = _load_fpsan_experiment()
    compiled = SimpleNamespace(
      asm={
        "ttgir": (
          '#loc1 = loc("generated_region_kernel.py":10:1)\n'
          '#loc2 = loc("generated_region_kernel.py":11:1)\n'
          "%twice = arith.addi %value, %value : tensor<16xi32> loc(#loc1)\n"
          "%difference = arith.subi %twice, %value : tensor<16xi32> loc(#loc2)"
        ),
        "llir": "%sum = add <4 x i32> %left, %right",
      }
    )

    result = experiment._validate_generated_region_payload_lowering(
      compiled,
      required_payload_operations=("add", "sub"),
      payload_source_lines={10: ("add",), 11: ("sub",)},
    )

    self.assertEqual(result["ttgir"]["payload_operation_counts"], {"add": 1, "sub": 1})
    self.assertEqual(result["llir"]["floating_operation_lines"], [])

  def test_fpsan_case_checkpoint_preserves_completed_work_before_later_failure(self):
    experiment = _load_fpsan_experiment()
    measurement = {"warmup_launches": 3, "measured_launches": 30}
    compiler = {"files": {"ttgir": "compiler.ttgir"}}
    output_record = {
      "status": "output_captured",
      "actual_words": [1, 2, 3, 4],
      "expected_words": [4, 3, 2, 1],
      "measurement": measurement,
      "compiler_ir": compiler,
    }

    def fail_during_second_control(
      _case_directory,
      *,
      source_path,
      target,
      target_index,
      timeout_s,
      checkpoint,
    ):
      del source_path, target, target_index, timeout_s
      for name in ("fpsan", "ordinary"):
        checkpoint(
          f"{name}_running",
          work_variant_updates={name: {"status": "running"}},
        )
        checkpoint(
          f"{name}_complete",
          work_variant_updates={
            name: {
              "status": "complete",
              "compiler": compiler,
              "execution_observed": True,
              "measurement": measurement,
            }
          },
        )
      checkpoint(
        "lost_iteration_running",
        work_variant_updates={"lost_iteration": {"status": "running"}},
      )
      checkpoint(
        "lost_iteration_captured",
        work_variant_updates={
          "lost_iteration": {
            "status": "complete",
            "compiler": compiler,
            "execution_observed": True,
            "measurement": measurement,
          }
        },
        calibration_mutant_update={
          "name": "lost_iteration",
          "record": output_record,
        },
      )
      checkpoint(
        "lost_update_running",
        work_variant_updates={"lost_update_acc_y": {"status": "running"}},
      )
      raise RuntimeError("synthetic later-control failure")

    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      source_path = root / "source.sir"
      source_path.write_text("fixture", encoding="utf-8")
      entry = {}
      report = {"generated_regions": {"programs": {"case": entry}}}
      with mock.patch.object(
        experiment,
        "_run_generated_region_case",
        side_effect=fail_during_second_control,
      ):
        result = experiment._run_generated_region_case_record(
          report,
          root / "report.json",
          root / "case",
          entry,
          source_path=source_path,
          target=1268,
          target_index=(0, 0),
          timeout_s=1.0,
        )

    accounting = result["measurement_accounting"]
    self.assertEqual(result["status"], "ERROR")
    self.assertEqual(accounting["status"], "partial")
    self.assertEqual(
      accounting["variant_counts"], {"useful": 2, "calibration": 1, "total": 3}
    )
    self.assertEqual(accounting["observed"]["total_compilations"], 3)
    self.assertEqual(accounting["observed"]["total_executions"], 3)
    self.assertEqual(accounting["observed"]["total_warmup_launches"], 9)
    self.assertEqual(accounting["observed"]["total_measured_launches"], 90)
    self.assertIsNone(accounting["total"]["total_compilations"])
    self.assertEqual(
      result["work_variants"]["lost_iteration"]["calibration_result"]["actual_words"],
      [1, 2, 3, 4],
    )
    self.assertEqual(
      result["calibration_mutants"]["lost_iteration"]["expected_words"],
      [4, 3, 2, 1],
    )
    self.assertEqual(result["work_variants"]["lost_update_acc_y"]["status"], "running")

  def test_ordinary_reference_admits_binary32_subtraction_region(self):
    experiment = _load_fpsan_experiment()
    from test.lib.tensor_fp32_reference import sub_f32_rne_bits

    values = tuple(index % 9 for index in range(256))
    input_bytes = struct.pack("<256f", *values)
    admission = experiment._ordinary_generated_region_admission(
      SUBTRACTION_REGION_SOURCE, input_bytes
    )
    actual = experiment._ordinary_generated_region_output_words(
      SUBTRACTION_REGION_SOURCE, input_bytes
    )
    self.assertEqual(sub_f32_rne_bits(0x3F800000, 0x3F800000), 0)

    self.assertEqual(admission["status"], "admitted")
    self.assertIn("sub", admission["operations"])
    self.assertEqual(
      actual,
      [struct.unpack("<I", struct.pack("<f", 3.0 * value))[0] for value in values],
    )

  def test_subtraction_rejects_domain_underflow_and_preserves_signed_zero(self):
    experiment = _load_fpsan_experiment()
    from test.lib import tensor_exact_f32
    from test.lib.tensor_fp32_reference import sub_f32_rne_bits

    underflow_source = SUBTRACTION_REGION_SOURCE.replace(
      "tensor sub %twice, %acc_x", "tensor sub %acc_x, %twice"
    )
    zero_bytes = struct.pack("<256f", *([0.0] * 256))
    with self.assertRaisesRegex(ValueError, "negative exact value"):
      tensor_exact_f32.certify(underflow_source, zero_bytes, (0, 8))

    positive_bytes = struct.pack("<256f", *([1.0] * 256))
    admission = experiment._ordinary_generated_region_admission(
      underflow_source, positive_bytes
    )

    self.assertEqual(admission["status"], "unclassified")
    self.assertIn("negative value", admission["reason"])
    self.assertEqual(sub_f32_rne_bits(0x80000000, 0), 0x80000000)
    self.assertEqual(sub_f32_rne_bits(0x3F800000, 0x3F800000), 0)
    self.assertEqual(sub_f32_rne_bits(0x3F800000, 0x33000000), 0x3F800000)
    self.assertEqual(sub_f32_rne_bits(0x3F800000, 0x33800000), 0x3F7FFFFF)

  def test_subtraction_graph_uses_a_computed_left_operand(self):
    from test.lib import tensor_exact_f32

    _shape, _iterations, _types, nodes, _updates = (
      tensor_exact_f32._generated_region_program(SUBTRACTION_REGION_SOURCE)
    )

    self.assertEqual(nodes[0], ("twice", "add", ("acc_x", "acc_x"), None))
    self.assertEqual(nodes[1], ("difference", "sub", ("twice", "acc_x"), None))

  def test_exact_certificate_bounds_the_nonzero_subtraction(self):
    from test.lib import tensor_exact_f32

    inputs = tuple(float(index % 9) for index in range(256))
    proof, output = tensor_exact_f32.certify(
      SUBTRACTION_REGION_SOURCE, struct.pack("<256f", *inputs), (0, 8)
    )
    bounds = {item["destination"]: item for item in proof["bounds"]}

    self.assertEqual(bounds["%twice"]["max_value"], 16)
    self.assertEqual(bounds["%difference"]["min_value"], 0)
    self.assertEqual(bounds["%difference"]["max_value"], 8)
    self.assertEqual(
      struct.unpack("<256f", output), tuple(3.0 * value for value in inputs)
    )

  def test_payload_query_subtracts_a_computed_payload(self):
    from test.lib import tensor_fpsan

    query = tensor_fpsan.build_generated_payload_query(SUBTRACTION_REGION_SOURCE)

    self.assertIn("(bvsub g_v0_n0_e0 x0)", query["script"])
    self.assertIn("(assert (bvuge g_v0_n0_e0 x0))", query["script"])

  def test_payload_replay_detects_replacing_active_subtraction_with_addition(self):
    from test.lib import tensor_fpsan

    values = (1,) * 256
    source_with_add = SUBTRACTION_REGION_SOURCE.replace(
      "tensor sub %twice, %acc_x", "tensor add %twice, %acc_x"
    )
    expected = tensor_fpsan.evaluate_generated_payload_region(
      SUBTRACTION_REGION_SOURCE, values
    )
    changed = tensor_fpsan.evaluate_generated_payload_region(source_with_add, values)

    self.assertEqual(expected, [3] * 256)
    self.assertEqual(changed, [5] * 256)

  def test_compiler_sources_retain_the_active_subtraction(self):
    from test.lib.tensor_fpsan import render_generated_region_triton_kernel
    from test.lib.tensor_gluon_lowering import render_gluon_kernel

    fpsan_source = render_generated_region_triton_kernel(SUBTRACTION_REGION_SOURCE)
    gluon_source = render_gluon_kernel(SUBTRACTION_REGION_SOURCE).source

    self.assertIn("value_1 = value_0 - acc_x", fpsan_source)
    self.assertIn("v1 = v1_left - v1_right", gluon_source)

  def test_affine_subtraction_accepts_zero_boundary_and_refuses_reverse(self):
    from test.lib import tensor_exact_f32

    zero_input = struct.pack("<256f", *([0.0] * 256))
    _proof, output = tensor_exact_f32.certify(
      SUBTRACTION_REGION_SOURCE, zero_input, (0, 8)
    )
    reversed_subtraction = SUBTRACTION_REGION_SOURCE.replace(
      "tensor sub %twice, %acc_x", "tensor sub %acc_x, %twice"
    )

    self.assertEqual(output, zero_input)
    with self.assertRaisesRegex(ValueError, "negative exact value"):
      tensor_exact_f32.certify(reversed_subtraction, zero_input, (0, 8))

  def test_orthogonal_reductions_feed_distinct_loop_state_roots(self):
    from test.lib import tensor_exact_f32

    source = (
      ROOT / "test/tensor/gluon/programs/orthogonal-reductions-region-v1.sir"
    ).read_text(encoding="utf-8")
    values = tuple(index % 9 for index in range(64))
    _shape, iterations, _types, nodes, updates = (
      tensor_exact_f32._generated_region_program(source)
    )
    _proof, output_bytes = tensor_exact_f32.certify(
      source, struct.pack("<64f", *values), (0, 8)
    )
    row_totals = [sum(values[row * 8 : (row + 1) * 8]) for row in range(8)]
    column_totals = [
      sum(values[row * 8 + column] for row in range(8)) for column in range(8)
    ]
    expected = tuple(
      values[row * 8 + column] + row_totals[row] + column_totals[column]
      for row in range(8)
      for column in range(8)
    )

    self.assertEqual(iterations, 1)
    self.assertEqual([node[3] for node in nodes if node[1] == "sum"], [1, 0])
    self.assertEqual(updates, [("acc_x", "next_x"), ("acc_y", "next_y")])
    self.assertEqual(struct.unpack("<64f", output_bytes), tuple(map(float, expected)))

  def test_region_parser_admits_elementwise_multiplication(self):
    from test.lib import tensor_exact_f32

    _shape, iterations, _types, nodes, updates = (
      tensor_exact_f32._generated_region_program(MULTIPLY_REGION_SOURCE)
    )

    self.assertEqual(iterations, 1)
    self.assertEqual(nodes[0], ("squares", "mul", ("acc_x", "acc_x"), None))
    self.assertEqual(updates, [("acc_x", "next_x"), ("acc_y", "next_y")])

  def test_exact_certificate_bounds_multiplication_and_full_output(self):
    from test.lib import tensor_exact_f32

    input_bytes = struct.pack("<4f", 1, 2, 3, 4)
    proof, output_bytes = tensor_exact_f32.certify(
      MULTIPLY_REGION_SOURCE, input_bytes, (0, 8)
    )

    self.assertEqual(struct.unpack("<4f", output_bytes), (7, 11, 37, 45))
    square_bound = next(
      item for item in proof["bounds"] if item["destination"] == "%squares"
    )
    self.assertEqual(square_bound["operation"], "mul")
    self.assertEqual(square_bound["max_value"], 64)
    self.assertIn("mul", proof["allowed_operations"])

  def test_payload_evaluator_and_query_model_tensor_multiplication(self):
    from test.lib import tensor_fpsan

    payloads = (1, 2, 3, 4)
    query = tensor_fpsan.build_generated_payload_query(MULTIPLY_REGION_SOURCE)
    output = tensor_fpsan.evaluate_generated_payload_region(
      MULTIPLY_REGION_SOURCE, payloads
    )

    self.assertIn("(bvmul x0 x0)", query["script"])
    self.assertEqual(output, [7, 11, 37, 45])

  def test_gluon_and_fpsan_lowerers_emit_multiplication(self):
    from test.lib.tensor_fpsan import render_generated_region_triton_kernel
    from test.lib.tensor_gluon_lowering import render_gluon_kernel

    gluon_source = render_gluon_kernel(MULTIPLY_REGION_SOURCE).source
    fpsan_source = render_generated_region_triton_kernel(MULTIPLY_REGION_SOURCE)

    self.assertIn("v0 = v0_left * v0_right", gluon_source)
    self.assertIn("value_0 = acc_x * acc_x", fpsan_source)

  def test_fpsan_lowering_requirements_follow_static_body_operations(self):
    experiment = _load_fpsan_experiment()

    self.assertEqual(
      experiment._required_fpsan_payload_operations(MULTIPLY_REGION_SOURCE, 1),
      ("add", "mul"),
    )
    self.assertEqual(
      experiment._required_fpsan_payload_operations(MULTIPLY_REGION_SOURCE, 0), ()
    )

  def test_fpsan_mutant_requirements_follow_live_state_updates(self):
    experiment = _load_fpsan_experiment()

    operations = experiment._required_fpsan_payload_operations(
      STATE_ONLY_SUBTRACTION_SOURCE, 1, omitted_updates=("acc_y",)
    )

    self.assertEqual(operations, ("add",))
    self.assertNotIn("sub", operations)

  def test_lost_update_payload_contract_accepts_legally_eliminated_subtraction(self):
    experiment = _load_fpsan_experiment()
    from test.lib import tensor_fpsan

    live_operations = experiment._required_fpsan_payload_operations(
      STATE_ONLY_SUBTRACTION_SOURCE, 1, omitted_updates=("acc_y",)
    )
    payload_source_lines = tensor_fpsan.generated_region_payload_operation_lines(
      STATE_ONLY_SUBTRACTION_SOURCE, omitted_updates=("acc_y",)
    )
    compiled = SimpleNamespace(
      asm={
        "ttgir": (
          '#loc1 = loc("generated_region_kernel.py":8:15)\n'
          "%offsets = arith.muli %offsets_1, %cst : tensor<2x1xi32> loc(#loc1)"
        ),
        "llir": "ret void",
      }
    )

    result = experiment._validate_generated_region_payload_lowering(
      compiled,
      required_payload_operations=(),
      payload_source_lines=payload_source_lines,
    )
    transition = experiment._payload_operation_transition(
      experiment._required_fpsan_payload_operations(STATE_ONLY_SUBTRACTION_SOURCE, 1),
      live_operations,
      result["ttgir"]["payload_operation_counts"],
    )

    self.assertEqual(live_operations, ("add",))
    self.assertEqual(result["ttgir"]["required_payload_operations"], [])
    self.assertEqual(result["ttgir"]["payload_operation_counts"], {})
    self.assertNotIn("sub", result["ttgir"]["payload_operation_counts"])
    self.assertEqual(transition["eliminated_payload_operations"], ["sub"])
    self.assertEqual(transition["unobserved_live_payload_operations"], ["add"])

  def test_ordinary_reference_uses_admitted_binary32_multiply(self):
    experiment = _load_fpsan_experiment()
    from test.lib.tensor_fp32_reference import add_f32_rne_bits, mul_f32_rne_bits

    words = (0x3F800001, 0x3F800002, 0x40000001, 0x40400001)
    input_bytes = struct.pack("<4I", *words)
    admission = experiment._ordinary_generated_region_admission(
      MULTIPLY_REGION_SOURCE, input_bytes
    )
    squares = [mul_f32_rne_bits(word, word) for word in words]
    row_sums = [
      add_f32_rne_bits(squares[index], squares[index + 1]) for index in (0, 2)
    ]
    expected = []
    for index, word in enumerate(words):
      row_sum = row_sums[index // 2]
      next_x = add_f32_rne_bits(word, row_sum)
      expected.append(add_f32_rne_bits(next_x, squares[index]))

    actual = experiment._ordinary_generated_region_output_words(
      MULTIPLY_REGION_SOURCE, input_bytes
    )

    self.assertEqual(admission["status"], "admitted")
    self.assertIn("mul", admission["operations"])
    self.assertEqual(actual, expected)

  def test_ordinary_admission_excludes_binary32_multiplication_overflow(self):
    experiment = _load_fpsan_experiment()
    input_bytes = struct.pack("<4I", *([0x7F7FFFFF] * 4))

    admission = experiment._ordinary_generated_region_admission(
      MULTIPLY_REGION_SOURCE, input_bytes
    )

    self.assertEqual(admission["status"], "unclassified")
    self.assertIn("overflowed finite f32", admission["reason"])

  def capture(self, ttir, ttgir):
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temporary:
      return runner._capture_region_compiler(
        SimpleNamespace(asm={"ttir": ttir, "ttgir": ttgir}),
        "generated",
        Path(temporary),
        {},
        expected_loop=True,
      )

  def test_capture_reports_a_retained_generated_loop(self):
    ir = "module { scf.for %i = %c0 to %c2 step %c1 { tt.add %a, %b } }"

    result = self.capture(ir, ir)

    self.assertEqual(result["observation"]["loop_status"], "retained")
    self.assertEqual(result["observation"]["stages"]["ttir"]["loop_construct_count"], 1)

  def test_capture_reports_an_eliminated_generated_loop(self):
    ir = "module { %0 = tt.add %a, %b : tensor<4xf32> }"

    result = self.capture(ir, ir)

    self.assertEqual(result["observation"]["loop_status"], "unrolled_or_removed")
    self.assertEqual(result["observation"]["stages"]["ttir"]["loop_construct_count"], 0)

  def test_fpsan_runner_retains_non_sat_generated_outcomes(self):
    experiment = _load_fpsan_experiment()
    for status in ("UNSAT", "UNKNOWN", "TIMEOUT"):
      with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
        outcome = {
          "status": status,
          "query_sha256": "a" * 64,
          "solver_outcome": {"phase": "check", "stdout": status.lower()},
        }
        with mock.patch.object(
          experiment, "synthesize_generated_region", return_value=outcome
        ):
          result = experiment._run_generated_region_case(Path(temporary))

        self.assertEqual(result["status"], status)
        self.assertEqual(result["solver_outcome"]["stdout"], status.lower())

  def test_fpsan_runner_loads_the_hashed_native_corpus(self):
    experiment = _load_fpsan_experiment()

    sources = experiment._generated_region_corpus_sources()

    self.assertEqual(
      [source["program_id"] for source in sources],
      [
        "native-seed-7",
        "native-seed-17",
        "native-seed-29",
        "native-seed-0-shape-2x4-iterations-3",
        "native-seed-3-shape-4x2-iterations-4",
        "native-seed-6-shape-2x4-iterations-2",
        "native-seed-2-shape-8x8-iterations-2",
        "native-subtraction-seed-0-shape-2x4-iterations-3",
        "native-subtraction-seed-3-shape-4x2-iterations-4",
        "native-multiplication-seed-6-shape-2x4-iterations-1",
        "native-mixed-seed-17-shape-2x2-iterations-1",
        "authored-elementwise-multiply-region-v1",
        "authored-elementwise-subtraction-region-v1",
        "authored-orthogonal-reductions-region-v1",
      ],
    )
    for source in sources:
      text = source["source_path"].read_text(encoding="utf-8")
      self.assertEqual(
        hashlib.sha256(text.encode("utf-8")).hexdigest(), source["source_sha256"]
      )

  def test_paired_compiler_summary_reports_observed_mode_differences(self):
    experiment = _load_fpsan_experiment()
    fpsan_ir = {
      "stage_sha256": {"ttir": "fpsan", "ttgir": "same"},
      "layout_conversion_operations_in_ttgir": 2,
      "warp_shuffle_operations_in_ptx": 3,
      "shared_memory_transfers_in_ptx": 0,
    }
    ordinary_ir = {
      "stage_sha256": {"ttir": "ordinary", "ttgir": "same"},
      "layout_conversion_operations_in_ttgir": 1,
      "warp_shuffle_operations_in_ptx": 0,
      "shared_memory_transfers_in_ptx": 0,
    }

    comparison = experiment._compare_generated_compiler_modes(fpsan_ir, ordinary_ir)

    self.assertEqual(comparison["stage_hashes_equal"], {"ttgir": True, "ttir": False})
    self.assertEqual(
      comparison["metric_delta_fpsan_minus_ordinary"],
      {
        "layout_conversion_operations_in_ttgir": 1,
        "warp_shuffle_operations_in_ptx": 3,
        "shared_memory_transfers_in_ptx": 0,
      },
    )
    self.assertIn(
      "do not establish rematerialization", comparison["interpretation_limit"]
    )

  def test_ordinary_reference_matches_the_exact_integer_region_contract(self):
    experiment = _load_fpsan_experiment()
    from test.lib import tensor_exact_f32

    source_path = ROOT / "test/tensor/gluon/programs/native-seed-7.sir"
    source = source_path.read_text(encoding="utf-8")
    input_bytes = struct.pack("<4f", 1.0, 2.0, 4.0, 8.0)
    _, expected = tensor_exact_f32.certify(source, input_bytes, (0, 8))

    actual = experiment._ordinary_generated_region_output_words(source, input_bytes)

    self.assertEqual(actual, list(struct.unpack("<4I", expected)))

  def test_ordinary_reference_handles_mixed_finite_transport_inputs(self):
    experiment = _load_fpsan_experiment()
    from test.lib import tensor_fpsan

    source = (ROOT / "test/tensor/gluon/programs/native-seed-7.sir").read_text(
      encoding="utf-8"
    )
    words = [tensor_fpsan.unembed_generated_payload(value) for value in (1, 2, 3, 4)]
    input_bytes = struct.pack("<4I", *words)

    output_words = experiment._ordinary_generated_region_output_words(
      source, input_bytes
    )

    self.assertEqual(len(output_words), 4)
    self.assertTrue(all(type(word) is int for word in output_words))

  def test_ordinary_reference_rejects_nonfinite_inputs(self):
    experiment = _load_fpsan_experiment()

    source = (ROOT / "test/tensor/gluon/programs/native-seed-7.sir").read_text(
      encoding="utf-8"
    )
    input_bytes = struct.pack("<4I", 0x7F800000, 0, 0, 0)

    with self.assertRaisesRegex(ValueError, "finite f32"):
      experiment._ordinary_generated_region_output_words(source, input_bytes)

  def test_ordinary_reference_rounds_each_binary32_add(self):
    experiment = _load_fpsan_experiment()

    result = experiment._round_f32_add(1.0, 2**-24)

    self.assertEqual(struct.unpack("<I", struct.pack("<f", result))[0], 0x3F800000)

  def test_ordinary_reference_preserves_all_output_lanes(self):
    experiment = _load_fpsan_experiment()

    source = (ROOT / "test/tensor/gluon/programs/native-seed-7.sir").read_text(
      encoding="utf-8"
    )
    words = experiment._ordinary_generated_region_output_words(
      source, struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    )

    self.assertEqual(len(words), 4)

  def _wide_counterexample(self, width=4):
    from test.lib import tensor_fpsan

    source = (ROOT / "test/tensor/gluon/programs/native-seed-7.sir").read_text(
      encoding="utf-8"
    )
    source = source.replace("tensor [2, 2]", f"tensor [2, {width}]")
    source = source.replace("tensor [1, 2]", f"tensor [1, {width}]")
    source = source.replace("[2, 2];", f"[2, {width}];")
    element_count = 2 * width
    zero_values = ", ".join("0.0" for _ in range(element_count))
    source = source.replace("{0.0, 0.0, 0.0, 0.0}", "{" + zero_values + "}")
    payloads = (
      (3, 3, 4, 2, 2, 8, 3, 0)
      if width == 4
      else tuple(index % 9 for index in range(element_count))
    )
    input_words = [tensor_fpsan.unembed_generated_payload(value) for value in payloads]
    return source, struct.pack(f"<{element_count}I", *input_words)

  def test_ordinary_reference_admits_nonnegative_two_term_reductions(self):
    experiment = _load_fpsan_experiment()
    from test.lib import tensor_fpsan

    source = (ROOT / "test/tensor/gluon/programs/native-seed-7.sir").read_text(
      encoding="utf-8"
    )
    words = [tensor_fpsan.unembed_generated_payload(value) for value in (1, 2, 3, 4)]

    admission = experiment._ordinary_generated_region_admission(
      source, struct.pack("<4I", *words)
    )

    self.assertEqual(admission["status"], "admitted")
    self.assertEqual(admission["max_reduction_width"], 2)
    self.assertEqual(
      admission["contract"],
      "nonnegative-finite-binary32-rne-ops-reduction-width-at-most-2-v3",
    )

  def test_ordinary_reference_excludes_reductions_with_four_terms(self):
    experiment = _load_fpsan_experiment()
    source, input_bytes = self._wide_counterexample()

    admission = experiment._ordinary_generated_region_admission(source, input_bytes)

    self.assertEqual(admission["status"], "unclassified")
    self.assertEqual(admission["max_reduction_width"], 4)
    self.assertIn("reduction width 4", admission["reason"])

  def test_ordinary_reference_admits_exact_integer_four_term_reductions(self):
    experiment = _load_fpsan_experiment()
    source, _input_bytes = self._wide_counterexample()
    input_bytes = struct.pack("<8f", 1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0)

    admission = experiment._ordinary_generated_region_admission(source, input_bytes)

    self.assertEqual(admission["status"], "admitted")
    self.assertEqual(admission["max_reduction_width"], 4)
    self.assertEqual(
      admission["contract"], "nonnegative-integer-f32-all-partials-within-2^24-v1"
    )

  def test_ordinary_reference_admits_eight_term_reductions_with_safe_integers(self):
    experiment = _load_fpsan_experiment()
    source, _input_bytes = self._wide_counterexample(width=8)
    input_bytes = struct.pack("<16f", *([1.0] * 16))

    admission = experiment._ordinary_generated_region_admission(source, input_bytes)

    self.assertEqual(admission["status"], "admitted")
    self.assertEqual(admission["max_reduction_width"], 8)
    self.assertEqual(
      admission["contract"], "nonnegative-integer-f32-all-partials-within-2^24-v1"
    )

  def test_wider_ordinary_contract_refuses_fractional_inputs(self):
    experiment = _load_fpsan_experiment()
    source, _input_bytes = self._wide_counterexample()
    input_bytes = struct.pack("<8f", *([0.5] * 8))

    admission = experiment._ordinary_generated_region_admission(source, input_bytes)

    self.assertEqual(admission["status"], "unclassified")
    self.assertIn("integer-valued", admission["reason"])

  def test_wider_ordinary_contract_refuses_unbounded_partial_sums(self):
    experiment = _load_fpsan_experiment()
    source, _input_bytes = self._wide_counterexample()
    input_bytes = struct.pack("<8f", *([6_000_000.0] * 8))

    admission = experiment._ordinary_generated_region_admission(source, input_bytes)

    self.assertEqual(admission["status"], "unclassified")
    self.assertIn("exceeds exact range", admission["reason"])

  def test_wider_ordinary_contract_records_its_partial_sum_proof(self):
    experiment = _load_fpsan_experiment()
    source, _input_bytes = self._wide_counterexample()
    input_bytes = struct.pack("<8f", *([1.0] * 8))

    admission = experiment._ordinary_generated_region_admission(source, input_bytes)

    proof = admission["exact_integer_proof"]
    self.assertEqual(proof["integer_exactness_bound"], 1 << 24)
    self.assertLessEqual(proof["max_reduction_partial_sum"], 1 << 24)
    self.assertEqual(len(proof["proof_sha256"]), 64)

  def test_generated_arithmetic_corpus_is_admitted_by_ordinary_numeric_contracts(self):
    experiment = _load_fpsan_experiment()
    from test.lib import tensor_exact_f32

    manifest = json.loads(
      (ROOT / "test/tensor/gluon/programs/native-generated-corpus.json").read_text(
        encoding="utf-8"
      )
    )
    records = [
      record
      for record in manifest["programs"]
      if record["generation"]["operation_mode"]
      in {"subtraction", "multiplication", "mixed"}
    ]

    self.assertEqual(len(records), 4)
    for record in records:
      with self.subTest(program_id=record["program_id"]):
        source = (
          ROOT / "test/tensor/gluon/programs" / record["source_file"]
        ).read_text(encoding="utf-8")
        shape, _iterations, _types, _nodes, _updates = (
          tensor_exact_f32._generated_region_program(source)
        )
        values = [index % 8 + 1 for index in range(math.prod(shape))]
        input_bytes = struct.pack(f"<{len(values)}f", *values)
        admission = experiment._ordinary_generated_region_admission(source, input_bytes)
        exact_proof, exact_output = tensor_exact_f32.certify(
          source, input_bytes, (0, 8)
        )
        actual_words = experiment._ordinary_generated_region_output_words(
          source, input_bytes
        )
        actual_output = struct.pack(f"<{len(actual_words)}I", *actual_words)

        self.assertEqual(admission["status"], "admitted", admission.get("reason"))
        self.assertEqual(actual_output, exact_output)
        if admission["max_reduction_width"] > 2:
          self.assertEqual(
            admission["contract"],
            "nonnegative-integer-f32-all-partials-within-2^24-v1",
          )
          self.assertLessEqual(
            admission["exact_integer_proof"]["max_reduction_partial_sum"],
            exact_proof["integer_exactness_bound"],
          )

  def test_unclassified_reduction_difference_keeps_actual_and_reference_words(self):
    experiment = _load_fpsan_experiment()
    source, input_bytes = self._wide_counterexample()
    expected_words = experiment._ordinary_generated_region_output_words(
      source, input_bytes
    )
    alternate_words = [
      0x3120CDD2,
      0x313F7AF7,
      0x3120609F,
      0x3120FEF6,
      0x314C8A3E,
      0x3179054C,
      0x314BEBE6,
      0x314C5919,
    ]

    comparison = experiment._ordinary_generated_region_comparison(
      source, input_bytes, alternate_words
    )

    self.assertEqual(comparison["status"], "unclassified")
    self.assertEqual(comparison["difference"], "different")
    self.assertEqual(comparison["mismatch_lanes"], [0, 1, 2])
    self.assertEqual(comparison["actual_words"], alternate_words)
    self.assertEqual(comparison["reference_words"], expected_words)

  def test_generated_region_corpus_checkpoints_success_before_later_failure(self):
    experiment = _load_fpsan_experiment()
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      sources = []
      for program_id in ("native-seed-7", "native-seed-17"):
        source_path = root / f"{program_id}.sir"
        source_path.write_text(f"fixture {program_id}\n", encoding="utf-8")
        source = source_path.read_bytes()
        sources.append(
          {
            "program_id": program_id,
            "source_path": source_path,
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "generation": {"seed": int(program_id.rsplit("-", 1)[1])},
          }
        )
      output_path = root / "report.json"
      report = {"generated_regions": {"programs": {}}}

      def run_case(case_ir, source_path, **kwargs):
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        programs = saved["generated_regions"]["programs"]
        self.assertEqual(set(programs), {"native-seed-7", "native-seed-17"})
        if source_path.name == "native-seed-7.sir":
          self.assertEqual(programs["native-seed-17"]["status"], "queued")
          return {
            "status": "pass",
            "elapsed_s": 0.25,
            "smt_witness": {"input_float_words": [0x3F800000]},
            "ordinary_reference": {"status": "pass"},
          }
        self.assertEqual(programs["native-seed-7"]["status"], "pass")
        raise AssertionError("synthetic output mismatch")

      with mock.patch.object(
        experiment,
        "_run_generated_region_case",
        side_effect=run_case,
      ):
        experiment._run_generated_region_corpus(
          report,
          output_path,
          root / "ir",
          sources,
          target=1268,
          target_index=(0, 0),
          timeout_s=30.0,
        )

      saved = json.loads(output_path.read_text(encoding="utf-8"))
      programs = saved["generated_regions"]["programs"]
      self.assertEqual(programs["native-seed-7"]["status"], "pass")
      self.assertEqual(
        programs["native-seed-7"]["smt_witness"]["input_float_words"],
        [0x3F800000],
      )
      self.assertEqual(
        programs["native-seed-7"]["ordinary_reference"]["status"], "pass"
      )
      self.assertEqual(programs["native-seed-17"]["status"], "ERROR")
      self.assertEqual(
        programs["native-seed-17"]["failure"]["message"],
        "synthetic output mismatch",
      )
      self.assertEqual(saved["generated_regions"]["status"], "failed")
      self.assertEqual(saved["status"], "failed")
      self.assertEqual(
        programs["native-seed-17"]["property"],
        {"kind": "output-payload-equals", "target": 1268, "index": [0, 0]},
      )
      self.assertGreaterEqual(programs["native-seed-17"]["elapsed_s"], 0.0)

  def test_generated_region_failure_saves_stage_witness_and_available_ir(self):
    experiment = _load_fpsan_experiment()
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      source_path = root / "native-seed-7.sir"
      source_path.write_text("fixture native-seed-7\n", encoding="utf-8")
      source = source_path.read_bytes()
      sources = [
        {
          "program_id": "native-seed-7",
          "source_path": source_path,
          "source_sha256": hashlib.sha256(source).hexdigest(),
          "generation": {"seed": 7},
        }
      ]
      output_path = root / "report.json"
      report = {"generated_regions": {"programs": {}}}

      def fail_after_partial_execution(case_ir, source_path, **kwargs):
        (case_ir / "ordinary.ttgir").write_text(
          "module { tensor.generate }", encoding="utf-8"
        )
        kwargs["checkpoint"](
          "ordinary_execution",
          smt_witness={"input_float_words": [0x3F800000]},
          ordinary_output_words=[0x3120CDD2],
        )
        raise RuntimeError("ordinary output mismatch")

      with mock.patch.object(
        experiment,
        "_run_generated_region_case",
        side_effect=fail_after_partial_execution,
      ):
        experiment._run_generated_region_corpus(
          report,
          output_path,
          root / "ir",
          sources,
          target=1268,
          target_index=(0, 0),
          timeout_s=30.0,
        )

      saved = json.loads(output_path.read_text(encoding="utf-8"))
      result = saved["generated_regions"]["programs"]["native-seed-7"]
      self.assertEqual(result["failure"]["stage"], "ordinary_execution")
      self.assertEqual(result["smt_witness"]["input_float_words"], [0x3F800000])
      self.assertEqual(result["ordinary_output_words"], [0x3120CDD2])
      self.assertEqual(result["source"]["sha256"], hashlib.sha256(source).hexdigest())
      self.assertEqual(result["property"]["target"], 1268)
      self.assertEqual(len(result["artifacts"]), 1)
      self.assertEqual(result["artifacts"][0]["path"], "ordinary.ttgir")

  def test_unavailable_finite_reference_is_unclassified_with_actual_words(self):
    experiment = _load_fpsan_experiment()
    source = (ROOT / "test/tensor/gluon/programs/native-seed-7.sir").read_text(
      encoding="utf-8"
    )
    max_finite = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]
    input_bytes = struct.pack("<4f", max_finite, max_finite, 0.0, 0.0)

    comparison = experiment._ordinary_generated_region_comparison(
      source, input_bytes, [0, 0, 0, 0]
    )

    self.assertEqual(comparison["status"], "unclassified")
    self.assertEqual(comparison["difference"], "unavailable")
    self.assertIsNone(comparison["reference_words"])
    self.assertEqual(comparison["actual_words"], [0, 0, 0, 0])
    self.assertIn("reference unavailable", comparison["admission"]["reason"])

  def test_expanded_four_term_reduction_retains_unclassified_gpu_words(self):
    experiment = _load_fpsan_experiment()
    source = (
      ROOT / "test/tensor/gluon/programs/" / "native-seed-6-shape-2x4-iterations-2.sir"
    ).read_text(encoding="utf-8")
    input_bytes = struct.pack("<8f", *([0.5] * 8))
    actual_words = [0x3F800000] * 8

    comparison = experiment._ordinary_generated_region_comparison(
      source, input_bytes, actual_words
    )

    self.assertEqual(comparison["status"], "unclassified")
    self.assertEqual(comparison["admission"]["max_reduction_width"], 4)
    self.assertEqual(comparison["actual_words"], actual_words)
    self.assertEqual(len(comparison["reference_words"]), 8)

  def test_paired_overhead_schedule_alternates_first_mode(self):
    experiment = _load_fpsan_experiment()

    orders = [list(experiment._paired_measurement_order(index)) for index in range(4)]

    self.assertEqual(
      orders,
      [
        ["fpsan", "ordinary"],
        ["ordinary", "fpsan"],
        ["fpsan", "ordinary"],
        ["ordinary", "fpsan"],
      ],
    )

  def test_paired_overhead_summary_requires_measurement_blocks(self):
    experiment = _load_fpsan_experiment()

    with self.assertRaisesRegex(ValueError, "nonempty"):
      experiment._summarize_paired_overhead([])

  def test_paired_overhead_summary_requires_counterbalanced_pair_order(self):
    experiment = _load_fpsan_experiment()
    block = {
      "order": ["fpsan", "ordinary", "fpsan"],
      "fpsan_ms": 0.03,
      "ordinary_ms": 0.02,
    }

    with self.assertRaisesRegex(ValueError, "counterbalanced"):
      experiment._summarize_paired_overhead([block])

  def test_paired_overhead_summary_rejects_invalid_latencies(self):
    experiment = _load_fpsan_experiment()
    order = ["fpsan", "ordinary"]
    for mode, value in (("fpsan_ms", 0.0), ("ordinary_ms", float("nan"))):
      with self.subTest(mode=mode, value=value):
        block = {
          "order": order,
          "fpsan_ms": 0.03,
          "ordinary_ms": 0.02,
          mode: value,
        }
        with self.assertRaisesRegex(ValueError, "finite positive"):
          experiment._summarize_paired_overhead([block])

  def test_paired_overhead_uses_the_median_of_within_block_ratios(self):
    experiment = _load_fpsan_experiment()
    blocks = [
      {
        "order": ["fpsan", "ordinary"],
        "fpsan_ms": 2.0,
        "ordinary_ms": 1.0,
      },
      {
        "order": ["ordinary", "fpsan"],
        "fpsan_ms": 4.0,
        "ordinary_ms": 2.0,
      },
      {
        "order": ["fpsan", "ordinary"],
        "fpsan_ms": 3.0,
        "ordinary_ms": 1.0,
      },
    ]

    summary = experiment._summarize_paired_overhead(blocks)

    self.assertEqual(summary["block_count"], 3)
    self.assertEqual(summary["ordinary_median_ms"], 1.0)
    self.assertEqual(summary["fpsan_median_ms"], 3.0)
    self.assertEqual(summary["paired_ratio_median"], 2.0)
    self.assertEqual(summary["paired_overhead_median_percent"], 100.0)
    self.assertEqual(summary["paired_ratio_range"], [2.0, 3.0])


if __name__ == "__main__":
  unittest.main()
