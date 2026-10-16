"""Generated Gluon compiler summaries must classify counted loops."""

import hashlib
import importlib.util
import json
import math
import os
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


ORTHOGONAL_REGION_PATH = (
  ROOT / "test/tensor/gluon/programs/orthogonal-reductions-region-v1.sir"
)
NATIVE_SEED_7_PATH = ROOT / "test/tensor/gluon/programs/native-seed-7.sir"
ORTHOGONAL_REGION_SOURCE = ORTHOGONAL_REGION_PATH.read_text(encoding="utf-8")
NATIVE_SEED_7_SOURCE = NATIVE_SEED_7_PATH.read_text(encoding="utf-8")
PLAN_SHAPE = (8, 8)
NATIVE_SHAPE = (2, 2)
PLAN_LAYOUT_SEED = 20261002


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


def _run_fpsan_case_with_paired_timing_failure():
  experiment = _load_fpsan_experiment()
  source_path = ROOT / "test/tensor/gluon/programs/elementwise-multiply-region-v1.sir"
  input_words = [0x3F800000] * 4
  output_words = [0x40800000] * 4
  witness = {
    "status": "SAT",
    "shape": [2, 2],
    "iterations": 1,
    "transport_contract": "test-payload-transport-v1",
    "query_sha256": "a" * 64,
    "input_payloads": input_words,
    "output_payloads": [4] * 4,
    "input_diversity": {"distinct_payloads": 1},
    "input_float_words": input_words,
    "output_float_words": output_words,
    "input_float_bits": [f"{word:08x}" for word in input_words],
    "output_float_bits": [f"{word:08x}" for word in output_words],
    "calibration_mutants": {},
    "property": {"kind": "output-payload-equals", "target": 0, "index": [0, 0]},
    "cost_s": 0.01,
    "solver": "test-solver",
    "solver_outcome": "sat",
  }

  class Tensor:
    def __init__(self, words):
      self.words = words

    def reshape(self, _shape):
      return self

  launch_modes = []

  def compile_mode(_kernel, _inputs, _shape, mode):
    launch_modes.append("fpsan" if mode else "ordinary")
    return Tensor(output_words), SimpleNamespace(asm={}), lambda: None, 0.0125

  def capture_ir(_compiled, prefix, ir_directory):
    content = f"synthetic IR for {prefix}\n".encode("utf-8")
    path = Path(ir_directory) / f"{prefix}.ttgir"
    path.write_bytes(content)
    return {
      "files": {"ttgir": str(path)},
      "stage_sha256": {"ttgir": hashlib.sha256(content).hexdigest()},
      "stage_bytes": {"ttgir": len(content)},
    }

  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    output_path = root / "report.json"
    entry = {}
    report = {"generated_regions": {"programs": {"case": entry}}}
    persisted_at_timing = []

    def fail_during_paired_timing(_launches):
      persisted_at_timing.append(
        json.loads(output_path.read_text(encoding="utf-8"))["generated_regions"][
          "programs"
        ]["case"]
      )
      raise RuntimeError("synthetic paired timing failure")

    fake_triton = ModuleType("triton")
    fake_triton.__path__ = []
    fake_triton.jit = lambda function: function
    fake_language = ModuleType("triton.language")
    with (
      mock.patch.dict(
        sys.modules,
        {"triton": fake_triton, "triton.language": fake_language},
      ),
      mock.patch.object(
        experiment, "synthesize_generated_region", return_value=witness
      ),
      mock.patch.object(experiment, "_tensor_from_words", side_effect=Tensor),
      mock.patch.object(
        experiment, "_raw_words", side_effect=lambda tensor: tensor.words
      ),
      mock.patch.object(
        experiment, "_compile_generated_kernel_mode", side_effect=compile_mode
      ),
      mock.patch.object(experiment, "_capture_ir", side_effect=capture_ir),
      mock.patch.object(
        experiment, "_validate_generated_region_payload_lowering", return_value={}
      ),
      mock.patch.object(
        experiment,
        "_measure_paired_generated_modes",
        side_effect=fail_during_paired_timing,
      ),
    ):
      experiment._run_generated_region_case_record(
        report,
        output_path,
        root / "case",
        entry,
        source_path=source_path,
        target=None,
        target_index=(0, 0),
        timeout_s=1.0,
      )

    return {
      "at_timing": persisted_at_timing[0],
      "result": json.loads(output_path.read_text(encoding="utf-8"))[
        "generated_regions"
      ]["programs"]["case"],
      "launch_modes": launch_modes,
      "expected_output_words": output_words,
    }


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

  def test_ordinary_first_launch_checkpoint_precedes_paired_timing(self):
    observation = _run_fpsan_case_with_paired_timing_failure()

    self.assertEqual(observation["launch_modes"], ["fpsan", "ordinary"])
    self.assertEqual(observation["at_timing"]["stage"], "ordinary_output_captured")

  def test_ordinary_first_launch_checkpoint_retains_output_comparison_and_ir(self):
    observation = _run_fpsan_case_with_paired_timing_failure()
    persisted = observation["at_timing"]
    ordinary_ir = persisted["ordinary_compiler_ir"]

    self.assertEqual(
      persisted["ordinary_output_words"], observation["expected_output_words"]
    )
    self.assertEqual(
      persisted["ordinary_reference"]["actual_words"],
      observation["expected_output_words"],
    )
    self.assertEqual(
      persisted["ordinary_reference"]["reference_words"],
      observation["expected_output_words"],
    )
    self.assertEqual(persisted["ordinary_reference"]["status"], "pass")
    self.assertEqual(persisted["work_variants"]["ordinary"]["compiler"], ordinary_ir)
    self.assertTrue(ordinary_ir["stage_sha256"]["ttgir"])

  def test_ordinary_first_launch_checkpoint_retains_compile_wall_time(self):
    observation = _run_fpsan_case_with_paired_timing_failure()
    persisted = observation["at_timing"]

    self.assertEqual(
      persisted["ordinary_first_launch_wall_s_including_compilation"], 0.0125
    )
    self.assertEqual(
      persisted["work_variants"]["ordinary"][
        "first_launch_wall_s_including_compilation"
      ],
      0.0125,
    )

  def test_paired_timing_failure_counts_both_successful_first_launches(self):
    observation = _run_fpsan_case_with_paired_timing_failure()
    accounting = observation["result"]["measurement_accounting"]

    self.assertEqual(accounting["observed"]["useful_compilations"], 2)
    self.assertEqual(accounting["observed"]["useful_executions"], 2)
    self.assertEqual(accounting["observed"]["total_compilations"], 2)
    self.assertEqual(accounting["observed"]["total_executions"], 2)

  def test_paired_timing_failure_keeps_work_unknown_and_totals_censored(self):
    observation = _run_fpsan_case_with_paired_timing_failure()
    accounting = observation["result"]["measurement_accounting"]

    self.assertEqual(accounting["status"], "unknown")
    self.assertEqual(accounting["observed"]["total_compilations"], 2)
    self.assertEqual(accounting["observed"]["total_executions"], 2)
    self.assertIsNone(accounting["total"]["total_compilations"])
    self.assertIsNone(accounting["total"]["total_executions"])
    self.assertIsNone(accounting["total"]["total_measured_launches"])

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


def _consumers(source):
  """The consumer identities the emitter sees in one recorded region source."""
  from test.lib.tensor_gluon import region_consumers

  return region_consumers(source)


def _layout_pool(
  runner, *, shape=PLAN_SHAPE, count=4, seed=PLAN_LAYOUT_SEED, tile=None
):
  """The runner's own deterministic layout pool for one region shape."""
  from test.lib.tensor_gluon_lowering import reduction_blocked_layout

  reduction_layout = reduction_blocked_layout(shape, {0, 1}, num_warps=1)
  return runner.region_layout_pool(
    shape, reduction_layout, num_warps=1, seed=seed, count=count, tile=tile
  )


class _RecordedTensor:
  """A tensor stand-in that reports the bytes the driver's state holds."""

  def __init__(self, state):
    self.state = state

  def reshape(self, _shape):
    return self

  def view(self, _dtype):
    return self

  def cpu(self):
    return self

  def numpy(self):
    return self

  def tobytes(self):
    return self.state["bytes"]


class _StagedTensor:
  """A whole physical allocation a fake launch can rewrite in place.

  The runner stages one allocation per case and hands the kernel a view that
  starts at the guard prefix, so a view reports the bytes of the allocation
  behind it while a launch writes the allocation it was staged into.
  """

  def __init__(self, data, *, base=0, length=None):
    self.data = data
    self.base = base
    self.length = len(data) // 4 if length is None else length

  def __getitem__(self, key):
    if not isinstance(key, slice):
      raise IndexError(f"staged allocation index {key!r} is not a recorded view")
    start = self.base + (key.start or 0)
    stop = self.base + (key.stop if key.stop is not None else self.length)
    return _StagedTensor(self.data, base=start, length=stop - start)

  def clone(self):
    return self

  def to(self, _device):
    return self

  def cpu(self):
    return self

  def view(self, _dtype):
    return self

  def numpy(self):
    return self

  def tobytes(self):
    return bytes(self.data[self.base * 4 : (self.base + self.length) * 4])


def _recording_torch(state):
  """The device the runner launches on, recording every staged allocation."""

  def frombuffer(buffer, *_args, **_kwargs):
    allocation = _StagedTensor(buffer, length=len(buffer) // 4)
    state["allocations"].append(allocation)
    return allocation

  return SimpleNamespace(
    tensor=lambda *_args, **_kwargs: _RecordedTensor(state),
    empty=lambda *_args, **_kwargs: _RecordedTensor(state),
    frombuffer=frombuffer,
    device=lambda name: name,
    float32="float32",
    uint8="uint8",
    __version__="2.5.0",
    cuda=SimpleNamespace(
      is_available=lambda: True,
      get_device_name=lambda: "NVIDIA V100-SXM2-16GB",
      get_device_capability=lambda: (7, 0),
      synchronize=lambda: None,
      Event=lambda **_kwargs: SimpleNamespace(
        record=lambda: None, elapsed_time=lambda _other: 0.0
      ),
    ),
  )


class _LaunchedKernel:
  """The kernel a case-directory run launches, and what one launch leaves."""

  def __init__(self, name, write):
    self.name = name
    self.asm = {}
    self._write = write

  def __getitem__(self, _grid):
    def launch(*args, **_options):
      self._write(args)
      return self

    return launch


def _qualified_environment(**overrides):
  """The environment record a qualified device run reports."""
  record = {
    "schema": "refractir.tensor-environment.v1",
    "status": "environment_available",
    "campaign_available": True,
    "failure_reasons": [],
    "gpus": [
      {
        "name": "Tesla V100-SXM2-32GB",
        "compute_capability": [7, 0],
        "driver_version": "535.274.02",
      }
    ],
    "driver_version": "535.274.02",
    "cuda_version": "12.1",
    "python_version": "3.12.15",
    "torch_version": "2.5.1+cu121",
    "triton_version": "3.9.0",
    "triton_source_revision": "32f06df4bf1c30885ff61ee3488836ca81ed65f0",
    "refractir_revision": "073a8f89c25a7c00c30ff7400fd02958541549b7",
    "package_lock_file": "package-lock.json",
    "package_lock_sha256": "b" * 64,
    "package_lock": {"schema": "refractir.tensor-environment-lock.v1", "packages": []},
  }
  record.update(overrides)
  return record


def _add_one_to_first_word(payload):
  """What the add-one calibration mutant writes into the first output lane."""
  return struct.pack("<f", struct.unpack("<f", payload[:4])[0] + 1.0) + payload[4:]


def _run_generated_region_program(
  *,
  plan_family=None,
  layout_count=1,
  source=None,
  source_path=None,
  input_values=(1, 2, 4, 8),
  tile=None,
  layout=None,
  guard_envelope=None,
  padding=None,
  canary=None,
  revision=None,
  compiler_flags=None,
  symirc=None,
  environment=None,
):
  """Run the generated-region runner against recorded fake launches.

  Only the device, the compiler capture, and the environment probe are faked;
  the exact reference, the layout sampling, and the emitter under test stay
  real. `source` replaces the recorded power-of-two program, which is how a
  tailed region is launched on a declared execution tile.

  `layout` is the physical layout record a case declares. The runner then
  stages that allocation, and the fake device leaves the exact reference at the
  addresses the record names, so the bytes the runner gathers are the bytes a
  real launch of the emitted kernel would leave behind. `environment`,
  `revision`, `compiler_flags`, and `symirc` state the toolchain identity the
  run records instead of whatever host happens to run the test.
  """
  from test.lib import tensor_exact_f32, tensor_layout

  runner = _load_runner()
  if source is None:
    source = NATIVE_SEED_7_SOURCE
    source_path = NATIVE_SEED_7_PATH
  input_values = tuple(input_values)
  _proof, expected_bytes = tensor_exact_f32.certify(
    source, struct.pack(f"<{len(input_values)}f", *input_values), (0, 8)
  )
  case_count = 1 if plan_family else layout_count
  launches = []
  tensor_state = {"bytes": expected_bytes, "allocations": []}
  kernels = {}
  try:
    addressing = None if layout is None else tensor_layout.storage_addressing(layout)
  except tensor_layout.LayoutError:
    # A record the runner refuses is never staged: the run stops before the
    # device, so the stand-in keeps its dense answers for the refusal itself.
    addressing = None
  envelope = tensor_layout.GuardEnvelope(*(guard_envelope or (0, 0)))
  sentinel = float("nan") if padding is None else float(padding)
  canary_word = sentinel if canary is None else float(canary)

  def physical_after(logical):
    """The allocation a correct launch leaves: the reference at its addresses."""
    return tensor_layout.stage_envelope(
      logical,
      addressing.as_record(),
      element_size=4,
      envelope=envelope,
      padding=struct.pack("<f", canary_word),
      profile=addressing.profile,
    )

  def measure(kernel, args, *, num_warps):
    (ir_directory / "mlir-pass-dump.txt").write_text("dump", encoding="utf-8")
    launches.append({"kernel": kernel, "args": args, "num_warps": num_warps})
    payload = expected_bytes
    if kernel.name == "calibration-add-one-output-mutant":
      payload = _add_one_to_first_word(expected_bytes)
    if addressing is None:
      tensor_state["bytes"] = payload
    else:
      # A correct launch writes the logical mapping and nothing else, so any
      # other slot that differs afterwards is the kernel's own write.
      args[1].data[:] = physical_after(payload)
    return SimpleNamespace(asm={}), {"median_gpu_event_ms_per_launch": 0.0}

  def compiler(_compiled, _name, _ir_directory, _source_lines, *, expected_loop):
    del expected_loop
    return {
      "ir_files": {},
      "observation": {},
      "num_warps": 1,
      "registers_per_thread": 0,
      "spill_bytes": 0,
    }

  def pass_observation(_text, *, source_lines):
    del source_lines
    return {
      "compilation_count": case_count + 1,
      "compilations": [{"snapshot_count": 0} for _ in range(case_count + 1)],
      "interpretation_limit": "test",
    }

  def load_kernel(generated, path):
    kernels[Path(path).stem] = generated.source
    return SimpleNamespace(name=Path(path).stem)

  environment_record = _qualified_environment(**(environment or {}))
  variables = {
    "REFRACTIR_REVISION": revision,
    "REFRACTIR_COMPILER_FLAGS": compiler_flags,
    "REFRACTIR_WORKTREE_DIRTY": "false",
    "REFRACTIR_SYMIRC": None if symirc is None else str(symirc),
  }
  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary).resolve()
    ir_directory = root / "ir"
    output_path = root / "report.json"
    # The recorded environment replaces whatever the host happens to export, so
    # what a run says about its toolchain is what the probe returned.
    with mock.patch.dict(os.environ, {}):
      for name, value in variables.items():
        if value is None:
          os.environ.pop(name, None)
        else:
          os.environ[name] = value
      with (
        mock.patch.object(runner, "torch", _recording_torch(tensor_state)),
        mock.patch.object(runner, "triton", SimpleNamespace(__version__="3.4.0")),
        mock.patch.object(runner.probe_triton_env, "require_qualified_triton"),
        mock.patch.object(
          runner.probe_triton_env, "collect_environment", lambda: environment_record
        ),
        mock.patch.object(runner, "_load_kernel", side_effect=load_kernel) as loader,
        mock.patch.object(runner, "_measure_region_kernel", side_effect=measure),
        mock.patch.object(runner, "_capture_region_compiler", side_effect=compiler),
        mock.patch.object(
          runner, "render_gluon_kernel", wraps=runner.render_gluon_kernel
        ) as render,
        mock.patch.object(runner, "observe_pass_dumps", side_effect=pass_observation),
        mock.patch.object(
          runner.BlockedLayoutPlan, "to_gluon_layout", lambda self: self
        ),
      ):
        try:
          result = runner.run_program(
            source_path,
            output_path,
            ir_directory,
            input_values=input_values,
            domain=(0, 8),
            layout_seed=PLAN_LAYOUT_SEED,
            num_warps=1,
            layout_count=layout_count,
            plan_family=plan_family,
            tile=tile,
            layout=layout,
            guard_envelope=guard_envelope,
            padding=padding,
            canary=canary,
          )
          refusal = None
        except Exception as failure:  # noqa: BLE001 - the refusal is the record
          result = None
          refusal = failure
    return {
      "result": result,
      "refusal": refusal,
      "report": (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.exists()
        else None
      ),
      "launches": launches,
      "kernels": kernels,
      "allocations": tensor_state["allocations"],
      "ir_files": (
        sorted(path.name for path in ir_directory.iterdir())
        if ir_directory.exists()
        else []
      ),
      "render": render,
      "loader": loader,
      "runner": runner,
      "ir_directory": ir_directory,
      "output_path": output_path,
    }


class GeneratedRegionPlanRunnerTests(unittest.TestCase):
  """Plan-mode selection and launch stay host-testable and replayable."""

  def test_plan_family_selection_is_reproducible_from_seed_and_shape(self):
    from test.lib.tensor_gluon_lowering import render_gluon_kernel

    runner = _load_runner()
    pool = _layout_pool(runner)
    consumers = _consumers(ORTHOGONAL_REGION_SOURCE)
    first = runner.select_plan_family(
      "loop-state-layout",
      PLAN_SHAPE,
      num_warps=1,
      layouts=_layout_pool(runner),
      consumers=consumers,
    )
    second = runner.select_plan_family(
      "loop-state-layout",
      PLAN_SHAPE,
      num_warps=1,
      layouts=_layout_pool(runner),
      consumers=consumers,
    )

    self.assertEqual(first.name, "loop-state-layout")
    self.assertEqual(first.as_record(), second.as_record())
    self.assertEqual(
      {
        first.load.name,
        first.compute.name,
        first.reduction.name,
        first.store.name,
        first.state_layout.name,
      },
      {plan.name for plan in pool},
    )
    self.assertEqual(
      render_gluon_kernel(ORTHOGONAL_REGION_SOURCE, plan=first).layout_parameters,
      render_gluon_kernel(ORTHOGONAL_REGION_SOURCE, plan=second).layout_parameters,
    )

  def test_unknown_plan_family_is_refused_with_the_known_names(self):
    runner = _load_runner()

    with self.assertRaisesRegex(ValueError, "unknown plan family") as refusal:
      runner.select_plan_family(
        "all-same-layouts",
        PLAN_SHAPE,
        num_warps=1,
        layouts=_layout_pool(runner),
        consumers=_consumers(ORTHOGONAL_REGION_SOURCE),
      )

    for family in (
      "all-same-layout",
      "load-to-compute",
      "shared-value-reused",
      "split-reduction-consumers",
      "reduction-result-expand",
      "loop-state-layout",
    ):
      self.assertIn(family, str(refusal.exception))

  def test_plan_family_beyond_the_sampled_pool_is_refused(self):
    runner = _load_runner()
    pool = _layout_pool(runner, count=2)

    with self.assertRaisesRegex(ValueError, "loop-state-layout") as refusal:
      runner.select_plan_family(
        "loop-state-layout",
        PLAN_SHAPE,
        num_warps=1,
        layouts=pool,
        consumers=_consumers(ORTHOGONAL_REGION_SOURCE),
      )

    self.assertIn("places 4 distinct layouts", str(refusal.exception))
    self.assertIn("sampled 2", str(refusal.exception))
    expressible = runner.select_plan_family(
      "load-to-compute",
      PLAN_SHAPE,
      num_warps=1,
      layouts=pool,
      consumers=_consumers(ORTHOGONAL_REGION_SOURCE),
    )
    self.assertEqual(
      [expressible.load.name, expressible.compute.name],
      [pool[1].name, pool[0].name],
    )

  def test_plan_launch_arguments_follow_the_kernel_layout_parameters(self):
    from test.lib.tensor_gluon_lowering import render_gluon_kernel

    runner = _load_runner()
    pool = _layout_pool(runner)
    family = runner.select_plan_family(
      "loop-state-layout",
      PLAN_SHAPE,
      num_warps=1,
      layouts=pool,
      consumers=_consumers(ORTHOGONAL_REGION_SOURCE),
    )
    generated = render_gluon_kernel(ORTHOGONAL_REGION_SOURCE, plan=family)

    with mock.patch.object(
      runner.BlockedLayoutPlan, "to_gluon_layout", lambda self: self
    ):
      arguments = runner.region_launch_arguments(
        generated, list(reversed(pool)), "input", "output"
      )

    self.assertEqual(arguments[:2], ("input", "output"))
    self.assertEqual(
      [layout.name.replace("-", "_") for layout in arguments[2:]],
      list(generated.layout_parameters),
    )
    self.assertEqual(
      [layout.name for layout in arguments[2:]],
      [
        family.load.name,
        family.reduction.name,
        family.store.name,
        family.state_layout.name,
      ],
    )
    self.assertNotEqual(
      [layout.name for layout in arguments[2:]],
      [plan.name for plan in reversed(pool)][: len(arguments) - 2],
    )

  def test_no_plan_run_keeps_the_role_layout_launch_protocol(self):
    runner = _load_runner()
    observation = _run_generated_region_program(layout_count=2)
    pool = _layout_pool(runner, shape=NATIVE_SHAPE, count=2)
    launches = observation["launches"]
    result = observation["result"]

    self.assertEqual(len(launches), 3)
    self.assertEqual(
      observation["render"].call_args, mock.call(NATIVE_SEED_7_SOURCE, tile=None)
    )
    self.assertEqual(
      [layout.name for layout in launches[1]["args"][2:]],
      [pool[1].name, pool[1].name, pool[0].name, pool[1].name],
    )
    self.assertEqual(list(result["cases"]), [pool[0].name, pool[1].name])
    self.assertEqual(result["contract"], "generated-region-exact-f32-layout-study-v1")
    self.assertNotIn("plan", result)
    self.assertNotIn("plan", observation["report"])
    self.assertEqual(result["measurement_protocol"]["expected_layout_count"], 2)

  def test_a_tailed_region_launches_on_its_declared_execution_tile(self):
    from test.lib.tensor_gluon_lowering import render_gluon_kernel
    from test.unit.test_tensor_exact_composed import tailed_region_source

    runner = _load_runner()
    source = tailed_region_source((2, 3))
    with tempfile.TemporaryDirectory() as temporary:
      program = Path(temporary) / "tailed.sir"
      program.write_text(source, encoding="utf-8")
      observation = _run_generated_region_program(
        source=source,
        source_path=program,
        input_values=(1, 2, 3, 4, 5, 6),
        tile=(2, 4),
        layout_count=2,
      )
    result = observation["result"]
    emitted = render_gluon_kernel(source, tile=(2, 4)).source
    pool = _layout_pool(runner, shape=(2, 3), count=2, tile=(2, 4))

    self.assertEqual(observation["render"].call_args.kwargs, {"tile": (2, 4)})
    self.assertEqual(result["shape"], [2, 3])
    self.assertEqual(result["execution_tile"], [2, 4])
    self.assertEqual(list(result["cases"]), [plan.name for plan in pool])
    for case in result["cases"].values():
      self.assertEqual(case["status"], "pass")
    # The launched kernel keeps the padded tile and masks load, reduction, and
    # store; the layouts it is launched with are the tile's, not the tail's.
    self.assertIn("gl.zeros((2, 4)", emitted)
    self.assertIn("valid_load_layout = ", emitted)
    self.assertIn("valid_reduction_layout", emitted)
    self.assertIn("mask=valid_store_layout", emitted)
    self.assertEqual(render_gluon_kernel(source, tile=(2, 4)).tile, (2, 4))
    for layout in pool:
      layout.validate((2, 3), num_warps=1, tile=(2, 4))

  def test_plan_mode_run_records_the_family_roles_and_launch_order(self):
    from test.lib.tensor_gluon_lowering import render_gluon_kernel

    runner = _load_runner()
    observation = _run_generated_region_program(
      plan_family="loop-state-layout", layout_count=4
    )
    result = observation["result"]
    record = result["plan"]
    pool = _layout_pool(runner, shape=NATIVE_SHAPE, count=4)
    family = runner.select_plan_family(
      "loop-state-layout",
      NATIVE_SHAPE,
      num_warps=1,
      layouts=pool,
      consumers=_consumers(NATIVE_SEED_7_SOURCE),
    )
    generated = render_gluon_kernel(NATIVE_SEED_7_SOURCE, plan=family)

    self.assertEqual(list(result["cases"]), ["loop-state-layout"])
    self.assertEqual(record["family"], "loop-state-layout")
    self.assertEqual(observation["report"]["plan"]["family"], record["family"])
    self.assertEqual(observation["report"]["plan"]["arguments"], record["arguments"])
    self.assertEqual(
      observation["report"]["plan"]["layout_parameters"], record["layout_parameters"]
    )
    self.assertEqual(record["roles"], family.as_record())
    self.assertEqual(record["layout_parameters"], list(generated.layout_parameters))
    self.assertEqual(
      [item["parameter"] for item in record["arguments"]],
      list(generated.layout_parameters),
    )
    self.assertEqual(
      [item["layout"] for item in record["arguments"]],
      [layout.name for layout in generated.layout_arguments(pool)],
    )
    self.assertEqual(
      [
        layout.name.replace("-", "_")
        for layout in observation["launches"][0]["args"][2:]
      ],
      list(generated.layout_parameters),
    )
    self.assertEqual(
      [layout.name for layout in observation["launches"][1]["args"][2:]],
      [layout.name for layout in observation["launches"][0]["args"][2:]],
    )
    named = {
      record["roles"][role]
      for role in ("load", "compute", "reduction", "store", "state_layout")
      if record["roles"][role]
    }
    named.update(record["roles"]["consumer_views"].values())
    self.assertEqual(sorted(record["layouts"]), sorted(named))
    for name, identity in record["layouts"].items():
      self.assertEqual(identity["name"], name)
    self.assertEqual(
      [layout["name"] for layout in record["pool"]],
      [plan.name for plan in pool],
    )

  def test_single_layout_pool_expresses_only_the_same_layout_family(self):
    runner = _load_runner()
    pool = _layout_pool(runner, count=1)
    family = runner.select_plan_family(
      "all-same-layout",
      PLAN_SHAPE,
      num_warps=1,
      layouts=pool,
      consumers=_consumers(ORTHOGONAL_REGION_SOURCE),
    )

    self.assertEqual(
      {
        family.load.name,
        family.compute.name,
        family.reduction.name,
        family.store.name,
      },
      {pool[0].name},
    )
    with self.assertRaisesRegex(ValueError, "load-to-compute"):
      runner.select_plan_family(
        "load-to-compute",
        PLAN_SHAPE,
        num_warps=1,
        layouts=pool,
        consumers=_consumers(ORTHOGONAL_REGION_SOURCE),
      )

  def test_plan_mode_refuses_an_unqualified_compiler_before_the_device(self):
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temporary:
      with mock.patch.object(
        runner.probe_triton_env,
        "require_qualified_triton",
        side_effect=RuntimeError("triton source revision is not qualified"),
      ):
        with self.assertRaisesRegex(RuntimeError, "source revision"):
          runner.run_program(
            NATIVE_SEED_7_PATH,
            Path(temporary) / "report.json",
            Path(temporary) / "ir",
            input_values=(1, 2, 4, 8),
            plan_family="loop-state-layout",
          )


TAIL_SHAPE = (2, 3)
TAIL_TILE = (2, 4)
TAIL_INPUT_VALUES = (1, 2, 3, 4, 5, 6)
# The physical layout the padded-tail cases declare: rows strided by the
# execution tile's width plus a gap, with the tensor starting at element 1.
STRIDED_TAIL_RECORD = {
  "shape": list(TAIL_SHAPE),
  "element_strides": [4, 1],
  "element_offset": 1,
  "storage_span": 10,
  "axis_order": [1, 0],
}


def _tailed_case_file(root):
  """One tailed region written where the runner can read it."""
  from test.unit.test_tensor_exact_composed import tailed_region_source

  source = tailed_region_source(TAIL_SHAPE)
  path = Path(root) / "tailed-2x3.sir"
  path.write_text(source, encoding="utf-8")
  return source, path


def _run_case_directory_runner(
  *,
  source,
  record,
  input_values=(1, 2, 4, 8),
  guard_envelope=(3, 5),
  padding="3f800001",
  canary="3f000002",
):
  """Run the case-directory route against a recorded manifest and fake device.

  `record` is restated into both buffer records of the manifest, which is what a
  case builder writes when both buffers are staged in one layout, and the fake
  device leaves the exact reference at the addresses that record names. The
  cross-lane route places the whole warp on one reduction axis, so the region it
  runs is the power-of-two shape that route is built for.
  """
  from test.lib import tensor_exact_f32, tensor_layout

  runner = _load_runner()
  input_bytes = struct.pack(f"<{len(input_values)}f", *input_values)
  _proof, expected_bytes = tensor_exact_f32.certify(source, input_bytes, (0, 8))
  envelope = tensor_layout.GuardEnvelope(*guard_envelope)
  addressing = None if record is None else tensor_layout.storage_addressing(record)
  state = {"bytes": expected_bytes, "allocations": []}
  kernels = {}

  def write(args):
    if not isinstance(args[1], _StagedTensor):
      # A run that launches the dense route records its own bytes, which is the
      # reading a record the route never received is missing.
      state["bytes"] = expected_bytes
      return
    args[1].data[:] = tensor_layout.stage_envelope(
      expected_bytes,
      record,
      element_size=4,
      envelope=envelope,
      padding=bytes.fromhex(canary),
      profile=addressing.profile,
    )

  # A manifest that declares no addressing is what the exact profiles carry: its
  # buffer records hold only what the worker needs to stage the dense buffers.
  layout_fields = {"shape": [2, 2]} if record is None else record
  execution = (
    {}
    if record is None
    else {
      "guard_envelope": {"prefix": guard_envelope[0], "suffix": guard_envelope[1]},
      "padding_by_dtype": {"f32": padding},
      "output_canary_by_dtype": {"f32": canary},
    }
  )
  case = SimpleNamespace(
    manifest={
      "profile": "tensor-v3",
      "inputs": [{**layout_fields, "name": "x", "dtype": "f32"}],
      "output": {**layout_fields, "name": "out", "dtype": "f32"},
      "execution": execution,
      "reference": {
        "exactness": {
          "kind": tensor_exact_f32.REGION_KIND,
          "input_domain": [0, 8],
        }
      },
    },
    program=source.encode("utf-8"),
    inputs=input_bytes,
    expected=expected_bytes,
  )

  def load_kernel(generated, path):
    kernels[Path(path).stem] = generated.source
    return _LaunchedKernel(Path(path).stem, write)

  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary).resolve()
    with mock.patch.dict(os.environ, {}):
      with (
        mock.patch.object(runner, "torch", _recording_torch(state)),
        mock.patch.object(runner, "triton", SimpleNamespace(__version__="3.4.0")),
        mock.patch.object(runner.probe_triton_env, "require_qualified_triton"),
        mock.patch.object(
          runner.probe_triton_env,
          "collect_environment",
          lambda: _qualified_environment(),
        ),
        mock.patch.object(
          runner.tensor_case, "load_case", lambda *_args, **_kwargs: case
        ),
        mock.patch.object(runner, "_load_kernel", side_effect=load_kernel),
        mock.patch.object(
          runner, "render_gluon_kernel", wraps=runner.render_gluon_kernel
        ) as render,
        mock.patch.object(
          runner,
          "observe_compiler_ir",
          lambda _stages, *, source_lines: {
            "layout_conversion_count": 1,
            "stages": {
              "ptx": {"warp_shuffle_count": 1, "shared_memory_transfer_count": 0}
            },
          },
        ),
        mock.patch.object(
          runner.BlockedLayoutPlan, "to_gluon_layout", lambda self: self
        ),
      ):
        try:
          report = runner.run(
            root / "case",
            root / "report.json",
            root / "ir",
            symiri="symiri",
          )
          refusal = None
        except Exception as failure:  # noqa: BLE001 - the refusal is the record
          report, refusal = None, failure
    return {
      "report": report,
      "refusal": refusal,
      "render": render,
      "kernels": kernels,
      "allocations": state["allocations"],
    }


class GeneratedRegionStrideRunnerTests(unittest.TestCase):
  """The runner addresses the physical layout record a case declares."""

  def test_the_runner_passes_the_layout_record_to_the_emitter(self):
    with tempfile.TemporaryDirectory() as temporary:
      source, path = _tailed_case_file(temporary)
      for plan_family in (None, "all-same-layout"):
        with self.subTest(plan_family=plan_family):
          observation = _run_generated_region_program(
            source=source,
            source_path=path,
            input_values=TAIL_INPUT_VALUES,
            tile=TAIL_TILE,
            layout=STRIDED_TAIL_RECORD,
            plan_family=plan_family,
            layout_count=1,
          )

          self.assertIsNone(observation["refusal"])
          self.assertEqual(
            observation["render"].call_args.kwargs["layout"], STRIDED_TAIL_RECORD
          )

  def test_a_strided_record_addresses_the_staged_allocation(self):
    from test.lib import tensor_layout

    with tempfile.TemporaryDirectory() as temporary:
      source, path = _tailed_case_file(temporary)
      observation = _run_generated_region_program(
        source=source,
        source_path=path,
        input_values=TAIL_INPUT_VALUES,
        tile=TAIL_TILE,
        layout=STRIDED_TAIL_RECORD,
        guard_envelope=(3, 5),
        padding=1.0000001,
        layout_count=1,
      )

      self.assertIsNone(observation["refusal"])
      emitted = observation["kernels"]["generated-region-kernel"]
      self.assertIn(
        "load_offsets = 1 + load_rows[:, None] * 4 + load_columns[None, :]", emitted
      )
      self.assertIn(
        "store_offsets = 1 + store_rows[:, None] * 4 + store_columns[None, :]",
        emitted,
      )
      self.assertNotIn("load_rows[:, None] * 3", emitted)
      # The allocation the kernel is handed places the logical input at the
      # recorded addresses, so the emitted strides and the staged bytes agree.
      self.assertEqual(
        bytes(observation["allocations"][0].data),
        tensor_layout.stage_envelope(
          struct.pack(f"<{len(TAIL_INPUT_VALUES)}f", *TAIL_INPUT_VALUES),
          STRIDED_TAIL_RECORD,
          element_size=4,
          envelope=tensor_layout.GuardEnvelope(3, 5),
          padding=struct.pack("<f", 1.0000001),
          profile="v3",
        ),
      )
      for case in observation["result"]["cases"].values():
        self.assertEqual(case["status"], "pass")
        self.assertEqual(case["guard_verdict"]["guard_or_gap_slots_changed"], [])
        self.assertEqual(case["guard_verdict"]["logical_slots_changed"], 6)

  def test_an_identity_record_emits_the_pre_change_kernel(self):
    identity = {
      "shape": [2, 2],
      "element_strides": [2, 1],
      "element_offset": 0,
      "storage_span": 4,
      "axis_order": [1, 0],
    }
    without = _run_generated_region_program(layout_count=1)
    with_record = _run_generated_region_program(layout=identity, layout_count=1)

    self.assertIsNone(without["refusal"])
    self.assertIsNone(with_record["refusal"])
    self.assertEqual(
      with_record["kernels"]["generated-region-kernel"],
      without["kernels"]["generated-region-kernel"],
    )

  def test_an_inconsistent_record_is_refused_before_emission(self):
    from test.lib import tensor_layout

    def record(**overrides):
      return {**STRIDED_TAIL_RECORD, **overrides}

    cases = (
      (record(storage_span=7), "exceeds storage span"),
      (record(axis_order=[0, 1]), "fastest to slowest stride"),
    )
    with tempfile.TemporaryDirectory() as temporary:
      source, path = _tailed_case_file(temporary)
      for layout, reason in cases:
        with self.subTest(layout=layout):
          observation = _run_generated_region_program(
            source=source,
            source_path=path,
            input_values=TAIL_INPUT_VALUES,
            tile=TAIL_TILE,
            layout=layout,
            layout_count=1,
          )

          self.assertIsInstance(observation["refusal"], tensor_layout.LayoutError)
          self.assertIn(reason, str(observation["refusal"]))
          self.assertEqual(observation["render"].call_count, 0)
          self.assertEqual(observation["loader"].call_count, 0)
          self.assertEqual(observation["ir_files"], [])

  def test_a_runner_invocation_records_the_toolchain_qualification(self):
    with tempfile.TemporaryDirectory() as temporary:
      symirc = Path(temporary).resolve() / "symirc"
      symirc.write_bytes(b"recorded-symirc-bytes")
      observation = _run_generated_region_program(
        layout_count=1,
        revision="073a8f89c25a7c00c30ff7400fd02958541549b7",
        compiler_flags="-O2 -DTEST",
        symirc=symirc,
      )

    self.assertIsNone(observation["refusal"])
    qualification = observation["report"]["qualification"]
    self.assertEqual(
      qualification["refractir_revision"],
      "073a8f89c25a7c00c30ff7400fd02958541549b7",
    )
    self.assertEqual(qualification["working_tree_dirty"], "false")
    self.assertEqual(qualification["compiler"]["flags"], "-O2 -DTEST")
    self.assertEqual(
      qualification["compiler"]["sha256"],
      hashlib.sha256(b"recorded-symirc-bytes").hexdigest(),
    )
    self.assertEqual(qualification["compiler"]["executable"], str(symirc))
    self.assertEqual(qualification["triton_version"], "3.9.0")
    self.assertEqual(
      qualification["triton_source_revision"],
      "32f06df4bf1c30885ff61ee3488836ca81ed65f0",
    )
    self.assertEqual(qualification["package_lock_sha256"], "b" * 64)
    self.assertEqual(qualification["device"]["name"], "NVIDIA V100-SXM2-16GB")
    self.assertEqual(qualification["device"]["compute_capability"], [7, 0])
    self.assertTrue(qualification["campaign_valid"])
    self.assertEqual(
      qualification["environment_file"],
      str(observation["ir_directory"] / "environment.json"),
    )
    self.assertIn("environment.json", observation["ir_files"])
    self.assertIn("package-lock.json", observation["ir_files"])
    for case in observation["report"]["cases"].values():
      self.assertTrue(case["qualification"]["campaign_valid"])
      self.assertEqual(
        case["qualification"]["refractir_revision"],
        qualification["refractir_revision"],
      )
      self.assertEqual(case["qualification"]["package_lock_sha256"], "b" * 64)

  def test_an_unqualified_environment_leaves_every_case_unqualified(self):
    with tempfile.TemporaryDirectory() as temporary:
      symirc = Path(temporary).resolve() / "symirc"
      symirc.write_bytes(b"recorded-symirc-bytes")
      observation = _run_generated_region_program(
        layout_count=1,
        revision="073a8f89c25a7c00c30ff7400fd02958541549b7",
        compiler_flags="-O2",
        symirc=symirc,
        environment={
          "status": "environment_unavailable",
          "campaign_available": False,
          "failure_reasons": ["gpu_unavailable"],
        },
      )

    self.assertIsNone(observation["refusal"])
    qualification = observation["report"]["qualification"]
    self.assertFalse(qualification["campaign_valid"])
    self.assertEqual(qualification["failure_reasons"], ["gpu_unavailable"])
    for case in observation["report"]["cases"].values():
      self.assertFalse(case["qualification"]["campaign_valid"])

  def test_a_manifest_without_addressing_keeps_the_dense_route(self):
    with tempfile.TemporaryDirectory():
      observation = _run_case_directory_runner(source=NATIVE_SEED_7_SOURCE, record=None)

    self.assertIsNone(observation["refusal"])
    report = observation["report"]
    self.assertEqual(observation["render"].call_args.kwargs, {})
    self.assertIsNone(report["physical_layout"])
    self.assertIsNone(report["guard_verdict"])
    self.assertEqual(report["actual_output_bits"], report["expected_output_bits"])

  def test_the_case_directory_runner_addresses_the_manifest_record(self):
    from test.lib import tensor_layout

    # The cross-lane route lowers the power-of-two shape it is built for, so its
    # record strides rows by more than the row length rather than by a tile.
    record = {
      "shape": [2, 2],
      "element_strides": [3, 1],
      "element_offset": 1,
      "storage_span": 8,
      "axis_order": [1, 0],
    }
    observation = _run_case_directory_runner(source=NATIVE_SEED_7_SOURCE, record=record)

    self.assertIsNone(observation["refusal"])
    report = observation["report"]
    self.assertEqual(observation["render"].call_args.kwargs, {"layout": record})
    self.assertEqual(report["physical_layout"]["record"], record)
    self.assertEqual(
      report["physical_layout"]["guard_envelope"], {"prefix": 3, "suffix": 5}
    )
    self.assertEqual(report["guard_verdict"]["guard_or_gap_slots_changed"], [])
    self.assertEqual(report["guard_verdict"]["logical_slots_changed"], 4)
    self.assertEqual(report["actual_output_bits"], report["expected_output_bits"])
    self.assertIn(
      "load_offsets = 1 + load_rows[:, None] * 3 + load_columns[None, :]",
      observation["kernels"]["generated-region-kernel"],
    )
    self.assertEqual(
      bytes(observation["allocations"][0].data),
      tensor_layout.stage_envelope(
        struct.pack("<4f", 1, 2, 4, 8),
        record,
        element_size=4,
        envelope=tensor_layout.GuardEnvelope(3, 5),
        padding=bytes.fromhex("3f800001"),
        profile="v2",
      ),
    )

  def test_a_dense_record_on_a_padded_tile_keeps_its_addressing(self):
    dense = {
      "shape": list(TAIL_SHAPE),
      "element_strides": [3, 1],
      "element_offset": 0,
      "storage_span": 6,
      "axis_order": [1, 0],
    }
    with tempfile.TemporaryDirectory() as temporary:
      source, path = _tailed_case_file(temporary)
      without = _run_generated_region_program(
        source=source,
        source_path=path,
        input_values=TAIL_INPUT_VALUES,
        tile=TAIL_TILE,
        layout_count=1,
      )
      with_record = _run_generated_region_program(
        source=source,
        source_path=path,
        input_values=TAIL_INPUT_VALUES,
        tile=TAIL_TILE,
        layout=dense,
        layout_count=1,
      )

    self.assertIsNone(without["refusal"])
    self.assertIsNone(with_record["refusal"])
    emitted = without["kernels"]["generated-region-kernel"]
    self.assertEqual(with_record["kernels"]["generated-region-kernel"], emitted)
    self.assertIn(
      "load_offsets = load_rows[:, None] * 3 + load_columns[None, :]", emitted
    )
    # The padded tile still keeps its validity predicates: a dense record states
    # where the tensor is, not how wide the tile is.
    self.assertIn("valid_load_layout", emitted)
    self.assertIn("mask=valid_store_layout", emitted)


if __name__ == "__main__":
  unittest.main()
