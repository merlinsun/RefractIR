"""Generated Gluon compiler summaries must classify counted loops."""

import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "test/tensor/gluon/run_generated_region.py"


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

  def _wide_counterexample(self):
    from test.lib import tensor_fpsan

    source = (ROOT / "test/tensor/gluon/programs/native-seed-7.sir").read_text(
      encoding="utf-8"
    )
    source = source.replace("tensor [2, 2]", "tensor [2, 4]")
    source = source.replace("tensor [1, 2]", "tensor [1, 4]")
    source = source.replace("[2, 2];", "[2, 4];")
    source = source.replace(
      "{0.0, 0.0, 0.0, 0.0}",
      "{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}",
    )
    payloads = (3, 3, 4, 2, 2, 8, 3, 0)
    input_words = [tensor_fpsan.unembed_generated_payload(value) for value in payloads]
    return source, struct.pack("<8I", *input_words)

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
      admission["contract"], "nonnegative-f32-reduction-width-at-most-2-v1"
    )

  def test_ordinary_reference_excludes_reductions_with_four_terms(self):
    experiment = _load_fpsan_experiment()
    source, input_bytes = self._wide_counterexample()

    admission = experiment._ordinary_generated_region_admission(source, input_bytes)

    self.assertEqual(admission["status"], "unclassified")
    self.assertEqual(admission["max_reduction_width"], 4)
    self.assertIn("reduction width 4", admission["reason"])

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
    input_bytes = struct.pack("<8f", *([1.0] * 8))
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
