"""Generated Gluon compiler summaries must classify counted loops."""

import importlib.util
import hashlib
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
      ["native-seed-7", "native-seed-17", "native-seed-29"],
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
    self.assertIn("do not establish rematerialization", comparison["interpretation_limit"])

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


if __name__ == "__main__":
  unittest.main()
