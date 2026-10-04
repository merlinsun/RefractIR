"""Validation for the frozen corpus emitted by the native tensor graph builder."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

from test.lib import tensor_exact_f32, tensor_fpsan

PROGRAM_DIR = Path(__file__).resolve().parents[1] / "tensor" / "gluon" / "programs"
MANIFEST_PATH = PROGRAM_DIR / "native-generated-corpus.json"
EXPECTED_SEEDS = (7, 17, 29)


class GeneratedRegionCorpusTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

  def test_manifest_hashes_and_options_admit_all_sources(self):
    self.assertEqual(
      tuple(record["generation"]["seed"] for record in self.manifest["programs"]),
      EXPECTED_SEEDS,
    )
    for record in self.manifest["programs"]:
      with self.subTest(program_id=record["program_id"]):
        source = (PROGRAM_DIR / record["source_file"]).read_text(encoding="utf-8")
        self.assertEqual(
          record["source_sha256"], hashlib.sha256(source.encode("utf-8")).hexdigest()
        )
        self.assertEqual(record["generation"]["shape"], [2, 2])
        self.assertEqual(record["generation"]["iterations"], 2)
        self.assertEqual(record["generation"]["dtype"], "f32")
        shape, iterations, _, _, _ = tensor_exact_f32._generated_region_program(source)
        self.assertEqual(shape, (2, 2))
        self.assertEqual(iterations, 2)
        payload_query = tensor_fpsan.build_generated_payload_query(source)
        self.assertEqual(payload_query["shape"], shape)
        self.assertIn("lost_iteration_output_0", payload_query["script"])
        self.assertIn("lost_update_acc_y_output_0", payload_query["script"])

  def test_native_exporter_reproduces_every_frozen_source(self):
    exporter = os.environ.get("REFRACTIR_TENSOR_GRAPH_EXPORT")
    if not exporter:
      self.skipTest("native graph exporter is supplied by the managed Make target")
    exporter_path = Path(exporter)
    self.assertTrue(exporter_path.is_file(), f"missing exporter: {exporter_path}")
    for record in self.manifest["programs"]:
      with self.subTest(program_id=record["program_id"]):
        result = subprocess.run(
          [str(exporter_path), "--seed", str(record["generation"]["seed"])],
          check=True,
          capture_output=True,
          text=True,
        )
        frozen = (PROGRAM_DIR / record["source_file"]).read_text(encoding="utf-8")
        self.assertEqual(result.stdout, frozen)

  def test_seed_seven_payload_solver_finds_both_loop_mutants(self):
    if not shutil.which("bitwuzla") and not shutil.which("z3"):
      self.skipTest("Bitwuzla or Z3 is needed for the native payload witness")
    source_path = PROGRAM_DIR / "native-seed-7.sir"
    witness = tensor_fpsan.synthesize_generated_region(
      source_path.read_text(encoding="utf-8")
    )

    self.assertEqual(witness["status"], "SAT")
    self.assertEqual(witness["shape"], [2, 2])
    self.assertEqual(witness["iterations"], 2)
    self.assertEqual(
      set(witness["calibration_mutants"]), {"lost_iteration", "lost_update_acc_y"}
    )
    self.assertTrue(
      all(
        mutant["mismatch_lanes"]
        for mutant in witness["calibration_mutants"].values()
      )
    )


if __name__ == "__main__":
  unittest.main()
