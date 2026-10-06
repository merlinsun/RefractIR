"""Validation for the frozen corpus emitted by the native tensor graph builder."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import unittest
from pathlib import Path

from test.lib import tensor_exact_f32, tensor_fpsan

PROGRAM_DIR = Path(__file__).resolve().parents[1] / "tensor" / "gluon" / "programs"
MANIFEST_PATH = PROGRAM_DIR / "native-generated-corpus.json"
EXPECTED_GENERATION = (
  ("native-seed-7", 7, (2, 2), 2),
  ("native-seed-17", 17, (2, 2), 2),
  ("native-seed-29", 29, (2, 2), 2),
  ("native-seed-0-shape-2x4-iterations-3", 0, (2, 4), 3),
  ("native-seed-3-shape-4x2-iterations-4", 3, (4, 2), 4),
  ("native-seed-6-shape-2x4-iterations-2", 6, (2, 4), 2),
  ("native-seed-2-shape-8x8-iterations-2", 2, (8, 8), 2),
)
LARGE_ORTHOGONAL_PROGRAM_ID = "native-seed-2-shape-8x8-iterations-2"


class GeneratedRegionCorpusTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

  def test_manifest_hashes_and_options_admit_all_sources(self):
    self.assertEqual(len(self.manifest["programs"]), len(EXPECTED_GENERATION))
    self.assertEqual(
      tuple(
        (
          record["program_id"],
          record["generation"]["seed"],
          tuple(record["generation"]["shape"]),
          record["generation"]["iterations"],
        )
        for record in self.manifest["programs"]
      ),
      EXPECTED_GENERATION,
    )
    for record in self.manifest["programs"]:
      with self.subTest(program_id=record["program_id"]):
        source = (PROGRAM_DIR / record["source_file"]).read_text(encoding="utf-8")
        self.assertEqual(
          record["source_sha256"], hashlib.sha256(source.encode("utf-8")).hexdigest()
        )
        self.assertEqual(record["generation"]["dtype"], "f32")
        shape, iterations, _, _, _ = tensor_exact_f32._generated_region_program(source)
        self.assertEqual(shape, tuple(record["generation"]["shape"]))
        self.assertEqual(iterations, record["generation"]["iterations"])
        payload_query = tensor_fpsan.build_generated_payload_query(source)
        self.assertEqual(payload_query["shape"], shape)
        self.assertIn("lost_iteration_output_0", payload_query["script"])
        self.assertIn("lost_update_acc_y_output_0", payload_query["script"])

  def test_expanded_dependency_profiles_include_single_and_double_reductions(self):
    reduction_counts = set()
    signatures = set()
    for record in self.manifest["programs"]:
      source = (PROGRAM_DIR / record["source_file"]).read_text(encoding="utf-8")
      _shape, _iterations, _types, nodes, _updates = (
        tensor_exact_f32._generated_region_program(source)
      )
      reductions = [node for node in nodes if node[1] == "sum"]
      reduction_counts.add(len(reductions))
      signatures.add(tuple((node[1], node[3]) for node in nodes))

    self.assertIn(1, reduction_counts)
    self.assertIn(2, reduction_counts)
    self.assertGreaterEqual(len(signatures), 4)

  def test_manifest_includes_a_64_element_orthogonal_reduction_candidate(self):
    record = next(
      item
      for item in self.manifest["programs"]
      if item["program_id"] == LARGE_ORTHOGONAL_PROGRAM_ID
    )

    self.assertEqual(record["generation"]["seed"], 2)
    self.assertEqual(record["generation"]["shape"], [8, 8])
    self.assertEqual(record["generation"]["iterations"], 2)
    self.assertEqual(record["generation"]["graph_limits"]["max_elements"], 64)

  def test_64_element_candidate_keeps_both_reduction_dependencies(self):
    record = next(
      item
      for item in self.manifest["programs"]
      if item["program_id"] == LARGE_ORTHOGONAL_PROGRAM_ID
    )
    source = (PROGRAM_DIR / record["source_file"]).read_text(encoding="utf-8")
    shape, iterations, _types, nodes, updates = (
      tensor_exact_f32._generated_region_program(source)
    )
    reduction_axes = tuple(node[3] for node in nodes if node[1] == "sum")
    node_by_name = {node[0]: node for node in nodes}
    sum_names = {node[0] for node in nodes if node[1] == "sum"}

    def ancestors(value):
      pending = [value]
      visited = set()
      while pending:
        current = pending.pop()
        if current in visited:
          continue
        visited.add(current)
        node = node_by_name.get(current)
        if node is not None:
          pending.extend(node[2])
      return visited

    state_roots = dict(updates)
    root_dependencies = [ancestors(state_roots[state]) for state in ("acc_x", "acc_y")]

    self.assertEqual(shape, (8, 8))
    self.assertEqual(iterations, 2)
    self.assertEqual(reduction_axes, (1, 0))
    self.assertTrue(
      all(sum_names <= dependencies for dependencies in root_dependencies)
    )
    self.assertIn("t6", root_dependencies[0] & root_dependencies[1])

  def test_64_element_candidate_has_a_full_exact_integer_certificate(self):
    record = next(
      item
      for item in self.manifest["programs"]
      if item["program_id"] == LARGE_ORTHOGONAL_PROGRAM_ID
    )
    source = (PROGRAM_DIR / record["source_file"]).read_text(encoding="utf-8")
    input_bytes = struct.pack("<64f", *range(1, 65))

    proof, output_bytes = tensor_exact_f32.certify(source, input_bytes, (0, 64))

    self.assertEqual(len(output_bytes), 64 * 4)
    self.assertEqual(proof["integer_exactness_bound"], 1 << 24)
    self.assertTrue(proof["bounds"])
    reduction_bounds = [
      bound for bound in proof["bounds"] if bound["operation"] == "sum"
    ]
    self.assertEqual(len(reduction_bounds), 4)
    self.assertTrue(all("max_partial_sum" in bound for bound in reduction_bounds))

  def test_payload_query_accepts_64_element_orthogonal_reductions(self):
    record = next(
      item
      for item in self.manifest["programs"]
      if item["program_id"] == LARGE_ORTHOGONAL_PROGRAM_ID
    )
    source = (PROGRAM_DIR / record["source_file"]).read_text(encoding="utf-8")
    query = tensor_fpsan.build_generated_payload_query(source)

    self.assertEqual(query["shape"], (8, 8))
    self.assertEqual(query["iterations"], 2)
    self.assertEqual(len(query["variables"]), 64)
    self.assertEqual(len(query["output_names"]), 64)
    self.assertIn("lost_iteration_output_0", query["script"])
    self.assertIn("lost_update_acc_y_output_0", query["script"])

  def test_native_exporter_reproduces_64_element_candidate(self):
    record = next(
      item
      for item in self.manifest["programs"]
      if item["program_id"] == LARGE_ORTHOGONAL_PROGRAM_ID
    )
    exporter = os.environ.get("REFRACTIR_TENSOR_GRAPH_EXPORT")
    if not exporter:
      self.skipTest("native graph exporter is supplied by the managed Make target")

    result = subprocess.run(
      [
        exporter,
        "--seed",
        str(record["generation"]["seed"]),
        "--shape",
        "8,8",
        "--iterations",
        str(record["generation"]["iterations"]),
      ],
      check=True,
      capture_output=True,
      text=True,
    )

    self.assertEqual(
      result.stdout,
      (PROGRAM_DIR / record["source_file"]).read_text(encoding="utf-8"),
    )

  def test_native_exporter_reproduces_every_frozen_source(self):
    exporter = os.environ.get("REFRACTIR_TENSOR_GRAPH_EXPORT")
    if not exporter:
      self.skipTest("native graph exporter is supplied by the managed Make target")
    exporter_path = Path(exporter)
    self.assertTrue(exporter_path.is_file(), f"missing exporter: {exporter_path}")
    for record in self.manifest["programs"]:
      with self.subTest(program_id=record["program_id"]):
        result = subprocess.run(
          [
            str(exporter_path),
            "--seed",
            str(record["generation"]["seed"]),
            "--shape",
            ",".join(str(size) for size in record["generation"]["shape"]),
            "--iterations",
            str(record["generation"]["iterations"]),
          ],
          check=True,
          capture_output=True,
          text=True,
        )
        frozen = (PROGRAM_DIR / record["source_file"]).read_text(encoding="utf-8")
        self.assertEqual(result.stdout, frozen)

  def test_exact_model_replays_full_expanded_tensor_and_shape_index(self):
    from test.lib import tensor_fixed_study

    program_id = "native-seed-0-shape-2x4-iterations-3"
    record = next(
      item for item in self.manifest["programs"] if item["program_id"] == program_id
    )
    source = (PROGRAM_DIR / record["source_file"]).read_text(encoding="utf-8")
    values = (1, 2, 3, 4, 5, 6, 7, 8)
    input_bytes = struct.pack("<8f", *values)
    _proof, output_bytes = tensor_exact_f32.certify(source, input_bytes, (0, 8))
    target = tensor_fixed_study.output_element(output_bytes, (2, 4), (1, 3))

    witness = tensor_fixed_study.validate_solver_model(
      {"@kernel": {"%?x": list(values)}},
      source,
      domain=(0, 8),
      target=target,
      target_index=(1, 3),
    )

    self.assertEqual(witness["values"], values)
    self.assertEqual(witness["output_bytes"], output_bytes)

  def test_payload_solver_handles_expanded_loop_and_both_mutants(self):
    if not shutil.which("bitwuzla") and not shutil.which("z3"):
      self.skipTest("Bitwuzla or Z3 is needed for the expanded payload witness")
    program_id = "native-seed-6-shape-2x4-iterations-2"
    record = next(
      item for item in self.manifest["programs"] if item["program_id"] == program_id
    )
    source = (PROGRAM_DIR / record["source_file"]).read_text(encoding="utf-8")

    witness = tensor_fpsan.synthesize_generated_region(source)

    self.assertEqual(witness["status"], "SAT")
    self.assertEqual(witness["shape"], [2, 4])
    self.assertEqual(witness["iterations"], 2)
    self.assertTrue(
      all(
        mutant["mismatch_lanes"] for mutant in witness["calibration_mutants"].values()
      )
    )

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
      set(witness["calibration_mutants"]),
      {"lost_iteration", "lost_update_acc_y", "lane_permutation"},
    )
    self.assertTrue(
      all(
        mutant["mismatch_lanes"] for mutant in witness["calibration_mutants"].values()
      )
    )


if __name__ == "__main__":
  unittest.main()
