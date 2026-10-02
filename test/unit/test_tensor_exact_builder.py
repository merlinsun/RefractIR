"""Authored case export refuses stale models and incomplete reference replay."""

import hashlib
import importlib
import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class ExactBuilderTests(unittest.TestCase):
  def api(self):
    return importlib.import_module("test.lib.build_tensor_exact_loop_case")

  def tools(self, command, **kwargs):
    if "--emit-model" in command:
      model = {"@kernel": {"%?x": [1, 2, 4, 8]}}
      Path(command[command.index("--emit-model") + 1]).write_text(json.dumps(model))
    if "--output" in command:
      Path(command[command.index("--output") + 1]).write_text(
        "fun @kernel() : tensor [2, 2] i32 { ret tensor [2, 2] i32 [1,2,4,8]; }\n"
      )
    elif "--tensor-output" in command:
      Path(command[command.index("--tensor-output") + 1]).write_bytes(
        struct.pack("<4f", 63, 84, 102, 126)
      )
    elif "--target" in command:
      Path(command[command.index("-o") + 1]).write_text("# test emitter output\n")
    return subprocess.CompletedProcess(command, 0, "SAT\n", "")

  def build(self, root, side_effect=None):
    api = self.api()
    with mock.patch.object(
      api.subprocess, "run", side_effect=side_effect or self.tools
    ):
      return api.build_case(
        root,
        symiri="symiri",
        symirc="symirc",
        solver="symirsolve",
        timeout_s=0.1,
      )

  def test_exports_complete_bitwise_case(self):
    with tempfile.TemporaryDirectory() as root:
      case = self.build(Path(root) / "case")
      manifest = json.loads((case / "case.json").read_text())
      self.assertEqual(manifest["comparison"]["mode"], "exact")
      self.assertEqual(
        (case / "expected.bin").read_bytes(), struct.pack("<4f", 63, 84, 102, 126)
      )
      self.assertEqual(
        manifest["reference"]["exactness"]["kind"], "composed-add-sum-f32-v1"
      )

  def test_solver_records_repeated_block_visits_and_domain(self):
    calls = []

    def run(command, **kwargs):
      calls.append(command)
      return self.tools(command, **kwargs)

    with tempfile.TemporaryDirectory() as root:
      case = self.build(Path(root) / "case", run)
      query = (case / "selection-query.sir").read_text()
    solve = next(command for command in calls if "--emit-model" in command)
    self.assertEqual(solve[solve.index("--path") + 1].count("^body"), 2)
    self.assertEqual(solve[solve.index("--main") + 1], "@kernel")
    self.assertNotIn("--emit-tensor-params", solve)
    self.assertNotIn("--tensor-param-domain", solve)
    self.assertIn("in [0, 8]", query)

  def test_archives_solved_program_as_selection_evidence(self):
    with tempfile.TemporaryDirectory() as root:
      case = self.build(Path(root) / "case")
      solved = case / "selection.solved.sir"
      manifest = json.loads((case / "case.json").read_text())
      self.assertTrue(solved.is_file())
      self.assertIn("selection.solved.sir", manifest["evidence"]["artifacts"])
      self.assertEqual(
        manifest["files"]["selection.solved.sir"]["sha256"],
        hashlib.sha256(solved.read_bytes()).hexdigest(),
      )

  def test_rejects_model_outside_frozen_domain(self):
    def run(command, **kwargs):
      result = self.tools(command, **kwargs)
      if "--emit-model" in command:
        path = Path(command[command.index("--emit-model") + 1])
        model = json.loads(path.read_text())
        model["@kernel"]["%?x"][0] = 9
        path.write_text(json.dumps(model))
      return result

    with tempfile.TemporaryDirectory() as root:
      with self.assertRaisesRegex(ValueError, "domain"):
        self.build(Path(root) / "case", run)
      self.assertFalse((Path(root) / "case/case.json").exists())

  def test_rejects_parameter_witness_shape_for_symbol_model(self):
    def run(command, **kwargs):
      result = self.tools(command, **kwargs)
      if "--emit-model" in command:
        path = Path(command[command.index("--emit-model") + 1])
        path.write_text(
          json.dumps(
            {
              "@kernel": {
                "%?x": {
                  "shape": [2, 2],
                  "element": "i32",
                  "leaves": [1, 2, 4, 8],
                }
              }
            }
          )
        )
      return result

    with tempfile.TemporaryDirectory() as root:
      with self.assertRaisesRegex(ValueError, "four i32 leaves"):
        self.build(Path(root) / "case", run)

  def test_rejects_wrong_non_target_output_element(self):
    def run(command, **kwargs):
      result = self.tools(command, **kwargs)
      if "--tensor-output" in command:
        Path(command[command.index("--tensor-output") + 1]).write_bytes(
          struct.pack("<4f", 63, 84, 102, 127)
        )
      return result

    with tempfile.TemporaryDirectory() as root:
      with self.assertRaisesRegex(ValueError, "complete f32 reference output"):
        self.build(Path(root) / "case", run)

  def test_retains_timeout_without_accepting_case(self):
    def timeout(command, **kwargs):
      raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    with tempfile.TemporaryDirectory() as root:
      with self.assertRaisesRegex(ValueError, "timed out"):
        self.build(Path(root) / "case", timeout)
      directory = Path(root) / "case"
      self.assertFalse((directory / "case.json").exists())
      self.assertEqual(
        json.loads((directory / "attempt.json").read_text())["status"], "timeout"
      )

  def _load(self, case):
    tensor_case = importlib.import_module("test.lib.tensor_case")
    signature = {
      "entry": "kernel",
      "inputs": [{"name": "x", "dtype": "f32", "shape": [2, 2], "byte_length": 16}],
      "output": {"name": "return", "dtype": "f32", "shape": [2, 2], "byte_length": 16},
    }
    with mock.patch.object(tensor_case, "_describe", return_value=signature):
      return tensor_case.load_case(case)

  def test_case_loader_accepts_complete_composed_proof_and_selection(self):
    with tempfile.TemporaryDirectory() as root:
      case = self.build(Path(root) / "case")
      loaded = self._load(case)
      self.assertEqual(loaded.manifest["solver_selection"]["domain"], [0, 8])

  def test_case_loader_rejects_missing_selection_evidence(self):
    with tempfile.TemporaryDirectory() as root:
      case = self.build(Path(root) / "case")
      manifest_path = case / "case.json"
      manifest = json.loads(manifest_path.read_text())
      del manifest["reference"]["exactness"]["selection"]
      manifest_path.write_text(json.dumps(manifest))
      with self.assertRaisesRegex(ValueError, "selection"):
        self._load(case)

  def test_case_loader_rejects_changed_solver_domain(self):
    with tempfile.TemporaryDirectory() as root:
      case = self.build(Path(root) / "case")
      manifest_path = case / "case.json"
      manifest = json.loads(manifest_path.read_text())
      manifest["reference"]["exactness"]["selection"]["domain"] = [0, 9]
      manifest_path.write_text(json.dumps(manifest))
      with self.assertRaisesRegex(ValueError, "domain"):
        self._load(case)

  def test_case_loader_rejects_tampered_query_hash(self):
    with tempfile.TemporaryDirectory() as root:
      case = self.build(Path(root) / "case")
      manifest_path = case / "case.json"
      manifest = json.loads(manifest_path.read_text())
      manifest["reference"]["exactness"]["selection"]["query_sha256"] = "0" * 64
      manifest_path.write_text(json.dumps(manifest))
      with self.assertRaisesRegex(ValueError, "query digest"):
        self._load(case)

  def test_case_loader_rejects_mismatched_solver_model(self):
    with tempfile.TemporaryDirectory() as root:
      case = self.build(Path(root) / "case")
      model_path = case / "selection-model.json"
      model = json.loads(model_path.read_text())
      model["@kernel"]["%?x"][0] = 2
      model_path.write_text(json.dumps(model))
      with self.assertRaisesRegex(ValueError, "SHA-256 digest mismatch"):
        self._load(case)


if __name__ == "__main__":
  unittest.main()
