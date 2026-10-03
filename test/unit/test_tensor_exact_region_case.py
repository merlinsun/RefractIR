"""Generated bounded regions must cross the ordinary case boundary exactly."""

import hashlib
import importlib
import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test.unit.test_tensor_exact_composed import generated_region_source

ROOT = Path(__file__).resolve().parents[2]


def _tool(environment_name, binary_name):
  value = os.environ.get(environment_name)
  return Path(value).resolve() if value else ROOT / "build/bin" / binary_name


SYMIRI = _tool("REFRACTIR_SYMIRI", "symiri")
SYMIRC = _tool("REFRACTIR_SYMIRC", "symirc")
SYMIR_SOLVE = _tool("REFRACTIR_SYMIR_SOLVE", "symirsolve")
TOOLS_AVAILABLE = all(path.is_file() for path in (SYMIRI, SYMIRC, SYMIR_SOLVE))


class GeneratedRegionCaseTests(unittest.TestCase):
  def setUp(self):
    self.exact = importlib.import_module("test.lib.tensor_exact_f32")
    self.builder = importlib.import_module("test.lib.build_tensor_exact_region_case")
    self.source = generated_region_source()
    self.input_bytes = struct.pack("<4f", 1, 2, 4, 8)
    self.expected = self.exact.certify(self.source, self.input_bytes, (0, 8))[1]

  def _tools(self, command, **kwargs):
    if "--emit-model" in command:
      Path(command[command.index("--emit-model") + 1]).write_text(
        json.dumps({"@kernel": {"%?x": [1, 2, 4, 8]}})
      )
    if "--output" in command:
      Path(command[command.index("--output") + 1]).write_text(
        "fun @kernel() : tensor [2, 2] i32 { ret %out; }\n"
      )
    if "--tensor-output" in command:
      Path(command[command.index("--tensor-output") + 1]).write_bytes(self.expected)
    if "--target" in command:
      Path(command[command.index("-o") + 1]).write_text("# generated kernel\n")
    return subprocess.CompletedProcess(command, 0, "SAT\n", "")

  def _build(self, root, side_effect=None):
    helper = importlib.import_module("test.lib.build_tensor_exact_loop_case")
    with mock.patch.object(
      helper.subprocess, "run", side_effect=side_effect or self._tools
    ):
      return self.builder.build_case(
        Path(root) / "case",
        source=self.source,
        target=69,
        symiri="symiri",
        symirc="symirc",
        solver="symirsolve",
        timeout_s=0.1,
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

  def _rewrite_file(self, case, name, data):
    path = case / name
    path.write_bytes(data)
    manifest_path = case / "case.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][name] = {
      "bytes": len(data),
      "sha256": hashlib.sha256(data).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

  def test_exports_and_loads_a_complete_bitwise_generated_case(self):
    with tempfile.TemporaryDirectory() as root:
      case = self._build(root)
      loaded = self._load(case)
      model = json.loads((case / "selection-model.json").read_text())
      self.assertEqual(loaded.inputs, struct.pack("<4f", *model["@kernel"]["%?x"]))
      self.assertEqual(loaded.expected, self.expected)
      self.assertEqual(loaded.manifest["comparison"]["mode"], "exact")
      self.assertEqual(
        loaded.manifest["reference"]["exactness"]["kind"],
        "bounded-region-add-sum-f32-v1",
      )

  def test_rejects_a_changed_complete_output_after_digest_refresh(self):
    with tempfile.TemporaryDirectory() as root:
      case = self._build(root)
      changed = bytearray(self.expected)
      changed[-1] ^= 1
      self._rewrite_file(case, "expected.bin", bytes(changed))
      with self.assertRaisesRegex(ValueError, "complete output bits"):
        self._load(case)

  def test_rejects_bounds_that_do_not_match_recomputation(self):
    with tempfile.TemporaryDirectory() as root:
      case = self._build(root)
      manifest_path = case / "case.json"
      manifest = json.loads(manifest_path.read_text())
      manifest["reference"]["exactness"]["bounds"][0]["max_value"] += 1
      manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
      with self.assertRaisesRegex(ValueError, "derived obligations"):
        self._load(case)

  def test_rejects_solver_query_changed_consistently_with_digests(self):
    with tempfile.TemporaryDirectory() as root:
      case = self._build(root)
      query_path = case / "selection-query.sir"
      query = query_path.read_bytes() + b"// changed\n"
      query_path.write_bytes(query)
      manifest_path = case / "case.json"
      manifest = json.loads(manifest_path.read_text())
      digest = hashlib.sha256(query).hexdigest()
      manifest["files"]["selection-query.sir"] = {"bytes": len(query), "sha256": digest}
      selection = manifest["reference"]["exactness"]["selection"]
      selection["query_sha256"] = digest
      manifest["solver_selection"]["query_sha256"] = digest
      manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
      with self.assertRaisesRegex(ValueError, "query differs"):
        self._load(case)

  def test_rejects_models_outside_the_proved_domain(self):
    def invalid_model(command, **kwargs):
      result = self._tools(command, **kwargs)
      if "--emit-model" in command:
        model_path = Path(command[command.index("--emit-model") + 1])
        model = json.loads(model_path.read_text())
        model["@kernel"]["%?x"][0] = 9
        model_path.write_text(json.dumps(model))
      return result

    with tempfile.TemporaryDirectory() as root:
      with self.assertRaisesRegex(ValueError, "outside the declared domain"):
        self._build(root, invalid_model)
      self.assertFalse((Path(root) / "case/case.json").exists())

  def test_single_visit_query_uses_its_derived_path_and_domain(self):
    source = generated_region_source(iterations=1)
    input_bytes = struct.pack("<4f", 1, 2, 3, 4)
    _, expected = self.exact.certify(source, input_bytes, (0, 4))
    target = struct.unpack("<4f", expected)[0]
    query = self.exact.render_generated_region_query(source, (0, 4), int(target))
    self.assertIn("in [0, 4]", query)
    self.assertIn(f"require %__exact_selected == {int(target)};", query)
    self.assertEqual(
      self.exact.generated_region_path(1),
      ["^entry", "^header", "^body", "^header", "^exit"],
    )

  def test_integer_twin_rewrites_positive_zero_initializers(self):
    query = self.exact.render_generated_region_query(self.source, (0, 8), 69)
    self.assertNotIn("0.0", query)
    self.assertIn("= {0, 0, 0, 0};", query)

  @unittest.skipUnless(
    TOOLS_AVAILABLE, "built symiri, symirc and symirsolve are required"
  )
  def test_native_solver_replay_and_case_loader(self):
    with tempfile.TemporaryDirectory() as root:
      case = self.builder.build_case(
        Path(root) / "case",
        source=self.source,
        target=69,
        symiri=SYMIRI,
        symirc=SYMIRC,
        solver=SYMIR_SOLVE,
      )
      loaded = importlib.import_module("test.lib.tensor_case").load_case(
        case, symiri=SYMIRI
      )
      _, independently_expected = self.exact.certify(self.source, loaded.inputs, (0, 8))
      self.assertEqual(loaded.manifest["comparison"]["mode"], "exact")
      self.assertEqual(loaded.expected, independently_expected)
      self.assertEqual(struct.unpack("<f", loaded.expected[:4])[0], 69.0)


if __name__ == "__main__":
  unittest.main()
