"""Target-only tensor bundles load, refuse tampering, and round-trip through write.

The structural tests use a frontend stub when the native interpreter is not
built, so layout and digest contracts are checked without a compiled tool. The
acceptance and diagnostic tests use the native interpreter and skip when it is
absent.
"""

import copy
import hashlib
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

try:
  from test.lib import tensor_target_case
except ImportError:
  tensor_target_case = None


SYMIRI = Path(os.environ.get("REFRACTIR_SYMIRI", "./symiri")).resolve()

VALID_SOURCE = """\
fun @kernel(%x: tensor [2, 2] f32) : tensor [2, 2] f32 {
  let mut %out: tensor [2, 2] f32 = undef;
^entry:
  %out = %x;
  ret %out;
}
"""

V3_SOURCE = """\
fun @kernel(%x: tensor [3] f32) : tensor [3] f32 {
  let mut %out: tensor [3] f32 = undef;
^entry:
  %out = %x;
  ret %out;
}
"""

INVALID_SOURCE = """\
fun @kernel(%x: tensor [2, 2] f32) : tensor [4] f32 {
  let mut %out: tensor [4] f32 = undef;
^entry:
  %out = %x;
  ret %out;
}
"""

INPUT_DATA = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
V3_INPUT_DATA = struct.pack("<3f", 1.5, 2.5, 3.5)

LAYOUT = {
  "shape": [2, 2],
  "element_strides": [2, 1],
  "element_offset": 0,
  "storage_span": 4,
  "axis_order": [1, 0],
}

V3_LAYOUT = {
  "shape": [3],
  "element_strides": [1],
  "element_offset": 0,
  "storage_span": 3,
  "axis_order": [0],
}

INPUT_RECORD = {"name": "%x", "dtype": "f32", **LAYOUT, "data": INPUT_DATA}
OUTPUT_RECORD = {"name": "return", "dtype": "f32", **LAYOUT}

LAUNCH = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}

EXECUTION = {
  "guard_envelope": {"prefix": 0, "suffix": 0},
  "gap_sentinel_by_dtype": {"f32": "3f000001", "i32": "55555555"},
  "output_canary_by_dtype": {"f32": "3f000002", "i32": "aaaaaaaa"},
  "padding_by_dtype": {"f32": "3f000001", "i32": "55555555"},
  "warp_probe": {"status": "not_probed", "num_warps": 4},
}

_STUB_DIRECTORY = None

# What the native frontend describes for VALID_SOURCE.
SIGNATURE = {
  "entry": "kernel",
  "inputs": [{"name": "x", "dtype": "f32", "shape": [2, 2], "byte_length": 16}],
  "output": {"name": "return", "dtype": "f32", "shape": [2, 2], "byte_length": 16},
}


def _frontend_script(directory, signature):
  """A stub frontend that describes one stated signature."""
  executable = Path(directory) / "symiri"
  executable.write_text(
    "#!/usr/bin/env python3\n"
    "import json\n"
    f"print(json.dumps({signature!r}))\n",
    encoding="utf-8",
  )
  executable.chmod(0o755)
  return executable


def _stub_frontend():
  """A frontend that accepts every source, for tests about the bundle format."""
  global _STUB_DIRECTORY
  if _STUB_DIRECTORY is None:
    _STUB_DIRECTORY = tempfile.TemporaryDirectory()
    _frontend_script(_STUB_DIRECTORY.name, SIGNATURE)
  return Path(_STUB_DIRECTORY.name) / "symiri"


def tearDownModule():
  global _STUB_DIRECTORY
  if _STUB_DIRECTORY is not None:
    _STUB_DIRECTORY.cleanup()
    _STUB_DIRECTORY = None


def _frontend():
  """The native interpreter when it exists, else the accepting stub."""
  return SYMIRI if SYMIRI.is_file() else _stub_frontend()


def _bundle(
  root,
  *,
  source=VALID_SOURCE,
  inputs=None,
  output=None,
  launch=None,
  execution=None,
  profile="tensor-v2",
):
  """Write one valid target-only bundle before any test-specific mutation."""
  return tensor_target_case.write(
    root,
    source=source,
    inputs=[dict(INPUT_RECORD)] if inputs is None else inputs,
    output=dict(OUTPUT_RECORD) if output is None else output,
    launch=dict(LAUNCH) if launch is None else launch,
    execution=copy.deepcopy(EXECUTION) if execution is None else execution,
    profile=profile,
  )


def _edit_manifest(root, edit):
  """Rewrite case.json through one mutation, leaving its artifacts untouched."""
  path = Path(root) / "case.json"
  manifest = json.loads(path.read_text(encoding="utf-8"))
  edit(manifest)
  path.write_text(json.dumps(manifest), encoding="utf-8")
  return manifest


class TargetOnlyBundleTests(unittest.TestCase):
  """A target-only bundle states the kernel, its bytes, and the run settings."""

  def setUp(self):
    self.assertIsNotNone(tensor_target_case, "missing target-only bundle module")
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.root = Path(self.directory.name) / "case"

  def _load(self, *, symiri=None):
    return tensor_target_case.load(
      self.root, symiri=_frontend() if symiri is None else symiri
    )

  def test_a_target_only_bundle_loads_without_a_reference(self):
    written = _bundle(self.root)

    case = self._load()

    self.assertEqual(case.path, self.root.resolve())
    self.assertEqual(case.record, written.record)
    self.assertIs(case.manifest, case.record)
    self.assertEqual(case.source, VALID_SOURCE)
    self.assertEqual(case.inputs, INPUT_DATA)
    self.assertFalse((self.root / "expected.bin").exists())
    self.assertEqual(case.record["reference"], {"kind": "none"})
    self.assertEqual(case.record["comparison"], {"kind": "none"})

    manifest = case.worker_manifest()
    self.assertEqual(manifest["schema"], tensor_target_case.SCHEMA)
    self.assertEqual(manifest["kind"], "target-only")
    self.assertEqual(manifest["profile"], "tensor-v2")
    self.assertEqual(manifest["entry"], "kernel")
    self.assertEqual(manifest["grid"], [1, 1, 1])
    self.assertEqual(manifest["launch"], LAUNCH)
    self.assertEqual(manifest["execution"], EXECUTION)
    for record in manifest["inputs"]:
      for key in (
        "name",
        "dtype",
        "shape",
        "element_strides",
        "element_offset",
        "storage_span",
        "axis_order",
        "byte_offset",
        "byte_length",
      ):
        self.assertIn(key, record)
    for key in (
      "name",
      "dtype",
      "shape",
      "element_strides",
      "element_offset",
      "storage_span",
      "axis_order",
      "byte_length",
    ):
      self.assertIn(key, manifest["output"])

  def test_a_tensor_v3_bundle_loads(self):
    written = _bundle(
      self.root,
      source=V3_SOURCE,
      inputs=[{"name": "%x", "dtype": "f32", **V3_LAYOUT, "data": V3_INPUT_DATA}],
      output={"name": "return", "dtype": "f32", **V3_LAYOUT},
      profile="tensor-v3",
    )

    v3_signature = {
      "entry": "kernel",
      "inputs": [{"name": "x", "dtype": "f32", "shape": [3], "byte_length": 12}],
      "output": {"name": "return", "dtype": "f32", "shape": [3], "byte_length": 12},
    }
    frontend = (
      SYMIRI if SYMIRI.is_file() else _frontend_script(self.directory.name, v3_signature)
    )
    case = self._load(symiri=frontend)

    self.assertEqual(case.record, written.record)
    self.assertEqual(case.record["profile"], "tensor-v3")
    self.assertEqual(case.inputs, V3_INPUT_DATA)

  def test_a_tampered_program_sir_is_refused(self):
    _bundle(self.root)
    path = self.root / "program.sir"
    tampered = bytearray(path.read_bytes())
    tampered[0] ^= 0xFF
    path.write_bytes(tampered)

    with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
      self._load()

    self.assertEqual(raised.exception.status, "digest_mismatch")
    self.assertIn("program.sir", str(raised.exception))

  def test_a_tampered_inputs_bin_is_refused(self):
    _bundle(self.root)
    path = self.root / "inputs.bin"
    tampered = bytearray(path.read_bytes())
    tampered[0] ^= 0xFF
    path.write_bytes(tampered)

    with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
      self._load()

    self.assertEqual(raised.exception.status, "digest_mismatch")
    self.assertIn("inputs.bin", str(raised.exception))


  def test_the_loader_records_the_described_signature(self):
    _bundle(self.root)
    case = self._load(symiri=_frontend_script(self.directory.name, SIGNATURE))
    self.assertEqual(case.signature, SIGNATURE)

  def test_a_signature_with_another_input_count_is_refused(self):
    _bundle(self.root)
    signature = dict(SIGNATURE)
    signature["inputs"] = SIGNATURE["inputs"] + [
      {"name": "y", "dtype": "f32", "shape": [2, 2], "byte_length": 16}
    ]
    with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
      self._load(symiri=_frontend_script(self.directory.name, signature))
    self.assertEqual(raised.exception.status, "signature_mismatch")
    self.assertIn("parameters", str(raised.exception))

  def test_a_signature_with_another_parameter_type_is_refused(self):
    _bundle(self.root)
    for field, value in (
      ("dtype", "i32"),
      ("shape", [3]),
      ("name", "y"),
    ):
      with self.subTest(field=field):
        signature = copy.deepcopy(SIGNATURE)
        signature["inputs"][0][field] = value
        with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
          self._load(symiri=_frontend_script(self.directory.name, signature))
        self.assertEqual(raised.exception.status, "signature_mismatch")

  def test_a_signature_with_another_return_shape_is_refused(self):
    signature = copy.deepcopy(SIGNATURE)
    signature["output"]["shape"] = [4]
    _bundle(self.root)
    with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
      self._load(symiri=_frontend_script(self.directory.name, signature))
    self.assertEqual(raised.exception.status, "signature_mismatch")
    self.assertIn("return shape", str(raised.exception))

  def test_a_frontend_that_describes_nothing_is_refused(self):
    _bundle(self.root)
    executable = Path(self.directory.name) / "silent-symiri"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
      self._load(symiri=executable)
    self.assertEqual(raised.exception.status, "invalid_source")
    self.assertIn("did not describe", str(raised.exception))

  def test_an_input_byte_length_that_disagrees_with_its_signature_is_refused(self):
    _bundle(self.root)

    def shrink(manifest):
      manifest["inputs"][0]["byte_length"] = 12

    _edit_manifest(self.root, shrink)

    with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
      self._load()

    self.assertEqual(raised.exception.status, "invalid_layout")
    self.assertIn("byte_length", str(raised.exception))

  def test_an_input_range_past_inputs_bin_is_refused(self):
    _bundle(self.root)

    def shift(manifest):
      manifest["inputs"][0]["byte_offset"] = 8

    _edit_manifest(self.root, shift)

    with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
      self._load()

    self.assertEqual(raised.exception.status, "invalid_layout")
    self.assertIn("inputs.bin", str(raised.exception))

  def test_a_declared_reference_is_refused(self):
    _bundle(self.root)
    _edit_manifest(
      self.root,
      lambda manifest: manifest.__setitem__("reference", {"kind": "expected-bytes"}),
    )

    with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
      self._load()

    self.assertEqual(raised.exception.status, "reference_not_allowed")
    self.assertIn("reference", str(raised.exception))

  def test_a_declared_comparison_is_refused(self):
    _bundle(self.root)
    _edit_manifest(
      self.root, lambda manifest: manifest.__setitem__("comparison", {"kind": "exact"})
    )

    with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
      self._load()

    self.assertEqual(raised.exception.status, "reference_not_allowed")
    self.assertIn("comparison", str(raised.exception))

  def test_an_expected_bin_artifact_is_refused(self):
    _bundle(self.root)
    (self.root / "expected.bin").write_bytes(b"\x00" * len(INPUT_DATA))

    with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
      self._load()

    self.assertEqual(raised.exception.status, "reference_not_allowed")
    self.assertIn("expected.bin", str(raised.exception))

  def test_a_missing_frontend_is_not_silently_accepted(self):
    _bundle(self.root)
    absent = Path(self.directory.name) / "absent-symiri"

    with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
      self._load(symiri=absent)

    self.assertEqual(raised.exception.status, "frontend_unavailable")

  @unittest.skipUnless(SYMIRI.is_file(), "native frontend unavailable")
  def test_a_source_the_frontend_rejects_is_refused_with_its_diagnostic(self):
    _bundle(self.root, source=INVALID_SOURCE)

    with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
      self._load(symiri=SYMIRI)

    self.assertEqual(raised.exception.status, "invalid_source")
    self.assertIn("tensor-type", str(raised.exception))

  def test_an_output_storage_span_that_disagrees_with_its_strides_is_refused(self):
    _bundle(self.root)
    _edit_manifest(
      self.root,
      lambda manifest: manifest["output"].__setitem__("storage_span", 5),
    )

    with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
      self._load()

    self.assertEqual(raised.exception.status, "invalid_layout")
    self.assertIn("storage_span", str(raised.exception))

  def test_an_unsupported_grid_is_refused(self):
    _bundle(self.root)
    _edit_manifest(self.root, lambda manifest: manifest.__setitem__("grid", [2, 1, 1]))

    with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
      self._load()

    self.assertEqual(raised.exception.status, "unsupported_grid")

  def test_an_invalid_launch_is_refused(self):
    _bundle(self.root)

    def unsupported_warps(manifest):
      manifest["launch"]["num_warps"] = 3

    _edit_manifest(self.root, unsupported_warps)

    with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
      self._load()

    self.assertEqual(raised.exception.status, "invalid_case")
    self.assertIn("num_warps", str(raised.exception))

  def test_a_foreign_schema_or_kind_is_refused(self):
    for field, value in (
      ("schema", "refractir.tensor-case.v2"),
      ("kind", "reference-bearing"),
    ):
      with self.subTest(field=field):
        root = Path(self.directory.name) / field
        _bundle(root)
        _edit_manifest(root, lambda manifest: manifest.__setitem__(field, value))

        with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
          tensor_target_case.load(root, symiri=_frontend())

        self.assertEqual(raised.exception.status, "invalid_case")

  def test_writing_a_bundle_recomputes_digests_and_round_trips(self):
    written = _bundle(self.root)
    case = self._load()

    self.assertEqual(case.record, written.record)
    self.assertEqual(
      case.record["source"]["sha256"],
      hashlib.sha256(VALID_SOURCE.encode("utf-8")).hexdigest(),
    )
    self.assertEqual(
      case.record["inputs_bin"]["sha256"], hashlib.sha256(INPUT_DATA).hexdigest()
    )
    self.assertEqual(case.record["inputs_bin"]["byte_length"], len(INPUT_DATA))
    self.assertEqual(case.record["inputs"][0]["byte_offset"], 0)
    self.assertEqual(case.record["inputs"][0]["byte_length"], len(INPUT_DATA))
    self.assertEqual(case.record["output"]["byte_length"], len(INPUT_DATA))

    other = Path(self.directory.name) / "other"
    other_data = struct.pack("<4f", 9.0, 8.0, 7.0, 6.0)
    changed = _bundle(
      other,
      source=VALID_SOURCE + "\n",
      inputs=[{**INPUT_RECORD, "data": other_data}],
    )

    self.assertNotEqual(
      changed.record["source"]["sha256"], case.record["source"]["sha256"]
    )
    self.assertEqual(
      changed.record["inputs_bin"]["sha256"], hashlib.sha256(other_data).hexdigest()
    )
    self.assertEqual(
      tensor_target_case.load(other, symiri=_frontend()).record, changed.record
    )

  def test_the_writer_refuses_input_data_that_does_not_fill_its_layout(self):
    with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
      _bundle(self.root, inputs=[{**INPUT_RECORD, "data": INPUT_DATA[:-4]}])

    self.assertEqual(raised.exception.status, "invalid_layout")


if __name__ == "__main__":
  unittest.main()
