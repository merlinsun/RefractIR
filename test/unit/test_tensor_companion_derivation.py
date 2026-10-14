"""The loader must check a companion's derivation against the counted artifacts.

A companion earns a bit-exact comparison by deriving its reference from a counted
case. The loader is what grants that comparison, so it is where the derivation
has to hold: the companion has to carry the counted artifacts it depends on, and
the loader has to compare inherited input and reference bytes, reproduce the
unrolled source, and validate the counted certificate.

These tests build a genuine certified counted region and a companion of it with
the real region exporter, the real exact-f32 certificate and the real unroller.
Only the frontend signature check is answered from the manifest instead of a
built ``symiri``, so the module runs wherever the exporter is built; the same
behaviour is also exercised through the real ``symiri`` by
``test_tensor_unrolled_companion`` on a host that has the tools.
"""

import hashlib
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test.lib import tensor_case, tensor_exact_f32, tensor_region_generation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(os.path.relpath(Path(tensor_case.__file__).resolve().parent, Path.cwd()))
SHAPE = (8, 8)
ITERATIONS = 3
SEED = 1
DOMAIN = (0, 8)
INPUT_VALUE = 8
PRECISIONS = (128, 256)
CARRIED_SOURCE = "counted/program.sir"


def _digest(data):
  return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _write_digest(directory, manifest, name):
  """Refresh one file record after a file has been rewritten in place."""
  path = directory / name
  if path.is_file():
    manifest["files"][name] = _digest(path.read_bytes())


def _certified(seed, iterations, family):
  """One generated bounded region, plus its certified reference bytes."""
  configuration = tensor_region_generation.RegionConfiguration(
    seed=seed, shape=SHAPE, iterations=iterations, family=family
  )
  source = tensor_region_generation.generate_source(configuration)
  inputs = struct.pack(f"<{SHAPE[0] * SHAPE[1]}f", *([float(INPUT_VALUE)] * 64))
  certificate, expected = tensor_exact_f32.certify(source, inputs, DOMAIN)
  return source, inputs, certificate, expected


def _tensor_source():
  return _certified(SEED, ITERATIONS, "mixed-depth")


def _selection_evidence(source, inputs, expected):
  """The solver artifacts a counted certificate records, written here.

  The solver chooses integer leaves in the domain; the fixture's inputs are all
  the domain's upper bound, so the exact witness is those bytes read as f32 and
  converted to integers.
  """
  values = [int(value) for value in struct.unpack(f"<{len(inputs) // 4}f", inputs)]
  target = int(struct.unpack("<f", expected[:4])[0])
  command = [
    f"{ROOT}/symirsolve",
    "--tensor-profile-version",
    "v2",
    "--path",
    ",".join(tensor_exact_f32.generated_region_path(ITERATIONS)),
    "--main",
    "@kernel",
    "--output",
    "selection.solved.sir",
    "--emit-model",
    "selection-model.json",
  ]
  artifacts = {
    "selection-query.sir": tensor_exact_f32.render_generated_region_query(
      source, DOMAIN, target
    ).encode(),
    "selection.solved.sir": source.encode(),
    "selection-model.json": json.dumps({"@kernel": {"%?x": values}}).encode(),
    "selection-result.json": (
      json.dumps(
        {"status": "SAT", "exit_code": 0, "stdout": "SAT", "command": command},
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
      )
      + "\n"
    ).encode(),
  }
  # `_validate_exact_f32_output_proof` requires the selection to have been
  # checked for an admitted region certificate.
  selection = {
    "method": "bounded-integer-twin-smt",
    "domain": list(DOMAIN),
    "target": {"index": [0, 0], "value": target},
    "path": tensor_exact_f32.generated_region_path(ITERATIONS),
    "replay": "complete-f32-output-bitwise",
  }
  for key, name in (
    ("query", "selection-query.sir"),
    ("solved", "selection.solved.sir"),
    ("model", "selection-model.json"),
    ("result", "selection-result.json"),
  ):
    selection[f"{key}_file"] = name
    selection[f"{key}_sha256"] = hashlib.sha256(artifacts[name]).hexdigest()
  return artifacts, selection


def _counted_case(directory):
  """The counted side, with the certificate a bit-exact derivation needs."""
  source, inputs, certificate, expected = _tensor_source()
  selection_files, selection = _selection_evidence(source, inputs, expected)
  certificate = {**certificate, "selection": selection}
  shape = list(SHAPE)
  path = Path(directory) / "counted"
  path.mkdir(parents=True, exist_ok=True)
  (path / "program.sir").write_text(source, encoding="utf-8")
  (path / "inputs.bin").write_bytes(inputs)
  (path / "expected.bin").write_bytes(expected)
  for name, data in selection_files.items():
    (path / name).write_bytes(data)
  files = {
    name: _digest((path / name).read_bytes())
    for name in sorted(
      {"program.sir", "inputs.bin", "expected.bin"} | set(selection_files)
    )
  }
  buffer = {
    "byte_offset": 0,
    "byte_length": len(inputs),
    "dtype": "f32",
    "shape": shape,
    "element_strides": [shape[1], 1],
    "element_offset": 0,
    "storage_span": len(inputs) // 4,
    "axis_order": [1, 0],
    "tile_shape": shape,
    "tile_origin": [0, 0],
  }
  manifest = {
    "schema": tensor_case.SCHEMA_V2,
    "profile": tensor_case.PROFILE_V2,
    "entry": "kernel",
    "source": {
      "file": "program.sir",
      "sha256": files["program.sir"]["sha256"],
      "generator": "generated-bounded-region",
      "seed": SEED,
    },
    "inputs": [{"name": "x", "file": "inputs.bin", **buffer}],
    "output": {"name": "return", "file": "expected.bin", **buffer},
    "grid": [1, 1, 1],
    "softmax_lowering": "direct",
    "launch": {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False},
    "property": {"id": "companion-derivation-fixture", "source": "fixture"},
    "evidence": {
      "kind": "exact_smt",
      "status": "exact_smt_checked",
      "artifacts": sorted(files),
    },
    "comparison": {"mode": "exact", "atol": [0, 1], "rtol": [0, 1]},
    "reference": {
      "defined": True,
      "stable": True,
      "precisions": [],
      "exactness": certificate,
    },
    "execution": {
      "full_shapes": [shape],
      "range_extents": [shape],
      "warp_probe": {"status": "not_probed", "num_warps": 4},
      "loop_bounds": [ITERATIONS],
      "total_body_executions": ITERATIONS,
      "gap_sentinel_by_dtype": {"f32": "3f000001", "i32": "55555555"},
      "output_canary_by_dtype": {"f32": "3f000002", "i32": "aaaaaaaa"},
    },
    "coverage": {
      "primary_stratum": "generated-bounded-region",
      "source_signature": hashlib.sha256(source.encode()).hexdigest(),
      "loop_structure": [{"bound": ITERATIONS}],
      "input_behavior": {
        "property_passed": True,
        "pattern": "smt-selected",
        "input_mode": "bounded-integer-twin-smt",
        "policy": "exact",
        "input_domain": {"lo": DOMAIN[0], "hi": DOMAIN[1]},
        "distinguishing_objective": "output-element-equality",
      },
    },
    "environment": None,
    "operator_coverage": tensor_exact_f32.generated_region_operations(source),
    "shape_axes": [],
    "files": files,
  }
  (path / "case.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
  )
  return path


def _counted_files(certificate):
  """Counted artifact names the certificate and the runtime buffers reference."""
  names = {"program.sir", "inputs.bin", "expected.bin"}

  def walk(value):
    if isinstance(value, dict):
      for key, item in value.items():
        if isinstance(item, str) and key.endswith("_file"):
          names.add(item)
        elif not isinstance(item, str):
          walk(item)
    elif isinstance(value, list):
      for item in value:
        walk(item)

  walk(certificate)
  return names


def _build_companion(counted, destination, *, input_bits=None):
  """The companion ``build_companion`` would write, without the tools.

  The carried counted artifacts, the unrolled source and the record of the
  derivation mirror the builder's own output; only the kernel and the reference
  replays a Triton host would produce are absent. ``input_bits`` rewrites the
  inherited runtime input to that many lanes, with the records that describe it,
  which gives the byte comparison a counted input of another length.
  """
  counted = Path(counted)
  destination = Path(destination)
  counted_manifest = json.loads((counted / "case.json").read_text(encoding="utf-8"))
  certificate = counted_manifest["reference"]["exactness"]
  counted_program = (counted / "program.sir").read_bytes()
  inputs = (counted / "inputs.bin").read_bytes()
  records = {"inputs": [dict(item) for item in counted_manifest["inputs"]]}
  if input_bits is not None:
    inputs = struct.pack(f"<{input_bits}f", *([float(INPUT_VALUE)] * input_bits))
    shape = [input_bits // SHAPE[0], SHAPE[1]]
    record = records["inputs"][0]
    record["byte_length"] = len(inputs)
    record["shape"] = shape
    record["tile_shape"] = shape
    record["storage_span"] = input_bits
    record["element_strides"] = [shape[1], 1]
  expected = (counted / "expected.bin").read_bytes()
  unrolled = tensor_region_generation.unroll_source(counted_program.decode("utf-8"))
  destination.mkdir(parents=True, exist_ok=True)
  (destination / "program.sir").write_text(unrolled, encoding="utf-8")
  (destination / "inputs.bin").write_bytes(inputs)
  (destination / "expected.bin").write_bytes(expected)
  carried = {}
  for name in sorted(
    {"program.sir", "inputs.bin", "expected.bin"} | _counted_files(certificate)
  ):
    carried[name] = f"counted/{name}"
    target = destination / "counted" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(counted / name, target)
  files = {
    "program.sir": _digest(unrolled.encode()),
    "inputs.bin": _digest(inputs),
    "expected.bin": _digest(expected),
  }
  for name, retained in carried.items():
    files[retained] = _digest((destination / retained).read_bytes())
  for bits in PRECISIONS:
    name = f"reference-{bits}.bin"
    (destination / name).write_bytes(expected)
    files[name] = _digest(expected)
  manifest = {
    "schema": tensor_case.SCHEMA_V2,
    "profile": counted_manifest["profile"],
    "entry": counted_manifest["entry"],
    "source": {
      "file": "program.sir",
      "sha256": files["program.sir"]["sha256"],
      "generator": "unrolled-companion",
      "seed": counted_manifest.get("source", {}).get("seed", 0),
    },
    # The derivation record carries the counted certificate itself, so the
    # loader can validate it the way the counted path does.
    "companion": {
      "of": {
        "entry": counted_manifest["entry"],
        "profile": counted_manifest["profile"],
        "program_sha256": hashlib.sha256(counted_program).hexdigest(),
        "manifest_sha256": hashlib.sha256(
          (counted / "case.json").read_bytes()
        ).hexdigest(),
        "certificate_sha256": hashlib.sha256(
          json.dumps(
            certificate, ensure_ascii=False, sort_keys=True, separators=(",", ":")
          ).encode("utf-8")
        ).hexdigest(),
        "reference_sha256": hashlib.sha256(expected).hexdigest(),
        "certificate": certificate,
        "operator_coverage": counted_manifest["operator_coverage"],
        "files": carried,
      },
      "transformation": "test.lib.tensor_region_generation.unroll_source",
      "counted_trips": ITERATIONS,
      "unrolled_trips": 1,
      "unrolled_body_copies": ITERATIONS,
      "unrolled_source_sha256": files["program.sir"]["sha256"],
      "inherited_files": ["inputs.bin"],
      "reference": "interpreter-replay-unrolled",
      "replays": [
        {
          "reference_bits": bits,
          "file": f"reference-{bits}.bin",
          "sha256": hashlib.sha256(expected).hexdigest(),
          "command": [
            f"{ROOT}/symiri",
            "--tensor-profile-version",
            "v2",
            "--entry",
            counted_manifest["entry"],
            "--tensor-input",
            str(destination / "inputs.bin"),
            "--tensor-output",
            str(destination / f"reference-{bits}.bin"),
            str(destination / "program.sir"),
          ],
        }
        for bits in PRECISIONS
      ],
    },
    "inputs": records["inputs"],
    "output": counted_manifest["output"],
    "grid": counted_manifest["grid"],
    "softmax_lowering": counted_manifest["softmax_lowering"],
    "launch": counted_manifest["launch"],
    "property": counted_manifest["property"],
    "evidence": {
      "kind": "numerical_reference",
      "status": "numerical_reference_checked",
      "artifacts": sorted(files),
    },
    "comparison": {
      "mode": "numeric",
      "policy": "derived-counted-certificate-bit-exact",
      "atol": [0, 1],
      "rtol": [0, 1],
      "zero_sign_exact": True,
    },
    "reference": {
      "defined": True,
      "stable": True,
      "precisions": list(PRECISIONS),
      "exactness": None,
    },
    "execution": counted_manifest["execution"],
    "coverage": counted_manifest["coverage"],
    "environment": None,
    "operator_coverage": counted_manifest["operator_coverage"],
    "shape_axes": [],
    "files": files,
  }
  (destination / "case.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
  )
  return destination


def _stub_describe(manifest):
  """The frontend signature of a case without running ``symiri``."""
  inputs = [
    {
      "name": item["name"],
      "dtype": item["dtype"],
      "shape": item["shape"],
      "byte_length": item["byte_length"],
    }
    for item in manifest["inputs"]
  ]
  output = manifest["output"]
  return {
    "entry": manifest["entry"],
    "inputs": inputs,
    "output": {key: output[key] for key in ("name", "dtype", "shape", "byte_length")},
  }


def _rewrite_carried_source(companion, manifest):
  """Append a byte to the carried counted source, refreshing its own record."""
  path = companion / CARRIED_SOURCE
  path.write_bytes(path.read_bytes() + b"\n")
  _write_digest(companion, manifest, CARRIED_SOURCE)


class CompanionDerivationTests(unittest.TestCase):
  """One certified counted fixture; each test attacks its companion."""

  @classmethod
  def setUpClass(cls):
    if not tensor_region_generation.EXPORTER.is_file():
      raise unittest.SkipTest("the built region exporter is required")
    cls.workspace = tempfile.TemporaryDirectory()
    cls.counted = _counted_case(cls.workspace.name)
    cls.load = tensor_case.load_case

  @classmethod
  def tearDownClass(cls):
    cls.workspace.cleanup()

  def setUp(self):
    self.root = Path(tempfile.mkdtemp(dir=self.workspace.name))

  def tearDown(self):
    shutil.rmtree(self.root, ignore_errors=True)

  def _load(self, root, manifest):
    """Load one case with only the frontend signature answered from JSON."""
    with mock.patch.object(
      tensor_case, "_describe", return_value=_stub_describe(manifest)
    ):
      return tensor_case.load_case(root)

  def _companion(self, *, input_bits=None):
    return _build_companion(
      self.counted, self.root / "companion", input_bits=input_bits
    )

  def _attacked(self, change, *, input_bits=None):
    """Apply ``change`` to a fresh companion and return the loader's refusal."""
    companion = self._companion(input_bits=input_bits)
    manifest = json.loads((companion / "case.json").read_text(encoding="utf-8"))
    change(companion, manifest)
    (companion / "case.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    with self.assertRaises(tensor_case.CaseValidationError) as raised:
      self._load(companion, manifest)
    return str(raised.exception)

  def test_a_genuine_companion_loads_and_keeps_its_bit_exact_comparison(self):
    """The positive control: the derivation admits the case it describes."""
    companion = self._companion()
    manifest = json.loads((companion / "case.json").read_text(encoding="utf-8"))
    loaded = self._load(companion, manifest)

    self.assertTrue(loaded.manifest["comparison"]["zero_sign_exact"])
    self.assertEqual(loaded.manifest["comparison"]["atol"], [0, 1])
    self.assertEqual(loaded.inputs, (self.counted / "inputs.bin").read_bytes())
    self.assertEqual(loaded.expected, (self.counted / "expected.bin").read_bytes())
    self.assertEqual(
      loaded.program.decode("utf-8"),
      tensor_region_generation.unroll_source(
        (self.counted / "program.sir").read_text(encoding="utf-8")
      ),
    )

  def test_a_shorter_inherited_input_is_refused(self):
    """The same attack when the stale input also changes the byte count."""
    error = self._attacked(lambda companion, manifest: None, input_bits=32)
    self.assertIn("runtime inputs", error)

  def test_a_changed_inherited_input_is_refused(self):
    """The reviewer's attack: change input [0,0] from 8 to 7, refresh digests."""

    def change(companion, manifest):
      data = bytearray((companion / "inputs.bin").read_bytes())
      data[0:4] = struct.pack("<f", 7.0)
      (companion / "inputs.bin").write_bytes(bytes(data))
      _write_digest(companion, manifest, "inputs.bin")

    error = self._attacked(change)
    self.assertIn("runtime inputs", error)

  def test_a_changed_inherited_input_of_another_length_is_refused(self):
    """The same attack on a shorter counted input, byte count refreshed too."""

    def change(companion, manifest):
      data = bytearray((companion / "inputs.bin").read_bytes())
      data[0:4] = struct.pack("<f", 7.0)
      (companion / "inputs.bin").write_bytes(bytes(data))
      _write_digest(companion, manifest, "inputs.bin")

    error = self._attacked(change, input_bits=32)
    self.assertIn("runtime inputs", error)

  def test_the_same_input_change_on_the_counted_case_is_refused_by_its_proof(self):
    """The counted control: its certificate no longer matches its inputs."""
    manifest = json.loads((self.counted / "case.json").read_text(encoding="utf-8"))
    data = bytearray((self.counted / "inputs.bin").read_bytes())
    data[0:4] = struct.pack("<f", 7.0)
    (self.counted / "inputs.bin").write_bytes(bytes(data))
    _write_digest(self.counted, manifest, "inputs.bin")
    (self.counted / "case.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    self.addCleanup(self._restore_counted)

    with self.assertRaises(tensor_case.CaseValidationError) as raised:
      self._load(self.counted, manifest)
    self.assertIn("complete output bits", str(raised.exception))

  def _restore_counted(self):
    """Rebuild the shared counted fixture after an in-place attack on it."""
    shutil.rmtree(self.counted)
    type(self).counted = _counted_case(self.workspace.name)

  def test_a_changed_loop_bound_is_refused(self):
    """The second reviewer attack: change the bound, refresh the source hashes."""

    def change(companion, manifest):
      source = (companion / "program.sir").read_text(encoding="utf-8")
      source = re.sub(r"(?m)^(\s*br %i < )1(, \^body)", r"\g<1>2\g<2>", source)
      (companion / "program.sir").write_text(source, encoding="utf-8")
      digest = _digest(source.encode())
      manifest["files"]["program.sir"] = digest
      manifest["source"]["sha256"] = digest["sha256"]
      manifest["companion"]["unrolled_source_sha256"] = digest["sha256"]

    error = self._attacked(change)
    self.assertIn("unrolling", error)

  def test_a_wrong_certificate_is_refused(self):
    """A syntactically valid certificate that does not certify the source."""

    def change(companion, manifest):
      certificate = manifest["companion"]["of"]["certificate"]
      certificate["bounds"][0]["max_value"] = 999
      manifest["companion"]["of"]["certificate_sha256"] = hashlib.sha256(
        json.dumps(
          certificate, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
      ).hexdigest()

    error = self._attacked(change)
    self.assertIn("certificate", error)

  def test_a_companion_without_its_counted_artifacts_is_refused(self):
    """The smallest edge case: the counted copy is gone entirely."""

    def change(companion, manifest):
      shutil.rmtree(companion / "counted")
      for name in list(manifest["files"]):
        if name.startswith("counted/"):
          del manifest["files"][name]

    error = self._attacked(change)
    self.assertIn("counted artifacts", error)

  def test_a_carried_counted_file_of_other_bytes_is_refused(self):
    """A carried counted artifact whose bytes do not match its own record."""
    error = self._attacked(_rewrite_carried_source)
    self.assertIn("counted source", error)


if __name__ == "__main__":
  unittest.main()
