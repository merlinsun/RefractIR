"""A counted loop region and its mechanically unrolled companion must agree."""

import hashlib
import importlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test.lib import tensor_region_generation

ROOT = Path(__file__).resolve().parents[2]


def _tool(environment_name, binary_name):
  value = os.environ.get(environment_name)
  return Path(value).resolve() if value else ROOT / "build/bin" / binary_name


SYMIRI = _tool("REFRACTIR_SYMIRI", "symiri")
SYMIRC = _tool("REFRACTIR_SYMIRC", "symirc")
SYMIR_SOLVE = _tool("REFRACTIR_SYMIR_SOLVE", "symirsolve")
TOOLS_AVAILABLE = (
  all(path.is_file() for path in (SYMIRI, SYMIRC, SYMIR_SOLVE))
  and tensor_region_generation.EXPORTER.is_file()
)


def _interpreter_available():
  """A triton that imports and a torch behind it, not just a module spec."""
  try:
    import torch  # noqa: F401
    import triton  # noqa: F401
  except Exception:
    return False
  return True


SHAPE = (8, 8)
ITERATIONS = 3
DOMAIN = (0, 8)
# Seed 1's certified reference keeps zero lanes (24 of 64), which the signed-zero
# probes below need: they flip one zero lane's sign and compare a genuine output.
SEED = 1


def _inputs(shape, value):
  count = shape[0] * shape[1]
  return struct.pack(f"<{count}f", *([float(value)] * count))


def _certified_target(exact, source, input_bytes, domain):
  """A property target the certified replay of `source` already witnesses."""
  _, expected = exact.certify(source, input_bytes, domain)
  return int(struct.unpack("<f", expected[:4])[0])


class UnrolledCompanionTests(unittest.TestCase):
  """One counted fixture; every test builds and inspects its own companion."""

  @classmethod
  def setUpClass(cls):
    if not TOOLS_AVAILABLE:
      raise unittest.SkipTest(
        "built symiri, symirc, symirsolve and the region exporter are required"
      )
    cls.regions = tensor_region_generation
    cls.exact = importlib.import_module("test.lib.tensor_exact_f32")
    cls.builder = importlib.import_module("test.lib.build_tensor_exact_region_case")
    cls.tensor_case = importlib.import_module("test.lib.tensor_case")
    cls.fixtures = tempfile.TemporaryDirectory()
    cls.configuration = cls.regions.RegionConfiguration(
      seed=SEED, shape=SHAPE, iterations=ITERATIONS, family="mixed-depth"
    )
    cls.source = cls.regions.generate_source(cls.configuration)
    cls.counted = cls.builder.build_case(
      Path(cls.fixtures.name) / "counted",
      source=cls.source,
      target=_certified_target(cls.exact, cls.source, _inputs(SHAPE, 1), DOMAIN),
      symiri=SYMIRI,
      symirc=SYMIRC,
      solver=SYMIR_SOLVE,
      domain=DOMAIN,
      configuration=cls.configuration,
      timeout_s=600.0,
    )

  @classmethod
  def tearDownClass(cls):
    cls.fixtures.cleanup()

  def setUp(self):
    self.companion_module = importlib.import_module(
      "test.lib.tensor_unrolled_companion"
    )
    self.workspace = tempfile.TemporaryDirectory()
    self.addCleanup(self.workspace.cleanup)
    self.destination = Path(self.workspace.name) / "companion"

  def _companion(self):
    return self.companion_module.build_companion(
      self.counted, self.destination, symiri=SYMIRI, symirc=SYMIRC
    )

  def _replay(self, companion, output_name):
    """Replay the companion's own source with the reference interpreter."""
    output = Path(self.workspace.name) / output_name
    result = subprocess.run(
      [
        str(SYMIRI),
        "--tensor-profile-version",
        "v2",
        "--entry",
        "kernel",
        "--tensor-input",
        str(companion / "inputs.bin"),
        "--tensor-output",
        str(output),
        str(companion / "program.sir"),
      ],
      capture_output=True,
      text=True,
      check=False,
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    return output.read_bytes()

  def _zero_sign_probe(self, case_dir):
    """Compare a genuine output against a reference with one zero lane flipped.

    The genuine bytes are the case's own reference: what a run has to produce.
    Only the reference's first positive-zero lane is tampered with, exactly as in
    the review's probe, so the verdict is a statement about the case's policy
    rather than about the buffers.
    """
    loaded = self.tensor_case.load_case(case_dir, symiri=SYMIRI)
    words = struct.unpack(f"<{len(loaded.expected) // 4}I", loaded.expected)
    lanes = [index for index, word in enumerate(words) if word == 0]
    self.assertTrue(lanes, "the fixture reference has no positive-zero lane")
    index = lanes[0]
    tampered = bytearray(loaded.expected)
    tampered[index * 4 : index * 4 + 4] = struct.pack("<I", 0x80000000)
    runner = importlib.import_module("test.lib.run_triton_tests")
    report = runner.compare_outputs(
      loaded.expected,
      bytes(tampered),
      loaded.manifest["output"]["dtype"],
      loaded.manifest["output"]["shape"],
      loaded.manifest["comparison"],
    )
    return index, report

  def test_companion_loads_through_the_case_loader(self):
    companion = self._companion()
    loaded = self.tensor_case.load_case(companion, symiri=SYMIRI)
    unrolled = self.regions.unroll_source(self.source)

    self.assertEqual(loaded.program.decode("utf-8"), unrolled)
    self.assertNotEqual(loaded.program.decode("utf-8"), self.source)
    self.assertEqual(loaded.manifest["profile"], "tensor-v2")
    self.assertEqual(loaded.manifest["entry"], "kernel")
    self.assertIsNone(loaded.manifest["reference"]["exactness"])
    self.assertEqual(
      loaded.manifest["companion"]["transformation"],
      "test.lib.tensor_region_generation.unroll_source",
    )
    self.assertEqual(loaded.inputs, (self.counted / "inputs.bin").read_bytes())
    self.assertEqual(loaded.expected, (self.counted / "expected.bin").read_bytes())
    # The derivation is checkable on its own: every counted artifact it depends
    # on is a digested file of the companion's case.
    carried = loaded.manifest["companion"]["of"]["files"]
    self.assertEqual(carried["program.sir"], "counted/program.sir")
    self.assertEqual(
      (companion / "counted" / "program.sir").read_bytes(),
      (self.counted / "program.sir").read_bytes(),
    )
    counted_manifest = json.loads(
      (self.counted / "case.json").read_text(encoding="utf-8")
    )
    self.assertEqual(
      loaded.manifest["companion"]["of"]["certificate"],
      counted_manifest["reference"]["exactness"],
    )
    for name in ("inputs.bin", "expected.bin"):
      record = loaded.manifest["files"][carried[name]]
      data = (companion / carried[name]).read_bytes()
      self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest(), name)

  def test_companion_replay_equals_the_counted_expected_bytes(self):
    companion = self._companion()
    counted_expected = (self.counted / "expected.bin").read_bytes()

    self.assertEqual(self._replay(companion, "replay.bin"), counted_expected)
    self.assertEqual((companion / "expected.bin").read_bytes(), counted_expected)
    self.assertEqual(
      (companion / "inputs.bin").read_bytes(),
      (self.counted / "inputs.bin").read_bytes(),
    )

  def test_every_companion_artifact_digest_matches_its_file(self):
    companion = self._companion()
    manifest = json.loads((companion / "case.json").read_text(encoding="utf-8"))

    self.assertLessEqual(
      {"program.sir", "inputs.bin", "expected.bin", "kernel.py"},
      set(manifest["files"]),
    )
    for name, record in manifest["files"].items():
      data = (companion / name).read_bytes()
      self.assertEqual(record["bytes"], len(data), name)
      self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest(), name)

  def test_companion_kernel_differs_and_the_runner_accepts_it(self):
    companion = self._companion()

    self.assertNotEqual(
      (companion / "kernel.py").read_bytes(), (self.counted / "kernel.py").read_bytes()
    )
    if not _interpreter_available():
      self.skipTest(
        "an importable triton and torch are required for an interpreter run"
      )
    runner = importlib.import_module("test.lib.run_triton_tests")
    report = runner.run_case(
      companion, "interpret", timeout_s=600, symiri=SYMIRI, symirc=SYMIRC
    )

    self.assertEqual(report["status"], "pass", report.get("reason"))
    self.assertEqual(
      (companion / "runs" / "interpret" / "actual.bin").read_bytes(),
      (self.counted / "expected.bin").read_bytes(),
    )

  def test_tampered_program_is_refused_by_the_loader(self):
    companion = self._companion()
    program = companion / "program.sir"
    data = bytearray(program.read_bytes())
    marker = data.index(b"%i = %i + 1;")
    data[marker + len("%i = %i + 1;") - 2] = ord("2")
    program.write_bytes(bytes(data))

    with self.assertRaisesRegex(
      self.tensor_case.CaseValidationError, "digest mismatch for program.sir"
    ):
      self.tensor_case.load_case(companion, symiri=SYMIRI)

  def test_a_changed_unrolled_form_is_refused_on_replay(self):
    """The builder trusts the replay, not the name of the transformation."""
    regions = self.companion_module.tensor_region_generation
    unroller = regions.unroll_source

    def changed(source, iterations=None):
      return unroller(source, iterations).replace("0.0, 0.0", "1.0, 1.0", 1)

    with mock.patch.object(regions, "unroll_source", changed):
      with self.assertRaises(self.companion_module.CompanionError) as raised:
        self.companion_module.build_companion(
          self.counted, self.destination, symiri=SYMIRI, symirc=SYMIRC
        )

    self.assertEqual(raised.exception.status, "replay_mismatch")
    attempt = json.loads((self.destination / "attempt.json").read_text())
    self.assertEqual(attempt["status"], "replay_mismatch")

  def test_single_trip_region_is_refused_because_unrolling_changes_nothing(self):
    """One trip runs the body once, so the unrolled text is the counted text."""
    configuration = self.regions.RegionConfiguration(
      seed=4, shape=(4, 8), iterations=1, family="cross-state"
    )
    source = self.regions.generate_source(configuration)
    counted = self.builder.build_case(
      Path(self.workspace.name) / "single-trip",
      source=source,
      target=_certified_target(self.exact, source, _inputs((4, 8), 1), (0, 2)),
      symiri=SYMIRI,
      symirc=SYMIRC,
      solver=SYMIR_SOLVE,
      domain=(0, 2),
      configuration=configuration,
      timeout_s=600.0,
    )
    destination = Path(self.workspace.name) / "single-trip-companion"

    self.assertEqual(self.regions.unroll_source(source), source)
    with self.assertRaises(self.companion_module.CompanionError) as raised:
      self.companion_module.build_companion(
        counted, destination, symiri=SYMIRI, symirc=SYMIRC
      )

    self.assertEqual(raised.exception.status, "unroll_is_identity")
    self.assertFalse(destination.exists())

  def test_a_flipped_zero_sign_in_the_reference_is_a_companion_mismatch(self):
    """The reviewer's probe: a zero lane's sign is part of the contract."""
    index, report = self._zero_sign_probe(self._companion())

    self.assertEqual(report["status"], "exact_mismatch")
    self.assertEqual([item["index"] for item in report["mismatches"]], [index])
    self.assertEqual(report["mismatches"][0]["expected_bits"], "0x80000000")
    self.assertEqual(report["mismatches"][0]["actual_bits"], "0x00000000")

  def test_the_counted_case_control_rejects_the_same_flipped_zero_reference(self):
    """The counted case is the control: both policies must reach one verdict."""
    counted_index, counted_report = self._zero_sign_probe(self.counted)
    companion_index, companion_report = self._zero_sign_probe(self._companion())

    self.assertEqual(counted_index, companion_index)
    self.assertEqual(counted_report["status"], "exact_mismatch")
    self.assertEqual(companion_report["status"], counted_report["status"])

  def test_a_genuine_companion_run_passes_byte_identically(self):
    """The bit-exact contract still admits the genuine complete output."""
    companion = self._companion()
    loaded = self.tensor_case.load_case(companion, symiri=SYMIRI)
    counted_expected = (self.counted / "expected.bin").read_bytes()

    self.assertTrue(loaded.manifest["comparison"].get("zero_sign_exact"))
    self.assertEqual(loaded.manifest["comparison"]["atol"], [0, 1])
    self.assertEqual(loaded.manifest["comparison"]["rtol"], [0, 1])
    if not _interpreter_available():
      self.skipTest(
        "an importable triton and torch are required for an interpreter run"
      )
    runner = importlib.import_module("test.lib.run_triton_tests")
    report = runner.run_case(
      companion, "interpret", timeout_s=600, symiri=SYMIRI, symirc=SYMIRC
    )
    actual = (companion / "runs" / "interpret" / "actual.bin").read_bytes()

    self.assertEqual(report["status"], "pass", report.get("reason"))
    self.assertEqual(report["comparison"]["status"], "pass")
    self.assertEqual(actual, counted_expected)
    self.assertEqual(actual, loaded.expected)

  def test_a_derivation_that_does_not_match_its_bytes_is_refused(self):
    """The loader checks every link the derivation record names."""
    companion = self._companion()
    manifest = json.loads((companion / "case.json").read_text(encoding="utf-8"))
    self.assertIsInstance(manifest.get("companion"), dict)
    path = companion / "case.json"

    def refusal(change):
      tampered = json.loads(json.dumps(manifest))
      change(tampered)
      path.write_text(json.dumps(tampered), encoding="utf-8")
      with self.assertRaises(self.tensor_case.CaseValidationError) as raised:
        self.tensor_case.load_case(companion, symiri=SYMIRI)
      return str(raised.exception)

    with self.subTest(field="counted reference digest"):
      self.assertIn(
        "counted reference",
        refusal(lambda data: data["companion"]["of"].update(reference_sha256="0" * 64)),
      )
    with self.subTest(field="replay digest"):
      self.assertIn(
        "replay 0",
        refusal(lambda data: data["companion"]["replays"][0].update(sha256="0" * 64)),
      )
    with self.subTest(field="unrolled source digest"):
      self.assertIn(
        "unrolled source",
        refusal(lambda data: data["companion"].update(unrolled_source_sha256="0" * 64)),
      )
    with self.subTest(field="trip counts"):
      self.assertIn(
        "trip counts",
        refusal(lambda data: data["companion"].update(unrolled_body_copies=1)),
      )

  def _refresh(self, companion, manifest, name):
    """Re-digest one artifact after the attack has rewritten it."""
    data = (companion / name).read_bytes()
    manifest["files"][name] = {
      "bytes": len(data),
      "sha256": hashlib.sha256(data).hexdigest(),
    }

  def test_a_changed_inherited_input_is_refused_by_the_real_loader(self):
    """The reviewer's attack on a companion the shipped builder wrote."""
    companion = self._companion()
    manifest = json.loads((companion / "case.json").read_text(encoding="utf-8"))
    data = bytearray((companion / "inputs.bin").read_bytes())
    self.assertNotEqual(struct.unpack("<f", data[0:4])[0], 7.0)
    data[0:4] = struct.pack("<f", 7.0)
    (companion / "inputs.bin").write_bytes(bytes(data))
    self._refresh(companion, manifest, "inputs.bin")
    (companion / "case.json").write_text(json.dumps(manifest), encoding="utf-8")

    with self.assertRaisesRegex(
      self.tensor_case.CaseValidationError, "not the counted runtime inputs"
    ):
      self.tensor_case.load_case(companion, symiri=SYMIRI)

  def test_a_changed_loop_bound_is_refused_by_the_real_loader(self):
    """The second reviewer attack: the bound changes, every hash is refreshed."""
    companion = self._companion()
    manifest = json.loads((companion / "case.json").read_text(encoding="utf-8"))
    source = (companion / "program.sir").read_text(encoding="utf-8")
    changed = re.sub(r"(?m)^(\s*br %i < )1(, \^body)", r"\g<1>2\g<2>", source)
    self.assertNotEqual(changed, source)
    (companion / "program.sir").write_text(changed, encoding="utf-8")
    self._refresh(companion, manifest, "program.sir")
    digest = manifest["files"]["program.sir"]["sha256"]
    manifest["source"]["sha256"] = digest
    manifest["companion"]["unrolled_source_sha256"] = digest
    (companion / "case.json").write_text(json.dumps(manifest), encoding="utf-8")

    with self.assertRaisesRegex(
      self.tensor_case.CaseValidationError, "mechanical unrolling"
    ):
      self.tensor_case.load_case(companion, symiri=SYMIRI)

  def test_a_wrong_certificate_is_refused_by_the_real_loader(self):
    """A syntactically valid certificate that does not certify the source."""
    companion = self._companion()
    manifest = json.loads((companion / "case.json").read_text(encoding="utf-8"))
    certificate = manifest["companion"]["of"]["certificate"]
    certificate["bounds"][0]["max_value"] = 999
    manifest["companion"]["of"]["certificate_sha256"] = hashlib.sha256(
      json.dumps(
        certificate, ensure_ascii=False, sort_keys=True, separators=(",", ":")
      ).encode("utf-8")
    ).hexdigest()
    (companion / "case.json").write_text(json.dumps(manifest), encoding="utf-8")

    with self.assertRaisesRegex(
      self.tensor_case.CaseValidationError, "counted certificate is not valid"
    ):
      self.tensor_case.load_case(companion, symiri=SYMIRI)

  def test_a_corrupted_carried_selection_artifact_is_refused(self):
    """The certificate check reads the carried solver artifacts, not their names."""
    companion = self._companion()
    manifest = json.loads((companion / "case.json").read_text(encoding="utf-8"))
    name = manifest["companion"]["of"]["files"]["selection-model.json"]
    model = json.loads((companion / name).read_text(encoding="utf-8"))
    model["@kernel"]["%?x"][0] = 0
    (companion / name).write_text(json.dumps(model), encoding="utf-8")
    self._refresh(companion, manifest, name)
    (companion / "case.json").write_text(json.dumps(manifest), encoding="utf-8")

    with self.assertRaisesRegex(
      self.tensor_case.CaseValidationError, "counted certificate is not valid"
    ):
      self.tensor_case.load_case(companion, symiri=SYMIRI)

  def test_a_companion_without_its_counted_artifacts_is_refused(self):
    """The smallest edge case: the carried counted copy is gone entirely."""
    companion = self._companion()
    manifest = json.loads((companion / "case.json").read_text(encoding="utf-8"))
    shutil.rmtree(companion / "counted")
    for name in list(manifest["files"]):
      if name.startswith("counted/"):
        del manifest["files"][name]
    (companion / "case.json").write_text(json.dumps(manifest), encoding="utf-8")

    with self.assertRaisesRegex(
      self.tensor_case.CaseValidationError, "counted artifacts"
    ):
      self.tensor_case.load_case(companion, symiri=SYMIRI)

  def test_the_counted_case_control_refuses_the_same_input_change(self):
    """The counted loader rejects it too, so both contracts now agree."""
    counted = Path(self.workspace.name) / "attacked-counted"
    shutil.copytree(self.counted, counted)
    manifest = json.loads((counted / "case.json").read_text(encoding="utf-8"))
    data = bytearray((counted / "inputs.bin").read_bytes())
    data[0:4] = struct.pack("<f", 7.0)
    (counted / "inputs.bin").write_bytes(bytes(data))
    self._refresh(counted, manifest, "inputs.bin")
    (counted / "case.json").write_text(json.dumps(manifest), encoding="utf-8")

    with self.assertRaises(self.tensor_case.CaseValidationError):
      self.tensor_case.load_case(counted, symiri=SYMIRI)

  def test_a_companion_without_a_derivation_cannot_claim_the_bit_exact_contract(
    self,
  ):
    """The flag is the derivation's to grant, not any v2 case's to state."""
    companion = self._companion()
    manifest = json.loads((companion / "case.json").read_text(encoding="utf-8"))
    manifest["comparison"]["zero_sign_exact"] = True
    del manifest["companion"]
    (companion / "case.json").write_text(json.dumps(manifest), encoding="utf-8")

    with self.assertRaisesRegex(self.tensor_case.CaseValidationError, "derivation"):
      self.tensor_case.load_case(companion, symiri=SYMIRI)

    counted_copy = Path(self.workspace.name) / "counted-copy"
    shutil.copytree(self.counted, counted_copy)
    counted = json.loads((counted_copy / "case.json").read_text(encoding="utf-8"))
    counted["comparison"]["zero_sign_exact"] = True
    (counted_copy / "case.json").write_text(json.dumps(counted), encoding="utf-8")
    with self.assertRaisesRegex(self.tensor_case.CaseValidationError, "derivation"):
      self.tensor_case.load_case(counted_copy, symiri=SYMIRI)

  def test_a_reference_whose_only_zero_lane_is_negative_zero_still_matches(self):
    """The smallest edge case: signed-zero equality survives the flag."""
    loaded = self.tensor_case.load_case(self._companion(), symiri=SYMIRI)
    shape = loaded.manifest["output"]["shape"]
    count = 1
    for size in shape:
      count *= size
    values = [float(index + 1) for index in range(count)]
    values[count // 2] = -0.0
    reference = struct.pack(f"<{count}f", *values)

    self.assertEqual(
      [index for index, value in enumerate(values) if value == 0.0], [count // 2]
    )
    self.assertEqual(
      struct.unpack("<I", reference[count // 2 * 4 : count // 2 * 4 + 4])[0],
      0x80000000,
    )
    runner = importlib.import_module("test.lib.run_triton_tests")
    report = runner.compare_outputs(
      reference, reference, "f32", shape, loaded.manifest["comparison"]
    )
    self.assertEqual(report["status"], "pass")

  def test_an_uncertified_counted_case_is_refused(self):
    """No counted certificate means there is no derivation to compare with."""
    uncertified = Path(self.workspace.name) / "uncertified-counted"
    shutil.copytree(self.counted, uncertified)
    manifest = json.loads((uncertified / "case.json").read_text(encoding="utf-8"))
    # The same program and reference, with the certificate replaced by a plain
    # numerical reference: still a loadable counted case, but nothing to derive
    # a bit-exact companion from.
    manifest["comparison"] = {"mode": "numeric", "atol": [0, 1], "rtol": [0, 1]}
    manifest["reference"] = {
      "defined": True,
      "stable": True,
      "precisions": [128, 256],
      "exactness": None,
    }
    (uncertified / "case.json").write_text(json.dumps(manifest), encoding="utf-8")
    self.tensor_case.load_case(uncertified, symiri=SYMIRI)
    destination = Path(self.workspace.name) / "uncertified-companion"

    with self.assertRaises(self.companion_module.CompanionError) as raised:
      self.companion_module.build_companion(
        uncertified, destination, symiri=SYMIRI, symirc=SYMIRC
      )

    self.assertEqual(raised.exception.status, "uncertified_counted_case")
    self.assertFalse(destination.exists())


if __name__ == "__main__":
  unittest.main()
