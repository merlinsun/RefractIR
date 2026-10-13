"""A counted loop region and its mechanically unrolled companion must agree."""

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
ITERATIONS = 2
DOMAIN = (0, 8)


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
      seed=9, shape=SHAPE, iterations=ITERATIONS, family="mixed-depth"
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


if __name__ == "__main__":
  unittest.main()
