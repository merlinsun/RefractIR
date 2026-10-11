"""Exact admission and replay for the configured long-axis region families."""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from test.lib import tensor_exact_f32, tensor_region_generation
from test.lib.tensor_region_generation import RegionConfiguration

ROOT = Path(__file__).resolve().parents[2]


def _tool(environment_name, binary_name):
  value = os.environ.get(environment_name)
  return Path(value).resolve() if value else ROOT / "build" / "bin" / binary_name


SYMIRI = _tool("REFRACTIR_SYMIRI", "symiri")
EXPORTER = tensor_region_generation.EXPORTER
READY = EXPORTER.is_file() and SYMIRI.is_file()


def _inputs(shape, domain_hi=2):
  count = shape[0] * shape[1]
  values = [float(1 + (index % domain_hi)) for index in range(count)]
  return struct.pack(f"<{count}f", *values)


def _replay(source, input_bytes):
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "program.sir").write_text(source, encoding="utf-8")
    (root / "inputs.bin").write_bytes(input_bytes)
    result = subprocess.run(
      [
        str(SYMIRI),
        "--tensor-profile-version",
        "v2",
        "--entry",
        "kernel",
        "--tensor-input",
        str(root / "inputs.bin"),
        "--tensor-output",
        str(root / "output.bin"),
        str(root / "program.sir"),
      ],
      capture_output=True,
      text=True,
      check=False,
    )
    if result.returncode != 0:
      raise AssertionError(f"reference replay failed: {result.stderr.strip()}")
    return (root / "output.bin").read_bytes()


class TensorRegionExactFamilyTests(unittest.TestCase):
  def setUp(self):
    if not READY:
      self.skipTest("the exporter and interpreter come from the managed Make target")

  def test_long_axis_family_receives_a_full_certificate(self):
    # Two trips over a 256-wide reduction stay inside the exact range; a third
    # trip is rejected below, so the admitted configuration states its bound.
    configuration = RegionConfiguration(
      seed=4, shape=(256, 2), iterations=2, family="orthogonal-reductions"
    )
    source = tensor_region_generation.generate_source(configuration)
    input_bytes = _inputs(configuration.shape)

    proof, output = tensor_exact_f32.certify(source, input_bytes, (0, 2))

    self.assertEqual(proof["kind"], tensor_exact_f32.REGION_KIND)
    self.assertTrue(proof["complete_output"])
    self.assertEqual(len(output), 512 * 4)
    self.assertEqual(proof["integer_exactness_bound"], 1 << 24)
    self.assertEqual(proof["signed_zero"], "positive-zero-only")
    self.assertEqual(len(proof["path"]), 2 + 2 * configuration.iterations + 1)
    self.assertEqual(_replay(source, input_bytes), output)

  def test_admitted_reduction_partials_fit_the_exactness_bound(self):
    configuration = RegionConfiguration(
      seed=4, shape=(256, 2), iterations=2, family="orthogonal-reductions"
    )
    source = tensor_region_generation.generate_source(configuration)

    proof, _output = tensor_exact_f32.certify(
      source, _inputs(configuration.shape), (0, 2)
    )
    reductions = [bound for bound in proof["bounds"] if bound["operation"] == "sum"]

    self.assertTrue(reductions, "a reduction family must record reduction bounds")
    for bound in reductions:
      self.assertIn("max_partial_sum", bound)
      self.assertLessEqual(bound["max_partial_sum"], proof["integer_exactness_bound"])
      self.assertLessEqual(bound["max_value"], bound["max_partial_sum"])

  def test_overflowing_recurrence_is_rejected(self):
    configuration = RegionConfiguration(
      seed=4, shape=(256, 2), iterations=3, family="orthogonal-reductions"
    )
    source = tensor_region_generation.generate_source(configuration)
    # The third trip multiplies the admitted two-trip bound by another 256-wide
    # reduction, so the recurrence leaves the exact range.
    input_bytes = _inputs(configuration.shape, domain_hi=2)

    with self.assertRaisesRegex(ValueError, "exceeds exact range"):
      tensor_exact_f32.certify(source, input_bytes, (0, 2))

  def test_subtraction_family_enforces_signed_zero_obligations(self):
    configuration = RegionConfiguration(
      seed=4,
      shape=(4, 8),
      iterations=2,
      family="cross-state",
      operation_mode="subtraction",
    )
    source = tensor_region_generation.generate_source(configuration)
    input_bytes = _inputs(configuration.shape)

    proof, output = tensor_exact_f32.certify(source, input_bytes, (0, 2))

    self.assertEqual(proof["signed_zero"], "positive-zero-only")
    self.assertIn("sub", proof["allowed_operations"])
    self.assertEqual(_replay(source, input_bytes), output)

    negative_zero = struct.pack("<32f", *([1.0] * 31 + [0.0]))
    negative_zero = negative_zero[:-4] + struct.pack("<I", 0x80000000)
    with self.assertRaisesRegex(ValueError, "positive zero"):
      tensor_exact_f32.certify(source, negative_zero, (0, 2))

  def test_unrolled_form_replays_to_the_same_full_output(self):
    configuration = RegionConfiguration(
      seed=4, shape=(32, 16), iterations=3, family="mixed-depth"
    )
    source = tensor_region_generation.generate_source(configuration)
    unrolled = tensor_region_generation.unroll_source(source)
    input_bytes = _inputs(configuration.shape)

    proof, expected = tensor_exact_f32.certify(source, input_bytes, (0, 2))

    self.assertTrue(proof["complete_output"])
    self.assertEqual(_replay(source, input_bytes), expected)
    self.assertEqual(_replay(unrolled, input_bytes), expected)
    self.assertNotEqual(unrolled, source)


class TensorRegionExactFamilyEdgeTests(unittest.TestCase):
  def setUp(self):
    if not READY:
      self.skipTest("the exporter and interpreter come from the managed Make target")

  def test_altered_domain_is_rejected(self):
    configuration = RegionConfiguration(
      seed=4, shape=(4, 8), iterations=2, family="cross-state"
    )
    source = tensor_region_generation.generate_source(configuration)
    input_bytes = _inputs(configuration.shape)

    for domain in ((-1, 2), (0, 1 << 25), (0, 2.0)):
      with self.subTest(domain=domain):
        with self.assertRaisesRegex(ValueError, "nonnegative bounded integers"):
          tensor_exact_f32.certify(source, input_bytes, domain)


if __name__ == "__main__":
  unittest.main()
