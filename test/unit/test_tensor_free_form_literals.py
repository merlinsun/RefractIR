"""Literal operands in free-form regions (generation plan G4); no GPU is required.

A region may use a literal wherever an operation takes a value: a
multiplicative or additive constant, a threshold in `where`, or `max`/`min`
against zero. The literal renders in place and is never declared. Behind the
builder's `literals` feature, growth may place such operations; with the
feature off the builder draws exactly as before.
"""

import os
import re
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from test.unit.tensor_support import SYMIRC, SYMIRI, check_source, run_typed
from test.unit.test_tensor_emit import emit_cli
from test.unit.triton_numpy_shim import np
from test.unit.triton_numpy_shim import run as run_shim

FREE_FORM_API = Path(
  os.environ.get("REFRACTIR_TENSOR_FREE_FORM_API", "build/bin/tensor_free_form_api")
)
FREE_FORM_EXPORT = Path(
  os.environ.get(
    "REFRACTIR_TENSOR_FREE_FORM_EXPORT", "build/bin/tensor_free_form_export"
  )
)

X = [0.5, 0.0005, -1.0, 0.001]
THRESHOLD = 2.0**-10


def f32(value):
  return struct.unpack("<f", struct.pack("<f", value))[0]


def literal_expected():
  """`2 * max(where(t < x, x, 0), 0)` with t = 2^-10, one trip from x."""
  return [f32(2.0 * max(x if THRESHOLD < f32(x) else 0.0, 0.0)) for x in map(f32, X)]


def api(name):
  return subprocess.run(
    [str(FREE_FORM_API), name], text=True, capture_output=True, check=False, timeout=30
  )


def export(seed, *features, dtype="f32"):
  with tempfile.TemporaryDirectory() as directory:
    arguments = [
      str(FREE_FORM_EXPORT),
      "--seed",
      str(seed),
      "--shape",
      "4,4",
      "--states",
      "2",
      "--trip-count",
      "2",
      "--dtype",
      dtype,
      "--out",
      directory,
    ]
    if features:
      arguments += ["--features", ",".join(features)]
    result = subprocess.run(
      arguments, text=True, capture_output=True, check=False, timeout=30
    )
    source_path = Path(directory) / "program.sir"
    source = source_path.read_text(encoding="utf-8") if source_path.exists() else None
  return result, source


def has_literal_operand(source):
  return any(
    re.search(r", -?\d+(\.\d+)?(e-?\d+)?[,;]$|tensor where \w+ -?\d", line)
    for line in source.splitlines()
    if "= tensor " in line
  )


@unittest.skipUnless(
  FREE_FORM_API.is_file() and SYMIRI.is_file(), "free-form API or symiri is unavailable"
)
class LiteralRegionTests(unittest.TestCase):
  def source(self):
    result = api("literal-region-source")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return result.stdout

  def test_literals_render_in_place_and_are_never_declared(self):
    source = self.source()
    self.assertIn("tensor where lt 0.0009765625, %acc_x, %acc_x, 0.0;", source)
    self.assertRegex(source, r"tensor max %\w+, 0\.0;")
    self.assertRegex(source, r"tensor mul %\w+, 2\.0;")
    self.assertNotIn("%lit", source)

  def test_the_literal_region_is_a_v3_and_a_v5_program(self):
    for profile in ("v3", "v5"):
      with self.subTest(profile=profile):
        result = check_source(self.source(), tensor_profile=profile)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_p0_computes_the_thresholded_relu(self):
    result, output = run_typed(
      self.source(), struct.pack("<4f", *X), tensor_profile="v5"
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(list(struct.unpack("<4f", output)), literal_expected())

  @unittest.skipUnless(
    SYMIRC.is_file() and np is not None, "symirc or numpy is unavailable"
  )
  def test_the_emitted_kernel_computes_p0s_output(self):
    kernel = emit_cli(self.source(), ("--tensor-profile-version", "v5"))[1]
    words = run_shim(kernel, [X], 4, np.float32)
    values = [struct.unpack("<f", struct.pack("<I", word))[0] for word in words]
    self.assertEqual(values, literal_expected())

  def test_a_literal_operand_bounds_its_magnitude(self):
    result = api("literal-magnitude-bound")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(
  FREE_FORM_EXPORT.is_file() and SYMIRI.is_file(),
  "the exporter or symiri is unavailable",
)
class LiteralBuilderTests(unittest.TestCase):
  def test_the_literals_feature_places_literal_operands(self):
    with_literal = 0
    for seed in range(24):
      with self.subTest(seed=seed):
        result, source = export(seed, "extrema", "where", "literals")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        checked = check_source(source, tensor_profile="v5")
        self.assertEqual(
          checked.returncode, 0, checked.stdout + checked.stderr + source
        )
        with_literal += has_literal_operand(source)
    self.assertGreaterEqual(with_literal, 12)

  def test_integer_regions_take_integer_literals(self):
    for seed in range(8):
      with self.subTest(seed=seed):
        result, source = export(seed, "literals", dtype="i32")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotRegex(source, r"= tensor [^;]*\d\.\d")
        self.assertEqual(check_source(source, tensor_profile="v5").returncode, 0)

  def test_without_the_feature_no_literal_operand_appears(self):
    for seed in range(24):
      with self.subTest(seed=seed):
        result, source = export(seed, "extrema", "where")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(has_literal_operand(source), source)


if __name__ == "__main__":
  unittest.main()
