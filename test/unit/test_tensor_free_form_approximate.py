"""Division, exp and remainder in free-form regions (generation plan G5).

tensor-v5 admits the operations whose GPU results are approximate. Behind the
builder's `div`, `exp` and `rem` features, growth may place them, a divisor
included that may be zero: Triton defines a float division by zero, and the
relation guards skip non-finite lanes. P0 still calls such a lane UB, so a
seed whose P0 run fails there keeps its other outputs. No GPU is required.
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

FREE_FORM_API = Path(
  os.environ.get("REFRACTIR_TENSOR_FREE_FORM_API", "build/bin/tensor_free_form_api")
)
FREE_FORM_EXPORT = Path(
  os.environ.get(
    "REFRACTIR_TENSOR_FREE_FORM_EXPORT", "build/bin/tensor_free_form_export"
  )
)


def export(seed, *features):
  with tempfile.TemporaryDirectory() as directory:
    result = subprocess.run(
      [
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
        "f32",
        "--profile",
        "v5",
        "--features",
        ",".join(features),
        "--out",
        directory,
      ],
      text=True,
      capture_output=True,
      check=False,
      timeout=30,
    )
    source_path = Path(directory) / "program.sir"
    source = source_path.read_text(encoding="utf-8") if source_path.exists() else None
  return result, source


def api(name):
  return subprocess.run(
    [str(FREE_FORM_API), name], text=True, capture_output=True, check=False, timeout=30
  )


@unittest.skipUnless(
  FREE_FORM_API.is_file() and SYMIRI.is_file(), "free-form API or symiri is unavailable"
)
class ApproximateRegionTests(unittest.TestCase):
  def source(self):
    result = api("approximate-region-source")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return result.stdout

  def test_div_exp_and_rem_render_and_pass_v5_only(self):
    source = self.source()
    for op in ("div", "exp", "rem"):
      self.assertRegex(source, rf"= tensor {op} ")
    self.assertEqual(check_source(source, tensor_profile="v5").returncode, 0)
    self.assertNotEqual(check_source(source, tensor_profile="v3").returncode, 0)

  def test_p0_runs_a_body_with_a_nonzero_divisor(self):
    data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0) + struct.pack(
      "<4f", 2.0, 4.0, 8.0, 16.0
    )
    result, output = run_typed(self.source(), data, tensor_profile="v5")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(len(output), 16)

  def test_p0_reports_ub_for_a_zero_divisor(self):
    data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0) + struct.pack(
      "<4f", 0.0, 4.0, 8.0, 16.0
    )
    result, _ = run_typed(self.source(), data, tensor_profile="v5")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("UB", result.stdout + result.stderr)

  @unittest.skipUnless(SYMIRC.is_file(), "symirc is unavailable")
  def test_the_emitter_lowers_each_operation(self):
    result, kernel = emit_cli(self.source(), ("--tensor-profile-version", "v5"))
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn(" / ", kernel)
    self.assertIn("tl.exp(", kernel)

  def test_magnitude_bounds_are_unbounded_past_an_approximate_operation(self):
    result = api("approximate-magnitude-unbounded")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(
  FREE_FORM_EXPORT.is_file() and SYMIRI.is_file(),
  "the exporter or symiri is unavailable",
)
class ApproximateBuilderTests(unittest.TestCase):
  def test_each_feature_places_its_operation_in_some_seed(self):
    for feature in ("div", "exp", "rem"):
      with self.subTest(feature=feature):
        placed = 0
        for seed in range(16):
          result, source = export(seed, feature)
          self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
          checked = check_source(source, tensor_profile="v5")
          self.assertEqual(
            checked.returncode, 0, checked.stdout + checked.stderr + source
          )
          placed += bool(re.search(rf"= tensor {feature} ", source))
        self.assertGreaterEqual(placed, 6)

  def test_the_exporter_refuses_an_approximate_feature_outside_v5(self):
    with tempfile.TemporaryDirectory() as directory:
      result = subprocess.run(
        [
          str(FREE_FORM_EXPORT),
          "--seed",
          "1",
          "--shape",
          "4,4",
          "--dtype",
          "f32",
          "--profile",
          "v3",
          "--features",
          "div",
          "--out",
          directory,
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
      )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("v5", result.stdout + result.stderr)


if __name__ == "__main__":
  unittest.main()
