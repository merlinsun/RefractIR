"""Pointwise regions and half-precision boundaries (generation plan G8).

The `pointwise` policy favours elementwise operations, narrowing above all, and
a `parallel_axis` request keeps one axis free of every operation that mixes
it, so grid tiling (G6) can split it and permutations of it stay exact. At the
kernel boundary an f32 input may be loaded as f16 or bf16 and widened, and an
output that is a narrowing result may be stored in its format; P0 is
unchanged because every value crossing the boundary is representable. No GPU
is required.
"""

import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from test.unit.tensor_support import SYMIRC, SYMIRI, check_source, run_typed
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

HALF = """\
fun @kernel(%x: tensor [6] f32, %y: tensor [6] f32) : tensor [6] f32 {
  let mut %p: tensor [6] f32 = undef;
  let mut %h: tensor [6] f32 = undef;
^entry:
  %p = tensor mul %x, %y;
  %p = tensor add %p, %x;
  %h = tensor narrow f16 %p;
  ret %h;
}
"""

NOT_NARROWED = HALF.replace(
  "  %h = tensor narrow f16 %p;\n", "  %h = tensor add %p, %y;\n"
)


def api(name):
  return subprocess.run(
    [str(FREE_FORM_API), name], text=True, capture_output=True, check=False, timeout=60
  )


def export(seed, directory):
  return subprocess.run(
    [
      str(FREE_FORM_EXPORT),
      "--seed",
      str(seed),
      "--shape",
      "8,4",
      "--states",
      "2",
      "--trip-count",
      "2",
      "--dtype",
      "f32",
      "--profile",
      "v5",
      "--policy",
      "pointwise",
      "--parallel-axis",
      "0",
      "--features",
      "extrema,where,narrow,scan,transpose,literals",
      "--out",
      directory,
    ],
    text=True,
    capture_output=True,
    check=False,
    timeout=30,
  )


def emit(source, *extra):
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "case.sir").write_text(source, encoding="utf-8")
    result = subprocess.run(
      [
        str(SYMIRC),
        "--target",
        "triton",
        "--entry",
        "kernel",
        "--tensor-profile-version",
        "v5",
        "--tensor-launch-json",
        str(root / "launch.json"),
        *extra,
        "-o",
        str(root / "kernel.py"),
        str(root / "case.sir"),
      ],
      capture_output=True,
      text=True,
      check=False,
      timeout=10,
    )
    kernel = (root / "kernel.py").read_text() if (root / "kernel.py").exists() else None
    launch = (
      json.loads((root / "launch.json").read_text())
      if (root / "launch.json").exists()
      else None
    )
  return result, kernel, launch


@unittest.skipUnless(FREE_FORM_API.is_file(), "free-form API is unavailable")
class PointwiseContractTests(unittest.TestCase):
  def test_every_pointwise_region_keeps_its_parallel_axis(self):
    result = api("pointwise-keeps-the-parallel-axis")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_the_pointwise_policy_favours_elementwise_operations(self):
    result = api("pointwise-policy-weights")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_a_parallel_axis_refuses_a_requirement_that_must_mix_it(self):
    result = api("parallel-axis-refuses-a-mixing-requirement")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(
  FREE_FORM_EXPORT.is_file() and SYMIRC.is_file() and SYMIRI.is_file(),
  "the exporter, symirc or symiri is unavailable",
)
class PointwiseExportTests(unittest.TestCase):
  def test_every_pointwise_export_can_be_grid_tiled(self):
    for seed in range(16):
      with self.subTest(seed=seed), tempfile.TemporaryDirectory() as directory:
        result = export(seed, directory)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        source = (Path(directory) / "program.sir").read_text(encoding="utf-8")
        self.assertEqual(check_source(source, tensor_profile="v5").returncode, 0)
        tiled, _, launch = emit(source, "--tensor-grid-block", "2")
        self.assertEqual(tiled.returncode, 0, tiled.stdout + tiled.stderr + source)
        self.assertEqual(launch["grid"], [4, 1, 1])


@unittest.skipUnless(
  SYMIRC.is_file() and SYMIRI.is_file(), "symirc or symiri is unavailable"
)
class BoundaryDtypeTests(unittest.TestCase):
  def test_half_inputs_are_widened_and_a_half_output_is_stored(self):
    result, kernel, launch = emit(
      HALF,
      "--tensor-boundary-dtype",
      "x=float16",
      "--tensor-boundary-dtype",
      "return=float16",
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertRegex(kernel, r"tl\.load\(x_ptr[^\n]*\)\.to\(tl\.float32\)")
    self.assertRegex(kernel, r"tl\.store\(out_ptr[^\n]*\.to\(tl\.float16\)")
    self.assertEqual(launch["dtypes"], {"x_ptr": "float16", "out_ptr": "float16"})

  def test_a_bf16_output_needs_a_bf16_narrowing(self):
    result, _, _ = emit(HALF, "--tensor-boundary-dtype", "return=bfloat16")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("narrow bf16", result.stdout + result.stderr)

  def test_an_output_that_is_not_a_narrowing_is_refused(self):
    result, _, _ = emit(NOT_NARROWED, "--tensor-boundary-dtype", "return=float16")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("narrow f16", result.stdout + result.stderr)

  def test_a_boundary_dtype_on_an_integer_tensor_is_refused(self):
    source = HALF.replace("f32", "i32").replace(
      "  %h = tensor narrow f16 %p;\n", "  %h = %p;\n"
    )
    result, _, _ = emit(source, "--tensor-boundary-dtype", "x=float16")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("f32 tensor", result.stdout + result.stderr)

  @unittest.skipUnless(np is not None, "numpy is unavailable")
  def test_p0s_output_narrowed_equals_the_stored_half_bytes(self):
    x = np.asarray([0.1, -2.5, 3.25, 1e-3, 7.0, -0.0], dtype=np.float16)
    y = [1.5, 0.75, -2.0, 1024.0, 0.3, 4.0]
    data = struct.pack("<6f", *x.astype(np.float32)) + struct.pack("<6f", *y)
    result, output = run_typed(HALF, data, tensor_profile="v5")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    expected = np.frombuffer(output, dtype="<f4").astype(np.float16).tobytes()
    _, kernel, _ = emit(
      HALF,
      "--tensor-boundary-dtype",
      "x=float16",
      "--tensor-boundary-dtype",
      "return=float16",
    )
    words = run_shim(kernel, [x, np.asarray(y, dtype=np.float32)], 6, np.float16)
    stored = b"".join(struct.pack("<H", word) for word in words)
    self.assertEqual(stored, expected)


if __name__ == "__main__":
  unittest.main()
