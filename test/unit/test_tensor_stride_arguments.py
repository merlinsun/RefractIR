"""Strides and offsets as kernel arguments (generation plan G7); no GPU is required.

Under `--tensor-stride-arguments` every tensor buffer's strides and element
offset are kernel arguments that default to its layout, so a launch can hand
the same kernel a transposed, row-padded or offset buffer. The launch record
names them in `strides` and `offsets`, the relations sidecar's `launch` fields.
"""

import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from test.unit.tensor_support import SYMIRC, SYMIRI, run_typed
from test.unit.triton_numpy_shim import np
from test.unit.triton_numpy_shim import run as run_shim

SOURCE = """\
fun @kernel(%x: tensor [3, 4] f32, %y: tensor [4] f32) : tensor [3, 4] f32 {
  let mut %out: tensor [3, 4] f32 = undef;
^entry:
  %out = tensor mul %x, %y;
  %out = tensor add %out, %x;
  ret %out;
}
"""

WINDOW = """\
fun @kernel(%src: tensor [8] f32, %i: i32) : tensor [4] f32 {
  let mut %w: tensor [4] f32 = undef;
^entry:
  %w = tensor window %src, %i, 0.0, [4, 1, 8];
  ret %w;
}
"""

X = [float(v) for v in range(1, 13)]
Y = [0.5, -1.0, 2.0, 0.25]


def emit(source, *extra, layouts=None):
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "case.sir").write_text(source, encoding="utf-8")
    command = [
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
    ]
    if layouts is not None:
      (root / "layout.json").write_text(
        json.dumps({"buffers": layouts}), encoding="utf-8"
      )
      command += ["--tensor-layout-manifest", str(root / "layout.json")]
    command += ["-o", str(root / "kernel.py"), str(root / "case.sir")]
    result = subprocess.run(
      command, capture_output=True, text=True, check=False, timeout=10
    )
    kernel = (root / "kernel.py").read_text() if (root / "kernel.py").exists() else None
    launch = (
      json.loads((root / "launch.json").read_text())
      if (root / "launch.json").exists()
      else None
    )
  return result, kernel, launch


def expected():
  data = struct.pack("<12f", *X) + struct.pack("<4f", *Y)
  result, output = run_typed(SOURCE, data, tensor_profile="v5")
  assert result.returncode == 0, result.stdout + result.stderr
  return list(struct.unpack("<12f", output))


def floats(words):
  return [struct.unpack("<f", struct.pack("<I", word))[0] for word in words]


@unittest.skipUnless(SYMIRC.is_file(), "symirc is unavailable")
class StrideArgumentEmissionTests(unittest.TestCase):
  def test_strides_and_offsets_become_arguments_with_row_major_defaults(self):
    result, kernel, launch = emit(SOURCE, "--tensor-stride-arguments")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("x_stride0=4, x_stride1=1, x_offset=0", kernel)
    self.assertIn("y_stride0=1, y_offset=0", kernel)
    self.assertIn("out_stride0=4, out_stride1=1, out_offset=0", kernel)
    self.assertEqual(launch["strides"]["x_ptr"], ["x_stride0", "x_stride1"])
    self.assertEqual(launch["offsets"]["out_ptr"], "out_offset")
    self.assertEqual(launch["scalars"]["x_stride0"], 4)

  def test_a_layout_manifest_sets_the_defaults(self):
    layouts = [
      {
        "name": "x",
        "role": "input",
        "element_strides": [1, 3],
        "element_offset": 1,
        "storage_span": 13,
        "axis_order": [0, 1],
      }
    ]
    result, kernel, launch = emit(SOURCE, "--tensor-stride-arguments", layouts=layouts)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("x_stride0=1, x_stride1=3, x_offset=1", kernel)
    self.assertEqual(launch["scalars"]["x_offset"], 1)

  def test_without_the_option_strides_stay_literals(self):
    result, kernel, launch = emit(SOURCE)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertNotIn("_stride0", kernel)
    self.assertNotIn("strides", launch)

  def test_a_window_is_refused_under_stride_arguments(self):
    result, _, _ = emit(WINDOW, "--tensor-stride-arguments")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("window", result.stdout + result.stderr)


@unittest.skipUnless(
  SYMIRC.is_file() and SYMIRI.is_file() and np is not None, "tools or numpy unavailable"
)
class StrideArgumentExecutionTests(unittest.TestCase):
  def test_the_default_layout_computes_p0s_output(self):
    _, kernel, launch = emit(SOURCE, "--tensor-stride-arguments")
    self.assertEqual(floats(run_shim(kernel, [X, Y], 12, np.float32)), expected())

  def test_a_transposed_buffer_passed_at_launch_computes_p0s_output(self):
    _, kernel, _ = emit(SOURCE, "--tensor-stride-arguments")
    transposed = [X[row * 4 + column] for column in range(4) for row in range(3)]
    words = run_shim(kernel, [transposed, Y], 12, np.float32, x_stride0=1, x_stride1=3)
    self.assertEqual(floats(words), expected())

  def test_an_offset_buffer_passed_at_launch_computes_p0s_output(self):
    _, kernel, _ = emit(SOURCE, "--tensor-stride-arguments")
    words = run_shim(kernel, [[99.0, *X], Y], 12, np.float32, x_offset=1)
    self.assertEqual(floats(words), expected())

  def test_stride_arguments_combine_with_grid_tiling(self):
    source = SOURCE.replace("[3, 4]", "[37, 4]")
    x = [float(v % 7) for v in range(148)]
    data = struct.pack("<148f", *x) + struct.pack("<4f", *Y)
    result, output = run_typed(source, data, tensor_profile="v5")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    _, kernel, launch = emit(
      source, "--tensor-stride-arguments", "--tensor-grid-block", "16"
    )
    transposed = [x[row * 4 + column] for column in range(4) for row in range(37)]
    words = run_shim(
      kernel,
      [transposed, Y],
      148,
      np.float32,
      grid=launch["grid"],
      **launch["constexprs"],
      N=37,
      x_stride0=1,
      x_stride1=37,
    )
    self.assertEqual(floats(words), list(struct.unpack("<148f", output)))


if __name__ == "__main__":
  unittest.main()
