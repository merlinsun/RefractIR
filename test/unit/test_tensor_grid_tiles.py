"""Grid-tiled Triton emission (generation plan G6); no GPU is required.

Under `--tensor-grid-block B` a tensor-v5 kernel whose leading axis is
parallel (no reduction, scan, dot or transpose mixes it) runs as `cdiv(N, B)`
programs. Each program takes rows `pid * B + arange(B)` of every tensor with
that leading extent, masked by `rows < N` with `N` a runtime argument, so a
non-divisible extent leaves a partial last tile. `--tensor-launch-json` writes
the launch the kernel needs: its grid, the block constexpr, the extent
argument, and the sidecar's `launch.tiles` entry.
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

POINTWISE = """\
fun @kernel(%x: tensor [1000] f32, %y: tensor [1000] f32) : tensor [1000] f32 {
  let mut %out: tensor [1000] f32 = undef;
^entry:
  %out = tensor mul %x, %y;
  %out = tensor add %out, %x;
  ret %out;
}
"""

ROW_SUM = """\
fun @kernel(%x: tensor [300, 4] f32) : tensor [300, 1] f32 {
  let mut %s: tensor [300, 1] f32 = undef;
^entry:
  %s = tensor sum %x, 1, keep;
  ret %s;
}
"""

COLUMN_SUM = """\
fun @kernel(%x: tensor [300, 4] f32) : tensor [4] f32 {
  let mut %s: tensor [4] f32 = undef;
^entry:
  %s = tensor sum %x, 0, drop;
  ret %s;
}
"""

TRANSPOSE = """\
fun @kernel(%x: tensor [16, 4] f32) : tensor [4, 16] f32 {
  let mut %t: tensor [4, 16] f32 = undef;
^entry:
  %t = tensor transpose %x, [1, 0];
  ret %t;
}
"""


def emit(source, block, *extra):
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
        "--tensor-grid-block",
        str(block),
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


def values(seed, count):
  rng = np.random.default_rng(seed)
  return [float(np.float32(v)) for v in rng.uniform(-1, 1, count)]


def p0(source, data):
  result, output = run_typed(source, data, tensor_profile="v5")
  assert result.returncode == 0, result.stdout + result.stderr
  return list(struct.unpack(f"<{len(output) // 4}f", output))


def floats(words):
  return [struct.unpack("<f", struct.pack("<I", word))[0] for word in words]


@unittest.skipUnless(SYMIRC.is_file(), "symirc is unavailable")
class GridEmissionTests(unittest.TestCase):
  def test_a_pointwise_kernel_runs_one_program_per_tile(self):
    result, kernel, launch = emit(POINTWISE, 256)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("tl.program_id(0)", kernel)
    self.assertIn("BLOCK: tl.constexpr", kernel)
    self.assertRegex(kernel, r"< N\b")
    self.assertEqual(launch["grid"], [4, 1, 1])
    self.assertEqual(launch["constexprs"], {"BLOCK": 256})
    self.assertEqual(launch["scalars"], {"N": 1000})
    self.assertEqual(launch["tiles"], {"BLOCK": {"extent_arg": "N", "grid_axis": 0}})

  def test_a_reduction_along_the_other_axis_keeps_the_rows_parallel(self):
    result, kernel, launch = emit(ROW_SUM, 64)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(launch["grid"], [5, 1, 1])

  def test_a_reduction_across_the_leading_axis_is_refused(self):
    result, _, _ = emit(COLUMN_SUM, 64)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("sum", result.stdout + result.stderr)

  def test_a_transpose_is_refused(self):
    result, _, _ = emit(TRANSPOSE, 8)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("transpose", result.stdout + result.stderr)

  def test_a_block_that_is_not_a_power_of_two_is_refused(self):
    result, _, _ = emit(POINTWISE, 100)
    self.assertNotEqual(result.returncode, 0)


@unittest.skipUnless(
  SYMIRC.is_file() and SYMIRI.is_file() and np is not None, "tools or numpy unavailable"
)
class GridExecutionTests(unittest.TestCase):
  def test_every_block_size_computes_p0s_output(self):
    x, y = values(1, 1000), values(2, 1000)
    expected = p0(POINTWISE, struct.pack("<1000f", *x) + struct.pack("<1000f", *y))
    for block in (128, 256, 512, 1024):
      with self.subTest(block=block):
        _, kernel, launch = emit(POINTWISE, block)
        words = run_shim(
          kernel,
          [x, y],
          1000,
          np.float32,
          grid=launch["grid"],
          **launch["constexprs"],
          **launch["scalars"],
        )
        self.assertEqual(floats(words), expected)

  def test_a_partial_tile_of_rows_computes_p0s_row_sums(self):
    # Integer values keep every sum exact, so summation order cannot matter.
    x = [float(v) for v in np.random.default_rng(3).integers(-8, 9, 1200)]
    expected = p0(ROW_SUM, struct.pack("<1200f", *x))
    _, kernel, launch = emit(ROW_SUM, 64)
    words = run_shim(
      kernel,
      [x],
      300,
      np.float32,
      grid=launch["grid"],
      **launch["constexprs"],
      **launch["scalars"],
    )
    self.assertEqual(floats(words), expected)

  def test_a_runtime_loop_carries_each_programs_rows(self):
    source = """\
fun @kernel(%x: tensor [300] f32, %n: i32) : tensor [300] f32 {
  let mut %i: i32 = undef;
  let mut %acc: tensor [300] f32 = undef;
^entry:
  assume %n >= 0;
  assume %n <= 5;
  %acc = tensor mul %x, 0.0;
  %i = 0;
  br ^header;
^header:
  br %i < %n, ^body, ^exit;
^body:
  %acc = tensor add %acc, %x;
  %acc = tensor max %acc, -2.0;
  %i = %i + 1;
  br ^header;
^exit:
  ret %acc;
}
"""
    x = values(4, 300)
    _, kernel, launch = emit(source, 128)
    for n in (0, 3, 5):
      with self.subTest(n=n):
        expected = p0(source, struct.pack("<300f", *x) + struct.pack("<i", n))
        words = run_shim(
          kernel,
          [x, [n]],
          300,
          np.float32,
          grid=launch["grid"],
          **launch["constexprs"],
          **launch["scalars"],
        )
        self.assertEqual(floats(words), expected)


if __name__ == "__main__":
  unittest.main()
