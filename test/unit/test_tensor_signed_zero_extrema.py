"""Triton extrema keep SIR's signed-zero tie-break; no GPU is required.

SIR resolves equal-magnitude signed zeros in `min`, `max`, `rmin`, and `rmax`
with the IEEE minNum/maxNum rule: a -0.0 operand forces a -0.0 minimum and a
+0.0 operand forces a +0.0 maximum. Triton leaves that tie unspecified, so the
emitted kernel must fix it itself. These tests run the emitted kernel against
a numpy stand-in for `triton.language` whose extrema return the wrong signed
zero on every tie, so only a lowering that does not depend on the tie passes.
"""

import struct
import sys
import tempfile
import unittest
from pathlib import Path

from test.unit.tensor_support import SYMIRC
from test.unit.test_tensor_emit import emit_cli
from test.unit.triton_numpy_shim import np
from test.unit.triton_numpy_shim import run as _run


def _bits(values):
  return [struct.unpack("<I", struct.pack("<f", value))[0] for value in values]


PZ, NZ = 0.0, -0.0


def _maximum(a, b):
  """SIR's max: a zero tie gives +0.0 unless both zeros are negative."""
  return a + b if a == 0 and b == 0 else max(a, b)


def _minimum(a, b):
  """SIR's min: a zero tie gives -0.0 unless both zeros are positive."""
  return -((-a) + (-b)) if a == 0 and b == 0 else min(a, b)


ELEMENTWISE = """\
fun @kernel(%x: tensor [8] {dtype}, %y: tensor [8] {dtype}) : tensor [8] {dtype} {{
  let mut %out: tensor [8] {dtype} = undef;
^entry:
  %out = tensor {op} %x, %y;
  ret %out;
}}
"""

REDUCTION = """\
fun @kernel(%x: tensor [4, 4] f32) : tensor [4, 1] f32 {{
  let mut %out: tensor [4, 1] f32 = undef;
^entry:
  %out = tensor {op} %x, 1, keep;
  ret %out;
}}
"""

V3_REDUCTION = """\
fun @kernel(%x: tensor [3] f32) : tensor [1] f32 {{
  let mut %out: tensor [1] f32 = undef;
^entry:
  %out = tensor {op} %x, 0, keep;
  ret %out;
}}
"""

# A rank-0 extremum result meeting a tile, the shape of the W1f interpreter
# crash: Triton's interpreter types the scalar's `== 0.0` as float32, so the
# tie must not combine the two comparisons with `&`.
SCALAR_TILE = """\
fun @kernel(%v: tensor [4] f32, %t: tensor [4, 4] f32) : tensor [4, 4] f32 {{
  let mut %s: f32 = undef;
  let mut %out: tensor [4, 4] f32 = undef;
^entry:
  %s = tensor {reduction} %v, 0, drop;
  %out = tensor {op} %s, %t;
  ret %out;
}}
"""

# The tie's comparisons joined by `&`, which the interpreter rejects there.
JOINED = "== 0.0) & ("

LEFT = [PZ, NZ, NZ, PZ, 1.0, -3.0, 5.0, NZ]
RIGHT = [NZ, PZ, NZ, PZ, 2.0, NZ, 5.0, -1.0]
ROWS = [[NZ, PZ, NZ, NZ], [NZ, NZ, NZ, NZ], [-1.0, NZ, -2.0, -4.0], [3.0, PZ, NZ, 1.0]]
TILE = [value for row in ROWS for value in row]


@unittest.skipUnless(
  SYMIRC.is_file() and np is not None, "symirc or numpy is unavailable"
)
class SignedZeroExtremaTests(unittest.TestCase):
  def emit(self, source, *arguments):
    result, output = emit_cli(source, arguments)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return output

  def test_f32_max_resolves_a_zero_tie_to_positive_zero(self):
    kernel = self.emit(ELEMENTWISE.format(dtype="f32", op="max"))
    self.assertEqual(
      _run(kernel, [LEFT, RIGHT], 8), _bits([PZ, PZ, NZ, PZ, 2.0, NZ, 5.0, NZ])
    )

  def test_f32_min_resolves_a_zero_tie_to_negative_zero(self):
    kernel = self.emit(ELEMENTWISE.format(dtype="f32", op="min"))
    self.assertEqual(
      _run(kernel, [LEFT, RIGHT], 8), _bits([NZ, NZ, NZ, PZ, 1.0, -3.0, 5.0, -1.0])
    )

  def test_f32_rmax_returns_positive_zero_when_any_zero_lane_is_positive(self):
    kernel = self.emit(REDUCTION.format(op="rmax"))
    flat = [value for row in ROWS for value in row]
    self.assertEqual(_run(kernel, [flat], 4), _bits([PZ, NZ, NZ, 3.0]))

  def test_f32_rmin_returns_negative_zero_when_any_zero_lane_is_negative(self):
    kernel = self.emit(REDUCTION.format(op="rmin"))
    flat = [value for row in ROWS for value in row]
    self.assertEqual(_run(kernel, [flat], 4), _bits([NZ, NZ, -4.0, NZ]))

  def test_v3_masked_rmax_keeps_the_tie_break_beside_its_neutral_lanes(self):
    kernel = self.emit(V3_REDUCTION.format(op="rmax"), "--tensor-profile-version", "v3")
    self.assertEqual(_run(kernel, [[NZ, PZ, NZ]], 1), _bits([PZ]))
    self.assertEqual(_run(kernel, [[NZ, NZ, NZ]], 1), _bits([NZ]))

  def test_the_max_tie_joins_no_comparisons_with_and(self):
    self.assertNotIn(JOINED, self.emit(ELEMENTWISE.format(dtype="f32", op="max")))

  def test_the_min_tie_joins_no_comparisons_with_and(self):
    self.assertNotIn(JOINED, self.emit(ELEMENTWISE.format(dtype="f32", op="min")))

  def test_a_rank_zero_max_against_a_tile_joins_no_comparisons(self):
    source = SCALAR_TILE.format(reduction="rmax", op="max")
    self.assertNotIn(JOINED, self.emit(source, "--tensor-profile-version", "v3"))

  def test_a_rank_zero_min_against_a_tile_joins_no_comparisons(self):
    source = SCALAR_TILE.format(reduction="rmin", op="min")
    self.assertNotIn(JOINED, self.emit(source, "--tensor-profile-version", "v3"))

  def test_the_tie_nests_one_zero_test_inside_the_other(self):
    kernel = self.emit(ELEMENTWISE.format(dtype="f32", op="max"))
    self.assertRegex(
      kernel,
      r"tl\.where\((\w+) == 0\.0, tl\.where\((\w+) == 0\.0, \(\1 \+ \2\), "
      r"tl\.maximum\(\1, \2\)\), tl\.maximum\(\1, \2\)\)",
    )

  def test_a_rank_zero_max_keeps_the_tie_break_against_a_tile(self):
    kernel = self.emit(
      SCALAR_TILE.format(reduction="rmax", op="max"), "--tensor-profile-version", "v3"
    )
    # rmax of four negative zeros is -0.0, so a +0.0 tile lane wins its tie.
    self.assertEqual(
      _run(kernel, [[NZ] * 4, TILE], 16), _bits([_maximum(NZ, value) for value in TILE])
    )

  def test_a_rank_zero_min_keeps_the_tie_break_against_a_tile(self):
    kernel = self.emit(
      SCALAR_TILE.format(reduction="rmin", op="min"), "--tensor-profile-version", "v3"
    )
    # rmin of four positive zeros is +0.0, so a -0.0 tile lane wins its tie.
    self.assertEqual(
      _run(kernel, [[PZ] * 4, TILE], 16), _bits([_minimum(PZ, value) for value in TILE])
    )

  def test_i32_extrema_keep_the_plain_triton_calls(self):
    kernel = self.emit(ELEMENTWISE.format(dtype="i32", op="max"))
    self.assertIn("tl.maximum(", kernel)
    self.assertNotIn("== 0.0", kernel)


def _triton_interpreter_available():
  try:
    import torch  # noqa: F401
    import triton  # noqa: F401
  except Exception:
    return False
  return True


INTERPRETER_RUNNER = """\
import importlib.util
import json
import struct
import sys

import torch

spec = importlib.util.spec_from_file_location("kernel", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
inputs = [torch.tensor(values, dtype=torch.float32) for values in json.loads(sys.argv[2])]
out = torch.zeros(int(sys.argv[3]), dtype=torch.float32)
module.kernel[(1, 1, 1)](*inputs, out)
print(json.dumps([struct.unpack("<I", struct.pack("<f", v))[0] for v in out.tolist()]))
"""


@unittest.skipUnless(
  SYMIRC.is_file() and _triton_interpreter_available(),
  "symirc or the Triton interpreter is unavailable",
)
class TritonInterpreterSignedZeroTests(unittest.TestCase):
  """The same tie cases through Triton's own interpreter."""

  def run_kernel(self, source, inputs, output_length, *arguments):
    import json
    import os
    import subprocess

    result, kernel = emit_cli(source, arguments)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / "kernel.py").write_text(kernel, encoding="utf-8")
      (root / "runner.py").write_text(INTERPRETER_RUNNER, encoding="utf-8")
      completed = subprocess.run(
        [
          sys.executable,
          str(root / "runner.py"),
          str(root / "kernel.py"),
          json.dumps(inputs),
          str(output_length),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env={**os.environ, "TRITON_INTERPRET": "1"},
      )
      self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
      return json.loads(completed.stdout.strip().splitlines()[-1])

  def test_elementwise_extrema_keep_the_tie_break(self):
    for op, expected in (
      ("max", [PZ, PZ, NZ, PZ, 2.0, NZ, 5.0, NZ]),
      ("min", [NZ, NZ, NZ, PZ, 1.0, -3.0, 5.0, -1.0]),
    ):
      with self.subTest(op=op):
        source = ELEMENTWISE.format(dtype="f32", op=op)
        self.assertEqual(self.run_kernel(source, [LEFT, RIGHT], 8), _bits(expected))

  def test_a_rank_zero_extremum_against_a_tile_runs_and_keeps_the_tie_break(self):
    for reduction, op, lanes, expected in (
      ("rmax", "max", [NZ] * 4, [_maximum(NZ, value) for value in TILE]),
      ("rmin", "min", [PZ] * 4, [_minimum(PZ, value) for value in TILE]),
    ):
      with self.subTest(op=op):
        source = SCALAR_TILE.format(reduction=reduction, op=op)
        self.assertEqual(
          self.run_kernel(source, [lanes, TILE], 16, "--tensor-profile-version", "v3"),
          _bits(expected),
        )

  def test_reductions_keep_the_tie_break(self):
    flat = [value for row in ROWS for value in row]
    for op, expected in (("rmax", [PZ, NZ, NZ, 3.0]), ("rmin", [NZ, NZ, -4.0, NZ])):
      with self.subTest(op=op):
        source = REDUCTION.format(op=op)
        self.assertEqual(self.run_kernel(source, [flat], 4), _bits(expected))


if __name__ == "__main__":
  unittest.main()
