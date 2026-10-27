"""Triton extrema keep SIR's signed-zero tie-break; no GPU is required.

SIR resolves equal-magnitude signed zeros in `min`, `max`, `rmin`, and `rmax`
with the IEEE minNum/maxNum rule: a -0.0 operand forces a -0.0 minimum and a
+0.0 operand forces a +0.0 maximum. Triton leaves that tie unspecified, so the
emitted kernel must fix it itself. These tests run the emitted kernel against
a numpy stand-in for `triton.language` whose extrema return the wrong signed
zero on every tie, so only a lowering that does not depend on the tie passes.
"""

import runpy
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from test.unit.tensor_support import SYMIRC
from test.unit.test_tensor_emit import emit_cli

try:
  import numpy as np
except ImportError:  # pragma: no cover - numpy is a test-environment dependency
  np = None


class _Pointer:
  def __init__(self, data, offsets=0):
    self.data, self.offsets = data, offsets

  def __add__(self, offsets):
    return _Pointer(self.data, self.offsets + offsets)


class _Tensor(np.ndarray if np is not None else object):
  """A numpy array with Triton's `.to(dtype, bitcast=...)`."""

  def to(self, dtype, bitcast=False):
    return _wrap(self.view(dtype) if bitcast else self.astype(dtype))


def _wrap(value):
  return np.asarray(value).view(_Tensor)


def _adversarial_language():
  """`triton.language` over numpy, breaking every extrema tie the wrong way."""
  tl = types.ModuleType("triton.language")
  tl.float32, tl.int32, tl.int64 = np.float32, np.int32, np.int64
  tl.arange = lambda start, end: np.arange(start, end, dtype=np.int32)
  tl.full = lambda shape, value, dtype=np.float32: np.full(shape, value, dtype=dtype)
  tl.where = lambda condition, left, right: _wrap(
    np.where(condition, left, right).astype(np.result_type(left, right))
  )

  def load(pointer, mask=None, other=0):
    offsets = np.asarray(pointer.offsets)
    if mask is None:
      return _wrap(pointer.data[offsets])
    safe = np.where(mask, offsets, 0)
    return _wrap(np.where(mask, pointer.data[safe], other).astype(pointer.data.dtype))

  def store(pointer, value, mask=None):
    offsets = np.asarray(pointer.offsets)
    value = np.broadcast_to(np.asarray(value, dtype=pointer.data.dtype), offsets.shape)
    if mask is None:
      pointer.data[offsets] = value
    else:
      pointer.data[offsets[mask]] = value[mask]

  def tie(result, zero_tie, wrong):
    if result.dtype.kind != "f":
      return _wrap(result)
    return _wrap(np.where(zero_tie, np.float32(wrong), result).astype(result.dtype))

  tl.load, tl.store = load, store
  tl.maximum = lambda a, b: tie(np.maximum(a, b), (a == 0) & (b == 0), -0.0)
  tl.minimum = lambda a, b: tie(np.minimum(a, b), (a == 0) & (b == 0), 0.0)

  def reduce(function, wrong):
    def apply(value, axis=None, keep_dims=False):
      result = function(value, axis=axis, keepdims=keep_dims)
      return tie(np.asarray(result), np.asarray(result) == 0, wrong)

    return apply

  tl.max = reduce(np.max, -0.0)
  tl.min = reduce(np.min, 0.0)
  tl.sum = lambda value, axis=None, keep_dims=False: _wrap(
    np.sum(value, axis=axis, keepdims=keep_dims)
  )
  return tl


def _run(kernel_source, inputs, output_length):
  """Run an emitted kernel on float32 inputs and return the output bits."""
  triton = types.ModuleType("triton")
  triton.jit = lambda function: function
  language = _adversarial_language()
  triton.language = language
  with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "kernel.py"
    path.write_text(kernel_source, encoding="utf-8")
    with mock.patch.dict(sys.modules, {"triton": triton, "triton.language": language}):
      kernel = runpy.run_path(str(path))["kernel"]
      out = np.zeros(output_length, dtype=np.float32)
      kernel(
        *[_Pointer(np.asarray(values, np.float32)) for values in inputs], _Pointer(out)
      )
  return [struct.unpack("<I", struct.pack("<f", value))[0] for value in out]


def _bits(values):
  return [struct.unpack("<I", struct.pack("<f", value))[0] for value in values]


PZ, NZ = 0.0, -0.0

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

LEFT = [PZ, NZ, NZ, PZ, 1.0, -3.0, 5.0, NZ]
RIGHT = [NZ, PZ, NZ, PZ, 2.0, NZ, 5.0, -1.0]
ROWS = [[NZ, PZ, NZ, NZ], [NZ, NZ, NZ, NZ], [-1.0, NZ, -2.0, -4.0], [3.0, PZ, NZ, 1.0]]


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

  def test_reductions_keep_the_tie_break(self):
    flat = [value for row in ROWS for value in row]
    for op, expected in (("rmax", [PZ, NZ, NZ, 3.0]), ("rmin", [NZ, NZ, -4.0, NZ])):
      with self.subTest(op=op):
        source = REDUCTION.format(op=op)
        self.assertEqual(self.run_kernel(source, [flat], 4), _bits(expected))


if __name__ == "__main__":
  unittest.main()
