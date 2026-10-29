"""A numpy stand-in for `triton.language`, enough to run emitted kernels on CPU.

The stand-in breaks every floating-point extrema tie the wrong way: an equal
pair of signed zeros gives the wrong signed zero, so a kernel whose result
depends on Triton's unspecified tie-break fails here. No Triton is required.
"""

import runpy
import struct
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

try:
  import numpy as np
except ImportError:  # pragma: no cover - numpy is a test-environment dependency
  np = None


class Pointer:
  def __init__(self, data, offsets=0):
    self.data, self.offsets = data, offsets

  def __add__(self, offsets):
    return Pointer(self.data, self.offsets + offsets)


class _Tensor(np.ndarray if np is not None else object):
  """A numpy array with Triton's `.to(dtype, bitcast=...)`."""

  def to(self, dtype, bitcast=False):
    return _wrap(self.view(dtype) if bitcast else self.astype(dtype))


def _wrap(value):
  return np.asarray(value).view(_Tensor)


def language():
  """`triton.language` over numpy, breaking every extrema tie the wrong way."""
  tl = types.ModuleType("triton.language")
  tl.float32, tl.int32, tl.int64 = np.float32, np.int32, np.int64
  tl.arange = lambda start, end: np.arange(start, end, dtype=np.int32)
  tl.full = lambda shape, value, dtype=np.float32: _wrap(
    np.full(shape, value, dtype=dtype)
  )
  tl.range = lambda start, end, **_hints: range(start, end)
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


def _jit(function=None, **_options):
  """`triton.jit` both bare and with keyword options such as `do_not_specialize`."""
  return function if function is not None else (lambda inner: inner)


def run(kernel_source, inputs, output_length, dtype=None, **arguments):
  """Run an emitted kernel on the inputs and return the output words as integers.

  `arguments` are passed by keyword, as a launcher passes a scalar argument.
  """
  dtype = np.float32 if dtype is None else dtype
  triton = types.ModuleType("triton")
  triton.jit = _jit
  tl = language()
  triton.language = tl
  with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "kernel.py"
    path.write_text(kernel_source, encoding="utf-8")
    with mock.patch.dict(sys.modules, {"triton": triton, "triton.language": tl}):
      kernel = runpy.run_path(str(path))["kernel"]
      out = np.zeros(output_length, dtype=dtype)
      kernel(
        *[Pointer(np.asarray(values, dtype)) for values in inputs],
        Pointer(out),
        **arguments,
      )
  return [struct.unpack("<I", value.tobytes())[0] for value in out]
