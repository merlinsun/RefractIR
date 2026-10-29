"""Enclose a free-form body's output over every valid summation order.

The two Triton paths may differ only where the comparison contract leaves a
choice to the backend. With fusion off, `+ - *`, `min`, `max`, their reductions,
and data movement are specified: each result is the round-to-nearest-even f32
of the exact result. `sum` is not: a backend adds its terms in any order.

This module evaluates an exported body (`program.sir`) on its recorded inputs
over exact rationals, carrying each value as an interval [lo, hi] of f32
values. A specified operation rounds the exact extremes of its operand
intervals to nearest; rounding is monotone, so the interval holds every IEEE
result, and point operands give a point. A `sum` of n terms x_i lies within
gamma_(n-1) * sum|x_i| of the exact sum in any order, where
gamma_k = k*u / (1 - k*u) and u = 2^-24; the bound is rounded inward to f32,
since the result is an f32. When every term is a multiple of one power of two
g and sum|x_i| <= 2^24 * g, every partial sum is representable, so every order
gives the exact sum and the lane stays a point.

The width of an output lane is the number of f32 steps from lo to hi.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

U = Fraction(1, 2**24)
_MIN_EXPONENT = -149
_OVERFLOW = Fraction(2**128)


class NonFinite(ArithmeticError):
  """A value left the finite f32 range, which SIR leaves undefined."""


class Unsupported(ValueError):
  """The program uses a statement this evaluator does not model."""


def _quantum(value):
  """The f32 spacing at a nonzero magnitude: 2^(e - 23), at least 2^-149."""
  exponent = value.numerator.bit_length() - value.denominator.bit_length()
  if Fraction(2) ** exponent > value:
    exponent -= 1
  return Fraction(2) ** max(exponent - 23, _MIN_EXPONENT)


def round_f32(value, mode="nearest"):
  """Round an exact rational to f32: to nearest even, or "down" / "up"."""
  value = Fraction(value)
  if value == 0:
    return Fraction(0)
  quantum = _quantum(abs(value))
  steps = value / quantum
  low = steps.numerator // steps.denominator
  if mode == "down":
    count = low
  elif mode == "up":
    count = low if steps == low else low + 1
  else:
    remainder = steps - low
    count = (
      low + 1
      if remainder > Fraction(1, 2) or (remainder == Fraction(1, 2) and low % 2)
      else low
    )
  result = count * quantum
  if abs(result) >= _OVERFLOW:
    raise NonFinite(f"{float(value):g} is outside the finite f32 range")
  return result


def _ordinal(value):
  """f32 values in order as integers; both zeros are 0."""
  bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
  return -(bits & 0x7FFFFFFF) if bits >> 31 else bits


def ulps(lo, hi):
  """The number of f32 steps from lo to hi."""
  return _ordinal(hi) - _ordinal(lo)


_round = np.vectorize(round_f32, otypes=[object])
_ulps = np.vectorize(ulps, otypes=[object])


@dataclass(frozen=True)
class Interval:
  """Elementwise f32 bounds, as object arrays of exact rationals."""

  lo: np.ndarray
  hi: np.ndarray

  @staticmethod
  def point(values):
    values = np.asarray(values, dtype=object)
    return Interval(values, values)

  def map(self, function):
    return Interval(function(self.lo), function(self.hi))

  def widths(self):
    return [int(width) for width in _ulps(self.lo, self.hi).flat]


def _monotone(function, a, b):
  """Round an elementwise operation nondecreasing in both operands."""
  return Interval(_round(function(a.lo, b.lo)), _round(function(a.hi, b.hi)))


def _product(a, b):
  corners = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
  low, high = corners[0], corners[0]
  for corner in corners[1:]:
    low, high = np.minimum(low, corner), np.maximum(high, corner)
  return Interval(_round(low), _round(high))


def _grain(value):
  """The largest power of two dividing a nonzero f32 value."""
  numerator = abs(value.numerator)
  twos = (numerator & -numerator).bit_length() - 1
  return Fraction(2**twos, value.denominator)


def _exact_in_any_order(terms):
  """Every partial sum of these point terms is an f32, whatever the order.

  Each partial sum is a multiple of the terms' common grain g of magnitude at
  most sum|x_i|, and every multiple of g up to 2^24 * g is an f32.
  """
  nonzero = [abs(term) for term in terms if term != 0]
  if not nonzero:
    return True
  return sum(nonzero) <= 2**24 * min(_grain(term) for term in nonzero)


def _sum(a, axis, keep):
  count = a.lo.shape[axis]
  gamma = (count - 1) * U / (1 - (count - 1) * U)
  low_lanes = np.moveaxis(a.lo, axis, -1)
  high_lanes = np.moveaxis(a.hi, axis, -1)
  low = np.empty(low_lanes.shape[:-1], dtype=object)
  high = np.empty(low_lanes.shape[:-1], dtype=object)
  for index in np.ndindex(low.shape):
    los, his = list(low_lanes[index]), list(high_lanes[index])
    if los == his and _exact_in_any_order(los):
      low[index] = high[index] = round_f32(sum(los))
      continue
    magnitude = sum(max(abs(lo), abs(hi)) for lo, hi in zip(los, his))
    low[index] = round_f32(sum(los) - gamma * magnitude, "up")
    high[index] = round_f32(sum(his) + gamma * magnitude, "down")
  if keep:
    low, high = np.expand_dims(low, axis), np.expand_dims(high, axis)
  return Interval(low, high)


def apply(op, operands, *, axis=None, keep=False):
  """One tensor operation on interval operands."""
  if op in ("add", "sub", "mul", "min", "max"):
    a, b = operands
    if op == "add":
      return _monotone(np.add, a, b)
    if op == "sub":
      return _monotone(np.subtract, a, Interval(b.hi, b.lo))
    if op == "mul":
      return _product(a, b)
    return _monotone(np.minimum if op == "min" else np.maximum, a, b)
  (a,) = operands
  if op == "sum":
    return _sum(a, axis, keep)
  if op in ("rmin", "rmax"):
    reduce = np.min if op == "rmin" else np.max
    return a.map(lambda values: reduce(values, axis=axis, keepdims=keep))
  raise Unsupported(f"tensor {op}")


_STATEMENT = re.compile(r"(%\w+) = (.+);")
_SHAPE = r"\[([\d, ]+)\]"


def _ints(text):
  return [int(part) for part in text.split(",")]


class _Machine:
  """Runs the straight-line blocks and counted loop of an exported body."""

  def __init__(self, source, inputs):
    self.blocks = {}
    label = None
    for line in source.splitlines():
      line = line.strip()
      if line.startswith("^"):
        label = line.rstrip(":")
        self.blocks[label] = []
      elif label is not None and line and line != "}":
        self.blocks[label].append(line)
    self.values = {name: Interval.point(values) for name, values in inputs.items()}
    self.scalars = {}

  def operand(self, text):
    text = text.strip()
    if text.startswith("%"):
      return self.values[text]
    return Interval.point(np.array(Fraction(text), dtype=object))

  def tensor(self, op, rest):
    if op == "window":
      match = re.fullmatch(rf"(%\w+), (%\w+), ([^,]+), {_SHAPE}", rest)
      rows, stride, active = _ints(match.group(4))
      start = self.scalars[match.group(2)] * stride
      pad = Fraction(match.group(3))
      source = self.values[match.group(1)]

      def window(values):
        columns = values.shape[1]
        return np.array(
          [
            list(values[start + row]) if start + row < active else [pad] * columns
            for row in range(rows)
          ],
          dtype=object,
        )

      return source.map(window)
    if op in ("sum", "rmin", "rmax"):
      name, axis, keep = (part.strip() for part in rest.split(","))
      return apply(op, [self.values[name]], axis=int(axis), keep=keep == "keep")
    if op == "transpose":
      name, permutation = re.fullmatch(rf"(%\w+), {_SHAPE}", rest).groups()
      return self.values[name].map(
        lambda values: np.transpose(values, _ints(permutation))
      )
    if op == "expand":
      name, axis = (part.strip() for part in rest.split(","))
      return self.values[name].map(lambda values: np.expand_dims(values, int(axis)))
    if op == "broadcast":
      name, shape = re.fullmatch(rf"(%\w+), {_SHAPE}", rest).groups()
      return self.values[name].map(lambda values: np.broadcast_to(values, _ints(shape)))
    left, right = rest.split(",")
    return apply(op, [self.operand(left), self.operand(right)])

  def assign(self, target, expression):
    tensor = re.fullmatch(r"tensor (\w+) (.+)", expression)
    if tensor:
      self.values[target] = self.tensor(*tensor.groups())
    elif re.fullmatch(r"%\w+", expression) and expression in self.values:
      self.values[target] = self.values[expression]
    elif re.fullmatch(r"-?\d+", expression):
      self.scalars[target] = int(expression)
    elif re.fullmatch(rf"{re.escape(target)} \+ 1", expression):
      self.scalars[target] += 1
    else:
      raise Unsupported(expression)

  def run(self, limit=10_000):
    label = "^entry"
    for _ in range(limit):
      for line in self.blocks[label]:
        statement = _STATEMENT.fullmatch(line)
        if statement:
          self.assign(*statement.groups())
          continue
        if line.startswith("ret "):
          return self.values[line[4:].rstrip(";")]
        jump = re.fullmatch(r"br (\^\w+);", line)
        branch = re.fullmatch(r"br (%\w+) < (\d+), (\^\w+), (\^\w+);", line)
        if jump:
          label = jump.group(1)
        elif branch:
          name, bound, taken, other = branch.groups()
          label = taken if self.scalars[name] < int(bound) else other
        else:
          raise Unsupported(line)
        break
    raise Unsupported("the program did not return")


def enclose(source, inputs):
  """The output interval of an exported body for named input arrays."""
  for op in re.findall(r"= tensor (\w+)", source):
    if op not in (
      "add",
      "sub",
      "mul",
      "min",
      "max",
      "sum",
      "rmin",
      "rmax",
      "window",
      "transpose",
      "expand",
      "broadcast",
    ):
      raise Unsupported(f"tensor {op}")
  return _Machine(source, inputs).run()


def enclose_bundle(bundle):
  """The output interval for a target bundle's program and recorded inputs."""
  bundle = Path(bundle)
  record = json.loads((bundle / "case.json").read_text(encoding="utf-8"))
  data = (bundle / "inputs.bin").read_bytes()
  inputs = {}
  for spec in record["inputs"]:
    chunk = data[spec["byte_offset"] : spec["byte_offset"] + spec["byte_length"]]
    values = [Fraction(value) for (value,) in struct.iter_unpack("<f", chunk)]
    inputs[spec["name"]] = np.array(values, dtype=object).reshape(spec["shape"])
  return enclose((bundle / "program.sir").read_text(encoding="utf-8"), inputs)
