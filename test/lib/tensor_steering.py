"""Steer a free-form body's inputs toward a floating-point property (plan W4b).

Lane-level floating-point solving is too slow to steer with (rule D3), so the
body is encoded over the reals instead: `add`, `sub`, and `sum` are linear,
`min`, `max`, and their reductions are `ite` terms, and a product of two
computed values is linearized by fixing one factor to its value at the base
input and pinning that factor to the same value; the second factor is fixed
first, and the first is tried when that choice is unsat. Z3 (the `z3` binary) solves a real-valued surrogate of the
property, and its model is rounded to f32 inputs. The solver never supplies an
expected value: P0, the SIR interpreter, runs the rounded input, and the input
is accepted only if P0's output shows the property.

`evaluate` is an exact model of P0's f32 semantics (`src/interp/tensor_numeric.cpp`),
signed zeros included: elementwise results and sums are the round-to-nearest-
even f32 of their exact value, and `min`/`max` use the minNum/maxNum tie rule.
It decides what P0's output cannot show (whether a sum's exact value lands on
a rounding tie), and every accepted input requires it to agree with P0.

Properties, each a real surrogate on one output lane (or one `sum`) that
rounding then decides:

  signed-zero  the lane is negative and above -2^-150, so it rounds to -0.0,
               and no input is 0; accepted if P0 gives -0.0. With nonzero
               inputs only an underflowing product can reach it: a sum of f32
               values is 0 (+0.0 when rounded to nearest even) or at least
               2^-149 in magnitude.
  cancel       the lane is 0 and its producer (`add`, `sub`, or `sum`) has an
               operand of magnitude at least 2^20; accepted if P0 gives 0.
  tie          a `sum` equals 2^24 + 1; accepted if its exact value is halfway
               between two f32 values.

A surrogate Z3 proves unsat is a refusal. A rounded model that misses is
excluded and the query solved again, up to `attempts` times.
"""

from __future__ import annotations

import math
import re
import struct
import subprocess
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from test.lib.tensor_enclosure import NonFinite, round_f32

PROPERTIES = ("signed-zero", "cancel", "tie")
INPUT_BOUND = 2**24
VALUE_BOUND = 2**64
CANCEL_MAGNITUDE = 2**20
TIE = 2**24 + 1
# Half the smallest subnormal: a negative value above this rounds to -0.0.
UNDERFLOW = Fraction(1, 2**150)


class Unsupported(ValueError):
  """The body uses a statement this module does not model."""


def _negative(value):
  return math.copysign(1.0, value) < 0


def _f32(exact, negative_zero):
  """The f32 nearest an exact value; an exact zero takes the given sign."""
  if exact == 0:
    return -0.0 if negative_zero else 0.0
  rounded = round_f32(exact)
  return math.copysign(0.0, exact) if rounded == 0 else float(rounded)


def _binary(op, a, b):
  """P0's f32 result of one elementwise operation."""
  if op == "add":
    return _f32(Fraction(a) + Fraction(b), _negative(a) and _negative(b))
  if op == "sub":
    return _f32(Fraction(a) - Fraction(b), _negative(a) and not _negative(b))
  if op == "mul":
    return _f32(Fraction(a) * Fraction(b), _negative(a) != _negative(b))
  if op == "min":
    if a < b or b < a:
      return min(a, b)
    return -0.0 if a == 0 and (_negative(a) or _negative(b)) else a
  if a > b or b > a:
    return max(a, b)
  return -0.0 if a == 0 and _negative(a) and _negative(b) else a


def _real(value):
  """An SMT-LIB real literal for a float or fraction."""
  value = Fraction(value)
  text = (
    str(abs(value.numerator))
    if value.denominator == 1
    else (f"(/ {abs(value.numerator)} {value.denominator})")
  )
  return f"(- {text})" if value < 0 else text


@dataclass(eq=False)
class Node:
  """One lane value: its real term, its P0 value, and how it was produced."""

  term: str
  value: float
  op: str
  operands: tuple = ()
  exact: Fraction | None = None


class _Body:
  """Runs an exported body once, building real terms and P0 values together."""

  def __init__(self, source, parameters, inputs, fix_first=False):
    self.fix_first = fix_first
    self.lines, self.computed, self.sums, self.inputs = [], [], [], []
    self.count, self.linearized, self.pinned = 0, False, []
    self.blocks = {}
    label = None
    for line in source.splitlines():
      line = line.strip()
      if line.startswith("^"):
        label = line.rstrip(":")
        self.blocks[label] = []
      elif label and line and line != "}":
        self.blocks[label].append(line)
    self.values, self.scalars = {}, {}
    # One entry per input element: a solver variable, or a fixed scalar i32
    # parameter (a trip count), which is the path and is never steered.
    self.slots = []
    position = 0
    for name, shape in parameters:
      if list(shape) == []:
        self.scalars[name] = int(inputs[position])
        self.slots.append(self.scalars[name])
        position += 1
        continue
      count = int(np.prod(shape))
      lanes = np.empty(count, dtype=object)
      for index in range(count):
        variable = f"in_{name[1:]}_{index}"
        self.lines.append(f"(declare-const {variable} Real)")
        lanes[index] = Node(variable, float(inputs[position + index]), "input")
        self.inputs.append(lanes[index])
        self.slots.append(lanes[index])
      position += count
      self.values[name] = lanes.reshape(shape)

  def _define(self, expression, value, op, operands=(), exact=None):
    name = f"t{self.count}"
    self.count += 1
    self.lines.append(f"(define-fun {name} () Real {expression})")
    node = Node(name, value, op, tuple(operands), exact)
    self.computed.append(node)
    return node

  def _binary(self, op, a, b):
    value = _binary(op, a.value, b.value)
    if op in ("add", "sub"):
      expression = f"({'+' if op == 'add' else '-'} {a.term} {b.term})"
    elif op == "mul":
      fixed, free = (
        (a, b) if a.op == "const" or (self.fix_first and b.op != "const") else (b, a)
      )
      expression = f"(* {_real(fixed.value)} {free.term})"
      if a.op != "const" and b.op != "const":
        # The fixed factor keeps its base value, so the model stays consistent.
        self.linearized = True
        self.pinned.append(f"(= {fixed.term} {_real(fixed.value)})")
    elif op == "min":
      expression = f"(ite (<= {a.term} {b.term}) {a.term} {b.term})"
    else:
      expression = f"(ite (>= {a.term} {b.term}) {a.term} {b.term})"
    return self._define(expression, value, op, (a, b))

  def _sum(self, lanes):
    exact = sum((Fraction(lane.value) for lane in lanes), Fraction(0))
    node = self._define(
      f"(+ {' '.join(lane.term for lane in lanes)})"
      if len(lanes) > 1
      else lanes[0].term,
      _f32(exact, False),
      "sum",
      lanes,
      exact,
    )
    self.sums.append(node)
    return node

  def _reduce(self, op, values, axis, keep):
    lanes = np.moveaxis(values, axis, -1)
    out = np.empty(lanes.shape[:-1], dtype=object)
    for index in np.ndindex(out.shape):
      group = list(lanes[index])
      if op == "sum":
        out[index] = self._sum(group)
      else:
        result = group[0]
        for lane in group[1:]:
          result = self._binary("min" if op == "rmin" else "max", result, lane)
        out[index] = result
    return np.expand_dims(out, axis) if keep else out

  def _operand(self, text):
    text = text.strip()
    if text.startswith("%"):
      return self.values[text]
    value = float(text)
    return np.array(Node(_real(value), value, "const"), dtype=object)

  def _tensor(self, op, rest):
    if op == "window":
      match = re.fullmatch(r"(%\w+), (%\w+), ([^,]+), \[(\d+), (\d+), (\d+)\]", rest)
      if not match:
        raise Unsupported(rest)
      source, index, pad, rows, stride, active = match.groups()
      start = self.scalars[index] * int(stride)
      backing = self.values[source]
      fill = Node(_real(float(pad)), float(pad), "const")
      return np.array(
        [
          list(backing[start + row])
          if start + row < int(active)
          else [fill] * backing.shape[1]
          for row in range(int(rows))
        ],
        dtype=object,
      )
    if op in ("sum", "rmin", "rmax"):
      name, axis, keep = (part.strip() for part in rest.split(","))
      return self._reduce(op, self.values[name], int(axis), keep == "keep")
    if op == "transpose":
      name, permutation = re.fullmatch(r"(%\w+), \[([\d, ]+)\]", rest).groups()
      return np.transpose(self.values[name], [int(p) for p in permutation.split(",")])
    if op == "expand":
      name, axis = (part.strip() for part in rest.split(","))
      return np.expand_dims(self.values[name], int(axis))
    if op == "broadcast":
      name, shape = re.fullmatch(r"(%\w+), \[([\d, ]+)\]", rest).groups()
      return np.broadcast_to(self.values[name], [int(s) for s in shape.split(",")])
    if op in ("add", "sub", "mul", "min", "max"):
      left, right = np.broadcast_arrays(
        *(self._operand(part) for part in rest.split(","))
      )
      out = np.empty(left.shape, dtype=object)
      for index in np.ndindex(left.shape):
        out[index] = self._binary(op, left[index], right[index])
      return out
    raise Unsupported(f"tensor {op}")

  def run(self):
    label = "^entry"
    for _ in range(10_000):
      for line in self.blocks[label]:
        assignment = re.fullmatch(r"(%\w+) = (.+);", line)
        if assignment:
          target, expression = assignment.groups()
          tensor = re.fullmatch(r"tensor (\w+) (.+)", expression)
          if tensor:
            self.values[target] = self._tensor(*tensor.groups())
          elif expression in self.values:
            self.values[target] = self.values[expression]
          elif re.fullmatch(r"-?\d+", expression):
            self.scalars[target] = int(expression)
          elif expression == f"{target} + 1":
            self.scalars[target] += 1
          else:
            raise Unsupported(expression)
          continue
        if line.startswith("assume "):
          # P0 enforces the assumed range of a scalar parameter.
          continue
        if line.startswith("ret "):
          return list(
            np.asarray(self.values[line[4:].rstrip(";")], dtype=object).reshape(-1)
          )
        jump = re.fullmatch(r"br (\^\w+);", line)
        branch = re.fullmatch(r"br (%\w+) < (%\w+|\d+), (\^\w+), (\^\w+);", line)
        if jump:
          label = jump.group(1)
        elif branch:
          name, bound, taken, other = branch.groups()
          limit = self.scalars[bound] if bound.startswith("%") else int(bound)
          label = taken if self.scalars[name] < limit else other
        else:
          raise Unsupported(line)
        break
    raise Unsupported("the body did not return")


def pack(parameters, values):
  """Input bytes for (name, shape) parameters: f32 lanes, an i32 for a scalar."""
  data, position = bytearray(), 0
  for _, shape in parameters:
    if list(shape) == []:
      data += struct.pack("<i", int(values[position]))
      position += 1
      continue
    count = int(np.prod(shape))
    data += struct.pack(f"<{count}f", *values[position : position + count])
    position += count
  return bytes(data)


def unpack(parameters, data):
  """The values `pack` wrote, in parameter order."""
  values, offset = [], 0
  for _, shape in parameters:
    count = 1 if list(shape) == [] else int(np.prod(shape))
    code = "<i" if list(shape) == [] else f"<{count}f"
    values += list(struct.unpack_from(code, data, offset))
    offset += 4 * count
  return values


def evaluate(source, parameters, inputs):
  """P0's output for these inputs, from the exact model."""
  return [node.value for node in _Body(source, parameters, inputs).run()]


# --- Z3 ---------------------------------------------------------------------

_TOKEN = re.compile(r"\(|\)|[^\s()]+")


def _parse(text):
  """Nested lists from an S-expression."""
  stack = [[]]
  for token in _TOKEN.findall(text):
    if token == "(":
      stack.append([])
    elif token == ")":
      item = stack.pop()
      stack[-1].append(item)
    else:
      stack[-1].append(token)
  return stack[0]


def _rational(expression):
  if isinstance(expression, str):
    return Fraction(expression)
  head, *rest = expression
  if head == "-":
    return (
      -_rational(rest[0]) if len(rest) == 1 else _rational(rest[0]) - _rational(rest[1])
    )
  if head == "/":
    return _rational(rest[0]) / _rational(rest[1])
  raise ValueError(f"not a rational: {expression}")


def _z3(text, *, scratch, name, timeout_s):
  scratch = Path(scratch)
  scratch.mkdir(parents=True, exist_ok=True)
  path = scratch / f"{name}.smt2"
  path.write_text(text, encoding="utf-8")
  started = time.monotonic()
  try:
    completed = subprocess.run(
      ["z3", f"-T:{timeout_s}", str(path)],
      capture_output=True,
      text=True,
      timeout=timeout_s + 30,
    )
    output = completed.stdout
  except subprocess.TimeoutExpired:
    output = "timeout"
  seconds = time.monotonic() - started
  lines = output.strip().splitlines()
  verdict = lines[0] if lines else "no output"
  model = {}
  if verdict == "sat":
    for pair in _parse(" ".join(lines[1:]))[0]:
      model[pair[0]] = _rational(pair[1])
  return verdict, model, seconds


def _bounds(body):
  facts = [f"(<= {-INPUT_BOUND} {node.term} {INPUT_BOUND})" for node in body.inputs]
  facts += [
    f"(<= {_real(-VALUE_BOUND)} {node.term} {_real(VALUE_BOUND)})"
    for node in body.computed
  ]
  return facts + sorted(set(body.pinned))


def _query(body, facts, wanted):
  return (
    "\n".join(
      [
        "(set-logic QF_LRA)",
        "(set-option :produce-models true)",
        *body.lines,
        *(f"(assert {fact})" for fact in facts),
        "(check-sat)",
        f"(get-value ({' '.join(wanted)}))",
      ]
    )
    + "\n"
  )


def real_outputs(source, parameters, inputs, *, scratch, timeout_s=60):
  """The real encoding's outputs with every input fixed: a check of the encoding."""
  body = _Body(source, parameters, inputs)
  outputs = body.run()
  facts = [f"(= {node.term} {_real(node.value)})" for node in body.inputs]
  verdict, model, _ = _z3(
    _query(body, facts, [node.term for node in outputs]),
    scratch=scratch,
    name="real-outputs",
    timeout_s=timeout_s,
  )
  if verdict != "sat":
    raise RuntimeError(f"the fixed-input encoding is {verdict}")
  return [model[node.term] for node in outputs]


@dataclass
class Steered:
  """One steering query's outcome."""

  status: str
  property: str
  lane: int | None
  reason: str = ""
  values: list | None = None
  attempts: list = field(default_factory=list)
  z3_seconds: float = 0.0
  linearized: bool = False
  evidence: dict = field(default_factory=dict)


def _abs_at_least(term, magnitude):
  return f"(or (>= {term} {magnitude}) (<= {term} (- {magnitude})))"


def _surrogate(body, outputs, prop, lane):
  """Facts for the property's real surrogate, the node it aims at, or a reason."""
  if prop == "signed-zero":
    node = outputs[lane]
    nonzero = [f"(not (= {i.term} 0))" for i in body.inputs]
    return [f"(< {_real(-UNDERFLOW)} {node.term} 0)"] + nonzero, node, None
  if prop == "cancel":
    node = outputs[lane]
    if node.op not in ("add", "sub", "sum"):
      return None, node, f"lane {lane} is not produced by add, sub, or sum"
    large = " ".join(
      _abs_at_least(operand.term, CANCEL_MAGNITUDE) for operand in node.operands
    )
    return [f"(= {node.term} 0)", f"(or {large})"], node, None
  if not body.sums:
    return None, None, "the body has no sum"
  node = body.sums[lane or 0]
  return [f"(= {node.term} {TIE})"], node, None


def _halfway(exact):
  if exact is None:
    return False
  try:
    down, up = round_f32(exact, "down"), round_f32(exact, "up")
  except NonFinite:
    return False
  return down != up and exact - down == up - exact


def _check(prop, lane, inputs, source, parameters, p0, aim_index, fix_first):
  """Whether P0 confirms the property for these inputs, or why not."""
  try:
    body = _Body(source, parameters, inputs, fix_first=fix_first)
    outputs = body.run()
  except NonFinite:
    return "the model leaves the finite range", {}
  if prop == "signed-zero" and any(node.value == 0 for node in body.inputs):
    return "an input rounded to zero", {}
  produced = p0(inputs)
  if produced is None:
    return "p0_not_completed", {}
  expected = struct.pack(f"<{len(outputs)}f", *(node.value for node in outputs))
  if produced != expected:
    return "p0_differs_from_the_model", {}
  if prop == "signed-zero":
    hit = _negative(outputs[lane].value) and outputs[lane].value == 0
    return (None if hit else "lane is not -0.0"), {}
  if prop == "cancel":
    node = outputs[lane]
    largest = max(abs(operand.value) for operand in node.operands)
    hit = node.value == 0 and largest >= CANCEL_MAGNITUDE
    return (None if hit else "no cancellation"), {"largest_operand": largest}
  node = body.sums[aim_index]
  hit = _halfway(node.exact)
  return (None if hit else "sum is not halfway"), {
    "halfway": hit,
    "sum_exact": str(node.exact),
  }


def steer(
  source, parameters, base, *, prop, p0, scratch, lane=None, attempts=5, timeout_s=60
):
  """Steer one property on one lane (or sum) of a body; see the module docstring."""
  if prop not in PROPERTIES:
    raise ValueError(f"unknown property {prop!r}")
  aim = lane if prop != "tie" else (lane or 0)
  result = Steered("refused", prop, aim)
  for fix_first in (False, True):
    body = _Body(source, parameters, base, fix_first=fix_first)
    outputs = body.run()
    facts, _, refusal = _surrogate(body, outputs, prop, aim)
    result.linearized = body.linearized
    if refusal:
      result.reason = refusal
      return result
    _attempt(
      result,
      body,
      facts,
      prop,
      aim,
      source,
      parameters,
      p0,
      scratch,
      attempts,
      timeout_s,
    )
    if result.status == "accepted" or result.reason != "unsat" or not body.linearized:
      return result
  return result


def _attempt(
  result, body, facts, prop, aim, source, parameters, p0, scratch, attempts, timeout_s
):
  """Solve, round, and check up to `attempts` times, recording each attempt."""
  blocked = []
  for attempt in range(attempts):
    verdict, model, seconds = _z3(
      _query(body, _bounds(body) + facts + blocked, [i.term for i in body.inputs]),
      scratch=scratch,
      name=f"{prop}-{aim}-{'first' if body.fix_first else 'second'}-{attempt}",
      timeout_s=timeout_s,
    )
    result.z3_seconds += seconds
    if verdict != "sat":
      result.reason = verdict
      return
    inputs = []
    for slot in body.slots:
      if not isinstance(slot, Node):
        inputs.append(slot)
        continue
      try:
        inputs.append(float(round_f32(model[slot.term])))
      except NonFinite:
        inputs.append(math.inf)
    miss, evidence = _check(
      prop, aim, inputs, source, parameters, p0, aim, body.fix_first
    )
    result.attempts.append({"miss": miss, "z3_seconds": round(seconds, 3)})
    if miss is None:
      result.status, result.values, result.evidence = "accepted", inputs, evidence
      result.reason = ""
      return
    blocked.append(
      "(or "
      + " ".join(f"(not (= {i.term} {_real(model[i.term])}))" for i in body.inputs)
      + ")"
    )
  result.reason = f"no accepted input in {attempts} attempts"


@dataclass
class BodySearch:
  """Every steering trial on one body and the first accepted input per property."""

  accepted: dict = field(default_factory=dict)
  trials: list = field(default_factory=list)
  z3_seconds: float = 0.0


def steer_body(
  source,
  parameters,
  bases,
  *,
  p0,
  scratch,
  properties=PROPERTIES,
  lanes=4,
  budget_s=60.0,
  attempts=5,
):
  """Try each property on up to `lanes` lanes (or sums), linearized at each base.

  `bases` maps a name to the input values a product is linearized at. A
  property stops at its first accepted input; the search stops when the body's
  Z3 time reaches `budget_s`. A query started just under the budget may run
  past it.
  """
  search = BodySearch()
  for prop in properties:
    for base_name, base in bases.items():
      for lane in range(lanes):
        if prop in search.accepted or search.z3_seconds >= budget_s:
          break
        result = steer(
          source,
          parameters,
          base,
          prop=prop,
          lane=lane,
          p0=p0,
          scratch=scratch,
          attempts=attempts,
          timeout_s=max(1, int(budget_s - search.z3_seconds)),
        )
        search.z3_seconds += result.z3_seconds
        search.trials.append(
          {
            "property": prop,
            "base": base_name,
            "lane": lane,
            "status": result.status,
            "reason": result.reason,
            "attempts": len(result.attempts),
            "z3_seconds": round(result.z3_seconds, 3),
            "linearized": result.linearized,
            "evidence": result.evidence,
          }
        )
        if result.status == "accepted":
          search.accepted[prop] = result
  return search
