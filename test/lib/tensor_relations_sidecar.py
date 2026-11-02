"""Relation facts of a generated body for the sidecar (seed-export plan, E2).

The body is run once, symbolically, along the case's own path (its trip count
and branch selector), by the statement machine of `tensor_enclosure`. Each value
carries the facts the relations need:

- a linear form over the float inputs' weights: the exponent its factor gets
  when every input i is multiplied by 2^(k * weight_i). Operations that need
  equal factors (add, sub, min, max, the arms and the comparison of `where`)
  record an equation; mul and dot add forms; div subtracts them;
- the operations in its backward slice, the one that produced it, and the axes
  reductions consumed;
- per axis, which input axes map onto it lane for lane, and which inputs reach
  it only broadcast.

The weights are an integer solution of the equations (all ones when that
solves them, else a generic combination of the nullspace, possibly negative);
parity solves the same equations mod 2, so
min/max and comparisons count as sign-preserving because the harness runs the
dual kernel under negation. A zero literal or zero fill takes any factor; a
nonzero literal pins its partner's weight to zero. `exp` makes a value
unknown.
"""

import itertools
import math
import re
from fractions import Fraction

from test.lib import tensor_enclosure

ZERO = "zero"
UNKNOWN = "unknown"
EXCLUDED_OPS = frozenset({"div", "exp", "rem"})
ORDER_DEPENDENT_OPS = frozenset({"sum", "cumsum"})


class _Fact:
  """What the relations need to know about one value."""

  def __init__(self, form, shape, *, ops=frozenset(), producer="input", axes=None):
    self.form = form
    self.shape = tuple(shape)
    self.ops = frozenset(ops)
    self.producer = producer
    # Per axis: (input, input axis) pairs mapped lane for lane, or (input, None)
    # for an input that reaches this axis only broadcast.
    self.axes = axes if axes is not None else [set() for _ in self.shape]
    self.tainted = set()
    self.reduced = set()

  @property
  def inputs(self):
    return {name for entries in self.axes for name, _ in entries} | {
      name for name, _ in self.tainted
    }


def _input(name, shape):
  form = {name: 1}
  return _Fact(form, shape, axes=[{(name, axis)} for axis in range(len(shape))])


def _literal(text):
  return _Fact(ZERO if float(text) == 0.0 else {}, ())


def _add_forms(left, right, sign=1):
  if UNKNOWN in (left, right):
    return UNKNOWN
  if left == ZERO or right == ZERO:
    return ZERO
  total = dict(left)
  for name, coefficient in right.items():
    total[name] = total.get(name, 0) + sign * coefficient
  return total


def _broadcast_shape(shapes):
  rank = max(len(shape) for shape in shapes)
  result = []
  for position in range(rank):
    extents = {
      shape[position - rank + len(shape)]
      for shape in shapes
      if position - rank + len(shape) >= 0
    }
    result.append(max(extents))
  return tuple(result)


def _aligned_axes(operands, shape):
  """Each result axis's entries from right-aligned operands; an operand of
  extent 1 (or missing the axis) reaches it only broadcast."""
  axes = [set() for _ in shape]
  for fact in operands:
    offset = len(shape) - len(fact.shape)
    for position in range(len(shape)):
      own = position - offset
      if own >= 0 and fact.shape[own] == shape[position]:
        axes[position] |= fact.axes[own]
      else:
        source = fact.axes[own] if own >= 0 else set()
        axes[position] |= {(name, None) for name in fact.inputs} | {
          (name, None) for name, _ in source
        }
  return axes


class _FactMachine(tensor_enclosure._Machine):
  """The enclosure's statement machine over relation facts."""

  def __init__(self, source, parameters, scalars):
    super().__init__(source, {}, scalars)
    self.equations = []
    self.values = {}
    for record in parameters:
      if record["dtype"] == "f32":
        self.values[record["name"]] = _input(record["name"], record["shape"])
    for name, shape in re.findall(
      r"let mut (%\w+): tensor \[([\d, ]+)\] f32 = \{", source
    ):
      self.values[name] = _Fact(ZERO, [int(d) for d in shape.split(",")])

  def operand(self, text):
    text = text.strip()
    return self.values[text] if text.startswith("%") else _literal(text)

  def equal(self, *facts):
    forms = [fact.form for fact in facts if fact.form not in (ZERO, UNKNOWN)]
    for left, right in zip(forms, forms[1:]):
      self.equations.append((left, right))

  def combine(self, op, operands, form, *, producer=None):
    shape = _broadcast_shape([fact.shape for fact in operands])
    result = _Fact(
      form,
      shape,
      ops=frozenset({op}).union(*(fact.ops for fact in operands)),
      producer=producer or op,
      axes=_aligned_axes(operands, shape),
    )
    for fact in operands:
      result.tainted |= fact.tainted
      result.reduced |= fact.reduced
    return result

  def tensor(self, op, rest):
    if op == "narrow":
      fmt, name = rest.split()
      fact = self.values[name]
      return self.combine(f"narrow {fmt}", [fact], fact.form)
    if op == "where":
      relation, names = rest.split(" ", 1)
      a, b, x, y = (self.operand(name) for name in names.split(","))
      self.equal(a, b)
      self.equal(x, y)
      form = x.form if x.form != ZERO else y.form
      return self.combine(f"where {relation}", [a, b, x, y], form)
    if op in ("sum", "rmin", "rmax", "cumsum"):
      parts = [part.strip() for part in rest.split(",")]
      return self.reduce(
        op, self.values[parts[0]], int(parts[1]), parts[2:] == ["keep"]
      )
    if op == "transpose":
      name, permutation = re.fullmatch(r"(%\w+), \[([\d, ]+)\]", rest).groups()
      return self.transpose(self.values[name], [int(p) for p in permutation.split(",")])
    if op == "expand":
      name, axis = (part.strip() for part in rest.split(","))
      return self.expand(self.values[name], int(axis))
    if op == "broadcast":
      name, shape = re.fullmatch(r"(%\w+), \[([\d, ]+)\]", rest).groups()
      fact = self.values[name]
      target = _Fact(fact.form, [int(d) for d in shape.split(",")])
      result = self.combine(
        "broadcast", [fact, target], fact.form, producer=fact.producer
      )
      result.ops = fact.ops
      return result
    if op == "window":
      return self.window(rest)
    if op == "dot":
      a, b = (self.operand(name) for name in rest.split(","))
      return self.dot(a, b)
    left, right = (self.operand(name) for name in rest.split(","))
    if op in ("add", "sub", "min", "max", "rem"):
      self.equal(left, right)
      form = left.form if left.form != ZERO else right.form
      if op == "rem" and ZERO in (left.form, right.form):
        form = ZERO
    elif op == "mul":
      form = _add_forms(left.form, right.form)
    elif op == "div":
      form = _add_forms(left.form, right.form, sign=-1)
    else:
      raise tensor_enclosure.Unsupported(f"tensor {op}")
    return self.combine(op, [left, right], form)

  def reduce(self, op, fact, axis, keep):
    axis %= len(fact.shape)
    result = self.combine(op, [fact], fact.form)
    result.tainted |= {entry for entry in fact.axes[axis] if entry[1] is not None}
    result.reduced.add(axis)
    if op == "cumsum":
      result.axes[axis] = set()
      return result
    result.shape = tuple(1 if a == axis else d for a, d in enumerate(fact.shape))
    result.axes[axis] = set()
    if not keep:
      result.shape = result.shape[:axis] + result.shape[axis + 1 :]
      del result.axes[axis]
    return result

  def transpose(self, fact, permutation):
    result = self.combine("transpose", [fact], fact.form, producer=fact.producer)
    result.ops = fact.ops
    result.shape = tuple(fact.shape[p] for p in permutation)
    result.axes = [set(fact.axes[p]) for p in permutation]
    return result

  def expand(self, fact, axis):
    result = self.combine("expand", [fact], fact.form, producer=fact.producer)
    result.ops = fact.ops
    axis %= len(fact.shape) + 1
    result.shape = fact.shape[:axis] + (1,) + fact.shape[axis:]
    result.axes = fact.axes[:axis] + [set()] + fact.axes[axis:]
    return result

  def window(self, rest):
    match = re.fullmatch(r"(%\w+), (%\w+), ([^,]+), \[(\d+), (\d+), (\d+)\]", rest)
    source = self.values[match.group(1)]
    fill = _literal(match.group(3))
    self.equal(source, fill)
    rows = int(match.group(4))
    result = self.combine("window", [source], source.form, producer=source.producer)
    result.ops = source.ops
    result.shape = (rows,) + source.shape[1:]
    # The rows a trip reads depend on the induction variable, so no row of
    # the source maps onto a fixed row of the tile.
    result.tainted |= {entry for entry in source.axes[0] if entry[1] is not None}
    result.axes = [set()] + [set(entries) for entries in source.axes[1:]]
    return result

  def dot(self, a, b):
    result = self.combine("dot", [a, b], _add_forms(a.form, b.form))
    result.shape = (a.shape[0], b.shape[1])
    result.tainted |= {e for e in a.axes[1] | b.axes[0] if e[1] is not None}
    result.axes = [set(a.axes[0]), set(b.axes[1])]
    return result


def _nullspace(rows, count):
  """A basis of {w : rows . w = 0} over the rationals, by Gauss-Jordan."""
  rows = [[Fraction(v) for v in row] for row in rows]
  pivots, rank = [], 0
  for column in range(count):
    pivot = next((r for r in range(rank, len(rows)) if rows[r][column]), None)
    if pivot is None:
      continue
    rows[rank], rows[pivot] = rows[pivot], rows[rank]
    lead = rows[rank][column]
    rows[rank] = [v / lead for v in rows[rank]]
    for r in range(len(rows)):
      if r != rank and rows[r][column]:
        factor = rows[r][column]
        rows[r] = [a - factor * b for a, b in zip(rows[r], rows[rank])]
    pivots.append(column)
    rank += 1
  basis = []
  for free in (c for c in range(count) if c not in pivots):
    vector = [Fraction(0)] * count
    vector[free] = Fraction(1)
    for row, column in zip(rows, pivots):
      vector[column] = -row[free]
    basis.append(vector)
  return basis


def _integral(vector):
  scale = math.lcm(*(v.denominator for v in vector))
  values = [int(v * scale) for v in vector]
  divisor = math.gcd(*values) or 1
  values = [v // divisor for v in values]
  first = next((v for v in values if v), 0)
  return [-v for v in values] if first < 0 else values


def _weights(equations, names):
  """Integer weights solving every equation: all ones when they do, else a
  generic combination of the nullspace, which scales the most inputs; None
  when only all zeros solve."""
  index = {name: i for i, name in enumerate(names)}
  rows = []
  for left, right in equations:
    row = [0] * len(names)
    for name, coefficient in left.items():
      row[index[name]] += coefficient
    for name, coefficient in right.items():
      row[index[name]] -= coefficient
    rows.append(row)
  if all(sum(row) == 0 for row in rows):
    return dict.fromkeys(names, 1)
  basis = _nullspace(rows, len(names))
  if not basis:
    return None
  # Distinct prime coefficients keep accidental cancellations out.
  primes = (1, 2, 3, 5, 7, 11, 13, 17, 19, 23)
  combined = [
    sum(primes[k] * vector[i] for k, vector in enumerate(basis))
    for i in range(len(names))
  ]
  return dict(zip(names, _integral(combined)))


def _parities(equations, names):
  """Signs (0 or 1) solving the equations mod 2, preferring every input
  negated; None when only all zeros solve."""
  best = None
  for candidate in itertools.product(range(2), repeat=len(names)):
    signs = dict(zip(names, candidate))
    if all(
      sum(c * signs[n] for n, c in left.items()) % 2
      == sum(c * signs[n] for n, c in right.items()) % 2
      for left, right in equations
    ) and any(candidate):
      if best is None or sum(candidate) > sum(best.values()):
        best = signs
  return best


def _evaluate(form, weights, modulus=None):
  if weights is None or form in (ZERO, UNKNOWN):
    return None
  total = sum(c * weights[n] for n, c in form.items())
  return total % modulus if modulus else total


def _class(fact, element, narrow_bf16):
  ops = set(fact.ops)
  if ops & EXCLUDED_OPS or (narrow_bf16 == "excluded" and "narrow bf16" in ops):
    return "excluded"
  if ops & ORDER_DEPENDENT_OPS or ("dot" in ops and element == "f32"):
    return "order_dependent"
  return "specified"


def _axis_lists(out, weights):
  """Per input, the axes a permutation, and a per-index factor, may act on."""
  permutable, scalable = {}, {}
  for position, entries in enumerate(out.axes):
    clean = all(entry not in out.tainted for entry in entries if entry[1] is not None)
    broadcast = any(
      axis is None and weights and weights.get(name, 0) for name, axis in entries
    )
    for name, axis in entries:
      if axis is None or axis != position or not clean:
        continue
      permutable.setdefault(name, []).append(axis)
      if not broadcast:
        scalable.setdefault(name, []).append(axis)
  return permutable, scalable


def _argument(name):
  return f"{name.lstrip('%')}_ptr"


def analyze(source, parameters, scalars=None, *, element="f32", narrow_bf16="excluded"):
  """The sidecar's `inputs` and `outputs` for one body on one path.

  `parameters` are the region's parameter records; `scalars` gives the values
  of its i32 scalars (trip count, branch selector). `narrow_bf16` is the class
  `narrow bf16` has on the host that runs, `specified` only once its PTX
  evidence shows one round-to-nearest-even conversion.
  """
  machine = _FactMachine(source, parameters, scalars)
  out = machine.run()
  names = [r["name"] for r in parameters if r["dtype"] == "f32"]
  weights = _weights(machine.equations, names) if names else None
  parities = _parities(machine.equations, names) if names else None
  cls = _class(out, element, narrow_bf16)
  permutable, scalable = _axis_lists(out, weights)
  inputs = {}
  for record in parameters:
    name = record["name"]
    if record["dtype"] != "f32":
      inputs[_argument(name)] = {"float": False}
      continue
    inputs[_argument(name)] = {
      "float": True,
      "weight": None if weights is None else weights[name],
      "parity": None if parities is None else parities[name],
      "permutable_axes": sorted(permutable.get(name, [])),
      "scale_axes": sorted(scalable.get(name, [])),
    }
  excluded = cls == "excluded"
  output = {
    "degree": None if excluded else _evaluate(out.form, weights),
    "parity": None if excluded else _evaluate(out.form, parities, modulus=2),
    "reduced_axes": sorted(out.reduced),
    "producer_op": out.producer,
    "class": cls,
    "slice_ops": sorted(out.ops),
  }
  return {"inputs": inputs, "outputs": {"out_ptr": output}}
