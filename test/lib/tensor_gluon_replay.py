"""Execute the operations one emitted Gluon kernel states, on the CPU.

The emitter is proved by execution rather than by reading the source it writes.
The replay evaluates the emitted operations, treating every layout conversion as
the logical identity it stands for, so a wrong mask shows up as a wrong sum. It
also records the shape of every mask beside the shape of the operand that mask
guards: a mask wider than its operand broadcasts that operand back across lanes
the reduction must not count, and the recorded shape pair states that without
anyone reading the emitted text.

Only the standard library is used, so the emitter's unit suite runs wherever the
exact reference does. Broadcasting follows the region grammar's own numpy-style
rule, which `test.lib.tensor_exact_f32` already implements.
"""

from dataclasses import dataclass

from test.lib.tensor_exact_f32 import _broadcast_shape, _broadcast_values


class _Array:
  """A rank-one or rank-two emitted value, flat and row-major.

  The emitted surface is small: aranges and their reshapes, comparisons,
  elementwise arithmetic, reductions, broadcasts, and the two pointers.
  Everything a kernel can write is one of those, so this class states exactly
  those operations instead of a general array library.
  """

  __slots__ = ("shape", "values")

  def __init__(self, shape, values):
    self.shape = tuple(shape)
    self.values = tuple(values)

  def _elementwise(self, other, operation):
    """One emitted binary operation, with the grammar's own broadcasting."""
    if not isinstance(other, _Array):
      return _Array(self.shape, [operation(value, other) for value in self.values])
    shape = _common_shape(self.shape, other.shape)
    left = _broadcast_values(self.values, self.shape, shape)
    right = _broadcast_values(other.values, other.shape, shape)
    return _Array(shape, [operation(a, b) for a, b in zip(left, right)])

  def __lt__(self, bound):
    return self._elementwise(bound, lambda value, other: value < other)

  def __and__(self, other):
    return self._elementwise(other, lambda left, right: bool(left) and bool(right))

  def __add__(self, other):
    return self._elementwise(other, lambda left, right: left + right)

  def __radd__(self, other):
    return self.__add__(other)

  def __sub__(self, other):
    return self._elementwise(other, lambda left, right: left - right)

  def __mul__(self, other):
    return self._elementwise(other, lambda left, right: left * right)

  def __rmul__(self, other):
    return self.__mul__(other)

  def __getitem__(self, key):
    """The emitted reshapes: a column vector and a row vector."""
    if key == (slice(None), None):
      return _Array((self.shape[0], 1), self.values)
    if key == (None, slice(None)):
      return _Array((1, self.shape[-1]), self.values)
    raise IndexError(f"emitted index {key!r} is outside the replayed surface")


def _volume(shape):
  volume = 1
  for extent in shape:
    volume *= extent
  return volume


def _common_shape(*shapes):
  """The shape a set of emitted operands broadcasts to, or a clear refusal."""
  common = shapes[0]
  for shape in shapes[1:]:
    try:
      common = _broadcast_shape(common, shape)
    except ValueError as error:
      raise ValueError(f"emitted shapes {shapes} do not broadcast") from error
  return common


def _as_array(value):
  """One emitted argument as an array: a literal scalar is a rank-zero value."""
  return value if isinstance(value, _Array) else _Array((), (value,))


def _record(records, entry):
  """Keep one entry per distinct emitted operation, in first-use order.

  A mask written once in a loop body is executed once per trip, and what the
  tests need to see is the set of masks the kernel builds, not how often the
  loop ran.
  """
  if entry not in records:
    records.append(entry)


def _broadcast(array, shape):
  """One array's values at the extents of a shape it broadcasts to."""
  return _Array(shape, _broadcast_values(array.values, array.shape, shape))


def _reduce(values, axis):
  """Sum one rank-two value over one axis, keeping the other axis in order."""
  rows, columns = values.shape
  if axis == 0:
    return _Array(
      (columns,),
      [
        sum(values.values[row * columns + column] for row in range(rows))
        for column in range(columns)
      ],
    )
  return _Array(
    (rows,),
    [sum(values.values[row * columns : (row + 1) * columns]) for row in range(rows)],
  )


def _expand_dims(values, axis):
  """Restore the reduced axis the emitter keeps at extent one."""
  shape = (1, values.shape[0]) if axis == 0 else (values.shape[0], 1)
  return _Array(shape, values.values)


class _Pointer:
  """A flat logical buffer and the offset the kernel's addressing computes."""

  def __init__(self, data, offset=0):
    self.data = data
    self.offset = offset

  def __add__(self, offset):
    return _Pointer(self.data, offset)


class _Surface:
  """The identifiers the emitted kernel reads from `gl`.

  Layout conversions are the logical identity they stand for, and a layout
  object is only ever a kernel argument, so the surface ignores every layout.
  """

  constexpr = object
  float32 = "float32"

  def __init__(self, where_calls, sum_calls):
    self._where_calls = where_calls
    self._sum_calls = sum_calls

  def SliceLayout(self, *args, **kwargs):
    return None

  def convert_layout(self, value, layout):
    return value

  def zeros(self, shape, dtype, **kwargs):
    return _Array(shape, [0.0] * _volume(shape))

  def arange(self, start, end, **kwargs):
    return _Array((end - start,), list(range(start, end)))

  def load(self, pointer, mask=None, other=0.0):
    addresses = pointer.offset.values
    if mask is None:
      return _Array(
        pointer.offset.shape, [pointer.data[address] for address in addresses]
      )
    return _Array(
      pointer.offset.shape,
      [
        pointer.data[address] if keep else other
        for address, keep in zip(addresses, mask.values)
      ],
    )

  def store(self, pointer, values, mask=None):
    keeps = mask.values if mask is not None else [True] * len(values.values)
    for address, value, keep in zip(pointer.offset.values, values.values, keeps):
      if keep:
        pointer.data[address] = value

  def where(self, condition, left, right):
    _record(self._where_calls, (_as_array(condition).shape, _as_array(left).shape))
    condition, left, right = (_as_array(value) for value in (condition, left, right))
    shape = _common_shape(condition.shape, left.shape, right.shape)
    return _Array(
      shape,
      [
        chosen if keep else rejected
        for chosen, rejected, keep in zip(
          _broadcast_values(left.values, left.shape, shape),
          _broadcast_values(right.values, right.shape, shape),
          _broadcast_values(condition.values, condition.shape, shape),
        )
      ],
    )

  def sum(self, values, axis):
    _record(self._sum_calls, (int(axis), values.shape))
    return _reduce(values, axis)

  def expand_dims(self, values, axis):
    return _expand_dims(values, axis)

  def broadcast(self, left, right):
    left, right = _as_array(left), _as_array(right)
    shape = _common_shape(left.shape, right.shape)
    return (_broadcast(left, shape), _broadcast(right, shape))


@dataclass(frozen=True)
class GluonReplay:
  """The logical output one emitted kernel computed, and the shapes it used.

  `output` is the logical row-major buffer the kernel stored. `where_calls`
  pairs the shape of each distinct `gl.where` mask with the shape of the operand
  that mask guards, and `sum_calls` pairs each distinct `gl.sum` axis with the
  shape of the summand it reduced; both keep first-use order, so a mask written
  inside the loop body is recorded once however many trips the loop made.
  """

  output: tuple
  where_calls: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
  sum_calls: tuple[tuple[int, tuple[int, ...]], ...]

  def masks_match_operands(self):
    """True when every mask is exactly as wide as the operand it guards."""
    return all(mask_shape == operand for mask_shape, operand in self.where_calls)


def replay_gluon_kernel(generated, values, *, layouts=None):
  """Run one emitted kernel on the CPU and report the shapes it observed.

  `values` is the logical row-major input buffer, so it holds one element per
  logical lane and no padded lane: the kernel's own load mask is what keeps a
  padded address from being dereferenced, exactly as on the device.
  """
  logical = [float(value) for value in values]
  if len(logical) != _volume(generated.shape):
    raise ValueError("replay input does not cover the logical extent")
  output = [0.0] * len(logical)
  where_calls = []
  sum_calls = []
  body = "\n".join(
    line
    for line in generated.source.splitlines()
    if not line.startswith(("import ", "from ", "@"))
  )
  namespace = {"gl": _Surface(where_calls, sum_calls)}
  exec(compile(body, "<emitted gluon kernel>", "exec"), namespace)
  arguments = (
    tuple(layouts)
    if layouts is not None
    else (None,) * len(generated.layout_parameters)
  )
  namespace["generated_region_kernel"](_Pointer(logical), _Pointer(output), *arguments)
  return GluonReplay(tuple(output), tuple(where_calls), tuple(sum_calls))
