"""Runtime inputs for a free-form region, derived from its record alone.

The free-form route states the shapes, dtype, and roles it needs. Input values
therefore come from that record rather than from the canonical certificate
parser, so a body the certificate grammar cannot express still receives inputs.
The values are small integers chosen to be exact in f32, which keeps a mismatch
between two execution paths a behavioural difference rather than an input
rounding. Exact input words do not establish that an execution is exact:
intermediate f32 operations may round, so the study compares observed outputs
rather than assuming a closed form.

Window probes are role-aware: the backing parameter is perturbed while the two
state tensors stay bit-identical, and each probe states its own pattern and
probe name so a study record can reference it.
"""

import hashlib
import struct
from dataclasses import dataclass

PROPOSAL_SCHEMA = "refractir.free-form-input-proposal.v1"
WINDOW_PROBE_SCHEMA = "refractir.free-form-window-probes.v1"
PATTERNS = ("distinct-integer", "mixed-sign", "repeated-value", "seeded-random")
WINDOW_PROBES = ("i0-base", "i1-read-plus-one", "i2-late-row", "i3-unused-rows")
CORNER_PROBES = ("i4-signed-zero", "i5-denormal", "i6-cancel", "i7-ties", "i8-large")
_MASK = 0xFFFFFFFFFFFFFFFF


class SpecError(ValueError):
  """The region record does not state inputs a proposal can honour."""


@dataclass(frozen=True)
class Proposal:
  """One frozen input set, with the values a study record can name."""

  schema: str
  pattern: str
  seed: int
  names: tuple
  shapes: tuple
  dtype: str
  values: tuple
  data: bytes
  probe: str | None = None

  def as_record(self):
    """The JSON-able statement of this proposal, without the payload."""
    return {
      "schema": self.schema,
      "pattern": self.pattern,
      "seed": self.seed,
      "dtype": self.dtype,
      "names": list(self.names),
      "shapes": [list(shape) for shape in self.shapes],
      "values": [list(row) for row in self.values],
      "byte_length": len(self.data),
      "sha256": hashlib.sha256(self.data).hexdigest(),
      "probe": self.probe,
    }


def _element_count(shape):
  count = 1
  for extent in shape:
    count *= extent
  return count


def parameters(spec):
  """The runtime parameters a region needs, in the recorded order."""
  if not isinstance(spec, dict):
    raise SpecError("region record must be an object")
  records = spec.get("parameters")
  if not isinstance(records, list) or not records:
    raise SpecError("region record needs at least one runtime parameter")
  result = []
  names = set()
  for record in records:
    if not isinstance(record, dict):
      raise SpecError("each parameter record must be an object")
    name = record.get("name")
    if not isinstance(name, str) or not name:
      raise SpecError("parameter name must be nonempty text")
    if name in names:
      raise SpecError(f"parameter {name} is declared twice")
    names.add(name)
    shape = record.get("shape")
    if (
      not isinstance(shape, list)
      or not shape
      or any(type(extent) is not int or extent <= 0 for extent in shape)
    ):
      raise SpecError(f"parameter {name} needs a positive shape")
    dtype = record.get("dtype")
    if dtype not in ("f32", "i32"):
      raise SpecError(f"parameter {name} has unsupported dtype {dtype!r}")
    result.append({"name": name, "shape": list(shape), "dtype": dtype})
  return result


def _splitmix64(value):
  value = (value + 0x9E3779B97F4A7C15) & _MASK
  value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
  value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK
  return value ^ (value >> 31)


def _pattern_values(pattern, count, seed):
  """Small integers, so every element is exact in f32 and i32."""
  values = []
  for index in range(count):
    if pattern == "distinct-integer":
      values.append(index % 7 + 1)
    elif pattern == "mixed-sign":
      magnitude = index % 5 + 1
      values.append(magnitude if index % 2 == 0 else -magnitude)
    elif pattern == "repeated-value":
      values.append(3)
    elif pattern == "seeded-random":
      values.append(_splitmix64(seed + index) % 9 - 4)
    else:
      raise SpecError(f"unknown free-form input pattern: {pattern}")
  return values


def proposal(spec, *, pattern, seed):
  """One proposal for the region's parameters under the stated pattern."""
  if pattern not in PATTERNS:
    raise SpecError(f"unknown free-form input pattern: {pattern}")
  if type(seed) is not int or seed < 0:
    raise SpecError("proposal seed must be a nonnegative integer")
  records = parameters(spec)
  dtype = records[0]["dtype"]
  for record in records:
    if record["dtype"] != dtype:
      raise SpecError("every runtime parameter must share one dtype")
  payload = bytearray()
  names = []
  shapes = []
  rows = []
  for record in records:
    count = _element_count(record["shape"])
    row = tuple(_pattern_values(pattern, count, seed + len(names)))
    names.append(record["name"])
    shapes.append(tuple(record["shape"]))
    rows.append(row)
    if dtype == "f32":
      payload.extend(struct.pack(f"<{count}f", *row))
    else:
      payload.extend(struct.pack(f"<{count}i", *row))
  return Proposal(
    schema=PROPOSAL_SCHEMA,
    pattern=pattern,
    seed=seed,
    names=tuple(names),
    shapes=tuple(shapes),
    dtype=dtype,
    values=tuple(rows),
    data=bytes(payload),
  )


def derive_proposals_from_spec(spec, *, count=1, seed=0, patterns=None):
  """`count` proposals for one region record, one per requested pattern.

  A caller that asks for more proposals than it names patterns gets the
  patterns again under a later seed, so two proposals never carry the same
  values unless the pattern itself does.
  """
  if type(count) is not int or count <= 0:
    raise SpecError("proposal count must be a positive integer")
  if type(seed) is not int or seed < 0:
    raise SpecError("proposal seed must be a nonnegative integer")
  sequence = tuple(patterns or PATTERNS)
  if not sequence:
    raise SpecError("proposal patterns must not be empty")
  return [
    proposal(spec, pattern=sequence[index % len(sequence)], seed=seed + index)
    for index in range(count)
  ]



def _window_roles(region):
  """Resolve the window backing parameter, the two states, and the window."""
  records = parameters(region)
  source = region.get("window_source")
  if not isinstance(source, dict):
    raise SpecError("a window region needs a window_source record")
  backing_name = source.get("parameter")
  records_by_name = {record["name"]: record for record in records}
  if backing_name not in records_by_name:
    raise SpecError("the window_source parameter is not a runtime parameter")
  backing = records_by_name[backing_name]
  if backing.get("dtype") != "f32":
    raise SpecError("window probes need f32 parameters")
  if len(backing["shape"]) != 2:
    raise SpecError("the window backing parameter must be a matrix")
  states = [record for record in records if record["name"] != backing_name]
  if len(states) != 2:
    raise SpecError("window probes need exactly two state parameters")
  if any(state["dtype"] != "f32" for state in states):
    raise SpecError("window probes need f32 states")
  for state in states:
    if len(state["shape"]) != 2:
      raise SpecError("window state parameters must be matrices")
  if states[0]["shape"] != states[1]["shape"]:
    raise SpecError("window state parameters must share one shape")
  window = region.get("window")
  if not isinstance(window, dict):
    raise SpecError("a window region needs a window record")
  for field in ("rows", "stride", "active"):
    if type(window.get(field)) is not int or window[field] < 0:
      raise SpecError(f"window {field} must be a nonnegative integer")
  if window["rows"] == 0 or window["active"] == 0:
    raise SpecError("window rows and active must be positive")
  trips = region.get("trip_count")
  if type(trips) is not int or trips < 1:
    raise SpecError("a window region needs a positive trip_count")
  rows, columns = backing["shape"]
  if states[0]["shape"] != [window["rows"], columns]:
    raise SpecError("the state shape must match the window rows and backing columns")
  if window["active"] > rows:
    raise SpecError("window active must fit the backing rows")
  if (trips - 1) * window["stride"] + window["rows"] > rows:
    raise SpecError("the window witness must fit the backing rows")
  return {
    "parameters": records,
    "backing": backing,
    "states": states,
    "window": window,
    "trips": trips,
  }


def _slot(group, column):
  """One source slot per (group, column).

  Columns 0-3 keep the slots the controlled study froze; the one-trip columns
  4 and 5 follow all of them, so no earlier probe changes its bytes.
  """
  if type(group) is not int or not 0 <= group < 3:
    raise SpecError("window probe group must be in [0, 3)")
  if type(column) is not int or not 0 <= column < 6:
    raise SpecError("window probe column must be in [0, 6)")
  return 4 * group + column if column < 4 else 12 + 2 * group + column - 4


def _window_rows(roles, *, group, column):
  """The base backing and state values for one source slot."""
  slot = _slot(group, column)
  rows, columns = roles["backing"]["shape"]
  state_rows = roles["window"]["rows"]
  states = roles["states"]
  state_zero = [
    [((r + 2 * c) % 5) - 2 for c in range(columns)]
    for r in range(state_rows)
  ]
  state_one = [
    [((2 * r + c + 1) % 5) - 2 for c in range(columns)] for r in range(state_rows)
  ]
  base = [[((3 * r + c + slot) % 7) - 3 for c in range(columns)] for r in range(rows)]
  values = {
    roles["backing"]["name"]: base,
    states[0]["name"]: state_zero,
    states[1]["name"]: state_one,
  }
  return slot, values


def _pack_window_proposal(roles, values, *, pattern, probe, seed):
  """Pack one probe's rows in the region's recorded parameter order."""
  names = tuple(record["name"] for record in roles["parameters"])
  shapes = tuple(tuple(record["shape"]) for record in roles["parameters"])
  payload = bytearray()
  rows = tuple(tuple(tuple(values[name][r]) for r in range(len(values[name]))) for name in names)
  for row in rows:
    for values_row in row:
      payload.extend(struct.pack(f"<{len(values_row)}f", *values_row))
  return Proposal(
    schema=WINDOW_PROBE_SCHEMA,
    pattern=pattern,
    seed=seed,
    names=names,
    shapes=shapes,
    dtype="f32",
    values=rows,
    data=bytes(payload),
    probe=probe,
  )


def _read_positions(roles):
  """Every backing coordinate a trip and row read under the witness."""
  window = roles["window"]
  read = set()
  for trip in range(roles["trips"]):
    for row in range(window["rows"]):
      address = trip * window["stride"] + row
      if address < window["active"]:
        read.add(address)
  return read


def window_base_proposal(region, *, group, column):
  """The I0 probe: role-separated states and backing, before any perturbation."""
  roles = _window_roles(region)
  slot, values = _window_rows(roles, group=group, column=column)
  return _pack_window_proposal(
    roles, values, pattern="window-base", probe="i0-base", seed=slot
  )


def window_probes(region, *, group, column):
  """The frozen window probes I0-I3 for one source slot.

  I1 adds one to every backing position any trip reads, I2 replaces the last
  active row, and I3 flips the rows at or above the active bound that no trip
  reads. The states are identical in all of them. I2 needs a later trip to read
  a row the first window does not, so a one-trip window has no I2.
  """
  roles = _window_roles(region)
  slot, base_values = _window_rows(roles, group=group, column=column)
  window = roles["window"]
  rows, columns = roles["backing"]["shape"]
  backing_name = roles["backing"]["name"]
  read = _read_positions(roles)
  late = window["active"] - 1
  one_trip = roles["trips"] == 1
  if late < window["rows"] and not one_trip:
    raise SpecError("the late row is read by the first window")
  if late not in read:
    raise SpecError("the late row is not reachable by any trip")
  if window["active"] > rows:
    raise SpecError("window active must fit the backing rows")

  def clone(values, backing):
    return {
      backing_name: backing,
      roles["states"][0]["name"]: [list(row) for row in values[roles["states"][0]["name"]]],
      roles["states"][1]["name"]: [list(row) for row in values[roles["states"][1]["name"]]],
    }

  base = clone(base_values, [list(row) for row in base_values[backing_name]])
  read_plus = clone(base_values, [list(row) for row in base_values[backing_name]])
  for address in read:
    for column in range(columns):
      read_plus[backing_name][address][column] += 1
  late_row = [(-5 if column % 2 == 0 else 5) for column in range(columns)]
  late_probe = clone(base_values, [list(row) for row in base_values[backing_name]])
  late_probe[backing_name][late] = late_row
  unused = clone(base_values, [list(row) for row in base_values[backing_name]])
  for address in range(window["active"], rows):
    unused[backing_name][address] = [1 - value for value in unused[backing_name][address]]
  probes = (
    _pack_window_proposal(roles, base, pattern="window-base", probe="i0-base", seed=slot),
    _pack_window_proposal(
      roles, read_plus, pattern="window-read-plus-one", probe="i1-read-plus-one", seed=slot
    ),
    _pack_window_proposal(
      roles, late_probe, pattern="window-late-row", probe="i2-late-row", seed=slot
    ),
    _pack_window_proposal(
      roles, unused, pattern="window-unused-rows", probe="i3-unused-rows", seed=slot
    ),
  )
  return tuple(probe for probe in probes if not (one_trip and probe.probe == "i2-late-row"))


def _odd(value):
  """1 or 3 from the base value: 2^24 plus either is a rounding tie, and
  round-to-nearest-even takes 2^24 + 1 down and 2^24 + 3 up. (Below 2^24 the
  spacing is 1, so 2^24 minus an odd integer is exact, not a tie.)"""
  return 1 if abs(value) % 4 < 2 else 3


# Each corner recipe maps one I0 element, at row r and column c of its
# parameter, to a value that exercises one floating-point corner. Even (r + c)
# and odd (r + c) elements alternate, so neighbours in a row or a column pair
# up. Every value is exactly representable in f32.
_CORNERS = {
  # Half the elements become a zero carrying the base value's sign.
  "i4-signed-zero": lambda v, even: (-0.0 if v < 0 else 0.0) if even else float(v),
  # 2^-127 is subnormal; 2 and 3 times it are normal: values straddle FLT_MIN.
  "i5-denormal": lambda v, even: v * 2.0**-127,
  # Large values of opposite sign whose sum cancels to the small base values.
  "i6-cancel": lambda v, even: (2.0**23 if even else -(2.0**23)) + v,
  # 2^24 beside 1 or 3: their sum lies halfway between two floats.
  "i7-ties": lambda v, even: 2.0**24 if even else float(_odd(v)),
  # Magnitudes near 2^30 beside magnitudes near 2^24.
  "i8-large": lambda v, even: v * (2.0**28 if even else 2.0**22),
}


def corner_probes(region, *, group, column):
  """The corner-value probes I4-I8 for one source slot, derived from I0.

  Unlike I0-I3 they are not bounded by the exact-input contract: they put
  signed zeros, subnormals, cancellation, rounding ties, and large magnitudes
  into every parameter, the window backing and both states alike.
  """
  roles = _window_roles(region)
  slot, base = _window_rows(roles, group=group, column=column)
  return tuple(
    _pack_window_proposal(
      roles,
      {
        name: [
          [_CORNERS[probe](value, (r + c) % 2 == 0) for c, value in enumerate(row)]
          for r, row in enumerate(matrix)
        ]
        for name, matrix in base.items()
      },
      pattern=f"window-{probe[3:]}",
      probe=probe,
      seed=slot,
    )
    for probe in CORNER_PROBES
  )
