"""Runtime inputs for a free-form region, derived from its record alone.

The free-form route states the shapes, dtype, and roles it needs. Input values
therefore come from that record rather than from the canonical certificate
parser, so a body the certificate grammar cannot express still receives inputs.
The values are small integers held exactly in f32, which keeps a mismatch
between two execution paths a behavioural difference rather than rounding.
"""

import hashlib
import struct
from dataclasses import dataclass

PROPOSAL_SCHEMA = "refractir.free-form-input-proposal.v1"
PATTERNS = ("distinct-integer", "mixed-sign", "repeated-value", "seeded-random")
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
