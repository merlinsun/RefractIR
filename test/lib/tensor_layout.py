"""Stride-aware storage helpers for single-program tensor cases."""

from dataclasses import dataclass
from itertools import product


class LayoutError(ValueError):
  """A tensor layout is malformed, overlapping, or outside its allocation."""


def _integer(value, name, minimum):
  if type(value) is not int or value < minimum:
    raise LayoutError(f"{name} must be an integer >= {minimum}")
  return value


def _shape(shape, profile="v2"):
  if profile not in {"v1", "v2", "v3"}:
    raise LayoutError(f"unknown tensor profile {profile!r}")
  max_rank = 2 if profile == "v3" else 3
  if not isinstance(shape, list) or not 1 <= len(shape) <= max_rank:
    raise LayoutError(f"shape rank must be in [1, {max_rank}]")
  result = [
    _integer(value, f"shape dimension {axis}", 1) for axis, value in enumerate(shape)
  ]
  count = 1
  for value in result:
    if profile != "v3" and value & (value - 1):
      raise LayoutError("shape dimensions must be powers of two")
    count *= value
  if profile == "v3":
    tile_count = 1
    for value in result:
      tile_count *= 1 << (value - 1).bit_length()
    if tile_count > 512:
      raise LayoutError("padded tile volume exceeds 512 elements")
  else:
    limit = 256 if profile == "v1" else 512
    if count > limit:
      raise LayoutError(f"shape exceeds the tensor-{profile} {limit}-element limit")
  return result


def execution_tile_shape(shape, *, profile="v2"):
  """Return the execution extents derived from a logical tensor shape."""
  logical = _shape(shape, profile)
  if profile != "v3":
    return logical
  return [1 << (value - 1).bit_length() for value in logical]


def logical_addresses(layout, *, profile="v2"):
  """Return row-major logical indices mapped to unique physical element offsets."""
  if not isinstance(layout, dict):
    raise LayoutError("layout must be an object")
  shape = _shape(layout.get("shape"), profile)
  strides = layout.get("element_strides")
  if not isinstance(strides, list) or len(strides) != len(shape):
    raise LayoutError("element strides must match shape rank")
  for axis, value in enumerate(strides):
    if type(value) is not int:
      raise LayoutError(f"element stride {axis} must be an integer")
    if value <= 0:
      raise LayoutError("element strides must be positive")
  axis_order = layout.get("axis_order")
  if (
    not isinstance(axis_order, list)
    or len(axis_order) != len(shape)
    or any(type(axis) is not int for axis in axis_order)
    or sorted(axis_order) != list(range(len(shape)))
  ):
    raise LayoutError("axis order must be a permutation of tensor axes")
  if sorted(range(len(shape)), key=lambda axis: (strides[axis], axis)) != axis_order:
    raise LayoutError("axis order must list axes from fastest to slowest stride")
  element_offset = _integer(layout.get("element_offset"), "element offset", 0)
  storage_span = _integer(layout.get("storage_span"), "storage span", 1)
  addresses = [
    element_offset
    + sum(coordinate * strides[axis] for axis, coordinate in enumerate(index))
    for index in product(*(range(size) for size in shape))
  ]
  if len(set(addresses)) != len(addresses):
    raise LayoutError("logical tensor coordinates overlap in physical storage")
  if max(addresses, default=element_offset) >= storage_span:
    raise LayoutError("logical tensor address exceeds storage span")
  return addresses


def _element_size(element_size):
  return _integer(element_size, "element size", 1)


def stage_buffer(logical_bytes, layout, *, element_size, gap_bytes, profile="v2"):
  """Place dense logical input values at their recorded physical addresses."""
  element_size = _element_size(element_size)
  shape = _shape(layout.get("shape"), profile)
  count = 1
  for dimension in shape:
    count *= dimension
  if len(logical_bytes) != count * element_size:
    raise LayoutError("logical input byte length does not match shape")
  if not isinstance(gap_bytes, bytes) or len(gap_bytes) != element_size:
    raise LayoutError("gap sentinel must contain exactly one element")
  addresses = logical_addresses(layout, profile=profile)
  staged = bytearray(gap_bytes * layout["storage_span"])
  for logical_index, address in enumerate(addresses):
    source_begin = logical_index * element_size
    target_begin = address * element_size
    staged[target_begin : target_begin + element_size] = logical_bytes[
      source_begin : source_begin + element_size
    ]
  return bytes(staged)


def gather_buffer(physical_bytes, layout, *, element_size, profile="v2"):
  """Gather a physical buffer into logical row-major element order."""
  element_size = _element_size(element_size)
  addresses = logical_addresses(layout, profile=profile)
  if len(physical_bytes) != layout["storage_span"] * element_size:
    raise LayoutError("physical byte length does not match storage span")
  return b"".join(
    physical_bytes[address * element_size : (address + 1) * element_size]
    for address in addresses
  )


def scatter_buffer(
  logical_bytes, physical_bytes, layout, *, element_size, profile="v2"
):
  """Write logical output values into a canary-filled physical allocation."""
  element_size = _element_size(element_size)
  addresses = logical_addresses(layout, profile=profile)
  if len(physical_bytes) != layout["storage_span"] * element_size:
    raise LayoutError("physical byte length does not match storage span")
  if len(logical_bytes) != len(addresses) * element_size:
    raise LayoutError("logical output byte length does not match shape")
  scattered = bytearray(physical_bytes)
  for logical_index, address in enumerate(addresses):
    source_begin = logical_index * element_size
    target_begin = address * element_size
    scattered[target_begin : target_begin + element_size] = logical_bytes[
      source_begin : source_begin + element_size
    ]
  return bytes(scattered)


def changed_canary_slots(
  before, after, layout, *, element_size, envelope=None, profile="v2"
):
  """Return changed physical slots outside the logical output mapping.

  With a guard envelope the changed slots are named, so a write into a prefix
  guard, a suffix guard, and an internal gap stay distinguishable.
  """
  element_size = _element_size(element_size)
  if envelope is not None:
    return [
      name
      for name in classify_changed_slots(
        before,
        after,
        layout,
        element_size=element_size,
        envelope=envelope,
        profile=profile,
      )
      if not name.startswith("input-")
    ]
  expected_length = layout["storage_span"] * element_size
  if len(before) != expected_length or len(after) != expected_length:
    raise LayoutError("canary buffers do not match storage span")
  logical = set(logical_addresses(layout, profile=profile))
  return [
    slot
    for slot in range(layout["storage_span"])
    if slot not in logical
    and before[slot * element_size : (slot + 1) * element_size]
    != after[slot * element_size : (slot + 1) * element_size]
  ]


@dataclass(frozen=True)
class GuardEnvelope:
  """Allocated element slots around one tensor's storage span.

  The prefix and suffix guards stay outside every recorded address, so a write
  observed there has left the tensor's own allocation. Together with the
  internal gaps they form the nonlogical region of the buffer.
  """

  prefix: int = 0
  suffix: int = 0

  def __post_init__(self):
    _integer(self.prefix, "guard prefix", 0)
    _integer(self.suffix, "guard suffix", 0)


def _envelope_slots(layout, envelope):
  if not isinstance(envelope, GuardEnvelope):
    raise LayoutError("guard envelope must be a GuardEnvelope")
  return envelope.prefix + layout["storage_span"] + envelope.suffix


def _slot_names(layout, envelope, *, profile="v2"):
  """Name every physical slot of an enveloped buffer."""
  logical = logical_addresses(layout, profile=profile)
  names = []
  for slot in range(_envelope_slots(layout, envelope)):
    if slot < envelope.prefix:
      names.append(f"prefix-guard-{slot}")
      continue
    physical = slot - envelope.prefix
    if physical >= layout["storage_span"]:
      names.append(f"suffix-guard-{physical - layout['storage_span']}")
      continue
    names.append(
      f"input-{logical.index(physical)}" if physical in logical else f"gap-{physical}"
    )
  return names


def stage_envelope(
  logical_bytes, layout, *, element_size, envelope, padding, profile="v2"
):
  """Stage logical values inside an allocation with guards and gap padding."""
  element_size = _element_size(element_size)
  if not isinstance(padding, bytes) or len(padding) != element_size:
    raise LayoutError("padding must contain exactly one element")
  staged = bytearray(padding * _envelope_slots(layout, envelope))
  staged[
    envelope.prefix * element_size : (envelope.prefix + layout["storage_span"])
    * element_size
  ] = stage_buffer(
    logical_bytes, layout, element_size=element_size, gap_bytes=padding, profile=profile
  )
  return bytes(staged)


def gather_envelope(physical_bytes, layout, *, element_size, envelope, profile="v2"):
  """Gather the logical values out of an enveloped buffer."""
  element_size = _element_size(element_size)
  if len(physical_bytes) != _envelope_slots(layout, envelope) * element_size:
    raise LayoutError("enveloped byte length does not match guards and storage span")
  begin = envelope.prefix * element_size
  end = (envelope.prefix + layout["storage_span"]) * element_size
  return gather_buffer(
    physical_bytes[begin:end], layout, element_size=element_size, profile=profile
  )


def classify_changed_slots(
  before, after, layout, *, element_size, envelope, profile="v2"
):
  """Name every physical slot whose bytes changed, logical or not."""
  element_size = _element_size(element_size)
  expected_length = _envelope_slots(layout, envelope) * element_size
  if len(before) != expected_length or len(after) != expected_length:
    raise LayoutError("enveloped buffers do not match guards and storage span")
  names = _slot_names(layout, envelope, profile=profile)
  return [
    names[slot]
    for slot in range(len(names))
    if before[slot * element_size : (slot + 1) * element_size]
    != after[slot * element_size : (slot + 1) * element_size]
  ]


def classify_physical_difference(expected, observed, *, control=None):
  """Compare logical outputs and classify a difference by its origin.

  A control variant is an intentional mutation the study applies to itself, so
  a difference it produces is expected and is never a compiler discovery.
  """
  if not isinstance(expected, bytes) or not isinstance(observed, bytes):
    raise LayoutError("logical comparison needs byte buffers")
  if len(expected) != len(observed):
    raise LayoutError("logical comparison needs equal-length buffers")
  lanes = [
    index
    for index in range(len(expected) // 4)
    if expected[index * 4 : (index + 1) * 4] != observed[index * 4 : (index + 1) * 4]
  ]
  if not lanes:
    return {
      "status": "equal",
      "classification": "equal",
      "control": control,
      "logical_mismatch_lanes": [],
    }
  return {
    "status": "different",
    "classification": "control_variant" if control else "unexpected",
    "control": control,
    "logical_mismatch_lanes": lanes,
  }
