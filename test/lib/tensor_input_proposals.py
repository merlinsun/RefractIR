"""Deterministic input proposals for one admitted generated loop region.

A behavior experiment needs more than the solver-selected witness a counted case
carries. This module derives four recorded proposals from the case itself, so the
same case always yields the same bytes:

* `independent-small` draws every lane independently from a small value range;
* `asymmetric-coordinates` makes the value depend on both coordinates with
  distinct per-axis weights, so transposing the tensor changes the input;
* `sparse-distinct-groups` marks one lane per reduction group with a group-
  specific value, so different groups carry different partial sums;
* `sparse-final-lanes` marks the final valid lane of every reduction group and
  the final element of the tensor, so the highest addresses carry data.

Every value is a nonnegative integer leaf inside the recorded domain, with
positive zero, which is what the exact f32 reference admits. A proposal is
therefore either certified over the whole output or refused with a reason; the
derivation never widens the domain to admit a pattern.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from test.lib import tensor_exact_f32

DERIVATION_VERSION = "tensor-input-proposal-v1"
PROPOSAL_NAMES = (
  "independent-small",
  "asymmetric-coordinates",
  "sparse-distinct-groups",
  "sparse-final-lanes",
)
SMALL_VALUE_LIMIT = 3
ASYMMETRIC_WEIGHTS = (3, 5)
ASYMMETRIC_OFFSET = 1
GROUP_VALUE_CYCLE = (1, 2, 3)
DEFAULT_GROUP_AXIS = 0


@dataclass(frozen=True)
class InputProposal:
  """One recorded proposal: its name, its derivation record, and its leaves."""

  name: str
  spec: dict
  values: tuple[int, ...]

  @property
  def data(self) -> bytes:
    """The proposal's little-endian f32 bytes, one word per lane."""
    return struct.pack(f"<{len(self.values)}f", *self.values)

  @property
  def nonzero(self) -> tuple[int, ...]:
    return tuple(index for index, value in enumerate(self.values) if value)

  def record(self) -> dict:
    """A machine-readable record of the proposal and the bytes it states."""
    words = struct.unpack(f"<{len(self.values)}I", self.data)
    return {
      "name": self.name,
      "derivation": DERIVATION_VERSION,
      "spec": dict(self.spec),
      "values": list(self.values),
      "byte_length": len(self.data),
      "sha256": hashlib.sha256(self.data).hexdigest(),
      "word_zero_count": sum(1 for word in words if word == 0),
      "nonzero_lanes": list(self.nonzero),
      "distinct_values": sorted(set(self.values)),
      "in_domain": True,
      "integer_valued": True,
      "positive_zero": True,
    }


def pack(proposal: InputProposal, *, domain: tuple[int, int]) -> bytes:
  """The proposal's bytes, refused unless every leaf is an admitted integer."""
  low, high = _domain(domain)
  for index, value in enumerate(proposal.values):
    if type(value) is not int or not low <= value <= high:
      raise ValueError(
        f"proposal {proposal.name} lane {index} carries {value!r}, outside [{low},{high}]"
      )
  return proposal.data


def _domain(domain):
  if (
    not isinstance(domain, (tuple, list))
    or len(domain) != 2
    or any(type(value) is not int for value in domain)
    or domain[0] < 0
    or domain[0] > domain[1]
  ):
    raise ValueError("proposal domain must be a nonnegative integer interval")
  return domain[0], domain[1]


def _case(source, domain, seed):
  """The case facts every proposal is derived from, with the exact admission."""
  shape, iterations, _types, nodes, _updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  if type(seed) is not int or seed < 0:
    raise ValueError("proposal seed must be a nonnegative integer")
  low, high = _domain(domain)
  axes = tuple(node[3] for node in nodes if node[1] == "sum")
  return {
    "shape": shape,
    "iterations": iterations,
    "axes": axes,
    "group_axes": axes or (DEFAULT_GROUP_AXIS,),
    "group_axis_source": "reduction-axis" if axes else "default-axis-0",
    "low": low,
    "high": high,
    "seed": seed,
    "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
  }


def _base_spec(case, kind, rule):
  return {
    "kind": kind,
    "derivation": DERIVATION_VERSION,
    "rule": rule,
    "source_sha256": case["source_sha256"],
    "seed": case["seed"],
    "domain": [case["low"], case["high"]],
    "shape": list(case["shape"]),
    "iterations": case["iterations"],
    "reduction_axes": list(case["axes"]),
    "group_axes": list(case["group_axes"]),
    "group_axis_source": case["group_axis_source"],
  }


def _coordinates(index, shape):
  coordinate = [0] * len(shape)
  for axis in range(len(shape) - 1, -1, -1):
    coordinate[axis] = index % shape[axis]
    index //= shape[axis]
  return coordinate


def _flat_index(coordinate, shape):
  index = 0
  for axis, value in enumerate(coordinate):
    index = index * shape[axis] + value
  return index


def _groups(shape, axis):
  """The lanes of every reduction group along `axis`, first lane order kept.

  A reduction along `axis` sums the lanes that share their other coordinates, so
  a group is named by the kept coordinates and its lanes differ only on `axis`.
  """
  kept_extents = [extent for index, extent in enumerate(shape) if index != axis]
  count = 1
  for extent in kept_extents:
    count *= extent
  groups = []
  for group in range(count):
    kept = _coordinates(group, kept_extents)
    lanes = []
    for position in range(shape[axis]):
      coordinate = []
      kept_position = 0
      for index in range(len(shape)):
        if index == axis:
          coordinate.append(position)
        else:
          coordinate.append(kept[kept_position])
          kept_position += 1
      lanes.append(_flat_index(coordinate, shape))
    groups.append(tuple(lanes))
  return groups


def _independent_small_values(case):
  count = case["shape"][0] * case["shape"][1]
  limit = min(SMALL_VALUE_LIMIT, case["high"])
  prefix = (
    f"{DERIVATION_VERSION}:{case['source_sha256']}:{case['seed']}:independent-small"
  )
  values = tuple(
    int.from_bytes(hashlib.sha256(f"{prefix}:{index}".encode()).digest()[:8], "big")
    % (limit + 1)
    for index in range(count)
  )
  spec = _base_spec(
    case,
    "independent-small-values",
    "per-lane sha256(derivation:source:seed:name:lane)[0:8] mod (min(3,hi)+1)",
  )
  spec["value_range"] = [0, limit]
  return InputProposal("independent-small", spec, values)


def _asymmetric_coordinates(case):
  rows, columns = case["shape"]
  modulus = case["high"] + 1
  row_weight, column_weight = ASYMMETRIC_WEIGHTS
  values = tuple(
    (row_weight * row + column_weight * column + ASYMMETRIC_OFFSET) % modulus
    for row in range(rows)
    for column in range(columns)
  )
  spec = _base_spec(
    case,
    "asymmetric-coordinate-pattern",
    f"({{row_weight}}*row + {{column_weight}}*column + {ASYMMETRIC_OFFSET}) mod (hi+1)",
  )
  spec["axis_weights"] = [row_weight, column_weight]
  spec["modulus"] = modulus
  return InputProposal("asymmetric-coordinates", spec, values)


def _sparse_distinct_groups(case):
  shape = case["shape"]
  axis = case["group_axes"][0]
  values = [0] * (shape[0] * shape[1])
  groups = _groups(shape, axis)
  for group, lanes in enumerate(groups):
    value = GROUP_VALUE_CYCLE[group % len(GROUP_VALUE_CYCLE)]
    values[lanes[0]] = min(value, case["high"])
  spec = _base_spec(
    case,
    "sparse-distinct-groups",
    "the first lane of each group on the first group axis cycles 1,2,3",
  )
  spec["group_count"] = len(groups)
  spec["group_lane"] = "first"
  return InputProposal("sparse-distinct-groups", spec, tuple(values))


def _sparse_final_lanes(case):
  shape = case["shape"]
  values = [0] * (shape[0] * shape[1])
  lanes = set()
  for axis in case["group_axes"]:
    for group_lanes in _groups(shape, axis):
      lanes.add(group_lanes[-1])
  for lane in lanes:
    values[lane] = min(1, case["high"])
  final = shape[0] * shape[1] - 1
  values[final] = case["high"]
  spec = _base_spec(
    case,
    "sparse-final-lanes",
    "the final lane of each group on every group axis holds 1; the final element "
    "holds the domain upper bound",
  )
  spec["group_lane"] = "final"
  spec["final_element_lane"] = final
  spec["final_element_value"] = case["high"]
  return InputProposal("sparse-final-lanes", spec, tuple(values))


def derive_proposals(source, *, domain=(0, 8), seed=0):
  """Derive the four recorded proposals for one admitted region source."""
  case = _case(source, domain, seed)
  proposals = (
    _independent_small_values(case),
    _asymmetric_coordinates(case),
    _sparse_distinct_groups(case),
    _sparse_final_lanes(case),
  )
  for proposal in proposals:
    pack(proposal, domain=domain)
  return proposals


def duplicate_groups(proposals):
  """Names of proposals that share identical bytes, grouped instead of dropped."""
  groups = {}
  for proposal in proposals:
    groups.setdefault(proposal.data, []).append(proposal.name)
  return [names for names in groups.values() if len(names) > 1]


def constant_output(data):
  """State whether every f32 word of a reference output is the same word."""
  if not isinstance(data, bytes) or not data or len(data) % 4:
    raise ValueError("an output must be a nonempty whole number of f32 words")
  words = struct.unpack(f"<{len(data) // 4}I", data)
  if len(set(words)) == 1:
    return {
      "constant": True,
      "word": f"0x{words[0]:08x}",
      "value": struct.unpack("<f", struct.pack("<I", words[0]))[0],
    }
  return {"constant": False, "distinct_words": len(set(words))}
