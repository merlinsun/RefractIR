"""Deterministic i32 input proposals for one explicit logical tensor case.

`test.lib.tensor_input_proposals` reads its case facts from the generated f32
region source. An i32 tail case cannot use that parser: the dtype is i32, a
logical extent may be a tail such as 31, and the returned tensor may be scalar.
This adapter takes the shape, dtype, reduction-group axes, domain, and seed from
the caller and builds the same four patterns over little-endian i32 words:

* `independent-small` draws every lane independently from a small value range;
* `asymmetric-coordinates` makes the value depend on every coordinate with a
  distinct per-axis weight;
* `sparse-distinct-groups` marks the first lane of every reduction group with a
  group-specific value;
* `sparse-final-lanes` marks the final lane of every reduction group and the
  final element of the tensor.

The lane geometry is imported from the f32 module, so both dtypes mark the same
coordinates; only the case facts and the word packing differ. The small-value
draw uses the explicit case digest where the f32 module uses the source digest,
because an i32 case has no generated f32 source text to hash.

`tensor_case.load_case` validates tensor-v3 `coverage.input_behavior.pattern`
against a closed set, and no proposal name is in that set. A proposal record
therefore keeps the proposal name in `name` and `v3_input_behavior.proposal`,
which the loader ignores, and states the accepted label a producer may write in
`v3_input_behavior.pattern`. The mapping beside `V3_INPUT_PATTERNS` records
which accepted label each pattern uses and why.

The reference replay is delegated to `tensor_reference.replay`, which reads the
profile and entry from the case itself. `replay_reference` exists so a caller
holding an i32 tail case does not state a profile version of its own.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass

from test.lib import tensor_case, tensor_reference
from test.lib.tensor_input_proposals import (
  ASYMMETRIC_OFFSET,
  ASYMMETRIC_WEIGHTS,
  DEFAULT_GROUP_AXIS,
  GROUP_VALUE_CYCLE,
  PROPOSAL_NAMES,
  SMALL_VALUE_LIMIT,
  _coordinates,
  _groups,
)

DTYPE = "i32"
DERIVATION_VERSION = "tensor-input-proposal-i32-v1"
I32_MAX = 2**31 - 1

# The accepted tensor-v3 label each proposal maps to. The proposal name stays
# out of this vocabulary: `tensor_case.load_case` validates the manifest's
# `coverage.input_behavior.pattern` against its closed set, and the record keeps
# the proposal name in the loader-ignored `proposal` field instead.
V3_INPUT_PATTERNS = {
  # A seeded per-lane draw from a bounded value range.
  "independent-small": "seeded-random",
  # The value is a weighted function of every coordinate.
  "asymmetric-coordinates": "coordinate-distinct",
  # One lane per reduction group carries a group-specific value.
  "sparse-distinct-groups": "distinct-groups",
  # The final lane of every group carries data and the final element carries
  # the domain upper bound, so the pattern places values at the boundary.
  "sparse-final-lanes": "boundary-value",
}

__all__ = [
  "DERIVATION_VERSION",
  "DTYPE",
  "I32InputProposal",
  "I32ProposalCase",
  "PROPOSAL_NAMES",
  "V3_INPUT_PATTERNS",
  "derive_proposals",
  "pack",
  "replay_reference",
]


@dataclass(frozen=True)
class I32ProposalCase:
  """One explicit i32 case; no fact is parsed from SIR text."""

  shape: tuple[int, ...]
  reduction_axes: tuple[int, ...] = ()
  dtype: str = DTYPE
  domain: tuple[int, int] = (0, 8)
  seed: int = 0

  def __post_init__(self):
    object.__setattr__(self, "shape", _shape(self.shape))
    object.__setattr__(
      self, "reduction_axes", _reduction_axes(self.reduction_axes, self.shape)
    )
    if self.dtype != DTYPE:
      raise ValueError(f"this adapter states i32 cases, not {self.dtype!r}")
    object.__setattr__(self, "domain", _domain(self.domain))
    if type(self.seed) is not int or self.seed < 0:
      raise ValueError("proposal seed must be a nonnegative integer")

  @property
  def count(self) -> int:
    """The number of logical lanes."""
    return math.prod(self.shape)

  @property
  def group_axes(self) -> tuple[int, ...]:
    """The axes whose lanes form reduction groups, defaulting to axis zero."""
    return self.reduction_axes or (DEFAULT_GROUP_AXIS,)

  @property
  def group_axis_source(self) -> str:
    return "reduction-axis" if self.reduction_axes else "default-axis-0"

  @property
  def identity(self) -> str:
    """Digest of every explicit fact that changes the proposal bytes."""
    facts = {
      "dtype": self.dtype,
      "shape": list(self.shape),
      "reduction_axes": list(self.reduction_axes),
      "domain": list(self.domain),
      "seed": self.seed,
    }
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class I32InputProposal:
  """One recorded proposal: its name, its derivation record, and its leaves."""

  name: str
  spec: dict
  values: tuple[int, ...]

  @property
  def data(self) -> bytes:
    """The proposal's little-endian i32 bytes, one word per lane."""
    return struct.pack(f"<{len(self.values)}i", *self.values)

  @property
  def v3_pattern(self) -> str:
    """The accepted tensor-v3 pattern label for this proposal."""
    try:
      return V3_INPUT_PATTERNS[self.name]
    except KeyError as error:
      raise ValueError(
        f"proposal {self.name!r} has no tensor-v3 pattern label"
      ) from error

  @property
  def nonzero(self) -> tuple[int, ...]:
    return tuple(index for index, value in enumerate(self.values) if value)

  def record(self) -> dict:
    """A machine-readable record of the proposal, its bytes, and its labels."""
    words = struct.unpack(f"<{len(self.values)}i", self.data)
    low, high = self.spec.get("domain", [0, 0])
    return {
      "name": self.name,
      "derivation": DERIVATION_VERSION,
      "dtype": DTYPE,
      "spec": dict(self.spec),
      "values": list(self.values),
      "byte_length": len(self.data),
      "sha256": hashlib.sha256(self.data).hexdigest(),
      "word_zero_count": sum(1 for word in words if word == 0),
      "nonzero_lanes": list(self.nonzero),
      "distinct_values": sorted(set(self.values)),
      "in_domain": True,
      "integer_valued": True,
      "v3_input_behavior": {
        "pattern": self.v3_pattern,
        "proposal": self.name,
        "input_mode": "deterministic-input-proposal",
        "input_domain": {"lo": low, "hi": high},
      },
    }


def pack(proposal, *, domain):
  """The proposal's bytes, refused unless every leaf is an admitted integer."""
  low, high = _domain(domain)
  for index, value in enumerate(proposal.values):
    if type(value) is not int or not low <= value <= high:
      raise ValueError(
        f"proposal {proposal.name} lane {index} carries {value!r}, "
        f"outside [{low},{high}]"
      )
  return proposal.data


def derive_proposals(case):
  """Derive the four recorded i32 proposals for one explicit case."""
  if not isinstance(case, I32ProposalCase):
    raise TypeError("derive_proposals requires an I32ProposalCase")
  proposals = (
    _independent_small_values(case),
    _asymmetric_coordinates(case),
    _sparse_distinct_groups(case),
    _sparse_final_lanes(case),
  )
  for proposal in proposals:
    pack(proposal, domain=case.domain)
  return proposals


def replay_reference(
  case,
  *,
  directory,
  symiri=tensor_case.DEFAULT_SYMIRI,
  timeout=tensor_reference.DEFAULT_TIMEOUT_S,
):
  """Replay the case's reference from its own profile and entry."""
  return tensor_reference.replay(
    case, directory=directory, symiri=symiri, timeout=timeout
  )


def _shape(shape):
  if not isinstance(shape, (tuple, list)) or len(shape) not in {1, 2}:
    raise ValueError("i32 proposal shape must hold one or two logical extents")
  extents = tuple(shape)
  for axis, extent in enumerate(extents):
    if type(extent) is not int or extent < 1:
      raise ValueError(f"i32 proposal extent {axis} must be an integer >= 1")
  return extents


def _reduction_axes(axes, shape):
  if not isinstance(axes, (tuple, list)):
    raise ValueError("i32 proposal reduction axes must be a tuple")
  seen = set()
  for axis in axes:
    if type(axis) is not int or not 0 <= axis < len(shape):
      raise ValueError(f"i32 proposal reduction axis {axis!r} is outside the rank")
    if axis in seen:
      raise ValueError("i32 proposal reduction axes must not repeat")
    seen.add(axis)
  return tuple(sorted(seen))


def _domain(domain):
  if (
    not isinstance(domain, (tuple, list))
    or len(domain) != 2
    or any(type(value) is not int for value in domain)
    or domain[0] < 0
    or domain[0] > domain[1]
    or domain[1] > I32_MAX
  ):
    raise ValueError("proposal domain must be a nonnegative i32 interval")
  return domain[0], domain[1]


def _base_spec(case, kind, rule):
  return {
    "kind": kind,
    "derivation": DERIVATION_VERSION,
    "rule": rule,
    "case_sha256": case.identity,
    "dtype": case.dtype,
    "seed": case.seed,
    "domain": [case.domain[0], case.domain[1]],
    "shape": list(case.shape),
    "reduction_axes": list(case.reduction_axes),
    "group_axes": list(case.group_axes),
    "group_axis_source": case.group_axis_source,
  }


def _independent_small_values(case):
  limit = min(SMALL_VALUE_LIMIT, case.domain[1])
  # No source text exists to seed the f32 module's stream, and the case digest
  # already covers every fact that may change the values.
  prefix = f"{DERIVATION_VERSION}:{case.identity}:independent-small"
  values = tuple(
    int.from_bytes(hashlib.sha256(f"{prefix}:{index}".encode()).digest()[:8], "big")
    % (limit + 1)
    for index in range(case.count)
  )
  spec = _base_spec(
    case,
    "independent-small-values",
    "per-lane sha256(derivation:case:name:lane)[0:8] mod (min(3,hi)+1)",
  )
  spec["value_range"] = [0, limit]
  return I32InputProposal("independent-small", spec, values)


def _asymmetric_coordinates(case):
  modulus = case.domain[1] + 1
  weights = ASYMMETRIC_WEIGHTS[: len(case.shape)]
  values = tuple(
    (
      sum(
        weight * coordinate
        for weight, coordinate in zip(weights, _coordinates(index, case.shape))
      )
      + ASYMMETRIC_OFFSET
    )
    % modulus
    for index in range(case.count)
  )
  terms = " + ".join(f"{weight}*c{axis}" for axis, weight in enumerate(weights))
  spec = _base_spec(
    case,
    "asymmetric-coordinate-pattern",
    f"({terms} + {ASYMMETRIC_OFFSET}) mod (hi+1)",
  )
  spec["axis_weights"] = list(weights)
  spec["modulus"] = modulus
  return I32InputProposal("asymmetric-coordinates", spec, values)


def _sparse_distinct_groups(case):
  values = [0] * case.count
  groups = _groups(case.shape, case.group_axes[0])
  for group, lanes in enumerate(groups):
    value = GROUP_VALUE_CYCLE[group % len(GROUP_VALUE_CYCLE)]
    values[lanes[0]] = min(value, case.domain[1])
  spec = _base_spec(
    case,
    "sparse-distinct-groups",
    "the first lane of each group on the first group axis cycles 1,2,3",
  )
  spec["group_count"] = len(groups)
  spec["group_lane"] = "first"
  return I32InputProposal("sparse-distinct-groups", spec, tuple(values))


def _sparse_final_lanes(case):
  values = [0] * case.count
  lanes = set()
  for axis in case.group_axes:
    for group_lanes in _groups(case.shape, axis):
      lanes.add(group_lanes[-1])
  for lane in lanes:
    values[lane] = min(1, case.domain[1])
  final = case.count - 1
  values[final] = case.domain[1]
  spec = _base_spec(
    case,
    "sparse-final-lanes",
    "the final lane of each group on every group axis holds 1; the final "
    "element holds the domain upper bound",
  )
  spec["group_lane"] = "final"
  spec["final_element_lane"] = final
  spec["final_element_value"] = case.domain[1]
  return I32InputProposal("sparse-final-lanes", spec, tuple(values))
