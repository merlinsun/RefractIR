"""Small, validated Gluon layout descriptions for the exact tensor pilot."""

import random
from dataclasses import asdict, dataclass
from functools import reduce
from operator import mul


def _is_power_of_two(value):
  return type(value) is int and value > 0 and value & (value - 1) == 0


def _product(values):
  return reduce(mul, values, 1)


@dataclass(frozen=True)
class BlockedLayoutPlan:
  """A rank-two BlockedLayout with launch-shape consistency checks."""

  name: str
  size_per_thread: tuple[int, int]
  threads_per_warp: tuple[int, int]
  warps_per_cta: tuple[int, int]
  order: tuple[int, int]

  def validate(self, shape, *, num_warps):
    if not isinstance(self.name, str) or not self.name:
      raise ValueError("layout name must be nonempty")
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
      raise ValueError("shape must have rank two")
    if any(not _is_power_of_two(dimension) for dimension in shape):
      raise ValueError("shape dimensions must be positive powers of two")
    for field in (self.size_per_thread, self.threads_per_warp, self.warps_per_cta):
      if (
        not isinstance(field, tuple)
        or len(field) != 2
        or any(not _is_power_of_two(value) for value in field)
      ):
        raise ValueError("layout tile dimensions must be rank-two powers of two")
    if _product(self.threads_per_warp) != 32:
      raise ValueError("threads_per_warp must allocate exactly 32 lanes")
    if not _is_power_of_two(num_warps) or _product(self.warps_per_cta) != num_warps:
      raise ValueError("warps_per_cta must match the launch num_warps")
    if len(self.order) != 2 or set(self.order) != {0, 1}:
      raise ValueError("order must be a permutation of tensor dimensions")

  def to_dict(self):
    return asdict(self)

  def to_gluon_layout(self):
    """Construct the corresponding Triton layout only when Triton is present."""
    from triton.experimental.gluon import language as gl

    return gl.BlockedLayout(
      list(self.size_per_thread),
      list(self.threads_per_warp),
      list(self.warps_per_cta),
      list(self.order),
    )


def _powers_of_two_at_most(limit):
  values = []
  value = 1
  while value <= limit:
    values.append(value)
    value *= 2
  return values


def _factor_pairs(product):
  return [
    (first, product // first)
    for first in _powers_of_two_at_most(product)
    if product % first == 0
  ]


def generate_layout_plans(shape, *, num_warps, seed, count):
  """Sample distinct rank-two blocked layouts from a bounded legal space."""
  if (
    not isinstance(shape, (tuple, list))
    or len(shape) != 2
    or any(not _is_power_of_two(size) for size in shape)
  ):
    raise ValueError("shape must contain two positive powers of two")
  if not _is_power_of_two(num_warps):
    raise ValueError("num_warps must be a positive power of two")
  if type(seed) is not int or seed < 0:
    raise ValueError("seed must be a nonnegative integer")
  if type(count) is not int or count <= 0:
    raise ValueError("count must be a positive integer")

  candidates = []
  thread_pairs = _factor_pairs(32)
  warp_pairs = _factor_pairs(num_warps)
  for size_per_thread in (
    (row_size, column_size)
    for row_size in _powers_of_two_at_most(shape[0])
    for column_size in _powers_of_two_at_most(shape[1])
  ):
    for threads_per_warp in thread_pairs:
      for warps_per_cta in warp_pairs:
        for order in ((0, 1), (1, 0)):
          name = f"blocked-{len(candidates):04d}"
          plan = BlockedLayoutPlan(
            name,
            size_per_thread,
            threads_per_warp,
            warps_per_cta,
            order,
          )
          plan.validate(shape, num_warps=num_warps)
          candidates.append(plan)
  if count > len(candidates):
    raise ValueError(
      f"requested {count} layouts, but only {len(candidates)} available legal layouts"
    )

  random.Random(seed).shuffle(candidates)
  return candidates[:count]
