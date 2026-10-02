"""Small, validated Gluon layout descriptions for the exact tensor pilot."""

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
