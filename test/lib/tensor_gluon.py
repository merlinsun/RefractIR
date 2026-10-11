"""Small, validated Gluon layout descriptions for the exact tensor pilot."""

import random
from dataclasses import asdict, dataclass, field
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
    for field_value in (
      self.size_per_thread,
      self.threads_per_warp,
      self.warps_per_cta,
    ):
      if (
        not isinstance(field_value, tuple)
        or len(field_value) != 2
        or any(not _is_power_of_two(value) for value in field_value)
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


def select_distinct_layouts(required, sampled, *, shape, num_warps, count):
  """Keep a required plan and fill the sample by physical layout identity."""
  if type(count) is not int or count <= 0:
    raise ValueError("count must be a positive integer")
  required.validate(shape, num_warps=num_warps)
  selected = [required]
  seen = {
    (
      required.size_per_thread,
      required.threads_per_warp,
      required.warps_per_cta,
      required.order,
    )
  }
  if count == 1:
    return selected

  for plan in sampled:
    plan.validate(shape, num_warps=num_warps)
    key = (
      plan.size_per_thread,
      plan.threads_per_warp,
      plan.warps_per_cta,
      plan.order,
    )
    if key in seen:
      continue
    seen.add(key)
    selected.append(plan)
    if len(selected) == count:
      return selected

  raise ValueError(
    f"requested {count} distinct layouts, but the required layout and sample "
    f"contain only {len(selected)}"
  )


def _bits(value):
  """Bit width one distribution factor contributes to a lane identifier."""
  return 0 if value <= 1 else value.bit_length() - 1


def _blocked_owner(plan, shape, coordinate):
  """Return (warp, lane, register) owning one element under a blocked layout.

  Each dimension is covered by a tile of sizePerThread x threadsPerWarp x
  warpsPerCTA; a larger extent repeats that tile, and the repetition lands in
  the thread's registers. Bits are concatenated across dimensions in `order`,
  which lists axes from fastest to slowest.
  """
  warp = lane = register = 0
  lane_stride = warp_stride = register_stride = 1
  for axis in plan.order:
    extent = shape[axis]
    per_thread = min(plan.size_per_thread[axis], extent)
    per_warp = min(plan.threads_per_warp[axis], max(1, extent // per_thread))
    per_cta = min(plan.warps_per_cta[axis], max(1, extent // (per_thread * per_warp)))
    coverage = per_thread * per_warp * per_cta
    within = coordinate[axis] % coverage
    tiles = coordinate[axis] // coverage
    lane += ((within // per_thread) % per_warp) * lane_stride
    warp += ((within // (per_thread * per_warp)) % per_cta) * warp_stride
    register += (tiles * per_thread + within % per_thread) * register_stride
    lane_stride *= max(per_warp, 1)
    warp_stride *= max(per_cta, 1)
    register_stride *= -(-extent // coverage) * per_thread
  return warp, lane, register


def effective_mapping(plan, shape):
  """Return the (warp, lane, register) owner of every element, row-major."""
  if not isinstance(shape, (tuple, list)) or len(shape) != 2:
    raise ValueError("effective mapping needs a rank-two shape")
  return tuple(
    _blocked_owner(plan, tuple(shape), (row, column))
    for row in range(shape[0])
    for column in range(shape[1])
  )


def _layout_identity(plan):
  return (
    plan.size_per_thread,
    plan.threads_per_warp,
    plan.warps_per_cta,
    plan.order,
  )


def classify_layout_pair(left, right, *, shape):
  """Name what a layout change between two plans actually moves."""
  if not isinstance(left, BlockedLayoutPlan) or not isinstance(
    right, BlockedLayoutPlan
  ):
    raise ValueError("layout classification needs two blocked layout plans")
  if _layout_identity(left) == _layout_identity(right):
    return "identical"
  left_mapping = effective_mapping(left, shape)
  right_mapping = effective_mapping(right, shape)
  if left_mapping == right_mapping:
    return "equivalent"
  if any(a[0] != b[0] for a, b in zip(left_mapping, right_mapping)):
    return "cross_warp"
  if any(a[1] != b[1] for a, b in zip(left_mapping, right_mapping)):
    return "cross_lane"
  return "register_only"


@dataclass(frozen=True)
class GluonRegionPlan:
  """Layout roles and per-consumer views for one generated region.

  A consumer view states the layout one use of a value needs. The emitter
  converts into a fresh view instead of rebinding the value, so a second
  consumer keeps the layout its own entry states.
  """

  name: str
  shape: tuple[int, int]
  load: BlockedLayoutPlan
  compute: BlockedLayoutPlan
  store: BlockedLayoutPlan
  reduction: BlockedLayoutPlan | None = None
  state_layout: BlockedLayoutPlan | None = None
  consumer_views: dict = field(default_factory=dict)

  def view_for(self, value, node_index):
    """The layout of one consumer's view, or the compute layout by default."""
    return self.consumer_views.get((value, node_index), self.compute)

  def as_record(self):
    return {
      "name": self.name,
      "shape": list(self.shape),
      "load": self.load.name,
      "compute": self.compute.name,
      "reduction": self.reduction.name if self.reduction else None,
      "store": self.store.name,
      "state_layout": self.state_layout.name if self.state_layout else None,
      "consumer_views": {
        f"{value}#{index}": layout.name
        for (value, index), layout in sorted(self.consumer_views.items())
      },
    }


# Distinct sampled layouts each family places on its roles. A pool that holds
# fewer layouts than a family has roles would put one layout on two roles, so
# the family would silently stop stating the difference its name describes.
PLAN_FAMILY_MINIMUM_LAYOUTS = {
  "all-same-layout": 1,
  "load-to-compute": 2,
  "shared-value-reused": 2,
  "split-reduction-consumers": 3,
  "reduction-result-expand": 3,
  "loop-state-layout": 4,
}


def region_plan_families(shape, *, num_warps, layouts):
  """Build the layout plan families one layout pool can faithfully express.

  `layouts` holds distinct sampled blocked layouts; the families place them on
  the load, compute, reduction, per-consumer, loop-state, and store roles. A
  family that needs more distinct layouts than the pool holds is left out
  rather than collapsing two of its roles onto one layout.
  """
  if not isinstance(layouts, (list, tuple)) or not layouts:
    raise ValueError("plan families need at least one distinct layout")
  for layout in layouts:
    layout.validate(shape, num_warps=num_warps)
  baseline = layouts[0]
  second = layouts[1 % len(layouts)]
  third = layouts[2 % len(layouts)]
  fourth = layouts[3 % len(layouts)]

  def region(name, load, compute, store, *, reduction=None, state=None, views=None):
    return GluonRegionPlan(
      name,
      tuple(shape),
      load,
      compute,
      store,
      reduction=reduction,
      state_layout=state,
      consumer_views=dict(views or {}),
    )

  families = {
    "all-same-layout": region(
      "all-same-layout", baseline, baseline, baseline, reduction=baseline
    ),
    "load-to-compute": region(
      "load-to-compute", second, baseline, baseline, reduction=baseline
    ),
    "shared-value-reused": region(
      "shared-value-reused",
      baseline,
      baseline,
      second,
      reduction=baseline,
      views={("acc_x", 0): second, ("acc_x", 2): second},
    ),
    "split-reduction-consumers": region(
      "split-reduction-consumers",
      baseline,
      baseline,
      second,
      reduction=second,
      views={("acc_x", 0): second, ("acc_x", 2): third},
    ),
    "reduction-result-expand": region(
      "reduction-result-expand",
      baseline,
      third,
      second,
      reduction=second,
      # The first broadcast reads the first reduction's result (`v0`) and the
      # second reads the second reduction's (`v2`), so each gets a view that
      # differs from the compute layout the emitter would otherwise inherit.
      views={("v0", 1): fourth, ("v2", 3): fourth},
    ),
    "loop-state-layout": region(
      "loop-state-layout",
      baseline,
      baseline,
      fourth,
      reduction=second,
      state=third,
    ),
  }
  return {
    name: family
    for name, family in families.items()
    if len(layouts) >= PLAN_FAMILY_MINIMUM_LAYOUTS[name]
  }
