"""Small, validated Gluon layout descriptions for the exact tensor pilot."""

import random
import re
from dataclasses import asdict, dataclass, field
from functools import reduce
from operator import mul

from test.lib import tensor_layout
from test.lib.tensor_exact_f32 import _generated_region_program


def _is_power_of_two(value):
  return type(value) is int and value > 0 and value & (value - 1) == 0


def _product(values):
  return reduce(mul, values, 1)


def padded_execution_tile(shape):
  """The execution tile the tensor-v3 profile pads one logical shape up to.

  The profile rounds every extent up to a power of two and keeps the padded
  volume inside its 512-element cap, so a logical tail never becomes a physical
  extent: the tile states the padded extents the hardware places values on and
  the mask states which of its lanes carry logical elements.
  """
  if not isinstance(shape, (tuple, list)) or len(shape) != 2:
    raise ValueError("padded execution tile needs a rank-two logical shape")
  return tuple(tensor_layout.execution_tile_shape(list(shape), profile="v3"))


def execution_tile(shape, *, tile=None):
  """The power-of-two execution tile one logical rank-two shape runs on.

  A layout plan is validated against this tile rather than against the logical
  extent, because the hardware places the padded tile. `tile` states the tile a
  caller already chose; without one the tile is the logical shape itself, which
  exists only when every extent is already a power of two, so a tailed shape
  without a declared tile is refused instead of silently rounded up.
  """
  if not isinstance(shape, (tuple, list)) or len(shape) != 2:
    raise ValueError("execution tile needs a rank-two logical shape")
  if tile is None:
    if any(not _is_power_of_two(extent) for extent in shape):
      raise ValueError("shape dimensions must be positive powers of two")
    return tuple(shape)
  if (
    not isinstance(tile, (tuple, list))
    or len(tile) != 2
    or any(not _is_power_of_two(extent) for extent in tile)
  ):
    raise ValueError("execution tile must be two positive powers of two")
  if any(extent > padded for extent, padded in zip(shape, tile)):
    raise ValueError("execution tile must cover the logical shape")
  return tuple(tile)


@dataclass(frozen=True)
class BlockedLayoutPlan:
  """A rank-two BlockedLayout with launch-shape consistency checks."""

  name: str
  size_per_thread: tuple[int, int]
  threads_per_warp: tuple[int, int]
  warps_per_cta: tuple[int, int]
  order: tuple[int, int]

  def validate(self, shape, *, num_warps, tile=None):
    if not isinstance(self.name, str) or not self.name:
      raise ValueError("layout name must be nonempty")
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
      raise ValueError("shape must have rank two")
    execution_tile(shape, tile=tile)
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


def generate_layout_plans(shape, *, num_warps, seed, count, tile=None):
  """Sample distinct rank-two blocked layouts from a bounded legal space.

  The sampled register shape is bounded by the execution tile, so a tile wider
  than the logical extent still offers the layouts that place its padding.
  """
  if not isinstance(shape, (tuple, list)) or len(shape) != 2:
    raise ValueError("shape must contain two positive powers of two")
  tile_extents = execution_tile(shape, tile=tile)
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
    for row_size in _powers_of_two_at_most(tile_extents[0])
    for column_size in _powers_of_two_at_most(tile_extents[1])
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
          plan.validate(shape, num_warps=num_warps, tile=tile)
          candidates.append(plan)
  if count > len(candidates):
    raise ValueError(
      f"requested {count} layouts, but only {len(candidates)} available legal layouts"
    )

  random.Random(seed).shuffle(candidates)
  return candidates[:count]


def select_distinct_layouts(required, sampled, *, shape, num_warps, count, tile=None):
  """Keep a required plan and fill the sample by physical layout identity."""
  if type(count) is not int or count <= 0:
    raise ValueError("count must be a positive integer")
  required.validate(shape, num_warps=num_warps, tile=tile)
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
    plan.validate(shape, num_warps=num_warps, tile=tile)

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
  """Bit width one declared distribution factor contributes."""
  return 0 if value <= 1 else value.bit_length() - 1


def _blocked_bases(plan, shape):
  """Return the (axis, coordinate value) bases of each hardware dimension.

  Triton reads a blocked layout as `register(sizePerThread) *
  lane(threadsPerWarp) * warp(warpsPerCTA)`, where `inner * outer` keeps the
  inner bases at the low bits of every axis and shifts the outer bases up by
  the inner extent. The register bits are therefore the low bits of each axis,
  the lane bits follow them, and the warp bits follow those. A tile smaller
  than the tensor is repeated by register bits appended after all three
  factors, in the layout's own axis order.

  A basis that reaches past the tensor is dropped when it is a register bit and
  blanked when it is a lane or warp bit: the register then does not exist,
  while the lane or warp keeps its physical identity and holds a copy of the
  element another lane or warp already holds.
  """
  registers, lanes, warps = [], [], []
  for axis in plan.order:
    per_thread = plan.size_per_thread[axis]
    per_warp = per_thread * plan.threads_per_warp[axis]
    for bit in range(_bits(per_thread)):
      registers.append((axis, 1 << bit))
    for bit in range(_bits(plan.threads_per_warp[axis])):
      lanes.append((axis, per_thread << bit))
    for bit in range(_bits(plan.warps_per_cta[axis])):
      warps.append((axis, per_warp << bit))
  for axis in plan.order:
    tile = (
      plan.size_per_thread[axis]
      * plan.threads_per_warp[axis]
      * plan.warps_per_cta[axis]
    )
    for bit in range(_bits(shape[axis] // tile)):
      registers.append((axis, tile << bit))
  registers = [basis for basis in registers if basis[1] < shape[basis[0]]]
  lanes = [(axis, value if value < shape[axis] else 0) for axis, value in lanes]
  warps = [(axis, value if value < shape[axis] else 0) for axis, value in warps]
  return registers, lanes, warps


def _basis_coordinate(bases, index):
  """The logical coordinate one hardware dimension's index selects."""
  row = column = 0
  for bit, (axis, value) in enumerate(bases):
    if index >> bit & 1:
      if axis == 0:
        row += value
      else:
        column += value
  return row, column


def effective_mapping(plan, shape, *, tile=None):
  """Return the owner set of every tile element, row-major.

  An owner is a `(warp, lane, register)` triple. A tensor smaller than the
  layout tile gives one element several owners, because the unused lane and
  warp bits repeat it across the hardware instead of renumbering it. An
  element the tile does not place has no owner at all, which the empty set
  states instead of hiding.

  The mapped elements are the declared execution tile's when `tile` states
  one, so a tailed shape reports the owners of its padded lanes as well: those
  lanes are placed by the same hardware and only the mask says they carry no
  logical element. Without a tile the map covers the logical shape itself,
  which keeps an element no layout places visible as unowned.
  """
  if not isinstance(shape, (tuple, list)) or len(shape) != 2:
    raise ValueError("effective mapping needs a rank-two shape")
  shape = execution_tile(shape, tile=tile) if tile is not None else tuple(shape)
  registers, lanes, warps = _blocked_bases(plan, shape)

  owners = {}
  for register in range(1 << len(registers)):
    register_row, register_column = _basis_coordinate(registers, register)
    for lane in range(1 << len(lanes)):
      lane_row, lane_column = _basis_coordinate(lanes, lane)
      for warp in range(1 << len(warps)):
        warp_row, warp_column = _basis_coordinate(warps, warp)
        coordinate = (
          register_row + lane_row + warp_row,
          register_column + lane_column + warp_column,
        )
        owners.setdefault(coordinate, []).append((warp, lane, register))
  return tuple(
    tuple(sorted(owners.get((row, column), ())))
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


def _owner_field(owners, field_index):
  return {owner[field_index] for owner in owners}


def _holders(owners):
  """The `(warp, lane)` holders of an element, whose local register is free."""
  return {(warp, lane) for warp, lane, _register in owners}


# What one element requires of a layout change, weakest first. A pair needs
# whatever its most demanding element needs: a register renumbering outranks a
# dropped replicated copy, and a movement outranks both.
_REQUIREMENT_RANK = {
  "equivalent": 0,
  "subsumed": 1,
  "register_only": 2,
  "cross_lane": 3,
  "cross_warp": 4,
}


def _element_requirement(source_owners, destination_owners):
  """Name what the owners of one element require of a layout change.

  A destination holder that also holds the element in the source needs no
  transfer, because the register a value sits in is local to its holder. One
  that does not hold it names the movement: another warp has to send the
  value, or another lane of a warp the source already uses has to.
  """
  if destination_owners == source_owners:
    return "equivalent"
  source_holders = _holders(source_owners)
  destination_holders = _holders(destination_owners)
  missing = destination_holders - source_holders
  if not missing:
    # Every destination holder already holds the element: a destination that
    # keeps fewer holders drops replicated copies, and one that keeps them all
    # asks for the same holders in another local register.
    return "subsumed" if destination_holders < source_holders else "register_only"
  if any(warp not in _owner_field(source_owners, 0) for warp, _lane in missing):
    return "cross_warp"
  if any(lane not in _owner_field(source_owners, 1) for _warp, lane in missing):
    return "cross_lane"
  # Both the warp and the lane are owners already, so only the register differs.
  return "register_only"


def classify_layout_pair(left, right, *, shape, tile=None):
  """Name what a layout change between two plans requires of ownership.

  Owners are the `(warp, lane, register)` triples `effective_mapping` reports,
  and the verdict is the strongest requirement any element states. The mapped
  elements are the execution tile's, so a padded tail is classified by what
  moving its whole tile costs, padding lanes included.

  `identical` means the plan parameters match and `equivalent` means every
  element has the same owners in both plans. `subsumed` means every
  destination holder already holds the element it must own: replication is
  dropped and the register that holds a value serves the destination view, so
  no value has to cross the hardware. `register_only` keeps exactly the
  source's holders and renumbers the local register they use. A destination
  holder the source does not have at all has to receive the value, named
  `cross_warp` when the missing holder sits in another warp and `cross_lane`
  when it is another lane of a warp the source already uses.

  The verdict states what ownership requires, not what a conversion costs. A
  `subsumed` or `register_only` pair still needs lowering evidence before it
  can be called free, because this classifier models who holds a value, not
  the instructions a backend emits for it. An element no owner places is
  never reported as movement-free, because such a layout cannot be shown to
  need nothing.
  """
  if not isinstance(left, BlockedLayoutPlan) or not isinstance(
    right, BlockedLayoutPlan
  ):
    raise ValueError("layout classification needs two blocked layout plans")
  if _layout_identity(left) == _layout_identity(right):
    return "identical"
  source = effective_mapping(left, shape, tile=tile)
  destination = effective_mapping(right, shape, tile=tile)

  if any(not owners for owners in source) or any(not owners for owners in destination):
    return "cross_warp"
  return max(
    (
      _element_requirement(source_owners, destination_owners)
      for source_owners, destination_owners in zip(source, destination)
    ),
    key=_REQUIREMENT_RANK.__getitem__,
  )


@dataclass(frozen=True)
class GluonRegionPlan:
  """Layout roles and per-consumer views for one generated region.

  A consumer view states the layout one use of a value needs. The emitter
  converts into a fresh view instead of rebinding the value, so a second
  consumer keeps the layout its own entry states.

  `shape` is the logical extent the region computes over and `tile` is the
  power-of-two execution tile its layouts are assigned to. A plan that leaves
  `tile` unset claims the logical extent is already the tile, which a tailed
  shape cannot support.
  """

  name: str
  shape: tuple[int, int]
  load: BlockedLayoutPlan
  compute: BlockedLayoutPlan
  store: BlockedLayoutPlan
  reduction: BlockedLayoutPlan | None = None
  state_layout: BlockedLayoutPlan | None = None
  consumer_views: dict = field(default_factory=dict)
  tile: tuple[int, int] | None = None

  def view_for(self, value, node_index):
    """The layout of one consumer's explicit view, or None when it has none.

    A consumer without an override keeps the layout of the role its operation
    declares, which only the emitter knows, so this reports no layout rather
    than inventing one.
    """
    return self.consumer_views.get((value, node_index))

  def as_record(self):
    return {
      "name": self.name,
      "shape": list(self.shape),
      "tile": list(self.tile) if self.tile is not None else None,
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


@dataclass(frozen=True)
class RegionConsumers:
  """The consumer identities one generated region offers a layout plan.

  `operations` names the operation of every node in graph order, and `uses`
  pairs the alias of each operand with the index of the node that reads it.
  Aliases follow the emitter: the loop states keep their names and a node
  result is `v<node index>`, whatever the graph calls it.
  """

  shape: tuple[int, int]
  roots: tuple[str, ...]
  operations: tuple[str, ...]
  uses: tuple[tuple[str, int], ...]

  def uses_of(self, operation):
    """The (alias, node index) pairs read by one kind of operation."""
    return tuple(
      (value, index)
      for value, index in self.uses
      if self.operations[index] == operation
    )

  def uses_of_node(self, index):
    """The (alias, node index) pairs one node of the region reads."""
    return tuple(use for use in self.uses if use[1] == index)

  def consumers_of_node(self, index):
    """The (alias, node index) pairs that read one node's result."""
    result = _node_alias(index)
    return tuple(use for use in self.uses if use[0] == result)

  def expansion_uses(self, index):
    """The uses one reduction-fed broadcast expands into, with their operands.

    A broadcast pairs its operand with a reference and a binary operation
    pairs its two operands, so Gluon reads every value such a consumer reads
    in one layout. The expansion therefore names the value each consumer reads
    after the broadcast, and every other value that consumer reads. The
    broadcast's own operand is not one of them: the emitter pairs it with a
    reference that no consumer view addresses, so a view there would leave the
    pair in two layouts. Empty when no node reads the broadcast's result.
    """
    uses = []
    for result, consumer in self.consumers_of_node(index):
      uses.append((result, consumer))
      uses.extend(
        use for use in self.uses_of_node(consumer) if use != (result, consumer)
      )
    return tuple(uses)

  def shared_reduction_value(self):
    """The value two reductions read, with their node indexes, if there is one."""
    readers = {}
    for value, index in self.uses_of("sum"):
      readers.setdefault(value, []).append(index)
    for value, indexes in readers.items():
      if len(indexes) >= 2:
        return value, (indexes[0], indexes[1])
    return None

  def reduction_result_broadcasts(self):
    """The (alias, node index) pairs of broadcasts that read a reduction."""
    return tuple(
      (value, index)
      for value, index in self.uses_of("broadcast")
      if self._is_reduction_result(value)
    )

  def _is_reduction_result(self, value):
    index = _node_index(value)
    return index is not None and self.operations[index] == "sum"


def _node_index(alias):
  """The node index an emitter alias names, or None for a loop state."""
  match = re.fullmatch(r"v(\d+)", str(alias))
  return int(match[1]) if match else None


def _node_alias(index):
  """The alias the emitter gives the value one node produces."""
  return f"v{index}"


def region_consumers(source):
  """Read the emitter-visible consumers of one generated region source.

  The region grammar names its temporaries itself; a plan has to state the
  consumers the emitter will actually write, so this translates every operand
  into the emitter's alias for it and rejects an operand the emitter cannot
  name instead of guessing one.
  """
  shape, _iterations, _tensor_types, nodes, updates = _generated_region_program(source)
  aliases = {}
  for index, (destination, _operation, _operands, _attribute) in enumerate(nodes):
    aliases[destination] = _node_alias(index)
  roots = tuple(state for state, _root in updates)
  for state in roots:
    if state in aliases:
      raise ValueError(f"generated region reads and updates {state!r}")
    aliases[state] = state
  uses = []
  for index, (_destination, _operation, operands, _attribute) in enumerate(nodes):
    for operand in operands:
      if operand not in aliases:
        raise ValueError(
          f"generated region node {index} reads {operand!r}, which the emitter "
          "cannot name"
        )
      uses.append((aliases[operand], index))
  return RegionConsumers(
    tuple(shape),
    roots,
    tuple(operation for _destination, operation, _o, _a in nodes),
    tuple(uses),
  )


@dataclass(frozen=True)
class PlanFamilies:
  """The plan families one layout pool and one region graph can express.

  `plans` holds the families that are expressible. `omitted` names every other
  family of `PLAN_FAMILY_MINIMUM_LAYOUTS` with the reason it is missing, so a
  caller never has to guess whether the pool or the graph left it out.
  """

  plans: dict
  omitted: dict

  def __getitem__(self, name):
    return self.plans[name]

  def __contains__(self, name):
    return name in self.plans

  def __iter__(self):
    return iter(self.plans)

  def __len__(self):
    return len(self.plans)

  def get(self, name, default=None):
    return self.plans.get(name, default)

  def keys(self):
    return self.plans.keys()

  def values(self):
    return self.plans.values()

  def items(self):
    return self.plans.items()


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


def region_plan_families(shape, *, num_warps, layouts, consumers, tile=None):
  """Build the layout plan families one layout pool and one region can express.

  `layouts` holds distinct sampled blocked layouts, which the families place on
  the load, compute, reduction, per-consumer, loop-state, and store roles.
  `consumers` states who the region's graph lets a plan address, so a family
  views the values the region really has. A family the pool cannot afford or
  the graph cannot express is omitted with its reason rather than collapsing
  two of its roles onto one layout or inventing a consumer. `tile` is the
  execution tile the layouts and the plan are assigned to.
  """
  if not isinstance(layouts, (list, tuple)) or not layouts:
    raise ValueError("plan families need at least one distinct layout")
  for layout in layouts:
    layout.validate(shape, num_warps=num_warps, tile=tile)
  if not isinstance(consumers, RegionConsumers):
    raise ValueError("plan families need the region's consumer identities")

  # A region with no reduction has no reduction role, so no family declares
  # one: a parameter the kernel body never reads would state nothing.
  reductions = bool(consumers.uses_of("sum"))
  baseline = layouts[0]
  second = layouts[1 % len(layouts)]
  third = layouts[2 % len(layouts)]
  fourth = layouts[3 % len(layouts)]

  plans = {}
  omitted = {
    name: (
      f"plan family {name!r} places {required} distinct layouts on its roles, "
      f"but the run sampled {len(layouts)}"
    )
    for name, required in PLAN_FAMILY_MINIMUM_LAYOUTS.items()
    if len(layouts) < required
  }

  def region(name, load, compute, store, *, reduction=None, state=None, views=None):
    if name in omitted:
      return
    plans[name] = GluonRegionPlan(
      name,
      tuple(shape),
      load,
      compute,
      store,
      reduction=reduction if reductions else None,
      state_layout=state,
      consumer_views=dict(views or {}),
      tile=tuple(tile) if tile is not None else None,
    )

  region("all-same-layout", baseline, baseline, baseline, reduction=baseline)
  region("load-to-compute", second, baseline, baseline, reduction=baseline)

  shared = consumers.shared_reduction_value()
  if shared is None:
    for name in ("shared-value-reused", "split-reduction-consumers"):
      omitted.setdefault(name, "no value is read by two reduction nodes")
  else:
    value, (first, last) = shared
    region(
      "shared-value-reused",
      baseline,
      baseline,
      second,
      reduction=baseline,
      views={(value, first): second, (value, last): second},
    )
    region(
      "split-reduction-consumers",
      baseline,
      baseline,
      second,
      reduction=second,
      views={(value, first): second, (value, last): third},
    )

  broadcasts = consumers.reduction_result_broadcasts()
  expansions = [consumers.expansion_uses(index) for _value, index in broadcasts]
  if len(broadcasts) < 2:
    omitted.setdefault(
      "reduction-result-expand", "fewer than two reductions feed a broadcast"
    )
  elif any(not uses for uses in expansions[:2]):
    # The two expansion views need a consumer node to sit at: the emitter
    # reads a loop state directly, so a broadcast whose result only reaches a
    # state update offers the plan no use to state a layout on.
    omitted.setdefault(
      "reduction-result-expand",
      "a reduction-fed broadcast result reaches no consumer",
    )
  else:
    region(
      "reduction-result-expand",
      baseline,
      third,
      second,
      reduction=second,
      # Each broadcast reads one reduction's result, and the consumers of the
      # expansion read it in the view that differs from the compute layout the
      # emitter would otherwise give them.
      views={use: fourth for use in expansions[0] + expansions[1]},
    )

  region("loop-state-layout", baseline, baseline, fourth, reduction=second, state=third)
  return PlanFamilies(plans, omitted)
