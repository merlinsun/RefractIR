"""Lower the admitted generated tensor-region subset into Gluon source."""

import re
from dataclasses import dataclass

from test.lib.tensor_exact_f32 import _generated_region_program
from test.lib.tensor_gluon import BlockedLayoutPlan, execution_tile


@dataclass(frozen=True)
class GeneratedGluonKernel:
  source: str
  shape: tuple[int, int]
  iterations: int
  conversion_source_lines: dict[int, str]
  # Identifiers of the constexpr layout parameters, in signature order. A
  # caller supplying a plan passes the matching layout objects positionally.
  layout_parameters: tuple[str, ...] = ()
  # The power-of-two execution tile the kernel places values on. It equals the
  # logical shape when the shape has no tail, and is the padded tile otherwise.
  tile: tuple[int, int] | None = None

  def layout_arguments(self, layouts):
    """Map each layout parameter to the plan object that names it."""
    by_identifier = {_layout_identifier(layout.name): layout for layout in layouts}
    missing = [
      identifier
      for identifier in self.layout_parameters
      if identifier not in by_identifier
    ]
    if missing:
      raise ValueError(f"layout parameters without a plan object: {missing}")
    return [by_identifier[identifier] for identifier in self.layout_parameters]


def _shape_layout(shape, full_shape, parent):
  if len(shape) != 2 or any(
    extent > full_extent for extent, full_extent in zip(shape, full_shape)
  ):
    raise ValueError("generated region tensor shape is outside the rank-two layout")
  return parent


def cross_lane_threads_per_warp(shape, reduction_axes):
  """Choose a legal rank-two warp layout that spans one reduction axis."""
  if (
    not isinstance(shape, (tuple, list))
    or len(shape) != 2
    or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
  ):
    raise ValueError("cross-lane reduction shape must have rank two")
  axes = tuple(reduction_axes)
  if any(type(axis) is not int or axis not in (0, 1) for axis in axes):
    raise ValueError("cross-lane reduction axes must be 0 or 1")
  for axis in (1, 0):
    if axis in axes and shape[axis] > 1:
      return (1, 32) if axis == 1 else (32, 1)
  raise ValueError("region has no nontrivial reduction axis for cross-lane testing")


def reduction_blocked_layout(shape, reduction_axes, *, num_warps):
  """Place CTA warps on the non-reduced axis, spilling across axes if needed."""
  if type(num_warps) is not int or num_warps <= 0 or num_warps & (num_warps - 1):
    raise ValueError("num_warps must be a positive power of two")
  threads_per_warp = cross_lane_threads_per_warp(shape, reduction_axes)
  reduction_axis = 1 if threads_per_warp == (1, 32) else 0
  parallel_axis = 1 - reduction_axis

  parallel_extent = shape[parallel_axis]
  warps_on_parallel_axis = 1 << min(
    num_warps.bit_length() - 1,
    parallel_extent.bit_length() - 1,
  )
  warps_on_reduction_axis = num_warps // warps_on_parallel_axis
  warps_per_cta = [1, 1]
  warps_per_cta[parallel_axis] = warps_on_parallel_axis
  warps_per_cta[reduction_axis] = warps_on_reduction_axis
  order = (1, 0) if reduction_axis == 1 else (0, 1)
  return BlockedLayoutPlan(
    f"reduction-blocked-{num_warps}-warps",
    (1, 1),
    threads_per_warp,
    tuple(warps_per_cta),
    order,
  )


def _layout_identifier(name):
  """Turn a layout name into the identifier the kernel body can reference."""
  identifier = re.sub(r"[^0-9a-zA-Z_]", "_", str(name))
  if not identifier or identifier[0].isdigit():
    identifier = "layout_" + identifier
  return identifier


def render_gluon_kernel(source, *, plan=None, tile=None):
  """Emit one Gluon kernel from the same typed region parsed by the exact proof.

  `plan` states the layout of each role and of each consumer's view. A consumer
  view is a fresh local, so converting for one consumer never changes the
  layout another consumer reads; views that ask for a layout a value already
  has are shared instead of recomputed. Without a plan the kernel keeps one
  compute layout and the column reduction layout, as before.

  `tile` is the power-of-two execution tile the region runs on. A logical
  extent that is already a power of two is its own tile and emits no mask. A
  tailed extent states its tile explicitly (or through its plan) and the
  emitter keeps that tile while honouring the logical extent: the load and the
  store carry the tile-shaped validity predicate, and a reduction zeroes the
  padded lanes before it sums, so no padded lane can reach a stored value.
  """
  shape, iterations, tensor_types, nodes, updates = _generated_region_program(source)
  if plan is not None and tuple(plan.shape) != tuple(shape):
    raise ValueError("layout plan shape differs from the generated region shape")
  plan_tile = plan.tile if plan is not None else None
  if tile is not None and plan_tile is not None and tuple(tile) != tuple(plan_tile):
    raise ValueError("declared execution tile differs from the layout plan's tile")
  declared = tile if tile is not None else plan_tile
  if declared is None:
    if any(dimension & (dimension - 1) for dimension in shape):
      raise ValueError("Gluon region lowering requires power-of-two dimensions")
    tile = shape
  else:
    tile = execution_tile(shape, tile=declared)
  padded = tuple(tile) != tuple(shape)

  if plan is None:
    load_layout, compute_layout = "load_layout", "compute_layout"
    reduction_layout, store_layout = "reduction_layout", "store_layout"
    state_layout = None
    constexpr_layouts = [
      load_layout,
      compute_layout,
      reduction_layout,
      store_layout,
    ]
  else:
    load_layout = _layout_identifier(plan.load.name)
    compute_layout = _layout_identifier(plan.compute.name)
    reduction_layout = (
      _layout_identifier(plan.reduction.name) if plan.reduction else compute_layout
    )
    store_layout = _layout_identifier(plan.store.name)
    state_layout = (
      _layout_identifier(plan.state_layout.name) if plan.state_layout else None
    )
    # Every layout the body names is a constexpr argument, so the caller keeps
    # control of the plan instead of the emitter inventing one.
    constexpr_layouts = []
    for identifier in (
      load_layout,
      compute_layout,
      reduction_layout,
      store_layout,
      state_layout,
      *(_layout_identifier(view.name) for view in plan.consumer_views.values()),
    ):
      if identifier is not None and identifier not in constexpr_layouts:
        constexpr_layouts.append(identifier)

  lines = [
    "import triton",
    "from triton.experimental import gluon",
    "from triton.experimental.gluon import language as gl",
    "",
    "@gluon.jit",
    "def generated_region_kernel(",
    "    input_ptr,",
    "    output_ptr,",
  ]
  lines.extend(f"    {identifier}: gl.constexpr," for identifier in constexpr_layouts)
  lines.append("):")
  conversion_source_lines = {}

  def emit(line, *, indent=4, conversion=None):
    lines.append(" " * indent + line)
    if conversion is not None:
      conversion_source_lines[len(lines)] = conversion

  current_layouts = {"acc_x": compute_layout, "acc_y": compute_layout}
  view_names = set()
  plan_uses = []
  # One validity predicate per layout that reads it. The predicate is built
  # from that layout's own index vectors, so the mask never has to change
  # layout itself; the scopes stay apart because a name bound inside the loop
  # body is not visible to the store after it.
  masks = {}
  loop_masks = {}

  def use(value, node_index, fallback_layout, label):
    """Name the local one consumer reads, and the layout that local carries.

    An explicit consumer view wins. Without one the consumer keeps the layout
    its operation's role declares, which is the fallback the call site states.
    """
    plan_uses.append((value, node_index, fallback_layout))
    override = plan.view_for(value, node_index) if plan is not None else None
    wanted = (
      _layout_identifier(override.name) if override is not None else fallback_layout
    )
    if plan is None:
      emit(
        f"{value} = gl.convert_layout({value}, {wanted})", indent=8, conversion=label
      )
      return value, wanted
    if current_layouts.get(value) == wanted:
      return value, wanted
    name = f"{value}_view_{wanted}"
    if name not in view_names:
      emit(f"{name} = gl.convert_layout({value}, {wanted})", indent=8, conversion=label)
      view_names.add(name)
    return name, wanted

  def mask_in(layout, *, indent, rows_name=None, columns_name=None):
    """The tile-shaped predicate that marks the logical elements of one layout.

    The predicate is built from index vectors in the layout that reads it, so
    the mask itself never has to change layout. A role that already emitted its
    index vectors passes their names in; any other layout gets its own. The
    predicate is emitted once per layout and scope and reused after that, so a
    plan that sends several consumers through one view pays for its mask once.
    Without padding there is no predicate, and every mask user then keeps the
    unmasked form.
    """
    if not padded:
      return None
    names = loop_masks if indent > 4 else masks
    if layout in names:
      return names[layout]
    if (rows_name is None) != (columns_name is None):
      raise ValueError("a mask either reuses both index vectors or emits its own")
    if rows_name is None:
      rows_name = f"{layout}_rows"
      columns_name = f"{layout}_columns"
      emit(
        f"{rows_name} = gl.arange(0, {tile[0]}, "
        f"layout=gl.SliceLayout(1, parent={layout}))",
        indent=indent,
      )
      emit(
        f"{columns_name} = gl.arange(0, {tile[1]}, "
        f"layout=gl.SliceLayout(0, parent={layout}))",
        indent=indent,
      )
    name = f"valid_{layout}"
    emit(
      f"{name} = ({rows_name}[:, None] < {shape[0]}) & "
      f"({columns_name}[None, :] < {shape[1]})",
      indent=indent,
    )
    names[layout] = name
    return name

  rows, columns = tile
  emit(
    f"load_rows = gl.arange(0, {rows}, layout=gl.SliceLayout(1, parent={load_layout}))"
  )
  emit(
    f"load_columns = gl.arange(0, {columns}, "
    f"layout=gl.SliceLayout(0, parent={load_layout}))"
  )
  # The tile addresses the logical row-major buffer: a padded lane computes an
  # address it never dereferences, because the load mask excludes it.
  emit(f"load_offsets = load_rows[:, None] * {shape[1]} + load_columns[None, :]")
  load_mask = mask_in(
    load_layout,
    indent=4,
    rows_name="load_rows",
    columns_name="load_columns",
  )
  load_call = "gl.load(input_ptr + load_offsets"
  if load_mask is not None:
    load_call += f", mask={load_mask}, other=0.0"
  emit(
    f"acc_x = gl.convert_layout({load_call}), {compute_layout})",
    conversion="load_to_compute",
  )
  emit(f"acc_y = gl.zeros(({rows}, {columns}), gl.float32, layout={compute_layout})")
  if state_layout is not None:
    # A loop-carried value keeps one type for the whole loop, so the state
    # layout is established before the loop and the assigned roots are
    # converted into it. The body's own operations stay in the compute layout.
    for state in ("acc_x", "acc_y"):
      emit(
        f"{state} = gl.convert_layout({state}, {state_layout})",
        conversion=f"{state}_to_state_layout",
      )
      current_layouts[state] = state_layout
  emit(f"for _ in range({iterations}):")

  value_names = {"acc_x": "acc_x", "acc_y": "acc_y"}
  for index, (destination, _operation, _operands, _attribute) in enumerate(nodes):
    value_names[destination] = f"v{index}"

  for index, (destination, operation, operands, attribute) in enumerate(nodes):
    value = value_names[destination]
    result_shape = tensor_types[destination]
    if operation == "sum":
      operand = value_names[operands[0]]
      axis = attribute if attribute >= 0 else attribute + len(shape)
      if plan is None:
        reduction_input = f"{operand}_reduction_input_{index}"
        reduction_input_layout = reduction_layout
        emit(
          f"{reduction_input} = gl.convert_layout({operand}, {reduction_layout})",
          indent=8,
          conversion=f"region_sum_{index}_to_reduction",
        )
      else:
        reduction_input, reduction_input_layout = use(
          operand,
          index,
          reduction_layout,
          f"region_sum_{index}_to_reduction",
        )
      # A reduction never sees a padded lane: the mask zeroes them in the
      # operand's own layout, so whatever the padding holds cannot join the
      # partial sums (and a sum of zeros is exactly what the logical extent
      # reduces over).
      summand = reduction_input
      reduction_mask = mask_in(reduction_input_layout, indent=8)
      if reduction_mask is not None:
        summand = f"{reduction_input}_masked"
        emit(
          f"{summand} = gl.where({reduction_mask}, {reduction_input}, 0.0)",
          indent=8,
        )
      emit(
        f"{value} = gl.convert_layout("
        f"gl.expand_dims(gl.sum({summand}, axis={axis}), {axis}), {compute_layout})",
        indent=8,
        conversion=f"region_sum_{index}_result_layout",
      )
      current_layouts[value] = compute_layout
    elif operation == "broadcast":
      operand_name = value_names[operands[0]]
      operand_shape = tensor_types[operands[0]]
      operand_layout = _shape_layout(operand_shape, shape, compute_layout)
      target_layout = _shape_layout(result_shape, shape, compute_layout)
      operand, _operand_layout = use(
        operand_name, index, operand_layout, f"region_broadcast_{index}_source_layout"
      )
      if tuple(result_shape) == shape:
        reference = "acc_x"
        if plan is not None and current_layouts.get("acc_x") != target_layout:
          # The broadcast reference is a value, not just a shape, so it has to
          # carry the layout the result will have.
          reference = f"acc_x_view_{target_layout}"
          if reference not in view_names:
            emit(
              f"{reference} = gl.convert_layout(acc_x, {target_layout})",
              indent=8,
              conversion=f"region_broadcast_{index}_reference_layout",
            )
            view_names.add(reference)
      else:
        emit(
          f"{value}_reference = gl.zeros({tuple(result_shape)!r}, "
          f"gl.float32, layout={target_layout})",
          indent=8,
        )
        reference = f"{value}_reference"
      emit(f"{value}, _ = gl.broadcast({operand}, {reference})", indent=8)
      emit(
        f"{value} = gl.convert_layout({value}, {target_layout})",
        indent=8,
        conversion=f"region_broadcast_{index}_result_layout",
      )
      current_layouts[value] = target_layout
    else:
      left_name, right_name = (value_names[name] for name in operands)
      left_shape, right_shape = (tensor_types[name] for name in operands)
      left_layout = _shape_layout(left_shape, shape, compute_layout)
      right_layout = _shape_layout(right_shape, shape, compute_layout)
      target_layout = _shape_layout(result_shape, shape, compute_layout)
      left, _left_layout = use(
        left_name, index, left_layout, f"region_{operation}_{index}_left_layout"
      )
      right, _right_layout = use(
        right_name, index, right_layout, f"region_{operation}_{index}_right_layout"
      )
      emit(f"{value}_left, {value}_right = gl.broadcast({left}, {right})", indent=8)
      operator = {"add": "+", "sub": "-", "mul": "*"}[operation]
      emit(f"{value} = {value}_left {operator} {value}_right", indent=8)
      emit(
        f"{value} = gl.convert_layout({value}, {target_layout})",
        indent=8,
        conversion=f"region_{operation}_{index}_result_layout",
      )
      current_layouts[value] = target_layout

  for state, root in updates:
    root_value = value_names[root]
    if state_layout is not None and current_layouts.get(root_value) != state_layout:
      emit(
        f"{state} = gl.convert_layout({root_value}, {state_layout})",
        indent=8,
        conversion=f"{state}_to_state_layout",
      )
      current_layouts[state] = state_layout
    else:
      emit(f"{state} = {root_value}", indent=8)
      current_layouts[state] = current_layouts.get(root_value, compute_layout)

  emit("out_left, out_right = gl.broadcast(acc_x, acc_y)")
  emit("output = out_left + out_right")
  emit(
    f"output = gl.convert_layout(output, {store_layout})",
    conversion="compute_to_store",
  )
  emit(
    f"store_rows = gl.arange(0, {rows}, "
    f"layout=gl.SliceLayout(1, parent={store_layout}))"
  )
  emit(
    f"store_columns = gl.arange(0, {columns}, "
    f"layout=gl.SliceLayout(0, parent={store_layout}))"
  )
  emit(f"store_offsets = store_rows[:, None] * {shape[1]} + store_columns[None, :]")
  store_mask = mask_in(
    store_layout,
    indent=4,
    rows_name="store_rows",
    columns_name="store_columns",
  )
  store_call = "gl.store(output_ptr + store_offsets, output"
  if store_mask is not None:
    store_call += f", mask={store_mask}"
  emit(store_call + ")")
  if plan is not None:
    # A declared view that no consumer ever queries means the plan and the
    # graph disagree about who reads what, so the family would silently do
    # nothing. Report it instead of emitting a kernel that ignores the plan.
    queried = {(value, index) for value, index, _layout in plan_uses}
    unqueried = sorted(set(plan.consumer_views) - queried)
    if unqueried:
      raise ValueError(
        "layout plan declares consumer views the region never queries: "
        + ", ".join(f"{value}#{index}" for value, index in unqueried)
      )
  return GeneratedGluonKernel(
    "\n".join(lines) + "\n",
    shape,
    iterations,
    conversion_source_lines,
    tuple(constexpr_layouts),
    tile,
  )
