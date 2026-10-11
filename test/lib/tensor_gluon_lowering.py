"""Lower the admitted generated tensor-region subset into Gluon source."""

import re
from dataclasses import dataclass

from test.lib.tensor_exact_f32 import _generated_region_program
from test.lib.tensor_gluon import BlockedLayoutPlan


@dataclass(frozen=True)
class GeneratedGluonKernel:
  source: str
  shape: tuple[int, int]
  iterations: int
  conversion_source_lines: dict[int, str]


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


def render_gluon_kernel(source, *, plan=None):
  """Emit one Gluon kernel from the same typed region parsed by the exact proof.

  `plan` states the layout of each role and of each consumer's view. A consumer
  view is a fresh local, so converting for one consumer never changes the
  layout another consumer reads; views that ask for a layout a value already
  has are shared instead of recomputed. Without a plan the kernel keeps one
  compute layout and the column reduction layout, as before.
  """
  shape, iterations, tensor_types, nodes, updates = _generated_region_program(source)
  if any(dimension & (dimension - 1) for dimension in shape):
    raise ValueError("Gluon region lowering requires power-of-two dimensions")
  if plan is not None and tuple(plan.shape) != tuple(shape):
    raise ValueError("layout plan shape differs from the generated region shape")

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

  def emit(line, *, conversion=None):
    lines.append(line)
    if conversion is not None:
      conversion_source_lines[len(lines)] = conversion

  current_layouts = {"acc_x": compute_layout, "acc_y": compute_layout}
  view_names = set()

  def use(value, node_index, fallback_layout, label):
    """Name the local one consumer reads, converting into a view if needed."""
    wanted = (
      _layout_identifier(plan.view_for(value, node_index).name)
      if plan is not None
      else fallback_layout
    )
    if plan is None:
      emit(f"        {value} = gl.convert_layout({value}, {wanted})", conversion=label)
      return value
    if current_layouts.get(value) == wanted:
      return value
    name = f"{value}_view_{wanted}"
    if name not in view_names:
      emit(f"        {name} = gl.convert_layout({value}, {wanted})", conversion=label)
      view_names.add(name)
    return name

  rows, columns = shape
  emit(
    f"    load_rows = gl.arange(0, {rows}, "
    f"layout=gl.SliceLayout(1, parent={load_layout}))"
  )
  emit(
    f"    load_columns = gl.arange(0, {columns}, "
    f"layout=gl.SliceLayout(0, parent={load_layout}))"
  )
  emit(
    "    load_offsets = load_rows[:, None] * "
    + str(columns)
    + " + load_columns[None, :]"
  )
  emit(
    f"    acc_x = gl.convert_layout(gl.load(input_ptr + load_offsets), {compute_layout})",
    conversion="load_to_compute",
  )
  emit(
    f"    acc_y = gl.zeros(({rows}, {columns}), gl.float32, layout={compute_layout})"
  )
  emit(f"    for _ in range({iterations}):")

  if state_layout is not None:
    for state in ("acc_x", "acc_y"):
      emit(
        f"        {state} = gl.convert_layout({state}, {compute_layout})",
        conversion=f"{state}_from_state_layout",
      )
      current_layouts[state] = compute_layout

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
        emit(
          f"        {reduction_input} = gl.convert_layout({operand}, {reduction_layout})",
          conversion=f"region_sum_{index}_to_reduction",
        )
      else:
        reduction_input = use(
          operand,
          index,
          reduction_layout,
          f"region_sum_{index}_to_reduction",
        )
      emit(
        f"        {value} = gl.convert_layout("
        f"gl.expand_dims(gl.sum({reduction_input}, axis={axis}), {axis}), {compute_layout})",
        conversion=f"region_sum_{index}_result_layout",
      )
      current_layouts[value] = compute_layout
    elif operation == "broadcast":
      operand_name = value_names[operands[0]]
      operand_shape = tensor_types[operands[0]]
      operand_layout = _shape_layout(operand_shape, shape, compute_layout)
      target_layout = _shape_layout(result_shape, shape, compute_layout)
      operand = use(
        operand_name, index, operand_layout, f"region_broadcast_{index}_source_layout"
      )
      if tuple(result_shape) == shape:
        reference = "acc_x"
      else:
        emit(
          f"        {value}_reference = gl.zeros({tuple(result_shape)!r}, "
          f"gl.float32, layout={target_layout})"
        )
        reference = f"{value}_reference"
      emit(f"        {value}, _ = gl.broadcast({operand}, {reference})")
      emit(
        f"        {value} = gl.convert_layout({value}, {target_layout})",
        conversion=f"region_broadcast_{index}_result_layout",
      )
      current_layouts[value] = target_layout
    else:
      left_name, right_name = (value_names[name] for name in operands)
      left_shape, right_shape = (tensor_types[name] for name in operands)
      left_layout = _shape_layout(left_shape, shape, compute_layout)
      right_layout = _shape_layout(right_shape, shape, compute_layout)
      target_layout = _shape_layout(result_shape, shape, compute_layout)
      left = use(
        left_name, index, left_layout, f"region_{operation}_{index}_left_layout"
      )
      right = use(
        right_name, index, right_layout, f"region_{operation}_{index}_right_layout"
      )
      emit(f"        {value}_left, {value}_right = gl.broadcast({left}, {right})")
      operator = {"add": "+", "sub": "-", "mul": "*"}[operation]
      emit(f"        {value} = {value}_left {operator} {value}_right")
      emit(
        f"        {value} = gl.convert_layout({value}, {target_layout})",
        conversion=f"region_{operation}_{index}_result_layout",
      )
      current_layouts[value] = target_layout

  for state, root in updates:
    emit(f"        {state} = {value_names[root]}")
    current_layouts[state] = current_layouts.get(value_names[root], compute_layout)

  if state_layout is not None:
    for state in ("acc_x", "acc_y"):
      emit(
        f"        {state} = gl.convert_layout({state}, {state_layout})",
        conversion=f"{state}_to_state_layout",
      )
      current_layouts[state] = state_layout

  emit("    out_left, out_right = gl.broadcast(acc_x, acc_y)")
  emit("    output = out_left + out_right")
  emit(
    f"    output = gl.convert_layout(output, {store_layout})",
    conversion="compute_to_store",
  )
  emit(
    f"    store_rows = gl.arange(0, {rows}, "
    f"layout=gl.SliceLayout(1, parent={store_layout}))"
  )
  emit(
    f"    store_columns = gl.arange(0, {columns}, "
    f"layout=gl.SliceLayout(0, parent={store_layout}))"
  )
  emit(
    "    store_offsets = store_rows[:, None] * "
    + str(columns)
    + " + store_columns[None, :]"
  )
  emit("    gl.store(output_ptr + store_offsets, output)")
  return GeneratedGluonKernel(
    "\n".join(lines) + "\n", shape, iterations, conversion_source_lines
  )
