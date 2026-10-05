"""Lower the admitted generated tensor-region subset into Gluon source."""

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


def render_gluon_kernel(source):
  """Emit one Gluon kernel from the same typed region parsed by the exact proof."""
  shape, iterations, tensor_types, nodes, updates = _generated_region_program(source)
  if any(dimension & (dimension - 1) for dimension in shape):
    raise ValueError("Gluon region lowering requires power-of-two dimensions")

  lines = [
    "import triton",
    "from triton.experimental import gluon",
    "from triton.experimental.gluon import language as gl",
    "",
    "@gluon.jit",
    "def generated_region_kernel(",
    "    input_ptr,",
    "    output_ptr,",
    "    load_layout: gl.constexpr,",
    "    compute_layout: gl.constexpr,",
    "    reduction_layout: gl.constexpr,",
    "    store_layout: gl.constexpr,",
    "):",
  ]
  conversion_source_lines = {}

  def emit(line, *, conversion=None):
    lines.append(line)
    if conversion is not None:
      conversion_source_lines[len(lines)] = conversion

  rows, columns = shape
  emit(
    f"    load_rows = gl.arange(0, {rows}, "
    "layout=gl.SliceLayout(1, parent=load_layout))"
  )
  emit(
    f"    load_columns = gl.arange(0, {columns}, "
    "layout=gl.SliceLayout(0, parent=load_layout))"
  )
  emit(
    "    load_offsets = load_rows[:, None] * "
    + str(columns)
    + " + load_columns[None, :]"
  )
  emit(
    "    acc_x = gl.convert_layout(gl.load(input_ptr + load_offsets), compute_layout)",
    conversion="load_to_compute",
  )
  emit(f"    acc_y = gl.zeros(({rows}, {columns}), gl.float32, layout=compute_layout)")
  emit(f"    for _ in range({iterations}):")

  value_names = {"acc_x": "acc_x", "acc_y": "acc_y"}
  for index, (destination, _operation, _operands, _attribute) in enumerate(nodes):
    value_names[destination] = f"v{index}"

  for index, (destination, operation, operands, attribute) in enumerate(nodes):
    value = value_names[destination]
    result_shape = tensor_types[destination]
    if operation == "sum":
      operand = value_names[operands[0]]
      axis = attribute if attribute >= 0 else attribute + len(shape)
      emit(
        f"        {operand} = gl.convert_layout({operand}, reduction_layout)",
        conversion=f"region_sum_{index}_to_reduction",
      )
      emit(
        f"        {value} = gl.convert_layout("
        f"gl.expand_dims(gl.sum({operand}, axis={axis}), {axis}), compute_layout)",
        conversion=f"region_sum_{index}_result_layout",
      )
    elif operation == "broadcast":
      operand = value_names[operands[0]]
      operand_shape = tensor_types[operands[0]]
      operand_layout = _shape_layout(operand_shape, shape, "compute_layout")
      target_layout = _shape_layout(result_shape, shape, "compute_layout")
      emit(
        f"        {operand} = gl.convert_layout({operand}, {operand_layout})",
        conversion=f"region_broadcast_{index}_source_layout",
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
    else:
      left, right = (value_names[name] for name in operands)
      left_shape, right_shape = (tensor_types[name] for name in operands)
      left_layout = _shape_layout(left_shape, shape, "compute_layout")
      right_layout = _shape_layout(right_shape, shape, "compute_layout")
      target_layout = _shape_layout(result_shape, shape, "compute_layout")
      operation_name = operation
      emit(
        f"        {left} = gl.convert_layout({left}, {left_layout})",
        conversion=f"region_{operation_name}_{index}_left_layout",
      )
      emit(
        f"        {right} = gl.convert_layout({right}, {right_layout})",
        conversion=f"region_{operation_name}_{index}_right_layout",
      )
      emit(f"        {value}_left, {value}_right = gl.broadcast({left}, {right})")
      operator = {"add": "+", "sub": "-", "mul": "*"}[operation]
      emit(f"        {value} = {value}_left {operator} {value}_right")
      emit(
        f"        {value} = gl.convert_layout({value}, {target_layout})",
        conversion=f"region_{operation_name}_{index}_result_layout",
      )

  for state, root in updates:
    emit(f"        {state} = {value_names[root]}")

  emit("    out_left, out_right = gl.broadcast(acc_x, acc_y)")
  emit("    output = out_left + out_right")
  emit(
    "    output = gl.convert_layout(output, store_layout)",
    conversion="compute_to_store",
  )
  emit(
    f"    store_rows = gl.arange(0, {rows}, "
    "layout=gl.SliceLayout(1, parent=store_layout))"
  )
  emit(
    f"    store_columns = gl.arange(0, {columns}, "
    "layout=gl.SliceLayout(0, parent=store_layout))"
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
