"""Turn one exported free-form region into a target-only bundle.

The C++ exporter states the SIR and the region record; this module states the
bundle around them: dense physical layouts, the guard envelope, and the input
bytes the record's parameters describe. Nothing here reads a certificate or an
expected output, so a body the canonical grammar cannot express still becomes a
bundle.
"""

from test.lib import tensor_free_form_inputs, tensor_target_case

DEFAULT_ENVELOPE = {"prefix": 1, "suffix": 1}
GAP_SENTINEL = {"f32": "3f000001", "i32": "55555555"}
OUTPUT_CANARY = {"f32": "3f000002", "i32": "aaaaaaaa"}
PADDING = {"f32": "3f000003", "i32": "55555556"}


def element_count(shape):
  count = 1
  for extent in shape:
    count *= extent
  return count


def dense_layout(shape):
  """The row-major layout of one dense tensor, with axes fastest first."""
  if not shape or any(type(extent) is not int or extent <= 0 for extent in shape):
    raise tensor_target_case.TargetCaseError(
      tensor_target_case.INVALID_LAYOUT, f"a dense layout needs a positive shape, got {shape!r}"
    )
  strides = [1] * len(shape)
  for axis in range(len(shape) - 2, -1, -1):
    strides[axis] = strides[axis + 1] * shape[axis + 1]
  return {
    "shape": list(shape),
    "element_strides": strides,
    "element_offset": 0,
    "storage_span": element_count(shape),
    "axis_order": sorted(range(len(shape)), key=lambda axis: (strides[axis], axis)),
  }


def strided_output(shape, row_padding=0, element_offset=0):
  """A row-major layout with padding between rows and a leading offset.

  `row_padding` adds that many unused elements to every row stride, which only
  a matrix has; a rank-one tensor takes the offset alone. The span is the
  highest addressed element plus one, so a padded or shifted output still
  states exactly the storage it needs.
  """
  if not shape or any(type(extent) is not int or extent <= 0 for extent in shape):
    raise tensor_target_case.TargetCaseError(
      tensor_target_case.INVALID_LAYOUT, f"a layout needs a positive shape, got {shape!r}"
    )
  if type(row_padding) is not int or row_padding < 0:
    raise tensor_target_case.TargetCaseError(
      tensor_target_case.INVALID_LAYOUT, "row padding must be a nonnegative integer"
    )
  if type(element_offset) is not int or element_offset < 0:
    raise tensor_target_case.TargetCaseError(
      tensor_target_case.INVALID_LAYOUT, "element offset must be a nonnegative integer"
    )
  if row_padding and len(shape) < 2:
    raise tensor_target_case.TargetCaseError(
      tensor_target_case.INVALID_LAYOUT, "row padding needs a matrix shape"
    )
  layout = dense_layout(shape)
  strides = list(layout["element_strides"])
  if row_padding:
    strides[-2] += row_padding
    # Padding widens the row stride, so every outer stride must be widened too
    # or the addresses of different outer coordinates would interleave. Each
    # outer stride covers one full padded inner extent.
    for axis in range(len(shape) - 3, -1, -1):
      strides[axis] = max(strides[axis], shape[axis + 1] * strides[axis + 1])
  span = element_offset + sum(
    (extent - 1) * stride for extent, stride in zip(shape, strides)
  ) + 1
  return {
    "shape": list(shape),
    "element_strides": strides,
    "element_offset": element_offset,
    "storage_span": span,
    "axis_order": sorted(range(len(shape)), key=lambda axis: (strides[axis], axis)),
  }


def execution_record(*, num_warps, envelope=None):
  """The staging record the shared worker reads for one bundle."""
  guards = dict(DEFAULT_ENVELOPE if envelope is None else envelope)
  return {
    "guard_envelope": guards,
    "gap_sentinel_by_dtype": dict(GAP_SENTINEL),
    "output_canary_by_dtype": dict(OUTPUT_CANARY),
    "padding_by_dtype": dict(PADDING),
    "warp_probe": {"status": "not_probed", "num_warps": num_warps},
  }


def input_records(region, proposal, *, layouts=None):
  """One record per runtime parameter, each carrying its own bytes.

  `proposal` packs the parameters in the order the region record lists them,
  so each record takes the next element-count slice. `layouts` optionally maps
  a parameter name to the physical layout the worker should stage; every
  unnamed parameter keeps the dense layout. A mismatch between the proposal
  and the record is refused rather than padded.
  """
  parameters = tensor_free_form_inputs.parameters(region)
  if layouts is not None:
    if not isinstance(layouts, dict):
      raise tensor_target_case.TargetCaseError(
        tensor_target_case.INVALID_LAYOUT, "input layouts must be a mapping by parameter name"
      )
    known = {parameter["name"] for parameter in parameters}
    unknown = sorted(set(layouts) - known)
    if unknown:
      raise tensor_target_case.TargetCaseError(
        tensor_target_case.INVALID_LAYOUT,
        "an input layout names an unknown parameter: " + ", ".join(unknown),
      )
  if tuple(proposal.names) != tuple(parameter["name"] for parameter in parameters):
    raise tensor_target_case.TargetCaseError(
      tensor_target_case.INVALID_CASE,
      "the proposal does not name the region's runtime parameters",
    )
  records = []
  offset = 0
  for parameter in parameters:
    length = element_count(parameter["shape"]) * 4
    data = proposal.data[offset : offset + length]
    if len(data) != length:
      raise tensor_target_case.TargetCaseError(
        tensor_target_case.INVALID_CASE,
        f"the proposal is shorter than parameter {parameter['name']} needs",
      )
    layout = dense_layout(parameter["shape"])
    if layouts is not None and parameter["name"] in layouts:
      given = layouts[parameter["name"]]
      if not isinstance(given, dict):
        raise tensor_target_case.TargetCaseError(
          tensor_target_case.INVALID_LAYOUT,
          f"the layout for {parameter['name']} must be an object",
        )
      if given.get("shape") != list(parameter["shape"]):
        raise tensor_target_case.TargetCaseError(
          tensor_target_case.INVALID_LAYOUT,
          f"the layout for {parameter['name']} states another shape",
        )
      tensor_target_case._validate_layout(given, "input")
      layout = {
        key: given[key]
        for key in (
          "shape",
          "element_strides",
          "element_offset",
          "storage_span",
          "axis_order",
        )
      }
    records.append(
      {
        "name": parameter["name"],
        "dtype": parameter["dtype"],
        **layout,
        "data": data,
      }
    )
    offset += length
  if offset != len(proposal.data):
    raise tensor_target_case.TargetCaseError(
      tensor_target_case.INVALID_CASE,
      "the proposal carries bytes the region's parameters do not describe",
    )
  return records


def build_bundle(
  bundle_dir,
  *,
  region,
  source,
  proposal,
  envelope=None,
  output_row_padding=0,
  output_offset=0,
  input_layouts=None,
  num_stages=1,
  loop_bound=tensor_target_case.LITERAL_LOOP_BOUND,
  loop_hint=None,
):
  """Write the bundle for one region record and return the loaded case."""
  num_warps = region.get("num_warps")
  if type(num_warps) is not int or num_warps <= 0:
    raise tensor_target_case.TargetCaseError(
      tensor_target_case.INVALID_CASE, "the region record must state num_warps"
    )
  output = {
    "name": "return",
    "dtype": region["dtype"],
    **strided_output(
      region["shape"],
      row_padding=output_row_padding,
      element_offset=output_offset,
    ),
  }
  return tensor_target_case.write(
    bundle_dir,
    source=source,
    inputs=input_records(region, proposal, layouts=input_layouts),
    output=output,
    launch={
      "num_warps": num_warps,
      "num_stages": num_stages,
      "enable_fp_fusion": False,
    },
    execution=execution_record(num_warps=num_warps, envelope=envelope),
    profile=region["profile"],
    loop_bound=loop_bound,
    loop_hint=loop_hint,
  )
