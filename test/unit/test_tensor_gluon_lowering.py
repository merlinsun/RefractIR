"""Shared generated-region graphs lower to Gluon source and run on the CPU."""

import ast
import hashlib
import re
import struct
import unittest
from pathlib import Path

from test.lib import tensor_input_proposals, tensor_layout
from test.lib.build_tensor_exact_region_case import physical_layout
from test.lib.tensor_exact_f32 import certify
from test.lib.tensor_gluon import (
  BlockedLayoutPlan,
  GluonRegionPlan,
  generate_layout_plans,
)
from test.lib.tensor_gluon_lowering import (
  cross_lane_threads_per_warp,
  render_gluon_kernel,
)
from test.lib.tensor_gluon_replay import replay_gluon_kernel
from test.unit.test_tensor_exact_composed import (
  chained_region_source,
  generated_region_source,
  generated_sequential_region_source,
  tailed_region_source,
)

ORTHOGONAL_PROGRAM_PATH = (
  Path(__file__).resolve().parents[1]
  / "tensor/gluon/programs/orthogonal-reductions-region-v1.sir"
)

TAIL_PROGRAM_PATH = (
  Path(__file__).resolve().parents[1]
  / "tensor/gluon/programs/padded-tail-8x31-region-v1.sir"
)

LOAD = BlockedLayoutPlan("load-a", (1, 1), (1, 32), (1, 1), (1, 0))
COMPUTE = BlockedLayoutPlan("compute-b", (2, 1), (1, 32), (1, 1), (1, 0))
REDUCTION = BlockedLayoutPlan("reduction-c", (1, 2), (1, 32), (1, 1), (0, 1))
STORE = BlockedLayoutPlan("store-d", (2, 2), (1, 32), (1, 1), (0, 1))
STATE = BlockedLayoutPlan("state-e", (4, 2), (1, 32), (1, 1), (0, 1))
CONSUMER = BlockedLayoutPlan("consumer-f", (4, 1), (1, 32), (1, 1), (1, 0))


def identifier(layout):
  """The kernel-body identifier the emitter derives from a layout name."""
  return str(layout.name).replace("-", "_")


def role_plan(**overrides):
  """One plan whose roles are all distinct, so no role hides behind another."""
  fields = {
    "name": "distinct-roles",
    "shape": (8, 8),
    "load": LOAD,
    "compute": COMPUTE,
    "reduction": REDUCTION,
    "store": STORE,
  }
  fields.update(overrides)
  return GluonRegionPlan(**fields)


def kernel_body(source):
  """The emitted body, i.e. everything after the kernel signature."""
  return source.split("\n):\n", 1)[1]


class TensorGluonLoweringTests(unittest.TestCase):
  def test_rendered_kernel_contains_the_shared_region_operations(self):
    generated = render_gluon_kernel(generated_region_source())

    self.assertEqual(generated.shape, (2, 2))
    self.assertEqual(generated.iterations, 2)
    self.assertIn("v0 =", generated.source)
    self.assertIn(
      "gl.expand_dims(gl.sum(v0_reduction_input_1, axis=1)", generated.source
    )
    self.assertIn("v2, _ = gl.broadcast", generated.source)
    self.assertIn("v3 =", generated.source)
    self.assertIn("for _ in range(2):", generated.source)

  def test_rendered_kernel_preserves_sequential_state_update_order(self):
    source = generated_sequential_region_source("  %acc_x = %t0;\n  %acc_y = %acc_x;")
    generated = render_gluon_kernel(source)

    self.assertLess(
      generated.source.index("acc_x = v0"), generated.source.index("acc_y = acc_x")
    )

  def test_rendered_kernel_places_reduction_and_broadcast_axes(self):
    generated = render_gluon_kernel(generated_region_source(axis=0, iterations=1))

    self.assertIn("gl.sum(v0_reduction_input_1, axis=0)", generated.source)
    self.assertIn(
      "gl.expand_dims(gl.sum(v0_reduction_input_1, axis=0), 0), compute_layout",
      generated.source,
    )
    self.assertIn("gl.convert_layout(v1, compute_layout)", generated.source)
    self.assertIn("gl.broadcast(v1, acc_x)", generated.source)

  def test_rendered_kernel_is_python_and_attributes_layout_conversions(self):
    generated = render_gluon_kernel(generated_region_source())

    ast.parse(generated.source)
    self.assertTrue(generated.conversion_source_lines)
    self.assertIn("load_to_compute", generated.conversion_source_lines.values())
    self.assertIn("compute_to_store", generated.conversion_source_lines.values())

  def test_rejects_unsupported_region_operations_before_emission(self):
    source = generated_region_source().replace("tensor sum", "tensor transpose", 1)

    with self.assertRaisesRegex(ValueError, "unsupported tensor operation"):
      render_gluon_kernel(source)

  def test_normalizes_negative_reduction_axes_for_gluon(self):
    source = generated_region_source(axis=1).replace(
      "tensor sum %t0, 1, keep;", "tensor sum %t0, -1, keep;"
    )

    generated = render_gluon_kernel(source)

    self.assertIn("gl.sum(v0_reduction_input_1, axis=1)", generated.source)

  def test_selects_warp_lanes_along_a_nontrivial_reduction_axis(self):
    self.assertEqual(cross_lane_threads_per_warp((2, 2), (0, 1)), (1, 32))
    self.assertEqual(cross_lane_threads_per_warp((2, 2), (0,)), (32, 1))
    with self.assertRaisesRegex(ValueError, "no nontrivial reduction axis"):
      cross_lane_threads_per_warp((1, 1), (0, 1))

  def test_reduction_uses_a_temporary_layout_view(self):
    generated = render_gluon_kernel(generated_region_source())

    self.assertIn(
      "v0_reduction_input_1 = gl.convert_layout(v0, reduction_layout)",
      generated.source,
    )
    self.assertIn("gl.sum(v0_reduction_input_1, axis=1)", generated.source)

  def test_reduction_conversion_does_not_reassign_the_graph_value(self):
    generated = render_gluon_kernel(generated_region_source())

    self.assertNotIn("v0 = gl.convert_layout(v0, reduction_layout)", generated.source)

  def test_orthogonal_reductions_create_independent_input_views(self):
    source = ORTHOGONAL_PROGRAM_PATH.read_text(encoding="utf-8")
    generated = render_gluon_kernel(source)

    self.assertIn(
      "acc_x_reduction_input_0 = gl.convert_layout(acc_x, reduction_layout)",
      generated.source,
    )
    self.assertIn(
      "acc_x_reduction_input_2 = gl.convert_layout(acc_x, reduction_layout)",
      generated.source,
    )
    self.assertIn("gl.sum(acc_x_reduction_input_0, axis=1)", generated.source)
    self.assertIn("gl.sum(acc_x_reduction_input_2, axis=0)", generated.source)

  def test_orthogonal_reductions_preserve_compute_layout_for_broadcast(self):
    source = ORTHOGONAL_PROGRAM_PATH.read_text(encoding="utf-8")
    generated = render_gluon_kernel(source)

    self.assertNotIn(
      "acc_x = gl.convert_layout(acc_x, reduction_layout)", generated.source
    )
    self.assertIn("v1, _ = gl.broadcast(v0, acc_x)", generated.source)
    self.assertIn("v3, _ = gl.broadcast(v2, acc_x)", generated.source)

  def test_reduction_conversion_source_map_keeps_each_operation_identity(self):
    source = ORTHOGONAL_PROGRAM_PATH.read_text(encoding="utf-8")
    generated = render_gluon_kernel(source)

    mapped_lines = [
      generated.source.splitlines()[line_number - 1]
      for line_number, conversion in generated.conversion_source_lines.items()
      if conversion.endswith("_to_reduction")
    ]

    self.assertEqual(len(mapped_lines), 2)
    self.assertTrue(
      all("_reduction_input_" in line for line in mapped_lines), mapped_lines
    )


TAIL_SHAPE = (8, 31)
TAIL_TILE = (8, 32)


def tile_layout(name, order=(1, 0)):
  """One rank-two blocked layout that a single warp can place on any tile."""
  return BlockedLayoutPlan(name, (1, 1), (1, 32), (1, 1), order)


def replay_region(source, *, shape, tile=None, plan=None, input_value=1.0):
  """One region's independent exact reference and the replay of its emission."""
  values = [input_value] * (shape[0] * shape[1])
  proof, expected = certify(source, struct.pack(f"<{len(values)}f", *values), (0, 2))
  generated = render_gluon_kernel(source, plan=plan, tile=tile)
  return proof, expected, replay_gluon_kernel(generated, values)


def replay_words(output):
  """The replayed output as one buffer of f32 words."""
  return b"".join(struct.pack("<f", value) for value in output)


def reference_words(expected):
  """An exact reference buffer as one f32 word per logical lane."""
  return [expected[offset : offset + 4] for offset in range(0, len(expected), 4)]


def lane_values(expected):
  """The distinct f32 values one exact reference buffer states."""
  return [struct.unpack("<f", word)[0] for word in reference_words(expected)]


class TensorGluonPaddedTailTests(unittest.TestCase):
  """A tailed logical extent runs on a declared power-of-two execution tile."""

  def test_a_tail_shape_is_admitted_by_the_emission_entry_point(self):
    generated = render_gluon_kernel(tailed_region_source(), tile=TAIL_TILE)

    ast.parse(generated.source)
    self.assertEqual(generated.shape, TAIL_SHAPE)
    self.assertEqual(generated.tile, TAIL_TILE)
    self.assertIn("gl.load(input_ptr + load_offsets, mask=", generated.source)
    self.assertIn("for _ in range(2):", generated.source)

  def test_the_mask_excludes_padded_lanes_from_a_reduction(self):
    generated = render_gluon_kernel(tailed_region_source(), tile=TAIL_TILE)

    self.assertIn(
      "valid_load_layout = (load_rows[:, None] < 8) & (load_columns[None, :] < 31)",
      generated.source,
    )
    # The reduction reads a view in its own layout, so its mask is stated in
    # that layout and zeroes the padded lanes before the sum sees them.
    self.assertIn(
      "valid_reduction_layout = "
      "(reduction_layout_rows[:, None] < 8) & "
      "(reduction_layout_columns[None, :] < 31)",
      generated.source,
    )
    self.assertIn(
      "v0_reduction_input_1_masked = "
      "gl.where(valid_reduction_layout, v0_reduction_input_1, 0.0)",
      generated.source,
    )
    self.assertIn("gl.sum(v0_reduction_input_1_masked, axis=1)", generated.source)

  def test_padded_lanes_never_appear_in_the_stored_output(self):
    generated = render_gluon_kernel(tailed_region_source(), tile=TAIL_TILE)

    self.assertIn(
      "load_offsets = load_rows[:, None] * 31 + load_columns[None, :]",
      generated.source,
    )
    self.assertIn(
      "store_offsets = store_rows[:, None] * 31 + store_columns[None, :]",
      generated.source,
    )
    self.assertRegex(
      generated.source,
      r"gl\.store\(output_ptr \+ store_offsets, output, mask=valid_\w+\)",
    )

  def test_a_plan_assigned_to_the_tile_is_preserved_for_a_tailed_shape(self):
    sampled = generate_layout_plans(
      TAIL_SHAPE, num_warps=1, seed=0, count=3, tile=TAIL_TILE
    )
    for layout in sampled:
      layout.validate(TAIL_SHAPE, num_warps=1, tile=TAIL_TILE)
    load, compute, reduction, store = (
      tile_layout("tile-load"),
      tile_layout("tile-compute"),
      tile_layout("tile-reduction", order=(0, 1)),
      tile_layout("tile-store", order=(0, 1)),
    )
    plan = GluonRegionPlan(
      "tile-plan",
      TAIL_SHAPE,
      load,
      compute,
      store,
      reduction=reduction,
      tile=TAIL_TILE,
    )

    generated = render_gluon_kernel(tailed_region_source(), plan=plan)
    body = kernel_body(generated.source)

    self.assertEqual(generated.tile, TAIL_TILE)
    self.assertEqual(plan.as_record()["tile"], list(TAIL_TILE))
    for layout in (load, compute, reduction, store):
      with self.subTest(layout=layout.name):
        self.assertIn(identifier(layout), generated.layout_parameters)
        self.assertRegex(body, rf"\b{re.escape(identifier(layout))}\b")

  def test_a_non_power_of_two_shape_without_a_declared_tile_is_refused(self):
    with self.assertRaisesRegex(ValueError, "power-of-two dimensions"):
      render_gluon_kernel(tailed_region_source())

    plan = GluonRegionPlan(
      "untiled",
      TAIL_SHAPE,
      tile_layout("untiled-load"),
      tile_layout("untiled-compute"),
      tile_layout("untiled-store"),
      reduction=tile_layout("untiled-reduction", order=(0, 1)),
    )
    with self.assertRaisesRegex(ValueError, "power-of-two dimensions"):
      render_gluon_kernel(tailed_region_source(), plan=plan)


class TensorGluonPaddedTailEdgeTests(unittest.TestCase):
  """The two tile extremes: no padding at all, and a one-element tail."""

  def test_a_tile_that_equals_the_logical_extent_emits_no_mask(self):
    generated = render_gluon_kernel(generated_region_source(), tile=(2, 2))

    self.assertEqual(generated.tile, (2, 2))
    self.assertNotIn("valid", generated.source)
    self.assertIn("gl.load(input_ptr + load_offsets)", generated.source)
    self.assertIn("gl.store(output_ptr + store_offsets, output)", generated.source)

  def test_a_one_element_tail_keeps_the_tile_and_masks_the_padded_axis(self):
    generated = render_gluon_kernel(tailed_region_source((1, 31)), tile=(1, 32))

    self.assertIn("gl.zeros((1, 32), gl.float32", generated.source)
    self.assertIn(
      "valid_load_layout = (load_rows[:, None] < 1) & (load_columns[None, :] < 31)",
      generated.source,
    )
    self.assertIn("mask=valid_load_layout", generated.source)


def case_layout(shape, *, profile="tensor-v3", row_padding=0, element_offset=0):
  """The physical layout record a tailed region case declares for its buffers.

  The record comes from the case builder itself, so the tests address the layout
  a built case states instead of an arithmetic they re-derive.
  """
  return {
    "shape": list(shape),
    **physical_layout(
      list(shape),
      profile=profile,
      row_padding=row_padding,
      element_offset=element_offset,
    ),
  }


def staged_values(shape):
  """A small mixed input pattern inside the certified [0, 8] domain."""
  return [float((index * 5) % 9) for index in range(shape[0] * shape[1])]


def replay_staged(source, *, shape, tile, layout, envelope=None, values=None):
  """One region's exact reference and the replay of its emission.

  The replay runs against the allocation `test.lib.tensor_layout` stages from
  the same record the emitter addresses, so a kernel that reads dense logical
  row-major storage reads a gap or the wrong element and fails instead of
  agreeing by accident.
  """
  values = staged_values(shape) if values is None else values
  proof, expected = certify(source, struct.pack(f"<{len(values)}f", *values), (0, 8))
  generated = render_gluon_kernel(source, tile=tile, layout=layout)
  return (
    proof,
    expected,
    generated,
    replay_gluon_kernel(generated, values, layout=layout, envelope=envelope),
  )


# The counted arm's logical result for the committed padded-tail case, as
# `test/tensor/campaigns/results/track-c-padded-tail-gluon-evidence.json` records
# it: the case's reference buffer, which the counted run's `actual.bin` matched
# byte for byte. The plan arm has to land on the same digest.
COUNTED_TAIL_OUTPUT_SHA256 = (
  "e523c11eb8c81b455ed69bfecd41e52b7236ca0dcaa0a27e5ee511bb74f81fa2"
)


class TensorGluonPhysicalStrideTests(unittest.TestCase):
  """A region addresses the allocation its own layout record declares."""

  def test_a_non_identity_stride_layout_addresses_the_recorded_elements(self):
    layout = case_layout(TAIL_SHAPE, row_padding=1)
    proof, expected, generated, replay = replay_staged(
      TAIL_PROGRAM_PATH.read_text(encoding="utf-8"),
      shape=TAIL_SHAPE,
      tile=TAIL_TILE,
      layout=layout,
    )

    self.assertTrue(proof["complete_output"])
    self.assertEqual(layout["element_strides"], [33, 1])
    self.assertIn(
      "load_offsets = load_rows[:, None] * 33 + load_columns[None, :]",
      generated.source,
    )
    self.assertIn(
      "store_offsets = store_rows[:, None] * 33 + store_columns[None, :]",
      generated.source,
    )
    self.assertEqual(replay_words(replay.output), expected)

  def test_a_nonzero_element_offset_is_honoured(self):
    layout = case_layout(TAIL_SHAPE, element_offset=1)
    proof, expected, generated, replay = replay_staged(
      TAIL_PROGRAM_PATH.read_text(encoding="utf-8"),
      shape=TAIL_SHAPE,
      tile=TAIL_TILE,
      layout=layout,
    )

    self.assertTrue(proof["complete_output"])
    self.assertEqual(layout["element_offset"], 1)
    self.assertIn(
      "load_offsets = 1 + load_rows[:, None] * 32 + load_columns[None, :]",
      generated.source,
    )
    self.assertIn(
      "store_offsets = 1 + store_rows[:, None] * 32 + store_columns[None, :]",
      generated.source,
    )
    self.assertEqual(replay_words(replay.output), expected)

  def test_a_padded_allocation_keeps_its_guards_and_gaps(self):
    layout = case_layout(TAIL_SHAPE, row_padding=1, element_offset=1)
    envelope = tensor_layout.GuardEnvelope(prefix=3, suffix=5)
    proof, expected, generated, replay = replay_staged(
      TAIL_PROGRAM_PATH.read_text(encoding="utf-8"),
      shape=TAIL_SHAPE,
      tile=TAIL_TILE,
      layout=layout,
      envelope=envelope,
    )

    self.assertTrue(proof["complete_output"])
    self.assertEqual(replay_words(replay.output), expected)
    addressing = tensor_layout.storage_addressing(layout)
    changed = tensor_layout.classify_changed_slots(
      replay_words(replay.initial_physical_output),
      replay_words(replay.physical_output),
      layout,
      element_size=4,
      envelope=envelope,
      profile=addressing.profile,
    )
    # Exactly the logical slots changed: a store into a prefix guard, a suffix
    # guard, or an internal gap would be named here instead.
    self.assertEqual(changed, [f"input-{index}" for index in range(len(expected) // 4)])

  def test_the_gathered_plan_mode_result_equals_the_counted_runs_bytes(self):
    source = TAIL_PROGRAM_PATH.read_text(encoding="utf-8")
    proposals = tensor_input_proposals.derive_proposals(source, domain=(0, 8), seed=0)
    witness = next(
      proposal for proposal in proposals if proposal.name == "sparse-final-lanes"
    )
    values = [float(value) for value in witness.values]
    layout = case_layout(TAIL_SHAPE, element_offset=1)
    proof, expected, generated, replay = replay_staged(
      source,
      shape=TAIL_SHAPE,
      tile=TAIL_TILE,
      layout=layout,
      envelope=tensor_layout.GuardEnvelope(prefix=3, suffix=5),
      values=values,
    )

    self.assertTrue(proof["complete_output"])
    # The local exact reference is the counted arm's recorded result, so the
    # gathered plan-mode bytes are compared against the counted run itself.
    self.assertEqual(hashlib.sha256(expected).hexdigest(), COUNTED_TAIL_OUTPUT_SHA256)
    self.assertEqual(
      hashlib.sha256(replay_words(replay.output)).hexdigest(),
      COUNTED_TAIL_OUTPUT_SHA256,
    )

  def test_a_layout_whose_strides_do_not_fit_its_storage_span_is_refused(self):
    source = TAIL_PROGRAM_PATH.read_text(encoding="utf-8")
    layout = case_layout(TAIL_SHAPE, row_padding=1, element_offset=1)
    layout["storage_span"] -= 1

    with self.assertRaisesRegex(tensor_layout.LayoutError, "exceeds storage span"):
      render_gluon_kernel(source, tile=TAIL_TILE, layout=layout)

    reordered = case_layout(TAIL_SHAPE)
    reordered["axis_order"] = [0, 1]
    with self.assertRaisesRegex(tensor_layout.LayoutError, "fastest to slowest"):
      render_gluon_kernel(source, tile=TAIL_TILE, layout=reordered)


class TensorGluonPhysicalStrideEdgeTests(unittest.TestCase):
  """The dense layout a region had before physical strides existed."""

  def test_an_identity_layout_with_a_zero_offset_emits_the_same_kernel(self):
    source = generated_region_source()
    layout = case_layout((2, 2), profile="tensor-v2")

    self.assertEqual(layout["element_strides"], [2, 1])
    self.assertEqual(layout["element_offset"], 0)
    self.assertEqual(
      render_gluon_kernel(source, layout=layout).source,
      render_gluon_kernel(source).source,
    )


class TensorGluonDeclaredRoleTests(unittest.TestCase):
  """A plan states each role; the emitter honors the role without an override."""

  def setUp(self):
    self.source = ORTHOGONAL_PROGRAM_PATH.read_text(encoding="utf-8")

  def test_declared_reduction_layout_reaches_the_sum_operand(self):
    generated = render_gluon_kernel(self.source, plan=role_plan())
    view = f"acc_x_view_{identifier(REDUCTION)}"

    self.assertIn(
      f"{view} = gl.convert_layout(acc_x, {identifier(REDUCTION)})",
      generated.source,
    )
    self.assertIn(f"gl.sum({view}, axis=1)", generated.source)
    self.assertIn(f"gl.sum({view}, axis=0)", generated.source)

  def test_two_reduction_consumers_share_one_role_view(self):
    generated = render_gluon_kernel(self.source, plan=role_plan())
    view = f"acc_x_view_{identifier(REDUCTION)}"
    conversions = [
      line
      for line in generated.source.splitlines()
      if line.strip() == f"{view} = gl.convert_layout(acc_x, {identifier(REDUCTION)})"
    ]

    self.assertEqual(len(conversions), 1)
    self.assertEqual(generated.source.count(f"gl.sum({view},"), 2)

  def test_consumer_override_takes_precedence_over_the_declared_role(self):
    plan = role_plan(consumer_views={("acc_x", 0): CONSUMER})
    generated = render_gluon_kernel(self.source, plan=plan)

    self.assertIn(
      f"gl.sum(acc_x_view_{identifier(CONSUMER)}, axis=1)", generated.source
    )
    self.assertIn(
      f"gl.sum(acc_x_view_{identifier(REDUCTION)}, axis=0)", generated.source
    )

  def test_every_declared_role_layout_is_used_by_the_body(self):
    generated = render_gluon_kernel(self.source, plan=role_plan(state_layout=STATE))
    body = kernel_body(generated.source)

    for name in generated.layout_parameters:
      with self.subTest(layout=name):
        self.assertRegex(body, rf"\b{re.escape(name)}\b")

  def test_view_for_states_only_explicit_consumer_overrides(self):
    plan = role_plan(consumer_views={("acc_x", 0): CONSUMER})

    self.assertIs(plan.view_for("acc_x", 0), CONSUMER)
    self.assertIsNone(plan.view_for("acc_x", 2))
    self.assertIsNone(plan.view_for("v0", 1))

  def test_a_plan_without_a_reduction_role_keeps_the_sums_in_compute(self):
    generated = render_gluon_kernel(self.source, plan=role_plan(reduction=None))

    self.assertIn("gl.sum(acc_x, axis=1)", generated.source)
    self.assertIn("gl.sum(acc_x, axis=0)", generated.source)
    self.assertNotIn(f"acc_x_view_{identifier(COMPUTE)}", generated.source)


class TensorGluonChainedReductionTests(unittest.TestCase):
  """A reduction masks its own operand's extent, not the region tile.

  Every test here compares the kernel's replayed output bits against the
  independent exact reference of the same program, and every mask it observes
  must be exactly as wide as the operand that mask guards.
  """

  def assert_replays_exactly(self, replay, expected):
    self.assertEqual(replay_words(replay.output), expected)
    self.assertTrue(replay.masks_match_operands(), replay.where_calls)

  def test_a_chained_reduction_sums_its_reduced_operand_once(self):
    # The reviewer's case: the second axis-1 sum reads the [8, 1] result of the
    # first, so a tile-shaped mask counts each of its lanes once per padded
    # column and the recurrence grows by that factor on every lane.
    source = chained_region_source(TAIL_SHAPE, axes=(1, 1))
    proof, expected, replay = replay_region(source, shape=TAIL_SHAPE, tile=TAIL_TILE)

    self.assertTrue(proof["complete_output"])
    self.assertEqual(set(lane_values(expected)), {65.0})
    self.assert_replays_exactly(replay, expected)
    # Only the first reduction reads a padded operand, so the second one carries
    # no mask at all instead of the full tile's.
    self.assertEqual(replay.where_calls, (((8, 32), (8, 32)),))

  def test_a_single_reduction_tail_is_unchanged(self):
    # The committed padded-tail program: one reduction over the padded axis, on
    # the tile the device evidence ran. Its mask is the region's own predicate.
    source = TAIL_PROGRAM_PATH.read_text(encoding="utf-8")
    proof, expected, replay = replay_region(source, shape=TAIL_SHAPE, tile=TAIL_TILE)

    self.assertTrue(proof["complete_output"])
    self.assert_replays_exactly(replay, expected)
    self.assertEqual(replay.where_calls, (((8, 32), (8, 32)),))

  def test_a_chained_reduction_on_the_other_axis_counts_each_lane_once(self):
    # The chain collapses the rows, so the second operand keeps one logical row
    # and its mask must state that extent rather than the region's eight rows.
    source = chained_region_source(TAIL_SHAPE, axes=(0, 0))
    proof, expected, replay = replay_region(source, shape=TAIL_SHAPE, tile=TAIL_TILE)

    self.assertTrue(proof["complete_output"])
    self.assertEqual(set(lane_values(expected)), {19.0})
    self.assert_replays_exactly(replay, expected)
    self.assertEqual(replay.where_calls, (((8, 32), (8, 32)), ((1, 32), (1, 32))))

  def test_a_chain_whose_intermediate_is_later_broadcast_keeps_its_extent(self):
    # The intermediate feeds a broadcast that restates its own extent before the
    # second reduction reads it, so the emitted value has to carry the padded
    # extent of that shape and not the region's.
    source = chained_region_source(TAIL_SHAPE, axes=(0, 0), bridge=True)
    proof, expected, replay = replay_region(source, shape=TAIL_SHAPE, tile=TAIL_TILE)

    self.assertTrue(proof["complete_output"])
    self.assertEqual(set(lane_values(expected)), {19.0})
    self.assert_replays_exactly(replay, expected)
    self.assertEqual(replay.where_calls, (((8, 32), (8, 32)), ((1, 32), (1, 32))))

  def test_two_reductions_sharing_one_layout_get_their_own_masks(self):
    # One layout for every role, so both reductions read the same layout at
    # different shapes: the mask cache is keyed by the operand, not the layout.
    shared = tile_layout("chain-shared")
    plan = GluonRegionPlan(
      "chain-shared-plan",
      TAIL_SHAPE,
      shared,
      shared,
      shared,
      reduction=shared,
      tile=TAIL_TILE,
    )
    source = chained_region_source(TAIL_SHAPE, axes=(1, 1))
    proof, expected, replay = replay_region(source, shape=TAIL_SHAPE, plan=plan)

    self.assertTrue(proof["complete_output"])
    self.assertEqual(set(lane_values(expected)), {65.0})
    self.assert_replays_exactly(replay, expected)
    self.assertEqual(replay.where_calls, (((8, 32), (8, 32)),))

  def test_a_chain_whose_second_operand_needs_no_mask_guards_only_the_padded_one(
    self,
  ):
    # The smallest padded chain: the reduced operand's extent is already its own
    # tile, so no predicate guards it and only the full-shaped operand has one.
    shape, tile = (2, 3), (2, 4)
    source = chained_region_source(shape, axes=(1, 1))
    proof, expected, replay = replay_region(source, shape=shape, tile=tile)

    self.assertTrue(proof["complete_output"])
    self.assertEqual(set(lane_values(expected)), {9.0})
    self.assert_replays_exactly(replay, expected)
    self.assertEqual(replay.where_calls, (((2, 4), (2, 4)),))

  def test_a_chain_without_padding_emits_no_mask(self):
    # The other extreme: every extent is a power of two, so every operand is
    # exactly its own tile and no predicate is needed anywhere.
    shape = (2, 2)
    source = chained_region_source(shape, axes=(1, 1))
    proof, expected, replay = replay_region(source, shape=shape)

    self.assertTrue(proof["complete_output"])
    self.assertEqual(set(lane_values(expected)), {7.0})
    self.assert_replays_exactly(replay, expected)
    self.assertEqual(replay.where_calls, ())


if __name__ == "__main__":
  unittest.main()
