"""Physical buffer layout validation and logical value staging."""

import unittest

try:
  from test.lib import tensor_layout
except ImportError:
  tensor_layout = None


PAD_A = bytes.fromhex("3f800001")
PAD_B = bytes.fromhex("3f800002")


class TensorLayoutGuardTests(unittest.TestCase):
  """Allocated guards and internal gaps around one padded tensor."""

  def setUp(self):
    self.assertIsNotNone(tensor_layout, "missing tensor layout module")
    self.layout = {
      "shape": [2, 4],
      "element_strides": [6, 1],
      "element_offset": 1,
      "storage_span": 11,
      "axis_order": [1, 0],
    }
    self.envelope = tensor_layout.GuardEnvelope(prefix=2, suffix=3)
    self.values = bytes(range(32))

  def test_guarded_staging_preserves_the_logical_reference(self):
    staged = tensor_layout.stage_envelope(
      self.values, self.layout, element_size=4, envelope=self.envelope, padding=PAD_A
    )

    self.assertEqual(
      len(staged), (2 + 11 + 3) * 4, "the envelope must cover both guards"
    )
    self.assertEqual(
      tensor_layout.gather_envelope(
        staged, self.layout, element_size=4, envelope=self.envelope
      ),
      self.values,
    )
    self.assertEqual(staged[:8], PAD_A * 2)
    self.assertEqual(staged[-12:], PAD_A * 3)

  def test_prefix_and_suffix_guard_writes_are_detected(self):
    staged = tensor_layout.stage_envelope(
      self.values, self.layout, element_size=4, envelope=self.envelope, padding=PAD_A
    )
    overwritten = bytearray(staged)
    overwritten[0:4] = b"\xde\xad\xbe\xef"
    overwritten[-4:] = b"\xde\xad\xbe\xef"

    changed = tensor_layout.classify_changed_slots(
      staged, bytes(overwritten), self.layout, element_size=4, envelope=self.envelope
    )

    self.assertEqual(changed, ["prefix-guard-0", "suffix-guard-2"])

  def test_internal_gap_and_input_writes_are_detected(self):
    staged = tensor_layout.stage_envelope(
      self.values, self.layout, element_size=4, envelope=self.envelope, padding=PAD_A
    )
    overwritten = bytearray(staged)
    overwritten[(2 + 5) * 4 : (2 + 6) * 4] = b"\xde\xad\xbe\xef"
    overwritten[(2 + 1) * 4 : (2 + 2) * 4] = b"\xde\xad\xbe\xef"

    changed = tensor_layout.classify_changed_slots(
      staged, bytes(overwritten), self.layout, element_size=4, envelope=self.envelope
    )

    self.assertEqual(changed, ["input-0", "gap-5"])
    self.assertEqual(
      tensor_layout.changed_canary_slots(
        staged, bytes(overwritten), self.layout, element_size=4, envelope=self.envelope
      ),
      ["gap-5"],
    )
    self.assertEqual(
      tensor_layout.changed_canary_slots(
        staged, staged, self.layout, element_size=4, envelope=self.envelope
      ),
      [],
    )

  def test_two_padding_patterns_leave_the_logical_output_unchanged(self):
    first = tensor_layout.stage_envelope(
      self.values, self.layout, element_size=4, envelope=self.envelope, padding=PAD_A
    )
    second = tensor_layout.stage_envelope(
      self.values, self.layout, element_size=4, envelope=self.envelope, padding=PAD_B
    )

    self.assertNotEqual(first, second)
    self.assertEqual(
      tensor_layout.gather_envelope(
        first, self.layout, element_size=4, envelope=self.envelope
      ),
      tensor_layout.gather_envelope(
        second, self.layout, element_size=4, envelope=self.envelope
      ),
    )
    self.assertTrue(
      tensor_layout.changed_canary_slots(
        first, second, self.layout, element_size=4, envelope=self.envelope
      ),
      "the padding change must be visible outside the logical mapping",
    )

  def test_padding_dependent_read_is_detected_and_classified_as_a_control(self):
    """A read of an inactive padded lane is a control, not a compiler finding."""
    logical = bytes(range(32))
    padded_read = bytearray(logical)
    padded_read[0:4] = PAD_A

    comparison = tensor_layout.classify_physical_difference(
      expected=logical,
      observed=bytes(padded_read),
      control="dropped-mask-reads-padding",
    )

    self.assertEqual(comparison["status"], "different")
    self.assertEqual(comparison["classification"], "control_variant")
    self.assertTrue(comparison["control"])
    self.assertEqual(comparison["logical_mismatch_lanes"], [0])


class TensorLayoutGuardEdgeTests(unittest.TestCase):
  def setUp(self):
    self.assertIsNotNone(tensor_layout, "missing tensor layout module")

  def test_zero_guards_and_no_internal_gap_reduce_to_the_plain_buffer(self):
    layout = {
      "shape": [2, 4],
      "element_strides": [4, 1],
      "element_offset": 0,
      "storage_span": 8,
      "axis_order": [1, 0],
    }
    values = bytes(range(32))
    envelope = tensor_layout.GuardEnvelope()

    staged = tensor_layout.stage_envelope(
      values, layout, element_size=4, envelope=envelope, padding=PAD_A
    )

    self.assertEqual(
      staged,
      tensor_layout.stage_buffer(values, layout, element_size=4, gap_bytes=PAD_A),
    )
    self.assertEqual(
      tensor_layout.classify_changed_slots(
        staged, staged, layout, element_size=4, envelope=envelope
      ),
      [],
    )


class TensorLayoutTests(unittest.TestCase):
  def setUp(self):
    self.assertIsNotNone(tensor_layout, "missing tensor layout module")

  def test_same_logical_input_across_layouts(self):
    values = bytes(range(32))
    row_major = {
      "shape": [2, 4],
      "element_strides": [4, 1],
      "element_offset": 0,
      "storage_span": 8,
      "axis_order": [1, 0],
    }
    column_major = {
      "shape": [2, 4],
      "element_strides": [1, 2],
      "element_offset": 0,
      "storage_span": 8,
      "axis_order": [0, 1],
    }

    first = tensor_layout.stage_buffer(
      values, row_major, element_size=4, gap_bytes=b"\xaa" * 4
    )
    second = tensor_layout.stage_buffer(
      values, column_major, element_size=4, gap_bytes=b"\xaa" * 4
    )

    self.assertEqual(
      tensor_layout.gather_buffer(first, row_major, element_size=4), values
    )
    self.assertEqual(
      tensor_layout.gather_buffer(second, column_major, element_size=4), values
    )
    self.assertNotEqual(first, second)

  def test_padded_stride_mapping(self):
    layout = {
      "shape": [2, 4],
      "element_strides": [6, 1],
      "element_offset": 1,
      "storage_span": 11,
      "axis_order": [1, 0],
    }

    addresses = tensor_layout.logical_addresses(layout)

    self.assertEqual(addresses, [1, 2, 3, 4, 7, 8, 9, 10])

  def test_overlapping_strides_are_rejected(self):
    layout = {
      "shape": [2, 4],
      "element_strides": [1, 1],
      "element_offset": 0,
      "storage_span": 5,
      "axis_order": [0, 1],
    }

    with self.assertRaisesRegex(tensor_layout.LayoutError, "overlap"):
      tensor_layout.logical_addresses(layout)

  def test_output_gather_uses_logical_order_and_preserves_canaries(self):
    layout = {
      "shape": [2, 4],
      "element_strides": [6, 1],
      "element_offset": 1,
      "storage_span": 11,
      "axis_order": [1, 0],
    }
    canary = b"\xcd" * 4
    physical = bytearray(canary * layout["storage_span"])
    logical = b"".join(value.to_bytes(4, "little") for value in range(8))
    staged = tensor_layout.scatter_buffer(logical, physical, layout, element_size=4)

    self.assertEqual(
      tensor_layout.gather_buffer(staged, layout, element_size=4), logical
    )
    self.assertEqual(
      tensor_layout.changed_canary_slots(physical, staged, layout, element_size=4), []
    )

  def test_out_of_span_layout_is_rejected(self):
    layout = {
      "shape": [2, 4],
      "element_strides": [6, 1],
      "element_offset": 1,
      "storage_span": 10,
      "axis_order": [1, 0],
    }

    with self.assertRaisesRegex(tensor_layout.LayoutError, "storage span"):
      tensor_layout.logical_addresses(layout)

  def test_invalid_axis_order_and_nonpositive_stride_are_rejected(self):
    layout = {
      "shape": [2, 4],
      "element_strides": [4, 1],
      "element_offset": 0,
      "storage_span": 8,
      "axis_order": [0, 1],
    }
    with self.assertRaisesRegex(tensor_layout.LayoutError, "axis order"):
      tensor_layout.logical_addresses(layout)
    layout["axis_order"] = [1, 0]
    layout["element_strides"] = [4, 0]
    with self.assertRaisesRegex(tensor_layout.LayoutError, "positive"):
      tensor_layout.logical_addresses(layout)

  def test_v3_accepts_logical_non_power_of_two_shape(self):
    layout = {
      "shape": [17],
      "element_strides": [1],
      "element_offset": 0,
      "storage_span": 17,
      "axis_order": [0],
    }

    self.assertEqual(
      tensor_layout.logical_addresses(layout, profile="v3"), list(range(17))
    )

  def test_v2_still_rejects_logical_non_power_of_two_shape(self):
    layout = {
      "shape": [17],
      "element_strides": [1],
      "element_offset": 0,
      "storage_span": 17,
      "axis_order": [0],
    }

    with self.assertRaisesRegex(tensor_layout.LayoutError, "powers of two"):
      tensor_layout.logical_addresses(layout, profile="v2")

  def test_v3_boundary_pairings_have_512_element_execution_tiles(self):
    for length, other in ((17, 16), (31, 16), (33, 8), (63, 8), (65, 4)):
      for shape in ([length, other], [other, length]):
        layout = {
          "shape": list(shape),
          "element_strides": [shape[1], 1],
          "element_offset": 0,
          "storage_span": shape[0] * shape[1],
          "axis_order": [1, 0],
        }
        addresses = tensor_layout.logical_addresses(layout, profile="v3")
        self.assertEqual(len(addresses), length * other)

  def test_v3_rejects_padded_tile_larger_than_512(self):
    layout = {
      "shape": [65, 5],
      "element_strides": [5, 1],
      "element_offset": 0,
      "storage_span": 325,
      "axis_order": [1, 0],
    }

    with self.assertRaisesRegex(tensor_layout.LayoutError, "padded tile volume"):
      tensor_layout.logical_addresses(layout, profile="v3")

  def test_v3_staging_uses_logical_addresses_and_keeps_gap_canaries(self):
    layout = {
      "shape": [17],
      "element_strides": [2],
      "element_offset": 1,
      "storage_span": 34,
      "axis_order": [0],
    }
    logical = b"".join(value.to_bytes(4, "little") for value in range(17))
    canary = b"\xaa" * 4

    staged = tensor_layout.stage_buffer(
      logical, layout, element_size=4, gap_bytes=canary, profile="v3"
    )

    self.assertEqual(
      tensor_layout.gather_buffer(staged, layout, element_size=4, profile="v3"),
      logical,
    )
    self.assertEqual(
      tensor_layout.changed_canary_slots(
        canary * 34, staged, layout, element_size=4, profile="v3"
      ),
      [],
    )


if __name__ == "__main__":
  unittest.main()
