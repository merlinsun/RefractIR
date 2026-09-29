"""Physical buffer layout validation and logical value staging."""

import unittest

try:
  from test.lib import tensor_layout
except ImportError:
  tensor_layout = None


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
