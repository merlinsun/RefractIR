"""Tensor profile version selection and profile-specific limits."""

import ast
import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from test.unit.tensor_support import SYMIRC, SYMIRI, run_typed


def check_cli(source, profile=None):
  with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "case.sir"
    path.write_text(source, encoding="utf-8")
    command = [str(SYMIRI), "--check"]
    if profile is not None:
      command.extend(["--tensor-profile-version", profile])
    command.append(str(path))
    return subprocess.run(
      command, capture_output=True, text=True, check=False, timeout=10
    )


def emit_cli(source, profile=None, layout_manifest=None):
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source_path = root / "case.sir"
    output_path = root / "kernel.py"
    source_path.write_text(source, encoding="utf-8")
    command = [str(SYMIRC), "--target", "triton"]
    if profile is not None:
      command.extend(["--tensor-profile-version", profile])
    if layout_manifest is not None:
      layout_path = root / "layout.json"
      layout_path.write_text(json.dumps(layout_manifest), encoding="utf-8")
      command.extend(["--tensor-layout-manifest", str(layout_path)])
    command.extend(["-o", str(output_path), str(source_path)])
    result = subprocess.run(
      command, capture_output=True, text=True, check=False, timeout=10
    )
    output = output_path.read_text(encoding="utf-8") if output_path.exists() else None
    return result, output


def tensor_source(shape="[512]", element="i32"):
  return f"""\
fun @kernel(%x: tensor {shape} {element}) : tensor {shape} {element} {{
^entry:
  ret %x;
}}
"""


LAYOUT_SOURCE = """\
fun @kernel(%x: tensor [256, 2] f32) : tensor [256, 2] f32 {
^entry:
  ret %x;
}
"""


def layout_manifest(
  *,
  input_strides=(1, 256),
  input_offset=3,
  input_span=515,
  input_order=(0, 1),
  output_strides=(2, 1),
  output_span=512,
  output_order=(1, 0),
):
  return {
    "buffers": [
      {
        "name": "x",
        "element_strides": list(input_strides),
        "element_offset": input_offset,
        "storage_span": input_span,
        "axis_order": list(input_order),
      },
      {
        "name": "return",
        "element_strides": list(output_strides),
        "element_offset": 0,
        "storage_span": output_span,
        "axis_order": list(output_order),
      },
    ]
  }


COUNTED_LOOP_SOURCE = """\
fun @kernel(%x: tensor [2] i32) : tensor [2] i32 {
  let mut %i: i32 = undef;
  let mut %out: tensor [2] i32 = undef;
^entry:
  %i = 0;
  br ^header;
^header:
  br %i < 2, ^body, ^exit;
^body:
  %i = %i + 1;
  br ^header;
^exit:
  %out = tensor add %x, %i;
  ret %out;
}
"""


NESTED_LOOP_SOURCE = """\
fun @kernel(%x: tensor [2] i32) : tensor [2] i32 {
  let mut %outer: i32 = undef;
  let mut %inner: i32 = undef;
  let mut %out: tensor [2] i32 = undef;
^entry:
  %outer = 0;
  %out = tensor mul %x, 0;
  br ^outer_header;
^outer_header:
  br %outer < 2, ^outer_body, ^exit;
^outer_body:
  %inner = 0;
  br ^inner_header;
^inner_header:
  br %inner < 2, ^inner_body, ^outer_latch;
^inner_body:
  %out = tensor add %out, %x;
  %inner = %inner + 1;
  br ^inner_header;
^outer_latch:
  %outer = %outer + 1;
  br ^outer_header;
^exit:
  ret %out;
}
"""


NESTED_LOOP_WITHOUT_INNER_RESET = NESTED_LOOP_SOURCE.replace(
  "  %inner = 0;\n", "", 1
).replace(
  "  %out = tensor mul %x, 0;\n", "  %inner = 0;\n  %out = tensor mul %x, 0;\n", 1
)

LOOP_INPUT_SOURCE = """\
fun @kernel(%x: tensor [2] i32) : tensor [2] i32 {
  let mut %i: i32 = undef;
  let mut %out: tensor [2] i32 = undef;
^entry:
  %out = tensor mul %x, 0;
  %i = 0;
  br ^header;
^header:
  br %i < 2, ^body, ^exit;
^body:
  %out = tensor add %out, %x;
  %i = %i + 1;
  br ^header;
^exit:
  ret %out;
}
"""


class TensorProfileVersionTests(unittest.TestCase):
  def setUp(self):
    self.assertTrue(SYMIRI.is_file(), f"symiri executable not found: {SYMIRI}")
    self.assertTrue(SYMIRC.is_file(), f"symirc executable not found: {SYMIRC}")

  def test_v2_accepts_512_element_tensor(self):
    result = check_cli(tensor_source(), "v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_v3_accepts_non_power_of_two_shape_without_changing_v2(self):
    source = tensor_source("[17]", "i32")

    v2_result = check_cli(source, "v2")
    v3_result = check_cli(source, "v3")

    self.assertNotEqual(v2_result.returncode, 0)
    self.assertIn("dimensions must be powers of two", v2_result.stderr)
    self.assertEqual(v3_result.returncode, 0, v3_result.stdout + v3_result.stderr)

  def test_v3_rejects_tile_volume_above_512(self):
    result = check_cli(tensor_source("[65, 5]", "i32"), "v3")

    self.assertNotEqual(result.returncode, 0)

  def test_v4_remainder_vector_limits(self):
    def source(length):
      return f"""\
fun @kernel(%x: tensor [{length}] f32, %y: tensor [{length}] f32) : tensor [{length}] f32 {{
  let mut %out: tensor [{length}] f32 = undef;
^entry:
  %out = tensor rem %x, %y;
  ret %out;
}}
"""

    for length in (1, 512):
      result = check_cli(source(length), "v4")
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    too_large = check_cli(source(1024), "v4")
    self.assertNotEqual(too_large.returncode, 0)
    self.assertIn("tensor exceeds 512 elements", too_large.stderr)

  def test_v3_emits_masked_strided_input_load_with_zero_fallback(self):
    source = """\
fun @kernel(%x: tensor [17] i32) : tensor [17] i32 {
  let mut %out: tensor [17] i32 = undef;
^entry:
  %out = tensor mul %x, 2;
  ret %out;
}
"""
    manifest = {
      "buffers": [
        {
          "name": "x",
          "role": "input",
          "element_strides": [2],
          "element_offset": 1,
          "storage_span": 34,
          "axis_order": [0],
        },
      ]
    }

    result, output = emit_cli(source, "v3", manifest)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("tl.arange(0, 32)", output)
    self.assertRegex(output, r"tl\.load\(x_ptr \+ .*mask=.*other=0\)")

  def test_v3_sum_uses_logical_validity_for_padded_lanes(self):
    source = """\
fun @kernel(%x: tensor [17] f32) : f32 {
  let mut %out: f32 = undef;
^entry:
  %out = tensor sum %x, 0, drop;
  ret %out;
}
"""

    result, output = emit_cli(source, "v3")

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertRegex(
      output, r"tl\.sum\(tl\.where\(.*?, .*?, 0\.0\), axis=0, keep_dims=False\)"
    )

  def test_v3_output_store_masks_tail_and_uses_output_stride(self):
    source = """\
fun @kernel(%x: tensor [17] i32) : tensor [17] i32 {
  let mut %out: tensor [17] i32 = undef;
^entry:
  %out = tensor add %x, 1;
  ret %out;
}
"""
    manifest = {
      "buffers": [
        {
          "name": "return",
          "role": "output",
          "element_strides": [2],
          "element_offset": 1,
          "storage_span": 34,
          "axis_order": [0],
        },
      ]
    }

    result, output = emit_cli(source, "v3", manifest)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertRegex(output, r"tl\.store\(out_ptr \+ .*?, .*?, mask=.*\)")
    self.assertIn("_sir_out_d0 * 2", output)

  def test_v3_rejects_tensor_division_without_mask_semantics(self):
    source = """\
fun @kernel(%x: tensor [17] f32) : tensor [17] f32 {
  let mut %out: tensor [17] f32 = undef;
^entry:
  %out = tensor div %x, 2.0;
  ret %out;
}
"""

    result = check_cli(source, "v3")

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("unsupported tensor operation", result.stderr)

  def test_v3_admits_tensor_min_and_the_backend_masks_it(self):
    source = """\
fun @kernel(%x: tensor [17] i32) : tensor [17] i32 {
  let mut %out: tensor [17] i32 = undef;
^entry:
  %out = tensor min %x, %x;
  ret %out;
}
"""

    result = check_cli(source, "v3")

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_v3_rejects_tensor_exp_without_mask_semantics(self):
    source = """\
fun @kernel(%x: tensor [17] f32) : tensor [17] f32 {
  let mut %out: tensor [17] f32 = undef;
^entry:
  %out = tensor exp %x;
  ret %out;
}
"""

    result = check_cli(source, "v3")

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("unsupported tensor operation", result.stderr)

  def test_v3_admits_an_extrema_reduction_and_masks_it(self):
    source = """\
fun @kernel(%x: tensor [17] i32) : i32 {
  let mut %out: i32 = undef;
^entry:
  %out = tensor rmin %x, 0, drop;
  ret %out;
}
"""

    result = check_cli(source, "v3")

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_v3_rejects_tensor_bitwise_ops_without_mask_semantics(self):
    source = """\
fun @kernel(%x: tensor [17] i32) : tensor [17] i32 {
  let mut %out: tensor [17] i32 = undef;
^entry:
  %out = tensor and %x, %x;
  ret %out;
}
"""

    result = check_cli(source, "v3")

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("unsupported tensor operation", result.stderr)

  def test_v3_rejects_iota_until_inactive_coordinate_semantics_are_added(self):
    source = """\
fun @kernel() : tensor [17] i32 {
  let mut %out: tensor [17] i32 = undef;
^entry:
  %out = tensor iota [17], 0;
  ret %out;
}
"""

    result = check_cli(source, "v3")

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("unsupported tensor operation", result.stderr)

  def test_v3_rejects_extract_until_validity_mapping_is_added(self):
    source = """\
fun @kernel(%x: tensor [17] i32) : i32 {
  let mut %out: i32 = undef;
^entry:
  %out = tensor extract %x, [0];
  ret %out;
}
"""

    result = check_cli(source, "v3")

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("unsupported tensor operation", result.stderr)

  def test_v2_accepts_512_element_rank_two_tensor(self):
    result = check_cli(tensor_source("[256, 2]", "f32"), "v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_v1_default_keeps_256_element_limit(self):
    result = check_cli(tensor_source())
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("256 elements", result.stderr)

  def test_v2_rejects_more_than_512_elements(self):
    result = check_cli(tensor_source("[1024]"), "v2")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("512 elements", result.stderr)

  def test_unknown_profile_version_has_specific_error(self):
    result = check_cli(tensor_source("[2]"), "v9")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("unknown tensor profile version", result.stderr)

  def test_v2_triton_emitter_uses_the_full_512_element_range(self):
    result, output = emit_cli(tensor_source(), "v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("tl.arange(0, 512)", output)

  def test_v2_accepts_a_canonical_counted_loop(self):
    result = check_cli(COUNTED_LOOP_SOURCE, "v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_post_loop_induction_variable_equals_bound(self):
    result, output = run_typed(
      COUNTED_LOOP_SOURCE, struct.pack("<2i", 10, -3), tensor_profile="v2"
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(output, struct.pack("<2i", 12, -1))

  def test_nested_loop_resets_inner_counter_on_each_entry(self):
    result = check_cli(NESTED_LOOP_SOURCE, "v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_nested_loop_without_inner_preheader_reset_is_rejected(self):
    result = check_cli(NESTED_LOOP_WITHOUT_INNER_RESET, "v2")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("preheader must reset induction variable", result.stderr)

  def test_loop_increment_outside_latch_is_rejected(self):
    source = """\
fun @kernel(%x: tensor [2] i32) : tensor [2] i32 {
  let mut %i: i32 = undef;
^entry:
  %i = 0;
  br ^header;
^header:
  br %i < 2, ^body, ^exit;
^body:
  %i = %i + 2;
  br ^latch;
^latch:
  br ^header;
^exit:
  ret %x;
}
"""

    result = check_cli(source, "v2")

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("latch must increment induction variable by one", result.stderr)

  def test_input_named_return_has_distinct_input_and_output_layouts(self):
    source = """\
fun @kernel(%return: tensor [2] i32) : tensor [2, 2] i32 {
  let mut %out: tensor [2, 2] i32 = undef;
^entry:
  %out = tensor broadcast %return, [2, 2];
  ret %out;
}
"""
    manifest = {
      "buffers": [
        {
          "name": "return",
          "role": "input",
          "element_strides": [1],
          "element_offset": 0,
          "storage_span": 2,
          "axis_order": [0],
        },
        {
          "name": "return",
          "role": "output",
          "element_strides": [2, 1],
          "element_offset": 0,
          "storage_span": 4,
          "axis_order": [1, 0],
        },
      ]
    }

    result, output = emit_cli(source, "v2", manifest)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("_sir_return_offsets = _sir_return_d0", output)
    self.assertIn("_sir_out_offsets = _sir_out_d0 * 2 + _sir_out_d1", output)

  def test_nested_loop_body_execution_cap_is_enforced(self):
    source = NESTED_LOOP_SOURCE.replace("%outer < 2", "%outer < 3")
    result = check_cli(source, "v2")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("eight total body executions", result.stderr)

  def test_triton_emits_counted_loop_as_tl_range(self):
    result, output = emit_cli(COUNTED_LOOP_SOURCE, "v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("for _sir_i in tl.range(0, 2):", output)
    self.assertIn("        pass", output)
    self.assertNotIn("tl.static_range", output)

  def test_triton_restores_induction_value_after_range(self):
    result, output = emit_cli(COUNTED_LOOP_SOURCE, "v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("_sir_i = tl.full((), 2, tl.int32)", output)

  def test_triton_emits_nested_counted_ranges(self):
    result, output = emit_cli(NESTED_LOOP_SOURCE, "v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(output.count(" in tl.range("), 2)

  def test_nested_preheader_reset_is_inside_outer_range(self):
    result, output = emit_cli(NESTED_LOOP_SOURCE, "v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    outer = output.index("for _sir_outer in tl.range(")
    reset = output.index("_sir_inner = tl.full((), 0, tl.int32)")
    inner = output.index("for _sir_inner in tl.range(")
    self.assertLess(outer, reset)
    self.assertLess(reset, inner)

  def test_tensor_accumulator_update_is_nested_in_loop_body(self):
    result, output = emit_cli(NESTED_LOOP_SOURCE, "v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    accumulator = output.index("_sir_out = (_sir_out + _sir_input_load_tmp")
    inner = output.index("for _sir_inner in tl.range(")
    self.assertLess(inner, accumulator)
    outer_exit = output.index("_sir_outer = tl.full((), 2, tl.int32)")
    self.assertLess(accumulator, outer_exit)
    accumulator_line = next(
      line
      for line in output.splitlines()
      if "_sir_out = (_sir_out + _sir_input_load_tmp" in line
    )
    self.assertTrue(accumulator_line.startswith("            "))

  def test_triton_input_layout_uses_recorded_strides_and_offset(self):
    result, output = emit_cli(LAYOUT_SOURCE, "v2", layout_manifest())
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("_sir_x_offsets = 3 + _sir_x_d0 + _sir_x_d1 * 256", output)
    self.assertIn("tl.load(x_ptr + _sir_x_offsets)", output)

  def test_triton_output_layout_uses_recorded_strides(self):
    manifest = layout_manifest(
      input_strides=(2, 1),
      input_offset=0,
      input_span=512,
      input_order=(1, 0),
      output_strides=(1, 256),
      output_span=512,
      output_order=(0, 1),
    )
    result, output = emit_cli(LAYOUT_SOURCE, "v2", manifest)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("_sir_out_d0 + _sir_out_d1 * 256", output)

  def test_overlapping_tensor_buffer_strides_are_rejected(self):
    manifest = layout_manifest(input_strides=(1, 1), input_order=(0, 1))
    result, output = emit_cli(LAYOUT_SOURCE, "v2", manifest)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("overlap", result.stderr.lower())
    self.assertIsNone(output)

  def test_axis_order_must_agree_with_strides(self):
    manifest = layout_manifest(input_order=(1, 0))
    result, output = emit_cli(LAYOUT_SOURCE, "v2", manifest)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("axis order", result.stderr.lower())
    self.assertIsNone(output)

  def test_tensor_buffer_span_must_cover_last_address(self):
    manifest = layout_manifest(input_span=514)
    result, output = emit_cli(LAYOUT_SOURCE, "v2", manifest)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("storage span", result.stderr.lower())
    self.assertIsNone(output)

  def test_parameter_load_is_emitted_inside_the_loop_body(self):
    result, output = emit_cli(LOOP_INPUT_SOURCE, "v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    load_line = [line for line in output.splitlines() if "tl.load(x_ptr" in line][-1]
    self.assertTrue(load_line.startswith("        "), load_line)

  def test_each_dynamic_parameter_use_gets_a_load_site(self):
    result, output = emit_cli(LOOP_INPUT_SOURCE, "v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(output.count("tl.load(x_ptr"), 2)

  def test_nested_loop_parameter_load_is_inside_inner_loop(self):
    result, output = emit_cli(NESTED_LOOP_SOURCE, "v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    load_line = [line for line in output.splitlines() if "tl.load(x_ptr" in line][-1]
    self.assertTrue(load_line.startswith("            "), load_line)

  def test_strided_parameter_load_uses_layout_inside_loop(self):
    manifest = layout_manifest(
      input_strides=(1, 2),
      input_offset=3,
      input_span=7,
      input_order=(0, 1),
      output_strides=(2, 1),
      output_span=4,
      output_order=(1, 0),
    )
    source = LOOP_INPUT_SOURCE.replace("tensor [2] i32", "tensor [2, 2] i32")
    result, output = emit_cli(source, "v2", manifest)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    load_line = [line for line in output.splitlines() if "tl.load(x_ptr" in line][-1]
    self.assertTrue(load_line.startswith("        "), load_line)

  def test_scalar_parameter_read_is_emitted_at_its_loop_use(self):
    source = LOOP_INPUT_SOURCE.replace(
      "fun @kernel(%x: tensor [2] i32) : tensor [2] i32 {",
      "fun @kernel(%x: tensor [2] i32, %bias: i32) : tensor [2] i32 {",
    ).replace("%out = tensor add %out, %x;", "%out = tensor add %out, %bias;")
    result, output = emit_cli(source, "v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    load_line = next(
      line for line in output.splitlines() if "tl.load(bias_ptr)" in line
    )
    self.assertTrue(load_line.startswith("        "), load_line)

  def test_direct_tensor_parameter_copy_emits_a_load(self):
    source = """\
fun @kernel(%x: tensor [2, 4] f32) : tensor [2, 4] f32 {
  let mut %out: tensor [2, 4] f32 = undef;
^entry:
  %out = %x;
  ret %out;
}
"""
    result, output = emit_cli(source)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("tl.load(x_ptr + _sir_x_offsets)", output)

  def test_direct_scalar_parameter_copy_emits_a_load(self):
    source = """\
fun @kernel(%x: f32, %v: tensor [2] f32) : f32 {
  let mut %dummy: tensor [2] f32 = undef;
  let mut %out: f32 = undef;
^entry:
  %dummy = tensor exp %v;
  %out = %x;
  ret %out;
}
"""
    result, output = emit_cli(source)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("tl.load(x_ptr)", output)

  def test_direct_integer_parameter_copy_emits_a_load(self):
    source = """\
fun @kernel(%x: i32, %v: tensor [2] f32) : i32 {
  let mut %dummy: tensor [2] f32 = undef;
  let mut %out: i32 = undef;
^entry:
  %dummy = tensor exp %v;
  %out = %x;
  ret %out;
}
"""
    result, output = emit_cli(source)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertIn("tl.load(x_ptr)", output)

  def test_direct_parameter_copy_inside_loop_loads_in_the_body(self):
    source = """\
fun @kernel(%x: tensor [2, 4] f32) : tensor [2, 4] f32 {
  let mut %i: i32 = undef;
  let mut %out: tensor [2, 4] f32 = undef;
^entry:
  %out = %x;
  %i = 0;
  br ^header;
^header:
  br %i < 2, ^body, ^exit;
^body:
  %out = %x;
  %i = %i + 1;
  br ^header;
^exit:
  ret %out;
}
"""
    result, output = emit_cli(source, "v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    load_lines = [line for line in output.splitlines() if "tl.load(x_ptr" in line]
    self.assertEqual(len(load_lines), 2)
    self.assertTrue(load_lines[0].startswith("    "), load_lines[0])
    self.assertTrue(load_lines[1].startswith("        "), load_lines[1])

  def test_each_plain_parameter_use_gets_its_own_load_site(self):
    source = """\
fun @kernel(%x: tensor [2, 4] f32) : tensor [2, 4] f32 {
  let mut %a: tensor [2, 4] f32 = undef;
  let mut %b: tensor [2, 4] f32 = undef;
  let mut %out: tensor [2, 4] f32 = undef;
^entry:
  %a = %x;
  %b = %x;
  %out = tensor add %a, %b;
  ret %out;
}
"""
    result, output = emit_cli(source)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    ast.parse(output)
    self.assertEqual(output.count("tl.load(x_ptr + _sir_x_offsets)"), 2)


if __name__ == "__main__":
  unittest.main()
