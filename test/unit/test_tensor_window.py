"""Contract tests for the tensor window operation's surface syntax."""

import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from test.unit.tensor_support import SYMIRI, check_source, run_typed, solve_source

ROOT = Path(__file__).resolve().parents[2]
WINDOW_API = Path(
  os.environ.get("REFRACTIR_TENSOR_WINDOW_API", ROOT / "build/bin/tensor_window_api")
)

WINDOW_SOURCE = """\
fun @kernel(%src: tensor [8, 2] f32, %i: i32) : tensor [4, 2] f32 {
  let mut %w: tensor [4, 2] f32 = undef;
^entry:
  %w = tensor window %src, %i, 0.0, [4, 1, 8];
  ret %w;
}
"""


def run_window_api(mode, source):
  with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "case.sir"
    path.write_text(source, encoding="utf-8")
    return subprocess.run(
      [str(WINDOW_API), mode, str(path)],
      capture_output=True,
      text=True,
      check=False,
      timeout=10,
    )


def fill_of(printed):
  """The fill token between the index operand and the attribute."""
  body = printed.split("tensor window ", 1)[1]
  operands = body.split(", [", 1)[0]
  return operands.rsplit(", ", 1)[1].rstrip(";").strip()


@unittest.skipUnless(WINDOW_API.is_file(), "tensor window API not built")
class TensorWindowSyntaxTests(unittest.TestCase):
  def test_window_parses(self):
    result = run_window_api("parse", WINDOW_SOURCE)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_window_prints_round_trip(self):
    result = run_window_api("print", WINDOW_SOURCE)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("tensor window %src, %i, ", result.stdout)
    self.assertIn(", [4, 1, 8];", result.stdout)
    self.assertEqual(float(fill_of(result.stdout)), 0.0)
    again = run_window_api("parse", result.stdout)
    self.assertEqual(again.returncode, 0, again.stdout + again.stderr)

  def test_window_dump_names_the_operator(self):
    result = run_window_api("dump", WINDOW_SOURCE)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("window", result.stdout)
    self.assertIn("4, 1, 8", result.stdout)

  def test_window_rejects_a_short_attribute(self):
    source = WINDOW_SOURCE.replace("[4, 1, 8]", "[3, 1]")
    result = run_window_api("parse", source)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("tensor-window", result.stderr)

  def test_window_rejects_a_trailing_operand(self):
    source = WINDOW_SOURCE.replace("0.0, [4, 1, 8]", "0.0, 1.0, [4, 1, 8]")
    result = run_window_api("parse", source)
    self.assertNotEqual(result.returncode, 0)
    self.assertNotIn("unknown tensor operator", result.stderr)

  def test_window_print_of_a_float_fill_is_bit_exact(self):
    literal = "0.30000001192092896"
    result = run_window_api(
      "print", WINDOW_SOURCE.replace("0.0, [4, 1, 8]", f"{literal}, [4, 1, 8]")
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    printed = fill_of(result.stdout)
    self.assertEqual(float(printed), float(literal))


BROADCAST_SOURCE = """\
fun @kernel(%scalar: f32, %i: i32) : tensor [4, 2] f32 {
  let mut %buf: tensor [8, 2] f32 = undef;
  let mut %w: tensor [4, 2] f32 = undef;
^entry:
  %buf = tensor broadcast %scalar, [8, 2];
  %w = tensor window %buf, %i, 0.0, [4, 1, 8];
  ret %w;
}
"""

FLOAT_INDEX_SOURCE = """\
fun @kernel(%src: tensor [8, 2] f32, %f: f32) : tensor [4, 2] f32 {
  let mut %w: tensor [4, 2] f32 = undef;
^entry:
  %w = tensor window %src, %f, 0.0, [4, 1, 8];
  ret %w;
}
"""

NON_LITERAL_FILL_SOURCE = """\
fun @kernel(%src: tensor [8, 2] f32, %i: i32) : tensor [4, 2] f32 {
  let mut %w: tensor [4, 2] f32 = undef;
^entry:
  %w = tensor window %src, %i, %i, [4, 1, 8];
  ret %w;
}
"""


@unittest.skipUnless(SYMIRI.is_file(), "symiri executable not found")
class TensorWindowTypingTests(unittest.TestCase):
  def check(self, source, profile="v2"):
    return check_source(source, tensor_profile=profile)

  def test_window_types_a_rank_two_source(self):
    result = self.check(WINDOW_SOURCE)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_window_rejects_a_non_parameter_source(self):
    result = self.check(BROADCAST_SOURCE)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("parameter", result.stderr)

  def test_window_rejects_a_float_index(self):
    result = self.check(FLOAT_INDEX_SOURCE)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("i32", result.stderr)

  def test_window_rejects_active_beyond_the_source(self):
    source = WINDOW_SOURCE.replace("[4, 1, 8]", "[4, 1, 9]")
    result = self.check(source)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("active", result.stderr)

  def test_window_rejects_a_non_literal_fill(self):
    result = self.check(NON_LITERAL_FILL_SOURCE)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("literal", result.stderr)

  def test_window_rows_equal_to_the_leading_extent(self):
    source = """\
fun @kernel(%src: tensor [8, 2] f32, %i: i32) : tensor [8, 2] f32 {
  let mut %w: tensor [8, 2] f32 = undef;
^entry:
  %w = tensor window %src, %i, 0.0, [8, 0, 8];
  ret %w;
}
"""
    result = self.check(source)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


WINDOW_INTERP_SOURCE = """\
fun @kernel(%src: tensor [8] f32, %i: i32) : tensor [4] f32 {
  let mut %w: tensor [4] f32 = undef;
^entry:
  %w = tensor window %src, %i, 0.0, [4, 1, 8];
  ret %w;
}
"""

WINDOW_FILL_SOURCE = """\
fun @kernel(%src: tensor [8] f32, %i: i32) : tensor [4] f32 {
  let mut %w: tensor [4] f32 = undef;
^entry:
  %w = tensor window %src, %i, -1.0, [4, 1, 2];
  ret %w;
}
"""

WINDOW_STRIDE_SOURCE = """\
fun @kernel(%src: tensor [8] f32, %i: i32) : tensor [4] f32 {
  let mut %w: tensor [4] f32 = undef;
^entry:
  %w = tensor window %src, %i, 0.0, [4, 2, 8];
  ret %w;
}
"""

WINDOW_TILE_SOURCE = """\
fun @kernel(%src: tensor [8, 2] f32, %i: i32) : tensor [8, 2] f32 {
  let mut %tile: tensor [1, 2] f32 = undef;
  let mut %state: tensor [8, 2] f32 = undef;
^entry:
  %tile = tensor window %src, %i, 0.0, [1, 1, 8];
  %state = tensor broadcast %tile, [8, 2];
  ret %state;
}
"""

SOURCE_VALUES = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)


def window_inputs(index, values=SOURCE_VALUES):
  return struct.pack(f"<{len(values)}f", *values) + struct.pack("<i", index)


@unittest.skipUnless(SYMIRI.is_file(), "symiri executable not found")
class TensorWindowInterpreterTests(unittest.TestCase):
  def run_window(self, source, index):
    result, output = run_typed(source, window_inputs(index), tensor_profile="v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return struct.unpack(f"<{len(output) // 4}f", output)

  def test_window_reads_the_active_rows(self):
    self.assertEqual(self.run_window(WINDOW_INTERP_SOURCE, 0), (10.0, 20.0, 30.0, 40.0))
    self.assertEqual(self.run_window(WINDOW_INTERP_SOURCE, 1), (20.0, 30.0, 40.0, 50.0))

  def test_window_fills_inactive_rows(self):
    self.assertEqual(self.run_window(WINDOW_FILL_SOURCE, 0), (10.0, 20.0, -1.0, -1.0))

  def test_window_stride_selects_rows(self):
    # The stride advances the block start: index 1 with stride 2 begins at row 2.
    self.assertEqual(self.run_window(WINDOW_STRIDE_SOURCE, 1), (30.0, 40.0, 50.0, 60.0))

  def test_window_positions_below_zero_read_the_fill(self):
    # Active positions are those in [0, active); row -1 reads the fill like
    # any other inactive position (generation plan G2).
    self.assertEqual(self.run_window(WINDOW_INTERP_SOURCE, -1), (0.0, 10.0, 20.0, 30.0))

  def test_window_large_index_masks_to_fill_without_access(self):
    values = struct.unpack("<4f", struct.pack("<4f", -1.0, -1.0, -1.0, -1.0))
    result, output = run_typed(
      WINDOW_FILL_SOURCE.replace("[4, 1, 2]", "[4, 4, 8]"),
      window_inputs(2**31 - 1),
      tensor_profile="v2",
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(struct.unpack("<4f", output), values)

  def test_window_row_tile_broadcasts_to_the_state(self):
    values = [float(value) for value in range(1, 17)]
    input_bytes = struct.pack("<16f", *values) + struct.pack("<i", 3)
    result, output = run_typed(WINDOW_TILE_SOURCE, input_bytes, tensor_profile="v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    row = (7.0, 8.0)
    self.assertEqual(struct.unpack("<16f", output), row * 8)


SOLVER_WINDOW_SOURCE = """\
fun @kernel(%src: tensor [8] f32, %i: i32) : i32 {
  let mut %w: tensor [4] f32 = undef;
  let mut %cell: f32 = undef;
^entry:
  %w = tensor window %src, %i, 0.0, [4, 1, 8];
  %cell = tensor extract %w, [0];
  require %cell > 0.0;
  ret 0;
}
"""


@unittest.skipUnless(SYMIRI.is_file(), "symiri executable not found")
class TensorWindowProfileTests(unittest.TestCase):
  def check(self, profile):
    return check_source(WINDOW_SOURCE, tensor_profile=profile)

  def test_v2_admits_a_window(self):
    result = self.check("v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_v3_admits_a_window(self):
    result = self.check("v3")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_v1_rejects_a_window(self):
    result = self.check("v1")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("window", result.stderr)

  def test_v4_rejects_a_window(self):
    result = self.check("v4")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("window", result.stderr)

  def test_exact_solver_names_window_as_unsupported(self):
    result, _, _ = solve_source(SOLVER_WINDOW_SOURCE, tensor_profile="v2")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("unsupported_theory", result.stderr)

  def test_window_in_a_straight_line_v1_function(self):
    result = self.check("v1")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("tensor window requires", result.stderr)
    self.assertNotIn("one basic block", result.stderr)


I32_WINDOW_SOURCE = """\
fun @kernel(%src: tensor [2] i32, %i: i32) : tensor [1] i32 {
  let mut %w: tensor [1] i32 = undef;
^entry:
  %w = tensor window %src, %i, <fill>, [1, 0, 2];
  ret %w;
}
"""


def i32_window(fill):
  return I32_WINDOW_SOURCE.replace("<fill>", fill)


@unittest.skipUnless(SYMIRI.is_file(), "symiri executable not found")
class TensorWindowFillRangeTests(unittest.TestCase):
  def test_an_i32_fill_above_the_signed_range_is_refused(self):
    result = check_source(i32_window("2147483648"), tensor_profile="v2")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("out of range", result.stderr)

  def test_an_i32_fill_below_the_signed_range_is_refused(self):
    result = check_source(i32_window("-2147483649"), tensor_profile="v2")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("out of range", result.stderr)


  def test_an_i32_fill_far_above_the_range_is_refused(self):
    result = check_source(i32_window("4294967296"), tensor_profile="v2")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("out of range", result.stderr)

  def test_an_i32_fill_far_below_the_range_is_refused(self):
    result = check_source(i32_window("-4294967297"), tensor_profile="v2")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("out of range", result.stderr)

  def test_an_i32_fill_above_the_range_with_a_wider_source_is_refused(self):
    source = I32_WINDOW_SOURCE.replace("tensor [2] i32", "tensor [4] i32").replace(
      "[1, 0, 2]", "[2, 1, 4]"
    )
    result = check_source(source.replace("<fill>", "2147483648"), tensor_profile="v2")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("out of range", result.stderr)
  def test_the_i32_signed_boundaries_are_admitted(self):
    for fill in ("2147483647", "-2147483648"):
      with self.subTest(fill=fill):
        result = check_source(i32_window(fill), tensor_profile="v2")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
  unittest.main()
