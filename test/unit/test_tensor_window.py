"""Contract tests for the tensor window operation's surface syntax."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
  unittest.main()
