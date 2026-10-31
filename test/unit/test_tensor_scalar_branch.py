"""Scalar `if` diamonds in tensor kernels (feature-ranked plan, L4)."""

import struct
import unittest

from test.unit.tensor_support import SYMIRC, SYMIRI, check_source, run_typed
from test.unit.test_tensor_emit import emit_cli

# A counted loop whose body swaps or keeps the two states, chosen by %sel.
LOOP_DIAMOND = """\
fun @kernel(%x: tensor [2, 4] f32, %y: tensor [2, 4] f32, %sel: i32) : tensor [2, 4] f32 {
  let mut %i: i32 = undef;
  let mut %a: tensor [2, 4] f32 = undef;
  let mut %b: tensor [2, 4] f32 = undef;
  let mut %na: tensor [2, 4] f32 = undef;
  let mut %nb: tensor [2, 4] f32 = undef;
  let mut %out: tensor [2, 4] f32 = undef;
^entry:
  %a = %x;
  %b = %y;
  %i = 0;
  br ^header;
^header:
  br %i < 2, ^body, ^exit;
^body:
  %na = tensor add %a, %b;
  %nb = tensor sub %a, %b;
  br %sel < 1, ^keep, ^swap;
^keep:
  %a = %na;
  %b = %nb;
  br ^latch;
^swap:
  %a = %nb;
  %b = %na;
  br ^latch;
^latch:
  %i = %i + 1;
  br ^header;
^exit:
  %out = tensor mul %a, %b;
  ret %out;
}
"""

# A diamond at the top level, outside every loop.
TOP_DIAMOND = """\
fun @kernel(%x: tensor [2, 4] f32, %sel: i32) : tensor [2, 4] f32 {
  let mut %out: tensor [2, 4] f32 = undef;
^entry:
  br %sel == 3, ^then, ^else;
^then:
  %out = tensor add %x, %x;
  br ^join;
^else:
  %out = tensor sub %x, %x;
  br ^join;
^join:
  ret %out;
}
"""


def run_loop(selector):
  x = [float(v) for v in range(1, 9)]
  y = [float(v) / 2 for v in range(8)]
  return run_typed(
    LOOP_DIAMOND, struct.pack("<16fi", *x, *y, selector), tensor_profile="v3"
  )


def reference(selector):
  a = [float(v) for v in range(1, 9)]
  b = [float(v) / 2 for v in range(8)]
  for _ in range(2):
    na = [p + q for p, q in zip(a, b)]
    nb = [p - q for p, q in zip(a, b)]
    a, b = (na, nb) if selector < 1 else (nb, na)
  return [p * q for p, q in zip(a, b)]


@unittest.skipUnless(SYMIRI.is_file(), "symiri is unavailable")
class ScalarBranchProfileTests(unittest.TestCase):
  def test_diamonds_in_and_outside_loops_check_under_v2_and_v3(self):
    for source in (LOOP_DIAMOND, TOP_DIAMOND):
      for profile in ("v2", "v3"):
        with self.subTest(profile=profile, looped="header" in source):
          result = check_source(source, tensor_profile=profile)
          self.assertEqual(result.returncode, 0, result.stderr)

  def test_a_branch_that_does_not_rejoin_is_refused(self):
    source = TOP_DIAMOND.replace(
      "^else:\n  %out = tensor sub %x, %x;\n  br ^join;",
      "^else:\n  %out = tensor sub %x, %x;\n  ret %out;",
    )
    result = check_source(source, tensor_profile="v3")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("tensor-profile: a scalar branch must form a diamond", result.stderr)

  def test_a_branch_on_a_tensor_value_is_refused(self):
    source = (
      TOP_DIAMOND.replace("br %sel == 3, ^then, ^else;", "br %sel == 3, ^then, ^else;")
      .replace(
        "fun @kernel(%x: tensor [2, 4] f32, %sel: i32)",
        "fun @kernel(%x: tensor [2, 4] f32, %sel: f32)",
      )
      .replace("%sel == 3", "%sel == 3.0")
    )
    result = check_source(source, tensor_profile="v3")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("i32 scalars", result.stderr)


@unittest.skipUnless(SYMIRI.is_file(), "symiri is unavailable")
class ScalarBranchReferenceTests(unittest.TestCase):
  def test_the_selector_chooses_the_arm_each_trip(self):
    for selector in (0, 1):
      with self.subTest(selector=selector):
        result, output = run_loop(selector)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(list(struct.unpack("<8f", output)), reference(selector))
    self.assertNotEqual(reference(0), reference(1))


@unittest.skipUnless(SYMIRC.is_file(), "symirc is unavailable")
class ScalarBranchEmissionTests(unittest.TestCase):
  def test_a_diamond_lowers_to_an_if_else_on_the_loaded_scalar(self):
    result, output = emit_cli(LOOP_DIAMOND, ("--tensor-profile-version", "v3"))
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertRegex(
      output, r"\n( +)if \(\w+ < 1\):\n(\1    .*\n)+\1else:\n(\1    .*\n)+"
    )
    result, output = emit_cli(TOP_DIAMOND, ("--tensor-profile-version", "v3"))
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertRegex(output, r"if \(\w+ == 3\):")


if __name__ == "__main__":
  unittest.main()
