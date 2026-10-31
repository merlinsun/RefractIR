"""`tensor where`: a per-lane choice by a comparison (feature-ranked plan, L3)."""

import struct
import unittest

from test.unit.tensor_support import SYMIRC, SYMIRI, check_source, run_typed
from test.unit.test_tensor_emit import emit_cli


def where_program(relop, element="f32", cond_shape="[8]", arm_shape="[8]"):
  return f"""\
fun @kernel(%a: tensor {cond_shape} {element}, %b: tensor {cond_shape} {element}, %x: tensor {arm_shape} {element}, %y: tensor {arm_shape} {element}) : tensor [8] {element} {{
  let mut %out: tensor [8] {element} = undef;
^entry:
  %out = tensor where {relop} %a, %b, %x, %y;
  ret %out;
}}
"""


def run(relop, element, a, b, x, y):
  code = "f" if element == "f32" else "i"
  return run_typed(
    where_program(relop, element),
    struct.pack(f"<32{code}", *a, *b, *x, *y),
    tensor_profile="v3",
    reference_bits=128,
  )


@unittest.skipUnless(SYMIRI.is_file(), "symiri is unavailable")
class WhereTypingTests(unittest.TestCase):
  def test_where_checks_under_v2_and_v3_and_broadcasts(self):
    for profile in ("v2", "v3"):
      for relop in ("lt", "le", "eq"):
        with self.subTest(profile=profile, relop=relop):
          result = check_source(where_program(relop), tensor_profile=profile)
          self.assertEqual(result.returncode, 0, result.stderr)
    broadcast = where_program("lt", cond_shape="[1]")
    result = check_source(broadcast, tensor_profile="v3")
    self.assertEqual(result.returncode, 0, result.stderr)

  def test_where_rejects_mixed_elements_and_unknown_relations(self):
    mixed = where_program("lt").replace("%x: tensor [8] f32", "%x: tensor [8] i32")
    result = check_source(mixed, tensor_profile="v3")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("tensor-where", result.stderr)
    result = check_source(where_program("gt"), tensor_profile="v3")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("where relation", result.stderr)

  def test_v1_does_not_admit_where(self):
    result = check_source(where_program("lt"), tensor_profile="v1")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("tensor-profile: where", result.stderr)


@unittest.skipUnless(SYMIRI.is_file(), "symiri is unavailable")
class WhereReferenceTests(unittest.TestCase):
  def test_i32_lanes_take_the_arm_their_comparison_picks(self):
    a = [0, 1, 2, 3, 4, 5, 6, 7]
    b = [3] * 8
    x = [100 + i for i in range(8)]
    y = [-i for i in range(8)]
    for relop, holds in (
      ("lt", lambda p, q: p < q),
      ("le", lambda p, q: p <= q),
      ("eq", lambda p, q: p == q),
    ):
      with self.subTest(relop=relop):
        result, output = run(relop, "i32", a, b, x, y)
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = [x[i] if holds(a[i], b[i]) else y[i] for i in range(8)]
        self.assertEqual(struct.unpack("<8i", output), tuple(expected))

  def test_a_one_lane_condition_broadcasts_over_the_arms(self):
    result, output = run_typed(
      where_program("lt", element="i32", cond_shape="[1]"),
      struct.pack("<18i", 1, 2, *range(8), *range(10, 18)),
      tensor_profile="v3",
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(struct.unpack("<8i", output), tuple(range(8)))

  def test_f32_zeros_compare_equal_and_the_arm_keeps_its_bits(self):
    a = [-0.0, 0.0, 1.0, -1.0, 2.5, 2.5, 0.1, 3.0]
    b = [0.0, -0.0, 1.0, -2.0, 2.5, 3.0, 0.2, 3.0]
    x = [-0.0] * 8
    y = [7.0] * 8
    result, output = run("eq", "f32", a, b, x, y)
    self.assertEqual(result.returncode, 0, result.stderr)
    words = struct.unpack("<8I", output)
    seven = struct.unpack("<I", struct.pack("<f", 7.0))[0]
    self.assertEqual(
      words,
      (0x80000000, 0x80000000, 0x80000000, seven, 0x80000000, seven, seven, 0x80000000),
    )


@unittest.skipUnless(SYMIRC.is_file(), "symirc is unavailable")
class WhereEmissionTests(unittest.TestCase):
  def test_where_lowers_to_tl_where_with_the_comparison(self):
    for relop, symbol in (("lt", "<"), ("le", "<="), ("eq", "==")):
      with self.subTest(relop=relop):
        result, output = emit_cli(
          where_program(relop), ("--tensor-profile-version", "v3")
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(output, rf"tl\.where\(\(\w+ {symbol} \w+\), \w+, \w+\)")


if __name__ == "__main__":
  unittest.main()
