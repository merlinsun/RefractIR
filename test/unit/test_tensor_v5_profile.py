"""The tensor-v5 generation profile; no GPU is required.

tensor-v5 serves generation without an expected output: v3's operations plus
div, exp, and rem, rank two, padded tiles of up to 65,536 lanes, and up to 64
loop body executions. Its kernels are judged by relations, not by an exact
solver, so the exact solver refuses the profile.
"""

import unittest

from test.unit.tensor_support import (
  SYMIR_SOLVE,
  SYMIRC,
  SYMIRI,
  check_source,
  solve_source,
)
from test.unit.test_tensor_emit import emit_cli

BINARY = """\
fun @kernel(%x: tensor [8] f32, %y: tensor [8] f32) : tensor [8] f32 {{
  let mut %out: tensor [8] f32 = undef;
^entry:
  %out = tensor {op} %x, %y;
  ret %out;
}}
"""

EXP = """\
fun @kernel(%x: tensor [8] f32) : tensor [8] f32 {
  let mut %out: tensor [8] f32 = undef;
^entry:
  %out = tensor exp %x;
  ret %out;
}
"""

LOOP = """\
fun @kernel(%x: tensor [8] i32) : tensor [8] i32 {{
  let mut %i: i32 = undef;
  let mut %acc: tensor [8] i32 = undef;
^entry:
  %acc = tensor mul %x, 0;
  %i = 0;
  br ^header;
^header:
  br %i < {trips}, ^body, ^exit;
^body:
  %acc = tensor add %acc, %x;
  %i = %i + 1;
  br ^header;
^exit:
  ret %acc;
}}
"""

WIDE = """\
fun @kernel(%x: tensor [1000] f32, %y: tensor [1000] f32) : tensor [1000] f32 {
  let mut %out: tensor [1000] f32 = undef;
^entry:
  %out = tensor add %x, %y;
  ret %out;
}
"""


def check(source, profile):
  return check_source(source, tensor_profile=profile)


@unittest.skipUnless(SYMIRI.is_file(), "symiri is unavailable")
class V5ProfileTests(unittest.TestCase):
  def assertAccepted(self, source, profile):
    result = check(source, profile)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def assertRefused(self, source, profile, message):
    result = check(source, profile)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn(message, result.stdout + result.stderr)

  def test_a_v3_program_is_a_v5_program(self):
    for source in (LOOP.format(trips=8), BINARY.format(op="add")):
      with self.subTest(source=source.splitlines()[3]):
        self.assertAccepted(source, "v3")
        self.assertAccepted(source, "v5")

  def test_div_is_admitted_by_v5_and_still_refused_by_v3(self):
    self.assertAccepted(BINARY.format(op="div"), "v5")
    self.assertRefused(BINARY.format(op="div"), "v3", "tensor-v3")

  def test_exp_is_admitted_by_v5_and_still_refused_by_v3(self):
    self.assertAccepted(EXP, "v5")
    self.assertRefused(EXP, "v3", "tensor-v3")

  def test_rem_is_admitted_by_v5_inside_any_body(self):
    source = BINARY.format(op="rem").replace(
      "ret %out;", "%out = tensor add %out, %x;\n  ret %out;"
    )
    self.assertAccepted(source, "v5")
    self.assertRefused(source, "v3", "tensor-v4")

  def test_v5_admits_64_loop_executions_and_refuses_65(self):
    self.assertAccepted(LOOP.format(trips=64), "v5")
    self.assertRefused(LOOP.format(trips=65), "v5", "64")
    self.assertRefused(LOOP.format(trips=9), "v3", "[1, 8]")

  def test_v5_keeps_v3s_rank_limit(self):
    source = BINARY.format(op="add").replace("[8]", "[2, 2, 2]")
    self.assertRefused(source, "v5", "rank")

  def test_v5_pads_a_1000_lane_tensor_that_v3_refuses(self):
    self.assertAccepted(WIDE, "v5")
    self.assertRefused(WIDE, "v3", "512")


@unittest.skipUnless(SYMIR_SOLVE.is_file(), "symirsolve is unavailable")
class V5SolverTests(unittest.TestCase):
  def test_the_exact_solver_refuses_v5(self):
    result = solve_source(BINARY.format(op="add"), tensor_profile="v5")[0]
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("tensor-v5", result.stdout + result.stderr)


@unittest.skipUnless(SYMIRC.is_file(), "symirc is unavailable")
class V5EmissionTests(unittest.TestCase):
  def test_v5_emits_div_and_masks_the_padded_lanes(self):
    source = BINARY.format(op="div").replace("[8]", "[5]")
    result, output = emit_cli(source, ("--tensor-profile-version", "v5"))
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn(" / ", output)
    self.assertIn("tl.arange(0, 8)", output)
    self.assertRegex(output, r"mask=\w+")


if __name__ == "__main__":
  unittest.main()
