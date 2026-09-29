"""Bounded exact tensor symbolic execution tests."""

import unittest

from test.unit.tensor_support import SYMIR_SOLVE, solve_source


class TensorSolverTests(unittest.TestCase):
  def setUp(self):
    self.assertTrue(
      SYMIR_SOLVE.is_file(), f"symirsolve executable not found: {SYMIR_SOLVE}"
    )

  def test_synthesized_coordinate_coefficients(self):
    source = """\
fun @kernel() : i32 {
  sym %?a : value i32 in [0, 10];
  sym %?b : value i32 in [0, 10];
  sym %?k : value i32 in [-10, 10];
  let mut %row: tensor [2, 4] i32 = undef;
  let mut %col: tensor [2, 4] i32 = undef;
  let mut %ar: tensor [2, 4] i32 = undef;
  let mut %bc: tensor [2, 4] i32 = undef;
  let mut %sum: tensor [2, 4] i32 = undef;
  let mut %out: tensor [2, 4] i32 = undef;
  let mut %p00: i32 = undef;
  let mut %p10: i32 = undef;
  let mut %p01: i32 = undef;
^entry:
  %row = tensor iota [2, 4], 0;
  %col = tensor iota [2, 4], 1;
  %ar = tensor mul %row, %?a;
  %bc = tensor mul %col, %?b;
  %sum = tensor add %ar, %bc;
  %out = tensor add %sum, %?k;
  %p00 = tensor extract %out, [0, 0];
  %p10 = tensor extract %out, [1, 0];
  %p01 = tensor extract %out, [0, 1];
  require %p00 == 1;
  require %p10 == 8;
  require %p01 == 4;
  ret 0;
}
"""

    result, _, model = solve_source(source)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("SAT", result.stdout)
    self.assertEqual(model["@kernel"]["%?a"], 7)
    self.assertEqual(model["@kernel"]["%?b"], 3)
    self.assertEqual(model["@kernel"]["%?k"], 1)

  def test_transpose_element_constraint(self):
    source = """\
fun @kernel() : i32 {
  sym %?x : value tensor [2, 4] i32 in [0, 9];
  let mut %out: tensor [4, 2] i32 = undef;
  let mut %cell: i32 = undef;
^entry:
  %out = tensor transpose %?x, [1, 0];
  %cell = tensor extract %out, [3, 1];
  require %cell == 7;
  ret %cell;
}
"""

    result, _, model = solve_source(source)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("SAT", result.stdout)
    self.assertEqual(len(model["@kernel"]["%?x"]), 8)
    self.assertGreaterEqual(model["@kernel"]["%?x"][1], 0)
    self.assertLessEqual(model["@kernel"]["%?x"][1], 9)
    self.assertEqual(model["@kernel"]["%?x"][7], 7)

  def test_broadcast_aliases_source_term(self):
    source = """\
fun @kernel() : i32 {
  sym %?x : value tensor [2, 1] i32;
  let mut %out: tensor [2, 4] i32 = undef;
  let mut %left: i32 = undef;
  let mut %right: i32 = undef;
^entry:
  %out = tensor broadcast %?x, [2, 4];
  %left = tensor extract %out, [0, 0];
  %right = tensor extract %out, [0, 3];
  require %left != %right;
  ret %left;
}
"""

    result, _, _ = solve_source(source)

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("UNSAT", result.stdout)

  def test_sum_checks_all_partial_ranges(self):
    source = """\
fun @kernel() : i32 {
  sym %?x : value tensor [4] i32;
  let mut %sum: i32 = undef;
  let mut %x0: i32 = undef;
  let mut %x1: i32 = undef;
  let mut %x2: i32 = undef;
  let mut %x3: i32 = undef;
^entry:
  %x0 = tensor extract %?x, [0];
  %x1 = tensor extract %?x, [1];
  %x2 = tensor extract %?x, [2];
  %x3 = tensor extract %?x, [3];
  require %x0 == 2147483647;
  require %x1 == 1;
  require %x2 == -1;
  require %x3 == 0;
  %sum = tensor sum %?x, 0, drop;
  require %sum == 2147483647;
  ret %sum;
}
"""

    result, _, _ = solve_source(source)

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("UNSAT", result.stdout)

  def test_softmax_is_unsupported_theory(self):
    control = """\
fun @kernel() : i32 {
  sym %?x : value i32 in [0, 1];
^entry:
  require %?x == 1;
  ret %?x;
}
"""
    numerical = """\
fun @kernel() : i32 {
  sym %?x : value tensor [2] f32;
  let mut %out: tensor [2] f32 = undef;
  let mut %cell: f32 = undef;
^entry:
  %out = tensor softmax %?x, 0;
  %cell = tensor extract %out, [0];
  require %cell > 0.0;
  ret 0;
}
"""

    control_result, _, _ = solve_source(control)
    unsupported_result, _, _ = solve_source(numerical)

    self.assertEqual(control_result.returncode, 0, control_result.stderr)
    self.assertIn("SAT", control_result.stdout)
    self.assertNotIn("UNSAT", unsupported_result.stdout)
    self.assertIn("unsupported_theory", unsupported_result.stderr)

  def test_singleton_sum_materializes_tensor_model(self):
    source = """\
fun @kernel() : i32 {
  sym %?x : value tensor [1] i32 in [5, 5];
  let mut %row: tensor [1] i32 = undef;
  let mut %cell: i32 = undef;
^entry:
  %row = tensor sum %?x, -1, keep;
  %cell = tensor extract %row, [0];
  require %cell == 5;
  ret %cell;
}
"""

    result, output, model = solve_source(source)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(model["@kernel"]["%?x"], [5])
    self.assertIn("let %x__solved: tensor [1] i32 = {5};", output)
    replay, _, _ = solve_source(output)
    self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)

  def test_direct_tensor_symbol_copy_retains_all_leaves(self):
    source = """\
fun @kernel() : i32 {
  sym %?x : value tensor [2] i32 in [0, 9];
  let mut %out: tensor [2] i32 = undef;
  let mut %first: i32 = undef;
  let mut %second: i32 = undef;
^entry:
  %out = %?x;
  %first = tensor extract %out, [0];
  %second = tensor extract %out, [1];
  require %first == 3;
  require %second == 7;
  ret %first;
}
"""

    result, output, model = solve_source(source)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("SAT", result.stdout)
    self.assertEqual(model["@kernel"]["%?x"], [3, 7])
    self.assertIn("let %x__solved: tensor [2] i32 = {3, 7};", output)


if __name__ == "__main__":
  unittest.main()
