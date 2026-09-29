"""Bounded exact tensor symbolic execution tests."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from test.unit.tensor_support import SYMIR_SOLVE, solve_source

ROOT = Path(__file__).resolve().parents[2]


def load_json(path):
  try:
    return json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return None


def solve_case(source, *, path="^entry", entry="@kernel", flags=()):
  """Solve one function over an explicit path, returning both model files.

  Kept local because the shared `tensor_support.solve_source` helper is the
  straight-line `^entry` case and does not take solver flags.
  """
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source_path = root / "case.sir"
    output_path = root / "solved.sir"
    model_path = root / "model.json"
    params_path = root / "params.json"
    source_path.write_text(source, encoding="utf-8")
    result = subprocess.run(
      [
        str(SYMIR_SOLVE),
        "--main",
        entry,
        "--path",
        path,
        "--output",
        str(output_path),
        "--emit-model",
        str(model_path),
        "--emit-tensor-params",
        str(params_path),
        *flags,
        str(source_path),
      ],
      capture_output=True,
      text=True,
      check=False,
      timeout=30,
    )
    output = output_path.read_text(encoding="utf-8") if output_path.exists() else None
    return result, output, load_json(model_path), load_json(params_path)


def assert_witness(test, record):
  """A witness record is well formed: non-empty shape and one leaf per element."""
  test.assertIsInstance(record, dict)
  shape = record["shape"]
  leaves = record["leaves"]
  test.assertIn(record["element"], ("i32", "f32"))
  test.assertTrue(shape, "a witness shape is never empty")
  test.assertTrue(all(dimension > 0 for dimension in shape))
  count = 1
  for dimension in shape:
    count *= dimension
  test.assertEqual(len(leaves), count)


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

  def test_tensor_parameter_witness_satisfies_complete_target(self):
    source = """\
fun @kernel(%x: tensor [4] i32) : i32 {
  let mut %out: i32 = undef;
^entry:
  %out = tensor sum %x, 0, drop;
  require %out == 100;
  ret %out;
}
"""

    result, _, model, params = solve_case(source)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("SAT", result.stdout)
    record = model["@kernel"]["%x"]
    assert_witness(self, record)
    self.assertEqual(record["shape"], [4])
    self.assertEqual(record["element"], "i32")
    self.assertEqual(len(record["leaves"]), 4)
    self.assertEqual(sum(record["leaves"]), 100)
    self.assertEqual(params["@kernel"]["%x"], record)

  def test_two_tensor_parameters_keep_distinct_identities(self):
    source = """\
fun @kernel(%a: tensor [2] i32, %b: tensor [2] i32) : i32 {
  let mut %sa: i32 = undef;
  let mut %sb: i32 = undef;
^entry:
  %sa = tensor sum %a, 0, drop;
  %sb = tensor sum %b, 0, drop;
  require %sa == 3;
  require %sb == 11;
  ret %sa;
}
"""

    result, _, model, _ = solve_case(source)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("SAT", result.stdout)
    self.assertEqual(sorted(model["@kernel"]), ["%a", "%b"])
    first = model["@kernel"]["%a"]
    second = model["@kernel"]["%b"]
    assert_witness(self, first)
    assert_witness(self, second)
    self.assertEqual(first["shape"], [2])
    self.assertEqual(second["shape"], [2])
    self.assertEqual(sum(first["leaves"]), 3)
    self.assertEqual(sum(second["leaves"]), 11)
    self.assertNotEqual(first["leaves"], second["leaves"])

  def test_scalar_and_tensor_parameters_are_disjoint(self):
    source = """\
fun @kernel(%s: i32, %x: tensor [2] i32) : i32 {
  let mut %out: i32 = undef;
^entry:
  %out = tensor sum %x, 0, drop;
  require %out == 10;
  require %s == 7;
  ret %s;
}
"""

    result, output, model, params = solve_case(source)

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("SAT", result.stdout)
    header = output.splitlines()[0]
    self.assertIn("%s=7", header)
    self.assertNotIn("%x", header)
    self.assertEqual(list(model["@kernel"]), ["%x"])
    self.assertEqual(list(params["@kernel"]), ["%x"])
    self.assertNotIn("%s", model["@kernel"])
    assert_witness(self, model["@kernel"]["%x"])

  def test_bounded_counted_loop_tensor_witness(self):
    source = """\
fun @kernel(%x: tensor [2] i32) : i32 {
  let mut %i: i32 = undef;
  let mut %acc: tensor [2] i32 = undef;
  let mut %out: i32 = undef;
^entry:
  %i = 0;
  %acc = %x;
  br ^header;
^header:
  br %i < 2, ^body, ^exit;
^body:
  %acc = tensor add %acc, %x;
  %i = %i + 1;
  br ^header;
^exit:
  %out = tensor sum %acc, 0, drop;
  require %out == 18;
  ret %out;
}
"""
    loop_path = "^entry,^header,^body,^header,^body,^header,^exit"

    result, _, model, _ = solve_case(
      source,
      path=loop_path,
      flags=("--tensor-profile-version", "v2", "--tensor-param-domain", "0,3"),
    )

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("SAT", result.stdout)
    record = model["@kernel"]["%x"]
    assert_witness(self, record)
    self.assertEqual(record["shape"], [2])
    self.assertEqual(record["leaves"], [3, 3])

  def test_v3_logical_tail_witness_has_no_padding(self):
    source = """\
fun @kernel(%x: tensor [3] i32) : i32 {
  let mut %out: i32 = undef;
^entry:
  %out = tensor sum %x, 0, drop;
  require %out == 300;
  ret %out;
}
"""

    result, _, model, _ = solve_case(source, flags=("--tensor-profile-version", "v3"))

    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("SAT", result.stdout)
    record = model["@kernel"]["%x"]
    assert_witness(self, record)
    self.assertEqual(record["shape"], [3])
    self.assertEqual(len(record["leaves"]), 3)
    self.assertEqual(sum(record["leaves"]), 300)

  def test_v3_profile_rejects_illegal_program(self):
    # `tensor extract` is legal under the default v1 profile and unsupported
    # in v3, so the same source separates "checked and solved" from
    # "rejected before solving".
    source = """\
fun @kernel(%x: tensor [2] i32) : i32 {
  let mut %cell: i32 = undef;
^entry:
  %cell = tensor extract %x, [1];
  require %cell == 5;
  ret %cell;
}
"""

    accepted, _, model, _ = solve_case(source)
    rejected, _, _, rejected_params = solve_case(
      source, flags=("--tensor-profile-version", "v3")
    )

    self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
    self.assertIn("SAT", accepted.stdout)
    self.assertEqual(model["@kernel"]["%x"]["leaves"][1], 5)
    self.assertNotEqual(rejected.returncode, 0)
    self.assertNotIn("SAT", rejected.stdout)
    self.assertIn("tensor-profile", rejected.stderr)
    self.assertIsNone(rejected_params)

  def test_malformed_witness_is_rejected_by_valid(self):
    compiler = next(
      (path for name in ("c++", "g++", "clang++") if (path := shutil.which(name))),
      None,
    )
    if compiler is None:
      self.skipTest("no C++ compiler to exercise TensorParamWitness::valid")

    probe = """\
#include <cstdint>
#include <vector>

#include "solver/solver.hpp"

using Witness = refractir::SymbolicExecutor::Result::TensorParamWitness;

int main() {
  Witness complete;
  complete.shape = {2, 2};
  complete.leaves = {int64_t{1}, int64_t{2}, int64_t{3}, int64_t{4}};
  if (!complete.valid())
    return 1;

  Witness shortLeaves;
  shortLeaves.shape = {2, 2};
  shortLeaves.leaves = {int64_t{1}};
  if (shortLeaves.valid())
    return 2;

  Witness zeroElement;
  zeroElement.shape = {0};
  if (zeroElement.valid())
    return 3;

  Witness emptyShape;
  if (emptyShape.valid())
    return 4;

  return 0;
}
"""
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source_path = root / "witness_probe.cpp"
      binary_path = root / "witness_probe"
      source_path.write_text(probe, encoding="utf-8")
      build = subprocess.run(
        [
          compiler,
          "-std=c++20",
          "-I",
          str(ROOT / "include"),
          "-o",
          str(binary_path),
          str(source_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
      )
      self.assertEqual(build.returncode, 0, build.stderr)
      outcome = subprocess.run(
        [str(binary_path)], capture_output=True, text=True, check=False, timeout=30
      )

    failures = {
      1: "a complete witness was rejected",
      2: "a leaf-count disagreement was accepted",
      3: "a zero-element shape was accepted",
      4: "an empty shape was accepted",
    }
    self.assertEqual(
      outcome.returncode, 0, failures.get(outcome.returncode, outcome.stderr)
    )


if __name__ == "__main__":
  unittest.main()
