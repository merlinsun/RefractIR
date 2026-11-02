"""Steering a kernel's integers toward corner paths (generation plan G3).

The scalar skeleton of a tensor-v5 kernel keeps its control flow and integer
locals, turns each i32 parameter into a ranged symbol, and drops every tensor
statement, which cannot affect control. Each target is a block path, with
`require` lines that pin a corner such as an inverted empty range;
`symirsolve` chooses the scalars, and P0 replays the real kernel with them and
must take exactly that path. No GPU is required.
"""

import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from test.lib import tensor_path_steering as steering
from test.unit.tensor_support import SYMIR_SOLVE, SYMIRI

FREE_FORM_API = Path(
  os.environ.get("REFRACTIR_TENSOR_FREE_FORM_API", "build/bin/tensor_free_form_api")
)

F2_LOOP = """\
fun @kernel(%x: tensor [4] i32, %lo: i32, %hi: i32) : tensor [4] i32 {
  let mut %i: i32 = undef;
  let mut %acc: tensor [4] i32 = undef;
^entry:
  assume %lo >= -16;
  assume %lo <= 16;
  assume %hi >= -16;
  assume %hi <= 16;
  %acc = tensor mul %x, 0;
  %i = %lo;
  br ^header;
^header:
  br %i > %hi, ^body, ^exit;
^body:
  %acc = tensor add %acc, %i;
  %i = %i + -2;
  br ^header;
^exit:
  ret %acc;
}
"""

BRANCH_LOOP = """\
fun @kernel(%x: tensor [4] i32, %n: i32, %sel: i32) : tensor [4] i32 {
  let mut %i: i32 = undef;
  let mut %acc: tensor [4] i32 = undef;
^entry:
  assume %n >= 0;
  assume %n <= 4;
  assume %sel >= 0;
  assume %sel <= 1;
  %acc = tensor mul %x, 0;
  %i = 0;
  br ^header;
^header:
  br %i < %n, ^body, ^exit;
^body:
  br %sel < 1, ^keep, ^swap;
^keep:
  %acc = tensor add %acc, %x;
  br ^latch;
^swap:
  %acc = tensor sub %acc, %x;
  br ^latch;
^latch:
  %i = %i + 1;
  br ^header;
^exit:
  ret %acc;
}
"""

NO_TRIP = (
  F2_LOOP.replace(
    "  assume %lo >= -16;\n  assume %lo <= 16;\n  assume %hi >= -16;\n  assume %hi <= 16;",
    "  assume %lo >= 5;\n  assume %lo <= 6;\n  assume %hi >= 0;\n  assume %hi <= 1;",
  )
  .replace("%i > %hi", "%i < %hi")
  .replace("%i + -2", "%i + 1")
)

STRAIGHT = """\
fun @kernel(%x: tensor [4] i32, %y: tensor [4] i32) : tensor [4] i32 {
  let mut %out: tensor [4] i32 = undef;
^entry:
  %out = tensor add %x, %y;
  ret %out;
}
"""

READY = steering.STEERING.is_file() and SYMIRI.is_file() and SYMIR_SOLVE.is_file()


def names(source):
  return [target["name"] for target in steering.targets(source)]


@unittest.skipUnless(SYMIRI.is_file(), "symiri is unavailable")
class TypedTraceTests(unittest.TestCase):
  def test_a_typed_run_dumps_the_executed_blocks(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / "case.sir").write_text(F2_LOOP, encoding="utf-8")
      (root / "in.bin").write_bytes(
        struct.pack("<4i", 1, 2, 3, 4) + struct.pack("<2i", 3, 0)
      )
      completed = subprocess.run(
        [
          str(SYMIRI),
          "--tensor-profile-version",
          "v5",
          "--dump-trace",
          "--entry",
          "kernel",
          "--tensor-input",
          str(root / "in.bin"),
          "--tensor-output",
          str(root / "out.bin"),
          str(root / "case.sir"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
      )
    self.assertEqual(completed.returncode, 0, completed.stderr)
    self.assertEqual(
      steering.executed_blocks(completed.stdout),
      ["^entry", "^header", "^body", "^header", "^body", "^header", "^exit"],
    )


@unittest.skipUnless(READY, "the steering driver, symiri or symirsolve is unavailable")
class SteeringTests(unittest.TestCase):
  def test_a_runtime_loop_offers_its_corner_targets(self):
    self.assertEqual(
      names(F2_LOOP),
      [
        "baseline",
        "loop0-empty-equal",
        "loop0-empty-inverted",
        "loop0-one",
        "loop0-most",
      ],
    )

  def test_every_target_is_solved_and_reached_in_p0(self):
    for result in steering.steer(F2_LOOP):
      with self.subTest(target=result["name"]):
        self.assertEqual(result["status"], "reached", result)

  def test_f2s_inverted_empty_range_comes_from_the_target_alone(self):
    result = next(
      r for r in steering.steer(F2_LOOP) if r["name"] == "loop0-empty-inverted"
    )
    self.assertLess(result["scalars"]["lo"], result["scalars"]["hi"])
    self.assertEqual(result["path"], ["^entry", "^header", "^exit"])

  def test_an_unreachable_target_is_reported_unsatisfiable(self):
    results = {r["name"]: r["status"] for r in steering.steer(NO_TRIP)}
    self.assertEqual(results["loop0-one"], "unsatisfiable")
    self.assertEqual(results["loop0-empty-inverted"], "reached")

  def test_a_kernel_without_control_flow_has_one_path(self):
    self.assertEqual(names(STRAIGHT), ["baseline"])
    self.assertEqual([r["status"] for r in steering.steer(STRAIGHT)], ["reached"])

  def test_both_arms_of_a_branch_are_reached(self):
    results = {r["name"]: r for r in steering.steer(BRANCH_LOOP)}
    for arm, label in (("then", "^keep"), ("else", "^swap")):
      with self.subTest(arm=arm):
        result = results[f"branch0-{arm}"]
        self.assertEqual(result["status"], "reached", result)
        self.assertIn(label, result["path"])


@unittest.skipUnless(
  READY and FREE_FORM_API.is_file(), "the free-form API is unavailable"
)
class WindowSteeringTests(unittest.TestCase):
  def test_a_window_is_steered_across_the_start_and_the_end_of_its_source(self):
    source = subprocess.run(
      [str(FREE_FORM_API), "range-window-source"],
      capture_output=True,
      text=True,
      check=True,
    ).stdout
    results = {r["name"]: r for r in steering.steer(source)}
    below = results["window0-below"]
    self.assertEqual(below["status"], "reached", below)
    self.assertLess(below["scalars"]["lo"], 0)
    self.assertGreater(below["scalars"]["lo"] + 2, 0)
    above = results["window0-above"]
    self.assertEqual(above["status"], "reached", above)
    self.assertLess(above["scalars"]["lo"], 4)
    self.assertGreater(above["scalars"]["lo"] + 2, 4)

  def test_every_target_of_nested_runtime_loops_is_reached(self):
    source = subprocess.run(
      [str(FREE_FORM_API), "range-nested-source"],
      capture_output=True,
      text=True,
      check=True,
    ).stdout
    statuses = {r["name"]: r["status"] for r in steering.steer(source)}
    # The outer loop is range(%m, -1, -1) with %m in [0, 3]: it always runs.
    # The inner loop is range(0, %n, 2) with %n in [0, 6]: it cannot invert.
    impossible = {"loop0-empty-equal", "loop0-empty-inverted", "loop1-empty-inverted"}
    self.assertIn("loop1-most", statuses)
    for name, status in statuses.items():
      with self.subTest(target=name):
        self.assertEqual(status, "unsatisfiable" if name in impossible else "reached")


if __name__ == "__main__":
  unittest.main()
