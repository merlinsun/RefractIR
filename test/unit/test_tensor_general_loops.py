"""Counted loops with runtime bounds and any nonzero step (generation plan G1).

Under tensor-v5 a counted loop is `for i in range(lo, hi, s)`: the start and
the bound are each a literal or an i32 entry parameter with an assumed range,
the latch adds a nonzero literal step, and a negative step pairs with a `>`
header. Empty ranges are legal, which is the shape of upstream F2
(triton-lang/triton#12525). After the loop the induction variable holds
`lo + k*s` for the `k` trips run, as Python's `range` would leave it. No GPU is
required; the emitted kernel runs on the numpy stand-in.
"""

import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from test.unit.tensor_support import SYMIRC, SYMIRI, check_source
from test.unit.test_tensor_emit import emit_cli
from test.unit.triton_numpy_shim import np
from test.unit.triton_numpy_shim import run as run_shim

LOOP = """\
fun @kernel(%x: tensor [4] i32, %lo: i32, %hi: i32) : tensor [4] i32 {{
  let mut %i: i32 = undef;
  let mut %acc: tensor [4] i32 = undef;
^entry:
{assumes}
  %acc = tensor mul %x, 0;
  %i = {start};
  br ^header;
^header:
  br %i {relation} {bound}, ^body, ^exit;
^body:
  %acc = tensor add %acc, %i;
  %i = %i + {step};
  br ^header;
^exit:
  %acc = tensor add %acc, %i;
  ret %acc;
}}
"""

RANGES = (
  "  assume %lo >= -16;\n  assume %lo <= 16;\n  assume %hi >= -16;\n  assume %hi <= 16;"
)
X = [3, -1, 4, 1]


def loop(step, *, assumes=RANGES, start="%lo", bound="%hi"):
  relation = "<" if not str(step).startswith("-") else ">"
  return LOOP.format(
    assumes=assumes, start=start, bound=bound, relation=relation, step=step
  )


def expected(lo, hi, step):
  """Every lane: the sum of the range plus the exit value of the induction variable."""
  values = list(range(lo, hi, step))
  exit_value = lo + len(values) * step
  return [sum(values) + exit_value] * len(X)


def run_p0(source, lo, hi, profile="v5"):
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "case.sir").write_text(source, encoding="utf-8")
    (root / "in.bin").write_bytes(struct.pack("<4i", *X) + struct.pack("<2i", lo, hi))
    completed = subprocess.run(
      [
        str(SYMIRI),
        "--tensor-profile-version",
        profile,
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
    if completed.returncode != 0:
      return completed.stdout + completed.stderr
    return list(struct.unpack("<4i", (root / "out.bin").read_bytes()))


def signed(words):
  return [struct.unpack("<i", struct.pack("<I", word))[0] for word in words]


# F2's bound sets: three empty ranges and one ordinary range.
F2_CASES = ((-3, -1, -2), (0, 2, -2), (2, 0, 2), (0, 10, 1))
# Thirty (lo, hi, step) cases over both signs, both empty forms and uneven strides.
EXIT_CASES = [
  (lo, hi, step)
  for step in (1, 2, 3, -1, -2, -3)
  for lo, hi in ((0, 5), (5, 0), (-4, 4), (4, -4), (-7, -7))
]


@unittest.skipUnless(SYMIRI.is_file(), "symiri is unavailable")
class GeneralLoopProfileTests(unittest.TestCase):
  def check(self, source, profile="v5"):
    return check_source(source, tensor_profile=profile)

  def test_v5_accepts_a_negative_step_with_a_greater_than_header(self):
    result = self.check(loop(-2))
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_v3_still_refuses_a_negative_step(self):
    result = self.check(loop(-2), "v3")
    self.assertNotEqual(result.returncode, 0)

  def test_a_zero_step_is_refused(self):
    result = self.check(loop(0))
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("step", result.stdout + result.stderr)

  def test_a_negative_step_needs_a_greater_than_header(self):
    source = loop(-2).replace("%i > %hi", "%i < %hi")
    result = self.check(source)
    self.assertNotEqual(result.returncode, 0)

  def test_a_range_whose_trips_exceed_the_budget_is_refused(self):
    wide = "  assume %lo >= -100;\n  assume %lo <= 100;\n  assume %hi >= -100;\n  assume %hi <= 100;"
    result = self.check(loop(1, assumes=wide))
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("64", result.stdout + result.stderr)

  def test_an_empty_literal_range_is_legal_under_v5(self):
    source = loop(1, assumes="", start="5", bound="0")
    self.assertEqual(self.check(source).returncode, 0)
    self.assertNotEqual(self.check(source, "v3").returncode, 0)


@unittest.skipUnless(SYMIRI.is_file(), "symiri is unavailable")
class GeneralLoopP0Tests(unittest.TestCase):
  def test_p0_runs_f2s_bound_sets(self):
    for lo, hi, step in F2_CASES:
      with self.subTest(lo=lo, hi=hi, step=step):
        self.assertEqual(run_p0(loop(step), lo, hi), expected(lo, hi, step))

  def test_p0s_exit_value_matches_pythons_range(self):
    for lo, hi, step in EXIT_CASES:
      with self.subTest(lo=lo, hi=hi, step=step):
        self.assertEqual(run_p0(loop(step), lo, hi), expected(lo, hi, step))


@unittest.skipUnless(SYMIRC.is_file(), "symirc is unavailable")
class GeneralLoopEmissionTests(unittest.TestCase):
  def emit(self, source):
    result, output = emit_cli(source, ("--tensor-profile-version", "v5"))
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return output

  def test_the_loop_is_a_tl_range_with_runtime_bounds_and_the_literal_step(self):
    output = self.emit(loop(-2))
    self.assertIn("tl.load(lo_ptr)", output)
    self.assertIn("tl.load(hi_ptr)", output)
    self.assertRegex(output, r"for _sir_i in tl\.range\(_sir_\w+, _sir_\w+, -2\):")

  def test_a_literal_range_keeps_its_literals(self):
    output = self.emit(loop(3, assumes="", start="-4", bound="4"))
    self.assertRegex(output, r"tl\.range\(-4, 4, 3\)")


@unittest.skipUnless(
  SYMIRC.is_file() and np is not None, "symirc or numpy is unavailable"
)
class GeneralLoopExecutionTests(unittest.TestCase):
  def test_a_bound_below_a_literal_start_leaves_the_variable_at_the_start(self):
    source = loop(1, start="0")
    self.assertEqual(run_p0(source, 0, -3), expected(0, -3, 1))
    kernel = emit_cli(source, ("--tensor-profile-version", "v5"))[1]
    self.assertEqual(
      signed(run_shim(kernel, [X, [0], [-3]], len(X), np.int32)), expected(0, -3, 1)
    )

  def test_the_kernel_computes_p0s_output_for_every_case(self):
    kernels = {}
    for lo, hi, step in (*F2_CASES, *EXIT_CASES):
      with self.subTest(lo=lo, hi=hi, step=step):
        if step not in kernels:
          result, kernels[step] = emit_cli(
            loop(step), ("--tensor-profile-version", "v5")
          )
          self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        words = run_shim(kernels[step], [X, [lo], [hi]], len(X), np.int32)
        self.assertEqual(signed(words), expected(lo, hi, step))


if __name__ == "__main__":
  unittest.main()
