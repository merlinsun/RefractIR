"""Counted loops bounded by an entry parameter (plan W4a); no GPU is required.

A counted loop's bound may be an i32 entry parameter whose range `^entry`
states with `assume %n >= lo;` and `assume %n <= hi;`, where the loop starts
at or below `lo` and `hi` allows one to eight trips. The loop may then run
zero times. P0 runs the trip count the argument gives, and the emitted kernel
loops to the loaded argument, so the compiler never sees a constant bound.
"""

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test.unit.tensor_support import SYMIRC, SYMIRI, check_source
from test.unit.test_tensor_emit import emit_cli
from test.unit.triton_numpy_shim import np
from test.unit.triton_numpy_shim import run as run_shim

LOOP = """\
fun @kernel(%x: tensor [8] i32, %n: i32) : tensor [8] i32 {{
  let mut %i: i32 = undef;
  let mut %acc: tensor [8] i32 = undef;
^entry:
{assumes}
  %acc = tensor mul %x, 0;
  %i = {start};
  br ^header;
^header:
  br %i < {bound}, ^body, ^exit;
^body:
  %acc = tensor add %acc, %x;
  %i = %i + 1;
  br ^header;
^exit:
  %acc = tensor add %acc, %i;
  ret %acc;
}}
"""

RANGED = "  assume %n >= 0;\n  assume %n <= 8;"
X = [1, -2, 3, 0, 5, 7, -9, 4]


def loop(assumes=RANGED, start=0, bound="%n"):
  return LOOP.format(assumes=assumes, start=start, bound=bound)


def run_p0(source, n):
  """P0's output for x = X and the given trip count, or its stderr."""
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "case.sir").write_text(source, encoding="utf-8")
    (root / "in.bin").write_bytes(struct.pack("<8i", *X) + struct.pack("<i", n))
    completed = subprocess.run(
      [
        str(SYMIRI),
        "--tensor-profile-version",
        "v3",
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
    return list(struct.unpack("<8i", (root / "out.bin").read_bytes()))


def expected(n):
  """n trips of acc += x, then acc + i, where i leaves the loop at n."""
  return [n * value + n for value in X]


@unittest.skipUnless(SYMIRI.is_file(), "symiri is unavailable")
class ParameterLoopProfileTests(unittest.TestCase):
  def check(self, source):
    return check_source(source, tensor_profile="v3")

  def test_a_ranged_parameter_bound_is_accepted(self):
    result = self.check(loop())
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_a_parameter_bound_needs_both_assumes(self):
    for assumes in ("  assume %n >= 0;", "  assume %n <= 8;", ""):
      with self.subTest(assumes=assumes):
        result = self.check(loop(assumes=assumes))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("assume", result.stdout + result.stderr)

  def test_a_range_beyond_eight_trips_is_refused(self):
    result = self.check(loop(assumes="  assume %n >= 0;\n  assume %n <= 9;"))
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("one to eight", result.stdout + result.stderr)

  def test_a_range_below_the_start_is_refused(self):
    result = self.check(loop(assumes="  assume %n >= 0;\n  assume %n <= 8;", start=1))
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("starts", result.stdout + result.stderr)

  def test_a_bound_that_is_not_an_i32_entry_parameter_is_refused(self):
    source = loop().replace("%n: i32", "%n: tensor [1] i32")
    result = self.check(source)
    self.assertNotEqual(result.returncode, 0)


@unittest.skipUnless(SYMIRI.is_file(), "symiri is unavailable")
class ParameterLoopP0Tests(unittest.TestCase):
  def test_p0_runs_zero_one_and_eight_trips(self):
    for n in (0, 1, 8):
      with self.subTest(n=n):
        self.assertEqual(run_p0(loop(), n), expected(n))

  def test_p0_refuses_an_argument_outside_the_assumed_range(self):
    self.assertIn("Assumption failed", run_p0(loop(), 9))


@unittest.skipUnless(SYMIRC.is_file(), "symirc is unavailable")
class ParameterLoopEmissionTests(unittest.TestCase):
  def emit(self, *extra):
    result, output = emit_cli(loop(), ("--tensor-profile-version", "v3", *extra))
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return output

  def test_the_loop_runs_to_the_loaded_argument(self):
    output = self.emit()
    self.assertIn("tl.load(n_ptr)", output)
    self.assertRegex(output, r"for _sir_i in tl\.range\(0, (_sir_\w+)\):")
    self.assertNotRegex(output, r"tl\.range\(0, \d+\)")

  def test_the_induction_variable_leaves_the_loop_at_the_argument(self):
    output = self.emit()
    bound = __import__("re").search(r"tl\.range\(0, (_sir_\w+)\)", output).group(1)
    self.assertIn(f"_sir_i = tl.full((), {bound}, tl.int32)", output)

  def test_a_runtime_loop_bound_option_adds_no_parameter_for_it(self):
    output = self.emit("--tensor-loop-bound", "runtime")
    self.assertIn("@triton.jit\ndef kernel(x_ptr, n_ptr, out_ptr):", output)


@unittest.skipUnless(
  SYMIRC.is_file() and np is not None, "symirc or numpy is unavailable"
)
class ParameterLoopExecutionTests(unittest.TestCase):
  def test_the_kernel_computes_p0s_output_for_zero_one_and_eight_trips(self):
    kernel = emit_cli(loop(), ("--tensor-profile-version", "v3"))[1]
    for n in (0, 1, 8):
      with self.subTest(n=n):
        words = run_shim(kernel, [X, [n]], 8, np.int32)
        self.assertEqual(
          [struct.unpack("<i", struct.pack("<I", word))[0] for word in words],
          expected(n),
        )


def _triton_interpreter_available():
  try:
    import torch  # noqa: F401
    import triton  # noqa: F401
  except Exception:
    return False
  return True


INTERPRETER_RUNNER = """\
import importlib.util
import json
import sys

import torch

spec = importlib.util.spec_from_file_location("kernel", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
x = torch.tensor(json.loads(sys.argv[2]), dtype=torch.int32)
n = torch.tensor([int(sys.argv[3])], dtype=torch.int32)
out = torch.zeros(8, dtype=torch.int32)
module.kernel[(1, 1, 1)](x, n, out)
print(json.dumps(out.tolist()))
"""


@unittest.skipUnless(
  SYMIRC.is_file() and _triton_interpreter_available(),
  "symirc or the Triton interpreter is unavailable",
)
class ParameterLoopInterpreterTests(unittest.TestCase):
  def test_triton_runs_zero_one_and_eight_trips(self):
    kernel = emit_cli(loop(), ("--tensor-profile-version", "v3"))[1]
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / "kernel.py").write_text(kernel, encoding="utf-8")
      (root / "runner.py").write_text(INTERPRETER_RUNNER, encoding="utf-8")
      for n in (0, 1, 8):
        with self.subTest(n=n):
          completed = subprocess.run(
            [
              sys.executable,
              str(root / "runner.py"),
              str(root / "kernel.py"),
              json.dumps(X),
              str(n),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            env={**os.environ, "TRITON_INTERPRET": "1"},
          )
          self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
          self.assertEqual(
            json.loads(completed.stdout.strip().splitlines()[-1]), expected(n)
          )


if __name__ == "__main__":
  unittest.main()
