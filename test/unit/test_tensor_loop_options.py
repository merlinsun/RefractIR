"""Counted-loop emission options: runtime bounds and loop hints; no GPU is required.

With `--tensor-loop-bound runtime` each counted loop takes its bound from a
kernel parameter whose default is the SIR literal, so the launcher need not
change and the semantics are the literal program's. Triton no longer sees a
constant trip count: the parameter is excluded from specialization, so even a
bound of one stays a runtime value. `--tensor-loop-hint` adds one `tl.range`
keyword to every counted loop; a hint changes how Triton schedules the loop,
never what it computes.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test.unit.tensor_support import SYMIRC
from test.unit.test_tensor_emit import emit_cli
from test.unit.triton_numpy_shim import np
from test.unit.triton_numpy_shim import run as run_shim

RUNTIME = ("--tensor-profile-version", "v3", "--tensor-loop-bound", "runtime")

ACCUMULATE = """\
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
  %acc = tensor add %acc, %i;
  ret %acc;
}}
"""

NESTED = """\
fun @kernel(%x: tensor [8] i32) : tensor [8] i32 {
  let mut %outer: i32 = undef;
  let mut %inner: i32 = undef;
  let mut %acc: tensor [8] i32 = undef;
^entry:
  %acc = tensor mul %x, 0;
  %outer = 0;
  br ^outer_header;
^outer_header:
  br %outer < 2, ^outer_body, ^exit;
^outer_body:
  %inner = 0;
  br ^inner_header;
^inner_header:
  br %inner < 3, ^inner_body, ^outer_latch;
^inner_body:
  %acc = tensor add %acc, %x;
  %inner = %inner + 1;
  br ^inner_header;
^outer_latch:
  %outer = %outer + 1;
  br ^outer_header;
^exit:
  ret %acc;
}
"""

VALUES = [1, -2, 3, 0, 5, 7, -9, 4]


@unittest.skipUnless(SYMIRC.is_file(), "symirc is unavailable")
class RuntimeLoopBoundEmissionTests(unittest.TestCase):
  def emit(self, source, arguments=RUNTIME):
    result, output = emit_cli(source, arguments)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return output

  def test_bound_is_a_kernel_parameter_defaulting_to_the_literal(self):
    output = self.emit(ACCUMULATE.format(trips=4))
    self.assertIn("def kernel(x_ptr, out_ptr, _sir_i_bound=4):", output)

  def test_range_ends_at_the_parameter_instead_of_the_literal(self):
    output = self.emit(ACCUMULATE.format(trips=4))
    self.assertIn("for _sir_i in tl.range(0, _sir_i_bound):", output)
    self.assertNotIn("tl.range(0, 4)", output)

  def test_bound_parameter_is_excluded_from_specialization(self):
    output = self.emit(ACCUMULATE.format(trips=1))
    self.assertIn('@triton.jit(do_not_specialize=["_sir_i_bound"])\n', output)

  def test_induction_variable_leaves_the_loop_at_the_parameter(self):
    output = self.emit(ACCUMULATE.format(trips=4))
    self.assertIn("_sir_i = tl.full((), _sir_i_bound, tl.int32)", output)
    self.assertNotIn("tl.full((), 4,", output)

  def test_nested_loops_take_one_parameter_each(self):
    output = self.emit(NESTED)
    self.assertIn(
      "def kernel(x_ptr, out_ptr, _sir_outer_bound=2, _sir_inner_bound=3):", output
    )
    self.assertIn(
      '@triton.jit(do_not_specialize=["_sir_outer_bound", "_sir_inner_bound"])',
      output,
    )

  def test_literal_bounds_remain_the_default(self):
    output = self.emit(ACCUMULATE.format(trips=4), ("--tensor-profile-version", "v3"))
    self.assertIn("@triton.jit\ndef kernel(x_ptr, out_ptr):", output)
    self.assertIn("for _sir_i in tl.range(0, 4):", output)

  def test_a_loop_free_kernel_is_unchanged_by_runtime_bounds(self):
    source = ACCUMULATE.format(trips=4)
    loop_free = source[: source.index("  %i = 0;")] + "  ret %acc;\n}\n"
    loop_free = loop_free.replace("  let mut %i: i32 = undef;\n", "")
    self.assertEqual(
      self.emit(loop_free), self.emit(loop_free, ("--tensor-profile-version", "v3"))
    )
    self.assertIn("@triton.jit\ndef kernel(x_ptr, out_ptr):", self.emit(loop_free))

  def test_an_unknown_loop_bound_mode_is_refused(self):
    result, _ = emit_cli(
      ACCUMULATE.format(trips=4),
      ("--tensor-profile-version", "v3", "--tensor-loop-bound", "symbolic"),
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("literal or runtime", result.stdout + result.stderr)


@unittest.skipUnless(
  SYMIRC.is_file() and np is not None, "symirc or numpy is unavailable"
)
class RuntimeLoopBoundExecutionTests(unittest.TestCase):
  """The runtime-bound kernel computes the literal kernel's output."""

  def outputs(self, source, **arguments):
    literal = emit_cli(source, ("--tensor-profile-version", "v3"))[1]
    runtime = emit_cli(source, RUNTIME)[1]
    return (
      run_shim(literal, [VALUES], 8, np.int32),
      run_shim(runtime, [VALUES], 8, np.int32, **arguments),
    )

  def test_default_bound_matches_the_literal_kernel(self):
    for trips in (1, 2, 4):
      with self.subTest(trips=trips):
        literal, runtime = self.outputs(ACCUMULATE.format(trips=trips))
        self.assertEqual(runtime, literal)

  def test_nested_default_bounds_match_the_literal_kernel(self):
    literal, runtime = self.outputs(NESTED)
    self.assertEqual(runtime, literal)

  def test_a_passed_bound_runs_that_many_trips(self):
    _, runtime = self.outputs(ACCUMULATE.format(trips=4), _sir_i_bound=1)
    literal_one, _ = self.outputs(ACCUMULATE.format(trips=1))
    self.assertEqual(runtime, literal_one)


@unittest.skipUnless(SYMIRC.is_file(), "symirc is unavailable")
class LoopHintEmissionTests(unittest.TestCase):
  def emit(self, *hint, trips=4):
    result, output = emit_cli(
      ACCUMULATE.format(trips=trips), ("--tensor-profile-version", "v3", *hint)
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return output

  def test_a_num_stages_hint_reaches_the_range(self):
    output = self.emit("--tensor-loop-hint", "num_stages=2")
    self.assertIn("for _sir_i in tl.range(0, 4, num_stages=2):", output)

  def test_an_unroll_hint_reaches_the_range(self):
    output = self.emit("--tensor-loop-hint", "loop_unroll_factor=2")
    self.assertIn("for _sir_i in tl.range(0, 4, loop_unroll_factor=2):", output)

  def test_a_flatten_hint_reaches_the_range(self):
    output = self.emit("--tensor-loop-hint", "flatten")
    self.assertIn("for _sir_i in tl.range(0, 4, flatten=True):", output)

  def test_a_hint_combines_with_a_runtime_bound(self):
    output = self.emit(
      "--tensor-loop-hint", "loop_unroll_factor=2", "--tensor-loop-bound", "runtime"
    )
    self.assertIn(
      "for _sir_i in tl.range(0, _sir_i_bound, loop_unroll_factor=2):", output
    )

  def test_every_counted_loop_takes_the_hint(self):
    result, output = emit_cli(
      NESTED, ("--tensor-profile-version", "v3", "--tensor-loop-hint", "num_stages=3")
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("for _sir_outer in tl.range(0, 2, num_stages=3):", output)
    self.assertIn("for _sir_inner in tl.range(0, 3, num_stages=3):", output)

  def test_an_unknown_or_out_of_range_hint_is_refused(self):
    for hint in (
      "warp_specialize=1",
      "num_stages=0",
      "loop_unroll_factor=x",
      "flatten=1",
    ):
      with self.subTest(hint=hint):
        result, _ = emit_cli(
          ACCUMULATE.format(trips=4),
          ("--tensor-profile-version", "v3", "--tensor-loop-hint", hint),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--tensor-loop-hint", result.stdout + result.stderr)


@unittest.skipUnless(
  SYMIRC.is_file() and np is not None, "symirc or numpy is unavailable"
)
class LoopHintExecutionTests(unittest.TestCase):
  def test_a_hinted_kernel_computes_the_unhinted_output(self):
    source = ACCUMULATE.format(trips=3)
    plain = emit_cli(source, ("--tensor-profile-version", "v3"))[1]
    for hint in ("num_stages=2", "loop_unroll_factor=2", "flatten"):
      with self.subTest(hint=hint):
        hinted = emit_cli(
          source, ("--tensor-profile-version", "v3", "--tensor-loop-hint", hint)
        )[1]
        self.assertEqual(
          run_shim(hinted, [VALUES], 8, np.int32),
          run_shim(plain, [VALUES], 8, np.int32),
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
out = torch.zeros(8, dtype=torch.int32)
module.kernel[(1, 1, 1)](x, out)
print(json.dumps(out.tolist()))
"""


@unittest.skipUnless(
  SYMIRC.is_file() and _triton_interpreter_available(),
  "symirc or the Triton interpreter is unavailable",
)
class RuntimeLoopBoundInterpreterTests(unittest.TestCase):
  """Triton's interpreter accepts the defaulted, unspecialized bound."""

  def run_kernel(self, arguments, source):
    result, kernel = emit_cli(source, arguments)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / "kernel.py").write_text(kernel, encoding="utf-8")
      (root / "runner.py").write_text(INTERPRETER_RUNNER, encoding="utf-8")
      completed = subprocess.run(
        [
          sys.executable,
          str(root / "runner.py"),
          str(root / "kernel.py"),
          json.dumps(VALUES),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env={**os.environ, "TRITON_INTERPRET": "1"},
      )
      self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
      return json.loads(completed.stdout.strip().splitlines()[-1])

  def test_runtime_bounds_match_literal_bounds(self):
    for source in (ACCUMULATE.format(trips=1), ACCUMULATE.format(trips=4), NESTED):
      with self.subTest(source=source.splitlines()[8]):
        self.assertEqual(
          self.run_kernel(RUNTIME, source),
          self.run_kernel(("--tensor-profile-version", "v3"), source),
        )


if __name__ == "__main__":
  unittest.main()
