"""Free-form loops over general ranges (generation plan G2); no GPU is required.

A free-form loop may run over `range(start, stop, step)`, each endpoint a
literal or a ranged i32 parameter, with an outer loop over a range of its own.
Its windows may index any row: a position outside `[0, active)` reads the fill,
so a loop that walks rows in reverse or from below zero stays defined.
"""

import os
import struct
import subprocess
import unittest
from pathlib import Path

from test.unit.tensor_support import SYMIRC, SYMIRI, check_source, run_typed
from test.unit.test_tensor_emit import emit_cli
from test.unit.triton_numpy_shim import np
from test.unit.triton_numpy_shim import run as run_shim

FREE_FORM_API = Path(
  os.environ.get("REFRACTIR_TENSOR_FREE_FORM_API", "build/bin/tensor_free_form_api")
)

X = [1.0, 2.0, 3.0, 4.0]
SOURCE = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
Y = [0.5, -1.0, 2.0, 0.25]


def api(name):
  return subprocess.run(
    [str(FREE_FORM_API), name], text=True, capture_output=True, check=False, timeout=30
  )


def window_expected(lo, hi):
  """`acc_x = x + sum of the [2, 2] tiles read at each i in range(lo, hi, -1)`."""
  acc = list(X)
  for i in range(lo, hi, -1):
    for row in range(2):
      position = i + row
      for column in range(2):
        value = SOURCE[position * 2 + column] if 0 <= position < 4 else 0.0
        acc[row * 2 + column] += value
  return acc


def nested_expected(n, m):
  """The three-state recurrence run once per inner trip of every outer trip."""
  x, y, z = list(X), list(Y), [0.0] * 4
  trips = len(range(0, n, 2)) * len(range(m, -1, -1))
  for _ in range(trips):
    s01 = [a + b for a, b in zip(x, y)]
    x, y, z = (
      [a + b for a, b in zip(s01, x)],
      [a + b for a, b in zip(s01, y)],
      [a + b + b for a, b in zip(s01, z)],
    )
  return [a + b + c for a, b, c in zip(x, y, z)]


def floats(data):
  return list(struct.unpack(f"<{len(data) // 4}f", data))


@unittest.skipUnless(FREE_FORM_API.is_file(), "free-form API binary is unavailable")
class RangeContractTests(unittest.TestCase):
  def run_contract(self, name):
    result = api(name)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_a_range_refuses_a_trip_parameter(self):
    self.run_contract("range-refuses-a-trip-parameter")

  def test_a_range_refuses_a_zero_step(self):
    self.run_contract("range-refuses-a-zero-step")

  def test_a_range_refuses_an_empty_parameter_range(self):
    self.run_contract("range-refuses-an-empty-parameter-range")

  def test_a_range_window_may_reach_past_the_source(self):
    self.run_contract("range-window-may-reach-past-the-source")


@unittest.skipUnless(
  FREE_FORM_API.is_file() and SYMIRI.is_file(), "free-form API or symiri is unavailable"
)
class RangeWindowTests(unittest.TestCase):
  def source(self):
    result = api("range-window-source")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return result.stdout

  def run_p0(self, lo, hi):
    data = (
      struct.pack("<4f", *X) + struct.pack("<8f", *SOURCE) + struct.pack("<2i", lo, hi)
    )
    result, output = run_typed(self.source(), data, tensor_profile="v5")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return floats(output)

  def test_the_range_window_loop_is_a_v5_program_and_not_a_v3_program(self):
    source = self.source()
    self.assertEqual(check_source(source, tensor_profile="v5").returncode, 0)
    self.assertNotEqual(check_source(source, tensor_profile="v3").returncode, 0)
    self.assertIn("br %i > %hi, ^body, ^exit;", source)
    self.assertIn("%i = %i + -1;", source)

  def test_a_negative_step_reads_the_rows_in_reverse(self):
    self.assertEqual(self.run_p0(3, -1), window_expected(3, -1))

  def test_a_row_index_of_minus_one_reads_the_fill(self):
    self.assertEqual(self.run_p0(-1, -2), window_expected(-1, -2))

  def test_an_empty_range_leaves_the_state_unchanged(self):
    self.assertEqual(self.run_p0(2, 2), X)


@unittest.skipUnless(
  FREE_FORM_API.is_file() and SYMIRC.is_file(), "free-form API or symirc is unavailable"
)
class RangeWindowEmissionTests(unittest.TestCase):
  def test_a_window_whose_index_may_be_negative_masks_below_zero(self):
    source = api("range-window-source").stdout
    result, output = emit_cli(source, ("--tensor-profile-version", "v5"))
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertRegex(output, r"\(_sir_window_lin\w* >= 0\)")

  def test_a_counted_window_keeps_its_emission(self):
    source = api("window-assembled-source").stdout
    result, output = emit_cli(source, ("--tensor-profile-version", "v2"))
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertNotIn(">= 0)", output)

  @unittest.skipUnless(np is not None, "numpy is unavailable")
  def test_the_emitted_kernel_computes_p0s_window_output(self):
    source = api("range-window-source").stdout
    kernel = emit_cli(source, ("--tensor-profile-version", "v5"))[1]
    for lo, hi in ((3, -1), (-1, -2), (4, -3)):
      with self.subTest(lo=lo, hi=hi):
        words = run_shim(kernel, [X, SOURCE, [lo], [hi]], 4, np.float32)
        values = [struct.unpack("<f", struct.pack("<I", word))[0] for word in words]
        self.assertEqual(values, window_expected(lo, hi))


@unittest.skipUnless(
  FREE_FORM_API.is_file() and SYMIRI.is_file(), "free-form API or symiri is unavailable"
)
class RangeNestedTests(unittest.TestCase):
  def test_nested_runtime_ranges_assemble_into_a_v5_program(self):
    result = api("range-nested-source")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    source = result.stdout
    self.assertEqual(check_source(source, tensor_profile="v5").returncode, 0)
    self.assertRegex(source, r"%n: i32, %m: i32")
    for n, m in ((6, 3), (0, 3), (5, 0)):
      with self.subTest(n=n, m=m):
        data = (
          struct.pack("<4f", *X) + struct.pack("<4f", *Y) + struct.pack("<2i", n, m)
        )
        run, output = run_typed(source, data, tensor_profile="v5")
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(floats(output), nested_expected(n, m))


if __name__ == "__main__":
  unittest.main()
