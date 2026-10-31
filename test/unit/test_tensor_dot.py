"""`tensor dot`: typing, profile, P0, and Triton emission (plan L1)."""

import struct
import unittest

from test.unit.tensor_support import SYMIRC, SYMIRI, check_source, run_typed
from test.unit.test_tensor_emit import emit_cli


def dot_program(m, k, n, element="i32"):
  return f"""\
fun @kernel(%a: tensor [{m}, {k}] {element}, %b: tensor [{k}, {n}] {element}) : tensor [{m}, {n}] {element} {{
  let mut %out: tensor [{m}, {n}] {element} = undef;
^entry:
  %out = tensor dot %a, %b;
  ret %out;
}}
"""


def matmul(a, b, m, k, n):
  return [
    sum(a[i * k + t] * b[t * n + j] for t in range(k))
    for i in range(m)
    for j in range(n)
  ]


@unittest.skipUnless(SYMIRI.is_file(), "symiri is unavailable")
class DotTypingTests(unittest.TestCase):
  def test_dot_of_matching_tiles_checks_under_v2_and_v3(self):
    for profile in ("v2", "v3"):
      with self.subTest(profile=profile):
        result = check_source(dot_program(16, 32, 16), tensor_profile=profile)
        self.assertEqual(result.returncode, 0, result.stderr)

  def test_dot_rejects_mismatched_or_non_matrix_operands(self):
    mismatched = dot_program(16, 16, 16).replace(
      "%b: tensor [16, 16] i32", "%b: tensor [32, 16] i32"
    )
    vector = """\
fun @kernel(%a: tensor [16] i32, %b: tensor [16] i32) : tensor [16] i32 {
  let mut %out: tensor [16] i32 = undef;
^entry:
  %out = tensor dot %a, %b;
  ret %out;
}
"""
    for source in (mismatched, vector):
      result = check_source(source, tensor_profile="v3")
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("tensor-dot", result.stderr)

  def test_profile_requires_power_of_two_extents_of_at_least_16(self):
    for m, k, n in ((8, 16, 16), (16, 8, 16), (16, 16, 8), (16, 24, 16)):
      with self.subTest(shape=(m, k, n)):
        result = check_source(dot_program(m, k, n), tensor_profile="v3")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tensor-profile: dot", result.stderr)
    for profile in ("v1", "v4"):
      result = check_source(dot_program(16, 16, 16), tensor_profile=profile)
      self.assertNotEqual(result.returncode, 0, profile)


@unittest.skipUnless(SYMIRI.is_file(), "symiri is unavailable")
class DotReferenceTests(unittest.TestCase):
  def test_i32_dot_is_exact_and_operands_must_fit_int8(self):
    m = k = n = 16
    a = [((i * 7) % 255) - 127 for i in range(m * k)]
    b = [((i * 11) % 256) - 128 for i in range(k * n)]
    result, output = run_typed(
      dot_program(m, k, n),
      struct.pack(f"<{m * k + k * n}i", *a, *b),
      tensor_profile="v3",
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(output, struct.pack(f"<{m * n}i", *matmul(a, b, m, k, n)))

    a[5] = 128
    result, _ = run_typed(
      dot_program(m, k, n),
      struct.pack(f"<{m * k + k * n}i", *a, *b),
      tensor_profile="v3",
    )
    self.assertEqual(result.returncode, 5, result.stderr)
    self.assertIn("UB: tensor dot operand outside int8", result.stderr)

  def test_i32_dot_indexes_rectangular_tiles(self):
    for m, k, n in ((16, 32, 16), (16, 16, 32)):
      with self.subTest(shape=(m, k, n)):
        a = [((i * 5) % 17) - 8 for i in range(m * k)]
        b = [((i * 3) % 13) - 6 for i in range(k * n)]
        result, output = run_typed(
          dot_program(m, k, n),
          struct.pack(f"<{m * k + k * n}i", *a, *b),
          tensor_profile="v3",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output, struct.pack(f"<{m * n}i", *matmul(a, b, m, k, n)))

  def test_f32_dot_is_the_correctly_rounded_inner_product(self):
    m = k = n = 16
    # Row 0 of a is 2^24 followed by ones; b is all ones. Accumulating in f32
    # loses every one (2^24 + 1 rounds to 2^24); the exact sum 2^24 + 15
    # rounds to 2^24 + 16.
    a = [0.0] * (m * k)
    a[0] = float(2**24)
    for t in range(1, k):
      a[t] = 1.0
    b = [1.0] * (k * n)
    result, output = run_typed(
      dot_program(m, k, n, "f32"),
      struct.pack(f"<{m * k + k * n}f", *a, *b),
      tensor_profile="v3",
      reference_bits=128,
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    values = struct.unpack(f"<{m * n}f", output)
    self.assertEqual(values[:n], (float(2**24 + 16),) * n)
    self.assertEqual(values[n:], (0.0,) * (m * n - n))


@unittest.skipUnless(SYMIRC.is_file(), "symirc is unavailable")
class DotEmissionTests(unittest.TestCase):
  def test_dot_lowers_to_tl_dot_with_a_fixed_precision(self):
    result, output = emit_cli(
      dot_program(16, 16, 16), ("--tensor-profile-version", "v3")
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertRegex(
      output,
      r"tl\.dot\(\w+\.to\(tl\.int8\), \w+\.to\(tl\.int8\), out_dtype=tl\.int32\)",
    )
    result, output = emit_cli(
      dot_program(16, 16, 16, "f32"), ("--tensor-profile-version", "v3")
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertRegex(
      output, r"tl\.dot\(\w+, \w+, input_precision=\"ieee\", out_dtype=tl\.float32\)"
    )


if __name__ == "__main__":
  unittest.main()
