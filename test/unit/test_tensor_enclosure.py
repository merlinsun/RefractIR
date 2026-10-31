"""Enclosures of a free-form body's output over every summation order (plan W2)."""

import struct
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from test.lib import (
  tensor_free_form_case,
  tensor_free_form_inputs,
  tensor_target_reference,
)
from test.unit.tensor_support import SYMIRI

try:
  import numpy as np

  from test.lib import tensor_enclosure
except ImportError:  # pragma: no cover - numpy is a test-environment dependency
  np = tensor_enclosure = None

REGION = {
  "schema": "refractir.free-form-region.v1",
  "entry": "kernel",
  "profile": "tensor-v3",
  "dtype": "f32",
  "shape": [2, 8],
  "trip_count": 2,
  "state_count": 2,
  "num_warps": 1,
  "parameters": [
    {"name": "%window_buffer", "shape": [5, 8], "dtype": "f32"},
    {"name": "%x", "shape": [2, 8], "dtype": "f32"},
    {"name": "%y", "shape": [2, 8], "dtype": "f32"},
  ],
  "window_source": {"parameter": "%window_buffer", "shape": [5, 8], "dtype": "f32"},
  "window": {"rows": 2, "stride": 1, "active": 4, "source": "%window_buffer"},
}

HEADER = """\
fun @kernel(%window_buffer: tensor [5, 8] f32, %x: tensor [2, 8] f32, %y: tensor [2, 8] f32) : tensor [2, 8] f32 {
  let mut %i: i32 = undef;
  let mut %acc_x: tensor [2, 8] f32 = undef;
  let mut %acc_y: tensor [2, 8] f32 = undef;
  let mut %tile: tensor [2, 8] f32 = undef;
  let mut %a: tensor [2, 8] f32 = undef;
  let mut %s: tensor [2, 1] f32 = undef;
  let mut %out: tensor [2, 8] f32 = undef;
^entry:
  %acc_x = %x;
  %acc_y = %y;
  %i = 0;
  br ^header;
^header:
  br %i < 2, ^body, ^exit;
^body:
  %tile = tensor window %window_buffer, %i, 0.0, [2, 1, 4];
"""

FOOTER = """\
  %i = %i + 1;
  br ^header;
^exit:
  %out = tensor add %acc_x, %acc_y;
  ret %out;
}
"""

SPECIFIED = (
  HEADER
  + """\
  %a = tensor mul %acc_x, %tile;
  %acc_x = tensor max %a, %acc_y;
  %acc_y = tensor sub %acc_y, %tile;
"""
  + FOOTER
)

SUMMED = (
  HEADER
  + """\
  %s = tensor sum %acc_x, 1, keep;
  %acc_x = tensor add %s, %tile;
  %acc_y = tensor max %acc_y, %tile;
"""
  + FOOTER
)


def with_trip_parameter(source):
  """The same body, its loop run to an i32 `%trips` parameter ranged 0..2."""
  return (
    source.replace(
      "%y: tensor [2, 8] f32) : tensor", "%y: tensor [2, 8] f32, %trips: i32) : tensor"
    )
    .replace("^entry:\n", "^entry:\n  assume %trips >= 0;\n  assume %trips <= 2;\n")
    .replace("br %i < 2,", "br %i < %trips,")
  )


TRIP_REGION = {
  **REGION,
  "parameters": REGION["parameters"]
  + [{"name": "%trips", "shape": [], "dtype": "i32", "role": "trip_count"}],
  "trip_parameter": {"name": "%trips", "max": 2},
}


def proposals():
  base = tensor_free_form_inputs.window_base_proposal(REGION, group=0, column=0)
  return [base, *tensor_free_form_inputs.corner_probes(REGION, group=0, column=0)]


def f32(value):
  return struct.unpack("<f", struct.pack("<f", value))[0]


@unittest.skipUnless(np is not None, "numpy is unavailable")
class RoundingTests(unittest.TestCase):
  def test_directed_rounding_matches_nextafter(self):
    for value in (1.0, -3.5, 2.0**-140, 1e30, -(2.0**-126)):
      exact = f32(value)
      with self.subTest(value=value):
        up = np.nextafter(np.float32(exact), np.float32(np.inf))
        down = np.nextafter(np.float32(exact), np.float32(-np.inf))
        between = (Fraction(exact) + Fraction(float(up))) / 2 + Fraction(1, 2**200)
        self.assertEqual(tensor_enclosure.round_f32(between, "up"), Fraction(float(up)))
        self.assertEqual(tensor_enclosure.round_f32(between, "down"), Fraction(exact))
        self.assertEqual(
          tensor_enclosure.round_f32(Fraction(exact), "up"), Fraction(exact)
        )
        below = (Fraction(exact) + Fraction(float(down))) / 2 - Fraction(1, 2**200)
        self.assertEqual(
          tensor_enclosure.round_f32(below, "down"), Fraction(float(down))
        )

  def test_nearest_rounding_breaks_ties_to_even(self):
    self.assertEqual(tensor_enclosure.round_f32(Fraction(2**24 + 1)), 2**24)
    self.assertEqual(tensor_enclosure.round_f32(Fraction(2**24 + 3)), 2**24 + 4)

  def test_overflow_is_not_finite(self):
    with self.assertRaises(tensor_enclosure.NonFinite):
      tensor_enclosure.round_f32(Fraction(2**128))

  def test_ulps_count_f32_steps_and_identify_signed_zeros(self):
    self.assertEqual(tensor_enclosure.ulps(Fraction(1), Fraction(1)), 0)
    step = Fraction(float(np.nextafter(np.float32(1), np.float32(2))))
    self.assertEqual(tensor_enclosure.ulps(Fraction(1), step), 1)
    self.assertEqual(tensor_enclosure.ulps(Fraction(-(2**-149)), Fraction(2**-149)), 2)


@unittest.skipUnless(
  np is not None and SYMIRI.is_file(), "numpy or symiri is unavailable"
)
class EnclosureTests(unittest.TestCase):
  def run_both(self, source, proposal):
    with tempfile.TemporaryDirectory() as directory:
      case = tensor_free_form_case.build_bundle(
        Path(directory) / "bundle", region=REGION, source=source, proposal=proposal
      )
      enclosure = tensor_enclosure.enclose_bundle(case.path)
      observation = tensor_target_reference.observe(case, symiri=SYMIRI)
    return enclosure, observation

  def assert_contains(self, enclosure, observation):
    self.assertTrue(observation.completed, observation.reason + observation.stderr)
    p0 = struct.unpack(f"<{len(observation.output) // 4}f", observation.output)
    for value, low, high in zip(p0, enclosure.lo.flat, enclosure.hi.flat):
      self.assertLessEqual(low, Fraction(value))
      self.assertLessEqual(Fraction(value), high)

  def test_add_mul_max_enclosures_contain_p0_and_are_points(self):
    for proposal in proposals():
      with self.subTest(probe=proposal.probe):
        enclosure, observation = self.run_both(SPECIFIED, proposal)
        self.assert_contains(enclosure, observation)
        self.assertEqual(set(enclosure.widths()), {0})

  def test_a_sum_of_eight_terms_stays_within_its_reordering_bound(self):
    cancel = next(p for p in proposals() if p.probe == "i6-cancel")
    enclosure, observation = self.run_both(SUMMED, cancel)
    self.assert_contains(enclosure, observation)
    self.assertGreater(max(enclosure.widths()), 0)
    gamma = Fraction(7, 2**24) / (1 - Fraction(7, 2**24))
    terms = np.array([[Fraction(f32(2.0**23 + 3))] * 8], dtype=object)
    lane = tensor_enclosure.Interval(terms, terms)
    summed = tensor_enclosure.apply("sum", [lane], axis=1, keep=True)
    exact = sum(terms.flat)
    self.assertGreaterEqual(summed.lo.flat[0], exact - gamma * exact)
    self.assertLessEqual(summed.hi.flat[0], exact + gamma * exact)

  def test_an_integer_valued_body_has_width_zero(self):
    enclosure, observation = self.run_both(SUMMED, proposals()[0])
    self.assert_contains(enclosure, observation)
    self.assertEqual(set(enclosure.widths()), {0})

  def test_sum_enclosures_contain_p0_for_every_probe(self):
    for proposal in proposals():
      with self.subTest(probe=proposal.probe):
        self.assert_contains(*self.run_both(SUMMED, proposal))

  def test_a_sum_past_the_exact_range_is_widened(self):
    exact = tensor_enclosure.Interval.point(
      np.array([[Fraction(2**24 - 1), Fraction(1)]], dtype=object)
    )
    self.assertEqual(tensor_enclosure.apply("sum", [exact], axis=1).widths(), [0])
    past = tensor_enclosure.Interval.point(
      np.array([[Fraction(2**24), Fraction(1)]], dtype=object)
    )
    self.assertGreater(tensor_enclosure.apply("sum", [past], axis=1).widths()[0], 0)

  def test_an_interval_operand_widens_the_ops_that_consume_it(self):
    one, two = Fraction(1), Fraction(2)
    wide = tensor_enclosure.Interval(
      np.array([one], dtype=object), np.array([two], dtype=object)
    )
    point = tensor_enclosure.Interval(
      np.array([-two], dtype=object), np.array([-two], dtype=object)
    )
    product = tensor_enclosure.apply("mul", [wide, point])
    self.assertEqual((product.lo[0], product.hi[0]), (-4, -2))
    self.assertEqual(tensor_enclosure.apply("max", [wide, point]).widths(), [2**23])

  def test_a_trip_parameter_body_is_enclosed_for_each_trip_count(self):
    for source in (SPECIFIED, SUMMED):
      for trips in (0, 1, 2):
        with self.subTest(trips=trips, summed="sum" in source):
          probe = tensor_free_form_inputs.window_base_proposal(
            TRIP_REGION, group=0, column=0, trips=trips
          )
          with tempfile.TemporaryDirectory() as directory:
            case = tensor_free_form_case.build_bundle(
              Path(directory) / "bundle",
              region=TRIP_REGION,
              source=with_trip_parameter(source),
              proposal=probe,
            )
            enclosure = tensor_enclosure.enclose_bundle(case.path)
            observation = tensor_target_reference.observe(case, symiri=SYMIRI)
          self.assert_contains(enclosure, observation)
          self.assertEqual(set(enclosure.widths()), {0})

  def test_an_unsupported_statement_is_refused(self):
    with self.assertRaises(tensor_enclosure.Unsupported):
      tensor_enclosure.enclose(SPECIFIED.replace("tensor sub", "tensor div"), {})


def f32_bits(value):
  return struct.pack("<f", value)


@unittest.skipUnless(np is not None, "numpy is unavailable")
class LaneCheckTests(unittest.TestCase):
  """One output's lanes against its enclosure (enclosure-oracle plan, E1)."""

  def interval(self, pairs):
    lo = np.array([Fraction(low) for low, _ in pairs], dtype=object)
    hi = np.array([Fraction(high) for _, high in pairs], dtype=object)
    return tensor_enclosure.Interval(lo, hi)

  def classes(self, pairs, values):
    report = tensor_enclosure.check_lanes(
      self.interval(pairs), b"".join(f32_bits(value) for value in values)
    )
    return report

  def test_a_point_lane_with_other_bits_is_a_point_mismatch(self):
    report = self.classes([(3, 3), (1, 1)], [3.0, f32(1.0 + 2.0**-23)])
    self.assertEqual(report["lanes"], ["point", "point_mismatch"])
    self.assertEqual([v["index"] for v in report["violations"]], [1])

  def test_a_value_one_step_outside_a_widened_lane_is_outside(self):
    low, high = Fraction(1), Fraction(f32(1.0 + 4 * 2.0**-23))
    below = float(np.nextafter(np.float32(1.0), np.float32(0.0)))
    report = self.classes([(low, high)] * 4, [1.0, float(high), 1.0 + 2.0**-23, below])
    self.assertEqual(report["lanes"], ["inside", "inside", "inside", "outside"])
    self.assertEqual(report["violations"][0]["class"], "outside")

  def test_a_zero_point_lane_accepts_either_zero_sign(self):
    report = self.classes([(0, 0), (0, 0)], [0.0, -0.0])
    self.assertEqual(report["lanes"], ["point", "point_zero"])
    self.assertEqual(report["violations"], [])

  def test_output_bytes_of_the_wrong_length_are_refused(self):
    with self.assertRaises(ValueError):
      tensor_enclosure.check_lanes(self.interval([(1, 1), (2, 2)]), f32_bits(1.0))

  def test_the_report_counts_point_and_widened_lanes_and_the_widest(self):
    step = Fraction(f32(1.0 + 2.0**-23))
    report = self.classes([(1, 1), (1, step), (2, 2)], [1.0, 1.0, 2.0])
    self.assertEqual(
      (report["point_lanes"], report["widened_lanes"], report["widest_ulps"]), (2, 1, 1)
    )

  def test_a_non_finite_value_is_outside_every_lane(self):
    report = self.classes([(1, 1), (1, 2)], [float("inf"), float("nan")])
    self.assertEqual(report["lanes"], ["point_mismatch", "outside"])


@unittest.skipUnless(np is not None, "numpy is unavailable")
class EncloseValuesTests(unittest.TestCase):
  def test_input_bytes_that_do_not_fit_the_parameters_are_refused(self):
    probe = tensor_free_form_inputs.window_base_proposal(
      TRIP_REGION, group=0, column=0, trips=2
    )
    with self.assertRaises(ValueError):
      tensor_enclosure.enclose_values(
        with_trip_parameter(SUMMED), TRIP_REGION["parameters"], probe.data + b"\0\0\0\0"
      )

  def test_values_give_the_bundles_enclosure(self):
    probe = tensor_free_form_inputs.window_base_proposal(
      TRIP_REGION, group=0, column=0, trips=2
    )
    source = with_trip_parameter(SUMMED)
    with tempfile.TemporaryDirectory() as directory:
      case = tensor_free_form_case.build_bundle(
        Path(directory) / "bundle", region=TRIP_REGION, source=source, proposal=probe
      )
      from_bundle = tensor_enclosure.enclose_bundle(case.path)
    from_values = tensor_enclosure.enclose_values(
      source, TRIP_REGION["parameters"], probe.data
    )
    self.assertEqual(list(from_values.lo.flat), list(from_bundle.lo.flat))
    self.assertEqual(list(from_values.hi.flat), list(from_bundle.hi.flat))


if __name__ == "__main__":
  unittest.main()


DOT_SOURCE = """\
fun @kernel(%a: tensor [16, 16] f32, %b: tensor [16, 16] f32) : tensor [16, 16] f32 {
  let mut %out: tensor [16, 16] f32 = undef;
^entry:
  %out = tensor dot %a, %b;
  ret %out;
}
"""


def dot_point(values, shape=(16, 16)):
  array = np.array([Fraction(value) for value in values], dtype=object).reshape(shape)
  return tensor_enclosure.Interval.point(array)


@unittest.skipUnless(np is not None, "numpy is unavailable")
class DotEnclosureTests(unittest.TestCase):
  """f32 `tensor dot` over every accumulation order (plan L1)."""

  def test_an_exact_inner_product_is_a_point(self):
    a = [(i % 7) - 3 for i in range(256)]
    b = [(i % 5) - 2 for i in range(256)]
    result = tensor_enclosure.apply("dot", [dot_point(a), dot_point(b)])
    expected = np.array(a).reshape(16, 16) @ np.array(b).reshape(16, 16)
    self.assertEqual(set(result.widths()), {0})
    self.assertEqual([int(v) for v in result.lo.flat], [int(v) for v in expected.flat])

  def test_a_rounded_inner_product_holds_both_orders(self):
    a = [0] * 256
    a[0] = 2**24
    for k in range(1, 16):
      a[k] = 1
    result = tensor_enclosure.apply("dot", [dot_point(a), dot_point([1] * 256)])
    # Accumulating in order loses every one; P0 rounds 2^24 + 15 once.
    self.assertLessEqual(result.lo[0, 0], 2**24)
    self.assertGreaterEqual(result.hi[0, 0], 2**24 + 16)
    self.assertEqual(result.widths()[16:], [0] * 240)

  def test_the_bound_is_gamma_k_of_the_absolute_products(self):
    a = dot_point([Fraction(2**23 + 3)] * 256)
    b = dot_point([1] * 256)
    result = tensor_enclosure.apply("dot", [a, b])
    exact = 16 * (2**23 + 3)
    gamma = Fraction(16, 2**24) / (1 - Fraction(16, 2**24))
    self.assertGreaterEqual(result.lo[0, 0], exact - gamma * exact)
    self.assertLessEqual(result.hi[0, 0], exact + gamma * exact)
    self.assertGreater(result.widths()[0], 0)

  def test_an_interval_operand_takes_every_corner_product(self):
    one, two = Fraction(1), Fraction(2)
    wide = tensor_enclosure.Interval(
      np.full((16, 16), -two, dtype=object), np.full((16, 16), one, dtype=object)
    )
    result = tensor_enclosure.apply("dot", [wide, dot_point([1] * 256)])
    low, high = result.lo[0, 0], result.hi[0, 0]
    self.assertTrue(-33 < low <= -32 and 16 <= high < 17, (low, high))

  def test_a_body_with_dot_is_enclosed_and_contains_p0(self):
    a = [((i * 7) % 11) / 4 for i in range(256)]
    b = [((i * 3) % 13) / 8 - 0.5 for i in range(256)]
    inputs = {
      "%a": np.array([Fraction(v) for v in a], dtype=object).reshape(16, 16),
      "%b": np.array([Fraction(v) for v in b], dtype=object).reshape(16, 16),
    }
    result = tensor_enclosure.enclose(DOT_SOURCE, inputs)
    expected = np.array(a).reshape(16, 16) @ np.array(b).reshape(16, 16)
    for low, high, value in zip(result.lo.flat, result.hi.flat, expected.flat):
      self.assertLessEqual(low, Fraction(float(np.float32(value))))
      self.assertGreaterEqual(high, Fraction(float(np.float32(value))))


@unittest.skipUnless(np is not None, "numpy is unavailable")
class NarrowEnclosureTests(unittest.TestCase):
  """`tensor narrow` is one correctly rounded conversion (plan L2)."""

  def test_a_point_narrows_to_the_format_and_an_interval_to_its_rounded_ends(self):
    values = [0.1, 1 + 3 * 2**-11, 65519.0, -(2.0**-30)]
    point = tensor_enclosure.Interval.point(
      np.array([Fraction(v) for v in values], dtype=object)
    )
    f16 = tensor_enclosure.apply("narrow f16", [point])
    self.assertEqual(
      [float(v) for v in f16.lo.flat], [float(np.float16(v)) for v in values]
    )
    self.assertEqual(set(f16.widths()), {0})
    wide = tensor_enclosure.Interval(
      np.array([Fraction(1)], dtype=object),
      np.array([Fraction(1 + 2**-9)], dtype=object),
    )
    bf16 = tensor_enclosure.apply("narrow bf16", [wide])
    self.assertEqual((bf16.lo[0], bf16.hi[0]), (1, 1))
    with self.assertRaises(tensor_enclosure.NonFinite):
      tensor_enclosure.apply(
        "narrow f16",
        [tensor_enclosure.Interval.point(np.array([Fraction(65520)], dtype=object))],
      )


@unittest.skipUnless(np is not None, "numpy is unavailable")
class WhereEnclosureTests(unittest.TestCase):
  """`tensor where` picks an arm, or either when the comparison is open (L3)."""

  def lanes(self, pairs):
    lo = np.array([Fraction(low) for low, _ in pairs], dtype=object)
    hi = np.array([Fraction(high) for _, high in pairs], dtype=object)
    return tensor_enclosure.Interval(lo, hi)

  def test_a_decided_comparison_takes_its_arm_and_an_open_one_the_hull(self):
    a = self.lanes([(0, 0), (5, 5), (1, 3)])
    b = self.lanes([(1, 1), (1, 1), (2, 2)])
    x = self.lanes([(10, 10), (10, 10), (10, 10)])
    y = self.lanes([(-4, -2), (-4, -2), (-4, -2)])
    result = tensor_enclosure.apply("where lt", [a, b, x, y])
    self.assertEqual(
      list(zip(result.lo.flat, result.hi.flat)), [(10, 10), (-4, -2), (-4, 10)]
    )
    self.assertEqual(
      [lo for lo in tensor_enclosure.apply("where eq", [a, a, x, y]).lo.flat][:2],
      [10, 10],
    )

  def test_a_body_with_where_is_enclosed(self):
    source = """\
fun @kernel(%a: tensor [4] f32, %b: tensor [4] f32) : tensor [4] f32 {
  let mut %out: tensor [4] f32 = undef;
^entry:
  %out = tensor where le %a, %b, %b, %a;
  ret %out;
}
"""
    inputs = {
      "%a": np.array([Fraction(v) for v in (1, 2, 3, 4)], dtype=object),
      "%b": np.array([Fraction(v) for v in (4, 2, 1, 0)], dtype=object),
    }
    result = tensor_enclosure.enclose(source, inputs)
    self.assertEqual([int(v) for v in result.lo.flat], [4, 2, 3, 4])


@unittest.skipUnless(np is not None, "numpy is unavailable")
class CumsumEnclosureTests(unittest.TestCase):
  """Each prefix of a `tensor cumsum` is enclosed like a sum of its terms (L6)."""

  def test_exact_prefixes_are_points_and_rounded_ones_are_widened(self):
    exact = tensor_enclosure.Interval.point(
      np.array([[Fraction(v) for v in (1, 2, 3, 4)]], dtype=object)
    )
    result = tensor_enclosure.apply("cumsum", [exact], axis=1)
    self.assertEqual([int(v) for v in result.lo.flat], [1, 3, 6, 10])
    self.assertEqual(set(result.widths()), {0})
    rounded = tensor_enclosure.Interval.point(
      np.array([[Fraction(2**24), Fraction(1), Fraction(1)]], dtype=object)
    )
    result = tensor_enclosure.apply("cumsum", [rounded], axis=1)
    self.assertEqual(result.widths()[0], 0)
    self.assertLessEqual(result.lo[0, 1], 2**24)
    self.assertGreaterEqual(result.hi[0, 2], 2**24 + 2)
    self.assertGreater(result.widths()[2], 0)

  def test_a_body_with_cumsum_is_enclosed(self):
    source = """\
fun @kernel(%a: tensor [2, 4] f32) : tensor [2, 4] f32 {
  let mut %out: tensor [2, 4] f32 = undef;
^entry:
  %out = tensor cumsum %a, 0;
  ret %out;
}
"""
    inputs = {
      "%a": np.array([Fraction(v) for v in range(8)], dtype=object).reshape(2, 4)
    }
    result = tensor_enclosure.enclose(source, inputs)
    self.assertEqual([int(v) for v in result.lo.flat], [0, 1, 2, 3, 4, 6, 8, 10])
