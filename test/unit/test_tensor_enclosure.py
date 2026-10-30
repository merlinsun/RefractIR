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


if __name__ == "__main__":
  unittest.main()
