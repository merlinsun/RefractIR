"""Inputs steered toward floating-point properties, confirmed by P0 (plan W4b)."""

import math
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

from test.lib import (
  tensor_free_form_case,
  tensor_free_form_inputs,
  tensor_target_reference,
)
from test.unit.tensor_support import SYMIRI

try:
  import numpy
except ImportError:  # pragma: no cover - numpy is a test-environment dependency
  numpy = None
else:
  from test.unit.test_tensor_enclosure import REGION, SPECIFIED, SUMMED, proposals

  PARAMETERS = [(record["name"], record["shape"]) for record in REGION["parameters"]]

try:
  from test.lib import tensor_steering
except ImportError:
  tensor_steering = None

SQUARE = """\
fun @kernel(%x: tensor [8] f32) : tensor [8] f32 {
  let mut %out: tensor [8] f32 = undef;
^entry:
  %out = tensor mul %x, %x;
  ret %out;
}
"""


PRODUCT = """\
fun @kernel(%x: tensor [2, 8] f32, %y: tensor [2, 8] f32) : tensor [2, 8] f32 {
  let mut %out: tensor [2, 8] f32 = undef;
^entry:
  %out = tensor mul %x, %y;
  ret %out;
}
"""
PRODUCT_REGION = (
  {
    **REGION,
    "parameters": [
      {"name": "%x", "shape": [2, 8], "dtype": "f32"},
      {"name": "%y", "shape": [2, 8], "dtype": "f32"},
    ],
  }
  if numpy is not None
  else None
)


def values_of(proposal):
  return list(struct.unpack(f"<{len(proposal.data) // 4}f", proposal.data))


def bits(value):
  return struct.unpack("<I", struct.pack("<f", value))[0]


class P0:
  """Runs SIR's interpreter on one body's bundle for a list of input values."""

  def __init__(self, source, region, scratch):
    self.source, self.region, self.scratch, self.runs = source, region, Path(scratch), 0

  def __call__(self, values):
    self.runs += 1
    parameters = tensor_free_form_inputs.parameters(self.region)
    proposal = tensor_free_form_inputs.Proposal(
      schema=tensor_free_form_inputs.WINDOW_PROBE_SCHEMA,
      pattern="steered",
      seed=0,
      names=tuple(record["name"] for record in parameters),
      shapes=tuple(tuple(record["shape"]) for record in parameters),
      dtype="f32",
      values=(),
      data=struct.pack(f"<{len(values)}f", *values),
    )
    case = tensor_free_form_case.build_bundle(
      self.scratch / f"run-{self.runs}",
      region=self.region,
      source=self.source,
      proposal=proposal,
    )
    observation = tensor_target_reference.observe(case, symiri=SYMIRI)
    return observation.output if observation.completed else None


@unittest.skipUnless(
  numpy is not None and SYMIRI.is_file() and shutil.which("z3"),
  "numpy, symiri, or z3 is unavailable",
)
class SteeringTests(unittest.TestCase):
  def setUp(self):
    self.assertIsNotNone(tensor_steering, "test.lib.tensor_steering is missing")
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.scratch = Path(temporary.name)
    self.base = values_of(proposals()[0])

  def p0(self, source):
    return P0(source, REGION, self.scratch / "p0")

  def steer(self, source, prop, **options):
    return tensor_steering.steer(
      source,
      PARAMETERS,
      self.base,
      prop=prop,
      p0=options.pop("p0", self.p0(source)),
      scratch=self.scratch / "smt",
      **options,
    )

  def test_the_point_model_matches_p0_on_every_probe(self):
    for source in (SPECIFIED, SUMMED):
      for proposal in proposals():
        with self.subTest(probe=proposal.probe, source=source.count("sum")):
          expected = self.p0(source)(values_of(proposal))
          model = tensor_steering.evaluate(source, PARAMETERS, values_of(proposal))
          self.assertEqual(
            [bits(value) for value in model],
            list(struct.unpack(f"<{len(expected) // 4}I", expected)),
          )

  def test_the_real_encoding_at_integer_inputs_equals_p0(self):
    outputs = tensor_steering.real_outputs(
      SUMMED, PARAMETERS, self.base, scratch=self.scratch / "smt"
    )
    expected = struct.unpack("<16f", self.p0(SUMMED)(self.base))
    self.assertEqual([float(value) for value in outputs], list(expected))

  def test_a_signed_zero_is_steered_through_an_underflowing_product(self):
    # Linearized around small values, the free factor can stay a nonzero f32.
    base = [2.0**-100] * 16 + [2.0**-60] * 16
    p0 = P0(PRODUCT, PRODUCT_REGION, self.scratch / "p0")
    result = tensor_steering.steer(
      PRODUCT,
      [("%x", [2, 8]), ("%y", [2, 8])],
      base,
      prop="signed-zero",
      lane=0,
      p0=p0,
      scratch=self.scratch / "smt",
    )
    self.assertEqual(result.status, "accepted", result.reason)
    self.assertTrue(all(value != 0 for value in result.values))
    self.assertEqual(struct.unpack("<I", p0(result.values)[:4])[0], 0x80000000)

  def test_a_signed_zero_from_sums_of_nonzero_inputs_is_refused(self):
    # SPECIFIED's lane 0 adds f32 values: 0 rounds to +0.0, and any nonzero
    # result is at least 2^-149 in magnitude, so -0.0 is out of reach.
    self.assertEqual(self.steer(SPECIFIED, "signed-zero", lane=0).status, "refused")

  def test_an_unreachable_surrogate_is_a_refusal(self):
    region = {**REGION, "parameters": [{"name": "%x", "shape": [8], "dtype": "f32"}]}
    result = tensor_steering.steer(
      SQUARE,
      [("%x", [8])],
      [1.0, 2.0, 3.0, -1.0, 2.0, 1.0, 3.0, 2.0],
      prop="signed-zero",
      lane=0,
      p0=P0(SQUARE, region, self.scratch / "p0"),
      scratch=self.scratch / "smt",
    )
    self.assertEqual((result.status, result.reason), ("refused", "unsat"))
    self.assertTrue(result.linearized)

  def test_a_cancellation_is_steered_with_large_operands(self):
    result = self.steer(SPECIFIED, "cancel", lane=0)
    self.assertEqual(result.status, "accepted", result.reason)
    output = struct.unpack("<f", self.p0(SPECIFIED)(result.values)[:4])[0]
    self.assertEqual(output, 0.0)
    self.assertGreaterEqual(result.evidence["largest_operand"], 2**20)

  def test_a_tie_is_steered_into_a_sum(self):
    result = self.steer(SUMMED, "tie")
    self.assertEqual(result.status, "accepted", result.reason)
    self.assertTrue(result.evidence["halfway"])

  def test_a_tie_needs_a_sum(self):
    result = self.steer(SPECIFIED, "tie")
    self.assertEqual((result.status, result.reason), ("refused", "the body has no sum"))

  def test_a_product_of_two_computed_values_is_linearized(self):
    self.assertTrue(self.steer(SPECIFIED, "signed-zero", lane=0).linearized)

  def test_a_body_search_returns_each_accepted_property_once(self):
    bases = {"i0-base": self.base}
    outcome = tensor_steering.steer_body(
      SUMMED, PARAMETERS, bases, p0=self.p0(SUMMED), scratch=self.scratch / "smt"
    )
    self.assertEqual(sorted(outcome.accepted), ["cancel", "tie"])
    tie = outcome.accepted["tie"]
    self.assertEqual((tie.status, tie.property), ("accepted", "tie"))
    self.assertTrue(
      all(trial["property"] in tensor_steering.PROPERTIES for trial in outcome.trials)
    )

  def test_a_body_search_stops_at_its_budget(self):
    outcome = tensor_steering.steer_body(
      SUMMED,
      PARAMETERS,
      {"i0-base": self.base},
      p0=self.p0(SUMMED),
      scratch=self.scratch / "smt",
      budget_s=0,
    )
    self.assertEqual((outcome.accepted, outcome.trials), ({}, []))

  def test_a_body_search_tries_only_the_requested_properties(self):
    outcome = tensor_steering.steer_body(
      SUMMED,
      PARAMETERS,
      {"i0-base": self.base},
      p0=self.p0(SUMMED),
      scratch=self.scratch / "smt",
      properties=("tie",),
    )
    self.assertEqual(sorted(outcome.accepted), ["tie"])
    self.assertEqual({trial["property"] for trial in outcome.trials}, {"tie"})

  def test_a_miss_is_retried_then_refused(self):
    result = self.steer(SPECIFIED, "cancel", lane=0, p0=lambda values: None, attempts=3)
    self.assertEqual(result.status, "refused")
    self.assertEqual(len(result.attempts), 3)
    self.assertTrue(
      all(attempt["miss"] == "p0_not_completed" for attempt in result.attempts)
    )
    self.assertTrue(math.isfinite(result.z3_seconds))


if __name__ == "__main__":
  unittest.main()
