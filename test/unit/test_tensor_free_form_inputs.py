"""Contract tests for free-form runtime inputs derived without a certificate."""

import struct
import unittest
from unittest import mock

from test.lib import tensor_free_form_inputs as inputs

SPEC = {
  "schema": "refractir.free-form-region.v1",
  "parameters": [
    {"name": "%x", "shape": [2, 2], "dtype": "f32"},
    {"name": "%y", "shape": [4], "dtype": "f32"},
  ],
}


class FreeFormInputTests(unittest.TestCase):
  def test_parameters_are_read_in_the_recorded_order(self):
    parameters = inputs.parameters(SPEC)
    self.assertEqual([record["name"] for record in parameters], ["%x", "%y"])
    self.assertEqual([record["shape"] for record in parameters], [[2, 2], [4]])
    self.assertEqual([record["dtype"] for record in parameters], ["f32", "f32"])

  def test_payload_length_matches_each_parameter_signature(self):
    proposal = inputs.proposal(SPEC, pattern="distinct-integer", seed=0)
    self.assertEqual(len(proposal.data), (4 + 4) * 4)
    first = struct.unpack("<4f", proposal.data[:16])
    second = struct.unpack("<4f", proposal.data[16:])
    self.assertEqual(tuple(proposal.values[0]), first)
    self.assertEqual(tuple(proposal.values[1]), second)
    self.assertEqual(first, (1.0, 2.0, 3.0, 4.0))

  def test_same_seed_replays_and_patterns_differ(self):
    first = inputs.proposal(SPEC, pattern="mixed-sign", seed=9)
    second = inputs.proposal(SPEC, pattern="mixed-sign", seed=9)
    self.assertEqual(first.data, second.data)
    other = inputs.proposal(SPEC, pattern="distinct-integer", seed=9)
    self.assertNotEqual(first.data, other.data)

  def test_proposals_do_not_read_the_certificate_parser(self):
    from test.lib import tensor_exact_f32

    with mock.patch.object(
      tensor_exact_f32,
      "_generated_region_program",
      side_effect=AssertionError("the general path must not need the canonical parser"),
    ):
      proposals = inputs.derive_proposals_from_spec(SPEC, count=4, seed=3)
    self.assertEqual(len(proposals), 4)
    self.assertEqual(
      [proposal.pattern for proposal in proposals],
      list(inputs.PATTERNS),
    )
    self.assertTrue(all(len(proposal.data) == 32 for proposal in proposals))

  def test_more_proposals_than_patterns_repeat_a_pattern_under_a_later_seed(self):
    proposals = inputs.derive_proposals_from_spec(SPEC, count=8, seed=0)
    self.assertEqual(proposals[0].pattern, proposals[4].pattern)
    self.assertEqual(proposals[0].seed, 0)
    self.assertEqual(proposals[4].seed, 4)
    random_pair = [proposal for proposal in proposals if proposal.pattern == "seeded-random"]
    self.assertNotEqual(random_pair[0].data, random_pair[1].data)

  def test_an_unusable_record_is_refused(self):
    with self.assertRaises(inputs.SpecError):
      inputs.parameters({})
    with self.assertRaises(inputs.SpecError):
      inputs.parameters({"parameters": [{"name": "%x", "shape": [0], "dtype": "f32"}]})
    with self.assertRaises(inputs.SpecError):
      inputs.parameters({"parameters": [{"name": "%x", "shape": [2], "dtype": "f64"}]})
    with self.assertRaises(inputs.SpecError):
      inputs.proposal(SPEC, pattern="unknown", seed=0)
    with self.assertRaises(inputs.SpecError):
      inputs.derive_proposals_from_spec(SPEC, count=0)


if __name__ == "__main__":
  unittest.main()
