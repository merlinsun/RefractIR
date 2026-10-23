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


WINDOW_REGION = {
  "schema": "refractir.free-form-region.v1",
  "profile": "tensor-v3",
  "dtype": "f32",
  "trip_count": 4,
  "parameters": [
    {"name": "%window_buffer", "shape": [7, 8], "dtype": "f32"},
    {"name": "%x", "shape": [2, 8], "dtype": "f32"},
    {"name": "%y", "shape": [2, 8], "dtype": "f32"},
  ],
  "window_source": {"parameter": "%window_buffer", "shape": [7, 8], "dtype": "f32"},
  "window": {"rows": 2, "stride": 1, "active": 5, "source": "%window_buffer"},
}


class WindowProbeTests(unittest.TestCase):
  def test_probes_separate_states_and_backing(self):
    probes = inputs.window_probes(WINDOW_REGION, group=0, column=0)
    self.assertEqual([probe.probe for probe in probes], list(inputs.WINDOW_PROBES))
    state_zero = tuple(probes[0].values[1])
    state_one = tuple(probes[0].values[2])
    self.assertNotEqual(state_zero, state_one)
    self.assertEqual(
      state_zero, tuple(tuple((r + 2 * c) % 5 - 2 for c in range(8)) for r in range(2))
    )
    self.assertEqual(
      state_one, tuple(tuple((2 * r + c + 1) % 5 - 2 for c in range(8)) for r in range(2))
    )
    base = probes[0].values[0]
    self.assertEqual(base[0], tuple((c + 0) % 7 - 3 for c in range(8)))
    self.assertNotEqual(base, state_zero)
    for probe in probes[1:]:
      self.assertEqual(probe.values[1], state_zero)
      self.assertEqual(probe.values[2], state_one)
      self.assertNotEqual(probe.values[0], base)

  def test_shapes_that_match_still_separate_the_roles(self):
    region = {
      **WINDOW_REGION,
      "trip_count": 1,
      "parameters": [
        {"name": "%window_buffer", "shape": [4, 4], "dtype": "f32"},
        {"name": "%x", "shape": [4, 4], "dtype": "f32"},
        {"name": "%y", "shape": [4, 4], "dtype": "f32"},
      ],
      "window_source": {"parameter": "%window_buffer", "shape": [4, 4], "dtype": "f32"},
      "window": {"rows": 4, "stride": 0, "active": 4, "source": "%window_buffer"},
    }
    probe = inputs.window_base_proposal(region, group=1, column=1)
    self.assertEqual(probe.names, ("%window_buffer", "%x", "%y"))
    self.assertEqual(len({tuple(map(tuple, probe.values[0])), tuple(map(tuple, probe.values[1])),
                          tuple(map(tuple, probe.values[2]))}), 3)
    self.assertEqual(struct.unpack("<16f", probe.data[:64]), tuple(
      value for row in probe.values[0] for value in row
    ))

  def test_probes_change_only_the_declared_backing_positions(self):
    probes = inputs.window_probes(WINDOW_REGION, group=0, column=0)
    base = probes[0].values[0]
    read = {(trip * 1 + row, c) for trip in range(4) for row in range(2)
            if trip * 1 + row < 5 for c in range(8)}
    plus = probes[1].values[0]
    for address in range(7):
      for column in range(8):
        expected = base[address][column] + (1 if (address, column) in read else 0)
        self.assertEqual(plus[address][column], expected)
    late = probes[2].values[0]
    for address in range(7):
      if address == 4:
        self.assertEqual(late[address], tuple(-5 if c % 2 == 0 else 5 for c in range(8)))
      else:
        self.assertEqual(late[address], base[address])
    unused = probes[3].values[0]
    for address in range(7):
      if address >= 5:
        self.assertEqual(unused[address], tuple(1 - value for value in base[address]))
      else:
        self.assertEqual(unused[address], base[address])

  def test_the_late_row_is_read_only_after_the_first_window(self):
    probes = inputs.window_probes(WINDOW_REGION, group=0, column=0)
    window = WINDOW_REGION["window"]
    read = {trip * window["stride"] + row for trip in range(4) for row in range(window["rows"])
            if trip * window["stride"] + row < window["active"]}
    self.assertIn(window["active"] - 1, read)
    self.assertNotIn(window["active"] - 1, set(range(window["rows"])))
    self.assertFalse(any(address >= window["active"] for address in read))
    self.assertEqual(probes[3].probe, "i3-unused-rows")

  def test_an_unusable_window_is_refused(self):
    with self.assertRaises(inputs.SpecError):
      inputs.window_probes(WINDOW_REGION, group=3, column=0)
    with self.assertRaises(inputs.SpecError):
      inputs.window_probes(WINDOW_REGION, group=0, column=4)
    with self.assertRaises(inputs.SpecError):
      inputs.window_probes(
        {**WINDOW_REGION, "window": {**WINDOW_REGION["window"], "stride": 0, "active": 1}},
        group=0, column=0,
      )
    with self.assertRaises(inputs.SpecError):
      inputs.window_base_proposal(
        {**WINDOW_REGION, "window": {**WINDOW_REGION["window"], "active": 8}},
        group=0, column=0,
      )
    with self.assertRaises(inputs.SpecError):
      inputs.window_probes(
        {**WINDOW_REGION, "parameters": WINDOW_REGION["parameters"][:2]},
        group=0, column=0,
      )



if __name__ == "__main__":
  unittest.main()
