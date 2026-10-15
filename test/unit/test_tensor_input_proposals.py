"""Tests for deterministic input proposals derived from an admitted region."""

import hashlib
import struct
import unittest

from test.lib import tensor_exact_f32, tensor_input_proposals


def _source(body, temporaries, shape=(4, 3), iterations=2):
  """A minimal canonical generated region with the given body operations."""
  count = shape[0] * shape[1]
  declarations = "\n".join(
    f"  let mut %{name}: tensor [{extent}] f32 = undef;" for name, extent in temporaries
  )
  return (
    "fun @kernel(%x: tensor [4, 3] f32) : tensor [4, 3] f32 {\n"
    "  let mut %i: i32 = undef;\n"
    "  let mut %acc_x: tensor [4, 3] f32 = undef;\n"
    "  let mut %acc_y: tensor [4, 3] f32 = {" + ", ".join(["0.0"] * count) + "};\n"
    "  let mut %out: tensor [4, 3] f32 = undef;\n"
    f"{declarations}\n"
    "^entry:\n"
    "  %acc_x = %x;\n"
    "  %i = 0;\n"
    "  br ^header;\n"
    "^header:\n"
    f"  br %i < {iterations}, ^body, ^exit;\n"
    "^body:\n" + "\n".join(f"  {line}" for line in body) + "\n"
    "  %i = %i + 1;\n"
    "  br ^header;\n"
    "^exit:\n"
    "  %out = tensor add %acc_x, %acc_y;\n"
    "  ret %out;\n"
    "}\n"
  )


ONE_REDUCTION = _source(
  [
    "%t0 = tensor add %acc_x, %acc_y;",
    "%t1 = tensor sum %t0, 0, keep;",
    "%t3 = tensor broadcast %t1, [4, 3];",
    "%next_x = tensor add %t0, %t3;",
    "%next_y = tensor add %acc_y, %t0;",
    "%acc_x = %next_x;",
    "%acc_y = %next_y;",
  ],
  [
    ("t0", "4, 3"),
    ("t1", "1, 3"),
    ("t3", "4, 3"),
    ("next_x", "4, 3"),
    ("next_y", "4, 3"),
  ],
)

BOTH_REDUCTIONS = _source(
  [
    "%t0 = tensor add %acc_x, %acc_y;",
    "%t1 = tensor sum %t0, 0, keep;",
    "%t2 = tensor sum %t0, 1, keep;",
    "%t3 = tensor broadcast %t1, [4, 3];",
    "%t4 = tensor broadcast %t2, [4, 3];",
    "%next_x = tensor add %t0, %t3;",
    "%next_y = tensor add %acc_y, %t4;",
    "%acc_x = %next_x;",
    "%acc_y = %next_y;",
  ],
  [
    ("t0", "4, 3"),
    ("t1", "1, 3"),
    ("t2", "4, 1"),
    ("t3", "4, 3"),
    ("t4", "4, 3"),
    ("next_x", "4, 3"),
    ("next_y", "4, 3"),
  ],
)

NO_REDUCTION = _source(
  [
    "%t0 = tensor add %acc_x, %acc_y;",
    "%next_x = tensor add %t0, %acc_x;",
    "%next_y = tensor add %acc_y, %t0;",
    "%acc_x = %next_x;",
    "%acc_y = %next_y;",
  ],
  [("t0", "4, 3"), ("next_x", "4, 3"), ("next_y", "4, 3")],
)


def _lane(proposal, shape, coordinates):
  index = 0
  for axis, coordinate in enumerate(coordinates):
    index = index * shape[axis] + coordinate
  return proposal.values[index]


class TensorInputProposalTests(unittest.TestCase):
  def test_the_four_required_proposals_are_derived_in_order(self):
    found = tensor_input_proposals.derive_proposals(
      ONE_REDUCTION, domain=(0, 8), seed=9
    )
    self.assertEqual(
      [proposal.name for proposal in found],
      [
        "independent-small",
        "asymmetric-coordinates",
        "sparse-distinct-groups",
        "sparse-final-lanes",
      ],
    )

  def test_the_same_case_derives_identical_bytes(self):
    first = tensor_input_proposals.derive_proposals(
      ONE_REDUCTION, domain=(0, 8), seed=9
    )
    again = tensor_input_proposals.derive_proposals(
      ONE_REDUCTION, domain=(0, 8), seed=9
    )
    other_seed = tensor_input_proposals.derive_proposals(
      ONE_REDUCTION, domain=(0, 8), seed=10
    )
    self.assertEqual(
      [proposal.data for proposal in first], [proposal.data for proposal in again]
    )
    self.assertNotEqual(first[0].data, other_seed[0].data)

  def test_every_proposal_stays_in_the_exact_domain(self):
    for proposal in tensor_input_proposals.derive_proposals(
      BOTH_REDUCTIONS, domain=(0, 8), seed=9
    ):
      self.assertEqual(len(proposal.values), 12)
      words = struct.unpack(f"<{len(proposal.values)}I", proposal.data)
      for value, word in zip(proposal.values, words):
        self.assertIsInstance(value, int)
        self.assertTrue(0 <= value <= 8)
        self.assertEqual(float(value), struct.unpack("<f", struct.pack("<I", word))[0])
        if value == 0:
          self.assertEqual(word, 0)

  def test_independent_small_values_are_small_and_lane_independent(self):
    proposal = tensor_input_proposals.derive_proposals(
      ONE_REDUCTION, domain=(0, 8), seed=9
    )[0]
    self.assertLessEqual(max(proposal.values), 3)
    self.assertGreaterEqual(len(set(proposal.values)), 2)
    # Independence: two lanes of the same reduction group may differ, which a
    # coordinate pattern cannot do when it is constant along the reduced axis.
    self.assertNotEqual(
      _lane(proposal, (4, 3), (0, 0)), _lane(proposal, (4, 3), (1, 0))
    )

  def test_asymmetric_coordinates_depend_on_both_axes(self):
    proposal = tensor_input_proposals.derive_proposals(
      ONE_REDUCTION, domain=(0, 8), seed=9
    )[1]
    self.assertNotEqual(
      _lane(proposal, (4, 3), (0, 0)), _lane(proposal, (4, 3), (1, 0))
    )
    self.assertNotEqual(
      _lane(proposal, (4, 3), (0, 0)), _lane(proposal, (4, 3), (0, 1))
    )
    self.assertNotEqual(
      _lane(proposal, (4, 3), (1, 2)), _lane(proposal, (4, 3), (2, 1))
    )

  def test_sparse_distinct_groups_marks_the_first_lane_of_every_group(self):
    proposal = tensor_input_proposals.derive_proposals(
      ONE_REDUCTION, domain=(0, 8), seed=9
    )[2]
    marked = [
      (index // 3, index % 3) for index, value in enumerate(proposal.values) if value
    ]
    self.assertEqual(marked, [(0, column) for column in range(3)])
    self.assertGreaterEqual(len(set(proposal.values)), 2)

  def test_sparse_final_lanes_marks_the_last_lane_of_every_group(self):
    one = tensor_input_proposals.derive_proposals(ONE_REDUCTION, domain=(0, 8), seed=9)[
      3
    ]
    marked = [
      (index // 3, index % 3) for index, value in enumerate(one.values) if value
    ]
    self.assertEqual(marked, [(3, column) for column in range(3)])
    self.assertEqual(_lane(one, (4, 3), (3, 2)), 8)

    both = tensor_input_proposals.derive_proposals(
      BOTH_REDUCTIONS, domain=(0, 8), seed=9
    )[3]
    marked = {
      (index // 3, index % 3) for index, value in enumerate(both.values) if value
    }
    self.assertEqual(
      marked, {(3, column) for column in range(3)} | {(row, 2) for row in range(4)}
    )
    self.assertEqual(_lane(both, (4, 3), (3, 2)), 8)

  def test_a_region_without_a_reduction_keeps_the_axis_zero_groups(self):
    proposals = tensor_input_proposals.derive_proposals(
      NO_REDUCTION, domain=(0, 8), seed=9
    )
    first_group = [
      (index // 3, index % 3)
      for index, value in enumerate(proposals[2].values)
      if value
    ]
    final_group = [
      (index // 3, index % 3)
      for index, value in enumerate(proposals[3].values)
      if value
    ]
    self.assertEqual(first_group, [(0, column) for column in range(3)])
    self.assertEqual(final_group, [(3, column) for column in range(3)])
    self.assertEqual(proposals[2].spec["group_axis_source"], "default-axis-0")

  def test_coincident_proposals_are_grouped_instead_of_dropped(self):
    first = tensor_input_proposals.derive_proposals(
      ONE_REDUCTION, domain=(0, 8), seed=9
    )[0]
    copy = tensor_input_proposals.InputProposal(
      name="independent-small-copy", spec=dict(first.spec), values=first.values
    )
    self.assertEqual(
      tensor_input_proposals.duplicate_groups((first, copy)),
      [["independent-small", "independent-small-copy"]],
    )
    self.assertEqual(
      tensor_input_proposals.duplicate_groups(
        tensor_input_proposals.derive_proposals(ONE_REDUCTION, domain=(0, 8), seed=9)
      ),
      [],
    )

  def test_smaller_domains_bound_every_pattern(self):
    for proposal in tensor_input_proposals.derive_proposals(
      BOTH_REDUCTIONS, domain=(0, 1), seed=9
    ):
      self.assertTrue(all(0 <= value <= 1 for value in proposal.values))

  def test_a_negative_domain_is_refused(self):
    with self.assertRaises(ValueError):
      tensor_input_proposals.derive_proposals(ONE_REDUCTION, domain=(-1, 8), seed=9)

  def test_records_carry_the_bytes_and_the_derivation(self):
    proposal = tensor_input_proposals.derive_proposals(
      ONE_REDUCTION, domain=(0, 8), seed=9
    )[0]
    record = proposal.record()
    self.assertEqual(record["sha256"], hashlib.sha256(proposal.data).hexdigest())
    self.assertEqual(record["byte_length"], len(proposal.data))
    self.assertEqual(record["values"], list(proposal.values))
    self.assertEqual(record["name"], "independent-small")
    self.assertTrue(record["in_domain"])
    self.assertEqual(record["derivation"], tensor_input_proposals.DERIVATION_VERSION)


class TensorInputProposalRefusalTests(unittest.TestCase):
  def test_a_source_outside_the_generated_region_is_refused(self):
    with self.assertRaises(ValueError):
      tensor_input_proposals.derive_proposals(
        "fun @kernel(%x: tensor [2, 2] f32) : tensor [2, 2] f32 {\n}\n",
        domain=(0, 8),
        seed=9,
      )

  def test_values_outside_the_domain_cannot_be_packed(self):
    proposal = tensor_input_proposals.InputProposal(
      name="out-of-domain", spec={}, values=(9,) + (0,) * 11
    )
    with self.assertRaises(ValueError):
      tensor_input_proposals.pack(proposal, domain=(0, 8))

  def test_the_derived_proposals_decode_back_to_their_values(self):
    for proposal in tensor_input_proposals.derive_proposals(
      ONE_REDUCTION, domain=(0, 8), seed=9
    ):
      values = struct.unpack(f"<{len(proposal.values)}f", proposal.data)
      self.assertEqual(tuple(int(value) for value in values), proposal.values)
      self.assertEqual(
        tensor_exact_f32._generated_region_program(ONE_REDUCTION)[0], (4, 3)
      )

  def test_a_constant_reference_output_is_reported_with_its_word(self):
    constant = tensor_input_proposals.constant_output(struct.pack("<3f", 0.0, 0.0, 0.0))
    self.assertEqual(constant, {"constant": True, "word": "0x00000000", "value": 0.0})
    varying = tensor_input_proposals.constant_output(struct.pack("<3f", 0.0, 1.0, 2.0))
    self.assertEqual(varying, {"constant": False, "distinct_words": 3})
    with self.assertRaises(ValueError):
      tensor_input_proposals.constant_output(b"")


if __name__ == "__main__":
  unittest.main()
