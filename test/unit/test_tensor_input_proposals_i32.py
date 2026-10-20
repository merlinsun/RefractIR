"""Tests for the explicit i32 input-proposal adapter.

The adapter exists for tensor-v3 tail cases: the dtype is i32 and an extent may
be a logical tail such as 31 or 63, so the f32 generated-region parser cannot
state either fact. The tests pass the case facts explicitly, then check the
four patterns, the v3 pattern metadata, the case-derived reference replay, and
the refusal boundary.
"""

from __future__ import annotations

import inspect
import re
import shlex
import struct
import tempfile
import unittest
from pathlib import Path

from test.lib import tensor_case
from test.lib import tensor_input_proposals_i32 as proposals_i32

DOMAIN = (0, 8)
TAIL_SHAPE = (16, 31)
RANK_ONE_TAIL = (31,)


def _case(shape=TAIL_SHAPE, reduction_axes=(1,), **overrides):
  """One explicit i32 case, with every fact the adapter is allowed to read."""
  facts = {
    "shape": shape,
    "reduction_axes": reduction_axes,
    "dtype": "i32",
    "domain": DOMAIN,
    "seed": 9,
  }
  facts.update(overrides)
  return proposals_i32.I32ProposalCase(**facts)


def _words(proposal):
  return struct.unpack(f"<{len(proposal.values)}i", proposal.data)


def _loader_v3_patterns():
  """The closed pattern set `tensor_case.load_case` validates for tensor-v3.

  Reading the loader's own source keeps this test pinned to the live
  vocabulary instead of to a second copy that could drift.
  """
  source = inspect.getsource(tensor_case.load_case)
  block = re.search(r"pattern\s+in\s*\{(.*?)\}", source, re.DOTALL)
  if block is None:
    raise AssertionError("the loader no longer states its tensor-v3 pattern set")
  return frozenset(re.findall(r'"([^"]+)"', block[1]))


class TensorInputProposalI32Tests(unittest.TestCase):
  def test_a_rank_two_i32_tail_case_produces_all_four_in_domain_patterns(self):
    proposals = proposals_i32.derive_proposals(_case())
    self.assertEqual(
      [proposal.name for proposal in proposals], list(proposals_i32.PROPOSAL_NAMES)
    )
    rows, columns = TAIL_SHAPE
    count = rows * columns
    for proposal in proposals:
      self.assertEqual(len(proposal.values), count)
      self.assertEqual(len(proposal.data), 4 * count)
      self.assertEqual(_words(proposal), proposal.values)
      self.assertEqual(proposal.spec["dtype"], "i32")
      self.assertEqual(proposal.spec["shape"], list(TAIL_SHAPE))
      self.assertEqual(proposal.spec["domain"], list(DOMAIN))
      self.assertEqual(proposal.spec["reduction_axes"], [1])
      self.assertEqual(proposal.spec["group_axes"], [1])
      self.assertEqual(proposal.spec["group_axis_source"], "reduction-axis")
      for value in proposal.values:
        self.assertIs(type(value), int)
        self.assertGreaterEqual(value, 0)
        self.assertLessEqual(value, 8)

    independent, asymmetric, distinct, final = proposals
    self.assertLessEqual(max(independent.values), 3)
    self.assertGreaterEqual(len(set(independent.values)), 2)
    for row, column in ((0, 0), (1, 0), (0, 1), (rows - 1, columns - 1)):
      index = row * columns + column
      self.assertEqual(asymmetric.values[index], (3 * row + 5 * column + 1) % 9)

    self.assertEqual(distinct.spec["group_count"], rows)
    marked = {index: value for index, value in enumerate(distinct.values) if value != 0}
    self.assertEqual(set(marked), {row * columns for row in range(rows)})
    self.assertEqual(set(marked.values()), {1, 2, 3})

    expected = {row * columns + columns - 1: 1 for row in range(rows)}
    expected[count - 1] = 8
    self.assertEqual(
      {index: value for index, value in enumerate(final.values) if value != 0},
      expected,
    )

  def test_a_rank_one_scalar_tail_case_produces_the_same_four_patterns(self):
    proposals = proposals_i32.derive_proposals(
      _case(shape=RANK_ONE_TAIL, reduction_axes=())
    )
    self.assertEqual(
      [proposal.name for proposal in proposals], list(proposals_i32.PROPOSAL_NAMES)
    )
    count = RANK_ONE_TAIL[0]
    for proposal in proposals:
      self.assertEqual(len(proposal.values), count)
      self.assertEqual(len(proposal.data), 4 * count)
      self.assertEqual(_words(proposal), proposal.values)
      self.assertEqual(proposal.spec["shape"], list(RANK_ONE_TAIL))
      for value in proposal.values:
        self.assertGreaterEqual(value, 0)
        self.assertLessEqual(value, 8)

    independent, asymmetric, distinct, final = proposals
    self.assertLessEqual(max(independent.values), 3)
    self.assertEqual(distinct.spec["group_axis_source"], "default-axis-0")
    self.assertEqual(distinct.spec["group_count"], 1)
    self.assertEqual(distinct.values, (1,) + (0,) * (count - 1))
    self.assertEqual(final.values, (0,) * (count - 1) + (8,))
    self.assertEqual(asymmetric.values[0], 1)
    self.assertEqual(asymmetric.values[1], 4)
    self.assertEqual(asymmetric.values[count - 1], (3 * (count - 1) + 1) % 9)

  def test_the_v3_pattern_metadata_is_present_and_accepted(self):
    vocabulary = _loader_v3_patterns()
    self.assertEqual(
      proposals_i32.V3_INPUT_PATTERNS,
      {
        "independent-small": "seeded-random",
        "asymmetric-coordinates": "coordinate-distinct",
        "sparse-distinct-groups": "distinct-groups",
        "sparse-final-lanes": "boundary-value",
      },
    )
    for proposal in proposals_i32.derive_proposals(_case()):
      record = proposal.record()
      behavior = record["v3_input_behavior"]
      self.assertEqual(record["dtype"], "i32")
      self.assertEqual(record["name"], proposal.name)
      self.assertIn(behavior["pattern"], vocabulary)
      self.assertEqual(
        behavior["pattern"], proposals_i32.V3_INPUT_PATTERNS[proposal.name]
      )
      self.assertEqual(behavior["proposal"], proposal.name)
      self.assertEqual(behavior["input_mode"], "deterministic-input-proposal")
      self.assertEqual(behavior["input_domain"], {"lo": 0, "hi": 8})
      self.assertNotIn(proposal.name, vocabulary)

  def test_the_reference_replay_uses_the_v3_cases_own_profile_and_entry(self):
    with tempfile.TemporaryDirectory() as root:
      root = Path(root)
      case_dir = root / "case"
      case_dir.mkdir()
      expected = struct.pack("<4i", 1, 2, 3, 4)
      program = "fun @tail_sum(%x: tensor [31] i32) : i32 {\n  ret 0;\n}\n"
      (case_dir / "program.sir").write_text(program, encoding="utf-8")
      (case_dir / "inputs.bin").write_bytes(struct.pack("<31i", *range(31)))
      (case_dir / "expected.bin").write_bytes(expected)
      manifest = {
        "profile": "tensor-v3",
        "entry": "tail_sum",
        "comparison": {"mode": "exact"},
        "reference": {"precisions": [], "stable": True},
      }
      case = tensor_case.TensorCase(
        case_dir, manifest, program.encode(), b"", expected, {"entry": "tail_sum"}
      )
      result = proposals_i32.replay_reference(
        case,
        directory=root / "reference",
        symiri=self._fake_symiri(root, expected),
      )

      self.assertEqual(result.classification, "reference_ok", result.reason)
      self.assertEqual(result.profile, "v3")
      self.assertEqual(result.entry, "tail_sum")
      self.assertEqual(result.reference, expected)
      command = list(result.replays[0].command)
      self.assertEqual(
        command[command.index("--tensor-profile-version") + 1],
        "v3",
        "the adapter must pass the profile the case records",
      )
      self.assertEqual(
        command[command.index("--entry") + 1],
        "tail_sum",
        "the adapter must pass the entry the case records",
      )

  def test_the_adapter_refuses_out_of_domain_values_and_is_deterministic(self):
    case = _case()
    first = proposals_i32.derive_proposals(case)
    again = proposals_i32.derive_proposals(_case())
    self.assertEqual(
      [proposal.data for proposal in first],
      [proposal.data for proposal in again],
    )
    self.assertEqual(
      [proposal.record()["sha256"] for proposal in first],
      [proposal.record()["sha256"] for proposal in again],
    )
    for proposal in first:
      self.assertEqual(proposals_i32.pack(proposal, domain=DOMAIN), proposal.data)
    self.assertNotEqual(
      first[0].data, proposals_i32.derive_proposals(_case(seed=10))[0].data
    )

    out_of_domain = proposals_i32.I32InputProposal(
      name="out-of-domain", spec={}, values=(9,) + (0,) * 495
    )
    with self.assertRaises(ValueError):
      proposals_i32.pack(out_of_domain, domain=DOMAIN)
    with self.assertRaises(ValueError):
      proposals_i32.pack(first[2], domain=(1, 8))
    for refused in (
      {"dtype": "f32"},
      {"shape": (2, 3, 4)},
      {"shape": ()},
      {"shape": (0, 31)},
      {"reduction_axes": (2,)},
      {"reduction_axes": (0, 1, 1)},
      {"domain": (-1, 8)},
      {"domain": (8, 0)},
      {"seed": -1},
    ):
      with self.subTest(refused=refused), self.assertRaises(ValueError):
        _case(**refused)

  @staticmethod
  def _fake_symiri(root, expected):
    """An interpreter that only succeeds for the case's own profile and entry."""
    fixture = root / "reference-fixture.bin"
    fixture.write_bytes(expected)
    script = root / "fake-symiri"
    script.write_text(
      "#!/bin/sh\n"
      "set -u\n"
      "out=\n"
      "profile=\n"
      "entry=\n"
      'while [ "$#" -gt 0 ]; do\n'
      '  case "$1" in\n'
      "    --tensor-output) out=$2; shift 2 ;;\n"
      "    --tensor-profile-version) profile=$2; shift 2 ;;\n"
      "    --entry) entry=$2; shift 2 ;;\n"
      "    --tensor-reference-bits) shift 2 ;;\n"
      "    *) shift ;;\n"
      "  esac\n"
      "done\n"
      '[ "$profile" = v3 ] || exit 71\n'
      '[ "$entry" = tail_sum ] || exit 72\n'
      f'cp {shlex.quote(str(fixture))} "$out"\n',
      encoding="utf-8",
    )
    script.chmod(0o755)
    return script


class TensorInputProposalI32EdgeTests(unittest.TestCase):
  def test_a_single_lane_rank_one_case_marks_the_final_element_once(self):
    proposals = proposals_i32.derive_proposals(_case(shape=(1,), reduction_axes=()))
    for proposal in proposals:
      self.assertEqual(len(proposal.values), 1)
      self.assertEqual(len(proposal.data), 4)
    self.assertTrue(0 <= proposals[0].values[0] <= 3)
    self.assertEqual(proposals[1].values, (1,))
    self.assertEqual(proposals[2].values, (1,))
    self.assertEqual(proposals[3].values, (8,))


if __name__ == "__main__":
  unittest.main()
