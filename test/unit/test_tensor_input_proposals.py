"""Tests for deterministic input proposals derived from an admitted region."""

import argparse
import hashlib
import json
import struct
import tempfile
import time
import unittest
from pathlib import Path

from test.lib import (
  build_tensor_exact_region_case,
  tensor_case,
  tensor_exact_f32,
  tensor_input_proposals,
  tensor_region_generation,
)
from test.tensor.campaigns import run_input_proposals
from test.unit.tensor_support import SYMIRC, SYMIRI

DOMAIN = (0, 8)
EXPORTER = tensor_region_generation.EXPORTER
TOOLS_AVAILABLE = SYMIRI.is_file() and SYMIRC.is_file() and EXPORTER.is_file()


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


def _producer_arguments(shape, iterations, family, seed=9):
  """The generation options the proposal driver states for one region."""
  return argparse.Namespace(
    seed=seed,
    shape=",".join(str(value) for value in shape),
    iterations=iterations,
    operation_mode="additive",
    max_operators=10,
    exporter=str(EXPORTER),
    force=False,
    domain=",".join(str(value) for value in DOMAIN),
    representative=family,
  )


def _digest(data):
  return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _build_counted_case(work, source, configuration, domain=DOMAIN):
  """A witness-route counted case, so the proposal inherits a real manifest.

  The witness is a fixed in-domain input rather than the tested proposal: the
  proposal route needs the counted case's profile, launch, and execution
  records, not its runtime bytes.
  """
  shape = tensor_exact_f32._generated_region_program(source)[0]
  values = (1,) * (shape[0] * shape[1])
  _, expected = tensor_exact_f32.certify(
    source, struct.pack(f"<{len(values)}f", *values), domain
  )
  target = int(struct.unpack("<f", expected[:4])[0])
  directory = Path(work) / "counted"
  build_tensor_exact_region_case.build_case(
    directory,
    source=source,
    target=target,
    symiri=SYMIRI,
    symirc=SYMIRC,
    domain=domain,
    configuration=configuration,
    selection="witness",
    witness=values,
  )
  return directory, json.loads((directory / "case.json").read_text(encoding="utf-8"))


def _build_proposal_case(
  work,
  *,
  family,
  shape,
  iterations,
  proposal_name,
  certificate_domain=DOMAIN,
  counted_domain=DOMAIN,
):
  """Generate the region, count it, and build one proposal through the driver."""
  source, configuration, generation = run_input_proposals.build_family(
    work, _producer_arguments(shape, iterations, family), family
  )
  proposals = tensor_input_proposals.derive_proposals(
    source, domain=certificate_domain, seed=generation["configuration"]["seed"]
  )
  proposal = next(item for item in proposals if item.name == proposal_name)
  counted_dir, counted_manifest = _build_counted_case(
    work, source, configuration, counted_domain
  )
  provenance = run_input_proposals._provenance(
    family,
    source,
    generation,
    run_input_proposals._reference_case(counted_manifest, counted_dir),
    certificate_domain,
  )
  directory = Path(work) / "suite" / f"{family}--{proposal_name}"
  run_input_proposals.build_proposal_case(
    directory,
    source=source,
    proposal=proposal,
    domain=certificate_domain,
    counted_manifest=counted_manifest,
    provenance=provenance,
    symiri=SYMIRI,
    ledger=[],
    deadline=time.monotonic() + 120,
  )
  manifest = json.loads((directory / "case.json").read_text(encoding="utf-8"))
  loaded = tensor_case.load_case(directory, symiri=SYMIRI)
  return source, proposal, directory, manifest, loaded


def _assert_exact_route(
  test, source, proposal, directory, manifest, loaded, domain=DOMAIN
):
  """The case and its proof both have to survive the loader unchanged."""
  proof, expected = tensor_exact_f32.certify(source, loaded.inputs, domain)
  test.assertEqual(loaded.inputs, proposal.data)
  test.assertEqual(loaded.expected, expected)
  test.assertEqual(manifest["comparison"]["mode"], "exact")
  test.assertEqual(manifest["comparison"]["atol"], [0, 1])
  test.assertEqual(manifest["comparison"]["rtol"], [0, 1])
  test.assertEqual(manifest["reference"]["precisions"], [])
  test.assertEqual(manifest["reference"]["exactness"], proof)
  test.assertEqual(manifest["reference"]["kind"], "exact-certificate")
  test.assertEqual(manifest["evidence"]["kind"], "construction")
  test.assertEqual(manifest["evidence"]["status"], "construction_checked")
  record = json.loads((directory / "proposal-record.json").read_text(encoding="utf-8"))
  test.assertEqual(record["admission"], "admitted-exact-certificate")
  test.assertEqual(record["reference_kind"], "exact-certificate")
  test.assertEqual(record["comparison_mode"], "exact")


@unittest.skipUnless(
  TOOLS_AVAILABLE, "built symiri, symirc and the region exporter are required"
)
class SampledExactRouteProducerTests(unittest.TestCase):
  """Five sampled proposals that carry their own exact whole-output proof."""

  def test_a_non_square_multi_iteration_broadcast_region_admits_an_exact_case(self):
    with tempfile.TemporaryDirectory() as root:
      source, proposal, directory, manifest, loaded = _build_proposal_case(
        Path(root),
        family="broadcast-hoist",
        shape=(4, 8),
        iterations=4,
        proposal_name="independent-small",
      )
      self.assertEqual(manifest["inputs"][0]["shape"], [4, 8])
      self.assertEqual(manifest["coverage"]["loop_structure"], [{"bound": 4}])
      _assert_exact_route(self, source, proposal, directory, manifest, loaded)

  def test_a_wide_non_square_region_admits_an_exact_case(self):
    with tempfile.TemporaryDirectory() as root:
      source, proposal, directory, manifest, loaded = _build_proposal_case(
        Path(root),
        family="broadcast-hoist",
        shape=(32, 16),
        iterations=2,
        proposal_name="asymmetric-coordinates",
      )
      self.assertEqual(manifest["inputs"][0]["shape"], [32, 16])
      self.assertEqual(manifest["coverage"]["loop_structure"], [{"bound": 2}])
      _assert_exact_route(self, source, proposal, directory, manifest, loaded)

  def test_a_square_multi_iteration_region_admits_an_exact_case(self):
    with tempfile.TemporaryDirectory() as root:
      source, proposal, directory, manifest, loaded = _build_proposal_case(
        Path(root),
        family="cross-state",
        shape=(8, 8),
        iterations=3,
        proposal_name="sparse-distinct-groups",
      )
      self.assertEqual(manifest["inputs"][0]["shape"], [8, 8])
      self.assertEqual(manifest["coverage"]["loop_structure"], [{"bound": 3}])
      _assert_exact_route(self, source, proposal, directory, manifest, loaded)

  def test_a_square_mixed_depth_region_admits_an_exact_case(self):
    with tempfile.TemporaryDirectory() as root:
      source, proposal, directory, manifest, loaded = _build_proposal_case(
        Path(root),
        family="mixed-depth",
        shape=(8, 8),
        iterations=3,
        proposal_name="sparse-final-lanes",
      )
      self.assertEqual(manifest["inputs"][0]["shape"], [8, 8])
      self.assertEqual(manifest["coverage"]["loop_structure"], [{"bound": 3}])
      _assert_exact_route(self, source, proposal, directory, manifest, loaded)

  def test_the_counted_representative_admits_an_exact_case(self):
    with tempfile.TemporaryDirectory() as root:
      source, proposal, directory, manifest, loaded = _build_proposal_case(
        Path(root),
        family="mixed-depth",
        shape=(32, 16),
        iterations=4,
        proposal_name="independent-small",
      )
      self.assertEqual(manifest["inputs"][0]["shape"], [32, 16])
      self.assertEqual(manifest["coverage"]["loop_structure"], [{"bound": 4}])
      _assert_exact_route(self, source, proposal, directory, manifest, loaded)


@unittest.skipUnless(
  TOOLS_AVAILABLE, "built symiri, symirc and the region exporter are required"
)
class SampledExactRouteEdgeTests(unittest.TestCase):
  def test_a_proof_is_rejected_for_different_input_or_expected_bytes(self):
    with tempfile.TemporaryDirectory() as root:
      source, _proposal, directory, manifest, loaded = _build_proposal_case(
        Path(root),
        family="broadcast-hoist",
        shape=(4, 8),
        iterations=4,
        proposal_name="independent-small",
      )
      inputs = bytearray(loaded.inputs)
      values = list(struct.unpack(f"<{len(inputs) // 4}f", inputs))
      replacement = None
      for candidate in (values[0] + 1.0, values[0] - 1.0, values[0] + 2.0):
        if not DOMAIN[0] <= candidate <= DOMAIN[1]:
          continue
        trial = list(values)
        trial[0] = candidate
        trial_bytes = struct.pack(f"<{len(trial)}f", *trial)
        _, trial_expected = tensor_exact_f32.certify(source, trial_bytes, DOMAIN)
        if trial_expected != loaded.expected:
          replacement = (trial_bytes, trial_expected)
          break
      self.assertIsNotNone(replacement, "no in-domain input changes the output")
      input_bytes, expected_bytes = replacement
      inputs_record = manifest["files"]["inputs.bin"]
      expected_record = manifest["files"]["expected.bin"]

      with self.subTest(stale="input bytes"):
        (directory / "inputs.bin").write_bytes(input_bytes)
        manifest["files"]["inputs.bin"] = _digest(input_bytes)
        (directory / "case.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(
          tensor_case.CaseValidationError, "complete output bits differ"
        ):
          tensor_case.load_case(directory, symiri=SYMIRI)
        (directory / "inputs.bin").write_bytes(loaded.inputs)
        manifest["files"]["inputs.bin"] = inputs_record

      with self.subTest(stale="expected bytes"):
        self.assertNotEqual(expected_bytes, loaded.expected)
        (directory / "expected.bin").write_bytes(expected_bytes)
        manifest["files"]["expected.bin"] = _digest(expected_bytes)
        (directory / "case.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(
          tensor_case.CaseValidationError, "complete output bits differ"
        ):
          tensor_case.load_case(directory, symiri=SYMIRI)
        (directory / "expected.bin").write_bytes(loaded.expected)
        manifest["files"]["expected.bin"] = expected_record

      with self.subTest(stale="proof"):
        proof = json.loads(json.dumps(manifest["reference"]["exactness"]))
        proof["bounds"][0]["max_value"] += 1
        manifest["reference"]["exactness"] = proof
        (directory / "case.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(
          tensor_case.CaseValidationError, "differs from derived obligations"
        ):
          tensor_case.load_case(directory, symiri=SYMIRI)

  def test_a_refused_certificate_stays_numeric_and_labelled(self):
    with tempfile.TemporaryDirectory() as root:
      _source, _proposal, directory, manifest, loaded = _build_proposal_case(
        Path(root),
        family="fold",
        shape=(256, 2),
        iterations=2,
        proposal_name="independent-small",
        certificate_domain=DOMAIN,
        counted_domain=(0, 2),
      )
      record = json.loads(
        (directory / "proposal-record.json").read_text(encoding="utf-8")
      )
      self.assertEqual(manifest["comparison"]["mode"], "numeric")
      self.assertEqual(manifest["comparison"]["atol"], [0, 1])
      self.assertEqual(manifest["comparison"]["rtol"], [0, 1])
      self.assertEqual(manifest["reference"]["precisions"], [128, 256])
      self.assertIsNone(manifest["reference"]["exactness"])
      self.assertEqual(manifest["reference"]["kind"], "native-refractir-replay")
      self.assertEqual(manifest["evidence"]["kind"], "numerical_reference")
      self.assertEqual(manifest["evidence"]["status"], "numerical_reference_checked")
      self.assertEqual(loaded.manifest["comparison"]["mode"], "numeric")
      self.assertEqual(record["admission"], "refused-exact-certificate")
      self.assertEqual(record["reference_kind"], "native-refractir-replay")
      self.assertEqual(record["comparison_mode"], "numeric")
      self.assertIn("exceeds exact range", record["certificate"]["refusal_reason"])


def _report_entry(name, admission, kind, mode, status="pass"):
  return {
    "name": name,
    "admission": admission,
    "reference": {"kind": kind},
    "comparison": {"mode": mode},
    "execution": {"status": status, "comparison": {"status": "pass"}},
    "output": {"constant": False},
    "duplicate_inputs": {},
  }


class SampledExactRouteSummaryTests(unittest.TestCase):
  """The report keeps exact-certified and fallback executions apart."""

  def test_the_summary_counts_the_two_reference_routes_separately(self):
    report = {
      "kernels": [
        {
          "family": "mixed-depth",
          "admission": {"status": "admitted"},
          "proposals": [
            _report_entry(
              "independent-small",
              "admitted-exact-certificate",
              "exact-certificate",
              "exact",
            ),
            _report_entry(
              "asymmetric-coordinates",
              "refused-exact-certificate",
              "native-refractir-replay",
              "numeric",
            ),
            _report_entry(
              "sparse-final-lanes",
              "refused-exact-certificate",
              "native-refractir-replay",
              "numeric",
              status="not_run",
            ),
          ],
        }
      ]
    }
    summary = run_input_proposals._summary(report)
    exact = summary["reference_routes"]["exact-certificate"]
    fallback = summary["reference_routes"]["native-refractir-replay"]
    self.assertEqual((exact["cases"], exact["executed"], exact["passed"]), (1, 1, 1))
    self.assertEqual(
      (fallback["cases"], fallback["executed"], fallback["passed"]), (2, 1, 1)
    )
    self.assertEqual(exact["proposals"], ["mixed-depth/independent-small"])
    self.assertEqual(summary["proposal_refusals"], [])
    self.assertEqual(
      [row["proposal"] for row in summary["certificate_refusals"]],
      ["asymmetric-coordinates", "sparse-final-lanes"],
    )
    self.assertEqual(
      [row["comparison_mode"] for row in summary["rows"]],
      ["exact", "numeric", "numeric"],
    )


if __name__ == "__main__":
  unittest.main()
