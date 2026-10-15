"""Tests for the Track A bounded discovery batch driver.

The batch itself runs on the device; what can be checked without one is the
instrument: that the manifest is frozen with the declared allocation, that a
changed allocation cannot silently replace a frozen one, that the proposal seeds
are fresh and distinct, that a case's observation outcome is read separately
from its execution outcome, and that the budget statement counts every rejection
rather than a net number.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from test.tensor.campaigns import run_track_a_discovery as discovery


class ManifestFreezeTests(unittest.TestCase):
  def test_allocation_is_eight_configurations_across_two_halves(self):
    manifest = discovery.build_manifest(frozen_unix=1.0, frozen_utc="1970-01-01T00:00:01Z")
    configurations = manifest["configurations"]
    self.assertEqual(len(configurations), manifest["configuration_count"])
    self.assertEqual(len(configurations), 8)
    self.assertEqual(
      sum(entry["proposals"] for entry in configurations),
      manifest["proposal_total"],
    )
    self.assertEqual(manifest["proposal_total"], 128)
    established = [
      entry for entry in configurations if entry["half"] == "established"
    ]
    underexplored = [
      entry for entry in configurations if entry["half"] == "underexplored"
    ]
    self.assertEqual(len(established), 4)
    self.assertEqual(len(underexplored), 4)
    self.assertEqual(manifest["half_allocation"]["established"], 4)
    self.assertEqual(manifest["half_allocation"]["underexplored"], 4)
    for entry in configurations:
      self.assertEqual(entry["proposals"], discovery.PROPOSALS_PER_CONFIGURATION)
      self.assertEqual(len(entry["seeds"]), entry["proposals"])
      self.assertIn(entry["half"], {"established", "underexplored"})

  def test_established_half_is_the_recorded_families_at_the_recorded_shapes(self):
    manifest = discovery.build_manifest(frozen_unix=1.0, frozen_utc="now")
    established = {
      entry["family"]: entry
      for entry in manifest["configurations"]
      if entry["half"] == "established"
    }
    self.assertEqual(
      set(established),
      {
        "mixed-depth",
        "cross-state",
        "orthogonal-reductions",
        "reduction-arithmetic",
      },
    )
    for entry in established.values():
      # The recorded Track A first run measured exactly these shapes, bounds,
      # arithmetic mode, and domain.
      self.assertEqual(entry["shape"], [4, 8])
      self.assertEqual(entry["operation_mode"], "additive")
      self.assertEqual(entry["domain"], [0, 2])
    self.assertEqual(established["mixed-depth"]["iterations"], 3)
    for family in ("cross-state", "orthogonal-reductions", "reduction-arithmetic"):
      self.assertEqual(established[family]["iterations"], 2)

  def test_underexplored_half_covers_four_distinct_legal_categories(self):
    manifest = discovery.build_manifest(frozen_unix=1.0, frozen_utc="now")
    underexplored = [
      entry
      for entry in manifest["configurations"]
      if entry["half"] == "underexplored"
    ]
    categories = {entry["category"] for entry in underexplored}
    self.assertEqual(len(categories), 4)
    self.assertIn("arithmetic mode", categories)
    self.assertIn("larger trip count", categories)
    self.assertIn("shape at the profile's element cap", categories)
    self.assertIn("family at a shape it has not been run at", categories)
    for entry in underexplored:
      self.assertTrue(entry["evidence"])
      self.assertLessEqual(entry["elements"], discovery.tensor_region_generation.ELEMENT_CAP)

  def test_proposal_seeds_are_fresh_and_distinct(self):
    manifest = discovery.build_manifest(frozen_unix=1.0, frozen_utc="now")
    seeds = [
      seed for entry in manifest["configurations"] for seed in entry["seeds"]
    ]
    self.assertEqual(len(seeds), 128)
    self.assertEqual(len(set(seeds)), 128)
    # The recorded Track A evidence used seeds 0-9; a fresh proposal is a seed
    # that evidence never drew.
    self.assertTrue(all(seed >= 100000 for seed in seeds))
    self.assertFalse(set(seeds) & set(range(10)))

  def test_freeze_refuses_to_replace_a_different_manifest(self):
    manifest = discovery.build_manifest(frozen_unix=1.0, frozen_utc="first")
    with tempfile.TemporaryDirectory() as directory:
      args = _args(directory, frozen_unix=1.0, frozen_utc="first")
      first = discovery.step_freeze(Path(directory), args, {})
      self.assertEqual(first["status"], "written")
      again = discovery.step_freeze(Path(directory), args, {})
      self.assertEqual(again["status"], "verified")
      self.assertEqual(again["digest"], first["digest"])
      changed = _args(directory, frozen_unix=2.0, frozen_utc="second")
      with self.assertRaises(SystemExit):
        discovery.step_freeze(Path(directory), changed, {})
      on_disk = json.loads(
        (Path(directory) / discovery.MANIFEST_FILE).read_text(encoding="utf-8")
      )
      self.assertEqual(on_disk["digest"], first["digest"])
      self.assertEqual(on_disk["frozen_at_utc"], "first")
    self.assertEqual(
      manifest["digest"] if "digest" in manifest else first["digest"],
      first["digest"],
    )


class BudgetAccountingTests(unittest.TestCase):
  def test_observation_outcome_is_read_apart_from_execution(self):
    observed = {
      "observations": {
        "status": "observed",
        "surviving_layout_conversions": 2,
        "effective_layout_conversions_by_stage": {"ttgir": 3},
      },
      "passes": {
        "status": "observed",
        "conversion_movement_total": 0,
        "rematerialization": {"confirmed_total": 0},
      },
    }
    self.assertEqual(
      discovery.observation_outcome(observed)["status"],
      "target-behavior-observed",
    )
    quiet = {
      "observations": {
        "status": "observed",
        "surviving_layout_conversions": 0,
        "effective_layout_conversions_by_stage": {"ttgir": 3, "ttir": 0},
      },
      "passes": {
        "status": "observed",
        "conversion_movement_total": 0,
        "rematerialization": {"confirmed_total": 0},
      },
    }
    self.assertEqual(discovery.observation_outcome(quiet)["status"], "not-observed")
    unavailable = {
      "observations": {"status": "unavailable", "reason": "no archived compiler IR"}
    }
    outcome = discovery.observation_outcome(unavailable)
    self.assertEqual(outcome["status"], "unavailable")
    self.assertEqual(outcome["reason"], "no archived compiler IR")
    rematerialized = {
      "observations": {
        "status": "observed",
        "surviving_layout_conversions": 0,
        "effective_layout_conversions_by_stage": {},
      },
      "passes": {
        "status": "observed",
        "conversion_movement_total": 0,
        "rematerialization": {"confirmed_total": 1},
      },
    }
    self.assertEqual(
      discovery.observation_outcome(rematerialized)["status"],
      "target-behavior-observed",
    )

  def test_budget_keeps_rejections_by_reason_and_flags_all_rejected_half(self):
    report = {
      "frozen_manifest": {
        "configurations": discovery.configuration_records(),
      },
      "proposals": [
        {
          "id": "U1--seed100004",
          "configuration_id": "U1-mixed-depth-subtraction-4x8-it2",
          "half": "underexplored",
          "outcome": "rejected",
          "rejection_reason": "generated region may produce a negative exact value",
          "inputs": [],
        },
        {
          "id": "E1--seed100000",
          "configuration_id": "E1-mixed-depth-4x8-it3",
          "half": "established",
          "outcome": "admitted",
          "inputs": [
            {
              "name": "independent-small",
              "status": "built",
              "directory": "/nonexistent",
              "execution": {"status": "pass", "comparison": {"status": "pass"}},
              "observation": {"status": "not-observed"},
            }
          ],
        },
      ],
      "configurations": [],
    }
    report["configurations"] = discovery._configuration_summary(report)
    budget = discovery._budget(
      report,
      {"seconds": {"propose": 1.0, "build": 2.0, "run": 3.0}},
      _args("/tmp", shards=8),
    )
    self.assertEqual(budget["proposals_drawn"], 2)
    self.assertEqual(budget["proposals_admitted"], 1)
    self.assertEqual(budget["proposals_rejected"], 1)
    self.assertIn(
      "generated region may produce a negative exact value",
      budget["rejections_by_reason"],
    )
    self.assertEqual(budget["executions_attempted"], 1)
    self.assertEqual(budget["execution_outcomes"], {"pass": 1})
    self.assertEqual(budget["seconds"]["total_measured"], 6.0)
    self.assertFalse(budget["compilations"]["cache_reuse_visible"])

  def test_in_loop_removal_is_the_target_behavior_and_the_device_loop_is_separate(self):
    # The recorded Track A behavior: every conversion is removed, but a named
    # pass removes them while a loop construct is present before and after.
    in_loop = {
      "observations": {
        "status": "observed",
        "surviving_layout_conversions": 0,
        "effective_layout_conversions_by_stage": {"ttgir": 0, "ttir": 0},
        "device_backedge": False,
      },
      "passes": {
        "status": "observed",
        "conversion_movement_total": 0,
        "rematerialization": {"confirmed_total": 0},
        "in_loop_removals": [
          {
            "pass": "TritonGPURemoveLayoutConversions",
            "transition_index": 12,
            "loop_constructs_before": 1,
            "loop_constructs_after": 1,
            "conversions_removed": 9,
          }
        ],
      },
    }
    outcome = discovery.observation_outcome(in_loop)
    self.assertEqual(outcome["status"], "target-behavior-observed")
    self.assertEqual(outcome["removed_inside_a_loop"], 9)
    self.assertFalse(outcome["device_loop"])
    device = {
      "observations": {
        "status": "observed",
        "surviving_layout_conversions": 0,
        "effective_layout_conversions_by_stage": {},
        "device_backedge": True,
      },
      "passes": {
        "status": "observed",
        "conversion_movement_total": 0,
        "rematerialization": {"confirmed_total": 0},
      },
    }
    # A device backedge alone is stated, not counted as the layout behavior.
    self.assertEqual(
      discovery.observation_outcome(device)["status"], "not-observed"
    )
    self.assertTrue(discovery.observation_outcome(device)["device_loop"])

  def test_compaction_keeps_every_measurement_and_drops_the_duplicates(self):
    case = {
      "name": "independent-small",
      "status": "built",
      "admission": "admitted-exact-certificate",
      "reference_kind": "exact-certificate",
      "input_bytes_sha256": "a" * 64,
      "expected_sha256": "b" * 64,
      "replays": [
        {
          "reference_bits": 128,
          "sha256": "c" * 64,
          "matches_expected": True,
          "command": ["symiri", "--tensor-input", "/tmp/inputs.bin"],
        }
      ],
      "execution": {
        "status": "pass",
        "comparison": {"status": "pass"},
        "campaign_valid": True,
        "artifact_digests": {
          "runs/compiled/kernel.py": {"sha256": "d" * 64},
          "runs/compiled/actual.bin": {"sha256": "e" * 64},
        },
      },
      "certificate": {
        "admission": "admitted-exact-certificate",
        "reference_kind": "exact-certificate",
        "proof_summary": {"kind": "bounded-region-add-sum-f32-v1"},
        "replays": [{"reference_bits": 128, "command": ["symiri"]}],
      },
      "observations": {
        "status": "observed",
        "loop_status": "unrolled_or_removed",
        "backedges_by_stage": {"ptx": 0, "llir": 0},
        "surviving_layout_conversions": 0,
        "full_ir_digests": {"ttgir": {"sha256": "f" * 64}},
        "structural_ir_digests": {"ttgir": {"sha256": "0" * 64}},
      },
      "passes": {
        "status": "observed",
        "conversion_movement_total": 0,
        "rematerialization": {"confirmed_total": 0},
        "in_loop_removals": [{"pass": "P", "conversions_removed": 9}],
        "pass_totals": {"P": {"transitions": 1, "conversions_removed": 9}},
        "named_pass_occurrences": {"P": [{"transition_index": 1}]},
        "target_pass_evidence": {
          "P": {
            "status": "observed",
            "transition_indices": [1],
            "occurrences": [
              {
                "transition_index": 1,
                "conversion_match": {"removed": 9, "added": 0, "preserved": 0},
              }
            ],
          }
        },
        "changing_transitions": [
          {
            "pass": "P",
            "pipeline": "ttgir",
            "loop_construct_delta": 0,
            "layout_conversion_delta": -9,
            "effective_layout_conversion_delta": -9,
            "conversion_match": {"added": 0, "removed": 9},
            "conversion_movement": 0,
            "rematerialization": {"confirmed_count": 0},
          }
        ],
      },
    }
    report = {
      "schema": discovery.SCHEMA,
      "frozen_manifest": {"storage_policy": {"declared_before_the_run": True}},
      "proposals": [{"id": "P1", "inputs": [case]}],
      "budget": {"proposals_drawn": 1},
    }
    compact = discovery.compact_report(report, full_record="/tmp/full.json")
    kept = compact["proposals"][0]["inputs"][0]
    self.assertNotIn("passes", kept["observations"])
    self.assertNotIn("full_ir_digests", kept["observations"])
    self.assertIn("structural_ir_digests", kept["observations"])
    self.assertNotIn("named_pass_occurrences", kept["passes"])
    self.assertEqual(
      kept["passes"]["target_pass_evidence"]["P"]["occurrence_count"], 1
    )
    self.assertEqual(
      kept["passes"]["pass_totals"]["P"]["conversions_removed"], 9
    )
    self.assertEqual(kept["passes"]["in_loop_removals"][0]["conversions_removed"], 9)
    self.assertEqual(
      kept["passes"]["changing_transitions"][0]["conversion_match"]["removed"], 9
    )
    self.assertEqual(kept["execution"]["actual_sha256"], "e" * 64)
    self.assertEqual(kept["replays"][0]["sha256"], "c" * 64)
    self.assertNotIn("command", kept["replays"][0])
    self.assertNotIn("replays", kept["certificate"])
    self.assertEqual(compact["budget"], {"proposals_drawn": 1})
    self.assertEqual(compact["schema"], discovery.SCHEMA)


class ProvenanceTests(unittest.TestCase):
  def test_provenance_states_the_regions_own_request(self):
    manifest = discovery.build_manifest(frozen_unix=1.0, frozen_utc="now")
    configuration = manifest["configurations"][0]
    region = {
      "seed": 100000,
      "shape": [4, 8],
      "iterations": 3,
      "family": "mixed-depth",
      "operation_mode": "additive",
      "max_operators": 10,
    }
    source = "define void @kernel() { ret void }\n"
    provenance = discovery._provenance(
      configuration, region, source, {"status": "admitted", "target": 1}, (0, 2)
    )
    self.assertEqual(provenance["region"], region)
    self.assertEqual(provenance["family"], configuration["family"])
    self.assertEqual(provenance["input_domain"], {"lo": 0, "hi": 2})
    self.assertEqual(
      provenance["program_sha256"], discovery._sha256(source.encode("utf-8"))
    )
    self.assertEqual(provenance["reference_case"]["admission"], "admitted")
    # A configuration without a region is not a region statement: the region is
    # an argument, so it cannot be silently taken from the wrong place.
    with self.assertRaises(TypeError):
      discovery._provenance(configuration, source, {"status": "admitted"}, (0, 2))


def _args(work, **overrides):
  class Args:
    pass

  args = Args()
  args.work = str(work)
  args.output = str(Path(work) / "out.json")
  args.steps = "freeze"
  args.exporter = "exporter"
  args.symiri = "symiri"
  args.symirc = "symirc"
  args.stage_timeout = 1.0
  args.run_timeout = 1.0
  args.shards = 1
  args.devices = 1
  args.shard = None
  args.revision = "revision"
  args.compiler_flags = "flags"
  args.import_root = str(discovery.TREE)
  args.pass_dump = True
  args.frozen_unix = 1.0
  args.frozen_utc = "now"
  args.force = False
  args.environment = {}
  args.controls = ""
  args.control_seed = 9
  args.case = None
  args.smoke_devices = False
  for key, value in overrides.items():
    setattr(args, key, value)
  return args


if __name__ == "__main__":
  unittest.main()
