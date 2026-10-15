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
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test.tensor.campaigns import run_track_a_discovery as discovery


class ManifestFreezeTests(unittest.TestCase):
  def test_allocation_is_eight_configurations_across_two_halves(self):
    manifest = discovery.build_manifest(
      frozen_unix=1.0, frozen_utc="1970-01-01T00:00:01Z"
    )
    configurations = manifest["configurations"]
    self.assertEqual(len(configurations), manifest["configuration_count"])
    self.assertEqual(len(configurations), 8)
    self.assertEqual(
      sum(entry["proposals"] for entry in configurations),
      manifest["proposal_total"],
    )
    self.assertEqual(manifest["proposal_total"], 128)
    established = [entry for entry in configurations if entry["half"] == "established"]
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
      entry for entry in manifest["configurations"] if entry["half"] == "underexplored"
    ]
    categories = {entry["category"] for entry in underexplored}
    self.assertEqual(len(categories), 4)
    self.assertIn("arithmetic mode", categories)
    self.assertIn("larger trip count", categories)
    self.assertIn("shape at the profile's element cap", categories)
    self.assertIn("family at a shape it has not been run at", categories)
    for entry in underexplored:
      self.assertTrue(entry["evidence"])
      self.assertLessEqual(
        entry["elements"], discovery.tensor_region_generation.ELEMENT_CAP
      )

  def test_proposal_seeds_are_fresh_and_distinct(self):
    manifest = discovery.build_manifest(frozen_unix=1.0, frozen_utc="now")
    seeds = [seed for entry in manifest["configurations"] for seed in entry["seeds"]]
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
    # The recorded Track A behavior: a named pass removes conversions from
    # inside a loop that is present before and after, and the nine that the
    # loop enclosed are the ones that count, not the whole transition's twelve.
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
            "conversions_removed": 12,
            "removed_inside_a_loop": 9,
            "removed_in_loop_regions": {"scf.for": 9},
            "removed_outside_a_loop": 3,
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
    self.assertEqual(discovery.observation_outcome(device)["status"], "not-observed")
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
        "in_loop_removals": [
          {"pass": "P", "conversions_removed": 9, "removed_inside_a_loop": 9}
        ],
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
    self.assertEqual(kept["passes"]["target_pass_evidence"]["P"]["occurrence_count"], 1)
    self.assertEqual(kept["passes"]["pass_totals"]["P"]["conversions_removed"], 9)
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


class TrackAVerdictTests(unittest.TestCase):
  """In-loop attribution and the byte check, at the verdict they produce.

  The pass dump of one compilation is a sequence of module snapshots, each
  printed before its own pass runs, so a transition is measured from the
  snapshot named for the pass to the snapshot that follows it.
  """

  def test_removal_outside_a_loop_does_not_qualify_the_run(self):
    # The recorded cross-state shape: the named pass removes a conversion while
    # the loop is present before and after, but the removed conversion sits at
    # function scope, so no conversion was removed from inside the loop.
    with tempfile.TemporaryDirectory() as directory:
      case = _case_directory(
        directory,
        "independent-small",
        dump=_removal_dump(_ttgir(), _ttgir(function_conversion=False)),
        ir={
          "ttir": _ttgir(),
          "ttgir": _ttgir(function_conversion=False, loop_conversion=False),
        },
      )
      summary = discovery._summarize_observations(case)
      passes = summary.pop("passes")

    self.assertEqual(passes["in_loop_removals"], [])
    outcome = discovery.observation_outcome({"observations": summary, "passes": passes})
    self.assertEqual(outcome["status"], "not-observed")
    # The removal is not dropped from the reading: the outcome says what the
    # pass removed and where it did not remove it from.
    self.assertEqual(
      outcome["reasons"],
      [
        "1 layout conversion(s) removed by TritonGPURemoveLayoutConversions, "
        "none from inside a loop"
      ],
    )

  def test_removal_inside_a_loop_qualifies_the_run(self):
    with tempfile.TemporaryDirectory() as directory:
      case = _case_directory(
        directory,
        "independent-small",
        dump=_removal_dump(_ttgir(), _ttgir(loop_conversion=False)),
        ir={
          "ttir": _ttgir(),
          "ttgir": _ttgir(function_conversion=False, loop_conversion=False),
        },
      )
      summary = discovery._summarize_observations(case)
      passes = summary.pop("passes")

    entries = passes["in_loop_removals"]
    self.assertEqual(len(entries), 1)
    self.assertEqual(entries[0]["conversions_removed"], 1)
    self.assertEqual(entries[0]["removed_inside_a_loop"], 1)
    self.assertEqual(entries[0]["removed_in_loop_regions"], {"scf.for": 1})
    self.assertEqual(entries[0]["removed_outside_a_loop"], 0)
    outcome = discovery.observation_outcome({"observations": summary, "passes": passes})
    self.assertEqual(outcome["status"], "target-behavior-observed")
    self.assertEqual(outcome["removed_inside_a_loop"], 1)

  def test_configuration_with_no_in_loop_removal_observes_nothing(self):
    # A whole configuration whose every removal is at function scope: the run
    # rate is re-derived, and the published number stays visible beside it.
    with tempfile.TemporaryDirectory() as directory:
      configuration = discovery.configuration_records()[0]
      published = Path(directory) / "published.json"
      published.write_text(
        json.dumps(
          {
            "schema": discovery.SCHEMA,
            "ledger": {"driver_sha256": "0" * 64},
            "configurations": [
              {
                "id": configuration["id"],
                "observation_outcomes": {"target-behavior-observed": 2},
              }
            ],
            "budget": {"observation_outcomes": {"target-behavior-observed": 2}},
          }
        ),
        encoding="utf-8",
      )
      cases = [
        {
          "name": name,
          "expected": _ZERO,
          "actual": _ZERO,
          "dump": _removal_dump(_ttgir(), _ttgir(function_conversion=False)),
          "ir": {
            "ttir": _ttgir(),
            "ttgir": _ttgir(function_conversion=False, loop_conversion=False),
          },
        }
        for name in ("independent-small", "independent-large")
      ]
      record = _discovery_record(directory, cases, supersedes=published)

    summary = record["configurations"][0]
    self.assertEqual(summary["observation_outcomes"], {"not-observed": 2})
    self.assertEqual(record["budget"]["observation_outcomes"], {"not-observed": 2})
    self.assertEqual(
      summary["observation_outcomes_superseded"]["counts"],
      {"target-behavior-observed": 2},
    )
    self.assertEqual(record["corrections"]["supersedes"]["driver_sha256"], "0" * 64)

  def test_signed_zero_byte_mismatch_fails_the_execution_verdict(self):
    # The reviewer's reproduction: a numeric comparison that passes while the
    # reference word `+0.0` is not the device word `-0.0`.
    with tempfile.TemporaryDirectory() as directory:
      record = _discovery_record(
        directory,
        [
          {
            "name": "independent-small",
            "expected": _ZERO,
            "actual": struct.pack("<I", 0x80000000),
          }
        ],
      )

    self.assertEqual(record["budget"]["executions_attempted"], 1)
    self.assertEqual(record["budget"]["execution_outcomes"], {"fail": 1})
    self.assertEqual(record["budget"]["comparison_outcomes"], {"fail": 1})
    self.assertEqual(record["configurations"][0]["executions_passed"], 0)
    case = record["proposals"][0]["inputs"][0]
    self.assertEqual(case["bit_level_check"]["status"], "exact_mismatch")
    self.assertEqual(case["execution"]["status"], "fail")
    self.assertEqual(case["execution"]["comparison"]["status"], "fail")
    self.assertTrue(case["execution"]["byte_check_applied"])
    self.assertEqual(case["execution"]["byte_check_status"], "exact_mismatch")

  def test_equal_bytes_still_pass_the_execution_verdict(self):
    with tempfile.TemporaryDirectory() as directory:
      record = _discovery_record(
        directory,
        [{"name": "independent-small", "expected": _ZERO, "actual": _ZERO}],
      )

    self.assertEqual(record["budget"]["execution_outcomes"], {"pass": 1})
    self.assertEqual(record["budget"]["comparison_outcomes"], {"pass": 1})
    self.assertEqual(record["configurations"][0]["executions_passed"], 1)
    case = record["proposals"][0]["inputs"][0]
    self.assertEqual(case["execution"]["status"], "pass")
    self.assertEqual(case["execution"]["comparison"]["status"], "pass")
    self.assertTrue(case["execution"]["byte_check_applied"])
    self.assertEqual(case["execution"]["byte_check_status"], "pass")

  def test_one_pass_removing_inside_and_outside_a_loop_counts_the_inside(self):
    # The smallest mixed case: one pass removes two identical-shape conversions
    # of which exactly one sits inside the loop it retains.
    with tempfile.TemporaryDirectory() as directory:
      case = _case_directory(
        directory,
        "independent-small",
        dump=_removal_dump(
          _ttgir(), _ttgir(function_conversion=False, loop_conversion=False)
        ),
        ir={
          "ttir": _ttgir(),
          "ttgir": _ttgir(function_conversion=False, loop_conversion=False),
        },
      )
      summary = discovery._summarize_observations(case)
      passes = summary.pop("passes")

    entry = passes["in_loop_removals"][0]
    self.assertEqual(entry["conversions_removed"], 2)
    self.assertEqual(entry["removed_inside_a_loop"], 1)
    self.assertEqual(entry["removed_outside_a_loop"], 1)
    outcome = discovery.observation_outcome({"observations": summary, "passes": passes})
    self.assertEqual(outcome["removed_inside_a_loop"], 1)
    self.assertEqual(outcome["status"], "target-behavior-observed")


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
  args.full_output = None
  args.supersedes = None
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


_ZERO = struct.pack("<f", 0.0)

# The four stages the driver reads are named by the files the runner archives;
# one module of TTGIR carries a function-scope conversion, a loop construct,
# and a conversion inside that construct, which is the only difference the
# attribution has to see.
_TTGIR_HEADER = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>
#slice = #ttg.slice<{dim = 0, parent = #blocked}>
#loc1 = loc("kernel.py":11:1)
#loc2 = loc("kernel.py":12:1)
"""

_FUNCTION_CONVERSION = (
  "    %0 = ttg.convert_layout %a : tensor<4xf32, #blocked> -> "
  "tensor<4xf32, #slice> loc(#loc1)\n"
)
_LOOP_CONVERSION = (
  "      %1 = ttg.convert_layout %b : tensor<4xf32, #blocked> -> "
  "tensor<4xf32, #slice> loc(#loc2)\n"
)


def _ttgir(*, function_conversion=True, loop_conversion=True):
  """One TTGIR module, with or without each of its two conversions."""
  return (
    _TTGIR_HEADER
    + "module {\n"
    + "  tt.func public @kernel() {\n"
    + (_FUNCTION_CONVERSION if function_conversion else "")
    + "    scf.for %i = %c0 to %c2 step %c1 {\n"
    + (_LOOP_CONVERSION if loop_conversion else "")
    + "      scf.yield\n"
    + "    }\n"
    + "    return\n"
    + "  }\n"
    + "}\n"
  )


def _pass_snapshot(name, ir):
  return (
    f"// -----// IR Dump Before {name}: {name.lower()} "
    "('builtin.module' operation) //----- //\n"
    f"{ir}\n"
  )


def _removal_dump(before, after, *, pass_name="TritonGPURemoveLayoutConversions"):
  """A compilation whose named pass turns `before` into `after`."""
  return "".join(
    (
      _pass_snapshot("GluonInline", before),
      _pass_snapshot(pass_name, before),
      _pass_snapshot("LLVMDIScope", after),
    )
  )


def _case_directory(root, name, *, expected=_ZERO, actual=_ZERO, dump=None, ir=None):
  """One built case directory, with the files the report step reads."""
  directory = Path(root) / name
  compiled = directory / "runs" / "compiled"
  compiled.mkdir(parents=True)
  (directory / "expected.bin").write_bytes(expected)
  (compiled / "actual.bin").write_bytes(actual)
  if dump is not None:
    (compiled / "compiler-pass-dump.txt").write_text(dump, encoding="utf-8")
  for stage, text in (ir or {}).items():
    (compiled / f"compiler-{stage}.ir").write_text(text, encoding="utf-8")
  (directory / "proposal-record.json").write_text(
    json.dumps(
      {
        "proposal": {"name": name},
        "certificate": {
          "admission": "admitted-exact-certificate",
          "reference_kind": "exact-certificate",
          "proof_summary": {},
        },
        "replays": [],
        "loader": {"status": "admitted"},
      }
    ),
    encoding="utf-8",
  )
  return directory


_NUMERIC_PASS = {
  "status": "pass",
  "comparison": {"status": "pass", "mismatches": []},
  "campaign_valid": True,
  "exit_code": 0,
  "reason": None,
  "artifact_digests": {},
}

_ENVIRONMENT = {
  "gpus": [],
  "driver_version": "535.274.02",
  "cuda_version": "12.1",
  "python_version": "3.12.15",
  "torch_version": "2.5.1+cu121",
  "triton_version": "3.9.0+git0000000",
  "triton_source_revision": "0" * 40,
  "campaign_available": True,
  "failure_reasons": [],
  "package_lock_sha256": "e" * 64,
}


def _discovery_record(root, cases, *, supersedes=None):
  """Drive the report step over synthetic cases and return its deliverable.

  The environment probe is stubbed: it hashes the host's installed packages,
  which says nothing about the aggregation these cases exercise.
  """
  work = Path(root) / "work"
  work.mkdir()
  args = _args(work, supersedes=str(supersedes) if supersedes else None)
  discovery.step_freeze(work, args, {})
  manifest = discovery.build_manifest(
    frozen_unix=args.frozen_unix, frozen_utc=args.frozen_utc
  )
  configuration = manifest["configurations"][0]
  seed = configuration["seeds"][0]
  identifier = discovery.proposal_id(configuration, seed)
  inputs = []
  runs = {}
  for case in cases:
    directory = _case_directory(
      work / "suite" / identifier,
      case["name"],
      expected=case.get("expected", _ZERO),
      actual=case.get("actual", _ZERO),
      dump=case.get("dump"),
      ir=case.get("ir"),
    )
    inputs.append(
      {
        "name": case["name"],
        "status": "built",
        "directory": str(directory),
        "admission": "admitted-exact-certificate",
        "reference_kind": "exact-certificate",
        "expected_sha256": "a" * 64,
        "input_bytes_sha256": "b" * 64,
      }
    )
    runs[f"{identifier}/{case['name']}"] = {
      "report": dict(case.get("numeric") or _NUMERIC_PASS),
      "seconds": 1.0,
    }
  discovery._write_json(
    work / "proposals.json",
    {
      proposal_id: {
        "configuration_id": entry["id"],
        "half": entry["half"],
        "category": entry["category"],
        "seed": drawn_seed,
        "region": {},
        "region_identity": {},
        "generation": {},
        "admission": {"status": "admitted"},
      }
      for entry in manifest["configurations"]
      for drawn_seed in entry["seeds"]
      for proposal_id in (discovery.proposal_id(entry, drawn_seed),)
    },
  )
  discovery._write_json(
    work / "build-records.json",
    {identifier: {"status": "built", "inputs": inputs}},
  )
  discovery._write_json(work / "run-records.json", runs)
  with mock.patch.object(
    discovery.probe_triton_env, "collect_environment", return_value=_ENVIRONMENT
  ):
    discovery.step_report(work, args, {})
  return json.loads(Path(args.output).read_text(encoding="utf-8"))


if __name__ == "__main__":
  unittest.main()
