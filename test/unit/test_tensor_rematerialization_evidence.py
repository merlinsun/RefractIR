"""Host tests for the broadcast-hoist rematerialization evidence.

Two readings are covered here and nowhere else. A consumer count is a count of
the operations that read a value - a definition is not its own consumer, and one
operation is one consumer however often it prints its operand - and the summary
keeps what the raw build and run records hold: per-arm provenance, the
qualification record, the input, reference, and observed digests, and every
refusal, instead of dropping them when it reports a pass. The readings build on
the shared operand scan in `tensor_compiler_observation` and live in the evidence
module, which is covered here, so a rewrite of that scan cannot take them with it.
"""

import json
import tempfile
import unittest
from pathlib import Path

from test.lib import tensor_rematerialization_evidence

_IMPORT_ROOT = "/host/remat-20261016/refractir"

_PROVENANCE = {
  "import_root": _IMPORT_ROOT,
  "imported_modules": {
    "root": _IMPORT_ROOT,
    "sha256": {
      "test/lib/tensor_compiler_observation.py": "ab" * 32,
      "test/lib/tensor_region_generation.py": "cd" * 32,
    },
  },
  "regenerated_by": {
    "library": {
      "module": "test.lib.tensor_rematerialization_evidence",
      "sha256": "cd" * 32,
    },
    "raw_records": {"build": "work/build.json", "runs": "work/runs.json"},
  },
  "field_provenance": {
    "cases[].analysis": "test.lib.tensor_compiler_observation",
  },
}

_BUILD = {
  "broadcast-hoist-4x8-it4": {
    "id": "broadcast-hoist-4x8-it4",
    "status": "built",
    "selection": "solver",
    "program_sha256": "11" * 32,
    "kernel_sha256": "22" * 32,
    "configuration": {
      "family": "broadcast-hoist",
      "shape": [4, 8],
      "iterations": 4,
      "seed": 9,
      "operation_mode": "additive",
      "max_operators": 10,
    },
    "bounds": {"integer_exactness_bound": 16777216, "max_partial_sum": 9268480},
    "region_pattern": {"view": "t2", "consumers": ["t3", "next_y"]},
  },
  "broadcast-hoist-32x16-it4": {
    "id": "broadcast-hoist-32x16-it4",
    "status": "certificate_refused",
    "reason": "materialized value or reduction partial sum exceeds exact range",
    "source_sha256": "33" * 32,
    "configuration": {
      "family": "broadcast-hoist",
      "shape": [32, 16],
      "iterations": 4,
      "seed": 9,
      "operation_mode": "additive",
      "max_operators": 10,
    },
    "steps": [{"step": "certify", "seconds": 0.84}],
  },
}

_RUNS = {
  "broadcast-hoist-4x8-it4": {
    "status": "pass",
    "program_sha256": "11" * 32,
    "kernel_sha256": "22" * 32,
    "manifest": {"profile": "tensor-v2"},
    "pass_dump": {"path": "/work/dump.txt", "bytes": 12, "sha256": "44" * 32},
    "report": {
      "status": "pass",
      "mode": "compiled",
      "exit_code": 0,
      "comparison": {"status": "pass", "mismatches": []},
      "campaign_valid": False,
      "actual_sha256": "55" * 32,
      "actual_bytes": 128,
      "artifact_digests": {
        "inputs.bin": {"bytes": 128, "sha256": "66" * 32},
        "runs/compiled/compiler-ttgir.ir": {"bytes": 7492, "sha256": "77" * 32},
      },
    },
  },
}

_ANALYSIS = {
  "cases": {
    "broadcast-hoist-4x8-it4": {
      "status": "pass",
      "source_pattern": {"status": "view_present", "view": "t2"},
      "reduction_axes": [0, 1],
      "analysis": {"confirmed_total": 0},
    }
  },
  "matcher": "observe_pass_dumps(..., rematerialization=True)",
  "structural_reading": "one operation repeated on one unchanged operand set",
  "interpretation_limit": "the structural reading is not the confirmation criterion",
  "matcher_boundary_probe": {"same_encoding_copy": {"confirmed_count": 1}},
}


def _write_case(directory):
  """Write the raw records a built and executed arm leaves in its case dir."""
  case = {
    "schema": "refractir.tensor-case.v2",
    "profile": "tensor-v2",
    "label": "broadcast-hoist [4,8] x 4",
    "comparison": {"mode": "exact", "atol": [0, 1], "rtol": [0, 1]},
    "evidence": {
      "kind": "exact_smt",
      "status": "exact_smt_checked",
      "artifacts": ["program.sir", "inputs.bin", "expected.bin"],
    },
    "property": {"id": "output-element-0-0-equals-10788808", "source": "..."},
    "coverage": {
      "input_behavior": {"input_mode": "bounded-integer-twin-smt", "policy": "exact"},
      "loop_structure": [{"bound": 4}],
    },
    "reference": {"defined": True, "exactness": {"bound_method": "affine forms"}},
    "files": {
      "inputs.bin": {"bytes": 128, "sha256": "66" * 32},
      "expected.bin": {"bytes": 128, "sha256": "88" * 32},
    },
  }
  run = {
    "schema": "refractir.tensor-run.v1",
    "started_at": "2026-10-15T22:36:22.124612+00:00",
    "mode": "compiled",
    "refractir_revision": "unavailable",
    "working_tree_dirty": "unknown",
    "compiler": {
      "executable": f"{_IMPORT_ROOT}/symirc",
      "flags": "unavailable",
      "sha256": "99" * 32,
    },
    "environment": "runs/compiled/environment.json",
  }
  environment = {
    "schema": "refractir.tensor-environment.v1",
    "status": "environment_available",
    "campaign_available": True,
    "cuda_version": "12.1",
    "driver_version": "535.274.02",
    "python_version": "3.12.15",
    "torch_version": "2.5.1+cu121",
    "triton_version": "3.9.0+git32f06df4",
    "triton_source_revision": "32f06df4bf1c30885ff61ee3488836ca81ed65f0",
    "gpus": [
      {"name": "Tesla V100-SXM2-32GB", "compute_capability": "7.0"},
    ]
    * 8,
  }
  directory.mkdir(parents=True, exist_ok=True)
  (directory / "case.json").write_text(json.dumps(case), encoding="utf-8")
  compiled = directory / "runs" / "compiled"
  compiled.mkdir(parents=True, exist_ok=True)
  (compiled / "run.json").write_text(json.dumps(run), encoding="utf-8")
  (compiled / "environment.json").write_text(json.dumps(environment), encoding="utf-8")


class TensorConsumerReadingTests(unittest.TestCase):
  """A consumer count states the operations that read a value."""

  def test_a_value_with_one_consumer_reports_one_consumer(self):
    ir = """\
%v = tt.broadcast %x : tensor<1x8xf32, #blocked> -> tensor<4x8xf32, #blocked>
%c = ttg.convert_layout %v : tensor<4x8xf32, #blocked> -> tensor<4x8xf32, #slice>
"""
    self.assertEqual(
      tensor_rematerialization_evidence.consuming_operations(ir, "%v"), ("%c",)
    )

  def test_a_value_mentioned_only_by_its_own_definition_reports_no_consumer(self):
    ir = """\
%v = tt.broadcast %x : tensor<1x8xf32, #blocked> -> tensor<4x8xf32, #blocked>
"""
    self.assertEqual(
      tensor_rematerialization_evidence.consuming_operations(ir, "%v"), ()
    )
    self.assertEqual(
      tensor_rematerialization_evidence.consuming_operations(ir, "%x"), ("%v",)
    )

  def test_the_generated_state_view_is_located_by_its_reduction_or_reported_absent(
    self,
  ):
    ir = """\
%offsets = tt.broadcast %offsets_1 : tensor<4x1xi32, #blocked> -> tensor<4x8xi32, #blocked>
%offsets_2 = ttg.convert_layout %offsets : tensor<4x8xi32, #blocked> -> tensor<4x8xi32, #slice>
%t1 = "tt.reduce"(%acc_x) <{axis = 0 : i32}> ({
^bb0(%a: f32, %b: f32):
  %sum = arith.addf %a, %b : f32
  tt.reduce.return %sum : f32
}) : (tensor<4x8xf32, #blocked>) -> tensor<8xf32, #blocked>
%t1e = tt.expand_dims %t1 {axis = 0 : i32} : tensor<8xf32, #blocked> -> tensor<1x8xf32, #blocked>
%t2 = tt.broadcast %t1e : tensor<1x8xf32, #blocked> -> tensor<4x8xf32, #blocked>
%t3 = "tt.reduce"(%t2) <{axis = 1 : i32}> ({
^bb0(%c: f32, %d: f32):
  %sum2 = arith.addf %c, %d : f32
  tt.reduce.return %sum2 : f32
}) : (tensor<4x8xf32, #blocked>) -> tensor<4xf32, #blocked>
%next_y = arith.addf %acc_y, %t2 : tensor<4x8xf32, #blocked>
"""
    reading = tensor_rematerialization_evidence.broadcast_state_views(ir)

    self.assertEqual(reading["status"], "view_present")
    view = [entry for entry in reading["views"] if entry["value"] == "%t2"][0]
    self.assertEqual(view["reduction"], "%t1")
    self.assertEqual(view["consumers"], ("%next_y", "%t3"))
    self.assertNotIn("%offsets", [entry["value"] for entry in reading["views"]])

    absent = tensor_rematerialization_evidence.broadcast_state_views(
      """\
%offsets = tt.broadcast %offsets_1 : tensor<4x1xi32, #blocked> -> tensor<4x8xi32, #blocked>
%offsets_2 = ttg.convert_layout %offsets : tensor<4x8xi32, #blocked> -> tensor<4x8xi32, #slice>
"""
    )
    self.assertEqual(absent["status"], "no_view")
    self.assertTrue(absent["reason"], "an absent view states why it is absent")

  def test_an_operand_named_inside_a_nested_generic_form_counts_one_consumer(self):
    """One operation stays one consumer when its regions name the operand too."""
    ir = """\
%v = tt.broadcast %x : tensor<1x8xf32, #blocked> -> tensor<4x8xf32, #blocked>
%r = "scf.if"(%cond, %v) ({^bb0: scf.yield %v : tensor<4x8xf32, #blocked>}) ({^bb0: scf.yield %v : tensor<4x8xf32, #blocked>}) : (i1, tensor<4x8xf32, #blocked>) -> tensor<4x8xf32, #blocked>
"""
    self.assertEqual(
      tensor_rematerialization_evidence.consuming_operations(ir, "%v"), ("%r",)
    )


class TensorRematerializationRecordTests(unittest.TestCase):
  """The summary is rebuilt from the raw records, refusals included."""

  def test_a_certificate_refusal_is_a_first_class_arm(self):
    record = tensor_rematerialization_evidence.assemble(
      analysis=_ANALYSIS,
      build=_BUILD,
      runs=_RUNS,
      case_dirs={},
      frozen_corpus={},
      probes={},
      provenance=_PROVENANCE,
    )

    arm = record["cases"]["broadcast-hoist-32x16-it4"]
    self.assertEqual(arm["status"], "certificate_refused")
    self.assertIn("exceeds exact range", arm["refusal"]["reason"])
    self.assertEqual(arm["refusal"]["steps"], [{"step": "certify", "seconds": 0.84}])
    self.assertIsNone(arm["execution"])
    self.assertEqual(arm["configuration"]["iterations"], 4)

  def test_every_arm_carries_its_provenance_and_digests(self):
    previous = {
      "schema": "refractir.tensor-rematerialization-broadcast-hoist.v1",
      "module_digests": {
        "test/lib/tensor_compiler_observation.py": "ab" * 32,
        "test/lib/tensor_region_generation.py": "42b15b3d" + "0" * 56,
      },
      "matcher_boundary_probe": {
        "same_encoding_copy": {
          "status": "confirmed_only",
          "confirmed_count": 1,
          "unattributed_count": 0,
        },
        "layout_rematerialization": {
          "status": "none",
          "confirmed_count": 0,
          "unattributed_count": 0,
        },
      },
      "cases": {
        "broadcast-hoist-4x8-it4": {
          "run_status": "pass",
          "analysis": {
            "transitions": [
              {
                "index": 12,
                "matcher": {"status": "none", "confirmed_count": 0},
                "opportunities_before": [
                  {"conversion": "%view", "source_consumers": 2},
                ],
              }
            ]
          },
        }
      },
    }
    analysis = json.loads(json.dumps(_ANALYSIS))
    analysis["matcher_boundary_probe"] = {
      "same_encoding_copy": {
        "status": "confirmed_only",
        "confirmed_count": 1,
        "unattributed_count": 0,
      },
      "layout_rematerialization": {
        "status": "confirmed_only",
        "confirmed_count": 1,
        "unattributed_count": 0,
        "encoding_change": {"changed": True},
      },
      "library": {
        "module": "test.lib.tensor_compiler_observation",
        "sha256": "ef" * 32,
      },
    }
    analysis["cases"]["broadcast-hoist-4x8-it4"]["analysis"] = {
      "confirmed_total": 0,
      "transitions": [
        {
          "index": 12,
          "matcher": {"status": "confirmed_only", "confirmed_count": 1},
          "opportunities_before": [
            {"conversion": "%view", "source_consumer_count": 1},
          ],
        }
      ],
    }
    with tempfile.TemporaryDirectory() as directory:
      case_dir = Path(directory) / "broadcast-hoist-4x8-it4"
      _write_case(case_dir)
      record = tensor_rematerialization_evidence.assemble(
        analysis=analysis,
        build=_BUILD,
        runs=_RUNS,
        case_dirs={"broadcast-hoist-4x8-it4": case_dir},
        frozen_corpus={},
        probes={},
        provenance=_PROVENANCE,
        previous=previous,
      )

    for case_id, arm in record["cases"].items():
      with self.subTest(case=case_id):
        self.assertEqual(arm["provenance"]["import_root"], _IMPORT_ROOT)
        self.assertTrue(arm["provenance"]["recorded_by"])

    arm = record["cases"]["broadcast-hoist-4x8-it4"]
    self.assertEqual(arm["run_status"], "pass")
    self.assertEqual(arm["execution"]["status"], "pass")
    self.assertEqual(arm["execution"]["campaign_valid"], False)
    self.assertEqual(arm["provenance"]["refractir"]["revision"], "unavailable")
    self.assertEqual(
      arm["provenance"]["refractir"]["tree"], "/host/remat-20261016/refractir"
    )
    self.assertEqual(arm["provenance"]["compiler"]["flags"], "unavailable")
    self.assertEqual(arm["provenance"]["compiler"]["sha256"], "99" * 32)
    self.assertEqual(arm["provenance"]["triton"]["version"], "3.9.0+git32f06df4")
    self.assertEqual(
      arm["provenance"]["environment"]["device"], "Tesla V100-SXM2-32GB (sm_70) x8"
    )
    self.assertEqual(arm["provenance"]["environment"]["driver"], "535.274.02")
    self.assertEqual(arm["digests"]["inputs"]["sha256"], "66" * 32)
    self.assertEqual(arm["digests"]["reference"]["sha256"], "88" * 32)
    self.assertEqual(arm["digests"]["observed"]["sha256"], "55" * 32)
    self.assertEqual(arm["digests"]["pass_dump"]["sha256"], "44" * 32)
    self.assertEqual(arm["qualification"]["evidence"]["status"], "exact_smt_checked")
    self.assertEqual(
      arm["qualification"]["property"]["id"], "output-element-0-0-equals-10788808"
    )

    superseded = {entry["path"]: entry for entry in record["superseded"]}
    consumer_path = (
      "cases[broadcast-hoist-4x8-it4].analysis.transitions[12]"
      ".opportunities_before[%view].source_consumers"
    )
    self.assertEqual(superseded[consumer_path]["previous"], 2)
    self.assertEqual(superseded[consumer_path]["corrected"], 1)
    self.assertEqual(
      superseded["cases[broadcast-hoist-32x16-it4]"]["corrected"],
      "certificate_refused",
    )
    self.assertEqual(
      superseded["imported_modules[test/lib/tensor_region_generation.py]"]["corrected"],
      "cd" * 32,
    )
    probe_path = "matcher_boundary_probe[layout_rematerialization].confirmed_count"
    self.assertEqual(superseded[probe_path]["previous"], 0)
    self.assertEqual(superseded[probe_path]["corrected"], 1)
    self.assertNotIn(
      "matcher_boundary_probe[same_encoding_copy].status",
      superseded,
      "a probe reading that did not move is not restated as superseded",
    )

    drift = record["matcher_rederivation"]
    self.assertEqual(drift["arms_compared"], ["broadcast-hoist-4x8-it4"])
    self.assertEqual([entry["field"] for entry in drift["changed"]], ["matcher"])
    self.assertEqual(
      drift["changed"][0]["previous"], {"status": "none", "confirmed_count": 0}
    )
    self.assertEqual(
      drift["changed"][0]["corrected"],
      {"status": "confirmed_only", "confirmed_count": 1},
    )


def _arm(transitions):
  return {"analysis": {"transitions": transitions}}


def _ledger_record(count, consumers):
  """One case with one transition and one opportunity, in either reading shape."""
  opportunity = {"conversion": "ttg.convert_layout %c"}
  if count is not None:
    opportunity["source_consumer_count"] = count
  if consumers is not None:
    opportunity["source_consumers"] = consumers
  transition = {"index": 0, "pass": "Pass0", "opportunities_before": [opportunity]}
  return {
    "cases": {"arm": {"analysis": {"transitions": [transition]}}},
    "provenance": {},
  }


class TensorRematerializationDriftTests(unittest.TestCase):
  """The drift check compares transition identities, not list positions."""

  _PREVIOUS = [
    {"index": index, "pass": f"Pass{index}", "matcher": {"confirmed_count": 0}}
    for index in range(26)
  ]

  def test_an_arm_that_lost_every_transition_reports_them_missing(self):
    # The corrected record holds no transition at all while the recorded one
    # held 26: comparing overlapping positions alone let the whole arm pass as
    # unchanged. Every missing identity is named, and the arm stays compared.
    previous = {"cases": {"broadcast-hoist-32x16-it2": _arm(list(self._PREVIOUS))}}
    cases = {"broadcast-hoist-32x16-it2": _arm([])}

    drift = tensor_rematerialization_evidence.matcher_drift(previous, cases)

    self.assertEqual(drift["arms_compared"], ["broadcast-hoist-32x16-it2"])
    self.assertEqual(drift["changed"], [])
    self.assertEqual(drift["removed_transition_count"], 26)
    self.assertEqual(drift["added_transition_count"], 0)
    self.assertEqual(len(drift["removed_transitions"]), 26)
    self.assertEqual(
      [entry["index"] for entry in drift["removed_transitions"]], list(range(26))
    )
    self.assertEqual(drift["removed_transitions"][0]["pass"], "Pass0")

  def test_a_shortened_list_reports_only_the_identities_that_are_gone(self):
    # Ten of the 26 recorded transitions survive in the corrected record. The
    # 16 that are gone are reported one by one, with the pass that produced
    # each, so a shortened list cannot pass silently.
    previous = {"cases": {"arm": _arm(list(self._PREVIOUS))}}
    cases = {"arm": _arm([dict(entry) for entry in self._PREVIOUS[:10]])}

    drift = tensor_rematerialization_evidence.matcher_drift(previous, cases)

    self.assertEqual(drift["changed"], [])
    self.assertEqual(drift["removed_transition_count"], 16)
    self.assertEqual(
      [entry["index"] for entry in drift["removed_transitions"]],
      list(range(10, 26)),
    )

  def test_a_reordered_list_is_not_a_removal(self):
    # The same 26 identities in another order are the same transitions: the
    # comparison is keyed by identity, so a reordering moves nothing.
    previous = {"cases": {"arm": _arm(list(self._PREVIOUS))}}
    cases = {"arm": _arm(list(reversed(self._PREVIOUS)))}

    drift = tensor_rematerialization_evidence.matcher_drift(previous, cases)

    self.assertEqual(drift["arms_compared"], ["arm"])
    self.assertEqual(drift["removed_transitions"], [])
    self.assertEqual(drift["added_transitions"], [])
    self.assertEqual(drift["changed"], [])

  def test_a_one_sided_missing_record_is_reported_rather_than_skipped(self):
    # A recorded arm of 26 transitions that the corrected record no longer
    # holds must not pass as unchanged, whether the arm is gone, its analysis
    # is gone, or its transitions field is gone. Each absence is named, so a
    # dropped arm, a dropped analysis, and a dropped field stay apart from one
    # another and from an arm that is still there and states no transitions at
    # all - which stays a compared arm with 26 removals.
    previous = {"cases": {"arm": _arm(list(self._PREVIOUS))}}
    variants = {
      "missing_arm": {},
      "missing_analysis": {"arm": {}},
      "missing_transitions": {"arm": {"analysis": {}}},
    }
    for reason, cases in variants.items():
      with self.subTest(reason=reason):
        drift = tensor_rematerialization_evidence.matcher_drift(previous, cases)

        self.assertNotIn("arm", drift["arms_compared"])
        self.assertEqual(drift["removed_transition_count"], 26)
        self.assertEqual(len(drift["removed_transitions"]), 26)
        self.assertEqual(
          [entry["index"] for entry in drift["removed_transitions"]],
          list(range(26)),
        )
        self.assertEqual(
          drift["unavailable_arms"],
          [
            {
              "case": "arm",
              "recorded": None,
              "corrected": reason,
              "removed_transition_count": 26,
              "added_transition_count": 0,
            }
          ],
        )

    empty = tensor_rematerialization_evidence.matcher_drift(previous, {"arm": _arm([])})
    self.assertEqual(empty["arms_compared"], ["arm"])
    self.assertEqual(empty["unavailable_arms"], [])
    self.assertEqual(empty["removed_transition_count"], 26)

  def test_an_arm_that_neither_record_analysed_is_named_as_skipped(self):
    # An arm both records hold but neither analyses has nothing comparable, so
    # the check skips it - and the skip names the arm and the reason each side
    # gave, so it cannot pass for an arm that was compared and found unchanged.
    drift = tensor_rematerialization_evidence.matcher_drift(
      {"cases": {"arm": {"analysis": {}}}},
      {"arm": {}},
    )

    self.assertEqual(drift["arms_compared"], [])
    self.assertEqual(drift["removed_transitions"], [])
    self.assertEqual(drift["added_transitions"], [])
    self.assertEqual(drift["unavailable_arms"], [])
    self.assertEqual(
      drift["skipped_arms"],
      [
        {
          "case": "arm",
          "recorded": "missing_transitions",
          "corrected": "missing_analysis",
        }
      ],
    )


if __name__ == "__main__":
  unittest.main()


class TensorCorrectionLedgerTests(unittest.TestCase):
  """The correction ledger compares like with like, or not at all.

  The ledger exists so a regeneration cannot silently rewrite a reading: every
  value this record changes is listed with the value the previous one held. That
  promise breaks if the two sides are different kinds of thing - a previous
  record that listed consumer *names* compared against a current count of
  consumers reports a correction for every opportunity it touches, even when the
  record is compared with itself.
  """

  def test_a_record_compared_with_itself_reports_no_corrections(self):
    # The whole point of the ledger: nothing changed, so nothing is listed.
    record = _ledger_record(count=1, consumers=None)
    corrections = tensor_rematerialization_evidence.superseded_readings(
      record, record["cases"], record["provenance"]
    )
    self.assertEqual(corrections, [])

  def test_a_legacy_consumer_list_is_compared_by_its_count(self):
    # A previous record that listed the consumers agrees with a current record
    # that counts them when the counts match; the legacy shape is read as what it
    # is, a list of that many consumers.
    previous = _ledger_record(count=None, consumers=["%_sir_x_offsets"])
    current = _ledger_record(count=1, consumers=None)
    corrections = tensor_rematerialization_evidence.superseded_readings(
      previous, current["cases"], current["provenance"]
    )
    self.assertEqual(corrections, [])

  def test_a_legacy_consumer_list_of_another_length_is_a_correction(self):
    # Two names against a count of one is a real change, reported with both
    # readings so a reader sees what moved.
    previous = _ledger_record(count=None, consumers=["%a", "%b"])
    current = _ledger_record(count=1, consumers=None)
    corrections = tensor_rematerialization_evidence.superseded_readings(
      previous, current["cases"], current["provenance"]
    )
    self.assertEqual(len(corrections), 1)
    # The comparison is by count, so the reading is a count; the names the
    # previous record listed are kept beside it rather than thrown away.
    self.assertEqual(corrections[0]["previous"], 2)
    self.assertEqual(corrections[0]["previous_consumers"], ["%a", "%b"])
    self.assertEqual(corrections[0]["corrected"], 1)

  def test_a_count_change_is_a_correction(self):
    previous = _ledger_record(count=2, consumers=None)
    current = _ledger_record(count=1, consumers=None)
    corrections = tensor_rematerialization_evidence.superseded_readings(
      previous, current["cases"], current["provenance"]
    )
    self.assertEqual(len(corrections), 1)
    self.assertEqual(corrections[0]["previous"], 2)
    self.assertEqual(corrections[0]["corrected"], 1)

  def test_an_opportunity_neither_side_counted_is_not_a_correction(self):
    # Neither side carries a reading: unknown on both sides is not a change.
    previous = _ledger_record(count=None, consumers=None)
    current = _ledger_record(count=None, consumers=None)
    corrections = tensor_rematerialization_evidence.superseded_readings(
      previous, current["cases"], current["provenance"]
    )
    self.assertEqual(corrections, [])
