"""Combined tensor test coverage: test_tensor_evaluation.py, test_tensor_campaign_driver.py."""

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# Consolidated from test_tensor_evaluation.py.
try:
  from test.lib import tensor_directed, tensor_evaluation
except ImportError:
  tensor_directed = None
  tensor_evaluation = None


class TensorEvaluationTests(unittest.TestCase):
  def test_model_selection_summary_counts_applicability_rescues_and_cost(self):
    def arm(
      name,
      *,
      status,
      property_passed,
      distinguished,
      proposal_count,
      objective_applicable,
      selection_s,
      reference_evaluations,
      reference_s,
      graph_id="graph-a",
    ):
      return {
        "arm": name,
        "graph_id": graph_id,
        "target_model": "swapped_reduction_groups",
        "P": "defined-stable-reference",
        "D": "fault-model:swapped_reduction_groups",
        "input_domain_id": "positive-integers-1-to-4",
        "proposal_mechanism_id": "mt19937-v1",
        "seed_schedule_id": "single-seed-test",
        "proposal_budget": 8,
        "status": status,
        "property_passed": property_passed,
        "distinguished": distinguished,
        "proposal_count": proposal_count,
        "objective_applicable": objective_applicable,
        "selection_cost": {
          "selection_s": selection_s,
          "property_reference_evaluations": reference_evaluations,
          "property_reference_s": reference_s,
          "model_reference_evaluations": reference_evaluations,
          "model_reference_s": reference_s,
        },
      }

    pairs = [
      {
        "graph_id": "graph-a",
        "source_signature": "sig-a",
        "arms": [
          arm(
            "ordinary",
            status="accepted",
            property_passed=True,
            distinguished=False,
            proposal_count=1,
            objective_applicable=True,
            selection_s=0.2,
            reference_evaluations=1,
            reference_s=0.03,
          ),
          arm(
            "directed",
            status="accepted",
            property_passed=True,
            distinguished=True,
            proposal_count=3,
            objective_applicable=True,
            selection_s=0.7,
            reference_evaluations=3,
            reference_s=0.09,
          ),
        ],
      },
      {
        "graph_id": "graph-b",
        "source_signature": "sig-a",
        "arms": [
          arm(
            "ordinary",
            status="no_applicable_model",
            property_passed=False,
            distinguished=False,
            proposal_count=0,
            objective_applicable=False,
            selection_s=0,
            reference_evaluations=0,
            reference_s=0,
            graph_id="graph-b",
          ),
          arm(
            "directed",
            status="no_applicable_model",
            property_passed=False,
            distinguished=False,
            proposal_count=0,
            objective_applicable=False,
            selection_s=0,
            reference_evaluations=0,
            reference_s=0,
            graph_id="graph-b",
          ),
        ],
      },
      {
        "graph_id": "graph-c",
        "source_signature": "sig-b",
        "arms": [
          arm(
            "ordinary",
            status="accepted",
            property_passed=True,
            distinguished=False,
            proposal_count=1,
            objective_applicable=True,
            selection_s=0.2,
            reference_evaluations=1,
            reference_s=0.03,
            graph_id="graph-c",
          ),
          arm(
            "directed",
            status="proposal_exhausted",
            property_passed=True,
            distinguished=False,
            proposal_count=8,
            objective_applicable=True,
            selection_s=1.4,
            reference_evaluations=8,
            reference_s=0.24,
            graph_id="graph-c",
          ),
        ],
      },
    ]

    summary = tensor_evaluation.model_specific_selection_summary(pairs)
    model = summary["models"]["swapped_reduction_groups"]

    self.assertEqual(model["graphs"], 3)
    self.assertEqual(model["unique_structures"], 2)
    self.assertEqual(model["no_applicable_model"], 1)
    self.assertEqual(model["applicable_graphs"], 2)
    self.assertEqual(model["ordinary_not_distinguished"], 2)
    self.assertEqual(model["directed_proposal_exhausted"], 1)
    self.assertEqual(model["rescues"], 1)
    self.assertEqual(model["rescue_cost"]["directed_property_reference_evaluations"], 3)
    self.assertEqual(model["rescue_cost"]["directed_model_reference_evaluations"], 3)
    self.assertEqual(model["rescue_cost"]["ordinary_property_reference_evaluations"], 1)
    self.assertAlmostEqual(model["rescue_cost"]["directed_selection_s"], 0.7)
    self.assertAlmostEqual(model["rescue_cost_per_rescue"]["directed_selection_s"], 0.7)
    changed_domain = json.loads(json.dumps(pairs))
    changed_domain[0]["arms"][1]["input_domain_id"] = "different-domain"
    with self.assertRaisesRegex(ValueError, "input_domain_id"):
      tensor_evaluation.model_specific_selection_summary(changed_domain)

  def test_campaign_summary_reports_model_specific_selection(self):
    def record(arm, status, distinguished, proposals):
      return {
        "arm": f"tensor-v2-compositional-{arm}",
        "seed": 14,
        "source_signature": "sig-14",
        "target_model": "swapped_reduction_groups",
        "P": "defined-stable-reference",
        "D": "fault-model:swapped_reduction_groups",
        "input_domain_id": "positive-integers-1-to-4",
        "proposal_mechanism_id": "mt19937-v1",
        "seed_schedule_id": "single-seed-14",
        "proposal_budget": 8,
        "selection_status": status,
        "property_passed": True,
        "objective_applicable": True,
        "distinguished": distinguished,
        "proposal_count": proposals,
        "selection_cost": {
          "selection_s": 0.1,
          "property_reference_evaluations": proposals,
          "property_reference_s": 0.01 * proposals,
          "model_reference_evaluations": proposals,
          "model_reference_s": 0.02 * proposals,
        },
      }

    ledger = {
      "protocol": "fixed-count",
      "target_count": 1,
      "records": [
        {"arm": "tensor-v2-compositional-ordinary", "completed_target_runs": 1},
        {"arm": "tensor-v2-compositional-directed", "completed_target_runs": 1},
      ],
      "cases": [
        record("ordinary", "accepted", False, 1),
        record("directed", "accepted", True, 3),
      ],
    }

    summary = tensor_evaluation.summarize_ledger(ledger)

    model = summary["model_selection"]["models"]["swapped_reduction_groups"]
    self.assertEqual(model["rescues"], 1)
    self.assertEqual(model["unique_structures"], 1)

  def setUp(self):
    self.assertIsNotNone(tensor_evaluation, "missing tensor evaluation support")

  def test_fixed_count_shortfall_withholds_equal_count_comparison(self):
    summary = tensor_evaluation.fixed_count_summary(
      [
        {"arm": "existing", "completed_target_runs": 10},
        {"arm": "ordinary", "completed_target_runs": 10},
        {"arm": "directed", "completed_target_runs": 8},
      ],
      target_count=10,
    )

    self.assertEqual(summary["status"], "count_shortfall")
    self.assertFalse(summary["equal_count_comparison_valid"])
    self.assertEqual(summary["arms"]["directed"]["shortfall"], 2)

  def test_fixed_time_keeps_unequal_completed_counts(self):
    summary = tensor_evaluation.fixed_time_summary(
      [
        {
          "arm": "ordinary",
          "budget_s": 60,
          "elapsed_s": 60,
          "completed_target_runs": 8,
          "generation_s": 1,
          "selection_s": 1,
          "input_materialization_s": 1,
          "reference_validation_s": 1,
          "compilation_s": 1,
          "gpu_execution_s": 1,
        },
        {
          "arm": "directed",
          "budget_s": 60,
          "elapsed_s": 58,
          "completed_target_runs": 3,
          "generation_s": 1,
          "selection_s": 1,
          "input_materialization_s": 1,
          "reference_validation_s": 1,
          "compilation_s": 1,
          "gpu_execution_s": 1,
        },
      ],
      budget_s=60,
    )

    self.assertEqual(summary["status"], "complete")
    self.assertEqual(summary["arms"]["ordinary"]["completed_target_runs"], 8)
    self.assertEqual(summary["arms"]["directed"]["completed_target_runs"], 3)

  def test_cache_must_be_empty_and_isolated_by_arm_trial(self):
    records = [
      {
        "arm": "ordinary",
        "trial": "a",
        "cache_id": "cache-a",
        "cache_empty_at_start": True,
      },
      {
        "arm": "directed",
        "trial": "a",
        "cache_id": "cache-b",
        "cache_empty_at_start": True,
      },
    ]

    self.assertTrue(tensor_evaluation.validate_cache_isolation(records)["valid"])
    invalid = records + [
      {
        "arm": "directed",
        "trial": "b",
        "cache_id": "cache-a",
        "cache_empty_at_start": False,
      },
    ]
    self.assertFalse(tensor_evaluation.validate_cache_isolation(invalid)["valid"])

  def test_fixed_time_cost_includes_generation_and_reference(self):
    record = {
      "generation_s": 1,
      "selection_s": 2,
      "input_materialization_s": 3,
      "reference_validation_s": 4,
      "compilation_s": 5,
      "gpu_execution_s": 6,
    }

    self.assertEqual(tensor_evaluation.end_to_end_seconds(record), 21)

  def test_archived_v1_requires_measured_preparation_cost(self):
    self.assertFalse(tensor_evaluation.archived_v1_end_to_end_eligible({}))
    self.assertTrue(
      tensor_evaluation.archived_v1_end_to_end_eligible(
        {"generation_s": 2, "reference_validation_s": 3}
      )
    )

  def test_rediscovery_requires_fresh_matched_affected_fixed_and_triage(self):
    attempt = {
      "fresh_case": True,
      "affected_status": "numeric_mismatch",
      "corrected_status": "pass",
      "same_source_sha256": True,
      "same_inputs_sha256": True,
      "same_settings": True,
      "triage_same_root_cause": True,
    }

    self.assertTrue(tensor_evaluation.qualifies_rediscovery(attempt))
    self.assertFalse(
      tensor_evaluation.qualifies_rediscovery({**attempt, "fresh_case": False})
    )

  def test_fixed_time_rejects_uncharged_preparation_stages(self):
    with self.assertRaisesRegex(ValueError, "generation_s"):
      tensor_evaluation.fixed_time_summary(
        [
          {
            "arm": "ordinary",
            "budget_s": 60,
            "elapsed_s": 60,
            "completed_target_runs": 3,
          }
        ],
        budget_s=60,
      )

  def test_selection_time_is_required_for_fixed_time_accounting(self):
    record = {
      "generation_s": 1,
      "selection_s": 2,
      "input_materialization_s": 3,
      "reference_validation_s": 4,
      "compilation_s": 5,
      "gpu_execution_s": 6,
    }

    self.assertEqual(tensor_evaluation.end_to_end_seconds(record), 21)

  def test_fixed_time_summary_accepts_the_declared_selection_stage(self):
    stages = {
      "generation_s": 1,
      "selection_s": 1,
      "input_materialization_s": 1,
      "reference_validation_s": 1,
      "compilation_s": 1,
      "gpu_execution_s": 1,
    }
    record = {
      "arm": "ordinary",
      "budget_s": 10,
      "elapsed_s": 10,
      "completed_target_runs": 2,
      **stages,
    }

    report = tensor_evaluation.fixed_time_summary([record], budget_s=10)

    self.assertEqual(report["status"], "complete")

  def test_legacy_solving_stage_is_not_silently_counted_as_selection(self):
    record = {
      "generation_s": 0,
      "solving_s": 1,
      "input_materialization_s": 0,
      "reference_validation_s": 0,
      "compilation_s": 0,
      "gpu_execution_s": 0,
    }

    with self.assertRaisesRegex(ValueError, "selection_s"):
      tensor_evaluation.end_to_end_seconds(record)

  def test_selection_and_solving_cannot_both_be_charged(self):
    record = {
      "generation_s": 0,
      "selection_s": 1,
      "solving_s": 1,
      "input_materialization_s": 0,
      "reference_validation_s": 0,
      "compilation_s": 0,
      "gpu_execution_s": 0,
    }

    with self.assertRaisesRegex(ValueError, "solving_s"):
      tensor_evaluation.end_to_end_seconds(record)

  def test_selection_stage_is_included_when_checking_wall_clock_budget(self):
    record = {
      "arm": "ordinary",
      "budget_s": 10,
      "elapsed_s": 5,
      "completed_target_runs": 1,
      "generation_s": 1,
      "selection_s": 2,
      "input_materialization_s": 1,
      "reference_validation_s": 1,
      "compilation_s": 1,
      "gpu_execution_s": 1,
    }

    with self.assertRaisesRegex(ValueError, "stage costs exceed"):
      tensor_evaluation.fixed_time_summary([record], budget_s=10)

  def test_authored_holdout_replay_does_not_claim_generator_generalization(self):
    report = tensor_evaluation.campaign_summary(
      {
        "cases": [],
        "authored_holdout_replay": {"status": "pass"},
        "generator_generalization_status": "exposed",
      }
    )

    self.assertEqual(report["authored_holdout_replay_status"], "pass")
    self.assertNotEqual(report["generator_generalization_status"], "exposed")

  def test_case_failures_are_counted_separately_from_root_causes(self):
    report = tensor_evaluation.campaign_summary(
      {
        "cases": [
          {"status": "numeric_mismatch", "root_cause_id": "bug-a", "triaged": True},
          {"status": "numeric_mismatch", "root_cause_id": "bug-a", "triaged": True},
          {"status": "exact_mismatch", "root_cause_id": "candidate", "triaged": False},
        ],
      }
    )

    self.assertEqual(report["failing_case_count"], 3)
    self.assertEqual(report["investigated_defect_count"], 1)

  def test_generalization_requires_same_fresh_case_on_both_builds(self):
    report = tensor_evaluation.campaign_summary(
      {
        "holdout_status": "verified",
        "authored_holdout_replay": {"status": "pass"},
        "holdout_attempts": [
          {
            "fresh_generated_case": True,
            "affected_status": "numeric_mismatch",
            "corrected_status": "pass",
            "same_source_sha256": False,
            "same_inputs_sha256": True,
            "same_settings": True,
            "holdout_root_cause": True,
            "triaged": True,
          }
        ],
      }
    )

    self.assertEqual(report["holdout_status"], "verified")
    self.assertEqual(
      report["generator_generalization_status"], "not_exposed_within_budget"
    )

  def test_rediscovery_manifest_requires_frozen_distinct_builds(self):
    with self.assertRaisesRegex(ValueError, "distinct"):
      tensor_evaluation.summarize_ledger(
        {
          "protocol": "rediscovery",
          "frozen_configuration": {
            "generator_version": "frozen",
            "seed_schedule": [1, 2],
            "affected_revision": "same",
            "corrected_revision": "same",
            "candidate_attempt_cap": 4,
            "wall_clock_budget_s": 60,
          },
          "attempts": [],
        }
      )

  def test_matched_graphs_require_identical_execution_configuration(self):
    ordinary = {
      "graph_id": "graph-a",
      "input_domain_id": "positive",
      "shape": [4, 4],
      "layout_id": "row-major",
      "num_warps": 1,
      "compiler_revision": "triton-a",
      "proposal_mechanism_id": "pcg-v1",
      "seed_schedule_id": "seed-1",
      "comparison_policy_id": "exact",
      "proposal_budget": 100,
      "property_passed": True,
    }
    directed = {**ordinary, "property_passed": True, "distinguished": True}
    paired = tensor_evaluation.matched_graph_pairs([ordinary], [directed])
    changed = tensor_evaluation.matched_graph_pairs(
      [ordinary], [{**directed, "num_warps": 4}]
    )
    different_objective = tensor_evaluation.matched_graph_pairs(
      [
        {
          **ordinary,
          "D": "fault-model:wrong_axis",
          "target_model": "wrong_axis",
        }
      ],
      [
        {
          **directed,
          "D": "fault-model:wrong_stride",
          "target_model": "wrong_stride",
        }
      ],
    )

    self.assertEqual(len(paired["pairs"]), 1)
    self.assertEqual(changed["pairs"], [])
    self.assertEqual(changed["unmatched"]["ordinary"], ["graph-a"])
    self.assertEqual(different_objective["pairs"], [])

  def test_directed_shortfall_does_not_replace_graph(self):
    ordinary = {
      "graph_id": "graph-a",
      "input_domain_id": "positive",
      "shape": [4, 4],
      "layout_id": "row-major",
      "num_warps": 1,
      "compiler_revision": "triton-a",
      "proposal_mechanism_id": "pcg-v1",
      "seed_schedule_id": "seed-1",
      "comparison_policy_id": "exact",
      "proposal_budget": 100,
      "property_passed": True,
    }
    directed = {**ordinary, "property_passed": False}

    paired = tensor_evaluation.matched_graph_pairs([ordinary], [directed])

    self.assertEqual(paired["pairs"], [])
    self.assertEqual(paired["unmatched"]["directed"], ["graph-a"])

  def test_pair_preparation_uses_same_graph_seed_and_domain(self):
    source = (
      "fun @kernel(%x: tensor [2, 2] f32, %y: tensor [2, 2] f32) : "
      "tensor [2, 2] f32 { ret %x; }\n"
    )
    manifest = {
      "profile": "tensor-v2",
      "coverage": {"primary_stratum": "multi-input-composition"},
      "comparison": {"mode": "numeric", "atol": [1, 1000], "rtol": [1, 1000]},
      "inputs": [],
      "output": {"shape": [2, 2]},
      "launch": {"num_warps": 2},
    }
    with tempfile.TemporaryDirectory() as directory:
      case = Path(directory) / "case"
      case.mkdir()
      manifest["files"] = {
        name: {"bytes": 0, "sha256": "0" * 64}
        for name in ("program.sir", "inputs.bin", "expected.bin")
      }
      (case / "case.json").write_text(json.dumps(manifest))
      (case / "program.sir").write_text(source)
      (case / "inputs.bin").write_bytes(b"")
      (case / "expected.bin").write_bytes(b"")
      stale = case / "runs" / "compiled"
      stale.mkdir(parents=True)
      (stale / "case.json").write_text("stale run artifact")

      def select(path, *, arm, seed, proposal_limit, symiri, target_model=None):
        return {
          "status": "accepted" if arm == "ordinary" else "proposal_exhausted",
          "proposal_count": 4,
          "property_passed": True,
          "distinguished": False,
        }

      with mock.patch.object(tensor_directed, "prepare_case_arm", select):
        report = tensor_evaluation.prepare_matched_pair(
          case,
          Path(directory) / "pair",
          seed=9,
          proposal_budget=20,
          compiler_revision="triton-pin",
        )
      self.assertFalse((Path(directory) / "pair" / "ordinary" / "runs").exists())

    self.assertEqual(report["arms"][0]["graph_id"], report["arms"][1]["graph_id"])
    self.assertEqual(
      report["arms"][0]["seed_schedule_id"], report["arms"][1]["seed_schedule_id"]
    )
    self.assertTrue(report["arms"][1]["property_passed"])
    self.assertFalse(report["arms"][1]["distinguished"])
    self.assertEqual(report["pairing"]["pairs"], [])

  def test_pair_preparation_preserves_the_model_specific_objective(self):
    source = (
      "fun @kernel(%x: tensor [2, 2] i32, %y: tensor [2, 2] i32) : "
      "tensor [2, 2] i32 { ret %x; }\n"
    )
    manifest = {
      "profile": "tensor-v2",
      "coverage": {"primary_stratum": "multi-input-composition"},
      "comparison": {"mode": "exact", "atol": [0, 1], "rtol": [0, 1]},
      "inputs": [],
      "output": {"shape": [2, 2]},
      "launch": {"num_warps": 2},
    }
    target_models = []
    with tempfile.TemporaryDirectory() as directory:
      case = Path(directory) / "case"
      case.mkdir()
      manifest["files"] = {
        name: {"bytes": 0, "sha256": "0" * 64}
        for name in ("program.sir", "inputs.bin", "expected.bin")
      }
      (case / "case.json").write_text(json.dumps(manifest))
      (case / "program.sir").write_text(source)
      (case / "inputs.bin").write_bytes(b"")
      (case / "expected.bin").write_bytes(b"")

      def select(path, *, arm, seed, proposal_limit, symiri, target_model=None):
        target_models.append(target_model)
        return {
          "status": "accepted",
          "proposal_count": 1,
          "property_passed": True,
          "distinguished": True,
        }

      with mock.patch.object(tensor_directed, "prepare_case_arm", select):
        report = tensor_evaluation.prepare_matched_pair(
          case,
          Path(directory) / "pair",
          seed=9,
          proposal_budget=20,
          compiler_revision="triton-pin",
          target_model="swapped_reduction_groups",
        )

    self.assertEqual(target_models, ["swapped_reduction_groups"] * 2)
    self.assertEqual(
      [record["target_model"] for record in report["arms"]],
      ["swapped_reduction_groups"] * 2,
    )
    self.assertEqual(
      [record["D"] for record in report["arms"]],
      ["fault-model:swapped_reduction_groups"] * 2,
    )
    summary = tensor_evaluation.model_specific_selection_summary([report])
    self.assertEqual(summary["models"]["swapped_reduction_groups"]["graphs"], 1)

  def test_pair_execution_runs_both_accepted_arms_once(self):
    with tempfile.TemporaryDirectory() as directory:
      pair = make_pair(Path(directory) / "pair")
      calls = []
      report = tensor_evaluation.execute_matched_pair(
        pair,
        mode="compiled",
        case_runner=lambda path, mode: calls.append((Path(path).name, mode))
        or {"status": "pass", "campaign_valid": True},
      )

    self.assertEqual(len(calls), 2)
    self.assertEqual(report["completed_target_runs"], 2)
    self.assertEqual([arm["run_status"] for arm in report["arms"]], ["pass", "pass"])

  def test_pair_execution_keeps_directed_shortfall_without_substitution(self):
    with tempfile.TemporaryDirectory() as directory:
      pair = make_pair(Path(directory) / "pair", directed_status="proposal_exhausted")
      calls = []
      report = tensor_evaluation.execute_matched_pair(
        pair,
        mode="compiled",
        case_runner=lambda path, _mode: calls.append(Path(path).name)
        or {"status": "pass", "campaign_valid": True},
      )

    self.assertEqual(calls, ["ordinary"])
    self.assertEqual(report["arms"][1]["run_status"], "directed_shortfall")
    self.assertEqual(report["completed_target_runs"], 1)

  def test_pair_execution_records_runner_cache_identities(self):
    with tempfile.TemporaryDirectory() as directory:
      pair = make_pair(Path(directory) / "pair")

      def runner(case_path, _mode):
        cache = {"cache_id": Path(case_path).name, "empty_at_start": True}
        run_path = Path(case_path) / "runs" / "compiled" / "run.json"
        run_path.parent.mkdir(parents=True)
        run_path.write_text(json.dumps({"compiler_cache": cache}))
        return {"status": "pass", "campaign_valid": True}

      report = tensor_evaluation.execute_matched_pair(
        pair, mode="compiled", case_runner=runner
      )

    self.assertEqual(
      {item["cache_id"] for item in report["cache_records"]},
      {"ordinary", "directed"},
    )

  def test_pair_execution_does_not_count_compile_rejection_as_completed(self):
    with tempfile.TemporaryDirectory() as directory:
      pair = make_pair(Path(directory) / "pair")
      report = tensor_evaluation.execute_matched_pair(
        pair,
        mode="compiled",
        case_runner=lambda _path, _mode: {
          "status": "compile_failure",
          "campaign_valid": True,
        },
      )

    self.assertEqual(report["completed_target_runs"], 0)
    self.assertEqual([arm["completed_target_runs"] for arm in report["arms"]], [0, 0])

  def test_pair_execution_keeps_arm_results_in_pair_report(self):
    with tempfile.TemporaryDirectory() as directory:
      pair = make_pair(Path(directory) / "pair")
      report = tensor_evaluation.execute_matched_pair(
        pair,
        mode="compiled",
        case_runner=lambda path, _mode: {
          "status": "numeric_mismatch" if Path(path).name == "directed" else "pass",
          "campaign_valid": True,
        },
      )
      on_disk = json.loads((pair / "pair.json").read_text())

    self.assertEqual(report["arms"][1]["run_status"], "numeric_mismatch")
    self.assertEqual(
      on_disk["runs"]["compiled"]["arms"][1]["result"]["status"], "numeric_mismatch"
    )

  def test_pair_execution_resume_does_not_repeat_completed_arms(self):
    with tempfile.TemporaryDirectory() as directory:
      pair = make_pair(Path(directory) / "pair")
      calls = []

      def runner(path, _mode):
        calls.append(Path(path).name)
        return {"status": "pass", "campaign_valid": True}

      first = tensor_evaluation.execute_matched_pair(
        pair, mode="interpret", case_runner=runner
      )
      resumed = tensor_evaluation.execute_matched_pair(
        pair, mode="interpret", case_runner=runner
      )

    self.assertEqual(calls, ["ordinary", "directed"])
    self.assertEqual(first["completed_target_runs"], 2)
    self.assertEqual(resumed["completed_target_runs"], 2)

  def test_pair_execution_resume_retries_only_interrupted_arm(self):
    with tempfile.TemporaryDirectory() as directory:
      pair = make_pair(Path(directory) / "pair")
      calls = []

      def interrupted_runner(path, _mode):
        name = Path(path).name
        calls.append(name)
        if name == "directed":
          raise RuntimeError("temporary worker interruption")
        return {"status": "pass", "campaign_valid": True}

      tensor_evaluation.execute_matched_pair(
        pair, mode="interpret", case_runner=interrupted_runner
      )
      retry = tensor_evaluation.execute_matched_pair(
        pair,
        mode="interpret",
        case_runner=lambda path, _mode: calls.append(Path(path).name)
        or {"status": "pass", "campaign_valid": True},
      )

    self.assertEqual(calls, ["ordinary", "directed", "directed"])
    self.assertEqual(retry["completed_target_runs"], 2)

  def test_pair_execution_persists_first_arm_before_starting_second(self):
    with tempfile.TemporaryDirectory() as directory:
      pair = make_pair(Path(directory) / "pair")
      observed = []

      def runner(path, _mode):
        if Path(path).name == "directed":
          saved = json.loads((pair / "pair.json").read_text())
          observed.append(saved["runs"]["interpret"]["arms"][0]["run_status"])
        return {"status": "pass", "campaign_valid": True}

      tensor_evaluation.execute_matched_pair(pair, mode="interpret", case_runner=runner)

    self.assertEqual(observed, ["pass"])

  def test_pair_execution_reruns_arm_left_running_after_process_interruption(self):
    with tempfile.TemporaryDirectory() as directory:
      pair = make_pair(Path(directory) / "pair")
      calls = []

      def interrupted_runner(path, _mode):
        name = Path(path).name
        calls.append(name)
        if name == "directed":
          raise KeyboardInterrupt()
        return {"status": "pass", "campaign_valid": True}

      with self.assertRaises(KeyboardInterrupt):
        tensor_evaluation.execute_matched_pair(
          pair, mode="interpret", case_runner=interrupted_runner
        )
      saved = json.loads((pair / "pair.json").read_text())
      self.assertEqual(saved["runs"]["interpret"]["arms"][1]["run_status"], "running")
      retry = tensor_evaluation.execute_matched_pair(
        pair,
        mode="interpret",
        case_runner=lambda path, _mode: calls.append(Path(path).name)
        or {"status": "pass", "campaign_valid": True},
      )

    self.assertEqual(calls, ["ordinary", "directed", "directed"])
    self.assertEqual(retry["completed_target_runs"], 2)

  def test_pair_execution_does_not_resume_completed_shortfall_as_a_new_case(self):
    with tempfile.TemporaryDirectory() as directory:
      pair = make_pair(Path(directory) / "pair", directed_status="proposal_exhausted")
      calls = []

      def runner(path, _mode):
        calls.append(Path(path).name)
        return {"status": "pass", "campaign_valid": True}

      tensor_evaluation.execute_matched_pair(pair, mode="interpret", case_runner=runner)
      tensor_evaluation.execute_matched_pair(pair, mode="interpret", case_runner=runner)

    self.assertEqual(calls, ["ordinary"])

  def test_pair_execution_passes_selected_gpu_to_default_case_runner(self):
    from test.lib import run_triton_tests

    with tempfile.TemporaryDirectory() as directory:
      pair = make_pair(Path(directory) / "pair")
      calls = []

      def fake_run(case_path, mode, *, device_id):
        calls.append((Path(case_path).name, mode, device_id))
        return {"status": "pass", "campaign_valid": True}

      with mock.patch.object(run_triton_tests, "run_case", side_effect=fake_run):
        report = tensor_evaluation.execute_matched_pair(
          pair, mode="compiled", device_id=3
        )

    self.assertEqual(report["completed_target_runs"], 2)
    self.assertEqual(
      calls,
      [
        ("ordinary", "compiled", 3),
        ("directed", "compiled", 3),
      ],
    )

  def test_pair_execution_records_selected_gpu_in_pair_ledger(self):
    with tempfile.TemporaryDirectory() as directory:
      pair = make_pair(Path(directory) / "pair")
      tensor_evaluation.execute_matched_pair(
        pair,
        mode="compiled",
        device_id=2,
        case_runner=lambda _path, _mode: {"status": "compile_failure"},
      )
      saved = json.loads((pair / "pair.json").read_text())

    self.assertEqual(saved["runs"]["compiled"]["device_id"], 2)

  def test_pair_execution_refuses_resume_on_a_different_gpu(self):
    with tempfile.TemporaryDirectory() as directory:
      pair = make_pair(Path(directory) / "pair")
      tensor_evaluation.execute_matched_pair(
        pair,
        mode="compiled",
        device_id=2,
        case_runner=lambda _path, _mode: {"status": "compile_failure"},
      )

      with self.assertRaisesRegex(ValueError, "different selected device"):
        tensor_evaluation.execute_matched_pair(
          pair,
          mode="compiled",
          device_id=3,
          case_runner=lambda _path, _mode: {"status": "pass"},
        )

  def test_pair_execution_rejects_negative_gpu_index(self):
    with tempfile.TemporaryDirectory() as directory:
      pair = make_pair(Path(directory) / "pair")
      with self.assertRaisesRegex(ValueError, "nonnegative"):
        tensor_evaluation.execute_matched_pair(
          pair,
          mode="compiled",
          device_id=-1,
          case_runner=lambda _path, _mode: {"status": "pass"},
        )

  def test_interpreter_pair_records_no_selected_gpu(self):
    with tempfile.TemporaryDirectory() as directory:
      pair = make_pair(Path(directory) / "pair")
      tensor_evaluation.execute_matched_pair(
        pair,
        mode="interpret",
        case_runner=lambda _path, _mode: {"status": "pass"},
      )
      saved = json.loads((pair / "pair.json").read_text())

    self.assertIsNone(saved["runs"]["interpret"]["device_id"])


def make_pair(root, *, directed_status="accepted"):
  root = Path(root)
  root.mkdir()
  arms = []
  for name in ("ordinary", "directed"):
    (root / name).mkdir()
    arms.append(
      {
        "arm": name,
        "case_directory": name,
        "status": "accepted" if name == "ordinary" else directed_status,
        "property_passed": True,
        "distinguished": name == "directed" and directed_status == "accepted",
        "completed_target_runs": 0,
      }
    )
  (root / "pair.json").write_text(json.dumps({"arms": arms}))
  return root


if __name__ == "__main__":
  unittest.main()

# Consolidated from test_tensor_campaign_driver.py.
DRIVER_PATH = (
  Path(__file__).resolve().parents[1]
  / "tensor"
  / "campaigns"
  / "run_matched_tensor_evaluation.py"
)
SPEC = importlib.util.spec_from_file_location("tensor_campaign_driver", DRIVER_PATH)
campaign_driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign_driver)


class TensorCampaignDriverTests(unittest.TestCase):
  def test_campaign_rejects_different_model_objectives_for_paired_arms(self):
    config = {
      "fixed_count": {
        "arms": {
          "ordinary": {
            "profile": "tensor-v2",
            "selection": {
              "arm": "ordinary",
              "target_model": "wrong_axis",
              "D": "fault-model:wrong_axis",
            },
          },
          "directed": {
            "profile": "tensor-v2",
            "selection": {
              "arm": "directed",
              "target_model": "wrong_stride",
              "D": "fault-model:wrong_stride",
            },
          },
        }
      }
    }

    with self.assertRaisesRegex(ValueError, "same target model"):
      campaign_driver._validate_model_objective_pairs(config)

  def test_selection_status_preserves_model_applicability_and_budget_outcomes(self):
    self.assertEqual(
      campaign_driver._selection_outcome_status(
        "directed", {"status": "no_applicable_model", "property_passed": False}
      ),
      "no_applicable_model",
    )
    self.assertEqual(
      campaign_driver._selection_outcome_status(
        "directed", {"status": "proposal_exhausted", "property_passed": True}
      ),
      "directed_shortfall",
    )
    self.assertEqual(
      campaign_driver._selection_outcome_status(
        "directed", {"status": "model_invalid_for_proposals", "property_passed": True}
      ),
      "model_invalid_for_proposals",
    )

  def test_fixed_time_arm_config_uses_manifest_budget(self):
    protocol = {"wall_clock_budget_s_per_arm": 3600, "device_pool": [0]}

    arm_config = campaign_driver._fixed_time_arm_config(protocol)

    self.assertEqual(arm_config["wall_clock_budget_s"], 3600.0)

  def test_fixed_time_arm_config_preserves_other_settings(self):
    protocol = {
      "wall_clock_budget_s_per_arm": 120,
      "device_pool": [0],
      "max_concurrent_runs_per_device": 1,
    }

    arm_config = campaign_driver._fixed_time_arm_config(protocol)

    self.assertEqual(arm_config["device_pool"], [0])
    self.assertEqual(arm_config["max_concurrent_runs_per_device"], 1)
    self.assertNotIn("wall_clock_budget_s", protocol)

  def test_fixed_time_arm_config_rejects_zero_budget(self):
    with self.assertRaisesRegex(ValueError, "positive"):
      campaign_driver._fixed_time_arm_config({"wall_clock_budget_s_per_arm": 0})

  def test_fixed_time_arm_config_rejects_negative_budget(self):
    with self.assertRaisesRegex(ValueError, "positive"):
      campaign_driver._fixed_time_arm_config({"wall_clock_budget_s_per_arm": -1})

  def test_fixed_time_arm_config_rejects_boolean_budget(self):
    with self.assertRaisesRegex(ValueError, "finite positive number"):
      campaign_driver._fixed_time_arm_config({"wall_clock_budget_s_per_arm": True})

  def test_fixed_time_arm_config_rejects_missing_budget(self):
    with self.assertRaisesRegex(ValueError, "wall_clock_budget_s_per_arm"):
      campaign_driver._fixed_time_arm_config({"device_pool": [0]})

  def test_fixed_time_arm_config_rejects_nonfinite_budget(self):
    for budget in (math.nan, math.inf, -math.inf):
      with self.subTest(budget=budget):
        with self.assertRaisesRegex(ValueError, "finite positive number"):
          campaign_driver._fixed_time_arm_config(
            {"wall_clock_budget_s_per_arm": budget}
          )

  def test_run_fixed_time_passes_normalized_budget_to_arm_runner(self):
    seen_configs = []

    def run_arm(config, arm_name, *_args):
      seen_configs.append(config)
      return {
        "record": {"arm": arm_name, "budget_s": config["wall_clock_budget_s"]},
        "cases": [],
        "cache_records": [],
        "failure": None,
      }

    config = {
      "fixed_time": {
        "wall_clock_budget_s_per_arm": 60,
        "device_pool": [0],
        "arms": {"ordinary": {}},
      }
    }
    tensor_evaluation = SimpleNamespace(summarize_ledger=lambda _ledger: {})

    with tempfile.TemporaryDirectory() as directory:
      with (
        mock.patch.object(campaign_driver, "_fixed_time_arm", side_effect=run_arm),
        mock.patch.object(campaign_driver, "_json_write"),
      ):
        campaign_driver.run_fixed_time(
          config,
          directory,
          hooks=None,
          tensor_case=None,
          tensor_coverage=None,
          tensor_directed=None,
          tensor_evaluation=tensor_evaluation,
        )

    self.assertEqual(seen_configs[0]["wall_clock_budget_s"], 60.0)


if __name__ == "__main__":
  unittest.main()
