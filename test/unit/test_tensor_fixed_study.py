"""Fixed-domain study accounting for selection and compiler evidence."""

import json
import struct
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from test.lib import tensor_evaluation
from test.tensor.gluon import run_selection_study


def _trial(policy, *, route="exact", trial="seed-1", status="admitted"):
  return {
    "route": route,
    "semantic_contract": f"{route}-contract-v1",
    "property_id": f"{route}-output-property",
    "selection_policy": policy,
    "trial": trial,
    "program_id": "composed-loop-v1",
    "input_domain_id": "nonnegative-integer-f32-0-through-8",
    "compiler_settings_id": "triton-3.8.0-v100-settings-v1",
    "budget_s": 10,
    "elapsed_s": 2,
    "admission_status": status,
    "execution_status": "pass" if status == "admitted" else "not_run",
    "cost_s": {
      "generation_s": 0.1,
      "selection_s": 0.3 if policy == "random" else 0.8,
      "input_materialization_s": 0.0,
      "reference_validation_s": 0.3,
      "compilation_s": 0.5,
      "gpu_execution_s": 0.1,
    },
    "calibration_controls": [
      {
        "control_id": "known-routing-mutant",
        "expected_detection": True,
        "observed_detection": True,
      },
      {
        "control_id": "equivalent-rewrite",
        "expected_detection": False,
        "observed_detection": False,
      },
    ],
    "compiler_observation": {
      "layout_conversion_count": 1,
      "layout_conversions": [
        {
          "source_operation": "compute_to_reduction",
          "layout_changed": True,
          "stage": "ttgir",
        }
      ],
      "effective_conversion_operations": ["compute_to_reduction"],
      "loop_status": "retained",
      "stages": {
        "ttgir": {
          "reused_values": [{"value": "%v", "use_count": 2}],
          "duplicate_expressions": [
            {"expression": "tt.add %a, %b", "values": ["%r", "%s"], "count": 2}
          ],
          "loop_construct_count": 1,
        },
        "ptx": {
          "warp_shuffle_count": 2,
          "shared_memory_transfer_count": 0,
        },
      },
    },
  }


def _ledger():
  return {
    "schema": "refractir.tensor-fixed-contract-study.v1",
    "fixed": {
      "program_id": "composed-loop-v1",
      "input_domain_id": "nonnegative-integer-f32-0-through-8",
      "compiler_settings_id": "triton-3.8.0-v100-settings-v1",
      "budget_s": 10,
    },
    "trials": [_trial("random"), _trial("solver")],
  }


class TensorFixedStudyTests(unittest.TestCase):
  def test_summarizes_admission_and_useful_executions_per_budget(self):
    ledger = _ledger()
    ledger["trials"][0]["admission_status"] = "refused"
    ledger["trials"][0]["execution_status"] = "not_run"

    report = tensor_evaluation.summarize_fixed_contract_study(ledger)
    route = report["routes"]["exact:exact-contract-v1:exact-output-property"]

    self.assertEqual(route["policies"]["random"]["admitted"], 0)
    self.assertEqual(route["policies"]["random"]["refused"], 1)
    self.assertEqual(route["policies"]["solver"]["useful_executions"], 1)
    self.assertEqual(route["policies"]["solver"]["allocated_budget_s"], 10)

  def test_rejects_stage_costs_or_elapsed_time_over_the_fixed_budget(self):
    ledger = _ledger()
    ledger["trials"][0]["elapsed_s"] = 11

    with self.assertRaisesRegex(ValueError, "fixed budget"):
      tensor_evaluation.summarize_fixed_contract_study(ledger)

  def test_end_to_end_cost_uses_observed_elapsed_time(self):
    report = tensor_evaluation.summarize_fixed_contract_study(_ledger())
    summary = report["routes"]["exact:exact-contract-v1:exact-output-property"][
      "policies"
    ]["random"]

    self.assertEqual(summary["end_to_end_cost_per_useful_execution_s"], 2.0)
    self.assertAlmostEqual(summary["stage_accounted_cost_per_useful_execution_s"], 1.3)

  def test_unrun_controls_are_recorded_without_false_negatives(self):
    ledger = _ledger()
    ledger["protocol"] = {
      "calibration_control_contract": [
        {"control_id": "correct-layout-baseline", "expected_detection": False},
        {"control_id": "add-one-output-mutant", "expected_detection": True},
      ]
    }
    for trial in ledger["trials"]:
      trial["admission_status"] = "refused"
      trial["execution_status"] = "not_run"
      trial["calibration_controls"] = []

    report = tensor_evaluation.summarize_fixed_contract_study(ledger)
    calibration = report["routes"]["exact:exact-contract-v1:exact-output-property"][
      "policies"
    ]["random"]["calibration"]

    self.assertEqual(calibration["unobserved_controls"], 2)
    self.assertEqual(calibration["false_negative"], 0)
    self.assertIsNone(calibration["sensitivity"])
    self.assertIsNone(calibration["specificity"])

  def test_partial_control_observations_do_not_invent_missing_results(self):
    ledger = _ledger()
    ledger["protocol"] = {
      "calibration_control_contract": [
        {"control_id": "correct-layout-baseline", "expected_detection": False},
        {"control_id": "add-one-output-mutant", "expected_detection": True},
      ]
    }
    for trial in ledger["trials"]:
      trial["execution_status"] = "execution_failure"
      trial["calibration_controls"] = [
        {
          "control_id": "correct-layout-baseline",
          "expected_detection": False,
          "observed_detection": False,
        }
      ]

    report = tensor_evaluation.summarize_fixed_contract_study(ledger)
    calibration = report["routes"]["exact:exact-contract-v1:exact-output-property"][
      "policies"
    ]["solver"]["calibration"]

    self.assertEqual(calibration["true_negative"], 1)
    self.assertEqual(calibration["false_negative"], 0)
    self.assertEqual(calibration["unobserved_controls"], 1)
    self.assertIsNone(calibration["sensitivity"])
    self.assertEqual(calibration["specificity"], 1.0)

  def test_reports_budget_overrun_on_timeout_without_dropping_the_trial(self):
    ledger = _ledger()
    ledger["trials"][0]["admission_status"] = "timeout"
    ledger["trials"][0]["execution_status"] = "not_run"
    ledger["trials"][0]["elapsed_s"] = 11

    report = tensor_evaluation.summarize_fixed_contract_study(ledger)
    random_policy = report["routes"]["exact:exact-contract-v1:exact-output-property"][
      "policies"
    ]["random"]

    self.assertEqual(random_policy["timeouts"], 1)
    self.assertEqual(random_policy["budget_overrun_s"], 1.0)

  def test_runner_persists_refused_arms_before_summary(self):
    selected = {
      "status": "proposal_exhausted",
      "generation_s": 0.0,
      "selection_s": 0.0,
      "reference_validation_s": 0.0,
      "proposal_count": 1,
      "values": None,
      "input_bytes": None,
    }
    with tempfile.TemporaryDirectory() as root:
      output = Path(root) / "study"
      with (
        mock.patch.object(
          run_selection_study.tensor_fixed_study,
          "select_random_witness",
          return_value=selected,
        ) as random_selector,
        mock.patch.object(
          run_selection_study.subprocess,
          "run",
          side_effect=FileNotFoundError("missing"),
        ),
      ):
        report = run_selection_study.run(
          output,
          solver="missing-symirsolve",
          seed=17,
          proposal_budget=1,
          budget_s=1.0,
        )

      ledger = json.loads((output / "study.json").read_text())
      self.assertEqual(len(ledger["trials"]), 2)
      self.assertEqual(ledger["trials"][0]["admission_status"], "refused")
      self.assertEqual(ledger["trials"][1]["admission_status"], "refused")
      self.assertTrue((output / "summary.json").is_file())
      self.assertIn("deadline", random_selector.call_args.kwargs)
      self.assertEqual(report["ledger"]["trials"], ledger["trials"])

  def test_runner_persists_unsat_solver_arm_without_gpu_execution(self):
    selected = {
      "status": "proposal_exhausted",
      "generation_s": 0.0,
      "selection_s": 0.0,
      "reference_validation_s": 0.0,
      "proposal_count": 1,
      "values": None,
      "input_bytes": None,
    }
    with tempfile.TemporaryDirectory() as root:
      output = Path(root) / "study"
      with (
        mock.patch.object(
          run_selection_study.tensor_fixed_study,
          "select_random_witness",
          return_value=selected,
        ),
        mock.patch.object(
          run_selection_study.subprocess,
          "run",
          side_effect=lambda command, **kwargs: __import__(
            "subprocess"
          ).CompletedProcess(command, 1, "UNSAT\n", ""),
        ),
      ):
        run_selection_study.run(
          output,
          solver="unsat-symirsolve",
          seed=17,
          proposal_budget=1,
          budget_s=1.0,
        )

      trials = json.loads((output / "study.json").read_text())["trials"]
      self.assertEqual(trials[0]["admission_status"], "refused")
      self.assertEqual(trials[1]["admission_status"], "unsat")
      self.assertEqual(trials[1]["execution_status"], "not_run")

  def test_runner_persists_solver_timeout_and_partial_output(self):
    selected = {
      "status": "proposal_exhausted",
      "generation_s": 0.0,
      "selection_s": 0.0,
      "reference_validation_s": 0.0,
      "proposal_count": 1,
      "values": None,
      "input_bytes": None,
    }

    def timeout(command, **kwargs):
      raise __import__("subprocess").TimeoutExpired(
        command, kwargs["timeout"], output=b"partial SAT", stderr=b"partial log"
      )

    with tempfile.TemporaryDirectory() as root:
      output = Path(root) / "study"
      with (
        mock.patch.object(
          run_selection_study.tensor_fixed_study,
          "select_random_witness",
          return_value=selected,
        ),
        mock.patch.object(run_selection_study.subprocess, "run", side_effect=timeout),
      ):
        run_selection_study.run(
          output,
          solver="timeout-symirsolve",
          seed=17,
          proposal_budget=1,
          budget_s=2.0,
        )

      trials = json.loads((output / "study.json").read_text())["trials"]
      self.assertEqual(trials[1]["admission_status"], "timeout")
      self.assertEqual((output / "solver/selection.stdout").read_text(), "partial SAT")
      self.assertEqual((output / "solver/selection.stderr").read_text(), "partial log")

  def test_runner_persists_failed_gpu_child_execution(self):
    selected = {
      "status": "accepted",
      "generation_s": 0.0,
      "selection_s": 0.0,
      "reference_validation_s": 0.0,
      "proposal_count": 1,
      "values": [1, 2, 4, 8],
      "input_bytes": struct.pack("<4f", 1, 2, 4, 8),
    }

    def fail_gpu_then_unsat(command, **kwargs):
      completed = __import__("subprocess").CompletedProcess
      if any("run_exact_loop" in argument for argument in command):
        return completed(command, 1, "partial child output", "child failed")
      return completed(command, 1, "UNSAT\n", "")

    with tempfile.TemporaryDirectory() as root:
      output = Path(root) / "study"
      with (
        mock.patch.object(
          run_selection_study.tensor_fixed_study,
          "select_random_witness",
          return_value=selected,
        ),
        mock.patch.object(
          run_selection_study.subprocess, "run", side_effect=fail_gpu_then_unsat
        ),
      ):
        run_selection_study.run(
          output,
          solver="unsat-symirsolve",
          seed=17,
          proposal_budget=1,
          budget_s=1.0,
        )

      trials = json.loads((output / "study.json").read_text())["trials"]
      self.assertEqual(trials[0]["execution_status"], "execution_failure")
      self.assertEqual(trials[1]["admission_status"], "unsat")
      self.assertEqual((output / "random/gluon.stderr").read_text(), "child failed")

  def test_random_selection_stops_at_its_deadline(self):
    selected = run_selection_study.tensor_fixed_study.select_random_witness(
      source="unused",
      domain=(0, 8),
      target=63,
      target_index=(0, 0),
      seed=1,
      proposal_budget=10,
      deadline=time.perf_counter() - 1,
    )

    self.assertEqual(selected["status"], "timeout")
    self.assertIsNone(selected["input_bytes"])

    ledger = _ledger()
    ledger["trials"][0]["cost_s"]["compilation_s"] = 9
    with self.assertRaisesRegex(ValueError, "exceed elapsed_s"):
      tensor_evaluation.summarize_fixed_contract_study(ledger)

  def test_rejects_a_policy_pair_that_does_not_share_the_frozen_domain(self):
    ledger = _ledger()
    ledger["trials"][1]["input_domain_id"] = "different-domain"

    with self.assertRaisesRegex(ValueError, "frozen input_domain_id"):
      tensor_evaluation.summarize_fixed_contract_study(ledger)

  def test_reports_calibration_sensitivity_and_specificity(self):
    report = tensor_evaluation.summarize_fixed_contract_study(_ledger())
    calibration = report["routes"]["exact:exact-contract-v1:exact-output-property"][
      "policies"
    ]["solver"]["calibration"]

    self.assertEqual(calibration["true_positive"], 1)
    self.assertEqual(calibration["true_negative"], 1)
    self.assertEqual(calibration["sensitivity"], 1.0)
    self.assertEqual(calibration["specificity"], 1.0)

  def test_requires_positive_and_negative_calibration_controls(self):
    ledger = _ledger()
    ledger["trials"][0]["calibration_controls"] = []

    report = tensor_evaluation.summarize_fixed_contract_study(ledger)
    calibration = report["routes"]["exact:exact-contract-v1:exact-output-property"][
      "policies"
    ]["random"]["calibration"]
    self.assertEqual(calibration["unobserved_controls"], 2)
    self.assertEqual(calibration["false_negative"], 0)

  def test_rejects_calibration_with_only_positive_controls(self):
    ledger = _ledger()
    ledger["trials"][0]["calibration_controls"] = ledger["trials"][0][
      "calibration_controls"
    ][:1]

    with self.assertRaisesRegex(ValueError, "positive and negative controls"):
      tensor_evaluation.summarize_fixed_contract_study(ledger)

  def test_rejects_duplicate_calibration_control_ids(self):
    ledger = _ledger()
    controls = ledger["trials"][0]["calibration_controls"]
    controls.append(dict(controls[0]))

    with self.assertRaisesRegex(ValueError, "duplicate ids"):
      tensor_evaluation.summarize_fixed_contract_study(ledger)

  def test_requires_policy_mates_to_use_the_same_calibration_controls(self):
    ledger = _ledger()
    ledger["trials"][1]["calibration_controls"][0]["control_id"] = (
      "different-known-mutant"
    )

    with self.assertRaisesRegex(ValueError, "undeclared calibration control"):
      tensor_evaluation.summarize_fixed_contract_study(ledger)

  def test_requires_policy_mates_to_agree_on_control_polarity(self):
    ledger = _ledger()
    controls = ledger["trials"][1]["calibration_controls"]
    controls[0]["expected_detection"], controls[1]["expected_detection"] = (
      controls[1]["expected_detection"],
      controls[0]["expected_detection"],
    )

    with self.assertRaisesRegex(
      ValueError, "changed a calibration control expectation"
    ):
      tensor_evaluation.summarize_fixed_contract_study(ledger)

  def test_keeps_compiler_observations_separate_for_different_contracts(self):
    ledger = _ledger()
    ledger["trials"].extend(
      [
        _trial("random", route="payload"),
        _trial("solver", route="payload"),
      ]
    )

    report = tensor_evaluation.summarize_fixed_contract_study(ledger)
    exact = report["routes"]["exact:exact-contract-v1:exact-output-property"]
    payload = report["routes"]["payload:payload-contract-v1:payload-output-property"]
    solver_compiler = exact["policies"]["solver"]["compiler"]

    self.assertEqual(exact["paired_trials"], 1)
    self.assertEqual(payload["paired_trials"], 1)
    self.assertEqual(solver_compiler["effective_layout_conversions"], 1)
    self.assertEqual(
      solver_compiler["conversion_operations"]["compute_to_reduction"], 1
    )
    self.assertEqual(solver_compiler["reused_values"], 1)
    self.assertEqual(solver_compiler["duplicate_expression_groups"], 1)
    self.assertEqual(solver_compiler["retained_loop_compilations"], 1)
    self.assertEqual(solver_compiler["ptx_warp_shuffles"], 2)
    self.assertIn("comparison is route/property local", report["comparison_scope"])

  def test_rejects_missing_mate_in_paired_policy_protocol(self):
    ledger = _ledger()
    ledger["trials"] = ledger["trials"][:1]

    with self.assertRaisesRegex(ValueError, "missing selection-policy mate"):
      tensor_evaluation.summarize_fixed_contract_study(ledger)


class TensorSelectionStudyAggregationTests(unittest.TestCase):
  def setUp(self):
    self.gpu_report = {
      "cases": {
        "baseline": {
          "compiler": {
            "observation": {
              "layout_conversions": [],
              "layout_conversion_count": 0,
              "effective_conversion_operations": [],
              "loop_status": "retained",
              "stages": {
                "ttgir": {
                  "reused_values": [{"value": "%a", "use_count": 2}],
                  "duplicate_expressions": [],
                  "loop_construct_count": 1,
                },
                "ptx": {
                  "warp_shuffle_count": 2,
                  "shared_memory_transfer_count": 1,
                },
              },
            }
          },
          "measurement": {
            "first_launch_wall_s_including_compilation": 1.0,
            "median_gpu_event_ms_per_launch": 0.5,
            "measured_launches": 10,
          },
        }
      },
      "calibration_compiler": {
        "observation": {
          "layout_conversions": [
            {
              "layout_changed": True,
              "source_operation": "compute_to_store",
              "stage": "ttgir",
            }
          ],
          "layout_conversion_count": 1,
          "effective_conversion_operations": ["compute_to_store"],
          "loop_status": "unrolled_or_removed",
          "stages": {
            "ttgir": {
              "reused_values": [],
              "duplicate_expressions": [
                {"expression": "tt.add %a, %b", "values": ["%x", "%y"], "count": 2}
              ],
              "loop_construct_count": 0,
            },
            "ptx": {
              "warp_shuffle_count": 3,
              "shared_memory_transfer_count": 4,
            },
          },
        },
        "measurement": {
          "first_launch_wall_s_including_compilation": 2.0,
          "median_gpu_event_ms_per_launch": 1.5,
          "measured_launches": 10,
        },
      },
    }

  def test_counts_calibration_mutant_compilation(self):
    observation = run_selection_study._combined_compiler_observation(self.gpu_report)

    self.assertEqual(observation["compilation_count"], 2)

  def test_counts_calibration_mutant_effective_conversion(self):
    observation = run_selection_study._combined_compiler_observation(self.gpu_report)

    self.assertEqual(observation["layout_conversion_count"], 1)
    self.assertEqual(
      observation["effective_conversion_operations"], ["compute_to_store"]
    )

  def test_counts_calibration_mutant_loop_form(self):
    observation = run_selection_study._combined_compiler_observation(self.gpu_report)

    self.assertEqual(observation["loop_status"], "unrolled_or_removed")
    self.assertEqual(observation["stages"]["ttgir"]["loop_construct_count"], 1)

  def test_counts_calibration_mutant_ptx_communication(self):
    observation = run_selection_study._combined_compiler_observation(self.gpu_report)

    self.assertEqual(observation["stages"]["ptx"]["warp_shuffle_count"], 5)
    self.assertEqual(observation["stages"]["ptx"]["shared_memory_transfer_count"], 5)

  def test_counts_calibration_mutant_compile_and_execution_cost(self):
    compilation_s, execution_s = run_selection_study._gpu_costs(self.gpu_report)

    self.assertEqual(compilation_s, 3.0)
    self.assertEqual(execution_s, 0.02)

  def test_timeout_preserves_partial_byte_output(self):
    with tempfile.TemporaryDirectory() as root:
      output = Path(root)
      arm = output / "random"
      arm.mkdir()
      trial = run_selection_study._trial_record("random", 17, 10.0)
      trial["admission_status"] = "admitted"
      trial["selection_record"] = {}
      error = __import__("subprocess").TimeoutExpired(
        ["gluon"], 1, output=b"partial stdout", stderr=b"partial stderr"
      )
      with mock.patch.object(run_selection_study.subprocess, "run", side_effect=error):
        run_selection_study._run_gpu_policy(
          trial,
          output,
          {"values": [1, 2, 4, 8], "input_bytes": b"\0" * 16},
          budget_s=10.0,
          started=time.perf_counter(),
        )

      self.assertEqual((arm / "gluon.stdout").read_text(), "partial stdout")
      self.assertEqual((arm / "gluon.stderr").read_text(), "partial stderr")
      self.assertEqual(trial["execution_status"], "execution_failure")


if __name__ == "__main__":
  unittest.main()
