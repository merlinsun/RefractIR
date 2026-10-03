"""Fixed-domain study accounting for selection and compiler evidence."""

import unittest

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

    with self.assertRaisesRegex(ValueError, "positive and negative calibration"):
      tensor_evaluation.summarize_fixed_contract_study(ledger)

  def test_rejects_calibration_with_only_positive_controls(self):
    ledger = _ledger()
    ledger["trials"][0]["calibration_controls"] = ledger["trials"][0][
      "calibration_controls"
    ][:1]

    with self.assertRaisesRegex(ValueError, "positive and negative calibration"):
      tensor_evaluation.summarize_fixed_contract_study(ledger)

  def test_rejects_duplicate_calibration_control_ids(self):
    ledger = _ledger()
    controls = ledger["trials"][0]["calibration_controls"]
    controls.append(dict(controls[0]))

    with self.assertRaisesRegex(ValueError, "duplicate calibration control"):
      tensor_evaluation.summarize_fixed_contract_study(ledger)

  def test_requires_policy_mates_to_use_the_same_calibration_controls(self):
    ledger = _ledger()
    ledger["trials"][1]["calibration_controls"][0]["control_id"] = (
      "different-known-mutant"
    )

    with self.assertRaisesRegex(ValueError, "same calibration controls"):
      tensor_evaluation.summarize_fixed_contract_study(ledger)

  def test_requires_policy_mates_to_agree_on_control_polarity(self):
    ledger = _ledger()
    controls = ledger["trials"][1]["calibration_controls"]
    controls[0]["expected_detection"], controls[1]["expected_detection"] = (
      controls[1]["expected_detection"],
      controls[0]["expected_detection"],
    )

    with self.assertRaisesRegex(ValueError, "same calibration controls"):
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


if __name__ == "__main__":
  unittest.main()
