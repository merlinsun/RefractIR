"""Contracts for paired FpSan and exact-route accounting."""

import importlib
import tempfile
import unittest
from pathlib import Path


def _matching_reports():
  program_id = "native-seed-7"
  source_sha256 = "a" * 64
  policy = {
    "attempts": 1,
    "admitted": 1,
    "admission_rate": 1.0,
    "gpu_executions": 1,
    "useful_executions": 1,
    "mismatches": 0,
    "refused": 0,
    "timeouts": 0,
    "unsat": 0,
    "cost_s": {"compilation_s": 4.0, "gpu_execution_s": 0.02},
    "end_to_end_cost_per_useful_execution_s": 5.0,
    "calibration": {"sensitivity": 1.0, "specificity": 1.0},
    "compiler": {
      "compilations": 6,
      "retained_loop_compilations": 5,
      "effective_layout_conversions": 12,
      "ptx_warp_shuffles": 8,
      "ptx_shared_memory_transfers": 2,
    },
  }
  fpsan = {
    "generated_region_protocol": {
      "semantic_contract": "generated-region-qf-bv-payload-ring-v1",
      "logical_input_domain": [0, 8],
      "input_transport": "pinned-mixer raw f32 words",
      "property": {
        "kind": "output-payload-equals",
        "target": 1268,
        "index": [0, 0],
      },
    },
    "generated_regions": {
      "programs": {
        program_id: {
          "status": "pass",
          "region_source": {"sha256": source_sha256},
          "selection": {
            "status": "SAT",
            "cost_s": {"solver_session_s": 0.25, "solver_invocations": 1},
          },
          "measurement": {"median_gpu_event_ms_per_launch": 0.04},
          "ordinary_reference": {
            "status": "pass",
            "admission": {"status": "admitted"},
            "measurement": {"median_gpu_event_ms_per_launch": 0.03},
          },
          "calibration_mutants": {
            "lost_iteration": {"status": "detected"},
            "lost_update_acc_y": {"status": "detected"},
          },
          "measurement_accounting": {
            "useful_compilations": 2,
            "calibration_compilations": 2,
            "useful_executions": 2,
            "calibration_executions": 2,
          },
          "compiler_ir": {
            "layout_conversion_operations_in_ttgir": 3,
            "warp_shuffle_operations_in_ptx": 1,
            "shared_memory_transfers_in_ptx": 0,
          },
        }
      }
    },
  }
  exact = {
    "status": "complete",
    "protocol": {
      "semantic_contract": "generated-multi-output-region-f32-exact-loop-v1",
      "input_domain": [0, 8],
      "property_target": 1268,
    },
    "cells": [
      {
        "program_id": program_id,
        "program_sha256": source_sha256,
        "property_id": "output-element-0-0-equals-1268",
        "target": 1268,
        "target_index": [0, 0],
        "summary": {
          "routes": {
            "exact-f32:contract:output-element-0-0-equals-1268": {
              "property_id": "output-element-0-0-equals-1268",
              "policies": {"random": policy, "solver": dict(policy)},
            }
          }
        },
      }
    ],
  }
  return fpsan, exact


def _gpu_measurement(*, warmups, measured):
  return {
    "first_launch_wall_s_including_compilation": 0.1,
    "warmup_launches": warmups,
    "measured_launches": measured,
  }


def _gpu_case(*, warmups=3, measured=30):
  return {
    "compiler": {"observation": {"compilation_count": 1}},
    "measurement": _gpu_measurement(warmups=warmups, measured=measured),
  }


def _gpu_report(*, cases=None, expected_layout_count=2, calibration=True):
  return {
    "cases": cases or {"layout-a": _gpu_case(), "layout-b": _gpu_case()},
    "calibration_compiler": (
      {
        "observation": {"compilation_count": 1},
        "measurement": _gpu_measurement(warmups=2, measured=7),
      }
      if calibration
      else None
    ),
    "measurement_protocol": {
      "expected_layout_count": expected_layout_count,
      "measurement_accounting": {"calibration_compilations": 1},
    },
  }


class TensorRouteProtocolTests(unittest.TestCase):
  def test_cache_policy_creates_only_a_fresh_empty_directory(self):
    protocol = importlib.import_module("test.lib.tensor_route_protocol")
    with tempfile.TemporaryDirectory() as temporary:
      cache_path = Path(temporary) / "cold-cache"

      record = protocol.prepare_fresh_cache(cache_path)

      self.assertEqual(record["path"], str(cache_path.resolve()))
      self.assertTrue(record["empty_at_start"])
      (cache_path / "compiled.bin").write_text("stale")
      with self.assertRaisesRegex(ValueError, "must be empty"):
        protocol.prepare_fresh_cache(cache_path)

  def test_accounting_separates_useful_and_calibration_compiles(self):
    protocol = importlib.import_module("test.lib.tensor_route_protocol")

    accounting = protocol.measurement_accounting(
      useful_compilations=5,
      calibration_compilations=1,
      useful_executions=5,
      calibration_executions=1,
    )

    self.assertEqual(accounting["total_compilations"], 6)
    self.assertEqual(accounting["total_executions"], 6)
    self.assertEqual(accounting["useful_executions"], 5)
    self.assertEqual(accounting["calibration_executions"], 1)
    self.assertEqual(accounting["total_warmup_launches"], 18)
    self.assertEqual(accounting["total_measured_launches"], 180)

  def test_route_intersection_requires_same_hash_domain_and_target(self):
    protocol = importlib.import_module("test.lib.tensor_route_protocol")
    fpsan, exact = _matching_reports()

    comparison = protocol.compare_route_reports(fpsan, exact)

    self.assertEqual(comparison["status"], "complete")
    self.assertEqual(comparison["common_admitted_program_ids"], ["native-seed-7"])
    self.assertEqual(comparison["programs"][0]["target"], 1268)
    self.assertTrue(comparison["programs"][0]["source_hash_matches"])

    exact["cells"][0]["program_sha256"] = "b" * 64
    comparison = protocol.compare_route_reports(fpsan, exact)
    self.assertEqual(comparison["common_admitted_program_ids"], [])
    self.assertIn("source hash differs", comparison["programs"][0]["exclusions"])

  def test_route_intersection_excludes_different_properties(self):
    protocol = importlib.import_module("test.lib.tensor_route_protocol")
    fpsan, exact = _matching_reports()
    exact["cells"][0]["target"] = 1408

    comparison = protocol.compare_route_reports(fpsan, exact)

    self.assertEqual(comparison["common_admitted_program_ids"], [])
    self.assertIn("target differs", comparison["programs"][0]["exclusions"])

  def test_route_intersection_excludes_different_target_indices(self):
    protocol = importlib.import_module("test.lib.tensor_route_protocol")
    fpsan, exact = _matching_reports()
    exact["cells"][0]["target_index"] = [1, 1]

    comparison = protocol.compare_route_reports(fpsan, exact)

    self.assertEqual(comparison["common_admitted_program_ids"], [])
    self.assertIn("target index differs", comparison["programs"][0]["exclusions"])

  def test_route_intersection_excludes_different_input_domains(self):
    protocol = importlib.import_module("test.lib.tensor_route_protocol")
    fpsan, exact = _matching_reports()
    exact["protocol"]["input_domain"] = [1, 8]

    comparison = protocol.compare_route_reports(fpsan, exact)

    self.assertEqual(comparison["common_admitted_program_ids"], [])
    self.assertIn("input domain differs", comparison["programs"][0]["exclusions"])

  def test_route_intersection_preserves_distinct_numeric_contracts(self):
    protocol = importlib.import_module("test.lib.tensor_route_protocol")
    fpsan, exact = _matching_reports()

    comparison = protocol.compare_route_reports(fpsan, exact)

    self.assertFalse(comparison["semantic_equivalence_claim"])
    self.assertNotEqual(
      comparison["routes"]["fpsan"]["semantic_contract"],
      comparison["routes"]["exact"]["semantic_contract"],
    )
    self.assertIn("raw input words differ", comparison["interpretation_limit"])

  def test_route_metrics_report_selection_and_gpu_costs_separately(self):
    protocol = importlib.import_module("test.lib.tensor_route_protocol")
    fpsan, exact = _matching_reports()

    metrics = protocol.compare_route_reports(fpsan, exact)["programs"][0][
      "route_metrics"
    ]

    self.assertEqual(metrics["fpsan"]["selection_status"], "SAT")
    self.assertEqual(metrics["fpsan"]["selection_cost_s"]["solver_session_s"], 0.25)
    self.assertEqual(
      metrics["exact"]["policies"]["random"]["cost_s"]["compilation_s"], 4.0
    )

  def test_route_metrics_retain_calibration_sensitivity_and_mutants(self):
    protocol = importlib.import_module("test.lib.tensor_route_protocol")
    fpsan, exact = _matching_reports()

    metrics = protocol.compare_route_reports(fpsan, exact)["programs"][0][
      "route_metrics"
    ]

    self.assertEqual(
      metrics["exact"]["policies"]["solver"]["calibration"]["sensitivity"], 1.0
    )
    self.assertEqual(metrics["fpsan"]["calibration_mutants_detected"], 2)

  def test_route_metrics_retain_observed_compiler_transformations(self):
    protocol = importlib.import_module("test.lib.tensor_route_protocol")
    fpsan, exact = _matching_reports()

    metrics = protocol.compare_route_reports(fpsan, exact)["programs"][0][
      "route_metrics"
    ]

    self.assertEqual(
      metrics["exact"]["policies"]["random"]["compiler"][
        "effective_layout_conversions"
      ],
      12,
    )
    self.assertEqual(metrics["fpsan"]["compiler"]["warp_shuffles"], 1)

  def test_route_metrics_separate_useful_from_calibration_work(self):
    protocol = importlib.import_module("test.lib.tensor_route_protocol")
    fpsan, exact = _matching_reports()

    metrics = protocol.compare_route_reports(fpsan, exact)["programs"][0][
      "route_metrics"
    ]

    self.assertEqual(
      metrics["fpsan"]["measurement_accounting"]["useful_compilations"], 2
    )
    self.assertEqual(
      metrics["fpsan"]["measurement_accounting"]["calibration_compilations"], 2
    )
    exact_accounting = metrics["exact"]["policies"]["random"]["measurement_accounting"]
    self.assertEqual(exact_accounting["status"], "unknown")
    self.assertIsNone(exact_accounting["total"]["total_compilations"])

  def test_gpu_work_accounting_uses_observed_layout_and_calibration_records(self):
    protocol = importlib.import_module("test.lib.tensor_route_protocol")

    accounting = protocol.gpu_report_work_accounting(_gpu_report())

    self.assertEqual(accounting["status"], "complete")
    self.assertEqual(accounting["observed"]["useful_compilations"], 2)
    self.assertEqual(accounting["observed"]["calibration_compilations"], 1)
    self.assertEqual(accounting["observed"]["useful_warmup_launches"], 6)
    self.assertEqual(accounting["observed"]["calibration_warmup_launches"], 2)
    self.assertEqual(accounting["observed"]["total_measured_launches"], 67)
    self.assertEqual(accounting["total"], accounting["observed"])

  def test_partial_gpu_report_keeps_lower_bounds_and_withholds_totals(self):
    protocol = importlib.import_module("test.lib.tensor_route_protocol")

    accounting = protocol.gpu_report_work_accounting(
      _gpu_report(
        cases={"layout-a": _gpu_case(warmups=1, measured=4)}, calibration=False
      )
    )

    self.assertEqual(accounting["status"], "partial")
    self.assertEqual(accounting["observed"]["useful_compilations"], 1)
    self.assertEqual(accounting["observed"]["useful_measured_launches"], 4)
    self.assertIsNone(accounting["total"]["total_compilations"])
    self.assertIsNone(accounting["total"]["total_measured_launches"])

  def test_empty_gpu_report_is_unknown_instead_of_zero_work(self):
    protocol = importlib.import_module("test.lib.tensor_route_protocol")

    accounting = protocol.gpu_report_work_accounting({})

    self.assertEqual(accounting["status"], "unknown")
    self.assertEqual(accounting["observed"]["total_compilations"], 0)
    self.assertIsNone(accounting["total"]["total_compilations"])

  def test_exact_route_metrics_use_accounting_retained_in_policy_summary(self):
    protocol = importlib.import_module("test.lib.tensor_route_protocol")
    fpsan, exact = _matching_reports()
    exact["cells"][0]["summary"]["routes"][
      "exact-f32:contract:output-element-0-0-equals-1268"
    ]["policies"]["random"]["work_accounting"] = protocol.gpu_report_work_accounting(
      _gpu_report(
        cases={
          "layout-a": _gpu_case(warmups=1, measured=4),
          "layout-b": _gpu_case(warmups=2, measured=5),
        }
      )
    )

    accounting = protocol.compare_route_reports(fpsan, exact)["programs"][0][
      "route_metrics"
    ]["exact"]["policies"]["random"]["measurement_accounting"]

    self.assertEqual(accounting["status"], "complete")
    self.assertEqual(accounting["observed"]["useful_compilations"], 2)
    self.assertEqual(accounting["observed"]["total_measured_launches"], 16)

  def test_route_comparison_records_budgets_without_merging_them(self):
    protocol = importlib.import_module("test.lib.tensor_route_protocol")
    fpsan, exact = _matching_reports()
    fpsan["generated_region_protocol"]["selection_budget_s_per_program"] = 60
    exact["protocol"]["budget_s_per_policy"] = 120

    comparison = protocol.compare_route_reports(fpsan, exact)

    self.assertEqual(comparison["routes"]["fpsan"]["budget_s_per_program"], 60)
    self.assertEqual(comparison["routes"]["exact"]["budget_s_per_policy"], 120)
    self.assertFalse(comparison["semantic_equivalence_claim"])


if __name__ == "__main__":
  unittest.main()
