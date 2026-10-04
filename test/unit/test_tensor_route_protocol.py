"""Contracts for paired FpSan and exact-route accounting."""

import importlib
import tempfile
import unittest
from pathlib import Path


def _matching_reports():
  program_id = "native-seed-7"
  source_sha256 = "a" * 64
  policy = {
    "admitted": 1,
    "gpu_executions": 1,
    "mismatches": 0,
    "refused": 0,
    "timeouts": 0,
    "unsat": 0,
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


if __name__ == "__main__":
  unittest.main()
