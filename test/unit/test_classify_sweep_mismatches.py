"""Host-only first triage step: the mismatch bits of a sweep's finite leads."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from test.unit import test_moving_window_sweep as fixtures
from test.unit.test_moving_window_sweep import I0, I3, S0, TWO, Script, ticking

sweep = fixtures.sweep
TOOL_PATH = fixtures.DRIVER_PATH.with_name("classify_sweep_mismatches.py")
_spec = importlib.util.spec_from_file_location("classify_mismatches", TOOL_PATH)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


class BitClassTests(unittest.TestCase):
  def test_each_mismatch_falls_into_one_class(self):
    cases = {
      ("0x00000000", "0x80000000"): "signed_zero",
      ("0x80000000", "0x00000000"): "signed_zero",
      ("0x3f800000", "0xbf800000"): "sign_flip",
      ("0x7fc00000", "0x3f800000"): "nan",
      ("0x3f800000", "0x3f800001"): "rounding",
      ("0x40400000", "0x40a00000"): "value",
      ("length=16", "length=12"): "length",
    }
    for (first, second), expected in cases.items():
      with self.subTest(first=first, second=second):
        self.assertEqual(tool.classify(first, second), expected)


class LeadTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name)

  def shard(self, index, script):
    work = self.root / f"shard-{index}"
    declared = sweep.sweep_spec(
      seed_start=S0 + 1000 * index,
      seed_count=1,
      configurations=fixtures.CONFIGURATIONS,
      sample_every=20,
      gpu_workers=1000,
      wall_clock_s=10**6,
    )
    sweep.run_sweep(
      work,
      declared,
      script.operations,
      free_bytes=lambda path: 10**12,
      clock=ticking(60),
    )
    return work

  def record(self, work, mismatches):
    """Write the paired run's record where the binding keeps it."""
    for line in (work / "candidates.jsonl").read_text().splitlines():
      candidate = json.loads(line)
      artifacts = Path(candidate["directory"]) / "bundle" / "target-runs"
      artifacts.mkdir(parents=True, exist_ok=True)
      record = {
        "agreement": {
          "kind": "different",
          "mismatches": [
            {"index": index, "interpreter_bits": first, "compiled_bits": second}
            for index, first, second in mismatches
          ],
        }
      }
      (artifacts / "target-run.json").write_text(json.dumps(record))

  def difference(self, seed):
    return Script(outputs={(seed, 1, I0, "p2"): TWO, (seed, 1, I3, "p2"): TWO})

  def test_each_lead_reports_its_classes_and_the_totals_count_dominant_classes(self):
    zero = self.shard(0, self.difference(S0))
    self.record(zero, [(3, "0x00000000", "0x80000000")])
    value = self.shard(1, self.difference(S0 + 1000))
    self.record(
      value, [(3, "0x40800000", "0x40a00000"), (5, "0x00000000", "0x80000000")]
    )
    result = tool.classify_leads([zero, value])
    leads = {lead["seed"]: lead for lead in result["leads"]}
    self.assertEqual(leads[S0]["classes"], {"signed_zero": 2})
    self.assertEqual(leads[S0]["dominant"], "signed_zero")
    self.assertEqual(leads[S0 + 1000]["classes"], {"signed_zero": 2, "value": 2})
    self.assertEqual(leads[S0 + 1000]["dominant"], "value")
    self.assertEqual(result["by_dominant_class"], {"signed_zero": 1, "value": 1})
    example = leads[S0 + 1000]["examples"][0]
    self.assertEqual((example["interpreter"], example["compiled"]), (4.0, 5.0))

  def test_a_missing_or_truncated_record_is_reported(self):
    missing = self.shard(0, self.difference(S0))
    truncated = self.shard(1, self.difference(S0 + 1000))
    self.record(truncated, [(i, "0x3f800000", "0x40000000") for i in range(16)])
    result = tool.classify_leads([missing, truncated])
    leads = {lead["seed"]: lead for lead in result["leads"]}
    self.assertEqual(leads[S0]["dominant"], "record_missing")
    self.assertEqual(leads[S0]["missing_records"], 2)
    self.assertTrue(leads[S0 + 1000]["truncated"])
    self.assertFalse(leads[S0]["truncated"])


if __name__ == "__main__":
  unittest.main()
