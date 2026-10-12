"""Committed device evidence records must be machine-readable."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "tensor" / "campaigns" / "results"


class TensorEvidenceRecordTests(unittest.TestCase):
  """A record cited as measurement evidence has to parse as its own format."""

  def records(self):
    self.assertTrue(RESULTS.is_dir(), f"missing results directory: {RESULTS}")
    # A copy through a non-native filesystem can leave AppleDouble companions
    # (`._name.json`) beside the records; they are metadata, not evidence.
    return sorted(path for path in RESULTS.glob("*.json") if not path.name.startswith("._"))

  def test_every_record_parses_as_json(self):
    files = self.records()
    self.assertTrue(files, "no evidence records were found")

    for path in files:
      with self.subTest(record=path.name):
        text = path.read_text(encoding="utf-8")
        try:
          json.loads(text)
        except json.JSONDecodeError as error:
          self.fail(
            f"{path.name} is not JSON ({error}); a prose report belongs in the "
            "matching .md file, not in a record the evidence cites"
          )

  def test_device_records_are_structured_and_not_prose(self):
    for path in self.records():
      if not path.name.startswith(("track-", "sweep-", "summary-")):
        continue
      with self.subTest(record=path.name):
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsInstance(value, (dict, list), path.name)
        self.assertTrue(value, f"{path.name} is empty")
        self.assertFalse(
          path.read_text(encoding="utf-8").lstrip().startswith("#"),
          f"{path.name} starts with a Markdown heading",
        )

  def test_the_guard_control_record_carries_its_verdict_and_artifacts(self):
    path = RESULTS / "track-b4-guard-control-evidence.json"
    self.assertTrue(path.is_file(), "the guard control record is missing")
    record = json.loads(path.read_text(encoding="utf-8"))

    self.assertTrue(record["refused"], "the control must have been refused")
    self.assertIn("prefix-guard-0", record["reason"])
    self.assertTrue(
      all(record["artifacts_present"].values()),
      "a refusal must still leave the run's artifacts",
    )


if __name__ == "__main__":
  unittest.main()
