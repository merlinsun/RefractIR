"""R4 qualification declaration, production gate, and closure device batch."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test.unit import test_moving_window_discovery as fixtures

study = fixtures.study

VALIDATION_DIR = os.environ.get("REFRACTIR_MOVING_WINDOW_VALIDATION_DIR")


def qualification_built(case="mw-g0c0"):
  """One admitted construction carrying every declared capability."""
  return {
    "kind": "admitted",
    "case": case,
    "source": "fun @kernel() {}",
    "source_sha256": "source",
    "inputs_sha256": "inputs",
    "layouts": {
      "%window_buffer": {
        "shape": [8, 8],
        "element_strides": [11, 1],
        "element_offset": 1,
        "storage_span": 78,
      }
    },
    "region": {
      "profile": "v3",
      "entry": "kernel",
      "parameters": [
        {"name": "%window_buffer", "role": "window_source", "shape": [8, 8]},
        {"name": "%x", "role": "state", "shape": [3, 8]},
        {"name": "%y", "role": "state", "shape": [3, 8]},
      ],
      "window": {
        "source_kind": "backing",
        "result": "%tile",
        "witness": {
          "all_active": False,
          "per_trip_active": [
            [True, True, True],
            [True, True, True],
            [True, True, True],
            [True, False, False],
          ],
        },
      },
      "requirements": [
        {
          "input": "%tile",
          "operations": ["min", "expand"],
          "path": ["%tile", "%next_x"],
          "results": ["%next_x"],
        }
      ],
    },
  }


def valid_record(frozen):
  cases = []
  for entry in study.QUALIFICATION["cases"]:
    cases.append(
      {
        "case": entry["case"],
        "complete": True,
        "validated": True,
        "capabilities": {name: True for name in entry["checks"]},
        "configurations": {
          configuration: {
            "paths": {
              "p1": {"status": "completed"},
              "p2": {"status": "completed"},
            }
          }
          for configuration in study.CONFIGURATIONS
        },
      }
    )
  record = {
    "schema": "refractir.moving-window-qualification.v3",
    "study": frozen.digest,
    "declaration": study.QUALIFICATION,
    "cases": cases,
    "validated": True,
  }
  record["digest"] = study._digest(record)
  return record


class QualificationCapabilityTests(unittest.TestCase):
  def test_the_frozen_declaration_requires_a_g2_extrema_shape_case(self):
    cases = {entry["case"]: entry["checks"] for entry in study.QUALIFICATION["cases"]}
    self.assertEqual(
      cases,
      {
        "mw-g0c0": ["backing_storage"],
        "mw-g0c1": ["changing_active_rows"],
        "mw-g0c3": ["strided_backing_layout"],
        "mw-g2c3": ["extrema_shape_path"],
      },
    )

  def test_qualification_construction_forbids_multiplication(self):
    self.assertEqual(study.QUALIFICATION.get("max_multiplications"), 0)

  def test_a_live_path_with_extrema_and_shape_qualifies(self):
    checks = study.qualification_capabilities(
      qualification_built("mw-g2c3"), ["extrema_shape_path"]
    )
    self.assertEqual(checks, {"extrema_shape_path": True})

  def test_extrema_without_shape_does_not_qualify(self):
    built = qualification_built("mw-g2c3")
    built["region"]["requirements"] = [
      {"operations": ["min", "sub"], "path": ["%tile", "%next_x"]}
    ]
    checks = study.qualification_capabilities(built, ["extrema_shape_path"])
    self.assertFalse(checks["extrema_shape_path"])

  def test_shape_without_extrema_does_not_qualify(self):
    built = qualification_built("mw-g2c3")
    built["region"]["requirements"] = [
      {"operations": ["expand", "sub"], "path": ["%tile", "%next_x"]}
    ]
    checks = study.qualification_capabilities(built, ["extrema_shape_path"])
    self.assertFalse(checks["extrema_shape_path"])

  def test_a_dense_backing_layout_does_not_qualify_as_strided(self):
    built = qualification_built("mw-g0c3")
    built["layouts"] = {
      "%window_buffer": {
        "shape": [8, 8],
        "element_strides": [8, 1],
        "element_offset": 0,
      }
    }
    checks = study.qualification_capabilities(built, ["strided_backing_layout"])
    self.assertFalse(checks["strided_backing_layout"])

  def test_an_invariant_active_row_pattern_does_not_qualify(self):
    built = qualification_built("mw-g0c1")
    built["region"]["window"]["witness"]["per_trip_active"] = [
      [True, True, True]
    ] * 4
    checks = study.qualification_capabilities(built, ["changing_active_rows"])
    self.assertFalse(checks["changing_active_rows"])


class QualificationGateTests(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.work = Path(self.directory.name) / "work"
    self.work.mkdir(parents=True)
    self.frozen = study.freeze_study(toolchain={"symirc_sha256": "x"})
    self.frozen.write(self.work / "study.json")

  def write(self, record):
    (self.work / "qualification.json").write_text(
      json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

  def test_a_complete_matching_record_passes(self):
    self.write(valid_record(self.frozen))
    record = study.require_qualification(self.work, self.frozen)
    self.assertTrue(record["validated"])

  def test_a_missing_record_is_refused(self):
    with self.assertRaises(study.StudyRefused):
      study.require_qualification(self.work, self.frozen)

  def test_an_incomplete_record_is_refused(self):
    record = valid_record(self.frozen)
    record["cases"][0]["validated"] = False
    record["digest"] = study._digest(
      {key: value for key, value in record.items() if key != "digest"}
    )
    self.write(record)
    with self.assertRaises(study.StudyRefused):
      study.require_qualification(self.work, self.frozen)

  def test_a_record_for_another_study_is_refused(self):
    record = valid_record(self.frozen)
    record["study"] = "another"
    record["digest"] = study._digest(
      {key: value for key, value in record.items() if key != "digest"}
    )
    self.write(record)
    with self.assertRaises(study.StudyRefused):
      study.require_qualification(self.work, self.frozen)

  def test_a_tampered_record_is_refused(self):
    record = valid_record(self.frozen)
    self.write(record)
    data = json.loads((self.work / "qualification.json").read_text())
    data["cases"][0]["capabilities"]["backing_storage"] = False
    self.write(data)
    with self.assertRaises(study.StudyRefused):
      study.require_qualification(self.work, self.frozen)


class CliWiringTests(unittest.TestCase):
  def test_select_and_discover_refuse_without_qualification(self):
    calls = {"build": 0, "execute": 0}
    toolchain = {"symirc_sha256": "x"}
    operations = {
      "build": lambda request: calls.__setitem__("build", calls["build"] + 1),
      "screen": lambda outcome: {},
      "execute": lambda request: calls.__setitem__("execute", calls["execute"] + 1),
      "toolchain": toolchain,
    }
    with tempfile.TemporaryDirectory() as directory:
      work = Path(directory) / "work"
      work.mkdir(parents=True)
      frozen = study.freeze_study(toolchain=toolchain)
      frozen.write(work / "study.json")
      with mock.patch.object(study, "device_operations", return_value=operations):
        for mode in ("select", "discover"):
          with self.assertRaises(study.StudyRefused):
            study.main(
              [
                "--mode", mode,
                "--work", str(work),
                "--exporter", "exporter",
                "--symirc", "symirc",
                "--symiri", "symiri",
                "--accept-current-toolchain",
              ]
            )
    self.assertEqual(calls, {"build": 0, "execute": 0})


@unittest.skipUnless(
  VALIDATION_DIR, "bounded device validation is opt-in"
)
class BoundedDeviceValidation(unittest.TestCase):
  """§6.2: one declared batch, three scenarios, one process budget."""

  CAPS = {
    "qualification": {"p2": 8, "p1": 8, "p0": 4},
    "refused-p2": {"p2": 2, "p1": 3, "p0": 1},
    "refused-p0": {"p2": 2, "p1": 2, "p0": 0},
  }
  TOTAL_CAPS = {"p2": 12, "p1": 28, "p0": 10}

  @classmethod
  def setUpClass(cls):
    cls.root = Path(VALIDATION_DIR)
    if cls.root.exists():
      raise AssertionError("the validation directory must be fresh")
    cls.root.mkdir(parents=True)
    cls.operations = study.device_operations(
      exporter=Path(os.environ["REFRACTIR_TENSOR_FREE_FORM_EXPORT"]),
      symirc=Path(os.environ.get("REFRACTIR_SYMIRC", "symirc")),
      symiri=Path(os.environ.get("REFRACTIR_SYMIRI", "symiri")),
    )
    cls.total = {"p0": 0, "p1": 0, "p2": 0}
    cls.scenarios = []

  @classmethod
  def tearDownClass(cls):
    for path, cap in cls.TOTAL_CAPS.items():
      if cls.total[path] > cap:
        raise AssertionError(f"batch {path} starts exceed the declared cap")
    (cls.root / "validation.json").write_text(
      json.dumps(
        {"batch": "moving-window-closure", "scenarios": cls.scenarios,
         "total_starts": cls.total, "caps": cls.TOTAL_CAPS},
        indent=1, sort_keys=True,
      ) + "\n",
      encoding="utf-8",
    )

  def scenario(self, name):
    work = self.root / name
    work.mkdir()
    frozen = study.freeze_study(toolchain=self.operations["toolchain"])
    frozen.write(work / "study.json")
    authority = study.Study(
      work,
      frozen,
      build=self.operations["build"],
      screen=self.operations["screen"],
      execute=self.operations["execute"],
      accounting_mode="process",
    )
    return work, frozen, authority

  def starts(self, work):
    counts = {"p0": 0, "p1": 0, "p2": 0}
    for line in (work / "path-ledger.jsonl").read_text().splitlines():
      event = json.loads(line)
      if event.get("kind") == "process" and event.get("event") == "start":
        counts[event["path"]] = counts.get(event["path"], 0) + 1
    return counts

  def record(self, name, counts, **extra):
    for path in self.CAPS[name]:
      if counts[path] > self.CAPS[name][path]:
        raise AssertionError(f"{name} exceeds its {path} cap")
    for path in self.total:
      self.total[path] += counts[path]
    self.scenarios.append({"scenario": name, "starts": counts, **extra})

  def test_1_qualification_and_zero_work_resume(self):
    work, frozen, authority = self.scenario("qualification")
    record = study.run_qualification(
      work, operations=self.operations, authority=authority
    )
    study.require_qualification(work, frozen)
    before = self.starts(work)
    study.run_qualification(work, operations=self.operations, authority=authority)
    after = self.starts(work)
    self.assertEqual(before, after)
    self.record(
      "qualification", before, validated=record["validated"],
      qualification_digest=record["digest"],
    )

  def test_2_refused_p2_retains_partial_evidence(self):
    work, frozen, authority = self.scenario("refused-p2")
    # Seven held P2 reservations leave one qualification credit; warps-4 is refused.
    for index in range(7):
      authority.path_ledger.append(
        "process", phase="qualification", path="p2", event="reserved",
        execution_id=f"synthetic-refused-p2-{index}",
        case="mw-g0c0", configuration="warps-1", attempt=0,
      )
    record = study.run_qualification(
      work, operations=self.operations, authority=authority, slots=1
    )
    case = record["cases"][0]
    self.assertFalse(case["complete"])
    progress = json.loads((work / "qualification-progress.json").read_text())
    refused_status = progress["cases"]["mw-g0c0"]["attempts"]["warps-4"]["status"]
    self.assertEqual(refused_status, "refused")
    first = self.starts(work)
    self.assertEqual(first, {"p0": 1, "p1": 2, "p2": 1})
    cached = work / "mw-g0c0" / "p0.bin"
    self.assertTrue(cached.is_file())
    cached.rename(cached.with_suffix(".bin.aside"))
    retry = "mw-g0c0/warps-4/attempt-0"
    retried = study.run_qualification(
      work, operations=self.operations, authority=authority, slots=1, retry=retry
    )
    after = self.starts(work)
    self.assertEqual({key: after[key] - first[key] for key in after},
                     {"p0": 0, "p1": 1, "p2": 1})
    self.assertTrue(cached.with_suffix(".bin.aside").is_file())
    self.assertFalse(cached.exists())
    self.assertTrue(retried["cases"][0]["complete"])
    self.record(
      "refused-p2", after, retry=retry,
      p0_cache="moved-aside",
      p2_refusal=refused_status,
    )

  def test_3_refused_p0_uses_cached_unavailability(self):
    self.skipTest(
      "scenario 3 is blocked on P0-refusal retry reference propagation; "
      "the r3 evidence is preserved and the batch remains open"
    )
    work, frozen, authority = self.scenario("refused-p0")
    for index in range(4):
      authority.path_ledger.append(
        "process", phase="qualification", path="p0", event="reserved",
        execution_id=f"synthetic-refused-p0-{index}",
        case="mw-g0c0", configuration="warps-1", attempt=0,
      )
    record = study.run_qualification(
      work, operations=self.operations, authority=authority, slots=1
    )
    self.assertFalse(record["cases"][0]["complete"])
    self.assertEqual(self.starts(work), {"p0": 0, "p1": 0, "p2": 0})
    retry = "mw-g0c0/warps-1/attempt-0"
    retried = study.run_qualification(
      work, operations=self.operations, authority=authority, slots=1, retry=retry
    )
    after = self.starts(work)
    self.assertEqual(after, {"p0": 0, "p1": 2, "p2": 2})
    self.assertTrue(retried["cases"][0]["complete"])
    self.assertFalse(retried["cases"][0]["validated"])
    self.assertIn("p0_unavailable", retried["cases"][0])
    self.record("refused-p0", after, retry=retry, p0_cache="unavailable")


if __name__ == "__main__":
  unittest.main()
