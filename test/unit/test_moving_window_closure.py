"""R4 qualification declaration, production gate, and closure device batch."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test.unit import test_moving_window_discovery as fixtures
from test.unit.test_moving_window_repairs import with_capabilities

study = fixtures.study

VALIDATION_DIR = os.environ.get("REFRACTIR_MOVING_WINDOW_VALIDATION_DIR")


def qualification_built(case="mw-g0c0"):
  """One admitted construction carrying every declared capability."""
  return with_capabilities(
    {
      "kind": "admitted",
      "case": case,
      "region": {},
    }
  )


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

  def test_qualification_construction_allows_the_study_multiplication_limit(self):
    self.assertEqual(study.QUALIFICATION.get("max_multiplications"), 1)

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

  def test_legacy_version_one_declarations_remain_readable(self):
    for multiplication in (0, 1):
      spec = study.study_spec()
      spec["qualification"] = {
        **study.QUALIFICATION, "version": 1,
        "max_multiplications": multiplication,
      }
      study._validate_spec(spec, allow_legacy_qualification=True)
      with self.assertRaises(Exception):
        study._validate_spec(spec)

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
  """§6.2: one fail-stop batch with checks before each process reservation."""

  CAPS = {
    "qualification": {"p2": 8, "p1": 8, "p0": 4},
    "refused-p2": {"p2": 2, "p1": 3, "p0": 1},
    "refused-p0": {"p2": 2, "p1": 2, "p0": 0},
  }
  TOTAL_CAPS = {"p2": 12, "p1": 13, "p0": 5}

  def test_bounded_batch(self):
    from test.lib.tensor_repair_validation import run_batch

    operations = study.device_operations(
      exporter=Path(os.environ["REFRACTIR_TENSOR_FREE_FORM_EXPORT"]),
      symirc=Path(os.environ.get("REFRACTIR_SYMIRC", "symirc")),
      symiri=Path(os.environ.get("REFRACTIR_SYMIRI", "symiri")),
    )
    result = run_batch(
      Path(VALIDATION_DIR),
      operations,
      driver=study,
      caps=self.CAPS,
      total_caps=self.TOTAL_CAPS,
    )
    self.assertEqual(result["status"], "passed")
    self.assertEqual(result["total_starts"], self.TOTAL_CAPS)


if __name__ == "__main__":
  unittest.main()
