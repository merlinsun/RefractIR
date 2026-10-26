"""Qualification-history and repair-delivery contracts."""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test.lib import tensor_campaign_evidence, tensor_candidate_triage, tensor_execution
from test.lib.tensor_target_reference import COMPLETED, ReferenceObservation
from test.unit import test_moving_window_closure as closure
from test.unit.test_moving_window_review_fixes import (
  CandidateFixture,
  QualificationFixture,
  authority,
  study,
)

SOURCE_NAMES = (
  "test/lib/tensor_repair_validation.py",
  "test/tensor/campaigns/run_moving_window_discovery.py",
  "test/unit/test_moving_window_closure.py",
)


def read(path):
  return json.loads(Path(path).read_text())


def write(path, value):
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  Path(path).write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")


class QualificationHistoryTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name)
    self.f = CandidateFixture(self.root, refused=True)
    study.run_qualification(self.root, operations=self.f.operations)
    self.original = (
      self.root / self.f.case / "warps-1-run-0" / "target-run.json"
    ).read_bytes()

  def decision(self, classification="unresolved"):
    evidence = []
    if classification != "unresolved":
      note = self.root / "diagnosis.txt"
      note.write_text("Host fixture: the deliberately injected failure is resolved.\n")
      evidence = [
        {"path": note.name, "sha256": hashlib.sha256(note.read_bytes()).hexdigest()}
      ]
    return authority(self.f).record_triage_decision(
      {
        "action": "resume",
        "disposition": {
          "classification": classification,
          "explanation": "Keep the original outcome while testing qualification recovery.",
          "evidence": evidence,
        },
      }
    )

  def retry(self):
    return study.run_qualification(
      self.root,
      operations=self.f.operations,
      retry=f"{self.f.case}/warps-1/attempt-0",
    )

  def test_successful_diagnostic_retry_cannot_validate_an_open_candidate(self):
    self.decision()
    self.assertFalse(self.retry()["validated"])
    self.assertFalse(read(self.root / "qualification.json")["validated"])

  def test_unresolved_retry_runs_only_the_named_configuration(self):
    self.decision()
    before = len(self.f.calls)
    self.retry()
    self.assertEqual(len(self.f.calls) - before, 1)
    self.assertEqual(self.f.calls[-1]["configuration"], "warps-1")

  def test_unresolved_resume_without_retry_cannot_dispatch(self):
    self.decision()
    before = len(self.f.calls)
    with self.assertRaisesRegex(study.StudyRefused, "unresolved|paused"):
      study.run_qualification(self.root, operations=self.f.operations)
    self.assertEqual(len(self.f.calls), before)

  def test_production_gate_checks_history_even_with_a_green_summary(self):
    self.decision()
    self.retry()
    a = authority(self.f)
    write(self.root / "qualification.json", closure.valid_record(a.frozen))
    with self.assertRaisesRegex(study.StudyRefused, "unresolved"):
      study.require_qualification(self.root, a.frozen)

  def test_resolved_evidence_is_revalidated_after_the_pause_was_cleared(self):
    self.decision("infrastructure_defect")
    self.assertTrue(self.retry()["validated"])
    (self.root / "diagnosis.txt").write_text("changed after triage\n")
    with self.assertRaisesRegex(study.StudyRefused, "digest|evidence"):
      study.require_qualification(self.root, authority(self.f).frozen)

  def test_edge_resolution_after_retry_closes_one_candidate_without_erasing_it(self):
    self.decision()
    self.retry()
    a = authority(self.f)
    self.assertTrue(a.is_paused())
    self.assertEqual(
      tensor_candidate_triage.summary(self.root, a.path_ledger.events)["unresolved"], 1
    )
    self.decision("infrastructure_defect")
    completed = study.run_qualification(self.root, operations=self.f.operations)
    self.assertTrue(completed["validated"])
    a = authority(self.f)
    self.assertTrue(study.require_qualification(self.root, a.frozen)["validated"])
    summary = tensor_candidate_triage.summary(self.root, a.path_ledger.events)
    self.assertEqual((summary["unresolved"], summary["resolved"]), (0, 1))
    self.assertEqual(
      (self.root / self.f.case / "warps-1-run-0/target-run.json").read_bytes(),
      self.original,
    )

  def test_edge_select_and_discover_check_open_history_before_workers(self):
    self.decision()
    self.retry()
    a = authority(self.f)
    write(self.root / "qualification.json", closure.valid_record(a.frozen))
    calls = len(self.f.calls)
    with mock.patch.object(study, "device_operations", return_value=self.f.operations):
      for mode in ("select", "discover"):
        with self.assertRaisesRegex(study.StudyRefused, "unresolved"):
          study.main(
            [
              "--mode",
              mode,
              "--work",
              str(self.root),
              "--exporter",
              "exporter",
              "--symirc",
              "symirc",
              "--symiri",
              "symiri",
            ]
          )
    self.assertEqual(len(self.f.calls), calls)


class RepairIdentityTests(unittest.TestCase):
  def setUp(self):
    from test.lib import tensor_repair_validation

    self.validation = tensor_repair_validation
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    self.source = self.root / "source"
    self.names = SOURCE_NAMES
    for name in self.names:
      path = self.source / name
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text("# controlled source bytes\n")
    self.patch = mock.patch.object(
      self.validation, "SOURCE_ROOT", self.source, create=True
    )
    self.patch.start()
    self.addCleanup(self.patch.stop)
    self.f = QualificationFixture(self.root / "batch")

  def run_batch(self, *, destination=None):
    return self.validation.run_batch(
      destination or self.f.root,
      self.f.operations,
      driver=study,
      caps=closure.BoundedDeviceValidation.CAPS,
      total_caps=closure.BoundedDeviceValidation.TOTAL_CAPS,
    )

  def test_saved_identity_binds_the_applied_python_sources(self):
    record = self.run_batch()
    self.assertIn("implementation", record)
    name = self.names[1]
    self.assertEqual(
      record["implementation"]["files"][name],
      hashlib.sha256((self.source / name).read_bytes()).hexdigest(),
    )

  def test_source_snapshot_exists_before_the_first_builder(self):
    build = self.f.operations["build"]

    def check(request):
      snapshot = self.f.root / "implementation" / self.names[1]
      self.assertTrue(snapshot.is_file(), "source bytes must precede the first build")
      self.assertEqual(
        snapshot.read_bytes(), (self.source / self.names[1]).read_bytes()
      )
      self.assertIn("implementation", read(self.f.root / "validation.json"))
      return build(request)

    self.f.operations["build"] = check
    self.run_batch()

  def test_changed_source_stops_before_the_next_target(self):
    execute = self.f.operations["execute"]

    def change(request):
      result = execute(request)
      (self.source / self.names[1]).write_text("# changed during the batch\n")
      return result

    self.f.operations["execute"] = change
    with self.assertRaisesRegex(ValueError, "implementation|source"):
      self.run_batch()
    self.assertEqual(len(self.f.calls), 1)
    self.assertEqual(read(self.f.root / "validation.json")["status"], "failed")

  def test_missing_runtime_source_refuses_before_any_builder(self):
    (self.source / self.names[0]).unlink()
    with self.assertRaisesRegex(ValueError, "source|implementation"):
      self.run_batch()
    self.assertEqual(self.f.builds, [])
    self.assertEqual(self.f.calls, [])

  def test_different_source_bytes_have_different_identities(self):
    first = self.run_batch()
    (self.source / self.names[1]).write_text("# a separately applied revision\n")
    second = self.run_batch(destination=self.root / "other-batch")
    self.assertIn("implementation", first)
    self.assertIn("implementation", second)
    self.assertNotEqual(
      first["implementation"]["digest"], second["implementation"]["digest"]
    )

  def test_edge_archived_sources_verify_after_original_sources_are_removed(self):
    record = self.run_batch()
    shutil.rmtree(self.source)
    self.validation.verify_implementation(
      self.f.root / "implementation", record["implementation"]
    )


class ArtifactFixture(QualificationFixture):
  """Host workers with real artifact bytes and recorded command shapes."""

  def execute(self, request):
    self.calls.append(request)
    directory = Path(request["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    bundle = Path(request["corpus"]["bundles"]["i0-base"])
    observed = request.get("reference")
    paths = {p: {"status": "not_run"} for p in ("p0", "p1", "p2")}
    target = {"status": "target_unavailable", "candidate": False}
    target["emission"] = {
      "kernel_sha256": hashlib.sha256(b"# fixture kernel\n").hexdigest()
    }
    refusal = None
    for path, mode in (("p0", None), ("p1", "interpret"), ("p2", "compiled")):
      if path == "p0" and observed is not None:
        paths[path] = {"status": observed.status, "cached": True}
        continue
      dest = directory / (mode or "reference")
      dest.mkdir(exist_ok=True)
      if path == "p0":
        output = Path(request["reference_output_path"])
        command = [
          "symiri",
          "--tensor-input",
          str(bundle / "inputs.bin"),
          "--tensor-output",
          str(output),
          str(bundle / "program.sir"),
        ]
      else:
        output = dest / "actual.bin"
        command = [
          "python",
          "worker",
          "--case",
          str(bundle),
          "--artifacts-dir",
          str(dest),
          "--mode",
          mode,
        ]

      def event(kind):
        request["on_process"](
          {
            "event": kind,
            "path": path,
            "pid": 123,
            "command": command,
            "status": "completed",
          }
        )

      try:
        event("reserved")
      except tensor_execution.DispatchRefused as error:
        refusal = str(error)
        if path == "p0":
          observed = ReferenceObservation("not_run", reason=refusal)
          request["on_reference"](observed)
        break
      event("start")
      output.write_bytes(b"ABCD")
      if path == "p0":
        observed = ReferenceObservation(COMPLETED, output=b"ABCD")
        request["on_reference"](observed)
      else:
        (dest / "kernel.py").write_bytes(b"# fixture kernel\n")
        (dest / "worker.log").write_text("")
        target["interpreter" if path == "p1" else "compiled"] = {
          "status": "completed",
          "actual_sha256": hashlib.sha256(b"ABCD").hexdigest(),
        }
      event("finish")
      paths[path] = {"status": "completed"}
    target["status"] = "dispatch_refused" if refusal else "target_agreement"
    target["dispatch_refused"] = refusal
    write(directory / "target-run.json", target)
    return {
      **target,
      "paths": paths,
      "reference": observed,
      "compiler": {"status": "unavailable"},
    }


class RepairDeliveryTests(unittest.TestCase):
  def setUp(self):
    from test.lib import tensor_repair_validation as validation

    temp = tempfile.TemporaryDirectory()
    self.addCleanup(temp.cleanup)
    self.root = Path(temp.name)
    self.work = self.root / "batch"
    self.delivered = self.root / "delivered"
    self.f = ArtifactFixture(self.work)
    self.source = self.root / "source"
    for name in SOURCE_NAMES:
      path = self.source / name
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text("# host fixture source\n")
    with mock.patch.object(validation, "SOURCE_ROOT", self.source, create=True):
      validation.run_batch(
        self.work,
        self.f.operations,
        driver=study,
        caps=closure.BoundedDeviceValidation.CAPS,
        total_caps=closure.BoundedDeviceValidation.TOTAL_CAPS,
      )

  def pack(self):
    return tensor_campaign_evidence.package(self.work, self.delivered)

  def test_package_retains_authoritative_ledgers_and_payloads(self):
    self.pack()
    for name in ("qualification", "refused-p2", "refused-p0"):
      self.assertTrue((self.delivered / name / "path-ledger.jsonl").is_file())
      self.assertTrue(list((self.delivered / name).rglob("actual.bin")))
    self.assertEqual(
      (self.delivered / "validation.json").read_bytes(),
      (self.work / "validation.json").read_bytes(),
    )

  def test_package_verifies_without_the_original_host_tree(self):
    self.pack()
    shutil.rmtree(self.work)
    relocated = self.root / "relocated"
    self.delivered.rename(relocated)
    result = tensor_campaign_evidence.verify(relocated)
    self.assertEqual(result["schema"], "refractir.moving-window-repair-evidence.v1")

  def test_missing_scenario_ledger_prevents_a_completion_manifest(self):
    (self.work / "refused-p2/path-ledger.jsonl").unlink()
    with self.assertRaisesRegex(ValueError, "ledger"):
      self.pack()
    self.assertFalse((self.delivered / "evidence-manifest.json").exists())

  def test_missing_completed_output_prevents_delivery(self):
    next((self.work / "qualification").rglob("actual.bin")).unlink()
    with self.assertRaisesRegex(ValueError, "actual.bin|artifact"):
      self.pack()
    self.assertFalse((self.delivered / "evidence-manifest.json").exists())

  def test_derived_index_cannot_replace_or_disagree_with_ledgers(self):
    index = self.work / "processes.jsonl"
    lines = index.read_text().splitlines()
    index.write_text("\n".join(lines[1:]) + "\n")
    with self.assertRaisesRegex(ValueError, "index|process"):
      self.pack()

  def test_edge_missing_decision_journal_cannot_be_replaced_by_progress(self):
    (self.work / "refused-p2/triage-decisions.jsonl").unlink()
    with self.assertRaisesRegex(ValueError, "triage-decisions"):
      self.pack()

  def test_edge_missing_source_identity_is_not_backfilled_by_packaging(self):
    record = read(self.work / "validation.json")
    record.pop("implementation")
    write(self.work / "validation.json", record)
    with self.assertRaisesRegex(ValueError, "implementation"):
      self.pack()
    self.assertNotIn("implementation", read(self.work / "validation.json"))

  def test_edge_missing_retired_outcome_cannot_be_replaced_by_a_green_retry(self):
    ledger = self.work / "refused-p2/path-ledger.jsonl"
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    events = [
      e
      for e in events
      if not (
        e.get("kind") == "qualification_outcome"
        and e.get("configuration") == "warps-4"
        and e.get("attempt") == 0
      )
    ]
    ledger.write_text("".join(json.dumps(e) + "\n" for e in events))
    with self.assertRaisesRegex(ValueError, "outcome"):
      self.pack()

  def test_edge_manifest_detects_changed_packaged_source_bytes(self):
    self.pack()
    (self.delivered / "implementation" / SOURCE_NAMES[1]).write_text(
      "# changed after delivery\n"
    )
    with self.assertRaisesRegex(ValueError, "digest"):
      tensor_campaign_evidence.verify(self.delivered)


if __name__ == "__main__":
  unittest.main()
