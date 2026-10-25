"""Contracts for the exporter, candidate triage, and portable evidence boundary."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from test.lib import (
  native_triton_family,
  probe_triton_env,
  tensor_free_form_case,
  tensor_free_form_inputs,
)
from test.unit import test_moving_window_discovery as fixtures

study = fixtures.study


def write(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")


def digest(data):
  return hashlib.sha256(data).hexdigest()


class ExportLimitTests(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    self.commands = []
    self.limit = 16
    self.nodes = 5
    self.missing = False
    with (
      mock.patch.object(probe_triton_env, "collect_environment", return_value={}),
      mock.patch.object(native_triton_family, "triton_identity", return_value={}),
    ):
      self.operations = study.device_operations(
        exporter=self.root / "exporter",
        symirc=self.root / "symirc",
        symiri=self.root / "symiri",
      )
    self.request = {
      "slot": study.SOURCE_SLOTS[8],
      "seed": 20261087,
      "directory": str(self.root / "attempt"),
      "max_nodes": 16,
    }

  def exported(self, command, **unused):
    self.commands.append(command)
    export = Path(command[command.index("--out") + 1])
    record = {"structure": {"nodes": self.nodes}}
    if not self.missing:
      record["max_nodes"] = self.limit
    write(export / "region.json", record)
    (export / "program.sir").write_text("fun @kernel() {}\n^exit:\n")
    return SimpleNamespace(returncode=0, stdout="", stderr="")

  def build(self):
    with (
      mock.patch.object(study.subprocess, "run", side_effect=self.exported),
      mock.patch.object(
        tensor_free_form_inputs, "window_probes", return_value=[]
      ) as probes,
      mock.patch.object(tensor_free_form_case, "build_bundle"),
    ):
      # Stop after export validation without requiring input-generation metadata.
      probes.side_effect = RuntimeError("reached input construction")
      try:
        return self.operations["build"](self.request)
      except RuntimeError as error:
        if str(error) != "reached input construction":
          raise
        return {"kind": "validated-export"}

  def test_frozen_limit_reaches_real_export_command(self):
    self.build()
    self.assertIn("--max-nodes", self.commands[0])
    self.assertEqual(self.commands[0][self.commands[0].index("--max-nodes") + 1], "16")

  def test_legacy_default_is_explicit_at_export_boundary(self):
    del self.request["max_nodes"]
    self.limit = 12
    self.build()
    self.assertIn("--max-nodes", self.commands[0])
    self.assertEqual(self.commands[0][self.commands[0].index("--max-nodes") + 1], "12")

  def test_exporter_cannot_return_a_different_limit(self):
    self.limit = 12
    with self.assertRaisesRegex(study.StudyRefused, "node limit"):
      self.build()

  def test_live_node_count_cannot_exceed_the_limit(self):
    self.nodes = 17
    with self.assertRaisesRegex(study.StudyRefused, "node count"):
      self.build()

  def test_exported_limit_cannot_be_missing(self):
    self.missing = True
    with self.assertRaisesRegex(study.StudyRefused, "node limit"):
      self.build()

  def test_edge_legacy_declaration_loads_without_opening_dispatch(self):
    frozen = study.freeze_study()
    record = json.loads(json.dumps(frozen.as_record()))
    record["spec"]["qualification"]["version"] = 1
    record["digest"] = study._digest(record["spec"])
    path = self.root / "study.json"
    write(path, record)
    loaded = study.FrozenStudy.load(path)
    self.assertEqual(loaded.digest, record["digest"])
    with self.assertRaisesRegex(study.StudyRefused, "current qualification"):
      study.require_qualification(self.root, loaded)

  def test_edge_exact_limit_is_admitted(self):
    self.nodes = 16
    self.assertEqual(self.build()["kind"], "validated-export")


class TriageDispositionTests(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    self.run = study.Study(
      self.root,
      study.freeze_study(),
      build=fixtures.fake_builder([slot["id"] for slot in study.SOURCE_SLOTS]),
      screen=fixtures.fake_screen(),
      execute=fixtures.fake_execute()[0],
    ).prepare()
    self.run.pause(
      [{"case": "mw-g1c3", "status": "non_finite", "actual_sha256": "a" * 64}],
      reason="candidate",
      phase="selection",
    )
    self.run._save_pending(
      {"case": "mw-g1c3", "attempt": 0, "screening_complete": True}
    )
    self.proof = self.root / "triage" / "memo.md"
    self.proof.parent.mkdir()
    self.proof.write_text("A recorded, independently justified diagnosis.\n")

  def disposition(self, classification="invalid_execution"):
    return {
      "classification": classification,
      "explanation": "See the preserved diagnostic argument.",
      "evidence": [
        {"path": "triage/memo.md", "sha256": digest(self.proof.read_bytes())}
      ],
    }

  def test_rejecting_a_candidate_requires_a_disposition(self):
    with self.assertRaisesRegex(study.StudyRefused, "disposition"):
      self.run.record_triage_decision("reject")
    self.assertTrue(self.run.is_paused())

  def test_resolved_disposition_requires_evidence(self):
    value = self.disposition()
    value["evidence"] = []
    with self.assertRaisesRegex(study.StudyRefused, "evidence"):
      self.run.record_triage_decision({"action": "reject", "disposition": value})

  def test_changed_evidence_cannot_clear_the_pause(self):
    value = self.disposition()
    self.proof.write_text("changed")
    with self.assertRaisesRegex(study.StudyRefused, "digest"):
      self.run.record_triage_decision({"action": "reject", "disposition": value})
    self.assertTrue(self.run.is_paused())
    self.assertFalse((self.root / "triage-decisions.jsonl").exists())

  def test_saved_disposition_binds_the_candidate(self):
    pause = json.loads((self.root / "study-paused.json").read_text())
    value = self.run.record_triage_decision(
      {"action": "reject", "disposition": self.disposition()}
    )
    saved = json.loads((self.root / "triage-decisions.jsonl").read_text())
    self.assertEqual(saved.get("candidate_sha256"), study._digest(pause["candidates"]))
    self.assertEqual(saved["disposition"], value["disposition"])

  def test_explicit_unresolved_disposition_stays_open(self):
    value = self.disposition("unresolved")
    value["evidence"] = []
    self.run.record_triage_decision({"action": "reject", "disposition": value})
    self.run.select_sources()
    self.run.freeze_corpus()
    record = study.report_from_records(self.root)
    self.assertIn("candidate_triage", record)
    self.assertEqual(record["candidate_triage"]["unresolved"], 1)
    self.assertEqual(record["candidate_triage"]["resolved"], 0)

  def test_edge_recovery_revalidates_disposition_evidence(self):
    with mock.patch.object(self.run, "_clear_pending", side_effect=KeyboardInterrupt):
      with self.assertRaises(KeyboardInterrupt):
        self.run.record_triage_decision(
          {"action": "reject", "disposition": self.disposition()}
        )
    self.proof.write_text("changed after decision append")
    with self.assertRaisesRegex(study.StudyRefused, "digest"):
      self.run.prepare()
    self.assertTrue(self.run.is_paused())
    self.assertIsNotNone(self.run._pending_admission())

  def test_edge_resource_resume_needs_no_semantic_disposition(self):
    (self.root / "pending-admission.json").unlink(missing_ok=True)
    self.run._clear_pending()
    self.run.pause([], reason="disk", phase="resource")
    self.run.record_triage_decision("resume")
    self.assertFalse(self.run.is_paused())


def evidence_fixture(root):
  run = study.Study(
    root,
    study.freeze_study(),
    build=fixtures.fake_builder([slot["id"] for slot in study.SOURCE_SLOTS]),
    screen=fixtures.fake_screen(),
    execute=fixtures.fake_execute()[0],
  ).prepare()
  run.select_sources()
  run.freeze_corpus()
  bundle = root / "selection" / "rejected" / "bundle"
  source = b"fun @kernel() {}\n"
  data = b"ABCD"
  kernel = b"def kernel(): pass\n"
  for name, content in [("program.sir", source), ("inputs.bin", data)]:
    (bundle / name).parent.mkdir(parents=True, exist_ok=True)
    (bundle / name).write_bytes(content)
  write(bundle / "case.json", {"launch": {"num_warps": 1}})
  artifact = bundle / "target-runs"
  emission = {"kernel_sha256": digest(kernel)}
  modes = {}
  for path, mode in [("p1", "interpret"), ("p2", "compiled")]:
    dest = artifact / mode
    dest.mkdir(parents=True)
    (dest / "kernel.py").write_bytes(kernel)
    (dest / "actual.bin").write_bytes(data)
    (dest / "worker.log").write_text("")
    modes["interpreter" if path == "p1" else "compiled"] = {
      "status": "completed",
      "actual_sha256": digest(data),
    }
    for event in ["reserved", "start", "finish"]:
      run.path_ledger.append(
        "process",
        event=event,
        phase="discovery",
        path=path,
        case="rejected",
        slot="rejected/i0/warps-1",
        execution_id=path,
        status="completed",
        command=[
          "python",
          "worker",
          "--case",
          str(bundle),
          "--artifacts-dir",
          str(dest),
          "--mode",
          mode,
        ],
      )
  write(
    artifact / "target-run.json",
    {"emission": emission, **modes, "status": "target_agreement"},
  )
  ir = artifact / "compiled" / "compiler-ttir.ir"
  ir.write_text("module { }\n")
  run.path_ledger.append(
    "classification",
    slot="rejected/i0/warps-1",
    case="rejected",
    status="target_agreement",
    candidate=False,
    compiler={
      "status": "observed",
      "artifact_sha256": {"ttir": digest(ir.read_bytes())},
    },
  )
  return run, bundle, artifact


def inventory(root):
  files = []
  for path in sorted(root.rglob("*")):
    if path.is_file() and path.name != "evidence-manifest.json":
      data = path.read_bytes()
      files.append(
        {
          "path": path.relative_to(root).as_posix(),
          "sha256": digest(data),
          "size": len(data),
        }
      )
  study_digest = json.loads((root / "study.json").read_text())["digest"]
  write(
    root / "evidence-manifest.json",
    {
      "schema": "refractir.moving-window-evidence.v1",
      "study": study_digest,
      "origin": str(root.resolve()),
      "files": files,
    },
  )


class EvidenceArchiveTests(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name) / "work"
    self.run, self.bundle, self.artifacts = evidence_fixture(self.root)

  def check(self):
    inventory(self.root)
    return study.report_from_records(self.root)

  def test_summary_with_missing_kernel_is_not_raw_evidence(self):
    (self.artifacts / "compiled" / "kernel.py").unlink()
    with self.assertRaisesRegex(ValueError, "kernel.py"):
      self.check()

  def test_completed_output_must_be_delivered(self):
    (self.artifacts / "interpret" / "actual.bin").unlink()
    with self.assertRaisesRegex(ValueError, "actual.bin"):
      self.check()

  def test_output_must_match_the_saved_run_digest(self):
    (self.artifacts / "interpret" / "actual.bin").write_bytes(b"WRONG")
    with self.assertRaisesRegex(ValueError, "actual.bin"):
      self.check()

  def test_observed_compiler_stage_must_be_delivered(self):
    (self.artifacts / "compiled" / "compiler-ttir.ir").unlink()
    with self.assertRaisesRegex(ValueError, "compiler-ttir.ir"):
      self.check()

  def test_rejected_source_bundle_is_not_optional(self):
    (self.bundle / "program.sir").unlink()
    with self.assertRaisesRegex(ValueError, "program.sir"):
      self.check()

  def test_edge_claimed_agreement_is_checked_against_raw_outputs(self):
    target = self.artifacts / "compiled" / "actual.bin"
    target.write_bytes(b"DIFF")
    record_path = self.artifacts / "target-run.json"
    record = json.loads(record_path.read_text())
    record["compiled"]["actual_sha256"] = digest(target.read_bytes())
    write(record_path, record)
    with self.assertRaisesRegex(ValueError, "agreement"):
      self.check()

  def test_edge_external_controls_are_copied_with_explicit_mapping(self):
    from test.lib import tensor_campaign_evidence

    external = Path(self.temp.name) / "controls"
    external.mkdir()
    for name in ("program.sir", "inputs.bin", "case.json"):
      (self.bundle / name).replace(external / name)
    ledger = self.root / "path-ledger.jsonl"
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    for event in events:
      command = event.get("command")
      if command and "--case" in command:
        command[command.index("--case") + 1] = str(external)
    ledger.write_text("".join(json.dumps(event) + "\n" for event in events))
    with self.assertRaisesRegex(ValueError, "external"):
      tensor_campaign_evidence.package(self.root, Path(self.temp.name) / "missing")
    archive = Path(self.temp.name) / "delivered"
    tensor_campaign_evidence.package(self.root, archive, external_roots=[external])
    self.assertTrue((archive / "external/000/program.sir").is_file())
    self.assertEqual(
      study.report_from_records(archive), study.report_from_records(self.root)
    )

  def test_edge_pack_writes_manifest_only_after_verification(self):
    from test.lib import tensor_campaign_evidence

    archive = Path(self.temp.name) / "archive"
    expected = study.report_from_records(self.root)
    tensor_campaign_evidence.package(self.root, archive)
    self.assertEqual(study.report_from_records(archive), expected)
    with self.assertRaisesRegex(ValueError, "new"):
      tensor_campaign_evidence.package(self.root, archive)

  def test_edge_portable_archive_verifies_after_relocation(self):
    import shutil

    expected = self.check()
    copy = Path(self.temp.name) / "delivered"
    shutil.copytree(self.root, copy)
    self.assertEqual(study.report_from_records(copy), expected)
