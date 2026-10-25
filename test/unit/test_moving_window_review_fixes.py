"""Acceptance contracts for the prerequisite qualification repairs."""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test.lib import tensor_execution
from test.unit import test_moving_window_closure as closure
from test.unit.test_moving_window_repairs import (
  QualificationFixture as BaseFixture,
)
from test.unit.test_moving_window_repairs import (
  fixtures,
  read,
  study,
  with_capabilities,
)


class QualificationFixture(BaseFixture):
  def build(self, request):
    return with_capabilities(super().build(request))


class CandidateFixture(QualificationFixture):
  """Inject one completed candidate, optionally followed by a dispatch refusal."""

  def __init__(self, root, *, refused=False, candidate=True):
    super().__init__(root)
    execute = self.operations["execute"]

    def once(request):
      first = not self.calls
      observed = []
      reference_callback = request.get("on_reference")

      def retain(value):
        observed.append(value)
        if reference_callback:
          reference_callback(value)

      def account(event):
        if first and refused and event["path"] == "p2" and event["event"] == "reserved":
          raise tensor_execution.DispatchRefused("fixture P2 refusal")
        request["on_process"](event)

      try:
        result = execute({**request, "on_process": account, "on_reference": retain})
      except tensor_execution.DispatchRefused as error:
        result = {
          "status": "dispatch_refused",
          "dispatch_refused": str(error),
          "paths": {
            "p0": {"status": "completed"},
            "p1": {"status": "completed"},
            "p2": {"status": "not_run"},
          },
          "reference": observed[-1],
          "compiler": {"status": "unavailable"},
        }
      if first:
        result["candidate"] = candidate
        if candidate:
          result["paths"]["p1"]["status"] = "runtime_failure"
      (Path(request["directory"]) / "target-run.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "reference"})
      )
      return result

    self.operations["execute"] = once


def authority(fixture):
  path = fixture.root / "study.json"
  frozen = (
    study.FrozenStudy.load(path)
    if path.exists()
    else study.freeze_study(toolchain=fixture.toolchain)
  )
  return study.Study(
    fixture.root, frozen, build=None, screen=None, execute=None
  ).prepare()


def resume_if_paused(fixture):
  runner = authority(fixture)
  if runner.is_paused():
    runner.record_triage_decision(
      {
        "action": "resume",
        "reason": "fixture triage",
        "disposition": {
          "classification": "unresolved",
          "explanation": "Retain the fixture candidate while testing recovery.",
          "evidence": [],
        },
      }
    )
  return runner


class QualificationOutcomeTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name)

  def test_candidate_stops_before_the_next_configuration(self):
    f = CandidateFixture(self.root)
    study.run_qualification(f.root, operations=f.operations)
    self.assertEqual(len(f.calls), 1)
    self.assertTrue(authority(f).is_paused())
    self.assertEqual(read(f.root / "study-paused.json")["phase"], "qualification")

  def test_refusal_stops_before_the_next_case(self):
    f = CandidateFixture(self.root, refused=True, candidate=False)
    study.run_qualification(f.root, operations=f.operations)
    self.assertEqual(len(f.calls), 1)
    self.assertTrue(authority(f).is_paused())
    self.assertFalse(
      read(f.root / f.case / "warps-1-run-0" / "target-run.json")["candidate"]
    )

  def test_final_report_keeps_qualification_candidate(self):
    f = CandidateFixture(self.root)
    f.run()
    runner = resume_if_paused(f)
    f.run()
    runner = authority(f)
    runner.build = fixtures.fake_builder(
      [s["id"] for s in runner.frozen.spec["sources"]]
    )
    runner.screen = fixtures.fake_screen()
    runner.select_sources()
    runner.freeze_corpus()
    saved = runner.finalize()
    self.assertEqual(len(saved["candidates"]), 1)
    self.assertEqual(saved["candidates"][0]["phase"], "qualification")
    self.assertEqual(saved, study.report_from_records(f.root))

  def test_retry_keeps_both_attempt_outcomes(self):
    f = CandidateFixture(self.root, refused=True)
    f.run()
    before = (f.root / f.case / "warps-1-run-0" / "target-run.json").read_bytes()
    resume_if_paused(f)
    f.run(retry=f"{f.case}/warps-1/attempt-0")
    events = [
      e
      for e in authority(f).path_ledger.events
      if e["kind"] == "qualification_outcome" and e["configuration"] == "warps-1"
    ]
    self.assertEqual([e["attempt"] for e in events], [0, 1])
    self.assertEqual([e["outcome"]["candidate"] for e in events], [True, False])
    self.assertEqual(
      (f.root / f.case / "warps-1-run-0" / "target-run.json").read_bytes(), before
    )

  def test_outcome_survives_interruption_before_progress_save(self):
    f = CandidateFixture(self.root)
    write = tensor_execution.write_atomic

    def interrupt(path, data):
      if Path(path).name == "qualification-progress.json":
        cases = json.loads(data)["cases"].values()
        if any(
          a["status"] == "completed" for c in cases for a in c["attempts"].values()
        ):
          raise KeyboardInterrupt()
      return write(path, data)

    with mock.patch.object(tensor_execution, "write_atomic", side_effect=interrupt):
      with self.assertRaises(KeyboardInterrupt):
        f.run()
    f.calls.clear()
    with self.assertRaises(study.StudyRefused):
      f.run()
    self.assertEqual(f.calls, [])
    outcomes = [
      e for e in authority(f).path_ledger.events if e["kind"] == "qualification_outcome"
    ]
    self.assertEqual(len(outcomes), 1)
    self.assertTrue(outcomes[0]["outcome"]["candidate"])

  def test_edge_stop_is_terminal_across_qualification_restarts(self):
    f = CandidateFixture(self.root)
    f.run()
    calls = len(f.calls)
    with self.assertRaises(study.StudyRefused):
      f.run(triage_decision="stop")
    self.assertTrue(authority(f).is_stopped())
    with self.assertRaises(study.StudyRefused):
      f.run()
    self.assertEqual(len(f.calls), calls)


class UnifiedImportTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.f = QualificationFixture(temporary.name)
    self.runner = authority(self.f)

  def local(self, *, finish=True):
    callback = self.runner.process_observer("triage", case="case", attempt=0)
    callback({"event": "reserved", "path": "p2"})
    if finish:
      callback({"event": "start", "path": "p2", "pid": 123})
      callback({"event": "finish", "path": "p2", "status": "completed"})
    return next(
      e["execution_id"]
      for e in self.runner.path_ledger.events
      if e["kind"] == "process"
    )

  def import_row(self, identity, **fields):
    row = {
      "execution_id": identity,
      "case": "case",
      "attempt": 0,
      "path": "p2",
      "allocation": "triage",
      "result": "completed",
      **fields,
    }
    path = self.f.root / "import.jsonl"
    path.write_text(json.dumps(row) + "\n")
    self.runner.triage_ledger = path
    self.runner.prepare()

  def test_same_execution_is_charged_once_after_import_and_restart(self):
    self.import_row(self.local())
    self.assertEqual(authority(self.f).reserve_spent("p2"), 1)

  def assert_conflicting_import_refused(self, **fields):
    key = self.local()
    with self.assertRaises(study.StudyRefused):
      self.import_row(key, **fields)
    self.assertFalse((self.f.root / "accepted-imports.json").exists())
    self.assertEqual(authority(self.f).reserve_spent("p2"), 1)

  def test_import_path_cannot_conflict_with_local_identity(self):
    self.assert_conflicting_import_refused(path="p1")

  def test_import_allocation_cannot_conflict_with_local_identity(self):
    self.assert_conflicting_import_refused(allocation="discovery")

  def test_import_outcome_cannot_conflict_with_local_completion(self):
    self.assert_conflicting_import_refused(result="failed")

  def test_imported_identity_cannot_be_dispatched_again(self):
    key = study._digest(
      {"phase": "triage", "case": "case", "attempt": 0, "path": "p2", "ordinal": 0}
    )
    self.import_row(key)
    with self.assertRaises(study.StudyRefused):
      authority(self.f).process_observer("triage", case="case", attempt=0)(
        {"event": "reserved", "path": "p2"}
      )
    self.assertEqual(authority(self.f).reserve_spent("p2"), 1)

  def test_edge_imported_spawn_failure_releases_only_its_hold(self):
    self.import_row(self.local(finish=False), result="spawn_failed")
    self.assertEqual(authority(self.f).reserve_spent("p2"), 0)
    self.runner.prepare()
    self.assertEqual(authority(self.f).reserve_spent("p2"), 0)


class ValidationBatchTests(unittest.TestCase):
  def setUp(self):
    temp = tempfile.TemporaryDirectory()
    self.addCleanup(temp.cleanup)
    self.root = Path(temp.name) / "validation"

  def run_batch(self, *, candidate=False, interrupted=False, limits=None):
    f = QualificationFixture(self.root)
    f.operations["screen"] = lambda *_: None
    execute = f.operations["execute"]

    def wrapped(request):
      if interrupted:
        f.fail_at = ("warps-1", "p1", "start")
      result = execute(request)
      if candidate and len(f.calls) == 1:
        result["candidate"] = True
        result["status"] = "compiled_path_candidate"
      return result

    f.operations["execute"] = wrapped
    cls = closure.BoundedDeviceValidation
    patches = [
      mock.patch.object(closure, "VALIDATION_DIR", str(self.root)),
      mock.patch.object(cls, "__unittest_skip__", False),
      mock.patch.object(study, "device_operations", return_value=f.operations),
      mock.patch.dict(os.environ, {"REFRACTIR_TENSOR_FREE_FORM_EXPORT": "fixture"}),
    ]
    if limits is not None:
      patches.append(mock.patch.object(cls, "TOTAL_CAPS", limits))
    from contextlib import ExitStack

    with ExitStack() as stack:
      for patch in patches:
        stack.enter_context(patch)
      try:
        result = unittest.TextTestRunner(stream=io.StringIO()).run(
          unittest.defaultTestLoader.loadTestsFromTestCase(cls)
        )
      except KeyboardInterrupt:
        result = None
    return f, result

  def actual(self):
    counts = {p: 0 for p in ("p0", "p1", "p2")}
    for path in self.root.rglob("path-ledger.jsonl"):
      for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get("kind") == "process" and event.get("event") == "start":
          counts[event["path"]] += 1
    return counts

  def test_candidate_stops_later_validation_scenarios(self):
    f, _ = self.run_batch(candidate=True)
    self.assertEqual(len(f.calls), 1)
    self.assertFalse((self.root / "refused-p2").exists())

  def test_failed_scenario_starts_remain_in_saved_totals(self):
    self.run_batch(candidate=True)
    record = read(self.root / "validation.json")
    self.assertEqual(record["total_starts"], self.actual())
    self.assertEqual(record["status"], "failed")

  def test_validation_limits_match_section_6_2(self):
    self.assertEqual(
      closure.BoundedDeviceValidation.TOTAL_CAPS, {"p2": 12, "p1": 13, "p0": 5}
    )

  def test_batch_limit_is_checked_before_spawn(self):
    self.run_batch(limits={"p2": 0, "p1": 13, "p0": 5})
    self.assertEqual(self.actual()["p2"], 0)

  def test_interruption_retains_running_scenario_and_starts(self):
    self.run_batch(interrupted=True)
    self.assertTrue(
      (self.root / "validation.json").is_file(),
      "interruption must persist the validation record",
    )
    record = read(self.root / "validation.json")
    self.assertEqual(record["status"], "interrupted")
    self.assertEqual(record["total_starts"], self.actual())
    self.assertEqual(record["scenarios"][0]["status"], "interrupted")

  def test_edge_complete_host_batch_matches_the_declared_counts(self):
    _, result = self.run_batch()
    self.assertTrue(result.wasSuccessful(), result.errors or result.failures)
    saved = read(self.root / "validation.json")
    self.assertEqual(saved["status"], "passed")
    self.assertEqual(saved["total_starts"], {"p2": 12, "p1": 13, "p0": 5})
    rows = [
      json.loads(line)
      for line in (self.root / "processes.jsonl").read_text().splitlines()
    ]
    self.assertEqual(sum(row.get("event") == "start" for row in rows), 30)
    self.assertEqual(len(saved["scenarios"]), 3)


class CapabilityWitnessTests(unittest.TestCase):
  def setUp(self):
    temp = tempfile.TemporaryDirectory()
    self.addCleanup(temp.cleanup)
    self.fixture = QualificationFixture(temp.name)
    self.built = self.fixture.build({"directory": str(Path(temp.name) / "build")})

  def check(self, name):
    return study.qualification_capabilities(self.built, [name])[name]

  def test_offset_without_non_dense_stride_is_not_strided(self):
    layout = self.built["layouts"]["%window_buffer"]
    layout["element_strides"] = [8, 1]
    path = Path(self.built["bundles"]["i0-base"]) / "case.json"
    data = read(path)
    data["inputs"] = [layout]
    path.write_text(json.dumps(data))
    self.assertFalse(self.check("strided_backing_layout"))

  def test_staged_layout_must_support_the_stride_claim(self):
    path = Path(self.built["bundles"]["i0-base"]) / "case.json"
    data = read(path)
    data["inputs"][0]["element_strides"] = [8, 1]
    path.write_text(json.dumps(data))
    self.assertFalse(self.check("strided_backing_layout"))

  def test_extrema_shape_path_must_start_at_the_window(self):
    witness = self.built["region"]["requirements"][0]
    witness["input"] = "%acc_x"
    witness["path"][0] = "%acc_x"
    self.assertFalse(self.check("extrema_shape_path"))

  def test_path_edges_must_exist_in_the_live_graph(self):
    self.built["region"]["live_dag"]["operands"][1] = ["%acc_x"]
    self.assertFalse(self.check("extrema_shape_path"))

  def test_broadcast_alone_cannot_replace_transpose_or_expand(self):
    region = self.built["region"]
    region["requirements"][0].update(
      path=["%tile", "%r", "%next_x"],
      operations=["rmin", "broadcast"],
      operation_ids=[0, 1],
    )
    region["live_dag"] = {
      "values": ["%tile", "%r", "%next_x"],
      "ops": ["rmin", "broadcast"],
      "operands": [["%tile"], ["%r"]],
      "roots": ["%next_x"],
    }
    self.assertFalse(self.check("extrema_shape_path"))

  def test_edge_separate_witnesses_cannot_establish_a_composed_path(self):
    self.assertTrue(self.check("extrema_shape_path"))
    region = self.built["region"]
    witness = region["requirements"][0]
    region["requirements"] = [
      {
        **witness,
        "path": ["%tile", "%r"],
        "operations": ["rmin"],
        "operation_ids": [0],
      },
      {
        **witness,
        "input": "%r",
        "path": ["%r", "%e", "%next_x"],
        "operations": ["expand", "broadcast"],
        "operation_ids": [1, 2],
      },
    ]
    self.assertFalse(self.check("extrema_shape_path"))

  def test_edge_older_qualification_declaration_cannot_open_production_gate(self):
    frozen = study.freeze_study(toolchain=self.fixture.toolchain)
    record = closure.valid_record(frozen)
    record["declaration"] = {**record["declaration"], "version": 1}
    record.pop("digest")
    record["digest"] = study._digest(record)
    (self.fixture.root / "qualification.json").write_text(json.dumps(record))
    with self.assertRaises(study.StudyRefused):
      study.require_qualification(self.fixture.root, frozen)


class ConstructionStopTests(unittest.TestCase):
  def setUp(self):
    temp = tempfile.TemporaryDirectory()
    self.addCleanup(temp.cleanup)
    self.fixture = QualificationFixture(temp.name)
    self.requests = []

  def refuse(self, case="mw-g0c0", *, admitted_without_capability=False):
    f = self.fixture
    build = f.operations["build"]

    def controlled(request):
      self.requests.append(request)
      if request["slot"]["id"] != case:
        return build(request)
      if admitted_without_capability:
        result = build(request)
        result["region"]["window"]["source_kind"] = "state"
        return result
      Path(request["directory"]).mkdir(parents=True)
      return {"kind": "refused", "reason": "required_path_missing"}

    f.operations["build"] = controlled

  def run_qualification(self):
    return study.run_qualification(
      self.fixture.root, operations=self.fixture.operations
    )

  def test_exhausted_first_case_dispatches_no_workers(self):
    self.refuse()
    self.run_qualification()
    self.assertEqual(self.fixture.calls, [])
    self.assertEqual(len(self.requests), 4)

  def test_exhaustion_after_a_completed_case_stops_later_cases(self):
    self.refuse("mw-g0c1")
    self.run_qualification()
    self.assertEqual(len(self.fixture.calls), 2)
    self.assertEqual({r["slot"]["id"] for r in self.requests}, {"mw-g0c0", "mw-g0c1"})

  def test_exhaustion_is_a_durable_qualification_pause(self):
    self.refuse()
    self.run_qualification()
    self.assertTrue(authority(self.fixture).is_paused())
    pause = read(self.fixture.root / "study-paused.json")
    self.assertEqual(pause["phase"], "qualification")
    self.assertEqual(pause["candidates"], [])

  def test_restart_does_not_advance_past_an_exhausted_case(self):
    self.refuse()
    self.run_qualification()
    before = (len(self.requests), len(self.fixture.calls))
    with self.assertRaises(study.StudyRefused):
      self.run_qualification()
    self.assertEqual((len(self.requests), len(self.fixture.calls)), before)

  def test_missing_capability_pauses_before_any_worker(self):
    self.refuse(admitted_without_capability=True)
    self.run_qualification()
    self.assertEqual(self.fixture.calls, [])
    self.assertTrue(authority(self.fixture).is_paused())

  def test_edge_fourth_construction_attempt_can_qualify(self):
    build = self.fixture.operations["build"]
    calls = []

    def last_attempt(request):
      calls.append(request)
      if request["attempt"] < 3:
        Path(request["directory"]).mkdir(parents=True)
        return {"kind": "refused", "reason": "required_path_missing"}
      return build(request)

    self.fixture.operations["build"] = last_attempt
    result = study.run_qualification(
      self.fixture.root, operations=self.fixture.operations
    )
    self.assertTrue(result["validated"])
    self.assertEqual(len(calls), 16)
    self.assertEqual(len(self.fixture.calls), 8)
    self.assertFalse(authority(self.fixture).is_paused())


if __name__ == "__main__":
  unittest.main()
