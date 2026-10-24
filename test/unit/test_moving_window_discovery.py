"""Contract tests for the frozen moving-window study driver."""

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from test.lib import tensor_tranche

DRIVER_PATH = (
  Path(__file__).resolve().parents[2]
  / "test"
  / "tensor"
  / "campaigns"
  / "run_moving_window_discovery.py"
)
_spec = importlib.util.spec_from_file_location("moving_window_discovery", DRIVER_PATH)
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)


def fake_builder(admit=(), log=None, outcomes=None):
  """A builder that admits named slots and refuses the rest."""
  admitted = set(admit)
  outcomes = dict(outcomes or {})

  def build(request):
    if log is not None:
      log.append(request)
    case = request["slot"]["id"]
    if case in outcomes:
      return outcomes[case]
    if case not in admitted:
      return {"kind": "refused", "reason": "no body"}
    return {
      "kind": "admitted",
      "source_sha256": f"source-{case}",
      "inputs_sha256": f"inputs-{case}",
      "layouts": {"dense": True},
      "witnesses": [{"input": "%tile", "results": ["%next_x"]}],
      "compiler_identity": {"revision": "deadbeef"},
      "region": {"live_dag": {"connectivity_signature": f"sig-{case}"}},
    }

  return build


def fake_screen(selected_for=None, log=None):
  selected_for = None if selected_for is None else set(selected_for)

  def screen(outcome):
    case = next(
      (key for key in (outcome.get("case") or "",) if key), outcome.get("source_sha256")
    )
    if log is not None:
      log.append(outcome)
    picked = selected_for is None or case in selected_for
    return {
      "selected": picked,
      "reason": None if picked else "insensitive",
      "launches": 6,
      "p0": 4,
      "record": {"trip_count": 4},
    }

  return screen


def fake_execute(compiler_status="observed", candidate=False, cached_p0=False):
  calls = []

  def execute(request):
    calls.append(request)
    return {
      "status": "target_agreement",
      "paths": {
        "p0": {"status": "completed", "cached": cached_p0, "key": "p0-key"},
        "p1": {"status": "completed", "directory": request["directory"] + "/p1"},
        "p2": {"status": "completed", "directory": request["directory"] + "/p2"},
      },
      "candidate": candidate,
      "unavailable": [],
      "compiler": {"status": compiler_status},
    }

  return execute, calls


class FreezeTests(unittest.TestCase):
  def test_freezing_declares_lanes_slots_attempts_and_budgets(self):
    frozen = study.freeze_study()
    self.assertEqual(frozen.spec["lanes"], ["G0", "G1", "G2", "control"])
    self.assertEqual(len(frozen.spec["sources"]), 12)
    self.assertEqual(len(frozen.spec["controls"]), 4)
    self.assertEqual(len(frozen.tranche.slots), 128)
    self.assertEqual(frozen.spec["attempts"]["per_slot"], 8)
    self.assertEqual(frozen.spec["attempts"]["internal_max_attempts"], 1)
    self.assertEqual(frozen.spec["budgets"]["discovery_gpu"], 128)
    self.assertEqual(frozen.spec["budgets"]["reserve_gpu"], 32)
    self.assertEqual(len({slot.case_id for slot in frozen.tranche.slots}), 16)

  def test_a_changed_spec_is_refused_on_resume(self):
    frozen = study.freeze_study()
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "study.json"
      frozen.write(path)
      record = json.loads(path.read_text(encoding="utf-8"))
      record["spec"]["budgets"]["reserve_gpu"] = 99
      path.write_text(json.dumps(record), encoding="utf-8")
      with self.assertRaises(tensor_tranche.ManifestRefused):
        study.FrozenStudy.load(path)
      frozen.write(path)
      work = Path(directory) / "work"
      study.Study(work, frozen, build=fake_builder(), screen=fake_screen(),
                  execute=fake_execute()[0]).prepare()
      changed = study.FrozenStudy(
        spec=study.study_spec(), digest="other", tranche=frozen.tranche
      )
      with self.assertRaises(tensor_tranche.ManifestRefused):
        study.Study(work, changed, build=fake_builder(), screen=fake_screen(),
                    execute=fake_execute()[0]).prepare()

  def test_the_work_directory_refuses_another_study(self):
    with tempfile.TemporaryDirectory() as directory:
      work = Path(directory) / "work"
      frozen = study.freeze_study()
      study.Study(work, frozen, build=fake_builder(), screen=fake_screen(),
                  execute=fake_execute()[0]).prepare()
      other = study.freeze_study()
      other.digest = "different"
      with self.assertRaises(tensor_tranche.ManifestRefused):
        study.Study(work, other, build=fake_builder(), screen=fake_screen(),
                    execute=fake_execute()[0]).prepare()


class SelectionTests(unittest.TestCase):
  def test_each_slot_uses_at_most_eight_builds_with_one_internal_attempt(self):
    log = []
    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      run = study.Study(
        Path(directory) / "work", frozen, build=fake_builder(["mw-g0c0"], log),
        screen=fake_screen(), execute=fake_execute()[0],
      ).prepare()
      run.select_sources()
      by_case = {}
      for request in log:
        by_case.setdefault(request["slot"]["id"], []).append(request)
      self.assertEqual(len(by_case["mw-g0c0"]), 1)
      for case, requests in by_case.items():
        self.assertLessEqual(len(requests), 8)
        for request in requests:
          self.assertEqual(request["max_attempts"], 1)
      expected = sum(
        study.ATTEMPT_BASE + 8 * slot["index"] + attempt
        for slot in frozen.spec["sources"]
        for attempt in range(8)
        if slot["id"] != "mw-g0c0"
      )
      actual = sum(req["seed"] for req in log if req["slot"]["id"] != "mw-g0c0")
      self.assertEqual(actual, expected)
      self.assertEqual(len(run.corpus()), 1)
      self.assertEqual(len(run._shortfalls()), 11)

  def test_new_cases_are_screened_before_the_corpus_freezes(self):
    screens = []
    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      admitted = [slot["id"] for slot in frozen.spec["sources"]]
      run = study.Study(
        Path(directory) / "work", frozen, build=fake_builder(admitted),
        screen=fake_screen(log=screens), execute=fake_execute()[0],
      ).prepare()
      run.select_sources()
      self.assertEqual(len(run.corpus()), 12)
      self.assertEqual(len(screens), 12)
      before = len(run.frozen.spec["sources"])
      corpus = run.freeze_corpus()
      self.assertEqual(len(corpus["sources"]), before)
      self.assertEqual(len(corpus["controls"]), 4)
      self.assertEqual(corpus["shortfalls"], [])
      with self.assertRaises(study.StudyRefused):
        run.select_sources()

  def test_a_shortfall_is_reported_with_a_case_identity(self):
    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      run = study.Study(
        Path(directory) / "work", frozen, build=fake_builder(["mw-g0c0"]),
        screen=fake_screen(), execute=fake_execute()[0],
      ).prepare()
      run.select_sources()
      run.freeze_corpus()
      final = run._final_record(run.ledger.counts())
      shortfalls = [item for item in final["refusals"] if item["reason"] == "attempts_exhausted"]
      self.assertIn(
        {"slot": "mw-g0c1", "reason": "attempts_exhausted"}, shortfalls
      )

  def test_resume_keeps_the_corpus_and_does_not_rebuild_it(self):
    log = []
    with tempfile.TemporaryDirectory() as directory:
      work = Path(directory) / "work"
      frozen = study.freeze_study()
      first = study.Study(work, frozen, build=fake_builder(["mw-g0c0"], log),
                          screen=fake_screen(), execute=fake_execute()[0]).prepare()
      first.select_sources(max_slots=1)
      self.assertEqual(len(first.corpus()), 1)
      resumed_log = []
      resumed = study.Study(
        work, study.FrozenStudy.load(work / "study.json"),
        build=fake_builder(["mw-g0c1"], resumed_log), screen=fake_screen(),
        execute=fake_execute()[0],
      ).prepare()
      resumed.select_sources()
      self.assertEqual(len(resumed.corpus()), 2)
      rebuilt = [request["slot"]["id"] for request in resumed_log if request["slot"]["id"] == "mw-g0c0"]
      self.assertEqual(rebuilt, [])
      self.assertEqual(
        resumed.corpus()[0]["compiler_identity"], {"revision": "deadbeef"}
      )
      self.assertEqual(
        resumed.corpus()[0]["witnesses"], [{"input": "%tile", "results": ["%next_x"]}]
      )


class DiscoveryTests(unittest.TestCase):
  def _frozen_run(self, directory, **execute_kwargs):
    frozen = study.freeze_study()
    work = Path(directory) / "work"
    admitted = [slot["id"] for slot in frozen.spec["sources"]]
    run = study.Study(
      work, frozen, build=fake_builder(admitted), screen=fake_screen(),
      execute=fake_execute(**execute_kwargs)[0], legacy_accounting=True,
    ).prepare()
    run.select_sources()
    run.freeze_corpus()
    return run

  def test_paths_reconcile_and_reserve_stops_launches(self):
    with tempfile.TemporaryDirectory() as directory:
      run = self._frozen_run(directory, cached_p0=True)
      run.run_discovery()
      final = run.finalize()
      self.assertEqual(final["execution"]["states"]["completed"], 96)
      self.assertEqual(len(final["execution"]["unexecuted"]), 32)
      self.assertEqual(final["paths"]["p0"]["cached"], 96)
      self.assertEqual(final["paths"]["p0"]["launches"], 0)
      self.assertEqual(final["paths"]["p1"]["launches"], 96)
      self.assertEqual(final["paths"]["p2"]["launches"], 96)
      for _ in range(32):
        run.retry("mw-g0c0/i0-base/warps-1", case_id="mw-g0c0", input="i0-base",
                  configuration="warps-1", reason="repeat")
      self.assertEqual(run.reserve_spent(), 32)
      self.assertIsNone(
        run.retry("mw-g0c0/i0-base/warps-1", case_id="mw-g0c0", input="i0-base",
                  configuration="warps-1", reason="repeat")
      )
      self.assertTrue(
        any(event.get("kind") == "launch_refused" for event in run.path_ledger.events)
      )
      self.assertEqual(final["reserve"]["spent"], 0)

  def test_the_diversity_report_requires_three_bodies_and_three_signatures(self):
    with tempfile.TemporaryDirectory() as directory:
      run = self._frozen_run(directory)
      report = run.diversity_report()
      for lane in ("G0", "G1", "G2"):
        self.assertTrue(report[lane]["sufficient"], report)

  def test_discovery_refuses_below_the_diversity_minimum(self):
    with tempfile.TemporaryDirectory() as directory:
      run = self._frozen_run(directory)
      record = json.loads((run.work / "corpus-frozen.json").read_text(encoding="utf-8"))
      record["diversity"] = {
        "G0": {"bodies": 2, "distinct_signatures": 2, "sufficient": False},
        "G1": {"bodies": 4, "distinct_signatures": 4, "sufficient": True},
        "G2": {"bodies": 3, "distinct_signatures": 3, "sufficient": True},
        "sufficient": False,
      }
      payload = {key: record[key] for key in record if key != "digest"}
      record["digest"] = study._digest(payload)
      (run.work / "corpus-frozen.json").write_text(json.dumps(record), encoding="utf-8")
      with self.assertRaises(study.StudyRefused):
        run.run_discovery()

  def test_a_missing_compiler_witness_cannot_suppress_a_candidate(self):
    calls = {"count": 0}

    def execute(request):
      calls["count"] += 1
      return {
        "status": "target_agreement",
        "paths": {
          "p0": {"status": "completed", "cached": False},
          "p1": {"status": "completed"},
          "p2": {"status": "completed"},
        },
        "candidate": calls["count"] == 1,
        "unavailable": [],
        "compiler": {"status": "unavailable"},
      }

    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      admitted = [slot["id"] for slot in frozen.spec["sources"]]
      run = study.Study(
        Path(directory) / "work", frozen, build=fake_builder(admitted),
        screen=fake_screen(), execute=execute,
      ).prepare()
      run.select_sources()
      run.freeze_corpus()
      run.run_discovery()
      self.assertTrue(run.is_paused())
      self.assertGreater(run.ledger.counts()["pending"], 0)
      with self.assertRaises(study.StudyRefused):
        run.finalize()
      run.record_triage_decision("resume")
      run.run_discovery()
      final = run.finalize()
      self.assertGreater(len(final["candidates"]), 0)
      self.assertGreater(final["compiler"]["unavailable"], 0)
      self.assertTrue(final["corpus"], "the final record keeps the admitted corpus")
      self.assertTrue(final["observations"], "the final record keeps its observations")


  def test_new_runs_derive_launches_from_process_starts(self):
    def execute(request):
      for path in ("p1", "p2"):
        request["on_process"]({"event": "reserved", "mode": "x", "path": path})
        request["on_process"]({"event": "start", "mode": "x", "path": path, "pid": 1})
        request["on_process"]({"event": "finish", "mode": "x", "path": path, "status": "completed"})
      return {
        "status": "completed",
        "paths": {
          "p0": {"status": "unavailable"},
          "p1": {"status": "completed"},
          "p2": {"status": "completed"},
        },
        "candidate": False, "unavailable": [], "compiler": {"status": "unavailable"},
      }

    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      admitted = [slot["id"] for slot in frozen.spec["sources"]]
      run = study.Study(
        Path(directory) / "work", frozen, build=fake_builder(admitted),
        screen=fake_screen(), execute=execute,
      ).prepare()
      run.select_sources()
      run.freeze_corpus()
      run.run_discovery()
      final = run.finalize()
      self.assertEqual(final["paths"]["p0"]["launches"], 0)
      self.assertEqual(final["paths"]["p0"]["not_run"], 96)
      self.assertEqual(final["paths"]["p1"]["launches"], 96)
      self.assertEqual(final["paths"]["p2"]["launches"], 96)

  def test_a_pending_admission_cannot_be_resumed_without_a_disposition(self):
    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      run = study.Study(
        Path(directory) / "work", frozen,
        build=fake_builder(["mw-g0c0"]),
        screen=DiscoveryTests().candidate_screen if hasattr(DiscoveryTests(), "candidate_screen") else fake_screen(),
        execute=fake_execute()[0],
      ).prepare()
      # Use the candidate screen explicitly.
      run.screen = TriageAndAccountingTests().candidate_screen
      run.select_sources()
      self.assertTrue(run.is_paused())
      with self.assertRaises(study.StudyRefused):
        run.record_triage_decision("resume")
      self.assertTrue(run.is_paused())
      self.assertIsNotNone(run._pending_admission())

  def test_an_unapplied_admit_is_replayed_once_after_restart(self):
    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      work = Path(directory) / "work"
      run = study.Study(
        work, frozen, build=fake_builder(["mw-g0c0"]),
        screen=TriageAndAccountingTests().candidate_screen,
        execute=fake_execute()[0],
      ).prepare()
      run.select_sources()
      pending = run._pending_admission()
      pause = json.loads((work / "study-paused.json").read_text(encoding="utf-8"))
      decision = {
        "action": "admit", "case": pending["case"], "attempt": pending["attempt"],
        "pause_id": pause["pause_id"], "phase": pause["phase"],
      }
      decision["decision_id"] = hashlib.sha256(
        json.dumps(decision, sort_keys=True).encode("utf-8")
      ).hexdigest()[:16]
      path = work / "triage-decisions.jsonl"
      path.write_text(json.dumps(decision) + "\n", encoding="utf-8")
      # The pause and pending admission remain; a new process replays the decision.
      resumed = study.Study(
        work, study.FrozenStudy.load(work / "study.json"),
        build=fake_builder(["mw-g0c0"]), screen=fake_screen(),
        execute=fake_execute()[0],
      ).prepare()
      self.assertFalse(resumed.is_paused())
      self.assertEqual([entry["case"] for entry in resumed.corpus()], ["mw-g0c0"])
      self.assertIsNone(resumed._pending_admission())
      again = study.Study(
        work, study.FrozenStudy.load(work / "study.json"),
        build=fake_builder(["mw-g0c0"]), screen=fake_screen(),
        execute=fake_execute()[0],
      ).prepare()
      self.assertEqual(len(again.corpus()), 1)


  def test_qualification_persists_events_and_skips_completed_configurations(self):
    calls = []

    def build(request):
      return {
        "kind": "admitted", "source_sha256": "s", "inputs_sha256": "i",
        "layouts": {}, "witnesses": [], "region": {}, "source": "sir",
        "trip_variant": "sir", "probe_values": {}, "bundles": {},
        "compiler_identity": {},
      }

    def execute(request):
      request["on_process"]({"event": "reserved", "mode": "x"})
      request["on_process"]({"event": "start", "mode": "x", "pid": 1})
      request["on_process"]({"event": "finish", "mode": "x", "status": "completed"})
      calls.append(request["configuration"])
      return {
        "status": "completed",
        "paths": {
          "p0": {"status": "completed", "cached": False},
          "p1": {"status": "completed"},
          "p2": {"status": "completed"},
        },
        "candidate": False, "compiler": {"status": "unavailable"},
      }

    with tempfile.TemporaryDirectory() as directory:
      work = Path(directory) / "qualification"
      record = study.run_qualification(
        work, operations={"build": build, "execute": execute}
      )
      self.assertEqual(len(record["cases"]), 4)
      self.assertEqual(len(calls), 8)
      lines = (work / "qualification-processes.jsonl").read_text().splitlines()
      self.assertTrue(any('"event": "start"' in line for line in lines))
      calls.clear()
      study.run_qualification(work, operations={"build": build, "execute": execute})
      self.assertEqual(calls, [])

  def test_an_interrupted_discovery_stop_is_replayed(self):
    def execute(request):
      return {
        "status": "completed",
        "paths": {
          "p0": {"status": "completed", "cached": False},
          "p1": {"status": "completed"},
          "p2": {"status": "completed"},
        },
        "candidate": True,
        "unavailable": [], "compiler": {"status": "unavailable"},
      }

    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      work = Path(directory) / "work"
      run = study.Study(
        work, frozen,
        build=fake_builder([slot["id"] for slot in frozen.spec["sources"]]),
        screen=fake_screen(), execute=execute,
      ).prepare()
      run.select_sources()
      run.freeze_corpus()
      run.run_discovery()
      self.assertTrue(run.is_paused())
      pause = json.loads((work / "study-paused.json").read_text(encoding="utf-8"))
      decision = {
        "action": "stop", "pause_id": pause["pause_id"], "phase": pause["phase"],
      }
      decision["decision_id"] = hashlib.sha256(
        json.dumps(decision, sort_keys=True).encode("utf-8")
      ).hexdigest()[:16]
      with (work / "triage-decisions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(decision) + "\n")
      resumed = study.Study(
        work, study.FrozenStudy.load(work / "study.json"),
        build=fake_builder(), screen=fake_screen(), execute=execute,
      ).prepare()
      self.assertTrue(resumed.is_stopped())
      self.assertFalse(resumed.is_paused())
      with self.assertRaises(study.StudyRefused):
        resumed.run_discovery()

  def test_an_accepted_triage_import_survives_without_the_option(self):
    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      work = Path(directory) / "work"
      ledger = Path(directory) / "triage-ledger.jsonl"
      ledger.write_text(triage_rows(10), encoding="utf-8")
      study.Study(
        work, frozen, build=fake_builder(), screen=fake_screen(),
        execute=fake_execute()[0], triage_ledger=ledger,
      ).prepare()
      resumed = study.Study(
        work, frozen, build=fake_builder(), screen=fake_screen(),
        execute=fake_execute()[0],
      ).prepare()
      self.assertEqual(resumed.triage_charge(), 10)
      self.assertEqual(resumed.triage_charge("p2"), 10)
      self.assertEqual(resumed.triage_charge("p0"), 0)
      self.assertEqual(resumed.triage_charge("p1"), 0)
      with self.assertRaises(study.StudyRefused):
        study.Study(
          work, frozen, build=fake_builder(), screen=fake_screen(),
          execute=fake_execute()[0],
          triage_ledger=Path(directory) / "missing.jsonl",
        ).prepare()

  def test_changed_accepted_import_bytes_are_refused(self):
    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      work = Path(directory) / "work"
      ledger = Path(directory) / "triage-ledger.jsonl"
      ledger.write_text(triage_rows(1), encoding="utf-8")
      study.Study(
        work, frozen, build=fake_builder(), screen=fake_screen(),
        execute=fake_execute()[0], triage_ledger=ledger,
      ).prepare()
      ledger.write_text(triage_rows(2), encoding="utf-8")
      with self.assertRaises(study.StudyRefused):
        study.Study(
          work, frozen, build=fake_builder(), screen=fake_screen(),
          execute=fake_execute()[0], triage_ledger=ledger,
        ).prepare()

  def test_a_screening_candidate_appears_in_the_final_report(self):
    with tempfile.TemporaryDirectory() as directory:
      run = self._frozen_run(directory)
      run.path_ledger.append(
        "selection_candidates", case="mw-g0c0",
        candidates=[{"kind": "unused_rows_changed", "probe": "i0-base vs i3-unused-rows"}],
      )
      run.run_discovery()
      final = run.finalize()
      phases = {candidate["phase"] for candidate in final["candidates"]}
      self.assertIn("screening", phases)
      self.assertTrue(
        any(candidate.get("case") == "mw-g0c0" for candidate in final["candidates"])
      )

  def test_report_from_records_reproduces_the_final_summary(self):
    with tempfile.TemporaryDirectory() as directory:
      run = self._frozen_run(directory)
      run.run_discovery()
      final = run.finalize()
      reproduced = study.report_from_records(run.work)
      self.assertEqual(reproduced, final)

  def test_an_interrupted_claim_is_reconciled_before_retry(self):
    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      work = Path(directory) / "work"
      first = study.Study(work, frozen, build=fake_builder(), screen=fake_screen(),
                          execute=fake_execute()[0]).prepare()
      claim = first.ledger.claim("mw-g0c0/i0-base/warps-1")
      self.assertEqual(claim.slot_id, "mw-g0c0/i0-base/warps-1")
      resumed = study.Study(
        work, study.FrozenStudy.load(work / "study.json"),
        build=fake_builder(), screen=fake_screen(), execute=fake_execute()[0],
      ).prepare()
      self.assertEqual(resumed.ledger.state("mw-g0c0/i0-base/warps-1"), "failed")
      reconciled = [e for e in resumed.path_ledger.events if e.get("kind") == "reconciled"]
      self.assertEqual(len(reconciled), 1)
      resumed.path_ledger.append(
        "slot_started",
        slot="mw-g0c0/i0-base/warps-1",
        case="mw-g0c0",
        attempt=0,
        directory=str(resumed.work / "interrupted-attempt"),
      )
      directory_path = resumed.retry(
        "mw-g0c0/i0-base/warps-1", case_id="mw-g0c0", input="i0-base",
        configuration="warps-1", reason="interrupted",
      )
      self.assertIsNotNone(directory_path)
      self.assertTrue(str(directory_path).endswith("attempt-1"))
      self.assertFalse((resumed.work / "discovery" / "mw-g0c0" / "i0-base" / "warps-1" / "attempt-0").exists())
      self.assertEqual(resumed.reserve_spent(), 1)


class QualificationTests(unittest.TestCase):
  def test_qualification_runs_four_cases_under_two_warps_and_reuses_p0(self):
    calls = []
    built = {
      "kind": "admitted",
      "source_sha256": "source",
      "inputs_sha256": "inputs",
      "layouts": {},
      "witnesses": [],
      "region": {"shape": [2, 8]},
      "source": "fun @kernel() {}",
      "probe_values": {"i0-base": []},
    }

    def build(request):
      return built

    def execute(request):
      calls.append(request)
      return {
        "status": "target_agreement",
        "paths": {
          "p0": {"status": "completed", "cached": request.get("reference") is not None},
          "p1": {"status": "completed"},
          "p2": {"status": "completed"},
        },
        "candidate": False,
        "compiler": {"status": "observed"},
        "reference": {"kind": "ref"},
      }

    with tempfile.TemporaryDirectory() as directory:
      record = study.run_qualification(
        Path(directory) / "qualification",
        operations={"build": build, "execute": execute},
      )
      self.assertEqual(len(record["cases"]), 4)
      self.assertEqual(len(calls), 8)
      for case in record["cases"]:
        self.assertEqual(sorted(case["configurations"]), ["warps-1", "warps-4"])
        self.assertFalse(case["configurations"]["warps-1"]["paths"]["p0"]["cached"])
        self.assertTrue(case["configurations"]["warps-4"]["paths"]["p0"]["cached"])
      self.assertTrue((Path(directory) / "qualification" / "qualification.json").is_file())


  def test_an_interrupted_qualification_needs_an_explicit_retry(self):
    calls = []

    def build(request):
      return built

    def failing(request):
      calls.append(request["configuration"])
      request["on_process"]({"event": "start", "mode": "x", "pid": 1})
      raise RuntimeError("interrupted")

    def execute(request):
      calls.append(request["configuration"])
      return {
        "status": "completed",
        "paths": {
          "p0": {"status": "completed", "cached": request.get("reference") is not None},
          "p1": {"status": "completed"}, "p2": {"status": "completed"},
        },
        "candidate": False, "compiler": {"status": "unavailable"},
        "reference": {"kind": "ref"},
      }

    built = {
      "kind": "admitted", "source_sha256": "s", "inputs_sha256": "i",
      "layouts": {}, "witnesses": [], "region": {}, "source": "sir",
      "trip_variant": "sir", "probe_values": {}, "bundles": {}, "compiler_identity": {},
    }
    with tempfile.TemporaryDirectory() as directory:
      work = Path(directory) / "qualification"
      with self.assertRaises(RuntimeError):
        study.run_qualification(
          work, slots=1, operations={"build": build, "execute": failing}
        )
      calls.clear()
      record = study.run_qualification(
        work, slots=1, operations={"build": build, "execute": execute}
      )
      self.assertEqual(calls, ["warps-4"])
      self.assertTrue(record["cases"][0].get("interrupted"))
      calls.clear()
      study.run_qualification(
        work, slots=1, operations={"build": build, "execute": execute},
        retry=f"{record['cases'][0]['case']}/warps-1",
      )
      self.assertEqual(calls, ["warps-1"])


class EvidenceTests(unittest.TestCase):
  def test_screening_candidates_pause_and_stop_selection(self):
    triggered = []

    def candidate_screen(outcome):
      triggered.append(outcome["source_sha256"])
      return {
        "selected": True,
        "reason": "selected-but-candidate",
        "launches": 6,
        "p0": 4,
        "candidates": [{"kind": "p0_p1_disagreement", "probe": "i0-base"}],
        "record": {},
      }

    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      work = Path(directory) / "work"
      run = study.Study(
        work, frozen, build=fake_builder(["mw-g0c0"]), screen=candidate_screen,
        execute=fake_execute()[0],
      ).prepare()
      run.select_sources()
      self.assertTrue(run.is_paused())
      self.assertIn("screening candidates", run._paused_reason())
      self.assertEqual(len(triggered), 1)
      self.assertEqual(run.corpus(), [])
      with self.assertRaises(study.StudyRefused):
        run.run_discovery()

  def test_an_interrupted_selection_attempt_consumes_its_ordinal(self):
    resumed_log = []
    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      work = Path(directory) / "work"
      first = study.Study(
        work, frozen, build=fake_builder(["mw-g0c0"]), screen=fake_screen(),
        execute=fake_execute()[0],
      ).prepare()
      first.select_sources(max_slots=1)
      self.assertEqual(len(first.corpus()), 1)
      # Simulate a crash after reserving attempt 0 for the next slot.
      slot = frozen.spec["sources"][1]
      first.path_ledger.append(
        "selection_attempt_reserved", case=slot["id"], attempt=0,
        seed=123, directory=str(first.work / "selection" / slot["id"] / "attempt-0"),
      )
      (first.work / "selection" / slot["id"] / "attempt-0").mkdir(parents=True)
      resumed = study.Study(
        work, study.FrozenStudy.load(work / "study.json"),
        build=fake_builder([slot["id"]], resumed_log), screen=fake_screen(),
        execute=fake_execute()[0],
      ).prepare()
      resumed.select_sources()
      ordinals = [
        event["attempt"] for event in resumed.path_ledger.events
        if event.get("case") == slot["id"] and event.get("kind") == "selection_attempt"
      ]
      self.assertEqual(ordinals, [0, 1])
      self.assertEqual(resumed_log[0]["attempt"], 1)
      self.assertLessEqual(len(ordinals), 8)
      self.assertEqual(len(resumed.corpus()), 2)

  def test_successive_retries_get_distinct_directories(self):
    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      run = study.Study(
        Path(directory) / "work", frozen, build=fake_builder(), screen=fake_screen(),
        execute=fake_execute()[0],
      ).prepare()
      first = run.retry("s", case_id="mw-g0c0", input="i0-base",
                        configuration="warps-1", reason="first")
      second = run.retry("s", case_id="mw-g0c0", input="i0-base",
                         configuration="warps-1", reason="second")
      self.assertIsNotNone(first)
      self.assertIsNotNone(second)
      self.assertNotEqual(first, second)
      self.assertTrue(first.is_dir() and second.is_dir())

  def test_the_report_reconstructs_the_corpus_without_selection_history(self):
    with tempfile.TemporaryDirectory() as directory:
      run = DiscoveryTests()._frozen_run(directory) if False else None
      frozen = study.freeze_study()
      work = Path(directory) / "work"
      run = study.Study(
        work, frozen, build=fake_builder([slot["id"] for slot in frozen.spec["sources"]]),
        screen=fake_screen(), execute=fake_execute()[0], legacy_accounting=True,
      ).prepare()
      run.select_sources()
      run.freeze_corpus()
      run.run_discovery()
      final = run.finalize()
      packaged = Path(directory) / "packaged"
      packaged.mkdir()
      for name in ("study.json", "corpus-frozen.json", "path-ledger.jsonl",
                   "tranche-ledger.jsonl", "final.json"):
        (packaged / name).write_bytes((work / name).read_bytes())
      reproduced = study.report_from_records(packaged)
      self.assertEqual(reproduced, final)

  def test_a_conflicting_selection_history_is_refused(self):
    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      work = Path(directory) / "work"
      run = study.Study(
        work, frozen, build=fake_builder(["mw-g0c0"]), screen=fake_screen(),
        execute=fake_execute()[0],
      ).prepare()
      run.select_sources()
      run.freeze_corpus()
      entry = run.corpus()[0]
      for field, value in (
        ("source_sha256", "contradiction"),
        ("inputs_sha256", "contradiction"),
        ("layouts", {"bad": True}),
        ("witnesses", [{"bad": True}]),
      ):
        mutated = dict(entry)
        mutated[field] = value
        run.corpus_path.write_text(json.dumps(mutated) + "\n", encoding="utf-8")
        with self.assertRaises(study.StudyRefused):
          run.corpus_for_report()

  def test_toolchain_identity_is_enforced(self):
    frozen = study.freeze_study(toolchain={"symirc_sha256": "a"})
    study.require_toolchain(frozen, {"symirc_sha256": "a"})
    with self.assertRaises(study.StudyRefused):
      study.require_toolchain(frozen, {"symirc_sha256": "b"})
    historical = study.freeze_study()
    with self.assertRaises(study.StudyRefused):
      study.require_toolchain(historical, {"symirc_sha256": "a"})
    study.require_toolchain(historical, {"symirc_sha256": "a"}, accept_current=True)

  def test_compiler_observation_reports_stages_or_unavailable(self):
    with tempfile.TemporaryDirectory() as directory:
      compiled = Path(directory)
      missing = study.compiler_stage_observation(compiled)
      self.assertEqual(missing["status"], "unavailable")
      (compiled / "compiler-ttir.ir").write_text(
        "module {\n  scf.for %i = %c0 to %c1 step %c1 {\n    %v = tt.load %p : !tt.ptr<f32>\n  }\n}\n",
        encoding="utf-8",
      )
      (compiled / "compiler-ptx.ir").write_text(
        "%r1 = ld.global.b32 { %r2 }, [ %rd1 + 0 ];\n", encoding="utf-8"
      )
      observed = study.compiler_stage_observation(compiled)
      self.assertEqual(observed["status"], "observed")
      self.assertIn("ttir", observed["stages"])
      self.assertIn("ptx", observed["artifact_sha256"])
      self.assertEqual(observed["predicate_attribution"], "unavailable")


def triage_rows(count):
  return "\n".join(
    json.dumps({"case": f"c{index}", "result": "timed_out"})
    for index in range(count)
  ) + "\n"


DEVICE_TESTS = os.environ.get("REFRACTIR_DEVICE_TESTS") == "1"


class TriageAndAccountingTests(unittest.TestCase):
  def candidate_screen(self, outcome):
    return {
      "selected": True, "reason": "candidate", "launches": 6, "p0": 4,
      "candidates": [{"kind": "unused_rows_changed", "probe": "i0-base vs i3-unused-rows"}],
      "record": {},
    }

  def test_a_stop_survives_restart_and_blocks_dispatch(self):
    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      work = Path(directory) / "work"
      run = study.Study(
        work, frozen, build=fake_builder(["mw-g0c0"]),
        screen=self.candidate_screen, execute=fake_execute()[0],
      ).prepare()
      run.select_sources()
      self.assertTrue(run.is_paused())
      run.record_triage_decision("stop")
      self.assertTrue(run.is_stopped())
      self.assertFalse(run.is_paused())
      resumed = study.Study(
        work, study.FrozenStudy.load(work / "study.json"),
        build=fake_builder(["mw-g0c0"]), screen=fake_screen(),
        execute=fake_execute()[0],
      ).prepare()
      with self.assertRaises(study.StudyRefused):
        resumed.select_sources()
      with self.assertRaises(study.StudyRefused):
        resumed.run_discovery()

  def test_an_eighth_attempt_candidate_can_be_admitted_by_triage(self):
    def builder(request):
      if request["attempt"] < 7:
        return {"kind": "refused", "reason": "seed"}
      return {
        "kind": "admitted", "source_sha256": "s", "inputs_sha256": "i",
        "layouts": {}, "witnesses": [], "region": {}, "source": "sir",
        "trip_variant": "sir", "probe_values": {}, "bundles": {},
        "compiler_identity": {},
      }

    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      run = study.Study(
        Path(directory) / "work", frozen, build=builder,
        screen=self.candidate_screen, execute=fake_execute()[0],
      ).prepare()
      run.select_sources(max_slots=1)
      self.assertTrue(run.is_paused())
      self.assertIsNotNone(run._pending_admission())
      run.record_triage_decision("admit")
      self.assertEqual([entry["case"] for entry in run.corpus()], ["mw-g0c0"])
      self.assertNotIn(
        "mw-g0c0", [e.get("case") for e in run.path_ledger.events if e.get("kind") == "shortfall"]
      )
      attempts = [
        e["attempt"] for e in run.path_ledger.events
        if e.get("kind") == "selection_attempt" and e.get("case") == "mw-g0c0"
      ]
      self.assertEqual(attempts, list(range(8)))
      run.select_sources()
      self.assertEqual(len(run.corpus()), 1)

  def test_a_rejected_candidate_consumes_one_attempt(self):
    calls = {"count": 0}

    def builder(request):
      calls["count"] += 1
      return {
        "kind": "admitted", "source_sha256": "s", "inputs_sha256": "i",
        "layouts": {}, "witnesses": [], "region": {}, "source": "sir",
        "trip_variant": "sir", "probe_values": {}, "bundles": {},
        "compiler_identity": {},
      }

    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      run = study.Study(
        Path(directory) / "work", frozen, build=builder,
        screen=self.candidate_screen, execute=fake_execute()[0],
      ).prepare()
      run.select_sources(max_slots=1)
      run.record_triage_decision("reject")
      self.assertIsNone(run._pending_admission())
      run.select_sources(max_slots=1)
      attempts = [
        e["attempt"] for e in run.path_ledger.events
        if e.get("kind") == "selection_attempt" and e.get("case") == "mw-g0c0"
      ]
      self.assertEqual(attempts, [0, 1])

  def test_an_invalid_execution_is_not_a_report_candidate(self):
    with tempfile.TemporaryDirectory() as directory:
      run = DiscoveryTests()._frozen_run(directory)
      run.path_ledger.append(
        "selection_invalid_execution", case="mw-g0c0",
        candidates=[{"kind": "invalid_execution", "probe": "i0-base"}],
      )
      final = run.finalize()
      self.assertEqual(len(final["invalid_executions"]), 1)
      self.assertFalse(
        any(candidate.get("kind") == "invalid_execution" for candidate in final["candidates"])
      )

  def test_an_absent_archive_marks_every_control_missing(self):
    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      admitted = [slot["id"] for slot in frozen.spec["sources"]]
      run = study.Study(
        Path(directory) / "work", frozen, build=fake_builder(admitted),
        screen=fake_screen(), execute=fake_execute()[0],
      ).prepare()
      run.select_sources()
      run.freeze_corpus()
      controls = run.frozen_corpus()["controls"]
      self.assertTrue(all(control.get("status") == "missing" for control in controls))
      run.run_discovery()
      final = run.finalize()
      self.assertEqual(len(final["execution"]["unexecuted"]), 32)

  def test_low_disk_after_the_first_attempt_stops_selection(self):
    import unittest.mock as mock
    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      built = []
      run = study.Study(
        Path(directory) / "work", frozen,
        build=lambda request: built.append(request) or {
          "kind": "admitted", "source_sha256": "s", "inputs_sha256": "i",
          "layouts": {}, "witnesses": [], "region": {}, "source": "sir",
          "trip_variant": "sir", "probe_values": {}, "bundles": {},
          "compiler_identity": {},
        },
        screen=fake_screen(), execute=fake_execute()[0],
      ).prepare()
      usage = type("Usage", (), {"free": 8 * 1024**3, "total": 8 * 1024**3, "used": 0})()
      low = type("Usage", (), {"free": 0, "total": 8 * 1024**3, "used": 0})()
      with mock.patch.object(study.shutil, "disk_usage", side_effect=[usage, usage, low, low]):
        run.select_sources()
      self.assertEqual(len(built), 1)
      self.assertTrue(run.is_paused())

  def test_reconciliation_charges_the_triage_ledger_without_changing_the_final(self):
    with tempfile.TemporaryDirectory() as directory:
      run = DiscoveryTests()._frozen_run(directory)
      final = run.finalize()
      ledger = Path(directory) / "triage-ledger.jsonl"
      ledger.write_text(triage_rows(1), encoding="utf-8")
      plain = study.report_from_records(run.work)
      reconciled = study.report_from_records(run.work, triage_ledger=ledger)
      self.assertEqual(plain, final)
      self.assertEqual(reconciled["reserve"]["spent"], 1)
      self.assertEqual(reconciled["reserve"]["remaining"], 31)

  def test_a_path_allocation_is_exhausted_before_a_spawn(self):
    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      run = study.Study(
        Path(directory) / "work", frozen, build=fake_builder(),
        screen=fake_screen(), execute=fake_execute()[0],
      ).prepare()
      for _ in range(frozen.spec["budgets"]["discovery_p0"]):
        run.path_ledger.append(
          "process", phase="discovery", path="p0", event="reserved",
          slot="s", attempt=0,
        )
      with self.assertRaises(study.StudyRefused):
        run._reserve_process_budget(
          "discovery", {"event": "reserved", "path": "p0"}
        )
      self.assertTrue(
        any(e.get("kind") == "launch_refused" for e in run.path_ledger.events)
      )

  def test_the_triage_ledger_is_charged_to_the_executable_reserve(self):
    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      work = Path(directory) / "work"
      ledger = Path(directory) / "triage-ledger.jsonl"
      ledger.write_text(triage_rows(10), encoding="utf-8")
      run = study.Study(
        work, frozen, build=fake_builder(), screen=fake_screen(),
        execute=fake_execute()[0], triage_ledger=ledger,
      ).prepare()
      self.assertEqual(run.triage_charge(), 10)
      self.assertEqual(run.reserve_spent(), 10)
      for _ in range(22):
        run.path_ledger.append("retry", slot="s", case="c", attempt=0)
      self.assertEqual(run.reserve_spent(), 32)
      self.assertIsNone(
        run.retry("s", case_id="mw-g0c0", input="i0-base",
                  configuration="warps-1", reason="over")
      )
      self.assertEqual(run.triage_charge(), 10)

  def test_low_disk_during_discovery_stops_before_the_next_slot(self):
    import unittest.mock as mock
    with tempfile.TemporaryDirectory() as directory:
      run = DiscoveryTests()._frozen_run(directory)
      usage = type("Usage", (), {"free": 8 * 1024**3, "total": 8 * 1024**3, "used": 0})()
      low = type("Usage", (), {"free": 0, "total": 8 * 1024**3, "used": 0})()
      with mock.patch.object(
        study.shutil, "disk_usage", side_effect=[usage, usage, low, low]
      ):
        run.run_discovery()
      self.assertTrue(run.is_paused())
      self.assertEqual(run.ledger.counts()["completed"], 1)
      self.assertGreater(run.ledger.counts()["pending"], 0)

  def test_discovery_wires_process_callbacks_into_the_request(self):
    seen = []

    def execute(request):
      request["on_process"]({"event": "reserved", "mode": "reference"})
      request["on_process"]({"event": "start", "mode": "reference", "pid": 1})
      request["on_process"]({"event": "finish", "mode": "reference", "status": "completed"})
      seen.append(True)
      return {
        "status": "completed",
        "paths": {
          "p0": {"status": "completed", "cached": False},
          "p1": {"status": "completed"},
          "p2": {"status": "completed"},
        },
        "candidate": False, "unavailable": [], "compiler": {"status": "unavailable"},
      }

    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      admitted = [slot["id"] for slot in frozen.spec["sources"]]
      run = study.Study(
        Path(directory) / "work", frozen, build=fake_builder(admitted),
        screen=fake_screen(), execute=execute,
      ).prepare()
      run.select_sources()
      run.freeze_corpus()
      run.run_discovery()
      final = run.finalize()
      self.assertEqual(final["execution"]["processes"]["started"], 96)
      self.assertTrue(seen)


class ReportAndControlTests(unittest.TestCase):
  def test_low_disk_space_pauses_without_building(self):
    import unittest.mock as mock
    with tempfile.TemporaryDirectory() as directory:
      frozen = study.freeze_study()
      built = []
      run = study.Study(
        Path(directory) / "work", frozen,
        build=lambda request: built.append(request) or {"kind": "refused"},
        screen=fake_screen(), execute=fake_execute()[0],
      ).prepare()
      with mock.patch.object(study.shutil, "disk_usage") as usage:
        usage.return_value = type("Usage", (), {"free": 0, "total": 0, "used": 0})()
        run.select_sources()
      self.assertTrue(run.is_paused())
      self.assertEqual(built, [])
      self.assertEqual(
        [e for e in run.path_ledger.events if e.get("kind") == "disk_guard"][0]["free_bytes"],
        0,
      )

  def test_process_events_are_counted_in_the_final_record(self):
    with tempfile.TemporaryDirectory() as directory:
      run = DiscoveryTests()._frozen_run(directory)
      for path in ("p0", "p1", "p2"):
        run.path_ledger.append("process", slot="s", path=path, event="start", mode="x")
        run.path_ledger.append(
          "process", slot="s", path=path, event="finish", mode="x", status="completed"
        )
      final = run.finalize()
      processes = final["execution"]["processes"]
      self.assertEqual(processes["started"], 3)
      self.assertEqual(processes["finished"], 3)
      self.assertEqual(processes["by_path"]["p1"]["started"], 1)

  def controls_root(self, directory):
    root = Path(directory) / "controls"
    for control in study.study_spec()["controls"]:
      for pattern in study.CONTROL_PATTERNS.values():
        for warps in (1, 4):
          bundle = root / control["source"] / f"{pattern}-warps-{warps}"
          bundle.mkdir(parents=True)
          (bundle / "program.sir").write_text(
            "fun @kernel() : i32 { ^entry: ret 0; }\n", encoding="utf-8"
          )
          (bundle / "inputs.bin").write_bytes(b"\x00" * 4)
          (bundle / "case.json").write_text(
            json.dumps({"launch": {"num_warps": warps}}), encoding="utf-8"
          )
    return root

  def frozen_with_controls(self, directory, controls_root):
    frozen = study.freeze_study()
    admitted = [slot["id"] for slot in frozen.spec["sources"]]
    run = study.Study(
      Path(directory) / "work", frozen, build=fake_builder(admitted),
      screen=fake_screen(), execute=fake_execute()[0],
      controls_root=controls_root,
    ).prepare()
    run.select_sources()
    run.freeze_corpus()
    return run

  def test_freeze_records_control_payload_and_case_hashes(self):
    with tempfile.TemporaryDirectory() as directory:
      root = self.controls_root(directory)
      run = self.frozen_with_controls(directory, root)
      controls = run.frozen_corpus()["controls"]
      self.assertEqual(len(controls), 4)
      for control in controls:
        self.assertEqual(len(control["hashes"]), 4)
        for entry in control["hashes"].values():
          self.assertEqual(set(entry["case_sha256"]), {"1", "4"})
          self.assertTrue(entry["program_sha256"])
          self.assertTrue(entry["inputs_sha256"])

  def test_a_changed_control_case_is_refused_at_freeze(self):
    with tempfile.TemporaryDirectory() as directory:
      root = self.controls_root(directory)
      control = study.study_spec()["controls"][0]
      pattern = study.CONTROL_PATTERNS["i0-base"]
      run = self.frozen_with_controls(directory, root)
      frozen_control = next(
        entry for entry in run.frozen_corpus()["controls"]
        if entry["case"] == control["id"]
      )
      (root / control["source"] / f"{pattern}-warps-4" / "case.json").write_text(
        json.dumps({"launch": {"num_warps": 4, "grid": 99}}), encoding="utf-8"
      )
      with self.assertRaises(study.StudyRefused):
        study.verify_control_payload(
          frozen_control, "i0-base",
          root / control["source"] / f"{pattern}-warps-4", 4,
        )

  def test_a_missing_control_is_recorded_and_left_unexecuted(self):
    with tempfile.TemporaryDirectory() as directory:
      root = self.controls_root(directory)
      control = study.study_spec()["controls"][0]
      pattern = study.CONTROL_PATTERNS["i0-base"]
      import shutil as shutil_module
      shutil_module.rmtree(root / control["source"] / f"{pattern}-warps-1")
      run = self.frozen_with_controls(directory, root)
      controls = {entry["case"]: entry for entry in run.frozen_corpus()["controls"]}
      self.assertEqual(controls[control["id"]]["status"], "missing")
      run.run_discovery()
      final = run.finalize()
      missing_slots = [
        slot_id for slot_id in final["execution"]["unexecuted"]
        if slot_id.startswith(control["id"])
      ]
      self.assertEqual(len(missing_slots), 8)


@unittest.skipUnless(DEVICE_TESTS, "device integration tests are opt-in")
class DeviceIntegrationTests(unittest.TestCase):
  def operations(self):
    root = Path(os.environ.get("REFRACTIR_ROOT", ".")).resolve()
    return study.device_operations(
      exporter=Path(os.environ.get(
        "REFRACTIR_TENSOR_FREE_FORM_EXPORT", root / "build/bin/tensor_free_form_export"
      )),
      symirc=Path(os.environ.get("REFRACTIR_SYMIRC", root / "symirc")),
      symiri=Path(os.environ.get("REFRACTIR_SYMIRI", root / "symiri")),
      controls_root=Path(os.environ["REFRACTIR_CONTROLS_ROOT"])
      if os.environ.get("REFRACTIR_CONTROLS_ROOT") else None,
    )

  def build(self, operations, directory, attempt=1, case_id=None):
    frozen = study.freeze_study()
    slot = (
      next(entry for entry in frozen.spec["sources"] if entry["id"] == case_id)
      if case_id is not None else frozen.spec["sources"][0]
    )
    seed = frozen.spec["attempts"]["seed_base"] + 8 * slot["index"] + attempt
    return slot, operations["build"]({
      "slot": slot, "attempt": attempt, "seed": seed,
      "max_attempts": frozen.spec["attempts"]["internal_max_attempts"],
      "directory": str(Path(directory) / f"build-{case_id or attempt}-{attempt}"),
    })

  def test_build_then_execute_reaches_the_workers(self):
    operations = self.operations()
    with tempfile.TemporaryDirectory() as directory:
      slot, built = self.build(operations, directory)
      self.assertEqual(built["kind"], "admitted", built.get("reason"))
      self.assertIn("directory", built)
      events = []
      outcome = operations["execute"]({
        "slot": "device", "case": slot["id"],
        "corpus": {"kind": "source", **built, "case": slot["id"]},
        "input": "i0-base", "configuration": "warps-1", "attempt": 0,
        "directory": str(Path(directory) / "run"),
        "on_process": events.append,
      })
      self.assertEqual(outcome["paths"]["p0"]["status"], "completed")
      self.assertEqual(outcome["paths"]["p1"]["status"], "completed")
      self.assertEqual(outcome["paths"]["p2"]["status"], "completed")
      paths = {event.get("path") for event in events}
      self.assertIn("p0", paths)
      self.assertIn("p1", paths)
      self.assertIn("p2", paths)
      self.assertEqual(
        [event["event"] for event in events if event.get("path") == "p2"],
        ["reserved", "start", "finish"],
      )

  def test_a_corrupt_cached_p0_does_not_block_the_targets(self):
    operations = self.operations()
    with tempfile.TemporaryDirectory() as directory:
      slot, built = self.build(operations, directory)
      built["screen"] = {
        "p0": [{
          "probe": "i0-base", "status": "completed",
          "output_sha256": "0" * 64, "output_path": str(Path(directory) / "missing.bin"),
        }]
      }
      outcome = operations["execute"]({
        "slot": "device", "case": slot["id"],
        "corpus": {"kind": "source", **built, "case": slot["id"]},
        "input": "i0-base", "configuration": "warps-1", "attempt": 0,
        "directory": str(Path(directory) / "run-corrupt"),
      })
      self.assertEqual(outcome["paths"]["p0"]["status"], "unavailable")
      self.assertEqual(outcome["paths"]["p1"]["status"], "completed")
      self.assertEqual(outcome["paths"]["p2"]["status"], "completed")

  def test_an_unavailable_selection_p0_is_not_replayed(self):
    operations = self.operations()
    with tempfile.TemporaryDirectory() as directory:
      for attempt in range(1, 5):
        slot, built = self.build(operations, directory, attempt)
        if built["kind"] != "admitted":
          continue
        screened = operations["screen"](built)
        if not screened["selected"]:
          continue
        built["screen"] = {
          "p0": [
            {"probe": probe, "status": "unavailable", "reason": "fixture"}
            for probe in ("i0-base", "i1-read-plus-one", "i2-late-row", "i3-unused-rows")
          ]
        }
        for warps in (1, 4):
          events = []
          outcome = operations["execute"]({
            "slot": "device", "case": slot["id"],
            "corpus": {"kind": "source", **built, "case": slot["id"]},
            "input": "i0-base", "configuration": f"warps-{warps}",
            "attempt": attempt,
            "directory": str(Path(directory) / f"run-{warps}"),
            "on_process": events.append,
          })
          self.assertEqual(outcome["paths"]["p0"]["status"], "unavailable")
          self.assertEqual(outcome["paths"]["p1"]["status"], "completed")
          self.assertEqual(outcome["paths"]["p2"]["status"], "completed")
          self.assertFalse(
            any(e.get("path") == "p0" and e.get("event") == "start" for e in events)
          )
        return
      self.fail("no selected body in the bound")

  def test_conditional_state_variants_execute(self):
    operations = self.operations()
    with tempfile.TemporaryDirectory() as directory:
      # The preserved selection record rejected this attempt for insensitivity,
      # so the six base probes run and the conditional state probes follow.
      slot, built = self.build(operations, directory, attempt=0, case_id="mw-g0c2")
      self.assertEqual(built["kind"], "admitted", built.get("reason"))
      screened = operations["screen"](built)
      conditional = [
        run for run in screened["record"]["runs"]
        if run.get("variant") != "combined"
      ]
      self.assertTrue(conditional, screened["record"]["selection"])
      self.assertTrue(
        any(run["status"] == "completed" for run in conditional),
        [run["status"] for run in conditional],
      )
      self.assertLessEqual(len(screened["record"]["runs"]), 10)

  def test_screening_records_p0_output_identity(self):
    operations = self.operations()
    with tempfile.TemporaryDirectory() as directory:
      for attempt in (1, 2, 3):
        slot, built = self.build(operations, directory, attempt=attempt)
        self.assertEqual(built["kind"], "admitted", built.get("reason"))
        events = []
        built["on_process"] = events.append
        screened = operations["screen"](built)
        self.assertIn("p1", {event.get("path") for event in events})
        if screened["selected"]:
          self.assertIn("p0", {event.get("path") for event in events})
        if not screened["selected"]:
          continue
        self.assertEqual(screened["p0"], 4)
        for entry in screened["record"]["p0"]:
          self.assertTrue(entry["output_sha256"])
          self.assertTrue(Path(entry["output_path"]).is_file())
          self.assertIn(entry["agreement"], (True, False, None))
        self.assertEqual(len(screened["record"]["p0"]), 4)
        return
      self.fail("no admitted body passed screening in the bound")


if __name__ == "__main__":
  unittest.main()
