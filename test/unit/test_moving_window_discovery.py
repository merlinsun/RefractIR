"""Contract tests for the frozen moving-window study driver."""

import importlib.util
import json
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
      execute=fake_execute(**execute_kwargs)[0],
    ).prepare()
    run.select_sources()
    run.freeze_corpus()
    return run

  def test_paths_reconcile_and_reserve_stops_launches(self):
    with tempfile.TemporaryDirectory() as directory:
      run = self._frozen_run(directory, cached_p0=True)
      run.run_discovery()
      final = run.finalize()
      self.assertEqual(final["execution"]["states"]["completed"], 128)
      self.assertEqual(final["execution"]["unexecuted"], [])
      self.assertEqual(final["paths"]["p0"]["cached"], 128)
      self.assertEqual(final["paths"]["p0"]["launches"], 0)
      self.assertEqual(final["paths"]["p1"]["launches"], 128)
      self.assertEqual(final["paths"]["p2"]["launches"], 128)
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
    with tempfile.TemporaryDirectory() as directory:
      run = self._frozen_run(directory, compiler_status="unavailable", candidate=True)
      run.run_discovery()
      final = run.finalize()
      self.assertGreater(len(final["candidates"]), 0)
      self.assertGreater(final["compiler"]["unavailable"], 0)
      self.assertTrue(final["corpus"], "the final record keeps the admitted corpus")
      self.assertTrue(final["observations"], "the final record keeps its observations")

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
        screen=fake_screen(), execute=fake_execute()[0],
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
      entry["source_sha256"] = "contradiction"
      run.corpus_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
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


if __name__ == "__main__":
  unittest.main()
