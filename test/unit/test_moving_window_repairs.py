"""Regression contracts for partial evidence and restart-safe study accounting."""

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test.lib import tensor_execution
from test.lib import tensor_state_diagnostics as diagnostics
from test.lib import tensor_target_reference as reference
from test.unit import test_moving_window_discovery as fixtures
from test.unit import test_tensor_target_run as targets

study = fixtures.study


def read(path):
  return json.loads(Path(path).read_text(encoding="utf-8"))


def tree(path):
  return {
    str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()
  }


class _TargetFixture:
  def setUp(self):
    self.worker = targets.TargetRunTests("runTest")
    self.worker.setUp()
    self.addCleanup(self.worker.doCleanups)
    self.worker.write_bundle()

  def refused(self, *, p1=None, p0=True, different=False, mode="compiled"):
    self.worker.configure(
      {"interpret": p1 or {"actual": targets.AGREE_HEX}, "compiled": {}}
    )
    observed = None
    if p0:
      observed = reference.ReferenceObservation(
        reference.COMPLETED,
        output=bytes.fromhex(targets.AGREE_HEX) if not different else bytes(16),
      )

    def account(event):
      if event["event"] == "reserved" and event["mode"] == mode:
        raise tensor_execution.DispatchRefused("reserve exhausted")

    self.worker.run_target(reference=observed, on_process=account)
    return read(self.worker.bundle / "target-runs" / "target-run.json")


class PartialTargetTests(_TargetFixture, unittest.TestCase):
  def test_agreement_is_not_a_candidate_after_p2_refusal(self):
    saved = self.refused()
    self.assertEqual(saved["compiled"]["status"], "not_run")
    self.assertEqual(saved["diagnosis"]["observations"], {"R-T": "equal"})
    self.assertFalse(saved["candidate"])

  def test_difference_survives_p2_refusal(self):
    saved = self.refused(different=True)
    self.assertEqual(saved["compiled"]["status"], "not_run")
    self.assertEqual(saved["diagnosis"]["observations"], {"R-T": "different"})
    self.assertTrue(saved["candidate"])

  def test_absent_reference_and_p2_refusal_are_not_a_difference(self):
    saved = self.refused(p0=False)
    self.assertEqual(saved["compiled"]["status"], "not_run")
    self.assertFalse(saved["candidate"])

  def test_interpreter_failure_survives_p2_refusal(self):
    saved = self.refused(p1={"exit": 1, "phase": "runtime", "stderr": "failed"})
    self.assertEqual(saved["compiled"]["status"], "not_run")
    self.assertTrue(saved["candidate"])
    self.assertEqual(saved["diagnosis"]["diagnosis"], "interpreter_candidate")

  def test_unsupported_interpreter_and_refusal_are_not_numerical(self):
    saved = self.refused(p1={"exit": 1, "phase": "runtime", "stderr": "unsupported"})
    self.assertEqual(saved["compiled"]["status"], "not_run")
    self.assertFalse(saved["candidate"])

  def test_not_run_has_no_exit_code_or_output_hash(self):
    saved = self.refused()
    self.assertIsNone(saved["compiled"]["exit_code"])
    self.assertIsNone(saved["compiled"]["actual_sha256"])
    self.assertEqual(saved["compiled"]["reason"], "reserve exhausted")


class PartialScreenTests(_TargetFixture, unittest.TestCase):
  def test_p1_refusal_saves_p0_and_both_not_run_paths(self):
    saved = self.refused(mode="interpret")
    self.assertEqual(saved["reference"]["status"], reference.COMPLETED)
    self.assertEqual(saved["interpreter"]["status"], "not_run")
    self.assertEqual(saved["compiled"]["status"], "not_run")
    self.assertFalse(saved["candidate"])

  def screen_until(self, limit, *, failure=None, insensitive=False):
    calls = []

    def probe(name, trips, variant):
      if len(calls) == limit:
        raise tensor_execution.DispatchRefused("screen allocation exhausted")
      calls.append((name, trips, variant))
      if failure and len(calls) == failure:
        return "runtime_failure", None
      values = {
        "i0-base": 1,
        "i1-read-plus-one": 2,
        "i2-late-row": 3,
        "i3-unused-rows": 4,
      }
      value = 1 if insensitive else values[name]
      return "completed", struct.pack("<f", value)

    result = diagnostics.window_screen(trip_count=4, run_probe=probe)
    self.assertEqual(len(calls), limit)
    self.assertFalse(result["selection"]["selected"])
    self.assertEqual(result["selection"]["reason"], "incomplete_observation")
    self.assertTrue(result["dispatch_refused"])
    return result

  def test_i0_i3_candidate_survives_fifth_probe_refusal(self):
    result = self.screen_until(4)
    self.assertTrue(
      any(c.get("kind") == "unused_rows_changed" for c in result["candidates"])
    )
    self.assertEqual(sum(r["status"] == "completed" for r in result["runs"]), 4)

  def test_earlier_failure_survives_later_refusal(self):
    result = self.screen_until(3, failure=2)
    self.assertTrue(
      any(c.get("status") == "runtime_failure" for c in result["candidates"])
    )

  def test_missing_pairs_are_incomplete_not_insensitive(self):
    result = self.screen_until(1)
    self.assertEqual(result["conditional_runs"], 0)
    self.assertEqual(result["candidates"], [])

  def test_partial_screen_is_persisted_and_cannot_be_admitted(self):
    with tempfile.TemporaryDirectory() as directory:
      work = Path(directory)
      frozen = study.freeze_study()
      slot = frozen.spec["sources"][0]["id"]
      builds = []

      def screen(outcome):
        result = self.screen_until(4)
        return {
          "selected": False,
          "dispatch_refused": result["dispatch_refused"],
          "record": result,
          "candidates": result["candidates"],
        }

      runner = study.Study(
        work,
        frozen,
        build=fixtures.fake_builder([slot], log=builds),
        screen=screen,
        execute=None,
      ).prepare()
      runner.select_sources(max_slots=1)
      self.assertTrue(runner.is_paused())
      rows = [e for e in runner.path_ledger.events if e["kind"] == "selection_screen"]
      self.assertTrue(rows[-1]["record"]["dispatch_refused"])
      self.assertEqual(len(builds), 1)
      with self.assertRaises(study.StudyRefused):
        runner.record_triage_decision("admit")
      with self.assertRaises(study.StudyRefused):
        runner.select_sources(max_slots=1)

  def test_conditional_refusal_keeps_the_base_observations(self):
    result = self.screen_until(7, insensitive=True)
    self.assertEqual(sum(r["status"] == "completed" for r in result["runs"]), 7)
    self.assertEqual(sum(r["variant"] == "combined" for r in result["runs"]), 6)

  def test_p0_dispatch_refusal_saves_record_without_starting_targets(self):
    from test.lib import tensor_target_case

    case = tensor_target_case.load(self.worker.bundle, symiri=self.worker.symiri)

    def refuse(event):
      raise tensor_execution.DispatchRefused("P0 budget exhausted")

    with mock.patch.object(reference.subprocess, "Popen") as spawn:
      observed = reference.observe(case, symiri=self.worker.symiri, on_process=refuse)
    spawn.assert_not_called()
    self.assertEqual(observed.status, "not_run")
    events = []
    self.worker.run_target(
      reference=observed, dispatch_refused=observed.reason, on_process=events.append
    )
    self.assertEqual(events, [])
    saved = read(self.worker.bundle / "target-runs" / "target-run.json")
    self.assertEqual(saved["reference"]["status"], "not_run")
    self.assertEqual(saved["interpreter"]["status"], "not_run")
    self.assertEqual(saved["compiled"]["status"], "not_run")
    self.assertFalse(saved["candidate"])


class QualificationFixture:
  def __init__(self, root):
    self.root = Path(root)
    self.builds = []
    self.calls = []
    self.events = []
    self.fail_at = None
    self.p0_status = reference.COMPLETED
    self.toolchain = {
      "exporter_sha256": "1" * 64,
      "symirc_sha256": "2" * 64,
      "symiri_sha256": "3" * 64,
      "triton_library_sha256": "4" * 64,
      "triton": {"revision": "pinned", "device": "fixture"},
      "python": "fixture",
      "device": "fixture",
    }
    self.operations = {
      "build": self.build,
      "execute": self.execute,
      "toolchain": self.toolchain,
    }

  def build(self, request):
    self.builds.append(request)
    directory = Path(request["directory"])
    directory.mkdir(parents=True)
    source = "fun @kernel() {}\n"
    data = struct.pack("<f", 1)
    bundle = directory / "bundle"
    bundle.mkdir()
    (bundle / "program.sir").write_text(source)
    (bundle / "inputs.bin").write_bytes(data)
    (bundle / "case.json").write_text(json.dumps({"launch": {"num_warps": 1}}))
    return {
      "kind": "admitted",
      "synthetic": True,
      "directory": str(directory),
      "source": source,
      "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
      "inputs_sha256": hashlib.sha256(data).hexdigest(),
      "probe_values": {"i0-base": list(data)},
      "layouts": {},
      "region": {"profile": "v3", "entry": "kernel"},
      "bundles": {"i0-base": str(bundle)},
    }

  def execute(self, request):
    self.calls.append(request)
    directory = Path(request["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    observed = request.get("reference")
    paths = {}
    for path in ("p0", "p1", "p2"):
      if path == "p0" and observed is not None:
        paths[path] = {"status": observed.status, "cached": True}
        continue
      for event in ("reserved", "start", "finish"):
        row = {"event": event, "path": path, "pid": 123, "status": "completed"}
        request["on_process"](row)
        self.events.append((path, event))
        if self.fail_at == (request["configuration"], path, event):
          raise KeyboardInterrupt()
      if path == "p0":
        observed = reference.ReferenceObservation(
          self.p0_status,
          output=b"ABCD" if self.p0_status == reference.COMPLETED else None,
          reason="fixture observation",
        )
        if request.get("on_reference"):
          request["on_reference"](observed)
      paths[path] = {"status": "completed", "cached": False}
    (directory / "target-run.json").write_text('{"status":"target_agreement"}')
    return {
      "status": "target_agreement",
      "paths": paths,
      "candidate": False,
      "compiler": {"status": "unavailable"},
      "reference": observed,
    }

  def run(self, **options):
    return study.run_qualification(
      self.root, operations=self.operations, slots=1, **options
    )

  @property
  def case(self):
    return study.SOURCE_SLOTS[0]["id"]

  def interrupt(self):
    self.fail_at = ("warps-4", "p2", "start")
    try:
      self.run()
    except KeyboardInterrupt:
      pass
    else:
      raise AssertionError("expected interruption")
    self.fail_at = None
    self.calls.clear()
    self.events.clear()

  def token(self):
    return f"{self.case}/warps-4/attempt-0"


class SharedBudgetTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.fixture = QualificationFixture(temporary.name)

  def authority(self):
    f = self.fixture
    path = f.root / "study.json"
    frozen = (
      study.FrozenStudy.load(path)
      if path.exists()
      else study.freeze_study(toolchain=f.toolchain)
    )
    return study.Study(f.root, frozen, build=None, screen=None, execute=None).prepare()

  def charge(self, runner, path, count):
    for i in range(count):
      account = runner.process_observer("triage", case="triage", attempt=i)
      account({"event": "reserved", "path": path})
      account({"event": "start", "path": path, "pid": i + 1})
      account({"event": "finish", "path": path, "status": "completed"})

  def test_shared_p2_exhaustion_blocks_qualification_retry(self):
    f = self.fixture
    f.interrupt()
    self.charge(self.authority(), "p2", 32)
    f.run(retry=f.token())
    self.assertNotIn(("p2", "start"), f.events)

  def test_qualification_retry_charges_are_visible_to_discovery(self):
    f = self.fixture
    f.interrupt()
    f.run(retry=f.token())
    runner = self.authority()
    self.assertEqual(runner.reserve_spent("p2"), 1)
    self.assertEqual(runner.reserve_spent("p1"), 1)
    self.assertEqual(runner.reserve_spent("p0"), 0)

  def test_imports_count_by_path_across_restart(self):
    runner = self.authority()
    path = self.fixture.root / "import.jsonl"
    path.write_text(
      "\n".join(
        json.dumps(
          {
            "execution_id": p,
            "path": p,
            "allocation": "triage",
            "result": "completed",
            "case": "imported",
          }
        )
        for p in ("p0", "p1", "p2")
      )
      + "\n"
    )
    runner.triage_ledger = path
    runner.prepare()
    runner = self.authority()
    for p in ("p0", "p1", "p2"):
      self.charge(runner, p, 31)
      with self.assertRaises(tensor_execution.DispatchRefused):
        runner.process_observer("triage", case="over", attempt=32)(
          {"event": "reserved", "path": p}
        )

  def test_cached_p0_does_not_need_p0_retry_capacity(self):
    f = self.fixture
    f.interrupt()
    self.charge(self.authority(), "p0", 32)
    result = f.run(retry=f.token())
    self.assertEqual(len(result["cases"][0]["configurations"]), 2)
    self.assertNotIn(("p0", "start"), f.events)
    self.assertIn(("p2", "start"), f.events)

  def test_reserved_process_remains_charged_on_restart(self):
    runner = self.authority()
    account = runner.process_observer("triage", case="interrupted", attempt=0)
    account({"event": "reserved", "path": "p2"})
    self.assertEqual(self.authority().reserve_spent("p2"), 1)
    with self.assertRaises(study.StudyRefused):
      self.authority().process_observer("triage", case="interrupted", attempt=0)(
        {"event": "reserved", "path": "p2"}
      )

  def test_spawn_failed_releases_only_its_own_hold_once(self):
    runner = self.authority()
    account = runner.process_observer("triage", case="failed-spawn", attempt=0)
    account({"event": "reserved", "path": "p2"})
    self.assertEqual(runner.reserve_spent("p2"), 1)
    event = {"event": "finish", "path": "p2", "status": "spawn_failed"}
    account(event)
    account(event)
    self.assertEqual(self.authority().reserve_spent("p2"), 0)

  def test_saved_and_rebuilt_reports_include_separate_reserves(self):
    runner = self.authority()
    runner.build = fixtures.fake_builder(
      [s["id"] for s in runner.frozen.spec["sources"]]
    )
    runner.screen = fixtures.fake_screen()
    runner.select_sources()
    runner.freeze_corpus()
    self.charge(runner, "p2", 1)
    saved = runner.finalize()
    rebuilt = study.report_from_records(runner.work)
    self.assertEqual(saved, rebuilt)
    self.assertEqual(saved["reserve"]["by_path"]["p2"], {"spent": 1, "remaining": 31})
    self.assertEqual(saved["reserve"]["by_path"]["p0"], {"spent": 0, "remaining": 32})


class QualificationIdentityTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.fixture = QualificationFixture(temporary.name)
    self.fixture.run()
    self.fixture.calls.clear()
    self.fixture.builds.clear()

  def refused_without_mutation(self):
    f = self.fixture
    before = tree(f.root)
    with self.assertRaises(study.StudyRefused):
      f.run()
    self.assertEqual(tree(f.root), before)
    self.assertEqual(f.calls, [])
    self.assertEqual(f.builds, [])

  def test_changed_symirc_is_refused(self):
    self.fixture.toolchain["symirc_sha256"] = "a" * 64
    self.refused_without_mutation()

  def test_changed_triton_or_device_is_refused(self):
    self.fixture.toolchain["triton"]["revision"] = "other"
    self.fixture.toolchain["device"] = "other"
    self.refused_without_mutation()

  def test_changed_layout_is_refused(self):
    path = self.fixture.root / self.fixture.case / "build.json"
    built = read(path)
    built["layouts"] = {"%x": {"element_offset": 1}}
    path.write_text(json.dumps(built))
    self.refused_without_mutation()

  def test_changed_input_payload_is_refused(self):
    built = read(self.fixture.root / self.fixture.case / "build.json")
    (Path(built["bundles"]["i0-base"]) / "inputs.bin").write_bytes(b"WXYZ")
    self.refused_without_mutation()

  def test_missing_completed_artifact_is_refused(self):
    next(self.fixture.root.glob("*/warps-1-run-0/target-run.json")).unlink()
    self.refused_without_mutation()

  def test_unchanged_completed_cases_make_no_worker_or_builder_calls(self):
    result = self.fixture.run()
    self.assertTrue(result["cases"][0]["validated"])
    self.assertEqual(self.fixture.builds, [])
    self.assertEqual(self.fixture.calls, [])


class ReferenceRetryTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.fixture = QualificationFixture(temporary.name)
    self.fixture.interrupt()
    (self.fixture.root / self.fixture.case / "p0.bin").unlink()

  def invalid_token(self, token):
    f = self.fixture
    before = tree(f.root)
    with self.assertRaises(study.StudyRefused):
      f.run(retry=token)
    self.assertEqual(tree(f.root), before)
    self.assertEqual(f.calls, [])

  def test_unknown_case_cannot_authorize_reference_replay(self):
    self.invalid_token("unknown/warps-1/attempt-0")

  def test_completed_configuration_cannot_authorize_reference_replay(self):
    self.invalid_token(f"{self.fixture.case}/warps-1/attempt-0")

  def test_stale_attempt_cannot_authorize_reference_replay(self):
    self.invalid_token(f"{self.fixture.case}/warps-4/attempt-99")

  def test_target_retry_with_missing_cache_does_not_replay_p0(self):
    f = self.fixture
    result = f.run(retry=f.token())
    self.assertEqual(len(f.calls), 1)
    self.assertEqual(f.calls[0]["reference"].status, reference.UNAVAILABLE)
    self.assertNotIn(("p0", "start"), f.events)
    self.assertFalse(result["cases"][0]["validated"])

  def test_explicit_reference_recheck_is_charged_and_keeps_prior_record(self):
    f = self.fixture
    original = (f.root / f.case / "p0.json").read_bytes()
    record = read(f.root / f.case / "p0.json")
    token = f"{f.case}/{record['observation_id']}"
    f.run(retry=f.token(), reference_recheck=token)
    self.assertIn(("p0", "start"), f.events)
    self.assertEqual((f.root / f.case / "p0.json").read_bytes(), original)
    self.assertTrue(list((f.root / f.case / "reference-rechecks").glob("*/p0.json")))
    frozen = study.FrozenStudy.load(f.root / "study.json")
    runner = study.Study(
      f.root, frozen, build=None, screen=None, execute=None
    ).prepare()
    self.assertEqual(runner.reserve_spent("p0"), 1)

  def test_failed_p0_observation_is_reused_across_restart(self):
    with tempfile.TemporaryDirectory() as directory:
      f = QualificationFixture(directory)
      f.p0_status = reference.FAILED
      f.interrupt()
      f.run(retry=f.token())
      self.assertEqual(f.calls[0]["reference"].status, reference.FAILED)
      self.assertNotIn(("p0", "start"), f.events)


class AccountingModeTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name)
    spec = study.study_spec()
    spec.pop("accounting")
    self.frozen = study.freeze_study(spec)
    self.frozen.write(self.root / "study.json")
    study.PathLedger(self.root / "path-ledger.jsonl").append(
      "process", phase="discovery", path="p1", event="start", pid=1
    )

  def runner(self, **kw):
    return study.Study(
      self.root, self.frozen, build=None, screen=None, execute=None, **kw
    )

  def test_unversioned_process_requires_choice_before_mutation(self):
    before = tree(self.root)
    with self.assertRaises(study.StudyRefused):
      self.runner().prepare()
    self.assertEqual(tree(self.root), before)

  def test_api_cannot_override_process_version_with_legacy(self):
    with self.assertRaises(study.StudyRefused):
      study.Study(
        self.root / "v2",
        study.freeze_study(),
        build=None,
        screen=None,
        execute=None,
        legacy_accounting=True,
      ).prepare()

  def test_process_adoption_is_durable_without_rewriting_freeze(self):
    before = (self.root / "study.json").read_bytes()
    self.runner(legacy_accounting=False).prepare()
    resumed = self.runner().prepare()
    self.assertFalse(resumed.legacy_accounting)
    self.assertEqual((self.root / "study.json").read_bytes(), before)

  def test_live_and_reconstructed_reports_share_mode(self):
    runner = self.runner(legacy_accounting=False).prepare()
    with mock.patch.object(
      runner,
      "_final_record",
      side_effect=lambda counts: {
        "paths": study.path_counts(
          runner.path_ledger.events, legacy=runner.legacy_accounting
        )
      },
    ):
      live = runner.finalize()
    with mock.patch.object(
      study.Study,
      "_final_record",
      autospec=True,
      side_effect=lambda s, counts: {
        "paths": study.path_counts(s.path_ledger.events, legacy=s.legacy_accounting)
      },
    ):
      rebuilt = study.report_from_records(self.root)
    self.assertEqual(live, rebuilt)
    self.assertEqual(live["paths"]["p1"]["launches"], 1)

  def test_conflicting_mode_is_refused_before_decisions(self):
    self.runner(legacy_accounting=False).prepare()
    before = tree(self.root)
    with self.assertRaises(study.StudyRefused):
      self.runner(legacy_accounting=True).prepare()
    self.assertEqual(tree(self.root), before)

  def test_legacy_report_preserves_its_field_set(self):
    with tempfile.TemporaryDirectory() as directory:
      runner = fixtures.DiscoveryTests()._frozen_run(directory, cached_p0=True)
      runner.run_discovery()
      expected = runner.finalize()
      actual = study.report_from_records(runner.work)
      self.assertEqual(actual, expected)
      self.assertNotIn("not_run", actual["paths"]["p0"])


if __name__ == "__main__":
  unittest.main()
