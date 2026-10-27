"""Multi-shard sweep summary contracts, over records written by the sweep driver."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from test.unit import test_moving_window_sweep as fixtures
from test.unit.test_moving_window_sweep import I0, I3, S0, TWO, Script, ticking

sweep = fixtures.sweep
SUMMARY_PATH = fixtures.DRIVER_PATH.with_name("summarize_moving_window_sweep.py")
_spec = importlib.util.spec_from_file_location("sweep_summary", SUMMARY_PATH)
summary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summary)


def declared(start, count, wall_clock_s=10**6):
  return sweep.sweep_spec(
    seed_start=start,
    seed_count=count,
    configurations=fixtures.CONFIGURATIONS,
    sample_every=20,
    gpu_workers=1000,
    wall_clock_s=wall_clock_s,
  )


class SummaryTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name)

  def shard(self, index, count, script, wall_clock_s=10**6, **kwargs):
    kwargs.setdefault("free_bytes", lambda path: 10**12)
    kwargs.setdefault("clock", ticking(600))
    work = self.root / f"shard-{index}"
    declaration = declared(S0 + 1000 * index, count, wall_clock_s)
    sweep.run_sweep(work, declaration, script.operations, **kwargs)
    return work

  def test_rates_count_admitted_kernels_only(self):
    first = self.shard(0, 3, Script(refused={S0 + 1}))
    second = self.shard(1, 2, Script())
    result = summary.summarize([first, second])
    totals = result["totals"]
    self.assertEqual(
      totals["counts"],
      {"agreement": 4, "candidate": 0, "infrastructure": 0, "refused": 1},
    )
    self.assertEqual(totals["admitted"], 4)
    self.assertEqual(totals["shard_seconds"], 3000)
    self.assertEqual(totals["admitted_per_shard_hour"], 4 * 3600 / 3000)
    self.assertEqual(totals["gpu_workers_dispatched"], 16)

  def test_disk_is_averaged_over_admitted_kernels(self):
    first = self.shard(0, 3, Script(refused={S0 + 1}))
    kernels = [
      json.loads(line)
      for line in (first / "run-log.jsonl").read_text().splitlines()
      if json.loads(line)["event"] == "kernel"
    ]
    admitted = [line["disk_bytes"] for line in kernels if line["status"] != "refused"]
    disk = summary.summarize([first])["totals"]["disk_bytes_per_admitted_kernel"]
    self.assertEqual(
      disk,
      {"mean": sum(admitted) / 2, "max": max(admitted), "total": sum(admitted)},
    )

  def test_compiled_timeouts_report_their_kernels_time_and_skips(self):
    script = Script(
      statuses={
        (S0 + 1, 1, I0, "p2"): ("timed_out", "worker exceeded", "compilation"),
        (S0 + 1, 4, I0, "p2"): ("timed_out", "worker exceeded", "compilation"),
      }
    )
    first = self.shard(0, 2, script)
    timeouts = summary.summarize([first])["totals"]["p2_timeouts"]
    self.assertEqual(timeouts["kernels"], [S0 + 1])
    self.assertEqual(timeouts["runs"], 2)
    self.assertEqual(timeouts["skipped_pairs"], 2)
    self.assertEqual(timeouts["shard_seconds"], 600)
    self.assertEqual(timeouts["share_of_shard_time"], 0.5)

  def test_finite_path_differences_are_listed_apart_from_non_finite_ones(self):
    finite = self.shard(
      0, 1, Script(outputs={(S0, 1, I0, "p2"): TWO, (S0, 1, I3, "p2"): TWO})
    )
    seed = S0 + 1000
    older = self.shard(1, 1, Script())
    # Records written before non-finite differences were tagged call them
    # path_difference; the summary must still treat them as non-finite.
    lines = [
      {
        "kind": "non_finite",
        "path": "p1",
        "configuration": {"num_warps": 1, "num_stages": 1},
        "input": I0,
      },
      {
        "kind": "path_difference",
        "positions": [1],
        "configuration": {"num_warps": 1, "num_stages": 1},
        "input": I0,
      },
    ]
    with (older / "candidates.jsonl").open("a") as handle:
      for line in lines:
        record = {**line, "seed": seed, "directory": "d"}
        record["signature"] = sweep.signature(record)
        handle.write(json.dumps(record) + "\n")
    result = summary.summarize([finite, older])
    self.assertEqual(
      [entry["seed"] for entry in result["finite_path_differences"]], [S0]
    )
    self.assertEqual(result["finite_path_differences"][0]["shard"], str(finite))
    kinds = {entry["kind"] for entry in result["signatures"].values()}
    self.assertIn("non_finite", kinds)
    self.assertTrue(
      any(
        entry.get("non_finite_seeds") == [seed]
        for entry in result["signatures"].values()
      )
    )

  def test_each_shard_reports_its_sessions_stops_and_declared_digest(self):
    # The first session stops on its wall clock after one kernel; the restart
    # finishes the two-seed block.
    first = self.shard(0, 2, Script(), wall_clock_s=2500, clock=ticking(1000))
    sweep.run_sweep(
      first,
      declared(S0, 2, 2500),
      Script().operations,
      free_bytes=lambda path: 10**12,
      clock=ticking(1000),
    )
    digest = json.loads((first / "sweep.json").read_text())["spec_sha256"]
    other = self.shard(1, 1, Script())
    declaration = {
      "shards": [
        {"work": "/host/shard-0", "spec_sha256": digest},
        {"work": "/host/shard-1", "spec_sha256": "0" * 64},
      ]
    }
    result = summary.summarize([first, other], declaration=declaration)
    shards = result["shards"]
    self.assertEqual([s["spec_matches_declaration"] for s in shards], [True, False])
    self.assertEqual(
      [session["stop_reason"] for session in shards[0]["sessions"]],
      ["wall_clock", "seed_block_complete"],
    )
    self.assertEqual(shards[0]["stop_reason"], "seed_block_complete")
    self.assertEqual([session["kernels"] for session in shards[0]["sessions"]], [1, 1])


if __name__ == "__main__":
  unittest.main()
