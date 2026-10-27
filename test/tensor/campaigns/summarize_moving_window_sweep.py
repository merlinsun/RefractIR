"""Combine moving-window sweep shards into one checked summary.

Each shard is a sweep work directory. The summary is computed from the
retained run logs and candidate records, not from each shard's `report.json`,
so it also covers shards that are still running or were interrupted. Rates
and disk use count admitted kernels only, using the driver's own
`measurements`, and shard time is the summed per-kernel session-clock time.

Compiled-path timeouts are reported with the kernels, runs, skipped pairs,
and shard time they took. Candidates are grouped by signature across
shards. A path difference at the same configuration and input as a
non-finite output is undefined in SIR, so it is counted with the non-finite
evidence even in records written before the driver tagged such differences;
only the remaining path differences are listed as finite leads.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_DRIVER = Path(__file__).with_name("run_moving_window_sweep.py")
_spec = importlib.util.spec_from_file_location("moving_window_sweep", _DRIVER)
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

SCHEMA = "refractir.moving-window-sweep-summary.v1"


def _lines(path):
  path = Path(path)
  if not path.is_file():
    return []
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sessions(history):
  """One entry per start or restart, closed by its stop line if it has one."""
  sessions = []
  for line in history:
    if line["event"] in ("start", "restart"):
      sessions.append(
        {
          "session": len(sessions) + 1,
          "device": line.get("device"),
          "timeout_s": line.get("timeout_s"),
          "started": line["time"],
          "stop_reason": None,
          "duration_s": None,
          "kernels": 0,
        }
      )
    elif sessions and line["event"] == "kernel":
      sessions[-1]["kernels"] += 1
    elif sessions and line["event"] == "stop":
      sessions[-1]["stop_reason"] = line["reason"]
      sessions[-1]["duration_s"] = line["time"] - sessions[-1]["started"]
  return sessions


def _declared_digest(work, declaration):
  for shard in (declaration or {}).get("shards", ()):
    if Path(shard["work"]).name == Path(work).name:
      return shard["spec_sha256"]
  return None


def _shard(work, declaration):
  work = Path(work)
  history = _lines(work / "run-log.jsonl")
  kernels = [line for line in history if line["event"] == "kernel"]
  frozen = json.loads((work / "sweep.json").read_text())
  declared = _declared_digest(work, declaration)
  sessions = _sessions(history)
  return (
    {
      "shard": str(work),
      "spec_sha256": frozen["spec_sha256"],
      "spec_matches_declaration": None
      if declared is None
      else declared == frozen["spec_sha256"],
      "sessions": sessions,
      "stop_reason": sessions[-1]["stop_reason"] if sessions else None,
      "counts": _counts(kernels),
      "gpu_workers_dispatched": sum(line["event"] == "dispatch" for line in history),
      "measurements": sweep.measurements(kernels),
    },
    kernels,
    history,
  )


def _counts(kernels):
  return {
    status: sum(line["status"] == status for line in kernels)
    for status in ("agreement", "candidate", "infrastructure", "refused")
  }


def _p2_timeouts(kernels):
  hit = [
    line
    for line in kernels
    if any(pair["p2"]["status"] == "timed_out" for pair in line.get("pairs", ()))
  ]
  total = sum(line.get("elapsed_s", 0) for line in kernels)
  spent = sum(line.get("elapsed_s", 0) for line in hit)
  return {
    "kernels": sorted(line["seed"] for line in hit),
    "runs": sum(
      pair["p2"]["status"] == "timed_out" for line in hit for pair in line["pairs"]
    ),
    "skipped_pairs": sum(
      pair["p1"]["status"] == sweep.SKIPPED
      for line in kernels
      for pair in line.get("pairs", ())
    ),
    "shard_seconds": spent,
    "share_of_shard_time": spent / total if total else None,
  }


def _where(candidate):
  return (
    candidate["seed"],
    json.dumps(candidate.get("configuration"), sort_keys=True),
    candidate.get("input"),
  )


def _candidates(shards):
  """Signatures across shards, and the path differences between defined outputs."""
  signatures = {}
  finite = {}
  for shard in shards:
    records = _lines(Path(shard) / "candidates.jsonl")
    non_finite = {_where(c) for c in records if c["kind"] == "non_finite"}
    for candidate in records:
      entry = signatures.setdefault(
        candidate["signature"],
        {"kind": candidate["kind"], "count": 0, "seeds": set(), "shards": set()},
      )
      entry["count"] += 1
      entry["seeds"].add(candidate["seed"])
      entry["shards"].add(str(shard))
      if candidate["kind"] != "path_difference":
        continue
      if _where(candidate) in non_finite:
        entry.setdefault("non_finite_seeds", set()).add(candidate["seed"])
        continue
      lead = finite.setdefault(
        (str(shard), candidate["seed"]),
        {"shard": str(shard), "seed": candidate["seed"], "runs": 0, "positions": set()},
      )
      lead["runs"] += 1
      lead["positions"].update(candidate.get("positions") or ())
  for entry in signatures.values():
    for key in ("seeds", "shards", "non_finite_seeds"):
      if key in entry:
        entry[key] = sorted(entry[key])
  leads = sorted(finite.values(), key=lambda lead: (lead["shard"], lead["seed"]))
  for lead in leads:
    # Word indices in numeric order; a length mismatch is recorded as text.
    positions = lead["positions"]
    lead["positions"] = sorted(p for p in positions if isinstance(p, int)) + sorted(
      p for p in positions if not isinstance(p, int)
    )
  return signatures, leads


def summarize(shards, declaration=None):
  """The combined summary for the given shard work directories."""
  shards = [Path(shard) for shard in shards]
  records, kernels, starts, stops = [], [], [], []
  dispatched = 0
  for work in shards:
    record, shard_kernels, history = _shard(work, declaration)
    records.append(record)
    kernels += shard_kernels
    dispatched += record["gpu_workers_dispatched"]
    starts += [
      line["time"] for line in history if line["event"] in ("start", "restart")
    ]
    stops += [line["time"] for line in history if line["event"] == "stop"]
  measured = sweep.measurements(kernels)
  admitted = measured["infrastructure"]["admitted"]
  wall = max(stops) - min(starts) if starts and stops else None
  signatures, leads = _candidates(shards)
  return {
    "schema": SCHEMA,
    "shards": records,
    "totals": {
      "counts": _counts(kernels),
      "admitted": admitted,
      "gpu_workers_dispatched": dispatched,
      "shard_seconds": measured["elapsed_s"],
      "admitted_per_shard_hour": measured["kernels_per_hour"],
      "wall_clock_s": wall,
      "admitted_per_wall_hour": admitted * 3600 / wall if wall else None,
      "disk_bytes_per_admitted_kernel": measured["disk_bytes_per_kernel"],
      "infrastructure": measured["infrastructure"],
      "live_nodes": measured["live_nodes"],
      "exposure": measured["exposure"],
      "p2_timeouts": _p2_timeouts(kernels),
    },
    "signatures": signatures,
    "finite_path_differences": leads,
  }


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("shards", type=Path, nargs="+")
  parser.add_argument("--declaration", type=Path)
  parser.add_argument("--out", type=Path, required=True)
  arguments = parser.parse_args(argv)
  declaration = (
    json.loads(arguments.declaration.read_text()) if arguments.declaration else None
  )
  result = summarize(arguments.shards, declaration=declaration)
  arguments.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
  totals = result["totals"]
  print(
    json.dumps(
      {
        "admitted": totals["admitted"],
        "counts": totals["counts"],
        "admitted_per_shard_hour": totals["admitted_per_shard_hour"],
        "admitted_per_wall_hour": totals["admitted_per_wall_hour"],
        "p2_timeout_share": totals["p2_timeouts"]["share_of_shard_time"],
        "stop_reasons": [shard["stop_reason"] for shard in result["shards"]],
        "finite_path_differences": [
          lead["seed"] for lead in result["finite_path_differences"]
        ],
      },
      indent=1,
    )
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
