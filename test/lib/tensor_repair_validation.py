"""Fail-stop runner for the declared moving-window repair validation batch.

This is a test driver, not another campaign scheduler. Per-study process
ledgers remain authoritative; batch limits are checked from those ledgers
before forwarding a reservation to the study's own accounting authority.
"""

import json
from pathlib import Path

from test.lib import tensor_execution
from test.lib import tensor_study_accounting as accounting


def _require(condition, message):
  if not condition:
    raise AssertionError(message)


def run_batch(root, operations, *, driver, caps, total_caps):
  root = Path(root)
  root.mkdir(parents=True, exist_ok=False)
  names = ("qualification", "refused-p2", "refused-p0")
  record = {
    "batch": "moving-window-closure",
    "status": "running",
    "caps": dict(total_caps),
    "toolchain": operations["toolchain"],
    "scenarios": [
      {"scenario": name, "status": "pending", "caps": caps[name]} for name in names
    ],
  }
  active = None

  def events(name):
    path = root / name / "path-ledger.jsonl"
    if not path.exists():
      return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

  def starts(name):
    return {
      path: sum(
        e.get("kind") == "process"
        and e.get("event") == "start"
        and e.get("path") == path
        for e in events(name)
      )
      for path in accounting.PATHS
    }

  def held(name, path):
    # Normalize only in this derived view; never edit an execution ledger.
    rows = [
      {**e, "phase": "qualification"}
      for e in events(name)
      if e.get("kind") == "process"
    ]
    return accounting.used(rows, "qualification", path)

  def save():
    record["total_starts"] = {path: 0 for path in accounting.PATHS}
    for scenario in record["scenarios"]:
      scenario["starts"] = starts(scenario["scenario"])
      for path in accounting.PATHS:
        record["total_starts"][path] += scenario["starts"][path]
    # This file is a derived index; the scenario ledgers retain original events.
    rows = [
      {"scenario": name, **event}
      for name in names
      for event in events(name)
      if event.get("kind") in ("process", "launch_refused")
    ]
    tensor_execution.write_atomic(
      root / "processes.jsonl",
      ("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)).encode(),
    )
    tensor_execution.write_atomic(
      root / "validation.json",
      (json.dumps(record, indent=1, sort_keys=True) + "\n").encode(),
    )

  def execute(request):
    callback = request["on_process"]
    name = active["scenario"]

    def guard(event):
      path = event["path"]
      if event["event"] == "reserved":
        expected = (
          name == "refused-p2"
          and path == "p2"
          and request["configuration"] == "warps-4"
          and request["attempt"] == 0
        ) or (name == "refused-p0" and path == "p0" and request["attempt"] == 0)
        if expected and not active.get("injected_refusal"):
          active["injected_refusal"] = {"path": path, "request": request["slot"]}
          save()
          raise tensor_execution.DispatchRefused("declared validation refusal control")
        if (
          held(name, path) >= caps[name][path]
          or sum(held(n, path) for n in names) >= total_caps[path]
        ):
          active["budget_refusal"] = {"path": path, "request": request["slot"]}
          save()
          raise tensor_execution.DispatchRefused(
            "repair validation allocation exhausted"
          )
      try:
        callback(event)
      finally:
        save()

    return operations["execute"]({**request, "on_process": guard})

  wrapped = {**operations, "execute": execute}

  def scenario(name):
    work = root / name
    work.mkdir()
    frozen = driver.freeze_study(toolchain=operations["toolchain"])
    frozen.write(work / "study.json")
    authority = driver.Study(
      work,
      frozen,
      build=operations["build"],
      screen=operations.get("screen"),
      execute=execute,
      accounting_mode="process",
    )
    return work, frozen, authority

  def run(work, authority, **kwargs):
    result = driver.run_qualification(
      work, operations=wrapped, authority=authority, **kwargs
    )
    _require(not active.get("budget_refusal"), "validation allocation was exhausted")
    _require(
      not any(
        e.get("kind") == "qualification_outcome" and e["outcome"].get("candidate")
        for e in authority.path_ledger.events
      ),
      "qualification produced a candidate; stop the validation batch for triage",
    )
    return result

  def resume(authority):
    _require(
      authority.is_paused(), "expected refusal did not durably pause qualification"
    )
    authority.record_triage_decision(
      {"action": "resume", "reason": "declared refusal control"}
    )

  save()
  try:
    for active in record["scenarios"]:
      name = active["scenario"]
      active["status"] = "running"
      save()
      work, frozen, authority = scenario(name)
      if name == "qualification":
        result = run(work, authority)
        driver.require_qualification(work, frozen)
        before = starts(name)
        run(work, authority)
        _require(starts(name) == before, "qualification resume launched a worker")
        active.update(
          validated=result["validated"], qualification_digest=result["digest"]
        )
      elif name == "refused-p2":
        result = run(work, authority, slots=1)
        _require(not result["cases"][0]["complete"], "P2 refusal was not retained")
        before = starts(name)
        _require(before == {"p0": 1, "p1": 2, "p2": 1}, "unexpected partial-run starts")
        cache = work / "mw-g0c0" / "p0.bin"
        cache.rename(cache.with_suffix(".bin.aside"))
        resume(authority)
        retry = "mw-g0c0/warps-4/attempt-0"
        retried = run(work, authority, slots=1, retry=retry)
        after = starts(name)
        _require(
          {p: after[p] - before[p] for p in after} == {"p0": 0, "p1": 1, "p2": 1},
          "retry did not preserve cached P0 unavailability",
        )
        _require(
          not cache.exists() and cache.with_suffix(".bin.aside").exists(),
          "retry replaced the missing reference",
        )
        _require(retried["cases"][0]["complete"], "retry did not complete both targets")
        active.update(retry=retry, p0_cache="moved-aside")
      else:
        result = run(work, authority, slots=1)
        _require(not result["cases"][0]["complete"], "P0 refusal was not retained")
        _require(
          starts(name) == {"p0": 0, "p1": 0, "p2": 0}, "P0 control launched a child"
        )
        resume(authority)
        retry = "mw-g0c0/warps-1/attempt-0"
        retried = run(work, authority, slots=1, retry=retry)
        _require(
          retried["cases"][0]["complete"], "cached-unavailable targets did not complete"
        )
        _require(
          not retried["cases"][0]["validated"], "missing P0 became accepted reference"
        )
        _require("p0_unavailable" in retried["cases"][0], "missing P0 was not reported")
        active.update(retry=retry, p0_cache="unavailable")
      _require(
        starts(name) == caps[name],
        "scenario does not match its declared execution count",
      )
      active["status"] = "completed"
      save()
    record["status"] = "passed"
  except BaseException as error:
    status = (
      "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
    )
    record["status"] = status
    record["reason"] = f"{type(error).__name__}: {error}"
    if active is not None:
      active["status"] = status
    raise
  finally:
    save()
  return record
