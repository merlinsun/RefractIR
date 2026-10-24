"""Frozen moving-window discovery study.

The study is frozen before any builder call: twelve source slots, four
archived controls, four input probes, two warp settings, eight seeds per slot,
the probe rules, and the path budgets. The driver reuses the tranche ledger for
the 128 paired slots and adds an append-only path ledger for the work outside
those slots (selection attempts, screening, P0/P1/P2 launches, cached
observations, retries, and refusals). Nothing here deletes evidence: a spent
reserve stops launches and leaves the record.

The builder, screener, and executor are injected so a host fixture can exercise
the accounting without a GPU; the campaign binds them to the exporter, the
interpreter-only runner, and the paired runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from test.lib import tensor_tranche

MIN_FREE_BYTES = 2 * 1024**3
SCHEMA = "refractir.moving-window-study.v1"
FINAL_SCHEMA = "refractir.moving-window-final.v1"
CORPUS_SCHEMA = "refractir.moving-window-corpus.v1"
LANES = ("G0", "G1", "G2", "control")
PROBES = ("i0-base", "i1-read-plus-one", "i2-late-row", "i3-unused-rows")
CONFIGURATIONS = ("warps-1", "warps-4")
POLICIES = ("unguided", "deep", "wide", "shared", "reduction-heavy", "cross-state")
COLUMNS = (
  {"column": 0, "shape": [2, 8], "trips": 4, "stride": 1, "backing_rows": 7, "active": 5, "regime": "interior"},
  {"column": 1, "shape": [3, 4], "trips": 4, "stride": 1, "backing_rows": 8, "active": 5, "regime": "tail"},
  {"column": 2, "shape": [4, 4], "trips": 8, "stride": 2, "backing_rows": 20, "active": 18, "regime": "interior"},
  {"column": 3, "shape": [3, 8], "trips": 8, "stride": 2, "backing_rows": 19, "active": 15, "regime": "tail"},
)
CONTROL_PATTERNS = {
  "i0-base": "distinct-integer-0",
  "i1-read-plus-one": "mixed-sign-1",
  "i2-late-row": "repeated-value-2",
  "i3-unused-rows": "seeded-random-3",
}
CONTROLS = (
  {"id": "control-e7-unguided-1", "source": "e7-unguided-1-20296002"},
  {"id": "control-e7-shared-1", "source": "e7-shared-1-20296052"},
  {"id": "control-e7-deep-3", "source": "e7-deep-3-20296023"},
  {"id": "control-e7-shared-3", "source": "e7-shared-3-20296060"},
)
ATTEMPT_BASE = 20261023
ATTEMPTS_PER_SLOT = 8
INTERNAL_MAX_ATTEMPTS = 1
PROBE_RULES = {
  "schedule": [
    "i0-base/T", "i1-read-plus-one/T", "i2-late-row/T", "i3-unused-rows/T",
    "i0-base/T-1", "i2-late-row/T-1",
  ],
  "conditional_cap": 4,
  "launch_cap": 10,
  "select_when": {
    "i0_differs_from": ["i1-read-plus-one", "i2-late-row"],
    "i0_agrees_with": ["i3-unused-rows"],
    "trip_changes": True,
  },
}
BUDGETS = {
  "qualification_gpu": 8,
  "qualification_interpreter": 8,
  "qualification_p0": 4,
  "selection_builder_calls": 96,
  "selection_interpreter": 960,
  "selection_p0": 384,
  "discovery_gpu": 128,
  "discovery_interpreter": 128,
  "discovery_p0": 16,
  "triage_gpu": 32,
  "triage_interpreter": 32,
  "triage_p0": 32,
  "gpu_ceiling": 160,
  "reserve_gpu": 32,
  "reserve_interpreter": 32,
  "reserve_p0": 32,
  "attempts_per_slot": ATTEMPTS_PER_SLOT,
}


class StudyRefused(RuntimeError):
  """The study cannot proceed as frozen."""


def _canonical(value):
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value):
  return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sha256_bytes(data):
  return hashlib.sha256(data).hexdigest()


def source_slots():
  """The twelve source slots: three requirement groups by four columns."""
  slots = []
  for group in range(3):
    for column in range(4):
      requirement = (
        "reduction" if group == 0 else
        "shared" if group == 1 else
        "extrema-transpose" if column % 2 == 0 else "extrema-expand"
      )
      spec = dict(COLUMNS[column])
      slots.append(
        {
          "id": f"mw-g{group}c{column}",
          "lane": f"G{group}",
          "index": 4 * group + column,
          "group": group,
          "requirement": requirement,
          "policy": POLICIES[(4 * group + column) % len(POLICIES)],
          "padded": (group + column) % 2 == 1,
          **spec,
        }
      )
  return tuple(slots)


SOURCE_SLOTS = source_slots()


def study_spec():
  """The frozen study specification."""
  return {
    "schema": SCHEMA,
    "lanes": list(LANES),
    "sources": [dict(slot) for slot in SOURCE_SLOTS],
    "controls": [dict(control) for control in CONTROLS],
    "probes": list(PROBES),
    "configurations": list(CONFIGURATIONS),
    "attempts": {
      "per_slot": ATTEMPTS_PER_SLOT,
      "seed_base": ATTEMPT_BASE,
      "internal_max_attempts": INTERNAL_MAX_ATTEMPTS,
    },
    "probe_rules": PROBE_RULES,
    "budgets": dict(BUDGETS),
  }


def _validate_spec(spec):
  if not isinstance(spec, dict):
    raise tensor_tranche.ManifestRefused("a study spec must be an object")
  if spec.get("schema") != SCHEMA:
    raise tensor_tranche.ManifestRefused("unknown study schema")
  if spec.get("lanes") != list(LANES):
    raise tensor_tranche.ManifestRefused("the study lanes changed")
  sources = spec.get("sources")
  controls = spec.get("controls")
  if not isinstance(sources, list) or len(sources) != 12:
    raise tensor_tranche.ManifestRefused("the study needs twelve source slots")
  if not isinstance(controls, list) or len(controls) != 4:
    raise tensor_tranche.ManifestRefused("the study needs four controls")
  if spec.get("probes") != list(PROBES):
    raise tensor_tranche.ManifestRefused("the study probe list changed")
  if spec.get("configurations") != list(CONFIGURATIONS):
    raise tensor_tranche.ManifestRefused("the study configurations changed")
  attempts = spec.get("attempts")
  if not isinstance(attempts, dict) or attempts.get("per_slot") != ATTEMPTS_PER_SLOT:
    raise tensor_tranche.ManifestRefused("the study attempt count changed")
  if attempts.get("seed_base") != ATTEMPT_BASE:
    raise tensor_tranche.ManifestRefused("the study seed base changed")
  if attempts.get("internal_max_attempts") != INTERNAL_MAX_ATTEMPTS:
    raise tensor_tranche.ManifestRefused("the internal attempt limit changed")
  if spec.get("probe_rules") != PROBE_RULES:
    raise tensor_tranche.ManifestRefused("the study probe rules changed")
  if spec.get("budgets") != BUDGETS:
    raise tensor_tranche.ManifestRefused("the study budgets changed")
  seen = set()
  for slot in sources:
    if slot.get("id") in seen:
      raise tensor_tranche.ManifestRefused("duplicate source slot")
    seen.add(slot.get("id"))
    if slot.get("lane") not in LANES[:3]:
      raise tensor_tranche.ManifestRefused("a source slot names an unknown lane")
    if slot.get("requirement") not in ("reduction", "shared", "extrema-transpose", "extrema-expand"):
      raise tensor_tranche.ManifestRefused("a source slot names an unknown requirement")
    if slot.get("policy") not in POLICIES:
      raise tensor_tranche.ManifestRefused("a source slot names an unknown policy")
    for field in ("column", "trips", "stride", "backing_rows", "active"):
      if type(slot.get(field)) is not int:
        raise tensor_tranche.ManifestRefused(f"a source slot has no integer {field}")
    if slot["trips"] < 2 or slot["stride"] < 0 or slot["active"] < 1:
      raise tensor_tranche.ManifestRefused("a source slot has an invalid window")
    if (slot["trips"] - 1) * slot["stride"] + slot["shape"][0] > slot["backing_rows"]:
      raise tensor_tranche.ManifestRefused("a source slot leaves its backing witness")
  for control in controls:
    if control.get("lane", "control") != "control":
      raise tensor_tranche.ManifestRefused("a control names another lane")
    if not control.get("id") or not control.get("source"):
      raise tensor_tranche.ManifestRefused("a control needs an id and an archived source")


class FrozenStudy:
  """A frozen spec, its digest, and the tranche slots it expands to."""

  def __init__(self, spec, digest, tranche):
    self.spec = spec
    self.digest = digest
    self.tranche = tranche

  def as_record(self):
    return {
      "schema": SCHEMA,
      "digest": self.digest,
      "spec": self.spec,
      "tranche": self.tranche.as_record(),
    }

  def write(self, path):
    Path(path).write_text(json.dumps(self.as_record(), indent=1) + "\n", encoding="utf-8")
    return Path(path)

  @classmethod
  def load(cls, path):
    record = tensor_tranche.read(path)
    spec = record.get("spec")
    digest = record.get("digest")
    if not isinstance(spec, dict) or digest != _digest(spec):
      raise tensor_tranche.ManifestRefused("the study spec changed after it was frozen")
    _validate_spec(spec)
    tranche = tensor_tranche.FrozenTranche.from_payload(record.get("tranche"))
    return cls(spec=spec, digest=digest, tranche=tranche)


def freeze_study(spec=None, *, toolchain=None):
  spec = study_spec() if spec is None else spec
  if toolchain is not None:
    spec = {**spec, "toolchain": dict(toolchain)}
  _validate_spec(spec)
  cases = [
    {
      "id": slot["id"],
      "lane": slot["lane"],
      "inputs": list(spec["probes"]),
      "configurations": list(spec["configurations"]),
    }
    for slot in spec["sources"]
  ] + [
    {
      "id": control["id"],
      "lane": "control",
      "inputs": list(spec["probes"]),
      "configurations": list(spec["configurations"]),
    }
    for control in spec["controls"]
  ]
  manifest = {
    "tranche": "moving-window-2026-10-23",
    "lanes": list(spec["lanes"]),
    "study": spec["schema"],
    "budget": {
      "executions": spec["budgets"]["discovery_gpu"],
      "controls": len(spec["controls"]) * len(spec["probes"]) * len(spec["configurations"]),
    },
    "cases": cases,
  }
  return FrozenStudy(spec=spec, digest=_digest(spec), tranche=tensor_tranche.freeze(manifest))


class PathLedger:
  """An append-only event log for work outside the frozen paired slots."""

  def __init__(self, path):
    self.path = Path(path)
    self.events = []

  def load(self):
    self.events = []
    if self.path.is_file():
      for line in self.path.read_text(encoding="utf-8").splitlines():
        if line.strip():
          self.events.append(json.loads(line))
    return self

  def append(self, kind, **fields):
    event = {"kind": kind, **fields}
    self.events.append(event)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    with self.path.open("a", encoding="utf-8") as handle:
      handle.write(json.dumps(event, sort_keys=True) + "\n")
    return event


def path_counts(events):
  """Launches, cached observations, failures, and statuses by path."""
  counts = {
    path: {"launches": 0, "cached": 0, "failures": 0, "statuses": {}}
    for path in ("p0", "p1", "p2")
  }
  for event in events:
    if event.get("kind") == "launch":
      path = event.get("path")
      if path not in counts:
        continue
      counts[path]["launches"] += 1
      status = event.get("status", "unknown")
      counts[path]["statuses"][status] = counts[path]["statuses"].get(status, 0) + 1
      if status != "completed":
        counts[path]["failures"] += 1
    elif event.get("kind") == "cached":
      path = event.get("path")
      if path in counts:
        counts[path]["cached"] += 1
  return counts


class Study:
  """One run of the frozen study: selection, discovery, and accounting."""

  def __init__(self, work, frozen, *, build, screen, execute, controls_root=None,
               triage_ledger=None):
    self.work = Path(work)
    self.frozen = frozen
    self.build = build
    self.screen = screen
    self.execute = execute
    self.controls_root = None if controls_root is None else Path(controls_root)
    self.triage_ledger = None if triage_ledger is None else Path(triage_ledger)
    self.path_ledger = PathLedger(self.work / "path-ledger.jsonl")
    self.corpus_path = self.work / "corpus.jsonl"
    self.ledger = None

  def prepare(self):
    self.work.mkdir(parents=True, exist_ok=True)
    study_path = self.work / "study.json"
    if study_path.is_file():
      existing = FrozenStudy.load(study_path)
      if existing.digest != self.frozen.digest:
        raise tensor_tranche.ManifestRefused(
          "the work directory belongs to another frozen study"
        )
    else:
      self.frozen.write(study_path)
    self.path_ledger.load()
    self.ledger = tensor_tranche.Ledger.open(
      self.work / "tranche-ledger.jsonl", self.frozen.tranche
    )
    for slot in self.frozen.tranche.slots:
      if self.ledger.state(slot.slot_id) == "running":
        self.ledger.finish(slot.slot_id, "failed", reason="interrupted")
        self.path_ledger.append(
          "reconciled", slot=slot.slot_id, state="failed", reason="interrupted"
        )
    return self

  def corpus(self):
    if not self.corpus_path.is_file():
      return []
    return [
      json.loads(line)
      for line in self.corpus_path.read_text(encoding="utf-8").splitlines()
      if line.strip()
    ]

  def _selected_ids(self):
    return {entry["case"] for entry in self.corpus()}

  def is_paused(self):
    return (self.work / "study-paused.json").is_file()

  def pause(self, candidates, *, reason):
    """Persist candidates and stop routine dispatch until triage decides."""
    record = {
      "reason": reason,
      "candidates": candidates,
    }
    (self.work / "study-paused.json").write_text(
      json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    self.path_ledger.append("paused", reason=reason, candidates=len(candidates))

  def record_triage_decision(self, decision):
    """Apply a validated disposition and only then clear the pause."""
    if not self.is_paused():
      raise StudyRefused("the study is not paused")
    record = dict(decision) if isinstance(decision, dict) else {"action": decision}
    action = record.get("action")
    if action not in ("admit", "reject", "resume", "stop"):
      raise StudyRefused(f"unknown triage action: {action}")
    pending = self._pending_admission()
    if action in ("admit", "reject") and pending is None:
      raise StudyRefused("no pending admission to decide")
    record = {
      **record,
      "case": (pending or {}).get("case"),
      "attempt": (pending or {}).get("attempt"),
    }
    path = self.work / "triage-decisions.jsonl"
    with path.open("a", encoding="utf-8") as handle:
      handle.write(json.dumps(record, sort_keys=True) + "\n")
    self.path_ledger.append("triage_decision", decision=record)
    if action == "admit":
      self._admit_pending()
    elif action == "reject":
      self.path_ledger.append(
        "triage_rejected", case=pending["case"], attempt=pending["attempt"]
      )
      self._clear_pending()
    elif action == "stop":
      (self.work / "study-stopped.json").write_text(
        json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8"
      )
    (self.work / "study-paused.json").unlink()
    return record

  def _paused_reason(self):
    path = self.work / "study-paused.json"
    if not path.is_file():
      return None
    return json.loads(path.read_text(encoding="utf-8")).get("reason")

  def _shortfalls(self):
    return [e["case"] for e in self.path_ledger.events if e.get("kind") == "shortfall"]

  def _selection_closed(self):
    return len(self._selected_ids()) + len(self._shortfalls()) >= len(
      self.frozen.spec["sources"]
    )

  def _corpus_entry(self, slot, outcome, screened):
    return {
      "case": slot["id"],
      "lane": slot["lane"],
      "kind": "source",
      "source_sha256": outcome.get("source_sha256"),
      "inputs_sha256": outcome.get("inputs_sha256"),
      "layouts": outcome.get("layouts", {}),
      "witnesses": outcome.get("witnesses", []),
      "region": outcome.get("region"),
      "source": outcome.get("source"),
      "trip_variant": outcome.get("trip_variant"),
      "probe_values": outcome.get("probe_values"),
      "bundles": outcome.get("bundles"),
      "compiler_identity": outcome.get("compiler_identity"),
      "screen": screened.get("record", {}),
    }

  def _pending_path(self):
    return self.work / "pending-admission.json"

  def _pending_admission(self):
    path = self._pending_path()
    if not path.is_file():
      return None
    return json.loads(path.read_text(encoding="utf-8"))

  def _save_pending(self, pending):
    self._pending_path().write_text(
      json.dumps(pending, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

  def _clear_pending(self):
    self._pending_path().unlink(missing_ok=True)

  def _admit_pending(self):
    pending = self._pending_admission()
    if pending is None:
      return False
    entry = pending["entry"]
    if entry["case"] not in self._selected_ids():
      self.path_ledger.append(
        "corpus", case=entry["case"],
        source_sha256=entry.get("source_sha256"),
        inputs_sha256=entry.get("inputs_sha256"),
      )
      with self.corpus_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    self._clear_pending()
    return True

  def _decisions(self):
    path = self.work / "triage-decisions.jsonl"
    if not path.is_file():
      return []
    return [
      json.loads(line)
      for line in path.read_text(encoding="utf-8").splitlines()
      if line.strip()
    ]

  def _reconcile_pending(self):
    pending = self._pending_admission()
    if pending is None:
      return
    decisions = [
      decision for decision in self._decisions()
      if decision.get("case") == pending.get("case")
      and decision.get("attempt") == pending.get("attempt")
    ]
    if not decisions:
      raise StudyRefused(
        f"a pending admission for {pending.get('case')} needs a triage decision"
      )
    action = decisions[-1].get("action")
    if action == "admit":
      self._admit_pending()
    elif action == "reject":
      self._clear_pending()
    elif action == "stop":
      raise StudyRefused("the study is stopped")
    else:
      raise StudyRefused(f"unsupported triage action for a pending admission: {action}")

  def is_stopped(self):
    return (self.work / "study-stopped.json").is_file()

  def _disk_guard(self):
    free = shutil.disk_usage(self.work).free
    if free < MIN_FREE_BYTES:
      self.path_ledger.append("disk_guard", free_bytes=free, required=MIN_FREE_BYTES)
      self.pause(
        [{"kind": "disk_space", "free_bytes": free, "required": MIN_FREE_BYTES}],
        reason="free disk space is below the dispatch guard",
      )
      return False
    return True

  def select_sources(self, max_slots=None):
    """Run the bounded builder calls and screen every admitted body.

    A selected or shortfall slot is settled: resume never rebuilds it and never
    re-rolls a shortfall. `max_slots` is a host-fixture hook for interrupting a
    selection run; the campaign passes no limit.
    """
    if self.is_paused():
      raise StudyRefused(f"the study is paused: {self._paused_reason()}")
    if self.is_stopped():
      raise StudyRefused("the study is stopped")
    self._reconcile_pending()
    if not self._disk_guard():
      return self
    if (self.work / "corpus-frozen.json").is_file():
      raise StudyRefused("the corpus is already frozen")
    selected = self._selected_ids()
    shortfalls = set(self._shortfalls())
    processed = 0
    for slot in self.frozen.spec["sources"]:
      if slot["id"] in selected or slot["id"] in shortfalls:
        continue
      if max_slots is not None and processed >= max_slots:
        break
      processed += 1
      admitted = None
      # An interrupted reservation already consumed its ordinal. It is never
      # retried automatically; the next ordinal continues the bounded search.
      for reservation in self._unfinished_reservations(slot["id"]):
        self.path_ledger.append(
          "selection_attempt",
          case=slot["id"],
          attempt=reservation["attempt"],
          seed=reservation.get("seed"),
          outcome="interrupted",
          reason="reservation has no terminal outcome",
        )
      used = self._attempts_used(slot["id"])
      for attempt in range(self.frozen.spec["attempts"]["per_slot"]):
        if attempt in used:
          continue
        if not self._disk_guard():
          return self
        seed = (
          self.frozen.spec["attempts"]["seed_base"] + 8 * slot["index"] + attempt
        )
        directory = self.work / "selection" / slot["id"] / f"attempt-{attempt}"
        if directory.exists():
          raise StudyRefused(f"selection attempt directory already exists: {directory}")
        self.path_ledger.append(
          "selection_attempt_reserved",
          case=slot["id"],
          attempt=attempt,
          seed=seed,
          directory=str(directory),
        )
        directory.mkdir(parents=True)
        used.add(attempt)
        request = {
          "slot": dict(slot),
          "attempt": attempt,
          "seed": seed,
          "max_attempts": self.frozen.spec["attempts"]["internal_max_attempts"],
          "directory": str(directory),
        }
        try:
          outcome = self.build(request)
        except Exception as error:  # a builder crash is an attempt result
          outcome = {"kind": "error", "reason": f"{type(error).__name__}: {error}"}
        self.path_ledger.append(
          "selection_attempt",
          case=slot["id"],
          attempt=attempt,
          seed=seed,
          outcome=outcome.get("kind", "error"),
          reason=outcome.get("reason"),
        )
        if outcome.get("kind") != "admitted":
          continue
        outcome["on_process"] = lambda event, case=slot["id"], attempt=attempt:           self.path_ledger.append(
            "process", phase="screening", case=case, attempt=attempt, **event
          )
        screened = self.screen(outcome)
        self.path_ledger.append(
          "selection_screen",
          case=slot["id"],
          selected=bool(screened.get("selected")),
          reason=screened.get("reason"),
          launches=screened.get("launches", 0),
          p0=screened.get("p0", 0),
        )
        candidates = list(screened.get("candidates", []))
        invalid = [c for c in candidates if c.get("kind") == "invalid_execution"]
        if invalid:
          self.path_ledger.append(
            "selection_invalid_execution", case=slot["id"], candidates=invalid
          )
          candidates = [c for c in candidates if c.get("kind") != "invalid_execution"]
        if candidates:
          self.path_ledger.append(
            "selection_candidates",
            case=slot["id"],
            candidates=candidates,
          )
          self._save_pending(
            {
              "case": slot["id"],
              "attempt": attempt,
              "seed": seed,
              "entry": self._corpus_entry(slot, outcome, screened),
              "candidates": candidates,
            }
          )
          self.pause(
            candidates,
            reason=f"screening candidates for {slot['id']}",
          )
          return self
        if screened.get("unavailable"):
          self.path_ledger.append(
            "selection_unavailable", case=slot["id"],
            unavailable=screened["unavailable"],
          )
        if screened.get("selected"):
          admitted = self._corpus_entry(slot, outcome, screened)
          break
      if admitted is None:
        self.path_ledger.append(
          "shortfall", case=slot["id"], reason="attempts_exhausted"
        )
        continue
      self.path_ledger.append(
        "corpus",
        case=slot["id"],
        source_sha256=admitted["source_sha256"],
        inputs_sha256=admitted["inputs_sha256"],
      )
      with self.corpus_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(admitted, sort_keys=True) + "\n")
      selected.add(slot["id"])
    return self

  def diversity_report(self):
    """Per-group admitted bodies and distinct connectivity signatures."""
    groups = {}
    for entry in self.corpus():
      lane = entry.get("lane")
      if lane not in ("G0", "G1", "G2"):
        continue
      signature = (
        (entry.get("region") or {}).get("live_dag") or {}
      ).get("connectivity_signature")
      group = groups.setdefault(lane, {"bodies": 0, "signatures": []})
      group["bodies"] += 1
      if signature:
        group["signatures"].append(signature)
    report = {}
    for lane in ("G0", "G1", "G2"):
      group = groups.get(lane, {"bodies": 0, "signatures": []})
      distinct = len(set(group["signatures"]))
      report[lane] = {
        "bodies": group["bodies"],
        "distinct_signatures": distinct,
        "sufficient": group["bodies"] >= 3 and distinct >= 3,
      }
    report["sufficient"] = all(report[lane]["sufficient"] for lane in ("G0", "G1", "G2"))
    return report

  def freeze_corpus(self):
    """Freeze the admitted sources and every archived control before dispatch."""
    if not self._selection_closed():
      raise StudyRefused("the corpus cannot freeze before every source slot is settled")
    entries = self.corpus()
    if not entries:
      raise StudyRefused("the corpus has no admitted source")
    controls = []
    for control in self.frozen.spec["controls"]:
      entry = {
        "case": control["id"],
        "lane": "control",
        "kind": "control",
        "source": control["source"],
        "patterns": dict(CONTROL_PATTERNS),
      }
      if self.controls_root is None:
        entry["status"] = "missing"
        entry["reason"] = "no control archive was supplied"
      else:
        hashes = {}
        missing = []
        for probe, pattern in CONTROL_PATTERNS.items():
          bundles = [
            self.controls_root / control["source"] / f"{pattern}-warps-{warps}"
            for warps in (1, 4)
          ]
          if not all(bundle.is_dir() for bundle in bundles):
            missing.append(f"{control['source']}/{pattern}")
            continue
          program_sha = hashlib.sha256(
            (bundles[0] / "program.sir").read_bytes()
          ).hexdigest()
          inputs_sha = hashlib.sha256(
            (bundles[0] / "inputs.bin").read_bytes()
          ).hexdigest()
          case_hashes = {}
          for warps, bundle in zip((1, 4), bundles):
            case_hashes[str(warps)] = hashlib.sha256(
              (bundle / "case.json").read_bytes()
            ).hexdigest()
          for bundle in bundles[1:]:
            if hashlib.sha256((bundle / "program.sir").read_bytes()).hexdigest() != program_sha:
              raise StudyRefused(f"archived control program changed across warps: {bundle}")
            if hashlib.sha256((bundle / "inputs.bin").read_bytes()).hexdigest() != inputs_sha:
              raise StudyRefused(f"archived control input changed across warps: {bundle}")
          hashes[probe] = {
            "pattern": pattern,
            "program_sha256": program_sha,
            "inputs_sha256": inputs_sha,
            "case_sha256": case_hashes,
          }
        entry["hashes"] = hashes
        if missing:
          entry["status"] = "missing"
          entry["reason"] = "missing archived controls: " + ", ".join(missing)
      controls.append(entry)
    record = {
      "schema": CORPUS_SCHEMA,
      "sources": entries,
      "controls": controls,
      "shortfalls": self._shortfalls(),
      "diversity": self.diversity_report(),
    }
    record["digest"] = _digest(record)
    (self.work / "corpus-frozen.json").write_text(
      json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record

  def frozen_corpus(self):
    path = self.work / "corpus-frozen.json"
    if not path.is_file():
      raise StudyRefused("the corpus has not been frozen")
    record = json.loads(path.read_text(encoding="utf-8"))
    digest = record.pop("digest")
    if digest != _digest(record):
      raise StudyRefused("the frozen corpus changed after its digest")
    record["digest"] = digest
    return record

  ATTEMPT_KINDS = (
    "selection_attempt", "selection_attempt_reserved", "launch",
    "launch_reserved", "retry", "slot_started",
  )

  def _attempts_used(self, case_id):
    return {
      event["attempt"]
      for event in self.path_ledger.events
      if event.get("case") == case_id
      and event.get("kind") in self.ATTEMPT_KINDS
      and type(event.get("attempt")) is int
    }

  def _unfinished_reservations(self, case_id):
    finished = {
      event["attempt"]
      for event in self.path_ledger.events
      if event.get("case") == case_id and event.get("kind") == "selection_attempt"
    }
    return [
      event for event in self.path_ledger.events
      if event.get("case") == case_id
      and event.get("kind") == "selection_attempt_reserved"
      and event["attempt"] not in finished
    ]

  IMMUTABLE_FIELDS = (
    "source_sha256", "inputs_sha256", "layouts", "witnesses", "region",
    "probes", "compiler_identity",
  )

  def corpus_for_report(self):
    """The frozen corpus; selection history is cross-checked, never returned."""
    frozen = self.frozen_corpus()
    selected = self.corpus()
    if not selected:
      return frozen["sources"]
    frozen_by = {entry["case"]: entry for entry in frozen["sources"]}
    for entry in selected:
      frozen_entry = frozen_by.get(entry.get("case"))
      if frozen_entry is None:
        raise StudyRefused(
          f"selection history names a case outside the frozen corpus: {entry.get('case')}"
        )
      for field in self.IMMUTABLE_FIELDS:
        if entry.get(field) != frozen_entry.get(field):
          raise StudyRefused(
            f"selection history contradicts the frozen {field} for {entry['case']}"
          )
    if len(selected) != len(frozen["sources"]):
      raise StudyRefused("selection history and the frozen corpus disagree on source count")
    return frozen["sources"]

  def _next_attempt(self, slot_id):
    used = [
      event["attempt"]
      for event in self.path_ledger.events
      if event.get("slot") == slot_id
      and event.get("kind") in self.ATTEMPT_KINDS
      and type(event.get("attempt")) is int
    ]
    return max(used, default=-1) + 1

  def run_discovery(self):
    """Execute every frozen paired slot that has a corpus entry."""
    if self.ledger is None:
      raise StudyRefused("prepare the study before dispatch")
    if self.is_paused():
      raise StudyRefused(f"the study is paused: {self._paused_reason()}")
    if self.is_stopped():
      raise StudyRefused("the study is stopped")
    if not self._disk_guard():
      return self
    corpus = self.frozen_corpus()
    if not corpus.get("diversity", {}).get("sufficient", False):
      raise StudyRefused("the generator diversity minimum is not met")
    available = {entry["case"] for entry in corpus["sources"] + corpus["controls"]}
    cases = {
      case["id"]: case
      for case in self.frozen.tranche.payload["manifest"]["cases"]
    }
    for slot in self.frozen.tranche.slots:
      if self.ledger.state(slot.slot_id) != "pending":
        continue
      if not self._disk_guard():
        return self
      if slot.case_id not in available:
        continue
      corpus_entry = next(
        entry for entry in corpus["sources"] + corpus["controls"]
        if entry["case"] == slot.case_id
      )
      if corpus_entry.get("status") == "missing":
        self.path_ledger.append(
          "control_missing", case=slot.case_id, slot=slot.slot_id,
          reason=corpus_entry.get("reason"),
        )
        continue
      self.ledger.claim(slot.slot_id)
      attempt = self._next_attempt(slot.slot_id)
      directory = (
        self.work
        / "discovery"
        / slot.case_id
        / slot.input
        / slot.configuration
        / f"attempt-{attempt}"
      )
      if directory.exists():
        raise StudyRefused(f"discovery attempt directory exists: {directory}")
      directory.mkdir(parents=True)
      self.path_ledger.append(
        "slot_started",
        slot=slot.slot_id,
        case=slot.case_id,
        attempt=attempt,
        directory=str(directory),
      )
      def account(event):
        self.path_ledger.append(
          "process", phase="discovery", slot=slot.slot_id, case=slot.case_id,
          attempt=attempt, directory=str(directory), **event,
        )

      request = {
        "slot": slot.slot_id,
        "case": cases[slot.case_id],
        "corpus": next(
          entry for entry in corpus["sources"] + corpus["controls"]
          if entry["case"] == slot.case_id
        ),
        "input": slot.input,
        "configuration": slot.configuration,
        "attempt": attempt,
        "directory": str(directory),
        "on_process": account,
      }
      try:
        outcome = self.execute(request)
      except Exception as error:
        self.ledger.finish(slot.slot_id, "failed", reason=f"{type(error).__name__}: {error}")
        classification = {
          "slot": slot.slot_id,
          "case": slot.case_id,
          "status": "driver_error",
          "candidate": True,
          "unavailable": [f"driver error: {type(error).__name__}: {error}"],
          "compiler": {"status": "unavailable"},
        }
        self.path_ledger.append("classification", **classification)
        self.pause(
          [classification],
          reason=f"infrastructure error during discovery at {slot.slot_id}",
        )
        return self
      for path in ("p0", "p1", "p2"):
        info = outcome.get("paths", {}).get(path)
        if info is None:
          continue
        if info.get("cached"):
          self.path_ledger.append(
            "cached", path=path, slot=slot.slot_id, case=slot.case_id,
            attempt=attempt, key=info.get("key"),
          )
        else:
          self.path_ledger.append(
            "launch", path=path, slot=slot.slot_id, case=slot.case_id,
            attempt=attempt, directory=info.get("directory", str(directory)),
            status=info.get("status", "completed"),
          )
      classification = {
        "slot": slot.slot_id,
        "case": slot.case_id,
        "status": outcome.get("status", "completed"),
        "candidate": bool(outcome.get("candidate")),
        "unavailable": list(outcome.get("unavailable", [])),
        "compiler": outcome.get("compiler", {"status": "unavailable"}),
      }
      self.path_ledger.append("classification", **classification)
      self.ledger.finish(slot.slot_id, "completed", detail={"status": outcome.get("status")})
      if classification["candidate"]:
        self.pause(
          [classification],
          reason=f"candidate during discovery at {slot.slot_id}",
        )
        return self
    return self

  def triage_charge(self):
    """The documented triage attempts imported into the executable reserve."""
    if self.triage_ledger is None or not self.triage_ledger.is_file():
      return 0
    return sum(
      1 for line in self.triage_ledger.read_text(encoding="utf-8").splitlines()
      if line.strip()
    )

  def reserve_spent(self):
    return sum(
      1 for e in self.path_ledger.events if e.get("kind") == "retry"
    ) + self.triage_charge()

  def retry(self, slot_id, *, case_id, input, configuration, reason):
    """Reserve a unique retry directory, charging every paired-path reserve."""
    if not self._disk_guard():
      return None
    budgets = self.frozen.spec["budgets"]
    spent = self.reserve_spent()
    limits = {
      "gpu": budgets["reserve_gpu"],
      "interpreter": budgets["reserve_interpreter"],
      "p0": budgets["reserve_p0"],
    }
    if any(spent >= limit for limit in limits.values()):
      self.path_ledger.append(
        "launch_refused", slot=slot_id, case=case_id, reason="reserve_spent"
      )
      return None
    attempt = self._next_attempt(slot_id)
    directory = (
      self.work / "discovery" / case_id / input / configuration / f"attempt-{attempt}"
    )
    if directory.exists():
      raise StudyRefused(f"retry directory exists: {directory}")
    directory.mkdir(parents=True)
    self.path_ledger.append(
      "retry",
      slot=slot_id,
      case=case_id,
      attempt=attempt,
      directory=str(directory),
      reserve=spent + 1,
      reason=reason,
    )
    return directory

  def _final_record(self, counts):
    events = self.path_ledger.events
    sources = self.corpus_for_report()
    classifications = [e for e in events if e.get("kind") == "classification"]
    screening_candidates = [
      {"phase": "screening", **event}
      for event in events
      if event.get("kind") == "selection_candidates"
    ]
    invalid_executions = [
      {"phase": "screening", **event}
      for event in events
      if event.get("kind") == "selection_invalid_execution"
    ]
    discovery_candidates = [
      {"phase": "discovery", **event}
      for event in classifications
      if event.get("candidate")
    ]
    # Historical records predate screening-candidate provenance; keep their
    # shape so a preserved final record still reproduces exactly.
    if screening_candidates:
      candidates = screening_candidates + discovery_candidates
    else:
      candidates = [e for e in classifications if e.get("candidate")]
    unavailable = [
      {"slot": e["slot"], "reason": reason}
      for e in classifications
      for reason in e.get("unavailable", [])
    ]
    compiler_observed = sum(
      1 for e in classifications if (e.get("compiler") or {}).get("status") == "observed"
    )
    unexecuted = [
      slot.slot_id
      for slot in self.frozen.tranche.slots
      if self.ledger.state(slot.slot_id) == "unexecuted"
    ]
    return {
      "schema": FINAL_SCHEMA,
      "study": self.frozen.digest,
      "lanes": list(self.frozen.spec["lanes"]),
      "sources": {
        "declared": len(self.frozen.spec["sources"]),
        "selected": len(sources),
        "attempts": sum(1 for e in events if e.get("kind") == "selection_attempt"),
        "shortfalls": self._shortfalls(),
      },
      "controls": {
        "declared": len(self.frozen.spec["controls"]),
        "archived": self.frozen_corpus()["controls"],
      },
      "corpus": sources,
      "corpus_digest": self.frozen_corpus()["digest"],
      "diversity": self.frozen_corpus().get("diversity"),
      "observations": [
        {
          "slot": e["slot"],
          "status": e.get("status"),
          "unavailable": list(e.get("unavailable", [])),
          "compiler": e.get("compiler", {"status": "unavailable"}),
        }
        for e in classifications
      ],
      "execution": {
        "slots": len(self.frozen.tranche.slots),
        "states": counts,
        "unexecuted": unexecuted,
        **(
          {
            "processes": {
              "started": sum(
                1 for e in events if e.get("kind") == "process" and e.get("event") == "start"
              ),
              "finished": sum(
                1 for e in events if e.get("kind") == "process" and e.get("event") == "finish"
              ),
              "by_path": {
                path: {
                  "started": sum(
                    1 for e in events
                    if e.get("kind") == "process"
                    and e.get("event") == "start"
                    and e.get("path") == path
                  ),
                  "finished": sum(
                    1 for e in events
                    if e.get("kind") == "process"
                    and e.get("event") == "finish"
                    and e.get("path") == path
                  ),
                }
                for path in ("p0", "p1", "p2")
              },
            }
          }
          if any(e.get("kind") == "process" for e in events)
          else {}
        ),
      },
      "paths": path_counts(events),
      "candidates": candidates,
      **({"invalid_executions": invalid_executions} if invalid_executions else {}),
      "unavailable": unavailable,
      "compiler": {
        "observed": compiler_observed,
        "unavailable": len(classifications) - compiler_observed,
      },
      "reserve": {
        "spent": self.reserve_spent(),
        "remaining": max(
          0, self.frozen.spec["budgets"]["reserve_gpu"] - self.reserve_spent()
        ),
      },
      "refusals": [
        {"slot": e.get("slot") or e.get("case"), "reason": e.get("reason")}
        for e in events
        if e.get("kind") in ("launch_refused", "reconciled", "shortfall")
      ],
    }

  def finalize(self):
    if self.ledger is None:
      raise StudyRefused("prepare the study before finalizing")
    if self.is_paused():
      raise StudyRefused(
        f"cannot finalize a paused study: {self._paused_reason()}"
      )
    counts = self.ledger.close()
    record = self._final_record(counts)
    (self.work / "final.json").write_text(
      json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def report_from_records(work, *, triage_ledger=None):
  """Recompute the final summary from the preserved records alone."""
  work = Path(work)
  frozen = FrozenStudy.load(work / "study.json")
  study = Study(
    work, frozen, build=None, screen=None, execute=None,
    triage_ledger=triage_ledger,
  )
  study.path_ledger.load()
  study.ledger = tensor_tranche.Ledger.open(
    work / "tranche-ledger.jsonl", frozen.tranche
  )
  return study._final_record(study.ledger.counts())


def compiler_stage_observation(compiled):
  """Run the existing observer over the retained stages and summarize it."""
  from test.lib import tensor_compiler_observation

  compiled = Path(compiled)
  artifacts = {}
  hashes = {}
  for stage, name in (
    ("ttir", "compiler-ttir.ir"),
    ("ttgir", "compiler-ttgir.ir"),
    ("ptx", "compiler-ptx.ir"),
  ):
    path = compiled / name
    if path.is_file():
      artifacts[stage] = path.read_text(encoding="utf-8")
      hashes[stage] = hashlib.sha256(path.read_bytes()).hexdigest()
  if not artifacts:
    return {"status": "unavailable", "reason": "no compiler IR artifacts", "artifact_sha256": hashes}
  try:
    observation = tensor_compiler_observation.observe_compiler_ir(
      artifacts,
      expected_loop=True,
      loop_required_stages=("ttir",),
    )
  except ValueError as error:
    return {
      "status": "unavailable",
      "reason": str(error),
      "artifact_sha256": hashes,
    }
  stages = {
    stage: {
      "loop_construct_count": data.get("loop_construct_count"),
      "layout_conversion_count": len(data.get("layout_conversions") or []),
      "duplicate_work_count": len(data.get("duplicate_work") or []),
    }
    for stage, data in (observation.get("stages") or {}).items()
  }
  return {
    "status": "observed",
    "observer": "tensor_compiler_observation.observe_compiler_ir",
    "artifact_sha256": hashes,
    "stages": stages,
    "loop_evidence": observation.get("loop_evidence"),
    "predicate_attribution": "unavailable",
    "predicate_reason": (
      "the retained IR does not identify the window load's predicate apart "
      "from other predicated loads"
    ),
    "later_loop_retention": "unavailable",
  }



def require_toolchain(frozen, current, *, accept_current=False):
  """Refuse execution unless the current toolchain matches the frozen one."""
  frozen_toolchain = frozen.spec.get("toolchain")
  if frozen_toolchain is None:
    if not accept_current:
      raise StudyRefused(
        "the frozen study carries no toolchain identity; an explicit "
        "acceptance is required before execution"
      )
    return
  if frozen_toolchain != current:
    raise StudyRefused("the current toolchain differs from the frozen study identity")


def _trip_variant(source, trips):
  """The same body at a smaller trip count, with the window witness checked."""
  marker = "br %i < "
  start = source.find(marker)
  if start < 0:
    raise StudyRefused("the assembled source has no counted trip branch")
  end = source.find(", ^body, ^exit;", start)
  if end < 0:
    raise StudyRefused("the assembled source has no counted trip branch")
  return source[:start] + marker + str(trips) + source[end:]


def _file_sha256(path):
  try:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
  except OSError:
    return None


def verify_control_payload(corpus, probe, bundle, warps):
  """Load a frozen control only after verifying every mutable payload part."""
  bundle = Path(bundle)
  archived_program = (bundle / "program.sir").read_bytes()
  archived_inputs = (bundle / "inputs.bin").read_bytes()
  archived_case = (bundle / "case.json").read_bytes()
  frozen_hashes = (corpus.get("hashes") or {}).get(probe)
  if frozen_hashes:
    if hashlib.sha256(archived_program).hexdigest() != frozen_hashes.get("program_sha256"):
      raise StudyRefused(f"archived control program changed: {corpus.get('source')}")
    if hashlib.sha256(archived_inputs).hexdigest() != frozen_hashes.get("inputs_sha256"):
      raise StudyRefused(f"archived control input changed: {corpus.get('source')}")
    if hashlib.sha256(archived_case).hexdigest() != (frozen_hashes.get("case_sha256") or {}).get(str(warps)):
      raise StudyRefused(f"archived control case changed: {corpus.get('source')}")
  case_record = json.loads(archived_case)
  if case_record.get("launch", {}).get("num_warps") != warps:
    raise StudyRefused(
      f"archived control warp setting differs from the slot: {warps}"
    )
  return archived_program, archived_inputs, archived_case


def device_operations(*, exporter, symirc, symiri, controls_root=None, timeout_s=300.0):
  """Bind the driver to the exporter, the interpreter screen, and the runner."""
  from pathlib import Path as _Path

  from test.lib import (
    native_triton_family,
    probe_triton_env,
    tensor_free_form_case,
    tensor_free_form_inputs,
    tensor_compiler_observation,
    tensor_state_diagnostics,
    tensor_target_case,
    tensor_target_reference,
    tensor_target_run,
  )

  exporter = Path(exporter).resolve()
  symirc = Path(symirc).resolve()
  symiri = Path(symiri).resolve()
  environment = probe_triton_env.collect_environment()
  identity = native_triton_family.triton_identity(environment)
  try:
    import triton as _triton
    triton_library = Path(_triton.__file__).parent / "_C" / "libtriton.so"
  except Exception:
    triton_library = None
  toolchain = {
    "exporter_sha256": _file_sha256(exporter),
    "symirc_sha256": _file_sha256(symirc),
    "symiri_sha256": _file_sha256(symiri),
    "triton": identity,
    "triton_library_sha256": None if triton_library is None else _file_sha256(triton_library),
    "python": sys.version.split()[0],
    "device": identity.get("device"),
  }

  def build(request):
    slot = request["slot"]
    seed = request["seed"]
    directory = _Path(request["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    rows, columns = slot["shape"]
    command = [
      str(exporter),
      "--policy", slot["policy"],
      "--seed", str(seed),
      "--shape", f"{rows},{columns}",
      "--states", "2",
      "--trip-count", str(slot["trips"]),
      "--window", f"{rows},{slot['stride']},{slot['active']}",
      "--window-buffer-rows", str(slot["backing_rows"]),
      "--profile", "v3",
      "--features", "extrema,transpose,expand",
      "--window-requirement", slot["requirement"],
      "--max-attempts", "1",
      "--max-multiplications", "1",
      "--require-computed-roots",
      "--num-warps", "1",
      "--out", str(directory / "export"),
    ]
    completed = subprocess.run(
      command, capture_output=True, text=True, check=False, timeout=timeout_s
    )
    if completed.returncode != 0:
      return {
        "kind": "refused",
        "reason": (completed.stderr or completed.stdout).strip().splitlines()[-1:]
        and (completed.stderr or completed.stdout).strip()
      }
    export = directory / "export"
    region = json.loads((export / "region.json").read_text(encoding="utf-8"))
    source = (export / "program.sir").read_text(encoding="utf-8")
    if (slot["trips"] - 1) * slot["stride"] + rows > slot["backing_rows"]:
      return {"kind": "refused", "reason": "window witness leaves the backing"}
    probes = tensor_free_form_inputs.window_probes(
      region, group=slot["group"], column=slot["column"]
    )
    layouts = None
    if slot["padded"]:
      layouts = {
        "%window_buffer": tensor_free_form_case.strided_output(
          [slot["backing_rows"], columns], row_padding=3, element_offset=1
        )
      }
    bundles = {}
    for probe in probes:
      bundle = directory / "bundles" / probe.probe
      tensor_free_form_case.build_bundle(
        bundle,
        region=region,
        source=source,
        proposal=probe,
        input_layouts=layouts,
        output_row_padding=1,
        output_offset=1,
      )
      bundles[probe.probe] = str(bundle)
    tminus = _trip_variant(source, slot["trips"] - 1)
    for probe in probes:
      if probe.probe in ("i0-base", "i2-late-row"):
        bundle = directory / "bundles" / f"{probe.probe}-tminus"
        tensor_free_form_case.build_bundle(
          bundle,
          region=region,
          source=tminus,
          proposal=probe,
          input_layouts=layouts,
          output_row_padding=1,
          output_offset=1,
        )
        bundles[f"{probe.probe}-tminus"] = str(bundle)
    return {
      "kind": "admitted",
      "directory": str(directory),
      "export_command": list(command),
      "region": region,
      "source": source,
      "trip_variant": tminus,
      "probes": {probe.probe: probe.as_record() for probe in probes},
      "probe_values": {probe.probe: list(probe.data) for probe in probes},
      "layouts": layouts or {},
      "bundles": bundles,
      "case": slot["id"],
      "trip_count": slot["trips"],
      "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
      "inputs_sha256": hashlib.sha256(probes[0].data).hexdigest(),
      "witnesses": region.get("requirements", []),
      "compiler_identity": identity,
    }

  def _screen_bundle(bundle_path, on_process=None):
    return tensor_target_run.run_interpreter(
      _Path(bundle_path),
      symirc=symirc,
      symiri=symiri,
      timeout_s=timeout_s,
      compiler_identity=identity,
      on_process=on_process,
    )

  def _state_variant(outcome, probe, trips, variant, p1_process=None):
    """Export the same body returning one state, then run the requested probe."""
    state = variant.rsplit("-", 1)[-1]
    if state not in ("0", "1"):
      return "unsupported", None
    base = _Path(outcome["directory"]) / "variants" / f"state-{state}"
    base.mkdir(parents=True, exist_ok=True)
    command = list(outcome.get("export_command") or [])
    if len(command) < 2 or command[-2] != "--out":
      return "unsupported", None
    command = command[:-2] + ["--return", f"state:{state}", "--out", str(base / "export")]
    completed = subprocess.run(
      command, capture_output=True, text=True, check=False, timeout=timeout_s
    )
    if completed.returncode != 0:
      (base / "reason.txt").write_text(
        (completed.stderr or completed.stdout).strip()[-2000:], encoding="utf-8"
      )
      return "runtime_failure", None
    source = (base / "export" / "program.sir").read_text(encoding="utf-8")

    def body(text):
      # Returning one state omits the unused combined-output declaration; every
      # loop declaration and statement must otherwise be identical.
      return [
        line for line in text.split("^exit:")[0].splitlines()
        if not line.strip().startswith("let mut %out:")
      ]

    if body(source) != body(outcome["source"]):
      (base / "reason.txt").write_text("the variant body changed", encoding="utf-8")
      return "runtime_failure", None
    if trips != outcome["trip_count"]:
      source = _trip_variant(source, trips)
    data = bytes(outcome["probe_values"][probe])
    proposal = tensor_free_form_inputs.Proposal(
      schema=tensor_free_form_inputs.WINDOW_PROBE_SCHEMA,
      pattern=probe,
      seed=0,
      names=tuple(record["name"] for record in
                  tensor_free_form_inputs.parameters(outcome["region"])),
      shapes=tuple(tuple(record["shape"]) for record in
                   tensor_free_form_inputs.parameters(outcome["region"])),
      dtype="f32",
      values=(),
      data=data,
    )
    bundle = base / f"bundle-{probe}-{trips}"
    tensor_free_form_case.build_bundle(
      bundle,
      region={**outcome["region"], "num_warps": 1},
      source=source,
      proposal=proposal,
      input_layouts=outcome.get("layouts") or None,
      output_row_padding=1,
      output_offset=1,
    )
    run = _screen_bundle(bundle, p1_process)
    return run.status, run.interpreter.actual

  def screen(outcome):
    """Use the shared window schedule; keep candidates and P0 separate."""
    process = outcome.get("on_process")
    trip_count = outcome["trip_count"]

    def p1_process(event):
      if process is not None:
        process({**event, "path": "p1"})

    def p0_process(event):
      if process is not None:
        process({**event, "path": "p0"})

    def run_probe(probe, trips, variant):
      if variant != "combined":
        return _state_variant(outcome, probe, trips, variant, p1_process)
      key = probe if trips == trip_count else f"{probe}-tminus"
      bundle = outcome["bundles"].get(key)
      if bundle is None:
        return "unsupported", None
      screened = _screen_bundle(bundle, p1_process)
      return screened.status, screened.interpreter.actual

    result = tensor_state_diagnostics.window_screen(
      trip_count=trip_count, run_probe=run_probe
    )
    selected = bool(result["selection"]["selected"])
    candidates = list(result["candidates"])
    unavailable = list(result["unavailable"])
    p0 = []
    if selected:
      for probe in ("i0-base", "i1-read-plus-one", "i2-late-row", "i3-unused-rows"):
        bundle = outcome["bundles"][probe]
        case = tensor_target_case.load(bundle, symiri=symiri)
        output_path = _Path(outcome["directory"]) / f"p0-{probe}.bin"
        observation = tensor_target_reference.observe(
          case, symiri=symiri, timeout_s=timeout_s, output_path=output_path,
          on_process=p0_process,
        )
        record = observation.as_record()
        base_run = next(
          run for run in result["runs"]
          if run["probe"] == probe and run["trip_count"] == trip_count
          and run["variant"] == "combined"
        )
        p1_sha = base_run.get("actual_sha256")
        p0_sha = record.get("output_sha256")
        agreement = (
          None if p0_sha is None or p1_sha is None else p0_sha == p1_sha
        )
        p0.append(
          {
            "probe": probe,
            "status": observation.status,
            "output_sha256": p0_sha,
            "output_path": str(output_path),
            "p1_sha256": p1_sha,
            "agreement": agreement,
          }
        )
        if observation.status == tensor_target_reference.UB:
          selected = False
          candidates.append(
            {
              "kind": "invalid_execution",
              "probe": probe,
              "status": observation.status,
            }
          )
        elif observation.status == tensor_target_reference.COMPLETED:
          if agreement is False:
            candidates.append(
              {
                "kind": "p0_p1_disagreement",
                "probe": probe,
                "p0_sha256": p0_sha,
                "p1_sha256": p1_sha,
              }
            )
        else:
          unavailable.append(
            {"probe": probe, "status": observation.status, "reason": observation.reason}
          )
    launches = 6 + sum(
      1 for run in result["runs"]
      if run["variant"] != "combined" and run["status"] not in ("unsupported",)
    )
    return {
      "selected": selected,
      "reason": result["selection"]["reason"],
      "launches": launches,
      "p0": len(p0),
      "candidates": candidates,
      "unavailable": unavailable,
      "record": {**result, "p0": p0},
    }

  reference_cache = {}

  def execute(request):
    slot = request["slot"]
    corpus = request["corpus"]
    probe = request["input"]
    configuration = request["configuration"]
    directory = _Path(request["directory"])
    process = request.get("on_process")

    def p0_process(event):
      if process is not None:
        process({**event, "path": "p0"})

    def run_process(event):
      if process is not None:
        path = "p1" if event.get("mode") == "interpret" else "p2"
        process({**event, "path": path})
    directory.mkdir(parents=True, exist_ok=True)
    if corpus.get("kind") == "control":
      if controls_root is None:
        return {
          "status": "unavailable",
          "paths": {},
          "candidate": False,
          "unavailable": ["no controls root was supplied"],
          "compiler": {"status": "unavailable"},
        }
      warps = int(configuration.rsplit("-", 1)[1])
      pattern = CONTROL_PATTERNS[probe]
      bundle = (
        Path(controls_root) / str(corpus["source"]) / f"{pattern}-warps-{warps}"
      )
      if not bundle.is_dir():
        return {
          "status": "unavailable",
          "paths": {},
          "candidate": False,
          "unavailable": [f"archived control bundle is missing: {bundle}"],
          "compiler": {"status": "unavailable"},
        }
      if corpus.get("status") == "missing":
        return {
          "status": "unavailable",
          "paths": {},
          "candidate": False,
          "unavailable": [corpus.get("reason", "archived control is missing")],
          "compiler": {"status": "unavailable"},
        }
      archived_program, archived_inputs, archived_case = verify_control_payload(
        corpus, probe, bundle, warps
      )
      case = tensor_target_case.load(bundle, symiri=symiri)
      cache_key = (
        "control",
        corpus["source"],
        probe,
        hashlib.sha256(archived_inputs).hexdigest(),
        case.record.get("profile"),
        case.record.get("entry"),
        "interpret",
        toolchain["symiri_sha256"],
      )
      reference = reference_cache.get(cache_key)
      cached = reference is not None
      if reference is None:
        reference = tensor_target_reference.observe(
          case, symiri=symiri, timeout_s=timeout_s,
          output_path=directory / "p0.bin",
          on_process=p0_process,
        )
        if reference.status == tensor_target_reference.COMPLETED:
          reference_cache[cache_key] = reference
      run = tensor_target_run.run(
        bundle,
        symirc=symirc,
        symiri=symiri,
        reference=reference,
        timeout_s=timeout_s,
        artifact_dir=directory / "target-runs",
        compiler_identity=identity,
        on_process=run_process,
      )
      paths = {
        "p0": {
          "status": reference.status,
          "cached": cached,
          "directory": str(directory / "p0.bin"),
          "cache_key": "|".join(str(part) for part in cache_key),
        },
        "p1": {
          "status": run.interpreter.status,
          "directory": str(directory / "target-runs" / "interpret"),
        },
        "p2": {
          "status": run.compiled.status,
          "directory": str(directory / "target-runs" / "compiled"),
        },
      }
      compiled = directory / "target-runs" / "compiled"
      compiler = compiler_stage_observation(compiled)
      return {
        "status": run.status,
        "paths": paths,
        "candidate": run.candidate,
        "unavailable": [] if reference.status == tensor_target_reference.COMPLETED else [
          f"p0:{reference.status}"
        ],
        "compiler": compiler,
        "control": {
          "program_sha256": hashlib.sha256(archived_program).hexdigest(),
          "inputs_sha256": hashlib.sha256(archived_inputs).hexdigest(),
          "pattern": pattern,
          "warps": warps,
        },
        "reference": reference,
        "record": run.as_record(),
      }
    if corpus.get("compiler_identity") not in (None, identity):
      raise StudyRefused("the corpus compiler identity differs from the current environment")
    region = corpus["region"]
    source = corpus["source"]
    if corpus.get("source_sha256") and hashlib.sha256(source.encode("utf-8")).hexdigest() != corpus["source_sha256"]:
      raise StudyRefused("the corpus source changed after the freeze")
    probe_bytes = bytes(corpus["probe_values"][probe])
    stored_probe = (corpus.get("probes") or {}).get(probe) or {}
    if stored_probe.get("sha256") and hashlib.sha256(probe_bytes).hexdigest() != stored_probe["sha256"]:
      raise StudyRefused("the corpus probe inputs changed after the freeze")
    proposal = tensor_free_form_inputs.Proposal(
      schema=tensor_free_form_inputs.WINDOW_PROBE_SCHEMA,
      pattern=probe,
      seed=0,
      names=tuple(record["name"] for record in
                  tensor_free_form_inputs.parameters(region)),
      shapes=tuple(tuple(record["shape"]) for record in
                   tensor_free_form_inputs.parameters(region)),
      dtype="f32",
      values=(),
      data=probe_bytes,
    )
    warps = int(configuration.rsplit("-", 1)[1])
    bundled_region = {**region, "num_warps": warps}
    bundle = directory / "bundle"
    tensor_free_form_case.build_bundle(
      bundle,
      region=bundled_region,
      source=source,
      proposal=proposal,
      input_layouts=corpus.get("layouts") or None,
      output_row_padding=1,
      output_offset=1,
    )
    case = tensor_target_case.load(bundle, symiri=symiri)
    reference = request.get("reference")
    cached = reference is not None
    cache_state = "hit" if cached else "miss"
    cache_key = (
      "source",
      corpus.get("source_sha256"),
      hashlib.sha256(probe_bytes).hexdigest(),
      case.record.get("profile"),
      case.record.get("entry"),
      "interpret",
      toolchain["symiri_sha256"],
    )
    if reference is None:
      reference = reference_cache.get(cache_key)
      cached = reference is not None
      cache_state = "hit" if cached else "miss"
    if reference is None:
      selection_p0 = [
        entry
        for entry in ((corpus.get("screen") or {}).get("p0") or [])
        if entry.get("probe") == probe
      ]
      if selection_p0:
        entry = selection_p0[0]
        if entry.get("status") == tensor_target_reference.COMPLETED and entry.get("output_sha256"):
          output_path = entry.get("output_path")
          data = None
          if output_path and _Path(output_path).is_file():
            data = _Path(output_path).read_bytes()
          if data is not None and hashlib.sha256(data).hexdigest() == entry["output_sha256"]:
            reference = tensor_target_reference.ReferenceObservation(
              tensor_target_reference.COMPLETED,
              output=data,
              path="selection-cache",
            )
            cached = True
            cache_state = "selection-hit"
          else:
            cache_state = "selection-corrupt"
        else:
          cache_state = "selection-incomplete"
      if reference is None and cache_state == "selection-corrupt":
        reference = tensor_target_reference.ReferenceObservation(
          tensor_target_reference.UNAVAILABLE,
          reason=f"cached P0 output for {probe} is missing or fails identity validation",
        )
      if reference is None:
        reference = tensor_target_reference.observe(
          case, symiri=symiri, timeout_s=timeout_s, on_process=p0_process
        )
    run = tensor_target_run.run(
      bundle,
      symirc=symirc,
      symiri=symiri,
      reference=reference,
      timeout_s=timeout_s,
      compiler_identity=identity,
      on_process=run_process,
    )
    if reference.status == tensor_target_reference.COMPLETED and not cached:
      reference_cache[cache_key] = reference
      cache_state = "cached-for-later-warps"
    paths = {
      "p0": {
        "status": reference.status,
        "cached": cached,
        "cache_state": cache_state,
        "directory": str(bundle / "runs" / "reference"),
      },
      "p1": {
        "status": run.interpreter.status,
        "directory": str(bundle / "target-runs" / "interpret"),
      },
      "p2": {
        "status": run.compiled.status,
        "directory": str(bundle / "target-runs" / "compiled"),
      },
    }
    compiled = bundle / "target-runs" / "compiled"
    compiler = compiler_stage_observation(compiled)
    unavailable = []
    if reference.status != tensor_target_reference.COMPLETED:
      unavailable.append(f"p0:{reference.status}")
    return {
      "status": run.status,
      "paths": paths,
      "candidate": run.candidate,
      "unavailable": unavailable,
      "compiler": compiler,
      "reference": reference,
      "record": run.as_record(),
    }

  return {"build": build, "screen": screen, "execute": execute, "toolchain": toolchain}


def run_qualification(work, *, operations, slots=4, timeout_s=300.0):
  """Four integration cases under two warp settings, reusing each P0."""
  frozen = freeze_study()
  chosen = frozen.spec["sources"][:slots]
  work = Path(work)
  work.mkdir(parents=True, exist_ok=True)
  record = {"schema": "refractir.moving-window-qualification.v1", "cases": []}
  for slot in chosen:
    case_record = {"case": slot["id"], "padded": slot["padded"], "configurations": {}}
    built = None
    for attempt in range(4):
      candidate = operations["build"](
        {
          "slot": slot,
          "attempt": attempt,
          "seed": frozen.spec["attempts"]["seed_base"] + 8 * slot["index"] + attempt,
          "max_attempts": 1,
          "directory": str(work / slot["id"] / f"attempt-{attempt}"),
        }
      )
      if candidate.get("kind") == "admitted":
        built = candidate
        case_record["attempt"] = attempt
        break
      case_record.setdefault("refusals", []).append(candidate.get("reason"))
    if built is None:
      record["cases"].append(case_record)
      continue
    p0_reference = None
    for configuration in frozen.spec["configurations"]:
      events = []

      def account(event, events=events):
        events.append(event)

      outcome = operations["execute"](
        {
          "slot": f"{slot['id']}/i0-base/{configuration}",
          "case": slot["id"],
          "corpus": {"kind": "source", **built, "case": slot["id"]},
          "input": "i0-base",
          "configuration": configuration,
          "attempt": case_record.get("attempt", 0),
          "directory": str(work / slot["id"] / configuration),
          "reference": p0_reference,
          "on_process": account,
        }
      )
      if p0_reference is None:
        p0_reference = outcome.get("reference")
      case_record["configurations"][configuration] = {
        "status": outcome.get("status"),
        "paths": outcome.get("paths"),
        "candidate": outcome.get("candidate"),
        "compiler": outcome.get("compiler"),
        "processes": events,
      }
    record["cases"].append(case_record)
  record["digest"] = _digest(record)
  (work / "qualification.json").write_text(
    json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8"
  )
  return record


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--work", required=True)
  parser.add_argument("--exporter", required=True)
  parser.add_argument("--symirc", required=True)
  parser.add_argument("--symiri", required=True)
  parser.add_argument("--timeout", type=float, default=300.0)
  parser.add_argument("--controls-root", default=None)
  parser.add_argument("--mode", choices=("qualification", "select", "discover", "close", "reconcile"),
                      required=True)
  parser.add_argument("--triage-ledger", default=None)
  parser.add_argument("--accept-current-toolchain", action="store_true")
  parser.add_argument("--triage-decision", default=None)
  arguments = parser.parse_args(argv)
  if arguments.mode == "reconcile":
    report = report_from_records(
      Path(arguments.work),
      triage_ledger=Path(arguments.triage_ledger) if arguments.triage_ledger else None,
    )
    (Path(arguments.work) / "final-reconciled.json").write_text(
      json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["reserve"]))
    return 0
  operations = device_operations(
    exporter=Path(arguments.exporter),
    symirc=Path(arguments.symirc),
    symiri=Path(arguments.symiri),
    controls_root=Path(arguments.controls_root) if arguments.controls_root else None,
    timeout_s=arguments.timeout,
  )
  if arguments.mode == "qualification":
    record = run_qualification(arguments.work, operations=operations)
    print(json.dumps({"cases": len(record["cases"]), "digest": record["digest"]}))
    return 0
  toolchain = operations.pop("toolchain")
  work = Path(arguments.work)
  study_path = work / "study.json"
  frozen = FrozenStudy.load(study_path) if study_path.is_file() else freeze_study(toolchain=toolchain)
  require_toolchain(
    frozen, toolchain, accept_current=arguments.accept_current_toolchain
  )
  study = Study(
    work, frozen, controls_root=Path(arguments.controls_root) if arguments.controls_root else None,
    triage_ledger=Path(arguments.triage_ledger) if arguments.triage_ledger else None,
    **operations,
  ).prepare()
  if study.is_paused():
    if arguments.triage_decision is None:
      raise StudyRefused(
        f"the study is paused ({study._paused_reason()}); pass --triage-decision"
      )
    study.record_triage_decision(arguments.triage_decision)
    if arguments.triage_decision == "stop":
      print(json.dumps({"paused": "stopped"}))
      return 2
  if arguments.mode == "select":
    study.select_sources()
    if study.is_paused():
      print(json.dumps({"paused": study._paused_reason()}))
      return 2
    corpus = study.freeze_corpus()
    print(json.dumps({"selected": len(corpus["sources"]),
                      "shortfalls": corpus["shortfalls"]}))
    return 0
  if arguments.mode == "discover":
    study.run_discovery()
    if study.is_paused():
      print(json.dumps({"paused": study._paused_reason()}))
      return 2
    print(json.dumps({"execution": study.ledger.counts()}))
    return 0
  final = study.finalize()
  print(json.dumps({"execution": final["execution"]["states"],
                    "candidates": len(final["candidates"])}))
  return 0


if __name__ == "__main__":
  sys.exit(main())
