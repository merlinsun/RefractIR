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
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from test.lib import tensor_tranche

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


def freeze_study(spec=None):
  spec = study_spec() if spec is None else spec
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

  def __init__(self, work, frozen, *, build, screen, execute):
    self.work = Path(work)
    self.frozen = frozen
    self.build = build
    self.screen = screen
    self.execute = execute
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

  def _shortfalls(self):
    return [e["case"] for e in self.path_ledger.events if e.get("kind") == "shortfall"]

  def _selection_closed(self):
    return len(self._selected_ids()) + len(self._shortfalls()) >= len(
      self.frozen.spec["sources"]
    )

  def select_sources(self, max_slots=None):
    """Run the bounded builder calls and screen every admitted body.

    A selected or shortfall slot is settled: resume never rebuilds it and never
    re-rolls a shortfall. `max_slots` is a host-fixture hook for interrupting a
    selection run; the campaign passes no limit.
    """
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
      for attempt in range(self.frozen.spec["attempts"]["per_slot"]):
        seed = (
          self.frozen.spec["attempts"]["seed_base"] + 8 * slot["index"] + attempt
        )
        directory = self.work / "selection" / slot["id"] / f"attempt-{attempt}"
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
        screened = self.screen(outcome)
        self.path_ledger.append(
          "selection_screen",
          case=slot["id"],
          selected=bool(screened.get("selected")),
          reason=screened.get("reason"),
          launches=screened.get("launches", 0),
          p0=screened.get("p0", 0),
        )
        if screened.get("selected"):
          admitted = {
            "case": slot["id"],
            "lane": slot["lane"],
            "kind": "source",
            "source_sha256": outcome.get("source_sha256"),
            "inputs_sha256": outcome.get("inputs_sha256"),
            "layouts": outcome.get("layouts", {}),
            "witnesses": outcome.get("witnesses", []),
            "compiler_identity": outcome.get("compiler_identity"),
            "screen": screened.get("record", {}),
          }
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

  def freeze_corpus(self):
    """Freeze the admitted sources and every archived control before dispatch."""
    if not self._selection_closed():
      raise StudyRefused("the corpus cannot freeze before every source slot is settled")
    entries = self.corpus()
    if not entries:
      raise StudyRefused("the corpus has no admitted source")
    controls = []
    for control in self.frozen.spec["controls"]:
      controls.append(
        {
          "case": control["id"],
          "lane": "control",
          "kind": "control",
          "source": control["source"],
        }
      )
    record = {
      "schema": CORPUS_SCHEMA,
      "sources": entries,
      "controls": controls,
      "shortfalls": self._shortfalls(),
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

  def _next_attempt(self, slot_id):
    used = [
      event["attempt"]
      for event in self.path_ledger.events
      if event.get("slot") == slot_id and event.get("kind") == "launch"
    ]
    return max(used, default=-1) + 1

  def run_discovery(self):
    """Execute every frozen paired slot that has a corpus entry."""
    if self.ledger is None:
      raise StudyRefused("prepare the study before dispatch")
    corpus = self.frozen_corpus()
    available = {entry["case"] for entry in corpus["sources"] + corpus["controls"]}
    cases = {
      case["id"]: case
      for case in self.frozen.tranche.payload["manifest"]["cases"]
    }
    for slot in self.frozen.tranche.slots:
      if self.ledger.state(slot.slot_id) != "pending":
        continue
      if slot.case_id not in available:
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
      }
      try:
        outcome = self.execute(request)
      except Exception as error:
        self.ledger.finish(slot.slot_id, "failed", reason=f"{type(error).__name__}: {error}")
        self.path_ledger.append(
          "classification",
          slot=slot.slot_id,
          status="driver_error",
          candidate=False,
          unavailable=[],
          compiler={"status": "unavailable"},
        )
        continue
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
      self.path_ledger.append(
        "classification",
        slot=slot.slot_id,
        status=outcome.get("status", "completed"),
        candidate=bool(outcome.get("candidate")),
        unavailable=list(outcome.get("unavailable", [])),
        compiler=outcome.get("compiler", {"status": "unavailable"}),
      )
      self.ledger.finish(slot.slot_id, "completed", detail={"status": outcome.get("status")})
    return self

  def reserve_spent(self):
    return sum(1 for e in self.path_ledger.events if e.get("kind") == "retry")

  def retry(self, slot_id, *, case_id, input, configuration, reason):
    """Charge the reserve for a fresh attempt, or refuse and keep evidence."""
    limit = self.frozen.spec["budgets"]["reserve_gpu"]
    if self.reserve_spent() >= limit:
      self.path_ledger.append(
        "launch_refused", slot=slot_id, case=case_id, reason="reserve_spent"
      )
      return None
    attempt = self._next_attempt(slot_id) + 1
    directory = (
      self.work / "discovery" / case_id / input / configuration / f"attempt-{attempt}"
    )
    self.path_ledger.append(
      "retry",
      slot=slot_id,
      case=case_id,
      attempt=attempt,
      reserve=self.reserve_spent() + 1,
      reason=reason,
    )
    return directory

  def _final_record(self, counts):
    events = self.path_ledger.events
    sources = self.corpus()
    classifications = [e for e in events if e.get("kind") == "classification"]
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
      },
      "paths": path_counts(events),
      "candidates": candidates,
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
        {"slot": e["slot"], "reason": e.get("reason")}
        for e in events
        if e.get("kind") in ("launch_refused", "reconciled", "shortfall")
      ],
    }

  def finalize(self):
    if self.ledger is None:
      raise StudyRefused("prepare the study before finalizing")
    counts = self.ledger.close()
    record = self._final_record(counts)
    (self.work / "final.json").write_text(
      json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def report_from_records(work):
  """Recompute the final summary from the preserved records alone."""
  work = Path(work)
  frozen = FrozenStudy.load(work / "study.json")
  study = Study(work, frozen, build=None, screen=None, execute=None)
  study.path_ledger.load()
  study.ledger = tensor_tranche.Ledger.open(
    work / "tranche-ledger.jsonl", frozen.tranche
  )
  return study._final_record(study.ledger.counts())


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


def device_operations(*, exporter, symirc, symiri, timeout_s=300.0):
  """Bind the driver to the exporter, the interpreter screen, and the runner."""
  from pathlib import Path as _Path

  from test.lib import (
    native_triton_family,
    probe_triton_env,
    tensor_free_form_case,
    tensor_free_form_inputs,
    tensor_target_case,
    tensor_target_reference,
    tensor_target_run,
  )

  exporter = Path(exporter).resolve()
  symirc = Path(symirc).resolve()
  symiri = Path(symiri).resolve()
  environment = probe_triton_env.collect_environment()
  identity = native_triton_family.triton_identity(environment)

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
      "region": region,
      "source": source,
      "trip_variant": tminus,
      "probes": {probe.probe: probe.as_record() for probe in probes},
      "probe_values": {probe.probe: list(probe.data) for probe in probes},
      "layouts": layouts or {},
      "bundles": bundles,
      "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
      "inputs_sha256": hashlib.sha256(probes[0].data).hexdigest(),
      "witnesses": region.get("requirements", []),
      "compiler_identity": identity,
    }

  def _screen_bundle(bundle_path):
    return tensor_target_run.run_interpreter(
      _Path(bundle_path),
      symirc=symirc,
      symiri=symiri,
      timeout_s=timeout_s,
      compiler_identity=identity,
    )

  def screen(outcome):
    runs = {}
    for probe in ("i0-base", "i1-read-plus-one", "i2-late-row", "i3-unused-rows"):
      runs[probe] = _screen_bundle(outcome["bundles"][probe])
    tminus = {
      probe: _screen_bundle(outcome["bundles"][f"{probe}-tminus"])
      for probe in ("i0-base", "i2-late-row")
    }

    def value(run):
      return run.interpreter.actual if run.status == "completed" else None

    base = value(runs["i0-base"])
    broad = value(runs["i1-read-plus-one"])
    late = value(runs["i2-late-row"])
    unused = value(runs["i3-unused-rows"])
    trip_base = value(tminus["i0-base"])
    finished = all(part is not None for part in (base, broad, late, unused, trip_base))
    selected = bool(
      finished and base != broad and base != late and base == unused and base != trip_base
    )
    reason = None
    if not finished:
      reason = "incomplete_observation"
    elif base == unused and not (base != broad and base != late):
      reason = "insensitive_to_window_probes"
    elif base != unused:
      reason = "unused_rows_changed"
    elif base == trip_base:
      reason = "trip_invariant"
    record = {
      "runs": {
        name: {
          "status": run.status,
          "output_sha256": run.as_record()["output_sha256"],
        }
        for name, run in {**runs, **{f"{k}-tminus": v for k, v in tminus.items()}}.items()
      },
      "selected": selected,
      "reason": reason,
    }
    p0 = []
    if selected:
      for probe in ("i0-base", "i1-read-plus-one", "i2-late-row", "i3-unused-rows"):
        case = tensor_target_case.load(outcome["bundles"][probe], symiri=symiri)
        observation = tensor_target_reference.observe(
          case, symiri=symiri, timeout_s=timeout_s
        )
        p0.append(
          {
            "probe": probe,
            "status": observation.status,
            "output_sha256": getattr(observation, "output_sha256", None),
          }
        )
        if observation.status == tensor_target_reference.UB:
          selected = False
          reason = "invalid_execution"
          record["selected"] = False
          record["reason"] = reason
    return {
      "selected": selected,
      "reason": reason,
      "launches": len(runs) + len(tminus),
      "p0": len(p0),
      "record": {**record, "p0": p0},
    }

  def execute(request):
    slot = request["slot"]
    corpus = request["corpus"]
    probe = request["input"]
    configuration = request["configuration"]
    directory = _Path(request["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    if corpus.get("kind") == "control":
      return {
        "status": "unavailable",
        "paths": {},
        "candidate": False,
        "unavailable": ["archived control is not bound to a bundle in this driver"],
        "compiler": {"status": "unavailable"},
      }
    region = corpus["region"]
    source = corpus["source"]
    probe_bytes = bytes(corpus["probe_values"][probe])
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
    if reference is None:
      reference = tensor_target_reference.observe(
        case, symiri=symiri, timeout_s=timeout_s
      )
    run = tensor_target_run.run(
      bundle,
      symirc=symirc,
      symiri=symiri,
      reference=reference,
      timeout_s=timeout_s,
      compiler_identity=identity,
    )
    paths = {
      "p0": {
        "status": reference.status,
        "cached": request.get("reference") is not None,
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
    compiler = {"status": "unavailable"}
    ttir = compiled / "compiler-ttir.ir"
    ptx = compiled / "compiler-ptx.ir"
    if ttir.is_file() and ptx.is_file():
      compiler = {
        "status": "observed",
        "ttir_sha256": hashlib.sha256(ttir.read_bytes()).hexdigest(),
        "ptx_sha256": hashlib.sha256(ptx.read_bytes()).hexdigest(),
      }
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

  return {"build": build, "screen": screen, "execute": execute}


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
        }
      )
      if p0_reference is None:
        p0_reference = outcome.get("reference")
      case_record["configurations"][configuration] = {
        "status": outcome.get("status"),
        "paths": outcome.get("paths"),
        "candidate": outcome.get("candidate"),
        "compiler": outcome.get("compiler"),
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
  parser.add_argument("--mode", choices=("qualification", "select", "discover"),
                      required=True)
  arguments = parser.parse_args(argv)
  operations = device_operations(
    exporter=Path(arguments.exporter),
    symirc=Path(arguments.symirc),
    symiri=Path(arguments.symiri),
    timeout_s=arguments.timeout,
  )
  if arguments.mode == "qualification":
    record = run_qualification(arguments.work, operations=operations)
    print(json.dumps({"cases": len(record["cases"]), "digest": record["digest"]}))
    return 0
  frozen = freeze_study()
  study = Study(Path(arguments.work), frozen, **operations).prepare()
  if arguments.mode == "select":
    study.select_sources()
    corpus = study.freeze_corpus()
    print(json.dumps({"selected": len(corpus["sources"]),
                      "shortfalls": corpus["shortfalls"]}))
    return 0
  study.run_discovery()
  final = study.finalize()
  print(json.dumps({"execution": final["execution"]["states"],
                    "candidates": len(final["candidates"])}))
  return 0


if __name__ == "__main__":
  sys.exit(main())
