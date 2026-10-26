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

from test.lib import (  # noqa: E402
  tensor_campaign_evidence,  # noqa: E402
  tensor_execution,
  tensor_tranche,
)
from test.lib import tensor_candidate_triage as candidate_triage  # noqa: E402
from test.lib import tensor_study_accounting as accounting  # noqa: E402

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
  """A refusal that keeps evidence and stops the requested operation."""


DispatchRefused = tensor_execution.DispatchRefused


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


QUALIFICATION = {
  "version": 2,
  "max_multiplications": 1,
  "cases": [
    {"case": "mw-g0c0", "checks": ["backing_storage"]},
    {"case": "mw-g0c1", "checks": ["changing_active_rows"]},
    {"case": "mw-g0c3", "checks": ["strided_backing_layout"]},
    {"case": "mw-g2c3", "checks": ["extrema_shape_path"]},
  ],
}


def _qualification_live_path(region):
  """Check a window-to-result witness against the exported live DAG."""
  dag = region.get("live_dag") or {}
  values, ops, operands = (dag.get(key, []) for key in ("values", "ops", "operands"))
  if not (len(values) > len(ops) == len(operands)) or len(set(values)) != len(values):
    return False
  # The export lists live inputs first, followed by live operation results.
  input_count = len(values) - len(ops)
  producers = {value: index for index, value in enumerate(values[input_count:])}
  window = (region.get("window") or {}).get("result")
  roots = set(dag.get("roots") or [])
  for witness in region.get("requirements") or []:
    path, operations = witness.get("path") or [], witness.get("operations") or []
    if (
      not window
      or witness.get("input") != window
      or len(path) < 3
      or path[0] != window
      or path[-1] not in roots
      or path[-1] not in (witness.get("results") or [])
      or len(operations) != len(path) - 1
      or len(set(path)) != len(path)
    ):
      continue
    ids = [producers.get(value) for value in path[1:]]
    if any(index is None for index in ids) or witness.get("operation_ids") != ids:
      continue
    if not all(
      ops[index] == op and source in operands[index]
      for source, index, op in zip(path, ids, operations)
    ):
      continue
    if set(operations) & {"max", "min", "rmin", "rmax"} and set(operations) & {
      "transpose",
      "expand",
    }:
      return True
  return False


def qualification_capabilities(built, required):
  """Check exported roles, connected witnesses, and the layout actually staged."""
  from test.lib.tensor_free_form_inputs import SpecError, _window_roles

  checks = {name: False for name in required}
  region = built.get("region") or {}
  try:
    roles = _window_roles(region)
  except (SpecError, TypeError, KeyError):
    return checks
  window, backing, states = roles["window"], roles["backing"], roles["states"]
  if "backing_storage" in checks:
    checks["backing_storage"] = bool(
      window.get("source_kind") == "backing"
      and backing["shape"][0] > states[0]["shape"][0]
    )
  if "changing_active_rows" in checks:
    active = [
      [
        trip * window["stride"] + row < window["active"]
        for row in range(window["rows"])
      ]
      for trip in range(roles["trips"])
    ]
    witness = window.get("witness") or {}
    checks["changing_active_rows"] = bool(
      active == witness.get("per_trip_active")
      and any(len({trip[row] for trip in active}) > 1 for row in range(window["rows"]))
      and witness.get("all_active") is False
    )
  if "strided_backing_layout" in checks:
    bundles = built.get("bundles") or {}
    staged = []
    for bundle in bundles.values():
      try:
        manifest = json.loads((Path(bundle) / "case.json").read_text())
        layouts = [
          entry for entry in manifest["inputs"] if entry["name"] == backing["name"]
        ]
      except (OSError, ValueError, KeyError, TypeError):
        return checks
      if len(layouts) != 1:
        return checks
      layout = layouts[0]
      shape, strides = layout.get("shape"), layout.get("element_strides")
      dense = []
      product = 1
      for extent in reversed(backing["shape"]):
        dense.insert(0, product)
        product *= extent
      staged.append(
        bool(
          shape == backing["shape"]
          and isinstance(strides, list)
          and len(strides) == len(dense)
          and all(type(value) is int and value > 0 for value in strides)
          and any(
            extent > 1 and stride != contiguous
            for extent, stride, contiguous in zip(shape, strides, dense)
          )
        )
      )
    checks["strided_backing_layout"] = bool(staged and all(staged))
  if "extrema_shape_path" in checks:
    checks["extrema_shape_path"] = _qualification_live_path(region)
  return checks


def require_qualification(work, frozen):
  """Refuse production dispatch without a complete, matching R4 record."""
  work = Path(work)
  declaration = frozen.spec.get("qualification")
  if declaration != QUALIFICATION:
    raise StudyRefused(
      "the frozen study has no current qualification declaration"
    )
  ledger = work / "path-ledger.jsonl"
  if not ledger.is_file():
    raise StudyRefused("qualification outcome history is missing")
  reviews = _qualification_reviews(work, PathLedger(ledger).load().events, frozen)
  if any(review["unresolved"] for review in reviews):
    raise StudyRefused("qualification has unresolved candidate outcomes")
  path = work / "qualification.json"
  if not path.is_file():
    raise StudyRefused("qualification has not been recorded")
  record = json.loads(path.read_text(encoding="utf-8"))
  digest = record.pop("digest", None)
  if digest != _digest(record):
    raise StudyRefused("the qualification record changed after its digest")
  record["digest"] = digest
  if record.get("schema") != "refractir.moving-window-qualification.v3":
    raise StudyRefused("the qualification record has an unsupported schema")
  if record.get("study") != frozen.digest:
    raise StudyRefused("the qualification record belongs to another study")
  if record.get("declaration") != declaration:
    raise StudyRefused("the qualification declaration changed")
  cases = record.get("cases") or []
  by_case = {case.get("case"): case for case in cases}
  if sorted(by_case) != sorted(entry["case"] for entry in declaration["cases"]):
    raise StudyRefused("qualification does not cover the four declared cases")
  for entry in declaration["cases"]:
    case = by_case[entry["case"]]
    if not case.get("complete") or not case.get("validated"):
      raise StudyRefused(f"qualification case {entry['case']} is incomplete")
    if case.get("capabilities") != {name: True for name in entry["checks"]}:
      raise StudyRefused(f"qualification case {entry['case']} lost a capability")
    configurations = case.get("configurations") or {}
    for configuration in CONFIGURATIONS:
      outcome = configurations.get(configuration) or {}
      if outcome.get("paths", {}).get("p1", {}).get("status") != "completed":
        raise StudyRefused(f"qualification case {entry['case']} lacks {configuration} P1")
      if outcome.get("paths", {}).get("p2", {}).get("status") != "completed":
        raise StudyRefused(f"qualification case {entry['case']} lacks {configuration} P2")
  if not record.get("validated"):
    raise StudyRefused("the qualification record is not validated")
  return record


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
    "accounting": {"version": 2, "mode": "process-events"},
    "limits": {"max_nodes": 16},
    "qualification": QUALIFICATION,
    "triage_policy": candidate_triage.POLICY,
  }


def _validate_spec(spec, *, allow_legacy_qualification=False):
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
  if "accounting" in spec and spec["accounting"] != {
    "version": 2, "mode": "process-events"
  }:
    raise tensor_tranche.ManifestRefused("the study accounting version is unknown")
  if "qualification" in spec and spec["qualification"] != QUALIFICATION:
    legacy = {
      json.dumps({**QUALIFICATION, "version": 1, "max_multiplications": value},
                 sort_keys=True)
      for value in (0, 1)
    }
    legacy.add(json.dumps({**QUALIFICATION, "version": 1}, sort_keys=True))
    if (
      not allow_legacy_qualification
      or json.dumps(spec["qualification"], sort_keys=True) not in legacy
    ):
      raise tensor_tranche.ManifestRefused("the qualification declaration changed")
  if "limits" in spec and spec["limits"] != {"max_nodes": 16}:
    raise tensor_tranche.ManifestRefused("the study node limit changed")
  if "triage_policy" in spec and spec["triage_policy"] != candidate_triage.POLICY:
    raise tensor_tranche.ManifestRefused("the candidate triage policy changed")
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
    # Reading an archived declaration does not authorize current dispatch.
    _validate_spec(spec, allow_legacy_qualification=True)
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


def path_counts(events, *, legacy=False):
  """Launches, cached observations, failures, and statuses by path.

  New runs count only recorded process starts. Historical records carry
  launch events instead; `legacy=True` is their explicit compatibility mode.
  """
  if legacy:
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
  counts = {
    path: {
      "launches": 0, "cached": 0, "failures": 0, "not_run": 0,
      "spawn_failed": 0, "interrupted": 0, "statuses": {},
    }
    for path in ("p0", "p1", "p2")
  }
  starts = {}
  finishes = {}
  for event in events:
    path = event.get("path")
    if path not in counts:
      continue
    if event.get("kind") == "process":
      if event.get("event") == "start":
        counts[path]["launches"] += 1
        starts[path] = starts.get(path, 0) + 1
      elif event.get("event") == "finish":
        status = event.get("status", "failed")
        if status == "spawn_failed":
          counts[path]["spawn_failed"] += 1
        else:
          finishes[path] = finishes.get(path, 0) + 1
          counts[path]["statuses"][status] = counts[path]["statuses"].get(status, 0) + 1
          if status != "completed":
            counts[path]["failures"] += 1
    elif event.get("kind") == "cached":
      counts[path]["cached"] += 1
    elif event.get("kind") == "not_run":
      counts[path]["not_run"] += 1
  for path in counts:
    counts[path]["interrupted"] = max(0, starts.get(path, 0) - finishes.get(path, 0))
  return counts

def parse_triage_ledger(data, source=None):
  """Validate and normalize triage rows; arbitrary line counts never charge."""
  try:
    text = data.decode("utf-8") if isinstance(data, bytes) else str(data)
  except UnicodeDecodeError as error:
    raise StudyRefused(f"the triage ledger is not UTF-8: {error}") from error
  entries = []
  seen = {}
  for number, line in enumerate(text.splitlines(), start=1):
    if not line.strip():
      continue
    try:
      row = json.loads(line)
    except json.JSONDecodeError as error:
      raise StudyRefused(
        f"the triage ledger row {number} is not JSON: {error}"
      ) from error
    if not isinstance(row, dict):
      raise StudyRefused(f"the triage ledger row {number} is not an object")
    if not isinstance(row.get("case"), str) or not row["case"]:
      raise StudyRefused(f"the triage ledger row {number} has no case")
    if not isinstance(row.get("result"), str) or not row["result"]:
      raise StudyRefused(f"the triage ledger row {number} has no result")
    path = row.get("path", "p2")
    allocation = row.get("allocation", "triage")
    if path not in ("p0", "p1", "p2"):
      raise StudyRefused(f"the triage ledger row {number} has an unknown path")
    if allocation not in ("triage", "qualification", "selection", "discovery"):
      raise StudyRefused(f"the triage ledger row {number} has an unknown allocation")
    if row["result"] not in (
      "completed", "failed", "timed_out", "hung", "refused", "spawn_failed",
    ):
      raise StudyRefused(f"the triage ledger row {number} has an unknown outcome")
    supplied = row.get("execution_id")
    if supplied is not None and (not isinstance(supplied, str) or not supplied):
      raise StudyRefused(
        f"the triage ledger row {number} has an invalid execution_id"
      )
    attempt = row.get("attempt")
    if supplied is None and (type(attempt) is not int or attempt < 0):
      raise StudyRefused(
        f"the triage ledger row {number} needs an execution_id or a valid attempt"
      )
    key = supplied or hashlib.sha256(
      json.dumps(
        {
          "case": row["case"],
          "attempt": row.get("attempt"),
          "path": path,
          "allocation": allocation,
        },
        sort_keys=True,
      ).encode("utf-8")
    ).hexdigest()
    entry = {
      "execution_id": key,
      "case": row["case"],
      "attempt": row.get("attempt"),
      "result": row["result"],
      "path": path,
      "allocation": allocation,
    }
    if key in seen:
      previous = seen[key]
      if any(
        previous.get(field) != entry.get(field)
        for field in ("case", "attempt", "result", "path", "allocation")
      ):
        raise StudyRefused(
          f"the triage ledger has conflicting duplicate rows at {number}"
        )
      continue
    seen[key] = entry
    entries.append(entry)
  if not entries:
    raise StudyRefused("the triage ledger has no entries")
  return entries


class Study:
  """One run of the frozen study: selection, discovery, and accounting."""

  def __init__(
    self,
    work,
    frozen,
    *,
    build,
    screen,
    execute,
    controls_root=None,
    triage_ledger=None,
    legacy_accounting=None,
    accounting_mode=None,
  ):
    self.work = Path(work)
    self.frozen = frozen
    self.build = build
    self.screen = screen
    self.execute = execute
    self.controls_root = None if controls_root is None else Path(controls_root)
    self.triage_ledger = None if triage_ledger is None else Path(triage_ledger)
    self._imports_override = None
    if legacy_accounting is not None:
      legacy_mode = "legacy" if legacy_accounting else "process"
      if accounting_mode is not None and accounting_mode != legacy_mode:
        raise StudyRefused("conflicting accounting arguments")
      accounting_mode = legacy_mode
    self.accounting_mode = accounting_mode
    self.legacy_accounting = frozen.spec.get("accounting") is None
    self.path_ledger = PathLedger(self.work / "path-ledger.jsonl")
    self.corpus_path = self.work / "corpus.jsonl"
    self.ledger = None

  def _resolve_accounting(self, *, persist=False):
    path = self.work / "accounting-adoption.json"
    adoption = json.loads(path.read_text()) if path.is_file() else None
    try:
      mode = accounting.resolve_mode(
        self.frozen.spec,
        self.path_ledger.events,
        requested=self.accounting_mode,
        adoption=adoption,
        digest=self.frozen.digest,
      )
    except ValueError as error:
      raise StudyRefused(str(error)) from error
    self.legacy_accounting = mode == "legacy"
    if (
      persist
      and self.frozen.spec.get("accounting") is None
      and adoption is None
      and self.accounting_mode is not None
    ):
      tensor_execution.write_atomic(
        path,
        _canonical(
          {
            "study_digest": self.frozen.digest,
            "mode": mode,
          }
        ).encode("utf-8"),
      )

  def _accepted_imports_path(self):
    return self.work / "accepted-imports.json"

  def accepted_imports(self):
    if self._imports_override is not None:
      return list(self._imports_override)
    path = self._accepted_imports_path()
    if not path.is_file():
      return []
    return json.loads(path.read_text(encoding="utf-8"))

  def _accept_triage_ledger(self):
    if self.triage_ledger is None:
      return
    if not self.triage_ledger.is_file():
      raise StudyRefused(f"the supplied triage ledger is missing: {self.triage_ledger}")
    data = self.triage_ledger.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    entries = parse_triage_ledger(data, str(self.triage_ledger))
    imports = self.accepted_imports()
    if any(record.get("ledger_sha256") == digest for record in imports):
      return
    if imports:
      raise StudyRefused(
        "a different triage ledger is already accepted for this study"
      )
    paths = {}
    for entry in entries:
      paths[entry["path"]] = paths.get(entry["path"], 0) + 1
    imports.append(
      {
        "ledger_sha256": digest,
        "entries": entries,
        "paths": paths,
        "source": str(self.triage_ledger),
      }
    )
    try:
      accounting.used(self.path_ledger.events, "triage", "p2", imports)
    except ValueError as error:
      raise StudyRefused(str(error)) from error
    tensor_execution.write_atomic(
      self._accepted_imports_path(),
      (json.dumps(imports, indent=1, sort_keys=True) + "\n").encode(),
    )

  def prepare(self):
    self.work.mkdir(parents=True, exist_ok=True)
    # Validate every existing identity before any import, decision, or ledger
    # mutation. A request that is going to be refused must leave evidence
    # byte-identical.
    study_path = self.work / "study.json"
    if study_path.is_file():
      existing = FrozenStudy.load(study_path)
      if existing.digest != self.frozen.digest:
        raise tensor_tranche.ManifestRefused(
          "the work directory belongs to another frozen study"
        )
    self.path_ledger.load()
    self._resolve_accounting()
    if not study_path.is_file():
      self.frozen.write(study_path)
    self.ledger = tensor_tranche.Ledger.open(
      self.work / "tranche-ledger.jsonl", self.frozen.tranche
    )
    self._resolve_accounting(persist=True)
    self._accept_triage_ledger()
    self._replay_decisions()
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

  def pause(self, candidates, *, reason, phase="discovery", qualification_attempt=None):
    """Persist candidates and stop routine dispatch until triage decides."""
    pause_id = hashlib.sha256(
      (reason + ":" + str(len(self.path_ledger.events))).encode("utf-8")
    ).hexdigest()[:16]
    record = {
      "pause_id": pause_id,
      "phase": phase,
      "reason": reason,
      "candidates": candidates,
      **(
        {"qualification_attempt": qualification_attempt}
        if qualification_attempt is not None
        else {}
      ),
    }
    (self.work / "study-paused.json").write_text(
      json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    self.path_ledger.append(
      "paused",
      reason=reason,
      candidates=len(candidates),
      phase=phase,
      pause_id=pause_id,
      candidate_sha256=_digest(candidates) if candidates else None,
    )

  def record_triage_decision(self, decision):
    """Persist, apply, and mark one decision for the active pause."""
    if not self.is_paused():
      raise StudyRefused("the study is not paused")
    pause = json.loads(
      (self.work / "study-paused.json").read_text(encoding="utf-8")
    )
    record = dict(decision) if isinstance(decision, dict) else {"action": decision}
    action = record.get("action")
    if action not in ("admit", "reject", "resume", "stop"):
      raise StudyRefused(f"unknown triage action: {action}")
    pending = self._pending_admission()
    if action in ("admit", "reject") and pending is None:
      raise StudyRefused("no pending admission to decide")
    if (
      action == "admit"
      and pending is not None
      and pending.get("screening_complete") is False
    ):
      raise StudyRefused(
        "an incomplete screen cannot be admitted; reject or stop this attempt"
      )
    if action == "resume" and pending is not None:
      raise StudyRefused(
        "resume cannot resolve a pending admission; use admit, reject, or stop"
      )
    if (
      self.frozen.spec.get("triage_policy") == candidate_triage.POLICY
      and pause.get("candidates")
      and action != "stop"
    ):
      try:
        record["disposition"] = candidate_triage.validate(
          self.work, record.get("disposition")
        )
      except ValueError as error:
        raise StudyRefused(str(error)) from error
      if (
        action == "admit"
        and record["disposition"]["classification"] == "invalid_execution"
      ):
        raise StudyRefused("an invalid execution cannot be admitted")
      record["candidate_sha256"] = _digest(pause["candidates"])
    record = {
      **record,
      "pause_id": pause.get("pause_id"),
      "phase": pause.get("phase"),
      **(
        {"qualification_attempt": pause["qualification_attempt"]}
        if "qualification_attempt" in pause
        else {}
      ),
      "case": (pending or {}).get("case"),
      "attempt": (pending or {}).get("attempt"),
    }
    record["decision_id"] = hashlib.sha256(
      json.dumps(record, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    with (self.work / "triage-decisions.jsonl").open("a", encoding="utf-8") as handle:
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
    with (self.work / "triage-applied.jsonl").open("a", encoding="utf-8") as handle:
      handle.write(
        json.dumps({"decision_id": record["decision_id"]}, sort_keys=True) + "\n"
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
    if pending.get("screening_complete") is False:
      raise StudyRefused("an incomplete screen cannot be admitted")
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

  def _applied_decisions(self):
    path = self.work / "triage-applied.jsonl"
    if not path.is_file():
      return set()
    return {
      json.loads(line).get("decision_id")
      for line in path.read_text(encoding="utf-8").splitlines()
      if line.strip()
    }

  def _replay_decisions(self):
    """Finish or apply the last decision for the active pause, exactly once."""
    pause_path = self.work / "study-paused.json"
    if not pause_path.is_file():
      return False
    pause = json.loads(pause_path.read_text(encoding="utf-8"))
    pause_id = pause.get("pause_id")
    decisions = [
      decision for decision in self._decisions()
      if decision.get("pause_id") == pause_id
    ]
    if not decisions:
      return False
    decision = decisions[-1]
    if (
      self.frozen.spec.get("triage_policy") == candidate_triage.POLICY
      and pause.get("candidates")
      and decision.get("action") != "stop"
    ):
      if decision.get("candidate_sha256") != _digest(pause["candidates"]):
        raise StudyRefused("triage decision names different candidate evidence")
      try:
        candidate_triage.validate(self.work, decision.get("disposition"))
      except ValueError as error:
        raise StudyRefused(str(error)) from error
    applied = decision.get("decision_id") in self._applied_decisions()
    action = decision.get("action")
    pending = self._pending_admission()
    if not applied:
      if action == "admit":
        if pending is not None:
          self._admit_pending()
        elif decision.get("case") not in self._selected_ids():
          return False
      elif action == "reject":
        if pending is not None:
          self._clear_pending()
      elif action == "stop":
        (self.work / "study-stopped.json").write_text(
          json.dumps(decision, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
      elif action != "resume":
        return False
      with (self.work / "triage-applied.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
          json.dumps({"decision_id": decision["decision_id"]}, sort_keys=True) + "\n"
        )
    elif action == "stop":
      (self.work / "study-stopped.json").write_text(
        json.dumps(decision, indent=1, sort_keys=True) + "\n", encoding="utf-8"
      )
    self._clear_pending()
    pause_path.unlink(missing_ok=True)
    return True

  def _reconcile_pending(self):
    if self._pending_admission() is None:
      return
    if self._replay_decisions():
      return
    raise StudyRefused(
      "a pending admission needs an admit, reject, or stop decision"
    )

  def is_stopped(self):
    return (self.work / "study-stopped.json").is_file()

  def _reserve_process_budget(self, phase, event):
    if event.get("event") != "reserved":
      return
    path = event.get("path")
    try:
      key = accounting.budget_key(phase, path)
      used = accounting.used(
        self.path_ledger.events, phase, path, self.accepted_imports()
      )
    except ValueError as error:
      raise StudyRefused(str(error)) from error
    limit = 0 if key is None else self.frozen.spec["budgets"][key]
    if used >= limit:
      self.path_ledger.append(
        "launch_refused", phase=phase, path=path, reason="budget_exhausted"
      )
      raise DispatchRefused(
        f"the {phase} {path} allocation is exhausted ({used}/{limit})"
      )

  def process_observer(self, phase, **context):
    """Reserve and record one sequential callback stream, before each Popen."""
    ordinals = {path: 0 for path in accounting.PATHS}
    active = {}

    def account(event):
      path = event.get("path")
      if path not in accounting.PATHS:
        raise StudyRefused("process callback is missing its execution path")
      kind = event.get("event")
      if kind == "reserved":
        identity = _digest(
          {"phase": phase, **context, "path": path, "ordinal": ordinals[path]}
        )
        imported = (
          entry
          for record in self.accepted_imports()
          for entry in record.get("entries", [])
        )
        if any(
          e.get("execution_id") == identity for e in self.path_ledger.events
        ) or any(e["execution_id"] == identity for e in imported):
          raise StudyRefused(
            "a process reservation already exists; use a fresh explicit attempt"
          )
        self._reserve_process_budget(phase, event)
        active[path] = identity
        ordinals[path] += 1
      else:
        identity = active.get(path)
        if identity is None:
          raise StudyRefused("a process event has no reservation")
        history = [
          e for e in self.path_ledger.events if e.get("execution_id") == identity
        ]
        if any(e.get("event") == "finish" for e in history):
          if kind == "finish" and all(
            history[-1].get(k) == v for k, v in event.items()
          ):
            return
          raise StudyRefused("process already finished")
        started = any(e.get("event") == "start" for e in history)
        if kind == "start" and started:
          raise StudyRefused("duplicate process start")
        if kind not in ("start", "finish") or (
          kind == "finish" and not started and event.get("status") != "spawn_failed"
        ):
          raise StudyRefused("invalid process lifecycle transition")
      self.path_ledger.append(
        "process", **context, **event, phase=phase, execution_id=identity
      )

    return account

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
          "max_nodes": self.frozen.spec.get("limits", {}).get("max_nodes", 12),
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
        screening_account = self.process_observer(
          "screening",
          case=slot["id"],
          attempt=attempt,
        )

        outcome["on_process"] = screening_account
        try:
          screened = self.screen(outcome)
        except DispatchRefused as error:
          self.path_ledger.append(
            "dispatch_refused", case=slot["id"], attempt=attempt,
            phase="screening", reason=str(error),
          )
          self._save_pending(
            {
              "case": slot["id"],
              "attempt": attempt,
              "seed": seed,
              "entry": self._corpus_entry(slot, outcome, {}),
              "candidates": [],
              "screening_complete": False,
            }
          )
          self.pause(
            [{"kind": "dispatch_refused", "phase": "screening", "reason": str(error)}],
            reason=f"dispatch refused during screening for {slot['id']}",
            phase="selection",
          )
          return self
        self.path_ledger.append(
          "selection_screen",
          case=slot["id"],
          selected=bool(screened.get("selected")),
          reason=screened.get("reason"),
          launches=screened.get("launches", 0),
          p0=screened.get("p0", 0),
          record=screened.get("record", {}),
        )
        candidates = list(screened.get("candidates", []))
        invalid = [c for c in candidates if c.get("kind") == "invalid_execution"]
        if invalid:
          self.path_ledger.append(
            "selection_invalid_execution", case=slot["id"], candidates=invalid
          )
          candidates = [c for c in candidates if c.get("kind") != "invalid_execution"]
        if screened.get("dispatch_refused"):
          if candidates:
            self.path_ledger.append(
              "selection_candidates", case=slot["id"], candidates=candidates
            )
          self._save_pending(
            {
              "case": slot["id"],
              "attempt": attempt,
              "seed": seed,
              "entry": self._corpus_entry(slot, outcome, screened),
              "candidates": candidates,
              "screening_complete": False,
            }
          )
          self.pause(candidates, reason=screened["dispatch_refused"], phase="selection")
          return self
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
            phase="selection",
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
      account = self.process_observer(
        "discovery",
        slot=slot.slot_id,
        case=slot.case_id,
        attempt=attempt,
        directory=str(directory),
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
      except DispatchRefused as error:
        classification = {
          "slot": slot.slot_id,
          "case": slot.case_id,
          "status": "dispatch_refused",
          "candidate": False,
          "unavailable": [str(error)],
          "compiler": {"status": "unavailable"},
        }
        self.path_ledger.append("dispatch_refused", **classification)
        self.path_ledger.append("classification", **classification)
        self.ledger.finish(
          slot.slot_id, "refused", reason=str(error),
          detail={"status": "dispatch_refused"},
        )
        self.pause(
          [{"kind": "dispatch_refused", "phase": "discovery", "reason": str(error)}],
          reason=f"dispatch refused during discovery at {slot.slot_id}",
          phase="resource",
        )
        return self
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
      slot_processes = [
        event for event in self.path_ledger.events
        if event.get("kind") == "process"
        and event.get("slot") == slot.slot_id
        and event.get("attempt") == attempt
      ]
      started = {
        event.get("path") for event in slot_processes
        if event.get("event") == "start"
      }
      for path in ("p0", "p1", "p2"):
        info = outcome.get("paths", {}).get(path)
        if info is None:
          continue
        if info.get("cached"):
          self.path_ledger.append(
            "cached", path=path, slot=slot.slot_id, case=slot.case_id,
            attempt=attempt, key=info.get("key"),
          )
        elif slot_processes or not self.legacy_accounting:
          # New runs trust lifecycle evidence: no start means no launch.
          if path in started:
            self.path_ledger.append(
              "launch", path=path, slot=slot.slot_id, case=slot.case_id,
              attempt=attempt, directory=info.get("directory", str(directory)),
              status=info.get("status", "completed"),
            )
          else:
            self.path_ledger.append(
              "not_run", path=path, slot=slot.slot_id, case=slot.case_id,
              attempt=attempt, status=info.get("status", "unavailable"),
              reason="no process start was recorded",
            )
        else:
          # Host fixtures and historical records supply no lifecycle events.
          self.path_ledger.append(
            "launch", path=path, slot=slot.slot_id, case=slot.case_id,
            attempt=attempt, directory=info.get("directory", str(directory)),
            status=info.get("status", "completed"),
          )
      if outcome.get("dispatch_refused"):
        classification = {
          "slot": slot.slot_id,
          "case": slot.case_id,
          "status": "dispatch_refused",
          "candidate": bool(outcome.get("candidate")),
          "dispatch_refused": outcome["dispatch_refused"],
          "unavailable": list(outcome.get("unavailable", [])),
          "compiler": outcome.get("compiler", {"status": "unavailable"}),
        }
        self.path_ledger.append("dispatch_refused", **classification)
        self.path_ledger.append("classification", **classification)
        self.ledger.finish(
          slot.slot_id, "refused", reason=outcome["dispatch_refused"],
          detail={"status": "dispatch_refused", "candidate": classification["candidate"]},
        )
        self.pause(
          [classification], reason=f"dispatch refused at {slot.slot_id}",
          phase="resource",
        )
        return self
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

  def triage_charge(self, path=None):
    """Charges from validated imports, counted by their execution path."""
    total = 0
    for record in self.accepted_imports():
      if "paths" in record:
        paths = record["paths"]
        total += sum(paths.values()) if path is None else paths.get(path, 0)
      else:  # accepted-imports from the earlier scalar format
        if path is None or record.get("path") == path:
          total += record.get("charge", 0)
    return total

  def reserve_spent(self, path=None):
    if not self.legacy_accounting:
      return accounting.used(
        self.path_ledger.events, "triage", path or "p2", self.accepted_imports()
      )
    retries = sum(1 for e in self.path_ledger.events if e.get("kind") == "retry")
    if path is None:
      return retries + self.triage_charge()
    return retries + self.triage_charge(path)

  def retry(self, slot_id, *, case_id, input, configuration, reason):
    """Reserve a fresh directory; process-mode retries charge only actual paths.

    Pass process_observer("triage", slot=slot_id, directory=str(directory))
    to the executor. Directory creation alone authorizes no subprocess.
    """
    if not self._disk_guard():
      return None
    budgets = self.frozen.spec["budgets"]
    spent = self.reserve_spent()
    limits = {
      "p2": budgets["reserve_gpu"],
      "p1": budgets["reserve_interpreter"],
      "p0": budgets["reserve_p0"],
    }
    needed = limits if self.legacy_accounting else {p: limits[p] for p in ("p1", "p2")}
    if any(self.reserve_spent(path) >= limit for path, limit in needed.items()):
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
      reserve=spent + int(self.legacy_accounting),
      process_accounting=not self.legacy_accounting,
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
    qualification = [e for e in events if e.get("kind") == "qualification_outcome"]
    candidates += [
      {"phase": "qualification", **e}
      for e in qualification
      if e["outcome"].get("candidate")
    ]
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
                1
                for e in events
                if e.get("kind") == "process" and e.get("event") == "start"
              ),
              "finished": sum(
                1
                for e in events
                if e.get("kind") == "process" and e.get("event") == "finish"
              ),
              "by_path": {
                path: {
                  "started": sum(
                    1
                    for e in events
                    if e.get("kind") == "process"
                    and e.get("event") == "start"
                    and e.get("path") == path
                  ),
                  "finished": sum(
                    1
                    for e in events
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
      "paths": path_counts(events, legacy=self.legacy_accounting),
      "candidates": candidates,
      **(
        {"candidate_triage": candidate_triage.summary(self.work, events)}
        if self.frozen.spec.get("triage_policy") == candidate_triage.POLICY
        else {}
      ),
      **({"qualification": qualification} if qualification else {}),
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
        **(
          {
            "accounting": "reserved-or-started",
            "by_path": {
              path: {
                "spent": self.reserve_spent(path),
                "remaining": max(
                  0,
                  self.frozen.spec["budgets"][accounting.budget_key("triage", path)]
                  - self.reserve_spent(path),
                ),
              }
              for path in accounting.PATHS
            },
          }
          if not self.legacy_accounting
          else {}
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


def report_from_records(work, *, triage_ledger=None, accounting_mode=None):
  """Recompute the final summary from the preserved records alone."""
  work = Path(work)
  frozen = FrozenStudy.load(work / "study.json")
  if (work / tensor_campaign_evidence.NAME).is_file():
    tensor_campaign_evidence.verify(work, study_digest=frozen.digest)
  study = Study(
    work,
    frozen,
    build=None,
    screen=None,
    execute=None,
    triage_ledger=triage_ledger,
    accounting_mode=accounting_mode,
  )
  if triage_ledger is not None:
    ledger = Path(triage_ledger)
    if not ledger.is_file():
      raise StudyRefused(f"the triage ledger is missing: {ledger}")
    data = ledger.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    entries = parse_triage_ledger(data, str(ledger))
    accepted = study.accepted_imports()
    if accepted:
      if not any(record.get("ledger_sha256") == digest for record in accepted):
        raise StudyRefused(
          "the supplied triage ledger is not the accepted import for this study"
        )
    else:
      paths = {}
      for entry in entries:
        paths[entry["path"]] = paths.get(entry["path"], 0) + 1
      study._imports_override = [
        {
          "ledger_sha256": digest,
          "entries": entries,
          "paths": paths,
          "source": str(ledger),
        }
      ]
  study.path_ledger.load()
  study._resolve_accounting()
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
    node_limit = request.get("max_nodes", 12)
    if type(node_limit) is not int or node_limit <= 0:
      raise StudyRefused("the requested node limit is invalid")
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
      "--max-nodes", str(node_limit),
      "--max-multiplications", str(request.get("max_multiplications", 1)),
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
    if type(region.get("max_nodes")) is not int or region["max_nodes"] != node_limit:
      raise StudyRefused("the exporter node limit differs from the frozen request")
    nodes = (region.get("structure") or {}).get("nodes")
    if type(nodes) is not int or not 0 <= nodes <= node_limit:
      raise StudyRefused("the exported live node count exceeds the frozen limit")
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
    starts = {"p0": 0, "p1": 0}

    def p1_process(event):
      if process is not None:
        process({**event, "path": "p1"})
      if event.get("event") == "start":
        starts["p1"] += 1

    def p0_process(event):
      if process is not None:
        process({**event, "path": "p0"})
      if event.get("event") == "start":
        starts["p0"] += 1

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
        if observation.status == tensor_target_reference.NOT_RUN:
          result["dispatch_refused"] = observation.reason
          result["selection"]["reason"] = "incomplete_observation"
          result["selection"]["selected"] = False
          selected = False
          break
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
    result = {
      **result,
      "p0": p0,
      "candidates": candidates,
      "unavailable": unavailable,
      "admission_selected": selected,
    }
    tensor_execution.write_atomic(
      Path(outcome["directory"]) / "screening.json",
      tensor_execution.canonical_json(result).encode("utf-8"),
    )
    return {
      "selected": selected,
      "dispatch_refused": result.get("dispatch_refused"),
      "reason": result["selection"]["reason"],
      "launches": starts["p1"],
      "p0": starts["p0"],
      "candidates": candidates,
      "unavailable": unavailable,
      "record": {**result, "p0": p0},
    }

  reference_cache = {}

  def execute(request):
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
      current_p0_refused = False
      if reference is None:
        reference = tensor_target_reference.observe(
          case, symiri=symiri, timeout_s=timeout_s,
          output_path=directory / "p0.bin",
          on_process=p0_process,
        )
        current_p0_refused = (
          reference.status == tensor_target_reference.NOT_RUN
        )
        if reference.status == tensor_target_reference.COMPLETED:
          reference_cache[cache_key] = reference
      run = tensor_target_run.run(
        bundle,
        symirc=symirc,
        symiri=symiri,
        reference=reference,
        dispatch_refused=reference.reason if current_p0_refused else None,
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
        "dispatch_refused": run.dispatch_refused,
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
    current_p0_refused = False
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
    if reference is None and not request.get("force_reference"):
      reference = reference_cache.get(cache_key)
      cached = reference is not None
      cache_state = "hit" if cached else "miss"
    if reference is None and not request.get("force_reference"):
      selection_p0 = [
        entry
        for entry in ((corpus.get("screen") or {}).get("p0") or [])
        if entry.get("probe") == probe
      ]
      if selection_p0:
        entry = selection_p0[0]
        status = entry.get("status")
        if status == tensor_target_reference.COMPLETED and entry.get("output_sha256"):
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
        elif status in (
          tensor_target_reference.UNAVAILABLE,
          tensor_target_reference.FAILED,
          tensor_target_reference.TIMED_OUT,
        ):
          reference = tensor_target_reference.ReferenceObservation(
            status,
            reason=f"the selection observation was {status}",
          )
          cache_state = f"selection-{status}"
        else:
          cache_state = "selection-incomplete"
      if reference is None and cache_state == "selection-corrupt":
        reference = tensor_target_reference.ReferenceObservation(
          tensor_target_reference.UNAVAILABLE,
          reason=f"cached P0 output for {probe} is missing or fails identity validation",
        )
    if reference is None:
      reference = tensor_target_reference.observe(
        case,
        symiri=symiri,
        timeout_s=timeout_s,
        on_process=p0_process,
        output_path=request.get("reference_output_path"),
      )
      current_p0_refused = (
        reference.status == tensor_target_reference.NOT_RUN
      )
    if request.get("on_reference") is not None:
      request["on_reference"](reference)
    run = tensor_target_run.run(
      bundle,
      symirc=symirc,
      symiri=symiri,
      reference=reference,
      dispatch_refused=reference.reason if current_p0_refused else None,
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
        "reason": run.compiled.reason,
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
      "dispatch_refused": run.dispatch_refused,
      "unavailable": unavailable,
      "compiler": compiler,
      "reference": reference,
      "record": run.as_record(),
    }

  return {"build": build, "screen": screen, "execute": execute, "toolchain": toolchain}


def _qualification_toolchain(toolchain):
  required = (
    "exporter_sha256",
    "symirc_sha256",
    "symiri_sha256",
    "triton_library_sha256",
    "triton",
    "python",
    "device",
  )
  if not all(toolchain.get(key) for key in required):
    raise StudyRefused(
      "qualification requires the complete qualified toolchain identity"
    )


def _qualification_identity(built, frozen, toolchain):
  """Bind the build record and actual bundle payloads before reusing a result."""
  _qualification_toolchain(toolchain)
  region = built.get("region") or {}
  probes = built.get("probe_values") or {}
  if (
    not built.get("source")
    or "i0-base" not in probes
    or not region.get("profile")
    or not region.get("entry")
  ):
    raise StudyRefused(
      "qualification build has incomplete source/input/profile/entry identity"
    )
  bundles = built.get("bundles") or {}
  if "i0-base" not in bundles:
    raise StudyRefused("qualification build has no base input bundle")
  payloads = {}
  for name, directory in sorted(bundles.items()):
    for filename in ("program.sir", "inputs.bin", "case.json"):
      path = Path(directory) / filename
      if not path.is_file():
        raise StudyRefused(f"qualification artifact is missing: {path}")
      payloads[f"{name}/{filename}"] = _sha256_bytes(path.read_bytes())
  if payloads["i0-base/program.sir"] != _sha256_bytes(built["source"].encode("utf-8")):
    raise StudyRefused("qualification source payload differs from the build")
  if payloads["i0-base/inputs.bin"] != _sha256_bytes(bytes(probes["i0-base"])):
    raise StudyRefused("qualification input payload differs from the build")
  return {
    "study_digest": frozen.digest,
    "toolchain": toolchain,
    "build_digest": _digest(built),
    "payloads": payloads,
    "source_sha256": payloads["i0-base/program.sir"],
    "input_sha256": payloads["i0-base/inputs.bin"],
    "profile": region["profile"],
    "entry": region["entry"],
    "layouts": built.get("layouts"),
    "configurations": frozen.spec["configurations"],
  }


def _qualification_artifacts(directory):
  # P0 is a diagnostic cache with its own availability/identity handling. A
  # lost cache must not make a completed target look as if it never ran.
  directory = Path(directory)
  return {
    str(path.relative_to(directory)): _sha256_bytes(path.read_bytes())
    for path in sorted(directory.rglob("*"))
    if path.is_file()
    and path.name not in ("p0.bin", "p0.json", "reference.bin")
    and "reference" not in path.relative_to(directory).parts
  }


def _qualification_retry(token, chosen, progress):
  if token is None:
    return None
  valid = {}
  for slot in chosen:
    for configuration, attempt in (
      progress["cases"].get(slot["id"], {}).get("attempts", {}).items()
    ):
      if attempt.get("status") in ("running", "interrupted", "refused"):
        key = f"{slot['id']}/{configuration}/attempt-{attempt['attempt']}"
        valid[key] = (slot["id"], configuration, attempt["attempt"])
  if token not in valid:
    raise StudyRefused(
      "qualification retry must name an existing unfinished case/configuration/attempt-N"
    )
  return valid[token]


def _qualification_reference(work, case, progress, identity):
  from test.lib.tensor_target_reference import (
    COMPLETED,
    STATUSES,
    UNAVAILABLE,
    ReferenceObservation,
  )

  stored = progress["cases"].get(case, {})
  meta_path = Path(stored.get("reference_record", work / case / "p0.json"))
  missing_meta = (
    {"observation_id": stored["reference_observation_id"]}
    if stored.get("reference_observation_id")
    else None
  )
  if not meta_path.is_file():
    if stored.get("p0_attempted") or stored.get("attempts"):
      return ReferenceObservation(
        UNAVAILABLE, reason="qualification P0 record is missing"
      ), missing_meta
    return None, None
  try:
    meta = json.loads(meta_path.read_text())
  except (ValueError, OSError):
    return ReferenceObservation(
      UNAVAILABLE, reason="qualification P0 record is corrupt"
    ), missing_meta
  if stored.get("reference_digest") != _digest(meta):
    return ReferenceObservation(
      UNAVAILABLE, reason="qualification P0 record digest differs"
    ), missing_meta
  if meta.get("identity") != identity:
    raise StudyRefused("qualification P0 identity differs from its build")
  if meta.get("status") not in STATUSES:
    return ReferenceObservation(
      UNAVAILABLE, reason="qualification P0 status is invalid"
    ), meta
  if meta["status"] != COMPLETED:
    return ReferenceObservation(meta["status"], reason=meta.get("reason", "")), meta
  output = meta_path.with_name("p0.bin")
  data = output.read_bytes() if output.is_file() else None
  if data is None or _sha256_bytes(data) != meta.get("output_sha256"):
    return ReferenceObservation(
      UNAVAILABLE, reason="qualification P0 output is missing or corrupt"
    ), meta
  return ReferenceObservation(COMPLETED, output=data, path="qualification-cache"), meta


def _qualification_outcome(
  authority, case, configuration, attempt, outcome, record, *, provenance="execution"
):
  """Append one immutable outcome; progress.json is only the latest-state cache."""
  key = _digest(
    {
      "study": authority.frozen.digest,
      "case": case,
      "configuration": configuration,
      "attempt": attempt,
    }
  )
  fields = {
    "qualification_attempt": key,
    "case": case,
    "configuration": configuration,
    "attempt": attempt,
    "outcome": json.loads(_canonical(outcome)),
    "attempt_record": json.loads(_canonical(record)),
    "provenance": provenance,
  }
  previous = next(
    (
      e
      for e in authority.path_ledger.events
      if e.get("qualification_attempt") == key
      and e.get("kind") == "qualification_outcome"
    ),
    None,
  )
  if previous is not None:
    if any(previous.get(k) != fields[k] for k in fields if k != "provenance"):
      raise StudyRefused("a qualification attempt has conflicting outcomes")
    return previous
  return authority.path_ledger.append("qualification_outcome", **fields)


def _restore_qualification_outcomes(progress, events):
  """Recover an outcome committed before its progress-cache update."""
  for event in events:
    if event.get("kind") != "qualification_outcome":
      continue
    state = progress["cases"].get(event["case"])
    if state is None:
      raise StudyRefused("qualification outcome has no persisted build")
    record = event["attempt_record"]
    if _qualification_artifacts(record["directory"]) != record["artifacts"]:
      raise StudyRefused("qualification attempt evidence changed or is missing")
    configuration = event["configuration"]
    current = state["attempts"].get(configuration)
    if current is None or current["attempt"] <= event["attempt"]:
      state["attempts"][configuration] = dict(record)
      state["outcomes"][configuration] = dict(event["outcome"])


def _qualification_reviews(work, events, frozen):
  """Resolve candidate history from applied, evidence-bound decisions."""

  def rows(name):
    path = Path(work) / name
    return (
      [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
      if path.is_file()
      else []
    )

  applied = {row["decision_id"] for row in rows("triage-applied.jsonl")}
  decisions = {
    row["qualification_attempt"]: row
    for row in rows("triage-decisions.jsonl")
    if row.get("decision_id") in applied and row.get("qualification_attempt")
  }
  reviews = []
  for event in events:
    if event.get("kind") != "qualification_outcome":
      continue
    outcome = event["outcome"]
    if not (outcome.get("candidate") or outcome.get("dispatch_refused")):
      continue
    key = _digest(
      {
        "study": frozen.digest,
        "case": event["case"],
        "configuration": event["configuration"],
        "attempt": event["attempt"],
      }
    )
    if key != event.get("qualification_attempt"):
      raise StudyRefused("qualification outcome belongs to another attempt or study")
    decision = decisions.get(key)
    acknowledged = decision is not None and decision.get("action") == "resume"
    unresolved = bool(outcome.get("candidate"))
    if acknowledged and unresolved:
      if decision.get("candidate_sha256") != _digest([event]):
        raise StudyRefused("qualification decision names different candidate evidence")
      try:
        disposition = candidate_triage.validate(work, decision.get("disposition"))
      except ValueError as error:
        raise StudyRefused(str(error)) from error
      unresolved = disposition["classification"] == "unresolved"
    reviews.append(
      {"event": event, "acknowledged": acknowledged, "unresolved": unresolved}
    )
  return reviews


def _pause_qualification_outcomes(authority, *, diagnostic_retry=None):
  for review in _qualification_reviews(
    authority.work, authority.path_ledger.events, authority.frozen
  ):
    event = review["event"]
    if review["acknowledged"] and not review["unresolved"]:
      continue
    identity = (event["case"], event["configuration"], event["attempt"])
    # A named retry permits this diagnostic attempt only. It is not a
    # disposition and cannot authorize the next configuration or case.
    if review["unresolved"] and review["acknowledged"] and identity == diagnostic_retry:
      continue
    if not authority.is_paused():
      authority.pause(
        [event] if event["outcome"].get("candidate") else [],
        reason="qualification candidate"
        if event["outcome"].get("candidate")
        else "qualification dispatch refused",
        phase="qualification",
        qualification_attempt=event["qualification_attempt"],
      )
    return True
  return False


def _qualification_construction_stop(authority, case, reason, **details):
  event = authority.path_ledger.append(
    "qualification_construction_stopped", case=case, reason=reason, **details
  )
  authority.pause([], reason=reason, phase="qualification")
  return event


def run_qualification(
  work,
  *,
  operations,
  slots=4,
  timeout_s=300.0,
  retry=None,
  authority=None,
  reference_recheck=None,
  accounting_mode=None,
  triage_decision=None,
):
  """Qualify four cases using the same per-process authority as discovery.

  A retry names case/configuration/attempt-N. It never implies a fresh P0.
  Re-observation additionally names case/observation-id, is charged to P0
  triage, and preserves the previous observation and all target attempts.
  """
  from test.lib.tensor_target_reference import (
    COMPLETED,
    ReferenceObservation,
  )

  work = Path(work)
  toolchain = operations.get("toolchain") or {}
  _qualification_toolchain(toolchain)
  study_path = work / "study.json"
  frozen = (
    FrozenStudy.load(study_path)
    if study_path.is_file()
    else freeze_study(toolchain=toolchain)
  )
  require_toolchain(frozen, toolchain)
  if authority is None:
    authority = Study(
      work,
      frozen,
      build=None,
      screen=None,
      execute=None,
      accounting_mode=accounting_mode,
    )
  if (
    authority.work.resolve() != work.resolve()
    or authority.frozen.digest != frozen.digest
  ):
    raise StudyRefused("qualification must use the study's accounting authority")
  if type(slots) is not int or not 1 <= slots <= 4:
    raise StudyRefused("qualification has at most four frozen cases")
  declaration = frozen.spec.get("qualification")
  if declaration != QUALIFICATION:
    raise StudyRefused("the frozen study has no current qualification declaration")
  by_id = {slot["id"]: slot for slot in frozen.spec["sources"]}
  chosen = [by_id[entry["case"]] for entry in declaration["cases"]][:slots]
  progress_path = work / "qualification-progress.json"
  progress = (
    json.loads(progress_path.read_text())
    if progress_path.is_file()
    else {
      "schema": "refractir.moving-window-qualification-progress.v2",
      "cases": {},
    }
  )
  if progress.get("schema") != "refractir.moving-window-qualification-progress.v2":
    raise StudyRefused(
      "legacy qualification progress needs explicit evidence migration"
    )
  authority.path_ledger.load()
  _restore_qualification_outcomes(progress, authority.path_ledger.events)
  retry_request = _qualification_retry(retry, chosen, progress)
  recheck_request = None
  builds = {}
  identities = {}
  references = {}
  # Validate every reusable case before prepare() can import charges, recover
  # decisions, or reconcile running slots. Rejection leaves all files intact.
  for slot in chosen:
    case = slot["id"]
    path = work / case / "build.json"
    stored = progress["cases"].get(case)
    if not path.is_file():
      if stored:
        raise StudyRefused("a persisted qualification build is missing")
      continue
    built = json.loads(path.read_text())
    identity = _qualification_identity(built, frozen, toolchain)
    if stored is None or stored.get("identity") != identity:
      raise StudyRefused(
        "qualification build identity differs or was not durably frozen"
      )
    for attempt in stored.get("attempts", {}).values():
      if attempt.get("status") == "completed":
        actual = _qualification_artifacts(attempt["directory"])
        if not attempt.get("artifacts") or actual != attempt["artifacts"]:
          raise StudyRefused("completed qualification artifacts changed or are missing")
    observed, meta = _qualification_reference(work, case, progress, identity)
    builds[case], identities[case], references[case] = built, identity, observed
    if reference_recheck is not None and meta is not None:
      if reference_recheck == f"{case}/{meta.get('observation_id')}":
        recheck_request = case
  if reference_recheck is not None:
    if (
      recheck_request is None
      or retry_request is None
      or recheck_request != retry_request[0]
    ):
      raise StudyRefused(
        "reference recheck must match the retried case and its prior observation identity"
      )
  authority.prepare()
  # Migrate only preserved outcomes, never earlier overwritten attempts.
  for case, state in progress["cases"].items():
    for configuration, outcome in state["outcomes"].items():
      attempt_record = state["attempts"][configuration]
      if attempt_record["status"] in ("completed", "refused"):
        _qualification_outcome(
          authority,
          case,
          configuration,
          attempt_record["attempt"],
          outcome,
          attempt_record,
          provenance="progress-recovery",
        )
  construction_stops = [
    e
    for e in authority.path_ledger.events
    if e.get("kind") == "qualification_construction_stopped"
  ]
  if construction_stops and not authority.is_paused() and not authority.is_stopped():
    authority.pause([], reason=construction_stops[-1]["reason"], phase="qualification")
  _pause_qualification_outcomes(authority, diagnostic_retry=retry_request)
  if triage_decision is not None:
    authority.record_triage_decision(triage_decision)
    _pause_qualification_outcomes(authority, diagnostic_retry=retry_request)
  if construction_stops:
    raise StudyRefused(
      "qualification construction failed; preserve it and use a new approved freeze"
    )
  if authority.is_stopped() or authority.is_paused():
    raise StudyRefused(
      "qualification cannot dispatch while the study is paused or stopped"
    )
  diagnostic_only = any(
    review["unresolved"] for review in _qualification_reviews(
      work, authority.path_ledger.events, frozen
    )
  )
  if authority.legacy_accounting:
    raise StudyRefused("new qualification needs process accounting")
  record = {
    "schema": "refractir.moving-window-qualification.v3",
    "study": frozen.digest,
    "declaration": declaration,
    "cases": [],
  }

  def save_progress():
    tensor_execution.write_atomic(
      progress_path, (json.dumps(progress, indent=1, sort_keys=True) + "\n").encode()
    )

  for slot in chosen:
    if diagnostic_only and slot["id"] != retry_request[0]:
      continue
    if authority.is_paused() or authority.is_stopped():
      break
    case = slot["id"]
    built = builds.get(case)
    if built is None:
      reserved = any(
        e.get("kind") == "qualification_build_reserved" and e.get("case") == case
        for e in authority.path_ledger.events
      )
      if reserved or ((work / case).is_dir() and any((work / case).glob("attempt-*"))):
        reason = "uncommitted qualification build"
        _qualification_construction_stop(authority, case, reason)
        record["cases"].append({"case": case, "interrupted": reason})
        break
      for attempt in range(4):
        if not authority._disk_guard():
          break
        request = {
          "slot": slot,
          "attempt": attempt,
          "seed": frozen.spec["attempts"]["seed_base"] + 8 * slot["index"] + attempt,
          "max_attempts": 1,
          "max_multiplications": declaration.get("max_multiplications", 1),
          "max_nodes": frozen.spec.get("limits", {}).get("max_nodes", 12),
          "directory": str(work / case / f"attempt-{attempt}"),
        }
        authority.path_ledger.append(
          "qualification_build_reserved", case=case, attempt=attempt, request=request
        )
        try:
          candidate = operations["build"](request)
        except BaseException:
          _qualification_construction_stop(
            authority,
            case,
            "qualification builder interrupted",
            attempt=attempt,
            request=request,
          )
          raise
        if candidate.get("kind") != "admitted":
          authority.path_ledger.append(
            "qualification_build_refused",
            case=case,
            attempt=attempt,
            reason=candidate.get("reason"),
            request=request,
            outcome=candidate,
          )
          continue
        built = {**candidate, "attempt": attempt}
        identities[case] = _qualification_identity(built, frozen, toolchain)
        tensor_execution.write_atomic(
          work / case / "build.json",
          (json.dumps(built, indent=1, sort_keys=True) + "\n").encode(),
        )
        progress["cases"][case] = {
          "identity": identities[case],
          "outcomes": {},
          "attempts": {},
          "p0_attempted": False,
        }
        save_progress()
        break
    if built is None:
      if authority.is_paused():
        reason = json.loads((work / "study-paused.json").read_text())["reason"]
      else:
        reason = "qualification construction exhausted four attempts"
        _qualification_construction_stop(authority, case, reason)
      record["cases"].append({"case": case, "refused": reason})
      break
    required = next(
      entry["checks"] for entry in declaration["cases"] if entry["case"] == case
    )
    capabilities = qualification_capabilities(built, required)
    if not all(capabilities.get(name) for name in required):
      _qualification_construction_stop(
        authority,
        case,
        "the declared qualification capability is missing",
        capabilities=capabilities,
        identity=identities[case],
      )
      record["cases"].append(
        {
          "case": case,
          "capabilities": capabilities,
          "refused": "the declared qualification capability is missing",
        }
      )
      break
    state = progress["cases"][case]
    observed = references.get(case)
    for configuration in frozen.spec["configurations"]:
      if diagnostic_only and configuration != retry_request[1]:
        continue
      previous = state["attempts"].get(configuration)
      if previous is not None:
        if previous["status"] == "completed":
          continue
        if retry_request != (case, configuration, previous["attempt"]):
          continue
      if not authority._disk_guard():
        break
      run_index = 0 if previous is None else previous["attempt"] + 1
      directory = work / case / f"{configuration}-run-{run_index}"
      if directory.exists():
        raise StudyRefused("qualification attempt directory already exists")
      directory.mkdir(parents=True)
      phase = "qualification" if previous is None else "triage"
      accountant = authority.process_observer(
        phase,
        case=case,
        configuration=configuration,
        attempt=run_index,
        directory=str(directory),
      )
      events = []
      reference_destination = work / case / "p0.json"
      if recheck_request == case and previous is not None:
        observed = None
        reference_destination = (
          work
          / case
          / "reference-rechecks"
          / f"{configuration}-run-{run_index}"
          / "p0.json"
        )
      needs_reference = observed is None
      state["attempts"][configuration] = {
        "status": "running",
        "attempt": run_index,
        "directory": str(directory),
      }
      state["p0_attempted"] |= needs_reference
      save_progress()

      def account(event):
        accountant(event)
        events.append(event)

      def retain_reference(value):
        # An observation is saved immediately after P0, before P1/P2 can fail.
        nonlocal observed
        if not isinstance(value, ReferenceObservation):
          raise StudyRefused(
            "qualification executor did not return a typed P0 observation"
          )
        observed = value
        if not needs_reference:
          return
        meta = {**value.as_record(), "identity": identities[case]}
        meta["observation_id"] = _digest({**meta, "attempt": str(directory)})
        if reference_destination.is_file():
          if json.loads(reference_destination.read_text()) != meta:
            raise StudyRefused("refusing to overwrite an earlier P0 observation")
        else:
          if value.output is not None:
            tensor_execution.write_atomic(
              reference_destination.with_name("p0.bin"), value.output
            )
          tensor_execution.write_atomic(
            reference_destination,
            (json.dumps(meta, indent=1, sort_keys=True) + "\n").encode(),
          )
        state["reference_record"] = str(reference_destination)
        state["reference_observation_id"] = meta["observation_id"]
        state["reference_digest"] = _digest(meta)
        save_progress()

      try:
        outcome = operations["execute"](
          {
            "slot": f"{case}/i0-base/{configuration}",
            "case": case,
            "corpus": {"kind": "source", **built, "case": case},
            "input": "i0-base",
            "configuration": configuration,
            "attempt": run_index,
            "directory": str(directory),
            "reference": observed,
            "on_process": account,
            "on_reference": retain_reference,
            "reference_output_path": str(reference_destination.with_name("p0.bin")),
            "force_reference": needs_reference,
          }
        )
      except DispatchRefused as error:
        if needs_reference and observed is None:
          from test.lib.tensor_target_reference import (
            UNAVAILABLE,
            ReferenceObservation,
          )
          failed = ReferenceObservation(
            UNAVAILABLE, reason=f"P0 dispatch refused: {error}"
          )
          retain_reference(failed)
        outcome = {
          "status": "dispatch_refused",
          "dispatch_refused": str(error),
          "paths": {},
          "candidate": False,
          "compiler": {"status": "unavailable"},
        }
      except BaseException:
        state["attempts"][configuration]["status"] = "interrupted"
        save_progress()
        raise
      if isinstance(outcome.get("reference"), ReferenceObservation):
        retain_reference(outcome["reference"])
      paths = outcome.get("paths", {})
      complete = not outcome.get("dispatch_refused")
      entry = {
        "status": outcome.get("status"),
        "paths": paths,
        "candidate": outcome.get("candidate", False),
        "dispatch_refused": outcome.get("dispatch_refused"),
        "compiler": outcome.get("compiler"),
        "processes": events,
        "reference_status": getattr(observed, "status", "unavailable"),
      }
      attempt_record = {
        **state["attempts"][configuration],
        "status": "completed" if complete else "refused",
        "artifacts": _qualification_artifacts(directory),
      }
      _qualification_outcome(
        authority, case, configuration, run_index, entry, attempt_record
      )
      state["outcomes"][configuration] = entry
      state["attempts"][configuration] = attempt_record
      save_progress()
      if _pause_qualification_outcomes(authority):
        break
    configurations = state["outcomes"]
    complete = all(
      state["attempts"].get(c, {}).get("status") == "completed"
      for c in frozen.spec["configurations"]
    )
    open_candidates = [
      review["event"]["qualification_attempt"]
      for review in _qualification_reviews(work, authority.path_ledger.events, frozen)
      if review["unresolved"] and review["event"]["case"] == case
    ]
    validated = (
      complete
      and not open_candidates
      and getattr(observed, "status", None) == COMPLETED
      and all(
        not value.get("candidate")
        and all(
          value.get("paths", {}).get(p, {}).get("status") == "completed"
          for p in ("p1", "p2")
        )
        for value in configurations.values()
      )
    )
    capabilities = qualification_capabilities(built, required)
    validated = bool(
      validated and all(capabilities.get(name) for name in required)
    )
    case_record = {
      "case": case,
      "padded": slot["padded"],
      "configurations": configurations,
      "capabilities": capabilities,
      "complete": complete,
      "validated": validated,
      **({"unresolved_candidates": open_candidates} if open_candidates else {}),
    }
    if not complete:
      case_record["interrupted"] = (
        "an unfinished configuration needs an explicit attempt retry"
      )
    if getattr(observed, "status", None) != COMPLETED:
      case_record["p0_unavailable"] = getattr(observed, "reason", "no P0 observation")
    state["complete"], state["validated"] = complete, validated
    save_progress()
    record["cases"].append(case_record)
    if authority.is_paused() or authority.is_stopped():
      break
  record["validated"] = (
    slots == 4
    and len(record["cases"]) == 4
    and all(case.get("validated", False) for case in record["cases"])
  )
  record["digest"] = _digest(record)
  tensor_execution.write_atomic(
    work / "qualification.json",
    (json.dumps(record, indent=1, sort_keys=True) + "\n").encode(),
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
  parser.add_argument(
    "--qualification-retry",
    default=None,
    help="case/configuration/attempt-N of an unfinished attempt",
  )
  parser.add_argument(
    "--qualification-reference-recheck",
    default=None,
    help="case/observation-id; requires a matching explicit retry",
  )
  parser.add_argument("--accounting-mode", choices=("legacy", "process"), default=None)
  parser.add_argument("--accept-current-toolchain", action="store_true")
  parser.add_argument("--triage-decision", default=None)
  parser.add_argument("--triage-disposition-file", type=Path, default=None)
  arguments = parser.parse_args(argv)
  decision = arguments.triage_decision
  if arguments.triage_disposition_file is not None:
    if decision is None:
      parser.error("--triage-disposition-file requires --triage-decision")
    decision = {"action": decision, "disposition": json.loads(arguments.triage_disposition_file.read_text())}
  if arguments.mode == "reconcile":
    report = report_from_records(
      Path(arguments.work),
      triage_ledger=Path(arguments.triage_ledger) if arguments.triage_ledger else None,
      accounting_mode=arguments.accounting_mode,
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
  toolchain = operations["toolchain"]
  work = Path(arguments.work)
  study_path = work / "study.json"
  frozen = FrozenStudy.load(study_path) if study_path.is_file() else freeze_study(toolchain=toolchain)
  require_toolchain(
    frozen, toolchain, accept_current=arguments.accept_current_toolchain
  )
  if arguments.mode in ("select", "discover"):
    require_qualification(work, frozen)
  study = Study(
    work,
    frozen,
    controls_root=Path(arguments.controls_root) if arguments.controls_root else None,
    triage_ledger=Path(arguments.triage_ledger) if arguments.triage_ledger else None,
    accounting_mode=arguments.accounting_mode,
    **{key: value for key, value in operations.items() if key != "toolchain"},
  )
  if arguments.mode == "qualification":
    record = run_qualification(
      work,
      operations=operations,
      retry=arguments.qualification_retry,
      reference_recheck=arguments.qualification_reference_recheck,
      triage_decision=decision,
      authority=study,
    )
    print(
      json.dumps(
        {
          "cases": len(record["cases"]),
          "digest": record["digest"],
          "validated": record["validated"],
        }
      )
    )
    return 0 if record["validated"] else 2
  study.prepare()
  if study.is_paused():
    if arguments.triage_decision is None:
      raise StudyRefused(
        f"the study is paused ({study._paused_reason()}); pass --triage-decision"
      )
    study.record_triage_decision(decision)
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
