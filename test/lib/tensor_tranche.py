"""Frozen manifest and append-only ledger for one bounded tranche.

A tranche is decided before dispatch and then identified by a digest of that
decision. Slots are derived from the manifest in a fixed order, so resuming an
interrupted tranche re-derives the same slots instead of a similar-looking set.
Work is claimed before it starts and every claim and outcome is appended, which
makes the evidence a transcript rather than a summary rewritten at the end.

Three refusals are distinct on purpose:

* `ManifestRefused` - the manifest changed after the freeze, or is malformed;
* `SlotRefused` - a slot is terminal, or is already running;
* `BudgetExhausted` - the frozen execution budget is spent.

None of them deletes anything. A slot that was refused for domain reasons is
terminal, because the refusal is a result; a slot that never ran is recorded as
`unexecuted` when the tranche is closed, because an unspent slot is also a fact
worth reporting.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

TRANCHES = ("A", "B", "C", "FP", "companion")
STATES = (
  "pending",
  "running",
  "completed",
  "refused",
  "failed",
  "timed_out",
  "unexecuted",
)
TERMINAL = ("completed", "refused", "failed", "timed_out", "unexecuted")


class ManifestRefused(ValueError):
  """The tranche manifest is malformed or no longer matches its digest."""


class SlotRefused(ValueError):
  """The requested slot cannot be claimed."""


class BudgetExhausted(RuntimeError):
  """The frozen execution budget is spent."""


def _canonical(value):
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(payload):
  return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def read(path):
  """Read a frozen manifest file without verifying it."""
  try:
    return json.loads(Path(path).read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise ManifestRefused(f"cannot read the frozen manifest: {error}") from error


@dataclasses.dataclass(frozen=True)
class Slot:
  """One planned execution: a case, one input, one configuration."""

  slot_id: str
  case_id: str
  lane: str
  input: str
  configuration: str
  index: int


@dataclasses.dataclass(frozen=True)
class FrozenTranche:
  """A manifest and the slots it expands to, identified by one digest."""

  payload: dict
  digest: str
  slots: tuple[Slot, ...]

  def write(self, path):
    Path(path).write_text(json.dumps(self.as_record(), indent=1) + "\n", encoding="utf-8")
    return Path(path)

  def as_record(self):
    return {"digest": self.digest, "payload": self.payload}

  @classmethod
  def from_payload(cls, record):
    payload = record.get("payload") if isinstance(record, dict) else None
    if not isinstance(payload, dict) or "manifest" not in payload:
      raise ManifestRefused("a frozen tranche record must carry a manifest payload")
    if record.get("digest") != payload.get("digest"):
      raise ManifestRefused("the frozen digest does not match its payload")
    manifest = payload["manifest"]
    recomputed = _digest(manifest)
    if payload.get("digest") != recomputed:
      raise ManifestRefused("the manifest changed after it was frozen")
    return freeze(manifest)

  @classmethod
  def load(cls, path):
    return cls.from_payload(read(path))

  def slot(self, slot_id):
    for slot in self.slots:
      if slot.slot_id == slot_id:
        return slot
    raise SlotRefused(f"no slot {slot_id!r} in the frozen tranche")

  def budget(self, name):
    value = self.payload["manifest"]["budget"].get(name)
    return value


def freeze(manifest):
  """Expand a manifest into slots and freeze it."""
  if not isinstance(manifest, dict):
    raise ManifestRefused("a tranche manifest must be an object")
  cases = manifest.get("cases")
  if not isinstance(cases, list) or not cases:
    raise ManifestRefused("a tranche manifest must list at least one case")
  budget = manifest.get("budget")
  if not isinstance(budget, dict) or not isinstance(budget.get("executions"), int):
    raise ManifestRefused("a tranche manifest must state an integer execution budget")

  lanes = manifest.get("lanes", list(TRANCHES))
  if (
    not isinstance(lanes, list)
    or not lanes
    or any(not isinstance(lane, str) or not lane for lane in lanes)
  ):
    raise ManifestRefused("a tranche manifest must state a nonempty lane list")
  if len(set(lanes)) != len(lanes):
    raise ManifestRefused("a tranche manifest repeats a lane")

  slots = []
  seen = set()
  for case in cases:
    case_id = case.get("id")
    lane = case.get("lane")
    inputs = case.get("inputs")
    configurations = case.get("configurations")
    if not isinstance(case_id, str) or not case_id:
      raise ManifestRefused("every case needs a string id")
    if lane not in lanes:
      raise ManifestRefused(f"case {case_id!r} names an unknown lane {lane!r}")
    if not isinstance(inputs, list) or not inputs:
      raise ManifestRefused(f"case {case_id!r} lists no inputs")
    if not isinstance(configurations, list) or not configurations:
      raise ManifestRefused(f"case {case_id!r} lists no configurations")
    for name in inputs:
      for configuration in configurations:
        slot_id = f"{case_id}/{name}/{configuration}"
        if slot_id in seen:
          raise ManifestRefused(f"duplicate slot {slot_id!r}")
        seen.add(slot_id)
        slots.append(
          Slot(
            slot_id=slot_id,
            case_id=case_id,
            lane=lane,
            input=name,
            configuration=configuration,
            index=len(slots),
          )
        )
  payload = {"manifest": manifest, "digest": _digest(manifest)}
  return FrozenTranche(payload=payload, digest=payload["digest"], slots=tuple(slots))


@dataclasses.dataclass(frozen=True)
class Claim:
  """A slot handed out for execution."""

  slot_id: str
  index: int


class Ledger:
  """An append-only record of claims and outcomes for one frozen tranche."""

  def __init__(self, frozen, path=None):
    self.frozen = frozen
    self.path = None if path is None else Path(path)
    self.events = []
    self._states = {}
    self._claims = 0

  @property
  def digest(self):
    return self.frozen.digest

  @classmethod
  def in_memory(cls, frozen):
    return cls(frozen)

  @classmethod
  def open(cls, path, frozen):
    """Open a ledger, replaying what is already on disk."""
    ledger = cls(frozen, path=path)
    path = Path(path)
    if path.is_file():
      for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
          continue
        try:
          event = json.loads(line)
        except json.JSONDecodeError as error:
          raise ManifestRefused(f"a ledger line is not JSON: {error}") from error
        if event.get("kind") == "header":
          if event.get("tranche_digest") != frozen.digest:
            raise ManifestRefused("the ledger belongs to another frozen tranche")
          continue
        ledger._replay(event)
    else:
      ledger._append({"kind": "header", "tranche_digest": frozen.digest})
    return ledger

  def _append(self, event):
    self.events.append(event)
    if self.path is not None:
      self.path.parent.mkdir(parents=True, exist_ok=True)
      with self.path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")

  def _replay(self, event):
    self.events.append(event)
    slot_id = event.get("slot")
    if event.get("kind") == "claim":
      self._claims += 1
      self._states[slot_id] = "running"
    elif event.get("kind") == "finish":
      self._states[slot_id] = event.get("state", "failed")

  def state(self, slot_id):
    return self._states.get(slot_id, "pending")

  def counts(self):
    counts = {state: 0 for state in STATES}
    for slot in self.frozen.slots:
      counts[self.state(slot.slot_id)] += 1
    return counts

  def completed_ids(self):
    return [
      slot.slot_id
      for slot in self.frozen.slots
      if self.state(slot.slot_id) == "completed"
    ]

  def claim(self, slot_id):
    """Claim one slot, recording the claim before any work starts."""
    slot = self.frozen.slot(slot_id)
    state = self.state(slot_id)
    if state in TERMINAL:
      raise SlotRefused(f"slot {slot_id!r} is already {state}")
    if state == "running":
      raise SlotRefused(f"slot {slot_id!r} is already running")
    budget = self.frozen.budget("executions")
    if budget is not None and self._claims >= budget:
      raise BudgetExhausted(
        f"the frozen execution budget of {budget} is spent; {slot_id!r} stays unexecuted"
      )
    self._claims += 1
    self._append({"kind": "claim", "slot": slot_id, "index": slot.index})
    self._states[slot_id] = "running"
    return Claim(slot_id=slot_id, index=slot.index)

  def claim_next(self):
    """Claim the first slot that is still pending, in frozen order."""
    for slot in self.frozen.slots:
      if self.state(slot.slot_id) == "pending":
        return self.claim(slot.slot_id)
    raise SlotRefused("every slot in the frozen tranche is terminal")

  def finish(self, slot_id, state, *, reason=None, detail=None):
    """Record a terminal state for a claimed slot."""
    if state not in TERMINAL:
      raise ValueError(f"finish state must be terminal, not {state!r}")
    if self.state(slot_id) != "running":
      raise SlotRefused(f"slot {slot_id!r} is not running")
    event = {"kind": "finish", "slot": slot_id, "state": state}
    if reason is not None:
      event["reason"] = reason
    if detail is not None:
      event["detail"] = detail
    self._append(event)
    self._states[slot_id] = state
    return state

  def close(self):
    """Record every slot that never ran as unexecuted, then return the counts."""
    for slot in self.frozen.slots:
      if self.state(slot.slot_id) == "pending":
        self._append(
          {"kind": "finish", "slot": slot.slot_id, "state": "unexecuted", "reason": "not dispatched"}
        )
        self._states[slot.slot_id] = "unexecuted"
    return self.counts()
