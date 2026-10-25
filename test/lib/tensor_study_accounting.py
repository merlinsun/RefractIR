"""Pure accounting rules shared by live dispatch and report reconstruction.

Process reservations hold capacity even if the parent is interrupted. Only a
confirmed pre-spawn failure releases a hold. A start, including an interrupted
start, consumes its path's allocation. This module does not launch or schedule
work; the study's append-only ledger remains the authority.
"""

PATHS = ("p0", "p1", "p2")
PHASE_KEYS = {
  "qualification": (
    "qualification_p0",
    "qualification_interpreter",
    "qualification_gpu",
  ),
  "screening": ("selection_p0", "selection_interpreter", None),
  "discovery": ("discovery_p0", "discovery_interpreter", "discovery_gpu"),
  "triage": ("triage_p0", "triage_interpreter", "triage_gpu"),
}


def resolve_mode(spec, events, *, requested=None, adoption=None, digest=None):
  if requested not in (None, "legacy", "process"):
    raise ValueError("unknown accounting mode")
  declared = spec.get("accounting")
  if declared is not None and declared != {"version": 2, "mode": "process-events"}:
    raise ValueError("unsupported accounting declaration")
  adopted = None
  if adoption is not None:
    if adoption.get("study_digest") != digest or adoption.get("mode") not in (
      "process",
      "legacy",
    ):
      raise ValueError("accounting adoption does not match this study")
    adopted = adoption["mode"]
  mode = "process" if declared is not None else adopted
  if mode is not None and requested is not None and requested != mode:
    raise ValueError("accounting mode conflicts with the frozen or adopted mode")
  if mode is None:
    mode = requested
  if mode is None:
    if any(e.get("kind") == "process" for e in events):
      raise ValueError("unversioned process evidence needs an explicit accounting mode")
    mode = "legacy"
  if mode == "legacy" and any(e.get("kind") == "process" for e in events):
    raise ValueError("process evidence cannot be discarded by a legacy interpretation")
  return mode


def budget_key(phase, path):
  if phase not in PHASE_KEYS or path not in PATHS:
    raise ValueError("unknown process phase or path")
  return PHASE_KEYS[phase][PATHS.index(path)]


def _execution_records(events, imports):
  """Merge explicit identities before counting any path or allocation."""
  local = {}
  for event in events:
    if event.get("kind") == "process" and event.get("execution_id") is not None:
      local.setdefault(event["execution_id"], []).append(event)
  imported = {}
  for record in imports:
    if "entries" not in record:
      raise ValueError("process accounting requires normalized import entries")
    for entry in record["entries"]:
      key = entry["execution_id"]
      if key in imported and imported[key] != entry:
        raise ValueError("conflicting imported execution identity")
      imported[key] = entry

  records = []
  for key, history in local.items():
    first = history[0]
    phase, path = first["phase"], first["path"]
    if any(e["phase"] != phase or e["path"] != path for e in history):
      raise ValueError("conflicting local execution identity")
    started = any(e.get("event") == "start" for e in history)
    reserved = any(e.get("event") == "reserved" for e in history)
    finishes = {e.get("status") for e in history if e.get("event") == "finish"}
    charged = started or (reserved and "spawn_failed" not in finishes)
    entry = imported.pop(key, None)
    if entry is not None:
      allocation = "selection" if phase == "screening" else phase
      if entry["allocation"] != allocation or entry["path"] != path:
        raise ValueError("import path/allocation conflicts with local execution")
      for field in ("case", "attempt"):
        if first.get(field) is not None and entry.get(field) != first[field]:
          raise ValueError("import provenance conflicts with local execution")
      result = "timed_out" if entry["result"] == "hung" else entry["result"]
      terminal = finishes - {"interrupted"}
      if terminal and terminal != {result}:
        raise ValueError("import outcome conflicts with local execution")
      if started and result in ("spawn_failed", "refused"):
        raise ValueError("an imported refusal cannot erase a recorded launch")
      charged = started or result not in ("spawn_failed", "refused")
    records.append((phase, path, int(charged)))
  for entry in imported.values():
    phase = "screening" if entry["allocation"] == "selection" else entry["allocation"]
    records.append(
      (phase, entry["path"], int(entry["result"] not in ("spawn_failed", "refused")))
    )
  return records


def used(events, phase, path, imports=()):
  """Count each held/started/imported execution once in its own allocation."""
  total = sum(
    charge
    for p, target, charge in _execution_records(events, imports)
    if p == phase and target == path
  )
  # Unidentified historical events cannot be deduplicated against an import.
  # Retain their conservative charge; new records always carry identities.
  old = [
    e
    for e in events
    if e.get("kind") == "process"
    and e.get("execution_id") is None
    and e.get("phase") == phase
    and e.get("path") == path
  ]
  reserved = sum(e.get("event") == "reserved" for e in old)
  started = sum(e.get("event") == "start" for e in old)
  failed = sum(
    e.get("event") == "finish" and e.get("status") == "spawn_failed" for e in old
  )
  total += max(started, reserved - failed)
  if phase == "triage":
    total += sum(
      e.get("kind") == "retry" and not e.get("process_accounting") for e in events
    )
  return total
