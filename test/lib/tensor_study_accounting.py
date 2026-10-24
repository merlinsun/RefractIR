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


def used(events, phase, path, imports=()):
  """Count held/started processes once, plus imported work in this allocation."""
  identities = {}
  old = []
  for event in events:
    if (
      event.get("kind") != "process"
      or event.get("phase") != phase
      or event.get("path") != path
    ):
      continue
    identity = event.get("execution_id")
    if identity is None:
      old.append(event)
    else:
      identities.setdefault(identity, []).append(event)
  total = 0
  for history in identities.values():
    started = any(e.get("event") == "start" for e in history)
    reserved = any(e.get("event") == "reserved" for e in history)
    spawn_failed = any(
      e.get("event") == "finish" and e.get("status") == "spawn_failed" for e in history
    )
    total += int(started or (reserved and not spawn_failed))
  # Earlier process records have no execution IDs. Retain their conservative
  # charge; new records never use this adapter.
  reserved = sum(e.get("event") == "reserved" for e in old)
  started = sum(e.get("event") == "start" for e in old)
  failed = sum(
    e.get("event") == "finish" and e.get("status") == "spawn_failed" for e in old
  )
  total += max(started, reserved - failed)
  allocation = "selection" if phase == "screening" else phase
  seen = {}
  for record in imports:
    if "entries" not in record:
      raise ValueError("process accounting requires normalized import entries")
    for entry in record["entries"]:
      identity = entry["execution_id"]
      if identity in seen and seen[identity] != entry:
        raise ValueError("conflicting imported execution identity")
      if identity in seen:
        continue
      seen[identity] = entry
      if entry["path"] == path and entry["allocation"] == allocation:
        if entry["result"] not in ("spawn_failed", "refused"):
          total += 1
  if phase == "triage":
    # Old retry records reserved all paths without per-process callbacks.
    total += sum(
      e.get("kind") == "retry" and not e.get("process_accounting") for e in events
    )
  return total
