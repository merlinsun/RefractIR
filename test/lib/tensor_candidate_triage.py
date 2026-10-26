"""Keep corpus decisions separate from evidence-backed candidate dispositions."""

import hashlib
import json
from pathlib import Path

POLICY = {"version": 1, "require_disposition": True}
CLASSIFICATIONS = {
  "unresolved",
  "invalid_execution",
  "infrastructure_defect",
  "permitted_difference",
  "known_bug",
  "new_bug",
}


def validate(work, disposition):
  if not isinstance(disposition, dict):
    raise ValueError("a candidate decision requires a disposition")
  classification = disposition.get("classification")
  if classification not in CLASSIFICATIONS:
    raise ValueError("unknown candidate disposition")
  explanation = disposition.get("explanation")
  if not isinstance(explanation, str) or not explanation.strip():
    raise ValueError("a candidate disposition requires an explanation")
  evidence = disposition.get("evidence")
  if not isinstance(evidence, list) or (
    classification != "unresolved" and not evidence
  ):
    raise ValueError("a resolved candidate disposition requires evidence")
  root = Path(work).resolve()
  for item in evidence:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
      raise ValueError("malformed triage evidence")
    relative = Path(item["path"])
    if relative.is_absolute() or ".." in relative.parts:
      raise ValueError("triage evidence must be relative to the study")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
      raise ValueError("triage evidence is missing or outside the study")
    if hashlib.sha256(path.read_bytes()).hexdigest() != item.get("sha256"):
      raise ValueError("triage evidence digest changed")
  # A validated record establishes provenance, not the truth of its diagnosis.
  return {
    "classification": classification,
    "explanation": explanation,
    "evidence": evidence,
  }


def summary(work, events):
  pauses = {
    event["pause_id"]: event
    for event in events
    if event.get("kind") == "paused" and event.get("candidate_sha256")
  }
  decisions = {}
  # The decision journal is authoritative even if interruption precedes the
  # derived path-ledger event.
  journal = Path(work) / "triage-decisions.jsonl"
  saved = (
    [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    if journal.is_file()
    else []
  )
  for decision in saved:
    if decision.get("pause_id") in pauses and decision.get("disposition"):
      if (
        decision.get("candidate_sha256")
        != pauses[decision["pause_id"]]["candidate_sha256"]
      ):
        raise ValueError("triage decision names different candidate evidence")
      decisions[decision["pause_id"]] = validate(work, decision["disposition"])
  # Qualification may pause the same retained outcome again after a diagnostic
  # retry. Its latest disposition replaces the open status, not its history.
  latest_qualification = {
    pause["candidate_sha256"]: pause_id
    for pause_id, pause in pauses.items()
    if pause.get("phase") == "qualification"
  }
  records = []
  for pause_id, pause in pauses.items():
    if (
      pause.get("phase") == "qualification"
      and latest_qualification[pause["candidate_sha256"]] != pause_id
    ):
      continue
    disposition = decisions.get(pause_id)
    records.append(
      {
        "pause_id": pause_id,
        "phase": pause.get("phase"),
        "candidate_sha256": pause["candidate_sha256"],
        "disposition": disposition,
        "status": "unresolved"
        if disposition is None
        else disposition["classification"],
      }
    )
  unresolved = sum(record["status"] == "unresolved" for record in records)
  return {
    "records": records,
    "unresolved": unresolved,
    "resolved": len(records) - unresolved,
  }
