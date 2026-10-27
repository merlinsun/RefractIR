"""Classify the retained mismatch bits of a sweep's finite leads.

A host-only first triage step: no worker starts and no GPU is used. The
finite leads are the path differences the multi-shard summary keeps after
discarding those in non-finite output. For each lead, the paired-run records
kept in its quarantined directories (`bundle/target-runs/target-run.json`)
list up to the first 16 differing output words as raw bits, and each
difference is put in one class:

* `signed_zero`: +0.0 against -0.0
* `sign_flip`: the same magnitude with the opposite sign
* `nan`: either word is a NaN
* `rounding`: the same sign, at most 4 units in the last place apart
* `value`: any other difference
* `length`: the outputs differ in length

A lead's dominant class is the most severe class among its differences, so a
single value difference is not hidden by any number of signed zeros.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

_SUMMARY = Path(__file__).with_name("summarize_moving_window_sweep.py")
_spec = importlib.util.spec_from_file_location("sweep_summary", _SUMMARY)
summary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summary)

SCHEMA = "refractir.sweep-mismatch-classes.v1"
# The paired runner reports at most this many differing words per run.
REPORTED_LIMIT = 16
SEVERITY = ("value", "sign_flip", "length", "nan", "rounding", "signed_zero")
RECORD = Path("bundle") / "target-runs" / "target-run.json"
ROUNDING_ULPS = 4
SIGN = 0x80000000


def _word(bits):
  return int(bits, 16)


def _float(word):
  return struct.unpack("<f", word.to_bytes(4, "little"))[0]


def classify(interpreter_bits, compiled_bits):
  """The class of one differing output word."""
  if interpreter_bits.startswith("length=") or compiled_bits.startswith("length="):
    return "length"
  first, second = _word(interpreter_bits), _word(compiled_bits)
  if {first, second} == {0, SIGN}:
    return "signed_zero"
  if first ^ second == SIGN:
    return "sign_flip"
  if math.isnan(_float(first)) or math.isnan(_float(second)):
    return "nan"
  if first & SIGN == second & SIGN and abs(first - second) <= ROUNDING_ULPS:
    return "rounding"
  return "value"


def _example(mismatch):
  example = {
    "index": mismatch["index"],
    "interpreter_bits": mismatch["interpreter_bits"],
    "compiled_bits": mismatch["compiled_bits"],
  }
  if not mismatch["interpreter_bits"].startswith("length="):
    example["interpreter"] = _float(_word(mismatch["interpreter_bits"]))
    example["compiled"] = _float(_word(mismatch["compiled_bits"]))
  return example


def _candidates(path):
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _lead(lead, candidates):
  """Classes, examples, and record gaps for one finite lead."""
  classes, examples = {}, []
  missing, truncated = 0, False
  for candidate in candidates:
    path = Path(candidate["directory"]) / RECORD
    if not path.is_file():
      missing += 1
      continue
    mismatches = json.loads(path.read_text())["agreement"]["mismatches"]
    truncated = truncated or len(mismatches) >= REPORTED_LIMIT
    for mismatch in mismatches:
      kind = classify(mismatch["interpreter_bits"], mismatch["compiled_bits"])
      classes[kind] = classes.get(kind, 0) + 1
      if len(examples) < 3:
        examples.append({"class": kind, **_example(mismatch)})
  present = [kind for kind in SEVERITY if kind in classes]
  return {
    "shard": lead["shard"],
    "seed": lead["seed"],
    "runs": len(candidates),
    "classes": classes,
    "dominant": present[0] if present else "record_missing",
    "missing_records": missing,
    "truncated": truncated,
    "examples": examples,
  }


def classify_leads(shards):
  """Classify every finite lead the summary finds in these shards."""
  leads = summary.summarize(shards)["finite_path_differences"]
  results = []
  for lead in leads:
    candidates = [
      candidate
      for candidate in _candidates(Path(lead["shard"]) / "candidates.jsonl")
      if candidate["seed"] == lead["seed"] and candidate["kind"] == "path_difference"
    ]
    results.append(_lead(lead, candidates))
  dominant = {}
  for result in results:
    dominant[result["dominant"]] = dominant.get(result["dominant"], 0) + 1
  return {"schema": SCHEMA, "leads": results, "by_dominant_class": dominant}


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("shards", type=Path, nargs="+")
  parser.add_argument("--out", type=Path, required=True)
  arguments = parser.parse_args(argv)
  result = classify_leads(arguments.shards)
  arguments.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
  print(json.dumps(result["by_dominant_class"], indent=1, sort_keys=True))
  for lead in result["leads"]:
    print(lead["seed"], lead["dominant"], lead["classes"])
  return 0


if __name__ == "__main__":
  sys.exit(main())
