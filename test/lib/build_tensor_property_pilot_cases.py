"""Export balanced, independently revalidated property-pilot witnesses."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import subprocess
from collections import defaultdict
from pathlib import Path

from test.lib import tensor_case, tensor_property_inputs
from test.lib.tensor_case_manifest import record_buffer, write_v1_manifest
from test.lib.tensor_property_inputs import validate_witness
from test.lib.tensor_property_workload import (
  PilotProblem,
  load_workload,
  render_runtime,
)

METHODS = ("ordinary", "boundary", "construction", "smt")
FAMILIES = (
  "independent_sum",
  "affine_sum",
  "layout_sum",
  "shared_reductions",
  "nonlinear",
)
GENERATOR_REVISION = "property-selection-pilot-v1"


def _sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _problem_index(problems: list[PilotProblem]) -> dict[str, PilotProblem]:
  return {problem.problem_id: problem for problem in problems}


def select_pilot_witnesses(
  attempt_rows: list[dict[str, object]],
  problems: list[PilotProblem],
  *,
  limit_per_method: int = 32,
  seed: int = 281,
) -> dict[str, list[dict[str, object]]]:
  """Choose unique validated problem witnesses with family/size round-robin."""
  if type(limit_per_method) is not int or limit_per_method < 0:
    raise ValueError("limit_per_method must be a nonnegative integer")
  if type(seed) is not int:
    raise ValueError("seed must be an integer")
  problem_by_id = _problem_index(problems)
  candidates: dict[str, dict[tuple[str, int], list[dict[str, object]]]] = {
    method: defaultdict(list) for method in METHODS
  }
  for row in attempt_rows:
    method = row.get("method")
    if method not in METHODS or row.get("status") != "validated":
      continue
    problem_id = row.get("problem_id")
    if not isinstance(problem_id, str) or problem_id not in problem_by_id:
      raise ValueError(f"validated attempt references unknown problem {problem_id!r}")
    problem = problem_by_id[problem_id]
    if row.get("family") not in (None, problem.family) or row.get(
      "symbol_count"
    ) not in (None, problem.symbol_count):
      raise ValueError(f"attempt metadata disagrees with problem {problem_id}")
    if not isinstance(row.get("input_hex"), str) or not isinstance(
      row.get("output_hex"), str
    ):
      raise ValueError(f"validated attempt {problem_id}/{method} lacks witness bytes")
    cell = (problem.family, problem.symbol_count)
    candidates[str(method)][cell].append(row)

  selected: dict[str, list[dict[str, object]]] = {}
  for method_index, method in enumerate(METHODS):
    rng = random.Random(seed ^ (method_index * 0x9E3779B1))
    by_cell = candidates[method]
    unique_cells: dict[tuple[str, int], list[dict[str, object]]] = {}
    for cell, rows in by_cell.items():
      by_problem: dict[str, list[dict[str, object]]] = defaultdict(list)
      for row in rows:
        by_problem[str(row["problem_id"])].append(row)
      problems_in_cell = list(by_problem.values())
      rng.shuffle(problems_in_cell)
      unique_cells[cell] = []
      for repeated_trials in problems_in_cell:
        unique_cells[cell].append(rng.choice(repeated_trials))

    family_order = [
      family for family in FAMILIES if any(p.family == family for p in problems)
    ]
    symbol_order = sorted({problem.symbol_count for problem in problems})
    cell_order = [
      (family, symbol_count) for family in family_order for symbol_count in symbol_order
    ]
    chosen = []
    while len(chosen) < limit_per_method:
      made_progress = False
      for cell in cell_order:
        rows = unique_cells.get(cell, [])
        if not rows:
          continue
        chosen.append(rows.pop())
        made_progress = True
        if len(chosen) == limit_per_method:
          break
      if not made_progress:
        break
    selected[method] = chosen
  return selected


def _read_attempts(path: Path) -> list[dict[str, object]]:
  rows = []
  with path.open(encoding="utf-8") as source:
    for line_number, line in enumerate(source, start=1):
      if not line.strip():
        raise ValueError(f"attempt ledger has a blank line at {line_number}")
      row = json.loads(line)
      if not isinstance(row, dict):
        raise ValueError(f"attempt ledger line {line_number} is not an object")
      rows.append(row)
  return rows


def _describe(source_path: Path, *, symiri: str | Path) -> dict[str, object]:
  command = [
    str(symiri),
    "--entry",
    "kernel",
    "--tensor-describe",
    str(source_path),
  ]
  try:
    result = subprocess.run(
      command, capture_output=True, text=True, timeout=10, check=False
    )
  except (OSError, subprocess.TimeoutExpired) as error:
    raise RuntimeError(f"cannot describe exported kernel signature: {error}") from error
  if result.returncode != 0:
    raise RuntimeError(f"kernel signature check failed: {result.stderr.strip()}")
  try:
    signature = json.loads(result.stdout)
  except json.JSONDecodeError as error:
    raise RuntimeError("symiri returned invalid tensor signature JSON") from error
  if signature.get("entry") != "kernel":
    raise ValueError("exported kernel signature has the wrong entry")
  return signature


def _input_records(signature: dict[str, object], problem: PilotProblem) -> list[dict]:
  described = signature.get("inputs")
  if not isinstance(described, list) or len(described) != len(problem.inputs):
    raise ValueError("described input count differs from the frozen problem")
  records = []
  offset = 0
  for description, input_tensor in zip(described, problem.inputs, strict=True):
    if (
      description.get("name") != input_tensor.name
      or description.get("dtype") != "i32"
      or description.get("shape") != list(input_tensor.shape)
    ):
      raise ValueError(f"input signature differs for {input_tensor.name}")
    length = int(description["byte_length"])
    element_count = 1
    for dimension in input_tensor.shape:
      element_count *= dimension
    if length != element_count * 4:
      raise ValueError("input byte length differs from the frozen input shape")
    record = record_buffer(
      input_tensor.name, "i32", list(input_tensor.shape), "inputs.bin", length
    )
    record["byte_offset"] = offset
    records.append(record)
    offset += length
  return records


def _case_groups(
  selected: dict[str, list[dict[str, object]]],
  problem_by_id: dict[str, PilotProblem],
) -> list[dict[str, object]]:
  groups: dict[tuple[bytes, bytes, tuple[int, ...]], dict[str, object]] = {}
  for method in METHODS:
    for row in selected[method]:
      problem = problem_by_id[str(row["problem_id"])]
      source = render_runtime(problem).encode("utf-8")
      try:
        input_bytes = bytes.fromhex(str(row["input_hex"]))
        output_bytes = bytes.fromhex(str(row["output_hex"]))
      except ValueError as error:
        raise ValueError(
          "selected witness contains invalid hexadecimal bytes"
        ) from error
      key = (source, input_bytes, (4,))
      group = groups.setdefault(
        key,
        {
          "problem": problem,
          "source": source,
          "input_bytes": input_bytes,
          "target_bytes": output_bytes,
          "associations": [],
        },
      )
      if group["target_bytes"] != output_bytes:
        raise ValueError("duplicate source/input selected with conflicting outputs")
      group["associations"].append(
        {
          "method": method,
          "problem_id": problem.problem_id,
          "trial": row.get("trial"),
          "schedule_index": row.get("schedule_index"),
          "seed": row.get("seed"),
          "input_sha256": _sha256(input_bytes),
        }
      )
  return list(groups.values())


def build_cases(
  workload_path: str | Path,
  attempts_path: str | Path,
  output_dir: str | Path,
  *,
  seed: int,
  limit_per_method: int = 32,
  symiri: str | Path | None = None,
) -> dict[str, object]:
  """Select, replay, and export a balanced corpus of runtime-input cases."""
  workload_path = Path(workload_path).resolve()
  attempts_path = Path(attempts_path).resolve()
  output_dir = Path(output_dir).resolve()
  if output_dir.exists() and any(output_dir.iterdir()):
    raise ValueError(f"case output directory is not empty: {output_dir}")
  output_dir.mkdir(parents=True, exist_ok=True)
  _, problems = load_workload(workload_path)
  problem_by_id = _problem_index(problems)
  selected = select_pilot_witnesses(
    _read_attempts(attempts_path),
    problems,
    limit_per_method=limit_per_method,
    seed=seed,
  )
  symiri = Path(symiri or tensor_property_inputs.SYMIRI).resolve()
  all_cases = []
  validation_root = output_dir / "validation"
  for index, group in enumerate(_case_groups(selected, problem_by_id)):
    problem = group["problem"]
    input_bytes = group["input_bytes"]
    target_bytes = struct.pack(f"<{len(problem.target)}i", *problem.target)
    if group["target_bytes"] != target_bytes:
      raise ValueError(
        f"attempt output bytes disagree with target for {problem.problem_id}"
      )
    validation = validate_witness(
      problem,
      input_bytes,
      artifact_dir=validation_root / f"case-{index:04d}",
    )
    if validation.status != "validated":
      raise ValueError(
        f"witness validation failed for {problem.problem_id}: "
        f"{validation.status} ({validation.reason})"
      )
    if validation.output_bytes != target_bytes:
      raise ValueError(f"independent reference output differs for {problem.problem_id}")
    if validation.output_bytes != group["target_bytes"]:
      raise ValueError(
        f"attempt output differs from independent replay for {problem.problem_id}"
      )

    case_name = f"case_{index:04d}"
    case_dir = output_dir / case_name
    case_dir.mkdir()
    source = group["source"]
    signature_path = case_dir / "program.sir"
    signature_path.write_bytes(source)
    signature = _describe(signature_path, symiri=symiri)
    input_records = _input_records(signature, problem)
    input_size = sum(record["byte_length"] for record in input_records)
    if input_size != len(input_bytes):
      raise ValueError(
        f"input bytes do not match the described signature for {problem.problem_id}"
      )
    expected_size = int(signature["output"]["byte_length"])
    if expected_size != len(target_bytes):
      raise ValueError(
        f"target bytes do not match the output signature for {problem.problem_id}"
      )
    (case_dir / "inputs.bin").write_bytes(input_bytes)
    (case_dir / "expected.bin").write_bytes(validation.output_bytes)
    methods = sorted({str(item["method"]) for item in group["associations"]})
    evidence_kind = "exact_smt" if "smt" in methods else "construction"
    evidence_status = (
      "exact_smt_checked" if evidence_kind == "exact_smt" else "construction_checked"
    )
    write_v1_manifest(
      case_dir,
      source,
      input_bytes,
      validation.output_bytes,
      signature,
      input_records,
      case_name=case_name,
      property_id=f"pilot.{problem.problem_id}",
      property_source=(
        f"prescribed target {list(problem.target)}; selected by {', '.join(methods)}"
      ),
      evidence_kind=evidence_kind,
      evidence_status=evidence_status,
      generator_revision=GENERATOR_REVISION,
      seed=next(
        (
          int(item["seed"])
          for item in group["associations"]
          if item["seed"] is not None
        ),
        seed,
      ),
    )
    tensor_case.load_case(case_dir, symiri=symiri)
    all_cases.append(
      {
        "case": case_name,
        "problem_id": problem.problem_id,
        "graph_id": problem.graph_id,
        "family": problem.family,
        "symbol_count": problem.symbol_count,
        "target_pattern": problem.target_pattern,
        "target": list(problem.target),
        "methods": methods,
        "associations": group["associations"],
        "input_sha256": _sha256(input_bytes),
        "expected_sha256": _sha256(validation.output_bytes),
      }
    )

  try:
    workload_bytes = workload_path.read_bytes()
    attempts_bytes = attempts_path.read_bytes()
  except OSError as error:
    raise ValueError(f"cannot hash pilot inputs: {error}") from error
  sidecar = {
    "schema": "refractir.tensor-property-case-corpus.v1",
    "generator_revision": GENERATOR_REVISION,
    "selection_seed": seed,
    "per_method_limit": limit_per_method,
    "workload_sha256": _sha256(workload_bytes),
    "attempts_sha256": _sha256(attempts_bytes),
    "selected_by_method": {method: len(selected[method]) for method in METHODS},
    "case_count": len(all_cases),
    "cases": all_cases,
  }
  (output_dir / "selection.json").write_text(
    json.dumps(sidecar, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
  )
  return sidecar


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--workload", required=True)
  parser.add_argument("--attempts", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--seed", type=int, default=281)
  parser.add_argument("--limit-per-method", type=int, default=32)
  parser.add_argument("--symiri", default=None)
  args = parser.parse_args(argv)
  summary = build_cases(
    args.workload,
    args.attempts,
    args.output,
    seed=args.seed,
    limit_per_method=args.limit_per_method,
    symiri=args.symiri,
  )
  print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
