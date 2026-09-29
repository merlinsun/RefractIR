"""Bounded SMT input selection and independent replay for the property pilot."""

from __future__ import annotations

import itertools
import json
import math
import random
import re
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from test.lib.tensor_property_workload import (
  PilotProblem,
  _shape_text,
  render_query,
  render_reference,
  validate_problem,
)
from test.unit.tensor_support import SYMIR_SOLVE, SYMIRI

_PROCESS_LIMIT_S = 20.0
_CLEANUP_GRACE_S = 1.0


@dataclass(frozen=True)
class Attempt:
  status: str
  input_bytes: bytes | None = None
  output_bytes: bytes | None = None
  elapsed_s: float = 0.0
  stage_timings_s: tuple[tuple[str, float], ...] = ()
  attempt_count: int = 1
  artifact_paths: tuple[tuple[str, str], ...] = ()
  reason: str | None = None
  supported: bool | None = None
  witness_validated: bool = False


def _attempt_directory(
  problem: PilotProblem, artifact_dir: str | Path, child: str
) -> Path:
  directory = Path(artifact_dir) / problem.problem_id / child
  directory.mkdir(parents=True, exist_ok=True)
  return directory


def _monotonic() -> float:
  return time.monotonic()


def _time_left(deadline: float | None, fallback: float = _PROCESS_LIMIT_S) -> float:
  return fallback if deadline is None else max(0.0, deadline - _monotonic())


def _process_timeout(deadline: float | None) -> float:
  if deadline is None:
    return _PROCESS_LIMIT_S
  return min(_PROCESS_LIMIT_S, _time_left(deadline) + _CLEANUP_GRACE_S)


def _format_record(attempt: Attempt) -> dict[str, object]:
  return {
    "status": attempt.status,
    "reason": attempt.reason,
    "input_hex": attempt.input_bytes.hex() if attempt.input_bytes is not None else None,
    "output_hex": attempt.output_bytes.hex()
    if attempt.output_bytes is not None
    else None,
    "elapsed_s": attempt.elapsed_s,
    "stage_timings_s": dict(attempt.stage_timings_s),
    "attempt_count": attempt.attempt_count,
    "artifact_paths": dict(attempt.artifact_paths),
    "supported": attempt.supported,
    "witness_validated": attempt.witness_validated,
  }


def _finish(
  status: str,
  started: float,
  *,
  attempt_dir: Path,
  input_bytes: bytes | None = None,
  output_bytes: bytes | None = None,
  timings: dict[str, float] | None = None,
  paths: dict[str, str] | None = None,
  reason: str | None = None,
  attempt_count: int = 1,
  supported: bool | None = None,
  witness_validated: bool = False,
) -> Attempt:
  elapsed = max(0.0, _monotonic() - started)
  artifact_paths = dict(paths or {})
  record_path = attempt_dir / "attempt.json"
  attempt = Attempt(
    status=status,
    input_bytes=input_bytes,
    output_bytes=output_bytes,
    elapsed_s=elapsed,
    stage_timings_s=tuple(sorted((timings or {}).items())),
    attempt_count=attempt_count,
    artifact_paths=tuple(sorted(artifact_paths.items())),
    reason=reason,
    supported=supported,
    witness_validated=witness_validated or status == "validated",
  )
  record_path.write_text(
    json.dumps(_format_record(attempt), indent=2) + "\n", encoding="utf-8"
  )
  artifact_paths["attempt_json"] = str(record_path)
  return Attempt(
    status=attempt.status,
    input_bytes=attempt.input_bytes,
    output_bytes=attempt.output_bytes,
    elapsed_s=attempt.elapsed_s,
    stage_timings_s=attempt.stage_timings_s,
    attempt_count=attempt.attempt_count,
    artifact_paths=tuple(sorted(artifact_paths.items())),
    reason=attempt.reason,
    supported=attempt.supported,
    witness_validated=attempt.witness_validated,
  )


def _input_leaf_count(problem: PilotProblem) -> int:
  return sum(len(_flatten_shape(item.shape)) for item in problem.inputs)


def _flatten_shape(shape: tuple[int, ...]) -> tuple[None, ...]:
  count = 1
  for dimension in shape:
    count *= dimension
  return (None,) * count


def _decode_input(
  problem: PilotProblem, input_bytes: object
) -> tuple[dict[str, tuple[int, ...]] | None, str | None]:
  if not isinstance(input_bytes, bytes):
    return None, "input_type"
  element_count = _input_leaf_count(problem)
  expected_length = element_count * 4
  if len(input_bytes) != expected_length:
    return None, "input_length"
  values = struct.unpack(f"<{element_count}i", input_bytes)
  lower, upper = problem.domain
  if any(value < lower or value > upper for value in values):
    return None, "input_domain"
  decoded: dict[str, tuple[int, ...]] = {}
  offset = 0
  for item in problem.inputs:
    count = 1
    for dimension in item.shape:
      count *= dimension
    decoded[item.name] = tuple(values[offset : offset + count])
    offset += count
  return decoded, None


def _flatten_model(value: object) -> list[int] | None:
  if isinstance(value, bool):
    return None
  if isinstance(value, int):
    return [value]
  if not isinstance(value, list):
    return None
  result: list[int] = []
  for item in value:
    flattened = _flatten_model(item)
    if flattened is None:
      return None
    result.extend(flattened)
  return result


def _decode_model(
  problem: PilotProblem, model_path: Path
) -> tuple[bytes | None, str | None]:
  try:
    model = json.loads(model_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return None, "input_type"
  if not isinstance(model, dict) or not isinstance(model.get("@kernel"), dict):
    return None, "input_type"
  function_model = model["@kernel"]
  packed: list[int] = []
  lower, upper = problem.domain
  for item in problem.inputs:
    values = _flatten_model(function_model.get(f"%?{item.name}"))
    expected_count = 1
    for dimension in item.shape:
      expected_count *= dimension
    if values is None or len(values) != expected_count:
      return None, "input_length"
    if any(value < lower or value > upper for value in values):
      return None, "input_domain"
    if any(value < -(1 << 31) or value >= 1 << 31 for value in values):
      return None, "input_domain"
    packed.extend(values)
  return struct.pack(f"<{len(packed)}i", *packed), None


def _materialized_query(
  problem: PilotProblem, values: dict[str, tuple[int, ...]]
) -> str:
  source = render_query(problem)
  for item in problem.inputs:
    scalar_values = values[item.name]
    literal = "{" + ", ".join(str(value) for value in scalar_values) + "}"
    symbol_declaration = (
      f"  sym %?{item.name} : value {_shape_text(item.shape)} "
      f"in [{problem.domain[0]}, {problem.domain[1]}];"
    )
    concrete_declaration = (
      f"  let %{item.name}__pilot: {_shape_text(item.shape)} = {literal};"
    )
    if symbol_declaration not in source:
      raise ValueError(f"query is missing the symbol declaration for {item.name}")
    source = source.replace(symbol_declaration, concrete_declaration)
    source = re.sub(
      re.escape(f"%?{item.name}") + r"(?![A-Za-z0-9_])",
      f"%{item.name}__pilot",
      source,
    )
  return source


def _write_process_result(
  result: subprocess.CompletedProcess[str], prefix: Path
) -> dict[str, str]:
  stdout_path = prefix.with_suffix(".stdout.txt")
  stderr_path = prefix.with_suffix(".stderr.txt")
  stdout_path.write_text(result.stdout or "", encoding="utf-8")
  stderr_path.write_text(result.stderr or "", encoding="utf-8")
  return {"stdout": str(stdout_path), "stderr": str(stderr_path)}


def _run_query_file(
  source_path: Path, *, deadline: float | None, output_prefix: Path
) -> tuple[str, dict[str, str], str | None]:
  remaining = _time_left(deadline)
  if remaining <= 0:
    return "budget_exhausted", {}, "query replay deadline expired before launch"
  command = [
    str(SYMIRI),
    "--tensor-profile-version",
    "v1",
    "--main",
    "@kernel",
    str(source_path),
  ]
  try:
    result = subprocess.run(
      command,
      capture_output=True,
      text=True,
      check=False,
      timeout=_process_timeout(deadline),
    )
  except subprocess.TimeoutExpired:
    return "timeout", {}, "query replay timed out"
  except OSError as error:
    return "reference_execution_error", {}, str(error)
  paths = _write_process_result(result, output_prefix)
  paths["exit_code"] = str(result.returncode)
  if deadline is not None and _monotonic() > deadline:
    return "budget_exhausted", paths, "query replay completed after its deadline"
  if result.returncode != 0:
    return (
      "query_replay_disagreement",
      paths,
      result.stderr.strip() or result.stdout.strip(),
    )
  return "pass", paths, None


def _run_reference(
  problem: PilotProblem,
  input_bytes: bytes,
  attempt_dir: Path,
  deadline: float | None,
) -> tuple[str, bytes | None, dict[str, str], str | None]:
  reference_path = attempt_dir / "reference.sir"
  reference_path.write_text(render_reference(problem), encoding="utf-8")
  inputs_path = attempt_dir / "inputs.bin"
  inputs_path.write_bytes(input_bytes)
  output_path = attempt_dir / "reference.bin"
  remaining = _time_left(deadline)
  if remaining <= 0:
    return (
      "budget_exhausted",
      None,
      {"reference_sir": str(reference_path)},
      "deadline expired before reference launch",
    )
  command = [
    str(SYMIRI),
    "--tensor-profile-version",
    "v1",
    "--entry",
    "kernel",
    "--tensor-input",
    str(inputs_path),
    "--tensor-output",
    str(output_path),
    str(reference_path),
  ]
  try:
    result = subprocess.run(
      command,
      capture_output=True,
      text=True,
      check=False,
      timeout=_process_timeout(deadline),
    )
  except subprocess.TimeoutExpired:
    return (
      "timeout",
      None,
      {"reference_sir": str(reference_path)},
      "reference execution timed out",
    )
  except OSError as error:
    return (
      "reference_execution_error",
      None,
      {"reference_sir": str(reference_path)},
      str(error),
    )
  paths = {"reference_sir": str(reference_path), "inputs": str(inputs_path)}
  paths.update(_write_process_result(result, attempt_dir / "reference"))
  paths["reference_exit_code"] = str(result.returncode)
  if deadline is not None and _monotonic() > deadline:
    return (
      "budget_exhausted",
      None,
      paths,
      "reference execution completed after its deadline",
    )
  if result.returncode != 0:
    output = (result.stdout + result.stderr).lower()
    if "undefined behavior" in output or "ub:" in output:
      return "reference_ub", None, paths, result.stderr.strip() or result.stdout.strip()
    return (
      "reference_execution_error",
      None,
      paths,
      result.stderr.strip() or result.stdout.strip(),
    )
  try:
    output_bytes = output_path.read_bytes()
  except OSError as error:
    return "reference_execution_error", None, paths, str(error)
  paths["reference_output"] = str(output_path)
  expected_count = max(1, len(problem.target))
  if len(output_bytes) != expected_count * 4:
    return (
      "reference_execution_error",
      None,
      paths,
      "reference output has an unexpected byte count",
    )
  return "pass", output_bytes, paths, None


def validate_witness(
  problem: PilotProblem,
  input_bytes: bytes,
  *,
  artifact_dir: str | Path,
  budget_s: float | None = None,
) -> Attempt:
  """Validate packed input bytes against the independent reference and query."""
  started = _monotonic()
  attempt_dir = _attempt_directory(problem, artifact_dir, "witness")
  paths: dict[str, str] = {}
  timings: dict[str, float] = {}
  try:
    validate_problem(problem)
  except (TypeError, ValueError) as error:
    return _finish(
      "invalid_problem",
      started,
      attempt_dir=attempt_dir,
      paths=paths,
      reason=str(error),
    )
  deadline = None
  if budget_s is not None:
    if (
      not isinstance(budget_s, (int, float))
      or isinstance(budget_s, bool)
      or not math.isfinite(budget_s)
    ):
      return _finish(
        "invalid_problem",
        started,
        attempt_dir=attempt_dir,
        reason="invalid replay budget",
      )
    deadline = started + max(0.0, float(budget_s))
  decode_started = _monotonic()
  decoded, reason = _decode_input(problem, input_bytes)
  timings["input_decode"] = _monotonic() - decode_started
  if reason is not None:
    return _finish(
      "validation_failure",
      started,
      attempt_dir=attempt_dir,
      timings=timings,
      reason=reason,
    )
  assert decoded is not None
  reference_started = _monotonic()
  reference_status, output_bytes, reference_paths, detail = _run_reference(
    problem, input_bytes, attempt_dir, deadline
  )
  timings["reference_execution"] = _monotonic() - reference_started
  paths.update(reference_paths)
  if reference_status == "budget_exhausted":
    return _finish(
      "budget_exhausted",
      started,
      attempt_dir=attempt_dir,
      input_bytes=input_bytes,
      timings=timings,
      paths=paths,
      reason=detail,
    )
  if reference_status == "timeout":
    return _finish(
      "timeout",
      started,
      attempt_dir=attempt_dir,
      input_bytes=input_bytes,
      timings=timings,
      paths=paths,
      reason=detail,
    )
  if reference_status == "reference_ub":
    return _finish(
      "validation_failure",
      started,
      attempt_dir=attempt_dir,
      input_bytes=input_bytes,
      timings=timings,
      paths=paths,
      reason="reference_ub",
    )
  if reference_status != "pass" or output_bytes is None:
    return _finish(
      "validation_failure",
      started,
      attempt_dir=attempt_dir,
      input_bytes=input_bytes,
      timings=timings,
      paths=paths,
      reason="reference_execution_error" if detail else "reference_execution_error",
    )
  expected = struct.pack(f"<{len(problem.target)}i", *problem.target)
  if output_bytes != expected:
    return _finish(
      "validation_failure",
      started,
      attempt_dir=attempt_dir,
      input_bytes=input_bytes,
      output_bytes=output_bytes,
      timings=timings,
      paths=paths,
      reason="target_mismatch",
    )

  query_started = _monotonic()
  materialized_path = attempt_dir / "materialized-query.sir"
  try:
    materialized_path.write_text(
      _materialized_query(problem, decoded), encoding="utf-8"
    )
  except (TypeError, ValueError):
    return _finish(
      "validation_failure",
      started,
      attempt_dir=attempt_dir,
      input_bytes=input_bytes,
      output_bytes=output_bytes,
      timings=timings,
      paths=paths,
      reason="query_replay_disagreement",
    )
  paths["materialized_query"] = str(materialized_path)
  query_status, query_paths, query_detail = _run_query_file(
    materialized_path,
    deadline=deadline,
    output_prefix=attempt_dir / "materialized-query",
  )
  paths.update(
    {f"materialized_query_{key}": value for key, value in query_paths.items()}
  )
  timings["materialized_query_replay"] = _monotonic() - query_started
  if query_status == "budget_exhausted":
    return _finish(
      "budget_exhausted",
      started,
      attempt_dir=attempt_dir,
      input_bytes=input_bytes,
      output_bytes=output_bytes,
      timings=timings,
      paths=paths,
      reason=query_detail,
    )
  if query_status == "timeout":
    return _finish(
      "timeout",
      started,
      attempt_dir=attempt_dir,
      input_bytes=input_bytes,
      output_bytes=output_bytes,
      timings=timings,
      paths=paths,
      reason=query_detail,
    )
  if query_status != "pass":
    return _finish(
      "validation_failure",
      started,
      attempt_dir=attempt_dir,
      input_bytes=input_bytes,
      output_bytes=output_bytes,
      timings=timings,
      paths=paths,
      reason="query_replay_disagreement",
    )
  if deadline is not None and _monotonic() > deadline:
    return _finish(
      "budget_exhausted",
      started,
      attempt_dir=attempt_dir,
      input_bytes=input_bytes,
      output_bytes=output_bytes,
      timings=timings,
      paths=paths,
      reason="witness replay completed after its deadline",
      witness_validated=True,
    )
  return _finish(
    "validated",
    started,
    attempt_dir=attempt_dir,
    input_bytes=input_bytes,
    output_bytes=output_bytes,
    timings=timings,
    paths=paths,
  )


def _solver_status(stdout: str, stderr: str) -> str:
  text = stdout + "\n" + stderr
  if "unsupported_theory" in text.lower():
    return "unsupported_theory"
  if re.search(r"\bUNSAT(?:ISFIABLE)?\b", text, re.IGNORECASE):
    return "unsat"
  if re.search(r"\bUNKNOWN\b", text, re.IGNORECASE):
    return "unknown"
  if re.search(r"\bSAT(?:ISFIABLE)?\b", text, re.IGNORECASE):
    return "sat"
  return "invalid_problem"


def solve_problem(
  problem: PilotProblem,
  *,
  seed: int,
  budget_s: float,
  artifact_dir: str | Path,
) -> Attempt:
  """Solve the prescribed-output query and replay a strictly decoded witness."""
  started = _monotonic()
  attempt_dir = _attempt_directory(problem, artifact_dir, "solver")
  paths: dict[str, str] = {}
  timings: dict[str, float] = {}
  try:
    validate_problem(problem)
  except (TypeError, ValueError) as error:
    return _finish(
      "invalid_problem",
      started,
      attempt_dir=attempt_dir,
      paths=paths,
      reason=str(error),
    )
  if isinstance(seed, bool) or not isinstance(seed, int):
    return _finish(
      "invalid_problem", started, attempt_dir=attempt_dir, reason="seed must be integer"
    )
  if (
    isinstance(budget_s, bool)
    or not isinstance(budget_s, (int, float))
    or not math.isfinite(budget_s)
    or budget_s < 0
  ):
    return _finish(
      "invalid_problem",
      started,
      attempt_dir=attempt_dir,
      reason="invalid solver budget",
    )
  deadline = started + float(budget_s)
  query_path = attempt_dir / "query.sir"
  solved_path = attempt_dir / "solved.sir"
  model_path = attempt_dir / "model.json"
  query_started = _monotonic()
  try:
    query_path.write_text(render_query(problem), encoding="utf-8")
  except (TypeError, ValueError) as error:
    return _finish(
      "invalid_problem", started, attempt_dir=attempt_dir, reason=str(error)
    )
  timings["query_render"] = _monotonic() - query_started
  paths["query"] = str(query_path)
  if _time_left(deadline) <= 0:
    return _finish(
      "budget_exhausted",
      started,
      attempt_dir=attempt_dir,
      timings=timings,
      paths=paths,
      reason="deadline expired before solver launch",
    )
  remaining = _time_left(deadline)
  solver_timeout_ms = max(1, int(remaining * 1000))
  command = [
    str(SYMIR_SOLVE),
    "--main",
    "@kernel",
    "--path",
    "^entry",
    "--seed",
    str(seed),
    "--num-smt-threads",
    "1",
    "--timeout-ms",
    str(solver_timeout_ms),
    "--output",
    str(solved_path),
    "--emit-model",
    str(model_path),
    str(query_path),
  ]
  solver_started = _monotonic()
  try:
    result = subprocess.run(
      command,
      capture_output=True,
      text=True,
      check=False,
      timeout=min(_PROCESS_LIMIT_S, remaining + _CLEANUP_GRACE_S),
    )
  except subprocess.TimeoutExpired:
    timings["solver"] = _monotonic() - solver_started
    return _finish(
      "timeout",
      started,
      attempt_dir=attempt_dir,
      timings=timings,
      paths=paths,
      reason="external solver deadline expired",
    )
  except OSError as error:
    timings["solver"] = _monotonic() - solver_started
    paths.update(
      _write_process_result(
        subprocess.CompletedProcess(command, 127, "", str(error)),
        attempt_dir / "solver",
      )
    )
    return _finish(
      "invalid_problem",
      started,
      attempt_dir=attempt_dir,
      timings=timings,
      paths=paths,
      reason=str(error),
    )
  timings["solver"] = _monotonic() - solver_started
  paths.update(_write_process_result(result, attempt_dir / "solver"))
  paths["solver_exit_code"] = str(result.returncode)
  status = _solver_status(result.stdout, result.stderr)
  if _monotonic() > deadline:
    return _finish(
      "budget_exhausted",
      started,
      attempt_dir=attempt_dir,
      timings=timings,
      paths=paths,
      reason=f"solver completed after its deadline with status {status}",
    )
  if status != "sat":
    return _finish(
      status,
      started,
      attempt_dir=attempt_dir,
      timings=timings,
      paths=paths,
      reason=(result.stderr.strip() or result.stdout.strip() or None),
    )
  paths["solved_query"] = str(solved_path)
  paths["model"] = str(model_path)
  input_bytes, model_error = _decode_model(problem, model_path)
  if model_error is not None or input_bytes is None:
    return _finish(
      "validation_failure",
      started,
      attempt_dir=attempt_dir,
      timings=timings,
      paths=paths,
      reason=model_error or "input_type",
    )
  if _time_left(deadline) <= 0:
    return _finish(
      "budget_exhausted",
      started,
      attempt_dir=attempt_dir,
      input_bytes=input_bytes,
      timings=timings,
      paths=paths,
      reason="solver exhausted the selection deadline",
    )
  validation = validate_witness(
    problem,
    input_bytes,
    artifact_dir=attempt_dir / "replay",
    budget_s=_time_left(deadline),
  )
  timings.update(
    {f"replay_{name}": value for name, value in validation.stage_timings_s}
  )
  paths.update({f"replay_{name}": value for name, value in validation.artifact_paths})
  if validation.status != "validated":
    return _finish(
      validation.status,
      started,
      attempt_dir=attempt_dir,
      input_bytes=input_bytes,
      output_bytes=validation.output_bytes,
      timings=timings,
      paths=paths,
      reason=validation.reason or "SAT witness failed independent validation",
      witness_validated=validation.witness_validated,
    )
  if _time_left(deadline) <= 0:
    return _finish(
      "budget_exhausted",
      started,
      attempt_dir=attempt_dir,
      input_bytes=input_bytes,
      output_bytes=validation.output_bytes,
      timings=timings,
      paths=paths,
      reason="SAT witness validation completed after the selection deadline",
      witness_validated=True,
    )

  solved_started = _monotonic()
  solved_status, solved_paths, solved_detail = _run_query_file(
    solved_path, deadline=deadline, output_prefix=attempt_dir / "solved-query"
  )
  timings["solved_query_replay"] = _monotonic() - solved_started
  paths.update({f"solved_query_{key}": value for key, value in solved_paths.items()})
  if solved_status == "budget_exhausted":
    return _finish(
      "budget_exhausted",
      started,
      attempt_dir=attempt_dir,
      input_bytes=input_bytes,
      output_bytes=validation.output_bytes,
      timings=timings,
      paths=paths,
      reason=solved_detail,
      witness_validated=True,
    )
  if solved_status == "timeout":
    return _finish(
      "timeout",
      started,
      attempt_dir=attempt_dir,
      input_bytes=input_bytes,
      output_bytes=validation.output_bytes,
      timings=timings,
      paths=paths,
      reason=solved_detail,
      witness_validated=True,
    )
  if solved_status != "pass":
    return _finish(
      "validation_failure",
      started,
      attempt_dir=attempt_dir,
      input_bytes=input_bytes,
      output_bytes=validation.output_bytes,
      timings=timings,
      paths=paths,
      reason="query_replay_disagreement",
      witness_validated=True,
    )
  if _monotonic() > deadline:
    return _finish(
      "budget_exhausted",
      started,
      attempt_dir=attempt_dir,
      input_bytes=input_bytes,
      output_bytes=validation.output_bytes,
      timings=timings,
      paths=paths,
      reason="solved-query replay completed after the selection deadline",
      witness_validated=True,
    )
  return _finish(
    "validated",
    started,
    attempt_dir=attempt_dir,
    input_bytes=input_bytes,
    output_bytes=validation.output_bytes,
    timings=timings,
    paths=paths,
  )


_BOUNDARY_VALUES = (-8, -4, -2, -1, 0, 1, 2, 4, 8)
_METHODS = {"ordinary", "boundary", "construction", "smt"}


def _coordinates(shape: tuple[int, ...]):
  if not shape:
    yield ()
    return
  yield from itertools.product(*(range(size) for size in shape))


def _linear_index(coords: tuple[int, ...], shape: tuple[int, ...]) -> int:
  index = 0
  for coordinate, dimension in zip(coords, shape, strict=True):
    index = index * dimension + coordinate
  return index


def _construction_layout(
  problem: PilotProblem,
) -> tuple[str, tuple[int, ...], int, tuple[int, ...] | None] | None:
  if len(problem.inputs) != 1:
    return None
  input_tensor = problem.inputs[0]
  nodes = problem.nodes
  transpose_perm: tuple[int, ...] | None = None
  if len(nodes) == 1:
    reduction = nodes[0]
    if reduction.op != "sum" or reduction.operands != (input_tensor.name,):
      return None
    reduction_shape = input_tensor.shape
  elif len(nodes) == 2:
    transpose, reduction = nodes
    if (
      transpose.op != "transpose"
      or transpose.operands != (input_tensor.name,)
      or reduction.op != "sum"
      or reduction.operands != (transpose.node_id,)
    ):
      return None
    transpose_perm = transpose.permutation
    reduction_shape = transpose.shape
  else:
    return None
  if reduction.keep or reduction.axis is None:
    return None
  axis = reduction.axis % len(reduction_shape)
  return input_tensor.name, input_tensor.shape, axis, transpose_perm


def _construction_groups(
  problem: PilotProblem,
  layout: tuple[str, tuple[int, ...], int, tuple[int, ...] | None],
) -> tuple[tuple[int, ...], ...]:
  _, input_shape, axis, permutation = layout
  reduction_shape = (
    tuple(input_shape[index] for index in permutation)
    if permutation is not None
    else input_shape
  )
  groups: list[tuple[int, ...]] = []
  for output_coords in _coordinates(problem.output_shape):
    group = []
    for axis_index in range(reduction_shape[axis]):
      reduction_coords = list(output_coords)
      reduction_coords.insert(axis, axis_index)
      if permutation is None:
        input_coords = tuple(reduction_coords)
      else:
        mapped = [0] * len(input_shape)
        for output_axis, input_axis in enumerate(permutation):
          mapped[input_axis] = reduction_coords[output_axis]
        input_coords = tuple(mapped)
      group.append(_linear_index(input_coords, input_shape))
    groups.append(tuple(group))
  return tuple(groups)


def _constructed_input(
  problem: PilotProblem,
  groups: tuple[tuple[int, ...], ...],
  rng: random.Random,
) -> tuple[bytes | None, str | None]:
  lower, upper = problem.domain
  input_count = _input_leaf_count(problem)
  values: list[int | None] = [None] * input_count
  for target, group in zip(problem.target, groups, strict=True):
    if target < lower * len(group) or target > upper * len(group):
      return None, "analytic_infeasibility_certificate"
    residual = target
    for offset, input_index in enumerate(group):
      remaining = len(group) - offset - 1
      choice_lower = max(lower, residual - upper * remaining)
      choice_upper = min(upper, residual - lower * remaining)
      if choice_lower > choice_upper:
        return None, "analytic_infeasibility_certificate"
      value = rng.randint(choice_lower, choice_upper)
      values[input_index] = value
      residual -= value
  if any(value is None for value in values):
    return None, "constructor_mapping_incomplete"
  return struct.pack(f"<{len(values)}i", *values), None


def _proposal(problem: PilotProblem, rng: random.Random, *, boundary: bool) -> bytes:
  lower, upper = problem.domain
  choices = tuple(value for value in _BOUNDARY_VALUES if lower <= value <= upper)
  if boundary and not choices:
    raise ValueError("boundary menu has no values in the declared domain")
  values = []
  for _ in range(_input_leaf_count(problem)):
    use_uniform = not boundary or rng.random() < 0.5
    values.append(rng.randint(lower, upper) if use_uniform else rng.choice(choices))
  return struct.pack(f"<{len(values)}i", *values)


def _selection_directory(
  problem: PilotProblem, artifact_dir: str | Path, method: str
) -> Path:
  directory = Path(artifact_dir) / problem.problem_id / f"selection-{method}"
  directory.mkdir(parents=True, exist_ok=True)
  return directory


def select_input(
  problem: PilotProblem,
  *,
  method: str,
  seed: int,
  budget_s: float,
  artifact_dir: str | Path,
) -> Attempt:
  """Select one target-satisfying input under a shared end-to-end deadline."""
  started = _monotonic()
  if method not in _METHODS:
    directory = Path(artifact_dir) / "invalid-selection"
    directory.mkdir(parents=True, exist_ok=True)
    return _finish(
      "invalid_problem",
      started,
      attempt_dir=directory,
      reason=f"unknown input-selection method {method!r}",
    )
  try:
    validate_problem(problem)
  except (TypeError, ValueError) as error:
    directory = Path(artifact_dir) / "invalid-selection"
    directory.mkdir(parents=True, exist_ok=True)
    return _finish("invalid_problem", started, attempt_dir=directory, reason=str(error))
  directory = _selection_directory(problem, artifact_dir, method)
  if isinstance(seed, bool) or not isinstance(seed, int):
    return _finish(
      "invalid_problem", started, attempt_dir=directory, reason="seed must be integer"
    )
  if (
    isinstance(budget_s, bool)
    or not isinstance(budget_s, (int, float))
    or not math.isfinite(budget_s)
    or budget_s < 0
  ):
    return _finish(
      "invalid_problem",
      started,
      attempt_dir=directory,
      reason="invalid selection budget",
    )
  deadline = started + float(budget_s)
  if method == "construction":
    layout = _construction_layout(problem)
    if layout is None:
      return _finish(
        "unsupported_baseline",
        started,
        attempt_dir=directory,
        reason="construction supports only disjoint sums and pure transpose-sum layouts",
        supported=False,
      )
    groups = _construction_groups(problem, layout)
    input_bytes, reason = _constructed_input(problem, groups, random.Random(seed))
    if reason is not None or input_bytes is None:
      return _finish(
        "budget_exhausted",
        started,
        attempt_dir=directory,
        reason=reason or "constructor found no witness",
        supported=True,
      )
    if _monotonic() > deadline:
      return _finish(
        "budget_exhausted",
        started,
        attempt_dir=directory,
        input_bytes=input_bytes,
        reason="construction completed after its deadline",
        supported=True,
      )
    proposal_dir = directory / "proposal-000001"
    witness = validate_witness(
      problem,
      input_bytes,
      artifact_dir=proposal_dir,
      budget_s=_time_left(deadline),
    )
    if witness.status == "validated" and _monotonic() <= deadline:
      status = "validated"
    elif witness.status == "validated" or witness.status == "budget_exhausted":
      status = "budget_exhausted"
    else:
      status = witness.status
    return _finish(
      status,
      started,
      attempt_dir=directory,
      input_bytes=input_bytes,
      output_bytes=witness.output_bytes,
      timings=dict(witness.stage_timings_s),
      paths=dict(witness.artifact_paths),
      reason=witness.reason
      or (
        "constructed witness completed after its deadline"
        if status == "budget_exhausted"
        else None
      ),
      supported=True,
      witness_validated=witness.witness_validated or witness.status == "validated",
    )

  if method == "smt":
    return solve_problem(
      problem,
      seed=seed,
      budget_s=_time_left(deadline),
      artifact_dir=directory,
    )

  rng = random.Random(seed)
  boundary_choices = tuple(
    value
    for value in _BOUNDARY_VALUES
    if problem.domain[0] <= value <= problem.domain[1]
  )
  if method == "boundary" and not boundary_choices:
    return _finish(
      "unsupported_baseline",
      started,
      attempt_dir=directory,
      reason="boundary menu has no values in the declared domain",
      supported=False,
    )
  attempt_count = 0
  total_timings: dict[str, float] = {"proposal_generation": 0.0}
  paths: dict[str, str] = {}
  while _monotonic() < deadline:
    proposal_started = _monotonic()
    input_bytes = _proposal(problem, rng, boundary=method == "boundary")
    total_timings["proposal_generation"] += _monotonic() - proposal_started
    attempt_count += 1
    witness = validate_witness(
      problem,
      input_bytes,
      artifact_dir=directory / f"proposal-{attempt_count:06d}",
      budget_s=_time_left(deadline),
    )
    for stage, elapsed in witness.stage_timings_s:
      total_timings[stage] = total_timings.get(stage, 0.0) + elapsed
    paths.update(
      {
        f"proposal_{attempt_count:06d}_{key}": value
        for key, value in witness.artifact_paths
      }
    )
    if witness.status == "validated":
      if _monotonic() > deadline:
        return _finish(
          "budget_exhausted",
          started,
          attempt_dir=directory,
          input_bytes=input_bytes,
          output_bytes=witness.output_bytes,
          timings=total_timings,
          paths=paths,
          reason="validated witness completed after its deadline",
          attempt_count=attempt_count,
          supported=True if method == "boundary" else None,
          witness_validated=True,
        )
      return _finish(
        "validated",
        started,
        attempt_dir=directory,
        input_bytes=input_bytes,
        output_bytes=witness.output_bytes,
        timings=total_timings,
        paths=paths,
        attempt_count=attempt_count,
        supported=True if method == "boundary" else None,
      )
    if witness.status == "budget_exhausted":
      return _finish(
        "budget_exhausted",
        started,
        attempt_dir=directory,
        input_bytes=input_bytes,
        output_bytes=witness.output_bytes,
        timings=total_timings,
        paths=paths,
        reason=witness.reason or "selection deadline exhausted during witness replay",
        attempt_count=attempt_count,
        supported=True if method == "boundary" else None,
        witness_validated=witness.witness_validated,
      )
    if witness.status == "timeout":
      status = "timeout"
      reason = witness.reason
      break
    if witness.status == "validation_failure" and witness.reason in {
      "target_mismatch",
      "reference_ub",
    }:
      if _monotonic() >= deadline:
        status = "budget_exhausted"
        reason = "proposal budget exhausted without a validated input"
        break
      continue
    status = witness.status
    reason = witness.reason or "proposal validation failed"
    break
  else:
    status = "budget_exhausted"
    reason = "proposal budget exhausted without a validated input"
  return _finish(
    status,
    started,
    attempt_dir=directory,
    timings=total_timings,
    paths=paths,
    reason=reason,
    attempt_count=attempt_count,
    supported=True if method == "boundary" else None,
  )
