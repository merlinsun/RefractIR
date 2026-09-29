"""Run bounded cvc5 transcendental probes without trusting model text."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

_EXPECTED_VERSION = "1.3.4"
_DEFAULT_MENU = (-2, -1, 0, 1, 2, 3)


def _text(value):
  if value is None:
    return ""
  if isinstance(value, bytes):
    return value.decode("utf-8", errors="replace")
  return value


def _result(status, **fields):
  result = {
    "status": status,
    "elapsed_ns": 0,
    "stdout": "",
    "stderr": "",
    "solver_version": None,
    "external_timeout": False,
    "accepted_witness": False,
    "materialization_status": "not_attempted",
    "materialized_inputs": {},
    "materialization_error": None,
  }
  result.update(fields)
  return result


def _resolve_binary(binary):
  candidate = os.fspath(binary)
  if os.sep in candidate:
    return (
      candidate if os.path.isfile(candidate) and os.access(candidate, os.X_OK) else None
    )
  return shutil.which(candidate)


def _check_version(binary, expected_version, deadline_s):
  try:
    completed = subprocess.run(
      [binary, "--version"],
      capture_output=True,
      text=True,
      timeout=deadline_s,
      check=False,
    )
  except subprocess.TimeoutExpired as error:
    return None, _result(
      "synthesis_timeout",
      stderr=_text(error.stderr),
      stdout=_text(error.stdout),
      external_timeout=True,
    )
  except OSError as error:
    return None, _result("environment_unavailable", stderr=str(error))

  version_output = completed.stdout + completed.stderr
  match = re.search(r"\bcvc5\s+version\s+(\d+\.\d+\.\d+)\b", version_output)
  if completed.returncode != 0 or not match:
    return None, _result(
      "invalid_solver_version",
      stdout=completed.stdout,
      stderr=completed.stderr,
    )
  version = match.group(1)
  if version != expected_version:
    return version, _result(
      "invalid_solver_version",
      solver_version=version,
      stdout=completed.stdout,
      stderr=completed.stderr,
    )
  return version, None


def _parse_sexprs(text):
  text = re.sub(r";[^\n]*", "", text)
  tokens = re.findall(r"\(|\)|[^\s()]+", text)
  stack = [[]]
  for token in tokens:
    if token == "(":
      child = []
      stack[-1].append(child)
      stack.append(child)
    elif token == ")":
      if len(stack) == 1:
        raise ValueError("unmatched closing parenthesis")
      stack.pop()
    else:
      stack[-1].append(token)
  if len(stack) != 1:
    raise ValueError("unmatched opening parenthesis")
  return stack[0]


def _requested_variables(query):
  match = re.search(r"\(\s*get-value\s+\(([^()]*)\)\s*\)", query)
  if not match:
    return []
  return [token for token in match.group(1).split() if token]


def _assignments(expressions, variables):
  wanted = set(variables)
  found = {}

  def visit(expression):
    if not isinstance(expression, list):
      return
    if len(expression) == 2 and isinstance(expression[0], str):
      if expression[0] in wanted:
        found[expression[0]] = expression[1]
        return
    for child in expression:
      visit(child)

  for expression in expressions:
    visit(expression)
  return found


def _fraction(expression):
  if isinstance(expression, str):
    try:
      return Fraction(expression)
    except (ValueError, ZeroDivisionError):
      return None
  if not isinstance(expression, list):
    return None
  if len(expression) == 2 and expression[0] in ("-", "to_real"):
    value = _fraction(expression[1])
    if value is None:
      return None
    return -value if expression[0] == "-" else value
  if len(expression) == 3 and expression[0] == "/":
    numerator = _fraction(expression[1])
    denominator = _fraction(expression[2])
    if numerator is None or denominator in (None, 0):
      return None
    return numerator / denominator
  return None


def _expr_text(expression):
  if isinstance(expression, list):
    return "(" + " ".join(_expr_text(item) for item in expression) + ")"
  return str(expression)


def _materialize(stdout, query, finite_menu):
  variables = _requested_variables(query)
  if not variables:
    return "not_requested", {}, None
  try:
    expressions = _parse_sexprs(stdout)
  except ValueError as error:
    return "rejected", {}, f"invalid model response: {error}"
  values = _assignments(expressions, variables)
  allowed = {Fraction(str(value)) for value in finite_menu}
  materialized = {}
  for variable in variables:
    if variable not in values:
      return "rejected", {}, f"model has no value for {variable}"
    number = _fraction(values[variable])
    if number is None:
      return (
        "rejected",
        {},
        f"model value for {variable} is not a rational menu literal: "
        f"{_expr_text(values[variable])}",
      )
    if number not in allowed:
      return (
        "rejected",
        {},
        f"model value for {variable} is outside the finite menu: {number}",
      )
    materialized[variable] = (
      number.numerator
      if number.denominator == 1
      else [number.numerator, number.denominator]
    )
  return "materialized", materialized, None


def run_probe(
  binary,
  query_path,
  solver_ms=5000,
  deadline_s=6,
  finite_menu=_DEFAULT_MENU,
  expected_version=_EXPECTED_VERSION,
):
  """Run one query and classify solver, deadline, and model materialization.

  A materialized model is only a candidate. This function deliberately never
  marks it accepted; independent tensor-reference validation happens later.
  """
  resolved = _resolve_binary(binary) if binary else None
  if resolved is None:
    return _result("environment_unavailable", stderr="cvc5 executable is unavailable")

  version, version_failure = _check_version(resolved, expected_version, deadline_s)
  if version_failure is not None:
    version_failure["solver_version"] = version
    return version_failure

  try:
    query_text = Path(query_path).read_text(encoding="utf-8")
  except OSError as error:
    return _result("invalid_query", solver_version=version, stderr=str(error))

  started = time.perf_counter_ns()
  try:
    completed = subprocess.run(
      [
        resolved,
        "--lang=smt2",
        f"--tlimit-per={int(solver_ms)}",
        os.fspath(query_path),
      ],
      capture_output=True,
      text=True,
      timeout=deadline_s,
      check=False,
    )
  except subprocess.TimeoutExpired as error:
    return _result(
      "synthesis_timeout",
      elapsed_ns=time.perf_counter_ns() - started,
      stdout=_text(error.stdout),
      stderr=_text(error.stderr),
      solver_version=version,
      external_timeout=True,
    )
  except OSError as error:
    return _result(
      "environment_unavailable",
      elapsed_ns=time.perf_counter_ns() - started,
      stderr=str(error),
      solver_version=version,
    )

  elapsed = time.perf_counter_ns() - started
  output_lines = [
    line.strip() for line in completed.stdout.splitlines() if line.strip()
  ]
  status_line = next(
    (line for line in output_lines if line in {"sat", "unsat", "unknown"}), None
  )
  if completed.returncode != 0 or status_line is None:
    return _result(
      "probe_error",
      elapsed_ns=elapsed,
      stdout=completed.stdout,
      stderr=completed.stderr,
      solver_version=version,
    )
  if status_line == "unsat":
    return _result(
      "synthesis_unsat",
      elapsed_ns=elapsed,
      stdout=completed.stdout,
      stderr=completed.stderr,
      solver_version=version,
    )
  if status_line == "unknown":
    return _result(
      "synthesis_unknown",
      elapsed_ns=elapsed,
      stdout=completed.stdout,
      stderr=completed.stderr,
      solver_version=version,
    )

  materialization_status, inputs, error = _materialize(
    completed.stdout, query_text, finite_menu
  )
  result_status = (
    "materialization_rejected"
    if materialization_status == "rejected"
    else "synthesis_sat"
  )
  return _result(
    result_status,
    elapsed_ns=elapsed,
    stdout=completed.stdout,
    stderr=completed.stderr,
    solver_version=version,
    materialization_status=materialization_status,
    materialized_inputs=inputs,
    materialization_error=error,
  )


def run_feasibility_suite(
  binary, query_dir, solver_ms=5000, deadline_s=6, environment_id=None
):
  query_dir = Path(query_dir)
  manifest_path = query_dir / "queries.json"
  try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    return {
      "schema": "refractir.tensor-cvc5-probe.v1",
      "gate_status": "invalid_probe_manifest",
      "route_available": False,
      "error": str(error),
      "queries": [],
    }

  records = []
  for query in manifest["queries"]:
    result = run_probe(
      binary,
      query_dir / query["file"],
      solver_ms=solver_ms,
      deadline_s=deadline_s,
      finite_menu=manifest["finite_menu"],
    )
    records.append(
      {
        "id": query["id"],
        "expected_status": query["expected_status"],
        "result": result,
        "matches_expected": result["status"] == f"synthesis_{query['expected_status']}",
        "concrete_validation": "pending",
      }
    )

  if any(record["result"]["status"] == "environment_unavailable" for record in records):
    gate_status = "environment_unavailable"
  elif not all(record["matches_expected"] for record in records):
    gate_status = "failed"
  elif any(
    record["expected_status"] == "sat"
    and record["result"]["materialization_status"] != "materialized"
    for record in records
  ):
    gate_status = "failed"
  else:
    gate_status = "awaiting_reference_validation"

  report = {
    "schema": "refractir.tensor-cvc5-probe.v1",
    "solver_version_expected": _EXPECTED_VERSION,
    "solver_time_limit_ms": int(solver_ms),
    "external_deadline_s": float(deadline_s),
    "gate_status": gate_status,
    "route_available": False,
    "queries": records,
  }
  if environment_id:
    report["probe_environment"] = environment_id
  return report


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--binary", default=shutil.which("cvc5"))
  parser.add_argument(
    "--query-dir",
    default=str(Path(__file__).resolve().parents[1] / "tensor" / "cvc5"),
  )
  parser.add_argument("--output", required=True)
  parser.add_argument("--solver-ms", type=int, default=5000)
  parser.add_argument("--deadline-s", type=float, default=6)
  parser.add_argument("--environment-id")
  args = parser.parse_args(argv)

  report = run_feasibility_suite(
    args.binary,
    args.query_dir,
    solver_ms=args.solver_ms,
    deadline_s=args.deadline_s,
    environment_id=args.environment_id,
  )
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(
    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
  return 0 if report["gate_status"] == "awaiting_reference_validation" else 2


if __name__ == "__main__":
  sys.exit(main())
