"""Exact nonnegative integer domain for a canonical two-state recurrence.

Admission recognizes the complete source skeleton and derives bounds from the
input domain. Nonnegative group totals bound every possible reduction tree.
"""

import hashlib
import json
import math
import re
import struct

KIND = "composed-add-sum-f32-v1"
EXACT_BOUND = 2**24
OPERATIONS = ["add", "sum", "sum", "broadcast", "broadcast", "add", "add", "add"]


def render_program(shape=(2, 2), iterations=2, *, dtype="f32"):
  """Compute both next-state values before assigning either state."""
  if (
    len(shape) != 2
    or any(type(dim) is not int or not 1 <= dim <= 512 for dim in shape)
    or math.prod(shape) > 512
    or type(iterations) is not int
    or not 1 <= iterations <= 8
    or dtype not in {"f32", "i32"}
  ):
    raise ValueError("invalid bounded recurrence shape, iterations or dtype")
  rows, columns = shape
  tensor = f"tensor [{rows}, {columns}] {dtype}"
  zero = "0.0" if dtype == "f32" else "0"
  initial = "{" + ", ".join([zero] * (rows * columns)) + "}"
  declarations = "  let mut %i: i32 = undef;\n"
  for name in ("a", "b", "v", "r", "c", "rb", "cb", "a_next", "b_next", "out"):
    ty = tensor
    if name == "r":
      ty = f"tensor [{rows}, 1] {dtype}"
    if name == "c":
      ty = f"tensor [1, {columns}] {dtype}"
    declarations += (
      f"  let mut %{name}: {ty} = {initial if name == 'b' else 'undef'};\n"
    )
  return (
    f"fun @kernel(%x: {tensor}) : {tensor} {{\n"
    + declarations
    + "^entry:\n  %a = %x;\n  %i = 0;\n  br ^header;\n"
    + f"^header:\n  br %i < {iterations}, ^body, ^exit;\n"
    + "^body:\n"
    + "  %v = tensor add %a, %b;\n"
    + "  %r = tensor sum %v, 1, keep;\n"
    + "  %c = tensor sum %v, 0, keep;\n"
    + f"  %rb = tensor broadcast %r, [{rows}, {columns}];\n"
    + f"  %cb = tensor broadcast %c, [{rows}, {columns}];\n"
    + "  %a_next = tensor add %a, %rb;\n"
    + "  %b_next = tensor add %b, %cb;\n"
    + "  %a = %a_next;\n  %b = %b_next;\n"
    + "  %i = %i + 1;\n  br ^header;\n"
    + "^exit:\n  %out = tensor add %a, %b;\n  ret %out;\n}\n"
  )


def _recognize(source):
  signature = re.search(r"fun @kernel\(%x: tensor \[(\d+),\s*(\d+)\] f32\)", source)
  loop = re.search(r"br %i < (\d+), \^body, \^exit;", source)
  if signature is None or loop is None:
    raise ValueError("source is outside the canonical composed exact family")
  shape = tuple(int(value) for value in signature.groups())
  iterations = int(loop[1])
  canonical = render_program(shape, iterations)
  # Preserve token separation when normalizing whitespace. Native case loading
  # additionally checks the source through the shared frontend.
  if " ".join(source.split()) != " ".join(canonical.split()):
    raise ValueError("source differs from the canonical composed exact recurrence")
  return shape, iterations


def certify(source, input_bytes, domain=(0, 8)):
  """Return recomputed proof evidence and complete independent output bits."""
  shape, iterations = _recognize(source)
  if (
    not isinstance(domain, (list, tuple))
    or len(domain) != 2
    or any(type(value) is not int for value in domain)
    or not 0 <= domain[0] <= domain[1] <= EXACT_BOUND
  ):
    raise ValueError("exact domain requires nonnegative bounded integers")
  rows, columns = shape
  count = rows * columns
  if not isinstance(input_bytes, bytes) or len(input_bytes) != count * 4:
    raise ValueError("input bytes do not match the recognized tensor shape")
  words = struct.unpack(f"<{count}I", input_bytes)
  leaves = struct.unpack(f"<{count}f", input_bytes)
  if any(
    not math.isfinite(value)
    or not value.is_integer()
    or not domain[0] <= value <= domain[1]
    or (value == 0 and word != 0)
    for value, word in zip(leaves, words)
  ):
    raise ValueError("input requires nonnegative integer f32 values and positive zero")
  values = {"x": tuple(map(int, leaves)), "b": (0,) * count}
  upper = {"x": (domain[1],) * count, "b": (0,) * count}
  bounds = []

  def record(destination, operation, actual, maxima, visit, *, partial=False):
    maximum = max(maxima)
    if maximum > EXACT_BOUND:
      raise ValueError(
        "materialized value or reduction partial sum exceeds exact range"
      )
    values[destination] = actual
    upper[destination] = maxima
    evidence = {
      "block": "^body" if visit else ("^exit" if destination == "out" else "^entry"),
      "visit": visit,
      "destination": "%" + destination,
      "operation": operation,
      "max_value": maximum,
    }
    if partial:
      evidence["max_partial_sum"] = maximum
    bounds.append(evidence)

  record("x", "input", values["x"], upper["x"], 0)
  record("a", "copy", values["x"], upper["x"], 0)
  record("b", "positive-zero", values["b"], upper["b"], 0)

  def add(destination, left, right, visit):
    record(
      destination,
      "add",
      tuple(a + b for a, b in zip(values[left], values[right])),
      tuple(a + b for a, b in zip(upper[left], upper[right])),
      visit,
    )

  for visit in range(1, iterations + 1):
    add("v", "a", "b", visit)
    for destination, groups in (
      ("r", [range(row * columns, (row + 1) * columns) for row in range(rows)]),
      ("c", [range(column, count, columns) for column in range(columns)]),
    ):
      record(
        destination,
        "sum",
        tuple(sum(values["v"][i] for i in group) for group in groups),
        tuple(sum(upper["v"][i] for i in group) for group in groups),
        visit,
        partial=True,
      )
    for destination, operand, indices in (
      ("rb", "r", [i // columns for i in range(count)]),
      ("cb", "c", [i % columns for i in range(count)]),
    ):
      record(
        destination,
        "broadcast",
        tuple(values[operand][i] for i in indices),
        tuple(upper[operand][i] for i in indices),
        visit,
      )
    add("a_next", "a", "rb", visit)
    add("b_next", "b", "cb", visit)
    for state in ("a", "b"):
      values[state] = values[state + "_next"]
      upper[state] = upper[state + "_next"]
  add("out", "a", "b", 0)
  proof = {
    "kind": KIND,
    "complete_output": True,
    "integer_exactness_bound": EXACT_BOUND,
    "input_domain": list(domain),
    "allowed_operations": ["add", "sum", "broadcast"],
    "signed_zero": "positive-zero-only",
    "path": ["^entry", "^header"]
    + [block for _ in range(iterations) for block in ("^body", "^header")]
    + ["^exit"],
    "bounds": bounds,
  }
  return proof, struct.pack(f"<{count}f", *values["out"])


def validate(source, input_bytes, expected_bytes, proof, *, artifacts=None):
  """Reject missing obligations, stale bounds, changed paths and wrong outputs."""
  if not isinstance(proof, dict):
    raise ValueError("composed exactness evidence must be an object")
  recomputed, expected = certify(source, input_bytes, proof.get("input_domain"))
  recorded_selection = proof.get("selection")
  comparable = dict(proof)
  comparable.pop("selection", None)
  try:
    # JSON comparison preserves the difference between booleans and integers.
    identical = json.dumps(comparable, sort_keys=True, allow_nan=False) == json.dumps(
      recomputed, sort_keys=True, allow_nan=False
    )
  except (TypeError, ValueError) as error:
    raise ValueError("composed exactness evidence is not canonical JSON") from error
  if not identical:
    raise ValueError("composed exactness evidence differs from derived obligations")
  if expected_bytes != expected:
    raise ValueError("complete output bits differ from independent exact execution")
  if recorded_selection is not None:
    if artifacts is None:
      raise ValueError("solver selection artifacts are required")
    validate_selection(
      source, input_bytes, expected_bytes, recorded_selection, artifacts
    )
  return expected


def validate_selection(source, input_bytes, expected_bytes, selection, artifacts):
  """Check the exact i32 query, returned model and solver terminal record."""
  required = {
    "method",
    "domain",
    "target",
    "path",
    "query_file",
    "query_sha256",
    "model_file",
    "model_sha256",
    "result_file",
    "result_sha256",
    "replay",
  }
  if not isinstance(selection, dict) or set(selection) != required:
    raise ValueError("solver selection evidence is incomplete")
  if selection["method"] != "bounded-integer-twin-smt" or selection["domain"] != [0, 8]:
    raise ValueError("solver selection domain or method is invalid")
  if selection["target"] != {"index": [0, 0], "value": 63}:
    raise ValueError("solver selection target is invalid")
  if selection["path"] != [
    "^entry",
    "^header",
    "^body",
    "^header",
    "^body",
    "^header",
    "^exit",
  ]:
    raise ValueError("solver selection path does not visit both loop iterations")
  if selection["replay"] != "complete-f32-output-bitwise":
    raise ValueError("solver selection lacks complete bitwise replay")
  try:
    query = artifacts[selection["query_file"]]
    model_bytes = artifacts[selection["model_file"]]
    result_bytes = artifacts[selection["result_file"]]
    if hashlib.sha256(query).hexdigest() != selection["query_sha256"]:
      raise ValueError("solver query digest does not match")
    if hashlib.sha256(model_bytes).hexdigest() != selection["model_sha256"]:
      raise ValueError("solver model digest does not match")
    if hashlib.sha256(result_bytes).hexdigest() != selection["result_sha256"]:
      raise ValueError("solver result digest does not match")
  except (KeyError, TypeError) as error:
    raise ValueError("solver selection artifact is missing") from error
  from test.lib.build_tensor_exact_loop_case import render_query

  if query != render_query().encode():
    raise ValueError("solver query differs from the admitted integer twin")
  try:
    model = json.loads(model_bytes)
    witness = model["@kernel"]["%?x"]
    values = witness if isinstance(witness, list) else witness["leaves"]
    solver_result = json.loads(result_bytes)
  except (KeyError, TypeError, json.JSONDecodeError) as error:
    raise ValueError("solver model or terminal result is malformed") from error
  if (
    not isinstance(values, list)
    or len(values) != 4
    or any(type(value) is not int or not 0 <= value <= 8 for value in values)
  ):
    raise ValueError("solver model has invalid leaves for the declared domain")
  if struct.pack("<4f", *values) != input_bytes:
    raise ValueError("solver model leaves differ from the exported input bits")
  if (
    not isinstance(solver_result, dict)
    or solver_result.get("status") != "SAT"
    or solver_result.get("exit_code") != 0
    or not isinstance(solver_result.get("stdout"), str)
    or solver_result["stdout"].strip().split()[:1] != ["SAT"]
    or not isinstance(solver_result.get("command"), list)
  ):
    raise ValueError("solver result is not a successful SAT witness")
  command = solver_result["command"]
  for option, expected_value in (
    ("--path", ",".join(selection["path"])),
    ("--tensor-param-domain", "0,8"),
    ("--main", "@kernel"),
  ):
    try:
      actual = command[command.index(option) + 1]
    except (ValueError, IndexError) as error:
      raise ValueError(f"solver command omits {option}") from error
    if actual != expected_value:
      raise ValueError(f"solver command {option} disagrees with recorded selection")
  output = struct.unpack("<4f", expected_bytes)
  if output[0] != selection["target"]["value"]:
    raise ValueError("complete reference output does not satisfy the solver target")
