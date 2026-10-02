"""Exact nonnegative integer domain for a canonical two-state recurrence.

Admission recognizes the complete source skeleton and derives bounds from the
input domain. Nonnegative group totals bound every possible reduction tree.
"""

import hashlib
import itertools
import json
import math
import re
import struct
from pathlib import Path

KIND = "composed-add-sum-f32-v1"
EXACT_BOUND = 2**24
OPERATIONS = ["add", "sum", "sum", "broadcast", "broadcast", "add", "add", "add"]
REGION_KIND = "bounded-region-add-sum-f32-v1"


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


def _shape(text):
  try:
    result = tuple(int(part.strip()) for part in text.split(","))
  except (AttributeError, ValueError) as error:
    raise ValueError("generated region contains a malformed tensor shape") from error
  if not result or any(size <= 0 for size in result) or math.prod(result) > 512:
    raise ValueError("generated region tensor shape is outside the bounded subset")
  return result


def _coordinates(shape):
  return itertools.product(*(range(size) for size in shape))


def _flat_index(coordinates, shape):
  result = 0
  for coordinate, extent in zip(coordinates, shape):
    result = result * extent + coordinate
  return result


def _broadcast_shape(left, right):
  rank = max(len(left), len(right))
  result = []
  for offset in range(1, rank + 1):
    lhs = left[-offset] if offset <= len(left) else 1
    rhs = right[-offset] if offset <= len(right) else 1
    if lhs != rhs and lhs != 1 and rhs != 1:
      raise ValueError("generated region add shapes do not broadcast")
    result.append(max(lhs, rhs))
  return tuple(reversed(result))


def _broadcast_values(values, source_shape, target_shape):
  if len(source_shape) > len(target_shape):
    raise ValueError("generated region broadcast cannot lower rank")
  padding = len(target_shape) - len(source_shape)
  aligned = (1,) * padding + tuple(source_shape)
  if any(src != 1 and src != dst for src, dst in zip(aligned, target_shape)):
    raise ValueError("generated region broadcast shapes are incompatible")
  result = []
  for target_coordinates in _coordinates(target_shape):
    source_coordinates = tuple(
      0 if source_extent == 1 else target_coordinates[axis + padding]
      for axis, source_extent in enumerate(source_shape)
    )
    result.append(values[_flat_index(source_coordinates, source_shape)])
  return tuple(result)


def _generated_region_program(source):
  """Parse one bounded two-state loop containing only generated region ops."""
  signature = re.search(
    r"(?m)^fun @kernel\(%x: tensor \[([^\]]+)\] f32\) : "
    r"tensor \[([^\]]+)\] f32 \{$",
    source,
  )
  if signature is None:
    raise ValueError("source is outside the bounded generated-region signature")
  input_shape = _shape(signature[1])
  output_shape = _shape(signature[2])
  if input_shape != output_shape or len(input_shape) != 2:
    raise ValueError("generated region requires one rank-two input and matching output")

  block_names = []
  blocks = {}
  current = None
  for line in source.splitlines():
    text = line.strip()
    label = re.fullmatch(r"\^([A-Za-z_][A-Za-z_0-9]*):", text)
    if label:
      current = label[1]
      block_names.append(current)
      blocks[current] = []
    elif current is not None and text and text != "}":
      blocks[current].append(text)
  if block_names != ["entry", "header", "body", "exit"]:
    raise ValueError("generated region requires one canonical bounded loop")

  tensor_types = {}
  initializers = {}
  declaration = re.compile(
    r"^\s*let mut (%[A-Za-z_][A-Za-z_0-9.$-]*): tensor "
    r"\[([^\]]+)\] f32 = (.*);\s*$",
    re.MULTILINE,
  )
  for match in declaration.finditer(source):
    name = match[1][1:]
    if name in tensor_types:
      raise ValueError("generated region redeclares a tensor value")
    tensor_types[name] = _shape(match[2])
    initializers[name] = match[3].strip()
  for name, shape in (
    ("acc_x", input_shape),
    ("acc_y", input_shape),
    ("out", output_shape),
  ):
    if tensor_types.get(name) != shape:
      raise ValueError(
        "generated region state or return declarations do not match the signature"
      )
  if initializers.get("acc_x") != "undef":
    raise ValueError("generated region first state must be initialized from its input")
  if initializers.get("out") != "undef":
    raise ValueError("generated region output must be uninitialized before the loop")
  zero_initializer = initializers.get("acc_y", "")
  zero_values = re.fullmatch(r"\{\s*(.*?)\s*\}", zero_initializer)
  if zero_values is None:
    raise ValueError("generated region second state must start at positive zero")
  zero_leaves = [leaf.strip() for leaf in zero_values[1].split(",")]
  if len(zero_leaves) != math.prod(input_shape) or any(
    leaf != "0.0" for leaf in zero_leaves
  ):
    raise ValueError("generated region second state must start at positive zero")
  if any(
    name not in {"acc_x", "acc_y", "out"} and initializer != "undef"
    for name, initializer in initializers.items()
  ):
    raise ValueError(
      "generated region temporaries must be uninitialized before the loop"
    )

  scalar_index = re.search(r"(?m)^\s*let mut %i: i32 = undef;\s*$", source)
  declarations = re.findall(r"(?m)^\s*let mut ", source)
  if scalar_index is None or len(declarations) != len(tensor_types) + 1:
    raise ValueError("generated region loop counter is missing")
  if "x" in tensor_types:
    raise ValueError("generated region cannot redeclare its input parameter")
  tensor_types["x"] = input_shape
  if blocks["entry"] != ["%acc_x = %x;", "%i = 0;", "br ^header;"]:
    raise ValueError(
      "generated region entry does not initialize the two states canonically"
    )
  header = re.fullmatch(r"br %i < (\d+), \^body, \^exit;", " ".join(blocks["header"]))
  if header is None or not 1 <= int(header[1]) <= 8:
    raise ValueError("generated region loop bound is outside the bounded subset")
  iterations = int(header[1])
  if blocks["exit"] != [
    "%out = tensor add %acc_x, %acc_y;",
    "ret %out;",
  ]:
    raise ValueError("generated region return must combine both final states")

  body = blocks["body"]
  if len(body) < 5 or body[-2:] != ["%i = %i + 1;", "br ^header;"]:
    raise ValueError("generated region body has an invalid loop backedge")
  state_update_lines = body[-4:-2]
  updates = {}
  for line in state_update_lines:
    update = re.fullmatch(r"(%acc_[xy]) = (%[A-Za-z_][A-Za-z_0-9.$-]*);", line)
    if update is None or update[1][1:] in updates:
      raise ValueError(
        "generated region must update both states after computing its roots"
      )
    updates[update[1][1:]] = update[2][1:]
  if set(updates) != {"acc_x", "acc_y"}:
    raise ValueError("generated region must update both loop-carried states")

  nodes = []
  definitions = {}
  for line in body[:-4]:
    assignment = re.fullmatch(r"(%[A-Za-z_][A-Za-z_0-9.$-]*) = (.*);", line)
    if assignment is None:
      raise ValueError("generated region body contains a non-tensor operation")
    destination = assignment[1][1:]
    if destination not in tensor_types or destination in {"acc_x", "acc_y", "out"}:
      raise ValueError("generated region writes an undeclared or loop-carried value")
    if destination in definitions:
      raise ValueError(
        "generated region assigns a temporary more than once in its body"
      )
    expression = assignment[2]
    binary = re.fullmatch(
      r"tensor add (%[A-Za-z_][A-Za-z_0-9.$-]*), (%[A-Za-z_][A-Za-z_0-9.$-]*)",
      expression,
    )
    reduction = re.fullmatch(
      r"tensor sum (%[A-Za-z_][A-Za-z_0-9.$-]*), (-?\d+), keep", expression
    )
    broadcast = re.fullmatch(
      r"tensor broadcast (%[A-Za-z_][A-Za-z_0-9.$-]*), \[([^\]]+)\]", expression
    )
    if binary:
      operands = (binary[1][1:], binary[2][1:])
      node = (destination, "add", operands, None)
    elif reduction:
      operands = (reduction[1][1:],)
      node = (destination, "sum", operands, int(reduction[2]))
    elif broadcast:
      operands = (broadcast[1][1:],)
      node = (destination, "broadcast", operands, _shape(broadcast[2]))
    else:
      if expression.startswith("tensor "):
        raise ValueError("unsupported tensor operation in generated region")
      raise ValueError("generated region contains an unsupported assignment")
    definitions[destination] = operands
    nodes.append(node)
  if not nodes:
    raise ValueError("generated region has no operations")

  live = set()
  pending = list(updates.values())
  while pending:
    value = pending.pop()
    if value in live or value not in definitions:
      continue
    live.add(value)
    pending.extend(definitions[value])
  if live != set(definitions):
    raise ValueError(
      "generated region contains a dead operation outside the output union"
    )
  if any(tensor_types.get(root) != input_shape for root in updates.values()):
    raise ValueError("generated region result shapes must match their loop states")

  available = {"acc_x", "acc_y"}
  for destination, operation, operands, attribute in nodes:
    if any(operand not in available for operand in operands):
      raise ValueError("generated region operations must be in dependency order")
    output_shape = tensor_types[destination]
    if operation == "add":
      inferred = _broadcast_shape(*(tensor_types[name] for name in operands))
      if inferred != output_shape:
        raise ValueError("generated region add result has an invalid shape")
    elif operation == "sum":
      source_shape = tensor_types[operands[0]]
      axis = attribute if attribute >= 0 else attribute + len(source_shape)
      if not 0 <= axis < len(source_shape):
        raise ValueError("generated region reduction axis is outside the input rank")
      expected_shape = list(source_shape)
      expected_shape[axis] = 1
      if tuple(expected_shape) != output_shape:
        raise ValueError("generated region keep-dim sum has an invalid result shape")
    else:
      if attribute != output_shape:
        raise ValueError(
          "generated region broadcast target differs from its result shape"
        )
      _broadcast_values(
        (0,) * math.prod(tensor_types[operands[0]]),
        tensor_types[operands[0]],
        output_shape,
      )
    available.add(destination)
  if any(root not in available for root in updates.values()):
    raise ValueError(
      "generated region state roots must be computed before state updates"
    )
  return input_shape, iterations, tensor_types, nodes, updates


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


def _certify_canonical(source, input_bytes, domain):
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


def _certify_generated_region(source, input_bytes, domain):
  shape, iterations, tensor_types, nodes, updates = _generated_region_program(source)
  if (
    not isinstance(domain, (list, tuple))
    or len(domain) != 2
    or any(type(value) is not int for value in domain)
    or not 0 <= domain[0] <= domain[1] <= EXACT_BOUND
  ):
    raise ValueError("exact domain requires nonnegative bounded integers")

  count = math.prod(shape)
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

  values = {
    "x": tuple(map(int, leaves)),
    "acc_x": tuple(map(int, leaves)),
    "acc_y": (0,) * count,
  }
  upper = {
    "x": (domain[1],) * count,
    "acc_x": (domain[1],) * count,
    "acc_y": (0,) * count,
  }
  bounds = []

  def record(destination, operation, actual, maxima, block, visit, *, partial=False):
    if len(actual) != len(maxima) or len(actual) != math.prod(
      tensor_types[destination]
    ):
      raise ValueError("generated region evaluator produced a malformed tensor")
    maximum = max(maxima)
    if maximum > EXACT_BOUND:
      raise ValueError(
        "materialized value or reduction partial sum exceeds exact range"
      )
    values[destination] = actual
    upper[destination] = maxima
    evidence = {
      "block": block,
      "visit": visit,
      "destination": "%" + destination,
      "operation": operation,
      "max_value": maximum,
    }
    if partial:
      evidence["max_partial_sum"] = maximum
    bounds.append(evidence)

  record("x", "input", values["x"], upper["x"], "^entry", 0)
  record("acc_x", "input-copy", values["acc_x"], upper["acc_x"], "^entry", 0)
  record(
    "acc_y",
    "positive-zero",
    values["acc_y"],
    upper["acc_y"],
    "^entry",
    0,
  )

  def add_values(left, right, output_shape):
    broadcast_left = _broadcast_values(values[left], tensor_types[left], output_shape)
    broadcast_right = _broadcast_values(
      values[right], tensor_types[right], output_shape
    )
    upper_left = _broadcast_values(upper[left], tensor_types[left], output_shape)
    upper_right = _broadcast_values(upper[right], tensor_types[right], output_shape)
    return (
      tuple(a + b for a, b in zip(broadcast_left, broadcast_right)),
      tuple(a + b for a, b in zip(upper_left, upper_right)),
    )

  def sum_values(source_name, axis, output_shape):
    source_shape = tensor_types[source_name]
    normalized_axis = axis if axis >= 0 else axis + len(source_shape)
    actual_results = []
    upper_results = []
    for destination_coordinates in _coordinates(output_shape):
      actual_total = 0
      upper_total = 0
      for axis_coordinate in range(source_shape[normalized_axis]):
        source_coordinates = list(destination_coordinates)
        source_coordinates[normalized_axis] = axis_coordinate
        source_index = _flat_index(source_coordinates, source_shape)
        actual_total += values[source_name][source_index]
        upper_total += upper[source_name][source_index]
      actual_results.append(actual_total)
      upper_results.append(upper_total)
    return tuple(actual_results), tuple(upper_results)

  for visit in range(1, iterations + 1):
    for destination, operation, operands, attribute in nodes:
      output_shape = tensor_types[destination]
      if operation == "add":
        actual, maxima = add_values(operands[0], operands[1], output_shape)
      elif operation == "sum":
        actual, maxima = sum_values(operands[0], attribute, output_shape)
      else:
        actual = _broadcast_values(
          values[operands[0]], tensor_types[operands[0]], output_shape
        )
        maxima = _broadcast_values(
          upper[operands[0]], tensor_types[operands[0]], output_shape
        )
      record(
        destination,
        operation,
        actual,
        maxima,
        "^body",
        visit,
        partial=operation == "sum",
      )

    next_states = {
      state: (values[root], upper[root]) for state, root in updates.items()
    }
    for state, (actual, maxima) in next_states.items():
      record(state, "state-update", actual, maxima, "^body", visit)

  output, output_upper = add_values("acc_x", "acc_y", shape)
  record("out", "add", output, output_upper, "^exit", 0)
  proof = {
    "kind": REGION_KIND,
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


def certify(source, input_bytes, domain=(0, 8)):
  """Return recomputed proof evidence and complete independent output bits."""
  try:
    _recognize(source)
  except ValueError:
    return _certify_generated_region(source, input_bytes, domain)
  return _certify_canonical(source, input_bytes, domain)


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
    "solved_file",
    "solved_sha256",
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
    solved = artifacts[selection["solved_file"]]
    model_bytes = artifacts[selection["model_file"]]
    result_bytes = artifacts[selection["result_file"]]
    if hashlib.sha256(query).hexdigest() != selection["query_sha256"]:
      raise ValueError("solver query digest does not match")
    if hashlib.sha256(solved).hexdigest() != selection["solved_sha256"]:
      raise ValueError("solver output digest does not match")
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
    values = model["@kernel"]["%?x"]
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
    ("--main", "@kernel"),
  ):
    try:
      actual = command[command.index(option) + 1]
    except (ValueError, IndexError) as error:
      raise ValueError(f"solver command omits {option}") from error
    if actual != expected_value:
      raise ValueError(f"solver command {option} disagrees with recorded selection")
  for option, artifact_name in (
    ("--output", selection["solved_file"]),
    ("--emit-model", selection["model_file"]),
  ):
    try:
      actual_path = command[command.index(option) + 1]
    except (ValueError, IndexError) as error:
      raise ValueError(f"solver command omits {option}") from error
    if Path(actual_path).name != artifact_name:
      raise ValueError(f"solver command {option} differs from recorded selection")
  if "--emit-tensor-params" in command or "--tensor-param-domain" in command:
    raise ValueError("solver command uses tensor parameter options for a symbol query")
  output = struct.unpack("<4f", expected_bytes)
  if output[0] != selection["target"]["value"]:
    raise ValueError("complete reference output does not satisfy the solver target")
