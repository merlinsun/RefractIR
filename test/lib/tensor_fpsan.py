"""Independent host model for Triton's FpSan payload-ring subset."""

import hashlib
import math
import re
import shutil
import struct
import subprocess
from pathlib import Path

from test.lib import tensor_exact_f32

PAYLOAD_MASK = 0xFFFFFFFF
KNOWN_INPUT_PAYLOADS = (0, 1, PAYLOAD_MASK)
PINNED_TRITON_REVISION = "e2e356723ce35f46b5144373b145c55c4d712237"
PINNED_FPSAN_SOURCE_SHA256 = {
  "python/triton/backends/nvidia/compiler.py": (
    "e4326bac1f3d3a9bf76cff03b66e5c849aaac384562efa6f4caf59c936709ccb"
  ),
  "lib/Dialect/TritonInstrument/Transforms/FpSanitizer.cpp": (
    "55288bde0dfe83d68acd84c284afe929ab17b3f46d268513d39701c11d4e41c8"
  ),
  "lib/Conversion/TritonInstrumentToLLVM/FpSanToLLVM.cpp": (
    "04bc0482782501db8768c520e0afdab64538e1de521ec12c220f9214a2129f33"
  ),
}
_WORD_ZERO = "#x00000000"
_WORD_ONE = "#x00000001"
_WORD_NEG_ONE = "#xFFFFFFFF"


def _sum_left(names):
  result = names[0]
  for name in names[1:]:
    result = f"(bvadd {result} {name})"
  return result


def _sum_balanced(names):
  current = list(names)
  while len(current) > 1:
    next_level = [
      f"(bvadd {current[index]} {current[index + 1]})"
      for index in range(0, len(current) - 1, 2)
    ]
    if len(current) % 2:
      next_level.append(current[-1])
    current = next_level
  return current[0]


def _sum_constraints(output_name, nonzero_names=()):
  return (f"(= {output_name} {_WORD_ONE})",) + tuple(
    f"(not (= {name} {_WORD_ZERO}))" for name in nonzero_names
  )


_PAYLOAD_CASES = {
  "add_tree": {
    "variables": ("a", "b", "c", "d"),
    "reference": ("(bvadd (bvadd a b) (bvadd c d))",),
    "candidate": ("(bvadd (bvadd a c) (bvadd b d))",),
    "constraints": _sum_constraints("reference_0", ("a", "b", "c")),
    "equivalent": True,
  },
  "add_cancellation": {
    "variables": ("a", "b"),
    "reference": ("(bvadd a b)",),
    "candidate": ("(bvadd a b)",),
    "constraints": ("(= reference_0 #x00000000)", "(not (= a #x00000000))"),
    "equivalent": True,
  },
  "multiply_fixed_points": {
    "variables": ("a", "b"),
    "reference": ("(bvmul a b)",),
    "candidate": ("(bvmul a b)",),
    "constraints": (f"(= reference_0 {_WORD_NEG_ONE})",),
    "equivalent": True,
  },
  "subtract_zero_identity": {
    "variables": ("a", "b"),
    "reference": ("(bvsub a b)",),
    "candidate": ("(bvsub a b)",),
    "constraints": (f"(= reference_0 {_WORD_ONE})", "(= b #x00000000)"),
    "equivalent": True,
  },
  "sum_reduction": {
    "variables": ("x0", "x1", "x2", "x3"),
    "reference": (_sum_left(("x0", "x1", "x2", "x3")),),
    "candidate": (_sum_balanced(("x0", "x1", "x2", "x3")),),
    "constraints": _sum_constraints("reference_0", ("x0", "x1", "x2")),
    "equivalent": True,
  },
  "transpose_copy": {
    "variables": ("x0", "x1", "x2", "x3"),
    "reference": ("x0", "x2", "x1", "x3"),
    "candidate": ("x0", "x2", "x1", "x3"),
    "constraints": ("(not (= x0 x2))", "(not (= x1 x3))"),
    "equivalent": True,
  },
  "broadcast_and_add_zero": {
    "variables": ("value",),
    "reference": tuple("(bvadd value #x00000000)" for _ in range(4)),
    "candidate": tuple("(bvadd #x00000000 value)" for _ in range(4)),
    "constraints": (f"(= value {_WORD_NEG_ONE})",),
    "equivalent": True,
  },
  "operand_routing_fault": {
    "variables": ("x0", "x1", "x2", "x3"),
    "reference": ("x0", "x1", "x2", "x3"),
    "candidate": ("x1", "x2", "x3", "x0"),
    "constraints": (
      "(or (not (= reference_0 candidate_0)) "
      "(not (= reference_1 candidate_1)) "
      "(not (= reference_2 candidate_2)) "
      "(not (= reference_3 candidate_3)))",
    ),
    "equivalent": False,
  },
  "lost_update_fault": {
    "variables": ("a", "b"),
    "reference": ("(bvadd a b)",),
    "candidate": ("a",),
    "constraints": (
      "(= reference_0 #x00000000)",
      "(not (= reference_0 candidate_0))",
    ),
    "equivalent": False,
  },
}


def embed_known_f32(value):
  """Return the documented fixed payload for +0, +1, or -1 only."""
  bits = struct.unpack("<I", struct.pack("<f", value))[0]
  fixed_points = {0x00000000: 0, 0x3F800000: 1, 0xBF800000: PAYLOAD_MASK}
  try:
    return fixed_points[bits]
  except KeyError as error:
    raise ValueError(
      "host FpSan model only admits the three documented fixed points"
    ) from error


def unembed_known_payload(payload):
  fixed_points = {0: 0x00000000, 1: 0x3F800000, PAYLOAD_MASK: 0xBF800000}
  try:
    return fixed_points[payload & PAYLOAD_MASK]
  except KeyError as error:
    raise ValueError(
      "host FpSan model only decodes the three documented fixed points"
    ) from error


def add32(left, right):
  return (left + right) & PAYLOAD_MASK


def sub32(left, right):
  return (left - right) & PAYLOAD_MASK


def mul32(left, right):
  return (left * right) & PAYLOAD_MASK


def compare_words(actual, expected):
  """Return differing lanes after validating complete raw-word outputs."""
  if not isinstance(actual, (list, tuple)) or not isinstance(expected, (list, tuple)):
    raise ValueError("raw-word outputs must be sequences")
  if len(actual) != len(expected):
    raise ValueError("raw-word outputs must have equal length")
  for word in (*actual, *expected):
    if type(word) is not int or not 0 <= word <= PAYLOAD_MASK:
      raise ValueError("raw-word outputs must contain unsigned 32-bit words")
  return [
    index for index, (left, right) in enumerate(zip(actual, expected)) if left != right
  ]


def flatten_word_values(values):
  """Flatten scalar words returned by tensor libraries at any output rank."""
  if isinstance(values, (list, tuple)):
    return [word for item in values for word in flatten_word_values(item)]
  if type(values) is not int:
    raise ValueError("raw output words must be integers or nested sequences")
  return [values]


def validate_instrumentation_sensitivity(instrumented, ordinary, ordinary_expected):
  """Require an enabled FpSan run to differ from its verified IEEE control."""
  if any(
    not isinstance(words, (list, tuple)) or not words
    for words in (instrumented, ordinary, ordinary_expected)
  ):
    raise ValueError("instrumentation sensitivity outputs must be non-empty sequences")
  ordinary_mismatches = compare_words(ordinary, ordinary_expected)
  if ordinary_mismatches:
    raise AssertionError("disabled-mode control did not match its expected IEEE result")
  mismatch_lanes = compare_words(instrumented, ordinary)
  if not mismatch_lanes:
    raise AssertionError(
      "FpSan output matched ordinary IEEE execution; instrumentation is inactive"
    )
  return {"status": "pass", "mismatch_lanes": mismatch_lanes}


def validate_fpsan_payload_lowering(llvm_ir):
  """Require integer payload addition and reject residual IEEE arithmetic."""
  if not isinstance(llvm_ir, str) or not llvm_ir.strip():
    raise ValueError("FpSan payload lowering requires non-empty LLVM IR")
  lines = llvm_ir.splitlines()
  integer_operations = [
    line.strip()
    for line in lines
    if re.search(
      r"\badd\s+(?:i\d+|<\d+\s*x\s*i\d+>)|@llvm\.vector\.reduce\.add\b", line
    )
  ]
  floating_operations = [
    line.strip() for line in lines if re.search(r"\bf(?:add|sub|mul|div|rem)\b", line)
  ]
  if floating_operations:
    raise AssertionError(
      "FpSan payload lowering retained floating-point arithmetic: "
      + "; ".join(floating_operations)
    )
  if not integer_operations:
    raise AssertionError("FpSan payload lowering has no integer payload addition")
  return {
    "status": "pass",
    "integer_operation_lines": integer_operations,
    "floating_operation_lines": floating_operations,
  }


def verify_fpsan_source_identity(
  source_root,
  git_revision,
  *,
  expected_revision=PINNED_TRITON_REVISION,
  expected_sources=None,
):
  """Verify the pinned Triton revision and hashes of its FpSan implementation."""
  source_root = Path(source_root).resolve()
  source_manifest = (
    PINNED_FPSAN_SOURCE_SHA256 if expected_sources is None else expected_sources
  )
  if not re.fullmatch(r"[0-9a-f]{40}", git_revision):
    raise ValueError("Triton git revision must be a full lowercase commit hash")
  if git_revision != expected_revision:
    raise RuntimeError(
      f"expected pinned Triton revision {expected_revision}, found {git_revision}"
    )
  if not source_manifest:
    raise ValueError("FpSan source manifest must not be empty")

  actual_hashes = {}
  for relative_path, expected_hash in source_manifest.items():
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
      raise ValueError("FpSan source manifest paths must stay inside the source root")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
      raise ValueError(f"invalid SHA-256 in FpSan source manifest: {relative_path}")
    source_path = (source_root / path).resolve()
    try:
      source_path.relative_to(source_root)
    except ValueError as error:
      raise ValueError(
        "FpSan source manifest paths must stay inside the source root"
      ) from error
    try:
      contents = source_path.read_bytes()
    except OSError as error:
      raise RuntimeError(f"FpSan source file is missing: {relative_path}") from error
    actual_hash = hashlib.sha256(contents).hexdigest()
    if actual_hash != expected_hash:
      raise RuntimeError(
        f"FpSan source hash mismatch for {relative_path}: {actual_hash}"
      )
    actual_hashes[relative_path] = actual_hash

  return {
    "git_commit": git_revision,
    "source_sha256": actual_hashes,
  }


def verify_installed_fpsan_source(triton_module):
  """Locate Triton's source checkout and verify the FpSan implementation pin."""
  module_file = Path(triton_module.__file__).resolve()
  source_root = next(
    (
      parent
      for parent in module_file.parents
      if (parent / ".git").exists()
      and (parent / "python/triton/__init__.py").resolve() == module_file
    ),
    None,
  )
  if source_root is None:
    raise RuntimeError("installed Triton source checkout cannot be identified")
  result = subprocess.run(
    ["git", "-C", str(source_root), "rev-parse", "HEAD"],
    capture_output=True,
    text=True,
    check=False,
  )
  if result.returncode != 0:
    raise RuntimeError(f"cannot read Triton source revision: {result.stderr.strip()}")
  identity = verify_fpsan_source_identity(source_root, result.stdout.strip())
  identity["source_root"] = str(source_root)
  return identity


def _payload_inputs(name, inputs):
  variables = _PAYLOAD_CASES[name]["variables"]
  if not isinstance(inputs, dict) or set(inputs) != set(variables):
    raise ValueError(f"{name} inputs must define exactly {variables}")
  if any(
    type(inputs[variable]) is not int or not 0 <= inputs[variable] <= PAYLOAD_MASK
    for variable in variables
  ):
    raise ValueError("payload inputs must be unsigned 32-bit words")
  return [inputs[variable] for variable in variables]


def evaluate_payload_case(name, inputs):
  """Evaluate one named reference/kernel pair without using SMT or Triton."""
  values = _payload_inputs(name, inputs)
  if name == "add_tree":
    a, b, c, d = values
    reference = [add32(add32(a, b), add32(c, d))]
    candidate = [add32(add32(a, c), add32(b, d))]
  elif name == "add_cancellation":
    reference = candidate = [add32(*values)]
  elif name == "multiply_fixed_points":
    reference = candidate = [mul32(*values)]
  elif name == "subtract_zero_identity":
    reference = candidate = [sub32(*values)]
  elif name == "sum_reduction":
    reference = [reduce_tree(values, "left")]
    candidate = [reduce_tree(values, "balanced")]
  elif name == "transpose_copy":
    reference = candidate = transpose_2x2(values)
  elif name == "broadcast_and_add_zero":
    reference = [add32(0, values[0])] * 4
    candidate = [add32(values[0], 0)] * 4
  elif name == "operand_routing_fault":
    reference = list(values)
    candidate = list(values[1:]) + list(values[:1])
  elif name == "lost_update_fault":
    reference = [add32(*values)]
    candidate = [values[0]]
  else:
    raise ValueError(f"unknown FpSan payload case: {name}")
  return {"reference": reference, "candidate": candidate}


def _generated_payload_inputs(shape, inputs):
  count = math.prod(shape)
  if not isinstance(inputs, (tuple, list)) or len(inputs) != count:
    raise ValueError(f"generated region needs {count} payload input words")
  if any(type(value) is not int or not 0 <= value <= PAYLOAD_MASK for value in inputs):
    raise ValueError("payload inputs must be unsigned 32-bit words")
  return tuple(inputs)


def _payload_reduce(values, input_shape, output_shape, axis):
  axis = axis if axis >= 0 else axis + len(input_shape)
  reduced = []
  for output_coordinates in tensor_exact_f32._coordinates(output_shape):
    total = 0
    for axis_coordinate in range(input_shape[axis]):
      coordinates = list(output_coordinates)
      coordinates[axis] = axis_coordinate
      index = tensor_exact_f32._flat_index(coordinates, input_shape)
      total = add32(total, values[index])
    reduced.append(total)
  return tuple(reduced)


def evaluate_generated_payload_region(source, input_payloads):
  """Evaluate an admitted generated loop as 32-bit FpSan payload arithmetic."""
  shape, iterations, tensor_types, nodes, updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  initial = _generated_payload_inputs(shape, input_payloads)
  count = math.prod(shape)
  values = {
    "x": initial,
    "acc_x": initial,
    "acc_y": (0,) * count,
  }

  for _ in range(iterations):
    for destination, operation, operands, attribute in nodes:
      output_shape = tensor_types[destination]
      if operation == "add":
        left = tensor_exact_f32._broadcast_values(
          values[operands[0]], tensor_types[operands[0]], output_shape
        )
        right = tensor_exact_f32._broadcast_values(
          values[operands[1]], tensor_types[operands[1]], output_shape
        )
        result = tuple(add32(a, b) for a, b in zip(left, right))
      elif operation == "sum":
        result = _payload_reduce(
          values[operands[0]],
          tensor_types[operands[0]],
          output_shape,
          attribute,
        )
      else:
        result = tensor_exact_f32._broadcast_values(
          values[operands[0]], tensor_types[operands[0]], output_shape
        )
      values[destination] = result

    for state, root in updates:
      values[state] = values[root]

  left = tensor_exact_f32._broadcast_values(values["acc_x"], shape, shape)
  right = tensor_exact_f32._broadcast_values(values["acc_y"], shape, shape)
  return [add32(a, b) for a, b in zip(left, right)]


def _generated_payload_smt(source):
  shape, iterations, tensor_types, nodes, updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  count = math.prod(shape)
  variables = tuple(f"x{index}" for index in range(count))
  definitions = []

  def define(name, expression):
    definitions.append(f"(define-fun {name} () (_ BitVec 32) {expression})")

  values = {
    "x": list(variables),
    "acc_x": list(variables),
    "acc_y": ["#x00000000"] * count,
  }
  for visit in range(iterations):
    for node_index, (destination, operation, operands, attribute) in enumerate(nodes):
      output_shape = tensor_types[destination]
      if operation == "add":
        left = tensor_exact_f32._broadcast_values(
          values[operands[0]], tensor_types[operands[0]], output_shape
        )
        right = tensor_exact_f32._broadcast_values(
          values[operands[1]], tensor_types[operands[1]], output_shape
        )
        expressions = [f"(bvadd {a} {b})" for a, b in zip(left, right)]
      elif operation == "sum":
        input_shape = tensor_types[operands[0]]
        axis = attribute if attribute >= 0 else attribute + len(input_shape)
        expressions = []
        for output_coordinates in tensor_exact_f32._coordinates(output_shape):
          terms = []
          for axis_coordinate in range(input_shape[axis]):
            coordinates = list(output_coordinates)
            coordinates[axis] = axis_coordinate
            index = tensor_exact_f32._flat_index(coordinates, input_shape)
            terms.append(values[operands[0]][index])
          expressions.append(_sum_left(terms))
      else:
        expressions = list(
          tensor_exact_f32._broadcast_values(
            values[operands[0]], tensor_types[operands[0]], output_shape
          )
        )

      names = []
      for element_index, expression in enumerate(expressions):
        name = f"g_v{visit}_n{node_index}_e{element_index}"
        define(name, expression)
        names.append(name)
      values[destination] = names

    for state, root in updates:
      values[state] = values[root]

  output_names = tuple(f"output_{index}" for index in range(count))
  for name, left, right in zip(output_names, values["acc_x"], values["acc_y"]):
    define(name, f"(bvadd {left} {right})")
  return shape, iterations, variables, output_names, definitions


def build_generated_payload_query(source):
  """Encode an admitted generated loop and fixed-point I/O in QF_BV."""
  shape, iterations, variables, output_names, definitions = _generated_payload_smt(
    source
  )
  lines = ["(set-logic QF_BV)", "(set-option :produce-models true)"]
  lines.extend(f"(declare-const {name} (_ BitVec 32))" for name in variables)
  for name in variables:
    choices = " ".join(
      f"(= {name} #x{payload:08X})" for payload in KNOWN_INPUT_PAYLOADS
    )
    lines.append(f"(assert (or {choices}))")
  lines.extend(definitions)
  for name in output_names:
    choices = " ".join(
      f"(= {name} #x{payload:08X})" for payload in KNOWN_INPUT_PAYLOADS
    )
    lines.append(f"(assert (or {choices}))")
  nonzero_inputs = " ".join(f"(not (= {name} #x00000000))" for name in variables)
  lines.append(f"(assert (or {nonzero_inputs}))")
  lines.append("(check-sat)")
  requested = [*variables, *output_names]
  lines.append("(get-value (" + " ".join(requested) + "))")
  script = "\n".join(lines) + "\n"
  return {
    "shape": shape,
    "iterations": iterations,
    "variables": variables,
    "output_names": output_names,
    "script": script,
    "sha256": hashlib.sha256(script.encode("ascii")).hexdigest(),
  }


def synthesize_generated_region(source):
  """Solve one generated-region query and replay it with the host payload model."""
  query = build_generated_payload_query(source)
  command = _solver_command()
  result = subprocess.run(
    command, input=query["script"], text=True, capture_output=True, check=False
  )
  if result.returncode != 0:
    raise RuntimeError(
      f"{command[0]} failed ({result.returncode}): {result.stderr.strip()}"
    )
  requested = [*query["variables"], *query["output_names"]]
  model = parse_payload_model(result.stdout, requested)
  input_payloads = [model[name] for name in query["variables"]]
  output_payloads = [model[name] for name in query["output_names"]]
  replayed = evaluate_generated_payload_region(source, input_payloads)
  if output_payloads != replayed:
    raise AssertionError("SMT model outputs disagree with the host payload evaluator")
  try:
    input_words = [unembed_known_payload(value) for value in input_payloads]
    output_words = [unembed_known_payload(value) for value in output_payloads]
  except ValueError as error:
    raise AssertionError(
      "generated region witness cannot be transported through fixed f32 points"
    ) from error
  return {
    "query_sha256": query["sha256"],
    "query_smt2": query["script"],
    "shape": list(query["shape"]),
    "iterations": query["iterations"],
    "input_payloads": dict(zip(query["variables"], input_payloads)),
    "output_payloads": output_payloads,
    "input_float_words": input_words,
    "output_float_words": output_words,
    "input_float_bits": [f"{value:08x}" for value in input_words],
    "output_float_bits": [f"{value:08x}" for value in output_words],
  }


def render_generated_region_triton_kernel(source):
  """Emit a Triton kernel with the same dataflow and ordered updates as SIR."""
  shape, iterations, tensor_types, nodes, updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  if len(shape) != 2:
    raise ValueError("FpSan Triton lowering requires a rank-two generated region")
  rows, columns = shape
  lines = [
    "import triton",
    "import triton.language as tl",
    "",
    "@triton.jit",
    "def fpsan_generated_region(input_ptr, output_ptr):",
    f"    rows = tl.arange(0, {rows})",
    f"    columns = tl.arange(0, {columns})",
    f"    offsets = rows[:, None] * {columns} + columns[None, :]",
    "    acc_x = tl.load(input_ptr + offsets)",
    f"    acc_y = tl.full(({rows}, {columns}), 0.0, tl.float32)",
    f"    for _ in tl.static_range({iterations}):",
  ]
  names = {"acc_x": "acc_x", "acc_y": "acc_y"}
  for index, (destination, _operation, _operands, _attribute) in enumerate(nodes):
    names[destination] = f"value_{index}"

  for destination, operation, operands, attribute in nodes:
    target = names[destination]
    if operation == "add":
      left, right = (names[name] for name in operands)
      expression = f"{left} + {right}"
    elif operation == "sum":
      axis = attribute if attribute >= 0 else attribute + len(shape)
      operand = names[operands[0]]
      lines.append(f"        {target}_reduced = tl.sum({operand}, axis={axis})")
      expression = f"tl.expand_dims({target}_reduced, {axis})"
    else:
      operand = names[operands[0]]
      target_shape = tensor_types[destination]
      expression = f"tl.broadcast_to({operand}, {target_shape!r})"
    lines.append(f"        {target} = {expression}")

  for state, root in updates:
    lines.append(f"        {state} = {names[root]}")
  lines.extend(
    (
      "    output = acc_x + acc_y",
      "    tl.store(output_ptr + offsets, output)",
    )
  )
  return "\n".join(lines) + "\n"


def build_payload_query(case_name):
  """Build one auditable QF_BV query for a named instrumented kernel."""
  try:
    spec = _PAYLOAD_CASES[case_name]
  except KeyError as error:
    raise ValueError(f"unknown FpSan payload case: {case_name}") from error

  variables = spec["variables"]
  references = spec["reference"]
  candidates = spec["candidate"]
  lines = ["(set-logic QF_BV)", "(set-option :produce-models true)"]
  lines.extend(f"(declare-const {name} (_ BitVec 32))" for name in variables)
  for name in variables:
    allowed = " ".join(
      f"(= {name} #x{payload:08X})" for payload in KNOWN_INPUT_PAYLOADS
    )
    lines.append(f"(assert (or {allowed}))")
  for prefix, expressions in (("reference", references), ("candidate", candidates)):
    lines.extend(
      f"(define-fun {prefix}_{index} () (_ BitVec 32) {expression})"
      for index, expression in enumerate(expressions)
    )
  lines.extend(f"(assert {constraint})" for constraint in spec["constraints"])
  lines.append("(check-sat)")
  requested = list(variables)
  requested.extend(f"reference_{index}" for index in range(len(references)))
  requested.extend(f"candidate_{index}" for index in range(len(candidates)))
  lines.append("(get-value (" + " ".join(requested) + "))")
  script = "\n".join(lines) + "\n"
  return {
    "case": case_name,
    "variables": variables,
    "reference_expressions": references,
    "candidate_expressions": candidates,
    "constraints": spec["constraints"],
    "script": script,
    "sha256": hashlib.sha256(script.encode("ascii")).hexdigest(),
  }


def parse_payload_model(output, requested_names):
  """Parse solver get-value words without converting them through float."""
  tokens = output.split()
  if not tokens or tokens[0] != "sat":
    raise ValueError("payload synthesis solver did not return sat")
  values = {}
  pattern = re.compile(
    r"\(\s*([A-Za-z_][A-Za-z_0-9]*)\s+"
    r"(#x[0-9a-fA-F]+|#b[01]+|\(_\s*bv\d+\s+32\s*\))\s*\)"
  )
  for match in pattern.finditer(output):
    name, raw = match.groups()
    if raw.startswith("#x"):
      digits = raw[2:]
      if len(digits) != 8:
        raise ValueError("payload model hexadecimal words must be 32 bits")
      value = int(digits, 16)
    elif raw.startswith("#b"):
      digits = raw[2:]
      if len(digits) != 32:
        raise ValueError("payload model binary words must be 32 bits")
      value = int(digits, 2)
    else:
      value = int(re.search(r"bv(\d+)", raw)[1])
    if name in values:
      raise ValueError(f"payload solver repeated model value for {name}")
    values[name] = value
  requested_names = tuple(requested_names)
  if set(values) != set(requested_names):
    raise ValueError("payload solver model is missing or added requested words")
  return values


def _solver_command():
  bitwuzla = shutil.which("bitwuzla")
  if bitwuzla is not None:
    return [bitwuzla]
  z3 = shutil.which("z3")
  if z3 is not None:
    return [z3, "-in"]
  raise RuntimeError(
    "bitwuzla or z3 executable is required for FpSan bit-vector checks"
  )


def synthesize_payload_case(case_name):
  """Solve a kernel-specific query and cross-check it with the host evaluator."""
  query = build_payload_query(case_name)
  command = _solver_command()
  result = subprocess.run(
    command, input=query["script"], text=True, capture_output=True, check=False
  )
  if result.returncode != 0:
    raise RuntimeError(
      f"{command[0]} failed ({result.returncode}): {result.stderr.strip()}"
    )
  requested = list(query["variables"])
  requested.extend(
    f"reference_{index}" for index in range(len(query["reference_expressions"]))
  )
  requested.extend(
    f"candidate_{index}" for index in range(len(query["candidate_expressions"]))
  )
  model = parse_payload_model(result.stdout, requested)
  inputs = {name: model[name] for name in query["variables"]}
  evaluated = evaluate_payload_case(case_name, inputs)
  reference = [
    model[f"reference_{index}"] for index in range(len(query["reference_expressions"]))
  ]
  candidate = [
    model[f"candidate_{index}"] for index in range(len(query["candidate_expressions"]))
  ]
  if reference != evaluated["reference"] or candidate != evaluated["candidate"]:
    raise AssertionError(
      "SMT model outputs disagree with the independent host evaluator"
    )
  try:
    input_words = [unembed_known_payload(inputs[name]) for name in query["variables"]]
    reference_words = [unembed_known_payload(value) for value in reference]
    candidate_words = [unembed_known_payload(value) for value in candidate]
  except ValueError as error:
    raise AssertionError(
      "selected payload cannot be transported as a pinned f32 word"
    ) from error
  return {
    "case": case_name,
    "solver": command[0],
    "query_sha256": query["sha256"],
    "query_smt2": query["script"],
    "variables": list(query["variables"]),
    "reference_equations": list(query["reference_expressions"]),
    "candidate_equations": list(query["candidate_expressions"]),
    "selection_constraints": list(query["constraints"]),
    "input_payloads": inputs,
    "input_float_words": input_words,
    "input_float_bits": [f"{value:08x}" for value in input_words],
    "reference_payloads": reference,
    "candidate_payloads": candidate,
    "reference_float_words": reference_words,
    "candidate_float_words": candidate_words,
    "equivalent_outputs": reference == candidate,
  }


def prove_payload_kernel_equations():
  """Prove all non-mutant reference/candidate equations by QF_BV exhaustion."""
  command = _solver_command()
  proven = []
  for case_name, spec in _PAYLOAD_CASES.items():
    if not spec["equivalent"]:
      continue
    lines = ["(set-logic QF_BV)"]
    lines.extend(f"(declare-const {name} (_ BitVec 32))" for name in spec["variables"])
    for prefix, expressions in (
      ("reference", spec["reference"]),
      ("candidate", spec["candidate"]),
    ):
      lines.extend(
        f"(define-fun {prefix}_{index} () (_ BitVec 32) {expression})"
        for index, expression in enumerate(expressions)
      )
    differences = [
      f"(not (= reference_{index} candidate_{index}))"
      for index in range(len(spec["reference"]))
    ]
    mismatch = (
      differences[0] if len(differences) == 1 else "(or " + " ".join(differences) + ")"
    )
    lines.extend((f"(assert {mismatch})", "(check-sat)"))
    script = "\n".join(lines) + "\n"
    result = subprocess.run(
      command, input=script, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
      raise RuntimeError(
        f"{command[0]} failed ({result.returncode}): {result.stderr.strip()}"
      )
    if result.stdout.split() != ["unsat"]:
      raise AssertionError(
        f"{case_name} reference/candidate equation was not proved: "
        f"{result.stdout.split()}"
      )
    proven.append(
      {
        "case": case_name,
        "status": "unsat",
        "query_sha256": hashlib.sha256(script.encode("ascii")).hexdigest(),
      }
    )
  return proven


def _round_f32(value):
  return struct.unpack("<f", struct.pack("<f", value))[0]


def _f32_word(value):
  return struct.unpack("<I", struct.pack("<f", value))[0]


def rounding_reassociation_bits():
  """Return the two IEEE-f32 results for (1 + 2^-24) - 1 and 1 + (2^-24 - 1)."""
  a, b, c = 1.0, 2.0**-24, -1.0
  left_associated = _round_f32(_round_f32(a + b) + c)
  right_associated = _round_f32(a + _round_f32(b + c))
  return _f32_word(left_associated), _f32_word(right_associated)


def f32_fmod_bits(dividend, divisor):
  """Evaluate f32 fmod on materialized binary32 inputs and return its bits."""
  dividend = _round_f32(dividend)
  divisor = _round_f32(divisor)
  if not math.isfinite(dividend) or not math.isfinite(divisor) or divisor == 0.0:
    raise ValueError("f32 fmod control requires finite operands and nonzero divisor")
  return _f32_word(_round_f32(math.fmod(dividend, divisor)))


def reduce_tree(values, tree):
  if not values:
    return 0
  if tree == "left":
    result = values[0]
    for value in values[1:]:
      result = add32(result, value)
    return result
  if tree != "balanced":
    raise ValueError(f"unknown reduction tree: {tree}")
  current = list(values)
  while len(current) > 1:
    next_level = [
      add32(current[index], current[index + 1])
      for index in range(0, len(current) - 1, 2)
    ]
    if len(current) % 2:
      next_level.append(current[-1])
    current = next_level
  return current[0]


def transpose_2x2(values):
  if len(values) != 4:
    raise ValueError("2x2 transpose needs four row-major payloads")
  return [values[0], values[2], values[1], values[3]]


def broadcast_scalar(value, count):
  if count < 0:
    raise ValueError("broadcast count must be nonnegative")
  return [value] * count


def prove_payload_laws():
  bitwuzla = shutil.which("bitwuzla")
  z3 = shutil.which("z3")
  if bitwuzla is not None:
    command = [bitwuzla]
  elif z3 is not None:
    command = [z3, "-in"]
  else:
    raise RuntimeError(
      "bitwuzla or z3 executable is required for the FpSan bit-vector checks"
    )
  queries = [
    "(= (bvadd (bvadd a b) c) (bvadd a (bvadd b c)))",
    "(= (bvadd a b) (bvadd b a))",
    "(= (bvsub a a) #x00000000)",
    "(= (bvsub a b) (bvadd a (bvneg b)))",
    "(= (bvmul a (bvadd b c)) (bvadd (bvmul a b) (bvmul a c)))",
    "(= (bvadd (bvadd a b) (bvadd c d)) (bvadd (bvadd a c) (bvadd b d)))",
  ]
  declarations = "\n".join(
    f"(declare-const {name} (_ BitVec 32))" for name in ("a", "b", "c", "d")
  )
  failures = " ".join(f"(not {query})" for query in queries)
  script = (
    "(set-logic QF_BV)\n" + declarations + f"\n(assert (or {failures}))\n(check-sat)\n"
  )
  result = subprocess.run(
    command, input=script, text=True, capture_output=True, check=False
  )
  if result.returncode != 0:
    raise RuntimeError(
      f"{command[0]} failed ({result.returncode}): {result.stderr.strip()}"
    )
  outcomes = result.stdout.split()
  if outcomes != ["unsat"]:
    raise AssertionError(f"payload ring law queries were not all UNSAT: {outcomes}")
  return True
