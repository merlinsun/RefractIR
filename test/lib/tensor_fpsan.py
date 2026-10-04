"""Independent host model for Triton's FpSan payload-ring subset."""

import hashlib
import math
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path

from test.lib import tensor_exact_f32

PAYLOAD_MASK = 0xFFFFFFFF
KNOWN_INPUT_PAYLOADS = (0, 1, PAYLOAD_MASK)
GENERATED_INPUT_MAX = 8
GENERATED_OUTPUT_MAX = 1 << 24
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
_FPSAN_F32_SIGN_MASK = 0x80000000
_FPSAN_F32_MAGNITUDE_MASK = 0x7FFFFFFF
_FPSAN_F32_SHIFT = 23
_FPSAN_F32_MUL_A = 0x00E3F789
_FPSAN_F32_MUL_A_INVERSE = 0x6D2BAEB9
_FPSAN_F32_MUL_B_POSITIVE = 0x0BEAC0C7
_FPSAN_F32_MUL_B_NEGATIVE = 0x74153F39
_FPSAN_F32_MUL_B_POSITIVE_INVERSE = 0x7B8000F7
_FPSAN_F32_MUL_B_NEGATIVE_INVERSE = 0x047FFF09
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


def _evaluate_generated_payload_region(
  source, input_payloads, *, iteration_count=None, omitted_updates=()
):
  shape, iterations, tensor_types, nodes, updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  initial = _generated_payload_inputs(shape, input_payloads)
  if iteration_count is None:
    iteration_count = iterations
  if (
    type(iteration_count) is not int
    or iteration_count < 0
    or iteration_count > iterations
  ):
    raise ValueError("payload iteration count must be within the source loop bound")
  omitted_updates = tuple(omitted_updates)
  update_names = {state for state, _root in updates}
  if any(name not in update_names for name in omitted_updates):
    raise ValueError("payload mutant names an unknown loop-carried state")
  count = math.prod(shape)
  values = {
    "x": initial,
    "acc_x": initial,
    "acc_y": (0,) * count,
  }

  for _ in range(iteration_count):
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
      elif operation == "mul":
        left = tensor_exact_f32._broadcast_values(
          values[operands[0]], tensor_types[operands[0]], output_shape
        )
        right = tensor_exact_f32._broadcast_values(
          values[operands[1]], tensor_types[operands[1]], output_shape
        )
        result = tuple((a * b) & PAYLOAD_MASK for a, b in zip(left, right))
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
      if state not in omitted_updates:
        values[state] = values[root]

  left = tensor_exact_f32._broadcast_values(values["acc_x"], shape, shape)
  right = tensor_exact_f32._broadcast_values(values["acc_y"], shape, shape)
  return [add32(a, b) for a, b in zip(left, right)]


def evaluate_generated_payload_region(source, input_payloads):
  """Evaluate an admitted generated loop as 32-bit FpSan payload arithmetic."""
  return _evaluate_generated_payload_region(source, input_payloads)


def evaluate_generated_payload_mutant(source, input_payloads, mutant):
  """Evaluate one generated-loop omission used to calibrate witness quality."""
  _shape, iterations, _types, _nodes, _updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  if mutant == "lost_iteration":
    return _evaluate_generated_payload_region(
      source, input_payloads, iteration_count=iterations - 1
    )
  if mutant == "lost_update_acc_y":
    return _evaluate_generated_payload_region(
      source, input_payloads, omitted_updates=("acc_y",)
    )
  raise ValueError(f"unknown generated-region calibration mutant: {mutant}")


def _generated_payload_smt(
  source, *, iteration_count=None, omitted_updates=(), prefix=""
):
  shape, iterations, tensor_types, nodes, updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  if iteration_count is None:
    iteration_count = iterations
  if (
    type(iteration_count) is not int
    or iteration_count < 0
    or iteration_count > iterations
  ):
    raise ValueError("payload iteration count must be within the source loop bound")
  omitted_updates = tuple(omitted_updates)
  update_names = {state for state, _root in updates}
  if any(name not in update_names for name in omitted_updates):
    raise ValueError("payload mutant names an unknown loop-carried state")
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
  for visit in range(iteration_count):
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
      elif operation == "mul":
        left = tensor_exact_f32._broadcast_values(
          values[operands[0]], tensor_types[operands[0]], output_shape
        )
        right = tensor_exact_f32._broadcast_values(
          values[operands[1]], tensor_types[operands[1]], output_shape
        )
        expressions = [f"(bvmul {a} {b})" for a, b in zip(left, right)]
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
        name = f"{prefix}g_v{visit}_n{node_index}_e{element_index}"
        define(name, expression)
        names.append(name)
      values[destination] = names

    for state, root in updates:
      if state not in omitted_updates:
        values[state] = values[root]

  output_names = tuple(f"{prefix}output_{index}" for index in range(count))
  for name, left, right in zip(output_names, values["acc_x"], values["acc_y"]):
    define(name, f"(bvadd {left} {right})")
  return shape, iterations, variables, output_names, definitions


def _fpsan_unmix_payload_smt():
  """SMT bit-vector inverse of the pinned Triton f32 payload mixer."""
  return (
    "(define-fun fpsan_unmix_payload ((payload (_ BitVec 32))) (_ BitVec 32)\n"
    "  (let ((sign (bvand payload #x80000000)))\n"
    "    (let ((negative (not (= sign #x00000000))))\n"
    "      (let ((multiplier_inverse "
    "(ite negative #x047FFF09 #x7B8000F7)))\n"
    "        (let ((unshifted (bvand (bvmul "
    "(bvand payload #x7FFFFFFF) multiplier_inverse) #x7FFFFFFF)))\n"
    "          (let ((unmixed (bvxor unshifted "
    "(bvlshr unshifted #x00000017))))\n"
    "            (bvor (bvand (bvmul unmixed #x6D2BAEB9) "
    "#x7FFFFFFF) sign)))))))\n"
  )


def build_generated_payload_query(source, *, target=None, target_index=(0, 0)):
  """Encode transportable loop I/O and mutation-sensitive witnesses in QF_BV."""
  shape, iterations, variables, output_names, definitions = _generated_payload_smt(
    source
  )
  property_spec = None
  if target is not None:
    if type(target) is not int or not 0 <= target <= GENERATED_OUTPUT_MAX:
      raise ValueError("generated output payload target is outside the bounded range")
    if (
      not isinstance(target_index, (tuple, list))
      or len(target_index) != len(shape)
      or any(
        type(value) is not int or not 0 <= value < shape[axis]
        for axis, value in enumerate(target_index)
      )
    ):
      raise ValueError("generated output target index is outside the output shape")
    target_index = tuple(target_index)
    flat_index = tensor_exact_f32._flat_index(target_index, shape)
    property_spec = {
      "kind": "output-payload-equals",
      "target": target,
      "index": list(target_index),
    }
  _, _, _, lost_iteration_names, lost_iteration_definitions = _generated_payload_smt(
    source,
    iteration_count=iterations - 1,
    prefix="lost_iteration_",
  )
  _, _, _, lost_update_names, lost_update_definitions = _generated_payload_smt(
    source,
    omitted_updates=("acc_y",),
    prefix="lost_update_acc_y_",
  )
  count = math.prod(shape)
  input_transport_words = tuple(
    unembed_generated_payload(value) for value in range(GENERATED_INPUT_MAX + 1)
  )
  bound_input = struct.pack(f"<{count}f", *([float(GENERATED_INPUT_MAX)] * count))
  tensor_exact_f32.certify(source, bound_input, (0, GENERATED_INPUT_MAX))

  lines = ["(set-logic QF_BV)", "(set-option :produce-models true)"]
  lines.append(_fpsan_unmix_payload_smt())
  lines.extend(f"(declare-const {name} (_ BitVec 32))" for name in variables)
  for name in variables:
    lines.append(f"(assert (bvule {name} #x{GENERATED_INPUT_MAX:08X}))")
  lines.extend(definitions)
  lines.extend(lost_iteration_definitions)
  lines.extend(lost_update_definitions)
  all_outputs = (*output_names, *lost_iteration_names, *lost_update_names)
  lines.extend(
    f"(assert (bvule {name} #x{GENERATED_OUTPUT_MAX:08X}))" for name in all_outputs
  )
  lines.extend(
    f"(assert (not (= (bvand (fpsan_unmix_payload {name}) #x7F800000) #x7F800000)))"
    for name in all_outputs
  )
  lines.append(
    "(assert (or "
    + " ".join(
      f"(not (= {output} {input_name}))"
      for output, input_name in zip(output_names, variables)
    )
    + "))"
  )
  for mutant_names in (lost_iteration_names, lost_update_names):
    lines.append(
      "(assert (or "
      + " ".join(
        f"(not (= {output} {mutant}))"
        for output, mutant in zip(output_names, mutant_names)
      )
      + "))"
    )
  if property_spec is not None:
    lines.append(f"(assert (= {output_names[flat_index]} #x{target:08X}))")
  lines.append("(check-sat)")
  requested = [*variables, *output_names]
  check_script = "\n".join(lines) + "\n"
  script = check_script + "(get-value (" + " ".join(requested) + "))\n"
  return {
    "shape": shape,
    "iterations": iterations,
    "variables": variables,
    "output_names": output_names,
    "input_transport_words": input_transport_words,
    "transport_contract": "triton-fpsan-f32-mixer-e2e356723ce35f46b5144373b145c55c4d712237",
    "mutant_output_names": {
      "lost_iteration": lost_iteration_names,
      "lost_update_acc_y": lost_update_names,
    },
    "property": property_spec,
    "check_script": check_script,
    "script": script,
    "sha256": hashlib.sha256(script.encode("ascii")).hexdigest(),
  }


def _solver_status(output):
  for line in output.splitlines():
    match = re.match(r"^\s*(sat|unsat|unknown)\b", line, re.IGNORECASE)
    if match:
      return match[1].upper()
  return "ERROR"


def _solver_outcome(status, query, command, phase, result=None, error=None):
  output = getattr(result, "stdout", "") if result is not None else ""
  stderr = getattr(result, "stderr", "") if result is not None else ""
  if isinstance(output, bytes):
    output = output.decode("utf-8", errors="replace")
  if isinstance(stderr, bytes):
    stderr = stderr.decode("utf-8", errors="replace")
  return {
    "status": status,
    "solver": command[0],
    "transport_contract": query["transport_contract"],
    "query_sha256": query["sha256"],
    "query_smt2": query["script"],
    "shape": list(query["shape"]),
    "iterations": query["iterations"],
    "property": query["property"],
    "solver_outcome": {
      "phase": phase,
      "stdout": output,
      "stderr": stderr,
      "returncode": getattr(result, "returncode", None),
      "error": str(error) if error is not None else None,
    },
  }


def embed_generated_float_word(word):
  """Apply the pinned Triton f32 payload mixer to one raw IEEE word."""
  if type(word) is not int or not 0 <= word <= PAYLOAD_MASK:
    raise ValueError("generated f32 transport word must be an unsigned 32-bit integer")
  sign = word & _FPSAN_F32_SIGN_MASK
  magnitude = word & _FPSAN_F32_MAGNITUDE_MASK
  mixed = (magnitude * _FPSAN_F32_MUL_A) & _FPSAN_F32_MAGNITUDE_MASK
  mixed ^= mixed >> _FPSAN_F32_SHIFT
  multiplier = _FPSAN_F32_MUL_B_NEGATIVE if sign else _FPSAN_F32_MUL_B_POSITIVE
  return ((mixed * multiplier) & _FPSAN_F32_MAGNITUDE_MASK) | sign


def unembed_generated_payload(payload):
  """Encode one bounded payload as a finite f32 word using pinned FpSan mixing."""
  if type(payload) is not int or not 0 <= payload <= GENERATED_OUTPUT_MAX:
    raise ValueError("generated payload is outside the bounded transport range")
  sign = payload & _FPSAN_F32_SIGN_MASK
  magnitude = payload & _FPSAN_F32_MAGNITUDE_MASK
  multiplier_inverse = (
    _FPSAN_F32_MUL_B_NEGATIVE_INVERSE if sign else _FPSAN_F32_MUL_B_POSITIVE_INVERSE
  )
  unshifted = (magnitude * multiplier_inverse) & _FPSAN_F32_MAGNITUDE_MASK
  unmixed = unshifted
  shift = _FPSAN_F32_SHIFT
  while shift < 32:
    unmixed ^= unshifted >> shift
    shift *= 2
  raw_word = ((unmixed * _FPSAN_F32_MUL_A_INVERSE) & _FPSAN_F32_MAGNITUDE_MASK) | sign
  if raw_word & 0x7F800000 == 0x7F800000:
    raise ValueError("generated payload does not map to a finite f32 transport word")
  return raw_word


def synthesize_generated_region(
  source, *, target=None, target_index=(0, 0), timeout_s=30.0
):
  """Solve a bounded payload query and retain each solver terminal outcome."""
  if (
    isinstance(timeout_s, bool)
    or not isinstance(timeout_s, (int, float))
    or not math.isfinite(timeout_s)
    or timeout_s <= 0
  ):
    raise ValueError("generated payload solver timeout must be positive and finite")
  started = time.perf_counter()
  deadline = time.monotonic() + timeout_s
  query_started = time.perf_counter()
  query = build_generated_payload_query(
    source, target=target, target_index=target_index
  )
  costs = {
    "query_build_s": time.perf_counter() - query_started,
    "solver_check_s": 0.0,
    "solver_model_s": 0.0,
    "host_validation_s": 0.0,
  }

  def finish(result):
    result["cost_s"] = {**costs, "total_s": time.perf_counter() - started}
    return result

  command = _solver_command()
  remaining_s = deadline - time.monotonic()
  if remaining_s <= 0:
    return finish(_solver_outcome("TIMEOUT", query, command, "check"))
  check_started = time.perf_counter()
  try:
    check_result = subprocess.run(
      command,
      input=query["check_script"],
      text=True,
      capture_output=True,
      check=False,
      timeout=remaining_s,
    )
  except subprocess.TimeoutExpired as error:
    costs["solver_check_s"] = time.perf_counter() - check_started
    return finish(_solver_outcome("TIMEOUT", query, command, "check", error=error))
  costs["solver_check_s"] = time.perf_counter() - check_started
  status = _solver_status(check_result.stdout)
  if status in {"UNSAT", "UNKNOWN"}:
    return finish(_solver_outcome(status, query, command, "check", check_result))
  if status != "SAT" or check_result.returncode != 0:
    return finish(_solver_outcome("ERROR", query, command, "check", check_result))

  remaining_s = deadline - time.monotonic()
  if remaining_s <= 0:
    return finish(_solver_outcome("TIMEOUT", query, command, "model"))
  model_started = time.perf_counter()
  try:
    model_result = subprocess.run(
      command,
      input=query["script"],
      text=True,
      capture_output=True,
      check=False,
      timeout=remaining_s,
    )
  except subprocess.TimeoutExpired as error:
    costs["solver_model_s"] = time.perf_counter() - model_started
    return finish(_solver_outcome("TIMEOUT", query, command, "model", error=error))
  costs["solver_model_s"] = time.perf_counter() - model_started
  model_status = _solver_status(model_result.stdout)
  if model_status in {"UNSAT", "UNKNOWN"}:
    return finish(_solver_outcome(model_status, query, command, "model", model_result))
  if model_status != "SAT" or model_result.returncode != 0:
    return finish(_solver_outcome("ERROR", query, command, "model", model_result))

  validation_started = time.perf_counter()
  requested = [*query["variables"], *query["output_names"]]
  model = parse_payload_model(model_result.stdout, requested)
  input_payloads = [model[name] for name in query["variables"]]
  output_payloads = [model[name] for name in query["output_names"]]
  replayed = evaluate_generated_payload_region(source, input_payloads)
  if output_payloads != replayed:
    raise AssertionError("SMT model outputs disagree with the host payload evaluator")
  calibration_mutants = {}
  for name in ("lost_iteration", "lost_update_acc_y"):
    mutant_payloads = evaluate_generated_payload_mutant(source, input_payloads, name)
    mismatch_lanes = compare_words(output_payloads, mutant_payloads)
    if not mismatch_lanes:
      raise AssertionError(f"generated payload witness missed {name} calibration")
    calibration_mutants[name] = {
      "status": "detected",
      "candidate_payloads": mutant_payloads,
      "candidate_float_bits": [
        f"{unembed_generated_payload(value):08x}" for value in mutant_payloads
      ],
      "mismatch_lanes": mismatch_lanes,
    }
  if query["property"] is not None:
    target = query["property"]["target"]
    index = query["property"]["index"]
    flat_index = tensor_exact_f32._flat_index(index, query["shape"])
    if output_payloads[flat_index] != target:
      raise AssertionError(
        "generated payload witness violates its fixed output property"
      )
  costs["host_validation_s"] = time.perf_counter() - validation_started
  return finish(
    {
      "status": "SAT",
      "solver": command[0],
      "transport_contract": query["transport_contract"],
      "query_sha256": query["sha256"],
      "query_smt2": query["script"],
      "shape": list(query["shape"]),
      "iterations": query["iterations"],
      "property": query["property"],
      "input_payloads": dict(zip(query["variables"], input_payloads)),
      "output_payloads": output_payloads,
      "input_float_words": [
        unembed_generated_payload(value) for value in input_payloads
      ],
      "output_float_words": [
        unembed_generated_payload(value) for value in output_payloads
      ],
      "input_float_bits": [
        f"{unembed_generated_payload(value):08x}" for value in input_payloads
      ],
      "output_float_bits": [
        f"{unembed_generated_payload(value):08x}" for value in output_payloads
      ],
      "calibration_mutants": calibration_mutants,
    }
  )


def render_generated_region_triton_kernel(
  source, *, iteration_count=None, omitted_updates=()
):
  """Emit a Triton kernel with the same dataflow and ordered updates as SIR."""
  shape, iterations, tensor_types, nodes, updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  if iteration_count is None:
    iteration_count = iterations
  if (
    type(iteration_count) is not int
    or iteration_count < 0
    or iteration_count > iterations
  ):
    raise ValueError("generated Triton iteration count must fit the source loop bound")
  omitted_updates = tuple(omitted_updates)
  update_names = {state for state, _root in updates}
  if any(name not in update_names for name in omitted_updates):
    raise ValueError("generated Triton mutant names an unknown loop-carried state")
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
    f"    for _ in tl.static_range({iteration_count}):",
  ]
  names = {"acc_x": "acc_x", "acc_y": "acc_y"}
  for index, (destination, _operation, _operands, _attribute) in enumerate(nodes):
    names[destination] = f"value_{index}"

  for destination, operation, operands, attribute in nodes:
    target = names[destination]
    if operation == "add":
      left, right = (names[name] for name in operands)
      expression = f"{left} + {right}"
    elif operation == "mul":
      left, right = (names[name] for name in operands)
      expression = f"{left} * {right}"
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
    if state not in omitted_updates:
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
