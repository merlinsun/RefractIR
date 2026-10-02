"""Independent host model for Triton's FpSan payload-ring subset."""

import hashlib
import math
import re
import shutil
import struct
import subprocess

PAYLOAD_MASK = 0xFFFFFFFF
KNOWN_INPUT_PAYLOADS = (0, 1, PAYLOAD_MASK)
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
