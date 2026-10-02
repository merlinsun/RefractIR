"""Independent host model for Triton's FpSan payload-ring subset."""

import math
import shutil
import struct
import subprocess

PAYLOAD_MASK = 0xFFFFFFFF


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
