"""Exact or rational-tolerance comparison for complete tensor buffers."""

import math
import struct
from fractions import Fraction


def _tolerance(pair):
  if not isinstance(pair, (tuple, list)) or len(pair) != 2:
    raise ValueError("tolerance must be a numerator/denominator pair")
  numerator, denominator = pair
  if not isinstance(numerator, int) or not isinstance(denominator, int):
    raise ValueError("tolerance pair must contain integers")
  if numerator < 0 or denominator <= 0:
    raise ValueError("tolerance numerator must be nonnegative and denominator positive")
  return Fraction(numerator, denominator)


def _as_pair(value):
  return [value.numerator, value.denominator]


def _decode_f32(data):
  count = len(data) // 4
  words = struct.unpack(f"<{count}I", data)
  values = struct.unpack(f"<{count}f", data)
  return words, values


def compare(
  actual_bytes,
  reference_bytes,
  dtype,
  exact,
  atol_pair,
  rtol_pair,
  *,
  zero_sign_exact=False,
):
  """Compare all little-endian elements and report zero-based linear indices."""
  if dtype not in {"i32", "f32"}:
    return {
      "status": "invalid_case",
      "mismatching_indices": [],
      "reason": "unsupported dtype",
    }
  if not isinstance(zero_sign_exact, bool) or (zero_sign_exact and dtype != "f32"):
    return {
      "status": "invalid_case",
      "mismatching_indices": [],
      "reason": "zero_sign_exact is only valid for f32 comparisons",
    }
  if len(actual_bytes) != len(reference_bytes) or len(actual_bytes) % 4:
    return {
      "status": "invalid_case",
      "mismatching_indices": [],
      "reason": "buffer lengths differ or are not a multiple of four",
    }
  try:
    atol = _tolerance(atol_pair)
    rtol = _tolerance(rtol_pair)
  except ValueError as error:
    return {"status": "invalid_case", "mismatching_indices": [], "reason": str(error)}

  if dtype == "i32":
    actual = struct.unpack(f"<{len(actual_bytes) // 4}i", actual_bytes)
    reference = struct.unpack(f"<{len(reference_bytes) // 4}i", reference_bytes)
    mismatches = [i for i, (a, r) in enumerate(zip(actual, reference)) if a != r]
    return {
      "status": ("pass" if not mismatches else "exact_mismatch"),
      "mismatching_indices": mismatches,
    }

  actual_words, actual = _decode_f32(actual_bytes)
  reference_words, reference = _decode_f32(reference_bytes)
  if any(not math.isfinite(value) for value in reference):
    return {
      "status": "invalid_case",
      "mismatching_indices": [],
      "reason": "reference contains a non-finite f32 value",
    }

  if exact:
    mismatches = [
      i for i, (a, r) in enumerate(zip(actual_words, reference_words)) if a != r
    ]
    return {
      "status": ("pass" if not mismatches else "exact_mismatch"),
      "mismatching_indices": mismatches,
    }

  # Zero tolerance plus the signed-zero rule is a bit-exact contract: two
  # distinct f32 bit patterns are two distinct values except for the two zeros,
  # which the rule compares by word. A failure is then an exact mismatch, and
  # saying so keeps the verdict vocabulary equal to the contract.
  bit_exact = zero_sign_exact and atol == 0 and rtol == 0
  mismatches = []
  max_absolute = Fraction(0)
  max_relative = Fraction(0)
  for index, (a, r) in enumerate(zip(actual, reference)):
    if not math.isfinite(a):
      mismatches.append(index)
      continue
    if zero_sign_exact and r == 0.0:
      if actual_words[index] != reference_words[index]:
        mismatches.append(index)
      continue
    absolute = abs(Fraction.from_float(a) - Fraction.from_float(r))
    allowance = atol + rtol * abs(Fraction.from_float(r))
    if absolute > allowance:
      mismatches.append(index)
    max_absolute = max(max_absolute, absolute)
    if r != 0.0:
      max_relative = max(max_relative, absolute / abs(Fraction.from_float(r)))

  if not mismatches:
    status = "pass"
  elif bit_exact:
    status = "exact_mismatch"
  else:
    status = "numeric_mismatch"
  return {
    "status": status,
    "mismatching_indices": mismatches,
    "max_absolute_error": _as_pair(max_absolute),
    "max_relative_error": _as_pair(max_relative),
  }


def run_if_reference_stable(trace_128, trace_256, target_runner):
  """Withhold target execution when any materialized reference node changes."""
  if not all(isinstance(item, bytes) for item in (*trace_128, *trace_256)):
    raise TypeError("reference traces must contain canonical byte snapshots")
  trace_count = max(len(trace_128), len(trace_256))
  mismatches = [
    index
    for index in range(trace_count)
    if index >= len(trace_128)
    or index >= len(trace_256)
    or trace_128[index] != trace_256[index]
  ]
  if mismatches:
    return {
      "status": "oracle_unstable",
      "stable": False,
      "target_eligible": False,
      "target_ran": False,
      "compiler_claim": False,
      "mismatching_trace_indices": mismatches,
    }

  result = dict(target_runner())
  result.update(
    {
      "stable": True,
      "target_eligible": True,
      "target_ran": True,
      "compiler_claim": result.get("status") == "pass",
    }
  )
  return result
