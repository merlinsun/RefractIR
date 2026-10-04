"""Input selection and independent replay for a fixed exact-f32 study."""

import math
import random
import struct
import time

from test.lib import tensor_exact_f32


def output_element(output_bytes, shape, index):
  if (
    not isinstance(shape, (tuple, list))
    or len(shape) != 2
    or any(type(size) is not int or size <= 0 for size in shape)
    or not isinstance(index, (tuple, list))
    or len(index) != 2
    or any(type(value) is not int for value in index)
    or any(value < 0 or value >= shape[axis] for axis, value in enumerate(index))
  ):
    raise ValueError("output shape or index is invalid")
  values = struct.unpack(f"<{math.prod(shape)}f", output_bytes)
  offset = index[0] * shape[1] + index[1]
  return values[offset]


def _source_shape(source):
  try:
    return tensor_exact_f32._generated_region_program(source)[0]
  except ValueError:
    return tensor_exact_f32._recognize(source)[0]


def _validate_contract(domain, target, target_index, shape=(2, 2)):
  if (
    not isinstance(domain, (tuple, list))
    or len(domain) != 2
    or any(type(value) is not int for value in domain)
    or domain[0] < 0
    or domain[0] > domain[1]
  ):
    raise ValueError("exact study domain must be a nonnegative integer interval")
  if isinstance(target, bool) or not isinstance(target, (int, float)):
    raise ValueError("target must be a finite numeric value")
  if not math.isfinite(target):
    raise ValueError("target must be a finite numeric value")
  if shape is not None and (
    not isinstance(target_index, (tuple, list))
    or len(target_index) != len(shape)
    or any(
      type(value) is not int or not 0 <= value < shape[axis]
      for axis, value in enumerate(target_index)
    )
  ):
    raise ValueError("target_index must identify an element of the output shape")


def _certify_values(values, source, domain, target, target_index, shape):
  count = math.prod(shape)
  if (
    not isinstance(values, (tuple, list))
    or len(values) != count
    or any(
      type(value) is not int or not domain[0] <= value <= domain[1] for value in values
    )
  ):
    raise ValueError("witness input must match the output shape and frozen domain")
  input_bytes = struct.pack(f"<{count}f", *values)
  proof, output_bytes = tensor_exact_f32.certify(source, input_bytes, domain)
  observed = output_element(output_bytes, shape, target_index)
  target_bytes = struct.pack("<f", float(target))
  if struct.pack("<f", observed) != target_bytes:
    raise ValueError(
      f"selected witness fails property at {tuple(target_index)}: "
      f"expected {target}, got {observed}"
    )
  return {
    "values": tuple(values),
    "input_bytes": input_bytes,
    "output_bytes": output_bytes,
    "proof": proof,
  }


def validate_solver_model(
  model,
  source,
  *,
  domain,
  target,
  target_index,
  entry="@kernel",
  symbol="%?x",
):
  """Extract one i32 model witness and replay the full exact-f32 output."""
  shape = _source_shape(source)
  _validate_contract(domain, target, target_index, shape)
  try:
    values = model[entry][symbol]
  except (KeyError, TypeError) as error:
    raise ValueError("solver model has no input witness") from error
  return _certify_values(values, source, domain, target, target_index, shape)


def select_random_witness(
  source,
  *,
  domain,
  target,
  target_index,
  seed,
  proposal_budget,
  deadline=None,
):
  """Sample one common-domain input policy and check every proposal independently."""
  _validate_contract(domain, target, target_index, shape=None)
  if type(seed) is not int or seed < 0:
    raise ValueError("seed must be a nonnegative integer")
  if type(proposal_budget) is not int or proposal_budget <= 0:
    raise ValueError("proposal_budget must be a positive integer")
  if deadline is not None and (
    isinstance(deadline, bool)
    or not isinstance(deadline, (int, float))
    or not math.isfinite(deadline)
  ):
    raise ValueError("deadline must be a finite monotonic timestamp")
  generator = random.Random(seed)
  started = time.perf_counter()
  if deadline is not None and started >= deadline:
    return {
      "values": None,
      "input_bytes": None,
      "output_bytes": None,
      "proof": None,
      "status": "timeout",
      "proposal_count": 0,
      "generation_s": 0.0,
      "reference_validation_s": 0.0,
      "selection_s": 0.0,
      "elapsed_s": 0.0,
      "seed": seed,
      "proposal_budget": proposal_budget,
      "input_domain": list(domain),
      "property": {"target": target, "index": list(target_index)},
    }
  shape = _source_shape(source)
  _validate_contract(domain, target, target_index, shape)
  generation_s = 0.0
  reference_validation_s = 0.0
  last_values = None
  count = math.prod(shape)
  proposal_count = 0
  status = None
  for _ in range(proposal_budget):
    if deadline is not None and time.perf_counter() >= deadline:
      status = "timeout"
      break
    proposal_count += 1
    generation_started = time.perf_counter()
    values = tuple(generator.randint(domain[0], domain[1]) for _ in range(count))
    generation_s += time.perf_counter() - generation_started
    reference_started = time.perf_counter()
    proof, output_bytes = tensor_exact_f32.certify(
      source, struct.pack(f"<{count}f", *values), domain
    )
    reference_validation_s += time.perf_counter() - reference_started
    last_values = values
    if deadline is not None and time.perf_counter() >= deadline:
      status = "timeout"
      break
    if struct.pack(
      "<f", output_element(output_bytes, shape, target_index)
    ) == struct.pack("<f", float(target)):
      status = "accepted"
      result = {
        "values": values,
        "input_bytes": struct.pack(f"<{count}f", *values),
        "output_bytes": output_bytes,
        "proof": proof,
      }
      break
  if status == "timeout":
    result = {
      "values": None,
      "input_bytes": None,
      "output_bytes": None,
      "proof": None,
      "last_proposed_values": last_values,
    }
  elif status != "accepted":
    status = "proposal_exhausted"
    result = {
      "values": None,
      "input_bytes": None,
      "output_bytes": None,
      "proof": None,
      "proposal_count": proposal_budget,
      "last_proposed_values": last_values,
    }
  elapsed = time.perf_counter() - started
  result.update(
    {
      "status": status,
      "proposal_count": proposal_count,
      "generation_s": generation_s,
      "reference_validation_s": reference_validation_s,
      "selection_s": max(0.0, elapsed - generation_s - reference_validation_s),
      "elapsed_s": elapsed,
      "seed": seed,
      "proposal_budget": proposal_budget,
      "input_domain": list(domain),
      "property": {"target": target, "index": list(target_index)},
    }
  )
  return result
