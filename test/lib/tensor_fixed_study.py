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


def _validate_contract(domain, target, target_index):
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
  if (
    not isinstance(target_index, (tuple, list))
    or len(target_index) != 2
    or any(type(value) is not int or not 0 <= value < 2 for value in target_index)
  ):
    raise ValueError("target_index must identify an element of the 2x2 output")


def _certify_values(values, source, domain, target, target_index):
  if (
    not isinstance(values, (tuple, list))
    or len(values) != 4
    or any(
      type(value) is not int or not domain[0] <= value <= domain[1] for value in values
    )
  ):
    raise ValueError("witness must contain four integers in the frozen domain")
  input_bytes = struct.pack("<4f", *values)
  proof, output_bytes = tensor_exact_f32.certify(source, input_bytes, domain)
  observed = output_element(output_bytes, (2, 2), target_index)
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
  _validate_contract(domain, target, target_index)
  try:
    values = model[entry][symbol]
  except (KeyError, TypeError) as error:
    raise ValueError("solver model has no input witness") from error
  return _certify_values(values, source, domain, target, target_index)


def select_random_witness(
  source,
  *,
  domain,
  target,
  target_index,
  seed,
  proposal_budget,
):
  """Sample one common-domain input policy and check every proposal independently."""
  _validate_contract(domain, target, target_index)
  if type(seed) is not int or seed < 0:
    raise ValueError("seed must be a nonnegative integer")
  if type(proposal_budget) is not int or proposal_budget <= 0:
    raise ValueError("proposal_budget must be a positive integer")
  generator = random.Random(seed)
  started = time.perf_counter()
  generation_s = 0.0
  reference_validation_s = 0.0
  last_values = None
  for proposal_count in range(1, proposal_budget + 1):
    generation_started = time.perf_counter()
    values = tuple(generator.randint(domain[0], domain[1]) for _ in range(4))
    generation_s += time.perf_counter() - generation_started
    reference_started = time.perf_counter()
    proof, output_bytes = tensor_exact_f32.certify(
      source, struct.pack("<4f", *values), domain
    )
    reference_validation_s += time.perf_counter() - reference_started
    last_values = values
    if struct.pack(
      "<f", output_element(output_bytes, (2, 2), target_index)
    ) == struct.pack("<f", float(target)):
      result = {
        "status": "accepted",
        "values": values,
        "input_bytes": struct.pack("<4f", *values),
        "output_bytes": output_bytes,
        "proof": proof,
        "proposal_count": proposal_count,
      }
      break
  else:
    result = {
      "status": "proposal_exhausted",
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
