"""Deterministic corpus, proposal, control, and ledger contracts for the
boundary-input sensitivity experiment.
"""

import copy
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

from test.lib.tensor_coverage import source_signature

CORPUS_SCHEMA = "refractir.tensor-boundary-sensitivity-corpus.v1"
LEDGER_SCHEMA = "refractir.tensor-boundary-sensitivity-ledger.v1"
LEGACY_SAMPLER_VERSION = "legacy-four-phase-cycle-v0"
BOUNDARY_SAMPLER_VERSION = "boundary-splitmix64-v1"
ARMS = ("legacy_cycle", "boundary")
CONTROL_IDS = (
  "operand_substitution",
  "tensor_permutation",
  "reduction_axis",
  "broadcast_source",
  "negative",
)
CONTROL_STATUSES = (
  "invalid_program",
  "inapplicable",
  "invalid_on_input",
  "distinguished",
  "not_distinguished",
  "timeout",
  "not_run",
)
PROPOSAL_STATUSES = {
  "accepted",
  "refused",
  "unsupported",
  "invalid",
  "invalid_on_input",
  "timeout",
  "error",
}


def _canonical_bytes(value):
  return json.dumps(
    value,
    ensure_ascii=False,
    allow_nan=False,
    sort_keys=True,
    separators=(",", ":"),
  ).encode("utf-8")


def _sha256(value):
  return hashlib.sha256(value).hexdigest()


def _corpus_payload(corpus):
  return {key: value for key, value in corpus.items() if key != "corpus_sha256"}


def _validate_config(config):
  if not isinstance(config, dict):
    raise ValueError("frozen corpus config must be an object")
  if config.get("profile") != "tensor-v2" or config.get("dtype") != "f32":
    raise ValueError("frozen corpus requires tensor-v2 f32 programs")
  if config.get("input_domain") != [-1, 2]:
    raise ValueError("frozen corpus requires the common input domain [-1, 2]")
  if config.get("proposals_per_arm") != 4:
    raise ValueError("frozen corpus requires four proposals per arm")
  comparison = config.get("comparison")
  if comparison != {
    "mode": "numeric",
    "atol": [2, 100000],
    "rtol": [2, 1000],
  }:
    raise ValueError("frozen corpus requires the declared numeric comparator")
  launch = config.get("launch")
  if (
    not isinstance(launch, dict)
    or launch.get("num_warps") not in {1, 2, 4, 8}
    or launch.get("num_stages") != 1
    or launch.get("enable_fp_fusion") is not False
  ):
    raise ValueError("frozen corpus requires fixed tensor-v2 launch settings")
  if config.get("physical_layouts") != {
    "policy": "contiguous-row-major-v1",
    "buffer_offsets": 0,
    "grid": [1, 1, 1],
  }:
    raise ValueError("frozen corpus requires declared physical layouts")
  if (
    not isinstance(config.get("source_snapshot_sha256"), str)
    or not config["source_snapshot_sha256"]
  ):
    raise ValueError("frozen corpus requires a source snapshot digest")
  try:
    _canonical_bytes(config)
  except (TypeError, ValueError) as error:
    raise ValueError(f"frozen corpus config is not canonical JSON: {error}") from error


def freeze_corpus(candidates, *, config, base_revision, target_count=256):
  """Freeze the first ``target_count`` unique checked sources in candidate order.

  Candidate records with status ``candidate`` must have a source string and are
  assumed to have passed the shared native frontend check. Other statuses are
  retained in the enumeration as generation or frontend outcomes. Deduplication
  uses the existing alpha-normalized SIR token signature, which retains literals,
  shapes, types, and operators.
  """
  _validate_config(config)
  if not isinstance(base_revision, str) or not base_revision:
    raise ValueError("frozen corpus base revision must be nonempty")
  if type(target_count) is not int or target_count < 1:
    raise ValueError("frozen corpus target_count must be a positive integer")

  enumeration = []
  kernels = []
  signature_to_kernel = {}
  for candidate in candidates:
    if len(kernels) == target_count:
      break
    if not isinstance(candidate, dict):
      raise ValueError("frozen corpus candidate must be an object")
    seed = candidate.get("seed")
    if type(seed) is not int or seed < 0 or seed > 0xFFFFFFFF:
      raise ValueError("frozen corpus candidate seed must be uint32")
    status = candidate.get("status")
    if not isinstance(status, str) or not status:
      raise ValueError("frozen corpus candidate status must be nonempty")
    record = {"seed": seed, "status": status}

    if status != "candidate":
      reason = candidate.get("reason")
      if reason is not None:
        if not isinstance(reason, str):
          raise ValueError("frozen corpus candidate reason must be text")
        record["reason"] = reason
      enumeration.append(record)
      continue

    source = candidate.get("source")
    if not isinstance(source, str) or not source.strip():
      raise ValueError("frozen corpus candidate source must be nonempty text")
    signature = source_signature(source)
    source_digest = _sha256(source.encode("utf-8"))
    record.update(
      {"source": source, "source_sha256": source_digest, "source_signature": signature}
    )
    existing = signature_to_kernel.get(signature)
    if existing is not None:
      record["status"] = "duplicate"
      record["duplicate_of"] = existing
      enumeration.append(record)
      continue

    kernel_id = f"kernel-{signature}"
    record["status"] = "selected"
    record["kernel_id"] = kernel_id
    enumeration.append(record)
    kernel = {
      "kernel_id": kernel_id,
      "candidate_seed": seed,
      "source": source,
      "source_sha256": source_digest,
      "source_signature": signature,
    }
    kernels.append(kernel)
    signature_to_kernel[signature] = kernel_id

  if len(kernels) != target_count:
    raise ValueError(
      f"frozen corpus candidate schedule produced {len(kernels)} unique kernels; "
      f"expected {target_count}"
    )
  corpus = {
    "schema": CORPUS_SCHEMA,
    "base_revision": base_revision,
    "target_count": target_count,
    "config": copy.deepcopy(config),
    "candidate_enumeration": enumeration,
    "kernels": kernels,
  }
  corpus["corpus_sha256"] = _sha256(_canonical_bytes(corpus))
  return corpus


def validate_frozen_corpus(corpus, *, expected_sha256=None):
  """Validate the frozen manifest digest, source identities, and exact count."""
  message = "frozen corpus"
  if not isinstance(corpus, dict) or corpus.get("schema") != CORPUS_SCHEMA:
    raise ValueError(f"{message} schema is invalid")
  _validate_config(corpus.get("config"))
  target_count = corpus.get("target_count")
  kernels = corpus.get("kernels")
  if type(target_count) is not int or target_count < 1 or not isinstance(kernels, list):
    raise ValueError(f"{message} count or kernels are invalid")
  if len(kernels) != target_count:
    raise ValueError(f"{message} does not contain its exact target count")
  if not isinstance(corpus.get("candidate_enumeration"), list):
    raise ValueError(f"{message} candidate enumeration is missing")
  signatures = set()
  ids = set()
  for kernel in kernels:
    if not isinstance(kernel, dict):
      raise ValueError(f"{message} contains an invalid kernel record")
    source = kernel.get("source")
    if not isinstance(source, str) or not source.strip():
      raise ValueError(f"{message} source is missing")
    digest = _sha256(source.encode("utf-8"))
    signature = source_signature(source)
    if (
      kernel.get("source_sha256") != digest
      or kernel.get("source_signature") != signature
    ):
      raise ValueError(f"{message} source identity changed")
    kernel_id = f"kernel-{signature}"
    if (
      kernel.get("kernel_id") != kernel_id
      or signature in signatures
      or kernel_id in ids
    ):
      raise ValueError(f"{message} contains duplicate or invalid kernel identity")
    signatures.add(signature)
    ids.add(kernel_id)
  digest = _sha256(_canonical_bytes(_corpus_payload(corpus)))
  if corpus.get("corpus_sha256") != digest:
    raise ValueError(f"{message} digest mismatch")
  if expected_sha256 is not None and digest != expected_sha256:
    raise ValueError(f"{message} differs from the expected frozen digest")
  return digest


def generate_candidate_schedule(
  generator_api,
  *,
  target_count=256,
  start_seed=2027,
  max_candidates=None,
  timeout_s=30,
  frontend_checker=None,
):
  """Use the native compositional builder before any input-policy selection."""
  api = Path(generator_api).resolve()
  if type(target_count) is not int or target_count < 1:
    raise ValueError("target_count must be a positive integer")
  if type(start_seed) is not int or start_seed < 0 or start_seed > 0xFFFFFFFF:
    raise ValueError("start_seed must be uint32")
  if max_candidates is None:
    max_candidates = target_count * 50
  if type(max_candidates) is not int or max_candidates < target_count:
    raise ValueError("max_candidates must be at least target_count")
  if type(timeout_s) not in (int, float) or timeout_s <= 0:
    raise ValueError("timeout_s must be positive")
  if frontend_checker is None:
    from test.unit.tensor_support import check_source

    def frontend_checker(source):
      return check_source(source, tensor_profile="v2")

  candidates = []
  signatures = set()
  for offset in range(max_candidates):
    seed = start_seed + offset
    if seed > 0xFFFFFFFF:
      raise ValueError("candidate seed schedule exceeds uint32")
    try:
      result = subprocess.run(
        [str(api), "--generate", str(seed)],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_s,
      )
    except subprocess.TimeoutExpired as error:
      candidates.append(
        {"seed": seed, "status": "generation_timeout", "reason": str(error)}
      )
      continue
    except OSError as error:
      raise RuntimeError(f"native graph generator is unavailable: {error}") from error

    if result.returncode != 0:
      detail = (result.stderr or result.stdout).strip()
      status = (
        "graph_generation_failure"
        if detail.startswith("graph_exhaustion:")
        else "frontend_failure"
      )
      candidates.append({"seed": seed, "status": status, "reason": detail})
      continue
    source = result.stdout
    if not source.strip():
      candidates.append(
        {"seed": seed, "status": "frontend_failure", "reason": "empty generated source"}
      )
      continue
    checked = frontend_checker(source)
    if checked.returncode != 0:
      candidates.append(
        {
          "seed": seed,
          "status": "frontend_failure",
          "reason": (checked.stderr or checked.stdout).strip(),
        }
      )
      continue
    signature = source_signature(source)
    candidates.append({"seed": seed, "status": "candidate", "source": source})
    if signature not in signatures:
      signatures.add(signature)
      if len(signatures) == target_count:
        return candidates

  raise RuntimeError(
    f"native graph schedule produced {len(signatures)} unique kernels from "
    f"{max_candidates} candidates; target was {target_count}"
  )


def input_signature_from_source(source):
  """Parse the ordered scalar/tensor input signature of a generated kernel."""
  function = re.search(r"fun\s+@kernel\((.*?)\)\s*:", source, re.DOTALL)
  if function is None:
    raise ValueError("generated source has no @kernel entry")
  parameters = function.group(1)
  pattern = re.compile(
    r"(%[A-Za-z_][A-Za-z_0-9.$-]*)\s*:\s*"
    r"(?:tensor\s*\[([^]]*)\]\s*)?(i32|f32)"
  )
  matches = list(pattern.finditer(parameters))
  if parameters.strip() and not matches:
    raise ValueError("generated kernel input signature is unsupported")
  if len(matches) != len(re.findall(r"%[A-Za-z_][A-Za-z_0-9.$-]*\s*:", parameters)):
    raise ValueError("generated kernel input signature contains an unsupported type")
  signature = []
  for match in matches:
    name, shape_text, dtype = match.groups()
    shape = (
      list(_parse_shape(shape_text))
      if shape_text is not None and shape_text.strip()
      else []
    )
    if len(shape) > 3 or any(size <= 0 or size & (size - 1) for size in shape):
      raise ValueError(f"generated kernel input {name} has an invalid tensor-v2 shape")
    count = math.prod(shape) if shape else 1
    if count > 512:
      raise ValueError(f"generated kernel input {name} exceeds tensor-v2 size")
    signature.append(
      {
        "name": name.removeprefix("%"),
        "dtype": dtype,
        "shape": shape,
        "byte_length": count * 4,
      }
    )
  return signature


def propose_boundary_inputs(generator_api, source, sampler_seed, *, timeout_s=10):
  """Ask the production C++ boundary sampler for inputs to a frozen source."""
  if type(sampler_seed) is not int or sampler_seed < 0 or sampler_seed > 0xFFFFFFFF:
    raise ValueError("sampler_seed must be uint32")
  with tempfile.TemporaryDirectory(prefix="refractir-boundary-proposal-") as directory:
    source_path = Path(directory) / "program.sir"
    source_path.write_text(source, encoding="utf-8")
    try:
      result = subprocess.run(
        [str(Path(generator_api).resolve()), str(source_path), str(sampler_seed)],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_s,
      )
    except subprocess.TimeoutExpired as error:
      return {
        "status": "timeout",
        "reason": str(error),
        "sampler_version": BOUNDARY_SAMPLER_VERSION,
        "buffers": [],
      }
  if result.returncode != 0:
    return {
      "status": "refused",
      "reason": (result.stderr or result.stdout).strip(),
      "sampler_version": BOUNDARY_SAMPLER_VERSION,
      "buffers": [],
    }
  try:
    proposal = json.loads(result.stdout)
  except json.JSONDecodeError as error:
    return {
      "status": "error",
      "reason": f"native proposal returned invalid JSON: {error}",
      "sampler_version": BOUNDARY_SAMPLER_VERSION,
      "buffers": [],
    }
  if proposal.get("outcome") != "accepted":
    return {
      "status": "refused",
      "reason": proposal.get("reason", proposal.get("outcome", "unknown")),
      "sampler_version": proposal.get("sampler_version"),
      "buffers": [],
    }
  if proposal.get("sampler_version") != BOUNDARY_SAMPLER_VERSION:
    raise ValueError("native boundary sampler version differs from frozen protocol")
  buffers = []
  for item in proposal.get("inputs", []):
    if item.get("element") not in {"i32", "f32"} or not isinstance(
      item.get("bytes"), str
    ):
      raise ValueError("native boundary sampler returned a malformed buffer")
    try:
      data = bytes.fromhex(item["bytes"])
    except ValueError as error:
      raise ValueError("native boundary sampler returned malformed bytes") from error
    buffers.append(
      {
        "name": item["name"],
        "dtype": item["element"],
        "shape": item["shape"],
        "bytes": data,
        "bytes_hex": data.hex(),
      }
    )
  return {
    "status": "accepted",
    "sampler_version": proposal["sampler_version"],
    "buffers": buffers,
  }


def legacy_cycle_buffers(input_records, phase):
  """Materialize one phase of the old shared four-value boundary cycle."""
  if type(phase) is not int or phase not in range(4):
    raise ValueError("legacy cycle phase must be in [0, 3]")
  buffers = []
  for record in input_records:
    if not isinstance(record, dict) or record.get("dtype") not in {"i32", "f32"}:
      raise ValueError("legacy cycle input record has an unsupported dtype")
    shape = record.get("shape")
    if not isinstance(shape, list) or any(
      type(size) is not int or size <= 0 for size in shape
    ):
      raise ValueError("legacy cycle input shape is invalid")
    count = math.prod(shape) if shape else 1
    values = [((index + phase) % 4) - 1 for index in range(count)]
    format_code = "i" if record["dtype"] == "i32" else "f"
    data = struct.pack(f"<{count}{format_code}", *values)
    buffers.append(
      {
        "name": record["name"],
        "dtype": record["dtype"],
        "shape": list(shape),
        "bytes": data,
        "bytes_hex": data.hex(),
      }
    )
  return buffers


def combine_input_buffers(buffers, input_records):
  """Concatenate proposed inputs in the order declared by the frozen case."""
  by_name = {buffer["name"]: buffer for buffer in buffers}
  if len(by_name) != len(buffers) or set(by_name) != {
    item["name"] for item in input_records
  }:
    raise ValueError("proposal inputs do not match the frozen input signature")
  ordered = []
  for record in input_records:
    buffer = by_name[record["name"]]
    if buffer.get("dtype") != record["dtype"] or buffer.get("shape") != record["shape"]:
      raise ValueError(f"proposal signature changed for input {record['name']}")
    data = buffer.get("bytes")
    if data is None:
      try:
        data = bytes.fromhex(buffer["bytes_hex"])
      except (KeyError, ValueError) as error:
        raise ValueError("proposal input is missing its byte buffer") from error
    if len(data) != record["byte_length"]:
      raise ValueError(f"proposal byte length changed for input {record['name']}")
    ordered.append(data)
  return b"".join(ordered)


def replay_reference(source, input_bytes, *, reference_runner=None):
  """Replay the complete program through the existing tensor-v2 interpreter."""
  if reference_runner is None:
    from test.unit.tensor_support import run_typed

    reference_runner = run_typed
  try:
    result, output = reference_runner(
      source,
      input_bytes,
      tensor_profile="v2",
      reference_stability=True,
    )
  except subprocess.TimeoutExpired as error:
    return {"status": "timeout", "reason": str(error), "output_bytes": None}
  if result.returncode != 0 or output is None:
    return {
      "status": "invalid_on_input",
      "reason": (result.stderr or result.stdout).strip(),
      "returncode": result.returncode,
      "output_bytes": None,
    }
  return {
    "status": "accepted",
    "returncode": 0,
    "output_bytes": output,
    "output_bytes_hex": output.hex(),
  }


def _parse_shape(text):
  return tuple(int(value.strip()) for value in text.split(",") if value.strip())


def _broadcast_shape(left, right):
  rank = max(len(left), len(right))
  left = (1,) * (rank - len(left)) + tuple(left)
  right = (1,) * (rank - len(right)) + tuple(right)
  output = []
  for lhs, rhs in zip(left, right, strict=True):
    if lhs != rhs and lhs != 1 and rhs != 1:
      return None
    output.append(max(lhs, rhs))
  return tuple(output)


def _control_shape_environment(source):
  function = re.search(r"fun\s+@kernel\((.*?)\)\s*:", source, re.DOTALL)
  if function is None:
    return {}, []
  shapes = {}
  parameters = []
  for match in re.finditer(
    r"(%[A-Za-z_][A-Za-z_0-9.$-]*)\s*:\s*tensor\s*\[([^]]+)\]\s*f32",
    function.group(1),
  ):
    name, shape_text = match.groups()
    shape = _parse_shape(shape_text)
    shapes[name] = shape
    parameters.append((name, shape))
  return shapes, parameters


def _shape_before_lines(source):
  shapes, parameters = _control_shape_environment(source)
  declared_shapes = {
    match.group(1): _parse_shape(match.group(2))
    for match in re.finditer(
      r"\blet(?:\s+mut)?\s+(%[A-Za-z_][A-Za-z_0-9.$-]*)\s*:\s*"
      r"tensor\s*\[([^]]+)\]\s*f32\b",
      source,
    )
  }
  states = []
  for line in source.splitlines(keepends=True):
    states.append(dict(shapes))
    alias = re.match(
      r"\s*(%[A-Za-z_][A-Za-z_0-9.$-]*)\s*=\s*"
      r"(%[A-Za-z_][A-Za-z_0-9.$-]*)\s*;\s*$",
      line,
    )
    if alias is not None:
      destination, operand = alias.groups()
      shape = shapes.get(operand, declared_shapes.get(destination))
      if shape is not None:
        shapes[destination] = shape
      continue

    assignment = re.match(
      r"\s*(%[A-Za-z_][A-Za-z_0-9.$-]*)\s*=\s*tensor\s+([A-Za-z_]+)\s+(.+);\s*$",
      line,
    )
    if assignment is None:
      continue
    destination, operation, arguments = assignment.groups()
    operands = re.match(
      r"(%[A-Za-z_][A-Za-z_0-9.$-]*)\s*,\s*(%[A-Za-z_][A-Za-z_0-9.$-]*)",
      arguments,
    )
    if operation in {"add", "sub", "mul", "min", "max"}:
      inferred_shape = None
      if operands is not None:
        left, right = operands.groups()
        if left in shapes and right in shapes:
          inferred_shape = _broadcast_shape(shapes[left], shapes[right])
      else:
        operand = re.match(r"(%[A-Za-z_][A-Za-z_0-9.$-]*)", arguments)
        if operand is not None:
          inferred_shape = shapes.get(operand.group(1))
      shape = declared_shapes.get(destination, inferred_shape)
      if shape is not None:
        shapes[destination] = shape
      continue

    inferred_shape = None
    operand = re.match(r"(%[A-Za-z_][A-Za-z_0-9.$-]*)", arguments)
    if operand is None or operand.group(1) not in shapes:
      shape = declared_shapes.get(destination)
      if shape is not None:
        shapes[destination] = shape
      continue
    source_shape = shapes[operand.group(1)]
    if operation == "transpose":
      permutation = re.search(r",\s*\[([^]]+)\]", arguments)
      if permutation:
        axes = _parse_shape(permutation.group(1))
        if len(axes) == len(source_shape) and sorted(axes) == list(range(len(axes))):
          inferred_shape = tuple(source_shape[axis] for axis in axes)
    elif operation == "sum":
      reduction = re.search(r",\s*(-?\d+)\s*,\s*(keep|drop)\b", arguments)
      if reduction:
        axis = int(reduction.group(1))
        axis = axis if axis >= 0 else axis + len(source_shape)
        if 0 <= axis < len(source_shape):
          if reduction.group(2) == "keep":
            result = list(source_shape)
            result[axis] = 1
            inferred_shape = tuple(result)
          else:
            inferred_shape = source_shape[:axis] + source_shape[axis + 1 :]
    elif operation == "expand":
      expansion = re.search(r",\s*(-?\d+)\s*$", arguments)
      if expansion:
        axis = int(expansion.group(1))
        rank = len(source_shape) + 1
        axis = axis if axis >= 0 else axis + rank
        if 0 <= axis < rank:
          result = list(source_shape)
          result.insert(axis, 1)
          inferred_shape = tuple(result)
    elif operation == "broadcast":
      target = re.search(r",\s*\[([^]]+)\]", arguments)
      if target:
        target_shape = _parse_shape(target.group(1))
        if _broadcast_shape(source_shape, target_shape) == tuple(target_shape):
          inferred_shape = tuple(target_shape)
    shape = declared_shapes.get(destination, inferred_shape)
    if shape is not None:
      shapes[destination] = shape
  return states, parameters


def _unavailable_control(control_id, reason):
  return {"control_id": control_id, "status": "inapplicable", "reason": reason}


def build_control_catalogue(source):
  """Build the predeclared, shape-preserving source-level controls."""
  states, parameters = _shape_before_lines(source)
  lines = source.splitlines(keepends=True)
  controls = {}

  operand_source = None
  for line_index, line in enumerate(lines):
    operation = re.match(
      r"\s*%[A-Za-z_][A-Za-z_0-9.$-]*\s*=\s*tensor\s+"
      r"(?:add|sub|mul|min|max)\s+(%[A-Za-z_][A-Za-z_0-9.$-]*),\s*"
      r"(%[A-Za-z_][A-Za-z_0-9.$-]*)\s*;",
      line,
    )
    if operation is None or line_index >= len(states):
      continue
    first_operand, _second_operand = operation.groups()
    shape = states[line_index].get(first_operand)
    replacement = next(
      (
        name
        for name, candidate_shape in parameters
        if name != first_operand and candidate_shape == shape
      ),
      None,
    )
    if replacement is not None:
      operand_source = "".join(
        lines[:line_index]
        + [line.replace(first_operand, replacement, 1)]
        + lines[line_index + 1 :]
      )
      break
  controls["operand_substitution"] = (
    {
      "control_id": "operand_substitution",
      "status": "available",
      "source": operand_source,
    }
    if operand_source is not None
    else _unavailable_control(
      "operand_substitution", "no equal-shape alternate parameter"
    )
  )

  permutation_source = None
  for line_index, line in enumerate(lines):
    operation = re.match(
      r"(\s*%[A-Za-z_][A-Za-z_0-9.$-]*\s*=\s*tensor\s+transpose\s+)"
      r"(%[A-Za-z_][A-Za-z_0-9.$-]*),\s*\[([^]]+)\](\s*;\s*\n?)$",
      line,
    )
    if operation is None or line_index >= len(states):
      continue
    prefix, operand, permutation_text, suffix = operation.groups()
    shape = states[line_index].get(operand)
    permutation = _parse_shape(permutation_text)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
      continue
    if sorted(permutation) != [0, 1]:
      continue
    alternative = tuple(reversed(permutation))
    if alternative == permutation:
      continue
    text = f"{prefix}{operand}, [{', '.join(map(str, alternative))}]{suffix}"
    permutation_source = "".join(lines[:line_index] + [text] + lines[line_index + 1 :])
    break
  controls["tensor_permutation"] = (
    {
      "control_id": "tensor_permutation",
      "status": "available",
      "source": permutation_source,
    }
    if permutation_source is not None
    else _unavailable_control(
      "tensor_permutation", "no square tensor transpose with a distinct permutation"
    )
  )

  reduction_source = None
  for line_index, line in enumerate(lines):
    operation = re.match(
      r"(\s*%[A-Za-z_][A-Za-z_0-9.$-]*\s*=\s*tensor\s+sum\s+)"
      r"(%[A-Za-z_][A-Za-z_0-9.$-]*),\s*(-?\d+),\s*drop(\s*;\s*\n?)$",
      line,
    )
    if operation is None or line_index >= len(states):
      continue
    prefix, operand, axis_text, suffix = operation.groups()
    shape = states[line_index].get(operand)
    if shape is None or len(shape) < 2:
      continue
    axis = int(axis_text)
    axis = axis if axis >= 0 else axis + len(shape)
    if axis < 0 or axis >= len(shape):
      continue
    alternative = next(
      (
        index
        for index, extent in enumerate(shape)
        if index != axis and extent == shape[axis]
      ),
      None,
    )
    if alternative is None:
      continue
    text = f"{prefix}{operand}, {alternative}, drop{suffix}"
    reduction_source = "".join(lines[:line_index] + [text] + lines[line_index + 1 :])
    break
  controls["reduction_axis"] = (
    {"control_id": "reduction_axis", "status": "available", "source": reduction_source}
    if reduction_source is not None
    else _unavailable_control(
      "reduction_axis", "no drop reduction has an equal-sized alternate axis"
    )
  )

  broadcast_source = None
  for line_index, line in enumerate(lines):
    operation = re.match(
      r"(\s*%[A-Za-z_][A-Za-z_0-9.$-]*\s*=\s*tensor\s+broadcast\s+)"
      r"(%[A-Za-z_][A-Za-z_0-9.$-]*),\s*\[([^]]+)\](\s*;\s*\n?)$",
      line,
    )
    if operation is None or line_index >= len(states):
      continue
    prefix, operand, target_text, suffix = operation.groups()
    current = states[line_index]
    source_shape = current.get(operand)
    target_shape = _parse_shape(target_text)
    if source_shape is None:
      continue
    replacement = next(
      (
        name
        for name, candidate_shape in current.items()
        if name != operand
        and candidate_shape != source_shape
        and _broadcast_shape(candidate_shape, target_shape) == target_shape
      ),
      None,
    )
    if replacement is None:
      continue
    text = f"{prefix}{replacement}, [{target_text}]{suffix}"
    broadcast_source = "".join(lines[:line_index] + [text] + lines[line_index + 1 :])
    break
  controls["broadcast_source"] = (
    {
      "control_id": "broadcast_source",
      "status": "available",
      "source": broadcast_source,
    }
    if broadcast_source is not None
    else _unavailable_control(
      "broadcast_source", "no differently shaped value broadcasts to the frozen target"
    )
  )
  controls["negative"] = {
    "control_id": "negative",
    "status": "available",
    "source": source,
  }
  return controls


def evaluate_control(
  control,
  *,
  original_source,
  input_bytes,
  expected_bytes,
  output_shape,
  checker=None,
  reference_runner=None,
  comparison=None,
):
  """Check a fixed control, replay it, and compare the complete output."""
  control_id = control.get("control_id")
  status = control.get("status")
  if status == "inapplicable":
    return {
      "control_id": control_id,
      "status": "inapplicable",
      "reason": control.get("reason"),
    }
  if status != "available":
    raise ValueError("control must be available or inapplicable")
  source = control.get("source")
  if not isinstance(source, str):
    raise ValueError("available control is missing source")
  if checker is None:
    from test.unit.tensor_support import check_source

    def checker(candidate):
      return check_source(candidate, tensor_profile="v2")

  try:
    checked = checker(source)
  except subprocess.TimeoutExpired as error:
    return {"control_id": control_id, "status": "timeout", "reason": str(error)}
  if checked.returncode != 0:
    return {
      "control_id": control_id,
      "status": "invalid_program",
      "reason": (checked.stderr or checked.stdout).strip(),
    }
  replay = replay_reference(source, input_bytes, reference_runner=reference_runner)
  if replay["status"] == "timeout":
    return {"control_id": control_id, "status": "timeout", "reason": replay["reason"]}
  if replay["status"] != "accepted":
    return {
      "control_id": control_id,
      "status": "invalid_on_input",
      "reason": replay.get("reason"),
    }
  if comparison is None:
    comparison = {"mode": "numeric", "atol": [2, 100000], "rtol": [2, 1000]}
  from test.lib.run_triton_tests import compare_outputs

  compared = compare_outputs(
    replay["output_bytes"], expected_bytes, "f32", output_shape, comparison
  )
  if compared["status"] == "invalid_case":
    return {
      "control_id": control_id,
      "status": "invalid_program",
      "reason": compared.get("reason", "control output shape is invalid"),
    }
  return {
    "control_id": control_id,
    "status": "not_distinguished" if compared["status"] == "pass" else "distinguished",
    "comparison": compared,
    "output_bytes_hex": replay["output_bytes_hex"],
  }


def materialize_proposal_case(
  template_directory,
  destination,
  *,
  input_bytes,
  expected_bytes,
  policy,
  sampler_version,
  proposal_id,
  corpus_sha256,
  input_seed=None,
):
  """Clone a validated case and replace its runtime bytes and provenance."""
  source_root = Path(template_directory).resolve()
  target = Path(destination).resolve()
  if target.exists():
    raise FileExistsError(target)
  shutil.copytree(source_root, target)
  try:
    manifest_path = target / "case.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list):
      raise ValueError("template has no frozen input signature")
    expected_input_bytes = sum(record["byte_length"] for record in inputs)
    if len(input_bytes) != expected_input_bytes:
      raise ValueError("proposal input byte length differs from frozen signature")
    output_length = manifest.get("output", {}).get("byte_length")
    if type(output_length) is not int or len(expected_bytes) != output_length:
      raise ValueError("reference output byte length differs from frozen signature")

    (target / "inputs.bin").write_bytes(input_bytes)
    (target / "expected.bin").write_bytes(expected_bytes)
    for name, payload in (
      ("inputs.bin", input_bytes),
      ("expected.bin", expected_bytes),
    ):
      manifest["files"][name] = {"bytes": len(payload), "sha256": _sha256(payload)}
    domain = {"lo": -1, "hi": 2}
    generator = manifest.setdefault("generator", {})
    behavior = manifest.setdefault("coverage", {}).setdefault("input_behavior", {})
    generator["input_policy"] = policy
    generator["input_sampler_version"] = sampler_version
    generator["input_domain"] = domain
    if input_seed is not None:
      generator["input_seed"] = input_seed
    behavior["policy"] = policy
    behavior["sampler_version"] = sampler_version
    behavior["input_domain"] = domain
    manifest["boundary_sensitivity"] = {
      "corpus_sha256": corpus_sha256,
      "proposal_id": proposal_id,
    }
    manifest_path.write_text(
      json.dumps(
        manifest, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
      )
      + "\n",
      encoding="utf-8",
    )
  except BaseException:
    shutil.rmtree(target, ignore_errors=True)
    raise
  return target


def _splitmix64(value):
  value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
  value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
  value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
  return value ^ (value >> 31)


def _stream_seed(seed, stream):
  return _splitmix64(seed ^ stream) & 0xFFFFFFFF


def allocate_proposals(corpus, *, proposals_per_arm=4):
  """Allocate immutable proposal slots; a failed slot is never replaced."""
  validate_frozen_corpus(corpus)
  if proposals_per_arm != corpus["config"]["proposals_per_arm"]:
    raise ValueError("proposal allocation differs from frozen corpus config")
  if type(proposals_per_arm) is not int or proposals_per_arm < 1:
    raise ValueError("proposals_per_arm must be a positive integer")
  slots = []
  for kernel in corpus["kernels"]:
    for arm in ARMS:
      for index in range(proposals_per_arm):
        slot = {
          "proposal_id": f"{kernel['kernel_id']}:{arm}:{index}",
          "kernel_id": kernel["kernel_id"],
          "candidate_seed": kernel["candidate_seed"],
          "arm": arm,
          "proposal_index": index,
          "sampler_version": (
            LEGACY_SAMPLER_VERSION
            if arm == "legacy_cycle"
            else BOUNDARY_SAMPLER_VERSION
          ),
        }
        if arm == "legacy_cycle":
          slot["legacy_phase"] = index
        else:
          slot["sampler_seed"] = _stream_seed(
            kernel["candidate_seed"], 0x696E707574 + index
          )
        slots.append(slot)
  return slots


def new_study_ledger(corpus, *, proposals_per_arm=4):
  digest = validate_frozen_corpus(corpus)
  proposals = allocate_proposals(corpus, proposals_per_arm=proposals_per_arm)
  return {
    "schema": LEDGER_SCHEMA,
    "corpus_sha256": digest,
    "proposals_per_arm": proposals_per_arm,
    "proposal_count_planned": len(proposals),
    "proposals": [
      dict(slot, proposal=None, reference=None, result=None, control_results=[])
      for slot in proposals
    ],
    "controls": [],
  }


def record_proposal_input(ledger, proposal_id, proposal):
  """Durably stage immutable proposal bytes before reference or GPU work."""
  if not isinstance(ledger, dict) or ledger.get("schema") != LEDGER_SCHEMA:
    raise ValueError("study ledger schema is invalid")
  if (
    not isinstance(proposal, dict)
    or proposal.get("proposal_status") not in PROPOSAL_STATUSES
  ):
    raise ValueError("input proposal has an invalid status")
  matches = [
    slot
    for slot in ledger.get("proposals", [])
    if slot.get("proposal_id") == proposal_id
  ]
  if len(matches) != 1:
    raise ValueError("proposal id is not a unique allocated slot")
  slot = matches[0]
  sampler_version = proposal.get("sampler_version")
  if sampler_version is not None and sampler_version != slot["sampler_version"]:
    raise ValueError("input proposal sampler version differs from allocated slot")
  input_hex = proposal.get("input_bytes_hex")
  if proposal["proposal_status"] == "accepted":
    if not isinstance(input_hex, str):
      raise ValueError("accepted input proposal must retain its input bytes")
    try:
      bytes.fromhex(input_hex)
    except ValueError as error:
      raise ValueError("input proposal bytes are not valid hex") from error
  if slot.get("proposal") is not None:
    if _canonical_bytes(slot["proposal"]) != _canonical_bytes(proposal):
      raise ValueError("proposal slot already has a different input proposal")
    return False
  if slot.get("result") is not None:
    raise ValueError("cannot stage input after the proposal result is complete")
  slot["proposal"] = copy.deepcopy(proposal)
  return True


def record_proposal_result(ledger, proposal_id, result):
  """Record a proposal once; an identical replay is a no-op."""
  if not isinstance(ledger, dict) or ledger.get("schema") != LEDGER_SCHEMA:
    raise ValueError("study ledger schema is invalid")
  if (
    not isinstance(result, dict)
    or result.get("proposal_status") not in PROPOSAL_STATUSES
  ):
    raise ValueError("proposal result has an invalid status")
  matches = [
    slot
    for slot in ledger.get("proposals", [])
    if slot.get("proposal_id") == proposal_id
  ]
  if len(matches) != 1:
    raise ValueError("proposal id is not a unique allocated slot")
  slot = matches[0]
  staged = slot.get("proposal")
  if staged is not None:
    staged_bytes = staged.get("input_bytes_hex")
    result_bytes = result.get("input_bytes_hex")
    if result_bytes is not None and staged_bytes != result_bytes:
      raise ValueError("proposal result input bytes differ from the staged input")
  if slot.get("result") is not None:
    if _canonical_bytes(slot["result"]) != _canonical_bytes(result):
      raise ValueError("proposal slot already has a different result")
    return False
  slot["result"] = copy.deepcopy(result)
  return True


def record_control_result(ledger, proposal_id, control):
  """Append one immutable control observation to its allocated proposal."""
  if (
    not isinstance(control, dict)
    or control.get("control_id") not in CONTROL_IDS
    or control.get("status") not in CONTROL_STATUSES
  ):
    raise ValueError("control result has an invalid id or status")
  matches = [
    slot
    for slot in ledger.get("proposals", [])
    if slot.get("proposal_id") == proposal_id
  ]
  if len(matches) != 1:
    raise ValueError("proposal id is not a unique allocated slot")
  slot = matches[0]
  normalized = {
    **copy.deepcopy(control),
    "proposal_id": proposal_id,
    "kernel_id": slot["kernel_id"],
    "arm": slot["arm"],
  }
  identity = (proposal_id, normalized["control_id"])
  existing = [
    item
    for item in ledger.get("controls", [])
    if (item.get("proposal_id"), item.get("control_id")) == identity
  ]
  if existing:
    if len(existing) != 1 or _canonical_bytes(existing[0]) != _canonical_bytes(
      normalized
    ):
      raise ValueError("proposal already has a different control result")
    return False
  slot.setdefault("control_results", []).append(normalized)
  ledger.setdefault("controls", []).append(normalized)
  return True


def record_reference_result(ledger, proposal_id, reference):
  """Checkpoint the full reference output before queuing compilation."""
  if not isinstance(reference, dict) or reference.get("status") not in {
    "accepted",
    "invalid_on_input",
    "timeout",
  }:
    raise ValueError("reference result has an invalid status")
  matches = [
    slot
    for slot in ledger.get("proposals", [])
    if slot.get("proposal_id") == proposal_id
  ]
  if len(matches) != 1:
    raise ValueError("proposal id is not a unique allocated slot")
  slot = matches[0]
  if reference["status"] == "accepted":
    output_hex = reference.get("output_bytes_hex")
    if not isinstance(output_hex, str):
      raise ValueError("accepted reference must retain complete output bytes")
    try:
      bytes.fromhex(output_hex)
    except ValueError as error:
      raise ValueError("reference output bytes are not valid hex") from error
  if slot.get("reference") is not None:
    if _canonical_bytes(slot["reference"]) != _canonical_bytes(reference):
      raise ValueError("proposal slot already has a different reference result")
    return False
  slot["reference"] = copy.deepcopy(reference)
  return True


def _atomic_json(path, value):
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  payload = (
    json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
    + "\n"
  )
  descriptor, temporary = tempfile.mkstemp(
    prefix=f".{destination.name}.", dir=destination.parent
  )
  try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
      output.write(payload)
      output.flush()
      os.fsync(output.fileno())
    os.replace(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
  finally:
    if os.path.exists(temporary):
      os.unlink(temporary)


def save_ledger(path, ledger):
  _atomic_json(path, ledger)


def save_proposal_record(path, ledger, proposal_id):
  """Write one staged proposal/result and its controls as an atomic sidecar."""
  if not isinstance(ledger, dict) or ledger.get("schema") != LEDGER_SCHEMA:
    raise ValueError("study ledger schema is invalid")
  matches = [
    slot
    for slot in ledger.get("proposals", [])
    if slot.get("proposal_id") == proposal_id
  ]
  if len(matches) != 1:
    raise ValueError("proposal id is not a unique allocated slot")
  record = {
    "schema": f"{LEDGER_SCHEMA}.proposal-record",
    "corpus_sha256": ledger["corpus_sha256"],
    "slot": copy.deepcopy(matches[0]),
  }
  records_dir = Path(f"{Path(path)}.proposals")
  name = f"{_sha256(proposal_id.encode('utf-8'))}.json"
  _atomic_json(records_dir / name, record)


def source_snapshot_manifest(source_root):
  """Hash every source-tree file except VCS, build, and interpreter caches."""
  root = Path(source_root).resolve()
  if not root.is_dir():
    raise ValueError("source snapshot root must be a directory")
  excluded = {
    ".git",
    "build",
    "venv",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
  }
  files = []
  for path in sorted(root.rglob("*")):
    relative = path.relative_to(root)
    if any(part in excluded for part in relative.parts) or not path.is_file():
      continue
    if path.is_symlink():
      raise ValueError(f"source snapshot contains a symlink: {relative.as_posix()}")
    data = path.read_bytes()
    files.append(
      {
        "path": relative.as_posix(),
        "bytes": len(data),
        "sha256": _sha256(data),
      }
    )
  payload = {"schema": "refractir.source-snapshot.v1", "files": files}
  payload["snapshot_sha256"] = _sha256(_canonical_bytes(payload))
  return payload


def save_corpus(path, corpus):
  digest = validate_frozen_corpus(corpus)
  destination = Path(path)
  _atomic_json(destination, corpus)
  _atomic_json(Path(f"{destination}.sha256"), {"corpus_sha256": digest})


def load_corpus(path, *, expected_sha256=None):
  source = Path(path)
  try:
    corpus = json.loads(source.read_text(encoding="utf-8"))
    digest_record = json.loads(Path(f"{source}.sha256").read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(f"cannot read frozen corpus: {error}") from error
  expected = expected_sha256 or digest_record.get("corpus_sha256")
  if digest_record.get("corpus_sha256") != expected:
    raise ValueError("frozen corpus sidecar digest differs from the expected value")
  validate_frozen_corpus(corpus, expected_sha256=expected)
  return corpus


def load_ledger(path, corpus):
  validate_frozen_corpus(corpus)
  try:
    ledger = json.loads(Path(path).read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(f"cannot recover study ledger: {error}") from error
  if not isinstance(ledger, dict) or ledger.get("schema") != LEDGER_SCHEMA:
    raise ValueError("study ledger schema is invalid")
  if ledger.get("corpus_sha256") != corpus["corpus_sha256"]:
    raise ValueError("study ledger belongs to a different frozen corpus")
  expected = allocate_proposals(
    corpus, proposals_per_arm=ledger.get("proposals_per_arm")
  )
  proposals = ledger.get("proposals")
  if not isinstance(proposals, list) or len(proposals) != len(expected):
    raise ValueError("study ledger proposal allocation is incomplete")
  slots_by_id = {slot["proposal_id"]: slot for slot in proposals}
  allocated_by_id = {slot["proposal_id"]: slot for slot in expected}
  records_dir = Path(f"{Path(path)}.proposals")
  if records_dir.is_dir():
    for record_path in sorted(records_dir.glob("*.json")):
      try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
      except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
          f"cannot recover proposal record {record_path.name}: {error}"
        ) from error
      if (
        not isinstance(record, dict)
        or record.get("schema") != f"{LEDGER_SCHEMA}.proposal-record"
        or record.get("corpus_sha256") != corpus["corpus_sha256"]
        or not isinstance(record.get("slot"), dict)
      ):
        raise ValueError("proposal sidecar belongs to a different study")
      saved_slot = record["slot"]
      proposal_id = saved_slot.get("proposal_id")
      if proposal_id not in slots_by_id:
        raise ValueError("proposal sidecar has an unknown allocated id")
      if any(
        saved_slot.get(key) != value
        for key, value in allocated_by_id[proposal_id].items()
      ):
        raise ValueError("proposal sidecar identity differs from frozen allocation")
      target_slot = slots_by_id[proposal_id]
      target_slot["proposal"] = saved_slot.get("proposal")
      target_slot["reference"] = saved_slot.get("reference")
      target_slot["result"] = saved_slot.get("result")
      target_slot["control_results"] = saved_slot.get("control_results", [])
      for key in (
        "gpu_submitted",
        "gpu_device_id",
        "gpu_case_directory",
        "gpu_submitted_at",
        "gpu_command",
      ):
        if key in saved_slot:
          target_slot[key] = saved_slot[key]
  for actual, allocated in zip(proposals, expected, strict=True):
    if any(actual.get(key) != value for key, value in allocated.items()):
      raise ValueError("study ledger proposal slots differ from the frozen allocation")
    if actual.get("result") is not None:
      result = actual["result"]
      if (
        not isinstance(result, dict)
        or result.get("proposal_status") not in PROPOSAL_STATUSES
      ):
        raise ValueError("study ledger contains an invalid proposal result")
    staged = actual.get("proposal")
    if staged is not None:
      if (
        not isinstance(staged, dict)
        or staged.get("proposal_status") not in PROPOSAL_STATUSES
        or staged.get("sampler_version") != allocated["sampler_version"]
      ):
        raise ValueError("study ledger contains an invalid staged input proposal")
      input_hex = staged.get("input_bytes_hex")
      if staged["proposal_status"] == "accepted":
        if not isinstance(input_hex, str):
          raise ValueError("staged accepted proposal omitted its input bytes")
        try:
          bytes.fromhex(input_hex)
        except ValueError as error:
          raise ValueError("staged proposal input bytes are not valid hex") from error
    reference = actual.get("reference")
    if reference is not None:
      if not isinstance(reference, dict) or reference.get("status") not in {
        "accepted",
        "invalid_on_input",
        "timeout",
      }:
        raise ValueError("study ledger contains an invalid reference result")
      if reference["status"] == "accepted":
        output_hex = reference.get("output_bytes_hex")
        if not isinstance(output_hex, str):
          raise ValueError("accepted reference omitted complete output bytes")
        try:
          bytes.fromhex(output_hex)
        except ValueError as error:
          raise ValueError("reference output bytes are not valid hex") from error
    control_results = actual.get("control_results", [])
    if not isinstance(control_results, list):
      raise ValueError("study ledger proposal control_results must be a list")
    for control in control_results:
      if (
        not isinstance(control, dict)
        or control.get("proposal_id") != actual["proposal_id"]
        or control.get("control_id") not in CONTROL_IDS
        or control.get("status") not in CONTROL_STATUSES
      ):
        raise ValueError("study ledger contains an invalid proposal control result")
  ledger["proposal_count_planned"] = len(expected)
  if not isinstance(ledger.get("controls"), list):
    raise ValueError("study ledger controls must be a list")
  for slot in proposals:
    submitted = slot.get("gpu_submitted")
    if submitted is not None and type(submitted) is not bool:
      raise ValueError("study ledger GPU submission marker must be boolean")
    device_id = slot.get("gpu_device_id")
    if device_id is not None and (type(device_id) is not int or device_id < 0):
      raise ValueError("study ledger GPU device id is invalid")
    case_directory = slot.get("gpu_case_directory")
    if case_directory is not None and (
      not isinstance(case_directory, str) or not case_directory
    ):
      raise ValueError("study ledger GPU case directory is invalid")
    if submitted is True and (device_id is None or case_directory is None):
      raise ValueError("submitted GPU proposal omitted its device or case directory")
    gpu_command = slot.get("gpu_command")
    if gpu_command is not None and not isinstance(gpu_command, dict):
      raise ValueError("study ledger GPU command record is invalid")
    if submitted is True and gpu_command is None:
      raise ValueError("submitted GPU proposal omitted its command record")
  merged_controls = {}
  for control in ledger["controls"] + [
    control for slot in proposals for control in slot.get("control_results", [])
  ]:
    identity = (control.get("proposal_id"), control.get("control_id"))
    previous = merged_controls.get(identity)
    if previous is not None and _canonical_bytes(previous) != _canonical_bytes(control):
      raise ValueError("study ledger has conflicting control result records")
    merged_controls[identity] = control
  ledger["controls"] = [merged_controls[key] for key in sorted(merged_controls)]
  return ledger


def summarize_control_results(controls):
  if not isinstance(controls, list):
    raise ValueError("control results must be a list")
  counts = {status: 0 for status in CONTROL_STATUSES}
  by_control = {}
  for control in controls:
    if not isinstance(control, dict):
      raise ValueError("control result must be an object")
    control_id = control.get("control_id")
    status = control.get("status")
    if not isinstance(control_id, str) or not control_id:
      raise ValueError("control result needs a control_id")
    if status not in counts:
      raise ValueError(f"unknown control status: {status!r}")
    counts[status] += 1
    by_control.setdefault(control_id, {name: 0 for name in CONTROL_STATUSES})[
      status
    ] += 1
  applicable = counts["distinguished"] + counts["not_distinguished"]
  return {
    "total": len(controls),
    "counts": counts,
    "by_control": by_control,
    "applicable": applicable,
    "detection_rate": counts["distinguished"] / applicable if applicable else None,
  }


def summarize_input_patterns(proposals):
  """Measure cross-buffer equality, repeated rows, and unique byte patterns."""
  if not isinstance(proposals, list):
    raise ValueError("proposal records must be a list")
  arms = {
    arm: {
      "accepted_proposals": 0,
      "input_buffer_pair_comparisons": 0,
      "equal_input_buffer_pairs": 0,
      "rank2_rows": 0,
      "repeated_rows": 0,
      "unique_input_patterns": 0,
      "unique_output_patterns": 0,
    }
    for arm in ARMS
  }
  input_patterns = {arm: set() for arm in ARMS}
  output_patterns = {arm: set() for arm in ARMS}
  for slot in proposals:
    if not isinstance(slot, dict) or slot.get("arm") not in ARMS:
      raise ValueError("proposal record has an unknown arm")
    arm = slot["arm"]
    proposal = slot.get("proposal") or {}
    if proposal.get("proposal_status") != "accepted":
      continue
    buffers = proposal.get("input_buffers", [])
    if not isinstance(buffers, list):
      raise ValueError("proposal input_buffers must be a list")
    summary = arms[arm]
    summary["accepted_proposals"] += 1
    buffer_words = []
    for buffer in buffers:
      if not isinstance(buffer, dict) or not isinstance(buffer.get("bytes_hex"), str):
        raise ValueError("accepted proposal has a malformed input buffer")
      try:
        data = bytes.fromhex(buffer["bytes_hex"])
      except ValueError as error:
        raise ValueError("accepted proposal input buffer is not hex") from error
      buffer_words.append(data)
      shape = buffer.get("shape")
      if isinstance(shape, list) and len(shape) == 2:
        rows, columns = shape
        if (
          type(rows) is int
          and rows > 0
          and type(columns) is int
          and columns > 0
          and len(data) == rows * columns * 4
        ):
          row_bytes = columns * 4
          row_slices = [
            data[index * row_bytes : (index + 1) * row_bytes] for index in range(rows)
          ]
          summary["rank2_rows"] += rows
          summary["repeated_rows"] += rows - len(set(row_slices))
    for left_index, left in enumerate(buffer_words):
      for right in buffer_words[left_index + 1 :]:
        summary["input_buffer_pair_comparisons"] += 1
        summary["equal_input_buffer_pairs"] += left == right
    combined = proposal.get("input_bytes_hex")
    if combined is None:
      combined = b"".join(buffer_words).hex()
    try:
      input_patterns[arm].add(_sha256(bytes.fromhex(combined)))
    except ValueError as error:
      raise ValueError("proposal input pattern is not hex") from error
    result = slot.get("result") or {}
    output_hex = result.get("output_bytes_hex")
    if output_hex is not None:
      try:
        output_patterns[arm].add(_sha256(bytes.fromhex(output_hex)))
      except ValueError as error:
        raise ValueError("proposal output pattern is not hex") from error
  for arm in ARMS:
    arms[arm]["unique_input_patterns"] = len(input_patterns[arm])
    arms[arm]["unique_output_patterns"] = len(output_patterns[arm])
  return {"arms": arms}


def paired_sensitivity_summary(ledger):
  """Summarize four-proposal detection unions and paired policy gains/losses."""
  if not isinstance(ledger, dict) or ledger.get("schema") != LEDGER_SCHEMA:
    raise ValueError("study ledger schema is invalid")
  proposals = ledger.get("proposals")
  controls = ledger.get("controls")
  if not isinstance(proposals, list) or not isinstance(controls, list):
    raise ValueError("study ledger proposals or controls are missing")
  kernel_ids = sorted({slot.get("kernel_id") for slot in proposals})
  records_by_pair = {}
  by_control = {}
  seen_controls = set()
  for control in controls:
    if not isinstance(control, dict):
      raise ValueError("control result must be an object")
    kernel_id = control.get("kernel_id")
    arm = control.get("arm")
    status = control.get("status")
    control_id = control.get("control_id")
    proposal_id = control.get("proposal_id")
    if (
      kernel_id not in kernel_ids
      or arm not in ARMS
      or status not in CONTROL_STATUSES
      or not isinstance(control_id, str)
      or not control_id
      or not isinstance(proposal_id, str)
      or not proposal_id
    ):
      raise ValueError("control result has an invalid kernel, arm, or status")
    identity = (proposal_id, control_id)
    if identity in seen_controls:
      raise ValueError("study ledger contains a duplicate control result")
    seen_controls.add(identity)
    records_by_pair.setdefault((kernel_id, arm), []).append(control)
    by_control.setdefault(control_id, {}).setdefault((kernel_id, arm), []).append(
      status
    )

  detected_by_pair = {}
  classifiable_by_pair = {}
  complete_by_pair = {}
  for kernel_id in kernel_ids:
    for arm in ARMS:
      slots = [
        slot
        for slot in proposals
        if slot.get("kernel_id") == kernel_id and slot.get("arm") == arm
      ]
      completed_control_slots = {
        (record.get("proposal_id"), record.get("control_id"))
        for record in records_by_pair.get((kernel_id, arm), [])
      }
      expected_control_slots = {
        (slot.get("proposal_id"), control_id)
        for slot in slots
        for control_id in CONTROL_IDS
      }
      complete_by_pair[(kernel_id, arm)] = (
        len(slots) == ledger.get("proposals_per_arm") == 4
        and completed_control_slots == expected_control_slots
      )
      statuses = [
        record["status"] for record in records_by_pair.get((kernel_id, arm), [])
      ]
      applicable = [
        status
        for status in statuses
        if status in {"distinguished", "not_distinguished"}
      ]
      detected_by_pair[(kernel_id, arm)] = "distinguished" in applicable
      classifiable_by_pair[(kernel_id, arm)] = bool(applicable)

  paired_kernels = wins = losses = ties = unclassifiable = incomplete = 0
  detected_counts = {arm: 0 for arm in ARMS}
  per_kernel = []
  for kernel_id in kernel_ids:
    legacy_key = (kernel_id, "legacy_cycle")
    boundary_key = (kernel_id, "boundary")
    legacy_detected = detected_by_pair[legacy_key]
    boundary_detected = detected_by_pair[boundary_key]
    detected_counts["legacy_cycle"] += legacy_detected
    detected_counts["boundary"] += boundary_detected
    both_complete = complete_by_pair[legacy_key] and complete_by_pair[boundary_key]
    if not both_complete:
      incomplete += 1
      outcome = "incomplete"
    elif not classifiable_by_pair[legacy_key] or not classifiable_by_pair[boundary_key]:
      unclassifiable += 1
      outcome = "unclassifiable"
    else:
      paired_kernels += 1
      if boundary_detected and not legacy_detected:
        wins += 1
        outcome = "boundary_win"
      elif legacy_detected and not boundary_detected:
        losses += 1
        outcome = "boundary_loss"
      else:
        ties += 1
        outcome = "tie"
    per_kernel.append(
      {
        "kernel_id": kernel_id,
        "legacy_detected": legacy_detected,
        "boundary_detected": boundary_detected,
        "legacy_complete": complete_by_pair[legacy_key],
        "boundary_complete": complete_by_pair[boundary_key],
        "outcome": outcome,
      }
    )

  per_control = {}
  for control_id, pairs in sorted(by_control.items()):
    record = {
      "legacy_cycle_detected": 0,
      "boundary_detected": 0,
      "legacy_cycle_classified": 0,
      "boundary_classified": 0,
    }
    for kernel_id in kernel_ids:
      for arm in ARMS:
        statuses = pairs.get((kernel_id, arm), [])
        classified = any(
          status in {"distinguished", "not_distinguished"} for status in statuses
        )
        if classified:
          record[f"{arm}_classified"] += 1
          if "distinguished" in statuses:
            record[f"{arm}_detected"] += 1
    for arm in ARMS:
      numerator = record[f"{arm}_detected"]
      denominator = record[f"{arm}_classified"]
      record[f"{arm}_detection_rate"] = numerator / denominator if denominator else None
    per_control[control_id] = record
  return {
    "kernel_count": len(kernel_ids),
    "paired_kernels": paired_kernels,
    "unclassifiable_kernels": unclassifiable,
    "incomplete_kernels": incomplete,
    "observed_four_proposal_union_detections": detected_counts,
    "wins": wins,
    "losses": losses,
    "ties": ties,
    "per_control": per_control,
    "per_kernel": per_kernel,
  }


def summarize_study(ledger):
  if not isinstance(ledger, dict) or ledger.get("schema") != LEDGER_SCHEMA:
    raise ValueError("study ledger schema is invalid")
  proposals = ledger.get("proposals")
  if not isinstance(proposals, list):
    raise ValueError("study ledger proposals are missing")
  completed = [slot for slot in proposals if slot.get("result") is not None]
  by_arm = {
    arm: {"planned": 0, "completed": 0, "proposal_statuses": {}} for arm in ARMS
  }
  gpu_work = {
    "compilation_count": 0,
    "execution_count": 0,
    "compilation_unknown_count": 0,
    "execution_unknown_count": 0,
    "compilation_skipped_count": 0,
    "execution_skipped_count": 0,
    "compilation_timing_observed_count": 0,
    "execution_timing_observed_count": 0,
  }
  cost_s = {
    "reference": 0.0,
    "compiled": 0.0,
    "control": 0.0,
    "triton_compilation": 0.0,
    "gpu_execution": 0.0,
  }
  cost_unknown_count = {stage: 0 for stage in cost_s}
  for slot in proposals:
    arm = slot.get("arm")
    if arm not in by_arm:
      raise ValueError("study ledger has an unknown arm")
    by_arm[arm]["planned"] += 1
    result = slot.get("result")
    if result is None:
      continue
    by_arm[arm]["completed"] += 1
    status = result.get("proposal_status")
    if status not in PROPOSAL_STATUSES:
      raise ValueError("study ledger has an invalid proposal status")
    statuses = by_arm[arm]["proposal_statuses"]
    statuses[status] = statuses.get(status, 0) + 1
    gpu = result.get("gpu") or {}
    for stage in ("compilation", "execution"):
      count_key = f"{stage}_count"
      state_key = f"{stage}_state"
      count = gpu.get(count_key, 0)
      state = gpu.get(state_key)
      if count is None:
        if state not in {None, "unknown"}:
          raise ValueError(f"GPU {count_key} is unknown but its state is not")
        gpu_work[f"{stage}_unknown_count"] += 1
        continue
      if type(count) is not int or count < 0:
        raise ValueError(f"GPU {count_key} must be a nonnegative integer or unknown")
      if state == "unknown":
        raise ValueError(f"GPU {count_key} is known but its state is unknown")
      if state == "skipped" and count != 0:
        raise ValueError(f"GPU {count_key} is skipped but has a nonzero count")
      if state == "attempted" and count == 0:
        raise ValueError(f"GPU {count_key} is attempted but has a zero count")
      if state not in {None, "attempted", "skipped"}:
        raise ValueError(f"GPU {state_key} has an invalid state")
      gpu_work[count_key] += count
      if state == "skipped":
        gpu_work[f"{stage}_skipped_count"] += 1
    timing = gpu.get("worker_timing") or {}
    if not isinstance(timing, dict):
      raise ValueError("GPU worker timing must be an object")
    for timing_key, cost_key, observed_key in (
      ("compilation_s", "triton_compilation", "compilation_timing_observed_count"),
      ("gpu_execution_s", "gpu_execution", "execution_timing_observed_count"),
    ):
      value = timing.get(timing_key)
      if value is None:
        continue
      if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"GPU worker timing {timing_key} is invalid")
      cost_s[cost_key] += value
      gpu_work[observed_key] += 1
    costs = result.get("cost_s") or {}
    if not isinstance(costs, dict):
      raise ValueError("proposal costs must be an object")
    for stage in cost_s:
      value = costs.get(stage, 0.0)
      if value is None:
        cost_unknown_count[stage] += 1
        continue
      if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"proposal cost {stage} must be finite and nonnegative")
      cost_s[stage] += value
  controls = ledger.get("controls", [])
  if not isinstance(controls, list):
    raise ValueError("study ledger controls must be a list")
  for control in controls:
    if not isinstance(control, dict):
      raise ValueError("study ledger control result must be an object")
    value = control.get("elapsed_s")
    if value is None:
      if control.get("status") in {"inapplicable", "not_run"}:
        value = 0.0
      else:
        cost_unknown_count["control"] += 1
        continue
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
      raise ValueError("control cost must be finite and nonnegative")
    cost_s["control"] += value
  return {
    "planned_proposals": len(proposals),
    "completed_proposals": len(completed),
    "pending_proposals": len(proposals) - len(completed),
    "by_arm": by_arm,
    "gpu_work": gpu_work,
    "cost_s": cost_s,
    "cost_unknown_count": cost_unknown_count,
    "input_patterns": summarize_input_patterns(proposals),
    "controls": summarize_control_results(ledger.get("controls", [])),
    "paired_sensitivity": paired_sensitivity_summary(ledger),
  }
