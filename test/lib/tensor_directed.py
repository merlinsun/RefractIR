"""P/D input selection and reference-checked tensor fault models."""

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import struct
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

from test.lib import tensor_case, tensor_layout

ROOT = Path(__file__).resolve().parents[2]
MAX_PROPOSALS = 100
REFERENCE_TIMEOUT_S = 60
_TENSOR_TYPE = re.compile(
  r"(%[A-Za-z_][A-Za-z_0-9]*)\s*:\s*tensor\s*\[([^\]]+)\]\s*(i32|f32)"
)
_SUM = re.compile(
  r"(?m)^(\s*)(%[A-Za-z_][A-Za-z_0-9]*)\s*=\s*tensor\s+sum\s+"
  r"(%[A-Za-z_][A-Za-z_0-9]*)\s*,\s*(\d+)\s*,\s*drop\s*;"
)
_TRANSPOSE = re.compile(
  r"(?m)^(\s*)(%[A-Za-z_][A-Za-z_0-9]*)\s*=\s*tensor\s+transpose\s+"
  r"(%[A-Za-z_][A-Za-z_0-9]*)\s*,\s*\[([^\]]+)\]\s*;"
)
_EXPAND = re.compile(
  r"(?m)^(\s*)(%[A-Za-z_][A-Za-z_0-9]*)\s*=\s*tensor\s+expand\s+"
  r"(%[A-Za-z_][A-Za-z_0-9]*)\s*,\s*(\d+)\s*;"
)
_LOOP_BRANCH = re.compile(
  r"(?m)^\s*br\s+(%[A-Za-z_][A-Za-z_0-9]*)\s*<\s*(\d+)\s*,\s*"
  r"\^([A-Za-z_][A-Za-z_0-9.-]*)\s*,\s*\^([A-Za-z_][A-Za-z_0-9.-]*)\s*;"
)
_BLOCK_LABEL = re.compile(r"(?m)^\^([A-Za-z_][A-Za-z_0-9.-]*):\s*$")
_BRANCH_TERMINATOR = re.compile(r"(?m)^\s*br\s+([^;\n]+);\s*$")
_LABEL_REFERENCE = re.compile(r"\^([A-Za-z_][A-Za-z_0-9.-]*)")
_TENSOR_ADD = re.compile(
  r"(?m)^(\s*)(%[A-Za-z_][A-Za-z_0-9]*)\s*=\s*tensor\s+add\s+"
  r"(%[A-Za-z_][A-Za-z_0-9]*)\s*,\s*(%[A-Za-z_][A-Za-z_0-9]*)\s*;"
)
_TENSOR_OUTPUT = re.compile(
  r"(?m)^\s*fun\s+@[A-Za-z_][A-Za-z_0-9]*\s*\([^)]*\)\s*:\s*"
  r"tensor\s*\[([^\]]+)\]\s*(i32|f32)\s*\{"
)


class ReferenceEvaluationError(RuntimeError):
  """A concrete input or fault model cannot be evaluated by the reference."""


def _shape_tokens(source):
  shapes = {}
  elements = {}
  for match in _TENSOR_TYPE.finditer(source):
    name, dimensions, element = match.groups()
    shapes[name] = tuple(int(token.strip()) for token in dimensions.split(","))
    elements[name] = element
  return shapes, elements


def _drop_axis(shape, axis):
  if axis < 0 or axis >= len(shape):
    return None
  return shape[:axis] + shape[axis + 1 :]


def _insert_axis(shape, axis):
  if axis < 0 or axis > len(shape):
    return None
  return shape[:axis] + (1,) + shape[axis:]


def _inapplicable(name, reason):
  return {"name": name, "status": "inapplicable", "reason": reason}


def _reduction_axis_model(source, shapes, *, exchange=False):
  matches = list(_SUM.finditer(source))
  name = "exchanged_reduction_groups" if exchange else "wrong_reduction_axis"
  if not matches:
    return _inapplicable(name, "source has no tensor reduction")
  match = matches[0]
  _, destination, operand, axis_text = match.groups()
  shape = shapes.get(operand)
  output_shape = shapes.get(destination)
  if shape is None or output_shape is None:
    return _inapplicable(name, "reduction tensor shape is unavailable")
  axis = int(axis_text)
  choices = [
    candidate
    for candidate in range(len(shape))
    if candidate != axis and _drop_axis(shape, candidate) == output_shape
  ]
  if not choices:
    return _inapplicable(
      name,
      "no alternate reduction axis preserves the original logical output shape",
    )
  alternate = choices[-1] if exchange else choices[0]
  replacement = match.group(0).replace(f", {axis}, drop", f", {alternate}, drop")
  mutated = source[: match.start()] + replacement + source[match.end() :]
  return {
    "name": name,
    "status": "applicable",
    "source": mutated,
    "edit": {"from_axis": axis, "to_axis": alternate},
  }


def _transpose_model(source, shapes):
  name = "wrong_transpose_permutation"
  matches = list(_TRANSPOSE.finditer(source))
  if not matches:
    return _inapplicable(name, "source has no tensor transpose")
  match = matches[0]
  _, destination, operand, permutation_text = match.groups()
  shape = shapes.get(operand)
  output_shape = shapes.get(destination)
  if shape is None or output_shape is None:
    return _inapplicable(name, "transpose tensor shape is unavailable")
  permutation = tuple(int(token.strip()) for token in permutation_text.split(","))
  alternatives = [
    candidate
    for candidate in itertools.permutations(range(len(shape)))
    if candidate != permutation
    and tuple(shape[axis] for axis in candidate) == output_shape
  ]
  if not alternatives:
    return _inapplicable(
      name,
      "no alternate transpose permutation preserves the original logical output shape",
    )
  alternate = alternatives[0]
  replacement = match.group(0).replace(
    f"[{permutation_text}]", "[" + ", ".join(map(str, alternate)) + "]"
  )
  mutated = source[: match.start()] + replacement + source[match.end() :]
  return {
    "name": name,
    "status": "applicable",
    "source": mutated,
    "edit": {"from_permutation": permutation, "to_permutation": alternate},
  }


def _broadcast_model(source, shapes):
  name = "incorrect_broadcast_correspondence"
  matches = list(_EXPAND.finditer(source))
  if not matches:
    return _inapplicable(name, "source has no tensor expand")
  match = matches[0]
  _, destination, operand, axis_text = match.groups()
  shape = shapes.get(operand)
  if shape is None:
    return _inapplicable(name, "expanded tensor shape is unavailable")
  axis = int(axis_text)
  target_match = re.search(
    rf"(?m)^\s*%[A-Za-z_][A-Za-z_0-9]*\s*=\s*tensor\s+broadcast\s+"
    rf"{re.escape(destination)}\s*,\s*\[([^\]]+)\]\s*;",
    source,
  )
  if target_match is None:
    return _inapplicable(name, "expanded value has no direct broadcast consumer")
  target_shape = tuple(int(token.strip()) for token in target_match.group(1).split(","))
  alternatives = [
    candidate
    for candidate in range(len(shape) + 1)
    if candidate != axis
    and len(_insert_axis(shape, candidate) or ()) == len(target_shape)
    and all(
      expanded_dimension == 1 or expanded_dimension == target_dimension
      for expanded_dimension, target_dimension in zip(
        _insert_axis(shape, candidate) or (), target_shape
      )
    )
  ]
  if not alternatives:
    return _inapplicable(
      name, "no alternate expand axis is valid for the original broadcast shape"
    )
  alternate = alternatives[0]
  new_shape = _insert_axis(shape, alternate)
  declaration = re.search(
    rf"(?m)^(\s*let\s+mut\s+{re.escape(destination)}\s*:\s*tensor\s*)"
    rf"\[[^\]]+\](\s+(?:i32|f32)\s*=\s*undef\s*;)\s*$",
    source,
  )
  if declaration is None:
    return _inapplicable(name, "expanded destination declaration is unavailable")
  old_shape_text = re.search(r"\[([^\]]+)\]", declaration.group(0)).group(1)
  new_decl = declaration.group(0).replace(
    f"[{old_shape_text}]", "[" + ", ".join(map(str, new_shape)) + "]"
  )
  mutated = source[: declaration.start()] + new_decl + source[declaration.end() :]
  new_expand = match.group(0).replace(f", {axis};", f", {alternate};")
  changed = _EXPAND.search(mutated)
  assert changed is not None
  mutated = mutated[: changed.start()] + new_expand + mutated[changed.end() :]
  return {
    "name": name,
    "status": "applicable",
    "source": mutated,
    "edit": {"from_axis": axis, "to_axis": alternate},
  }


def _contiguous_strides(shape):
  stride = 1
  result = [0] * len(shape)
  for axis in range(len(shape) - 1, -1, -1):
    result[axis] = stride
    stride *= shape[axis]
  return result


def _stride_model(manifest):
  name = "wrong_physical_stride"
  records = manifest.get("inputs") if isinstance(manifest, dict) else None
  if not isinstance(records, list) or not records:
    return _inapplicable(name, "case has no runtime input buffer records")
  for record in records:
    shape = record.get("shape", [])
    if len(shape) < 2:
      continue
    wrong_strides = _contiguous_strides(shape)
    if wrong_strides == record.get("element_strides"):
      wrong_strides = list(reversed(wrong_strides))
    layout = {
      "shape": shape,
      "element_strides": wrong_strides,
      "element_offset": record["element_offset"],
      "storage_span": record["storage_span"],
      "axis_order": record["axis_order"],
    }
    max_address = layout["element_offset"] + sum(
      (size - 1) * stride for size, stride in zip(shape, wrong_strides)
    )
    if max_address >= layout["storage_span"]:
      return {
        "name": name,
        "status": "invalid_model",
        "reason": "the modeled stride reads outside the declared allocation",
        "input": record["name"],
        "strides": wrong_strides,
      }
    return {
      "name": name,
      "status": "applicable",
      "input": record["name"],
      "strides": wrong_strides,
      "layout": layout,
    }
  return _inapplicable(name, "no rank-two-or-higher input can exercise a stride model")


def _blocks(source):
  labels = list(_BLOCK_LABEL.finditer(source))
  result = {}
  for index, match in enumerate(labels):
    end = labels[index + 1].start() if index + 1 < len(labels) else len(source)
    result[match.group(1)] = (match.end(), end, source[match.end() : end])
  return result


def _block_for_position(source, blocks, position):
  for label, (start, end, _text) in blocks.items():
    if start <= position < end:
      return label
  return None


def _validated_counted_loops(source):
  if not _SUM.search(source):
    return []
  blocks = _blocks(source)
  result = []
  branches = list(_LOOP_BRANCH.finditer(source))
  for branch in branches:
    counter, bound_text, body_label, exit_label = branch.groups()
    bound = int(bound_text)
    identifier = re.escape(counter)
    init_pattern = re.compile(rf"(?m)^\s*{identifier}\s*=\s*(-?\d+)\s*;")
    increment_pattern = re.compile(
      rf"(?m)^\s*{identifier}\s*=\s*{identifier}\s*\+\s*1\s*;"
    )
    write_pattern = re.compile(rf"(?m)^\s*{identifier}\s*=")
    declaration_pattern = re.compile(
      rf"(?m)^\s*let\s+mut\s+{identifier}\s*:\s*i32\s*=\s*undef\s*;"
    )
    initializers = list(init_pattern.finditer(source))
    increments = list(increment_pattern.finditer(source))
    writes = list(write_pattern.finditer(source))
    if (
      len(initializers) != 1
      or len(increments) != 1
      or len(writes) != 2
      or declaration_pattern.search(source) is None
      or sum(item.group(1) == counter for item in branches) != 1
    ):
      continue
    start = int(initializers[0].group(1))
    if bound - start < 2 or bound - start > 8:
      continue
    header_label = _block_for_position(source, blocks, branch.start())
    init_label = _block_for_position(source, blocks, initializers[0].start())
    latch_label = _block_for_position(source, blocks, increments[0].start())
    if (
      header_label is None
      or body_label not in blocks
      or exit_label not in blocks
      or init_label is None
      or latch_label is None
      or init_label == latch_label
    ):
      continue
    predecessors = []
    for label, (_begin, _end, block_text) in blocks.items():
      terminators = list(_BRANCH_TERMINATOR.finditer(block_text))
      if not terminators:
        continue
      targets = _LABEL_REFERENCE.findall(terminators[-1].group(1))
      if header_label in targets:
        predecessors.append(label)
    if set(predecessors) != {init_label, latch_label}:
      continue
    backedge = rf"(?m)^\s*br\s+\^{re.escape(header_label)}\s*;\s*$"
    if not re.search(backedge, blocks[init_label][2]) or not re.search(
      backedge, blocks[latch_label][2]
    ):
      continue
    result.append(
      {
        "counter": counter,
        "start": start,
        "bound": bound,
        "header": header_label,
        "body": body_label,
        "latch": latch_label,
        "branch": branch,
        "blocks": blocks,
      }
    )
  return result


def _loop_bound_model(source):
  name = "loop_bound_minus_one"
  loops = _validated_counted_loops(source)
  if not loops:
    return _inapplicable(
      name,
      "source has no counted reduction loop with a preheader reset and unit latch increment",
    )
  loop = loops[0]
  branch = loop["branch"]
  replacement_bound = loop["bound"] - 1
  mutated = source[: branch.start(2)] + str(replacement_bound) + source[branch.end(2) :]
  return {
    "name": name,
    "status": "applicable",
    "source": mutated,
    "edit": {
      "counter": loop["counter"],
      "from_bound": loop["bound"],
      "to_bound": replacement_bound,
    },
  }


def _loop_accumulator_model(source, shapes):
  name = "loop_accumulator_overwrite"
  loops = _validated_counted_loops(source)
  if not loops:
    return _inapplicable(name, "source has no validated counted reduction loop")
  loop_blocks = {label for loop in loops for label in (loop["body"], loop["latch"])}
  blocks = loops[0]["blocks"]
  reduction_destinations = {match.group(2) for match in _SUM.finditer(source)}
  for match in _TENSOR_ADD.finditer(source):
    _indent, destination, left, right = match.groups()
    block = _block_for_position(source, blocks, match.start())
    if (
      block not in loop_blocks
      or destination != left
      or right not in reduction_destinations
      or right == destination
      or shapes.get(destination) != shapes.get(right)
    ):
      continue
    replacement = f"{match.group(1)}{destination} = {right};"
    mutated = source[: match.start()] + replacement + source[match.end() :]
    return {
      "name": name,
      "status": "applicable",
      "source": mutated,
      "edit": {"accumulator": destination, "reduction_result": right},
    }
  return _inapplicable(
    name, "no same-shaped tensor reduction is accumulated in a counted loop body"
  )


def _tensor_output_signature(source, manifest):
  output = manifest.get("output") if isinstance(manifest, dict) else None
  if isinstance(output, dict):
    shape = output.get("shape")
    dtype = output.get("dtype")
    if isinstance(shape, list) and dtype in {"i32", "f32"}:
      return tuple(shape), dtype
  match = _TENSOR_OUTPUT.search(source)
  if match is None:
    return None, None
  return tuple(int(token.strip()) for token in match.group(1).split(",")), match.group(
    2
  )


def _reduction_group_model(source, manifest):
  name = "swapped_reduction_groups"
  if not _SUM.search(source):
    return _inapplicable(name, "source has no tensor reduction")
  shape, dtype = _tensor_output_signature(source, manifest)
  if shape is None:
    return _inapplicable(name, "tensor output shape is unavailable")
  if not shape or shape[0] < 2:
    return _inapplicable(name, "output has fewer than two reduction groups")
  return {
    "name": name,
    "status": "applicable",
    "output_transform": "reverse_first_axis_groups",
    "output_shape": list(shape),
    "output_dtype": dtype,
    "edit": {"axis": 0, "group_order": "reverse"},
  }


def _reverse_first_axis_groups(data, shape):
  if not shape or shape[0] < 2 or len(data) != math.prod(shape) * 4:
    raise ValueError("reduction output cannot be partitioned into at least two groups")
  group_bytes = len(data) // shape[0]
  result = bytearray(len(data))
  for destination in range(shape[0]):
    source = shape[0] - destination - 1
    result[destination * group_bytes : (destination + 1) * group_bytes] = data[
      source * group_bytes : (source + 1) * group_bytes
    ]
  return bytes(result)


def build_fault_models(source, manifest=None):
  """Build only mutations whose modeled result can retain the source shape."""
  shapes, _ = _shape_tokens(source)
  return [
    _reduction_axis_model(source, shapes),
    _reduction_axis_model(source, shapes, exchange=True),
    _transpose_model(source, shapes),
    _broadcast_model(source, shapes),
    _stride_model(manifest),
    _loop_bound_model(source),
    _loop_accumulator_model(source, shapes),
    _reduction_group_model(source, manifest),
  ]


def _fraction_pair(value):
  if not isinstance(value, (list, tuple)) or len(value) != 2:
    raise ValueError("rational tolerance must be a numerator/denominator pair")
  numerator, denominator = value
  if type(numerator) is not int or type(denominator) is not int:
    raise ValueError("rational tolerance components must be integers")
  if numerator < 0 or denominator <= 0:
    raise ValueError("rational tolerance must be nonnegative with positive denominator")
  return Fraction(numerator, denominator)


def compare_model_output(
  modeled,
  reference,
  dtype,
  shape,
  comparison,
  *,
  modeled_shape=None,
  modeled_dtype=None,
):
  """Apply exact policy or the strict >4× numerical distinction margin."""
  if modeled_shape is not None and list(modeled_shape) != list(shape):
    return {
      "status": "inapplicable",
      "distinguished": False,
      "reason": "modeled output shape differs from the original shape",
    }
  if modeled_dtype is not None and modeled_dtype != dtype:
    return {
      "status": "inapplicable",
      "distinguished": False,
      "reason": "modeled output dtype differs from the original dtype",
    }
  count = math.prod(shape) if shape else 1
  if len(modeled) != count * 4 or len(reference) != count * 4:
    return {
      "status": "inapplicable",
      "distinguished": False,
      "reason": "modeled output byte length differs from the original shape",
    }
  if comparison.get("mode") == "exact":
    if dtype == "i32":
      left = struct.unpack(f"<{count}i", modeled)
      right = struct.unpack(f"<{count}i", reference)
      indices = [
        index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]
      ]
    elif dtype == "f32":
      left = struct.unpack(f"<{count}I", modeled)
      right = struct.unpack(f"<{count}I", reference)
      indices = [
        index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]
      ]
    else:
      return {
        "status": "invalid_model",
        "distinguished": False,
        "reason": "unsupported exact output dtype",
      }
    return {
      "status": "applicable_distinguished"
      if indices
      else "applicable_not_distinguished",
      "defined": True,
      "distinguished": bool(indices),
      "differing_indices": indices,
      "comparison": "exact",
    }

  if dtype != "f32" or comparison.get("mode") != "numeric":
    return {
      "status": "invalid_model",
      "distinguished": False,
      "reason": "numerical distinction requires f32 and numeric comparison",
    }
  actual = struct.unpack(f"<{count}f", modeled)
  expected = struct.unpack(f"<{count}f", reference)
  if any(not math.isfinite(value) for value in (*actual, *expected)):
    return {
      "status": "invalid_model",
      "defined": False,
      "distinguished": False,
      "reason": "modeled or reference output is non-finite",
    }
  atol = _fraction_pair(comparison["atol"])
  rtol = _fraction_pair(comparison["rtol"])
  differing = []
  margins = []
  for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
    reference_value = Fraction.from_float(expected_value)
    difference = abs(Fraction.from_float(actual_value) - reference_value)
    allowance = atol + rtol * abs(reference_value)
    threshold = 4 * allowance
    margins.append(
      {
        "index": index,
        "difference": [difference.numerator, difference.denominator],
        "four_times_allowance": [threshold.numerator, threshold.denominator],
      }
    )
    if difference > threshold:
      differing.append(index)
  return {
    "status": "applicable_distinguished"
    if differing
    else "applicable_not_distinguished",
    "defined": True,
    "distinguished": bool(differing),
    "differing_indices": differing,
    "margins": margins,
    "comparison": "numeric_gt_4x_allowance",
  }


def _predicate_value(value):
  if isinstance(value, dict):
    return bool(value.get("passed", value.get("distinguished", False))), value
  return bool(value), {"passed": bool(value)}


def _distinction_objective(value, target_model):
  """Return the selected D result without letting unrelated models satisfy it."""
  if not isinstance(value, dict):
    if target_model is not None:
      raise ValueError("targeted distinction evaluator omitted per-model results")
    distinguished = bool(value)
    return distinguished, None, distinguished
  if target_model is None:
    records = value.get("models")
    applicable = isinstance(records, list) and any(
      record.get("applicability") == "applicable" for record in records
    )
    return bool(value.get("distinguished", False)), None, applicable
  records = value.get("models")
  if not isinstance(records, list):
    raise ValueError("distinction evaluator omitted per-model results")
  matches = [record for record in records if record.get("name") == target_model]
  if len(matches) != 1:
    raise ValueError(
      f"distinction evaluator must return exactly one result for {target_model}"
    )
  record = matches[0]
  applicable = record.get("applicability") == "applicable"
  return (
    applicable and record.get("distinguished") is True,
    record,
    applicable,
  )


def select_input_candidate(
  candidates,
  *,
  arm,
  property_validator,
  distinction_evaluator,
  proposal_limit=MAX_PROPOSALS,
  target_model=None,
):
  """Ordinary accepts P; directed accepts P∧D over the same proposal stream.

  When ``target_model`` is set, D means that specific model is distinguished;
  another model's result cannot satisfy the directed objective.
  """
  if arm not in {"ordinary", "directed"}:
    raise ValueError("arm must be ordinary or directed")
  if type(proposal_limit) is not int or not 1 <= proposal_limit <= MAX_PROPOSALS:
    raise ValueError(f"proposal_limit must be in [1, {MAX_PROPOSALS}]")
  if target_model is not None and (
    not isinstance(target_model, str) or not target_model
  ):
    raise ValueError("target_model must be a nonempty fault-model name")
  attempts = []
  property_count = 0
  objective_applicable = False
  objective_invalid = False
  selection_started = time.perf_counter()
  for index, candidate in enumerate(candidates):
    if index >= proposal_limit:
      break
    property_value = property_validator(candidate)
    property_passed, property_record = _predicate_value(property_value)
    if property_passed:
      property_count += 1
      distinction_value = distinction_evaluator(candidate)
      distinguished, objective_record, target_applicable = _distinction_objective(
        distinction_value, target_model
      )
      if target_model is None:
        distinction_record = (
          distinction_value
          if isinstance(distinction_value, dict)
          else {"distinguished": distinguished}
        )
        objective_applicable = objective_applicable or target_applicable
      else:
        distinction_record = {
          "distinguished": distinguished,
          "target_model": target_model,
          "models": [objective_record],
          "reference_evaluations": distinction_value.get("reference_evaluations", 0),
          "reference_evaluation_s": distinction_value.get(
            "reference_evaluation_s", 0.0
          ),
        }
        objective_applicable = objective_applicable or target_applicable
        if objective_record.get("applicability") == "invalid_model":
          objective_invalid = True
    else:
      distinguished = False
      distinction_record = {"distinguished": False, "status": "not_evaluated"}
    attempts.append(
      {
        "index": index,
        "property_passed": property_passed,
        "property": property_record,
        "distinguished": distinguished,
        "distinction": distinction_record,
        "objective_applicable": target_applicable if property_passed else None,
      }
    )
    if property_passed and (arm == "ordinary" or distinguished):
      return {
        "status": "accepted",
        "arm": arm,
        "target_model": target_model,
        "selected": candidate,
        "property_passed": True,
        "distinguished": distinguished,
        "objective_applicable": target_applicable,
        "proposal_count": index + 1,
        "attempts": attempts,
        "elapsed_s": time.perf_counter() - selection_started,
      }
  status = "proposal_exhausted"
  if target_model is not None:
    if property_count == 0:
      status = "property_exhausted"
    elif not objective_applicable:
      status = (
        "model_invalid_for_proposals" if objective_invalid else "no_applicable_model"
      )
  return {
    "status": status,
    "arm": arm,
    "target_model": target_model,
    "selected": None,
    "property_passed": any(attempt["property_passed"] for attempt in attempts),
    "distinguished": any(
      attempt["property_passed"] and attempt["distinguished"] for attempt in attempts
    ),
    "objective_applicable": objective_applicable,
    "proposal_count": len(attempts),
    "attempts": attempts,
    "elapsed_s": time.perf_counter() - selection_started,
  }


def _run_reference(source, input_bytes, *, entry, symiri, profile, stable):
  with tempfile.TemporaryDirectory(prefix="tensor-directed-") as directory:
    root = Path(directory)
    source_path = root / "program.sir"
    input_path = root / "inputs.bin"
    output_path = root / "expected.bin"
    source_path.write_text(source, encoding="utf-8")
    input_path.write_bytes(input_bytes)
    command = [
      str(symiri),
      "--tensor-profile-version",
      "v2" if profile == tensor_case.PROFILE_V2 else "v1",
      "--entry",
      entry,
      "--tensor-input",
      str(input_path),
      "--tensor-output",
      str(output_path),
    ]
    if stable:
      command.append("--tensor-reference-stability")
    command.append(str(source_path))
    try:
      result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=REFERENCE_TIMEOUT_S,
      )
    except subprocess.TimeoutExpired as error:
      raise ReferenceEvaluationError("reference_timeout") from error
    except OSError as error:
      raise ReferenceEvaluationError(f"reference_unavailable: {error}") from error
    if result.returncode != 0 or not output_path.is_file():
      detail = (result.stderr or result.stdout).strip()
      raise ReferenceEvaluationError(detail or "reference evaluation failed")
    return output_path.read_bytes()


def _proposals(manifest, seed, count):
  if type(count) is not int or not 1 <= count <= MAX_PROPOSALS:
    raise ValueError(f"proposal count must be in [1, {MAX_PROPOSALS}]")
  inputs = manifest["inputs"]
  total_bytes = sum(record["byte_length"] for record in inputs)
  for proposal_index in range(count):
    random_source = random.Random((seed + proposal_index * 0x9E3779B9) & 0xFFFFFFFF)
    data = bytearray(total_bytes)
    for record in inputs:
      count_elements = math.prod(record["shape"]) if record["shape"] else 1
      begin = record["byte_offset"]
      dtype = record["dtype"]
      for element_index in range(count_elements):
        value = random_source.randint(1, 4)
        format_code = "i" if dtype == "i32" else "f"
        struct.pack_into(f"<{format_code}", data, begin + element_index * 4, value)
    yield bytes(data)


def _logical_output_shape(source, entry):
  match = re.search(
    rf"(?m)^\s*fun\s+@{re.escape(entry)}\([^)]*\)\s*:\s*"
    rf"tensor\s*\[([^\]]+)\]\s*(i32|f32)\s*\{{",
    source,
  )
  if match is None:
    return [], None
  return [int(token.strip()) for token in match.group(1).split(",")], match.group(2)


def _apply_stride_model(model, input_bytes, manifest):
  if model.get("status") != "applicable":
    raise ValueError(model.get("reason", "stride model is not applicable"))
  record = next(item for item in manifest["inputs"] if item["name"] == model["input"])
  byte_offset = record["byte_offset"]
  byte_length = record["byte_length"]
  logical = input_bytes[byte_offset : byte_offset + byte_length]
  dtype = record["dtype"]
  gap = bytes.fromhex(manifest["execution"]["gap_sentinel_by_dtype"][dtype])
  element_size = 4
  layout = {
    "shape": record["shape"],
    "element_strides": record["element_strides"],
    "element_offset": record["element_offset"],
    "storage_span": record["storage_span"],
    "axis_order": record["axis_order"],
  }
  physical = tensor_layout.stage_buffer(
    logical, layout, element_size=element_size, gap_bytes=gap
  )
  strides = model["strides"]
  coordinates = itertools.product(*(range(size) for size in record["shape"]))
  wrong_logical = bytearray()
  for coordinate in coordinates:
    address = record["element_offset"] + sum(
      value * stride for value, stride in zip(coordinate, strides)
    )
    if address < 0 or address >= record["storage_span"]:
      raise ValueError("modeled stride reads outside the declared allocation")
    begin = address * element_size
    wrong_logical.extend(physical[begin : begin + element_size])
  if dtype == "f32":
    values = struct.unpack(f"<{len(wrong_logical) // 4}f", wrong_logical)
    if any(not math.isfinite(value) for value in values):
      raise ValueError("modeled stride reads a non-finite gap sentinel")
  result = bytearray(input_bytes)
  result[byte_offset : byte_offset + byte_length] = wrong_logical
  return bytes(result)


def _run_models(models, source, input_bytes, reference, manifest, symiri):
  output = manifest["output"]
  entry = manifest["entry"]
  stable = output["dtype"] == "f32"
  result_records = []
  reference_evaluations = 0
  reference_evaluation_s = 0.0
  for model in models:
    base = {
      "name": model["name"],
      "applicability": model["status"],
      "defined": False,
      "output_dtype": None,
      "output_shape": None,
      "distinguished": False,
      "differing_indices": [],
      "modeled_result_sha256": None,
    }
    if model["status"] != "applicable":
      base["reason"] = model.get("reason")
      result_records.append(base)
      continue
    modeled_source = model.get("source", source)
    modeled_inputs = input_bytes
    if model.get("output_transform") == "reverse_first_axis_groups":
      try:
        modeled = _reverse_first_axis_groups(reference, output["shape"])
      except ValueError as error:
        base["applicability"] = "invalid_model"
        base["reason"] = str(error)
        result_records.append(base)
        continue
    elif "strides" in model:
      try:
        modeled_inputs = _apply_stride_model(model, input_bytes, manifest)
      except (ValueError, tensor_layout.LayoutError) as error:
        base["applicability"] = "invalid_model"
        base["reason"] = str(error)
        result_records.append(base)
        continue
      started = time.perf_counter()
      reference_evaluations += 1
      try:
        modeled = _run_reference(
          modeled_source,
          modeled_inputs,
          entry=entry,
          symiri=symiri,
          profile=manifest["profile"],
          stable=stable,
        )
      except ReferenceEvaluationError as error:
        base["applicability"] = "invalid_model"
        base["reason"] = str(error)
        result_records.append(base)
        continue
      finally:
        reference_evaluation_s += time.perf_counter() - started
    else:
      started = time.perf_counter()
      reference_evaluations += 1
      try:
        modeled = _run_reference(
          modeled_source,
          modeled_inputs,
          entry=entry,
          symiri=symiri,
          profile=manifest["profile"],
          stable=stable,
        )
      except ReferenceEvaluationError as error:
        base["applicability"] = "invalid_model"
        base["reason"] = str(error)
        result_records.append(base)
        continue
      finally:
        reference_evaluation_s += time.perf_counter() - started
    shape, dtype = _logical_output_shape(modeled_source, entry)
    comparison = compare_model_output(
      modeled,
      reference,
      output["dtype"],
      output["shape"],
      manifest["comparison"],
      modeled_shape=shape,
      modeled_dtype=dtype,
    )
    comparison_status = comparison.get("status")
    applicability = (
      comparison_status
      if comparison_status in {"inapplicable", "invalid_model"}
      else "applicable"
    )
    base.update(
      {
        "applicability": applicability,
        "defined": comparison.get("defined", comparison_status != "invalid_model"),
        "output_dtype": dtype,
        "output_shape": shape,
        "distinguished": comparison.get("distinguished", False),
        "differing_indices": comparison.get("differing_indices", []),
        "modeled_result_sha256": hashlib.sha256(modeled).hexdigest(),
        "comparison": comparison,
      }
    )
    if comparison.get("reason"):
      base["reason"] = comparison["reason"]
    result_records.append(base)
  return {
    "distinguished": any(
      record.get("applicability") == "applicable" and record.get("distinguished")
      for record in result_records
    ),
    "models": result_records,
    "reference_evaluations": reference_evaluations,
    "reference_evaluation_s": reference_evaluation_s,
  }


def _write_atomic(path, data):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
  try:
    with os.fdopen(descriptor, "wb") as output:
      output.write(data)
    os.replace(temporary, path)
  finally:
    if os.path.exists(temporary):
      os.unlink(temporary)


def _canonical_json(value):
  return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _update_file_digest(manifest, name, data):
  manifest["files"][name] = {
    "bytes": len(data),
    "sha256": hashlib.sha256(data).hexdigest(),
  }


def prepare_case_arm(
  case_path,
  *,
  arm,
  seed,
  proposal_limit=MAX_PROPOSALS,
  symiri=None,
  target_model=None,
):
  """Select and materialize one ordinary or fault-directed v2 composition input."""
  root = Path(case_path).resolve()
  symiri = Path(symiri or tensor_case.DEFAULT_SYMIRI).resolve()
  case = tensor_case.load_case(root, symiri=symiri)
  manifest = json.loads(json.dumps(case.manifest))
  if manifest["profile"] != tensor_case.PROFILE_V2:
    raise ValueError("P/D input selection requires a tensor-v2 case")
  if manifest.get("coverage", {}).get("primary_stratum") not in {
    "multi-input-composition",
    "loop-reduction",
  }:
    raise ValueError(
      "P/D input selection requires a composition or loop-reduction case"
    )

  source = case.program.decode("utf-8")
  models = build_fault_models(source, manifest)
  target_record = None
  if target_model is not None:
    matches = [model for model in models if model["name"] == target_model]
    if len(matches) != 1:
      raise ValueError(f"unknown fault model: {target_model}")
    target_record = matches[0]
    if target_record["status"] != "applicable":
      selection_record = {
        "schema": "refractir.tensor-input-selection.v1",
        "arm": arm,
        "seed": seed,
        "input_domain": "finite-integer-valued-elements-in-[1,4]",
        "input_domain_id": "finite-integer-valued-elements-in-[1,4]",
        "proposal_mechanism_id": "python-mt19937-seed-plus-golden-ratio-step-v1",
        "seed_schedule_id": f"single-seed-{seed}",
        "P": "defined-stable-reference",
        "D": f"fault-model:{target_model}",
        "target_model": target_model,
        "proposal_limit": proposal_limit,
        "proposal_budget": proposal_limit,
        "proposal_count": 0,
        "status": "no_applicable_model",
        "property_passed": False,
        "distinguished": False,
        "objective_applicable": False,
        "attempts": [],
        "cost": {
          "selection_elapsed_s": 0.0,
          "selection_s": 0.0,
          "property_reference_evaluations": 0,
          "property_reference_s": 0.0,
          "model_reference_evaluations": 0,
          "model_reference_s": 0.0,
        },
      }
      _write_atomic(
        root / "selection.json",
        _canonical_json(selection_record).encode("utf-8"),
      )
      return selection_record
  cache = {}
  stable = manifest["output"]["dtype"] == "f32"
  property_reference_evaluations = 0
  property_reference_evaluation_s = 0.0

  def validate_property(input_bytes):
    nonlocal property_reference_evaluations, property_reference_evaluation_s
    started = time.perf_counter()
    property_reference_evaluations += 1
    try:
      expected = _run_reference(
        source,
        input_bytes,
        entry=manifest["entry"],
        symiri=symiri,
        profile=manifest["profile"],
        stable=stable,
      )
    except ReferenceEvaluationError as error:
      return {
        "passed": False,
        "reason": str(error),
        "reference_evaluations": 1,
        "reference_evaluation_s": time.perf_counter() - started,
      }
    finally:
      property_reference_evaluation_s += time.perf_counter() - started
    cache[input_bytes] = expected
    return {
      "passed": True,
      "defined": True,
      "stable": stable,
      "reference_evaluations": 1,
      "reference_evaluation_s": time.perf_counter() - started,
    }

  def evaluate_distinction(input_bytes):
    reference = cache[input_bytes]
    evaluated_models = [target_record] if target_record is not None else models
    return _run_models(
      evaluated_models, source, input_bytes, reference, manifest, symiri
    )

  selection = select_input_candidate(
    _proposals(manifest, seed, proposal_limit),
    arm=arm,
    property_validator=validate_property,
    distinction_evaluator=evaluate_distinction,
    proposal_limit=proposal_limit,
    target_model=target_model,
  )
  model_reference_evaluations = sum(
    attempt.get("distinction", {}).get("reference_evaluations", 0)
    for attempt in selection["attempts"]
  )
  model_reference_s = sum(
    attempt.get("distinction", {}).get("reference_evaluation_s", 0.0)
    for attempt in selection["attempts"]
  )
  selection_elapsed_s = selection.get("elapsed_s", 0.0)
  selection_record = {
    "schema": "refractir.tensor-input-selection.v1",
    "arm": arm,
    "seed": seed,
    "input_domain": "finite-integer-valued-elements-in-[1,4]",
    "input_domain_id": "finite-integer-valued-elements-in-[1,4]",
    "proposal_mechanism_id": "python-mt19937-seed-plus-golden-ratio-step-v1",
    "seed_schedule_id": f"single-seed-{seed}",
    "P": "defined-stable-reference",
    "D": f"fault-model:{target_model}"
    if target_model is not None
    else "at-least-one-applicable-model-distinguished",
    "target_model": target_model,
    "proposal_limit": proposal_limit,
    "proposal_budget": proposal_limit,
    "proposal_count": selection["proposal_count"],
    "status": selection["status"],
    "property_passed": selection["property_passed"],
    "distinguished": selection["distinguished"],
    "objective_applicable": selection.get("objective_applicable"),
    "attempts": selection["attempts"],
    "cost": {
      "selection_elapsed_s": selection_elapsed_s,
      "selection_s": max(
        0.0,
        selection_elapsed_s - property_reference_evaluation_s - model_reference_s,
      ),
      "property_reference_evaluations": property_reference_evaluations,
      "property_reference_s": property_reference_evaluation_s,
      "model_reference_evaluations": model_reference_evaluations,
      "model_reference_s": model_reference_s,
    },
  }
  if selection["status"] != "accepted":
    _write_atomic(
      root / "selection.json",
      _canonical_json(selection_record).encode("utf-8"),
    )
    return selection_record

  selected = selection["selected"]
  expected = cache[selected]
  old_inputs = case.inputs
  _write_atomic(root / "inputs.bin", selected)
  _write_atomic(root / "expected.bin", expected)
  _update_file_digest(manifest, "inputs.bin", selected)
  _update_file_digest(manifest, "expected.bin", expected)
  manifest["property"].update(
    {
      "P": selection_record["P"],
      "D": selection_record["D"],
      "selection_arm": arm,
      "validation": {"passed": True, "defined": True, "stable": stable},
    }
  )
  chosen_attempt = selection["attempts"][-1]
  model_records = chosen_attempt["distinction"].get("models", [])
  manifest["coverage"]["input_behavior"] = {
    "property_passed": True,
    "property_id": selection_record["P"],
    "distinguishing_objective": selection_record["D"],
    "selection_arm": arm,
    "proposal_count": selection["proposal_count"],
    "distinguished": selection["distinguished"],
    "fault_models": model_records,
  }
  manifest["evidence"] = {
    "kind": "error_directed" if arm == "directed" else manifest["evidence"]["kind"],
    "status": "error_directed_checked"
    if arm == "directed"
    else manifest["evidence"]["status"],
    "artifacts": manifest["evidence"]["artifacts"],
  }
  manifest["reference"]["defined"] = True
  manifest["reference"]["stable"] = True
  selection_payload = {
    **selection_record,
    "selected_inputs_sha256": hashlib.sha256(selected).hexdigest(),
    "expected_sha256": hashlib.sha256(expected).hexdigest(),
    "previous_inputs_sha256": hashlib.sha256(old_inputs).hexdigest(),
    "models": model_records,
  }
  selection_bytes = _canonical_json(selection_payload).encode("utf-8")
  _write_atomic(root / "selection.json", selection_bytes)
  _update_file_digest(manifest, "selection.json", selection_bytes)
  if "selection.json" not in manifest["evidence"]["artifacts"]:
    manifest["evidence"]["artifacts"].append("selection.json")
  _write_atomic(root / "case.json", _canonical_json(manifest).encode("utf-8"))
  tensor_case.load_case(root, symiri=symiri)
  selection_record.update(
    {
      key: selection_payload[key]
      for key in (
        "selected_inputs_sha256",
        "expected_sha256",
        "previous_inputs_sha256",
      )
    }
  )
  return selection_record


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--case", required=True)
  parser.add_argument("--arm", choices=("ordinary", "directed"), required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--proposal-limit", type=int, default=MAX_PROPOSALS)
  parser.add_argument("--target-model")
  parser.add_argument("--symiri", default=str(tensor_case.DEFAULT_SYMIRI))
  args = parser.parse_args(argv)
  result = prepare_case_arm(
    args.case,
    arm=args.arm,
    seed=args.seed,
    proposal_limit=args.proposal_limit,
    symiri=args.symiri,
    target_model=args.target_model,
  )
  print(_canonical_json(result), end="")
  return 0 if result["status"] == "accepted" else 1


if __name__ == "__main__":
  sys.exit(main())
