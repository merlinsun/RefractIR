"""Summarize observable Triton IR transformations without inferring semantics."""

import collections
import re

_SSA = re.compile(r"%[-a-zA-Z$._0-9]+")
_LOOP = re.compile(r"\b(?:scf\.(?:for|while)|tt\.(?:loop|loop_unroll))\b")
_CONVERSION = re.compile(r"\b(?:ttg\.)?convert_layout\b")
_LOCATION_DEFINITION = re.compile(
  r'^\s*#loc(\d+)\s*=\s*loc\("(?:[^"\\]|\\.)*":(\d+):(\d+)\)'
)
_ANY_LOCATION_DEFINITION = re.compile(r"^\s*#loc(\d+)\s*=\s*loc\((.*)\)")
_LOCATION_USE = re.compile(r"\bloc\(#loc(\d+)\)")
_LOCATION_REFERENCE = re.compile(r"#loc(\d+)")
_TENSOR_ENCODING = re.compile(r"tensor<[^>]*,\s*(#[a-zA-Z0-9_.-]+)>")
_INLINE_LOCATION = re.compile(r'\bloc\("(?:[^"\\]|\\.)*":(\d+):(\d+)\)')


def _location_definitions(text):
  direct, parents = {}, {}
  for line in text.splitlines():
    direct_match = _LOCATION_DEFINITION.match(line)
    if direct_match:
      direct[direct_match[1]] = int(direct_match[2])
      continue
    definition = _ANY_LOCATION_DEFINITION.match(line)
    if definition:
      references = _LOCATION_REFERENCE.findall(definition[2])
      if references:
        parents[definition[1]] = references[-1]
  return direct, parents


def _source_line(instruction, locations):
  direct, parents = locations
  indirect = _LOCATION_USE.search(instruction)
  if indirect:
    label = indirect[1]
    visited = set()
    while label not in direct and label in parents and label not in visited:
      visited.add(label)
      label = parents[label]
    return direct.get(label)
  direct = _INLINE_LOCATION.search(instruction)
  return int(direct[1]) if direct else None


def _operation_rhs(instruction):
  if "=" not in instruction:
    return None
  _result, rhs = instruction.split("=", 1)
  rhs = re.sub(r"\bloc\([^)]*\)", "", rhs)
  rhs = re.sub(r"\s+//.*$", "", rhs).strip()
  if not re.search(r"\b(?:tt|ttg|arith|math|scf)\.[a-zA-Z_][\w.]*", rhs):
    return None
  return re.sub(r"\s+", " ", rhs)


def _stage_observation(text):
  lines = text.splitlines()
  locations = _location_definitions(text)
  definitions = {}
  instructions = []
  for line in lines:
    match = re.match(r"^\s*(%[-a-zA-Z$._0-9]+)\s*=\s*(.*)$", line)
    if not match:
      continue
    value, rhs = match.groups()
    definitions[value] = rhs
    instructions.append((value, rhs))

  uses = collections.Counter()
  for _value, rhs in instructions:
    uses.update(_SSA.findall(rhs))
  reused = [
    {"value": value, "use_count": uses[value]}
    for value in sorted(definitions)
    if uses[value] > 1
  ]

  duplicate_rhs = collections.defaultdict(list)
  operation_counts = collections.Counter()
  for value, rhs in instructions:
    operation = _operation_rhs(f"{value} = {rhs}")
    if operation is None:
      continue
    duplicate_rhs[operation].append(value)
    opcode = re.match(r"((?:tt|ttg|arith|math|scf)\.[a-zA-Z_][\w.]*)", operation)
    if opcode:
      operation_counts[opcode[1]] += 1
  duplicates = [
    {"expression": expression, "values": values, "count": len(values)}
    for expression, values in sorted(duplicate_rhs.items())
    if len(values) > 1
  ]

  conversions = []
  for line_number, line in enumerate(lines, start=1):
    if not _CONVERSION.search(line):
      continue
    result_match = re.match(r"^\s*(%[-a-zA-Z$._0-9]+)\s*=", line)
    rhs = line.split("=", 1)[1] if "=" in line else line
    operands = _SSA.findall(rhs)
    encodings = _TENSOR_ENCODING.findall(line)
    source_line = _source_line(line, locations)
    conversions.append(
      {
        "result": result_match[1] if result_match else None,
        "source_value": operands[0] if operands else None,
        "source_line": source_line,
        "ir_line": line_number,
        "source_operation": None,
        "source_layout": encodings[0] if encodings else None,
        "target_layout": encodings[-1] if encodings else None,
        "layout_changed": len(encodings) >= 2 and encodings[0] != encodings[-1],
        "instruction": line.strip(),
      }
    )

  loops = _LOOP.findall(text)
  return {
    "defined_value_count": len(definitions),
    "operation_counts": dict(sorted(operation_counts.items())),
    "reused_values": reused,
    "duplicate_expressions": duplicates,
    "loop_construct_count": len(loops),
    "loop_constructs": loops,
    "layout_conversions": conversions,
  }


def observe_compiler_ir(
  artifacts,
  *,
  source_lines=None,
  expected_source_operations=(),
  exact_source_operations=False,
  expected_loop=False,
):
  """Measure conversions, value fanout, duplicate work, and loop form per IR stage.

  `source_lines` maps source line numbers from compiler locations to stable
  logical operation names. Duplicate expressions are evidence of repeated
  IR work, not proof that a particular source-level value was rematerialized.
  """
  if not isinstance(artifacts, dict) or not artifacts:
    raise ValueError("compiler artifacts must be a nonempty stage mapping")
  if type(expected_loop) is not bool or type(exact_source_operations) is not bool:
    raise ValueError("loop and exact-conversion expectations must be boolean")
  if source_lines is None:
    source_lines = {}
  if not isinstance(source_lines, dict) or any(
    type(line) is not int or not isinstance(name, str) or not name
    for line, name in source_lines.items()
  ):
    raise ValueError("source_lines must map integer lines to operation names")
  if not isinstance(expected_source_operations, (tuple, list, set)) or any(
    not isinstance(name, str) or not name for name in expected_source_operations
  ):
    raise ValueError("expected source operations must be nonempty names")

  stages = {}
  conversions = []
  for stage, text in artifacts.items():
    if not isinstance(stage, str) or not stage:
      raise ValueError("compiler stage names must be nonempty strings")
    if not isinstance(text, str) or not text.strip():
      raise ValueError(f"compiler artifact {stage!r} must contain IR text")
    observation = _stage_observation(text)
    for conversion in observation["layout_conversions"]:
      conversion["stage"] = stage
      conversion["source_operation"] = source_lines.get(conversion["source_line"])
      conversions.append(conversion)
    if stage.lower() in {"ptx", "sass"}:
      observation["warp_shuffle_count"] = len(re.findall(r"\bshfl(?:\.sync)?\.", text))
      observation["shared_memory_transfer_count"] = len(
        re.findall(r"\b(?:ld|st)\.shared(?:\.[a-z0-9]+)?\b", text)
      )
    stages[stage] = observation

  observed_operations = {
    conversion["source_operation"]
    for conversion in conversions
    if conversion["source_operation"] is not None and conversion["layout_changed"]
  }
  expected_operations = set(expected_source_operations)
  missing = sorted(expected_operations - observed_operations)
  if missing:
    raise ValueError(
      "expected layout conversion did not change layout or survive compilation: "
      + ", ".join(missing)
    )
  if exact_source_operations:
    unexpected = sorted(observed_operations - expected_operations)
    unattributed = [
      conversion
      for conversion in conversions
      if conversion["layout_changed"] and conversion["source_operation"] is None
    ]
    if unexpected or unattributed:
      names = unexpected + ["unattributed effective conversion"] * bool(unattributed)
      raise ValueError("unplanned effective layout conversion: " + ", ".join(names))

  loop_stages = [
    stage for stage, result in stages.items() if result["loop_construct_count"] > 0
  ]
  loop_status = (
    "not_applicable"
    if not expected_loop
    else "retained"
    if loop_stages
    else "unrolled_or_removed"
  )
  return {
    "stages": stages,
    "layout_conversions": conversions,
    "layout_conversion_operation_count": len(conversions),
    "layout_conversion_count": sum(
      conversion["layout_changed"] for conversion in conversions
    ),
    "effective_conversion_operations": sorted(observed_operations),
    "loop_status": loop_status,
    "loop_stages": loop_stages,
    "interpretation_limit": (
      "IR shape records compiler observations; it does not prove semantic correctness"
    ),
  }
