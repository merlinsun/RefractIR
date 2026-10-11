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
_PASS_DUMP_HEADER = re.compile(
  r"^// -----// IR Dump Before (.+?): (.+?) \('([^']+)' operation(?:: @([^)]+))?\) "
  r"//----- //\s*$",
  re.MULTILINE,
)
_MODULE_SCOPE = "builtin.module"
_PASS_DUMP_TERMINAL = "LLVMDIScope"
_ENCODING_DEFINITION = re.compile(r"^\s*(#[a-zA-Z0-9_.-]+)\s*=\s*(\S.*?)\s*$")
_ALIAS_REFERENCE = re.compile(r"^#[a-zA-Z0-9_.-]+$")
_ENCODING_REFERENCE = re.compile(r"#[a-zA-Z0-9_.-]+")
_LOCATION_ALIAS = re.compile(r"^#loc\d+$")
_OPCODE = re.compile(r"(?:tt|ttg|arith|math|scf|cf|memref)\.[a-zA-Z_][\w.]*")
_IMPURE_OPCODE = re.compile(
  r"\b(?:tt\.(?:load|store|atomic[\w.]*|print|assert|call)|"
  r"ttg\.(?:local_alloc|local_load|local_store|async_[\w.]*)|memref\.[\w.]+)\b"
)


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


def _resolve_encoding_text(text, raw, visited):
  """Expand nested encoding aliases so equal layouts compare equal."""

  def substitute(match):
    name = match.group(0)
    if name in visited or name not in raw:
      return name
    return _resolve_encoding_text(raw[name], raw, visited | {name})

  return re.sub(r"\s+", "", _ENCODING_REFERENCE.sub(substitute, text))


def _encoding_definitions(text):
  """Resolve every encoding alias to its definition text.

  Layout aliases are renamed freely across passes, so a comparison that reads
  names rather than definitions reports a change where none happened.
  """
  raw = {}
  for line in text.splitlines():
    match = _ENCODING_DEFINITION.match(line)
    if match and not _LOCATION_ALIAS.match(match[1]):
      raw[match[1]] = match[2]
  return {
    name: _resolve_encoding_text(definition, raw, {name})
    for name, definition in raw.items()
  }


def _encoding_identity(name, definitions):
  """Return the comparison key for an encoding reference."""
  if name is None:
    return None
  return definitions.get(name) or name


def _region_map(text):
  """Map each 1-based line to its innermost enclosing region operation."""
  regions = {}
  stack = []
  for line_number, line in enumerate(text.splitlines(), start=1):
    stripped = line.strip()
    while stripped.startswith("}"):
      if stack:
        stack.pop()
      stripped = stripped[1:].strip()
    regions[line_number] = stack[-1] if stack else "module"
    if stripped.endswith("{"):
      opener = re.match(r"(?:%[-a-zA-Z$._0-9]+\s*=\s*)?([a-z][\w.]*)", stripped)
      if opener:
        stack.append(opener[1])
  return regions


_PTX_LABEL = re.compile(r"^\s*(\$[\w$.]+):")
_LLVM_LABEL = re.compile(r"^\s*([A-Za-z_0-9][\w.$]*):")
_PTX_BRANCH = re.compile(r"^\s*(?:@[%!]?[\w$.]+\s+)?bra(?:\.uni)?\s+(\$[\w$.]+)")
_LLVM_BRANCH = re.compile(r"\blabel\s+%([\w.$]+)")
_MLIR_LABEL = re.compile(r"^\s*(\^[\w$.]+):")
_MLIR_BRANCH = re.compile(r"\b(?:cf\.)?br\s+(\^[\w$.]+)")


def _backedge_count(text):
  """Count branches whose target label is already defined earlier in the text.

  PTX and LLVM IR encode a loop as a backward branch, and a lowered MLIR stage
  encodes it as a branch to an earlier block. Counting only structured MLIR
  operations would report a device loop as removed the moment it is lowered out
  of `scf`, which is a lowering rather than an unrolling decision.
  """
  definitions = {}
  backedges = 0
  for index, line in enumerate(text.splitlines()):
    ptx_label = _PTX_LABEL.match(line)
    llvm_label = _LLVM_LABEL.match(line)
    mlir_label = _MLIR_LABEL.match(line)
    if ptx_label:
      definitions.setdefault(("ptx", ptx_label[1]), index)
    elif mlir_label:
      definitions.setdefault(("mlir", mlir_label[1]), index)
    elif llvm_label and not line.strip().startswith("//"):
      definitions.setdefault(("llvm", llvm_label[1]), index)
    ptx_branch = _PTX_BRANCH.match(line)
    if ptx_branch and definitions.get(("ptx", ptx_branch[1]), index) < index:
      backedges += 1
      continue
    for target in _MLIR_BRANCH.findall(line):
      if definitions.get(("mlir", target), index) < index:
        backedges += 1
        break
    else:
      for target in _LLVM_BRANCH.findall(line):
        if definitions.get(("llvm", target), index) < index:
          backedges += 1
  return backedges


def _stage_observation(text):
  lines = text.splitlines()
  locations = _location_definitions(text)
  layout_definitions = _encoding_definitions(text)
  regions = _region_map(text)
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
  consumers = collections.defaultdict(list)
  for value, rhs in instructions:
    for operand in _SSA.findall(rhs):
      uses[operand] += 1
      consumers[operand].append(value)
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
    result_value = result_match[1] if result_match else None
    source_layout = encodings[0] if encodings else None
    target_layout = encodings[-1] if encodings else None
    source_definition = _encoding_identity(source_layout, layout_definitions)
    target_definition = _encoding_identity(target_layout, layout_definitions)
    layout_changed = len(encodings) >= 2 and source_definition != target_definition
    conversions.append(
      {
        "result": result_value,
        "source_value": operands[0] if operands else None,
        "source_line": source_line,
        "ir_line": line_number,
        "source_operation": None,
        "source_layout": source_layout,
        "target_layout": target_layout,
        "source_layout_definition": source_definition,
        "target_layout_definition": target_definition,
        "consumers": sorted(set(consumers.get(result_value, []))),
        "enclosing_region": regions.get(line_number, "module"),
        "layout_changed": layout_changed,
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
    "backedge_count": _backedge_count(text),
    "layout_conversions": conversions,
  }


def observe_compiler_ir(
  artifacts,
  *,
  source_lines=None,
  expected_source_operations=(),
  exact_source_operations=False,
  expected_loop=False,
  loop_required_stages=None,
):
  """Measure conversions, value fanout, duplicate work, and loop form per IR stage.

  `source_lines` maps source line numbers from compiler locations to stable
  logical operation names. Duplicate expressions are evidence of repeated
  IR work, not proof that a particular source-level value was rematerialized.

  `loop_required_stages` names the stages that must still contain a loop. It
  defaults to every supplied stage; a loop that survives only in an earlier
  stage such as TTIR therefore does not satisfy a later-stage requirement.
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
  if loop_required_stages is not None and (
    not isinstance(loop_required_stages, (tuple, list))
    or any(not isinstance(stage, str) or not stage for stage in loop_required_stages)
  ):
    raise ValueError("loop_required_stages must be a sequence of stage names")

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
  required_stages = (
    list(artifacts) if loop_required_stages is None else list(loop_required_stages)
  )
  loop_evidence = {}
  loop_presence = {}
  for stage in dict.fromkeys(list(artifacts) + required_stages):
    if stage not in stages:
      # A stage that was never measured is unavailable, not loop-free.
      loop_evidence[stage] = None
      loop_presence[stage] = None
      continue
    if stages[stage]["loop_construct_count"] > 0:
      loop_evidence[stage] = "mlir-construct"
    elif stages[stage]["backedge_count"] > 0:
      loop_evidence[stage] = "backedge"
    else:
      loop_evidence[stage] = None
    loop_presence[stage] = loop_evidence[stage] is not None
  required_presence = [loop_presence[stage] for stage in required_stages]
  if not expected_loop:
    loop_status = "not_applicable"
  elif not required_stages:
    loop_status = "unavailable"
  elif any(present is None for present in required_presence):
    loop_status = "unavailable"
  elif all(required_presence):
    loop_status = "retained"
  else:
    loop_status = "unrolled_or_removed"
  return {
    "stages": stages,
    "layout_conversions": conversions,
    "layout_conversion_operation_count": len(conversions),
    "layout_conversion_count": sum(
      conversion["layout_changed"] for conversion in conversions
    ),
    "effective_conversion_operations": sorted(observed_operations),
    "loop_status": loop_status,
    "loop_presence": loop_presence,
    "loop_evidence": loop_evidence,
    "loop_required_stages": required_stages,
    "loop_stages": loop_stages,
    "interpretation_limit": (
      "IR shape records compiler observations; it does not prove semantic correctness"
    ),
  }


def _conversion_key(conversion):
  """Identity of a conversion fragment, independent of SSA and alias names."""
  return (
    conversion["source_line"],
    conversion["source_layout_definition"],
    conversion["target_layout_definition"],
  )


def match_conversion_fragments(before_text, after_text, *, source_lines=None):
  """Match layout conversions between two IR snapshots by fragment identity.

  Values are matched by source location, effective encodings, and enclosing
  region rather than by SSA name, so a renamed alias or renumbered value stays
  the same fragment. A preserved fragment that changed its enclosing region is
  reported as moved even when both snapshots hold equal conversion counts.
  """
  if source_lines is None:
    source_lines = {}
  observations = []
  for text in (before_text, after_text):
    if not isinstance(text, str) or not text.strip():
      raise ValueError("conversion matching needs two nonempty IR snapshots")
    conversions = _stage_observation(text)["layout_conversions"]
    for conversion in conversions:
      conversion["source_operation"] = source_lines.get(conversion["source_line"])
    observations.append(conversions)
  before, after = observations

  grouped = []
  for conversions in (before, after):
    counter = collections.Counter()
    entries = collections.defaultdict(list)
    for conversion in conversions:
      key = _conversion_key(conversion)
      counter[key] += 1
      entries[key].append(conversion)
    grouped.append((counter, entries))

  fragments = []
  counts = collections.Counter()
  for key in sorted(
    set(grouped[0][0]) | set(grouped[1][0]), key=lambda item: repr(item)
  ):
    before_entries = sorted(
      grouped[0][1].get(key, []), key=lambda entry: entry["enclosing_region"]
    )
    after_entries = sorted(
      grouped[1][1].get(key, []), key=lambda entry: entry["enclosing_region"]
    )
    paired = min(len(before_entries), len(after_entries))
    for index in range(paired):
      left, right = before_entries[index], after_entries[index]
      moved = left["enclosing_region"] != right["enclosing_region"]
      counts["moved" if moved else "preserved"] += 1
      fragments.append(
        {
          "status": "moved" if moved else "preserved",
          "source_line": key[0],
          "source_operation": right["source_operation"] or left["source_operation"],
          "source_layout": right["source_layout"],
          "target_layout": right["target_layout"],
          "enclosing_region_before": left["enclosing_region"],
          "enclosing_region_after": right["enclosing_region"],
          "value_before": left["result"],
          "value_after": right["result"],
        }
      )
    for entry in before_entries[paired:]:
      counts["removed"] += 1
      fragments.append(
        {
          "status": "removed",
          "source_line": entry["source_line"],
          "source_operation": entry["source_operation"],
          "source_layout": entry["source_layout"],
          "target_layout": entry["target_layout"],
          "enclosing_region_before": entry["enclosing_region"],
          "enclosing_region_after": None,
          "value_before": entry["result"],
          "value_after": None,
        }
      )
    for entry in after_entries[paired:]:
      counts["added"] += 1
      fragments.append(
        {
          "status": "added",
          "source_line": entry["source_line"],
          "source_operation": entry["source_operation"],
          "source_layout": entry["source_layout"],
          "target_layout": entry["target_layout"],
          "enclosing_region_before": None,
          "enclosing_region_after": entry["enclosing_region"],
          "value_before": None,
          "value_after": entry["result"],
        }
      )

  return {
    "matched": counts["preserved"] + counts["moved"],
    "preserved": counts["preserved"],
    "removed": counts["removed"],
    "added": counts["added"],
    "moved": counts["moved"],
    "fragments": fragments,
  }


def _definitions_and_consumers(text):
  locations = _location_definitions(text)
  definitions = {}
  lines = {}
  consumers = collections.defaultdict(list)
  for line in text.splitlines():
    match = re.match(r"^\s*(%[-a-zA-Z$._0-9]+)\s*=\s*(.*)$", line)
    if not match:
      continue
    value, rhs = match.groups()
    definitions[value] = _normalized_expression(rhs)
    lines[value] = _source_line(line, locations)
    for operand in _SSA.findall(rhs):
      consumers[operand].append(value)
  return definitions, consumers, lines


def _normalized_expression(rhs):
  stripped = re.sub(r"\bloc\([^)]*\)", "", rhs)
  stripped = re.sub(r"\s+//.*$", "", stripped)
  return re.sub(r"\s+", " ", stripped).strip()


def _canonical_expression(name, definitions, depth=4):
  """Inline pure definitions so renamed SSA values still compare equal."""
  if depth <= 0:
    return name
  expression = definitions.get(name)
  if expression is None or _IMPURE_OPCODE.search(expression):
    return name
  if not _OPCODE.search(expression):
    return name

  def substitute(match):
    operand = match.group(0)
    if operand == name:
      return operand
    return "(" + _canonical_expression(operand, definitions, depth - 1) + ")"

  return re.sub(r"\s+", " ", _SSA.sub(substitute, expression)).strip()


def observe_rematerialization(before_text, after_text, *, source_lines=None):
  """Match recomputed values whose original uses were redirected.

  A candidate needs all three of: an equivalent pure definition, at least one
  consumer that stopped using the original, and at least one consumer of the
  recomputed value that structurally matches a former consumer of the original.
  Duplicate expressions alone are reported as unattributed work.
  """
  if source_lines is None:
    source_lines = {}
  for text in (before_text, after_text):
    if not isinstance(text, str) or not text.strip():
      raise ValueError("rematerialization matching needs two nonempty IR snapshots")

  before_definitions, before_consumers, before_lines = _definitions_and_consumers(
    before_text
  )
  after_definitions, after_consumers, _after_lines = _definitions_and_consumers(
    after_text
  )
  before_canonical = {}
  after_canonical = {}
  for name in before_definitions:
    before_canonical[name] = _canonical_expression(name, before_definitions)
  for name in after_definitions:
    after_canonical[name] = _canonical_expression(name, after_definitions)

  candidates = collections.defaultdict(lambda: {"redirected": [], "consumers": []})
  for original, canonical in sorted(before_canonical.items()):
    if canonical == original:
      continue
    before_uses = sorted(set(before_consumers.get(original, [])))
    after_uses = sorted(set(after_consumers.get(original, [])))
    if len(after_uses) >= len(before_uses):
      continue
    former_consumers = {before_canonical.get(consumer) for consumer in before_uses}
    for recomputed, recomputed_canonical in sorted(after_canonical.items()):
      if recomputed == original or recomputed_canonical != canonical:
        continue
      redirected = [
        consumer
        for consumer in sorted(set(after_consumers.get(recomputed, [])))
        if consumer != original and after_canonical.get(consumer) in former_consumers
      ]
      if not redirected:
        continue
      entry = candidates[original]
      entry["recomputed"] = recomputed
      entry["redirected"] = sorted(set(entry["redirected"]) | set(redirected))
      entry["consumers"] = after_uses
      entry["expression"] = after_definitions[recomputed]

  confirmed = []
  claimed = set()
  for original in sorted(candidates):
    entry = candidates[original]
    if entry["recomputed"] in claimed:
      continue
    claimed.add(entry["recomputed"])
    confirmed.append(
      {
        "original_value": original,
        "recomputed_values": [entry["recomputed"]],
        "expression": entry["expression"],
        "redirected_consumers": entry["redirected"],
        "remaining_consumers": entry["consumers"],
        "source_line": before_lines.get(original),
      }
    )

  confirmed_expressions = {candidate["expression"] for candidate in confirmed}
  unattributed = []
  for duplicate in _stage_observation(after_text)["duplicate_expressions"]:
    if duplicate["expression"] in confirmed_expressions:
      continue
    unattributed.append(
      {
        "expression": duplicate["expression"],
        "values": duplicate["values"],
        "count": duplicate["count"],
      }
    )

  if confirmed and unattributed:
    status = "confirmed_and_unattributed"
  elif confirmed:
    status = "confirmed_only"
  elif unattributed:
    status = "unattributed_only"
  else:
    status = "none"
  return {
    "status": status,
    "confirmed": confirmed,
    "confirmed_count": len(confirmed),
    "unattributed": unattributed,
    "unattributed_count": len(unattributed),
    "interpretation_limit": (
      "matched recomputation with redirected uses; duplicate expressions alone "
      "are reported as unattributed work"
    ),
  }


def _pass_snapshot_observation(text, source_lines):
  observation = _stage_observation(text)
  conversions = observation["layout_conversions"]
  for conversion in conversions:
    conversion["source_operation"] = source_lines.get(conversion["source_line"])
  effective_by_operation = collections.Counter(
    conversion["source_operation"] or "unattributed"
    for conversion in conversions
    if conversion["layout_changed"]
  )
  duplicate_opcode_counts = collections.Counter()
  repeated_operation_count = 0
  for duplicate in observation["duplicate_expressions"]:
    repeated_operation_count += duplicate["count"] - 1
    opcode = re.match(
      r"((?:tt|ttg|arith|math|scf)\.[a-zA-Z_][\w.]*)", duplicate["expression"]
    )
    if opcode:
      duplicate_opcode_counts[opcode[1]] += duplicate["count"] - 1
  return {
    "layout_conversion_count": len(conversions),
    "effective_layout_conversion_count": sum(
      conversion["layout_changed"] for conversion in conversions
    ),
    "effective_conversion_operation_counts": dict(
      sorted(effective_by_operation.items())
    ),
    "duplicate_expression_group_count": len(observation["duplicate_expressions"]),
    "duplicate_repeated_operation_count": repeated_operation_count,
    "duplicate_operation_counts": dict(sorted(duplicate_opcode_counts.items())),
    "loop_construct_count": observation["loop_construct_count"],
  }


def _counter_delta(before, after):
  return {
    name: after.get(name, 0) - before.get(name, 0)
    for name in sorted(set(before) | set(after))
    if after.get(name, 0) != before.get(name, 0)
  }


def _target_pass_evidence(compilation, observed, target_passes):
  """Locate the transition that measures each named pass in one compilation.

  A pass name that is absent, duplicated, or last in the compilation has no
  measurable before/after pair, so its evidence stays unavailable instead of
  falling back to aggregate counts from other stages.
  """
  evidence = {}
  for target in target_passes:
    positions = [
      index
      for index, snapshot in enumerate(compilation)
      if snapshot["pass"] == target and index + 1 < len(compilation)
    ]
    occurrences = [
      {
        "transition_index": index,
        "loop_before": observed[index]["loop_construct_count"] > 0,
        "loop_after": observed[index + 1]["loop_construct_count"] > 0,
      }
      for index in positions
    ]
    evidence[target] = {
      "status": (
        "unavailable"
        if not occurrences
        else "observed"
        if len(occurrences) == 1
        else "repeated"
      ),
      "transition_index": positions[0] if len(positions) == 1 else None,
      "transition_indices": positions,
      "occurrences": occurrences,
    }
  return evidence


def observe_pass_dumps(
  dump_text, *, source_lines=None, target_passes=(), rematerialization=False
):
  """Measure adjacent MLIR pass snapshots, grouped by kernel compilation.

  A transition from ``Before A`` to ``Before B`` measures the observable IR
  change across pass A. Duplicate-expression counts describe repeated IR work;
  they do not establish that a source value was rematerialized.

  `target_passes` names the passes whose own transition must be reported;
  `rematerialization` additionally matches recomputed values with redirected
  uses across each transition.
  """
  if not isinstance(dump_text, str) or not dump_text.strip():
    raise ValueError("pass dump must contain MLIR snapshots")
  if source_lines is None:
    source_lines = {}
  if not isinstance(source_lines, dict) or any(
    type(line) is not int or not isinstance(name, str) or not name
    for line, name in source_lines.items()
  ):
    raise ValueError("source_lines must map integer lines to operation names")
  if not isinstance(target_passes, (tuple, list)) or any(
    not isinstance(name, str) or not name for name in target_passes
  ):
    raise ValueError("target_passes must be a sequence of pass names")
  if type(rematerialization) is not bool:
    raise ValueError("rematerialization must be boolean")

  headers = list(_PASS_DUMP_HEADER.finditer(dump_text))
  if not headers:
    raise ValueError("pass dump contains no recognized MLIR snapshot headers")
  if dump_text[: headers[0].start()].strip():
    raise ValueError("pass dump contains text before the first snapshot")

  # A nested pipeline prints its own snapshots while a module-level pass is
  # running. The module snapshot is complete before the first nested header, so
  # the trailing nested text is excluded and counted instead of being read as
  # part of the module IR.
  snapshots = []
  for index, header in enumerate(headers):
    if header[3] != _MODULE_SCOPE:
      continue
    end = headers[index + 1].start() if index + 1 < len(headers) else len(dump_text)
    nested = 0
    for following in headers[index + 1 :]:
      if following[3] == _MODULE_SCOPE:
        break
      nested += 1
    snapshots.append(
      {
        "pass": header[1],
        "pipeline": header[2],
        "ir": dump_text[header.end() : end],
        "nested_snapshot_count": nested,
      }
    )
  if not snapshots:
    raise ValueError("pass dump contains no module-scope snapshots")

  compilations = []
  current = []
  for snapshot in snapshots:
    current.append(snapshot)
    if snapshot["pass"] == _PASS_DUMP_TERMINAL:
      compilations.append(current)
      current = []
  if current:
    raise ValueError("pass dump compilation does not end at the final pass")
  if not compilations:
    raise ValueError("pass dump contains no complete compilations")

  observed_compilations = []
  for compilation in compilations:
    transitions = []
    observed = [
      _pass_snapshot_observation(snapshot["ir"], source_lines)
      for snapshot in compilation
    ]
    for index in range(len(compilation) - 1):
      before, after = observed[index], observed[index + 1]
      conversion_match = match_conversion_fragments(
        compilation[index]["ir"],
        compilation[index + 1]["ir"],
        source_lines=source_lines,
      )
      transition = {
        "pass": compilation[index]["pass"],
        "pipeline": compilation[index]["pipeline"],
        "next_snapshot_pass": compilation[index + 1]["pass"],
        "before": before,
        "after": after,
        "layout_conversion_delta": (
          after["layout_conversion_count"] - before["layout_conversion_count"]
        ),
        "effective_layout_conversion_delta": (
          after["effective_layout_conversion_count"]
          - before["effective_layout_conversion_count"]
        ),
        "effective_conversion_operation_deltas": _counter_delta(
          before["effective_conversion_operation_counts"],
          after["effective_conversion_operation_counts"],
        ),
        "conversion_match": {
          name: conversion_match[name]
          for name in ("matched", "preserved", "removed", "added", "moved")
        },
        "conversion_movement": [
          fragment
          for fragment in conversion_match["fragments"]
          if fragment["status"] == "moved"
        ],
        "duplicate_expression_group_delta": (
          after["duplicate_expression_group_count"]
          - before["duplicate_expression_group_count"]
        ),
        "duplicate_repeated_operation_delta": (
          after["duplicate_repeated_operation_count"]
          - before["duplicate_repeated_operation_count"]
        ),
        "duplicate_operation_deltas": _counter_delta(
          before["duplicate_operation_counts"],
          after["duplicate_operation_counts"],
        ),
        "loop_construct_delta": (
          after["loop_construct_count"] - before["loop_construct_count"]
        ),
      }
      if rematerialization:
        transition["rematerialization"] = observe_rematerialization(
          compilation[index]["ir"],
          compilation[index + 1]["ir"],
          source_lines=source_lines,
        )
      transitions.append(transition)
    observed_compilations.append(
      {
        "snapshot_count": len(compilation),
        "nested_snapshot_counts": [
          snapshot["nested_snapshot_count"] for snapshot in compilation
        ],
        "terminal_pass": compilation[-1]["pass"],
        "final_loop_construct_count": observed[-1]["loop_construct_count"],
        "target_pass_evidence": _target_pass_evidence(
          compilation, observed, target_passes
        ),
        "transitions": transitions,
      }
    )

  return {
    "compilation_count": len(observed_compilations),
    "compilations": observed_compilations,
    "interpretation_limit": (
      "pass snapshots measure IR shape changes; duplicate-expression changes are "
      "not a proof of rematerialization"
    ),
  }
