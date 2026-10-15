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
_TENSOR_OPEN = re.compile(r"\btensor<")
_ENCODING_TEXT = re.compile(r'(?:#[\w.-]+|"[\w.-]+")(?:<.*>)?$')
_QUOTED_ENCODING = re.compile(r'^\s*"([\w.]+)"')
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
_REGION_OPENER = re.compile(
  r'(?:%[-a-zA-Z$._0-9]+(?::\d+)?\s*=\s*)?(?:"([\w.]+)"|([a-z][\w.]*))'
)
_RESOLUTION_DEPTH = 4
_UNRESOLVED = object()


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


def _split_top_level(text):
  """Split a printed type body on the commas that sit outside angle brackets."""
  parts, depth, start = [], 0, 0
  for index, char in enumerate(text):
    if char == "<":
      depth += 1
    elif char == ">":
      depth -= 1
    elif char == "," and depth == 0:
      parts.append(text[start:index])
      start = index + 1
  parts.append(text[start:])
  return parts


def _tensor_encodings(line):
  """Return the trailing encoding of every tensor type printed on a line.

  MLIR prints an encoding as a bare alias (`#blocked2`) or inline in
  parameterized form (`#ttg.slice<{dim = 1, parent = #blocked}>`,
  `#ttg.blocked<{...}>`), and prints a dialect name in generic quoted form
  (`"ttg.slice"<{...}>`) when the attribute has no custom spelling. The inline
  forms hold `>` characters that no single regular expression can pair with the
  `>` that closes the tensor type, so the body is scanned with a bracket counter
  and split on the commas outside every bracket, and the trailing component is
  read as the encoding.

  A type the line does not close, or one printed without a trailing encoding,
  contributes None rather than a truncated reading. The encoding is taken as
  printed: its name is not validated against a dialect.
  """
  encodings = []
  for match in _TENSOR_OPEN.finditer(line):
    depth, index = 1, match.end()
    while index < len(line) and depth:
      if line[index] == "<":
        depth += 1
      elif line[index] == ">":
        depth -= 1
      index += 1
    body = line[match.end() : index - 1] if not depth else ""
    components = _split_top_level(body)
    candidate = components[-1].strip() if len(components) > 1 else ""
    encodings.append(candidate if _ENCODING_TEXT.match(candidate) else None)
  return encodings


def _resolve_encoding_text(text, raw, visited):
  """Expand nested encoding aliases so equal layouts compare equal.

  A dialect name spelled generically (`"ttg.slice"<{...}>`) denotes the attribute
  its custom form spells (`#ttg.slice<{...}>`), so the quoted spelling is
  rewritten to that form before the nested references are expanded. Only the
  root of the encoding is rewritten: a quoted name nested inside a parameter
  keeps its printed text.
  """

  def substitute(match):
    name = match.group(0)
    if name in visited or name not in raw:
      return name
    return _resolve_encoding_text(raw[name], raw, visited | {name})

  text = _QUOTED_ENCODING.sub(r"#\1", text)
  return re.sub(r"\s+", "", _ENCODING_REFERENCE.sub(substitute, text))


def _encoding_definitions(text):
  """Resolve every encoding alias to its definition text.

  Layout aliases are renamed freely across passes, so a comparison that reads
  names rather than definitions reports a change where none happened. The
  resolved text is the comparison key for every spelling of that layout, whether
  a snapshot printed the alias or the parameters inline (`_encoding_identity`).
  A name the snapshot repeats keeps its last definition.
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
  """Return the comparison key for one printed encoding, None when none is.

  The key is the encoding's parameters with every nested alias expanded and all
  whitespace removed, so a bare alias, an inline parameter list, and the generic
  quoted spelling of one layout all compare equal, while two parameter sets that
  differ anywhere stay unequal. A reference the snapshot defines nowhere keeps
  its printed text, which no resolved definition equals.
  """
  if name is None:
    return None
  return _resolve_encoding_text(name, definitions, set()) or name


def _region_opener(line):
  """Name the region operation a line opens, or None when it opens no region.

  MLIR prints a repeated result as a counted list (`%name:2 = scf.for ...`)
  rather than by repeating the name, and prints an operation in generic form
  with a quoted name (`"tt.reduce"(...) ({`). Both open a region, so both must
  be pushed; otherwise every region opened that way is missing from the stack
  and the `})` that closes it pops the enclosing loop or function instead.
  """
  opener = _REGION_OPENER.match(line)
  if not opener:
    return None
  return opener[1] or opener[2]


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
      opener = _region_opener(stripped)
      if opener:
        stack.append(opener)
  return regions


_ENTRY_BLOCK = "<entry>"
_LLVM_MARKER = re.compile(r"^\s*define\b", re.MULTILINE)
_PTX_MARKER = re.compile(
  r"^\s*(?:\.(?:version|target|address_size|visible|extern|weak)\b|\$[\w$.]+:)",
  re.MULTILINE,
)
_PTX_SCOPE = re.compile(r"^\s*(?:\.(?:visible|extern|weak)\s+)*\.(?:entry|func)\b")
_PTX_LABEL = re.compile(r"^\s*(\$?[\w$.]+):(?:\s|$)")
_PTX_BRANCH = re.compile(r"^\s*(?:@[!\w$.%]+\s+)?bra(?:\.uni)?\s+([^\s,;]+)")
_PTX_BRANCH_PREFIX = re.compile(r"^\s*(?:@[!\w$.%]+\s+)?br[ax]\b")
_PTX_PREDICATED = re.compile(r"^\s*@")
_PTX_TERMINATOR = re.compile(r"^\s*(?:@[!\w$.%]+\s+)?(?:ret|exit|trap)\b")
_PTX_TARGET = re.compile(r"^\$?[\w$.]+$")
_LLVM_SCOPE = re.compile(r"^\s*define\b")
_LLVM_SCOPE_END = re.compile(r"^\}")
_LLVM_LABEL = re.compile(r"^\s*([A-Za-z_0-9][\w.$]*):")
_LLVM_TERMINATOR = re.compile(r"^\s*(br|switch|indirectbr|callbr|invoke)\b")
_LLVM_TARGET = re.compile(r"\blabel\s+%([\w.$]+)")
_MLIR_LABEL = re.compile(r"^\s*(\^[\w$.]+)(?:\([^)]*\))?:")
_MLIR_TERMINATOR = re.compile(r"^\s*(?:(?:cf|llvm)\.)?(br|cond_br|switch)\b")
_MLIR_GENERIC_TERMINATOR = re.compile(r'^\s*"(?:(?:cf|llvm)\.)?(br|cond_br|switch)"')
_MLIR_TARGET = re.compile(r"\^[\w$.]+")


def _ptx_branch_graph(lines):
  """Collect PTX branch edges, one scope per `.entry`/`.func` body.

  PTX blocks need no terminator, so a block that does not end in an
  unconditional transfer also reaches the next block in program order. Those
  implicit edges are returned separately: they close cycles, but they are not
  branches and must not be counted as such.
  """
  edges, fallthrough, unparsed = [], [], 0
  defined = collections.defaultdict(set)
  blocks = collections.defaultdict(lambda: _ENTRY_BLOCK)
  closed = {}
  scope = "ptx:0"
  count = 0
  for line in lines:
    if _PTX_SCOPE.match(line):
      count += 1
      scope = f"ptx:{count}"
      closed.pop(scope, None)
      continue
    label = _PTX_LABEL.match(line)
    if label:
      previous = closed.get(scope)
      if previous is not None and previous[1]:
        fallthrough.append((scope, previous[0], label[1]))
      blocks[scope] = label[1]
      closed[scope] = (label[1], True)
      defined[scope].add(label[1])
      continue
    branch = _PTX_BRANCH.match(line)
    if branch:
      target = branch[1]
      if _PTX_TARGET.match(target):
        edges.append((scope, blocks[scope], target))
        if not _PTX_PREDICATED.match(line):
          closed[scope] = (blocks[scope], False)
      else:
        # A register or table operand names no block in this text.
        unparsed += 1
      continue
    if _PTX_BRANCH_PREFIX.match(line):
      # `brx.idx` and truncated `bra` lines reach a destination we cannot read.
      unparsed += 1
      continue
    if _PTX_TERMINATOR.match(line) and not _PTX_PREDICATED.match(line):
      closed[scope] = (blocks[scope], False)
  return edges, fallthrough, defined, unparsed


def _llvm_scopes(lines):
  """Assign each line to the innermost enclosing `define` body."""
  scopes = []
  scope = "llvm:0"
  count = 0
  for line in lines:
    if _LLVM_SCOPE.match(line):
      count += 1
      scope = f"llvm:{count}"
    scopes.append(scope)
    if _LLVM_SCOPE_END.match(line):
      scope = "llvm:0"
  return scopes


def _llvm_branch_graph(lines):
  """Collect LLVM IR branch edges, one scope per `define` body."""
  edges, unparsed = [], 0
  defined = collections.defaultdict(set)
  blocks = collections.defaultdict(lambda: _ENTRY_BLOCK)
  scopes = _llvm_scopes(lines)
  in_switch = False
  for index, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith(";") or stripped.startswith("//"):
      continue
    scope = scopes[index]
    if in_switch:
      for target in _LLVM_TARGET.findall(line):
        edges.append((scope, blocks[scope], target))
      if "]" in line:
        in_switch = False
      continue
    label = _LLVM_LABEL.match(line)
    if label:
      blocks[scope] = label[1]
      defined[scope].add(label[1])
      continue
    terminator = _LLVM_TERMINATOR.match(line)
    if terminator is None:
      continue
    operation = terminator[1]
    targets = _LLVM_TARGET.findall(line)
    if operation in ("indirectbr", "callbr") or not targets:
      # An indirect destination, a computed transfer, or a truncated line.
      unparsed += 1
      continue
    for target in targets:
      edges.append((scope, blocks[scope], target))
    if operation == "switch" and "]" not in line:
      in_switch = True
  return edges, [], defined, unparsed


def _mlir_scopes(lines):
  """Assign each line to the innermost enclosing MLIR region."""
  scopes = []
  stack = ["mlir:0"]
  count = 0
  for line in lines:
    stripped = line.strip()
    while stripped.startswith("}"):
      if len(stack) > 1:
        stack.pop()
      stripped = stripped[1:].strip()
    scopes.append(stack[-1])
    if stripped.endswith("{"):
      count += 1
      stack.append(f"mlir:{count}")
  return scopes


def _mlir_branch_graph(lines):
  """Collect lowered MLIR branch edges, one scope per region."""
  edges, unparsed = [], 0
  defined = collections.defaultdict(set)
  blocks = collections.defaultdict(lambda: _ENTRY_BLOCK)
  scopes = _mlir_scopes(lines)
  in_switch = False
  for index, line in enumerate(lines):
    if line.strip().startswith("//"):
      continue
    scope = scopes[index]
    if in_switch:
      for target in _MLIR_TARGET.findall(line):
        edges.append((scope, blocks[scope], target))
      if "]" in line:
        in_switch = False
      continue
    label = _MLIR_LABEL.match(line)
    if label:
      blocks[scope] = label[1]
      defined[scope].add(label[1])
      continue
    terminator = _MLIR_TERMINATOR.match(line) or _MLIR_GENERIC_TERMINATOR.match(line)
    if terminator is None:
      continue
    targets = _MLIR_TARGET.findall(line)
    if not targets:
      unparsed += 1
      continue
    for target in targets:
      edges.append((scope, blocks[scope], target))
    if terminator[1] == "switch" and "]" not in line:
      in_switch = True
  return edges, [], defined, unparsed


def _strongly_connected_components(nodes, adjacency):
  """Tarjan's algorithm, iteratively, so a deep CFG cannot exhaust the stack."""
  visited = {}
  low = {}
  on_stack = set()
  stack = []
  components = {}
  counter = 0
  for root in nodes:
    if root in visited:
      continue
    visited[root] = low[root] = counter
    counter += 1
    stack.append(root)
    on_stack.add(root)
    work = [(root, iter(adjacency.get(root, ())))]
    while work:
      node, neighbours = work[-1]
      descended = False
      for neighbour in neighbours:
        if neighbour not in visited:
          visited[neighbour] = low[neighbour] = counter
          counter += 1
          stack.append(neighbour)
          on_stack.add(neighbour)
          work.append((neighbour, iter(adjacency.get(neighbour, ()))))
          descended = True
          break
        if neighbour in on_stack:
          low[node] = min(low[node], visited[neighbour])
      if descended:
        continue
      work.pop()
      if low[node] == visited[node]:
        component = set()
        while True:
          member = stack.pop()
          on_stack.discard(member)
          component.add(member)
          if member == node:
            break
        for member in component:
          components[member] = component
      if work:
        parent = work[-1][0]
        low[parent] = min(low[parent], low[node])
  return components


def _branch_cycle_evidence(text):
  """Count branches on a CFG cycle and branches whose target is unreadable.

  A branch participates in a cycle when its source and target blocks belong to
  the same strongly connected component of the control-flow graph of their
  function. Text order says nothing about dominance, so a block printed before
  its predecessor is not a loop. Edges and block names are scoped per function,
  so the same label text in two functions never merges. Implicit fall-through
  edges complete the graph but are never counted: only branch instructions are.
  A branch whose target is not a block of its own function is unreadable
  evidence: it is counted, and the caller must treat the stage as unavailable
  rather than loop-free.
  """
  lines = text.splitlines()
  if _LLVM_MARKER.search(text):
    graphs = (_llvm_branch_graph(lines),)
  elif _PTX_MARKER.search(text):
    graphs = (_ptx_branch_graph(lines),)
  else:
    graphs = (_mlir_branch_graph(lines),)
  edges, fallthrough, defined, unparsed = [], [], {}, 0
  for found, implicit, blocks, unknown in graphs:
    edges.extend(found)
    fallthrough.extend(implicit)
    for scope, labels in blocks.items():
      defined.setdefault(scope, set()).update(labels)
    unparsed += unknown
  readable = []
  for scope, source, target in edges:
    if target not in defined.get(scope, ()):
      unparsed += 1
      continue
    readable.append((scope, source, target))
  adjacency = collections.defaultdict(set)
  nodes = {}
  for scope, source, target in readable + [
    edge for edge in fallthrough if edge[2] in defined.get(edge[0], ())
  ]:
    nodes.setdefault((scope, source))
    nodes.setdefault((scope, target))
    adjacency[(scope, source)].add((scope, target))
  components = _strongly_connected_components(nodes, adjacency)
  cycles = 0
  for scope, source, target in readable:
    component = components.get((scope, source))
    if component is not None and component is components.get((scope, target)):
      cycles += 1
  return cycles, unparsed


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
    encodings = _tensor_encodings(line)
    source_line = _source_line(line, locations)
    result_value = result_match[1] if result_match else None
    source_layout = encodings[0] if encodings else None
    target_layout = encodings[-1] if encodings else None
    source_definition = _encoding_identity(source_layout, layout_definitions)
    target_definition = _encoding_identity(target_layout, layout_definitions)
    # A conversion whose operand types print no encoding, or print an unclosed
    # one, has no readable pair: the effect is unproven rather than absent.
    readable = len(encodings) >= 2 and all(
      encoding is not None for encoding in encodings[:2]
    )
    layout_changed = readable and source_definition != target_definition
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
  branch_cycles, unparsed_branches = _branch_cycle_evidence(text)
  return {
    "defined_value_count": len(definitions),
    "operation_counts": dict(sorted(operation_counts.items())),
    "reused_values": reused,
    "duplicate_expressions": duplicates,
    "loop_construct_count": len(loops),
    "loop_constructs": loops,
    "backedge_count": branch_cycles,
    "unparsed_branch_count": unparsed_branches,
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
  A stage holds a loop when it keeps a structured loop construct (`mlir-construct`)
  or when its branch graph has a cycle (`branch-cycle`); a stage with a branch
  whose target cannot be read stays unavailable rather than loop-free.
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
      loop_evidence[stage] = "branch-cycle"
    else:
      loop_evidence[stage] = None
    if loop_evidence[stage] is not None:
      loop_presence[stage] = True
    elif stages[stage]["unparsed_branch_count"] > 0:
      # Part of the CFG could not be read, so "no cycle found" is not
      # evidence that the stage is loop-free.
      loop_presence[stage] = None
    else:
      loop_presence[stage] = False
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


def _is_pure_definition(expression):
  """Report whether a definition denotes a computation this module can compare."""
  return bool(_OPCODE.search(expression)) and not _IMPURE_OPCODE.search(expression)


def _canonical_expression(name, definitions, depth=_RESOLUTION_DEPTH):
  """Inline pure definitions so renamed SSA values still compare equal.

  Only the root keeps this contract: an impure or unparsed definition denotes no
  structure of its own, so it canonicalizes to its printed name. Operands are
  substituted with the identity their own definition supplies instead, so that
  leaves compare by what they compute rather than by how they are printed. An
  operand whose identity the budget cannot establish leaves the whole expression
  without one, so a truncated expression is never compared as if it were whole.
  """
  if depth <= 0:
    # The budget was spent before this definition could be inlined. A value the
    # snapshot defines has no identity in its printed name.
    return name if name not in definitions else _UNRESOLVED
  expression = definitions.get(name)
  if expression is None or not _is_pure_definition(expression):
    return name
  unresolved = False

  def substitute(match):
    nonlocal unresolved
    operand = match.group(0)
    if operand == name:
      return operand
    identity = _operand_identity(operand, definitions, depth - 1)
    if identity is _UNRESOLVED:
      unresolved = True
      return operand
    if identity is None:
      # The snapshot defines nothing here, so the printed name is all there is.
      return operand
    return "(" + identity + ")"

  text = re.sub(r"\s+", " ", _SSA.sub(substitute, expression)).strip()
  return _UNRESOLVED if unresolved else text


def _operand_identity(operand, definitions, depth):
  """Return what one operand denotes, or None when only its name denotes it.

  A definition in the same snapshot is what the operand computes: its text when
  the definition is impure or unparsed, and its recursively canonicalized
  expression when it is pure. The operands printed inside a text definition are
  resolved the same way, so a pointer whose own computation changed is not the
  same pointer, and one that was merely renamed still is. An operand with no
  definition in the snapshot (an argument, or a value defined outside it) has no
  identity beyond its printed name. Callers wrap a returned identity in
  parentheses, so an operand that a definition identifies never compares equal
  to one that only a name identifies: a leaf that cannot be resolved the same
  way on both sides leaves the correspondence unproven. That includes a leaf the
  snapshot does define but the inline budget cannot reach, for which _UNRESOLVED
  is returned instead: its printed name is not its identity, so neither the leaf
  nor any expression containing it may be compared by text.
  """
  expression = definitions.get(operand)
  if expression is None:
    return None
  if not _is_pure_definition(expression) or depth <= 0:
    # Impure, unparsed, or past the inline budget: the definition text says what
    # the operand computes, while its printed name would not.
    return _resolved_definition_text(expression, operand, definitions, depth)
  return _canonical_expression(operand, definitions, depth)


def _resolved_definition_text(expression, name, definitions, depth):
  """Substitute the operands a definition text prints with their identities.

  The text is an identity only while every operand it prints is either resolved
  or left as the printed name of a value the snapshot defines nowhere. Once the
  budget is spent, an operand the snapshot does define is a correspondence the
  text cannot show: the snapshot may have changed what that name computes, so
  the whole text counts as unresolved rather than as the name it prints.
  """
  unresolved = False

  def substitute(match):
    nonlocal unresolved
    inner = match.group(0)
    if inner == name:
      return inner
    if depth <= 0:
      if inner in definitions:
        unresolved = True
      return inner
    identity = _operand_identity(inner, definitions, depth - 1)
    if identity is _UNRESOLVED:
      unresolved = True
      return inner
    return inner if identity is None else "(" + identity + ")"

  text = re.sub(r"\s+", " ", _SSA.sub(substitute, expression)).strip()
  return _UNRESOLVED if unresolved else text


def _comparable_expression(name, definitions):
  """Return the expression a name denotes, or None when it has no identity.

  A definition whose resolution ran out at a leaf the snapshot defines carries
  no identity to compare: the pass may have changed what that leaf computes, so
  the containing expression must not be matched by the text that remains.
  """
  expression = _canonical_expression(name, definitions)
  return None if expression is _UNRESOLVED else expression


def _denoted_expression(name, canonical):
  """Return the computation a name denotes, or None when it is only a name.

  An impure or unparsed definition canonicalizes to its own name, and an
  unresolved one is None, so neither carries structure to compare, and snapshot
  names are rebound freely. Such a definition cannot establish that two
  snapshots hold the same computation.
  """
  expression = canonical.get(name)
  if expression is None or expression == name:
    return None
  return expression


def _pure_expression_counts(canonical):
  """Count the pure definitions that share each canonical expression.

  An impure or unparsed definition canonicalizes to its own name and an
  unresolved one to None, so counting by canonical expression only ever groups
  equal resolved pure computations.
  """
  counts = collections.Counter()
  for name, expression in canonical.items():
    if expression is not None and expression != name:
      counts[expression] += 1
  return counts


def observe_rematerialization(before_text, after_text, *, source_lines=None):
  """Match recomputed values whose original uses were redirected.

  A candidate needs all of: an equivalent pure definition (a), strictly more
  pure definitions with that canonical expression after the pass than before it
  (b), at least one consumer the original lost (c), at least one consumer of the
  recomputed value that structurally matches a former consumer of the original
  (d), and the original producer still present after the pass under the same
  name with the same canonical expression (e).

  Correspondence is decided by canonical expressions in which every operand is
  replaced by the identity its own definition supplies: the definition text when
  that definition is impure or unparsed, the recursively canonicalized
  expression when it is pure, and the printed name when the snapshot defines the
  operand nowhere. Two defined leaves are therefore never equated by their
  printed names, and a leaf that resolves on one side but not the other leaves
  the correspondence unproven. A definition whose resolution budget runs out at
  a leaf the snapshot does define has no identity at all: the text that remains
  prints a name whose computation the pass may have changed, so no candidate
  that would compare it is confirmed. That refusal does not depend on how large
  the budget is, only on whether the resolution completed. Requirement (b) is
  what separates a recomputation from a rename, and (e) is what ties a
  recomputed value to a producer that survived the pass: renumbering alone, an
  unchanged graph, a name rebound to a load, a call, or any different
  definition, and ambiguous correspondences stay unconfirmed. Duplicate
  expressions alone are reported as unattributed work.
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
    before_canonical[name] = _comparable_expression(name, before_definitions)
  for name in after_definitions:
    after_canonical[name] = _comparable_expression(name, after_definitions)
  before_expressions = _pure_expression_counts(before_canonical)
  after_expressions = _pure_expression_counts(after_canonical)

  candidates = collections.defaultdict(lambda: {"recomputed": [], "redirected": []})
  for original, canonical in sorted(before_canonical.items()):
    if canonical is None or canonical == original:
      # An unresolved original has no identity to compare, and an impure or
      # unparsed one canonicalizes to its own name.
      continue
    if _denoted_expression(original, after_canonical) != canonical:
      # The producer survived only if the name still denotes the same pure
      # computation: a name rebound to a load, a call, or any different
      # definition is not the original.
      continue
    if after_expressions[canonical] <= before_expressions[canonical]:
      # No additional computation exists, so this is a rename, not a copy.
      continue
    before_uses = sorted(set(before_consumers.get(original, [])))
    after_uses = sorted(set(after_consumers.get(original, [])))
    if len(after_uses) >= len(before_uses):
      continue
    former_consumers = set()
    for consumer in before_uses:
      expression = _denoted_expression(consumer, before_canonical)
      if expression is not None:
        former_consumers.add(expression)
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
      entry["recomputed"].append(recomputed)
      entry["redirected"] = sorted(set(entry["redirected"]) | set(redirected))
      entry["consumers"] = after_uses
      entry["expression"] = after_definitions[recomputed]

  confirmed = []
  claims = collections.Counter(
    entry["recomputed"][0]
    for entry in candidates.values()
    if len(entry["recomputed"]) == 1
  )
  for original in sorted(candidates):
    entry = candidates[original]
    if len(entry["recomputed"]) != 1:
      # Several equivalent copies took over the original's consumers, so no
      # single copy can be called its recomputation.
      continue
    recomputed = entry["recomputed"][0]
    if claims[recomputed] > 1:
      # Several producers lost consumers to this one copy: which of them it
      # recomputes is ambiguous, so none of them is confirmed.
      continue
    confirmed.append(
      {
        "original_value": original,
        "recomputed_values": [recomputed],
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
      "a confirmed recomputation needs an equivalent surviving producer, an "
      "additional pure definition, and a redirected former consumer; duplicate "
      "expressions alone are reported as unattributed work"
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
