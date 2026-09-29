"""Deterministic dependency audit for the compositional tensor graph builder.

The audit measures the dependency structures the builder reaches rather than
their spelling. Each generated case has three separate identities: its source
bytes, its ordered dependency topology, and its operator-labelled graph.
Literal operand values, variable and parameter names, declared shapes, and axis
attributes enter none of them; operator names enter only the labelled graph.
Dimensions, constants, and axes are recorded beside those identities so they
cannot inflate the structural claim.

Every generation failure stays in the denominator. A case is accepted only
after `symiri --check` accepts the emitted source for the selected profile, the
source parses as one admitted straight-line `^entry` block whose every
computation lies in the returned value's backward slice, and an independent
reference run reproduces the case's expected bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from test.lib.tensor_case import CaseValidationError, TensorCase, load_case
from test.unit.tensor_support import check_source, run_typed

ROOT = Path(__file__).resolve().parents[2]
RYSMITH = Path(os.environ.get("REFRACTIR_RYSMITH", ROOT / "rysmith")).resolve()

PROFILES = ("v2", "v3")
SEED_COUNT = 256

# The structural features the audit requires a witness for. The names describe
# the live dependency structure; the last is the scalar reduction edge case.
FEATURES = (
  "chain",
  "shared-intermediate-diamond",
  "independent-branch-join",
  "dual-live-reductions",
  "transpose-or-broadcast-upstream-of-reduction",
  "scalar-reduction-then-expand",
)

REDUCTIONS = frozenset({"sum", "rmin", "rmax"})
SHAPE_OPERATORS = frozenset({"transpose", "broadcast"})
BINARY_OPERATORS = frozenset(
  {"add", "sub", "mul", "div", "rem", "min", "max", "and", "or", "xor"}
)

_FUNCTION = re.compile(r"fun (@[A-Za-z_][\w.$-]*)\((.*)\)\s*:\s*(.*?)\s*\{$")
_LET = re.compile(r"let mut (%[A-Za-z_][\w.$-]*)\s*:\s*(.*?)\s*=\s*(.*?);$")
_BLOCK = re.compile(r"\^[A-Za-z_][\w.$-]*:$")
_RETURN = re.compile(r"ret (%[A-Za-z_][\w.$-]*);$")
_ASSIGN = re.compile(r"(%[A-Za-z_][\w.$-]*)\s*=\s*(.*?);$")
_LOCAL = re.compile(r"%[A-Za-z_][\w.$-]*")
_LITERAL = re.compile(r"-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")
_PARAMETER = re.compile(r"(%[A-Za-z_][\w.$-]*)\s*:\s*(.*?)$")
_TENSOR_TYPE = re.compile(r"tensor \[([0-9, ]*)\]\s*(?:i32|f32)$")
_SCALAR_TYPE = re.compile(r"(?:i32|f32)$")
_TENSOR_HEAD = re.compile(r"([A-Za-z_]\w*)\s+(.*)$")


class UnsupportedSource(ValueError):
  """The source is outside the admitted straight-line compositional subset."""


@dataclass(frozen=True)
class Literal:
  """A literal operand. Its value stays out of every structural identity."""

  text: str


@dataclass(frozen=True)
class Value:
  """One admitted value: an entry parameter, a copy, or a tensor operator."""

  name: str
  operator: str
  operands: tuple[str | Literal, ...]
  shape: tuple[int, ...]
  axis: int | None = None
  keep_dims: bool | None = None
  attribute: tuple[int, ...] = ()

  @property
  def is_input(self):
    return self.operator == "input"


@dataclass(frozen=True)
class Graph:
  """The straight-line slice an admitted source declares, in source order."""

  parameters: tuple[str, ...]
  values: tuple[Value, ...]
  result: str

  @property
  def computations(self):
    return tuple(value for value in self.values if not value.is_input)


def _split_top_level(text):
  """Split on commas that are not inside brackets."""
  parts = []
  current = []
  depth = 0
  for character in text:
    if character in "[(":
      depth += 1
    elif character in "])":
      depth -= 1
    if character == "," and depth == 0:
      parts.append("".join(current).strip())
      current = []
    else:
      current.append(character)
  tail = "".join(current).strip()
  if tail:
    parts.append(tail)
  return parts


def _parse_type(text):
  """Return the declared shape; a scalar and a rank-zero tensor both look empty."""
  tensor = _TENSOR_TYPE.match(text)
  if tensor is not None:
    inner = tensor.group(1).strip()
    return tuple(int(value) for value in inner.split(",")) if inner else ()
  if _SCALAR_TYPE.match(text):
    return ()
  raise UnsupportedSource(f"unsupported type {text!r}")


def _parse_int(text, what):
  try:
    return int(text)
  except ValueError as error:
    raise UnsupportedSource(f"{what}: expected an integer, got {text!r}") from error


def _parse_list(text, what):
  if not (text.startswith("[") and text.endswith("]")):
    raise UnsupportedSource(f"{what}: expected a bracketed list, got {text!r}")
  inner = text[1:-1].strip()
  if not inner:
    return ()
  return tuple(_parse_int(value.strip(), what) for value in inner.split(","))


def _parse_operand(text, defined, what):
  if _LOCAL.fullmatch(text):
    if text not in defined:
      raise UnsupportedSource(f"{what} reads {text} before it is assigned")
    return text
  if _LITERAL.fullmatch(text):
    return Literal(text=text)
  raise UnsupportedSource(f"{what}: unsupported operand {text!r}")


def _parse_rhs(name, rhs, shapes, defined):
  """Build one admitted value from an assignment right-hand side."""
  if _LOCAL.fullmatch(rhs):
    if rhs not in defined:
      raise UnsupportedSource(f"{name} reads {rhs} before it is assigned")
    return Value(name=name, operator="copy", operands=(rhs,), shape=shapes[name])
  if not rhs.startswith("tensor "):
    raise UnsupportedSource(f"unsupported right-hand side {rhs!r}")
  head = _TENSOR_HEAD.match(rhs[len("tensor ") :])
  if head is None:
    raise UnsupportedSource(f"unsupported tensor expression {rhs!r}")
  operator = head.group(1)
  parts = _split_top_level(head.group(2))
  what = f"{name} = tensor {operator}"

  def operand(position):
    return _parse_operand(parts[position], defined, what)

  if operator in BINARY_OPERATORS:
    if len(parts) != 2:
      raise UnsupportedSource(f"{what}: a binary operator takes two operands")
    return Value(
      name=name,
      operator=operator,
      operands=(operand(0), operand(1)),
      shape=shapes[name],
    )
  if operator == "exp":
    if len(parts) != 1:
      raise UnsupportedSource(f"{what}: exp takes one operand")
    return Value(
      name=name, operator=operator, operands=(operand(0),), shape=shapes[name]
    )
  if operator in ("transpose", "broadcast"):
    if len(parts) != 2:
      raise UnsupportedSource(f"{what}: expected an operand and an attribute list")
    return Value(
      name=name,
      operator=operator,
      operands=(operand(0),),
      shape=shapes[name],
      attribute=_parse_list(parts[1], what),
    )
  if operator in ("expand",):
    if len(parts) != 2:
      raise UnsupportedSource(f"{what}: expected an operand and an axis")
    return Value(
      name=name,
      operator=operator,
      operands=(operand(0),),
      shape=shapes[name],
      axis=_parse_int(parts[1], what),
    )
  if operator in REDUCTIONS:
    if len(parts) != 3 or parts[2] not in ("keep", "drop"):
      raise UnsupportedSource(f"{what}: expected an operand, an axis, and keep or drop")
    return Value(
      name=name,
      operator=operator,
      operands=(operand(0),),
      shape=shapes[name],
      axis=_parse_int(parts[1], what),
      keep_dims=parts[2] == "keep",
    )
  raise UnsupportedSource(f"unsupported tensor operator {operator!r}")


def parse_graph(source):
  """Parse one admitted straight-line compositional source.

  Anything outside that subset, including control flow, extra blocks, `sym`,
  `require`, `select`, and non-`undef` initializers, raises UnsupportedSource
  rather than yielding a partial graph.
  """
  if not isinstance(source, str):
    raise TypeError("source must be text")
  lines = [line.strip() for line in source.splitlines()]
  lines = [line for line in lines if line and not line.startswith("//")]
  if not lines:
    raise UnsupportedSource("empty source")
  header = _FUNCTION.match(lines[0])
  if header is None:
    raise UnsupportedSource(f"unsupported function header {lines[0]!r}")

  parameters = []
  shapes = {}
  for entry in _split_top_level(header.group(2)):
    match = _PARAMETER.match(entry)
    if match is None:
      raise UnsupportedSource(f"unsupported parameter {entry!r}")
    name = match.group(1)
    if name in shapes:
      raise UnsupportedSource(f"duplicate parameter {name}")
    shapes[name] = _parse_type(match.group(2))
    parameters.append(name)
  _parse_type(header.group(3))
  if not parameters:
    raise UnsupportedSource("the audit admits at least one runtime parameter")

  index = 1
  while index < len(lines):
    declaration = _LET.match(lines[index])
    if declaration is None:
      break
    name, type_text, initializer = declaration.groups()
    if initializer != "undef":
      raise UnsupportedSource(f"{name}: only undef initializers are admitted")
    if name in shapes:
      raise UnsupportedSource(f"duplicate declaration {name}")
    shapes[name] = _parse_type(type_text)
    index += 1

  if index >= len(lines) or lines[index] != "^entry:":
    raise UnsupportedSource("the admitted program is exactly one ^entry block")
  index += 1
  statements = []
  while index < len(lines) and lines[index] != "}":
    statements.append(lines[index])
    index += 1
  if index >= len(lines):
    raise UnsupportedSource("unterminated function body")
  if index != len(lines) - 1:
    raise UnsupportedSource("statements after the closing brace")

  if not statements:
    raise UnsupportedSource("the entry block has no statements")
  returned = _RETURN.match(statements[-1])
  if returned is None:
    raise UnsupportedSource("the entry block must end with a return")

  values = [
    Value(name=name, operator="input", operands=(), shape=shapes[name])
    for name in parameters
  ]
  defined = {name: shapes[name] for name in parameters}
  declared = set(shapes) - set(parameters)
  for statement in statements[:-1]:
    match = _ASSIGN.match(statement)
    if match is None:
      raise UnsupportedSource(f"unsupported entry statement {statement!r}")
    name, rhs = match.group(1), match.group(2)
    if name not in declared:
      raise UnsupportedSource(f"{name} is assigned without a mutable declaration")
    if name in defined:
      raise UnsupportedSource(f"{name} is assigned more than once")
    values.append(_parse_rhs(name, rhs, shapes, defined))
    defined[name] = shapes[name]
  missing = declared - set(defined)
  if missing:
    raise UnsupportedSource(
      f"declared but never assigned: {', '.join(sorted(missing))}"
    )
  if returned.group(1) not in defined:
    raise UnsupportedSource(f"the return value {returned.group(1)} is never assigned")

  return Graph(tuple(parameters), tuple(values), returned.group(1))


def values_by_name(graph):
  return {value.name: value for value in graph.values}


def return_slice(graph):
  """Return every parameter and computation the returned value is built from."""
  producers = values_by_name(graph)
  live = set()
  pending = [graph.result]
  while pending:
    name = pending.pop()
    if name in live:
      continue
    live.add(name)
    value = producers.get(name)
    if value is None:
      raise UnsupportedSource(f"the return slice reads undefined {name}")
    pending.extend(slot for slot in value.operands if isinstance(slot, str))
  return frozenset(live)


def input_dependencies(graph):
  """Map every admitted value to the entry parameters it is computed from."""
  dependencies = {}
  for value in graph.values:
    if value.is_input:
      dependencies[value.name] = frozenset({value.name})
      continue
    merged = set()
    for slot in value.operands:
      if isinstance(slot, str):
        if slot not in dependencies:
          raise UnsupportedSource(f"{value.name} reads {slot} before it is defined")
        merged |= dependencies[slot]
    dependencies[value.name] = frozenset(merged)
  return dependencies


def consumers(graph):
  """Map every value name to the indices of the values that read it, in order.

  A value read twice by the same statement is one consumer: reuse inside a
  node is not a second branch.
  """
  sites = {}
  for index, value in enumerate(graph.values):
    for slot in set(value.operands):
      if isinstance(slot, str):
        sites.setdefault(slot, []).append(index)
  for index in sites.values():
    index.sort()
  return sites


def _digest(records):
  body = json.dumps(records, separators=(",", ":"), ensure_ascii=True)
  return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _canonical(graph, *, labelled):
  """Assign canonical node identifiers from the returned value's operands.

  Identifiers are handed out on first visit, in operand order, so a shared
  intermediate keeps one identity while two identical but separate
  computations keep two. Literals are operand markers without a value.
  """
  producers = values_by_name(graph)
  identifiers = {}
  records = []

  def visit(name):
    if name in identifiers:
      return identifiers[name]
    index = len(records)
    identifiers[name] = index
    records.append(None)
    value = producers[name]
    if value.is_input:
      record = ["input"]
    else:
      slots = [
        "literal" if isinstance(slot, Literal) else visit(slot)
        for slot in value.operands
      ]
      # Every operation carries the same unlabelled marker, a whole-value copy
      # included: the topology identity separates dependency structure from the
      # operators that fill it, so two graphs that differ only in which
      # operation occupies a node must hash alike.
      record = [f"operator:{value.operator}" if labelled else "operator", slots]
    records[index] = record
    return index

  return [visit(graph.result), records]


def topology_digest(graph):
  """Hash the ordered dependency topology, without operators, names, or values."""
  return _digest(["topology-v1", *_canonical(graph, labelled=False)])


def operator_digest(graph):
  """Hash the ordered dependency topology with operator names attached."""
  return _digest(["operator-graph-v1", *_canonical(graph, labelled=True)])


def source_digest(source):
  """Hash the source bytes, kept apart from both structural identities."""
  return hashlib.sha256(source.encode("utf-8")).hexdigest()


def shape_signature(graph):
  """Digest the declared dimensions only."""
  return _digest(["shapes-v1", [list(value.shape) for value in graph.values]])


def literal_signature(graph):
  """Digest the literal operand values only."""
  return _digest(
    [
      "literals-v1",
      [
        [slot.text for slot in value.operands if isinstance(slot, Literal)]
        for value in graph.values
      ],
    ]
  )


def axis_signature(graph):
  """Digest the axis, keep/drop, and attribute records only."""
  return _digest(
    [
      "axes-v1",
      [[value.axis, value.keep_dims, list(value.attribute)] for value in graph.values],
    ]
  )


def _render(value):
  """Render one admitted value the way the source spells it."""
  if value.is_input:
    return value.name
  if value.operator == "copy":
    return f"{value.name} = {value.operands[0]}"
  slots = ", ".join(
    slot.text if isinstance(slot, Literal) else slot for slot in value.operands
  )
  suffix = ""
  if value.operator in ("transpose", "broadcast"):
    suffix = ", [" + ", ".join(str(value_) for value_ in value.attribute) + "]"
  elif value.operator == "expand":
    suffix = f", {value.axis}"
  elif value.operator in REDUCTIONS:
    suffix = f", {value.axis}, {'keep' if value.keep_dims else 'drop'}"
  return f"{value.name} = tensor {value.operator} {slots}{suffix}"


def _ancestors(graph, name):
  """Return the strict ancestors of a value, restricted to the return slice."""
  producers = values_by_name(graph)
  live = return_slice(graph)
  seen = set()
  pending = [
    slot for slot in producers[name].operands if isinstance(slot, str) and slot in live
  ]
  while pending:
    current = pending.pop()
    if current in seen:
      continue
    seen.add(current)
    pending.extend(
      slot
      for slot in producers[current].operands
      if isinstance(slot, str) and slot in live
    )
  return seen


def _live_consumers(graph, name, live):
  sites = consumers(graph)
  return [
    graph.values[index].name
    for index in sites.get(name, [])
    if graph.values[index].name in live
  ]


def _descendants(graph, name, live):
  """Return the values reachable from a value, itself included."""
  sites = consumers(graph)
  seen = set()
  pending = [name]
  while pending:
    current = pending.pop()
    if current in seen or current not in live:
      continue
    seen.add(current)
    pending.extend(graph.values[index].name for index in sites.get(current, []))
  return seen


def feature_witnesses(graph):
  """Return one concrete witness per structural feature the graph reaches."""
  producers = values_by_name(graph)
  dependencies = input_dependencies(graph)
  live = return_slice(graph)
  witnesses = {}

  depths = {}
  deepest = {}
  for value in graph.values:
    operands = [slot for slot in value.operands if isinstance(slot, str)]
    if value.is_input:
      depths[value.name] = 0
    elif value.operator == "copy" or not operands:
      depths[value.name] = max((depths[slot] for slot in operands), default=0)
      deepest[value.name] = max(operands, key=lambda slot: depths[slot], default=None)
    else:
      deepest[value.name] = max(operands, key=lambda slot: depths[slot])
      depths[value.name] = depths[deepest[value.name]] + 1
  if depths.get(graph.result, 0) >= 2:
    path = [graph.result]
    while deepest.get(path[-1]) is not None:
      path.append(deepest[path[-1]])
    witnesses["chain"] = " -> ".join(reversed(path))

  order = {value.name: index for index, value in enumerate(graph.values)}
  for value in graph.values:
    if value.is_input:
      continue
    live_sites = [
      index
      for index in consumers(graph).get(value.name, [])
      if graph.values[index].name in live
    ]
    if len(live_sites) < 2:
      continue
    closures = [
      _descendants(graph, graph.values[index].name, live) for index in live_sites
    ]
    common = set.intersection(*closures)
    names = [graph.values[index].name for index in live_sites]
    # A reconvergence at one of the two consumers still counts: one branch can
    # feed the other and meet again there. Prefer a join the branches merely
    # meet at, so the witness names the reconvergence rather than a branch.
    elsewhere = [name for name in common if name not in names]
    if common:
      join = min(elsewhere or common, key=lambda name: order[name])
      witnesses["shared-intermediate-diamond"] = (
        f"{value.name} -> {{{', '.join(names)}}} -> {join}"
      )
      break

  for value in graph.values:
    if value.is_input:
      continue
    branches = [
      slot
      for slot in value.operands
      if isinstance(slot, str) and producers[slot].operator not in ("input", "copy")
    ]
    for position, left in enumerate(branches):
      for right in branches[position + 1 :]:
        if (
          dependencies[left]
          and dependencies[right]
          and not (dependencies[left] & dependencies[right])
        ):
          witnesses["independent-branch-join"] = (
            f"{_render(producers[left])}; {_render(producers[right])}; "
            f"joined at {_render(value)}"
          )
          break
      if "independent-branch-join" in witnesses:
        break
    if "independent-branch-join" in witnesses:
      break

  reductions = {}
  for value in graph.values:
    if value.operator not in REDUCTIONS or value.name not in live:
      continue
    live_consumers = frozenset(_live_consumers(graph, value.name, live))
    if live_consumers:
      reductions[value.name] = live_consumers
  names = sorted(reductions)
  for position, first in enumerate(names):
    for second in names[position + 1 :]:
      if reductions[first] != reductions[second]:
        witnesses["dual-live-reductions"] = (
          f"{_render(producers[first])}; {_render(producers[second])}"
        )
        break
    if "dual-live-reductions" in witnesses:
      break

  for value in graph.values:
    if value.operator not in REDUCTIONS or value.name not in live:
      continue
    upstream = sorted(
      name
      for name in _ancestors(graph, value.name)
      if producers[name].operator in SHAPE_OPERATORS
    )
    if upstream:
      witnesses["transpose-or-broadcast-upstream-of-reduction"] = (
        f"{_render(producers[upstream[0]])}; consumed by {_render(value)}"
      )
      break

  for value in graph.values:
    if value.operator not in REDUCTIONS or value.name not in live or value.shape:
      continue
    expanded = sorted(
      name
      for name in _live_consumers(graph, value.name, live)
      if producers[name].operator == "expand"
    )
    if expanded:
      witnesses["scalar-reduction-then-expand"] = (
        f"{_render(value)}; {_render(producers[expanded[0]])}"
      )
      break

  return witnesses


@dataclass(frozen=True)
class GeneratedCase:
  """One generation attempt: its source when it produced a case."""

  profile: str
  seed: int
  accepted: bool
  reason: str
  source: str
  case: TensorCase | None = None


@dataclass(frozen=True)
class CaseAudit:
  """One audited seed. A refusal keeps its reason in the denominator."""

  profile: str
  seed: int
  accepted: bool
  reason: str
  source: str
  statements: tuple[str, ...]
  source_id: str
  topology_id: str
  operator_id: str
  shape_id: str
  literal_id: str
  axis_id: str
  features: tuple[str, ...]
  witnesses: dict[str, str]
  declared_inputs: tuple[str, ...] = ()
  concrete_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProfileAudit:
  """Aggregate of one profile's audit."""

  profile: str
  attempted: int
  accepted: int
  refusals: dict[str, int]
  source_count: int
  topology_count: int
  operator_count: int
  shape_count: int
  literal_count: int
  axis_count: int
  feature_counts: dict[str, int]
  feature_witnesses: dict[str, dict]
  concrete_dependence_cases: int
  cases: tuple[CaseAudit, ...]


@dataclass(frozen=True)
class AuditReport:
  """The whole bounded qualification: one profile audit per profile."""

  seed_count: int
  profiles: tuple[ProfileAudit, ...]


def refusal_key(reason):
  """Bucket one refusal reason under a stable report key."""
  head = reason.split(":", 1)[0].strip()
  return head or "refused"


def _read_summary(output_dir):
  path = Path(output_dir) / "generation.json"
  if not path.is_file():
    return {}
  try:
    return json.loads(path.read_text(encoding="utf-8"))
  except json.JSONDecodeError:
    return {}


def _first_line(text):
  stripped = (text or "").strip()
  return stripped.splitlines()[0] if stripped else ""


def _generation_reason(completed, summary):
  status = summary.get("status") or "failed"
  attempts = summary.get("attempts") or []
  detail = attempts[-1].get("reason") if attempts else summary.get("reason")
  if detail:
    return f"{status}: {detail}"
  if completed.returncode != 0:
    return f"{status}: rysmith exited {completed.returncode}"
  return status


def generate_case(profile, seed, *, root, rysmith=RYSMITH, timeout=120):
  """Run one compositional generation attempt under default graph limits."""
  output_dir = Path(root) / f"{profile}-{seed:06d}"
  command = [
    str(rysmith),
    "--profile",
    "tensor",
    "--target",
    "sir",
    "-n",
    "1",
    "--seed",
    str(seed),
    "--max-retries",
    "1",
    "--output-dir",
    str(output_dir),
    "--tensor-dtype",
    "i32",
    "--tensor-profile-version",
    profile,
    "--tensor-generator",
    "compositional",
    "--tensor-input-policy",
    "ordinary",
  ]
  try:
    completed = subprocess.run(
      command, capture_output=True, text=True, check=False, timeout=timeout
    )
  except subprocess.TimeoutExpired:
    return GeneratedCase(profile, seed, False, "generation_timeout", "", None)
  summary = _read_summary(output_dir)
  if completed.returncode != 0 or summary.get("emitted") != 1:
    return GeneratedCase(
      profile, seed, False, _generation_reason(completed, summary), "", None
    )
  try:
    case = load_case(output_dir / "case_000000")
  except CaseValidationError as error:
    return GeneratedCase(profile, seed, False, f"invalid_case: {error}", "", None)
  return GeneratedCase(profile, seed, True, "", case.program.decode("utf-8"), case)


def _refused(profile, seed, reason, source="", statements=(), **identities):
  return CaseAudit(
    profile=profile,
    seed=seed,
    accepted=False,
    reason=reason,
    source=source,
    statements=statements,
    source_id=identities.get("source_id", ""),
    topology_id=identities.get("topology_id", ""),
    operator_id=identities.get("operator_id", ""),
    shape_id=identities.get("shape_id", ""),
    literal_id=identities.get("literal_id", ""),
    axis_id=identities.get("axis_id", ""),
    features=(),
    witnesses={},
  )


def audit_case(profile, seed, *, root, rysmith=RYSMITH, timeout=120):
  """Generate, check, parse, and independently replay one seeded case."""
  generated = generate_case(profile, seed, root=root, rysmith=rysmith, timeout=timeout)
  if not generated.accepted:
    return _refused(profile, seed, generated.reason)
  source = generated.source
  statements = tuple(
    line.strip()
    for line in source.splitlines()
    if line.strip() and not line.strip().startswith(("fun ", "let ", "^entry:", "}"))
  )
  checked = check_source(source, tensor_profile=profile)
  if checked.returncode != 0:
    return _refused(
      profile,
      seed,
      f"check_failed: {_first_line(checked.stderr) or _first_line(checked.stdout)}",
      source=source,
    )
  try:
    graph = parse_graph(source)
  except UnsupportedSource as error:
    return _refused(profile, seed, f"unsupported_source: {error}", source=source)
  identities = {
    "source_id": source_digest(source),
    "topology_id": topology_digest(graph),
    "operator_id": operator_digest(graph),
    "shape_id": shape_signature(graph),
    "literal_id": literal_signature(graph),
    "axis_id": axis_signature(graph),
  }
  dead = sorted(
    value.name for value in graph.computations if value.name not in return_slice(graph)
  )
  if dead:
    return _refused(
      profile,
      seed,
      f"dead_computation: {', '.join(dead)}",
      source,
      statements,
      **identities,
    )
  manifest = generated.case.manifest
  policy = manifest.get("generator", {}).get("input_policy")
  if policy != "ordinary":
    return _refused(
      profile,
      seed,
      f"unexpected_input_policy: {policy}",
      source,
      statements,
      **identities,
    )
  dtypes = {record["dtype"] for record in manifest["inputs"]}
  dtypes.add(manifest["output"]["dtype"])
  if dtypes != {"i32"}:
    return _refused(
      profile,
      seed,
      f"unexpected_dtype: {', '.join(sorted(dtypes))}",
      source,
      statements,
      **identities,
    )
  replay, actual = run_typed(source, generated.case.inputs, tensor_profile=profile)
  if replay.returncode != 0:
    return _refused(
      profile,
      seed,
      f"reference_failed: {_first_line(replay.stderr) or _first_line(replay.stdout)}",
      source,
      statements,
      **identities,
    )
  if actual != generated.case.expected:
    return _refused(
      profile, seed, "reference_mismatch", source, statements, **identities
    )
  witnesses = feature_witnesses(graph)
  # Concrete dependence is the builder's own counterfactual record: it replaces
  # each input buffer and observes a different output. It is reported apart from
  # the syntactic input reachability of the topology, which alone does not show
  # that two inputs are semantically interchangeable.
  behavior = manifest.get("coverage", {}).get("input_behavior", {})
  concrete = tuple(
    sorted(
      item["input"]
      for item in behavior.get("counterfactual_inputs", [])
      if item.get("outputs_differ")
    )
  )
  return CaseAudit(
    profile=profile,
    seed=seed,
    accepted=True,
    reason="",
    source=source,
    statements=statements,
    features=tuple(sorted(witnesses)),
    witnesses=witnesses,
    declared_inputs=tuple(record["name"] for record in manifest["inputs"]),
    concrete_inputs=concrete,
    **identities,
  )


def _witness_record(case, feature):
  return {
    "seed": case.seed,
    "witness": case.witnesses[feature],
    "statements": list(case.statements),
    "source": case.source,
  }


def audit_profile(
  profile, *, seeds=None, root=None, jobs=1, rysmith=RYSMITH, timeout=120
):
  """Audit one profile over `seeds`, keeping every attempt in the denominator."""
  if profile not in PROFILES:
    raise ValueError(f"the audit covers {', '.join(PROFILES)}, not {profile!r}")
  if not Path(rysmith).is_file():
    raise RuntimeError(f"rysmith executable not found: {rysmith}")
  seeds = list(range(SEED_COUNT)) if seeds is None else [int(seed) for seed in seeds]

  def run_all(directory):
    def audit(seed):
      return audit_case(profile, seed, root=directory, rysmith=rysmith, timeout=timeout)

    if jobs > 1:
      with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(audit, seeds))
    return [audit(seed) for seed in seeds]

  if root is None:
    with tempfile.TemporaryDirectory() as directory:
      cases = run_all(directory)
  else:
    cases = run_all(root)

  accepted = [case for case in cases if case.accepted]
  refusals = Counter(refusal_key(case.reason) for case in cases if not case.accepted)
  feature_counts = Counter(feature for case in accepted for feature in case.features)
  witnesses = {}
  for case in accepted:
    for feature in case.features:
      witnesses.setdefault(feature, _witness_record(case, feature))
  return ProfileAudit(
    profile=profile,
    attempted=len(cases),
    accepted=len(accepted),
    refusals=dict(sorted(refusals.items())),
    source_count=len({case.source_id for case in accepted}),
    topology_count=len({case.topology_id for case in accepted}),
    operator_count=len({case.operator_id for case in accepted}),
    shape_count=len({case.shape_id for case in accepted}),
    literal_count=len({case.literal_id for case in accepted}),
    axis_count=len({case.axis_id for case in accepted}),
    feature_counts={feature: feature_counts[feature] for feature in FEATURES},
    feature_witnesses=witnesses,
    concrete_dependence_cases=sum(
      1
      for case in accepted
      if case.concrete_inputs and set(case.concrete_inputs) == set(case.declared_inputs)
    ),
    cases=tuple(cases),
  )


def audit(
  *, seeds=None, profiles=PROFILES, root=None, jobs=1, rysmith=RYSMITH, timeout=120
):
  """Audit every requested profile over the same seeds."""
  seed_list = (
    list(range(SEED_COUNT)) if seeds is None else [int(seed) for seed in seeds]
  )
  reports = tuple(
    audit_profile(
      profile, seeds=seed_list, root=root, jobs=jobs, rysmith=rysmith, timeout=timeout
    )
    for profile in profiles
  )
  return AuditReport(seed_count=len(seed_list), profiles=reports)


def _report_record(report):
  """The JSON-serializable form of an audit report."""
  return {
    "schema": "refractir.tensor-graph-audit.v1",
    "seed_count": report.seed_count,
    "profiles": [
      {
        "profile": profile.profile,
        "attempted": profile.attempted,
        "accepted": profile.accepted,
        "refusals": profile.refusals,
        "identities": {
          "sources": profile.source_count,
          "topologies": profile.topology_count,
          "operator_graphs": profile.operator_count,
        },
        "recorded_separately": {
          "dimension_signatures": profile.shape_count,
          "literal_signatures": profile.literal_count,
          "axis_signatures": profile.axis_count,
          "concrete_dependence_cases": profile.concrete_dependence_cases,
        },
        "features": {
          feature: {
            "count": profile.feature_counts[feature],
            "witness": profile.feature_witnesses.get(feature),
          }
          for feature in FEATURES
        },
        "cases": [
          {
            "seed": case.seed,
            "accepted": case.accepted,
            "reason": case.reason,
            "source_id": case.source_id,
            "topology_id": case.topology_id,
            "operator_id": case.operator_id,
            "shape_id": case.shape_id,
            "literal_id": case.literal_id,
            "axis_id": case.axis_id,
            "features": list(case.features),
            "statements": list(case.statements),
          }
          for case in profile.cases
        ],
      }
      for profile in report.profiles
    ],
  }


def render_report(report):
  """Render the audit report as markdown, keeping each identity separate."""
  lines = [
    "# Compositional tensor graph dependency audit",
    "",
    f"Bounded qualification: {report.seed_count} seeds per profile, v2 and v3, "
    "default graph limits, ordinary i32 inputs. Every generation failure stays "
    "in the denominator. Accepted means generated, accepted by "
    "`symiri --check --tensor-profile-version <profile>`, parsed as one admitted "
    "straight-line `^entry` block whose every computation lies in the return "
    "slice, and reproduced byte for byte by an independent reference run.",
    "",
  ]
  for profile in report.profiles:
    lines.extend(
      [
        f"## tensor-{profile.profile}",
        "",
        f"Attempted {profile.attempted}, accepted {profile.accepted}.",
        "",
      ]
    )
    if profile.refusals:
      lines.append("| refusal | count |")
      lines.append("|---|---|")
      for reason, count in profile.refusals.items():
        lines.append(f"| {reason} | {count} |")
      lines.append("")
    lines.extend(
      [
        "| identity | distinct |",
        "|---|---|",
        f"| source bytes | {profile.source_count} |",
        f"| ordered dependency topologies | {profile.topology_count} |",
        f"| operator-labelled graphs | {profile.operator_count} |",
        f"| declared dimensions (recorded separately) | {profile.shape_count} |",
        f"| literal operands (recorded separately) | {profile.literal_count} |",
        f"| axis attributes (recorded separately) | {profile.axis_count} |",
        f"| accepted cases with recorded concrete input dependence "
        f"| {profile.concrete_dependence_cases} |",
        "",
        "The last row is concrete dependence: the builder replaced each input "
        "buffer and observed a different output. It is not the syntactic input "
        "reachability of the topology, which alone does not show that two inputs "
        "are interchangeable.",
        "",
        "Distinct sources count generated programs. They are not a measure of "
        "optimized compiler coverage, and source diversity alone is not a claim "
        "about compiled Triton behavior.",
        "",
        "| feature | cases | witness seed | witness |",
        "|---|---|---|---|",
      ]
    )
    for feature in FEATURES:
      record = profile.feature_witnesses.get(feature)
      seed = record["seed"] if record else "-"
      witness = record["witness"] if record else "absent"
      lines.append(
        f"| {feature} | {profile.feature_counts[feature]} | {seed} | {witness} |"
      )
    lines.append("")
    for feature in FEATURES:
      record = profile.feature_witnesses.get(feature)
      if record is None:
        continue
      lines.extend(
        [
          f"### {profile.profile} witness: {feature} (seed {record['seed']})",
          "",
          "```sir",
          record["source"].rstrip(),
          "```",
          "",
        ]
      )
  return "\n".join(lines).rstrip() + "\n"


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--seeds", type=int, default=SEED_COUNT)
  parser.add_argument("--profiles", default=",".join(PROFILES))
  parser.add_argument("--jobs", type=int, default=1)
  parser.add_argument("--root", type=Path, default=None)
  arguments = parser.parse_args(argv)

  profiles = tuple(
    value.strip() for value in arguments.profiles.split(",") if value.strip()
  )
  report = audit(
    seeds=range(arguments.seeds),
    profiles=profiles,
    root=arguments.root,
    jobs=arguments.jobs,
  )
  arguments.output_dir.mkdir(parents=True, exist_ok=True)
  json_path = arguments.output_dir / "tensor-graph-audit.json"
  markdown_path = arguments.output_dir / "tensor-graph-audit.md"
  json_path.write_text(
    json.dumps(_report_record(report), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  markdown_path.write_text(render_report(report), encoding="utf-8")
  for profile in report.profiles:
    print(
      f"tensor-{profile.profile}: attempted {profile.attempted}, "
      f"accepted {profile.accepted}, topologies {profile.topology_count}, "
      f"operator graphs {profile.operator_count}, "
      f"refusals {profile.refusals or 'none'}"
    )
    for feature in FEATURES:
      record = profile.feature_witnesses.get(feature)
      print(
        f"  {feature}: {profile.feature_counts[feature]}"
        + (f" (seed {record['seed']}: {record['witness']})" if record else " (absent)")
      )
  print(f"wrote {json_path} and {markdown_path}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
