"""Deterministic workload and source rendering for the tensor-property pilot."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeAlias

SCHEMA = "refractir.tensor-property-problem.v1"
WORKLOAD_SCHEMA = "refractir.tensor-property-workload.v1"
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BINARY = {"add", "sub", "mul"}
_FAMILIES = (
  "independent_sum",
  "affine_sum",
  "layout_sum",
  "shared_reductions",
  "nonlinear",
)

Operand: TypeAlias = str | int


@dataclass(frozen=True, slots=True)
class InputTensor:
  name: str
  shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GraphNode:
  node_id: str
  op: str
  operands: tuple[Operand, ...]
  shape: tuple[int, ...]
  axis: int | None = None
  keep: bool = False
  permutation: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class PilotProblem:
  graph_id: str
  problem_id: str
  family: str
  symbol_count: int
  inputs: tuple[InputTensor, ...]
  output_shape: tuple[int, ...]
  domain: tuple[int, int]
  target: tuple[int, ...]
  target_pattern: str
  nodes: tuple[GraphNode, ...]
  output_id: str

  @property
  def input_names(self) -> tuple[str, ...]:
    return tuple(item.name for item in self.inputs)

  @property
  def input_shapes(self) -> tuple[tuple[int, ...], ...]:
    return tuple(item.shape for item in self.inputs)


def _element_count(shape: tuple[int, ...]) -> int:
  count = 1
  for dimension in shape:
    count *= dimension
  return count


def _check_integer(value: object, what: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise ValueError(f"{what} must be an integer")
  return value


def _check_shape(
  shape: tuple[int, ...], what: str, *, allow_scalar: bool = False
) -> None:
  if not isinstance(shape, tuple):
    raise ValueError(f"{what} must be a tuple")
  if (not shape and not allow_scalar) or len(shape) > 3:
    raise ValueError(f"{what} rank must be between one and three")
  for dimension in shape:
    if isinstance(dimension, bool) or not isinstance(dimension, int):
      raise ValueError(f"{what} dimensions must be integers")
    if dimension <= 0 or dimension & (dimension - 1):
      raise ValueError(f"{what} dimensions must be positive powers of two")
  if _element_count(shape) > 256:
    raise ValueError(f"{what} exceeds 256 elements")


def _broadcast_shape(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
  rank = max(len(left), len(right))
  padded_left = (1,) * (rank - len(left)) + left
  padded_right = (1,) * (rank - len(right)) + right
  result = []
  for lhs, rhs in zip(padded_left, padded_right):
    if lhs != rhs and lhs != 1 and rhs != 1:
      raise ValueError("binary tensor operands cannot be broadcast together")
    result.append(max(lhs, rhs))
  return tuple(result)


def _operand_shape(
  operand: Operand, shapes: dict[str, tuple[int, ...]]
) -> tuple[int, ...]:
  if isinstance(operand, str):
    try:
      return shapes[operand]
    except KeyError as error:
      raise ValueError(
        f"graph operand {operand!r} is not defined before use"
      ) from error
  _check_integer(operand, "literal operand")
  return ()


def validate_problem(problem: PilotProblem) -> None:
  if not isinstance(problem, PilotProblem):
    raise ValueError("problem must be a PilotProblem")
  if not _NAME.fullmatch(problem.graph_id) or not _NAME.fullmatch(problem.problem_id):
    raise ValueError("graph and problem identifiers must be simple names")
  if problem.family not in _FAMILIES:
    raise ValueError(f"unknown graph family {problem.family!r}")
  if not isinstance(problem.inputs, tuple) or not problem.inputs:
    raise ValueError("a problem requires an ordered tuple of inputs")
  seen: set[str] = set()
  symbol_count = 0
  shapes: dict[str, tuple[int, ...]] = {}
  for item in problem.inputs:
    if (
      not isinstance(item, InputTensor)
      or not _NAME.fullmatch(item.name)
      or item.name in seen
    ):
      raise ValueError("input names must be unique simple identifiers")
    _check_shape(item.shape, f"input {item.name}")
    seen.add(item.name)
    shapes[item.name] = item.shape
    symbol_count += _element_count(item.shape)
  if _check_integer(problem.symbol_count, "symbol_count") != symbol_count:
    raise ValueError("symbol_count does not match the ordered input tensors")
  _check_shape(problem.output_shape, "output", allow_scalar=True)
  if _element_count(problem.output_shape) > 8:
    raise ValueError("returned values may contain at most eight elements")
  if not isinstance(problem.domain, tuple) or len(problem.domain) != 2:
    raise ValueError("domain must be a pair of integer bounds")
  lower, upper = (_check_integer(v, "domain bound") for v in problem.domain)
  if lower > upper or lower < -(1 << 31) or upper >= 1 << 31:
    raise ValueError("domain must be an ordered signed-i32 interval")
  if not isinstance(problem.target, tuple) or len(problem.target) != _element_count(
    problem.output_shape
  ):
    raise ValueError("target element count does not match the output shape")
  for value in problem.target:
    value = _check_integer(value, "target element")
    if value < -(1 << 31) or value >= 1 << 31:
      raise ValueError("target element is outside signed-i32 range")
  if not isinstance(problem.target_pattern, str) or not problem.target_pattern:
    raise ValueError("target_pattern must be a non-empty string")
  if (
    not isinstance(problem.nodes, tuple) or not problem.nodes or len(problem.nodes) > 8
  ):
    raise ValueError("a graph must contain between one and eight operator nodes")

  nodes_by_id: dict[str, GraphNode] = {}
  for node in problem.nodes:
    if not isinstance(node, GraphNode) or not _NAME.fullmatch(node.node_id):
      raise ValueError("node identifiers must be simple names")
    if node.node_id in shapes or node.node_id in nodes_by_id:
      raise ValueError(f"duplicate graph value {node.node_id!r}")
    _check_shape(node.shape, f"node {node.node_id}", allow_scalar=True)
    if not isinstance(node.operands, tuple):
      raise ValueError("node operands must be an ordered tuple")
    if node.axis is not None:
      _check_integer(node.axis, "node axis")
    if not isinstance(node.keep, bool):
      raise ValueError("node keep attribute must be boolean")
    if not isinstance(node.permutation, tuple):
      raise ValueError("node permutation must be a tuple")
    for axis in node.permutation:
      _check_integer(axis, "permutation axis")
    operand_shapes = tuple(_operand_shape(operand, shapes) for operand in node.operands)
    if node.op in _BINARY:
      if len(node.operands) != 2 or _broadcast_shape(*operand_shapes) != node.shape:
        raise ValueError(f"node {node.node_id} has invalid {node.op} shape or arity")
      if node.axis is not None or node.keep or node.permutation:
        raise ValueError(f"node {node.node_id} has irrelevant binary attributes")
    elif node.op == "sum":
      if len(node.operands) != 1 or not operand_shapes[0]:
        raise ValueError("sum requires one tensor operand")
      rank = len(operand_shapes[0])
      if node.axis is None or not -rank <= node.axis < rank:
        raise ValueError(f"node {node.node_id} has an invalid reduction axis")
      axis = node.axis % rank
      expected = (
        operand_shapes[0][:axis] + (1,) + operand_shapes[0][axis + 1 :]
        if node.keep
        else operand_shapes[0][:axis] + operand_shapes[0][axis + 1 :]
      )
      if expected != node.shape or node.permutation:
        raise ValueError(f"node {node.node_id} has invalid sum shape or attributes")
    elif node.op == "transpose":
      if len(node.operands) != 1 or not operand_shapes[0]:
        raise ValueError("transpose requires one tensor operand")
      rank = len(operand_shapes[0])
      if (
        len(node.permutation) != rank
        or set(node.permutation) != set(range(rank))
        or tuple(operand_shapes[0][axis] for axis in node.permutation) != node.shape
        or node.axis is not None
        or node.keep
      ):
        raise ValueError(f"node {node.node_id} has invalid transpose attributes")
    elif node.op == "broadcast":
      if (
        len(node.operands) != 1
        or node.axis is not None
        or node.keep
        or node.permutation
      ):
        raise ValueError("broadcast requires one operand and no other attributes")
      source = operand_shapes[0]
      if len(source) > len(node.shape):
        raise ValueError("broadcast cannot reduce rank")
      padded = (1,) * (len(node.shape) - len(source)) + source
      if any(src not in (1, dst) for src, dst in zip(padded, node.shape)):
        raise ValueError("broadcast target is incompatible with its source")
    else:
      raise ValueError(f"unknown graph operation {node.op!r}")
    shapes[node.node_id] = node.shape
    nodes_by_id[node.node_id] = node

  if (
    problem.output_id not in nodes_by_id
    or shapes[problem.output_id] != problem.output_shape
  ):
    raise ValueError("output identity or shape does not match the returned graph value")
  dependencies: set[str] = set()
  visited_nodes: set[str] = set()

  def visit(value: str) -> None:
    if value in seen:
      dependencies.add(value)
      return
    if value in visited_nodes:
      return
    node = nodes_by_id.get(value)
    if node is None:
      raise ValueError(f"returned graph depends on undefined value {value!r}")
    visited_nodes.add(value)
    for operand in node.operands:
      if isinstance(operand, str):
        visit(operand)

  visit(problem.output_id)
  if dependencies != seen:
    unused = sorted(seen - dependencies)
    raise ValueError(f"runtime inputs do not influence the result: {unused}")
  if visited_nodes != set(nodes_by_id):
    dead = sorted(set(nodes_by_id) - visited_nodes)
    raise ValueError(f"graph contains dead operator nodes: {dead}")
  if problem.family == "nonlinear":
    nonlinear_products = [
      node
      for node in problem.nodes
      if node.op == "mul"
      and len(node.operands) == 2
      and all(isinstance(operand, str) for operand in node.operands)
    ]
    if len(nonlinear_products) != 1:
      raise ValueError(
        "nonlinear family requires one unknown-by-unknown multiplication"
      )


def _shape_text(shape: tuple[int, ...]) -> str:
  return "i32" if not shape else f"tensor [{', '.join(str(d) for d in shape)}] i32"


def _reference(name: str, query: bool, input_names: set[str]) -> str:
  if name in input_names:
    return f"%?{name}" if query else f"%{name}"
  return f"%{name}"


def _coordinates(linear: int, shape: tuple[int, ...]) -> tuple[int, ...]:
  coordinates = [0] * len(shape)
  for axis in range(len(shape) - 1, -1, -1):
    coordinates[axis] = linear % shape[axis]
    linear //= shape[axis]
  return tuple(coordinates)


def _node_statement(node: GraphNode, *, query: bool, input_names: set[str]) -> str:
  operands = [
    _reference(operand, query, input_names)
    if isinstance(operand, str)
    else str(operand)
    for operand in node.operands
  ]
  if node.op in _BINARY:
    expression = f"tensor {node.op} {operands[0]}, {operands[1]}"
  elif node.op == "sum":
    assert node.axis is not None
    expression = (
      f"tensor sum {operands[0]}, {node.axis}, {'keep' if node.keep else 'drop'}"
    )
  elif node.op == "transpose":
    permutation = ", ".join(str(axis) for axis in node.permutation)
    expression = f"tensor transpose {operands[0]}, [{permutation}]"
  elif node.op == "broadcast":
    dimensions = ", ".join(str(dimension) for dimension in node.shape)
    expression = f"tensor broadcast {operands[0]}, [{dimensions}]"
  else:
    raise ValueError(f"cannot render operation {node.op!r}")
  return f"  %{node.node_id} = {expression};"


def _render(problem: PilotProblem, *, query: bool, with_target: bool) -> str:
  validate_problem(problem)
  input_names = set(problem.input_names)
  if query:
    parameters = ""
    return_type = "i32"
  else:
    parameters = ", ".join(
      f"%{item.name}: {_shape_text(item.shape)}" for item in problem.inputs
    )
    return_type = _shape_text(problem.output_shape)
  lines = [f"fun @kernel({parameters}) : {return_type} {{"]
  if query:
    lower, upper = problem.domain
    for item in problem.inputs:
      lines.append(
        f"  sym %?{item.name} : value {_shape_text(item.shape)} in [{lower}, {upper}];"
      )
  for node in problem.nodes:
    lines.append(f"  let mut %{node.node_id}: {_shape_text(node.shape)} = undef;")
  target_count = _element_count(problem.output_shape)
  if with_target and problem.output_shape:
    for index in range(target_count):
      lines.append(f"  let mut %__target_{index}: i32 = undef;")
  lines.append("^entry:")
  lines.extend(
    _node_statement(node, query=query, input_names=input_names)
    for node in problem.nodes
  )
  if with_target:
    for index, target in enumerate(problem.target):
      if problem.output_shape:
        coordinates = _coordinates(index, problem.output_shape)
        index_text = ", ".join(str(value) for value in coordinates)
        lines.append(
          f"  %__target_{index} = tensor extract %{problem.output_id}, [{index_text}];"
        )
        actual = f"%__target_{index}"
      else:
        actual = f"%{problem.output_id}"
      lines.append(f"  require {actual} == {target};")
  lines.append("  ret 0;" if query else f"  ret %{problem.output_id};")
  lines.append("}")
  return "\n".join(lines) + "\n"


def render_query(problem: PilotProblem) -> str:
  return _render(problem, query=True, with_target=True)


def render_runtime(problem: PilotProblem) -> str:
  return _render(problem, query=False, with_target=True)


def render_reference(problem: PilotProblem) -> str:
  return _render(problem, query=False, with_target=False)


def normalized_computation(source: str) -> tuple[str, ...]:
  """Return normalized graph operations, excluding property-only extraction."""
  lines = source.splitlines()
  try:
    start = (
      next(index for index, line in enumerate(lines) if line.strip() == "^entry:") + 1
    )
  except StopIteration as error:
    raise ValueError("source has no ^entry block") from error
  result = []
  for line in lines[start:]:
    statement = line.strip()
    if statement.startswith("ret "):
      break
    if " = tensor " not in statement or " = tensor extract " in statement:
      continue
    result.append(statement.replace("%?", "%"))
  return tuple(result)


def _matrix_shape(symbol_count: int, rng: random.Random) -> tuple[int, int]:
  choices = {16: ((2, 8), (4, 4), (8, 2)), 32: ((4, 8), (8, 4)), 64: ((8, 8),)}
  return rng.choice(choices[symbol_count])


def _sum_shape(symbol_count: int, rng: random.Random) -> tuple[int, ...]:
  choices = {
    16: ((16,), (2, 8), (4, 4), (8, 2)),
    32: ((32,), (4, 8), (8, 4)),
    64: ((64,), (8, 8)),
  }
  return rng.choice(choices[symbol_count])


def _target_candidate(
  shape: tuple[int, ...], rng: random.Random
) -> tuple[str, tuple[int, ...]]:
  count = _element_count(shape)
  if count == 1:
    return "scalar_menu", (rng.choice((0, -3, -1, 1, 3)),)
  pattern = rng.choice(("zero", "repeated", "mixed_sign", "distinct"))
  if pattern == "zero":
    return pattern, (0,) * count
  if pattern == "repeated":
    value = rng.choice((-3, -1, 1, 3))
    return pattern, (value,) * count
  if pattern == "distinct":
    return pattern, tuple(rng.sample(range(-8, 9), count))
  negative = rng.choice((-3, -1))
  positive = rng.choice((1, 3))
  values = [negative, positive]
  values.extend(rng.choice((-3, -1, 1, 3)) for _ in range(count - 2))
  rng.shuffle(values)
  return pattern, tuple(values)


def _target_pair(
  shape: tuple[int, ...], rng: random.Random
) -> tuple[tuple[str, tuple[int, ...]], tuple[str, tuple[int, ...]]]:
  first = _target_candidate(shape, rng)
  for _ in range(64):
    second = _target_candidate(shape, rng)
    if second[1] != first[1]:
      return first, second
  count = _element_count(shape)
  fallback = next(
    (value,) * count for value in (-3, -1, 1, 3) if (value,) * count != first[1]
  )
  return first, ("repeated", fallback)


def select_targets(
  output_shape: tuple[int, ...], seed: int
) -> tuple[tuple[str, tuple[int, ...]], tuple[str, tuple[int, ...]]]:
  """Choose two distinct targets using only a local seeded random stream."""
  _check_shape(output_shape, "target output", allow_scalar=True)
  return _target_pair(output_shape, random.Random(_check_integer(seed, "seed")))


def _make_graph(
  graph_id: str,
  family: str,
  symbol_count: int,
  rng: random.Random,
  variant: int,
) -> tuple[tuple[InputTensor, ...], tuple[GraphNode, ...], tuple[int, ...], str]:
  if family == "independent_sum":
    shape = _sum_shape(symbol_count, rng)
    axis = (variant + rng.randrange(3)) % len(shape)
    if len(shape) == 2 and shape[1 - axis] > 8:
      axis = 1 - axis
    out_shape = shape[:axis] + shape[axis + 1 :]
    node = GraphNode("n0", "sum", ("x",), out_shape, axis=axis)
    return (InputTensor("x", shape),), (node,), out_shape, "n0"

  if family == "affine_sum":
    shape = _sum_shape(symbol_count, rng)
    coefficient = rng.choice((-3, -2, -1, 1, 2, 3))
    offset = rng.choice((-3, -1, 1, 3))
    axis = rng.randrange(len(shape))
    out_shape = shape[:axis] + shape[axis + 1 :]
    if _element_count(out_shape) > 8:
      axis = len(shape) - 1
      out_shape = shape[:axis] + shape[axis + 1 :]
    nodes = (
      GraphNode("n0", "mul", ("x", coefficient), shape),
      GraphNode("n1", "add", ("n0", offset), shape),
      GraphNode("n2", "sum", ("n1",), out_shape, axis=axis),
    )
    return (InputTensor("x", shape),), nodes, out_shape, "n2"

  if family == "layout_sum":
    if variant % 2 == 0:
      input_shape = _matrix_shape(symbol_count, rng)
      transpose_shape = (input_shape[1], input_shape[0])
      axis = rng.randrange(2)
      out_shape = transpose_shape[:axis] + transpose_shape[axis + 1 :]
      arithmetic = rng.choice(("add", "sub", "mul"))
      literal = rng.choice((-3, -2, -1, 1, 2, 3))
      nodes = (
        GraphNode("n0", "transpose", ("x",), transpose_shape, permutation=(1, 0)),
        GraphNode("n1", arithmetic, ("n0", literal), transpose_shape),
        GraphNode("n2", "sum", ("n1",), out_shape, axis=axis),
      )
      return (InputTensor("x", input_shape),), nodes, out_shape, "n2"
    input_shape = (1, symbol_count)
    output_shape = (2, symbol_count)
    literal = rng.choice((-3, -2, -1, 1, 2, 3))
    arithmetic = rng.choice(("add", "sub", "mul"))
    nodes = (
      GraphNode("n0", "broadcast", ("x",), output_shape),
      GraphNode("n1", arithmetic, ("n0", literal), output_shape),
      GraphNode("n2", "sum", ("n1",), (2,), axis=1),
    )
    return (InputTensor("x", input_shape),), nodes, (2,), "n2"

  if family == "shared_reductions":
    rows, columns = _matrix_shape(symbol_count, rng)
    shape = (rows, columns)
    row_shape = (rows, 1)
    column_shape = (columns,)
    nodes = (
      GraphNode("n0", "add", ("x", "x"), shape),
      GraphNode("n1", "sum", ("n0",), row_shape, axis=1, keep=True),
      GraphNode("n2", "broadcast", ("n1",), shape),
      GraphNode("n3", "add", ("n0", "n2"), shape),
      GraphNode("n4", "sum", ("n3",), column_shape, axis=0),
      GraphNode("n5", "sum", ("n0",), column_shape, axis=0),
      GraphNode("n6", "add", ("n4", "n5"), column_shape),
    )
    return (InputTensor("x", shape),), nodes, column_shape, "n6"

  if family == "nonlinear":
    per_input = symbol_count // 2
    shape_choices = {
      8: ((8,), (2, 4), (4, 2)),
      16: ((16,), (2, 8), (4, 4), (8, 2)),
      32: ((32,), (4, 8), (8, 4)),
    }
    shape = rng.choice(shape_choices[per_input])
    axis = rng.randrange(len(shape))
    out_shape = shape[:axis] + shape[axis + 1 :]
    if _element_count(out_shape) > 8:
      axis = len(shape) - 1
      out_shape = shape[:axis] + shape[axis + 1 :]
    nodes = (
      GraphNode("n0", "mul", ("x", "y"), shape),
      GraphNode("n1", "sum", ("n0",), out_shape, axis=axis),
    )
    return (InputTensor("x", shape), InputTensor("y", shape)), nodes, out_shape, "n1"

  raise ValueError(f"unknown family {family!r} for {graph_id}")


def _make_development_shared_reductions(seed: int) -> list[PilotProblem]:
  """Provide a second reduction orientation for the held-apart dev set."""
  shape = (4, 8)
  output_shape = (4,)
  nodes = (
    GraphNode("n0", "add", ("x", "x"), shape),
    GraphNode("n1", "sum", ("n0",), (1, 8), axis=0, keep=True),
    GraphNode("n2", "broadcast", ("n1",), shape),
    GraphNode("n3", "add", ("n0", "n2"), shape),
    GraphNode("n4", "sum", ("n3",), output_shape, axis=1),
    GraphNode("n5", "sum", ("n0",), output_shape, axis=1),
    GraphNode("n6", "add", ("n4", "n5"), output_shape),
  )
  target_rng = random.Random((seed ^ 0x9E3779B9) & 0xFFFFFFFF)
  targets = _target_pair(output_shape, target_rng)
  return [
    _make_problem(
      "development_shared_reductions",
      f"development_shared_reductions_t{target_index}",
      "shared_reductions",
      (InputTensor("x", shape),),
      nodes,
      output_shape,
      "n6",
      pattern,
      values,
      (-8, 8),
    )
    for target_index, (pattern, values) in enumerate(targets)
  ]


def _make_problem(
  graph_id: str,
  problem_id: str,
  family: str,
  inputs: tuple[InputTensor, ...],
  nodes: tuple[GraphNode, ...],
  output_shape: tuple[int, ...],
  output_id: str,
  target_pattern: str,
  target: tuple[int, ...],
  domain: tuple[int, int],
) -> PilotProblem:
  problem = PilotProblem(
    graph_id=graph_id,
    problem_id=problem_id,
    family=family,
    symbol_count=sum(_element_count(item.shape) for item in inputs),
    inputs=inputs,
    output_shape=output_shape,
    domain=domain,
    target=target,
    target_pattern=target_pattern,
    nodes=nodes,
    output_id=output_id,
  )
  validate_problem(problem)
  return problem


def freeze_workload(seed: int) -> list[PilotProblem]:
  """Build the 60-graph, 120-target-pair evaluation workload."""
  seed = _check_integer(seed, "seed")
  graph_rng = random.Random(seed)
  target_rng = random.Random((seed ^ 0x9E3779B9) & 0xFFFFFFFF)
  problems: list[PilotProblem] = []
  graph_index = 0
  for symbol_count in (16, 32, 64):
    allocations = (
      ("independent_sum", 4),
      ("affine_sum", 4),
      ("layout_sum", 4),
      ("shared_reductions", 3),
      ("nonlinear", 5),
    )
    for family, count in allocations:
      for variant in range(count):
        graph_index += 1
        graph_id = f"g{graph_index:03d}"
        rng = random.Random(graph_rng.getrandbits(64))
        inputs, nodes, output_shape, output_id = _make_graph(
          graph_id, family, symbol_count, rng, variant
        )
        targets = _target_pair(output_shape, target_rng)
        for target_index, (pattern, values) in enumerate(targets):
          problems.append(
            _make_problem(
              graph_id,
              f"{graph_id}_t{target_index}",
              family,
              inputs,
              nodes,
              output_shape,
              output_id,
              pattern,
              values,
              (-8, 8),
            )
          )
  return problems


def freeze_development_workload(
  seed: int, *, evaluation_workload: list[PilotProblem]
) -> list[PilotProblem]:
  """Build six deterministic development graphs disjoint from evaluation."""
  seed = _check_integer(seed, "development seed")
  if not isinstance(evaluation_workload, list) or any(
    not isinstance(problem, PilotProblem) for problem in evaluation_workload
  ):
    raise ValueError("evaluation_workload must be a list of PilotProblem records")
  evaluation_signatures = {_graph_signature(problem) for problem in evaluation_workload}
  strata = (
    ("independent_sum", 16, 0),
    ("layout_sum", 16, 0),
    ("affine_sum", 32, 0),
    ("shared_reductions", 32, 0),
    ("nonlinear", 64, 0),
    ("layout_sum", 64, 1),
  )
  required = {(family, count): index + 1 for family, count, index in strata}
  candidates: dict[tuple[str, int], list[tuple[str, list[PilotProblem]]]] = {
    key: [] for key in required
  }
  seen_signatures = set(evaluation_signatures)
  shared_alternative = _make_development_shared_reductions(seed)
  shared_signature = _graph_signature(shared_alternative[0])
  if shared_signature in seen_signatures:
    raise ValueError("development shared-reduction alternative overlaps evaluation")
  candidates[("shared_reductions", 32)].append((shared_signature, shared_alternative))
  seen_signatures.add(shared_signature)
  for seed_offset in range(512):
    by_graph: dict[str, list[PilotProblem]] = {}
    for problem in freeze_workload(seed + seed_offset):
      by_graph.setdefault(problem.graph_id, []).append(problem)
    for rows in by_graph.values():
      first = rows[0]
      key = (first.family, first.symbol_count)
      if key not in candidates or key == ("shared_reductions", 32):
        continue
      signature = _graph_signature(first)
      if signature in seen_signatures:
        continue
      seen_signatures.add(signature)
      candidates[key].append((signature, rows))
    if all(len(candidates[key]) >= count for key, count in required.items()):
      break
  else:
    raise ValueError("could not freeze enough development graphs outside evaluation")

  chosen = []
  for family, symbol_count, index in strata:
    _signature, rows = candidates[(family, symbol_count)][index]
    chosen.append(rows)
  result = []
  for graph_index, rows in enumerate(chosen, start=1):
    graph_id = f"dev{graph_index:03d}"
    for target_index, problem in enumerate(rows):
      result.append(
        replace(
          problem,
          graph_id=graph_id,
          problem_id=f"{graph_id}_t{target_index}",
        )
      )
  return result


def sum_problem(
  shape: tuple[int, ...], target: tuple[int, ...], domain: tuple[int, int]
) -> PilotProblem:
  _check_shape(shape, "sum input")
  if not isinstance(target, tuple):
    raise ValueError("target must be a tuple")
  output_shape = shape[:-1]
  if len(target) != _element_count(output_shape):
    raise ValueError("target element count does not match the sum output shape")
  digest = hashlib.sha256(repr((shape, target, domain)).encode("ascii")).hexdigest()[
    :12
  ]
  return _make_problem(
    f"adhoc_sum_{digest}",
    f"adhoc_sum_{digest}",
    "independent_sum",
    (InputTensor("x", shape),),
    (GraphNode("n0", "sum", ("x",), output_shape, axis=-1),),
    output_shape,
    "n0",
    "custom",
    target,
    domain,
  )


def broadcast_problem(target: tuple[int, int], domain: tuple[int, int]) -> PilotProblem:
  if not isinstance(target, tuple) or len(target) != 2:
    raise ValueError("broadcast target must contain two output elements")
  digest = hashlib.sha256(repr((target, domain)).encode("ascii")).hexdigest()[:12]
  return _make_problem(
    f"adhoc_broadcast_{digest}",
    f"adhoc_broadcast_{digest}",
    "layout_sum",
    (InputTensor("x", (1,)),),
    (GraphNode("n0", "broadcast", ("x",), (2,)),),
    (2,),
    "n0",
    "custom",
    target,
    domain,
  )


def problem_to_dict(problem: PilotProblem) -> dict[str, object]:
  validate_problem(problem)
  return {
    "schema": SCHEMA,
    "graph_id": problem.graph_id,
    "problem_id": problem.problem_id,
    "family": problem.family,
    "symbol_count": problem.symbol_count,
    "inputs": [
      {"name": item.name, "shape": list(item.shape)} for item in problem.inputs
    ],
    "output_shape": list(problem.output_shape),
    "domain": list(problem.domain),
    "target": list(problem.target),
    "target_pattern": problem.target_pattern,
    "nodes": [
      {
        "node_id": node.node_id,
        "op": node.op,
        "operands": list(node.operands),
        "shape": list(node.shape),
        "axis": node.axis,
        "keep": node.keep,
        "permutation": list(node.permutation),
      }
      for node in problem.nodes
    ],
    "output_id": problem.output_id,
  }


def _require_keys(value: object, expected: set[str], what: str) -> dict[str, object]:
  if not isinstance(value, dict):
    raise ValueError(f"{what} must be a JSON object")
  actual = set(value)
  if actual != expected:
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    raise ValueError(f"{what} fields mismatch (unknown={unknown}, missing={missing})")
  return value


def problem_from_dict(value: object) -> PilotProblem:
  fields = {
    "schema",
    "graph_id",
    "problem_id",
    "family",
    "symbol_count",
    "inputs",
    "output_shape",
    "domain",
    "target",
    "target_pattern",
    "nodes",
    "output_id",
  }
  record = _require_keys(value, fields, "problem")
  if record["schema"] != SCHEMA:
    raise ValueError(f"unsupported problem schema {record['schema']!r}")
  inputs_value = record["inputs"]
  if not isinstance(inputs_value, list):
    raise ValueError("inputs must be a JSON array")
  inputs = []
  for item in inputs_value:
    input_record = _require_keys(item, {"name", "shape"}, "input")
    inputs.append(InputTensor(input_record["name"], tuple(input_record["shape"])))
  nodes_value = record["nodes"]
  if not isinstance(nodes_value, list):
    raise ValueError("nodes must be a JSON array")
  nodes = []
  node_fields = {"node_id", "op", "operands", "shape", "axis", "keep", "permutation"}
  for item in nodes_value:
    node_record = _require_keys(item, node_fields, "node")
    operands = tuple(node_record["operands"])
    if any(
      isinstance(operand, bool) or not isinstance(operand, (str, int))
      for operand in operands
    ):
      raise ValueError("node operands must be strings or integers")
    nodes.append(
      GraphNode(
        node_record["node_id"],
        node_record["op"],
        operands,
        tuple(node_record["shape"]),
        node_record["axis"],
        node_record["keep"],
        tuple(node_record["permutation"]),
      )
    )
  problem = PilotProblem(
    graph_id=record["graph_id"],
    problem_id=record["problem_id"],
    family=record["family"],
    symbol_count=record["symbol_count"],
    inputs=tuple(inputs),
    output_shape=tuple(record["output_shape"]),
    domain=tuple(record["domain"]),
    target=tuple(record["target"]),
    target_pattern=record["target_pattern"],
    nodes=tuple(nodes),
    output_id=record["output_id"],
  )
  validate_problem(problem)
  return problem


def save_workload(path: Path, problems: list[PilotProblem], seed: int) -> str:
  payload = {
    "schema": WORKLOAD_SCHEMA,
    "seed": seed,
    "problems": [problem_to_dict(problem) for problem in problems],
  }
  data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(data)
  return hashlib.sha256(data).hexdigest()


def load_workload(path: Path) -> tuple[int, list[PilotProblem]]:
  value = json.loads(Path(path).read_text(encoding="utf-8"))
  payload = _require_keys(value, {"schema", "seed", "problems"}, "workload")
  if payload["schema"] != WORKLOAD_SCHEMA:
    raise ValueError(f"unsupported workload schema {payload['schema']!r}")
  seed = _check_integer(payload["seed"], "seed")
  if not isinstance(payload["problems"], list):
    raise ValueError("workload problems must be a JSON array")
  return seed, [problem_from_dict(item) for item in payload["problems"]]


def _graph_signature(problem: PilotProblem) -> str:
  input_indices = {item.name: index for index, item in enumerate(problem.inputs)}
  node_indices = {node.node_id: index for index, node in enumerate(problem.nodes)}

  def normalize_operand(operand: Operand) -> dict[str, int]:
    if isinstance(operand, str):
      if operand in input_indices:
        return {"input": input_indices[operand]}
      if operand in node_indices:
        return {"node": node_indices[operand]}
    return {"literal": int(operand)}

  graph = {
    "domain": problem.domain,
    "inputs": [{"shape": item.shape} for item in problem.inputs],
    "output_shape": problem.output_shape,
    "nodes": [
      {
        "op": node.op,
        "operands": [normalize_operand(operand) for operand in node.operands],
        "shape": node.shape,
        "axis": node.axis,
        "keep": node.keep,
        "permutation": node.permutation,
      }
      for node in problem.nodes
    ],
    "output": node_indices[problem.output_id],
  }
  return hashlib.sha256(
    json.dumps(graph, sort_keys=True, default=list).encode()
  ).hexdigest()


def write_frozen_workloads(
  directory: Path, *, development_seed: int, evaluation_seed: int
) -> dict:
  directory = Path(directory)
  evaluation = freeze_workload(evaluation_seed)
  development = freeze_development_workload(
    development_seed, evaluation_workload=evaluation
  )
  development_hash = save_workload(
    directory / "development.json", development, development_seed
  )
  evaluation_hash = save_workload(
    directory / "workload.json", evaluation, evaluation_seed
  )
  graph_rows: dict[str, PilotProblem] = {}
  for problem in evaluation:
    graph_rows.setdefault(problem.graph_id, problem)
  signatures: dict[str, list[str]] = {}
  for graph_id, problem in graph_rows.items():
    signatures.setdefault(_graph_signature(problem), []).append(graph_id)
  collisions = [ids for ids in signatures.values() if len(ids) > 1]
  development_signatures = {_graph_signature(problem) for problem in development}
  evaluation_signatures = {_graph_signature(problem) for problem in evaluation}
  overlap = sorted(development_signatures & evaluation_signatures)
  if overlap:
    raise ValueError("development workload overlaps evaluation graph structures")
  family_counts = {
    family: sum(problem.family == family for problem in evaluation)
    for family in _FAMILIES
  }
  symbol_counts = {
    str(count): sum(problem.symbol_count == count for problem in evaluation)
    for count in (16, 32, 64)
  }
  pattern_counts: dict[str, int] = {}
  for problem in evaluation:
    pattern_counts[problem.target_pattern] = (
      pattern_counts.get(problem.target_pattern, 0) + 1
    )
  manifest = {
    "schema": WORKLOAD_SCHEMA,
    "development_seed": development_seed,
    "evaluation_seed": evaluation_seed,
    "development_sha256": development_hash,
    "evaluation_sha256": evaluation_hash,
    "development_problem_count": len(development),
    "development_graph_count": len(development_signatures),
    "development_evaluation_graph_overlap": overlap,
    "evaluation_problem_count": len(evaluation),
    "evaluation_graph_count": len(graph_rows),
    "family_problem_counts": family_counts,
    "symbol_count_problem_counts": symbol_counts,
    "target_pattern_counts": pattern_counts,
    "rejected_structurally_invalid_graphs": [],
    "normalization_collisions": collisions,
  }
  (directory / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  return manifest
