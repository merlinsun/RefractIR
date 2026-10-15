"""Explicit generation options for bounded multi-result tensor loop regions.

The native exporter owns graph construction. This module states the
configuration a caller wants, asks the exporter for exactly that region, and
checks the emitted source against the request, so a case carries the shape,
bounds, and graph family it was generated from instead of a seed alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from test.lib import tensor_exact_f32

ROOT = Path(__file__).resolve().parents[2]
EXPORTER = Path(
  os.environ.get(
    "REFRACTIR_TENSOR_GRAPH_EXPORT",
    ROOT / "build" / "bin" / "tensor_graph_region_export",
  )
).resolve()

# Track A works on long axes at the fixed 512-element volume cap of the tensor
# profile, so a 256-element axis and its 16-element counterpart are both inside
# the same budget.
TARGET_SHAPES = ((256, 2), (2, 256), (128, 4), (4, 128), (16, 32), (32, 16))
BOUND_SCHEDULE = (2, 3, 4)
FAMILIES = (
  "fold",
  "orthogonal-reductions",
  "reduction-arithmetic",
  "cross-state",
  "mixed-depth",
  "broadcast-hoist",
)
# The families that draw their connection skeleton from the seeded engine; the
# original fold family keeps its frozen policy.
NEW_FAMILIES = FAMILIES[1:]
PHYSICAL_LAYOUTS = ("row-major", "column-major", "padded")
WARP_COUNTS = (1, 2, 4, 8)
ELEMENT_CAP = 512


class RegionGenerationError(RuntimeError):
  """A region request the exporter refuses or a source that breaks it."""


@dataclass(frozen=True)
class RegionConfiguration:
  """One frozen logical program request for the native region exporter."""

  seed: int
  shape: tuple[int, int]
  iterations: int
  family: str = "fold"
  operation_mode: str = "additive"
  max_operators: int = 10

  def __post_init__(self):
    if type(self.seed) is not int or self.seed < 0:
      raise ValueError("region seed must be a nonnegative integer")
    if (
      not isinstance(self.shape, tuple)
      or len(self.shape) != 2
      or any(type(extent) is not int or extent < 1 for extent in self.shape)
      or self.shape[0] * self.shape[1] > ELEMENT_CAP
    ):
      raise ValueError(f"region shape must hold at most {ELEMENT_CAP} elements")
    if type(self.iterations) is not int or not 1 <= self.iterations <= 8:
      raise ValueError("region bounds must be in [1,8]")
    if self.family not in FAMILIES:
      raise ValueError(f"region family must be one of {', '.join(FAMILIES)}")
    if type(self.max_operators) is not int or not 1 <= self.max_operators <= 64:
      raise ValueError("region operator budget must be in [1,64]")

  @property
  def identity(self) -> str:
    """Digest of every choice that changes the generated logical program."""
    return hashlib.sha256(
      json.dumps(self.as_record(), sort_keys=True).encode("utf-8")
    ).hexdigest()

  def as_record(self) -> dict:
    return {
      "seed": self.seed,
      "shape": list(self.shape),
      "iterations": self.iterations,
      "family": self.family,
      "operation_mode": self.operation_mode,
      "max_operators": self.max_operators,
    }

  def with_shape(self, shape):
    return replace(self, shape=tuple(shape))

  def with_bounds(self, iterations):
    return replace(self, iterations=iterations)

  def to_arguments(self) -> list[str]:
    return [
      "--seed",
      str(self.seed),
      "--shape",
      f"{self.shape[0]},{self.shape[1]}",
      "--iterations",
      str(self.iterations),
      "--operation-mode",
      self.operation_mode,
      "--family",
      self.family,
      "--max-operators",
      str(self.max_operators),
    ]


def generate_source(configuration, *, exporter=None, timeout_s=60.0):
  """Run the native exporter for one explicit configuration."""
  if not isinstance(configuration, RegionConfiguration):
    raise ValueError("generate_source needs a RegionConfiguration")
  binary = Path(exporter) if exporter is not None else EXPORTER
  try:
    result = subprocess.run(
      [str(binary), *configuration.to_arguments()],
      capture_output=True,
      text=True,
      timeout=timeout_s,
      check=False,
    )
  except OSError as error:
    raise RegionGenerationError(f"region exporter could not run: {error}") from error
  except subprocess.TimeoutExpired as error:
    raise RegionGenerationError("region exporter timed out") from error
  if result.returncode != 0:
    raise RegionGenerationError(
      result.stderr.strip() or f"region exporter exited with {result.returncode}"
    )
  source = result.stdout
  validate_source(source, configuration)
  return source


def validate_source(source, configuration):
  """Check that an emitted source is the requested logical program."""
  shape, iterations, _types, nodes, updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  if shape != configuration.shape:
    raise RegionGenerationError(
      f"emitted region shape {shape} does not match the requested {configuration.shape}"
    )
  if iterations != configuration.iterations:
    raise RegionGenerationError(
      f"emitted bounds {iterations} do not match the requested {configuration.iterations}"
    )
  if [state for state, _root in updates] != ["acc_x", "acc_y"]:
    raise RegionGenerationError("emitted region must update both loop states")
  if any(not root for _state, root in updates):
    raise RegionGenerationError("emitted region results must name the two roots")
  return nodes


def region_matrix(
  seeds, *, shapes=TARGET_SHAPES, bounds=BOUND_SCHEDULE, families=FAMILIES
):
  """One configuration per shape, bound, family, and seed.

  Shapes, bounds, families, and seeds are chosen independently, so a change in
  one axis cannot silently follow from another.
  """
  if isinstance(seeds, int):
    seeds = (seeds,)
  configurations = []
  for seed in seeds:
    for index, family in enumerate(families):
      for offset, shape in enumerate(shapes):
        bounds_value = bounds[(index + offset) % len(bounds)]
        configurations.append(
          RegionConfiguration(
            seed=seed, shape=tuple(shape), iterations=bounds_value, family=family
          )
        )
  return configurations


def launch_plan(
  launch_seed, *, count=None, layouts=PHYSICAL_LAYOUTS, warp_counts=WARP_COUNTS
):
  """Physical layouts and launch settings for one launch seed.

  The launch seed is deliberately separate from the source seed: a launch
  configuration is a variant of the same logical program, not part of it.
  """
  random_ = random.Random(f"launch:{launch_seed}")
  entries = []
  for index in range(count if count is not None else len(layouts)):
    entries.append(
      {
        "layout": layouts[index % len(layouts)],
        "num_warps": warp_counts[(index // len(layouts)) % len(warp_counts)],
        "num_stages": 1,
        "grid": [1, 1, 1],
      }
    )
  random_.shuffle(entries)
  return entries


def root_dependencies(source):
  """Ancestor sets of the two loop-carried roots, keyed by root name."""
  _shape, _iterations, _types, nodes, updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  producers = {node[0]: node for node in nodes}

  def ancestors(value):
    pending = [value]
    visited = set()
    while pending:
      current = pending.pop()
      if current in visited:
        continue
      visited.add(current)
      node = producers.get(current)
      if node is not None:
        pending.extend(node[2])
    return visited

  return {state: ancestors(root) for state, root in updates}


def operation_nodes(source):
  _shape, _iterations, _types, nodes, _updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  return nodes


def arithmetic_operations(source):
  """Element-wise arithmetic the body uses: `add`, `sub`, and `mul` only.

  Reductions and broadcasts are structural, so they are not part of the
  arithmetic a `TensorRegionArithmeticMode` selects.
  """
  return {
    node[1] for node in operation_nodes(source) if node[1] in {"add", "sub", "mul"}
  }


def shared_producer(source):
  """The one operation node that reads both loop states, in source order."""
  states = {"acc_x", "acc_y"}
  for node in operation_nodes(source):
    if set(node[2]) == states:
      return node
  raise RegionGenerationError("the region has no shared producer over both states")


def producer_operation(source):
  """Operation of the shared producer: the family's arithmetic seed choice."""
  return shared_producer(source)[1]


def producer_operands(source):
  """Operands of the shared producer, in emitted order."""
  return shared_producer(source)[2]


def reduction_axes(source):
  """Reduction axes in emission order, one entry per keep-dimension sum."""
  return tuple(node[3] for node in operation_nodes(source) if node[1] == "sum")


def root_join_patterns(source):
  """(operation, operands) of each named root's producer, in root order."""
  _shape, _iterations, _types, nodes, updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  producers = {node[0]: node for node in nodes}
  return tuple((producers[root][1], producers[root][2]) for _state, root in updates)


def consumer_values(source):
  """Consumers of every defined value, keyed by the value they read."""
  _shape, _iterations, _types, nodes, _updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  consumers = {node[0]: [] for node in nodes}
  for node in nodes:
    for operand in node[2]:
      consumers.setdefault(operand, []).append(node[0])
  return consumers


def broadcast_hoist_view(source):
  """The state-derived broadcast view the `broadcast-hoist` family places.

  The record states what the layout pipeline is handed: the loop-carried value
  the view is derived from and the axis it is reduced on, the one-extent
  reduction, the broadcast that re-expands it, every consumer of that broadcast,
  the second reduction (on the other axis, which is what asks for the other
  slice layout), the broadcast that carries it back, the elementwise consumer
  that keeps the view's own path, and the named roots the view reaches. The
  layout conversion the view provokes is inserted by the Triton pipeline rather
  than by the region, so the record stops at the source operations.

  A source without that view is refused: the family's whole point is the view,
  so a body that lost it is not this family's body.
  """
  _shape, _iterations, _types, nodes, updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  producers = {node[0]: node for node in nodes}
  consumers = consumer_values(source)
  dependencies = root_dependencies(source)
  for node in nodes:
    if node[1] != "broadcast":
      continue
    reduced = producers.get(node[2][0])
    if (
      reduced is None or reduced[1] != "sum" or reduced[2][0] not in {"acc_x", "acc_y"}
    ):
      continue
    second_reductions = [
      consumer
      for consumer in consumers.get(node[0], [])
      if producers.get(consumer, (None, None))[1] == "sum"
    ]
    if len(second_reductions) != 1:
      continue
    second_reduction = producers[second_reductions[0]]
    second_views = [
      consumer
      for consumer in consumers.get(second_reduction[0], [])
      if producers.get(consumer, (None, None))[1] == "broadcast"
    ]
    if len(second_views) != 1:
      continue
    shallow = [
      consumer
      for consumer in consumers.get(node[0], [])
      if consumer != second_reduction[0]
    ]
    if len(shallow) != 1:
      continue
    reached = [root for state, root in updates if node[0] in dependencies[state]]
    return {
      "state": reduced[2][0],
      "state_axis": reduced[3],
      "reduced": reduced[0],
      "view": node[0],
      "consumers": list(consumers[node[0]]),
      "second_reduction": second_reduction[0],
      "second_axis": second_reduction[3],
      "second_view": second_views[0],
      "shallow_consumer": shallow[0],
      "roots": reached,
    }
  raise RegionGenerationError("the region has no state-derived broadcast view")


def unroll_source(source, iterations=None):
  """Expand one bounded region into its mechanically unrolled equivalent.

  Each copy keeps the original body text, so the unrolled form is the same
  sequence of operations the counted loop performs, with a single trip and no
  backedge. Both forms must produce the same complete output.
  """
  shape, bounds, _types, _nodes, _updates = tensor_exact_f32._generated_region_program(
    source
  )
  trips = bounds if iterations is None else iterations
  if type(trips) is not int or trips != bounds:
    raise ValueError("unrolling must keep the counted number of trips")
  lines = source.splitlines()
  body_start = lines.index("^body:")
  exit_start = lines.index("^exit:")
  body = [line.strip() for line in lines[body_start + 1 : exit_start] if line.strip()]
  increment = [line for line in body if line == "%i = %i + 1;"]
  backedge = [line for line in body if line == "br ^header;"]
  content = [line for line in body if line not in {"%i = %i + 1;", "br ^header;"}]
  if increment != ["%i = %i + 1;"] or backedge != ["br ^header;"]:
    raise ValueError(
      "generated region body does not have one loop increment and backedge"
    )
  head = lines[: body_start + 1]
  head = [line.replace(f"br %i < {bounds},", "br %i < 1,") for line in head]
  tail = lines[exit_start:]
  unrolled = [
    *head,
    *[f"  {line}" for _trip in range(trips) for line in content],
    "  %i = %i + 1;",
    "  br ^header;",
    *tail,
  ]
  return "\n".join(unrolled) + "\n"
