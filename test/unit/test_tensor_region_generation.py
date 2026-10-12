"""Explicit configuration tests for bounded multi-result tensor regions."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from test.lib import tensor_exact_f32, tensor_region_generation
from test.lib.tensor_region_generation import RegionConfiguration, RegionGenerationError

ROOT = Path(__file__).resolve().parents[2]


def _tool(environment_name, binary_name):
  value = os.environ.get(environment_name)
  return Path(value).resolve() if value else ROOT / "build" / "bin" / binary_name


SYMIRI = _tool("REFRACTIR_SYMIRI", "symiri")
SYMIRC = _tool("REFRACTIR_SYMIRC", "symirc")
SYMIR_SOLVE = _tool("REFRACTIR_SYMIR_SOLVE", "symirsolve")
EXPORTER = tensor_region_generation.EXPORTER
EXPORTER_AVAILABLE = EXPORTER.is_file()
TOOLS_AVAILABLE = all(path.is_file() for path in (SYMIRI, SYMIRC, SYMIR_SOLVE))


def _replay(source, input_bytes):
  """Run one region through the reference interpreter and return its output."""
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source_path = root / "program.sir"
    input_path = root / "inputs.bin"
    output_path = root / "output.bin"
    source_path.write_text(source, encoding="utf-8")
    input_path.write_bytes(input_bytes)
    result = subprocess.run(
      [
        str(SYMIRI),
        "--tensor-profile-version",
        "v2",
        "--entry",
        "kernel",
        "--tensor-input",
        str(input_path),
        "--tensor-output",
        str(output_path),
        str(source_path),
      ],
      capture_output=True,
      text=True,
      check=False,
    )
    if result.returncode != 0:
      raise AssertionError(f"reference replay failed: {result.stderr.strip()}")
    return output_path.read_bytes()


class TensorRegionGenerationTests(unittest.TestCase):
  def setUp(self):
    if not EXPORTER_AVAILABLE:
      self.skipTest("the native graph exporter is supplied by the managed Make target")

  def test_long_axis_regions_are_reachable_for_every_family(self):
    shapes = ((256, 2), (2, 256), (128, 4))
    for family in tensor_region_generation.FAMILIES:
      for shape in shapes:
        with self.subTest(family=family, shape=shape):
          configuration = RegionConfiguration(
            seed=11, shape=shape, iterations=2, family=family
          )
          source = tensor_region_generation.generate_source(configuration)
          parsed = tensor_exact_f32._generated_region_program(source)
          self.assertEqual(parsed[0], shape)
          self.assertEqual(parsed[1], 2)

  def test_shared_producer_feeds_two_independent_reduction_consumers(self):
    configuration = RegionConfiguration(
      seed=5, shape=(16, 32), iterations=3, family="orthogonal-reductions"
    )
    source = tensor_region_generation.generate_source(configuration)
    nodes = tensor_region_generation.operation_nodes(source)
    reductions = [node for node in nodes if node[1] == "sum"]

    self.assertEqual(len(reductions), 2)
    self.assertEqual({node[3] for node in reductions}, {0, 1})
    self.assertEqual(
      reductions[0][2], reductions[1][2], "both reductions must share one producer"
    )
    dependencies = tensor_region_generation.root_dependencies(source)
    live = dependencies["acc_x"] | dependencies["acc_y"]
    for reduction in reductions:
      self.assertIn(reduction[0], live)
    self.assertNotIn(reductions[0][0], dependencies["acc_x"])
    self.assertNotIn(reductions[1][0], dependencies["acc_y"])

  def test_cross_state_roots_depend_on_both_current_states(self):
    if not SYMIRI.is_file():
      self.skipTest("the reference interpreter is supplied by the managed Make target")
    configuration = RegionConfiguration(
      seed=3, shape=(4, 8), iterations=2, family="cross-state"
    )
    source = tensor_region_generation.generate_source(configuration)
    values = [1 + (index % 3) for index in range(32)]
    input_bytes = struct.pack("<32f", *values)

    replayed = _replay(source, input_bytes)
    _proof, expected = tensor_exact_f32.certify(source, input_bytes, (0, 8))

    self.assertEqual(replayed, expected)
    nodes = tensor_region_generation.operation_nodes(source)
    # The producer's operand order is one of the seeded choices, so the
    # contract is which states it reads, not the order it reads them in.
    shared = [node for node in nodes if set(node[2]) == {"acc_x", "acc_y"}]
    self.assertEqual(len(shared), 1, "one producer must read both current states")

  def test_configuration_reproduces_independently_of_launch_settings(self):
    configuration = RegionConfiguration(
      seed=13, shape=(32, 16), iterations=4, family="mixed-depth"
    )
    first = tensor_region_generation.generate_source(configuration)
    plan = tensor_region_generation.launch_plan(1001, count=6)

    self.assertEqual(
      {entry["layout"] for entry in plan},
      set(tensor_region_generation.PHYSICAL_LAYOUTS),
    )
    self.assertGreater(len({entry["num_warps"] for entry in plan}), 1)
    self.assertEqual(plan, tensor_region_generation.launch_plan(1001, count=6))
    for _entry in plan:
      self.assertEqual(tensor_region_generation.generate_source(configuration), first)

    # A launch setting is not part of the logical program: the same seed with a
    # different shape or bound is a different program with a different identity.
    self.assertNotEqual(
      configuration.identity, configuration.with_shape((16, 32)).identity
    )
    self.assertNotEqual(configuration.identity, configuration.with_bounds(2).identity)
    self.assertEqual(tensor_region_generation.region_matrix((7,))[0].seed, 7)

  def test_case_export_records_the_region_configuration(self):
    if not TOOLS_AVAILABLE:
      self.skipTest("the tensor toolchain is supplied by the managed Make target")
    from test.lib import build_tensor_exact_region_case

    configuration = RegionConfiguration(
      seed=21, shape=(4, 8), iterations=2, family="reduction-arithmetic"
    )
    source = tensor_region_generation.generate_source(configuration)
    # The property has to be satisfiable inside the domain, so take the target
    # from the admitted reference at a concrete member of that domain.
    probe = struct.pack("<32f", *([1.0] * 32))
    _proof, reference = tensor_exact_f32.certify(source, probe, (0, 2))
    target = int(struct.unpack("<32f", reference)[0])
    with tempfile.TemporaryDirectory() as directory:
      case = build_tensor_exact_region_case.build_case(
        Path(directory) / "case",
        source=source,
        target=target,
        domain=(0, 2),
        symiri=SYMIRI,
        symirc=SYMIRC,
        solver=SYMIR_SOLVE,
        configuration=configuration,
      )
      manifest = json.loads((case / "case.json").read_text(encoding="utf-8"))
      outputs = (case / "expected.bin").read_bytes()

    self.assertEqual(manifest["generator"]["region"], configuration.as_record())
    self.assertEqual(manifest["source"]["seed"], configuration.seed)
    self.assertEqual(len(outputs), 32 * 4)
    self.assertEqual(manifest["execution"]["loop_bounds"], [2])


class TensorRegionFamilyDiversityTests(unittest.TestCase):
  """A fixed shape, bound, mode, and family still leaves the body to the seed."""

  def setUp(self):
    if not EXPORTER_AVAILABLE:
      self.skipTest("the native graph exporter is supplied by the managed Make target")

  def sources(self, family, *, mode="additive", seeds=range(1, 9), shape=(16, 32)):
    return {
      seed: tensor_region_generation.generate_source(
        RegionConfiguration(
          seed=seed, shape=shape, iterations=2, family=family, operation_mode=mode
        )
      )
      for seed in seeds
    }

  def test_each_family_draws_a_different_body_per_seed(self):
    for family in tensor_region_generation.NEW_FAMILIES:
      with self.subTest(family=family):
        bodies = set(self.sources(family).values())

        self.assertGreaterEqual(
          len(bodies),
          3,
          f"{family} emitted {len(bodies)} distinct bodies across eight seeds",
        )

  def test_shared_producer_is_never_multiplied_against_the_zero_state(self):
    """Every mode leaves the body a real computation, not a constant.

    The shared producer reads both loop states and the second one is
    initialized to zero, so a product there is a constant zero on every trip:
    the producer must use another operation, and the requested mode stays
    visible on the root placements instead.
    """
    for family in tensor_region_generation.NEW_FAMILIES:
      for mode in ("additive", "subtraction", "multiplication", "mixed"):
        for seed in range(1, 5):
          with self.subTest(family=family, mode=mode, seed=seed):
            source = self.sources(family, mode=mode, seeds=(seed,))[seed]

            self.assertEqual(
              set(tensor_region_generation.producer_operands(source)),
              {"acc_x", "acc_y"},
              "the shared producer is the operation that reads both states",
            )
            self.assertNotEqual(
              tensor_region_generation.producer_operation(source),
              "mul",
              f"{family} in {mode} mode multiplies the zero-initialized state",
            )

  def test_every_mode_is_visible_in_the_emitted_arithmetic(self):
    """Each mode selects arithmetic the emitted body actually performs."""
    expected = {
      "additive": {"add"},
      "subtraction": {"add", "sub"},
      "multiplication": {"add", "mul"},
      "mixed": {"add", "mul", "sub"},
    }
    for family in tensor_region_generation.NEW_FAMILIES:
      for mode, operations in expected.items():
        with self.subTest(family=family, mode=mode):
          observed = {
            frozenset(tensor_region_generation.arithmetic_operations(source))
            for source in self.sources(family, mode=mode, seeds=range(1, 5)).values()
          }

          self.assertEqual(observed, {frozenset(operations)})

  def test_mixed_arithmetic_differs_from_additive_for_every_family(self):
    for family in tensor_region_generation.NEW_FAMILIES:
      additive = self.sources(family, mode="additive")
      mixed = self.sources(family, mode="mixed")
      for seed, source in additive.items():
        with self.subTest(family=family, seed=seed):
          self.assertNotEqual(source, mixed[seed])
          # The mode is carried by the root placements: additive adds at every
          # root, mixed multiplies at one placement and subtracts at another.
          self.assertEqual(
            tensor_region_generation.arithmetic_operations(source), {"add"}
          )
          self.assertEqual(
            tensor_region_generation.arithmetic_operations(mixed[seed]),
            {"add", "mul", "sub"},
          )

  def test_reduction_axis_follows_the_seed(self):
    for family in (
      "orthogonal-reductions",
      "reduction-arithmetic",
      "mixed-depth",
    ):
      with self.subTest(family=family):
        axes = {
          seed: tensor_region_generation.reduction_axes(source)
          for seed, source in self.sources(family).items()
        }

        self.assertGreater(
          len(set(axes.values())),
          1,
          f"{family} keeps one reduction schedule for every seed: {axes}",
        )

  def test_producer_order_and_root_joins_follow_the_seed(self):
    for family in tensor_region_generation.NEW_FAMILIES:
      with self.subTest(family=family):
        patterns = {
          seed: (
            tensor_region_generation.producer_operands(source),
            tensor_region_generation.root_join_patterns(source),
          )
          for seed, source in self.sources(family).items()
        }

        for operands, _joins in patterns.values():
          self.assertEqual(set(operands), {"acc_x", "acc_y"})
        self.assertGreater(
          len(set(patterns.values())),
          1,
          f"{family} joins the same states in the same order for every seed",
        )


class TensorRegionModeSensitivityTests(unittest.TestCase):
  """No mode may leave a family with an output that ignores its input.

  Every check runs the emitted region over one explicitly stated input, the
  2x2 tensor `[1, 2, 4, 8]` (f32, complete output read through the reference
  interpreter): its lanes, rows, and columns all differ, so a body that
  collapses to a constant, or to zero, cannot pass. A second input differs
  only in its last lane, so a body that ignores the input cannot tell them
  apart.
  """

  SHAPE = (2, 2)
  ITERATIONS = 2
  INPUT = struct.pack("<4f", 1.0, 2.0, 4.0, 8.0)
  OTHER_INPUT = struct.pack("<4f", 1.0, 2.0, 4.0, 9.0)
  DOMAIN = (0, 8)
  MODES = ("additive", "subtraction", "multiplication", "mixed")
  SEEDS = (1, 2)

  def setUp(self):
    if not EXPORTER_AVAILABLE:
      self.skipTest("the native graph exporter is supplied by the managed Make target")
    if not SYMIRI.is_file():
      self.skipTest("the reference interpreter is supplied by the managed Make target")

  def source_for(self, family, mode, seed, *, iterations=None):
    return tensor_region_generation.generate_source(
      RegionConfiguration(
        seed=seed,
        shape=self.SHAPE,
        iterations=self.ITERATIONS if iterations is None else iterations,
        family=family,
        operation_mode=mode,
      )
    )

  def output_for(self, source, input_bytes):
    return struct.unpack("<4f", _replay(source, input_bytes))

  def assert_not_identically_constant(self, family, mode, seed):
    output = self.output_for(self.source_for(family, mode, seed), self.INPUT)

    self.assertGreater(
      len(set(output)),
      1,
      f"{family} in {mode} mode emitted the constant output {output}",
    )
    self.assertTrue(
      any(output),
      f"{family} in {mode} mode emitted the all-zero output {output}",
    )

  def test_multiplication_mode_is_not_identically_constant(self):
    for family in tensor_region_generation.NEW_FAMILIES:
      for seed in self.SEEDS:
        with self.subTest(family=family, seed=seed):
          self.assert_not_identically_constant(family, "multiplication", seed)

  def test_mixed_mode_is_not_identically_constant(self):
    for family in tensor_region_generation.NEW_FAMILIES:
      for seed in self.SEEDS:
        with self.subTest(family=family, seed=seed):
          self.assert_not_identically_constant(family, "mixed", seed)

  def test_every_family_and_mode_output_follows_the_trip_count(self):
    """A body that ignores the trip count is not a loop body.

    The output is read after one, two, and three trips: a region whose result
    is the same at every count computes nothing across the backedge, which no
    amount of input sensitivity can establish. A body that alternates is
    sensitive, so the check compares the whole set of outputs rather than
    consecutive pairs.

    One combination is trip-invariant by construction and stays admitted:
    `cross-state` with subtraction writes each state to the other's value,
    because the producer adds the two states and subtracting one from that sum
    leaves the other. The exact certificate's subtraction obligations are
    stated over exactly that configuration, so it is named here rather than
    refused, and a region with that shape studies the compiler's loop machinery
    without carrying a loop-dependent value.
    """
    trip_invariant = {("cross-state", "subtraction")}
    insensitive = set()
    for family in tensor_region_generation.NEW_FAMILIES:
      for mode in self.MODES:
        for seed in self.SEEDS:
          outputs = {
            count: self.output_for(
              self.source_for(family, mode, seed, iterations=count), self.INPUT
            )
            for count in (1, 2, 3)
          }
          if len(set(outputs.values())) == 1:
            insensitive.add((family, mode))
            continue
          with self.subTest(family=family, mode=mode, seed=seed):
            self.assertGreater(len(set(outputs.values())), 1)

    self.assertEqual(
      insensitive,
      trip_invariant,
      "the set of trip-invariant family and mode combinations changed",
    )

  def test_every_family_and_mode_output_follows_the_input(self):
    for family in tensor_region_generation.NEW_FAMILIES:
      for mode in self.MODES:
        for seed in self.SEEDS:
          with self.subTest(family=family, mode=mode, seed=seed):
            source = self.source_for(family, mode, seed)

            self.assertNotEqual(
              self.output_for(source, self.INPUT),
              self.output_for(source, self.OTHER_INPUT),
              f"{family} in {mode} mode returns the same output for two inputs",
            )

  def test_multiplication_keeps_the_second_state_live(self):
    """A state whose update cannot reach the output is not loop-carried.

    The region is replayed once as emitted and once with the second state's
    write-back replaced by a self-copy. Multiplication mode is the smallest
    case that distinguishes a live update from a dead one: its product is the
    one operation the zero-initialized state annihilates.
    """
    for family in tensor_region_generation.NEW_FAMILIES:
      for seed in self.SEEDS:
        with self.subTest(family=family, seed=seed):
          source = self.source_for(family, "multiplication", seed)
          muted = source.replace("  %acc_y = %next_y;", "  %acc_y = %acc_y;")

          self.assertNotEqual(muted, source, "the region must update both loop states")
          self.assertNotEqual(
            self.output_for(muted, self.INPUT),
            self.output_for(source, self.INPUT),
            f"{family} drops the second state's contribution to the output",
          )

  def test_every_family_receives_an_exact_subtraction_certificate(self):
    """The mode's subtraction must stay inside the exact certificate."""
    for family in tensor_region_generation.NEW_FAMILIES:
      for seed in self.SEEDS:
        with self.subTest(family=family, seed=seed):
          source = self.source_for(family, "subtraction", seed)

          proof, output = tensor_exact_f32.certify(source, self.INPUT, self.DOMAIN)

          self.assertIn("sub", proof["allowed_operations"])
          self.assertEqual(_replay(source, self.INPUT), output)


class TensorRegionGenerationEdgeTests(unittest.TestCase):
  """Degenerate axes and budgets are refused instead of silently weakened."""

  def setUp(self):
    if not EXPORTER_AVAILABLE:
      self.skipTest("the native graph exporter is supplied by the managed Make target")

  def test_every_seed_keeps_both_states_live_within_the_budget(self):
    for family in tensor_region_generation.NEW_FAMILIES:
      for seed in range(1, 13):
        with self.subTest(family=family, seed=seed):
          configuration = RegionConfiguration(
            seed=seed, shape=(4, 8), iterations=2, family=family
          )
          source = tensor_region_generation.generate_source(configuration)
          dependencies = tensor_region_generation.root_dependencies(source)
          live = dependencies["acc_x"] | dependencies["acc_y"]

          self.assertIn("acc_x", live)
          self.assertIn("acc_y", live)
          self.assertLessEqual(
            len(tensor_region_generation.operation_nodes(source)),
            configuration.max_operators,
          )

  def test_singleton_axis_refusal_does_not_depend_on_the_seed(self):
    for seed in range(1, 13):
      configuration = RegionConfiguration(
        seed=seed, shape=(1, 64), iterations=2, family="orthogonal-reductions"
      )
      with self.subTest(seed=seed):
        with self.assertRaisesRegex(RegionGenerationError, "graph_exhaustion"):
          tensor_region_generation.generate_source(configuration)

  def test_singleton_axis_is_rejected_for_independent_reductions(self):
    configuration = RegionConfiguration(
      seed=1, shape=(1, 64), iterations=2, family="orthogonal-reductions"
    )

    with self.assertRaisesRegex(RegionGenerationError, "graph_exhaustion"):
      tensor_region_generation.generate_source(configuration)

  def test_minimal_operator_budget_hosts_the_smallest_family(self):
    minimal = RegionConfiguration(
      seed=1, shape=(4, 8), iterations=2, family="cross-state", max_operators=3
    )
    source = tensor_region_generation.generate_source(minimal)
    self.assertEqual(len(tensor_region_generation.operation_nodes(source)), 3)

    starved = RegionConfiguration(
      seed=1, shape=(4, 8), iterations=2, family="cross-state", max_operators=2
    )
    with self.assertRaisesRegex(RegionGenerationError, "graph_exhaustion"):
      tensor_region_generation.generate_source(starved)


if __name__ == "__main__":
  unittest.main()
