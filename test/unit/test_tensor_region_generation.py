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
    shared = [node for node in nodes if node[2] == ("acc_x", "acc_y")]
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


class TensorRegionGenerationEdgeTests(unittest.TestCase):
  """Degenerate axes and budgets are refused instead of silently weakened."""

  def setUp(self):
    if not EXPORTER_AVAILABLE:
      self.skipTest("the native graph exporter is supplied by the managed Make target")

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
