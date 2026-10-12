"""Shape, bound, and arithmetic access to the bounded loop regions.

`test_tensor_region_family_option` owns `--tensor-region-family`, which names
the connection skeleton of the tensor-v2 `loop-reduction` stratum's region. The
region's other three choices - its two-dimensional shape, its bounded trip
count, and the arithmetic its roots use - are a separate request surface with
their own admitted domain, so they are tested here: a request either reaches
the generated source and the case manifest, or it is refused with the reason it
is outside that domain instead of being adjusted to something admitted.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from test.lib import tensor_exact_f32, tensor_region_generation

ROOT = Path(__file__).resolve().parents[2]
RYSMITH = Path(os.environ.get("REFRACTIR_RYSMITH", ROOT / "rysmith")).resolve()
READY = RYSMITH.is_file()

STRATUM = "loop-reduction"
# The region policy an unnamed request keeps: the 2x2 two-state body the frozen
# fold sources use, two iterations, additive roots.
DEFAULT_SHAPE = (2, 2)
DEFAULT_ITERATIONS = 2
DEFAULT_ARITHMETIC = "additive"

# Root placements are positional: the first root takes the mode's own operation
# and the second the joining one. `mixed` multiplies first and subtracts second.
ROOT_OPERATIONS = {
  "additive": ("add", "add"),
  "subtraction": ("sub", "add"),
  "multiplication": ("mul", "add"),
  "mixed": ("mul", "sub"),
}
ARITHMETIC_OPERATIONS = {
  "additive": {"add"},
  "subtraction": {"add", "sub"},
  "multiplication": {"add", "mul"},
  "mixed": {"add", "sub", "mul"},
}


def generate(
  output_dir,
  *,
  seed=5,
  count=2,
  family=None,
  shape=None,
  iterations=None,
  arithmetic=None,
  stratum=STRATUM,
  timeout_s=300,
):
  arguments = [
    str(RYSMITH),
    "--profile",
    "tensor",
    "--target",
    "sir",
    "--seed",
    str(seed),
    "-n",
    str(count),
    "--tensor-dtype",
    "f32",
    "--tensor-profile-version",
    "v2",
    "--tensor-generator",
    "constructor",
    "--output-dir",
    str(output_dir),
  ]
  if stratum is not None:
    arguments += ["--tensor-stratum", stratum]
  if family is not None:
    arguments += ["--tensor-region-family", family]
  if shape is not None:
    arguments += ["--tensor-region-shape", f"{shape[0]},{shape[1]}"]
  if iterations is not None:
    arguments += ["--tensor-region-iterations", str(iterations)]
  if arithmetic is not None:
    arguments += ["--tensor-region-arithmetic", arithmetic]
  return subprocess.run(
    arguments, capture_output=True, text=True, check=False, timeout=timeout_s
  )


def case_sources(output_dir):
  """Program sources of one run, keyed by case directory name."""
  return {
    path.parent.name: path.read_text(encoding="utf-8")
    for path in sorted(Path(output_dir).glob("case_*/program.sir"))
  }


def case_manifests(output_dir):
  """Case manifests of one run, keyed by case directory name."""
  return {
    path.parent.name: json.loads(path.read_text(encoding="utf-8"))
    for path in sorted(Path(output_dir).glob("case_*/case.json"))
  }


def is_region(source):
  """A loop region updates both named states from a shared two-state body."""
  return "%acc_x = %next_x;" in source and "%acc_y = %next_y;" in source


def region_shape_and_iterations(source):
  shape, iterations, _types, _nodes, _updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  return shape, iterations


def region_record(manifest):
  return manifest["generator"].get("region")


class TensorRegionGeneratedCase(unittest.TestCase):
  """Common guard and accessors for one generated run."""

  def setUp(self):
    if not READY:
      self.skipTest("rysmith is supplied by the managed Make target")

  def sources_and_manifests(self, result, output_dir):
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    sources = case_sources(output_dir)
    manifests = case_manifests(output_dir)
    self.assertTrue(sources)
    self.assertEqual(set(sources), set(manifests))
    return sources, manifests


class TensorRegionShapeOptionTests(TensorRegionGeneratedCase):
  def test_requested_shape_reaches_the_generated_source(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      result = generate(root, count=2, shape=(4, 8))
      sources, manifests = self.sources_and_manifests(result, root)

    for name, source in sources.items():
      with self.subTest(case=name):
        self.assertTrue(
          is_region(source), "the request must force the loop-region form"
        )
        self.assertEqual(region_shape_and_iterations(source)[0], (4, 8))
        self.assertIn("tensor [4, 8] f32", source)

    for name, manifest in manifests.items():
      with self.subTest(case=name):
        self.assertEqual(
          region_record(manifest),
          {
            "shape": [4, 8],
            "iterations": DEFAULT_ITERATIONS,
            "family": "fold",
            "operation_mode": DEFAULT_ARITHMETIC,
          },
          "the case must record the shape, bound, family, and arithmetic it used",
        )


class TensorRegionBoundOptionTests(TensorRegionGeneratedCase):
  def test_requested_iterations_reach_the_generated_source(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      result = generate(root, count=2, iterations=3)
      sources, manifests = self.sources_and_manifests(result, root)

    for name, source in sources.items():
      with self.subTest(case=name):
        shape, iterations = region_shape_and_iterations(source)
        self.assertEqual(iterations, 3)
        self.assertEqual(shape, DEFAULT_SHAPE)
        self.assertIn("br %i < 3, ^body, ^exit;", source)

    for name, manifest in manifests.items():
      with self.subTest(case=name):
        self.assertEqual(manifest["execution"]["loop_bounds"], [3])
        self.assertEqual(manifest["execution"]["total_body_executions"], 3)
        self.assertEqual(
          region_record(manifest),
          {
            "shape": list(DEFAULT_SHAPE),
            "iterations": 3,
            "family": "fold",
            "operation_mode": DEFAULT_ARITHMETIC,
          },
        )


class TensorRegionArithmeticOptionTests(TensorRegionGeneratedCase):
  def test_requested_arithmetic_reaches_the_generated_source(self):
    for arithmetic, operations in ROOT_OPERATIONS.items():
      with self.subTest(arithmetic=arithmetic):
        with tempfile.TemporaryDirectory() as directory:
          root = Path(directory)
          result = generate(root, count=1, arithmetic=arithmetic)
          sources, manifests = self.sources_and_manifests(result, root)

        for source in sources.values():
          self.assertEqual(
            tuple(
              pattern[0]
              for pattern in tensor_region_generation.root_join_patterns(source)
            ),
            operations,
            "the mode must place its operation on the named roots",
          )
          self.assertEqual(
            set(tensor_region_generation.arithmetic_operations(source)),
            ARITHMETIC_OPERATIONS[arithmetic],
          )
          self.assertEqual(
            tensor_region_generation.producer_operation(source),
            "add",
            "the shared producer stays additive in every mode",
          )
        for manifest in manifests.values():
          self.assertEqual(region_record(manifest)["operation_mode"], arithmetic)


class TensorRegionOptionDefaultTests(TensorRegionGeneratedCase):
  def test_unnamed_request_keeps_the_region_defaults(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      unnamed = generate(root / "unnamed", seed=5, count=64)
      named = generate(
        root / "named",
        seed=5,
        count=64,
        family="fold",
        shape=DEFAULT_SHAPE,
        iterations=DEFAULT_ITERATIONS,
        arithmetic=DEFAULT_ARITHMETIC,
      )
      unnamed_sources, unnamed_manifests = self.sources_and_manifests(
        unnamed, root / "unnamed"
      )
      named_sources, _named_manifests = self.sources_and_manifests(
        named, root / "named"
      )

    regions = {
      name: source for name, source in unnamed_sources.items() if is_region(source)
    }
    self.assertTrue(regions, "the default route must still reach the fold region")
    self.assertGreater(
      len(unnamed_sources) - len(regions),
      0,
      "an unnamed request must leave the other loop forms scheduled",
    )
    for name, source in regions.items():
      with self.subTest(case=name):
        self.assertEqual(region_shape_and_iterations(source), (DEFAULT_SHAPE, 2))
        self.assertEqual(
          set(tensor_region_generation.arithmetic_operations(source)),
          ARITHMETIC_OPERATIONS[DEFAULT_ARITHMETIC],
        )
        self.assertIsNone(
          region_record(unnamed_manifests[name]),
          "an unnamed region keeps the manifest it had",
        )
        self.assertEqual(
          named_sources[name],
          source,
          "naming the defaults must reproduce the unnamed region byte for byte",
        )


class TensorRegionOptionRefusalTests(TensorRegionGeneratedCase):
  def assert_refused(self, arguments, reason):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      result = generate(root, count=1, **arguments)
      emitted = list(root.glob("case_*"))

    self.assertNotEqual(result.returncode, 0)
    combined = result.stdout + result.stderr
    self.assertIn(reason, combined, combined)
    self.assertEqual(emitted, [], "a refused request must not emit a case")

  def test_out_of_domain_request_is_refused_with_its_reason(self):
    cases = (
      ({"shape": (3, 3)}, "powers of two"),
      ({"shape": (64, 64)}, "512-element"),
      ({"shape": (512, 1)}, "at least 2"),
      ({"iterations": 9}, "must be in [1,8]"),
      ({"iterations": 0}, "must be in [1,8]"),
      ({"arithmetic": "xor"}, "unknown tensor region arithmetic"),
    )
    for arguments, reason in cases:
      with self.subTest(arguments=arguments):
        self.assert_refused(arguments, reason)


class TensorRegionOptionEdgeTests(TensorRegionGeneratedCase):
  def test_boundary_shape_at_the_largest_admitted_trip_count(self):
    shape = (32, 16)
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      result = generate(root, seed=7, count=1, shape=shape, iterations=8)
      sources, manifests = self.sources_and_manifests(result, root)
      # The returned output is an artifact of the run, so it is measured while
      # the case directory still exists.
      output_bytes = {
        name: (root / name / "expected.bin").stat().st_size for name in manifests
      }

    expected_elements = shape[0] * shape[1]
    self.assertEqual(expected_elements, 512, "the boundary shape fills the profile cap")
    for source in sources.values():
      self.assertEqual(region_shape_and_iterations(source), (shape, 8))
    for name, manifest in manifests.items():
      with self.subTest(case=name):
        self.assertEqual(manifest["execution"]["loop_bounds"], [8])
        self.assertEqual(manifest["execution"]["total_body_executions"], 8)
        self.assertEqual(
          manifest["output"]["shape"],
          [shape[0], shape[1]],
        )
        self.assertEqual(output_bytes[name], expected_elements * 4)
        self.assertEqual(
          region_record(manifest),
          {
            "shape": list(shape),
            "iterations": 8,
            "family": "fold",
            "operation_mode": DEFAULT_ARITHMETIC,
          },
        )


if __name__ == "__main__":
  unittest.main()
