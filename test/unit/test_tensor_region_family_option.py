"""Normal-generator access to the bounded loop-region families.

`rysmith --profile tensor` reaches the loop-region stratum through its
constructor route. `--tensor-region-family` names the connection skeleton that
stratum passes to `generateTensorRegion`, so a caller can ask for the new
families instead of only the frozen fold body.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from test.lib import tensor_region_generation

ROOT = Path(__file__).resolve().parents[2]
RYSMITH = Path(os.environ.get("REFRACTIR_RYSMITH", ROOT / "rysmith")).resolve()
READY = RYSMITH.is_file()

FAMILIES = tensor_region_generation.FAMILIES
STRATUM = "loop-reduction"


def generate(
  output_dir,
  *,
  seed=5,
  count=2,
  family=None,
  stratum=STRATUM,
  generator="constructor",
  version="v2",
  dtype="f32",
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
    dtype,
    "--tensor-profile-version",
    version,
    "--tensor-generator",
    generator,
    "--output-dir",
    str(output_dir),
  ]
  if stratum is not None:
    arguments += ["--tensor-stratum", stratum]
  if family is not None:
    arguments += ["--tensor-region-family", family]
  return subprocess.run(
    arguments, capture_output=True, text=True, check=False, timeout=timeout_s
  )


def case_sources(output_dir):
  """Program sources of one run, keyed by case directory name."""
  return {
    path.parent.name: path.read_text(encoding="utf-8")
    for path in sorted(Path(output_dir).glob("case_*/program.sir"))
  }


def is_region(source):
  """A loop region updates both named states from a shared two-state body."""
  return "%acc_x = %next_x;" in source and "%acc_y = %next_y;" in source


class TensorRegionFamilyOptionTests(unittest.TestCase):
  def setUp(self):
    if not READY:
      self.skipTest("rysmith is supplied by the managed Make target")

  def test_named_family_changes_the_emitted_source(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      fold = generate(root / "fold", count=2, family="fold")
      cross_state = generate(root / "cross-state", count=2, family="cross-state")

      self.assertEqual(fold.returncode, 0, fold.stdout + fold.stderr)
      self.assertEqual(
        cross_state.returncode, 0, cross_state.stdout + cross_state.stderr
      )
      fold_sources = case_sources(root / "fold")
      cross_state_sources = case_sources(root / "cross-state")

    self.assertEqual(len(fold_sources), 2)
    self.assertEqual(len(cross_state_sources), 2)
    self.assertTrue(all(is_region(source) for source in fold_sources.values()))
    self.assertTrue(all(is_region(source) for source in cross_state_sources.values()))
    self.assertNotEqual(
      set(fold_sources.values()),
      set(cross_state_sources.values()),
      "naming a family must change the emitted source",
    )

  def test_every_named_family_is_accepted(self):
    for family in FAMILIES:
      with self.subTest(family=family):
        with tempfile.TemporaryDirectory() as directory:
          result = generate(Path(directory), count=2, family=family)
          emitted = case_sources(Path(directory))

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(emitted), 2)
        self.assertTrue(all(is_region(source) for source in emitted.values()))

  def test_unknown_region_family_is_refused(self):
    with tempfile.TemporaryDirectory() as directory:
      result = generate(Path(directory), count=1, family="transpose")

    self.assertNotEqual(result.returncode, 0)
    combined = result.stdout + result.stderr
    self.assertIn("tensor region family", combined)
    self.assertIn("mixed-depth", combined)

  def test_absent_option_keeps_the_unforced_loop_form(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      absent = generate(root / "absent", seed=5, count=24)
      named = generate(root / "fold", seed=5, count=24, family="fold")

      self.assertEqual(absent.returncode, 0, absent.stdout + absent.stderr)
      self.assertEqual(named.returncode, 0, named.stdout + named.stderr)
      absent_sources = case_sources(root / "absent")
      named_sources = case_sources(root / "fold")

    regions = {
      name: source for name, source in absent_sources.items() if is_region(source)
    }
    self.assertTrue(regions, "the default route must still reach the fold region")
    self.assertGreater(
      len(absent_sources) - len(regions),
      0,
      "an absent option must leave the other loop forms scheduled",
    )
    for name, source in regions.items():
      self.assertEqual(
        named_sources[name],
        source,
        "an unnamed region must be the fold family, unchanged",
      )

  def test_named_family_reaches_the_exported_graph(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      orthogonal = generate(
        root / "orthogonal", seed=7, count=2, family="orthogonal-reductions"
      )
      cross_state = generate(
        root / "cross-state", seed=7, count=2, family="cross-state"
      )

      self.assertEqual(orthogonal.returncode, 0, orthogonal.stdout + orthogonal.stderr)
      self.assertEqual(
        cross_state.returncode, 0, cross_state.stdout + cross_state.stderr
      )
      orthogonal_sources = case_sources(root / "orthogonal")
      cross_state_sources = case_sources(root / "cross-state")

    for source in orthogonal_sources.values():
      self.assertEqual(
        set(tensor_region_generation.reduction_axes(source)),
        {0, 1},
        "the orthogonal family must reduce both axes",
      )
    for source in cross_state_sources.values():
      dependencies = tensor_region_generation.root_dependencies(source)
      self.assertEqual(
        set(tensor_region_generation.producer_operands(source)),
        {"acc_x", "acc_y"},
      )
      for root in ("acc_x", "acc_y"):
        self.assertTrue(
          dependencies[root] >= {"acc_x", "acc_y"},
          "each cross-state root must depend on both current states",
        )


class TensorRegionFamilyOptionEdgeTests(unittest.TestCase):
  def setUp(self):
    if not READY:
      self.skipTest("rysmith is supplied by the managed Make target")

  def test_named_family_is_refused_without_the_constructor_loop_route(self):
    with tempfile.TemporaryDirectory() as directory:
      result = generate(
        Path(directory), count=1, family="cross-state", generator="compositional"
      )

    self.assertNotEqual(result.returncode, 0)
    combined = result.stdout + result.stderr
    self.assertIn("constructor", combined)
    self.assertIn("does not build loop regions", combined)

  def test_named_family_is_refused_with_a_foreign_stratum(self):
    with tempfile.TemporaryDirectory() as directory:
      result = generate(
        Path(directory), count=1, family="cross-state", stratum="untargeted-exploration"
      )

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("loop-reduction", result.stdout + result.stderr)


if __name__ == "__main__":
  unittest.main()
