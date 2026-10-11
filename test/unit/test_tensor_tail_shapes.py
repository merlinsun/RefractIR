"""Rank-two logical tails in the compositional tensor generator (v3)."""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RYSMITH = Path(os.environ.get("REFRACTIR_RYSMITH", ROOT / "rysmith")).resolve()
SYMIRI = Path(os.environ.get("REFRACTIR_SYMIRI", ROOT / "symiri")).resolve()
SYMIRC = Path(os.environ.get("REFRACTIR_SYMIRC", ROOT / "symirc")).resolve()
READY = RYSMITH.is_file()

# The rank-two tails the compositional policy may choose, each paired so its
# padded tile stays inside the 512-element profile volume.
PLANNED_TAILS = ((31, 8), (33, 8), (8, 31), (8, 33), (63, 4), (4, 65))


def _load_case_module():
  spec = importlib.util.find_spec("test.lib.tensor_case")
  if spec is None:
    return None
  from test.lib import tensor_case

  return tensor_case


def generate(output_dir, *, seed, count, generator="compositional", target="sir"):
  return subprocess.run(
    [
      str(RYSMITH),
      "--profile",
      "tensor",
      "--target",
      target,
      "--seed",
      str(seed),
      "-n",
      str(count),
      "--tensor-dtype",
      "i32",
      "--tensor-profile-version",
      "v3",
      "--tensor-generator",
      generator,
      "--output-dir",
      str(output_dir),
    ],
    capture_output=True,
    text=True,
    check=False,
    timeout=300,
  )


def next_power_of_two(value):
  extent = 1
  while extent < value:
    extent <<= 1
  return extent


def is_boundary_tail(shape):
  """A rank-two shape with a published boundary length on one axis.

  The generator also produces tiny non-power-of-two shapes when a reducibility
  fixup fires; those are not the rank-two tails this policy adds.
  """
  return (
    len(shape) == 2
    and all(extent > 1 for extent in shape)
    and any(extent >= 17 and extent != next_power_of_two(extent) for extent in shape)
  )


def tailed_axis(shape):
  """The axis whose extent is the published boundary length, if there is one."""
  axes = [
    index
    for index, extent in enumerate(shape)
    if extent >= 17 and extent != next_power_of_two(extent)
  ]
  return axes[0] if len(axes) == 1 else None


class TensorCompositionalTailTests(unittest.TestCase):
  def setUp(self):
    if not READY:
      self.skipTest("rysmith is supplied by the managed Make target")

  @classmethod
  def generated_cases(cls):
    """Generate one frozen sweep shared by the assertions in this class."""
    if not hasattr(cls, "_cases"):
      cases = []
      with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for seed in range(1, 9):
          output_dir = root / f"seed-{seed}"
          result = generate(output_dir, seed=seed, count=6)
          if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)
          for case_dir in sorted(output_dir.glob("case_*")):
            import json

            cases.append(
              (
                case_dir,
                json.loads((case_dir / "case.json").read_text(encoding="utf-8")),
                (case_dir / "program.sir").read_text(encoding="utf-8"),
                (case_dir / "inputs.bin").read_bytes(),
                (case_dir / "expected.bin").read_bytes(),
              )
            )
      cls._cases = cases
    return cls._cases

  def test_rank_two_tails_enter_compositional_generation(self):
    shapes = set()
    for _dir, manifest, _source, _inputs, _expected in self.generated_cases():
      shape = tuple(manifest["inputs"][0]["shape"])
      if len(shape) == 2 and shape != (
        next_power_of_two(shape[0]),
        next_power_of_two(shape[1]),
      ):
        shapes.add(shape)

    self.assertTrue(shapes, "compositional v3 generation must reach rank-two tails")
    self.assertGreaterEqual(
      len(shapes & set(PLANNED_TAILS)),
      2,
      f"the planned tail pairs must be reachable, observed {sorted(shapes)}",
    )

  def test_padded_tiles_are_the_next_power_of_two_within_the_volume(self):
    observed_tiles = 0
    for _dir, manifest, _source, _inputs, _expected in self.generated_cases():
      for record in (*manifest["inputs"], manifest["output"]):
        logical = tuple(record["shape"])
        tile = tuple(record["tile_shape"])
        self.assertEqual(
          tile,
          tuple(next_power_of_two(extent) for extent in logical),
          f"tile {tile} is not the next power of two of {logical}",
        )
        self.assertLessEqual(math.prod(tile), 512)
        observed_tiles += 1

    self.assertGreater(observed_tiles, 0)

  def test_tailed_shapes_cover_both_axis_orientations(self):
    tails = set()
    for _dir, manifest, _source, _inputs, _expected in self.generated_cases():
      shape = tuple(manifest["inputs"][0]["shape"])
      if is_boundary_tail(shape):
        tails.add(shape)

    self.assertGreaterEqual(
      len(tails),
      2,
      f"the sweep must reach more than one planned tail pair, observed {sorted(tails)}",
    )
    self.assertEqual(
      {tailed_axis(shape) for shape in tails},
      {0, 1},
      f"both tailed-axis orientations must be reachable, observed {sorted(tails)}",
    )

  def test_tailed_case_padding_stays_out_of_the_logical_payloads(self):
    tailed = [
      case
      for case in self.generated_cases()
      if is_boundary_tail(tuple(case[1]["inputs"][0]["shape"]))
    ]
    self.assertTrue(tailed, "no rank-two tail case was generated")

    for _dir, manifest, source, inputs, expected in tailed:
      shape = tuple(manifest["inputs"][0]["shape"])
      record = manifest["inputs"][0]
      tile = tuple(record["tile_shape"])
      with self.subTest(shape=shape):
        self.assertIn(shape, PLANNED_TAILS)
        self.assertNotEqual(
          shape,
          tile,
          "a logical tail must be padded instead of rounded up in place",
        )
        self.assertIn(f"tensor [{shape[0]}, {shape[1]}]", source)
        output_shape = tuple(manifest["output"]["shape"])
        # The reference output is written at its logical size too, whatever the
        # physical tile the kernel stores it through.
        self.assertEqual(manifest["output"]["byte_length"], len(expected))
        self.assertEqual(len(expected), math.prod(output_shape) * 4)
        # The logical payload is the strided sub-region of the padded storage,
        # so its byte length is the logical element count and never the tile's.
        self.assertEqual(record["byte_length"], math.prod(shape) * 4)
        self.assertGreaterEqual(len(inputs), record["byte_length"])
        self.assertEqual(len(expected), math.prod(shape) * 4)
        covered = (
          record["element_offset"]
          + 1
          + sum(
            (extent - 1) * stride
            for extent, stride in zip(shape, record["element_strides"])
          )
        )
        self.assertLessEqual(covered, record["storage_span"])

  def test_tailed_case_reference_is_complete_and_replayable(self):
    if not SYMIRI.is_file():
      self.skipTest("the reference interpreter comes from the managed Make target")
    cases = [
      case
      for case in self.generated_cases()
      if is_boundary_tail(tuple(case[1]["inputs"][0]["shape"]))
    ]
    self.assertTrue(cases, "no rank-two tail case was generated")
    _dir, manifest, source, inputs, expected = cases[0]
    element = manifest["inputs"][0]["dtype"]
    count = math.prod(manifest["output"]["shape"])
    self.assertEqual(len(expected), count * 4)

    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / "program.sir").write_text(source, encoding="utf-8")
      (root / "inputs.bin").write_bytes(inputs)
      result = subprocess.run(
        [
          str(SYMIRI),
          "--tensor-profile-version",
          "v3",
          "--entry",
          "kernel",
          "--tensor-input",
          str(root / "inputs.bin"),
          "--tensor-output",
          str(root / "output.bin"),
          str(root / "program.sir"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
      )
      self.assertEqual(result.returncode, 0, result.stderr)
      replayed = (root / "output.bin").read_bytes()

    self.assertEqual(element, "i32")
    self.assertEqual(replayed, expected)

  def test_tailed_program_emits_a_masked_triton_tile(self):
    if not SYMIRC.is_file():
      self.skipTest("the compiler comes from the managed Make target")
    import json

    tailed = []
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for seed in range(1, 9):
        result = generate(root / f"seed-{seed}", seed=seed, count=6, target="triton")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for kernel_path in sorted((root / f"seed-{seed}").glob("case_*/kernel.py")):
          manifest = json.loads(
            (kernel_path.parent / "case.json").read_text(encoding="utf-8")
          )
          shape = tuple(manifest["inputs"][0]["shape"])
          if not is_boundary_tail(shape):
            continue
          tailed.append((shape, kernel_path.read_text(encoding="utf-8")))

    self.assertTrue(tailed, "no rank-two tail kernel was emitted")
    for shape, kernel in tailed:
      axis = next(
        index
        for index, extent in enumerate(shape)
        if extent != next_power_of_two(extent)
      )
      self.assertIn("mask=", kernel)
      self.assertIn(f"< {shape[axis]}", kernel)
      self.assertIn(f"tl.arange(0, {next_power_of_two(shape[axis])})", kernel)
      self.assertIn("other=0", kernel)


class TensorCompositionalTailEdgeTests(unittest.TestCase):
  def setUp(self):
    if not READY:
      self.skipTest("rysmith is supplied by the managed Make target")

  def test_power_of_two_rank_two_shape_keeps_tile_equal_to_shape(self):
    for (
      _dir,
      manifest,
      _source,
      _inputs,
      _expected,
    ) in TensorCompositionalTailTests.generated_cases():
      logical = tuple(manifest["inputs"][0]["shape"])
      if len(logical) == 2 and logical == (
        next_power_of_two(logical[0]),
        next_power_of_two(logical[1]),
      ):
        self.assertEqual(tuple(manifest["inputs"][0]["tile_shape"]), logical)
        return
    self.skipTest("the sweep produced no power-of-two rank-two shape")

  def test_padded_tile_over_the_profile_volume_is_rejected(self):
    if not SYMIRI.is_file():
      self.skipTest("the interpreter comes from the managed Make target")
    source = """\
fun @kernel(%x: tensor [65, 16] i32) : tensor [65, 16] i32 {
  let mut %out: tensor [65, 16] i32 = undef;
^entry:
  %out = %x;
  ret %out;
}
"""
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "wide.sir"
      path.write_text(source, encoding="utf-8")
      result = subprocess.run(
        [str(SYMIRI), "--check", "--tensor-profile-version", "v3", str(path)],
        capture_output=True,
        text=True,
        check=False,
      )

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("padded tile volume exceeds 512", result.stdout + result.stderr)


if __name__ == "__main__":
  unittest.main()
