"""Generation blocks for the relations harness (generation plan G9 and G10).

A generation block builds tensor-v5 bodies without any rule that exists only to
certify P0: no exact-input bound or contract, node and multiplication caps of
32, inputs uniform in [-1, 1]. Each body is steered (G3), and every reached
target becomes one seed directory sharing the body's kernel, with
`path.target` in its sidecar; the first is the seed itself and the rest are
variants. Tiled, strided and half-precision blocks carry the launch arguments
the relations vary and the sidecar's `launch` fields. No GPU is required.
"""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from test.unit.tensor_support import SYMIR_SOLVE, SYMIRC, SYMIRI

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
  "export_seed_block", ROOT / "test" / "tensor" / "campaigns" / "export_seed_block.py"
)
block = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(block)

EXPORTER = Path(
  os.environ.get(
    "REFRACTIR_TENSOR_FREE_FORM_EXPORT", "build/bin/tensor_free_form_export"
  )
)
STEERING = Path(
  os.environ.get("REFRACTIR_TENSOR_PATH_STEERING", "build/bin/tensor_path_steering")
)
READY = all(
  path.is_file() for path in (EXPORTER, STEERING, SYMIRC, SYMIRI, SYMIR_SOLVE)
)


def export(kind, count=4, start=900, exporter=EXPORTER):
  directory = tempfile.mkdtemp()
  record = block.export_block(
    kind,
    seed_start=start,
    seed_count=count,
    out=directory,
    exporter=exporter,
    symirc=SYMIRC,
    symiri=SYMIRI,
  )
  return Path(directory), record


def seeds(directory):
  return sorted(path for path in directory.glob("seed-*") if path.is_dir())


def read(path, name):
  return json.loads((path / name).read_text(encoding="utf-8"))


@unittest.skipUnless(READY, "the exporter, steering driver or tools are unavailable")
class GenerationBlockTests(unittest.TestCase):
  def test_a_loops_block_exports_every_body_with_its_targets(self):
    directory, record = export("generation-loops")
    self.assertEqual(record["exported"], 4, record["entries"])
    for seed in seeds(directory):
      with self.subTest(seed=seed.name):
        relations = read(seed, "relations.json")
        self.assertIn("target", relations["path"])
        self.assertEqual(relations["path"]["step"], -2)
        targets = {relations["path"]["target"]} | {
          read(variant, "relations.json")["path"]["target"]
          for variant in (seed / "variants").iterdir()
        }
        self.assertIn("loop0-empty-inverted", targets)
        launch = read(seed, "launch.json")
        self.assertEqual(launch["names"][-1], "out_ptr")
        self.assertEqual(
          (seed / "kernel.py").read_text(),
          (
            seed / "variants" / sorted(os.listdir(seed / "variants"))[0] / "kernel.py"
          ).read_text(),
        )

  def test_a_refused_body_is_logged_and_not_exported(self):
    directory, record = export(
      "generation-loops", count=2, exporter=Path("/usr/bin/false")
    )
    self.assertEqual(record["exported"], 0)
    self.assertEqual({entry["status"] for entry in record["entries"]}, {"refused"})
    self.assertEqual(seeds(directory), [])

  def test_no_generation_request_states_an_exact_rule(self):
    for kind in block.GENERATION_BLOCKS:
      with self.subTest(kind=kind):
        command = block.generation_command(EXPORTER, kind, 900, Path("/tmp/x"))
        self.assertFalse([flag for flag in command if "exact" in flag], command)
        self.assertIn("v5", command)
        self.assertEqual(command[command.index("--max-nodes") + 1], "32")

  def test_a_tiled_block_names_its_extent_and_block(self):
    directory, record = export("generation-tiled")
    self.assertGreaterEqual(record["exported"], 3, record["entries"])
    for seed in seeds(directory):
      with self.subTest(seed=seed.name):
        launch = read(seed, "launch.json")
        arguments = dict(zip(launch["names"], launch["args"]))
        self.assertEqual(arguments["BLOCK"], {"kind": "constexpr", "value": 16})
        extent = arguments["N"]["value"]
        self.assertEqual(launch["grid"][0], -(-extent // 16))
        relations = read(seed, "relations.json")
        self.assertEqual(
          relations["launch"]["tiles"], {"BLOCK": {"extent_arg": "N", "grid_axis": 0}}
        )
        self.assertIn(read(seed, "seed.json")["tile"]["corner"], block.TILE_CORNERS)

  def test_a_strided_block_names_its_stride_arguments(self):
    directory, _ = export("generation-strides", count=2)
    for seed in seeds(directory):
      with self.subTest(seed=seed.name):
        launch = read(seed, "launch.json")
        self.assertIn("x_stride0", launch["names"])
        relations = read(seed, "relations.json")
        self.assertEqual(relations["launch"]["offsets"]["x_ptr"], "x_offset")

  def test_a_half_block_stores_its_input_as_float16(self):
    directory, _ = export("generation-dtypes", count=2)
    for seed in seeds(directory):
      with self.subTest(seed=seed.name):
        launch = read(seed, "launch.json")
        x = launch["args"][launch["names"].index("x_ptr")]
        self.assertEqual(x["dtype"], "float16")
        elements = 1
        for extent in x["shape"]:
          elements *= extent
        self.assertEqual((seed / "inputs" / "x.bin").stat().st_size, 2 * elements)

  def test_a_literals_block_lists_its_literal_operands(self):
    directory, _ = export("generation-literals")
    listed = [read(seed, "relations.json")["literals"] for seed in seeds(directory)]
    self.assertTrue(any(listed), listed)

  def test_every_generated_seed_loads_through_layout_mutators_reader(self):
    from test.unit import layout_mutator_seed_format as seed_format

    for kind in ("generation-tiled", "generation-strides", "generation-dtypes"):
      directory, _ = export(kind, count=1)
      for seed in seeds(directory):
        with self.subTest(kind=kind, seed=seed.name):
          loaded = seed_format.read_seed(seed)
          self.assertEqual(loaded.launch.names, read(seed, "launch.json")["names"])
          for variant in (seed / "variants").iterdir():
            seed_format.read_seed(variant)


if __name__ == "__main__":
  unittest.main()
