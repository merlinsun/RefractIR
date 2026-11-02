"""Free-form bundles exported as layout-mutator seeds (seed-export plan, E1)."""

import dataclasses
import hashlib
import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from test.lib import (
  tensor_free_form_case,
  tensor_free_form_inputs,
  tensor_relations_sidecar,
  tensor_seed_export,
  tensor_target_reference,
)
from test.unit import layout_mutator_seed_format as seed_format
from test.unit.tensor_support import SYMIRC, SYMIRI

ROOT = Path(__file__).resolve().parents[2]
EXPORTER = Path(
  os.environ.get(
    "REFRACTIR_TENSOR_FREE_FORM_EXPORT",
    ROOT / "build" / "bin" / "tensor_free_form_export",
  )
)
WINDOW = ["--trip-count", "4", "--window", "2,1,5", "--window-buffer-rows", "7"]


def export_body(directory, *extra, seed=2):
  out = Path(directory) / "export"
  subprocess.run(
    [
      str(EXPORTER),
      "--policy",
      "unguided",
      "--seed",
      str(seed),
      "--shape",
      "2,8",
      "--profile",
      "v3",
      *WINDOW,
      "--trip-parameter",
      *extra,
      "--out",
      str(out),
    ],
    check=True,
    capture_output=True,
  )
  region = json.loads((out / "region.json").read_text(encoding="utf-8"))
  return region, (out / "program.sir").read_text(encoding="utf-8")


def bundle(directory, *, trips=3, layouts=None):
  region, source = export_body(directory)
  proposal = tensor_free_form_inputs.window_base_proposal(
    region, group=0, column=0, trips=trips
  )
  case = tensor_free_form_case.build_bundle(
    Path(directory) / "bundle",
    region=region,
    source=source,
    proposal=proposal,
    input_layouts=layouts,
  )
  return region, case


@unittest.skipUnless(
  EXPORTER.is_file() and SYMIRC.is_file(), "the exporter and symirc must be built"
)
class SeedExportTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name)

  def export(self, **options):
    region, case = bundle(self.root, **options)
    seed_dir = tensor_seed_export.export_seed(
      case.path, region, self.root / "seed", symirc=SYMIRC, provenance={"seed": 2}
    )
    return region, case, seed_dir

  def test_a_seed_round_trips_through_the_harness_reader(self):
    _, case, seed_dir = self.export()
    seed = seed_format.read_seed(seed_dir)
    self.assertEqual(seed.launch.kernel, "kernel")
    self.assertEqual(
      seed.launch.names, ["window_buffer_ptr", "x_ptr", "y_ptr", "trips_ptr", "out_ptr"]
    )
    self.assertIn("@triton.jit", seed.kernel_source)
    self.assertEqual(tuple(seed.launch.grid), tuple(case.record["grid"]))
    self.assertEqual(seed.launch.options, case.record["launch"])

  def test_tensors_keep_the_bundles_shape_dtype_strides_and_bytes(self):
    _, case, seed_dir = self.export()
    launch = seed_format.read_seed(seed_dir).launch
    records = {record["name"].lstrip("%"): record for record in case.record["inputs"]}
    for name, arg in zip(launch.names, launch.args):
      if name == "out_ptr":
        self.assertEqual(arg.fill, {"rule": "zeros"})
        self.assertEqual(list(arg.shape), case.record["output"]["shape"])
        continue
      record = records[name.removesuffix("_ptr")]
      self.assertEqual(list(arg.shape), record["shape"])
      self.assertEqual(arg.dtype, {"f32": "float32", "i32": "int32"}[record["dtype"]])
      self.assertEqual(list(arg.strides), record["element_strides"])
      data = (seed_dir / "inputs" / arg.raw).read_bytes()
      start = record["byte_offset"]
      self.assertEqual(data, case.inputs[start : start + record["byte_length"]])

  def test_the_trip_parameter_is_an_int32_scalar_tensor_in_the_path(self):
    _, _, seed_dir = self.export(trips=3)
    launch = seed_format.read_seed(seed_dir).launch
    trips = launch.args[launch.names.index("trips_ptr")]
    self.assertEqual((trips.shape, trips.dtype, trips.strides), ((), "int32", ()))
    self.assertEqual(
      (seed_dir / "inputs" / trips.raw).read_bytes(), (3).to_bytes(4, "little")
    )
    sidecar = json.loads((seed_dir / "relations.json").read_text())
    self.assertEqual(sidecar["path"], {"trip_counts": {"trips": 3}})

  def test_the_sidecar_and_provenance_name_their_sources(self):
    _, case, seed_dir = self.export()
    sidecar = json.loads((seed_dir / "relations.json").read_text())
    self.assertEqual(sidecar["schema"], "layout-mutator.relations-sidecar.v1")
    self.assertEqual(sidecar["source"], "refractir")
    self.assertEqual(sidecar["expected"], {"p0_sha256": None, "p0_file": None})
    provenance = seed_format.read_seed(seed_dir).provenance
    self.assertNotIn("p0", provenance, "no symiri, so no P0 record")
    self.assertEqual(provenance["source"], "refractir")
    self.assertEqual(provenance["seed"], 2)
    self.assertEqual(
      provenance["program_sha256"],
      hashlib.sha256(case.source.encode("utf-8")).hexdigest(),
    )
    self.assertEqual(
      provenance["kernel_sha256"],
      hashlib.sha256((seed_dir / "kernel.py").read_bytes()).hexdigest(),
    )

  def test_the_sidecar_carries_the_bodys_relation_facts(self):
    region, case, seed_dir = self.export(trips=3)
    written = json.loads((seed_dir / "relations.json").read_text())
    facts = tensor_relations_sidecar.analyze(
      case.source, tensor_free_form_inputs.parameters(region), {"%trips": 3}
    )
    self.assertEqual(written["outputs"], facts["outputs"])
    self.assertEqual(written["inputs"], facts["inputs"])
    self.assertEqual(written["inputs"]["trips_ptr"], {"float": False})
    self.assertIn(
      written["outputs"]["out_ptr"]["class"], ("specified", "order_dependent")
    )

  def test_a_branch_selector_is_recorded_in_the_provenance(self):
    for seed in range(1, 21):
      try:
        region, source = export_body(
          self.root / str(seed), "--branch-parameter", seed=seed
        )
      except subprocess.CalledProcessError:
        continue
      proposal = tensor_free_form_inputs.window_base_proposal(
        region, group=0, column=0, trips=2, branch=1
      )
      case = tensor_free_form_case.build_bundle(
        self.root / str(seed) / "bundle",
        region=region,
        source=source,
        proposal=proposal,
      )
      seed_dir = tensor_seed_export.export_seed(
        case.path, region, self.root / "seed", symirc=SYMIRC
      )
      launch = seed_format.read_seed(seed_dir).launch
      self.assertEqual(launch.names[-2:], ["sel_ptr", "out_ptr"])
      self.assertEqual(seed_format.read_seed(seed_dir).provenance["branch"], {"sel": 1})
      return
    self.fail("no exportable branch body within 20 seeds")

  def test_a_padded_layout_is_refused(self):
    layouts = {
      "%window_buffer": tensor_free_form_case.strided_output([7, 8], row_padding=3)
    }
    region, case = bundle(self.root, layouts=layouts)
    with self.assertRaises(tensor_seed_export.ExportError):
      tensor_seed_export.export_seed(
        case.path, region, self.root / "seed", symirc=SYMIRC
      )


def overflowing_bundle(root):
  """The first exported body that P0 refuses on inputs near the f32 maximum."""
  for seed in range(1, 31):
    directory = Path(root) / f"overflow-{seed}"
    try:
      region, source = export_body(directory, seed=seed)
    except subprocess.CalledProcessError:
      continue
    proposal = tensor_free_form_inputs.window_base_proposal(
      region, group=0, column=0, trips=3
    )
    values = bytearray(proposal.data)
    for offset in range(0, len(values) - 4, 4):
      values[offset : offset + 4] = struct.pack("<f", 3.0e38)
    huge = dataclasses.replace(proposal, data=bytes(values))
    case = tensor_free_form_case.build_bundle(
      directory / "bundle", region=region, source=source, proposal=huge
    )
    observation = tensor_target_reference.observe(case, symiri=SYMIRI)
    if observation.status == "ub":
      return region, case
  raise AssertionError("no body overflowed within 30 seeds")


@unittest.skipUnless(
  EXPORTER.is_file() and SYMIRC.is_file() and SYMIRI.is_file(),
  "the exporter, symirc and symiri must be built",
)
class ExpectedOutputTests(unittest.TestCase):
  """P0's output as the seed's expected output (seed-export plan, E3)."""

  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name)

  def export(self, region, case, **options):
    return tensor_seed_export.export_seed(
      case.path, region, self.root / "seed", symirc=SYMIRC, **options
    )

  def test_the_expected_hash_is_p0s_output(self):
    region, case = bundle(self.root, trips=3)
    seed_dir = self.export(region, case, symiri=SYMIRI)
    observation = tensor_target_reference.observe(
      case, symiri=SYMIRI, output_path=self.root / "direct.bin"
    )
    expected = json.loads((seed_dir / "relations.json").read_text())["expected"]
    self.assertEqual(
      expected["p0_sha256"], hashlib.sha256(observation.output).hexdigest()
    )

  def test_the_expected_bytes_are_written_beside_the_kernel(self):
    region, case = bundle(self.root, trips=3)
    seed_dir = self.export(region, case, symiri=SYMIRI)
    expected = json.loads((seed_dir / "relations.json").read_text())["expected"]
    self.assertEqual(expected["p0_file"], "outputs/out_ptr.p0.bin")
    data = (seed_dir / expected["p0_file"]).read_bytes()
    self.assertEqual(hashlib.sha256(data).hexdigest(), expected["p0_sha256"])
    self.assertEqual(len(data), 4 * 16)

  def test_an_overflowing_input_leaves_the_expected_output_empty(self):
    region, case = overflowing_bundle(self.root)
    seed_dir = self.export(region, case, symiri=SYMIRI)
    expected = json.loads((seed_dir / "relations.json").read_text())["expected"]
    self.assertEqual(expected, {"p0_sha256": None, "p0_file": None})
    self.assertFalse((seed_dir / "outputs").exists())

  def test_a_p0_refusal_is_recorded_with_its_reason(self):
    region, case = overflowing_bundle(self.root)
    seed_dir = self.export(region, case, symiri=SYMIRI)
    p0 = json.loads((seed_dir / "seed.json").read_text())["p0"]
    self.assertEqual(p0["status"], "p0_ub")
    self.assertTrue(p0["reason"])

  def test_a_completed_run_is_recorded_too(self):
    region, case = bundle(self.root, trips=3)
    seed_dir = self.export(region, case, symiri=SYMIRI)
    p0 = json.loads((seed_dir / "seed.json").read_text())["p0"]
    self.assertEqual(p0["status"], "completed")


if __name__ == "__main__":
  unittest.main()
