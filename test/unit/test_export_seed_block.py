"""A seed block exported for layout-mutator (seed-export plan, E4)."""

import importlib.util
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test.unit import layout_mutator_seed_format as seed_format
from test.unit.tensor_support import SYMIRC, SYMIRI

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
  "export_seed_block", ROOT / "test" / "tensor" / "campaigns" / "export_seed_block.py"
)
block = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(block)

EXPORTER = Path(
  os.environ.get(
    "REFRACTIR_TENSOR_FREE_FORM_EXPORT",
    ROOT / "build" / "bin" / "tensor_free_form_export",
  )
)
START = 47000000


@unittest.skipUnless(
  EXPORTER.is_file() and SYMIRC.is_file() and SYMIRI.is_file(),
  "the exporter, symirc and symiri must be built",
)
class SeedBlockTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name)

  def export(self, out, *, exporter=EXPORTER, count=8, feature="where"):
    return block.export_block(
      feature,
      seed_start=START,
      seed_count=count,
      out=Path(out),
      exporter=exporter,
      symirc=SYMIRC,
      symiri=SYMIRI,
    )

  def test_every_exported_body_is_a_seed_with_a_sidecar_and_p0(self):
    record = self.export(self.root / "block")
    exported = [entry for entry in record["entries"] if entry["status"] == "exported"]
    self.assertEqual(len(record["entries"]), 8)
    self.assertTrue(exported)
    for entry in exported:
      seed_dir = self.root / "block" / entry["directory"]
      seed_format.read_seed(seed_dir)
      sidecar = json.loads((seed_dir / "relations.json").read_text())
      self.assertIn("out_ptr", sidecar["outputs"])
      self.assertIn("p0", seed_format.read_seed(seed_dir).provenance)

  def test_a_refused_body_is_logged_and_not_exported(self):
    failing = self.root / "refuse.sh"
    failing.write_text("#!/bin/sh\necho 'free_form_exhaustion: planted' >&2\nexit 1\n")
    failing.chmod(failing.stat().st_mode | stat.S_IEXEC)
    record = self.export(self.root / "block", exporter=failing, count=2)
    self.assertEqual([e["status"] for e in record["entries"]], ["refused", "refused"])
    self.assertIn("planted", record["entries"][0]["reason"])
    self.assertEqual(
      sorted(p.name for p in (self.root / "block").iterdir()), ["BLOCK.json"]
    )

  def test_an_emission_failure_is_logged_and_leaves_no_partial_seed(self):
    def refuse(*args, **kwargs):
      raise block.tensor_seed_export.ExportError("symirc refused the bundle: planted")

    with mock.patch.object(block.tensor_seed_export, "export_seed", refuse):
      record = self.export(self.root / "block", count=2)
    statuses = {entry["status"] for entry in record["entries"]}
    self.assertLessEqual(statuses, {"export_failed", "refused"})
    failed = [e for e in record["entries"] if e["status"] == "export_failed"]
    self.assertTrue(failed)
    self.assertIn("planted", failed[0]["reason"])
    self.assertEqual(
      sorted(p.name for p in (self.root / "block").iterdir()), ["BLOCK.json"]
    )

  def test_each_seed_carries_its_probes_as_named_variants(self):
    record = self.export(self.root / "block", count=4)
    entry = next(e for e in record["entries"] if e["status"] == "exported")
    seed_dir = self.root / "block" / entry["directory"]
    variants = sorted(p.name for p in (seed_dir / "variants").iterdir())
    self.assertEqual(variants, sorted(block.VARIANTS[1:]))
    base = seed_format.read_seed(seed_dir)
    for name in variants:
      variant = seed_format.read_seed(seed_dir / "variants" / name)
      self.assertEqual(variant.kernel_source, base.kernel_source)
      self.assertEqual(variant.provenance["input"], name)
    self.assertEqual(base.provenance["input"], block.VARIANTS[0])

  def test_the_block_digest_is_recorded_and_reproducible(self):
    first = self.export(self.root / "first", count=3)
    second = self.export(self.root / "second", count=3)
    self.assertEqual(first["digest"], second["digest"])
    written = json.loads((self.root / "first" / "BLOCK.json").read_text())
    self.assertEqual(written["digest"], first["digest"])
    self.assertEqual(len(first["digest"]), 64)

  def test_no_certification_rule_reaches_the_exporter(self):
    request = block.block_request("where", START, 0)
    argv = block.export_command(EXPORTER, request, self.root / "out")
    self.assertNotIn("--exact-input-bound", argv)
    self.assertNotIn("--exact-contract", argv)
    self.assertNotIn("--window-requirement", argv)
    self.assertEqual(argv[argv.index("--max-nodes") + 1], str(block.EXPORT_MAX_NODES))
    self.assertIn("--trip-parameter", argv)
    self.assertEqual(argv[argv.index("--features") + 1], "where")


if __name__ == "__main__":
  unittest.main()
