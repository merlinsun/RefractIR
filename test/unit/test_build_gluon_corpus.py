"""Free-form bodies routed to the Gluon layout runner (feature-ranked plan, L5)."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
  "build_gluon_corpus",
  _ROOT / "test" / "tensor" / "campaigns" / "build_gluon_corpus.py",
)
corpus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(corpus)

EXPORTER = Path(
  os.environ.get(
    "REFRACTIR_TENSOR_FREE_FORM_EXPORT",
    _ROOT / "build" / "bin" / "tensor_free_form_export",
  )
)


@unittest.skipUnless(EXPORTER.is_file(), "the free-form exporter is not built")
class GluonCorpusTests(unittest.TestCase):
  def build(self, seeds, out):
    return corpus.build(EXPORTER, seeds=seeds, shape=(2, 8), out=Path(out))

  def test_rendered_bodies_keep_their_canonical_program_and_kernel(self):
    with tempfile.TemporaryDirectory() as directory:
      manifest = self.build(range(1, 11), directory)
      rendered = [
        entry for entry in manifest["entries"] if entry["status"] == "rendered"
      ]
      self.assertTrue(rendered, manifest)
      for entry in rendered:
        program = Path(directory) / entry["program"]
        self.assertEqual(corpus.sha256(program), entry["program_sha256"])
        kernel = Path(directory) / entry["kernel"]
        self.assertIn("@gluon.jit", kernel.read_text())
        self.assertEqual(corpus.sha256(kernel), entry["kernel_sha256"])

  def test_every_seed_has_one_outcome_with_its_reason(self):
    with tempfile.TemporaryDirectory() as directory:
      manifest = self.build(range(1, 11), directory)
      self.assertEqual(
        [entry["seed"] for entry in manifest["entries"]], list(range(1, 11))
      )
      for entry in manifest["entries"]:
        self.assertIn(
          entry["status"], ("rendered", "unsupported", "no_canonical", "refused")
        )
        if entry["status"] != "rendered":
          self.assertTrue(entry["reason"])

  def test_a_build_is_reproducible(self):
    with (
      tempfile.TemporaryDirectory() as first,
      tempfile.TemporaryDirectory() as second,
    ):
      self.assertEqual(self.build(range(1, 6), first), self.build(range(1, 6), second))

  def test_the_manifest_is_written_beside_the_bodies(self):
    with tempfile.TemporaryDirectory() as directory:
      manifest = self.build(range(1, 4), directory)
      written = json.loads((Path(directory) / "CORPUS.json").read_text())
      self.assertEqual(written, manifest)
      self.assertEqual(written["schema"], corpus.SCHEMA)

  def test_the_command_line_builds_a_seed_range(self):
    with tempfile.TemporaryDirectory() as directory:
      code = corpus.main(
        [
          "--exporter",
          str(EXPORTER),
          "--seed-start",
          "1",
          "--seed-count",
          "3",
          "--shape",
          "2,8",
          "--out",
          directory,
        ]
      )
      self.assertEqual(code, 0)
      written = json.loads((Path(directory) / "CORPUS.json").read_text())
      self.assertEqual([entry["seed"] for entry in written["entries"]], [1, 2, 3])


if __name__ == "__main__":
  unittest.main()
