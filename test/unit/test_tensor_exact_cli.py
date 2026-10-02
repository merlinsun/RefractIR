"""Exercise the composed exact case with the built RefractIR command-line tools."""

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

from test.lib import build_tensor_exact_loop_case, tensor_case, tensor_exact_f32

ROOT = Path(__file__).resolve().parents[2]


def _tool(environment_name, binary_name):
  value = os.environ.get(environment_name)
  return Path(value).resolve() if value else ROOT / "build/bin" / binary_name


SYMIRI = _tool("REFRACTIR_SYMIRI", "symiri")
SYMIRC = _tool("REFRACTIR_SYMIRC", "symirc")
SYMIR_SOLVE = _tool("REFRACTIR_SYMIR_SOLVE", "symirsolve")
TOOLS_AVAILABLE = all(path.is_file() for path in (SYMIRI, SYMIRC, SYMIR_SOLVE))


@unittest.skipUnless(
  TOOLS_AVAILABLE, "built symiri, symirc, and symirsolve are required"
)
class ExactComposedCliTests(unittest.TestCase):
  def test_solver_model_replay_validation_and_compilation(self):
    with tempfile.TemporaryDirectory() as root:
      case_dir = build_tensor_exact_loop_case.build_case(
        Path(root) / "case",
        symiri=SYMIRI,
        symirc=SYMIRC,
        solver=SYMIR_SOLVE,
      )
      loaded = tensor_case.load_case(case_dir, symiri=SYMIRI)
      model = json.loads((case_dir / "selection-model.json").read_text())
      witness = model["@kernel"]["%?x"]

      self.assertEqual(loaded.inputs, struct.pack("<4f", *witness))
      _, independently_expected = tensor_exact_f32.certify(
        loaded.program.decode("utf-8"), loaded.inputs, (0, 8)
      )
      self.assertEqual(loaded.expected, independently_expected)
      self.assertEqual(struct.unpack("<4f", loaded.expected)[0], 63.0)
      self.assertIn("selection.solved.sir", loaded.manifest["files"])
      self.assertTrue((case_dir / "selection.solved.sir").is_file())
      self.assertTrue((case_dir / "kernel.py").is_file())
      attempt = json.loads((case_dir / "attempt.json").read_text())
      self.assertEqual(attempt["status"], "accepted")
      self.assertEqual(
        [stage["stage"] for stage in attempt["stages"]],
        ["solve", "reference", "compile"],
      )
      self.assertTrue(all(stage["status"] == "pass" for stage in attempt["stages"]))


if __name__ == "__main__":
  unittest.main()
