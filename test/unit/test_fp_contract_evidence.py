"""PTX evidence for the floating-point comparison contract (plan W3); no GPU."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from test.unit.tensor_support import SYMIRC

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
  "fp_contract_evidence",
  _ROOT / "test" / "tensor" / "campaigns" / "fp_contract_evidence.py",
)
evidence = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evidence)


class ClassifyTests(unittest.TestCase):
  def test_a_full_range_division_is_approximate_and_a_rounded_one_is_not(self):
    self.assertEqual(
      evidence.classify_ptx("div.full.f32 %f3, %f1, %f2;"),
      {
        "instructions": ["div.full.f32"],
        "approximate": True,
        "ftz": False,
        "fma": False,
        "specified": False,
      },
    )
    self.assertFalse(evidence.classify_ptx("div.rn.f32 %f3, %f1, %f2;")["approximate"])

  def test_an_approximate_flushing_exponential_is_flagged_twice(self):
    found = evidence.classify_ptx(
      "mul.f32 %f2, %f1, 0f3FB8AA3B;\nex2.approx.ftz.f32 %f3, %f2;"
    )
    self.assertEqual(found["instructions"], ["ex2.approx.ftz.f32", "mul.f32"])
    self.assertTrue(found["approximate"] and found["ftz"])

  def test_rounded_arithmetic_has_no_flags_and_a_fused_one_is_seen(self):
    self.assertEqual(
      evidence.classify_ptx("add.rn.f32 %f3, %f1, %f2;\nmax.f32 %f4, %f3, %f1;"),
      {
        "instructions": ["add.rn.f32", "max.f32"],
        "approximate": False,
        "ftz": False,
        "fma": False,
        "specified": True,
      },
    )
    self.assertTrue(evidence.classify_ptx("fma.rn.f32 %f4, %f1, %f2, %f3;")["fma"])

  def test_only_correctly_rounded_unfused_instructions_are_specified(self):
    self.assertTrue(evidence.classify_ptx("add.rn.f32 %f3, %f1, %f2;")["specified"])
    self.assertTrue(evidence.classify_ptx("div.rn.f32 %f3, %f1, %f2;")["specified"])
    self.assertFalse(evidence.classify_ptx("div.full.f32 %f3, %f1, %f2;")["specified"])
    # The server's rem kernel: a - trunc(a / b) * b, not one IEEE operation.
    rem = "div.rn.f32 %f3, %f1, %f2;\ncvt.rzi.f32.f32 %f4, %f3;\nneg.f32 %f5, %f4;\nfma.rn.f32 %f6, %f5, %f2, %f1;"
    self.assertFalse(evidence.classify_ptx(rem)["specified"])

  def test_non_float_instructions_are_ignored(self):
    self.assertEqual(evidence.classify_ptx("add.s32 %r1, %r2, 1;")["instructions"], [])


@unittest.skipUnless(SYMIRC.is_file(), "symirc is unavailable")
class EvidenceRunTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.out = Path(temporary.name) / "evidence"
    self.compiled = []

  def fake_compile(self, ptx_by_op, fail=()):
    def compile_matrix(kernel_path, *, warps, stages, out):
      op = Path(out).name
      self.compiled.append((op, Path(kernel_path).read_text()))
      if op in fail:
        return {"settings": [{"status": "failed", "error": "PassManager::run failed"}]}
      directory = Path(out) / f"warps-{warps[0]}-stages-{stages[0]}"
      directory.mkdir(parents=True)
      (directory / "ptx.ir").write_text(ptx_by_op.get(op, "add.rn.f32 %f1, %f2, %f3;"))
      return {"settings": [{"status": "compiled"}]}

    return compile_matrix

  def test_every_operation_is_emitted_compiled_and_classified(self):
    table = evidence.collect(
      self.out,
      symirc=SYMIRC,
      compile_matrix=self.fake_compile({"div": "div.full.f32 %f1;"}),
    )
    self.assertEqual([row["op"] for row in table["ops"]], list(evidence.OPS))
    by_op = {row["op"]: row for row in table["ops"]}
    self.assertTrue(by_op["div"]["approximate"])
    self.assertFalse(by_op["add"]["approximate"])
    kernels = dict(self.compiled)
    self.assertIn(" / ", kernels["div"])
    self.assertIn("tl.exp(", kernels["exp"])
    self.assertIn("tl.sum(", kernels["sum"])
    self.assertIn('input_precision="ieee"', kernels["dot"])
    self.assertIn("tl.cumsum(", kernels["cumsum"])
    self.assertIn("tl.float16", kernels["narrow-f16"])
    self.assertIn("tl.bfloat16", kernels["narrow-bf16"])
    self.assertEqual(json.loads((self.out / "evidence.json").read_text()), table)

  def test_a_failed_compile_is_recorded_and_the_others_still_run(self):
    table = evidence.collect(
      self.out, symirc=SYMIRC, compile_matrix=self.fake_compile({}, fail={"exp"})
    )
    by_op = {row["op"]: row for row in table["ops"]}
    self.assertEqual(by_op["exp"]["status"], "failed")
    self.assertIn("PassManager", by_op["exp"]["error"])
    self.assertEqual(by_op["rem"]["status"], "compiled")


if __name__ == "__main__":
  unittest.main()
