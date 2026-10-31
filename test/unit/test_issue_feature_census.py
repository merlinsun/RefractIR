"""Feature census of Triton issue reproducers (plan L0)."""

import importlib.util
import json
import tempfile
import textwrap
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
  "issue_feature_census",
  _ROOT / "test" / "tensor" / "campaigns" / "issue_feature_census.py",
)
census = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(census)


def source(text):
  return textwrap.dedent(text).lstrip()


DOT_CONVERT = source(
  """
  import torch
  import triton
  import triton.language as tl

  @triton.jit
  def kernel(a_ptr, b_ptr, c_ptr):
      a = tl.load(a_ptr).to(tl.float16)
      c = tl.dot(a, tl.load(b_ptr))
      c = tl.cumsum(c, axis=0)
      tl.atomic_add(c_ptr, tl.where(c > 0, c, 0.0))
      for i in tl.range(0, 4, loop_unroll_factor=2):
          pass

  x = torch.ones(4).to(torch.float16)
  """
)

NESTED = source(
  """
  import triton
  import triton.language as tl

  @triton.jit
  def kernel(x_ptr, n):
      for i in range(n):
          for j in range(n):
              tl.store(x_ptr + i, j)
      while n > 0:
          n -= 1
      if n == 0:
          tl.store(x_ptr, 0)

  def host():
      for i in range(4):
          for j in range(4):
              pass
  """
)

GLUON = source(
  """
  from triton.experimental import gluon
  from triton.experimental.gluon import language as gl

  @gluon.jit
  def kernel(x_ptr):
      layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
      gl.store(x_ptr, gl.arange(0, 128, layout=layout))
  """
)


class FeatureTests(unittest.TestCase):
  def test_calls_and_conversions_are_detected(self):
    found = census.features(DOT_CONVERT)
    for name in ("dot", "conversion", "fp16", "scan", "atomic", "where", "range_hint"):
      self.assertIn(name, found)
    self.assertNotIn("gluon", found)
    self.assertNotIn("nested_control", found)

  def test_control_flow_counts_only_inside_kernels(self):
    found = census.features(NESTED)
    self.assertIn("nested_loop", found)
    self.assertIn("while", found)
    self.assertIn("if", found)
    self.assertIn("nested_control", found)
    host_only = "def host():" + NESTED.split("def host():")[1]
    self.assertNotIn("nested_loop", census.features(host_only))

  def test_gluon_kernels_are_detected(self):
    found = census.features(GLUON)
    self.assertIn("gluon", found)
    self.assertNotIn("dot", found)

  def test_any_to_inside_a_kernel_is_a_conversion(self):
    kernel = source(
      """
      @triton.jit
      def kernel(x_ptr, y_ptr):
          tl.store(y_ptr, tl.load(x_ptr).to(y_ptr.dtype.element_ty))
      """
    )
    self.assertIn("conversion", census.features(kernel))
    self.assertNotIn("conversion", census.features("x = t.to(device)\n"))

  def test_block_pointers_are_not_tensor_descriptors(self):
    block = "p = tl.make_block_ptr(base=x, shape=(4,), strides=(1,))\n"
    host = "d = TensorDescriptor.from_tensor(x, [16, 16])\n"
    self.assertEqual(
      census.features(block) & {"block_ptr", "descriptor"}, {"block_ptr"}
    )
    self.assertIn("descriptor", census.features(host))

  def test_unparseable_source_falls_back_to_names(self):
    found = census.features("@triton.jit\ndef k(x):\n    y = tl.dot(x, x\n    ...")
    self.assertIn("dot", found)
    self.assertNotIn("nested_loop", found)


class SymptomTests(unittest.TestCase):
  def test_label_words_come_before_the_file_check(self):
    cases = [
      ("fdiv ieee rounding", "pass\n", "wrong_result"),
      ("argmax nan interpreter", "pass\n", "wrong_result"),
      ("compile assert", "assert x == 1\n", "crash"),
      ("autotune hang", "pass\n", "hang"),
      ("transpose", "torch.testing.assert_close(a, b)\n", "wrong_result"),
      ("transpose", "assert torch.allclose(a, b)\n", "wrong_result"),
      ("transpose", "print(a)\n", "crash"),
      ("matmul slow performance", "pass\n", "other"),
    ]
    for label, text, expected in cases:
      with self.subTest(label=label, text=text):
        row = census.symptom(label, text)
        self.assertEqual(row["symptom"], expected)
    both = census.symptom("compile assert", "assert x == 1\n")
    self.assertEqual(both["label_signal"], "crash")
    self.assertTrue(both["checks_value"])


class CensusTests(unittest.TestCase):
  def test_rows_carry_manifest_fields_and_excerpt_marks(self):
    with tempfile.TemporaryDirectory() as name:
      seeds = Path(name)
      (seeds / "issue_1_dot_wrong.py").write_text(DOT_CONVERT)
      (seeds / "issue_2_loop_excerpt.py").write_text(
        "# Extracted from issue-body code blocks; may be a fragment\n" + NESTED
      )
      manifest = {
        "seeds": [
          {
            "issue": 1,
            "seed_path": "issue_1_dot_wrong.py",
            "status": "verified_merged_fix",
            "title": "dot wrong result",
            "fix_status": "pr_reference_unverified",
          },
          {"issue": 2, "seed_path": "issue_2_loop_excerpt.py", "status": "x"},
        ]
      }
      (seeds / "manifest.json").write_text(json.dumps(manifest))
      rows = census.census(seeds)
    self.assertEqual([row["issue"] for row in rows], [1, 2])
    self.assertEqual(rows[0]["label"], "dot wrong result")
    self.assertEqual(rows[0]["symptom"], "wrong_result")
    self.assertEqual(rows[0]["fix_status"], "pr_reference_unverified")
    self.assertFalse(rows[0]["excerpt"])
    self.assertIsNone(rows[1]["label"])
    self.assertTrue(rows[1]["excerpt"])
    self.assertIn("nested_control", rows[1]["features"])


class RankingTests(unittest.TestCase):
  @staticmethod
  def row(features, symptom):
    return {"features": features, "symptom": symptom}

  def test_d7_ranks_by_wrong_results_then_density_and_promotes(self):
    rows = (
      [self.row(["dot"], "wrong_result")] * 4
      + [self.row(["dot"], "crash")] * 4
      + [self.row(["conversion"], "wrong_result")] * 4
      + [self.row(["scan"], "wrong_result")] * 2
      + [self.row(["nested_control"], "wrong_result")] * 1
      + [self.row(["gluon"], "wrong_result")] * 1
      + [self.row(["gluon"], "crash")] * 2
      + [self.row(["where"], "wrong_result")] * 3
      + [self.row(["atomic"], "wrong_result")] * 1
    )
    result = census.rank(rows)
    self.assertEqual(
      [entry["feature"] for entry in result["order"]],
      ["conversion", "dot", "where", "scan", "nested_control", "gluon"],
    )
    dot = next(entry for entry in result["order"] if entry["feature"] == "dot")
    self.assertEqual((dot["files"], dot["wrong_result"], dot["crash"]), (8, 4, 4))
    self.assertAlmostEqual(dot["density"], 0.5)
    self.assertEqual(result["promoted"], ["where"])
    atomic = next(e for e in result["outside"] if e["feature"] == "atomic")
    self.assertEqual(atomic["wrong_result"], 1)

  def test_math_is_tallied_but_never_promoted(self):
    rows = [self.row([name], "wrong_result") for name in CANDIDATES_ONCE]
    rows += [self.row(["math"], "wrong_result")] * 5
    result = census.rank(rows)
    self.assertEqual(result["promoted"], [])
    self.assertNotIn("math", [entry["feature"] for entry in result["outside"]])


CANDIDATES_ONCE = ("gluon", "conversion", "dot", "scan", "nested_control")


if __name__ == "__main__":
  unittest.main()
