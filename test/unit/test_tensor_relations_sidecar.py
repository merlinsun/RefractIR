"""Relation facts of a generated body for the sidecar (seed-export plan, E2)."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test.lib import tensor_free_form_inputs
from test.lib import tensor_relations_sidecar as sidecar
from test.unit.tensor_support import SYMIRI

EXPORTER = Path(
  os.environ.get(
    "REFRACTIR_TENSOR_FREE_FORM_EXPORT",
    Path(__file__).resolve().parents[2] / "build" / "bin" / "tensor_free_form_export",
  )
)


def program(params, body, result_shape="[2, 4]"):
  """A straight-line body: `params` maps names to shapes, `body` is SIR lines."""
  signature = ", ".join(
    f"%{name}: tensor {shape} f32" for name, shape in params.items()
  )
  locals_ = sorted({line.split("=")[0].strip() for line in body})
  declarations = "\n".join(
    f"  let mut {name}: tensor {result_shape} f32 = undef;" for name in locals_
  )
  return (
    f"fun @kernel({signature}) : tensor {result_shape} f32 {{\n{declarations}\n"
    "^entry:\n" + "\n".join(f"  {line}" for line in body) + "\n  ret %out;\n}\n"
  )


def records(params):
  return [
    {
      "name": f"%{name}",
      "shape": [int(d) for d in shape.strip("[]").split(",")],
      "dtype": "f32",
    }
    for name, shape in params.items()
  ]


def analyze(params, body, **options):
  return sidecar.analyze(program(params, body), records(params), **options)


TWO = {"a": "[2, 4]", "b": "[2, 4]"}


class SidecarTests(unittest.TestCase):
  def test_a_pointwise_body_is_linear_odd_and_permutable(self):
    facts = analyze(TWO, ["%out = tensor add %a, %b;"])
    out = facts["outputs"]["out_ptr"]
    self.assertEqual((out["degree"], out["parity"]), (1, 1))
    self.assertEqual((out["class"], out["producer_op"]), ("specified", "add"))
    self.assertEqual(out["reduced_axes"], [])
    for name in ("a_ptr", "b_ptr"):
      self.assertEqual(facts["inputs"][name]["weight"], 1)
      self.assertEqual(facts["inputs"][name]["parity"], 1)
      self.assertEqual(facts["inputs"][name]["permutable_axes"], [0, 1])
      self.assertEqual(facts["inputs"][name]["scale_axes"], [0, 1])

  def test_a_product_body_has_degree_two_and_even_parity(self):
    out = analyze(TWO, ["%out = tensor mul %a, %b;"])["outputs"]["out_ptr"]
    self.assertEqual((out["degree"], out["parity"]), (2, 0))
    self.assertEqual(out["slice_ops"], ["mul"])

  def test_a_mixed_body_solves_its_weights(self):
    params = {"a": "[2, 4]", "b": "[2, 4]", "c": "[2, 4]"}
    facts = analyze(params, ["%p = tensor mul %a, %b;", "%out = tensor add %p, %c;"])
    weights = [facts["inputs"][f"{n}_ptr"]["weight"] for n in "abc"]
    self.assertEqual(weights, [1, 1, 2])
    self.assertEqual(facts["outputs"]["out_ptr"]["degree"], 2)

  def test_extrema_keep_odd_parity_through_the_dual(self):
    facts = analyze(TWO, ["%m = tensor max %a, %b;", "%out = tensor min %m, %a;"])
    out = facts["outputs"]["out_ptr"]
    self.assertEqual((out["degree"], out["parity"], out["producer_op"]), (1, 1, "min"))
    self.assertEqual(out["class"], "specified")

  def test_a_sum_is_order_dependent_and_consumes_its_axis(self):
    facts = analyze(TWO, ["%s = tensor sum %a, 1, keep;", "%out = tensor add %s, %b;"])
    out = facts["outputs"]["out_ptr"]
    self.assertEqual(out["class"], "order_dependent")
    self.assertEqual(out["reduced_axes"], [1])
    # a's axis 1 is summed; b's axis 1 still maps lane for lane onto the output.
    self.assertEqual(facts["inputs"]["a_ptr"]["permutable_axes"], [0])
    self.assertEqual(facts["inputs"]["b_ptr"]["permutable_axes"], [0, 1])
    # The broadcast sum cannot take a factor varying along axis 1.
    self.assertEqual(facts["inputs"]["a_ptr"]["scale_axes"], [0])
    self.assertEqual(facts["inputs"]["b_ptr"]["scale_axes"], [0])

  def test_a_division_excludes_the_output(self):
    out = analyze(TWO, ["%out = tensor div %a, %b;"])["outputs"]["out_ptr"]
    self.assertEqual(out["class"], "excluded")
    self.assertEqual(out["degree"], None)


if __name__ == "__main__":
  unittest.main()


class SidecarEdgeTests(unittest.TestCase):
  def test_a_nonzero_literal_pins_its_partner_and_leaves_only_the_trivial_solution(
    self,
  ):
    facts = analyze({"a": "[2, 4]"}, ["%out = tensor add %a, 1.0;"])
    self.assertIsNone(facts["inputs"]["a_ptr"]["weight"])
    self.assertIsNone(facts["outputs"]["out_ptr"]["degree"])
    zero = analyze({"a": "[2, 4]"}, ["%out = tensor add %a, 0.0;"])
    self.assertEqual(zero["outputs"]["out_ptr"]["degree"], 1)


@unittest.skipUnless(SYMIRI.is_file() and EXPORTER.is_file(), "symiri and the exporter")
class SidecarAgainstP0Tests(unittest.TestCase):
  """The weights and degree predict P0's output under scaling, bit for bit."""

  def test_scaling_the_inputs_by_their_weights_scales_p0_by_the_degree(self):
    checked = 0
    with tempfile.TemporaryDirectory() as directory:
      for seed in range(1, 31):
        out = Path(directory) / str(seed)
        completed = subprocess.run(
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
            "--trip-count",
            "4",
            "--window",
            "2,1,5",
            "--window-buffer-rows",
            "7",
            "--trip-parameter",
            "--out",
            str(out),
          ],
          capture_output=True,
          check=False,
        )
        if completed.returncode != 0:
          continue
        region = json.loads((out / "region.json").read_text())
        source = (out / "program.sir").read_text()
        parameters = tensor_free_form_inputs.parameters(region)
        probe = tensor_free_form_inputs.window_base_proposal(
          region, group=0, column=0, trips=3
        )
        facts = sidecar.analyze(source, parameters, {"%trips": 3})
        output = facts["outputs"]["out_ptr"]
        if output["class"] != "specified" or output["degree"] is None:
          continue
        scaled = bytearray()
        offset = 0
        for record in parameters:
          count = max(1, int(np.prod(record["shape"])))
          chunk = probe.data[offset : offset + 4 * count]
          offset += 4 * count
          if record["dtype"] != "f32":
            scaled += chunk
            continue
          weight = facts["inputs"][f"{record['name'][1:]}_ptr"]["weight"]
          values = np.frombuffer(chunk, dtype="<f4") * np.float32(2.0**weight)
          scaled += values.astype("<f4").tobytes()
        base = p0(source, probe.data, directory)
        variant = p0(source, bytes(scaled), directory)
        expected = np.frombuffer(base, dtype="<f4") * np.float32(
          2.0 ** output["degree"]
        )
        self.assertEqual(variant, expected.astype("<f4").tobytes(), seed)
        checked += 1
    self.assertGreaterEqual(checked, 3, "too few specified bodies to check")


def p0(source, data, directory):
  root = Path(directory)
  (root / "p.sir").write_text(source)
  (root / "in.bin").write_bytes(data)
  subprocess.run(
    [
      str(SYMIRI),
      "--tensor-profile-version",
      "v3",
      "--entry",
      "kernel",
      "--tensor-input",
      str(root / "in.bin"),
      "--tensor-output",
      str(root / "out.bin"),
      str(root / "p.sir"),
    ],
    check=True,
    capture_output=True,
  )
  return (root / "out.bin").read_bytes()
