"""Rank-0 inputs in target bundles: a scalar parameter is a one-element buffer.

Every SIR parameter reaches an emitted kernel as a pointer, a scalar included,
so a scalar input travels like any tensor input: a dense layout of shape `[]`
with one element, staged and passed by the paired worker. The trip-count
parameter of a parameter-bounded loop (plan W4a) is the first user.
"""

import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test.lib import (
  tensor_execution,
  tensor_free_form_case,
  tensor_free_form_inputs,
  tensor_target_case,
  tensor_target_reference,
)
from test.unit.tensor_support import SYMIRC, SYMIRI
from test.unit.test_tensor_parameter_loops import X, expected, loop

REGION = {
  "schema": "refractir.free-form-region.v1",
  "entry": "kernel",
  "profile": "tensor-v3",
  "dtype": "i32",
  "shape": [8],
  "num_warps": 1,
  "parameters": [
    {"name": "%x", "shape": [8], "dtype": "i32"},
    {"name": "%n", "shape": [], "dtype": "i32"},
  ],
}


def proposal(n):
  return tensor_free_form_inputs.Proposal(
    schema=tensor_free_form_inputs.WINDOW_PROBE_SCHEMA,
    pattern="trips",
    seed=0,
    names=("%x", "%n"),
    shapes=((8,), ()),
    dtype="i32",
    values=(),
    data=struct.pack("<8i", *X) + struct.pack("<i", n),
  )


class ScalarLayoutTests(unittest.TestCase):
  def test_a_scalar_has_a_one_element_dense_layout(self):
    self.assertEqual(
      tensor_free_form_case.dense_layout([]),
      {
        "shape": [],
        "element_strides": [],
        "element_offset": 0,
        "storage_span": 1,
        "axis_order": [],
      },
    )

  def test_a_region_may_declare_a_scalar_parameter(self):
    self.assertEqual(
      tensor_free_form_inputs.parameters(REGION)[1],
      {"name": "%n", "shape": [], "dtype": "i32"},
    )


@unittest.skipUnless(SYMIRI.is_file(), "symiri is unavailable")
class ScalarBundleTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name)

  def bundle(self, n, name="bundle"):
    tensor_free_form_case.build_bundle(
      self.root / name, region=REGION, source=loop(), proposal=proposal(n)
    )
    return tensor_target_case.load(self.root / name, symiri=SYMIRI)

  def test_a_scalar_input_occupies_four_bytes_of_the_bundle(self):
    case = self.bundle(3)
    record = case.record["inputs"][1]
    self.assertEqual(
      (record["name"], record["shape"], record["byte_length"]), ("%n", [], 4)
    )
    self.assertEqual(case.inputs[-4:], struct.pack("<i", 3))

  def test_p0_reads_the_scalar_from_the_bundle(self):
    for n in (0, 1, 8):
      with self.subTest(n=n):
        observation = tensor_target_reference.observe(
          self.bundle(n, f"bundle-{n}"), symiri=SYMIRI
        )
        self.assertTrue(observation.completed, observation.stderr)
        self.assertEqual(list(struct.unpack("<8i", observation.output)), expected(n))

  def test_a_scalar_layout_spanning_more_than_one_element_is_refused(self):
    self.bundle(3)
    path = self.root / "bundle" / "case.json"
    text = path.read_text(encoding="utf-8")
    import json

    manifest = json.loads(text)
    manifest["inputs"][1]["storage_span"] = 2
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with self.assertRaises(tensor_target_case.TargetCaseError):
      tensor_target_case.load(self.root / "bundle", symiri=SYMIRI)


def _triton_interpreter_available():
  try:
    import torch  # noqa: F401
    import triton  # noqa: F401
  except Exception:
    return False
  return True


@unittest.skipUnless(
  SYMIRC.is_file() and SYMIRI.is_file() and _triton_interpreter_available(),
  "native tools or the Triton interpreter are unavailable",
)
class ScalarWorkerTests(unittest.TestCase):
  def test_the_worker_stages_the_scalar_and_runs_its_trip_count(self):
    for n in (0, 1, 8):
      with self.subTest(n=n), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tensor_free_form_case.build_bundle(
          root / "bundle", region=REGION, source=loop(), proposal=proposal(n)
        )
        case = tensor_target_case.load(root / "bundle", symiri=SYMIRI)
        artifacts = root / "artifacts"
        kernel, stderr, stdout = tensor_execution.emit_kernel(case, SYMIRC, artifacts)
        self.assertIsNotNone(kernel, stdout + stderr)
        environment = {
          key: value for key, value in os.environ.items() if key != "TRITON_INTERPRET"
        }
        completed = subprocess.run(
          [
            sys.executable,
            "-m",
            "test.lib.triton_case_worker",
            "--case",
            str(root / "bundle"),
            "--artifacts-dir",
            str(artifacts),
            "--mode",
            "interpret",
          ],
          capture_output=True,
          text=True,
          check=False,
          timeout=120,
          env={**environment, "TRITON_INTERPRET": "1"},
          cwd=Path(__file__).resolve().parents[2],
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        actual = (artifacts / "actual.bin").read_bytes()
        self.assertEqual(list(struct.unpack("<8i", actual)), expected(n))


if __name__ == "__main__":
  unittest.main()
