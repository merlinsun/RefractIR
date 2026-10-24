"""Contract tests for the diagnostic SIR observation of a target-only bundle."""

import json
import stat
import struct
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from test.lib import tensor_target_case, tensor_target_reference
from test.unit.tensor_support import SYMIRI

SOURCE = """\
fun @kernel(%x: tensor [2, 2] f32, %y: tensor [2, 2] f32) : tensor [2, 2] f32 {
  let mut %i: i32 = undef;
  let mut %acc_x: tensor [2, 2] f32 = undef;
  let mut %acc_y: tensor [2, 2] f32 = undef;
  let mut %out: tensor [2, 2] f32 = undef;
^entry:
  %acc_x = %x;
  %acc_y = %y;
  %i = 0;
  br ^header;
^header:
  br %i < 2, ^body, ^exit;
^body:
  %acc_x = tensor add %acc_x, %acc_y;
  %i = %i + 1;
  br ^header;
^exit:
  %out = tensor add %acc_x, %acc_y;
  ret %out;
}
"""

LAYOUT = {
  "shape": [2, 2],
  "element_strides": [2, 1],
  "element_offset": 0,
  "storage_span": 4,
  "axis_order": [1, 0],
}
X = (1.0, 2.0, 3.0, 4.0)
Y = (10.0, 20.0, 30.0, 40.0)
# Two iterations add y twice, then the exit adds y once more.
EXPECTED = struct.pack("<4f", *(x + 3 * y for x, y in zip(X, Y)))
LAUNCH = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}
EXECUTION = {
  "guard_envelope": {"prefix": 1, "suffix": 1},
  "gap_sentinel_by_dtype": {"f32": "3f000001", "i32": "55555555"},
  "output_canary_by_dtype": {"f32": "3f000002", "i32": "aaaaaaaa"},
  "padding_by_dtype": {"f32": "3f000003", "i32": "55555556"},
  "warp_probe": {"status": "not_probed", "num_warps": 4},
}


def build_bundle(root):
  tensor_target_case.write(
    root,
    source=SOURCE,
    inputs=[
      {"name": "%x", "dtype": "f32", **LAYOUT, "data": struct.pack("<4f", *X)},
      {"name": "%y", "dtype": "f32", **LAYOUT, "data": struct.pack("<4f", *Y)},
    ],
    output={"name": "return", "dtype": "f32", **LAYOUT},
    launch=LAUNCH,
    execution=EXECUTION,
  )
  record = json.loads((root / "case.json").read_text(encoding="utf-8"))
  return SimpleNamespace(path=root, record=record)


def stub_symiri(directory, body, exit_code):
  path = Path(directory) / "symiri"
  path.write_text(f"#!/bin/sh\n{body}\nexit {exit_code}\n", encoding="utf-8")
  path.chmod(path.stat().st_mode | stat.S_IXUSR)
  return path


class DiagnosticReferenceTests(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.root = Path(self.directory.name)
    self.case = build_bundle(self.root / "bundle")

  def test_the_inputs_bin_order_matches_the_parameter_order(self):
    if not SYMIRI.is_file():
      self.skipTest("symiri unavailable")
    observation = tensor_target_reference.observe(self.case, symiri=SYMIRI)
    self.assertEqual(observation.status, tensor_target_reference.COMPLETED)
    self.assertEqual(observation.output, EXPECTED)

  def test_a_failing_start_callback_reaps_the_reference_child(self):
    marker = self.root / "marker"
    symiri = stub_symiri(self.root, f"sleep 1.5; touch {marker}", 0)

    def on_process(event):
      if event["event"] == "start":
        raise OSError("ledger write failed")

    with self.assertRaises(OSError):
      tensor_target_reference.observe(
        self.case, symiri=symiri, on_process=on_process
      )
    time.sleep(2.0)
    self.assertFalse(marker.exists())

  def test_a_missing_interpreter_is_unavailable(self):
    observation = tensor_target_reference.observe(
      self.case, symiri=Path("/nonexistent/symiri")
    )
    self.assertEqual(observation.status, tensor_target_reference.UNAVAILABLE)
    self.assertIn("cannot start", observation.reason)

  def test_a_bundle_without_inputs_bin_is_unavailable(self):
    (self.case.path / "inputs.bin").unlink()
    observation = tensor_target_reference.observe(self.case, symiri=SYMIRI)
    self.assertEqual(observation.status, tensor_target_reference.UNAVAILABLE)
    self.assertIn("inputs.bin", observation.reason)

  def test_a_timeout_is_recorded(self):
    symiri = stub_symiri(self.root, "sleep 5", 0)
    observation = tensor_target_reference.observe(
      self.case, symiri=symiri, timeout_s=0.1
    )
    self.assertEqual(observation.status, tensor_target_reference.TIMED_OUT)

  def test_a_plain_failure_is_failed(self):
    symiri = stub_symiri(self.root, "echo 'Error: broken' >&2", 2)
    observation = tensor_target_reference.observe(self.case, symiri=symiri)
    self.assertEqual(observation.status, tensor_target_reference.FAILED)
    self.assertEqual(observation.reason, "Error: broken")

  def test_undefined_behaviour_is_reported(self):
    symiri = stub_symiri(self.root, "echo 'UB: division by zero' >&2", 5)
    observation = tensor_target_reference.observe(self.case, symiri=symiri)
    self.assertEqual(observation.status, tensor_target_reference.UB)

  def test_an_observation_marks_itself_diagnostic(self):
    observation = tensor_target_reference.observe(
      self.case, symiri=Path("/nonexistent/symiri")
    )
    record = observation.as_record()
    self.assertEqual(record["kind"], "diagnostic")
    self.assertEqual(record["path"], "refractir_interpreter")
    self.assertEqual(record["status"], "unavailable")
    self.assertIsNone(record["output_sha256"])


if __name__ == "__main__":
  unittest.main()
