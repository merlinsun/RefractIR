"""Shared execution and emission helpers have one home and one set of names."""

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


class TensorExecutionTests(unittest.TestCase):
  """The runner keeps its patch points while the helpers live in one module."""

  def test_the_shared_helpers_have_one_home(self):
    from test.lib import tensor_execution

    for name in (
      "canonical_json",
      "write_atomic",
      "run_child",
      "gpu_available",
      "emit_kernel",
    ):
      self.assertTrue(callable(getattr(tensor_execution, name, None)), name)
    self.assertEqual(tensor_execution.DEFAULT_TIMEOUT_S, 60)
    self.assertEqual(tensor_execution.ROOT, Path(__file__).resolve().parents[2])

  def test_the_runner_reexports_the_shared_helpers_unchanged(self):
    from test.lib import run_triton_tests, tensor_execution

    self.assertIs(run_triton_tests._emit_kernel, tensor_execution.emit_kernel)
    self.assertIs(run_triton_tests.run_child, tensor_execution.run_child)
    self.assertIs(run_triton_tests._gpu_available, tensor_execution.gpu_available)
    self.assertIs(run_triton_tests._canonical_json, tensor_execution.canonical_json)
    self.assertIs(run_triton_tests._write_atomic, tensor_execution.write_atomic)
    for name in (
      "run_case",
      "compare_outputs",
      "probe_warp_settings",
      "_emit_kernel",
    ):
      self.assertTrue(callable(getattr(run_triton_tests, name, None)), name)

  def test_canonical_json_is_sorted_newline_terminated_text(self):
    from test.lib import tensor_execution

    self.assertEqual(
      tensor_execution.canonical_json({"b": 1, "a": [2]}),
      '{\n  "a": [\n    2\n  ],\n  "b": 1\n}\n',
    )
    self.assertEqual(
      tensor_execution.canonical_json({"name": "café"}),
      '{\n  "name": "café"\n}\n',
    )

  def test_write_atomic_replaces_contents_without_a_temporary_file(self):
    from test.lib import tensor_execution

    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "nested" / "artifact.json"
      tensor_execution.write_atomic(path, b"first")
      tensor_execution.write_atomic(path, b"second")

      self.assertEqual(path.read_bytes(), b"second")
      self.assertEqual(
        sorted(child.name for child in path.parent.iterdir()), ["artifact.json"]
      )

  def test_a_failing_start_callback_reaps_the_child(self):
    from test.lib import tensor_execution

    with tempfile.TemporaryDirectory() as directory:
      marker = Path(directory) / "marker"
      command = [
        sys.executable, "-c",
        f"import time; time.sleep(1.5); open({str(marker)!r}, 'w').write('x')",
      ]

      def on_process(event):
        if event["event"] == "start":
          raise OSError("ledger write failed")

      with self.assertRaises(OSError):
        tensor_execution.run_child(
          command, "interpret", timeout_s=5, on_process=on_process
        )
      time.sleep(2.0)
      self.assertFalse(marker.exists())

  def test_gpu_availability_is_false_without_torch(self):
    from test.lib import tensor_execution

    with mock.patch.dict(sys.modules, {"torch": None}):
      self.assertIs(tensor_execution.gpu_available(), False)


if __name__ == "__main__":
  unittest.main()
