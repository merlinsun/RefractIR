"""Worker phase timing and runner artifact retention tests."""

import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from test.lib import triton_case_worker
from test.unit.tensor_support import SYMIRI
from test.unit.test_tensor_runner import available_environment, make_case


class FakeTensor:
  def __init__(self, data=b""):
    self.data = bytes(data)

  def clone(self):
    return FakeTensor(self.data)

  def to(self, _device):
    return self

  def all(self):
    return self

  def __bool__(self):
    return True

  def detach(self):
    return self

  def contiguous(self):
    return self

  def view(self, _dtype):
    return self

  def numpy(self):
    return self

  def tobytes(self):
    return self.data


class FakeKernel:
  def __init__(self, output_bytes, *, compile_error=None, runtime_error=None):
    self.output_bytes = output_bytes
    self.compile_error = compile_error
    self.runtime_error = runtime_error

  def warmup(self, *_args, **_kwargs):
    if self.compile_error:
      raise self.compile_error
    return SimpleNamespace(asm={})

  def __getitem__(self, _grid):
    def launch(*args, **_kwargs):
      if self.runtime_error:
        raise self.runtime_error
      args[-1].data = self.output_bytes

    return launch


def fake_torch():
  return SimpleNamespace(
    cuda=SimpleNamespace(
      is_available=lambda: True,
      set_device=lambda _device: None,
      synchronize=lambda: None,
    ),
    device=lambda name: name,
    int32=object(),
    float32=object(),
    uint8=object(),
    frombuffer=lambda data, dtype: FakeTensor(data),
    isfinite=lambda _tensor: FakeTensor(),
    full=lambda shape, _value, **_kwargs: FakeTensor(bytes(4 * shape[0])),
  )


class TritonWorkerTimingTests(unittest.TestCase):
  def run_worker(self, root, kernel, *, mode="compiled"):
    case = make_case(root / "case")
    artifact_dir = case / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "kernel.py").write_text("# fake kernel\n", encoding="utf-8")
    with (
      mock.patch.dict("sys.modules", {"torch": fake_torch()}),
      mock.patch.object(
        triton_case_worker.runpy, "run_path", return_value={"kernel": kernel}
      ),
    ):
      triton_case_worker.execute(case, mode, artifact_dir)
    return artifact_dir

  def test_success_records_compilation_and_gpu_execution_durations(self):
    expected = struct.pack("<8f", *range(8))
    with tempfile.TemporaryDirectory() as directory:
      artifact_dir = self.run_worker(Path(directory), FakeKernel(expected))
      timing_path = artifact_dir / "worker-timing.json"
      self.assertTrue(timing_path.is_file())
      timing = json.loads(timing_path.read_text(encoding="utf-8"))

    self.assertGreaterEqual(timing["compilation_s"], 0)
    self.assertGreaterEqual(timing["gpu_execution_s"], 0)

  def test_success_records_input_staging_and_output_materialization(self):
    expected = struct.pack("<8f", *range(8))
    with tempfile.TemporaryDirectory() as directory:
      artifact_dir = self.run_worker(Path(directory), FakeKernel(expected))
      timing_path = artifact_dir / "worker-timing.json"
      self.assertTrue(timing_path.is_file())
      timing = json.loads(timing_path.read_text())

    self.assertIn("input_staging_s", timing)
    self.assertIn("output_materialization_s", timing)
    self.assertGreaterEqual(timing["worker_total_s"], timing["input_staging_s"])

  def test_compile_failure_retains_partial_timing_and_phase(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      with self.assertRaisesRegex(RuntimeError, "compile failed"):
        self.run_worker(
          root, FakeKernel(b"", compile_error=RuntimeError("compile failed"))
        )
      artifacts = root / "case" / "artifacts"
      self.assertTrue((artifacts / "worker-timing.json").is_file())
      timing = json.loads((artifacts / "worker-timing.json").read_text())
      phase = json.loads((artifacts / "worker-phase.json").read_text())

    self.assertIn("compilation_s", timing)
    self.assertEqual(phase["phase"], "compilation")

  def test_runtime_failure_retains_completed_compilation_timing(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      with self.assertRaisesRegex(RuntimeError, "launch failed"):
        self.run_worker(
          root, FakeKernel(b"", runtime_error=RuntimeError("launch failed"))
        )
      artifacts = root / "case" / "artifacts"
      self.assertTrue((artifacts / "worker-timing.json").is_file())
      timing = json.loads((artifacts / "worker-timing.json").read_text())
      phase = json.loads((artifacts / "worker-phase.json").read_text())

    self.assertIn("compilation_s", timing)
    self.assertIn("gpu_execution_s", timing)
    self.assertEqual(phase["phase"], "runtime")

  def test_runner_lists_worker_timing_artifact(self):
    from test.lib import run_triton_tests
    from test.lib.tensor_case import TensorCase

    with tempfile.TemporaryDirectory() as directory:
      case = make_case(Path(directory) / "case")
      manifest = json.loads((case / "case.json").read_text())
      case_data = TensorCase(
        case,
        manifest,
        (case / "program.sir").read_bytes(),
        (case / "inputs.bin").read_bytes(),
        (case / "expected.bin").read_bytes(),
        {},
      )
      artifact_dir = case / "runs" / "compiled"
      actual = artifact_dir / "actual.bin"
      timing = artifact_dir / "worker-timing.json"

      def fake_emit(_case, _symirc, output_dir):
        kernel = Path(output_dir) / "kernel.py"
        kernel.write_text("# emitted\n", encoding="utf-8")
        return kernel, "", ""

      worker_code = (
        "from pathlib import Path; import json; "
        f"Path({str(actual)!r}).write_bytes(Path({str(case / 'expected.bin')!r}).read_bytes()); "
        f"Path({str(timing)!r}).write_text(json.dumps({{'worker_total_s': 1.0}}))"
      )
      with (
        mock.patch.object(run_triton_tests, "_emit_kernel", side_effect=fake_emit),
        mock.patch.object(
          run_triton_tests.tensor_case, "load_case", return_value=case_data
        ),
      ):
        report = run_triton_tests.run_case(
          case,
          "compiled",
          worker_command=["python3", "-c", worker_code],
          gpu_probe=lambda: True,
          environment_probe=available_environment,
          symiri=SYMIRI,
          symirc=Path(__file__).resolve(),
        )

    self.assertEqual(report["status"], "pass")
    self.assertIn("runs/compiled/worker-timing.json", report["artifacts"])

  def test_interpreter_timing_identifies_cpu_execution(self):
    expected = struct.pack("<8f", *range(8))
    with tempfile.TemporaryDirectory() as directory:
      artifact_dir = self.run_worker(
        Path(directory), FakeKernel(expected), mode="interpret"
      )
      timing_path = artifact_dir / "worker-timing.json"
      self.assertTrue(timing_path.is_file())
      timing = json.loads(timing_path.read_text())

    self.assertEqual(timing["execution_kind"], "interpreter")
    self.assertEqual(timing["compilation_s"], 0)


if __name__ == "__main__":
  unittest.main()
