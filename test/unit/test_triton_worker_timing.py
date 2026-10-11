"""Worker phase timing and runner artifact retention tests."""

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from test.lib import tensor_layout, triton_case_worker
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
  def __init__(
    self, output_bytes, *, compile_error=None, runtime_error=None, corrupt=None
  ):
    self.output_bytes = output_bytes
    self.compile_error = compile_error
    self.runtime_error = runtime_error
    self.corrupt = corrupt

  def warmup(self, *_args, **_kwargs):
    if self.compile_error:
      raise self.compile_error
    return SimpleNamespace(asm={})

  def __getitem__(self, _grid):
    def launch(*args, **_kwargs):
      if self.runtime_error:
        raise self.runtime_error
      args[-1].data = self.output_bytes
      if self.corrupt is not None:
        args[0].data = self.corrupt

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


class RecordingTensor(FakeTensor):
  """A stand-in device tensor that copies buffers and records its transfers.

  A real tensor copy owns its storage, so mutating one buffer must never be
  visible in another; the tape keeps every buffer handed to the device.
  """

  def __init__(self, data=b"", *, dtype=None, tape=None):
    super().__init__(data)
    self.dtype = dtype
    self.tape = tape

  def clone(self):
    return RecordingTensor(self.data, dtype=self.dtype, tape=self.tape)

  def to(self, _device):
    device = RecordingTensor(self.data, dtype=self.dtype, tape=self.tape)
    if self.tape is not None:
      self.tape.append(device)
    return device


def recording_torch(tape):
  return SimpleNamespace(
    cuda=SimpleNamespace(
      is_available=lambda: True,
      set_device=lambda _device: None,
      synchronize=lambda: None,
    ),
    device=lambda name: name,
    int32="int32",
    float32="float32",
    uint8="uint8",
    frombuffer=lambda data, dtype: RecordingTensor(data, dtype=dtype, tape=tape),
    isfinite=lambda _tensor: FakeTensor(),
    full=lambda shape, _value, **_kwargs: FakeTensor(bytes(4 * shape[0])),
  )


def scatter(logical, layout, *, padding, envelope, element_size=4):
  """Place logical values in the enveloped slots the device writes them to."""
  addresses = tensor_layout.logical_addresses(layout)
  slots = envelope.prefix + layout["storage_span"] + envelope.suffix
  physical = bytearray(padding * slots)
  for index, address in enumerate(addresses):
    target = (envelope.prefix + address) * element_size
    physical[target : target + element_size] = logical[
      index * element_size : (index + 1) * element_size
    ]
  return bytes(physical)


class TritonWorkerGuardEnvelopeTests(unittest.TestCase):
  """The device route allocates and validates allocated guard envelopes."""

  prefix = 2
  suffix = 3
  inputs = struct.pack("<8f", *range(8))
  expected = inputs
  canary = bytes.fromhex("aaaaaaaa")
  gap_sentinel = bytes.fromhex("3f000001")
  padding = bytes.fromhex("55555555")
  layout = {
    "shape": [2, 4],
    "element_strides": [6, 1],
    "element_offset": 1,
    "storage_span": 11,
    "axis_order": [1, 0],
  }
  envelope = tensor_layout.GuardEnvelope(prefix=2, suffix=3)

  def setUp(self):
    self.device = []
    self.failure = None
    self.artifacts = None

  def case_dir(self, root, *, padding=True, envelope=(None, None), dense=None):
    """Write one case manifest; `envelope=(0, 0)` declares an empty one."""
    case = make_case(root / "case")
    manifest_path = case / "case.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = "refractir.tensor-case.v2"
    manifest["profile"] = "tensor-v2"
    for record in (*manifest["inputs"], manifest["output"]):
      record.update(dense or self.layout)
      record["tile_shape"] = list(record["shape"])
      record["tile_origin"] = [0, 0]
    manifest["execution"] = {
      "warp_probe": {"status": "not_probed", "num_warps": 4},
      "gap_sentinel_by_dtype": {
        "i32": self.gap_sentinel.hex(),
        "f32": self.gap_sentinel.hex(),
      },
      "output_canary_by_dtype": {"i32": self.canary.hex(), "f32": self.canary.hex()},
      "guard_envelope": {
        "prefix": self.prefix if envelope[0] is None else envelope[0],
        "suffix": self.suffix if envelope[1] is None else envelope[1],
      },
    }
    if padding:
      manifest["execution"]["padding_by_dtype"] = {
        "i32": self.padding.hex(),
        "f32": self.padding.hex(),
      }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return case

  def run_case(self, case, kernel):
    """Run one prepared case, keeping the device buffers and any refusal."""
    artifact_dir = case / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "kernel.py").write_text("# fake kernel\n", encoding="utf-8")
    with (
      mock.patch.dict("sys.modules", {"torch": recording_torch(self.device)}),
      mock.patch.object(
        triton_case_worker.runpy, "run_path", return_value={"kernel": kernel}
      ),
    ):
      try:
        triton_case_worker.execute(case, "compiled", artifact_dir)
      except (RuntimeError, tensor_layout.LayoutError) as error:
        self.failure = error
    self.artifacts = artifact_dir

  def run_worker(self, root, kernel, *, padding=True):
    """Prepare and run one guarded case."""
    self.run_case(self.case_dir(root, padding=padding), kernel)

  def device_bytes(self, index):
    """Return one buffer handed to the device, refusing to pass vacuously."""
    self.assertTrue(
      len(self.device) > index,
      f"the worker transferred {len(self.device)} buffers, expected {index + 1}",
    )
    return self.device[index].data

  def refusal(self):
    """Return the worker's refusal, refusing to pass vacuously."""
    self.assertIsNotNone(self.failure, "the worker accepted a case it must refuse")
    return str(self.failure)

  def staging(self):
    path = self.artifacts / "worker-staging.json"
    self.assertTrue(path.is_file(), "the worker must record what it staged")
    return json.loads(path.read_text(encoding="utf-8"))

  def test_inputs_reach_the_device_with_allocated_guards(self):
    with tempfile.TemporaryDirectory() as directory:
      self.run_worker(
        Path(directory),
        FakeKernel(
          scatter(
            self.inputs, self.layout, padding=self.padding, envelope=self.envelope
          )
        ),
      )

    input_buffer = self.device_bytes(0)
    self.assertEqual(
      len(input_buffer),
      (self.prefix + self.layout["storage_span"] + self.suffix) * 4,
      "the device input must cover both guards",
    )
    self.assertEqual(input_buffer[: self.prefix * 4], self.padding * self.prefix)
    self.assertEqual(input_buffer[-self.suffix * 4 :], self.padding * self.suffix)
    self.assertEqual(
      tensor_layout.gather_envelope(
        input_buffer, self.layout, element_size=4, envelope=self.envelope
      ),
      self.inputs,
    )

  def test_output_allocation_is_canary_filled_before_the_run(self):
    with tempfile.TemporaryDirectory() as directory:
      self.run_worker(
        Path(directory),
        FakeKernel(
          scatter(
            self.expected, self.layout, padding=self.canary, envelope=self.envelope
          )
        ),
      )
      actual = (self.artifacts / "actual.bin").read_bytes()

    output_buffer = self.device_bytes(1)
    self.assertEqual(
      len(output_buffer),
      (self.prefix + self.layout["storage_span"] + self.suffix) * 4,
    )
    self.assertEqual(output_buffer[: self.prefix * 4], self.canary * self.prefix)
    self.assertEqual(output_buffer[-self.suffix * 4 :], self.canary * self.suffix)
    self.assertIsNone(self.failure, "a guarded case with canaries must be accepted")
    self.assertEqual(actual, self.expected)

  def test_guard_gap_and_input_writes_are_named_in_the_rejection(self):
    corrupt = bytearray(
      scatter(self.expected, self.layout, padding=self.canary, envelope=self.envelope)
    )
    corrupt[0:4] = b"\xde\xad\xbe\xef"
    corrupt[(self.prefix + 5) * 4 : (self.prefix + 6) * 4] = b"\xde\xad\xbe\xef"
    corrupt[(self.prefix + 1) * 4 : (self.prefix + 2) * 4] = b"\xde\xad\xbe\xef"
    accepted = scatter(
      self.expected, self.layout, padding=self.canary, envelope=self.envelope
    )
    with tempfile.TemporaryDirectory() as directory:
      self.run_worker(Path(directory), FakeKernel(accepted, corrupt=bytes(corrupt)))

    message = self.refusal()
    self.assertIn("prefix-guard-0", message)
    self.assertIn("gap-5", message)
    self.assertIn("input-0", message)

  def test_staging_record_reports_the_guards_and_padding_used(self):
    with tempfile.TemporaryDirectory() as directory:
      self.run_worker(
        Path(directory),
        FakeKernel(
          scatter(
            self.inputs, self.layout, padding=self.padding, envelope=self.envelope
          )
        ),
      )
      staging = self.staging()

    self.assertEqual(
      staging["guard_envelope"], {"prefix": self.prefix, "suffix": self.suffix}
    )
    self.assertEqual(staging["padding_by_dtype"]["f32"], self.padding.hex())

  def test_envelope_padding_falls_back_to_the_gap_sentinel(self):
    with tempfile.TemporaryDirectory() as directory:
      self.run_worker(
        Path(directory),
        FakeKernel(
          scatter(
            self.inputs,
            self.layout,
            padding=self.gap_sentinel,
            envelope=self.envelope,
          )
        ),
        padding=False,
      )
      staging = self.staging()

    self.assertEqual(staging["padding_by_dtype"]["f32"], self.gap_sentinel.hex())

  def test_zero_guards_stage_the_same_bytes_as_no_envelope(self):
    """A declared but empty envelope must reduce to the plain allocation."""
    dense = {
      "shape": [2, 4],
      "element_strides": [4, 1],
      "element_offset": 0,
      "storage_span": 8,
      "axis_order": [1, 0],
    }
    payload = tensor_layout.stage_buffer(
      self.expected, dense, element_size=4, gap_bytes=self.gap_sentinel
    )
    with tempfile.TemporaryDirectory() as directory:
      guarded = self.case_dir(Path(directory) / "guarded", envelope=(0, 0), dense=dense)
      self.run_case(guarded, FakeKernel(payload))
      guarded_bytes = (self.artifacts / "actual.bin").read_bytes()
      guarded_staging = self.staging()
      plain = self.case_dir(Path(directory) / "plain", dense=dense)
      manifest_path = plain / "case.json"
      manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
      del manifest["execution"]["guard_envelope"]
      manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
      self.run_case(plain, FakeKernel(payload))
      plain_bytes = (self.artifacts / "actual.bin").read_bytes()

    self.assertEqual(guarded_bytes, plain_bytes)
    self.assertEqual(guarded_staging["guard_envelope"], {"prefix": 0, "suffix": 0})
    self.assertEqual(guarded_bytes, self.expected)


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

  def test_pass_dump_capture_is_off_by_default(self):
    with (
      tempfile.TemporaryDirectory() as directory,
      mock.patch.dict("os.environ", {}, clear=True),
    ):
      requested = triton_case_worker.configure_pass_dumps(Path(directory))

      self.assertIsNone(requested)
      self.assertNotIn("MLIR_DUMP_PATH", os.environ)

  def test_pass_dump_capture_routes_one_file_per_compilation(self):
    with (
      tempfile.TemporaryDirectory() as directory,
      mock.patch.dict("os.environ", {"REFRACTIR_TRITON_PASS_DUMP": "1"}, clear=True),
    ):
      artifact_dir = Path(directory)
      requested = triton_case_worker.configure_pass_dumps(artifact_dir)

      self.assertEqual(requested, artifact_dir / "compiler-pass-dump.txt")
      self.assertEqual(os.environ["MLIR_ENABLE_DUMP"], "1")
      self.assertEqual(os.environ["MLIR_DUMP_PATH"], str(requested))
      self.assertIn("pass", requested.name)
      self.assertIn("dump", requested.name)

  def test_compiled_run_exposes_the_dump_path_during_compilation(self):
    observed = []
    kernel = FakeKernel(struct.pack("<8f", *range(8)))
    original_warmup = kernel.warmup

    def recording_warmup(*args, **kwargs):
      observed.append(os.environ.get("MLIR_DUMP_PATH"))
      return original_warmup(*args, **kwargs)

    kernel.warmup = recording_warmup
    with (
      tempfile.TemporaryDirectory() as directory,
      mock.patch.dict("os.environ", {"REFRACTIR_TRITON_PASS_DUMP": "1"}, clear=True),
    ):
      artifact_dir = self.run_worker(Path(directory), kernel)

    self.assertEqual(observed, [str(artifact_dir / "compiler-pass-dump.txt")])

  def test_interpreter_run_does_not_request_pass_dumps(self):
    with (
      tempfile.TemporaryDirectory() as directory,
      mock.patch.dict("os.environ", {"REFRACTIR_TRITON_PASS_DUMP": "1"}, clear=True),
    ):
      self.run_worker(
        Path(directory), FakeKernel(struct.pack("<8f", *range(8))), mode="interpret"
      )

      self.assertNotIn("MLIR_DUMP_PATH", os.environ)

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
