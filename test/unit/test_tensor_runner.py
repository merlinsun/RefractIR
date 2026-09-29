"""Validation and subprocess tests for the isolated Triton case runner."""

import contextlib
import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from test.unit.tensor_support import SYMIRI

try:
  from test.lib import run_triton_tests, tensor_case
except ImportError:
  run_triton_tests = None
  tensor_case = None


def fnv1a64(data):
  value = 14695981039346656037
  for byte in data:
    value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
  return f"{value:016x}"


def make_case(root, input_shape=(2, 4), manifest_shape=None):
  root = Path(root)
  root.mkdir(parents=True, exist_ok=True)
  source = f"""\
fun @kernel(%x: tensor [{input_shape[0]}, {input_shape[1]}] f32) : tensor [{input_shape[0]}, {input_shape[1]}] f32 {{
  let mut %out: tensor [{input_shape[0]}, {input_shape[1]}] f32 = undef;
^entry:
  %out = %x;
  ret %out;
}}
""".encode()
  inputs = struct.pack("<8f", *range(8))
  expected = inputs
  kernel = b"def kernel(*args):\n  pass\n"
  (root / "program.sir").write_bytes(source)
  (root / "inputs.bin").write_bytes(inputs)
  (root / "expected.bin").write_bytes(expected)
  (root / "kernel.py").write_bytes(kernel)

  shape = list(manifest_shape or input_shape)
  strides = []
  stride = 1
  for dimension in reversed(shape):
    strides.insert(0, stride)
    stride *= dimension
  files = {}
  for name, data in (
    ("program.sir", source),
    ("inputs.bin", inputs),
    ("expected.bin", expected),
    ("kernel.py", kernel),
  ):
    files[name] = {"bytes": len(data), "fnv1a64": fnv1a64(data)}
  manifest = {
    "schema": "refractir.tensor-case.v1",
    "profile": "tensor-v1",
    "entry": "kernel",
    "seed": 17,
    "generator_revision": "test",
    "files": files,
    "inputs": [
      {
        "name": "x",
        "dtype": "f32",
        "shape": shape,
        "file": "inputs.bin",
        "byte_offset": 0,
        "byte_length": 32,
        "element_strides": strides,
        "element_offset": 0,
        "tile_shape": shape,
        "tile_origin": [0, 0],
      }
    ],
    "output": {
      "name": "return",
      "dtype": "f32",
      "shape": shape,
      "file": "expected.bin",
      "byte_offset": 0,
      "byte_length": 32,
      "element_strides": strides,
      "element_offset": 0,
      "tile_shape": shape,
      "tile_origin": [0, 0],
    },
    "grid": [1, 1, 1],
    "softmax_lowering": "direct",
    "launch": {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False},
    "property": {"id": "copy", "source": "true"},
    "evidence": {"kind": "construction", "status": "validated", "artifacts": []},
    "comparison": {
      "mode": "exact",
      "policy": "bit-exact",
      "atol": [0, 1],
      "rtol": [0, 1],
    },
    "reference": {"precisions": [], "stable": True},
    "environment": None,
  }
  (root / "case.json").write_text(json.dumps(manifest), encoding="utf-8")
  return root


def make_v2_case(
  root,
  *,
  input_strides=(1, 2),
  input_offset=1,
  input_span=10,
  input_order=(0, 1),
  output_strides=(4, 1),
  output_offset=0,
  output_span=8,
  output_order=(1, 0),
):
  root = make_case(root)
  manifest_path = root / "case.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  manifest["schema"] = "refractir.tensor-case.v2"
  manifest["profile"] = "tensor-v2"
  manifest["comparison"] = {
    "mode": "numeric",
    "policy": "absolute-relative",
    "atol": [1, 1000000],
    "rtol": [1, 1000000],
  }
  manifest["reference"]["precisions"] = [128, 256]
  manifest["execution"] = {
    "gap_sentinel_by_dtype": {"i32": "5a5a5a5a", "f32": "7fc00001"},
    "output_canary_by_dtype": {"i32": "a5a5a5a5", "f32": "7fc00002"},
    "full_shapes": [[2, 4]],
    "range_extents": [[2, 4]],
    "warp_probe": {
      "status": "not_probed",
      "num_warps": manifest["launch"]["num_warps"],
    },
    "loop_bounds": [],
    "total_body_executions": 0,
  }
  for name, strides, offset, span, order in (
    ("input", input_strides, input_offset, input_span, input_order),
    ("output", output_strides, output_offset, output_span, output_order),
  ):
    record = manifest["inputs"][0] if name == "input" else manifest["output"]
    record["element_strides"] = list(strides)
    record["element_offset"] = offset
    record["storage_span"] = span
    record["axis_order"] = list(order)
  for name, data in (
    ("program.sir", (root / "program.sir").read_bytes()),
    ("inputs.bin", (root / "inputs.bin").read_bytes()),
    ("expected.bin", (root / "expected.bin").read_bytes()),
    ("kernel.py", (root / "kernel.py").read_bytes()),
  ):
    manifest["files"][name] = {
      "bytes": len(data),
      "sha256": hashlib.sha256(data).hexdigest(),
    }
  manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
  return root


def available_environment():
  return {
    "schema": "refractir.tensor-environment.v1",
    "status": "environment_available",
    "campaign_available": True,
    "failure_reasons": [],
    "package_lock_file": "package-lock.json",
    "package_lock_sha256": "0" * 64,
    "package_lock": {
      "schema": "refractir.tensor-environment-lock.v1",
      "packages": [],
    },
  }


class TensorRunnerTests(unittest.TestCase):
  def setUp(self):
    self.assertTrue(SYMIRI.is_file(), f"symiri executable not found: {SYMIRI}")

  def test_v4_emit_uses_v4_tensor_profile_without_layout_manifest(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      case = SimpleNamespace(
        path=root,
        manifest={
          "entry": "kernel",
          "softmax_lowering": "direct",
          "profile": "tensor-v4",
        },
      )
      artifact_dir = root / "artifacts"
      commands = []

      def fake_run(command, **_kwargs):
        commands.append(command)
        output = Path(command[command.index("-o") + 1])
        output.write_text("def kernel(): pass\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

      with mock.patch.object(run_triton_tests.subprocess, "run", side_effect=fake_run):
        kernel, _, _ = run_triton_tests._emit_kernel(
          case, Path("/fake/symirc"), artifact_dir
        )

      self.assertIsNotNone(kernel)
      command = commands[0]
      profile_flag = command.index("--tensor-profile-version")
      self.assertEqual(command[profile_flag + 1], "v4")
      self.assertNotIn("--tensor-layout-manifest", command)

  def test_v2_kernel_layout_roles_disambiguate_input_named_return(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      program = root / "program.sir"
      program.write_text("", encoding="utf-8")
      case = SimpleNamespace(
        path=root,
        manifest={
          "entry": "kernel",
          "softmax_lowering": "direct",
          "profile": "tensor-v2",
          "inputs": [
            {
              "name": "return",
              "shape": [2],
              "element_strides": [1],
              "element_offset": 0,
              "storage_span": 2,
              "axis_order": [0],
            }
          ],
          "output": {
            "name": "return",
            "shape": [2, 2],
            "element_strides": [2, 1],
            "element_offset": 0,
            "storage_span": 4,
            "axis_order": [1, 0],
          },
        },
      )
      artifact_dir = root / "artifacts"

      def fake_run(command, **_kwargs):
        output = Path(command[command.index("-o") + 1])
        output.write_text("def kernel(): pass\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

      with mock.patch.object(run_triton_tests.subprocess, "run", side_effect=fake_run):
        kernel, _, _ = run_triton_tests._emit_kernel(
          case, Path("/fake/symirc"), artifact_dir
        )

      self.assertIsNotNone(kernel)
      layouts = json.loads(
        (artifact_dir / "tensor-layout.json").read_text(encoding="utf-8")
      )["buffers"]
      self.assertEqual(
        [(layout["name"], layout["role"]) for layout in layouts],
        [("return", "input"), ("return", "output")],
      )

  def test_full_output_mismatch_is_localized(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    expected = struct.pack("<8f", *range(8))
    actual = bytearray(expected)
    actual[-4:] = struct.pack("<f", 99.0)

    result = run_triton_tests.compare_outputs(
      bytes(actual),
      expected,
      "f32",
      [2, 4],
      {"mode": "exact", "policy": "bit-exact", "atol": [0, 1], "rtol": [0, 1]},
    )

    self.assertEqual(result["status"], "exact_mismatch")
    self.assertEqual(
      result["mismatches"],
      [
        {
          "index": 7,
          "coordinate": [1, 3],
          "expected_bits": "0x40e00000",
          "actual_bits": "0x42c60000",
        }
      ],
    )

  def test_manifest_source_shape_mismatch(self):
    self.assertIsNotNone(tensor_case, "missing tensor case validator module")
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_case(Path(directory) / "case", manifest_shape=(4, 2))
      with self.assertRaises(tensor_case.CaseValidationError) as raised:
        tensor_case.load_case(case_path, symiri=SYMIRI)
      self.assertEqual(raised.exception.status, "invalid_case")
      self.assertIn("shape", str(raised.exception).lower())
      marker = case_path / "worker-started"
      report = run_triton_tests.run_case(
        case_path,
        "compiled",
        worker_command=[
          sys.executable,
          "-c",
          "from pathlib import Path; Path(" + repr(str(marker)) + ").touch()",
        ],
        gpu_probe=lambda: True,
      )

      self.assertEqual(report["status"], "invalid_case")
      self.assertFalse(marker.exists(), "invalid manifest reached the child process")

  def test_unstable_reference_is_rejected_before_worker(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_case(Path(directory) / "case")
      manifest_path = case_path / "case.json"
      manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
      manifest["reference"]["stable"] = False
      manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
      marker = case_path / "worker-started"
      report = run_triton_tests.run_case(
        case_path,
        "interpret",
        worker_command=[
          sys.executable,
          "-c",
          "from pathlib import Path; Path(" + repr(str(marker)) + ").touch()",
        ],
        environment_probe=available_environment,
      )

      self.assertEqual(report["status"], "invalid_case")
      self.assertFalse(marker.exists(), "unstable reference reached the worker")

  def test_rejected_evidence_is_rejected_before_worker(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_case(Path(directory) / "case")
      manifest_path = case_path / "case.json"
      manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
      manifest["evidence"]["status"] = "rejected"
      manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
      marker = case_path / "worker-started"
      report = run_triton_tests.run_case(
        case_path,
        "interpret",
        worker_command=[
          sys.executable,
          "-c",
          "from pathlib import Path; Path(" + repr(str(marker)) + ").touch()",
        ],
        environment_probe=available_environment,
      )

      self.assertEqual(report["status"], "invalid_case")
      self.assertFalse(marker.exists(), "rejected evidence reached the worker")

  def test_smoke_and_regression_evidence_kinds_are_accepted(self):
    self.assertIsNotNone(tensor_case, "missing tensor case validator module")
    with tempfile.TemporaryDirectory() as directory:
      evidence_pairs = (
        ("handwritten", "validated"),
        ("conformance_matrix", "validated"),
        ("historical_regression", "validated"),
        ("error_directed", "error_directed_checked"),
      )
      for kind, status in evidence_pairs:
        with self.subTest(kind=kind):
          case_path = make_case(Path(directory) / kind)
          manifest_path = case_path / "case.json"
          manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
          manifest["evidence"] = {
            "kind": kind,
            "status": status,
            "artifacts": [],
          }
          manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

          case = tensor_case.load_case(case_path, symiri=SYMIRI)

          self.assertEqual(case.manifest["evidence"]["kind"], kind)

  def test_mode_isolation(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    child = (
      "import json,os; "
      "print(json.dumps({'pid':os.getpid(),'interpret':os.environ.get('TRITON_INTERPRET')}))"
    )
    os.environ["TRITON_INTERPRET"] = "stale-parent-value"
    try:
      compiled = run_triton_tests.run_child(
        [sys.executable, "-c", child], "compiled", 10
      )
      interpreted = run_triton_tests.run_child(
        [sys.executable, "-c", child], "interpret", 10
      )
    finally:
      os.environ.pop("TRITON_INTERPRET", None)

    compiled_result = json.loads(compiled.stdout)
    interpreted_result = json.loads(interpreted.stdout)
    self.assertIsNone(compiled_result["interpret"])
    self.assertEqual(interpreted_result["interpret"], "1")
    self.assertNotEqual(compiled_result["pid"], interpreted_result["pid"])

  def test_compiled_child_uses_isolated_cache_directory(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    with tempfile.TemporaryDirectory() as directory:
      cache = Path(directory) / "compiled-cache"
      cache.mkdir()
      (cache / "stale-entry").write_text("old")
      result_path = Path(directory) / "cache-result.json"
      script = (
        "import json,os; from pathlib import Path; p=Path(os.environ['TRITON_CACHE_DIR']); "
        "Path("
        + repr(str(result_path))
        + ").write_text(json.dumps({'path':str(p),'entries':list(p.iterdir())}))"
      )

      run_triton_tests.run_child(
        [sys.executable, "-c", script], "compiled", 10, cache_dir=cache
      )
      result = json.loads(result_path.read_text())

    self.assertEqual(Path(result["path"]), cache.resolve())
    self.assertEqual(result["entries"], [])

  def test_compiled_child_is_pinned_to_selected_visible_gpu(self):
    child = "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES'))"
    result = run_triton_tests.run_child(
      [sys.executable, "-c", child], "compiled", 10, device_id=3
    )

    self.assertEqual(result.stdout.decode().strip(), "3")

  def test_run_metadata_records_selected_physical_gpu(self):
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_case(Path(directory) / "case")
      artifact_dir = case_path / "runs" / "compiled"
      expected = case_path / "expected.bin"
      actual = artifact_dir / "actual.bin"
      worker = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path("
        + repr(str(actual))
        + ").write_bytes(Path("
        + repr(str(expected))
        + ").read_bytes())",
      ]

      def fake_emit(_case, _symirc, output_dir):
        kernel = Path(output_dir) / "kernel.py"
        kernel.write_text("def kernel(): pass\n")
        return kernel, "", ""

      with mock.patch.object(run_triton_tests, "_emit_kernel", side_effect=fake_emit):
        run_triton_tests.run_case(
          case_path,
          "compiled",
          device_id=5,
          worker_command=worker,
          gpu_probe=lambda: True,
          environment_probe=available_environment,
          symiri=SYMIRI,
          symirc=Path(__file__).resolve(),
        )
      metadata = json.loads((artifact_dir / "run.json").read_text())

    self.assertEqual(metadata["selected_device"], 5)

  def test_worker_sees_only_the_selected_gpu(self):
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_case(Path(directory) / "case")
      artifact_dir = case_path / "runs" / "compiled"
      expected = case_path / "expected.bin"
      actual = artifact_dir / "actual.bin"
      observed = Path(directory) / "visible-gpu.json"
      worker = [
        sys.executable,
        "-c",
        "import json,os; from pathlib import Path; Path("
        + repr(str(observed))
        + ").write_text(json.dumps({'visible':os.environ.get('CUDA_VISIBLE_DEVICES')})); Path("
        + repr(str(actual))
        + ").write_bytes(Path("
        + repr(str(expected))
        + ").read_bytes())",
      ]

      def fake_emit(_case, _symirc, output_dir):
        kernel = Path(output_dir) / "kernel.py"
        kernel.write_text("def kernel(): pass\n")
        return kernel, "", ""

      with mock.patch.object(run_triton_tests, "_emit_kernel", side_effect=fake_emit):
        run_triton_tests.run_case(
          case_path,
          "compiled",
          device_id=6,
          worker_command=worker,
          gpu_probe=lambda: True,
          environment_probe=available_environment,
          symiri=SYMIRI,
          symirc=Path(__file__).resolve(),
        )
      visible = json.loads(observed.read_text())["visible"]

    self.assertEqual(visible, "6")

  def test_invalid_device_index_is_rejected_before_kernel_emission(self):
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_case(Path(directory) / "case")
      with mock.patch.object(run_triton_tests, "_emit_kernel") as emit:
        report = run_triton_tests.run_case(
          case_path,
          "compiled",
          device_id=-1,
          gpu_probe=lambda: True,
          symiri=SYMIRI,
        )

    self.assertEqual(report["status"], "invalid_case")
    emit.assert_not_called()

  def test_interpreter_does_not_probe_for_a_gpu(self):
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_case(Path(directory) / "case")
      artifact_dir = case_path / "runs" / "interpret"
      expected = case_path / "expected.bin"
      actual = artifact_dir / "actual.bin"
      worker = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path("
        + repr(str(actual))
        + ").write_bytes(Path("
        + repr(str(expected))
        + ").read_bytes())",
      ]

      def fake_emit(_case, _symirc, output_dir):
        kernel = Path(output_dir) / "kernel.py"
        kernel.write_text("def kernel(): pass\n")
        return kernel, "", ""

      with mock.patch.object(run_triton_tests, "_emit_kernel", side_effect=fake_emit):
        report = run_triton_tests.run_case(
          case_path,
          "interpret",
          worker_command=worker,
          gpu_probe=lambda: (_ for _ in ()).throw(
            AssertionError("unexpected GPU probe")
          ),
          environment_probe=available_environment,
          symiri=SYMIRI,
          symirc=Path(__file__).resolve(),
        )

    self.assertEqual(report["status"], "pass")

  def test_command_line_accepts_selected_gpu_index(self):
    reports = []

    def fake_run(case, mode, **kwargs):
      reports.append((case, mode, kwargs.get("device_id")))
      return {"status": "pass", "campaign_valid": True}

    with mock.patch.object(run_triton_tests, "run_case", side_effect=fake_run):
      code = run_triton_tests.main(
        ["--case", "case-dir", "--mode", "compiled", "--device", "4"],
        gpu_probe=lambda: True,
      )

    self.assertEqual(code, 0)
    self.assertEqual(reports, [("case-dir", "compiled", 4)])

  def test_interpreter_child_does_not_inherit_compiler_cache(self):
    child = "import os; print(os.environ.get('TRITON_CACHE_DIR'))"
    os.environ["TRITON_CACHE_DIR"] = "/shared/stale-cache"
    try:
      result = run_triton_tests.run_child(
        [sys.executable, "-c", child], "interpret", 10
      )
    finally:
      os.environ.pop("TRITON_CACHE_DIR", None)

    self.assertEqual(result.stdout.decode().strip(), "None")

  def test_run_metadata_records_fresh_compiler_cache(self):
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_case(Path(directory) / "case")
      artifact_dir = case_path / "runs" / "compiled"
      cache = artifact_dir / "triton-cache"
      cache.mkdir(parents=True)
      (cache / "stale-entry").write_text("old")
      expected = case_path / "expected.bin"
      actual = artifact_dir / "actual.bin"
      worker = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path("
        + repr(str(actual))
        + ").write_bytes(Path("
        + repr(str(expected))
        + ").read_bytes())",
      ]

      def fake_emit(_case, _symirc, output_dir):
        kernel = Path(output_dir) / "kernel.py"
        kernel.write_text("def kernel(): pass\n")
        return kernel, "", ""

      with mock.patch.object(run_triton_tests, "_emit_kernel", side_effect=fake_emit):
        report = run_triton_tests.run_case(
          case_path,
          "compiled",
          worker_command=worker,
          gpu_probe=lambda: True,
          environment_probe=available_environment,
          symiri=SYMIRI,
          symirc=Path(__file__).resolve(),
        )
      run = json.loads((artifact_dir / "run.json").read_text())

    self.assertEqual(report["status"], "pass")
    self.assertTrue(run["compiler_cache"]["empty_at_start"])
    self.assertTrue(run["compiler_cache"]["cache_id"])
    self.assertFalse((cache / "stale-entry").exists())

  def test_compiled_and_interpreter_runs_have_separate_cache_policy(self):
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_case(Path(directory) / "case")
      expected = case_path / "expected.bin"

      def fake_emit(_case, _symirc, output_dir):
        kernel = Path(output_dir) / "kernel.py"
        kernel.write_text("def kernel(): pass\n")
        return kernel, "", ""

      reports = []
      with mock.patch.object(run_triton_tests, "_emit_kernel", side_effect=fake_emit):
        for mode in ("compiled", "interpret"):
          actual = case_path / "runs" / mode / "actual.bin"
          worker = [
            sys.executable,
            "-c",
            "from pathlib import Path; Path("
            + repr(str(actual))
            + ").write_bytes(Path("
            + repr(str(expected))
            + ").read_bytes())",
          ]
          reports.append(
            run_triton_tests.run_case(
              case_path,
              mode,
              worker_command=worker,
              gpu_probe=lambda: True,
              environment_probe=available_environment,
              symiri=SYMIRI,
              symirc=Path(__file__).resolve(),
            )
          )
      compiled = json.loads((case_path / "runs" / "compiled" / "run.json").read_text())
      interpreted = json.loads(
        (case_path / "runs" / "interpret" / "run.json").read_text()
      )

    self.assertEqual([report["status"] for report in reports], ["pass", "pass"])
    self.assertIsNotNone(compiled["compiler_cache"])
    self.assertIsNone(interpreted["compiler_cache"])

  def test_separate_case_trials_receive_distinct_cache_ids(self):
    with tempfile.TemporaryDirectory() as directory:
      case_paths = [
        make_case(Path(directory) / "ordinary"),
        make_case(Path(directory) / "directed"),
      ]

      def fake_emit(_case, _symirc, output_dir):
        kernel = Path(output_dir) / "kernel.py"
        kernel.write_text("def kernel(): pass\n")
        return kernel, "", ""

      cache_records = []
      with mock.patch.object(run_triton_tests, "_emit_kernel", side_effect=fake_emit):
        for case_path in case_paths:
          artifact_dir = case_path / "runs" / "compiled"
          actual = artifact_dir / "actual.bin"
          expected = case_path / "expected.bin"
          worker = [
            sys.executable,
            "-c",
            "from pathlib import Path; Path("
            + repr(str(actual))
            + ").write_bytes(Path("
            + repr(str(expected))
            + ").read_bytes())",
          ]
          run_triton_tests.run_case(
            case_path,
            "compiled",
            worker_command=worker,
            gpu_probe=lambda: True,
            environment_probe=available_environment,
            symiri=SYMIRI,
            symirc=Path(__file__).resolve(),
          )
          metadata = json.loads((artifact_dir / "run.json").read_text())
          cache_records.append(metadata["compiler_cache"])

    self.assertNotEqual(cache_records[0]["cache_id"], cache_records[1]["cache_id"])
    self.assertEqual(cache_records[0]["path"], "runs/compiled/triton-cache")

  def test_timeout_retains_reproducer(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_case(Path(directory) / "case")
      report = run_triton_tests.run_case(
        case_path,
        "compiled",
        timeout_s=0.1,
        worker_command=[sys.executable, "-c", "import time; time.sleep(30)"],
        gpu_probe=lambda: True,
        environment_probe=available_environment,
      )

      self.assertEqual(report["status"], "runtime_timeout")
      self.assertIn("program.sir", report["artifacts"])
      self.assertIn("inputs.bin", report["artifacts"])
      self.assertTrue((case_path / "runs" / "compiled" / "stdout.log").is_file())
      self.assertTrue((case_path / "runs" / "compiled" / "stderr.log").is_file())

  def test_missing_gpu_is_not_pass(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_case(Path(directory) / "case")
      stdout = StringIO()
      with contextlib.redirect_stdout(stdout):
        exit_code = run_triton_tests.main(
          ["--case", str(case_path), "--mode", "compiled"],
          gpu_probe=lambda: False,
        )

    report = json.loads(stdout.getvalue())
    self.assertNotEqual(exit_code, 0)
    self.assertEqual(report["status"], "environment_unavailable")

  def test_partial_output_file_is_runtime_failure(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_case(Path(directory) / "case")
      partial = case_path / "runs" / "compiled" / "actual.bin"
      child = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path(" + repr(str(partial)) + ").write_bytes(b'x')",
      ]
      report = run_triton_tests.run_case(
        case_path,
        "compiled",
        worker_command=child,
        gpu_probe=lambda: True,
        environment_probe=available_environment,
      )

    self.assertEqual(report["status"], "runtime_failure")
    self.assertEqual(report["actual_bytes"], 1)

  def test_execution_artifacts_are_isolated_by_mode(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_case(Path(directory) / "case")

      def worker(mode):
        actual = case_path / "runs" / mode / "actual.bin"
        script = (
          "from pathlib import Path; p=Path(" + repr(str(actual)) + "); "
          "p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(Path("
          + repr(str(case_path / "expected.bin"))
          + ").read_bytes())"
        )
        return [sys.executable, "-c", script]

      compiled = run_triton_tests.run_case(
        case_path,
        "compiled",
        worker_command=worker("compiled"),
        gpu_probe=lambda: True,
        environment_probe=available_environment,
      )
      interpreted = run_triton_tests.run_case(
        case_path,
        "interpret",
        worker_command=worker("interpret"),
        environment_probe=available_environment,
      )

      self.assertEqual(compiled["status"], "pass")
      self.assertEqual(interpreted["status"], "pass")
      compiled_digests = compiled["artifact_digests"]
      for relative in (
        "runs/compiled/actual.bin",
        "runs/compiled/run.json",
        "runs/compiled/environment.json",
        "runs/compiled/package-lock.json",
        "runs/compiled/kernel.py",
      ):
        self.assertIn(relative, compiled_digests)
        data = (case_path / relative).read_bytes()
        self.assertEqual(len(data), compiled_digests[relative]["bytes"])
        self.assertEqual(
          hashlib.sha256(data).hexdigest(), compiled_digests[relative]["sha256"]
        )
      self.assertTrue((case_path / "runs" / "interpret" / "actual.bin").is_file())

  def test_interpret_report_is_retained_when_compiled_is_saved(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    with tempfile.TemporaryDirectory() as directory:
      case_path = Path(directory) / "case"
      case_path.mkdir()
      run_triton_tests._save_report(case_path, {"mode": "interpret", "status": "pass"})
      run_triton_tests._save_report(case_path, {"mode": "compiled", "status": "pass"})

      interpreted = case_path / "result-interpret.json"
      compiled = case_path / "result-compiled.json"
      self.assertTrue(interpreted.is_file())
      self.assertTrue(compiled.is_file())
      self.assertEqual(json.loads(interpreted.read_text())["mode"], "interpret")
      self.assertEqual(json.loads(compiled.read_text())["mode"], "compiled")

  def test_environment_failure_report_is_mode_scoped(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    with tempfile.TemporaryDirectory() as directory:
      case_path = Path(directory) / "case"
      case_path.mkdir()
      run_triton_tests._save_report(
        case_path,
        {"mode": "compiled", "status": "environment_unavailable", "exit_code": 2},
      )

      report = json.loads((case_path / "result-compiled.json").read_text())
      self.assertEqual(report["status"], "environment_unavailable")
      self.assertEqual(report["exit_code"], 2)

  def test_compile_failure_report_is_mode_scoped(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    with tempfile.TemporaryDirectory() as directory:
      case_path = Path(directory) / "case"
      case_path.mkdir()
      run_triton_tests._save_report(
        case_path,
        {"mode": "interpret", "status": "compile_failure", "artifacts": ["kernel.py"]},
      )

      report = json.loads((case_path / "result-interpret.json").read_text())
      self.assertEqual(report["status"], "compile_failure")
      self.assertEqual(report["artifacts"], ["kernel.py"])

  def test_runtime_timeout_report_is_mode_scoped(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    with tempfile.TemporaryDirectory() as directory:
      case_path = Path(directory) / "case"
      case_path.mkdir()
      run_triton_tests._save_report(
        case_path,
        {"mode": "compiled", "status": "runtime_timeout", "timeout_s": 0.25},
      )

      report = json.loads((case_path / "result-compiled.json").read_text())
      self.assertEqual(report["status"], "runtime_timeout")
      self.assertEqual(report["timeout_s"], 0.25)

  def test_mismatch_report_is_mode_scoped(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    with tempfile.TemporaryDirectory() as directory:
      case_path = Path(directory) / "case"
      case_path.mkdir()
      run_triton_tests._save_report(
        case_path,
        {"mode": "compiled", "status": "numeric_mismatch", "mismatching_indices": [3]},
      )

      report = json.loads((case_path / "result-compiled.json").read_text())
      self.assertEqual(report["status"], "numeric_mismatch")
      self.assertEqual(report["mismatching_indices"], [3])

  def test_result_json_remains_latest_report(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    with tempfile.TemporaryDirectory() as directory:
      case_path = Path(directory) / "case"
      case_path.mkdir()
      run_triton_tests._save_report(case_path, {"mode": "interpret", "status": "pass"})

      report = json.loads((case_path / "result.json").read_text())
      self.assertEqual(report["mode"], "interpret")

  def test_v2_manifest_accepts_a_valid_noncontiguous_layout(self):
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_v2_case(Path(directory) / "case")
      case = tensor_case.load_case(case_path, symiri=SYMIRI)

    self.assertEqual(case.manifest["profile"], "tensor-v2")
    self.assertEqual(case.manifest["inputs"][0]["element_strides"], [1, 2])

  def test_v2_manifest_rejects_overlapping_strides(self):
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_v2_case(
        Path(directory) / "case",
        input_strides=(1, 1),
        input_span=8,
        input_order=(0, 1),
      )
      with self.assertRaisesRegex(tensor_case.CaseValidationError, "overlap"):
        tensor_case.load_case(case_path, symiri=SYMIRI)

  def test_v2_manifest_rejects_inconsistent_axis_order(self):
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_v2_case(
        Path(directory) / "case",
        input_strides=(4, 1),
        input_order=(0, 1),
      )
      with self.assertRaisesRegex(tensor_case.CaseValidationError, "axis order"):
        tensor_case.load_case(case_path, symiri=SYMIRI)

  def test_v2_manifest_rejects_storage_span_outside_last_address(self):
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_v2_case(
        Path(directory) / "case",
        input_strides=(1, 2),
        input_offset=1,
        input_span=8,
      )
      with self.assertRaisesRegex(tensor_case.CaseValidationError, "storage span"):
        tensor_case.load_case(case_path, symiri=SYMIRI)

  def test_v2_manifest_requires_matching_sha256_artifacts(self):
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_v2_case(Path(directory) / "case")
      manifest_path = case_path / "case.json"
      manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
      manifest["files"]["inputs.bin"]["sha256"] = "0" * 64
      manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
      with self.assertRaisesRegex(tensor_case.CaseValidationError, "SHA-256"):
        tensor_case.load_case(case_path, symiri=SYMIRI)

  def test_warp_preflight_runs_every_legal_setting_and_records_results(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_v2_case(Path(directory) / "case")
      calls = []

      def fake_run(case, mode, **_kwargs):
        manifest = json.loads((Path(case) / "case.json").read_text())
        num_warps = manifest["launch"]["num_warps"]
        calls.append(num_warps)
        artifact = Path(case) / "runs" / mode / "actual.bin"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(bytes([num_warps]))
        return {
          "status": "pass",
          "campaign_valid": True,
          "artifacts": [artifact.relative_to(case).as_posix()],
        }

      report = run_triton_tests.probe_warp_settings(
        case_path,
        warp_counts=(1, 2, 4, 8),
        run_case_fn=fake_run,
        symiri=SYMIRI,
      )

      self.assertEqual(calls, [1, 2, 4, 8])
      self.assertEqual(report["status"], "probed")
      self.assertEqual(
        [item["classification"] for item in report["results"]],
        ["pass"] * 4,
      )
      manifest = json.loads((case_path / "case.json").read_text())
      self.assertEqual(manifest["execution"]["warp_probe"]["status"], "probed")
      self.assertEqual(len(manifest["execution"]["warp_probe"]["results"]), 4)
      validated = tensor_case.load_case(case_path, symiri=SYMIRI)
      self.assertEqual(validated.manifest["profile"], "tensor-v2")

  def test_warp_preflight_marks_only_profile_exclusions_unsupported(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_v2_case(Path(directory) / "case")
      calls = []
      report = run_triton_tests.probe_warp_settings(
        case_path,
        warp_counts=(3,),
        run_case_fn=lambda *_args, **_kwargs: calls.append(True),
        symiri=SYMIRI,
      )

      self.assertEqual(calls, [])
      self.assertEqual(
        report["results"][0]["classification"], "unsupported_configuration"
      )
      self.assertIn("profile", report["results"][0]["reason"].lower())

  def test_warp_preflight_continues_after_a_legal_wrong_result(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_v2_case(Path(directory) / "case")
      calls = []

      def fake_run(case, _mode, **_kwargs):
        num_warps = json.loads((Path(case) / "case.json").read_text())["launch"][
          "num_warps"
        ]
        calls.append(num_warps)
        status = "numeric_mismatch" if num_warps == 1 else "pass"
        return {
          "status": status,
          "campaign_valid": status == "pass",
          "artifacts": [],
        }

      report = run_triton_tests.probe_warp_settings(
        case_path,
        warp_counts=(1, 2),
        run_case_fn=fake_run,
        symiri=SYMIRI,
      )

      self.assertEqual(calls, [1, 2])
      self.assertEqual(
        [item["classification"] for item in report["results"]],
        ["candidate", "pass"],
      )

  def test_warp_preflight_separates_resource_and_environment_failures(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_v2_case(Path(directory) / "case")

      def fake_run(case, _mode, **_kwargs):
        num_warps = json.loads((Path(case) / "case.json").read_text())["launch"][
          "num_warps"
        ]
        status = {
          1: "resource_limit",
          2: "environment_unavailable",
          4: "unsupported_configuration",
          8: "runtime_failure",
        }[num_warps]
        return {"status": status, "reason": f"probe {num_warps}", "artifacts": []}

      report = run_triton_tests.probe_warp_settings(
        case_path,
        warp_counts=(1, 2, 4, 8),
        run_case_fn=fake_run,
        symiri=SYMIRI,
      )

      self.assertEqual(
        [item["classification"] for item in report["results"]],
        [
          "resource_limit",
          "environment_unavailable",
          "unsupported_configuration",
          "candidate",
        ],
      )

  def test_warp_preflight_artifacts_are_isolated_by_setting(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_v2_case(Path(directory) / "case")

      def fake_run(case, mode, **_kwargs):
        case = Path(case)
        num_warps = json.loads((case / "case.json").read_text())["launch"]["num_warps"]
        artifact = case / "runs" / mode / "actual.bin"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"warp-{num_warps}".encode())
        return {
          "status": "pass",
          "campaign_valid": True,
          "artifacts": [artifact.relative_to(case).as_posix()],
        }

      report = run_triton_tests.probe_warp_settings(
        case_path,
        warp_counts=(1, 4),
        run_case_fn=fake_run,
        symiri=SYMIRI,
      )
      artifact_paths = [
        case_path / item["case_dir"] / "runs" / "compiled" / "actual.bin"
        for item in report["results"]
      ]

      self.assertEqual(len(set(artifact_paths)), 2)
      self.assertEqual(
        [path.read_bytes() for path in artifact_paths], [b"warp-1", b"warp-4"]
      )
      for result in report["results"]:
        with self.subTest(num_warps=result["num_warps"]):
          clone = case_path / result["case_dir"]
          self.assertEqual(
            tensor_case.load_case(clone, symiri=SYMIRI).manifest["profile"],
            "tensor-v2",
          )

  def test_warp_preflight_requires_a_pinned_campaign_environment(self):
    self.assertIsNotNone(run_triton_tests, "missing isolated Triton runner module")
    with tempfile.TemporaryDirectory() as directory:
      case_path = make_v2_case(Path(directory) / "case")
      report = run_triton_tests.probe_warp_settings(
        case_path,
        warp_counts=(1,),
        run_case_fn=lambda *_args, **_kwargs: {
          "status": "pass",
          "campaign_valid": False,
          "artifacts": [],
        },
        symiri=SYMIRI,
      )

      self.assertEqual(report["results"][0]["status"], "pass")
      self.assertEqual(
        report["results"][0]["classification"], "environment_unavailable"
      )


if __name__ == "__main__":
  unittest.main()
