"""Combined tensor test coverage: test_tensor_case_v3.py, test_tensor_case_v4.py, test_tensor_v4_calibration.py."""

import ast
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test.lib import tensor_case
from test.lib.build_tensor_11986_case import build_case
from test.lib.run_triton_tests import compare_outputs
from test.lib.tensor_case import load_case
from test.unit.tensor_support import SYMIRC, SYMIRI


# Consolidated from test_tensor_case_v3.py.
def _manifest_case(
  root,
  shape=(17,),
  *,
  input_stride=1,
  output_stride=1,
  tile_shape=None,
  output_canary="a5a5a5a5",
):
  root = Path(root)
  root.mkdir(parents=True, exist_ok=True)
  elements = 1
  for dim in shape:
    elements *= dim
  values = list(range(elements))
  sir_shape = "[" + ", ".join(str(dim) for dim in shape) + "]"
  source = (
    f"fun @kernel(%x: tensor {sir_shape} i32) : tensor {sir_shape} i32 {{\n"
    f"  let mut %out: tensor {sir_shape} i32 = undef;\n"
    "^entry:\n  %out = %x;\n  ret %out;\n}\n"
  )
  inputs = struct.pack(f"<{elements}i", *values)
  expected = inputs
  (root / "program.sir").write_bytes(source.encode())
  (root / "inputs.bin").write_bytes(inputs)
  (root / "expected.bin").write_bytes(expected)
  strides = [input_stride] * len(shape)
  output_strides = [output_stride] * len(shape)
  input_span = 1 + sum((dim - 1) * stride for dim, stride in zip(shape, strides))
  output_span = 1 + sum(
    (dim - 1) * stride for dim, stride in zip(shape, output_strides)
  )
  tile = tile_shape or [1 << (dim - 1).bit_length() for dim in shape]
  files = {
    name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    for name, data in (
      ("program.sir", source.encode()),
      ("inputs.bin", inputs),
      ("expected.bin", expected),
    )
  }
  manifest = {
    "schema": tensor_case.SCHEMA_V3,
    "profile": tensor_case.PROFILE_V3,
    "entry": "kernel",
    "source": {"file": "program.sir", "sha256": files["program.sir"]["sha256"]},
    "files": files,
    "inputs": [
      {
        "name": "x",
        "dtype": "i32",
        "shape": list(shape),
        "file": "inputs.bin",
        "byte_offset": 0,
        "byte_length": len(inputs),
        "element_strides": strides,
        "element_offset": 0,
        "storage_span": input_span,
        "axis_order": list(sorted(range(len(shape)), key=lambda axis: strides[axis])),
        "tile_shape": tile,
        "tile_origin": [0] * len(shape),
      }
    ],
    "output": {
      "name": "return",
      "dtype": "i32",
      "shape": list(shape),
      "file": "expected.bin",
      "byte_offset": 0,
      "byte_length": len(expected),
      "element_strides": output_strides,
      "element_offset": 0,
      "storage_span": output_span,
      "axis_order": list(
        sorted(range(len(shape)), key=lambda axis: output_strides[axis])
      ),
      "tile_shape": tile,
      "tile_origin": [0] * len(shape),
    },
    "grid": [1, 1, 1],
    "softmax_lowering": "direct",
    "launch": {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False},
    "execution": {
      "gap_sentinel_by_dtype": {"i32": "5a5a5a5a", "f32": "7fc00001"},
      "output_canary_by_dtype": {"i32": output_canary, "f32": "7fc00002"},
      "full_shapes": [list(shape)],
      "range_extents": [list(shape)],
      "warp_probe": {"status": "not_probed", "num_warps": 4},
      "loop_bounds": [],
      "total_body_executions": 0,
    },
    "coverage": {"input_behavior": {"pattern": "coordinate-distinct"}},
    "generator": {"input_pattern": "coordinate-distinct"},
    "property": {"id": "copy", "source": "true"},
    "evidence": {"kind": "construction", "status": "validated", "artifacts": []},
    "comparison": {"mode": "exact", "atol": [0, 1], "rtol": [0, 1]},
    "reference": {"defined": True, "stable": True, "precisions": []},
    "environment": None,
  }
  (root / "case.json").write_text(json.dumps(manifest), encoding="utf-8")
  signature = {
    "entry": "kernel",
    "inputs": [
      {"name": "x", "dtype": "i32", "shape": list(shape), "byte_length": len(inputs)}
    ],
    "output": {
      "name": "return",
      "dtype": "i32",
      "shape": list(shape),
      "byte_length": len(expected),
    },
  }
  return root, signature


class TensorV3CaseTests(unittest.TestCase):
  def _load(self, root, signature):
    with mock.patch.object(tensor_case, "_describe", return_value=signature):
      return tensor_case.load_case(root, symiri=Path("/unused/symiri"))

  def test_v3_accepts_non_power_of_two_logical_shape_and_strides(self):
    with tempfile.TemporaryDirectory() as directory:
      root, signature = _manifest_case(directory, input_stride=2, output_stride=2)

      case = self._load(root, signature)

      self.assertEqual(case.manifest["inputs"][0]["shape"], [17])
      self.assertEqual(case.manifest["inputs"][0]["tile_shape"], [32])

  def test_v3_rejects_tile_metadata_that_disagrees_with_logical_shape(self):
    with tempfile.TemporaryDirectory() as directory:
      root, signature = _manifest_case(directory, tile_shape=[17])

      with self.assertRaisesRegex(tensor_case.CaseValidationError, "tile shape"):
        self._load(root, signature)

  def test_v3_rejects_output_canary_matching_any_expected_value(self):
    with tempfile.TemporaryDirectory() as directory:
      root, signature = _manifest_case(directory, output_canary="02000000")

      with self.assertRaisesRegex(tensor_case.CaseValidationError, "output canary"):
        self._load(root, signature)

  def test_v3_rejects_padded_volume_above_512(self):
    with tempfile.TemporaryDirectory() as directory:
      root, signature = _manifest_case(directory, shape=(65, 5))

      with self.assertRaisesRegex(
        tensor_case.CaseValidationError, "padded tile volume"
      ):
        self._load(root, signature)

  def test_v2_manifest_cannot_use_v3_non_power_of_two_shape(self):
    with tempfile.TemporaryDirectory() as directory:
      root, signature = _manifest_case(directory)
      manifest_path = root / "case.json"
      manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
      manifest["schema"] = tensor_case.SCHEMA_V2
      manifest["profile"] = tensor_case.PROFILE_V2
      manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

      with self.assertRaisesRegex(tensor_case.CaseValidationError, "powers of two"):
        self._load(root, signature)


if __name__ == "__main__":
  unittest.main()

# Consolidated from test_tensor_case_v4.py.
SOURCE = """\
fun @kernel(%x: tensor [2] f32, %y: tensor [2] f32) : tensor [2] f32 {
  let mut %out: tensor [2] f32 = undef;
^entry:
  %out = tensor rem %x, %y;
  ret %out;
}
"""


def write_case(
  root,
  *,
  x=(-0.0, 0.0),
  y=(3.0, 3.0),
  expected=(-0.0, 0.0),
  zero_sign_exact=True,
  input_strides=(1,),
):
  root.mkdir(parents=True, exist_ok=True)
  artifacts = {
    "program.sir": SOURCE.encode(),
    "inputs.bin": struct.pack("<4f", *(x + y)),
    "expected.bin": struct.pack("<2f", *expected),
  }
  for name, data in artifacts.items():
    (root / name).write_bytes(data)

  def digest(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

  def buffer(name, values, file_name, offset):
    return {
      "name": name,
      "dtype": "f32",
      "shape": [2],
      "file": file_name,
      "byte_offset": offset,
      "byte_length": 8,
      "element_strides": list(input_strides),
      "element_offset": 0,
      "storage_span": 2,
      "axis_order": [0],
      "tile_shape": [2],
      "tile_origin": [0],
    }

  comparison = {"mode": "exact", "atol": [0, 1], "rtol": [0, 1]}
  if zero_sign_exact:
    comparison["zero_sign_exact"] = True
  manifest = {
    "schema": tensor_case.SCHEMA_V4,
    "profile": tensor_case.PROFILE_V4,
    "entry": "kernel",
    "source": {
      "file": "program.sir",
      "sha256": digest(artifacts["program.sir"])["sha256"],
    },
    "inputs": [buffer("x", x, "inputs.bin", 0), buffer("y", y, "inputs.bin", 8)],
    "output": buffer("return", expected, "expected.bin", 0),
    "grid": [1, 1, 1],
    "softmax_lowering": "direct",
    "launch": {"num_warps": 1, "num_stages": 1, "enable_fp_fusion": False},
    "execution": {
      "full_shapes": [[2], [2]],
      "range_extents": [[2], [2]],
      "warp_probe": {"status": "not_probed", "num_warps": 1},
      "loop_bounds": [],
      "total_body_executions": 0,
      "gap_sentinel_by_dtype": {"i32": "55555555", "f32": "3f000001"},
      "output_canary_by_dtype": {"i32": "aaaaaaaa", "f32": "7fc00002"},
    },
    "property": {"id": "defined-inputs", "source": "finite f32 fmod operands"},
    "evidence": {
      "kind": "authored_calibration",
      "status": "validated",
      "artifacts": ["program.sir", "inputs.bin", "expected.bin"],
    },
    "comparison": comparison,
    "reference": {"defined": True, "stable": True, "precisions": []},
    "coverage": {"primary_stratum": "authored-calibration"},
    "operator_coverage": ["rem"],
    "generator": {
      "version": "authored-calibration",
      "seed": 0,
      "synthesis": "construction",
      "primary_stratum": "authored-calibration",
    },
    "files": {name: digest(data) for name, data in artifacts.items()},
  }
  (root / "case.json").write_text(json.dumps(manifest), encoding="utf-8")
  return manifest


class TensorCaseV4Tests(unittest.TestCase):
  def test_loads_canonical_signed_zero_calibration_case(self):
    with tempfile.TemporaryDirectory() as directory:
      case_dir = Path(directory) / "case"
      manifest = write_case(case_dir)

      case = tensor_case.load_case(case_dir, symiri=SYMIRI)

      self.assertEqual(case.manifest["profile"], "tensor-v4")
      self.assertTrue(manifest["comparison"]["zero_sign_exact"])

  def test_rejects_zero_divisor_and_missing_zero_sign_policy(self):
    with tempfile.TemporaryDirectory() as directory:
      case_dir = Path(directory) / "zero-divisor"
      write_case(case_dir, y=(0.0, 3.0))
      with self.assertRaisesRegex(tensor_case.CaseValidationError, "divisor lane 0"):
        tensor_case.load_case(case_dir, symiri=SYMIRI)

    with tempfile.TemporaryDirectory() as directory:
      case_dir = Path(directory) / "missing-policy"
      write_case(case_dir, zero_sign_exact=False)
      with self.assertRaisesRegex(tensor_case.CaseValidationError, "signed-zero"):
        tensor_case.load_case(case_dir, symiri=SYMIRI)

  def test_rejects_noncontiguous_layout_and_out_of_bounds_quotient(self):
    with tempfile.TemporaryDirectory() as directory:
      case_dir = Path(directory) / "strided"
      write_case(case_dir, input_strides=(2,))
      with self.assertRaisesRegex(tensor_case.CaseValidationError, "layout is invalid"):
        tensor_case.load_case(case_dir, symiri=SYMIRI)

    with tempfile.TemporaryDirectory() as directory:
      case_dir = Path(directory) / "large-quotient"
      manifest = write_case(case_dir, x=(65536.0, 0.0), y=(2.0**-64, 3.0))
      manifest["files"]["inputs.bin"] = {
        "bytes": len((case_dir / "inputs.bin").read_bytes()),
        "sha256": hashlib.sha256((case_dir / "inputs.bin").read_bytes()).hexdigest(),
      }
      (case_dir / "case.json").write_text(json.dumps(manifest), encoding="utf-8")
      with self.assertRaisesRegex(tensor_case.CaseValidationError, "quotient lane 0"):
        tensor_case.load_case(case_dir, symiri=SYMIRI)

  def test_numeric_v4_comparison_rejects_tolerance_hidden_zero_changes(self):
    reference = struct.pack("<I", 0x80000000)
    for actual in (struct.pack("<I", 0), struct.pack("<f", -(2.0**-80))):
      result = compare_outputs(
        actual,
        reference,
        "f32",
        [1],
        {
          "mode": "numeric",
          "atol": [2, 100000],
          "rtol": [2, 1000],
          "zero_sign_exact": True,
        },
      )
      self.assertEqual(result["status"], "numeric_mismatch")


# Consolidated from test_tensor_v4_calibration.py.
class TensorV4CalibrationTests(unittest.TestCase):
  def test_authored_case_contains_canonical_signed_zero_and_native_remainder(self):
    with tempfile.TemporaryDirectory() as directory:
      case_dir = Path(directory) / "authored-11986"
      build_case(case_dir, symiri=SYMIRI, symirc=SYMIRC)
      case = load_case(case_dir, symiri=SYMIRI)

      self.assertEqual(case.manifest["evidence"]["kind"], "authored_calibration")
      self.assertEqual(case.manifest["generator"]["version"], "authored-calibration")
      self.assertEqual(
        struct.unpack("<4I", case.inputs),
        (0x80000000, 0x00000000, 0x40400000, 0x40400000),
      )
      self.assertEqual(struct.unpack("<2I", case.expected), (0x80000000, 0x00000000))
      kernel = ast.parse((case_dir / "kernel.py").read_text(encoding="utf-8"))
      self.assertTrue(
        any(
          isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod)
          for node in ast.walk(kernel)
        )
      )


if __name__ == "__main__":
  unittest.main()
