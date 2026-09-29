"""Build the authored signed-zero calibration case for Triton issue #11986."""

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

from test.lib.tensor_coverage import source_signature

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYMIRI = Path(os.environ.get("REFRACTIR_SYMIRI", ROOT / "symiri")).resolve()
DEFAULT_SYMIRC = Path(os.environ.get("REFRACTIR_SYMIRC", ROOT / "symirc")).resolve()
SOURCE = """\
fun @kernel(%x: tensor [2] f32, %y: tensor [2] f32) : tensor [2] f32 {
  let mut %out: tensor [2] f32 = undef;
^entry:
  %out = tensor rem %x, %y;
  ret %out;
}
"""
INPUT_BYTES = struct.pack("<4I", 0x80000000, 0x00000000, 0x40400000, 0x40400000)


def _run(command):
  result = subprocess.run(
    command, capture_output=True, text=True, timeout=30, check=False
  )
  if result.returncode != 0:
    raise RuntimeError(
      "command failed: "
      + " ".join(map(str, command))
      + "\n"
      + result.stdout
      + result.stderr
    )
  return result


def _digest(data):
  return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _buffer(name, file_name, offset, byte_length):
  return {
    "name": name,
    "dtype": "f32",
    "shape": [2],
    "file": file_name,
    "byte_offset": offset,
    "byte_length": byte_length,
    "element_strides": [1],
    "element_offset": 0,
    "storage_span": 2,
    "axis_order": [0],
    "tile_shape": [2],
    "tile_origin": [0],
  }


def build_case(output_dir, *, symiri=DEFAULT_SYMIRI, symirc=DEFAULT_SYMIRC):
  """Build a fresh authored-replay bundle; it is not generator rediscovery."""
  output_dir = Path(output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=False)
  source_bytes = SOURCE.encode("utf-8")
  source_path = output_dir / "program.sir"
  inputs_path = output_dir / "inputs.bin"
  expected_path = output_dir / "expected.bin"
  kernel_path = output_dir / "kernel.py"
  source_path.write_bytes(source_bytes)
  inputs_path.write_bytes(INPUT_BYTES)

  _run(
    [
      str(symiri),
      "--tensor-profile-version",
      "v4",
      "--entry",
      "kernel",
      "--tensor-input",
      str(inputs_path),
      "--tensor-output",
      str(expected_path),
      str(source_path),
    ]
  )
  _run(
    [
      str(symirc),
      "--tensor-profile-version",
      "v4",
      "--target",
      "triton",
      "--entry",
      "kernel",
      "-o",
      str(kernel_path),
      str(source_path),
    ]
  )

  inputs = inputs_path.read_bytes()
  expected = expected_path.read_bytes()
  kernel = kernel_path.read_bytes()
  if inputs != INPUT_BYTES:
    raise ValueError("authored #11986 input bytes changed during fixture generation")
  if expected != struct.pack("<2I", 0x80000000, 0x00000000):
    raise ValueError("independent reference returned unexpected signed-zero bits")
  records = {
    "program.sir": source_bytes,
    "inputs.bin": inputs,
    "expected.bin": expected,
    "kernel.py": kernel,
  }
  manifest = {
    "schema": "refractir.tensor-case.v4",
    "profile": "tensor-v4",
    "entry": "kernel",
    "source": {
      "file": "program.sir",
      "sha256": _digest(source_bytes)["sha256"],
      "generator": "authored-calibration",
      "seed": 0,
    },
    "inputs": [
      _buffer("x", "inputs.bin", 0, 8),
      _buffer("y", "inputs.bin", 8, 8),
    ],
    "output": _buffer("return", "expected.bin", 0, 8),
    "grid": [1, 1, 1],
    "softmax_lowering": "direct",
    "launch": {"num_warps": 1, "num_stages": 1, "enable_fp_fusion": False},
    "property": {
      "id": "defined-inputs",
      "source": "Authored signed-zero remainder calibration fixture.",
      "P": "defined-inputs",
      "D": "untargeted",
      "validated": True,
    },
    "evidence": {
      "kind": "authored_calibration",
      "status": "validated",
      "artifacts": ["program.sir", "inputs.bin", "expected.bin", "kernel.py"],
    },
    "comparison": {
      "mode": "exact",
      "atol": [0, 1],
      "rtol": [0, 1],
      "zero_sign_exact": True,
    },
    "reference": {"defined": True, "stable": True, "precisions": []},
    "execution": {
      "full_shapes": [[2], [2], [2]],
      "range_extents": [[2], [2], [2]],
      "warp_probe": {"status": "not_probed", "num_warps": 1},
      "loop_bounds": [],
      "total_body_executions": 0,
      "gap_sentinel_by_dtype": {"i32": "55555555", "f32": "3f000001"},
      "output_canary_by_dtype": {"i32": "aaaaaaaa", "f32": "7fc00002"},
    },
    "coverage": {
      "primary_stratum": "authored-calibration",
      "source_signature": source_signature(SOURCE),
      "loop_structure": [],
      "input_behavior": {
        "property_passed": True,
        "pattern": "signed-zero",
        "input_mode": "authored-calibration",
        "distinguishing_objective": "untargeted",
        "counterfactual_inputs": [],
      },
    },
    "environment": None,
    "operator_coverage": ["rem"],
    "shape_axes": [],
    "generator": {
      "version": "authored-calibration",
      "seed": 0,
      "input_mode": "authored-calibration",
      "input_pattern": "signed-zero",
      "synthesis": "construction",
      "primary_stratum": "authored-calibration",
      "node_limit": 12,
    },
    "files": {name: _digest(data) for name, data in records.items()},
  }
  (output_dir / "case.json").write_text(
    json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
  )
  return output_dir


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", required=True)
  args = parser.parse_args(argv)
  build_case(args.output_dir)
  print(
    json.dumps(
      {"case": str(Path(args.output_dir).resolve()), "kind": "authored_calibration"}
    )
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
