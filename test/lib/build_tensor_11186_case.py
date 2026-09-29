"""Freeze the authored PR #11186 SIR, input, and independent reference bundle."""

import argparse
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "test" / "tensor" / "regressions" / "triton-11186"
DEFAULT_SYMIRI = Path(os.environ.get("REFRACTIR_SYMIRI", ROOT / "symiri")).resolve()
INPUT_SHAPE = [256, 2]
OUTPUT_SHAPE = [256]


def _canonical_json(value):
  return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _file_record(path):
  data = path.read_bytes()
  return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _run(command):
  result = subprocess.run(
    command, capture_output=True, text=True, timeout=60, check=False
  )
  if result.returncode:
    raise RuntimeError(
      "command failed: "
      + " ".join(map(str, command))
      + "\n"
      + result.stdout
      + result.stderr
    )
  return result


def _strides(shape):
  result = [0] * len(shape)
  stride = 1
  for axis in range(len(shape) - 1, -1, -1):
    result[axis] = stride
    stride *= shape[axis]
  return result


def _layout(name, shape, strides, span, axis_order):
  return {
    "name": name,
    "shape": shape,
    "element_strides": strides,
    "element_offset": 0,
    "storage_span": span,
    "axis_order": axis_order,
    "tile_shape": shape,
    "tile_origin": [0] * len(shape),
  }


def build_case(output_dir=FIXTURE, *, symiri=DEFAULT_SYMIRI):
  """Regenerate canonical artifacts and hashes without changing the source graph."""
  output_dir = Path(output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  if output_dir != FIXTURE.resolve():
    for name in ("program.sir", "layout.json", "upstream_reproducer.py"):
      source = FIXTURE / name
      if source.is_file():
        (output_dir / name).write_bytes(source.read_bytes())
  source_path = output_dir / "program.sir"
  layout_path = output_dir / "layout.json"
  values = [float(row) for row in range(256) for _column in range(2)]
  inputs = struct.pack(f"<{len(values)}f", *values)
  (output_dir / "inputs.bin").write_bytes(inputs)

  describe = json.loads(
    _run(
      [
        str(symiri),
        "--tensor-profile-version",
        "v2",
        "--entry",
        "kernel",
        "--tensor-describe",
        str(source_path),
      ]
    ).stdout
  )
  described_inputs = [
    {key: record.get(key) for key in ("name", "dtype", "shape", "byte_length")}
    for record in describe.get("inputs", [])
  ]
  described_output = {
    key: describe.get("output", {}).get(key)
    for key in ("name", "dtype", "shape", "byte_length")
  }
  if described_inputs != [
    {"name": "x", "dtype": "f32", "shape": INPUT_SHAPE, "byte_length": len(inputs)}
  ] or described_output != {
    "name": "return",
    "dtype": "f32",
    "shape": OUTPUT_SHAPE,
    "byte_length": 4 * OUTPUT_SHAPE[0],
  }:
    raise ValueError("PR #11186 source signature differs from the frozen fixture")

  layouts = {
    "buffers": [
      {
        "name": "x",
        "element_strides": [1, 256],
        "element_offset": 0,
        "storage_span": 512,
        "axis_order": [0, 1],
      },
      {
        "name": "return",
        "element_strides": [1],
        "element_offset": 0,
        "storage_span": 256,
        "axis_order": [0],
      },
    ]
  }
  layout_path.write_text(_canonical_json(layouts), encoding="utf-8")
  expected_path = output_dir / "expected.bin"
  _run(
    [
      str(symiri),
      "--tensor-profile-version",
      "v2",
      "--entry",
      "kernel",
      "--tensor-input",
      str(output_dir / "inputs.bin"),
      "--tensor-output",
      str(expected_path),
      str(source_path),
    ]
  )

  input_record = _layout("x", INPUT_SHAPE, [1, 256], 512, [0, 1]) | {
    "dtype": "f32",
    "file": "inputs.bin",
    "byte_offset": 0,
    "byte_length": len(inputs),
  }
  output_record = _layout("return", OUTPUT_SHAPE, [1], 256, [0]) | {
    "dtype": "f32",
    "file": "expected.bin",
    "byte_offset": 0,
    "byte_length": 4 * OUTPUT_SHAPE[0],
  }
  files = {
    name: _file_record(output_dir / name)
    for name in (
      "program.sir",
      "inputs.bin",
      "expected.bin",
      "layout.json",
      "upstream_reproducer.py",
    )
  }
  manifest = {
    "schema": "refractir.tensor-case.v2",
    "profile": "tensor-v2",
    "entry": "kernel",
    "seed": 11186,
    "generator_revision": "authored-triton-11186-calibration",
    "files": files,
    "inputs": [input_record],
    "output": output_record,
    "grid": [1, 1, 1],
    "softmax_lowering": "direct",
    "launch": {"num_warps": 1, "num_stages": 1, "enable_fp_fusion": False},
    "execution": {
      "gap_sentinel_by_dtype": {"i32": "5a5a5a5a", "f32": "7fc00000"},
      "output_canary_by_dtype": {"i32": "a5a5a5a5", "f32": "7fc00001"},
    },
    "property": {
      "id": "historical.triton-11186",
      "source": "The strided [256,2] reduction executes twice inside a counted loop and updates a tensor accumulator.",
    },
    "evidence": {
      "kind": "historical_regression",
      "status": "validated",
      "artifacts": ["upstream_reproducer.py", "layout.json"],
    },
    "comparison": {
      "mode": "numeric",
      "policy": "numeric-v1",
      "atol": [2, 100000],
      "rtol": [2, 1000],
    },
    "reference": {"precisions": [128, 256], "stable": True},
    "environment": None,
    "coverage": {
      "primary_stratum": "authored-calibration",
      "input_behavior": {"groups_distinct": True, "input_pattern": "x[row,col]=row"},
    },
  }
  (output_dir / "case.json").write_text(_canonical_json(manifest), encoding="utf-8")
  return output_dir


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", default=str(FIXTURE))
  parser.add_argument("--symiri", default=str(DEFAULT_SYMIRI))
  args = parser.parse_args(argv)
  case_dir = build_case(args.output_dir, symiri=args.symiri)
  print(json.dumps({"case": str(case_dir), "status": "built"}, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
