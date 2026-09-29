"""Materialize the hand-written tensor-v1 smoke corpus from `.sir.in` fixtures."""

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from test.lib.tensor_case_manifest import record_buffer, write_v1_manifest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "test" / "tensor" / "cases"
SYMIRI = Path(os.environ.get("REFRACTIR_SYMIRI", ROOT / "symiri")).resolve()


def _pack(dtype, values, *, words=False):
  if dtype == "i32":
    return struct.pack(f"<{len(values)}i", *values)
  if words:
    return struct.pack(f"<{len(values)}I", *values)
  if not all(math.isfinite(value) for value in values):
    raise ValueError("smoke input contains a non-finite value")
  return struct.pack(f"<{len(values)}f", *values)


def _record_buffer(name, dtype, shape, file_name, byte_length):
  return record_buffer(name, dtype, shape, file_name, byte_length)


def _write_manifest(
  case_dir,
  source,
  inputs,
  expected,
  signature,
  input_records,
  *,
  case_name,
  softmax_lowering,
  numeric,
):
  write_v1_manifest(
    case_dir,
    source,
    inputs,
    expected,
    signature,
    input_records,
    case_name=case_name,
    property_id=f"hand.{case_name}",
    property_source="hand-written conformance stimulus",
    evidence_kind="handwritten",
    evidence_status="validated",
    generator_revision="hand-written-tensor-v1",
    softmax_lowering=softmax_lowering,
    numeric=numeric,
  )


def _run(command):
  result = subprocess.run(
    command, capture_output=True, text=True, timeout=20, check=False
  )
  if result.returncode != 0:
    raise RuntimeError(
      "command failed: " + " ".join(command) + "\n" + result.stdout + result.stderr
    )
  return result


def build_smoke_cases(output_dir, *, symiri=SYMIRI):
  """Create reproducible runtime bundles outside the recursive `.sir` tests."""
  output_dir = Path(output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  definitions = [
    {
      "name": "coordinate",
      "fixture": "coordinate.sir.in",
      "parameters": [("i32", [7]), ("i32", [3]), ("i32", [1])],
      "numeric": False,
    },
    {
      "name": "transpose",
      "fixture": "transpose.sir.in",
      "parameters": [("i32", list(range(8)))],
      "numeric": False,
    },
    {
      "name": "singleton_broadcast",
      "fixture": "singleton_broadcast.sir.in",
      "parameters": [("i32", [10, 20]), ("i32", [1, 2, 3, 4])],
      "numeric": False,
    },
    {
      "name": "integer_sum",
      "fixture": "integer_sum.sir.in",
      "parameters": [("i32", [5, -1, 2, 4])],
      "numeric": False,
    },
    {
      "name": "softmax",
      "fixture": "softmax.sir.in",
      "parameters": [("f32", [0.0, 1.0, 2.0, 3.0, 1.0, 0.0, -1.0, -2.0])],
      "numeric": True,
      "lowerings": ["direct", "decomposed"],
    },
    {
      "name": "extract_negative_zero",
      "fixture": "extract_negative_zero.sir.in",
      "parameters": [("f32", [0x80000000, 0x00000001, 0x3F800000, 0x00000000], True)],
      "numeric": False,
    },
    {
      "name": "extract_subnormal",
      "fixture": "extract_subnormal.sir.in",
      "parameters": [("f32", [0x80000000, 0x00000001, 0x3F800000, 0x00000000], True)],
      "numeric": False,
    },
  ]
  case_names = []
  for definition in definitions:
    source = (FIXTURES / definition["fixture"]).read_bytes()
    signature_dir = Path(tempfile.mkdtemp(prefix="refractir-tensor-signature-"))
    try:
      source_path = signature_dir / "program.sir"
      source_path.write_bytes(source)
      signature = json.loads(
        _run(
          [
            str(symiri),
            "--entry",
            "kernel",
            "--tensor-describe",
            str(source_path),
          ]
        ).stdout
      )
    finally:
      import shutil

      shutil.rmtree(signature_dir)

    if len(signature["inputs"]) != len(definition["parameters"]):
      raise ValueError(f"fixture input mismatch for {definition['name']}")
    input_parts = []
    input_records = []
    offset = 0
    for input_record, value_spec in zip(signature["inputs"], definition["parameters"]):
      dtype, values, *raw_words = value_spec
      if input_record["dtype"] != dtype:
        raise ValueError(f"fixture type mismatch for {input_record['name']}")
      data = _pack(dtype, values, words=bool(raw_words and raw_words[0]))
      if len(data) != input_record["byte_length"]:
        raise ValueError(f"fixture byte length mismatch for {input_record['name']}")
      record = _record_buffer(
        input_record["name"], dtype, input_record["shape"], "inputs.bin", len(data)
      )
      record["byte_offset"] = offset
      input_records.append(record)
      input_parts.append(data)
      offset += len(data)
    inputs = b"".join(input_parts)
    lowerings = definition.get("lowerings", ["direct"])
    for lowering in lowerings:
      case_name = (
        definition["name"]
        if lowering == "direct"
        else f"{definition['name']}_{lowering}"
      )
      case_dir = output_dir / case_name
      case_dir.mkdir(parents=True, exist_ok=True)
      (case_dir / "program.sir").write_bytes(source)
      (case_dir / "inputs.bin").write_bytes(inputs)
      expected_path = case_dir / "expected.bin"
      reference_command = [
        str(symiri),
        "--entry",
        "kernel",
      ]
      if definition["numeric"]:
        reference_command.append("--tensor-reference-stability")
      reference_command.extend(
        [
          "--tensor-input",
          str(case_dir / "inputs.bin"),
          "--tensor-output",
          str(expected_path),
          str(case_dir / "program.sir"),
        ]
      )
      _run(reference_command)
      expected = expected_path.read_bytes()
      _write_manifest(
        case_dir,
        source,
        inputs,
        expected,
        signature,
        input_records,
        case_name=case_name,
        softmax_lowering=lowering,
        numeric=definition["numeric"],
      )
      case_names.append(case_name)
  return case_names


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", required=True)
  args = parser.parse_args(argv)
  print(json.dumps({"cases": build_smoke_cases(args.output_dir)}, sort_keys=True))
  return 0


if __name__ == "__main__":
  sys.exit(main())
