"""Writers for the strict tensor-v1 case manifest."""

from __future__ import annotations

import json
from pathlib import Path


def fnv1a64(data: bytes) -> str:
  value = 14695981039346656037
  for byte in data:
    value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
  return f"{value:016x}"


def record_buffer(name, dtype, shape, file_name, byte_length):
  stride = 1
  strides = [0] * len(shape)
  for axis in range(len(shape) - 1, -1, -1):
    strides[axis] = stride
    stride *= shape[axis]
  return {
    "name": name,
    "dtype": dtype,
    "shape": shape,
    "file": file_name,
    "byte_offset": 0,
    "byte_length": byte_length,
    "element_strides": strides,
    "element_offset": 0,
    "tile_shape": shape,
    "tile_origin": [0] * len(shape),
  }


def write_v1_manifest(
  case_dir,
  source: bytes,
  inputs: bytes,
  expected: bytes,
  signature: dict,
  input_records: list[dict],
  *,
  case_name: str,
  property_id: str,
  property_source: str,
  evidence_kind: str,
  evidence_status: str,
  generator_revision: str,
  seed: int = 0,
  softmax_lowering: str = "direct",
  numeric: bool = False,
) -> dict:
  """Write a v1 case manifest without embedding assumptions about provenance."""
  if evidence_kind == "exact_smt" and evidence_status != "exact_smt_checked":
    raise ValueError("exact_smt evidence must be marked exact_smt_checked")
  if evidence_kind == "construction" and evidence_status != "construction_checked":
    raise ValueError("construction evidence must be marked construction_checked")
  if evidence_kind == "handwritten" and evidence_status != "validated":
    raise ValueError("handwritten evidence must be marked validated")
  if evidence_kind not in {"exact_smt", "construction", "handwritten"}:
    raise ValueError(f"unsupported tensor-v1 evidence kind {evidence_kind!r}")
  if softmax_lowering not in {"direct", "decomposed"}:
    raise ValueError("softmax_lowering must be direct or decomposed")

  files = {
    name: {"bytes": len(data), "fnv1a64": fnv1a64(data)}
    for name, data in (
      ("program.sir", source),
      ("inputs.bin", inputs),
      ("expected.bin", expected),
    )
  }
  output_description = signature["output"]
  output = record_buffer(
    output_description["name"],
    output_description["dtype"],
    output_description["shape"],
    "expected.bin",
    output_description["byte_length"],
  )
  manifest = {
    "schema": "refractir.tensor-case.v1",
    "profile": "tensor-v1",
    "entry": signature["entry"],
    "seed": seed,
    "generator_revision": generator_revision,
    "files": files,
    "inputs": input_records,
    "output": output,
    "grid": [1, 1, 1],
    "softmax_lowering": softmax_lowering,
    "launch": {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False},
    "property": {"id": property_id, "source": property_source},
    "evidence": {"kind": evidence_kind, "status": evidence_status, "artifacts": []},
    "comparison": {
      "mode": "numeric" if numeric else "exact",
      "policy": "numeric-v1" if numeric else "bit-exact",
      "atol": [2, 100000] if numeric else [0, 1],
      "rtol": [2, 1000] if numeric else [0, 1],
    },
    "reference": {"precisions": [128, 256] if numeric else [], "stable": True},
    "environment": None,
  }
  path = Path(case_dir) / "case.json"
  path.write_text(
    json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
  )
  return manifest
