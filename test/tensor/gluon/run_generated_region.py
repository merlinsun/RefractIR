#!/usr/bin/env python3
"""Run an exact generated tensor-region case through a cross-lane Gluon layout."""

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
import time
from pathlib import Path

import torch
import triton

from test.lib import tensor_case, tensor_exact_f32
from test.lib.tensor_compiler_observation import observe_compiler_ir
from test.lib.tensor_gluon import BlockedLayoutPlan
from test.lib.tensor_gluon_lowering import (
  cross_lane_threads_per_warp,
  render_gluon_kernel,
)

TRITON_VERSION = "3.8.0"
WARMUP_LAUNCHES = 3
MEASURED_LAUNCHES = 30


def _word_hex(data):
  return [f"{word:08x}" for word in struct.unpack(f"<{len(data) // 4}I", data)]


def _load_kernel(generated, source_path):
  source_path.write_text(generated.source, encoding="utf-8")
  module_name = (
    f"generated_gluon_{hashlib.sha256(generated.source.encode()).hexdigest()}"
  )
  spec = importlib.util.spec_from_file_location(module_name, source_path)
  if spec is None or spec.loader is None:
    raise RuntimeError("cannot load the generated Gluon source")
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module.generated_region_kernel


def run(case_directory, output_path, ir_directory, *, symiri=None):
  if triton.__version__ != TRITON_VERSION:
    raise RuntimeError(f"expected Triton {TRITON_VERSION}, found {triton.__version__}")
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA device is unavailable")
  device_name = torch.cuda.get_device_name()
  if "V100" not in device_name:
    raise RuntimeError(f"expected a V100 for the cross-lane pilot, found {device_name}")

  case = tensor_case.load_case(
    case_directory, symiri=symiri or tensor_case.DEFAULT_SYMIRI
  )
  proof = case.manifest.get("reference", {}).get("exactness", {})
  if proof.get("kind") != tensor_exact_f32.REGION_KIND:
    raise ValueError("Gluon lowering requires a certified generated-region case")
  source = case.program.decode("utf-8")
  input_shape, iterations, tensor_types, nodes, _updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  if case.manifest["inputs"][0]["shape"] != list(input_shape):
    raise ValueError("case input shape differs from the generated tensor region")
  if case.manifest["output"]["shape"] != list(input_shape):
    raise ValueError("case output shape differs from the generated tensor region")
  derived_proof, expected_bytes = tensor_exact_f32.certify(
    source, case.inputs, proof["input_domain"]
  )
  if expected_bytes != case.expected:
    raise ValueError("case output differs from the independent exact region result")
  reduction_axes = {
    attribute if attribute >= 0 else attribute + len(input_shape)
    for destination, operation, operands, attribute in nodes
    if operation == "sum" and tensor_types[operands[0]] == input_shape
  }
  if not reduction_axes:
    raise ValueError("generated region has no full-shape reduction to distribute")

  threads_per_warp = cross_lane_threads_per_warp(input_shape, reduction_axes)
  compute_threads_per_warp = (32, 1) if threads_per_warp == (1, 32) else (1, 32)
  compute_order = (0, 1) if compute_threads_per_warp == (32, 1) else (1, 0)
  reduction_order = (1, 0) if threads_per_warp == (1, 32) else (0, 1)
  compute_layout = BlockedLayoutPlan(
    "compute-blocked",
    (1, 1),
    compute_threads_per_warp,
    (1, 1),
    compute_order,
  )
  reduction_layout = BlockedLayoutPlan(
    "reduction-blocked-cross-lane",
    (1, 1),
    threads_per_warp,
    (1, 1),
    reduction_order,
  )
  for plan in (compute_layout, reduction_layout):
    plan.validate(input_shape, num_warps=1)
  generated = render_gluon_kernel(source)
  ir_directory = Path(ir_directory)
  ir_directory.mkdir(parents=True, exist_ok=True)
  kernel = _load_kernel(generated, ir_directory / "generated-region-kernel.py")

  values = struct.unpack(f"<{math.prod(input_shape)}f", case.inputs)
  input_tensor = torch.tensor(values, dtype=torch.float32, device="cuda").reshape(
    input_shape
  )
  output_tensor = torch.empty(input_shape, dtype=torch.float32, device="cuda")
  gluon_compute_layout = compute_layout.to_gluon_layout()
  gluon_reduction_layout = reduction_layout.to_gluon_layout()
  started = time.perf_counter()
  compiled = kernel[(1,)](
    input_tensor,
    output_tensor,
    gluon_compute_layout,
    gluon_compute_layout,
    gluon_reduction_layout,
    gluon_compute_layout,
    num_warps=1,
    enable_fp_fusion=False,
  )
  torch.cuda.synchronize()
  compilation_wall_s = time.perf_counter() - started
  actual_bytes = output_tensor.view(torch.uint8).cpu().numpy().tobytes()
  if actual_bytes != case.expected:
    raise AssertionError(
      "generated Gluon output differs from complete exact f32 bits: "
      f"actual={_word_hex(actual_bytes)}, expected={_word_hex(case.expected)}"
    )

  for _ in range(WARMUP_LAUNCHES):
    kernel[(1,)](
      input_tensor,
      output_tensor,
      gluon_compute_layout,
      gluon_compute_layout,
      gluon_reduction_layout,
      gluon_compute_layout,
      num_warps=1,
      enable_fp_fusion=False,
    )
  torch.cuda.synchronize()
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)
  start_event.record()
  for _ in range(MEASURED_LAUNCHES):
    kernel[(1,)](
      input_tensor,
      output_tensor,
      gluon_compute_layout,
      gluon_compute_layout,
      gluon_reduction_layout,
      gluon_compute_layout,
      num_warps=1,
      enable_fp_fusion=False,
    )
  end_event.record()
  torch.cuda.synchronize()
  median_event_ms = start_event.elapsed_time(end_event) / MEASURED_LAUNCHES

  assembly = getattr(compiled, "asm", {})
  ir_files = {}
  compiler_stages = {}
  for stage in ("ttir", "ttgir", "llir", "ptx", "sass"):
    content = assembly.get(stage)
    if isinstance(content, str) and content.strip():
      ir_path = ir_directory / f"generated-region.{stage}"
      ir_path.write_text(content, encoding="utf-8")
      ir_files[stage] = str(ir_path)
      compiler_stages[stage] = content
  compiler_observation = observe_compiler_ir(
    compiler_stages,
    source_lines=generated.conversion_source_lines,
  )
  if compiler_observation["layout_conversion_count"] == 0:
    raise AssertionError("Gluon lowering retained no effective layout conversion")
  device_stage = compiler_observation["stages"].get(
    "ptx", compiler_observation["stages"].get("sass", {})
  )
  shuffle_count = device_stage.get("warp_shuffle_count", 0)
  shared_transfer_count = device_stage.get("shared_memory_transfer_count", 0)
  if shuffle_count + shared_transfer_count == 0:
    raise AssertionError("cross-lane reduction emitted no shuffle or shared transfer")

  report = {
    "status": "pass",
    "case_path": str(Path(case_directory).resolve()),
    "case_source_sha256": hashlib.sha256(case.program).hexdigest(),
    "input_sha256": hashlib.sha256(case.inputs).hexdigest(),
    "expected_output_bits": _word_hex(case.expected),
    "actual_output_bits": _word_hex(actual_bytes),
    "proof_kind": derived_proof["kind"],
    "proof_bound": derived_proof["integer_exactness_bound"],
    "iterations": iterations,
    "shape": list(input_shape),
    "reduction_axes": sorted(reduction_axes),
    "device_name": device_name,
    "compute_capability": list(torch.cuda.get_device_capability()),
    "triton_version": triton.__version__,
    "torch_version": torch.__version__,
    "layouts": {
      "load": compute_layout.to_dict(),
      "compute": compute_layout.to_dict(),
      "reduction": reduction_layout.to_dict(),
      "store": compute_layout.to_dict(),
    },
    "compiler_observation": compiler_observation,
    "cross_lane_evidence": {
      "warp_shuffle_count": shuffle_count,
      "shared_memory_transfer_count": shared_transfer_count,
      "status": "observed",
    },
    "measurement": {
      "num_warps": 1,
      "enable_fp_fusion": False,
      "warmup_launches": WARMUP_LAUNCHES,
      "measured_launches": MEASURED_LAUNCHES,
      "first_launch_wall_s_including_compilation": compilation_wall_s,
      "mean_gpu_event_ms_per_launch": median_event_ms,
    },
    "ir_files": ir_files,
  }
  Path(output_path).write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--case-dir", required=True)
  parser.add_argument("--output", default="generated-region-gluon.json")
  parser.add_argument("--ir-dir", default="generated-region-gluon-ir")
  parser.add_argument("--symiri", default=None)
  args = parser.parse_args()
  report = run(args.case_dir, args.output, args.ir_dir, symiri=args.symiri)
  print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
  main()
