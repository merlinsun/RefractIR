#!/usr/bin/env python3
"""Compare two Gluon layouts on the certified composed exact-f32 loop."""

import argparse
import hashlib
import json
import re
import statistics
import struct
import time
from pathlib import Path

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from test.lib.tensor_exact_f32 import certify, render_program
from test.lib.tensor_gluon import BlockedLayoutPlan

TRITON_VERSION = "3.8.0"
INPUT_VALUES = (1.0, 2.0, 4.0, 8.0)
DOMAIN = (0, 8)
ITERATIONS = 2
WARMUP_LAUNCHES = 3
MEASURED_LAUNCHES = 30


@gluon.jit
def _composed_exact_kernel(
  input_ptr,
  output_ptr,
  load_layout: gl.constexpr,
  compute_layout: gl.constexpr,
  reduction_layout: gl.constexpr,
  store_layout: gl.constexpr,
  iterations: gl.constexpr,
):
  load_rows = gl.arange(0, 2, layout=gl.SliceLayout(1, parent=load_layout))
  load_columns = gl.arange(0, 2, layout=gl.SliceLayout(0, parent=load_layout))
  load_offsets = load_rows[:, None] * 2 + load_columns[None, :]
  value = gl.convert_layout(gl.load(input_ptr + load_offsets), compute_layout)
  state_a = value
  state_b = gl.zeros((2, 2), gl.float32, layout=compute_layout)

  for _ in range(iterations):
    reduced = gl.convert_layout(state_a + state_b, reduction_layout)
    row_totals = gl.sum(reduced, axis=1)
    column_totals = gl.sum(reduced, axis=0)
    row_tile = gl.convert_layout(gl.expand_dims(row_totals, 1), compute_layout)
    column_tile = gl.convert_layout(gl.expand_dims(column_totals, 0), compute_layout)
    row_tile, _ = gl.broadcast(row_tile, state_a)
    column_tile, _ = gl.broadcast(column_tile, state_b)

    next_a = state_a + row_tile
    next_b = state_b + column_tile
    state_a = next_a
    state_b = next_b

  output = gl.convert_layout(state_a + state_b, store_layout)
  store_rows = gl.arange(0, 2, layout=gl.SliceLayout(1, parent=store_layout))
  store_columns = gl.arange(0, 2, layout=gl.SliceLayout(0, parent=store_layout))
  store_offsets = store_rows[:, None] * 2 + store_columns[None, :]
  gl.store(output_ptr + store_offsets, output)


def _layouts():
  row = BlockedLayoutPlan(
    "row-major",
    (1, 1),
    (1, 32),
    (1, 1),
    (1, 0),
  )
  column = BlockedLayoutPlan(
    "column-major",
    (1, 1),
    (32, 1),
    (1, 1),
    (0, 1),
  )
  return (
    {
      "name": row.name,
      "load": row,
      "compute": row,
      "reduction": column,
      "store": row,
    },
    {
      "name": column.name,
      "load": column,
      "compute": column,
      "reduction": row,
      "store": column,
    },
  )


def _word_hex(data):
  return [f"{word:08x}" for word in struct.unpack("<4I", data)]


def _capture_compiler_output(compiled, layout_name, ir_directory):
  assembly = getattr(compiled, "asm", {})
  metadata = getattr(compiled, "metadata", None)
  shared_memory = (
    metadata.get("shared")
    if isinstance(metadata, dict)
    else getattr(metadata, "shared", None)
  )
  ir_files = {}
  for name in ("ttir", "ttgir", "llir", "ptx", "sass"):
    content = assembly.get(name)
    if isinstance(content, str) and content:
      path = ir_directory / f"{layout_name}.{name}"
      path.write_text(content)
      ir_files[name] = str(path)
  ttgir = assembly.get("ttgir", "")
  ptx = assembly.get("ptx", "")
  layout_conversions = len(
    re.findall(r"(?:ttg\.)?convert_layout|convert_layout", ttgir)
  )
  return {
    "ir_files": ir_files,
    "layout_conversion_operations_in_ttgir": layout_conversions,
    "warp_shuffle_operations_in_ptx": len(re.findall(r"\bshfl\.", ptx)),
    "shared_memory_transfers_in_ptx": len(re.findall(r"\b(?:ld|st)\.shared", ptx)),
    "num_warps": getattr(compiled, "n_warps", None),
    "registers_per_thread": getattr(compiled, "n_regs", None),
    "spill_bytes": getattr(compiled, "n_spills", None),
    "shared_memory_bytes": shared_memory,
  }


def _measure_launches(kernel, args):
  start_wall = time.perf_counter()
  compiled = kernel[(1,)](*args, num_warps=1)
  torch.cuda.synchronize()
  first_launch_wall_s = time.perf_counter() - start_wall
  for _ in range(WARMUP_LAUNCHES):
    kernel[(1,)](*args, num_warps=1)
  torch.cuda.synchronize()

  samples_ms = []
  for _ in range(3):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(MEASURED_LAUNCHES // 3):
      kernel[(1,)](*args, num_warps=1)
    end.record()
    torch.cuda.synchronize()
    samples_ms.append(start.elapsed_time(end) / (MEASURED_LAUNCHES // 3))
  return compiled, first_launch_wall_s, statistics.median(samples_ms)


def run(output_path, ir_directory):
  if triton.__version__ != TRITON_VERSION:
    raise RuntimeError(f"expected Triton {TRITON_VERSION}, found {triton.__version__}")
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA device is unavailable")
  device_name = torch.cuda.get_device_name()
  if "V100" not in device_name:
    raise RuntimeError(
      f"expected a V100 for the pinned layout pilot, found {device_name}"
    )
  source = render_program()
  input_bytes = struct.pack("<4f", *INPUT_VALUES)
  proof, expected_bytes = certify(source, input_bytes, DOMAIN)
  input_tensor = torch.tensor(INPUT_VALUES, dtype=torch.float32, device="cuda").reshape(
    2, 2
  )
  output_tensor = torch.empty_like(input_tensor)
  ir_directory.mkdir(parents=True, exist_ok=True)

  cases = {}
  for case in _layouts():
    layouts = {key: case[key] for key in ("load", "compute", "reduction", "store")}
    for plan in layouts.values():
      plan.validate((2, 2), num_warps=1)
    args = (
      input_tensor,
      output_tensor,
      layouts["load"].to_gluon_layout(),
      layouts["compute"].to_gluon_layout(),
      layouts["reduction"].to_gluon_layout(),
      layouts["store"].to_gluon_layout(),
      ITERATIONS,
    )
    compiled, first_launch_wall_s, median_launch_ms = _measure_launches(
      _composed_exact_kernel, args
    )
    actual_bytes = output_tensor.view(torch.uint8).cpu().numpy().tobytes()
    if actual_bytes != expected_bytes:
      raise AssertionError(
        f"{case['name']} output differs from the certified full-output f32 bits: "
        f"actual={_word_hex(actual_bytes)}, expected={_word_hex(expected_bytes)}"
      )
    compiler = _capture_compiler_output(compiled, case["name"], ir_directory)
    if not (
      compiler["layout_conversion_operations_in_ttgir"]
      or compiler["warp_shuffle_operations_in_ptx"]
      or compiler["shared_memory_transfers_in_ptx"]
    ):
      raise AssertionError(
        f"{case['name']} produced no retained layout conversion or communication"
      )
    cases[case["name"]] = {
      "status": "pass",
      "layouts": {key: plan.to_dict() for key, plan in layouts.items()},
      "actual_output_bits": _word_hex(actual_bytes),
      "expected_output_bits": _word_hex(expected_bytes),
      "measurement": {
        "warmup_launches": WARMUP_LAUNCHES,
        "measured_launches": MEASURED_LAUNCHES,
        "first_launch_wall_s_including_compilation": first_launch_wall_s,
        "median_gpu_event_ms_per_launch": median_launch_ms,
      },
      "compiler": compiler,
    }

  torch.cuda.synchronize()
  result = {
    "contract": "composed-add-sum-f32-exact-loop-v1",
    "triton_version": triton.__version__,
    "torch_version": torch.__version__,
    "device_name": device_name,
    "compute_capability": list(torch.cuda.get_device_capability()),
    "input_domain": list(DOMAIN),
    "input_bits_sha256": hashlib.sha256(input_bytes).hexdigest(),
    "iterations": ITERATIONS,
    "proof_bound": proof["integer_exactness_bound"],
    "proof_obligations": len(proof["bounds"]),
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "measurement_protocol": {
      "warmup_launches_per_layout": WARMUP_LAUNCHES,
      "measured_launches_per_layout": MEASURED_LAUNCHES,
      "num_warps": 1,
      "comparison": "same input bytes and logical program for each blocked layout",
    },
    "complete_output_bits": _word_hex(expected_bytes),
    "cases": cases,
  }
  Path(output_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", default="gluon-exact-loop.json")
  parser.add_argument("--ir-dir", default="gluon-ir")
  args = parser.parse_args()
  result = run(args.output, Path(args.ir_dir))
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
