#!/usr/bin/env python3
"""Compare two Gluon layouts on the certified composed exact-f32 loop."""

import argparse
import hashlib
import inspect
import json
import statistics
import struct
import time
from pathlib import Path

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from test.lib.tensor_compiler_observation import observe_compiler_ir
from test.lib.tensor_exact_f32 import certify, render_program
from test.lib.tensor_gluon import generate_layout_plans

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
  mutate_output: gl.constexpr,
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
  if mutate_output:
    output = output + 1.0
  store_rows = gl.arange(0, 2, layout=gl.SliceLayout(1, parent=store_layout))
  store_columns = gl.arange(0, 2, layout=gl.SliceLayout(0, parent=store_layout))
  store_offsets = store_rows[:, None] * 2 + store_columns[None, :]
  gl.store(output_ptr + store_offsets, output)


def _layout_key(plan):
  return (
    plan.size_per_thread,
    plan.threads_per_warp,
    plan.warps_per_cta,
    plan.order,
  )


def _layouts():
  first, second = generate_layout_plans((2, 2), num_warps=1, seed=20261002, count=2)
  return (
    {
      "name": "blocked-a-baseline",
      "load": first,
      "compute": first,
      "reduction": first,
      "store": first,
      "expected_conversions": (),
    },
    {
      "name": "blocked-a-compute-to-reduction",
      "load": first,
      "compute": first,
      "reduction": second,
      "store": first,
      "expected_conversions": (
        "compute_to_reduction",
        "reduction_to_compute_row",
        "reduction_to_compute_column",
      ),
    },
    {
      "name": "blocked-b-compute-to-reduction",
      "load": second,
      "compute": second,
      "reduction": first,
      "store": second,
      "expected_conversions": (
        "compute_to_reduction",
        "reduction_to_compute_row",
        "reduction_to_compute_column",
      ),
    },
    {
      "name": "blocked-a-load-to-compute",
      "load": second,
      "compute": first,
      "reduction": first,
      "store": first,
      "expected_conversions": ("load_to_compute",),
    },
    {
      "name": "blocked-a-compute-to-store",
      "load": first,
      "compute": first,
      "reduction": first,
      "store": second,
      "expected_conversions": ("compute_to_store",),
    },
  )


def _conversion_source_lines():
  source_lines, first_line = inspect.getsourcelines(_composed_exact_kernel.fn)
  operation_fragments = {
    "load_to_compute": "gl.convert_layout(gl.load(input_ptr + load_offsets)",
    "compute_to_reduction": "reduced = gl.convert_layout(state_a + state_b",
    "reduction_to_compute_row": "gl.convert_layout(gl.expand_dims(row_totals",
    "reduction_to_compute_column": "gl.convert_layout(gl.expand_dims(column_totals",
    "compute_to_store": "output = gl.convert_layout(state_a + state_b, store_layout)",
  }
  result = {}
  for operation, fragment in operation_fragments.items():
    matches = [
      first_line + index for index, line in enumerate(source_lines) if fragment in line
    ]
    if len(matches) != 1:
      raise RuntimeError(f"cannot locate unique source operation {operation}")
    result[matches[0]] = operation
  return result


def _word_hex(data):
  return [f"{word:08x}" for word in struct.unpack("<4I", data)]


def _capture_compiler_output(compiled, layout_name, ir_directory, expected_conversions):
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
  compiler_stages = {
    name: content
    for name, content in assembly.items()
    if name in {"ttir", "ttgir", "llir", "ptx", "sass"}
    and isinstance(content, str)
    and content.strip()
  }
  observation = observe_compiler_ir(
    compiler_stages,
    source_lines=_conversion_source_lines(),
    expected_source_operations=expected_conversions,
    exact_source_operations=True,
    expected_loop=True,
  )
  compiler = {
    "ir_files": ir_files,
    "observation": observation,
    "num_warps": getattr(compiled, "n_warps", None),
    "registers_per_thread": getattr(compiled, "n_regs", None),
    "spill_bytes": getattr(compiled, "n_spills", None),
    "shared_memory_bytes": shared_memory,
  }
  return compiler


def _device_communication(observation):
  stages = observation["stages"]
  stage = stages.get("ptx", stages.get("sass", {}))
  return {
    "warp_shuffle_count": stage.get("warp_shuffle_count", 0),
    "shared_memory_transfer_count": stage.get("shared_memory_transfer_count", 0),
  }


def _measure_launches(kernel, args):
  start_wall = time.perf_counter()
  compiled = kernel[(1,)](*args, num_warps=1, enable_fp_fusion=False)
  torch.cuda.synchronize()
  first_launch_wall_s = time.perf_counter() - start_wall
  for _ in range(WARMUP_LAUNCHES):
    kernel[(1,)](*args, num_warps=1, enable_fp_fusion=False)
  torch.cuda.synchronize()

  samples_ms = []
  for _ in range(3):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(MEASURED_LAUNCHES // 3):
      kernel[(1,)](*args, num_warps=1, enable_fp_fusion=False)
    end.record()
    torch.cuda.synchronize()
    samples_ms.append(start.elapsed_time(end) / (MEASURED_LAUNCHES // 3))
  return compiled, first_launch_wall_s, statistics.median(samples_ms)


def run(output_path, ir_directory, *, input_values=INPUT_VALUES):
  if triton.__version__ != TRITON_VERSION:
    raise RuntimeError(f"expected Triton {TRITON_VERSION}, found {triton.__version__}")
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA device is unavailable")
  device_name = torch.cuda.get_device_name()
  if "V100" not in device_name:
    raise RuntimeError(
      f"expected a V100 for the pinned layout pilot, found {device_name}"
    )
  if (
    not isinstance(input_values, (tuple, list))
    or len(input_values) != 4
    or any(
      type(value) is not int or not DOMAIN[0] <= value <= DOMAIN[1]
      for value in input_values
    )
  ):
    raise ValueError(
      "input_values must contain four integers in the frozen [0,8] domain"
    )
  input_values = tuple(input_values)
  source = render_program()
  input_bytes = struct.pack("<4f", *input_values)
  reference_started = time.perf_counter()
  proof, expected_bytes = certify(source, input_bytes, DOMAIN)
  reference_validation_s = time.perf_counter() - reference_started
  if expected_bytes[:4] != struct.pack("<f", 63.0):
    raise ValueError("input_values do not satisfy the shared output-element property")
  input_started = time.perf_counter()
  input_tensor = torch.tensor(input_values, dtype=torch.float32, device="cuda").reshape(
    2, 2
  )
  output_tensor = torch.empty_like(input_tensor)
  input_materialization_s = time.perf_counter() - input_started
  ir_directory.mkdir(parents=True, exist_ok=True)

  cases = {}
  plan_fields = ("load", "compute", "reduction", "store")
  for case in _layouts():
    layouts = {key: case[key] for key in plan_fields}
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
      False,
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
    compiler = _capture_compiler_output(
      compiled,
      case["name"],
      ir_directory,
      case["expected_conversions"],
    )
    cases[case["name"]] = {
      "status": "pass",
      "layouts": {key: plan.to_dict() for key, plan in layouts.items()},
      "layout_keys": {key: _layout_key(plan) for key, plan in layouts.items()},
      "expected_conversions": list(case["expected_conversions"]),
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

  baseline_name = "blocked-a-baseline"
  baseline_communication = _device_communication(
    cases[baseline_name]["compiler"]["observation"]
  )
  for name, case in cases.items():
    communication = _device_communication(case["compiler"]["observation"])
    case["compiler"]["communication_vs_baseline"] = {
      "baseline_case": baseline_name,
      "baseline": baseline_communication,
      "observed": communication,
      "delta": {
        key: communication[key] - baseline_communication[key] for key in communication
      },
      "effective_conversion_operations": case["compiler"]["observation"][
        "effective_conversion_operations"
      ],
    }

  calibration_reference_started = time.perf_counter()
  calibration_input_bytes = struct.pack("<4f", *INPUT_VALUES)
  _, calibration_expected_bytes = certify(source, calibration_input_bytes, DOMAIN)
  reference_validation_s += time.perf_counter() - calibration_reference_started
  calibration_input_started = time.perf_counter()
  calibration_input = torch.tensor(
    INPUT_VALUES, dtype=torch.float32, device="cuda"
  ).reshape(2, 2)
  calibration_output = torch.empty_like(calibration_input)
  input_materialization_s += time.perf_counter() - calibration_input_started
  baseline_layouts = _layouts()[0]
  calibration_args = (
    calibration_input,
    calibration_output,
    baseline_layouts["load"].to_gluon_layout(),
    baseline_layouts["compute"].to_gluon_layout(),
    baseline_layouts["reduction"].to_gluon_layout(),
    baseline_layouts["store"].to_gluon_layout(),
    ITERATIONS,
    True,
  )
  calibration_compiled, calibration_first_launch_s, calibration_median_launch_ms = (
    _measure_launches(_composed_exact_kernel, calibration_args)
  )
  calibration_actual_bytes = (
    calibration_output.view(torch.uint8).cpu().numpy().tobytes()
  )
  calibration_detected = calibration_actual_bytes != calibration_expected_bytes
  calibration_compiler = _capture_compiler_output(
    calibration_compiled,
    "calibration-add-one-output-mutant",
    ir_directory,
    (),
  )
  calibration_controls = [
    {
      "control_id": "correct-layout-baseline",
      "expected_detection": False,
      "observed_detection": cases[baseline_name]["actual_output_bits"]
      != _word_hex(expected_bytes),
    },
    {
      "control_id": "add-one-output-mutant",
      "expected_detection": True,
      "observed_detection": calibration_detected,
      "input_sha256": hashlib.sha256(calibration_input_bytes).hexdigest(),
      "expected_output_bits": _word_hex(calibration_expected_bytes),
      "actual_output_bits": _word_hex(calibration_actual_bytes),
    },
  ]

  torch.cuda.synchronize()
  result = {
    "contract": "composed-add-sum-f32-exact-loop-v1",
    "triton_version": triton.__version__,
    "torch_version": torch.__version__,
    "device_name": device_name,
    "compute_capability": list(torch.cuda.get_device_capability()),
    "input_domain": list(DOMAIN),
    "input_bits_sha256": hashlib.sha256(input_bytes).hexdigest(),
    "input_values": list(input_values),
    "iterations": ITERATIONS,
    "reference_validation_s": reference_validation_s,
    "input_materialization_s": input_materialization_s,
    "calibration_controls": calibration_controls,
    "calibration_compiler": {
      "observation": calibration_compiler["observation"],
      "ir_files": calibration_compiler["ir_files"],
      "registers_per_thread": calibration_compiler["registers_per_thread"],
      "spill_bytes": calibration_compiler["spill_bytes"],
      "measurement": {
        "warmup_launches": WARMUP_LAUNCHES,
        "measured_launches": MEASURED_LAUNCHES,
        "first_launch_wall_s_including_compilation": calibration_first_launch_s,
        "median_gpu_event_ms_per_launch": calibration_median_launch_ms,
      },
    },
    "proof_bound": proof["integer_exactness_bound"],
    "proof_obligations": len(proof["bounds"]),
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "measurement_protocol": {
      "warmup_launches_per_layout": WARMUP_LAUNCHES,
      "measured_launches_per_layout": MEASURED_LAUNCHES,
      "num_warps": 1,
      "enable_fp_fusion": False,
      "comparison": "same input bytes and logical program for each blocked layout",
      "baseline": baseline_name,
      "compiler_observation": (
        "effective layout changes are mapped to JIT source lines; PTX communication "
        "counts are compared with the all-same-layout baseline"
      ),
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
  parser.add_argument(
    "--input-values",
    default=",".join(str(int(value)) for value in INPUT_VALUES),
    help="four comma-separated integer leaves in the frozen [0,8] domain",
  )
  args = parser.parse_args()
  try:
    input_values = tuple(int(value.strip()) for value in args.input_values.split(","))
  except ValueError as error:
    parser.error(f"invalid --input-values: {error}")
  result = run(args.output, Path(args.ir_dir), input_values=input_values)
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
