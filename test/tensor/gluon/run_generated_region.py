#!/usr/bin/env python3
"""Run certified generated tensor regions through Gluon layouts."""

import argparse
import functools
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
import triton

from test.lib import tensor_case, tensor_exact_f32
from test.lib.tensor_compiler_observation import (
  observe_compiler_ir,
  observe_pass_dumps,
)
from test.lib.tensor_gluon import BlockedLayoutPlan, generate_layout_plans
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


def _with_generated_pass_observations(function):
  @functools.wraps(function)
  def wrapped(program_path, output_path, ir_directory, *args, **kwargs):
    ir_directory = Path(ir_directory)
    pass_dump_path = ir_directory / "mlir-pass-dump.txt"
    pass_dump_path.parent.mkdir(parents=True, exist_ok=True)
    pass_dump_path.unlink(missing_ok=True)
    variable_names = ("MLIR_ENABLE_DUMP", "MLIR_DUMP_PATH")
    previous = {name: os.environ.get(name) for name in variable_names}
    os.environ["MLIR_ENABLE_DUMP"] = "generated_region_kernel"
    os.environ["MLIR_DUMP_PATH"] = str(pass_dump_path)
    try:
      result = function(program_path, output_path, ir_directory, *args, **kwargs)
    finally:
      for name, value in previous.items():
        if value is None:
          os.environ.pop(name, None)
        else:
          os.environ[name] = value

    source_lines = result.pop("_conversion_source_lines")
    pass_observation = observe_pass_dumps(
      pass_dump_path.read_text(encoding="utf-8"), source_lines=source_lines
    )
    expected_names = list(result["cases"]) + ["calibration-add-one-output-mutant"]
    if pass_observation["compilation_count"] != len(expected_names):
      raise RuntimeError(
        "compiler pass dump did not contain one compilation per generated layout "
        f"and calibration case: expected {len(expected_names)}, observed "
        f"{pass_observation['compilation_count']}"
      )
    for name, compilation in zip(
      expected_names, pass_observation["compilations"], strict=True
    ):
      if name == "calibration-add-one-output-mutant":
        result["calibration_compiler"]["pass_observation"] = compilation
      else:
        result["cases"][name]["compiler"]["pass_observation"] = compilation
    result["compiler_pass_observation"] = {
      "compilation_count": pass_observation["compilation_count"],
      "snapshot_counts": [
        item["snapshot_count"] for item in pass_observation["compilations"]
      ],
      "interpretation_limit": pass_observation["interpretation_limit"],
      "dump_file": str(pass_dump_path),
    }
    Path(output_path).write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result

  return wrapped


def _measure_region_kernel(kernel, args):
  started = time.perf_counter()
  compiled = kernel[(1,)](*args, num_warps=1, enable_fp_fusion=False)
  torch.cuda.synchronize()
  compilation_wall_s = time.perf_counter() - started
  for _ in range(WARMUP_LAUNCHES):
    kernel[(1,)](*args, num_warps=1, enable_fp_fusion=False)
  torch.cuda.synchronize()
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()
  for _ in range(MEASURED_LAUNCHES):
    kernel[(1,)](*args, num_warps=1, enable_fp_fusion=False)
  end.record()
  torch.cuda.synchronize()
  median_event_ms = start.elapsed_time(end) / MEASURED_LAUNCHES
  return compiled, {
    "warmup_launches": WARMUP_LAUNCHES,
    "measured_launches": MEASURED_LAUNCHES,
    "first_launch_wall_s_including_compilation": compilation_wall_s,
    "median_gpu_event_ms_per_launch": median_event_ms,
  }


def _capture_region_compiler(
  compiled, name, ir_directory, source_lines, *, expected_loop
):
  assembly = getattr(compiled, "asm", {})
  ir_files = {}
  compiler_stages = {}
  for stage in ("ttir", "ttgir", "llir", "ptx", "sass"):
    content = assembly.get(stage)
    if isinstance(content, str) and content.strip():
      ir_path = ir_directory / f"{name}.{stage}"
      ir_path.write_text(content, encoding="utf-8")
      ir_files[stage] = str(ir_path)
      compiler_stages[stage] = content
  return {
    "ir_files": ir_files,
    "observation": observe_compiler_ir(
      compiler_stages, source_lines=source_lines, expected_loop=expected_loop
    ),
    "num_warps": getattr(compiled, "n_warps", None),
    "registers_per_thread": getattr(compiled, "n_regs", None),
    "spill_bytes": getattr(compiled, "n_spills", None),
  }


@_with_generated_pass_observations
def run_program(
  program_path,
  output_path,
  ir_directory,
  *,
  input_values,
  domain=(0, 8),
  target=None,
  target_index=(0, 0),
  layout_seed=20261002,
):
  """Compare sampled Gluon layouts on one admitted generated-region source."""
  if triton.__version__ != TRITON_VERSION:
    raise RuntimeError(f"expected Triton {TRITON_VERSION}, found {triton.__version__}")
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA device is unavailable")
  device_name = torch.cuda.get_device_name()
  if "V100" not in device_name:
    raise RuntimeError(
      f"expected a V100 for the generated-region study, found {device_name}"
    )
  if type(layout_seed) is not int or layout_seed < 0:
    raise ValueError("layout_seed must be a nonnegative integer")

  source_path = Path(program_path).resolve()
  source = source_path.read_text(encoding="utf-8")
  shape, iterations, tensor_types, nodes, _updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  count = math.prod(shape)
  if (
    not isinstance(input_values, (tuple, list))
    or len(input_values) != count
    or any(
      type(value) is not int or not domain[0] <= value <= domain[1]
      for value in input_values
    )
  ):
    raise ValueError("input_values must fit the generated shape and frozen domain")
  if (
    not isinstance(target_index, (tuple, list))
    or len(target_index) != len(shape)
    or any(
      type(value) is not int or not 0 <= value < shape[axis]
      for axis, value in enumerate(target_index)
    )
  ):
    raise ValueError("target_index is outside the generated output shape")
  input_values = tuple(input_values)
  target_index = tuple(target_index)
  input_bytes = struct.pack(f"<{count}f", *input_values)
  reference_started = time.perf_counter()
  proof, expected_bytes = tensor_exact_f32.certify(source, input_bytes, domain)
  reference_validation_s = time.perf_counter() - reference_started
  output_values = struct.unpack(f"<{count}f", expected_bytes)
  target_offset = target_index[0] * shape[1] + target_index[1]
  if target is not None and struct.pack(
    "<f", output_values[target_offset]
  ) != struct.pack("<f", float(target)):
    raise ValueError("input_values do not satisfy the requested output property")

  ir_directory = Path(ir_directory).resolve()
  ir_directory.mkdir(parents=True, exist_ok=True)
  materialization_started = time.perf_counter()
  input_tensor = torch.tensor(input_values, dtype=torch.float32, device="cuda").reshape(
    shape
  )
  output_tensor = torch.empty(shape, dtype=torch.float32, device="cuda")
  input_materialization_s = time.perf_counter() - materialization_started
  reduction_axes = {
    attribute if attribute >= 0 else attribute + len(shape)
    for _destination, operation, operands, attribute in nodes
    if operation == "sum" and tensor_types[operands[0]] == shape
  }
  if not reduction_axes:
    raise ValueError("generated region has no full-shape reduction to distribute")
  reduction_threads = cross_lane_threads_per_warp(shape, reduction_axes)
  reduction_order = (1, 0) if reduction_threads == (1, 32) else (0, 1)
  reduction_layout = BlockedLayoutPlan(
    "reduction-blocked-cross-lane",
    (1, 1),
    reduction_threads,
    (1, 1),
    reduction_order,
  )
  plans = generate_layout_plans(shape, num_warps=1, seed=layout_seed, count=5)
  generated = render_gluon_kernel(source)
  kernel = _load_kernel(generated, ir_directory / "generated-region-kernel.py")
  cases = {}
  for plan in plans:
    plan.validate(shape, num_warps=1)
    reduction_layout.validate(shape, num_warps=1)
    args = (
      input_tensor,
      output_tensor,
      plan.to_gluon_layout(),
      plan.to_gluon_layout(),
      reduction_layout.to_gluon_layout(),
      plan.to_gluon_layout(),
    )
    compiled, measurement = _measure_region_kernel(kernel, args)
    actual_bytes = output_tensor.view(torch.uint8).cpu().numpy().tobytes()
    if actual_bytes != expected_bytes:
      raise AssertionError(
        f"{plan.name} differs from complete exact output bits: "
        f"actual={_word_hex(actual_bytes)}, expected={_word_hex(expected_bytes)}"
      )
    cases[plan.name] = {
      "status": "pass",
      "layouts": {
        "load": plan.to_dict(),
        "compute": plan.to_dict(),
        "reduction": reduction_layout.to_dict(),
        "store": plan.to_dict(),
      },
      "actual_output_bits": _word_hex(actual_bytes),
      "expected_output_bits": _word_hex(expected_bytes),
      "measurement": measurement,
      "compiler": _capture_region_compiler(
        compiled,
        plan.name,
        ir_directory,
        generated.conversion_source_lines,
        expected_loop=iterations > 0,
      ),
    }

  store_line = "    gl.store(output_ptr + store_offsets, output)\n"
  if generated.source.count(store_line) != 1:
    raise RuntimeError("generated Gluon kernel has no unique output store")
  mutant = replace(
    generated,
    source=generated.source.replace(
      store_line, "    output = output + 1.0\n" + store_line
    ),
  )
  mutant_kernel = _load_kernel(
    mutant, ir_directory / "calibration-add-one-output-mutant.py"
  )
  baseline_plan = plans[0]
  mutant_args = (
    input_tensor,
    output_tensor,
    baseline_plan.to_gluon_layout(),
    baseline_plan.to_gluon_layout(),
    reduction_layout.to_gluon_layout(),
    baseline_plan.to_gluon_layout(),
  )
  mutant_compiled, mutant_measurement = _measure_region_kernel(
    mutant_kernel, mutant_args
  )
  mutant_bytes = output_tensor.view(torch.uint8).cpu().numpy().tobytes()
  mutant_detected = mutant_bytes != expected_bytes
  if not mutant_detected:
    raise AssertionError("generated-region output mutation was not detected")
  calibration_compiler = _capture_region_compiler(
    mutant_compiled,
    "calibration-add-one-output-mutant",
    ir_directory,
    mutant.conversion_source_lines,
    expected_loop=iterations > 0,
  )
  calibration_compiler["measurement"] = mutant_measurement
  baseline_detected = cases[baseline_plan.name]["actual_output_bits"] != _word_hex(
    expected_bytes
  )
  result = {
    "contract": "generated-region-exact-f32-layout-study-v1",
    "triton_version": triton.__version__,
    "torch_version": torch.__version__,
    "device_name": device_name,
    "compute_capability": list(torch.cuda.get_device_capability()),
    "program_sha256": hashlib.sha256(source.encode()).hexdigest(),
    "input_domain": list(domain),
    "input_values": list(input_values),
    "input_bits_sha256": hashlib.sha256(input_bytes).hexdigest(),
    "input_materialization_s": input_materialization_s,
    "reference_validation_s": reference_validation_s,
    "iterations": iterations,
    "shape": list(shape),
    "property": {"target": target, "index": list(target_index)},
    "layout_seed": layout_seed,
    "proof_bound": proof["integer_exactness_bound"],
    "proof_obligations": len(proof["bounds"]),
    "complete_output_bits": _word_hex(expected_bytes),
    "calibration_controls": [
      {
        "control_id": "correct-layout-baseline",
        "expected_detection": False,
        "observed_detection": baseline_detected,
      },
      {
        "control_id": "add-one-output-mutant",
        "expected_detection": True,
        "observed_detection": mutant_detected,
        "actual_output_bits": _word_hex(mutant_bytes),
        "expected_output_bits": _word_hex(expected_bytes),
      },
    ],
    "calibration_compiler": {
      "observation": calibration_compiler["observation"],
      "ir_files": calibration_compiler["ir_files"],
      "measurement": mutant_measurement,
    },
    "measurement_protocol": {
      "num_warps": 1,
      "enable_fp_fusion": False,
      "warmup_launches_per_layout": WARMUP_LAUNCHES,
      "measured_launches_per_layout": MEASURED_LAUNCHES,
      "comparison": "same exact SIR, selected input, domain, and property across layouts",
    },
    "cases": cases,
    "_conversion_source_lines": generated.conversion_source_lines,
  }
  Path(output_path).write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  return result


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
  parser.add_argument("--case-dir")
  parser.add_argument("--program")
  parser.add_argument("--output", default="generated-region-gluon.json")
  parser.add_argument("--ir-dir", default="generated-region-gluon-ir")
  parser.add_argument("--symiri", default=None)
  parser.add_argument("--input-values", default="1,2,4,8")
  parser.add_argument("--domain", default="0,8")
  parser.add_argument("--target", type=float)
  parser.add_argument("--target-index", default="0,0")
  parser.add_argument("--layout-seed", type=int, default=20261002)
  args = parser.parse_args()
  if args.program:
    try:
      input_values = tuple(int(value.strip()) for value in args.input_values.split(","))
      domain = tuple(int(value.strip()) for value in args.domain.split(","))
      target_index = tuple(int(value.strip()) for value in args.target_index.split(","))
    except ValueError as error:
      parser.error(f"invalid generated-region input, domain or target: {error}")
    if len(domain) != 2:
      parser.error("--domain must contain two integers")
    report = run_program(
      args.program,
      args.output,
      args.ir_dir,
      input_values=input_values,
      domain=domain,
      target=args.target,
      target_index=target_index,
      layout_seed=args.layout_seed,
    )
  elif args.case_dir:
    report = run(args.case_dir, args.output, args.ir_dir, symiri=args.symiri)
  else:
    parser.error("one of --case-dir or --program is required")
  print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
  main()
