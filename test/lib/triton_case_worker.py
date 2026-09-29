"""Subprocess worker for one complete Triton tensor tile."""

import argparse
import json
import runpy
import sys
import time
import traceback
from pathlib import Path

from test.lib import tensor_layout


def _write_phase(path, phase):
  path.write_text(json.dumps({"phase": phase}) + "\n", encoding="utf-8")


def _write_timing(path, timing):
  path.write_text(json.dumps(timing, sort_keys=True) + "\n", encoding="utf-8")


def _read_inputs(case_dir, manifest, *, device):
  raw = (case_dir / "inputs.bin").read_bytes()
  inputs = []
  physical_inputs = []
  profile = manifest["profile"]
  has_layout = profile in {"tensor-v2", "tensor-v3"}
  layout_profile = profile.removeprefix("tensor-")
  for record in manifest["inputs"]:
    begin = record["byte_offset"]
    end = begin + record["byte_length"]
    element = record["dtype"]
    dtype = {"i32": torch.int32, "f32": torch.float32}[element]
    host = torch.frombuffer(bytearray(raw[begin:end]), dtype=dtype).clone()
    if record["dtype"] == "f32" and not bool(torch.isfinite(host).all()):
      raise ValueError(f"input {record['name']} contains non-finite values")
    physical = raw[begin:end]
    if has_layout and record["shape"]:
      execution = manifest["execution"]
      layout = {
        "shape": record["shape"],
        "element_strides": record["element_strides"],
        "element_offset": record["element_offset"],
        "storage_span": record["storage_span"],
        "axis_order": record["axis_order"],
      }
      sentinel = bytes.fromhex(execution["gap_sentinel_by_dtype"][record["dtype"]])
      staged = tensor_layout.stage_buffer(
        physical, layout, element_size=4, gap_bytes=sentinel, profile=layout_profile
      )
      host = torch.frombuffer(bytearray(staged), dtype=dtype).clone()
      physical = staged
    inputs.append(host.to(device))
    physical_inputs.append(physical)
  return inputs, physical_inputs


def _dump_ir(compiled, artifact_dir):
  for kind, content in getattr(compiled, "asm", {}).items():
    if not isinstance(content, str):
      continue
    safe_kind = "".join(char.lower() if char.isalnum() else "_" for char in kind)
    (artifact_dir / f"compiler-{safe_kind}.ir").write_text(content, encoding="utf-8")


def execute(case_dir, mode, artifacts_dir=None):
  global torch
  worker_started = time.perf_counter()
  import torch as torch_module

  torch = torch_module
  case_dir = Path(case_dir)
  artifacts_dir = Path(artifacts_dir) if artifacts_dir else case_dir
  artifacts_dir.mkdir(parents=True, exist_ok=True)
  timing_path = artifacts_dir / "worker-timing.json"
  timing = {
    "schema": "refractir.tensor-worker-timing.v1",
    "mode": mode,
    "execution_kind": "gpu" if mode == "compiled" else "interpreter",
    "input_staging_s": 0.0,
    "compilation_s": 0.0,
    "gpu_execution_s": 0.0,
    "interpreter_execution_s": 0.0,
    "output_materialization_s": 0.0,
    "worker_total_s": 0.0,
  }

  def save_timing():
    timing["worker_total_s"] = max(0.0, time.perf_counter() - worker_started)
    _write_timing(timing_path, timing)

  try:
    manifest = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    if mode == "compiled":
      if not torch.cuda.is_available():
        raise RuntimeError("compiled mode requires a CUDA device")
      torch.cuda.set_device(0)
      device = torch.device("cuda:0")
    else:
      device = torch.device("cpu")

    stage_started = time.perf_counter()
    try:
      inputs, initial_inputs = _read_inputs(case_dir, manifest, device=device)
      output_record = manifest["output"]
      output_dtype = {"i32": torch.int32, "f32": torch.float32}[output_record["dtype"]]
      has_layout = manifest["profile"] in {"tensor-v2", "tensor-v3"}
      layout_profile = manifest["profile"].removeprefix("tensor-")
      if has_layout:
        output_layout = {
          "shape": output_record["shape"],
          "element_strides": output_record["element_strides"],
          "element_offset": output_record["element_offset"],
          "storage_span": output_record["storage_span"],
          "axis_order": output_record["axis_order"],
        }
        output_count = output_record["storage_span"]
        canary = bytes.fromhex(
          manifest["execution"]["output_canary_by_dtype"][output_record["dtype"]]
        )
        initial_output = canary * output_count
        output = (
          torch.frombuffer(bytearray(initial_output), dtype=output_dtype)
          .clone()
          .to(device)
        )
      else:
        output_count = output_record["byte_length"] // 4
        fill_value = _I32_SENTINEL if output_record["dtype"] == "i32" else float("nan")
        output = torch.full(
          (output_count,), fill_value, dtype=output_dtype, device=device
        )
    finally:
      timing["input_staging_s"] = max(0.0, time.perf_counter() - stage_started)
      save_timing()

    module = runpy.run_path(str(artifacts_dir / "kernel.py"))
    kernel = module.get("kernel")
    if kernel is None:
      raise RuntimeError("generated module does not export kernel")
    args = [*inputs, output]
    launch = manifest["launch"]
    options = {
      "num_warps": launch["num_warps"],
      "num_stages": launch["num_stages"],
      "enable_fp_fusion": launch["enable_fp_fusion"],
    }
    phase_path = artifacts_dir / "worker-phase.json"
    grid = (1, 1, 1) if has_layout else (1,)
    if mode == "compiled":
      _write_phase(phase_path, "compilation")
      compile_started = time.perf_counter()
      try:
        compiled = kernel.warmup(*args, grid=grid, **options)
        if compiled is None:
          raise RuntimeError("Triton warmup did not return a compiled kernel")
        _dump_ir(compiled, artifacts_dir)
      finally:
        timing["compilation_s"] = max(0.0, time.perf_counter() - compile_started)
        save_timing()
      _write_phase(phase_path, "runtime")

    execution_started = time.perf_counter()
    try:
      if mode != "compiled":
        _write_phase(phase_path, "runtime")
      kernel[grid](*args, **options)
      if mode == "compiled":
        torch.cuda.synchronize()
    finally:
      execution_time = max(0.0, time.perf_counter() - execution_started)
      execution_field = (
        "gpu_execution_s" if mode == "compiled" else "interpreter_execution_s"
      )
      timing[execution_field] = execution_time
      save_timing()

    if manifest["profile"] == "tensor-v3":
      for index, (input_tensor, initial) in enumerate(zip(inputs, initial_inputs)):
        actual_input = (
          input_tensor.detach()
          .to("cpu")
          .contiguous()
          .view(torch.uint8)
          .numpy()
          .tobytes()
        )
        if actual_input != initial:
          raise RuntimeError(
            f"kernel modified input allocation or gap canary for input {index}"
          )

    materialize_started = time.perf_counter()
    try:
      physical = (
        output.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
      )
      if has_layout:
        initial_bytes = initial_output
        if output_record["shape"]:
          changed = tensor_layout.changed_canary_slots(
            initial_bytes,
            physical,
            output_layout,
            element_size=4,
            profile=layout_profile,
          )
          if changed:
            raise RuntimeError(f"kernel modified output canary slots: {changed[:16]}")
          result = tensor_layout.gather_buffer(
            physical, output_layout, element_size=4, profile=layout_profile
          )
        else:
          result = physical[: output_record["byte_length"]]
        (artifacts_dir / "actual-physical.bin").write_bytes(physical)
      else:
        result = physical
      (artifacts_dir / "actual.bin").write_bytes(result)
    finally:
      timing["output_materialization_s"] = max(
        0.0, time.perf_counter() - materialize_started
      )
      save_timing()
  finally:
    save_timing()


_I32_SENTINEL = 0x5A5A5A5A


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--case", required=True)
  parser.add_argument("--artifacts-dir")
  parser.add_argument("--mode", choices=("compiled", "interpret"), required=True)
  args = parser.parse_args(argv)
  try:
    execute(args.case, args.mode, args.artifacts_dir)
  except Exception:
    print(traceback.format_exc(), file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())
