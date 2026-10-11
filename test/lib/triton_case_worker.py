"""Subprocess worker for one complete Triton tensor tile."""

import argparse
import json
import os
import runpy
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from test.lib import tensor_layout

_ELEMENT_SIZE = 4
_LAYOUT_PROFILES = {"tensor-v2": "v2", "tensor-v3": "v3"}


@dataclass(frozen=True)
class Envelope:
  """The physical allocation a case asks for around one tensor."""

  guards: tensor_layout.GuardEnvelope
  padding_by_dtype: dict
  declared: bool

  def padding(self, dtype):
    return self.padding_by_dtype[dtype]

  def slots(self, record):
    return self.guards.prefix + record["storage_span"] + self.guards.suffix

  def bytes_length(self, record):
    return self.slots(record) * _ELEMENT_SIZE


def _write_phase(path, phase):
  path.write_text(json.dumps({"phase": phase}) + "\n", encoding="utf-8")


def _write_timing(path, timing):
  path.write_text(json.dumps(timing, sort_keys=True) + "\n", encoding="utf-8")


def _guard_envelope(execution):
  declared = execution.get("guard_envelope")
  if declared is None:
    return tensor_layout.GuardEnvelope(), False
  if not isinstance(declared, dict):
    raise tensor_layout.LayoutError("guard envelope must be an object")
  return (
    tensor_layout.GuardEnvelope(
      prefix=declared.get("prefix", 0), suffix=declared.get("suffix", 0)
    ),
    True,
  )


def _padding_by_dtype(execution):
  """Return the per-variant padding, defaulting to the gap sentinel.

  Guards and internal gaps share one filler, so a case that already declares a
  gap sentinel gets guards for free; explicit padding keeps the two readable
  apart when a study wants to attribute a write to one region.
  """
  declared = execution.get("padding_by_dtype")
  if declared is None:
    return dict(execution["gap_sentinel_by_dtype"])
  if not isinstance(declared, dict):
    raise tensor_layout.LayoutError("padding by dtype must be an object")
  padding = {}
  for dtype, sentinel in execution["gap_sentinel_by_dtype"].items():
    value = declared.get(dtype, sentinel)
    if not isinstance(value, str):
      raise tensor_layout.LayoutError(f"padding for {dtype} must be hexadecimal")
    padding[dtype] = value
  return padding


def _envelope(manifest):
  execution = manifest["execution"]
  guards, declared = _guard_envelope(execution)
  if declared and manifest["profile"] not in _LAYOUT_PROFILES:
    raise tensor_layout.LayoutError(
      f"a guard envelope requires a physical layout profile, not {manifest['profile']}"
    )
  return Envelope(
    guards=guards,
    padding_by_dtype=_padding_by_dtype(execution),
    declared=declared,
  )


def _no_envelope():
  """The plain allocation: no guards, no padding, no logical input check."""
  return Envelope(
    guards=tensor_layout.GuardEnvelope(), padding_by_dtype={}, declared=False
  )


def _buffer_layout(record, profile):
  return {
    "shape": record["shape"],
    "element_strides": record["element_strides"],
    "element_offset": record["element_offset"],
    "storage_span": record["storage_span"],
    "axis_order": record["axis_order"],
  }


def _read_inputs(case_dir, manifest, *, device, envelope, profile):
  """Stage every input, in an envelope when the case declares guards.

  Returns the kernel arguments, the host bytes each staged allocation started
  with, and the device buffer a write outside the tensor would land in. A
  guarded input is handed to the kernel as a view that starts at the guard
  prefix, because the recorded offsets address the tensor inside its
  allocation, while the view keeps sharing storage with the whole buffer.
  """
  raw = (case_dir / "inputs.bin").read_bytes()
  inputs = []
  physical_inputs = []
  staged_buffers = []
  for record in manifest["inputs"]:
    begin = record["byte_offset"]
    end = begin + record["byte_length"]
    element = record["dtype"]
    dtype = {"i32": torch.int32, "f32": torch.float32}[element]
    host = torch.frombuffer(bytearray(raw[begin:end]), dtype=dtype).clone()
    if record["dtype"] == "f32" and not bool(torch.isfinite(host).all()):
      raise ValueError(f"input {record['name']} contains non-finite values")
    physical = raw[begin:end]
    guarded = envelope.declared and bool(profile) and bool(record["shape"])
    if profile and record["shape"]:
      layout = _buffer_layout(record, profile)
      if envelope.declared:
        staged = tensor_layout.stage_envelope(
          physical,
          layout,
          element_size=_ELEMENT_SIZE,
          envelope=envelope.guards,
          padding=bytes.fromhex(envelope.padding(record["dtype"])),
          profile=profile,
        )
      else:
        sentinel = bytes.fromhex(
          manifest["execution"]["gap_sentinel_by_dtype"][element]
        )
        staged = tensor_layout.stage_buffer(
          physical,
          layout,
          element_size=_ELEMENT_SIZE,
          gap_bytes=sentinel,
          profile=profile,
        )
      host = torch.frombuffer(bytearray(staged), dtype=dtype).clone()
      physical = staged
    device_buffer = host.to(device)
    inputs.append(
      device_buffer[
        envelope.guards.prefix : envelope.guards.prefix + record["storage_span"]
      ]
      if guarded
      else device_buffer
    )
    physical_inputs.append(physical)
    staged_buffers.append(device_buffer)
  return inputs, physical_inputs, staged_buffers


def _initial_output(output_record, canary, envelope, *, profile):
  """Fill every nonlogical slot of the output allocation with the canary."""
  if envelope.declared and profile and output_record["shape"]:
    return canary * envelope.slots(output_record)
  return canary * output_record["storage_span"]


def _staging_record(envelope, *, execution, inputs, output):
  """Record the physical allocation so any run stays reproducible."""
  if not envelope.declared:
    return {"guard_envelope": None}
  return {
    "guard_envelope": {
      "prefix": envelope.guards.prefix,
      "suffix": envelope.guards.suffix,
    },
    "padding_by_dtype": envelope.padding_by_dtype,
    "gap_sentinel_by_dtype": execution["gap_sentinel_by_dtype"],
    "output_canary_by_dtype": execution["output_canary_by_dtype"],
    "buffers": [
      {
        "role": role,
        "name": record["name"],
        "storage_span": record["storage_span"],
        "physical_length": envelope.bytes_length(record) if record["shape"] else None,
      }
      for role, records in (("input", inputs), ("output", [output]))
      for record in records
    ],
  }


def _host_bytes(tensor):
  """Copy one device tensor back as raw bytes."""
  return tensor.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()


def _enveloped_changes(before, after, layout, envelope, *, profile):
  """Name every changed slot, in the order the physical buffer is laid out.

  A logical slot (`input-N`) carries element N of the tensor, so it may hold a
  result; a guard or gap slot lies outside every recorded address, so a change
  there has left the allocation.
  """
  return list(
    tensor_layout.classify_changed_slots(
      before,
      after,
      layout,
      element_size=_ELEMENT_SIZE,
      envelope=envelope,
      profile=profile,
    )
  )


def _modified_input_slots(record, before, after, envelope, profile):
  """Name the input slots a kernel has no right to touch.

  The whole allocation is compared, so a write into an input guard or gap is
  reported with the same slot vocabulary as an output write.
  """
  if not profile or not record["shape"]:
    return []
  return _enveloped_changes(
    before,
    after,
    _buffer_layout(record, profile),
    envelope,
    profile=profile,
  )


def _dump_ir(compiled, artifact_dir):
  for kind, content in getattr(compiled, "asm", {}).items():
    if not isinstance(content, str):
      continue
    safe_kind = "".join(char.lower() if char.isalnum() else "_" for char in kind)
    (artifact_dir / f"compiler-{safe_kind}.ir").write_text(content, encoding="utf-8")


def configure_pass_dumps(artifact_dir):
  """Route MLIR pass dumps into one file per worker compilation.

  A worker process compiles a single kernel, so the dump file cannot mix
  compilations. Dumping stays opt-in because it serializes the pass pipeline.
  """
  if os.environ.get("REFRACTIR_TRITON_PASS_DUMP") not in {"1", "true", "True"}:
    return None
  path = Path(artifact_dir) / "compiler-pass-dump.txt"
  os.environ["MLIR_ENABLE_DUMP"] = "1"
  os.environ["MLIR_DUMP_PATH"] = str(path)
  return path


def execute(case_dir, mode, artifacts_dir=None):
  """Run one case, staging and validating its guard envelope when declared."""
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
      envelope = _envelope(manifest) if "execution" in manifest else _no_envelope()
      profile = _LAYOUT_PROFILES.get(manifest["profile"])
      inputs, initial_inputs, staged_inputs = _read_inputs(
        case_dir, manifest, device=device, envelope=envelope, profile=profile
      )
      output_record = manifest["output"]
      output_dtype = {"i32": torch.int32, "f32": torch.float32}[output_record["dtype"]]
      if profile:
        output_layout = _buffer_layout(output_record, profile)
        canary = bytes.fromhex(
          manifest["execution"]["output_canary_by_dtype"][output_record["dtype"]]
        )
        initial_output = _initial_output(
          output_record, canary, envelope, profile=profile
        )
        output_buffer = (
          torch.frombuffer(bytearray(initial_output), dtype=output_dtype)
          .clone()
          .to(device)
        )
        output = (
          output_buffer[
            envelope.guards.prefix : envelope.guards.prefix
            + output_record["storage_span"]
          ]
          if envelope.declared
          else output_buffer
        )
      else:
        output_count = output_record["byte_length"] // 4
        fill_value = _I32_SENTINEL if output_record["dtype"] == "i32" else float("nan")
        output_buffer = torch.full(
          (output_count,), fill_value, dtype=output_dtype, device=device
        )
        output = output_buffer
      if envelope.declared:
        (artifacts_dir / "worker-staging.json").write_text(
          json.dumps(
            _staging_record(
              envelope,
              execution=manifest["execution"],
              inputs=manifest["inputs"],
              output=output_record,
            ),
            sort_keys=True,
          )
          + "\n",
          encoding="utf-8",
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
    grid = (1, 1, 1) if profile else (1,)
    if mode == "compiled":
      configure_pass_dumps(artifacts_dir)
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

    materialize_started = time.perf_counter()
    try:
      # The checks compare the whole allocation, so they read the buffer the
      # kernel's view points into rather than the view itself.
      physical = _host_bytes(
        output_buffer if envelope.declared and output_record["shape"] else output
      )
      if profile:
        if envelope.declared and output_record["shape"]:
          # The observed bytes are saved before any storage verdict: a wrong
          # logical value belongs to the numerical comparator, and a storage
          # violation must not cost the run its evidence.
          result = tensor_layout.gather_envelope(
            physical,
            output_layout,
            element_size=_ELEMENT_SIZE,
            envelope=envelope.guards,
            profile=profile,
          )
          (artifacts_dir / "actual-physical.bin").write_bytes(physical)
          (artifacts_dir / "actual.bin").write_bytes(result)
          # Every nonlogical slot still holds its canary, so a guarded write
          # is detected without judging the logical result here.
          changed = tensor_layout.changed_canary_slots(
            initial_output,
            physical,
            output_layout,
            element_size=_ELEMENT_SIZE,
            envelope=envelope.guards,
            profile=profile,
          )
          if changed:
            raise RuntimeError(
              "kernel wrote outside its logical output mapping: "
              f"guards and gaps differ at {changed[:16]}"
            )
        elif output_record["shape"]:
          changed = tensor_layout.changed_canary_slots(
            initial_output,
            physical,
            output_layout,
            element_size=_ELEMENT_SIZE,
            profile=profile,
          )
          if changed:
            raise RuntimeError(f"kernel modified output canary slots: {changed[:16]}")
          result = tensor_layout.gather_buffer(
            physical, output_layout, element_size=_ELEMENT_SIZE, profile=profile
          )
        else:
          result = physical[: output_record["byte_length"]]
        (artifacts_dir / "actual-physical.bin").write_bytes(physical)
      else:
        result = physical
      (artifacts_dir / "actual.bin").write_bytes(result)
      if profile == "v3" or (envelope.declared and profile):
        for record, (staged_input, initial) in zip(
          manifest["inputs"], zip(staged_inputs, initial_inputs)
        ):
          actual_input = _host_bytes(staged_input)
          if actual_input == initial:
            continue
          changed = _modified_input_slots(
            record, initial, actual_input, envelope.guards, profile
          )
          raise RuntimeError(
            f"kernel modified input allocation or gap canary for {record['name']}: "
            f"{changed[:16]}"
          )
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
