#!/usr/bin/env python3
"""Run certified generated tensor regions through Gluon layouts."""

import argparse
import functools
import hashlib
import importlib.util
import json
import math
import os
import statistics
import struct
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import triton

from test.lib import (
  probe_triton_env,
  tensor_case,
  tensor_exact_f32,
  tensor_layout,
  tensor_route_protocol,
)
from test.lib.tensor_compiler_observation import (
  observe_compiler_ir,
  observe_pass_dumps,
)
from test.lib.tensor_gluon import (
  PLAN_FAMILY_MINIMUM_LAYOUTS,
  BlockedLayoutPlan,
  execution_tile,
  generate_layout_plans,
  region_consumers,
  region_plan_families,
  select_distinct_layouts,
)
from test.lib.tensor_gluon_lowering import (
  cross_lane_threads_per_warp,
  reduction_blocked_layout,
  render_gluon_kernel,
)

WARMUP_LAUNCHES = tensor_route_protocol.WARMUP_LAUNCHES
MEASURED_LAUNCHES = tensor_route_protocol.MEASURED_LAUNCHES
ROOT = Path(__file__).resolve().parents[3]

# The modules that decide what a plan-mode run emits and how its addresses are
# validated. A run states their digests beside the toolchain fields, so an
# addressing record can be traced to the emitter that produced it.
EMITTER_MODULES = (
  "test.lib.tensor_layout",
  "test.lib.tensor_gluon",
  "test.lib.tensor_gluon_lowering",
  "test.lib.tensor_exact_f32",
)
LAYOUT_FIELDS = (
  "shape",
  "element_strides",
  "element_offset",
  "storage_span",
  "axis_order",
)


def _word_bytes(values):
  """One sequence of f32 values as the little-endian bytes a buffer holds."""
  return b"".join(struct.pack("<f", float(value)) for value in values)


def _word_float(word):
  """One recorded f32 sentinel, written as the hex word the manifest holds."""
  return struct.unpack("<f", bytes.fromhex(word))[0]


def _float_word(value):
  """One f32 sentinel as the hex word a record states it in."""
  return f"{struct.unpack('<I', struct.pack('<f', float(value)))[0]:08x}"


def _addressing_lines(source):
  """The addresses and validity predicates one emitted kernel states."""
  return [
    line.strip()
    for line in source.splitlines()
    if "_offsets =" in line or (line.strip().startswith("valid_") and "=" in line)
  ]


def _staging_record(staging):
  """The physical allocation one run staged, stated as the record declares it."""
  return {
    "record": staging.record(),
    "profile": staging.addressing.profile,
    "guard_envelope": {
      "prefix": staging.envelope.prefix,
      "suffix": staging.envelope.suffix,
    },
    "allocation_slots": staging.slots(),
    "padding_bits": _float_word(staging.padding),
    "canary_bits": _float_word(staging.canary),
  }


def declared_layout_record(record):
  """The addressing record one case buffer declares, or None without one.

  A manifest states the physical addressing of a buffer in the same five fields
  `test.lib.tensor_layout` validates, so a record is read here exactly as the
  worker reads it and nothing about the addressing is restated by hand.
  """
  if not isinstance(record, dict):
    return None
  if any(field not in record for field in LAYOUT_FIELDS):
    return None
  return {field: record[field] for field in LAYOUT_FIELDS}


def _guard_envelope(value):
  """The guard envelope one run stages around its recorded allocation."""
  if value is None:
    return tensor_layout.GuardEnvelope()
  if (
    not isinstance(value, (tuple, list))
    or len(value) != 2
    or any(type(item) is not int or item < 0 for item in value)
  ):
    raise ValueError("guard envelope must be two nonnegative integers")
  return tensor_layout.GuardEnvelope(prefix=value[0], suffix=value[1])


@dataclass(frozen=True)
class _Staging:
  """One physical allocation a launch addresses, and the readings it leaves.

  `addressing` is the validated restatement of the case's layout record, so the
  addresses the emitter writes and the addresses the buffers are staged at come
  from one validation of one record. Every nonlogical slot of the input holds
  `padding` and the whole output allocation holds `canary` before the launch, and
  the staged input bytes are retained, so a write that leaves the logical mapping
  changes a slot the verdict names and a write into the input is refused by the
  slots it changed.
  """

  addressing: object
  envelope: object
  padding: float
  canary: float

  def record(self):
    """The layout record this staging was validated from."""
    return self.addressing.as_record()

  def slots(self):
    """The element count of one whole allocation, guards included."""
    return self.envelope.prefix + self.addressing.storage_span + self.envelope.suffix

  def allocate(self, values):
    """Stage the logical input and a canary output into one allocation each."""
    record = self.record()
    staged_input = tensor_layout.stage_envelope(
      _word_bytes(values),
      record,
      element_size=4,
      envelope=self.envelope,
      padding=_word_bytes([self.padding]),
      profile=self.addressing.profile,
    )
    initial_output = _word_bytes([self.canary]) * self.slots()
    begin = self.envelope.prefix
    end = begin + self.addressing.storage_span
    input_buffer = _device_buffer(staged_input)
    output_buffer = _device_buffer(initial_output)
    return {
      "input_buffer": input_buffer,
      "input": input_buffer[begin:end],
      "output_buffer": output_buffer,
      "output": output_buffer[begin:end],
      "initial_input": staged_input,
      "initial_output": initial_output,
    }

  def read(self, allocation, name):
    """The gathered logical output and the storage verdict of one launch.

    The whole input allocation is compared with the bytes it was staged with
    before the output is judged, so a kernel that writes through its input
    pointer - into a guard, a gap, or a logical element - is refused with the
    slots it changed, named by role exactly as an output write is. A kernel has
    no right to any of them, and an input reading is only meaningful if the
    bytes it read were never rewritten.
    """
    record = self.record()
    changed_input = tensor_layout.classify_changed_slots(
      allocation["initial_input"],
      _host_bytes(allocation["input_buffer"]),
      record,
      element_size=4,
      envelope=self.envelope,
      profile=self.addressing.profile,
    )
    if changed_input:
      raise RuntimeError(
        f"{name} modified its input allocation: slots differ at {changed_input[:16]}"
      )
    physical = _host_bytes(allocation["output_buffer"])
    logical = tensor_layout.gather_envelope(
      physical,
      record,
      element_size=4,
      envelope=self.envelope,
      profile=self.addressing.profile,
    )
    changed = tensor_layout.classify_changed_slots(
      allocation["initial_output"],
      physical,
      record,
      element_size=4,
      envelope=self.envelope,
      profile=self.addressing.profile,
    )
    verdict = {
      "profile": self.addressing.profile,
      "allocation_slots": self.slots(),
      "changed_slot_count": len(changed),
      "logical_slots_changed": sum(1 for slot in changed if slot.startswith("input-")),
      "guard_or_gap_slots_changed": [
        slot for slot in changed if not slot.startswith("input-")
      ],
      "input_slots_changed": changed_input,
    }
    return logical, verdict


def _staging(addressing, guard_envelope, padding, canary):
  """One validated record as the allocation a run stages it in."""
  sentinel = float("nan") if padding is None else float(padding)
  return _Staging(
    addressing=addressing,
    envelope=_guard_envelope(guard_envelope),
    padding=sentinel,
    canary=sentinel if canary is None else float(canary),
  )


def _device_buffer(data):
  """One staged allocation on the device the run launches on."""
  return torch.frombuffer(bytearray(data), dtype=torch.float32).clone().to("cuda")


def _host_bytes(tensor):
  """Copy one device buffer back as the raw bytes it holds."""
  return tensor.view(torch.uint8).cpu().numpy().tobytes()


def _module_digests(names):
  """The digest of every imported module that decided what a run emitted."""
  digests = {}
  for name in names:
    path = Path(getattr(sys.modules.get(name), "__file__", "") or "")
    if not path.is_file():
      continue
    digests[name] = {
      "path": str(path),
      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
  return digests


def _qualification_record(
  *, triton_version, device_name, compute_capability, ir_directory
):
  """What one runner invocation states about the toolchain that produced it.

  The fields are the ones the locked runner records for a counted case: the
  RefractIR revision and compiler flags the driver exported, the Triton version
  and its source revision, the package lock, and the device. `campaign_valid`
  repeats the locked runner's predicate, so a plan-mode case is qualified on the
  same terms as the counted case it is compared with.
  """
  environment = probe_triton_env.collect_environment()
  probe_triton_env.write_environment(Path(ir_directory), environment)
  source_revision = os.environ.get(
    "REFRACTIR_REVISION", environment.get("refractir_revision") or "unavailable"
  )
  compiler_flags = os.environ.get("REFRACTIR_COMPILER_FLAGS", "unavailable")
  compiler = Path(os.environ.get("REFRACTIR_SYMIRC", ROOT / "symirc")).resolve()
  compiler_digest = None
  try:
    compiler_digest = hashlib.sha256(compiler.read_bytes()).hexdigest()
  except OSError:
    pass
  campaign_valid = bool(
    environment.get("campaign_available")
    and source_revision != "unavailable"
    and compiler_flags != "unavailable"
    and compiler_digest
    and environment.get("triton_source_revision") not in (None, "unavailable")
    and environment.get("package_lock_sha256")
  )
  return {
    "schema": "refractir.tensor-run-qualification.v1",
    "refractir_revision": source_revision,
    "working_tree_dirty": os.environ.get("REFRACTIR_WORKTREE_DIRTY", "unknown"),
    "compiler": {
      "executable": str(compiler),
      "sha256": compiler_digest,
      "flags": compiler_flags,
    },
    "triton_version": environment.get("triton_version") or triton_version,
    "triton_source_revision": environment.get("triton_source_revision"),
    "torch_version": torch.__version__,
    "device": {
      "name": device_name,
      "compute_capability": list(compute_capability),
    },
    "driver_version": environment.get("driver_version"),
    "python_version": environment.get("python_version"),
    "package_lock_sha256": environment.get("package_lock_sha256"),
    "environment_status": environment.get("status"),
    "failure_reasons": list(environment.get("failure_reasons", [])),
    "campaign_available": bool(environment.get("campaign_available")),
    "campaign_valid": campaign_valid,
    "emitter_modules": _emitter_digests(),
    "environment_file": str(Path(ir_directory) / "environment.json"),
    "package_lock_file": str(Path(ir_directory) / "package-lock.json"),
  }


def _emitter_digests():
  """The digest of every module that decided what this run emitted.

  The runner itself is named from its own file, because a run started with `-m`
  is imported as `__main__` and would otherwise state no digest for the code
  that chose the addresses under test.
  """
  digests = _module_digests(EMITTER_MODULES)
  runner = Path(__file__).resolve()
  digests["test.tensor.gluon.run_generated_region"] = {
    "path": str(runner),
    "sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
  }
  return digests


def _case_qualification(qualification):
  """The qualification one case record repeats, so it stands on its own."""
  return {
    "campaign_valid": qualification["campaign_valid"],
    "refractir_revision": qualification["refractir_revision"],
    "triton_source_revision": qualification["triton_source_revision"],
    "compiler_sha256": qualification["compiler"]["sha256"],
    "package_lock_sha256": qualification["package_lock_sha256"],
  }


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


def _measure_region_kernel(kernel, args, *, num_warps):
  started = time.perf_counter()
  compiled = kernel[(1,)](*args, num_warps=num_warps, enable_fp_fusion=False)
  torch.cuda.synchronize()
  compilation_wall_s = time.perf_counter() - started
  for _ in range(WARMUP_LAUNCHES):
    kernel[(1,)](*args, num_warps=num_warps, enable_fp_fusion=False)
  torch.cuda.synchronize()
  launches_per_batch = MEASURED_LAUNCHES // tensor_route_protocol.MEASUREMENT_BATCHES
  samples_ms = []
  for _ in range(tensor_route_protocol.MEASUREMENT_BATCHES):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(launches_per_batch):
      kernel[(1,)](*args, num_warps=num_warps, enable_fp_fusion=False)
    end.record()
    torch.cuda.synchronize()
    samples_ms.append(start.elapsed_time(end) / launches_per_batch)
  median_event_ms = statistics.median(samples_ms)
  return compiled, {
    "warmup_launches": WARMUP_LAUNCHES,
    "measured_launches": MEASURED_LAUNCHES,
    "measurement_batches": tensor_route_protocol.MEASUREMENT_BATCHES,
    "launches_per_batch": launches_per_batch,
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


def region_layout_pool(shape, reduction_layout, *, num_warps, seed, count, tile=None):
  """Sample the distinct layouts one run may place on the region roles.

  The required reduction layout comes first so every pool can express the
  baseline role assignment; the rest is drawn deterministically from the seed.
  `tile` is the execution tile the layouts are assigned to, so a tailed shape
  samples the layouts of its padded tile rather than of its logical extent.
  """
  sampled_plans = (
    generate_layout_plans(shape, num_warps=num_warps, seed=seed, count=count, tile=tile)
    if count > 1
    else ()
  )
  return select_distinct_layouts(
    reduction_layout,
    sampled_plans,
    shape=shape,
    num_warps=num_warps,
    count=count,
    tile=tile,
  )


def select_plan_family(name, shape, *, num_warps, layouts, consumers, tile=None):
  """Select one named plan family the sampled pool and the region can express.

  A family states its roles from the pool's distinct layouts and its consumer
  views from the region's own consumers, so a request the pool or the graph
  cannot express is refused with the reason instead of silently collapsing.
  """
  if name not in PLAN_FAMILY_MINIMUM_LAYOUTS:
    raise ValueError(
      f"unknown plan family {name!r}; expected one of "
      + ", ".join(sorted(PLAN_FAMILY_MINIMUM_LAYOUTS))
    )
  families = region_plan_families(
    shape, num_warps=num_warps, layouts=layouts, consumers=consumers, tile=tile
  )
  family = families.get(name)
  if family is None:
    raise ValueError(
      f"plan family {name!r} cannot be expressed: {families.omitted[name]}"
    )
  return family


def region_launch_arguments(generated, layouts, input_tensor, output_tensor):
  """The positional plan-mode kernel arguments: two pointers, then layouts.

  The layout objects follow `generated.layout_parameters`, which is the order
  the emitter wrote into the kernel signature.
  """
  return (input_tensor, output_tensor) + tuple(
    plan.to_gluon_layout() for plan in generated.layout_arguments(layouts)
  )


def plan_launch_record(generated, plan, pool):
  """The replayable record of one plan-mode launch.

  `roles` names the layout of every role and consumer view, `layouts` gives the
  physical identity behind each name, and `arguments` lists the layouts the
  kernel receives in its own signature order.
  """
  named = [plan.load, plan.compute, plan.store, plan.reduction, plan.state_layout]
  named.extend(plan.consumer_views.values())
  identities = {layout.name: layout.to_dict() for layout in named if layout is not None}
  return {
    "family": plan.name,
    "shape": list(plan.shape),
    "tile": list(plan.tile) if plan.tile is not None else None,
    "roles": plan.as_record(),
    "layouts": {name: identities[name] for name in sorted(identities)},
    "pool": [layout.to_dict() for layout in pool],
    "layout_parameters": list(generated.layout_parameters),
    "arguments": [
      {"parameter": parameter, "layout": layout.name}
      for parameter, layout in zip(
        generated.layout_parameters, generated.layout_arguments(pool), strict=True
      )
    ],
  }


def _region_case_record(
  kernel,
  args,
  output_buffer,
  *,
  name,
  layouts,
  ir_directory,
  generated,
  num_warps,
  expected_bytes,
  staging=None,
  allocation=None,
  qualification=None,
):
  """One launched case: its logical bytes, its storage verdict, its identity.

  Without a layout record the bytes are the dense output the kernel stored.
  With one the kernel wrote into a physical allocation, so the input allocation
  is compared with the bytes it was staged with and the bytes are gathered from
  the addresses the record names; the slots outside them are named: a write into
  either allocation has left the tensor's own mapping and is refused with the
  slots it touched, before the logical values are judged.
  """
  compiled, measurement = _measure_region_kernel(kernel, args, num_warps=num_warps)
  guard_verdict = None
  if staging is None:
    actual_bytes = output_buffer.view(torch.uint8).cpu().numpy().tobytes()
  else:
    actual_bytes, guard_verdict = staging.read(allocation, name)
    if guard_verdict["guard_or_gap_slots_changed"]:
      raise RuntimeError(
        f"{name} wrote outside its logical output mapping: guards and gaps "
        f"differ at {guard_verdict['guard_or_gap_slots_changed'][:16]}"
      )
  if actual_bytes != expected_bytes:
    raise AssertionError(
      f"{name} differs from complete exact output bits: "
      f"actual={_word_hex(actual_bytes)}, expected={_word_hex(expected_bytes)}"
    )
  record = {
    "status": "pass",
    "layouts": layouts,
    "actual_output_bits": _word_hex(actual_bytes),
    "expected_output_bits": _word_hex(expected_bytes),
    "measurement": measurement,
    "compiler": _capture_region_compiler(
      compiled,
      name,
      ir_directory,
      generated.conversion_source_lines,
      expected_loop=generated.iterations > 0,
    ),
  }
  if guard_verdict is not None:
    record["guard_verdict"] = guard_verdict
  if qualification is not None:
    record["qualification"] = _case_qualification(qualification)
  return record


def _program_staging(layout, guard_envelope, padding, canary, shape):
  """The physical allocation one program run addresses, or None when dense.

  The record is validated here rather than left to the emitter, so a case whose
  record cannot address its own storage is refused before any kernel is written
  and before the device is asked for a buffer. The shape check states the same
  refusal the emitter states, because the record has to describe the region the
  kernel is emitted from.
  """
  if layout is None:
    if guard_envelope is not None or padding is not None or canary is not None:
      raise ValueError("a guard envelope, padding, or canary requires a layout record")
    return None
  addressing = tensor_layout.storage_addressing(layout)
  if addressing.shape != tuple(shape):
    raise ValueError("declared layout shape differs from the generated region shape")
  return _staging(addressing, guard_envelope, padding, canary)


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
  num_warps=1,
  layout_count=5,
  plan_family=None,
  tile=None,
  layout=None,
  guard_envelope=None,
  padding=None,
  canary=None,
):
  """Compare sampled Gluon layouts on one admitted generated-region source.

  Without `plan_family` the run samples one layout per role and launches the
  historical emitter. With one it launches a single plan-mode kernel whose
  layouts the family names.

  `tile` is the power-of-two execution tile a logical tail runs on. It is
  required when the region's extent is not a power of two, and refused when it
  is stated but does not cover the shape.

  `layout` is the physical layout record the case declares: the element
  strides, element offset, storage span, and axis order that
  `test.lib.tensor_layout` validates. It is validated here, before anything is
  emitted, and restated to the emitter, so the addresses the kernel computes
  are the addresses the input is staged at and the output is gathered from. A
  record that leaves its own storage span, or whose axis order contradicts its
  strides, is refused before emission. `guard_envelope` allocates prefix and
  suffix guards around the recorded storage span, `padding` fills every
  nonlogical slot of the input allocation, and `canary` fills the output
  allocation before the launch, so a write outside the logical mapping is
  reported as the slots it changed, and a write into any slot of the input
  allocation - guard, gap, or logical element - is refused with the slots it
  changed before the output is judged.
  """
  probe_triton_env.require_qualified_triton()
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA device is unavailable")
  device_name = torch.cuda.get_device_name()
  if "V100" not in device_name:
    raise RuntimeError(
      f"expected a V100 for the generated-region study, found {device_name}"
    )
  if type(layout_seed) is not int or layout_seed < 0:
    raise ValueError("layout_seed must be a nonnegative integer")
  if type(num_warps) is not int or num_warps <= 0 or num_warps & (num_warps - 1):
    raise ValueError("num_warps must be a positive power of two")
  if type(layout_count) is not int or layout_count <= 0:
    raise ValueError("layout_count must be a positive integer")

  source_path = Path(program_path).resolve()
  source = source_path.read_text(encoding="utf-8")
  shape, iterations, tensor_types, nodes, _updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  tile = execution_tile(shape, tile=tile) if tile is not None else None
  staging = _program_staging(layout, guard_envelope, padding, canary, shape)
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
  qualification = _qualification_record(
    triton_version=triton.__version__,
    device_name=device_name,
    compute_capability=torch.cuda.get_device_capability(),
    ir_directory=ir_directory,
  )
  materialization_started = time.perf_counter()
  input_tensor = output_tensor = None
  if staging is None:
    input_tensor = torch.tensor(
      input_values, dtype=torch.float32, device="cuda"
    ).reshape(shape)
    output_tensor = torch.empty(shape, dtype=torch.float32, device="cuda")
  materialization_s = time.perf_counter() - materialization_started
  staged_s = 0.0

  def case_pointers():
    """The pointers one case launch addresses, and what to read back from it.

    A recorded layout stages its own allocation per case, so no case is judged
    against what another case left in the buffer; a dense run keeps the two
    buffers the study has always launched.
    """
    nonlocal staged_s
    if staging is None:
      return (input_tensor, output_tensor), output_tensor, None
    started = time.perf_counter()
    allocation = staging.allocate(input_values)
    staged_s += time.perf_counter() - started
    return (
      (allocation["input"], allocation["output"]),
      allocation["output_buffer"],
      allocation,
    )

  reduction_axes = {
    attribute if attribute >= 0 else attribute + len(shape)
    for _destination, operation, operands, attribute in nodes
    if operation == "sum" and tensor_types[operands[0]] == shape
  }
  if not reduction_axes:
    raise ValueError("generated region has no full-shape reduction to distribute")
  reduction_layout = reduction_blocked_layout(
    shape, reduction_axes, num_warps=num_warps
  )
  reduction_layout.validate(shape, num_warps=num_warps, tile=tile)
  plans = region_layout_pool(
    shape,
    reduction_layout,
    num_warps=num_warps,
    seed=layout_seed,
    count=layout_count,
    tile=tile,
  )
  emission = {} if staging is None else {"layout": staging.record()}
  cases = {}
  if plan_family is None:
    generated = render_gluon_kernel(source, tile=tile, **emission)
    kernel = _load_kernel(generated, ir_directory / "generated-region-kernel.py")
    for plan in plans:
      plan.validate(shape, num_warps=num_warps, tile=tile)
      pointers, output_buffer, allocation = case_pointers()
      args = (
        *pointers,
        plan.to_gluon_layout(),
        plan.to_gluon_layout(),
        reduction_layout.to_gluon_layout(),
        plan.to_gluon_layout(),
      )
      cases[plan.name] = _region_case_record(
        kernel,
        args,
        output_buffer,
        name=plan.name,
        layouts={
          "load": plan.to_dict(),
          "compute": plan.to_dict(),
          "reduction": reduction_layout.to_dict(),
          "store": plan.to_dict(),
        },
        ir_directory=ir_directory,
        generated=generated,
        num_warps=num_warps,
        expected_bytes=expected_bytes,
        staging=staging,
        allocation=allocation,
        qualification=qualification,
      )
    baseline_case = plans[0].name
    baseline_layout_arguments = (
      plans[0].to_gluon_layout(),
      plans[0].to_gluon_layout(),
      reduction_layout.to_gluon_layout(),
      plans[0].to_gluon_layout(),
    )
    plan_record = None
  else:
    family_plan = select_plan_family(
      plan_family,
      shape,
      num_warps=num_warps,
      layouts=plans,
      consumers=region_consumers(source),
      tile=tile,
    )
    generated = render_gluon_kernel(source, plan=family_plan, **emission)
    kernel = _load_kernel(generated, ir_directory / "generated-region-kernel.py")
    pointers, output_buffer, allocation = case_pointers()
    arguments = region_launch_arguments(generated, plans, *pointers)
    cases[plan_family] = _region_case_record(
      kernel,
      arguments,
      output_buffer,
      name=plan_family,
      layouts={
        "load": family_plan.load.to_dict(),
        "compute": family_plan.compute.to_dict(),
        "reduction": (
          family_plan.reduction.to_dict() if family_plan.reduction else None
        ),
        "store": family_plan.store.to_dict(),
      },
      ir_directory=ir_directory,
      generated=generated,
      num_warps=num_warps,
      expected_bytes=expected_bytes,
      staging=staging,
      allocation=allocation,
      qualification=qualification,
    )
    baseline_case = plan_family
    baseline_layout_arguments = tuple(arguments[2:])
    plan_record = plan_launch_record(generated, family_plan, plans)

  store_lines = [
    line
    for line in generated.source.splitlines()
    if line.startswith("    gl.store(output_ptr + store_offsets, output")
  ]
  if len(store_lines) != 1:
    raise RuntimeError("generated Gluon kernel has no unique output store")
  store_line = store_lines[0] + "\n"
  mutant = replace(
    generated,
    source=generated.source.replace(
      store_line, "    output = output + 1.0\n" + store_line
    ),
  )
  mutant_kernel = _load_kernel(
    mutant, ir_directory / "calibration-add-one-output-mutant.py"
  )
  mutant_pointers, mutant_output, mutant_allocation = case_pointers()
  mutant_args = (*mutant_pointers, *baseline_layout_arguments)
  mutant_compiled, mutant_measurement = _measure_region_kernel(
    mutant_kernel, mutant_args, num_warps=num_warps
  )
  mutant_verdict = None
  if staging is None:
    mutant_bytes = mutant_output.view(torch.uint8).cpu().numpy().tobytes()
  else:
    # The mutant adds one to every stored value, so its own copy of the
    # allocation is read rather than the baseline case's.
    mutant_bytes, mutant_verdict = staging.read(
      mutant_allocation, "calibration-add-one-output-mutant"
    )
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
  baseline_detected = cases[baseline_case]["actual_output_bits"] != _word_hex(
    expected_bytes
  )
  result = {
    "status": "pass",
    "contract": "generated-region-exact-f32-layout-study-v1",
    "triton_version": triton.__version__,
    "torch_version": torch.__version__,
    "device_name": device_name,
    "compute_capability": list(torch.cuda.get_device_capability()),
    "program_sha256": hashlib.sha256(source.encode()).hexdigest(),
    "input_domain": list(domain),
    "input_values": list(input_values),
    "input_bits_sha256": hashlib.sha256(input_bytes).hexdigest(),
    "input_materialization_s": materialization_s + staged_s,
    "reference_validation_s": reference_validation_s,
    "iterations": iterations,
    "shape": list(shape),
    "execution_tile": list(tile) if tile is not None else None,
    "property": {"target": target, "index": list(target_index)},
    "layout_seed": layout_seed,
    "proof_bound": proof["integer_exactness_bound"],
    "proof_obligations": len(proof["bounds"]),
    "complete_output_bits": _word_hex(expected_bytes),
    "qualification": qualification,
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
      "guard_verdict": mutant_verdict,
    },
    "measurement_protocol": {
      "num_warps": num_warps,
      "enable_fp_fusion": False,
      "warmup_launches_per_layout": WARMUP_LAUNCHES,
      "measured_launches_per_layout": MEASURED_LAUNCHES,
      "measurement_batches": tensor_route_protocol.MEASUREMENT_BATCHES,
      "expected_layout_count": layout_count,
      "cache": tensor_route_protocol.cache_record_from_environment(),
      "measurement_accounting": tensor_route_protocol.measurement_accounting(
        useful_compilations=len(cases),
        calibration_compilations=1,
        useful_executions=len(cases),
        calibration_executions=1,
      ),
      "comparison": "same exact SIR, selected input, domain, and property across layouts",
    },
    "cases": cases,
    "physical_layout": None if staging is None else _staging_record(staging),
    "emission": {
      "sha256": hashlib.sha256(generated.source.encode()).hexdigest(),
      "addressing": _addressing_lines(generated.source),
    },
    "_conversion_source_lines": generated.conversion_source_lines,
  }
  if plan_record is not None:
    # Only a plan-mode run states a family; no-plan reports keep their keys.
    result["plan"] = plan_record
  Path(output_path).write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  return result


def _declared_sentinel(execution, key, dtype="f32"):
  """One f32 sentinel a case manifest declares, or None when it declares none."""
  word = (execution.get(key) or {}).get(dtype)
  return None if word is None else _word_float(word)


def _case_staging(manifest, shape):
  """The physical allocation one case manifest declares, once it is validated.

  The emitter addresses both pointers from one record, so a case whose buffers
  declare different addressing is refused rather than addressed wrongly, and a
  manifest that declares no addressing keeps the dense allocation the
  cross-lane pilot has always launched.
  """
  records = [declared_layout_record(record) for record in manifest.get("inputs", [])]
  output = declared_layout_record(manifest.get("output", {}))
  declared = [record for record in records if record is not None]
  if not declared and output is None:
    return None
  if len(declared) != len(records) or output is None:
    raise ValueError("every case buffer must declare its physical layout record")
  if any(record != declared[0] for record in declared) or output != declared[0]:
    raise ValueError("case buffers declare different physical layout records")
  addressing = tensor_layout.storage_addressing(declared[0])
  if addressing.shape != tuple(shape):
    raise ValueError("declared layout shape differs from the generated region shape")
  execution = manifest.get("execution") or {}
  guard = execution.get("guard_envelope")
  padding = _declared_sentinel(execution, "padding_by_dtype")
  if padding is None:
    padding = _declared_sentinel(execution, "gap_sentinel_by_dtype")
  return _staging(
    addressing,
    (guard.get("prefix", 0), guard.get("suffix", 0)) if guard else None,
    padding,
    _declared_sentinel(execution, "output_canary_by_dtype"),
  )


def run(case_directory, output_path, ir_directory, *, symiri=None):
  probe_triton_env.require_qualified_triton()
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
  staging = _case_staging(case.manifest, input_shape)
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
  emission = {} if staging is None else {"layout": staging.record()}
  generated = render_gluon_kernel(source, **emission)
  ir_directory = Path(ir_directory).resolve()
  ir_directory.mkdir(parents=True, exist_ok=True)
  qualification = _qualification_record(
    triton_version=triton.__version__,
    device_name=device_name,
    compute_capability=torch.cuda.get_device_capability(),
    ir_directory=ir_directory,
  )
  kernel = _load_kernel(generated, ir_directory / "generated-region-kernel.py")

  values = struct.unpack(f"<{math.prod(input_shape)}f", case.inputs)
  allocation = None if staging is None else staging.allocate(values)
  if allocation is None:
    input_tensor = torch.tensor(values, dtype=torch.float32, device="cuda").reshape(
      input_shape
    )
    output_tensor = torch.empty(input_shape, dtype=torch.float32, device="cuda")
  else:
    input_tensor = allocation["input"]
    output_tensor = allocation["output"]
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
  guard_verdict = None
  if staging is None:
    actual_bytes = output_tensor.view(torch.uint8).cpu().numpy().tobytes()
  else:
    actual_bytes, guard_verdict = staging.read(allocation, "generated Gluon kernel")
    if guard_verdict["guard_or_gap_slots_changed"]:
      raise RuntimeError(
        "generated Gluon kernel wrote outside its logical output mapping: "
        f"guards and gaps differ at "
        f"{guard_verdict['guard_or_gap_slots_changed'][:16]}"
      )
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
  launches_per_batch = MEASURED_LAUNCHES // tensor_route_protocol.MEASUREMENT_BATCHES
  samples_ms = []
  for _ in range(tensor_route_protocol.MEASUREMENT_BATCHES):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(launches_per_batch):
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
    samples_ms.append(start_event.elapsed_time(end_event) / launches_per_batch)
  median_event_ms = statistics.median(samples_ms)

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
    "physical_layout": None if staging is None else _staging_record(staging),
    "guard_verdict": guard_verdict,
    "qualification": qualification,
    "emission": {
      "sha256": hashlib.sha256(generated.source.encode()).hexdigest(),
      "addressing": _addressing_lines(generated.source),
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
      "measurement_batches": tensor_route_protocol.MEASUREMENT_BATCHES,
      "launches_per_batch": launches_per_batch,
      "first_launch_wall_s_including_compilation": compilation_wall_s,
      "mean_gpu_event_ms_per_launch": median_event_ms,
    },
    "ir_files": ir_files,
  }
  Path(output_path).write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  return report


def _sentinel_argument(text):
  """One sentinel the command line states, as a float or an f32 bit pattern."""
  if text is None:
    return None
  if text.lower().startswith("0x"):
    return struct.unpack("<f", struct.pack("<I", int(text, 16)))[0]
  return float(text)


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
  parser.add_argument("--num-warps", type=int, default=1)
  parser.add_argument("--layout-count", type=int, default=5)
  parser.add_argument(
    "--tile",
    default=None,
    help="power-of-two execution tile as rows,columns for a tailed logical extent",
  )
  parser.add_argument(
    "--plan-family", choices=sorted(PLAN_FAMILY_MINIMUM_LAYOUTS), default=None
  )
  parser.add_argument(
    "--layout",
    default=None,
    help="path to the JSON physical layout record the case declares",
  )
  parser.add_argument(
    "--guard-envelope",
    default=None,
    help="prefix,suffix guards the case allocates around one buffer",
  )
  parser.add_argument(
    "--padding",
    default=None,
    help="f32 sentinel for guards and gaps, as a float or an 0x bit pattern",
  )
  parser.add_argument(
    "--canary",
    default=None,
    help="f32 initial value of the output allocation, as a float or an 0x word",
  )
  args = parser.parse_args()
  if args.plan_family and not args.program:
    parser.error("--plan-family requires --program")
  if args.layout and not args.program:
    parser.error("--layout requires --program")
  layout = None
  if args.layout:
    try:
      layout = json.loads(Path(args.layout).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
      parser.error(f"invalid layout record: {error}")
  guard_envelope = None
  if args.guard_envelope:
    try:
      guard_envelope = tuple(
        int(value.strip()) for value in args.guard_envelope.split(",")
      )
    except ValueError as error:
      parser.error(f"invalid guard envelope: {error}")
    if len(guard_envelope) != 2:
      parser.error("--guard-envelope must contain two integers")
  try:
    padding = _sentinel_argument(args.padding)
    canary = _sentinel_argument(args.canary)
  except ValueError as error:
    parser.error(f"invalid sentinel: {error}")
  tile = None
  if args.tile is not None:
    try:
      tile = tuple(int(value.strip()) for value in args.tile.split(","))
    except ValueError as error:
      parser.error(f"invalid execution tile: {error}")
    if len(tile) != 2:
      parser.error("--tile must contain two integers")
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
      num_warps=args.num_warps,
      layout_count=args.layout_count,
      plan_family=args.plan_family,
      tile=tile,
      layout=layout,
      guard_envelope=guard_envelope,
      padding=padding,
      canary=canary,
    )
  elif args.case_dir:
    report = run(args.case_dir, args.output, args.ir_dir, symiri=args.symiri)
  else:
    parser.error("one of --case-dir or --program is required")
  print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
  main()
