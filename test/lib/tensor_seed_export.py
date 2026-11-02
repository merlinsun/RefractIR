"""Export a free-form bundle as a layout-mutator seed (seed-export plan, E1).

A seed directory holds `kernel.py`, `launch.json`, `seed.json`, `inputs/`, and
the `relations.json` sidecar both projects share. The kernel is the one the
RefractIR sweep runs, emitted from the bundle by the same `symirc` call, and is
left unchanged: every argument keeps the position, shape, dtype and strides the
emitter compiled in. A scalar SIR parameter is read through a pointer by that
kernel, so it travels as a zero-dimensional tensor, not by value.

Only dense row-major layouts are exported. The emitted kernel bakes its
addressing into the source, so a padded bundle would need its storage
reconstructed, which no exported block needs yet.
"""

import ast
import hashlib
import json
import re
import shutil
import struct
import tempfile
from pathlib import Path

from test.lib import (
  tensor_execution,
  tensor_free_form_inputs,
  tensor_relations_sidecar,
  tensor_target_case,
  tensor_target_reference,
)

SIDECAR_SCHEMA = "layout-mutator.relations-sidecar.v1"
DTYPES = {"f32": "float32", "i32": "int32"}


class ExportError(ValueError):
  """A bundle the exporter cannot turn into a seed."""


def _sha256(data):
  return hashlib.sha256(data).hexdigest()


def _plain(name):
  return name.lstrip("%")


def _dense_strides(shape):
  strides, step = [], 1
  for extent in reversed(shape):
    strides.insert(0, step)
    step *= extent
  return strides


def _require_dense(record):
  if (
    record["element_strides"] != _dense_strides(record["shape"])
    or record["element_offset"]
  ):
    raise ExportError(f"{record['name']}: only dense row-major layouts are exported")


def _kernel_parameters(source):
  """The kernel's positional parameters; defaulted ones (runtime loop bounds)
  keep their defaults and are not part of the launch."""
  for node in ast.walk(ast.parse(source)):
    if isinstance(node, ast.FunctionDef) and node.name == "kernel":
      positional = node.args.args
      return [
        arg.arg for arg in positional[: len(positional) - len(node.args.defaults)]
      ]
  raise ExportError("the emitted source has no kernel function")


def _emit(case, symirc, extra_args=(), launch_record=False):
  """The kernel text, and with `launch_record` the launch record symirc wrote."""
  with tempfile.TemporaryDirectory() as scratch:
    record_path = Path(scratch) / "launch-record.json"
    arguments = (
      [*extra_args, "--tensor-launch-json", str(record_path)] if launch_record else []
    )
    path, stderr, stdout = tensor_execution.emit_kernel(
      case, symirc, Path(scratch) / "kernel", extra_args=arguments
    )
    if path is None:
      raise ExportError(
        f"symirc refused the bundle: {(stderr or stdout).strip()[-500:]}"
      )
    record = json.loads(record_path.read_text("utf-8")) if launch_record else None
    return path.read_text(encoding="utf-8"), record


def _tensor(record, **storage):
  return {
    "kind": "tensor",
    "shape": list(record["shape"]),
    "dtype": DTYPES[record["dtype"]],
    "strides": list(record["element_strides"]),
    **storage,
  }


def _launch(case, names):
  inputs = case.record["inputs"]
  if names != [f"{_plain(r['name'])}_ptr" for r in inputs] + ["out_ptr"]:
    raise ExportError(f"kernel parameters {names} do not follow the bundle's inputs")
  args = [
    _tensor(record, fill=None, raw=f"{_plain(record['name'])}.bin") for record in inputs
  ]
  args.append(_tensor(case.record["output"], fill={"rule": "zeros"}, raw=None))
  return {
    "kernel": "kernel",
    "grid": list(case.record["grid"]),
    "options": dict(case.record["launch"]),
    "names": names,
    "args": args,
  }


def _raw_inputs(case, halves=None):
  """Each input's bytes; an f32 input held as float16 is narrowed, exactly,
  since its values were rounded to f16 before P0 saw them."""
  halves = halves or {}
  raw = {}
  for r in case.record["inputs"]:
    data = case.inputs[r["byte_offset"] : r["byte_offset"] + r["byte_length"]]
    if halves.get(_plain(r["name"])) == "float16":
      values = struct.unpack(f"<{len(data) // 4}f", data)
      data = struct.pack(f"<{len(values)}e", *values)
    raw[f"{_plain(r['name'])}.bin"] = data
  return raw


def _signature(source):
  """Every kernel parameter in order, as (name, default, constexpr)."""
  for node in ast.walk(ast.parse(source)):
    if isinstance(node, ast.FunctionDef) and node.name == "kernel":
      positional = node.args.args
      defaults = [None] * (len(positional) - len(node.args.defaults)) + list(
        node.args.defaults
      )
      return [
        (
          arg.arg,
          None if default is None else ast.literal_eval(default),
          arg.annotation is not None and "constexpr" in ast.unparse(arg.annotation),
        )
        for arg, default in zip(positional, defaults)
      ]
  raise ExportError("the emitted source has no kernel function")


def _generation_launch(case, kernel, record, halves):
  """A launch naming every kernel parameter in signature order: layout-mutator
  passes arguments positionally, and its launch relation varies the block
  constexpr and the stride and offset scalars by name."""
  inputs = {f"{_plain(r['name'])}_ptr": r for r in case.record["inputs"]}
  names, args = [], []
  for name, default, constexpr in _signature(kernel):
    names.append(name)
    if name in inputs or name == "out_ptr":
      tensor = inputs.get(name, case.record["output"])
      buffer = "return" if name == "out_ptr" else _plain(tensor["name"])
      storage = (
        {"fill": {"rule": "zeros"}, "raw": None}
        if name == "out_ptr"
        else {"fill": None, "raw": f"{_plain(tensor['name'])}.bin"}
      )
      arg = _tensor(tensor, **storage)
      if buffer in halves:
        arg["dtype"] = halves[buffer]
      args.append(arg)
    elif constexpr:
      args.append(
        {"kind": "constexpr", "value": record["constexprs"].get(name, default)}
      )
    else:
      args.append(
        {"kind": "scalar", "value": record["scalars"].get(name, default), "type": "int"}
      )
  return {
    "kernel": "kernel",
    "grid": list(record["grid"]),
    "options": dict(case.record["launch"]),
    "names": names,
    "args": args,
  }


_LITERAL = re.compile(r"(?<![%\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])")


def literals(source):
  """Each literal operand of a tensor operation, with the operation's name."""
  found = []
  for line in source.splitlines():
    match = re.search(r"= tensor (\w+)(?: (?:lt|le|eq|f16|bf16))? (.*);", line)
    if not match or match.group(1) in {
      "sum",
      "rmin",
      "rmax",
      "cumsum",
      "expand",
      "transpose",
      "broadcast",
      "window",
    }:
      continue
    for token in _LITERAL.findall(match.group(2)):
      found.append({"value": float(token), "role": match.group(1), "lifted_to": None})
  return found


def _scalar(case, name):
  for record in case.record["inputs"]:
    if record["name"] == name:
      data = case.inputs[record["byte_offset"] : record["byte_offset"] + 4]
      return int.from_bytes(data, "little", signed=True)
  raise ExportError(f"the bundle has no input {name}")


def sidecar(case, region, *, narrow_bf16="excluded"):
  """The sidecar for the case's path: the body's relation facts on that path,
  and no expected output yet.

  The trip count is the path component the contract names. A branch selector
  is recorded in `seed.json` instead, since the contract has no field for it.
  """
  trip = region.get("trip_parameter")
  trip_counts = (
    {} if trip is None else {_plain(trip["name"]): _scalar(case, trip["name"])}
  )
  parameters = tensor_free_form_inputs.parameters(region)
  scalars = {
    record["name"]: _scalar(case, record["name"])
    for record in parameters
    if record["dtype"] == "i32"
  }
  facts = tensor_relations_sidecar.analyze(
    case.source,
    parameters,
    scalars,
    element=region["dtype"],
    narrow_bf16=narrow_bf16,
  )
  path = {"trip_counts": trip_counts}
  if region.get("range"):
    path["scalars"] = {_plain(name): value for name, value in scalars.items()}
    path["step"] = region["range"]["step"]
  return {
    "schema": SIDECAR_SCHEMA,
    "source": "refractir",
    "inputs": facts["inputs"],
    "outputs": facts["outputs"],
    "path": path,
    "expected": {"p0_sha256": None, "p0_file": None},
  }


P0_FILE = "outputs/out_ptr.p0.bin"


def _expected(case, out, symiri):
  """Run P0 on the bundle; write its output and return the sidecar's
  `expected` with the run's record for `seed.json`. A refusal (undefined
  behaviour, including a non-finite value) leaves `expected` empty."""
  with tempfile.TemporaryDirectory() as scratch:
    observation = tensor_target_reference.observe(
      case, symiri=symiri, output_path=Path(scratch) / "p0.bin"
    )
  if not observation.completed:
    reason = observation.reason or observation.stderr.strip()[-500:]
    return {"p0_sha256": None, "p0_file": None}, {
      "status": f"p0_{observation.status}",
      "reason": reason or "P0 did not complete",
    }
  path = out / P0_FILE
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(observation.output)
  return {"p0_sha256": _sha256(observation.output), "p0_file": P0_FILE}, {
    "status": "completed"
  }


def export_seed(
  bundle_dir,
  region,
  out_dir,
  *,
  symirc,
  symiri=None,
  provenance=None,
  narrow_bf16="excluded",
  generation=None,
):
  """Write one seed directory for a bundle and its region record; return it.

  With `symiri`, P0's output becomes the seed's expected output. `narrow_bf16`
  is the class `narrow bf16` has on the host that will run the seed (see
  `tensor_relations_sidecar.analyze`). `generation` (generation plan G9/G10)
  states `emit_args` for symirc, `halves` (buffer name or `return` to
  `float16`/`bfloat16`), and the steering `target`; the launch then names every
  kernel parameter and the sidecar gains `launch`, `literals` and `path.target`.
  """
  case = tensor_target_case.load(bundle_dir)
  for record in [*case.record["inputs"], case.record["output"]]:
    _require_dense(record)
  generation = generation or {}
  halves = dict(generation.get("halves", {}))
  emit_args = list(generation.get("emit_args", ()))
  for name, dtype in halves.items():
    emit_args += ["--tensor-boundary-dtype", f"{name}={dtype}"]
  kernel, record = _emit(case, symirc, emit_args, launch_record=bool(generation))
  launch = (
    _generation_launch(case, kernel, record, halves)
    if record is not None
    else _launch(case, _kernel_parameters(kernel))
  )
  out = Path(out_dir)
  if out.exists():
    shutil.rmtree(out)
  (out / "inputs").mkdir(parents=True)
  (out / "kernel.py").write_text(kernel, encoding="utf-8")
  (out / "launch.json").write_text(
    json.dumps(launch, indent=2) + "\n", encoding="utf-8"
  )
  for name, data in _raw_inputs(case, halves).items():
    (out / "inputs" / name).write_bytes(data)
  relations = sidecar(case, region, narrow_bf16=narrow_bf16)
  if record is not None:
    relations["launch"] = {
      key: record[key] for key in ("tiles", "strides", "offsets") if record.get(key)
    }
    relations["literals"] = literals(case.source)
    if generation.get("target"):
      relations["path"]["target"] = generation["target"]
  p0 = None
  if symiri is not None:
    relations["expected"], p0 = _expected(case, out, symiri)
  branch = region.get("branch_parameter")
  record = {
    "source": "refractir",
    "bundle_sha256": _sha256((Path(bundle_dir) / "case.json").read_bytes()),
    "program_sha256": _sha256(case.source.encode("utf-8")),
    "kernel_sha256": _sha256(kernel.encode("utf-8")),
    **({"branch": {_plain(branch): _scalar(case, branch)}} if branch else {}),
    **({"p0": p0} if p0 is not None else {}),
    **(provenance or {}),
  }
  (out / "seed.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  (out / "relations.json").write_text(
    json.dumps(relations, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  return out
