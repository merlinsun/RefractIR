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
import shutil
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


def _emit(case, symirc):
  with tempfile.TemporaryDirectory() as scratch:
    path, stderr, stdout = tensor_execution.emit_kernel(case, symirc, Path(scratch))
    if path is None:
      raise ExportError(
        f"symirc refused the bundle: {(stderr or stdout).strip()[-500:]}"
      )
    return path.read_text(encoding="utf-8")


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


def _raw_inputs(case):
  return {
    f"{_plain(r['name'])}.bin": case.inputs[
      r["byte_offset"] : r["byte_offset"] + r["byte_length"]
    ]
    for r in case.record["inputs"]
  }


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
  return {
    "schema": SIDECAR_SCHEMA,
    "source": "refractir",
    "inputs": facts["inputs"],
    "outputs": facts["outputs"],
    "path": {"trip_counts": trip_counts},
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
):
  """Write one seed directory for a bundle and its region record; return it.

  With `symiri`, P0's output becomes the seed's expected output. `narrow_bf16`
  is the class `narrow bf16` has on the host that will run the seed (see
  `tensor_relations_sidecar.analyze`).
  """
  case = tensor_target_case.load(bundle_dir)
  for record in [*case.record["inputs"], case.record["output"]]:
    _require_dense(record)
  kernel = _emit(case, symirc)
  launch = _launch(case, _kernel_parameters(kernel))
  out = Path(out_dir)
  if out.exists():
    shutil.rmtree(out)
  (out / "inputs").mkdir(parents=True)
  (out / "kernel.py").write_text(kernel, encoding="utf-8")
  (out / "launch.json").write_text(
    json.dumps(launch, indent=2) + "\n", encoding="utf-8"
  )
  for name, data in _raw_inputs(case).items():
    (out / "inputs" / name).write_bytes(data)
  relations = sidecar(case, region, narrow_bf16=narrow_bf16)
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
