"""Select and export an exact case from an admitted generated loop region."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path

from test.lib import tensor_exact_f32, tensor_layout
from test.lib.build_tensor_exact_loop_case import (
  CaseBuildError,
  _digest,
  _record,
  _run,
  _write_attempt,
)
from test.lib.tensor_case import NUMERIC_PRECISIONS
from test.lib.tensor_coverage import source_signature

DOMAIN = (0, 8)
TIMEOUT_S = 60.0
GAP_SENTINEL_BY_DTYPE = {"i32": "55555555", "f32": "3f000001"}
OUTPUT_CANARY_BY_DTYPE = {"i32": "aaaaaaaa", "f32": "3f000002"}
PROFILE_VERSIONS = {"tensor-v2": "v2", "tensor-v3": "v3"}
PROFILE_SCHEMAS = {
  "tensor-v2": "refractir.tensor-case.v2",
  "tensor-v3": "refractir.tensor-case.v3",
}


def physical_layout(shape, *, profile="tensor-v2", row_padding=0, element_offset=0):
  """The row-major physical layout a case's buffers are staged in.

  The tile is the profile's execution tile: the logical extents under
  tensor-v2, and the padded power-of-two extents under tensor-v3. The row
  stride is the tile's column extent plus any requested gap, so a tailed
  shape's rows are strided by the padded tile width and its logical payload
  occupies a sub-region of the allocation rather than the whole of it.
  """
  if profile not in PROFILE_VERSIONS:
    raise CaseBuildError("unsupported_profile", f"unknown tensor profile {profile!r}")
  if (
    not isinstance(row_padding, int)
    or row_padding < 0
    or not isinstance(element_offset, int)
    or element_offset < 0
  ):
    raise CaseBuildError(
      "invalid_layout", "row padding and element offset must be nonnegative"
    )
  if len(shape) != 2:
    raise CaseBuildError("invalid_layout", "a region case stores a rank-two tensor")
  tile = tensor_layout.execution_tile_shape(
    list(shape), profile=PROFILE_VERSIONS[profile]
  )
  strides = [tile[1] + row_padding, 1]
  return {
    "tile_shape": tile,
    "element_strides": strides,
    "element_offset": element_offset,
    "storage_span": element_offset
    + (shape[0] - 1) * strides[0]
    + (shape[1] - 1) * strides[1]
    + 1,
    "axis_order": [1, 0],
  }


def _stage_record(record, layout):
  """Restate one buffer record in the profile's physical layout."""
  return {**record, **layout, "tile_origin": [0, 0]}


def _layout_manifest(records):
  """The addressing manifest the compiler emits a kernel against."""
  return {
    "buffers": [
      {
        "name": record["name"],
        "role": role,
        "element_strides": record["element_strides"],
        "element_offset": record["element_offset"],
        "storage_span": record["storage_span"],
        "axis_order": record["axis_order"],
      }
      for role, record in records
    ]
  }


def _hex_words(values, name):
  if not isinstance(values, dict):
    raise CaseBuildError("invalid_guards", f"{name} must map dtypes to hex words")
  words = {}
  for dtype, value in values.items():
    if (
      not isinstance(value, str)
      or len(value) != 8
      or any(char not in "0123456789abcdefABCDEF" for char in value)
    ):
      raise CaseBuildError(
        "invalid_guards", f"{name} entry {dtype} must be four hexadecimal bytes"
      )
    words[dtype] = value.lower()
  return words


def _guard_record(guard_envelope, padding):
  """Translate a requested guard envelope into the worker's execution record."""
  if guard_envelope is None:
    return {}, {}
  if (
    not isinstance(guard_envelope, tuple)
    or len(guard_envelope) != 2
    or any(type(value) is not int or value < 0 for value in guard_envelope)
  ):
    raise CaseBuildError(
      "invalid_guards", "guard envelope must be a pair of nonnegative integers"
    )
  prefix, suffix = guard_envelope
  if prefix + suffix == 0:
    raise CaseBuildError(
      "invalid_guards", "a guard envelope must allocate at least one slot"
    )
  envelope = {"prefix": prefix, "suffix": suffix}
  if padding is None:
    return envelope, dict(GAP_SENTINEL_BY_DTYPE)
  return envelope, _hex_words(padding, "padding")


def _declared_witness(witness, count, domain):
  """Check the caller's declared input against the certified domain."""
  if (
    not isinstance(witness, (tuple, list))
    or len(witness) != count
    or any(
      type(value) is not int or not domain[0] <= value <= domain[1] for value in witness
    )
  ):
    raise CaseBuildError(
      "invalid_witness",
      "declared witness must hold one in-domain integer per tensor element",
    )
  return tuple(witness)


def _model_inputs(model_path, count, domain):
  try:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    values = model["@kernel"]["%?x"]
  except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
    raise CaseBuildError(
      "invalid_model", "solver model has no tensor witness for %?x"
    ) from error
  if (
    not isinstance(values, list)
    or len(values) != count
    or any(
      type(value) is not int or not domain[0] <= value <= domain[1] for value in values
    )
  ):
    raise CaseBuildError(
      "invalid_model",
      "solver witness is outside the declared domain or has the wrong shape",
    )
  return tuple(values)


def _selection_record(domain, target, path, files, result_bytes):
  return {
    "method": "bounded-integer-twin-smt",
    "domain": list(domain),
    "target": {"index": [0, 0], "value": target},
    "path": path,
    "query_file": "selection-query.sir",
    "query_sha256": hashlib.sha256(files["selection-query.sir"]).hexdigest(),
    "solved_file": "selection.solved.sir",
    "solved_sha256": hashlib.sha256(files["selection.solved.sir"]).hexdigest(),
    "model_file": "selection-model.json",
    "model_sha256": hashlib.sha256(files["selection-model.json"]).hexdigest(),
    "result_file": "selection-result.json",
    "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
    "replay": "complete-f32-output-bitwise",
  }


def build_case(
  output_dir: str | Path,
  *,
  source: str,
  target: int,
  symiri: str | Path = "symiri",
  symirc: str | Path = "symirc",
  solver: str | Path = "symirsolve",
  domain: tuple[int, int] = DOMAIN,
  timeout_s: float = TIMEOUT_S,
  configuration=None,
  guard_envelope: tuple[int, int] | None = None,
  padding: dict[str, str] | None = None,
  profile: str = "tensor-v2",
  row_padding: int = 0,
  element_offset: int = 0,
  selection: str = "solver",
  witness: tuple[int, ...] | None = None,
) -> Path:
  """Prove bounds, solve an i32 twin, replay f32, then write a tensor case.

  `configuration` is the explicit region request the source was generated from.
  It is recorded so a case keeps the shape, bounds, and graph family that
  produced it instead of only the source bytes.

  `guard_envelope` asks the device route to allocate prefix and suffix guards
  around every buffer, with `padding` filling the guards and internal gaps. It
  changes only physical storage, never the logical inputs or reference output.

  `profile` selects the tensor profile the case is typed against: tensor-v2
  admits only power-of-two extents, while tensor-v3 admits logical tails and
  stages them on the padded execution tile that `row_padding` and
  `element_offset` then place inside the physical allocation. Every command,
  the manifest, and the kernel all state that profile, so a tailed case cannot
  be recorded as if its logical extent were its physical one.

  `selection` states how the runtime input is admitted. "solver" solves the
  bounded-integer twin for a witness that reaches `target` and records the
  selection as exact_smt evidence. "witness" executes the caller's declared
  in-domain `witness` instead: the reference is still the exact obligation
  proof of that input replayed by the interpreter, but the case states a
  zero-tolerance numerical comparison rather than a solver selection. A tailed
  tensor-v3 region needs the witness route, because the tensor-v3 profile's
  operation set excludes the `tensor extract` the selection query states its
  target with.
  """
  if profile not in PROFILE_VERSIONS:
    raise CaseBuildError("unsupported_profile", f"unknown tensor profile {profile!r}")
  if selection not in {"solver", "witness"}:
    raise CaseBuildError("unsupported_selection", f"unknown selection {selection!r}")
  if selection == "solver" and witness is not None:
    raise CaseBuildError(
      "invalid_witness", "a declared witness requires selection='witness'"
    )
  profile_version = PROFILE_VERSIONS[profile]
  envelope_record, padding_record = _guard_record(guard_envelope, padding)

  directory = Path(output_dir).resolve()
  directory.mkdir(parents=True, exist_ok=False)
  region_record = configuration.as_record() if configuration is not None else None
  region_seed = configuration.seed if configuration is not None else 0
  ledger: list[dict] = []
  deadline = time.monotonic() + timeout_s
  source_path = directory / "program.sir"
  query_path = directory / "selection-query.sir"
  solved_path = directory / "selection.solved.sir"
  model_path = directory / "selection-model.json"
  input_path = directory / "inputs.bin"
  output_path = directory / "expected.bin"
  kernel_path = directory / "kernel.py"
  try:
    shape, iterations, _, _, _ = tensor_exact_f32._generated_region_program(source)
    if configuration is not None:
      from test.lib import tensor_region_generation

      tensor_region_generation.validate_source(source, configuration)
    if type(target) is not int:
      raise CaseBuildError("unsupported_region", "target must be an integer")
    layout = physical_layout(
      shape, profile=profile, row_padding=row_padding, element_offset=element_offset
    )
    count = math.prod(shape)
    preflight_value = float(domain[0])
    preflight_input = struct.pack(f"<{count}f", *([preflight_value] * count))
    # Bounds cover the declared domain; the concrete preflight must be a member
    # of that domain because certification also checks the supplied input.
    tensor_exact_f32.certify(source, preflight_input, domain)
    path = tensor_exact_f32.generated_region_path(iterations)
    source_path.write_text(source, encoding="utf-8")
    query = None
    solve_result = None
    if selection == "witness":
      values = _declared_witness(witness, count, domain)
    else:
      query = tensor_exact_f32.render_generated_region_query(source, domain, target)
      query_path.write_text(query, encoding="utf-8")
      solve_result = _run(
        "solve",
        [
          str(solver),
          "--tensor-profile-version",
          profile_version,
          "--main",
          "@kernel",
          "--path",
          ",".join(path),
          "--timeout-ms",
          str(int(timeout_s * 1000)),
          "--output",
          str(solved_path),
          "--emit-model",
          str(model_path),
          str(query_path),
        ],
        deadline=deadline,
        ledger=ledger,
      )
      values = _model_inputs(model_path, count, domain)
    input_bytes = struct.pack(f"<{len(values)}f", *values)
    proof, independently_expected = tensor_exact_f32.certify(
      source, input_bytes, domain
    )
    output_values = struct.unpack(
      f"<{len(independently_expected) // 4}f", independently_expected
    )
    if selection == "solver" and output_values[0] != target:
      raise CaseBuildError(
        "property_mismatch", "selected output does not satisfy the target"
      )
    input_path.write_bytes(input_bytes)
    _run(
      "reference",
      [
        str(symiri),
        "--tensor-profile-version",
        profile_version,
        "--entry",
        "kernel",
        "--tensor-input",
        str(input_path),
        "--tensor-output",
        str(output_path),
        str(source_path),
      ],
      deadline=deadline,
      ledger=ledger,
    )
    expected = output_path.read_bytes()
    if expected != independently_expected:
      raise CaseBuildError(
        "replay_mismatch", "complete f32 reference output differs from exact proof bits"
      )
    input_record = _stage_record(
      _record("x", "f32", list(shape), "inputs.bin", len(input_bytes)), layout
    )
    output_record = _stage_record(
      _record("return", "f32", list(shape), "expected.bin", len(expected)), layout
    )
    layout_path = directory / "tensor-layout.json"
    layout_path.write_text(
      json.dumps(
        _layout_manifest((("input", input_record), ("output", output_record))),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
      )
      + "\n",
      encoding="utf-8",
    )
    _run(
      "compile",
      [
        str(symirc),
        "--tensor-profile-version",
        profile_version,
        "--tensor-layout-manifest",
        str(layout_path),
        "--target",
        "triton",
        "--entry",
        "kernel",
        "-o",
        str(kernel_path),
        str(source_path),
      ],
      deadline=deadline,
      ledger=ledger,
    )

    files = {
      "program.sir": source.encode(),
      "inputs.bin": input_bytes,
      "expected.bin": expected,
      "kernel.py": kernel_path.read_bytes(),
      "tensor-layout.json": layout_path.read_bytes(),
    }
    if selection == "solver":
      solve_record = ledger[0]
      result_bytes = (
        json.dumps(
          {
            "status": "SAT",
            "exit_code": solve_result.returncode,
            "stdout": solve_result.stdout,
            "command": solve_record["command"],
          },
          ensure_ascii=False,
          sort_keys=True,
          indent=2,
        )
        + "\n"
      ).encode()
      (directory / "selection-result.json").write_bytes(result_bytes)
      files.update(
        {
          "selection-query.sir": query.encode(),
          "selection.solved.sir": solved_path.read_bytes(),
          "selection-model.json": model_path.read_bytes(),
          "selection-result.json": result_bytes,
        }
      )
      proof["selection"] = _selection_record(domain, target, path, files, result_bytes)
    operations = tensor_exact_f32.generated_region_operations(source)
    tensor_types = tensor_exact_f32._generated_region_program(source)[2]
    shapes = [
      list(value)
      for value in sorted(
        set(tensor_types.values()), key=lambda item: (len(item), item)
      )
    ]
    witness_route = selection == "witness"
    pattern = (
      "repeated-value"
      if witness_route and len(set(values)) == 1
      else "seeded-random"
      if witness_route
      else "smt-selected"
    )
    input_behavior = {
      "property_passed": True,
      "pattern": pattern,
      "input_mode": "declared-witness" if witness_route else "bounded-integer-twin-smt",
      "policy": "declared-witness" if witness_route else "exact",
      "input_domain": {"lo": domain[0], "hi": domain[1]},
    }
    if not witness_route:
      input_behavior["distinguishing_objective"] = "output-element-equality"
    coverage = {
      "primary_stratum": "generated-bounded-region",
      "source_signature": source_signature(source),
      "loop_structure": [{"bound": iterations}],
      "input_behavior": input_behavior,
    }
    manifest = {
      "schema": PROFILE_SCHEMAS[profile],
      "profile": profile,
      "entry": "kernel",
      "source": {
        "file": "program.sir",
        "sha256": _digest(files["program.sir"])["sha256"],
        "generator": "generated-bounded-region",
        "seed": region_seed,
      },
      "inputs": [input_record],
      "output": output_record,
      "grid": [1, 1, 1],
      "softmax_lowering": "direct",
      "launch": {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False},
      "property": {
        "id": (
          f"declared-witness-output-element-0-0-equals-{output_values[0]}"
          if witness_route
          else f"output-element-0-0-equals-{target}"
        ),
        "source": (
          f"Execute the declared integer-valued f32 witness in "
          f"[{domain[0]},{domain[1]}] and replay the complete output."
          if witness_route
          else f"Select integer-valued f32 leaves in [{domain[0]},{domain[1]}] "
          f"so output[0,0] equals {target}."
        ),
      },
      "evidence": {
        "kind": "construction" if witness_route else "exact_smt",
        "status": "construction_checked" if witness_route else "exact_smt_checked",
        "artifacts": list(files),
      },
      "comparison": {
        "mode": "numeric" if witness_route else "exact",
        "atol": [0, 1],
        "rtol": [0, 1],
      },
      "reference": {
        "defined": True,
        "stable": True,
        "precisions": NUMERIC_PRECISIONS if witness_route else [],
        "exactness": None if witness_route else proof,
      },
      "execution": {
        "full_shapes": shapes,
        "range_extents": shapes,
        "warp_probe": {"status": "not_probed", "num_warps": 4},
        "loop_bounds": [iterations],
        "total_body_executions": iterations,
        "gap_sentinel_by_dtype": dict(GAP_SENTINEL_BY_DTYPE),
        "output_canary_by_dtype": dict(OUTPUT_CANARY_BY_DTYPE),
      },
      "coverage": coverage,
      "environment": None,
      "operator_coverage": operations,
      "shape_axes": [],
      "generator": {
        "version": "generated-bounded-region-v1",
        "seed": region_seed,
        "region": region_record,
        "input_mode": input_behavior["input_mode"],
        "input_policy": input_behavior["policy"],
        "input_domain": input_behavior["input_domain"],
        "input_pattern": pattern,
        "synthesis": "declared-witness" if witness_route else "integer-twin-smt",
        "primary_stratum": "generated-bounded-region",
        "node_limit": 10,
      },
      "files": {name: _digest(data) for name, data in files.items()},
    }
    if witness_route:
      # The declared witness is not a solver selection, so the case states
      # what it actually ran instead of naming a query that never existed.
      manifest["generator"]["witness_output_sha256"] = hashlib.sha256(
        independently_expected
      ).hexdigest()
    else:
      manifest["solver_selection"] = {
        "query_file": "selection-query.sir",
        "solved_file": "selection.solved.sir",
        "model_file": "selection-model.json",
        "query_sha256": hashlib.sha256(files["selection-query.sir"]).hexdigest(),
        "model_sha256": hashlib.sha256(files["selection-model.json"]).hexdigest(),
        "domain": list(domain),
        "target": {"index": [0, 0], "value": target},
        "path": path,
      }
    if envelope_record:
      manifest["execution"]["guard_envelope"] = envelope_record
      manifest["execution"]["padding_by_dtype"] = padding_record
    (directory / "case.json").write_text(
      json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
      encoding="utf-8",
    )
    _write_attempt(directory, "accepted", ledger, None)
    return directory
  except CaseBuildError as error:
    _write_attempt(directory, error.status, ledger, str(error))
    raise
  except BaseException as error:
    _write_attempt(directory, "failed", ledger, f"{type(error).__name__}: {error}")
    raise


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", required=True)
  parser.add_argument("--target", required=True, type=int)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--symiri", default="symiri")
  parser.add_argument("--symirc", default="symirc")
  parser.add_argument("--symirsolve", default="symirsolve")
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--shape", default=None)
  parser.add_argument("--iterations", type=int, default=None)
  parser.add_argument("--family", default=None)
  parser.add_argument("--guard-prefix", type=int, default=0)
  parser.add_argument("--guard-suffix", type=int, default=0)
  parser.add_argument("--padding", default=None)
  args = parser.parse_args(argv)
  source = Path(args.source).read_text(encoding="utf-8")
  configuration = None
  if args.seed is not None:
    from test.lib import tensor_region_generation

    if args.shape is None or args.iterations is None:
      parser.error("--seed requires --shape and --iterations")
    configuration = tensor_region_generation.RegionConfiguration(
      seed=args.seed,
      shape=tuple(int(value) for value in args.shape.split(",")),
      iterations=args.iterations,
      family=args.family or "fold",
    )
  guard_envelope = (args.guard_prefix, args.guard_suffix)
  if guard_envelope == (0, 0):
    if args.padding is not None:
      parser.error("--padding requires --guard-prefix or --guard-suffix")
    guard_envelope = None
  padding = None
  if args.padding is not None:
    try:
      padding = dict(pair.split("=", 1) for pair in args.padding.split(","))
    except ValueError:
      parser.error("--padding must be a comma-separated list of dtype=hex pairs")
  case = build_case(
    args.output_dir,
    source=source,
    target=args.target,
    symiri=args.symiri,
    symirc=args.symirc,
    solver=args.symirsolve,
    configuration=configuration,
    guard_envelope=guard_envelope,
    padding=padding,
  )
  print(json.dumps({"case": str(case), "status": "accepted"}))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
