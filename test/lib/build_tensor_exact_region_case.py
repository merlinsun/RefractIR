"""Select and export an exact case from an admitted generated loop region."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path

from test.lib import tensor_exact_f32
from test.lib.build_tensor_exact_loop_case import (
  CaseBuildError,
  _digest,
  _record,
  _run,
  _write_attempt,
)
from test.lib.tensor_coverage import source_signature

DOMAIN = (0, 8)
TIMEOUT_S = 60.0


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
) -> Path:
  """Prove bounds, solve an i32 twin, replay f32, then write a tensor-v2 case.

  `configuration` is the explicit region request the source was generated from.
  It is recorded so a case keeps the shape, bounds, and graph family that
  produced it instead of only the source bytes.
  """
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
    preflight_value = float(domain[0])
    preflight_input = struct.pack(
      f"<{math.prod(shape)}f", *([preflight_value] * math.prod(shape))
    )
    # Bounds cover the declared domain; the concrete preflight must be a member
    # of that domain because certification also checks the supplied input.
    tensor_exact_f32.certify(source, preflight_input, domain)
    query = tensor_exact_f32.render_generated_region_query(source, domain, target)
    path = tensor_exact_f32.generated_region_path(iterations)
    source_path.write_text(source, encoding="utf-8")
    query_path.write_text(query, encoding="utf-8")
    solve_result = _run(
      "solve",
      [
        str(solver),
        "--tensor-profile-version",
        "v2",
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
    values = _model_inputs(model_path, math.prod(shape), domain)
    input_bytes = struct.pack(f"<{len(values)}f", *values)
    proof, independently_expected = tensor_exact_f32.certify(
      source, input_bytes, domain
    )
    output_values = struct.unpack(
      f"<{len(independently_expected) // 4}f", independently_expected
    )
    if output_values[0] != target:
      raise CaseBuildError(
        "property_mismatch", "selected output does not satisfy the target"
      )
    input_path.write_bytes(input_bytes)
    _run(
      "reference",
      [
        str(symiri),
        "--tensor-profile-version",
        "v2",
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
    _run(
      "compile",
      [
        str(symirc),
        "--tensor-profile-version",
        "v2",
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
    files = {
      "program.sir": source.encode(),
      "inputs.bin": input_bytes,
      "expected.bin": expected,
      "kernel.py": kernel_path.read_bytes(),
      "selection-query.sir": query.encode(),
      "selection.solved.sir": solved_path.read_bytes(),
      "selection-model.json": model_path.read_bytes(),
      "selection-result.json": result_bytes,
    }
    proof["selection"] = _selection_record(domain, target, path, files, result_bytes)
    operations = tensor_exact_f32.generated_region_operations(source)
    tensor_types = tensor_exact_f32._generated_region_program(source)[2]
    shapes = [
      list(value)
      for value in sorted(
        set(tensor_types.values()), key=lambda item: (len(item), item)
      )
    ]
    coverage = {
      "primary_stratum": "generated-bounded-region",
      "source_signature": source_signature(source),
      "loop_structure": [{"bound": iterations}],
      "input_behavior": {
        "property_passed": True,
        "pattern": "smt-selected",
        "input_mode": "bounded-integer-twin-smt",
        "policy": "exact",
        "input_domain": {"lo": domain[0], "hi": domain[1]},
        "distinguishing_objective": "output-element-equality",
      },
    }
    manifest = {
      "schema": "refractir.tensor-case.v2",
      "profile": "tensor-v2",
      "entry": "kernel",
      "source": {
        "file": "program.sir",
        "sha256": _digest(files["program.sir"])["sha256"],
        "generator": "generated-bounded-region",
        "seed": region_seed,
      },
      "inputs": [_record("x", "f32", list(shape), "inputs.bin", len(input_bytes))],
      "output": _record("return", "f32", list(shape), "expected.bin", len(expected)),
      "grid": [1, 1, 1],
      "softmax_lowering": "direct",
      "launch": {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False},
      "property": {
        "id": f"output-element-0-0-equals-{target}",
        "source": f"Select integer-valued f32 leaves in [{domain[0]},{domain[1]}] so output[0,0] equals {target}.",
      },
      "evidence": {
        "kind": "exact_smt",
        "status": "exact_smt_checked",
        "artifacts": list(files),
      },
      "comparison": {"mode": "exact", "atol": [0, 1], "rtol": [0, 1]},
      "reference": {
        "defined": True,
        "stable": True,
        "precisions": [],
        "exactness": proof,
      },
      "execution": {
        "full_shapes": shapes,
        "range_extents": shapes,
        "warp_probe": {"status": "not_probed", "num_warps": 4},
        "loop_bounds": [iterations],
        "total_body_executions": iterations,
        "gap_sentinel_by_dtype": {"i32": "55555555", "f32": "3f000001"},
        "output_canary_by_dtype": {"i32": "aaaaaaaa", "f32": "3f000002"},
      },
      "coverage": coverage,
      "environment": None,
      "operator_coverage": operations,
      "shape_axes": [],
      "generator": {
        "version": "generated-bounded-region-v1",
        "seed": region_seed,
        "region": region_record,
        "input_mode": "bounded-integer-twin-smt",
        "input_policy": "exact",
        "input_domain": {"lo": domain[0], "hi": domain[1]},
        "input_pattern": "smt-selected",
        "synthesis": "integer-twin-smt",
        "primary_stratum": "generated-bounded-region",
        "node_limit": 10,
      },
      "solver_selection": {
        "query_file": "selection-query.sir",
        "solved_file": "selection.solved.sir",
        "model_file": "selection-model.json",
        "query_sha256": hashlib.sha256(files["selection-query.sir"]).hexdigest(),
        "model_sha256": hashlib.sha256(files["selection-model.json"]).hexdigest(),
        "domain": list(domain),
        "target": {"index": [0, 0], "value": target},
        "path": path,
      },
      "files": {name: _digest(data) for name, data in files.items()},
    }
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
  case = build_case(
    args.output_dir,
    source=source,
    target=args.target,
    symiri=args.symiri,
    symirc=args.symirc,
    solver=args.symirsolve,
    configuration=configuration,
  )
  print(json.dumps({"case": str(case), "status": "accepted"}))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
