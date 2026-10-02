"""Solve and export the authored composed f32 loop as an exact case."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import time
from pathlib import Path

from test.lib import tensor_exact_f32
from test.lib.tensor_case_manifest import record_buffer
from test.lib.tensor_coverage import source_signature

DOMAIN = (0, 8)
TARGET = 63
PATH = ["^entry", "^header", "^body", "^header", "^body", "^header", "^exit"]
TIMEOUT_S = 60.0


class CaseBuildError(ValueError):
  """A refusal, timeout or failed stage that did not produce an accepted case."""

  def __init__(self, status: str, message: str):
    super().__init__(message)
    self.status = status


def render_query() -> str:
  """Select an exact i32 witness for the canonical f32 computation."""
  source = tensor_exact_f32.render_program(dtype="i32")
  source = source.replace(
    "fun @kernel(%x: tensor [2, 2] i32) : tensor [2, 2] i32 {",
    "fun @kernel() : tensor [2, 2] i32 {\n  sym %?x : value tensor [2, 2] i32 in [0, 8];",
  )
  source = source.replace("%x", "%?x")
  source = source.replace("^entry:\n", "  let mut %selected: i32 = undef;\n^entry:\n")
  source = source.replace(
    "  ret %out;",
    "  %selected = tensor extract %out, [0, 0];\n"
    "  require %selected == 63;\n  ret %out;",
  )
  return source


def _digest(data: bytes) -> dict[str, object]:
  return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _record(name: str, dtype: str, shape: list[int], file_name: str, length: int):
  record = record_buffer(name, dtype, shape, file_name, length)
  stride = 1
  strides = [0] * len(shape)
  for axis in range(len(shape) - 1, -1, -1):
    strides[axis] = stride
    stride *= shape[axis]
  record.update(
    {
      "element_strides": strides,
      "element_offset": 0,
      "storage_span": math.prod(shape),
      "axis_order": list(reversed(range(len(shape)))),
      "tile_shape": shape[:],
      "tile_origin": [0] * len(shape),
    }
  )
  return record


def _run(stage: str, command: list[str], *, deadline: float, ledger: list[dict]):
  remaining = deadline - time.monotonic()
  if remaining <= 0:
    raise CaseBuildError("timeout", f"deadline expired before {stage}")
  try:
    result = subprocess.run(
      command,
      capture_output=True,
      text=True,
      check=False,
      timeout=min(TIMEOUT_S, remaining),
    )
  except subprocess.TimeoutExpired as error:
    ledger.append({"stage": stage, "status": "timeout", "command": command})
    raise CaseBuildError("timeout", f"{stage} timed out") from error
  except OSError as error:
    ledger.append({"stage": stage, "status": "environment_error", "command": command})
    raise CaseBuildError("environment_error", f"cannot run {stage}: {error}") from error
  ledger.append(
    {
      "stage": stage,
      "status": "pass" if result.returncode == 0 else "failed",
      "exit_code": result.returncode,
      "command": command,
      "stdout": result.stdout,
      "stderr": result.stderr,
    }
  )
  if time.monotonic() > deadline:
    raise CaseBuildError("timeout", f"{stage} completed after the shared deadline")
  if result.returncode != 0:
    status = (
      "unsat"
      if "UNSAT" in result.stdout
      else "solver_error"
      if stage == "solve"
      else "stage_error"
    )
    raise CaseBuildError(status, result.stderr.strip() or result.stdout.strip())
  return result


def _model_inputs(model_path: Path) -> tuple[int, ...]:
  try:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    entry = model["@kernel"]
    witness = entry["%?x"]
    values = witness if isinstance(witness, list) else witness["leaves"]
  except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
    raise CaseBuildError(
      "invalid_model", "solver model has no tensor witness for %?x"
    ) from error
  if (
    not isinstance(values, list)
    or len(values) != 4
    or any(
      type(value) is not int or not DOMAIN[0] <= value <= DOMAIN[1] for value in values
    )
  ):
    raise CaseBuildError(
      "invalid_model",
      "solver witness is outside the frozen domain or not four i32 leaves",
    )
  return tuple(values)


def _write_attempt(directory: Path, status: str, ledger: list[dict], error: str | None):
  (directory / "attempt.json").write_text(
    json.dumps({"status": status, "stages": ledger, "error": error}, indent=2) + "\n",
    encoding="utf-8",
  )


def build_case(
  output_dir: str | Path,
  *,
  symiri: str | Path = "symiri",
  symirc: str | Path = "symirc",
  solver: str | Path = "symirsolve",
  timeout_s: float = TIMEOUT_S,
) -> Path:
  """Build a checked selection, f32 replay and Triton bundle in a new directory."""
  directory = Path(output_dir).resolve()
  directory.mkdir(parents=True, exist_ok=False)
  ledger: list[dict] = []
  deadline = time.monotonic() + timeout_s
  source = tensor_exact_f32.render_program()
  query = render_query()
  input_path = directory / "inputs.bin"
  output_path = directory / "expected.bin"
  model_path = directory / "selection-model.json"
  solved_path = directory / "selection.solved.sir"
  query_path = directory / "selection-query.sir"
  source_path = directory / "program.sir"
  kernel_path = directory / "kernel.py"
  try:
    # Prove an input-independent upper bound before replacing f32 with i32 in
    # the solver query. A zero witness is only a carrier for the declared range.
    proof, _ = tensor_exact_f32.certify(source, struct.pack("<4f", 0, 0, 0, 0), DOMAIN)
    query_path.write_text(query, encoding="utf-8")
    source_path.write_text(source, encoding="utf-8")
    solve_result = _run(
      "solve",
      [
        str(solver),
        "--tensor-profile-version",
        "v2",
        "--main",
        "@kernel",
        "--path",
        ",".join(PATH),
        "--tensor-param-domain",
        "0,8",
        "--timeout-ms",
        str(int(timeout_s * 1000)),
        "--output",
        str(solved_path),
        "--emit-tensor-params",
        str(model_path),
        str(query_path),
      ],
      deadline=deadline,
      ledger=ledger,
    )
    values = _model_inputs(model_path)
    selection_result_path = directory / "selection-result.json"
    selection_result = {
      "status": "SAT",
      "exit_code": solve_result.returncode,
      "stdout": solve_result.stdout,
      "command": ledger[-1]["command"],
    }
    selection_result_path.write_text(
      json.dumps(selection_result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
      encoding="utf-8",
    )
    input_bytes = struct.pack("<4f", *values)
    proof, independently_expected = tensor_exact_f32.certify(
      source, input_bytes, DOMAIN
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
        "replay_mismatch",
        "complete f32 reference output differs bitwise from exact proof",
      )
    observed = struct.unpack("<4f", expected)
    if observed[0] != TARGET:
      raise CaseBuildError(
        "property_mismatch", f"selected output[0,0] is {observed[0]}, expected {TARGET}"
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
    kernel = kernel_path.read_bytes()
    query_bytes = query.encode()
    solver_model = model_path.read_bytes()
    selection_result_bytes = selection_result_path.read_bytes()
    source_bytes = source.encode()
    files = {
      "program.sir": source_bytes,
      "inputs.bin": input_bytes,
      "expected.bin": expected,
      "kernel.py": kernel,
      "selection-query.sir": query_bytes,
      "selection-model.json": solver_model,
      "selection-result.json": selection_result_bytes,
    }
    proof["selection"] = {
      "method": "bounded-integer-twin-smt",
      "domain": list(DOMAIN),
      "target": {"index": [0, 0], "value": TARGET},
      "path": PATH,
      "query_file": "selection-query.sir",
      "query_sha256": hashlib.sha256(query_bytes).hexdigest(),
      "model_file": "selection-model.json",
      "model_sha256": hashlib.sha256(solver_model).hexdigest(),
      "result_file": "selection-result.json",
      "result_sha256": hashlib.sha256(selection_result_bytes).hexdigest(),
      "replay": "complete-f32-output-bitwise",
    }
    manifest = {
      "schema": "refractir.tensor-case.v2",
      "profile": "tensor-v2",
      "entry": "kernel",
      "source": {
        "file": "program.sir",
        "sha256": _digest(source_bytes)["sha256"],
        "generator": "authored-composed-loop",
        "seed": 0,
      },
      "inputs": [_record("x", "f32", [2, 2], "inputs.bin", len(input_bytes))],
      "output": _record("return", "f32", [2, 2], "expected.bin", len(expected)),
      "grid": [1, 1, 1],
      "softmax_lowering": "direct",
      "launch": {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False},
      "property": {
        "id": "output-element-0-0-equals-63",
        "source": "Select four nonnegative integer f32 leaves in [0,8] so Y[0,0] equals 63.",
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
        "full_shapes": [[2, 2], [2, 2], [2, 1], [1, 2], [2, 2], [2, 2], [2, 2]],
        "range_extents": [[2, 2], [2, 2], [2, 1], [1, 2], [2, 2], [2, 2], [2, 2]],
        "warp_probe": {"status": "not_probed", "num_warps": 4},
        "loop_bounds": [2],
        "total_body_executions": 2,
        "gap_sentinel_by_dtype": {"i32": "55555555", "f32": "3f000001"},
        "output_canary_by_dtype": {"i32": "aaaaaaaa", "f32": "3f000002"},
      },
      "coverage": {
        "primary_stratum": "authored-composed-loop",
        "source_signature": source_signature(source),
        "loop_structure": [{"bound": 2}],
        "input_behavior": {
          "property_passed": True,
          "pattern": "smt-selected",
          "input_mode": "bounded-integer-twin-smt",
          "input_domain": {"lo": 0, "hi": 8},
          "policy": "exact",
          "distinguishing_objective": "output-element-equality",
        },
      },
      "environment": None,
      "operator_coverage": tensor_exact_f32.OPERATIONS,
      "shape_axes": [[[2, 2], 1], [[2, 2], 0]],
      "generator": {
        "version": "authored-composed-loop-v1",
        "seed": 0,
        "input_mode": "bounded-integer-twin-smt",
        "input_policy": "exact",
        "input_domain": {"lo": 0, "hi": 8},
        "input_pattern": "smt-selected",
        "synthesis": "integer-twin-smt",
        "primary_stratum": "authored-composed-loop",
        "node_limit": 8,
      },
      "solver_selection": {
        "query_file": "selection-query.sir",
        "model_file": "selection-model.json",
        "query_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "model_sha256": hashlib.sha256(solver_model).hexdigest(),
        "domain": list(DOMAIN),
        "target": {"index": [0, 0], "value": TARGET},
        "path": PATH,
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
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--symiri", default="symiri")
  parser.add_argument("--symirc", default="symirc")
  parser.add_argument("--symirsolve", default="symirsolve")
  args = parser.parse_args(argv)
  case = build_case(
    args.output_dir, symiri=args.symiri, symirc=args.symirc, solver=args.symirsolve
  )
  print(json.dumps({"case": str(case), "status": "accepted"}))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
