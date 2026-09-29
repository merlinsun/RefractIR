"""Build the shape, axis, permutation, and composition conformance corpus."""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

from test.lib import build_tensor_smoke_cases as smoke

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "build" / "tensor_cases" / "regressions"
HISTORICAL_INPUT = [0.0, 2.0, 3.0, 1.0]


def required_shapes():
  return [(2, 2), (2, 4), (4, 2), (1, 4), (4, 1), (2, 2, 4)]


def permutations_for_rank(rank):
  return list(itertools.permutations(range(rank)))


def softmax_sweep():
  cases = []
  for shape in required_shapes():
    rank = len(shape)
    for axis in range(rank):
      for spelling in (axis, axis - rank):
        cases.append(
          {"shape": list(shape), "axis": axis, "axis_spelling": str(spelling)}
        )
  return cases


def transpose_sweep():
  shapes = [(4,), (2, 2), (2, 4), (4, 2), (2, 2, 4)]
  return [
    {"shape": list(shape), "permutation": list(permutation)}
    for shape in shapes
    for permutation in permutations_for_rank(len(shape))
  ]


def composition_pairs():
  return [
    {
      "id": "transpose-softmax",
      "orders": ["softmax-then-transpose", "transpose-then-softmax"],
    },
    {"id": "sum-expand", "orders": ["sum-then-expand", "expand-then-sum"]},
    {
      "id": "broadcast-softmax",
      "orders": ["broadcast-then-softmax", "softmax-then-broadcast"],
    },
    {
      "id": "transpose-sum-expand",
      "orders": ["sum-then-expand-then-transpose", "transpose-then-sum-then-expand"],
    },
  ]


def historical_case():
  return {
    "issue": 11406,
    "shape": [2, 2],
    "axis": 1,
    "input_values": HISTORICAL_INPUT,
    "legacy_keep_dims": False,
    "corrected_keep_dims": None,
    "affected_revision": "f4fca2c0bc0e19930d2be76bb06ee7b232b60e86",
    "corrected_revision": "31b1bf9aed2714af9a816d023b405499e8b27883",
    "fix_pr": "https://github.com/triton-lang/triton/pull/11409",
    "compatibility_pr": "https://github.com/triton-lang/triton/pull/11430",
  }


def _sir(param_types, result_type, declarations, statements, result="%out"):
  return (
    f"fun @kernel({param_types}) : {result_type} {{\n"
    + declarations
    + "\n^entry:\n"
    + statements
    + f"\n  ret {result};\n}}\n"
  )


def _tensor_type(shape, dtype):
  return f"tensor [{', '.join(str(size) for size in shape)}] {dtype}"


def _input_values(shape, dtype):
  count = 1
  for size in shape:
    count *= size
  if dtype == "i32":
    return [
      ((index * 13 + index // max(1, shape[-1]) * 7) % 31) - 15
      for index in range(count)
    ]
  values = []
  for index in range(count):
    residual = index
    coordinates = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
      coordinates[axis] = residual % shape[axis]
      residual //= shape[axis]
    weighted = sum(
      (coordinate + 1) * (axis + 1) * 0.375
      for axis, coordinate in enumerate(coordinates)
    )
    values.append(weighted + ((index % 3) - 1) * 0.25)
  return values


def _run(command):
  result = subprocess.run(
    command, cwd=ROOT, capture_output=True, text=True, timeout=30, check=False
  )
  if result.returncode:
    raise RuntimeError(
      "command failed: " + " ".join(map(str, command)) + "\n" + result.stderr
    )
  return result


def _build_case(
  case_root,
  case_name,
  source,
  parameter_values,
  *,
  numeric,
  group=None,
  symiri=smoke.SYMIRI,
  property_id=None,
):
  case_dir = case_root / case_name
  case_dir.mkdir(parents=True, exist_ok=False)
  source_path = case_dir / "program.sir"
  source_path.write_text(source, encoding="utf-8")
  signature = json.loads(
    _run(
      [
        str(symiri),
        "--entry",
        "kernel",
        "--tensor-describe",
        str(source_path),
      ]
    ).stdout
  )
  input_parts = []
  input_records = []
  offset = 0
  for input_info, (dtype, values, raw_words) in zip(
    signature["inputs"], parameter_values
  ):
    data = smoke._pack(dtype, values, words=raw_words)
    if len(data) != input_info["byte_length"]:
      raise ValueError(
        f"input size disagrees with signature for {case_name}:{input_info['name']}"
      )
    input_records.append(
      smoke._record_buffer(
        input_info["name"], dtype, input_info["shape"], "inputs.bin", len(data)
      )
      | {"byte_offset": offset}
    )
    input_parts.append(data)
    offset += len(data)
  inputs = b"".join(input_parts)
  (case_dir / "inputs.bin").write_bytes(inputs)
  expected_path = case_dir / "expected.bin"
  reference_command = [str(symiri), "--entry", "kernel"]
  if numeric:
    reference_command.append("--tensor-reference-stability")
  reference_command.extend(
    [
      "--tensor-input",
      str(case_dir / "inputs.bin"),
      "--tensor-output",
      str(expected_path),
      str(source_path),
    ]
  )
  _run(reference_command)
  expected = expected_path.read_bytes()
  smoke._write_manifest(
    case_dir,
    source.encode("utf-8"),
    inputs,
    expected,
    signature,
    input_records,
    case_name=case_name,
    softmax_lowering="direct",
    numeric=numeric,
  )
  manifest_path = case_dir / "case.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  manifest["property"]["id"] = property_id or f"regression.{case_name}"
  manifest["property"]["source"] = (
    "Complete output compared against the independent tensor reference."
  )
  manifest["evidence"]["kind"] = (
    "historical_regression" if case_name == "issue_11406" else "conformance_matrix"
  )
  if group:
    manifest["regression_group"] = group
  manifest["axis_choices"] = []
  if case_name.startswith("softmax_"):
    manifest["axis_choices"] = [int(case_name.rsplit("_axis", 1)[1].split("_", 1)[0])]
  manifest_path.write_text(
    json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  return case_dir


def _softmax_source(shape, spelling, declarations=None, statements=None, result="%out"):
  tensor_type = _tensor_type(shape, "f32")
  declarations = declarations or f"  let mut %out: {tensor_type} = undef;"
  statements = statements or f"  %out = tensor softmax %x, {spelling};"
  return _sir(f"%x: {tensor_type}", tensor_type, declarations, statements, result)


def _transpose_source(shape, permutation):
  out_shape = [shape[axis] for axis in permutation]
  input_type = _tensor_type(shape, "i32")
  output_type = _tensor_type(out_shape, "i32")
  perm = ", ".join(map(str, permutation))
  return _sir(
    f"%x: {input_type}",
    output_type,
    f"  let mut %out: {output_type} = undef;",
    f"  %out = tensor transpose %x, [{perm}];",
  )


def _composition_definitions():
  softmax_then_transpose = _sir(
    "%x: tensor [2, 4] f32",
    "tensor [4, 2] f32",
    "  let mut %soft: tensor [2, 4] f32 = undef;\n  let mut %out: tensor [4, 2] f32 = undef;",
    "  %soft = tensor softmax %x, 1;\n  %out = tensor transpose %soft, [1, 0];",
  )
  transpose_then_softmax = _sir(
    "%x: tensor [2, 4] f32",
    "tensor [4, 2] f32",
    "  let mut %trans: tensor [4, 2] f32 = undef;\n  let mut %out: tensor [4, 2] f32 = undef;",
    "  %trans = tensor transpose %x, [1, 0];\n  %out = tensor softmax %trans, 0;",
  )
  sum_then_expand = _sir(
    "%x: tensor [2, 4] i32",
    "tensor [2, 4] i32",
    "  let mut %rows: tensor [2] i32 = undef;\n  let mut %expanded: tensor [2, 1] i32 = undef;\n  let mut %out: tensor [2, 4] i32 = undef;",
    "  %rows = tensor sum %x, 1, drop;\n  %expanded = tensor expand %rows, 1;\n  %out = tensor broadcast %expanded, [2, 4];",
  )
  expand_then_sum = _sir(
    "%x: tensor [2, 4] i32",
    "tensor [2, 4] i32",
    "  let mut %expanded_input: tensor [2, 1, 4] i32 = undef;\n  let mut %rows: tensor [2, 1] i32 = undef;\n  let mut %out: tensor [2, 4] i32 = undef;",
    "  %expanded_input = tensor expand %x, 1;\n  %rows = tensor sum %expanded_input, 2, drop;\n  %out = tensor broadcast %rows, [2, 4];",
  )
  broadcast_then_softmax = _sir(
    "%a: tensor [2, 1] f32, %b: tensor [1, 4] f32",
    "tensor [2, 4] f32",
    "  let mut %broadcast: tensor [2, 4] f32 = undef;\n  let mut %out: tensor [2, 4] f32 = undef;",
    "  %broadcast = tensor add %a, %b;\n  %out = tensor softmax %broadcast, 1;",
  )
  softmax_then_broadcast = _sir(
    "%a: tensor [2, 1] f32, %b: tensor [1, 4] f32",
    "tensor [2, 4] f32",
    "  let mut %soft: tensor [1, 4] f32 = undef;\n  let mut %out: tensor [2, 4] f32 = undef;",
    "  %soft = tensor softmax %b, 1;\n  %out = tensor broadcast %soft, [2, 4];",
  )
  sum_expand_transpose = _sir(
    "%x: tensor [2, 4] i32",
    "tensor [2, 1] i32",
    "  let mut %rows: tensor [2] i32 = undef;\n  let mut %expanded: tensor [1, 2] i32 = undef;\n  let mut %out: tensor [2, 1] i32 = undef;",
    "  %rows = tensor sum %x, 1, drop;\n  %expanded = tensor expand %rows, 0;\n  %out = tensor transpose %expanded, [1, 0];",
  )
  transpose_sum_expand = _sir(
    "%x: tensor [2, 4] i32",
    "tensor [2, 1] i32",
    "  let mut %trans: tensor [4, 2] i32 = undef;\n  let mut %rows: tensor [2] i32 = undef;\n  let mut %out: tensor [2, 1] i32 = undef;",
    "  %trans = tensor transpose %x, [1, 0];\n  %rows = tensor sum %trans, 0, drop;\n  %out = tensor expand %rows, 1;",
  )
  axis_restoration_add = _sir(
    "%x: tensor [2, 2] i32, %bias: tensor [2, 2] i32",
    "tensor [2, 2] i32",
    "  let mut %rows: tensor [2] i32 = undef;\n  let mut %expanded: tensor [2, 1] i32 = undef;\n  let mut %out: tensor [2, 2] i32 = undef;",
    "  %rows = tensor sum %x, 1, drop;\n  %expanded = tensor expand %rows, 1;\n  %out = tensor add %expanded, %bias;",
  )
  return [
    (
      "transpose-softmax",
      "softmax-then-transpose",
      softmax_then_transpose,
      [("f32", _input_values((2, 4), "f32"), False)],
      True,
    ),
    (
      "transpose-softmax",
      "transpose-then-softmax",
      transpose_then_softmax,
      [("f32", _input_values((2, 4), "f32"), False)],
      True,
    ),
    (
      "sum-expand",
      "sum-then-expand",
      sum_then_expand,
      [("i32", _input_values((2, 4), "i32"), False)],
      False,
    ),
    (
      "sum-expand",
      "expand-then-sum",
      expand_then_sum,
      [("i32", _input_values((2, 4), "i32"), False)],
      False,
    ),
    (
      "broadcast-softmax",
      "broadcast-then-softmax",
      broadcast_then_softmax,
      [("f32", [2.0, -1.0], False), ("f32", [0.0, 2.0, 1.0, -1.0], False)],
      True,
    ),
    (
      "broadcast-softmax",
      "softmax-then-broadcast",
      softmax_then_broadcast,
      [("f32", [2.0, -1.0], False), ("f32", [0.0, 2.0, 1.0, -1.0], False)],
      True,
    ),
    (
      "transpose-sum-expand",
      "sum-expand-transpose",
      sum_expand_transpose,
      [("i32", _input_values((2, 4), "i32"), False)],
      False,
    ),
    (
      "transpose-sum-expand",
      "transpose-sum-expand",
      transpose_sum_expand,
      [("i32", _input_values((2, 4), "i32"), False)],
      False,
    ),
    (
      "axis-restoration",
      "sum-expand-add",
      axis_restoration_add,
      [("i32", [1, 2, 3, 4], False), ("i32", [10, 20, 30, 40], False)],
      False,
    ),
  ]


def _make_matrix(output_dir, *, symiri=smoke.SYMIRI):
  if output_dir.exists():
    manifest_path = output_dir / "regression-matrix.json"
    if not manifest_path.is_file():
      raise FileExistsError(
        f"refusing to reuse an unrecognized regression directory: {output_dir}"
      )
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing.get("schema") != "refractir.tensor-regression-matrix.v1":
      raise ValueError(f"unknown regression manifest in {output_dir}")
    if any(
      not (output_dir / name / "case.json").is_file()
      for name in existing.get("cases", [])
    ):
      raise ValueError(f"regression matrix is incomplete in {output_dir}")
    return existing
  output_dir.mkdir(parents=True)
  cases = []
  pair_paths = {}
  for case in softmax_sweep():
    shape = tuple(case["shape"])
    axis = case["axis"]
    spelling = case["axis_spelling"]
    name = f"softmax_{'x'.join(map(str, shape))}_axis{axis}_{'neg' if spelling.startswith('-') else 'pos'}"
    path = _build_case(
      output_dir,
      name,
      _softmax_source(shape, spelling),
      [("f32", _input_values(shape, "f32"), False)],
      numeric=True,
      group=f"softmax.{shape}.{axis}",
      symiri=symiri,
    )
    cases.append(name)
    pair_paths.setdefault(f"softmax.{shape}.{axis}", []).append(path)

  for case in transpose_sweep():
    shape = tuple(case["shape"])
    permutation = tuple(case["permutation"])
    name = (
      f"transpose_{'x'.join(map(str, shape))}_perm{'_'.join(map(str, permutation))}"
    )
    path = _build_case(
      output_dir,
      name,
      _transpose_source(shape, permutation),
      [("i32", _input_values(shape, "i32"), False)],
      numeric=False,
      group=f"transpose.rank{len(shape)}",
      symiri=symiri,
    )
    cases.append(name)

  for group, order, source, parameters, numeric in _composition_definitions():
    name = f"composition_{group}_{order}"
    path = _build_case(
      output_dir,
      name,
      source,
      parameters,
      numeric=numeric,
      group=group,
      symiri=symiri,
    )
    cases.append(name)
    pair_paths.setdefault(group, []).append(path)

  historical_source = (ROOT / "test/tensor/regressions/issue_11406.sir.in").read_text(
    encoding="utf-8"
  )
  historical = historical_case()
  path = _build_case(
    output_dir,
    "issue_11406",
    historical_source,
    [("f32", historical["input_values"], False)],
    numeric=True,
    group="triton.issue.11406",
    symiri=symiri,
  )
  case_manifest = json.loads((path / "case.json").read_text(encoding="utf-8"))
  case_manifest["historical"] = historical
  (path / "case.json").write_text(
    json.dumps(case_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  cases.append("issue_11406")

  pairs = []
  for group, paths in sorted(pair_paths.items()):
    if len(paths) < 2:
      continue
    input_blobs = {(path / "inputs.bin").read_bytes() for path in paths}
    expected_blobs = {(path / "expected.bin").read_bytes() for path in paths}
    if len(input_blobs) != 1 or len(expected_blobs) != 1:
      raise ValueError(
        f"equivalent source-order cases disagree before target execution: {group}"
      )
    pairs.append(
      {
        "id": group,
        "cases": [path.name for path in paths],
        "inputs_equal": True,
        "reference_equal": True,
      }
    )
  matrix = {
    "schema": "refractir.tensor-regression-matrix.v1",
    "cases": cases,
    "case_count": len(cases),
    "softmax_axis_cases": len(softmax_sweep()),
    "transpose_cases": len(transpose_sweep()),
    "composition_pairs": pairs,
    "historical": historical,
  }
  (output_dir / "regression-matrix.json").write_text(
    json.dumps(matrix, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  return matrix


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--symiri", type=Path, default=smoke.SYMIRI)
  args = parser.parse_args(argv)
  matrix = _make_matrix(args.output_dir.resolve(), symiri=args.symiri.resolve())
  print(
    json.dumps(
      {"case_count": matrix["case_count"], "cases": matrix["cases"]}, sort_keys=True
    )
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
