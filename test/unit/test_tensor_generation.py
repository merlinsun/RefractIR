"""CLI tests for deterministic, property-directed tensor generation."""

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from test.lib.tensor_case import CaseValidationError, load_case
from test.lib.tensor_coverage import source_signature
from test.unit.tensor_support import ROOT, check_source, run_typed

RYSMITH = Path(os.environ.get("REFRACTIR_RYSMITH", ROOT / "rysmith")).resolve()


# The only outcome a sweep may step over: the solver reached its budget without
# deciding. Every other status is a regression, so a sweep that skipped it
# silently would hide one.
SWEEPABLE_OUTCOMES = frozenset({"input_policy_unknown"})


def admit_outcome(test, status, seed, outcomes):
  """Record an unsuccessful seed, refusing an unexpected failure."""
  test.assertIn(
    status,
    SWEEPABLE_OUTCOMES,
    f"seed {seed} failed with {status}, which is not an undecided solver result",
  )
  outcomes.append(status)


def generate(output_dir, *arguments, seed=3, target="sir"):
  command = [
    str(RYSMITH),
    "--profile",
    "tensor",
    "--target",
    target,
    "--seed",
    str(seed),
    "--output-dir",
    str(output_dir),
    *arguments,
  ]
  return subprocess.run(
    command,
    capture_output=True,
    text=True,
    check=False,
    timeout=30,
  )


def generate_v2(output_dir, *, seed=3):
  return generate(
    output_dir,
    "-n",
    "1",
    "--tensor-dtype",
    "i32",
    "--tensor-profile-version",
    "v2",
    "--tensor-stratum",
    "multi-input-composition",
    seed=seed,
  )


def generate_v2_loop(output_dir, *, seed=3, dtype="i32", count=1):
  return generate(
    output_dir,
    "-n",
    str(count),
    "--tensor-dtype",
    dtype,
    "--tensor-profile-version",
    "v2",
    "--tensor-stratum",
    "loop-reduction",
    seed=seed,
  )


def generate_v2_stratum(output_dir, stratum, *, seed=3, dtype="f32", count=1):
  return generate(
    output_dir,
    "-n",
    str(count),
    "--tensor-dtype",
    dtype,
    "--tensor-profile-version",
    "v2",
    "--tensor-stratum",
    stratum,
    seed=seed,
  )


def generate_v3(
  output_dir, *, stratum=None, pattern=None, seed=3, dtype="i32", count=1, target="sir"
):
  arguments = [
    "-n",
    str(count),
    "--tensor-dtype",
    dtype,
    "--tensor-profile-version",
    "v3",
  ]
  if stratum is not None:
    arguments.extend(["--tensor-stratum", stratum])
  if pattern is not None:
    arguments.extend(["--tensor-input-pattern", pattern])
  return generate(output_dir, *arguments, seed=seed, target=target)


def generate_v4(
  output_dir, *, mode, seed=3, count=1, target="sir", synthesis="construction"
):
  return generate(
    output_dir,
    "-n",
    str(count),
    "--tensor-dtype",
    "f32",
    "--tensor-profile-version",
    "v4",
    "--tensor-input-pattern",
    mode,
    "--tensor-synthesis",
    synthesis,
    seed=seed,
    target=target,
  )


def generate_compositional(
  output_dir,
  *,
  profile="v2",
  dtype="i32",
  seed=3,
  count=1,
  target="sir",
  stratum=None,
  policy="ordinary",
  property_target=None,
  pattern=None,
  elements=None,
):
  arguments = [
    "-n",
    str(count),
    "--tensor-dtype",
    dtype,
    "--tensor-profile-version",
    profile,
    "--tensor-generator",
    "compositional",
    "--tensor-input-policy",
    policy,
  ]
  if property_target is not None:
    arguments.extend(["--tensor-property-target", str(property_target)])
  for index, value in elements or ():
    arguments.extend(
      ["--tensor-output-element", f"{','.join(str(axis) for axis in index)}={value}"]
    )
  if pattern is not None:
    arguments.extend(["--tensor-input-pattern", pattern])
  if stratum is not None:
    arguments.extend(["--tensor-stratum", stratum])
  return generate(output_dir, *arguments, seed=seed, target=target)


def row_major_coordinates(shape, offset):
  """The coordinate a row-major linear offset names in `shape`."""
  coordinates = [0] * len(shape)
  for axis in range(len(shape) - 1, -1, -1):
    coordinates[axis] = offset % shape[axis]
    offset //= shape[axis]
  return coordinates


def case_output_values(case):
  """The i32 reference output of a case, in row-major order."""
  record = case.manifest["output"]
  if record["dtype"] != "i32":
    raise AssertionError("the element-property tests drive integer outputs only")
  return list(struct.unpack(f"<{len(case.expected) // 4}i", case.expected))


def read_summary(directory):
  summary = Path(directory) / "generation.json"
  if not summary.is_file():
    raise AssertionError("tensor profile did not write its generation summary")
  return json.loads(summary.read_text(encoding="utf-8"))


def entry_statements(source):
  """Return the assignment lines of the single entry block, in source order."""
  match = re.search(r"(?ms)^\^entry:\n(.*)$", source)
  if match is None:
    raise AssertionError("generated source has no entry block")
  return [line.strip() for line in match.group(1).splitlines() if line.strip()]


def referenced_locals(statement):
  return set(re.findall(r"%[A-Za-z_][\w.$-]*", statement))


def tensor_local_shapes(source):
  """Map every parameter and local to its declared shape, scalar shapes empty."""
  shapes = {}
  header = source.split(") :", 1)[0]
  for match in re.finditer(r"(%[A-Za-z_][\w.$-]*): tensor \[([^\]]*)\]", header):
    shapes[match.group(1)] = [int(value) for value in match.group(2).split(",")]
  for match in re.finditer(r"(%[A-Za-z_][\w.$-]*): (?:i32|f32)", header):
    shapes.setdefault(match.group(1), [])
  for match in re.finditer(
    r"let mut (%[A-Za-z_][\w.$-]*): tensor \[([^\]]*)\]", source
  ):
    shapes[match.group(1)] = [int(value) for value in match.group(2).split(",")]
  for match in re.finditer(r"let mut (%[A-Za-z_][\w.$-]*): (?:i32|f32)", source):
    shapes.setdefault(match.group(1), [])
  return shapes


def sum_records(source):
  """Return (operand, operand shape, normalized axis, keep) per tensor sum."""
  shapes = tensor_local_shapes(source)
  records = []
  for match in re.finditer(
    r"tensor sum (%[A-Za-z_][\w.$-]*), (-?\d+), (keep|drop)", source
  ):
    name, axis, keep = match.group(1), int(match.group(2)), match.group(3)
    shape = shapes[name]
    normalized = axis if axis >= 0 else axis + len(shape)
    records.append((name, shape, normalized, keep))
  return records


def local_consumers(statements):
  """Map each referenced local to the indices of the statements that read it."""
  consumers = {}
  for index, statement in enumerate(statements):
    assignment = re.match(r"(%[A-Za-z_][\w.$-]*)\s*=", statement)
    target = assignment.group(1) if assignment else None
    for name in referenced_locals(statement):
      if name == target:
        continue
      consumers.setdefault(name, []).append(index)
  return consumers


def independently_sum_loop_case(case):
  """Compute generated row-loop results from logical input values and source."""
  manifest = case.manifest
  buffer = manifest["inputs"][0]
  rows, columns = buffer["shape"]
  dtype_code = "i" if buffer["dtype"] == "i32" else "f"
  input_values = struct.unpack(f"<{rows * columns}{dtype_code}", case.inputs)
  bounds = manifest["execution"]["loop_bounds"]
  iterations = 1
  for bound in bounds:
    iterations *= bound
  source = case.program.decode("utf-8")
  updates = source.count("%acc = tensor add %acc, %rows;")
  repeat_scale = 1
  if "tensor transpose %x" in source:
    repeat_scale = columns
  result = []
  for row in range(rows):
    row_sum = sum(input_values[row * columns : (row + 1) * columns])
    result.append(row_sum * repeat_scale * iterations * updates)
  return result


def find_multi_input_loop_case(output_dir):
  """Return a loop that reduces the two unchanged runtime inputs."""
  for case_dir in sorted(Path(output_dir).glob("case_*")):
    case = load_case(case_dir)
    source = case.program.decode("utf-8")
    if (
      len(case.manifest["inputs"]) == 2
      and case.manifest["execution"]["loop_bounds"]
      and "tensor sum %x," in source
      and "tensor sum %y," in source
    ):
      return case
  return None


def find_loop_carried_reduction_case(output_dir):
  """Return a loop case that reduces its changing tensor accumulator state."""
  for case_dir in sorted(Path(output_dir).glob("case_*")):
    case = load_case(case_dir)
    source = case.program.decode("utf-8")
    if (
      "tensor sum %acc_x, 1, drop" in source and "tensor sum %acc_y, 0, drop" in source
    ):
      return case
  return None


def loop_carried_tensor_updates(source):
  """Map self-updated tensor loop variables to their body right-hand sides."""
  tensor_variables = set(
    re.findall(r"let mut (%[A-Za-z_][\w.$-]*):\s*tensor\b", source)
  )
  body = re.search(r"(?ms)^\^body:\n(.*?)(?=\^[A-Za-z_][A-Za-z0-9_]*:)", source)
  if body is None:
    return {}
  updates = {}
  for line in body.group(1).splitlines():
    assignment = re.match(
      r"\s*(%[A-Za-z_][\w.$-]*)\s*=\s*tensor\s+\w+\s+(.*?);\s*$", line
    )
    if assignment is None:
      continue
    name, rhs = assignment.groups()
    if name in tensor_variables and name in re.findall(r"%[A-Za-z_][\w.$-]*", rhs):
      updates.setdefault(name, []).append(rhs)
  return updates


class TensorGenerationTests(unittest.TestCase):
  def setUp(self):
    self.assertTrue(RYSMITH.is_file(), f"rysmith executable not found: {RYSMITH}")

  def test_property_precedes_candidate_acceptance(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate(
        output_dir,
        "-n",
        "1",
        "--tensor-dtype",
        "f32",
        "--tensor-property",
        "softmax-dominance",
        "--tensor-synthesis",
        "construction",
        "--max-retries",
        "8",
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      summary = read_summary(output_dir)
      attempts = summary["attempts"]
      self.assertTrue(any(item["outcome"] == "rejected" for item in attempts), attempts)
      self.assertTrue(any(item["outcome"] == "accepted" for item in attempts), attempts)
      property_ids = {item["property_id"] for item in attempts}
      self.assertEqual(property_ids, {"softmax-dominance"})
      source = (output_dir / "case_000000" / "program.sir").read_text(encoding="utf-8")
      self.assertIn("require", source)
      self.assertIn("require %last > 0.8;", source)
      manifest = json.loads(
        (output_dir / "case_000000" / "case.json").read_text(encoding="utf-8")
      )
      self.assertEqual(manifest["property"]["threshold_f32_bits"], "3f4ccccd")

  def test_seed_reproduces_case_bytes(self):
    with tempfile.TemporaryDirectory() as directory:
      first = Path(directory) / "first"
      second = Path(directory) / "second"
      args = ("-n", "2", "--tensor-dtype", "i32", "--tensor-property", "coordinates")
      first_result = generate(first, *args)
      second_result = generate(second, *args)

      self.assertEqual(first_result.returncode, 0, first_result.stderr)
      self.assertEqual(second_result.returncode, 0, second_result.stderr)
      self.assertEqual(read_summary(first), read_summary(second))
      for case_name in ("case_000000", "case_000001"):
        for artifact in ("program.sir", "inputs.bin", "expected.bin", "case.json"):
          with self.subTest(case=case_name, artifact=artifact):
            self.assertEqual(
              (first / case_name / artifact).read_bytes(),
              (second / case_name / artifact).read_bytes(),
            )

  def test_tensor_profile_bypasses_scalar_checksum(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate(output_dir, "-n", "1", "--tensor-dtype", "i32")

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case_dir = output_dir / "case_000000"
      self.assertTrue((case_dir / "case.json").is_file())
      self.assertTrue((case_dir / "program.sir").is_file())
      self.assertTrue((case_dir / "inputs.bin").is_file())
      self.assertTrue((case_dir / "expected.bin").is_file())
      self.assertFalse(any(case_dir.glob("func_*.json")))
      source = (case_dir / "program.sir").read_text(encoding="utf-8")
      self.assertNotIn("SOLVED:", source)
      self.assertNotIn("crc32", source.lower())

  def test_requested_route_has_no_silent_fallback(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate(
        output_dir,
        "-n",
        "1",
        "--tensor-dtype",
        "f32",
        "--tensor-property",
        "softmax-dominance",
        "--tensor-synthesis",
        "exact",
      )

      self.assertNotEqual(result.returncode, 0)
      summary = read_summary(output_dir)
      self.assertEqual(summary["status"], "unsupported_theory")
      self.assertEqual(summary["emitted"], 0)
      self.assertFalse((output_dir / "case_000000").exists())
      self.assertIn("exact", summary["reason"])

  def test_attempt_budget_reports_shortfall(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate(
        output_dir,
        "-n",
        "1",
        "--tensor-dtype",
        "i32",
        "--max-retries",
        "0",
      )

      self.assertNotEqual(result.returncode, 0)
      summary = read_summary(output_dir)
      self.assertEqual(summary["status"], "shortfall")
      self.assertEqual(summary["requested"], 1)
      self.assertEqual(summary["emitted"], 0)
      self.assertEqual(summary["rejected"], 0)

  def test_target_sir_is_reemittable(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate(output_dir, "-n", "1", "--tensor-dtype", "i32")

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = output_dir / "case_000000"
      manifest = json.loads((case / "case.json").read_text(encoding="utf-8"))
      self.assertEqual(manifest["profile"], "tensor-v1")
      self.assertFalse((case / "kernel.py").exists())
      emitted = subprocess.run(
        [
          str(ROOT / "symirc"),
          "--target",
          "triton",
          "--entry",
          "kernel",
          "-o",
          str(case / "kernel.py"),
          str(case / "program.sir"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
      )

      self.assertEqual(emitted.returncode, 0, emitted.stderr)
      self.assertTrue((case / "kernel.py").is_file())
      (case / "case.json").write_text(json.dumps(manifest), encoding="utf-8")

  def test_exact_i32_route_materializes_a_replayable_case(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate(
        output_dir,
        "-n",
        "1",
        "--tensor-dtype",
        "i32",
        "--tensor-property",
        "coordinates",
        "--tensor-synthesis",
        "exact",
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case_dir = output_dir / "case_000000"
      case = load_case(case_dir)
      manifest = case.manifest
      self.assertEqual(manifest["evidence"]["kind"], "exact_smt")
      self.assertEqual(manifest["generator"]["synthesis"], "exact")
      replay, actual = run_typed(case.program.decode("utf-8"), case.inputs)
      self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
      self.assertEqual(actual, case.expected)

  def test_softmax_samples_shapes_axes_and_operator_compositions(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate(
        output_dir,
        "-n",
        "24",
        "--tensor-dtype",
        "f32",
        "--tensor-property",
        "softmax-dominance",
        "--tensor-synthesis",
        "construction",
        seed=0,
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      sources = set()
      shapes = set()
      axes = set()
      operators = set()
      for index in range(24):
        case = Path(directory) / "cases" / f"case_{index:06d}"
        source = (case / "program.sir").read_text(encoding="utf-8")
        manifest = json.loads((case / "case.json").read_text(encoding="utf-8"))
        sources.add(source)
        shapes.add(tuple(manifest["inputs"][0]["shape"]))
        axes.update(item["axis"] for item in manifest["shape_axes"])
        operators.update(manifest["operator_coverage"])

      self.assertGreaterEqual(len(sources), 8)
      self.assertGreaterEqual(len(shapes), 6)
      self.assertGreaterEqual(len(axes), 3)
      self.assertTrue(any(1 in shape for shape in shapes), shapes)
      self.assertTrue(any(", -" in source for source in sources), sources)
      self.assertTrue(
        {
          "softmax",
          "transpose",
          "rmax",
          "expand",
          "broadcast",
          "min",
          "max",
          "exp",
          "sum",
          "div",
        }
        <= operators,
        operators,
      )

  def test_exact_sum_target_keeps_solved_input_in_runtime_buffer(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate(
        output_dir,
        "-n",
        "1",
        "--tensor-dtype",
        "i32",
        "--tensor-property",
        "sum-target",
        "--tensor-synthesis",
        "exact",
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case_dir = output_dir / "case_000000"
      case = load_case(case_dir)
      source = case.program.decode("utf-8")
      self.assertIn("fun @kernel(%x: tensor [2, 4] i32)", source)
      self.assertNotIn("sym %?x", source)
      self.assertEqual(len(case.inputs), 32)
      replay, actual = run_typed(source, case.inputs)
      self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
      self.assertEqual(actual, case.expected)

  def test_v2_two_runtime_inputs_affect_the_return(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2(output_dir, seed=17)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      self.assertEqual(len(case.manifest["inputs"]), 2)
      self.assertEqual([item["name"] for item in case.manifest["inputs"]], ["a", "b"])
      baseline, baseline_bytes = run_typed(
        case.program.decode("utf-8"), case.inputs, tensor_profile="v2"
      )
      self.assertEqual(baseline.returncode, 0, baseline.stderr)
      self.assertEqual(baseline_bytes, case.expected)
      for buffer in case.manifest["inputs"]:
        changed = bytearray(case.inputs)
        offset = buffer["byte_offset"]
        length = buffer["byte_length"]
        changed[offset : offset + length] = bytes(length)
        replay, changed_output = run_typed(
          case.program.decode("utf-8"), bytes(changed), tensor_profile="v2"
        )
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertNotEqual(changed_output, case.expected)

  def test_v2_shared_intermediate_has_multiple_live_consumers(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2(output_dir)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      source = (output_dir / "case_000000" / "program.sir").read_text(encoding="utf-8")
      self.assertGreaterEqual(source.count("%transposed"), 4, source)
      self.assertIn("tensor sum", source)
      self.assertRegex(
        source,
        r"%out = tensor (add|sub|min|max) %transposed, %broadcasted;",
      )

  def test_v2_transpose_reduce_broadcast_arithmetic_shapes_match_reference(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2(output_dir)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      source = case.program.decode("utf-8")
      operators = case.manifest["operator_coverage"]
      self.assertTrue({"transpose", "sum", "expand", "broadcast"} <= set(operators))
      self.assertTrue(set(operators) & {"add", "sub", "min", "max"})
      self.assertLess(source.index("tensor transpose"), source.index("tensor sum"))
      self.assertLess(source.index("tensor sum"), source.index("tensor expand"))
      self.assertLess(source.index("tensor expand"), source.index("tensor broadcast"))
      self.assertLess(source.index("tensor broadcast"), source.index("%out = tensor "))
      input_shape = case.manifest["inputs"][0]["shape"]
      transposed_shape = case.manifest["execution"]["full_shapes"][1]
      permutation = [
        int(value)
        for value in re.search(r"tensor transpose %[^,]+, \[([^]]+)\]", source)
        .group(1)
        .split(",")
      ]
      self.assertNotEqual(permutation, list(range(len(input_shape))))
      self.assertEqual(transposed_shape, [input_shape[axis] for axis in permutation])
      self.assertEqual(case.signature["output"]["shape"], transposed_shape)
      self.assertEqual(case.manifest["execution"]["full_shapes"][0], input_shape)
      self.assertEqual(case.manifest["execution"]["full_shapes"][1], transposed_shape)
      replay, actual = run_typed(source, case.inputs, tensor_profile="v2")
      self.assertEqual(replay.returncode, 0, replay.stderr)
      self.assertEqual(actual, case.expected)

  def test_v2_seed_reproduces_graph_inputs_output_and_layout(self):
    with tempfile.TemporaryDirectory() as directory:
      first = Path(directory) / "first"
      second = Path(directory) / "second"
      first_result = generate_v2(first, seed=71)
      second_result = generate_v2(second, seed=71)

      self.assertEqual(first_result.returncode, 0, first_result.stderr)
      self.assertEqual(second_result.returncode, 0, second_result.stderr)
      for name in ("program.sir", "inputs.bin", "expected.bin", "case.json"):
        self.assertEqual(
          (first / "case_000000" / name).read_bytes(),
          (second / "case_000000" / name).read_bytes(),
          name,
        )
      first_case = json.loads((first / "case_000000" / "case.json").read_text())
      first_source = (first / "case_000000" / "program.sir").read_text()
      self.assertEqual(
        first_case["coverage"]["source_signature"], source_signature(first_source)
      )

  def test_v2_manifest_uses_profile_schema_and_sha256_artifact_hashes(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2(output_dir)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case_dir = output_dir / "case_000000"
      manifest = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
      self.assertEqual(manifest["schema"], "refractir.tensor-case.v2")
      self.assertEqual(manifest["profile"], "tensor-v2")
      self.assertEqual(
        manifest["coverage"]["primary_stratum"], "multi-input-composition"
      )
      self.assertEqual(load_case(case_dir).manifest, manifest)

  def test_v2_loop_stratum_emits_counted_accumulator_reduction(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_loop(output_dir, seed=83)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      source = case.program.decode("utf-8")
      self.assertRegex(source, r"br %i < [234], \^body, \^exit;")
      self.assertRegex(source, r"tensor sum %(?:x|transposed|acc_x|acc_y), [01], drop")
      self.assertRegex(
        source,
        r"%acc(?:_[xy])? = tensor add %acc(?:_[xy])?, "
        r"%(?:rows(?:_[xy])?|broadcast_[xy]);",
      )
      self.assertEqual(case.manifest["coverage"]["primary_stratum"], "loop-reduction")

  def test_v2_loop_input_is_runtime_and_physically_strided(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_loop(output_dir)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      shape = case.manifest["inputs"][0]["shape"]
      self.assertEqual([item["name"] for item in case.manifest["inputs"]], ["x"])
      self.assertEqual(len(shape), 2)
      self.assertEqual(case.manifest["inputs"][0]["element_strides"], [1, shape[0]])
      self.assertEqual(case.manifest["inputs"][0]["axis_order"], [0, 1])
      self.assertEqual(len(case.inputs), shape[0] * shape[1] * 4)
      self.assertIn("%x", case.program.decode("utf-8"))

  def test_v2_loop_manifest_records_bound_and_execution_count(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_loop(output_dir)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      execution = case.manifest["execution"]
      bounds = execution["loop_bounds"]
      self.assertIn(len(bounds), {1, 2})
      self.assertTrue(all(2 <= bound <= 4 for bound in bounds), bounds)
      expected_body_executions = (
        bounds[0] if len(bounds) == 1 else bounds[0] + bounds[0] * bounds[1]
      )
      self.assertEqual(execution["total_body_executions"], expected_body_executions)
      self.assertLessEqual(execution["total_body_executions"], 8)
      self.assertEqual(execution["full_shapes"][0], case.manifest["inputs"][0]["shape"])
      self.assertEqual(execution["full_shapes"][-1], case.manifest["output"]["shape"])

  def test_v2_loop_seed_reproduces_program_inputs_and_expected_output(self):
    with tempfile.TemporaryDirectory() as directory:
      first = Path(directory) / "first"
      second = Path(directory) / "second"
      first_result = generate_v2_loop(first, seed=109)
      second_result = generate_v2_loop(second, seed=109)

      self.assertEqual(first_result.returncode, 0, first_result.stderr)
      self.assertEqual(second_result.returncode, 0, second_result.stderr)
      for name in ("program.sir", "inputs.bin", "expected.bin", "case.json"):
        self.assertEqual(
          (first / "case_000000" / name).read_bytes(),
          (second / "case_000000" / name).read_bytes(),
          name,
        )

  def test_v2_loop_expected_values_match_independent_row_sum(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_loop(output_dir, seed=211)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      scalar_code = "i" if case.manifest["output"]["dtype"] == "i32" else "f"
      actual = struct.unpack(f"<{len(case.expected) // 4}{scalar_code}", case.expected)
      self.assertEqual(list(actual), independently_sum_loop_case(case))

  def test_v2_loop_schedule_emits_nested_counted_loop(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_loop(output_dir, seed=2, count=20)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      nested_sources = [
        (output_dir / f"case_{index:06d}" / "program.sir").read_text(encoding="utf-8")
        for index in range(20)
        if "^outer_header:"
        in (output_dir / f"case_{index:06d}" / "program.sir").read_text(
          encoding="utf-8"
        )
      ]
      self.assertTrue(nested_sources, "no nested counted loop was generated")
      self.assertTrue(
        all(
          "^inner_header:" in source and "%inner = 0;" in source
          for source in nested_sources
        )
      )

  def test_v2_nested_loop_records_total_body_execution_bound(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_loop(output_dir, seed=4, count=20)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      nested_cases = [
        load_case(output_dir / f"case_{index:06d}")
        for index in range(20)
        if len(
          json.loads(
            (output_dir / f"case_{index:06d}" / "case.json").read_text(encoding="utf-8")
          )["execution"]["loop_bounds"]
        )
        == 2
      ]
      self.assertTrue(nested_cases, "no nested counted loop was generated")
      execution = nested_cases[0].manifest["execution"]
      outer, inner = execution["loop_bounds"]
      self.assertEqual(outer, 2)
      self.assertIn(inner, {2, 3})
      self.assertEqual(execution["total_body_executions"], outer + outer * inner)

  def test_v2_nested_loop_resets_inner_counter_on_each_outer_entry(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_loop(output_dir, seed=6, count=20)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      nested_sources = [
        (output_dir / f"case_{index:06d}" / "program.sir").read_text(encoding="utf-8")
        for index in range(20)
        if "^outer_header:"
        in (output_dir / f"case_{index:06d}" / "program.sir").read_text(
          encoding="utf-8"
        )
      ]
      self.assertTrue(nested_sources, "no nested counted loop was generated")
      source = nested_sources[0]
      outer_body = source.split("^outer_body:", maxsplit=1)[1].split(
        "^inner_header:", maxsplit=1
      )[0]
      self.assertIn("%inner = 0;", outer_body)

  def test_v2_loop_accumulator_matches_all_loop_iterations(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_loop(output_dir, seed=8)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      expected = struct.unpack(f"<{len(case.expected) // 4}i", case.expected)
      self.assertEqual(list(expected), independently_sum_loop_case(case))

  def test_v2_loop_seed_reproduces_all_case_artifacts(self):
    with tempfile.TemporaryDirectory() as directory:
      first = Path(directory) / "first"
      second = Path(directory) / "second"
      first_result = generate_v2_loop(first, seed=10, count=20)
      second_result = generate_v2_loop(second, seed=10, count=20)

      self.assertEqual(first_result.returncode, 0, first_result.stderr)
      self.assertEqual(second_result.returncode, 0, second_result.stderr)
      first_bounds = []
      for index in range(20):
        first_case = first / f"case_{index:06d}"
        second_case = second / f"case_{index:06d}"
        manifest = json.loads((first_case / "case.json").read_text(encoding="utf-8"))
        first_bounds.append(manifest["execution"]["loop_bounds"])
        for name in ("program.sir", "inputs.bin", "expected.bin", "case.json"):
          self.assertEqual(
            (first_case / name).read_bytes(),
            (second_case / name).read_bytes(),
            f"{index}: {name}",
          )
      self.assertTrue(any(len(bounds) == 2 for bounds in first_bounds))

  def test_v2_loop_f32_reference_is_stable_and_runtime_backed(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_loop(output_dir, seed=307, dtype="f32")

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      self.assertEqual(case.manifest["comparison"]["mode"], "numeric")
      self.assertEqual(case.manifest["reference"]["stable"], True)
      replay, actual = run_typed(
        case.program.decode("utf-8"),
        case.inputs,
        tensor_profile="v2",
        reference_stability=True,
      )
      self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
      self.assertEqual(actual, case.expected)

  def test_v2_compositions_sample_shapes_axes_and_source_structures(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate(
        output_dir,
        "-n",
        "12",
        "--tensor-dtype",
        "i32",
        "--tensor-profile-version",
        "v2",
        "--tensor-stratum",
        "multi-input-composition",
        seed=0,
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      signatures = set()
      input_shapes = set()
      reductions = set()
      for index in range(12):
        case_dir = output_dir / f"case_{index:06d}"
        case = load_case(case_dir)
        signatures.add(case.manifest["coverage"]["source_signature"])
        input_shapes.add(tuple(case.manifest["inputs"][0]["shape"]))
        reductions.update(item["axis"] for item in case.manifest["shape_axes"])
      self.assertGreaterEqual(len(signatures), 4)
      self.assertGreaterEqual(len(input_shapes), 3)
      self.assertGreaterEqual(len(reductions), 2)

  def test_v2_layout_stratum_generates_unary_bit_preserving_case(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_stratum(output_dir, "bit-preserving-f32-layout", seed=401)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      manifest = case.manifest
      self.assertEqual(
        manifest["coverage"]["primary_stratum"], "bit-preserving-f32-layout"
      )
      self.assertEqual(len(manifest["inputs"]), 1)
      self.assertEqual(manifest["comparison"]["mode"], "exact")
      self.assertEqual(manifest["output"]["dtype"], "f32")
      self.assertIn("tensor transpose", case.program.decode("utf-8"))

  def test_v2_layout_case_preserves_signed_zero_and_subnormal_bits(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_stratum(output_dir, "bit-preserving-f32-layout", seed=409)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      self.assertIn(struct.pack("<I", 0x80000000), case.inputs)
      self.assertIn(struct.pack("<I", 0x00000001), case.inputs)
      replay, actual = run_typed(
        case.program.decode("utf-8"), case.inputs, tensor_profile="v2"
      )
      self.assertEqual(replay.returncode, 0, replay.stderr)
      self.assertEqual(actual, case.expected)

  def test_v2_exact_reduction_records_complete_output_proof(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_stratum(output_dir, "exact-f32-reduction", seed=419)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      self.assertEqual(case.manifest["comparison"]["mode"], "exact")
      proof = case.manifest["reference"]["exactness"]
      self.assertEqual(proof["kind"], "integer-valued-f32-reduction")
      self.assertTrue(proof["complete_output"])
      self.assertLessEqual(proof["max_abs_leaf_sum"], 2**24)
      self.assertEqual(proof["signed_zero"], "positive-zero-only")

  def test_v2_exact_reduction_uses_runtime_input_and_bitwise_reference(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_stratum(output_dir, "exact-f32-reduction", seed=421)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      self.assertIn("%x", case.program.decode("utf-8"))
      self.assertEqual(case.manifest["inputs"][0]["name"], "x")
      replay, actual = run_typed(
        case.program.decode("utf-8"),
        case.inputs,
        tensor_profile="v2",
        reference_stability=True,
      )
      self.assertEqual(replay.returncode, 0, replay.stderr)
      self.assertEqual(actual, case.expected)

  def test_v2_exact_reduction_proof_does_not_extend_through_exp(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_stratum(output_dir, "exact-f32-reduction", seed=427)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case_dir = output_dir / "case_000000"
      case = load_case(case_dir)
      output_shape = case.manifest["output"]["shape"]
      output_type = (
        "f32"
        if not output_shape
        else f"tensor [{', '.join(map(str, output_shape))}] f32"
      )
      source = case.program.decode("utf-8")
      source = source.replace(
        f"  let mut %out: {output_type} = undef;",
        f"  let mut %tmp: {output_type} = undef;\n  let mut %out: {output_type} = undef;",
      )
      source = source.replace("%out = tensor sum", "%tmp = tensor sum")
      source = source.replace("  ret %out;", "  %out = tensor exp %tmp;\n  ret %out;")
      (case_dir / "program.sir").write_text(source, encoding="utf-8")
      manifest = case.manifest
      digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
      manifest["source"]["sha256"] = digest
      manifest["files"]["program.sir"]["bytes"] = len(source.encode("utf-8"))
      manifest["files"]["program.sir"]["sha256"] = digest
      manifest["coverage"]["source_signature"] = source_signature(source)
      manifest["operator_coverage"] = ["sum", "exp"]
      (case_dir / "case.json").write_text(json.dumps(manifest), encoding="utf-8")

      with self.assertRaisesRegex(CaseValidationError, "exact f32 proof"):
        load_case(case_dir)

  def test_v2_f32_composition_with_arithmetic_remains_numerical(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_stratum(
        output_dir, "multi-input-composition", seed=431, dtype="f32"
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      self.assertEqual(case.manifest["comparison"]["mode"], "numeric")
      self.assertNotIn("exactness", case.manifest["reference"])

  def test_v2_primary_quota_constructs_layout_and_exact_reductions(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_stratum(output_dir, "", seed=443, dtype="f32", count=10)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      strata = [
        load_case(output_dir / f"case_{index:06d}").manifest["coverage"][
          "primary_stratum"
        ]
        for index in range(10)
      ]
      self.assertEqual(strata.count("bit-preserving-f32-layout"), 2)
      self.assertEqual(strata.count("exact-f32-reduction"), 2)
      quotas = read_summary(output_dir)["stratum_quotas"]
      self.assertEqual([item["shortfall"] for item in quotas], [0] * 5)

  def test_v2_untargeted_stratum_uses_its_own_unary_graph(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_stratum(
        output_dir, "untargeted-exploration", seed=449, dtype="i32"
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      self.assertEqual(len(case.manifest["inputs"]), 1)
      self.assertEqual(
        case.manifest["coverage"]["primary_stratum"], "untargeted-exploration"
      )
      self.assertIn("tensor sum", case.program.decode("utf-8"))

  def test_v2_launch_warp_count_is_selected_from_case_seed(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      seed = 55
      result = generate_v2_stratum(
        output_dir, "multi-input-composition", seed=seed, dtype="i32"
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      manifest = json.loads(
        (output_dir / "case_000000" / "case.json").read_text(encoding="utf-8")
      )
      case_seed = manifest["source"]["seed"]
      self.assertGreaterEqual(case_seed, seed)
      self.assertIn(manifest["launch"]["num_warps"], {1, 2, 4, 8})

  def test_v2_warp_probe_initial_record_matches_launch(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      seed = 62
      result = generate_v2_stratum(
        output_dir, "multi-input-composition", seed=seed, dtype="i32"
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      manifest = json.loads(
        (output_dir / "case_000000" / "case.json").read_text(encoding="utf-8")
      )
      self.assertEqual(
        manifest["execution"]["warp_probe"]["num_warps"],
        manifest["launch"]["num_warps"],
      )

  def test_v2_warp_schedule_covers_all_profile_counts(self):
    selected = []
    with tempfile.TemporaryDirectory() as directory:
      for seed in range(16):
        output_dir = Path(directory) / str(seed)
        result = generate_v2_stratum(
          output_dir, "multi-input-composition", seed=seed, dtype="i32"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads(
          (output_dir / "case_000000" / "case.json").read_text(encoding="utf-8")
        )
        selected.append(manifest["launch"]["num_warps"])

    self.assertEqual(set(selected), {1, 2, 4, 8})

  def test_v2_adjacent_case_seeds_change_warp_setting(self):
    selected = []
    with tempfile.TemporaryDirectory() as directory:
      for seed in (100, 101):
        output_dir = Path(directory) / str(seed)
        result = generate_v2_stratum(
          output_dir, "multi-input-composition", seed=seed, dtype="i32"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads(
          (output_dir / "case_000000" / "case.json").read_text(encoding="utf-8")
        )
        selected.append(manifest["launch"]["num_warps"])

      self.assertNotEqual(selected[0], selected[1])

  def test_v4_paired_input_modes_keep_graph_and_launch_fixed(self):
    with tempfile.TemporaryDirectory() as directory:
      ordinary_dir = Path(directory) / "ordinary"
      boundary_dir = Path(directory) / "boundary"
      ordinary_result = generate_v4(ordinary_dir, mode="ordinary", seed=907)
      boundary_result = generate_v4(boundary_dir, mode="boundary-value", seed=907)

      self.assertEqual(
        ordinary_result.returncode, 0, ordinary_result.stdout + ordinary_result.stderr
      )
      self.assertEqual(
        boundary_result.returncode, 0, boundary_result.stdout + boundary_result.stderr
      )
      ordinary = load_case(ordinary_dir / "case_000000")
      boundary = load_case(boundary_dir / "case_000000")
      self.assertEqual(ordinary.program, boundary.program)
      self.assertEqual(
        ordinary.manifest["coverage"]["source_signature"],
        boundary.manifest["coverage"]["source_signature"],
      )
      self.assertEqual(ordinary.manifest["launch"], boundary.manifest["launch"])
      self.assertEqual(ordinary.manifest["launch"]["num_warps"], 1)
      self.assertEqual(ordinary.manifest["generator"]["input_mode"], "ordinary")
      self.assertEqual(boundary.manifest["generator"]["input_mode"], "boundary-value")
      self.assertNotEqual(ordinary.inputs, boundary.inputs)
      for case in (ordinary, boundary):
        self.assertEqual(case.manifest["schema"], "refractir.tensor-case.v4")
        self.assertEqual(case.manifest["comparison"]["zero_sign_exact"], True)
        replay, actual = run_typed(
          case.program.decode("utf-8"), case.inputs, tensor_profile="v4"
        )
        self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
        self.assertEqual(actual, case.expected)

  def test_v4_boundary_schedule_covers_each_declared_family(self):
    required = {
      "signed-zero",
      "small-integer",
      "power-of-two",
      "repeated-value",
      "exact-multiple",
      "mixed-sign",
    }
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v4(output_dir, mode="boundary-value", seed=911, count=12)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      cases = [load_case(output_dir / f"case_{index:06d}") for index in range(12)]
      patterns = {
        case.manifest["coverage"]["input_behavior"]["pattern"] for case in cases
      }
      self.assertTrue(required <= patterns, patterns)
      for case in cases:
        self.assertEqual(
          case.manifest["coverage"]["input_behavior"]["input_mode"], "boundary-value"
        )
        self.assertEqual(case.manifest["launch"]["num_warps"], 1)

  def test_v4_rejects_solver_synthesis_route(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v4(output_dir, mode="ordinary", synthesis="exact")

      self.assertNotEqual(result.returncode, 0)
      summary = read_summary(output_dir)
      self.assertEqual(summary["status"], "unsupported_route")
      self.assertEqual(summary["emitted"], 0)

  def test_v2_warp_setting_is_repeatable_for_same_seed(self):
    selected = []
    with tempfile.TemporaryDirectory() as directory:
      for repeat in range(2):
        output_dir = Path(directory) / str(repeat)
        result = generate_v2_stratum(
          output_dir, "multi-input-composition", seed=111, dtype="i32"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads(
          (output_dir / "case_000000" / "case.json").read_text(encoding="utf-8")
        )
        selected.append((manifest["source"]["seed"], manifest["launch"]["num_warps"]))

    self.assertEqual(selected[0], selected[1])
    self.assertIn(selected[0][1], {1, 2, 4, 8})

  def test_v2_composition_graph_depth_varies_with_seed(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_stratum(
        output_dir,
        "multi-input-composition",
        seed=17,
        dtype="i32",
        count=8,
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      depths = {
        len(
          json.loads(
            (output_dir / f"case_{index:06d}" / "case.json").read_text(encoding="utf-8")
          )["operator_coverage"]
        )
        for index in range(8)
      }
      self.assertGreater(len(depths), 1)
      self.assertTrue(all(3 <= depth <= 12 for depth in depths), depths)

  def test_v2_composition_samples_min_and_max_operations(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_stratum(
        output_dir,
        "multi-input-composition",
        seed=23,
        dtype="i32",
        count=24,
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      operators = set()
      for index in range(24):
        manifest = json.loads(
          (output_dir / f"case_{index:06d}" / "case.json").read_text(encoding="utf-8")
        )
        operators.update(manifest["operator_coverage"])
      self.assertTrue({"min", "max"}.issubset(operators), operators)

  def test_v2_f32_composition_can_include_exp(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_stratum(
        output_dir,
        "multi-input-composition",
        seed=29,
        dtype="f32",
        count=24,
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      operators = set()
      for index in range(24):
        manifest = json.loads(
          (output_dir / f"case_{index:06d}" / "case.json").read_text(encoding="utf-8")
        )
        operators.update(manifest["operator_coverage"])
      self.assertIn("exp", operators)

  def test_v2_composition_includes_symmetric_reduction_shapes(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_stratum(
        output_dir,
        "multi-input-composition",
        seed=31,
        dtype="i32",
        count=64,
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      input_shapes = {
        tuple(
          json.loads(
            (output_dir / f"case_{index:06d}" / "case.json").read_text(encoding="utf-8")
          )["inputs"][0]["shape"]
        )
        for index in range(64)
      }
      self.assertTrue({(4, 4), (4, 4, 4)}.issubset(input_shapes), input_shapes)

  def test_v2_composition_has_shared_runtime_fanout(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_stratum(
        output_dir, "multi-input-composition", seed=37, dtype="i32"
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      source = (output_dir / "case_000000" / "program.sir").read_text(encoding="utf-8")
      self.assertGreaterEqual(source.count("%mixed"), 4, source)

  def test_v2_composition_counterfactually_depends_on_each_input(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2(output_dir, seed=2802362399)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case_dir = output_dir / "case_000000"
      source = (case_dir / "program.sir").read_text(encoding="utf-8")
      manifest = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
      witnesses = manifest["coverage"]["input_behavior"]["counterfactual_inputs"]
      self.assertEqual({item["input"] for item in witnesses}, {"a", "b"})
      self.assertTrue(all(item["outputs_differ"] for item in witnesses))
      original = bytearray((case_dir / "inputs.bin").read_bytes())
      baseline_result, baseline = run_typed(
        source, bytes(original), tensor_profile="v2"
      )
      self.assertEqual(
        baseline_result.returncode, 0, baseline_result.stdout + baseline_result.stderr
      )
      influenced_inputs = set()
      for input_index, buffer in enumerate(manifest["inputs"]):
        begin = buffer["byte_offset"]
        end = begin + buffer["byte_length"]
        for replacement in (1, 4):
          changed = bytearray(original)
          for offset in range(begin, end, 4):
            changed[offset : offset + 4] = struct.pack("<i", replacement)
          probe_result, output = run_typed(source, bytes(changed), tensor_profile="v2")
          self.assertEqual(
            probe_result.returncode, 0, probe_result.stdout + probe_result.stderr
          )
          if output != baseline:
            influenced_inputs.add(input_index)
            break

      self.assertEqual(influenced_inputs, set(range(len(manifest["inputs"]))), source)

  def test_v2_triton_generation_uses_each_case_input_layout(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate(
        output_dir,
        "-n",
        "1",
        "--tensor-dtype",
        "i32",
        "--tensor-profile-version",
        "v2",
        "--tensor-stratum",
        "loop-reduction",
        seed=1,
        target="triton",
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case_dir = output_dir / "case_000000"
      manifest = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
      kernel = (case_dir / "kernel.py").read_text(encoding="utf-8")

      row_count = manifest["inputs"][0]["shape"][0]
      self.assertEqual(manifest["inputs"][0]["element_strides"], [1, row_count])
      self.assertIn(f"_sir_x_offsets = _sir_x_d0 + _sir_x_d1 * {row_count}", kernel)

  def test_v2_loop_reduction_has_at_least_three_structural_signatures(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_stratum(
        output_dir, "loop-reduction", seed=7300, dtype="f32", count=40
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      signatures = set()
      for index in range(40):
        case_dir = output_dir / f"case_{index:06d}"
        source = (case_dir / "program.sir").read_text(encoding="utf-8")
        signatures.add(source_signature(source))

      self.assertGreaterEqual(len(signatures), 3)

  def test_v2_loop_reduction_places_reductions_inside_and_before_loops(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_stratum(
        output_dir, "loop-reduction", seed=8100, dtype="f32", count=40
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      placements = set()
      for index in range(40):
        source = (output_dir / f"case_{index:06d}" / "program.sir").read_text(
          encoding="utf-8"
        )
        header = re.search(r"\^(?:header|inner_header):", source)
        self.assertIsNotNone(header, source)
        reduction = source.index("tensor sum")
        placements.add("before" if reduction < header.start() else "inside")

      self.assertEqual(placements, {"before", "inside"})

  def test_v2_loop_reduction_composes_live_transpose_and_broadcast(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_stratum(
        output_dir, "loop-reduction", seed=9100, dtype="f32", count=40
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      candidate = None
      for index in range(40):
        case = load_case(output_dir / f"case_{index:06d}")
        source = case.program.decode("utf-8")
        if all(
          operation in source
          for operation in (
            "tensor transpose %x",
            "tensor sum",
            "tensor expand",
            "tensor broadcast",
            "tensor add %acc",
          )
        ):
          candidate = case
          break

      self.assertIsNotNone(candidate, "no loop candidate contains the composition")
      source = candidate.program.decode("utf-8")
      self.assertIn("ret %acc;", source)
      replay, baseline = run_typed(source, candidate.inputs, tensor_profile="v2")
      self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
      self.assertEqual(baseline, candidate.expected)
      replay, zeroed = run_typed(
        source, bytes(len(candidate.inputs)), tensor_profile="v2"
      )
      self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
      self.assertNotEqual(zeroed, baseline)

  def test_v2_loop_structure_and_warp_count_are_independently_sampled(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_stratum(
        output_dir, "loop-reduction", seed=10100, dtype="f32", count=80
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      warp_counts = {"single": set(), "nested": set()}
      for index in range(80):
        case_dir = output_dir / f"case_{index:06d}"
        source = (case_dir / "program.sir").read_text(encoding="utf-8")
        manifest = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        form = "nested" if "^outer_header:" in source else "single"
        warp_counts[form].add(manifest["launch"]["num_warps"])

      self.assertEqual(warp_counts["single"], {1, 2, 4, 8})
      self.assertEqual(warp_counts["nested"], {1, 2, 4, 8})

  def test_v2_default_schedule_reaches_single_loop_f32_at_every_warp_count(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate(
        output_dir,
        "-n",
        "100",
        "--tensor-dtype",
        "auto",
        "--tensor-profile-version",
        "v2",
        seed=20260926,
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      single_loop_warps = set()
      nested_loop_count = 0
      for index in range(100):
        case_dir = output_dir / f"case_{index:06d}"
        source = (case_dir / "program.sir").read_text(encoding="utf-8")
        manifest = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        if manifest["coverage"]["primary_stratum"] != "loop-reduction":
          continue
        self.assertEqual(manifest["inputs"][0]["dtype"], "f32")
        if "^outer_header:" in source:
          nested_loop_count += 1
        else:
          single_loop_warps.add(manifest["launch"]["num_warps"])

      self.assertGreater(nested_loop_count, 0)
      self.assertEqual(single_loop_warps, {1, 2, 4, 8})

  def test_v2_loop_reduction_varies_shapes_and_bounded_trip_counts(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_stratum(
        output_dir, "loop-reduction", seed=12100, dtype="f32", count=40
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      shapes = set()
      loop_bounds = set()
      for index in range(40):
        case = load_case(output_dir / f"case_{index:06d}")
        execution = case.manifest["execution"]
        shapes.add(tuple(case.manifest["inputs"][0]["shape"]))
        loop_bounds.add(tuple(execution["loop_bounds"]))
        self.assertGreaterEqual(execution["total_body_executions"], 1)
        self.assertLessEqual(execution["total_body_executions"], 8)
        self.assertLessEqual(max(execution["loop_bounds"]), 4)

      self.assertGreaterEqual(len(shapes), 2)
      self.assertGreaterEqual(len(loop_bounds), 2)
      self.assertIn((256, 2), shapes)

  def test_v2_loop_reduction_generates_two_runtime_inputs(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_loop(output_dir, seed=17001, dtype="i32", count=48)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = find_multi_input_loop_case(output_dir)
      self.assertIsNotNone(case, "no loop-reduction case has two runtime inputs")
      if case is None:
        return
      self.assertEqual({item["name"] for item in case.manifest["inputs"]}, {"x", "y"})

  def test_v2_multi_input_loop_has_two_tensor_carried_accumulators(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_loop(output_dir, seed=17002, dtype="i32", count=48)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = find_multi_input_loop_case(output_dir)
      self.assertIsNotNone(case, "no loop-reduction case has two runtime inputs")
      if case is None:
        return
      updates = loop_carried_tensor_updates(case.program.decode("utf-8"))
      self.assertGreaterEqual(len(updates), 2, updates)

  def test_v2_multi_input_loop_accumulators_have_a_cross_dependency(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_loop(output_dir, seed=17003, dtype="i32", count=48)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = find_multi_input_loop_case(output_dir)
      self.assertIsNotNone(case, "no loop-reduction case has two runtime inputs")
      if case is None:
        return
      updates = loop_carried_tensor_updates(case.program.decode("utf-8"))
      self.assertGreaterEqual(len(updates), 2, updates)
      cross_dependency = any(
        other != target
        and any(other in re.findall(r"%[A-Za-z_][\w.$-]*", rhs) for rhs in rhs_values)
        for target, rhs_values in updates.items()
        for other in updates
      )
      self.assertTrue(cross_dependency, updates)

  def test_v2_multi_input_loop_uses_distinct_physical_layouts(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_loop(output_dir, seed=17004, dtype="i32", count=48)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = find_multi_input_loop_case(output_dir)
      self.assertIsNotNone(case, "no loop-reduction case has two runtime inputs")
      if case is None:
        return
      layouts = {
        (tuple(item["element_strides"]), tuple(item["axis_order"]))
        for item in case.manifest["inputs"]
      }
      self.assertEqual(len(layouts), 2, layouts)
      self.assertEqual({axis_order for _, axis_order in layouts}, {(0, 1), (1, 0)})

  def test_v2_multi_input_loop_proves_each_input_changes_the_returned_output(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_loop(output_dir, seed=17005, dtype="i32", count=48)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = find_multi_input_loop_case(output_dir)
      self.assertIsNotNone(case, "no loop-reduction case has two runtime inputs")
      if case is None:
        return
      source = case.program.decode("utf-8")
      witnesses = case.manifest["coverage"]["input_behavior"]["counterfactual_inputs"]
      self.assertEqual(
        {item["input"] for item in witnesses},
        {item["name"] for item in case.manifest["inputs"]},
      )
      self.assertTrue(all(item["outputs_differ"] for item in witnesses), witnesses)

      rows, columns = case.manifest["inputs"][0]["shape"]
      logical_values = {}
      for item in case.manifest["inputs"]:
        begin = item["byte_offset"]
        end = begin + item["byte_length"]
        logical_values[item["name"]] = struct.unpack(
          f"<{rows * columns}i", case.inputs[begin:end]
        )
      row_sums_x = [
        sum(logical_values["x"][row * columns + column] for row in range(rows))
        for column in range(columns)
      ]
      row_sums_y = [
        sum(logical_values["y"][row * columns : (row + 1) * columns])
        for row in range(rows)
      ]
      acc_x = [0] * columns
      acc_y = [0] * rows
      loop_bounds = case.manifest["execution"]["loop_bounds"]
      self.assertEqual(len(loop_bounds), 1, loop_bounds)
      trip_count = loop_bounds[0]
      for _ in range(trip_count):
        acc_x = [left + right for left, right in zip(acc_x, row_sums_x)]
        acc_y = [left + right for left, right in zip(acc_y, row_sums_y)]
        acc_y = [left + right for left, right in zip(acc_y, acc_x)]
      independent = [left + right for left, right in zip(acc_x, acc_y)]
      self.assertEqual(
        list(struct.unpack(f"<{len(case.expected) // 4}i", case.expected)),
        independent,
      )

      baseline_run, baseline = run_typed(source, case.inputs, tensor_profile="v2")
      self.assertEqual(
        baseline_run.returncode, 0, baseline_run.stdout + baseline_run.stderr
      )
      self.assertEqual(baseline, case.expected)
      for item in case.manifest["inputs"]:
        changed = bytearray(case.inputs)
        start = item["byte_offset"]
        end = start + item["byte_length"]
        changed[start:end] = bytes(end - start)
        probe_run, output = run_typed(source, bytes(changed), tensor_profile="v2")
        self.assertEqual(probe_run.returncode, 0, probe_run.stdout + probe_run.stderr)
        self.assertNotEqual(output, baseline, item["name"])

  def test_v2_multi_input_loop_emits_each_physical_input_layout(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate(
        output_dir,
        "-n",
        "48",
        "--tensor-dtype",
        "i32",
        "--tensor-profile-version",
        "v2",
        "--tensor-stratum",
        "loop-reduction",
        seed=17006,
        target="triton",
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = find_multi_input_loop_case(output_dir)
      self.assertIsNotNone(case, "no Triton loop case has two runtime inputs")
      if case is None:
        return
      kernel = (case.path / "kernel.py").read_text(encoding="utf-8")
      x_shape = case.manifest["inputs"][0]["shape"]
      y_shape = case.manifest["inputs"][1]["shape"]
      self.assertEqual(x_shape, y_shape)
      rows, columns = x_shape
      self.assertEqual(case.manifest["inputs"][0]["element_strides"], [1, rows])
      self.assertEqual(case.manifest["inputs"][1]["element_strides"], [columns, 1])
      self.assertIn(f"_sir_x_offsets = _sir_x_d0 + _sir_x_d1 * {rows}", kernel)
      self.assertIn(f"_sir_y_offsets = _sir_y_d0 * {columns} + _sir_y_d1", kernel)

  def test_v2_dynamic_loop_reduces_both_carried_accumulators(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_loop(output_dir, seed=27001, dtype="i32", count=96)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = find_loop_carried_reduction_case(output_dir)
      self.assertIsNotNone(case, "no loop reduces its changing tensor state")
      if case is None:
        return
      source = case.program.decode("utf-8")
      self.assertIn("tensor sum %acc_x, 1, drop", source)
      self.assertIn("tensor sum %acc_y, 0, drop", source)
      self.assertEqual(len(case.manifest["inputs"]), 2)

  def test_v2_dynamic_loop_reductions_execute_inside_the_counted_body(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_loop(output_dir, seed=27002, dtype="i32", count=96)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = find_loop_carried_reduction_case(output_dir)
      self.assertIsNotNone(case, "no loop reduces its changing tensor state")
      if case is None:
        return
      source = case.program.decode("utf-8")
      body = re.search(r"(?ms)^\^body:\n(.*?)(?=^\^[A-Za-z_][A-Za-z0-9_]*:)", source)
      self.assertIsNotNone(body, source)
      if body is None:
        return
      body_source = body.group(1)
      for accumulator in ("%acc_x", "%acc_y"):
        self.assertRegex(body_source, rf"tensor sum {re.escape(accumulator)},")
      self.assertLess(
        body_source.index("tensor sum %acc_x"),
        body_source.index("%acc_x = tensor add"),
      )
      self.assertLess(
        body_source.index("tensor sum %acc_y"),
        body_source.index("%acc_y = tensor add"),
      )
      self.assertEqual(len(case.manifest["execution"]["loop_bounds"]), 1)

  def test_v2_dynamic_loop_accumulators_are_cross_dependent(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_loop(output_dir, seed=27003, dtype="i32", count=96)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = find_loop_carried_reduction_case(output_dir)
      self.assertIsNotNone(case, "no loop reduces its changing tensor state")
      if case is None:
        return
      source = case.program.decode("utf-8")
      header = source.index("^header:")
      self.assertIn("%acc_x = %x;", source[:header])
      self.assertIn("%acc_y = %y;", source[:header])
      self.assertIn("%acc_x = tensor add %acc_x, %broadcast_y;", source)
      self.assertIn("%acc_y = tensor add %acc_y, %broadcast_x;", source)

  def test_v2_dynamic_loop_has_exact_inputs_with_distinct_layouts_and_witnesses(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_loop(output_dir, seed=27004, dtype="i32", count=96)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = find_loop_carried_reduction_case(output_dir)
      self.assertIsNotNone(case, "no loop reduces its changing tensor state")
      if case is None:
        return
      self.assertEqual(case.manifest["comparison"]["mode"], "exact")
      layouts = {
        (tuple(item["element_strides"]), tuple(item["axis_order"]))
        for item in case.manifest["inputs"]
      }
      self.assertEqual(len(layouts), 2, layouts)
      self.assertEqual({axis_order for _, axis_order in layouts}, {(0, 1), (1, 0)})
      witnesses = case.manifest["coverage"]["input_behavior"]["counterfactual_inputs"]
      self.assertEqual(
        {item["input"] for item in witnesses},
        {item["name"] for item in case.manifest["inputs"]},
      )
      self.assertTrue(all(item["outputs_differ"] for item in witnesses), witnesses)

  def test_v2_dynamic_loop_reference_matches_independent_recurrence(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_loop(output_dir, seed=27005, dtype="i32", count=96)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = find_loop_carried_reduction_case(output_dir)
      self.assertIsNotNone(case, "no loop reduces its changing tensor state")
      if case is None:
        return
      rows, columns = case.manifest["inputs"][0]["shape"]
      input_values = {}
      for item in case.manifest["inputs"]:
        start = item["byte_offset"]
        end = start + item["byte_length"]
        input_values[item["name"]] = list(
          struct.unpack(f"<{rows * columns}i", case.inputs[start:end])
        )
      acc_x = input_values["x"]
      acc_y = input_values["y"]
      trip_count = case.manifest["execution"]["loop_bounds"][0]
      for _ in range(trip_count):
        row_sums_x = [
          sum(acc_x[row * columns : (row + 1) * columns]) for row in range(rows)
        ]
        column_sums_y = [
          sum(acc_y[row * columns + column] for row in range(rows))
          for column in range(columns)
        ]
        next_x = [
          acc_x[row * columns + column] + column_sums_y[column]
          for row in range(rows)
          for column in range(columns)
        ]
        next_y = [
          acc_y[row * columns + column] + row_sums_x[row]
          for row in range(rows)
          for column in range(columns)
        ]
        acc_x, acc_y = next_x, next_y
      expected = [left + right for left, right in zip(acc_x, acc_y)]
      actual = list(struct.unpack(f"<{rows * columns}i", case.expected))
      self.assertEqual(actual, expected)

      source = case.program.decode("utf-8")
      run, interpreted = run_typed(source, case.inputs, tensor_profile="v2")
      self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
      self.assertEqual(interpreted, case.expected)

  def test_v2_dynamic_loop_f32_is_numeric_and_reaches_every_warp_count(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v2_loop(output_dir, seed=27006, dtype="f32", count=192)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      dynamic_warps = set()
      for case_dir in sorted(Path(output_dir).glob("case_*")):
        case = load_case(case_dir)
        source = case.program.decode("utf-8")
        if "tensor sum %acc_x, 1, drop" not in source:
          continue
        dynamic_warps.add(case.manifest["launch"]["num_warps"])
        self.assertEqual(case.manifest["comparison"]["mode"], "numeric")
        self.assertTrue(case.manifest["reference"]["stable"])

      self.assertEqual(dynamic_warps, {1, 2, 4, 8})

  def test_v3_generator_records_logical_and_execution_shapes(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v3(output_dir, stratum="tail-arithmetic")

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      self.assertEqual(case.manifest["profile"], "tensor-v3")
      self.assertEqual(case.manifest["inputs"][0]["shape"], [17])
      self.assertEqual(case.manifest["inputs"][0]["tile_shape"], [32])
      self.assertEqual(case.manifest["output"]["shape"], [17])
      self.assertEqual(case.manifest["output"]["tile_shape"], [32])

  def test_v3_loop_reduction_reduces_the_loop_carried_tail_tensor(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v3(
        output_dir,
        stratum="tail-loop",
        pattern="coordinate-distinct",
        seed=20260928,
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      source = case.program.decode("utf-8")
      loop_body = source.split("^body:", maxsplit=1)[1].split("^exit:", maxsplit=1)[0]
      self.assertIn("tensor sum %acc, 0, drop", loop_body)

  def test_v3_loop_reduction_has_a_live_loop_carried_dependency(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v3(output_dir, stratum="tail-loop", pattern="mixed-sign")

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      source = case.program.decode("utf-8")
      self.assertIn("%acc = tensor add %acc, %x", source)
      self.assertIn("tensor sum %acc, 0, drop", source)
      self.assertIn("sum", case.manifest["operator_coverage"])

  def test_v3_one_dimensional_loop_reduction_records_scalar_output_and_axis(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v3(output_dir, stratum="tail-loop", pattern="repeated-value")

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      self.assertEqual(case.manifest["inputs"][0]["shape"], [17])
      self.assertEqual(case.manifest["output"]["shape"], [])
      self.assertEqual(case.manifest["shape_axes"], [{"shape": [17], "axis": 0}])

  def test_v3_one_dimensional_loop_reduction_reference_matches_trip_count(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v3(
        output_dir,
        stratum="tail-loop",
        pattern="coordinate-distinct",
        seed=20260928,
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      input_values = struct.unpack(f"<{len(case.inputs) // 4}i", case.inputs)
      expected = struct.unpack("<i", case.expected)[0]
      self.assertEqual(expected, 6 * sum(input_values))

  def test_v3_f32_loop_reduction_uses_stable_numerical_comparison(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v3(
        output_dir,
        stratum="tail-loop",
        pattern="coordinate-distinct",
        dtype="f32",
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      self.assertEqual(case.manifest["output"]["shape"], [])
      self.assertEqual(case.manifest["comparison"]["mode"], "numeric")
      self.assertTrue(case.manifest["reference"]["stable"])

  def test_v3_loop_reduction_covers_both_axes_of_each_2d_boundary_placement(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v3(
        output_dir,
        stratum="tail-loop",
        pattern="distinct-groups",
        seed=2026092802,
        count=25,
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      axes_by_shape = {}
      for case_dir in sorted(output_dir.glob("case_*")):
        case = load_case(case_dir)
        input_shape = case.manifest["inputs"][0]["shape"]
        if len(input_shape) != 2:
          continue
        axis_record = case.manifest["shape_axes"][0]
        axis = axis_record["axis"]
        expected_shape = input_shape[:axis] + input_shape[axis + 1 :]
        self.assertEqual(case.manifest["output"]["shape"], expected_shape)
        self.assertIn("tensor sum %acc", case.program.decode("utf-8"))
        axes_by_shape.setdefault(tuple(input_shape), set()).add(axis)

      expected_shapes = {
        (17, 16),
        (16, 17),
        (31, 16),
        (16, 31),
        (33, 8),
        (8, 33),
        (63, 8),
        (8, 63),
        (65, 4),
        (4, 65),
      }
      self.assertEqual(set(axes_by_shape), expected_shapes)
      self.assertTrue(all(axes == {0, 1} for axes in axes_by_shape.values()))

  def test_v3_generator_covers_required_boundary_shapes_within_tile_limit(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v3(output_dir, stratum="tail-sum", count=25)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      shapes = set()
      shape_axes = {}
      for case_dir in sorted(output_dir.glob("case_*")):
        case = load_case(case_dir)
        record = case.manifest["inputs"][0]
        shape = tuple(record["shape"])
        tile = record["tile_shape"]
        volume = 1
        for extent in tile:
          volume *= extent
        self.assertLessEqual(volume, 512)
        shapes.add(shape)
        if len(shape) == 2:
          shape_axes.setdefault(shape, set()).add(
            case.manifest["shape_axes"][0]["axis"]
          )

      expected = {
        (17,),
        (31,),
        (33,),
        (63,),
        (65,),
        (17, 16),
        (16, 17),
        (31, 16),
        (16, 31),
        (33, 8),
        (8, 33),
        (63, 8),
        (8, 63),
        (65, 4),
        (4, 65),
      }
      self.assertEqual(shapes, expected)
      self.assertEqual(
        set(shape_axes), {shape for shape in expected if len(shape) == 2}
      )
      self.assertTrue(all(axes == {0, 1} for axes in shape_axes.values()), shape_axes)

  def test_v3_generator_records_each_input_pattern_in_both_ledgers(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v3(output_dir, count=6)

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      patterns = set()
      for case_dir in sorted(output_dir.glob("case_*")):
        case = load_case(case_dir)
        pattern = case.manifest["generator"]["input_pattern"]
        self.assertEqual(
          case.manifest["coverage"]["input_behavior"]["pattern"], pattern
        )
        patterns.add(pattern)

      self.assertEqual(
        patterns,
        {
          "coordinate-distinct",
          "mixed-sign",
          "cancellation",
          "repeated-value",
          "distinct-groups",
          "seeded-random",
        },
      )

  def _assert_v3_f32_arithmetic_generates(self, pattern):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v3(
        output_dir,
        stratum="tail-arithmetic",
        pattern=pattern,
        dtype="f32",
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      source = (case.path / "program.sir").read_text(encoding="utf-8")
      self.assertIn("tensor mul %x, 2.0", source)

  def test_v3_f32_arithmetic_types_coordinate_distinct_literal(self):
    self._assert_v3_f32_arithmetic_generates("coordinate-distinct")

  def test_v3_f32_arithmetic_types_mixed_sign_literal(self):
    self._assert_v3_f32_arithmetic_generates("mixed-sign")

  def test_v3_f32_arithmetic_types_cancellation_literal(self):
    self._assert_v3_f32_arithmetic_generates("cancellation")

  def test_v3_f32_arithmetic_types_repeated_value_literal(self):
    self._assert_v3_f32_arithmetic_generates("repeated-value")

  def test_v3_f32_arithmetic_types_distinct_group_literal(self):
    self._assert_v3_f32_arithmetic_generates("distinct-groups")

  def test_v3_f32_untargeted_composition_types_arithmetic_literal(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v3(
        output_dir,
        stratum="tail-untargeted",
        pattern="seeded-random",
        dtype="f32",
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      source = (case.path / "program.sir").read_text(encoding="utf-8")
      self.assertIn("tensor mul %x, 2.0", source)
      self.assertIn("tensor add %tmp, %x", source)

  def test_v3_exact_f32_sum_has_complete_output_proof(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v3(
        output_dir, stratum="tail-sum", pattern="cancellation", dtype="f32"
      )

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      self.assertEqual(case.manifest["comparison"]["mode"], "exact")
      proof = case.manifest["reference"]["exactness"]
      self.assertTrue(proof["complete_output"])
      self.assertEqual(proof["signed_zero"], "positive-zero-only")
      self.assertLessEqual(proof["max_abs_leaf_sum"], 1 << 24)

  def test_v3_triton_generation_uses_masked_tail_and_recorded_layouts(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory) / "cases"
      result = generate_v3(output_dir, stratum="tail-arithmetic", target="triton")

      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      case = load_case(output_dir / "case_000000")
      kernel = (case.path / "kernel.py").read_text(encoding="utf-8")
      self.assertIn("mask=", kernel)
      self.assertIn("tl.store(out_ptr +", kernel)
      input_record = case.manifest["inputs"][0]
      output_record = case.manifest["output"]
      self.assertGreater(input_record["element_strides"][0], 1)
      self.assertIn(f"* {input_record['element_strides'][0]}", kernel)
      self.assertIn(f"* {output_record['element_strides'][0]}", kernel)


class TensorGraphGenerationTests(unittest.TestCase):
  """Structural contracts for the opt-in compositional tensor generator."""

  def setUp(self):
    self.assertTrue(RYSMITH.is_file(), f"rysmith executable not found: {RYSMITH}")

  def generate_cases(self, directory, *, count, seed, profile="v2", dtype="i32"):
    output_dir = Path(directory) / "cases"
    result = generate_compositional(
      output_dir,
      profile=profile,
      dtype=dtype,
      seed=seed,
      count=count,
      stratum="multi-input-composition" if profile == "v2" else None,
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return output_dir

  def test_compositional_seed_reproduces_source_and_manifest(self):
    with tempfile.TemporaryDirectory() as directory:
      first = Path(directory) / "first"
      second = Path(directory) / "second"
      different = Path(directory) / "different"
      for output in (first, second, different):
        seed = 17 if output is not different else 18
        result = generate_compositional(
          output, count=2, seed=seed, stratum="multi-input-composition"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

      self.assertEqual(read_summary(first), read_summary(second))
      for index in range(2):
        case_name = f"case_{index:06d}"
        for artifact in ("program.sir", "inputs.bin", "expected.bin", "case.json"):
          self.assertEqual(
            (first / case_name / artifact).read_bytes(),
            (second / case_name / artifact).read_bytes(),
            f"{case_name}/{artifact}",
          )
        self.assertNotEqual(
          (first / case_name / "program.sir").read_bytes(),
          (different / case_name / "program.sir").read_bytes(),
          case_name,
        )

  def test_compositional_source_checks_and_manifest_agrees_with_it(self):
    with tempfile.TemporaryDirectory() as directory:
      for profile in ("v2", "v3"):
        with self.subTest(profile=profile):
          output_dir = self.generate_cases(
            Path(directory) / profile, count=4, seed=23, profile=profile
          )

          for case_dir in sorted(output_dir.glob("case_*")):
            case = load_case(case_dir)
            source = case.program.decode("utf-8")
            check = check_source(source, tensor_profile=profile)
            self.assertEqual(check.returncode, 0, check.stdout + check.stderr)
            manifest = case.manifest
            self.assertEqual(manifest["profile"], f"tensor-{profile}")
            tensor_inputs = [item for item in manifest["inputs"] if item["shape"]]
            self.assertGreaterEqual(len(tensor_inputs), 2)
            self.assertLessEqual(len(manifest["inputs"]), 3)
            atoms = set(re.findall(r"tensor ([a-z]+) ", source))
            copied = (
              {"copy"}
              if re.search(r"(?m)^\s*%[A-Za-z_][\w.$-]* = %[A-Za-z_][\w.$-]*;$", source)
              else set()
            )
            self.assertEqual(set(manifest["operator_coverage"]), atoms | copied)
            self.assertEqual(
              manifest["shape_axes"],
              [
                {"shape": shape, "axis": axis}
                for _, shape, axis, _ in sum_records(source)
              ],
            )
            replay, actual = run_typed(source, case.inputs, tensor_profile=profile)
            self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
            self.assertEqual(actual, case.expected)

  def test_compositional_generates_a_shared_intermediate_diamond(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = self.generate_cases(directory, count=6, seed=91)

      diamonds = []
      for case_dir in sorted(output_dir.glob("case_*")):
        source = (case_dir / "program.sir").read_text(encoding="utf-8")
        statements = entry_statements(source)
        parameters = set(re.findall(r"%[A-Za-z_][\w.$-]*", source.split(") :", 1)[0]))
        consumers = local_consumers(statements)
        for name, sites in consumers.items():
          if name not in parameters and len(set(sites)) >= 2:
            diamonds.append((case_dir.name, name))

      self.assertTrue(diamonds, "no generated intermediate had two live consumers")

  def test_compositional_generates_two_reductions_with_different_consumers(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = self.generate_cases(directory, count=6, seed=91)

      observed = []
      for case_dir in sorted(output_dir.glob("case_*")):
        source = (case_dir / "program.sir").read_text(encoding="utf-8")
        statements = entry_statements(source)
        consumers = local_consumers(statements)
        reductions = [
          (match.group(1), sorted(set(consumers.get(match.group(1), []))))
          for match in (
            re.match(r"(%[A-Za-z_][\w.$-]*)\s*=\s*tensor sum\b", statement)
            for statement in statements
          )
          if match is not None
        ]
        if len(reductions) >= 2 and all(sites for _, sites in reductions):
          if len({sites[0] for _, sites in reductions}) >= 2:
            observed.append(case_dir.name)

      self.assertTrue(
        observed, "no case had two reductions with distinct live consumers"
      )

  def test_compositional_can_place_a_shape_change_before_a_reduction(self):
    """A layout change upstream of a reduction is a distinct structure.

    Appending every shape operator downstream of every mandatory feature would
    leave the generator unable to reach a graph whose reduction consumes a
    transposed, expanded, or broadcast operand.
    """
    observed = []
    for seed in range(1, 61):
      with tempfile.TemporaryDirectory() as directory:
        output_dir = self.generate_cases(directory, count=4, seed=seed)
        for case_dir in sorted(output_dir.glob("case_*")):
          source = (case_dir / "program.sir").read_text(encoding="utf-8")
          statements = entry_statements(source)
          producers = {}
          for index, statement in enumerate(statements):
            match = re.match(r"(%[A-Za-z_][\w.$-]*)\s*=", statement)
            if match is not None:
              producers[match.group(1)] = index
          if self._reduce_consumes_a_shape_change(statements, producers):
            observed.append((seed, case_dir.name))
            break
      if observed:
        break

    self.assertTrue(observed, "no generated reduction consumed a shape-changed operand")

  @staticmethod
  def _reduce_consumes_a_shape_change(statements, producers):
    """Whether some tensor sum reads a value a transpose/broadcast/expand fed."""

    def ancestors(index):
      seen = set()
      pending = [index]
      while pending:
        current = pending.pop()
        for name in referenced_locals(statements[current]):
          producer = producers.get(name)
          if producer is None or producer in seen:
            continue
          seen.add(producer)
          pending.append(producer)
      return seen

    shape = re.compile(
      r"%[A-Za-z_][\w.$-]*\s*=\s*tensor (transpose|broadcast|expand)\b"
    )
    for index, statement in enumerate(statements):
      if not re.match(r"%[A-Za-z_][\w.$-]*\s*=\s*tensor sum\b", statement):
        continue
      if any(shape.match(statements[upstream]) for upstream in ancestors(index)):
        return True
    return False

  def test_compositional_keeps_every_computation_in_the_backward_slice(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = self.generate_cases(directory, count=4, seed=29)

      for case_dir in sorted(output_dir.glob("case_*")):
        source = (case_dir / "program.sir").read_text(encoding="utf-8")
        statements = entry_statements(source)
        parameters = set(re.findall(r"%[A-Za-z_][\w.$-]*", source.split(") :", 1)[0]))
        returned = re.search(r"ret (%[A-Za-z_][\w.$-]*);", source)
        self.assertIsNotNone(returned, source)
        producers = {}
        for statement in statements:
          match = re.match(r"(%[A-Za-z_][\w.$-]*)\s*=", statement)
          if match is not None:
            producers.setdefault(match.group(1), statement)
        live = set()
        pending = [returned.group(1)]
        while pending:
          name = pending.pop()
          if name in live:
            continue
          live.add(name)
          producer = producers.get(name)
          if producer is not None:
            pending.extend(referenced_locals(producer) - {name})
        self.assertEqual(set(producers) - live, set(), case_dir.name)
        self.assertEqual(
          set(re.findall(r"let mut (%[A-Za-z_][\w.$-]*)", source)),
          set(producers),
          case_dir.name,
        )
        self.assertTrue(parameters, case_dir.name)
        self.assertEqual(parameters - live, set(), case_dir.name)

  def test_compositional_expands_a_scalar_reduction_back_to_a_tensor(self):
    with tempfile.TemporaryDirectory() as directory:
      output_dir = self.generate_cases(directory, count=8, seed=5)

      observed = []
      for case_dir in sorted(output_dir.glob("case_*")):
        source = (case_dir / "program.sir").read_text(encoding="utf-8")
        shapes = tensor_local_shapes(source)
        for match in re.finditer(
          r"(%[A-Za-z_][\w.$-]*) = tensor sum (%[A-Za-z_][\w.$-]*), (-?\d+), drop;",
          source,
        ):
          result_name, operand = match.group(1), match.group(2)
          if len(shapes[operand]) != 1:
            continue
          if re.search(rf"tensor expand {re.escape(result_name)}, ", source):
            observed.append(case_dir.name)

      self.assertTrue(
        observed, "no rank-one drop reduction was expanded back to a tensor"
      )


def case_output_target(case):
  """The value the exact policy's stated property names for this case."""
  record = case.manifest["output"]
  if record["dtype"] != "i32":
    raise AssertionError("the policy tests drive integer outputs only")
  values = struct.unpack(f"<{len(case.expected) // 4}i", case.expected)
  return values[0] if not record["shape"] else sum(values)


def load_compositional_case(test, output_dir, **kwargs):
  """Generate exactly one compositional case and load it through the validator."""
  result = generate_compositional(output_dir, **kwargs)
  test.assertEqual(result.returncode, 0, result.stdout + result.stderr)
  directories = sorted(Path(output_dir).glob("case_*"))
  test.assertEqual(len(directories), 1, result.stdout + result.stderr)
  return load_case(directories[0])


class TensorInputPolicyTests(unittest.TestCase):
  """Input selection is a separate step over one frozen program."""

  def compositional_case(
    self, output_dir, *, profile="v2", policy="ordinary", property_target=None, seed=3
  ):
    return load_compositional_case(
      self,
      output_dir,
      profile=profile,
      policy=policy,
      property_target=property_target,
      seed=seed,
    )

  def replay(self, case, *, profile="v2"):
    result, output = run_typed(
      case.program.decode("utf-8"), case.inputs, tensor_profile=profile
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return output

  def test_ordinary_and_boundary_policies_share_one_frozen_program(self):
    with tempfile.TemporaryDirectory() as ordinary_dir:
      with tempfile.TemporaryDirectory() as boundary_dir:
        ordinary = self.compositional_case(ordinary_dir, seed=11, policy="ordinary")
        boundary = self.compositional_case(boundary_dir, seed=11, policy="boundary")

    self.assertEqual(ordinary.program, boundary.program)
    self.assertNotEqual(ordinary.inputs, boundary.inputs)
    self.assertEqual(self.replay(ordinary), ordinary.expected)
    self.assertEqual(self.replay(boundary), boundary.expected)

  def test_exact_policy_selects_a_witness_for_the_ordinary_target(self):
    with tempfile.TemporaryDirectory() as ordinary_dir:
      ordinary = self.compositional_case(ordinary_dir, seed=17, policy="ordinary")
      target = case_output_target(ordinary)
      with tempfile.TemporaryDirectory() as exact_dir:
        exact = self.compositional_case(
          exact_dir, seed=17, policy="exact", property_target=target
        )

    # The property query added constraints and nothing else.
    self.assertEqual(exact.program, ordinary.program)
    replay = self.replay(exact)
    self.assertEqual(replay, exact.expected)
    self.assertEqual(case_output_target(exact), target)

  def test_infeasible_target_is_recorded_and_no_case_is_emitted(self):
    with tempfile.TemporaryDirectory() as directory:
      result = generate_compositional(
        directory, seed=17, policy="exact", property_target=2_000_000_000
      )
      summary = read_summary(directory)
      cases = list(Path(directory).glob("case_*"))

    self.assertNotEqual(result.returncode, 0)
    self.assertEqual(cases, [])
    self.assertEqual(summary["status"], "input_policy_unsat")
    self.assertIn("input_policy_unsat", summary["reason"])

  def test_boundary_inputs_stay_inside_the_integer_definedness_contract(self):
    allowed = {0, 1, -1, 2}
    for seed in (3, 4, 5):
      with self.subTest(seed=seed), tempfile.TemporaryDirectory() as directory:
        case = self.compositional_case(directory, seed=seed, policy="boundary")

        values = struct.unpack(f"<{len(case.inputs) // 4}i", case.inputs)
        self.assertLessEqual(set(values), allowed)
        self.assertEqual(self.replay(case), case.expected)

  def test_v3_exact_witness_carries_logical_elements_only(self):
    with tempfile.TemporaryDirectory() as ordinary_dir:
      ordinary = self.compositional_case(
        ordinary_dir, seed=31, profile="v3", policy="ordinary"
      )
      target = case_output_target(ordinary)
      with tempfile.TemporaryDirectory() as exact_dir:
        exact = self.compositional_case(
          exact_dir, seed=31, profile="v3", policy="exact", property_target=target
        )

    self.assertEqual(exact.program, ordinary.program)
    self.assertTrue(exact.manifest["inputs"])
    for record in exact.manifest["inputs"]:
      logical = 1
      for dimension in record["shape"]:
        logical *= dimension
      self.assertEqual(record["byte_length"], 4 * logical)
    self.assertEqual(self.replay(exact, profile="v3"), exact.expected)

  def test_exact_policy_requires_a_target_before_selection(self):
    with tempfile.TemporaryDirectory() as directory:
      result = generate_compositional(directory, seed=5, policy="exact")
      summary = read_summary(directory)

    self.assertNotEqual(result.returncode, 0)
    self.assertEqual(summary["status"], "unsupported_route")
    self.assertIn("tensor-property-target", summary["reason"])


class TensorSelectionProvenanceTests(unittest.TestCase):
  """A saved case carries how its inputs were selected and what they prove."""

  def exact_case(self, seed=17, profile="v2"):
    """One exact case and the target it was asked to satisfy."""
    with tempfile.TemporaryDirectory() as ordinary_dir:
      ordinary = load_compositional_case(
        self, ordinary_dir, seed=seed, profile=profile, policy="ordinary"
      )
      target = case_output_target(ordinary)
    directory = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, directory, True)
    case = load_compositional_case(
      self,
      directory,
      seed=seed,
      profile=profile,
      policy="exact",
      property_target=target,
    )
    return case, target

  def test_exact_case_records_its_policy_target_and_domain(self):
    case, target = self.exact_case()

    generator = case.manifest["generator"]
    self.assertEqual(generator["input_policy"], "exact")
    self.assertEqual(generator["property_target"], target)
    self.assertEqual(generator["input_domain"], {"lo": -1, "hi": 2})
    behavior = case.manifest["coverage"]["input_behavior"]
    self.assertEqual(behavior["policy"], "exact")
    self.assertEqual(behavior["property_target"], target)
    self.assertEqual(behavior["input_domain"], {"lo": -1, "hi": 2})

  def test_exact_selection_is_recorded_as_smt_evidence(self):
    case, target = self.exact_case()

    # The stated target was solved for exactly and then checked against the
    # replay, which is what the evidence kind has to say.
    self.assertEqual(case.manifest["evidence"]["kind"], "exact_smt")
    self.assertEqual(case.manifest["evidence"]["status"], "exact_smt_checked")
    self.assertEqual(case.manifest["property"]["target"], target)
    self.assertEqual(case.manifest["property"]["D"], "output-target")

  def test_ordinary_case_records_its_policy_without_a_target(self):
    with tempfile.TemporaryDirectory() as directory:
      case = load_compositional_case(self, directory, seed=17, policy="ordinary")

    generator = case.manifest["generator"]
    self.assertEqual(generator["input_policy"], "ordinary")
    self.assertNotIn("property_target", generator)
    self.assertEqual(generator["input_domain"], {"lo": 1, "hi": 2})
    self.assertEqual(case.manifest["coverage"]["input_behavior"]["policy"], "ordinary")
    self.assertNotIn("target", case.manifest["property"])
    self.assertEqual(case.manifest["evidence"]["kind"], "construction")

  def tampered_case(self, mutate):
    """An exact case with `mutate` applied to its manifest, digests untouched.

    A case states its target in three places, so every copy has to agree with
    the bytes. Mutating the manifest alone leaves every artifact digest valid
    and isolates the consistency check under test.
    """
    case, _ = self.exact_case()
    manifest_path = case.path / "case.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return case

  def test_tampered_exact_case_is_rejected_by_the_validator(self):
    """A deliberately incorrect witness must not load as an accepted case."""

    def raise_recorded_target(manifest):
      behavior = manifest["coverage"]["input_behavior"]
      behavior["property_target"] = behavior["property_target"] + 1

    case = self.tampered_case(raise_recorded_target)

    with self.assertRaises(CaseValidationError):
      load_case(case.path)

  def test_conflicting_generator_target_is_rejected(self):
    """generator.property_target must agree with the record the loader checks."""
    case = self.tampered_case(
      lambda manifest: manifest["generator"].__setitem__(
        "property_target", manifest["generator"]["property_target"] + 1
      )
    )

    with self.assertRaises(CaseValidationError):
      load_case(case.path)

  def test_conflicting_property_objective_is_rejected(self):
    """property.target must agree with the record the loader checks."""
    case = self.tampered_case(
      lambda manifest: manifest["property"].__setitem__(
        "target", manifest["property"]["target"] + 1
      )
    )

    with self.assertRaises(CaseValidationError):
      load_case(case.path)

  def test_f32_inputs_are_checked_against_the_recorded_domain(self):
    """The recorded domain bounds floating inputs as well as integer ones."""
    directory = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, directory, True)
    case = load_compositional_case(
      self, directory, seed=13, dtype="f32", policy="boundary"
    )
    values = struct.unpack(f"<{len(case.inputs) // 4}f", case.inputs)
    self.assertTrue(any(value != 0.0 for value in values), values)
    manifest_path = case.path / "case.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Both copies move together, so only the value check can reject this.
    manifest["coverage"]["input_behavior"]["input_domain"] = {"lo": 0, "hi": 0}
    manifest["generator"]["input_domain"] = {"lo": 0, "hi": 0}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with self.assertRaises(CaseValidationError):
      load_case(case.path)

  def exact_sum_case(self, seeds=range(1, 41)):
    """A rank-one case whose exact sum target the solver decides.

    The sum route expresses a scalar or rank-one return only, and a seed's
    shape is the builder's choice, so the sweep takes the first seed that both
    fits the route and decides inside the default budget.
    """
    outcomes = []
    for seed in seeds:
      boundary_dir = tempfile.mkdtemp()
      self.addCleanup(shutil.rmtree, boundary_dir, True)
      exact_dir = tempfile.mkdtemp()
      self.addCleanup(shutil.rmtree, exact_dir, True)
      boundary = load_compositional_case(
        self, boundary_dir, seed=seed, policy="boundary"
      )
      if len(boundary.manifest["output"]["shape"]) != 1:
        continue
      target = case_output_target(boundary)
      result = generate_compositional(
        exact_dir, seed=seed, policy="exact", property_target=target
      )
      if result.returncode == 0:
        return (
          boundary,
          load_case(sorted(Path(exact_dir).glob("case_*"))[0]),
          target,
        )
      admit_outcome(self, read_summary(exact_dir)["status"], seed, outcomes)
    self.fail(f"no exact sum target decided within the sweep: {sorted(set(outcomes))}")

  def test_pinned_exact_sum_seed_keeps_its_regression(self):
    """A deterministic fixture beside the sweep: seed 2 decides the sum route.

    The sweep steps over an undecided solver result, so on its own it could
    absorb a regression by finding a later seed. This pin fails if the route
    stops working, and the builder's seed-to-graph mapping changes only
    deliberately.
    """
    boundary, exact, target = self.exact_sum_case(seeds=range(2, 3))

    self.assertEqual(
      exact.manifest["coverage"]["input_behavior"]["property_target"], target
    )

  def test_boundary_feasible_target_is_exact_feasible(self):
    """The exact domain contains every value the sampling policies draw."""
    boundary, exact, target = self.exact_sum_case()

    self.assertEqual(
      exact.manifest["coverage"]["input_behavior"]["property_target"], target
    )
    self.assertEqual(case_output_target(exact), target)

  def test_compositional_v3_records_the_sampler_pattern(self):
    with tempfile.TemporaryDirectory() as ordinary_dir:
      ordinary = load_compositional_case(
        self, ordinary_dir, seed=31, profile="v3", policy="ordinary"
      )
      target = case_output_target(ordinary)
    with tempfile.TemporaryDirectory() as exact_dir:
      exact = load_compositional_case(
        self, exact_dir, seed=31, profile="v3", policy="exact", property_target=target
      )

    # The pattern names the sampler that ran, not a scheduled value family the
    # sampler never implemented.
    self.assertEqual(
      ordinary.manifest["coverage"]["input_behavior"]["pattern"], "seeded-random"
    )
    self.assertEqual(
      exact.manifest["coverage"]["input_behavior"]["pattern"], "smt-selected"
    )
    self.assertEqual(
      exact.manifest["generator"]["input_pattern"],
      exact.manifest["coverage"]["input_behavior"]["pattern"],
    )

  def test_compositional_generation_rejects_an_explicit_input_pattern(self):
    with tempfile.TemporaryDirectory() as directory:
      result = generate_compositional(
        directory, seed=7, profile="v3", pattern="mixed-sign"
      )
      summary = read_summary(directory)

    self.assertNotEqual(result.returncode, 0)
    self.assertEqual(summary["status"], "unsupported_route")
    self.assertIn("input pattern", summary["reason"])
    self.assertEqual(list(Path(directory).glob("case_*")), [])


class TensorElementPropertyTests(unittest.TestCase):
  """Per-element output targets, solved exactly and checked after replay."""

  def sample_case(self, *, minimum_rank=1, distinct=False, seeds=range(1, 61)):
    """A compositional i32 case whose output shape suits the test."""
    for seed in seeds:
      directory = tempfile.mkdtemp()
      self.addCleanup(shutil.rmtree, directory, True)
      case = load_compositional_case(self, directory, seed=seed, policy="ordinary")
      shape = case.manifest["output"]["shape"]
      if len(shape) < minimum_rank:
        continue
      if distinct and len(set(case_output_values(case))) < 2:
        continue
      return seed, case
    self.fail(
      f"no compositional case in the first {len(list(seeds))} seeds had rank "
      f"{minimum_rank}" + (" and two distinct output values" if distinct else "")
    )

  def complete_element_case(
    self, *, distinct=False, minimum_rank=2, seeds=range(1, 41)
  ):
    """Solve a complete element target on the first seed whose query decides.

    A complete target pins every output coordinate, which the solver usually
    decides at once, but a few graphs still reach the five-second budget; the
    sweep skips those rather than turning a timeout into a case.
    """
    outcomes = []
    for seed in seeds:
      ordinary_dir = tempfile.mkdtemp()
      self.addCleanup(shutil.rmtree, ordinary_dir, True)
      exact_dir = tempfile.mkdtemp()
      self.addCleanup(shutil.rmtree, exact_dir, True)
      ordinary = load_compositional_case(
        self, ordinary_dir, seed=seed, policy="ordinary"
      )
      shape = ordinary.manifest["output"]["shape"]
      if len(shape) < minimum_rank:
        continue
      values = case_output_values(ordinary)
      if distinct and len(set(values)) < 2:
        continue
      elements = [
        (row_major_coordinates(shape, offset), value)
        for offset, value in enumerate(values)
      ]
      result = generate_compositional(
        exact_dir, seed=seed, policy="exact", elements=elements
      )
      if result.returncode == 0:
        return (
          seed,
          ordinary,
          load_case(sorted(Path(exact_dir).glob("case_*"))[0]),
          elements,
        )
      admit_outcome(self, read_summary(exact_dir)["status"], seed, outcomes)
    self.fail(
      f"no complete element target decided within the sweep: {sorted(set(outcomes))}"
    )

  def exact_element_case(self, directory, seed, elements, profile="v2"):
    """Solve one element target on the graph the same seed builds."""
    return load_compositional_case(
      self,
      directory,
      seed=seed,
      profile=profile,
      policy="exact",
      elements=elements,
    )

  def replay(self, case, *, profile="v2"):
    result, output = run_typed(
      case.program.decode("utf-8"), case.inputs, tensor_profile=profile
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return output

  def test_v2_complete_element_target_is_solved_and_validated(self):
    _, ordinary, case, _ = self.complete_element_case()
    values = case_output_values(ordinary)

    self.assertEqual(case.program, ordinary.program)
    constraint = case.manifest["property"]["constraint"]
    self.assertEqual(constraint["kind"], "elements_equal")
    self.assertTrue(constraint["complete"])
    self.assertEqual(len(constraint["elements"]), len(values))
    self.assertEqual(self.replay(case), case.expected)

  def partial_element_case(self, *, indices=None, seeds=range(1, 41)):
    """Solve a partial element target on the first seed whose query decides.

    A partial target leaves the solver a wide space, and Bitwuzla reaches its
    five-second budget on some of these graphs. The budget is the documented
    default, so the sweep picks a graph the solver can decide and the recorded
    outcome of the others stays UNKNOWN rather than becoming an accepted case.
    """
    outcomes = []
    for seed in seeds:
      directory = tempfile.mkdtemp()
      self.addCleanup(shutil.rmtree, directory, True)
      ordinary_directory = tempfile.mkdtemp()
      self.addCleanup(shutil.rmtree, ordinary_directory, True)
      ordinary = load_compositional_case(
        self, ordinary_directory, seed=seed, policy="ordinary"
      )
      shape = ordinary.manifest["output"]["shape"]
      if len(shape) < 2:
        continue
      values = case_output_values(ordinary)
      wanted = indices if indices is not None else [0, len(values) - 1]
      elements = [
        (row_major_coordinates(shape, offset), values[offset]) for offset in wanted
      ]
      result = generate_compositional(
        directory, seed=seed, policy="exact", elements=elements
      )
      if result.returncode == 0:
        return seed, load_case(sorted(Path(directory).glob("case_*"))[0]), elements
      admit_outcome(self, read_summary(directory)["status"], seed, outcomes)
    self.fail(
      f"no partial element target decided within the sweep: {sorted(set(outcomes))}"
    )

  def test_partial_element_target_keeps_unconstrained_output_from_the_reference(self):
    seed, case, elements = self.partial_element_case()

    constraint = case.manifest["property"]["constraint"]
    self.assertFalse(constraint["complete"])
    self.assertEqual(
      [record["index"] for record in constraint["elements"]],
      [list(index) for index, _ in elements],
    )
    # The unconstrained coordinates are the reference's own output, not the
    # request: expected.bin is an independent replay of the witnesses.
    self.assertEqual(self.replay(case), case.expected)
    shape = case.manifest["output"]["shape"]
    replayed = case_output_values(case)
    for index, value in elements:
      offset = 0
      for axis, coordinate in enumerate(index):
        offset = offset * shape[axis] + coordinate
      self.assertEqual(replayed[offset], value)

  def test_equal_totals_with_different_elements_are_distinguished(self):
    """The same sum is not the same output: elements are compared, not totals."""
    _, ordinary, case, elements = self.complete_element_case(distinct=True)
    values = case_output_values(ordinary)
    first, second = (
      0,
      next(index for index, value in enumerate(values) if value != values[0]),
    )

    # Keep the coordinates and exchange the required values: the total is
    # unchanged, and each named element now requires the other one's value.
    swapped = [(index, value) for index, value in elements]
    swapped[first] = (elements[first][0], elements[second][1])
    swapped[second] = (elements[second][0], elements[first][1])
    self.assertEqual(sum(values), sum(value for _, value in swapped))
    self.assertNotEqual(
      [value for _, value in swapped], [value for _, value in elements]
    )

    manifest_path = case.path / "case.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["property"]["constraint"]["elements"] = [
      {"index": list(index), "value": value} for index, value in swapped
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with self.assertRaises(CaseValidationError):
      load_case(case.path)

  def test_element_target_rejects_invalid_or_duplicate_coordinates(self):
    seed, ordinary = self.sample_case(minimum_rank=1)
    shape = ordinary.manifest["output"]["shape"]
    in_bounds = [0] * len(shape)
    # 99 exceeds every admissible dimension, so this coordinate is out of
    # bounds whatever shape the seed produced.
    cases = {
      "outside the output shape": [([99] * len(shape), 1)],
      "same coordinate twice": [(list(in_bounds), 1), (list(in_bounds), 2)],
    }
    for label, elements in cases.items():
      with self.subTest(rejection=label), tempfile.TemporaryDirectory() as directory:
        result = generate_compositional(
          directory, seed=seed, policy="exact", elements=elements
        )
        summary = read_summary(directory)

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(summary["status"], "property_invalid")
        self.assertEqual(list(Path(directory).glob("case_*")), [])

    with tempfile.TemporaryDirectory() as directory:
      result = generate(
        directory,
        "-n",
        "1",
        "--tensor-dtype",
        "i32",
        "--tensor-profile-version",
        "v2",
        "--tensor-generator",
        "compositional",
        "--tensor-input-policy",
        "exact",
        "--tensor-output-element",
        "nonsense",
        seed=seed,
      )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("tensor output element", result.stderr)

  def test_element_argument_rejects_an_incomplete_number(self):
    """A prefix of a longer token is not a value: it would certify a different
    property than the one requested."""
    arguments = ("0=1.5", "0=1junk", "4294967296=1", "0x10=3", "-1=3", "0=", "=1")
    for argument in arguments:
      with (
        self.subTest(argument=argument),
        tempfile.TemporaryDirectory() as directory,
      ):
        result = generate(
          directory,
          "-n",
          "1",
          "--tensor-dtype",
          "i32",
          "--tensor-profile-version",
          "v2",
          "--tensor-generator",
          "compositional",
          "--tensor-input-policy",
          "exact",
          "--tensor-output-element",
          argument,
          seed=4,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tensor output element", result.stderr)

  def test_element_value_accepts_every_base_with_either_sign(self):
    """A sign before a base prefix is part of the literal, not zero.

    `+0x10` has to read as 16: a parser that looks past a leading minus only
    would stop at the `0` and certify a different property. The recorded
    request is checked directly, so the assertion is about parsing rather than
    about whether the solver then decides the query.
    """
    cases = {
      "+0x10": 16,
      "0x10": 16,
      "+0b11": 3,
      "0b11": 3,
      "+0o17": 15,
      "0o17": 15,
      "+10": 10,
      "-10": -10,
    }
    seed, ordinary = self.sample_case(minimum_rank=1, seeds=range(1, 40))
    coordinate = ",".join("0" for _ in ordinary.manifest["output"]["shape"])
    for literal, value in cases.items():
      with self.subTest(literal=literal), tempfile.TemporaryDirectory() as directory:
        result = generate(
          directory,
          "-n",
          "1",
          "--tensor-dtype",
          "i32",
          "--tensor-profile-version",
          "v2",
          "--tensor-generator",
          "compositional",
          "--tensor-input-policy",
          "exact",
          "--tensor-output-element",
          f"{coordinate}={literal}",
          seed=seed,
        )
        summary = read_summary(directory)

        self.assertNotIn("tensor output element", result.stderr)
        self.assertEqual(summary["output_elements"][0]["value"], value)

  def test_positive_sign_overflow_is_rejected(self):
    """`+0x80000000` is outside i32, not zero."""
    with tempfile.TemporaryDirectory() as directory:
      result = generate(
        directory,
        "-n",
        "1",
        "--tensor-dtype",
        "i32",
        "--tensor-profile-version",
        "v2",
        "--tensor-generator",
        "compositional",
        "--tensor-input-policy",
        "exact",
        "--tensor-output-element",
        "0=+0x80000000",
        seed=4,
      )

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("outside i32", result.stderr)

  def test_element_arguments_outside_option_parsing_are_not_constraints(self):
    """`--` ends option parsing, and another option's value is not an occurrence."""
    invocations = {
      "after the separator": ["--", "--tensor-output-element=0=1"],
      "as another option's value": ["--target", "--tensor-output-element=0=1"],
    }
    for label, extra in invocations.items():
      with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
        result = generate(directory, "-n", "1", *extra, seed=4)

        # Neither invocation names an element target, so none may be parsed as
        # one; whatever else they are rejected for is a different matter.
        self.assertNotIn("tensor output element", result.stderr)

  def test_separate_element_arguments_keep_their_boundaries(self):
    """A malformed occurrence fails even when a later one is well formed.

    `--tensor-output-element 0 --tensor-output-element 1=2` is two arguments,
    not the single constraint [0,1]=2.
    """
    with tempfile.TemporaryDirectory() as directory:
      result = generate(
        directory,
        "-n",
        "1",
        "--tensor-dtype",
        "i32",
        "--tensor-profile-version",
        "v2",
        "--tensor-generator",
        "compositional",
        "--tensor-input-policy",
        "exact",
        "--tensor-output-element",
        "0",
        "--tensor-output-element",
        "1=2",
        seed=4,
      )

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("tensor output element", result.stderr)

  def test_element_target_is_rejected_under_v3(self):
    with tempfile.TemporaryDirectory() as directory:
      result = generate_compositional(
        directory,
        seed=5,
        profile="v3",
        policy="exact",
        elements=[([0], 1)],
      )
      summary = read_summary(directory)

    self.assertNotEqual(result.returncode, 0)
    self.assertEqual(summary["status"], "invalid_profile")
    self.assertIn("tensor-v2", summary["reason"])

  def test_pinned_element_seeds_keep_their_regression(self):
    """Deterministic fixtures beside the sweeps: seed 3 decides both routes."""
    _, ordinary, complete_case, _ = self.complete_element_case(seeds=range(3, 4))
    self.assertEqual(complete_case.program, ordinary.program)
    self.assertTrue(complete_case.manifest["property"]["constraint"]["complete"])
    self.assertEqual(self.replay(complete_case), complete_case.expected)

    _, partial_case, _ = self.partial_element_case(seeds=range(3, 4))
    self.assertFalse(partial_case.manifest["property"]["constraint"]["complete"])
    self.assertEqual(self.replay(partial_case), partial_case.expected)

  def test_single_coordinate_partial_target_on_a_rank_one_output(self):
    """The smallest partial request: one coordinate, the rest from the reference."""
    seed, case, elements = self.partial_element_case(indices=[0])

    constraint = case.manifest["property"]["constraint"]
    self.assertFalse(constraint["complete"])
    self.assertEqual(len(constraint["elements"]), 1)
    self.assertEqual(self.replay(case), case.expected)


if __name__ == "__main__":
  unittest.main()
