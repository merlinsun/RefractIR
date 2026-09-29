"""Combined tensor test coverage: test_tensor_coverage.py, test_tensor_directed.py."""

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# Consolidated from test_tensor_coverage.py.
try:
  from test.lib import tensor_coverage
except ImportError:
  tensor_coverage = None


def make_case(
  root,
  source,
  input_bytes,
  *,
  stratum="composition",
  exactness="proved",
  loop_bounds=None,
):
  root = Path(root)
  root.mkdir(parents=True, exist_ok=True)
  (root / "program.sir").write_text(source, encoding="utf-8")
  (root / "inputs.bin").write_bytes(input_bytes)
  (root / "kernel.py").write_text(
    "@triton.jit\ndef kernel():\n  pass\n", encoding="utf-8"
  )
  (root / "compiler-ttgir.ir").write_text(
    "#blocked = #ttg.blocked<{sizePerThread = [1, 4], order = [1, 0]}>\n"
    "tt.func public @kernel(%arg0: tensor<2x4xf32, #blocked>) {\n"
    "  %0 = tt.reduce %arg0 axis = 1 : tensor<2x4xf32, #blocked>\n"
    "}\n",
    encoding="utf-8",
  )
  manifest = {
    "schema": "refractir.tensor-case.v1",
    "profile": "tensor-v1",
    "generator": {"primary_stratum": stratum, "exactness_evidence": exactness},
    "comparison": {"mode": "exact" if exactness == "proved" else "numeric"},
    "property": {"id": "sum"},
  }
  if loop_bounds:
    manifest["execution"] = {
      "loop_bounds": loop_bounds,
      "total_body_executions": sum(loop_bounds),
    }
    manifest["coverage"] = {
      "loop_structure": [{"bound": bound} for bound in loop_bounds]
    }
  (root / "case.json").write_text(json.dumps(manifest), encoding="utf-8")
  return root


class TensorCoverageTests(unittest.TestCase):
  def setUp(self):
    self.assertIsNotNone(tensor_coverage, "missing tensor coverage module")

  def test_archived_v1_baseline_has_case_and_structure_counts(self):
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads(
      (root / "tensor" / "campaigns" / "v1-baseline.json").read_text(encoding="utf-8")
    )

    self.assertEqual(baseline["profile"], "tensor-v1")
    self.assertEqual(baseline["emitted_cases"], 100)
    self.assertEqual(baseline["source_structure"]["unique_program_sir_sha256"], 85)

  def test_archived_baseline_marks_missing_compiler_ir_as_unavailable(self):
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads(
      (root / "tensor" / "campaigns" / "v1-baseline.json").read_text(encoding="utf-8")
    )

    self.assertEqual(
      baseline["compiled_structure"]["status"],
      "unavailable_in_archived_seed_index",
    )
    self.assertEqual(
      baseline["compiled_structure"]["normalized_ir_signatures"], "not_recorded"
    )

  def test_source_signature_ignores_names(self):
    first = "fun @kernel(%x: tensor [2, 4] f32) : tensor [2] f32 { let mut %r: tensor [2] f32 = undef; ^entry: %r = tensor sum %x, 1, drop; ret %r; }"
    renamed = "fun @f(%input: tensor [2, 4] f32) : tensor [2] f32 { let mut %result: tensor [2] f32 = undef; ^start: %result = tensor sum %input, 1, drop; ret %result; }"

    self.assertEqual(
      tensor_coverage.source_signature(first),
      tensor_coverage.source_signature(renamed),
    )

  def test_source_signature_keeps_shape_and_fanout(self):
    source = "fun @kernel(%x: tensor [2, 4] f32) : tensor [2] f32 { let mut %r: tensor [2] f32 = undef; ^entry: %r = tensor sum %x, 1, drop; ret %r; }"
    changed_shape = source.replace("[2, 4]", "[4, 2]")
    changed_fanout = source.replace("ret %r;", "%r = %r + %r; ret %r;")

    signature = tensor_coverage.source_signature(source)
    self.assertNotEqual(signature, tensor_coverage.source_signature(changed_shape))
    self.assertNotEqual(signature, tensor_coverage.source_signature(changed_fanout))

  def test_source_signature_treats_minus_as_an_operator(self):
    left_minus_right = "fun @kernel(%a: i32, %b: i32) : i32 { ^entry: ret %a- %b; }"
    right_minus_right = "fun @kernel(%a: i32, %b: i32) : i32 { ^entry: ret %b- %b; }"
    spaced_left_minus_right = (
      "fun @kernel(%a: i32, %b: i32): i32 { ^entry: ret %a - %b; }"
    )

    self.assertNotEqual(
      tensor_coverage.source_signature(left_minus_right),
      tensor_coverage.source_signature(right_minus_right),
    )
    self.assertEqual(
      tensor_coverage.source_signature(left_minus_right),
      tensor_coverage.source_signature(spaced_left_minus_right),
    )

  def test_generation_labels_and_input_exactness_do_not_change_source_identity(self):
    source = "fun @kernel(%x: tensor [2, 4] f32) : tensor [2] f32 { let mut %r: tensor [2] f32 = undef; ^entry: %r = tensor sum %x, 1, drop; ret %r; }"
    with tempfile.TemporaryDirectory() as temporary:
      exact = make_case(Path(temporary) / "exact", source, b"exact-input")
      exploratory = make_case(Path(temporary) / "exploratory", source, b"other-input")
      for case, stratum, proof in (
        (exact, "exact-f32-reduction", {"kind": "integer-valued-f32-reduction"}),
        (exploratory, "untargeted-exploration", None),
      ):
        manifest_path = case / "case.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["generator"]["primary_stratum"] = stratum
        manifest["generator"]["exactness_evidence"] = proof
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

      records = [
        tensor_coverage.case_coverage(exact),
        tensor_coverage.case_coverage(exploratory),
      ]

    self.assertEqual(records[0]["source_signature"], records[1]["source_signature"])
    self.assertNotEqual(
      records[0]["input_behavior"]["primary_stratum"],
      records[1]["input_behavior"]["primary_stratum"],
    )
    self.assertNotEqual(
      records[0]["input_behavior"]["exactness_evidence"],
      records[1]["input_behavior"]["exactness_evidence"],
    )

  def test_ir_signature_strips_unstable_ids(self):
    first = 'tt.func public @kernel(%arg0: tensor<2x4xf32>) { %0 = tt.reduce %arg0 axis = 1 : tensor<2x4xf32> loc("a.sir":10:2) }'
    renamed = 'tt.func public @other(%input: tensor<2x4xf32>) { %value = tt.reduce %input axis = 1 : tensor<2x4xf32> loc("b.sir":99:8) }'

    self.assertEqual(
      tensor_coverage.normalize_triton_ir("ttgir", first),
      tensor_coverage.normalize_triton_ir("ttgir", renamed),
    )

  def test_ir_signature_keeps_layout_attributes(self):
    first = "#blocked = #ttg.blocked<{sizePerThread = [1, 4], order = [1, 0]}>\n%0 = tt.reduce %arg0 axis = 1 : tensor<2x4xf32, #blocked>"
    changed = first.replace("order = [1, 0]", "order = [0, 1]")

    self.assertNotEqual(
      tensor_coverage.compiled_signature("ttgir", first),
      tensor_coverage.compiled_signature("ttgir", changed),
    )

  def test_compiled_signature_keeps_launch_settings(self):
    ir = "tt.func public @kernel(%arg0: tensor<2x4xf32>) { tt.return }"

    self.assertNotEqual(
      tensor_coverage.compiled_signature("ttgir", ir, context={"num_warps": 1}),
      tensor_coverage.compiled_signature("ttgir", ir, context={"num_warps": 4}),
    )

  def test_same_ir_keeps_distinct_inputs(self):
    source = "fun @kernel(%x: tensor [2, 4] f32) : tensor [2] f32 { let mut %r: tensor [2] f32 = undef; ^entry: %r = tensor sum %x, 1, drop; ret %r; }"
    with tempfile.TemporaryDirectory() as temporary:
      first = make_case(Path(temporary) / "case-a", source, b"first")
      second = make_case(Path(temporary) / "case-b", source, b"second")

      coverage = tensor_coverage.index_cases([first, second])

    self.assertEqual(len(coverage["cases"]), 2)
    self.assertEqual(
      coverage["cases"][0]["compiled_signatures"],
      coverage["cases"][1]["compiled_signatures"],
    )
    self.assertNotEqual(
      coverage["cases"][0]["input_behavior"]["inputs_sha256"],
      coverage["cases"][1]["input_behavior"]["inputs_sha256"],
    )

  def test_source_signature_ignores_labels_and_input_exactness(self):
    source = "fun @kernel(%x: tensor [2, 4] f32) : tensor [2] f32 { let mut %r: tensor [2] f32 = undef; ^entry: %r = tensor sum %x, 1, drop; ret %r; }"
    with tempfile.TemporaryDirectory() as temporary:
      exact = make_case(Path(temporary) / "exact", source, b"integer leaves")
      numeric = make_case(
        Path(temporary) / "numeric",
        source,
        b"fractional leaves",
        stratum="untargeted",
        exactness="none",
      )

      exact_record = tensor_coverage.case_coverage(exact)
      numeric_record = tensor_coverage.case_coverage(numeric)

    self.assertEqual(
      exact_record["source_signature"], numeric_record["source_signature"]
    )
    self.assertNotEqual(
      exact_record["input_behavior"]["primary_stratum"],
      numeric_record["input_behavior"]["primary_stratum"],
    )
    self.assertNotEqual(
      exact_record["input_behavior"]["exactness_evidence"],
      numeric_record["input_behavior"]["exactness_evidence"],
    )

  def test_unavailable_ir_stage_does_not_drop_other_stages(self):
    with tempfile.TemporaryDirectory() as temporary:
      case = make_case(
        Path(temporary) / "case",
        "fun @kernel(): i32 { ^entry: ret 0; }",
        b"input",
      )
      record = tensor_coverage.case_coverage(
        case, unavailable_stages={"ptx": "compiler did not expose PTX"}
      )

    self.assertIn("ttgir", record["compiled_signatures"])
    self.assertEqual(record["unavailable_stages"]["ptx"], "compiler did not expose PTX")

  def test_v2_coverage_labels_are_read_from_joint_coverage_record(self):
    with tempfile.TemporaryDirectory() as temporary:
      case = make_case(
        Path(temporary) / "case",
        "fun @kernel(): i32 { ^entry: ret 0; }",
        b"input",
      )
      manifest = json.loads((case / "case.json").read_text())
      manifest.pop("generator")
      manifest["coverage"] = {
        "primary_stratum": "exact-f32-reduction",
        "exactness_evidence": {"bound": 16777216},
      }
      (case / "case.json").write_text(json.dumps(manifest))
      record = tensor_coverage.case_coverage(case)

    self.assertEqual(record["input_behavior"]["primary_stratum"], "exact-f32-reduction")
    self.assertEqual(
      record["input_behavior"]["exactness_evidence"], {"bound": 16777216}
    )

  def test_nested_compiled_ir_is_indexed_per_run(self):
    with tempfile.TemporaryDirectory() as temporary:
      case = make_case(
        Path(temporary) / "case",
        "fun @kernel(): i32 { ^entry: ret 0; }",
        b"input",
      )
      artifact = case / "runs" / "affected" / "runs" / "compiled"
      artifact.mkdir(parents=True)
      (artifact / "compiler-ttgir.ir").write_text("tt.func @affected {}")
      record = tensor_coverage.case_coverage(case)

    self.assertIn("runs/affected/runs/compiled", record["compiled_runs"])
    self.assertIn("ttgir", record["compiled_runs"]["runs/affected/runs/compiled"])

  def test_raw_hashes_are_retained_for_nested_ir(self):
    with tempfile.TemporaryDirectory() as temporary:
      case = make_case(
        Path(temporary) / "case",
        "fun @kernel(): i32 { ^entry: ret 0; }",
        b"input",
      )
      artifact = case / "runs" / "corrected" / "runs" / "compiled"
      artifact.mkdir(parents=True)
      ir_path = artifact / "compiler-ptx.ir"
      ir_path.write_text("ret;")
      record = tensor_coverage.case_coverage(case)
      expected_digest = hashlib.sha256(ir_path.read_bytes()).hexdigest()

    self.assertEqual(
      record["run_raw_artifact_sha256"]["runs/corrected/runs/compiled"]["ptx"],
      expected_digest,
    )

  def test_two_builds_keep_compiled_signatures_separate(self):
    with tempfile.TemporaryDirectory() as temporary:
      case = make_case(
        Path(temporary) / "case",
        "fun @kernel(): i32 { ^entry: ret 0; }",
        b"input",
      )
      (case / "compiler-ttgir.ir").unlink()
      for build, layout in (
        ("affected", "order = [1, 0]"),
        ("corrected", "order = [0, 1]"),
      ):
        artifact = case / "runs" / build / "runs" / "compiled"
        artifact.mkdir(parents=True)
        (artifact / "compiler-ttgir.ir").write_text(
          f"#blocked = #ttg.blocked<{{{layout}}}>"
        )
      record = tensor_coverage.case_coverage(case)

    affected = record["compiled_runs"]["runs/affected/runs/compiled"]["ttgir"]
    corrected = record["compiled_runs"]["runs/corrected/runs/compiled"]["ttgir"]
    self.assertNotEqual(affected, corrected)

  def test_source_loop_is_not_counted_as_compiled_loop_coverage(self):
    with tempfile.TemporaryDirectory() as temporary:
      case = make_case(
        Path(temporary) / "case",
        "fun @kernel(): i32 { ^entry: ret 0; }",
        b"input",
        loop_bounds=[2],
      )
      record = tensor_coverage.case_coverage(case)

    execution = record["execution_coverage"]
    self.assertTrue(execution["source_loop_present"])
    self.assertFalse(execution["compiled_loop_coverage"]["case-artifacts"]["covered"])

  def test_scalar_ir_loop_does_not_count_for_tensor_loop_coverage(self):
    scalar_loop = """\
tt.func public @kernel(%arg: i32) {
  %0 = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %arg) -> (i32) : i32 {
    %1 = arith.addi %acc, %i : i32
    scf.yield %1 : i32
  }
  tt.return
}
"""
    coverage = tensor_coverage._ttgir_loop_coverage(scalar_loop)

    self.assertEqual(coverage["loop_count"], 1)
    self.assertFalse(coverage["tensor_carried_loop"])
    self.assertFalse(coverage["covered"])

  def test_tensor_carried_loop_without_reduction_or_update_is_not_covered(self):
    loop_ir = """\
tt.func public @kernel(%arg: tensor<2xf32>) {
  %0 = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %arg) -> (tensor<2xf32>) : i32 {
    scf.yield %acc : tensor<2xf32>
  }
  tt.return
}
"""
    coverage = tensor_coverage._ttgir_loop_coverage(loop_ir)

    self.assertTrue(coverage["tensor_carried_loop"])
    self.assertFalse(coverage["covered"])

  def test_tensor_loop_with_reduction_and_carried_update_is_covered(self):
    loop_ir = """\
tt.func public @kernel(%arg: tensor<2x2xf32>, %initial: tensor<2xf32>) {
  %rows = "tt.reduce"(%arg) <{axis = 1 : i32}> : tensor<2x2xf32> -> tensor<2xf32>
  %0 = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %initial) -> (tensor<2xf32>) : i32 {
    %1 = arith.addf %acc, %rows : tensor<2xf32>
    scf.yield %1 : tensor<2xf32>
  }
  tt.return
}
"""
    coverage = tensor_coverage._ttgir_loop_coverage(loop_ir)

    self.assertEqual(coverage["loop_count"], 1)
    self.assertTrue(coverage["tensor_carried_loop"])
    self.assertEqual(coverage["reduction_count"], 1)
    self.assertTrue(coverage["covered"])

  def test_coverage_report_separates_source_loops_from_compiled_loops(self):
    with tempfile.TemporaryDirectory() as temporary:
      source = "fun @kernel(): i32 { ^entry: ret 0; }"
      compiled = make_case(
        Path(temporary) / "compiled", source, b"a", loop_bounds=[2, 2]
      )
      missing = make_case(Path(temporary) / "missing", source, b"b", loop_bounds=[2])
      (compiled / "compiler-ttgir.ir").write_text(
        "scf.for %i = %c0 to %c2 iter_args(%acc = %x) -> (tensor<2xf32>) : i32 {\n"
        "  %r = tt.reduce %x\n  %v = arith.addf %acc, %r\n  scf.yield %v\n}\n"
      )
      report = tensor_coverage.index_cases([compiled, missing])

    self.assertEqual(report["source_loop_cases"], 2)
    self.assertEqual(report["compiled_loop_cases"], 1)

  def test_index_counts_compiled_structures_from_each_pinned_run(self):
    with tempfile.TemporaryDirectory() as temporary:
      case = make_case(
        Path(temporary) / "case",
        "fun @kernel(): i32 { ^entry: ret 0; }",
        b"input",
      )
      (case / "compiler-ttgir.ir").unlink()
      for build, layout in (
        ("affected", "order = [1, 0]"),
        ("corrected", "order = [0, 1]"),
      ):
        artifact = case / "runs" / build / "runs" / "compiled"
        artifact.mkdir(parents=True)
        (artifact / "compiler-ttgir.ir").write_text(
          f"#blocked = #ttg.blocked<{{{layout}}}>"
        )
      report = tensor_coverage.index_cases([case])

    self.assertEqual(report["unique_compiled_signatures"]["ttgir"], 2)


if __name__ == "__main__":
  unittest.main()

# Consolidated from test_tensor_directed.py.
try:
  from test.lib import tensor_directed
except ImportError:
  tensor_directed = None

_SQUARE_SUM = """\
fun @kernel(%x: tensor [4, 4] i32) : tensor [4] i32 {
  let mut %rows: tensor [4] i32 = undef;
^entry:
  %rows = tensor sum %x, 0, drop;
  ret %rows;
}
"""

_RECTANGULAR_SUM = """\
fun @kernel(%x: tensor [2, 4] i32) : tensor [4] i32 {
  let mut %rows: tensor [4] i32 = undef;
^entry:
  %rows = tensor sum %x, 0, drop;
  ret %rows;
}
"""

_LOOP_REDUCTION = """\
fun @kernel(%x: tensor [2, 2] i32) : tensor [2] i32 {
  let mut %i: i32 = undef;
  let mut %acc: tensor [2] i32 = {0, 0};
  let mut %rows: tensor [2] i32 = undef;
^entry:
  %i = 0;
  br ^header;
^header:
  br %i < 2, ^body, ^exit;
^body:
  %rows = tensor sum %x, 1, drop;
  %acc = tensor add %acc, %rows;
  %i = %i + 1;
  br ^header;
^exit:
  ret %acc;
}
"""

_SINGLE_REDUCTION_GROUP = """\
fun @kernel(%x: tensor [1, 2] i32) : tensor [1] i32 {
  let mut %rows: tensor [1] i32 = undef;
^entry:
  %rows = tensor sum %x, 1, drop;
  ret %rows;
}
"""


def _stride_model_probe(
  *, dtype="i32", values=(1, 2, 3, 4), physical_strides=(1, 2), storage_span=4
):
  format_code = "i" if dtype == "i32" else "f"
  input_bytes = struct.pack(f"<4{format_code}", *values)
  source = (
    f"fun @kernel(%x: tensor [2, 2] {dtype}) : tensor [2, 2] {dtype} {{ ret %x; }}"
  )
  manifest = {
    "profile": "tensor-v2",
    "entry": "kernel",
    "inputs": [
      {
        "name": "x",
        "dtype": dtype,
        "shape": [2, 2],
        "element_strides": list(physical_strides),
        "element_offset": 0,
        "storage_span": storage_span,
        "axis_order": sorted(range(2), key=lambda axis: physical_strides[axis]),
        "byte_offset": 0,
        "byte_length": len(input_bytes),
      }
    ],
    "output": {"dtype": dtype, "shape": [2, 2]},
    "comparison": {"mode": "exact", "atol": [0, 1], "rtol": [0, 1]},
    "execution": {"gap_sentinel_by_dtype": {"i32": "5a5a5a5a", "f32": "7fc00000"}},
  }
  model = {
    "name": "wrong_physical_stride",
    "status": "applicable",
    "input": "x",
    "strides": [2, 1],
  }
  return source, input_bytes, manifest, model


class TensorDirectedTests(unittest.TestCase):
  def setUp(self):
    self.assertIsNotNone(tensor_directed, "missing tensor-directed input selector")

  def test_wrong_axis_model_requires_same_output_shape(self):
    models = tensor_directed.build_fault_models(_RECTANGULAR_SUM)

    wrong_axis = [model for model in models if model["name"] == "wrong_reduction_axis"]
    self.assertEqual(len(wrong_axis), 1)
    self.assertEqual(wrong_axis[0]["status"], "inapplicable")
    self.assertIsNone(wrong_axis[0].get("source"))

  def test_square_wrong_axis_model_preserves_output_shape(self):
    models = tensor_directed.build_fault_models(_SQUARE_SUM)

    wrong_axis = [model for model in models if model["name"] == "wrong_reduction_axis"]
    self.assertEqual(len(wrong_axis), 1)
    self.assertEqual(wrong_axis[0]["status"], "applicable")
    self.assertIn("tensor sum %x, 1, drop", wrong_axis[0]["source"])

  def test_numerical_distinction_uses_strict_four_tolerance_margin(self):
    expected = struct.pack("<f", 1.0)
    equal_margin = struct.pack("<f", 3.0)
    above_margin = struct.pack("<f", 3.000001)
    comparison = {"mode": "numeric", "atol": [1, 2], "rtol": [0, 1]}

    equal = tensor_directed.compare_model_output(
      equal_margin, expected, "f32", [1], comparison
    )
    above = tensor_directed.compare_model_output(
      above_margin, expected, "f32", [1], comparison
    )

    self.assertFalse(equal["distinguished"])
    self.assertTrue(above["distinguished"])
    self.assertEqual(equal["status"], "applicable_not_distinguished")

  def test_ordinary_arm_keeps_first_admissible_candidate_without_d(self):
    candidates = [b"ordinary-first", b"directed-second"]
    result = tensor_directed.select_input_candidate(
      candidates,
      arm="ordinary",
      property_validator=lambda _candidate: True,
      distinction_evaluator=lambda candidate: candidate == b"directed-second",
    )

    self.assertEqual(result["status"], "accepted")
    self.assertEqual(result["selected"], b"ordinary-first")
    self.assertFalse(result["distinguished"])

  def test_directed_arm_requires_both_p_and_d(self):
    candidates = [b"p-only", b"d-only", b"p-and-d"]
    result = tensor_directed.select_input_candidate(
      candidates,
      arm="directed",
      property_validator=lambda candidate: candidate != b"d-only",
      distinction_evaluator=lambda candidate: candidate != b"p-only",
    )

    self.assertEqual(result["status"], "accepted")
    self.assertEqual(result["selected"], b"p-and-d")
    self.assertTrue(result["distinguished"])
    self.assertEqual(
      [attempt["property_passed"] for attempt in result["attempts"]],
      [True, False, True],
    )

  def test_directed_exhaustion_retains_p_success_without_d(self):
    result = tensor_directed.select_input_candidate(
      [b"p-only"],
      arm="directed",
      property_validator=lambda _candidate: True,
      distinction_evaluator=lambda _candidate: False,
    )

    self.assertEqual(result["status"], "proposal_exhausted")
    self.assertTrue(result["property_passed"])
    self.assertFalse(result["distinguished"])

  def test_target_model_is_the_ordinary_arm_objective(self):
    distinction = {
      "distinguished": True,
      "models": [
        {"name": "target", "applicability": "applicable", "distinguished": False},
        {"name": "other", "applicability": "applicable", "distinguished": True},
      ],
    }

    result = tensor_directed.select_input_candidate(
      [b"first-p-valid"],
      arm="ordinary",
      property_validator=lambda _candidate: True,
      distinction_evaluator=lambda _candidate: distinction,
      target_model="target",
    )

    self.assertEqual(result["selected"], b"first-p-valid")
    self.assertFalse(result["distinguished"])
    self.assertEqual(result["target_model"], "target")

  def test_directed_target_model_ignores_other_models_and_finds_later_witness(self):
    outcomes = {
      b"other-only": {
        "distinguished": True,
        "models": [
          {"name": "target", "applicability": "applicable", "distinguished": False},
          {"name": "other", "applicability": "applicable", "distinguished": True},
        ],
      },
      b"target-witness": {
        "distinguished": True,
        "models": [
          {"name": "target", "applicability": "applicable", "distinguished": True},
          {"name": "other", "applicability": "applicable", "distinguished": False},
        ],
      },
    }

    result = tensor_directed.select_input_candidate(
      list(outcomes),
      arm="directed",
      property_validator=lambda _candidate: True,
      distinction_evaluator=outcomes.__getitem__,
      target_model="target",
    )

    self.assertEqual(result["selected"], b"target-witness")
    self.assertEqual(result["proposal_count"], 2)
    self.assertTrue(result["distinguished"])

  def test_inapplicable_target_model_is_classified_before_proposals(self):
    manifest = {
      "profile": "tensor-v2",
      "entry": "kernel",
      "output": {"dtype": "i32", "shape": [4]},
      "coverage": {"primary_stratum": "multi-input-composition"},
    }
    loaded_case = SimpleNamespace(
      manifest=manifest,
      program=_RECTANGULAR_SUM.encode("utf-8"),
      inputs=b"input",
    )

    with tempfile.TemporaryDirectory() as directory:
      with (
        mock.patch.object(
          tensor_directed.tensor_case, "load_case", return_value=loaded_case
        ),
        mock.patch.object(tensor_directed, "_run_reference") as run_reference,
      ):
        result = tensor_directed.prepare_case_arm(
          directory,
          arm="directed",
          seed=31,
          target_model="wrong_reduction_axis",
        )

    self.assertEqual(result["status"], "no_applicable_model")
    self.assertEqual(result["proposal_count"], 0)
    self.assertEqual(result["target_model"], "wrong_reduction_axis")
    run_reference.assert_not_called()

  def test_unknown_target_model_is_rejected_before_reference_evaluation(self):
    manifest = {
      "profile": "tensor-v2",
      "entry": "kernel",
      "output": {"dtype": "i32", "shape": [4]},
      "coverage": {"primary_stratum": "multi-input-composition"},
    }
    loaded_case = SimpleNamespace(
      manifest=manifest,
      program=_RECTANGULAR_SUM.encode("utf-8"),
      inputs=b"input",
    )

    with tempfile.TemporaryDirectory() as directory:
      with (
        mock.patch.object(
          tensor_directed.tensor_case, "load_case", return_value=loaded_case
        ),
        mock.patch.object(tensor_directed, "_run_reference") as run_reference,
      ):
        with self.assertRaisesRegex(ValueError, "unknown fault model"):
          tensor_directed.prepare_case_arm(
            directory,
            arm="directed",
            seed=31,
            target_model="not-a-model",
          )

    run_reference.assert_not_called()

  def test_targeted_selection_records_only_the_selected_model_result(self):
    manifest = {
      "schema": "refractir.tensor-case.v2",
      "profile": "tensor-v2",
      "entry": "kernel",
      "inputs": [
        {
          "name": "x",
          "dtype": "i32",
          "shape": [4, 4],
          "element_strides": [4, 1],
          "element_offset": 0,
          "storage_span": 16,
          "axis_order": [0, 1],
          "byte_offset": 0,
          "byte_length": 64,
        }
      ],
      "output": {"dtype": "i32", "shape": [4]},
      "comparison": {"mode": "exact", "atol": [0, 1], "rtol": [0, 1]},
      "coverage": {"primary_stratum": "multi-input-composition"},
      "property": {},
      "evidence": {
        "kind": "construction",
        "status": "construction_checked",
        "artifacts": [],
      },
      "reference": {},
      "files": {},
    }
    loaded_case = SimpleNamespace(
      manifest=manifest,
      program=_SQUARE_SUM.encode("utf-8"),
      inputs=b"old-input",
    )
    expected = struct.pack("<4i", 4, 8, 12, 16)
    input_bytes = bytes(64)
    model_result = {
      "distinguished": False,
      "models": [
        {
          "name": "wrong_reduction_axis",
          "applicability": "applicable",
          "distinguished": False,
        }
      ],
    }

    with tempfile.TemporaryDirectory() as directory:
      with (
        mock.patch.object(
          tensor_directed.tensor_case, "load_case", return_value=loaded_case
        ),
        mock.patch.object(
          tensor_directed, "_proposals", return_value=iter([input_bytes])
        ),
        mock.patch.object(tensor_directed, "_run_reference", return_value=expected),
        mock.patch.object(
          tensor_directed, "_run_models", return_value=model_result
        ) as run_models,
      ):
        result = tensor_directed.prepare_case_arm(
          directory,
          arm="ordinary",
          seed=12,
          target_model="wrong_reduction_axis",
        )
      written_manifest = json.loads(
        (Path(directory) / "case.json").read_text(encoding="utf-8")
      )

    self.assertEqual(result["D"], "fault-model:wrong_reduction_axis")
    self.assertFalse(result["distinguished"])
    self.assertEqual(result["cost"]["property_reference_evaluations"], 1)
    self.assertEqual(result["cost"]["model_reference_evaluations"], 0)
    self.assertEqual(
      written_manifest["coverage"]["input_behavior"]["fault_models"],
      model_result["models"],
    )
    self.assertEqual(run_models.call_args.args[0][0]["name"], "wrong_reduction_axis")

  def test_applicable_target_without_witness_is_budget_exhaustion(self):
    result = tensor_directed.select_input_candidate(
      [b"other-only"],
      arm="directed",
      property_validator=lambda _candidate: True,
      distinction_evaluator=lambda _candidate: {
        "distinguished": True,
        "models": [
          {"name": "target", "applicability": "applicable", "distinguished": False},
          {"name": "other", "applicability": "applicable", "distinguished": True},
        ],
      },
      target_model="target",
    )

    self.assertEqual(result["status"], "proposal_exhausted")
    self.assertTrue(result["objective_applicable"])
    self.assertEqual(result["target_model"], "target")

  def test_undefined_modeled_output_is_invalid_not_applicable(self):
    source = "fun @kernel(%x: tensor [1] f32) : tensor [1] f32 { ret %x; }"
    manifest = {
      "profile": "tensor-v2",
      "output": {"dtype": "f32", "shape": [1]},
      "entry": "kernel",
      "comparison": {"mode": "numeric", "atol": [1, 1000], "rtol": [1, 1000]},
    }
    model = {"name": "bad", "status": "applicable"}
    nan = struct.pack("<I", 0x7FC00000)

    with mock.patch.object(tensor_directed, "_run_reference", return_value=nan):
      result = tensor_directed._run_models(
        [model], source, b"input", struct.pack("<f", 1.0), manifest, None
      )

    self.assertEqual(result["models"][0]["applicability"], "invalid_model")
    self.assertFalse(result["models"][0]["defined"])
    self.assertFalse(result["distinguished"])
    self.assertEqual(result["reference_evaluations"], 1)
    self.assertGreaterEqual(result["reference_evaluation_s"], 0.0)

  def test_loop_bound_fault_model_decrements_a_counted_reduction_loop(self):
    models = tensor_directed.build_fault_models(_LOOP_REDUCTION)

    loop_bound = [model for model in models if model["name"] == "loop_bound_minus_one"]
    self.assertEqual(len(loop_bound), 1)
    self.assertEqual(loop_bound[0]["status"], "applicable")
    self.assertIn("br %i < 1, ^body, ^exit;", loop_bound[0]["source"])

  def test_loop_accumulator_fault_model_drops_only_the_accumulation_update(self):
    models = tensor_directed.build_fault_models(_LOOP_REDUCTION)

    accumulator = [
      model for model in models if model["name"] == "loop_accumulator_overwrite"
    ]
    self.assertEqual(len(accumulator), 1)
    self.assertEqual(accumulator[0]["status"], "applicable")
    self.assertIn("%acc = %rows;", accumulator[0]["source"])
    self.assertIn("tensor [2] i32", accumulator[0]["source"])

  def test_loop_models_reject_unvalidated_counter_updates(self):
    outside_latch = _LOOP_REDUCTION.replace(
      "  %i = %i + 1;\n  br ^header;",
      "  %i = %i + 1;\n  br ^latch;\n^latch:\n  br ^header;",
    )
    extra_write = _LOOP_REDUCTION.replace(
      "^exit:\n  ret %acc;", "^exit:\n  %i = 0;\n  ret %acc;"
    )
    nonunit_increment = _LOOP_REDUCTION.replace("%i = %i + 1;", "%i = %i + 2;")

    for source in (outside_latch, extra_write, nonunit_increment):
      with self.subTest(source=source):
        models = tensor_directed.build_fault_models(source)
        loop_models = [
          model
          for model in models
          if model["name"] in {"loop_bound_minus_one", "loop_accumulator_overwrite"}
        ]

        self.assertEqual(len(loop_models), 2)
        self.assertTrue(all(model["status"] == "inapplicable" for model in loop_models))

  def test_reduction_group_fault_model_reverses_defined_output_groups(self):
    reference = struct.pack("<2i", 7, 11)
    manifest = {
      "entry": "kernel",
      "output": {"dtype": "i32", "shape": [2]},
      "comparison": {"mode": "exact", "atol": [0, 1], "rtol": [0, 1]},
    }
    models = tensor_directed.build_fault_models(_LOOP_REDUCTION)
    groups = [model for model in models if model["name"] == "swapped_reduction_groups"]

    self.assertEqual(len(groups), 1)
    self.assertEqual(groups[0]["edit"], {"axis": 0, "group_order": "reverse"})
    result = tensor_directed._run_models(
      groups, _LOOP_REDUCTION, b"", reference, manifest, None
    )

    record = result["models"][0]
    self.assertTrue(result["distinguished"])
    self.assertTrue(record["defined"])
    self.assertEqual(record["output_shape"], [2])
    self.assertEqual(record["comparison"]["differing_indices"], [0, 1])

    equal_groups = struct.pack("<2i", 8, 8)
    equal_result = tensor_directed._run_models(
      groups, _LOOP_REDUCTION, b"", equal_groups, manifest, None
    )
    self.assertFalse(equal_result["distinguished"])
    self.assertEqual(equal_result["models"][0]["applicability"], "applicable")

  def test_reduction_group_model_is_inapplicable_to_a_single_output_group(self):
    models = tensor_directed.build_fault_models(_SINGLE_REDUCTION_GROUP)

    groups = [model for model in models if model["name"] == "swapped_reduction_groups"]
    self.assertEqual(len(groups), 1)
    self.assertEqual(groups[0]["status"], "inapplicable")

  def test_stride_model_executes_the_mutated_i32_input(self):
    source, input_bytes, manifest, model = _stride_model_probe()
    modeled_inputs = struct.pack("<4i", 1, 3, 2, 4)

    with mock.patch.object(
      tensor_directed, "_run_reference", side_effect=lambda _s, data, **_kw: data
    ) as run_reference:
      result = tensor_directed._run_models(
        [model], source, input_bytes, input_bytes, manifest, None
      )

    self.assertEqual(run_reference.call_args.args[1], modeled_inputs)
    self.assertTrue(result["distinguished"])
    self.assertEqual(result["reference_evaluations"], 1)

  def test_stride_model_executes_the_mutated_f32_input(self):
    source, input_bytes, manifest, model = _stride_model_probe(dtype="f32")
    modeled_inputs = struct.pack("<4f", 1.0, 3.0, 2.0, 4.0)

    with mock.patch.object(
      tensor_directed, "_run_reference", side_effect=lambda _s, data, **_kw: data
    ) as run_reference:
      result = tensor_directed._run_models(
        [model], source, input_bytes, input_bytes, manifest, None
      )

    self.assertEqual(run_reference.call_args.args[1], modeled_inputs)
    self.assertTrue(result["distinguished"])
    self.assertEqual(result["reference_evaluations"], 1)

  def test_stride_model_can_be_applicable_without_distinguishing(self):
    source, input_bytes, manifest, model = _stride_model_probe(values=(7, 7, 7, 7))

    with mock.patch.object(
      tensor_directed, "_run_reference", side_effect=lambda _s, data, **_kw: data
    ):
      result = tensor_directed._run_models(
        [model], source, input_bytes, input_bytes, manifest, None
      )

    record = result["models"][0]
    self.assertEqual(record["applicability"], "applicable")
    self.assertTrue(record["defined"])
    self.assertFalse(record["distinguished"])

  def test_stride_model_reads_deterministic_physical_gap_bytes(self):
    source, input_bytes, manifest, model = _stride_model_probe(
      physical_strides=(3, 1), storage_span=5
    )
    expected_modeled = struct.pack("<4i", 1, 2, 0x5A5A5A5A, 3)

    with mock.patch.object(
      tensor_directed, "_run_reference", side_effect=lambda _s, data, **_kw: data
    ) as run_reference:
      tensor_directed._run_models(
        [model], source, input_bytes, input_bytes, manifest, None
      )

    self.assertEqual(run_reference.call_args.args[1], expected_modeled)

  def test_undefined_stride_model_result_is_classified_invalid(self):
    source, input_bytes, manifest, model = _stride_model_probe()

    with mock.patch.object(
      tensor_directed,
      "_run_reference",
      side_effect=tensor_directed.ReferenceEvaluationError(
        "modeled input is undefined"
      ),
    ) as run_reference:
      result = tensor_directed._run_models(
        [model], source, input_bytes, input_bytes, manifest, None
      )

    self.assertEqual(run_reference.call_count, 1)
    self.assertEqual(result["reference_evaluations"], 1)
    self.assertEqual(result["models"][0]["applicability"], "invalid_model")
    self.assertFalse(result["models"][0]["defined"])

  def test_loop_reduction_stratum_can_use_directed_case_preparation(self):
    input_bytes = struct.pack("<4i", 1, 2, 3, 4)
    expected = struct.pack("<2i", 6, 14)
    manifest = {
      "schema": "refractir.tensor-case.v2",
      "profile": "tensor-v2",
      "entry": "kernel",
      "inputs": [
        {
          "name": "x",
          "dtype": "i32",
          "shape": [2, 2],
          "element_strides": [1, 2],
          "element_offset": 0,
          "storage_span": 4,
          "axis_order": [0, 1],
          "byte_offset": 0,
          "byte_length": 16,
        }
      ],
      "output": {
        "name": "return",
        "dtype": "i32",
        "shape": [2],
        "element_strides": [1],
        "element_offset": 0,
        "storage_span": 2,
        "axis_order": [0],
        "tile_shape": [2],
        "tile_origin": [0],
        "file": "expected.bin",
        "byte_offset": 0,
        "byte_length": 8,
      },
      "comparison": {
        "mode": "exact",
        "policy": "exact-v1",
        "atol": [0, 1],
        "rtol": [0, 1],
      },
      "reference": {"precisions": [], "stable": True},
      "property": {"id": "loop-reduction", "source": "bounded accumulator loop"},
      "evidence": {"kind": "construction", "status": "validated", "artifacts": []},
      "execution": {
        "gap_sentinel_by_dtype": {"i32": "5a5a5a5a", "f32": "7fc00000"},
        "output_canary_by_dtype": {"i32": "a5a5a5a5", "f32": "7fc00001"},
      },
      "coverage": {"primary_stratum": "loop-reduction"},
      "files": {},
    }
    case = SimpleNamespace(
      manifest=manifest,
      program=_LOOP_REDUCTION.encode("utf-8"),
      inputs=input_bytes,
      expected=expected,
    )
    with tempfile.TemporaryDirectory() as directory:
      with mock.patch.object(
        tensor_directed.tensor_case, "load_case", return_value=case
      ):
        with mock.patch.object(
          tensor_directed, "_run_reference", return_value=expected
        ):
          result = tensor_directed.prepare_case_arm(
            Path(directory),
            arm="ordinary",
            seed=5,
            proposal_limit=1,
            symiri=Path("/bin/true"),
          )

    self.assertEqual(result["status"], "accepted")
    self.assertTrue(result["property_passed"])


if __name__ == "__main__":
  unittest.main()
