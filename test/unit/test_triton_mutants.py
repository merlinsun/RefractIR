import json
import tempfile
import unittest
from pathlib import Path

from test.lib import build_tensor_regression_cases, triton_mutants
from test.lib.triton_mutants import mutate_kernel


class TritonMutantTests(unittest.TestCase):
  @staticmethod
  def _classify(result):
    classifier = getattr(triton_mutants, "classify_mutant_result", None)
    if classifier is not None:
      return classifier(result)
    return triton_mutants.is_fault_detected(result["status"])

  def test_exact_mismatch_is_detected(self):
    result = self._classify(
      {"status": "exact_mismatch", "compiled": True, "executed": True}
    )

    self.assertEqual(result, "exact_kill")

  def test_numeric_mismatch_is_detected(self):
    result = self._classify(
      {"status": "numeric_mismatch", "compiled": True, "executed": True}
    )

    self.assertEqual(result, "numeric_kill")

  def test_compile_failure_is_separate(self):
    result = self._classify(
      {"status": "compile_failure", "compiled": False, "executed": False}
    )

    self.assertEqual(result, "compile_rejection")

  def test_runtime_failure_is_separate(self):
    result = self._classify(
      {"status": "runtime_failure", "compiled": True, "executed": False}
    )

    self.assertEqual(result, "runtime_failure")

  def test_passed_mutant_is_a_survivor(self):
    result = self._classify({"status": "pass", "compiled": True, "executed": True})

    self.assertEqual(result, "survivor")

  def test_invalid_mutant_syntax_is_separate(self):
    result = self._classify(
      {"status": "invalid_mutant", "compiled": False, "executed": False}
    )

    self.assertEqual(result, "invalid")

  def test_summary_keeps_denominators_and_separate_outcomes(self):
    results = [
      {"status": "exact_mismatch", "compiled": True, "executed": True},
      {"status": "numeric_mismatch", "compiled": True, "executed": True},
      {"status": "compile_failure", "compiled": False, "executed": False},
      {"status": "runtime_failure", "compiled": True, "executed": False},
      {"status": "pass", "compiled": True, "executed": True},
    ]

    summarize = getattr(triton_mutants, "summarize_mutant_results", None)
    summary = summarize(results) if summarize is not None else None

    self.assertEqual(summary["attempted"], 5)
    self.assertEqual(summary["compiled"], 4)
    self.assertEqual(summary["executed"], 3)
    self.assertEqual(summary["killed"], 2)
    self.assertEqual(summary["compile_rejection"], 1)
    self.assertEqual(summary["runtime_failure"], 1)
    self.assertEqual(summary["survivor"], 1)

  def test_wrong_softmax_axis_changes_only_the_axis(self):
    source = "_sir_out = tl.softmax(_sir_x, dim=1)\n"
    mutant, _ = mutate_kernel("wrong_softmax_axis", source, (2, 4))
    self.assertEqual(mutant, "_sir_out = tl.softmax(_sir_x, dim=0)\n")

  def test_missing_axis_restoration_removes_expand(self):
    source = "_sir_out = tl.expand_dims(_sir_sum, 1)\n"
    mutant, _ = mutate_kernel("missing_axis_restoration", source, (2, 4))
    self.assertEqual(mutant, "_sir_out = _sir_sum\n")

  def test_wrong_transpose_permutation_changes_axes(self):
    source = "_sir_out = tl.permute(_sir_x, (0, 1))\n"
    mutant, _ = mutate_kernel("wrong_transpose_permutation", source, (2, 4))
    self.assertEqual(mutant, "_sir_out = tl.permute(_sir_x, (1, 0))\n")

  def test_transpose_mutant_uses_shape_preserving_case(self):
    self.assertEqual(
      triton_mutants._MUTANT_CASES["wrong_transpose_permutation"],
      "transpose_2x2_perm0_1",
    )

  def test_transpose_mutant_case_exists_in_regression_matrix(self):
    case_name = triton_mutants._MUTANT_CASES["wrong_transpose_permutation"]
    case = self._transpose_mutant_case()

    self.assertEqual(case_name, self._transpose_case_name(case))
    self.assertEqual(len(set(case["shape"])), 1)

  def test_transpose_mutant_source_keeps_square_output_shape(self):
    case = self._transpose_mutant_case()
    source = build_tensor_regression_cases._transpose_source(
      case["shape"], case["permutation"]
    )

    self.assertIn("%x: tensor [2, 2] i32", source)
    self.assertIn(": tensor [2, 2] i32 {", source)

  def test_transpose_mutation_preserves_selected_case_shape(self):
    case = self._transpose_mutant_case()
    shape = tuple(case["shape"])
    permutation = tuple(case["permutation"])
    source = f"_sir_out = tl.permute(_sir_x, {permutation})\n"
    mutant, _ = mutate_kernel("wrong_transpose_permutation", source, shape)
    original_shape = tuple(shape[axis] for axis in permutation)
    mutant_shape = tuple(shape[axis] for axis in reversed(permutation))

    self.assertNotEqual(mutant, source)
    self.assertEqual(original_shape, mutant_shape)

  def test_transpose_mutant_case_has_distinguishing_values(self):
    case = self._transpose_mutant_case()
    shape = tuple(case["shape"])
    values = build_tensor_regression_cases._input_values(shape, "i32")
    transposed = [values[0], values[2], values[1], values[3]]

    self.assertEqual(shape, (2, 2))
    self.assertEqual(len(set(values)), 4)
    self.assertNotEqual(values, transposed)

  def test_axis_restoration_mutant_uses_broadcast_compatible_case(self):
    self.assertEqual(
      triton_mutants._MUTANT_CASES["missing_axis_restoration"],
      "composition_axis-restoration_sum-expand-add",
    )

  def test_axis_restoration_case_is_in_the_regression_matrix(self):
    group, order, _, _, _ = self._axis_restoration_definition()
    case_name = f"composition_{group}_{order}"

    self.assertEqual(
      triton_mutants._MUTANT_CASES["missing_axis_restoration"], case_name
    )

  def test_axis_restoration_case_has_shape_preserving_broadcast(self):
    _, _, source, _, _ = self._axis_restoration_definition()

    self.assertIn("%x: tensor [2, 2] i32", source)
    self.assertIn("%bias: tensor [2, 2] i32", source)
    self.assertIn("%expanded: tensor [2, 1] i32", source)
    self.assertIn("tensor add %expanded, %bias", source)
    self.assertIn(": tensor [2, 2] i32 {", source)

  def test_missing_axis_mutation_remains_broadcastable(self):
    _, _, sir_source, _, _ = self._axis_restoration_definition()
    self.assertIn("%expanded = tensor expand %rows, 1;", sir_source)
    self.assertIn("tensor add %expanded, %bias", sir_source)
    source = (
      "_sir_expanded = tl.expand_dims(_sir_rows, axis=1)\n"
      "_sir_out = _sir_expanded + _sir_bias\n"
    )
    mutant, _ = mutate_kernel("missing_axis_restoration", source, (2, 2))

    self.assertEqual(
      mutant,
      "_sir_expanded = _sir_rows\n_sir_out = _sir_expanded + _sir_bias\n",
    )
    self.assertEqual(self._broadcast_shape((2, 1), (2, 2)), (2, 2))
    self.assertEqual(self._broadcast_shape((2,), (2, 2)), (2, 2))

  def test_missing_axis_mutation_changes_output_values(self):
    _, _, _, parameters, _ = self._axis_restoration_definition()
    x_values = parameters[0][1]
    bias_values = parameters[1][1]
    rows = [x_values[0] + x_values[1], x_values[2] + x_values[3]]
    correct = [
      rows[row] + bias_values[row * 2 + column]
      for row in range(2)
      for column in range(2)
    ]
    missing_axis = [
      rows[column] + bias_values[row * 2 + column]
      for row in range(2)
      for column in range(2)
    ]

    self.assertNotEqual(correct, missing_axis)
    self.assertEqual(correct, [13, 23, 37, 47])
    self.assertEqual(missing_axis, [13, 27, 33, 47])

  @staticmethod
  def _axis_restoration_definition():
    return next(
      definition
      for definition in build_tensor_regression_cases._composition_definitions()
      if definition[0] == "axis-restoration"
    )

  @staticmethod
  def _broadcast_shape(left, right):
    rank = max(len(left), len(right))
    left = (1,) * (rank - len(left)) + left
    right = (1,) * (rank - len(right)) + right
    return tuple(max(a, b) for a, b in zip(left, right))

  @staticmethod
  def _transpose_case_name(case):
    shape = "x".join(map(str, case["shape"]))
    permutation = "_".join(map(str, case["permutation"]))
    return f"transpose_{shape}_perm{permutation}"

  @staticmethod
  def _transpose_mutant_case():
    case_name = triton_mutants._MUTANT_CASES["wrong_transpose_permutation"]
    return next(
      case
      for case in build_tensor_regression_cases.transpose_sweep()
      if TritonMutantTests._transpose_case_name(case) == case_name
    )

  def test_wrong_output_stride_changes_row_pitch(self):
    source = "_sir_out_offsets = _sir_out_d0 * 4 + _sir_out_d1\n"
    mutant, _ = mutate_kernel("wrong_output_stride", source, (2, 4))
    self.assertEqual(mutant, "_sir_out_offsets = _sir_out_d0 * 3 + _sir_out_d1\n")

  def test_uniform_distribution_replaces_softmax(self):
    source = "_sir_out = tl.softmax(_sir_x, dim=1)\n"
    mutant, _ = mutate_kernel("uniform_distribution", source, (2, 4))
    self.assertEqual(
      mutant,
      "_sir_out = tl.full((2, 4), 0.25, tl.float32)\n",
    )

  def test_missing_fault_site_is_rejected(self):
    source = "_sir_out = tl.copy(_sir_x)\n"
    with self.assertRaisesRegex(ValueError, "softmax not found"):
      mutate_kernel("wrong_softmax_axis", source, (2, 4))
    self.assertEqual(source, "_sir_out = tl.copy(_sir_x)\n")

  @staticmethod
  def _case(root, *, root_kernel=None, root_environment=None, runs=None):
    root.mkdir(parents=True)
    (root / "case.json").write_text(
      json.dumps({"environment": {}, "launch": {}, "grid": [1, 1, 1]}),
      encoding="utf-8",
    )
    for name in ("program.sir", "inputs.bin", "expected.bin"):
      (root / name).write_bytes(name.encode())
    if root_kernel is not None:
      (root / "kernel.py").write_text(root_kernel, encoding="utf-8")
    if root_environment is not None:
      (root / "environment.json").write_text(
        json.dumps(root_environment), encoding="utf-8"
      )
    for mode, artifacts in (runs or {}).items():
      run_dir = root / "runs" / mode
      run_dir.mkdir(parents=True)
      if "kernel.py" in artifacts:
        (run_dir / "kernel.py").write_text(artifacts["kernel.py"], encoding="utf-8")
      if "environment.json" in artifacts:
        (run_dir / "environment.json").write_text(
          json.dumps(artifacts["environment.json"]), encoding="utf-8"
        )
    return root

  @staticmethod
  def _environment(revision):
    return {"triton_source_revision": revision, "package_lock_sha256": revision}

  def test_copy_case_uses_compiled_kernel_and_environment_artifacts(self):
    with tempfile.TemporaryDirectory() as directory:
      source = self._case(
        Path(directory) / "source",
        runs={
          "compiled": {
            "kernel.py": "compiled kernel\n",
            "environment.json": self._environment("compiled"),
          }
        },
      )

      triton_mutants._copy_case(source, Path(directory) / "mutant")

      mutant = Path(directory) / "mutant"
      self.assertEqual((mutant / "kernel.py").read_text(), "compiled kernel\n")
      environment = json.loads((mutant / "environment.json").read_text())
      self.assertEqual(environment["triton_source_revision"], "compiled")

  def test_copy_case_uses_compiled_environment_when_root_kernel_exists(self):
    with tempfile.TemporaryDirectory() as directory:
      source = self._case(
        Path(directory) / "source",
        root_kernel="root kernel\n",
        runs={"compiled": {"environment.json": self._environment("compiled")}},
      )

      triton_mutants._copy_case(source, Path(directory) / "mutant")

      environment = json.loads(
        (Path(directory) / "mutant" / "environment.json").read_text()
      )
      self.assertEqual(environment["triton_source_revision"], "compiled")

  def test_copy_case_prefers_kernel_from_selected_run(self):
    with tempfile.TemporaryDirectory() as directory:
      source = self._case(
        Path(directory) / "source",
        root_kernel="stale root kernel\n",
        root_environment=self._environment("root"),
        runs={
          "compiled": {
            "kernel.py": "compiled kernel\n",
            "environment.json": self._environment("compiled"),
          }
        },
      )

      triton_mutants._copy_case(source, Path(directory) / "mutant")

      self.assertEqual(
        (Path(directory) / "mutant" / "kernel.py").read_text(),
        "compiled kernel\n",
      )

  def test_copy_case_prefers_environment_from_selected_run(self):
    with tempfile.TemporaryDirectory() as directory:
      source = self._case(
        Path(directory) / "source",
        root_kernel="root kernel\n",
        root_environment=self._environment("stale-root"),
        runs={
          "compiled": {
            "kernel.py": "compiled kernel\n",
            "environment.json": self._environment("compiled"),
          }
        },
      )

      triton_mutants._copy_case(source, Path(directory) / "mutant")

      environment = json.loads(
        (Path(directory) / "mutant" / "environment.json").read_text()
      )
      self.assertEqual(environment["triton_source_revision"], "compiled")

  def test_copy_case_selects_artifacts_for_requested_mode(self):
    with tempfile.TemporaryDirectory() as directory:
      source = self._case(
        Path(directory) / "source",
        runs={
          "compiled": {
            "kernel.py": "compiled kernel\n",
            "environment.json": self._environment("compiled"),
          },
          "interpret": {
            "kernel.py": "interpreted kernel\n",
            "environment.json": self._environment("interpret"),
          },
        },
      )

      triton_mutants._copy_case(source, Path(directory) / "mutant", mode="interpret")

      mutant = Path(directory) / "mutant"
      self.assertEqual((mutant / "kernel.py").read_text(), "interpreted kernel\n")
      environment = json.loads((mutant / "environment.json").read_text())
      self.assertEqual(environment["triton_source_revision"], "interpret")


if __name__ == "__main__":
  unittest.main()
