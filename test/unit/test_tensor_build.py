"""Build-graph checks for the tensor generation profile."""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def makefile_target(target: str) -> tuple[str, str]:
  lines = (ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
  for index, line in enumerate(lines):
    if line.startswith(f"{target}:"):
      recipe = []
      for candidate in lines[index + 1 :]:
        if candidate.startswith("\t"):
          recipe.append(candidate)
        elif candidate:
          break
      return line, "\n".join(recipe)
  return "", ""


def dry_run_link_lines(target: str) -> dict[str, str]:
  solver = os.environ.get("REFRACTIR_TEST_SOLVER", "bitwuzla")
  result = subprocess.run(
    [
      "make",
      "--dry-run",
      "--always-make",
      target,
      f"SOLVER={solver}",
    ],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
  )
  links: dict[str, str] = {}
  for line in result.stdout.splitlines():
    match = re.search(r"\s-o (rylink|rytwin|rysmith)\s", line)
    if match:
      links[match.group(1)] = line
  return links


class TensorBuildTests(unittest.TestCase):
  def assert_test_group_has_classes(self, module_name: str, *class_names: str):
    module = importlib.import_module(f"test.unit.{module_name}")
    for class_name in class_names:
      with self.subTest(module=module_name, test_class=class_name):
        self.assertTrue(hasattr(module, class_name))

  def test_case_export_tests_are_grouped(self):
    self.assert_test_group_has_classes(
      "test_tensor_case",
      "TensorV3CaseTests",
      "TensorCaseV4Tests",
      "TensorV4CalibrationTests",
    )

  def test_exact_and_numerical_semantics_tests_are_grouped(self):
    self.assert_test_group_has_classes(
      "test_tensor_semantics", "TensorExactTests", "TensorNumericTests"
    )

  def test_coverage_and_directed_selection_tests_are_grouped(self):
    self.assert_test_group_has_classes(
      "test_tensor_selection", "TensorCoverageTests", "TensorDirectedTests"
    )

  def test_property_pilot_stages_are_grouped(self):
    self.assert_test_group_has_classes(
      "test_tensor_property_pilot",
      "WorkloadTests",
      "InputTests",
      "PropertyPilotCaseTests",
      "PropertyPilotTests",
    )

  def test_campaign_driver_tests_are_grouped_with_evaluation(self):
    self.assert_test_group_has_classes(
      "test_tensor_evaluation",
      "TensorEvaluationTests",
      "TensorCampaignDriverTests",
    )

  def test_historical_replay_is_grouped_with_regressions(self):
    self.assert_test_group_has_classes(
      "test_tensor_regressions",
      "HistoricalReplayTests",
      "TensorRegressionMatrixTests",
    )

  def test_coverage_target_runs_only_coverage_class(self):
    _, recipe = makefile_target("test-tensor-coverage")
    self.assertIn("test.unit.test_tensor_selection.TensorCoverageTests", recipe)

  def test_directed_target_runs_only_directed_class(self):
    declaration, recipe = makefile_target("test-tensor-directed")
    self.assertTrue(declaration.startswith("test-tensor-directed:"))
    self.assertIn("test.unit.test_tensor_selection.TensorDirectedTests", recipe)
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    phony = next(line for line in makefile.splitlines() if line.startswith(".PHONY:"))
    self.assertIn("test-tensor-directed", phony.split())

  def test_v2_profile_target_keeps_v3_case_scope(self):
    _, recipe = makefile_target("test-tensor-v2-profile")
    self.assertIn("test.unit.test_tensor_case.TensorV3CaseTests", recipe)
    self.assertNotIn("test.unit.test_tensor_case.TensorCaseV4Tests", recipe)

  def test_v4_generation_target_keeps_v4_case_scope(self):
    _, recipe = makefile_target("test-tensor-v4-generation")
    self.assertIn("test.unit.test_tensor_case.TensorCaseV4Tests", recipe)
    self.assertNotIn("test.unit.test_tensor_case.TensorV3CaseTests", recipe)

  def test_v4_target_keeps_calibration_case_scope(self):
    _, recipe = makefile_target("test-tensor-v4")
    self.assertIn("test.unit.test_tensor_case.TensorCaseV4Tests", recipe)
    self.assertIn("test.unit.test_tensor_case.TensorV4CalibrationTests", recipe)
    self.assertNotIn("test.unit.test_tensor_case.TensorV3CaseTests", recipe)

  def test_rylink_target_avoids_solver_bound_tensor_generator(self):
    link = dry_run_link_lines("rylink")["rylink"]
    self.assertNotIn("src/reify/tensor_gen.o", link)
    self.assertNotIn("src/reify/tensor_graph_gen.o", link)

  def test_rytwin_target_avoids_solver_bound_tensor_generator(self):
    link = dry_run_link_lines("rytwin")["rytwin"]
    self.assertNotIn("src/reify/tensor_gen.o", link)
    self.assertNotIn("src/reify/tensor_graph_gen.o", link)

  def test_test_unit_graph_keeps_generator_out_of_rylink(self):
    link = dry_run_link_lines("test-unit")["rylink"]
    self.assertNotIn("src/reify/tensor_gen.o", link)
    self.assertNotIn("src/reify/tensor_graph_gen.o", link)

  def test_reify_graph_keeps_generator_out_of_rylink(self):
    link = dry_run_link_lines("test-reify")["rylink"]
    self.assertNotIn("src/reify/tensor_gen.o", link)
    self.assertNotIn("src/reify/tensor_graph_gen.o", link)

  def test_all_graph_keeps_generator_out_of_solver_free_tools(self):
    links = dry_run_link_lines("all")
    self.assertNotIn("src/reify/tensor_gen.o", links["rylink"])
    self.assertNotIn("src/reify/tensor_gen.o", links["rytwin"])
    self.assertNotIn("src/reify/tensor_graph_gen.o", links["rylink"])
    self.assertNotIn("src/reify/tensor_graph_gen.o", links["rytwin"])

  def test_rysmith_owns_tensor_generator_and_solver_adapter(self):
    link = dry_run_link_lines("rysmith")["rysmith"]
    self.assertEqual(link.count("src/reify/tensor_gen.o"), 1)
    self.assertEqual(link.count("src/reify/tensor_graph_gen.o"), 1)
    self.assertIn("src/solver/bitwuzla_impl.o", link)

  def test_property_pilot_has_a_dedicated_make_target(self):
    declaration, _ = makefile_target("test-tensor-property-pilot")
    self.assertTrue(declaration.startswith("test-tensor-property-pilot:"))

  def test_property_pilot_target_is_phony(self):
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    phony = next(line for line in makefile.splitlines() if line.startswith(".PHONY:"))
    self.assertIn("test-tensor-property-pilot", phony.split())

  def test_property_pilot_requires_built_interpreter_and_solver(self):
    declaration, _ = makefile_target("test-tensor-property-pilot")
    self.assertIn("$(TARGET_INTERP)", declaration)
    self.assertIn("$(TARGET_SOLVER)", declaration)

  def test_property_pilot_uses_built_interpreter_and_solver(self):
    _, recipe = makefile_target("test-tensor-property-pilot")
    self.assertIn("REFRACTIR_SYMIRI=./$(TARGET_INTERP)", recipe)
    self.assertIn("REFRACTIR_SYMIR_SOLVE=./$(TARGET_SOLVER)", recipe)

  def test_property_pilot_runs_frozen_workload_tests(self):
    _, recipe = makefile_target("test-tensor-property-pilot")
    self.assertIn("test.unit.test_tensor_property_pilot", recipe)
    self.assertNotIn("test.unit.test_tensor_property_workload", recipe)

  def test_property_pilot_runs_selection_and_export_tests(self):
    _, recipe = makefile_target("test-tensor-property-pilot")
    self.assertIn("test.unit.test_tensor_property_pilot", recipe)
    self.assertNotIn("test.unit.test_tensor_property_inputs", recipe)
    self.assertNotIn("test.unit.test_tensor_property_cases", recipe)

  def test_tensor_gate_includes_property_pilot(self):
    _, recipe = makefile_target("test-tensor")
    self.assertIn("test-tensor-property-pilot", recipe)

  def test_fixed_study_target_runs_generated_selection_contracts(self):
    _, recipe = makefile_target("test-tensor-fixed-study")
    self.assertIn("test.unit.test_tensor_generated_selection_study", recipe)

  def test_generated_region_corpus_is_built_and_checked_by_tensor_gate(self):
    declaration, recipe = makefile_target("test-tensor-generated-corpus")
    self.assertTrue(declaration.startswith("test-tensor-generated-corpus:"))
    self.assertIn("tensor_graph_region_export", declaration)
    self.assertIn("test.unit.test_tensor_generated_region_corpus", recipe)
    _, tensor_recipe = makefile_target("test-tensor")
    self.assertIn("test-tensor-generated-corpus", tensor_recipe)


class TensorRegressionGateTests(unittest.TestCase):
  """Critical proposal and companion checks remain reachable from Make targets."""

  def test_a_missing_target_selects_no_tests(self):
    self.assertEqual(self.selected_tests("test-tensor-no-such-target"), set())

  @staticmethod
  def selected_tests(target):
    modules = set()
    seen = set()

    def visit(name):
      if name in seen:
        return
      seen.add(name)
      _, recipe = makefile_target(name)
      modules.update(re.findall(r"\btest\.unit\.[A-Za-z0-9_.]+", recipe))
      for child in re.findall(r"\btest-tensor[\w-]*", recipe):
        visit(child)

    def flatten(suite):
      for item in suite:
        if isinstance(item, unittest.TestSuite):
          yield from flatten(item)
        else:
          yield item.id()

    visit(target)
    return set(flatten(unittest.defaultTestLoader.loadTestsFromNames(sorted(modules))))

  def test_region_gate_checks_the_four_recorded_proposals(self):
    self.assertIn(
      "test.unit.test_tensor_input_proposals.TensorInputProposalTests."
      "test_the_four_required_proposals_are_derived_in_order",
      self.selected_tests("test-tensor-region-generation"),
    )

  def test_region_gate_checks_proposal_domain_refusals(self):
    self.assertIn(
      "test.unit.test_tensor_input_proposals.TensorInputProposalTests."
      "test_a_negative_domain_is_refused",
      self.selected_tests("test-tensor-region-generation"),
    )

  def test_exact_gate_checks_inherited_input_changes(self):
    self.assertIn(
      "test.unit.test_tensor_companion_derivation.CompanionDerivationTests."
      "test_a_changed_inherited_input_is_refused",
      self.selected_tests("test-tensor-exact"),
    )

  def test_exact_gate_checks_changed_loop_bounds(self):
    self.assertIn(
      "test.unit.test_tensor_companion_derivation.CompanionDerivationTests."
      "test_a_changed_loop_bound_is_refused",
      self.selected_tests("test-tensor-exact"),
    )

  def test_tensor_aggregate_checks_the_counted_certificate(self):
    self.assertIn(
      "test.unit.test_tensor_companion_derivation.CompanionDerivationTests."
      "test_a_wrong_certificate_is_refused",
      self.selected_tests("test-tensor"),
    )
