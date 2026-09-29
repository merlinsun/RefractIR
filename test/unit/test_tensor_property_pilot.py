"""Combined tensor test coverage: test_tensor_property_workload.py, test_tensor_property_inputs.py, test_tensor_property_cases.py, test_tensor_property_pilot.py."""

import collections
import importlib
import itertools
import json
import struct
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock
from unittest.mock import patch

from test.lib import run_tensor_property_pilot as pilot
from test.lib import tensor_case
from test.lib import tensor_property_inputs as input_selection
from test.lib.build_tensor_property_pilot_cases import (
  build_cases,
  select_pilot_witnesses,
)
from test.lib.tensor_property_inputs import (
  Attempt,
  select_input,
  solve_problem,
  validate_witness,
)
from test.lib.tensor_property_workload import (
  GraphNode,
  InputTensor,
  PilotProblem,
  broadcast_problem,
  freeze_workload,
  load_workload,
  render_query,
  render_runtime,
  save_workload,
  sum_problem,
  validate_problem,
)


# Consolidated from test_tensor_property_workload.py.
class WorkloadTests(unittest.TestCase):
  def test_deterministic_freeze(self):
    self.assertEqual(freeze_workload(281), freeze_workload(281))

  def test_workload_counts(self):
    problems = freeze_workload(281)
    self.assertEqual(len(problems), 120)
    self.assertEqual(len({problem.graph_id for problem in problems}), 60)
    self.assertEqual(
      collections.Counter(problem.symbol_count for problem in problems),
      {16: 40, 32: 40, 64: 40},
    )
    self.assertEqual(sum(problem.family == "nonlinear" for problem in problems), 30)

  def test_complete_target_constraints(self):
    problem = sum_problem((2, 2), (7, -3), (-8, 8))
    source = render_query(problem)
    self.assertEqual(source.count("require "), 2)
    self.assertIn("== 7;", source)
    self.assertIn("== -3;", source)

  def test_runtime_inputs_remain_parameters(self):
    problem = sum_problem((2, 2), (7, -3), (-8, 8))
    self.assertIn("sym %?x", render_query(problem))
    self.assertIn("%x: tensor [2, 2] i32", render_runtime(problem))
    self.assertNotIn("sym %?x", render_runtime(problem))

  def test_target_shape_rejected(self):
    with self.assertRaises(ValueError):
      sum_problem((2, 2), (7,), (-8, 8))

  def test_normalized_computation_matches_all_source_forms(self):
    from test.lib.tensor_property_workload import (
      normalized_computation,
      render_reference,
    )

    problem = sum_problem((2, 2), (7, -3), (-8, 8))
    self.assertEqual(
      normalized_computation(render_query(problem)),
      normalized_computation(render_runtime(problem)),
    )
    self.assertEqual(
      normalized_computation(render_runtime(problem)),
      normalized_computation(render_reference(problem)),
    )

  def test_target_selection_does_not_solve_or_replay(self):
    module = importlib.import_module("test.lib.tensor_property_workload")
    with (
      patch.object(module, "solve_problem", create=True, side_effect=AssertionError),
      patch.object(module, "run_reference", create=True, side_effect=AssertionError),
    ):
      first = module.select_targets((2,), 281)
      self.assertEqual(first, module.select_targets((2,), 281))

  def test_serialization_rejects_unknown_execution_field(self):
    module = importlib.import_module("test.lib.tensor_property_workload")
    value = module.problem_to_dict(sum_problem((2, 2), (7, -3), (-8, 8)))
    value["launch_grid"] = [2]
    with self.assertRaisesRegex(ValueError, "unknown=.*launch_grid"):
      module.problem_from_dict(value)

  def test_serialization_rejects_boolean_axis(self):
    module = importlib.import_module("test.lib.tensor_property_workload")
    value = module.problem_to_dict(sum_problem((2, 2), (7, -3), (-8, 8)))
    value["nodes"][0]["axis"] = True
    with self.assertRaisesRegex(ValueError, "axis must be an integer"):
      module.problem_from_dict(value)

  def test_development_set_is_disjoint_from_evaluation_graphs(self):
    module = importlib.import_module("test.lib.tensor_property_workload")
    with tempfile.TemporaryDirectory() as temporary:
      module.write_frozen_workloads(
        Path(temporary), development_seed=282, evaluation_seed=281
      )
      _, development = load_workload(Path(temporary) / "development.json")
      _, evaluation = load_workload(Path(temporary) / "workload.json")

    self.assertEqual(len(development), 12)
    self.assertEqual(
      development,
      module.freeze_development_workload(282, evaluation_workload=evaluation),
    )
    self.assertEqual({problem.family for problem in development}, set(module._FAMILIES))
    self.assertEqual(
      collections.Counter(problem.symbol_count for problem in development),
      {16: 4, 32: 4, 64: 4},
    )
    development_graphs = {module._graph_signature(problem) for problem in development}
    evaluation_graphs = {module._graph_signature(problem) for problem in evaluation}
    self.assertFalse(development_graphs & evaluation_graphs)


# Consolidated from test_tensor_property_inputs.py.
class InputTests(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)

  def test_feasible_rows(self):
    problem = sum_problem((2, 2), (7, -3), (-8, 8))
    result = solve_problem(
      problem, seed=1, budget_s=5, artifact_dir=self.directory.name
    )
    self.assertEqual(result.status, "validated")
    self.assertEqual(result.output_bytes, struct.pack("<2i", 7, -3))

  def test_singleton_tensor_symbol(self):
    problem = sum_problem((1,), (1,), (-1, 1))
    witness = struct.pack("<i", 1)
    replay = validate_witness(problem, witness, artifact_dir=self.directory.name)
    self.assertEqual(replay.status, "validated")
    self.assertEqual(replay.output_bytes, struct.pack("<i", 1))

  def test_singleton_tensor_symbol_solves_exact_target(self):
    problem = sum_problem((1,), (1,), (-1, 1))
    result = solve_problem(
      problem, seed=19, budget_s=5, artifact_dir=self.directory.name
    )
    self.assertEqual(result.status, "validated")
    self.assertEqual(result.input_bytes, struct.pack("<i", 1))
    self.assertEqual(result.output_bytes, struct.pack("<i", 1))

  def test_impossible_bounded_sum(self):
    problem = sum_problem((2,), (3,), (-1, 1))
    result = solve_problem(
      problem, seed=1, budget_s=5, artifact_dir=self.directory.name
    )
    self.assertEqual(result.status, "unsat")

  def test_contradictory_broadcast_aliases(self):
    problem = broadcast_problem((0, 1), (-1, 1))
    result = solve_problem(
      problem, seed=1, budget_s=5, artifact_dir=self.directory.name
    )
    self.assertEqual(result.status, "unsat")

  def test_missing_input_bytes(self):
    problem = sum_problem((2, 2), (7, -3), (-8, 8))
    result = validate_witness(
      problem,
      struct.pack("<3i", 3, 4, -1),
      artifact_dir=self.directory.name,
    )
    self.assertEqual(result.status, "validation_failure")
    self.assertEqual(result.reason, "input_length")

  def test_non_target_witness(self):
    problem = sum_problem((2, 2), (7, -3), (-8, 8))
    result = validate_witness(
      problem,
      struct.pack("<4i", 0, 0, 0, 0),
      artifact_dir=self.directory.name,
    )
    self.assertEqual(result.status, "validation_failure")
    self.assertEqual(result.reason, "target_mismatch")

  def test_out_of_domain_model_is_rejected(self):
    problem = sum_problem((1,), (0,), (-1, 1))
    model_path = Path(self.directory.name) / "bad-model.json"
    model_path.write_text(json.dumps({"@kernel": {"%?x": [2]}}), encoding="utf-8")
    input_bytes, reason = input_selection._decode_model(problem, model_path)
    self.assertIsNone(input_bytes)
    self.assertEqual(reason, "input_domain")

  def test_nonzero_solver_exit_is_not_assumed_unsat(self):
    problem = sum_problem((1,), (0,), (-1, 1))
    failure = subprocess.CompletedProcess(
      args=["symirsolve"], returncode=1, stdout="", stderr="solver crashed"
    )
    with mock.patch.object(input_selection.subprocess, "run", return_value=failure):
      result = solve_problem(
        problem, seed=29, budget_s=5, artifact_dir=self.directory.name
      )
    self.assertEqual(result.status, "invalid_problem")
    self.assertNotEqual(result.status, "unsat")

  def test_in_range_final_sum_does_not_hide_partial_overflow(self):
    problem = sum_problem((4,), (2**31 - 1,), (-(2**31), 2**31 - 1))
    witness = struct.pack("<4i", 2**31 - 1, 1, -1, 0)
    result = validate_witness(problem, witness, artifact_dir=self.directory.name)
    self.assertEqual(result.status, "validation_failure")
    self.assertEqual(result.reason, "reference_ub")

  def test_small_domain_feasibility_matches_exhaustive_enumeration(self):
    transpose = PilotProblem(
      graph_id="exhaustive_transpose",
      problem_id="exhaustive_transpose_target",
      family="layout_sum",
      symbol_count=4,
      inputs=(InputTensor("x", (2, 2)),),
      output_shape=(2, 2),
      domain=(-1, 1),
      target=(-1, 0, 1, -1),
      target_pattern="test",
      nodes=(GraphNode("n0", "transpose", ("x",), (2, 2), permutation=(1, 0)),),
      output_id="n0",
    )
    product_sum = PilotProblem(
      graph_id="exhaustive_product_sum",
      problem_id="exhaustive_product_sum_target",
      family="nonlinear",
      symbol_count=4,
      inputs=(InputTensor("x", (2,)), InputTensor("y", (2,))),
      output_shape=(),
      domain=(-1, 1),
      target=(1,),
      target_pattern="test",
      nodes=(
        GraphNode("n0", "mul", ("x", "y"), (2,)),
        GraphNode("n1", "sum", ("n0",), (), axis=0),
      ),
      output_id="n1",
    )
    cases = (
      (
        sum_problem((2,), (0,), (-1, 1)),
        lambda assignment: (assignment[0] + assignment[1],),
        ((-2,), (0,), (2,), (3,)),
      ),
      (
        broadcast_problem((-1, -1), (-1, 1)),
        lambda assignment: (assignment[0], assignment[0]),
        ((-1, -1), (0, 0), (1, 1), (0, 1)),
      ),
      (
        transpose,
        lambda assignment: (assignment[0], assignment[2], assignment[1], assignment[3]),
        ((-1, 0, 1, -1), (1, -1, 0, 1), (2, 0, 1, -1)),
      ),
      (
        product_sum,
        lambda assignment: (
          assignment[0] * assignment[2] + assignment[1] * assignment[3],
        ),
        ((-2,), (0,), (2,), (3,)),
      ),
    )

    for base, evaluate, targets in cases:
      with self.subTest(problem=base.problem_id):
        validate_problem(base)
        domains = itertools.product(
          range(base.domain[0], base.domain[1] + 1), repeat=base.symbol_count
        )
        feasible = {evaluate(assignment) for assignment in domains}
        for index, target in enumerate(targets):
          problem = replace(
            base, problem_id=f"{base.graph_id}_target_{index}", target=target
          )
          result = solve_problem(
            problem,
            seed=100 + index,
            budget_s=5,
            artifact_dir=self.directory.name,
          )
          expected = "sat" if target in feasible else "unsat"
          actual = "sat" if result.status == "validated" else result.status
          self.assertEqual(actual, expected, result.reason)


class SelectionTests(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)

  def test_construction_solves_bounded_disjoint_sums(self):
    problem = sum_problem((2, 2), (7, -3), (-8, 8))
    result = select_input(
      problem,
      method="construction",
      seed=5,
      budget_s=5,
      artifact_dir=self.directory.name,
    )
    self.assertEqual(result.status, "validated")
    self.assertTrue(result.supported)
    self.assertEqual(result.output_bytes, struct.pack("<2i", 7, -3))

  def test_construction_maps_transposed_reduction_groups(self):
    problem = PilotProblem(
      graph_id="constructor_transpose",
      problem_id="constructor_transpose_target",
      family="layout_sum",
      symbol_count=4,
      inputs=(InputTensor("x", (2, 2)),),
      output_shape=(2,),
      domain=(-8, 8),
      target=(7, -3),
      target_pattern="test",
      nodes=(
        GraphNode("n0", "transpose", ("x",), (2, 2), permutation=(1, 0)),
        GraphNode("n1", "sum", ("n0",), (2,), axis=0),
      ),
      output_id="n1",
    )
    result = select_input(
      problem,
      method="construction",
      seed=23,
      budget_s=5,
      artifact_dir=self.directory.name,
    )
    self.assertEqual(result.status, "validated")
    self.assertTrue(result.supported)
    self.assertEqual(result.output_bytes, struct.pack("<2i", 7, -3))

  def test_unsupported_construction_is_distinct(self):
    problem = next(item for item in freeze_workload(281) if item.family == "nonlinear")
    result = select_input(
      problem,
      method="construction",
      seed=5,
      budget_s=5,
      artifact_dir=self.directory.name,
    )
    self.assertEqual(result.status, "unsupported_baseline")
    self.assertFalse(result.supported)

  def test_ordinary_exhaustion_is_not_unsat(self):
    problem = sum_problem((2,), (3,), (-1, 1))
    clock = _FakeClock()

    def reject_every_input(*args, **kwargs):
      clock.advance(0.25)
      return Attempt(status="validation_failure", reason="target_mismatch")

    with (
      mock.patch.object(input_selection, "_monotonic", clock),
      mock.patch.object(input_selection, "validate_witness", reject_every_input),
    ):
      result = select_input(
        problem,
        method="ordinary",
        seed=9,
        budget_s=1,
        artifact_dir=self.directory.name,
      )
    self.assertEqual(result.status, "budget_exhausted")
    self.assertGreater(result.attempt_count, 0)

  def test_boundary_proposals_preserve_declared_domain(self):
    problem = sum_problem((16,), (0,), (-8, 8))
    clock = _FakeClock()
    observed = []
    boundary_values = {-8, -4, -2, -1, 0, 1, 2, 4, 8}

    def reject_every_input(_problem, input_bytes, **kwargs):
      observed.append(struct.unpack("<16i", input_bytes))
      clock.advance(0.001)
      return Attempt(status="validation_failure", reason="target_mismatch")

    with (
      mock.patch.object(input_selection, "_monotonic", clock),
      mock.patch.object(input_selection, "validate_witness", reject_every_input),
    ):
      result = select_input(
        problem,
        method="boundary",
        seed=13,
        budget_s=0.2,
        artifact_dir=self.directory.name,
      )
    self.assertEqual(result.status, "budget_exhausted")
    self.assertTrue(observed)
    self.assertTrue(
      all(-8 <= value <= 8 for candidate in observed for value in candidate)
    )
    self.assertTrue(
      any(any(value not in boundary_values for value in row) for row in observed)
    )

  def test_boundary_sampling_mixes_uniform_and_boundary_per_element(self):
    problem = sum_problem((2,), (999,), (-8, 8))
    clock = _FakeClock()
    observed = []
    rng = _SequenceRandom(random_values=(0.1, 0.9), integers=(7,), choices=(-4,))

    def reject_after_capture(_problem, input_bytes, **kwargs):
      observed.append(struct.unpack("<2i", input_bytes))
      clock.advance(2.0)
      return Attempt(status="validation_failure", reason="target_mismatch")

    with (
      mock.patch.object(input_selection, "_monotonic", clock),
      mock.patch.object(input_selection, "validate_witness", reject_after_capture),
      mock.patch.object(input_selection.random, "Random", return_value=rng),
    ):
      result = select_input(
        problem,
        method="boundary",
        seed=37,
        budget_s=1,
        artifact_dir=self.directory.name,
      )

    self.assertEqual(result.status, "budget_exhausted")
    self.assertEqual(observed, [(7, -4)])
    self.assertEqual(rng.random_calls, 2)

  def test_boundary_sampling_chooses_source_independently_for_each_value(self):
    problem = sum_problem((4,), (999,), (-8, 8))
    clock = _FakeClock()
    observed = []
    rng = _SequenceRandom(
      random_values=(0.1, 0.9, 0.1, 0.9),
      integers=(7, 6),
      choices=(-4, -2),
    )

    def reject_after_capture(_problem, input_bytes, **kwargs):
      observed.append(struct.unpack("<4i", input_bytes))
      clock.advance(2.0)
      return Attempt(status="validation_failure", reason="target_mismatch")

    with (
      mock.patch.object(input_selection, "_monotonic", clock),
      mock.patch.object(input_selection, "validate_witness", reject_after_capture),
      mock.patch.object(input_selection.random, "Random", return_value=rng),
    ):
      result = select_input(
        problem,
        method="boundary",
        seed=41,
        budget_s=1,
        artifact_dir=self.directory.name,
      )

    self.assertEqual(result.status, "budget_exhausted")
    self.assertEqual(observed, [(7, -4, 6, -2)])
    self.assertEqual((rng.random_calls, rng.randint_calls, rng.choice_calls), (4, 2, 2))

  def test_late_validated_witness_is_not_within_budget_success(self):
    problem = sum_problem((2,), (0,), (-1, 1))
    clock = _FakeClock()
    proposed = []

    def late_success(_problem, input_bytes, **kwargs):
      clock.advance(1.01)
      proposed.append(input_bytes)
      return Attempt(
        status="validated",
        input_bytes=input_bytes,
        output_bytes=struct.pack("<i", 0),
      )

    with (
      mock.patch.object(input_selection, "_monotonic", clock),
      mock.patch.object(input_selection, "validate_witness", late_success),
    ):
      result = select_input(
        problem,
        method="ordinary",
        seed=17,
        budget_s=1,
        artifact_dir=self.directory.name,
      )
    self.assertEqual(result.status, "budget_exhausted")
    self.assertEqual(result.input_bytes, proposed[0])
    self.assertTrue(result.witness_validated)


class _FakeClock:
  def __init__(self):
    self.now = 0.0

  def __call__(self):
    return self.now

  def advance(self, seconds):
    self.now += seconds


class _SequenceRandom:
  def __init__(self, *, random_values, integers, choices):
    self.random_values = list(random_values)
    self.integers = list(integers)
    self.choices = list(choices)
    self.random_calls = 0
    self.randint_calls = 0
    self.choice_calls = 0

  def random(self):
    value = self.random_values[self.random_calls]
    self.random_calls += 1
    return value

  def randint(self, _lower, _upper):
    value = self.integers[min(self.randint_calls, len(self.integers) - 1)]
    self.randint_calls += 1
    return value

  def choice(self, _values):
    value = self.choices[min(self.choice_calls, len(self.choices) - 1)]
    self.choice_calls += 1
    return value


# Consolidated from test_tensor_property_cases.py.
class PropertyPilotCaseTests(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)
    self.problem = sum_problem((2, 2), (7, -3), (-8, 8))
    self.workload = self.root / "workload.json"
    save_workload(self.workload, [self.problem], seed=31)
    self.attempts = self.root / "attempts.jsonl"
    self.input_bytes = struct.pack("<4i", 3, 4, -1, -2)
    self.output_bytes = struct.pack("<2i", 7, -3)
    self.signature = {
      "entry": "kernel",
      "inputs": [{"name": "x", "dtype": "i32", "shape": [2, 2], "byte_length": 16}],
      "output": {
        "name": "return",
        "dtype": "i32",
        "shape": [2],
        "byte_length": 8,
      },
    }

  def _attempt_record(self, method, input_bytes=None):
    return {
      "problem_id": self.problem.problem_id,
      "graph_id": self.problem.graph_id,
      "family": self.problem.family,
      "symbol_count": self.problem.symbol_count,
      "method": method,
      "trial": 1,
      "schedule_index": len(method),
      "status": "validated",
      "input_hex": (input_bytes or self.input_bytes).hex(),
      "output_hex": self.output_bytes.hex(),
    }

  def _write_attempts(self, rows):
    self.attempts.write_text(
      "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

  def test_selection_respects_per_method_limit_and_strata(self):
    rows = []
    for method in ("ordinary", "boundary", "construction", "smt"):
      rows.append(self._attempt_record(method))
    selected = select_pilot_witnesses(rows, [self.problem], limit_per_method=1, seed=5)
    self.assertEqual(set(selected), {"ordinary", "boundary", "construction", "smt"})
    self.assertTrue(all(len(items) == 1 for items in selected.values()))

  def test_case_manifest_records_selection_method_and_property(self):
    self._write_attempts([self._attempt_record("ordinary")])
    output = self.root / "cases"
    with (
      mock.patch(
        "test.lib.build_tensor_property_pilot_cases.validate_witness",
        return_value=Attempt(
          status="validated",
          input_bytes=self.input_bytes,
          output_bytes=self.output_bytes,
        ),
      ),
      mock.patch(
        "test.lib.build_tensor_property_pilot_cases._describe",
        return_value=self.signature,
      ),
      mock.patch.object(tensor_case, "_describe", return_value=self.signature),
    ):
      summary = build_cases(self.workload, self.attempts, output, seed=19)
      case = tensor_case.load_case(output / "case_0000", symiri="unused")
    self.assertEqual(summary["case_count"], 1)
    self.assertEqual(case.manifest["evidence"]["kind"], "construction")
    self.assertEqual(case.manifest["evidence"]["status"], "construction_checked")
    self.assertIn("ordinary", case.manifest["property"]["source"])
    self.assertEqual(case.inputs, self.input_bytes)
    self.assertEqual(case.expected, self.output_bytes)

  def test_smt_case_uses_exact_smt_evidence(self):
    self._write_attempts([self._attempt_record("smt")])
    output = self.root / "cases"
    with (
      mock.patch(
        "test.lib.build_tensor_property_pilot_cases.validate_witness",
        return_value=Attempt(
          status="validated",
          input_bytes=self.input_bytes,
          output_bytes=self.output_bytes,
        ),
      ),
      mock.patch(
        "test.lib.build_tensor_property_pilot_cases._describe",
        return_value=self.signature,
      ),
      mock.patch.object(tensor_case, "_describe", return_value=self.signature),
    ):
      build_cases(self.workload, self.attempts, output, seed=19)
      case = tensor_case.load_case(output / "case_0000", symiri="unused")
    self.assertEqual(case.manifest["evidence"]["kind"], "exact_smt")
    self.assertEqual(case.manifest["evidence"]["status"], "exact_smt_checked")

  def test_non_little_endian_or_out_of_domain_witness_is_rejected(self):
    self._write_attempts(
      [self._attempt_record("ordinary", struct.pack(">4i", 3, 4, -1, -2))]
    )
    output = self.root / "cases"
    with (
      mock.patch(
        "test.lib.build_tensor_property_pilot_cases.validate_witness",
        return_value=Attempt(status="validation_failure", reason="input_domain"),
      ),
      mock.patch(
        "test.lib.build_tensor_property_pilot_cases._describe",
        return_value=self.signature,
      ),
    ):
      with self.assertRaisesRegex(ValueError, "witness validation failed"):
        build_cases(self.workload, self.attempts, output, seed=19)

  def test_digest_mutation_is_rejected_by_strict_loader(self):
    self._write_attempts([self._attempt_record("ordinary")])
    output = self.root / "cases"
    with (
      mock.patch(
        "test.lib.build_tensor_property_pilot_cases.validate_witness",
        return_value=Attempt(
          status="validated",
          input_bytes=self.input_bytes,
          output_bytes=self.output_bytes,
        ),
      ),
      mock.patch(
        "test.lib.build_tensor_property_pilot_cases._describe",
        return_value=self.signature,
      ),
      mock.patch.object(tensor_case, "_describe", return_value=self.signature),
    ):
      build_cases(self.workload, self.attempts, output, seed=19)
    case_dir = output / "case_0000"
    with (case_dir / "inputs.bin").open("ab") as changed:
      changed.write(b"\x00")
    with mock.patch.object(tensor_case, "_describe", return_value=self.signature):
      with self.assertRaisesRegex(tensor_case.CaseValidationError, "byte length"):
        tensor_case.load_case(case_dir, symiri="unused")


if __name__ == "__main__":
  unittest.main()


# Consolidated from test_tensor_property_pilot.py.
class PropertyPilotTests(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)
    self.workload_path = self.root / "workload.json"
    self.output_path = self.root / "run-001"
    self.problem = sum_problem((2, 2), (7, -3), (-8, 8))
    save_workload(self.workload_path, [self.problem], seed=31)

  def _run(self, selector, *, trials=1, archive_writer=None):
    with mock.patch.object(pilot, "select_input", side_effect=selector):
      return pilot.run_campaign(
        self.workload_path,
        self.output_path,
        trials=trials,
        budget_seconds=0.25,
        seed=17,
        workers=1,
        archive_writer=archive_writer or (lambda _path: "created"),
      )

  def test_every_scheduled_method_problem_trial_has_terminal_record(self):
    def accepted(_problem, *, method, seed, **kwargs):
      return Attempt(
        status="validated",
        input_bytes=struct.pack("<4i", seed % 3, 0, 0, 0),
        output_bytes=struct.pack("<2i", 7, -3),
      )

    summary = self._run(accepted, trials=2)
    records = [
      json.loads(line)
      for line in (self.output_path / "attempts.jsonl").read_text().splitlines()
    ]
    keys = {(row["problem_id"], row["trial"], row["method"]) for row in records}
    self.assertEqual(len(records), 8)
    self.assertEqual(len(keys), 8)
    self.assertEqual(summary["terminal_attempts"], summary["scheduled_attempts"])

  def test_unsupported_constructor_is_excluded_from_supported_denominator(self):
    def results(_problem, *, method, **kwargs):
      if method == "construction":
        return Attempt(status="unsupported_baseline", supported=False)
      return Attempt(status="validated", input_bytes=b"witness", output_bytes=b"target")

    summary = self._run(results)
    construction = summary["methods"]["construction"]
    self.assertEqual(construction["status_counts"]["unsupported_baseline"], 1)
    self.assertEqual(construction["constructor_supported_attempts"], 0)
    self.assertEqual(construction["constructor_validated_supported"], 0)

  def test_changed_workload_bytes_invalidate_resume(self):
    def accepted(_problem, *, method, **kwargs):
      return Attempt(status="budget_exhausted")

    self._run(accepted)
    self.workload_path.write_bytes(self.workload_path.read_bytes() + b"\n")
    with mock.patch.object(pilot, "select_input") as selector:
      with self.assertRaisesRegex(ValueError, "workload hash"):
        pilot.run_campaign(
          self.workload_path,
          self.output_path,
          trials=1,
          budget_seconds=0.25,
          seed=17,
          workers=1,
          archive_writer=lambda _path: "created",
        )
    selector.assert_not_called()

  def test_methods_get_independent_witnesses_and_seed_streams(self):
    observed = []

    def independent(_problem, *, method, seed, **kwargs):
      observed.append((method, seed))
      value = pilot.METHODS.index(method) + 1
      return Attempt(
        status="validated",
        input_bytes=struct.pack("<4i", value, value, value, value),
        output_bytes=struct.pack("<2i", 7, -3),
      )

    self._run(independent)
    self.assertEqual(len({seed for _, seed in observed}), 4)
    records = [
      json.loads(line)
      for line in (self.output_path / "attempts.jsonl").read_text().splitlines()
    ]
    witnesses = {row["method"]: row["input_hex"] for row in records}
    self.assertEqual(len(set(witnesses.values())), 4)

  def test_archive_exception_preserves_completed_selection_results(self):
    def accepted(_problem, *, method, seed, **kwargs):
      return Attempt(
        status="validated",
        input_bytes=struct.pack("<4i", 1, 2, 3, 4),
        output_bytes=struct.pack("<2i", 7, -3),
      )

    def failed_archive(path):
      self.assertTrue((path / "attempts.jsonl").is_file())
      self.assertTrue((path / "selection-summary.json").is_file())
      raise OSError("simulated archive write failure")

    summary = self._run(accepted, archive_writer=failed_archive)
    self.assertEqual(summary["terminal_attempts"], 4)
    self.assertEqual(
      len((self.output_path / "attempts.jsonl").read_text().splitlines()), 4
    )
    saved = json.loads((self.output_path / "selection-summary.json").read_text())
    self.assertEqual(saved["terminal_attempts"], 4)
    archive_status = json.loads((self.output_path / "archive-status.json").read_text())
    self.assertEqual(archive_status["status"], "failed")

  def test_archive_is_rebuilt_when_durable_run_contents_change(self):
    def accepted(_problem, *, method, seed, **kwargs):
      return Attempt(
        status="validated",
        input_bytes=struct.pack("<4i", 1, 2, 3, 4),
        output_bytes=struct.pack("<2i", 7, -3),
      )

    self._run(accepted, trials=1)
    pilot._write_archive(self.output_path)
    first_archive = self.output_path.with_suffix(".tar.gz")
    self.assertTrue(first_archive.is_file())
    report = self.output_path / "analysis.txt"
    report.write_text("reconciled\n", encoding="utf-8")
    result = pilot._write_archive(self.output_path)
    self.assertEqual(result["status"], "created_and_verified")
    with pilot.tarfile.open(first_archive, "r:gz") as archive:
      member = archive.extractfile(f"{self.output_path.name}/analysis.txt")
      self.assertIsNotNone(member)
      self.assertEqual(member.read(), b"reconciled\n")

  def test_contradictory_unsat_and_witness_blocks_comparative_conclusions(self):
    def contradictory(_problem, *, method, **kwargs):
      if method == "ordinary":
        return Attempt(status="unsat")
      if method == "smt":
        return Attempt(
          status="validated",
          input_bytes=struct.pack("<4i", 1, 2, 3, 4),
          output_bytes=struct.pack("<2i", 7, -3),
        )
      return Attempt(status="budget_exhausted")

    summary = self._run(contradictory)
    self.assertEqual(summary["status"], "contradictory_evidence")
    self.assertFalse(summary["comparative_conclusion_valid"])
    self.assertEqual(len(summary["incidents"]), 1)

  def test_late_validated_witness_still_contradicts_an_unsat_claim(self):
    rows = {
      (self.problem.problem_id, 1, "ordinary"): {
        "problem_id": self.problem.problem_id,
        "trial": 1,
        "method": "ordinary",
        "status": "unsat",
      },
      (self.problem.problem_id, 1, "boundary"): {
        "problem_id": self.problem.problem_id,
        "trial": 1,
        "method": "boundary",
        "status": "budget_exhausted",
        "witness_validated": True,
        "input_hex": "01000000020000000300000004000000",
        "output_hex": "07000000fdffffff",
      },
    }
    summary = pilot._summarize(
      [
        {"problem_id": self.problem.problem_id, "method": "ordinary"},
        {"problem_id": self.problem.problem_id, "method": "boundary"},
      ],
      rows,
      [self.problem],
      trials=1,
      budget_seconds=1,
      seed=43,
    )

    self.assertEqual(summary["status"], "contradictory_evidence")
    self.assertFalse(summary["comparative_conclusion_valid"])
    self.assertEqual(
      summary["feasibility_by_problem"][self.problem.problem_id],
      "confirmed_feasible",
    )

  def test_attempt_record_persists_late_witness_evidence(self):
    attempt = Attempt(
      status="budget_exhausted",
      input_bytes=struct.pack("<4i", 1, 2, 3, 4),
      output_bytes=struct.pack("<2i", 7, -3),
      witness_validated=True,
    )
    record = pilot._attempt_record(
      {
        "schedule_index": 0,
        "problem_id": self.problem.problem_id,
        "graph_id": self.problem.graph_id,
        "trial": 1,
        "method": "boundary",
        "method_order": 0,
        "seed": 47,
      },
      self.problem,
      attempt,
      identity_sha256="identity",
      driver_elapsed_s=1.1,
    )

    self.assertEqual(record["status"], "budget_exhausted")
    self.assertIs(record["witness_validated"], True)

  def test_witness_bytes_without_validation_evidence_do_not_confirm_feasibility(self):
    rows = {
      (self.problem.problem_id, 1, "ordinary"): {
        "problem_id": self.problem.problem_id,
        "trial": 1,
        "method": "ordinary",
        "status": "unsat",
      },
      (self.problem.problem_id, 1, "boundary"): {
        "problem_id": self.problem.problem_id,
        "trial": 1,
        "method": "boundary",
        "status": "budget_exhausted",
        "input_hex": "01000000020000000300000004000000",
        "output_hex": "07000000fdffffff",
      },
    }
    summary = pilot._summarize(
      [
        {"problem_id": self.problem.problem_id, "method": "ordinary"},
        {"problem_id": self.problem.problem_id, "method": "boundary"},
      ],
      rows,
      [self.problem],
      trials=1,
      budget_seconds=1,
      seed=53,
    )

    self.assertEqual(summary["status"], "complete")
    self.assertEqual(
      summary["feasibility_by_problem"][self.problem.problem_id],
      "solver_proved_infeasible",
    )


if __name__ == "__main__":
  unittest.main()
