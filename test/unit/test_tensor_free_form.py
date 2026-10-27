"""Executable contract tests for free-form regions and their counted-loop assembly."""

import json
import re
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from test.lib import tensor_free_form_case, tensor_free_form_inputs, tensor_target_case
from test.unit.tensor_support import SYMIRI, check_source, describe_entry

FREE_FORM_API = Path(
  os.environ.get("REFRACTIR_TENSOR_FREE_FORM_API", "build/bin/tensor_free_form_api")
)
FREE_FORM_EXPORT = Path(
  os.environ.get("REFRACTIR_TENSOR_FREE_FORM_EXPORT", "build/bin/tensor_free_form_export")
)


class FreeFormContractTests(unittest.TestCase):
  """The builder and the assembler each keep their own stated contract."""

  def run_contract(self, name):
    self.assertTrue(
      FREE_FORM_API.is_file(), f"free-form API binary not found: {FREE_FORM_API}"
    )
    result = subprocess.run(
      [str(FREE_FORM_API), name], text=True, capture_output=True, check=False, timeout=30
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def render(self, name):
    self.assertTrue(
      FREE_FORM_API.is_file(), f"free-form API binary not found: {FREE_FORM_API}"
    )
    result = subprocess.run(
      [str(FREE_FORM_API), name], text=True, capture_output=True, check=False, timeout=30
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return result.stdout

  def test_two_typed_results_of_requested_shapes(self):
    self.run_contract("two-typed-results")

  def test_inputs_may_be_invariants_not_only_states(self):
    self.run_contract("inputs-not-only-states")

  def test_one_consumer_counts_a_value_named_twice_once(self):
    self.run_contract("add-of-one-value-is-one-consumer")

  def test_pruning_omits_dead_values_and_counts_live_consumers(self):
    self.run_contract("pruning-omits-dead-values")

  def test_same_seed_replays_and_impossible_limits_refuse(self):
    self.run_contract("deterministic-and-impossible")

  def test_three_state_region_assembles_into_valid_sir(self):
    self.run_contract("three-state-assembly")

  def test_additive_states_are_bounded_over_their_trips(self):
    self.run_contract("magnitude-three-state-trips")

  def test_a_compounding_product_leaves_the_exact_range(self):
    self.run_contract("magnitude-product-compounds")

  def test_a_sum_scales_its_bound_and_extrema_keep_it(self):
    self.run_contract("magnitude-reductions-and-shapes")

  def test_a_window_fill_bounds_its_tile(self):
    self.run_contract("magnitude-window-fill")

  def test_only_live_sums_are_order_sensitive(self):
    self.run_contract("order-sensitive-values-are-live-sums")

  def test_the_sum_bound_tracks_sums_alone(self):
    self.run_contract("magnitudes-track-sums")

  def test_the_reduction_contract_bounds_sums_and_overflow(self):
    self.run_contract("reduction-contract-bounds")

  def test_most_live_roots_keep_more_nodes(self):
    self.run_contract("most-live-roots-keep-more-nodes")

  def test_deepest_roots_are_the_default(self):
    self.run_contract("deepest-roots-are-the-default")

  def test_state_and_invariant_initialization_is_respected(self):
    self.run_contract("invariant-initialization")

  def test_simultaneous_updates_read_saved_values(self):
    self.run_contract("simultaneous-updates")

  def test_incompatible_updates_and_unknown_terms_are_rejected(self):
    self.run_contract("rejected-updates")

  def test_canonical_adapter_emits_the_canonical_loop_byte_for_byte(self):
    self.run_contract("canonical-bytes")

  def test_policy_names_round_trip(self):
    self.run_contract("policy-names")

  def test_selection_scores_bias_without_excluding_actions(self):
    self.run_contract("selection-bounds")

  def test_unguided_can_select_outside_a_biased_rank(self):
    self.run_contract("policy-unguided-ignores-rank")

  def test_a_depth_bias_changes_the_selected_action(self):
    self.run_contract("policy-depth-changes-choice")

  def test_a_sharing_bias_selects_a_second_consumer(self):
    self.run_contract("policy-sharing-selects-a-consumer")

  def test_a_reduction_bias_selects_a_reduction(self):
    self.run_contract("policy-reduction-selects-a-reduction")

  def test_replay_is_separate_from_the_distribution(self):
    self.run_contract("replay-vs-distribution")

  def test_features_stay_off_unless_the_request_enables_them(self):
    self.run_contract("features-off-keep-the-core")

  def test_enabled_features_appear_in_generated_bodies(self):
    self.run_contract("features-enable-transpose-and-expand")

  def test_features_respect_the_rank_limit(self):
    self.run_contract("features-respect-the-rank-limit")

  def test_a_feature_request_replays_exactly(self):
    self.run_contract("features-replay-exactly")

  def test_extrema_binary_operations_broadcast(self):
    self.run_contract("extrema-binary-types")

  def test_extrema_reductions_drop_their_axis(self):
    self.run_contract("extrema-reduce-drop-shape")

  def test_extrema_reductions_keep_their_axis(self):
    self.run_contract("extrema-reduce-keep-shape")

  def test_extrema_reject_an_axis_outside_the_rank(self):
    self.run_contract("extrema-reject-axis")

  def test_every_extrema_form_renders_its_sir_spelling(self):
    self.run_contract("extrema-render-round-trip")

  def test_a_rank_one_kept_reduction_stays_rank_one(self):
    self.run_contract("extrema-rank-one-keep-is-rank-one")

  def test_extrema_actions_are_enumerated(self):
    self.run_contract("extrema-action-enumerated")

  def test_extrema_reductions_carry_the_reduction_bias(self):
    self.run_contract("extrema-action-weighted")

  def test_extrema_nodes_count_as_reductions(self):
    self.run_contract("extrema-counted-as-reduction")

  def test_extrema_do_not_consume_the_multiplication_budget(self):
    self.run_contract("extrema-multiplication-budget-untouched")

  def test_an_extrema_body_replays_deterministically(self):
    self.run_contract("extrema-replay-deterministic")

  def test_a_rank_one_dropped_extrema_is_a_scalar(self):
    self.run_contract("extrema-rank-one-drop-is-scalar")

  def test_interior_needs_every_position_active_on_every_trip(self):
    self.run_contract("window-facts-interior")

  def test_a_zero_stride_with_two_rows_and_one_active_is_a_tail(self):
    self.run_contract("window-facts-stride-zero-tail")

  def test_a_zero_stride_with_one_row_is_interior(self):
    self.run_contract("window-facts-stride-zero-rows-one")

  def test_a_tail_may_have_a_constant_partial_predicate(self):
    self.run_contract("window-facts-constant-partial")

  def test_a_padded_v3_interior_records_nonconstant_shape_validity(self):
    self.run_contract("window-facts-padded-v3")

  def test_assembly_never_reports_a_compiled_mask(self):
    self.run_contract("window-facts-mask-not-observed")

  def test_a_window_needs_a_window_initialized_invariant(self):
    self.run_contract("window-init-requires-declared-invariant")

  def test_a_window_source_must_be_a_runtime_parameter(self):
    self.run_contract("window-source-must-be-a-parameter")

  def test_a_window_index_must_be_the_induction_variable(self):
    self.run_contract("window-index-must-be-the-induction-variable")

  def test_an_update_cannot_target_a_window_result(self):
    self.run_contract("window-cannot-be-a-state-destination")

  def test_a_window_range_must_fit_the_source(self):
    self.run_contract("window-range-must-fit-the-source")

  def test_two_windows_may_share_one_source(self):
    self.run_contract("two-windows-share-one-source")

  def test_a_window_is_produced_before_the_region_body(self):
    self.run_contract("window-produced-before-region-body")

  def test_a_window_is_recomputed_each_trip(self):
    self.run_contract("window-recomputed-each-trip")

  def test_a_window_does_not_change_the_update_order(self):
    self.run_contract("window-does-not-disturb-snapshots")

  @unittest.skipUnless(SYMIRI.is_file(), "symiri executable not found")
  def test_the_assembled_window_source_passes_check(self):
    source = self.render("window-assembled-source")
    checked = check_source(source, tensor_profile="v2")
    self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

  @unittest.skipUnless(SYMIRI.is_file(), "symiri executable not found")
  def test_the_assembled_window_source_passes_describe(self):
    source = self.render("window-assembled-source")
    result, description = describe_entry(source, tensor_profile="v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIsNotNone(description, result.stdout)
    self.assertEqual(
      [parameter["name"] for parameter in description["inputs"]],
      ["x", "src"],
    )
    self.assertEqual(description["output"]["shape"], [2, 2])

  def test_a_computed_root_requirement_refuses_an_input(self):
    self.run_contract("computed-root-refuses-an-input")

  def test_both_computed_roots_are_produced(self):
    self.run_contract("computed-roots-are-both-produced")

  def test_computed_roots_respect_the_budget(self):
    self.run_contract("computed-roots-respect-the-budget")

  def test_graph_keys_ignore_names_and_declaration_order(self):
    self.run_contract("graph-keys-ignore-names-and-order")

  def test_graph_keys_include_input_types(self):
    self.run_contract("graph-keys-include-input-types")

  def test_graph_keys_follow_roots_edges_and_roles(self):
    self.run_contract("graph-keys-follow-roots-edges-roles")

  def test_a_requirement_keeps_the_input_in_the_result(self):
    self.run_contract("requirement-keeps-the-input-in-the-result")

  def test_a_requirement_builds_a_live_reduction(self):
    self.run_contract("requirement-builds-a-live-reduction")

  def test_a_requirement_counts_two_distinct_consumers(self):
    self.run_contract("requirement-counts-two-distinct-consumers")

  def test_a_requirement_shares_one_extrema_shape_path(self):
    self.run_contract("requirement-shares-one-extrema-shape-path")

  def test_a_requirement_reserves_its_bridge_budget(self):
    self.run_contract("requirement-reserves-bridge-budget")

  def test_a_requirement_rejects_unknown_names(self):
    self.run_contract("requirement-rejects-unknown-names")

  def test_a_window_backing_parameter_supplies_the_window(self):
    self.run_contract("window-buffer-supplies-its-own-source")

  def test_a_window_backing_parameter_stays_out_of_the_value_pool(self):
    self.run_contract("window-buffer-stays-out-of-the-value-pool")

  def test_a_window_backing_witness_uses_its_own_extent(self):
    self.run_contract("window-buffer-witness-uses-its-own-extent")

  def test_a_window_backing_name_collision_is_refused(self):
    self.run_contract("window-buffer-name-collision-is-refused")

  def test_the_witness_enumerates_computed_addresses(self):
    self.run_contract("window-witness-enumerates-computed-addresses")

  def test_the_witness_reports_the_active_set_per_trip(self):
    self.run_contract("window-witness-reports-active-set-per-trip")

  def test_the_witness_refuses_active_beyond_the_source(self):
    self.run_contract("window-witness-refuses-active-beyond-source")

  def test_the_witness_keeps_the_max_address_inside_the_source(self):
    self.run_contract("window-witness-max-address-inside-source")

  def test_stride_two_rows_two_active_three_is_a_tail(self):
    self.run_contract("window-witness-stride-two-rows-two-active-three-is-tail")

  def test_the_witness_agrees_with_the_assembled_window(self):
    source = self.render("window-assembled-source")
    match = re.search(r"tensor window %src, %i, [-0-9.eE+]+, \[(\d+), (\d+), (\d+)\];", source)
    self.assertIsNotNone(match, source)
    rows, stride, active = (int(value) for value in match.groups())
    trips = int(re.search(r"br %i < (\d+), \^body, \^exit;", source).group(1))
    extent = int(re.search(r"%src: tensor \[(\d+),", source).group(1))
    witness = json.loads(self.render("window-witness-json"))
    self.assertEqual(witness["computed_max"], (trips - 1) * stride + rows - 1)
    self.assertEqual(
      witness["per_trip_active"],
      [[trip * stride + row < active for row in range(rows)] for trip in range(trips)],
    )
    self.assertEqual(witness["all_active"], all(all(row) for row in witness["per_trip_active"]))
    self.assertLess(witness["computed_max"], extent)

  @unittest.skipUnless(SYMIRI.is_file(), "symiri executable not found")
  def test_a_three_state_loop_with_a_window_passes_check(self):
    source = self.render("window-three-state-assembled-source")
    checked = check_source(source, tensor_profile="v2")
    self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
    self.assertIn("tensor window %src, %i, 0.0, [2, 1, 4];", source)








@unittest.skipUnless(SYMIRI.is_file(), "native tensor tools unavailable")
class FreeFormFrontendTests(unittest.TestCase):
  """The assembled source is what the native frontend accepts and describes."""

  def render(self, name):
    self.assertTrue(
      FREE_FORM_API.is_file(), f"free-form API binary not found: {FREE_FORM_API}"
    )
    result = subprocess.run(
      [str(FREE_FORM_API), name], text=True, capture_output=True, check=False, timeout=30
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return result.stdout

  def test_three_state_source_passes_the_native_frontend(self):
    source = self.render("three-state-source")
    result = check_source(source, tensor_profile="v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_three_state_source_describes_its_typed_signature(self):
    source = self.render("three-state-source")
    result, description = describe_entry(source, tensor_profile="v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIsNotNone(description, result.stdout)
    self.assertEqual(description["entry"], "kernel")
    self.assertEqual(description["output"]["shape"], [2, 2])
    self.assertEqual(description["output"]["dtype"], "f32")
    self.assertEqual(
      [parameter["name"] for parameter in description["inputs"]], ["x", "y"]
    )

  def test_canonical_source_passes_the_native_frontend(self):
    source = self.render("canonical-source")
    result = check_source(source, tensor_profile="v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    _, description = describe_entry(source, tensor_profile="v2")
    self.assertEqual([parameter["name"] for parameter in description["inputs"]], ["x"])



@unittest.skipUnless(SYMIRI.is_file(), "native tensor tools unavailable")
class FreeFormExportTests(unittest.TestCase):
  """The exporter writes one case directory whose record states what it built."""

  def export(self, *arguments):
    self.assertTrue(
      FREE_FORM_EXPORT.is_file(), f"free-form export binary not found: {FREE_FORM_EXPORT}"
    )
    return subprocess.run(
      [str(FREE_FORM_EXPORT), *arguments],
      text=True,
      capture_output=True,
      check=False,
      timeout=30,
    )

  def test_exporter_writes_a_program_and_its_structure_record(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.export(
        "--policy", "deep", "--seed", "11", "--shape", "2,2",
        "--trip-count", "2", "--states", "3", "--out", directory,
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      source = (Path(directory) / "program.sir").read_text(encoding="utf-8")
      record = json.loads((Path(directory) / "region.json").read_text(encoding="utf-8"))
      self.assertEqual(record["schema"], "refractir.free-form-region.v1")
      self.assertEqual(record["policy"], "deep")
      self.assertEqual(record["seed"], 11)
      self.assertEqual(record["state_count"], 3)
      self.assertEqual(record["trip_count"], 2)
      self.assertEqual(record["route"], "target-only")
      self.assertEqual([parameter["name"] for parameter in record["parameters"]], ["%x", "%y"])
      self.assertEqual(record["results"], ["%next_x", "%next_y", "%next_z"])
      self.assertGreater(record["structure"]["nodes"], 0)
      self.assertEqual(record["structure"]["policy"], "deep")
      self.assertEqual(len(record["structure"]["result_dependencies"]), 3)
      self.assertNotIn("program_sha256", record, "the bundle writer computes the digests")
      checked = check_source(source, tensor_profile="v2")
      self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)


  def test_a_feature_request_records_its_features(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.export(
        "--policy", "wide", "--seed", "3", "--shape", "2,2", "--max-rank", "3",
        "--features", "transpose,expand", "--out", directory,
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      record = json.loads((Path(directory) / "region.json").read_text(encoding="utf-8"))
      self.assertEqual(
        record["features"], {"transpose": True, "expand": True, "extrema": False}
      )
      self.assertEqual(record["max_rank"], 3)
      checked = check_source(
        (Path(directory) / "program.sir").read_text(encoding="utf-8"),
        tensor_profile="v2",
      )
      self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

  def test_features_are_off_unless_asked_for(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.export(
        "--policy", "unguided", "--seed", "7", "--shape", "2,2", "--out", directory
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      record = json.loads((Path(directory) / "region.json").read_text(encoding="utf-8"))
      self.assertEqual(
        record["features"], {"transpose": False, "expand": False, "extrema": False}
      )
      self.assertNotIn("tensor transpose", (Path(directory) / "program.sir").read_text(encoding="utf-8"))
      self.assertNotIn("tensor expand", (Path(directory) / "program.sir").read_text(encoding="utf-8"))

  def test_a_feature_body_uses_the_operation_it_was_given(self):
    seen = {"transpose": False, "expand": False}
    with tempfile.TemporaryDirectory() as directory:
      for seed in range(1, 21):
        if all(seen.values()):
          break
        output = Path(directory) / f"seed-{seed}"
        result = self.export(
          "--policy", "unguided", "--seed", str(seed), "--shape", "2,2",
          "--max-rank", "3", "--features", "transpose,expand", "--out", str(output),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record = json.loads((output / "region.json").read_text(encoding="utf-8"))
        source = (output / "program.sir").read_text(encoding="utf-8")
        for operation in ("transpose", "expand"):
          if operation in record["structure"]["operator_counts"]:
            self.assertIn(f"tensor {operation} ", source)
            seen[operation] = True
      self.assertTrue(seen["transpose"], "a transposable body must appear within 20 seeds")
      self.assertTrue(seen["expand"], "an expandable body must appear within 20 seeds")

  def test_an_unsupported_feature_is_refused(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.export("--features", "rotate", "--out", directory)
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("unknown free-form feature", result.stderr)
      result = self.export("--max-rank", "4", "--out", directory)
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("max rank", result.stderr)


  def _export(self, directory, *arguments):
    result = self.export(*arguments, "--out", str(directory))
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    region = json.loads((Path(directory) / "region.json").read_text(encoding="utf-8"))
    source = (Path(directory) / "program.sir").read_text(encoding="utf-8")
    return region, source

  def test_profile_reaches_the_bundle(self):
    with tempfile.TemporaryDirectory() as directory:
      region, source = self._export(
        directory, "--policy", "unguided", "--seed", "3", "--shape", "2,2",
        "--profile", "v3",
      )
      self.assertEqual(region["profile"], "tensor-v3")
      proposal = tensor_free_form_inputs.proposal(region, pattern="distinct-integer", seed=0)
      bundle = Path(directory) / "bundle"
      case = tensor_free_form_case.build_bundle(
        bundle, region=region, source=source, proposal=proposal
      )
      self.assertEqual(case.record["profile"], "tensor-v3")

  def test_v3_admits_a_non_power_of_two_extent(self):
    with tempfile.TemporaryDirectory() as directory:
      region, source = self._export(
        directory, "--policy", "unguided", "--seed", "3", "--shape", "3,5",
        "--profile", "v3",
      )
      proposal = tensor_free_form_inputs.proposal(region, pattern="distinct-integer", seed=0)
      bundle = Path(directory) / "bundle"
      tensor_free_form_case.build_bundle(
        bundle, region=region, source=source, proposal=proposal
      )
      case = tensor_target_case.load(bundle, symiri=SYMIRI)
      self.assertEqual(case.record["profile"], "tensor-v3")
      checked = check_source(source, tensor_profile="v3")
      self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

  def test_v2_refuses_a_non_power_of_two_extent(self):
    with tempfile.TemporaryDirectory() as directory:
      region, source = self._export(
        directory, "--policy", "unguided", "--seed", "3", "--shape", "3,5",
        "--profile", "v2",
      )
      proposal = tensor_free_form_inputs.proposal(region, pattern="distinct-integer", seed=0)
      bundle = Path(directory) / "bundle"
      tensor_free_form_case.build_bundle(
        bundle, region=region, source=source, proposal=proposal
      )
      with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
        tensor_target_case.load(bundle, symiri=SYMIRI)
      self.assertEqual(raised.exception.status, "invalid_source")

  def test_i32_reaches_parameters_and_output(self):
    with tempfile.TemporaryDirectory() as directory:
      region, source = self._export(
        directory, "--policy", "unguided", "--seed", "3", "--shape", "2,2",
        "--dtype", "i32",
      )
      self.assertEqual(region["dtype"], "i32")
      self.assertEqual(
        [parameter["dtype"] for parameter in region["parameters"]], ["i32", "i32"]
      )
      proposal = tensor_free_form_inputs.proposal(region, pattern="distinct-integer", seed=0)
      bundle = Path(directory) / "bundle"
      case = tensor_free_form_case.build_bundle(
        bundle, region=region, source=source, proposal=proposal
      )
      self.assertEqual(
        [record["dtype"] for record in case.record["inputs"]], ["i32", "i32"]
      )
      self.assertEqual(case.record["output"]["dtype"], "i32")
      checked = check_source(source, tensor_profile="v2")
      self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

  def test_i32_body_passes_the_native_frontend(self):
    with tempfile.TemporaryDirectory() as directory:
      region, source = self._export(
        directory, "--policy", "deep", "--seed", "9", "--shape", "2,2",
        "--dtype", "i32",
      )
      self.assertIn("tensor [2, 2] i32", source)
      checked = check_source(source, tensor_profile="v2")
      self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

  def test_power_of_two_v3_matches_v2_bytes(self):
    with tempfile.TemporaryDirectory() as directory:
      v2_region, v2_source = self._export(
        directory + "/v2", "--policy", "shared", "--seed", "7", "--shape", "2,2",
        "--profile", "v2",
      )
      v3_region, v3_source = self._export(
        directory + "/v3", "--policy", "shared", "--seed", "7", "--shape", "2,2",
        "--profile", "v3",
      )
      self.assertEqual(v2_source, v3_source)
      self.assertEqual(v2_region["profile"], "tensor-v2")
      self.assertEqual(v3_region["profile"], "tensor-v3")


  def window_export(self, directory, *extra):
    return self.export(
      "--policy", "unguided", "--seed", "2", "--states", "2", "--shape", "4,2",
      "--window", "2,1,4", "--window-source", "%x", *extra, "--out", directory,
    )

  def test_window_record_round_trips(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.window_export(directory)
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      record = json.loads((Path(directory) / "region.json").read_text(encoding="utf-8"))
      window = record["window"]
      self.assertEqual(window["result"], "%tile")
      self.assertEqual(window["source"], "%x")
      self.assertEqual(window["source_kind"], "input")
      self.assertEqual(window["source_shape"], [4, 2])
      self.assertIsNone(record["window_source"])
      self.assertEqual(
        [(parameter["role"], parameter["init"]) for parameter in record["parameters"]],
        [("state", "runtime"), ("state", "runtime")],
      )
      self.assertEqual((window["rows"], window["stride"], window["active"]), (2, 1, 4))
      self.assertEqual(window["fill"], 0.0)
      self.assertEqual(window["witness"]["computed_max"], 2)
      self.assertEqual(
        window["witness"]["per_trip_active"], [[True, True], [True, True]]
      )
      self.assertTrue(window["witness"]["all_active"])
      self.assertEqual(
        window["facts"],
        {
          "regime": "interior",
          "shape_validity": "absent",
          "active_predicate": "all_true",
          "mask_after_compilation": "not_observed",
        },
      )
      self.assertFalse(record["canonical"]["accepted"])
      self.assertIn("cannot carry a window", record["canonical"]["refusal"])
      source = (Path(directory) / "program.sir").read_text(encoding="utf-8")
      self.assertIn("tensor window %x, %i, 0.0, [2, 1, 4];", source)

  def test_window_dependency_is_recorded(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.window_export(directory)
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      record = json.loads((Path(directory) / "region.json").read_text(encoding="utf-8"))
      masks = record["structure"]["result_dependencies"]
      self.assertTrue(masks)
      for mask in masks:
        self.assertTrue(mask & (1 << 2), f"the tile must feed every result: {mask}")

  def test_the_exporter_refuses_a_body_that_ignores_the_window(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.export(
        "--policy", "unguided", "--seed", "1", "--states", "2", "--shape", "4,2",
        "--window", "2,1,4", "--window-source", "%x", "--out", directory,
      )
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("must consume the window input", result.stderr)

  def test_window_shape_reaches_inputs(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.window_export(directory)
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      record = json.loads((Path(directory) / "region.json").read_text(encoding="utf-8"))
      tile = [entry for entry in record["inputs"] if entry["value"] == "%tile"]
      self.assertEqual(len(tile), 1)
      self.assertEqual(tile[0]["init"], "window")
      self.assertEqual(tile[0]["shape"], [2, 2])
      self.assertEqual(tile[0]["role"], "invariant")
      source_parameter = record["parameters"][0]
      self.assertEqual((source_parameter["name"], source_parameter["shape"]), ("%x", [4, 2]))

  def _window_buffer_export(self, directory, *extra, seeds=range(1, 17)):
    """Export the first seed whose body consumes the tile with a backing buffer."""
    for seed in seeds:
      output = Path(directory) / f"seed-{seed}"
      result = self.export(
        "--policy", "reduction-heavy", "--seed", str(seed), "--shape", "4,4",
        "--trip-count", "8", "--states", "2", "--window", "4,2,18",
        "--window-buffer-rows", "20", "--profile", "v3", *extra, "--out", str(output),
      )
      if result.returncode == 0:
        return output
    self.fail("no seed within the bound consumed the window buffer")

  def _requirement_export(self, directory, requirement, *extra, seeds=range(1, 25)):
    for seed in seeds:
      output = Path(directory) / f"seed-{seed}"
      result = self.export(
        "--policy", "reduction-heavy", "--seed", str(seed), "--shape", "4,4",
        "--trip-count", "4", "--states", "2", "--window", "2,1,6",
        "--window-buffer-rows", "8", "--profile", "v3", "--features",
        "extrema,transpose,expand", "--window-requirement", requirement,
        "--max-attempts", "4", *extra, "--out", str(output),
      )
      if result.returncode == 0:
        return output
    self.fail(f"no seed within the bound satisfied the {requirement} requirement")

  def test_a_reduction_requirement_reaches_the_window_result(self):
    with tempfile.TemporaryDirectory() as directory:
      output = self._requirement_export(directory, "reduction")
      record = json.loads((output / "region.json").read_text(encoding="utf-8"))
      self.assertEqual(record["window_requirement"], "reduction")
      witnesses = [witness for witness in record["requirements"] if witness["input"] == "%tile"]
      self.assertTrue(witnesses, "the satisfied requirement must be recorded")
      witness = witnesses[0]
      self.assertEqual(witness["results"], ["%next_x"])
      self.assertTrue(witness["path"][0] == "%tile")
      self.assertTrue(
        any(operation in witness["operations"] for operation in ("sum", "rmin", "rmax")),
        witness["operations"],
      )
      self.assertIn("tensor window %window_buffer", (output / "program.sir").read_text(encoding="utf-8"))

  def test_a_shared_requirement_witnesses_two_consumers(self):
    with tempfile.TemporaryDirectory() as directory:
      output = self._requirement_export(directory, "shared")
      record = json.loads((output / "region.json").read_text(encoding="utf-8"))
      filled = [witness for witness in record["requirements"] if witness["distinct_consumers"]]
      self.assertTrue(filled, record["requirements"])
      witness = filled[0]
      self.assertEqual(len(witness["consumers"]), 2)
      self.assertNotEqual(witness["consumers"][0], witness["consumers"][1])
      self.assertEqual(len(set(witness["consumer_results"])), 2)

  def g2c0_export(self, directory, attempt):
      return self.export(
        "--policy", "wide", "--seed", str(20261087 + attempt),
        "--shape", "2,8", "--states", "2", "--trip-count", "4",
        "--window", "2,1,5", "--window-buffer-rows", "7",
        "--profile", "v3", "--features", "extrema,transpose,expand",
        "--max-attempts", "16", "--max-multiplications", "1",
        "--require-computed-roots", "--num-warps", "1",
        "--window-requirement", "extrema-transpose", "--out", directory,
      )

  def test_non_square_extrema_transpose_admits_attempt_zero(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.g2c0_export(directory, 0)
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_non_square_extrema_transpose_witness_has_both_operations(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.g2c0_export(directory, 1)
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      record = json.loads((Path(directory) / "region.json").read_text(encoding="utf-8"))
      witness = next(item for item in record["requirements"] if item["input"] == "%tile")
      self.assertTrue(
        any(operation in witness["operations"] for operation in ("rmin", "rmax", "min", "max")),
        witness["operations"],
      )
      self.assertIn("transpose", witness["operations"])

  def test_non_square_extrema_transpose_stays_within_the_node_budget(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.g2c0_export(directory, 2)
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      record = json.loads((Path(directory) / "region.json").read_text(encoding="utf-8"))
      self.assertLessEqual(record["structure"]["nodes"], record["max_nodes"])
      self.assertEqual(record["max_nodes"], 12)

  def test_non_square_extrema_transpose_roots_are_computed(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.g2c0_export(directory, 3)
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      record = json.loads((Path(directory) / "region.json").read_text(encoding="utf-8"))
      parameters = {parameter["name"] for parameter in record["parameters"]}
      self.assertFalse(set(record["results"]) & parameters)
      self.assertTrue(
        all(update["state"] != update["result"] for update in record["updates"])
      )

  def test_non_square_extrema_transpose_path_reaches_result_zero(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.g2c0_export(directory, 7)
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      record = json.loads((Path(directory) / "region.json").read_text(encoding="utf-8"))
      witness = next(item for item in record["requirements"] if item["input"] == "%tile")
      self.assertEqual(witness["results"], [record["results"][0]])
      self.assertEqual(witness["path"][0], "%tile")
      self.assertEqual(witness["path"][-1], record["results"][0])

  def test_square_extrema_transpose_control_remains_short(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.export(
        "--policy", "reduction-heavy", "--seed", "20261103",
        "--shape", "4,4", "--states", "2", "--trip-count", "8",
        "--window", "4,2,18", "--window-buffer-rows", "20",
        "--profile", "v3", "--features", "extrema,transpose,expand",
        "--max-attempts", "16", "--max-multiplications", "1",
        "--require-computed-roots", "--num-warps", "1",
        "--window-requirement", "extrema-transpose", "--out", directory,
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      record = json.loads((Path(directory) / "region.json").read_text(encoding="utf-8"))
      witness = next(item for item in record["requirements"] if item["input"] == "%tile")
      self.assertIn("transpose", witness["operations"])
      self.assertLessEqual(record["structure"]["nodes"], record["max_nodes"])

  def test_an_extrema_transpose_requirement_shares_a_path(self):
    with tempfile.TemporaryDirectory() as directory:
      output = self._requirement_export(directory, "extrema-transpose")
      record = json.loads((output / "region.json").read_text(encoding="utf-8"))
      witness = record["requirements"][0]
      self.assertTrue(
        any(operation in witness["operations"] for operation in ("rmin", "rmax")),
        witness["operations"],
      )
      self.assertIn("transpose", witness["operations"])

  def test_exporter_limits_reach_the_record(self):
    with tempfile.TemporaryDirectory() as directory:
      output = self._requirement_export(directory, "reduction")
      record = json.loads((output / "region.json").read_text(encoding="utf-8"))
      self.assertEqual(record["max_attempts"], 4)
      self.assertEqual(record["max_multiplications"], 2)
      result = self.export(
        "--policy", "unguided", "--seed", "1", "--shape", "2,2",
        "--max-attempts", "0", "--out", directory,
      )
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("max attempts", result.stderr)

  def test_an_exact_input_bound_refuses_a_body_beyond_the_exact_range(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.export(
        "--policy", "deep", "--seed", "11", "--shape", "2,2", "--trip-count", "2",
        "--states", "2", "--exact-input-bound", "100000000", "--out", directory,
      )
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("exact f32 integer range", result.stderr)
      self.assertFalse((Path(directory) / "program.sir").exists())

  def test_the_reduction_contract_is_recorded_with_its_sum_bound(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.export(
        "--policy", "deep", "--seed", "11", "--shape", "2,2", "--trip-count", "2",
        "--states", "2", "--max-multiplications", "0", "--exact-input-bound", "1",
        "--exact-contract", "reduction", "--out", directory,
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      record = json.loads((Path(directory) / "region.json").read_text(encoding="utf-8"))
      self.assertEqual(record["exact_range"]["contract"], "reduction")
      self.assertLessEqual(record["exact_range"]["sum_bound"], 2**23)
      result = self.export("--exact-contract", "loose", "--out", directory)
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("unknown free-form exact contract", result.stderr)

  def test_most_live_root_selection_is_recorded(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.export(
        "--policy", "deep", "--seed", "11", "--shape", "2,2", "--trip-count", "2",
        "--states", "2", "--root-selection", "most-live", "--out", directory,
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      record = json.loads((Path(directory) / "region.json").read_text(encoding="utf-8"))
      self.assertEqual(record["root_selection"], "most-live")
      result = self.export("--root-selection", "widest", "--out", directory)
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("unknown free-form root selection", result.stderr)

  def test_without_the_flag_the_record_states_no_exact_range(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.export(
        "--policy", "deep", "--seed", "11", "--shape", "2,2", "--trip-count", "2",
        "--states", "2", "--out", directory,
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      record = json.loads((Path(directory) / "region.json").read_text(encoding="utf-8"))
      self.assertNotIn("exact_range", record)
      self.assertNotIn("root_selection", record)

  def test_an_admitted_body_records_its_magnitude_bound(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.export(
        "--policy", "deep", "--seed", "11", "--shape", "2,2", "--trip-count", "2",
        "--states", "2", "--max-multiplications", "0", "--exact-input-bound", "1",
        "--out", directory,
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      record = json.loads((Path(directory) / "region.json").read_text(encoding="utf-8"))
      self.assertEqual(record["exact_range"]["input_bound"], 1.0)
      bound = record["exact_range"]["magnitude_bound"]
      self.assertGreaterEqual(bound, 1.0)
      self.assertLessEqual(bound, 2**24)

  def test_a_requirement_refusal_names_the_subgoal(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.export(
        "--policy", "reduction-heavy", "--seed", "1", "--shape", "4,4",
        "--trip-count", "4", "--states", "2", "--window", "2,1,6",
        "--window-buffer-rows", "8", "--profile", "v3", "--features", "extrema",
        "--window-requirement", "extrema-expand", "--max-attempts", "2",
        "--out", directory,
      )
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("required_path_missing", result.stderr)
      result = self.export(
        "--shape", "2,2", "--window-requirement", "reduction", "--out", directory,
      )
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("needs --window", result.stderr)
      result = self.export("--window-requirement", "rotate", "--out", directory)
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("unknown free-form window requirement", result.stderr)

  def test_live_dag_states_roots_and_edges(self):
    with tempfile.TemporaryDirectory() as directory:
      output = self._requirement_export(directory, "reduction")
      record = json.loads((output / "region.json").read_text(encoding="utf-8"))
      dag = record["live_dag"]
      self.assertEqual(dag["roots"], ["%next_x", "%next_y"])
      self.assertEqual(dag["ops"], [op for op in dag["ops"]])
      self.assertTrue(dag["connectivity_signature"])
      self.assertTrue(dag["typed_fingerprint"])
      self.assertGreaterEqual(len(dag["values"]), len(dag["ops"]))

  def test_a_window_buffer_is_a_separate_parameter(self):
    with tempfile.TemporaryDirectory() as directory:
      output = self._window_buffer_export(directory)
      record = json.loads((output / "region.json").read_text(encoding="utf-8"))
      source = (output / "program.sir").read_text(encoding="utf-8")
      parameters = record["parameters"]
      self.assertEqual(
        [parameter["name"] for parameter in parameters],
        ["%window_buffer", "%x", "%y"],
      )
      self.assertEqual(
        [(parameter["role"], parameter["init"]) for parameter in parameters],
        [("window_source", "window_backing"), ("state", "runtime"), ("state", "runtime")],
      )
      self.assertEqual(parameters[0]["shape"], [20, 4])
      self.assertEqual(
        record["window_source"],
        {
          "parameter": "%window_buffer",
          "shape": [20, 4],
          "dtype": "f32",
          "role": "window_source",
        },
      )
      window = record["window"]
      self.assertEqual(window["source"], "%window_buffer")
      self.assertEqual(window["source_kind"], "backing")
      self.assertEqual(window["source_shape"], [20, 4])
      self.assertEqual(window["witness"]["computed_max"], 17)
      self.assertEqual(record["shape"], [4, 4])
      self.assertEqual(
        [entry["shape"] for entry in record["inputs"] if entry["init"] != "window"],
        [[4, 4], [4, 4]],
      )
      self.assertNotIn("%window_buffer", [entry["value"] for entry in record["inputs"]])
      self.assertIn(
        "%window_buffer: tensor [20, 4] f32", source.splitlines()[0]
      )
      self.assertIn("tensor window %window_buffer, %i, 0.0, [4, 2, 18];", source)
      self.assertNotIn("let mut %window_buffer", source)
      checked = check_source(source, tensor_profile="v3")
      self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
      result, description = describe_entry(source, tensor_profile="v3")
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      self.assertEqual(
        [parameter["name"] for parameter in description["inputs"]],
        ["window_buffer", "x", "y"],
      )
      self.assertEqual(description["output"]["shape"], [4, 4])

  def test_a_window_buffer_witness_violation_is_refused_before_emission(self):
    with tempfile.TemporaryDirectory() as directory:
      for window, rows in (("4,2,12", "8"), ("4,2,4", "2")):
        output = Path(directory) / f"rows-{rows}"
        result = self.export(
          "--policy", "reduction-heavy", "--seed", "1", "--shape", "4,4",
          "--trip-count", "4", "--states", "2", "--window", window,
          "--window-buffer-rows", rows, "--profile", "v3", "--out", str(output),
        )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("fit the source extent", result.stderr)
        self.assertFalse((output / "program.sir").exists())

  def test_a_window_buffer_conflicts_with_an_explicit_source(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.export(
        "--shape", "4,4", "--trip-count", "8", "--window", "4,2,18",
        "--window-buffer-rows", "20", "--window-source", "%x", "--out", directory,
      )
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("conflicts", result.stderr)
      result = self.export("--window-buffer-rows", "20", "--out", directory)
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("needs --window", result.stderr)
      result = self.export("--window", "4,2,18", "--out", directory)
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("window source", result.stderr)

  def test_computed_roots_are_required_for_every_result(self):
    for seed in range(1, 25):
      with tempfile.TemporaryDirectory() as directory:
        result = self.export(
          "--policy", "reduction-heavy", "--seed", str(seed), "--shape", "4,2",
          "--trip-count", "2", "--window", "2,1,4", "--window-source", "%x",
          "--require-computed-roots", "--out", directory,
        )
        if result.returncode != 0:
          continue
        record = json.loads((Path(directory) / "region.json").read_text(encoding="utf-8"))
        source = (Path(directory) / "program.sir").read_text(encoding="utf-8")
        self.assertNotIn("%acc_y = %acc_y;", source)
        self.assertFalse(set(record["results"]) & {"%x", "%y", "%acc_x", "%acc_y"})
        self.assertTrue(
          all(update["state"] != update["result"] for update in record["updates"])
        )
        return
    self.fail("no seed within the bound admitted a computed-root export")

  def test_a_backing_only_change_leaves_state_initialization_identical(self):
    with tempfile.TemporaryDirectory() as directory:
      first = self._window_buffer_export(directory)
      # Rebuild the same seed with a wider backing, so only the parameter type
      # and the witness can change.
      wider = Path(directory) / "wider"
      seed = int(first.name.removeprefix("seed-"))
      result = self.export(
        "--policy", "reduction-heavy", "--seed", str(seed), "--shape", "4,4",
        "--trip-count", "8", "--states", "2", "--window", "4,2,18",
        "--window-buffer-rows", "24", "--profile", "v3", "--out", str(wider / "out"),
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      narrow_source = (first / "program.sir").read_text(encoding="utf-8")
      wide_source = (wider / "out" / "program.sir").read_text(encoding="utf-8")
      def entry_block(source):
        start = source.index("^entry:")
        end = source.index("^header:")
        return source[start:end]
      self.assertEqual(entry_block(narrow_source), entry_block(wide_source))
      self.assertIn("%window_buffer: tensor [20, 4]", narrow_source)
      self.assertIn("%window_buffer: tensor [24, 4]", wide_source)
      for source in (narrow_source, wide_source):
        self.assertIn("tensor window %window_buffer, %i, 0.0, [4, 2, 18];", source)

  def test_window_replay_is_deterministic(self):
    with tempfile.TemporaryDirectory() as directory:
      first = Path(directory) / "first"
      second = Path(directory) / "second"
      for target in (first, second):
        result = self.window_export(target)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      self.assertEqual(
        (first / "program.sir").read_text(encoding="utf-8"),
        (second / "program.sir").read_text(encoding="utf-8"),
      )
      self.assertEqual(
        (first / "region.json").read_text(encoding="utf-8"),
        (second / "region.json").read_text(encoding="utf-8"),
      )

  def test_a_window_with_two_states_and_a_reduction(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.export(
        "--policy", "reduction-heavy", "--seed", "3", "--states", "2", "--shape", "4,2",
        "--window", "2,1,4", "--window-source", "%x", "--out", directory,
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      source = (Path(directory) / "program.sir").read_text(encoding="utf-8")
      self.assertIn("tensor window %x, %i, 0.0, [2, 1, 4];", source)
      self.assertIn("tensor sum", source)
      checked = check_source(source, tensor_profile="v2")
      self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
  def test_exporter_refuses_an_impossible_configuration(self):
    with tempfile.TemporaryDirectory() as directory:
      result = self.export("--states", "5", "--out", directory)
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("two to four states", result.stderr)
      self.assertFalse((Path(directory) / "program.sir").exists())

if __name__ == "__main__":
  unittest.main()
