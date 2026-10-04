"""Contracts for paired studies over distinct generated loop regions."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test.lib import tensor_exact_f32, tensor_fpsan
from test.tensor.gluon import run_selection_study

PROGRAM_IDS = (
  "row-reduction-v1",
  "column-reduction-v1",
  "two-axis-reduction-v1",
)
NATIVE_GENERATED_PROGRAM_IDS = (
  "native-seed-7",
  "native-seed-17",
  "native-seed-29",
  "native-seed-0-shape-2x4-iterations-3",
  "native-seed-3-shape-4x2-iterations-4",
  "native-seed-6-shape-2x4-iterations-2",
  "native-seed-2-shape-8x8-iterations-2",
)


class GeneratedSelectionStudyTests(unittest.TestCase):
  def test_native_generator_corpus_freezes_sources_and_generation_options(self):
    manifest_path = run_selection_study.PROGRAM_DIR / "native-generated-corpus.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    self.assertEqual(manifest["schema"], "refractir.generated-region-corpus.v1")
    self.assertEqual(
      tuple(record["program_id"] for record in manifest["programs"]),
      NATIVE_GENERATED_PROGRAM_IDS,
    )
    for record in manifest["programs"]:
      source_path = run_selection_study.PROGRAM_DIR / record["source_file"]
      source = source_path.read_text(encoding="utf-8")
      self.assertEqual(
        record["source_sha256"], hashlib.sha256(source.encode()).hexdigest()
      )
      self.assertIn(f"seed-{record['generation']['seed']}", record["program_id"])
      self.assertEqual(len(record["generation"]["shape"]), 2)
      self.assertGreaterEqual(record["generation"]["iterations"], 1)
      tensor_exact_f32._generated_region_program(source)

  def test_selection_catalog_admits_frozen_native_graph_sources(self):
    sources = [
      run_selection_study._matrix_program_source(program_id)
      for program_id in NATIVE_GENERATED_PROGRAM_IDS
    ]

    self.assertEqual(
      len({hashlib.sha256(source.encode()).hexdigest() for source in sources}), 7
    )

  def test_default_matrix_includes_all_frozen_native_graph_programs(self):
    self.assertTrue(
      set(NATIVE_GENERATED_PROGRAM_IDS).issubset(run_selection_study.PROGRAM_SOURCES)
    )
    cells = run_selection_study._matrix_cell_specs(
      program_ids=NATIVE_GENERATED_PROGRAM_IDS,
      target_indices=((0, 0), (1, 1)),
      seeds=(17, 29),
      layout_seeds=(20261002, 20261003),
    )

    self.assertEqual(len(cells), 56)
    self.assertEqual(
      {cell["program_id"] for cell in cells}, set(NATIVE_GENERATED_PROGRAM_IDS)
    )

  def test_matrix_accepts_an_index_in_an_expanded_program_shape(self):
    cells = run_selection_study._matrix_cell_specs(
      program_ids=("native-seed-0-shape-2x4-iterations-3",),
      target_indices=((1, 3),),
      seeds=(7,),
      layout_seeds=(20261002,),
    )

    self.assertEqual(len(cells), 1)
    self.assertEqual(cells[0]["target_index"], (1, 3))

  def test_native_seed_payload_query_keeps_loop_mutant_calibrations(self):
    source = run_selection_study._matrix_program_source("native-seed-7")

    query = tensor_fpsan.build_generated_payload_query(source)

    self.assertEqual(query["shape"], (2, 2))
    self.assertEqual(query["iterations"], 2)
    self.assertIn("lost_iteration_output_0", query["script"])
    self.assertIn("lost_update_acc_y_output_0", query["script"])

  def test_program_catalog_contains_distinct_admitted_graphs(self):
    sources = [run_selection_study._matrix_program_source(name) for name in PROGRAM_IDS]
    metadata = [
      tensor_exact_f32._generated_region_program(source) for source in sources
    ]
    signatures = [
      tuple(item[3][index][1] for index in range(len(item[3]))) for item in metadata
    ]

    self.assertEqual(
      len({hashlib.sha256(source.encode()).hexdigest() for source in sources}), 3
    )
    self.assertEqual(signatures[0], ("add", "sum", "broadcast", "add", "add"))
    self.assertEqual(signatures[1], ("add", "sum", "broadcast", "add", "add"))
    self.assertEqual(
      signatures[2], ("add", "sum", "broadcast", "sum", "broadcast", "add", "add")
    )

  def test_matrix_specs_expand_program_property_seed_and_layout_axes(self):
    cells = run_selection_study._matrix_cell_specs(
      program_ids=PROGRAM_IDS[:2],
      target_indices=((0, 0), (1, 1)),
      seeds=(17, 29),
      layout_seeds=(20261002, 20261003),
    )

    self.assertEqual(len(cells), 16)
    self.assertEqual(len({cell["cell_id"] for cell in cells}), 16)
    self.assertEqual({cell["program_id"] for cell in cells}, set(PROGRAM_IDS[:2]))
    self.assertEqual(
      {(cell["target_index"], cell["seed"], cell["layout_seed"]) for cell in cells},
      {
        (index, seed, layout_seed)
        for index in ((0, 0), (1, 1))
        for seed in (17, 29)
        for layout_seed in (20261002, 20261003)
      },
    )

  def test_generated_selection_query_preserves_program_and_property_index(self):
    source = run_selection_study._matrix_program_source("two-axis-reduction-v1")
    query = run_selection_study.render_selection_query(
      source=source,
      iterations=2,
      target=84,
      target_index=(0, 1),
    )

    self.assertIn("tensor extract %out, [0, 1]", query)
    self.assertIn("sym %?x : value tensor [2, 2] i32 in [0, 8]", query)
    self.assertIn("%t3 = tensor sum %t0, 0, keep", query)
    self.assertNotIn(" f32", query)

  def test_generated_selection_query_rejects_an_out_of_shape_property(self):
    source = run_selection_study._matrix_program_source("row-reduction-v1")

    with self.assertRaisesRegex(ValueError, "outside the output shape"):
      tensor_exact_f32.render_generated_region_query(
        source, (0, 8), 37, target_index=(2, 0)
      )

  def test_generated_gpu_child_receives_source_and_layout_seed(self):
    command = run_selection_study._gpu_child_command(
      Path("gpu.json"),
      Path("ir"),
      input_values=(1, 2, 4, 8),
      iterations=2,
      target=84,
      target_index=(0, 1),
      layout_seed=20261003,
      program_source=Path("program.sir"),
    )

    self.assertIn("test.tensor.gluon.run_generated_region", command)
    self.assertIn("--program", command)
    self.assertIn("program.sir", command)
    self.assertIn("20261003", command)

  def test_matrix_runner_pairs_distinct_programs_with_derived_targets(self):
    def fake_run(output_dir, **kwargs):
      return {
        "ledger": {
          "fixed": {
            "program_id": kwargs["program_id"],
            "program_sha256": hashlib.sha256(
              kwargs["program_source"].encode()
            ).hexdigest(),
          },
          "trials": [{"property_id": str(kwargs["target_index"])}],
        },
        "summary": {"status": "complete", "cell": Path(output_dir).name},
      }

    with tempfile.TemporaryDirectory() as temporary:
      output = Path(temporary) / "matrix"
      with mock.patch.object(run_selection_study, "run", side_effect=fake_run) as run:
        matrix = run_selection_study.run_matrix(
          output,
          solver="symirsolve",
          program_ids=PROGRAM_IDS[:2],
          target_indices=((0, 0),),
          seeds=(17,),
          layout_seeds=(20261002,),
          budget_s=1.0,
        )

    self.assertEqual(matrix["status"], "complete")
    self.assertEqual(run.call_count, 2)
    self.assertEqual(
      {call.kwargs["program_id"] for call in run.call_args_list}, set(PROGRAM_IDS[:2])
    )
    self.assertEqual({call.kwargs["target"] for call in run.call_args_list}, {37, 61})
    self.assertTrue(all(call.kwargs["program_source"] for call in run.call_args_list))

  def test_matrix_runner_can_hold_one_target_fixed_across_programs(self):
    def fake_run(output_dir, **kwargs):
      return {
        "ledger": {
          "fixed": {
            "program_id": kwargs["program_id"],
            "program_sha256": hashlib.sha256(
              kwargs["program_source"].encode()
            ).hexdigest(),
          },
          "trials": [{"property_id": f"target-{kwargs['target']}"}],
        },
        "summary": {"status": "complete", "cell": Path(output_dir).name},
      }

    with tempfile.TemporaryDirectory() as temporary:
      output = Path(temporary) / "matrix"
      with mock.patch.object(run_selection_study, "run", side_effect=fake_run) as run:
        matrix = run_selection_study.run_matrix(
          output,
          solver="symirsolve",
          program_ids=NATIVE_GENERATED_PROGRAM_IDS,
          target_indices=((0, 0),),
          seeds=(17,),
          layout_seeds=(20261002,),
          target=1268,
          budget_s=1.0,
        )

    self.assertEqual(matrix["status"], "complete")
    self.assertEqual(matrix["protocol"]["property_target"], 1268)
    self.assertEqual(matrix["protocol"]["route"], "exact-f32")
    self.assertEqual(
      matrix["protocol"]["semantic_contract"],
      "generated-multi-output-region-f32-exact-loop-v1",
    )
    self.assertEqual(
      matrix["protocol"]["property_contract"],
      "numeric output element equals an integer-valued target under the exact-f32 model",
    )
    self.assertIn("finite f32", matrix["protocol"]["input_transport"])
    self.assertEqual(matrix["protocol"]["solver_timeout_s"], 60.0)
    self.assertEqual({call.kwargs["target"] for call in run.call_args_list}, {1268})
    self.assertEqual({cell["target"] for cell in matrix["cells"]}, {1268})


if __name__ == "__main__":
  unittest.main()
