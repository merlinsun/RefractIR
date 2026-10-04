"""Contracts for paired studies over distinct generated loop regions."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test.lib import tensor_exact_f32
from test.tensor.gluon import run_selection_study

PROGRAM_IDS = (
  "row-reduction-v1",
  "column-reduction-v1",
  "two-axis-reduction-v1",
)


class GeneratedSelectionStudyTests(unittest.TestCase):
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


if __name__ == "__main__":
  unittest.main()
