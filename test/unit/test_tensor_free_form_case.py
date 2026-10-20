"""Contract tests for the free-form region to target-only bundle bridge."""

import tempfile
import unittest
from pathlib import Path

from test.lib import tensor_free_form_case, tensor_free_form_inputs, tensor_target_case

REGION = {
  "schema": "refractir.free-form-region.v1",
  "entry": "kernel",
  "profile": "tensor-v2",
  "dtype": "f32",
  "policy": "unguided",
  "seed": 20261020,
  "shape": [2, 2],
  "trip_count": 2,
  "state_count": 3,
  "num_warps": 4,
  "parameters": [
    {"name": "%x", "shape": [2, 2], "dtype": "f32"},
    {"name": "%y", "shape": [2, 2], "dtype": "f32"},
  ],
  "results": ["%next_x", "%next_y", "%next_z"],
}

SOURCE = """\
fun @kernel(%x: tensor [2, 2] f32, %y: tensor [2, 2] f32) : tensor [2, 2] f32 {
  let mut %out: tensor [2, 2] f32 = undef;
^entry:
  %out = tensor add %x, %y;
  ret %out;
}
"""


class FreeFormCaseTests(unittest.TestCase):
  """A bundle carries the proposal's bytes and no reference."""

  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.bundle = Path(self.directory.name) / "bundle"

  def test_a_bundle_states_the_proposals_bytes_and_no_reference(self):
    proposal = tensor_free_form_inputs.proposal(REGION, pattern="distinct-integer", seed=0)
    case = tensor_free_form_case.build_bundle(
      self.bundle, region=REGION, source=SOURCE, proposal=proposal
    )
    self.assertEqual(case.inputs, proposal.data)
    self.assertEqual(case.record["profile"], "tensor-v2")
    self.assertEqual(case.record["reference"], {"kind": "none"})
    self.assertEqual(case.record["comparison"], {"kind": "none"})
    self.assertFalse((self.bundle / "expected.bin").exists())
    self.assertEqual([record["name"] for record in case.record["inputs"]], ["%x", "%y"])
    self.assertEqual(case.record["output"]["shape"], [2, 2])
    self.assertEqual(case.record["output"]["byte_length"], 16)
    self.assertEqual(case.record["launch"]["num_warps"], 4)
    self.assertEqual(case.record["execution"]["guard_envelope"], {"prefix": 1, "suffix": 1})

  def test_the_dense_layout_lists_axes_fastest_first(self):
    layout = tensor_free_form_case.dense_layout([2, 3])
    self.assertEqual(layout["element_strides"], [3, 1])
    self.assertEqual(layout["axis_order"], [1, 0])
    self.assertEqual(layout["storage_span"], 6)
    square = tensor_free_form_case.dense_layout([2, 2])
    self.assertEqual(square["axis_order"], [1, 0])

  def test_a_proposal_that_does_not_name_the_parameters_is_refused(self):
    proposal = tensor_free_form_inputs.proposal(REGION, pattern="distinct-integer", seed=0)
    renamed = tensor_free_form_inputs.proposal(
      {"parameters": [{"name": "%z", "shape": [2, 2], "dtype": "f32"}]},
      pattern="distinct-integer",
      seed=0,
    )
    with self.assertRaises(tensor_target_case.TargetCaseError):
      tensor_free_form_case.input_records(REGION, renamed)
    with self.assertRaises(tensor_target_case.TargetCaseError):
      tensor_free_form_case.build_bundle(
        self.bundle, region={**REGION, "num_warps": 0}, source=SOURCE, proposal=proposal
      )


if __name__ == "__main__":
  unittest.main()
