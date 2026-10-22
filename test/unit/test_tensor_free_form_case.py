"""Contract tests for the free-form region to target-only bundle bridge."""

import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test.lib import (
  tensor_execution,
  tensor_free_form_case,
  tensor_free_form_inputs,
  tensor_target_case,
)
from test.unit.tensor_support import SYMIRC, SYMIRI


def _triton_interpreter_available():
  try:
    import torch  # noqa: F401
    import triton  # noqa: F401
  except Exception:
    return False
  return True


@unittest.skipUnless(
  SYMIRC.is_file() and SYMIRI.is_file() and _triton_interpreter_available(),
  "native tools or the Triton interpreter are unavailable",
)
class StridedOutputExecutionTests(unittest.TestCase):
  """A padded output leaves every gap slot holding its canary."""

  def test_gap_slots_keep_the_canary(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      bundle = root / "bundle"
      proposal = tensor_free_form_inputs.proposal(
        REGION, pattern="distinct-integer", seed=0
      )
      tensor_free_form_case.build_bundle(
        bundle,
        region=REGION,
        source=SOURCE,
        proposal=proposal,
        output_row_padding=1,
      )
      case = tensor_target_case.load(bundle, symiri=SYMIRI)
      artifacts = root / "artifacts"
      kernel, stderr, stdout = tensor_execution.emit_kernel(case, SYMIRC, artifacts)
      self.assertIsNotNone(kernel, stdout + stderr)
      environment = dict(os.environ)
      environment.pop("TRITON_INTERPRET", None)
      completed = subprocess.run(
        [
          sys.executable,
          "-m",
          "test.lib.triton_case_worker",
          "--case",
          str(bundle),
          "--artifacts-dir",
          str(artifacts),
          "--mode",
          "interpret",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env={**environment, "TRITON_INTERPRET": "1"},
        cwd=Path(__file__).resolve().parents[2],
      )
      self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
      actual = (artifacts / "actual.bin").read_bytes()
      self.assertEqual(actual, struct.pack("<4f", 2.0, 4.0, 6.0, 8.0))


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


  def test_strided_output_round_trips(self):
    layout = tensor_free_form_case.strided_output([2, 2], row_padding=1)
    self.assertEqual(layout["element_strides"], [3, 1])
    self.assertEqual(layout["storage_span"], 5)
    self.assertEqual(layout["axis_order"], [1, 0])
    proposal = tensor_free_form_inputs.proposal(REGION, pattern="distinct-integer", seed=0)
    case = tensor_free_form_case.build_bundle(
      self.bundle,
      region=REGION,
      source=SOURCE,
      proposal=proposal,
      output_row_padding=1,
    )
    output = case.record["output"]
    self.assertEqual(output["element_strides"], [3, 1])
    self.assertEqual(output["storage_span"], 5)
    self.assertEqual(output["element_offset"], 0)
    self.assertEqual(output["byte_length"], 16)

  def test_offset_output_shifts_every_address(self):
    dense = tensor_free_form_case.dense_layout([2, 2])
    offset = tensor_free_form_case.strided_output([2, 2], row_padding=0, element_offset=2)
    self.assertEqual(offset["element_offset"], 2)
    self.assertEqual(offset["storage_span"], dense["storage_span"] + 2)
    self.assertEqual(offset["element_strides"], dense["element_strides"])
    proposal = tensor_free_form_inputs.proposal(REGION, pattern="distinct-integer", seed=0)
    case = tensor_free_form_case.build_bundle(
      self.bundle,
      region=REGION,
      source=SOURCE,
      proposal=proposal,
      output_offset=2,
    )
    self.assertEqual(case.record["output"]["element_offset"], 2)
    self.assertEqual(case.record["output"]["storage_span"], 6)

  def test_output_span_matches_the_layout(self):
    for shape, padding, offset in (
      ([2, 2], 0, 0),
      ([2, 2], 1, 0),
      ([2, 2], 1, 3),
      ([2, 3], 2, 1),
      ([2], 0, 3),
    ):
      with self.subTest(shape=shape, padding=padding, offset=offset):
        layout = tensor_free_form_case.strided_output(
          shape, row_padding=padding, element_offset=offset
        )
        expected = offset + sum(
          (extent - 1) * stride
          for extent, stride in zip(shape, layout["element_strides"])
        ) + 1
        self.assertEqual(layout["storage_span"], expected)

  def test_offset_output_with_rank_one_state(self):
    layout = tensor_free_form_case.strided_output([2], row_padding=0, element_offset=3)
    self.assertEqual(layout["element_strides"], [1])
    self.assertEqual(layout["element_offset"], 3)
    self.assertEqual(layout["storage_span"], 5)

  def test_row_padding_needs_a_matrix(self):
    with self.assertRaises(tensor_target_case.TargetCaseError):
      tensor_free_form_case.strided_output([2], row_padding=1)
    with self.assertRaises(tensor_target_case.TargetCaseError):
      tensor_free_form_case.strided_output([2, 2], row_padding=-1)
    with self.assertRaises(tensor_target_case.TargetCaseError):
      tensor_free_form_case.strided_output([2, 2], row_padding=0, element_offset=-1)

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
