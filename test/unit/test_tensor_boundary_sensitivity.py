"""Accounting contracts for the frozen boundary-input sensitivity study."""

import copy
import hashlib
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test.lib import tensor_boundary_sensitivity as study

_SOURCE = """fun @kernel(%x: tensor [2] f32) : tensor [2] f32 {
^entry:
  ret %x;
}
"""
_CONFIG = {
  "profile": "tensor-v2",
  "dtype": "f32",
  "input_domain": [-1, 2],
  "proposals_per_arm": 4,
  "comparison": {
    "mode": "numeric",
    "atol": [2, 100000],
    "rtol": [2, 1000],
  },
  "launch": {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False},
  "physical_layouts": {
    "policy": "contiguous-row-major-v1",
    "buffer_offsets": 0,
    "grid": [1, 1, 1],
  },
  "source_snapshot_sha256": "test-snapshot-digest",
}
_ADDED = """fun @kernel(%x: tensor [2] f32) : tensor [2] f32 {
^entry:
  %out = tensor add %x, %x;
  ret %out;
}
"""
_RENAMED_SOURCE = _SOURCE.replace("%x", "%y")
_CONTROL_SOURCE = """fun @kernel(%x: tensor [2, 2] f32, %y: tensor [2, 2] f32) : tensor [2, 2] f32 {
^entry:
  %sum = tensor add %x, %y;
  %rows = tensor sum %sum, 1, drop;
  %expanded = tensor expand %rows, 1;
  %out = tensor broadcast %expanded, [2, 2];
  ret %out;
}
"""
_TRANSPOSE_SOURCE = """fun @kernel(%x: tensor [2, 2] f32) : tensor [2, 2] f32 {
^entry:
  %out = tensor transpose %x, [1, 0];
  ret %out;
}
"""
_GENERATED_STYLE_SOURCE = """fun @kernel(%a: tensor [2, 2] f32, %b: tensor [2, 2] f32) : tensor [2, 2] f32 {
  let mut %t0: tensor [2, 2] f32 = undef;
  let mut %tCopy: tensor [2, 2] f32 = undef;
  let mut %t1: tensor [2, 2] f32 = undef;
  let mut %t2: tensor [2, 2] f32 = undef;
  let mut %t3: tensor [2] f32 = undef;
  let mut %t4: tensor [2, 1] f32 = undef;
  let mut %t5: tensor [2, 2] f32 = undef;
  let mut %t6: tensor [2, 2] f32 = undef;
  let mut %t7: tensor [2, 2] f32 = undef;
^entry:
  %t0 = tensor add %a, 1.0;
  %tCopy = %t0;
  %t1 = tensor add %b, 2.0;
  %t2 = tensor add %tCopy, %t1;
  %t3 = tensor sum %t2, 1, drop;
  %t4 = tensor expand %t3, 1;
  %t5 = tensor broadcast %t4, [2, 2];
  %t6 = tensor transpose %t5, [1, 0];
  %t7 = tensor add %t6, %a;
  ret %t7;
}
"""


def _corpus():
  return study.freeze_corpus(
    [{"seed": 2027, "status": "candidate", "source": _SOURCE}],
    config=_CONFIG,
    base_revision="base-revision",
    target_count=1,
  )


class TensorBoundarySensitivityTests(unittest.TestCase):
  def test_rejects_changed_frozen_source_or_configuration(self):
    corpus = _corpus()
    digest = corpus["corpus_sha256"]

    changed_source = copy.deepcopy(corpus)
    changed_source["kernels"][0]["source"] += "// changed\n"
    with self.assertRaisesRegex(ValueError, "frozen corpus"):
      study.validate_frozen_corpus(changed_source, expected_sha256=digest)

    changed_config = copy.deepcopy(corpus)
    changed_config["config"]["launch"]["num_warps"] = 8
    with self.assertRaisesRegex(ValueError, "frozen corpus"):
      study.validate_frozen_corpus(changed_config, expected_sha256=digest)

  def test_source_snapshot_hashes_all_nonbuild_files_deterministically(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / "src").mkdir()
      (root / "build").mkdir()
      (root / ".git").mkdir()
      (root / "src" / "kernel.cpp").write_text("first", encoding="utf-8")
      (root / "build" / "generated.bin").write_bytes(b"ignored build output")
      (root / ".git" / "index").write_bytes(b"ignored git state")
      first = study.source_snapshot_manifest(root)
      (root / "src" / "kernel.cpp").write_text("second", encoding="utf-8")
      second = study.source_snapshot_manifest(root)

    self.assertEqual([item["path"] for item in first["files"]], ["src/kernel.cpp"])
    self.assertNotEqual(first["snapshot_sha256"], second["snapshot_sha256"])

  def test_saved_corpus_checks_the_separate_frozen_digest(self):
    corpus = _corpus()
    digest = corpus["corpus_sha256"]
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "corpus.json"
      study.save_corpus(path, corpus)
      self.assertEqual(study.load_corpus(path, expected_sha256=digest), corpus)

      changed = copy.deepcopy(corpus)
      changed["kernels"][0]["source"] += "// edited after freeze\n"
      path.write_text(json.dumps(changed), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "frozen corpus"):
        study.load_corpus(path, expected_sha256=digest)

  def test_allocates_exactly_four_proposals_per_arm_without_retry_slots(self):
    slots = study.allocate_proposals(_corpus(), proposals_per_arm=4)

    self.assertEqual(len(slots), 8)
    for arm in ("legacy_cycle", "boundary"):
      proposals = [slot for slot in slots if slot["arm"] == arm]
      self.assertEqual(len(proposals), 4)
      self.assertEqual([slot["proposal_index"] for slot in proposals], [0, 1, 2, 3])
      self.assertEqual(len({slot["proposal_id"] for slot in proposals}), 4)

  def test_retains_refused_and_timed_out_slots_without_replacing_kernels(self):
    corpus = _corpus()
    ledger = study.new_study_ledger(corpus, proposals_per_arm=4)
    slots = ledger["proposals"]
    original_kernel_ids = [kernel["kernel_id"] for kernel in corpus["kernels"]]

    refused = slots[0]
    timeout = slots[1]
    study.record_proposal_result(
      ledger,
      refused["proposal_id"],
      {"proposal_status": "refused", "input_bytes_hex": "01020304"},
    )
    study.record_proposal_result(
      ledger,
      timeout["proposal_id"],
      {"proposal_status": "timeout", "input_bytes_hex": "05060708"},
    )

    self.assertEqual(
      [kernel["kernel_id"] for kernel in corpus["kernels"]], original_kernel_ids
    )
    self.assertEqual(len(ledger["proposals"]), 8)
    self.assertEqual(ledger["proposals"][0]["result"]["proposal_status"], "refused")
    self.assertEqual(ledger["proposals"][1]["result"]["proposal_status"], "timeout")
    self.assertEqual(ledger["proposals"][1]["result"]["input_bytes_hex"], "05060708")

  def test_keeps_invalid_inapplicable_and_undistinguished_controls_separate(self):
    controls = [
      {"control_id": "operand", "status": "invalid_program"},
      {"control_id": "permutation", "status": "inapplicable"},
      {"control_id": "reduction", "status": "invalid_on_input"},
      {"control_id": "broadcast", "status": "not_distinguished"},
      {"control_id": "negative", "status": "not_distinguished"},
      {"control_id": "operand", "status": "distinguished"},
    ]

    summary = study.summarize_control_results(controls)

    self.assertEqual(summary["counts"]["invalid_program"], 1)
    self.assertEqual(summary["counts"]["inapplicable"], 1)
    self.assertEqual(summary["counts"]["invalid_on_input"], 1)
    self.assertEqual(summary["counts"]["not_distinguished"], 2)
    self.assertEqual(summary["counts"]["distinguished"], 1)
    self.assertEqual(summary["by_control"]["operand"]["distinguished"], 1)

  def test_recovery_is_idempotent_and_does_not_double_count_executions(self):
    corpus = _corpus()
    ledger = study.new_study_ledger(corpus, proposals_per_arm=4)
    slot = ledger["proposals"][0]
    result = {
      "proposal_status": "accepted",
      "input_bytes_hex": "01020304",
      "gpu": {
        "status": "pass",
        "compiled": True,
        "executed": True,
        "compilation_count": 1,
        "execution_count": 1,
      },
    }
    study.record_proposal_result(ledger, slot["proposal_id"], result)
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "study.json"
      study.save_ledger(path, ledger)
      resumed = study.load_ledger(path, corpus)

    inserted = study.record_proposal_result(resumed, slot["proposal_id"], result)
    summary = study.summarize_study(resumed)

    self.assertFalse(inserted)
    self.assertEqual(summary["gpu_work"]["compilation_count"], 1)
    self.assertEqual(summary["gpu_work"]["execution_count"], 1)
    self.assertEqual(summary["completed_proposals"], 1)

  def test_sampled_input_is_durable_before_reference_or_gpu_execution(self):
    corpus = _corpus()
    ledger = study.new_study_ledger(corpus, proposals_per_arm=4)
    slot = ledger["proposals"][0]
    proposal = {
      "proposal_status": "accepted",
      "sampler_version": study.LEGACY_SAMPLER_VERSION,
      "input_bytes_hex": "01020304",
    }
    self.assertTrue(study.record_proposal_input(ledger, slot["proposal_id"], proposal))

    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "study.json"
      study.save_ledger(path, ledger)
      resumed = study.load_ledger(path, corpus)

    recovered = resumed["proposals"][0]
    self.assertEqual(recovered["proposal"]["input_bytes_hex"], "01020304")
    self.assertIsNone(recovered["result"])
    with self.assertRaisesRegex(ValueError, "different input proposal"):
      study.record_proposal_input(
        resumed,
        slot["proposal_id"],
        {**proposal, "input_bytes_hex": "05060708"},
      )

  def test_recovery_restores_gpu_submission_checkpoint_from_proposal_sidecar(self):
    corpus = _corpus()
    ledger = study.new_study_ledger(corpus, proposals_per_arm=4)
    slot = ledger["proposals"][0]
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "study.json"
      study.save_ledger(path, ledger)
      slot["gpu_submitted"] = True
      slot["gpu_device_id"] = 4
      slot["gpu_case_directory"] = "cases/kernel/legacy_cycle/proposal-00"
      slot["gpu_command"] = {"runner": "test.lib.run_triton_tests.run_case"}
      study.save_proposal_record(path, ledger, slot["proposal_id"])

      resumed = study.load_ledger(path, corpus)

    recovered = resumed["proposals"][0]
    self.assertTrue(recovered["gpu_submitted"])
    self.assertEqual(recovered["gpu_device_id"], 4)
    self.assertEqual(recovered["gpu_case_directory"], slot["gpu_case_directory"])
    self.assertEqual(recovered["gpu_command"], slot["gpu_command"])

  def test_proposal_journal_recovers_staged_input_and_controls_after_crash(self):
    corpus = _corpus()
    ledger = study.new_study_ledger(corpus, proposals_per_arm=4)
    slot = ledger["proposals"][0]
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "study.json"
      study.save_ledger(path, ledger)
      study.record_proposal_input(
        ledger,
        slot["proposal_id"],
        {
          "proposal_status": "accepted",
          "sampler_version": slot["sampler_version"],
          "input_bytes_hex": "01020304",
        },
      )
      control = {
        "control_id": "negative",
        "status": "not_distinguished",
        "comparison": {"status": "pass"},
      }
      study.record_control_result(ledger, slot["proposal_id"], control)
      study.save_proposal_record(path, ledger, slot["proposal_id"])
      recovered = study.load_ledger(path, corpus)

    self.assertEqual(
      recovered["proposals"][0]["proposal"]["input_bytes_hex"], "01020304"
    )
    self.assertEqual(len(recovered["controls"]), 1)
    self.assertEqual(recovered["controls"][0]["status"], "not_distinguished")

  def test_reference_output_is_checkpointed_before_compilation(self):
    corpus = _corpus()
    ledger = study.new_study_ledger(corpus, proposals_per_arm=4)
    slot = ledger["proposals"][0]
    reference = {
      "status": "accepted",
      "output_bytes_hex": "01020304",
      "elapsed_s": 0.25,
    }
    self.assertTrue(
      study.record_reference_result(ledger, slot["proposal_id"], reference)
    )

    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "study.json"
      study.save_ledger(path, ledger)
      study.save_proposal_record(path, ledger, slot["proposal_id"])
      recovered = study.load_ledger(path, corpus)

    self.assertEqual(recovered["proposals"][0]["reference"], reference)

  def test_control_summary_handles_no_applicable_controls(self):
    summary = study.summarize_control_results([])

    self.assertEqual(summary["total"], 0)
    self.assertIsNone(summary["detection_rate"])
    self.assertEqual(summary["counts"]["inapplicable"], 0)

  def test_candidate_schedule_retains_failures_and_deduplicates_sources(self):
    outputs = {
      2027: SimpleNamespace(returncode=0, stdout=_SOURCE, stderr=""),
      2028: SimpleNamespace(
        returncode=1, stdout="", stderr="graph_exhaustion: no admissible graph"
      ),
      2029: SimpleNamespace(returncode=0, stdout=_RENAMED_SOURCE, stderr=""),
      2030: SimpleNamespace(returncode=0, stdout=_ADDED, stderr=""),
    }

    def run(command, **_kwargs):
      return outputs[int(command[-1])]

    with patch.object(study.subprocess, "run", side_effect=run):
      candidates = study.generate_candidate_schedule(
        "/unused/native-api",
        start_seed=2027,
        target_count=2,
        frontend_checker=lambda _source: SimpleNamespace(
          returncode=0, stderr="", stdout=""
        ),
      )

    corpus = study.freeze_corpus(
      candidates,
      config=_CONFIG,
      base_revision="base-revision",
      target_count=2,
    )
    self.assertEqual(
      [record["status"] for record in corpus["candidate_enumeration"]],
      ["selected", "graph_generation_failure", "duplicate", "selected"],
    )
    self.assertEqual(
      [kernel["candidate_seed"] for kernel in corpus["kernels"]], [2027, 2030]
    )

  def test_legacy_cycle_materializes_all_four_declared_phases(self):
    input_records = [
      {"name": "x", "dtype": "f32", "shape": [8], "byte_length": 32},
      {"name": "y", "dtype": "f32", "shape": [4, 2], "byte_length": 32},
    ]
    phases = [study.legacy_cycle_buffers(input_records, phase) for phase in range(4)]

    values = [struct.unpack("<8f", buffers[0]["bytes"]) for buffers in phases]
    self.assertEqual(values[0], (-1.0, 0.0, 1.0, 2.0, -1.0, 0.0, 1.0, 2.0))
    self.assertEqual(len({buffers[0]["bytes"] for buffers in phases}), 4)
    self.assertEqual(phases[0][0]["bytes"], phases[0][1]["bytes"])

  def test_source_input_signature_preserves_parameter_order_and_layout_sizes(self):
    signature = study.input_signature_from_source(_CONTROL_SOURCE)

    self.assertEqual(
      signature,
      [
        {"name": "x", "dtype": "f32", "shape": [2, 2], "byte_length": 16},
        {"name": "y", "dtype": "f32", "shape": [2, 2], "byte_length": 16},
      ],
    )

  def test_proposal_case_updates_payload_hashes_and_sampler_provenance(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      template = root / "template"
      template.mkdir()
      input_bytes = struct.pack("<2f", -1.0, 2.0)
      expected_bytes = struct.pack("<2f", -1.0, 2.0)
      source_bytes = _SOURCE.encode()
      (template / "program.sir").write_bytes(source_bytes)
      (template / "inputs.bin").write_bytes(bytes(8))
      (template / "expected.bin").write_bytes(bytes(8))
      template_manifest = {
        "inputs": [{"name": "x", "dtype": "f32", "shape": [2], "byte_length": 8}],
        "output": {"byte_length": 8},
        "files": {
          "program.sir": {
            "bytes": len(source_bytes),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
          },
          "inputs.bin": {"bytes": 8, "sha256": hashlib.sha256(bytes(8)).hexdigest()},
          "expected.bin": {
            "bytes": 8,
            "sha256": hashlib.sha256(bytes(8)).hexdigest(),
          },
        },
      }
      (template / "case.json").write_text(json.dumps(template_manifest))

      case = study.materialize_proposal_case(
        template,
        root / "proposal",
        input_bytes=input_bytes,
        expected_bytes=expected_bytes,
        policy="boundary",
        sampler_version=study.BOUNDARY_SAMPLER_VERSION,
        proposal_id="kernel:boundary:0",
        corpus_sha256="frozen-digest",
        input_seed=17,
      )
      manifest = json.loads((case / "case.json").read_text())
      observed_inputs = (case / "inputs.bin").read_bytes()
      observed_expected = (case / "expected.bin").read_bytes()

    self.assertEqual(observed_inputs, input_bytes)
    self.assertEqual(observed_expected, expected_bytes)
    self.assertEqual(
      manifest["files"]["inputs.bin"]["sha256"], hashlib.sha256(input_bytes).hexdigest()
    )
    self.assertEqual(
      manifest["generator"]["input_sampler_version"], study.BOUNDARY_SAMPLER_VERSION
    )
    self.assertEqual(
      manifest["coverage"]["input_behavior"]["sampler_version"],
      study.BOUNDARY_SAMPLER_VERSION,
    )

  def test_control_catalogue_keeps_its_predeclared_ids(self):
    controls = study.build_control_catalogue(_CONTROL_SOURCE)

    self.assertEqual(
      list(controls),
      [
        "operand_substitution",
        "tensor_permutation",
        "reduction_axis",
        "broadcast_source",
        "negative",
      ],
    )
    self.assertEqual(controls["negative"]["source"], _CONTROL_SOURCE)
    self.assertEqual(controls["tensor_permutation"]["status"], "inapplicable")

  def test_operand_substitution_requires_equal_shape_and_changes_only_operand(self):
    controls = study.build_control_catalogue(_CONTROL_SOURCE)
    mutation = controls["operand_substitution"]["source"]
    self.assertIn("tensor add %y, %y", mutation)
    self.assertIn("tensor sum %sum, 1, drop", mutation)

    no_alternative = study.build_control_catalogue(_SOURCE)
    self.assertEqual(no_alternative["operand_substitution"]["status"], "inapplicable")

  def test_reduction_and_broadcast_controls_preserve_output_shapes(self):
    controls = study.build_control_catalogue(_CONTROL_SOURCE)

    self.assertIn("tensor sum %sum, 0, drop", controls["reduction_axis"]["source"])
    self.assertIn("tensor broadcast %x, [2, 2]", controls["broadcast_source"]["source"])

  def test_tensor_permutation_control_requires_shape_preserving_square_input(self):
    square = study.build_control_catalogue(_TRANSPOSE_SOURCE)
    nonsquare_source = _TRANSPOSE_SOURCE.replace("[2, 2]", "[2, 4]").replace(
      "[1, 0]", "[1, 0]"
    )
    nonsquare = study.build_control_catalogue(nonsquare_source)

    self.assertIn("tensor transpose %x, [0, 1]", square["tensor_permutation"]["source"])
    self.assertEqual(nonsquare["tensor_permutation"]["status"], "inapplicable")

  def test_generated_style_shape_tracking_handles_scalar_ops_and_aliases(self):
    states, _parameters = study._shape_before_lines(_GENERATED_STYLE_SOURCE)
    lines = _GENERATED_STYLE_SOURCE.splitlines()
    use_line = next(
      index for index, line in enumerate(lines) if "%t2 = tensor add" in line
    )

    self.assertEqual(states[use_line]["%t0"], (2, 2))
    self.assertEqual(states[use_line]["%tCopy"], (2, 2))

  def test_generated_style_controls_use_declared_intermediate_shapes(self):
    controls = study.build_control_catalogue(_GENERATED_STYLE_SOURCE)

    self.assertEqual(
      {
        control_id: controls[control_id]["status"]
        for control_id in (
          "operand_substitution",
          "tensor_permutation",
          "reduction_axis",
          "broadcast_source",
        )
      },
      {
        control_id: "available"
        for control_id in (
          "operand_substitution",
          "tensor_permutation",
          "reduction_axis",
          "broadcast_source",
        )
      },
    )

  def test_control_replay_separates_invalid_and_distinguished_outcomes(self):
    expected = struct.pack("<2f", 1.0, 2.0)

    def checker(source):
      return SimpleNamespace(
        returncode=1 if source == "invalid" else 0,
        stdout="type error",
        stderr="",
      )

    def reference_runner(source, _input_bytes, **_kwargs):
      data = struct.pack("<2f", 20.0, -30.0) if source == "different" else expected
      return SimpleNamespace(returncode=0, stdout="", stderr=""), data

    invalid = study.evaluate_control(
      {"control_id": "test", "status": "available", "source": "invalid"},
      original_source="original",
      input_bytes=bytes(8),
      expected_bytes=expected,
      output_shape=[2],
      checker=checker,
      reference_runner=reference_runner,
    )
    distinguished = study.evaluate_control(
      {"control_id": "test", "status": "available", "source": "different"},
      original_source="original",
      input_bytes=bytes(8),
      expected_bytes=expected,
      output_shape=[2],
      checker=checker,
      reference_runner=reference_runner,
    )
    negative = study.evaluate_control(
      {"control_id": "negative", "status": "available", "source": "original"},
      original_source="original",
      input_bytes=bytes(8),
      expected_bytes=expected,
      output_shape=[2],
      checker=checker,
      reference_runner=reference_runner,
    )

    self.assertEqual(invalid["status"], "invalid_program")
    self.assertEqual(distinguished["status"], "distinguished")
    self.assertEqual(negative["status"], "not_distinguished")

  def test_input_metrics_count_buffer_equality_repeated_rows_and_uniques(self):
    repeated = struct.pack("<4f", -1.0, 0.0, -1.0, 0.0).hex()
    proposals = [
      {
        "arm": "boundary",
        "proposal": {
          "proposal_status": "accepted",
          "input_bytes_hex": repeated + repeated,
          "input_buffers": [
            {"name": "x", "dtype": "f32", "shape": [2, 2], "bytes_hex": repeated},
            {"name": "y", "dtype": "f32", "shape": [2, 2], "bytes_hex": repeated},
          ],
        },
        "result": {"output_bytes_hex": repeated},
      }
    ]

    summary = study.summarize_input_patterns(proposals)
    boundary = summary["arms"]["boundary"]

    self.assertEqual(boundary["input_buffer_pair_comparisons"], 1)
    self.assertEqual(boundary["equal_input_buffer_pairs"], 1)
    self.assertEqual(boundary["repeated_rows"], 2)
    self.assertEqual(boundary["unique_input_patterns"], 1)
    self.assertEqual(boundary["unique_output_patterns"], 1)

  def test_paired_four_proposal_union_counts_wins_losses_and_ties(self):
    corpus = study.freeze_corpus(
      [
        {"seed": 2027, "status": "candidate", "source": _SOURCE},
        {"seed": 2028, "status": "candidate", "source": _ADDED},
      ],
      config=_CONFIG,
      base_revision="base-revision",
      target_count=2,
    )
    ledger = study.new_study_ledger(corpus, proposals_per_arm=4)
    for kernel_index, kernel in enumerate(corpus["kernels"]):
      for slot in ledger["proposals"]:
        if slot["kernel_id"] != kernel["kernel_id"]:
          continue
        detected_arm = "boundary" if kernel_index == 0 else "legacy_cycle"
        for control_id in study.CONTROL_IDS:
          ledger["controls"].append(
            {
              "kernel_id": slot["kernel_id"],
              "proposal_id": slot["proposal_id"],
              "arm": slot["arm"],
              "control_id": control_id,
              "status": (
                "distinguished"
                if control_id == "operand_substitution"
                and slot["arm"] == detected_arm
                and slot["proposal_index"] == 0
                else "not_distinguished"
                if control_id == "operand_substitution"
                else "inapplicable"
              ),
            }
          )

    summary = study.paired_sensitivity_summary(ledger)

    self.assertEqual(summary["paired_kernels"], 2)
    self.assertEqual(summary["wins"], 1)
    self.assertEqual(summary["losses"], 1)
    self.assertEqual(summary["ties"], 0)
    self.assertEqual(
      summary["per_control"]["operand_substitution"]["boundary_detected"], 1
    )

  def test_paired_detection_keeps_unclassifiable_kernels_in_denominator(self):
    ledger = study.new_study_ledger(_corpus(), proposals_per_arm=4)
    for slot in ledger["proposals"]:
      for control_id in study.CONTROL_IDS:
        ledger["controls"].append(
          {
            "kernel_id": slot["kernel_id"],
            "proposal_id": slot["proposal_id"],
            "arm": slot["arm"],
            "control_id": control_id,
            "status": "inapplicable",
          }
        )
    summary = study.paired_sensitivity_summary(ledger)

    self.assertEqual(summary["kernel_count"], 1)
    self.assertEqual(summary["paired_kernels"], 0)
    self.assertEqual(summary["unclassifiable_kernels"], 1)

  def test_study_summary_reports_reference_and_compiled_costs_once(self):
    ledger = study.new_study_ledger(_corpus(), proposals_per_arm=4)
    slot = ledger["proposals"][0]
    study.record_proposal_result(
      ledger,
      slot["proposal_id"],
      {
        "proposal_status": "accepted",
        "cost_s": {"reference": 0.25, "compiled": 1.5},
        "gpu": {
          "compiled": True,
          "executed": True,
          "compilation_count": 1,
          "execution_count": 1,
        },
      },
    )

    summary = study.summarize_study(ledger)

    self.assertEqual(summary["cost_s"]["reference"], 0.25)
    self.assertEqual(summary["cost_s"]["compiled"], 1.5)
    self.assertEqual(summary["gpu_work"]["compilation_count"], 1)
    self.assertEqual(summary["gpu_work"]["execution_count"], 1)

  def test_study_summary_separates_unknown_and_skipped_gpu_work(self):
    ledger = study.new_study_ledger(_corpus(), proposals_per_arm=4)
    unknown_slot, skipped_slot = ledger["proposals"][:2]
    study.record_proposal_result(
      ledger,
      unknown_slot["proposal_id"],
      {
        "proposal_status": "timeout",
        "gpu": {
          "compilation_count": None,
          "execution_count": None,
          "compilation_state": "unknown",
          "execution_state": "unknown",
        },
      },
    )
    study.record_proposal_result(
      ledger,
      skipped_slot["proposal_id"],
      {
        "proposal_status": "invalid_on_input",
        "gpu": {
          "compilation_count": 0,
          "execution_count": 0,
          "compilation_state": "skipped",
          "execution_state": "skipped",
        },
      },
    )

    summary = study.summarize_study(ledger)

    self.assertEqual(summary["gpu_work"]["compilation_count"], 0)
    self.assertEqual(summary["gpu_work"]["execution_count"], 0)
    self.assertEqual(summary["gpu_work"]["compilation_unknown_count"], 1)
    self.assertEqual(summary["gpu_work"]["execution_unknown_count"], 1)
    self.assertEqual(summary["gpu_work"]["compilation_skipped_count"], 1)
    self.assertEqual(summary["gpu_work"]["execution_skipped_count"], 1)

  def test_study_summary_includes_control_replay_cost(self):
    ledger = study.new_study_ledger(_corpus(), proposals_per_arm=4)
    slot = ledger["proposals"][0]
    study.record_control_result(
      ledger,
      slot["proposal_id"],
      {
        "control_id": "negative",
        "status": "not_distinguished",
        "elapsed_s": 0.75,
      },
    )

    summary = study.summarize_study(ledger)

    self.assertEqual(summary["cost_s"]["control"], 0.75)

  def test_study_summary_does_not_turn_unknown_gpu_duration_into_zero(self):
    ledger = study.new_study_ledger(_corpus(), proposals_per_arm=4)
    slot = ledger["proposals"][0]
    study.record_proposal_result(
      ledger,
      slot["proposal_id"],
      {
        "proposal_status": "accepted",
        "gpu": {
          "compilation_count": None,
          "execution_count": None,
          "compilation_state": "unknown",
          "execution_state": "unknown",
        },
        "cost_s": {"reference": 0.1, "compiled": None},
      },
    )

    summary = study.summarize_study(ledger)

    self.assertEqual(summary["cost_s"]["compiled"], 0.0)
    self.assertEqual(summary["cost_unknown_count"]["compiled"], 1)


class TensorBoundarySensitivityNativeTests(unittest.TestCase):
  @unittest.skipUnless(
    os.environ.get("REFRACTIR_TENSOR_INPUT_POLICY_API"),
    "native tensor input-policy API is not configured",
  )
  def test_native_builder_freezes_checked_source_and_boundary_proposal(self):
    api = os.environ["REFRACTIR_TENSOR_INPUT_POLICY_API"]
    candidates = study.generate_candidate_schedule(
      api,
      target_count=1,
      start_seed=2027,
      max_candidates=1,
    )
    corpus = study.freeze_corpus(
      candidates,
      config=_CONFIG,
      base_revision="native-test-revision",
      target_count=1,
    )
    kernel = corpus["kernels"][0]
    proposal = study.propose_boundary_inputs(api, kernel["source"], 91)

    self.assertEqual(proposal["status"], "accepted")
    self.assertEqual(proposal["sampler_version"], study.BOUNDARY_SAMPLER_VERSION)
    self.assertTrue(proposal["buffers"])
    for buffer in proposal["buffers"]:
      data = buffer["bytes"]
      values = struct.unpack(f"<{len(data) // 4}f", data)
      self.assertTrue(all(value in {-1.0, 0.0, 1.0, 2.0} for value in values))
