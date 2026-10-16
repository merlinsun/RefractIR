"""Contracts for selectable RefractIR tensor campaign routes."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from test.tensor.campaigns import continuous_triton_campaign as campaign
from test.tensor.campaigns.continuous_triton_campaign import (
  build_generation_command,
  parse_arguments,
)


class ContinuousTensorCampaignRouteTests(unittest.TestCase):
  def test_constructor_baseline_is_explicit(self):
    command = build_generation_command(
      "/repo/rysmith",
      "/tmp/cases",
      41,
      generator="constructor",
      input_policy="ordinary",
    )

    self.assertIn("--tensor-generator", command)
    self.assertEqual(command[command.index("--tensor-generator") + 1], "constructor")
    self.assertEqual(command[command.index("--tensor-input-policy") + 1], "ordinary")

  def test_compositional_boundary_route_is_passed_to_rysmith(self):
    command = build_generation_command(
      "/repo/rysmith",
      "/tmp/cases",
      42,
      generator="compositional",
      input_policy="boundary",
    )

    self.assertEqual(command[command.index("--tensor-generator") + 1], "compositional")
    self.assertEqual(command[command.index("--tensor-input-policy") + 1], "boundary")

  def test_exact_policy_forwards_scalar_property_target(self):
    command = build_generation_command(
      "/repo/rysmith",
      "/tmp/cases",
      43,
      generator="compositional",
      input_policy="exact",
      dtype="i32",
      property_target=18,
    )

    self.assertEqual(command[command.index("--tensor-property-target") + 1], "18")

  def test_exact_policy_forwards_every_output_element_target(self):
    command = build_generation_command(
      "/repo/rysmith",
      "/tmp/cases",
      44,
      generator="compositional",
      input_policy="exact",
      dtype="i32",
      output_elements=("0,1=-1", "1,0=3"),
    )

    targets = [
      command[index + 1]
      for index, argument in enumerate(command[:-1])
      if argument == "--tensor-output-element"
    ]
    self.assertEqual(targets, ["0,1=-1", "1,0=3"])

  def test_exact_policy_without_a_target_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "requires exactly one target form"):
      build_generation_command(
        "/repo/rysmith",
        "/tmp/cases",
        45,
        generator="compositional",
        input_policy="exact",
        dtype="i32",
      )

  def test_exact_policy_rejects_both_target_forms(self):
    with self.assertRaisesRegex(ValueError, "requires exactly one target form"):
      build_generation_command(
        "/repo/rysmith",
        "/tmp/cases",
        46,
        generator="compositional",
        input_policy="exact",
        dtype="i32",
        property_target=18,
        output_elements=("0,0=18",),
      )

  def test_cli_can_isolate_workers_to_a_selected_gpu_set(self):
    args = parse_arguments(
      [
        "--generator",
        "compositional",
        "--input-policy",
        "boundary",
        "--gpu-ids",
        "4,5,6,7",
        "--workers",
        "4",
      ]
    )

    self.assertEqual(args.gpu_ids, (4, 5, 6, 7))
    self.assertEqual(args.workers, 4)


def _case_fixture(root, *, case_id="case-0001", include_ir=True):
  case_dir = Path(root) / case_id
  case_dir.mkdir(parents=True)
  source = """fun @kernel(%x: tensor [2] f32) : tensor [2] f32 {
^entry:
  %out = tensor add %x, %x;
  ret %out;
}
"""
  inputs = bytes.fromhex("0000803f00000040")
  expected = bytes.fromhex("0000004000008040")
  (case_dir / "program.sir").write_text(source, encoding="utf-8")
  (case_dir / "inputs.bin").write_bytes(inputs)
  (case_dir / "expected.bin").write_bytes(expected)
  manifest = {
    "schema": "refractir.tensor-case.v2",
    "profile": "tensor-v2",
    "entry": "kernel",
    "grid": [1, 1, 1],
    "launch": {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False},
    "inputs": [
      {
        "name": "x",
        "dtype": "f32",
        "shape": [2],
        "byte_offset": 0,
        "byte_length": 8,
        "element_strides": [1],
        "element_offset": 0,
        "storage_span": 2,
        "axis_order": [0],
      }
    ],
    "output": {
      "dtype": "f32",
      "shape": [2],
      "byte_length": 8,
      "element_strides": [1],
      "element_offset": 0,
      "storage_span": 2,
      "axis_order": [0],
    },
    "comparison": {"mode": "numeric", "atol": [2, 100000], "rtol": [2, 1000]},
    "generator": {"seed": 2027},
  }
  (case_dir / "case.json").write_text(json.dumps(manifest), encoding="utf-8")
  if include_ir:
    compiled = case_dir / "runs" / "compiled"
    compiled.mkdir(parents=True)
    (compiled / "run.json").write_text(
      json.dumps(
        {
          "launch": manifest["launch"],
          "grid": manifest["grid"],
          "compiler": {"executable": "/repo/symirc", "sha256": "compiler-hash"},
        }
      ),
      encoding="utf-8",
    )
    (compiled / "environment.json").write_text(
      json.dumps(
        {
          "target_arch": "sm_70",
          "triton_source_revision": "triton-revision",
        }
      ),
      encoding="utf-8",
    )
    (compiled / "compiler-ttgir.ir").write_text(
      """#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module {
  %0 = ttg.convert_layout %arg0 : tensor<2xf32, #blocked> -> tensor<2xf32, #blocked>
  %1 = tt.reduce %0, %cst : (tensor<2xf32, #blocked>, tensor<2xf32, #blocked>) -> tensor<f32>
  %2 = arith.addf %arg0, %arg1 : f32
  %3 = arith.addf %arg0, %arg1 : f32
  %4 = ttg.convert_layout %0 : tensor<2xf32, #blocked> -> tensor<2xf32, #blocked>
  scf.for %i = %c0 to %c2 step %c1 {
    scf.yield
  }
}
""",
      encoding="utf-8",
    )
    (compiled / "compiler-ptx.ir").write_text(
      ".visible .entry kernel() {\n"
      "  shfl.sync.bfly.b32 %r1, %r2, 1, 31, -1;\n"
      "  st.shared.b32 [%r3], %r1;\n"
      "  ld.shared.b32 %r4, [%r3];\n"
      "}\n",
      encoding="utf-8",
    )
  return case_dir


def _passed_event(case_dir, *, case_id="case-0001"):
  def digest(name):
    return hashlib.sha256((Path(case_dir) / name).read_bytes()).hexdigest()

  return {
    "schema": "refractir.continuous-triton-pass.v1",
    "passed_at": "2026-10-10T00:00:00+00:00",
    "seed": 2027,
    "case_id": case_id,
    "source_sha256": digest("program.sir"),
    "manifest_sha256": digest("case.json"),
    "inputs_sha256": digest("inputs.bin"),
    "expected_sha256": digest("expected.bin"),
    "compiled_status": "pass",
    "campaign_valid": True,
  }


class ContinuousTensorCampaignSummaryTests(unittest.TestCase):
  def test_successful_cleanup_leaves_durable_case_summary(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      campaign_dir = root / "campaign"
      seed_dir = campaign_dir / "seed-2027"
      case_dir = _case_fixture(seed_dir)
      event = _passed_event(case_dir)

      recorded = campaign.finalize_passed_case(
        seed_dir,
        "case-0001",
        case_dir,
        event,
        campaign_dir=campaign_dir,
        source_revision="refractir-revision",
        triton_build="3.9.0+test",
      )

      summary_path = Path(recorded["case_summary_path"])
      self.assertFalse(case_dir.exists())
      self.assertTrue(summary_path.is_file())
      summary = json.loads(summary_path.read_text(encoding="utf-8"))
      self.assertEqual(summary["case_identity"]["case_id"], "case-0001")
      self.assertEqual(
        summary["compiler_observations"]["runs"]["runs/compiled"]["stages"]["ttgir"][
          "status"
        ],
        "available",
      )
      ttgir = summary["compiler_observations"]["runs"]["runs/compiled"]
      stage = ttgir["stages"]["ttgir"]
      self.assertTrue(stage["raw_sha256"])
      self.assertTrue(stage["normalized_sha256"])
      self.assertEqual(stage["observations"]["operation_counts"]["tt.reduce"], 1)
      self.assertEqual(stage["observations"]["loop_construct_count"], 1)
      self.assertTrue(stage["observations"]["reused_values"])
      self.assertTrue(stage["observations"]["duplicate_expressions"])
      self.assertTrue(ttgir["ttgir_layout_encodings"])
      ptx = ttgir["stages"]["ptx"]["observations"]
      self.assertEqual(ptx["warp_shuffle_count"], 1)
      self.assertEqual(ptx["shared_memory_transfer_count"], 2)
      archive = json.loads(
        (campaign_dir / "representatives" / "index.json").read_text(encoding="utf-8")
      )
      self.assertEqual(
        archive["buckets"][summary["behavior_bucket"]]["case_summary_sha256"],
        hashlib.sha256(summary_path.read_bytes()).hexdigest(),
      )

  def test_summary_persistence_failure_retains_case_and_pass_state(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      campaign_dir = root / "campaign"
      seed_dir = campaign_dir / "seed-2027"
      case_dir = _case_fixture(seed_dir)
      event = _passed_event(case_dir)

      with patch.object(
        campaign, "persist_case_summary", side_effect=OSError("disk full")
      ):
        with self.assertRaisesRegex(OSError, "disk full"):
          campaign.finalize_passed_case(
            seed_dir,
            "case-0001",
            case_dir,
            event,
            campaign_dir=campaign_dir,
            source_revision="refractir-revision",
            triton_build="3.9.0+test",
          )

      self.assertTrue(case_dir.is_dir())
      self.assertFalse((seed_dir / ".passed-cases").exists())
      self.assertFalse((campaign_dir / "passes.jsonl").exists())

  def test_restart_after_summary_publication_reuses_summary_and_archive(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      campaign_dir = root / "campaign"
      seed_dir = campaign_dir / "seed-2027"
      case_dir = _case_fixture(seed_dir)
      event = _passed_event(case_dir)
      first = campaign.persist_case_summary(
        case_dir,
        campaign_dir,
        event,
        source_revision="refractir-revision",
        triton_build="3.9.0+test",
      )

      recorded = campaign.finalize_passed_case(
        seed_dir,
        "case-0001",
        case_dir,
        event,
        campaign_dir=campaign_dir,
        source_revision="refractir-revision",
        triton_build="3.9.0+test",
      )
      recovered = campaign.recover_passed_case(
        seed_dir,
        "case-0001",
        None,
        campaign_dir=campaign_dir,
        source_revision="refractir-revision",
        triton_build="3.9.0+test",
      )

      self.assertEqual(first["case_summary_sha256"], recorded["case_summary_sha256"])
      self.assertEqual(
        recorded["case_summary_sha256"], recovered["case_summary_sha256"]
      )
      pass_records = (
        (campaign_dir / "passes.jsonl").read_text(encoding="utf-8").splitlines()
      )
      self.assertEqual(len(pass_records), 1)
      archive_index = json.loads(
        (campaign_dir / "representatives" / "index.json").read_text(encoding="utf-8")
      )
      self.assertEqual(len(archive_index["buckets"]), 1)

  def test_existing_pass_marker_requires_summary_before_cleanup(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      campaign_dir = root / "campaign"
      seed_dir = campaign_dir / "seed-2027"
      case_dir = _case_fixture(seed_dir)
      event = _passed_event(case_dir)
      seed_dir.joinpath(".passed-cases").write_text("case-0001\n", encoding="utf-8")
      campaign_dir.mkdir(parents=True, exist_ok=True)
      seed_dir.parent.mkdir(parents=True, exist_ok=True)
      (campaign_dir / "passes.jsonl").write_text(
        json.dumps(event) + "\n", encoding="utf-8"
      )

      recovered = campaign.recover_passed_case(
        seed_dir,
        "case-0001",
        case_dir,
        campaign_dir=campaign_dir,
        source_revision="refractir-revision",
        triton_build="3.9.0+test",
      )

      self.assertFalse(case_dir.exists())
      self.assertTrue(Path(recovered["case_summary_path"]).is_file())
      updated_event = json.loads(
        (campaign_dir / "passes.jsonl").read_text(encoding="utf-8").splitlines()[0]
      )
      self.assertEqual(
        updated_event["case_summary_sha256"], recovered["case_summary_sha256"]
      )

  def test_missing_compiler_ir_is_unavailable_not_zero_observations(self):
    with tempfile.TemporaryDirectory() as directory:
      case_dir = _case_fixture(directory, include_ir=False)
      summary = campaign.build_compiler_summary(
        case_dir,
        _passed_event(case_dir),
        source_revision="refractir-revision",
        triton_build="3.9.0+test",
      )

    stages = summary["compiler_observations"]["runs"]["case-artifacts"]["stages"]
    self.assertEqual(stages["ttir"]["status"], "unavailable")
    self.assertNotIn("operation_counts", stages["ttir"])
    self.assertIn("ttir", summary["compiler_observations"]["unavailable_stages"])

  def test_historical_pass_without_case_artifacts_is_marked_unavailable(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      campaign_dir = root / "campaign"
      seed_dir = campaign_dir / "seed-2027"
      case_dir = seed_dir / "case-0001"
      seed_dir.mkdir(parents=True)
      event = {
        "schema": "refractir.continuous-triton-pass.v1",
        "passed_at": "2026-10-10T00:00:00+00:00",
        "seed": 2027,
        "case_id": "case-0001",
        "source_sha256": "historical-source-hash",
        "manifest_sha256": "historical-manifest-hash",
        "inputs_sha256": "historical-input-hash",
        "expected_sha256": "historical-expected-hash",
      }
      seed_dir.joinpath(".passed-cases").write_text("case-0001\n", encoding="utf-8")
      campaign_dir.mkdir(parents=True, exist_ok=True)
      (campaign_dir / "passes.jsonl").write_text(
        json.dumps(event) + "\n", encoding="utf-8"
      )

      recovered = campaign.recover_passed_case(
        seed_dir,
        "case-0001",
        case_dir,
        campaign_dir=campaign_dir,
        source_revision="refractir-revision",
        triton_build="3.9.0+test",
      )

      summary = json.loads(
        Path(recovered["case_summary_path"]).read_text(encoding="utf-8")
      )
      self.assertEqual(summary["status"], "historical_artifacts_unavailable")
      self.assertEqual(
        summary["missing_case_artifacts"],
        ["case.json", "program.sir", "inputs.bin", "expected.bin"],
      )
      self.assertEqual(
        set(summary["compiler_observations"]["unavailable_stages"]),
        set(campaign.SUMMARY_STAGES),
      )
      self.assertFalse(case_dir.exists())

  def test_concurrent_worker_results_are_serialized_by_parent(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      campaign_dir = root / "campaign"
      seed_dir = campaign_dir / "seed-2027"
      cases = {
        case_id: _case_fixture(seed_dir, case_id=case_id)
        for case_id in ("case-0001", "case-0002")
      }

      def worker_result(case_id):
        return {
          "passed": True,
          "seed": 2027,
          "case_id": case_id,
          "case_dir": str(cases[case_id]),
          "event": _passed_event(cases[case_id], case_id=case_id),
        }

      with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(worker_result, cases))
      completed = [
        campaign.finalize_parent_result(
          seed_dir,
          result,
          campaign_dir=campaign_dir,
          source_revision="refractir-revision",
          triton_build="3.9.0+test",
        )
        for result in results
      ]

      self.assertEqual(len(completed), 2)
      self.assertTrue(
        all(not Path(result["case_dir"]).exists() for result in completed)
      )
      events = (campaign_dir / "passes.jsonl").read_text(encoding="utf-8").splitlines()
      self.assertEqual(len(events), 2)
      identities = [json.loads(line)["case_summary_sha256"] for line in events]
      self.assertEqual(len(set(identities)), 2)
      archive = json.loads(
        (campaign_dir / "representatives" / "index.json").read_text(encoding="utf-8")
      )
      self.assertEqual(len(archive["buckets"]), 1)

  def test_representative_archive_reports_quota_exhaustion(self):
    with (
      tempfile.TemporaryDirectory() as directory,
      patch.object(campaign, "REPRESENTATIVE_QUOTA", 1),
    ):
      root = Path(directory)
      campaign_dir = root / "campaign"
      seed_dir = campaign_dir / "seed-2027"
      first_case = _case_fixture(seed_dir, case_id="case-0001")
      second_case = _case_fixture(seed_dir, case_id="case-0002")
      second_ir = second_case / "runs" / "compiled" / "compiler-ttgir.ir"
      second_ir.write_text(
        second_ir.read_text(encoding="utf-8").replace("tt.reduce", "ttg.gather"),
        encoding="utf-8",
      )

      campaign.persist_case_summary(
        first_case,
        campaign_dir,
        _passed_event(first_case),
        source_revision="refractir-revision",
        triton_build="3.9.0+test",
      )
      second = campaign.persist_case_summary(
        second_case,
        campaign_dir,
        _passed_event(second_case, case_id="case-0002"),
        source_revision="refractir-revision",
        triton_build="3.9.0+test",
      )

      self.assertEqual(second["representative_archive"]["status"], "quota_exhausted")
      self.assertTrue(Path(second["case_summary_path"]).is_file())
      archive = json.loads(
        (campaign_dir / "representatives" / "index.json").read_text(encoding="utf-8")
      )
      self.assertEqual(len(archive["buckets"]), 1)


class CampaignPassDumpIsolationTests(unittest.TestCase):
  def test_a_case_without_pass_dumps_is_explicitly_unavailable(self):
    record = self.summary({})
    self.assertEqual(record["status"], "unavailable")
    self.assertEqual(record["files"], {})
    self.assertIn("not saved", record["reason"])

  @staticmethod
  def snapshot(name):
    return (
      f"// -----// IR Dump Before {name}: {name.lower()} "
      "('builtin.module' operation) //----- //\nmodule {}\n"
    ).encode()

  def summary(self, dumps):
    with tempfile.TemporaryDirectory() as directory:
      case_dir = _case_fixture(Path(directory))
      for relative, payload in dumps.items():
        path = case_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
      summary = campaign.build_compiler_summary(
        case_dir, _passed_event(case_dir), source_revision="review", triton_build="test"
      )
      return summary["compiler_observations"]["pass_deltas"]

  def test_separate_runs_cannot_complete_each_others_pass_dump(self):
    record = self.summary(
      {
        "runs/a/pass-dump.txt": self.snapshot("PassA"),
        "runs/b/pass-dump.txt": self.snapshot("LLVMDIScope"),
      }
    )
    self.assertNotEqual(record["status"], "available")
    self.assertIn("files", record)
    self.assertEqual(record["files"]["runs/a/pass-dump.txt"]["status"], "unavailable")
    self.assertEqual(
      record["files"]["runs/b/pass-dump.txt"]["compilations"][0]["transitions"], []
    )

  def test_truncated_dump_does_not_erase_a_complete_run(self):
    record = self.summary(
      {
        "runs/a/pass-dump.txt": self.snapshot("PassA") + self.snapshot("LLVMDIScope"),
        "runs/b/pass-dump.txt": self.snapshot("PassB"),
      }
    )
    self.assertEqual(record["status"], "partial")
    self.assertEqual(record["compilation_count"], 1)
    self.assertEqual(record["files"]["runs/a/pass-dump.txt"]["status"], "available")

  def test_undecodable_dump_does_not_erase_a_complete_run(self):
    record = self.summary(
      {
        "runs/a/pass-dump.txt": self.snapshot("PassA") + self.snapshot("LLVMDIScope"),
        "runs/b/pass-dump.txt": b"\xff",
      }
    )
    self.assertEqual(record["status"], "partial")
    self.assertEqual(record["files"]["runs/b/pass-dump.txt"]["status"], "unavailable")
    self.assertEqual(record["compilation_count"], 1)

  def test_complete_runs_keep_their_own_transition_and_file_identity(self):
    dumps = {
      f"runs/{name}/pass-dump.txt": self.snapshot(name) + self.snapshot("LLVMDIScope")
      for name in ("PassA", "PassB")
    }
    record = self.summary(dumps)
    self.assertIn("files", record)
    self.assertEqual(record["status"], "available")
    self.assertEqual(record["compilation_count"], 2)
    for name in ("PassA", "PassB"):
      relative = f"runs/{name}/pass-dump.txt"
      item = record["files"][relative]
      self.assertEqual(item["raw_sha256"], hashlib.sha256(dumps[relative]).hexdigest())
      self.assertEqual(item["compilations"][0]["transitions"][0]["pass"], name)

  def test_unavailable_dump_keeps_its_digest_after_case_cleanup(self):
    data = self.snapshot("PassA")
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      seed = root / "seed-2027"
      case_dir = _case_fixture(seed)
      dump = case_dir / "runs" / "compiled" / "pass-dump.txt"
      dump.write_bytes(data)
      result = campaign.finalize_passed_case(
        seed,
        case_dir.name,
        case_dir,
        _passed_event(case_dir),
        campaign_dir=root,
        source_revision="review",
        triton_build="test",
      )
      self.assertFalse(case_dir.exists())
      summary = json.loads(Path(result["case_summary_path"]).read_text())
      record = summary["compiler_observations"]["pass_deltas"]
      self.assertIn("files", record)
      item = record["files"]["runs/compiled/pass-dump.txt"]
      self.assertEqual(item["status"], "unavailable")
      self.assertEqual(item["raw_sha256"], hashlib.sha256(data).hexdigest())


class CampaignProvenanceTests(unittest.TestCase):
  def test_an_explicit_clean_snapshot_is_preserved(self):
    with patch.dict(os.environ, {"REFRACTIR_WORKTREE_DIRTY": "false"}, clear=True):
      self.assertEqual(
        campaign.prepare_environment()["REFRACTIR_WORKTREE_DIRTY"], "false"
      )

  def test_missing_flags_are_not_replaced_with_a_build_claim(self):
    with patch.dict(os.environ, {}, clear=True):
      self.assertEqual(
        campaign.prepare_environment()["REFRACTIR_COMPILER_FLAGS"], "unavailable"
      )

  def test_explicit_flags_reach_the_runner_unchanged(self):
    with patch.dict(
      os.environ, {"REFRACTIR_COMPILER_FLAGS": "clang++ -O1"}, clear=True
    ):
      self.assertEqual(
        campaign.prepare_environment()["REFRACTIR_COMPILER_FLAGS"], "clang++ -O1"
      )

  def test_a_dirty_snapshot_is_not_reported_as_clean(self):
    with patch.dict(os.environ, {"REFRACTIR_WORKTREE_DIRTY": "true"}, clear=True):
      self.assertEqual(
        campaign.prepare_environment()["REFRACTIR_WORKTREE_DIRTY"], "true"
      )

  def test_an_unmeasured_snapshot_is_unknown(self):
    with patch.dict(os.environ, {}, clear=True):
      self.assertEqual(
        campaign.prepare_environment()["REFRACTIR_WORKTREE_DIRTY"], "unknown"
      )

  def test_missing_revision_does_not_borrow_the_old_campaign_revision(self):
    env = {
      key: value for key, value in os.environ.items() if key != "REFRACTIR_REVISION"
    }
    result = subprocess.run(
      [
        sys.executable,
        "-c",
        "from test.tensor.campaigns.continuous_triton_campaign import REVISION; print(REVISION)",
      ],
      cwd=Path(__file__).resolve().parents[2],
      env=env,
      capture_output=True,
      text=True,
      check=True,
    )
    self.assertEqual(result.stdout.strip(), "unavailable")


class CampaignSourceAttributionTests(unittest.TestCase):
  def test_absent_ir_keeps_source_attribution_unavailable(self):
    with tempfile.TemporaryDirectory() as directory:
      case_dir = _case_fixture(Path(directory), include_ir=False)
      summary = campaign.build_compiler_summary(
        case_dir, _passed_event(case_dir), source_revision="review", triton_build="test"
      )
      self.assertEqual(
        summary["compiler_observations"]["source_attribution"]["status"], "unavailable"
      )

  def check_unmapped_source(self, operation):
    with tempfile.TemporaryDirectory() as directory:
      case_dir = _case_fixture(Path(directory))
      source = (
        (case_dir / "program.sir")
        .read_text()
        .replace("tensor add", f"tensor {operation}")
      )
      (case_dir / "program.sir").write_text(source)
      ir = case_dir / "runs" / "compiled" / "compiler-ttgir.ir"
      text = ir.read_text().replace(
        "-> tensor<2xf32, #blocked>",
        "-> tensor<2xf32, #blocked> loc(#loc1)",
      )
      ir.write_text('#loc1 = loc("kernel.py":3:5)\n' + text)
      summary = campaign.build_compiler_summary(
        case_dir, _passed_event(case_dir), source_revision="review", triton_build="test"
      )
      conversions = summary["compiler_observations"]["runs"]["runs/compiled"][
        "observation"
      ]["layout_conversions"]
      self.assertTrue(conversions)
      self.assertTrue(any(item["source_line"] == 3 for item in conversions))
      self.assertTrue(all(item["source_operation"] is None for item in conversions))

  def test_python_line_is_not_a_sir_add_attribution(self):
    self.check_unmapped_source("add")

  def test_python_line_is_not_a_sir_sub_attribution(self):
    self.check_unmapped_source("sub")

  def test_python_line_is_not_a_sir_mul_attribution(self):
    self.check_unmapped_source("mul")

  def test_python_line_is_not_a_sir_min_attribution(self):
    self.check_unmapped_source("min")

  def test_python_line_is_not_a_sir_max_attribution(self):
    self.check_unmapped_source("max")


if __name__ == "__main__":
  unittest.main()
