"""Restart and accounting contracts for the bounded study driver."""

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test.lib import tensor_boundary_sensitivity as study
from test.tensor import boundary_sensitivity_study as driver

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
  "comparison": {"mode": "numeric", "atol": [2, 100000], "rtol": [2, 1000]},
  "launch": {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False},
  "physical_layouts": {
    "policy": "contiguous-row-major-v1",
    "buffer_offsets": 0,
    "grid": [1, 1, 1],
  },
  "source_snapshot_sha256": "snapshot-v1",
}
_INPUTS = [{"name": "x", "dtype": "f32", "shape": [2], "byte_length": 8}]


class StudyProvenanceTests(unittest.TestCase):
  def configured_environment(self, supplied):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      executable = root / "tool"
      executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
      executable.chmod(0o700)
      with patch.dict(os.environ, supplied, clear=True):
        driver.configure_runtime(
          driver.SOURCE_ROOT,
          symiri=executable,
          symirc=executable,
          revision="reviewed-revision",
          compiler_flags="recorded flags",
        )
        return dict(os.environ)

  def test_cli_without_build_flags_records_unavailable(self):
    with patch.dict(os.environ, {}, clear=True):
      args = driver.parse_arguments(["--output-dir", "/tmp/study"])
    self.assertEqual(args.compiler_flags, "unavailable")

  def test_cli_preserves_supplied_compiler_environment(self):
    with patch.dict(
      os.environ, {"REFRACTIR_COMPILER_FLAGS": "clang++ -O0"}, clear=True
    ):
      args = driver.parse_arguments(["--output-dir", "/tmp/study"])
    self.assertEqual(args.compiler_flags, "clang++ -O0")

  def test_explicit_clean_state_is_preserved(self):
    environment = self.configured_environment({"REFRACTIR_WORKTREE_DIRTY": "false"})
    self.assertEqual(environment["REFRACTIR_WORKTREE_DIRTY"], "false")

  def test_unknown_dirty_state_is_preserved(self):
    environment = self.configured_environment({"REFRACTIR_WORKTREE_DIRTY": "unknown"})
    self.assertEqual(environment["REFRACTIR_WORKTREE_DIRTY"], "unknown")

  def test_missing_dirty_state_is_unknown(self):
    environment = self.configured_environment({})
    self.assertEqual(environment["REFRACTIR_WORKTREE_DIRTY"], "unknown")

  def test_explicit_flags_override_environment_defaults(self):
    with patch.dict(
      os.environ,
      {
        "REFRACTIR_COMPILER_FLAGS": "generic",
        "REFRACTIR_STUDY_COMPILER_FLAGS": "study",
      },
      clear=True,
    ):
      args = driver.parse_arguments(
        ["--output-dir", "/tmp/study", "--compiler-flags", "explicit"]
      )
    self.assertEqual(args.compiler_flags, "explicit")


class StudyReferenceBindingTests(unittest.TestCase):
  def setUp(self):
    self.input_bytes = struct.pack("<2f", 1.0, 2.0)
    self.reference = {
      "status": "accepted",
      "source_sha256": driver._sha256(_SOURCE.encode()),
      "input_sha256": driver._sha256(self.input_bytes),
      "output_bytes_hex": self.input_bytes.hex(),
    }

  def assert_refused(self, source=_SOURCE, input_bytes=None):
    with patch.object(
      study, "replay_reference", side_effect=AssertionError("replayed")
    ):
      with self.assertRaisesRegex(ValueError, "reference checkpoint"):
        driver.replay_or_reuse_reference(
          {"reference": self.reference},
          source,
          self.input_bytes if input_bytes is None else input_bytes,
        )

  def test_checkpoint_for_a_different_program_is_refused(self):
    self.assert_refused(source=_SOURCE.replace("ret %x;", "ret 0;"))

  def test_checkpoint_for_different_inputs_is_refused(self):
    self.assert_refused(input_bytes=struct.pack("<2f", 0.0, 2.0))

  def test_checkpoint_without_program_identity_is_refused(self):
    del self.reference["source_sha256"]
    self.assert_refused()

  def test_checkpoint_without_input_identity_is_refused(self):
    del self.reference["input_sha256"]
    self.assert_refused()

  def test_checkpoint_with_an_incomplete_output_is_refused(self):
    self.reference["output_bytes_hex"] = self.input_bytes[:4].hex()
    self.assert_refused()

  def test_scalar_checkpoint_reuses_its_single_complete_word(self):
    source = "fun @kernel(%x: f32) : f32 {\n^entry:\n  ret %x;\n}\n"
    data = struct.pack("<f", -0.0)
    checkpoint = {
      "status": "accepted",
      "source_sha256": driver._sha256(source.encode()),
      "input_sha256": driver._sha256(data),
      "output_bytes_hex": data.hex(),
    }
    with patch.object(
      study, "replay_reference", side_effect=AssertionError("replayed")
    ):
      reference, recovered = driver.replay_or_reuse_reference(
        {"reference": checkpoint}, source, data
      )
    self.assertEqual(reference, checkpoint)
    self.assertEqual(recovered, data)


class StudyImportRootTests(unittest.TestCase):
  def configure(self, root):
    with tempfile.TemporaryDirectory() as directory:
      executable = Path(directory) / "tool"
      executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
      executable.chmod(0o700)
      with patch.dict(os.environ, {}, clear=True):
        return driver.configure_runtime(
          root,
          symiri=executable,
          symirc=executable,
          revision="reviewed-revision",
          compiler_flags="recorded flags",
        )

  def test_source_snapshot_cannot_name_a_different_checkout(self):
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(ValueError, "import root"):
        self.configure(Path(directory))

  def test_sampler_helper_cannot_come_from_another_checkout(self):
    with patch.object(
      study, "__file__", "/other/test/lib/tensor_boundary_sensitivity.py"
    ):
      with self.assertRaisesRegex(ValueError, "import root"):
        self.configure(driver.SOURCE_ROOT)

  def assert_foreign_helper_refused(self, name):
    module = SimpleNamespace(__file__="/other/" + name.replace(".", "/") + ".py")
    with patch.dict(sys.modules, {name: module}):
      with self.assertRaisesRegex(ValueError, "import root"):
        self.configure(driver.SOURCE_ROOT)

  def test_case_loader_cannot_be_preloaded_from_another_checkout(self):
    self.assert_foreign_helper_refused("test.lib.tensor_case")

  def test_environment_probe_cannot_be_preloaded_from_another_checkout(self):
    self.assert_foreign_helper_refused("test.lib.probe_triton_env")

  def test_runner_cannot_be_preloaded_from_another_checkout(self):
    self.assert_foreign_helper_refused("test.lib.run_triton_tests")

  def test_a_symlink_to_the_actual_import_root_is_accepted(self):
    with tempfile.TemporaryDirectory() as directory:
      link = Path(directory) / "source"
      link.symlink_to(driver.SOURCE_ROOT, target_is_directory=True)
      paths = self.configure(link)
    self.assertIn("symiri", paths)


class TensorBoundarySensitivityDriverTests(unittest.TestCase):
  def test_frozen_corpus_is_reused_and_configuration_is_locked(self):
    corpus = study.freeze_corpus(
      [{"seed": 2027, "status": "candidate", "source": _SOURCE}],
      config=_CONFIG,
      base_revision="base-revision",
      target_count=1,
    )
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "corpus.json"
      study.save_corpus(path, corpus)
      reused = driver.load_or_create_corpus(
        path,
        config=_CONFIG,
        base_revision="base-revision",
        target_count=1,
        candidates_factory=lambda: self.fail("frozen candidates were regenerated"),
      )
      self.assertEqual(reused["corpus_sha256"], corpus["corpus_sha256"])
      changed = dict(_CONFIG, source_snapshot_sha256="snapshot-v2")
      with self.assertRaisesRegex(ValueError, "configuration"):
        driver.load_or_create_corpus(
          path,
          config=changed,
          base_revision="base-revision",
          target_count=1,
          candidates_factory=lambda: self.fail("mismatched corpus was regenerated"),
        )

  def test_staged_boundary_proposal_is_reused_without_resampling(self):
    input_bytes = struct.pack("<2f", -1.0, 2.0)
    staged = {
      "proposal_status": "accepted",
      "sampler_version": study.BOUNDARY_SAMPLER_VERSION,
      "input_bytes_hex": input_bytes.hex(),
      "input_buffers": [
        {
          "name": "x",
          "dtype": "f32",
          "shape": [2],
          "bytes_hex": input_bytes.hex(),
        }
      ],
    }
    slot = {
      "arm": "boundary",
      "sampler_seed": 91,
      "sampler_version": study.BOUNDARY_SAMPLER_VERSION,
    }
    with patch.object(
      study, "propose_boundary_inputs", side_effect=AssertionError("resampled")
    ):
      proposal, reused_bytes = driver.prepare_proposal(
        slot, _SOURCE, _INPUTS, "/unused/native-api", staged=staged
      )

    self.assertEqual(reused_bytes, input_bytes)
    self.assertEqual(proposal["input_bytes_hex"], input_bytes.hex())

  def test_accepted_reference_checkpoint_is_reused_without_replay(self):
    input_bytes = struct.pack("<2f", 1.0, 2.0)
    output_bytes = struct.pack("<2f", 3.0, 4.0)
    slot = {
      "reference": {
        "status": "accepted",
        "output_bytes_hex": output_bytes.hex(),
        "elapsed_s": 0.25,
        "source_sha256": driver._sha256(_SOURCE.encode()),
        "input_sha256": driver._sha256(input_bytes),
      }
    }
    with patch.object(
      study, "replay_reference", side_effect=AssertionError("replayed")
    ):
      reference, recovered = driver.replay_or_reuse_reference(
        slot, _SOURCE, input_bytes
      )

    self.assertEqual(reference["output_bytes_hex"], output_bytes.hex())
    self.assertEqual(recovered, output_bytes)

  def test_new_reference_checkpoint_records_command_and_input_identity(self):
    input_bytes = struct.pack("<2f", 1.0, 2.0)
    output_bytes = struct.pack("<2f", 3.0, 4.0)
    with patch.object(
      study,
      "replay_reference",
      return_value={
        "status": "accepted",
        "output_bytes": output_bytes,
        "output_bytes_hex": output_bytes.hex(),
      },
    ):
      reference, recovered = driver.replay_or_reuse_reference({}, _SOURCE, input_bytes)

    self.assertEqual(reference["source_sha256"], driver._sha256(_SOURCE.encode()))
    self.assertEqual(reference["input_sha256"], driver._sha256(input_bytes))
    self.assertIn("--tensor-reference-stability", reference["command"]["arguments"])
    self.assertEqual(reference["command"]["timeout_s"], 10)
    self.assertEqual(recovered, output_bytes)

  def test_boundary_proposal_records_command_and_buffer_identity(self):
    input_bytes = struct.pack("<2f", -1.0, 2.0)
    slot = {
      "arm": "boundary",
      "sampler_seed": 91,
      "sampler_version": study.BOUNDARY_SAMPLER_VERSION,
    }
    with patch.object(
      study,
      "propose_boundary_inputs",
      return_value={
        "status": "accepted",
        "sampler_version": study.BOUNDARY_SAMPLER_VERSION,
        "buffers": [
          {
            "name": "x",
            "dtype": "f32",
            "shape": [2],
            "bytes": input_bytes,
          }
        ],
      },
    ) as sample:
      proposal, combined = driver.prepare_proposal(
        slot, _SOURCE, _INPUTS, "/isolated/native-api"
      )

    sample.assert_called_once_with("/isolated/native-api", _SOURCE, 91, timeout_s=10)
    self.assertEqual(proposal["source_sha256"], driver._sha256(_SOURCE.encode()))
    self.assertEqual(proposal["input_sha256"], driver._sha256(input_bytes))
    self.assertEqual(
      proposal["sampling_command"]["argv"],
      ["/isolated/native-api", "<program.sir>", "91"],
    )
    self.assertEqual(combined, input_bytes)

  def test_compile_failure_counts_compilation_and_skips_execution(self):
    with tempfile.TemporaryDirectory() as directory:
      run = Path(directory) / "runs" / "compiled"
      run.mkdir(parents=True)
      (run / "worker-phase.json").write_text('{"phase":"compilation"}\n')
      (run / "worker-timing.json").write_text(
        json.dumps({"compilation_s": 0.5, "gpu_execution_s": 0.0})
      )
      work = driver.classify_gpu_run(
        Path(directory), {"status": "compile_failure"}, elapsed_s=1.0
      )

    self.assertEqual(work["compilation_count"], 1)
    self.assertEqual(work["execution_count"], 0)
    self.assertEqual(work["compilation_state"], "attempted")
    self.assertEqual(work["execution_state"], "skipped")

  def test_timeout_without_worker_phase_is_unknown_not_zero_work(self):
    with tempfile.TemporaryDirectory() as directory:
      work = driver.classify_gpu_run(
        Path(directory), {"status": "runtime_timeout"}, elapsed_s=3.0
      )

    self.assertIsNone(work["compilation_count"])
    self.assertIsNone(work["execution_count"])
    self.assertEqual(work["compilation_state"], "unknown")
    self.assertEqual(work["execution_state"], "unknown")

  def test_terminal_gpu_report_is_recovered_without_a_second_compile(self):
    with tempfile.TemporaryDirectory() as directory:
      case_dir = Path(directory)
      (case_dir / "result-compiled.json").write_text(
        json.dumps({"status": "pass", "campaign_valid": True}), encoding="utf-8"
      )
      runner = patch.object(
        driver, "run_compiled_case", side_effect=AssertionError("recompiled")
      )
      with runner:
        outcome = driver.execute_gpu_case(case_dir, device_id=4)

    self.assertEqual(outcome["report"]["status"], "pass")
    self.assertTrue(outcome["recovered"])

  def test_interrupted_compiled_run_is_not_retried(self):
    with tempfile.TemporaryDirectory() as directory:
      case_dir = Path(directory)
      run_dir = case_dir / "runs" / "compiled"
      run_dir.mkdir(parents=True)
      (run_dir / "worker-phase.json").write_text('{"phase":"compilation"}\n')
      with patch.object(
        driver, "run_compiled_case", side_effect=AssertionError("retried")
      ):
        outcome = driver.execute_gpu_case(case_dir, device_id=4)

    self.assertEqual(outcome["report"]["status"], "interrupted")
    self.assertTrue(outcome["interrupted"])

  def test_checkpointed_gpu_submission_without_report_is_not_restarted(self):
    with tempfile.TemporaryDirectory() as directory:
      case_dir = Path(directory)
      with patch.object(
        driver, "run_compiled_case", side_effect=AssertionError("restarted")
      ):
        outcome = driver.recover_submitted_gpu_case(case_dir)

    self.assertEqual(outcome["report"]["status"], "interrupted")
    self.assertTrue(outcome["recovered"])
    work = driver.classify_gpu_run(case_dir, outcome["report"], elapsed_s=0.0)
    self.assertIsNone(work["compilation_count"])
    self.assertIsNone(work["execution_count"])

  def test_source_snapshot_refuses_changed_files_on_resume(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / "source"
      root.mkdir()
      source = root / "kernel.cpp"
      source.write_text("first revision\n", encoding="utf-8")
      snapshot_path = Path(directory) / "source-snapshot.json"
      first = driver.load_or_create_source_snapshot(snapshot_path, root)
      self.assertEqual(
        driver.load_or_create_source_snapshot(snapshot_path, root), first
      )

      source.write_text("changed revision\n", encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "source snapshot"):
        driver.load_or_create_source_snapshot(snapshot_path, root)

  def test_frozen_control_catalogue_is_reused_and_locked(self):
    corpus = study.freeze_corpus(
      [{"seed": 2027, "status": "candidate", "source": _SOURCE}],
      config=_CONFIG,
      base_revision="base-revision",
      target_count=1,
    )
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "control-catalogue.json"
      catalogue = driver.load_or_create_control_catalogue(path, corpus)
      with patch.object(
        study, "build_control_catalogue", side_effect=AssertionError("rebuilt")
      ):
        self.assertEqual(
          driver.load_or_create_control_catalogue(path, corpus), catalogue
        )

      corrupted = json.loads(path.read_text(encoding="utf-8"))
      corrupted["kernels"].clear()
      path.write_text(json.dumps(corrupted), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "control catalogue"):
        driver.load_or_create_control_catalogue(path, corpus)

  def test_template_validation_rejects_source_drift(self):
    manifest = {
      "entry": "kernel",
      "profile": "tensor-v2",
      "grid": [1, 1, 1],
      "launch": _CONFIG["launch"],
      "comparison": _CONFIG["comparison"],
      "inputs": [
        {
          "name": "x",
          "dtype": "f32",
          "shape": [2],
          "byte_length": 8,
          "element_strides": [1],
          "element_offset": 0,
          "storage_span": 2,
          "axis_order": [0],
        }
      ],
      "output": {
        "name": "return",
        "dtype": "f32",
        "shape": [2],
        "byte_length": 8,
        "element_strides": [1],
        "element_offset": 0,
        "storage_span": 2,
        "axis_order": [0],
      },
    }
    case = SimpleNamespace(program=(_SOURCE + " ").encode(), manifest=manifest)
    with patch("test.lib.tensor_case.load_case", return_value=case):
      with self.assertRaisesRegex(ValueError, "source differs"):
        driver.validate_case_template(
          "/unused/case",
          source=_SOURCE,
          symiri="/unused/symiri",
          config=_CONFIG,
        )

  def test_template_launch_override_updates_probe_and_is_idempotent(self):
    with tempfile.TemporaryDirectory() as directory:
      case_dir = Path(directory)
      manifest = {
        "profile": "tensor-v2",
        "launch": {
          "num_warps": 8,
          "num_stages": 1,
          "enable_fp_fusion": False,
        },
        "execution": {
          "warp_probe": {"status": "not_probed", "num_warps": 8},
          "output_canary_by_dtype": {"f32": "0000807f"},
        },
        "source": {"sha256": "source-digest"},
      }
      manifest_path = case_dir / "case.json"
      manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

      normalized = driver.normalize_case_template_launch(case_dir, _CONFIG["launch"])
      first_bytes = manifest_path.read_bytes()
      repeated = driver.normalize_case_template_launch(case_dir, _CONFIG["launch"])
      repeated_bytes = manifest_path.read_bytes()

    self.assertEqual(normalized["launch"], _CONFIG["launch"])
    self.assertEqual(normalized["execution"]["warp_probe"]["num_warps"], 4)
    self.assertEqual(
      normalized["execution"]["output_canary_by_dtype"], {"f32": "0000807f"}
    )
    self.assertEqual(normalized["source"], {"sha256": "source-digest"})
    self.assertEqual(repeated, normalized)
    self.assertEqual(repeated_bytes, first_bytes)

  def test_ensure_case_template_normalizes_launch_before_validation(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      template_root = root / "templates"
      case_dir = template_root / "kernel-test" / "generation" / "case_000000"
      kernel = {"kernel_id": "kernel-test", "candidate_seed": 2027, "source": _SOURCE}

      def generate(_command, **_kwargs):
        case_dir.mkdir(parents=True)
        (case_dir / "case.json").write_text(
          json.dumps(
            {
              "profile": "tensor-v2",
              "launch": {
                "num_warps": 8,
                "num_stages": 1,
                "enable_fp_fusion": False,
              },
              "execution": {"warp_probe": {"status": "not_probed", "num_warps": 8}},
            }
          ),
          encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

      def validate(path, **_kwargs):
        manifest = json.loads((Path(path) / "case.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["launch"], _CONFIG["launch"])
        self.assertEqual(manifest["execution"]["warp_probe"]["num_warps"], 4)
        return manifest

      with (
        patch(
          "test.tensor.campaigns.continuous_triton_campaign.build_generation_command",
          return_value=["rysmith"],
        ),
        patch.object(driver.subprocess, "run", side_effect=generate),
        patch.object(driver, "validate_case_template", side_effect=validate),
      ):
        status = driver.ensure_case_template(
          kernel,
          template_root=template_root,
          rysmith="rysmith",
          source_root=root,
          symiri="symiri",
          config=_CONFIG,
        )

    self.assertEqual(status["status"], "accepted")

  def test_package_lock_artifacts_share_storage_without_changing_bytes(self):
    lock_bytes = b"frozen package-lock artifact" * 16
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      first = root / "cases" / "first" / "runs" / "compiled"
      second = root / "cases" / "second" / "runs" / "compiled"
      first.mkdir(parents=True)
      second.mkdir(parents=True)
      (first / "package-lock.json").write_bytes(lock_bytes)
      (second / "package-lock.json").write_bytes(lock_bytes)

      driver.deduplicate_package_lock_artifact(first.parent.parent, root)
      driver.deduplicate_package_lock_artifact(second.parent.parent, root)

      first_lock = first / "package-lock.json"
      second_lock = second / "package-lock.json"
      canonical = root / "shared" / "package-lock.json"
      self.assertEqual(first_lock.read_bytes(), lock_bytes)
      self.assertEqual(second_lock.read_bytes(), lock_bytes)
      self.assertEqual(canonical.read_bytes(), lock_bytes)
      self.assertEqual(first_lock.stat().st_ino, canonical.stat().st_ino)
      self.assertEqual(second_lock.stat().st_ino, canonical.stat().st_ino)

      driver.deduplicate_package_lock_artifact(second.parent.parent, root)
      self.assertEqual(second_lock.stat().st_ino, canonical.stat().st_ino)

  def test_package_lock_deduplication_skips_missing_artifacts(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      case_dir = root / "cases" / "empty"
      case_dir.mkdir(parents=True)

      self.assertFalse(driver.deduplicate_package_lock_artifact(case_dir, root))
      self.assertFalse((root / "shared" / "package-lock.json").exists())

  def test_materialized_case_is_reused_only_for_identical_proposal_bytes(self):
    input_bytes = struct.pack("<2f", -1.0, 2.0)
    expected_bytes = struct.pack("<2f", 3.0, 4.0)
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      template = root / "template"
      template.mkdir()
      (template / "program.sir").write_text(_SOURCE, encoding="utf-8")
      (template / "inputs.bin").write_bytes(bytes(8))
      (template / "expected.bin").write_bytes(bytes(8))
      (template / "case.json").write_text(
        json.dumps(
          {
            "inputs": [{"name": "x", "dtype": "f32", "shape": [2], "byte_length": 8}],
            "output": {"byte_length": 8},
            "files": {},
          }
        ),
        encoding="utf-8",
      )
      case_path, reused = driver.materialize_or_reuse_case(
        template,
        root / "proposal",
        input_bytes=input_bytes,
        expected_bytes=expected_bytes,
        policy="boundary",
        sampler_version=study.BOUNDARY_SAMPLER_VERSION,
        proposal_id="proposal-1",
        corpus_sha256="corpus-v1",
        input_seed=91,
      )
      self.assertFalse(reused)
      recovered, reused = driver.materialize_or_reuse_case(
        template,
        case_path,
        input_bytes=input_bytes,
        expected_bytes=expected_bytes,
        policy="boundary",
        sampler_version=study.BOUNDARY_SAMPLER_VERSION,
        proposal_id="proposal-1",
        corpus_sha256="corpus-v1",
        input_seed=91,
      )
      self.assertEqual(recovered, case_path)
      self.assertTrue(reused)

      with self.assertRaisesRegex(ValueError, "proposal case identity"):
        driver.materialize_or_reuse_case(
          template,
          case_path,
          input_bytes=struct.pack("<2f", 1.0, 1.0),
          expected_bytes=expected_bytes,
          policy="boundary",
          sampler_version=study.BOUNDARY_SAMPLER_VERSION,
          proposal_id="proposal-1",
          corpus_sha256="corpus-v1",
          input_seed=91,
        )

  def test_gpu_jobs_use_process_workers_and_parent_collects_reports(self):
    with tempfile.TemporaryDirectory() as directory:
      jobs = []
      for index, device_id in enumerate((4, 5)):
        case_dir = Path(directory) / f"case-{index}"
        case_dir.mkdir()
        (case_dir / "result-compiled.json").write_text(
          json.dumps({"status": "pass", "campaign_valid": True}),
          encoding="utf-8",
        )
        jobs.append(
          {
            "proposal_id": f"proposal-{index}",
            "case_directory": str(case_dir),
            "device_id": device_id,
            "timeout_s": 10,
          }
        )
      outcomes = []

      driver.run_compiled_jobs(
        jobs,
        workers=2,
        environment={"campaign_available": True},
        on_result=outcomes.append,
      )

    self.assertEqual(
      {item["proposal_id"] for item in outcomes}, {"proposal-0", "proposal-1"}
    )
    self.assertTrue(all(item["recovered"] for item in outcomes))
    self.assertTrue(all(item["report"]["status"] == "pass" for item in outcomes))

  def test_candidate_start_seed_must_fit_native_uint32(self):
    with self.assertRaises(SystemExit):
      driver.parse_arguments(
        ["--output-dir", "/tmp/boundary-study", "--start-seed", "4294967296"]
      )

  def test_command_manifest_captures_each_experiment_stage(self):
    commands = driver.command_manifest(
      source_root="/isolated/source",
      generator_api="/isolated/source/build/bin/tensor_input_policy_api",
      rysmith="/isolated/source/rysmith",
      symiri="/isolated/source/symiri",
      symirc="/isolated/source/symirc",
      compiler_flags="clang++ -std=c++20 -O2",
      candidate_timeout_s=30,
      template_timeout_s=120,
      reference_timeout_s=10,
      gpu_timeout_s=90,
    )

    self.assertEqual(
      commands["candidate_generation"]["argv"],
      ["/isolated/source/build/bin/tensor_input_policy_api", "--generate", "<seed>"],
    )
    self.assertIn("--tensor-generator", commands["template_generation"]["argv"])
    self.assertIn("compositional", commands["template_generation"]["argv"])
    self.assertIn("--tensor-input-policy", commands["template_generation"]["argv"])
    self.assertIn("ordinary", commands["template_generation"]["argv"])
    self.assertEqual(commands["boundary_sampling"]["timeout_s"], 10)
    self.assertIn("--tensor-reference-stability", commands["reference_replay"]["argv"])
    self.assertEqual(commands["compiled_execution"]["timeout_s"], 90)
    self.assertEqual(
      commands["compiled_execution"]["compiler_flags"], "clang++ -std=c++20 -O2"
    )
    self.assertEqual(commands["symir_emission"]["argv"][0], "/isolated/source/symirc")
    self.assertEqual(
      commands["symir_emission"]["argv"][1:4],
      ["--target", "triton", "--entry"],
    )

  def test_control_provenance_binds_source_input_and_comparator(self):
    control_source = _SOURCE.replace("ret %x;", "%y = tensor add %x, %x;\n  ret %y;")
    input_bytes = struct.pack("<2f", -1.0, 2.0)
    provenance = driver.control_execution_provenance(
      {"control_id": "negative", "status": "available", "source": control_source},
      input_bytes=input_bytes,
      comparison=_CONFIG["comparison"],
      symiri="/isolated/source/symiri",
    )

    self.assertEqual(
      provenance["control_source_sha256"],
      driver._sha256(control_source.encode("utf-8")),
    )
    self.assertEqual(provenance["input_sha256"], driver._sha256(input_bytes))
    self.assertEqual(provenance["comparison"], _CONFIG["comparison"])
    self.assertEqual(
      provenance["checker_command"]["executable"], "/isolated/source/symiri"
    )
    self.assertEqual(
      provenance["reference_command"]["executable"], "/isolated/source/symiri"
    )

  def test_gpu_command_record_captures_worker_invocation(self):
    record = driver.gpu_command_record(
      case_directory="/study/cases/kernel-1/boundary/proposal-00",
      source_root="/isolated/source",
      symiri="/isolated/source/symiri",
      symirc="/isolated/source/symirc",
      device_id=5,
      timeout_s=90,
    )

    self.assertEqual(record["runner"], "test.lib.run_triton_tests.run_case")
    self.assertEqual(record["device_id"], 5)
    self.assertEqual(record["timeout_s"], 90)
    self.assertIn("test.lib.triton_case_worker", record["worker_argv"])
    self.assertEqual(record["symirc_argv_template"][0], "/isolated/source/symirc")
    self.assertEqual(record["environment"]["CUDA_VISIBLE_DEVICES"], "5")
    self.assertEqual(
      record["environment"]["REFRACTIR_SYMIRC"], "/isolated/source/symirc"
    )

  def test_gpu_command_marker_is_restored_from_proposal_sidecar(self):
    corpus = study.freeze_corpus(
      [{"seed": 2027, "status": "candidate", "source": _SOURCE}],
      config=_CONFIG,
      base_revision="base-revision",
      target_count=1,
    )
    ledger = study.new_study_ledger(corpus)
    slot = ledger["proposals"][0]
    slot["gpu_submitted"] = True
    slot["gpu_device_id"] = 5
    slot["gpu_case_directory"] = "cases/proposal-00"
    slot["gpu_submitted_at"] = "2026-10-10T00:00:00+00:00"

    with tempfile.TemporaryDirectory() as directory:
      ledger_path = Path(directory) / "study-ledger.json"
      study.save_ledger(ledger_path, ledger)
      slot["gpu_command"] = {"runner": "test.lib.run_triton_tests.run_case"}
      study.save_proposal_record(ledger_path, ledger, slot["proposal_id"])
      recovered = study.load_ledger(ledger_path, corpus)

    self.assertEqual(
      recovered["proposals"][0]["gpu_command"],
      {"runner": "test.lib.run_triton_tests.run_case"},
    )

  def test_summary_links_the_frozen_command_manifest(self):
    corpus = study.freeze_corpus(
      [{"seed": 2027, "status": "candidate", "source": _SOURCE}],
      config=_CONFIG,
      base_revision="base-revision",
      target_count=1,
    )
    summary = driver._report_summary(
      study.new_study_ledger(corpus),
      output_dir="/study",
      corpus=corpus,
      run_configuration={"command_manifest_sha256": "commands-digest"},
    )

    self.assertEqual(summary["command_manifest_path"], "commands.json")
    self.assertEqual(summary["command_manifest_sha256"], "commands-digest")

  def test_command_manifest_preserves_executable_paths_with_spaces(self):
    commands = driver.command_manifest(
      source_root="/isolated tree/source",
      generator_api="/isolated tree/source/api binary",
      rysmith="/isolated tree/source/rysmith",
      symiri="/isolated tree/source/symiri",
      symirc="/isolated tree/source/symirc",
      compiler_flags="clang++ -std=c++20 -O2",
      candidate_timeout_s=30,
      template_timeout_s=120,
      reference_timeout_s=10,
      gpu_timeout_s=90,
    )

    self.assertEqual(
      commands["candidate_generation"]["argv"][0],
      "/isolated tree/source/api binary",
    )
    self.assertEqual(commands["candidate_generation"]["cwd"], "/isolated tree/source")


if __name__ == "__main__":
  unittest.main()
