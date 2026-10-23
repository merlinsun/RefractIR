"""Contract tests for the target-only interpreter/GPU run and its classifications."""

import hashlib
import json
import os
import stat
import struct
from dataclasses import replace
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test.lib import (
  tensor_emission,
  tensor_target_case,
  tensor_target_reference,
  tensor_target_run,
)

SOURCE = """\
fun @kernel(%x: tensor [2, 2] f32, %y: tensor [2, 2] f32) : tensor [2, 2] f32 {
  let mut %i: i32 = undef;
  let mut %acc_x: tensor [2, 2] f32 = undef;
  let mut %acc_y: tensor [2, 2] f32 = undef;
  let mut %out: tensor [2, 2] f32 = undef;
^entry:
  %acc_x = %x;
  %acc_y = %y;
  %i = 0;
  br ^header;
^header:
  br %i < 2, ^body, ^exit;
^body:
  %acc_x = tensor add %acc_x, %acc_y;
  %i = %i + 1;
  br ^header;
^exit:
  %out = tensor add %acc_x, %acc_y;
  ret %out;
}
"""

LAYOUT = {
  "shape": [2, 2],
  "element_strides": [2, 1],
  "element_offset": 0,
  "storage_span": 4,
  "axis_order": [1, 0],
}
INPUT_DATA = (1.0, 2.0, 3.0, 4.0)
INPUT_BYTES = struct.pack("<4f", *INPUT_DATA)
# Each parameter is one dense [2, 2] f32 tensor.
INPUT_RECORD = {"name": "%x", "dtype": "f32", **LAYOUT, "data": INPUT_BYTES}
SECOND_INPUT_RECORD = {"name": "%y", "dtype": "f32", **LAYOUT, "data": INPUT_BYTES}
OUTPUT_RECORD = {"name": "return", "dtype": "f32", **LAYOUT}
LAUNCH = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}
EXECUTION = {
  "guard_envelope": {"prefix": 1, "suffix": 1},
  "gap_sentinel_by_dtype": {"f32": "3f000001", "i32": "55555555"},
  "output_canary_by_dtype": {"f32": "3f000002", "i32": "aaaaaaaa"},
  "padding_by_dtype": {"f32": "3f000003", "i32": "55555556"},
  "warp_probe": {"status": "not_probed", "num_warps": 4},
}
AGREE = (1.0, 6.0, 3.0, 12.0)
TRITON_IDENTITY = {
  "version": "3.9.0+git32f06df4",
  "revision": "32f06df4bf1c30885ff61ee3488836ca81ed65f0",
  "torch": "2.5.1+cu121",
  "device": "Tesla V100-SXM2-32GB",
  "cache_mode": "default",
}
NATIVE_KERNEL = """\
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, y_ptr, out_ptr):
    pass
"""
CANARY = bytes.fromhex(EXECUTION["output_canary_by_dtype"]["f32"])
PARTIAL_HEX = (CANARY + struct.pack("<3f", 2.0, 3.0, 4.0)).hex()
AGREE_HEX = struct.pack("<4f", *AGREE).hex()

STUB_SYMBOLS = """\
import json
import sys

print(
    json.dumps(
        {
            "entry": "kernel",
            "inputs": [
                {"name": "x", "dtype": "f32", "shape": [2, 2], "byte_length": 16},
                {"name": "y", "dtype": "f32", "shape": [2, 2], "byte_length": 16},
            ],
            "output": {"name": "return", "dtype": "f32", "shape": [2, 2], "byte_length": 16},
        }
    )
)
"""

STUB_SYMIRC = """\
import sys
path = sys.argv[sys.argv.index("-o") + 1]
with open(path, "w", encoding="utf-8") as output:
    output.write("kernel = object()\\n")
"""

STUB_WORKER = """\
import argparse
import json
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--mode", required=True)
parser.add_argument("--artifacts-dir", required=True)
arguments = parser.parse_args()
config = json.loads(open(arguments.config, encoding="utf-8").read())[arguments.mode]
if config.get("phase"):
    with open(os.path.join(arguments.artifacts_dir, "worker-phase.json"), "w", encoding="utf-8") as output:
        json.dump({"phase": config["phase"]}, output)
if config.get("sleep"):
    time.sleep(config["sleep"])
if config.get("actual"):
    with open(os.path.join(arguments.artifacts_dir, "actual.bin"), "wb") as output:
        output.write(bytes.fromhex(config["actual"]))
if config.get("stderr"):
    sys.stderr.write(config["stderr"])
raise SystemExit(config.get("exit", 0))
"""


def _write_script(path, text):
  path.write_text(f"#!{sys.executable}\n" + text, encoding="utf-8")
  path.chmod(path.stat().st_mode | stat.S_IXUSR)
  return path


class TargetRunTests(unittest.TestCase):
  """A target-only run states what each path did and never implies a reference."""

  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    root = Path(self.directory.name)
    self.symiri = _write_script(root / "symiri", STUB_SYMBOLS)
    self.symirc = _write_script(root / "symirc", STUB_SYMIRC)
    self.worker = _write_script(root / "worker.py", STUB_WORKER)
    self.config = root / "worker.json"
    self.bundle = root / "bundle"
    self.configure({"interpret": {}, "compiled": {}})

  def configure(self, config):
    self.config.write_text(json.dumps(config), encoding="utf-8")

  def write_bundle(self, **fields):
    arguments = {
      "source": SOURCE,
      "inputs": [INPUT_RECORD, SECOND_INPUT_RECORD],
      "output": OUTPUT_RECORD,
      "launch": LAUNCH,
      "execution": EXECUTION,
    }
    arguments.update(fields)
    return tensor_target_case.write(self.bundle, **arguments)

  def worker_command(self, case, mode, artifact_dir):
    return [
      sys.executable,
      str(self.worker),
      "--config",
      str(self.config),
      "--mode",
      mode,
      "--artifacts-dir",
      str(artifact_dir),
    ]

  def run_target(self, *, gpu=True, timeout_s=5, **kwargs):
    return tensor_target_run.run(
      self.bundle,
      symirc=self.symirc,
      symiri=self.symiri,
      worker_command=self.worker_command,
      gpu_probe=lambda: gpu,
      timeout_s=timeout_s,
      **kwargs,
    )

  def test_a_reference_observation_is_recorded_as_diagnostic(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": AGREE_HEX},
        "compiled": {"phase": "runtime", "actual": AGREE_HEX},
      }
    )
    reference = tensor_target_reference.ReferenceObservation(
      tensor_target_reference.COMPLETED, output=bytes.fromhex(AGREE_HEX)
    )
    run = self.run_target(reference=reference)
    record = run.as_record()
    self.assertEqual(record["reference"]["kind"], "diagnostic")
    self.assertEqual(record["reference"]["path"], "refractir_interpreter")
    self.assertEqual(record["reference"]["status"], "completed")
    self.assertEqual(record["diagnosis"]["diagnosis"], "agreement")
    self.assertEqual(record["diagnosis"]["p0_availability"], "completed")
    bundle = json.loads((self.bundle / "case.json").read_text(encoding="utf-8"))
    self.assertEqual(bundle["reference"], {"kind": "none"})

  def test_both_modes_consume_one_emission_and_agree(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": AGREE_HEX},
        "compiled": {"phase": "runtime", "actual": AGREE_HEX},
      }
    )
    run = self.run_target()
    self.assertEqual(run.status, tensor_target_run.TARGET_AGREEMENT)
    self.assertFalse(run.candidate)
    record = run.as_record()
    self.assertEqual(record["reference"], {"kind": "none"})
    self.assertEqual(record["agreement"]["kind"], "identical-bytes")
    self.assertEqual(record["emission"]["kind"], "symirc")
    self.assertIsNotNone(record["emission"]["emitter"]["binary_sha256"])
    self.assertIsNotNone(record["emission"]["layout_manifest_sha256"])
    for mode in ("interpret", "compiled"):
      staged = (self.bundle / "target-runs" / mode / "kernel.py").read_bytes()
      self.assertEqual(
        record["emission"]["kernel_sha256"],
        hashlib.sha256(staged).hexdigest(),
        f"{mode} must consume the recorded kernel",
      )
      self.assertEqual(
        record["interpreter" if mode == "interpret" else "compiled"]["actual_sha256"],
        hashlib.sha256(bytes.fromhex(AGREE_HEX)).hexdigest(),
      )

  def run_native(self, **overrides):
    arguments = {
      "kernel": NATIVE_KERNEL,
      "launch": LAUNCH,
      "triton": TRITON_IDENTITY,
      "symiri": self.symiri,
      "worker_command": self.worker_command,
      "gpu_probe": lambda: True,
      "timeout_s": 5,
    }
    arguments.update(overrides)
    return tensor_target_run.run_native(self.bundle, **arguments)

  def test_a_frozen_native_kernel_runs_both_modes_without_symirc(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": AGREE_HEX},
        "compiled": {"phase": "runtime", "actual": AGREE_HEX},
      }
    )
    run = self.run_native()
    self.assertEqual(run.status, tensor_target_run.TARGET_AGREEMENT)
    self.assertEqual(run.interpreter.status, tensor_target_run.COMPLETED)
    self.assertEqual(run.compiled.status, tensor_target_run.COMPLETED)
    staged = self.bundle / "native-runs" / "interpret" / "kernel.py"
    self.assertEqual(staged.read_text(encoding="utf-8"), NATIVE_KERNEL)
    record = run.as_record()
    emitter = record["emission"]["emitter"]
    self.assertEqual(record["emission"]["path"], "native")
    self.assertEqual(emitter["kind"], "native")
    self.assertEqual(emitter["sir_role"], "signature-only")
    self.assertNotIn("executable", emitter)
    self.assertNotIn("binary_sha256", emitter)
    self.assertEqual(emitter["triton"], TRITON_IDENTITY)

  def test_native_both_modes_use_the_same_kernel_digest(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": AGREE_HEX},
        "compiled": {"phase": "runtime", "actual": AGREE_HEX},
      }
    )
    run = self.run_native()
    record = run.as_record()
    for mode in ("interpret", "compiled"):
      staged = self.bundle / "native-runs" / mode / "kernel.py"
      self.assertEqual(
        record["emission"]["kernel_sha256"],
        hashlib.sha256(staged.read_bytes()).hexdigest(),
      )

  def test_native_run_reports_agreement_or_disagreement(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": AGREE_HEX},
        "compiled": {"phase": "runtime", "actual": AGREE_HEX},
      }
    )
    self.assertEqual(self.run_native().status, tensor_target_run.TARGET_AGREEMENT)
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": AGREE_HEX},
        "compiled": {"phase": "runtime", "actual": PARTIAL_HEX},
      }
    )
    self.assertEqual(self.run_native().status, tensor_target_run.TARGET_DISAGREEMENT)

  def test_native_missing_kernel_refused(self):
    self.write_bundle()
    with self.assertRaises(tensor_emission.EmissionRefused):
      self.run_native(kernel="")

  def test_native_record_states_the_native_path(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": AGREE_HEX},
        "compiled": {"phase": "runtime", "actual": AGREE_HEX},
      }
    )
    record = self.run_native().as_record()
    self.assertEqual(record["emission"]["path"], "native")
    self.assertIn("native_digest", record["emission"])
    self.assertEqual(record["emission"]["emitter"]["triton"], TRITON_IDENTITY)

  def test_a_tampered_native_record_is_refused(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": AGREE_HEX},
        "compiled": {"phase": "runtime", "actual": AGREE_HEX},
      }
    )
    record = self.run_native().emission
    tampered = replace(
      record,
      kernel=tensor_emission.EmittedKernel(source="def kernel():\n  pass\n"),
    )
    with self.assertRaisesRegex(tensor_emission.EmissionRefused, "tampered"):
      self.run_native(record=tampered)

  def test_p0_ub_marks_concrete_execution_invalid_in_the_saved_record(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": AGREE_HEX},
        "compiled": {"phase": "runtime", "actual": AGREE_HEX},
      }
    )
    reference = tensor_target_reference.ReferenceObservation(tensor_target_reference.UB)
    run = self.run_target(reference=reference)
    self.assertEqual(run.status, tensor_target_run.TARGET_AGREEMENT)
    self.assertEqual(
      run.as_record()["legality"]["concrete_execution"]["status"], "invalid_execution"
    )
    self.assertTrue(run.candidate)
    saved = json.loads(
      (self.bundle / "target-runs" / "target-run.json").read_text(encoding="utf-8")
    )
    self.assertEqual(
      saved["legality"]["concrete_execution"]["status"], "invalid_execution"
    )
    self.assertEqual(saved["diagnosis"]["diagnosis"], tensor_target_run.INVALID_EXECUTION)
    self.assertTrue(saved["candidate"])

  def test_an_emission_suspect_diagnosis_is_a_candidate(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": AGREE_HEX},
        "compiled": {"phase": "runtime", "actual": AGREE_HEX},
      }
    )
    reference = tensor_target_reference.ReferenceObservation(
      tensor_target_reference.COMPLETED, output=bytes.fromhex(PARTIAL_HEX)
    )
    run = self.run_target(reference=reference)
    self.assertEqual(run.status, tensor_target_run.TARGET_AGREEMENT)
    self.assertEqual(run.as_record()["diagnosis"]["diagnosis"], "emission suspect")
    self.assertTrue(run.candidate)
    self.assertTrue(run.as_record()["candidate"])

  def test_an_unavailable_gpu_is_not_a_numerical_candidate(self):
    self.write_bundle()
    self.configure({"interpret": {"phase": "runtime", "actual": AGREE_HEX}})
    reference = tensor_target_reference.ReferenceObservation(
      tensor_target_reference.COMPLETED, output=bytes.fromhex(AGREE_HEX)
    )
    run = self.run_target(gpu=False, reference=reference)
    self.assertEqual(run.status, tensor_target_run.ENVIRONMENT_UNAVAILABLE)
    record = run.as_record()
    self.assertEqual(record["diagnosis"]["diagnosis"], tensor_target_run.TARGET_UNAVAILABLE)
    self.assertFalse(record["candidate"])
    saved = json.loads(
      (self.bundle / "target-runs" / "target-run.json").read_text(encoding="utf-8")
    )
    self.assertEqual(saved["diagnosis"]["diagnosis"], tensor_target_run.TARGET_UNAVAILABLE)
    self.assertFalse(saved["candidate"])

  def test_an_unsupported_interpreter_keeps_a_p0_target_difference(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {
          "phase": "runtime",
          "stderr": "InterpreterError: NotImplementedError('inline_asm not supported in interpreter mode')",
          "exit": 1,
        },
        "compiled": {"phase": "runtime", "actual": PARTIAL_HEX},
      }
    )
    reference = tensor_target_reference.ReferenceObservation(
      tensor_target_reference.COMPLETED, output=bytes.fromhex(AGREE_HEX)
    )
    run = self.run_target(reference=reference)
    self.assertEqual(run.status, tensor_target_run.COMPARISON_UNAVAILABLE)
    record = run.as_record()
    self.assertEqual(
      record["diagnosis"]["diagnosis"], tensor_target_run.UNRESOLVED_NUMERICAL
    )
    self.assertEqual(record["diagnosis"]["observations"], {"R-G": "different"})
    self.assertTrue(record["candidate"])
    saved = json.loads(
      (self.bundle / "target-runs" / "target-run.json").read_text(encoding="utf-8")
    )
    self.assertEqual(saved["diagnosis"]["observations"], {"R-G": "different"})
    self.assertTrue(saved["candidate"])

  def test_an_unavailable_gpu_keeps_a_p0_interpreter_difference(self):
    self.write_bundle()
    self.configure({"interpret": {"phase": "runtime", "actual": PARTIAL_HEX}})
    reference = tensor_target_reference.ReferenceObservation(
      tensor_target_reference.COMPLETED, output=bytes.fromhex(AGREE_HEX)
    )
    run = self.run_target(gpu=False, reference=reference)
    self.assertEqual(run.status, tensor_target_run.ENVIRONMENT_UNAVAILABLE)
    record = run.as_record()
    self.assertEqual(
      record["diagnosis"]["diagnosis"], tensor_target_run.UNRESOLVED_NUMERICAL
    )
    self.assertEqual(record["diagnosis"]["observations"], {"R-T": "different"})
    self.assertTrue(record["candidate"])

  def test_an_unavailable_gpu_with_an_interpreter_failure_is_still_a_candidate(self):
    self.write_bundle()
    self.configure(
      {"interpret": {"phase": "runtime", "stderr": "RuntimeError: boom", "exit": 1}}
    )
    reference = tensor_target_reference.ReferenceObservation(
      tensor_target_reference.COMPLETED, output=bytes.fromhex(AGREE_HEX)
    )
    run = self.run_target(gpu=False, reference=reference)
    # The unavailable GPU owns the run's status, but the interpreter's actual
    # failure is still the diagnosis and still enters triage.
    self.assertEqual(run.status, tensor_target_run.ENVIRONMENT_UNAVAILABLE)
    self.assertEqual(
      run.as_record()["diagnosis"]["diagnosis"], tensor_target_run.INTERPRETER_CANDIDATE
    )
    self.assertTrue(run.candidate)

  def test_an_unsupported_interpreter_is_not_a_candidate(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {
          "phase": "runtime",
          "stderr": "InterpreterError: NotImplementedError('inline_asm not supported in interpreter mode')",
          "exit": 1,
        },
        "compiled": {"phase": "runtime", "actual": AGREE_HEX},
      }
    )
    reference = tensor_target_reference.ReferenceObservation(
      tensor_target_reference.COMPLETED, output=bytes.fromhex(AGREE_HEX)
    )
    run = self.run_target(reference=reference)
    self.assertEqual(run.status, tensor_target_run.COMPARISON_UNAVAILABLE)
    self.assertEqual(
      run.as_record()["diagnosis"]["diagnosis"], tensor_target_run.TARGET_UNAVAILABLE
    )
    self.assertFalse(run.candidate)

  def test_a_reference_unavailable_run_is_not_a_candidate(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": AGREE_HEX},
        "compiled": {"phase": "runtime", "actual": AGREE_HEX},
      }
    )
    reference = tensor_target_reference.ReferenceObservation(
      tensor_target_reference.UNAVAILABLE
    )
    run = self.run_target(reference=reference)
    self.assertEqual(
      run.as_record()["diagnosis"]["diagnosis"], tensor_target_run.REFERENCE_UNAVAILABLE
    )
    self.assertFalse(run.candidate)

  def test_a_compile_failure_is_classified_separately(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": AGREE_HEX},
        "compiled": {"phase": "compilation", "stderr": "ptxas: boom", "exit": 1},
      }
    )
    run = self.run_target()
    self.assertEqual(run.status, tensor_target_run.COMPILE_FAILURE)
    self.assertEqual(run.compiled.status, tensor_target_run.COMPILE_FAILURE)
    self.assertTrue(run.candidate)

  def test_a_runtime_failure_and_a_timeout_are_classified_separately(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": AGREE_HEX},
        "compiled": {"phase": "runtime", "stderr": "RuntimeError: boom", "exit": 1},
      }
    )
    self.assertEqual(self.run_target().status, tensor_target_run.RUNTIME_FAILURE)
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": AGREE_HEX},
        "compiled": {"phase": "runtime", "sleep": 2, "actual": AGREE_HEX},
      }
    )
    timed_out = self.run_target(timeout_s=0.2)
    self.assertEqual(timed_out.status, tensor_target_run.TIMED_OUT)
    self.assertTrue(
      timed_out.candidate,
      "a timeout is an incomplete observation that needs diagnosis",
    )
    self.assertTrue(timed_out.as_record()["candidate"])

  def test_a_storage_violation_is_classified_before_a_runtime_failure(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": AGREE_HEX},
        "compiled": {
          "phase": "runtime",
          "stderr": "RuntimeError: kernel wrote outside its logical output mapping: [3]",
          "exit": 1,
        },
      }
    )
    run = self.run_target()
    self.assertEqual(run.status, tensor_target_run.STORAGE_VIOLATION)
    self.assertTrue(run.candidate)

  def test_interpreter_unsupported_with_a_completed_gpu_is_not_agreement(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {"phase": "runtime", "stderr": "RuntimeError: unsupported op", "exit": 1},
        "compiled": {"phase": "runtime", "actual": AGREE_HEX},
      }
    )
    run = self.run_target()
    self.assertEqual(run.status, tensor_target_run.COMPARISON_UNAVAILABLE)
    self.assertEqual(run.interpreter.status, tensor_target_run.INTERPRETER_UNSUPPORTED)
    self.assertFalse(run.candidate)
    self.assertEqual(run.as_record()["agreement"]["kind"], "unavailable")


  def test_an_interpreter_that_reports_not_supported_is_not_runtime_failure(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {
          "phase": "runtime",
          "stderr": "InterpreterError: NotImplementedError('inline_asm not supported in interpreter mode')",
          "exit": 1,
        },
        "compiled": {"phase": "runtime", "actual": AGREE_HEX},
      }
    )
    run = self.run_target()
    self.assertEqual(run.interpreter.status, tensor_target_run.INTERPRETER_UNSUPPORTED)
    self.assertEqual(run.status, tensor_target_run.COMPARISON_UNAVAILABLE)
    self.assertFalse(run.candidate)

  def test_a_target_disagreement_is_preserved_as_a_candidate(self):
    self.write_bundle()
    different = bytes.fromhex(AGREE_HEX)
    changed = different[:12] + struct.pack("<f", 99.0)
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": AGREE_HEX},
        "compiled": {"phase": "runtime", "actual": changed.hex()},
      }
    )
    run = self.run_target()
    self.assertEqual(run.status, tensor_target_run.TARGET_DISAGREEMENT)
    self.assertTrue(run.candidate)
    self.assertEqual(len(run.mismatches), 1)
    index, first, second = run.mismatches[0]
    self.assertEqual(index, 3)
    self.assertEqual(first, "0x41400000")
    self.assertNotEqual(first, second)
    record = json.loads(
      (self.bundle / "target-runs" / "target-run.json").read_text(encoding="utf-8")
    )
    self.assertEqual(record["status"], tensor_target_run.TARGET_DISAGREEMENT)
    self.assertEqual(record["agreement"]["mismatches"][0]["index"], 3)
    self.assertTrue(record["candidate"])

  def test_a_stated_cache_directory_reaches_the_compiled_worker(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": AGREE_HEX},
        "compiled": {"phase": "runtime", "actual": AGREE_HEX},
      }
    )
    self.worker.write_text(
      f"#!{sys.executable}\n"
      + STUB_WORKER.replace(
        'raise SystemExit(config.get("exit", 0))',
        'json.dump({"cache": os.environ.get("TRITON_CACHE_DIR"), "interpret": os.environ.get("TRITON_INTERPRET")}, open(arguments.config + "." + arguments.mode, "w"))\n'
        'raise SystemExit(config.get("exit", 0))',
      ),
      encoding="utf-8",
    )
    cache = Path(self.directory.name) / "shared-cache"
    run = self.run_target(cache_dir=cache)
    self.assertEqual(run.status, tensor_target_run.TARGET_AGREEMENT)
    compiled = json.loads(Path(f"{self.config}.compiled").read_text(encoding="utf-8"))
    interpreted = json.loads(Path(f"{self.config}.interpret").read_text(encoding="utf-8"))
    self.assertTrue(compiled["cache"].startswith(str(cache.resolve())))
    self.assertIsNone(interpreted["cache"])
    self.assertEqual(interpreted["interpret"], "1")

  def test_an_unavailable_device_is_not_a_pass(self):
    self.write_bundle()
    self.configure({"interpret": {"phase": "runtime", "actual": AGREE_HEX}, "compiled": {}})
    run = self.run_target(gpu=False)
    self.assertEqual(run.status, tensor_target_run.ENVIRONMENT_UNAVAILABLE)
    self.assertEqual(run.compiled.status, tensor_target_run.ENVIRONMENT_UNAVAILABLE)
    self.assertFalse(run.candidate)


  def test_a_short_output_is_a_runtime_failure_not_an_agreement(self):
    self.write_bundle()
    short = struct.pack("<f", 1.0).hex()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": short},
        "compiled": {"phase": "runtime", "actual": short},
      }
    )
    run = self.run_target()
    self.assertEqual(run.status, tensor_target_run.RUNTIME_FAILURE)
    self.assertTrue(run.candidate)
    self.assertEqual(run.interpreter.status, tensor_target_run.RUNTIME_FAILURE)
    self.assertEqual(run.interpreter.reason, "output is 4 bytes, expected 16")
    record = run.as_record()
    self.assertEqual(record["compiled"]["reason"], "output is 4 bytes, expected 16")
    self.assertNotEqual(record["agreement"]["kind"], "identical-bytes")

  def test_two_different_short_outputs_are_not_reported_as_agreement(self):
    self.write_bundle()
    first = struct.pack("<f", 1.0).hex()
    second = struct.pack("<f", 2.0).hex()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": first},
        "compiled": {"phase": "runtime", "actual": second},
      }
    )
    run = self.run_target()
    self.assertEqual(run.status, tensor_target_run.RUNTIME_FAILURE)
    self.assertNotEqual(run.as_record()["agreement"]["kind"], "identical-bytes")

  def test_every_byte_of_a_word_is_compared(self):
    first = b"\x01\x02\x03\x04\x05\x06"
    second = b"\x01\x02\x03\x04\x05\x07"
    differences = tensor_target_run._differences(first, second)
    self.assertEqual(len(differences), 1)
    self.assertEqual(differences[0][0], 1)
    self.assertEqual(differences[0][1], "0x00000605")
    self.assertEqual(differences[0][2], "0x00000705")



  def test_a_fully_written_output_reports_no_unwritten_slots(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": AGREE_HEX},
        "compiled": {"phase": "runtime", "actual": AGREE_HEX},
      }
    )
    run = self.run_target()
    self.assertEqual(run.interpreter.unwritten, [])
    self.assertEqual(run.compiled.unwritten, [])
    self.assertEqual(run.as_record()["interpreter"]["unwritten_logical_slots"], [])

  def test_a_partial_write_reports_the_unwritten_slots(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": PARTIAL_HEX},
        "compiled": {"phase": "runtime", "actual": PARTIAL_HEX},
      }
    )
    run = self.run_target()
    self.assertEqual(run.status, tensor_target_run.TARGET_AGREEMENT)
    self.assertEqual(run.interpreter.unwritten, [0])
    self.assertEqual(run.as_record()["compiled"]["unwritten_logical_slots"], [0])

  def test_unwritten_slots_do_not_change_the_status(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": PARTIAL_HEX},
        "compiled": {"phase": "runtime", "actual": PARTIAL_HEX},
      }
    )
    state = self.run_target()
    self.assertEqual(state.status, tensor_target_run.TARGET_AGREEMENT)
    self.assertFalse(state.candidate)

  def test_unwritten_slots_are_recorded_on_disagreement(self):
    self.write_bundle()
    first = CANARY + struct.pack("<3f", 2.0, 3.0, 4.0)
    second = CANARY + struct.pack("<3f", 2.0, 3.0, 5.0)
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": first.hex()},
        "compiled": {"phase": "runtime", "actual": second.hex()},
      }
    )
    run = self.run_target()
    self.assertEqual(run.status, tensor_target_run.TARGET_DISAGREEMENT)
    record = run.as_record()
    self.assertEqual(record["interpreter"]["unwritten_logical_slots"], [0])
    self.assertEqual(record["compiled"]["unwritten_logical_slots"], [0])

  def test_unwritten_slots_are_absent_without_a_profile(self):
    record = {"profile": "tensor-v1", "execution": {"output_canary_by_dtype": {"f32": "3f000002"}}}
    self.assertIsNone(tensor_target_run._unwritten_logical_slots(bytes(4), record))
    self.assertIsNone(tensor_target_run._unwritten_logical_slots(bytes(4), {"execution": {}}))

  def test_a_canary_equal_result_is_reported_not_failed(self):
    self.write_bundle()
    canary_hex = CANARY.hex()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": canary_hex * 4},
        "compiled": {"phase": "runtime", "actual": canary_hex * 4},
      }
    )
    run = self.run_target()
    self.assertEqual(run.status, tensor_target_run.TARGET_AGREEMENT)
    self.assertEqual(run.interpreter.unwritten, [0, 1, 2, 3])
    self.assertFalse(run.candidate)

  def test_the_record_separates_frontend_input_and_concrete_legality(self):
    self.write_bundle()
    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": AGREE_HEX},
        "compiled": {"phase": "runtime", "actual": AGREE_HEX},
      }
    )
    agreement = self.run_target().as_record()["legality"]
    self.assertEqual(agreement["frontend"]["status"], "accepted")
    self.assertEqual(agreement["frontend"]["signature"]["entry"], "kernel")
    self.assertEqual(agreement["input_and_layout"]["status"], "valid")
    self.assertEqual(agreement["concrete_execution"]["status"], "established")

    self.configure(
      {
        "interpret": {"phase": "runtime", "actual": AGREE_HEX},
        "compiled": {
          "phase": "runtime",
          "stderr": "RuntimeError: kernel wrote outside its logical output mapping: [3]",
          "exit": 1,
        },
      }
    )
    violated = self.run_target().as_record()["legality"]
    self.assertEqual(violated["frontend"]["status"], "accepted")
    self.assertEqual(violated["concrete_execution"]["status"], "violated")

    self.configure(
      {
        "interpret": {"phase": "runtime", "stderr": "RuntimeError: unsupported op", "exit": 1},
        "compiled": {"phase": "runtime", "actual": AGREE_HEX},
      }
    )
    unresolved = self.run_target().as_record()["legality"]
    self.assertEqual(unresolved["frontend"]["status"], "accepted")
    self.assertEqual(unresolved["concrete_execution"]["status"], "unresolved")

  def test_a_bundle_with_a_reference_is_refused_before_any_run(self):
    self.write_bundle()
    (self.bundle / "expected.bin").write_bytes(b"\0" * 16)
    with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
      self.run_target()
    self.assertEqual(raised.exception.status, tensor_target_case.REFERENCE_NOT_ALLOWED)
    self.assertFalse((self.bundle / "target-runs").exists())


class OptionalReferenceDiagnosisTests(unittest.TestCase):
  """A diagnostic P0 must not be mistaken for an accepted reference."""

  def observation(self, status, output=None):
    return tensor_target_reference.ReferenceObservation(status, output=output)

  def mode(self, status, actual=None):
    return tensor_target_run.ModeRun(
      mode="compiled",
      status=status,
      exit_code=0,
      phase="runtime",
      actual=actual,
      stderr="",
      log="",
    )

  def diagnose(self, reference, interpreter, compiled):
    return tensor_target_run.diagnose_with_optional_reference(
      reference=reference, interpreter=interpreter, compiled=compiled
    )

  def test_p0_ub_is_invalid_execution_and_stays_in_the_denominator(self):
    result = self.diagnose(
      self.observation(tensor_target_reference.UB),
      self.mode(tensor_target_run.COMPLETED, bytes.fromhex(AGREE_HEX)),
      self.mode(tensor_target_run.COMPLETED, bytes.fromhex(AGREE_HEX)),
    )
    self.assertEqual(result["diagnosis"], tensor_target_run.INVALID_EXECUTION)
    self.assertEqual(result["p0_availability"], "ub")

  def test_p0_unavailable_with_only_an_interpreter_failure_is_interpreter_candidate(self):
    result = self.diagnose(
      self.observation(tensor_target_reference.UNAVAILABLE),
      self.mode(tensor_target_run.RUNTIME_FAILURE),
      self.mode(tensor_target_run.COMPLETED, bytes.fromhex(AGREE_HEX)),
    )
    self.assertEqual(result["diagnosis"], tensor_target_run.INTERPRETER_CANDIDATE)
    self.assertEqual([candidate["path"] for candidate in result["candidates"]], ["P1"])

  def test_p0_unavailable_with_only_a_gpu_compile_failure_stays_compile_failure(self):
    result = self.diagnose(
      self.observation(tensor_target_reference.UNAVAILABLE),
      self.mode(tensor_target_run.COMPLETED, bytes.fromhex(AGREE_HEX)),
      self.mode(tensor_target_run.COMPILE_FAILURE),
    )
    self.assertEqual(result["diagnosis"], tensor_target_run.COMPILED_PATH_CANDIDATE)
    self.assertEqual(result["candidates"][0]["kind"], "compile_failure")
    self.assertEqual(result["candidates"][0]["path"], "P2")

  def test_p0_unavailable_with_both_paths_failing_is_both_paths_failed(self):
    result = self.diagnose(
      self.observation(tensor_target_reference.UNAVAILABLE),
      self.mode(tensor_target_run.RUNTIME_FAILURE),
      self.mode(tensor_target_run.COMPILE_FAILURE),
    )
    self.assertEqual(result["diagnosis"], tensor_target_run.BOTH_PATHS_FAILED)
    self.assertEqual(
      {candidate["path"] for candidate in result["candidates"]}, {"P1", "P2"}
    )

  def test_p0_completed_with_only_an_interpreter_failure_is_interpreter_candidate(self):
    result = self.diagnose(
      self.observation(tensor_target_reference.COMPLETED, bytes.fromhex(AGREE_HEX)),
      self.mode(tensor_target_run.RUNTIME_FAILURE),
      self.mode(tensor_target_run.COMPLETED, bytes.fromhex(AGREE_HEX)),
    )
    self.assertEqual(result["diagnosis"], tensor_target_run.INTERPRETER_CANDIDATE)

  def test_p0_unavailable_with_t_and_g_disagreeing_is_reference_unavailable(self):
    result = self.diagnose(
      self.observation(tensor_target_reference.UNAVAILABLE),
      self.mode(tensor_target_run.COMPLETED, bytes.fromhex(AGREE_HEX)),
      self.mode(tensor_target_run.COMPLETED, bytes.fromhex(PARTIAL_HEX)),
    )
    self.assertEqual(result["diagnosis"], tensor_target_run.REFERENCE_UNAVAILABLE)
    self.assertEqual(result["p0_availability"], "unavailable")

  def test_targets_agreeing_on_a_wrong_value_are_emission_suspect(self):
    reference = bytes.fromhex(AGREE_HEX)
    wrong = bytes.fromhex(PARTIAL_HEX)
    result = self.diagnose(
      self.observation(tensor_target_reference.COMPLETED, reference),
      self.mode(tensor_target_run.COMPLETED, wrong),
      self.mode(tensor_target_run.COMPLETED, wrong),
    )
    self.assertEqual(result["diagnosis"], "emission suspect")

  def test_candidate_diagnosis_still_records_p0_availability(self):
    result = self.diagnose(
      self.observation(tensor_target_reference.UNAVAILABLE),
      self.mode(tensor_target_run.COMPLETED, bytes.fromhex(AGREE_HEX)),
      self.mode(tensor_target_run.COMPILE_FAILURE),
    )
    self.assertEqual(result["diagnosis"], tensor_target_run.COMPILED_PATH_CANDIDATE)
    self.assertEqual(result["p0_availability"], "unavailable")

  def test_a_bundle_with_expected_bin_is_still_refused(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      tensor_target_case.write(
        root / "case",
        source=SOURCE,
        inputs=[INPUT_RECORD, SECOND_INPUT_RECORD],
        output=OUTPUT_RECORD,
        launch=LAUNCH,
        execution=EXECUTION,
      )
      (root / "case" / "expected.bin").write_bytes(struct.pack("<4f", *AGREE))
      with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
        tensor_target_case.load(root / "case", symiri=root / "symiri")
      self.assertEqual(raised.exception.status, "reference_not_allowed")


if __name__ == "__main__":
  unittest.main()
