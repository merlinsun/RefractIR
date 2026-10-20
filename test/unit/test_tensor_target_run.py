"""Contract tests for the target-only interpreter/GPU run and its classifications."""

import hashlib
import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test.lib import tensor_target_case, tensor_target_run

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
AGREE_HEX = struct.pack("<4f", *AGREE).hex()

STUB_SYMIRI = """\
import sys
raise SystemExit(0)
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
    self.symiri = _write_script(root / "symiri", STUB_SYMIRI)
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
      "inputs": [INPUT_RECORD],
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
    self.assertEqual(self.run_target(timeout_s=0.2).status, tensor_target_run.TIMED_OUT)

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

  def test_a_bundle_with_a_reference_is_refused_before_any_run(self):
    self.write_bundle()
    (self.bundle / "expected.bin").write_bytes(b"\0" * 16)
    with self.assertRaises(tensor_target_case.TargetCaseError) as raised:
      self.run_target()
    self.assertEqual(raised.exception.status, tensor_target_case.REFERENCE_NOT_ALLOWED)
    self.assertFalse((self.bundle / "target-runs").exists())


if __name__ == "__main__":
  unittest.main()
