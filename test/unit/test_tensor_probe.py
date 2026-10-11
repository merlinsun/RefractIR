"""Behavioral tests for the environment and real-solver probes."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_ENV_PROBE_AVAILABLE = importlib.util.find_spec("test.lib.probe_triton_env") is not None
_SOLVER_PROBE_AVAILABLE = (
  importlib.util.find_spec("test.lib.probe_cvc5_tensor") is not None
)

if _ENV_PROBE_AVAILABLE:
  from test.lib.probe_triton_env import (
    _triton_source_revision,
    collect_environment,
    qualified_compiler_identity,
    require_qualified_triton,
  )
else:
  collect_environment = None
  _triton_source_revision = None
  qualified_compiler_identity = None
  require_qualified_triton = None

if _SOLVER_PROBE_AVAILABLE:
  from test.lib.probe_cvc5_tensor import run_probe
else:
  run_probe = None


class QualifiedCompilerIdentityTests(unittest.TestCase):
  """Gluon runners accept one pinned compiler revision, not a version string."""

  def setUp(self):
    self.assertTrue(_ENV_PROBE_AVAILABLE, "tensor probe APIs are not implemented")

  def test_accepts_the_qualified_revision(self):
    identity = qualified_compiler_identity()
    require_qualified_triton(
      version=identity["version"], source_revision=identity["source_revision"]
    )

  def test_rejects_a_declared_revision_without_the_qualified_source(self):
    identity = qualified_compiler_identity()

    with self.assertRaisesRegex(RuntimeError, "source revision"):
      require_qualified_triton(version=identity["version"], source_revision="0" * 40)

  def test_rejects_an_unqualified_version(self):
    identity = qualified_compiler_identity()
    with self.assertRaisesRegex(RuntimeError, "3.8.0"):
      require_qualified_triton(
        version="3.8.0", source_revision=identity["source_revision"]
      )

  def test_rejects_an_unavailable_source_revision(self):
    identity = qualified_compiler_identity()

    with self.assertRaisesRegex(RuntimeError, "source revision"):
      require_qualified_triton(
        version=identity["version"], source_revision="unavailable"
      )


class TensorProbeTests(unittest.TestCase):
  def setUp(self):
    self.assertTrue(
      _ENV_PROBE_AVAILABLE and _SOLVER_PROBE_AVAILABLE,
      "tensor probe APIs are not implemented",
    )

  @staticmethod
  def environment_snapshot(gpus):
    return {
      "gpus": gpus,
      "cuda_version": "12.8",
      "python_version": "3.10.12",
      "torch_version": "2.10.0+cu128",
      "triton_version": "3.8.0",
      "triton_source_revision": "e2e356723ce35f46b5144373b145c55c4d712237",
      "packages": [
        {"name": "torch", "version": "2.10.0+cu128", "sha256": "a" * 64},
        {"name": "triton", "version": "3.8.0", "sha256": "b" * 64},
      ],
    }

  @staticmethod
  def fake_solver(directory, stdout, delay_s=0):
    path = Path(directory) / "fake-cvc5"
    script = f"""#!{sys.executable}
import sys
import time
if "--version" in sys.argv:
    print("This is cvc5 version 1.3.4")
else:
    time.sleep({delay_s!r})
    sys.stdout.write({stdout!r})
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path

  @staticmethod
  def query_file(directory):
    path = Path(directory) / "probe.smt2"
    path.write_text(
      "(set-logic QF_NRAT)\n(declare-const x Real)\n(check-sat)\n(get-value (x))\n",
      encoding="utf-8",
    )
    return path

  @staticmethod
  def local_git_repo(directory):
    root = Path(directory)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
      ["git", "-C", str(root), "config", "user.email", "test@example.invalid"],
      check=True,
    )
    subprocess.run(
      ["git", "-C", str(root), "config", "user.name", "Tensor Probe Test"], check=True
    )
    (root / "source.txt").write_text("pinned source\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "source.txt"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "source"], check=True)
    return root, subprocess.check_output(
      ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()

  def source_revision_from_file_url(self, url):
    class Distribution:
      @staticmethod
      def read_text(name):
        self.assertEqual(name, "direct_url.json")
        return json.dumps({"url": url, "dir_info": {"editable": True}})

    with (
      mock.patch(
        "test.lib.probe_triton_env.importlib.metadata.distribution",
        return_value=Distribution(),
      ),
      mock.patch.dict(
        sys.modules,
        {"triton": SimpleNamespace(__file__="/tmp/no-triton-git/__init__.py")},
      ),
    ):
      return _triton_source_revision()

  def test_local_file_install_reports_source_commit(self):
    with tempfile.TemporaryDirectory() as directory:
      root, expected = self.local_git_repo(Path(directory) / "triton-src")
      self.assertEqual(self.source_revision_from_file_url(root.as_uri()), expected)

  def test_local_file_install_resolves_checkout_parent(self):
    with tempfile.TemporaryDirectory() as directory:
      root, expected = self.local_git_repo(Path(directory) / "triton-src")
      nested = root / "python" / "triton"
      nested.mkdir(parents=True)
      self.assertEqual(self.source_revision_from_file_url(nested.as_uri()), expected)

  def test_local_file_install_decodes_escaped_repository_path(self):
    with tempfile.TemporaryDirectory() as directory:
      root, expected = self.local_git_repo(Path(directory) / "triton source")
      self.assertEqual(self.source_revision_from_file_url(root.as_uri()), expected)

  def test_local_file_install_resolves_git_worktree(self):
    with tempfile.TemporaryDirectory() as directory:
      root, expected = self.local_git_repo(Path(directory) / "triton-src")
      worktree = Path(directory) / "triton-worktree"
      subprocess.run(
        [
          "git",
          "-C",
          str(root),
          "worktree",
          "add",
          "-q",
          "--detach",
          str(worktree),
          "HEAD",
        ],
        check=True,
      )
      self.assertTrue((worktree / ".git").is_file())
      self.assertEqual(self.source_revision_from_file_url(worktree.as_uri()), expected)

  def test_local_file_install_reports_detached_commit(self):
    with tempfile.TemporaryDirectory() as directory:
      root, expected = self.local_git_repo(Path(directory) / "triton-src")
      subprocess.run(
        ["git", "-C", str(root), "checkout", "-q", "--detach", expected], check=True
      )
      self.assertEqual(self.source_revision_from_file_url(root.as_uri()), expected)

  def test_missing_gpu_is_unavailable(self):
    result = collect_environment(probe=lambda: self.environment_snapshot(gpus=[]))

    self.assertEqual(result["status"], "environment_unavailable")
    self.assertFalse(result["campaign_available"])

  def test_environment_records_identity(self):
    snapshot = self.environment_snapshot(
      gpus=[
        {
          "name": "Tesla V100-SXM2-32GB",
          "compute_capability": "7.0",
          "driver_version": "535.274.02",
        }
      ]
    )
    result = collect_environment(probe=lambda: snapshot)

    self.assertEqual(result["status"], "environment_available")
    self.assertEqual(result["gpus"][0]["compute_capability"], "7.0")
    self.assertEqual(result["driver_version"], "535.274.02")
    self.assertEqual(
      result["triton_source_revision"], snapshot["triton_source_revision"]
    )
    self.assertRegex(result["package_lock_sha256"], r"^[0-9a-f]{64}$")

  def test_unknown_is_not_sat(self):
    with tempfile.TemporaryDirectory() as directory:
      solver = self.fake_solver(directory, "unknown\n((x 0))\n")
      query = self.query_file(directory)
      result = run_probe(
        str(solver),
        query,
        solver_ms=100,
        deadline_s=2,
        finite_menu=(-2, -1, 0, 1, 2, 3),
      )

    self.assertEqual(result["status"], "synthesis_unknown")
    self.assertFalse(result["accepted_witness"])
    self.assertEqual(result["materialization_status"], "not_attempted")

  def test_external_deadline(self):
    with tempfile.TemporaryDirectory() as directory:
      solver = self.fake_solver(directory, "sat\n", delay_s=2)
      query = self.query_file(directory)
      result = run_probe(
        str(solver),
        query,
        solver_ms=1,
        deadline_s=0.1,
        finite_menu=(-2, -1, 0, 1, 2, 3),
      )

    self.assertEqual(result["status"], "synthesis_timeout")
    self.assertTrue(result["external_timeout"])
    self.assertLess(result["elapsed_ns"], 1_000_000_000)

  def test_nonliteral_model_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      solver = self.fake_solver(directory, "sat\n((x (exp 0)))\n")
      query = self.query_file(directory)
      result = run_probe(
        str(solver),
        query,
        solver_ms=100,
        deadline_s=2,
        finite_menu=(-2, -1, 0, 1, 2, 3),
      )

    self.assertEqual(result["status"], "materialization_rejected")
    self.assertEqual(result["materialization_status"], "rejected")
    self.assertFalse(result["accepted_witness"])
    self.assertIn("x", result["materialization_error"])

  def test_absent_cvc5_does_not_disable_cpu_tests(self):
    with tempfile.TemporaryDirectory() as directory:
      query = self.query_file(directory)
      result = run_probe(
        str(Path(directory) / "missing-cvc5"),
        query,
        finite_menu=(-2, -1, 0, 1, 2, 3),
      )

    self.assertEqual(result["status"], "environment_unavailable")
    self.assertFalse(result["accepted_witness"])


if __name__ == "__main__":
  unittest.main()
