"""Compile the mutually-recursive name visitors with Clang."""

import os
import shlex
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class UnusedNameClangPortabilityTests(unittest.TestCase):
  def test_unused_name_analysis_compiles_with_clang(self):
    configured = os.environ.get("REFRACTIR_TEST_CLANG")
    compiler = (
      shlex.split(configured) if configured else [shutil.which("clang++") or ""]
    )
    if not compiler[0]:
      self.skipTest("clang++ is not installed")

    result = subprocess.run(
      [
        *compiler,
        "-std=c++20",
        "-I",
        str(ROOT / "include"),
        "-Wall",
        "-Wextra",
        "-DREFRACTIR_TENSOR_REF",
        "-fsyntax-only",
        str(ROOT / "src/analysis/unused_name.cpp"),
      ],
      capture_output=True,
      text=True,
      check=False,
      timeout=60,
    )
    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
  unittest.main()
