"""Frontend and interpreter checks for rendered state snapshots."""

import os
import struct
import subprocess
import unittest
from pathlib import Path

from test.unit.tensor_support import SYMIRI, check_source, run_typed

REGION_API = Path(
  os.environ.get("REFRACTIR_TENSOR_REGION_API", "build/bin/tensor_region_api")
).resolve()


def rendered_swap_source():
  result = subprocess.run(
    [str(REGION_API), "render-swap-source"],
    capture_output=True,
    text=True,
    check=False,
    timeout=10,
  )
  if result.returncode != 0:
    raise AssertionError(result.stdout + result.stderr)
  return result.stdout


@unittest.skipUnless(
  REGION_API.is_file() and SYMIRI.is_file(), "native tensor tools unavailable"
)
class TensorRegionIntegrationTests(unittest.TestCase):
  def test_rendered_swap_is_valid_sir(self):
    source = rendered_swap_source()
    result = check_source(source, tensor_profile="v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_interpreter_observes_the_simultaneous_swap(self):
    source = rendered_swap_source()
    result, output = run_typed(source, struct.pack("<2i", 1, 2), tensor_profile="v2")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(output, struct.pack("<2i", 3, 4))


if __name__ == "__main__":
  unittest.main()
