"""Host contracts for ordinary binary32 GPU edge controls."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
CONTROLS_PATH = ROOT / "test/tensor/fpsan/run_ordinary_fp_controls.py"


def _load_controls():
  specification = importlib.util.spec_from_file_location(
    "test_tensor_ordinary_fp_controls_target", CONTROLS_PATH
  )
  if specification is None or specification.loader is None:
    raise RuntimeError("cannot load ordinary binary32 controls")
  module = importlib.util.module_from_spec(specification)
  torch = ModuleType("torch")
  triton = ModuleType("triton")
  triton.__path__ = []
  triton.jit = lambda function: function
  language = ModuleType("triton.language")
  language.__path__ = []
  extra = ModuleType("triton.language.extra")
  extra.__path__ = []
  cuda = ModuleType("triton.language.extra.cuda")
  cuda.__path__ = []
  libdevice = ModuleType("triton.language.extra.cuda.libdevice")
  with mock.patch.dict(
    sys.modules,
    {
      "torch": torch,
      "triton": triton,
      "triton.language": language,
      "triton.language.extra": extra,
      "triton.language.extra.cuda": cuda,
      "triton.language.extra.cuda.libdevice": libdevice,
    },
  ):
    specification.loader.exec_module(module)
  return module


class OrdinaryFpControlTests(unittest.TestCase):
  def test_subtraction_reference_requires_equal_nonempty_vectors(self):
    controls = _load_controls()

    with self.assertRaisesRegex(ValueError, "same nonzero length"):
      controls._binary32_subtraction_reference((0x3F800000,), ())
    with self.assertRaisesRegex(ValueError, "same nonzero length"):
      controls._binary32_subtraction_reference((0x3F800000,), (0x3F800000, 0))

  def test_subtraction_reference_rounds_even_half_ulp_tie_down(self):
    controls = _load_controls()

    result = controls._binary32_subtraction_reference((0x3F800000,), (0x33000000,))

    self.assertEqual(result, [0x3F800000])

  def test_subtraction_reference_rounds_odd_half_ulp_tie_to_even_neighbor(self):
    controls = _load_controls()

    result = controls._binary32_subtraction_reference((0x3F800001,), (0xB3800000,))

    self.assertEqual(result, [0x3F800002])

  def test_subtraction_reference_preserves_signed_zero_and_subnormals(self):
    controls = _load_controls()

    result = controls._binary32_subtraction_reference(
      (0x80000000, 0x00000001, 0x00000001, 0x00000000),
      (0x00000000, 0x00000000, 0x00000001, 0x80000000),
    )

    self.assertEqual(result, [0x80000000, 0x00000001, 0x00000000, 0x00000000])

  def test_ptx_ftz_detection_includes_subtraction(self):
    controls = _load_controls()

    self.assertTrue(controls._ptx_has_ftz_arithmetic("sub.ftz.f32 %f1, %f2, %f3;"))
    self.assertTrue(controls._ptx_has_ftz_arithmetic("sub.rn.ftz.f32 %f1, %f2, %f3;"))
    self.assertFalse(controls._ptx_has_ftz_arithmetic("sub.rn.f32 %f1, %f2, %f3;"))

  def test_subtraction_reference_refuses_finite_overflow(self):
    controls = _load_controls()

    with self.assertRaisesRegex(OverflowError, "binary32 overflow"):
      controls._binary32_subtraction_reference((0x7F7FFFFF,), (0xFF7FFFFF,))


if __name__ == "__main__":
  unittest.main()
