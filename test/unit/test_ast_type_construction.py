"""The AST Type wrapper must accept each variant through C++20 construction."""

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TYPE_VARIANTS = (
  ("integer", "IntType{}", 0),
  ("float", "FloatType{}", 1),
  ("struct", 'StructType{GlobalId{"@S", {}}, {}}', 2),
  ("array", "ArrayType{4, TypePtr{}, {}}", 3),
  ("pointer", "PtrType{TypePtr{}, {}}", 4),
  ("vector", "VecType{4, TypePtr{}, {}}", 5),
  ("tensor", "TensorType{Shape{2, 3}, TypePtr{}, {}}", 6),
)


def _compiler_command():
  configured = os.environ.get("REFRACTIR_TEST_CXX") or os.environ.get("CXX")
  if configured:
    return shlex.split(configured)
  return [shutil.which("clang++") or shutil.which("g++") or "c++"]


class AstTypeConstructionTests(unittest.TestCase):
  def assert_variant_compiles(self, name, initializer, index, with_span=True):
    if initializer is None:
      arguments = ""
    elif with_span:
      arguments = f"{initializer}, SourceSpan{{}}"
    else:
      arguments = initializer
    source = (
      '#include "ast/ast.hpp"\n'
      "#include <memory>\n"
      "using namespace refractir;\n"
      "int main() {\n"
      f"  auto value = std::make_shared<Type>({arguments});\n"
      f"  return value->v.index() == {index} ? 0 : 1;\n"
      "}\n"
    )
    with tempfile.TemporaryDirectory() as temporary:
      directory = Path(temporary)
      source_path = directory / f"{name}.cpp"
      binary_path = directory / name
      source_path.write_text(source, encoding="utf-8")
      result = subprocess.run(
        [
          *_compiler_command(),
          "-std=c++20",
          "-I",
          str(ROOT / "include"),
          str(source_path),
          "-o",
          str(binary_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
      )
      self.assertEqual(result.returncode, 0, result.stderr)
      executed = subprocess.run(
        [str(binary_path)], capture_output=True, text=True, check=False, timeout=10
      )
      self.assertEqual(executed.returncode, 0, executed.stderr)

  def test_integer_variant_construction_compiles(self):
    self.assert_variant_compiles(*TYPE_VARIANTS[0])

  def test_default_type_construction_compiles(self):
    self.assert_variant_compiles("default", None, 0)

  def test_float_variant_construction_compiles(self):
    self.assert_variant_compiles(*TYPE_VARIANTS[1])

  def test_struct_variant_construction_compiles(self):
    self.assert_variant_compiles(*TYPE_VARIANTS[2])

  def test_array_variant_construction_compiles(self):
    self.assert_variant_compiles(*TYPE_VARIANTS[3])

  def test_pointer_variant_construction_compiles(self):
    self.assert_variant_compiles(*TYPE_VARIANTS[4])

  def test_vector_variant_construction_compiles(self):
    self.assert_variant_compiles(*TYPE_VARIANTS[5])

  def test_tensor_variant_construction_compiles(self):
    self.assert_variant_compiles(*TYPE_VARIANTS[6])

  def test_integer_variant_without_wrapper_span_compiles(self):
    self.assert_variant_compiles(*TYPE_VARIANTS[0], with_span=False)

  def test_float_variant_without_wrapper_span_compiles(self):
    self.assert_variant_compiles(*TYPE_VARIANTS[1], with_span=False)

  def test_struct_variant_without_wrapper_span_compiles(self):
    self.assert_variant_compiles(*TYPE_VARIANTS[2], with_span=False)

  def test_array_variant_without_wrapper_span_compiles(self):
    self.assert_variant_compiles(*TYPE_VARIANTS[3], with_span=False)

  def test_pointer_variant_without_wrapper_span_compiles(self):
    self.assert_variant_compiles(*TYPE_VARIANTS[4], with_span=False)

  def test_vector_variant_without_wrapper_span_compiles(self):
    self.assert_variant_compiles(*TYPE_VARIANTS[5], with_span=False)

  def test_tensor_variant_without_wrapper_span_compiles(self):
    self.assert_variant_compiles(*TYPE_VARIANTS[6], with_span=False)


if __name__ == "__main__":
  unittest.main()
