// A small consumer of the installed `libsymir.a`.
//
// The archive is a list of object files, so building it proves nothing about
// whether its objects resolve each other: a tensor interpretation that reaches
// `readTensorArguments`, `writeTensorResult`, or a numerical tensor helper
// fails only when something links against the archive and calls it. This
// consumer links the archive the way an out-of-tree tool would and interprets
// one scalar, one exact tensor, and one numerical tensor program, so a missing
// translation unit is a link error or a wrong result here rather than a
// surprise for the first external user.
//
// It is deliberately small and standalone: the frontend entry points, the
// interpreter, and the tensor transport are the whole surface it touches.

#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ast/tensor.hpp"
#include "analysis/tensor.hpp"
#include "frontend/pipeline.hpp"
#include "interp/interpreter.hpp"
#include "interp/tensor_io.hpp"

namespace {

  int failures = 0;

  void report(bool ok, const std::string &what, const std::string &detail) {
    if (ok) {
      std::cout << "ok: " << what << "\n";
      return;
    }
    std::cout << "FAIL: " << what << ": " << detail << "\n";
    ++failures;
  }

  refractir::Program checked(const std::string &source) {
    refractir::Program program = refractir::parseSource(source);
    refractir::DiagBag diagnostics;
    if (!refractir::checkProgram(program, diagnostics))
      throw std::runtime_error("consumer program failed frontend checking");
    return program;
  }

  std::uint32_t readI32(const std::vector<std::uint8_t> &bytes, std::size_t offset) {
    std::uint32_t word = 0;
    for (unsigned index = 0; index < 4; ++index)
      word |= static_cast<std::uint32_t>(bytes[offset + index]) << (index * 8);
    return word;
  }

  void appendI32(std::vector<std::uint8_t> &bytes, std::int32_t value) {
    const auto word = static_cast<std::uint32_t>(value);
    for (unsigned shift = 0; shift < 32; shift += 8)
      bytes.push_back(static_cast<std::uint8_t>(word >> shift));
  }

  /// Scalar interpretation: no tensor type is involved at all.
  void checkScalarInterpretation() {
    const auto program = checked(
        "fun @kernel(%a: i32, %b: i32) : i32 {\n"
        "  let mut %out: i32 = undef;\n"
        "^entry:\n"
        "  %out = %a * %b;\n"
        "  ret %out;\n"
        "}\n"
    );
    std::ostringstream sink;
    refractir::Interpreter interpreter(program, sink);
    const auto result = interpreter.run("@kernel", {}, {"7", "3"});
    const bool ok = result && result->kind == refractir::RuntimeValue::Kind::Int &&
                    result->intVal == 21;
    report(ok, "scalar interpretation", "expected 21");
  }

  /// Exact tensor interpretation through the canonical binary transport.
  void checkExactTensorInterpretation() {
    const auto program = checked(
        "fun @kernel(%x: tensor [4] i32) : i32 {\n"
        "  let mut %sum: i32 = undef;\n"
        "^entry:\n"
        "  %sum = tensor sum %x, 0, drop;\n"
        "  ret %sum;\n"
        "}\n"
    );
    std::vector<std::uint8_t> input;
    for (const std::int32_t value: {5, -1, 2, 4})
      appendI32(input, value);
    const std::string binary(input.begin(), input.end());
    std::istringstream stream(binary, std::ios::binary);
    const auto args =
        refractir::readTensorArguments(stream, {refractir::TensorTypeInfo{{4}, refractir::TensorElement::I32}});
    std::ostringstream sink;
    refractir::Interpreter interpreter(program, sink);
    const auto result = interpreter.runTyped("@kernel", {}, args);
    const bool ok = result && result->intVal == 10;
    report(ok, "exact tensor interpretation", "expected 10");
  }

  /// Numerical tensor interpretation: reaches the multiprecision helpers.
  void checkNumericalTensorInterpretation() {
    const auto program = checked(
        "fun @kernel(%x: tensor [2] f32) : tensor [2] f32 {\n"
        "  let mut %out: tensor [2] f32 = undef;\n"
        "^entry:\n"
        "  %out = tensor exp %x;\n"
        "  ret %out;\n"
        "}\n"
    );
    const float values[2] = {0.0f, 1.0f};
    std::vector<std::uint8_t> input;
    for (const float value: values) {
      std::uint32_t word = 0;
      std::memcpy(&word, &value, sizeof(word));
      for (unsigned shift = 0; shift < 32; shift += 8)
        input.push_back(static_cast<std::uint8_t>(word >> shift));
    }
    const std::string binary(input.begin(), input.end());
    std::istringstream stream(binary, std::ios::binary);
    const auto args = refractir::readTensorArguments(
        stream, {refractir::TensorTypeInfo{{2}, refractir::TensorElement::F32}}
    );
    std::ostringstream sink;
    refractir::Interpreter interpreter(program, sink);
    const auto result = interpreter.runTyped("@kernel", {}, args);
    if (!result || result->kind != refractir::RuntimeValue::Kind::Tensor) {
      report(false, "numerical tensor interpretation", "no tensor result");
      return;
    }
    std::ostringstream bytes(std::ios::binary);
    refractir::writeTensorResult(
        bytes, *result, refractir::TensorTypeInfo{{2}, refractir::TensorElement::F32}
    );
    const std::string out = bytes.str();
    std::vector<std::uint8_t> raw(out.begin(), out.end());
    const bool ok = raw.size() == 8 && readI32(raw, 0) == 0x3f800000u &&
                    readI32(raw, 4) == 0x402df854u;
    report(ok, "numerical tensor interpretation", "expected exp(0)=1 and exp(1)=e");
  }

} // namespace

int main() {
  try {
    checkScalarInterpretation();
    checkExactTensorInterpretation();
    checkNumericalTensorInterpretation();
  } catch (const std::exception &error) {
    std::cout << "FAIL: consumer threw: " << error.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cout << failures << " library consumer check(s) failed\n";
    return 1;
  }
  std::cout << "library consumer ok\n";
  return 0;
}
