#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "ast/tensor.hpp"

namespace refractir::reify {

  struct TensorCaseBuffer {
    std::string name;
    TensorElement element = TensorElement::I32;
    Shape shape;
    std::vector<std::uint8_t> bytes;
  };

  struct TensorCase {
    std::string source;
    std::string target = "sir";
    std::uint32_t seed = 0;
    std::string synthesis = "construction";
    std::string propertyId;
    std::string propertySource;
    std::string propertyThresholdF32Bits;
    std::string evidenceKind;
    std::string evidenceStatus;
    std::vector<std::string> operators;
    std::vector<std::pair<Shape, std::int32_t>> shapeAxes;
    std::vector<TensorCaseBuffer> inputs;
    Shape outputShape;
    TensorElement outputElement = TensorElement::I32;
    std::vector<std::uint8_t> expected;
    bool referenceStable = true;
    bool numericComparison = false;
  };

  /// Write a complete tensor-v1 case directory. The directory is installed only
  /// after its SIR, binary data, optional Triton kernel, and manifest are ready.
  void writeTensorCase(const std::filesystem::path &directory, const TensorCase &testCase);

} // namespace refractir::reify
