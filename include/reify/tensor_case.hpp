#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "analysis/tensor.hpp"
#include "ast/tensor.hpp"

namespace refractir::reify {

struct TensorCaseBuffer {
  std::string name;
  TensorElement element = TensorElement::I32;
  Shape shape;
  std::vector<std::uint8_t> bytes;
  Shape elementStrides;
  std::uint32_t elementOffset = 0;
  std::uint32_t storageSpan = 0;
  Shape axisOrder;

  TensorCaseBuffer(std::string bufferName, TensorElement bufferElement,
                   Shape bufferShape, std::vector<std::uint8_t> bufferBytes,
                   Shape strides = {}, std::uint32_t offset = 0,
                   std::uint32_t span = 0, Shape order = {})
      : name(std::move(bufferName)), element(bufferElement),
        shape(std::move(bufferShape)), bytes(std::move(bufferBytes)),
        elementStrides(std::move(strides)), elementOffset(offset),
        storageSpan(span), axisOrder(std::move(order)) {}
};

struct TensorCase {
  TensorProfileId profileVersion = TensorProfileId::V1;
  std::string primaryStratum;
  std::string source;
  std::string target = "sir";
  std::uint32_t seed = 0;
  std::uint32_t numWarps = 4;
  std::string synthesis = "construction";
  std::string propertyId;
  std::string propertySource;
  std::string propertyThresholdF32Bits;
  std::string evidenceKind;
  std::string evidenceStatus;
  std::string exactnessKind;
  std::uint64_t exactnessMaxAbsLeafSum = 0;
  std::string inputMode;
  std::uint32_t inputSeed = 0;
  std::string inputPattern;
  // Selection provenance, so a saved case states how its runtime inputs were
  // chosen rather than leaving that to the campaign summary: the policy that
  // proposed them, the inclusive leaf domain it respected, and the output
  // target it was asked to satisfy. Empty and unset mean "not applicable".
  std::string inputPolicy;
  std::optional<std::pair<std::int64_t, std::int64_t>> inputDomain;
  std::optional<std::int64_t> propertyTarget;
  std::vector<std::string> operators;
  std::vector<std::pair<Shape, std::int32_t>> shapeAxes;
  std::vector<std::pair<std::string, std::int32_t>> inputCounterfactualWitnesses;
  std::vector<std::int32_t> loopBounds;
  std::uint32_t totalBodyExecutions = 0;
  std::vector<Shape> fullShapes;
  std::vector<Shape> rangeExtents;
  std::vector<TensorCaseBuffer> inputs;
  Shape outputShape;
  TensorElement outputElement = TensorElement::I32;
  Shape outputElementStrides;
  std::uint32_t outputElementOffset = 0;
  std::uint32_t outputStorageSpan = 0;
  Shape outputAxisOrder;
  std::vector<std::uint8_t> expected;
  bool referenceStable = true;
  bool numericComparison = false;
};

/// Write a complete tensor-v1 case directory. The directory is installed only
/// after its SIR, binary data, optional Triton kernel, and manifest are ready.
void writeTensorCase(const std::filesystem::path &directory,
                     const TensorCase &testCase);

} // namespace refractir::reify
