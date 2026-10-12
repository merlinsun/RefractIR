#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "analysis/tensor.hpp"
#include "ast/tensor.hpp"
#include "reify/tensor_region_family.hpp"

namespace refractir::reify {

/// One required value at one logical output coordinate. The coordinate is
/// row-major and fully padded to the output rank, so a record is unambiguous
/// without consulting the shape it was written against.
struct TensorOutputElementConstraint {
  std::vector<std::uint32_t> index;
  std::int32_t value = 0;
};

/// The bounded loop-region choices one case was generated from: the two-state
/// shape, the counted trip count, the connection family, and the arithmetic
/// its roots use. It is written to the manifest whenever a caller named a
/// region option, so a reader sees the values that were used instead of
/// having to infer them from the seed.
struct TensorRegionConfiguration {
  Shape shape;
  std::uint32_t iterations = 0;
  TensorRegionFamily family = TensorRegionFamily::Fold;
  TensorRegionArithmeticMode arithmetic = TensorRegionArithmeticMode::Additive;
};

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
  // The loop-region request this case was generated from. Unset for a case
  // whose caller named no region option, which keeps its manifest unchanged.
  std::optional<TensorRegionConfiguration> region;
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
  // The structured form of the same request. A scalar target states one number
  // for the whole output; an element target states one per coordinate and says
  // whether it covers every coordinate. Serialized beside the legacy scalar
  // fields so a reader can check the two representations against each other.
  std::string propertyKind = "none";
  std::vector<TensorOutputElementConstraint> propertyElements;
  bool propertyComplete = false;
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
