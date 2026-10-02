#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ast/tensor.hpp"

namespace refractir::reify {

  /// A typed tensor input to a reusable straight-line CFG region.
  struct TensorRegionInput {
    std::string name;
    TensorTypeInfo type;
  };

  /// A small typed DAG whose several result roots are emitted as one region.
  /// Dead nodes outside the union of result dependencies are omitted.
  class TensorRegion {
  public:
    explicit TensorRegion(std::vector<TensorRegionInput> inputs);

    void
    add(std::string name, TensorOp op, std::vector<std::string> operands, std::int64_t axis = 0,
        bool keepDims = false, Shape attribute = {});
    void setResults(std::vector<std::string> results);

    [[nodiscard]] const TensorTypeInfo &typeOf(const std::string &name) const;
    [[nodiscard]] std::vector<std::string> renderDeclarations() const;
    [[nodiscard]] std::vector<std::string> renderInstructions() const;
    [[nodiscard]] std::vector<std::string>
    renderBody(const std::vector<std::pair<std::string, std::string>> &updates) const;
    [[nodiscard]] std::vector<std::string>
    renderStateUpdates(const std::vector<std::pair<std::string, std::string>> &updates) const;

  private:
    struct Value {
      std::string name;
      TensorTypeInfo type;
      std::int32_t producer = -1;
    };

    struct Node {
      TensorOp op;
      std::vector<std::int32_t> operands;
      std::int64_t axis;
      bool keepDims;
      Shape attribute;
      std::int32_t result;
    };

    [[nodiscard]] std::int32_t requireValue(const std::string &name) const;
    [[nodiscard]] std::string renderNode(const Node &node) const;
    [[nodiscard]] std::vector<char> liveNodes() const;

    std::vector<Value> values_;
    std::vector<Node> nodes_;
    std::vector<std::int32_t> results_;
  };

} // namespace refractir::reify
