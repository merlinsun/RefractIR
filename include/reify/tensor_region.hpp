#pragma once

#include <cstdint>
#include <map>
#include <optional>
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
    /// A literal operand: a rank-0 value of the region's element type that an
    /// operation may name like any other value. It renders as its literal text
    /// where it is used and is never declared.
    void addLiteral(std::string name, TensorElement element, double value);
    void setResults(std::vector<std::string> results);

    [[nodiscard]] const TensorTypeInfo &typeOf(const std::string &name) const;
    [[nodiscard]] std::vector<std::string> renderDeclarations() const;
    [[nodiscard]] std::vector<std::string> renderDeclarations(
        const std::vector<std::pair<std::string, std::string>> &updates,
        const std::vector<std::string> &reservedNames = {}
    ) const;
    [[nodiscard]] std::vector<std::string> renderInstructions() const;
    /// An upper bound on the magnitude of every live value, given one for each
    /// input. Throws std::invalid_argument when an input has no bound.
    [[nodiscard]] std::map<std::string, double>
    magnitudeBounds(const std::map<std::string, double> &inputs) const;
    /// The live values whose rounding depends on evaluation order: the
    /// results of `sum`, whose lanes a backend may add in any order.
    [[nodiscard]] std::vector<std::string> orderSensitiveValues() const;
    /// Whether no live node mixes the lanes along `axis` or moves the axis:
    /// no reduction or scan along it, no dropped axis before it, no transpose
    /// that moves it, no dot, no expansion at or before it, and no broadcast
    /// that changes rank.
    [[nodiscard]] bool keepsAxisParallel(std::uint32_t axis) const;
    [[nodiscard]] std::vector<std::string> renderBody(
        const std::vector<std::pair<std::string, std::string>> &updates,
        const std::vector<std::string> &reservedNames = {}
    ) const;
    [[nodiscard]] std::vector<std::string> renderStateUpdates(
        const std::vector<std::pair<std::string, std::string>> &updates,
        const std::vector<std::string> &reservedNames = {}
    ) const;

  private:
    struct Value {
      std::string name;
      TensorTypeInfo type;
      std::int32_t producer = -1;
      std::optional<double> literal;
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
    [[nodiscard]] std::string operandText(std::int32_t value) const;
    [[nodiscard]] std::vector<char> liveNodes() const;
    using IndexedUpdate = std::pair<std::int32_t, std::int32_t>;
    using Snapshot = std::pair<std::int32_t, std::string>;
    [[nodiscard]] std::vector<IndexedUpdate>
    validateStateUpdates(const std::vector<std::pair<std::string, std::string>> &updates) const;
    [[nodiscard]] std::vector<Snapshot> stateSnapshots(
        const std::vector<IndexedUpdate> &updates, const std::vector<std::string> &reservedNames
    ) const;

    std::vector<Value> values_;
    std::vector<Node> nodes_;
    std::vector<std::int32_t> results_;
  };

} // namespace refractir::reify
