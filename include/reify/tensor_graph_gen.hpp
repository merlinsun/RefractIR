#pragma once

#include <cstdint>

#include "ast/ast.hpp"
#include "reify/tensor_gen.hpp"

namespace refractir::reify {

  /// Bounds honoured by compositional tensor graph construction. `maxOperators`
  /// counts emitted tensor-operation nodes, not the whole-value copies that
  /// only rename an existing value.
  struct TensorGraphLimits {
    std::uint32_t maxInputs = 3;
    std::uint32_t maxRank = 2;
    std::uint32_t maxElements = 512;
    std::uint32_t maxOperators = 16;
  };

  /// Build a straight-line typed tensor dependency graph for the selected
  /// tensor profile and return it as an ordinary unchecked Program. Every
  /// parameter and computation of the result lies in the return value's
  /// backward slice. Throws std::runtime_error prefixed "graph_exhaustion:"
  /// when the limits admit no graph; the caller re-checks the emitted source.
  Program generateTensorProgram(
      const TensorGenerationOptions &options, const TensorGraphLimits &limits
  );

} // namespace refractir::reify
