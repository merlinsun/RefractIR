#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ast/ast.hpp"
#include "reify/tensor_gen.hpp"
#include "reify/tensor_region.hpp"
#include "reify/tensor_region_family.hpp"

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
  Program
  generateTensorProgram(const TensorGenerationOptions &options, const TensorGraphLimits &limits);

  /// Build a seeded multi-result DAG region from same-shaped tensor state
  /// inputs. Its bounded loop-safe subset uses add, sub, mul, keep-dimension
  /// sum and broadcast. Result names are also the graph roots, so dead branches
  /// are excluded and the dependency union of every loop-carried result is kept.
  /// The states are supplied in a fixed order: the caller initializes the first
  /// to the runtime input and the second to zero. The families' additive
  /// producer, nonnegative subtraction, and live state updates are stated over
  /// that contract; another initialization changes what the region computes.
  TensorRegion generateTensorRegion(
      const TensorGenerationOptions &options, const TensorGraphLimits &limits,
      std::vector<TensorRegionInput> inputs, std::vector<std::string> results,
      TensorRegionArithmeticMode arithmetic = TensorRegionArithmeticMode::Additive,
      TensorRegionFamily family = TensorRegionFamily::Fold
  );

} // namespace refractir::reify
