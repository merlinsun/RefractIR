#pragma once

#include <cstdint>
#include <stdexcept>
#include "ast/ast.hpp"
#include "frontend/diagnostics.hpp"

namespace refractir {

  enum class TensorConsumer { Reference, ExactSolver, Triton, C, Wasm, Python };

  struct TensorProfile {
    std::uint32_t maxRank = 3;
    std::uint32_t maxElements = 256;
  };

  /// Product of dimensions; an empty shape is the scalar boundary shape 1.
  std::uint32_t tensorElementCount(const Shape &shape);

  /// Normalize an axis in [-rank, rank) to [0, rank).
  std::uint32_t normalizeTensorAxis(std::int64_t axis, std::uint32_t rank);

  /// Compute the right-aligned NumPy-style result shape or throw on mismatch.
  Shape tensorBroadcastShape(const Shape &lhs, const Shape &rhs);

  /// Compute a reduction result shape; rank-one drop yields the scalar shape {}.
  Shape tensorReduceShape(const Shape &input, std::int64_t axis, bool keep);

  /// Check profile and backend capability constraints after ordinary checking.
  bool checkTensorCapabilities(const Program &prog, TensorConsumer consumer, DiagBag &diags);

} // namespace refractir
