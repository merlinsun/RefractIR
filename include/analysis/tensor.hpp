#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <string>
#include <vector>
#include "ast/ast.hpp"
#include "frontend/diagnostics.hpp"

namespace refractir {

  enum class TensorConsumer { Reference, ExactSolver, Triton, C, Wasm, Python };

  enum class TensorProfileId { V1, V2, V3, V4 };

  struct TensorProfile {
    TensorProfileId id = TensorProfileId::V1;
    std::uint32_t maxRank = 3;
    std::uint32_t maxElements = 256;
    std::uint32_t maxLoopBodyExecutions = 0;
  };

  struct TensorCountedLoop {
    std::size_t header = 0;
    std::size_t preheader = 0;
    std::size_t latch = 0;
    std::size_t exit = 0;
    std::string inductionVariable;
    std::int32_t initialValue = 0;
    std::int32_t bound = 0;
    std::uint32_t tripCount = 0;
    int parentIndex = -1;
    std::vector<std::size_t> blocks;
  };

  struct TensorLoopAnalysis {
    std::vector<TensorCountedLoop> loops;
    std::uint32_t totalBodyExecutions = 0;
  };

  /// Select the immutable limits belonging to a tensor profile version.
  TensorProfile tensorProfile(TensorProfileId id);
  TensorProfileId parseTensorProfileVersion(std::string_view version);

  /// Recognize the v2 canonical counted-loop subset and enforce its static
  /// execution cap. Returns an empty analysis for a loop-free function.
  std::optional<TensorLoopAnalysis> analyzeTensorLoops(
      const FunDecl &function, const TensorProfile &profile, DiagBag &diags
  );

  /// Product of dimensions; an empty shape is the scalar boundary shape 1.
  std::uint32_t tensorElementCount(const Shape &shape);

  /// Padded Triton tile extents for a logical tensor shape under a profile.
  /// For v1/v2 this also validates the existing power-of-two shape contract.
  Shape tensorExecutionTileShape(const Shape &shape, const TensorProfile &profile);

  /// Normalize an axis in [-rank, rank) to [0, rank).
  std::uint32_t normalizeTensorAxis(std::int64_t axis, std::uint32_t rank);

  /// Compute the right-aligned NumPy-style result shape or throw on mismatch.
  Shape tensorBroadcastShape(const Shape &lhs, const Shape &rhs);

  /// Compute a reduction result shape; rank-one drop yields the scalar shape {}.
  Shape tensorReduceShape(const Shape &input, std::int64_t axis, bool keep);

  /// Check profile and backend capability constraints after ordinary checking.
  bool checkTensorCapabilities(
      const Program &prog, TensorConsumer consumer, const TensorProfile &profile,
      DiagBag &diags
  );

} // namespace refractir
