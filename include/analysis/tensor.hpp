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

  enum class TensorProfileId { V1, V2, V3, V4, V5 };

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
    // The literal start; meaningless when `startParameter` names the start.
    std::int32_t initialValue = 0;
    // The i32 entry parameter the loop starts from (tensor-v5); empty for a
    // literal start.
    std::string startParameter;
    // The literal bound, or for a parameter bound its assumed maximum.
    std::int32_t bound = 0;
    // The i32 entry parameter the loop runs to; empty for a literal bound.
    std::string boundParameter;
    // The literal added by the latch: one under v2 and v3, any nonzero value
    // under tensor-v5, where a negative step pairs with a `>` header.
    std::int32_t step = 1;
    // Whether the induction variable leaves the loop holding the bound. It
    // does for the v2/v3 form; otherwise it holds `start + k * step` for the
    // `k` trips run, which may be zero.
    bool exitsAtBound = true;
    // A lower bound on every value the induction variable takes in the body.
    std::int64_t minimumValue = 0;
    // The most trips the loop can run; a parameter-bounded loop may run none.
    std::uint32_t tripCount = 0;
    int parentIndex = -1;
    std::vector<std::size_t> blocks;
  };

  /// A scalar `if` diamond: `block` branches on a comparison of i32 scalars
  /// to two single-block arms that both fall through to `join`.
  struct TensorScalarBranch {
    std::size_t block = 0;
    std::size_t thenBlock = 0;
    std::size_t elseBlock = 0;
    std::size_t join = 0;
  };

  struct TensorLoopAnalysis {
    std::vector<TensorCountedLoop> loops;
    std::vector<TensorScalarBranch> branches;
    std::uint32_t totalBodyExecutions = 0;
  };

  /// Whether a profile pads non-power-of-two extents to power-of-two tiles and
  /// masks the padded lanes (v3, and v5 which extends it).
  bool tensorProfilePadsTiles(const TensorProfile &profile);

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
