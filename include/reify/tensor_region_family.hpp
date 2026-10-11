#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

// Configuration of a bounded multi-result loop region. The enums live apart
// from the generator so the option a caller states (TensorGenerationOptions)
// and the builder that honours it can share one definition.

namespace refractir::reify {

  /// Arithmetic sampled at the named roots of a bounded loop-safe region.
  /// Additive preserves the original generation policy; Subtraction and
  /// Multiplication place their operation on every root placement; Mixed
  /// places one multiplication and one subtraction on the first two.
  enum class TensorRegionArithmeticMode : std::uint8_t {
    Additive,
    Subtraction,
    Multiplication,
    Mixed,
  };

  /// Connection skeleton of a bounded loop region. `Fold` is the original
  /// shared-producer-then-folds shape. The other families place the same
  /// operation set behind different graph connections, so a pass that folds,
  /// rematerializes, or converts layouts meets a different neighbourhood
  /// without changing the admitted arithmetic or the loop skeleton.
  enum class TensorRegionFamily : std::uint8_t {
    /// Shared add of the two states, up to two sequential reduction folds, then
    /// one arithmetic root per state.
    Fold,
    /// One shared producer feeding independent reductions on the first two
    /// axes, each broadcast into a different state root.
    OrthogonalReductions,
    /// A reduction feeds bounded arithmetic before it becomes loop state.
    ReductionArithmetic,
    /// Each next-state root depends on both current states.
    CrossState,
    /// One shared producer has consumers at different depths: a reduction path
    /// and a later arithmetic join.
    MixedDepth,
  };

  /// Every family name the command line accepts, in the documented order.
  inline constexpr std::string_view tensorRegionFamilyNames =
      "fold, orthogonal-reductions, reduction-arithmetic, cross-state, mixed-depth";

  /// Parse the command-line spelling of a region family. An unknown name has no
  /// family, so a caller refuses the request instead of generating the default.
  [[nodiscard]] inline std::optional<TensorRegionFamily>
  parseTensorRegionFamily(std::string_view text) {
    if (text == "fold")
      return TensorRegionFamily::Fold;
    if (text == "orthogonal-reductions")
      return TensorRegionFamily::OrthogonalReductions;
    if (text == "reduction-arithmetic")
      return TensorRegionFamily::ReductionArithmetic;
    if (text == "cross-state")
      return TensorRegionFamily::CrossState;
    if (text == "mixed-depth")
      return TensorRegionFamily::MixedDepth;
    return std::nullopt;
  }

} // namespace refractir::reify
