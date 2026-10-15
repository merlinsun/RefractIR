#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

// Configuration of a bounded multi-result loop region. The enums live apart
// from the generator so the option a caller states (TensorGenerationOptions)
// and the builder that honours it can share one definition. Every spelling
// here is the one the command line and the generated manifest both use.

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
    /// A loop-carried state is reduced, broadcast back over the state shape, and
    /// the view is consumed twice: by a second reduction on the other axis, and
    /// by the elementwise path into a root. The two consumers can be reached by
    /// a recomputed view of a small operand, so a pass that would rather
    /// recompute the broadcast in a consumer's layout than convert the
    /// broadcast's result meets a value whose original computation stays live.
    BroadcastHoist,
  };

  /// Every family name the command line accepts, in the documented order.
  inline constexpr std::string_view tensorRegionFamilyNames =
      "fold, orthogonal-reductions, reduction-arithmetic, cross-state, mixed-depth, "
      "broadcast-hoist";

  /// Every arithmetic mode name the command line accepts, in the documented
  /// order. Additive is the policy an unnamed request keeps.
  inline constexpr std::string_view tensorRegionArithmeticNames =
      "additive, subtraction, multiplication, mixed";

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
    if (text == "broadcast-hoist")
      return TensorRegionFamily::BroadcastHoist;
    return std::nullopt;
  }

  /// Parse the command-line spelling of a region arithmetic mode. An unknown
  /// name has no mode, so a caller refuses the request instead of generating
  /// the default.
  [[nodiscard]] inline std::optional<TensorRegionArithmeticMode>
  parseTensorRegionArithmeticMode(std::string_view text) {
    if (text == "additive")
      return TensorRegionArithmeticMode::Additive;
    if (text == "subtraction")
      return TensorRegionArithmeticMode::Subtraction;
    if (text == "multiplication")
      return TensorRegionArithmeticMode::Multiplication;
    if (text == "mixed")
      return TensorRegionArithmeticMode::Mixed;
    return std::nullopt;
  }

  /// Manifest spelling of one family.
  [[nodiscard]] inline std::string_view tensorRegionFamilyName(TensorRegionFamily family) {
    switch (family) {
      case TensorRegionFamily::Fold:
        return "fold";
      case TensorRegionFamily::OrthogonalReductions:
        return "orthogonal-reductions";
      case TensorRegionFamily::ReductionArithmetic:
        return "reduction-arithmetic";
      case TensorRegionFamily::CrossState:
        return "cross-state";
      case TensorRegionFamily::MixedDepth:
        return "mixed-depth";
      case TensorRegionFamily::BroadcastHoist:
        return "broadcast-hoist";
    }
    return "unknown";
  }

  /// Manifest spelling of one arithmetic mode.
  [[nodiscard]] inline std::string_view
  tensorRegionArithmeticName(TensorRegionArithmeticMode mode) {
    switch (mode) {
      case TensorRegionArithmeticMode::Additive:
        return "additive";
      case TensorRegionArithmeticMode::Subtraction:
        return "subtraction";
      case TensorRegionArithmeticMode::Multiplication:
        return "multiplication";
      case TensorRegionArithmeticMode::Mixed:
        return "mixed";
    }
    return "unknown";
  }

} // namespace refractir::reify
