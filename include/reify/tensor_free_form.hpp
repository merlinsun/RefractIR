#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ast/tensor.hpp"
#include "reify/tensor_region.hpp"

namespace refractir::reify {

  /// Structural policy a free-form request states. `Unguided` scores every
  /// legal action the same and carries no required motif; every other policy
  /// biases the scores and never sets a hard feature minimum.
  enum class FreeFormPolicy : std::uint8_t {
    Unguided,
    Balanced,
    Deep,
    Wide,
    Shared,
    ReductionHeavy,
    CrossState,
  };

  /// Every policy name the command line accepts, in the documented order.
  inline constexpr std::string_view freeFormPolicyNames =
      "unguided, balanced, deep, wide, shared, reduction-heavy, cross-state";

  /// Parse the command-line spelling of a free-form policy. An unknown name
  /// has no policy, so a caller refuses the request instead of generating the
  /// unguided one.
  [[nodiscard]] std::optional<FreeFormPolicy> parseFreeFormPolicy(std::string_view text);

  /// Manifest spelling of one policy.
  [[nodiscard]] std::string_view freeFormPolicyName(FreeFormPolicy policy) noexcept;

  /// How an input takes part in a counted loop. A state is rewritten by one
  /// update; an invariant is read-only, so no update may target it.
  enum class FreeFormRole : std::uint8_t { State, Invariant };

  /// What initializes an enclosing local before the loop starts.
  enum class FreeFormInit : std::uint8_t { RuntimeInput, Zero };

  /// One typed region input together with its loop role and initialization.
  struct FreeFormInput {
    TensorRegionInput value;
    FreeFormRole role = FreeFormRole::State;
    FreeFormInit init = FreeFormInit::RuntimeInput;
    /// The function parameter a runtime-initialized local copies. Empty derives
    /// `%in_<index>`; a zero-initialized input reads no parameter.
    std::string parameter;
  };

  /// One requested result root. An empty name derives `%next_<index>`.
  struct FreeFormResult {
    std::string name;
    TensorTypeInfo type;
  };

  /// Operations beyond the additive, reduction, and broadcast core. Both are
  /// off by default: the frozen corpus was built without them, and enabling a
  /// feature changes which action a seed selects.
  struct FreeFormFeatures {
    bool transpose = false;
    bool expand = false;
    bool extrema = false;
  };

  /// Bounds the builder honours. `maxNodes` counts emitted tensor operations;
  /// `maxAttempts` bounds how many seeds one request may try before refusing.
  struct FreeFormLimits {
    std::uint32_t maxNodes = 12;
    std::uint32_t maxRank = 2;
    std::uint32_t maxElements = 512;
    std::uint32_t maxAttempts = 32;
    std::uint32_t maxMultiplications = 2;
  };

  /// One free-form construction request. It names no loop, no state count, and
  /// no final expression: those belong to the assembly that consumes the
  /// region, not to the region itself.
  struct FreeFormRequest {
    std::vector<FreeFormInput> inputs;
    std::vector<FreeFormResult> results;
    FreeFormPolicy policy = FreeFormPolicy::Unguided;
    FreeFormFeatures features;
    FreeFormLimits limits;
    std::uint32_t seed = 0;
  };

  /// The observable traits one legal action has. A policy can only bias what
  /// this record states, so the bias is testable without building a graph.
  struct FreeFormActionTraits {
    TensorOp op = TensorOp::Add;
    /// Depth of the deepest operand; an input has depth zero.
    std::uint32_t depth = 0;
    /// Distinct consumers the most reused operand already has.
    std::uint32_t reusedConsumers = 0;
    /// Operands that no live node reads yet.
    std::uint32_t unusedOperands = 0;
    /// Distinct inputs the operands together depend on.
    std::uint32_t inputs = 0;
    /// Whether the action changes the value's shape.
    bool shapeChange = false;
  };

  /// The weight one policy gives one action. Every weight is positive, so a
  /// bias reorders actions without excluding any, and `Unguided` returns the
  /// same weight for every action.
  double freeFormActionWeight(FreeFormPolicy policy, const FreeFormActionTraits &traits);

  /// Choose an index from nonnegative weights with one draw in `[0, total)`.
  /// Throws std::invalid_argument when the weights are empty, negative, or
  /// carry no positive weight, and when the draw lies outside the total.
  std::uint32_t freeFormSelectWeighted(const std::vector<double> &weights, double draw);

  /// Distinct-consumer bookkeeping for one dependency graph. `consumerOperands`
  /// holds one operand list per consuming operation: a value named twice in one
  /// list is one consumer, and a value read by two operations is two.
  struct FreeFormConsumers {
    std::uint32_t total = 0;
    std::uint32_t shared = 0;
    std::vector<std::uint32_t> perValue;
  };

  /// The facts a window assembly records about its masks, one field per fact
  /// so a reader never has to infer one from another.
  struct FreeFormWindowFacts {
    /// "interior" when every row is active on every trip, otherwise "tail".
    std::string_view regime;
    /// "absent" without a physical layout, otherwise "constant" for an exact
    /// tile and "nonconstant" when v3 padding contributes lanes.
    std::string_view shapeValidity;
    /// "all_true" for an interior window, "constant_partial" when a zero
    /// stride makes the predicate independent of the trip, otherwise
    /// "runtime".
    std::string_view activePredicate;
    /// Mask survival is an observation of compiled code, so assembly states
    /// only that no observation has been made.
    std::string_view maskAfterCompilation;
  };

  /// Classify a window from active positions over every trip.
  ///
  /// Interior means the highest address any trip and row can form is below
  /// `active`; equality is a tail because the predicate is strict.
  FreeFormWindowFacts classifyWindow(
      std::uint32_t rows, std::uint32_t stride, std::uint32_t active,
      std::uint64_t tripCount, bool v3, bool paddedTile
  );

  FreeFormConsumers countFreeFormConsumers(
      std::uint32_t valueCount, const std::vector<std::vector<std::int32_t>> &consumerOperands
  );

  /// A dependency graph stated without the builder's own pool, so the
  /// statistics rules can be checked against consumers a test already knows.
  /// Values are indexed in creation order: the inputs first, then each node's
  /// result in the order the node appears.
  struct FreeFormGraphView {
    std::vector<std::string> valueNames;
    std::vector<TensorTypeInfo> valueTypes;
    std::uint32_t inputCount = 0;
    std::vector<TensorOp> nodeOps;
    std::vector<std::vector<std::int32_t>> nodeOperands;
    std::vector<std::uint32_t> roots;
  };

  /// The post-pruning statistics of one graph. Every count covers live nodes
  /// and live consumer edges only. A shape change is counted once per
  /// operation that changes its operand's shape, so a reduction and the
  /// broadcast that follows it are two changes, and neither is counted twice.
  struct FreeFormStructure {
    std::string policy;
    /// The trait the policy biased towards, named before any action is chosen.
    std::string target;
    /// The trait the pruned graph exhibits, under the precedence documented on
    /// `summarizeFreeFormGraph`.
    std::string actual;
    std::uint32_t nodes = 0;
    std::uint32_t depth = 0;
    std::uint32_t reductions = 0;
    std::uint32_t shapeChanges = 0;
    std::uint32_t distinctConsumers = 0;
    std::uint32_t sharedValues = 0;
    std::uint32_t maxDefUseDistance = 0;
    /// One mask per root, bit `k` naming input `k`. Every name in a mask is an
    /// input the root transitively depends on.
    std::vector<std::uint64_t> resultDependencies;
    /// Operator counts over live nodes, ordered by operator name.
    std::vector<std::pair<std::string, std::uint32_t>> operatorCounts;
    /// Distinct consumers of every live value, ordered by value name.
    std::vector<std::pair<std::string, std::uint32_t>> consumers;
  };

  /// Summarize the pruned graph. `actual` is the first of these traits that
  /// holds: `reduction-heavy` when at least two reductions cover half the
  /// nodes; `shared` when at least two values have two consumers and they
  /// cover half the live values; `deep` when the live depth reaches four;
  /// `cross-state` when some root depends on two or more inputs; `wide` when
  /// distinct consumers cover half the live values; otherwise `unguided`.
  [[nodiscard]] FreeFormStructure
  summarizeFreeFormGraph(const FreeFormGraphView &graph, FreeFormPolicy policy);

  /// The request's resolved inputs and result names. The assembler reads it to
  /// state roles and initializations without re-deriving them.
  struct FreeFormInputSpec {
    std::vector<FreeFormInput> inputs;
    std::vector<std::string> results;
  };

  struct FreeFormRegion {
    TensorRegion region;
    FreeFormStructure structure;
    FreeFormInputSpec inputSpec;
  };

  /// Build one region from the request. Throws std::invalid_argument for a
  /// request outside the admitted domain and std::runtime_error prefixed
  /// "free_form_exhaustion:" when no attempt places every requested result.
  [[nodiscard]] FreeFormRegion buildFreeFormRegion(const FreeFormRequest &request);

  /// One counted loop assembled from a region.
  struct FreeFormLoop {
    std::string entry = "kernel";
    /// The region's inputs, with the role and initialization each one has.
    std::vector<FreeFormInput> inputs;
    /// The region's result names, in the order the updates consume them.
    std::vector<std::string> results;
    /// Each state together with the result that rewrites it.
    std::vector<std::pair<std::string, std::string>> updates;
    std::uint32_t tripCount = 2;
    /// The returned expression: the terms folded left with `finalOp` after the
    /// loop. Each term names a state or an invariant.
    std::vector<std::string> finalTerms;
    TensorOp finalOp = TensorOp::Add;
    /// Locals the enclosing function already owns, so region temporaries and
    /// snapshots do not reuse them.
    std::vector<std::string> reservedNames{"%i", "%out"};
  };

  /// Assemble the region into one counted loop. Throws std::invalid_argument
  /// when an update targets an invariant or a state twice, when a result is
  /// not a region result, when a result type differs from its state, when a
  /// final term is not an enclosing local, or when a stated name collides with
  /// another local.
  [[nodiscard]] std::string assembleCountedLoop(const TensorRegion &region, const FreeFormLoop &loop);

  /// The canonical two-state configuration: `%acc_x` from `%x`, `%acc_y` from
  /// zero, results `%next_x`/`%next_y`, and the final `%acc_x + %acc_y`.
  struct FreeFormAssembly {
    std::string source;
    std::string refusal;
  };

  /// Assemble the canonical loop for a region, or refuse with the reason. A
  /// region that does not fit travels the target-only path unchanged instead of
  /// being forced into a canonical program.
  [[nodiscard]] FreeFormAssembly assembleCanonicalLoop(const TensorRegion &region, std::uint32_t tripCount);

} // namespace refractir::reify
