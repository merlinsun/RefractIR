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
  enum class FreeFormInit : std::uint8_t { RuntimeInput, Zero, Window };

  /// The literal kind a window fill uses. The element type decides which one
  /// is admissible.
  enum class FreeFormFillKind : std::uint8_t { Integer, Float };

  /// How each requested result picks its root among the grown values once
  /// requirements and computed-root preferences are met. `Deepest` takes the
  /// deepest value. `MostLive` takes the value that keeps the most grown nodes
  /// alive beyond the roots already chosen, so less of the body is pruned.
  enum class FreeFormRootSelection : std::uint8_t { Deepest, MostLive };

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
  /// `requireComputed` refuses a direct input as this result's root; it is a
  /// construction requirement, not a language rule, and is off by default.
  struct FreeFormResult {
    std::string name;
    TensorTypeInfo type;
    bool requireComputed = false;
  };

  /// Operations beyond the additive, reduction, and broadcast core. Both are
  /// off by default: the frozen corpus was built without them, and enabling a
  /// feature changes which action a seed selects.
  struct FreeFormFeatures {
    bool transpose = false;
    bool expand = false;
    bool extrema = false;
    /// `tensor dot` of two rank-two values whose extents Triton's `tl.dot`
    /// accepts; it counts as a multiplication.
    bool dot = false;
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

  /// One bounded live-graph requirement. It names an input and result roots
  /// the input must reach, operations the connecting path must contain, and
  /// optionally two distinct consumers of a value derived from that input that
  /// reach two distinct roots. This is a small fixed vocabulary, not a graph
  /// constraint language.
  struct FreeFormRequirement {
    std::string input;
    std::vector<std::uint32_t> results;
    /// The connecting path must contain one operation from this set.
    std::vector<TensorOp> pathOps;
    /// When nonempty, the same path must also contain one of these shape
    /// operations.
    std::vector<TensorOp> shapeOps;
    /// A value derived from the input must have two distinct live consumers
    /// whose branches reach two distinct listed roots.
    bool distinctConsumers = false;
  };

  /// One satisfied requirement, stated over the live graph and emitted names.
  struct FreeFormRequirementWitness {
    std::string input;
    std::vector<std::string> results;
    /// The witnessed source-to-result value path, input first.
    std::vector<std::string> path;
    /// Operation names on the path, in path order.
    std::vector<std::string> operations;
    /// Normalized live operation IDs on the path.
    std::vector<std::uint32_t> operationIds;
    /// Two distinct consumer values and the distinct roots they reach.
    std::vector<std::string> consumers;
    std::vector<std::string> consumerResults;
    bool distinctConsumersSatisfied = false;
  };

  /// The normalized live graph, for identity and grouping. The connectivity
  /// signature ignores extents, names, attributes, seeds, and trips; the typed
  /// fingerprint keeps the types.
  struct FreeFormLiveDag {
    std::vector<std::string> values;
    std::vector<std::string> ops;
    std::vector<std::vector<std::string>> operands;
    std::vector<std::string> roots;
    std::string connectivitySignature;
    std::string typedFingerprint;
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
    std::vector<FreeFormRequirement> requirements;
    FreeFormRootSelection rootSelection = FreeFormRootSelection::Deepest;
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
    /// Attributes of the producer nodes, parallel to `nodeOps`; empty vectors
    /// mean the view does not state them.
    std::vector<std::int64_t> nodeAxes;
    std::vector<char> nodeKeepDims;
    std::vector<Shape> nodeAttributes;
    std::vector<std::uint32_t> roots;
  };

  /// The two normalized identities of one live graph. The connectivity
  /// signature ignores value names, concrete extents, operator attributes, and
  /// creation order; the typed fingerprint keeps types and attributes.
  struct FreeFormGraphKeys {
    std::string connectivitySignature;
    std::string typedFingerprint;
  };

  /// Derive the graph keys from ordered roots and input roles. Producer IDs
  /// are assigned by a deterministic traversal from the roots, not by
  /// creation order; roots and inputs are named by position only.
  FreeFormGraphKeys freeFormGraphKeys(
      const FreeFormGraphView &graph, const std::vector<std::string> &inputRoles
  );

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
    /// One witness per requested requirement, in request order.
    std::vector<FreeFormRequirementWitness> requirementWitnesses;
    /// The normalized live graph the witnesses were derived from.
    FreeFormLiveDag liveDag;
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
  /// One window production: the invariant it writes each trip, the runtime
  /// parameter it reads, the induction variable, and the literal fill.
  struct FreeFormWindow {
    std::string result;
    std::string source;
    std::string index = "%i";
    FreeFormFillKind fillKind = FreeFormFillKind::Float;
    std::int64_t fillInteger = 0;
    double fillFloat = 0.0;
    std::uint32_t rows = 0;
    std::uint32_t stride = 0;
    std::uint32_t active = 0;
  };

  /// One assembly-only window backing parameter. It supplies window data but
  /// is not a region value, state, local copy, or result: the window reads it
  /// directly, so a backing buffer larger than the tile never enters the
  /// region's value pool.
  struct FreeFormWindowSource {
    std::string parameter;
    TensorTypeInfo type;
  };

  struct FreeFormLoop {
    std::string entry = "kernel";
    /// The region's inputs, with the role and initialization each one has.
    std::vector<FreeFormInput> inputs;
    /// Backing parameters that supply window data without entering the region.
    /// They are emitted before the runtime inputs; each must be named by a
    /// window's `source`.
    std::vector<FreeFormWindowSource> windowSources;
    /// The region's result names, in the order the updates consume them.
    std::vector<std::string> results;
    /// Each state together with the result that rewrites it.
    std::vector<std::pair<std::string, std::string>> updates;
    std::uint32_t tripCount = 2;
    /// Nonempty: the loop runs to this i32 entry parameter, emitted after the
    /// runtime inputs and ranged by `assume` from 0 to `tripCount`, which is
    /// then the most trips; the loop may run none.
    std::string tripParameter;
    /// The returned expression: the terms folded left with `finalOp` after the
    /// loop. Each term names a state or an invariant.
    std::vector<std::string> finalTerms;
    TensorOp finalOp = TensorOp::Add;
    /// Locals the enclosing function already owns, so region temporaries and
    /// snapshots do not reuse them.
    std::vector<std::string> reservedNames{"%i", "%out"};
    /// Window productions, one per window-initialized invariant.
    std::vector<FreeFormWindow> windows;
  };

  /// The address facts of one window: the highest address any trip and row
  /// can form, which positions are active on each trip, and whether every
  /// position is active on every trip.
  struct FreeFormWindowWitness {
    std::uint64_t computedMax = 0;
    std::vector<std::vector<bool>> perTripActive;
    bool allActive = false;
  };

  /// Compute the witness, refusing a spec whose rows or active bound leave the
  /// source or whose maximum address reaches the source extent.
  FreeFormWindowWitness witnessWindow(
      const FreeFormWindow &window, std::uint32_t tripCount, std::uint32_t sourceExtent
  );

  /// Check the windows a loop states before any text is emitted. Throws
  /// std::invalid_argument when a window-initialized invariant lacks a window
  /// or has more than one, when the source is not a runtime parameter, when
  /// the index is not the induction variable, when an update targets the
  /// window result, when the rows, stride, or active bound leave the source,
  /// or when the fill kind and result type disagree.
  void validateFreeFormWindows(const FreeFormLoop &loop);

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

  /// The largest magnitude f32 holds every integer up to exactly (2^24).
  inline constexpr double kExactF32IntegerBound = 16777216.0;

  /// An upper bound on the magnitude of every value the assembled loop computes
  /// in any trip, including the final expression, when every runtime input is
  /// bounded by `inputBound` in magnitude.
  [[nodiscard]] double
  freeFormMagnitudeBound(const TensorRegion &region, const FreeFormLoop &loop, double inputBound);

  /// The largest magnitude bound over every value, and over the
  /// order-sensitive values alone, across every trip of the loop.
  struct FreeFormMagnitudes {
    double largest = 0.0;
    double largestSum = 0.0;
  };

  [[nodiscard]] FreeFormMagnitudes
  freeFormMagnitudes(const TensorRegion &region, const FreeFormLoop &loop, double inputBound);

  /// Which values must stay exact for a byte-identical path comparison.
  /// `AllValues` keeps every value within 2^24. `Reductions` relies on
  /// elementwise f32 operations rounding identically on both paths when
  /// fusion is off, so it keeps only `sum` results within 2^23 (a factor-of-two
  /// margin for rounding earlier in the body) and every value within 2^64,
  /// far below the f32 overflow threshold.
  enum class FreeFormExactContract : std::uint8_t { AllValues, Reductions };

  inline constexpr double kExactReductionBound = 8388608.0;
  inline constexpr double kFiniteMarginBound = 18446744073709551616.0;

  [[nodiscard]] bool
  freeFormWithinExactContract(const FreeFormMagnitudes &magnitudes, FreeFormExactContract contract);

} // namespace refractir::reify
