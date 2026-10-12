#include "reify/tensor_graph_gen.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "analysis/tensor.hpp"
#include "analysis/type_utils.hpp"

// The compositional builder grows a typed DAG forward from the runtime entry
// parameters and only then emits SIR. Every value it creates is consumed by a
// later value on the path to the return value, so the emitted program is
// straight-line, topologically ordered, and free of dead stores. Types and
// shapes are inferred before a node is created; the caller re-checks the
// printed source, so inference only has to stay conservative.
namespace refractir::reify {
  namespace {

    /// Independent construction attempts inside one builder call. The driver
    /// still owns the outer attempt budget; this bound only stops a builder
    /// that cannot place a node from looping.
    constexpr std::uint32_t constructAttempts = 32;

    /// The tensor case manifest records a twelve-node generation limit, so the
    /// builder never exceeds it even when the caller passes a larger cap.
    constexpr std::uint32_t nodeLimit = 12;

    /// Multiplication is admitted only against a small nonzero literal, and
    /// only this often. Every other operator is additive, so with at most four
    /// doublings a starting magnitude of two stays far below the signed 32-bit
    /// range even after a 512-element reduction and its re-expansion.
    constexpr std::uint32_t mulLimit = 4;

    /// A per-attempt construction failure that another RNG stream may resolve.
    /// Structural impossibility is reported as graph_exhaustion instead.
    struct AttemptFailure : std::runtime_error {
      explicit AttemptFailure(const std::string &message) : std::runtime_error(message) {}
    };

    struct GraphValue {
      std::string name;
      TypePtr type;
      Shape shape; // empty for a scalar
      TensorElement element = TensorElement::I32;
      int node = -1; // producing node, -1 for an entry parameter, -2 if pruned
    };

    struct GraphNode {
      TensorOp op = TensorOp::Add;
      bool copy = false;
      std::vector<int> operands;            // value indices, -1 selects a literal
      std::vector<double> literals;         // literal at the same position
      std::vector<std::uint32_t> attribute; // broadcast shape or permutation
      std::int64_t axis = 0;
      bool keepDims = false;
      int result = 0;
    };

    struct Graph {
      std::vector<GraphValue> values;
      std::vector<GraphNode> nodes;
      std::vector<int> roots;
    };

    void prune(Graph &graph);

    /// Mark every value `value` was transitively computed from.
    std::vector<char> collectAncestors(const Graph &graph, int value);

    enum class StepKind { Merge, Binary, Shape, Reduce, Diamond, DoubleReduction, Copy };

    constexpr std::uint32_t stepCost(StepKind kind) {
      switch (kind) {
        case StepKind::Merge:
        case StepKind::Binary:
        case StepKind::Shape:
        case StepKind::Copy:
          return 1;
        case StepKind::Reduce:
          return 3;
        case StepKind::Diamond:
          return 4;
        case StepKind::DoubleReduction:
          return 5;
      }
      return 1;
    }

    TypePtr scalarType(TensorElement element) {
      auto type = std::make_shared<Type>();
      if (element == TensorElement::I32) {
        IntType value;
        value.kind = IntType::Kind::I32;
        type->v = value;
      } else {
        FloatType value;
        value.kind = FloatType::Kind::F32;
        type->v = value;
      }
      return type;
    }

    TypePtr tensorType(const Shape &shape, TensorElement element) {
      auto type = std::make_shared<Type>();
      TensorType value;
      value.shape = shape;
      value.elem = scalarType(element);
      type->v = std::move(value);
      return type;
    }

    Coef literalCoef(TensorElement element, double value) {
      if (element == TensorElement::I32)
        return IntLit{static_cast<std::int64_t>(value), {}};
      return FloatLit{value, {}};
    }

    Expr localExpression(const std::string &name) {
      LValue lvalue;
      lvalue.base = LocalId{name, {}};
      return Expr{Atom{RValueAtom{std::move(lvalue), {}}, {}}, {}, {}};
    }

    class Builder {
    public:
      Builder(
          std::mt19937 &random, const TensorGenerationOptions &options,
          const TensorGraphLimits &limits,
          TensorRegionArithmeticMode arithmetic = TensorRegionArithmeticMode::Additive,
          TensorRegionFamily family = TensorRegionFamily::Fold
      ) :
          random_(random),
          profile_(tensorProfile(options.profileVersion)),
          element_(options.dtype == "i32" ? TensorElement::I32 : TensorElement::F32),
          regionArithmetic_(arithmetic),
          regionFamily_(family) {
        if (options.profileVersion != TensorProfileId::V2 &&
            options.profileVersion != TensorProfileId::V3)
          throw std::runtime_error("compositional generation supports tensor-v2 and tensor-v3 only"
          );
        if (options.dtype != "i32" && options.dtype != "f32")
          throw std::invalid_argument("compositional tensor elements must be i32 or f32");
        maxRank_ = std::min(limits.maxRank, profile_.maxRank);
        maxElements_ = std::min(limits.maxElements, profile_.maxElements);
        budget_ = std::min(limits.maxOperators, nodeLimit);
        maxInputs_ = limits.maxInputs;
      }

      Graph build();
      Graph buildRegion(
          const std::vector<TensorRegionInput> &inputs, const std::vector<std::string> &resultNames
      );

    private:
      /// The region's two loop states, in the order the caller supplies them:
      /// the first carries the runtime input, the second starts at zero. The
      /// families' additive producer, nonnegative subtraction, and live state
      /// updates are stated over that contract.
      static constexpr std::uint32_t inputState = 0;
      static constexpr std::uint32_t zeroState = 1;

      Shape validateRegion(
          const std::vector<TensorRegionInput> &inputs, const std::vector<std::string> &resultNames
      );
      Graph seedRegion(const std::vector<TensorRegionInput> &inputs) const;
      Graph buildRegionFolds(
          const std::vector<TensorRegionInput> &inputs, const std::vector<std::string> &resultNames,
          const Shape &shape
      );
      Graph buildRegionFamily(
          const std::vector<TensorRegionInput> &inputs, const std::vector<std::string> &resultNames,
          const Shape &shape
      );
      int emitSum(Graph &graph, int value, std::uint32_t axis);
      int emitBroadcast(Graph &graph, int value, const Shape &shape);
      int emitBinaryOp(Graph &graph, int lhs, int rhs, TensorOp op, const Shape &shape);
      void requireNodeBudget(std::uint32_t nodes) const;
      int emitProducer(Graph &graph, const Shape &shape);
      int
      emitPlacement(Graph &graph, int value, int other, std::size_t placement, const Shape &shape);
      std::vector<std::uint32_t> reducibleAxes(const Shape &shape) const;
      std::uint32_t
      chooseAxis(const Shape &shape, std::optional<std::uint32_t> excluded = std::nullopt);
      TensorOp placementOperation(std::size_t placement) const;
      std::array<std::uint32_t, 2> placementStates();

      bool admissible(const Shape &shape) const;
      Shape makeWorkingShape();
      Shape chooseTailedShape();
      Shape makeInputShape(const Shape &working);
      std::uint32_t chooseExtent(bool allowTail);
      std::optional<Shape> applyBinary(const Shape &lhs, const Shape &rhs) const;
      int appendValue(Graph &graph, const Shape &shape);
      int appendNode(Graph &graph, GraphNode node, const Shape &shape);
      int emitPair(Graph &graph, int current, int other, double literal, TensorOp op);
      int emitForkJoin(Graph &graph, int left, int right);
      int emitMerge(Graph &graph, int current, int input);
      int emitBinary(Graph &graph, int current);
      int emitShape(Graph &graph, int current);
      int emitReduce(Graph &graph, int current);
      int emitDiamond(Graph &graph, int current);
      int emitDoubleReduction(Graph &graph, int current);
      int emitCopy(Graph &graph, int current);
      int emitRegionFold(Graph &graph, int current, const Shape &shape);
      std::uint32_t pickAxisWithExtent(const Shape &shape);
      double chooseLiteral(bool multiplication);
      StepKind pickFiller(std::uint32_t remaining);

      std::mt19937 &random_;
      TensorProfile profile_;
      TensorElement element_;
      std::uint32_t maxRank_ = 1;
      std::uint32_t maxElements_ = 1;
      std::uint32_t budget_ = 0;
      std::uint32_t maxInputs_ = 0;
      std::uint32_t temporaries_ = 0;
      std::uint32_t multiplications_ = 0;
      TensorRegionArithmeticMode regionArithmetic_ = TensorRegionArithmeticMode::Additive;
      TensorRegionFamily regionFamily_ = TensorRegionFamily::Fold;
      std::vector<std::string> reservedNames_;
    };

    bool Builder::admissible(const Shape &shape) const {
      if (shape.empty())
        return true; // a scalar is only ever a reduction result
      if (shape.size() > maxRank_)
        return false;
      std::uint64_t elements = 1;
      for (const auto dimension: shape) {
        if (dimension == 0)
          return false;
        elements *= dimension;
      }
      if (elements > maxElements_)
        return false;
      try {
        (void) tensorExecutionTileShape(shape, profile_);
      } catch (const std::exception &) {
        return false;
      }
      return true;
    }

    std::uint32_t Builder::chooseExtent(bool allowTail) {
      // v2 dimensions stay powers of two; v3 additionally admits the logical
      // tails that exercise its canonicalization.
      static constexpr std::array<std::uint32_t, 10> small{1, 1, 1, 2, 2, 4, 4, 8, 16, 32};
      static constexpr std::array<std::uint32_t, 5> tails{17, 31, 33, 63, 65};
      if (allowTail && profile_.id == TensorProfileId::V3 && random_() % 3 == 0)
        return tails[random_() % tails.size()];
      return small[random_() % small.size()];
    }

    Shape Builder::chooseTailedShape() {
      // Rank-two logical tails whose padded tile stays inside the profile
      // volume. Each pair keeps one tailed axis and one power-of-two companion,
      // so a reduce or broadcast meets a masked axis beside a full one.
      static const std::array<Shape, 6> tails{
          Shape{31, 8}, Shape{33, 8}, Shape{8, 31}, Shape{8, 33}, Shape{63, 4}, Shape{4, 65}};
      return tails[random_() % tails.size()];
    }

    Shape Builder::makeWorkingShape() {
      for (std::uint32_t attempt = 0; attempt < 64; ++attempt) {
        const auto rank = 1 + random_() % maxRank_;
        Shape shape(rank);
        if (rank == 2 && profile_.id == TensorProfileId::V3 && random_() % 3 == 0)
          shape = chooseTailedShape();
        else
          for (auto &dimension: shape)
            dimension = chooseExtent(rank == 1);
        const bool reducible = std::any_of(shape.begin(), shape.end(), [](std::uint32_t dimension) {
          return dimension > 1;
        });
        if (!reducible)
          shape[random_() % rank] = profile_.id == TensorProfileId::V3 ? 3 : 2;
        if (admissible(shape))
          return shape;
      }
      throw AttemptFailure("cannot choose an admissible working shape");
    }

    Shape Builder::makeInputShape(const Shape &working) {
      if (working.size() == 1)
        return working;
      const auto keep = 1 + random_() % static_cast<std::uint32_t>(working.size());
      return Shape(working.end() - static_cast<std::ptrdiff_t>(keep), working.end());
    }

    std::optional<Shape> Builder::applyBinary(const Shape &lhs, const Shape &rhs) const {
      try {
        return tensorBroadcastShape(lhs, rhs);
      } catch (const std::exception &) {
        return std::nullopt;
      }
    }

    int Builder::appendValue(Graph &graph, const Shape &shape) {
      GraphValue value;
      do {
        value.name = "%t" + std::to_string(temporaries_++);
      } while (std::find(reservedNames_.begin(), reservedNames_.end(), value.name) !=
                   reservedNames_.end() ||
               std::any_of(graph.values.begin(), graph.values.end(), [&](const GraphValue &prior) {
                 return prior.name == value.name;
               }));
      value.shape = shape;
      value.element = element_;
      value.type = shape.empty() ? scalarType(element_) : tensorType(shape, element_);
      graph.values.push_back(std::move(value));
      return static_cast<int>(graph.values.size()) - 1;
    }

    int Builder::appendNode(Graph &graph, GraphNode node, const Shape &shape) {
      const auto result = appendValue(graph, shape);
      node.result = result;
      graph.values[static_cast<std::size_t>(result)].node = static_cast<int>(graph.nodes.size());
      graph.nodes.push_back(std::move(node));
      return result;
    }

    double Builder::chooseLiteral(bool multiplication) {
      // Zero would erase the dependency of every later value on its operands.
      // Multiplication also avoids the exact +1 identity.
      static constexpr std::array<double, 4> additive{1.0, -1.0, 2.0, -2.0};
      static constexpr std::array<double, 3> multiplicative{-1.0, 2.0, -2.0};
      if (multiplication)
        return multiplicative[random_() % multiplicative.size()];
      return additive[random_() % additive.size()];
    }

    std::uint32_t Builder::pickAxisWithExtent(const Shape &shape) {
      std::vector<std::uint32_t> axes;
      for (std::uint32_t axis = 0; axis < shape.size(); ++axis)
        if (shape[axis] > 1)
          axes.push_back(axis);
      return axes[random_() % axes.size()];
    }

    int Builder::emitPair(Graph &graph, int current, int other, double literal, TensorOp op) {
      const Shape currentShape = graph.values[static_cast<std::size_t>(current)].shape;
      Shape result = currentShape;
      if (other >= 0) {
        const auto merged =
            applyBinary(currentShape, graph.values[static_cast<std::size_t>(other)].shape);
        if (!merged)
          throw AttemptFailure("binary operands do not broadcast");
        result = *merged;
      }
      if (!admissible(result))
        throw AttemptFailure("binary result shape is outside the configured limits");
      GraphNode node;
      node.op = op;
      node.operands = {current, other};
      node.literals = {0.0, literal};
      return appendNode(graph, std::move(node), result);
    }

    int Builder::emitForkJoin(Graph &graph, int left, int right) {
      // Two bounded operations on separate entry parameters, combined by a join
      // node. Neither branch is an ancestor of the other, so the join carries a
      // dependency on two different runtime inputs instead of on one running
      // value. The join broadcasts to the same shape a plain merge would have
      // produced, so every later merge still sees a broadcast superset of the
      // remaining inputs.
      const auto merged = applyBinary(
          graph.values[static_cast<std::size_t>(left)].shape,
          graph.values[static_cast<std::size_t>(right)].shape
      );
      if (!merged || !admissible(*merged))
        return emitMerge(graph, left, right);
      const bool multiply = multiplications_ < mulLimit && random_() % 4 == 0;
      if (multiply)
        ++multiplications_;
      // A subtraction on the join operand could cancel the running value that
      // already carries this input, so the join itself stays additive.
      const auto leftOperator =
          multiply ? TensorOp::Mul : (random_() % 2 == 0 ? TensorOp::Add : TensorOp::Sub);
      const auto first = emitPair(graph, left, -1, chooseLiteral(multiply), leftOperator);
      const auto second = emitPair(graph, right, -1, chooseLiteral(false), TensorOp::Add);
      return emitPair(graph, first, second, 0.0, TensorOp::Add);
    }

    int Builder::emitMerge(Graph &graph, int current, int input) {
      // Entry parameters are merged before any shape-changing step, so the
      // running value stays a broadcast-compatible superset of every input.
      return emitPair(
          graph, current, input, 0.0, random_() % 2 == 0 ? TensorOp::Add : TensorOp::Sub
      );
    }

    int Builder::emitBinary(Graph &graph, int current) {
      const bool multiply = multiplications_ < mulLimit && random_() % 4 == 0;
      const auto op =
          multiply ? TensorOp::Mul : (random_() % 2 == 0 ? TensorOp::Add : TensorOp::Sub);
      int other = -1;
      if (!multiply && random_() % 2 == 0) {
        // Fan-out from an earlier value keeps shared intermediates observable.
        // Subtracting an ancestor of the running value would cancel the very
        // dependency the later counterfactual check looks for, so that pairing
        // is not offered.
        const auto ancestors =
            op == TensorOp::Sub ? collectAncestors(graph, current) : std::vector<char>();
        std::vector<int> candidates;
        const Shape currentShape = graph.values[static_cast<std::size_t>(current)].shape;
        for (std::size_t index = 0; index < graph.values.size(); ++index) {
          if (static_cast<int>(index) == current || graph.values[index].shape.empty())
            continue;
          if (index < ancestors.size() && ancestors[index])
            continue;
          const auto merged = applyBinary(currentShape, graph.values[index].shape);
          if (merged && admissible(*merged))
            candidates.push_back(static_cast<int>(index));
        }
        if (!candidates.empty())
          other = candidates[random_() % candidates.size()];
      }
      if (multiply)
        ++multiplications_;
      return emitPair(graph, current, other, chooseLiteral(multiply), op);
    }

    int Builder::emitShape(Graph &graph, int current) {
      const Shape shape = graph.values[static_cast<std::size_t>(current)].shape;
      std::vector<int> choices;
      if (shape.size() >= 2)
        choices.push_back(0); // transpose
      if (shape.size() < maxRank_)
        choices.push_back(1); // expand
      if (std::find(shape.begin(), shape.end(), 1) != shape.end())
        choices.push_back(2); // broadcast
      if (choices.empty())
        return emitBinary(graph, current);
      const auto choice = choices[random_() % choices.size()];

      if (choice == 0) {
        std::vector<std::uint32_t> permutation(shape.size());
        std::iota(permutation.begin(), permutation.end(), 0u);
        do {
          std::shuffle(permutation.begin(), permutation.end(), random_);
        } while (std::is_sorted(permutation.begin(), permutation.end()));
        Shape output(shape.size());
        for (std::size_t axis = 0; axis < permutation.size(); ++axis)
          output[axis] = shape[permutation[axis]];
        if (!admissible(output))
          return emitBinary(graph, current);
        GraphNode node;
        node.op = TensorOp::Transpose;
        node.operands = {current};
        node.attribute = std::move(permutation);
        return appendNode(graph, std::move(node), output);
      }

      if (choice == 1) {
        const auto axis = random_() % (shape.size() + 1);
        Shape output = shape;
        output.insert(output.begin() + static_cast<std::ptrdiff_t>(axis), 1);
        if (!admissible(output))
          return emitBinary(graph, current);
        GraphNode node;
        node.op = TensorOp::Expand;
        node.operands = {current};
        node.axis = static_cast<std::int64_t>(axis);
        return appendNode(graph, std::move(node), output);
      }

      std::vector<std::uint32_t> singletonAxes;
      for (std::uint32_t axis = 0; axis < shape.size(); ++axis)
        if (shape[axis] == 1)
          singletonAxes.push_back(axis);
      std::shuffle(singletonAxes.begin(), singletonAxes.end(), random_);
      for (const auto axis: singletonAxes) {
        for (std::uint32_t attempt = 0; attempt < 8; ++attempt) {
          Shape output = shape;
          output[axis] = chooseExtent(false);
          // Broadcasting to the shape the value already has would erase the
          // dimension change the test exists to observe.
          if (output[axis] == 1 || output == shape || !admissible(output))
            continue;
          GraphNode node;
          node.op = TensorOp::Broadcast;
          node.operands = {current};
          node.attribute = output;
          return appendNode(graph, std::move(node), output);
        }
      }
      return emitBinary(graph, current);
    }

    int Builder::emitReduce(Graph &graph, int current) {
      const Shape shape = graph.values[static_cast<std::size_t>(current)].shape;
      const bool reducible = std::any_of(shape.begin(), shape.end(), [](std::uint32_t dimension) {
        return dimension > 1;
      });
      if (!reducible)
        return emitBinary(graph, current);
      const auto axis = pickAxisWithExtent(shape);
      const bool keep = random_() % 2 == 0;

      GraphNode reduction;
      reduction.op = TensorOp::Sum;
      reduction.operands = {current};
      reduction.axis = static_cast<std::int64_t>(axis);
      reduction.keepDims = keep;
      Shape reduced = shape;
      if (keep)
        reduced[axis] = 1;
      else
        reduced.erase(reduced.begin() + static_cast<std::ptrdiff_t>(axis));
      const auto rows = appendNode(graph, std::move(reduction), reduced);

      if (keep)
        return emitPair(
            graph, current, rows, 0.0, random_() % 2 == 0 ? TensorOp::Add : TensorOp::Sub
        );

      // A rank-one drop is the scalar reduction boundary: expand turns the
      // scalar back into a tensor before the result rejoins the running value.
      GraphNode expansion;
      expansion.op = TensorOp::Expand;
      expansion.operands = {rows};
      expansion.axis = static_cast<std::int64_t>(axis);
      Shape expanded = reduced;
      expanded.insert(expanded.begin() + static_cast<std::ptrdiff_t>(axis), 1);
      const auto restored = appendNode(graph, std::move(expansion), expanded);
      return emitPair(
          graph, current, restored, 0.0, random_() % 2 == 0 ? TensorOp::Add : TensorOp::Sub
      );
    }

    int Builder::emitDiamond(Graph &graph, int current) {
      // A fan-out whose branches are combined with `add` keeps the shared
      // value's coefficient at two; subtracting the two branches would cancel
      // the shared value and every input behind it.
      const auto shared = emitPair(
          graph, current, -1, chooseLiteral(false),
          random_() % 2 == 0 ? TensorOp::Add : TensorOp::Sub
      );
      const auto left = emitPair(
          graph, shared, -1, chooseLiteral(false),
          random_() % 2 == 0 ? TensorOp::Add : TensorOp::Sub
      );
      const bool scaleRight = multiplications_ < mulLimit && random_() % 3 == 0;
      int right = 0;
      if (scaleRight) {
        ++multiplications_;
        right = emitPair(graph, shared, -1, 2.0, TensorOp::Mul);
      } else {
        right = emitPair(
            graph, shared, -1, chooseLiteral(false),
            random_() % 2 == 0 ? TensorOp::Add : TensorOp::Sub
        );
      }
      return emitPair(graph, left, right, 0.0, TensorOp::Add);
    }

    int Builder::emitDoubleReduction(Graph &graph, int current) {
      const Shape shape = graph.values[static_cast<std::size_t>(current)].shape;
      const bool reducible = std::any_of(shape.begin(), shape.end(), [](std::uint32_t dimension) {
        return dimension > 1;
      });
      if (!reducible)
        return emitBinary(graph, current);
      const auto firstAxis = pickAxisWithExtent(shape);
      const auto secondAxis = pickAxisWithExtent(shape);

      // The first reduction is consumed by the subtraction and the second by
      // the expansion that restores the running shape: distinct live consumers.
      GraphNode first;
      first.op = TensorOp::Sum;
      first.operands = {current};
      first.axis = static_cast<std::int64_t>(firstAxis);
      first.keepDims = true;
      Shape kept = shape;
      kept[firstAxis] = 1;
      const auto firstRows = appendNode(graph, std::move(first), kept);
      const auto mixed = emitPair(
          graph, current, firstRows, 0.0, random_() % 2 == 0 ? TensorOp::Add : TensorOp::Sub
      );

      GraphNode second;
      second.op = TensorOp::Sum;
      second.operands = {mixed};
      second.axis = static_cast<std::int64_t>(secondAxis);
      second.keepDims = false;
      Shape dropped = shape;
      dropped.erase(dropped.begin() + static_cast<std::ptrdiff_t>(secondAxis));
      const auto secondRows = appendNode(graph, std::move(second), dropped);

      GraphNode expansion;
      expansion.op = TensorOp::Expand;
      expansion.operands = {secondRows};
      expansion.axis = static_cast<std::int64_t>(secondAxis);
      Shape expanded = dropped;
      expanded.insert(expanded.begin() + static_cast<std::ptrdiff_t>(secondAxis), 1);
      const auto restored = appendNode(graph, std::move(expansion), expanded);
      return emitPair(
          graph, mixed, restored, 0.0, random_() % 2 == 0 ? TensorOp::Add : TensorOp::Sub
      );
    }

    int Builder::emitCopy(Graph &graph, int current) {
      GraphNode node;
      node.copy = true;
      node.operands = {current};
      const Shape shape = graph.values[static_cast<std::size_t>(current)].shape;
      return appendNode(graph, std::move(node), shape);
    }

    int Builder::emitRegionFold(Graph &graph, int current, const Shape &shape) {
      if (shape.empty())
        throw AttemptFailure("multi-output region folding needs a tensor shape");
      const auto axis = static_cast<std::uint32_t>(random_() % shape.size());

      GraphNode reduction;
      reduction.op = TensorOp::Sum;
      reduction.operands = {current};
      reduction.axis = static_cast<std::int64_t>(axis);
      reduction.keepDims = true;
      Shape reducedShape = shape;
      reducedShape[axis] = 1;
      const auto reduced = appendNode(graph, std::move(reduction), reducedShape);

      GraphNode broadcast;
      broadcast.op = TensorOp::Broadcast;
      broadcast.operands = {reduced};
      broadcast.attribute = shape;
      const auto expanded = appendNode(graph, std::move(broadcast), shape);
      return emitPair(graph, current, expanded, 0.0, TensorOp::Add);
    }

    StepKind Builder::pickFiller(std::uint32_t remaining) {
      static constexpr std::array<StepKind, 4> kinds{
          StepKind::Binary, StepKind::Shape, StepKind::Reduce, StepKind::Copy};
      static constexpr std::array<std::uint32_t, 4> weights{4, 3, 3, 1};
      std::array<StepKind, 4> choices{};
      std::array<std::uint32_t, 4> choiceWeights{};
      std::size_t count = 0;
      for (std::size_t index = 0; index < kinds.size(); ++index) {
        if (stepCost(kinds[index]) > remaining)
          continue;
        choices[count] = kinds[index];
        choiceWeights[count] = weights[index];
        ++count;
      }
      if (count == 0)
        return StepKind::Binary;
      const auto total =
          std::accumulate(choiceWeights.begin(), choiceWeights.begin() + count, std::uint32_t{0});
      auto pick = random_() % total;
      for (std::size_t index = 0; index < count; ++index) {
        if (pick < choiceWeights[index])
          return choices[index];
        pick -= choiceWeights[index];
      }
      return choices[0];
    }

    Graph Builder::build() {
      if (maxInputs_ < 2)
        throw std::runtime_error(
            "graph_exhaustion: compositional graphs need at least two runtime inputs"
        );
      if (maxRank_ == 0 || maxElements_ == 0)
        throw std::runtime_error("graph_exhaustion: compositional limits admit no tensor shape");

      const auto tensorInputs = 2 + random_() % (std::min(maxInputs_, 3u) - 1u);
      const bool scalarInput = maxInputs_ >= 3 && tensorInputs == 2 && random_() % 3 == 0;
      const auto inputCount = tensorInputs + (scalarInput ? 1u : 0u);
      const auto mergeCount = inputCount - 1;
      // The merge phase opens with a fork-join: one bounded operation on each of
      // two separate runtime inputs, combined by a join node. It replaces the
      // first merge and costs two nodes more, so the budget left for the
      // remaining features shrinks while the operator budget does not change.
      const auto mergeSteps = mergeCount - 1;
      const auto mergeCost = mergeCount + 2;
      const auto required = mergeCost + stepCost(StepKind::DoubleReduction);
      if (budget_ < required)
        throw std::runtime_error(
            "graph_exhaustion: operator budget cannot host the required graph features"
        );
      // The twelve-node limit cannot hold the explicit diamond beside the
      // independent-branch join and the two-reduction feature. A shared
      // intermediate with two live consumers that rejoin still exists without
      // it: the first reduction and the mixer of `emitDoubleReduction` both read
      // the running value and meet again at the mixer.
      const bool wantDiamond = budget_ >= required + stepCost(StepKind::Diamond);
      // The target draw keeps the position it has always had in this stream, so
      // a fixed seed still selects the same working shape and input shapes. The
      // graph around them changes; the tensors the solver has to reason about do
      // not, which is what keeps an exact-selection query inside its budget.
      const auto featureCost = stepCost(StepKind::DoubleReduction) +
                               (wantDiamond ? stepCost(StepKind::Diamond) : std::uint32_t{0});
      const auto minimum = mergeCost + featureCost;
      const auto target = minimum + random_() % (budget_ - minimum + 1);

      const Shape working = makeWorkingShape();
      static constexpr std::array<std::string_view, 3> tensorNames{"%a", "%b", "%c"};
      Graph graph;
      for (std::uint32_t index = 0; index < tensorInputs; ++index) {
        GraphValue value;
        value.name = std::string(tensorNames[index]);
        value.shape = index == 0 ? working : makeInputShape(working);
        value.element = element_;
        value.type = tensorType(value.shape, element_);
        graph.values.push_back(std::move(value));
      }
      if (scalarInput) {
        GraphValue value;
        value.name = "%s";
        value.element = element_;
        value.type = scalarType(element_);
        graph.values.push_back(std::move(value));
      }

      std::vector<int> pending;
      for (std::uint32_t index = 1; index < inputCount; ++index)
        pending.push_back(static_cast<int>(index));
      std::shuffle(pending.begin(), pending.end(), random_);
      // Each fork branch applies a tensor operator, so the first pending input
      // has to be a tensor; a scalar input still merges like any other.
      const auto tensorPartner = std::find_if(pending.begin(), pending.end(), [&](int index) {
        return !graph.values[static_cast<std::size_t>(index)].shape.empty();
      });
      if (tensorPartner != pending.end())
        std::iter_swap(pending.begin(), tensorPartner);

      // The required features are placed at random positions after the input
      // merges; the remaining budget is filled with independently chosen
      // operators, so neither placement nor operand choice repeats per seed.
      std::vector<StepKind> schedule(mergeSteps, StepKind::Merge);
      std::vector<StepKind> features{StepKind::DoubleReduction};
      if (wantDiamond)
        features.push_back(StepKind::Diamond);
      std::shuffle(features.begin(), features.end(), random_);
      for (const auto feature: features) {
        const auto span = schedule.size() - mergeSteps + 1;
        const auto position = mergeSteps + random_() % span;
        schedule.insert(schedule.begin() + static_cast<std::ptrdiff_t>(position), feature);
      }
      std::uint32_t used = minimum;
      // Fillers land anywhere after the input merges rather than only at the
      // end, so a shape change can sit upstream of a reduction. Appending them
      // would leave every transpose, expand, and broadcast downstream of every
      // sum, and a reduction that consumes a reshaped operand is a different
      // structure to compile.
      while (used < target) {
        const auto step = pickFiller(target - used);
        const auto span = schedule.size() - mergeSteps + 1;
        const auto position = mergeSteps + random_() % span;
        schedule.insert(schedule.begin() + static_cast<std::ptrdiff_t>(position), step);
        used += stepCost(step);
      }

      int current = 0;
      std::size_t merges = 0;
      current = emitForkJoin(graph, current, pending[merges++]);
      for (const auto step: schedule) {
        switch (step) {
          case StepKind::Merge:
            current = emitMerge(graph, current, pending[merges++]);
            break;
          case StepKind::Binary:
            current = emitBinary(graph, current);
            break;
          case StepKind::Shape:
            current = emitShape(graph, current);
            break;
          case StepKind::Reduce:
            current = emitReduce(graph, current);
            break;
          case StepKind::Diamond:
            current = emitDiamond(graph, current);
            break;
          case StepKind::DoubleReduction:
            current = emitDoubleReduction(graph, current);
            break;
          case StepKind::Copy:
            current = emitCopy(graph, current);
            break;
        }
      }
      graph.roots = {current};
      prune(graph);
      return graph;
    }

    void Builder::requireNodeBudget(std::uint32_t nodes) const {
      if (budget_ < nodes)
        throw AttemptFailure("operator budget cannot host the requested region family");
    }

    int Builder::emitSum(Graph &graph, int value, std::uint32_t axis) {
      const Shape shape = graph.values[static_cast<std::size_t>(value)].shape;
      if (axis >= shape.size())
        throw AttemptFailure("reduction axis is outside the region rank");
      if (shape[axis] < 2)
        throw AttemptFailure("reduction axis is a singleton and carries no reduction");
      GraphNode reduction;
      reduction.op = TensorOp::Sum;
      reduction.operands = {value};
      reduction.axis = static_cast<std::int64_t>(axis);
      reduction.keepDims = true;
      Shape reduced = shape;
      reduced[axis] = 1;
      return appendNode(graph, std::move(reduction), reduced);
    }

    int Builder::emitBroadcast(Graph &graph, int value, const Shape &shape) {
      const Shape current = graph.values[static_cast<std::size_t>(value)].shape;
      if (current == shape)
        throw AttemptFailure("broadcast target equals the operand shape");
      if (applyBinary(current, shape) != shape)
        throw AttemptFailure("broadcast target is not a superset of the operand shape");
      GraphNode broadcast;
      broadcast.op = TensorOp::Broadcast;
      broadcast.operands = {value};
      broadcast.attribute = shape;
      return appendNode(graph, std::move(broadcast), shape);
    }

    int Builder::emitBinaryOp(Graph &graph, int lhs, int rhs, TensorOp op, const Shape &shape) {
      const auto merged = applyBinary(
          graph.values[static_cast<std::size_t>(lhs)].shape,
          graph.values[static_cast<std::size_t>(rhs)].shape
      );
      if (!merged || *merged != shape)
        throw AttemptFailure("region family binary result is not the state shape");
      GraphNode node;
      node.op = op;
      node.operands = {lhs, rhs};
      node.literals = {0.0, 0.0};
      return appendNode(graph, std::move(node), *merged);
    }

    int Builder::emitProducer(Graph &graph, const Shape &shape) {
      // The producer reads both states, and the second one starts at zero: a
      // product with it is a constant zero on every trip, and a difference
      // cancels one state out of the update. It therefore stays additive, which
      // is also the skeleton every exact region certificate is stated over. The
      // requested mode is carried by the root placements instead.
      if (random_() % 2 == 0)
        return emitBinaryOp(graph, 1, 0, TensorOp::Add, shape);
      return emitBinaryOp(graph, 0, 1, TensorOp::Add, shape);
    }

    int Builder::emitPlacement(
        Graph &graph, int value, int other, std::size_t placement, const Shape &shape
    ) {
      auto operation = placementOperation(placement);
      // A product that would still join the zero-initialized state is a
      // constant zero, which would leave that root's update dead: the placement
      // adds instead, and the state it writes stays live.
      if (operation == TensorOp::Mul && other == static_cast<int>(zeroState))
        operation = TensorOp::Add;
      // A subtractive placement keeps its minuend first: a placement whose
      // operands swap under a non-commutative operation leaves the nonnegative
      // domain every exact certificate is stated over. The commutative
      // placements take their operand order from the engine.
      if (operation != TensorOp::Sub && random_() % 2 == 0)
        return emitBinaryOp(graph, other, value, operation, shape);
      return emitBinaryOp(graph, value, other, operation, shape);
    }

    std::vector<std::uint32_t> Builder::reducibleAxes(const Shape &shape) const {
      std::vector<std::uint32_t> axes;
      for (std::uint32_t axis = 0; axis < shape.size(); ++axis)
        if (shape[axis] > 1)
          axes.push_back(axis);
      return axes;
    }

    std::uint32_t Builder::chooseAxis(const Shape &shape, std::optional<std::uint32_t> excluded) {
      std::vector<std::uint32_t> admitted;
      for (const auto axis: reducibleAxes(shape))
        if (axis != excluded)
          admitted.push_back(axis);
      if (admitted.empty())
        throw AttemptFailure("every axis is a singleton and carries no reduction");
      return admitted[random_() % admitted.size()];
    }

    /// The operation root placement `placement` joins with. Mixed follows the
    /// fold family: one multiplication and one subtraction on the first two
    /// placements, additive afterwards.
    TensorOp Builder::placementOperation(std::size_t placement) const {
      switch (regionArithmetic_) {
        case TensorRegionArithmeticMode::Additive:
          return TensorOp::Add;
        case TensorRegionArithmeticMode::Subtraction:
          return TensorOp::Sub;
        case TensorRegionArithmeticMode::Multiplication:
          return TensorOp::Mul;
        case TensorRegionArithmeticMode::Mixed:
          return placement == 0 ? TensorOp::Mul : placement == 1 ? TensorOp::Sub : TensorOp::Add;
      }
      return TensorOp::Add;
    }

    /// The state each root placement joins. A subtractive placement keeps the
    /// assignment that cancels the other state, the only one whose recurrence
    /// stays inside the nonnegative domain; a multiplicative placement joins
    /// the state that carries the runtime input, so the mode's product is not
    /// annihilated by the zero-initialized state and stays visible in the body;
    /// an additive placement draws its assignment from the engine.
    std::array<std::uint32_t, 2> Builder::placementStates() {
      static constexpr std::array<std::uint32_t, 2> inputFirst{inputState, zeroState};
      static constexpr std::array<std::uint32_t, 2> zeroFirst{zeroState, inputState};
      if (placementOperation(0) == TensorOp::Sub)
        return zeroFirst;
      if (placementOperation(0) == TensorOp::Mul)
        return inputFirst;
      return random_() % 2 != 0 ? inputFirst : zeroFirst;
    }

    Graph Builder::seedRegion(const std::vector<TensorRegionInput> &inputs) const {
      Graph graph;
      for (const auto &input: inputs) {
        GraphValue value;
        value.name = input.name;
        value.shape = input.type.shape;
        value.element = input.type.element;
        value.type = tensorType(value.shape, value.element);
        graph.values.push_back(std::move(value));
      }
      return graph;
    }

    Graph Builder::buildRegionFamily(
        const std::vector<TensorRegionInput> &inputs, const std::vector<std::string> &resultNames,
        const Shape &shape
    ) {
      if (shape.size() < 2)
        throw AttemptFailure("region families need a rank-two state shape");
      if (inputs.size() != 2 || resultNames.size() != 2)
        throw std::invalid_argument(
            "a region family connects exactly two states to two named results"
        );

      Graph graph = seedRegion(inputs);
      const auto root = [&](int value, std::size_t index) {
        graph.values[static_cast<std::size_t>(value)].name = resultNames[index];
        graph.roots.push_back(value);
        return value;
      };
      // The connection skeleton is seeded: which operation produces the shared
      // value, in which operand order, which axis each reduction uses, and which
      // state each root joins all come from the engine, so one configuration
      // reaches many bodies instead of one.
      const auto states = placementStates();
      const auto join = [&](int value, std::size_t placement) {
        return emitPlacement(graph, value, static_cast<int>(states[placement]), placement, shape);
      };

      switch (regionFamily_) {
        case TensorRegionFamily::OrthogonalReductions: {
          const auto axes = reducibleAxes(shape);
          if (axes.size() < 2)
            throw AttemptFailure(
                "independent axis reductions need two extents above one, found a singleton axis"
            );
          requireNodeBudget(7);
          const auto shared = emitProducer(graph, shape);
          const auto firstAxis = chooseAxis(shape);
          const auto firstReduction = emitSum(graph, shared, firstAxis);
          const auto secondReduction = emitSum(graph, shared, chooseAxis(shape, firstAxis));
          const auto firstView = emitBroadcast(graph, firstReduction, shape);
          const auto secondView = emitBroadcast(graph, secondReduction, shape);
          // Each root takes the reduction the other root leaves alone, so both
          // reductions stay live without either root being a copy of the other.
          root(join(secondView, 0), 0);
          root(join(firstView, 1), 1);
          break;
        }
        case TensorRegionFamily::ReductionArithmetic: {
          requireNodeBudget(5);
          const auto shared = emitProducer(graph, shape);
          const auto reduced = emitSum(graph, shared, chooseAxis(shape));
          // The reduction reaches the state only through a bounded broadcast
          // placement, so the update is not a copy of the reduction.
          const auto scaled = join(reduced, 0);
          const auto reducedView = emitBroadcast(graph, reduced, shape);
          root(scaled, 0);
          root(join(reducedView, 1), 1);
          break;
        }
        case TensorRegionFamily::CrossState: {
          requireNodeBudget(3);
          // Both roots read both current states, so neither can be folded into
          // the other's update.
          const auto shared = emitProducer(graph, shape);
          root(join(shared, 0), 0);
          root(join(shared, 1), 1);
          break;
        }
        case TensorRegionFamily::MixedDepth: {
          requireNodeBudget(6);
          const auto shared = emitProducer(graph, shape);
          const auto reduced = emitSum(graph, shared, chooseAxis(shape));
          // The producer's shallow consumer joins the reduction path one
          // broadcast later, so one root sits at a different depth than the
          // other without changing the admitted operation set.
          const auto shallow = join(shared, 0);
          const auto view = emitBroadcast(graph, reduced, shape);
          root(emitPlacement(graph, view, shallow, 1, shape), 0);
          root(join(shared, 1), 1);
          break;
        }
        case TensorRegionFamily::Fold:
          throw std::logic_error("fold regions are built by the fold path");
      }

      prune(graph);
      if (graph.roots.size() != resultNames.size())
        throw std::logic_error("region family lost a named result root");
      return graph;
    }

    Graph Builder::buildRegion(
        const std::vector<TensorRegionInput> &inputs, const std::vector<std::string> &resultNames
    ) {
      const Shape shape = validateRegion(inputs, resultNames);
      if (regionFamily_ != TensorRegionFamily::Fold)
        return buildRegionFamily(inputs, resultNames, shape);
      return buildRegionFolds(inputs, resultNames, shape);
    }

    Shape Builder::validateRegion(
        const std::vector<TensorRegionInput> &inputs, const std::vector<std::string> &resultNames
    ) {
      if (inputs.size() < 2 || inputs.size() > maxInputs_)
        throw std::invalid_argument("multi-output regions need 2..maxInputs state tensors");
      if (resultNames.size() < 2)
        throw std::invalid_argument("multi-output regions need at least two named results");
      if (resultNames.size() >= nodeLimit)
        throw std::runtime_error(
            "graph_exhaustion: result roots leave no budget for a shared graph producer"
        );
      if (maxRank_ == 0 || maxElements_ == 0)
        throw std::runtime_error("graph_exhaustion: limits admit no region tensor shape");

      const Shape &shape = inputs.front().type.shape;
      if (shape.empty() || !admissible(shape))
        throw std::invalid_argument("multi-output region input shape is outside the limits");
      if (std::any_of(inputs.begin(), inputs.end(), [&](const TensorRegionInput &input) {
            return input.name.empty() || input.name.front() != '%' || input.type.shape != shape ||
                   input.type.element != element_;
          }))
        throw std::invalid_argument(
            "multi-output region state inputs must be named tensors with one matching type"
        );

      std::vector<std::string> names;
      names.reserve(inputs.size() + resultNames.size());
      for (const auto &input: inputs)
        names.push_back(input.name);
      for (const auto &result: resultNames) {
        if (result.empty() || result.front() != '%' ||
            std::find(names.begin(), names.end(), result) != names.end())
          throw std::invalid_argument("multi-output region result names must be unique locals");
        names.push_back(result);
      }
      reservedNames_ = std::move(names);

      const auto minimumNodes = static_cast<std::uint32_t>(1 + resultNames.size());
      if (budget_ < minimumNodes)
        throw std::runtime_error(
            "graph_exhaustion: operator budget cannot host the shared producer and every root"
        );

      return shape;
    }

    Graph Builder::buildRegionFolds(
        const std::vector<TensorRegionInput> &inputs, const std::vector<std::string> &resultNames,
        const Shape &shape
    ) {
      const auto minimumNodes = static_cast<std::uint32_t>(1 + resultNames.size());
      Graph graph = seedRegion(inputs);
      int current = emitPair(graph, 0, 1, 0.0, TensorOp::Add);
      const auto availableFolds = std::min<std::uint32_t>(2, (budget_ - minimumNodes) / 3);
      const auto foldCount = availableFolds == 0 ? 0 : 1 + random_() % availableFolds;
      for (std::uint32_t fold = 0; fold < foldCount; ++fold)
        current = emitRegionFold(graph, current, shape);

      for (std::size_t index = 0; index < resultNames.size(); ++index) {
        auto operation = TensorOp::Add;
        if (index == 0 && regionArithmetic_ == TensorRegionArithmeticMode::Subtraction)
          operation = TensorOp::Sub;
        else if (index == 0 && regionArithmetic_ == TensorRegionArithmeticMode::Multiplication)
          operation = TensorOp::Mul;
        else if (regionArithmetic_ == TensorRegionArithmeticMode::Mixed)
          operation = index == 0 ? TensorOp::Mul : (index == 1 ? TensorOp::Sub : TensorOp::Add);

        auto state = static_cast<int>(index % inputs.size());
        if (regionArithmetic_ == TensorRegionArithmeticMode::Mixed && index == 1) {
          // Keep the subtraction observable without requiring a prior state update.
          state = 0;
        }
        const auto result = operation == TensorOp::Add
                                ? emitPair(graph, state, current, 0.0, operation)
                                : emitPair(graph, current, state, 0.0, operation);
        graph.values[static_cast<std::size_t>(result)].name = resultNames[index];
        graph.roots.push_back(result);
      }

      prune(graph);
      return graph;
    }

    std::vector<char> collectAncestors(const Graph &graph, int value) {
      std::vector<char> seen(graph.values.size(), 0);
      std::vector<int> pending;
      const auto producer = graph.values[static_cast<std::size_t>(value)].node;
      if (producer >= 0)
        for (const auto operand: graph.nodes[static_cast<std::size_t>(producer)].operands)
          if (operand >= 0)
            pending.push_back(operand);
      while (!pending.empty()) {
        const auto index = pending.back();
        pending.pop_back();
        if (seen[static_cast<std::size_t>(index)])
          continue;
        seen[static_cast<std::size_t>(index)] = 1;
        const auto node = graph.values[static_cast<std::size_t>(index)].node;
        if (node < 0)
          continue;
        for (const auto operand: graph.nodes[static_cast<std::size_t>(node)].operands)
          if (operand >= 0)
            pending.push_back(operand);
      }
      return seen;
    }

    void prune(Graph &graph) {
      std::vector<char> reachable(graph.nodes.size(), 0);
      std::vector<int> pending;
      pending.reserve(graph.roots.size());
      for (const auto root: graph.roots)
        pending.push_back(graph.values[static_cast<std::size_t>(root)].node);
      while (!pending.empty()) {
        const auto node = pending.back();
        pending.pop_back();
        if (node < 0 || reachable[static_cast<std::size_t>(node)])
          continue;
        reachable[static_cast<std::size_t>(node)] = 1;
        for (const auto operand: graph.nodes[static_cast<std::size_t>(node)].operands)
          if (operand >= 0)
            pending.push_back(graph.values[static_cast<std::size_t>(operand)].node);
      }
      std::vector<GraphNode> live;
      std::vector<int> remapped(graph.nodes.size(), -1);
      for (std::size_t index = 0; index < graph.nodes.size(); ++index)
        if (reachable[index]) {
          remapped[index] = static_cast<int>(live.size());
          live.push_back(std::move(graph.nodes[index]));
        }
      graph.nodes = std::move(live);
      for (auto &value: graph.values) {
        if (value.node < 0)
          continue;
        const auto replacement = remapped[static_cast<std::size_t>(value.node)];
        value.node = replacement >= 0 ? replacement : -2;
      }
    }

    Expr nodeExpression(const Graph &graph, const GraphNode &node) {
      if (node.copy)
        return localExpression(graph.values[static_cast<std::size_t>(node.operands[0])].name);
      TensorAtom atom;
      atom.op = node.op;
      for (std::size_t index = 0; index < node.operands.size(); ++index) {
        if (node.operands[index] < 0)
          atom.operands.push_back(literalCoef(
              graph.values[static_cast<std::size_t>(node.result)].element, node.literals[index]
          ));
        else
          atom.operands.push_back(Coef{
              LocalId{graph.values[static_cast<std::size_t>(node.operands[index])].name, {}}});
      }
      atom.attribute = node.attribute;
      atom.axis = node.axis;
      atom.keepDims = node.keepDims;
      return Expr{Atom{std::move(atom), {}}, {}, {}};
    }

    Program emitProgram(const Graph &graph) {
      Program program;
      FunDecl function;
      function.name = GlobalId{"@kernel", {}};
      for (const auto &value: graph.values) {
        if (value.node != -1)
          continue;
        function.params.push_back(ParamDecl{LocalId{value.name, {}}, value.type, {}});
      }
      const auto root = graph.roots.front();
      function.retType = graph.values[static_cast<std::size_t>(root)].type;
      for (const auto &value: graph.values) {
        if (value.node < 0)
          continue;
        LetDecl local;
        local.isMutable = true;
        local.name = LocalId{value.name, {}};
        local.type = value.type;
        local.init = InitVal{InitVal::Kind::Undef, {}, {}};
        function.lets.push_back(std::move(local));
      }

      Block entry;
      entry.label = BlockLabel{"^entry", {}};
      for (const auto &node: graph.nodes) {
        LValue target;
        target.base = LocalId{graph.values[static_cast<std::size_t>(node.result)].name, {}};
        entry.instrs.push_back(Instr{
            AssignInstr{std::move(target), nodeExpression(graph, node), {}}});
      }
      entry.term = Terminator{
          RetTerm{localExpression(graph.values[static_cast<std::size_t>(root)].name), {}}};
      function.blocks.push_back(std::move(entry));
      program.funs.push_back(std::move(function));
      return program;
    }

    TensorRegion emitRegion(
        const Graph &graph, std::vector<TensorRegionInput> inputs,
        const std::vector<std::string> &resultNames
    ) {
      TensorRegion region(std::move(inputs));
      for (const auto &node: graph.nodes) {
        std::vector<std::string> operands;
        operands.reserve(node.operands.size());
        for (const auto operand: node.operands) {
          if (operand < 0)
            throw std::logic_error("generated loop region contains a literal operand");
          operands.push_back(graph.values[static_cast<std::size_t>(operand)].name);
        }
        region.add(
            graph.values[static_cast<std::size_t>(node.result)].name, node.op, std::move(operands),
            node.axis, node.keepDims, Shape(node.attribute)
        );
      }
      std::vector<std::string> roots;
      roots.reserve(graph.roots.size());
      for (const auto root: graph.roots)
        roots.push_back(graph.values[static_cast<std::size_t>(root)].name);
      if (roots != resultNames)
        throw std::logic_error("generated loop region lost a named result root");
      region.setResults(std::move(roots));
      return region;
    }

  } // namespace

  Program
  generateTensorProgram(const TensorGenerationOptions &options, const TensorGraphLimits &limits) {
    for (std::uint32_t attempt = 0; attempt < constructAttempts; ++attempt) {
      std::mt19937 random(options.seed ^ (0x9e3779b9u * (attempt + 1)));
      try {
        Builder builder(random, options, limits);
        return emitProgram(builder.build());
      } catch (const AttemptFailure &) {
        continue;
      }
    }
    throw std::runtime_error(
        "graph_exhaustion: no admissible compositional tensor graph within " +
        std::to_string(constructAttempts) + " construction attempts"
    );
  }

  TensorRegion generateTensorRegion(
      const TensorGenerationOptions &options, const TensorGraphLimits &limits,
      std::vector<TensorRegionInput> inputs, std::vector<std::string> results,
      TensorRegionArithmeticMode arithmetic, TensorRegionFamily family
  ) {
    switch (arithmetic) {
      case TensorRegionArithmeticMode::Additive:
      case TensorRegionArithmeticMode::Subtraction:
      case TensorRegionArithmeticMode::Multiplication:
      case TensorRegionArithmeticMode::Mixed:
        break;
      default:
        throw std::invalid_argument("unknown tensor region arithmetic mode");
    }
    switch (family) {
      case TensorRegionFamily::Fold:
      case TensorRegionFamily::OrthogonalReductions:
      case TensorRegionFamily::ReductionArithmetic:
      case TensorRegionFamily::CrossState:
      case TensorRegionFamily::MixedDepth:
        break;
      default:
        throw std::invalid_argument("unknown tensor region family");
    }
    for (std::uint32_t attempt = 0; attempt < constructAttempts; ++attempt) {
      std::mt19937 random(options.seed ^ (0x9e3779b9u * (attempt + 1)));
      try {
        Builder builder(random, options, limits, arithmetic, family);
        const auto graph = builder.buildRegion(inputs, results);
        return emitRegion(graph, std::move(inputs), results);
      } catch (const AttemptFailure &) {
        continue;
      }
    }
    throw std::runtime_error(
        "graph_exhaustion: no admissible multi-output tensor region within " +
        std::to_string(constructAttempts) + " construction attempts"
    );
  }

} // namespace refractir::reify
