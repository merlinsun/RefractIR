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
      int root = 0;
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
          const TensorGraphLimits &limits
      )
          : random_(random), profile_(tensorProfile(options.profileVersion)),
            element_(options.dtype == "i32" ? TensorElement::I32 : TensorElement::F32) {
        if (options.profileVersion != TensorProfileId::V2 &&
            options.profileVersion != TensorProfileId::V3)
          throw std::runtime_error(
              "compositional generation supports tensor-v2 and tensor-v3 only"
          );
        if (options.dtype != "i32" && options.dtype != "f32")
          throw std::invalid_argument("compositional tensor elements must be i32 or f32");
        maxRank_ = std::min(limits.maxRank, profile_.maxRank);
        maxElements_ = std::min(limits.maxElements, profile_.maxElements);
        budget_ = std::min(limits.maxOperators, nodeLimit);
        maxInputs_ = limits.maxInputs;
      }

      Graph build();

    private:
      bool admissible(const Shape &shape) const;
      Shape makeWorkingShape();
      Shape makeInputShape(const Shape &working);
      std::uint32_t chooseExtent(bool allowTail);
      std::optional<Shape> applyBinary(const Shape &lhs, const Shape &rhs) const;
      int appendValue(Graph &graph, const Shape &shape);
      int appendNode(Graph &graph, GraphNode node, const Shape &shape);
      int emitPair(Graph &graph, int current, int other, double literal, TensorOp op);
      int emitMerge(Graph &graph, int current, int input);
      int emitBinary(Graph &graph, int current);
      int emitShape(Graph &graph, int current);
      int emitReduce(Graph &graph, int current);
      int emitDiamond(Graph &graph, int current);
      int emitDoubleReduction(Graph &graph, int current);
      int emitCopy(Graph &graph, int current);
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

    Shape Builder::makeWorkingShape() {
      for (std::uint32_t attempt = 0; attempt < 64; ++attempt) {
        const auto rank = 1 + random_() % maxRank_;
        Shape shape(rank);
        for (auto &dimension: shape)
          dimension = chooseExtent(rank == 1);
        const bool reducible =
            std::any_of(shape.begin(), shape.end(), [](std::uint32_t dimension) {
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
      value.name = "%t" + std::to_string(temporaries_++);
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
        const auto merged = applyBinary(
            currentShape, graph.values[static_cast<std::size_t>(other)].shape
        );
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
      const bool reducible =
          std::any_of(shape.begin(), shape.end(), [](std::uint32_t dimension) {
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
      const bool reducible =
          std::any_of(shape.begin(), shape.end(), [](std::uint32_t dimension) {
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
        throw std::runtime_error(
            "graph_exhaustion: compositional limits admit no tensor shape"
        );

      const auto tensorInputs = 2 + random_() % (std::min(maxInputs_, 3u) - 1u);
      const bool scalarInput = maxInputs_ >= 3 && tensorInputs == 2 && random_() % 3 == 0;
      const auto inputCount = tensorInputs + (scalarInput ? 1u : 0u);
      const auto mergeCount = inputCount - 1;
      if (budget_ < mergeCount + 4)
        throw std::runtime_error(
            "graph_exhaustion: operator budget cannot host a shared intermediate"
        );

      const auto required = mergeCount + 4;
      const auto withReductions = required + 5;
      const bool wantReductions = budget_ >= withReductions;
      const auto target = wantReductions
                              ? withReductions + random_() % (budget_ - withReductions + 1)
                              : required + random_() % (budget_ - required + 1);

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

      // The required features are placed at random positions after the input
      // merges; the remaining budget is filled with independently chosen
      // operators, so neither placement nor operand choice repeats per seed.
      std::vector<StepKind> schedule(mergeCount, StepKind::Merge);
      std::vector<StepKind> features{StepKind::Diamond};
      if (wantReductions)
        features.push_back(StepKind::DoubleReduction);
      std::shuffle(features.begin(), features.end(), random_);
      for (const auto feature: features) {
        const auto span = schedule.size() - mergeCount + 1;
        const auto position = mergeCount + random_() % span;
        schedule.insert(schedule.begin() + static_cast<std::ptrdiff_t>(position), feature);
      }
      std::uint32_t used = 0;
      for (const auto step: schedule)
        used += stepCost(step);
      while (used < target) {
        const auto step = pickFiller(target - used);
        schedule.push_back(step);
        used += stepCost(step);
      }

      int current = 0;
      std::size_t merges = 0;
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
      graph.root = current;
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
      std::vector<int> pending{graph.values[static_cast<std::size_t>(graph.root)].node};
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
      for (std::size_t index = 0; index < graph.nodes.size(); ++index)
        if (reachable[index])
          live.push_back(std::move(graph.nodes[index]));
      graph.nodes = std::move(live);
      for (auto &value: graph.values)
        if (value.node >= 0 && !reachable[static_cast<std::size_t>(value.node)])
          value.node = -2;
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
      function.retType = graph.values[static_cast<std::size_t>(graph.root)].type;
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
        entry.instrs.push_back(
            Instr{AssignInstr{std::move(target), nodeExpression(graph, node), {}}}
        );
      }
      entry.term = Terminator{
          RetTerm{localExpression(graph.values[static_cast<std::size_t>(graph.root)].name), {}}};
      function.blocks.push_back(std::move(entry));
      program.funs.push_back(std::move(function));
      return program;
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

} // namespace refractir::reify
