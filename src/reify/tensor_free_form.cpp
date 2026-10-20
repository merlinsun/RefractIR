#include "reify/tensor_free_form.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "analysis/tensor.hpp"

// A free-form region is one typed DAG grown forward from its inputs. The
// builder decides what to place from the traits of each legal action, so a
// policy can only bias construction, never guarantee a motif. The same
// summarize function produces the record the caller keeps, which keeps the
// recorded structure and the construction it describes in one place.
namespace refractir::reify {
  namespace {

    std::uint64_t splitMix64(std::uint64_t value) {
      value += 0x9e3779b97f4a7c15ULL;
      value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
      value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
      return value ^ (value >> 31);
    }

    std::string shapeText(const Shape &shape) {
      std::ostringstream out;
      out << '[';
      for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index)
          out << ", ";
        out << shape[index];
      }
      out << ']';
      return out.str();
    }

    std::string tensorTypeText(const TensorTypeInfo &type) {
      const std::string element = type.element == TensorElement::F32 ? "f32" : "i32";
      if (type.shape.empty())
        return element;
      return "tensor " + shapeText(type.shape) + " " + element;
    }

    bool isBinary(TensorOp op) {
      return op == TensorOp::Add || op == TensorOp::Sub || op == TensorOp::Mul;
    }

    bool isReduction(TensorOp op) { return op == TensorOp::Sum; }

    bool sameType(const TensorTypeInfo &left, const TensorTypeInfo &right) {
      return left.shape == right.shape && left.element == right.element;
    }

    bool tensorShape(const Shape &shape) { return !shape.empty(); }

    bool withinLimits(const Shape &shape, const FreeFormLimits &limits) {
      if (shape.size() > limits.maxRank)
        return false;
      std::uint64_t elements = 1;
      for (const auto extent: shape) {
        if (extent == 0)
          return false;
        elements *= extent;
      }
      return elements <= limits.maxElements;
    }

    /// The result shape of broadcasting `shape` to `target`, or no shape when
    /// the pair does not broadcast. A scalar broadcasts to any tensor.
    std::optional<Shape> broadcastShape(const Shape &left, const Shape &right) {
      try {
        return tensorBroadcastShape(left, right);
      } catch (const std::exception &) {
        return std::nullopt;
      }
    }

    bool broadcastsTo(const Shape &source, const Shape &target) {
      if (target.size() < source.size())
        return false;
      for (std::size_t offset = 0; offset < source.size(); ++offset) {
        const auto from = source[source.size() - 1 - offset];
        const auto to = target[target.size() - 1 - offset];
        if (from != 1 && from != to)
          return false;
      }
      return true;
    }

    Shape reduceShape(const Shape &shape, std::int64_t axis, bool keepDims) {
      Shape result = shape;
      if (keepDims)
        result[static_cast<std::size_t>(axis)] = 1;
      else
        result.erase(result.begin() + axis);
      return result;
    }

    std::string actionKey(
        TensorOp op, std::int32_t lhs, std::int32_t rhs, std::int64_t axis, bool keepDims,
        const Shape &attribute
    ) {
      std::ostringstream key;
      key << static_cast<int>(op) << ':' << lhs << ':' << rhs << ':' << axis << ':' << keepDims
          << ':';
      for (const auto extent: attribute)
        key << extent << ',';
      return key.str();
    }

    struct PoolValue {
      std::string name;
      TensorTypeInfo type;
      std::uint32_t depth = 0;
      std::uint64_t mask = 0;
      std::uint32_t consumers = 0;
      std::int32_t node = -1;
      TensorOp op = TensorOp::Add;
    };

    struct NodeRecord {
      TensorOp op = TensorOp::Add;
      std::vector<std::int32_t> operands;
      std::int64_t axis = 0;
      bool keepDims = false;
      Shape attribute;
    };

    struct Action {
      TensorOp op = TensorOp::Add;
      std::int32_t lhs = 0;
      std::int32_t rhs = -1;
      std::int64_t axis = 0;
      bool keepDims = false;
      Shape attribute;
      TensorTypeInfo type;
      FreeFormActionTraits traits;
    };

    std::uint32_t distinctInputs(std::uint64_t mask) {
      std::uint32_t count = 0;
      while (mask != 0) {
        mask &= mask - 1;
        ++count;
      }
      return count;
    }

    /// One growth attempt: a value pool, the placed nodes, and the temporary
    /// name counter. Construction stops when the pool admits no legal action.
    class Growth {
    public:
      Growth(const FreeFormRequest &request, std::uint32_t growthNodes) :
          limits_(request.limits), element_(request.inputs.front().value.type.element),
          growthNodes_(growthNodes), random_(splitMix64(request.seed ^ 0x667265652d666f72ULL)) {
        for (std::size_t index = 0; index < request.inputs.size(); ++index) {
          PoolValue value;
          value.name = request.inputs[index].value.name;
          value.type = request.inputs[index].value.type;
          value.depth = 0;
          value.mask = 1ULL << index;
          value.node = -1;
          pool_.push_back(std::move(value));
        }
      }

      /// Place up to `growthNodes_` actions. Returns whether any action was
      /// placed; an attempt that places none still has its inputs to root.
      bool grow() {
        bool placed = false;
        while (nodes_.size() < growthNodes_) {
          auto actions = enumerate();
          if (actions.empty())
            break;
          std::vector<double> weights;
          weights.reserve(actions.size());
          for (const auto &action: actions)
            weights.push_back(freeFormActionWeight(policy_, action.traits));
          const auto draw =
              (static_cast<double>(random_()) + 0.5) / 4294967296.0 * totalWeight(weights);
          apply(actions[freeFormSelectWeighted(weights, draw)]);
          placed = true;
        }
        return placed;
      }

      void setPolicy(FreeFormPolicy policy) { policy_ = policy; }

      [[nodiscard]] const std::vector<PoolValue> &pool() const { return pool_; }
      [[nodiscard]] const std::vector<NodeRecord> &nodes() const { return nodes_; }
      [[nodiscard]] std::uint32_t nodesPlaced() const {
        return static_cast<std::uint32_t>(nodes_.size());
      }
      [[nodiscard]] std::uint32_t multiplications() const { return multiplications_; }

      /// The typed pool entries, in creation order, with inputs first.
      [[nodiscard]] std::vector<TensorTypeInfo> types() const {
        std::vector<TensorTypeInfo> types;
        types.reserve(pool_.size());
        for (const auto &value: pool_)
          types.push_back(value.type);
        return types;
      }
      [[nodiscard]] std::vector<std::string> names() const {
        std::vector<std::string> names;
        names.reserve(pool_.size());
        for (const auto &value: pool_)
          names.push_back(value.name);
        return names;
      }

      using Candidates = std::vector<std::uint32_t>;
      [[nodiscard]] Candidates valuesOfType(const TensorTypeInfo &type) const {
        Candidates candidates;
        for (std::size_t index = 0; index < pool_.size(); ++index)
          if (sameType(pool_[index].type, type))
            candidates.push_back(static_cast<std::uint32_t>(index));
        return candidates;
      }

      /// Append one action that produces `type`, or return no index when no
      /// single legal action does. The node budget is the caller's to check.
      std::optional<std::uint32_t> placeForType(const TensorTypeInfo &type) {
        auto actions = enumerate();
        const auto found = std::find_if(actions.begin(), actions.end(), [&](const Action &action) {
          return sameType(action.type, type);
        });
        if (found == actions.end())
          return std::nullopt;
        apply(*found);
        return static_cast<std::uint32_t>(pool_.size() - 1);
      }

    private:
      static double totalWeight(const std::vector<double> &weights) {
        double total = 0;
        for (const auto weight: weights)
          total += weight;
        return total;
      }

      std::string nextName() {
        std::string name;
        do {
          name = "%ff_" + std::to_string(temporaries_++);
        } while (std::any_of(pool_.begin(), pool_.end(), [&](const PoolValue &value) {
          return value.name == name;
        }));
        return name;
      }

      std::vector<Action> enumerate() const {
        std::vector<Action> actions;
        std::unordered_set<std::string> seen;
        const auto push = [&](Action action) {
          auto key = actionKey(
              action.op, action.lhs, action.rhs, action.axis, action.keepDims, action.attribute
          );
          if (seen.insert(std::move(key)).second)
            actions.push_back(std::move(action));
        };
        const auto traitsOf = [&](TensorOp op, std::int32_t lhs, std::int32_t rhs,
                                  const TensorTypeInfo &type) {
          FreeFormActionTraits traits;
          traits.op = op;
          const auto &left = pool_[static_cast<std::size_t>(lhs)];
          traits.depth = left.depth;
          traits.reusedConsumers = left.consumers;
          traits.unusedOperands = left.consumers == 0 ? 1U : 0U;
          std::uint64_t mask = left.mask;
          if (rhs >= 0) {
            const auto &right = pool_[static_cast<std::size_t>(rhs)];
            traits.depth = std::max(traits.depth, right.depth);
            traits.reusedConsumers = std::max(traits.reusedConsumers, right.consumers);
            traits.unusedOperands += right.consumers == 0 ? 1U : 0U;
            mask |= right.mask;
          }
          traits.inputs = distinctInputs(mask);
          // A sum or a broadcast is the only action that changes its operand's
          // shape; a binary action's result is the broadcast of both operands.
          traits.shapeChange =
              (isReduction(op) || op == TensorOp::Broadcast) && type.shape != left.type.shape;
          return traits;
        };

        const auto element = element_;
        for (std::int32_t lhs = 0; lhs < static_cast<std::int32_t>(pool_.size()); ++lhs) {
          const auto &left = pool_[static_cast<std::size_t>(lhs)];
          for (std::int32_t rhs = lhs; rhs < static_cast<std::int32_t>(pool_.size()); ++rhs) {
            const auto &right = pool_[static_cast<std::size_t>(rhs)];
            if (right.type.element != element || left.type.element != element)
              continue;
            const auto shape = broadcastShape(left.type.shape, right.type.shape);
            if (!shape || !withinLimits(*shape, limits_))
              continue;
            TensorTypeInfo type{*shape, element};
            Action add{};
            add.op = TensorOp::Add;
            add.lhs = lhs;
            add.rhs = rhs;
            add.type = type;
            add.traits = traitsOf(TensorOp::Add, lhs, rhs, type);
            push(add);
            if (lhs != rhs) {
              Action sub{};
              sub.op = TensorOp::Sub;
              sub.lhs = lhs;
              sub.rhs = rhs;
              sub.type = type;
              sub.traits = traitsOf(TensorOp::Sub, lhs, rhs, type);
              push(sub);
              Action reversed{};
              reversed.op = TensorOp::Sub;
              reversed.lhs = rhs;
              reversed.rhs = lhs;
              reversed.type = type;
              reversed.traits = traitsOf(TensorOp::Sub, rhs, lhs, type);
              push(reversed);
            }
            // Multiplication is admitted for f32 only: repeated products of
            // integer leaves would leave the i32 domain before a case is run.
            if (element == TensorElement::F32 && multiplications_ < limits_.maxMultiplications) {
              Action mul{};
              mul.op = TensorOp::Mul;
              mul.lhs = lhs;
              mul.rhs = rhs;
              mul.type = type;
              mul.traits = traitsOf(TensorOp::Mul, lhs, rhs, type);
              push(mul);
            }
          }
        }

        for (std::int32_t index = 0; index < static_cast<std::int32_t>(pool_.size()); ++index) {
          const auto &value = pool_[static_cast<std::size_t>(index)];
          const auto rank = static_cast<std::int64_t>(value.type.shape.size());
          for (std::int64_t axis = 0; axis < rank; ++axis) {
            if (value.type.shape[static_cast<std::size_t>(axis)] <= 1)
              continue;
            for (const bool keep: {false, true}) {
              const auto shape = reduceShape(value.type.shape, axis, keep);
              if (!withinLimits(shape, limits_))
                continue;
              Action sum{};
              sum.op = TensorOp::Sum;
              sum.lhs = index;
              sum.axis = axis;
              sum.keepDims = keep;
              sum.type = TensorTypeInfo{shape, element};
              sum.traits = traitsOf(TensorOp::Sum, index, -1, sum.type);
              push(sum);
            }
          }
        }

        for (std::int32_t index = 0; index < static_cast<std::int32_t>(pool_.size()); ++index) {
          const auto &value = pool_[static_cast<std::size_t>(index)];
          std::vector<Shape> targets;
          for (const auto &other: pool_)
            if (other.type.shape != value.type.shape)
              targets.push_back(other.type.shape);
          for (const auto &candidate: targets) {
            if (!broadcastsTo(value.type.shape, candidate) ||
                !withinLimits(candidate, limits_) || candidate == value.type.shape)
              continue;
            Action broadcast{};
            broadcast.op = TensorOp::Broadcast;
            broadcast.lhs = index;
            broadcast.attribute = candidate;
            broadcast.type = TensorTypeInfo{candidate, element};
            broadcast.traits = traitsOf(TensorOp::Broadcast, index, -1, broadcast.type);
            push(broadcast);
          }
        }
        return actions;
      }

      void apply(const Action &action) {
        std::vector<std::int32_t> operands{action.lhs};
        if (action.rhs >= 0)
          operands.push_back(action.rhs);
        PoolValue value;
        value.name = nextName();
        value.type = action.type;
        value.depth = action.traits.depth + 1;
        value.mask = 0;
        for (const auto operand: operands)
          value.mask |= pool_[static_cast<std::size_t>(operand)].mask;
        value.node = static_cast<std::int32_t>(nodes_.size());
        value.op = action.op;
        pool_.push_back(std::move(value));
        for (const auto operand: operands)
          ++pool_[static_cast<std::size_t>(operand)].consumers;
        if (action.op == TensorOp::Mul)
          ++multiplications_;
        nodes_.push_back(
            NodeRecord{action.op, std::move(operands), action.axis, action.keepDims,
                       action.attribute}
        );
      }

      FreeFormLimits limits_;
      TensorElement element_ = TensorElement::I32;
      std::uint32_t growthNodes_ = 0;
      FreeFormPolicy policy_ = FreeFormPolicy::Unguided;
      std::uint32_t temporaries_ = 0;
      std::uint32_t multiplications_ = 0;
      std::mt19937 random_;
      std::vector<PoolValue> pool_;
      std::vector<NodeRecord> nodes_;
    };

    /// The value indices a root set depends on, in live closure.
    std::vector<char> liveValues(
        std::uint32_t valueCount, const std::vector<std::vector<std::int32_t>> &operands,
        std::uint32_t inputCount, const std::vector<std::uint32_t> &roots
    ) {
      std::vector<char> live(valueCount, 0);
      std::vector<std::uint32_t> stack(roots.begin(), roots.end());
      while (!stack.empty()) {
        const auto value = stack.back();
        stack.pop_back();
        if (value < valueCount && live[value])
          continue;
        if (value >= valueCount)
          throw std::invalid_argument("free-form graph names a value outside its pool");
        live[value] = 1;
        if (value < inputCount)
          continue;
        for (const auto operand: operands[value - inputCount])
          if (operand >= 0)
            stack.push_back(static_cast<std::uint32_t>(operand));
      }
      return live;
    }

    std::string traitTarget(FreeFormPolicy policy) {
      switch (policy) {
        case FreeFormPolicy::Deep:
          return "depth";
        case FreeFormPolicy::Wide:
          return "width";
        case FreeFormPolicy::Shared:
          return "sharing";
        case FreeFormPolicy::ReductionHeavy:
          return "reductions";
        case FreeFormPolicy::CrossState:
          return "cross-input";
        case FreeFormPolicy::Balanced:
          return "balanced";
        case FreeFormPolicy::Unguided:
          return "none";
      }
      return "none";
    }

  } // namespace

  std::optional<FreeFormPolicy> parseFreeFormPolicy(std::string_view text) {
    if (text == "unguided")
      return FreeFormPolicy::Unguided;
    if (text == "balanced")
      return FreeFormPolicy::Balanced;
    if (text == "deep")
      return FreeFormPolicy::Deep;
    if (text == "wide")
      return FreeFormPolicy::Wide;
    if (text == "shared")
      return FreeFormPolicy::Shared;
    if (text == "reduction-heavy")
      return FreeFormPolicy::ReductionHeavy;
    if (text == "cross-state")
      return FreeFormPolicy::CrossState;
    return std::nullopt;
  }

  std::string_view freeFormPolicyName(FreeFormPolicy policy) noexcept {
    switch (policy) {
      case FreeFormPolicy::Unguided:
        return "unguided";
      case FreeFormPolicy::Balanced:
        return "balanced";
      case FreeFormPolicy::Deep:
        return "deep";
      case FreeFormPolicy::Wide:
        return "wide";
      case FreeFormPolicy::Shared:
        return "shared";
      case FreeFormPolicy::ReductionHeavy:
        return "reduction-heavy";
      case FreeFormPolicy::CrossState:
        return "cross-state";
    }
    return "unknown";
  }

  double freeFormActionWeight(FreeFormPolicy policy, const FreeFormActionTraits &traits) {
    const double depth = static_cast<double>(traits.depth);
    const double reused = static_cast<double>(traits.reusedConsumers);
    const double unused = static_cast<double>(traits.unusedOperands);
    const double inputs = static_cast<double>(traits.inputs);
    const double shapeChange = traits.shapeChange ? 1.0 : 0.0;
    switch (policy) {
      case FreeFormPolicy::Unguided:
        return 1.0;
      case FreeFormPolicy::Balanced:
        // A mild mix: a fresh operand and a shape change carry the score, and
        // repeated reuse of one value decays towards the flat weight. The
        // reciprocal keeps every weight positive, so no action is excluded.
        return 1.0 + unused + shapeChange + 1.0 / (1.0 + reused);
      case FreeFormPolicy::Deep:
        return 1.0 + 3.0 * depth;
      case FreeFormPolicy::Wide:
        return 1.0 + 2.0 * unused;
      case FreeFormPolicy::Shared:
        return 1.0 + 3.0 * reused + 2.0 * (reused >= 2.0 ? 1.0 : 0.0);
      case FreeFormPolicy::ReductionHeavy:
        return 1.0 + 4.0 * (isReduction(traits.op) ? 1.0 : 0.0);
      case FreeFormPolicy::CrossState:
        return 1.0 + 4.0 * (inputs >= 2.0 ? 1.0 : 0.0) + 2.0 * (inputs >= 3.0 ? 1.0 : 0.0);
    }
    return 1.0;
  }

  std::uint32_t freeFormSelectWeighted(const std::vector<double> &weights, double draw) {
    if (weights.empty())
      throw std::invalid_argument("free-form selection needs at least one weight");
    double total = 0;
    for (const auto weight: weights) {
      if (!(weight > 0))
        throw std::invalid_argument("free-form selection weights must be positive");
      total += weight;
    }
    if (!(draw >= 0 && draw < total))
      throw std::invalid_argument("free-form selection draw lies outside the weight total");
    double accumulated = 0;
    for (std::size_t index = 0; index < weights.size(); ++index) {
      accumulated += weights[index];
      if (draw < accumulated)
        return static_cast<std::uint32_t>(index);
    }
    // Rounding can leave the draw a hair below the stated total.
    return static_cast<std::uint32_t>(weights.size() - 1);
  }

  FreeFormConsumers countFreeFormConsumers(
      std::uint32_t valueCount, const std::vector<std::vector<std::int32_t>> &consumerOperands
  ) {
    FreeFormConsumers consumers;
    consumers.perValue.assign(valueCount, 0);
    for (const auto &operands: consumerOperands) {
      std::unordered_set<std::int32_t> distinct;
      for (const auto operand: operands) {
        if (operand < 0)
          continue;
        if (operand >= static_cast<std::int32_t>(valueCount))
          throw std::invalid_argument("free-form consumer names a value outside its pool");
        distinct.insert(operand);
      }
      for (const auto operand: distinct)
        ++consumers.perValue[static_cast<std::size_t>(operand)];
    }
    for (std::size_t index = 0; index < consumers.perValue.size(); ++index) {
      consumers.total += consumers.perValue[index];
      if (consumers.perValue[index] >= 2)
        ++consumers.shared;
    }
    return consumers;
  }

  FreeFormStructure summarizeFreeFormGraph(const FreeFormGraphView &graph, FreeFormPolicy policy) {
    const auto valueCount = static_cast<std::uint32_t>(graph.valueTypes.size());
    if (valueCount < graph.inputCount)
      throw std::invalid_argument("free-form graph has more inputs than values");
    if (graph.nodeOps.size() != graph.nodeOperands.size())
      throw std::invalid_argument("free-form graph names a different operand list per node");
    if (valueCount != graph.inputCount + graph.nodeOps.size())
      throw std::invalid_argument("free-form graph value count does not match its nodes");
    if (graph.valueNames.size() != valueCount)
      throw std::invalid_argument("free-form graph names a different value count");
    if (graph.roots.empty())
      throw std::invalid_argument("free-form graph has no roots");
    for (const auto root: graph.roots)
      if (root >= valueCount)
        throw std::invalid_argument("free-form graph root is outside its pool");
    for (const auto &operands: graph.nodeOperands)
      for (const auto operand: operands)
        if (operand < 0 || operand >= static_cast<std::int32_t>(valueCount))
          throw std::invalid_argument("free-form graph operand is outside its pool");

    const auto live = liveValues(valueCount, graph.nodeOperands, graph.inputCount, graph.roots);

    std::vector<std::uint32_t> depth(valueCount, 0);
    std::vector<std::uint64_t> mask(valueCount, 0);
    for (std::uint32_t index = 0; index < graph.inputCount; ++index)
      mask[index] = live[index] ? (1ULL << index) : 0;
    std::uint32_t liveNodes = 0;
    std::uint32_t deepest = 0;
    std::uint32_t reductions = 0;
    std::uint32_t shapeChanges = 0;
    std::map<std::string, std::uint32_t> operatorCounts;
    for (std::uint32_t node = 0; node < graph.nodeOps.size(); ++node) {
      const auto value = graph.inputCount + node;
      if (live[value]) {
        ++liveNodes;
        std::uint32_t nodeDepth = 0;
        for (const auto operand: graph.nodeOperands[node]) {
          nodeDepth = std::max(nodeDepth, depth[static_cast<std::size_t>(operand)]);
          mask[value] |= mask[static_cast<std::size_t>(operand)];
        }
        depth[value] = nodeDepth + 1;
        deepest = std::max(deepest, depth[value]);
        const auto op = graph.nodeOps[node];
        if (isReduction(op))
          ++reductions;
        if ((isReduction(op) || op == TensorOp::Broadcast) &&
            !graph.nodeOperands[node].empty() &&
            graph.valueTypes[value].shape !=
                graph.valueTypes[static_cast<std::size_t>(graph.nodeOperands[node].front())].shape)
          ++shapeChanges;
        ++operatorCounts[std::string(tensorOpName(op))];
      }
    }

    // Only live consumers count: a pruned node cannot make its operand shared.
    std::vector<std::vector<std::int32_t>> liveOperands;
    for (std::uint32_t node = 0; node < graph.nodeOps.size(); ++node)
      if (live[graph.inputCount + node])
        liveOperands.push_back(graph.nodeOperands[node]);
    const auto consumers = countFreeFormConsumers(valueCount, liveOperands);

    std::uint32_t maxDefUseDistance = 0;
    for (std::uint32_t node = 0; node < graph.nodeOps.size(); ++node) {
      if (!live[graph.inputCount + node])
        continue;
      const auto consumer = graph.inputCount + node;
      for (const auto operand: graph.nodeOperands[node])
        maxDefUseDistance = std::max(maxDefUseDistance, consumer - static_cast<std::uint32_t>(operand));
    }

    FreeFormStructure structure;
    structure.policy = std::string(freeFormPolicyName(policy));
    structure.target = traitTarget(policy);
    structure.nodes = liveNodes;
    structure.depth = deepest;
    structure.reductions = reductions;
    structure.shapeChanges = shapeChanges;
    structure.distinctConsumers = consumers.total;
    structure.sharedValues = consumers.shared;
    structure.maxDefUseDistance = maxDefUseDistance;
    structure.operatorCounts.assign(operatorCounts.begin(), operatorCounts.end());

    std::uint32_t liveValueCount = 0;
    for (std::uint32_t index = 0; index < valueCount; ++index) {
      if (!live[index])
        continue;
      ++liveValueCount;
      if (consumers.perValue[index] > 0)
        structure.consumers.emplace_back(
            graph.valueNames[index], consumers.perValue[index]
        );
    }
    std::sort(
        structure.consumers.begin(), structure.consumers.end(),
        [](const auto &left, const auto &right) { return left.first < right.first; }
    );

    for (const auto root: graph.roots)
      structure.resultDependencies.push_back(mask[root]);

    const bool reductionHeavy = reductions >= 2 && reductions * 2 >= liveNodes && liveNodes > 0;
    const bool shared = consumers.shared >= 2 && consumers.shared * 2 >= liveValueCount;
    const bool deep = deepest >= 4;
    const bool crossInput = std::any_of(
        structure.resultDependencies.begin(), structure.resultDependencies.end(),
        [](std::uint64_t value) { return distinctInputs(value) >= 2; }
    );
    const bool wide = liveValueCount > 0 && consumers.total * 2 >= liveValueCount;
    if (reductionHeavy)
      structure.actual = "reduction-heavy";
    else if (shared)
      structure.actual = "shared";
    else if (deep)
      structure.actual = "deep";
    else if (crossInput)
      structure.actual = "cross-state";
    else if (wide)
      structure.actual = "wide";
    else
      structure.actual = "unguided";
    return structure;
  }

  namespace {

    void validateRequest(const FreeFormRequest &request) {
      if (request.inputs.empty())
        throw std::invalid_argument("free-form request needs at least one input");
      if (request.results.empty())
        throw std::invalid_argument("free-form request needs at least one result");
      if (request.limits.maxNodes == 0)
        throw std::invalid_argument("free-form limits admit no node");
      if (request.limits.maxAttempts == 0)
        throw std::invalid_argument("free-form limits admit no attempt");
      if (request.limits.maxRank == 0)
        throw std::invalid_argument("free-form limits admit no rank");
      if (request.limits.maxElements == 0)
        throw std::invalid_argument("free-form limits admit no elements");
      if (request.inputs.size() > 64)
        throw std::invalid_argument("free-form request admits at most 64 inputs");
      const auto element = request.inputs.front().value.type.element;
      std::unordered_set<std::string> names;
      for (const auto &input: request.inputs) {
        if (input.value.name.empty() || input.value.name.front() != '%')
          throw std::invalid_argument("free-form inputs must be local identifiers");
        if (!tensorShape(input.value.type.shape))
          throw std::invalid_argument("free-form inputs must have a tensor shape");
        if (input.value.type.element != element)
          throw std::invalid_argument("free-form values must share one element type");
        if (!withinLimits(input.value.type.shape, request.limits))
          throw std::invalid_argument("free-form input lies outside the stated limits");
        if (!names.insert(input.value.name).second)
          throw std::invalid_argument("free-form input is declared more than once");
        if (input.init == FreeFormInit::Zero && !input.parameter.empty())
          throw std::invalid_argument("a zero-initialized input takes no parameter");
      }
      for (const auto &result: request.results) {
        if (result.type.element != element)
          throw std::invalid_argument("free-form values must share one element type");
        if (!tensorShape(result.type.shape))
          throw std::invalid_argument("free-form results must have a tensor shape");
        if (!withinLimits(result.type.shape, request.limits))
          throw std::invalid_argument("free-form result lies outside the stated limits");
        if (!result.name.empty() && !names.insert(result.name).second)
          throw std::invalid_argument("free-form result name repeats another name");
      }
    }

    std::string resolveResultName(const FreeFormRequest &request, std::size_t index) {
      if (!request.results[index].name.empty())
        return request.results[index].name;
      return "%next_" + std::to_string(index);
    }

  } // namespace

  FreeFormRegion buildFreeFormRegion(const FreeFormRequest &request) {
    validateRequest(request);
    const auto resultCount = static_cast<std::uint32_t>(request.results.size());
    if (request.limits.maxNodes < resultCount)
      throw std::runtime_error(
          "free_form_exhaustion: the node budget is smaller than the requested result count"
      );
    // One node per result is held back: a requested result type that the grown
    // pool does not reach can still be placed from an existing value.
    const auto growthNodes = request.limits.maxNodes - resultCount;
    std::string lastReason = "no attempt placed every requested result";
    for (std::uint32_t attempt = 0; attempt < request.limits.maxAttempts; ++attempt) {
      FreeFormRequest seed = request;
      seed.seed = static_cast<std::uint32_t>(splitMix64(
          static_cast<std::uint64_t>(request.seed) ^ (0x9e3779b97f4a7c15ULL * (attempt + 1))
      ));
      Growth growth(seed, growthNodes);
      growth.setPolicy(request.policy);
      growth.grow();

      std::vector<std::uint32_t> roots;
      bool complete = true;
      for (std::uint32_t index = 0; index < resultCount; ++index) {
        const auto type = request.results[index].type;
        std::optional<std::uint32_t> best;
        std::uint32_t bestScore = 0;
        for (const auto candidate: growth.valuesOfType(type)) {
          if (std::find(roots.begin(), roots.end(), candidate) != roots.end())
            continue;
          const auto &value = growth.pool()[candidate];
          const std::uint32_t score =
              (value.node >= 0 ? 1U << 30 : 0U) + (value.depth << 16) + candidate;
          if (!best || score > bestScore) {
            best = candidate;
            bestScore = score;
          }
        }
        if (!best) {
          if (growth.nodesPlaced() >= request.limits.maxNodes) {
            lastReason = "the node budget was spent before result " + std::to_string(index) +
                         " could be placed";
            complete = false;
            break;
          }
          const auto placed = growth.placeForType(type);
          if (!placed) {
            lastReason = "no legal action produces result " + std::to_string(index);
            complete = false;
            break;
          }
          best = *placed;
        }
        if (std::find(roots.begin(), roots.end(), *best) != roots.end()) {
          lastReason = "two requested results resolved to one value";
          complete = false;
          break;
        }
        roots.push_back(*best);
      }
      if (!complete)
        continue;

      const auto &pool = growth.pool();
      const auto &nodes = growth.nodes();
      // A root keeps the name the caller asked for when the builder computed
      // it. An input root keeps its input name: the region cannot rename an
      // input, and an update from a state to itself is a legal no-op.
      std::vector<std::string> resultNames;
      resultNames.reserve(roots.size());
      std::vector<std::string> emittedNames;
      emittedNames.reserve(pool.size());
      for (const auto &value: pool)
        emittedNames.push_back(value.name);
      for (std::size_t index = 0; index < roots.size(); ++index) {
        if (pool[roots[index]].node < 0) {
          resultNames.push_back(pool[roots[index]].name);
          continue;
        }
        const auto name = resolveResultName(request, index);
        emittedNames[roots[index]] = name;
        resultNames.push_back(name);
      }
      std::unordered_set<std::string> uniqueNames;
      for (const auto &name: emittedNames)
        if (!uniqueNames.insert(name).second) {
          lastReason = "two emitted values would share the name " + name;
          complete = false;
          break;
        }
      if (!complete)
        continue;

      std::vector<TensorRegionInput> regionInputs;
      regionInputs.reserve(request.inputs.size());
      for (const auto &input: request.inputs)
        regionInputs.push_back(input.value);
      TensorRegion region(std::move(regionInputs));
      for (std::size_t node = 0; node < nodes.size(); ++node) {
        const auto &record = nodes[node];
        std::vector<std::string> operands;
        for (const auto operand: record.operands)
          operands.push_back(emittedNames[static_cast<std::size_t>(operand)]);
        region.add(
            emittedNames[request.inputs.size() + node], record.op, std::move(operands),
            record.axis, record.keepDims, record.attribute
        );
      }
      region.setResults(resultNames);

      FreeFormGraphView view;
      view.valueNames = growth.names();
      view.valueTypes = growth.types();
      view.inputCount = static_cast<std::uint32_t>(request.inputs.size());
      for (const auto &record: nodes)
        view.nodeOps.push_back(record.op);
      for (const auto &record: nodes)
        view.nodeOperands.push_back(record.operands);
      view.roots = roots;

      FreeFormRegion built{
          std::move(region), summarizeFreeFormGraph(view, request.policy), {}
      };
      built.inputSpec.inputs = request.inputs;
      built.inputSpec.results = std::move(resultNames);
      return built;
    }
    throw std::runtime_error("free_form_exhaustion: " + lastReason);
  }

  namespace {

    bool isFinalOp(TensorOp op) { return isBinary(op); }

    void requireLocalName(const std::string &name, const std::string &what) {
      if (name.empty() || name.front() != '%')
        throw std::invalid_argument(what + " must be a local identifier");
    }

    std::string parameterName(const FreeFormInput &input, std::size_t index) {
      if (!input.parameter.empty())
        return input.parameter;
      return "%in_" + std::to_string(index);
    }

    std::string zeroInitializer(const Shape &shape, TensorElement element) {
      const std::string zero = element == TensorElement::F32 ? "0.0" : "0";
      std::ostringstream out;
      out << '{';
      std::uint64_t elements = 1;
      for (const auto extent: shape)
        elements *= extent;
      for (std::uint64_t index = 0; index < elements; ++index) {
        if (index)
          out << ", ";
        out << zero;
      }
      out << '}';
      return out.str();
    }

  } // namespace

  std::string assembleCountedLoop(const TensorRegion &region, const FreeFormLoop &loop) {
    if (loop.entry.empty())
      throw std::invalid_argument("counted loop needs an entry name");
    if (loop.tripCount == 0 || loop.tripCount > 64)
      throw std::invalid_argument("counted loop trip count must be in [1, 64]");
    if (loop.inputs.empty())
      throw std::invalid_argument("counted loop needs at least one input");
    if (loop.results.empty())
      throw std::invalid_argument("counted loop needs at least one result");
    if (loop.updates.size() != loop.results.size())
      throw std::invalid_argument("counted loop needs one update per result");
    if (loop.finalTerms.empty())
      throw std::invalid_argument("counted loop needs a returned expression");
    if (!isFinalOp(loop.finalOp))
      throw std::invalid_argument("counted loop final operation must be add, sub, or mul");

    std::vector<std::string> stateNames;
    std::vector<std::string> invariantNames;
    std::unordered_set<std::string> localNames;
    for (const auto &input: loop.inputs) {
      requireLocalName(input.value.name, "counted loop input");
      if (input.value.name == "%i" || input.value.name == "%out")
        throw std::invalid_argument("counted loop input reuses a loop local");
      if (!localNames.insert(input.value.name).second)
        throw std::invalid_argument("counted loop input is declared more than once");
      if (input.role == FreeFormRole::State)
        stateNames.push_back(input.value.name);
      else
        invariantNames.push_back(input.value.name);
    }

    std::unordered_set<std::string> parameters;
    for (std::size_t index = 0; index < loop.inputs.size(); ++index) {
      const auto &input = loop.inputs[index];
      if (input.init != FreeFormInit::RuntimeInput)
        continue;
      const auto name = parameterName(input, index);
      requireLocalName(name, "counted loop parameter");
      if (name == "%i" || name == "%out")
        throw std::invalid_argument("counted loop parameter reuses a loop local");
      if (localNames.contains(name))
        throw std::invalid_argument("counted loop parameter repeats a loop local");
      if (!parameters.insert(name).second)
        throw std::invalid_argument("counted loop parameter is declared more than once");
    }

    if (loop.updates.size() != stateNames.size())
      throw std::invalid_argument("counted loop needs one update per state");
    std::unordered_set<std::string> updated;
    for (const auto &[state, result]: loop.updates) {
      if (std::find(invariantNames.begin(), invariantNames.end(), state) != invariantNames.end())
        throw std::invalid_argument("counted loop cannot update an invariant: " + state);
      if (!localNames.contains(state))
        throw std::invalid_argument("counted loop updates an unknown state: " + state);
      if (!updated.insert(state).second)
        throw std::invalid_argument("counted loop updates a state twice: " + state);
      if (std::find(loop.results.begin(), loop.results.end(), result) == loop.results.end())
        throw std::invalid_argument("counted loop update source is not a declared result: " + result);
    }
    for (const auto &state: stateNames)
      if (!updated.contains(state))
        throw std::invalid_argument("counted loop leaves a state unupdated: " + state);

    std::unordered_set<std::string> terminators(localNames.begin(), localNames.end());
    std::vector<TensorTypeInfo> termTypes;
    for (const auto &term: loop.finalTerms) {
      if (!terminators.contains(term))
        throw std::invalid_argument("counted loop returns a value that is not a local: " + term);
      termTypes.push_back(region.typeOf(term));
    }
    TensorTypeInfo returnType = termTypes.front();
    for (std::size_t index = 1; index < termTypes.size(); ++index) {
      if (termTypes[index].element != returnType.element)
        throw std::invalid_argument("counted loop returned terms have different element types");
      returnType.shape = tensorBroadcastShape(returnType.shape, termTypes[index].shape);
    }

    const auto updates = loop.updates;
    const auto reserved = loop.reservedNames;
    const auto declarations = region.renderDeclarations(updates, reserved);
    const auto body = region.renderBody(updates, reserved);

    std::ostringstream source;
    source << "fun @" << loop.entry << '(';
    bool first = true;
    for (std::size_t index = 0; index < loop.inputs.size(); ++index) {
      const auto &input = loop.inputs[index];
      if (input.init != FreeFormInit::RuntimeInput)
        continue;
      if (!first)
        source << ", ";
      first = false;
      source << parameterName(input, index) << ": " << tensorTypeText(input.value.type);
    }
    source << ") : " << tensorTypeText(returnType) << " {\n"
           << "  let mut %i: i32 = undef;\n";
    for (const auto &input: loop.inputs) {
      source << "  let mut " << input.value.name << ": " << tensorTypeText(input.value.type)
             << " = ";
      if (input.init == FreeFormInit::Zero)
        source << zeroInitializer(input.value.type.shape, input.value.type.element);
      else
        source << "undef";
      source << ";\n";
    }
    if (loop.finalTerms.size() > 1)
      source << "  let mut %out: " << tensorTypeText(returnType) << " = undef;\n";
    for (const auto &declaration: declarations)
      source << declaration << '\n';
    source << "^entry:\n";
    for (std::size_t index = 0; index < loop.inputs.size(); ++index) {
      const auto &input = loop.inputs[index];
      if (input.init != FreeFormInit::RuntimeInput)
        continue;
      source << "  " << input.value.name << " = " << parameterName(input, index) << ";\n";
    }
    source << "  %i = 0;\n"
           << "  br ^header;\n"
           << "^header:\n"
           << "  br %i < " << loop.tripCount << ", ^body, ^exit;\n"
           << "^body:\n";
    for (const auto &instruction: body)
      source << instruction << '\n';
    source << "  %i = %i + 1;\n"
           << "  br ^header;\n"
           << "^exit:\n";
    if (loop.finalTerms.size() == 1) {
      source << "  ret " << loop.finalTerms.front() << ";\n";
    } else {
      source << "  %out = tensor " << tensorOpName(loop.finalOp) << ' ' << loop.finalTerms[0]
             << ", " << loop.finalTerms[1] << ";\n";
      for (std::size_t index = 2; index < loop.finalTerms.size(); ++index)
        source << "  %out = tensor " << tensorOpName(loop.finalOp) << " %out, "
               << loop.finalTerms[index] << ";\n";
      source << "  ret %out;\n";
    }
    source << "}\n";
    return source.str();
  }

  FreeFormAssembly assembleCanonicalLoop(const TensorRegion &region, std::uint32_t tripCount) {
    FreeFormAssembly assembly;
    try {
      const auto &firstType = region.typeOf("%acc_x");
      const auto &secondType = region.typeOf("%acc_y");
      if (firstType.element != TensorElement::F32)
        throw std::invalid_argument("the two states must be f32");
      if (firstType.shape != secondType.shape || firstType.shape.empty())
        throw std::invalid_argument("the two states must share one tensor shape");
      const auto &firstResult = region.typeOf("%next_x");
      const auto &secondResult = region.typeOf("%next_y");
      if (!sameType(firstType, firstResult) || !sameType(secondType, secondResult))
        throw std::invalid_argument("each result must match its state's type");
      FreeFormLoop loop;
      loop.inputs = {
          FreeFormInput{
              {"%acc_x", firstType},
              FreeFormRole::State,
              FreeFormInit::RuntimeInput,
              "%x"
          },
          FreeFormInput{{"%acc_y", secondType}, FreeFormRole::State, FreeFormInit::Zero, {}},
      };
      loop.results = {"%next_x", "%next_y"};
      loop.updates = {{"%acc_x", "%next_x"}, {"%acc_y", "%next_y"}};
      loop.tripCount = tripCount;
      loop.finalTerms = {"%acc_x", "%acc_y"};
      loop.finalOp = TensorOp::Add;
      assembly.source = assembleCountedLoop(region, loop);
    } catch (const std::invalid_argument &error) {
      assembly.refusal = error.what();
    } catch (const std::logic_error &error) {
      assembly.refusal = error.what();
    }
    return assembly;
  }

} // namespace refractir::reify
