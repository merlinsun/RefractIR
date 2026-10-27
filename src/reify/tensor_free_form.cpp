#include "reify/tensor_free_form.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

    bool isReduction(TensorOp op) {
      return op == TensorOp::Sum || op == TensorOp::ReduceMin || op == TensorOp::ReduceMax;
    }

    bool isExtrema(TensorOp op) {
      return op == TensorOp::Min || op == TensorOp::Max || op == TensorOp::ReduceMin ||
             op == TensorOp::ReduceMax;
    }

    bool sameType(const TensorTypeInfo &left, const TensorTypeInfo &right) {
      return left.shape == right.shape && left.element == right.element;
    }

    bool containsOp(const std::vector<TensorOp> &ops, TensorOp op) {
      return std::find(ops.begin(), ops.end(), op) != ops.end();
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

    /// The bounded set of permutations one value of this rank offers. Rank two
    /// has one, rank three the five non-identity permutations, and a deeper
    /// value only the reversal, so the action list stays small.
    std::vector<Shape> transposePermutations(std::size_t rank) {
      std::vector<Shape> permutations;
      if (rank < 2)
        return permutations;
      if (rank == 2) {
        permutations.push_back(Shape{1, 0});
      } else if (rank == 3) {
        permutations.push_back(Shape{1, 0, 2});
        permutations.push_back(Shape{0, 2, 1});
        permutations.push_back(Shape{2, 1, 0});
        permutations.push_back(Shape{2, 0, 1});
        permutations.push_back(Shape{1, 2, 0});
      } else {
        Shape reversed(rank);
        for (std::size_t axis = 0; axis < rank; ++axis)
          reversed[axis] = static_cast<std::uint32_t>(rank - 1 - axis);
        permutations.push_back(std::move(reversed));
      }
      return permutations;
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
          limits_(request.limits), features_(request.features),
          element_(request.inputs.front().value.type.element), growthNodes_(growthNodes),
          random_(splitMix64(request.seed ^ 0x667265652d666f72ULL)) {
        for (std::size_t index = 0; index < request.inputs.size(); ++index) {
          PoolValue value;
          value.name = request.inputs[index].value.name;
          value.type = request.inputs[index].value.type;
          value.depth = 0;
          value.mask = 1ULL << index;
          value.node = -1;
          inputIndices_.emplace(value.name, static_cast<std::uint32_t>(index));
          pool_.push_back(std::move(value));
        }
      }

      void setRequirements(const std::vector<FreeFormRequirement> &requirements) {
        requirements_ = requirements;
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
          for (const auto &action: actions) {
            double weight = freeFormActionWeight(policy_, action.traits);
            // An enabled feature is an experimental choice rather than a
            // policy trait, so its actions carry a fixed multiplier. Without
            // it a seeded body would rarely reach the operation the request
            // asked to exercise.
            if ((action.op == TensorOp::Transpose && features_.transpose) ||
                (action.op == TensorOp::Expand && features_.expand) ||
                (isExtrema(action.op) && features_.extrema))
              weight *= 3.0;
            weight *= requirementMultiplier(action);
            weights.push_back(weight);
          }
          const auto draw =
              (static_cast<double>(random_()) + 0.5) / 4294967296.0 * totalWeight(weights);
          apply(actions[freeFormSelectWeighted(weights, draw)]);
          placed = true;
        }
        return placed;
      }

      void setPolicy(FreeFormPolicy policy) { policy_ = policy; }

      /// Bias an action towards the inputs, operations, and consumers the
      /// requirements still need. An empty requirement list returns 1, so the
      /// unconstrained corpus is byte-identical.
      [[nodiscard]] double requirementMultiplier(const Action &action) const {
        if (requirements_.empty())
          return 1.0;
        double bonus = 0.0;
        for (const auto &requirement: requirements_) {
          const auto input = inputIndices_.find(requirement.input);
          if (input == inputIndices_.end())
            continue;
          const std::uint64_t bit = 1ULL << input->second;
          const auto &left = pool_[static_cast<std::size_t>(action.lhs)];
          const bool derives =
              (left.mask & bit) != 0 ||
              (action.rhs >= 0 && (pool_[static_cast<std::size_t>(action.rhs)].mask & bit) != 0);
          if (!derives)
            continue;
          if (containsOp(requirement.pathOps, action.op) ||
              containsOp(requirement.shapeOps, action.op))
            bonus += 3.0;
          else
            bonus += 1.0;
          if (requirement.distinctConsumers) {
            if ((left.mask & bit) != 0 && left.consumers < 2)
              bonus += 1.5;
            if (action.rhs >= 0 && (pool_[static_cast<std::size_t>(action.rhs)].mask & bit) != 0 &&
                pool_[static_cast<std::size_t>(action.rhs)].consumers < 2)
              bonus += 1.5;
          }
        }
        return 1.0 + bonus;
      }

      struct PathHit {
        /// Value indices from the input to the value, inputs included.
        std::vector<std::uint32_t> values;
        /// Node indices along the path, in the same order as `values[1:]`.
        std::vector<std::uint32_t> nodes;
      };

      /// A path from `value` to the required input whose operations contain
      /// one of `pathOps` and, when requested, one of `shapeOps`. This checks
      /// a single path; operations on separate branches do not combine.
      [[nodiscard]] std::optional<PathHit> requirementPath(
          std::uint32_t value, const FreeFormRequirement &requirement
      ) const {
        const auto input = inputIndices_.find(requirement.input);
        if (input == inputIndices_.end())
          return std::nullopt;
        PathHit hit;
        std::vector<char> active(pool_.size(), 0);
        std::function<bool(std::uint32_t, bool, bool)> walk =
            [&](std::uint32_t current, bool hasPathOp, bool hasShapeOp) -> bool {
          if (current == input->second) {
            if (hasPathOp && (requirement.shapeOps.empty() || hasShapeOp)) {
              hit.values.push_back(current);
              return true;
            }
            return false;
          }
          if (current >= pool_.size() || active[current])
            return false;
          const auto &entry = pool_[current];
          if (entry.node < 0)
            return false;
          active[current] = 1;
          const auto &node = nodes_[static_cast<std::size_t>(entry.node)];
          const bool nextPathOp = hasPathOp || containsOp(requirement.pathOps, node.op);
          const bool nextShapeOp = hasShapeOp || containsOp(requirement.shapeOps, node.op);
          for (const auto operand: node.operands) {
            if (operand < 0)
              continue;
            if (walk(static_cast<std::uint32_t>(operand), nextPathOp, nextShapeOp)) {
              hit.values.push_back(current);
              hit.nodes.push_back(static_cast<std::uint32_t>(entry.node));
              active[current] = 0;
              return true;
            }
          }
          active[current] = 0;
          return false;
        };
        if (!walk(value, requirement.pathOps.empty(), requirement.shapeOps.empty()))
          return std::nullopt;
        return hit;
      }

      /// Try up to `limit` actions that close one requirement for a result of
      /// `type`. The bridge walks forward from a value the input already
      /// reaches and prefers the missing required operation, then a shape
      /// restoration to the result type. Actions are charged to the node
      /// budget by the caller.
      [[nodiscard]] std::optional<std::uint32_t> bridgeForRequirement(
          const FreeFormRequirement &requirement, const TensorTypeInfo &type,
          std::uint32_t limit
      ) {
        const auto input = inputIndices_.find(requirement.input);
        if (input == inputIndices_.end())
          return std::nullopt;
        const std::uint64_t bit = 1ULL << input->second;
        std::optional<std::uint32_t> start;
        std::uint32_t bestDepth = 0;
        for (std::uint32_t index = 0; index < pool_.size(); ++index) {
          const auto &value = pool_[index];
          if ((value.mask & bit) == 0)
            continue;
          if (sameType(value.type, type) && (!start || value.depth > bestDepth)) {
            start = index;
            bestDepth = value.depth;
          }
        }
        if (!start) {
          for (std::uint32_t index = 0; index < pool_.size(); ++index) {
            const auto &value = pool_[index];
            if ((value.mask & bit) == 0)
              continue;
            if (!start || value.depth > bestDepth) {
              start = index;
              bestDepth = value.depth;
            }
          }
        }
        if (!start)
          return std::nullopt;
        std::uint32_t current = *start;
        bool hasPathOp = false;
        bool hasShapeOp = false;
        std::vector<TensorOp> used;
        for (std::uint32_t step = 0; step < limit; ++step) {
          auto actions = enumerate();
          std::optional<Action> chosen;
          int bestScore = std::numeric_limits<int>::min();
          for (const auto &action: actions) {
            if (action.lhs != static_cast<std::int32_t>(current))
              continue;
            const bool nextPathOp =
                hasPathOp || containsOp(requirement.pathOps, action.op);
            const bool nextShapeOp =
                hasShapeOp || containsOp(requirement.shapeOps, action.op);
            const bool done = sameType(action.type, type) && nextPathOp &&
                              (requirement.shapeOps.empty() || nextShapeOp);
            int score = 0;
            if (done)
              score = 100;
            else if (!hasShapeOp && containsOp(requirement.shapeOps, action.op))
              score = 75;
            else if (sameType(action.type, type) && nextPathOp && !nextShapeOp)
              score = 70;
            else if (!hasPathOp && containsOp(requirement.pathOps, action.op))
              score = 60;
            else if (sameType(action.type, type))
              score = 40;
            else if (action.op == TensorOp::Broadcast)
              score = 30;
            else
              score = 10 + (action.traits.shapeChange ? 1 : 0);
            if (std::find(used.begin(), used.end(), action.op) != used.end())
              score -= 25;
            if (score > bestScore) {
              bestScore = score;
              chosen = action;
            }
          }
          if (!chosen)
            return std::nullopt;
          apply(*chosen);
          current = static_cast<std::uint32_t>(pool_.size() - 1);
          hasPathOp = hasPathOp || containsOp(requirement.pathOps, chosen->op);
          hasShapeOp = hasShapeOp || containsOp(requirement.shapeOps, chosen->op);
          used.push_back(chosen->op);
          if (sameType(pool_[current].type, type) && hasPathOp &&
              (requirement.shapeOps.empty() || hasShapeOp))
            return current;
        }
        return std::nullopt;
      }

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
              (isReduction(op) || op == TensorOp::Broadcast || op == TensorOp::Transpose ||
               op == TensorOp::Expand) &&
              type.shape != left.type.shape;
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
            if (features_.extrema) {
              Action minimum{};
              minimum.op = TensorOp::Min;
              minimum.lhs = lhs;
              minimum.rhs = rhs;
              minimum.type = type;
              minimum.traits = traitsOf(TensorOp::Min, lhs, rhs, type);
              push(minimum);
              Action maximum{};
              maximum.op = TensorOp::Max;
              maximum.lhs = lhs;
              maximum.rhs = rhs;
              maximum.type = type;
              maximum.traits = traitsOf(TensorOp::Max, lhs, rhs, type);
              push(maximum);
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
              if (features_.extrema) {
                for (const auto op: {TensorOp::ReduceMin, TensorOp::ReduceMax}) {
                  Action extrema{};
                  extrema.op = op;
                  extrema.lhs = index;
                  extrema.axis = axis;
                  extrema.keepDims = keep;
                  extrema.type = TensorTypeInfo{shape, element};
                  extrema.traits = traitsOf(op, index, -1, extrema.type);
                  push(extrema);
                }
              }
            }
          }
        }

        if (features_.transpose) {
          for (std::int32_t index = 0; index < static_cast<std::int32_t>(pool_.size()); ++index) {
            const auto &value = pool_[static_cast<std::size_t>(index)];
            const auto rank = value.type.shape.size();
            for (const auto &permutation: transposePermutations(rank)) {
              Shape shape(rank);
              for (std::size_t axis = 0; axis < rank; ++axis)
                shape[axis] = value.type.shape[permutation[axis]];
              if (!withinLimits(shape, limits_))
                continue;
              Action action{};
              action.op = TensorOp::Transpose;
              action.lhs = index;
              action.attribute = permutation;
              action.type = TensorTypeInfo{shape, element};
              action.traits = traitsOf(TensorOp::Transpose, index, -1, action.type);
              push(action);
            }
          }
        }

        if (features_.expand) {
          for (std::int32_t index = 0; index < static_cast<std::int32_t>(pool_.size()); ++index) {
            const auto &value = pool_[static_cast<std::size_t>(index)];
            const auto rank = value.type.shape.size();
            if (rank == 0 || rank + 1 > limits_.maxRank)
              continue;
            for (std::size_t axis = 0; axis <= rank; ++axis) {
              Shape shape = value.type.shape;
              shape.insert(shape.begin() + static_cast<std::ptrdiff_t>(axis), 1);
              if (!withinLimits(shape, limits_))
                continue;
              Action action{};
              action.op = TensorOp::Expand;
              action.lhs = index;
              action.axis = static_cast<std::int64_t>(axis);
              action.type = TensorTypeInfo{shape, element};
              action.traits = traitsOf(TensorOp::Expand, index, -1, action.type);
              push(action);
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
      FreeFormFeatures features_;
      TensorElement element_ = TensorElement::I32;
      std::uint32_t growthNodes_ = 0;
      FreeFormPolicy policy_ = FreeFormPolicy::Unguided;
      std::uint32_t temporaries_ = 0;
      std::uint32_t multiplications_ = 0;
      std::mt19937 random_;
      std::vector<PoolValue> pool_;
      std::vector<NodeRecord> nodes_;
      std::unordered_map<std::string, std::uint32_t> inputIndices_;
      std::vector<FreeFormRequirement> requirements_;
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

    /// One path from a graph value to an input, with the operations it uses.
    struct GraphPath {
      std::vector<std::uint32_t> values;
      std::vector<std::uint32_t> nodes;
    };

    [[nodiscard]] std::optional<GraphPath> requirementGraphPath(
        const FreeFormGraphView &graph, std::uint32_t value, std::uint32_t input,
        const FreeFormRequirement &requirement
    ) {
      GraphPath path;
      std::vector<char> active(graph.valueNames.size(), 0);
      std::function<bool(std::uint32_t, bool, bool)> walk =
          [&](std::uint32_t current, bool hasPathOp, bool hasShapeOp) -> bool {
        if (current == input) {
          if (hasPathOp && (requirement.shapeOps.empty() || hasShapeOp)) {
            path.values.push_back(current);
            return true;
          }
          return false;
        }
        if (current >= graph.valueNames.size() || current < graph.inputCount || active[current])
          return false;
        const auto node = current - graph.inputCount;
        if (node >= graph.nodeOps.size())
          return false;
        active[current] = 1;
        const bool nextPathOp =
            hasPathOp || containsOp(requirement.pathOps, graph.nodeOps[node]);
        const bool nextShapeOp =
            hasShapeOp || containsOp(requirement.shapeOps, graph.nodeOps[node]);
        for (const auto operand: graph.nodeOperands[node]) {
          if (operand < 0)
            continue;
          if (walk(static_cast<std::uint32_t>(operand), nextPathOp, nextShapeOp)) {
            path.values.push_back(current);
            path.nodes.push_back(node);
            active[current] = 0;
            return true;
          }
        }
        active[current] = 0;
        return false;
      };
      if (!walk(value, requirement.pathOps.empty(), requirement.shapeOps.empty()))
        return std::nullopt;
      return path;
    }

    [[nodiscard]] bool graphDependsOn(
        const FreeFormGraphView &graph, std::uint32_t value, std::uint32_t input
    ) {
      if (value == input)
        return true;
      if (value < graph.inputCount || value >= graph.valueNames.size())
        return false;
      const auto node = value - graph.inputCount;
      for (const auto operand: graph.nodeOperands[node])
        if (operand >= 0 && graphDependsOn(graph, static_cast<std::uint32_t>(operand), input))
          return true;
      return false;
    }

    /// The live values, indexed by pool position.
    [[nodiscard]] std::vector<char> liveClosure(
        const FreeFormGraphView &graph, std::uint32_t root
    ) {
      return liveValues(
          static_cast<std::uint32_t>(graph.valueNames.size()), graph.nodeOperands,
          graph.inputCount, std::vector<std::uint32_t>{root}
      );
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

  FreeFormWindowFacts classifyWindow(
      std::uint32_t rows, std::uint32_t stride, std::uint32_t active,
      std::uint64_t tripCount, bool v3, bool paddedTile
  ) {
    const bool interior =
        tripCount != 0 &&
        static_cast<std::uint64_t>(tripCount - 1) * stride + (rows - 1) < active;
    const char *predicate = interior ? "all_true" : (stride == 0 ? "constant_partial" : "runtime");
    const char *validity = !v3 ? "absent" : (paddedTile ? "nonconstant" : "constant");
    return FreeFormWindowFacts{
        interior ? "interior" : "tail", validity, predicate, "not_observed"
    };
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

  FreeFormGraphKeys freeFormGraphKeys(
      const FreeFormGraphView &graph, const std::vector<std::string> &inputRoles
  ) {
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
    if (inputRoles.size() != graph.inputCount)
      throw std::invalid_argument("free-form graph input roles do not match its inputs");
    for (const auto root: graph.roots)
      if (root >= valueCount)
        throw std::invalid_argument("free-form graph root is outside its pool");
    for (const auto &operands: graph.nodeOperands)
      for (const auto operand: operands)
        if (operand < 0 || operand >= static_cast<std::int32_t>(valueCount))
          throw std::invalid_argument("free-form graph operand is outside its pool");

    // Canonical producer IDs: one deterministic depth-first traversal from the
    // ordered roots, visiting operands in operand order. Creation order and
    // value names never enter the identity.
    std::vector<std::uint32_t> producer(valueCount, UINT32_MAX);
    std::vector<std::uint32_t> producerValues;
    std::function<void(std::uint32_t)> visit = [&](std::uint32_t value) {
      if (value < graph.inputCount || value >= valueCount)
        return;
      if (producer[value] != UINT32_MAX)
        return;
      producer[value] = static_cast<std::uint32_t>(producerValues.size());
      producerValues.push_back(value);
      const auto node = value - graph.inputCount;
      for (const auto operand: graph.nodeOperands[node])
        if (operand >= 0)
          visit(static_cast<std::uint32_t>(operand));
    };
    for (const auto root: graph.roots)
      visit(root);

    const auto reference = [&](std::int32_t value) -> std::string {
      if (value < 0)
        return "?";
      if (value < static_cast<std::int32_t>(graph.inputCount))
        return "i" + std::to_string(value);
      return "n" + std::to_string(producer[static_cast<std::size_t>(value)]);
    };

    std::ostringstream connectivity;
    std::ostringstream typed;
    connectivity << "sig2;roles:";
    typed << "fp3;roles:";
    for (std::size_t input = 0; input < inputRoles.size(); ++input) {
      if (input)
        connectivity << ',';
      if (input)
        typed << ',';
      connectivity << inputRoles[input];
      typed << inputRoles[input];
    }
    connectivity << ';';
    typed << ';';
    // Input types are part of the typed identity but not of connectivity.
    for (std::uint32_t input = 0; input < graph.inputCount; ++input) {
      typed << "i" << input << ':' << inputRoles[input] << ':'
            << shapeText(graph.valueTypes[input].shape) << ':'
            << (graph.valueTypes[input].element == TensorElement::F32 ? "f32" : "i32")
            << ';';
    }
    for (const auto value: producerValues) {
      const auto node = value - graph.inputCount;
      const auto axis = graph.nodeAxes.size() == graph.nodeOps.size()
                            ? graph.nodeAxes[node]
                            : 0;
      const bool keep = graph.nodeKeepDims.size() == graph.nodeOps.size() &&
                        graph.nodeKeepDims[node] != 0;
      connectivity << tensorOpName(graph.nodeOps[node]) << '(';
      typed << tensorOpName(graph.nodeOps[node]) << '[' << axis << ',' << (keep ? 'k' : 'd')
            << ']';
      if (graph.nodeAttributes.size() == graph.nodeOps.size() &&
          !graph.nodeAttributes[node].empty())
        typed << "attr" << shapeText(graph.nodeAttributes[node]);
      typed << '(';
      bool first = true;
      for (const auto operand: graph.nodeOperands[node]) {
        if (operand < 0)
          continue;
        if (!first) {
          connectivity << ',';
          typed << ',';
        }
        first = false;
        connectivity << reference(operand);
        typed << reference(operand);
      }
      connectivity << ");";
      typed << ") :" << shapeText(graph.valueTypes[value].shape) << ':'
            << (graph.valueTypes[value].element == TensorElement::F32 ? "f32" : "i32") << ';';
    }
    connectivity << "roots:";
    typed << "roots:";
    for (std::size_t root = 0; root < graph.roots.size(); ++root) {
      if (root) {
        connectivity << ',';
        typed << ',';
      }
      connectivity << "r" << root << '=' << reference(static_cast<std::int32_t>(graph.roots[root]));
      typed << "r" << root << '=' << reference(static_cast<std::int32_t>(graph.roots[root]));
    }
    return {connectivity.str(), typed.str()};
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
        if ((isReduction(op) || op == TensorOp::Broadcast || op == TensorOp::Transpose ||
             op == TensorOp::Expand) &&
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
      std::unordered_set<std::string> inputNames;
      for (const auto &input: request.inputs)
        inputNames.insert(input.value.name);
      for (const auto &requirement: request.requirements) {
        if (!inputNames.contains(requirement.input))
          throw std::invalid_argument(
              "free-form requirement input is not a declared input: " + requirement.input
          );
        if (requirement.results.empty())
          throw std::invalid_argument("free-form requirement needs at least one result");
        if (requirement.distinctConsumers && requirement.results.size() < 2)
          throw std::invalid_argument(
              "free-form requirement needs two results for distinct consumers"
          );
        std::unordered_set<std::uint32_t> seen;
        for (const auto index: requirement.results) {
          if (index >= request.results.size())
            throw std::invalid_argument(
                "free-form requirement result is outside the request"
            );
          if (!seen.insert(index).second)
            throw std::invalid_argument("free-form requirement repeats a result");
        }
      }
    }

    std::string resolveResultName(const FreeFormRequest &request, std::size_t index) {
      if (!request.results[index].name.empty())
        return request.results[index].name;
      return "%next_" + std::to_string(index);
    }

  } // namespace

  namespace {
    /// A transpose changes a non-square type, so the required path needs a
    /// bounded return bridge before it can land on the result type again.
    std::uint32_t bridgeLimitFor(
        const FreeFormRequirement &requirement, const TensorTypeInfo &type
    ) {
      const bool transposes = std::find(
          requirement.shapeOps.begin(), requirement.shapeOps.end(),
          TensorOp::Transpose
      ) != requirement.shapeOps.end();
      const bool nonSquare = type.shape.size() > 1 &&
          std::any_of(
              type.shape.begin(), type.shape.end(),
              [first = type.shape.front()](std::int64_t extent) {
                return extent != first;
              }
          );
      return transposes && nonSquare ? 6 : 3;
    }
  } // namespace

  FreeFormRegion buildFreeFormRegion(const FreeFormRequest &request) {
    validateRequest(request);
    const auto resultCount = static_cast<std::uint32_t>(request.results.size());
    if (request.limits.maxNodes < resultCount)
      throw std::runtime_error(
          "free_form_exhaustion: the node budget is smaller than the requested result count"
      );
    // Path requirements may need a bounded bridge after growth, so their
    // worst-case three nodes per requested result are reserved up front. A
    // distinct-consumer requirement is construction bias, not a type bridge,
    // and takes no reservation.
    std::uint32_t bridgeReserve = 0;
    for (const auto &requirement: request.requirements) {
      if (requirement.pathOps.empty() && requirement.shapeOps.empty())
        continue;
      for (const auto result: requirement.results) {
        if (result >= request.results.size())
          throw std::invalid_argument(
              "free-form requirement names a result outside the request"
          );
        bridgeReserve += bridgeLimitFor(requirement, request.results[result].type);
      }
    }
    if (request.limits.maxNodes < resultCount + bridgeReserve)
      throw std::runtime_error(
          "free_form_exhaustion: node_budget_exhausted: the node budget cannot reserve the "
          "requested path bridges"
      );
    // One node per result is held back: a requested result type that the grown
    // pool does not reach can still be placed from an existing value.
    const auto growthNodes = request.limits.maxNodes - resultCount - bridgeReserve;
    std::string lastReason = "no attempt placed every requested result";
    for (std::uint32_t attempt = 0; attempt < request.limits.maxAttempts; ++attempt) {
      FreeFormRequest seed = request;
      seed.seed = static_cast<std::uint32_t>(splitMix64(
          static_cast<std::uint64_t>(request.seed) ^ (0x9e3779b97f4a7c15ULL * (attempt + 1))
      ));
      Growth growth(seed, growthNodes);
      growth.setPolicy(request.policy);
      growth.setRequirements(request.requirements);
      growth.grow();

      std::unordered_map<std::string, std::uint64_t> inputBits;
      for (std::uint32_t input = 0; input < request.inputs.size(); ++input)
        inputBits.emplace(request.inputs[input].value.name, 1ULL << input);
      const auto satisfiesRequirement = [&](std::uint32_t candidate,
                                            const FreeFormRequirement &requirement) {
        const auto &value = growth.pool()[candidate];
        const auto bit = inputBits.find(requirement.input);
        if (bit == inputBits.end() || (value.mask & bit->second) == 0)
          return false;
        if (requirement.pathOps.empty() && requirement.shapeOps.empty())
          return true;
        return growth.requirementPath(candidate, requirement).has_value();
      };
      std::vector<std::uint32_t> roots;
      bool complete = true;
      for (std::uint32_t index = 0; index < resultCount; ++index) {
        const auto type = request.results[index].type;
        std::vector<const FreeFormRequirement *> covering;
        for (const auto &requirement: request.requirements)
          if (std::find(requirement.results.begin(), requirement.results.end(), index) !=
              requirement.results.end())
            covering.push_back(&requirement);
        std::optional<std::uint32_t> best;
        std::uint32_t bestSatisfied = 0;
        const bool requireComputed = request.results[index].requireComputed;
        const auto choose = [&]() {
          best.reset();
          bestSatisfied = 0;
          std::uint32_t bestScore = 0;
          for (const auto candidate: growth.valuesOfType(type)) {
            if (std::find(roots.begin(), roots.end(), candidate) != roots.end())
              continue;
            if (requireComputed && growth.pool()[candidate].node < 0)
              continue;
            std::uint32_t satisfied = 0;
            for (const auto *requirement: covering)
              satisfied += satisfiesRequirement(candidate, *requirement) ? 1U : 0U;
            const auto &value = growth.pool()[candidate];
            const std::uint32_t score =
                (covering.empty() || satisfied == covering.size() ? 1U << 30 : satisfied << 28) +
                (value.node >= 0 ? 1U << 27 : 0U) + (value.depth << 16) + candidate;
            if (!best || score > bestScore) {
              best = candidate;
              bestScore = score;
              bestSatisfied = satisfied;
            }
          }
        };
        choose();
        // A requirement the grown pool does not satisfy gets a bounded bridge,
        // charged to the reserved budget.
        for (const auto *requirement: covering) {
          if (best && satisfiesRequirement(*best, *requirement))
            continue;
          if (growth.nodesPlaced() >= request.limits.maxNodes)
            break;
          if (!growth.bridgeForRequirement(
                  *requirement, type, bridgeLimitFor(*requirement, type)
              ))
            break;
          choose();
        }
        if (!best) {
          if (growth.nodesPlaced() >= request.limits.maxNodes) {
            lastReason = "node_budget_exhausted: the node budget was spent before result " +
                         std::to_string(index) + " could be placed";
            complete = false;
            break;
          }
          const auto placed = growth.placeForType(type);
          if (!placed) {
            lastReason = "required_path_missing: no legal action produces result " +
                         std::to_string(index);
            complete = false;
            break;
          }
          best = *placed;
          choose();
        }
        if (!covering.empty() && (!best || bestSatisfied < covering.size())) {
          const auto *missing = covering.front();
          const auto bit = inputBits.find(missing->input);
          bool derives = false;
          for (const auto &value: growth.pool())
            derives = derives || (bit != inputBits.end() && (value.mask & bit->second) != 0);
          if (!derives)
            lastReason = "required_input_unreachable: " + missing->input + " -> result " +
                         std::to_string(index);
          else
            lastReason = "required_path_missing: " + missing->input + " -> result " +
                         std::to_string(index);
          complete = false;
          break;
        }
        if (requireComputed && (!best || growth.pool()[*best].node < 0)) {
          lastReason = "required_computed_root_missing: result " +
                       std::to_string(index) + " has no computed root";
          complete = false;
          break;
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
      for (std::uint32_t index = 0; index < resultCount; ++index)
        if (request.results[index].requireComputed && growth.pool()[roots[index]].node < 0) {
          lastReason = "required_computed_root_missing: result " + std::to_string(index) +
                       " kept an input root";
          complete = false;
          break;
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
      for (const auto &record: nodes) {
        view.nodeAxes.push_back(record.axis);
        view.nodeKeepDims.push_back(record.keepDims ? 1 : 0);
        view.nodeAttributes.push_back(record.attribute);
      }
      view.roots = roots;

      // Requirements are validated on the live graph after pruning. A path
      // counts only while it is live, and two consumers are two distinct nodes
      // reaching two distinct listed roots; a shared value selected as a root
      // is not a consumer.
      const auto live = liveValues(
          static_cast<std::uint32_t>(view.valueTypes.size()), view.nodeOperands,
          view.inputCount, view.roots);
      std::vector<std::uint32_t> normalized(view.valueTypes.size(), 0);
      std::uint32_t nextLive = 0;
      for (std::uint32_t value = view.inputCount; value < view.valueTypes.size(); ++value)
        if (live[value])
          normalized[value] = nextLive++;
      std::unordered_map<std::string, std::uint32_t> inputIndexByName;
      for (std::uint32_t input = 0; input < view.inputCount; ++input)
        inputIndexByName.emplace(view.valueNames[input], input);
      std::vector<std::vector<char>> reachesRoot(
          view.valueTypes.size(), std::vector<char>(roots.size(), 0));
      for (std::uint32_t root = 0; root < roots.size(); ++root) {
        const auto closure = liveClosure(view, roots[root]);
        for (std::uint32_t value = 0; value < closure.size(); ++value)
          if (closure[value])
            reachesRoot[value][root] = 1;
      }
      std::vector<FreeFormRequirementWitness> witnesses;
      bool requirementsComplete = true;
      std::string requirementReason;
      for (const auto &requirement: request.requirements) {
        FreeFormRequirementWitness witness;
        witness.input = requirement.input;
        const auto input = inputIndexByName.find(requirement.input);
        for (const auto resultIndex: requirement.results) {
          const auto root = roots[resultIndex];
          std::optional<GraphPath> path;
          if (input != inputIndexByName.end())
            path = requirementGraphPath(view, root, input->second, requirement);
          if (!path) {
            requirementsComplete = false;
            requirementReason = "required_path_missing: " + requirement.input + " -> result " +
                                std::to_string(resultIndex);
            break;
          }
          witness.results.push_back(resultNames[resultIndex]);
          if (witness.path.empty()) {
            for (const auto value: path->values)
              witness.path.push_back(emittedNames[value]);
            for (const auto node: path->nodes) {
              witness.operations.emplace_back(tensorOpName(view.nodeOps[node]));
              witness.operationIds.push_back(normalized[view.inputCount + node]);
            }
          }
        }
        if (!requirementsComplete)
          break;
        if (requirement.distinctConsumers) {
          bool found = false;
          std::uint32_t firstConsumer = 0;
          std::uint32_t secondConsumer = 0;
          std::uint32_t firstRoot = 0;
          std::uint32_t secondRoot = 0;
          for (std::uint32_t value = 0; value < view.valueTypes.size() && !found; ++value) {
            if (!live[value] || input == inputIndexByName.end() ||
                !graphDependsOn(view, value, input->second))
              continue;
            std::vector<std::uint32_t> consumers;
            for (std::uint32_t node = 0; node < view.nodeOps.size(); ++node) {
              const auto consumer = view.inputCount + node;
              if (!live[consumer])
                continue;
              const auto &operands = view.nodeOperands[node];
              if (std::find(
                      operands.begin(), operands.end(), static_cast<std::int32_t>(value)
                  ) != operands.end())
                consumers.push_back(consumer);
            }
            for (std::size_t left = 0; left < consumers.size() && !found; ++left)
              for (std::size_t right = left + 1; right < consumers.size() && !found; ++right)
                for (const auto one: requirement.results)
                  for (const auto two: requirement.results)
                    if (one != two && reachesRoot[consumers[left]][one] &&
                        reachesRoot[consumers[right]][two]) {
                      found = true;
                      firstConsumer = consumers[left];
                      secondConsumer = consumers[right];
                      firstRoot = one;
                      secondRoot = two;
                    }
          }
          if (!found) {
            requirementsComplete = false;
            requirementReason = "distinct_consumers_missing: " + requirement.input;
            break;
          }
          witness.distinctConsumersSatisfied = true;
          witness.consumers = {emittedNames[firstConsumer], emittedNames[secondConsumer]};
          witness.consumerResults = {resultNames[firstRoot], resultNames[secondRoot]};
        }
        witnesses.push_back(std::move(witness));
      }
      if (!requirementsComplete) {
        lastReason = requirementReason;
        complete = false;
      }
      if (!complete)
        continue;

      FreeFormLiveDag liveDag;
      for (std::uint32_t value = 0; value < view.valueTypes.size(); ++value)
        if (live[value])
          liveDag.values.push_back(emittedNames[value]);
      for (std::uint32_t node = 0; node < view.nodeOps.size(); ++node) {
        const auto value = view.inputCount + node;
        if (!live[value])
          continue;
        liveDag.ops.emplace_back(tensorOpName(view.nodeOps[node]));
        std::vector<std::string> operands;
        for (const auto operand: view.nodeOperands[node]) {
          if (operand < 0)
            continue;
          operands.push_back(emittedNames[static_cast<std::size_t>(operand)]);
        }
        liveDag.operands.push_back(std::move(operands));
      }
      liveDag.roots = resultNames;
      std::vector<std::string> inputRoles;
      inputRoles.reserve(request.inputs.size());
      for (const auto &input: request.inputs)
        inputRoles.emplace_back(input.role == FreeFormRole::State ? "state" : "invariant");
      const auto keys = freeFormGraphKeys(view, inputRoles);
      liveDag.connectivitySignature = keys.connectivitySignature;
      liveDag.typedFingerprint = keys.typedFingerprint;

      FreeFormRegion built{
          std::move(region), summarizeFreeFormGraph(view, request.policy), {}
      };
      built.structure.requirementWitnesses = std::move(witnesses);
      built.structure.liveDag = std::move(liveDag);
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

  FreeFormWindowWitness witnessWindow(
      const FreeFormWindow &window, std::uint32_t tripCount, std::uint32_t sourceExtent
  ) {
    if (window.rows == 0 || window.active == 0 || window.rows > sourceExtent ||
        window.active > sourceExtent)
      throw std::invalid_argument(
          "free-form window witness: rows and active must fit the source extent"
      );
    FreeFormWindowWitness witness;
    witness.allActive = true;
    witness.perTripActive.resize(tripCount);
    for (std::uint32_t trip = 0; trip < tripCount; ++trip) {
      witness.perTripActive[trip].resize(window.rows);
      for (std::uint32_t row = 0; row < window.rows; ++row) {
        const auto address =
            static_cast<std::uint64_t>(trip) * window.stride + static_cast<std::uint64_t>(row);
        const bool active = address < window.active;
        witness.perTripActive[trip][row] = active;
        witness.allActive = witness.allActive && active;
        witness.computedMax = std::max(witness.computedMax, address);
      }
    }
    if (witness.computedMax >= sourceExtent)
      throw std::invalid_argument(
          "free-form window witness: maximum address reaches the source extent"
      );
    return witness;
  }

  void validateFreeFormWindows(const FreeFormLoop &loop) {
    std::unordered_map<std::string, std::size_t> parameters;
    std::unordered_map<std::string, std::size_t> windowInputs;
    std::unordered_map<std::string, TensorTypeInfo> sources;
    for (std::size_t index = 0; index < loop.inputs.size(); ++index) {
      const auto &input = loop.inputs[index];
      if (input.init == FreeFormInit::RuntimeInput) {
        parameters.emplace(parameterName(input, index), index);
        sources.emplace(parameterName(input, index), input.value.type);
      } else if (input.init == FreeFormInit::Window) {
        windowInputs.emplace(input.value.name, index);
      }
    }
    std::unordered_set<std::string> sourceNames;
    std::unordered_set<std::string> localNames;
    for (const auto &input: loop.inputs)
      localNames.insert(input.value.name);
    for (const auto &source: loop.windowSources) {
      requireLocalName(source.parameter, "counted loop window source");
      if (source.parameter == "%i" || source.parameter == "%out" ||
          std::find(loop.reservedNames.begin(), loop.reservedNames.end(), source.parameter) !=
              loop.reservedNames.end())
        throw std::invalid_argument(
            "counted loop window source reuses a loop local: " + source.parameter
        );
      if (localNames.contains(source.parameter))
        throw std::invalid_argument(
            "counted loop window source repeats a loop local: " + source.parameter
        );
      if (parameters.contains(source.parameter))
        throw std::invalid_argument(
            "counted loop window source repeats a runtime parameter: " + source.parameter
        );
      if (!sourceNames.insert(source.parameter).second)
        throw std::invalid_argument(
            "counted loop window source is declared more than once: " + source.parameter
        );
      if (source.type.shape.empty())
        throw std::invalid_argument("counted loop window source must be a tensor");
      sources.emplace(source.parameter, source.type);
    }
    std::unordered_set<std::string> declared;
    for (const auto &window: loop.windows) {
      const auto target = windowInputs.find(window.result);
      if (target == windowInputs.end())
        throw std::invalid_argument(
            "free-form window: result must name a window-initialized input"
        );
      if (!declared.insert(window.result).second)
        throw std::invalid_argument(
            "free-form window: a window-initialized input has more than one window"
        );
      const auto source = sources.find(window.source);
      if (source == sources.end())
        throw std::invalid_argument(
            "free-form window: source must be a runtime parameter or a window backing "
            "parameter"
        );
      if (window.index != "%i")
        throw std::invalid_argument(
            "free-form window: index must be the induction variable"
        );
      for (const auto &[state, result]: loop.updates)
        if (state == window.result)
          throw std::invalid_argument(
              "free-form window: an update cannot target the window result"
          );
      const auto &sourceType = source->second;
      const auto &targetType = loop.inputs[target->second].value.type;
      if (sourceType.shape.empty())
        throw std::invalid_argument("free-form window: source must be a tensor");
      const auto extent = sourceType.shape.front();
      if (window.rows == 0 || window.active == 0 || window.rows > extent ||
          window.active > extent)
        throw std::invalid_argument(
            "free-form window: rows, stride, and active must fit the source extent"
        );
      const auto maximum =
          static_cast<std::uint64_t>(window.stride) *
              static_cast<std::uint64_t>(loop.tripCount - 1) +
          window.rows;
      if (maximum > extent)
        throw std::invalid_argument(
            "free-form window: rows, stride, and active must fit the source extent"
        );
      Shape expected = sourceType.shape;
      expected.front() = window.rows;
      if (targetType.shape != expected || targetType.element != sourceType.element)
        throw std::invalid_argument(
            "free-form window: result shape must match rows and the source's trailing extents"
        );
      const bool floatElement = targetType.element == TensorElement::F32;
      if (floatElement != (window.fillKind == FreeFormFillKind::Float))
        throw std::invalid_argument(
            "free-form window: fill kind must match the result element"
        );
    }
    for (const auto &[name, index]: windowInputs)
      if (!declared.contains(name))
        throw std::invalid_argument(
            "free-form window: every window-initialized input needs a window"
        );
  }

  std::string windowFillText(const FreeFormWindow &window) {
    if (window.fillKind == FreeFormFillKind::Integer)
      return std::to_string(window.fillInteger);
    auto text = formatDouble(window.fillFloat);
    if (text.find_first_of(".eE") == std::string::npos)
      text += ".0";
    return text;
  }

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
    for (const auto &source: loop.windowSources) {
      if (!parameters.insert(source.parameter).second)
        throw std::invalid_argument("counted loop parameter is declared more than once");
    }
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

    validateFreeFormWindows(loop);

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
    auto reserved = loop.reservedNames;
    for (const auto &source: loop.windowSources)
      reserved.push_back(source.parameter);
    const auto declarations = region.renderDeclarations(updates, reserved);
    const auto body = region.renderBody(updates, reserved);
    // A window backing parameter lives in the signature, so it cannot share a
    // name with any local the region declares (a live node or a snapshot).
    for (const auto &source: loop.windowSources) {
      const std::string declaration = "  let mut " + source.parameter + ":";
      if (std::any_of(
              declarations.begin(), declarations.end(),
              [&](const std::string &line) { return line.starts_with(declaration); }
          ))
        throw std::invalid_argument(
            "counted loop window source collides with a region local: " + source.parameter
        );
    }

    std::ostringstream source;
    source << "fun @" << loop.entry << '(';
    bool first = true;
    for (const auto &windowSource: loop.windowSources) {
      if (!first)
        source << ", ";
      first = false;
      source << windowSource.parameter << ": " << tensorTypeText(windowSource.type);
    }
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
    for (const auto &window: loop.windows) {
      source << "  " << window.result << " = tensor window " << window.source << ", "
             << window.index << ", " << windowFillText(window) << ", [" << window.rows << ", "
             << window.stride << ", " << window.active << "];\n";
    }
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

  namespace {
    /// A window tile reads runtime data inside the active rows and the fill
    /// outside them, so either can be the larger.
    double windowInputBound(const FreeFormLoop &loop, const std::string &name, double inputBound) {
      for (const auto &window: loop.windows) {
        if (window.result != name)
          continue;
        const auto fill = window.fillKind == FreeFormFillKind::Float
                              ? std::fabs(window.fillFloat)
                              : std::fabs(static_cast<double>(window.fillInteger));
        return std::max(inputBound, fill);
      }
      throw std::invalid_argument("free-form bound: no window produces " + name);
    }

    double foldBound(TensorOp op, double left, double right) {
      if (op == TensorOp::Mul)
        return left == 0.0 || right == 0.0 ? 0.0 : left * right;
      if (op == TensorOp::Min || op == TensorOp::Max)
        return std::max(left, right);
      return left + right;
    }
  } // namespace

  double
  freeFormMagnitudeBound(const TensorRegion &region, const FreeFormLoop &loop, double inputBound) {
    std::map<std::string, double> current;
    for (const auto &input: loop.inputs) {
      const auto &name = input.value.name;
      current[name] = input.init == FreeFormInit::Zero ? 0.0
                      : input.init == FreeFormInit::Window
                          ? windowInputBound(loop, name, inputBound)
                          : inputBound;
    }
    double worst = 0.0;
    for (const auto &[name, bound]: current)
      worst = std::max(worst, bound);
    // Each trip reads the states the previous trip wrote; invariants, window
    // tiles included, keep the bound they started with.
    for (std::uint32_t trip = 0; trip < loop.tripCount; ++trip) {
      const auto bounds = region.magnitudeBounds(current);
      for (const auto &[name, bound]: bounds)
        worst = std::max(worst, bound);
      for (const auto &[state, result]: loop.updates)
        current[state] = bounds.at(result);
    }
    if (!loop.finalTerms.empty()) {
      auto folded = current.at(loop.finalTerms.front());
      for (std::size_t index = 1; index < loop.finalTerms.size(); ++index)
        folded = foldBound(loop.finalOp, folded, current.at(loop.finalTerms[index]));
      worst = std::max(worst, folded);
    }
    return worst;
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
