#include "reify/tensor_region.hpp"

#include <algorithm>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace refractir::reify {
  namespace {

    Shape broadcastShape(const Shape &left, const Shape &right) {
      const std::size_t rank = std::max(left.size(), right.size());
      Shape result(rank, 1);
      for (std::size_t offset = 0; offset < rank; ++offset) {
        const auto lhs = offset < left.size() ? left[left.size() - 1 - offset] : 1;
        const auto rhs = offset < right.size() ? right[right.size() - 1 - offset] : 1;
        if (lhs != rhs && lhs != 1 && rhs != 1)
          throw std::invalid_argument("tensor region add operands do not broadcast");
        result[rank - 1 - offset] = std::max(lhs, rhs);
      }
      return result;
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
      if (type.shape.empty())
        return type.element == TensorElement::F32 ? "f32" : "i32";
      return "tensor " + shapeText(type.shape) +
             (type.element == TensorElement::F32 ? " f32" : " i32");
    }

  } // namespace

  TensorRegion::TensorRegion(std::vector<TensorRegionInput> inputs) {
    for (auto &input: inputs) {
      if (input.name.empty() || input.name.front() != '%')
        throw std::invalid_argument("tensor region names must be local identifiers");
      if (input.type.shape.empty())
        throw std::invalid_argument("tensor region inputs must have a tensor shape");
      if (std::any_of(input.type.shape.begin(), input.type.shape.end(), [](std::uint32_t size) {
            return size == 0;
          }))
        throw std::invalid_argument("tensor region dimensions must be positive");
      if (std::any_of(values_.begin(), values_.end(), [&](const Value &value) {
            return value.name == input.name;
          }))
        throw std::invalid_argument("tensor region value is declared more than once");
      values_.push_back(Value{std::move(input.name), std::move(input.type), -1});
    }
    if (values_.empty())
      throw std::invalid_argument("tensor region needs at least one input");
  }

  std::int32_t TensorRegion::requireValue(const std::string &name) const {
    const auto found = std::find_if(values_.begin(), values_.end(), [&](const Value &value) {
      return value.name == name;
    });
    if (found == values_.end())
      throw std::invalid_argument("tensor region references an unknown value: " + name);
    return static_cast<std::int32_t>(found - values_.begin());
  }

  const TensorTypeInfo &TensorRegion::typeOf(const std::string &name) const {
    return values_[static_cast<std::size_t>(requireValue(name))].type;
  }

  void TensorRegion::add(
      std::string name, TensorOp op, std::vector<std::string> operandNames, std::int64_t axis,
      bool keepDims, Shape attribute
  ) {
    if (name.empty() || name.front() != '%')
      throw std::invalid_argument("tensor region results must be local identifiers");
    if (std::any_of(values_.begin(), values_.end(), [&](const Value &value) {
          return value.name == name;
        }))
      throw std::invalid_argument("tensor region value is declared more than once");
    const bool binary = op == TensorOp::Add || op == TensorOp::Sub || op == TensorOp::Mul ||
                        op == TensorOp::Min || op == TensorOp::Max;
    const bool reduction = op == TensorOp::Sum || op == TensorOp::ReduceMin ||
                           op == TensorOp::ReduceMax;
    const bool broadcast = op == TensorOp::Broadcast;
    const bool transpose = op == TensorOp::Transpose;
    const bool expand = op == TensorOp::Expand;
    const bool dot = op == TensorOp::Dot;
    if (!binary && !reduction && !broadcast && !transpose && !expand && !dot)
      throw std::invalid_argument(
          "tensor region supports add, sub, mul, min, max, sum, rmin, rmax, broadcast, "
          "transpose, expand and dot"
      );
    const std::size_t expectedOperands = binary || dot ? 2 : 1;
    if (operandNames.size() != expectedOperands)
      throw std::invalid_argument("tensor region operation has the wrong operand count");
    std::vector<std::int32_t> operands;
    operands.reserve(operandNames.size());
    for (const auto &operand: operandNames)
      operands.push_back(requireValue(operand));
    const auto &firstType = values_[static_cast<std::size_t>(operands.front())].type;
    TensorTypeInfo resultType;
    resultType.element = firstType.element;
    if (binary) {
      const auto &secondType = values_[static_cast<std::size_t>(operands[1])].type;
      if (firstType.element != secondType.element)
        throw std::invalid_argument("tensor region binary operation element types differ");
      resultType.shape = broadcastShape(firstType.shape, secondType.shape);
      if (!attribute.empty())
        throw std::invalid_argument("tensor region binary operation does not accept attributes");
    } else if (dot) {
      const auto &secondType = values_[static_cast<std::size_t>(operands[1])].type;
      if (firstType.element != secondType.element)
        throw std::invalid_argument("tensor region dot element types differ");
      if (firstType.shape.size() != 2 || secondType.shape.size() != 2 ||
          firstType.shape[1] != secondType.shape[0])
        throw std::invalid_argument("tensor region dot needs [M, K] and [K, N] operands");
      if (!attribute.empty())
        throw std::invalid_argument("tensor region dot does not accept attributes");
      resultType.shape = {firstType.shape[0], secondType.shape[1]};
    } else if (reduction) {
      if (!attribute.empty())
        throw std::invalid_argument("tensor region reduction does not accept a shape attribute");
      if (axis < -static_cast<std::int64_t>(firstType.shape.size()) ||
          axis >= static_cast<std::int64_t>(firstType.shape.size()))
        throw std::invalid_argument("tensor region reduction axis is outside its input rank");
      if (axis < 0)
        axis += static_cast<std::int64_t>(firstType.shape.size());
      resultType.shape = firstType.shape;
      if (keepDims)
        resultType.shape[static_cast<std::size_t>(axis)] = 1;
      else
        resultType.shape.erase(resultType.shape.begin() + axis);
    } else if (transpose) {
      if (attribute.size() != firstType.shape.size())
        throw std::invalid_argument("tensor region transpose needs a permutation rank match");
      std::vector<char> seen(firstType.shape.size(), 0);
      Shape output(firstType.shape.size());
      for (std::size_t index = 0; index < attribute.size(); ++index) {
        const auto sourceAxis = attribute[index];
        if (sourceAxis >= firstType.shape.size() || seen[sourceAxis])
          throw std::invalid_argument(
              "tensor region transpose axes must form a permutation"
          );
        seen[sourceAxis] = 1;
        output[index] = firstType.shape[sourceAxis];
      }
      if (keepDims)
        throw std::invalid_argument("tensor region transpose does not keep dimensions");
      resultType.shape = std::move(output);
    } else if (expand) {
      if (!attribute.empty())
        throw std::invalid_argument("tensor region expand does not accept a shape attribute");
      const auto outputRank = static_cast<std::int64_t>(firstType.shape.size()) + 1;
      if (axis < -outputRank || axis >= outputRank)
        throw std::invalid_argument("tensor region expand axis is outside the output rank");
      if (axis < 0)
        axis += outputRank;
      if (keepDims)
        throw std::invalid_argument("tensor region expand does not keep dimensions");
      resultType.shape = firstType.shape;
      resultType.shape.insert(
          resultType.shape.begin() + static_cast<std::ptrdiff_t>(axis), 1
      );
    } else {
      if (attribute.empty() ||
          std::any_of(attribute.begin(), attribute.end(), [](std::uint32_t size) {
            return size == 0;
          }))
        throw std::invalid_argument("tensor region broadcast needs a positive target shape");
      if (attribute.size() < firstType.shape.size())
        throw std::invalid_argument("tensor region broadcast cannot reduce rank");
      for (std::size_t offset = 0; offset < firstType.shape.size(); ++offset) {
        const auto source = firstType.shape[firstType.shape.size() - 1 - offset];
        const auto target = attribute[attribute.size() - 1 - offset];
        if (source != 1 && source != target)
          throw std::invalid_argument("tensor region broadcast shapes are incompatible");
      }
      resultType.shape = attribute;
    }
    const auto resultIndex = static_cast<std::int32_t>(values_.size());
    const auto nodeIndex = static_cast<std::int32_t>(nodes_.size());
    values_.push_back(Value{std::move(name), std::move(resultType), nodeIndex});
    nodes_.push_back(Node{
        op, std::move(operands), axis, keepDims, std::move(attribute), resultIndex});
  }

  void TensorRegion::setResults(std::vector<std::string> resultNames) {
    if (resultNames.empty())
      throw std::invalid_argument("tensor region needs at least one result");
    std::vector<std::int32_t> results;
    std::unordered_set<std::int32_t> seen;
    for (const auto &name: resultNames) {
      const auto value = requireValue(name);
      if (!seen.insert(value).second)
        throw std::invalid_argument("tensor region result is repeated");
      results.push_back(value);
    }
    results_ = std::move(results);
  }

  std::string TensorRegion::renderNode(const Node &node) const {
    std::ostringstream out;
    const auto &result = values_[static_cast<std::size_t>(node.result)];
    out << "  " << result.name << " = tensor " << tensorOpName(node.op) << ' '
        << values_[static_cast<std::size_t>(node.operands.front())].name;
    if (node.op == TensorOp::Add || node.op == TensorOp::Sub || node.op == TensorOp::Mul ||
        node.op == TensorOp::Min || node.op == TensorOp::Max || node.op == TensorOp::Dot) {
      out << ", " << values_[static_cast<std::size_t>(node.operands[1])].name;
    } else if (node.op == TensorOp::Sum || node.op == TensorOp::ReduceMin ||
               node.op == TensorOp::ReduceMax) {
      out << ", " << node.axis << ", " << (node.keepDims ? "keep" : "drop");
    } else if (node.op == TensorOp::Transpose) {
      out << ", " << shapeText(node.attribute);
    } else if (node.op == TensorOp::Expand) {
      out << ", " << node.axis;
    } else {
      out << ", " << shapeText(node.attribute);
    }
    out << ';';
    return out.str();
  }

  std::vector<char> TensorRegion::liveNodes() const {
    if (results_.empty())
      throw std::logic_error("tensor region results have not been set");
    std::vector<char> live(nodes_.size(), 0);
    std::function<void(std::int32_t)> mark = [&](std::int32_t valueIndex) {
      const auto producer = values_[static_cast<std::size_t>(valueIndex)].producer;
      if (producer < 0 || live[static_cast<std::size_t>(producer)])
        return;
      live[static_cast<std::size_t>(producer)] = 1;
      for (const auto operand: nodes_[static_cast<std::size_t>(producer)].operands)
        mark(operand);
    };
    for (const auto result: results_)
      mark(result);
    return live;
  }

  std::vector<std::string> TensorRegion::renderDeclarations() const {
    const auto live = liveNodes();
    std::vector<std::string> declarations;
    for (std::size_t index = 0; index < nodes_.size(); ++index) {
      if (!live[index])
        continue;
      const auto &value = values_[static_cast<std::size_t>(nodes_[index].result)];
      declarations.push_back(
          "  let mut " + value.name + ": " + tensorTypeText(value.type) + " = undef;"
      );
    }
    return declarations;
  }

  std::vector<std::string> TensorRegion::renderDeclarations(
      const std::vector<std::pair<std::string, std::string>> &updates,
      const std::vector<std::string> &reservedNames
  ) const {
    auto declarations = renderDeclarations();
    const auto validated = validateStateUpdates(updates);
    for (const auto &[source, name]: stateSnapshots(validated, reservedNames)) {
      declarations.push_back(
          "  let mut " + name + ": " +
          tensorTypeText(values_[static_cast<std::size_t>(source)].type) + " = undef;"
      );
    }
    return declarations;
  }

  // Interval reasoning over magnitudes only: every operation the region can
  // hold is monotone in its operands' magnitudes, so one forward pass over the
  // live nodes bounds every value from its inputs' bounds.
  std::map<std::string, double>
  TensorRegion::magnitudeBounds(const std::map<std::string, double> &inputs) const {
    std::vector<double> bound(values_.size(), 0.0);
    for (std::size_t index = 0; index < values_.size(); ++index) {
      if (values_[index].producer >= 0)
        continue;
      const auto found = inputs.find(values_[index].name);
      if (found == inputs.end())
        throw std::invalid_argument(
            "tensor region: no magnitude bound for input " + values_[index].name
        );
      bound[index] = found->second;
    }
    const auto live = liveNodes();
    std::map<std::string, double> bounds(inputs.begin(), inputs.end());
    for (std::size_t position = 0; position < nodes_.size(); ++position) {
      if (!live[position])
        continue;
      const auto &node = nodes_[position];
      const auto operand = [&](std::size_t index) {
        return bound[static_cast<std::size_t>(node.operands.at(index))];
      };
      double result = 0.0;
      switch (node.op) {
        case TensorOp::Add:
        case TensorOp::Sub:
          result = operand(0) + operand(1);
          break;
        case TensorOp::Mul:
          // A zero bound means exactly zero, so the product is zero even when
          // the other operand is unbounded; this also keeps 0 * inf from NaN.
          result = operand(0) == 0.0 || operand(1) == 0.0 ? 0.0 : operand(0) * operand(1);
          break;
        case TensorOp::Min:
        case TensorOp::Max:
          result = std::max(operand(0), operand(1));
          break;
        case TensorOp::Dot: {
          // K products, each bounded like `mul`.
          const auto &shape = values_[static_cast<std::size_t>(node.operands.at(0))].type.shape;
          result = operand(0) == 0.0 || operand(1) == 0.0
                       ? 0.0
                       : operand(0) * operand(1) * static_cast<double>(shape.at(1));
          break;
        }
        case TensorOp::Sum: {
          const auto &shape = values_[static_cast<std::size_t>(node.operands.at(0))].type.shape;
          const auto rank = static_cast<std::int64_t>(shape.size());
          const auto axis = node.axis < 0 ? node.axis + rank : node.axis;
          result = operand(0) * static_cast<double>(shape.at(static_cast<std::size_t>(axis)));
          break;
        }
        case TensorOp::ReduceMin:
        case TensorOp::ReduceMax:
        case TensorOp::Transpose:
        case TensorOp::Expand:
        case TensorOp::Broadcast:
          result = operand(0);
          break;
        default:
          throw std::invalid_argument(
              "tensor region: no magnitude rule for " +
              values_[static_cast<std::size_t>(node.result)].name
          );
      }
      bound[static_cast<std::size_t>(node.result)] = result;
      bounds[values_[static_cast<std::size_t>(node.result)].name] = result;
    }
    return bounds;
  }

  std::vector<std::string> TensorRegion::orderSensitiveValues() const {
    const auto live = liveNodes();
    std::vector<std::string> names;
    for (std::size_t position = 0; position < nodes_.size(); ++position)
      if (live[position] &&
          (nodes_[position].op == TensorOp::Sum || nodes_[position].op == TensorOp::Dot))
        names.push_back(values_[static_cast<std::size_t>(nodes_[position].result)].name);
    return names;
  }

  std::vector<std::string> TensorRegion::renderInstructions() const {
    const auto live = liveNodes();
    std::vector<std::string> instructions;
    for (std::size_t node = 0; node < nodes_.size(); ++node)
      if (live[node])
        instructions.push_back(renderNode(nodes_[node]));
    return instructions;
  }

  std::vector<TensorRegion::IndexedUpdate>
  TensorRegion::validateStateUpdates(const std::vector<std::pair<std::string, std::string>> &updates
  ) const {
    if (results_.empty())
      throw std::logic_error("tensor region results have not been set");
    if (updates.size() != results_.size())
      throw std::invalid_argument("state update count differs from region result count");
    std::unordered_set<std::int32_t> destinations;
    std::unordered_set<std::int32_t> results(results_.begin(), results_.end());

    std::vector<IndexedUpdate> validated;
    validated.reserve(updates.size());
    for (const auto &[stateName, resultName]: updates) {
      const auto state = requireValue(stateName);
      const auto result = requireValue(resultName);
      if (!destinations.insert(state).second)
        throw std::invalid_argument("tensor region state is updated more than once");
      if (values_[static_cast<std::size_t>(state)].producer >= 0)
        throw std::invalid_argument("state update destination must be a region input");
      if (!results.contains(result))
        throw std::invalid_argument("state update source is not a region result");
      if (values_[static_cast<std::size_t>(state)].type.shape !=
              values_[static_cast<std::size_t>(result)].type.shape ||
          values_[static_cast<std::size_t>(state)].type.element !=
              values_[static_cast<std::size_t>(result)].type.element)
        throw std::invalid_argument("state update types do not match");
      validated.emplace_back(state, result);
    }
    return validated;
  }

  std::vector<TensorRegion::Snapshot> TensorRegion::stateSnapshots(
      const std::vector<IndexedUpdate> &updates, const std::vector<std::string> &reservedNames
  ) const {
    std::unordered_set<std::int32_t> destinations;
    for (const auto &[destination, source]: updates) {
      (void) source;
      destinations.insert(destination);
    }

    std::unordered_set<std::int32_t> snappedSources;
    std::unordered_set<std::string> usedNames;
    for (const auto &value: values_)
      usedNames.insert(value.name);
    usedNames.insert(reservedNames.begin(), reservedNames.end());

    std::vector<Snapshot> snapshots;
    snapshots.reserve(updates.size());
    std::size_t nextSnapshot = 0;
    for (const auto &[destination, source]: updates) {
      if (destination == source || values_[static_cast<std::size_t>(source)].producer >= 0 ||
          !destinations.contains(source) || !snappedSources.insert(source).second)
        continue;

      std::string name;
      do {
        name = "%__region_saved_" + std::to_string(nextSnapshot++);
      } while (!usedNames.insert(name).second);
      snapshots.emplace_back(source, std::move(name));
    }
    return snapshots;
  }

  std::vector<std::string> TensorRegion::renderStateUpdates(
      const std::vector<std::pair<std::string, std::string>> &updates,
      const std::vector<std::string> &reservedNames
  ) const {
    const auto validated = validateStateUpdates(updates);
    const auto snapshots = stateSnapshots(validated, reservedNames);
    std::unordered_map<std::int32_t, std::string> snapshotNames;
    std::vector<std::string> assignments;
    assignments.reserve(snapshots.size() + validated.size());
    for (const auto &[sourceIndex, name]: snapshots) {
      const auto &source = values_[static_cast<std::size_t>(sourceIndex)];
      snapshotNames.emplace(sourceIndex, name);
      assignments.push_back("  " + name + " = " + source.name + ";");
    }

    for (const auto &[destinationIndex, sourceIndex]: validated) {
      if (destinationIndex == sourceIndex)
        continue;
      const auto &destination = values_[static_cast<std::size_t>(destinationIndex)];
      const auto &source = values_[static_cast<std::size_t>(sourceIndex)];
      const auto saved = snapshotNames.find(sourceIndex);
      assignments.push_back(
          "  " + destination.name + " = " +
          (saved == snapshotNames.end() ? source.name : saved->second) + ";"
      );
    }
    return assignments;
  }

  std::vector<std::string> TensorRegion::renderBody(
      const std::vector<std::pair<std::string, std::string>> &updates,
      const std::vector<std::string> &reservedNames
  ) const {
    auto body = renderInstructions();
    auto assignments = renderStateUpdates(updates, reservedNames);
    body.insert(body.end(), assignments.begin(), assignments.end());
    return body;
  }

} // namespace refractir::reify
