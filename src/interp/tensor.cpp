#include "interp/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include "analysis/tensor.hpp"
#include "analysis/type_utils.hpp"
#include "error.hpp"

namespace refractir {
  namespace {

    using Coords = std::vector<std::uint32_t>;

    std::uint32_t linearIndex(const Coords &coords, const Shape &shape) {
      if (coords.size() != shape.size())
        throw std::runtime_error("tensor coordinate rank mismatch");
      std::uint64_t index = 0;
      for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        if (coords[axis] >= shape[axis])
          throw std::runtime_error("tensor coordinate is out of bounds");
        index = index * shape[axis] + coords[axis];
      }
      return static_cast<std::uint32_t>(index);
    }

    Coords coordinates(std::uint32_t index, const Shape &shape) {
      Coords result(shape.size());
      for (std::size_t i = shape.size(); i > 0; --i) {
        const auto dim = shape[i - 1];
        result[i - 1] = index % dim;
        index /= dim;
      }
      return result;
    }

    RuntimeValue scalarAt(const RuntimeValue &value, const Coords &coords, const Shape &shape) {
      if (shape.empty())
        return value;
      if (value.kind != RuntimeValue::Kind::Tensor || value.tensorShape != shape ||
          value.arrayVal.size() != tensorElementCount(shape))
        throw std::runtime_error("tensor runtime shape does not match checked shape");
      return value.arrayVal.at(linearIndex(coords, shape));
    }

    RuntimeValue
    projectedAt(const RuntimeValue &value, const Shape &sourceShape, const Coords &outputCoords) {
      if (sourceShape.empty())
        return value;
      if (sourceShape.size() > outputCoords.size())
        throw std::runtime_error("tensor broadcast rank mismatch");
      Coords sourceCoords(sourceShape.size());
      const auto padding = outputCoords.size() - sourceShape.size();
      for (std::size_t i = 0; i < sourceShape.size(); ++i)
        sourceCoords[i] = sourceShape[i] == 1 ? 0 : outputCoords[i + padding];
      return scalarAt(value, sourceCoords, sourceShape);
    }

    RuntimeValue requireDefined(RuntimeValue value) {
      if (value.kind == RuntimeValue::Kind::Undef)
        throw UndefinedBehaviorError("UB: Reading undef tensor element");
      return value;
    }

    RuntimeValue normalizeScalar(RuntimeValue value, TensorElement element) {
      value = requireDefined(std::move(value));
      if (element == TensorElement::I32) {
        if (value.kind != RuntimeValue::Kind::Int || value.intVal < INT32_MIN ||
            value.intVal > INT32_MAX)
          throw std::runtime_error("tensor i32 operand has the wrong runtime type");
        value.bits = 32;
      } else {
        if (value.kind != RuntimeValue::Kind::Float || value.bits != 32 ||
            !std::isfinite(value.floatVal))
          throw std::runtime_error("tensor f32 operand is not a finite f32 value");
        value.floatVal = static_cast<double>(static_cast<float>(value.floatVal));
        if (!std::isfinite(value.floatVal))
          throw UndefinedBehaviorError("UB: non-finite tensor f32 value");
      }
      return value;
    }

    RuntimeValue tensorOrScalar(Shape shape, std::vector<RuntimeValue> leaves) {
      if (shape.empty()) {
        if (leaves.size() != 1)
          throw std::runtime_error("scalar tensor operation produced an invalid element count");
        return leaves.front();
      }
      RuntimeValue result;
      result.kind = RuntimeValue::Kind::Tensor;
      result.tensorShape = std::move(shape);
      result.arrayVal = std::move(leaves);
      result.bits = 32;
      return result;
    }

    RuntimeValue integerBinary(TensorOp op, RuntimeValue left, RuntimeValue right) {
      left = normalizeScalar(std::move(left), TensorElement::I32);
      right = normalizeScalar(std::move(right), TensorElement::I32);
      RuntimeValue result;
      result.kind = RuntimeValue::Kind::Int;
      result.bits = 32;
      const auto a = left.intVal;
      const auto b = right.intVal;
      if (op == TensorOp::Min)
        result.intVal = std::min(a, b);
      else if (op == TensorOp::Max)
        result.intVal = std::max(a, b);
      else if (op == TensorOp::And)
        result.intVal = static_cast<std::int32_t>(
            static_cast<std::uint32_t>(a) & static_cast<std::uint32_t>(b)
        );
      else if (op == TensorOp::Or)
        result.intVal = static_cast<std::int32_t>(
            static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b)
        );
      else if (op == TensorOp::Xor)
        result.intVal = static_cast<std::int32_t>(
            static_cast<std::uint32_t>(a) ^ static_cast<std::uint32_t>(b)
        );
      else {
        __int128 wide = 0;
        if (op == TensorOp::Add)
          wide = static_cast<__int128>(a) + b;
        else if (op == TensorOp::Sub)
          wide = static_cast<__int128>(a) - b;
        else if (op == TensorOp::Mul)
          wide = static_cast<__int128>(a) * b;
        else if (op == TensorOp::Rem)
          throw std::runtime_error("unsupported_exact_tensor_operation: rem requires f32");
        else
          throw std::runtime_error("unsupported exact tensor binary operator");
        if (wide < INT32_MIN || wide > INT32_MAX)
          throw UndefinedBehaviorError("UB: signed i32 overflow in tensor elementwise operation");
        result.intVal = static_cast<std::int32_t>(wide);
      }
      return result;
    }

    std::vector<RuntimeValue> readGroup(
        const RuntimeValue &input, const Shape &inputShape, const Coords &outputCoords,
        std::uint32_t axis, bool keep
    ) {
      Coords inputCoords;
      if (keep) {
        inputCoords = outputCoords;
      } else {
        inputCoords = outputCoords;
        inputCoords.insert(inputCoords.begin() + axis, 0);
      }
      std::vector<RuntimeValue> group;
      group.reserve(inputShape[axis]);
      for (std::uint32_t i = 0; i < inputShape[axis]; ++i) {
        inputCoords[axis] = i;
        group.push_back(scalarAt(input, inputCoords, inputShape));
      }
      return group;
    }

    RuntimeValue
    integerReduce(TensorOp op, std::vector<RuntimeValue> group, TensorElement element) {
      if (group.empty())
        throw std::runtime_error("tensor reduction group is empty");
      for (auto &value: group)
        value = normalizeScalar(std::move(value), element);
      RuntimeValue result = group.front();
      if (op == TensorOp::Sum) {
        std::int64_t positive = 0;
        std::int64_t negative = 0;
        for (const auto &value: group) {
          if (value.intVal > 0)
            positive += value.intVal;
          else
            negative += value.intVal;
        }
        if (positive > INT32_MAX || negative < INT32_MIN)
          throw UndefinedBehaviorError("UB: tensor sum partial overflow");
        result.intVal = positive + negative;
      } else if (op == TensorOp::ReduceMin) {
        for (const auto &value: group)
          result.intVal = std::min(result.intVal, value.intVal);
      } else if (op == TensorOp::ReduceMax) {
        for (const auto &value: group)
          result.intVal = std::max(result.intVal, value.intVal);
      } else {
        throw std::runtime_error("unsupported exact tensor reduction operator");
      }
      return result;
    }

  } // namespace

  TensorTypeInfo tensorTypeInfo(const TypePtr &type) {
    const TypePtr *element = &type;
    Shape shape;
    if (const auto tensor = TypeUtils::asTensor(type)) {
      shape = tensor->shape;
      element = &tensor->elem;
    }
    if (TypeUtils::getIntBitWidth(*element) == 32)
      return TensorTypeInfo{std::move(shape), TensorElement::I32};
    if (TypeUtils::getFloatBitWidth(*element) == 32)
      return TensorTypeInfo{std::move(shape), TensorElement::F32};
    throw std::runtime_error("typed tensor boundary supports only i32 and f32 values");
  }

  RuntimeValue evalTensorExact(
      const TensorAtom &atom, const std::vector<RuntimeValue> &operands,
      const TensorTypeInfo &resultType
  ) {
    if (atom.op == TensorOp::Iota) {
      const auto axis =
          normalizeTensorAxis(atom.axis, static_cast<std::uint32_t>(atom.attribute.size()));
      const auto count = tensorElementCount(atom.attribute);
      std::vector<RuntimeValue> values;
      values.reserve(count);
      for (std::uint32_t i = 0; i < count; ++i) {
        RuntimeValue value;
        value.kind = RuntimeValue::Kind::Int;
        value.bits = 32;
        value.intVal = coordinates(i, atom.attribute)[axis];
        values.push_back(std::move(value));
      }
      return tensorOrScalar(atom.attribute, std::move(values));
    }

    if (operands.size() != atom.operands.size())
      throw std::runtime_error("tensor operand count mismatch during evaluation");
    if (atom.op == TensorOp::Extract) {
      const auto &input = operands.front();
      if (input.kind != RuntimeValue::Kind::Tensor)
        throw std::runtime_error("tensor extract input is not a tensor");
      return normalizeScalar(
          scalarAt(input, atom.attribute, input.tensorShape), resultType.element
      );
    }

    if (atom.op == TensorOp::Expand) {
      const auto &input = operands.front();
      const Shape inputShape =
          input.kind == RuntimeValue::Kind::Tensor ? input.tensorShape : Shape{};
      const auto axis =
          normalizeTensorAxis(atom.axis, static_cast<std::uint32_t>(inputShape.size() + 1));
      const auto count = tensorElementCount(resultType.shape);
      std::vector<RuntimeValue> output;
      output.reserve(count);
      for (std::uint32_t i = 0; i < count; ++i) {
        auto outputCoords = coordinates(i, resultType.shape);
        outputCoords.erase(outputCoords.begin() + axis);
        output.push_back(
            normalizeScalar(scalarAt(input, outputCoords, inputShape), resultType.element)
        );
      }
      return tensorOrScalar(resultType.shape, std::move(output));
    }

    if (atom.op == TensorOp::Broadcast) {
      const auto &input = operands.front();
      const Shape inputShape =
          input.kind == RuntimeValue::Kind::Tensor ? input.tensorShape : Shape{};
      std::vector<RuntimeValue> output;
      output.reserve(tensorElementCount(resultType.shape));
      for (std::uint32_t i = 0; i < tensorElementCount(resultType.shape); ++i)
        output.push_back(normalizeScalar(
            projectedAt(input, inputShape, coordinates(i, resultType.shape)), resultType.element
        ));
      return tensorOrScalar(resultType.shape, std::move(output));
    }

    if (atom.op == TensorOp::Transpose) {
      const auto &input = operands.front();
      if (input.kind != RuntimeValue::Kind::Tensor)
        throw std::runtime_error("tensor transpose input is not a tensor");
      std::vector<RuntimeValue> output;
      output.reserve(tensorElementCount(resultType.shape));
      for (std::uint32_t i = 0; i < tensorElementCount(resultType.shape); ++i) {
        const auto outputCoords = coordinates(i, resultType.shape);
        Coords inputCoords(input.tensorShape.size());
        for (std::size_t axis = 0; axis < atom.attribute.size(); ++axis)
          inputCoords[atom.attribute[axis]] = outputCoords[axis];
        output.push_back(
            normalizeScalar(scalarAt(input, inputCoords, input.tensorShape), resultType.element)
        );
      }
      return tensorOrScalar(resultType.shape, std::move(output));
    }

    if (atom.op == TensorOp::Sum || atom.op == TensorOp::ReduceMin ||
        atom.op == TensorOp::ReduceMax) {
      if (resultType.element != TensorElement::I32)
        throw std::runtime_error("numerical tensor reductions require the reference evaluator");
      const auto &input = operands.front();
      if (input.kind != RuntimeValue::Kind::Tensor)
        throw std::runtime_error("tensor reduction input is not a tensor");
      const auto axis =
          normalizeTensorAxis(atom.axis, static_cast<std::uint32_t>(input.tensorShape.size()));
      std::vector<RuntimeValue> output;
      output.reserve(tensorElementCount(resultType.shape));
      const auto outCount = resultType.shape.empty() ? 1 : tensorElementCount(resultType.shape);
      for (std::uint32_t i = 0; i < outCount; ++i) {
        const auto outputCoords =
            resultType.shape.empty() ? Coords{} : coordinates(i, resultType.shape);
        output.push_back(integerReduce(
            atom.op, readGroup(input, input.tensorShape, outputCoords, axis, atom.keepDims),
            resultType.element
        ));
      }
      return tensorOrScalar(resultType.shape, std::move(output));
    }


    if (atom.op == TensorOp::Window) {
      if (operands.size() != 2)
        throw std::runtime_error("tensor window expects an index operand");
      const auto &source = operands[0];
      const auto &index = operands[1];
      if (source.kind != RuntimeValue::Kind::Tensor)
        throw std::runtime_error("tensor window source is not a tensor");
      if (index.kind != RuntimeValue::Kind::Int || index.bits != 32)
        throw std::runtime_error("tensor window index is not i32");
      if (!atom.fill)
        throw std::runtime_error("tensor window has no fill literal");
      RuntimeValue fill;
      if (const auto *literal = std::get_if<IntLit>(&*atom.fill)) {
        fill.kind = RuntimeValue::Kind::Int;
        fill.bits = 32;
        fill.intVal = literal->value;
      } else if (const auto *literal = std::get_if<FloatLit>(&*atom.fill)) {
        fill.kind = RuntimeValue::Kind::Float;
        fill.bits = 32;
        fill.floatVal = literal->value;
      } else {
        throw std::runtime_error("tensor window fill is not a literal");
      }
      fill = normalizeScalar(std::move(fill), resultType.element);
      const auto rows = atom.attribute.size() == 3 ? atom.attribute[0] : 0;
      const auto stride = atom.attribute.size() == 3 ? atom.attribute[1] : 0;
      const auto active = atom.attribute.size() == 3 ? atom.attribute[2] : 0;
      if (resultType.shape.empty() || resultType.shape.front() != rows)
        throw std::runtime_error("tensor window result shape does not match its rows");
      // A position is active when it lies in [0, active); every other
      // position, below zero included, reads the fill.
      const auto base = static_cast<std::int64_t>(index.intVal) * static_cast<std::int64_t>(stride);
      std::vector<RuntimeValue> output;
      output.reserve(tensorElementCount(resultType.shape));
      for (std::uint32_t i = 0; i < tensorElementCount(resultType.shape); ++i) {
        const auto coords = coordinates(i, resultType.shape);
        const auto address = base + static_cast<std::int64_t>(coords.front());
        if (address >= 0 && address < static_cast<std::int64_t>(active)) {
          Coords sourceCoords = coords;
          sourceCoords.front() = static_cast<std::uint32_t>(address);
          output.push_back(
              normalizeScalar(scalarAt(source, sourceCoords, source.tensorShape), resultType.element)
          );
        } else {
          output.push_back(fill);
        }
      }
      return tensorOrScalar(resultType.shape, std::move(output));
    }

    if (atom.op == TensorOp::Cumsum) {
      if (resultType.element != TensorElement::I32)
        throw std::runtime_error("numerical tensor cumsum requires the reference evaluator");
      const auto &input = operands.at(0);
      const auto axis =
          normalizeTensorAxis(atom.axis, static_cast<std::uint32_t>(input.tensorShape.size()));
      std::vector<RuntimeValue> output;
      output.reserve(tensorElementCount(resultType.shape));
      for (std::uint32_t i = 0; i < tensorElementCount(resultType.shape); ++i) {
        auto coords = coordinates(i, resultType.shape);
        const auto last = coords[axis];
        std::int64_t total = 0;
        for (std::uint32_t k = 0; k <= last; ++k) {
          coords[axis] = k;
          total += normalizeScalar(scalarAt(input, coords, input.tensorShape), TensorElement::I32)
                       .intVal;
          if (total < INT32_MIN || total > INT32_MAX)
            throw UndefinedBehaviorError("UB: tensor cumsum prefix overflows i32");
        }
        RuntimeValue lane;
        lane.kind = RuntimeValue::Kind::Int;
        lane.bits = 32;
        lane.intVal = total;
        output.push_back(lane);
      }
      return tensorOrScalar(resultType.shape, std::move(output));
    }

    if (isTensorWhere(atom.op)) {
      if (resultType.element != TensorElement::I32)
        throw std::runtime_error("numerical tensor where requires the reference evaluator");
      std::vector<RuntimeValue> output;
      output.reserve(tensorElementCount(resultType.shape));
      for (std::uint32_t i = 0; i < tensorElementCount(resultType.shape); ++i) {
        const auto coords = coordinates(i, resultType.shape);
        std::vector<RuntimeValue> lanes;
        for (const auto &operand: operands) {
          const Shape shape =
              operand.kind == RuntimeValue::Kind::Tensor ? operand.tensorShape : Shape{};
          lanes.push_back(normalizeScalar(projectedAt(operand, shape, coords), TensorElement::I32));
        }
        const auto a = lanes[0].intVal;
        const auto b = lanes[1].intVal;
        const bool holds = atom.op == TensorOp::WhereLt   ? a < b
                           : atom.op == TensorOp::WhereLe ? a <= b
                                                          : a == b;
        output.push_back(holds ? lanes[2] : lanes[3]);
      }
      return tensorOrScalar(resultType.shape, std::move(output));
    }

    if (atom.op == TensorOp::Dot) {
      if (resultType.element != TensorElement::I32)
        throw std::runtime_error("numerical tensor dot requires the reference evaluator");
      const auto &left = operands.at(0);
      const auto &right = operands.at(1);
      const auto inner = left.tensorShape.at(1);
      // Triton multiplies integer tiles as int8; a wider operand is undefined
      // rather than silently truncated, so the program states the range.
      auto int8Operand = [](const RuntimeValue &value) {
        const auto lane = normalizeScalar(value, TensorElement::I32).intVal;
        if (lane < INT8_MIN || lane > INT8_MAX)
          throw UndefinedBehaviorError("UB: tensor dot operand outside int8");
        return static_cast<std::int64_t>(lane);
      };
      std::vector<RuntimeValue> output;
      output.reserve(tensorElementCount(resultType.shape));
      for (std::uint32_t i = 0; i < tensorElementCount(resultType.shape); ++i) {
        const auto coords = coordinates(i, resultType.shape);
        std::int64_t total = 0;
        for (std::uint32_t k = 0; k < inner; ++k)
          total += int8Operand(scalarAt(left, {coords[0], k}, left.tensorShape)) *
                   int8Operand(scalarAt(right, {k, coords[1]}, right.tensorShape));
        RuntimeValue lane;
        lane.kind = RuntimeValue::Kind::Int;
        lane.bits = 32;
        lane.intVal = total;
        output.push_back(lane);
      }
      return tensorOrScalar(resultType.shape, std::move(output));
    }

    if (atom.op == TensorOp::Add || atom.op == TensorOp::Sub || atom.op == TensorOp::Mul ||
        atom.op == TensorOp::Min || atom.op == TensorOp::Max || atom.op == TensorOp::And ||
        atom.op == TensorOp::Or || atom.op == TensorOp::Xor) {
      const auto sourceShape = [&](const RuntimeValue &value) -> Shape {
        if (value.kind == RuntimeValue::Kind::Tensor)
          return value.tensorShape;
        return {};
      };
      const Shape leftShape = sourceShape(operands[0]);
      const Shape rightShape = sourceShape(operands[1]);
      if (tensorBroadcastShape(leftShape, rightShape) != resultType.shape)
        throw std::runtime_error("tensor binary result shape does not match checked shape");
      std::vector<RuntimeValue> output;
      const auto count = resultType.shape.empty() ? 1 : tensorElementCount(resultType.shape);
      output.reserve(count);
      for (std::uint32_t i = 0; i < count; ++i) {
        const auto outputCoords =
            resultType.shape.empty() ? Coords{} : coordinates(i, resultType.shape);
        auto left = projectedAt(operands[0], leftShape, outputCoords);
        auto right = projectedAt(operands[1], rightShape, outputCoords);
        if (resultType.element == TensorElement::I32)
          output.push_back(integerBinary(atom.op, std::move(left), std::move(right)));
        else
          throw std::runtime_error("numerical tensor arithmetic requires the reference evaluator");
      }
      return tensorOrScalar(resultType.shape, std::move(output));
    }

    throw std::runtime_error("numerical tensor operator requires the reference evaluator");
  }

} // namespace refractir
