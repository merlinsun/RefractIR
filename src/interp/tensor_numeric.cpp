#include "interp/tensor_numeric.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(REFRACTIR_TENSOR_REF)
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_bin_float/transcendental.hpp>
#endif

#include "analysis/tensor.hpp"
#include "error.hpp"

namespace refractir {
  namespace {

    thread_local unsigned referenceBits = 128;
    thread_local TensorTraceHook tensorTraceHook;

#if defined(REFRACTIR_TENSOR_REF)
    using Coordinates = std::vector<std::uint32_t>;

    template<unsigned Bits>
    using RefFloat = boost::multiprecision::number<
        boost::multiprecision::cpp_bin_float<Bits, boost::multiprecision::digit_base_2>>;

    std::uint32_t linearIndex(const Coordinates &coords, const Shape &shape) {
      if (coords.size() != shape.size())
        throw std::runtime_error("tensor reference coordinate rank mismatch");
      std::uint64_t index = 0;
      for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        if (coords[axis] >= shape[axis])
          throw std::runtime_error("tensor reference coordinate is out of bounds");
        index = index * shape[axis] + coords[axis];
      }
      return static_cast<std::uint32_t>(index);
    }

    Coordinates coordinates(std::uint32_t index, const Shape &shape) {
      Coordinates result(shape.size());
      for (std::size_t i = shape.size(); i > 0; --i) {
        result[i - 1] = index % shape[i - 1];
        index /= shape[i - 1];
      }
      return result;
    }

    RuntimeValue
    scalarAt(const RuntimeValue &value, const Coordinates &coords, const Shape &shape) {
      if (shape.empty())
        return value;
      if (value.kind != RuntimeValue::Kind::Tensor || value.tensorShape != shape ||
          value.arrayVal.size() != tensorElementCount(shape))
        throw std::runtime_error("tensor reference value does not match its checked shape");
      return value.arrayVal.at(linearIndex(coords, shape));
    }

    RuntimeValue projectedAt(
        const RuntimeValue &value, const Shape &sourceShape, const Coordinates &outputCoords
    ) {
      if (sourceShape.empty())
        return value;
      if (sourceShape.size() > outputCoords.size())
        throw std::runtime_error("tensor reference broadcast rank mismatch");
      Coordinates sourceCoords(sourceShape.size());
      const auto padding = outputCoords.size() - sourceShape.size();
      for (std::size_t axis = 0; axis < sourceShape.size(); ++axis)
        sourceCoords[axis] = sourceShape[axis] == 1 ? 0 : outputCoords[axis + padding];
      return scalarAt(value, sourceCoords, sourceShape);
    }

    RuntimeValue tensorOrScalar(Shape shape, std::vector<RuntimeValue> leaves) {
      if (shape.empty()) {
        if (leaves.size() != 1)
          throw std::runtime_error("tensor reference produced an invalid scalar leaf count");
        return std::move(leaves.front());
      }
      RuntimeValue result;
      result.kind = RuntimeValue::Kind::Tensor;
      result.tensorShape = std::move(shape);
      result.arrayVal = std::move(leaves);
      result.bits = 32;
      return result;
    }

    template<unsigned Bits>
    RefFloat<Bits> readFloat(const RuntimeValue &value) {
      if (value.kind == RuntimeValue::Kind::Undef)
        throw UndefinedBehaviorError("UB: Reading undef tensor element");
      if (value.kind != RuntimeValue::Kind::Float || value.bits != 32 ||
          !std::isfinite(value.floatVal) || !std::isfinite(static_cast<float>(value.floatVal)))
        throw std::runtime_error("tensor numerical reference operand is not a finite f32 value");
      return RefFloat<Bits>(static_cast<float>(value.floatVal));
    }

    template<unsigned Bits>
    RuntimeValue roundToF32(const RefFloat<Bits> &wide) {
      const float rounded = wide.template convert_to<float>();
      if (!std::isfinite(rounded))
        throw UndefinedBehaviorError("UB: non-finite tensor f32 result");
      RuntimeValue result;
      result.kind = RuntimeValue::Kind::Float;
      result.bits = 32;
      result.floatVal = static_cast<double>(rounded);
      return result;
    }

    template<unsigned Bits>
    std::vector<RuntimeValue> readGroup(
        const RuntimeValue &input, const Shape &inputShape, const Coordinates &outputCoords,
        std::uint32_t axis, bool keep
    ) {
      Coordinates inputCoords = outputCoords;
      if (!keep)
        inputCoords.insert(inputCoords.begin() + axis, 0);
      std::vector<RuntimeValue> group;
      group.reserve(inputShape[axis]);
      for (std::uint32_t i = 0; i < inputShape[axis]; ++i) {
        inputCoords[axis] = i;
        group.push_back(scalarAt(input, inputCoords, inputShape));
      }
      return group;
    }

    template<unsigned Bits>
    bool isNegativeZero(const RefFloat<Bits> &value) {
      return value == 0 && std::signbit(value.template convert_to<double>());
    }

    /// IEEE 754 minNum/maxNum, the tie-break the scalar intrinsics and the
    /// specification require: a -0.0 operand forces a -0.0 minimum and a +0.0
    /// operand forces a +0.0 maximum, so the result does not depend on the
    /// fold order.
    template<unsigned Bits>
    RefFloat<Bits> minNum(const RefFloat<Bits> &left, const RefFloat<Bits> &right) {
      if (left < right)
        return left;
      if (right < left)
        return right;
      if (left == 0)
        return (isNegativeZero(left) || isNegativeZero(right))
                   ? RefFloat<Bits>(std::copysign(0.0, -1.0))
                   : RefFloat<Bits>(0.0);
      return left;
    }

    template<unsigned Bits>
    RefFloat<Bits> maxNum(const RefFloat<Bits> &left, const RefFloat<Bits> &right) {
      if (left > right)
        return left;
      if (right > left)
        return right;
      if (left == 0)
        return (isNegativeZero(left) && isNegativeZero(right))
                   ? RefFloat<Bits>(std::copysign(0.0, -1.0))
                   : RefFloat<Bits>(0.0);
      return left;
    }

    /// A binary floating-point format narrower than f32: significand bits
    /// (with the implicit one), least normal exponent, largest finite value.
    struct NarrowFormat {
      const char *name;
      int precision;
      int minExponent;
      double largest;
    };

    NarrowFormat narrowFormat(TensorOp op) {
      if (op == TensorOp::NarrowF16)
        return {"f16", 11, -14, 65504.0};
      return {"bf16", 8, -126, std::ldexp(255.0, 120)};
    }

    /// Round an f32 to the format, to nearest with ties to even, in exact
    /// double arithmetic: the quantum is a power of two and the scaled value
    /// has at most 24 significant bits.
    RuntimeValue narrowLane(const RuntimeValue &value, const NarrowFormat &format) {
      if (value.kind == RuntimeValue::Kind::Undef)
        throw UndefinedBehaviorError("UB: Reading undef tensor element");
      const double x = static_cast<double>(static_cast<float>(value.floatVal));
      double rounded = x;
      if (x != 0.0) {
        int exponent = 0;
        std::frexp(x, &exponent);
        const int leading = std::max(exponent - 1, format.minExponent);
        const double quantum = std::ldexp(1.0, leading - (format.precision - 1));
        const double scaled = x / quantum;
        const double floor = std::floor(scaled);
        const double fraction = scaled - floor;
        double integral = floor;
        if (fraction > 0.5 || (fraction == 0.5 && std::fmod(floor, 2.0) != 0.0))
          integral = floor + 1.0;
        rounded = std::copysign(integral * quantum, x);
      }
      if (std::fabs(rounded) > format.largest)
        throw UndefinedBehaviorError(std::string("UB: tensor narrow overflows ") + format.name);
      RuntimeValue result;
      result.kind = RuntimeValue::Kind::Float;
      result.bits = 32;
      result.floatVal = static_cast<double>(static_cast<float>(rounded));
      return result;
    }

    template<unsigned Bits>
    RuntimeValue binaryResult(TensorOp op, const RuntimeValue &left, const RuntimeValue &right) {
      if (op == TensorOp::Rem) {
        if (left.kind != RuntimeValue::Kind::Float || right.kind != RuntimeValue::Kind::Float ||
            left.bits != 32 || right.bits != 32)
          throw std::runtime_error("tensor f32 remainder requires materialized f32 operands");
        const float a = static_cast<float>(left.floatVal);
        const float b = static_cast<float>(right.floatVal);
        if (!std::isfinite(a) || !std::isfinite(b))
          throw std::runtime_error("tensor f32 remainder operand is not finite");
        if (b == 0.0f)
          throw UndefinedBehaviorError("UB: tensor f32 remainder by zero");
        const float remainder = std::fmod(a, b);
        if (!std::isfinite(remainder))
          throw UndefinedBehaviorError("UB: non-finite tensor f32 remainder result");
        RuntimeValue result;
        result.kind = RuntimeValue::Kind::Float;
        result.bits = 32;
        result.floatVal = static_cast<double>(remainder);
        return result;
      }
      const auto a = readFloat<Bits>(left);
      const auto b = readFloat<Bits>(right);
      if (op == TensorOp::Div && b == 0)
        throw UndefinedBehaviorError("UB: tensor f32 division by zero");
      if (op == TensorOp::Min)
        return roundToF32<Bits>(minNum<Bits>(a, b));
      if (op == TensorOp::Max)
        return roundToF32<Bits>(maxNum<Bits>(a, b));
      switch (op) {
        case TensorOp::Add:
          return roundToF32<Bits>(a + b);
        case TensorOp::Sub:
          return roundToF32<Bits>(a - b);
        case TensorOp::Mul:
          return roundToF32<Bits>(a * b);
        case TensorOp::Div:
          return roundToF32<Bits>(a / b);
        default:
          throw std::runtime_error("unsupported numerical tensor binary operator");
      }
    }

    template<unsigned Bits>
    RuntimeValue reduceGroup(TensorOp op, const std::vector<RuntimeValue> &group) {
      if (group.empty())
        throw std::runtime_error("tensor numerical reference reduction group is empty");
      auto accum = readFloat<Bits>(group.front());
      if (op == TensorOp::Sum) {
        accum = 0;
        for (const auto &value: group)
          accum += readFloat<Bits>(value);
      } else if (op == TensorOp::ReduceMin || op == TensorOp::ReduceMax) {
        for (std::size_t i = 1; i < group.size(); ++i) {
          const auto value = readFloat<Bits>(group[i]);
          accum = op == TensorOp::ReduceMin ? minNum<Bits>(accum, value)
                                            : maxNum<Bits>(accum, value);
        }
      } else {
        throw std::runtime_error("unsupported numerical tensor reduction operator");
      }
      return roundToF32<Bits>(accum);
    }

    template<unsigned Bits>
    RuntimeValue evalTensorNumericImpl(
        const TensorAtom &atom, const std::vector<RuntimeValue> &operands,
        const TensorTypeInfo &resultType
    ) {
      if (resultType.element != TensorElement::F32)
        throw std::runtime_error("numerical tensor reference requires f32 values");

      if (atom.op == TensorOp::Exp) {
        const auto &input = operands.at(0);
        const Shape inputShape =
            input.kind == RuntimeValue::Kind::Tensor ? input.tensorShape : Shape{};
        const auto count = inputShape.empty() ? 1 : tensorElementCount(inputShape);
        std::vector<RuntimeValue> output;
        output.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
          const auto coords = inputShape.empty() ? Coordinates{} : coordinates(i, inputShape);
          output.push_back(
              roundToF32<Bits>(
                  boost::multiprecision::exp(readFloat<Bits>(scalarAt(input, coords, inputShape)))
              )
          );
        }
        return tensorOrScalar(resultType.shape, std::move(output));
      }

      if (atom.op == TensorOp::Add || atom.op == TensorOp::Sub || atom.op == TensorOp::Mul ||
          atom.op == TensorOp::Div || atom.op == TensorOp::Rem || atom.op == TensorOp::Min ||
          atom.op == TensorOp::Max) {
        const auto sourceShape = [](const RuntimeValue &value) -> Shape {
          return value.kind == RuntimeValue::Kind::Tensor ? value.tensorShape : Shape{};
        };
        const Shape leftShape = sourceShape(operands.at(0));
        const Shape rightShape = sourceShape(operands.at(1));
        if (tensorBroadcastShape(leftShape, rightShape) != resultType.shape)
          throw std::runtime_error("tensor numerical result shape does not match checked shape");
        const auto count = resultType.shape.empty() ? 1 : tensorElementCount(resultType.shape);
        std::vector<RuntimeValue> output;
        output.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
          const auto coords =
              resultType.shape.empty() ? Coordinates{} : coordinates(i, resultType.shape);
          output.push_back(
              binaryResult<Bits>(
                  atom.op, projectedAt(operands[0], leftShape, coords),
                  projectedAt(operands[1], rightShape, coords)
              )
          );
        }
        return tensorOrScalar(resultType.shape, std::move(output));
      }

      if (atom.op == TensorOp::Sum || atom.op == TensorOp::ReduceMin ||
          atom.op == TensorOp::ReduceMax) {
        const auto &input = operands.at(0);
        if (input.kind != RuntimeValue::Kind::Tensor)
          throw std::runtime_error("tensor numerical reduction input is not a tensor");
        const auto axis =
            normalizeTensorAxis(atom.axis, static_cast<std::uint32_t>(input.tensorShape.size()));
        const auto count = resultType.shape.empty() ? 1 : tensorElementCount(resultType.shape);
        std::vector<RuntimeValue> output;
        output.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
          const auto outCoords =
              resultType.shape.empty() ? Coordinates{} : coordinates(i, resultType.shape);
          output.push_back(
              reduceGroup<Bits>(
                  atom.op, readGroup<Bits>(input, input.tensorShape, outCoords, axis, atom.keepDims)
              )
          );
        }
        return tensorOrScalar(resultType.shape, std::move(output));
      }

      if (atom.op == TensorOp::Cumsum) {
        // Each prefix is its exact sum rounded once, like `sum`.
        const auto &input = operands.at(0);
        const auto axis =
            normalizeTensorAxis(atom.axis, static_cast<std::uint32_t>(input.tensorShape.size()));
        std::vector<RuntimeValue> output;
        output.reserve(tensorElementCount(resultType.shape));
        for (std::uint32_t i = 0; i < tensorElementCount(resultType.shape); ++i) {
          auto coords = coordinates(i, resultType.shape);
          const auto last = coords[axis];
          RefFloat<Bits> total = 0;
          for (std::uint32_t k = 0; k <= last; ++k) {
            coords[axis] = k;
            total += readFloat<Bits>(scalarAt(input, coords, input.tensorShape));
          }
          output.push_back(roundToF32<Bits>(total));
        }
        return tensorOrScalar(resultType.shape, std::move(output));
      }

      if (isTensorWhere(atom.op)) {
        // The chosen arm is returned with its bits, so a signed zero survives;
        // the comparison treats -0 and +0 as equal, as IEEE does.
        std::vector<RuntimeValue> output;
        output.reserve(tensorElementCount(resultType.shape));
        for (std::uint32_t i = 0; i < tensorElementCount(resultType.shape); ++i) {
          const auto coords = coordinates(i, resultType.shape);
          std::vector<RuntimeValue> lanes;
          for (const auto &operand: operands) {
            const Shape shape =
                operand.kind == RuntimeValue::Kind::Tensor ? operand.tensorShape : Shape{};
            lanes.push_back(projectedAt(operand, shape, coords));
          }
          const auto a = readFloat<Bits>(lanes[0]);
          const auto b = readFloat<Bits>(lanes[1]);
          (void) readFloat<Bits>(lanes[2]);
          (void) readFloat<Bits>(lanes[3]);
          const bool holds = atom.op == TensorOp::WhereLt   ? a < b
                             : atom.op == TensorOp::WhereLe ? a <= b
                                                            : a == b;
          output.push_back(holds ? lanes[2] : lanes[3]);
        }
        return tensorOrScalar(resultType.shape, std::move(output));
      }

      if (atom.op == TensorOp::NarrowF16 || atom.op == TensorOp::NarrowBF16) {
        const auto &input = operands.at(0);
        const auto format = narrowFormat(atom.op);
        if (input.kind != RuntimeValue::Kind::Tensor)
          return narrowLane(input, format);
        std::vector<RuntimeValue> output;
        output.reserve(input.arrayVal.size());
        for (const auto &lane: input.arrayVal)
          output.push_back(narrowLane(lane, format));
        return tensorOrScalar(resultType.shape, std::move(output));
      }

      if (atom.op == TensorOp::Dot) {
        // The exact inner product rounded once, like `sum`: products of f32
        // values are exact in the reference precision.
        const auto &left = operands.at(0);
        const auto &right = operands.at(1);
        const auto inner = left.tensorShape.at(1);
        std::vector<RuntimeValue> output;
        output.reserve(tensorElementCount(resultType.shape));
        for (std::uint32_t i = 0; i < tensorElementCount(resultType.shape); ++i) {
          const auto coords = coordinates(i, resultType.shape);
          RefFloat<Bits> total = 0;
          for (std::uint32_t k = 0; k < inner; ++k)
            total += readFloat<Bits>(scalarAt(left, {coords[0], k}, left.tensorShape)) *
                     readFloat<Bits>(scalarAt(right, {k, coords[1]}, right.tensorShape));
          output.push_back(roundToF32<Bits>(total));
        }
        return tensorOrScalar(resultType.shape, std::move(output));
      }

      if (atom.op == TensorOp::Softmax) {
        const auto &input = operands.at(0);
        if (input.kind != RuntimeValue::Kind::Tensor || input.tensorShape != resultType.shape)
          throw std::runtime_error("tensor softmax shape does not match checked shape");
        const auto axis =
            normalizeTensorAxis(atom.axis, static_cast<std::uint32_t>(input.tensorShape.size()));
        Shape groupShape = input.tensorShape;
        groupShape.erase(groupShape.begin() + axis);
        std::vector<RuntimeValue> output(tensorElementCount(input.tensorShape));
        const auto groupCount = groupShape.empty() ? 1 : tensorElementCount(groupShape);
        for (std::uint32_t groupIndex = 0; groupIndex < groupCount; ++groupIndex) {
          auto groupCoords =
              groupShape.empty() ? Coordinates{} : coordinates(groupIndex, groupShape);
          groupCoords.insert(groupCoords.begin() + axis, 0);
          std::vector<RefFloat<Bits>> shifted;
          shifted.reserve(input.tensorShape[axis]);
          auto maxValue = readFloat<Bits>(scalarAt(input, groupCoords, input.tensorShape));
          for (std::uint32_t i = 0; i < input.tensorShape[axis]; ++i) {
            groupCoords[axis] = i;
            const auto value = readFloat<Bits>(scalarAt(input, groupCoords, input.tensorShape));
            if (value > maxValue)
              maxValue = value;
            shifted.push_back(value);
          }
          RefFloat<Bits> denominator = 0;
          for (auto &value: shifted) {
            value = boost::multiprecision::exp(value - maxValue);
            denominator += value;
          }
          for (std::uint32_t i = 0; i < input.tensorShape[axis]; ++i) {
            groupCoords[axis] = i;
            const auto index = linearIndex(groupCoords, input.tensorShape);
            output[index] = roundToF32<Bits>(shifted[i] / denominator);
          }
        }
        return tensorOrScalar(resultType.shape, std::move(output));
      }

      throw std::runtime_error("unsupported numerical tensor operator");
    }
#endif

  } // namespace

  void setTensorReferenceBits(unsigned bits) {
    if (bits != 128 && bits != 256)
      throw std::invalid_argument("tensor reference precision must be 128 or 256 bits");
    referenceBits = bits;
  }

  unsigned tensorReferenceBits() noexcept { return referenceBits; }

  void setTensorTraceHook(TensorTraceHook hook) { tensorTraceHook = std::move(hook); }

  void traceTensorValue(const SourceSpan &span, const RuntimeValue &value) {
    if (tensorTraceHook)
      tensorTraceHook(span, value);
  }

  bool tensorTracesEqual(
      const std::vector<TensorTraceEntry> &left, const std::vector<TensorTraceEntry> &right
  ) {
    if (left.size() != right.size())
      return false;
    const auto scalarEqual = [](const RuntimeValue &a, const RuntimeValue &b) {
      if (a.kind != b.kind || a.bits != b.bits)
        return false;
      if (a.kind == RuntimeValue::Kind::Int)
        return a.intVal == b.intVal;
      if (a.kind == RuntimeValue::Kind::Float)
        return std::bit_cast<std::uint64_t>(a.floatVal) == std::bit_cast<std::uint64_t>(b.floatVal);
      return a.kind == RuntimeValue::Kind::Undef;
    };
    const auto valueEqual = [&](const RuntimeValue &a, const RuntimeValue &b) {
      if (a.kind != b.kind)
        return false;
      if (a.kind != RuntimeValue::Kind::Tensor)
        return scalarEqual(a, b);
      if (a.tensorShape != b.tensorShape || a.arrayVal.size() != b.arrayVal.size())
        return false;
      for (std::size_t i = 0; i < a.arrayVal.size(); ++i)
        if (!scalarEqual(a.arrayVal[i], b.arrayVal[i]))
          return false;
      return true;
    };
    for (std::size_t i = 0; i < left.size(); ++i) {
      if (left[i].span.begin.offset != right[i].span.begin.offset ||
          !valueEqual(left[i].value, right[i].value))
        return false;
    }
    return true;
  }

  RuntimeValue evalTensorNumeric(
      const TensorAtom &atom, const std::vector<RuntimeValue> &operands,
      const TensorTypeInfo &resultType, unsigned precisionBits
  ) {
    if (precisionBits != 128 && precisionBits != 256)
      throw std::invalid_argument("tensor reference precision must be 128 or 256 bits");
#if defined(REFRACTIR_TENSOR_REF)
    if (precisionBits == 128)
      return evalTensorNumericImpl<128>(atom, operands, resultType);
    return evalTensorNumericImpl<256>(atom, operands, resultType);
#else
    (void) atom;
    (void) operands;
    (void) resultType;
    throw std::runtime_error(
        "numerical tensor reference is disabled; rebuild with TENSOR_REF=1 and Boost headers"
    );
#endif
  }

} // namespace refractir
