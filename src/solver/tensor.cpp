#include "analysis/tensor.hpp"
#include "analysis/type_utils.hpp"
#include "solver/solver.hpp"

#include <stdexcept>

namespace refractir {
  namespace {

    using Coordinates = std::vector<std::uint32_t>;

    Coordinates coordinatesOf(std::uint32_t linear, const Shape &shape) {
      Coordinates result(shape.size());
      for (std::size_t i = shape.size(); i > 0; --i) {
        result[i - 1] = linear % shape[i - 1];
        linear /= shape[i - 1];
      }
      return result;
    }

    std::uint32_t linearIndex(const Coordinates &coords, const Shape &shape) {
      if (coords.size() != shape.size())
        throw std::runtime_error("tensor coordinate rank mismatch in symbolic evaluation");
      std::uint32_t result = 0;
      for (std::size_t i = 0; i < shape.size(); ++i) {
        if (coords[i] >= shape[i])
          throw std::runtime_error("tensor coordinate out of bounds in symbolic evaluation");
        result = result * shape[i] + coords[i];
      }
      return result;
    }

    Shape shapeOf(const TypePtr &type) {
      if (const auto tensor = TypeUtils::asTensor(type))
        return tensor->shape;
      return {};
    }

    SymbolicValue resultFromLeaves(Shape shape, std::vector<SymbolicValue> leaves) {
      if (shape.empty()) {
        if (leaves.size() != 1)
          throw std::runtime_error("scalar tensor operation produced an invalid leaf count");
        return leaves.front();
      }
      SymbolicValue result;
      result.kind = SymbolicValue::Kind::Tensor;
      result.tensorShape = std::move(shape);
      result.arrayVal = std::move(leaves);
      return result;
    }

    void
    requireDefined(const SymbolicValue &value, smt::ISolver &solver, std::vector<smt::Term> &ub) {
      if (value.kind == SymbolicValue::Kind::Int && value.is_defined.internal)
        ub.push_back(value.is_defined);
      else if (value.kind == SymbolicValue::Kind::Undef)
        ub.push_back(solver.make_false());
    }

    SymbolicValue inputAt(
        const SymbolicValue &value, const Shape &sourceShape, const Coordinates &outputCoords,
        const Shape &outputShape
    ) {
      if (sourceShape.empty())
        return value;
      if (sourceShape.size() > outputShape.size())
        throw std::runtime_error("tensor input rank exceeds output rank in symbolic evaluation");
      const auto padding = outputShape.size() - sourceShape.size();
      Coordinates sourceCoords(sourceShape.size());
      for (std::size_t i = 0; i < sourceShape.size(); ++i) {
        sourceCoords[i] = sourceShape[i] == 1 ? 0 : outputCoords[padding + i];
      }
      return value.arrayVal.at(linearIndex(sourceCoords, sourceShape));
    }

    const SymbolicValue &
    sourceAt(const SymbolicValue &value, const Shape &shape, const Coordinates &coords) {
      if (shape.empty())
        return value;
      return value.arrayVal.at(linearIndex(coords, shape));
    }

    void requireAllDefined(
        const SymbolicValue &value, smt::ISolver &solver, std::vector<smt::Term> &ub
    ) {
      if (value.kind == SymbolicValue::Kind::Tensor) {
        for (const auto &leaf: value.arrayVal)
          requireDefined(leaf, solver, ub);
      } else {
        requireDefined(value, solver, ub);
      }
    }

    smt::Term signedExtendTo(smt::Term value, std::uint32_t width, smt::ISolver &solver) {
      const auto actual = solver.get_bv_width(solver.get_sort(value));
      if (actual < width)
        return solver.make_term(smt::Kind::BV_SIGN_EXTEND, {value}, {width - actual});
      return value;
    }

    smt::Term checkedIntegerBinary(
        TensorOp op, smt::Term lhs, smt::Term rhs, smt::ISolver &solver, std::vector<smt::Term> &ub
    ) {
      const auto sort = solver.get_sort(lhs);
      switch (op) {
        case TensorOp::Add: {
          const auto overflow = solver.make_term(smt::Kind::BV_SADD_OVERFLOW, {lhs, rhs});
          ub.push_back(solver.make_term(smt::Kind::NOT, {overflow}));
          return solver.make_term(smt::Kind::BV_ADD, {lhs, rhs});
        }
        case TensorOp::Sub: {
          const auto overflow = solver.make_term(smt::Kind::BV_SSUB_OVERFLOW, {lhs, rhs});
          ub.push_back(solver.make_term(smt::Kind::NOT, {overflow}));
          return solver.make_term(smt::Kind::BV_SUB, {lhs, rhs});
        }
        case TensorOp::Mul: {
          const auto overflow = solver.make_term(smt::Kind::BV_SMUL_OVERFLOW, {lhs, rhs});
          ub.push_back(solver.make_term(smt::Kind::NOT, {overflow}));
          return solver.make_term(smt::Kind::BV_MUL, {lhs, rhs});
        }
        case TensorOp::Div: {
          const auto zero = solver.make_bv_zero(sort);
          ub.push_back(solver.make_term(smt::Kind::DISTINCT, {rhs, zero}));
          const auto minimum = solver.make_bv_min_signed(sort);
          const auto minusOne = solver.make_bv_value_int64(sort, -1);
          const auto isMin = solver.make_term(smt::Kind::EQUAL, {lhs, minimum});
          const auto isMinusOne = solver.make_term(smt::Kind::EQUAL, {rhs, minusOne});
          const auto overflow = solver.make_term(smt::Kind::AND, {isMin, isMinusOne});
          ub.push_back(solver.make_term(smt::Kind::NOT, {overflow}));
          return solver.make_term(smt::Kind::BV_SDIV, {lhs, rhs});
        }
        case TensorOp::Min:
        case TensorOp::Max: {
          const auto cmp = solver.make_term(
              op == TensorOp::Min ? smt::Kind::BV_SLE : smt::Kind::BV_SGE, {lhs, rhs}
          );
          return solver.make_term(smt::Kind::ITE, {cmp, lhs, rhs});
        }
        case TensorOp::And:
          return solver.make_term(smt::Kind::BV_AND, {lhs, rhs});
        case TensorOp::Or:
          return solver.make_term(smt::Kind::BV_OR, {lhs, rhs});
        case TensorOp::Xor:
          return solver.make_term(smt::Kind::BV_XOR, {lhs, rhs});
        case TensorOp::Rem:
          throw std::runtime_error(
              "unsupported_theory: exact tensor rem has no f32 SMT encoding"
          );
        default:
          throw std::runtime_error("unsupported_theory: exact tensor binary operator");
      }
    }

  } // namespace

  SymbolicValue SymbolicExecutor::evalTensorAtom(
      const TensorAtom &arg, smt::ISolver &solver, SymbolicStore &store, std::vector<smt::Term> &ub
  ) {
    if (!arg.resolvedType)
      throw std::runtime_error(
          "tensor atom reached symbolic execution without checked type metadata"
      );

    const Shape outputShape = shapeOf(arg.resolvedType);
    const auto outputCount = tensorElementCount(outputShape);
    const TypePtr elementType = TypeUtils::asTensor(arg.resolvedType)
                                    ? TypeUtils::asTensor(arg.resolvedType)->elem
                                    : arg.resolvedType;
    const bool isFloat = TypeUtils::getFloatBitWidth(elementType).has_value();
    const bool numericalOp = arg.op == TensorOp::Add || arg.op == TensorOp::Sub ||
                             arg.op == TensorOp::Mul || arg.op == TensorOp::Div ||
                             arg.op == TensorOp::Rem || arg.op == TensorOp::Min ||
                             arg.op == TensorOp::Max || arg.op == TensorOp::Exp ||
                             arg.op == TensorOp::Sum || arg.op == TensorOp::ReduceMin ||
                             arg.op == TensorOp::ReduceMax || arg.op == TensorOp::Softmax ||
                             arg.op == TensorOp::NarrowF16 || arg.op == TensorOp::NarrowBF16;
    if (isFloat && numericalOp) {
      throw std::runtime_error(
          "unsupported_theory: numerical tensor operator '" + std::string(tensorOpName(arg.op)) +
          "' has no exact SMT encoding"
      );
    }

    auto readOperand = [&](std::size_t index) -> SymbolicValue {
      const Coef &coef = arg.operands.at(index);
      if (const auto id = std::get_if<LocalOrSymId>(&coef)) {
        return std::visit([&](const auto &name) { return store.at(name.name); }, *id);
      }
      smt::Term term = evalCoef(coef, solver, store, getSort(elementType, solver));
      return SymbolicValue(SymbolicValue::Kind::Int, term, solver.make_true());
    };

    if (arg.op == TensorOp::Iota) {
      const auto axis =
          normalizeTensorAxis(arg.axis, static_cast<std::uint32_t>(arg.attribute.size()));
      std::vector<SymbolicValue> leaves;
      leaves.reserve(outputCount);
      const auto sort = getSort(elementType, solver);
      for (std::uint32_t i = 0; i < outputCount; ++i) {
        const auto coords = coordinatesOf(i, arg.attribute);
        leaves.emplace_back(
            SymbolicValue::Kind::Int,
            solver.make_bv_value_int64(sort, static_cast<std::int64_t>(coords[axis])),
            solver.make_true()
        );
      }
      return resultFromLeaves(arg.attribute, std::move(leaves));
    }

    if (arg.operands.empty())
      throw std::runtime_error("tensor atom is missing a symbolic operand");
    std::vector<SymbolicValue> operands;
    operands.reserve(arg.operands.size());
    for (std::size_t i = 0; i < arg.operands.size(); ++i)
      operands.push_back(readOperand(i));

    auto sourceShape = [](const SymbolicValue &value) {
      return value.kind == SymbolicValue::Kind::Tensor ? value.tensorShape : Shape{};
    };
    const auto firstShape = sourceShape(operands.front());

    if (arg.op == TensorOp::Extract) {
      const auto index = linearIndex(arg.attribute, firstShape);
      const auto &leaf = operands.front().arrayVal.at(index);
      requireDefined(leaf, solver, ub);
      return leaf;
    }

    if (arg.op == TensorOp::Expand || arg.op == TensorOp::Broadcast ||
        arg.op == TensorOp::Transpose) {
      const auto &input = operands.front();
      requireAllDefined(input, solver, ub);
      const auto count = tensorElementCount(outputShape);
      std::vector<SymbolicValue> leaves;
      leaves.reserve(count);
      std::uint32_t axis = 0;
      if (arg.op == TensorOp::Expand)
        axis = normalizeTensorAxis(arg.axis, static_cast<std::uint32_t>(firstShape.size() + 1));
      for (std::uint32_t i = 0; i < count; ++i) {
        auto outputCoords = coordinatesOf(i, outputShape);
        Coordinates sourceCoords;
        if (arg.op == TensorOp::Expand) {
          sourceCoords = outputCoords;
          sourceCoords.erase(sourceCoords.begin() + axis);
        } else if (arg.op == TensorOp::Broadcast) {
          const auto padding = outputShape.size() - firstShape.size();
          sourceCoords.resize(firstShape.size());
          for (std::size_t j = 0; j < firstShape.size(); ++j)
            sourceCoords[j] = firstShape[j] == 1 ? 0 : outputCoords[padding + j];
        } else {
          sourceCoords.resize(firstShape.size());
          for (std::size_t j = 0; j < arg.attribute.size(); ++j)
            sourceCoords[arg.attribute[j]] = outputCoords[j];
        }
        leaves.push_back(sourceAt(input, firstShape, sourceCoords));
      }
      return resultFromLeaves(outputShape, std::move(leaves));
    }

    if (arg.op == TensorOp::Add || arg.op == TensorOp::Sub || arg.op == TensorOp::Mul ||
        arg.op == TensorOp::Div || arg.op == TensorOp::Min || arg.op == TensorOp::Max ||
        arg.op == TensorOp::And || arg.op == TensorOp::Or || arg.op == TensorOp::Xor) {
      if (operands.size() != 2)
        throw std::runtime_error("tensor binary operator has an invalid operand count");
      const auto lhsShape = sourceShape(operands[0]);
      const auto rhsShape = sourceShape(operands[1]);
      const auto count = tensorElementCount(outputShape);
      std::vector<SymbolicValue> leaves;
      leaves.reserve(count);
      for (std::uint32_t i = 0; i < count; ++i) {
        const auto coords = coordinatesOf(i, outputShape);
        auto lhs = inputAt(operands[0], lhsShape, coords, outputShape);
        auto rhs = inputAt(operands[1], rhsShape, coords, outputShape);
        requireDefined(lhs, solver, ub);
        requireDefined(rhs, solver, ub);
        auto left = lhs.term;
        auto right = rhs.term;
        const auto width = solver.get_bv_width(solver.get_sort(left));
        left = signedExtendTo(left, width, solver);
        right = signedExtendTo(right, width, solver);
        leaves.emplace_back(
            SymbolicValue::Kind::Int, checkedIntegerBinary(arg.op, left, right, solver, ub),
            solver.make_true()
        );
      }
      return resultFromLeaves(outputShape, std::move(leaves));
    }

    if (arg.op == TensorOp::Sum || arg.op == TensorOp::ReduceMin || arg.op == TensorOp::ReduceMax) {
      const auto axis =
          normalizeTensorAxis(arg.axis, static_cast<std::uint32_t>(firstShape.size()));
      const auto groupSize = firstShape[axis];
      const auto count = tensorElementCount(outputShape);
      const auto sourceSort = getSort(elementType, solver);
      const auto wideSort = solver.make_bv_sort(64);
      const auto zeroWide = solver.make_bv_value_int64(wideSort, 0);
      const auto maxWide = solver.make_bv_value_int64(wideSort, INT32_MAX);
      const auto minWide = solver.make_bv_value_int64(wideSort, INT32_MIN);
      std::vector<SymbolicValue> leaves;
      leaves.reserve(count);
      for (std::uint32_t i = 0; i < count; ++i) {
        auto outputCoords = coordinatesOf(i, outputShape);
        Coordinates sourceCoords;
        if (arg.keepDims) {
          sourceCoords = outputCoords;
        } else {
          sourceCoords = outputCoords;
          sourceCoords.insert(sourceCoords.begin() + axis, 0);
        }

        smt::Term result;
        if (arg.op == TensorOp::Sum) {
          auto positive = zeroWide;
          auto negative = zeroWide;
          for (std::uint32_t k = 0; k < groupSize; ++k) {
            sourceCoords[axis] = k;
            const auto &leaf = operands.front().arrayVal.at(linearIndex(sourceCoords, firstShape));
            requireDefined(leaf, solver, ub);
            const auto zero = solver.make_bv_zero(sourceSort);
            const auto nonnegative = solver.make_term(smt::Kind::BV_SGE, {leaf.term, zero});
            const auto positivePart =
                solver.make_term(smt::Kind::ITE, {nonnegative, leaf.term, zero});
            const auto negativePart =
                solver.make_term(smt::Kind::ITE, {nonnegative, zero, leaf.term});
            positive = solver.make_term(
                smt::Kind::BV_ADD, {positive, signedExtendTo(positivePart, 64, solver)}
            );
            negative = solver.make_term(
                smt::Kind::BV_ADD, {negative, signedExtendTo(negativePart, 64, solver)}
            );
          }
          ub.push_back(solver.make_term(smt::Kind::BV_ULE, {positive, maxWide}));
          ub.push_back(solver.make_term(smt::Kind::BV_SGE, {negative, minWide}));
          const auto wideResult = solver.make_term(smt::Kind::BV_ADD, {positive, negative});
          result = solver.make_term(smt::Kind::BV_EXTRACT, {wideResult}, {31, 0});
        } else {
          sourceCoords[axis] = 0;
          const auto &first = operands.front().arrayVal.at(linearIndex(sourceCoords, firstShape));
          requireDefined(first, solver, ub);
          result = first.term;
          for (std::uint32_t k = 1; k < groupSize; ++k) {
            sourceCoords[axis] = k;
            const auto &next = operands.front().arrayVal.at(linearIndex(sourceCoords, firstShape));
            requireDefined(next, solver, ub);
            const auto cmp = solver.make_term(
                arg.op == TensorOp::ReduceMin ? smt::Kind::BV_SLE : smt::Kind::BV_SGE,
                {result, next.term}
            );
            result = solver.make_term(smt::Kind::ITE, {cmp, result, next.term});
          }
        }
        leaves.emplace_back(SymbolicValue::Kind::Int, result, solver.make_true());
      }
      return resultFromLeaves(outputShape, std::move(leaves));
    }

    throw std::runtime_error(
        "unsupported_theory: tensor operator '" + std::string(tensorOpName(arg.op)) + "'"
    );
  }

} // namespace refractir
