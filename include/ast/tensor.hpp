#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace refractir {

  /// Ordered dimensions for a tensor value. An empty shape denotes a scalar
  /// only at evaluator and transport boundaries, never in an AST TensorType.
  using Shape = std::vector<std::uint32_t>;

  enum class TensorElement { I32, F32 };

  struct TensorTypeInfo {
    Shape shape;
    TensorElement element = TensorElement::I32;
  };

  enum class TensorOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Min,
    Max,
    And,
    Or,
    Xor,
    Exp,
    Iota,
    Extract,
    Expand,
    Broadcast,
    Transpose,
    Sum,
    ReduceMin,
    ReduceMax,
    Softmax,
    Window,
    Dot,
    NarrowF16,
    NarrowBF16,
  };

  inline std::string_view tensorOpName(TensorOp op) noexcept {
    switch (op) {
      case TensorOp::Add:
        return "add";
      case TensorOp::Sub:
        return "sub";
      case TensorOp::Mul:
        return "mul";
      case TensorOp::Div:
        return "div";
      case TensorOp::Rem:
        return "rem";
      case TensorOp::Min:
        return "min";
      case TensorOp::Max:
        return "max";
      case TensorOp::And:
        return "and";
      case TensorOp::Or:
        return "or";
      case TensorOp::Xor:
        return "xor";
      case TensorOp::Exp:
        return "exp";
      case TensorOp::Iota:
        return "iota";
      case TensorOp::Extract:
        return "extract";
      case TensorOp::Expand:
        return "expand";
      case TensorOp::Broadcast:
        return "broadcast";
      case TensorOp::Transpose:
        return "transpose";
      case TensorOp::Sum:
        return "sum";
      case TensorOp::ReduceMin:
        return "rmin";
      case TensorOp::ReduceMax:
        return "rmax";
      case TensorOp::Softmax:
        return "softmax";
      case TensorOp::Window:
        return "window";
      case TensorOp::Dot:
        return "dot";
      case TensorOp::NarrowF16:
        return "narrow f16";
      case TensorOp::NarrowBF16:
        return "narrow bf16";
    }
    return "?";
  }

} // namespace refractir
