#pragma once

#include <vector>
#include "ast/ast.hpp"
#include "interp/value.hpp"

namespace refractir {

  /// Evaluate the exact, layout-preserving tensor fragment and integer operators.
  RuntimeValue evalTensorExact(
      const TensorAtom &atom, const std::vector<RuntimeValue> &operands,
      const TensorTypeInfo &resultType
  );

  /// Convert an already checked scalar or tensor AST type to transport metadata.
  TensorTypeInfo tensorTypeInfo(const TypePtr &type);

} // namespace refractir
