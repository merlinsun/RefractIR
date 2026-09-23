#pragma once

#include <iosfwd>
#include <string>
#include <vector>
#include "ast/ast.hpp"
#include "ast/tensor.hpp"
#include "interp/value.hpp"

namespace refractir {

  std::vector<RuntimeValue>
  readTensorArguments(std::istream &input, const std::vector<TensorTypeInfo> &types);
  void
  writeTensorResult(std::ostream &output, const RuntimeValue &value, const TensorTypeInfo &type);
  std::string describeTensorEntry(const Program &program, const std::string &entryName);

} // namespace refractir
