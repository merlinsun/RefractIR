#pragma once

#include <functional>
#include <vector>
#include "ast/ast.hpp"
#include "ast/tensor.hpp"
#include "interp/tensor.hpp"

namespace refractir {

  using TensorTraceHook = std::function<void(const SourceSpan &, const RuntimeValue &)>;

  struct TensorTraceEntry {
    SourceSpan span;
    RuntimeValue value;
  };

  /// Select the precision used by the independent finite f32 tensor reference.
  /// Only 128 and 256 bits are defined by the tensor profile.
  void setTensorReferenceBits(unsigned bits);
  unsigned tensorReferenceBits() noexcept;

  /// Record a completed tensor atom when a reference replay is collecting a trace.
  void setTensorTraceHook(TensorTraceHook hook);
  void traceTensorValue(const SourceSpan &span, const RuntimeValue &value);
  bool tensorTracesEqual(
      const std::vector<TensorTraceEntry> &left, const std::vector<TensorTraceEntry> &right
  );

  RuntimeValue evalTensorNumeric(
      const TensorAtom &atom, const std::vector<RuntimeValue> &operands,
      const TensorTypeInfo &resultType, unsigned precisionBits
  );

} // namespace refractir
