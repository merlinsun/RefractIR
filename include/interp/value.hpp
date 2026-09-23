#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include "ast/tensor.hpp"

namespace refractir {

  /**
   * Represents a value during interpreter runtime.
   *
   * Namespace-scoped (rather than nested in Interpreter) so the memory and
   * type-layout collaborators can operate on it without depending on the
   * full Interpreter definition.
   */
  struct RuntimeValue {
    enum class Kind { Int, Float, Array, Struct, Undef, Ptr, Vec, Tensor } kind = Kind::Undef;
    std::int64_t intVal = 0;
    double floatVal = 0.0;
    std::uint32_t bits = 64;    // bitwidth for Int or Float (32/64)
    std::uint64_t ptrVal = 0;   // for Ptr kind: raw address
    std::uint64_t ptrBase = 0;  // for Ptr kind: provenance object ID (ObjectInfo::provId)
    std::uint64_t elemSize = 1; // for Ptr kind: static element size of the pointee type
    std::vector<RuntimeValue> arrayVal;
    // Tensor values use flat row-major leaves and keep their logical shape
    // separate from arrays and vectors.
    Shape tensorShape;
    // See SymbolicValue::structVal: GCC 11's unordered_map requires a
    // complete mapped value type at this recursive member declaration.
    std::map<std::string, RuntimeValue> structVal;
    // Vec: same shape as Array (per-lane RuntimeValue tuple),
    // but represents a vector value (no address; not in heap_; lane-wise
    // arithmetic). Element kind matches the lane scalar type.
  };

  /// Local variable store: name → current value.
  using Store = std::unordered_map<std::string, RuntimeValue>;

  /**
   * The canonical text of a runtime value — what the `Result:` line carries.
   *
   * Floats render as IEEE 754 hex (`printf %a`), which round-trips losslessly,
   * distinguishes +0 from -0, and handles subnormals. Cross-validating the
   * interpreter against compiled C compares these strings, so decimal would
   * lose bits exactly at that boundary.
   *
   * One formatter, so a caller that wants the value as text and a caller that
   * reads the printed line cannot disagree about what a value looks like.
   *
   * Returns an empty string for an aggregate, a vector, or undef: those have
   * no one-line form, and a caller decides what to do about it rather than
   * being handed a placeholder it might record as a value.
   */
  [[nodiscard]] std::string formatRuntimeValue(const RuntimeValue &v);

  /**
   * The text version of a runtime value, for a person reading along.
   *
   * Deliberately lossy where formatRuntimeValue is exact: floats print at
   * default decimal precision and aggregates collapse to an ellipsis. Every
   * value has some rendering here, which is what a trace wants and what a
   * recorded value must not have.
   */
  [[nodiscard]] std::string rvToString(const RuntimeValue &v);

} // namespace refractir
