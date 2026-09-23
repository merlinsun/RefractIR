#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include "ast/ast.hpp"

namespace refractir {

  struct TypeUtils {
    /**
     * Returns the bitwidth of the given type if it is an integer type.
     * Returns std::nullopt for floats, pointers, vectors, aggregates.
     */
    static std::optional<std::uint32_t> getIntBitWidth(const TypePtr &t);

    /**
     * Returns the bitwidth of the given type if it is a float type.
     * Returns std::nullopt for integers, pointers, vectors, aggregates.
     */
    static std::optional<std::uint32_t> getFloatBitWidth(const TypePtr &t);

    /**
     * Returns the bitwidth of any scalar type (integer or float).
     *   i32 → 32, i64 → 64, iN → N, f32 → 32, f64 → 64.
     * Returns std::nullopt for pointers, vectors, aggregates.
     */
    static std::optional<std::uint32_t> getScalarBitWidth(const TypePtr &t);

    /**
     * Returns the total bitwidth of a vector type: N * scalarBitWidth(elem).
     * Returns std::nullopt for non-vector types or if the element type
     * is not a scalar (which the typechecker prevents).
     */
    static std::optional<std::uint32_t> getVectorBitWidth(const TypePtr &t);

    /**
     * Returns the bitwidth of any scalar or vector type.
     * Equivalent to: getScalarBitWidth(t) || getVectorBitWidth(t).
     * Returns std::nullopt for pointers, structs, and arrays.
     */
    static std::optional<std::uint32_t> getBitWidth(const TypePtr &t);

    /**
     * Checks if two types are structurally equal.
     *
     * Integers compare by width, not by `IntType::Kind`. `I32` and `I64` are
     * a shorthand for the two widths a program names most often, and `iN` is
     * the same type however it was spelled; a caller that builds a type from
     * a computed width has no way to know which spelling to reach for, so
     * observing the distinction here would make equality depend on where a
     * type came from rather than what it denotes.
     */
    static bool areTypesEqual(const TypePtr &a, const TypePtr &b);

    /**
     * Casts to ArrayType if possible, otherwise returns nullptr.
     */
    static const ArrayType *asArray(const TypePtr &t);

    /**
     * Casts to StructType if possible, otherwise returns nullptr.
     */
    static const StructType *asStruct(const TypePtr &t);

    /**
     * Casts to VecType if possible, otherwise returns nullptr.
     */
    static const VecType *asVec(const TypePtr &t);

    /**
     * Casts to TensorType if possible, otherwise returns nullptr.
     */
    static const TensorType *asTensor(const TypePtr &t);

    /**
     * Casts to PtrType if possible, otherwise returns nullptr.
     */
    static const PtrType *asPtr(const TypePtr &t);

    /**
     * The pointee of a `ptr T`, or nullptr when `t` is not a pointer.
     */
    static TypePtr pointee(const TypePtr &t);

    /**
     * Returns true if the type is an array type.
     */
    static bool isArray(const TypePtr &t);

    /**
     * Returns true if the type is a struct type.
     */
    static bool isStruct(const TypePtr &t);

    /**
     * True iff the type is a vector type `<N> T`.
     */
    static bool isVec(const TypePtr &t);

    /**
     * True iff `t` is a fixed-shape tensor value.
     */
    static bool isTensor(const TypePtr &t);

    /**
     * True iff the type is an integer type `iN`.
     */
    static bool isInt(const TypePtr &t);

    /**
     * True iff the type is a float type (`f32` or `f64`).
     */
    static bool isFloat(const TypePtr &t);

    /**
     * True iff the type is a pointer type `ptr T`.
     */
    static bool isPtr(const TypePtr &t);

    /**
     * True iff the type is a scalar: an integer or a float. Vectors,
     * pointers and aggregates are not scalars, though a vector's element
     * type is.
     */
    static bool isScalar(const TypePtr &t);

    /**
     * True iff the type is an aggregate: an array or a struct. A vector is
     * not an aggregate — its lanes are not addressable (SPEC §6.8.1).
     */
    static bool isAggregate(const TypePtr &t);

    /// Struct registry shared by every layout query: name → declaration.
    using StructTable = std::unordered_map<std::string, const StructDecl *>;

    /**
     * Index `prog`'s struct declarations by name. The entries point into
     * `prog`, so the table is valid only while `prog` outlives it and its
     * `structs` vector is not reallocated.
     */
    static StructTable buildStructTable(const Program &prog);

    /**
     * The static type reached from `t` by one access step: an element of an
     * array or a vector, or a named field of a struct. Returns nullptr when
     * the step does not apply — an index into a scalar, a field of a
     * non-struct, an unknown struct, or a field the struct does not declare.
     *
     * An `AccessIndex` does not read its index. Whether the index is in
     * bounds is a separate question from what type it lands on, and only the
     * first is answerable statically.
     */
    static TypePtr stepType(const TypePtr &t, const Access &acc, const StructTable &structs);

    /**
     * The static type reached from `t` by following `accesses` in order.
     * Returns nullptr as soon as a step does not apply, and `t` itself when
     * `accesses` is empty.
     */
    /**
     * Every scalar `t` reaches, as the access path that reaches it paired with
     * the scalar's own type. An `iN` or `fN` is its own only leaf; arrays,
     * vectors and structs are walked through, at any depth.
     *
     * A vector is walked like an array. Its lanes are not addressable
     * (SPEC §6.8.1), but they are readable by subscript, which is what a
     * caller enumerating scalars is after. A pointer is a leaf: what it points
     * at is a question about a value, not about a type.
     */
    static std::vector<std::pair<std::vector<Access>, TypePtr>>
    scalarLeaves(const TypePtr &t, const StructTable &structs);

    static TypePtr accessPathType(
        const TypePtr &t, const std::vector<Access> &accesses, const StructTable &structs
    );

    /**
     * True iff a vector occurs anywhere inside `t`, at any depth through
     * arrays and struct fields. A type that contains one cannot be navigated
     * through a pointer, since vector lanes are not addressable
     * (SPEC §6.8.1).
     */
    static bool containsVec(const TypePtr &t, const StructTable &structs);

    /**
     * Packed byte size of `t` — the one object-layout authority (SPEC §4:
     * `sizeof(@S) = Σ sizeof(field_i)`, no padding).
     *
     * Every consumer that reasons about object extents must measure with
     * this: the interpreter's memory model, and the solver's pointer
     * provenance and arithmetic. A second, differently-scaled size model
     * silently disagrees about what is in bounds — counting one unit per
     * scalar *leaf* rather than per byte, for instance, agrees with this one
     * only while every scalar in an object has the same width, and otherwise
     * both misses real out-of-bounds arithmetic and rejects valid programs.
     */
    static std::uint64_t packedSizeof(const TypePtr &t, const StructTable &structs);

    /**
     * Byte offset of `field` within `s` — sequential, no padding, and
     * measured with packedSizeof so it shares its scale.
     */
    static std::uint64_t
    packedFieldOffset(const StructDecl &s, const std::string &field, const StructTable &structs);
  };

} // namespace refractir
