#pragma once

// StateProfile — a per-program-point record of the concrete program state
// a RefractIR program passes through under a specific input.
//
// rysmith generates a program `p1` together with the exact input `i` that
// concretizes it, so the whole execution is deterministic and known. A
// StateProfile captures the value of every initialized local / parameter at
// each executed program point (block entry, or after each instruction) by
// running the reference interpreter with a state-capture hook. It is the
// data an equivalence-preserving rewrite needs: to synthesize a block `B'`
// that reproduces `B`'s effect on the one state `B` actually sees, we must
// first know that state (`s` at B's entry) and the state it produces (`s'`
// at B's exit == the next block's entry).
//
// Serialized as a JSON sidecar by `rysmith --emit-state` and consumed by
// `rytwin`. Float leaves are carried as canonical decimal strings via
// refractir::formatDouble so the profile is bit-exact across the text
// boundary (see the FP-serialization invariant in CLAUDE.md).

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "analysis/type_utils.hpp"
#include "ast/ast.hpp"
#include "interp/value.hpp"

namespace refractir {
struct Program;
  class Interpreter;
  class Memory;
  class TypeLayout;
} // namespace refractir

namespace refractir::reify {

  // Granularity of the captured trace. The CLI / JSON tokens are "pbb"
  // (per basic block) and "ppp" (per program point).
  enum class StateGranularity {
    Pbb, // one record per executed block, at its entry
    Ppp, // additionally one record after each instruction
  };

  // A concrete value tree mirroring the RefractIR type shape.
  // Uninitialized (undef) leaves are their own kind and carry no value.
  // Pointer leaves carry their provenance when it resolves: the
  // originating local (`ptrRoot`) and the byte offset of the pointee cell
  // from that local's base (`ptrOfs`), or `ptrNull` for the null pointer.
  // A Ptr leaf with an empty root and no null flag is unresolved (opaque).
  // Scalars keep their bit-width; float scalars keep a canonical decimal
  // string (bit-exact) alongside the raw double.
  struct StateValue {
    enum class Kind { Int, Float, Array, Vec, Struct, Ptr, Undef } kind = Kind::Undef;
    std::int64_t intVal = 0;
    double floatVal = 0.0;
    std::uint32_t bits = 0;                                 // scalar bit-width (Int / Float)
    std::vector<StateValue> elems;                          // Array / Vec, in order
    std::vector<std::pair<std::string, StateValue>> fields; // Struct, sorted by field name
    std::string ptrRoot;                                    // Ptr: originating local
    std::uint64_t ptrOfs = 0;                               // Ptr: byte offset from root base
    bool ptrNull = false;                                   // Ptr: the null pointer

    bool ptrResolved() const { return ptrNull || !ptrRoot.empty(); }
  };

  // One captured program point.
  struct StatePoint {
    // Executing function ("@f") and its activation id, so interprocedural
    // traces keep frames apart even when block labels collide across
    // functions. Old sidecar files lack both: `func` stays empty (consumers
    // fall back to the profile's entry) and `frame` stays 0.
    std::string func;
    std::uint64_t frame = 0;
    std::string block; // block label, e.g. "^b3"
    int instr = -1;    // -1 = block entry; >= 0 = after instruction `instr`
    // Initialized locals + parameters visible at this point, in a stable
    // (sorted-by-name) order so two runs serialize identically.
    std::vector<std::pair<std::string, StateValue>> vars;
  };

  struct StateProfile {
    std::string func; // entry function name (canonical, "@...")
    StateGranularity granularity = StateGranularity::Pbb;
    std::vector<StatePoint> trace; // in execution order
  };

  // Convert one interpreter RuntimeValue into a StateValue tree. The
  // overload with a Memory resolves pointer leaves to their provenance
  // (root local + byte offset); without it pointers stay opaque.
  StateValue toStateValue(const RuntimeValue &rv);
  StateValue toStateValue(const RuntimeValue &rv, const Memory &memory);

  // One scalar leaf of a StateValue tree: its access path from the root
  // (concrete IntLit indices / field names) and its value.
  struct StateLeaf {
    std::vector<Access> path;
    StateValue val;
  };

  // Enumerate the scalar leaves of `v` in tree order, appending to `out`.
  // Pointer leaves are appended too, carrying their provenance; `hasPtr` /
  // `hasUndef` let a consumer that cannot handle them bail out early. Undef
  // leaves have no value and are the one kind not appended. Struct fields
  // follow the StateValue's own (name-sorted) order.
  void
  enumStateLeaves(const StateValue &v, std::vector<StateLeaf> &out, bool &hasPtr, bool &hasUndef);

  // One scalar the run recorded at a program point: the declaration it belongs
  // to, the steps that reach it, its leafKey, and what it held.
  struct RecordedLeaf {
    std::string root;
    std::vector<Access> path;
    std::string key;
    StateValue value;
  };

  // The scalars recorded at `blockLabel`, one set per visit the run made to
  // it. Points inside a block are ignored: a caller reads the state a block
  // starts from, and a block entered twice yields two sets rather than one.
  [[nodiscard]] std::vector<std::vector<RecordedLeaf>>
  leavesAtBlock(const StateProfile &profile, const std::string &blockLabel);

  // Where a recorded pointer leaf points, as an lvalue the function can name.
  //
  // A profile records provenance — the originating local and a byte offset —
  // rather than the cell itself, so recovering the cell means walking the
  // root's declared type to the access path that offset falls on. `ptrType` is
  // the leaf's own static type, which fixes what is being pointed *at*.
  //
  // Returns nullopt, and sets `why` when given, for a pointer that names
  // nothing reconstructible: the null pointer, provenance the interpreter
  // could not resolve, a root the function does not declare, or an offset no
  // access path reaches (one past the end).
  //
  // `mustBeAddressable` additionally refuses a root that is not a `let mut`,
  // which is what a caller rebuilding the pointer with `addr` needs and a
  // caller only reading through it does not.
  [[nodiscard]] std::optional<LValue> resolvePointee(
      const StateValue &leaf, const TypePtr &ptrType, const FunDecl &fn,
      const TypeUtils::StructTable &structs, const TypeLayout &layout, bool mustBeAddressable,
      const char **why = nullptr
  );

  // Bit-exact structural equality of two state trees. Floats compare by
  // bit pattern (so +0.0 != -0.0), ints by value, aggregates recursively;
  // Ptr / Undef leaves compare by kind only.
  bool bitExactEq(const StateValue &a, const StateValue &b);

  // Install a state-capture hook on `interp` that appends each program
  // point it sees to `out.trace` (at the granularity `gran`). The caller
  // owns `out` and should have set `out.func` / `out.granularity`. This is
  // the single place the capture hook is defined so both profileProgram
  // (in-process on a Program) and runSymiriCaptureResult (file-based, in
  // reify/common) fill a profile from the same run they already make.
  void attachStateProfile(Interpreter &interp, StateProfile &out, StateGranularity gran);

  // Run `prog`'s entry `func` under `paramArgs` (one decimal-int / hex-float
  // string per parameter, declaration order) with the given granularity and
  // return the resulting StateProfile. Throws whatever the interpreter
  // throws (e.g. UndefinedBehaviorError) — callers targeting UB-free
  // programs can let it propagate.
  // `maxBlockSteps` caps the number of basic blocks the profiling run may
  // enter (0 = unlimited); when exceeded the interpreter throws StepLimitError.
  // A caller that can't rule out a non-terminating program up front (no
  // descriptor) passes a bound so profiling fails cleanly instead of hanging.
  StateProfile profileProgram(
      const Program &prog, const std::string &func, const std::vector<std::string> &paramArgs,
      StateGranularity granularity = StateGranularity::Pbb, std::uint64_t maxBlockSteps = 0
  );

  // Serialize a StateProfile to JSON (bit-exact FP via formatDouble).
  void writeStateProfileJson(std::ostream &os, const StateProfile &profile);

  // Parse a StateProfile from the JSON that writeStateProfileJson emits
  // (rytwin reads the func_<id>_<i>.state.json sidecar). Floats are read
  // back through refractir::parseFloatLiteral so the round-trip is
  // bit-exact. Returns std::nullopt on malformed input. This is a focused
  // parser for our own regular format — not a general JSON reader — in
  // keeping with reify's no-JSON-dependency stance.
  std::optional<StateProfile> readStateProfileJson(const std::string &json);

} // namespace refractir::reify
