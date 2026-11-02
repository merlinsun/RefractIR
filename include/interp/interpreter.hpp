#pragma once

#include <functional>
#include <iosfwd>
#include <list>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include "ast/ast.hpp"
#include "interp/memory.hpp"
#include "interp/type_layout.hpp"
#include "interp/value.hpp"

namespace refractir {

  /**
   * A concrete interpreter for the RefractIR language.
   * Executes RefractIR programs by evaluating expressions and instructions
   * against concrete values for symbolic variables.
   */
  // Evaluate a built-in intrinsic on concrete arguments, with no program or
  // interpreter state around it. src/interp/intrinsics.cpp is the single
  // source of truth for what an intrinsic computes, so an analysis that wants
  // to fold a call asks here rather than restating the semantics — and gets
  // the UB conditions with them, as the same UndefinedBehaviorError the
  // interpreter would raise.
  RuntimeValue evalIntrinsic(const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args);

  class Interpreter {
  public:
    explicit Interpreter(const Program &prog);
    // Output sink for the `Result:` line and the `--dump-exec` trace.
    // Defaults to std::cout via the single-arg ctor; a caller that needs to
    // capture the output passes a local stream instead of redirecting the
    // process-global std::cout, which is unsafe when other threads are
    // running concurrently.
    Interpreter(const Program &prog, std::ostream &out);
    /**
     * Executes the specified entry function with given symbolic bindings.
     * @param entryFuncName The name of the function to start execution from.
     * @param symBindings Mapping of symbolic identifiers to concrete values.
     * @param dumpExec Whether to print execution trace to stderr.
     * @param dumpCall Whether to print one line per function activation.
     */
    // One binding is an ordered list of scalar values: a
    // single entry for scalar syms (or a vector splat), one entry per
    // lane for vector syms.
    using SymScalar = std::variant<std::int64_t, double>;
    using SymBindings = std::unordered_map<std::string, std::vector<SymScalar>>;

    // `paramArgs` supplies one decimal-int / hex-float string
    // per parameter of the entry function. Empty when the entry is
    // parameterless (the common case).
    //
    // Returns what the entry function returned, or nullopt when it returned
    // nothing. The `Result:` line still goes to the output stream: it is what
    // the command-line tools print, not how a caller reads the value. An
    // in-process caller takes the returned RuntimeValue and neither renders
    // nor re-parses it.
    std::optional<RuntimeValue>
    run(const std::string &entryFuncName, const SymBindings &symBindings,
        const std::vector<std::string> &paramArgs = {}, bool dumpExec = false,
        bool dumpCall = false);

    /// Execute with already-decoded, type-checked parameter values.
    /// `dumpExec` prints executed blocks and updates, as `run` does.
    std::optional<RuntimeValue> runTyped(
        const std::string &entryFuncName, const SymBindings &symBindings,
        const std::vector<RuntimeValue> &paramArgs, bool dumpExec = false
    );

    // RuntimeValue and Store are defined in interp/value.hpp at namespace
    // scope so the memory / type-layout collaborators can share them.

    // Optional state-capture hook. When set, it is invoked with the live
    // local Store at the entry of every executed block (`instrIdx == -1`)
    // and, when `perInstr` is true, again after each instruction
    // (`instrIdx == i`). `funcName` is the executing function and
    // `frameId` its activation (unique per function entry within one run),
    // so interprocedural traces keep frames apart even when block labels
    // collide across functions. `memory` is the live memory model, so the
    // consumer can resolve pointer values to their provenance (originating
    // local + byte offset). It is the low-level primitive a per-program-point
    // state profile is built on: recording the concrete state at each point
    // lets a consumer reproduce the exact state a block sees under a given
    // input, which is what an equivalence-preserving rewrite needs to check
    // it against. No-op when unset.
    using StateHook = std::function<void(
        const std::string &funcName, std::uint64_t frameId, const std::string &blockLabel,
        int instrIdx, const Store &store, const Memory &memory
    )>;

    void setStateHook(StateHook hook, bool perInstr = false) {
      stateHook_ = std::move(hook);
      stateHookPerInstr_ = perInstr;
    }

    // Bound the number of basic blocks a run may enter (across nested calls).
    // When exceeded, execution throws StepLimitError. 0 (default) is
    // unlimited. Lets a caller replay a possibly non-terminating program
    // under a bound instead of hanging. Reset per run().
    void setMaxBlockSteps(std::uint64_t n) { maxBlockSteps_ = n; }

  private:
    const Program &prog_;
    std::ostream &out_; // sink for Result: / dump-exec (default std::cout)
    bool dumpExec_ = false;
    bool dumpCall_ = false;
    // Owns the struct registry + pure type-layout queries (sizeofType,
    // getCellTypeAtOffset, fieldOffset). Frame-independent; see TypeLayout.
    TypeLayout typeLayout_;

    // Memory model (flat heap, provenance objects, address map, allocator)
    // for pointer operations. Owns ObjectInfo; see Memory.
    Memory memory_;
    // typeMap_: varName → TypePtr, rebuilt at start of each execFunction call.
    // Per-frame static-type state (stays on the interpreter, not in Memory).
    std::unordered_map<std::string, TypePtr> typeMap_;

    TypePtr getLValueType(const LValue &lv) const;
    TypePtr getCoefType(const Coef &coef) const;
    TypePtr getAtomType(const Atom &atom) const;
    TypePtr getExprType(const Expr &expr) const;

    // --- Runtime evaluation helpers ---
    RuntimeValue makeUndef(const TypePtr &t);
    RuntimeValue broadcast(const TypePtr &t, const RuntimeValue &v);
    // Materialize one sym binding (scalar, vector splat, or
    // per-lane list). Shared by the entry and nested-call paths.
    RuntimeValue bindSymValue(const SymDecl &s, const std::vector<SymScalar> &vals);
    RuntimeValue evalInit(const InitVal &iv, const TypePtr &t, const Store &store);

    std::optional<RuntimeValue> execFunction(
        const FunDecl &f, const std::vector<RuntimeValue> &args, const SymBindings &symBindings
    );

    // Nested function call (`call @fun(...)`). Evaluates the
    // callee's body with a fresh local frame and returns its `ret`
    // value. Heap state (objects, addresses) is preserved across the
    // call so pointer arguments remain valid. typeMap_ entries that
    // collide with callee parameter/local names are saved and restored.
    RuntimeValue callFunction(const FunDecl &f, std::vector<RuntimeValue> args);

    // §9.6.1 step 5: after a callee returns, refresh caller-side
    // Store entries for any local that was promoted to memory (addr was
    // taken). Reads through a pointer commit to `heap_`; without this
    // refresh, subsequent direct reads of the local from the caller's
    // Store would observe the stale pre-call value.
    void syncStoreFromHeap(Store &store);

    // Reset per-run interpreter state shared by the textual and typed entry
    // points.
    void beginRun(bool dumpExec, bool dumpCall);

    // Run the body of a function whose Store has already been
    // populated. Used by both execFunction (top-level) and callFunction
    // (nested). A `ret` with a value writes it to `outRet` when that is
    // non-null; a `ret` without one leaves it alone, so an empty optional
    // means the function returned nothing.
    void runBlocks(const FunDecl &f, Store &store, std::optional<RuntimeValue> *outRet);

    // Render the `Result:` line for the value the entry function returned.
    void printResult(const std::optional<RuntimeValue> &res);

    // Sym bindings live on the interpreter (so nested calls
    // share the same SAT model). Captured in run() / execFunction.
    const SymBindings *symBindings_ = nullptr;

    // Optional per-program-point state-capture hook (see setStateHook).
    StateHook stateHook_;
    bool stateHookPerInstr_ = false;
    // Activation counter for StateHook frame ids: each runBlocks entry
    // (top-level run or nested call) claims the next id.
    std::uint64_t nextFrameId_ = 0;

    // Block-step budget (see setMaxBlockSteps). `maxBlockSteps_ == 0` means
    // unlimited; `blockSteps_` counts blocks entered in the current run and
    // is reset at the top of run().
    std::uint64_t maxBlockSteps_ = 0;
    std::uint64_t blockSteps_ = 0;

    RuntimeValue evalExpr(const Expr &e, const Store &store);
    // evalAtom dispatches on the Atom variant; each alternative's evaluation
    // lives in a dedicated evalXxxAtom helper (src/interp/expr.cpp). The two
    // trivial alternatives (CoefAtom / RValueAtom) are handled inline in the
    // dispatcher and delegate straight to evalCoef / evalLValue.
    RuntimeValue evalAtom(const Atom &a, const Store &store);
    RuntimeValue evalOpAtom(const OpAtom &arg, const Store &store);
    RuntimeValue evalUnaryAtom(const UnaryAtom &arg, const Store &store);
    RuntimeValue evalSelectAtom(const SelectAtom &arg, const Store &store);
    RuntimeValue evalCmpAtom(const CmpAtom &arg, const Store &store);
    RuntimeValue evalAddrAtom(const AddrAtom &arg, const Store &store);
    RuntimeValue evalLoadAtom(const LoadAtom &arg, const Store &store);
    RuntimeValue evalPtrIndexAtom(const PtrIndexAtom &arg, const Store &store);
    RuntimeValue evalPtrFieldAtom(const PtrFieldAtom &arg, const Store &store);
    RuntimeValue evalTensorAtom(const TensorAtom &arg, const Store &store);
    RuntimeValue evalCastAtom(const CastAtom &arg, const Store &store);
    RuntimeValue evalCallAtom(const CallAtom &arg, const Store &store);
    // Execute a built-in intrinsic. Argument values are already
    // evaluated. Result has the intrinsic's declared return bitwidth.
    // Implemented in src/interp/intrinsics.cpp — the single source of
    // truth for interpreter-side intrinsic semantics.
    RuntimeValue callIntrinsic(
        const IntrinsicDecl &intr, const std::vector<RuntimeValue> &args, SourceSpan callSpan
    );
    RuntimeValue evalCoef(const Coef &c, const Store &store);
    RuntimeValue evalSelectVal(const SelectVal &sv, const Store &store);
    RuntimeValue evalLValue(const LValue &lv, const Store &store);
    void setLValue(const LValue &lv, RuntimeValue val, Store &store);

    bool evalCond(const Cond &c, const Store &store);
  };

} // namespace refractir
