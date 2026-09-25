#pragma once

// Turning RefractIR source text into a program that is safe to run, compile or
// solve.
//
// Every consumer of a `.sir` file takes the same two steps: lex and parse the
// text into a `Program`, then run the checks that establish the invariants the
// interpreter, the backends and the symbolic executor all assume. That shape
// is fixed, and this header owns it, so a check added here reaches every
// consumer instead of the ones that remembered to list it.
//
// The checks are ordered. `SemChecker` and `TypeChecker` run over the whole
// module first, because per-function analysis of a program that is not
// well-formed means nothing, and because the later passes read the types the
// type checker assigns. The function analyses then run over each function in
// turn. `CheckOptions` selects which of them apply.
//
// Diagnostics stay with the caller. `checkProgram` fills a `DiagBag` and
// answers whether the program is well-formed; rendering a diagnostic needs the
// source text and a choice of format, and neither belongs to the checking.

#include <string_view>

#include "analysis/tensor.hpp"
#include "ast/ast.hpp"
#include "frontend/diagnostics.hpp"

namespace refractir {

  // Lex and parse `src`. Throws `LexError` or `ParseError` describing the
  // first construct it could not read.
  //
  // The lexer reads `src` in place, so the text must outlive the call. It need
  // not outlive the returned `Program`: tokens own their spelling and the AST
  // owns every name it carries. A caller that renders diagnostics keeps the
  // source anyway, since a `Span` is an offset into it.
  [[nodiscard]] Program parseSource(std::string_view src);

  // Which checks `checkProgram` applies. The module-level semantic and type
  // checks always run: they establish that the program is a program at all.
  struct CheckOptions {
    // Also run the per-function analyses. Definite initialization is the one
    // that rejects a program, since reading an uninitialized local is UB the
    // checker can see; reachability and unused names only warn.
    bool functionAnalyses = true;

    // Also require every function's CFG to be reducible. Structured lowering
    // is total only on reducible control flow, so a consumer that structures
    // asks here rather than discovering the problem inside a backend.
    bool reducibility = false;

    // Report a program that produced any warning as not well-formed.
    bool warningsAreErrors = false;

    // Apply the selected tensor profile during type checking. Defaulting to
    // v1 keeps existing consumers and tools on the shipped contract.
    TensorProfileId tensorProfile = TensorProfileId::V1;
  };

  // Run the checks `opts` selects over `prog`, appending every diagnostic to
  // `diags`. Returns true when the program is well-formed under `opts`.
  //
  // A false return means `diags` holds at least one error, or at least one
  // warning under `warningsAreErrors`. A true return does not mean `diags` is
  // empty: a warning is reported and does not, on its own, reject.
  [[nodiscard]] bool checkProgram(Program &prog, DiagBag &diags, const CheckOptions &opts = {});

} // namespace refractir
