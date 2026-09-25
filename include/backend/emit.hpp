#pragma once

// Driving a backend end to end: check the program, pick the lowering, write
// the file.
//
// Every backend is configured the same way — the same flags in the same order,
// the same reducibility precondition when control flow is reconstructed, the
// same vector-lowering choice — and the sequence is easy to get subtly wrong
// one caller at a time. This header owns it, so a backend that grows a knob
// grows it once here and every caller has it.
//
// The functions report success or failure and write nothing to any stream but
// the output file. What a failure should say, and to whom, is the caller's:
// a command-line tool prints it, an embedding counts it. Pass `verbose` to get
// a one-line reason on stderr while diagnosing.

#include <filesystem>
#include <string>
#include <vector>

#include "analysis/tensor.hpp"
#include "ast/ast.hpp"
#include "backend/triton_layout.hpp"

namespace refractir {

  /**
   * How a program is lowered. Each field means the same thing whichever
   * backend reads it; a backend with no use for one ignores it.
   */
  struct EmitOptions {
    // Keep `require` statements in the emitted code rather than dropping them.
    bool keepRequire = false;
    // Drop the dynamic UB guards (see CBackend::setNoUbGuards). Sound only
    // when the program is known UB-free.
    bool noUbGuards = false;
    // Vector-lowering strategy name; empty selects the backend's default
    // ("vecext" for C and WASM, "array" for python). A name the target does
    // not implement is an error, not a fallback.
    std::string vecLowering;
    // Reconstruct while/if control flow instead of C's goto or WASM's `$__pc`
    // dispatch loop. Requires a reducible program. Python always structures
    // and ignores this.
    bool structuredLowering = false;
    // Emit the program's `main` unmangled, as a real entry point.
    bool emitMain = false;
    // C only: write one `<stem>.c` per source file instead of one unit.
    bool splitBySource = false;
    // Report the reason for a failure on stderr.
    bool verbose = false;
    // Triton selects one ordinary RefractIR function as a kernel entry.
    std::string entry = "kernel";
    // Triton numerical softmax lowering: direct or decomposed.
    std::string tensorSoftmaxLowering = "direct";
    // Tensor-v1 remains the default. Tensor-v2 selects its larger static
    // tensor bound and, as those capabilities ship, its loop subset.
    TensorProfileId tensorProfile = TensorProfileId::V1;
    // Physical input/output memory mappings selected by one case manifest.
    std::vector<TritonBufferLayout> tritonBufferLayouts;
  };

  // Compile to C. Writes `<outDir>/<primaryStem>.c`, or one file per source
  // stem under `opts.splitBySource`.
  bool emitC(
      Program &prog, const std::filesystem::path &outDir, const std::string &primaryStem,
      const EmitOptions &opts
  );

  // Compile to WebAssembly text.
  bool emitWasm(Program &prog, const std::filesystem::path &outFile, const EmitOptions &opts);

  // Compile to Python. Requires a reducible program, since the backend
  // structures unconditionally.
  bool emitPy(Program &prog, const std::filesystem::path &outFile, const EmitOptions &opts);

  // Emit one complete-tile Triton JIT kernel. The file is replaced only after
  // checking and rendering both succeed.
  bool emitTriton(Program &prog, const std::filesystem::path &outFile, const EmitOptions &opts);

  /// Read the validated JSON layout manifest consumed by symirc. Optional
  /// input/output roles preserve compatibility when buffer names are unique;
  /// explicit roles disambiguate an input named `return` from the result.
  std::vector<TritonBufferLayout> readTritonBufferLayouts(const std::filesystem::path &path);

  // Parse a `.sir` file and compile it for `target` ("c", "wasm" or
  // "python" or "triton"). The C target writes one file here whatever `opts` says, since
  // `outPath` names a file rather than a directory.
  bool emitSirFile(
      const std::filesystem::path &sirPath, const std::string &target,
      const std::filesystem::path &outPath, const EmitOptions &opts
  );

} // namespace refractir
