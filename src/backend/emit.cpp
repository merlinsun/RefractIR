#include "backend/emit.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <system_error>

#include "analysis/reducibility.hpp"
#include "analysis/tensor.hpp"
#include "backend/c_backend.hpp"
#include "backend/c_vec_lowering.hpp"
#include "backend/py_backend.hpp"
#include "backend/py_vec_lowering.hpp"
#include "backend/triton_backend.hpp"
#include "backend/wasm_backend.hpp"
#include "backend/wasm_vec_lowering.hpp"
#include "frontend/pipeline.hpp"

namespace fs = std::filesystem;

namespace refractir {

  namespace {

    void note(bool verbose, const std::string &msg) {
      if (verbose)
        std::cerr << "emit: " << msg << "\n";
    }

    // A program must be well-formed before a backend walks it: the backends
    // read the types the checker assigns and assume every branch resolves.
    bool checked(Program &prog, bool verbose, TensorConsumer consumer) {
      DiagBag diags;
      if (checkProgram(prog, diags) && checkTensorCapabilities(prog, consumer, diags))
        return true;
      if (verbose) {
        std::cerr << "emit: the program did not pass the checks:\n";
        for (const auto &d: diags.diags)
          if (d.level == DiagLevel::Error)
            std::cerr << "  error: " << d.message << "\n";
      }
      return false;
    }

    // Reconstructing control flow is total only on reducible CFGs. Callers
    // filter or repair upstream; verifying here turns a violation into a
    // clean failure instead of malformed output.
    bool everyFunReducible(const Program &prog, bool verbose) {
      for (const auto &f: prog.funs)
        if (!ReducibilityResult::isReducible(f)) {
          note(verbose, "structured lowering needs reducible control flow: " + f.name.name);
          return false;
        }
      return true;
    }

    bool openOut(std::ofstream &ofs, const fs::path &p, bool verbose) {
      ofs.open(p);
      if (!ofs) {
        note(verbose, "cannot open " + p.string());
        return false;
      }
      return true;
    }

  } // namespace

  bool emitC(
      Program &prog, const fs::path &outDir, const std::string &primaryStem, const EmitOptions &opts
  ) {
    if (!checked(prog, opts.verbose, TensorConsumer::C))
      return false;
    if (opts.structuredLowering && !everyFunReducible(prog, opts.verbose))
      return false;

    auto vl = makeCVecLowering(opts.vecLowering.empty() ? "vecext" : opts.vecLowering);
    if (!vl) {
      note(opts.verbose, "the C target has no vec-lowering '" + opts.vecLowering + "'");
      return false;
    }

    // The split and single-file forms differ only in where the output goes and
    // which emit call makes it; how the backend lowers is identical. Under
    // splitBySource the backend opens its own files, so `sink` stays closed.
    std::ofstream sink;
    if (!opts.splitBySource && !openOut(sink, outDir / (primaryStem + ".c"), opts.verbose))
      return false;

    CBackend cb(sink);
    cb.setNoRequire(!opts.keepRequire);
    cb.setNoUbGuards(opts.noUbGuards);
    cb.setNoMainMangle(opts.emitMain);
    cb.setStructuredLowering(opts.structuredLowering);
    cb.setVecLowering(std::move(vl));
    try {
      if (opts.splitBySource)
        cb.emitSplit(prog, outDir.string(), primaryStem);
      else
        cb.emit(prog);
    } catch (const std::exception &e) {
      note(opts.verbose, std::string("the C backend failed: ") + e.what());
      return false;
    }
    return true;
  }

  bool emitWasm(Program &prog, const fs::path &outFile, const EmitOptions &opts) {
    if (!checked(prog, opts.verbose, TensorConsumer::Wasm))
      return false;
    if (opts.structuredLowering && !everyFunReducible(prog, opts.verbose))
      return false;

    auto vl = makeWasmVecLowering(opts.vecLowering.empty() ? "vecext" : opts.vecLowering);
    if (!vl) {
      note(opts.verbose, "the WASM target has no vec-lowering '" + opts.vecLowering + "'");
      return false;
    }

    std::ofstream ofs;
    if (!openOut(ofs, outFile, opts.verbose))
      return false;

    WasmBackend wb(ofs);
    wb.setNoRequire(!opts.keepRequire);
    wb.setNoUbGuards(opts.noUbGuards);
    wb.setNoMainMangle(opts.emitMain);
    wb.setStructuredLowering(opts.structuredLowering);
    wb.setVecLowering(std::move(vl));
    try {
      wb.emit(prog);
    } catch (const std::exception &e) {
      note(opts.verbose, std::string("the WASM backend failed: ") + e.what());
      return false;
    }
    return true;
  }

  bool emitPy(Program &prog, const fs::path &outFile, const EmitOptions &opts) {
    if (!checked(prog, opts.verbose, TensorConsumer::Python))
      return false;
    // Python has no goto, so the backend structures whatever it is given.
    if (!everyFunReducible(prog, opts.verbose))
      return false;

    auto vl = makePyVecLowering(opts.vecLowering.empty() ? "array" : opts.vecLowering);
    if (!vl) {
      note(opts.verbose, "the python target has no vec-lowering '" + opts.vecLowering + "'");
      return false;
    }

    std::ofstream ofs;
    if (!openOut(ofs, outFile, opts.verbose))
      return false;

    PyBackend pb(ofs);
    pb.setNoRequire(!opts.keepRequire);
    pb.setNoUbGuards(opts.noUbGuards);
    pb.setNoMainMangle(opts.emitMain);
    pb.setVecLowering(std::move(vl));
    try {
      pb.emit(prog);
    } catch (const std::exception &e) {
      note(opts.verbose, std::string("the python backend failed: ") + e.what());
      return false;
    }
    return true;
  }

  bool emitTriton(Program &prog, const fs::path &outFile, const EmitOptions &opts) {
    if (opts.keepRequire) {
      note(opts.verbose, "tensor-profile: --keep-require is not supported for Triton emission");
      return false;
    }
    if (opts.splitBySource || opts.structuredLowering || !opts.vecLowering.empty()) {
      note(
          opts.verbose,
          "tensor-profile: the selected code-generation option does not apply to Triton"
      );
      return false;
    }
    if (!checked(prog, opts.verbose, TensorConsumer::Triton))
      return false;

    std::ostringstream rendered;
    try {
      TritonBackend backend(rendered, opts.entry, opts.tensorSoftmaxLowering);
      backend.emit(prog);
    } catch (const std::exception &e) {
      note(opts.verbose, e.what());
      return false;
    }

    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path temporaryPath = outFile;
    temporaryPath += ".tmp." + std::to_string(ticks);
    std::ofstream temporary(temporaryPath, std::ios::binary | std::ios::trunc);
    if (!temporary) {
      note(opts.verbose, "cannot open temporary output " + temporaryPath.string());
      return false;
    }
    temporary << rendered.str();
    temporary.close();
    if (!temporary) {
      std::error_code ignored;
      fs::remove(temporaryPath, ignored);
      note(opts.verbose, "cannot finish temporary output " + temporaryPath.string());
      return false;
    }
    std::error_code error;
    fs::rename(temporaryPath, outFile, error);
    if (error) {
      std::error_code ignored;
      fs::remove(temporaryPath, ignored);
      note(opts.verbose, "cannot replace " + outFile.string() + ": " + error.message());
      return false;
    }
    return true;
  }

  bool emitSirFile(
      const fs::path &sirPath, const std::string &target, const fs::path &outPath,
      const EmitOptions &opts
  ) {
    // Read separately so a missing file keeps its own message rather than
    // surfacing as a compilation failure.
    std::ifstream in(sirPath);
    if (!in) {
      note(opts.verbose, "cannot open " + sirPath.string());
      return false;
    }
    std::stringstream buf;
    buf << in.rdbuf();
    const std::string src = buf.str();

    try {
      Program prog = parseSource(src);
      if (target == "c") {
        // outPath names a file, so the split form has nowhere to go.
        EmitOptions single = opts;
        single.splitBySource = false;
        return emitC(prog, outPath.parent_path(), outPath.stem().string(), single);
      }
      if (target == "wasm")
        return emitWasm(prog, outPath, opts);
      if (target == "python")
        return emitPy(prog, outPath, opts);
      if (target == "triton")
        return emitTriton(prog, outPath, opts);
      note(opts.verbose, "unknown target " + target);
      return false;
    } catch (const std::exception &e) {
      note(opts.verbose, std::string("compilation failed: ") + e.what());
      return false;
    }
  }

} // namespace refractir
