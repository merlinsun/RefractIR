#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "analysis/dominators.hpp"
#include "analysis/loop_info.hpp"
#include "analysis/structured_lowering.hpp"
#include "analysis/structurizer.hpp"
#include "analysis/tensor.hpp"
#include "ast/ast_dumper.hpp"
#include "backend/c_backend.hpp"
#include "backend/emit.hpp"
#include "backend/py_backend.hpp"
#include "backend/wasm_backend.hpp"
#include "backend/wasm_vec_lowering.hpp"
#include "cxxopts.hpp"
#include "error.hpp"
#include "frontend/link_resolver.hpp"
#include "frontend/pipeline.hpp"

int main(int argc, char **argv) {
  using namespace refractir;

  cxxopts::Options options("symirc", "RefractIR Compiler");

  // clang-format off
  options.add_options()
    ("input", "Input .sir file", cxxopts::value<std::string>())
    ("o,output", "Output file (default: stdout)", cxxopts::value<std::string>())
    ("target", "Backend target (c, wasm, python, triton)", cxxopts::value<std::string>()->default_value("c"))
    ("tensor-profile-version", "Tensor profile version: v1|v2|v3|v4", cxxopts::value<std::string>()->default_value("v1"))
    ("tensor-layout-manifest", "Triton target: JSON file with physical tensor buffer layouts", cxxopts::value<std::string>())
    ("entry", "Triton entry function", cxxopts::value<std::string>()->default_value("kernel"))
    ("tensor-softmax-lowering", "Triton softmax lowering: direct|decomposed", cxxopts::value<std::string>()->default_value("direct"))
    ("keep-require", "Triton target: reject emission when source require checks are retained", cxxopts::value<bool>()->default_value("false"))
    ("require-reducible", "Reject functions with irreducible control flow", cxxopts::value<bool>()->default_value("false"))
    ("structured-lowering", "Emit structured control flow (loops/ifs) instead of goto; implies --require-reducible (C target)", cxxopts::value<bool>()->default_value("false"))
    ("dump-ast", "Dump AST to stdout and exit", cxxopts::value<bool>()->default_value("false"))
    ("dump-domtree", "Dump per-function dominator trees to stdout and exit", cxxopts::value<bool>()->default_value("false"))
    ("dump-loops", "Dump per-function loop nesting forests to stdout and exit", cxxopts::value<bool>()->default_value("false"))
    ("dump-control-tree", "Dump per-function structured control trees to stdout and exit (implies --require-reducible)", cxxopts::value<bool>()->default_value("false"))
    ("dump-lowered-tree", "Dump per-function control trees after structured lowering and exit (implies --require-reducible)", cxxopts::value<bool>()->default_value("false"))
    ("w", "Inhibit all warning messages", cxxopts::value<bool>()->default_value("false"))
    ("Werror", "Make all warnings into errors", cxxopts::value<bool>()->default_value("false"))
    ("no-module-tags", "Omit (module ...) tags in WASM output", cxxopts::value<bool>()->default_value("false"))
    ("no-require", "Omit require checks from emitted code (useful for compiler testing)", cxxopts::value<bool>()->default_value("false"))
    ("no-ub-guards", "Omit dynamic undefined-behavior guards (null/OOB, div/rem-by-zero, FP finiteness, intrinsic preconditions). Sound only for known-UB-free programs; value semantics are unaffected", cxxopts::value<bool>()->default_value("false"))
    ("vec-lowering", "Vector lowering strategy: vecext|scalars|array|structscalars|structarray (default: target specific)", cxxopts::value<std::string>())
    ("I", "Include path for resolving link-form `decl`s (may repeat)", cxxopts::value<std::vector<std::string>>())
    ("split-by-source", "C target: emit one <stem>.c per source file plus common.h into the directory given by -o (instead of a single bundled .c)", cxxopts::value<bool>()->default_value("false"))
    ("emit-main", "Do not mangle @main function in emitted target code", cxxopts::value<bool>()->default_value("false"))
    ("h,help", "Print usage");
  options.parse_positional({"input"});
  // clang-format on

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  if (!result.count("input")) {
    std::cerr << "Error: No input file specified." << std::endl;
    std::cerr << options.help() << std::endl;
    return 1;
  }

  std::string inputPath = result["input"].as<std::string>();
  std::ifstream ifs(inputPath);
  if (!ifs) {
    std::cerr << "Error: Could not open file " << inputPath << "\n";
    return 1;
  }
  std::stringstream ss;
  ss << ifs.rdbuf();
  std::string src = ss.str();

  try {
    // 1. Frontend
    Program prog = parseSource(src);

    // 1b. -I link-form resolution.
    std::vector<Program> libs;
    if (result.count("I")) {
      libs = loadIncludeDirs(result["I"].as<std::vector<std::string>>());
    }
    resolveLinkDecls(prog, libs);

    if (result["dump-ast"].as<bool>()) {
      ASTDumper dumper(std::cout);
      dumper.dump(prog);
      return 0;
    }

    std::string target = result["target"].as<std::string>();

    // 2. Analysis
    CheckOptions checkOpts;
    checkOpts.tensorProfile =
        parseTensorProfileVersion(result["tensor-profile-version"].as<std::string>());
    // The structurizer is only total on reducible CFGs, so the
    // control-tree dump flags, --structured-lowering, and the python
    // target (which has no goto to fall back on) imply the check.
    checkOpts.reducibility = result["require-reducible"].as<bool>() ||
                             result["structured-lowering"].as<bool>() ||
                             result["dump-control-tree"].as<bool>() ||
                             result["dump-lowered-tree"].as<bool>() || target == "python";
    checkOpts.warningsAreErrors = result["Werror"].as<bool>();

    bool werror = checkOpts.warningsAreErrors;
    bool nowarn = result["w"].as<bool>();

    DiagBag diags;
    if (!checkProgram(prog, diags, checkOpts)) {
      std::cerr << "Errors:\n";
      for (const auto &d: diags.diags) {
        // Notes only ever accompany errors (e.g. ReducibilityCheck points
        // at the multi-entry loop header), so print them alongside.
        if (d.level == DiagLevel::Error || d.level == DiagLevel::Note ||
            (werror && d.level == DiagLevel::Warning)) {
          printMessage(std::cerr, src, d.span, d.message, d.level);
        }
      }
      return ExitCode::StaticError;
    }

    const TensorConsumer tensorConsumer = target == "c"        ? TensorConsumer::C
                                          : target == "wasm"   ? TensorConsumer::Wasm
                                          : target == "python" ? TensorConsumer::Python
                                          : target == "triton" ? TensorConsumer::Triton
                                                               : TensorConsumer::Reference;
    if (!checkTensorCapabilities(
            prog, tensorConsumer, tensorProfile(checkOpts.tensorProfile), diags
        )) {
      std::cerr << "Errors:\n";
      for (const auto &d: diags.diags)
        if (d.level == DiagLevel::Error)
          printMessage(std::cerr, src, d.span, d.message, d.level);
      return ExitCode::StaticError;
    }

    // Print warnings
    if (!nowarn) {
      for (const auto &d: diags.diags) {
        if (d.level == DiagLevel::Warning) {
          printMessage(std::cerr, src, d.span, d.message, d.level);
        }
      }
    }

    // Analysis dump flags: print after the pipeline has validated the
    // program (so every CFG is well-formed) and exit without emitting.
    bool dumpDomtree = result["dump-domtree"].as<bool>();
    bool dumpLoops = result["dump-loops"].as<bool>();
    bool dumpControlTree = result["dump-control-tree"].as<bool>();
    bool dumpLoweredTree = result["dump-lowered-tree"].as<bool>();
    if (dumpDomtree || dumpLoops || dumpControlTree || dumpLoweredTree) {
      bool first = true;
      for (const auto &f: prog.funs) {
        if (!first)
          std::cout << "\n";
        first = false;
        CFG cfg = CFG::build(f, diags);
        DomTree dt = DomTree::build(cfg);
        if (dumpDomtree)
          dt.dump(std::cout, cfg, f.name.name);
        if (dumpLoops || dumpControlTree || dumpLoweredTree) {
          LoopInfo li = LoopInfo::build(cfg, dt);
          if (dumpLoops)
            li.dump(std::cout, cfg, f.name.name);
          if (dumpControlTree)
            Structurizer::build(f, cfg, dt, li).dump(std::cout, cfg, f.name.name);
          if (dumpLoweredTree) {
            ControlTree lowered =
                StructuredLowering::run(Structurizer::build(f, cfg, dt, li), f, cfg);
            lowered.dump(std::cout, cfg, f.name.name);
          }
        }
      }
      return 0;
    }

    if (target == "triton") {
      EmitOptions tensorOptions;
      tensorOptions.verbose = true;
      tensorOptions.entry = result["entry"].as<std::string>();
      tensorOptions.tensorSoftmaxLowering = result["tensor-softmax-lowering"].as<std::string>();
      tensorOptions.tensorProfile = checkOpts.tensorProfile;
      if (result.count("tensor-layout-manifest"))
        tensorOptions.tritonBufferLayouts =
            readTritonBufferLayouts(result["tensor-layout-manifest"].as<std::string>());
      tensorOptions.keepRequire = result["keep-require"].as<bool>();
      tensorOptions.structuredLowering = result["structured-lowering"].as<bool>();
      tensorOptions.splitBySource = result["split-by-source"].as<bool>();
      if (result.count("vec-lowering"))
        tensorOptions.vecLowering = result["vec-lowering"].as<std::string>();
      const auto outPath = result.count("output")
                               ? std::filesystem::path(result["output"].as<std::string>())
                               : std::filesystem::path(inputPath).replace_extension(".py");
      if (!emitTriton(prog, outPath, tensorOptions))
        return 1;
      return 0;
    }

    // 3. Backend
    std::ostream *outStream = &std::cout;
    std::ofstream ofs;

    bool splitBySource = result.count("split-by-source") && result["split-by-source"].as<bool>();
    if (result.count("output") && !splitBySource) {
      // In split-by-source mode the -o argument is a directory; the
      // split path opens its own per-file streams. Skip opening a
      // single-file output here.
      std::string outPath = result["output"].as<std::string>();
      ofs.open(outPath);
      if (!ofs) {
        std::cerr << "Error: Could not open output file " << outPath << "\n";
        return 1;
      }
      outStream = &ofs;
    }

    bool noRequire = result["no-require"].as<bool>();
    bool noUbGuards = result["no-ub-guards"].as<bool>();
    bool emitMain = result["emit-main"].as<bool>();
    bool structuredLowering = result["structured-lowering"].as<bool>();
    // The python backend always emits structured control flow; the flag
    // reconstructs block/loop/if for the C and wasm targets (both goto-
    // /dispatch-based by default).
    std::string vlName;
    if (result.count("vec-lowering") > 0) {
      vlName = result["vec-lowering"].as<std::string>();
      // The python backend supports every strategy except vecext (no
      // native SIMD value type); the wasm backend only "array". C
      // accepts all five (validated by makeCVecLowering below).
      if (target == "python" && !makePyVecLowering(vlName)) {
        std::cerr << "Error: python target does not support vector lowering strategy '" << vlName
                  << "' (try array|scalars|structscalars|structarray)\n";
        return 1;
      }
      if (target == "wasm" && !makeWasmVecLowering(vlName)) {
        std::cerr << "Error: Target 'wasm' does not support vector lowering strategy '" << vlName
                  << "' (try vecext|array|scalars)\n";
        return 1;
      }
    } else {
      // C and WASM default to their native SIMD value type ("vecext" —
      // v128 on WASM); python has none and defaults to lane lists.
      vlName = (target == "python") ? "array" : "vecext";
    }

    if (target == "c") {
      // --split-by-source: emit one <stem>.c per source file
      // + common.h into the directory specified by -o.
      if (result["split-by-source"].as<bool>()) {
        if (!result.count("output")) {
          std::cerr << "Error: --split-by-source requires -o <directory>\n";
          return 1;
        }
        std::string outDir = result["output"].as<std::string>();
        // We don't write to outStream in this branch; close the file if
        // -o opened one (it pointed at outDir as if it were a file).
        if (ofs.is_open())
          ofs.close();
        // Compute the primary file's stem from the input path.
        std::string primaryStem = std::filesystem::path(inputPath).stem().string();
        CBackend cb(std::cout); // sink; unused — emitSplit opens its own streams
        cb.setNoRequire(noRequire);
        cb.setNoUbGuards(noUbGuards);
        cb.setNoMainMangle(emitMain);
        cb.setStructuredLowering(structuredLowering);
        // Set up the vector-lowering strategy.
        auto vl = makeCVecLowering(vlName);
        if (!vl) {
          std::cerr << "Error: unknown --vec-lowering '" << vlName << "'\n";
          return 1;
        }
        cb.setVecLowering(std::move(vl));
        // No per-file `wrote …` chatter: a caller driving this path over
        // many programs wants a clean stderr so its own per-program log
        // line is the only thing the user sees. The returned list is
        // dropped, and the caller inspects the output directory if it
        // cares about which files landed.
        (void) cb.emitSplit(prog, outDir, primaryStem);
      } else {
        CBackend cb(*outStream);
        cb.setNoRequire(noRequire);
        cb.setNoUbGuards(noUbGuards);
        cb.setNoMainMangle(emitMain);
        cb.setStructuredLowering(structuredLowering);
        // Set up the vector-lowering strategy.
        auto vl = makeCVecLowering(vlName);
        if (!vl) {
          std::cerr << "Error: unknown --vec-lowering '" << vlName
                    << "' (try vecext|scalars|array|structscalars|structarray)\n";
          return 1;
        }
        cb.setVecLowering(std::move(vl));
        cb.emit(prog);
      }
    } else if (target == "wasm") {
      WasmBackend wb(*outStream);
      if (result.count("no-module-tags")) {
        wb.setNoModuleTags(result["no-module-tags"].as<bool>());
      }
      wb.setNoRequire(noRequire);
      wb.setNoUbGuards(noUbGuards);
      wb.setNoMainMangle(emitMain);
      wb.setStructuredLowering(structuredLowering);
      // Vector-lowering strategy (validated above).
      wb.setVecLowering(makeWasmVecLowering(vlName));
      wb.emit(prog);
    } else if (target == "python") {
      PyBackend pb(*outStream);
      pb.setNoRequire(noRequire);
      pb.setNoUbGuards(noUbGuards);
      pb.setNoMainMangle(emitMain);
      pb.setVecLowering(makePyVecLowering(vlName));
      pb.emit(prog);
    } else {
      std::cerr << "Error: Unsupported target: " << target << "\n";
      return 1;
    }

  } catch (const LexError &e) {
    printMessage(std::cerr, src, e.span, e.what(), DiagLevel::Error);
    return ExitCode::LexError;
  } catch (const ParseError &e) {
    printMessage(std::cerr, src, e.span, e.what(), DiagLevel::Error);
    return ExitCode::ParseError;
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return ExitCode::Error;
  }

  return 0;
}
