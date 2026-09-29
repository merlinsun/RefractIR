#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ast/ast_dumper.hpp"
#include "ast/sir_printer.hpp"
#include "analysis/tensor.hpp"
#include "cxxopts.hpp"
#include "error.hpp"
#include "frontend/link_resolver.hpp"
#include "frontend/pipeline.hpp"
#include "solver/solved_header.hpp"
#include "solver/solver.hpp"
#if defined(USE_ALIVESMT)
#include "solver/alive_impl.hpp"
#elif defined(USE_BITWUZLA)
#include "solver/bitwuzla_impl.hpp"
#endif

using namespace refractir;

std::vector<std::string> split(const std::string &s, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

namespace {

  // One entry-parameter witness as a JSON record. `--emit-model` embeds the
  // record and `--emit-tensor-params` writes it standalone; the two agree
  // byte for byte. An f32 leaf goes out through refractir::formatDouble, the
  // canonical bit-exact serializer (docs/float.md §9).
  void writeTensorParamWitness(
      std::ostream &out, const SymbolicExecutor::Result::TensorParamWitness &witness
  ) {
    out << "{\"shape\": [";
    for (std::size_t i = 0; i < witness.shape.size(); ++i) {
      if (i)
        out << ", ";
      out << witness.shape[i];
    }
    out << "], \"element\": \"" << (witness.element == TensorElement::F32 ? "f32" : "i32")
        << "\", \"leaves\": [";
    for (std::size_t i = 0; i < witness.leaves.size(); ++i) {
      if (i)
        out << ", ";
      if (std::holds_alternative<int64_t>(witness.leaves[i]))
        out << std::get<int64_t>(witness.leaves[i]);
      else
        out << refractir::formatDouble(std::get<double>(witness.leaves[i]));
    }
    out << "]}";
  }

} // namespace

/// The terminal result with the reason the backend gave for it, so a consumer
/// can tell a reported timeout from an undecided query without parsing stderr
/// or inferring from the exit code. The status names match the console output;
/// `reason` is present only for UNKNOWN.
void writeOutcomeRecord(
    const std::string &path, const std::string &status, smt::UnknownReason reason
) {
  std::ofstream out(path);
  out << "{\"status\":\"" << status << "\"";
  if (status == "unknown")
    out << ",\"reason\":\"" << smt::unknownReasonName(reason) << "\"";
  out << "}\n";
  // The stream buffers, so its state is only final once it is closed. Checking
  // before that reports success for a record whose flush failed, which is how
  // a full device leaves an empty file behind.
  out.close();
  if (!out)
    throw std::runtime_error("failed to write outcome record: " + path);
}

int main(int argc, char **argv) {
  cxxopts::Options options("symirsolve", "RefractIR SMT-based Concretizer");

  // clang-format off
  options.add_options()
    ("input", "Input .sir file", cxxopts::value<std::string>())
    ("main", "Function to concretize", cxxopts::value<std::string>()->default_value("@main"))
    ("path", "Comma-separated block labels for execution path (acts as prefix if --sample is used)", cxxopts::value<std::string>())
    ("sample", "Number of paths to sample randomly", cxxopts::value<uint32_t>())
    ("max-path-len", "Maximum random path length", cxxopts::value<uint32_t>()->default_value("100"))
    ("require-terminal", "Force paths to reach 'ret' by appending shortest path if needed", cxxopts::value<bool>()->default_value("false"))
    ("o,output", "Output .sir file", cxxopts::value<std::string>())
    ("dump-ast", "Dump concretized AST to stdout", cxxopts::value<bool>()->default_value("false"))
    ("timeout-ms", "Solver timeout in milliseconds", cxxopts::value<uint32_t>()->default_value("0"))
    ("seed", "Solver seed", cxxopts::value<uint32_t>()->default_value("0"))
    ("j,num-threads", "Number of threads for parallel solving (0 = hardware concurrency)", cxxopts::value<uint32_t>()->default_value("1"))
    ("num-smt-threads", "Number of threads for the SMT solver backend (Bitwuzla/Z3 internal parallelism)", cxxopts::value<uint32_t>()->default_value("1"))
    ("emit-model", "Emit symbol assignments to a JSON-like file", cxxopts::value<std::string>())
    ("emit-outcome", "Emit the terminal result with its typed reason to a JSON file", cxxopts::value<std::string>())
    ("emit-tensor-params", "Emit entry tensor-parameter witnesses to a JSON file", cxxopts::value<std::string>())
    ("tensor-profile-version", "Tensor profile to check against (v1|v2|v3|v4)", cxxopts::value<std::string>()->default_value("v1"))
    ("tensor-param-domain", "Bound every entry-parameter integer leaf to [lo,hi] (lo,hi)", cxxopts::value<std::string>())
    ("sym", "Fix a symbol to a value (name=val)", cxxopts::value<std::vector<std::string>>())
    ("I", "Include path for resolving link-form `decl`s (may repeat)", cxxopts::value<std::vector<std::string>>())
    ("require-ub", "Force at least one UB to be triggered on the chosen path", cxxopts::value<bool>()->default_value("false"))
    ("require-nonterm", "Solve for a lasso path (final block = loop header) whose header state recurs after one lap, so the program diverges UB-free", cxxopts::value<bool>()->default_value("false"))
    ("h,help", "Print usage");
  options.parse_positional({"input"});
  // clang-format on

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  if (!result.count("input") || (!result.count("path") && !result.count("sample"))) {
    std::cerr << "Error: input and either --path or --sample are required." << std::endl;
    std::cerr << options.help() << std::endl;
    return 1;
  }

  std::string inputPath = result["input"].as<std::string>();
  std::string funcName = result["main"].as<std::string>();

  std::vector<std::string> path;
  if (result.count("path")) {
    std::string path_str = result["path"].as<std::string>();
    if (path_str.size() >= 2 && ((path_str.front() == '\'' && path_str.back() == '\'') ||
                                 (path_str.front() == '\"' && path_str.back() == '\"'))) {
      path_str = path_str.substr(1, path_str.size() - 2);
    }
    path = split(path_str, ',');
  }

  std::unordered_map<std::string, int64_t> fixedSyms;
  if (result.count("sym")) {
    for (const auto &s: result["sym"].as<std::vector<std::string>>()) {
      auto eq = s.find('=');
      if (eq != std::string::npos) {
        fixedSyms[s.substr(0, eq)] = parseIntegerLiteral(s.substr(eq + 1));
      }
    }
  }

  std::ifstream ifs(inputPath);
  if (!ifs) {
    std::cerr << "Error: Could not open file " << inputPath << std::endl;
    return 1;
  }
  std::stringstream ss;
  ss << ifs.rdbuf();
  std::string src = ss.str();

  try {
    Program prog = parseSource(src);

    // -I link-form resolution.
    std::vector<Program> libs;
    if (result.count("I")) {
      libs = loadIncludeDirs(result["I"].as<std::vector<std::string>>());
    }
    resolveLinkDecls(prog, libs);

    // The symbolic executor needs a well-typed program, not a tidy one: it
    // explores one path at a time, so an unreachable block is never visited
    // and an uninitialized read is caught as UB along the path that makes it.
    CheckOptions checkOpts;
    checkOpts.functionAnalyses = false;
    checkOpts.tensorProfile =
        parseTensorProfileVersion(result["tensor-profile-version"].as<std::string>());

    DiagBag diags;
    if (!checkProgram(prog, diags, checkOpts)) {
      std::cerr << "Errors in input program:" << std::endl;
      for (const auto &d: diags.diags) {
        if (d.level == DiagLevel::Error)
          printMessage(std::cerr, src, d.span, d.message, d.level);
      }
      return ExitCode::StaticError;
    }

    // A tensor program outside the selected profile fails here rather than
    // reaching the solver, exactly as it does for every other consumer.
    if (!checkTensorCapabilities(
            prog, TensorConsumer::ExactSolver, tensorProfile(checkOpts.tensorProfile), diags
        )) {
      std::cerr << "Errors in input program:" << std::endl;
      for (const auto &d: diags.diags) {
        if (d.level == DiagLevel::Error)
          printMessage(std::cerr, src, d.span, d.message, d.level);
      }
      return ExitCode::StaticError;
    }

    SymbolicExecutor::Config config;
    config.timeout_ms = result["timeout-ms"].as<uint32_t>();
    config.seed = result["seed"].as<uint32_t>();
    config.num_threads = result["num-threads"].as<uint32_t>();
    config.num_smt_threads = result["num-smt-threads"].as<uint32_t>();
    if (result.count("tensor-param-domain")) {
      const std::string domain = result["tensor-param-domain"].as<std::string>();
      const auto comma = domain.find(',');
      if (comma == std::string::npos) {
        std::cerr << "Error: --tensor-param-domain expects <lo>,<hi>." << std::endl;
        return 1;
      }
      try {
        const int64_t lo = parseIntegerLiteral(domain.substr(0, comma));
        const int64_t hi = parseIntegerLiteral(domain.substr(comma + 1));
        if (lo > hi) {
          std::cerr << "Error: --tensor-param-domain requires lo <= hi." << std::endl;
          return 1;
        }
        config.tensorParamDomain = std::make_pair(lo, hi);
      } catch (const std::exception &e) {
        std::cerr << "Error: --tensor-param-domain: " << e.what() << std::endl;
        return 1;
      }
    }
    if (result["require-ub"].as<bool>()) {
      config.mode = SolvingMode::RequireUB;
    }
    if (result["require-nonterm"].as<bool>()) {
      config.mode = SolvingMode::RequireNonterm;
    }

#if defined(USE_ALIVESMT)
    // AliveSMT (Z3) uses a global context that is not thread-safe.
    // Force single-threaded execution for AliveSMT backend.
    if (config.num_threads > 1) {
      std::cerr << "Warning: AliveSMT backend does not support multi-threading. "
                << "Forcing single-threaded execution (num_threads=1).\n";
      config.num_threads = 1;
    }
#endif

    auto solverFactory =
        [](const SymbolicExecutor::Config &cfg) -> std::unique_ptr<refractir::smt::ISolver> {
#if defined(USE_ALIVESMT)
      return std::make_unique<refractir::solver::AliveSolver>(
          cfg.timeout_ms, cfg.seed, cfg.num_smt_threads
      );
#elif defined(USE_BITWUZLA)
      return std::make_unique<refractir::solver::BitwuzlaSolver>(
          cfg.timeout_ms, cfg.seed, cfg.num_smt_threads
      );
#else
      (void) cfg;
      throw std::runtime_error("No solver backend available");
#endif
    };

    SymbolicExecutor executor(prog, config, solverFactory);
    SymbolicExecutor::Result res;
    if (result.count("sample")) {
      res = executor.sample(
          funcName, result["sample"].as<uint32_t>(), result["max-path-len"].as<uint32_t>(),
          result["require-terminal"].as<bool>(), path, fixedSyms
      );
    } else {
      res = executor.solve(funcName, path, fixedSyms);
    }

    if (res.sat) {
      if (result.count("emit-outcome"))
        writeOutcomeRecord(
            result["emit-outcome"].as<std::string>(), "sat", smt::UnknownReason::Unspecified
        );
      std::cout << "SAT" << std::endl;

      // Sorted by parameter name: the witness map is unordered, and a file
      // whose field order varies run to run cannot be compared or hashed.
      // Existing symbol entries keep their current order.
      std::vector<std::pair<std::string, SymbolicExecutor::Result::TensorParamWitness>>
          tensorParams(res.tensorParamModel.begin(), res.tensorParamModel.end());
      std::sort(
          tensorParams.begin(), tensorParams.end(),
          [](const auto &a, const auto &b) { return a.first < b.first; }
      );

      if (result.count("emit-model")) {
        std::ofstream mfs(result["emit-model"].as<std::string>());
        mfs << "{\n";
        mfs << "  \"" << funcName << "\": {\n";
        bool first = true;
        for (const auto &[name, val]: res.model) {
          if (!first)
            mfs << ",\n";
          mfs << "    \"" << name << "\": ";
          if (std::holds_alternative<int64_t>(val))
            mfs << std::get<int64_t>(val);
          else
            // Bit-exact serialization — see refractir::formatDouble in
            // ast.hpp. Default `<<` would silently truncate to ~6
            // significant digits and lose the model's exact bits.
            mfs << refractir::formatDouble(std::get<double>(val));
          first = false;
        }
        for (const auto &[name, values]: res.tensorModel) {
          if (!first)
            mfs << ",\n";
          mfs << "    \"" << name << "\": [";
          for (size_t i = 0; i < values.size(); ++i) {
            if (i)
              mfs << ", ";
            if (std::holds_alternative<int64_t>(values[i]))
              mfs << std::get<int64_t>(values[i]);
            else
              mfs << refractir::formatDouble(std::get<double>(values[i]));
          }
          mfs << "]";
          first = false;
        }
        for (const auto &[name, values]: res.vecModel) {
          if (!first)
            mfs << ",\n";
          mfs << "    \"" << name << "\": [";
          for (size_t i = 0; i < values.size(); ++i) {
            if (i)
              mfs << ", ";
            if (std::holds_alternative<int64_t>(values[i]))
              mfs << std::get<int64_t>(values[i]);
            else
              mfs << refractir::formatDouble(std::get<double>(values[i]));
          }
          mfs << "]";
          first = false;
        }
        for (const auto &[name, witness]: tensorParams) {
          if (!first)
            mfs << ",\n";
          mfs << "    \"" << name << "\": ";
          writeTensorParamWitness(mfs, witness);
          first = false;
        }
        mfs << "\n  }\n";
        mfs << "}\n";
      }

      if (result.count("emit-tensor-params")) {
        const std::string path = result["emit-tensor-params"].as<std::string>();
        std::ofstream pfs(path);
        if (!pfs) {
          std::cerr << "Error: Could not open output file " << path << std::endl;
          return 1;
        }
        pfs << "{\n";
        pfs << "  \"" << funcName << "\": {";
        bool firstWitness = true;
        for (const auto &[name, witness]: tensorParams) {
          pfs << (firstWitness ? "\n" : ",\n");
          pfs << "    \"" << name << "\": ";
          writeTensorParamWitness(pfs, witness);
          firstWitness = false;
        }
        pfs << (firstWitness ? "}\n}\n" : "\n  }\n}\n");
      }

      if (result["dump-ast"].as<bool>()) {
        std::unordered_map<std::string, int64_t> intModel;
        for (const auto &[name, val]: res.model) {
          if (std::holds_alternative<int64_t>(val))
            intModel[name] = std::get<int64_t>(val);
        }
        ASTDumper dumper(std::cout, intModel);
        dumper.dump(prog);
      }

      if (result.count("output")) {
        std::ofstream ofs(result["output"].as<std::string>());
        if (!ofs) {
          std::cerr << "Error: Could not open output file " << result["output"].as<std::string>()
                    << std::endl;
          return 1;
        }
        // The values the solver chose, so a consumer can pin the entry point
        // without solving again. See writeSolvedHeader.
        writeSolvedHeader(
            ofs, res.paramModel,
            res.retModel.has_value() ? formatModelValue(*res.retModel) : std::string{}
        );
        SIRPrinter printer(ofs, res.model, res.vecModel, res.tensorModel);
        printer.print(prog);
      }

    } else if (res.unsat) {
      if (result.count("emit-outcome"))
        writeOutcomeRecord(
            result["emit-outcome"].as<std::string>(), "unsat", smt::UnknownReason::Unspecified
        );
      std::cout << "UNSAT" << std::endl;
      return 1;
    } else {
      if (result.count("emit-outcome"))
        writeOutcomeRecord(result["emit-outcome"].as<std::string>(), "unknown", res.unknownReason);
      std::cout << "UNKNOWN" << std::endl;
      return 1;
    }

  } catch (const LexError &e) {
    printMessage(std::cerr, src, e.span, e.what(), DiagLevel::Error);
    return ExitCode::LexError;
  } catch (const ParseError &e) {
    printMessage(std::cerr, src, e.span, e.what(), DiagLevel::Error);
    return ExitCode::ParseError;
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return ExitCode::Error;
  }

  return 0;
}
