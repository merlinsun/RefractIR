#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "analysis/cfg.hpp"
#include "analysis/tensor.hpp"
#include "cxxopts.hpp"
#include "error.hpp"
#include "frontend/link_resolver.hpp"
#include "frontend/pipeline.hpp"
#include "interp/interpreter.hpp"
#include "interp/tensor.hpp"
#include "interp/tensor_io.hpp"
#include "interp/tensor_numeric.hpp"

int main(int argc, char **argv) {
  using namespace refractir;

  cxxopts::Options options("symiri", "RefractIR Reference Interpreter");

  // clang-format off
  options.add_options()
    ("input", "Input .sir file", cxxopts::value<std::string>())
    ("args", "Argument values for the entry function's parameters", cxxopts::value<std::vector<std::string>>())
    ("main", "Entry function to execute", cxxopts::value<std::string>()->default_value("@main"))
    ("entry", "Entry function for typed tensor input/output", cxxopts::value<std::string>())
    ("tensor-describe", "Describe a checked entry's typed buffer signature", cxxopts::value<bool>()->default_value("false"))
    ("tensor-input", "Little-endian typed input buffer", cxxopts::value<std::string>())
    ("tensor-output", "Little-endian typed result buffer", cxxopts::value<std::string>())
    ("tensor-reference-bits", "Numerical tensor reference precision (128 or 256)", cxxopts::value<unsigned>()->default_value("128"))
    ("tensor-reference-stability", "Run tensor references at 128 and 256 bits and compare each atom", cxxopts::value<bool>()->default_value("false"))
    ("sym", "Bind a symbol (name=value)", cxxopts::value<std::vector<std::string>>())
    ("check", "Check semantics only (do not execute)", cxxopts::value<bool>()->default_value("false"))
    ("dump-trace", "Dump executed blocks and variable updates", cxxopts::value<bool>()->default_value("false"))
    ("dump-call", "Dump function calls only ('> @f(args)' enters a function, '< @f = result' exits it)", cxxopts::value<bool>()->default_value("false"))
    ("max-bbl-steps", "Abort after entering this many basic blocks (0 = unlimited); bounds a possibly non-terminating program", cxxopts::value<uint64_t>()->default_value("0"))
    ("w", "Inhibit all warning messages", cxxopts::value<bool>()->default_value("false"))
    ("Werror", "Make all warnings into errors", cxxopts::value<bool>()->default_value("false"))
    ("I", "Include path for resolving link-form `decl`s (may repeat)", cxxopts::value<std::vector<std::string>>())
    ("h,help", "Print usage");
  options.parse_positional({"input", "args"});
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
  std::string mainFunc = result["main"].as<std::string>();
  Interpreter::SymBindings symBindings;

  if (result.count("sym")) {
    // cxxopts splits each occurrence's value on commas, so a vector
    // lane list like `--sym %?v=1,2,3,4` arrives as ["%?v=1", "2",
    // "3", "4"]. Entries without '=' are comma-continuations of the
    // previous binding — unambiguous, since values are numeric
    // literals and never contain '='.
    std::vector<std::string> binds;
    for (const auto &piece: result["sym"].as<std::vector<std::string>>()) {
      if (piece.find('=') != std::string::npos) {
        binds.push_back(piece);
      } else if (!binds.empty()) {
        binds.back() += "," + piece;
      } else {
        std::cerr << "Error: Invalid symbol binding format (expected name=value): " << piece
                  << "\n";
        return 1;
      }
    }
    for (const auto &bind: binds) {
      size_t eq = bind.find('=');
      std::string name = bind.substr(0, eq);
      std::string valStr = bind.substr(eq + 1);
      // A single scalar, or a comma-joined lane list for vector syms
      // (one value per lane; a single value splats).
      std::vector<Interpreter::SymScalar> vals;
      try {
        size_t pos = 0;
        while (true) {
          size_t comma = valStr.find(',', pos);
          vals.push_back(parseNumberLiteral(
              valStr.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos)
          ));
          if (comma == std::string::npos)
            break;
          pos = comma + 1;
        }
      } catch (...) {
        std::cerr << "Error: Invalid number value for symbol " << name << ": " << valStr << "\n";
        return 1;
      }
      symBindings[name] = std::move(vals);
    }
  }

  std::ifstream ifs(inputPath);
  if (!ifs) {
    std::cerr << "Error: Could not open file " << inputPath << "\n";
    return 1;
  }
  std::stringstream ss;
  ss << ifs.rdbuf();
  std::string src = ss.str();

  try {
    // 1. Frontend: Lex & Parse
    Program prog = parseSource(src);

    // 1b. Load -I libraries and resolve link-form `decl`s.
    std::vector<Program> libs;
    if (result.count("I")) {
      libs = loadIncludeDirs(result["I"].as<std::vector<std::string>>());
    }
    resolveLinkDecls(prog, libs);

    // 2. Analysis
    CheckOptions checkOpts;
    checkOpts.warningsAreErrors = result["Werror"].as<bool>();

    bool werror = checkOpts.warningsAreErrors;
    bool nowarn = result["w"].as<bool>();
    const bool describeTensor = result["tensor-describe"].as<bool>();

    DiagBag diags;
    if (!checkProgram(prog, diags, checkOpts)) {
      std::cerr << "Errors:\n";
      for (const auto &d: diags.diags) {
        if (d.level == DiagLevel::Error || (werror && d.level == DiagLevel::Warning)) {
          printMessage(std::cerr, src, d.span, d.message, d.level);
        }
      }
      return ExitCode::StaticError;
    }

    if (!checkTensorCapabilities(prog, TensorConsumer::Reference, diags)) {
      std::cerr << "Errors:\n";
      for (const auto &d: diags.diags)
        if (d.level == DiagLevel::Error)
          printMessage(std::cerr, src, d.span, d.message, d.level);
      return ExitCode::StaticError;
    }

    if (!nowarn && !describeTensor) {
      for (const auto &d: diags.diags) {
        if (d.level == DiagLevel::Warning) {
          printMessage(std::cout, src, d.span, d.message, d.level);
        }
      }
    }

    if (result["check"].as<bool>()) {
      return 0;
    }

    const bool hasTensorInput = result.count("tensor-input") != 0;
    const bool hasTensorOutput = result.count("tensor-output") != 0;
    const bool typedTensorRun = describeTensor || hasTensorInput || hasTensorOutput;
    if (typedTensorRun) {
      const std::string entryName =
          result.count("entry") ? result["entry"].as<std::string>() : mainFunc;
      if (describeTensor) {
        if (hasTensorInput || hasTensorOutput)
          throw std::runtime_error("--tensor-describe cannot be combined with tensor input/output");
        std::cout << describeTensorEntry(prog, entryName);
        return 0;
      }
      if (!hasTensorInput || !hasTensorOutput)
        throw std::runtime_error("--tensor-input and --tensor-output must be supplied together");

      setTensorReferenceBits(result["tensor-reference-bits"].as<unsigned>());

      std::string lookupName = entryName;
      if (lookupName.empty() || lookupName.front() != '@')
        lookupName.insert(lookupName.begin(), '@');
      const auto entry =
          std::find_if(prog.funs.begin(), prog.funs.end(), [&](const FunDecl &function) {
            return function.name.name == lookupName;
          });
      if (entry == prog.funs.end())
        throw std::runtime_error("Entry function not found: " + entryName);
      std::vector<TensorTypeInfo> parameterTypes;
      parameterTypes.reserve(entry->params.size());
      for (const auto &param: entry->params)
        parameterTypes.push_back(tensorTypeInfo(param.type));
      std::ifstream tensorInput(result["tensor-input"].as<std::string>(), std::ios::binary);
      if (!tensorInput)
        throw std::runtime_error("Could not open tensor input file");
      auto arguments = readTensorArguments(tensorInput, parameterTypes);
      const bool checkStability = result["tensor-reference-stability"].as<bool>();
      std::optional<RuntimeValue> value;
      if (checkStability) {
        std::vector<TensorTraceEntry> trace128;
        std::vector<TensorTraceEntry> trace256;
        const auto runWithTrace = [&](unsigned bits, std::vector<TensorTraceEntry> &trace) {
          setTensorReferenceBits(bits);
          setTensorTraceHook([&trace](const SourceSpan &span, const RuntimeValue &nodeValue) {
            trace.push_back(TensorTraceEntry{span, nodeValue});
          });
          try {
            Interpreter interpreter(prog);
            auto output = interpreter.runTyped(entryName, symBindings, arguments);
            setTensorTraceHook({});
            return output;
          } catch (...) {
            setTensorTraceHook({});
            throw;
          }
        };
        (void) runWithTrace(128, trace128);
        value = runWithTrace(256, trace256);
        if (!tensorTracesEqual(trace128, trace256))
          throw std::runtime_error(
              "oracle_unstable: tensor atom results differ at 128 and 256 bits"
          );
      } else {
        setTensorReferenceBits(result["tensor-reference-bits"].as<unsigned>());
        Interpreter interpreter(prog);
        value = interpreter.runTyped(entryName, symBindings, arguments);
      }
      if (!value)
        throw std::runtime_error("typed tensor entry returned no value");

      std::ostringstream encoded(std::ios::out | std::ios::binary);
      writeTensorResult(encoded, *value, tensorTypeInfo(entry->retType));
      const auto outputPath = std::filesystem::path(result["tensor-output"].as<std::string>());
      auto temporaryPath = outputPath;
      temporaryPath += ".tmp";
      try {
        std::ofstream output(temporaryPath, std::ios::binary | std::ios::trunc);
        if (!output)
          throw std::runtime_error("Could not open temporary tensor output file");
        const auto contents = encoded.str();
        output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        output.close();
        if (!output)
          throw std::runtime_error("Failed to write tensor output file");
        std::filesystem::rename(temporaryPath, outputPath);
      } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporaryPath, ignored);
        throw;
      }
      return 0;
    }

    // 4. Interpret
    std::vector<std::string> paramArgs;
    if (result.count("args"))
      paramArgs = result["args"].as<std::vector<std::string>>();
    Interpreter interp(prog);
    interp.setMaxBlockSteps(result["max-bbl-steps"].as<uint64_t>());
    interp.run(
        mainFunc, symBindings, paramArgs, result["dump-trace"].as<bool>(),
        result["dump-call"].as<bool>()
    );

  } catch (const StepLimitError &e) {
    std::cerr << "Step limit exceeded: " << e.what() << "\n";
    return ExitCode::Error;
  } catch (const UndefinedBehaviorError &e) {
    std::cerr << e.what() << "\n";
    return ExitCode::UndefinedBehavior;
  } catch (const RequireViolationError &e) {
    std::cerr << "Requirement failed: " << e.what() << "\n";
    return ExitCode::RequireViolation;
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
