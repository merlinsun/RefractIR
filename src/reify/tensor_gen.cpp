#include "reify/tensor_gen.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>

#include "analysis/tensor.hpp"
#include "analysis/type_utils.hpp"
#include "ast/sir_printer.hpp"
#include "frontend/diagnostics.hpp"
#include "frontend/pipeline.hpp"
#include "interp/interpreter.hpp"
#include "interp/tensor_io.hpp"
#include "interp/tensor_numeric.hpp"
#include "reify/tensor_case.hpp"
#include "solver/solver.hpp"
#if defined(USE_BITWUZLA)
#include "solver/bitwuzla_impl.hpp"
#elif defined(USE_ALIVESMT)
#include "solver/alive_impl.hpp"
#endif

namespace refractir::reify {
  namespace {

    struct AttemptRecord {
      std::uint32_t caseIndex = 0;
      std::uint32_t attempt = 0;
      std::string property;
      std::string outcome;
      std::string reason;
    };

    struct Candidate {
      std::string solverSource;
      std::string finalSource;
      std::vector<TensorCaseBuffer> inputs;
      std::vector<std::string> operators;
      std::string outputShape;
      TensorElement outputElement = TensorElement::I32;
    };

    std::string jsonString(std::string_view value) {
      std::string out{"\""};
      constexpr char hex[] = "0123456789abcdef";
      for (const unsigned char ch: value) {
        switch (ch) {
          case '"':
            out += "\\\"";
            break;
          case '\\':
            out += "\\\\";
            break;
          case '\b':
            out += "\\b";
            break;
          case '\f':
            out += "\\f";
            break;
          case '\n':
            out += "\\n";
            break;
          case '\r':
            out += "\\r";
            break;
          case '\t':
            out += "\\t";
            break;
          default:
            if (ch < 0x20) {
              out += "\\u00";
              out.push_back(hex[ch >> 4]);
              out.push_back(hex[ch & 15]);
            } else {
              out.push_back(static_cast<char>(ch));
            }
        }
      }
      out.push_back('"');
      return out;
    }

    void writeSummary(
        const std::filesystem::path &path, const TensorGenerationOptions &options,
        const TensorGenerationResult &result, const std::vector<AttemptRecord> &attempts
    ) {
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      if (!out)
        throw std::runtime_error("cannot create tensor generation summary: " + path.string());
      out << "{\"schema\":\"refractir.tensor-generation.v1\","
          << "\"profile\":\"tensor-v1\",\"status\":" << jsonString(result.status)
          << ",\"requested\":" << result.requested << ",\"emitted\":" << result.emitted
          << ",\"rejected\":" << result.rejected << ",\"seed\":" << options.seed
          << ",\"dtype\":" << jsonString(options.dtype)
          << ",\"property\":" << jsonString(options.property)
          << ",\"synthesis\":" << jsonString(options.synthesis)
          << ",\"target\":" << jsonString(options.target)
          << ",\"attempt_budget\":" << options.attemptBudget;
      if (!result.reason.empty())
        out << ",\"reason\":" << jsonString(result.reason);
      out << ",\"attempts\":[";
      for (std::size_t i = 0; i < attempts.size(); ++i) {
        const auto &attempt = attempts[i];
        if (i)
          out << ',';
        out << "{\"case\":" << attempt.caseIndex << ",\"attempt\":" << attempt.attempt
            << ",\"property_id\":" << jsonString(attempt.property)
            << ",\"outcome\":" << jsonString(attempt.outcome);
        if (!attempt.reason.empty())
          out << ",\"reason\":" << jsonString(attempt.reason);
        out << '}';
      }
      out << "]}\n";
      if (!out)
        throw std::runtime_error("failed to write tensor generation summary: " + path.string());
    }

    void appendLittleWord(std::vector<std::uint8_t> &bytes, std::uint32_t word) {
      bytes.push_back(static_cast<std::uint8_t>(word));
      bytes.push_back(static_cast<std::uint8_t>(word >> 8));
      bytes.push_back(static_cast<std::uint8_t>(word >> 16));
      bytes.push_back(static_cast<std::uint8_t>(word >> 24));
    }

    TensorElement scalarElement(const TypePtr &type) {
      if (TypeUtils::getIntBitWidth(type) == 32)
        return TensorElement::I32;
      if (TypeUtils::getFloatBitWidth(type) == 32)
        return TensorElement::F32;
      throw std::runtime_error("tensor generator only supports i32 and f32 inputs");
    }

    TensorTypeInfo typeInfo(const TypePtr &type) {
      if (const auto tensor = TypeUtils::asTensor(type))
        return {tensor->shape, scalarElement(tensor->elem)};
      if (TypeUtils::getIntBitWidth(type) == 32 || TypeUtils::getFloatBitWidth(type) == 32)
        return {{}, scalarElement(type)};
      throw std::runtime_error("tensor generator encountered a non-profile entry type");
    }

    const FunDecl &entryFunction(const Program &program) {
      for (const auto &function: program.funs)
        if (function.name.name == "@kernel")
          return function;
      throw std::runtime_error("generated tensor program has no @kernel entry");
    }

    Program checkedProgram(const std::string &source) {
      Program program = parseSource(source);
      DiagBag diagnostics;
      if (!checkProgram(program, diagnostics))
        throw std::runtime_error("generated tensor source failed frontend checking");
      if (!checkTensorCapabilities(program, TensorConsumer::Reference, diagnostics))
        throw std::runtime_error("generated tensor source is outside tensor-v1");
      return program;
    }

    std::vector<std::uint8_t> encodeI32(std::int32_t value) {
      std::vector<std::uint8_t> bytes;
      appendLittleWord(bytes, std::bit_cast<std::uint32_t>(value));
      return bytes;
    }

    std::vector<std::uint8_t> encodeI32Tensor(const std::vector<std::int32_t> &values) {
      std::vector<std::uint8_t> bytes;
      bytes.reserve(values.size() * 4);
      for (const auto value: values)
        appendLittleWord(bytes, std::bit_cast<std::uint32_t>(value));
      return bytes;
    }

    std::vector<std::uint8_t> encodeF32Tensor(const std::vector<float> &values) {
      std::vector<std::uint8_t> bytes;
      bytes.reserve(values.size() * 4);
      for (const auto value: values) {
        if (!std::isfinite(value))
          throw std::runtime_error("tensor generator produced a non-finite f32 input");
        appendLittleWord(bytes, std::bit_cast<std::uint32_t>(value));
      }
      return bytes;
    }

    std::vector<std::uint8_t> inputBytes(const std::vector<TensorCaseBuffer> &inputs) {
      std::vector<std::uint8_t> bytes;
      for (const auto &input: inputs)
        bytes.insert(bytes.end(), input.bytes.begin(), input.bytes.end());
      return bytes;
    }

    std::vector<RuntimeValue>
    decodedInputs(const Program &program, const std::vector<TensorCaseBuffer> &inputs) {
      const auto &entry = entryFunction(program);
      if (entry.params.size() != inputs.size())
        throw std::runtime_error("generated case input count differs from its signature");
      std::vector<TensorTypeInfo> types;
      types.reserve(entry.params.size());
      for (const auto &param: entry.params)
        types.push_back(typeInfo(param.type));
      const auto bytes = inputBytes(inputs);
      std::string binary;
      if (!bytes.empty())
        binary.assign(reinterpret_cast<const char *>(bytes.data()), bytes.size());
      std::istringstream input(binary, std::ios::binary);
      return readTensorArguments(input, types);
    }

    std::vector<std::uint8_t> runReference(
        const Program &program, const std::vector<TensorCaseBuffer> &inputs, unsigned precision,
        std::vector<TensorTraceEntry> *trace
    ) {
      const auto &entry = entryFunction(program);
      const auto resultType = typeInfo(entry.retType);
      auto args = decodedInputs(program, inputs);
      std::ostringstream output;
      Interpreter interpreter(program, output);
      if (trace) {
        setTensorTraceHook([trace](const SourceSpan &span, const RuntimeValue &value) {
          trace->push_back({span, value});
        });
      } else {
        setTensorTraceHook({});
      }
      setTensorReferenceBits(precision);
      try {
        const auto result = interpreter.runTyped("@kernel", {}, args);
        setTensorTraceHook({});
        if (!result)
          throw std::runtime_error("generated tensor entry did not return a value");
        std::ostringstream binary(std::ios::binary);
        writeTensorResult(binary, *result, resultType);
        const auto value = binary.str();
        return {value.begin(), value.end()};
      } catch (...) {
        setTensorTraceHook({});
        throw;
      }
    }

    std::string printWithModel(
        const std::string &source,
        const std::unordered_map<std::string, SymbolicExecutor::Result::ModelVal> &model,
        const std::unordered_map<std::string, std::vector<SymbolicExecutor::Result::ModelVal>>
            &vecModel = {},
        const std::unordered_map<std::string, std::vector<SymbolicExecutor::Result::ModelVal>>
            &tensorModel = {}
    ) {
      Program program = parseSource(source);
      std::ostringstream output;
      SIRPrinter printer(output, model, vecModel, tensorModel);
      printer.print(program);
      return output.str();
    }

    std::string replaceOnce(std::string source, std::string_view from, std::string_view to) {
      const auto position = source.find(from);
      if (position == std::string::npos)
        throw std::runtime_error("cannot materialize generated tensor return type");
      source.replace(position, from.size(), to);
      return source;
    }

    std::string tensorReturnSource(
        const std::string &source, std::string_view outputType, std::string_view scalarReturn,
        std::string_view tensorReturn
    ) {
      auto result = replaceOnce(source, ") : i32 {", ") : " + std::string(outputType) + " {");
      const std::string scalarTerm = "ret " + std::string(scalarReturn) + ";";
      const std::string tensorTerm = "ret " + std::string(tensorReturn) + ";";
      return replaceOnce(std::move(result), scalarTerm, tensorTerm);
    }

    std::string coordinatesSource() {
      return R"(fun @kernel(%base: i32) : i32 {
  sym %?a : value i32 in [0, 10];
  sym %?b : value i32 in [0, 10];
  sym %?k : value i32 in [-10, 10];
  let mut %row: tensor [2, 4] i32 = undef;
  let mut %col: tensor [2, 4] i32 = undef;
  let mut %ar: tensor [2, 4] i32 = undef;
  let mut %bc: tensor [2, 4] i32 = undef;
  let mut %sum: tensor [2, 4] i32 = undef;
  let mut %partial: tensor [2, 4] i32 = undef;
  let mut %out: tensor [2, 4] i32 = undef;
  let mut %p00: i32 = undef;
  let mut %p10: i32 = undef;
  let mut %p01: i32 = undef;
^entry:
  %row = tensor iota [2, 4], 0;
  %col = tensor iota [2, 4], 1;
  %ar = tensor mul %row, %?a;
  %bc = tensor mul %col, %?b;
  %sum = tensor add %ar, %bc;
  %partial = tensor add %sum, %?k;
  %out = tensor add %partial, %base;
  %p00 = tensor extract %out, [0, 0];
  %p10 = tensor extract %out, [1, 0];
  %p01 = tensor extract %out, [0, 1];
  require %p00 == 1;
  require %p10 == 8;
  require %p01 == 4;
  ret %p00;
}
)";
    }

    std::string sumTargetSource(bool symbolic) {
      if (symbolic) {
        return R"(fun @kernel() : i32 {
  sym %?x : value tensor [2, 4] i32 in [-16, 16];
  let mut %rows: tensor [2] i32 = undef;
  let mut %r0: i32 = undef;
  let mut %r1: i32 = undef;
^entry:
  %rows = tensor sum %?x, 1, drop;
  %r0 = tensor extract %rows, [0];
  %r1 = tensor extract %rows, [1];
  require %r0 == 10;
  require %r1 == -6;
  ret %r0;
}
)";
      }
      return R"(fun @kernel(%x: tensor [2, 4] i32) : i32 {
  let mut %rows: tensor [2] i32 = undef;
  let mut %r0: i32 = undef;
  let mut %r1: i32 = undef;
^entry:
  %rows = tensor sum %x, 1, drop;
  %r0 = tensor extract %rows, [0];
  %r1 = tensor extract %rows, [1];
  require %r0 == 10;
  require %r1 == -6;
  ret %r0;
}
)";
    }

    std::string softmaxSource() {
      return R"(fun @kernel(%x: tensor [2, 4] f32) : tensor [2, 4] f32 {
  let mut %out: tensor [2, 4] f32 = undef;
  let mut %last: f32 = undef;
  let mut %first: f32 = undef;
^entry:
  %out = tensor softmax %x, 1;
  %last = tensor extract %out, [0, 3];
  %first = tensor extract %out, [1, 0];
  // SIR uses canonical decimal float literals; 0.8 rounds to f32 bits 3f4ccccd.
  require %last > 0.8;
  require %first > 0.8;
  ret %out;
}
)";
    }

    [[nodiscard]] auto makeSolverFactory() {
      return [](const SymbolicExecutor::Config &config) -> std::unique_ptr<smt::ISolver> {
#if defined(USE_BITWUZLA)
        return std::make_unique<solver::BitwuzlaSolver>(
            config.timeout_ms, config.seed, config.num_smt_threads
        );
#elif defined(USE_ALIVESMT)
        return std::make_unique<solver::AliveSolver>(
            config.timeout_ms, config.seed, config.num_smt_threads
        );
#else
        (void) config;
        throw std::runtime_error("No solver backend compiled in");
#endif
      };
    }

    SymbolicExecutor::Result solveCandidate(const std::string &source, std::uint32_t seed) {
      Program program = checkedProgram(source);
      SymbolicExecutor::Config config;
      config.timeout_ms = 5000;
      config.seed = seed;
      SymbolicExecutor executor(program, config, makeSolverFactory());
      return executor.solve("@kernel", {"^entry"});
    }

    std::int32_t modelInt(
        const std::unordered_map<std::string, SymbolicExecutor::Result::ModelVal> &model,
        const std::string &name
    ) {
      const auto found = model.find(name);
      if (found == model.end() || !std::holds_alternative<std::int64_t>(found->second))
        throw std::runtime_error("exact tensor model omitted integer " + name);
      const auto value = std::get<std::int64_t>(found->second);
      if (value < INT32_MIN || value > INT32_MAX)
        throw std::runtime_error("exact tensor model value is outside i32");
      return static_cast<std::int32_t>(value);
    }

    Candidate coordinatesCandidate(const TensorGenerationOptions &options, std::mt19937 &random) {
      Candidate candidate;
      candidate.solverSource = coordinatesSource();
      candidate.outputShape = "tensor [2, 4] i32";
      candidate.outputElement = TensorElement::I32;
      candidate.operators = {"iota", "mul", "add", "extract"};
      std::int32_t base = 0;
      std::int32_t intercept = 1;
      std::unordered_map<std::string, SymbolicExecutor::Result::ModelVal> model;
      if (options.synthesis == "exact") {
        auto solved = solveCandidate(candidate.solverSource, options.seed);
        if (!solved.sat) {
          const std::string status = solved.unsat ? "unsat" : "unknown";
          throw std::runtime_error("exact_smt_" + status + ": " + solved.message);
        }
        model = solved.model;
        base = modelInt(solved.paramModel, "%base");
        intercept = modelInt(solved.model, "%?k");
      } else {
        std::uniform_int_distribution<std::int32_t> pickBase(-8, 8);
        base = pickBase(random);
        intercept = 1 - base;
        model.emplace("%?a", std::int64_t{7});
        model.emplace("%?b", std::int64_t{3});
        model.emplace("%?k", static_cast<std::int64_t>(intercept));
      }
      (void) intercept;
      auto materialized = printWithModel(candidate.solverSource, model);
      candidate.finalSource = tensorReturnSource(materialized, "tensor [2, 4] i32", "%p00", "%out");
      candidate.inputs.push_back({"base", TensorElement::I32, {}, encodeI32(base)});
      return candidate;
    }

    Candidate sumTargetCandidate(const TensorGenerationOptions &options, std::uint32_t seed) {
      Candidate candidate;
      const bool exact = options.synthesis == "exact";
      candidate.solverSource = sumTargetSource(exact);
      candidate.outputShape = "tensor [2] i32";
      candidate.outputElement = TensorElement::I32;
      candidate.operators = {"sum", "extract"};
      if (exact) {
        const auto solved = solveCandidate(candidate.solverSource, seed);
        if (!solved.sat) {
          const std::string status = solved.unsat ? "unsat" : "unknown";
          throw std::runtime_error("exact_smt_" + status + ": " + solved.message);
        }
        candidate.finalSource = tensorReturnSource(
            printWithModel(
                candidate.solverSource, solved.model, solved.vecModel, solved.tensorModel
            ),
            "tensor [2] i32", "%r0", "%rows"
        );
      } else {
        std::mt19937 random(seed);
        std::uniform_int_distribution<std::int32_t> small(-3, 3);
        std::vector<std::int32_t> values(8);
        const std::array<std::int32_t, 2> sums{10, -6};
        for (std::size_t row = 0; row < 2; ++row) {
          std::int32_t partial = 0;
          for (std::size_t col = 0; col < 3; ++col) {
            const auto value = small(random);
            values[row * 4 + col] = value;
            partial += value;
          }
          values[row * 4 + 3] = sums[row] - partial;
        }
        candidate.finalSource =
            tensorReturnSource(candidate.solverSource, "tensor [2] i32", "%r0", "%rows");
        candidate.inputs.push_back({"x", TensorElement::I32, {2, 4}, encodeI32Tensor(values)});
      }
      return candidate;
    }

    Candidate softmaxCandidate(std::uint32_t seed, std::uint32_t attempt) {
      Candidate candidate;
      candidate.solverSource = softmaxSource();
      candidate.finalSource = candidate.solverSource;
      candidate.outputShape = "tensor [2, 4] f32";
      candidate.outputElement = TensorElement::F32;
      candidate.operators = {"softmax", "extract"};
      // The first bounded proposal intentionally sits below 4/5. Every
      // following candidate grows the gap and is validated by the reference.
      const std::int32_t gap = 2 + static_cast<std::int32_t>(attempt % 3);
      const std::int32_t row0 = static_cast<std::int32_t>((seed + attempt * 3) % 5) - 2;
      const std::int32_t row1 = row0 + 3;
      const std::vector<float> values{
          static_cast<float>(row0),       static_cast<float>(row0),       static_cast<float>(row0),
          static_cast<float>(row0 + gap), static_cast<float>(row1 + gap), static_cast<float>(row1),
          static_cast<float>(row1),       static_cast<float>(row1)
      };
      candidate.inputs.push_back({"x", TensorElement::F32, {2, 4}, encodeF32Tensor(values)});
      return candidate;
    }

    TensorCase validateCandidate(
        Candidate candidate, const TensorGenerationOptions &options, const std::string &propertyId,
        const std::string &propertySource
    ) {
      Program program = checkedProgram(candidate.finalSource);
      const auto &entry = entryFunction(program);
      TensorCase testCase;
      testCase.source = candidate.finalSource;
      testCase.target = options.target;
      testCase.seed = options.seed;
      testCase.synthesis = options.synthesis;
      testCase.propertyId = propertyId;
      testCase.propertySource = propertySource;
      testCase.inputs = std::move(candidate.inputs);
      testCase.outputShape =
          TypeUtils::asTensor(entry.retType) ? TypeUtils::asTensor(entry.retType)->shape : Shape{};
      testCase.outputElement = scalarElement(
          TypeUtils::asTensor(entry.retType) ? TypeUtils::asTensor(entry.retType)->elem
                                             : entry.retType
      );
      testCase.operators = std::move(candidate.operators);
      testCase.numericComparison = testCase.outputElement == TensorElement::F32;
      if (testCase.numericComparison)
        testCase.propertyThresholdF32Bits = "3f4ccccd";
      testCase.evidenceKind = options.synthesis == "exact" ? "exact_smt"
                              : testCase.numericComparison ? "numerical_reference"
                                                           : "construction";
      testCase.evidenceStatus = options.synthesis == "exact" ? "exact_smt_checked"
                                : testCase.numericComparison ? "numerical_reference_checked"
                                                             : "construction_checked";

      if (testCase.numericComparison) {
        std::vector<TensorTraceEntry> trace128;
        std::vector<TensorTraceEntry> trace256;
        const auto output128 = runReference(program, testCase.inputs, 128, &trace128);
        const auto output256 = runReference(program, testCase.inputs, 256, &trace256);
        if (output128 != output256 || !tensorTracesEqual(trace128, trace256))
          throw std::runtime_error("unstable_reference: 128-bit and 256-bit results differ");
        testCase.expected = output256;
        testCase.referenceStable = true;
      } else {
        testCase.expected = runReference(program, testCase.inputs, 128, nullptr);
      }
      return testCase;
    }

    std::string inferredProperty(const TensorGenerationOptions &options) {
      if (!options.property.empty())
        return options.property;
      return options.dtype == "i32" ? "coordinates" : "softmax-dominance";
    }

    std::string propertySource(std::string_view property) {
      if (property == "coordinates")
        return "On tensor [2,4], output[r,c] = 7*r + 3*c + 1.";
      if (property == "sum-target")
        return "For tensor [2,4], row sums equal [10,-6].";
      return "On tensor [2,4], softmax row 0 column 3 and row 1 column 0 exceed 4/5.";
    }

    bool isCandidateRejection(std::string_view message) {
      return message.starts_with("UB:") ||
             message.find("Requirement failed") != std::string_view::npos ||
             message.find("unstable_reference:") == 0;
    }

  } // namespace

  TensorGenerationResult generateTensorCases(const TensorGenerationOptions &options) {
    TensorGenerationResult result;
    result.requested = options.count;
    std::vector<AttemptRecord> attempts;
    std::filesystem::create_directories(options.outputDirectory);
    const std::string property = inferredProperty(options);
    TensorGenerationOptions recordedOptions = options;
    recordedOptions.property = property;

    auto failRoute = [&](std::string status, std::string reason) {
      result.status = std::move(status);
      result.reason = std::move(reason);
      writeSummary(options.outputDirectory / "generation.json", recordedOptions, result, attempts);
      return result;
    };

    if (options.target != "sir" && options.target != "triton")
      return failRoute("unsupported_target", "tensor target must be sir or triton");
    if (options.dtype != "i32" && options.dtype != "f32")
      return failRoute("invalid_profile", "tensor dtype must be i32 or f32");
    if (property != "coordinates" && property != "sum-target" && property != "softmax-dominance")
      return failRoute("invalid_profile", "unknown tensor property family: " + property);
    if ((property == "coordinates" || property == "sum-target") && options.dtype != "i32")
      return failRoute("invalid_profile", property + " requires tensor dtype i32");
    if (property == "softmax-dominance" && options.dtype != "f32")
      return failRoute("invalid_profile", "softmax-dominance requires tensor dtype f32");
    if (options.synthesis != "construction" && options.synthesis != "exact" &&
        options.synthesis != "cvc5")
      return failRoute(
          "unsupported_route", "tensor synthesis must be construction, exact, or cvc5"
      );
    if (options.synthesis == "exact" && options.dtype == "f32")
      return failRoute(
          "unsupported_theory", "exact synthesis does not support f32 softmax numerical operators"
      );
    if (options.synthesis == "cvc5")
      return failRoute(
          "unavailable", "cvc5 tensor feasibility gate did not pass in the captured environment"
      );
    if (options.count == 0)
      return failRoute("invalid_profile", "tensor case count must be at least one");

    std::mt19937 random(options.seed);
    for (std::uint32_t caseIndex = 0; caseIndex < options.count; ++caseIndex) {
      bool accepted = false;
      const std::string caseProperty = property;
      for (std::uint32_t attempt = 0; attempt < options.attemptBudget; ++attempt) {
        const auto attemptSeed = options.seed + caseIndex * 0x9e3779b9U + attempt;
        Candidate candidate;
        try {
          if (property == "coordinates")
            candidate = coordinatesCandidate(options, random);
          else if (property == "sum-target")
            candidate = sumTargetCandidate(options, attemptSeed);
          else
            candidate = softmaxCandidate(options.seed + caseIndex, attempt);

          auto caseData = validateCandidate(
              std::move(candidate), options, caseProperty, propertySource(caseProperty)
          );
          const auto directory = options.outputDirectory / ("case_" + [&] {
                                   std::ostringstream index;
                                   index << std::setw(6) << std::setfill('0') << caseIndex;
                                   return index.str();
                                 }());
          writeTensorCase(directory, caseData);
          ++result.emitted;
          attempts.push_back({caseIndex, attempt, caseProperty, "accepted", {}});
          accepted = true;
          break;
        } catch (const std::exception &error) {
          const std::string message = error.what();
          if (message.starts_with("exact_smt_") || message.starts_with("unsupported_theory")) {
            result.status = message.starts_with("unsupported_theory") ? "unsupported_theory"
                                                                      : "exact_smt_failure";
            result.reason = message;
            writeSummary(
                options.outputDirectory / "generation.json", recordedOptions, result, attempts
            );
            return result;
          }
          if (!isCandidateRejection(message)) {
            result.status = "generation_error";
            result.reason = message;
            writeSummary(
                options.outputDirectory / "generation.json", recordedOptions, result, attempts
            );
            return result;
          }
          ++result.rejected;
          attempts.push_back({caseIndex, attempt, caseProperty, "rejected", message});
        }
      }
      if (!accepted) {
        result.status = "shortfall";
        result.reason = "requested " + std::to_string(options.count) + " cases but emitted " +
                        std::to_string(result.emitted) + " within attempt budget " +
                        std::to_string(options.attemptBudget);
        writeSummary(
            options.outputDirectory / "generation.json", recordedOptions, result, attempts
        );
        return result;
      }
    }
    result.status = "complete";
    writeSummary(options.outputDirectory / "generation.json", recordedOptions, result, attempts);
    return result;
  }

} // namespace refractir::reify
