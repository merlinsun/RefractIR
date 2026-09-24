#include "reify/tensor_gen.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
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
      std::vector<std::pair<Shape, std::int32_t>> shapeAxes;
      std::string propertySource;
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

    std::string sirShape(const Shape &shape) {
      std::ostringstream text;
      text << '[';
      for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i)
          text << ", ";
        text << shape[i];
      }
      text << ']';
      return text.str();
    }

    std::string sirCoordinates(const Shape &coordinates) { return sirShape(coordinates); }

    std::string tensorI32Type(const Shape &shape) { return "tensor " + sirShape(shape) + " i32"; }

    std::string tensorF32Type(const Shape &shape) { return "tensor " + sirShape(shape) + " f32"; }

    Shape zeroCoordinates(const Shape &shape) { return Shape(shape.size(), 0); }

    std::uint32_t rowMajorIndex(const Shape &shape, const Shape &coordinates) {
      if (shape.size() != coordinates.size())
        throw std::runtime_error("generator coordinate rank does not match its shape");
      std::uint32_t index = 0;
      for (std::size_t i = 0; i < shape.size(); ++i) {
        if (coordinates[i] >= shape[i])
          throw std::runtime_error("generator coordinate is out of bounds");
        index = index * shape[i] + coordinates[i];
      }
      return index;
    }

    std::string coordinatesSource(const Shape &shape, bool swapAxes, bool reorderAdds) {
      const std::uint32_t rowAxis = swapAxes ? 1 : 0;
      const std::uint32_t colAxis = swapAxes ? 0 : 1;
      const std::string rowCoefficient = rowAxis == 0 ? "%?a" : "%?b";
      const std::string colCoefficient = colAxis == 0 ? "%?a" : "%?b";
      const Shape origin = zeroCoordinates(shape);
      Shape rowStep = origin;
      Shape colStep = origin;
      rowStep[0] = 1;
      colStep[1] = 1;
      std::ostringstream source;
      source << "fun @kernel(%base: i32) : i32 {\n"
             << "  sym %?a : value i32 in [0, 10];\n"
             << "  sym %?b : value i32 in [0, 10];\n"
             << "  sym %?k : value i32 in [-10, 10];\n"
             << "  let mut %row: " << tensorI32Type(shape) << " = undef;\n"
             << "  let mut %col: " << tensorI32Type(shape) << " = undef;\n"
             << "  let mut %ar: " << tensorI32Type(shape) << " = undef;\n"
             << "  let mut %bc: " << tensorI32Type(shape) << " = undef;\n"
             << "  let mut %sum: " << tensorI32Type(shape) << " = undef;\n"
             << "  let mut %partial: " << tensorI32Type(shape) << " = undef;\n"
             << "  let mut %out: " << tensorI32Type(shape) << " = undef;\n"
             << "  let mut %p00: i32 = undef;\n"
             << "  let mut %p10: i32 = undef;\n"
             << "  let mut %p01: i32 = undef;\n"
             << "^entry:\n"
             << "  %row = tensor iota " << sirShape(shape) << ", " << rowAxis << ";\n"
             << "  %col = tensor iota " << sirShape(shape) << ", " << colAxis << ";\n"
             << "  %ar = tensor mul %row, " << rowCoefficient << ";\n"
             << "  %bc = tensor mul %col, " << colCoefficient << ";\n";
      if (reorderAdds) {
        source << "  %sum = tensor add %ar, %?k;\n"
               << "  %partial = tensor add %sum, %bc;\n";
      } else {
        source << "  %sum = tensor add %ar, %bc;\n"
               << "  %partial = tensor add %sum, %?k;\n";
      }
      source << "  %out = tensor add %partial, %base;\n"
             << "  %p00 = tensor extract %out, " << sirCoordinates(origin) << ";\n"
             << "  %p10 = tensor extract %out, " << sirCoordinates(rowStep) << ";\n"
             << "  %p01 = tensor extract %out, " << sirCoordinates(colStep) << ";\n"
             << "  require %p00 == 1;\n"
             << "  require %p10 == 8;\n"
             << "  require %p01 == 4;\n"
             << "  ret %p00;\n"
             << "}\n";
      return source.str();
    }

    std::string coordinatesProperty(const Shape &shape) {
      return "On tensor " + sirShape(shape) + ", output[i] = 7*i0 + 3*i1 + 1.";
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

    std::string softmaxSource(
        const Shape &shape, std::uint32_t axis, std::int32_t axisSpelling,
        std::uint32_t composition, const Shape &lastCoordinates,
        const std::optional<Shape> &firstCoordinates
    ) {
      const Shape keptShape = [&] {
        auto result = shape;
        result[axis] = 1;
        return result;
      }();
      const Shape reducedShape = [&] {
        auto result = shape;
        result.erase(result.begin() + axis);
        return result;
      }();
      std::ostringstream source;
      source << "fun @kernel(%x: " << tensorF32Type(shape) << ") : " << tensorF32Type(shape)
             << " {\n"
             << "  let mut %out: " << tensorF32Type(shape) << " = undef;\n"
             << "  let mut %last: f32 = undef;\n";
      if (firstCoordinates)
        source << "  let mut %first: f32 = undef;\n";

      if (composition == 1) {
        source << "  let mut %maximum: "
               << (reducedShape.empty() ? "f32" : tensorF32Type(reducedShape)) << " = undef;\n"
               << "  let mut %expanded: " << tensorF32Type(keptShape) << " = undef;\n"
               << "  let mut %shifted: " << tensorF32Type(shape) << " = undef;\n"
               << "  let mut %exps: " << tensorF32Type(shape) << " = undef;\n"
               << "  let mut %denom: " << tensorF32Type(keptShape) << " = undef;\n";
      } else if (composition == 2) {
        std::vector<std::uint32_t> permutation(shape.size());
        std::iota(permutation.rbegin(), permutation.rend(), 0u);
        Shape transposedShape(shape.size());
        for (std::size_t i = 0; i < permutation.size(); ++i)
          transposedShape[i] = shape[permutation[i]];
        source << "  let mut %transposed: " << tensorF32Type(transposedShape) << " = undef;\n"
               << "  let mut %normalized: " << tensorF32Type(transposedShape) << " = undef;\n";
      } else if (composition == 3) {
        source << "  let mut %broadcasted: " << tensorF32Type(shape) << " = undef;\n";
      } else if (composition >= 4) {
        source << "  let mut %normalized: " << tensorF32Type(shape) << " = undef;\n";
      }

      source << "^entry:\n";
      if (composition == 1) {
        source << "  %maximum = tensor rmax %x, " << axisSpelling << ", drop;\n"
               << "  %expanded = tensor expand %maximum, " << axis << ";\n"
               << "  %shifted = tensor sub %x, %expanded;\n"
               << "  %exps = tensor exp %shifted;\n"
               << "  %denom = tensor sum %exps, " << axisSpelling << ", keep;\n"
               << "  %out = tensor div %exps, %denom;\n";
      } else if (composition == 2 && shape.size() > 1) {
        std::vector<std::uint32_t> permutation(shape.size());
        std::iota(permutation.rbegin(), permutation.rend(), 0u);
        Shape transposedShape(shape.size());
        Shape inverse(shape.size());
        std::uint32_t transposedAxis = 0;
        for (std::size_t i = 0; i < permutation.size(); ++i) {
          transposedShape[i] = shape[permutation[i]];
          inverse[permutation[i]] = static_cast<std::uint32_t>(i);
          if (permutation[i] == axis)
            transposedAxis = static_cast<std::uint32_t>(i);
        }
        const auto transposedAxisSpelling = axisSpelling < 0
                                                ? static_cast<std::int32_t>(transposedAxis) -
                                                      static_cast<std::int32_t>(shape.size())
                                                : static_cast<std::int32_t>(transposedAxis);
        source << "  %transposed = tensor transpose %x, "
               << sirShape(Shape(permutation.begin(), permutation.end())) << ";\n"
               << "  %normalized = tensor softmax %transposed, " << transposedAxisSpelling << ";\n"
               << "  %out = tensor transpose %normalized, " << sirShape(inverse) << ";\n";
      } else if (composition == 3) {
        source << "  %broadcasted = tensor broadcast %x, " << sirShape(shape) << ";\n"
               << "  %out = tensor softmax %broadcasted, " << axisSpelling << ";\n";
      } else if (composition == 4 || composition == 5) {
        source << "  %normalized = tensor softmax %x, " << axisSpelling << ";\n"
               << "  %out = tensor " << (composition == 4 ? "max" : "min")
               << " %normalized, %normalized;\n";
      } else {
        source << "  %out = tensor softmax %x, " << axisSpelling << ";\n";
      }
      source << "  %last = tensor extract %out, " << sirCoordinates(lastCoordinates) << ";\n";
      if (firstCoordinates)
        source << "  %first = tensor extract %out, " << sirCoordinates(*firstCoordinates) << ";\n";
      source << "  require %last > 0.8;\n";
      if (firstCoordinates)
        source << "  require %first > 0.8;\n";
      source << "  ret %out;\n}\n";
      return source.str();
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

    std::vector<std::int32_t> modelTensorI32(
        const std::unordered_map<std::string, std::vector<SymbolicExecutor::Result::ModelVal>>
            &model,
        const std::string &name
    ) {
      const auto found = model.find(name);
      if (found == model.end())
        throw std::runtime_error("exact tensor model omitted tensor " + name);
      std::vector<std::int32_t> values;
      values.reserve(found->second.size());
      for (const auto &element: found->second) {
        if (!std::holds_alternative<std::int64_t>(element))
          throw std::runtime_error("exact tensor model contains a non-integer element");
        const auto value = std::get<std::int64_t>(element);
        if (value < INT32_MIN || value > INT32_MAX)
          throw std::runtime_error("exact tensor model value is outside i32");
        values.push_back(static_cast<std::int32_t>(value));
      }
      return values;
    }

    Candidate coordinatesCandidate(const TensorGenerationOptions &options, std::mt19937 &random) {
      Candidate candidate;
      const std::array<Shape, 4> shapes{Shape{2, 4}, Shape{4, 2}, Shape{2, 2, 2}, Shape{4, 2, 1}};
      std::uniform_int_distribution<std::size_t> chooseShape(0, shapes.size() - 1);
      std::uniform_int_distribution<std::uint32_t> chooseBit(0, 1);
      const Shape &shape = shapes[chooseShape(random)];
      const bool swapAxes = chooseBit(random) != 0;
      const bool reorderAdds = chooseBit(random) != 0;
      const std::uint32_t rowAxis = swapAxes ? 1 : 0;
      const std::uint32_t colAxis = swapAxes ? 0 : 1;
      candidate.solverSource = coordinatesSource(shape, swapAxes, reorderAdds);
      candidate.outputShape = tensorI32Type(shape);
      candidate.outputElement = TensorElement::I32;
      candidate.operators = {"iota", "mul", "add", "extract"};
      candidate.shapeAxes = {
          {shape, static_cast<std::int32_t>(rowAxis)}, {shape, static_cast<std::int32_t>(colAxis)}
      };
      candidate.propertySource = coordinatesProperty(shape);
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
      candidate.finalSource =
          tensorReturnSource(materialized, candidate.outputShape, "%p00", "%out");
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
      candidate.shapeAxes = {{Shape{2, 4}, 1}};
      candidate.propertySource = "For tensor [2,4], row sums equal [10,-6].";
      if (exact) {
        const auto solved = solveCandidate(candidate.solverSource, seed);
        if (!solved.sat) {
          const std::string status = solved.unsat ? "unsat" : "unknown";
          throw std::runtime_error("exact_smt_" + status + ": " + solved.message);
        }
        const auto values = modelTensorI32(solved.tensorModel, "%?x");
        if (values.size() != 8)
          throw std::runtime_error("exact sum target produced the wrong input element count");
        candidate.finalSource =
            tensorReturnSource(sumTargetSource(false), "tensor [2] i32", "%r0", "%rows");
        candidate.inputs.push_back({"x", TensorElement::I32, {2, 4}, encodeI32Tensor(values)});
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
      std::mt19937 random(seed + attempt * 0x9e3779b9U);
      std::uniform_int_distribution<std::uint32_t> rankDistribution(1, 3);
      std::uniform_int_distribution<std::size_t> dimensionDistribution(0, 2);
      std::uniform_int_distribution<std::uint32_t> bitDistribution(0, 1);
      std::uniform_int_distribution<std::uint32_t> compositionDistribution(0, 5);
      constexpr std::array<std::uint32_t, 3> dimensionChoices{1, 2, 4};
      Shape shape(rankDistribution(random));
      for (auto &dimension: shape)
        dimension = dimensionChoices[dimensionDistribution(random)];

      std::vector<std::uint32_t> reducibleAxes;
      for (std::uint32_t candidateAxis = 0; candidateAxis < shape.size(); ++candidateAxis)
        if (shape[candidateAxis] > 1)
          reducibleAxes.push_back(candidateAxis);
      if (reducibleAxes.empty()) {
        std::uniform_int_distribution<std::size_t> chooseAxis(0, shape.size() - 1);
        shape[chooseAxis(random)] = 2;
        for (std::uint32_t candidateAxis = 0; candidateAxis < shape.size(); ++candidateAxis)
          if (shape[candidateAxis] > 1)
            reducibleAxes.push_back(candidateAxis);
      }
      std::uniform_int_distribution<std::size_t> axisDistribution(0, reducibleAxes.size() - 1);
      const std::uint32_t axis = reducibleAxes[axisDistribution(random)];
      std::int32_t axisSpelling = static_cast<std::int32_t>(axis);
      if (bitDistribution(random) != 0)
        axisSpelling -= static_cast<std::int32_t>(shape.size());
      const auto composition = compositionDistribution(random);
      const auto axisSize = shape[axis];
      const auto requiredGap =
          static_cast<std::int32_t>(std::floor(std::log(4.0 * static_cast<double>(axisSize - 1)))) +
          1;
      const auto gap = requiredGap - (attempt == 0 ? 1 : 0);
      Shape lastCoordinates = zeroCoordinates(shape);
      lastCoordinates[axis] = axisSize - 1;
      std::optional<Shape> firstCoordinates;
      for (std::size_t otherAxis = 0; otherAxis < shape.size(); ++otherAxis) {
        if (otherAxis != axis && shape[otherAxis] > 1) {
          Shape second = zeroCoordinates(shape);
          second[otherAxis] = 1;
          firstCoordinates = std::move(second);
          break;
        }
      }
      if (firstCoordinates)
        (*firstCoordinates)[axis] = 0;
      candidate.solverSource =
          softmaxSource(shape, axis, axisSpelling, composition, lastCoordinates, firstCoordinates);
      candidate.finalSource = candidate.solverSource;
      candidate.outputShape = tensorF32Type(shape);
      candidate.outputElement = TensorElement::F32;
      candidate.shapeAxes.push_back({shape, static_cast<std::int32_t>(axis)});
      if (composition == 1) {
        candidate.operators = {"rmax", "expand", "sub", "exp", "sum", "div", "extract"};
      } else if (composition == 2 && shape.size() > 1) {
        std::vector<std::uint32_t> permutation(shape.size());
        std::iota(permutation.rbegin(), permutation.rend(), 0u);
        Shape transposedShape(shape.size());
        std::uint32_t transposedAxis = 0;
        for (std::size_t i = 0; i < permutation.size(); ++i) {
          transposedShape[i] = shape[permutation[i]];
          if (permutation[i] == axis)
            transposedAxis = static_cast<std::uint32_t>(i);
        }
        candidate.operators = {"transpose", "softmax", "extract"};
        candidate.shapeAxes.push_back({transposedShape, static_cast<std::int32_t>(transposedAxis)});
      } else if (composition == 3) {
        candidate.operators = {"broadcast", "softmax", "extract"};
      } else if (composition == 4) {
        candidate.operators = {"softmax", "max", "extract"};
      } else if (composition == 5) {
        candidate.operators = {"softmax", "min", "extract"};
      } else {
        candidate.operators = {"softmax", "extract"};
      }
      candidate.propertySource = "On tensor " + sirShape(shape) + ", softmax axis " +
                                 std::to_string(axis) + " exceeds 4/5 at " +
                                 sirCoordinates(lastCoordinates) + ".";
      if (firstCoordinates)
        candidate.propertySource +=
            " A second group at " + sirCoordinates(*firstCoordinates) + " also exceeds 4/5.";

      Shape groupShape = shape;
      groupShape.erase(groupShape.begin() + axis);
      std::uint32_t groupCount = 1;
      for (const auto dimension: groupShape)
        groupCount *= dimension;
      std::uniform_int_distribution<std::int32_t> offsetDistribution(-4, 4);
      std::vector<std::int32_t> groupOffsets(groupCount);
      for (auto &offset: groupOffsets)
        offset = offsetDistribution(random);
      std::vector<std::int32_t> dominantAxes(groupCount, -1);
      dominantAxes[0] = static_cast<std::int32_t>(axisSize - 1);
      if (firstCoordinates) {
        Shape secondGroupCoordinates;
        secondGroupCoordinates.reserve(firstCoordinates->size() - 1);
        for (std::size_t coordinateAxis = 0; coordinateAxis < firstCoordinates->size();
             ++coordinateAxis)
          if (coordinateAxis != axis)
            secondGroupCoordinates.push_back((*firstCoordinates)[coordinateAxis]);
        const auto secondGroup = rowMajorIndex(groupShape, secondGroupCoordinates);
        if (groupOffsets[secondGroup] == groupOffsets[0])
          groupOffsets[secondGroup] = groupOffsets[secondGroup] == 4
                                          ? groupOffsets[secondGroup] - 1
                                          : groupOffsets[secondGroup] + 1;
        dominantAxes[secondGroup] = 0;
      }
      std::vector<float> values(tensorElementCount(shape), 0.0f);
      for (std::uint32_t flat = 0; flat < values.size(); ++flat) {
        Shape coordinates(shape.size());
        auto remaining = flat;
        for (std::size_t coordinateAxis = shape.size(); coordinateAxis-- > 0;) {
          coordinates[coordinateAxis] = remaining % shape[coordinateAxis];
          remaining /= shape[coordinateAxis];
        }
        Shape groupCoordinates;
        groupCoordinates.reserve(groupShape.size());
        for (std::size_t coordinateAxis = 0; coordinateAxis < coordinates.size(); ++coordinateAxis)
          if (coordinateAxis != axis)
            groupCoordinates.push_back(coordinates[coordinateAxis]);
        const auto group = rowMajorIndex(groupShape, groupCoordinates);
        values[flat] = static_cast<float>(groupOffsets[group]);
        if (dominantAxes[group] >= 0 &&
            coordinates[axis] == static_cast<std::uint32_t>(dominantAxes[group]))
          values[flat] += static_cast<float>(gap);
      }
      candidate.inputs.push_back({"x", TensorElement::F32, shape, encodeF32Tensor(values)});
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
      testCase.propertySource =
          candidate.propertySource.empty() ? propertySource : candidate.propertySource;
      testCase.inputs = std::move(candidate.inputs);
      testCase.outputShape =
          TypeUtils::asTensor(entry.retType) ? TypeUtils::asTensor(entry.retType)->shape : Shape{};
      testCase.outputElement = scalarElement(
          TypeUtils::asTensor(entry.retType) ? TypeUtils::asTensor(entry.retType)->elem
                                             : entry.retType
      );
      testCase.operators = std::move(candidate.operators);
      testCase.shapeAxes = std::move(candidate.shapeAxes);
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
