#include "reify/tensor_gen.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
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
      std::string stratum;
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
      std::string exactnessKind;
      std::string inputPattern;
      std::uint64_t exactnessMaxAbsLeafSum = 0;
      TensorElement outputElement = TensorElement::I32;
      std::vector<std::int32_t> loopBounds;
      std::uint32_t totalBodyExecutions = 0;
      std::vector<Shape> fullShapes;
      std::vector<Shape> rangeExtents;
      std::optional<std::uint32_t> caseSeed;
    };

    constexpr std::array<std::string_view, 10> v2StratumSchedule{
        "bit-preserving-f32-layout", "exact-f32-reduction",     "loop-reduction",
        "multi-input-composition",   "multi-input-composition", "untargeted-exploration",
        "bit-preserving-f32-layout", "exact-f32-reduction",     "loop-reduction",
        "multi-input-composition"};
    constexpr std::array<std::string_view, 5> v2Strata{
        "bit-preserving-f32-layout", "exact-f32-reduction", "loop-reduction",
        "multi-input-composition", "untargeted-exploration"};
    constexpr std::array<std::string_view, 6> v3StratumSchedule{
        "tail-arithmetic", "tail-sum", "tail-layout", "tail-broadcast",
        "tail-loop", "tail-untargeted"};
    constexpr std::array<std::string_view, 6> v3InputPatterns{
        "coordinate-distinct", "mixed-sign", "cancellation", "repeated-value",
        "distinct-groups", "seeded-random"};

    std::uint64_t splitMix64(std::uint64_t value) {
      value += 0x9e3779b97f4a7c15ULL;
      value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
      value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
      return value ^ (value >> 31);
    }

    std::uint32_t streamSeed(std::uint32_t seed, std::uint64_t stream) {
      return static_cast<std::uint32_t>(splitMix64(seed ^ stream));
    }

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
          << "\"profile\":"
          << jsonString(options.profileVersion == TensorProfileId::V3 ? "tensor-v3"
                       : options.profileVersion == TensorProfileId::V2 ? "tensor-v2"
                                                                       : "tensor-v1")
          << ",\"status\":" << jsonString(result.status) << ",\"requested\":" << result.requested
          << ",\"emitted\":" << result.emitted << ",\"rejected\":" << result.rejected
          << ",\"seed\":" << options.seed << ",\"dtype\":" << jsonString(options.dtype)
          << ",\"property\":" << jsonString(options.property)
          << ",\"primary_stratum\":" << jsonString(options.primaryStratum)
          << ",\"synthesis\":" << jsonString(options.synthesis)
          << ",\"target\":" << jsonString(options.target)
          << ",\"attempt_budget\":" << options.attemptBudget;
      if (!result.reason.empty())
        out << ",\"reason\":" << jsonString(result.reason);
      if (options.profileVersion == TensorProfileId::V2) {
        std::array<std::uint32_t, v2Strata.size()> scheduled{};
        std::array<std::uint32_t, v2Strata.size()> emitted{};
        for (std::uint32_t index = 0; index < result.requested; ++index) {
          const auto stratum = options.primaryStratum.empty()
                                   ? v2StratumSchedule[index % v2StratumSchedule.size()]
                                   : std::string_view(options.primaryStratum);
          const auto found = std::find(v2Strata.begin(), v2Strata.end(), stratum);
          if (found != v2Strata.end())
            ++scheduled[static_cast<std::size_t>(found - v2Strata.begin())];
        }
        for (const auto &attempt: attempts) {
          if (attempt.outcome != "accepted")
            continue;
          const auto found = std::find(v2Strata.begin(), v2Strata.end(), attempt.stratum);
          if (found != v2Strata.end())
            ++emitted[static_cast<std::size_t>(found - v2Strata.begin())];
        }
        out << ",\"stratum_quotas\":[";
        for (std::size_t index = 0; index < v2Strata.size(); ++index) {
          if (index)
            out << ',';
          const auto shortfall =
              scheduled[index] > emitted[index] ? scheduled[index] - emitted[index] : 0;
          out << "{\"stratum\":" << jsonString(v2Strata[index])
              << ",\"scheduled\":" << scheduled[index] << ",\"emitted\":" << emitted[index]
              << ",\"shortfall\":" << shortfall << '}';
        }
        out << ']';
      }
      out << ",\"attempts\":[";
      for (std::size_t i = 0; i < attempts.size(); ++i) {
        const auto &attempt = attempts[i];
        if (i)
          out << ',';
        out << "{\"case\":" << attempt.caseIndex << ",\"attempt\":" << attempt.attempt
            << ",\"property_id\":" << jsonString(attempt.property)
            << ",\"primary_stratum\":" << jsonString(attempt.stratum)
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

    Program checkedProgram(
        const std::string &source, TensorProfileId profileVersion = TensorProfileId::V1
    ) {
      Program program = parseSource(source);
      DiagBag diagnostics;
      CheckOptions checkOptions;
      checkOptions.tensorProfile = profileVersion;
      if (!checkProgram(program, diagnostics, checkOptions))
        throw std::runtime_error("generated tensor source failed frontend checking");
      if (!checkTensorCapabilities(
              program, TensorConsumer::Reference, tensorProfile(profileVersion), diagnostics
          ))
        throw std::runtime_error("generated tensor source is outside selected tensor profile");
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

    std::optional<std::int32_t> inputCounterfactualWitness(
        const Program &program, const std::vector<TensorCaseBuffer> &inputs, std::size_t inputIndex,
        const std::vector<std::uint8_t> &expected
    ) {
      for (const std::int32_t replacement: {1, 4}) {
        auto candidateInputs = inputs;
        auto &bytes = candidateInputs[inputIndex].bytes;
        for (std::size_t offset = 0; offset < bytes.size(); offset += 4) {
          const auto word = candidateInputs[inputIndex].element == TensorElement::I32
                                ? std::bit_cast<std::uint32_t>(replacement)
                                : std::bit_cast<std::uint32_t>(static_cast<float>(replacement));
          for (std::size_t byte = 0; byte < 4; ++byte)
            bytes[offset + byte] = static_cast<std::uint8_t>(word >> (byte * 8));
        }
        if (runReference(program, candidateInputs, 128, nullptr) != expected)
          return replacement;
      }
      return std::nullopt;
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

    Shape v3BoundaryShape(std::uint32_t index) {
      constexpr std::array<std::uint32_t, 5> lengths{17, 31, 33, 63, 65};
      constexpr std::array<std::uint32_t, 5> tileExtents{32, 32, 64, 64, 128};
      if (index < lengths.size())
        return {lengths[index]};
      const auto pair =
          ((index - static_cast<std::uint32_t>(lengths.size())) / 2) % 10;
      const auto lengthIndex = pair / 2;
      const auto tail = lengths[lengthIndex];
      const auto other = 512 / tileExtents[lengthIndex];
      return pair % 2 == 0 ? Shape{tail, other} : Shape{other, tail};
    }

    std::vector<std::int32_t> v3PatternValues(
        const Shape &shape, std::string_view pattern, std::uint32_t seed
    ) {
      const auto count = tensorElementCount(shape);
      std::vector<std::int32_t> values(count);
      std::mt19937 random(seed);
      std::uniform_int_distribution<std::int32_t> choose(-8, 8);
      for (std::uint32_t index = 0; index < count; ++index) {
        if (pattern == "coordinate-distinct") {
          values[index] = static_cast<std::int32_t>(index + 1);
        } else if (pattern == "mixed-sign") {
          values[index] = (index % 2 == 0 ? 1 : -1) *
                          static_cast<std::int32_t>(index % 7 + 1);
        } else if (pattern == "cancellation") {
          const auto magnitude = static_cast<std::int32_t>(index / 2 % 8 + 1);
          values[index] = index % 2 == 0 ? magnitude : -magnitude;
        } else if (pattern == "repeated-value") {
          values[index] = 3;
        } else if (pattern == "distinct-groups") {
          std::uint32_t remaining = index;
          Shape coordinates(shape.size());
          for (std::size_t axis = shape.size(); axis-- > 0;) {
            coordinates[axis] = remaining % shape[axis];
            remaining /= shape[axis];
          }
          std::int32_t value = 1;
          for (std::size_t axis = 0; axis < coordinates.size(); ++axis)
            value += static_cast<std::int32_t>((axis + 2) * coordinates[axis]);
          values[index] = value;
        } else if (pattern == "seeded-random") {
          values[index] = choose(random);
        } else {
          throw std::runtime_error("tensor-v3 has an unknown input pattern");
        }
      }
      return values;
    }

    void assignV3PhysicalLayout(TensorCaseBuffer &buffer, std::uint32_t seed) {
      if (buffer.shape.empty())
        return;
      const auto gap = static_cast<std::uint32_t>(seed % 3 + 1);
      buffer.elementStrides.resize(buffer.shape.size());
      if (buffer.shape.size() == 1) {
        buffer.elementStrides[0] = gap;
      } else if ((seed & 1u) == 0) {
        buffer.elementStrides = {buffer.shape[1] + gap, 1};
      } else {
        buffer.elementStrides = {1, buffer.shape[0] + gap};
      }
      buffer.elementOffset = (seed >> 2) % 3;
      std::uint64_t lastAddress = buffer.elementOffset;
      for (std::size_t axis = 0; axis < buffer.shape.size(); ++axis)
        lastAddress += static_cast<std::uint64_t>(buffer.shape[axis] - 1) *
                       buffer.elementStrides[axis];
      if (lastAddress >= std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("tensor-v3 physical layout exceeds its storage limit");
      buffer.storageSpan = static_cast<std::uint32_t>(lastAddress + 1);
      buffer.axisOrder.resize(buffer.shape.size());
      std::iota(buffer.axisOrder.begin(), buffer.axisOrder.end(), 0u);
      std::sort(buffer.axisOrder.begin(), buffer.axisOrder.end(), [&](auto lhs, auto rhs) {
        return buffer.elementStrides[lhs] == buffer.elementStrides[rhs]
                   ? lhs < rhs
                   : buffer.elementStrides[lhs] < buffer.elementStrides[rhs];
      });
    }

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
          {shape, static_cast<std::int32_t>(rowAxis)}, {shape, static_cast<std::int32_t>(colAxis)}};
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

    Candidate compositionCandidate(const TensorGenerationOptions &options, std::uint32_t seed) {
      Candidate candidate;
      std::mt19937 random(seed);
      const std::array<Shape, 9> shapes{Shape{2, 4},    Shape{4, 2},    Shape{2, 4, 8},
                                        Shape{2, 8, 4}, Shape{4, 2, 8}, Shape{8, 2, 4},
                                        Shape{4, 4},    Shape{4, 4, 2}, Shape{4, 4, 4}};
      std::uniform_int_distribution<std::size_t> chooseShape(0, shapes.size() - 1);
      const Shape inputShape = shapes[chooseShape(random)];
      std::vector<std::uint32_t> permutation(inputShape.size());
      std::iota(permutation.begin(), permutation.end(), 0u);
      do {
        std::shuffle(permutation.begin(), permutation.end(), random);
      } while (std::is_sorted(permutation.begin(), permutation.end()));
      Shape outputShape(inputShape.size());
      for (std::size_t axis = 0; axis < permutation.size(); ++axis)
        outputShape[axis] = inputShape[permutation[axis]];
      std::uniform_int_distribution<std::uint32_t> chooseAxis(
          0, static_cast<std::uint32_t>(outputShape.size() - 1)
      );
      const auto reductionAxis = chooseAxis(random);
      Shape reducedShape = outputShape;
      reducedShape.erase(reducedShape.begin() + reductionAxis);
      Shape expandedShape = reducedShape;
      expandedShape.insert(expandedShape.begin() + reductionAxis, 1);
      const std::string element = options.dtype;
      const std::string inputType = "tensor " + sirShape(inputShape) + " " + element;
      const std::string outputType = "tensor " + sirShape(outputShape) + " " + element;
      const std::string reducedType =
          reducedShape.empty() ? element : "tensor " + sirShape(reducedShape) + " " + element;
      const std::string expandedType = "tensor " + sirShape(expandedShape) + " " + element;

      const std::array<std::string_view, 4> binaryOps{"add", "sub", "min", "max"};
      std::uniform_int_distribution<std::size_t> chooseBinary(0, binaryOps.size() - 1);
      const auto chooseOp = [&]() { return binaryOps[chooseBinary(random)]; };
      const bool useExp = options.dtype == "f32" && seed % 4 == 0;
      const auto optionalBudget = seed % (4 - static_cast<std::uint32_t>(useExp));
      const auto preOperationCount = optionalBudget / 2;
      const auto postOperationCount = optionalBudget - preOperationCount;

      std::ostringstream declarations;
      std::ostringstream instructions;
      std::vector<std::string> operators;
      const auto declare = [&](std::string_view name, const std::string &type) {
        declarations << "  let mut " << name << ": " << type << " = undef;\n";
      };
      const auto emitBinary = [&](std::string_view name, const std::string &type,
                                  std::string_view op, std::string_view lhs, std::string_view rhs) {
        declare(name, type);
        instructions << "  " << name << " = tensor " << op << ' ' << lhs << ", " << rhs << ";\n";
        operators.emplace_back(op);
      };

      emitBinary("%mixed", inputType, chooseOp(), "%a", "%b");
      emitBinary("%left", inputType, chooseOp(), "%mixed", "%a");
      emitBinary("%right", inputType, chooseOp(), "%mixed", "%b");
      emitBinary("%joined", inputType, chooseOp(), "%left", "%right");
      std::string current = "%joined";
      for (std::uint32_t index = 0; index < preOperationCount; ++index) {
        const auto name = "%pre" + std::to_string(index);
        const auto rhs = index % 2 == 0 ? "%a" : "%b";
        emitBinary(name, inputType, chooseOp(), current, rhs);
        current = name;
      }

      declare("%transposed", outputType);
      instructions << "  %transposed = tensor transpose " << current << ", "
                   << sirShape(Shape(permutation.begin(), permutation.end())) << ";\n";
      operators.emplace_back("transpose");
      std::string reductionInput = "%transposed";
      if (useExp) {
        declare("%activated", outputType);
        instructions << "  %activated = tensor exp %transposed;\n";
        operators.emplace_back("exp");
        reductionInput = "%activated";
      }
      declare("%rows", reducedType);
      instructions << "  %rows = tensor sum " << reductionInput << ", " << reductionAxis
                   << ", drop;\n";
      operators.emplace_back("sum");
      declare("%expanded", expandedType);
      instructions << "  %expanded = tensor expand %rows, " << reductionAxis << ";\n";
      operators.emplace_back("expand");
      declare("%broadcasted", outputType);
      instructions << "  %broadcasted = tensor broadcast %expanded, " << sirShape(outputShape)
                   << ";\n";
      operators.emplace_back("broadcast");

      emitBinary("%out", outputType, chooseOp(), "%transposed", "%broadcasted");
      current = "%out";
      for (std::uint32_t index = 0; index < postOperationCount; ++index) {
        const auto name = "%post" + std::to_string(index);
        const auto rhs = index % 2 == 0 ? "%transposed" : "%broadcasted";
        emitBinary(name, outputType, chooseOp(), current, rhs);
        current = name;
      }

      std::ostringstream source;
      source << "fun @kernel(%a: " << inputType << ", %b: " << inputType << ") : " << outputType
             << " {\n"
             << declarations.str() << "^entry:\n"
             << instructions.str() << "  ret " << current << ";\n}\n";
      candidate.solverSource = source.str();
      candidate.finalSource = candidate.solverSource;
      candidate.outputShape = outputType;
      candidate.outputElement = options.dtype == "i32" ? TensorElement::I32 : TensorElement::F32;
      candidate.operators = std::move(operators);
      candidate.shapeAxes = {{outputShape, static_cast<std::int32_t>(reductionAxis)}};
      candidate.fullShapes = {inputShape, outputShape, reducedShape, expandedShape};
      candidate.rangeExtents = candidate.fullShapes;
      candidate.propertySource =
          "Both runtime inputs contain finite values in [1,4], and the complete "
          "tensor computation is defined.";

      candidate.caseSeed = seed;
      std::uniform_int_distribution<std::int32_t> chooseA(1, 3);
      std::uniform_int_distribution<std::int32_t> chooseB(1, 4);
      std::vector<std::int32_t> valuesA(tensorElementCount(inputShape));
      std::vector<std::int32_t> valuesB(valuesA.size());
      for (std::size_t i = 0; i < valuesA.size(); ++i) {
        valuesA[i] = chooseA(random);
        valuesB[i] = chooseB(random);
      }
      if (candidate.outputElement == TensorElement::I32) {
        candidate.inputs.push_back({"a", TensorElement::I32, inputShape, encodeI32Tensor(valuesA)});
        candidate.inputs.push_back({"b", TensorElement::I32, inputShape, encodeI32Tensor(valuesB)});
      } else {
        std::vector<float> floatsA(valuesA.begin(), valuesA.end());
        std::vector<float> floatsB(valuesB.begin(), valuesB.end());
        candidate.inputs.push_back({"a", TensorElement::F32, inputShape, encodeF32Tensor(floatsA)});
        candidate.inputs.push_back({"b", TensorElement::F32, inputShape, encodeF32Tensor(floatsB)});
      }
      return candidate;
    }

    Candidate layoutCandidate(const TensorGenerationOptions &options, std::uint32_t seed) {
      if (options.dtype != "f32")
        throw std::runtime_error("bit-preserving-f32-layout requires dtype f32");

      Candidate candidate;
      std::mt19937 random(seed);
      const std::array<Shape, 4> shapes{Shape{2, 4}, Shape{4, 2}, Shape{2, 4, 2}, Shape{4, 2, 2}};
      std::uniform_int_distribution<std::size_t> chooseShape(0, shapes.size() - 1);
      const Shape inputShape = shapes[chooseShape(random)];
      std::vector<std::uint32_t> permutation(inputShape.size());
      std::iota(permutation.begin(), permutation.end(), 0u);
      do {
        std::shuffle(permutation.begin(), permutation.end(), random);
      } while (std::is_sorted(permutation.begin(), permutation.end()));
      Shape outputShape(inputShape.size());
      for (std::size_t axis = 0; axis < permutation.size(); ++axis)
        outputShape[axis] = inputShape[permutation[axis]];

      const auto inputType = tensorF32Type(inputShape);
      const auto outputType = tensorF32Type(outputShape);
      std::ostringstream source;
      source << "fun @kernel(%x: " << inputType << ") : " << outputType << " {\n"
             << "  let mut %out: " << outputType << " = undef;\n"
             << "^entry:\n"
             << "  %out = tensor transpose %x, "
             << sirShape(Shape(permutation.begin(), permutation.end())) << ";\n"
             << "  ret %out;\n}\n";
      candidate.solverSource = source.str();
      candidate.finalSource = candidate.solverSource;
      candidate.outputShape = outputType;
      candidate.outputElement = TensorElement::F32;
      candidate.operators = {"transpose"};
      candidate.exactnessKind = "bit-preserving-f32-layout";
      candidate.shapeAxes = {{inputShape, -1}};
      candidate.fullShapes = {inputShape, outputShape};
      candidate.rangeExtents = candidate.fullShapes;
      candidate.propertySource = "The finite runtime f32 tensor is transformed "
                                 "only by a bit-preserving transpose.";
      candidate.caseSeed = seed;

      constexpr std::array<std::uint32_t, 8> words{0x80000000u, 0x00000001u, 0x3f800000u,
                                                   0xc0200000u, 0x00000002u, 0x40000000u,
                                                   0xbf000000u, 0x00800000u};
      std::vector<std::uint8_t> bytes;
      const auto count = tensorElementCount(inputShape);
      bytes.reserve(count * sizeof(std::uint32_t));
      for (std::size_t i = 0; i < count; ++i)
        appendLittleWord(bytes, words[i % words.size()]);

      Shape strides(inputShape.size());
      for (std::size_t axis = 0; axis < permutation.size(); ++axis)
        strides[permutation[axis]] = outputShape[axis];
      std::uint32_t stride = 1;
      for (std::size_t axis = permutation.size(); axis > 0; --axis) {
        const auto inputAxis = permutation[axis - 1];
        strides[inputAxis] = stride;
        stride *= inputShape[inputAxis];
      }
      Shape memoryOrder(permutation.rbegin(), permutation.rend());
      candidate.inputs.push_back(
          {"x", TensorElement::F32, inputShape, std::move(bytes), std::move(strides), 0,
           static_cast<std::uint32_t>(count), std::move(memoryOrder)}
      );
      return candidate;
    }

    Candidate exactReductionCandidate(const TensorGenerationOptions &options, std::uint32_t seed) {
      if (options.dtype != "f32")
        throw std::runtime_error("exact-f32-reduction requires dtype f32");

      Candidate candidate;
      std::mt19937 random(seed);
      const std::array<Shape, 3> shapes{Shape{128, 4}, Shape{4, 128}, Shape{8, 8, 8}};
      std::uniform_int_distribution<std::size_t> chooseShape(0, shapes.size() - 1);
      const Shape inputShape = shapes[chooseShape(random)];
      std::uniform_int_distribution<std::uint32_t> chooseAxis(
          0, static_cast<std::uint32_t>(inputShape.size() - 1)
      );
      const auto axis = chooseAxis(random);
      Shape outputShape = inputShape;
      outputShape.erase(outputShape.begin() + axis);
      const auto inputType = tensorF32Type(inputShape);
      const auto outputType = outputShape.empty() ? "f32" : tensorF32Type(outputShape);
      std::ostringstream source;
      source << "fun @kernel(%x: " << inputType << ") : " << outputType << " {\n"
             << "  let mut %out: " << outputType << " = undef;\n"
             << "^entry:\n"
             << "  %out = tensor sum %x, " << axis << ", drop;\n"
             << "  ret %out;\n}\n";
      candidate.solverSource = source.str();
      candidate.finalSource = candidate.solverSource;
      candidate.outputShape = outputType;
      candidate.outputElement = TensorElement::F32;
      candidate.operators = {"sum"};
      candidate.exactnessKind = "integer-valued-f32-reduction";
      candidate.shapeAxes = {{inputShape, static_cast<std::int32_t>(axis)}};
      candidate.fullShapes = {inputShape, outputShape};
      candidate.rangeExtents = candidate.fullShapes;
      candidate.propertySource = "Every runtime leaf is a positive integer-valued "
                                 "finite f32 and the total absolute "
                                 "leaf sum is at most 2^24.";
      candidate.caseSeed = seed;

      constexpr std::uint32_t maxValue = 8;
      const auto count = tensorElementCount(inputShape);
      std::uniform_int_distribution<std::uint32_t> chooseValue(1, maxValue);
      std::vector<std::uint8_t> bytes;
      bytes.reserve(count * sizeof(std::uint32_t));
      for (std::size_t i = 0; i < count; ++i) {
        const auto value = chooseValue(random);
        candidate.exactnessMaxAbsLeafSum += value;
        appendLittleWord(bytes, std::bit_cast<std::uint32_t>(static_cast<float>(value)));
      }
      if (candidate.exactnessMaxAbsLeafSum > (1u << 24))
        throw std::runtime_error("exact f32 reduction proof exceeds the integer exactness bound");
      candidate.inputs.push_back({"x", TensorElement::F32, inputShape, std::move(bytes)});
      return candidate;
    }

    Candidate untargetedCandidate(const TensorGenerationOptions &options, std::uint32_t seed) {
      Candidate candidate;
      std::mt19937 random(seed);
      const std::array<Shape, 4> shapes{Shape{2, 4}, Shape{4, 2}, Shape{2, 4, 2}, Shape{4, 2, 2}};
      std::uniform_int_distribution<std::size_t> chooseShape(0, shapes.size() - 1);
      const Shape inputShape = shapes[chooseShape(random)];
      std::vector<std::uint32_t> permutation(inputShape.size());
      std::iota(permutation.begin(), permutation.end(), 0u);
      do {
        std::shuffle(permutation.begin(), permutation.end(), random);
      } while (std::is_sorted(permutation.begin(), permutation.end()));
      Shape transposedShape(inputShape.size());
      for (std::size_t axis = 0; axis < permutation.size(); ++axis)
        transposedShape[axis] = inputShape[permutation[axis]];
      std::uniform_int_distribution<std::uint32_t> chooseAxis(
          0, static_cast<std::uint32_t>(transposedShape.size() - 1)
      );
      const auto axis = chooseAxis(random);
      Shape reducedShape = transposedShape;
      reducedShape.erase(reducedShape.begin() + axis);
      Shape expandedShape = reducedShape;
      expandedShape.insert(expandedShape.begin() + axis, 1);
      const auto inputType = "tensor " + sirShape(inputShape) + " " + options.dtype;
      const auto transposedType = "tensor " + sirShape(transposedShape) + " " + options.dtype;
      const auto reducedType = reducedShape.empty()
                                   ? options.dtype
                                   : "tensor " + sirShape(reducedShape) + " " + options.dtype;
      const auto expandedType = "tensor " + sirShape(expandedShape) + " " + options.dtype;
      const auto outputType = transposedType;
      const std::string finalOp = (seed & 1u) ? "add" : "sub";
      std::ostringstream source;
      source << "fun @kernel(%x: " << inputType << ") : " << outputType << " {\n"
             << "  let mut %transposed: " << transposedType << " = undef;\n"
             << "  let mut %rows: " << reducedType << " = undef;\n"
             << "  let mut %expanded: " << expandedType << " = undef;\n"
             << "  let mut %broadcasted: " << transposedType << " = undef;\n"
             << "  let mut %out: " << outputType << " = undef;\n"
             << "^entry:\n"
             << "  %transposed = tensor transpose %x, "
             << sirShape(Shape(permutation.begin(), permutation.end())) << ";\n"
             << "  %rows = tensor sum %transposed, " << axis << ", drop;\n"
             << "  %expanded = tensor expand %rows, " << axis << ";\n"
             << "  %broadcasted = tensor broadcast %expanded, " << sirShape(transposedShape)
             << ";\n"
             << "  %out = tensor " << finalOp << " %transposed, %broadcasted;\n"
             << "  ret %out;\n}\n";
      candidate.solverSource = source.str();
      candidate.finalSource = candidate.solverSource;
      candidate.outputShape = outputType;
      candidate.outputElement = options.dtype == "i32" ? TensorElement::I32 : TensorElement::F32;
      candidate.operators = {"transpose", "sum", "expand", "broadcast", finalOp};
      candidate.shapeAxes = {{transposedShape, static_cast<std::int32_t>(axis)}};
      candidate.fullShapes = {inputShape, transposedShape, reducedShape, expandedShape};
      candidate.rangeExtents = candidate.fullShapes;
      candidate.propertySource =
          "A finite runtime tensor feeds a seeded, type-correct tensor graph.";
      candidate.caseSeed = seed;
      std::uniform_int_distribution<std::int32_t> chooseValue(1, 4);
      const auto count = tensorElementCount(inputShape);
      if (candidate.outputElement == TensorElement::I32) {
        std::vector<std::int32_t> values(count);
        for (auto &value: values)
          value = chooseValue(random);
        candidate.inputs.push_back({"x", TensorElement::I32, inputShape, encodeI32Tensor(values)});
      } else {
        std::vector<float> values(count);
        for (auto &value: values)
          value = static_cast<float>(chooseValue(random));
        candidate.inputs.push_back({"x", TensorElement::F32, inputShape, encodeF32Tensor(values)});
      }
      return candidate;
    }

    Candidate loopReductionCandidate(const TensorGenerationOptions &options, std::uint32_t seed) {
      Candidate candidate;
      const std::array<Shape, 4> shapes{Shape{256, 2}, Shape{128, 4}, Shape{64, 2}, Shape{64, 4}};
      std::mt19937 structureRandom(streamSeed(seed, 0x6c6f6f702d737472ULL));
      std::uniform_int_distribution<std::uint32_t> chooseLoopForm(0, 6);
      const std::uint32_t loopForm = chooseLoopForm(structureRandom);
      const bool nested = loopForm == 0;
      const std::uint32_t variant = nested ? 0 : loopForm - 1;
      std::uniform_int_distribution<std::size_t> chooseShape(0, shapes.size() - 1);
      Shape inputShape;
      if (!nested && variant == 0)
        inputShape = {256, 2};
      else if (!nested && variant == 4)
        inputShape = {8, 8};
      else if (!nested && variant == 5) {
        const std::array<Shape, 3> dynamicShapes{Shape{2, 4}, Shape{4, 2}, Shape{4, 4}};
        std::uniform_int_distribution<std::size_t> chooseDynamicShape(0, dynamicShapes.size() - 1);
        inputShape = dynamicShapes[chooseDynamicShape(structureRandom)];
      } else
        inputShape = shapes[chooseShape(structureRandom)];
      const std::uint32_t rowCount = inputShape[0];
      const std::uint32_t columnCount = inputShape[1];
      const Shape outputShape = variant == 5 ? inputShape : Shape{rowCount};
      std::uniform_int_distribution<std::int32_t> chooseSingleTripCount(2, 4);
      std::uniform_int_distribution<std::int32_t> chooseInnerTripCount(2, 3);
      const std::int32_t singleTripCount =
          !nested && variant == 0 ? 2 : chooseSingleTripCount(structureRandom);
      const std::int32_t innerTripCount = chooseInnerTripCount(structureRandom);
      const std::string element = options.dtype;
      const std::string inputType = "tensor " + sirShape(inputShape) + " " + element;
      const std::string outputType = "tensor " + sirShape(outputShape) + " " + element;
      const std::string zero = options.dtype == "f32" ? "0.0" : "0";
      const Shape transposedShape{columnCount, rowCount};
      const Shape expandedShape{rowCount, 1};
      const Shape rowReducedShape{rowCount};
      const Shape columnReducedShape{columnCount};
      const Shape columnExpandedShape{1, columnCount};
      const std::string transposedType = "tensor " + sirShape(transposedShape) + " " + element;
      const std::string expandedType = "tensor " + sirShape(expandedShape) + " " + element;
      const std::string rowReducedType = "tensor " + sirShape(rowReducedShape) + " " + element;
      const std::string columnReducedType =
          "tensor " + sirShape(columnReducedShape) + " " + element;
      const std::string columnExpandedType =
          "tensor " + sirShape(columnExpandedShape) + " " + element;
      std::ostringstream initializer;
      initializer << '{';
      for (std::size_t i = 0; i < rowCount; ++i) {
        if (i)
          initializer << ", ";
        initializer << zero;
      }
      initializer << '}';

      std::ostringstream source;
      source << "fun @kernel(%x: " << inputType;
      if (variant == 4 || variant == 5)
        source << ", %y: " << inputType;
      source << ") : " << outputType << " {\n";
      if (nested)
        source << "  let mut %outer: i32 = undef;\n"
               << "  let mut %inner: i32 = undef;\n";
      else
        source << "  let mut %i: i32 = undef;\n";
      if (variant == 4) {
        source << "  let mut %acc_x: " << outputType << " = " << initializer.str() << ";\n"
               << "  let mut %acc_y: " << outputType << " = " << initializer.str() << ";\n"
               << "  let mut %rows_x: " << outputType << " = undef;\n"
               << "  let mut %rows_y: " << outputType << " = undef;\n"
               << "  let mut %out: " << outputType << " = undef;\n";
      } else if (variant == 5) {
        source << "  let mut %acc_x: " << inputType << " = undef;\n"
               << "  let mut %acc_y: " << inputType << " = undef;\n"
               << "  let mut %rows_x: " << rowReducedType << " = undef;\n"
               << "  let mut %rows_y: " << columnReducedType << " = undef;\n"
               << "  let mut %expanded_x: " << expandedType << " = undef;\n"
               << "  let mut %expanded_y: " << columnExpandedType << " = undef;\n"
               << "  let mut %broadcast_x: " << inputType << " = undef;\n"
               << "  let mut %broadcast_y: " << inputType << " = undef;\n"
               << "  let mut %out: " << outputType << " = undef;\n";
      } else {
        source << "  let mut %acc: " << outputType << " = " << initializer.str() << ";\n"
               << "  let mut %rows: " << outputType << " = undef;\n";
      }
      if (variant == 3)
        source << "  let mut %transposed: " << transposedType << " = undef;\n"
               << "  let mut %transposed_rows: " << outputType << " = undef;\n"
               << "  let mut %expanded: " << expandedType << " = undef;\n"
               << "  let mut %broadcasted: " << inputType << " = undef;\n";
      source << "^entry:\n";
      if (variant == 5)
        source << "  %acc_x = %x;\n"
               << "  %acc_y = %y;\n";
      const auto emitRows = [&](std::ostringstream &out) {
        if (variant == 4) {
          out << "  %rows_x = tensor sum %x, 0, drop;\n"
              << "  %rows_y = tensor sum %y, 1, drop;\n";
        } else if (variant == 3) {
          out << "  %transposed = tensor transpose %x, [1, 0];\n"
              << "  %transposed_rows = tensor sum %transposed, 0, drop;\n"
              << "  %expanded = tensor expand %transposed_rows, 1;\n"
              << "  %broadcasted = tensor broadcast %expanded, " << sirShape(inputShape) << ";\n"
              << "  %rows = tensor sum %broadcasted, 1, drop;\n";
        } else {
          out << "  %rows = tensor sum %x, 1, drop;\n";
        }
      };
      if (variant == 2)
        emitRows(source);
      if (nested) {
        source << "  %outer = 0;\n"
               << "  br ^outer_header;\n"
               << "^outer_header:\n"
               << "  br %outer < 2, ^outer_body, ^exit;\n"
               << "^outer_body:\n"
               << "  %inner = 0;\n"
               << "  br ^inner_header;\n"
               << "^inner_header:\n"
               << "  br %inner < " << innerTripCount << ", ^body, ^outer_latch;\n"
               << "^body:\n";
        emitRows(source);
        source << "  %acc = tensor add %acc, %rows;\n"
               << "  %inner = %inner + 1;\n"
               << "  br ^inner_header;\n"
               << "^outer_latch:\n"
               << "  %outer = %outer + 1;\n"
               << "  br ^outer_header;\n";
      } else {
        source << "  %i = 0;\n"
               << "  br ^header;\n"
               << "^header:\n"
               << "  br %i < " << singleTripCount << ", ^body, ^exit;\n"
               << "^body:\n";
        if (variant == 5) {
          source << "  %rows_x = tensor sum %acc_x, 1, drop;\n"
                 << "  %rows_y = tensor sum %acc_y, 0, drop;\n"
                 << "  %expanded_x = tensor expand %rows_x, 1;\n"
                 << "  %expanded_y = tensor expand %rows_y, 0;\n"
                 << "  %broadcast_x = tensor broadcast %expanded_x, " << sirShape(inputShape)
                 << ";\n"
                 << "  %broadcast_y = tensor broadcast %expanded_y, " << sirShape(inputShape)
                 << ";\n";
        } else if (variant != 2) {
          emitRows(source);
        }
        if (variant == 4) {
          source << "  %acc_x = tensor add %acc_x, %rows_x;\n"
                 << "  %acc_y = tensor add %acc_y, %rows_y;\n"
                 << "  %acc_y = tensor add %acc_y, %acc_x;\n";
        } else if (variant == 5) {
          source << "  %acc_x = tensor add %acc_x, %broadcast_y;\n"
                 << "  %acc_y = tensor add %acc_y, %broadcast_x;\n";
        } else {
          source << "  %acc = tensor add %acc, %rows;\n";
          if (variant == 1)
            source << "  %acc = tensor add %acc, %rows;\n";
        }
        source << "  %i = %i + 1;\n"
               << "  br ^header;\n";
      }
      source << "^exit:\n";
      if (variant == 4 || variant == 5)
        source << "  %out = tensor add %acc_x, %acc_y;\n"
               << "  ret %out;\n}\n";
      else
        source << "  ret %acc;\n}\n";
      candidate.solverSource = source.str();
      candidate.finalSource = candidate.solverSource;
      candidate.outputShape = outputType;
      candidate.outputElement = options.dtype == "i32" ? TensorElement::I32 : TensorElement::F32;
      candidate.operators = {"sum", "add"};
      if (variant == 4) {
        candidate.operators = {"sum", "sum", "add", "add", "add"};
        candidate.shapeAxes = {{inputShape, 0}, {inputShape, 1}};
        candidate.fullShapes = {inputShape, inputShape, outputShape, outputShape};
        candidate.rangeExtents = candidate.fullShapes;
      } else if (variant == 5) {
        candidate.operators = {"sum",       "sum", "expand", "expand", "broadcast",
                               "broadcast", "add", "add",    "add"};
        candidate.shapeAxes = {{inputShape, 1}, {inputShape, 0}};
        candidate.fullShapes = {inputShape,         inputShape,    rowReducedShape,
                                columnReducedShape, expandedShape, columnExpandedShape,
                                inputShape,         inputShape,    outputShape};
        candidate.rangeExtents = candidate.fullShapes;
      } else if (variant == 3) {
        candidate.operators.insert(candidate.operators.end(), {"transpose", "expand", "broadcast"});
        candidate.shapeAxes = {{transposedShape, 0}, {inputShape, 1}};
        candidate.fullShapes = {inputShape,    transposedShape, outputShape,
                                expandedShape, inputShape,      outputShape};
        candidate.rangeExtents = candidate.fullShapes;
      } else {
        candidate.shapeAxes = {{inputShape, 1}};
        candidate.fullShapes = {inputShape, outputShape};
        candidate.rangeExtents = candidate.fullShapes;
      }
      if (variant == 1)
        candidate.operators.push_back("add");
      candidate.propertySource = "Each row has equal finite integer-valued inputs in [-4,259]; the "
                                 "reduction and accumulator remain defined.";
      if (variant == 4)
        candidate.propertySource =
            "Two finite runtime tensors with small positive integer-valued "
            "elements use distinct physical axis orders; their reductions and "
            "dependent accumulator updates remain defined.";
      if (variant == 5)
        candidate.propertySource =
            "Two finite runtime tensors of shape at most [4,4] contain values in "
            "[1,2]. Their loop-carried reductions, broadcasts, and updates remain "
            "defined without i32 overflow.";
      candidate.caseSeed = seed;
      candidate.loopBounds = nested ? std::vector<std::int32_t>{2, innerTripCount}
                                    : std::vector<std::int32_t>{singleTripCount};
      candidate.totalBodyExecutions = nested ? 2 + 2 * static_cast<std::uint32_t>(innerTripCount)
                                             : static_cast<std::uint32_t>(singleTripCount);
      if (variant == 4 || variant == 5) {
        std::mt19937 inputRandom(streamSeed(seed, 0x6c6f6f702d74776fULL));
        std::uniform_int_distribution<std::int32_t> chooseValue(1, variant == 5 ? 2 : 4);
        const auto inputCount = tensorElementCount(inputShape);
        std::vector<std::int32_t> valuesX(inputCount);
        std::vector<std::int32_t> valuesY(inputCount);
        for (std::size_t index = 0; index < inputCount; ++index) {
          valuesX[index] = chooseValue(inputRandom);
          valuesY[index] = chooseValue(inputRandom);
        }
        const auto appendInput = [&](std::string name, TensorElement elementType,
                                     std::vector<std::uint8_t> bytes, Shape strides,
                                     Shape axisOrder) {
          candidate.inputs.emplace_back(
              std::move(name), elementType, inputShape, std::move(bytes), std::move(strides), 0,
              static_cast<std::uint32_t>(inputCount), std::move(axisOrder)
          );
        };
        if (candidate.outputElement == TensorElement::I32) {
          appendInput("x", TensorElement::I32, encodeI32Tensor(valuesX), {1, rowCount}, {0, 1});
          appendInput("y", TensorElement::I32, encodeI32Tensor(valuesY), {columnCount, 1}, {1, 0});
        } else {
          std::vector<float> floatsX(valuesX.begin(), valuesX.end());
          std::vector<float> floatsY(valuesY.begin(), valuesY.end());
          appendInput("x", TensorElement::F32, encodeF32Tensor(floatsX), {1, rowCount}, {0, 1});
          appendInput("y", TensorElement::F32, encodeF32Tensor(floatsY), {columnCount, 1}, {1, 0});
        }
        return candidate;
      }
      std::mt19937 inputRandom(streamSeed(seed, 0x6c6f6f702d696e70ULL));
      std::uniform_int_distribution<std::int32_t> perturb(-4, 4);
      const auto inputCount = tensorElementCount(inputShape);
      if (candidate.outputElement == TensorElement::I32) {
        std::vector<std::int32_t> values;
        values.reserve(inputCount);
        for (std::int32_t row = 0; row < static_cast<std::int32_t>(rowCount); ++row) {
          const auto value = std::max(0, row + perturb(inputRandom));
          for (std::uint32_t column = 0; column < columnCount; ++column)
            values.push_back(value);
        }
        candidate.inputs.push_back(
            {"x",
             TensorElement::I32,
             inputShape,
             encodeI32Tensor(values),
             {1, rowCount},
             0,
             static_cast<std::uint32_t>(inputCount),
             {0, 1}}
        );
      } else {
        std::vector<float> values;
        values.reserve(inputCount);
        for (std::int32_t row = 0; row < static_cast<std::int32_t>(rowCount); ++row) {
          const auto value = static_cast<float>(std::max(0, row + perturb(inputRandom)));
          for (std::uint32_t column = 0; column < columnCount; ++column)
            values.push_back(value);
        }
        candidate.inputs.push_back(
            {"x",
             TensorElement::F32,
             inputShape,
             encodeF32Tensor(values),
             {1, rowCount},
             0,
             static_cast<std::uint32_t>(inputCount),
             {0, 1}}
        );
      }
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

    Candidate v3Candidate(const TensorGenerationOptions &options,
                          std::uint32_t seed, std::uint32_t caseIndex,
                          std::string_view stratum) {
      Candidate candidate;
      candidate.caseSeed = seed;
      candidate.inputPattern = options.inputPattern;
      const auto targetShape = v3BoundaryShape(caseIndex);
      const std::string element = options.dtype;
      const auto tensorType = [&](const Shape &shape) {
        return "tensor " + sirShape(shape) + " " + element;
      };
      const auto scalarType = element;

      Shape inputShape = targetShape;
      Shape outputShape = targetShape;
      std::string operationBody;
      std::string returnValue = "%out";
      std::string outputType;
      std::string localDeclarations;
      const std::string zero = element == "f32" ? "0.0" : "0";
      const std::string two = element == "f32" ? "2.0" : "2";
      const auto axis = targetShape.size() == 1 ? 0u : (caseIndex - 5) % 2;
      const auto makeInput = [&](const Shape &shape) {
        auto values = v3PatternValues(shape, options.inputPattern, seed);
        TensorCaseBuffer input{"x", element == "i32" ? TensorElement::I32
                                                        : TensorElement::F32,
                               shape, {}};
        if (element == "i32") {
          input.bytes = encodeI32Tensor(values);
        } else {
          std::vector<float> floats(values.begin(), values.end());
          input.bytes = encodeF32Tensor(floats);
        }
        assignV3PhysicalLayout(input, seed ^ 0x696e707574ULL);
        candidate.inputs.push_back(std::move(input));
      };

      if (stratum == "tail-arithmetic" || stratum == "tail-untargeted") {
        outputType = tensorType(outputShape);
        localDeclarations = "  let mut %tmp: " + tensorType(outputShape) + " = undef;\n";
        if (stratum == "tail-arithmetic") {
          operationBody = "  %out = tensor mul %x, " + two + ";\n";
          candidate.operators = {"mul"};
        } else {
          operationBody = "  %tmp = tensor mul %x, " + two + ";\n"
                          "  %out = tensor add %tmp, %x;\n";
          candidate.operators = {"mul", "add"};
        }
        candidate.propertySource = "The generated arithmetic result is defined for every logical element.";
      } else if (stratum == "tail-sum") {
        Shape reducedShape = targetShape;
        reducedShape.erase(reducedShape.begin() + axis);
        outputShape = reducedShape;
        outputType = outputShape.empty() ? scalarType : tensorType(outputShape);
        operationBody = "  %out = tensor sum %x, " + std::to_string(axis) + ", drop;\n";
        candidate.operators = {"sum"};
        candidate.shapeAxes = {{targetShape, static_cast<std::int32_t>(axis)}};
        candidate.propertySource = "The output equals the logical sum along the selected axis.";
      } else if (stratum == "tail-layout") {
        outputShape = targetShape;
        if (targetShape.size() == 2) {
          outputShape = {targetShape[1], targetShape[0]};
          outputType = tensorType(outputShape);
          operationBody = "  %out = tensor transpose %x, [1, 0];\n";
          candidate.operators = {"transpose"};
          if (element == "f32")
            candidate.exactnessKind = "bit-preserving-f32-layout";
        } else {
          outputType = tensorType(outputShape);
          operationBody = "  %out = %x;\n";
          candidate.operators = {"copy"};
        }
        candidate.propertySource = "The logical output preserves the input values under the declared layout operation.";
      } else if (stratum == "tail-broadcast") {
        outputShape = targetShape;
        if (targetShape.size() == 1) {
          inputShape = {1};
          outputShape = {targetShape[0], 1};
          outputType = tensorType(outputShape);
          localDeclarations = "  let mut %expanded: " + tensorType(Shape{1, 1}) + " = undef;\n";
          operationBody = "  %expanded = tensor expand %x, 1;\n"
                          "  %out = tensor broadcast %expanded, " + sirShape(outputShape) + ";\n";
          candidate.operators = {"expand", "broadcast"};
        } else {
          inputShape = targetShape[0] < targetShape[1]
                           ? Shape{1, targetShape[1]}
                           : Shape{targetShape[0], 1};
          outputType = tensorType(outputShape);
          operationBody = "  %out = tensor broadcast %x, " + sirShape(outputShape) + ";\n";
          candidate.operators = {"broadcast"};
        }
        candidate.propertySource = "Broadcast values are defined at every logical output coordinate.";
      } else if (stratum == "tail-loop") {
        outputShape.erase(outputShape.begin() + axis);
        outputType = outputShape.empty() ? scalarType : tensorType(outputShape);
        const auto reductionType = outputShape.empty() ? scalarType : tensorType(outputShape);
        localDeclarations = "  let mut %i: i32 = undef;\n"
                            "  let mut %acc: " + tensorType(inputShape) + " = undef;\n"
                            "  let mut %partial: " + reductionType + " = undef;\n";
        operationBody = "  %acc = tensor mul %x, " + zero + ";\n";
        if (outputShape.empty()) {
          operationBody += "  %out = " + zero + ";\n";
        } else {
          operationBody += "  %out = tensor sum %acc, " + std::to_string(axis) + ", drop;\n";
        }
        operationBody += "  %i = 0;\n"
                        "  br ^header;\n"
                        "^header:\n"
                        "  br %i < 3, ^body, ^exit;\n"
                        "^body:\n"
                        "  %acc = tensor add %acc, %x;\n"
                        "  %partial = tensor sum %acc, " + std::to_string(axis) + ", drop;\n";
        if (outputShape.empty()) {
          operationBody += "  %out = %out + %partial;\n";
        } else {
          operationBody += "  %out = tensor add %out, %partial;\n";
        }
        operationBody += "  %i = %i + 1;\n"
                         "  br ^header;\n"
                         "^exit:\n";
        candidate.operators = {"mul", "add", "sum"};
        candidate.shapeAxes = {{targetShape, static_cast<std::int32_t>(axis)}};
        candidate.loopBounds = {3};
        candidate.totalBodyExecutions = 3;
        candidate.propertySource =
            "The loop-carried tensor is reduced on the selected logical axis each iteration.";
      } else {
        throw std::runtime_error("tensor-v3 has no constructor for stratum " +
                                 std::string(stratum));
      }

      const std::string inputType = tensorType(inputShape);
      std::ostringstream source;
      source << "fun @kernel(%x: " << inputType << ") : " << outputType << " {\n"
             << localDeclarations
             << "  let mut %out: " << outputType << " = undef;\n"
             << "^entry:\n";
      if (stratum == "tail-loop")
        source << operationBody;
      else
        source << operationBody << "  ret " << returnValue << ";\n}\n";
      if (stratum == "tail-loop")
        source << "  ret %out;\n}\n";
      candidate.solverSource = source.str();
      candidate.finalSource = candidate.solverSource;
      candidate.outputShape = outputType;
      candidate.outputElement = element == "i32" ? TensorElement::I32 : TensorElement::F32;
      candidate.fullShapes = {inputShape, outputShape};
      candidate.rangeExtents = candidate.fullShapes;
      candidate.caseSeed = seed;
      candidate.inputs.clear();
      makeInput(inputShape);

      if (stratum == "tail-sum" && element == "f32" &&
          options.inputPattern != "seeded-random") {
        candidate.exactnessKind = "integer-valued-f32-reduction";
        const auto values = v3PatternValues(inputShape, options.inputPattern, seed);
        for (const auto value: values)
          candidate.exactnessMaxAbsLeafSum += static_cast<std::uint64_t>(std::abs(value));
        if (candidate.exactnessMaxAbsLeafSum > (1u << 24)) {
          candidate.exactnessKind.clear();
          candidate.exactnessMaxAbsLeafSum = 0;
        }
      }
      return candidate;
    }

    TensorCase validateCandidate(
        Candidate candidate, const TensorGenerationOptions &options, const std::string &propertyId,
        const std::string &propertySource
    ) {
      Program program = checkedProgram(candidate.finalSource, options.profileVersion);
      const auto &entry = entryFunction(program);
      TensorCase testCase;
      testCase.profileVersion = options.profileVersion;
      testCase.primaryStratum = options.primaryStratum;
      testCase.source = candidate.finalSource;
      testCase.target = options.target;
      testCase.seed = candidate.caseSeed.value_or(options.seed);
      constexpr std::array<std::uint32_t, 4> warpCounts = {1, 2, 4, 8};
      testCase.numWarps =
          options.profileVersion != TensorProfileId::V1
              ? warpCounts[splitMix64(testCase.seed ^ 0x776172702d636667ULL) % warpCounts.size()]
              : 4;
      testCase.synthesis = options.synthesis;
      testCase.propertyId = propertyId;
      testCase.propertySource =
          candidate.propertySource.empty() ? propertySource : candidate.propertySource;
      testCase.inputs = std::move(candidate.inputs);
      testCase.exactnessKind = std::move(candidate.exactnessKind);
      testCase.exactnessMaxAbsLeafSum = candidate.exactnessMaxAbsLeafSum;
      testCase.inputPattern = std::move(candidate.inputPattern);
      testCase.outputShape =
          TypeUtils::asTensor(entry.retType) ? TypeUtils::asTensor(entry.retType)->shape : Shape{};
      testCase.outputElement = scalarElement(
          TypeUtils::asTensor(entry.retType) ? TypeUtils::asTensor(entry.retType)->elem
                                             : entry.retType
      );
      if (options.profileVersion == TensorProfileId::V3 && !testCase.outputShape.empty()) {
        TensorCaseBuffer output{"return", testCase.outputElement, testCase.outputShape, {}};
        assignV3PhysicalLayout(output, testCase.seed ^ 0x6f7574707574ULL);
        testCase.outputElementStrides = std::move(output.elementStrides);
        testCase.outputElementOffset = output.elementOffset;
        testCase.outputStorageSpan = output.storageSpan;
        testCase.outputAxisOrder = std::move(output.axisOrder);
      }
      testCase.operators = std::move(candidate.operators);
      testCase.shapeAxes = std::move(candidate.shapeAxes);
      testCase.loopBounds = std::move(candidate.loopBounds);
      testCase.totalBodyExecutions = candidate.totalBodyExecutions;
      testCase.fullShapes = std::move(candidate.fullShapes);
      testCase.rangeExtents = std::move(candidate.rangeExtents);
      testCase.numericComparison =
          testCase.outputElement == TensorElement::F32 && testCase.exactnessKind.empty();
      if (testCase.numericComparison)
        testCase.propertyThresholdF32Bits = "3f4ccccd";
      testCase.evidenceKind = options.synthesis == "exact" ? "exact_smt"
                              : testCase.numericComparison ? "numerical_reference"
                                                           : "construction";
      testCase.evidenceStatus = options.synthesis == "exact" ? "exact_smt_checked"
                                : testCase.numericComparison ? "numerical_reference_checked"
                                                             : "construction_checked";

      if (testCase.outputElement == TensorElement::F32) {
        std::vector<TensorTraceEntry> trace128;
        std::vector<TensorTraceEntry> trace256;
        const auto output128 = runReference(program, testCase.inputs, 128, &trace128);
        const auto output256 = runReference(program, testCase.inputs, 256, &trace256);
        if (output128 != output256 || !tensorTracesEqual(trace128, trace256))
          throw std::runtime_error("unstable_reference: 128-bit and 256-bit results differ");
        testCase.expected = testCase.numericComparison ? output256 : output128;
        testCase.referenceStable = true;
      } else {
        testCase.expected = runReference(program, testCase.inputs, 128, nullptr);
      }
      if (options.profileVersion == TensorProfileId::V2 && testCase.inputs.size() > 1) {
        for (std::size_t inputIndex = 0; inputIndex < testCase.inputs.size(); ++inputIndex) {
          const auto witness =
              inputCounterfactualWitness(program, testCase.inputs, inputIndex, testCase.expected);
          if (!witness)
            throw std::runtime_error(
                "counterfactual_dependency: returned computation does not depend "
                "on input " +
                testCase.inputs[inputIndex].name
            );
          testCase.inputCounterfactualWitnesses.emplace_back(
              testCase.inputs[inputIndex].name, *witness
          );
        }
      }
      return testCase;
    }

    std::string inferredProperty(const TensorGenerationOptions &options) {
      if (!options.property.empty())
        return options.property;
      if (options.profileVersion == TensorProfileId::V2 ||
          options.profileVersion == TensorProfileId::V3)
        return "defined-inputs";
      return options.dtype == "i32" ? "coordinates" : "softmax-dominance";
    }

    std::string propertySource(std::string_view property) {
      if (property == "defined-inputs")
        return "Every runtime input is materialized, finite when floating, and "
               "defined.";
      if (property == "coordinates")
        return "On tensor [2,4], output[r,c] = 7*r + 3*c + 1.";
      if (property == "sum-target")
        return "For tensor [2,4], row sums equal [10,-6].";
      return "On tensor [2,4], softmax row 0 column 3 and row 1 column 0 exceed "
             "4/5.";
    }

    bool isCandidateRejection(std::string_view message) {
      return message.starts_with("UB:") ||
             message.find("Requirement failed") != std::string_view::npos ||
             message.find("unstable_reference:") == 0 ||
             message.find("counterfactual_dependency:") == 0;
    }

    std::string v2StratumForCase(std::uint32_t caseIndex) {
      return std::string(v2StratumSchedule[caseIndex % v2StratumSchedule.size()]);
    }

    bool validV2Stratum(std::string_view stratum) {
      return stratum == "bit-preserving-f32-layout" || stratum == "exact-f32-reduction" ||
             stratum == "loop-reduction" || stratum == "multi-input-composition" ||
             stratum == "untargeted-exploration";
    }

    std::string v3StratumForCase(std::uint32_t caseIndex) {
      return std::string(v3StratumSchedule[caseIndex % v3StratumSchedule.size()]);
    }

    std::string v3InputPatternForCase(std::uint32_t caseIndex) {
      return std::string(v3InputPatterns[caseIndex % v3InputPatterns.size()]);
    }

    bool validV3Stratum(std::string_view stratum) {
      return std::find(v3StratumSchedule.begin(), v3StratumSchedule.end(), stratum) !=
             v3StratumSchedule.end();
    }

    bool validV3InputPattern(std::string_view pattern) {
      return std::find(v3InputPatterns.begin(), v3InputPatterns.end(), pattern) !=
             v3InputPatterns.end();
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
    if (options.profileVersion == TensorProfileId::V2 && !options.primaryStratum.empty() &&
        !validV2Stratum(options.primaryStratum))
      return failRoute(
          "invalid_profile", "unknown tensor-v2 primary stratum: " + options.primaryStratum
      );
    if (options.profileVersion == TensorProfileId::V3 && !options.primaryStratum.empty() &&
        !validV3Stratum(options.primaryStratum))
      return failRoute(
          "invalid_profile", "unknown tensor-v3 primary stratum: " + options.primaryStratum
      );
    if (options.profileVersion == TensorProfileId::V3 && !options.inputPattern.empty() &&
        !validV3InputPattern(options.inputPattern))
      return failRoute(
          "invalid_profile", "unknown tensor-v3 input pattern: " + options.inputPattern
      );
    if (options.profileVersion != TensorProfileId::V3 && !options.inputPattern.empty())
      return failRoute("invalid_profile", "tensor input patterns are available only in tensor-v3");
    if (options.dtype != "i32" && options.dtype != "f32")
      return failRoute("invalid_profile", "tensor dtype must be i32 or f32");
    if (options.profileVersion == TensorProfileId::V1 && property != "coordinates" &&
        property != "sum-target" && property != "softmax-dominance")
      return failRoute("invalid_profile", "unknown tensor property family: " + property);
    if (options.profileVersion == TensorProfileId::V2 && property != "defined-inputs")
      return failRoute("invalid_profile", "tensor-v2 composition requires defined-inputs property");
    if (options.profileVersion == TensorProfileId::V3 && property != "defined-inputs")
      return failRoute("invalid_profile", "tensor-v3 generation requires defined-inputs property");
    if (options.profileVersion == TensorProfileId::V1 &&
        (property == "coordinates" || property == "sum-target") && options.dtype != "i32")
      return failRoute("invalid_profile", property + " requires tensor dtype i32");
    if (options.profileVersion == TensorProfileId::V1 && property == "softmax-dominance" &&
        options.dtype != "f32")
      return failRoute("invalid_profile", "softmax-dominance requires tensor dtype f32");
    if (options.synthesis != "construction" && options.synthesis != "exact" &&
        options.synthesis != "cvc5")
      return failRoute(
          "unsupported_route", "tensor synthesis must be construction, exact, or cvc5"
      );
    if (options.profileVersion != TensorProfileId::V1 && options.synthesis != "construction")
      return failRoute(
          "unsupported_route", "tensor-v2/v3 graph generation currently requires construction"
      );
    if (options.profileVersion == TensorProfileId::V1 && options.synthesis == "exact" &&
        options.dtype == "f32")
      return failRoute(
          "unsupported_theory", "exact synthesis does not support f32 softmax numerical operators"
      );
    if (options.synthesis == "cvc5")
      return failRoute(
          "unavailable", "cvc5 tensor feasibility gate did not pass "
                         "in the captured environment"
      );
    if (options.count == 0)
      return failRoute("invalid_profile", "tensor case count must be at least one");

    std::mt19937 random(options.seed);
    for (std::uint32_t caseIndex = 0; caseIndex < options.count; ++caseIndex) {
      bool accepted = false;
      const std::string caseProperty = property;
      const std::string caseStratum =
          options.profileVersion == TensorProfileId::V2
              ? (options.primaryStratum.empty() ? v2StratumForCase(caseIndex)
                                                : options.primaryStratum)
              : options.profileVersion == TensorProfileId::V3
                  ? (options.primaryStratum.empty() ? v3StratumForCase(caseIndex)
                                                    : options.primaryStratum)
              : std::string{};
      TensorGenerationOptions caseOptions = options;
      caseOptions.primaryStratum = caseStratum;
      if (options.profileVersion == TensorProfileId::V3)
        caseOptions.inputPattern = options.inputPattern.empty()
                                       ? v3InputPatternForCase(caseIndex)
                                       : options.inputPattern;
      if (options.profileVersion == TensorProfileId::V2 &&
          (caseStratum == "bit-preserving-f32-layout" || caseStratum == "exact-f32-reduction") &&
          caseOptions.dtype != "f32") {
        result.status = "shortfall";
        result.reason =
            "stratum " + caseStratum + " requires f32; requested dtype is " + caseOptions.dtype;
        attempts.push_back({caseIndex, 0, caseProperty, caseStratum, "unavailable", result.reason});
        writeSummary(
            options.outputDirectory / "generation.json", recordedOptions, result, attempts
        );
        return result;
      }
      for (std::uint32_t attempt = 0; attempt < options.attemptBudget; ++attempt) {
        const auto attemptSeed = options.seed + caseIndex * 0x9e3779b9U + attempt;
        Candidate candidate;
        try {
          if (options.profileVersion == TensorProfileId::V1) {
            if (property == "coordinates")
              candidate = coordinatesCandidate(options, random);
            else if (property == "sum-target")
              candidate = sumTargetCandidate(options, attemptSeed);
            else
              candidate = softmaxCandidate(options.seed + caseIndex, attempt);
          } else if (caseStratum == "multi-input-composition") {
            candidate = compositionCandidate(caseOptions, attemptSeed);
          } else if (caseStratum == "loop-reduction") {
            candidate = loopReductionCandidate(caseOptions, attemptSeed);
          } else if (caseStratum == "bit-preserving-f32-layout") {
            candidate = layoutCandidate(caseOptions, attemptSeed);
          } else if (caseStratum == "exact-f32-reduction") {
            candidate = exactReductionCandidate(caseOptions, attemptSeed);
          } else if (caseStratum == "untargeted-exploration") {
            candidate = untargetedCandidate(caseOptions, attemptSeed);
          } else if (options.profileVersion == TensorProfileId::V3) {
            candidate = v3Candidate(caseOptions, attemptSeed, caseIndex, caseStratum);
          } else {
            result.status = "stratum_unavailable";
            result.reason = "no constructor is registered for primary stratum " + caseStratum;
            attempts.push_back(
                {caseIndex, attempt, caseProperty, caseStratum, "unavailable", result.reason}
            );
            writeSummary(
                options.outputDirectory / "generation.json", recordedOptions, result, attempts
            );
            return result;
          }

          auto caseData = validateCandidate(
              std::move(candidate), caseOptions, caseProperty, propertySource(caseProperty)
          );
          const auto directory = options.outputDirectory / ("case_" + [&] {
                                   std::ostringstream index;
                                   index << std::setw(6) << std::setfill('0') << caseIndex;
                                   return index.str();
                                 }());
          writeTensorCase(directory, caseData);
          ++result.emitted;
          attempts.push_back({caseIndex, attempt, caseProperty, caseStratum, "accepted", {}});
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
          attempts.push_back({caseIndex, attempt, caseProperty, caseStratum, "rejected", message});
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
