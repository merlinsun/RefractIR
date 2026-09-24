#include "reify/tensor_case.hpp"

#include <atomic>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>

#include "analysis/tensor.hpp"
#include "backend/emit.hpp"
#include "frontend/pipeline.hpp"

namespace refractir::reify {
  namespace {

    std::atomic<std::uint64_t> stagingId{0};

    std::string jsonString(std::string_view value) {
      std::string result{"\""};
      constexpr char hex[] = "0123456789abcdef";
      for (const unsigned char ch: value) {
        switch (ch) {
          case '"':
            result += "\\\"";
            break;
          case '\\':
            result += "\\\\";
            break;
          case '\b':
            result += "\\b";
            break;
          case '\f':
            result += "\\f";
            break;
          case '\n':
            result += "\\n";
            break;
          case '\r':
            result += "\\r";
            break;
          case '\t':
            result += "\\t";
            break;
          default:
            if (ch < 0x20) {
              result += "\\u00";
              result.push_back(hex[ch >> 4]);
              result.push_back(hex[ch & 15]);
            } else {
              result.push_back(static_cast<char>(ch));
            }
        }
      }
      result.push_back('"');
      return result;
    }

    std::string elementName(TensorElement element) {
      return element == TensorElement::I32 ? "i32" : "f32";
    }

    std::uint64_t fnv1a64(std::string_view bytes) {
      std::uint64_t hash = 14695981039346656037ULL;
      for (const unsigned char byte: bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
      }
      return hash;
    }

    std::string hexHash(std::string_view bytes) {
      std::ostringstream out;
      out << std::hex << std::setw(16) << std::setfill('0') << fnv1a64(bytes);
      return out.str();
    }

    std::string readFile(const std::filesystem::path &path) {
      std::ifstream input(path, std::ios::binary);
      if (!input)
        throw std::runtime_error("cannot read tensor case artifact: " + path.string());
      std::ostringstream contents;
      contents << input.rdbuf();
      return contents.str();
    }

    void writeFile(const std::filesystem::path &path, std::string_view bytes) {
      std::ofstream output(path, std::ios::binary | std::ios::trunc);
      if (!output)
        throw std::runtime_error("cannot create tensor case artifact: " + path.string());
      output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      if (!output)
        throw std::runtime_error("failed to write tensor case artifact: " + path.string());
    }

    void writeShape(std::ostream &out, const Shape &shape) {
      out << '[';
      for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index)
          out << ',';
        out << shape[index];
      }
      out << ']';
    }

    Shape contiguousStrides(const Shape &shape) {
      Shape strides(shape.size());
      std::uint32_t stride = 1;
      for (std::size_t i = shape.size(); i > 0; --i) {
        strides[i - 1] = stride;
        stride *= shape[i - 1];
      }
      return strides;
    }

    struct InputLocation {
      std::uint64_t offset = 0;
      std::uint64_t length = 0;
    };

    InputLocation appendInputs(std::string &bytes, const TensorCaseBuffer &input) {
      InputLocation location;
      location.offset = bytes.size();
      location.length = input.bytes.size();
      bytes.append(reinterpret_cast<const char *>(input.bytes.data()), input.bytes.size());
      return location;
    }

    void writeBufferRecord(
        std::ostream &out, const TensorCaseBuffer &buffer, std::uint64_t offset,
        std::uint64_t length, std::string_view file, bool comma
    ) {
      if (comma)
        out << ',';
      out << "{\"name\":" << jsonString(buffer.name)
          << ",\"dtype\":" << jsonString(elementName(buffer.element)) << ",\"shape\":";
      writeShape(out, buffer.shape);
      out << ",\"file\":" << jsonString(file) << ",\"byte_offset\":" << offset
          << ",\"byte_length\":" << length << ",\"element_strides\":";
      writeShape(out, contiguousStrides(buffer.shape));
      out << ",\"element_offset\":0,\"tile_shape\":";
      writeShape(out, buffer.shape);
      out << ",\"tile_origin\":[";
      for (std::size_t i = 0; i < buffer.shape.size(); ++i) {
        if (i)
          out << ',';
        out << '0';
      }
      out << "]}";
    }

    std::string buildManifest(
        const TensorCase &testCase, const std::string &program, const std::string &inputs,
        const std::string &expected, const std::string &kernel
    ) {
      std::ostringstream out;
      out << "{\"schema\":\"refractir.tensor-case.v1\","
          << "\"profile\":\"tensor-v1\",\"entry\":\"kernel\",\"inputs\":[";
      std::string inputFile;
      std::vector<InputLocation> locations;
      inputFile.reserve(inputs.size());
      std::uint64_t offset = 0;
      for (const auto &input: testCase.inputs) {
        locations.push_back({offset, input.bytes.size()});
        offset += input.bytes.size();
      }
      for (std::size_t i = 0; i < testCase.inputs.size(); ++i)
        writeBufferRecord(
            out, testCase.inputs[i], locations[i].offset, locations[i].length, "inputs.bin", i != 0
        );

      TensorCaseBuffer output{"return", testCase.outputElement, testCase.outputShape, {}};
      out << "],\"output\":";
      writeBufferRecord(out, output, 0, expected.size(), "expected.bin", false);
      out << ",\"grid\":[1,1,1],\"softmax_lowering\":\"direct\","
          << "\"launch\":{\"num_warps\":4,\"num_stages\":1,"
          << "\"enable_fp_fusion\":false},\"property\":{\"id\":" << jsonString(testCase.propertyId)
          << ",\"source\":" << jsonString(testCase.propertySource);
      if (!testCase.propertyThresholdF32Bits.empty())
        out << ",\"threshold_f32_bits\":" << jsonString(testCase.propertyThresholdF32Bits);
      out << "},\"evidence\":{\"kind\":" << jsonString(testCase.evidenceKind)
          << ",\"status\":" << jsonString(testCase.evidenceStatus)
          << ",\"artifacts\":[\"program.sir\",\"inputs.bin\",\"expected.bin\"";
      if (testCase.target == "triton")
        out << ",\"kernel.py\"";
      out << "]}"
          << ",\"comparison\":{\"mode\":"
          << jsonString(testCase.numericComparison ? "numeric" : "exact") << ",\"atol\":";
      if (testCase.numericComparison)
        out << "[2,100000]";
      else
        out << "[0,1]";
      out << ",\"rtol\":";
      if (testCase.numericComparison)
        out << "[2,1000]";
      else
        out << "[0,1]";
      out << "},\"reference\":{\"stable\":" << (testCase.referenceStable ? "true" : "false")
          << ",\"precisions\":[";
      if (testCase.numericComparison)
        out << "128,256";
      out << "]},\"environment\":null,\"operator_coverage\":[";
      for (std::size_t i = 0; i < testCase.operators.size(); ++i) {
        if (i)
          out << ',';
        out << jsonString(testCase.operators[i]);
      }
      out << "],\"shape_axes\":[";
      for (std::size_t i = 0; i < testCase.shapeAxes.size(); ++i) {
        if (i)
          out << ',';
        out << "{\"shape\":";
        writeShape(out, testCase.shapeAxes[i].first);
        out << ",\"axis\":" << testCase.shapeAxes[i].second << '}';
      }
      out << "],\"generator\":{\"version\":\"tensor-v1\",\"seed\":" << testCase.seed
          << ",\"synthesis\":" << jsonString(testCase.synthesis) << ",\"node_limit\":8},\"files\":{"
          << "\"program.sir\":{\"bytes\":" << program.size()
          << ",\"fnv1a64\":" << jsonString(hexHash(program)) << "},"
          << "\"inputs.bin\":{\"bytes\":" << inputs.size()
          << ",\"fnv1a64\":" << jsonString(hexHash(inputs)) << "},"
          << "\"expected.bin\":{\"bytes\":" << expected.size()
          << ",\"fnv1a64\":" << jsonString(hexHash(expected)) << '}';
      if (!kernel.empty())
        out << ",\"kernel.py\":{\"bytes\":" << kernel.size()
            << ",\"fnv1a64\":" << jsonString(hexHash(kernel)) << '}';
      out << "}}";
      return out.str();
    }

  } // namespace

  void writeTensorCase(const std::filesystem::path &directory, const TensorCase &testCase) {
    if (std::filesystem::exists(directory))
      throw std::runtime_error("tensor case directory already exists: " + directory.string());
    if (testCase.target != "sir" && testCase.target != "triton")
      throw std::invalid_argument("tensor case target must be sir or triton");
    std::filesystem::create_directories(directory.parent_path());
    auto staging = directory;
    staging += ".tmp." + std::to_string(stagingId.fetch_add(1));
    if (std::filesystem::exists(staging))
      throw std::runtime_error("tensor case staging path already exists: " + staging.string());
    std::filesystem::create_directory(staging);

    try {
      writeFile(staging / "program.sir", testCase.source);
      std::string inputBytes;
      for (const auto &input: testCase.inputs)
        (void) appendInputs(inputBytes, input);
      const std::string expectedBytes(
          reinterpret_cast<const char *>(testCase.expected.data()), testCase.expected.size()
      );
      writeFile(staging / "inputs.bin", inputBytes);
      writeFile(staging / "expected.bin", expectedBytes);

      std::string kernel;
      if (testCase.target == "triton") {
        Program program = parseSource(testCase.source);
        DiagBag diagnostics;
        if (!checkProgram(program, diagnostics))
          throw std::runtime_error("cannot emit Triton kernel from unchecked tensor source");
        EmitOptions emitOptions;
        emitOptions.entry = "kernel";
        emitOptions.noUbGuards = true;
        if (!emitTriton(program, staging / "kernel.py", emitOptions))
          throw std::runtime_error("Triton kernel emission failed for generated tensor case");
        kernel = readFile(staging / "kernel.py");
      }

      writeFile(
          staging / "case.json",
          buildManifest(testCase, testCase.source, inputBytes, expectedBytes, kernel)
      );
      std::filesystem::rename(staging, directory);
    } catch (...) {
      std::error_code ec;
      std::filesystem::remove_all(staging, ec);
      throw;
    }
  }

} // namespace refractir::reify
