#include "reify/tensor_case.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

#include "analysis/tensor.hpp"
#include "backend/emit.hpp"
#include "frontend/lexer.hpp"
#include "frontend/pipeline.hpp"

namespace refractir::reify {
namespace {

std::atomic<std::uint64_t> stagingId{0};

std::string jsonString(std::string_view value) {
  std::string result{"\""};
  constexpr char hex[] = "0123456789abcdef";
  for (const unsigned char ch : value) {
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
  for (const unsigned char byte : bytes) {
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

std::string sha256(std::string_view bytes) {
  constexpr std::array<std::uint32_t, 64> constants{
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
      0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
      0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  std::array<std::uint32_t, 8> state{0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                     0xa54ff53a, 0x510e527f, 0x9b05688c,
                                     0x1f83d9ab, 0x5be0cd19};

  std::vector<std::uint8_t> padded(bytes.begin(), bytes.end());
  const auto bitLength = static_cast<std::uint64_t>(padded.size()) * 8;
  padded.push_back(0x80);
  while (padded.size() % 64 != 56)
    padded.push_back(0);
  for (int shift = 56; shift >= 0; shift -= 8)
    padded.push_back(static_cast<std::uint8_t>(bitLength >> shift));

  for (std::size_t block = 0; block < padded.size(); block += 64) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t i = 0; i < 16; ++i) {
      const std::size_t offset = block + i * 4;
      words[i] = (static_cast<std::uint32_t>(padded[offset]) << 24) |
                 (static_cast<std::uint32_t>(padded[offset + 1]) << 16) |
                 (static_cast<std::uint32_t>(padded[offset + 2]) << 8) |
                 static_cast<std::uint32_t>(padded[offset + 3]);
    }
    for (std::size_t i = 16; i < words.size(); ++i) {
      const auto s0 = std::rotr(words[i - 15], 7) ^
                      std::rotr(words[i - 15], 18) ^ (words[i - 15] >> 3);
      const auto s1 = std::rotr(words[i - 2], 17) ^
                      std::rotr(words[i - 2], 19) ^ (words[i - 2] >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    auto a = state[0];
    auto b = state[1];
    auto c = state[2];
    auto d = state[3];
    auto e = state[4];
    auto f = state[5];
    auto g = state[6];
    auto h = state[7];
    for (std::size_t i = 0; i < words.size(); ++i) {
      const auto sum1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
      const auto choose = (e & f) ^ (~e & g);
      const auto temp1 = h + sum1 + choose + constants[i] + words[i];
      const auto sum0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
      const auto majority = (a & b) ^ (a & c) ^ (b & c);
      const auto temp2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }

  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (const auto word : state)
    out << std::setw(8) << word;
  return out.str();
}

std::string sourceSignature(std::string_view source) {
  Lexer lexer(source);
  const auto tokens = lexer.lexAll();
  std::unordered_map<int, std::unordered_map<std::string, std::size_t>> names;
  std::unordered_map<int, std::size_t> nextName;
  std::string canonical{"sir-source-v3\n"};
  for (const auto &token : tokens) {
    if (token.kind == TokenKind::End)
      continue;

    std::string value = token.lexeme;
    switch (token.kind) {
    case TokenKind::GlobalId:
    case TokenKind::LocalId:
    case TokenKind::SymId:
    case TokenKind::BlockLabel: {
      const auto kind = static_cast<int>(token.kind);
      auto &scope = names[kind];
      auto [found, inserted] = scope.emplace(token.lexeme, nextName[kind]);
      if (inserted)
        ++nextName[kind];
      value = "id" + std::to_string(found->second);
      break;
    }
    default:
      break;
    }

    canonical += std::to_string(static_cast<unsigned int>(token.kind));
    canonical.push_back(':');
    canonical += std::to_string(value.size());
    canonical.push_back(':');
    canonical += value;
    canonical.push_back('\n');
  }
  return sha256(canonical);
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot read tensor case artifact: " +
                             path.string());
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

void writeFile(const std::filesystem::path &path, std::string_view bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    throw std::runtime_error("cannot create tensor case artifact: " +
                             path.string());
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!output)
    throw std::runtime_error("failed to write tensor case artifact: " +
                             path.string());
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
  bytes.append(reinterpret_cast<const char *>(input.bytes.data()),
               input.bytes.size());
  return location;
}

void writeBufferRecord(std::ostream &out, const TensorCaseBuffer &buffer,
                       std::uint64_t offset, std::uint64_t length,
                       std::string_view file, bool comma) {
  if (comma)
    out << ',';
  out << "{\"name\":" << jsonString(buffer.name)
      << ",\"dtype\":" << jsonString(elementName(buffer.element))
      << ",\"shape\":";
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

std::uint64_t storageElements(const Shape &shape) {
  std::uint64_t count = 1;
  for (const auto dimension : shape)
    count *= dimension;
  return count;
}

TritonBufferLayout makeTritonLayout(const TensorCaseBuffer &buffer,
                                    TritonBufferRole role) {
  TritonBufferLayout layout;
  layout.name = buffer.name;
  layout.role = role;
  layout.elementStrides = buffer.elementStrides.empty()
                              ? contiguousStrides(buffer.shape)
                              : buffer.elementStrides;
  layout.elementOffset = buffer.elementOffset;
  layout.axisOrder = buffer.axisOrder;
  if (layout.axisOrder.empty()) {
    layout.axisOrder.resize(buffer.shape.size());
    std::iota(layout.axisOrder.begin(), layout.axisOrder.end(), 0u);
    std::sort(layout.axisOrder.begin(), layout.axisOrder.end(),
              [&](auto lhs, auto rhs) {
                return layout.elementStrides[lhs] == layout.elementStrides[rhs]
                           ? lhs < rhs
                           : layout.elementStrides[lhs] <
                                 layout.elementStrides[rhs];
              });
  }
  if (buffer.storageSpan != 0) {
    layout.storageSpan = buffer.storageSpan;
  } else {
    std::uint64_t lastAddress = layout.elementOffset;
    for (std::size_t axis = 0; axis < buffer.shape.size(); ++axis)
      lastAddress += static_cast<std::uint64_t>(buffer.shape[axis] - 1) *
                     layout.elementStrides[axis];
    layout.storageSpan = lastAddress + 1;
  }
  return layout;
}

void writeV2BufferRecord(std::ostream &out, const TensorCaseBuffer &buffer,
                         std::uint64_t offset, std::uint64_t length,
                         std::string_view file, bool comma,
                         TensorProfileId profile = TensorProfileId::V2) {
  if (comma)
    out << ',';
  out << "{\"name\":" << jsonString(buffer.name)
      << ",\"dtype\":" << jsonString(elementName(buffer.element))
      << ",\"shape\":";
  writeShape(out, buffer.shape);
  out << ",\"file\":" << jsonString(file) << ",\"byte_offset\":" << offset
      << ",\"byte_length\":" << length << ",\"element_strides\":";
  const auto defaultStrides = contiguousStrides(buffer.shape);
  const auto &strides =
      buffer.elementStrides.empty() ? defaultStrides : buffer.elementStrides;
  writeShape(out, strides);
  std::vector<std::uint32_t> axisOrder = buffer.axisOrder;
  if (axisOrder.empty() && !buffer.shape.empty()) {
    axisOrder.resize(buffer.shape.size());
    std::iota(axisOrder.begin(), axisOrder.end(), 0u);
    std::sort(axisOrder.begin(), axisOrder.end(), [&](auto lhs, auto rhs) {
      return strides[lhs] == strides[rhs] ? lhs < rhs
                                          : strides[lhs] < strides[rhs];
    });
  }
  const auto span =
      buffer.storageSpan == 0
          ? static_cast<std::uint32_t>(storageElements(buffer.shape))
          : buffer.storageSpan;
  out << ",\"element_offset\":" << buffer.elementOffset
      << ",\"storage_span\":" << span << ",\"axis_order\":[";
  for (std::size_t i = 0; i < axisOrder.size(); ++i) {
    if (i)
      out << ',';
    out << axisOrder[i];
  }
  out << "],\"tile_shape\":";
  if (profile == TensorProfileId::V3 && !buffer.shape.empty())
    writeShape(out, tensorExecutionTileShape(buffer.shape, tensorProfile(profile)));
  else
    writeShape(out, buffer.shape);
  out << ",\"tile_origin\":[";
  for (std::size_t i = 0; i < buffer.shape.size(); ++i) {
    if (i)
      out << ',';
    out << '0';
  }
  out << "]}";
}

std::string wordHex(std::uint32_t word) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result(8, '0');
  for (std::size_t index = 0; index < result.size(); ++index) {
    const auto shift = static_cast<unsigned int>((result.size() - index - 1) * 4);
    result[index] = digits[(word >> shift) & 0xf];
  }
  return result;
}

void writeV2Execution(std::ostream &out, const TensorCase &testCase,
                      bool v3 = false) {
  std::uint32_t i32OutputCanary = 0xaaaaaaaa;
  if (v3) {
    std::vector<std::uint32_t> expectedWords;
    for (std::size_t offset = 0; offset + 4 <= testCase.expected.size(); offset += 4) {
      expectedWords.push_back(
          static_cast<std::uint32_t>(testCase.expected[offset]) |
          (static_cast<std::uint32_t>(testCase.expected[offset + 1]) << 8) |
          (static_cast<std::uint32_t>(testCase.expected[offset + 2]) << 16) |
          (static_cast<std::uint32_t>(testCase.expected[offset + 3]) << 24));
    }
    while (std::find(expectedWords.begin(), expectedWords.end(), i32OutputCanary) !=
           expectedWords.end())
      ++i32OutputCanary;
  }
  out << "\"execution\":{\"full_shapes\":[";
  for (std::size_t i = 0; i < testCase.fullShapes.size(); ++i) {
    if (i)
      out << ',';
    writeShape(out, testCase.fullShapes[i]);
  }
  out << "],\"range_extents\":[";
  for (std::size_t i = 0; i < testCase.rangeExtents.size(); ++i) {
    if (i)
      out << ',';
    writeShape(out, testCase.rangeExtents[i]);
  }
  out << "],\"warp_probe\":{\"status\":\"not_probed\",\"num_warps\":"
      << testCase.numWarps << "},"
      << "\"loop_bounds\":[";
  for (std::size_t i = 0; i < testCase.loopBounds.size(); ++i) {
    if (i)
      out << ',';
    out << testCase.loopBounds[i];
  }
  out << "],\"total_body_executions\":" << testCase.totalBodyExecutions
      << ",\"gap_sentinel_by_dtype\":{\"i32\":\"55555555\",\"f32\":"
         "\"3f000001\"},"
      << "\"output_canary_by_dtype\":{\"i32\":"
      << jsonString(wordHex(i32OutputCanary)) << ",\"f32\":"
      << jsonString(v3 ? "7fc00002" : "3f000002") << "}}";
}

std::string buildManifest(const TensorCase &testCase,
                          const std::string &program, const std::string &inputs,
                          const std::string &expected,
                          const std::string &kernel) {
  std::ostringstream out;
  out << "{\"schema\":\"refractir.tensor-case.v1\","
      << "\"profile\":\"tensor-v1\",\"entry\":\"kernel\",\"inputs\":[";
  std::string inputFile;
  std::vector<InputLocation> locations;
  inputFile.reserve(inputs.size());
  std::uint64_t offset = 0;
  for (const auto &input : testCase.inputs) {
    locations.push_back({offset, input.bytes.size()});
    offset += input.bytes.size();
  }
  for (std::size_t i = 0; i < testCase.inputs.size(); ++i)
    writeBufferRecord(out, testCase.inputs[i], locations[i].offset,
                      locations[i].length, "inputs.bin", i != 0);

  TensorCaseBuffer output{
      "return", testCase.outputElement, testCase.outputShape, {}};
  out << "],\"output\":";
  writeBufferRecord(out, output, 0, expected.size(), "expected.bin", false);
  out << ",\"grid\":[1,1,1],\"softmax_lowering\":\"direct\","
      << "\"launch\":{\"num_warps\":4,\"num_stages\":1,"
      << "\"enable_fp_fusion\":false},\"property\":{\"id\":"
      << jsonString(testCase.propertyId)
      << ",\"source\":" << jsonString(testCase.propertySource);
  if (!testCase.propertyThresholdF32Bits.empty())
    out << ",\"threshold_f32_bits\":"
        << jsonString(testCase.propertyThresholdF32Bits);
  out << "},\"evidence\":{\"kind\":" << jsonString(testCase.evidenceKind)
      << ",\"status\":" << jsonString(testCase.evidenceStatus)
      << ",\"artifacts\":[\"program.sir\",\"inputs.bin\",\"expected.bin\"";
  if (testCase.target == "triton")
    out << ",\"kernel.py\"";
  out << "]}"
      << ",\"comparison\":{\"mode\":"
      << jsonString(testCase.numericComparison ? "numeric" : "exact")
      << ",\"atol\":";
  if (testCase.numericComparison)
    out << "[2,100000]";
  else
    out << "[0,1]";
  out << ",\"rtol\":";
  if (testCase.numericComparison)
    out << "[2,1000]";
  else
    out << "[0,1]";
  out << "},\"reference\":{\"stable\":"
      << (testCase.referenceStable ? "true" : "false") << ",\"precisions\":[";
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
      << ",\"synthesis\":" << jsonString(testCase.synthesis)
      << ",\"node_limit\":8},\"files\":{"
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

std::string buildManifestV2(const TensorCase &testCase,
                            const std::string &program,
                            const std::string &inputs,
                            const std::string &expected,
                            const std::string &kernel,
                            TensorProfileId profile = TensorProfileId::V2) {
  const bool v3 = profile == TensorProfileId::V3;
  const bool v4 = profile == TensorProfileId::V4;
  const std::string profileName = v4 ? "tensor-v4" : v3 ? "tensor-v3" : "tensor-v2";
  const std::string schemaName = v4   ? "refractir.tensor-case.v4"
                                 : v3 ? "refractir.tensor-case.v3"
                                      : "refractir.tensor-case.v2";
  const std::string generatorName = v4   ? "tensor-v4-construction"
                                    : v3 ? "tensor-v3-construction"
                                         : "tensor-v2-construction";
  std::ostringstream out;
  out << "{\"schema\":" << jsonString(schemaName) << ","
      << "\"profile\":" << jsonString(profileName)
      << ",\"entry\":\"kernel\",\"source\":{"
      << "\"file\":\"program.sir\",\"sha256\":" << jsonString(sha256(program))
      << ",\"generator\":" << jsonString(generatorName) << ",\"seed\":" << testCase.seed
      << "},"
      << "\"inputs\":[";
  std::vector<InputLocation> locations;
  std::uint64_t offset = 0;
  for (const auto &input : testCase.inputs) {
    locations.push_back({offset, input.bytes.size()});
    offset += input.bytes.size();
  }
  for (std::size_t i = 0; i < testCase.inputs.size(); ++i)
    writeV2BufferRecord(out, testCase.inputs[i], locations[i].offset,
                        locations[i].length, "inputs.bin", i != 0, profile);

  TensorCaseBuffer output{
      "return", testCase.outputElement, testCase.outputShape, {},
      testCase.outputElementStrides, testCase.outputElementOffset,
      testCase.outputStorageSpan, testCase.outputAxisOrder};
  out << "],\"output\":";
  writeV2BufferRecord(out, output, 0, expected.size(), "expected.bin", false, profile);
  out << ",\"grid\":[1,1,1],\"softmax_lowering\":\"direct\","
      << "\"launch\":{\"num_warps\":" << testCase.numWarps
      << ",\"num_stages\":1,"
      << "\"enable_fp_fusion\":false},\"property\":{\"id\":"
      << jsonString(testCase.propertyId)
      << ",\"source\":" << jsonString(testCase.propertySource)
      << ",\"P\":\"defined-inputs\","
      << "\"D\":\"untargeted\",\"validated\":true},"
      << "\"evidence\":{\"kind\":" << jsonString(testCase.evidenceKind)
      << ",\"status\":" << jsonString(testCase.evidenceStatus)
      << ",\"artifacts\":[\"program.sir\",\"inputs.bin\",\"expected.bin\"";
  if (testCase.target == "triton")
    out << ",\"kernel.py\"";
  out << "]},\"comparison\":{\"mode\":"
      << jsonString(testCase.numericComparison ? "numeric" : "exact")
      << ",\"atol\":";
  if (testCase.numericComparison)
    out << "[2,100000]";
  else
    out << "[0,1]";
  out << ",\"rtol\":";
  if (testCase.numericComparison)
    out << "[2,1000]";
  else
    out << "[0,1]";
  if (v4)
    out << ",\"zero_sign_exact\":true";
  out << "},\"reference\":{\"defined\":true,\"stable\":true,\"precisions\":[";
  if (testCase.numericComparison)
    out << "128,256";
  out << ']';
  if (!testCase.exactnessKind.empty()) {
    out << ",\"exactness\":{\"kind\":" << jsonString(testCase.exactnessKind)
        << ",\"complete_output\":true,\"signed_zero\":"
        << jsonString(testCase.exactnessKind == "integer-valued-f32-reduction"
                          ? "positive-zero-only"
                          : "preserve-bits");
    if (testCase.exactnessKind == "integer-valued-f32-reduction")
      out << ",\"leaf_values\":\"finite-nonzero-integer-valued-f32\","
          << "\"max_abs_leaf_sum\":" << testCase.exactnessMaxAbsLeafSum
          << ",\"integer_exactness_bound\":16777216,"
          << "\"before_operations\":[\"input\"],\"after_operations\":[]";
    else
      out << ",\"allowed_operations\":[\"transpose\"]";
    out << '}';
  }
  out << "},";
  writeV2Execution(out, testCase, v3);
  out << ",\"coverage\":{\"primary_stratum\":"
      << jsonString(testCase.primaryStratum)
      << ",\"source_signature\":" << jsonString(sourceSignature(program)) << ','
      << "\"loop_structure\":[";
  for (std::size_t i = 0; i < testCase.loopBounds.size(); ++i) {
    if (i)
      out << ',';
    out << "{\"bound\":" << testCase.loopBounds[i] << '}';
  }
  out << "],\"input_behavior\":{\"property_passed\":true,";
  if (v3 || v4)
    out << "\"pattern\":" << jsonString(testCase.inputPattern) << ',';
  if (v4)
    out << "\"input_mode\":" << jsonString(testCase.inputMode) << ',';
  out << "\"distinguishing_objective\":\"untargeted\","
      << "\"counterfactual_inputs\":[";
  for (std::size_t index = 0;
       index < testCase.inputCounterfactualWitnesses.size(); ++index) {
    if (index)
      out << ',';
    const auto &[name, replacement] =
        testCase.inputCounterfactualWitnesses[index];
    out << "{\"input\":" << jsonString(name)
        << ",\"replacement\":" << replacement
        << ",\"outputs_differ\":true}";
  }
  out << "]}},"
      << "\"environment\":null,\"operator_coverage\":[";
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
  out << "],\"generator\":{\"version\":" << jsonString(generatorName)
      << ",\"seed\":"
      << testCase.seed << ",\"synthesis\":" << jsonString(testCase.synthesis)
      << ",\"primary_stratum\":" << jsonString(testCase.primaryStratum)
      << ",\"node_limit\":12";
  if (v3)
    out << ",\"input_pattern\":" << jsonString(testCase.inputPattern);
  if (v4)
    out << ",\"input_mode\":" << jsonString(testCase.inputMode)
        << ",\"input_pattern\":" << jsonString(testCase.inputPattern)
        << ",\"input_seed\":" << testCase.inputSeed;
  out << "},\"files\":{";
  auto writeDigest = [&](std::string_view name, std::string_view data,
                         bool comma) {
    if (comma)
      out << ',';
    out << jsonString(name) << ":{\"bytes\":" << data.size()
        << ",\"sha256\":" << jsonString(sha256(data)) << '}';
  };
  writeDigest("program.sir", program, false);
  writeDigest("inputs.bin", inputs, true);
  writeDigest("expected.bin", expected, true);
  if (!kernel.empty())
    writeDigest("kernel.py", kernel, true);
  out << "}}";
  return out.str();
}

} // namespace

void writeTensorCase(const std::filesystem::path &directory,
                     const TensorCase &testCase) {
  if (std::filesystem::exists(directory))
    throw std::runtime_error("tensor case directory already exists: " +
                             directory.string());
  if (testCase.target != "sir" && testCase.target != "triton")
    throw std::invalid_argument("tensor case target must be sir or triton");
  std::filesystem::create_directories(directory.parent_path());
  auto staging = directory;
  staging += ".tmp." + std::to_string(stagingId.fetch_add(1));
  if (std::filesystem::exists(staging))
    throw std::runtime_error("tensor case staging path already exists: " +
                             staging.string());
  std::filesystem::create_directory(staging);

  try {
    writeFile(staging / "program.sir", testCase.source);
    std::string inputBytes;
    for (const auto &input : testCase.inputs)
      (void)appendInputs(inputBytes, input);
    const std::string expectedBytes(
        reinterpret_cast<const char *>(testCase.expected.data()),
        testCase.expected.size());
    writeFile(staging / "inputs.bin", inputBytes);
    writeFile(staging / "expected.bin", expectedBytes);

    std::string kernel;
    if (testCase.target == "triton") {
      Program program = parseSource(testCase.source);
      DiagBag diagnostics;
      CheckOptions checkOptions;
      checkOptions.tensorProfile = testCase.profileVersion;
      if (!checkProgram(program, diagnostics, checkOptions))
        throw std::runtime_error(
            "cannot emit Triton kernel from unchecked tensor source");
      EmitOptions emitOptions;
      emitOptions.entry = "kernel";
      emitOptions.noUbGuards = true;
      emitOptions.tensorProfile = testCase.profileVersion;
      for (const auto &input : testCase.inputs)
        if (!input.shape.empty())
          emitOptions.tritonBufferLayouts.push_back(
              makeTritonLayout(input, TritonBufferRole::Input));
      if (!testCase.outputShape.empty()) {
        const TensorCaseBuffer output{"return", testCase.outputElement,
                                      testCase.outputShape, {},
                                      testCase.outputElementStrides,
                                      testCase.outputElementOffset,
                                      testCase.outputStorageSpan,
                                      testCase.outputAxisOrder};
        emitOptions.tritonBufferLayouts.push_back(
            makeTritonLayout(output, TritonBufferRole::Output));
      }
      if (!emitTriton(program, staging / "kernel.py", emitOptions))
        throw std::runtime_error(
            "Triton kernel emission failed for generated tensor case");
      kernel = readFile(staging / "kernel.py");
    }

    const std::string manifest =
        testCase.profileVersion == TensorProfileId::V1
            ? buildManifest(testCase, testCase.source, inputBytes,
                            expectedBytes, kernel)
            : buildManifestV2(testCase, testCase.source, inputBytes,
                              expectedBytes, kernel, testCase.profileVersion);
    writeFile(staging / "case.json", manifest);
    std::filesystem::rename(staging, directory);
  } catch (...) {
    std::error_code ec;
    std::filesystem::remove_all(staging, ec);
    throw;
  }
}

} // namespace refractir::reify
