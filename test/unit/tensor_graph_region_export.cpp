#include <charconv>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "reify/tensor_graph_gen.hpp"

namespace {
  using namespace refractir;
  using namespace refractir::reify;

  struct ExportOptions {
    std::uint32_t seed = 0;
    Shape shape{2, 2};
    std::uint32_t iterations = 2;
    bool hasSeed = false;
  };

  std::uint32_t parseUnsigned(std::string_view text, std::string_view name) {
    std::uint32_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
      throw std::invalid_argument(std::string(name) + " must be a uint32 decimal integer");
    return value;
  }

  Shape parseShape(std::string_view text) {
    const auto comma = text.find(',');
    if (comma == std::string_view::npos || text.find(',', comma + 1) != std::string_view::npos)
      throw std::invalid_argument("shape must be two comma-separated dimensions");
    Shape shape{
        parseUnsigned(text.substr(0, comma), "shape dimension"),
        parseUnsigned(text.substr(comma + 1), "shape dimension")};
    if (shape[0] == 0 || shape[1] == 0 || shape[0] > 64 || shape[1] > 64 ||
        static_cast<std::uint64_t>(shape[0]) * shape[1] > 64)
      throw std::invalid_argument("shape must contain between one and 64 elements");
    return shape;
  }

  ExportOptions parseArguments(int argc, char **argv) {
    ExportOptions options;
    bool hasShape = false;
    bool hasIterations = false;
    for (int index = 1; index < argc; ++index) {
      const std::string_view name(argv[index]);
      if (index + 1 >= argc)
        throw std::invalid_argument("each option requires a value");
      const std::string_view value(argv[++index]);
      if (name == "--seed") {
        if (options.hasSeed)
          throw std::invalid_argument("--seed may be specified only once");
        options.seed = parseUnsigned(value, "seed");
        options.hasSeed = true;
      } else if (name == "--shape") {
        if (hasShape)
          throw std::invalid_argument("--shape may be specified only once");
        options.shape = parseShape(value);
        hasShape = true;
      } else if (name == "--iterations") {
        if (hasIterations)
          throw std::invalid_argument("--iterations may be specified only once");
        options.iterations = parseUnsigned(value, "iterations");
        if (options.iterations < 1 || options.iterations > 8)
          throw std::invalid_argument("iterations must be in [1,8]");
        hasIterations = true;
      } else {
        throw std::invalid_argument("unknown option: " + std::string(name));
      }
    }
    if (!options.hasSeed)
      throw std::invalid_argument("--seed is required");
    return options;
  }

  std::string shapeText(const Shape &shape) {
    std::string result;
    for (const auto dimension: shape) {
      if (!result.empty())
        result += ", ";
      result += std::to_string(dimension);
    }
    return result;
  }

  std::string zeroTensor(const Shape &shape) {
    std::string result = "{";
    for (std::size_t index = 0; index < shape[0] * shape[1]; ++index) {
      if (index != 0)
        result += ", ";
      result += "0.0";
    }
    result += "}";
    return result;
  }

  void exportRegion(const ExportOptions &configuration) {
    TensorGenerationOptions options;
    options.profileVersion = TensorProfileId::V2;
    options.dtype = "f32";
    options.seed = configuration.seed;

    TensorGraphLimits limits;
    limits.maxInputs = 2;
    limits.maxRank = 2;
    limits.maxElements = 64;
    limits.maxOperators = 10;

    const TensorTypeInfo type{configuration.shape, TensorElement::F32};
    const auto region = generateTensorRegion(
        options, limits, {{"%acc_x", type}, {"%acc_y", type}}, {"%next_x", "%next_y"}
    );
    const std::vector<std::pair<std::string, std::string>> updates{
        {"%acc_x", "%next_x"}, {"%acc_y", "%next_y"}};
    const std::vector<std::string> reserved{"%i", "%out"};
    const auto tensorType = "tensor [" + shapeText(configuration.shape) + "] f32";

    std::cout << "fun @kernel(%x: " << tensorType << ") : " << tensorType << " {\n"
              << "  let mut %i: i32 = undef;\n"
              << "  let mut %acc_x: " << tensorType << " = undef;\n"
              << "  let mut %acc_y: " << tensorType << " = " << zeroTensor(configuration.shape)
              << ";\n"
              << "  let mut %out: " << tensorType << " = undef;\n";
    for (const auto &line: region.renderDeclarations(updates, reserved))
      std::cout << line << '\n';
    std::cout << "^entry:\n  %acc_x = %x;\n  %i = 0;\n  br ^header;\n"
              << "^header:\n  br %i < " << configuration.iterations << ", ^body, ^exit;\n"
              << "^body:\n";
    for (const auto &line: region.renderBody(updates, reserved))
      std::cout << line << '\n';
    std::cout << "  %i = %i + 1;\n  br ^header;\n^exit:\n"
              << "  %out = tensor add %acc_x, %acc_y;\n  ret %out;\n}\n";
  }
} // namespace

int main(int argc, char **argv) {
  try {
    exportRegion(parseArguments(argc, argv));
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'
              << "usage: tensor_graph_region_export --seed <uint32> "
                 "[--shape <rows,columns>] [--iterations <1..8>]\n";
    return 2;
  }
}
