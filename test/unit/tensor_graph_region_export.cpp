#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "reify/tensor_graph_gen.hpp"

namespace {
  using namespace refractir;
  using namespace refractir::reify;

  std::uint32_t readSeed(const char *argument) {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(argument, &consumed);
    if (argument[consumed] != '\0' || parsed > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("seed must be a uint32 decimal integer");
    return static_cast<std::uint32_t>(parsed);
  }

  void exportRegion(std::uint32_t seed) {
    TensorGenerationOptions options;
    options.profileVersion = TensorProfileId::V2;
    options.dtype = "f32";
    options.seed = seed;

    TensorGraphLimits limits;
    limits.maxInputs = 2;
    limits.maxRank = 2;
    limits.maxElements = 64;
    limits.maxOperators = 10;

    const TensorTypeInfo type{Shape{2, 2}, TensorElement::F32};
    const auto region = generateTensorRegion(
        options, limits, {{"%acc_x", type}, {"%acc_y", type}}, {"%next_x", "%next_y"}
    );
    const std::vector<std::pair<std::string, std::string>> updates{
        {"%acc_x", "%next_x"}, {"%acc_y", "%next_y"}};
    const std::vector<std::string> reserved{"%i", "%out"};

    std::cout << "fun @kernel(%x: tensor [2, 2] f32) : tensor [2, 2] f32 {\n"
                 "  let mut %i: i32 = undef;\n"
                 "  let mut %acc_x: tensor [2, 2] f32 = undef;\n"
                 "  let mut %acc_y: tensor [2, 2] f32 = {0.0, 0.0, 0.0, 0.0};\n"
                 "  let mut %out: tensor [2, 2] f32 = undef;\n";
    for (const auto &line: region.renderDeclarations(updates, reserved))
      std::cout << line << '\n';
    std::cout << "^entry:\n  %acc_x = %x;\n  %i = 0;\n  br ^header;\n"
                 "^header:\n  br %i < 2, ^body, ^exit;\n^body:\n";
    for (const auto &line: region.renderBody(updates, reserved))
      std::cout << line << '\n';
    std::cout << "  %i = %i + 1;\n  br ^header;\n^exit:\n"
                 "  %out = tensor add %acc_x, %acc_y;\n  ret %out;\n}\n";
  }
} // namespace

int main(int argc, char **argv) {
  if (argc != 3 || std::string(argv[1]) != "--seed") {
    std::cerr << "usage: tensor_graph_region_export --seed <uint32>\n";
    return 2;
  }
  try {
    exportRegion(readSeed(argv[2]));
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
