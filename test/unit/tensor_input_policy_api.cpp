#include <charconv>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>

#include "ast/sir_printer.hpp"
#include "frontend/pipeline.hpp"
#include "reify/tensor_graph_gen.hpp"
#include "reify/tensor_input_policy.hpp"

namespace {
  using namespace refractir;
  using namespace refractir::reify;

  std::string elementName(TensorElement element) {
    switch (element) {
      case TensorElement::I32:
        return "i32";
      case TensorElement::F32:
        return "f32";
    }
    throw std::runtime_error("unknown tensor element type");
  }

  std::string hexBytes(const std::vector<std::uint8_t> &bytes) {
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (const auto byte: bytes)
      result << std::setw(2) << static_cast<unsigned>(byte);
    return result.str();
  }

  std::string readSource(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
      throw std::runtime_error("cannot open source file: " + path);
    std::ostringstream source;
    source << input.rdbuf();
    return source.str();
  }

  std::uint32_t parseSeed(const char *text) {
    std::uint32_t seed = 0;
    const std::string value(text);
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), seed);
    if (error != std::errc{} || end != value.data() + value.size())
      throw std::invalid_argument("seed must be an unsigned 32-bit integer");
    return seed;
  }

  void printProposal(const TensorInputProposal &proposal) {
    std::cout << "{\"outcome\":\"" << proposal.outcome << "\",\"inputs\":[";
    for (std::size_t index = 0; index < proposal.inputs.size(); ++index) {
      if (index != 0)
        std::cout << ',';
      const auto &buffer = proposal.inputs[index];
      std::cout << "{\"name\":\"" << buffer.name << "\",\"element\":\""
                << elementName(buffer.element) << "\",\"shape\":[";
      for (std::size_t axis = 0; axis < buffer.shape.size(); ++axis) {
        if (axis != 0)
          std::cout << ',';
        std::cout << buffer.shape[axis];
      }
      std::cout << "],\"bytes\":\"" << hexBytes(buffer.bytes) << "\"}";
    }
    std::cout << "],\"sampler_version\":\"" << proposal.samplerVersion << "\"}\n";
  }
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--generate") {
      TensorGenerationOptions options;
      options.profileVersion = TensorProfileId::V2;
      options.generator = "compositional";
      options.synthesis = "construction";
      options.dtype = "f32";
      options.seed = parseSeed(argv[2]);

      const Program generated = generateTensorProgram(options, TensorGraphLimits{});
      std::ostringstream text;
      SIRPrinter printer(text, {}, {}, {});
      printer.print(generated);
      const auto source = text.str();
      Program checked = parseSource(source);
      DiagBag diagnostics;
      CheckOptions check;
      check.functionAnalyses = false;
      check.tensorProfile = TensorProfileId::V2;
      if (!checkProgram(checked, diagnostics, check)) {
        for (const auto &diagnostic: diagnostics.diags)
          std::cerr << diagnostic.message << '\n';
        return 2;
      }
      std::cout << source;
      return 0;
    }
    if (argc != 3)
      throw std::invalid_argument(
          "usage: tensor_input_policy_api <program.sir> <seed> | --generate <seed>"
      );

    Program program = parseSource(readSource(argv[1]));
    DiagBag diagnostics;
    CheckOptions check;
    check.functionAnalyses = false;
    check.tensorProfile = TensorProfileId::V2;
    if (!checkProgram(program, diagnostics, check)) {
      for (const auto &diagnostic: diagnostics.diags)
        std::cerr << diagnostic.message << '\n';
      return 2;
    }

    TensorInputPolicyOptions options;
    options.policy = TensorInputPolicy::Boundary;
    options.profileVersion = TensorProfileId::V2;
    options.seed = parseSeed(argv[2]);
    const auto proposal = proposeTensorInputs(program, SymbolicExecutor::SolverFactory{}, options);
    printProposal(proposal);
    return proposal.outcome == "accepted" ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
