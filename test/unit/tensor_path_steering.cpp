// Print a kernel's steering targets as JSON for test/lib/tensor_path_steering.py.
//
//   tensor_path_steering <kernel.sir> [--entry kernel] [--profile v5]
//
// The kernel is parsed and checked under the profile first, so a target is
// only ever derived from a program the profile accepts.

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "analysis/tensor.hpp"
#include "frontend/diagnostics.hpp"
#include "frontend/pipeline.hpp"
#include "reify/tensor_path_steering.hpp"

namespace {

  std::string jsonString(const std::string &text) {
    std::string out = "\"";
    for (const char c: text) {
      if (c == '"' || c == '\\')
        out += std::string("\\") + c;
      else if (c == '\n')
        out += "\\n";
      else
        out += c;
    }
    return out + "\"";
  }

} // namespace

int main(int argc, char **argv) {
  using namespace refractir;
  if (argc < 2) {
    std::cerr << "usage: tensor_path_steering <kernel.sir> [--entry name] [--profile v5]\n";
    return 2;
  }
  std::string entry = "kernel";
  std::string version = "v5";
  for (int index = 2; index + 1 < argc; index += 2) {
    const std::string flag = argv[index];
    if (flag == "--entry")
      entry = argv[index + 1];
    else if (flag == "--profile")
      version = argv[index + 1];
    else {
      std::cerr << "unknown option " << flag << "\n";
      return 2;
    }
  }
  try {
    std::ifstream file(argv[1]);
    std::stringstream text;
    text << file.rdbuf();
    const auto source = text.str();
    auto program = parseSource(source);
    DiagBag diagnostics;
    CheckOptions options;
    options.tensorProfile = parseTensorProfileVersion(version);
    if (!checkProgram(program, diagnostics, options) ||
        !checkTensorCapabilities(
            program, TensorConsumer::Reference, tensorProfile(options.tensorProfile), diagnostics
        )) {
      std::cerr << (diagnostics.diags.empty() ? "check failed" : diagnostics.diags.front().message)
                << "\n";
      return 1;
    }
    const auto targets =
        reify::steeringTargets(program, entry, tensorProfile(options.tensorProfile));
    std::cout << "{\"targets\": [";
    for (std::size_t index = 0; index < targets.size(); ++index) {
      const auto &target = targets[index];
      std::cout << (index ? ",\n" : "\n") << "{\"name\": " << jsonString(target.name)
                << ", \"path\": [";
      for (std::size_t step = 0; step < target.path.size(); ++step)
        std::cout << (step ? ", " : "") << jsonString(target.path[step]);
      std::cout << "], \"skeleton\": " << jsonString(target.skeleton) << "}";
    }
    std::cout << "\n]}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
