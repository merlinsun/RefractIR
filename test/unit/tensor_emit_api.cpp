#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "backend/emit.hpp"
#include "frontend/pipeline.hpp"
#include "reify/tensor_case.hpp"

int main(int argc, char **argv) {
  using namespace refractir;
  if (argc != 4) {
    std::cerr << "usage: tensor_emit_api <target> <source.sir> <output>\n";
    return 2;
  }

  std::ifstream input(argv[2]);
  if (!input) {
    std::cerr << "cannot open input file\n";
    return 2;
  }
  std::stringstream source;
  source << input.rdbuf();
  const std::string target(argv[1]);
  const std::filesystem::path output(argv[3]);
  if (target == "signature") {
    reify::TensorCase testCase;
    testCase.profileVersion = TensorProfileId::V2;
    testCase.source = source.str();
    reify::writeTensorCase(output, testCase);
    return 0;
  }

  Program program = parseSource(source.str());
  EmitOptions options;
  options.verbose = true;
  bool emitted = false;

  if (target == "c") {
    emitted = emitC(program, output.parent_path(), output.stem().string(), options);
  } else if (target == "wasm") {
    emitted = emitWasm(program, output, options);
  } else if (target == "python") {
    emitted = emitPy(program, output, options);
  } else if (target == "triton") {
    emitted = emitSirFile(argv[2], target, output, options);
  } else {
    std::cerr << "unknown target\n";
    return 2;
  }
  return emitted ? 0 : 1;
}
