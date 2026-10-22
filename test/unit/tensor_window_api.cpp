// Test-only entry point for the tensor window's parser, printer, and dumper.
//
// The shared CLI has no general SIR-print mode, so the contract tests in
// test_tensor_window.py drive the same frontend entry points through this
// binary instead of duplicating them.

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

#include "ast/ast_dumper.hpp"
#include "ast/sir_printer.hpp"
#include "frontend/pipeline.hpp"

int main(int argc, char **argv) {
  using namespace refractir;
  if (argc != 3) {
    std::cerr << "usage: tensor_window_api <parse|print|dump> <source.sir>\n";
    return 2;
  }
  const std::string mode = argv[1];
  std::ifstream input(argv[2]);
  if (!input) {
    std::cerr << "cannot open " << argv[2] << "\n";
    return 2;
  }
  std::string source((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  try {
    const Program program = parseSource(source);
    if (mode == "parse")
      return 0;
    if (mode == "print") {
      SIRPrinter printer(std::cout, {}, {}, {});
      printer.print(program);
      return 0;
    }
    if (mode == "dump") {
      ASTDumper dumper(std::cout);
      dumper.dump(program);
      return 0;
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
  std::cerr << "unknown mode: " << mode << "\n";
  return 2;
}
