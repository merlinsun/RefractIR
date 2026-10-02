#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "reify/tensor_graph_gen.hpp"

namespace {
  using namespace refractir;
  using namespace refractir::reify;

  void require(bool condition, const std::string &message) {
    if (!condition)
      throw std::runtime_error(message);
  }

  TensorRegion makeRegion(std::uint32_t seed, std::uint32_t maxOperators = 12) {
    TensorGenerationOptions options;
    options.profileVersion = TensorProfileId::V2;
    options.dtype = "i32";
    options.seed = seed;
    TensorGraphLimits limits;
    limits.maxInputs = 2;
    limits.maxRank = 2;
    limits.maxElements = 64;
    limits.maxOperators = maxOperators;
    const TensorTypeInfo type{Shape{4, 4}, TensorElement::I32};
    return generateTensorRegion(
        options, limits, {{"%acc_x", type}, {"%acc_y", type}}, {"%next_x", "%next_y"}
    );
  }

  std::string joined(const std::vector<std::string> &lines) {
    std::string result;
    for (const auto &line: lines) {
      result += line;
      result += '\n';
    }
    return result;
  }

  void suppliedSignatureAndNamedResults() {
    const auto region = makeRegion(11);
    require(region.typeOf("%acc_x").shape == Shape({4, 4}), "first input shape changed");
    require(region.typeOf("%acc_y").shape == Shape({4, 4}), "second input shape changed");
    require(region.typeOf("%next_x").shape == Shape({4, 4}), "first root shape changed");
    require(region.typeOf("%next_y").shape == Shape({4, 4}), "second root shape changed");
    const auto instructions = region.renderInstructions();
    require(instructions.size() >= 3, "region must produce both roots from a shared DAG");
    require(
        instructions.back().find("%next_y =") != std::string::npos, "second named result is missing"
    );
  }

  void resultUnionKeepsBothInputDependencies() {
    const auto region = makeRegion(29);
    const auto body = region.renderBody({{"%acc_x", "%next_x"}, {"%acc_y", "%next_y"}});
    const auto text = joined(body);
    require(text.find("%acc_x") != std::string::npos, "first state input is dead");
    require(text.find("%acc_y") != std::string::npos, "second state input is dead");
    require(text.find("%next_x =") != std::string::npos, "first result dependency was pruned");
    require(text.find("%next_y =") != std::string::npos, "second result dependency was pruned");
    require(
        body.size() >= 5 && body[body.size() - 2] == "  %acc_x = %next_x;" &&
            body.back() == "  %acc_y = %next_y;",
        "all generated results must precede parallel loop-state writeback"
    );
  }

  void operatorBudgetIncludesEveryRoot() {
    const auto region = makeRegion(7, 3);
    require(
        region.renderInstructions().size() == 3,
        "the budget must include the shared producer and both output roots"
    );
  }

  void sameSeedReproducesRegion() {
    const auto first = makeRegion(41).renderInstructions();
    const auto second = makeRegion(41).renderInstructions();
    require(first == second, "equal seeds must reproduce node order and reduction axes");
  }

  void differentSeedsVaryTheRegion() {
    std::vector<std::string> seen;
    for (std::uint32_t seed = 1; seed <= 12; ++seed)
      seen.push_back(joined(makeRegion(seed).renderInstructions()));
    std::sort(seen.begin(), seen.end());
    seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
    require(seen.size() > 1, "different seeds must be able to change the generated DAG");
  }

  void mismatchedStateShapesAreRejected() {
    TensorGenerationOptions options;
    options.profileVersion = TensorProfileId::V2;
    options.dtype = "i32";
    TensorGraphLimits limits;
    limits.maxInputs = 2;
    limits.maxRank = 2;
    limits.maxElements = 64;
    limits.maxOperators = 12;
    try {
      (void) generateTensorRegion(
          options, limits,
          {{"%acc_x", TensorTypeInfo{Shape{4, 4}, TensorElement::I32}},
           {"%acc_y", TensorTypeInfo{Shape{2, 8}, TensorElement::I32}}},
          {"%next_x", "%next_y"}
      );
    } catch (const std::invalid_argument &) {
      return;
    }
    throw std::runtime_error("state tensors with incompatible shapes must be rejected");
  }
} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: tensor_graph_region_api <test-name>\n";
    return 2;
  }
  const std::string name(argv[1]);
  try {
    if (name == "supplied-signature")
      suppliedSignatureAndNamedResults();
    else if (name == "result-union")
      resultUnionKeepsBothInputDependencies();
    else if (name == "operator-budget")
      operatorBudgetIncludesEveryRoot();
    else if (name == "same-seed")
      sameSeedReproducesRegion();
    else if (name == "different-seeds")
      differentSeedsVaryTheRegion();
    else if (name == "mismatched-signature")
      mismatchedStateShapesAreRejected();
    else
      throw std::invalid_argument("unknown tensor graph region test: " + name);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
