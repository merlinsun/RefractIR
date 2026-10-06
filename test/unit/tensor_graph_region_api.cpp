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

  TensorRegion makeRegion(
      std::uint32_t seed, std::uint32_t maxOperators = 12,
      TensorRegionArithmeticMode arithmetic = TensorRegionArithmeticMode::Additive
  ) {
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
        options, limits, {{"%acc_x", type}, {"%acc_y", type}}, {"%next_x", "%next_y"}, arithmetic
    );
  }

  TensorRegion makeNamedRegion(
      std::uint32_t seed, std::uint32_t maxOperators, std::vector<std::string> results,
      Shape shape = {4, 4}
  ) {
    TensorGenerationOptions options;
    options.profileVersion = TensorProfileId::V2;
    options.dtype = "i32";
    options.seed = seed;
    TensorGraphLimits limits;
    limits.maxInputs = 2;
    limits.maxRank = 2;
    limits.maxElements = 64;
    limits.maxOperators = maxOperators;
    const TensorTypeInfo type{shape, TensorElement::I32};
    return generateTensorRegion(
        options, limits, {{"%acc_x", type}, {"%acc_y", type}}, std::move(results)
    );
  }

  void requireUniqueInstructionResults(const TensorRegion &region) {
    std::vector<std::string> names;
    for (const auto &instruction: region.renderInstructions()) {
      const auto equals = instruction.find(" =");
      require(equals != std::string::npos, "rendered region instruction has no result");
      names.push_back(instruction.substr(2, equals - 2));
    }
    std::sort(names.begin(), names.end());
    require(
        std::adjacent_find(names.begin(), names.end()) == names.end(),
        "generated temporary and requested result names must stay unique"
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

  void resultNameT0IsReservedBeforeTheSharedProducer() {
    const auto region = makeNamedRegion(3, 3, {"%t0", "%next_y"});
    requireUniqueInstructionResults(region);
  }

  void resultNameT0IsReservedWithFoldNodes() {
    const auto region = makeNamedRegion(7, 10, {"%t0", "%next_y"});
    requireUniqueInstructionResults(region);
  }

  void adjacentTemporaryNamesAreReserved() {
    const auto region = makeNamedRegion(11, 3, {"%t0", "%t1"});
    requireUniqueInstructionResults(region);
  }

  void firstResultNameStaysUniqueWithThreeRoots() {
    const auto region = makeNamedRegion(19, 4, {"%t0", "%first", "%second"});
    requireUniqueInstructionResults(region);
  }

  void reservedNamesDoNotDependOnSeed() {
    const auto region = makeNamedRegion(29, 8, {"%t0", "%other"});
    requireUniqueInstructionResults(region);
  }

  void subtractionArithmeticModeEmitsComputedLeftOperand() {
    const auto region = makeRegion(13, 10, TensorRegionArithmeticMode::Subtraction);
    const auto instructions = region.renderInstructions();
    const auto subtraction =
        std::find_if(instructions.begin(), instructions.end(), [](const auto &line) {
          return line.find("tensor sub") != std::string::npos;
        });
    require(subtraction != instructions.end(), "subtraction mode must emit tensor sub");
    require(
        subtraction->find(", %acc_x;") != std::string::npos,
        "subtraction must subtract the first state input from the shared graph value"
    );
  }

  void multiplicationArithmeticModeEmitsTensorMultiply() {
    const auto region = makeRegion(17, 10, TensorRegionArithmeticMode::Multiplication);
    const auto instructions = region.renderInstructions();
    require(
        std::any_of(
            instructions.begin(), instructions.end(),
            [](const auto &line) { return line.find("tensor mul") != std::string::npos; }
        ),
        "multiplication mode must emit tensor mul"
    );
  }

  void mixedArithmeticModeKeepsBothResultsAndDependencies() {
    const auto region = makeRegion(29, 10, TensorRegionArithmeticMode::Mixed);
    const auto instructions = region.renderInstructions();
    const auto text = joined(instructions);
    require(text.find("tensor mul") != std::string::npos, "mixed mode must emit tensor mul");
    require(text.find("tensor sub") != std::string::npos, "mixed mode must emit tensor sub");
    const auto subtraction =
        std::find_if(instructions.begin(), instructions.end(), [](const auto &line) {
          return line.find("tensor sub") != std::string::npos;
        });
    require(
        subtraction->find(", %acc_x;") != std::string::npos,
        "mixed subtraction must consume the nonzero first state input"
    );
    const auto body = joined(region.renderBody({{"%acc_x", "%next_x"}, {"%acc_y", "%next_y"}}));
    require(body.find("%next_x =") != std::string::npos, "first named root was lost");
    require(body.find("%next_y =") != std::string::npos, "second named root was lost");
    require(
        body.rfind("  %acc_x = %next_x;") < body.rfind("  %acc_y = %next_y;"),
        "both generated roots must be computed before ordered state writeback"
    );
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
    else if (name == "result-name-t0-minimal")
      resultNameT0IsReservedBeforeTheSharedProducer();
    else if (name == "result-name-t0-folds")
      resultNameT0IsReservedWithFoldNodes();
    else if (name == "adjacent-temporary-names")
      adjacentTemporaryNamesAreReserved();
    else if (name == "result-name-t0-three-roots")
      firstResultNameStaysUniqueWithThreeRoots();
    else if (name == "result-name-t0-seed-variation")
      reservedNamesDoNotDependOnSeed();
    else if (name == "region-subtraction")
      subtractionArithmeticModeEmitsComputedLeftOperand();
    else if (name == "region-multiplication")
      multiplicationArithmeticModeEmitsTensorMultiply();
    else if (name == "region-mixed-arithmetic")
      mixedArithmeticModeKeepsBothResultsAndDependencies();
    else
      throw std::invalid_argument("unknown tensor graph region test: " + name);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
