#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "reify/tensor_region.hpp"

namespace {
  using namespace refractir;
  using namespace refractir::reify;

  TensorTypeInfo type(Shape shape, TensorElement element = TensorElement::I32) {
    return TensorTypeInfo{std::move(shape), element};
  }

  void require(bool condition, const std::string &message) {
    if (!condition)
      throw std::runtime_error(message);
  }

  template<typename Action>
  void requireInvalid(Action action) {
    try {
      action();
    } catch (const std::invalid_argument &) {
      return;
    }
    throw std::runtime_error("expected an invalid tensor region to be rejected");
  }

  void multiResults() {
    TensorRegion region({{"%a", type({2, 2})}, {"%b", type({2, 2})}});
    region.add("%v", TensorOp::Add, {"%a", "%b"});
    region.add("%rows", TensorOp::Sum, {"%v"}, -1, true);
    region.add("%columns", TensorOp::Sum, {"%v"}, 0, true);
    region.add("%row_broadcast", TensorOp::Broadcast, {"%rows"}, 0, false, {2, 2});
    region.add("%column_broadcast", TensorOp::Broadcast, {"%columns"}, 0, false, {2, 2});
    region.add("%a_next", TensorOp::Add, {"%a", "%row_broadcast"});
    region.add("%b_next", TensorOp::Add, {"%b", "%column_broadcast"});
    region.setResults({"%a_next", "%b_next"});

    const auto instructions = region.renderInstructions();
    require(instructions.size() == 7, "both result dependencies must be retained");
    require(
        instructions[1].find("tensor sum %v, 1, keep") != std::string::npos,
        "negative reduction axis must be canonicalized"
    );
    require(region.typeOf("%a_next").shape == Shape({2, 2}), "first result shape is wrong");
    require(region.typeOf("%b_next").shape == Shape({2, 2}), "second result shape is wrong");
  }

  void declarationsAndDeadNodes() {
    TensorRegion region(
        {{"%a", type({2, 2}, TensorElement::F32)}, {"%b", type({2, 2}, TensorElement::F32)}}
    );
    region.add("%dead", TensorOp::Add, {"%a", "%b"});
    region.add("%live", TensorOp::Add, {"%a", "%b"});
    region.setResults({"%live"});
    const auto declarations = region.renderDeclarations();
    const auto instructions = region.renderInstructions();
    require(
        declarations == std::vector<std::string>{"  let mut %live: tensor [2, 2] f32 = undef;"},
        "only live region values should be declared"
    );
    require(
        instructions.size() == 1 && instructions.front().find("%live =") != std::string::npos,
        "dead graph nodes must not be emitted"
    );
  }

  void simultaneousStateUpdates() {
    TensorRegion region({{"%left", type({2})}, {"%right", type({2})}});
    region.add("%next_left", TensorOp::Add, {"%left", "%right"});
    region.add("%next_right", TensorOp::Add, {"%right", "%left"});
    region.setResults({"%next_left", "%next_right"});
    const auto updates =
        region.renderStateUpdates({{"%left", "%next_left"}, {"%right", "%next_right"}});
    require(
        updates == std::vector<std::string>{"  %left = %next_left;", "  %right = %next_right;"},
        "all body results must be computed before either state assignment"
    );
    const auto body = region.renderBody({{"%left", "%next_left"}, {"%right", "%next_right"}});
    require(
        body.size() == 4 && body[0].find("%next_left =") != std::string::npos &&
            body[1].find("%next_right =") != std::string::npos && body[2] == updates[0] &&
            body[3] == updates[1],
        "rendered loop bodies must compute every output before updating state"
    );
    requireInvalid([&] {
      (void) region.renderStateUpdates({{"%left", "%next_left"}, {"%left", "%next_right"}});
    });
  }

  void multipleOutputsKeepSharedDependencies() {
    TensorRegion region({{"%x", type({2, 2})}, {"%y", type({2, 2})}});
    region.add("%shared", TensorOp::Add, {"%x", "%y"});
    region.add("%row", TensorOp::Sum, {"%shared"}, 1, true);
    region.add("%column", TensorOp::Sum, {"%shared"}, 0, true);
    region.setResults({"%row", "%column"});
    const auto instructions = region.renderInstructions();
    require(instructions.size() == 3, "shared producer and both roots must be emitted once");
    require(
        instructions[0].find("%shared =") != std::string::npos,
        "shared producer must precede both result roots"
    );
  }

  void invalidGraphsAreRejected() {
    TensorRegion region({{"%a", type({2, 2})}, {"%b", type({3, 2})}});
    requireInvalid([&] { region.add("%bad", TensorOp::Add, {"%a", "%b"}); });
    requireInvalid([&] { region.add("%bad_axis", TensorOp::Sum, {"%a"}, 2, false); });
    requireInvalid([&] { region.add("%missing", TensorOp::Sum, {"%unknown"}, 0, false); });
    requireInvalid([&] { region.add("%wrong_arity", TensorOp::Add, {"%a"}); });
  }

  void singletonDimensionsBroadcastAcrossAxes() {
    TensorRegion region({{"%x", type({1, 2})}});
    region.add("%row", TensorOp::Sum, {"%x"}, 0, true);
    region.add("%expanded", TensorOp::Broadcast, {"%row"}, 0, false, {3, 2});
    region.setResults({"%expanded"});
    require(
        region.typeOf("%expanded").shape == Shape({3, 2}),
        "singleton axes must broadcast to the requested extent"
    );
  }

  void scalarReductionCanBroadcastBackToTensor() {
    TensorRegion region({{"%x", type({2})}});
    region.add("%total", TensorOp::Sum, {"%x"}, 0, false);
    region.add("%splat", TensorOp::Broadcast, {"%total"}, 0, false, {2});
    region.setResults({"%splat"});
    require(region.typeOf("%total").shape.empty(), "dropped rank-one sum must be scalar");
    require(
        region.renderDeclarations().front() == "  let mut %total: i32 = undef;",
        "scalar intermediate must use scalar SIR declaration syntax"
    );
  }

  void inputRootsCanBeSwapped() {
    TensorRegion region({{"%a", type({2})}, {"%b", type({2})}});
    region.setResults({"%b", "%a"});
    const auto updates = region.renderStateUpdates({{"%a", "%b"}, {"%b", "%a"}});
    require(
        updates ==
            std::vector<std::string>{
                "  let mut %__region_saved_0: tensor [2] i32 = %b;",
                "  let mut %__region_saved_1: tensor [2] i32 = %a;", "  %a = %__region_saved_0;",
                "  %b = %__region_saved_1;"},
        "swapped input results must be snapshotted before either destination changes"
    );
  }

  void inputRootsCanFormLongCycles() {
    TensorRegion region({{"%a", type({2})}, {"%b", type({2})}, {"%c", type({2})}, {"%d", type({2})}}
    );
    region.setResults({"%b", "%c", "%d", "%a"});
    const auto updates =
        region.renderStateUpdates({{"%a", "%b"}, {"%b", "%c"}, {"%c", "%d"}, {"%d", "%a"}});
    require(
        updates.size() == 8 && updates[4] == "  %a = %__region_saved_0;" &&
            updates[7] == "  %d = %__region_saved_3;",
        "all sources in a longer input-root cycle must be read from old state"
    );
  }

  void identityRootCanShareAParallelUpdate() {
    TensorRegion region({{"%a", type({2})}, {"%b", type({2})}, {"%c", type({2})}});
    region.setResults({"%b", "%a", "%c"});
    const auto updates = region.renderStateUpdates({{"%a", "%b"}, {"%b", "%a"}, {"%c", "%c"}});
    require(
        updates ==
            std::vector<std::string>{
                "  let mut %__region_saved_0: tensor [2] i32 = %b;",
                "  let mut %__region_saved_1: tensor [2] i32 = %a;", "  %a = %__region_saved_0;",
                "  %b = %__region_saved_1;"},
        "an identity result must remain unchanged alongside a concurrent swap"
    );
  }

  void passThroughSourceCanSurviveComputedWriteback() {
    TensorRegion region({{"%a", type({2})}, {"%b", type({2})}});
    region.add("%sum", TensorOp::Add, {"%a", "%b"});
    region.setResults({"%sum", "%a"});
    const auto body = region.renderBody({{"%a", "%sum"}, {"%b", "%a"}});
    require(
        body.size() == 4 && body[1] == "  let mut %__region_saved_0: tensor [2] i32 = %a;" &&
            body[2] == "  %a = %sum;" && body[3] == "  %b = %__region_saved_0;",
        "a pass-through result must survive a computed update to its input"
    );
  }

  void repeatedPassThroughSourcesKeepTheirOldValue() {
    TensorRegion region({{"%a", type({2})}, {"%b", type({2})}, {"%c", type({2})}});
    region.setResults({"%b", "%a", "%c"});
    const auto updates = region.renderStateUpdates({{"%a", "%b"}, {"%b", "%a"}, {"%c", "%a"}});
    require(
        updates.size() == 5 && updates[2] == "  %a = %__region_saved_0;" &&
            updates[3] == "  %b = %__region_saved_1;" && updates[4] == "  %c = %__region_saved_1;",
        "repeated pass-through roots must share one snapshot of their old input"
    );
  }

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: tensor_region_api <test-name>\n";
    return 2;
  }
  const std::string name(argv[1]);
  try {
    if (name == "multi-results")
      multiResults();
    else if (name == "declarations-and-dead-nodes")
      declarationsAndDeadNodes();
    else if (name == "simultaneous-state-updates")
      simultaneousStateUpdates();
    else if (name == "shared-dependencies")
      multipleOutputsKeepSharedDependencies();
    else if (name == "invalid-graphs")
      invalidGraphsAreRejected();
    else if (name == "singleton-broadcast")
      singletonDimensionsBroadcastAcrossAxes();
    else if (name == "scalar-reduction")
      scalarReductionCanBroadcastBackToTensor();
    else if (name == "state-input-swap")
      inputRootsCanBeSwapped();
    else if (name == "state-input-cycle")
      inputRootsCanFormLongCycles();
    else if (name == "state-input-identity")
      identityRootCanShareAParallelUpdate();
    else if (name == "state-input-computed-mix")
      passThroughSourceCanSurviveComputedWriteback();
    else if (name == "state-input-repeated")
      repeatedPassThroughSourcesKeepTheirOldValue();
    else
      throw std::invalid_argument("unknown test name: " + name);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
