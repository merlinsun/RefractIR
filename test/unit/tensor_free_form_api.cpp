#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "reify/tensor_free_form.hpp"

namespace {
  using namespace refractir;
  using namespace refractir::reify;

  TensorTypeInfo type(Shape shape, TensorElement element = TensorElement::F32) {
    return TensorTypeInfo{std::move(shape), element};
  }

  FreeFormInput input(
      std::string name, TensorTypeInfo value, FreeFormRole role = FreeFormRole::State,
      FreeFormInit init = FreeFormInit::RuntimeInput, std::string parameter = {}
  ) {
    return FreeFormInput{
        TensorRegionInput{std::move(name), std::move(value)}, role, init, std::move(parameter)
    };
  }

  FreeFormInput state(std::string name, Shape shape, std::string parameter) {
    return input(std::move(name), type(shape), FreeFormRole::State, FreeFormInit::RuntimeInput,
                 std::move(parameter));
  }

  FreeFormInput zeroState(std::string name, Shape shape) {
    return input(std::move(name), type(shape), FreeFormRole::State, FreeFormInit::Zero);
  }

  FreeFormInput invariant(std::string name, Shape shape, std::string parameter) {
    return input(std::move(name), type(shape), FreeFormRole::Invariant, FreeFormInit::RuntimeInput,
                 std::move(parameter));
  }

  void require(bool condition, const std::string &message) {
    if (!condition)
      throw std::runtime_error(message);
  }

  template<typename Action>
  void requireInvalid(Action action, const std::string &needle) {
    try {
      action();
    } catch (const std::invalid_argument &error) {
      require(
          std::string(error.what()).find(needle) != std::string::npos,
          "refusal did not name '" + needle + "': " + error.what()
      );
      return;
    }
    throw std::runtime_error("expected an invalid free-form request to be refused");
  }

  template<typename Action>
  void requireExhaustion(Action action) {
    try {
      action();
    } catch (const std::runtime_error &error) {
      const std::string message = error.what();
      require(
          message.starts_with("free_form_exhaustion:"),
          "exhaustion must name its own prefix: " + message
      );
      return;
    }
    throw std::runtime_error("expected a free-form request to refuse after its attempts");
  }

  std::string render(const TensorRegion &region) {
    std::string text;
    for (const auto &line: region.renderInstructions())
      text += line + "\n";
    return text;
  }

  bool sameStructure(const FreeFormStructure &left, const FreeFormStructure &right) {
    return left.policy == right.policy && left.target == right.target &&
           left.actual == right.actual && left.nodes == right.nodes && left.depth == right.depth &&
           left.reductions == right.reductions && left.shapeChanges == right.shapeChanges &&
           left.distinctConsumers == right.distinctConsumers &&
           left.sharedValues == right.sharedValues &&
           left.maxDefUseDistance == right.maxDefUseDistance &&
           left.resultDependencies == right.resultDependencies &&
           left.operatorCounts == right.operatorCounts && left.consumers == right.consumers;
  }

  void twoTypedResults() {
    FreeFormRequest request;
    request.inputs = {state("%a", {2, 2}, "%a_in"), state("%b", {2, 2}, "%b_in")};
    request.results = {{"%rows", type({2})}, {"%cols", type({1, 2})}};
    request.policy = FreeFormPolicy::Deep;
    request.seed = 7;
    const auto built = buildFreeFormRegion(request);
    require(built.region.typeOf("%rows").shape == Shape({2}), "first result has the wrong shape");
    require(
        built.region.typeOf("%cols").shape == Shape({1, 2}), "second result has the wrong shape"
    );
    require(built.inputSpec.results.size() == 2, "both result names must be resolved");
    require(
        built.structure.resultDependencies.size() == 2,
        "every result needs one dependency mask"
    );
    require(built.structure.resultDependencies[0] != 0, "first result must depend on an input");
    require(built.structure.resultDependencies[1] != 0, "second result must depend on an input");
    require(built.structure.nodes > 0, "the builder must place at least one node");
  }

  void inputsNotOnlyStates() {
    bool sawAllThree = false;
    for (std::uint32_t seed = 1; seed <= 16; ++seed) {
      FreeFormRequest request;
      request.inputs = {
          state("%acc_x", {2, 2}, "%x"),
          zeroState("%acc_y", {2, 2}),
          invariant("%wide", {2, 2}, "%w"),
      };
      request.results = {{"%next_x", type({2, 2})}};
      request.policy = FreeFormPolicy::Unguided;
      request.seed = seed;
      const auto built = buildFreeFormRegion(request);
      require(built.inputSpec.inputs.size() == 3, "every typed input must be carried");
      require(
          built.inputSpec.inputs[2].role == FreeFormRole::Invariant,
          "the invariant's role must survive construction"
      );
      const auto mask = built.structure.resultDependencies.front();
      require(mask != 0, "the result must depend on at least one input");
      sawAllThree = sawAllThree || mask == 0b111ULL;
    }
    require(sawAllThree, "some seed must combine the state, the zero state, and the invariant");
  }

  void addOfOneValueIsOneConsumer() {
    FreeFormGraphView view;
    view.valueNames = {"%a", "%b", "%t0"};
    view.valueTypes = {type({2, 2}), type({2, 2}), type({2, 2})};
    view.inputCount = 2;
    view.nodeOps = {TensorOp::Add};
    view.nodeOperands = {{0, 0}};
    view.roots = {2};
    const auto structure = summarizeFreeFormGraph(view, FreeFormPolicy::Shared);
    require(
        structure.distinctConsumers == 1,
        "one operation naming a value twice is one distinct consumer"
    );
    require(structure.sharedValues == 0, "a value read once is not shared");
    require(structure.consumers.size() == 1, "only the consumed input carries a consumer count");
    require(structure.consumers.front().first == "%a", "the consumed input must be recorded");
    require(structure.consumers.front().second == 1, "the input has one distinct consumer");
  }

  void pruningOmitsDeadValues() {
    FreeFormGraphView view;
    view.valueNames = {"%a", "%b", "%t0", "%t1", "%t2"};
    view.valueTypes = {type({2, 2}), type({2, 2}), type({2, 2}), type({2, 2}), type({2, 2})};
    view.inputCount = 2;
    view.nodeOps = {TensorOp::Add, TensorOp::Add, TensorOp::Add};
    view.nodeOperands = {{0, 1}, {1, 0}, {3, 0}};
    view.roots = {2};
    const auto structure = summarizeFreeFormGraph(view, FreeFormPolicy::Unguided);
    require(structure.nodes == 1, "only the live node may be counted");
    require(structure.distinctConsumers == 2, "the live node consumes both inputs once each");
    require(structure.consumers.size() == 2, "dead node results carry no consumer count");
    require(
        std::none_of(
            structure.consumers.begin(), structure.consumers.end(),
            [](const auto &entry) { return entry.first == "%t1" || entry.first == "%t2"; }
        ),
        "a dead value must not appear in the consumer record"
    );
    require(structure.operatorCounts.size() == 1, "only one operator kind is live");
    require(structure.operatorCounts.front().second == 1, "only the live operator is counted");
    require(structure.depth == 1, "the live depth is one");
    require(structure.shapeChanges == 0, "an add does not change its operand's shape");
    require(
        structure.maxDefUseDistance == 2,
        "the live edge spans from the first input to its consumer"
    );
  }

  void deterministicAndImpossible() {
    FreeFormRequest request;
    request.inputs = {state("%acc", {2, 2}, "%x")};
    request.results = {{"%next", type({2, 2})}};
    request.policy = FreeFormPolicy::CrossState;
    request.seed = 3;
    request.limits.maxNodes = 8;
    const auto first = buildFreeFormRegion(request);
    const auto second = buildFreeFormRegion(request);
    require(
        render(first.region) == render(second.region),
        "the same seed and policy must build the same body"
    );
    require(
        sameStructure(first.structure, second.structure),
        "the same seed and policy must record the same structure"
    );
    FreeFormRequest impossible = request;
    impossible.results = {{"%big", type({4, 4})}};
    requireExhaustion([&] { (void) buildFreeFormRegion(impossible); });
    FreeFormRequest noNodes = request;
    noNodes.limits.maxNodes = 0;
    requireInvalid(
        [&] { (void) buildFreeFormRegion(noNodes); }, "free-form limits admit no node"
    );
  }

  TensorRegion canonicalRegion() {
    TensorRegion region({{"%acc_x", type({2, 2})}, {"%acc_y", type({2, 2})}});
    region.add("%shared", TensorOp::Add, {"%acc_x", "%acc_y"});
    region.add("%next_x", TensorOp::Add, {"%shared", "%acc_x"});
    region.add("%next_y", TensorOp::Add, {"%shared", "%acc_y"});
    region.setResults({"%next_x", "%next_y"});
    return region;
  }

  const char *canonicalSource = R"(fun @kernel(%x: tensor [2, 2] f32) : tensor [2, 2] f32 {
  let mut %i: i32 = undef;
  let mut %acc_x: tensor [2, 2] f32 = undef;
  let mut %acc_y: tensor [2, 2] f32 = {0.0, 0.0, 0.0, 0.0};
  let mut %out: tensor [2, 2] f32 = undef;
  let mut %shared: tensor [2, 2] f32 = undef;
  let mut %next_x: tensor [2, 2] f32 = undef;
  let mut %next_y: tensor [2, 2] f32 = undef;
^entry:
  %acc_x = %x;
  %i = 0;
  br ^header;
^header:
  br %i < 2, ^body, ^exit;
^body:
  %shared = tensor add %acc_x, %acc_y;
  %next_x = tensor add %shared, %acc_x;
  %next_y = tensor add %shared, %acc_y;
  %acc_x = %next_x;
  %acc_y = %next_y;
  %i = %i + 1;
  br ^header;
^exit:
  %out = tensor add %acc_x, %acc_y;
  ret %out;
}
)";

  void canonicalBytes() {
    const auto region = canonicalRegion();
    const auto source = assembleCanonicalLoop(region, 2);
    require(source.refusal.empty(), "the canonical region must be admitted: " + source.refusal);
    require(
        source.source == canonicalSource,
        "the canonical adapter must emit the canonical counted loop byte for byte:\n" + source.source
    );
    TensorRegion renamed({{"%acc_x", type({2, 2})}, {"%acc_y", type({2, 2})}});
    renamed.add("%a_next", TensorOp::Add, {"%acc_x", "%acc_y"});
    renamed.add("%b_next", TensorOp::Add, {"%acc_y", "%acc_x"});
    renamed.setResults({"%a_next", "%b_next"});
    require(
        !assembleCanonicalLoop(renamed, 2).refusal.empty(),
        "a region whose results are not the canonical names must be refused"
    );
    TensorRegion twoStatesOnly({{"%acc_x", type({2, 2})}, {"%acc_y", type({2, 2})}});
    twoStatesOnly.add("%next_x", TensorOp::Add, {"%acc_x", "%acc_y"});
    twoStatesOnly.setResults({"%next_x"});
    require(
        !assembleCanonicalLoop(twoStatesOnly, 2).refusal.empty(),
        "a region without the second result must be refused"
    );
  }

  TensorRegion threeStateRegion() {
    TensorRegion region(
        {{"%acc_x", type({2, 2})}, {"%acc_y", type({2, 2})}, {"%acc_z", type({2, 2})}}
    );
    region.add("%s01", TensorOp::Add, {"%acc_x", "%acc_y"});
    region.add("%next_x", TensorOp::Add, {"%s01", "%acc_x"});
    region.add("%next_y", TensorOp::Add, {"%s01", "%acc_y"});
    region.add("%s23", TensorOp::Add, {"%s01", "%acc_z"});
    region.add("%next_z", TensorOp::Add, {"%s23", "%acc_z"});
    region.setResults({"%next_x", "%next_y", "%next_z"});
    return region;
  }

  FreeFormLoop threeStateLoop() {
    FreeFormLoop loop;
    loop.inputs = {
        state("%acc_x", {2, 2}, "%x"),
        state("%acc_y", {2, 2}, "%y"),
        zeroState("%acc_z", {2, 2}),
    };
    loop.results = {"%next_x", "%next_y", "%next_z"};
    loop.updates = {
        {"%acc_x", "%next_x"}, {"%acc_y", "%next_y"}, {"%acc_z", "%next_z"}
    };
    loop.tripCount = 2;
    loop.finalTerms = {"%acc_x", "%acc_y", "%acc_z"};
    loop.finalOp = TensorOp::Add;
    return loop;
  }

  void threeStateAssembly() {
    const auto region = threeStateRegion();
    const auto source = assembleCountedLoop(region, threeStateLoop());
    require(
        source.find("fun @kernel(%x: tensor [2, 2] f32, %y: tensor [2, 2] f32) : "
                    "tensor [2, 2] f32") != std::string::npos,
        "runtime inputs become parameters and zero states do not:\n" + source
    );
    require(
        source.find("let mut %acc_z: tensor [2, 2] f32 = {0.0, 0.0, 0.0, 0.0};") !=
            std::string::npos,
        "a zero state must be initialized from zero:\n" + source
    );
    require(
        source.find("  %acc_x = %x;\n  %acc_y = %y;\n") != std::string::npos,
        "a runtime state must copy its parameter:\n" + source
    );
    require(
        source.find("br %i < 2, ^body, ^exit;") != std::string::npos,
        "the declared trip count must reach the header:\n" + source
    );
    require(
        source.find("  %out = tensor add %acc_x, %acc_y;\n  %out = tensor add %out, %acc_z;\n"
                    "  ret %out;\n") != std::string::npos,
        "the declared final expression must be folded after the loop:\n" + source
    );
    const auto firstUpdate = source.find("  %acc_x = %next_x;");
    require(firstUpdate != std::string::npos, "the first state update is missing:\n" + source);
    for (const auto &result: {"%next_x", "%next_y", "%next_z"}) {
      const auto computed = source.find(std::string("  ") + result + " = tensor");
      require(
          computed != std::string::npos && computed < firstUpdate,
          std::string("every result must be computed before any state is written: ") + result
      );
    }
  }

  void invariantInitialization() {
    TensorRegion region({{"%acc_x", type({2, 2})}, {"%bias", type({2, 2})}});
    region.add("%next_x", TensorOp::Add, {"%acc_x", "%bias"});
    region.setResults({"%next_x"});
    FreeFormLoop loop;
    loop.inputs = {state("%acc_x", {2, 2}, "%x"),
                   input("%bias", type({2, 2}), FreeFormRole::Invariant, FreeFormInit::Zero)};
    loop.results = {"%next_x"};
    loop.updates = {{"%acc_x", "%next_x"}};
    loop.finalTerms = {"%acc_x"};
    const auto source = assembleCountedLoop(region, loop);
    require(
        source.find("let mut %bias: tensor [2, 2] f32 = {0.0, 0.0, 0.0, 0.0};") !=
            std::string::npos,
        "the invariant must be initialized at entry:\n" + source
    );
    require(
        source.find("fun @kernel(%x: tensor [2, 2] f32) : tensor [2, 2] f32") != std::string::npos,
        "an invariant initialized from zero takes no parameter:\n" + source
    );
    require(
        source.find("  %bias = ") == std::string::npos,
        "an invariant is never assigned inside the loop:\n" + source
    );
    FreeFormLoop invariantUpdate = loop;
    invariantUpdate.updates = {{"%bias", "%next_x"}};
    requireInvalid(
        [&] { (void) assembleCountedLoop(region, invariantUpdate); }, "cannot update an invariant"
    );
  }

  void simultaneousUpdates() {
    TensorRegion region({{"%acc_x", type({2, 2})}, {"%acc_y", type({2, 2})}});
    region.setResults({"%acc_y", "%acc_x"});
    FreeFormLoop loop;
    loop.inputs = {state("%acc_x", {2, 2}, "%x"), state("%acc_y", {2, 2}, "%y")};
    loop.results = {"%acc_y", "%acc_x"};
    loop.updates = {{"%acc_x", "%acc_y"}, {"%acc_y", "%acc_x"}};
    loop.finalTerms = {"%acc_x"};
    const auto source = assembleCountedLoop(region, loop);
    require(
        source.find("%__region_saved") != std::string::npos,
        "a swapped update must snapshot one source:\n" + source
    );
    const auto body = source.find("^body:");
    require(body != std::string::npos, "the assembled loop must have a body");
    const auto saved = source.find("  %__region_saved_", body);
    const auto firstUpdate = source.find("  %acc_x = %", body);
    require(
        saved != std::string::npos && firstUpdate != std::string::npos && saved < firstUpdate,
        "the snapshot must be taken before the state writes:\n" + source
    );
    require(
        source.find("  %acc_y = %__region_saved_1;") != std::string::npos,
        "the second state must read its saved old value:\n" + source
    );
  }

  void rejectedUpdates() {
    TensorRegion region({{"%acc_x", type({2, 2})}, {"%acc_y", type({2, 2})}});
    region.add("%next_x", TensorOp::Sum, {"%acc_x"}, 0, false);
    region.add("%next_y", TensorOp::Add, {"%acc_y", "%acc_y"});
    region.setResults({"%next_x", "%next_y"});
    FreeFormLoop loop;
    loop.inputs = {state("%acc_x", {2, 2}, "%x"), state("%acc_y", {2, 2}, "%y")};
    loop.results = {"%next_x", "%next_y"};
    loop.updates = {{"%acc_x", "%next_x"}, {"%acc_y", "%next_y"}};
    loop.finalTerms = {"%acc_x"};
    requireInvalid(
        [&] { (void) assembleCountedLoop(region, loop); }, "types do not match"
    );
    FreeFormLoop missing = loop;
    missing.updates = {{"%acc_x", "%next_x"}, {"%acc_x", "%next_y"}};
    requireInvalid(
        [&] { (void) assembleCountedLoop(region, missing); }, "updates a state twice"
    );
    FreeFormLoop foreign = loop;
    foreign.finalTerms = {"%not_a_local"};
    requireInvalid(
        [&] { (void) assembleCountedLoop(region, foreign); }, "is not a local"
    );
    FreeFormLoop zeroTrips = loop;
    zeroTrips.tripCount = 0;
    requireInvalid(
        [&] { (void) assembleCountedLoop(region, zeroTrips); }, "trip count"
    );
  }

  void policyNames() {
    require(
        !parseFreeFormPolicy("nope").has_value(), "an unknown policy must have no policy"
    );
    const std::vector<std::string> names{
        "unguided", "balanced", "deep", "wide", "shared", "reduction-heavy", "cross-state"
    };
    for (const auto &name: names) {
      const auto policy = parseFreeFormPolicy(name);
      require(policy.has_value(), "every documented policy must parse: " + name);
      require(freeFormPolicyName(*policy) == name, "a policy must round trip: " + name);
    }
    require(
        freeFormPolicyNames.find("reduction-heavy") != std::string_view::npos,
        "the documented list must name reduction-heavy"
    );
  }

  void selectionBounds() {
    FreeFormActionTraits shallow;
    shallow.op = TensorOp::Add;
    shallow.depth = 0;
    shallow.unusedOperands = 2;
    shallow.inputs = 1;
    FreeFormActionTraits deep = shallow;
    deep.depth = 6;
    FreeFormActionTraits reused = shallow;
    reused.reusedConsumers = 2;
    FreeFormActionTraits sum = shallow;
    sum.op = TensorOp::Sum;
    FreeFormActionTraits wide = shallow;
    wide.inputs = 2;

    require(
        freeFormActionWeight(FreeFormPolicy::Unguided, shallow) ==
            freeFormActionWeight(FreeFormPolicy::Unguided, reused),
        "unguided must score every action the same"
    );
    require(
        freeFormActionWeight(FreeFormPolicy::Deep, deep) >
            freeFormActionWeight(FreeFormPolicy::Deep, shallow),
        "a depth bias must prefer the deeper operand"
    );
    require(
        freeFormActionWeight(FreeFormPolicy::Shared, reused) >
            freeFormActionWeight(FreeFormPolicy::Shared, shallow),
        "a sharing bias must prefer a value that already has a consumer"
    );
    require(
        freeFormActionWeight(FreeFormPolicy::ReductionHeavy, sum) >
            freeFormActionWeight(FreeFormPolicy::ReductionHeavy, shallow),
        "a reduction bias must prefer a reduction"
    );
    require(
        freeFormActionWeight(FreeFormPolicy::CrossState, wide) >
            freeFormActionWeight(FreeFormPolicy::CrossState, shallow),
        "a cross-state bias must prefer an action over two inputs"
    );
    std::vector<double> flat{1.0, 1.0, 1.0};
    require(freeFormSelectWeighted(flat, 0.0) == 0, "a zero draw selects the first action");
    require(freeFormSelectWeighted(flat, 1.0) == 1, "a draw at one selects the second action");
    require(freeFormSelectWeighted(flat, 2.5) == 2, "a draw past two selects the third action");
    requireInvalid(
        [&] { (void) freeFormSelectWeighted({}, 0.0); }, "at least one weight"
    );
    requireInvalid(
        [&] { (void) freeFormSelectWeighted({1.0, -1.0}, 0.0); }, "must be positive"
    );
    requireInvalid(
        [&] { (void) freeFormSelectWeighted(flat, 3.0); }, "outside the weight total"
    );
  }

  std::string threeStateSource() {
    return assembleCountedLoop(threeStateRegion(), threeStateLoop());
  }

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: tensor_free_form_api <test-name>\n";
    return 2;
  }
  const std::string name(argv[1]);
  try {
    if (name == "two-typed-results")
      twoTypedResults();
    else if (name == "inputs-not-only-states")
      inputsNotOnlyStates();
    else if (name == "add-of-one-value-is-one-consumer")
      addOfOneValueIsOneConsumer();
    else if (name == "pruning-omits-dead-values")
      pruningOmitsDeadValues();
    else if (name == "deterministic-and-impossible")
      deterministicAndImpossible();
    else if (name == "canonical-bytes")
      canonicalBytes();
    else if (name == "three-state-assembly")
      threeStateAssembly();
    else if (name == "invariant-initialization")
      invariantInitialization();
    else if (name == "simultaneous-updates")
      simultaneousUpdates();
    else if (name == "rejected-updates")
      rejectedUpdates();
    else if (name == "policy-names")
      policyNames();
    else if (name == "selection-bounds")
      selectionBounds();
    else if (name == "three-state-source") {
      std::cout << threeStateSource();
      return 0;
    } else if (name == "canonical-source") {
      const auto assembly = assembleCanonicalLoop(canonicalRegion(), 2);
      if (!assembly.refusal.empty()) {
        std::cerr << assembly.refusal << '\n';
        return 1;
      }
      std::cout << assembly.source;
      return 0;
    } else
      throw std::invalid_argument("unknown test name: " + name);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
