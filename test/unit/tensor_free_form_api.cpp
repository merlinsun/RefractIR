#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <sstream>
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
    // Every policy must leave every action with a positive weight, including a
    // heavily reused value under the balanced policy.
    for (const auto policy: {FreeFormPolicy::Unguided, FreeFormPolicy::Balanced,
                             FreeFormPolicy::Deep, FreeFormPolicy::Wide,
                             FreeFormPolicy::Shared, FreeFormPolicy::ReductionHeavy,
                             FreeFormPolicy::CrossState}) {
      for (const auto reused: {0U, 5U}) {
        for (const auto unused: {0U, 2U}) {
          for (const auto depth: {0U, 12U}) {
            FreeFormActionTraits traits;
            traits.op = TensorOp::Add;
            traits.depth = depth;
            traits.reusedConsumers = reused;
            traits.unusedOperands = unused;
            traits.inputs = 1;
            require(
                freeFormActionWeight(policy, traits) > 0.0,
                "a policy must never exclude an action by weight"
            );
            traits.shapeChange = true;
            require(
                freeFormActionWeight(policy, traits) > 0.0,
                "a shape-changing action must keep a positive weight"
            );
          }
        }
      }
    }
  }

  FreeFormActionTraits candidate(
      TensorOp op, std::uint32_t depth, std::uint32_t reused, std::uint32_t unused
  ) {
    FreeFormActionTraits traits;
    traits.op = op;
    traits.depth = depth;
    traits.reusedConsumers = reused;
    traits.unusedOperands = unused;
    traits.inputs = 1;
    return traits;
  }

  std::vector<double> weightsFor(FreeFormPolicy policy, const std::vector<FreeFormActionTraits> &traits) {
    std::vector<double> weights;
    for (const auto &candidate: traits)
      weights.push_back(freeFormActionWeight(policy, candidate));
    return weights;
  }

  void unguidedIgnoresTheRank() {
    const std::vector<FreeFormActionTraits> traits{
        candidate(TensorOp::Add, 6, 0, 0), candidate(TensorOp::Add, 0, 0, 2)
    };
    const auto deep = weightsFor(FreeFormPolicy::Deep, traits);
    require(deep[0] > deep[1], "the depth bias must rank the deep action top");
    require(
        freeFormSelectWeighted(weightsFor(FreeFormPolicy::Unguided, traits), 1.5) == 1,
        "an unguided draw may select the action a biased policy ranks last"
    );
    require(
        freeFormSelectWeighted(deep, 1.5) == 0,
        "the same draw under the depth bias selects the action it ranks top"
    );
  }

  void depthBiasChangesTheChoice() {
    const std::vector<FreeFormActionTraits> traits{
        candidate(TensorOp::Add, 6, 0, 0), candidate(TensorOp::Add, 0, 0, 2)
    };
    require(
        freeFormSelectWeighted(weightsFor(FreeFormPolicy::Deep, traits), 1.5) == 0,
        "the depth bias selects the deep action for this draw"
    );
    require(
        freeFormSelectWeighted(weightsFor(FreeFormPolicy::Wide, traits), 1.5) == 1,
        "the width bias selects the fresh-operand action for the same draw"
    );
  }

  void sharingBiasSelectsASecondConsumer() {
    const std::vector<FreeFormActionTraits> traits{
        candidate(TensorOp::Add, 1, 1, 0), candidate(TensorOp::Add, 1, 0, 2)
    };
    const auto shared = weightsFor(FreeFormPolicy::Shared, traits);
    require(shared[0] > shared[1], "the sharing bias must prefer a second consumer");
    require(
        freeFormSelectWeighted(weightsFor(FreeFormPolicy::Unguided, traits), 1.5) == 1,
        "an unguided draw selects the fresh action for this draw"
    );
    require(
        freeFormSelectWeighted(shared, 1.5) == 0,
        "the sharing bias selects the action that adds a second consumer"
    );
  }

  void reductionBiasSelectsAReduction() {
    const std::vector<FreeFormActionTraits> traits{
        candidate(TensorOp::Sum, 1, 0, 1), candidate(TensorOp::Add, 1, 0, 1)
    };
    require(
        freeFormSelectWeighted(weightsFor(FreeFormPolicy::ReductionHeavy, traits), 1.5) == 0,
        "the reduction bias selects the reduction for this draw"
    );
    require(
        freeFormSelectWeighted(weightsFor(FreeFormPolicy::Unguided, traits), 1.5) == 1,
        "an unguided draw selects the action after the reduction"
    );
  }

  void replayIsSeparateFromDistribution() {
    FreeFormRequest request;
    request.inputs = {state("%acc_x", {2, 2}, "%x"), zeroState("%acc_y", {2, 2})};
    request.results = {{"%next_x", type({2, 2})}, {"%next_y", type({2, 2})}};
    request.policy = FreeFormPolicy::Shared;
    request.seed = 42;
    const auto first = buildFreeFormRegion(request);
    const auto second = buildFreeFormRegion(request);
    require(
        render(first.region) == render(second.region) &&
            sameStructure(first.structure, second.structure),
        "one seed must replay its body and its record"
    );
    std::vector<std::string> bodies;
    for (std::uint32_t seed = 1; seed <= 12; ++seed) {
      FreeFormRequest varied = request;
      varied.seed = seed;
      bodies.push_back(render(buildFreeFormRegion(varied).region));
    }
    require(
        std::adjacent_find(bodies.begin(), bodies.end(), std::not_equal_to<>()) != bodies.end(),
        "different seeds must not all produce the same body"
    );
  }

  void featuresOffKeepTheCore() {
    FreeFormRequest request;
    request.inputs = {state("%acc_x", {2, 2}, "%x"), zeroState("%acc_y", {2, 2})};
    request.results = {{"%next_x", type({2, 2})}, {"%next_y", type({2, 2})}};
    request.policy = FreeFormPolicy::Shared;
    for (std::uint32_t seed = 1; seed <= 12; ++seed) {
      request.seed = seed;
      const auto built = buildFreeFormRegion(request);
      for (const auto &count: built.structure.operatorCounts) {
        require(
            count.first != "transpose" && count.first != "expand",
            "a request without the feature must not build the operation"
        );
      }
    }
  }

  void featuresEnableTransposeAndExpand() {
    FreeFormRequest request;
    request.inputs = {state("%acc_x", {2, 2}, "%x"), state("%acc_y", {2, 2}, "%y")};
    request.results = {{"%next_x", type({2, 2})}, {"%next_y", type({2, 2})}};
    request.policy = FreeFormPolicy::Unguided;
    request.features = FreeFormFeatures{true, true};
    request.limits.maxRank = 3;
    bool sawTranspose = false;
    bool sawExpand = false;
    for (std::uint32_t seed = 1; seed <= 16; ++seed) {
      request.seed = seed;
      const auto built = buildFreeFormRegion(request);
      for (const auto &count: built.structure.operatorCounts) {
        sawTranspose = sawTranspose || count.first == "transpose";
        sawExpand = sawExpand || count.first == "expand";
      }
    }
    require(sawTranspose, "a transposable value must be selected at some seed");
    require(sawExpand, "an expandable value must be selected at some seed");
  }

  void featuresRespectTheRankLimit() {
    FreeFormRequest request;
    request.inputs = {state("%only", {2}, "%x")};
    request.results = {{"%next", type({2})}};
    request.features = FreeFormFeatures{true, true};
    request.limits.maxRank = 1;
    for (std::uint32_t seed = 1; seed <= 12; ++seed) {
      request.seed = seed;
      const auto built = buildFreeFormRegion(request);
      for (const auto &count: built.structure.operatorCounts) {
        require(
            count.first != "transpose" && count.first != "expand",
            "the rank limit must refuse an operation that raises the rank"
        );
      }
    }
  }

  void featuresReplayExactly() {
    FreeFormRequest request;
    request.inputs = {state("%acc_x", {2, 2}, "%x"), zeroState("%acc_y", {2, 2})};
    request.results = {{"%next_x", type({2, 2})}, {"%next_y", type({2, 2})}};
    request.policy = FreeFormPolicy::Wide;
    request.features = FreeFormFeatures{true, true};
    request.seed = 21;
    const auto first = buildFreeFormRegion(request);
    const auto second = buildFreeFormRegion(request);
    require(
        render(first.region) == render(second.region),
        "the same seed and features must build the same body"
    );
    require(
        sameStructure(first.structure, second.structure),
        "the same seed and features must record the same structure"
    );
    require(
        first.structure.shapeChanges >= 1,
        "a transpose or expand that changes the shape must be counted"
    );
  }

  void extremaBinaryTypes() {
    TensorRegion region({{"%a", type({2, 2})}, {"%b", type({2, 2})}});
    region.add("%lo", TensorOp::Min, {"%a", "%b"});
    region.add("%hi", TensorOp::Max, {"%a", "%b"});
    region.setResults({"%lo", "%hi"});
    require(region.typeOf("%lo").shape == Shape({2, 2}), "min must broadcast its operands");
    require(region.typeOf("%hi").shape == Shape({2, 2}), "max must broadcast its operands");
    const auto instructions = region.renderInstructions();
    require(
        instructions[0] == "  %lo = tensor min %a, %b;",
        "min must render two operands: " + instructions[0]
    );
    require(
        instructions[1] == "  %hi = tensor max %a, %b;",
        "max must render two operands: " + instructions[1]
    );
  }

  void extremaReduceDropShape() {
    TensorRegion region({{"%a", type({2, 3})}});
    region.add("%lo", TensorOp::ReduceMin, {"%a"}, 1, false);
    region.setResults({"%lo"});
    require(region.typeOf("%lo").shape == Shape({2}), "a dropped rmin must lose its axis");
    require(
        region.renderInstructions().front() == "  %lo = tensor rmin %a, 1, drop;",
        "rmin must render its axis and keep mode"
    );
  }

  void extremaReduceKeepShape() {
    TensorRegion region({{"%a", type({2, 3})}});
    region.add("%hi", TensorOp::ReduceMax, {"%a"}, 1, true);
    region.add("%neg", TensorOp::ReduceMin, {"%a"}, -1, true);
    region.setResults({"%hi", "%neg"});
    require(region.typeOf("%hi").shape == Shape({2, 1}), "a kept rmax must set its axis to one");
    require(region.typeOf("%neg").shape == Shape({2, 1}), "a negative axis must normalize");
    const auto instructions = region.renderInstructions();
    require(
        instructions[0] == "  %hi = tensor rmax %a, 1, keep;",
        "rmax must render keep: " + instructions[0]
    );
    require(
        instructions[1] == "  %neg = tensor rmin %a, 1, keep;",
        "rmin must render its normalized axis: " + instructions[1]
    );
  }

  void extremaRejectAxis() {
    requireInvalid(
        [] {
          TensorRegion region({{"%a", type({2, 3})}});
          region.add("%lo", TensorOp::ReduceMin, {"%a"}, 2, false);
        },
        "axis"
    );
    requireInvalid(
        [] {
          TensorRegion region({{"%a", type({2, 3})}});
          region.add("%hi", TensorOp::ReduceMax, {"%a"}, -3, false);
        },
        "axis"
    );
  }

  void extremaRenderRoundTrip() {
    TensorRegion region({{"%a", type({2, 2})}});
    region.add("%lo", TensorOp::Min, {"%a", "%a"});
    region.add("%hi", TensorOp::Max, {"%a", "%a"});
    region.add("%rlo", TensorOp::ReduceMin, {"%a"}, 0, false);
    region.add("%rhi", TensorOp::ReduceMax, {"%a"}, 1, true);
    region.setResults({"%lo", "%hi", "%rlo", "%rhi"});
    const auto instructions = region.renderInstructions();
    const std::vector<std::string> expected{
        "  %lo = tensor min %a, %a;",
        "  %hi = tensor max %a, %a;",
        "  %rlo = tensor rmin %a, 0, drop;",
        "  %rhi = tensor rmax %a, 1, keep;",
    };
    require(instructions == expected, "every extrema form must render its SIR spelling");
    require(
        region.typeOf("%rlo").shape == Shape({2}),
        "a rank-two dropped rmin keeps one axis"
    );
    require(
        region.typeOf("%rhi").shape == Shape({2, 1}),
        "a rank-two kept rmax keeps the rank"
    );
  }

  void extremaRankOneKeep() {
    TensorRegion region({{"%a", type({2})}});
    region.add("%lo", TensorOp::ReduceMin, {"%a"}, 0, true);
    region.setResults({"%lo"});
    require(region.typeOf("%lo").shape == Shape({1}), "a rank-one keep must yield [1]");
  }

  bool isExtremaOp(const std::string &name) {
    return name == "min" || name == "max" || name == "rmin" || name == "rmax";
  }

  void extremaActionEnumerated() {
    FreeFormRequest request;
    request.inputs = {state("%acc_x", {2, 2}, "%x"), zeroState("%acc_y", {2, 2})};
    request.results = {{"%next_x", type({2, 2})}, {"%next_y", type({2, 2})}};
    request.policy = FreeFormPolicy::Unguided;
    request.features.extrema = true;
    std::vector<std::string> seen;
    for (std::uint32_t seed = 1; seed <= 24; ++seed) {
      request.seed = seed;
      const auto built = buildFreeFormRegion(request);
      for (const auto &count: built.structure.operatorCounts)
        if (isExtremaOp(count.first) &&
            std::find(seen.begin(), seen.end(), count.first) == seen.end())
          seen.push_back(count.first);
    }
    for (const auto &expected: {"min", "max", "rmin", "rmax"})
      require(
          std::find(seen.begin(), seen.end(), expected) != seen.end(),
          std::string("the builder must enumerate extrema action ") + expected
      );
  }

  void extremaActionWeighted() {
    const auto traits = [](TensorOp op) {
      FreeFormActionTraits value;
      value.op = op;
      value.inputs = 1;
      return value;
    };
    const auto rminWeight = freeFormActionWeight(FreeFormPolicy::ReductionHeavy, traits(TensorOp::ReduceMin));
    const auto sumWeight = freeFormActionWeight(FreeFormPolicy::ReductionHeavy, traits(TensorOp::Sum));
    const auto addWeight = freeFormActionWeight(FreeFormPolicy::ReductionHeavy, traits(TensorOp::Add));
    require(rminWeight == sumWeight, "a reduction-heavy policy must score rmin like sum");
    require(rminWeight > addWeight, "a reduction-heavy policy must prefer rmin to an add");
    require(
        freeFormActionWeight(FreeFormPolicy::Unguided, traits(TensorOp::ReduceMin)) ==
            freeFormActionWeight(FreeFormPolicy::Unguided, traits(TensorOp::Add)),
        "an unguided policy must not prefer extrema to an add"
    );
  }

  void extremaCountedAsReduction() {
    FreeFormGraphView view;
    view.valueNames = {"%a", "%lo"};
    view.valueTypes = {type({2, 2}), type({2})};
    view.inputCount = 1;
    view.nodeOps = {TensorOp::ReduceMin};
    view.nodeOperands = {{0}};
    view.roots = {1};
    const auto structure = summarizeFreeFormGraph(view, FreeFormPolicy::ReductionHeavy);
    require(structure.reductions == 1, "an rmin node must count as a reduction");
    require(structure.operatorCounts.size() == 1, "only the rmin operator is live");
    require(structure.operatorCounts.front().first == "rmin", "the operator count must name rmin");
    require(structure.shapeChanges == 1, "a dropped rmin changes the shape");
  }

  void extremaMultiplicationBudgetUntouched() {
    FreeFormRequest request;
    request.inputs = {state("%acc_x", {2, 2}, "%x"), zeroState("%acc_y", {2, 2})};
    request.results = {{"%next_x", type({2, 2})}, {"%next_y", type({2, 2})}};
    request.limits.maxMultiplications = 0;
    request.features.extrema = true;
    bool sawExtrema = false;
    for (std::uint32_t seed = 1; seed <= 16; ++seed) {
      request.seed = seed;
      const auto built = buildFreeFormRegion(request);
      for (const auto &count: built.structure.operatorCounts) {
        require(count.first != "mul", "an exhausted multiplication budget must still refuse mul");
        sawExtrema = sawExtrema || isExtremaOp(count.first);
      }
    }
    require(sawExtrema, "extrema must not consume the multiplication budget");
  }

  void extremaReplayDeterministic() {
    FreeFormRequest request;
    request.inputs = {state("%acc_x", {2, 2}, "%x"), zeroState("%acc_y", {2, 2})};
    request.results = {{"%next_x", type({2, 2})}, {"%next_y", type({2, 2})}};
    request.policy = FreeFormPolicy::Wide;
    request.features.extrema = true;
    request.features.transpose = true;
    request.features.expand = true;
    request.limits.maxRank = 3;
    request.seed = 33;
    const auto first = buildFreeFormRegion(request);
    const auto second = buildFreeFormRegion(request);
    bool sawExtrema = false;
    for (const auto &count: first.structure.operatorCounts)
      sawExtrema = sawExtrema || isExtremaOp(count.first);
    require(sawExtrema, "the replayed body must use an extrema operation");
    require(
        render(first.region) == render(second.region),
        "the same seed and features must build the same extrema body"
    );
    require(sameStructure(first.structure, second.structure), "the same record must replay");
  }

  void extremaRankOneDropIsScalar() {
    TensorRegion region({{"%a", type({2})}});
    region.add("%lo", TensorOp::ReduceMin, {"%a"}, 0, false);
    region.setResults({"%lo"});
    require(region.typeOf("%lo").shape.empty(), "a rank-one dropped rmin must be a scalar");
    require(
        region.renderInstructions().front() == "  %lo = tensor rmin %a, 0, drop;",
        "a scalar extrema reduction must render drop"
    );
  }

  void windowFactsInterior() {
    const auto facts = classifyWindow(2, 2, 4, 2, false, false);
    require(facts.regime == "interior", "every row must be active on every trip");
    require(facts.shapeValidity == "absent", "v2 has no shape validity");
    require(facts.activePredicate == "all_true", "an interior predicate is always true");
    require(
        classifyWindow(2, 2, 3, 2, false, false).regime == "tail",
        "the boundary case equality is a tail"
    );
  }

  void windowFactsStrideZeroTail() {
    const auto facts = classifyWindow(2, 0, 1, 3, false, false);
    require(facts.regime == "tail", "rows two active one must be a tail");
    require(
        facts.activePredicate == "constant_partial",
        "a zero stride makes the predicate independent of the trip"
    );
  }

  void windowFactsStrideZeroRowsOne() {
    const auto facts = classifyWindow(1, 0, 1, 5, false, false);
    require(facts.regime == "interior", "one row never reaches past its own address");
    require(facts.activePredicate == "all_true", "the single row is always active");
  }

  void windowFactsConstantPartial() {
    const auto facts = classifyWindow(2, 0, 1, 1, false, false);
    require(facts.regime == "tail", "row one is inactive even on the first trip");
    require(
        facts.activePredicate == "constant_partial",
        "the predicate is constant but not all true"
    );
  }

  void windowFactsPaddedV3() {
    const auto padded = classifyWindow(2, 2, 8, 2, true, true);
    require(padded.regime == "interior", "the interior boundary is unchanged by padding");
    require(
        padded.shapeValidity == "nonconstant",
        "a padded v3 result carries nonconstant shape validity"
    );
    require(
        classifyWindow(2, 2, 8, 2, true, false).shapeValidity == "constant",
        "an exact tile carries constant shape validity"
    );
  }

  void windowFactsMaskNotObserved() {
    for (const auto &facts: {classifyWindow(1, 0, 1, 1, false, false),
                             classifyWindow(2, 4, 2, 3, false, false),
                             classifyWindow(3, 1, 8, 2, true, true)})
      require(
          facts.maskAfterCompilation == "not_observed",
          "assembly cannot observe compiled masks"
      );
  }

  FreeFormLoop windowLoop() {
    FreeFormLoop loop;
    loop.inputs = {
        state("%acc_x", {2, 2}, "%x"),
        zeroState("%acc_y", {2, 2}),
        FreeFormInput{
            TensorRegionInput{"%source", type({4, 2})}, FreeFormRole::Invariant,
            FreeFormInit::RuntimeInput, "%src"},
        FreeFormInput{
            TensorRegionInput{"%tile", type({2, 2})}, FreeFormRole::Invariant,
            FreeFormInit::Window, ""},
    };
    loop.results = {"%next_x", "%next_y"};
    loop.updates = {{"%acc_x", "%next_x"}, {"%acc_y", "%next_y"}};
    loop.finalTerms = {"%acc_x", "%acc_y"};
    FreeFormWindow window;
    window.result = "%tile";
    window.source = "%src";
    window.fillKind = FreeFormFillKind::Float;
    window.fillFloat = 0.0;
    window.rows = 2;
    window.stride = 1;
    window.active = 4;
    loop.windows = {window};
    return loop;
  }

  void windowInitRequiresDeclaredInvariant() {
    auto loop = windowLoop();
    loop.windows.front().result = "%missing";
    requireInvalid([&] { validateFreeFormWindows(loop); }, "window-initialized input");
  }

  void windowSourceMustBeAParameter() {
    auto loop = windowLoop();
    loop.windows.front().source = "%source";
    requireInvalid([&] { validateFreeFormWindows(loop); }, "runtime parameter");
  }

  void windowIndexMustBeTheInductionVariable() {
    auto loop = windowLoop();
    loop.windows.front().index = "%k";
    requireInvalid([&] { validateFreeFormWindows(loop); }, "induction variable");
  }

  void windowCannotBeAStateDestination() {
    auto loop = windowLoop();
    loop.updates = {{"%tile", "%next_x"}, {"%acc_y", "%next_y"}};
    requireInvalid([&] { validateFreeFormWindows(loop); }, "cannot target");
  }

  void windowRangeMustFitTheSource() {
    auto loop = windowLoop();
    loop.windows.front().rows = 4;
    loop.windows.front().stride = 2;
    requireInvalid([&] { validateFreeFormWindows(loop); }, "fit the source extent");
  }

  void twoWindowsShareOneSource() {
    auto loop = windowLoop();
    loop.inputs.push_back(FreeFormInput{
        TensorRegionInput{"%tile_b", type({2, 2})}, FreeFormRole::Invariant,
        FreeFormInit::Window, ""});
    auto second = loop.windows.front();
    second.result = "%tile_b";
    loop.windows.push_back(second);
    validateFreeFormWindows(loop);
  }

  TensorRegion windowRegion() {
    TensorRegion region(
        {{"%acc_x", type({2, 2})}, {"%acc_y", type({2, 2})},
         {"%source", type({4, 2})}, {"%tile", type({2, 2})}}
    );
    region.add("%next_x", TensorOp::Add, {"%acc_x", "%tile"});
    region.add("%next_y", TensorOp::Add, {"%acc_y", "%acc_y"});
    region.setResults({"%next_x", "%next_y"});
    return region;
  }

  TensorRegion windowThreeStateRegion() {
    TensorRegion region(
        {{"%acc_x", type({2, 2})}, {"%acc_y", type({2, 2})}, {"%acc_z", type({2, 2})},
         {"%source", type({4, 2})}, {"%tile", type({2, 2})}}
    );
    region.add("%s01", TensorOp::Add, {"%acc_x", "%tile"});
    region.add("%next_x", TensorOp::Add, {"%s01", "%acc_x"});
    region.add("%next_y", TensorOp::Add, {"%s01", "%acc_y"});
    region.add("%s23", TensorOp::Add, {"%s01", "%acc_z"});
    region.add("%next_z", TensorOp::Add, {"%s23", "%acc_z"});
    region.setResults({"%next_x", "%next_y", "%next_z"});
    return region;
  }

  FreeFormLoop windowThreeStateLoop() {
    FreeFormLoop loop;
    loop.inputs = {
        state("%acc_x", {2, 2}, "%x"),
        state("%acc_y", {2, 2}, "%y"),
        zeroState("%acc_z", {2, 2}),
        input("%source", type({4, 2}), FreeFormRole::Invariant, FreeFormInit::RuntimeInput,
              "%src"),
        input("%tile", type({2, 2}), FreeFormRole::Invariant, FreeFormInit::Window),
    };
    loop.results = {"%next_x", "%next_y", "%next_z"};
    loop.updates = {
        {"%acc_x", "%next_x"}, {"%acc_y", "%next_y"}, {"%acc_z", "%next_z"}};
    loop.finalTerms = {"%acc_x", "%acc_y", "%acc_z"};
    FreeFormWindow window;
    window.result = "%tile";
    window.source = "%src";
    window.fillKind = FreeFormFillKind::Float;
    window.fillFloat = 0.0;
    window.rows = 2;
    window.stride = 1;
    window.active = 4;
    loop.windows = {window};
    return loop;
  }

  std::string assembledWindowSource() {
    const auto source = assembleCountedLoop(windowRegion(), windowLoop());
    require(
        source.find("tensor window %src, %i, 0.0, [2, 1, 4];") != std::string::npos,
        "the assembled source must produce the window each trip:\n" + source
    );
    return source;
  }

  void windowProducedBeforeRegionBody() {
    const auto source = assembledWindowSource();
    const auto window = source.find("  %tile = tensor window");
    const auto firstResult = source.find("  %next_x = tensor");
    require(
        window != std::string::npos && firstResult != std::string::npos && window < firstResult,
        "the window must be produced before the region body:\n" + source
    );
  }

  void windowRecomputedEachTrip() {
    const auto source = assembledWindowSource();
    const auto body = source.find("^body:\n");
    const auto window = source.find("  %tile = tensor window");
    const auto increment = source.find("  %i = %i + 1;");
    require(
        body != std::string::npos && window != std::string::npos && increment != std::string::npos &&
            body < window && window < increment,
        "the window must be recomputed inside the loop body:\n" + source
    );
    require(
        source.find("tensor window") > body,
        "no window production may appear before the loop body:\n" + source
    );
  }

  void windowDoesNotDisturbSnapshots() {
    auto plain = windowLoop();
    plain.windows.clear();
    plain.inputs[3].init = FreeFormInit::Zero;
    const auto withWindow = assembleCountedLoop(windowRegion(), windowLoop());
    const auto withoutWindow = assembleCountedLoop(windowRegion(), plain);
    auto bodyOf = [](const std::string &source) {
      const auto start = source.find("^body:\n");
      const auto end = source.find("  %i = %i + 1;");
      require(start != std::string::npos && end != std::string::npos, "a body must exist");
      return source.substr(start, end - start);
    };
    auto stripWindow = [](std::string body) {
      std::string kept;
      std::istringstream stream(body);
      std::string line;
      while (std::getline(stream, line)) {
        if (line.find("tensor window") == std::string::npos)
          kept += line + "\n";
      }
      return kept;
    };
    require(
        stripWindow(bodyOf(withWindow)) == bodyOf(withoutWindow),
        "the window must not change the body's update order"
    );
  }

  void windowThreeStateSource() {
    const auto source = assembleCountedLoop(windowThreeStateRegion(), windowThreeStateLoop());
    require(
        source.find("tensor window %src, %i, 0.0, [2, 1, 4];") != std::string::npos,
        "the three-state window must be produced each trip:\n" + source
    );
    std::cout << source;
  }

  FreeFormWindow witnessSpec(std::uint32_t rows, std::uint32_t stride, std::uint32_t active) {
    FreeFormWindow window;
    window.result = "%tile";
    window.source = "%src";
    window.rows = rows;
    window.stride = stride;
    window.active = active;
    return window;
  }

  void windowWitnessEnumeratesComputedAddresses() {
    const auto witness = witnessWindow(witnessSpec(2, 1, 4), 2, 4);
    require(witness.computedMax == 2, "the highest address is (trips-1)*stride + rows-1");
    require(witness.perTripActive.size() == 2, "one active set per trip");
    require(witness.allActive, "every row of this window is active on every trip");
  }

  void windowWitnessReportsTheActiveSetPerTrip() {
    const auto witness = witnessWindow(witnessSpec(2, 2, 3), 2, 4);
    require(witness.perTripActive[0][0] && witness.perTripActive[0][1], "trip zero is active");
    require(witness.perTripActive[1][0] && !witness.perTripActive[1][1], "trip one stays active");
    require(!witness.allActive, "the window is a tail");
  }

  void windowWitnessRefusesActiveBeyondTheSource() {
    requireInvalid(
        [] { (void) witnessWindow(witnessSpec(2, 1, 5), 2, 4); }, "fit the source extent"
    );
  }

  void windowWitnessMaxAddressIsInsideTheSource() {
    require(
        witnessWindow(witnessSpec(2, 2, 4), 2, 4).computedMax == 3,
        "the maximum address is the last row of the last trip"
    );
    requireInvalid(
        [] { (void) witnessWindow(witnessSpec(2, 2, 4), 3, 4); },
        "maximum address reaches the source extent"
    );
  }

  void windowWitnessStrideTwoRowsTwoActiveThreeIsTail() {
    const auto witness = witnessWindow(witnessSpec(2, 2, 3), 2, 4);
    require(!witness.allActive, "stride two rows two active three is a tail");
  }

  void windowWitnessJson() {
    const auto witness = witnessWindow(witnessSpec(2, 1, 4), 2, 4);
    std::cout << "{\"computed_max\":" << witness.computedMax << ",\"all_active\":"
              << (witness.allActive ? "true" : "false") << ",\"per_trip_active\":[";
    for (std::size_t trip = 0; trip < witness.perTripActive.size(); ++trip) {
      if (trip)
        std::cout << ',';
      std::cout << '[';
      for (std::size_t row = 0; row < witness.perTripActive[trip].size(); ++row) {
        if (row)
          std::cout << ',';
        std::cout << (witness.perTripActive[trip][row] ? "true" : "false");
      }
      std::cout << ']';
    }
    std::cout << "]}\n";
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
    else if (name == "policy-unguided-ignores-rank")
      unguidedIgnoresTheRank();
    else if (name == "policy-depth-changes-choice")
      depthBiasChangesTheChoice();
    else if (name == "policy-sharing-selects-a-consumer")
      sharingBiasSelectsASecondConsumer();
    else if (name == "policy-reduction-selects-a-reduction")
      reductionBiasSelectsAReduction();
    else if (name == "replay-vs-distribution")
      replayIsSeparateFromDistribution();
    else if (name == "features-off-keep-the-core")
      featuresOffKeepTheCore();
    else if (name == "features-enable-transpose-and-expand")
      featuresEnableTransposeAndExpand();
    else if (name == "features-respect-the-rank-limit")
      featuresRespectTheRankLimit();
    else if (name == "features-replay-exactly")
      featuresReplayExactly();
    else if (name == "extrema-binary-types")
      extremaBinaryTypes();
    else if (name == "extrema-reduce-drop-shape")
      extremaReduceDropShape();
    else if (name == "extrema-reduce-keep-shape")
      extremaReduceKeepShape();
    else if (name == "extrema-reject-axis")
      extremaRejectAxis();
    else if (name == "extrema-render-round-trip")
      extremaRenderRoundTrip();
    else if (name == "extrema-rank-one-keep-is-rank-one")
      extremaRankOneKeep();
    else if (name == "extrema-action-enumerated")
      extremaActionEnumerated();
    else if (name == "extrema-action-weighted")
      extremaActionWeighted();
    else if (name == "extrema-counted-as-reduction")
      extremaCountedAsReduction();
    else if (name == "extrema-multiplication-budget-untouched")
      extremaMultiplicationBudgetUntouched();
    else if (name == "extrema-replay-deterministic")
      extremaReplayDeterministic();
    else if (name == "extrema-rank-one-drop-is-scalar")
      extremaRankOneDropIsScalar();
    else if (name == "window-facts-interior")
      windowFactsInterior();
    else if (name == "window-facts-stride-zero-tail")
      windowFactsStrideZeroTail();
    else if (name == "window-facts-stride-zero-rows-one")
      windowFactsStrideZeroRowsOne();
    else if (name == "window-facts-constant-partial")
      windowFactsConstantPartial();
    else if (name == "window-facts-padded-v3")
      windowFactsPaddedV3();
    else if (name == "window-facts-mask-not-observed")
      windowFactsMaskNotObserved();
    else if (name == "window-init-requires-declared-invariant")
      windowInitRequiresDeclaredInvariant();
    else if (name == "window-source-must-be-a-parameter")
      windowSourceMustBeAParameter();
    else if (name == "window-index-must-be-the-induction-variable")
      windowIndexMustBeTheInductionVariable();
    else if (name == "window-cannot-be-a-state-destination")
      windowCannotBeAStateDestination();
    else if (name == "window-range-must-fit-the-source")
      windowRangeMustFitTheSource();
    else if (name == "two-windows-share-one-source")
      twoWindowsShareOneSource();
    else if (name == "window-produced-before-region-body")
      windowProducedBeforeRegionBody();
    else if (name == "window-recomputed-each-trip")
      windowRecomputedEachTrip();
    else if (name == "window-does-not-disturb-snapshots")
      windowDoesNotDisturbSnapshots();
    else if (name == "window-assembled-source") {
      std::cout << assembledWindowSource();
    } else if (name == "window-three-state-assembled-source")
      windowThreeStateSource();
    else if (name == "window-witness-enumerates-computed-addresses")
      windowWitnessEnumeratesComputedAddresses();
    else if (name == "window-witness-reports-active-set-per-trip")
      windowWitnessReportsTheActiveSetPerTrip();
    else if (name == "window-witness-refuses-active-beyond-source")
      windowWitnessRefusesActiveBeyondTheSource();
    else if (name == "window-witness-max-address-inside-source")
      windowWitnessMaxAddressIsInsideTheSource();
    else if (name == "window-witness-stride-two-rows-two-active-three-is-tail")
      windowWitnessStrideTwoRowsTwoActiveThreeIsTail();
    else if (name == "window-witness-json")
      windowWitnessJson();
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
