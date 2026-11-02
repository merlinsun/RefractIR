#include <algorithm>
#include <cmath>
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
  void requireExhaustionReason(Action action, const std::string &needle) {
    try {
      action();
    } catch (const std::runtime_error &error) {
      const std::string message = error.what();
      require(
          message.starts_with("free_form_exhaustion:"),
          "exhaustion must name its own prefix: " + message
      );
      require(
          message.find(needle) != std::string::npos,
          "exhaustion must name '" + needle + "': " + message
      );
      return;
    }
    throw std::runtime_error("expected a free-form request to refuse with " + needle);
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

  FreeFormEndpoint literalEnd(std::int32_t value) {
    FreeFormEndpoint endpoint;
    endpoint.value = value;
    return endpoint;
  }

  FreeFormEndpoint parameterEnd(std::string name, std::int32_t low, std::int32_t high) {
    FreeFormEndpoint endpoint;
    endpoint.parameter = std::move(name);
    endpoint.low = low;
    endpoint.high = high;
    return endpoint;
  }

  /// The window loop over `range(%lo, %hi, -1)`: rows may be read in reverse
  /// and from below zero, where the window reads its fill.
  FreeFormLoop rangeWindowLoop() {
    auto loop = windowLoop();
    loop.range = FreeFormRange{parameterEnd("%lo", -2, 4), parameterEnd("%hi", -3, 4), -1};
    return loop;
  }

  void rangeWindowSource() { std::cout << assembleCountedLoop(windowRegion(), rangeWindowLoop()); }

  /// Two runtime ranges, one nested in the other.
  void rangeNestedSource() {
    auto loop = threeStateLoop();
    loop.range = FreeFormRange{literalEnd(0), parameterEnd("%n", 0, 6), 2};
    loop.outerRange = FreeFormRange{parameterEnd("%m", 0, 3), literalEnd(-1), -1};
    std::cout << assembleCountedLoop(threeStateRegion(), loop);
  }

  void rangeRefusesATripParameter() {
    auto loop = rangeWindowLoop();
    loop.tripParameter = "%trips";
    requireInvalid([&] { (void) assembleCountedLoop(windowRegion(), loop); }, "trip parameter");
  }

  void rangeRefusesAZeroStep() {
    auto loop = rangeWindowLoop();
    loop.range->step = 0;
    requireInvalid([&] { (void) assembleCountedLoop(windowRegion(), loop); }, "step");
  }

  void rangeRefusesAnEmptyParameterRange() {
    auto loop = rangeWindowLoop();
    loop.range->start = parameterEnd("%lo", 3, 2);
    requireInvalid([&] { (void) assembleCountedLoop(windowRegion(), loop); }, "range");
  }

  void rangeWindowMayReachPastTheSource() {
    // Every window position outside [0, active) reads the fill, so a range
    // loop does not need the counted loop's address bound.
    auto loop = rangeWindowLoop();
    loop.windows.front().stride = 3;
    validateFreeFormWindows(loop);
    auto counted = windowLoop();
    counted.windows.front().stride = 3;
    requireInvalid([&] { validateFreeFormWindows(counted); }, "fit the source extent");
  }

  /// `2 * max(where(2^-10 < x, x, 0), 0)`, one trip from `%x`.
  void literalRegionSource() {
    TensorRegion region({{"%acc_x", type({4})}});
    region.addLiteral("%t", TensorElement::F32, std::ldexp(1.0, -10));
    region.addLiteral("%zero", TensorElement::F32, 0.0);
    region.addLiteral("%two", TensorElement::F32, 2.0);
    region.add("%w", TensorOp::WhereLt, {"%t", "%acc_x", "%acc_x", "%zero"});
    region.add("%m", TensorOp::Max, {"%w", "%zero"});
    region.add("%next_x", TensorOp::Mul, {"%m", "%two"});
    region.setResults({"%next_x"});
    FreeFormLoop loop;
    loop.inputs = {state("%acc_x", {4}, "%x")};
    loop.results = {"%next_x"};
    loop.updates = {{"%acc_x", "%next_x"}};
    loop.tripCount = 1;
    loop.finalTerms = {"%acc_x"};
    std::cout << assembleCountedLoop(region, loop);
  }

  void literalMagnitudeBound() {
    TensorRegion region({{"%x", type({4})}});
    region.addLiteral("%three", TensorElement::F32, 3.0);
    region.addLiteral("%minus", TensorElement::F32, -1.0);
    region.add("%p", TensorOp::Mul, {"%x", "%three"});
    region.add("%q", TensorOp::Add, {"%p", "%minus"});
    region.setResults({"%q"});
    const auto bounds = region.magnitudeBounds({{"%x", 2.0}});
    require(bounds.at("%p") == 6.0, "a literal factor scales the bound by its magnitude");
    require(bounds.at("%q") == 7.0, "a literal term adds its magnitude");
    requireInvalid(
        [] {
          TensorRegion integer({{"%x", type({4}, TensorElement::I32)}});
          integer.addLiteral("%half", TensorElement::I32, 0.5);
        },
        "literal"
    );
  }

  /// `rem(exp(x / y), y)`, one trip, with `%y` a read-only input.
  void approximateRegionSource() {
    TensorRegion region({{"%acc_x", type({4})}, {"%acc_y", type({4})}});
    region.add("%d", TensorOp::Div, {"%acc_x", "%acc_y"});
    region.add("%e", TensorOp::Exp, {"%d"});
    region.add("%next_x", TensorOp::Rem, {"%e", "%acc_y"});
    region.setResults({"%next_x"});
    FreeFormLoop loop;
    loop.inputs = {
        state("%acc_x", {4}, "%x"),
        input("%acc_y", type({4}), FreeFormRole::Invariant, FreeFormInit::RuntimeInput, "%y"),
    };
    loop.results = {"%next_x"};
    loop.updates = {{"%acc_x", "%next_x"}};
    loop.tripCount = 1;
    loop.finalTerms = {"%acc_x"};
    std::cout << assembleCountedLoop(region, loop);
  }

  void approximateMagnitudeUnbounded() {
    TensorRegion region({{"%x", type({4})}, {"%y", type({4})}});
    region.add("%d", TensorOp::Div, {"%x", "%y"});
    region.add("%e", TensorOp::Exp, {"%x"});
    region.add("%r", TensorOp::Rem, {"%x", "%y"});
    region.setResults({"%d", "%e", "%r"});
    const auto bounds = region.magnitudeBounds({{"%x", 2.0}, {"%y", 1.0}});
    require(std::isinf(bounds.at("%d")), "a quotient has no magnitude bound");
    require(std::isinf(bounds.at("%e")), "an exponential has no magnitude bound");
    require(bounds.at("%r") == 2.0, "a remainder is bounded by its dividend");
  }

  FreeFormRequest pointwiseRequest(std::uint32_t seed) {
    FreeFormRequest request;
    request.inputs = {state("%a", {8, 4}, "%a_in"), state("%b", {8, 4}, "%b_in")};
    request.results = {{"%next_a", type({8, 4})}, {"%next_b", type({8, 4})}};
    request.policy = FreeFormPolicy::Pointwise;
    request.parallelAxis = 0;
    request.features.transpose = true;
    request.features.extrema = true;
    request.features.where = true;
    request.features.narrow = true;
    request.features.scan = true;
    request.features.literals = true;
    request.seed = seed;
    return request;
  }

  void pointwiseKeepsTheParallelAxis() {
    std::uint32_t mixingAvailable = 0;
    for (std::uint32_t seed = 0; seed < 32; ++seed) {
      const auto built = buildFreeFormRegion(pointwiseRequest(seed));
      require(
          built.region.keepsAxisParallel(0),
          "seed " + std::to_string(seed) + " mixed its parallel axis"
      );
      auto unconstrained = pointwiseRequest(seed);
      unconstrained.parallelAxis.reset();
      if (!buildFreeFormRegion(unconstrained).region.keepsAxisParallel(0))
        ++mixingAvailable;
    }
    require(mixingAvailable > 0, "without the requirement some seed should mix axis 0");
  }

  void pointwisePolicyWeights() {
    FreeFormActionTraits add;
    add.op = TensorOp::Add;
    FreeFormActionTraits narrow;
    narrow.op = TensorOp::NarrowF16;
    FreeFormActionTraits sum;
    sum.op = TensorOp::Sum;
    sum.shapeChange = true;
    const auto weight = [](const FreeFormActionTraits &traits) {
      return freeFormActionWeight(FreeFormPolicy::Pointwise, traits);
    };
    require(weight(narrow) > weight(add), "narrowing is favoured over other lane operations");
    require(weight(add) > weight(sum), "lane operations are favoured over reductions");
    require(weight(sum) > 0.0, "every weight stays positive");
    require(parseFreeFormPolicy("pointwise") == FreeFormPolicy::Pointwise, "policy parses");
  }

  void parallelAxisRefusesAMixingRequirement() {
    auto request = pointwiseRequest(1);
    FreeFormRequirement requirement;
    requirement.input = "%a";
    requirement.results = {0};
    requirement.pathOps = {TensorOp::Transpose};
    request.requirements = {requirement};
    requireInvalid([&] { (void) buildFreeFormRegion(request); }, "transpose");
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

  FreeFormLoop windowBufferLoop() {
    FreeFormLoop loop;
    loop.inputs = {
        state("%acc_x", {4, 4}, "%x"),
        state("%acc_y", {4, 4}, "%y"),
        input("%tile", type({4, 4}), FreeFormRole::Invariant, FreeFormInit::Window),
    };
    loop.windowSources = {{"%window_buffer", type({20, 4})}};
    loop.results = {"%next_x", "%next_y"};
    loop.updates = {{"%acc_x", "%next_x"}, {"%acc_y", "%next_y"}};
    loop.finalTerms = {"%acc_x", "%acc_y"};
    loop.tripCount = 8;
    FreeFormWindow window;
    window.result = "%tile";
    window.source = "%window_buffer";
    window.fillKind = FreeFormFillKind::Float;
    window.fillFloat = 0.0;
    window.rows = 4;
    window.stride = 2;
    window.active = 18;
    loop.windows = {window};
    return loop;
  }

  TensorRegion windowBufferRegion() {
    TensorRegion region(
        {{"%acc_x", type({4, 4})}, {"%acc_y", type({4, 4})}, {"%tile", type({4, 4})}}
    );
    region.add("%next_x", TensorOp::Add, {"%acc_x", "%tile"});
    region.add("%next_y", TensorOp::Add, {"%acc_y", "%tile"});
    region.setResults({"%next_x", "%next_y"});
    return region;
  }

  void windowBufferSuppliesItsOwnSource() {
    const auto source = assembleCountedLoop(windowBufferRegion(), windowBufferLoop());
    require(
        source.find("fun @kernel(%window_buffer: tensor [20, 4] f32, %x: tensor [4, 4] f32, "
                    "%y: tensor [4, 4] f32)") != std::string::npos,
        "the backing parameter must be a function parameter before the runtime inputs:\n" + source
    );
    require(
        source.find("tensor window %window_buffer, %i, 0.0, [4, 2, 18];") != std::string::npos,
        "the window must read the backing parameter:\n" + source
    );
    require(
        source.find("let mut %window_buffer") == std::string::npos,
        "the backing parameter must not be a local\n"
    );
    require(
        source.find("%window_buffer = ") == std::string::npos,
        "the backing parameter must not be copied into a local\n"
    );
  }

  void windowBufferStaysOutOfTheValuePool() {
    auto region = windowBufferRegion();
    bool absent = false;
    try {
      (void) region.typeOf("%window_buffer");
    } catch (const std::invalid_argument &) {
      absent = true;
    }
    require(absent, "the backing parameter must not be a region value");
    const auto loop = windowBufferLoop();
    for (const auto &input: loop.inputs)
      require(
          input.value.name != "%window_buffer",
          "the backing parameter must not be a region input"
      );
    validateFreeFormWindows(loop);
    const auto source = assembleCountedLoop(region, loop);
    std::size_t occurrences = 0;
    for (std::size_t at = source.find("%window_buffer"); at != std::string::npos;
         at = source.find("%window_buffer", at + 1))
      ++occurrences;
    require(
        occurrences == 2,
        "the backing parameter appears only in the signature and the window line:\n" + source
    );
  }

  void windowBufferWitnessUsesItsOwnExtent() {
    auto loop = windowBufferLoop();
    loop.windowSources.front().type = type({8, 4});
    loop.tripCount = 4;
    loop.windows.front().active = 8;
    requireInvalid(
        [&] { validateFreeFormWindows(loop); }, "fit the source extent"
    );
    bool witnessRefused = false;
    try {
      (void) witnessWindow(loop.windows.front(), loop.tripCount, 8);
    } catch (const std::invalid_argument &error) {
      witnessRefused = std::string(error.what()).find("maximum address") != std::string::npos;
    }
    require(witnessRefused, "the witness must refuse a maximum address outside the backing");
    validateFreeFormWindows(windowBufferLoop());
  }

  void windowBufferNameCollisionIsRefused() {
    for (const std::string &name: {"%acc_x", "%x", "%tile", "%i", "%out"}) {
      auto loop = windowBufferLoop();
      loop.windowSources.front().parameter = name;
      loop.windows.front().source = name;
      requireInvalid([&] { validateFreeFormWindows(loop); }, "window source");
    }
    auto loop = windowBufferLoop();
    loop.windowSources.front().parameter = "%window_buffer";
    TensorRegion region(
        {{"%acc_x", type({4, 4})}, {"%acc_y", type({4, 4})}, {"%tile", type({4, 4})}}
    );
    region.add("%window_buffer", TensorOp::Add, {"%acc_x", "%acc_y"});
    region.add("%next_x", TensorOp::Add, {"%window_buffer", "%tile"});
    region.add("%next_y", TensorOp::Add, {"%acc_y", "%tile"});
    region.setResults({"%next_x", "%next_y"});
    requireInvalid(
        [&] { (void) assembleCountedLoop(region, loop); }, "region local"
    );
  }

  FreeFormGraphView graphKeysView() {
    FreeFormGraphView view;
    view.valueNames = {"%a", "%b", "%n0", "%n1"};
    view.valueTypes = {type({2, 2}), type({2, 2}), type({2, 2}), type({2, 2})};
    view.inputCount = 2;
    view.nodeOps = {TensorOp::Add, TensorOp::Transpose};
    view.nodeOperands = {{0, 1}, {2}};
    view.nodeAxes = {0, 0};
    view.nodeKeepDims = {0, 0};
    view.nodeAttributes = {{}, {0, 1}};
    view.roots = {3, 2};
    return view;
  }

  void computedRootRefusesAnInput() {
    FreeFormRequest request;
    request.inputs = {input("%a", type({2, 2}), FreeFormRole::State)};
    request.results = {{"%r0", type({2, 2}), true}};
    request.limits.maxNodes = 1;
    const auto built = buildFreeFormRegion(request);
    require(
        built.inputSpec.results.size() == 1 && built.inputSpec.results[0] != "%a",
        "a computed-root result must not keep the input name"
    );
    require(
        built.region.typeOf(built.inputSpec.results[0]).shape == Shape({2, 2}),
        "the computed root keeps the requested type"
    );

    auto control = request;
    control.results = {{"%r0", type({2, 2}), false}};
    const auto admitted = buildFreeFormRegion(control);
    require(
        admitted.inputSpec.results == std::vector<std::string>{"%a"},
        "without the requirement an input root is still legal"
    );
  }

  void computedRootsAreBothProduced() {
    FreeFormRequest request;
    request.inputs = {input("%a", type({2, 2}), FreeFormRole::State),
                      input("%b", type({2, 2}), FreeFormRole::State)};
    request.results = {{"%r0", type({2, 2}), true}, {"%r1", type({2, 2}), true}};
    request.limits.maxNodes = 4;
    request.seed = 3;
    const auto built = buildFreeFormRegion(request);
    for (const auto &name: built.inputSpec.results)
      require(name != "%a" && name != "%b", "both roots must be computed");
    require(
        built.structure.nodes >= 2,
        "both computed roots need producers"
    );
  }

  void computedRootsRespectTheBudget() {
    FreeFormRequest request;
    request.inputs = {input("%a", type({2, 2}), FreeFormRole::State)};
    request.results = {{"%r0", type({2, 2}), true}, {"%r1", type({2, 2}), true},
                       {"%r2", type({2, 2}), true}, {"%r3", type({2, 2}), true}};
    request.limits.maxNodes = 3;
    request.limits.maxAttempts = 2;
    requireExhaustion([&] { (void) buildFreeFormRegion(request); });
  }

  void graphKeysIgnoreNamesAndDeclarationOrder() {
    auto first = graphKeysView();
    auto renamed = first;
    renamed.valueNames = {"%x", "%y", "%p", "%q"};
    const auto one = freeFormGraphKeys(first, {"state", "state"});
    const auto two = freeFormGraphKeys(renamed, {"state", "state"});
    require(
        one.connectivitySignature == two.connectivitySignature,
        "renaming values must not change connectivity"
    );
    require(one.typedFingerprint != "", "the typed fingerprint must be stated");

    // Same DAG, producer declarations listed in the other order.
    FreeFormGraphView reordered;
    reordered.valueNames = {"%a", "%b", "%p", "%q"};
    reordered.valueTypes = {type({2, 2}), type({2, 2}), type({2, 2}), type({2, 2})};
    reordered.inputCount = 2;
    reordered.nodeOps = {TensorOp::Transpose, TensorOp::Add};
    reordered.nodeOperands = {{3}, {0, 1}};
    reordered.nodeAxes = {0, 0};
    reordered.nodeKeepDims = {0, 0};
    reordered.nodeAttributes = {{0, 1}, {}};
    reordered.roots = {2, 3};
    const auto three = freeFormGraphKeys(reordered, {"state", "state"});
    require(
        one.connectivitySignature == three.connectivitySignature,
        "creation order must not change connectivity"
    );
  }

  void graphKeysIncludeInputTypes() {
    FreeFormGraphView narrow;
    narrow.valueNames = {"%a", "%b", "%r"};
    narrow.valueTypes = {type({2, 2}), type({2, 2}), type({2})};
    narrow.inputCount = 2;
    narrow.nodeOps = {TensorOp::Sum};
    narrow.nodeOperands = {{0}};
    narrow.nodeAxes = {0};
    narrow.nodeKeepDims = {0};
    narrow.nodeAttributes = {{}};
    narrow.roots = {2};
    auto wider = narrow;
    wider.valueTypes[0] = type({3, 2});
    const auto one = freeFormGraphKeys(narrow, {"state", "state"});
    const auto two = freeFormGraphKeys(wider, {"state", "state"});
    require(
        one.connectivitySignature == two.connectivitySignature,
        "input extents stay excluded from connectivity"
    );
    require(
        one.typedFingerprint != two.typedFingerprint,
        "input extents must be part of the typed fingerprint"
    );
    auto renamed = narrow;
    renamed.valueNames = {"%x", "%y", "%z"};
    require(
        one.typedFingerprint == freeFormGraphKeys(renamed, {"state", "state"}).typedFingerprint,
        "renaming values must not change the typed fingerprint"
    );
  }

  void graphKeysFollowRootsEdgesAndRoles() {
    const auto base = freeFormGraphKeys(graphKeysView(), {"state", "state"});
    auto changedRoot = graphKeysView();
    changedRoot.roots = {2, 3};
    require(
        base.connectivitySignature != freeFormGraphKeys(changedRoot, {"state", "state"}).connectivitySignature,
        "a changed root must change connectivity"
    );
    auto changedEdge = graphKeysView();
    changedEdge.nodeOperands[0] = {0, 0};
    require(
        base.connectivitySignature != freeFormGraphKeys(changedEdge, {"state", "state"}).connectivitySignature,
        "a changed operand connection must change connectivity"
    );
    require(
        base.connectivitySignature != freeFormGraphKeys(graphKeysView(), {"state", "invariant"}).connectivitySignature,
        "a changed input role must change connectivity"
    );
    auto wider = graphKeysView();
    wider.valueTypes[0] = type({2, 2});
    wider.valueTypes[1] = type({2, 2});
    wider.valueTypes[2] = type({4, 2});
    wider.valueTypes[3] = type({4, 2});
    const auto typed = freeFormGraphKeys(wider, {"state", "state"});
    require(
        base.connectivitySignature == typed.connectivitySignature,
        "an extent-only change must keep connectivity"
    );
    require(
        base.typedFingerprint != typed.typedFingerprint,
        "an extent-only change must change the typed fingerprint"
    );
  }

  FreeFormRequest requirementRequest() {
    FreeFormRequest request;
    request.inputs = {
        input("%a", type({4, 4}), FreeFormRole::State),
        input("%b", type({4, 4}), FreeFormRole::State),
    };
    request.results = {{"%r0", type({4, 4})}, {"%r1", type({4, 4})}};
    request.features.extrema = true;
    request.features.transpose = true;
    request.features.expand = true;
    request.limits.maxNodes = 16;
    request.seed = 5;
    return request;
  }

  void requirementKeepsTheInputInTheResult() {
    auto request = requirementRequest();
    request.requirements = {{"%a", {0}, {}, {}, false}};
    const auto built = buildFreeFormRegion(request);
    require(
        built.structure.requirementWitnesses.size() == 1,
        "the satisfied requirement must emit one witness"
    );
    require(
        (built.structure.resultDependencies[0] & 1ULL) != 0,
        "result zero must depend on the required input"
    );
    const auto &witness = built.structure.requirementWitnesses.front();
    require(!witness.path.empty(), "the witness must state a source-to-result path");
    require(witness.path.front() == "%a", "the witness path must start at the input");
    require(witness.results.size() == 1, "the witness must name the required result");
  }

  void requirementBuildsALiveReduction() {
    auto request = requirementRequest();
    request.requirements = {{
        "%a", {0}, {TensorOp::Sum, TensorOp::ReduceMin, TensorOp::ReduceMax}, {}, false
    }};
    const auto built = buildFreeFormRegion(request);
    require(
        built.structure.requirementWitnesses.size() == 1,
        "the satisfied requirement must emit one witness"
    );
        const auto &witness = built.structure.requirementWitnesses.front();
    bool sawReduction = false;
    for (const auto &operation: witness.operations)
      sawReduction = sawReduction || operation == "sum" || operation == "rmin" ||
                     operation == "rmax";
    require(sawReduction, "the witnessed path must contain the required reduction");
    require(
        witness.operations.size() == witness.operationIds.size(),
        "every witnessed operation needs one live ID"
    );
    require(
        built.structure.nodes >= witness.operationIds.size(),
        "the witnessed operations must be inside the live graph"
    );
  }

  void requirementCountsTwoDistinctConsumers() {
    auto request = requirementRequest();
    request.requirements = {{"%a", {0, 1}, {}, {}, true}};
    const auto built = buildFreeFormRegion(request);
    require(
        built.structure.requirementWitnesses.size() == 1,
        "the satisfied requirement must emit one witness"
    );
        const auto &witness = built.structure.requirementWitnesses.front();
    require(witness.distinctConsumersSatisfied, "two distinct consumers must be witnessed");
    require(witness.consumers.size() == 2, "the witness must name the two consumers");
    require(witness.consumers[0] != witness.consumers[1], "the consumers must be distinct");
    require(
        witness.consumerResults.size() == 2 &&
            witness.consumerResults[0] != witness.consumerResults[1],
        "the consumers must reach different result roots"
    );

    auto single = requirementRequest();
    single.limits.maxNodes = 3;
    single.requirements = {{"%a", {0, 1}, {}, {}, true}};
    single.seed = 1;
    requireExhaustionReason(
        [&] { (void) buildFreeFormRegion(single); }, "distinct_consumers_missing"
    );
  }

  void requirementSharesOneExtremaShapePath() {
    auto request = requirementRequest();
    request.requirements = {{
        "%a", {0}, {TensorOp::ReduceMin, TensorOp::ReduceMax}, {TensorOp::Transpose}, false
    }};
    const auto built = buildFreeFormRegion(request);
    require(
        built.structure.requirementWitnesses.size() == 1,
        "the satisfied requirement must emit one witness"
    );
        const auto &witness = built.structure.requirementWitnesses.front();
    bool sawExtrema = false;
    bool sawShape = false;
    for (const auto &operation: witness.operations) {
      sawExtrema = sawExtrema || operation == "rmin" || operation == "rmax";
      sawShape = sawShape || operation == "transpose";
    }
    require(sawExtrema && sawShape, "one path must carry both the extrema and the shape op");

    auto missing = requirementRequest();
    missing.features.transpose = false;
    missing.requirements = {{
        "%a", {0}, {TensorOp::ReduceMin, TensorOp::ReduceMax}, {TensorOp::Transpose}, false
    }};
    missing.limits.maxAttempts = 8;
    requireExhaustionReason(
        [&] { (void) buildFreeFormRegion(missing); }, "required_path_missing"
    );
  }

  void requirementReservesBridgeBudget() {
    auto request = requirementRequest();
    request.limits.maxNodes = 4;
    request.requirements = {{"%a", {0}, {TensorOp::Sum}, {}, false}};
    requireExhaustionReason(
        [&] { (void) buildFreeFormRegion(request); }, "node_budget_exhausted"
    );

    auto feasible = requirementRequest();
    feasible.requirements = {{
        "%a", {0}, {TensorOp::ReduceMin, TensorOp::ReduceMax}, {TensorOp::Transpose}, false
    }};
    const auto built = buildFreeFormRegion(feasible);
    require(
        built.structure.liveDag.roots.size() == 2,
        "the live DAG must state both roots"
    );
    require(
        built.structure.liveDag.values.size() >= 2,
        "the live DAG must state its values"
    );
  }

  void requirementRejectsUnknownNames() {
    auto request = requirementRequest();
    request.requirements = {{"%missing", {0}, {}, {}, false}};
    requireInvalid([&] { (void) buildFreeFormRegion(request); }, "requirement input");
    auto outOfRange = requirementRequest();
    outOfRange.requirements = {{"%a", {5}, {}, {}, false}};
    requireInvalid([&] { (void) buildFreeFormRegion(outOfRange); }, "requirement result");
    auto oneRoot = requirementRequest();
    oneRoot.requirements = {{"%a", {0}, {}, {}, true}};
    requireInvalid([&] { (void) buildFreeFormRegion(oneRoot); }, "two results");
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

  void requireBound(double actual, double expected, const std::string &what) {
    require(
        actual == expected,
        what + ": bound " + std::to_string(actual) + ", expected " + std::to_string(expected)
    );
  }

  // Three additive states over two trips, with every runtime input bounded by
  // one: trip 1 gives 2/3/3/2/2, trip 2 gives 6/9/9/8/10, and the final sum 28.
  void magnitudeThreeStateTrips() {
    requireBound(
        freeFormMagnitudeBound(threeStateRegion(), threeStateLoop(), 1.0), 28.0,
        "three additive states over two trips"
    );
  }

  // A state multiplied by itself squares its bound each trip, so eight trips
  // leave the exact range even from a bound of two.
  void magnitudeProductCompounds() {
    TensorRegion region({{"%acc_x", type({2, 2})}});
    region.add("%next_x", TensorOp::Mul, {"%acc_x", "%acc_x"});
    region.setResults({"%next_x"});
    FreeFormLoop loop;
    loop.inputs = {state("%acc_x", {2, 2}, "%x")};
    loop.results = {"%next_x"};
    loop.updates = {{"%acc_x", "%next_x"}};
    loop.finalTerms = {"%acc_x"};
    loop.tripCount = 2;
    requireBound(freeFormMagnitudeBound(region, loop, 2.0), 16.0, "two squaring trips");
    loop.tripCount = 8;
    require(
        freeFormMagnitudeBound(region, loop, 2.0) > kExactF32IntegerBound,
        "eight squaring trips must leave the exact range"
    );
  }

  // A sum scales its bound by the reduced extent; an extrema reduction and a
  // broadcast keep it.
  void magnitudeReductionsAndShapes() {
    for (const auto op: {TensorOp::Sum, TensorOp::ReduceMax}) {
      TensorRegion region({{"%acc_x", type({2, 4})}});
      region.add("%reduced", op, {"%acc_x"}, 1, true);
      region.add("%wide", TensorOp::Broadcast, {"%reduced"}, 0, false, {2, 4});
      region.add("%next_x", TensorOp::Add, {"%acc_x", "%wide"});
      region.setResults({"%next_x"});
      FreeFormLoop loop;
      loop.inputs = {state("%acc_x", {2, 4}, "%x")};
      loop.results = {"%next_x"};
      loop.updates = {{"%acc_x", "%next_x"}};
      loop.finalTerms = {"%acc_x"};
      loop.tripCount = 1;
      requireBound(
          freeFormMagnitudeBound(region, loop, 1.0), op == TensorOp::Sum ? 5.0 : 2.0,
          op == TensorOp::Sum ? "a sum over four lanes" : "a maximum over four lanes"
      );
    }
  }

  // A window tile is bounded by the larger of the input bound and its fill.
  void magnitudeWindowFill() {
    TensorRegion region(
        {{"%acc_x", type({2, 2})},
         {"%acc_y", type({2, 2})},
         {"%source", type({4, 2})},
         {"%tile", type({2, 2})}}
    );
    region.add("%next_x", TensorOp::Add, {"%acc_x", "%tile"});
    region.add("%next_y", TensorOp::Add, {"%acc_y", "%tile"});
    region.setResults({"%next_x", "%next_y"});
    auto loop = windowLoop();
    loop.tripCount = 2;
    loop.windows.front().fillFloat = -9.0;
    // Trip 1: 1 + 9 and 0 + 9; trip 2: 10 + 9 and 9 + 9; the final sum is 37.
    requireBound(freeFormMagnitudeBound(region, loop, 1.0), 37.0, "a window fill of -9");
  }

  // Only `sum` results are order-sensitive; a dead sum is not listed.
  void orderSensitiveValuesAreLiveSums() {
    TensorRegion region({{"%acc_x", type({2, 4})}});
    region.add("%total", TensorOp::Sum, {"%acc_x"}, 1, true);
    region.add("%dead", TensorOp::Sum, {"%acc_x"}, 0, true);
    region.add("%top", TensorOp::ReduceMax, {"%acc_x"}, 1, true);
    region.add("%both", TensorOp::Add, {"%total", "%top"});
    region.add("%wide", TensorOp::Broadcast, {"%both"}, 0, false, {2, 4});
    region.add("%next_x", TensorOp::Add, {"%acc_x", "%wide"});
    region.setResults({"%next_x"});
    require(
        region.orderSensitiveValues() == std::vector<std::string>{"%total"},
        "only the live sum is order-sensitive"
    );
  }

  // The sum bound tracks `sum` results alone; a body with no sum has none.
  void magnitudesTrackSums() {
    TensorRegion summed({{"%acc_x", type({2, 4})}});
    summed.add("%reduced", TensorOp::Sum, {"%acc_x"}, 1, true);
    summed.add("%wide", TensorOp::Broadcast, {"%reduced"}, 0, false, {2, 4});
    summed.add("%next_x", TensorOp::Add, {"%acc_x", "%wide"});
    summed.setResults({"%next_x"});
    FreeFormLoop loop;
    loop.inputs = {state("%acc_x", {2, 4}, "%x")};
    loop.results = {"%next_x"};
    loop.updates = {{"%acc_x", "%next_x"}};
    loop.finalTerms = {"%acc_x"};
    loop.tripCount = 2;
    // Trip 1: sum 4, next 5; trip 2: sum 20, next 25.
    const auto magnitudes = freeFormMagnitudes(summed, loop, 1.0);
    requireBound(magnitudes.largestSum, 20.0, "the sum bound over two trips");
    requireBound(magnitudes.largest, 25.0, "the overall bound over two trips");

    TensorRegion squared({{"%acc_x", type({2, 4})}});
    squared.add("%next_x", TensorOp::Mul, {"%acc_x", "%acc_x"});
    squared.setResults({"%next_x"});
    requireBound(freeFormMagnitudes(squared, loop, 2.0).largestSum, 0.0, "no sum, no sum bound");
  }

  // The reduction contract admits large elementwise values that stay finite,
  // refuses a sum beyond 2^23, and refuses anything that could overflow.
  void reductionContractBounds() {
    using Contract = FreeFormExactContract;
    const FreeFormMagnitudes elementwise{4294967296.0, 0.0};
    require(
        !freeFormWithinExactContract(elementwise, Contract::AllValues) &&
            freeFormWithinExactContract(elementwise, Contract::Reductions),
        "2^32 elementwise is exact only where order does not matter"
    );
    const FreeFormMagnitudes bigSum{16.0 * kExactReductionBound, 2.0 * kExactReductionBound};
    require(
        !freeFormWithinExactContract(bigSum, Contract::Reductions), "a sum beyond 2^23 is refused"
    );
    const FreeFormMagnitudes overflowing{2.0 * kFiniteMarginBound, 0.0};
    require(
        !freeFormWithinExactContract(overflowing, Contract::Reductions),
        "a value beyond 2^64 is refused"
    );
    require(
        freeFormWithinExactContract(
            {kExactReductionBound, kExactReductionBound}, Contract::Reductions
        ),
        "the limits themselves are admitted"
    );
  }

  FreeFormRequest growthRequest(std::uint32_t seed, FreeFormRootSelection selection) {
    FreeFormRequest request;
    request.inputs = {state("%acc_x", {4, 4}, "%x"), state("%acc_y", {4, 4}, "%y")};
    request.results = {{"%next_x", type({4, 4})}, {"%next_y", type({4, 4})}};
    request.policy = FreeFormPolicy::Unguided;
    request.features = {true, true, true};
    request.limits.maxNodes = 16;
    request.seed = seed;
    request.rootSelection = selection;
    return request;
  }

  // Root selection happens after growth, so both selections prune the same
  // grown graph; keeping the most nodes alive must keep more of it overall.
  void mostLiveRootsKeepMoreNodes() {
    std::uint32_t deepest = 0;
    std::uint32_t mostLive = 0;
    for (std::uint32_t seed = 1; seed <= 40; ++seed) {
      deepest +=
          buildFreeFormRegion(growthRequest(seed, FreeFormRootSelection::Deepest)).structure.nodes;
      mostLive +=
          buildFreeFormRegion(growthRequest(seed, FreeFormRootSelection::MostLive)).structure.nodes;
    }
    require(
        mostLive > deepest, "most-live roots kept " + std::to_string(mostLive) +
                                " live nodes against " + std::to_string(deepest)
    );
  }

  // The default stays the deepest root, so the frozen corpus replays.
  void deepestRootsAreTheDefault() {
    for (std::uint32_t seed = 1; seed <= 12; ++seed) {
      auto unset = growthRequest(seed, FreeFormRootSelection::Deepest);
      unset.rootSelection = FreeFormRequest{}.rootSelection;
      require(
          render(buildFreeFormRegion(unset).region) ==
              render(
                  buildFreeFormRegion(growthRequest(seed, FreeFormRootSelection::Deepest)).region
              ),
          "an unset root selection must build the deepest-root body"
      );
    }
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
    else if (name == "pointwise-keeps-the-parallel-axis")
      pointwiseKeepsTheParallelAxis();
    else if (name == "pointwise-policy-weights")
      pointwisePolicyWeights();
    else if (name == "parallel-axis-refuses-a-mixing-requirement")
      parallelAxisRefusesAMixingRequirement();
    else if (name == "approximate-region-source")
      approximateRegionSource();
    else if (name == "approximate-magnitude-unbounded")
      approximateMagnitudeUnbounded();
    else if (name == "literal-region-source")
      literalRegionSource();
    else if (name == "literal-magnitude-bound")
      literalMagnitudeBound();
    else if (name == "range-window-source")
      rangeWindowSource();
    else if (name == "range-nested-source")
      rangeNestedSource();
    else if (name == "range-refuses-a-trip-parameter")
      rangeRefusesATripParameter();
    else if (name == "range-refuses-a-zero-step")
      rangeRefusesAZeroStep();
    else if (name == "range-refuses-an-empty-parameter-range")
      rangeRefusesAnEmptyParameterRange();
    else if (name == "range-window-may-reach-past-the-source")
      rangeWindowMayReachPastTheSource();
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
    else if (name == "computed-root-refuses-an-input")
      computedRootRefusesAnInput();
    else if (name == "computed-roots-are-both-produced")
      computedRootsAreBothProduced();
    else if (name == "computed-roots-respect-the-budget")
      computedRootsRespectTheBudget();
    else if (name == "graph-keys-include-input-types")
      graphKeysIncludeInputTypes();
    else if (name == "graph-keys-ignore-names-and-order")
      graphKeysIgnoreNamesAndDeclarationOrder();
    else if (name == "graph-keys-follow-roots-edges-roles")
      graphKeysFollowRootsEdgesAndRoles();
    else if (name == "requirement-keeps-the-input-in-the-result")
      requirementKeepsTheInputInTheResult();
    else if (name == "requirement-builds-a-live-reduction")
      requirementBuildsALiveReduction();
    else if (name == "requirement-counts-two-distinct-consumers")
      requirementCountsTwoDistinctConsumers();
    else if (name == "requirement-shares-one-extrema-shape-path")
      requirementSharesOneExtremaShapePath();
    else if (name == "requirement-reserves-bridge-budget")
      requirementReservesBridgeBudget();
    else if (name == "requirement-rejects-unknown-names")
      requirementRejectsUnknownNames();
    else if (name == "window-buffer-supplies-its-own-source")
      windowBufferSuppliesItsOwnSource();
    else if (name == "window-buffer-stays-out-of-the-value-pool")
      windowBufferStaysOutOfTheValuePool();
    else if (name == "window-buffer-witness-uses-its-own-extent")
      windowBufferWitnessUsesItsOwnExtent();
    else if (name == "window-buffer-name-collision-is-refused")
      windowBufferNameCollisionIsRefused();
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
    else if (name == "magnitude-three-state-trips")
      magnitudeThreeStateTrips();
    else if (name == "magnitude-product-compounds")
      magnitudeProductCompounds();
    else if (name == "magnitude-reductions-and-shapes")
      magnitudeReductionsAndShapes();
    else if (name == "magnitude-window-fill")
      magnitudeWindowFill();
    else if (name == "order-sensitive-values-are-live-sums")
      orderSensitiveValuesAreLiveSums();
    else if (name == "magnitudes-track-sums")
      magnitudesTrackSums();
    else if (name == "reduction-contract-bounds")
      reductionContractBounds();
    else if (name == "most-live-roots-keep-more-nodes")
      mostLiveRootsKeepMoreNodes();
    else if (name == "deepest-roots-are-the-default")
      deepestRootsAreTheDefault();
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
