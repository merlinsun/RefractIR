#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "reify/tensor_free_form.hpp"

// Export one free-form region as a case directory: the assembled SIR plus the
// record a target-only bundle states. The builder decides the body; this tool
// only chooses the loop configuration around it, so a corpus program and the
// structure it was built from cannot disagree.
namespace {
  using namespace refractir;
  using namespace refractir::reify;

  std::string jsonString(std::string_view value) {
    std::string out{"\""};
    constexpr char hex[] = "0123456789abcdef";
    for (const unsigned char ch: value) {
      switch (ch) {
        case '"':
          out += "\\\"";
          break;
        case '\\':
          out += "\\\\";
          break;
        case '\n':
          out += "\\n";
          break;
        case '\r':
          out += "\\r";
          break;
        case '\t':
          out += "\\t";
          break;
        default:
          if (ch < 0x20) {
            out += "\\u00";
            out.push_back(hex[ch >> 4]);
            out.push_back(hex[ch & 15]);
          } else {
            out.push_back(static_cast<char>(ch));
          }
      }
    }
    out.push_back('"');
    return out;
  }

  struct Options {
    FreeFormPolicy policy = FreeFormPolicy::Unguided;
    std::uint32_t seed = 0;
    Shape shape{2, 2};
    std::uint32_t tripCount = 2;
    std::uint32_t stateCount = 2;
    std::uint32_t maxNodes = 12;
    std::uint32_t numWarps = 4;
    FreeFormFeatures features;
    std::uint32_t maxRank = 2;
    /// `combined` returns the sum of every state; `state:K` returns state K.
    bool returnCombined = true;
    std::uint32_t returnState = 0;
    std::filesystem::path output;
  };

  constexpr std::string_view stateNames[] = {"%acc_x", "%acc_y", "%acc_z", "%acc_w"};
  constexpr std::string_view resultNames[] = {"%next_x", "%next_y", "%next_z", "%next_w"};
  constexpr std::string_view parameterNames[] = {"%x", "%y", "%z", "%w"};

  Shape parseShape(const std::string &text) {
    Shape shape;
    std::istringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
      const auto extent = std::stoul(item);
      if (extent == 0 || extent > 4096)
        throw std::invalid_argument("free-form export shape extent is outside [1, 4096]");
      shape.push_back(static_cast<std::uint32_t>(extent));
    }
    if (shape.empty())
      throw std::invalid_argument("free-form export shape needs at least one extent");
    return shape;
  }

  Options parseOptions(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
      const std::string flag = argv[index];
      const auto value = [&]() -> std::string {
        if (index + 1 >= argc)
          throw std::invalid_argument("free-form export needs a value after " + flag);
        return argv[++index];
      };
      if (flag == "--policy") {
        const auto policy = parseFreeFormPolicy(value());
        if (!policy)
          throw std::invalid_argument("unknown free-form policy: " + std::string(argv[index]));
        options.policy = *policy;
      } else if (flag == "--seed") {
        options.seed = static_cast<std::uint32_t>(std::stoul(value()));
      } else if (flag == "--shape") {
        options.shape = parseShape(value());
      } else if (flag == "--trip-count") {
        options.tripCount = static_cast<std::uint32_t>(std::stoul(value()));
      } else if (flag == "--states") {
        options.stateCount = static_cast<std::uint32_t>(std::stoul(value()));
      } else if (flag == "--max-nodes") {
        options.maxNodes = static_cast<std::uint32_t>(std::stoul(value()));
      } else if (flag == "--max-rank") {
        options.maxRank = static_cast<std::uint32_t>(std::stoul(value()));
      } else if (flag == "--features") {
        const std::string text = value();
        if (text.empty() || text == "none")
          continue;
        std::size_t begin = 0;
        while (begin <= text.size()) {
          const auto end = text.find(',', begin);
          const std::string item = text.substr(begin, end == std::string::npos ? end : end - begin);
          if (item == "transpose")
            options.features.transpose = true;
          else if (item == "expand")
            options.features.expand = true;
          else
            throw std::invalid_argument("unknown free-form feature: " + item);
          if (end == std::string::npos)
            break;
          begin = end + 1;
        }
      } else if (flag == "--num-warps") {
        options.numWarps = static_cast<std::uint32_t>(std::stoul(value()));
      } else if (flag == "--return") {
        const std::string text = value();
        if (text == "combined") {
          options.returnCombined = true;
        } else if (text.starts_with("state:")) {
          options.returnCombined = false;
          options.returnState = static_cast<std::uint32_t>(std::stoul(text.substr(6)));
        } else {
          throw std::invalid_argument(
              "free-form export return must be combined or state:<index>"
          );
        }
      } else if (flag == "--out") {
        options.output = value();
      } else {
        throw std::invalid_argument("unknown free-form export option: " + flag);
      }
    }
    if (options.output.empty())
      throw std::invalid_argument("free-form export needs --out");
    if (options.maxRank == 0 || options.maxRank > 3)
      throw std::invalid_argument("free-form export max rank must be in [1, 3]");
    if (options.stateCount < 2 || options.stateCount > 4)
      throw std::invalid_argument("free-form export admits two to four states");
    if (options.tripCount == 0 || options.tripCount > 64)
      throw std::invalid_argument("free-form export trip count must be in [1, 64]");
    if (options.numWarps != 1 && options.numWarps != 2 && options.numWarps != 4 &&
        options.numWarps != 8)
      throw std::invalid_argument("free-form export num_warps must be 1, 2, 4, or 8");
    if (options.shape.size() != 2)
      throw std::invalid_argument("free-form export needs a two-dimensional shape");
    return options;
  }

  void writeFile(const std::filesystem::path &path, const std::string &text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
      throw std::runtime_error("cannot create " + path.string());
    out << text;
    if (!out)
      throw std::runtime_error("cannot write " + path.string());
  }

  std::string structureJson(const FreeFormStructure &structure) {
    std::ostringstream out;
    out << "{\"policy\":" << jsonString(structure.policy)
        << ",\"target\":" << jsonString(structure.target)
        << ",\"actual\":" << jsonString(structure.actual) << ",\"nodes\":" << structure.nodes
        << ",\"depth\":" << structure.depth << ",\"reductions\":" << structure.reductions
        << ",\"shape_changes\":" << structure.shapeChanges
        << ",\"distinct_consumers\":" << structure.distinctConsumers
        << ",\"shared_values\":" << structure.sharedValues
        << ",\"max_def_use_distance\":" << structure.maxDefUseDistance
        << ",\"result_dependencies\":[";
    for (std::size_t index = 0; index < structure.resultDependencies.size(); ++index) {
      if (index)
        out << ',';
      out << structure.resultDependencies[index];
    }
    out << "],\"operator_counts\":{";
    for (std::size_t index = 0; index < structure.operatorCounts.size(); ++index) {
      if (index)
        out << ',';
      out << jsonString(structure.operatorCounts[index].first) << ':'
          << structure.operatorCounts[index].second;
    }
    out << "},\"consumers\":{";
    for (std::size_t index = 0; index < structure.consumers.size(); ++index) {
      if (index)
        out << ',';
      out << jsonString(structure.consumers[index].first) << ':'
          << structure.consumers[index].second;
    }
    out << "}}";
    return out.str();
  }

  std::string scalarElement(const TensorTypeInfo &type) {
    return type.element == TensorElement::F32 ? "f32" : "i32";
  }

  std::string shapeJson(const Shape &shape) {
    std::ostringstream out;
    out << '[';
    for (std::size_t index = 0; index < shape.size(); ++index) {
      if (index)
        out << ',';
      out << shape[index];
    }
    out << ']';
    return out.str();
  }

  void run(const Options &options) {
    const TensorTypeInfo stateType{options.shape, TensorElement::F32};
    std::vector<FreeFormInput> inputs;
    for (std::uint32_t index = 0; index < options.stateCount; ++index) {
      FreeFormInput input;
      input.value = TensorRegionInput{std::string(stateNames[index]), stateType};
      input.role = FreeFormRole::State;
      // The first two states read runtime inputs; every later state starts at
      // zero. That keeps the two-state program canonical without making every
      // program's arithmetic start from the same place.
      if (index < 2) {
        input.init = FreeFormInit::RuntimeInput;
        input.parameter = std::string(parameterNames[index]);
      } else {
        input.init = FreeFormInit::Zero;
      }
      inputs.push_back(std::move(input));
    }
    std::vector<FreeFormResult> results;
    for (std::uint32_t index = 0; index < options.stateCount; ++index)
      results.push_back(FreeFormResult{std::string(resultNames[index]), stateType});
    FreeFormRequest request;
    request.inputs = inputs;
    request.results = results;
    request.policy = options.policy;
    request.features = options.features;
    request.seed = options.seed;
    request.limits.maxNodes = options.maxNodes;
    request.limits.maxRank = options.maxRank;
    const auto built = buildFreeFormRegion(request);

    FreeFormLoop loop;
    loop.inputs = inputs;
    loop.results = built.inputSpec.results;
    for (std::uint32_t index = 0; index < options.stateCount; ++index)
      loop.updates.emplace_back(
          std::string(stateNames[index]), built.inputSpec.results[index]
      );
    loop.tripCount = options.tripCount;
    if (options.returnCombined) {
      for (std::uint32_t index = 0; index < options.stateCount; ++index)
        loop.finalTerms.emplace_back(stateNames[index]);
      loop.finalOp = TensorOp::Add;
    } else {
      if (options.returnState >= options.stateCount)
        throw std::invalid_argument("free-form export return state is outside the state count");
      loop.finalTerms = {std::string(stateNames[options.returnState])};
    }
    const auto source = assembleCountedLoop(built.region, loop);
    // The canonical adapter is recorded, never substituted: the experiment
    // travels the target-only route, so whether the bytes also fit the
    // canonical configuration is evidence about the body, not a second program.
    const auto canonical =
        options.stateCount == 2
            ? assembleCanonicalLoop(built.region, options.tripCount)
            : FreeFormAssembly{"", "a canonical loop needs exactly two states"};

    std::filesystem::create_directories(options.output);
    writeFile(options.output / "program.sir", source);
    std::ostringstream record;
    record << "{\"schema\":\"refractir.free-form-region.v1\",\"entry\":\"kernel\","
           << "\"profile\":\"tensor-v2\",\"dtype\":\"f32\",\"policy\":"
           << jsonString(std::string(freeFormPolicyName(options.policy)))
           << ",\"seed\":" << options.seed << ",\"shape\":" << shapeJson(options.shape)
           << ",\"trip_count\":" << options.tripCount << ",\"state_count\":"
           << options.stateCount << ",\"max_nodes\":" << options.maxNodes
           << ",\"max_rank\":" << options.maxRank
           << ",\"num_warps\":" << options.numWarps << ",\"route\":\"target-only\","
           << "\"features\":{\"transpose\":"
           << (options.features.transpose ? "true" : "false")
           << ",\"expand\":" << (options.features.expand ? "true" : "false") << "},"
           << "\"canonical\":{"
           << "\"accepted\":" << (canonical.source.empty() ? "false" : "true")
           << ",\"refusal\":" << jsonString(canonical.refusal) << "},\"parameters\":[";
    bool first = true;
    for (const auto &input: inputs) {
      if (input.init != FreeFormInit::RuntimeInput)
        continue;
      if (!first)
        record << ',';
      first = false;
      record << "{\"name\":" << jsonString(input.parameter)
             << ",\"shape\":" << shapeJson(input.value.type.shape)
             << ",\"dtype\":" << jsonString(scalarElement(input.value.type)) << '}';
    }
    record << "],\"inputs\":[";
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      if (index)
        record << ',';
      const auto &input = inputs[index];
      record << "{\"value\":" << jsonString(input.value.name)
             << ",\"role\":" << jsonString(input.role == FreeFormRole::State ? "state" : "invariant")
             << ",\"init\":" << jsonString(input.init == FreeFormInit::Zero ? "zero" : "runtime")
             << ",\"parameter\":" << jsonString(input.parameter)
             << ",\"shape\":" << shapeJson(input.value.type.shape)
             << ",\"dtype\":" << jsonString(scalarElement(input.value.type)) << '}';
    }
    record << "],\"results\":[";
    for (std::size_t index = 0; index < built.inputSpec.results.size(); ++index) {
      if (index)
        record << ',';
      record << jsonString(built.inputSpec.results[index]);
    }
    record << "],\"updates\":[";
    for (std::size_t index = 0; index < loop.updates.size(); ++index) {
      if (index)
        record << ',';
      record << "{\"state\":" << jsonString(loop.updates[index].first)
             << ",\"result\":" << jsonString(loop.updates[index].second) << '}';
    }
    record << "],\"final\":{\"terms\":[";
    for (std::size_t index = 0; index < loop.finalTerms.size(); ++index) {
      if (index)
        record << ',';
      record << jsonString(loop.finalTerms[index]);
    }
    record << "],\"op\":" << jsonString(std::string(tensorOpName(loop.finalOp)))
           << "},\"structure\":" << structureJson(built.structure) << "}\n";
    writeFile(options.output / "region.json", record.str());
  }

} // namespace

int main(int argc, char **argv) {
  try {
    run(parseOptions(argc, argv));
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
