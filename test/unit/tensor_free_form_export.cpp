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

#include "ast/ast.hpp"
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
    /// The loop runs to an i32 `%trips` parameter ranged 0..tripCount.
    bool tripParameter = false;
    std::uint32_t stateCount = 2;
    std::uint32_t maxNodes = 12;
    std::uint32_t numWarps = 4;
    FreeFormFeatures features;
    std::uint32_t maxRank = 2;
    std::string profile = "tensor-v2";
    TensorElement element = TensorElement::F32;
    std::string dtype = "f32";
    bool windowEnabled = false;
    FreeFormWindow window;
    std::string windowSource;
    /// Nonzero adds an assembly-only backing parameter of this many rows.
    std::uint32_t windowBufferRows = 0;
    /// One of the four study requirements, or empty for an unconstrained body.
    std::string windowRequirement;
    std::uint32_t maxAttempts = 0;
    bool maxAttemptsSet = false;
    std::uint32_t maxMultiplications = 2;
    bool maxMultiplicationsSet = false;
    /// Positive enables the exact-range check for inputs of this magnitude.
    double exactInputBound = 0.0;
    FreeFormExactContract exactContract = FreeFormExactContract::AllValues;
    FreeFormRootSelection rootSelection = FreeFormRootSelection::Deepest;
    bool requireComputedRoots = false;
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
      } else if (flag == "--profile") {
        const std::string text = value();
        if (text == "v2") {
          options.profile = "tensor-v2";
        } else if (text == "v3") {
          options.profile = "tensor-v3";
        } else {
          throw std::invalid_argument("free-form export profile must be v2 or v3");
        }
      } else if (flag == "--dtype") {
        const std::string text = value();
        if (text == "f32") {
          options.element = TensorElement::F32;
          options.dtype = "f32";
        } else if (text == "i32") {
          options.element = TensorElement::I32;
          options.dtype = "i32";
        } else {
          throw std::invalid_argument("free-form export dtype must be f32 or i32");
        }
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
          else if (item == "extrema")
            options.features.extrema = true;
          else if (item == "dot")
            options.features.dot = true;
          else if (item == "narrow")
            options.features.narrow = true;
          else if (item == "where")
            options.features.where = true;
          else
            throw std::invalid_argument("unknown free-form feature: " + item);
          if (end == std::string::npos)
            break;
          begin = end + 1;
        }
      } else if (flag == "--window") {
        const std::string text = value();
        std::vector<std::uint32_t> fields;
        std::istringstream stream(text);
        std::string item;
        while (std::getline(stream, item, ',')) {
          const auto extent = std::stoul(item);
          if (extent > 4096)
            throw std::invalid_argument("free-form export window field is too large");
          fields.push_back(static_cast<std::uint32_t>(extent));
        }
        if (fields.size() != 3 || fields[0] == 0 || fields[2] == 0)
          throw std::invalid_argument("free-form export window needs rows,stride,active");
        options.windowEnabled = true;
        options.window.rows = fields[0];
        options.window.stride = fields[1];
        options.window.active = fields[2];
        options.window.result = "%tile";
        options.window.index = "%i";
      } else if (flag == "--window-source") {
        options.windowSource = value();
      } else if (flag == "--window-buffer-rows") {
        options.windowBufferRows = static_cast<std::uint32_t>(std::stoul(value()));
        if (options.windowBufferRows > 4096)
          throw std::invalid_argument("free-form export window buffer is too large");
      } else if (flag == "--window-requirement") {
        options.windowRequirement = value();
        if (options.windowRequirement != "reduction" && options.windowRequirement != "shared" &&
            options.windowRequirement != "extrema-transpose" &&
            options.windowRequirement != "extrema-expand")
          throw std::invalid_argument(
              "unknown free-form window requirement: " + options.windowRequirement
          );
      } else if (flag == "--max-attempts") {
        options.maxAttempts = static_cast<std::uint32_t>(std::stoul(value()));
        options.maxAttemptsSet = true;
        if (options.maxAttempts == 0)
          throw std::invalid_argument("free-form export max attempts must be positive");
      } else if (flag == "--max-multiplications") {
        options.maxMultiplications = static_cast<std::uint32_t>(std::stoul(value()));
        options.maxMultiplicationsSet = true;
      } else if (flag == "--exact-input-bound") {
        options.exactInputBound = refractir::parseFloatLiteral(value());
      } else if (flag == "--exact-contract") {
        const std::string contract = value();
        if (contract == "all")
          options.exactContract = FreeFormExactContract::AllValues;
        else if (contract == "reduction")
          options.exactContract = FreeFormExactContract::Reductions;
        else
          throw std::invalid_argument("unknown free-form exact contract: " + contract);
      } else if (flag == "--root-selection") {
        const std::string selection = value();
        if (selection == "deepest")
          options.rootSelection = FreeFormRootSelection::Deepest;
        else if (selection == "most-live")
          options.rootSelection = FreeFormRootSelection::MostLive;
        else
          throw std::invalid_argument("unknown free-form root selection: " + selection);
      } else if (flag == "--trip-parameter") {
        options.tripParameter = true;
      } else if (flag == "--require-computed-roots") {
        options.requireComputedRoots = true;
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
    if (options.tripParameter && options.tripCount > 8)
      throw std::invalid_argument(
          "free-form export: a trip parameter's range must allow one to eight trips"
      );
    if (options.numWarps != 1 && options.numWarps != 2 && options.numWarps != 4 &&
        options.numWarps != 8)
      throw std::invalid_argument("free-form export num_warps must be 1, 2, 4, or 8");
    if (options.shape.size() != 2)
      throw std::invalid_argument("free-form export needs a two-dimensional shape");
    if (options.windowEnabled && options.windowSource.empty() && options.windowBufferRows == 0)
      throw std::invalid_argument("free-form export needs both --window and a window source");
    if (!options.windowEnabled && (!options.windowSource.empty() || options.windowBufferRows != 0))
      throw std::invalid_argument("free-form export window needs --window");
    if (!options.windowSource.empty() && options.windowBufferRows != 0)
      throw std::invalid_argument(
          "free-form export window buffer conflicts with --window-source"
      );
    if (!options.windowRequirement.empty() && !options.windowEnabled)
      throw std::invalid_argument("free-form export window requirement needs --window");
    if (options.windowEnabled) {
      const std::uint32_t sourceExtent =
          options.windowBufferRows != 0 ? options.windowBufferRows : options.shape[0];
      if (options.windowBufferRows == 0) {
        bool known = false;
        for (std::uint32_t index = 0; index < std::min<std::uint32_t>(options.stateCount, 2);
             ++index)
          known = known || options.windowSource == parameterNames[index];
        if (!known)
          throw std::invalid_argument(
              "free-form export window source is not a runtime parameter"
          );
        options.window.source = options.windowSource;
      } else {
        options.window.source = "%window_buffer";
      }
      if (options.window.rows > sourceExtent || options.window.active > sourceExtent)
        throw std::invalid_argument(
            "free-form export window rows and active must fit the source extent"
        );
      // Check the witness before writing anything, so a refused case leaves no
      // program behind.
      (void) witnessWindow(options.window, options.tripCount, sourceExtent);
      if (options.element == TensorElement::F32) {
        options.window.fillKind = FreeFormFillKind::Float;
        options.window.fillFloat = 0.0;
      } else {
        options.window.fillKind = FreeFormFillKind::Integer;
        options.window.fillInteger = 0;
      }
    }
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

  std::vector<FreeFormRequirement> windowRequirements(const Options &options) {
    std::vector<FreeFormRequirement> requirements;
    // The study requires each result root to keep its own prior state; the
    // window relationship is layered on top of that.
    requirements.push_back({"%acc_x", {0}, {}, {}, false});
    requirements.push_back({"%acc_y", {1}, {}, {}, false});
    if (options.windowRequirement == "reduction") {
      requirements.push_back(
          {"%tile", {0}, {TensorOp::Sum, TensorOp::ReduceMin, TensorOp::ReduceMax}, {}, false}
      );
    } else if (options.windowRequirement == "shared") {
      requirements.push_back(
          {"%tile", {0}, {TensorOp::Sum, TensorOp::ReduceMin, TensorOp::ReduceMax}, {}, false}
      );
      requirements.push_back({"%tile", {0, 1}, {}, {}, true});
    } else if (options.windowRequirement == "extrema-transpose") {
      requirements.push_back(
          {"%tile", {0}, {TensorOp::ReduceMin, TensorOp::ReduceMax}, {TensorOp::Transpose}, false}
      );
    } else if (options.windowRequirement == "extrema-expand") {
      requirements.push_back(
          {"%tile", {0}, {TensorOp::ReduceMin, TensorOp::ReduceMax}, {TensorOp::Expand}, false}
      );
    }
    return requirements;
  }

  std::string requirementJson(const FreeFormRequirementWitness &witness) {
    std::ostringstream out;
    out << "{\"input\":" << jsonString(witness.input) << ",\"results\":[";
    for (std::size_t index = 0; index < witness.results.size(); ++index) {
      if (index)
        out << ',';
      out << jsonString(witness.results[index]);
    }
    out << "],\"path\":[";
    for (std::size_t index = 0; index < witness.path.size(); ++index) {
      if (index)
        out << ',';
      out << jsonString(witness.path[index]);
    }
    out << "],\"operations\":[";
    for (std::size_t index = 0; index < witness.operations.size(); ++index) {
      if (index)
        out << ',';
      out << jsonString(witness.operations[index]);
    }
    out << "],\"operation_ids\":[";
    for (std::size_t index = 0; index < witness.operationIds.size(); ++index) {
      if (index)
        out << ',';
      out << witness.operationIds[index];
    }
    out << "],\"consumers\":[";
    for (std::size_t index = 0; index < witness.consumers.size(); ++index) {
      if (index)
        out << ',';
      out << jsonString(witness.consumers[index]);
    }
    out << "],\"consumer_results\":[";
    for (std::size_t index = 0; index < witness.consumerResults.size(); ++index) {
      if (index)
        out << ',';
      out << jsonString(witness.consumerResults[index]);
    }
    out << "],\"distinct_consumers\":"
        << (witness.distinctConsumersSatisfied ? "true" : "false") << '}';
    return out.str();
  }

  std::string liveDagJson(const FreeFormLiveDag &dag) {
    std::ostringstream out;
    out << "{\"values\":[";
    for (std::size_t index = 0; index < dag.values.size(); ++index) {
      if (index)
        out << ',';
      out << jsonString(dag.values[index]);
    }
    out << "],\"ops\":[";
    for (std::size_t index = 0; index < dag.ops.size(); ++index) {
      if (index)
        out << ',';
      out << jsonString(dag.ops[index]);
    }
    out << "],\"operands\":[";
    for (std::size_t index = 0; index < dag.operands.size(); ++index) {
      if (index)
        out << ',';
      out << '[';
      for (std::size_t operand = 0; operand < dag.operands[index].size(); ++operand) {
        if (operand)
          out << ',';
        out << jsonString(dag.operands[index][operand]);
      }
      out << ']';
    }
    out << "],\"roots\":[";
    for (std::size_t index = 0; index < dag.roots.size(); ++index) {
      if (index)
        out << ',';
      out << jsonString(dag.roots[index]);
    }
    out << "],\"connectivity_signature\":" << jsonString(dag.connectivitySignature)
        << ",\"typed_fingerprint\":" << jsonString(dag.typedFingerprint) << '}';
    return out.str();
  }

  void run(const Options &options) {
    const TensorTypeInfo stateType{options.shape, options.element};
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
    if (options.windowEnabled) {
      FreeFormInput tile;
      tile.value = TensorRegionInput{
          "%tile", TensorTypeInfo{{options.window.rows, options.shape[1]}, options.element}};
      tile.role = FreeFormRole::Invariant;
      tile.init = FreeFormInit::Window;
      inputs.push_back(std::move(tile));
    }
    std::vector<FreeFormResult> results;
    for (std::uint32_t index = 0; index < options.stateCount; ++index)
      results.push_back(
          FreeFormResult{
              std::string(resultNames[index]), stateType, options.requireComputedRoots
          }
      );
    FreeFormRequest request;
    request.inputs = inputs;
    request.results = results;
    request.policy = options.policy;
    request.features = options.features;
    request.seed = options.seed;
    request.rootSelection = options.rootSelection;
    request.limits.maxNodes = options.maxNodes;
    request.limits.maxRank = options.maxRank;
    if (options.maxAttemptsSet)
      request.limits.maxAttempts = options.maxAttempts;
    if (options.maxMultiplicationsSet)
      request.limits.maxMultiplications = options.maxMultiplications;
    if (!options.windowRequirement.empty())
      request.requirements = windowRequirements(options);
    const auto built = buildFreeFormRegion(request);
    if (options.windowEnabled) {
      const auto tileIndex = static_cast<std::size_t>(inputs.size() - 1);
      bool consumed = false;
      for (const auto mask: built.structure.resultDependencies)
        consumed = consumed || ((mask >> tileIndex) & 1ULL) != 0;
      if (!consumed)
        throw std::invalid_argument(
            "free-form export: the body must consume the window input"
        );
    }

    FreeFormLoop loop;
    loop.inputs = inputs;
    loop.results = built.inputSpec.results;
    for (std::uint32_t index = 0; index < options.stateCount; ++index)
      loop.updates.emplace_back(
          std::string(stateNames[index]), built.inputSpec.results[index]
      );
    loop.tripCount = options.tripCount;
    if (options.tripParameter)
      loop.tripParameter = "%trips";
    if (options.windowEnabled) {
      loop.windows = {options.window};
      if (options.windowBufferRows != 0)
        loop.windowSources = {{
            "%window_buffer",
            TensorTypeInfo{{options.windowBufferRows, options.shape[1]}, options.element},
        }};
    }
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
    // Opt-in: with integer inputs and no division every value stays an
    // integer, so exactness is a magnitude question, and a sound bound below
    // the contract's limits also rules out overflow to a non-finite value.
    FreeFormMagnitudes magnitudes;
    if (options.exactInputBound > 0.0) {
      magnitudes = freeFormMagnitudes(built.region, loop, options.exactInputBound);
      if (!freeFormWithinExactContract(magnitudes, options.exactContract))
        throw std::invalid_argument(
            "free-form export: value magnitude bound " +
            refractir::formatDouble(magnitudes.largest) + " (sums " +
            refractir::formatDouble(magnitudes.largestSum) +
            ") exceeds the exact f32 integer range for inputs bounded by " +
            refractir::formatDouble(options.exactInputBound)
        );
    }
    // The canonical adapter is recorded, never substituted: the experiment
    // travels the target-only route, so whether the bytes also fit the
    // canonical configuration is evidence about the body, not a second program.
    const auto canonical =
        options.stateCount == 2 && !options.windowEnabled
            ? assembleCanonicalLoop(built.region, options.tripCount)
            : FreeFormAssembly{
                  "", options.windowEnabled
                          ? "a canonical loop cannot carry a window"
                          : "a canonical loop needs exactly two states"
              };

    std::filesystem::create_directories(options.output);
    writeFile(options.output / "program.sir", source);
    std::ostringstream record;
    record << "{\"schema\":\"refractir.free-form-region.v1\",\"entry\":\"kernel\","
           << "\"profile\":" << jsonString(options.profile)
           << ",\"dtype\":" << jsonString(options.dtype)
           << ",\"policy\":" << jsonString(std::string(freeFormPolicyName(options.policy)))
           << ",\"seed\":" << options.seed << ",\"shape\":" << shapeJson(options.shape)
           << ",\"trip_count\":" << options.tripCount << ",\"trip_parameter\":"
           << (options.tripParameter
                   ? "{\"name\":\"%trips\",\"max\":" + std::to_string(options.tripCount) + "}"
                   : std::string("null"))
           << ",\"state_count\":" << options.stateCount << ",\"max_nodes\":" << options.maxNodes
           << ",\"max_rank\":" << options.maxRank << ",\"num_warps\":" << options.numWarps;
    if (options.exactInputBound > 0.0)
      record << ",\"exact_range\":{\"input_bound\":"
             << refractir::formatDouble(options.exactInputBound) << ",\"contract\":\""
             << (options.exactContract == FreeFormExactContract::Reductions ? "reduction" : "all")
             << "\",\"magnitude_bound\":" << refractir::formatDouble(magnitudes.largest)
             << ",\"sum_bound\":" << refractir::formatDouble(magnitudes.largestSum) << "}";
    if (options.rootSelection == FreeFormRootSelection::MostLive)
      record << ",\"root_selection\":\"most-live\"";
    record << ",\"route\":\"target-only\","
           << "\"features\":{\"transpose\":" << (options.features.transpose ? "true" : "false")
           << ",\"expand\":" << (options.features.expand ? "true" : "false") << ",\"extrema\":"
           << (options.features.extrema ? "true" : "false")
           // Recorded only when asked for, so existing exports keep their bytes.
           << (options.features.dot ? ",\"dot\":true" : "")
           << (options.features.narrow ? ",\"narrow\":true" : "")
           << (options.features.where ? ",\"where\":true" : "") << "},"
           << "\"window\":";
    if (options.windowEnabled) {
      const auto sourceExtent =
          options.windowBufferRows != 0 ? options.windowBufferRows : options.shape[0];
      const auto witness = witnessWindow(options.window, options.tripCount, sourceExtent);
      const bool v3 = options.profile == "tensor-v3";
      const auto isPowerOfTwo = [](std::uint32_t value) {
        return value != 0 && (value & (value - 1)) == 0;
      };
      const bool padded =
          v3 && (!isPowerOfTwo(options.window.rows) || !isPowerOfTwo(options.shape[1]));
      const auto facts = classifyWindow(
          options.window.rows, options.window.stride, options.window.active,
          options.tripCount, v3, padded
      );
      record << "{\"result\":" << jsonString(options.window.result)
             << ",\"source\":" << jsonString(options.window.source)
             << ",\"rows\":" << options.window.rows << ",\"stride\":" << options.window.stride
             << ",\"active\":" << options.window.active << ",\"fill\":"
             << (options.window.fillKind == FreeFormFillKind::Float
                     ? "0.0"
                     : "0")
             << ",\"source_kind\":"
             << jsonString(
                    options.windowBufferRows != 0 ? "backing" : "input"
                )
             << ",\"source_shape\":["
             << (options.windowBufferRows != 0 ? options.windowBufferRows : options.shape[0])
             << ',' << options.shape[1] << ']'
             << ",\"witness\":{\"computed_max\":" << witness.computedMax
             << ",\"per_trip_active\":[";
      for (std::size_t trip = 0; trip < witness.perTripActive.size(); ++trip) {
        if (trip)
          record << ',';
        record << '[';
        for (std::size_t row = 0; row < witness.perTripActive[trip].size(); ++row) {
          if (row)
            record << ',';
          record << (witness.perTripActive[trip][row] ? "true" : "false");
        }
        record << ']';
      }
      record << "],\"all_active\":" << (witness.allActive ? "true" : "false") << "},"
             << "\"facts\":{\"regime\":" << jsonString(std::string(facts.regime))
             << ",\"shape_validity\":" << jsonString(std::string(facts.shapeValidity))
             << ",\"active_predicate\":" << jsonString(std::string(facts.activePredicate))
             << ",\"mask_after_compilation\":"
             << jsonString(std::string(facts.maskAfterCompilation)) << "}}";
    } else {
      record << "null";
    }
    record << ",\"canonical\":{"
           << "\"accepted\":" << (canonical.source.empty() ? "false" : "true")
           << ",\"refusal\":" << jsonString(canonical.refusal) << "},\"parameters\":[";
    bool first = true;
    for (const auto &windowSource: loop.windowSources) {
      if (!first)
        record << ',';
      first = false;
      record << "{\"name\":" << jsonString(windowSource.parameter)
             << ",\"shape\":" << shapeJson(windowSource.type.shape)
             << ",\"dtype\":" << jsonString(scalarElement(windowSource.type))
             << ",\"role\":\"window_source\",\"init\":\"window_backing\"}";
    }
    for (const auto &input: inputs) {
      if (input.init != FreeFormInit::RuntimeInput)
        continue;
      if (!first)
        record << ',';
      first = false;
      record << "{\"name\":" << jsonString(input.parameter)
             << ",\"shape\":" << shapeJson(input.value.type.shape)
             << ",\"dtype\":" << jsonString(scalarElement(input.value.type))
             << ",\"role\":"
             << jsonString(input.role == FreeFormRole::State ? "state" : "invariant")
             << ",\"init\":\"runtime\"}";
    }
    if (options.tripParameter)
      record << (first ? "" : ",")
             << "{\"name\":\"%trips\",\"shape\":[],\"dtype\":\"i32\",\"role\":\"trip_count\","
                "\"init\":\"runtime\"}";
    record << "],\"window_requirement\":";
    if (options.windowRequirement.empty())
      record << "null";
    else
      record << jsonString(options.windowRequirement);
    record << ",\"max_attempts\":" << request.limits.maxAttempts
           << ",\"max_multiplications\":" << request.limits.maxMultiplications
           << ",\"requirements\":[";
    for (std::size_t index = 0; index < built.structure.requirementWitnesses.size(); ++index) {
      if (index)
        record << ',';
      record << requirementJson(built.structure.requirementWitnesses[index]);
    }
    record << "],\"live_dag\":" << liveDagJson(built.structure.liveDag)
           << ",\"window_source\":";
    if (options.windowBufferRows != 0) {
      record << "{\"parameter\":\"%window_buffer\",\"shape\":["
             << options.windowBufferRows << ',' << options.shape[1]
             << "],\"dtype\":" << jsonString(options.dtype)
             << ",\"role\":\"window_source\"}";
    } else {
      record << "null";
    }
    record << ",\"inputs\":[";
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      if (index)
        record << ',';
      const auto &input = inputs[index];
      record << "{\"value\":" << jsonString(input.value.name)
             << ",\"role\":" << jsonString(input.role == FreeFormRole::State ? "state" : "invariant")
             << ",\"init\":"
             << jsonString(
                    input.init == FreeFormInit::Zero     ? "zero"
                    : input.init == FreeFormInit::Window ? "window"
                                                         : "runtime"
                )
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
