// One deadline for input selection, replay, and response validation.
//
// The selection runs in a child process so that a solver which will not stop
// cannot hold the tool. The parent owns the deadline: it starts the clock
// before the request is written, terminates the child's process group when the
// interval expires, and never publishes work that finished late.

#include "reify/tensor_selection.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "analysis/tensor.hpp"
#include "ast/sir_printer.hpp"
#include "frontend/pipeline.hpp"
#include "interp/interpreter.hpp"
#include "interp/tensor_io.hpp"
#include "interp/tensor_numeric.hpp"
#include "analysis/type_utils.hpp"
#include "solver/backend.hpp"
#include "solver/solver.hpp"

namespace refractir::reify {

  namespace {

    // The protocol is line-oriented and hex-encodes every byte payload, so a
    // program with any text in it survives the boundary and a payload's length
    // is explicit rather than inferred from a delimiter. It is versioned by
    // its first line, and both directions are bounded before they are read.
    constexpr std::string_view REQUEST_SCHEMA = "refractir.tensor-selection.request.v1";
    constexpr std::string_view RESPONSE_SCHEMA = "refractir.tensor-selection.response.v1";
    constexpr std::size_t MAX_TRANSPORT_BYTES = 1 << 20;

    const char *HEX_DIGITS = "0123456789abcdef";

    std::string hexEncode(const std::uint8_t *data, std::size_t size) {
      std::string text;
      text.reserve(size * 2);
      for (std::size_t index = 0; index < size; ++index) {
        text.push_back(HEX_DIGITS[data[index] >> 4]);
        text.push_back(HEX_DIGITS[data[index] & 0xf]);
      }
      return text;
    }

    std::string hexEncode(const std::vector<std::uint8_t> &bytes) {
      return hexEncode(bytes.data(), bytes.size());
    }

    std::string hexEncode(std::string_view text) {
      return hexEncode(reinterpret_cast<const std::uint8_t *>(text.data()), text.size());
    }

    std::optional<std::vector<std::uint8_t>> hexDecode(std::string_view text) {
      if (text.size() % 2 != 0)
        return std::nullopt;
      std::vector<std::uint8_t> bytes;
      bytes.reserve(text.size() / 2);
      for (std::size_t index = 0; index < text.size(); index += 2) {
        const auto digit = [](char character) -> int {
          if (character >= '0' && character <= '9')
            return character - '0';
          if (character >= 'a' && character <= 'f')
            return character - 'a' + 10;
          if (character >= 'A' && character <= 'F')
            return character - 'A' + 10;
          return -1;
        };
        const int high = digit(text[index]);
        const int low = digit(text[index + 1]);
        if (high < 0 || low < 0)
          return std::nullopt;
        bytes.push_back(static_cast<std::uint8_t>((high << 4) | low));
      }
      return bytes;
    }

    /// One `key=value` line per record, parsed into the fields the protocol
    /// defines. Reading stops at the first malformed line.
    struct Record {
      std::vector<std::pair<std::string, std::string>> fields;

      void add(const std::string &key, const std::string &value) {
        fields.emplace_back(key, value);
      }

      [[nodiscard]] std::optional<std::string> get(std::string_view key) const {
        for (const auto &field: fields) {
          if (field.first == key)
            return field.second;
        }
        return std::nullopt;
      }

      [[nodiscard]] std::vector<std::string> all(std::string_view key) const {
        std::vector<std::string> values;
        for (const auto &field: fields) {
          if (field.first == key)
            values.push_back(field.second);
        }
        return values;
      }
    };

    void writeRecord(const std::filesystem::path &path, const Record &record) {
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      for (const auto &field: record.fields)
        out << field.first << '=' << field.second << '\n';
      out.close();
      if (!out)
        throw std::runtime_error("failed to write selection record: " + path.string());
    }

    std::optional<Record> readRecord(const std::filesystem::path &path) {
      std::error_code error;
      const auto size = std::filesystem::file_size(path, error);
      if (error || size > MAX_TRANSPORT_BYTES)
        return std::nullopt;
      std::ifstream in(path, std::ios::binary);
      if (!in)
        return std::nullopt;
      Record record;
      std::string line;
      while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r')
          line.pop_back();
        if (line.empty())
          continue;
        const auto equals = line.find('=');
        if (equals == std::string::npos)
          return std::nullopt;
        record.add(line.substr(0, equals), line.substr(equals + 1));
      }
      return record;
    }

    std::string joinLabels(const std::vector<std::string> &labels) {
      std::string joined;
      for (const auto &label: labels) {
        if (!joined.empty())
          joined += ',';
        joined += label;
      }
      return joined;
    }

    std::vector<std::string> splitLabels(std::string_view text) {
      std::vector<std::string> labels;
      std::size_t begin = 0;
      while (begin <= text.size()) {
        const auto comma = text.find(',', begin);
        const auto piece = text.substr(
            begin, comma == std::string_view::npos ? std::string_view::npos : comma - begin
        );
        if (!piece.empty())
          labels.emplace_back(piece);
        if (comma == std::string_view::npos)
          break;
        begin = comma + 1;
      }
      return labels;
    }

    std::string profileName(TensorProfileId id) {
      switch (id) {
        case TensorProfileId::V1:
          return "v1";
        case TensorProfileId::V2:
          return "v2";
        case TensorProfileId::V3:
          return "v3";
        case TensorProfileId::V4:
          return "v4";
      }
      return "v1";
    }

    /// The single entry function a tensor program declares.
    const FunDecl *entryFunction(const Program &program) {
      if (program.funs.empty())
        throw std::runtime_error("the program declares no entry function");
      return &program.funs.front();
    }

    TensorElement scalarElement(const TypePtr &type) {
      if (TypeUtils::getIntBitWidth(type) == 32)
        return TensorElement::I32;
      if (TypeUtils::getFloatBitWidth(type) == 32)
        return TensorElement::F32;
      throw std::runtime_error("selection only transports i32 and f32 elements");
    }

    TensorTypeInfo typeInfo(const TypePtr &type) {
      if (const auto tensor = TypeUtils::asTensor(type))
        return {tensor->shape, scalarElement(tensor->elem)};
      return {{}, scalarElement(type)};
    }

    std::string printProgram(const Program &program) {
      std::ostringstream text;
      SIRPrinter printer(text);
      printer.print(program);
      return text.str();
    }

    std::string propertyText(const TensorInputProperty &property) {
      switch (property.kind) {
        case TensorInputProperty::Kind::None:
          return "none";
        case TensorInputProperty::Kind::OutputEquals:
          return "output_equals:" + std::to_string(property.target);
        case TensorInputProperty::Kind::OutputSumEquals:
          return "output_sum_equals:" + std::to_string(property.target);
        case TensorInputProperty::Kind::ElementsEqual:
          return "elements_equal";
      }
      return "none";
    }

    std::optional<TensorInputProperty> parseProperty(const Record &record) {
      const auto text = record.get("property");
      if (!text)
        return std::nullopt;
      TensorInputProperty property;
      if (*text == "none") {
        property.kind = TensorInputProperty::Kind::None;
        return property;
      }
      const auto split = [&](std::string_view prefix) -> std::optional<std::int32_t> {
        if (text->rfind(prefix, 0) != 0)
          return std::nullopt;
        return static_cast<std::int32_t>(std::stoll(text->substr(prefix.size())));
      };
      if (const auto target = split("output_equals:")) {
        property.kind = TensorInputProperty::Kind::OutputEquals;
        property.target = *target;
        return property;
      }
      if (const auto target = split("output_sum_equals:")) {
        property.kind = TensorInputProperty::Kind::OutputSumEquals;
        property.target = *target;
        return property;
      }
      if (*text == "elements_equal") {
        property.kind = TensorInputProperty::Kind::ElementsEqual;
        for (const auto &element: record.all("element")) {
          const auto equals = element.find('=');
          if (equals == std::string::npos)
            return std::nullopt;
          TensorOutputElementConstraint constraint;
          for (const auto &axis: splitLabels(element.substr(0, equals)))
            constraint.index.push_back(static_cast<std::uint32_t>(std::stoul(axis)));
          constraint.value = static_cast<std::int32_t>(std::stoll(element.substr(equals + 1)));
          property.elements.push_back(constraint);
        }
        return property;
      }
      return std::nullopt;
    }

    std::string elementName(TensorElement element) {
      return element == TensorElement::F32 ? "f32" : "i32";
    }

    std::optional<TensorElement> elementFromName(std::string_view name) {
      if (name == "i32")
        return TensorElement::I32;
      if (name == "f32")
        return TensorElement::F32;
      return std::nullopt;
    }

    std::string shapeText(const Shape &shape) {
      std::string text = "[";
      for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        if (axis)
          text += ',';
        text += std::to_string(shape[axis]);
      }
      return text + "]";
    }

    // ---------------------------------------------------------------- worker

    /// Answer one request. Every stage writes its name to the notification
    /// file first, so a parent that terminates this process can say where the
    /// work stopped instead of guessing.
    class Worker {
    public:
      Worker(
          const std::filesystem::path &requestPath, const std::filesystem::path &responsePath,
          const std::filesystem::path &stagePath
      ) : requestPath_(requestPath), responsePath_(responsePath), stagePath_(stagePath) {}

      int run() {
        Record response;
        response.add("schema", std::string(RESPONSE_SCHEMA));
        try {
          const auto request = readRecord(requestPath_);
          if (!request || request->get("schema") != std::string(REQUEST_SCHEMA))
            throw std::runtime_error("unreadable selection request");
          if (!prepare(*request)) {
            response.add("status", "invalid");
            response.add("reason", failure_);
            response.add("stage", stage_);
            writeRecord(responsePath_, response);
            return 0;
          }
          const auto outcome = select();
          response.add("status", outcome.status);
          response.add("reason", outcome.reason);
          response.add("stage", stage_);
          // The identity and the request are reported for every terminal
          // outcome, so a refusal is attributable to the same graph and domain
          // an acceptance would be.
          response.add("digest", digest_);
          response.add(
              "domain",
              std::to_string(outcome.domain.first) + "," + std::to_string(outcome.domain.second)
          );
          response.add("symbolic_elements", std::to_string(outcome.symbolicElements));
          if (!outcome.unknownReason.empty())
            response.add("unknown_reason", outcome.unknownReason);
          if (outcome.status == "accepted") {
            response.add("path", joinLabels(path_));
            response.add("expected", hexEncode(outcome.expected));
            response.add("replay", joinLabels(outcome.replayPath));
            response.add("property", "holds");
            for (const auto &buffer: outcome.inputs) {
              response.add("param", buffer.name);
              response.add("param_element", elementName(buffer.element));
              response.add("param_shape", shapeText(buffer.shape));
              response.add("param_bytes", hexEncode(buffer.bytes));
            }
            for (const auto &entry: outcome.dependence) {
              response.add("dependence", entry.first);
              response.add("dependence_value", std::to_string(entry.second));
            }
          }
          writeRecord(responsePath_, response);
          return 0;
        } catch (const std::exception &error) {
          response.add("status", "invalid");
          response.add("reason", error.what());
          response.add("stage", stage_);
          try {
            writeRecord(responsePath_, response);
          } catch (const std::exception &) {
            return 1;
          }
          return 0;
        }
      }

    private:
      struct Outcome {
        std::string status = "invalid";
        std::string reason;
        std::string unknownReason;
        std::vector<TensorCaseBuffer> inputs;
        std::vector<std::uint8_t> expected;
        std::vector<std::string> replayPath;
        std::pair<std::int64_t, std::int64_t> domain{0, 0};
        std::uint32_t symbolicElements = 0;
        std::vector<std::pair<std::string, std::int32_t>> dependence;
      };

      void note(const std::string &stage) {
        stage_ = stage;
        std::ofstream out(stagePath_, std::ios::trunc);
        out << "stage=" << stage << '\n';
      }

      bool prepare(const Record &request) {
        note("query");
        // The profile decides which shapes the checker admits, so it is read
        // before the program is checked rather than after.
        if (const auto profile = request.get("profile"))
          options_.profileVersion = parseTensorProfileVersion(*profile);
        const auto programHex = request.get("program");
        if (!programHex)
          return fail("selection request carries no program");
        const auto programBytes = hexDecode(*programHex);
        if (!programBytes)
          return fail("selection request program is not hex");
        source_.assign(programBytes->begin(), programBytes->end());
        program_ = std::make_unique<Program>(parseSource(source_));
        DiagBag diagnostics;
        CheckOptions check;
        check.tensorProfile = options_.profileVersion;
        if (!checkProgram(*program_, diagnostics, check))
          return fail("selection request program does not check");
        digest_ = computationDigest(*program_);
        if (const auto path = request.get("path"))
          path_ = splitLabels(*path);
        if (path_.empty())
          return fail("selection request carries no path");
        const auto property = parseProperty(request);
        if (!property)
          return fail("selection request property is malformed");
        options_.policy = TensorInputPolicy::Ordinary;
        if (const auto policy = request.get("policy")) {
          if (*policy == "boundary")
            options_.policy = TensorInputPolicy::Boundary;
          else if (*policy == "exact")
            options_.policy = TensorInputPolicy::Exact;
          else if (*policy != "ordinary")
            return fail("selection request policy is unknown");
        }
        options_.property = *property;
        if (const auto dtype = request.get("dtype")) {
          const auto element = elementFromName(*dtype);
          if (!element)
            return fail("selection request dtype is unknown");
          element_ = *element;
        }
        const auto output = tensorOutputSignature(*program_);
        outputShape_ = output.shape;
        declaredDomain_ = options_.domain.value_or(std::pair<std::int64_t, std::int64_t>{0, 0});
        // The declared element decides the transport, and the request has to
        // agree with the program it carries.
        if (element_ != output.element)
          return fail("selection request dtype differs from the program's output");
        if (const auto domain = request.get("domain")) {
          const auto comma = domain->find(',');
          if (comma == std::string::npos)
            return fail("selection request domain is malformed");
          options_.domain = std::pair{
              std::stoll(domain->substr(0, comma)), std::stoll(domain->substr(comma + 1))
          };
        }
        if (const auto seed = request.get("seed"))
          options_.seed = static_cast<std::uint32_t>(std::stoul(*seed));
        if (const auto budget = request.get("timeout_ms"))
          budgetMs_ = static_cast<std::uint32_t>(std::stoul(*budget));
        started_ = std::chrono::steady_clock::now();
        return true;
      }

      bool fail(const std::string &reason) {
        failure_ = reason;
        return false;
      }

      /// Milliseconds left on this worker's own budget, zero when it is spent.
      std::uint32_t remainingMs() const {
        const auto spent = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - started_
        )
                               .count();
        if (spent >= static_cast<std::int64_t>(budgetMs_))
          return 0;
        return static_cast<std::uint32_t>(budgetMs_ - spent);
      }

      /// Milliseconds to give a backend. A backend limit of zero means "no
      /// limit", so a spent budget skips the query instead of disabling the
      /// timer. The margin keeps materialization, replay, and the response
      /// inside the worker's own interval, which is what lets a backend
      /// timeout be reported as one instead of being cut off by the parent.
      std::uint32_t backendBudgetMs() const {
        const auto remaining = remainingMs();
        if (remaining == 0)
          return 0;
        const auto margin = std::max<std::uint32_t>(50, remaining / 20);
        return remaining > margin ? remaining - margin : 1;
      }

      Outcome select() {
        const auto backendMs = backendBudgetMs();
        if (backendMs == 0) {
          auto expired = timeoutOutcome("the budget expired before the query started");
          expired.domain = declaredDomain_;
          return expired;
        }

        note("solver");
        // The backend limit is this worker's own guard, never the authority:
        // the parent kills the worker at the real deadline. A zero limit would
        // disable the backend timer, so the budget only ever hands it a
        // positive value or skips the query.
        // A backend that cannot report why it stopped must not be given its own
        // timer: the expiry would arrive as an unexplained unknown before the
        // deadline this project owns fires, which is exactly the cause C2
        // exists to preserve. Such a backend runs unbounded here and the
        // parent's deadline is what stops it.
        const std::uint32_t backendTimer =
            solver::backendReportsTerminationReason() ? backendMs : 0;
        const auto factory = [backendTimer](
                                 const SymbolicExecutor::Config &config
                             ) -> std::unique_ptr<smt::ISolver> {
          SymbolicExecutor::Config bounded = config;
          bounded.timeout_ms = backendTimer;
          return solver::configuredSolverFactory()(bounded);
        };
        const auto proposal = proposeTensorInputs(*program_, factory, options_);
        if (proposal.outcome != "accepted") {
          Outcome refusal;
          refusal.status = proposal.outcome;
          refusal.reason = proposal.reason;
          refusal.unknownReason = proposal.unknownReason;
          refusal.domain = proposal.domain;
          refusal.symbolicElements = proposal.symbolicElements;
          return refusal;
        }

        note("model");
        Outcome outcome;
        outcome.inputs = proposal.inputs;
        outcome.domain = proposal.domain;
        outcome.symbolicElements = proposal.symbolicElements;

        note("replay");
        std::vector<std::string> visited;
        outcome.expected = replay(proposal.inputs, &visited);
        outcome.replayPath = std::move(visited);
        if (outcome.replayPath != path_) {
          outcome.status = "path_mismatch";
          outcome.reason = "the replayed path differs from the selected path";
          return outcome;
        }

        note("response");
        if (options_.profileVersion == TensorProfileId::V2 && outcome.inputs.size() > 1) {
          for (std::size_t index = 0; index < outcome.inputs.size(); ++index) {
            const auto witness =
                counterfactualWitness(outcome.inputs, index, outcome.expected);
            if (!witness) {
              outcome.status = "counterfactual_dependency";
              outcome.reason = "the returned computation does not depend on input " +
                               outcome.inputs[index].name;
              return outcome;
            }
            outcome.dependence.emplace_back(outcome.inputs[index].name, *witness);
          }
        }
        if (!tensorPropertyHolds(outcome.expected, element_, outputShape_, options_.property)) {
          outcome.status = "model_incident";
          outcome.reason = "the replayed output does not satisfy the stated property";
          return outcome;
        }
        outcome.status = "accepted";
        return outcome;
      }

      /// Whether replacing one input's leaves changes the output, which is how
      /// a case shows that its returned value depends on that input. The check
      /// replays, so it belongs inside this worker rather than in the parent
      /// after the deadline has already been consumed.
      std::optional<std::int32_t> counterfactualWitness(
          const std::vector<TensorCaseBuffer> &inputs, std::size_t inputIndex,
          const std::vector<std::uint8_t> &expected
      ) {
        for (const std::int32_t replacement: {1, 4}) {
          auto candidate = inputs;
          auto &bytes = candidate[inputIndex].bytes;
          const std::uint32_t word = candidate[inputIndex].element == TensorElement::I32
                                         ? std::bit_cast<std::uint32_t>(replacement)
                                         : std::bit_cast<std::uint32_t>(
                                               static_cast<float>(replacement)
                                           );
          for (std::size_t offset = 0; offset + 4 <= bytes.size(); offset += 4)
            std::memcpy(bytes.data() + offset, &word, sizeof(word));
          if (replay(candidate, nullptr) != expected)
            return replacement;
        }
        return std::nullopt;
      }

      Outcome timeoutOutcome(const std::string &reason) {
        Outcome outcome;
        outcome.status = "timeout";
        outcome.reason = reason;
        return outcome;
      }

      /// Replay the proposed inputs through the shared interpreter, collecting
      /// the blocks it enters in order so repeated visits stay visible.
      std::vector<std::uint8_t> replay(
          const std::vector<TensorCaseBuffer> &inputs, std::vector<std::string> *visited
      ) {
        const auto &entry = *entryFunction(*program_);
        const auto resultType = typeInfo(entry.retType);
        // The transport is the one the case writer uses: each buffer's bytes in
        // declaration order, decoded through the shared reader.
        std::vector<std::uint8_t> bytes;
        for (const auto &input: inputs)
          bytes.insert(bytes.end(), input.bytes.begin(), input.bytes.end());
        std::string binary;
        if (!bytes.empty())
          binary.assign(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        std::vector<TensorTypeInfo> types;
        for (const auto &param: entry.params)
          types.push_back(typeInfo(param.type));
        std::istringstream stream(binary, std::ios::binary);
        auto args = readTensorArguments(stream, types);

        std::ostringstream output;
        Interpreter interpreter(*program_, output);
        const std::string entryName = entry.name.name;
        interpreter.setStateHook(
            [&visited, &entryName](
                const std::string &funcName, std::uint64_t, const std::string &label, int instrIdx,
                const Store &, const Memory &
            ) {
              if (funcName == entryName && instrIdx == -1 && visited)
                visited->push_back(label);
            }
        );
        setTensorReferenceBits(128);
        std::vector<std::uint8_t> expected;
        try {
          const auto result = interpreter.runTyped(entry.name.name, {}, args);
          setTensorTraceHook({});
          if (!result)
            throw std::runtime_error("the replayed entry did not return a value");
          std::ostringstream encoded(std::ios::binary);
          writeTensorResult(encoded, *result, resultType);
          const auto value = encoded.str();
          expected.assign(value.begin(), value.end());
        } catch (...) {
          setTensorTraceHook({});
          throw;
        }
        return expected;
      }

      std::filesystem::path requestPath_;
      std::filesystem::path responsePath_;
      std::filesystem::path stagePath_;
      std::unique_ptr<Program> program_;
      std::string source_;
      std::string digest_;
      std::string failure_;
      std::string stage_ = "query";
      std::vector<std::string> path_;
      TensorInputPolicyOptions options_;
      TensorElement element_ = TensorElement::I32;
      Shape outputShape_;
      std::pair<std::int64_t, std::int64_t> declaredDomain_{0, 0};
      std::uint32_t budgetMs_ = 5000;
      std::chrono::steady_clock::time_point started_;
    };

    // -------------------------------------------------------------- parent

    /// The worker's own last words, so a failure before the response is
    /// written is diagnosable rather than silent.
    std::string workerTail(const std::filesystem::path &logPath) {
      std::ifstream in(logPath, std::ios::binary);
      if (!in)
        return {};
      std::string text;
      std::getline(in, text);
      if (text.empty())
        return {};
      if (text.size() > 200)
        text.resize(200);
      return ": " + text;
    }

    /// Read the interval a response reports. A missing or malformed field is
    /// reported as such rather than skipped: an unreadable domain is a
    /// malformed response, and accepting it would mean accepting an answer
    /// whose interval was never established.
    bool readDomain(const Record &response, std::pair<std::int64_t, std::int64_t> &domain) {
      const auto field = response.get("domain");
      if (!field)
        return false;
      const auto comma = field->find(',');
      if (comma == std::string::npos || comma == 0 || comma + 1 == field->size())
        return false;
      try {
        domain = std::pair{
            std::stoll(field->substr(0, comma)), std::stoll(field->substr(comma + 1))
        };
      } catch (const std::exception &) {
        return false;
      }
      return true;
    }

    /// Whether every leaf of a payload lies inside `domain`. The transport is
    /// raw little-endian elements, so the check reads them as the element type
    /// declares rather than trusting the worker's claim about them.
    bool leavesWithinDomain(
        const std::vector<std::uint8_t> &bytes, TensorElement element,
        const std::pair<std::int64_t, std::int64_t> &domain
    ) {
      for (std::size_t offset = 0; offset + 4 <= bytes.size(); offset += 4) {
        std::uint32_t word = 0;
        std::memcpy(&word, bytes.data() + offset, sizeof(word));
        if (element == TensorElement::I32) {
          const auto value = static_cast<std::int64_t>(std::bit_cast<std::int32_t>(word));
          if (value < domain.first || value > domain.second)
            return false;
        } else {
          const auto value = static_cast<double>(std::bit_cast<float>(word));
          if (!(value >= static_cast<double>(domain.first) &&
                value <= static_cast<double>(domain.second)))
            return false;
        }
      }
      return true;
    }

    struct Child {
      pid_t pid = -1;
      bool reaped = false;
      int status = 0;
    };

    /// Signal the child's process group. The group is signalled rather than
    /// the process, so a backend or helper the worker started cannot outlive
    /// the selection; this is repeated on every terminal path, because a
    /// descendant keeps the group alive after its leader has been reaped.
    void signalGroup(pid_t pid) {
      if (pid > 0) {
        ::kill(-pid, SIGKILL);
        ::kill(pid, SIGKILL);
      }
    }

    /// Terminate the child's group and reap the child. `cleanup` folds in the
    /// group signal so no terminal path can forget it.
    void terminateChild(Child &child) {
      signalGroup(child.pid);
      if (child.pid > 0 && !child.reaped) {
        while (::waitpid(child.pid, &child.status, 0) < 0 && errno == EINTR) {
        }
        child.reaped = true;
      }
    }

  } // namespace

  int serveTensorSelectionRequest(
      const std::filesystem::path &requestPath, const std::filesystem::path &responsePath
  ) {
    // The stage file sits beside the response so both are written where the
    // supervisor already looks.
    const auto stagePath = responsePath.string() + ".stage";
    Worker worker(requestPath, responsePath, stagePath);
    return worker.run();
  }

  TensorSelectionResult selectAndReplayTensorInputs(
      const Program &program, const TensorSelectionOptions &options,
      const std::filesystem::path &workerExecutable
  ) {
    if (options.timeoutMs == 0)
      throw std::invalid_argument("a selection deadline must be positive");
    TensorSelectionResult result;
    const auto started = std::chrono::steady_clock::now();
    const auto elapsedUs = [&started] {
      return static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - started
          )
              .count()
      );
    };
    const auto finish = [&]() -> TensorSelectionResult & {
      result.elapsedUs = elapsedUs();
      return result;
    };

    // The domain a witness has to lie in is resolved from the policy when the
    // caller states none, so the policy default is as binding as an explicit
    // interval and a response cannot widen it.
    const auto effectiveDomain =
        options.policy.domain.value_or(tensorPolicyDomain(options.policy.policy));
    const auto signature = tensorInputSignature(program);
    const auto requestedSymbolicElements = [&signature] {
      std::uint32_t total = 0;
      for (const auto &input: signature)
        total += input.elements;
      return total;
    }();
    // Recorded before anything can return, so every terminal outcome names the
    // graph, the interval, and the size the request was about.
    const auto digest = computationDigest(program);
    result.proposal.computationDigest = digest;
    result.proposal.domain = effectiveDomain;
    result.proposal.symbolicElements = requestedSymbolicElements;

    if (workerExecutable.empty() || !std::filesystem::exists(workerExecutable)) {
      result.proposal.outcome = "invalid";
      result.proposal.reason = "the selection worker executable is unavailable";
      return finish();
    }

    // Everything below happens inside the deadline, the request file included:
    // writing it is part of the interval the contract covers.
    std::error_code error;
    const auto directory =
        std::filesystem::temp_directory_path(error) /
        ("refractir-selection-" + std::to_string(::getpid()) + "-" +
         std::to_string(elapsedUs()));
    if (error || !std::filesystem::create_directories(directory)) {
      result.proposal.outcome = "invalid";
      result.proposal.reason = "cannot create a selection working directory";
      return finish();
    }
    const auto cleanup = [&directory] {
      std::error_code ignored;
      std::filesystem::remove_all(directory, ignored);
    };
    const auto requestPath = directory / "request.txt";
    const auto responsePath = directory / "response.txt";
    const auto stagePath = responsePath.string() + ".stage";
    const auto logPath = directory / "worker.log";

    Record request;
    request.add("schema", std::string(REQUEST_SCHEMA));
    request.add("program", hexEncode(printProgram(program)));
    const auto output = tensorOutputSignature(program);
    request.add("profile", profileName(options.policy.profileVersion));
    request.add("dtype", elementName(output.element));
    request.add("policy", std::string(tensorInputPolicyName(options.policy.policy)));
    request.add("path", joinLabels(options.policy.path));
    request.add(
        "domain",
        std::to_string(effectiveDomain.first) + "," + std::to_string(effectiveDomain.second)
    );
    request.add("property", propertyText(options.policy.property));
    for (const auto &element: options.policy.property.elements) {
      std::string index;
      for (std::size_t axis = 0; axis < element.index.size(); ++axis) {
        if (axis)
          index += ',';
        index += std::to_string(element.index[axis]);
      }
      request.add("element", index + "=" + std::to_string(element.value));
    }
    request.add("seed", std::to_string(options.policy.seed));
    request.add("timeout_ms", std::to_string(options.timeoutMs));
    try {
      writeRecord(requestPath, request);
    } catch (const std::exception &failure) {
      result.proposal.outcome = "invalid";
      result.proposal.reason = failure.what();
      cleanup();
      return finish();
    }

    const auto spawn = [&]() -> Child {
      Child child;
      std::fflush(nullptr);
      child.pid = ::fork();
      if (child.pid == 0) {
        ::setpgid(0, 0);
        const int log = ::open(logPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (log >= 0) {
          ::dup2(log, STDOUT_FILENO);
          ::dup2(log, STDERR_FILENO);
          if (log > STDERR_FILENO)
            ::close(log);
        }
        const auto worker = workerExecutable.string();
        const auto request = requestPath.string();
        const auto response = responsePath.string();
        ::execl(
            worker.c_str(), worker.c_str(), "--tensor-selection-worker", request.c_str(),
            response.c_str(), nullptr
        );
        _exit(127);
      }
      if (child.pid > 0) {
        // Both sides set the group, so the signal reaches it whichever runs
        // first.
        ::setpgid(child.pid, child.pid);
      }
      return child;
    };

    Child child = spawn();
    if (child.pid < 0) {
      result.proposal.outcome = "invalid";
      result.proposal.reason = "cannot start the selection worker";
      cleanup();
      return finish();
    }

    // The parent owns the deadline. The loop waits rather than blocking, so
    // expiry is decided by this clock and not by how long the child runs.
    const auto budgetUs = static_cast<std::uint64_t>(options.timeoutMs) * 1000;
    // Every later decision asks the clock, so validation time cannot push a
    // result past the interval it was supposed to fit in.
    const auto pastDeadline = [&] { return elapsedUs() >= budgetUs; };
    bool expired = false;
    while (true) {
      const auto waited = ::waitpid(child.pid, &child.status, WNOHANG);
      if (waited == child.pid) {
        child.reaped = true;
        // A result observed at or after the deadline is late, however promptly
        // the worker produced it: the interval covers receipt and validation,
        // so the clock decides rather than the moment the child exited.
        if (elapsedUs() >= budgetUs)
          expired = true;
        break;
      }
      if (waited < 0 && errno != EINTR)
        break;
      if (elapsedUs() >= budgetUs) {
        expired = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (expired) {
      const auto beforeCleanup = elapsedUs();
      terminateChild(child);
      result.cleanupUs = elapsedUs() - beforeCleanup;
      result.expired = true;
      // The worker's last notification names the stage. Without one the stage
      // stays unknown rather than being guessed.
      result.timeoutStage = "unknown_stage";
      if (const auto stage = readRecord(stagePath)) {
        if (const auto value = stage->get("stage"))
          result.timeoutStage = *value;
      }
      result.proposal.outcome = "timeout";
      result.proposal.timeoutSource = "selection_deadline";
      result.proposal.reason = "the selection deadline expired during " + result.timeoutStage;
      cleanup();
      return finish();
    }
    terminateChild(child);
    if (!WIFEXITED(child.status) || WEXITSTATUS(child.status) != 0) {
      result.proposal.outcome = "invalid";
      result.proposal.reason = "the selection worker did not exit cleanly" + workerTail(logPath);
      cleanup();
      return finish();
    }

    const auto response = readRecord(responsePath);
    if (!response) {
      result.proposal.outcome = "invalid";
      result.proposal.reason =
          "the selection worker produced no readable response" + workerTail(logPath);
      cleanup();
      return finish();
    }
    if (const auto schema = response->get("schema"); !schema || *schema != RESPONSE_SCHEMA) {
      result.proposal.outcome = "invalid";
      result.proposal.reason = "the selection response schema is unknown";
      cleanup();
      return finish();
    }

    const auto status = response->get("status").value_or("invalid");
    result.proposal.reason = response->get("reason").value_or("");
    if (status != "accepted") {
      result.proposal.outcome = status == "timeout" ? "timeout" : status;
      if (status == "timeout")
        result.proposal.timeoutSource = "solver";
      result.proposal.unknownReason = response->get("unknown_reason").value_or("");
      // A refusal is about the request the parent made, so its own identity,
      // interval, and size stand. A worker that refused before solving reports
      // none of them, and one that reports a wider interval is describing
      // something the request never asked about.
      result.proposal.computationDigest = digest;
      result.proposal.domain = effectiveDomain;
      result.proposal.symbolicElements = requestedSymbolicElements;
      if (const auto symbolic = response->get("symbolic_elements")) {
        const auto reported = static_cast<std::uint32_t>(std::stoul(*symbolic));
        if (reported != 0)
          result.proposal.symbolicElements = reported;
      }
      cleanup();
      return finish();
    }

    // A response is accepted only when it agrees with the program the parent
    // sent: same identity, same parameters, same path, and a payload whose
    // length matches the declaration. None of this needs a replay, so the
    // parent never redoes the worker's unbounded work.
    const auto reject = [&](const std::string &reason) -> TensorSelectionResult & {
      result.proposal.outcome = "invalid";
      result.proposal.reason = reason;
      result.replayValidated = false;
      // The identity, domain, and size of the refused request stay in place;
      // only the response's own claims are dropped.
      result.proposal.computationDigest = digest;
      result.proposal.domain = effectiveDomain;
      result.proposal.symbolicElements = requestedSymbolicElements;
      cleanup();
      return finish();
    };
    if (pastDeadline()) {
      result.expired = true;
      result.timeoutStage = "response";
      result.proposal.outcome = "timeout";
      result.proposal.timeoutSource = "selection_deadline";
      result.proposal.reason = "the selection deadline expired during response";
      cleanup();
      return finish();
    }
    if (response->get("digest").value_or("") != digest)
      return reject("the selection response describes a different program");
    if (response->get("replay").value_or("") != joinLabels(options.policy.path))
      return reject("the selection response replayed a different path");
    if (response->get("property").value_or("") != "holds")
      return reject("the selection response does not report the stated property");

    auto reportedDomain = effectiveDomain;
    if (!readDomain(*response, reportedDomain))
      return reject("the selection response does not state a readable domain");
    const auto names = response->all("param");
    const auto elements = response->all("param_element");
    const auto shapes = response->all("param_shape");
    const auto payloads = response->all("param_bytes");
    if (names.size() != signature.size() || elements.size() != names.size() ||
        shapes.size() != names.size() || payloads.size() != names.size())
      return reject("the selection response parameter count differs from the signature");
    std::vector<TensorCaseBuffer> buffers;
    for (std::size_t index = 0; index < signature.size(); ++index) {
      if (names[index] != signature[index].name)
        return reject("the selection response names parameters out of order");
      const auto element = elementFromName(elements[index]);
      if (!element || *element != signature[index].element)
        return reject("the selection response changes a parameter's element type");
      const auto bytes = hexDecode(payloads[index]);
      if (!bytes)
        return reject("the selection response carries a malformed parameter payload");
      if (bytes->size() != static_cast<std::size_t>(signature[index].elements) *
                                 tensorElementBytes(signature[index].element))
        return reject("the selection response parameter length differs from the declaration");
      // The request states the domain the witness has to lie in, so a leaf
      // outside it is not a witness however the response describes itself.
      if (reportedDomain != effectiveDomain)
        return reject("the selection response reports a domain the request did not state");
      if (!leavesWithinDomain(*bytes, signature[index].element, reportedDomain))
        return reject("the selection response carries a leaf outside the stated domain");
      buffers.push_back(TensorCaseBuffer{
          signature[index].name, signature[index].element, signature[index].shape, *bytes
      });
    }

    const auto expectedHex = response->get("expected");
    if (!expectedHex)
      return reject("the selection response carries no expected output");
    const auto expected = hexDecode(*expectedHex);
    if (!expected)
      return reject("the selection response expected output is malformed");
    if (expected->size() !=
        static_cast<std::size_t>(output.elements) * tensorElementBytes(output.element))
      return reject("the selection response output length differs from the declaration");
    // The property is a claim about these bytes, and checking it is bounded
    // work on what the worker returned rather than another replay.
    if (!tensorPropertyHolds(*expected, output.element, output.shape, options.policy.property))
      return reject("the returned output does not satisfy the stated property");

    // The worker checked that the returned value depends on each input; the
    // parent only checks that the evidence covers every input it declared.
    const auto dependenceNames = response->all("dependence");
    const auto dependenceValues = response->all("dependence_value");
    if (dependenceNames.size() != dependenceValues.size())
      return reject("the selection response dependence evidence is malformed");
    const bool needsDependence =
        options.policy.profileVersion == TensorProfileId::V2 && signature.size() > 1;
    // Counted before it is indexed, in every profile: the profile that does not
    // use the evidence is exactly the one where an unexpected record would
    // otherwise reach past the signature.
    if (dependenceNames.size() > signature.size())
      return reject("the selection response carries dependence evidence for more inputs than exist");
    if (!needsDependence && !dependenceNames.empty())
      return reject("the selection response carries dependence evidence this profile does not use");
    if (needsDependence && dependenceNames.size() != signature.size())
      return reject("the selection response carries no dependence evidence");
    for (std::size_t index = 0; index < dependenceNames.size(); ++index) {
      if (dependenceNames[index] != signature[index].name)
        return reject("the selection response dependence evidence names inputs out of order");
    }

    // The deadline covers validation too, so the clock is asked once more
    // immediately before a result becomes an acceptance.
    if (pastDeadline()) {
      result.expired = true;
      result.timeoutStage = "response";
      result.proposal.outcome = "timeout";
      result.proposal.timeoutSource = "selection_deadline";
      result.proposal.reason = "the selection deadline expired during response";
      cleanup();
      return finish();
    }

    result.proposal.outcome = "accepted";
    result.proposal.computationDigest = digest;
    result.proposal.inputs = std::move(buffers);
    for (std::size_t index = 0; index < dependenceNames.size(); ++index) {
      result.inputCounterfactualWitnesses.emplace_back(
          dependenceNames[index], static_cast<std::int32_t>(std::stoll(dependenceValues[index]))
      );
    }
    result.proposal.domain = reportedDomain;
    if (const auto symbolic = response->get("symbolic_elements"))
      result.proposal.symbolicElements = static_cast<std::uint32_t>(std::stoul(*symbolic));
    result.expected = *expected;
    result.replayPath = splitLabels(response->get("replay").value_or(""));
    result.replayValidated = true;
    cleanup();
    return finish();
  }

} // namespace refractir::reify
