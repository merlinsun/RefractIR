// A controlled backend for the solver-outcome classification tests.
//
// Timing a real solver to produce a timeout is not a reliable regression: a
// query that happens to be slow on one machine is fast on another, and a
// query that is genuinely hard is a different test every time. This driver
// injects a backend that returns whatever the test asks for, so the mapping
// from a backend's report to the executor's `unknownReason` is deterministic.
//
// It also exercises the diagnostic-string mapping directly, including a string
// no backend emits, which is the case that must stay Unspecified, and the input
// policy's mapping from a typed reason to a proposal outcome.
//
// Usage:
//   solver_outcome_probe <check-sat-result> <reason-name>
//     check-sat-result: sat | unsat | unknown
//     reason-name:      unspecified | timeout | resource_limit | interrupted
//   solver_outcome_probe --classify <diagnostic>...
//   solver_outcome_probe --propose <check-sat-result> <reason-name>
//   solver_outcome_probe --sample <threads> <samples> <reason-name>
//   solver_outcome_probe --attempt-record <check-sat-result> <reason-name> <directory>
//
// It prints one line per observation: `status=<sat|unsat|unknown>
// reason=<name>` for a solve, `classify <text> -> <name>` for each diagnostic,
// `outcome=<name> timeout_source=<name> unknown_reason=<name>` for a proposal,
// and `reason=<name>` for a sampling run. The attempt-record mode writes the
// generation summary into the given directory and prints its path. Exit status
// is 0 unless the invocation itself is malformed.

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "frontend/pipeline.hpp"
#include "reify/tensor_gen.hpp"
#include "reify/tensor_input_policy.hpp"
#include "solver/solver.hpp"

namespace {

  using refractir::smt::Result;
  using refractir::smt::Sort;
  using refractir::smt::Term;
  using refractir::smt::UnknownReason;

  UnknownReason parseReason(const std::string &name) {
    if (name == "unspecified")
      return UnknownReason::Unspecified;
    if (name == "timeout")
      return UnknownReason::Timeout;
    if (name == "resource_limit")
      return UnknownReason::ResourceLimit;
    if (name == "interrupted")
      return UnknownReason::Interrupted;
    throw std::invalid_argument("unknown reason name: " + name);
  }

  Result parseResult(const std::string &name) {
    if (name == "sat")
      return Result::SAT;
    if (name == "unsat")
      return Result::UNSAT;
    if (name == "unknown")
      return Result::UNKNOWN;
    throw std::invalid_argument("unknown check-sat result: " + name);
  }

  /// A backend that answers exactly as told and reports exactly the reason it
  /// was given. Every other operation is a placeholder: the executor only
  /// needs the terms and sorts it builds to stay distinguishable.
  class ScriptedSolver : public refractir::smt::ISolver {
  public:
    ScriptedSolver(Result result, UnknownReason reason) : result_(result), reason_(reason) {}

    UnknownReason unknownReason() const override { return reason_; }

    Sort make_bv_sort(uint32_t) override { return Sort{std::make_shared<int>(1)}; }
    Sort make_fp_sort(uint32_t, uint32_t) override { return Sort{std::make_shared<int>(2)}; }
    Sort make_bool_sort() override { return Sort{std::make_shared<int>(3)}; }

    bool is_bv_sort(Sort s) override { return s.internal && *static_cast<int *>(s.internal.get()) == 1; }
    bool is_fp_sort(Sort s) override { return s.internal && *static_cast<int *>(s.internal.get()) == 2; }
    bool is_bool_sort(Sort s) override { return s.internal && *static_cast<int *>(s.internal.get()) == 3; }
    uint32_t get_bv_width(Sort) override { return 32; }
    std::pair<uint32_t, uint32_t> get_fp_dims(Sort) override { return {8, 24}; }

    Term make_true() override { return term(1); }
    Term make_false() override { return term(0); }
    Term make_bv_value(Sort, const std::string &, uint8_t) override { return term(0); }
    Term make_bv_value_uint64(Sort, uint64_t) override { return term(0); }
    Term make_bv_value_int64(Sort, int64_t) override { return term(0); }
    Term make_bv_zero(Sort) override { return term(0); }
    Term make_bv_one(Sort) override { return term(0); }
    Term make_bv_min_signed(Sort) override { return term(0); }
    Term make_bv_max_signed(Sort) override { return term(0); }

    Term make_fp_value(Sort, const std::string &, refractir::smt::RoundingMode) override {
      return term(0);
    }
    Term make_fp_value_from_real(Sort, double, refractir::smt::RoundingMode) override {
      return term(0);
    }
    Term make_rm_value(refractir::smt::RoundingMode) override { return term(0); }

    Term make_const(Sort, const std::string &) override { return term(0); }

    Term make_term(
        refractir::smt::Kind, const std::vector<Term> &, const std::vector<uint32_t> & = {}
    ) override {
      return term(0);
    }

    Sort get_sort(Term) override { return make_bv_sort(32); }
    bool is_true(Term) override { return false; }
    bool is_false(Term) override { return false; }

    void assert_formula(Term) override {}
    Result check_sat() override { return result_; }

    Term get_value(Term t) override { return t; }
    std::string get_bv_value_string(Term, uint8_t) override { return "0"; }
    std::string get_fp_value_string(Term) override { return std::string(32, '0'); }

  private:
    static Term term(int tag) { return Term{std::make_shared<int>(tag)}; }

    Result result_;
    UnknownReason reason_;
  };

  /// One small program is enough: the executor's job here is to carry a
  /// backend's undecided result and its reason through to its own Result.
  refractir::Program checked(const std::string &source) {
    refractir::Program program = refractir::parseSource(source);
    refractir::DiagBag diagnostics;
    if (!refractir::checkProgram(program, diagnostics))
      throw std::runtime_error("probe program failed frontend checking");
    return program;
  }

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc >= 3 && std::string(argv[1]) == "--classify") {
      for (int index = 2; index < argc; ++index) {
        std::cout << "classify " << argv[index] << " -> "
                  << refractir::smt::unknownReasonName(
                         refractir::smt::classifyUnknownReason(argv[index])
                     )
                  << "\n";
      }
      return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "--propose") {
      const auto scripted = parseResult(argv[2]);
      const auto reason = parseReason(argv[3]);
      auto program = checked(
          "fun @kernel(%x: tensor [2] i32) : i32 {\n"
          "  let mut %out: i32 = undef;\n"
          "^entry:\n"
          "  %out = tensor sum %x, 0, drop;\n"
          "  ret %out;\n"
          "}\n"
      );
      const auto factory = [scripted, reason](const refractir::SymbolicExecutor::Config &) {
        return std::make_unique<ScriptedSolver>(scripted, reason);
      };
      refractir::reify::TensorInputPolicyOptions options;
      options.policy = refractir::reify::TensorInputPolicy::Exact;
      options.profileVersion = refractir::TensorProfileId::V2;
      options.property.kind = refractir::reify::TensorInputProperty::Kind::OutputEquals;
      options.property.target = 3;
      options.path = {"^entry"};
      const auto proposal =
          refractir::reify::proposeTensorInputs(program, factory, options);
      std::cout << "outcome=" << proposal.outcome
                << " timeout_source="
                << (proposal.timeoutSource.empty() ? "none" : proposal.timeoutSource)
                << " unknown_reason="
                << (proposal.unknownReason.empty() ? "none" : proposal.unknownReason) << "\n";
      return 0;
    }
    if (argc == 5 && std::string(argv[1]) == "--sample") {
      const auto threads = static_cast<std::uint32_t>(std::stoul(argv[2]));
      const auto samples = static_cast<std::uint32_t>(std::stoul(argv[3]));
      const auto reason = parseReason(argv[4]);
      auto program = checked(
          "fun @main(%a: i32) : i32 {\n"
          "  let mut %out: i32 = undef;\n"
          "^entry:\n"
          "  %out = %a;\n"
          "  require %out == 1;\n"
          "  ret %out;\n"
          "}\n"
      );
      // A sample that never runs a query must not publish a default result
      // over one that reported a reason, however the threads interleave.
      const auto factory = [reason](const refractir::SymbolicExecutor::Config &) {
        return std::make_unique<ScriptedSolver>(Result::UNKNOWN, reason);
      };
      refractir::SymbolicExecutor::Config config;
      config.num_threads = threads;
      refractir::SymbolicExecutor executor(program, config, factory);
      const auto result = executor.sample("@main", samples, 0, false, {"^entry"}, {});
      std::cout << "reason=" << refractir::smt::unknownReasonName(result.unknownReason) << "\n";
      return 0;
    }
    if ((argc == 5 || argc == 6) && std::string(argv[1]) == "--attempt-record") {
      const auto scripted = parseResult(argv[2]);
      const auto reason = parseReason(argv[3]);
      const std::filesystem::path directory = argv[4];
      std::filesystem::create_directories(directory);
      // The compositional generator builds the graph from the seed, so the
      // attempt record is the one a normal run would write; only the backend
      // is the scripted one.
      refractir::reify::TensorGenerationOptions options;
      options.profileVersion = refractir::TensorProfileId::V2;
      options.dtype = "i32";
      options.generator = "compositional";
      options.inputPolicy = "exact";
      options.propertyTarget = 3;
      options.count = 1;
      options.seed = argc == 6 ? static_cast<std::uint32_t>(std::stoul(argv[5])) : 2;
      options.target = "sir";
      options.outputDirectory = directory;
      options.solverFactory = [scripted, reason](const refractir::SymbolicExecutor::Config &) {
        return std::make_unique<ScriptedSolver>(scripted, reason);
      };
      const auto generated = refractir::reify::generateTensorCases(options);
      std::cout << "status=" << generated.status << "\n";
      return 0;
    }
    if (argc != 3) {
      std::cerr << "usage: solver_outcome_probe <sat|unsat|unknown> "
                   "<unspecified|timeout|resource_limit|interrupted>\n"
                   "       solver_outcome_probe --classify <diagnostic>...\n"
                   "       solver_outcome_probe --propose <sat|unsat|unknown> "
                   "<unspecified|timeout|resource_limit|interrupted>\n";
      return 2;
    }

    const auto scripted = parseResult(argv[1]);
    const auto reason = parseReason(argv[2]);
    auto program = checked(
        "fun @main(%a: i32) : i32 {\n"
        "  let mut %out: i32 = undef;\n"
        "^entry:\n"
        "  %out = %a;\n"
        "  require %out == 1;\n"
        "  ret %out;\n"
        "}\n"
    );
    const auto factory = [scripted, reason](const refractir::SymbolicExecutor::Config &) {
      return std::make_unique<ScriptedSolver>(scripted, reason);
    };
    refractir::SymbolicExecutor::Config config;
    refractir::SymbolicExecutor executor(program, config, factory);
    const auto result = executor.solve("@main", {"^entry"});

    const char *status = result.sat ? "sat" : result.unsat ? "unsat" : "unknown";
    std::cout << "status=" << status << " reason="
              << refractir::smt::unknownReasonName(result.unknownReason) << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "probe failed: " << error.what() << "\n";
    return 2;
  }
}
