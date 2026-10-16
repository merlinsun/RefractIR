#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "analysis/tensor.hpp"
#include "ast/ast.hpp"
#include "reify/tensor_case.hpp"
#include "solver/solver.hpp"

namespace refractir::reify {

  /// How one frozen program's runtime inputs are proposed.
  ///
  /// `Ordinary` and `Boundary` sample values directly. `Exact` hands a stated
  /// property to the shared symbolic executor and keeps the solver's witness.
  /// The driver's existing construction route (`--tensor-synthesis
  /// construction`) stays available beside them; it differs from `Ordinary`
  /// only in how the values are drawn, not in who accepts them.
  enum class TensorInputPolicy { Ordinary, Boundary, Exact };

  TensorInputPolicy parseTensorInputPolicy(std::string_view name);
  std::string_view tensorInputPolicyName(TensorInputPolicy policy);
  /// Stable algorithm identifier for the boundary sampler; ordinary sampling
  /// keeps its existing contract, and exact selection is identified by its
  /// constraints instead.
  std::string_view tensorInputSamplerVersion(TensorInputPolicy policy);

  /// Inclusive envelope of the values a policy can propose for one leaf.
  ///
  /// `Ordinary` draws from `[1, 2]`; `Boundary` and `Exact` both span
  /// `[-1, 2]`, the boundary sampler by drawing each leaf from that interval,
  /// and the exact route by bounding each leaf to the same interval. A target
  /// either sampler can witness is therefore never outside the exact route's
  /// domain.
  /// The intervals are not identical across policies, and every case records
  /// the one it used, so a comparison can state which values it compared.
  std::pair<std::int64_t, std::int64_t> tensorPolicyDomain(TensorInputPolicy policy);

  /// A property chosen before any input is accepted. `None` is the admissibility
  /// property every policy already carries: the inputs must be defined, finite
  /// when floating, and produce a defined reference result.
  struct TensorInputProperty {
    enum class Kind {
      None,
      /// The scalar return value equals `target`.
      OutputEquals,
      /// The element sum of a rank-one tensor return value equals `target`.
      OutputSumEquals,
      /// Named elements of the returned tensor equal their required values.
      /// `complete` records whether the constraints cover every coordinate;
      /// the unconstrained coordinates of a partial request take whatever the
      /// reference computes.
      ElementsEqual,
    };
    Kind kind = Kind::None;
    std::int64_t target = 0;
    std::vector<TensorOutputElementConstraint> elements;
  };

  /// Parse one `--tensor-output-element` argument: a comma-separated coordinate
  /// list, `=`, and an i32 value (`0,1=-1`). Throws std::invalid_argument when
  /// the text is malformed.
  TensorOutputElementConstraint parseTensorOutputElement(std::string_view text);

  /// What one runtime input has to look like: the name a case binds, its
  /// element type, and its element count. A supervisor derives this from the
  /// program it sent, so a worker's payload lengths are checked against the
  /// declaration rather than against the worker's own description of them.
  struct TensorInputSignature {
    std::string name;
    TensorElement element = TensorElement::I32;
    Shape shape;
    std::uint32_t elements = 0;
  };

  /// The output the entry function returns, in the same terms.
  struct TensorOutputSignature {
    TensorElement element = TensorElement::I32;
    Shape shape;
    std::uint32_t elements = 0;
  };

  std::vector<TensorInputSignature> tensorInputSignature(const Program &program);
  TensorOutputSignature tensorOutputSignature(const Program &program);

  /// Bytes one element occupies in the runtime transport.
  std::size_t tensorElementBytes(TensorElement element);

  /// Order and check element constraints against a declared output shape:
  /// row-major lexicographic order for deterministic serialization, every
  /// coordinate the full output rank and in bounds, no coordinate twice. Sets
  /// `complete` when the constraints cover every element exactly once.
  /// Returns false with `reason` set when a constraint cannot be admitted.
  bool checkTensorOutputElements(
      const Shape &shape, std::vector<TensorOutputElementConstraint> &elements, bool &complete,
      std::string &reason
  );

  struct TensorInputPolicyOptions {
    TensorInputPolicy policy = TensorInputPolicy::Ordinary;
    TensorInputProperty property;
    /// Profile the property query is checked under before it is solved.
    TensorProfileId profileVersion = TensorProfileId::V1;
    std::uint32_t seed = 0;
    /// Explicit solver path over the frozen program's block labels, repeats
    /// included. An empty path is accepted only for a single-block program,
    /// whose path is then that block; a loop or a diamond needs the caller to
    /// state the visits.
    std::vector<std::string> path;
    /// Optional bound asserted on every entry-parameter integer leaf.
    std::optional<std::pair<std::int64_t, std::int64_t>> domain;
    /// The exact i32 route refuses more symbolic input elements than this, so
    /// an oversized graph is reported as out of budget rather than solved
    /// slowly.
    std::uint32_t maxSymbolicElements = 64;
    std::uint32_t timeoutMs = 5000;
  };

  struct TensorInputProposal {
    TensorInputPolicy policy = TensorInputPolicy::Ordinary;
    TensorInputProperty property;
    /// Stable sampler identifier, empty when the policy has no sampler version.
    std::string samplerVersion;
    /// `accepted`, or the first reason the policy could not propose:
    /// `unsupported`, `invalid`, `budget_exceeded`, `unsat`, `unknown`,
    /// `timeout`. These are never conflated: an infeasible property is not a
    /// timeout, and a refused graph is not an unsatisfiable one.
    std::string outcome;
    std::string reason;
    /// Identity of the frozen computation with every query constraint
    /// removed. `queryDigest` covers the property-constrained clone; the two
    /// are equal exactly when the clone added constraints and nothing else.
    std::string computationDigest;
    std::string queryDigest;
    std::uint32_t symbolicElements = 0;
    /// Inclusive bound the proposal's leaf values respect. A sampling policy
    /// reports the envelope it draws from; the exact policy reports the bound
    /// it asserted, which defaults to `tensorPolicyDomain(Exact)`.
    std::pair<std::int64_t, std::int64_t> domain;
    /// Set only for a `timeout` outcome, naming what enforced the limit:
    /// `solver` when the backend reported its own timeout. The selection
    /// deadline of a supervising caller is a different source.
    std::string timeoutSource;
    /// The backend's typed reason for an undecided result, empty when the
    /// result was decided. `unspecified` means the backend did not say.
    std::string unknownReason;
    std::vector<TensorCaseBuffer> inputs;
  };

  /// Propose runtime inputs for `program` under `options.policy`.
  TensorInputProposal proposeTensorInputs(
      const Program &program, const SymbolicExecutor::SolverFactory &solverFactory,
      const TensorInputPolicyOptions &options
  );

  /// Identity of a program's computation: its printed form with every
  /// `assume`/`require` instruction and every `__query_`-prefixed local
  /// removed. Two programs with the same digest differ only in query
  /// constraints, so a witness solved on one applies to the other.
  std::string computationDigest(const Program &program);

  /// Whether `output` satisfies `property` for the declared element type and
  /// logical shape. A caller replays a proposal through the interpreter and
  /// then runs this check, so no proposal is accepted on the solver's word.
  bool tensorPropertyHolds(
      const std::vector<std::uint8_t> &output, TensorElement element, const Shape &shape,
      const TensorInputProperty &property
  );

} // namespace refractir::reify
