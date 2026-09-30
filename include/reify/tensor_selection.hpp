#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "reify/tensor_input_policy.hpp"

namespace refractir::reify {

  /// The outcome of one supervised selection.
  ///
  /// The proposal alone is not an accepted case: `expected` holds the bytes an
  /// independent replay produced, `replayPath` the blocks that replay entered,
  /// and `replayValidated` says the worker checked the stated property against
  /// them before the budget expired. A caller that publishes a case from this
  /// result does not replay again.
  struct TensorSelectionResult {
    TensorInputProposal proposal;
    std::vector<std::uint8_t> expected;
    std::vector<std::string> replayPath;
    /// One counterfactual witness per input, in declaration order: replacing
    /// that input's leaves changes the output. Empty when the profile does not
    /// require the check. The worker produces it inside the deadline, so the
    /// parent never replays to obtain it.
    std::vector<std::pair<std::string, std::int32_t>> inputCounterfactualWitnesses;
    std::uint64_t elapsedUs = 0;
    bool replayValidated = false;
    /// Where the budget expired: `query`, `solver`, `model`, `replay`,
    /// `response`, or `unknown_stage` when the worker's last notification does
    /// not say. Empty unless the selection timed out.
    std::string timeoutStage;
    /// Time spent terminating and reaping the worker after expiry, measured
    /// separately from the deadline. Cleanup is not a real-time guarantee.
    std::uint64_t cleanupUs = 0;
    /// True when the worker outlived its deadline, whether or not it later
    /// produced a result. A late result is never an accepted case.
    bool expired = false;
  };

  struct TensorSelectionOptions {
    TensorInputPolicyOptions policy;
    /// The whole interval, from request serialization through response
    /// validation. Must be positive.
    std::uint32_t timeoutMs = 5000;
  };

  /// Select inputs and replay them under one deadline.
  ///
  /// The work runs in a supervised child process of `workerExecutable`, which
  /// is started without a shell and terminated as a process group if it
  /// outlives the deadline. The parent validates the response against the
  /// program it sent: lengths, parameter identity, path, and the stated
  /// property, none of which needs a replay.
  TensorSelectionResult selectAndReplayTensorInputs(
      const Program &program, const TensorSelectionOptions &options,
      const std::filesystem::path &workerExecutable
  );

  /// The worker side of the protocol: answer one request file, writing the
  /// response file. Called by the tool's private worker entry point, which is
  /// the executable the supervisor starts.
  int serveTensorSelectionRequest(
      const std::filesystem::path &requestPath, const std::filesystem::path &responsePath
  );

} // namespace refractir::reify
