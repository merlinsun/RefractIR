#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "analysis/tensor.hpp"
#include "reify/tensor_case.hpp"
#include "solver/solver.hpp"

namespace refractir::reify {

  struct TensorGenerationOptions {
    TensorProfileId profileVersion = TensorProfileId::V1;
    std::string generator = "constructor";
    // How a compositional case's runtime inputs are proposed: ordinary,
    // boundary, or exact. The constructor route keeps its own strata.
    std::string inputPolicy = "ordinary";
    // Target the exact policy solves for. Stated before any input is
    // accepted, so the case cannot read its property back off a result.
    std::optional<std::int64_t> propertyTarget;
    // Per-element targets, canonicalized once the output shape is known.
    // Mutually exclusive with `propertyTarget`.
    std::vector<TensorOutputElementConstraint> outputElements;
    std::string primaryStratum;
    std::string inputPattern;
    std::string dtype = "i32";
    std::string property;
    std::string synthesis = "construction";
    std::string target = "sir";
    std::uint32_t seed = 0;
    std::uint32_t count = 1;
    std::uint32_t attemptBudget = 100;
    std::filesystem::path outputDirectory;
    std::filesystem::path cvc5Executable;
    std::filesystem::path cvc5FeasibilityRecord;
    /// Replaces the configured solver. A caller that has to reach a particular
    /// solver outcome injects a backend instead of waiting for a real query to
    /// produce it; empty means the configured solver.
    SymbolicExecutor::SolverFactory solverFactory;
    /// The whole selection interval, from the request through the validated
    /// replay response. Positive, and used only when a worker is set.
    std::uint32_t selectionTimeoutMs = 5000;
    /// The executable a supervised selection runs in. A tool sets this to
    /// itself so selection can be terminated without ending the tool; empty
    /// keeps selection in this process.
    std::filesystem::path selectionWorker;
  };

  struct TensorGenerationResult {
    std::uint32_t requested = 0;
    std::uint32_t emitted = 0;
    std::uint32_t rejected = 0;
    std::string status = "complete";
    std::string reason;

    [[nodiscard]] int exitCode() const noexcept { return status == "complete" ? 0 : 1; }
  };

  /// Generate bounded, property-directed tensor cases and their deterministic
  /// campaign summary. A shortfall or unsupported route returns a nonzero code.
  TensorGenerationResult generateTensorCases(const TensorGenerationOptions &options);

} // namespace refractir::reify
