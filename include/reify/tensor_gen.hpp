#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "analysis/tensor.hpp"

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
