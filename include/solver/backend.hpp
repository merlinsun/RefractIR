#pragma once

#include <memory>
#include <stdexcept>

#include "solver/solver.hpp"

#if defined(USE_BITWUZLA)
#include "solver/bitwuzla_impl.hpp"
#elif defined(USE_ALIVESMT)
#include "solver/alive_impl.hpp"
#endif

namespace refractir::solver {

  /// The backend this build is configured with, constructed from the
  /// executor's own timeout and seed. A caller that needs a bounded query
  /// builds a `Config` with the limit it wants and passes that here.
  [[nodiscard]] inline SymbolicExecutor::SolverFactory configuredSolverFactory() {
    return [](const SymbolicExecutor::Config &config) -> std::unique_ptr<smt::ISolver> {
#if defined(USE_BITWUZLA)
      return std::make_unique<BitwuzlaSolver>(
          config.timeout_ms, config.seed, config.num_smt_threads
      );
#elif defined(USE_ALIVESMT)
      return std::make_unique<AliveSolver>(
          config.timeout_ms, config.seed, config.num_smt_threads
      );
#else
      (void) config;
      throw std::runtime_error("No solver backend compiled in");
#endif
    };
  }

} // namespace refractir::solver
