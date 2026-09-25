#include "frontend/pipeline.hpp"

#include <memory>
#include <utility>

#include "analysis/definite_init.hpp"
#include "analysis/pass_manager.hpp"
#include "analysis/reachability.hpp"
#include "analysis/reducibility.hpp"
#include "analysis/unused_name.hpp"
#include "frontend/lexer.hpp"
#include "frontend/parser.hpp"
#include "frontend/semchecker.hpp"
#include "frontend/typechecker.hpp"

namespace refractir {

  Program parseSource(std::string_view src) {
    Lexer lx(src);
    Parser ps(lx.lexAll());
    return ps.parseProgram();
  }

  bool checkProgram(Program &prog, DiagBag &diags, const CheckOptions &opts) {
    PassManager pm(diags);
    pm.addModulePass(std::make_unique<SemChecker>());
    pm.addModulePass(std::make_unique<TypeChecker>(tensorProfile(opts.tensorProfile)));
    if (opts.functionAnalyses) {
      pm.addFunctionPass(std::make_unique<ReachabilityAnalysis>());
      pm.addFunctionPass(std::make_unique<DefiniteInitAnalysis>());
      pm.addFunctionPass(std::make_unique<UnusedNameAnalysis>());
    }
    if (opts.reducibility)
      pm.addFunctionPass(std::make_unique<ReducibilityCheck>());

    if (pm.run(prog) == PassResult::Error)
      return false;
    return !(opts.warningsAreErrors && diags.hasWarnings());
  }

} // namespace refractir
