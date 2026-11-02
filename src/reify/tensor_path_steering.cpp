#include "reify/tensor_path_steering.hpp"

#include <algorithm>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "analysis/cfg.hpp"
#include "analysis/type_utils.hpp"
#include "ast/clone.hpp"
#include "ast/sir_printer.hpp"
#include "frontend/pipeline.hpp"

namespace refractir::reify {

  namespace {

    bool isI32Scalar(const TypePtr &type) {
      return type && !TypeUtils::asTensor(type) && TypeUtils::isScalar(type) &&
             TypeUtils::getIntBitWidth(type) == 32U;
    }

    std::optional<std::string> localName(const Coef &coef) {
      const auto *id = std::get_if<LocalOrSymId>(&coef);
      const auto *local = id ? std::get_if<LocalId>(id) : nullptr;
      return local ? std::optional<std::string>(local->name) : std::nullopt;
    }

    /// Whether an atom reads only literals and the named scalar locals.
    bool atomIsScalar(const Atom &atom, const std::unordered_set<std::string> &scalars) {
      if (const auto *coef = std::get_if<CoefAtom>(&atom.v)) {
        const auto name = localName(coef->coef);
        return !name || scalars.contains(*name);
      }
      if (const auto *read = std::get_if<RValueAtom>(&atom.v))
        return read->rval.accesses.empty() && scalars.contains(read->rval.base.name);
      if (const auto *operation = std::get_if<OpAtom>(&atom.v)) {
        const auto name = localName(operation->coef);
        return (!name || scalars.contains(*name)) && operation->rval.accesses.empty() &&
               scalars.contains(operation->rval.base.name);
      }
      return false;
    }

    bool exprIsScalar(const Expr &expr, const std::unordered_set<std::string> &scalars) {
      return atomIsScalar(expr.first, scalars) &&
             std::all_of(expr.rest.begin(), expr.rest.end(), [&](const Expr::Tail &tail) {
               return atomIsScalar(tail.atom, scalars);
             });
    }

    bool condIsScalar(const Cond &cond, const std::unordered_set<std::string> &scalars) {
      return exprIsScalar(cond.lhs, scalars) && exprIsScalar(cond.rhs, scalars);
    }

    /// The range `assume %name >= lo;` and `assume %name <= hi;` state in the
    /// entry block, if both are present.
    std::optional<std::pair<std::int64_t, std::int64_t>>
    assumedRange(const FunDecl &function, const std::string &name) {
      std::optional<std::int64_t> low;
      std::optional<std::int64_t> high;
      for (const auto &instruction: function.blocks.front().instrs) {
        const auto *assume = std::get_if<AssumeInstr>(&instruction);
        if (!assume || !assume->cond.lhs.rest.empty() || !assume->cond.rhs.rest.empty())
          continue;
        const auto *subject = std::get_if<RValueAtom>(&assume->cond.lhs.first.v);
        const auto *bound = std::get_if<CoefAtom>(&assume->cond.rhs.first.v);
        const auto *literal = bound ? std::get_if<IntLit>(&bound->coef) : nullptr;
        if (!subject || subject->rval.base.name != name || !literal)
          continue;
        if (assume->cond.op == RelOp::GE)
          low = literal->value;
        else if (assume->cond.op == RelOp::LE)
          high = literal->value;
      }
      if (!low || !high)
        return std::nullopt;
      return std::pair{*low, *high};
    }

    const FunDecl &findEntry(const Program &program, const std::string &entry) {
      const auto name = entry.starts_with("@") ? entry : "@" + entry;
      for (const auto &function: program.funs)
        if (function.name.name == name)
          return function;
      throw std::invalid_argument("path steering: entry function not found: " + entry);
    }

    std::string symbolName(const std::string &local) { return "%?" + local.substr(1); }

    /// How a path walk chooses at each decision: the trips every loop runs
    /// each time it is entered, and the arm every branch takes.
    struct Policy {
      std::vector<std::int64_t> trips;
      std::vector<bool> thenArm;
    };

    /// The block labels a walk under `policy` executes, or none when it does
    /// not return within a step bound.
    std::optional<std::vector<std::string>> walk(
        const FunDecl &function, const CFG &cfg, const TensorLoopAnalysis &analysis,
        const Policy &policy
    ) {
      std::unordered_map<std::size_t, std::size_t> loopAtHeader;
      for (std::size_t index = 0; index < analysis.loops.size(); ++index)
        loopAtHeader.emplace(analysis.loops[index].header, index);
      std::unordered_map<std::size_t, std::size_t> branchAt;
      for (std::size_t index = 0; index < analysis.branches.size(); ++index)
        branchAt.emplace(analysis.branches[index].block, index);

      std::vector<std::int64_t> remaining(analysis.loops.size(), 0);
      std::vector<std::string> path;
      std::optional<std::size_t> previous;
      auto block = cfg.entry;
      for (std::size_t steps = 0; steps < 100000; ++steps) {
        path.push_back(function.blocks[block].label.name);
        const auto &term = function.blocks[block].term;
        if (std::holds_alternative<RetTerm>(term))
          return path;
        const auto *branch = std::get_if<BrTerm>(&term);
        if (!branch)
          return std::nullopt;
        std::string next = branch->dest.name;
        if (const auto loop = loopAtHeader.find(block); loop != loopAtHeader.end()) {
          const auto &counted = analysis.loops[loop->second];
          if (previous == counted.preheader)
            remaining[loop->second] = policy.trips[loop->second];
          const bool enter = remaining[loop->second] > 0;
          if (enter)
            --remaining[loop->second];
          next = enter ? branch->thenLabel.name : branch->elseLabel.name;
        } else if (const auto diamond = branchAt.find(block); diamond != branchAt.end()) {
          next = policy.thenArm[diamond->second] ? branch->thenLabel.name : branch->elseLabel.name;
        }
        previous = block;
        block = cfg.indexOf.at(next);
      }
      return std::nullopt;
    }

    std::string startText(const TensorCountedLoop &loop) {
      return loop.startParameter.empty() ? std::to_string(loop.initialValue) : loop.startParameter;
    }

    std::string stopText(const TensorCountedLoop &loop) {
      return loop.boundParameter.empty() ? std::to_string(loop.bound) : loop.boundParameter;
    }

    bool runtimeLoop(const TensorCountedLoop &loop) {
      return !loop.startParameter.empty() || !loop.boundParameter.empty();
    }

    /// A window whose index is a loop's induction variable.
    struct WindowUse {
      std::size_t loop = 0;
      std::uint32_t rows = 0;
      std::uint32_t stride = 0;
      std::uint32_t active = 0;
    };

    std::vector<WindowUse> windowUses(const FunDecl &function, const TensorLoopAnalysis &analysis) {
      std::vector<WindowUse> uses;
      for (const auto &block: function.blocks)
        for (const auto &instruction: block.instrs) {
          const auto *assignment = std::get_if<AssignInstr>(&instruction);
          const auto *atom =
              assignment ? std::get_if<TensorAtom>(&assignment->rhs.first.v) : nullptr;
          if (!atom || atom->op != TensorOp::Window || atom->operands.size() != 2 ||
              atom->attribute.size() != 3)
            continue;
          const auto index = localName(atom->operands[1]);
          for (std::size_t loop = 0; loop < analysis.loops.size(); ++loop)
            if (index && analysis.loops[loop].inductionVariable == *index)
              uses.push_back(
                  WindowUse{loop, atom->attribute[0], atom->attribute[1], atom->attribute[2]}
              );
        }
      return uses;
    }

    /// The scalar skeleton: i32 parameters become ranged symbols copied into
    /// locals of the same name, tensor statements and tensor locals go, the
    /// returned value becomes 0, and each block may gain `require` lines.
    class Skeleton {
    public:
      Skeleton(const FunDecl &function) : function_(function) {
        for (const auto &param: function.params)
          if (isI32Scalar(param.type)) {
            scalars_.insert(param.name.name);
            parameters_.push_back(param.name.name);
          }
        for (const auto &let: function.lets)
          if (isI32Scalar(let.type)) {
            scalars_.insert(let.name.name);
            locals_.push_back(let.name.name);
          }
      }

      /// SIR text for the skeleton with `added[label]` at the end of
      /// that block's instructions.
      std::string
      render(const std::unordered_map<std::string, std::vector<std::string>> &added) const {
        std::ostringstream text;
        text << "fun " << function_.name.name << "() : i32 {\n";
        for (const auto &name: parameters_) {
          text << "  sym " << symbolName(name) << ": value i32";
          if (const auto range = assumedRange(function_, name))
            text << " in [" << range->first << ", " << range->second << "]";
          text << ";\n";
        }
        for (const auto &name: parameters_)
          text << "  let mut " << name << ": i32 = undef;\n";
        for (const auto &name: locals_)
          text << "  let mut " << name << ": i32 = undef;\n";
        for (std::size_t index = 0; index < function_.blocks.size(); ++index) {
          const auto &label = function_.blocks[index].label.name;
          text << label << ":\n";
          if (index == 0)
            for (const auto &name: parameters_)
              text << "  " << name << " = " << symbolName(name) << ";\n";
          if (const auto block = added.find(label); block != added.end())
            for (const auto &condition: block->second)
              text << "  require " << condition << ", \"steering target\";\n";
          text << "  ret 0;\n";
        }
        text << "}\n";
        return splice(text.str());
      }

    private:
      /// Parse the frame (declarations, labels, copies, requires) and move the
      /// original scalar instructions and terminators into it, so every kept
      /// line is the kernel's own, not a re-rendering of it.
      std::string splice(const std::string &frameText) const {
        auto frame = parseSource(frameText);
        auto &skeleton = frame.funs.front();
        for (std::size_t index = 0; index < function_.blocks.size(); ++index) {
          const auto &original = function_.blocks[index];
          auto &block = skeleton.blocks[index];
          // The frame put the parameter copies first and the requires last;
          // the kernel's own lines go between them, in order.
          const auto copies = static_cast<std::ptrdiff_t>(index == 0 ? parameters_.size() : 0);
          std::vector<Instr> instructions;
          for (auto it = block.instrs.begin(); it != block.instrs.begin() + copies; ++it)
            instructions.push_back(std::move(*it));
          for (const auto &instruction: original.instrs)
            if (keeps(instruction))
              instructions.push_back(cloneInstr(instruction));
          for (auto it = block.instrs.begin() + copies; it != block.instrs.end(); ++it)
            instructions.push_back(std::move(*it));
          block.instrs = std::move(instructions);
          if (std::holds_alternative<BrTerm>(original.term))
            block.term = cloneTerminator(original.term);
        }
        std::ostringstream out;
        SIRPrinter printer(out);
        printer.print(frame);
        return out.str();
      }

      bool keeps(const Instr &instruction) const {
        if (const auto *assignment = std::get_if<AssignInstr>(&instruction))
          return assignment->lhs.accesses.empty() && scalars_.contains(assignment->lhs.base.name) &&
                 exprIsScalar(assignment->rhs, scalars_);
        if (const auto *assume = std::get_if<AssumeInstr>(&instruction))
          return condIsScalar(assume->cond, scalars_);
        if (const auto *require = std::get_if<RequireInstr>(&instruction))
          return condIsScalar(require->cond, scalars_);
        return false;
      }

      const FunDecl &function_;
      std::unordered_set<std::string> scalars_;
      std::vector<std::string> parameters_;
      std::vector<std::string> locals_;
    };

    using Requires = std::unordered_map<std::string, std::vector<std::string>>;

  } // namespace

  std::vector<SteeringTarget>
  steeringTargets(const Program &program, const std::string &entry, const TensorProfile &profile) {
    const auto &function = findEntry(program, entry);
    if (function.blocks.empty())
      throw std::invalid_argument("path steering: the entry has no blocks");
    DiagBag diagnostics;
    const auto analysis = analyzeTensorLoops(function, profile, diagnostics);
    if (!analysis)
      throw std::invalid_argument(
          "path steering: " + (diagnostics.diags.empty()
                                   ? std::string("control flow is not in the profile's form")
                                   : diagnostics.diags.front().message)
      );
    DiagBag cfgDiagnostics;
    const auto cfg = CFG::build(function, cfgDiagnostics);
    const Skeleton skeleton(function);

    Policy baseline;
    for (const auto &loop: analysis->loops)
      baseline.trips.push_back(
          runtimeLoop(loop) ? std::min<std::int64_t>(loop.tripCount, 1) : loop.tripCount
      );
    baseline.thenArm.assign(analysis->branches.size(), true);

    std::vector<SteeringTarget> targets;
    const auto add = [&](std::string name, const Policy &policy, const Requires &added) {
      const auto path = walk(function, cfg, *analysis, policy);
      if (!path)
        throw std::invalid_argument("path steering: no returning path for target " + name);
      targets.push_back(SteeringTarget{std::move(name), *path, skeleton.render(added)});
    };

    add("baseline", baseline, {});
    for (std::size_t index = 0; index < analysis->loops.size(); ++index) {
      const auto &loop = analysis->loops[index];
      if (!runtimeLoop(loop))
        continue;
      const auto stem = "loop" + std::to_string(index) + "-";
      const auto &preheader = function.blocks[loop.preheader].label.name;
      const auto start = startText(loop);
      const auto stop = stopText(loop);
      auto policy = baseline;
      policy.trips[index] = 0;
      add(stem + "empty-equal", policy, {{preheader, {start + " == " + stop}}});
      add(stem + "empty-inverted", policy,
          {{preheader, {start + (loop.step > 0 ? " > " : " < ") + stop}}});
      policy.trips[index] = 1;
      add(stem + "one", policy, {});
      if (loop.tripCount > 1) {
        policy.trips[index] = loop.tripCount;
        add(stem + "most", policy, {});
      }
    }
    for (std::size_t index = 0; index < analysis->branches.size(); ++index) {
      const auto stem = "branch" + std::to_string(index) + "-";
      auto policy = baseline;
      for (const bool arm: {true, false}) {
        policy.thenArm[index] = arm;
        add(stem + (arm ? "then" : "else"), policy, {});
      }
    }
    const auto windows = windowUses(function, *analysis);
    for (std::size_t index = 0; index < windows.size(); ++index) {
      const auto &use = windows[index];
      const auto &loop = analysis->loops[use.loop];
      if (loop.startParameter.empty())
        continue;
      // The first tile starts at `stride * start` and covers `rows` rows.
      const auto first = std::to_string(use.stride) + " * " + loop.startParameter;
      const auto last = first + " + " + std::to_string(use.rows);
      const auto &preheader = function.blocks[loop.preheader].label.name;
      auto policy = baseline;
      policy.trips[use.loop] = 1;
      const auto stem = "window" + std::to_string(index) + "-";
      add(stem + "below", policy, {{preheader, {first + " < 0", last + " > 0"}}});
      const auto active = std::to_string(use.active);
      add(stem + "above", policy, {{preheader, {first + " < " + active, last + " > " + active}}});
    }
    return targets;
  }

} // namespace refractir::reify
