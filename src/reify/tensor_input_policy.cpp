#include "reify/tensor_input_policy.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <random>
#include <sstream>
#include <stdexcept>

#include "analysis/type_utils.hpp"
#include "ast/clone.hpp"
#include "ast/sir_printer.hpp"
#include "frontend/diagnostics.hpp"
#include "frontend/pipeline.hpp"

namespace refractir::reify {
  namespace {

    /// Query locals carry this prefix so a computation digest can drop them
    /// and compare the constrained clone with the frozen program.
    constexpr std::string_view queryPrefix = "__query_";

    std::string digestHex(std::uint64_t value) {
      std::ostringstream out;
      out << std::hex << value;
      return out.str();
    }

    std::uint64_t fnv1a64(std::string_view text) {
      std::uint64_t hash = 0xcbf29ce484222325ULL;
      for (const unsigned char byte: text) {
        hash ^= byte;
        hash *= 0x100000001b3ULL;
      }
      return hash;
    }

    void appendLittleWord(std::vector<std::uint8_t> &bytes, std::uint32_t word) {
      for (unsigned shift = 0; shift < 32; shift += 8)
        bytes.push_back(static_cast<std::uint8_t>(word >> shift));
    }

    std::uint32_t readLittleWord(const std::vector<std::uint8_t> &bytes, std::size_t offset) {
      std::uint32_t word = 0;
      for (unsigned index = 0; index < 4; ++index)
        word |= static_cast<std::uint32_t>(bytes[offset + index]) << (index * 8);
      return word;
    }

    const FunDecl *entryFunction(const Program &program) {
      for (const auto &function: program.funs)
        if (function.name.name == "@kernel")
          return &function;
      return nullptr;
    }

    FunDecl *entryFunction(Program &program) {
      for (auto &function: program.funs)
        if (function.name.name == "@kernel")
          return &function;
      return nullptr;
    }

    /// Tensor and scalar parameters are the only shapes an input policy can
    /// carry; a pointer or aggregate parameter has no tensor-v1 buffer form.
    bool isSamplable(const TypePtr &type) {
      if (TypeUtils::asTensor(type))
        return true;
      return TypeUtils::getIntBitWidth(type).has_value() ||
             TypeUtils::getFloatBitWidth(type).has_value();
    }

    TensorElement elementOf(const TypePtr &type) {
      const auto *tensor = TypeUtils::asTensor(type);
      const TypePtr element = tensor ? tensor->elem : type;
      if (TypeUtils::getFloatBitWidth(element))
        return TensorElement::F32;
      return TensorElement::I32;
    }

    bool isIntegerElement(const TypePtr &type) {
      const auto *tensor = TypeUtils::asTensor(type);
      return TypeUtils::getIntBitWidth(tensor ? tensor->elem : type).has_value();
    }

    Shape shapeOf(const TypePtr &type) {
      const auto *tensor = TypeUtils::asTensor(type);
      return tensor ? tensor->shape : Shape{};
    }

    std::uint32_t elementCountOf(const TypePtr &type) {
      const auto shape = shapeOf(type);
      return shape.empty() ? 1u : tensorElementCount(shape);
    }

    std::string bufferName(const std::string &sirName) {
      std::string name = sirName;
      if (!name.empty() && name.front() == '%')
        name.erase(name.begin());
      return name;
    }

    Expr localExpr(const std::string &name) {
      LValue lvalue;
      lvalue.base = LocalId{name, {}};
      return Expr{Atom{RValueAtom{std::move(lvalue), {}}, {}}, {}, {}};
    }

    Expr integerExpr(std::int64_t value) {
      return Expr{Atom{CoefAtom{IntLit{value, {}}, {}}, {}}, {}, {}};
    }

    /// The local a terminator returns, or an empty name when the return value
    /// is not a bare local reference.
    std::string returnedLocal(const Block &block) {
      const auto *ret = std::get_if<RetTerm>(&block.term);
      if (!ret || !ret->value)
        return {};
      const auto *rvalue = std::get_if<RValueAtom>(&ret->value->first.v);
      if (!rvalue || !ret->value->rest.empty() || !rvalue->rval.accesses.empty())
        return {};
      return rvalue->rval.base.name;
    }

    /// Build the property-constrained clone: the same computation plus the
    /// `require` that states the target. The probe local carries the query
    /// prefix, so the digest ignores it and the clone matches the frozen
    /// program exactly when nothing but the query changed.
    std::optional<Program> buildQuery(const Program &frozen, const TensorInputProperty &property) {
      if (property.kind == TensorInputProperty::Kind::None)
        return std::nullopt;
      Program query = cloneProgram(frozen);
      FunDecl *entry = entryFunction(query);
      if (!entry)
        return std::nullopt;

      Block *returnBlock = nullptr;
      std::string returned;
      for (auto &block: entry->blocks) {
        if (!std::holds_alternative<RetTerm>(block.term))
          continue;
        returnBlock = &block;
        returned = returnedLocal(block);
        break;
      }
      if (!returnBlock || returned.empty())
        return std::nullopt;

      const auto *tensor = TypeUtils::asTensor(entry->retType);
      std::string probe;
      if (property.kind == TensorInputProperty::Kind::OutputEquals) {
        // A scalar integer return can be constrained directly. A tensor
        // return states the property per element, which the completed
        // extraction route does not admit yet.
        if (tensor || !TypeUtils::getIntBitWidth(entry->retType).has_value())
          return std::nullopt;
      } else {
        // A rank-one integer tensor return reduces to the scalar the
        // property names, using only operators the profile already admits.
        if (!tensor || tensor->shape.size() != 1 ||
            !TypeUtils::getIntBitWidth(tensor->elem).has_value())
          return std::nullopt;
        probe = std::string(queryPrefix) + "sum";
        LetDecl local;
        local.isMutable = true;
        local.name = LocalId{probe, {}};
        local.type = tensor->elem;
        local.init = InitVal{InitVal::Kind::Undef, {}, {}};
        entry->lets.push_back(std::move(local));

        TensorAtom reduce;
        reduce.op = TensorOp::Sum;
        reduce.operands.push_back(Coef{LocalId{returned, {}}});
        reduce.axis = 0;
        reduce.keepDims = false;

        LValue reduced;
        reduced.base = LocalId{probe, {}};
        AssignInstr assignment;
        assignment.lhs = std::move(reduced);
        assignment.rhs = Expr{Atom{std::move(reduce), {}}, {}, {}};
        returnBlock->instrs.push_back(Instr{std::move(assignment)});
      }

      RequireInstr requirement;
      requirement.cond =
          Cond{localExpr(probe.empty() ? returned : probe), RelOp::EQ, integerExpr(property.target), {}};
      returnBlock->instrs.push_back(Instr{std::move(requirement)});
      return query;
    }

    std::vector<std::string>
    resolvePath(const FunDecl &entry, const std::vector<std::string> &path) {
      if (!path.empty())
        return path;
      if (entry.blocks.size() == 1)
        return {entry.blocks.front().label.name};
      return {};
    }

    void sampleValues(
        const FunDecl &entry, TensorInputPolicy policy, std::uint32_t seed,
        std::vector<TensorCaseBuffer> &buffers
    ) {
      std::mt19937 random(
          seed ^ (policy == TensorInputPolicy::Boundary ? 0x626f756eULL : 0x6f726469ULL)
      );
      // Boundary sampling stays inside the integer definedness contract: an
      // extreme magnitude would overflow a sum in the reference, which is UB
      // rather than a boundary a case can carry.
      static constexpr std::array<std::int32_t, 4> boundary{0, 1, -1, 2};
      for (const auto &param: entry.params) {
        if (!isSamplable(param.type))
          throw std::runtime_error(
              "tensor input policy: parameter " + param.name.name +
              " is neither a tensor nor a scalar"
          );
        const TensorElement element = elementOf(param.type);
        const Shape shape = shapeOf(param.type);
        TensorCaseBuffer buffer{bufferName(param.name.name), element, shape, {}};
        const auto count = elementCountOf(param.type);
        for (std::uint32_t index = 0; index < count; ++index) {
          const std::int32_t value =
              policy == TensorInputPolicy::Boundary
                  ? boundary[(index + seed) % boundary.size()]
                  : 1 + static_cast<std::int32_t>(random() % 2);
          if (element == TensorElement::I32)
            appendLittleWord(buffer.bytes, std::bit_cast<std::uint32_t>(value));
          else
            appendLittleWord(buffer.bytes, std::bit_cast<std::uint32_t>(static_cast<float>(value)));
        }
        buffers.push_back(std::move(buffer));
      }
    }

    std::optional<std::vector<TensorCaseBuffer>> witnessBuffers(
        const FunDecl &entry, const SymbolicExecutor::Result &result, std::string &reason
    ) {
      std::vector<TensorCaseBuffer> buffers;
      for (const auto &param: entry.params) {
        const TensorElement element = elementOf(param.type);
        const Shape shape = shapeOf(param.type);
        const auto count = elementCountOf(param.type);

        std::vector<SymbolicExecutor::Result::ModelVal> leaves;
        if (TypeUtils::asTensor(param.type)) {
          const auto found = result.tensorParamModel.find(param.name.name);
          if (found == result.tensorParamModel.end()) {
            reason = "solver model omitted tensor parameter " + param.name.name;
            return std::nullopt;
          }
          if (!found->second.valid() ||
              found->second.leaves.size() != static_cast<std::size_t>(count)) {
            reason = "tensor parameter witness is malformed for " + param.name.name;
            return std::nullopt;
          }
          if (shape != found->second.shape) {
            reason =
                "tensor parameter witness shape disagrees with the declaration of " +
                param.name.name;
            return std::nullopt;
          }
          leaves = found->second.leaves;
        } else {
          const auto found = result.paramModel.find(param.name.name);
          if (found == result.paramModel.end()) {
            reason = "solver model omitted scalar parameter " + param.name.name;
            return std::nullopt;
          }
          leaves.push_back(found->second);
        }

        TensorCaseBuffer buffer{bufferName(param.name.name), element, shape, {}};
        for (const auto &leaf: leaves) {
          if (element == TensorElement::I32) {
            if (!std::holds_alternative<std::int64_t>(leaf)) {
              reason = "integer parameter leaf is not an integer in the model";
              return std::nullopt;
            }
            appendLittleWord(
                buffer.bytes,
                std::bit_cast<std::uint32_t>(static_cast<std::int32_t>(std::get<std::int64_t>(leaf)))
            );
          } else {
            if (!std::holds_alternative<double>(leaf)) {
              reason = "floating parameter leaf is not a float in the model";
              return std::nullopt;
            }
            const auto value = static_cast<float>(std::get<double>(leaf));
            if (!std::isfinite(value)) {
              reason = "floating parameter leaf is not finite in the model";
              return std::nullopt;
            }
            appendLittleWord(buffer.bytes, std::bit_cast<std::uint32_t>(value));
          }
        }
        buffers.push_back(std::move(buffer));
      }
      return buffers;
    }

  } // namespace

  TensorInputPolicy parseTensorInputPolicy(std::string_view name) {
    if (name == "ordinary")
      return TensorInputPolicy::Ordinary;
    if (name == "boundary")
      return TensorInputPolicy::Boundary;
    if (name == "exact")
      return TensorInputPolicy::Exact;
    throw std::invalid_argument(
        "tensor input policy must be ordinary, boundary, or exact: " + std::string(name)
    );
  }

  std::string_view tensorInputPolicyName(TensorInputPolicy policy) {
    switch (policy) {
      case TensorInputPolicy::Ordinary:
        return "ordinary";
      case TensorInputPolicy::Boundary:
        return "boundary";
      case TensorInputPolicy::Exact:
        return "exact";
    }
    return "ordinary";
  }

  std::string computationDigest(const Program &program) {
    Program filtered = cloneProgram(program);
    const auto isQuery = [](const std::string &name) {
      return name.rfind(queryPrefix, 0) == 0;
    };
    for (auto &function: filtered.funs) {
      for (auto &block: function.blocks) {
        block.instrs.erase(
            std::remove_if(
                block.instrs.begin(), block.instrs.end(),
                [&](const Instr &instruction) {
                  if (std::holds_alternative<AssumeInstr>(instruction) ||
                      std::holds_alternative<RequireInstr>(instruction))
                    return true;
                  // The probe assignment computes the query's own reduction;
                  // it is scaffolding, not part of the frozen computation.
                  if (const auto *assignment = std::get_if<AssignInstr>(&instruction))
                    return isQuery(assignment->lhs.base.name);
                  return false;
                }
            ),
            block.instrs.end()
        );
      }
      function.lets.erase(
          std::remove_if(
              function.lets.begin(), function.lets.end(),
              [&](const LetDecl &let) { return isQuery(let.name.name); }
          ),
          function.lets.end()
      );
    }
    std::ostringstream text;
    SIRPrinter printer(text, {}, {}, {});
    printer.print(filtered);
    return digestHex(fnv1a64(text.str()));
  }

  bool tensorPropertyHolds(
      const std::vector<std::uint8_t> &output, TensorElement element, const Shape &shape,
      const TensorInputProperty &property
  ) {
    if (property.kind == TensorInputProperty::Kind::None)
      return true;
    const auto count = shape.empty() ? 1u : tensorElementCount(shape);
    if (output.size() < static_cast<std::size_t>(count) * 4)
      return false;
    std::int64_t total = 0;
    for (std::uint32_t index = 0; index < count; ++index) {
      const auto word = readLittleWord(output, index * 4);
      const std::int64_t value = element == TensorElement::I32
                                     ? static_cast<std::int64_t>(std::bit_cast<std::int32_t>(word))
                                     : static_cast<std::int64_t>(std::bit_cast<float>(word));
      if (property.kind == TensorInputProperty::Kind::OutputEquals && index == 0)
        return value == property.target;
      total += value;
    }
    return total == property.target;
  }

  TensorInputProposal proposeTensorInputs(
      const Program &program, const SymbolicExecutor::SolverFactory &solverFactory,
      const TensorInputPolicyOptions &options
  ) {
    TensorInputProposal proposal;
    proposal.policy = options.policy;
    proposal.property = options.property;
    proposal.computationDigest = computationDigest(program);

    const FunDecl *entry = entryFunction(program);
    if (!entry) {
      proposal.outcome = "invalid";
      proposal.reason = "program has no @kernel entry";
      return proposal;
    }
    if (options.policy != TensorInputPolicy::Exact) {
      sampleValues(*entry, options.policy, options.seed, proposal.inputs);
      proposal.queryDigest = proposal.computationDigest;
      proposal.outcome = "accepted";
      return proposal;
    }

    if (options.property.kind == TensorInputProperty::Kind::None) {
      proposal.outcome = "unsupported";
      proposal.reason = "the exact policy needs a stated output property before it can select";
      return proposal;
    }
    const auto path = resolvePath(*entry, options.path);
    if (path.empty()) {
      proposal.outcome = "invalid";
      proposal.reason = "a branching program needs an explicit solver path";
      return proposal;
    }

    std::uint32_t symbolicElements = 0;
    for (const auto &param: entry->params) {
      if (!isIntegerElement(param.type)) {
        proposal.outcome = "unsupported";
        proposal.reason = "the exact i32 route has no domain for parameter " + param.name.name;
        return proposal;
      }
      symbolicElements += elementCountOf(param.type);
    }
    proposal.symbolicElements = symbolicElements;
    if (symbolicElements > options.maxSymbolicElements) {
      proposal.outcome = "budget_exceeded";
      proposal.reason = "the exact i32 route holds at most " +
                        std::to_string(options.maxSymbolicElements) +
                        " symbolic input elements, and this entry has " +
                        std::to_string(symbolicElements);
      return proposal;
    }

    auto query = buildQuery(program, options.property);
    if (!query) {
      proposal.outcome = "unsupported";
      proposal.reason =
          "the requested property cannot be expressed on this program's return value";
      return proposal;
    }
    proposal.queryDigest = computationDigest(*query);
    if (proposal.queryDigest != proposal.computationDigest) {
      proposal.outcome = "invalid";
      proposal.reason = "the property query changed the computation it was meant to constrain";
      return proposal;
    }

    DiagBag diags;
    CheckOptions checkOptions;
    checkOptions.functionAnalyses = false;
    checkOptions.tensorProfile = options.profileVersion;
    if (!checkProgram(*query, diags, checkOptions) ||
        !checkTensorCapabilities(
            *query, TensorConsumer::ExactSolver, tensorProfile(options.profileVersion), diags
        )) {
      proposal.outcome = "invalid";
      proposal.reason = "the property query is outside the selected tensor profile";
      return proposal;
    }

    SymbolicExecutor::Config config;
    config.timeout_ms = options.timeoutMs;
    config.seed = options.seed;
    config.tensorParamDomain = options.domain;
    SymbolicExecutor executor(*query, config, solverFactory);
    const auto result = executor.solve("@kernel", path);
    if (!result.sat) {
      if (result.unsat) {
        proposal.outcome = "unsat";
        proposal.reason = "no input satisfies the stated property on this path";
      } else {
        proposal.outcome = "unknown";
        proposal.reason = result.message.empty() ? "the solver did not decide" : result.message;
      }
      return proposal;
    }

    std::string reason;
    auto buffers = witnessBuffers(*entry, result, reason);
    if (!buffers) {
      proposal.outcome = "invalid";
      proposal.reason = std::move(reason);
      return proposal;
    }
    proposal.inputs = std::move(*buffers);
    proposal.outcome = "accepted";
    return proposal;
  }

} // namespace refractir::reify
