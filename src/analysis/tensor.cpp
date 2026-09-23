#include "analysis/tensor.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>
#include "analysis/type_utils.hpp"

namespace refractir {

  std::uint32_t tensorElementCount(const Shape &shape) {
    std::uint64_t count = 1;
    for (const auto dim: shape) {
      if (dim == 0)
        throw std::invalid_argument("tensor-shape: dimensions must be positive");
      count *= dim;
      if (count > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("tensor-shape: element count overflows uint32");
    }
    return static_cast<std::uint32_t>(count);
  }

  std::uint32_t normalizeTensorAxis(std::int64_t axis, std::uint32_t rank) {
    const auto signedRank = static_cast<std::int64_t>(rank);
    if (rank == 0 || axis < -signedRank || axis >= signedRank)
      throw std::out_of_range("tensor-axis: axis is outside the operand rank");
    return static_cast<std::uint32_t>(axis < 0 ? axis + signedRank : axis);
  }

  Shape tensorBroadcastShape(const Shape &lhs, const Shape &rhs) {
    const auto rank = std::max(lhs.size(), rhs.size());
    Shape out(rank, 1);
    for (std::size_t i = 0; i < rank; ++i) {
      const auto li = i < rank - lhs.size() ? 1 : lhs[i - (rank - lhs.size())];
      const auto ri = i < rank - rhs.size() ? 1 : rhs[i - (rank - rhs.size())];
      if (li == 0 || ri == 0)
        throw std::invalid_argument("tensor-shape: dimensions must be positive");
      if (li != ri && li != 1 && ri != 1)
        throw std::invalid_argument("tensor-broadcast: aligned dimensions are incompatible");
      out[i] = std::max(li, ri);
    }
    (void) tensorElementCount(out);
    return out;
  }

  Shape tensorReduceShape(const Shape &input, std::int64_t axis, bool keep) {
    const auto normalized = normalizeTensorAxis(axis, static_cast<std::uint32_t>(input.size()));
    Shape out = input;
    if (keep)
      out[normalized] = 1;
    else
      out.erase(out.begin() + normalized);
    return out;
  }

  namespace {

    bool containsTensorType(
        const TypePtr &type, const TypeUtils::StructTable &structs,
        std::vector<std::string> *errors = nullptr
    ) {
      if (!type)
        return false;
      if (const auto tensor = std::get_if<TensorType>(&type->v)) {
        if (errors) {
          if (tensor->shape.empty())
            errors->push_back("tensor-profile: rank zero is not supported");
          if (tensor->shape.size() > 3)
            errors->push_back("tensor-profile: tensor rank exceeds 3");
          try {
            const auto count = tensorElementCount(tensor->shape);
            if (count > TensorProfile{}.maxElements)
              errors->push_back("tensor-profile: tensor exceeds 256 elements");
          } catch (const std::exception &e) {
            errors->push_back(e.what());
          }
          for (const auto dim: tensor->shape) {
            if ((dim & (dim - 1)) != 0)
              errors->push_back("tensor-profile: dimensions must be powers of two");
          }
          if (!tensor->elem || (!std::holds_alternative<IntType>(tensor->elem->v) &&
                                !std::holds_alternative<FloatType>(tensor->elem->v)))
            errors->push_back("tensor-type: tensor element type must be i32 or f32");
          else if (const auto bits = TypeUtils::getIntBitWidth(tensor->elem); bits && *bits != 32)
            errors->push_back("tensor-type: tensor element type must be i32 or f32");
          else if (const auto bits = TypeUtils::getFloatBitWidth(tensor->elem); bits && *bits != 32)
            errors->push_back("tensor-type: tensor element type must be i32 or f32");
        }
        return true;
      }
      if (const auto array = std::get_if<ArrayType>(&type->v))
        return containsTensorType(array->elem, structs, errors);
      if (const auto ptr = std::get_if<PtrType>(&type->v))
        return containsTensorType(ptr->pointee, structs, errors);
      if (const auto structure = std::get_if<StructType>(&type->v)) {
        const auto it = structs.find(structure->name.name);
        if (it == structs.end())
          return false;
        bool result = false;
        for (const auto &field: it->second->fields)
          result = containsTensorType(field.type, structs, errors) || result;
        return result;
      }
      return false;
    }

    template<typename F>
    void eachProgramType(const Program &prog, F &&visit) {
      for (const auto &structure: prog.structs)
        for (const auto &field: structure.fields)
          visit(field.type, field.span);
      for (const auto &function: prog.funs) {
        for (const auto &param: function.params)
          visit(param.type, param.span);
        visit(function.retType, function.span);
        for (const auto &symbol: function.syms)
          visit(symbol.type, symbol.span);
        for (const auto &local: function.lets)
          visit(local.type, local.span);
      }
      for (const auto &decl: prog.extDecls) {
        for (const auto &param: decl.params)
          visit(param.type, param.span);
        visit(decl.retType, decl.span);
      }
      for (const auto &intrinsic: prog.intrinsics) {
        for (const auto &param: intrinsic.params)
          visit(param.type, param.span);
        visit(intrinsic.retType, intrinsic.span);
      }
    }

    bool containsTensorAtom(const Atom &atom) { return std::holds_alternative<TensorAtom>(atom.v); }

    bool programHasTensorAtom(const Program &prog) {
      auto hasExpr = [](const Expr &expr) {
        if (containsTensorAtom(expr.first))
          return true;
        return std::any_of(expr.rest.begin(), expr.rest.end(), [](const Expr::Tail &tail) {
          return containsTensorAtom(tail.atom);
        });
      };
      for (const auto &function: prog.funs) {
        for (const auto &local: function.lets)
          if (local.init && local.init->kind == InitVal::Kind::Atom &&
              containsTensorAtom(*std::get<AtomPtr>(local.init->value)))
            return true;
        for (const auto &block: function.blocks) {
          for (const auto &instruction: block.instrs) {
            bool found = std::visit(
                [&](const auto &ins) {
                  using T = std::decay_t<decltype(ins)>;
                  if constexpr (std::is_same_v<T, AssignInstr>)
                    return hasExpr(ins.rhs);
                  else if constexpr (std::is_same_v<T, StoreInstr>)
                    return hasExpr(ins.ptr) || hasExpr(ins.val);
                  else if constexpr (std::is_same_v<T, AssumeInstr> ||
                                     std::is_same_v<T, RequireInstr>)
                    return hasExpr(ins.cond.lhs) || hasExpr(ins.cond.rhs);
                  return false;
                },
                instruction
            );
            if (found)
              return true;
          }
          if (const auto ret = std::get_if<RetTerm>(&block.term);
              ret && ret->value && hasExpr(*ret->value))
            return true;
        }
      }
      return false;
    }

    bool expressionHasCallAtom(const Expr &expr) {
      auto isCall = [](const Atom &atom) { return std::holds_alternative<CallAtom>(atom.v); };
      if (isCall(expr.first))
        return true;
      return std::any_of(expr.rest.begin(), expr.rest.end(), [&](const Expr::Tail &tail) {
        return isCall(tail.atom);
      });
    }

    bool tensorFunctionHasCall(const FunDecl &function) {
      for (const auto &local: function.lets)
        if (local.init && local.init->kind == InitVal::Kind::Atom &&
            std::holds_alternative<CallAtom>(std::get<AtomPtr>(local.init->value)->v))
          return true;
      for (const auto &block: function.blocks) {
        for (const auto &instruction: block.instrs) {
          const bool hasCall = std::visit(
              [&](const auto &ins) {
                using T = std::decay_t<decltype(ins)>;
                if constexpr (std::is_same_v<T, AssignInstr>) {
                  return expressionHasCallAtom(ins.rhs);
                } else if constexpr (std::is_same_v<T, StoreInstr>) {
                  return expressionHasCallAtom(ins.ptr) || expressionHasCallAtom(ins.val);
                } else if constexpr (std::is_same_v<T, AssumeInstr> ||
                                     std::is_same_v<T, RequireInstr>) {
                  return expressionHasCallAtom(ins.cond.lhs) || expressionHasCallAtom(ins.cond.rhs);
                }
                return false;
              },
              instruction
          );
          if (hasCall)
            return true;
        }
        if (const auto branch = std::get_if<BrTerm>(&block.term);
            branch && branch->cond &&
            (expressionHasCallAtom(branch->cond->lhs) || expressionHasCallAtom(branch->cond->rhs)))
          return true;
        if (const auto ret = std::get_if<RetTerm>(&block.term);
            ret && ret->value && expressionHasCallAtom(*ret->value))
          return true;
      }
      return false;
    }

    bool functionHasTensorType(const FunDecl &function, const TypeUtils::StructTable &structs) {
      if (containsTensorType(function.retType, structs))
        return true;
      for (const auto &param: function.params)
        if (containsTensorType(param.type, structs))
          return true;
      for (const auto &symbol: function.syms)
        if (containsTensorType(symbol.type, structs))
          return true;
      for (const auto &local: function.lets)
        if (containsTensorType(local.type, structs))
          return true;
      return false;
    }

  } // namespace

  bool checkTensorCapabilities(const Program &prog, TensorConsumer consumer, DiagBag &diags) {
    const bool legacy = consumer == TensorConsumer::C || consumer == TensorConsumer::Wasm ||
                        consumer == TensorConsumer::Python;
    const auto structs = TypeUtils::buildStructTable(prog);
    bool hasTensor = programHasTensorAtom(prog);
    std::vector<std::pair<std::string, SourceSpan>> profileErrors;
    eachProgramType(prog, [&](const TypePtr &type, SourceSpan span) {
      std::vector<std::string> errors;
      hasTensor = containsTensorType(type, structs, &errors) || hasTensor;
      for (auto &error: errors)
        profileErrors.emplace_back(std::move(error), span);
    });

    if (legacy && hasTensor) {
      const char *target = consumer == TensorConsumer::C      ? "C"
                           : consumer == TensorConsumer::Wasm ? "WebAssembly"
                                                              : "Python";
      diags.error(
          std::string("tensor-target: ") + target + " emission does not support tensors", prog.span
      );
      return false;
    }

    if (consumer != TensorConsumer::Reference && consumer != TensorConsumer::ExactSolver &&
        consumer != TensorConsumer::Triton)
      return !diags.hasErrors();
    for (auto &[message, span]: profileErrors)
      diags.error(message, span);
    for (const auto &function: prog.funs) {
      if (!functionHasTensorType(function, structs))
        continue;
      if (function.blocks.size() != 1 ||
          !std::holds_alternative<RetTerm>(function.blocks.front().term))
        diags.error(
            "tensor-profile: tensor functions must have one basic block ending in ret",
            function.span
        );
      if (tensorFunctionHasCall(function))
        diags.error("tensor-profile: tensor functions cannot contain calls", function.span);
    }
    return !diags.hasErrors();
  }

} // namespace refractir
