#include "analysis/tensor.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include "analysis/cfg.hpp"
#include "analysis/dominators.hpp"
#include "analysis/loop_info.hpp"
#include "analysis/reducibility.hpp"
#include "analysis/type_utils.hpp"

namespace refractir {

  TensorProfile tensorProfile(TensorProfileId id) {
    if (id == TensorProfileId::V1)
      return TensorProfile{TensorProfileId::V1, 3, 256, 0};
    if (id == TensorProfileId::V2)
      return TensorProfile{TensorProfileId::V2, 3, 512, 8};
    if (id == TensorProfileId::V3)
      return TensorProfile{TensorProfileId::V3, 2, 512, 8};
    return TensorProfile{TensorProfileId::V4, 1, 512, 0};
  }

  TensorProfileId parseTensorProfileVersion(std::string_view version) {
    if (version == "v1")
      return TensorProfileId::V1;
    if (version == "v2")
      return TensorProfileId::V2;
    if (version == "v3")
      return TensorProfileId::V3;
    if (version == "v4")
      return TensorProfileId::V4;
    throw std::invalid_argument("tensor-profile: unknown tensor profile version '" +
                                std::string(version) + "'");
  }

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

  Shape tensorExecutionTileShape(const Shape &shape, const TensorProfile &profile) {
    if (shape.empty())
      throw std::invalid_argument("tensor-profile: rank-zero tensors are not supported");
    if (shape.size() > profile.maxRank)
      throw std::invalid_argument(
          "tensor-profile: tensor rank exceeds " + std::to_string(profile.maxRank)
      );

    Shape tileShape;
    tileShape.reserve(shape.size());
    std::uint64_t tileElements = 1;
    for (const auto dimension: shape) {
      if (dimension == 0)
        throw std::invalid_argument("tensor-shape: dimensions must be positive");
      if (profile.id != TensorProfileId::V3) {
        if ((dimension & (dimension - 1)) != 0)
          throw std::invalid_argument("tensor-profile: dimensions must be powers of two");
        tileShape.push_back(dimension);
        continue;
      }

      std::uint64_t extent = 1;
      while (extent < dimension)
        extent <<= 1;
      tileElements *= extent;
      if (tileElements > profile.maxElements)
        throw std::invalid_argument(
            "tensor-profile: padded tile volume exceeds " +
            std::to_string(profile.maxElements)
        );
      tileShape.push_back(static_cast<std::uint32_t>(extent));
    }
    if (profile.id != TensorProfileId::V3) {
      const auto elements = tensorElementCount(shape);
      if (elements > profile.maxElements)
        throw std::invalid_argument(
            "tensor-profile: tensor exceeds " + std::to_string(profile.maxElements) +
            " elements"
        );
    }
    return tileShape;
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
        std::unordered_set<std::string> &visitedStructs, const TensorProfile &profile,
        std::vector<std::string> *errors
    ) {
      if (!type)
        return false;
      if (const auto tensor = std::get_if<TensorType>(&type->v)) {
        if (errors) {
          try {
            (void) tensorExecutionTileShape(tensor->shape, profile);
          } catch (const std::exception &e) {
            errors->push_back(e.what());
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
        return containsTensorType(array->elem, structs, visitedStructs, profile, errors);
      if (const auto ptr = std::get_if<PtrType>(&type->v))
        return containsTensorType(ptr->pointee, structs, visitedStructs, profile, errors);
      if (const auto structure = std::get_if<StructType>(&type->v)) {
        const auto it = structs.find(structure->name.name);
        if (it == structs.end() || !visitedStructs.insert(structure->name.name).second)
          return false;
        bool result = false;
        for (const auto &field: it->second->fields)
          result = containsTensorType(field.type, structs, visitedStructs, profile, errors) || result;
        return result;
      }
      return false;
    }

    bool containsTensorType(
        const TypePtr &type, const TypeUtils::StructTable &structs,
        const TensorProfile &profile, std::vector<std::string> *errors = nullptr
    ) {
      std::unordered_set<std::string> visitedStructs;
      return containsTensorType(type, structs, visitedStructs, profile, errors);
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

    template<typename AtomType>
    bool expressionContainsAtom(const Expr &expr);

    template<typename AtomType>
    bool conditionContainsAtom(const Cond &cond);

    template<typename AtomType>
    bool atomContainsAtom(const Atom &atom) {
      if (std::holds_alternative<AtomType>(atom.v))
        return true;
      return std::visit(
          [](const auto &value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, SelectAtom>) {
              return (value.cond && conditionContainsAtom<AtomType>(*value.cond)) ||
                     (value.maskExpr && expressionContainsAtom<AtomType>(*value.maskExpr));
            } else if constexpr (std::is_same_v<T, CallAtom>) {
              return std::any_of(value.args.begin(), value.args.end(), [](const auto &arg) {
                return arg && expressionContainsAtom<AtomType>(*arg);
              });
            }
            return false;
          },
          atom.v
      );
    }

    template<typename AtomType>
    bool expressionContainsAtom(const Expr &expr) {
      if (atomContainsAtom<AtomType>(expr.first))
        return true;
      return std::any_of(expr.rest.begin(), expr.rest.end(), [](const Expr::Tail &tail) {
        return atomContainsAtom<AtomType>(tail.atom);
      });
    }

    template<typename AtomType>
    bool conditionContainsAtom(const Cond &cond) {
      return expressionContainsAtom<AtomType>(cond.lhs) ||
             expressionContainsAtom<AtomType>(cond.rhs);
    }

    bool functionHasTensorAtom(const FunDecl &function) {
      auto hasExpr = [](const Expr &expr) { return expressionContainsAtom<TensorAtom>(expr); };
      for (const auto &local: function.lets)
        if (local.init && local.init->kind == InitVal::Kind::Atom &&
            atomContainsAtom<TensorAtom>(*std::get<AtomPtr>(local.init->value)))
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
        if (const auto branch = std::get_if<BrTerm>(&block.term);
            branch && branch->cond && (hasExpr(branch->cond->lhs) || hasExpr(branch->cond->rhs)))
          return true;
        if (const auto ret = std::get_if<RetTerm>(&block.term);
            ret && ret->value && hasExpr(*ret->value))
          return true;
      }
      return false;
    }

    bool programHasTensorAtom(const Program &prog) {
      return std::any_of(prog.funs.begin(), prog.funs.end(), functionHasTensorAtom);
    }

    bool expressionHasCallAtom(const Expr &expr) { return expressionContainsAtom<CallAtom>(expr); }

    bool expressionHasTensorOperation(const Expr &expr, TensorOp operation);

    bool atomHasTensorOperation(const Atom &atom, TensorOp operation) {
      if (const auto *tensor = std::get_if<TensorAtom>(&atom.v))
        return tensor->op == operation;
      return std::visit(
          [operation](const auto &value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, SelectAtom>) {
              return (value.cond &&
                      (expressionHasTensorOperation(value.cond->lhs, operation) ||
                       expressionHasTensorOperation(value.cond->rhs, operation))) ||
                     (value.maskExpr && expressionHasTensorOperation(*value.maskExpr, operation));
            } else if constexpr (std::is_same_v<T, CallAtom>) {
              return std::any_of(value.args.begin(), value.args.end(), [operation](const auto &arg) {
                return arg && expressionHasTensorOperation(*arg, operation);
              });
            }
            return false;
          },
          atom.v
      );
    }

    bool expressionHasTensorOperation(const Expr &expr, TensorOp operation) {
      if (atomHasTensorOperation(expr.first, operation))
        return true;
      return std::any_of(expr.rest.begin(), expr.rest.end(), [operation](const Expr::Tail &tail) {
        return atomHasTensorOperation(tail.atom, operation);
      });
    }

    bool functionHasTensorOperation(const FunDecl &function, TensorOp operation) {
      auto has = [operation](const Expr &expr) {
        return expressionHasTensorOperation(expr, operation);
      };
      for (const auto &local: function.lets)
        if (local.init && local.init->kind == InitVal::Kind::Atom &&
            atomHasTensorOperation(*std::get<AtomPtr>(local.init->value), operation))
          return true;
      for (const auto &block: function.blocks) {
        for (const auto &instruction: block.instrs) {
          const bool found = std::visit(
              [&](const auto &ins) {
                using T = std::decay_t<decltype(ins)>;
                if constexpr (std::is_same_v<T, AssignInstr>)
                  return has(ins.rhs);
                else if constexpr (std::is_same_v<T, StoreInstr>)
                  return has(ins.ptr) || has(ins.val);
                else if constexpr (std::is_same_v<T, AssumeInstr> ||
                                   std::is_same_v<T, RequireInstr>)
                  return has(ins.cond.lhs) || has(ins.cond.rhs);
                return false;
              },
              instruction
          );
          if (found)
            return true;
        }
        if (const auto branch = std::get_if<BrTerm>(&block.term);
            branch && branch->cond && (has(branch->cond->lhs) || has(branch->cond->rhs)))
          return true;
        if (const auto ret = std::get_if<RetTerm>(&block.term);
            ret && ret->value && has(*ret->value))
          return true;
      }
      return false;
    }

    bool v3SupportsTensorOperation(TensorOp op) {
      switch (op) {
        case TensorOp::Add:
        case TensorOp::Sub:
        case TensorOp::Mul:
        case TensorOp::Min:
        case TensorOp::Max:
        case TensorOp::Expand:
        case TensorOp::Broadcast:
        case TensorOp::Transpose:
        case TensorOp::Sum:
        case TensorOp::ReduceMin:
        case TensorOp::ReduceMax:
        case TensorOp::Window:
          return true;
        default:
          return false;
      }
    }

    bool atomHasUnsupportedV3TensorOperation(const Atom &atom);

    bool expressionHasUnsupportedV3TensorOperation(const Expr &expr) {
      if (atomHasUnsupportedV3TensorOperation(expr.first))
        return true;
      return std::any_of(expr.rest.begin(), expr.rest.end(), [](const Expr::Tail &tail) {
        return atomHasUnsupportedV3TensorOperation(tail.atom);
      });
    }

    bool conditionHasUnsupportedV3TensorOperation(const Cond &condition) {
      return expressionHasUnsupportedV3TensorOperation(condition.lhs) ||
             expressionHasUnsupportedV3TensorOperation(condition.rhs);
    }

    bool atomHasUnsupportedV3TensorOperation(const Atom &atom) {
      if (const auto *tensor = std::get_if<TensorAtom>(&atom.v))
        return !v3SupportsTensorOperation(tensor->op);
      return std::visit(
          [](const auto &value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, SelectAtom>) {
              return (value.cond && conditionHasUnsupportedV3TensorOperation(*value.cond)) ||
                     (value.maskExpr &&
                      expressionHasUnsupportedV3TensorOperation(*value.maskExpr));
            } else if constexpr (std::is_same_v<T, CallAtom>) {
              return std::any_of(value.args.begin(), value.args.end(), [](const auto &arg) {
                return arg && expressionHasUnsupportedV3TensorOperation(*arg);
              });
            }
            return false;
          },
          atom.v
      );
    }

    bool functionHasUnsupportedV3TensorOperation(const FunDecl &function) {
      for (const auto &local: function.lets)
        if (local.init && local.init->kind == InitVal::Kind::Atom &&
            atomHasUnsupportedV3TensorOperation(*std::get<AtomPtr>(local.init->value)))
          return true;
      for (const auto &block: function.blocks) {
        for (const auto &instruction: block.instrs) {
          const bool unsupported = std::visit(
              [](const auto &ins) {
                using T = std::decay_t<decltype(ins)>;
                if constexpr (std::is_same_v<T, AssignInstr>)
                  return expressionHasUnsupportedV3TensorOperation(ins.rhs);
                else if constexpr (std::is_same_v<T, StoreInstr>)
                  return expressionHasUnsupportedV3TensorOperation(ins.ptr) ||
                         expressionHasUnsupportedV3TensorOperation(ins.val);
                else if constexpr (std::is_same_v<T, AssumeInstr> ||
                                   std::is_same_v<T, RequireInstr>)
                  return conditionHasUnsupportedV3TensorOperation(ins.cond);
                return false;
              },
              instruction
          );
          if (unsupported)
            return true;
        }
        if (const auto branch = std::get_if<BrTerm>(&block.term);
            branch && branch->cond &&
            conditionHasUnsupportedV3TensorOperation(*branch->cond))
          return true;
        if (const auto ret = std::get_if<RetTerm>(&block.term);
            ret && ret->value && expressionHasUnsupportedV3TensorOperation(*ret->value))
          return true;
      }
      return false;
    }

    bool tensorFunctionHasCall(const FunDecl &function) {
      for (const auto &local: function.lets)
        if (local.init && local.init->kind == InitVal::Kind::Atom &&
            atomContainsAtom<CallAtom>(*std::get<AtomPtr>(local.init->value)))
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

    bool functionHasTensorType(
        const FunDecl &function, const TypeUtils::StructTable &structs,
        const TensorProfile &profile
    ) {
      if (containsTensorType(function.retType, structs, profile))
        return true;
      for (const auto &param: function.params)
        if (containsTensorType(param.type, structs, profile))
          return true;
      for (const auto &symbol: function.syms)
        if (containsTensorType(symbol.type, structs, profile))
          return true;
      for (const auto &local: function.lets)
        if (containsTensorType(local.type, structs, profile))
          return true;
      return false;
    }

    bool isF32Tensor(const TypePtr &type, const Shape &shape) {
      const auto tensor = TypeUtils::asTensor(type);
      return tensor && tensor->shape == shape && TypeUtils::getFloatBitWidth(tensor->elem) == 32;
    }

    bool isLocalCoef(const Coef &coef, std::string_view expected) {
      const auto id = std::get_if<LocalOrSymId>(&coef);
      if (!id)
        return false;
      const auto local = std::get_if<LocalId>(id);
      return local && local->name == expected;
    }

    std::optional<std::string> simpleLocal(const Expr &expr);

    bool isV4RemainderKernel(const FunDecl &function) {
      if (function.params.size() != 2 || !function.syms.empty() || function.lets.size() != 1 ||
          function.blocks.size() != 1 || function.blocks.front().instrs.size() != 1)
        return false;
      const auto &left = function.params[0];
      const auto &right = function.params[1];
      const auto leftTensor = TypeUtils::asTensor(left.type);
      const auto rightTensor = TypeUtils::asTensor(right.type);
      if (!leftTensor || !rightTensor || leftTensor->shape != rightTensor->shape ||
          !isF32Tensor(left.type, leftTensor->shape) ||
          !isF32Tensor(right.type, leftTensor->shape) ||
          !isF32Tensor(function.retType, leftTensor->shape))
        return false;

      const auto &output = function.lets.front();
      if (!output.isMutable || !output.init || output.init->kind != InitVal::Kind::Undef ||
          !isF32Tensor(output.type, leftTensor->shape))
        return false;
      const auto *assignment = std::get_if<AssignInstr>(&function.blocks.front().instrs.front());
      if (!assignment || assignment->lhs.base.name != output.name.name ||
          !assignment->lhs.accesses.empty() || !assignment->rhs.rest.empty())
        return false;
      const auto *atom = std::get_if<TensorAtom>(&assignment->rhs.first.v);
      if (!atom || atom->op != TensorOp::Rem || atom->operands.size() != 2 ||
          !isLocalCoef(atom->operands[0], left.name.name) ||
          !isLocalCoef(atom->operands[1], right.name.name))
        return false;

      const auto *ret = std::get_if<RetTerm>(&function.blocks.front().term);
      if (!ret || !ret->value || !simpleLocal(*ret->value) ||
          *simpleLocal(*ret->value) != output.name.name)
        return false;
      return true;
    }

    std::optional<std::string> simpleLocalAtom(const Atom &atom) {
      if (const auto *value = std::get_if<CoefAtom>(&atom.v)) {
        const auto *id = std::get_if<LocalOrSymId>(&value->coef);
        if (id)
          if (const auto *local = std::get_if<LocalId>(id))
            return local->name;
      }
      if (const auto *value = std::get_if<RValueAtom>(&atom.v))
        if (value->rval.accesses.empty())
          return value->rval.base.name;
      return std::nullopt;
    }

    std::optional<std::string> simpleLocal(const Expr &expr) {
      return expr.rest.empty() ? simpleLocalAtom(expr.first) : std::nullopt;
    }

    std::optional<std::int64_t> simpleIntegerAtom(const Atom &atom) {
      const auto *value = std::get_if<CoefAtom>(&atom.v);
      if (!value)
        return std::nullopt;
      if (const auto *integer = std::get_if<IntLit>(&value->coef))
        return integer->value;
      return std::nullopt;
    }

    std::optional<std::int64_t> simpleInteger(const Expr &expr) {
      if (!expr.rest.empty())
        return std::nullopt;
      return simpleIntegerAtom(expr.first);
    }

    // The range `assume %name >= lo;` and `assume %name <= hi;` state in the
    // entry block for an i32 entry parameter, if both are present.
    std::optional<std::pair<std::int64_t, std::int64_t>>
    assumedParameterRange(const FunDecl &function, const std::string &name) {
      const auto param = std::find_if(
          function.params.begin(), function.params.end(),
          [&](const ParamDecl &candidate) { return candidate.name.name == name; }
      );
      if (param == function.params.end() || TypeUtils::asTensor(param->type) ||
          TypeUtils::getIntBitWidth(param->type) != 32 || function.blocks.empty())
        return std::nullopt;
      std::optional<std::int64_t> low;
      std::optional<std::int64_t> high;
      for (const auto &instruction: function.blocks.front().instrs) {
        const auto *assume = std::get_if<AssumeInstr>(&instruction);
        if (!assume || simpleLocal(assume->cond.lhs) != name)
          continue;
        const auto value = simpleInteger(assume->cond.rhs);
        if (value && assume->cond.op == RelOp::GE)
          low = *value;
        else if (value && assume->cond.op == RelOp::LE)
          high = *value;
      }
      if (!low || !high)
        return std::nullopt;
      return std::pair{*low, *high};
    }

    std::optional<std::string> assignedLocal(const Instr &instruction) {
      const auto *assignment = std::get_if<AssignInstr>(&instruction);
      if (!assignment || !assignment->lhs.accesses.empty())
        return std::nullopt;
      return assignment->lhs.base.name;
    }

    bool isIncrementByOne(const Expr &expr, const std::string &name) {
      if (expr.rest.size() != 1 || simpleLocalAtom(expr.first) != name ||
          expr.rest.front().op != AddOp::Plus)
        return false;
      return simpleIntegerAtom(expr.rest.front().atom) == 1;
    }

    bool assignsLiteral(const Instr &instruction, const std::string &name, std::int64_t *value) {
      const auto *assignment = std::get_if<AssignInstr>(&instruction);
      if (!assignment || assignment->lhs.base.name != name || !assignment->lhs.accesses.empty())
        return false;
      const auto literal = simpleInteger(assignment->rhs);
      if (!literal)
        return false;
      *value = *literal;
      return true;
    }

  } // namespace

  std::optional<TensorLoopAnalysis> analyzeTensorLoops(
      const FunDecl &function, const TensorProfile &profile, DiagBag &diags
  ) {
    TensorLoopAnalysis result;
    if (!profile.maxLoopBodyExecutions)
      return result;

    CFG cfg = CFG::build(function, diags);
    if (diags.hasErrors())
      return std::nullopt;
    if (!ReducibilityResult::isReducible(function)) {
      diags.error("tensor-profile: counted-loop control flow must be reducible", function.span);
      return std::nullopt;
    }
    const auto reachable = cfg.rpo();
    if (reachable.size() != cfg.blocks.size()) {
      diags.error("tensor-profile: unreachable blocks are not supported", function.span);
      return std::nullopt;
    }

    const DomTree dominators = DomTree::build(cfg);
    const LoopInfo loopInfo = LoopInfo::build(cfg, dominators);
    result.loops.reserve(loopInfo.loops.size());
    std::vector<std::vector<char>> loopMembers;
    std::vector<char> loopHeader(cfg.blocks.size(), 0);
    std::vector<char> loopLatch(cfg.blocks.size(), 0);
    for (const auto &loop: loopInfo.loops) {
      loopHeader[loop.header] = 1;
      for (const auto latch: loop.latches)
        loopLatch[latch] = 1;
    }

    auto fail = [&](const std::string &message, std::size_t block) {
      diags.error(message, function.blocks[block].span);
      return std::optional<TensorLoopAnalysis>{};
    };

    for (std::size_t index = 0; index < loopInfo.loops.size(); ++index) {
      const auto &natural = loopInfo.loops[index];
      const auto header = natural.header;
      if (natural.latches.size() != 1 || natural.exitEdges.size() != 1)
        return fail("tensor-profile: counted loop needs one latch and one false-edge exit", header);
      const auto *branch = std::get_if<BrTerm>(&function.blocks[header].term);
      if (!branch || !branch->isConditional || !branch->cond)
        return fail("tensor-profile: counted-loop header must compare induction variable", header);
      if (!function.blocks[header].instrs.empty())
        return fail("tensor-profile: counted-loop header cannot contain instructions", header);
      const auto thenIt = cfg.indexOf.find(branch->thenLabel.name);
      const auto elseIt = cfg.indexOf.find(branch->elseLabel.name);
      if (thenIt == cfg.indexOf.end() || elseIt == cfg.indexOf.end())
        return fail("tensor-profile: invalid counted-loop branch target", header);

      std::vector<char> members(cfg.blocks.size(), 0);
      for (const auto block: natural.blocks)
        members[block] = 1;
      if (!members[thenIt->second] || members[elseIt->second] ||
          natural.exitEdges.front() != std::pair{header, elseIt->second})
        return fail("tensor-profile: counted-loop true edge must enter body and false edge exit", header);

      std::vector<std::size_t> preheaders;
      for (const auto predecessor: cfg.pred[header])
        if (!members[predecessor])
          preheaders.push_back(predecessor);
      if (preheaders.size() != 1)
        return fail("tensor-profile: counted loop needs one preheader", header);
      const auto preheader = preheaders.front();
      const auto *preheaderBranch = std::get_if<BrTerm>(&function.blocks[preheader].term);
      if (!preheaderBranch || preheaderBranch->isConditional ||
          preheaderBranch->dest.name != function.blocks[header].label.name)
        return fail("tensor-profile: counted-loop preheader must branch directly to its header", header);
      if (branch->cond->op != RelOp::LT)
        return fail("tensor-profile: counted-loop condition must use <", header);
      const auto induction = simpleLocal(branch->cond->lhs);
      auto bound = simpleInteger(branch->cond->rhs);
      const auto boundParameter = bound ? std::nullopt : simpleLocal(branch->cond->rhs);
      std::optional<std::pair<std::int64_t, std::int64_t>> range;
      if (boundParameter) {
        range = assumedParameterRange(function, *boundParameter);
        if (!range)
          return fail(
              "tensor-profile: a parameter loop bound must be an i32 entry parameter ranged by "
              "`assume " +
                  *boundParameter + " >= lo;` and `assume " + *boundParameter +
                  " <= hi;` in ^entry",
              header
          );
        bound = range->second;
      }
      if (!induction || !bound)
        return fail("tensor-profile: counted-loop header must compare induction variable", header);

      const auto declaration = std::find_if(
          function.lets.begin(), function.lets.end(), [&](const LetDecl &let) {
            return let.name.name == *induction;
          }
      );
      if (declaration == function.lets.end() || !declaration->isMutable ||
          TypeUtils::getIntBitWidth(declaration->type) != 32 ||
          (declaration->init && declaration->init->kind != InitVal::Kind::Undef))
        return fail("tensor-profile: induction variable must be a mutable uninitialized i32 local", header);

      std::int64_t initial = 0;
      std::size_t preheaderInitializers = 0;
      std::size_t latchIncrements = 0;
      std::size_t writes = 0;
      for (std::size_t block = 0; block < function.blocks.size(); ++block) {
        const auto &instructions = function.blocks[block].instrs;
        for (std::size_t instructionIndex = 0; instructionIndex < instructions.size();
             ++instructionIndex) {
          const auto &instruction = instructions[instructionIndex];
          const auto target = assignedLocal(instruction);
          if (!target || *target != *induction)
            continue;
          ++writes;
          if (block == preheader && assignsLiteral(instruction, *induction, &initial))
            ++preheaderInitializers;
          if (block == natural.latches.front()) {
            const auto *assignment = std::get_if<AssignInstr>(&instruction);
            if (!assignment || !isIncrementByOne(assignment->rhs, *induction) ||
                instructionIndex + 1 != instructions.size())
              return fail("tensor-profile: latch must increment induction variable by one", header);
            ++latchIncrements;
          }
        }
      }
      if (preheaderInitializers != 1)
        return fail("tensor-profile: preheader must reset induction variable on every entry", header);
      if (latchIncrements != 1)
        return fail("tensor-profile: latch must increment induction variable by one", header);
      if (writes != 2)
        return fail("tensor-profile: induction variable may only be initialized and incremented", header);
      if (initial < INT32_MIN || initial > INT32_MAX || *bound < INT32_MIN || *bound > INT32_MAX ||
          *bound <= initial || *bound - initial > 8)
        return fail(
            range ? "tensor-profile: a parameter loop bound's range must allow one to eight trips"
                  : "tensor-profile: loop trip count must be in [1, 8] without i32 overflow",
            header
        );
      // A parameter bound below the start would leave the induction variable
      // at its start, not at the bound, after the loop.
      if (range && range->first < initial)
        return fail(
            "tensor-profile: a parameter loop bound's range starts below the loop's start", header
        );

      const auto latch = natural.latches.front();
      const auto *latchBranch = std::get_if<BrTerm>(&function.blocks[latch].term);
      if (!latchBranch || latchBranch->isConditional ||
          latchBranch->dest.name != function.blocks[header].label.name)
        return fail("tensor-profile: counted-loop latch must branch to its header", header);

      TensorCountedLoop loop;
      loop.header = header;
      loop.preheader = preheader;
      loop.latch = latch;
      loop.exit = elseIt->second;
      loop.inductionVariable = *induction;
      loop.initialValue = static_cast<std::int32_t>(initial);
      loop.bound = static_cast<std::int32_t>(*bound);
      if (boundParameter)
        loop.boundParameter = *boundParameter;
      loop.tripCount = static_cast<std::uint32_t>(*bound - initial);
      loop.parentIndex = natural.parent;
      loop.blocks = natural.blocks;
      result.loops.push_back(std::move(loop));
      loopMembers.push_back(std::move(members));
    }

    for (std::size_t index = 0; index < result.loops.size(); ++index) {
      const auto &loop = result.loops[index];
      for (const auto block: loop.blocks) {
        const auto *term = std::get_if<BrTerm>(&function.blocks[block].term);
        if (!term)
          return fail("tensor-profile: counted-loop body cannot return or terminate early", block);
        if (term->isConditional && !loopHeader[block])
          return fail("tensor-profile: only counted-loop headers may branch conditionally", block);
        if (!term->isConditional) {
          const auto destination = cfg.indexOf.find(term->dest.name);
          if (destination == cfg.indexOf.end())
            return fail("tensor-profile: invalid counted-loop branch target", block);
          if (block == loop.latch) {
            if (destination->second != loop.header)
              return fail("tensor-profile: counted-loop latch must branch to its header", block);
          } else if (!loopMembers[index][destination->second]) {
            return fail("tensor-profile: counted-loop body has an early exit", block);
          }
        }
      }
    }

    std::uint64_t totalExecutions = 0;
    for (std::size_t index = 0; index < result.loops.size(); ++index) {
      std::uint64_t multiplier = 1;
      for (int parent = result.loops[index].parentIndex; parent >= 0;
           parent = result.loops[static_cast<std::size_t>(parent)].parentIndex)
        multiplier *= result.loops[static_cast<std::size_t>(parent)].tripCount;
      totalExecutions += multiplier * result.loops[index].tripCount;
    }
    if (totalExecutions > profile.maxLoopBodyExecutions) {
      diags.error(
          "tensor-profile: counted loops exceed eight total body executions", function.span
      );
      return std::nullopt;
    }
    result.totalBodyExecutions = static_cast<std::uint32_t>(totalExecutions);

    std::vector<char> inAnyLoop(cfg.blocks.size(), 0);
    for (const auto &loop: result.loops)
      for (const auto block: loop.blocks)
        inAnyLoop[block] = 1;
    for (std::size_t block = 0; block < function.blocks.size(); ++block) {
      const auto *branch = std::get_if<BrTerm>(&function.blocks[block].term);
      if (branch && branch->isConditional && !loopHeader[block])
        return fail("tensor-profile: conditional branches must be canonical counted-loop headers", block);
      if (loopLatch[block] && !loopHeader[block] && !inAnyLoop[block])
        return fail("tensor-profile: loop latch is outside its natural loop", block);
    }
    return result;
  }

  bool checkTensorCapabilities(
      const Program &prog, TensorConsumer consumer, const TensorProfile &profile,
      DiagBag &diags
  ) {
    const bool legacy = consumer == TensorConsumer::C || consumer == TensorConsumer::Wasm ||
                        consumer == TensorConsumer::Python;
    const auto structs = TypeUtils::buildStructTable(prog);
    bool hasTensor = programHasTensorAtom(prog);
    std::vector<std::pair<std::string, SourceSpan>> profileErrors;
    eachProgramType(prog, [&](const TypePtr &type, SourceSpan span) {
      std::vector<std::string> errors;
      hasTensor = containsTensorType(type, structs, profile, &errors) || hasTensor;
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
      if (!functionHasTensorType(function, structs, profile) && !functionHasTensorAtom(function))
        continue;
      const bool hasRemainder = functionHasTensorOperation(function, TensorOp::Rem);
      if (profile.id != TensorProfileId::V4 && hasRemainder)
        diags.error("tensor-profile: tensor rem requires tensor-v4", function.span);
      if (profile.id == TensorProfileId::V4 && !isV4RemainderKernel(function))
        diags.error(
            "tensor-v4: only a directly returned equal-shaped f32 vector remainder is supported",
            function.span
        );
      if (profile.id == TensorProfileId::V4 && consumer == TensorConsumer::ExactSolver &&
          hasRemainder)
        diags.error(
            "unsupported_theory: exact solver does not implement f32 tensor rem", function.span
        );
      if (profile.id == TensorProfileId::V3 &&
          functionHasUnsupportedV3TensorOperation(function))
        diags.error(
            "tensor-profile: unsupported tensor operation in tensor-v3 profile",
            function.span
        );
      if (profile.id != TensorProfileId::V2 && profile.id != TensorProfileId::V3 &&
          functionHasTensorOperation(function, TensorOp::Window))
        diags.error(
            "tensor-profile: tensor window requires tensor-v2 or tensor-v3", function.span
        );
      if (profile.id == TensorProfileId::V1) {
        if (function.blocks.size() != 1 ||
            !std::holds_alternative<RetTerm>(function.blocks.front().term))
          diags.error(
              "tensor-profile: tensor functions must have one basic block ending in ret",
              function.span
          );
      } else if (!analyzeTensorLoops(function, profile, diags)) {
        continue;
      }
      if (tensorFunctionHasCall(function))
        diags.error("tensor-profile: tensor functions cannot contain calls", function.span);
    }
    return !diags.hasErrors();
  }

} // namespace refractir
