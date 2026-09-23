#include <memory>
#include "analysis/type_utils.hpp"
#include "interp/interpreter.hpp"

namespace refractir {

  TypePtr Interpreter::getLValueType(const LValue &lv) const {
    auto tit = typeMap_.find(lv.base.name);
    if (tit == typeMap_.end())
      return nullptr;
    TypePtr cur = tit->second;
    for (const auto &acc: lv.accesses) {
      if (!cur)
        break;
      if (auto ai = std::get_if<AccessIndex>(&acc)) {
        (void) ai;
        if (auto at = std::get_if<ArrayType>(&cur->v))
          cur = at->elem;
        else if (auto vt = std::get_if<VecType>(&cur->v))
          cur = vt->elem;
      } else if (auto af = std::get_if<AccessField>(&acc)) {
        if (auto st = std::get_if<StructType>(&cur->v)) {
          auto sit = typeLayout_.structs().find(st->name.name);
          if (sit != typeLayout_.structs().end()) {
            bool found = false;
            for (const auto &fd: sit->second->fields) {
              if (fd.name == af->field) {
                cur = fd.type;
                found = true;
                break;
              }
            }
            if (!found)
              cur = nullptr;
          } else {
            cur = nullptr;
          }
        } else {
          cur = nullptr;
        }
      }
    }
    return cur;
  }

  TypePtr Interpreter::getCoefType(const Coef &coef) const {
    return std::visit(
        [this](auto &&arg) -> TypePtr {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, IntLit>) {
            // Int literals are polymorphic in RefractIR; we return I32/I64
            // based on the literal value for overload resolution.
            auto t = std::make_shared<Type>();
            if (arg.value > 2147483647LL || arg.value < -2147483648LL) {
              t->v = IntType{IntType::Kind::I64, {}, {}};
            } else {
              t->v = IntType{IntType::Kind::I32, {}, {}};
            }
            return t;
          } else if constexpr (std::is_same_v<T, FloatLit>) {
            auto t = std::make_shared<Type>();
            t->v = FloatType{FloatType::Kind::F64, {}};
            return t;
          } else if constexpr (std::is_same_v<T, NullLit>) {
            auto t = std::make_shared<Type>();
            t->v = PtrType{nullptr, {}};
            return t;
          } else {
            return std::visit(
                [this](auto &&id) -> TypePtr {
                  auto it = typeMap_.find(id.name);
                  if (it != typeMap_.end()) {
                    return it->second;
                  }
                  return nullptr;
                },
                arg
            );
          }
        },
        coef
    );
  }

  TypePtr Interpreter::getAtomType(const Atom &atom) const {
    return std::visit(
        [this](auto &&arg) -> TypePtr {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, OpAtom>) {
            return getLValueType(arg.rval);
          } else if constexpr (std::is_same_v<T, UnaryAtom>) {
            return getLValueType(arg.rval);
          } else if constexpr (std::is_same_v<T, SelectAtom>) {
            if (std::holds_alternative<RValue>(arg.vtrue)) {
              return getLValueType(std::get<RValue>(arg.vtrue));
            } else {
              return getCoefType(std::get<Coef>(arg.vtrue));
            }
          } else if constexpr (std::is_same_v<T, CoefAtom>) {
            return getCoefType(arg.coef);
          } else if constexpr (std::is_same_v<T, RValueAtom>) {
            return getLValueType(arg.rval);
          } else if constexpr (std::is_same_v<T, CastAtom>) {
            return arg.dstType;
          } else if constexpr (std::is_same_v<T, AddrAtom>) {
            auto t = std::make_shared<Type>();
            t->v = PtrType{getLValueType(arg.lv), {}};
            return t;
          } else if constexpr (std::is_same_v<T, LoadAtom>) {
            auto pt = getLValueType(arg.rval);
            if (pt) {
              if (auto ptr = std::get_if<PtrType>(&pt->v)) {
                return ptr->pointee;
              }
            }
            return nullptr;
          } else if constexpr (std::is_same_v<T, PtrIndexAtom>) {
            auto pt = getLValueType(arg.rval);
            if (pt) {
              if (auto ptr = std::get_if<PtrType>(&pt->v)) {
                if (auto at = std::get_if<ArrayType>(&ptr->pointee->v)) {
                  auto resPtr = std::make_shared<Type>();
                  resPtr->v = PtrType{at->elem, {}};
                  return resPtr;
                }
              }
            }
            return nullptr;
          } else if constexpr (std::is_same_v<T, PtrFieldAtom>) {
            auto pt = getLValueType(arg.rval);
            if (pt) {
              if (auto ptr = std::get_if<PtrType>(&pt->v)) {
                if (auto st = std::get_if<StructType>(&ptr->pointee->v)) {
                  auto sit = typeLayout_.structs().find(st->name.name);
                  if (sit != typeLayout_.structs().end()) {
                    for (const auto &fd: sit->second->fields) {
                      if (fd.name == arg.field) {
                        auto resPtr = std::make_shared<Type>();
                        resPtr->v = PtrType{fd.type, {}};
                        return resPtr;
                      }
                    }
                  }
                }
              }
            }
            return nullptr;
          } else if constexpr (std::is_same_v<T, CallAtom>) {
            // Function/intrinsic call returns its return type. For
            // overloaded intrinsics, honour the resolution the type
            // checker pinned onto the AST node.
            if (arg.resolvedIntrinsic)
              return arg.resolvedIntrinsic->retType;
            for (const auto &f: prog_.funs) {
              if (f.name.name == arg.callee.name)
                return f.retType;
            }
            for (const auto &d: prog_.extDecls) {
              if (d.name.name == arg.callee.name)
                return d.retType;
            }
            const IntrinsicDecl *intr = nullptr;
            for (const auto &i: prog_.intrinsics) {
              if (i.name.name != arg.callee.name)
                continue;
              if (i.params.size() != arg.args.size())
                continue;
              if (!intr) {
                intr = &i;
                continue;
              }
              auto bw1 = TypeUtils::getIntBitWidth(intr->params[0].type);
              auto bw2 = TypeUtils::getIntBitWidth(i.params[0].type);
              if (!bw1 || !bw2 || *bw1 == *bw2)
                continue;
              uint32_t argBW = *bw1;
              if (!arg.args.empty()) {
                auto at = getExprType(*arg.args[0]);
                if (at) {
                  if (auto b = TypeUtils::getIntBitWidth(at))
                    argBW = *b;
                }
              }
              if (*bw2 == argBW && *bw1 != argBW)
                intr = &i;
            }
            if (intr)
              return intr->retType;
            return nullptr;
          } else if constexpr (std::is_same_v<T, TensorAtom>) {
            return arg.resolvedType;
          }
          return nullptr;
        },
        atom.v
    );
  }

  TypePtr Interpreter::getExprType(const Expr &expr) const { return getAtomType(expr.first); }
} // namespace refractir
