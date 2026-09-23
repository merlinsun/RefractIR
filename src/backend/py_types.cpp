#include <stdexcept>
#include "backend/py_backend.hpp"

#include "analysis/type_utils.hpp"

namespace refractir {

  std::uint32_t PyBackend::intWidth(const TypePtr &t) {
    if (!t)
      return 0;
    if (auto it = std::get_if<IntType>(&t->v)) {
      switch (it->kind) {
        case IntType::Kind::I32:
          return 32;
        case IntType::Kind::I64:
          return 64;
        case IntType::Kind::ICustom:
          return static_cast<std::uint32_t>(it->bits.value_or(32));
      }
    }
    return 0;
  }

  std::uint32_t PyBackend::floatWidth(const TypePtr &t) {
    if (!t)
      return 0;
    if (auto ft = std::get_if<FloatType>(&t->v))
      return ft->kind == FloatType::Kind::F32 ? 32 : 64;
    return 0;
  }

  bool PyBackend::containsF64(const TypePtr &t) {
    if (!t)
      return false;
    return std::visit(
        [](const auto &arg) -> bool {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, FloatType>) {
            return arg.kind == FloatType::Kind::F64;
          } else if constexpr (std::is_same_v<T, ArrayType>) {
            return containsF64(arg.elem);
          } else if constexpr (std::is_same_v<T, VecType>) {
            return containsF64(arg.elem);
          } else if constexpr (std::is_same_v<T, PtrType>) {
            return containsF64(arg.pointee);
          } else {
            return false; // int / struct (fields handled at access time)
          }
        },
        t->v
    );
  }

  void PyBackend::requireSupportedType(const TypePtr &t, const char *what) const {
    if (!t)
      return;
    if (auto pt = std::get_if<PtrType>(&t->v))
      requireSupportedType(pt->pointee, what);
    else if (auto at = std::get_if<ArrayType>(&t->v))
      requireSupportedType(at->elem, what);
  }

  std::uint64_t PyBackend::byteSize(const TypePtr &t) const {
    // Packed bytes, from the one layout authority shared with the
    // interpreter and the solver. The buffer below is byte-indexed, so a
    // leaf lives at its byte offset and the slots it spans past the first
    // are padding — see flattenInit.
    TypeUtils::StructTable tbl;
    if (prog_)
      for (const auto &sd: prog_->structs)
        tbl[sd.name.name] = &sd;
    return TypeUtils::packedSizeof(t, tbl);
  }

  std::uint64_t PyBackend::fieldByteOffset(
      const std::string &structName, const std::string &field, TypePtr *fieldType
  ) const {
    auto it = structFields_.find(structName);
    if (it == structFields_.end())
      throw std::runtime_error("python target: unknown struct type");
    std::uint64_t off = 0;
    for (const auto &[name, fty]: it->second) {
      if (name == field) {
        if (fieldType)
          *fieldType = fty;
        return off;
      }
      off += byteSize(fty);
    }
    throw std::runtime_error("python target: unknown struct field " + field);
  }

  TypePtr PyBackend::getLValueType(const LValue &lv) {
    auto it = varTypes_.find(lv.base.name);
    if (it == varTypes_.end())
      return nullptr;
    TypePtr cur = it->second;
    for (const auto &acc: lv.accesses) {
      if (!cur)
        return nullptr;
      if (std::holds_alternative<AccessIndex>(acc)) {
        if (auto at = std::get_if<ArrayType>(&cur->v))
          cur = at->elem;
        else if (auto vt = std::get_if<VecType>(&cur->v))
          cur = vt->elem;
        else if (auto pt = std::get_if<PtrType>(&cur->v))
          cur = pt->pointee;
        else
          return nullptr;
      } else if (auto af = std::get_if<AccessField>(&acc)) {
        auto st = std::get_if<StructType>(&cur->v);
        if (!st)
          return nullptr;
        TypePtr fieldTy;
        fieldByteOffset(st->name.name, af->field, &fieldTy);
        cur = fieldTy;
      }
    }
    return cur;
  }

  TypePtr PyBackend::getCoefType(const Coef &coef) {
    return std::visit(
        [this](auto &&arg) -> TypePtr {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, IntLit>) {
            auto t = std::make_shared<Type>();
            // Prefer the width the type checker pinned; fall back to
            // the value-range heuristic for un-typechecked input.
            std::uint32_t bits = arg.resolvedBits;
            if (bits == 0)
              bits = (arg.value > 2147483647LL || arg.value < -2147483648LL) ? 64 : 32;
            t->v = IntType{IntType::Kind::ICustom, static_cast<int>(bits), {}};
            return t;
          } else if constexpr (std::is_same_v<T, FloatLit>) {
            auto t = std::make_shared<Type>();
            t->v =
                FloatType{arg.resolvedBits == 32 ? FloatType::Kind::F32 : FloatType::Kind::F64, {}};
            return t;
          } else if constexpr (std::is_same_v<T, NullLit>) {
            auto t = std::make_shared<Type>();
            t->v = PtrType{nullptr, {}};
            return t;
          } else {
            return std::visit(
                [this](auto &&id) -> TypePtr {
                  auto it = varTypes_.find(id.name);
                  return it != varTypes_.end() ? it->second : nullptr;
                },
                arg
            );
          }
        },
        coef
    );
  }

  TypePtr PyBackend::getSelectValType(const SelectVal &sv) {
    if (std::holds_alternative<RValue>(sv))
      return getLValueType(std::get<RValue>(sv));
    return getCoefType(std::get<Coef>(sv));
  }

  TypePtr PyBackend::getAtomType(const Atom &atom) {
    return std::visit(
        [this](auto &&arg) -> TypePtr {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, OpAtom>) {
            return getLValueType(arg.rval);
          } else if constexpr (std::is_same_v<T, UnaryAtom>) {
            return getLValueType(arg.rval);
          } else if constexpr (std::is_same_v<T, SelectAtom>) {
            return getSelectValType(arg.vtrue);
          } else if constexpr (std::is_same_v<T, CmpAtom>) {
            auto i1 = std::make_shared<Type>();
            i1->v = IntType{IntType::Kind::ICustom, 1, {}};
            auto lt = getSelectValType(arg.lhs);
            if (lt) {
              if (auto vt = std::get_if<VecType>(&lt->v)) {
                auto t = std::make_shared<Type>();
                t->v = VecType{vt->size, i1, {}};
                return t;
              }
            }
            return i1;
          } else if constexpr (std::is_same_v<T, CoefAtom>) {
            return getCoefType(arg.coef);
          } else if constexpr (std::is_same_v<T, RValueAtom>) {
            return getLValueType(arg.rval);
          } else if constexpr (std::is_same_v<T, CastAtom>) {
            return arg.dstType;
          } else if constexpr (std::is_same_v<T, CallAtom>) {
            if (arg.resolvedIntrinsic)
              return arg.resolvedIntrinsic->retType;
            for (const auto &f: prog_->funs)
              if (f.name.name == arg.callee.name)
                return f.retType;
            for (const auto &d: prog_->extDecls)
              if (d.name.name == arg.callee.name)
                return d.retType;
            return nullptr;
          } else if constexpr (std::is_same_v<T, LoadAtom>) {
            auto pt = getLValueType(arg.rval);
            if (pt)
              if (auto ptr = std::get_if<PtrType>(&pt->v))
                return ptr->pointee;
            return nullptr;
          } else if constexpr (std::is_same_v<T, AddrAtom>) {
            auto t = std::make_shared<Type>();
            t->v = PtrType{getLValueType(arg.lv), {}};
            return t;
          } else if constexpr (std::is_same_v<T, PtrIndexAtom>) {
            // ptrindex p, i : ptr [N] T -> ptr T
            auto pt = getLValueType(arg.rval);
            if (pt)
              if (auto ptr = std::get_if<PtrType>(&pt->v))
                if (auto at = std::get_if<ArrayType>(&ptr->pointee->v)) {
                  auto res = std::make_shared<Type>();
                  res->v = PtrType{at->elem, {}};
                  return res;
                }
            return nullptr;
          } else if constexpr (std::is_same_v<T, TensorAtom>) {
            return nullptr;
          } else {
            static_assert(std::is_same_v<T, PtrFieldAtom>);
            // ptrfield p, f : ptr @S -> ptr FieldType
            auto pt = getLValueType(arg.rval);
            if (pt)
              if (auto ptr = std::get_if<PtrType>(&pt->v))
                if (auto st = std::get_if<StructType>(&ptr->pointee->v)) {
                  TypePtr fieldTy;
                  fieldByteOffset(st->name.name, arg.field, &fieldTy);
                  auto res = std::make_shared<Type>();
                  res->v = PtrType{fieldTy, {}};
                  return res;
                }
            return nullptr;
          }
        },
        atom.v
    );
  }

  // Every atom in an Expr shares one type, so the first atom decides.
  TypePtr PyBackend::getExprType(const Expr &expr) { return getAtomType(expr.first); }

} // namespace refractir
