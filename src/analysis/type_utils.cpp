#include "analysis/type_utils.hpp"

#include <functional>

namespace refractir {

  std::optional<std::uint32_t> TypeUtils::getIntBitWidth(const TypePtr &t) {
    if (!t)
      return std::nullopt;
    if (auto it = std::get_if<IntType>(&t->v)) {
      switch (it->kind) {
        case IntType::Kind::I32:
          return 32;
        case IntType::Kind::I64:
          return 64;
        case IntType::Kind::ICustom:
          return it->bits.value_or(0);
      }
    }
    return std::nullopt;
  }

  std::optional<std::uint32_t> TypeUtils::getFloatBitWidth(const TypePtr &t) {
    if (!t)
      return std::nullopt;
    if (auto ft = std::get_if<FloatType>(&t->v)) {
      return ft->kind == FloatType::Kind::F32 ? 32u : 64u;
    }
    return std::nullopt;
  }

  std::optional<std::uint32_t> TypeUtils::getScalarBitWidth(const TypePtr &t) {
    if (!t)
      return std::nullopt;
    // Integer types
    if (auto ib = getIntBitWidth(t))
      return ib;
    // Float types
    return getFloatBitWidth(t);
  }

  std::optional<std::uint32_t> TypeUtils::getVectorBitWidth(const TypePtr &t) {
    if (!t)
      return std::nullopt;
    if (auto vt = std::get_if<VecType>(&t->v)) {
      if (auto elemBits = getScalarBitWidth(vt->elem))
        return static_cast<std::uint32_t>(vt->size) * (*elemBits);
    }
    return std::nullopt;
  }

  std::optional<std::uint32_t> TypeUtils::getBitWidth(const TypePtr &t) {
    if (auto sb = getScalarBitWidth(t))
      return sb;
    return getVectorBitWidth(t);
  }

  bool TypeUtils::areTypesEqual(const TypePtr &a, const TypePtr &b) {
    if (a.get() == b.get())
      return true;
    if (!a || !b)
      return false;
    if (a->v.index() != b->v.index())
      return false;

    if (std::holds_alternative<IntType>(a->v))
      return getIntBitWidth(a) == getIntBitWidth(b);
    if (auto fa = std::get_if<FloatType>(&a->v)) {
      auto fb = std::get_if<FloatType>(&b->v);
      return fa->kind == fb->kind;
    }
    if (auto sa = std::get_if<StructType>(&a->v)) {
      return sa->name.name == std::get<StructType>(b->v).name.name;
    }
    if (auto aa = std::get_if<ArrayType>(&a->v)) {
      auto ab = std::get_if<ArrayType>(&b->v);
      return aa->size == ab->size && areTypesEqual(aa->elem, ab->elem);
    }
    if (auto pa = std::get_if<PtrType>(&a->v)) {
      auto pb = std::get_if<PtrType>(&b->v);
      return areTypesEqual(pa->pointee, pb->pointee);
    }
    if (auto va = std::get_if<VecType>(&a->v)) {
      auto vb = std::get_if<VecType>(&b->v);
      return va->size == vb->size && areTypesEqual(va->elem, vb->elem);
    }
    if (const auto ta = std::get_if<TensorType>(&a->v)) {
      const auto tb = std::get_if<TensorType>(&b->v);
      return ta->shape == tb->shape && areTypesEqual(ta->elem, tb->elem);
    }
    return false;
  }

  const ArrayType *TypeUtils::asArray(const TypePtr &t) {
    return t ? std::get_if<ArrayType>(&t->v) : nullptr;
  }

  const StructType *TypeUtils::asStruct(const TypePtr &t) {
    return t ? std::get_if<StructType>(&t->v) : nullptr;
  }

  bool TypeUtils::isArray(const TypePtr &t) { return t && std::holds_alternative<ArrayType>(t->v); }

  bool TypeUtils::isStruct(const TypePtr &t) {
    return t && std::holds_alternative<StructType>(t->v);
  }

  const VecType *TypeUtils::asVec(const TypePtr &t) {
    return t ? std::get_if<VecType>(&t->v) : nullptr;
  }

  const TensorType *TypeUtils::asTensor(const TypePtr &t) {
    return t ? std::get_if<TensorType>(&t->v) : nullptr;
  }

  bool TypeUtils::isVec(const TypePtr &t) { return t && std::holds_alternative<VecType>(t->v); }

  bool TypeUtils::isTensor(const TypePtr &t) {
    return t && std::holds_alternative<TensorType>(t->v);
  }

  const PtrType *TypeUtils::asPtr(const TypePtr &t) {
    return t ? std::get_if<PtrType>(&t->v) : nullptr;
  }

  TypePtr TypeUtils::pointee(const TypePtr &t) {
    const PtrType *pt = asPtr(t);
    return pt ? pt->pointee : nullptr;
  }

  bool TypeUtils::isInt(const TypePtr &t) { return t && std::holds_alternative<IntType>(t->v); }

  bool TypeUtils::isFloat(const TypePtr &t) { return t && std::holds_alternative<FloatType>(t->v); }

  bool TypeUtils::isPtr(const TypePtr &t) { return t && std::holds_alternative<PtrType>(t->v); }

  bool TypeUtils::isScalar(const TypePtr &t) { return isInt(t) || isFloat(t); }

  bool TypeUtils::isAggregate(const TypePtr &t) { return isArray(t) || isStruct(t); }

  TypeUtils::StructTable TypeUtils::buildStructTable(const Program &prog) {
    StructTable table;
    for (const auto &s: prog.structs)
      table[s.name.name] = &s;
    return table;
  }

  TypePtr TypeUtils::stepType(const TypePtr &t, const Access &acc, const StructTable &structs) {
    if (auto af = std::get_if<AccessField>(&acc)) {
      const StructType *st = asStruct(t);
      if (!st)
        return nullptr;
      auto it = structs.find(st->name.name);
      if (it == structs.end())
        return nullptr;
      for (const auto &f: it->second->fields)
        if (f.name == af->field)
          return f.type;
      return nullptr;
    }
    if (const ArrayType *at = asArray(t))
      return at->elem;
    if (const VecType *vt = asVec(t))
      return vt->elem;
    return nullptr;
  }

  TypePtr TypeUtils::accessPathType(
      const TypePtr &t, const std::vector<Access> &accesses, const StructTable &structs
  ) {
    TypePtr cur = t;
    for (const auto &acc: accesses) {
      cur = stepType(cur, acc, structs);
      if (!cur)
        return nullptr;
    }
    return cur;
  }

  bool TypeUtils::containsVec(const TypePtr &t, const StructTable &structs) {
    if (!t)
      return false;
    if (isVec(t))
      return true;
    if (const ArrayType *at = asArray(t))
      return containsVec(at->elem, structs);
    if (const StructType *st = asStruct(t)) {
      auto it = structs.find(st->name.name);
      if (it == structs.end())
        return false;
      for (const auto &f: it->second->fields)
        if (containsVec(f.type, structs))
          return true;
    }
    return false;
  }

  std::uint64_t TypeUtils::packedSizeof(const TypePtr &t, const StructTable &structs) {
    // The 8-byte fallbacks (null type, unknown struct) match the pointer
    // width; they stand in for "no layout information", not a real size.
    if (!t)
      return 8;
    if (auto it = std::get_if<IntType>(&t->v)) {
      std::uint32_t bits = it->bits.value_or(it->kind == IntType::Kind::I32 ? 32 : 64);
      return (bits + 7) / 8;
    }
    if (auto ft = std::get_if<FloatType>(&t->v))
      return ft->kind == FloatType::Kind::F32 ? 4 : 8;
    if (std::holds_alternative<PtrType>(t->v))
      return 8;
    if (auto at = std::get_if<ArrayType>(&t->v))
      return at->size * packedSizeof(at->elem, structs);
    if (auto st = std::get_if<StructType>(&t->v)) {
      auto it = structs.find(st->name.name);
      if (it == structs.end())
        return 8;
      std::uint64_t total = 0;
      for (const auto &f: it->second->fields)
        total += packedSizeof(f.type, structs);
      return total;
    }
    return 8;
  }

  std::uint64_t TypeUtils::packedFieldOffset(
      const StructDecl &s, const std::string &field, const StructTable &structs
  ) {
    std::uint64_t off = 0;
    for (const auto &f: s.fields) {
      if (f.name == field)
        return off;
      off += packedSizeof(f.type, structs);
    }
    return off;
  }

  std::vector<std::pair<std::vector<Access>, TypePtr>>
  TypeUtils::scalarLeaves(const TypePtr &t, const StructTable &structs) {
    std::vector<std::pair<std::vector<Access>, TypePtr>> leaves;
    std::vector<Access> path;
    const std::function<void(const TypePtr &)> walk = [&](const TypePtr &type) {
      if (!type)
        return;
      if (getIntBitWidth(type) || getFloatBitWidth(type) || isPtr(type)) {
        leaves.emplace_back(path, type);
        return;
      }
      const auto descend = [&](Access step, const TypePtr &elem) {
        path.push_back(std::move(step));
        walk(elem);
        path.pop_back();
      };
      const auto index = [](std::uint64_t i) {
        return Access{AccessIndex{Index{IntLit{static_cast<std::int64_t>(i), {}}}, {}}};
      };
      if (auto arr = std::get_if<ArrayType>(&type->v)) {
        for (std::uint64_t i = 0; i < arr->size; ++i)
          descend(index(i), arr->elem);
      } else if (auto vec = std::get_if<VecType>(&type->v)) {
        for (std::uint64_t i = 0; i < vec->size; ++i)
          descend(index(i), vec->elem);
      } else if (auto str = std::get_if<StructType>(&type->v)) {
        const auto decl = structs.find(str->name.name);
        if (decl == structs.end() || !decl->second)
          return;
        for (const auto &field: decl->second->fields)
          descend(Access{AccessField{field.name, {}}}, field.type);
      }
    };
    walk(t);
    return leaves;
  }

} // namespace refractir
