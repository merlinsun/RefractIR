#include <stdexcept>
#include "analysis/tensor.hpp"
#include "analysis/type_utils.hpp"
#include "internal.hpp"
#include "solver/solver.hpp"

namespace refractir {

  smt::Sort SymbolicExecutor::getSort(const TypePtr &t, smt::ISolver &solver) {
    if (auto it = std::get_if<IntType>(&t->v)) {
      uint32_t bits = 32;
      switch (it->kind) {
        case IntType::Kind::I32:
          bits = 32;
          break;
        case IntType::Kind::I64:
          bits = 64;
          break;
        case IntType::Kind::ICustom:
          bits = it->bits.value_or(32);
          break;
      }
      return solver.make_bv_sort(bits);
    }
    if (auto ft = std::get_if<FloatType>(&t->v)) {
      if (ft->kind == FloatType::Kind::F32)
        return solver.make_fp_sort(8, 24);
      else
        return solver.make_fp_sort(11, 53);
    }
    if (auto at = std::get_if<ArrayType>(&t->v)) {
      return getSort(at->elem, solver);
    }
    if (auto st = std::get_if<StructType>(&t->v)) {
      auto it = structs_.find(st->name.name);
      if (it != structs_.end() && !it->second->fields.empty()) {
        return getSort(it->second->fields[0].type, solver);
      }
    }
    if (std::holds_alternative<PtrType>(t->v)) {
      return solver.make_bv_sort(kPtrBits);
    }
    if (auto vt = std::get_if<VecType>(&t->v)) {
      // Vectors aren't a single SMT sort; lanes are held as N
      // independent terms in SymbolicValue::arrayVal. getSort returns the
      // lane sort so any downstream caller that wants "what kind of term
      // is in each lane?" gets the right answer.
      return getSort(vt->elem, solver);
    }
    if (auto tensor = std::get_if<TensorType>(&t->v))
      return getSort(tensor->elem, solver);
    throw std::runtime_error("Unknown type or empty struct in getSort");
  }

  SymbolicValue SymbolicExecutor::createSymbolicValue(
      const TypePtr &t, const std::string &name, smt::ISolver &solver,
      std::vector<smt::Term> *finiteSink
  ) {
    SymbolicValue res;
    if (auto tensor = std::get_if<TensorType>(&t->v)) {
      res.kind = SymbolicValue::Kind::Tensor;
      res.tensorShape = tensor->shape;
      const auto count = tensorElementCount(tensor->shape);
      for (std::uint32_t i = 0; i < count; ++i)
        res.arrayVal.push_back(createSymbolicValue(
            tensor->elem, name + "[" + std::to_string(i) + "]", solver, finiteSink
        ));
    } else if (auto at = std::get_if<ArrayType>(&t->v)) {
      res.kind = SymbolicValue::Kind::Array;
      for (size_t i = 0; i < at->size; ++i) {
        res.arrayVal.push_back(
            createSymbolicValue(at->elem, name + "[" + std::to_string(i) + "]", solver, finiteSink)
        );
      }
    } else if (auto vt = std::get_if<VecType>(&t->v)) {
      // Vector sym: N independent lane-symbolic constants
      // (§9.5.1). Same shape as Array but tagged Vec so downstream
      // dispatch picks the lane-wise UB path.
      res.kind = SymbolicValue::Kind::Vec;
      for (size_t i = 0; i < vt->size; ++i) {
        res.arrayVal.push_back(
            createSymbolicValue(vt->elem, name + "[" + std::to_string(i) + "]", solver, finiteSink)
        );
      }
    } else if (auto st = std::get_if<StructType>(&t->v)) {
      res.kind = SymbolicValue::Kind::Struct;
      auto it = structs_.find(st->name.name);
      if (it != structs_.end()) {
        for (const auto &f: it->second->fields) {
          res.structVal[f.name] =
              createSymbolicValue(f.type, name + "." + f.name, solver, finiteSink);
        }
      }
    } else {
      res.kind = SymbolicValue::Kind::Int;
      res.term = solver.make_const(getSort(t, solver), name);
      res.is_defined = solver.make_true();
      // Domain invariant: a fresh FP input is a RefractIR value, so it is
      // finite (SPEC v0.2.2 §5). Assert it as a hard path constraint — never a
      // negatable ubGuard — so RequireUB cannot satisfy "trigger UB" by
      // choosing a non-finite (thus non-representable) input/sym. Overflow to
      // ±∞ from *finite* inputs stays a genuine, negatable result guard.
      if (finiteSink && std::holds_alternative<FloatType>(t->v))
        assertFPFinite(res.term, solver, *finiteSink);
    }
    return res;
  }

  SymbolicValue SymbolicExecutor::makeUndef(const TypePtr &t, smt::ISolver &solver) {
    SymbolicValue res;
    if (auto tensor = std::get_if<TensorType>(&t->v)) {
      res.kind = SymbolicValue::Kind::Tensor;
      res.tensorShape = tensor->shape;
      const auto count = tensorElementCount(tensor->shape);
      for (std::uint32_t i = 0; i < count; ++i)
        res.arrayVal.push_back(makeUndef(tensor->elem, solver));
    } else if (auto at = std::get_if<ArrayType>(&t->v)) {
      res.kind = SymbolicValue::Kind::Array;
      for (size_t i = 0; i < at->size; ++i)
        res.arrayVal.push_back(makeUndef(at->elem, solver));
    } else if (auto vt = std::get_if<VecType>(&t->v)) {
      res.kind = SymbolicValue::Kind::Vec;
      for (size_t i = 0; i < vt->size; ++i)
        res.arrayVal.push_back(makeUndef(vt->elem, solver));
    } else if (auto st = std::get_if<StructType>(&t->v)) {
      res.kind = SymbolicValue::Kind::Struct;
      auto it = structs_.find(st->name.name);
      if (it != structs_.end()) {
        for (const auto &f: it->second->fields)
          res.structVal[f.name] = makeUndef(f.type, solver);
      }
    } else {
      res.kind = SymbolicValue::Kind::Int;
      res.term = solver.make_const(getSort(t, solver), "undef");
      res.is_defined = solver.make_false();
    }
    return res;
  }

  SymbolicValue SymbolicExecutor::broadcast(const TypePtr &t, smt::Term val, smt::ISolver &solver) {
    if (auto tensor = std::get_if<TensorType>(&t->v)) {
      SymbolicValue res;
      res.kind = SymbolicValue::Kind::Tensor;
      res.tensorShape = tensor->shape;
      const auto count = tensorElementCount(tensor->shape);
      for (std::uint32_t i = 0; i < count; ++i)
        res.arrayVal.push_back(broadcast(tensor->elem, val, solver));
      return res;
    } else if (std::holds_alternative<ArrayType>(t->v)) {
      SymbolicValue res;
      res.kind = SymbolicValue::Kind::Array;
      const auto &at = std::get<ArrayType>(t->v);
      for (size_t i = 0; i < at.size; ++i)
        res.arrayVal.push_back(broadcast(at.elem, val, solver));
      return res;
    } else if (std::holds_alternative<VecType>(t->v)) {
      SymbolicValue res;
      res.kind = SymbolicValue::Kind::Vec;
      const auto &vt = std::get<VecType>(t->v);
      for (size_t i = 0; i < vt.size; ++i)
        res.arrayVal.push_back(broadcast(vt.elem, val, solver));
      return res;
    } else if (std::holds_alternative<StructType>(t->v)) {
      SymbolicValue res;
      res.kind = SymbolicValue::Kind::Struct;
      auto it = structs_.find(std::get<StructType>(t->v).name.name);
      if (it != structs_.end()) {
        for (const auto &f: it->second->fields)
          res.structVal[f.name] = broadcast(f.type, val, solver);
      }
      return res;
    } else {
      // Scalar leaf (IntType, FloatType, PtrType).
      // The incoming `val` term may have been constructed for a *different*
      // scalar type (e.g., BV[8] when this leaf is i32) because evalInit
      // derives the zero from getSort(struct), which returns the sort of
      // the first field's element — not this particular field's sort.
      // Resize BV terms to the correct width so every field gets a
      // correctly-sorted SMT constant.  All callers that broadcast an
      // integer literal zero just want "zero of the right width", so
      // sign-extension (= zero-extension for 0) is always correct here.
      smt::Term leaf = val;
      auto targetSort = getSort(t, solver);
      auto valSort = solver.get_sort(val);
      if (solver.is_bv_sort(valSort) && solver.is_bv_sort(targetSort)) {
        uint32_t vw = solver.get_bv_width(valSort);
        uint32_t tw = solver.get_bv_width(targetSort);
        if (vw < tw)
          leaf = solver.make_term(smt::Kind::BV_SIGN_EXTEND, {val}, {tw - vw});
        else if (vw > tw)
          // Truncate: re-create the constant at the narrower sort.
          // We only reach here for literal broadcasts (= 0), so the
          // numeric value already fits in tw bits.
          leaf = solver.make_bv_value(targetSort, "0", 10);
      }
      SymbolicValue res;
      res.kind = SymbolicValue::Kind::Int;
      res.term = leaf;
      res.is_defined = solver.make_true();
      return res;
    }
  }

  SymbolicValue SymbolicExecutor::evalInit(
      const InitVal &iv, const TypePtr &t, smt::ISolver &solver, SymbolicStore &store,
      std::vector<smt::Term> &pc, std::vector<smt::Term> &ub
  ) {
    if (iv.kind == InitVal::Kind::Undef)
      return makeUndef(t, solver);

    // Atom-form initializer (addr, load, cmp, ptrindex, etc.)
    if (iv.kind == InitVal::Kind::Atom) {
      const auto &atom = *std::get<AtomPtr>(iv.value);
      if (auto vt = std::get_if<VecType>(&t->v)) {
        return evalVecExprAtom(atom, *vt, solver, store, pc, ub);
      }
      return evalAtom(atom, solver, store, pc, ub, getSort(t, solver));
    }

    if (iv.kind == InitVal::Kind::Aggregate) {
      const auto &elements = std::get<std::vector<InitValPtr>>(iv.value);
      if (auto at = std::get_if<ArrayType>(&t->v)) {
        SymbolicValue res;
        res.kind = SymbolicValue::Kind::Array;
        for (size_t i = 0; i < elements.size(); ++i)
          res.arrayVal.push_back(evalInit(*elements[i], at->elem, solver, store, pc, ub));
        return res;
      } else if (auto tt = std::get_if<TensorType>(&t->v)) {
        SymbolicValue res;
        res.kind = SymbolicValue::Kind::Tensor;
        res.tensorShape = tt->shape;
        for (const auto &element: elements)
          res.arrayVal.push_back(evalInit(*element, tt->elem, solver, store, pc, ub));
        return res;
      } else if (auto vt = std::get_if<VecType>(&t->v)) {
        SymbolicValue res;
        res.kind = SymbolicValue::Kind::Vec;
        for (size_t i = 0; i < elements.size(); ++i)
          res.arrayVal.push_back(evalInit(*elements[i], vt->elem, solver, store, pc, ub));
        return res;
      } else if (auto st = std::get_if<StructType>(&t->v)) {
        SymbolicValue res;
        res.kind = SymbolicValue::Kind::Struct;
        auto it = structs_.find(st->name.name);
        if (it != structs_.end()) {
          for (size_t i = 0; i < elements.size(); ++i)
            res.structVal[it->second->fields[i].name] =
                evalInit(*elements[i], it->second->fields[i].type, solver, store, pc, ub);
        }
        return res;
      }
    }

    // Scalar
    smt::Term val;
    if (iv.kind == InitVal::Kind::Null) {
      // null pointer: BV64 zero.
      val = solver.make_bv_value(solver.make_bv_sort(kPtrBits), "0", 10);
      return broadcast(t, val, solver);
    }
    if (iv.kind == InitVal::Kind::Int) {
      auto lit = std::get<IntLit>(iv.value);
      TypePtr leafType = t;
      while (auto *at = std::get_if<ArrayType>(&leafType->v)) {
        leafType = at->elem;
      }
      val = solver.make_bv_value(getSort(leafType, solver), std::to_string(lit.value), 10);
    } else if (iv.kind == InitVal::Kind::Float) {
      auto lit = std::get<FloatLit>(iv.value);
      TypePtr leafType = t;
      while (auto *at = std::get_if<ArrayType>(&leafType->v)) {
        leafType = at->elem;
      }
      // From the double itself, never a decimal rendering of it: a decimal
      // string parses through a rational, which cannot represent -0.0 and
      // (via `%f`) keeps only six fraction digits. See make_fp_value_from_real.
      val = solver.make_fp_value_from_real(
          getSort(leafType, solver), lit.value, smt::RoundingMode::RNE
      );
    } else if (iv.kind == InitVal::Kind::Sym) {
      return store.at(std::get<SymId>(iv.value).name);
    } else if (iv.kind == InitVal::Kind::Local) {
      return store.at(std::get<LocalId>(iv.value).name);
    }
    return broadcast(t, val, solver);
  }
} // namespace refractir
