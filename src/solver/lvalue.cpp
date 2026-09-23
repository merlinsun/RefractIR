#include <stdexcept>
#include "analysis/type_utils.hpp"
#include "internal.hpp"
#include "solver/solver.hpp"

namespace refractir {

  SymbolicValue SymbolicExecutor::mergeAggregate(
      const std::vector<SymbolicValue> &elements, smt::Term idx, smt::ISolver &solver
  ) {
    if (elements.empty())
      return {SymbolicValue::Kind::Undef};

    if (elements[0].kind == SymbolicValue::Kind::Int) {
      smt::Term res = elements[0].term;
      smt::Term defined = elements[0].is_defined;
      auto targetSort = solver.get_sort(res);
      auto idxSort = solver.get_sort(idx);
      for (size_t i = 1; i < elements.size(); ++i) {
        auto i_term = solver.make_bv_value(idxSort, std::to_string(i), 10);
        auto cond = solver.make_term(smt::Kind::EQUAL, {idx, i_term});

        smt::Term nextTerm = elements[i].term;
        auto nextSort = solver.get_sort(nextTerm);
        if (solver.is_bv_sort(targetSort) && solver.is_bv_sort(nextSort)) {
          auto targetWidth = solver.get_bv_width(targetSort);
          auto nextWidth = solver.get_bv_width(nextSort);
          if (nextWidth < targetWidth) {
            nextTerm =
                solver.make_term(smt::Kind::BV_SIGN_EXTEND, {nextTerm}, {targetWidth - nextWidth});
          } else if (nextWidth > targetWidth) {
            res = solver.make_term(smt::Kind::BV_SIGN_EXTEND, {res}, {nextWidth - targetWidth});
            targetSort = solver.get_sort(res);
          }
        }

        res = solver.make_term(smt::Kind::ITE, {cond, nextTerm, res});
        defined = solver.make_term(smt::Kind::ITE, {cond, elements[i].is_defined, defined});
      }
      return SymbolicValue(SymbolicValue::Kind::Int, res, defined);
    } else if (elements[0].kind == SymbolicValue::Kind::Array) {
      SymbolicValue res;
      res.kind = SymbolicValue::Kind::Array;
      size_t inner_size = elements[0].arrayVal.size();
      for (size_t j = 0; j < inner_size; ++j) {
        std::vector<SymbolicValue> inner_elements;
        for (size_t i = 0; i < elements.size(); ++i)
          inner_elements.push_back(elements[i].arrayVal[j]);
        res.arrayVal.push_back(mergeAggregate(inner_elements, idx, solver));
      }
      return res;
    } else if (elements[0].kind == SymbolicValue::Kind::Struct) {
      SymbolicValue res;
      res.kind = SymbolicValue::Kind::Struct;
      for (const auto &[fld, _]: elements[0].structVal) {
        std::vector<SymbolicValue> inner_elements;
        for (size_t i = 0; i < elements.size(); ++i)
          inner_elements.push_back(elements[i].structVal.at(fld));
        res.structVal[fld] = mergeAggregate(inner_elements, idx, solver);
      }
      return res;
    }
    return {SymbolicValue::Kind::Undef};
  }

  SymbolicValue SymbolicExecutor::evalLValue(
      const LValue &lv, smt::ISolver &solver, SymbolicStore &store, std::vector<smt::Term> & /*pc*/,
      std::vector<smt::Term> &ub, bool forWrite
  ) {
    SymbolicValue res = store.at(lv.base.name);
    for (const auto &acc: lv.accesses) {
      if (auto ai = std::get_if<AccessIndex>(&acc)) {
        if (res.kind != SymbolicValue::Kind::Array && res.kind != SymbolicValue::Kind::Vec)
          throw std::runtime_error("Indexing non-array");
        size_t array_size = res.arrayVal.size();
        smt::Term idx;
        if (auto lit = std::get_if<IntLit>(&ai->index)) {
          idx = solver.make_bv_value(solver.make_bv_sort(32), std::to_string(lit->value), 10);
          // Out-of-range literal index is UB at compile time —
          // emit the bounds constraint (it'll be false → UNSAT) without
          // crashing in arrayVal.at().
          if (lit->value < 0 || static_cast<uint64_t>(lit->value) >= array_size) {
            ub.push_back(solver.make_false());
            // Use the last element as a dummy value (we won't be used).
            if (!res.arrayVal.empty())
              res = res.arrayVal.back();
            else
              res = SymbolicValue{SymbolicValue::Kind::Undef};
          } else {
            SymbolicValue next = res.arrayVal.at(lit->value);
            res = std::move(next);
          }
        } else {
          auto id = std::get<LocalOrSymId>(ai->index);
          idx = std::visit([&](auto &&v) { return store.at(v.name).term; }, id);
          auto idxSort = solver.get_sort(idx);
          if (solver.get_bv_width(idxSort) != 32) {
            idx = solver.make_term(
                smt::Kind::BV_SIGN_EXTEND, {idx}, {32 - solver.get_bv_width(idxSort)}
            );
          }
          res = mergeAggregate(res.arrayVal, idx, solver);
        }
        // Strict UB: bounds check
        auto size_term =
            solver.make_bv_value(solver.make_bv_sort(32), std::to_string(array_size), 10);
        auto zero = solver.make_bv_zero(solver.make_bv_sort(32));
        ub.push_back(solver.make_term(smt::Kind::BV_SLE, {zero, idx}));
        ub.push_back(solver.make_term(smt::Kind::BV_SLT, {idx, size_term}));
      } else if (auto af = std::get_if<AccessField>(&acc)) {
        if (res.kind != SymbolicValue::Kind::Struct)
          throw std::runtime_error("Field access on non-struct");
        SymbolicValue next = res.structVal.at(af->field);
        res = std::move(next);
      }
    }
    // Strict UB rule 3: reading an `undef` scalar is UB. Add
    // is_defined as a path constraint. Suppressed on the LHS-eval of
    // an AssignInstr (the caller is about to overwrite the value).
    if (!forWrite && res.kind == SymbolicValue::Kind::Int && res.is_defined.internal)
      ub.push_back(res.is_defined);
    return res;
  }

  SymbolicValue SymbolicExecutor::muxSymbolicValue(
      smt::Term cond, const SymbolicValue &t, const SymbolicValue &f, smt::ISolver &solver
  ) {
    if (t.kind != f.kind) {
      throw std::runtime_error("Muxing different kinds of SymbolicValues");
    }

    SymbolicValue res;
    res.kind = t.kind;

    if (t.kind == SymbolicValue::Kind::Int) {
      auto tSort = solver.get_sort(t.term);
      auto fSort = solver.get_sort(f.term);
      smt::Term tTerm = t.term;
      smt::Term fTerm = f.term;
      if (solver.is_bv_sort(tSort) && solver.is_bv_sort(fSort)) {
        auto tWidth = solver.get_bv_width(tSort);
        auto fWidth = solver.get_bv_width(fSort);
        if (tWidth < fWidth) {
          tTerm = solver.make_term(smt::Kind::BV_SIGN_EXTEND, {tTerm}, {fWidth - tWidth});
        } else if (fWidth < tWidth) {
          fTerm = solver.make_term(smt::Kind::BV_SIGN_EXTEND, {fTerm}, {tWidth - fWidth});
        }
      }
      res.term = solver.make_term(smt::Kind::ITE, {cond, tTerm, fTerm});
      res.is_defined = solver.make_term(smt::Kind::ITE, {cond, t.is_defined, f.is_defined});
      auto bv64 = solver.make_bv_sort(64);
      auto zero = solver.make_bv_value_int64(bv64, 0);
      auto tBase = t.prov_base.internal ? t.prov_base : zero;
      auto fBase = f.prov_base.internal ? f.prov_base : zero;
      auto tSize = t.prov_size.internal ? t.prov_size : zero;
      auto fSize = f.prov_size.internal ? f.prov_size : zero;
      res.prov_base = solver.make_term(smt::Kind::ITE, {cond, tBase, fBase});
      res.prov_size = solver.make_term(smt::Kind::ITE, {cond, tSize, fSize});
    } else if (t.kind == SymbolicValue::Kind::Array || t.kind == SymbolicValue::Kind::Vec ||
               t.kind == SymbolicValue::Kind::Tensor) {
      if (t.arrayVal.size() != f.arrayVal.size())
        throw std::runtime_error("Muxing arrays/vectors of different sizes");
      if (t.kind == SymbolicValue::Kind::Tensor && t.tensorShape != f.tensorShape)
        throw std::runtime_error("Muxing tensors with different shapes");
      res.tensorShape = t.tensorShape;
      for (size_t i = 0; i < t.arrayVal.size(); ++i) {
        res.arrayVal.push_back(muxSymbolicValue(cond, t.arrayVal[i], f.arrayVal[i], solver));
      }
    } else if (t.kind == SymbolicValue::Kind::Struct) {
      for (const auto &[key, val]: t.structVal) {
        if (f.structVal.find(key) == f.structVal.end())
          throw std::runtime_error("Muxing structs with mismatching keys: " + key);
        res.structVal[key] = muxSymbolicValue(cond, val, f.structVal.at(key), solver);
      }
    }
    return res;
  }

  SymbolicValue SymbolicExecutor::updateLValueRec(
      const SymbolicValue &cur, std::span<const Access> accesses, const SymbolicValue &val,
      smt::Term pathCond, smt::ISolver &solver, SymbolicStore &store, std::vector<smt::Term> &pc,
      std::vector<smt::Term> &ub, int depth
  ) {
    if (depth > 100)
      throw std::runtime_error("Recursion depth exceeded in updateLValueRec");

    if (accesses.empty()) {
      return muxSymbolicValue(pathCond, val, cur, solver);
    }

    const Access &acc = accesses[0];
    auto nextAccesses = accesses.subspan(1);
    SymbolicValue newCur = cur; // Copy

    if (auto ai = std::get_if<AccessIndex>(&acc)) {
      if (cur.kind != SymbolicValue::Kind::Array && cur.kind != SymbolicValue::Kind::Vec)
        throw std::runtime_error("Indexing non-array in setLValue");

      smt::Term idx;
      if (auto lit = std::get_if<IntLit>(&ai->index)) {
        idx = solver.make_bv_value(solver.make_bv_sort(32), std::to_string(lit->value), 10);
      } else {
        auto id = std::get<LocalOrSymId>(ai->index);
        idx = std::visit([&](auto &&v) { return store.at(v.name).term; }, id);
        if (solver.get_bv_width(solver.get_sort(idx)) != 32) {
          idx = solver.make_term(
              smt::Kind::BV_SIGN_EXTEND, {idx}, {32 - solver.get_bv_width(solver.get_sort(idx))}
          );
        }
      }

      // Bounds check UB
      size_t size = cur.arrayVal.size();
      if (size == 0)
        throw std::runtime_error("Indexing empty array");

      auto size_term = solver.make_bv_value(solver.make_bv_sort(32), std::to_string(size), 10);
      auto zero = solver.make_bv_zero(solver.make_bv_sort(32));

      ub.push_back(solver.make_term(
          smt::Kind::IMPLIES, {pathCond, solver.make_term(smt::Kind::BV_SLE, {zero, idx})}
      ));
      ub.push_back(solver.make_term(
          smt::Kind::IMPLIES, {pathCond, solver.make_term(smt::Kind::BV_SLT, {idx, size_term})}
      ));

      if (auto lit = std::get_if<IntLit>(&ai->index)) {
        // Constant index
        uint64_t k = lit->value;
        if (k < newCur.arrayVal.size()) {
          newCur.arrayVal[k] = updateLValueRec(
              cur.arrayVal[k], nextAccesses, val, pathCond, solver, store, pc, ub, depth + 1
          );
        }
      } else {
        // Symbolic index
        auto idxSort = solver.get_sort(idx);
        for (size_t k = 0; k < newCur.arrayVal.size(); ++k) {
          auto k_term = solver.make_bv_value(idxSort, std::to_string(k), 10);
          auto match = solver.make_term(smt::Kind::EQUAL, {idx, k_term});
          auto cond = solver.make_term(smt::Kind::AND, {pathCond, match});
          newCur.arrayVal[k] = updateLValueRec(
              cur.arrayVal[k], nextAccesses, val, cond, solver, store, pc, ub, depth + 1
          );
        }
      }
    } else {
      auto &af = std::get<AccessField>(acc);
      if (cur.kind != SymbolicValue::Kind::Struct)
        throw std::runtime_error("Field access on non-struct in setLValue");
      if (cur.structVal.find(af.field) == cur.structVal.end())
        throw std::runtime_error("Field not found: " + af.field);

      newCur.structVal[af.field] = updateLValueRec(
          cur.structVal.at(af.field), nextAccesses, val, pathCond, solver, store, pc, ub, depth + 1
      );
    }
    return newCur;
  }

  void SymbolicExecutor::setLValue(
      const LValue &lv, const SymbolicValue &val, smt::ISolver &solver, SymbolicStore &store,
      std::vector<smt::Term> &pc, std::vector<smt::Term> &ub
  ) {
    SymbolicValue &root = store.at(lv.base.name);
    // Use true condition because the instruction itself is unconditional *at this point in the
    // trace* The path constraints handle the reachability of the instruction.
    root = updateLValueRec(root, lv.accesses, val, solver.make_true(), solver, store, pc, ub, 0);
  }
} // namespace refractir
