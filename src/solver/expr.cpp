#include <optional>
#include <stdexcept>
#include "analysis/type_utils.hpp"
#include "internal.hpp"
#include "solver/solver.hpp"

namespace refractir {

  SymbolicValue SymbolicExecutor::evalExpr(
      const Expr &e, smt::ISolver &solver, SymbolicStore &store, std::vector<smt::Term> &pc,
      std::vector<smt::Term> &ub, std::optional<smt::Sort> expectedSort
  ) {
    SymbolicValue res = evalAtom(e.first, solver, store, pc, ub, expectedSort);
    // The typechecker requires the +/- chain to be homogeneous in type. If the
    // caller didn't supply an expected sort, propagate the first atom's sort
    // to subsequent atoms.
    std::optional<smt::Sort> chainSort = expectedSort;
    if (!chainSort && res.term.internal)
      chainSort = solver.get_sort(res.term);

    // Detect pointer arithmetic or pointer subtraction dynamically.
    TypePtr firstTy = resolveAtomType(e.first);
    bool isPtrExpr = firstTy && std::holds_alternative<PtrType>(firstTy->v);
    uint64_t ptrStep = 0;
    if (isPtrExpr) {
      auto pointeeTy = std::get<PtrType>(firstTy->v).pointee;
      ptrStep = sizeofTagUnits(pointeeTy, structs_);
    }

    for (const auto &tail: e.rest) {
      TypePtr rightTy = resolveAtomType(tail.atom);
      bool rhsIsPtr = rightTy && std::holds_alternative<PtrType>(rightTy->v);

      SymbolicValue right = evalAtom(tail.atom, solver, store, pc, ub, chainSort);

      auto lSort = solver.get_sort(res.term);
      auto rSort = solver.get_sort(right.term);

      if (solver.is_fp_sort(lSort)) {
        auto rmRNE = solver.make_rm_value(smt::RoundingMode::RNE);
        if (tail.op == AddOp::Plus) {
          res.term = solver.make_term(smt::Kind::FP_ADD, {rmRNE, res.term, right.term});
        } else {
          res.term = solver.make_term(smt::Kind::FP_SUB, {rmRNE, res.term, right.term});
        }
        assertFPFinite(res.term, solver, ub);
      } else if (solver.is_bv_sort(lSort) && solver.is_bv_sort(rSort)) {
        auto lWidth = solver.get_bv_width(lSort);
        auto rWidth = solver.get_bv_width(rSort);
        if (lWidth < rWidth) {
          res.term = solver.make_term(smt::Kind::BV_SIGN_EXTEND, {res.term}, {rWidth - lWidth});
        } else if (rWidth < lWidth) {
          right.term = solver.make_term(smt::Kind::BV_SIGN_EXTEND, {right.term}, {lWidth - rWidth});
        }

        bool isPtrIntArith = isPtrExpr && !rhsIsPtr;
        bool isPtrSub = isPtrExpr && rhsIsPtr && tail.op == AddOp::Minus;

        if (isPtrIntArith && ptrStep != 1) {
          auto stepTerm = solver.make_bv_value_int64(
              solver.get_sort(right.term), static_cast<int64_t>(ptrStep)
          );
          right.term = solver.make_term(smt::Kind::BV_MUL, {right.term, stepTerm});
        }

        if (tail.op == AddOp::Plus) {
          auto overflow = solver.make_term(smt::Kind::BV_SADD_OVERFLOW, {res.term, right.term});
          ub.push_back(solver.make_term(smt::Kind::NOT, {overflow}));
          res.term = solver.make_term(smt::Kind::BV_ADD, {res.term, right.term});
        } else {
          auto overflow = solver.make_term(smt::Kind::BV_SSUB_OVERFLOW, {res.term, right.term});
          ub.push_back(solver.make_term(smt::Kind::NOT, {overflow}));
          if (isPtrSub) {
            // Pointer subtraction:
            // 1. Rule 12 dynamic assertion (matching bases, non-zero)
            auto bv64 = solver.make_bv_sort(64);
            auto zero = solver.make_bv_value_int64(bv64, 0);
            if (res.prov_base.internal && right.prov_base.internal) {
              auto eqBase = solver.make_term(smt::Kind::EQUAL, {res.prov_base, right.prov_base});
              auto nonZeroBase = solver.make_term(smt::Kind::DISTINCT, {res.prov_base, zero});
              ub.push_back(solver.make_term(smt::Kind::AND, {eqBase, nonZeroBase}));
            } else {
              // One or both have no provenance, which is UB for ptr subtraction
              ub.push_back(solver.make_false());
            }

            // 2. Subtract addresses
            res.term = solver.make_term(smt::Kind::BV_SUB, {res.term, right.term});
            // 3. Divide by element size to get distance
            if (ptrStep > 1) {
              auto stepTerm = solver.make_bv_value_int64(
                  solver.get_sort(res.term), static_cast<int64_t>(ptrStep)
              );
              res.term = solver.make_term(smt::Kind::BV_SDIV, {res.term, stepTerm});
            }
            // 4. Result is an integer, so clear provenance
            res.prov_base = {};
            res.prov_size = {};
          } else {
            res.term = solver.make_term(smt::Kind::BV_SUB, {res.term, right.term});
          }
        }

        // Rule 10 (ptr arith OOB): for ptr ± int, result must stay in [base, base + size].
        if (isPtrIntArith) {
          if (res.prov_base.internal && res.prov_size.internal) {
            auto end = solver.make_term(smt::Kind::BV_ADD, {res.prov_base, res.prov_size});
            ub.push_back(solver.make_term(smt::Kind::BV_ULE, {res.prov_base, res.term}));
            ub.push_back(solver.make_term(smt::Kind::BV_ULE, {res.term, end}));
          }
        }
      }
    }
    return res;
  }

  SymbolicValue SymbolicExecutor::evalAtom(
      const Atom &a, smt::ISolver &solver, SymbolicStore &store, std::vector<smt::Term> &pc,
      std::vector<smt::Term> &ub, std::optional<smt::Sort> expectedSort
  ) {
    return std::visit(
        [&](auto &&arg) -> SymbolicValue {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, OpAtom>) {
            return evalOpAtom(arg, solver, store, pc, ub, expectedSort);
          } else if constexpr (std::is_same_v<T, UnaryAtom>) {
            return evalUnaryAtom(arg, solver, store, pc, ub);
          } else if constexpr (std::is_same_v<T, SelectAtom>) {
            return evalSelectAtom(arg, solver, store, pc, ub, expectedSort);
          } else if constexpr (std::is_same_v<T, CoefAtom>) {

            if (auto id = std::get_if<LocalOrSymId>(&arg.coef)) {
              const auto &value = store.at(std::visit([](auto &&part) { return part.name; }, *id));
              if (value.kind == SymbolicValue::Kind::Tensor) {
                for (const auto &leaf: value.arrayVal) {
                  if (leaf.kind == SymbolicValue::Kind::Undef)
                    ub.push_back(solver.make_false());
                  else if (leaf.is_defined.internal)
                    ub.push_back(leaf.is_defined);
                }
                return value;
              }
            }

            smt::Term term = evalCoef(arg.coef, solver, store, expectedSort);
            auto bv64 = solver.make_bv_sort(64);
            auto zero = solver.make_bv_value_int64(bv64, 0);
            return SymbolicValue(SymbolicValue::Kind::Int, term, solver.make_true(), zero, zero);

          } else if constexpr (std::is_same_v<T, RValueAtom>) {

            return evalLValue(arg.rval, solver, store, pc, ub);

          } else if constexpr (std::is_same_v<T, CastAtom>) {
            return evalCastAtom(arg, solver, store, pc, ub);
          } else if constexpr (std::is_same_v<T, CmpAtom>) {
            return evalCmpAtom(arg, solver, store, pc, ub);
          } else if constexpr (std::is_same_v<T, AddrAtom>) {
            return evalAddrAtom(arg, solver, store);
          } else if constexpr (std::is_same_v<T, PtrIndexAtom>) {
            return evalPtrIndexAtom(arg, solver, store, pc, ub);
          } else if constexpr (std::is_same_v<T, PtrFieldAtom>) {
            return evalPtrFieldAtom(arg, solver, store, pc, ub);
          } else if constexpr (std::is_same_v<T, LoadAtom>) {
            return evalLoadAtom(arg, solver, store, pc, ub);
          } else if constexpr (std::is_same_v<T, CallAtom>) {
            return evalCallAtom(arg, solver, store, pc, ub);
          } else if constexpr (std::is_same_v<T, TensorAtom>) {
            return evalTensorAtom(arg, solver, store, ub);
          }
          return SymbolicValue(SymbolicValue::Kind::Undef);
        },
        a.v
    );
  }

  SymbolicValue SymbolicExecutor::evalOpAtom(
      const OpAtom &arg, smt::ISolver &solver, SymbolicStore &store, std::vector<smt::Term> &pc,
      std::vector<smt::Term> &ub, std::optional<smt::Sort> expectedSort
  ) {

    smt::Term r = evalLValue(arg.rval, solver, store, pc, ub).term;
    auto rSort = solver.get_sort(r);
    smt::Term c = evalCoef(arg.coef, solver, store, expectedSort.value_or(rSort));
    auto cSort = solver.get_sort(c);

    if (solver.is_fp_sort(cSort)) {
      // SPEC §2.9: all FP ops use RNE. The fmod encoding additionally
      // uses RTZ for the quotient-to-integer step.
      auto rmRNE = solver.make_rm_value(smt::RoundingMode::RNE);
      smt::Term fpRes;
      if (arg.op == AtomOpKind::Mul)
        fpRes = solver.make_term(smt::Kind::FP_MUL, {rmRNE, c, r});
      else if (arg.op == AtomOpKind::Div)
        fpRes = solver.make_term(smt::Kind::FP_DIV, {rmRNE, c, r});
      else if (arg.op == AtomOpKind::Mod) {
        // SMT fp.rem uses the nearest integral quotient, while C fmod uses
        // truncation. They differ when the nearest quotient crosses zero:
        // the IEEE remainder then has the opposite sign from x. Add signed
        // |y| in that case. Excluding a zero remainder preserves -0 directly
        // and avoids treating -0 as evidence of quotient rounding.
        auto zero = solver.make_fp_value_from_real(cSort, 0.0, smt::RoundingMode::RNE);
        auto ieeeRemainder = solver.make_term(smt::Kind::FP_REM, {c, r});
        auto negativeRemainder = solver.make_term(smt::Kind::FP_LT, {ieeeRemainder, zero});
        auto negativeDividend = solver.make_term(smt::Kind::FP_LT, {c, zero});
        auto signsDiffer =
            solver.make_term(smt::Kind::DISTINCT, {negativeRemainder, negativeDividend});
        auto remainderIsZero = solver.make_term(smt::Kind::FP_EQUAL, {ieeeRemainder, zero});
        auto correct = solver.make_term(
            smt::Kind::AND, {signsDiffer, solver.make_term(smt::Kind::NOT, {remainderIsZero})}
        );
        auto negativeDivisor = solver.make_term(smt::Kind::FP_LT, {r, zero});
        auto absDivisor = solver.make_term(
            smt::Kind::ITE,
            {negativeDivisor, solver.make_term(smt::Kind::FP_SUB, {rmRNE, zero, r}), r}
        );
        auto signedDivisor = solver.make_term(
            smt::Kind::ITE,
            {negativeDividend, solver.make_term(smt::Kind::FP_SUB, {rmRNE, zero, absDivisor}),
             absDivisor}
        );
        auto corrected = solver.make_term(smt::Kind::FP_ADD, {rmRNE, ieeeRemainder, signedDivisor});
        fpRes = solver.make_term(smt::Kind::ITE, {correct, corrected, ieeeRemainder});
      } else
        return {};
      assertFPFinite(fpRes, solver, ub);
      return SymbolicValue(SymbolicValue::Kind::Int, fpRes, solver.make_true());
    }

    if (solver.is_bv_sort(cSort) && solver.is_bv_sort(rSort)) {
      auto cWidth = solver.get_bv_width(cSort);
      auto rWidth = solver.get_bv_width(rSort);
      if (cWidth < rWidth) {
        c = solver.make_term(smt::Kind::BV_SIGN_EXTEND, {c}, {rWidth - cWidth});
        cSort = solver.get_sort(c);
      } else if (rWidth < cWidth) {
        r = solver.make_term(smt::Kind::BV_SIGN_EXTEND, {r}, {cWidth - rWidth});
        rSort = solver.get_sort(r);
      }
    }

    if (arg.op == AtomOpKind::Div || arg.op == AtomOpKind::Mod) {
      auto zero = solver.make_bv_zero(solver.get_sort(r));
      ub.push_back(solver.make_term(smt::Kind::DISTINCT, {r, zero}));
    }
    if (arg.op == AtomOpKind::Mul) {
      auto overflow = solver.make_term(smt::Kind::BV_SMUL_OVERFLOW, {c, r});
      ub.push_back(solver.make_term(smt::Kind::NOT, {overflow}));
      return SymbolicValue(
          SymbolicValue::Kind::Int, solver.make_term(smt::Kind::BV_MUL, {c, r}), solver.make_true()
      );
    }
    if (arg.op == AtomOpKind::Div) {
      // Check overflow: c == INT_MIN && r == -1
      auto min_signed = solver.make_bv_min_signed(solver.get_sort(c));
      auto minus_one = solver.make_bv_value_int64(solver.get_sort(r), -1);
      auto is_min = solver.make_term(smt::Kind::EQUAL, {c, min_signed});
      auto is_minus_one = solver.make_term(smt::Kind::EQUAL, {r, minus_one});
      auto div_overflow = solver.make_term(smt::Kind::AND, {is_min, is_minus_one});
      ub.push_back(solver.make_term(smt::Kind::NOT, {div_overflow}));
      return SymbolicValue(
          SymbolicValue::Kind::Int, solver.make_term(smt::Kind::BV_SDIV, {c, r}), solver.make_true()
      );
    }
    if (arg.op == AtomOpKind::Mod) {
      // Check overflow for mod
      auto min_signed = solver.make_bv_min_signed(solver.get_sort(c));
      auto minus_one = solver.make_bv_value_int64(solver.get_sort(r), -1);
      auto is_min = solver.make_term(smt::Kind::EQUAL, {c, min_signed});
      auto is_minus_one = solver.make_term(smt::Kind::EQUAL, {r, minus_one});
      auto mod_overflow = solver.make_term(smt::Kind::AND, {is_min, is_minus_one});
      ub.push_back(solver.make_term(smt::Kind::NOT, {mod_overflow}));
      return SymbolicValue(
          SymbolicValue::Kind::Int, solver.make_term(smt::Kind::BV_SREM, {c, r}), solver.make_true()
      );
    }
    if (arg.op == AtomOpKind::And) {
      return SymbolicValue(
          SymbolicValue::Kind::Int, solver.make_term(smt::Kind::BV_AND, {c, r}), solver.make_true()
      );
    }
    if (arg.op == AtomOpKind::Or) {
      return SymbolicValue(
          SymbolicValue::Kind::Int, solver.make_term(smt::Kind::BV_OR, {c, r}), solver.make_true()
      );
    }
    if (arg.op == AtomOpKind::Xor) {
      return SymbolicValue(
          SymbolicValue::Kind::Int, solver.make_term(smt::Kind::BV_XOR, {c, r}), solver.make_true()
      );
    }
    if (arg.op == AtomOpKind::Shl || arg.op == AtomOpKind::Shr || arg.op == AtomOpKind::LShr) {
      // Overshift UB: amount in [0, width). Negative amounts and
      // amounts >= width are both UB (rule 5). We assert `r` is in
      // range by reading it as signed (>= 0) AND unsigned (< width)
      // — the conjunction covers both cases regardless of sort.
      uint32_t width = solver.get_bv_width(solver.get_sort(c));
      auto width_term = solver.make_bv_value(solver.get_sort(r), std::to_string(width), 10);
      auto zero = solver.make_bv_zero(solver.get_sort(r));
      ub.push_back(solver.make_term(smt::Kind::BV_SLE, {zero, r}));
      ub.push_back(solver.make_term(smt::Kind::BV_ULT, {r, width_term}));

      if (arg.op == AtomOpKind::Shl) {
        // SPEC §7.1 rule 4: SHL on a negative operand is UB, and
        // result overflow is UB. The result-overflow path was
        // already checked via `(ashr (shl c, r), r) == c`; we now
        // also require `c >= 0` to catch `-1 << 1` which the
        // overflow check otherwise accepts (ashr-shl round-trips
        // on `-1` since the sign bit dominates).
        ub.push_back(solver.make_term(smt::Kind::BV_SLE, {zero, c}));
        auto shifted = solver.make_term(smt::Kind::BV_SHL, {c, r});
        auto unshifted = solver.make_term(smt::Kind::BV_ASHR, {shifted, r});
        ub.push_back(solver.make_term(smt::Kind::EQUAL, {unshifted, c}));
        return SymbolicValue(SymbolicValue::Kind::Int, shifted, solver.make_true());
      }
      if (arg.op == AtomOpKind::Shr)
        return SymbolicValue(
            SymbolicValue::Kind::Int, solver.make_term(smt::Kind::BV_ASHR, {c, r}),
            solver.make_true()
        );
      if (arg.op == AtomOpKind::LShr)
        return SymbolicValue(
            SymbolicValue::Kind::Int, solver.make_term(smt::Kind::BV_SHR, {c, r}),
            solver.make_true()
        );
    }
    return {};

    return SymbolicValue(SymbolicValue::Kind::Undef);
  }

  SymbolicValue SymbolicExecutor::evalUnaryAtom(
      const UnaryAtom &arg, smt::ISolver &solver, SymbolicStore &store, std::vector<smt::Term> &pc,
      std::vector<smt::Term> &ub
  ) {

    auto r = evalLValue(arg.rval, solver, store, pc, ub).term;
    if (arg.op == UnaryOpKind::Not) {
      return SymbolicValue(
          SymbolicValue::Kind::Int, solver.make_term(smt::Kind::BV_NOT, {r}), solver.make_true()
      );
    }
    return {};

    return SymbolicValue(SymbolicValue::Kind::Undef);
  }

  SymbolicValue SymbolicExecutor::evalSelectAtom(
      const SelectAtom &arg, smt::ISolver &solver, SymbolicStore &store, std::vector<smt::Term> &pc,
      std::vector<smt::Term> &ub, std::optional<smt::Sort> expectedSort
  ) {

    // Two forms: cond form (relational predicate) or mask form
    // (scalar i1 expression here; vector masks go through the
    // vector path).
    smt::Term cond;
    if (arg.cond) {
      cond = evalCond(*arg.cond, solver, store, pc, ub);
    } else if (arg.maskExpr) {
      SymbolicValue m = evalExpr(*arg.maskExpr, solver, store, pc, ub);
      auto mSort = solver.get_sort(m.term);
      if (solver.is_fp_sort(mSort))
        throw std::runtime_error("scalar select: mask must be integral");
      auto zero = solver.make_bv_zero(mSort);
      cond = solver.make_term(smt::Kind::DISTINCT, {m.term, zero});
    } else {
      throw std::runtime_error("SelectAtom: neither cond nor maskExpr set");
    }
    // §6.4 select is lazy: only the chosen arm's UB
    // constraints participate in the path condition. Evaluate
    // each arm into a private constraint list, then gate them
    // with the appropriate side of `cond` before pushing to pc.
    std::vector<smt::Term> tPc, tUb, fPc, fUb;
    SymbolicValue vt = evalSelectVal(arg.vtrue, solver, store, tPc, tUb, expectedSort);
    SymbolicValue vf = evalSelectVal(arg.vfalse, solver, store, fPc, fUb, expectedSort);
    auto notCond = solver.make_term(smt::Kind::NOT, {cond});
    for (auto &t: tPc)
      pc.push_back(solver.make_term(smt::Kind::IMPLIES, {cond, t}));
    for (auto &u: tUb)
      ub.push_back(solver.make_term(smt::Kind::IMPLIES, {cond, u}));
    for (auto &t: fPc)
      pc.push_back(solver.make_term(smt::Kind::IMPLIES, {notCond, t}));
    for (auto &u: fUb)
      ub.push_back(solver.make_term(smt::Kind::IMPLIES, {notCond, u}));

    return muxSymbolicValue(cond, vt, vf, solver);

    return SymbolicValue(SymbolicValue::Kind::Undef);
  }

  SymbolicValue SymbolicExecutor::evalCastAtom(
      const CastAtom &arg, smt::ISolver &solver, SymbolicStore &store, std::vector<smt::Term> &pc,
      std::vector<smt::Term> &ub
  ) {

    smt::Term src = std::visit(
        [&](auto &&s) -> smt::Term {
          using S = std::decay_t<decltype(s)>;
          if constexpr (std::is_same_v<S, IntLit>) {
            return solver.make_bv_value(solver.make_bv_sort(32), std::to_string(s.value), 10);
          } else if constexpr (std::is_same_v<S, FloatLit>) {
            // Default to f32 if implied. Built from the double, not a decimal
            // rendering of it (see make_fp_value_from_real).
            return solver.make_fp_value_from_real(
                solver.make_fp_sort(8, 24), s.value, smt::RoundingMode::RNE
            );
          } else if constexpr (std::is_same_v<S, SymId>) {
            return store.at(s.name).term;
          } else {
            return evalLValue(s, solver, store, pc, ub).term;
          }
        },
        arg.src
    );
    auto dstSort = getSort(arg.dstType, solver);

    bool srcIsFp = solver.is_fp_sort(solver.get_sort(src));
    bool dstIsFp = solver.is_fp_sort(dstSort);

    smt::Term casted;
    if (srcIsFp && !dstIsFp) { // FP -> BV (spec §7.4 rule 8: range check)
      uint32_t width = solver.get_bv_width(dstSort);
      auto srcSort = solver.get_sort(src);
      assertFPFinite(src, solver, ub);
      double lo = -std::ldexp(1.0, static_cast<int>(width) - 1);
      double hi = std::ldexp(1.0, static_cast<int>(width) - 1);
      auto loFp = solver.make_fp_value_from_real(srcSort, lo, smt::RoundingMode::RNE);
      auto hiFp = solver.make_fp_value_from_real(srcSort, hi, smt::RoundingMode::RNE);
      ub.push_back(solver.make_term(smt::Kind::FP_GEQ, {src, loFp}));
      ub.push_back(solver.make_term(smt::Kind::FP_LT, {src, hiFp}));
      auto rmRTZ = solver.make_rm_value(smt::RoundingMode::RTZ);
      casted = solver.make_term(smt::Kind::FP_TO_SBV, {rmRTZ, src}, {width});
    } else if (!srcIsFp && dstIsFp) { // BV -> FP
      auto [exp, sig] = solver.get_fp_dims(dstSort);
      auto rmRNE = solver.make_rm_value(smt::RoundingMode::RNE);
      casted = solver.make_term(smt::Kind::FP_TO_FP_FROM_SBV, {rmRNE, src}, {exp, sig});
      assertFPFinite(casted, solver, ub);
    } else if (srcIsFp && dstIsFp) { // FP -> FP
      auto [exp, sig] = solver.get_fp_dims(dstSort);
      auto rmRNE = solver.make_rm_value(smt::RoundingMode::RNE);
      casted = solver.make_term(smt::Kind::FP_TO_FP_FROM_FP, {rmRNE, src}, {exp, sig});
      assertFPFinite(casted, solver, ub);
    } else {
      // BV -> BV resizing
      uint32_t srcWidth = solver.get_bv_width(solver.get_sort(src));
      uint32_t dstWidth = solver.get_bv_width(dstSort);
      if (srcWidth == dstWidth)
        casted = src;
      else if (srcWidth < dstWidth) {
        casted = solver.make_term(smt::Kind::BV_SIGN_EXTEND, {src}, {dstWidth - srcWidth});
      } else {
        casted = solver.make_term(smt::Kind::BV_EXTRACT, {src}, {dstWidth - 1, 0});
      }
    }
    return SymbolicValue(SymbolicValue::Kind::Int, casted, solver.make_true());

    return SymbolicValue(SymbolicValue::Kind::Undef);
  }

  SymbolicValue SymbolicExecutor::evalCmpAtom(
      const CmpAtom &arg, smt::ISolver &solver, SymbolicStore &store, std::vector<smt::Term> &pc,
      std::vector<smt::Term> &ub
  ) {

    auto getOperandSort = [&]() -> std::optional<smt::Sort> {
      if (auto rv = std::get_if<RValue>(&arg.lhs))
        return solver.get_sort(evalLValue(*rv, solver, store, pc, ub).term);
      if (auto rv = std::get_if<RValue>(&arg.rhs))
        return solver.get_sort(evalLValue(*rv, solver, store, pc, ub).term);
      return std::nullopt;
    };
    auto opSort = getOperandSort();
    auto loadOperand = [&](const SelectVal &sv) -> SymbolicValue {
      return evalSelectVal(sv, solver, store, pc, ub, opSort);
    };

    SymbolicValue lVal = loadOperand(arg.lhs);
    SymbolicValue rVal = loadOperand(arg.rhs);

    TypePtr lhsTy = resolveSelectValType(arg.lhs);
    TypePtr rhsTy = resolveSelectValType(arg.rhs);
    bool isLhsPtr = lhsTy && std::holds_alternative<PtrType>(lhsTy->v);
    bool isRhsPtr = rhsTy && std::holds_alternative<PtrType>(rhsTy->v);
    if (isLhsPtr || isRhsPtr) {
      if (arg.op == RelOp::LT || arg.op == RelOp::LE || arg.op == RelOp::GT ||
          arg.op == RelOp::GE) {
        auto bv64 = solver.make_bv_sort(64);
        auto zero = solver.make_bv_value_int64(bv64, 0);
        if (lVal.prov_base.internal && rVal.prov_base.internal) {
          auto eqBase = solver.make_term(smt::Kind::EQUAL, {lVal.prov_base, rVal.prov_base});
          auto nonZeroBase = solver.make_term(smt::Kind::DISTINCT, {lVal.prov_base, zero});
          ub.push_back(solver.make_term(smt::Kind::AND, {eqBase, nonZeroBase}));
        } else {
          ub.push_back(solver.make_false());
        }
      }
    }

    smt::Term l = lVal.term;
    smt::Term r = rVal.term;

    auto lSort = solver.get_sort(l);
    auto rSort = solver.get_sort(r);
    if (solver.is_fp_sort(lSort) != solver.is_fp_sort(rSort))
      throw std::runtime_error("scalar cmp: mixed FP/BV operands");
    if (!solver.is_fp_sort(lSort)) {
      uint32_t lw = solver.get_bv_width(lSort);
      uint32_t rw = solver.get_bv_width(rSort);
      if (lw != rw) {
        uint32_t target = std::max(lw, rw);
        if (lw < target)
          l = solver.make_term(smt::Kind::BV_SIGN_EXTEND, {l}, {target - lw});
        if (rw < target)
          r = solver.make_term(smt::Kind::BV_SIGN_EXTEND, {r}, {target - rw});
      }
    }
    bool isFp = solver.is_fp_sort(solver.get_sort(l));
    smt::Kind k = relOpToSmtKind(arg.op, isFp);
    smt::Term cond = solver.make_term(k, {l, r});
    smt::Sort i1 = solver.make_bv_sort(1);
    smt::Term one = solver.make_bv_value(i1, "1", 10);
    smt::Term zero = solver.make_bv_value(i1, "0", 10);
    return SymbolicValue(
        SymbolicValue::Kind::Int, solver.make_term(smt::Kind::ITE, {cond, one, zero}),
        solver.make_true()
    );

    return SymbolicValue(SymbolicValue::Kind::Undef);
  }

  SymbolicValue
  SymbolicExecutor::evalAddrAtom(const AddrAtom &arg, smt::ISolver &solver, SymbolicStore &store) {

    const std::string targetName = arg.lv.base.name;
    auto bv64 = solver.make_bv_sort(kPtrBits);
    smt::Term tag = solver.make_bv_value_int64(
        bv64, static_cast<int64_t>(tagOfLocal(currentFrame_, targetName))
    );
    TypePtr cur;
    if (currentFun_) {
      for (const auto &l: currentFun_->lets)
        if (l.name.name == targetName) {
          cur = l.type;
          break;
        }
      if (!cur)
        for (const auto &p: currentFun_->params)
          if (p.name.name == targetName) {
            cur = p.type;
            break;
          }
    }
    smt::Term prov_base = tag;
    std::uint64_t initial_size = sizeofTagUnits(cur, structs_);
    smt::Term prov_size = solver.make_bv_value_int64(bv64, static_cast<int64_t>(initial_size));

    for (const auto &acc: arg.lv.accesses) {
      if (auto ai = std::get_if<AccessIndex>(&acc)) {
        auto at = std::get_if<ArrayType>(&cur->v);
        if (!at)
          throw std::runtime_error("addr: index on non-array");
        std::uint64_t stride = sizeofTagUnits(at->elem, structs_);
        prov_base = tag;
        prov_size = solver.make_bv_value_int64(bv64, static_cast<int64_t>(at->size * stride));

        if (auto il = std::get_if<IntLit>(&ai->index)) {
          std::uint64_t off = static_cast<std::uint64_t>(il->value) * stride;
          if (off != 0) {
            auto ofT = solver.make_bv_value_int64(bv64, static_cast<int64_t>(off));
            tag = solver.make_term(smt::Kind::BV_ADD, {tag, ofT});
          }
          cur = at->elem;
          continue;
        }
        if (auto id = std::get_if<LocalOrSymId>(&ai->index)) {
          smt::Term idxT = std::visit([&](auto &&v) { return store.at(v.name).term; }, *id);
          auto idxSort = solver.get_sort(idxT);
          uint32_t iw = solver.get_bv_width(idxSort);
          if (iw < kPtrBits)
            idxT = solver.make_term(smt::Kind::BV_SIGN_EXTEND, {idxT}, {kPtrBits - iw});
          if (stride != 1) {
            auto stT = solver.make_bv_value_int64(bv64, static_cast<int64_t>(stride));
            idxT = solver.make_term(smt::Kind::BV_MUL, {idxT, stT});
          }
          tag = solver.make_term(smt::Kind::BV_ADD, {tag, idxT});
          cur = at->elem;
          continue;
        }
      }
      if (auto af = std::get_if<AccessField>(&acc)) {
        auto st = std::get_if<StructType>(&cur->v);
        if (!st)
          throw std::runtime_error("'addr' field access on non-struct base: " + targetName);
        auto sIt = structs_.find(st->name.name);
        if (sIt == structs_.end())
          throw std::runtime_error("unknown struct in addr field access");
        prov_base = tag;
        std::uint64_t stSize = sizeofTagUnits(cur, structs_);
        prov_size = solver.make_bv_value_int64(bv64, static_cast<int64_t>(stSize));

        std::uint64_t off = fieldOffsetTagUnits(*sIt->second, af->field, structs_);
        TypePtr fieldTy;
        for (const auto &f: sIt->second->fields)
          if (f.name == af->field) {
            fieldTy = f.type;
            break;
          }
        if (!fieldTy)
          throw std::runtime_error("addr: unknown field " + af->field);
        if (off != 0) {
          auto ofT = solver.make_bv_value_int64(bv64, static_cast<int64_t>(off));
          tag = solver.make_term(smt::Kind::BV_ADD, {tag, ofT});
        }
        cur = fieldTy;
        continue;
      }
      throw std::runtime_error("'addr' with this access kind not yet supported in solver");
    }
    return SymbolicValue(SymbolicValue::Kind::Int, tag, solver.make_true(), prov_base, prov_size);

    return SymbolicValue(SymbolicValue::Kind::Undef);
  }

  SymbolicValue SymbolicExecutor::evalPtrIndexAtom(
      const PtrIndexAtom &arg, smt::ISolver &solver, SymbolicStore &store,
      std::vector<smt::Term> &pc, std::vector<smt::Term> &ub
  ) {

    auto bv64 = solver.make_bv_sort(kPtrBits);
    SymbolicValue ptrVal = evalLValue(arg.rval, solver, store, pc, ub);
    smt::Term ptrTerm = ptrVal.term;

    auto nullTerm = solver.make_bv_value_int64(bv64, 0);
    ub.push_back(solver.make_term(smt::Kind::DISTINCT, {ptrTerm, nullTerm}));

    if (ptrVal.prov_base.internal && ptrVal.prov_size.internal) {
      auto endAddr = solver.make_term(smt::Kind::BV_ADD, {ptrVal.prov_base, ptrVal.prov_size});
      ub.push_back(solver.make_term(smt::Kind::DISTINCT, {ptrTerm, endAddr}));
    }

    smt::Term idxT;
    if (auto il = std::get_if<IntLit>(&arg.index)) {
      idxT = solver.make_bv_value_int64(bv64, il->value);
    } else if (auto id = std::get_if<LocalOrSymId>(&arg.index)) {
      smt::Term raw = std::visit([&](auto &&v) { return store.at(v.name).term; }, *id);
      auto rawSort = solver.get_sort(raw);
      uint32_t rw = solver.get_bv_width(rawSort);
      if (rw < kPtrBits)
        raw = solver.make_term(smt::Kind::BV_SIGN_EXTEND, {raw}, {kPtrBits - rw});
      idxT = raw;
    }

    std::uint64_t elemUnits = 1;
    std::uint64_t arrSize = 0;
    if (currentFun_) {
      TypePtr baseType = resolveLValueType(arg.rval);
      if (baseType) {
        if (auto pt = std::get_if<PtrType>(&baseType->v)) {
          if (auto at = std::get_if<ArrayType>(&pt->pointee->v)) {
            elemUnits = sizeofTagUnits(at->elem, structs_);
            arrSize = at->size;
          }
        }
      }
    }

    if (arrSize > 0) {
      auto zero = solver.make_bv_value_int64(bv64, 0);
      auto N = solver.make_bv_value_int64(bv64, static_cast<int64_t>(arrSize));
      ub.push_back(solver.make_term(smt::Kind::BV_SLE, {zero, idxT}));
      ub.push_back(solver.make_term(smt::Kind::BV_SLE, {idxT, N}));
    }

    if (elemUnits != 1) {
      auto stride = solver.make_bv_value_int64(bv64, static_cast<int64_t>(elemUnits));
      idxT = solver.make_term(smt::Kind::BV_MUL, {idxT, stride});
    }
    smt::Term newAddr = solver.make_term(smt::Kind::BV_ADD, {ptrTerm, idxT});

    smt::Term prov_base = ptrTerm;
    smt::Term prov_size =
        solver.make_bv_value_int64(bv64, static_cast<int64_t>(arrSize * elemUnits));
    return SymbolicValue(
        SymbolicValue::Kind::Int, newAddr, solver.make_true(), prov_base, prov_size
    );

    return SymbolicValue(SymbolicValue::Kind::Undef);
  }

  SymbolicValue SymbolicExecutor::evalPtrFieldAtom(
      const PtrFieldAtom &arg, smt::ISolver &solver, SymbolicStore &store,
      std::vector<smt::Term> &pc, std::vector<smt::Term> &ub
  ) {

    if (!currentFun_)
      throw std::runtime_error("ptrfield encountered without active FunDecl");
    auto bv64 = solver.make_bv_sort(kPtrBits);
    SymbolicValue ptrVal = evalLValue(arg.rval, solver, store, pc, ub);
    smt::Term ptrTerm = ptrVal.term;

    auto nullTerm = solver.make_bv_value_int64(bv64, 0);
    ub.push_back(solver.make_term(smt::Kind::DISTINCT, {ptrTerm, nullTerm}));

    if (ptrVal.prov_base.internal && ptrVal.prov_size.internal) {
      auto endAddr = solver.make_term(smt::Kind::BV_ADD, {ptrVal.prov_base, ptrVal.prov_size});
      ub.push_back(solver.make_term(smt::Kind::DISTINCT, {ptrTerm, endAddr}));
    }

    TypePtr cur = resolveLValueType(arg.rval);
    if (!cur || !std::holds_alternative<PtrType>(cur->v))
      throw std::runtime_error("ptrfield: rval is not pointer-typed");
    auto &pt = std::get<PtrType>(cur->v);
    if (!std::holds_alternative<StructType>(pt.pointee->v))
      throw std::runtime_error("ptrfield: pointee is not a struct");
    auto &st = std::get<StructType>(pt.pointee->v);
    auto sIt = structs_.find(st.name.name);
    if (sIt == structs_.end())
      throw std::runtime_error("ptrfield: unknown struct " + st.name.name);
    bool found = false;
    for (const auto &f: sIt->second->fields)
      if (f.name == arg.field) {
        found = true;
        break;
      }
    if (!found)
      throw std::runtime_error("ptrfield: unknown field " + arg.field);
    std::uint64_t off = fieldOffsetTagUnits(*sIt->second, arg.field, structs_);
    smt::Term newAddr = ptrTerm;
    if (off != 0) {
      auto offT = solver.make_bv_value_int64(bv64, static_cast<int64_t>(off));
      newAddr = solver.make_term(smt::Kind::BV_ADD, {ptrTerm, offT});
    }

    TypePtr fieldTy;
    for (const auto &f: sIt->second->fields)
      if (f.name == arg.field) {
        fieldTy = f.type;
        break;
      }
    smt::Term prov_base = ptrTerm;
    std::uint64_t structSize = sizeofTagUnits(pt.pointee, structs_);
    smt::Term prov_size = solver.make_bv_value_int64(bv64, static_cast<int64_t>(structSize));
    return SymbolicValue(
        SymbolicValue::Kind::Int, newAddr, solver.make_true(), prov_base, prov_size
    );

    return SymbolicValue(SymbolicValue::Kind::Undef);
  }

  SymbolicValue SymbolicExecutor::evalLoadAtom(
      const LoadAtom &arg, smt::ISolver &solver, SymbolicStore &store, std::vector<smt::Term> &pc,
      std::vector<smt::Term> &ub
  ) {

    if (!currentFun_)
      throw std::runtime_error("load encountered without active FunDecl");

    SymbolicValue ptrVal = evalLValue(arg.rval, solver, store, pc, ub);
    smt::Term ptrTerm = ptrVal.term;

    const TypePtr baseType = resolveLValueType(arg.rval);
    if (!baseType || !std::holds_alternative<PtrType>(baseType->v))
      throw std::runtime_error("load target is not ptr-typed: " + arg.rval.base.name);
    const TypePtr pointeeType = std::get<PtrType>(baseType->v).pointee;

    auto pointeeSort = getSort(pointeeType, solver);
    smt::Term result;
    if (solver.is_fp_sort(pointeeSort)) {
      auto [exp, sig] = solver.get_fp_dims(pointeeSort);
      auto zeroBv = solver.make_bv_value(solver.make_bv_sort(exp + sig), "0", 10);
      result = solver.make_term(smt::Kind::FP_TO_FP_FROM_SBV, {zeroBv}, {exp, sig});
    } else {
      result = solver.make_bv_value(pointeeSort, "0", 10);
    }
    auto bv64 = solver.make_bv_sort(kPtrBits);
    auto zero = solver.make_bv_value_int64(bv64, 0);

    if (ptrVal.prov_base.internal && ptrVal.prov_size.internal) {
      auto hasProv = solver.make_term(smt::Kind::DISTINCT, {ptrVal.prov_base, zero});
      auto inBoundsLower = solver.make_term(smt::Kind::BV_ULE, {ptrVal.prov_base, ptrTerm});
      auto endAddr = solver.make_term(smt::Kind::BV_ADD, {ptrVal.prov_base, ptrVal.prov_size});
      auto inBoundsUpper = solver.make_term(smt::Kind::BV_ULT, {ptrTerm, endAddr});
      auto cond = solver.make_term(
          smt::Kind::IMPLIES,
          {hasProv, solver.make_term(smt::Kind::AND, {inBoundsLower, inBoundsUpper})}
      );
      ub.push_back(cond);
    }

    smt::Term res_prov_base = zero;
    smt::Term res_prov_size = zero;
    // Track the loaded cell's definedness: initialised from true and
    // ITE-merged per matching cell below.  Rule 3 enforcement (pushing
    // to `ub`) happens after the enumeration loop, before the return.
    smt::Term res_is_defined = solver.make_true();
    std::vector<smt::Term> matchConds;
    std::function<void(const TypePtr &, SymbolicValue &, std::uint64_t, std::uint64_t)> enumLoad;
    enumLoad = [&](const TypePtr &ty, SymbolicValue &sv, std::uint64_t baseTag, std::uint64_t off) {
      if (!ty)
        return;
      if (typeMatch(ty, pointeeType)) {
        auto tagTerm = solver.make_bv_value_int64(bv64, static_cast<int64_t>(baseTag + off));
        auto cond = solver.make_term(smt::Kind::EQUAL, {ptrTerm, tagTerm});
        matchConds.push_back(cond);
        result = solver.make_term(smt::Kind::ITE, {cond, sv.term, result});

        auto svBase = sv.prov_base.internal ? sv.prov_base : zero;
        auto svSize = sv.prov_size.internal ? sv.prov_size : zero;
        res_prov_base = solver.make_term(smt::Kind::ITE, {cond, svBase, res_prov_base});
        res_prov_size = solver.make_term(smt::Kind::ITE, {cond, svSize, res_prov_size});

        auto svDef = sv.is_defined.internal ? sv.is_defined : solver.make_true();
        res_is_defined = solver.make_term(smt::Kind::ITE, {cond, svDef, res_is_defined});
        return;
      }
      if (auto at = std::get_if<ArrayType>(&ty->v)) {
        std::uint64_t stride = sizeofTagUnits(at->elem, structs_);
        for (std::uint64_t k = 0; k < at->size && k < sv.arrayVal.size(); ++k)
          enumLoad(at->elem, sv.arrayVal[k], baseTag, off + k * stride);
        return;
      }
      if (auto st = std::get_if<StructType>(&ty->v)) {
        auto sIt = structs_.find(st->name.name);
        if (sIt == structs_.end())
          return;
        std::uint64_t fOff = 0;
        for (const auto &f: sIt->second->fields) {
          auto fIt = sv.structVal.find(f.name);
          if (fIt != sv.structVal.end())
            enumLoad(f.type, fIt->second, baseTag, off + fOff);
          fOff += sizeofTagUnits(f.type, structs_);
        }
        return;
      }
    };
    for (const auto &l: currentFun_->lets) {
      std::uint64_t baseTag = tagOfLocal(currentFrame_, l.name.name);
      enumLoad(l.type, store.at(l.name.name), baseTag, 0);
    }
    // SPEC §9.6.1 step 4: when this load fires inside a
    // callee, a pointer parameter may point at a caller-owned
    // `let mut`.  Walk the caller frame too so the load sees
    // the caller's current value.
    if (outerFun_ && outerStore_) {
      for (const auto &l: outerFun_->lets) {
        auto it = outerStore_->find(l.name.name);
        if (it == outerStore_->end())
          continue;
        enumLoad(l.type, it->second, tagOfLocal(outerFrame_, l.name.name), 0);
      }
    }
    auto nullPtr = solver.make_bv_value_int64(bv64, 0);
    ub.push_back(solver.make_term(smt::Kind::DISTINCT, {ptrTerm, nullPtr}));
    if (!matchConds.empty()) {
      smt::Term anyMatch = matchConds[0];
      for (size_t i = 1; i < matchConds.size(); ++i)
        anyMatch = solver.make_term(smt::Kind::OR, {anyMatch, matchConds[i]});
      ub.push_back(anyMatch);
    }
    // Rule 3: load is a read — reading a cell whose value is undef is
    // UB at the load site.  Push the loaded cell's definedness into the
    // UB guards so paths that load undef are infeasible.
    ub.push_back(res_is_defined);
    return SymbolicValue(
        SymbolicValue::Kind::Int, result, res_is_defined, res_prov_base, res_prov_size
    );

    return SymbolicValue(SymbolicValue::Kind::Undef);
  }

  SymbolicValue SymbolicExecutor::evalCallAtom(
      const CallAtom &arg, smt::ISolver &solver, SymbolicStore &store, std::vector<smt::Term> &pc,
      std::vector<smt::Term> &ub
  ) {

    // Phase 4 handles intrinsic calls only. Fun/decl
    // targets are stubbed for Phases 6/7/8.
    // Evaluate arguments left-to-right first (§2.12 strict commit)
    // so we can match intrinsic overloads by argument type.
    // When the typechecker pinned a resolved intrinsic
    // overload, propagate each parameter's sort to evalExpr so FP
    // literals (`@fmin(%?x, 2.0)`) bind at the param's precision
    // rather than the f32 default.  Without this, a literal arg
    // typed at f32 collides with an f64 sym at FP_LT/FP_GT inside
    // the @fmin / @fmax encodings.
    std::vector<SymbolicValue> argVals;
    argVals.reserve(arg.args.size());
    for (size_t k = 0; k < arg.args.size(); ++k) {
      std::optional<smt::Sort> expected;
      if (arg.resolvedIntrinsic && k < arg.resolvedIntrinsic->params.size())
        expected = getSort(arg.resolvedIntrinsic->params[k].type, solver);
      argVals.push_back(evalExpr(*arg.args[k], solver, store, pc, ub, expected));
    }
    // Honour the overload pinned by the type checker
    // when available. The width-comparison fallback below is
    // kept for un-typechecked inputs (raw unit tests).
    const IntrinsicDecl *intr = arg.resolvedIntrinsic;
    if (!intr) {
      for (const auto &i: prog_.intrinsics) {
        if (i.name.name != arg.callee.name)
          continue;
        if (i.params.size() != argVals.size())
          continue;
        bool match = true;
        for (size_t k = 0; k < argVals.size(); ++k) {
          auto pb = TypeUtils::getIntBitWidth(i.params[k].type);
          if (pb && argVals[k].kind == SymbolicValue::Kind::Int) {
            auto sort = solver.get_sort(argVals[k].term);
            if (solver.is_bv_sort(sort) && solver.get_bv_width(sort) != *pb) {
              match = false;
              break;
            }
          }
        }
        if (match) {
          intr = &i;
          break;
        }
      }
    }
    if (!intr) {
      // `fun` target -- nested symbolic exec.
      // Hand the caller frame down so callee `store`s can land
      // on caller-side `let mut` targets per SPEC §9.6.1 step 4.
      for (const auto &f: prog_.funs)
        if (f.name.name == arg.callee.name) {
          return callFunction(f, std::move(argVals), solver, pc, ub, currentFun_, &store);
        }
      // Contract-form `decl` target.
      for (const auto &d: prog_.extDecls)
        if (d.name.name == arg.callee.name) {
          if (!d.contract)
            throw std::runtime_error(
                "Solver: link-form `decl` has no body and no contract: " + arg.callee.name
            );
          return callContract(d, arg.args, std::move(argVals), solver, store, pc, ub);
        }
      throw std::runtime_error("Solver: call target not found: " + arg.callee.name);
    }
    // Delegate to the dedicated intrinsics module so
    // all solver-side intrinsic SMT lowering lives in one place.
    // See src/solver/intrinsics.cpp.
    return callBuiltinIntrinsicSMT(*intr, argVals, solver, pc, ub);

    return SymbolicValue(SymbolicValue::Kind::Undef);
  }

  smt::Term SymbolicExecutor::evalCoef(
      const Coef &c, smt::ISolver &solver, SymbolicStore &store,
      std::optional<smt::Sort> expectedSort
  ) {
    if (auto lit = std::get_if<IntLit>(&c)) {
      smt::Sort s = expectedSort.value_or(solver.make_bv_sort(32));
      return solver.make_bv_value(s, std::to_string(lit->value), 10);
    }
    if (auto flit = std::get_if<FloatLit>(&c)) {
      smt::Sort s = expectedSort.value_or(solver.make_fp_sort(8, 24));
      return solver.make_fp_value_from_real(s, flit->value, smt::RoundingMode::RNE);
    }
    if (std::get_if<NullLit>(&c)) {
      // null = 0 as BV64 (pointer width)
      return solver.make_bv_value(solver.make_bv_sort(64), "0", 10);
    }
    auto id = std::get<LocalOrSymId>(c);
    return std::visit([&](auto &&v) { return store.at(v.name).term; }, id);
  }

  SymbolicValue SymbolicExecutor::evalSelectVal(
      const SelectVal &sv, smt::ISolver &solver, SymbolicStore &store, std::vector<smt::Term> &pc,
      std::vector<smt::Term> &ub, std::optional<smt::Sort> expectedSort
  ) {
    if (auto rv = std::get_if<RValue>(&sv))
      return evalLValue(*rv, solver, store, pc, ub);
    smt::Term t = evalCoef(std::get<Coef>(sv), solver, store, expectedSort);
    return SymbolicValue(SymbolicValue::Kind::Int, t, solver.make_true());
  }

  smt::Term SymbolicExecutor::evalCond(
      const Cond &c, smt::ISolver &solver, SymbolicStore &store, std::vector<smt::Term> &pc,
      std::vector<smt::Term> &ub
  ) {
    SymbolicValue lhsVal = evalExpr(c.lhs, solver, store, pc, ub);
    smt::Term lhs = lhsVal.term;
    SymbolicValue rhsVal = evalExpr(c.rhs, solver, store, pc, ub, solver.get_sort(lhs));
    smt::Term rhs = rhsVal.term;

    // Dynamic Rule 14 Relational Comparison check
    TypePtr lhsType = resolveExprType(c.lhs);
    TypePtr rhsType = resolveExprType(c.rhs);
    bool isLhsPtr = lhsType && std::holds_alternative<PtrType>(lhsType->v);
    bool isRhsPtr = rhsType && std::holds_alternative<PtrType>(rhsType->v);
    bool isRelational =
        (c.op == RelOp::LT || c.op == RelOp::LE || c.op == RelOp::GT || c.op == RelOp::GE);
    if (isRelational && isLhsPtr && isRhsPtr) {
      auto bv64 = solver.make_bv_sort(64);
      auto zero = solver.make_bv_value_int64(bv64, 0);
      if (lhsVal.prov_base.internal && rhsVal.prov_base.internal) {
        auto eqBase = solver.make_term(smt::Kind::EQUAL, {lhsVal.prov_base, rhsVal.prov_base});
        auto nonZeroBase = solver.make_term(smt::Kind::DISTINCT, {lhsVal.prov_base, zero});
        ub.push_back(solver.make_term(smt::Kind::AND, {eqBase, nonZeroBase}));
      } else {
        ub.push_back(solver.make_false());
      }
    }

    auto lSort = solver.get_sort(lhs);
    auto rSort = solver.get_sort(rhs);

    if (solver.is_fp_sort(lSort)) {
      switch (c.op) {
        case RelOp::EQ:
          return solver.make_term(smt::Kind::FP_EQUAL, {lhs, rhs});
        case RelOp::NE:
          return solver.make_term(
              smt::Kind::NOT, {solver.make_term(smt::Kind::FP_EQUAL, {lhs, rhs})}
          );
        case RelOp::LT:
          return solver.make_term(smt::Kind::FP_LT, {lhs, rhs});
        case RelOp::LE:
          return solver.make_term(smt::Kind::FP_LEQ, {lhs, rhs});
        case RelOp::GT:
          return solver.make_term(smt::Kind::FP_GT, {lhs, rhs});
        case RelOp::GE:
          return solver.make_term(smt::Kind::FP_GEQ, {lhs, rhs});
      }
    }

    if (solver.is_bv_sort(lSort) && solver.is_bv_sort(rSort)) {
      auto lWidth = solver.get_bv_width(lSort);
      auto rWidth = solver.get_bv_width(rSort);
      if (lWidth < rWidth) {
        lhs = solver.make_term(smt::Kind::BV_SIGN_EXTEND, {lhs}, {rWidth - lWidth});
      } else if (rWidth < lWidth) {
        rhs = solver.make_term(smt::Kind::BV_SIGN_EXTEND, {rhs}, {lWidth - rWidth});
      }
    }

    switch (c.op) {
      case RelOp::EQ:
        return solver.make_term(smt::Kind::EQUAL, {lhs, rhs});
      case RelOp::NE:
        return solver.make_term(smt::Kind::DISTINCT, {lhs, rhs});
      case RelOp::LT:
        return solver.make_term(smt::Kind::BV_SLT, {lhs, rhs});
      case RelOp::LE:
        return solver.make_term(smt::Kind::BV_SLE, {lhs, rhs});
      case RelOp::GT:
        return solver.make_term(smt::Kind::BV_SGT, {lhs, rhs});
      case RelOp::GE:
        return solver.make_term(smt::Kind::BV_SGE, {lhs, rhs});
    }
    return {};
  }
} // namespace refractir
