#include "solver/solver.hpp"
#include <atomic>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include "analysis/cfg.hpp"
#include "analysis/type_utils.hpp"
#include "internal.hpp"

namespace refractir {

  thread_local const FunDecl *SymbolicExecutor::currentFun_ = nullptr;

  SymbolicExecutor::SymbolicExecutor(
      const Program &prog, const Config &config, SolverFactory solverFactory
  ) : prog_(prog), config_(config), solverFactory_(solverFactory) {
    for (const auto &s: prog_.structs) {
      structs_[s.name.name] = &s;
    }
  }

  void SymbolicExecutor::execInstr(
      const Instr &ins, smt::ISolver &solver, SymbolicStore &store,
      std::vector<smt::Term> &pathConstraints, std::vector<smt::Term> &ubGuards,
      std::vector<smt::Term> &requirements
  ) {
    std::visit(
        [&](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, AssignInstr>) {
            auto lhsVal =
                evalLValue(arg.lhs, solver, store, pathConstraints, ubGuards, /*forWrite=*/true);
            // Vector LHS: evaluate RHS as a SymbolicValue::Vec.
            if (lhsVal.kind == SymbolicValue::Kind::Vec && arg.lhs.accesses.empty()) {
              // Find the LHS local's declared VecType.
              TypePtr lhsType;
              if (currentFun_) {
                for (const auto &l: currentFun_->lets)
                  if (l.name.name == arg.lhs.base.name) {
                    lhsType = l.type;
                    break;
                  }
                if (!lhsType)
                  for (const auto &p: currentFun_->params)
                    if (p.name.name == arg.lhs.base.name) {
                      lhsType = p.type;
                      break;
                    }
              }
              if (lhsType && std::holds_alternative<VecType>(lhsType->v)) {
                auto &vt = std::get<VecType>(lhsType->v);
                SymbolicValue rhsV =
                    evalVecExpr(arg.rhs, vt, solver, store, pathConstraints, ubGuards);
                setLValue(arg.lhs, rhsV, solver, store, pathConstraints, ubGuards);
                return;
              }
            }
            auto rhs = evalExpr(
                arg.rhs, solver, store, pathConstraints, ubGuards,
                lhsVal.kind == SymbolicValue::Kind::Int
                    ? std::optional(solver.get_sort(lhsVal.term))
                    : std::nullopt
            );
            setLValue(arg.lhs, rhs, solver, store, pathConstraints, ubGuards);
            // Track ptr provenance for cross-object and one-
            // past-end UB checks. The LHS is either a whole-local ptr
            // or a ptr-typed struct field path (`%s.p1`); we mirror
            // the RHS atom's provenance (addr / ptrindex / ptrfield
            // set a known provenance, pointer arithmetic preserves
            // it, load-derived ptrs are unknown).
            if (currentFun_) {
              // Resolve the LHS type by walking accesses.
              auto resolveLhsType = [&]() -> TypePtr {
                TypePtr cur;
                for (const auto &l: currentFun_->lets)
                  if (l.name.name == arg.lhs.base.name) {
                    cur = l.type;
                    break;
                  }
                if (!cur)
                  for (const auto &p: currentFun_->params)
                    if (p.name.name == arg.lhs.base.name) {
                      cur = p.type;
                      break;
                    }
                for (const auto &acc: arg.lhs.accesses) {
                  if (!cur)
                    return nullptr;
                  if (auto af = std::get_if<AccessField>(&acc)) {
                    auto st = std::get_if<StructType>(&cur->v);
                    if (!st)
                      return nullptr;
                    auto sIt = structs_.find(st->name.name);
                    if (sIt == structs_.end())
                      return nullptr;
                    cur = nullptr;
                    for (const auto &f: sIt->second->fields)
                      if (f.name == af->field) {
                        cur = f.type;
                        break;
                      }
                  } else if (auto ai = std::get_if<AccessIndex>(&acc)) {
                    (void) ai;
                    if (auto at = std::get_if<ArrayType>(&cur->v))
                      cur = at->elem;
                    else if (auto vt = std::get_if<VecType>(&cur->v))
                      cur = vt->elem;
                    else
                      return nullptr;
                  }
                }
                return cur;
              };
              auto isPtr = [&]() {
                auto t = resolveLhsType();
                return t && std::holds_alternative<PtrType>(t->v);
              };
              auto buildLhsKey = [&]() -> std::string {
                // Only field-keyed accesses (no dynamic indices).
                std::string key = arg.lhs.base.name;
                for (const auto &acc: arg.lhs.accesses) {
                  if (auto af = std::get_if<AccessField>(&acc)) {
                    key += "." + af->field;
                  } else {
                    return {};
                  }
                }
                return key;
              };
              std::string lhsKey = isPtr() ? buildLhsKey() : "";
              if (!lhsKey.empty()) {
                auto provFromName = [&](const std::string &src) { return prov_.lookup(src); };
                auto compute = [&]() -> std::optional<PtrProvenance> {
                  if (arg.rhs.rest.empty()) {
                    const auto &a = arg.rhs.first.v;
                    if (auto addr = std::get_if<AddrAtom>(&a)) {
                      // Provenance = the addressed local; size is the
                      // immediate containing object's total tag-unit
                      // span (spec rule 15). For `addr %arr[k]` that
                      // remains the whole array; for `addr %s.f` the
                      // whole struct.
                      uint64_t baseTag = tagOfLocal(currentFrame_, addr->lv.base.name);
                      TypePtr ty;
                      for (const auto &l: currentFun_->lets)
                        if (l.name.name == addr->lv.base.name) {
                          ty = l.type;
                          break;
                        }
                      if (!ty)
                        for (const auto &p: currentFun_->params)
                          if (p.name.name == addr->lv.base.name) {
                            ty = p.type;
                            break;
                          }
                      std::uint64_t size = sizeofTagUnits(ty, structs_);
                      return PtrProvenance{baseTag, size};
                    }
                    if (auto pi = std::get_if<PtrIndexAtom>(&a)) {
                      // Narrow provenance for ptrindex.
                      // The result pointer's provenance = the array the
                      // source points to. Compute the narrowed sub-array
                      // range from the source's type (ptr [N] T).
                      auto srcProv = provFromName(buildLValueKey(pi->rval));
                      if (srcProv && currentFun_) {
                        TypePtr baseType = resolveLValueType(pi->rval);
                        if (baseType) {
                          if (auto pt = std::get_if<PtrType>(&baseType->v)) {
                            if (auto at = std::get_if<ArrayType>(&pt->pointee->v)) {
                              // Narrowed size = N * sizeofTagUnits(T)
                              std::uint64_t elemUnits = sizeofTagUnits(at->elem, structs_);
                              std::uint64_t narrowSize = at->size * elemUnits;
                              // The narrowed base = source ptrVal at
                              // assignment time. We don't have the
                              // concrete tag here, so approximate:
                              // baseTag stays at the source's base but
                              // size is narrowed.
                              return PtrProvenance{srcProv->baseTag, narrowSize};
                            }
                          }
                        }
                      }
                      return srcProv;
                    }
                    if (auto pf = std::get_if<PtrFieldAtom>(&a))
                      return provFromName(buildLValueKey(pf->rval));
                    if (auto ca = std::get_if<CoefAtom>(&a)) {
                      if (auto id = std::get_if<LocalOrSymId>(&ca->coef))
                        if (auto lid = std::get_if<LocalId>(id))
                          return provFromName(lid->name);
                    }
                    if (auto rv = std::get_if<RValueAtom>(&a)) {
                      return provFromName(buildLValueKey(rv->rval));
                    }
                    // LoadAtom-derived ptrs: provenance unknown.
                  } else {
                    // `%p = %q + i` style: provenance carries from %q.
                    const auto &a = arg.rhs.first.v;
                    if (auto ca = std::get_if<CoefAtom>(&a)) {
                      if (auto id = std::get_if<LocalOrSymId>(&ca->coef))
                        if (auto lid = std::get_if<LocalId>(id))
                          return provFromName(lid->name);
                    }
                    if (auto rv = std::get_if<RValueAtom>(&a)) {
                      return provFromName(buildLValueKey(rv->rval));
                    }
                  }
                  return std::nullopt;
                };
                auto newProv = compute();
                if (newProv)
                  prov_.set(lhsKey, *newProv);
                else
                  prov_.erase(lhsKey);
              }
            }
          } else if constexpr (std::is_same_v<T, AssumeInstr>) {
            pathConstraints.push_back(evalCond(arg.cond, solver, store, pathConstraints, ubGuards));
          } else if constexpr (std::is_same_v<T, RequireInstr>) {
            requirements.push_back(evalCond(arg.cond, solver, store, pathConstraints, ubGuards));
          } else if constexpr (std::is_same_v<T, StoreInstr>) {
            // store %p, %v — mux-update every candidate target's value:
            //   for each %t of pointee type: %t := ite(p == tag_t, v, %t)
            if (!currentFun_)
              throw std::runtime_error("store encountered without active FunDecl");

            // Evaluate ptr term (BV64) and stored value term.
            SymbolicValue ptrVal = evalExpr(arg.ptr, solver, store, pathConstraints, ubGuards);
            smt::Term ptrTerm = ptrVal.term;

            // Rule 9/11: store through null or OOB is UB.
            // The evalExpr above already pushes rule-10 OOB constraints
            // for ptr-arith expressions, but a bare `store %pa, v`
            // where %pa was set earlier still needs a null check.
            auto bv64Store = solver.make_bv_sort(kPtrBits);
            auto nullStore = solver.make_bv_value_int64(bv64Store, 0);
            ubGuards.push_back(solver.make_term(smt::Kind::DISTINCT, {ptrTerm, nullStore}));

            if (ptrVal.prov_base.internal && ptrVal.prov_size.internal) {
              auto zero = solver.make_bv_value_int64(bv64Store, 0);
              auto hasProv = solver.make_term(smt::Kind::DISTINCT, {ptrVal.prov_base, zero});
              auto inBoundsLower = solver.make_term(smt::Kind::BV_ULE, {ptrVal.prov_base, ptrTerm});
              auto endAddr =
                  solver.make_term(smt::Kind::BV_ADD, {ptrVal.prov_base, ptrVal.prov_size});
              auto inBoundsUpper = solver.make_term(smt::Kind::BV_ULT, {ptrTerm, endAddr});
              auto cond = solver.make_term(
                  smt::Kind::IMPLIES,
                  {hasProv, solver.make_term(smt::Kind::AND, {inBoundsLower, inBoundsUpper})}
              );
              ubGuards.push_back(cond);
            }

            // Determine pointee type from the ptr expression's first atom.
            TypePtr pointeeType;
            if (auto *rv = std::get_if<RValueAtom>(&arg.ptr.first.v)) {
              TypePtr rvalType = resolveLValueType(rv->rval);
              if (rvalType) {
                if (auto pt = std::get_if<PtrType>(&rvalType->v)) {
                  pointeeType = pt->pointee;
                }
              }
            }
            if (!pointeeType)
              throw std::runtime_error(
                  "store: cannot derive pointee type (only `store %p, ...` "
                  "with a ptr-typed local or parameter %p is currently supported)"
              );

            auto pointeeSort = getSort(pointeeType, solver);
            SymbolicValue valVal = evalExpr(
                arg.val, solver, store, pathConstraints, ubGuards, std::optional(pointeeSort)
            );
            smt::Term valTerm = valVal.term;

            auto bv64 = solver.make_bv_sort(kPtrBits);
            // Mirror the load enumeration: recurse over (type, value,
            // offset) so a store can target any scalar leaf of a
            // nested aggregate — array-of-structs, struct-of-arrays.
            std::function<void(const TypePtr &, SymbolicValue &, std::uint64_t, std::uint64_t)>
                enumStore;
            std::vector<smt::Term> storeMatchConds;
            enumStore = [&](const TypePtr &ty, SymbolicValue &sv, std::uint64_t baseTag,
                            std::uint64_t off) {
              if (!ty)
                return;
              if (typeMatch(ty, pointeeType)) {
                auto tagTerm =
                    solver.make_bv_value_int64(bv64, static_cast<int64_t>(baseTag + off));
                auto cond = solver.make_term(smt::Kind::EQUAL, {ptrTerm, tagTerm});
                storeMatchConds.push_back(cond);
                sv.term = solver.make_term(smt::Kind::ITE, {cond, valTerm, sv.term});
                // When the pointee is itself a pointer, the stored
                // value carries its own provenance (base/size). The target's
                // provenance must follow the value under the same `cond`,
                // mirroring the select-arm ITE — otherwise a later store/load
                // through this redirected pointer bounds-checks `ptrTerm`
                // against the stale base the target held before this store,
                // a spurious UNSAT. Absent provenance reads as tag 0 (the
                // bounds check is gated on base != 0, so it self-disables).
                if (std::get_if<PtrType>(&pointeeType->v)) {
                  auto zeroP = solver.make_bv_value_int64(bv64, 0);
                  auto curBase = sv.prov_base.internal ? sv.prov_base : zeroP;
                  auto curSize = sv.prov_size.internal ? sv.prov_size : zeroP;
                  auto newBase = valVal.prov_base.internal ? valVal.prov_base : zeroP;
                  auto newSize = valVal.prov_size.internal ? valVal.prov_size : zeroP;
                  sv.prov_base = solver.make_term(smt::Kind::ITE, {cond, newBase, curBase});
                  sv.prov_size = solver.make_term(smt::Kind::ITE, {cond, newSize, curSize});
                }
                // Writing makes the cell as defined as the stored value, so
                // a later load of a now-written cell no longer trips the
                // rule-3 undef constraint (paired with the load's
                // is_defined propagation).
                auto curDef = sv.is_defined.internal ? sv.is_defined : solver.make_true();
                auto valDef = valVal.is_defined.internal ? valVal.is_defined : solver.make_true();
                sv.is_defined = solver.make_term(smt::Kind::ITE, {cond, valDef, curDef});
                return;
              }
              if (auto at = std::get_if<ArrayType>(&ty->v)) {
                std::uint64_t stride = sizeofTagUnits(at->elem, structs_);
                for (std::uint64_t k = 0; k < at->size && k < sv.arrayVal.size(); ++k)
                  enumStore(at->elem, sv.arrayVal[k], baseTag, off + k * stride);
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
                    enumStore(f.type, fIt->second, baseTag, off + fOff);
                  fOff += sizeofTagUnits(f.type, structs_);
                }
                return;
              }
            };
            for (const auto &l: currentFun_->lets) {
              std::uint64_t baseTag = tagOfLocal(currentFrame_, l.name.name);
              enumStore(l.type, store.at(l.name.name), baseTag, 0);
            }
            // Rule 11/15b: the store must land on a valid
            // T-typed cell — same as load's anyMatch constraint.
            if (!storeMatchConds.empty()) {
              smt::Term anyMatch = storeMatchConds[0];
              for (size_t j = 1; j < storeMatchConds.size(); ++j)
                anyMatch = solver.make_term(smt::Kind::OR, {anyMatch, storeMatchConds[j]});
              ubGuards.push_back(anyMatch);
            }
          }
        },
        ins
    );
  }

  void SymbolicExecutor::bindParamDomain(
      const TypePtr &type, const SymbolicValue &value, smt::ISolver &solver,
      std::vector<smt::Term> &pathConstraints
  ) {
    if (!config_.tensorParamDomain)
      return;
    const auto [lo, hi] = *config_.tensorParamDomain;
    if (lo > hi)
      throw std::invalid_argument("tensor-param-domain: lower bound exceeds upper bound");

    // Every leaf takes the bound, so the leaves and their scalar type come
    // from what the parameter declared: a tensor or vector parameter is one
    // term per element, a scalar parameter is one term.
    TypePtr leafType = type;
    std::vector<smt::Term> leaves;
    switch (value.kind) {
      case SymbolicValue::Kind::Tensor:
      case SymbolicValue::Kind::Vec:
        for (const auto &leaf: value.arrayVal)
          leaves.push_back(leaf.term);
        if (const auto *tensor = TypeUtils::asTensor(type))
          leafType = tensor->elem;
        else if (const auto *vec = std::get_if<VecType>(&type->v))
          leafType = vec->elem;
        break;
      case SymbolicValue::Kind::Int:
        leaves.push_back(value.term);
        break;
      default:
        throw std::invalid_argument(
            "tensor-param-domain: entry parameter has a pointer or aggregate leaf"
        );
    }
    if (!std::holds_alternative<IntType>(leafType->v))
      throw std::invalid_argument("tensor-param-domain: entry parameter leaf is not an integer");

    // Clamp to the leaf type's range so make_bv_value_int64 cannot wrap the
    // bound. A requested range that holds no value of the type is an
    // unsatisfiable query, not a malformed one.
    const std::uint32_t bits = TypeUtils::getIntBitWidth(leafType).value_or(0);
    int64_t effectiveLo = lo;
    int64_t effectiveHi = hi;
    if (bits > 0 && bits < 64) {
      effectiveLo = std::max(effectiveLo, -(int64_t{1} << (bits - 1)));
      effectiveHi = std::min(effectiveHi, (int64_t{1} << (bits - 1)) - 1);
    }
    if (effectiveLo > effectiveHi) {
      pathConstraints.push_back(solver.make_false());
      return;
    }

    const auto sort = getSort(leafType, solver);
    const auto loTerm = solver.make_bv_value_int64(sort, effectiveLo);
    const auto hiTerm = solver.make_bv_value_int64(sort, effectiveHi);
    for (const auto &leaf: leaves) {
      pathConstraints.push_back(solver.make_term(smt::Kind::BV_SLE, {loTerm, leaf}));
      pathConstraints.push_back(solver.make_term(smt::Kind::BV_SLE, {leaf, hiTerm}));
    }
  }

  SymbolicExecutor::Result SymbolicExecutor::solve(
      const std::string &funcName, const std::vector<std::string> &path,
      const std::unordered_map<std::string, int64_t> &fixedSyms
  ) {

    const FunDecl *entry = nullptr;
    for (const auto &f: prog_.funs) {
      if (f.name.name == funcName) {
        entry = &f;
        break;
      }
    }
    if (!entry)
      throw std::runtime_error("Function not found: " + funcName);

    // Make the current FunDecl visible to evalAtom/StoreInstr handlers via
    // thread_local storage. Restored on scope exit so nested or concurrent
    // solve() invocations on different threads see their own value.
    struct FunGuard {
      const FunDecl *prev;

      ~FunGuard() { SymbolicExecutor::currentFun_ = prev; }
    } funGuard{currentFun_};

    currentFun_ = entry;

    auto solverPtr = solverFactory_(config_);
    smt::ISolver &solver = *solverPtr;

    SymbolicStore store;
    std::vector<smt::Term> pathConstraints;
    std::vector<smt::Term> ubGuards;
    std::vector<smt::Term> requirements;

    // Make `requirements` reachable from nested callees.
    currentReq_ = &requirements;

    struct ReqGuard {
      std::vector<smt::Term> **slot;

      ~ReqGuard() { *slot = nullptr; }
    } reqGuard{&currentReq_};

    // Reset shared sym cache for each solve call so
    // independent paths don't reuse stale sym constants. Clear at
    // start AND on scope exit so the smt::Term destructors fire while
    // their owning solver is still alive (declared above).
    calleeSyms_.clear();
    // Seed the callee-branch RNG. Same seed reproduces.
    calleeRng_.seed(config_.seed ? config_.seed : 0xC0DEFACE);

    struct CacheGuard {
      std::unordered_map<const FunDecl *, std::unordered_map<std::string, SymbolicValue>> &cache;

      ~CacheGuard() { cache.clear(); }
    } cacheGuard{calleeSyms_};

    // Reset per-solve provenance tracking. Each call to solve()
    // walks a fresh CFG path, so prior provenance state must not leak.
    prov_.clear();
    // Activation ids likewise. They are handed out along the path being
    // walked, so a fresh walk starts at the entry frame and counts again —
    // which is what keeps a tag reproducible for a given path.
    currentFrame_ = kEntryFrame;
    outerFrame_ = kEntryFrame;
    nextFrame_ = kEntryFrame + 1;
    // Local for the captured top-level RetTerm term. The
    // path-traversal lambda captures by reference and writes into it
    // when it sees `ret <expr>;`. Lives on the stack of solve() so
    // the shared_ptr<bitwuzla::Term> goes away before the solver
    // does. Storing it as a class member would race when sample()
    // runs workers concurrently against the same SymbolicExecutor.
    smt::Term retTermLocal;

    // 1. Declare symbols and fix values if requested
    for (const auto &s: entry->syms) {
      auto sv = createSymbolicValue(s.type, s.name.name, solver, &pathConstraints);
      store[s.name.name] = sv;

      // Vector and tensor symbols expand to independent scalar terms for
      // domain/fix constraints. Scalar symbols contribute one term.
      std::vector<smt::Term> symLaneTerms;
      TypePtr symLaneType;
      if (sv.kind == SymbolicValue::Kind::Vec || sv.kind == SymbolicValue::Kind::Tensor) {
        for (const auto &lane: sv.arrayVal)
          symLaneTerms.push_back(lane.term);
        if (auto vt = std::get_if<VecType>(&s.type->v))
          symLaneType = vt->elem;
        else if (auto tt = std::get_if<TensorType>(&s.type->v))
          symLaneType = tt->elem;
      } else {
        symLaneTerms.push_back(sv.term);
        symLaneType = s.type;
      }

      // Add domain constraints
      if (s.domain) {
        std::visit(
            [&](auto &&d) {
              using T = std::decay_t<decltype(d)>;
              // Domain constraints apply per-lane for vector syms (§3.4.1
              // says each lane gets the same domain).
              auto laneSort = getSort(symLaneType, solver);
              if constexpr (std::is_same_v<T, DomainInterval>) {
                uint32_t bits = 64;
                if (auto *it = std::get_if<IntType>(&symLaneType->v)) {
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
                }
                int64_t effLo = d.lo, effHi = d.hi;
                if (bits < 64) {
                  int64_t typeLo = -(1LL << (bits - 1));
                  int64_t typeHi = (1LL << (bits - 1)) - 1;
                  effLo = std::max(effLo, typeLo);
                  effHi = std::min(effHi, typeHi);
                }
                if (effLo <= effHi) {
                  auto lo = solver.make_bv_value_int64(laneSort, effLo);
                  auto hi = solver.make_bv_value_int64(laneSort, effHi);
                  for (const auto &t: symLaneTerms) {
                    pathConstraints.push_back(solver.make_term(smt::Kind::BV_SLE, {lo, t}));
                    pathConstraints.push_back(solver.make_term(smt::Kind::BV_SLE, {t, hi}));
                  }
                }
              } else if constexpr (std::is_same_v<T, DomainSet>) {
                for (const auto &t: symLaneTerms) {
                  std::vector<smt::Term> or_terms;
                  for (auto v: d.values) {
                    auto vt = solver.make_bv_value_int64(laneSort, v);
                    or_terms.push_back(solver.make_term(smt::Kind::EQUAL, {t, vt}));
                  }
                  if (!or_terms.empty()) {
                    smt::Term or_all = or_terms[0];
                    for (size_t i = 1; i < or_terms.size(); ++i)
                      or_all = solver.make_term(smt::Kind::OR, {or_all, or_terms[i]});
                    pathConstraints.push_back(or_all);
                  }
                }
              }
            },
            *s.domain
        );
      }

      if (fixedSyms.count(s.name.name)) {
        auto val =
            solver.make_bv_value_int64(getSort(symLaneType, solver), fixedSyms.at(s.name.name));
        for (const auto &t: symLaneTerms)
          pathConstraints.push_back(solver.make_term(smt::Kind::EQUAL, {t, val}));
      }
    }

    // 2. Declare locals (parameters are also in store)
    for (const auto &p: entry->params) {
      auto sv = createSymbolicValue(p.type, p.name.name, solver, &pathConstraints);
      bindParamDomain(p.type, sv, solver, pathConstraints);
      store[p.name.name] = sv;
    }
    for (const auto &l: entry->lets) {
      if (l.init) {
        store[l.name.name] = evalInit(*l.init, l.type, solver, store, pathConstraints, ubGuards);
      } else {
        store[l.name.name] = makeUndef(l.type, solver);
      }
    }

    // 3. CFG Build for label mapping
    DiagBag diags;
    CFG cfg = CFG::build(*entry, diags);
    if (diags.hasErrors())
      throw std::runtime_error("CFG build failed");

    // RequireNonterm: the path is a lasso whose final block is
    // the loop header (it must reappear earlier — its first occurrence is
    // the orbit entry). We snapshot the complete mutable state at the
    // header's entry on *every* arrival. Step 5 then asserts the last equals
    // the first, so one UB-free orbit recurs forever.
    //
    // The number of arrivals is the requested period: k+1 arrivals means a
    // k-lap orbit, so the path itself carries k and no separate knob is
    // needed. For k > 1, step 5 additionally requires each intermediate
    // arrival to *differ* from the header state, which pins the period at
    // exactly k — otherwise any shorter orbit (in particular a period-1 one,
    // which the additive corrections can always produce) would satisfy the
    // closing equality and silently downgrade the request.
    std::string nontermHeader;
    std::vector<size_t> nontermVisits;
    bool nontermLasso = false;
    if (config_.mode == SolvingMode::RequireNonterm && !path.empty()) {
      nontermHeader = path.back();
      for (size_t i = 0; i < path.size(); ++i)
        if (path[i] == nontermHeader)
          nontermVisits.push_back(i);
      nontermLasso = (nontermVisits.size() >= 2);
    }
    std::vector<SymbolicStore> nontermSnaps(nontermVisits.size());

    // 4. Path traversal
    for (size_t i = 0; i < path.size(); ++i) {
      const std::string &label = path[i];
      if (cfg.indexOf.find(label) == cfg.indexOf.end())
        throw std::runtime_error("Invalid block label in path: " + label);

      const Block &block = entry->blocks[cfg.indexOf.at(label)];

      // Capture the header-entry state (before any of the block's
      // instructions run) on every arrival at the lasso header.
      if (config_.mode == SolvingMode::RequireNonterm && nontermLasso) {
        for (size_t v = 0; v < nontermVisits.size(); ++v)
          if (nontermVisits[v] == i)
            nontermSnaps[v] = store;
      }

      for (const auto &ins: block.instrs) {
        execInstr(ins, solver, store, pathConstraints, ubGuards, requirements);
      }

      // Evaluate the terminator. For a conditional br on a non-final block
      // we also pick a side (then/else) per the path; the cond is evaluated
      // either way so that any UB triggered by computing the cond (e.g.
      // rule 14 cross-object pointer compare) is captured as a path
      // constraint even when the br is the final block.
      const std::string *nextLabel = (i + 1 < path.size()) ? &path[i + 1] : nullptr;
      std::visit(
          [&](auto &&term) {
            using T = std::decay_t<decltype(term)>;
            if constexpr (std::is_same_v<T, BrTerm>) {
              if (term.isConditional) {
                auto cond = evalCond(*term.cond, solver, store, pathConstraints, ubGuards);
                if (nextLabel) {
                  if (term.thenLabel.name == *nextLabel) {
                    pathConstraints.push_back(cond);
                  } else if (term.elseLabel.name == *nextLabel) {
                    pathConstraints.push_back(solver.make_term(smt::Kind::NOT, {cond}));
                  } else {
                    throw std::runtime_error(
                        "Path edge not in CFG: " + label + " -> " + *nextLabel
                    );
                  }
                }
                // No next block: cond was evaluated for UB side-effects only.
              } else if (nextLabel) {
                if (term.dest.name != *nextLabel)
                  throw std::runtime_error("Path edge not in CFG: " + label + " -> " + *nextLabel);
              }
            } else if constexpr (std::is_same_v<T, RetTerm>) {
              if (nextLabel)
                throw std::runtime_error(
                    "Block " + label + " ends with ret but path has more blocks"
                );
              // Evaluate the return value's expression so that
              // any UB checks it raises (div/mod by zero, signed overflow,
              // load OOB, read of undef, etc.) become path constraints.
              // Stash the term into retTermSlot_ (set by the
              // caller of this lambda) so the SOLVED header can extract
              // its model value.
              if (term.value) {
                retTermLocal = evalExpr(*term.value, solver, store, pathConstraints, ubGuards).term;
              }
            } else {
              if (nextLabel)
                throw std::runtime_error(
                    "Block " + label + " ends with non-branch terminator but path has more blocks"
                );
            }
          },
          block.term
      );
    }

    // 5. Solve
    for (auto c: pathConstraints)
      solver.assert_formula(c);
    for (auto r: requirements)
      solver.assert_formula(r);

    if (config_.mode == SolvingMode::UBFree) {
      for (auto g: ubGuards)
        solver.assert_formula(g);
    } else if (config_.mode == SolvingMode::RequireUB) {
      if (!ubGuards.empty()) {
        smt::Term allSafe = ubGuards[0];
        for (size_t i = 1; i < ubGuards.size(); ++i) {
          allSafe = solver.make_term(smt::Kind::AND, {allSafe, ubGuards[i]});
        }
        solver.assert_formula(solver.make_term(smt::Kind::NOT, {allSafe}));
      } else {
        // No UB guards possible on this path — force UNSAT
        solver.assert_formula(solver.make_false());
      }
    } else if (config_.mode == SolvingMode::RequireNonterm) {
      // Non-termination via a recurrent header state (lasso). Assert every
      // operation on the stem + the orbit is UB-free (exactly as UBFree),
      // then assert the complete mutable state at the header's entry recurs
      // after the orbit's k laps, and — for k > 1 — that it does not recur
      // any earlier. Determinism lifts the single safe orbit to an infinite,
      // UB-free run.
      for (auto g: ubGuards)
        solver.assert_formula(g);

      if (!nontermLasso) {
        // The path never revisits its final block, so there is no cycle
        // whose recurrence could certify divergence — unsatisfiable.
        solver.assert_formula(solver.make_false());
      } else {
        // Bit-exact equality between two SymbolicValues of the same
        // declared type, recursing through aggregate/vector leaves. A
        // pointer let is a scalar Int whose term is its address, so the
        // Int case compares pointer values too. Undef leaves add no
        // constraint (an unread undef cannot influence the next lap).
        std::function<void(const SymbolicValue &, const SymbolicValue &, std::vector<smt::Term> &)>
            collectEq =
                [&](const SymbolicValue &a, const SymbolicValue &b, std::vector<smt::Term> &out) {
                  switch (a.kind) {
                    case SymbolicValue::Kind::Int:
                      if (a.term.internal && b.term.internal)
                        out.push_back(solver.make_term(smt::Kind::EQUAL, {a.term, b.term}));
                      return;
                    case SymbolicValue::Kind::Array:
                    case SymbolicValue::Kind::Vec: {
                      size_t n = std::min(a.arrayVal.size(), b.arrayVal.size());
                      for (size_t k = 0; k < n; ++k)
                        collectEq(a.arrayVal[k], b.arrayVal[k], out);
                      return;
                    }
                    case SymbolicValue::Kind::Tensor: {
                      if (a.tensorShape != b.tensorShape)
                        return;
                      size_t n = std::min(a.arrayVal.size(), b.arrayVal.size());
                      for (size_t k = 0; k < n; ++k)
                        collectEq(a.arrayVal[k], b.arrayVal[k], out);
                      return;
                    }
                    case SymbolicValue::Kind::Struct:
                      for (const auto &[fname, fv]: a.structVal) {
                        auto it = b.structVal.find(fname);
                        if (it != b.structVal.end())
                          collectEq(fv, it->second, out);
                      }
                      return;
                    case SymbolicValue::Kind::Undef:
                    default:
                      return;
                  }
                };

        // Per-leaf equalities between the header state at two arrivals.
        auto stateEqs = [&](const SymbolicStore &x, const SymbolicStore &y) {
          std::vector<smt::Term> eqs;
          for (const auto &l: entry->lets) {
            auto i1 = x.find(l.name.name);
            auto i2 = y.find(l.name.name);
            if (i1 != x.end() && i2 != y.end())
              collectEq(i1->second, i2->second, eqs);
          }
          return eqs;
        };

        // Closing equality: the last arrival restores the header state.
        const SymbolicStore &first = nontermSnaps.front();
        for (auto &e: stateEqs(first, nontermSnaps.back()))
          solver.assert_formula(e);

        // Primitive period: every intermediate arrival differs from the
        // header state, i.e. NOT(all leaves equal). A state with no
        // comparable leaf at all cannot differ, so the orbit is rejected
        // rather than admitted by an empty conjunction.
        for (size_t v = 1; v + 1 < nontermSnaps.size(); ++v) {
          auto eqs = stateEqs(first, nontermSnaps[v]);
          if (eqs.empty()) {
            solver.assert_formula(solver.make_false());
            break;
          }
          smt::Term same = eqs[0];
          for (size_t j = 1; j < eqs.size(); ++j)
            same = solver.make_term(smt::Kind::AND, {same, eqs[j]});
          solver.assert_formula(solver.make_term(smt::Kind::NOT, {same}));
        }
      }
    }

    smt::Result res = solver.check_sat();
    Result finalRes;
    if (res == smt::Result::SAT) {
      finalRes.sat = true;
      // Local helper: extract one BV/FP lane's concrete value into a
      // ModelVal. Pulled out so the vector branch below can reuse it
      // without duplicating the FP bit-pattern reconstruction.
      auto extractValue = [&](const smt::Term &term) -> Result::ModelVal {
        auto val_term = solver.get_value(term);
        if (solver.is_fp_sort(solver.get_sort(term))) {
          std::string bin = solver.get_fp_value_string(val_term);
          uint64_t bits = 0;
          for (char c: bin)
            bits = (bits << 1) | (c - '0');
          double d;
          if (bin.size() <= 32) {
            uint32_t b32 = (uint32_t) bits;
            float f;
            std::memcpy(&f, &b32, sizeof(f));
            d = f;
          } else {
            std::memcpy(&d, &bits, sizeof(d));
          }
          return d;
        } else {
          auto val_str = solver.get_bv_value_string(val_term, 10);
          uint64_t uraw = std::stoull(val_str, nullptr, 10);
          uint32_t width = solver.get_bv_width(solver.get_sort(term));
          int64_t raw;
          if (width < 64) {
            uint64_t mask = (uint64_t(1) << width) - 1;
            uraw &= mask;
            if (uraw >= (uint64_t(1) << (width - 1)))
              raw = static_cast<int64_t>(uraw) - static_cast<int64_t>(mask + 1);
            else
              raw = static_cast<int64_t>(uraw);
          } else {
            raw = static_cast<int64_t>(uraw);
          }
          return raw;
        }
      };

      for (const auto &s: entry->syms) {
        const auto &sv = store.at(s.name.name);
        // Vector sym: extract one model value per lane.
        if (sv.kind == SymbolicValue::Kind::Vec) {
          std::vector<Result::ModelVal> lanes;
          lanes.reserve(sv.arrayVal.size());
          for (const auto &lane: sv.arrayVal)
            lanes.push_back(extractValue(lane.term));
          finalRes.vecModel[s.name.name] = std::move(lanes);
          continue;
        }
        if (sv.kind == SymbolicValue::Kind::Tensor) {
          std::vector<Result::ModelVal> leaves;
          leaves.reserve(sv.arrayVal.size());
          for (const auto &leaf: sv.arrayVal)
            leaves.push_back(extractValue(leaf.term));
          finalRes.tensorModel[s.name.name] = std::move(leaves);
          continue;
        }
        finalRes.model[s.name.name] = extractValue(sv.term);
      }

      // Solved values for entry-fun params. We only handle
      // scalar Int/Float here — aggregate / pointer / vector params
      // are out of scope for the SOLVED header (printer leaves them
      // verbatim).
      for (const auto &p: entry->params) {
        auto it = store.find(p.name.name);
        if (it == store.end())
          continue;
        const auto &sv = it->second;
        if (sv.kind == SymbolicValue::Kind::Int && sv.term.internal)
          finalRes.paramModel[p.name.name] = extractValue(sv.term);
      }

      // Tensor parameters: the declared type carries the logical shape and
      // element kind, the symbolic store carries one leaf per logical
      // element, and extraction is the same one every other model value
      // uses. Leaves stay witnesses of a runtime argument; nothing here
      // substitutes them into the program.
      for (const auto &p: entry->params) {
        const auto *tensor = TypeUtils::asTensor(p.type);
        if (!tensor)
          continue;
        auto it = store.find(p.name.name);
        if (it == store.end() || it->second.kind != SymbolicValue::Kind::Tensor)
          continue;
        Result::TensorParamWitness witness;
        witness.shape = tensor->shape;
        witness.element = std::holds_alternative<FloatType>(tensor->elem->v)
                              ? TensorElement::F32
                              : TensorElement::I32;
        witness.leaves.reserve(it->second.arrayVal.size());
        for (const auto &leaf: it->second.arrayVal)
          witness.leaves.push_back(extractValue(leaf.term));
        finalRes.tensorParamModel.emplace(p.name.name, std::move(witness));
      }
      if (retTermLocal.internal)
        finalRes.retModel = extractValue(retTermLocal);

      // Map an exit-time address tag back to the entry-fun local/param
      // that *contains* it, returning {name, base tag}. The forward map
      // (`tagOfLocal`) is FNV-1a and every cell of an aggregate gets
      // `base + offset`, so a whole-object pointer hashes to a local's
      // bare tag while a sub-object pointer — e.g. `addr %a[2].f0`, whose
      // prov_base/address land *inside* the aggregate — does not. An
      // exact-match lookup misses the sub-object case (the prov_base then
      // resolves to no local, and the consumer silently falls back to the
      // entry-init, dropping body retargets). Range containment maps the
      // offset address back to its root object; the in-object delta is
      // recovered as `addr - baseTag`. Returns {"", 0} when no object
      // contains the tag — cross-object arithmetic, undef pointer, etc.
      auto reverseLookupContaining = [&](uint64_t tag) -> std::pair<std::string, uint64_t> {
        auto contains = [&](const std::string &nm, const TypePtr &ty) -> bool {
          uint64_t base = tagOfLocal(kEntryFrame, nm);
          uint64_t size = sizeofTagUnits(ty, structs_);
          return base <= tag && tag < base + size;
        };
        for (const auto &l: entry->lets)
          if (contains(l.name.name, l.type))
            return {l.name.name, tagOfLocal(kEntryFrame, l.name.name)};
        for (const auto &p: entry->params)
          if (contains(p.name.name, p.type))
            return {p.name.name, tagOfLocal(kEntryFrame, p.name.name)};
        return {{}, 0};
      };

      // Concrete model value of every entry-function let at the end of
      // the solved path. The tree shape mirrors the declared type and
      // recurses through the matching SymbolicValue.arrayVal /
      // structVal that path execution built up. For pointer cells we
      // ask the solver for the concrete tag and reverse-FNV it back
      // to a local name so consumers can identify the pointee
      // symbolically rather than as a raw 64-bit address.
      std::function<LetExitValue(const SymbolicValue &, const TypePtr &)> extractLet =
          [&](const SymbolicValue &sv, const TypePtr &declType) -> LetExitValue {
        LetExitValue out;
        if (declType && std::holds_alternative<PtrType>(declType->v)) {
          out.kind = LetExitValue::Kind::Ptr;
          // `prov_base` names the underlying object (even after
          // ptrfield / ptrindex / pointer arithmetic shifted the
          // pointer inside it); `sv.term` is the actual exit-time
          // address. The delta is the in-object offset in packed bytes.
          if (sv.term.internal && sv.prov_base.internal) {
            try {
              auto addrTerm = solver.get_value(sv.term);
              auto addrStr = solver.get_bv_value_string(addrTerm, 10);
              uint64_t addrTag = std::stoull(addrStr, nullptr, 10);
              // Identify the pointee from the actual exit address, not the
              // prov_base: a sub-lvalue `addr` (`addr %a[2].f0`) leaves
              // prov_base offset into the aggregate, which no bare local
              // tag matches. The containing-object lookup recovers the
              // root local; the in-object delta is `addr - root base`.
              auto [name, rootTag] = reverseLookupContaining(addrTag);
              out.targetLocal = name;
              out.targetOffset = addrTag - rootTag; // wraps mod 2^64
            } catch (...) {
              // get_value may fail when the term escaped its solver
              // context or wasn't asserted on the SAT path. Leave
              // the pointer unresolved — the consumer's fallback
              // path will replay the entry-block addr-init.
            }
          }
          return out;
        }
        switch (sv.kind) {
          case SymbolicValue::Kind::Int: {
            if (!sv.term.internal) {
              out.kind = LetExitValue::Kind::Undef;
              return out;
            }
            bool isFp = declType && std::holds_alternative<FloatType>(declType->v);
            out.kind = isFp ? LetExitValue::Kind::Float : LetExitValue::Kind::Int;
            out.scalar = extractValue(sv.term);
            return out;
          }
          case SymbolicValue::Kind::Array: {
            out.kind = LetExitValue::Kind::Array;
            TypePtr elemTy;
            if (declType && std::holds_alternative<ArrayType>(declType->v))
              elemTy = std::get<ArrayType>(declType->v).elem;
            out.elems.reserve(sv.arrayVal.size());
            for (const auto &e: sv.arrayVal)
              out.elems.push_back(extractLet(e, elemTy));
            return out;
          }
          case SymbolicValue::Kind::Struct: {
            out.kind = LetExitValue::Kind::Struct;
            const StructDecl *sd = nullptr;
            if (declType && std::holds_alternative<StructType>(declType->v)) {
              const auto &sn = std::get<StructType>(declType->v).name.name;
              for (const auto &s: prog_.structs) {
                if (s.name.name == sn) {
                  sd = &s;
                  break;
                }
              }
            }
            for (const auto &[fname, fv]: sv.structVal) {
              TypePtr fieldTy;
              if (sd) {
                for (const auto &f: sd->fields) {
                  if (f.name == fname) {
                    fieldTy = f.type;
                    break;
                  }
                }
              }
              out.fields.emplace(fname, extractLet(fv, fieldTy));
            }
            return out;
          }
          case SymbolicValue::Kind::Vec: {
            out.kind = LetExitValue::Kind::Vec;
            TypePtr elemTy;
            if (declType && std::holds_alternative<VecType>(declType->v))
              elemTy = std::get<VecType>(declType->v).elem;
            out.elems.reserve(sv.arrayVal.size());
            for (const auto &e: sv.arrayVal)
              out.elems.push_back(extractLet(e, elemTy));
            return out;
          }
          case SymbolicValue::Kind::Tensor: {
            out.kind = LetExitValue::Kind::Tensor;
            out.elems.reserve(sv.arrayVal.size());
            const auto tensorType = declType ? std::get_if<TensorType>(&declType->v) : nullptr;
            const auto elemTy = tensorType ? tensorType->elem : TypePtr{};
            for (const auto &e: sv.arrayVal)
              out.elems.push_back(extractLet(e, elemTy));
            return out;
          }
          case SymbolicValue::Kind::Undef:
          default:
            out.kind = LetExitValue::Kind::Undef;
            return out;
        }
      };
      for (const auto &letd: entry->lets) {
        auto it = store.find(letd.name.name);
        if (it == store.end())
          continue;
        finalRes.letExitValues.emplace(letd.name.name, extractLet(it->second, letd.type));
      }
    } else if (res == smt::Result::UNSAT) {
      finalRes.unsat = true;
    } else {
      finalRes.unknown = true;
      // The backend's own report, kept as it stands: an undecided result with
      // no reported cause stays Unspecified rather than becoming a timeout.
      finalRes.unknownReason = solver.unknownReason();
    }
    return finalRes;
  }

  SymbolicExecutor::Result SymbolicExecutor::sample(
      const std::string &funcName, uint32_t n, uint32_t maxPathLen, bool requireTerminal,
      const std::vector<std::string> &prefixPath,
      const std::unordered_map<std::string, int64_t> &fixedSyms
  ) {
    const FunDecl *entry = nullptr;
    for (const auto &f: prog_.funs) {
      if (f.name.name == funcName) {
        entry = &f;
        break;
      }
    }
    if (!entry)
      throw std::runtime_error("Function not found: " + funcName);

    DiagBag diags;
    CFG cfg = CFG::build(*entry, diags);
    if (diags.hasErrors())
      throw std::runtime_error("CFG build failed");

    auto nextToRet = cfg.shortestPathToRet(*entry);

    // Determine number of threads
    uint32_t num_threads = config_.num_threads;
    if (num_threads == 0) {
      num_threads = std::thread::hardware_concurrency();
      if (num_threads == 0)
        num_threads = 1; // Fallback if hardware_concurrency returns 0
    }

    // Lambda to generate a random path and attempt to solve it
    // Returns optional Result: nullopt if path should be skipped, otherwise the solve result
    auto tryOneSample = [&](std::mt19937 &rng) -> std::optional<Result> {
      std::vector<std::string> path = prefixPath;
      if (path.empty()) {
        path.push_back(cfg.blocks[cfg.entry]);
      }

      std::size_t currentIdx = cfg.indexOf.at(path.back());
      bool terminated = std::holds_alternative<RetTerm>(entry->blocks[currentIdx].term) ||
                        std::holds_alternative<UnreachableTerm>(entry->blocks[currentIdx].term);

      // Random walk
      while (!terminated && path.size() < maxPathLen) {
        const auto &successors = cfg.succ[currentIdx];
        if (successors.empty())
          break;

        std::uniform_int_distribution<std::size_t> dist(0, successors.size() - 1);
        std::size_t nextIdx = successors[dist(rng)];
        path.push_back(cfg.blocks[nextIdx]);
        currentIdx = nextIdx;
        terminated = std::holds_alternative<RetTerm>(entry->blocks[currentIdx].term) ||
                     std::holds_alternative<UnreachableTerm>(entry->blocks[currentIdx].term);
      }

      // Handle non-terminated paths
      if (!terminated) {
        if (requireTerminal) {
          // Append shortest path to ret
          while (!std::holds_alternative<RetTerm>(entry->blocks[currentIdx].term)) {
            auto it = nextToRet.find(currentIdx);
            if (it == nextToRet.end()) {
              // Cannot reach ret from here - skip this sample
              return std::nullopt;
            }
            currentIdx = it->second;
            path.push_back(cfg.blocks[currentIdx]);
          }
        } else {
          // Discard if not terminated
          return std::nullopt;
        }
      }

      // Try to solve this path
      try {
        return solve(funcName, path, fixedSyms);
      } catch (const std::exception &e) {
        Result errRes;
        errRes.unknown = true;
        errRes.message = e.what();
        return errRes;
      }
    };

    // Single-threaded execution
    if (num_threads == 1) {
      std::mt19937 rng(config_.seed);
      Result lastRes;
      lastRes.unknown = true;

      for (uint32_t i = 0; i < n; ++i) {
        auto res = tryOneSample(rng);
        if (!res)
          continue; // Path was skipped

        if (res->sat)
          return *res;
        lastRes = std::move(*res);
      }

      return lastRes;
    }

    // Multi-threaded execution
    std::atomic<bool> found(false);
    std::atomic<uint32_t> samplesProcessed(0);
    std::mutex resultMutex;
    Result satResult;
    Result lastRes;
    lastRes.unknown = true;
    bool hasLastRes = false;

    auto workerFunc = [&](uint32_t threadId) {
      std::mt19937 rng(config_.seed + threadId);
      Result threadLastRes;
      bool produced = false;

      while (samplesProcessed.fetch_add(1) < n && !found.load()) {
        if (found.load())
          break;

        auto res = tryOneSample(rng);
        if (!res)
          continue; // Path was skipped

        if (res->sat) {
          std::lock_guard<std::mutex> lock(resultMutex);
          if (!found.load()) {
            satResult = std::move(*res);
            found.store(true);
          }
          return;
        }
        threadLastRes = std::move(*res);
        produced = true;
      }

      // A worker that never reached a query has nothing to publish: its
      // default result would replace another worker's reported reason.
      if (!produced)
        return;
      std::lock_guard<std::mutex> lock(resultMutex);
      // A reported reason survives a later worker that has none to give, so
      // the aggregate does not depend on which thread finished last.
      if (!hasLastRes || lastRes.unknownReason == smt::UnknownReason::Unspecified ||
          threadLastRes.unknownReason != smt::UnknownReason::Unspecified) {
        lastRes = std::move(threadLastRes);
        hasLastRes = true;
      }
    };

    // Launch worker threads
    std::vector<std::thread> threads;
    for (uint32_t i = 0; i < num_threads; ++i) {
      threads.emplace_back(workerFunc, i);
    }

    // Wait for all threads to complete
    for (auto &t: threads) {
      t.join();
    }

    if (found.load()) {
      return satResult;
    }

    return lastRes;
  }

} // namespace refractir

namespace refractir::smt {

  UnknownReason classifyUnknownReason(std::string_view diagnostic) {
    // The complete diagnostics Z3 hands back through the wrapper. This is a
    // closed set on purpose: a report has to say which cause it is, and a text
    // that merely mentions a word does not, so "unknown timeout configuration"
    // and "not interrupted" stay unspecified rather than becoming a cause the
    // backend never reported.
    if (diagnostic == "timeout")
      return UnknownReason::Timeout;
    if (diagnostic == "canceled" || diagnostic == "cancelled" || diagnostic == "interrupted")
      return UnknownReason::Interrupted;
    if (diagnostic == "resource limits" || diagnostic == "max. memory" ||
        diagnostic == "max. memory out" || diagnostic == "memout")
      return UnknownReason::ResourceLimit;
    return UnknownReason::Unspecified;
  }

  std::string_view unknownReasonName(UnknownReason reason) {
    switch (reason) {
      case UnknownReason::Unspecified:
        return "unspecified";
      case UnknownReason::Timeout:
        return "timeout";
      case UnknownReason::ResourceLimit:
        return "resource_limit";
      case UnknownReason::Interrupted:
        return "interrupted";
    }
    return "unspecified";
  }

} // namespace refractir::smt
