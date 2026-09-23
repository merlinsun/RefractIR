#include "reify/twin_transform.hpp"
#include "ast/build.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <optional>
#include <ostream>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cstdio>
#include <cstring>
#include "analysis/cfg.hpp"
#include "analysis/definite_init.hpp"
#include "analysis/dominators.hpp"
#include "analysis/type_utils.hpp"

#include "ast/ast.hpp"
#include "ast/clone.hpp"
#include "frontend/diagnostics.hpp"
#include "interp/interpreter.hpp"
#include "interp/type_layout.hpp"
#include "reify/antiopt.hpp"
#include "reify/expr_gen.hpp"
#include "reify/hyperparameters.hpp"
#include "reify/state_profile.hpp"
#include "reify/twin_box.hpp"
#include "reify/twin_mini.hpp"
#include "reify/twin_probe.hpp"
#include "reify/twin_trace.hpp"
#include "reify/type_gen.hpp"

namespace refractir::reify {

  namespace {

    // --- small AST builders (mirroring func_gen's helpers) ---------------

    TypePtr makePtr(TypePtr pointee) {
      return std::make_shared<Type>(Type{PtrType{std::move(pointee), {}}, {}});
    }

    Instr buildAssign(const std::string &lhs, Expr rhs) {
      return Instr{AssignInstr{buildLValue(lhs), std::move(rhs), {}}};
    }

    // `%dst = cmp == %a, %b` — both operands are same-typed locals, so the
    // equality is exact without any width coercion.
    Instr
    cmpRelInstr(const std::string &dst, const std::string &a, RelOp op, const std::string &b) {
      CmpAtom c;
      c.op = op;
      c.lhs = SelectVal{RValue{buildLValue(a)}};
      c.rhs = SelectVal{RValue{buildLValue(b)}};
      return buildAssign(dst, buildExpr(Atom{std::move(c), {}}));
    }

    Instr cmpEqInstr(const std::string &dst, const std::string &a, const std::string &b) {
      CmpAtom c;
      c.op = RelOp::EQ;
      c.lhs = SelectVal{RValue{buildLValue(a)}};
      c.rhs = SelectVal{RValue{buildLValue(b)}};
      return buildAssign(dst, buildExpr(Atom{std::move(c), {}}));
    }

    // `%dst = %left <op> %right` — the left operand is an id (coef), the
    // right an lvalue, matching OpAtom's shape. The type checker requires
    // both operands share bit-width, so callers keep them same-typed.
    Instr opInstr(
        const std::string &dst, const std::string &left, AtomOpKind op, const std::string &right
    ) {
      OpAtom o;
      o.op = op;
      o.coef = Coef{LocalOrSymId{LocalId{left, {}}}};
      o.rval = buildLValue(right);
      return buildAssign(dst, buildExpr(Atom{std::move(o), {}}));
    }

    Instr andInstr(const std::string &acc, const std::string &cur) {
      return opInstr(acc, acc, AtomOpKind::And, cur);
    }

    Terminator brTo(const std::string &dest) {
      BrTerm b;
      b.dest = BlockLabel{dest, {}};
      b.thenLabel = b.dest;
      b.elseLabel = b.dest;
      b.isConditional = false;
      return Terminator{std::move(b)};
    }

    // `br <cond-expr> != 0, ^then, ^else` — the guard call site.
    Terminator brIfExpr(Expr cond, const std::string &thenL, const std::string &elseL) {
      Cond c;
      c.lhs = std::move(cond);
      c.op = RelOp::NE;
      c.rhs = buildExpr(buildCoefAtom(Coef{IntLit{0, {}}}));
      BrTerm b;
      b.cond = std::move(c);
      b.thenLabel = BlockLabel{thenL, {}};
      b.elseLabel = BlockLabel{elseL, {}};
      b.dest = b.thenLabel;
      b.isConditional = true;
      return Terminator{std::move(b)};
    }

    // --- read-set scanning ----------------------------------------------
    //
    // Collect the base names an expression *reads* and whether it touches
    // memory (load / addr / ptr navigation; stores are flagged by the
    // instruction scan). Memory-op blocks are twin candidates: their effect
    // is the frame-state diff, and the full-state guard pins every value a
    // load can observe — planRegion therefore requires the WHOLE state to be
    // guardable for such blocks. Only non-intrinsic calls still reject: a
    // callee given a pointer into an outer frame could mutate state this
    // frame's diff does not see.

    struct ReadScan {
      std::unordered_set<std::string> reads;
      bool mem = false;
    };

    void scanIndices(const LValue &lv, ReadScan &rs) {
      for (const auto &acc: lv.accesses)
        if (auto ai = std::get_if<AccessIndex>(&acc))
          if (auto id = std::get_if<LocalOrSymId>(&ai->index))
            std::visit([&](auto &&v) { rs.reads.insert(v.name); }, *id);
    }

    void scanLV(const LValue &lv, ReadScan &rs) {
      rs.reads.insert(lv.base.name);
      scanIndices(lv, rs);
    }

    void scanCoef(const Coef &c, ReadScan &rs) {
      if (auto id = std::get_if<LocalOrSymId>(&c))
        std::visit([&](auto &&v) { rs.reads.insert(v.name); }, *id);
    }

    void scanSelectVal(const SelectVal &sv, ReadScan &rs) {
      if (auto rv = std::get_if<RValue>(&sv))
        scanLV(*rv, rs);
      else if (auto co = std::get_if<Coef>(&sv))
        scanCoef(*co, rs);
    }

    bool scanAtom(const Atom &a, ReadScan &rs);

    bool scanExpr(const Expr &e, ReadScan &rs) {
      if (!scanAtom(e.first, rs))
        return false;
      for (const auto &t: e.rest)
        if (!scanAtom(t.atom, rs))
          return false;
      return true;
    }

    bool scanCond(const Cond &c, ReadScan &rs) {
      return scanExpr(c.lhs, rs) && scanExpr(c.rhs, rs);
    }

    bool scanAtom(const Atom &a, ReadScan &rs) {
      return std::visit(
          [&](auto &&arg) -> bool {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, CoefAtom>) {
              scanCoef(arg.coef, rs);
              return true;
            } else if constexpr (std::is_same_v<T, RValueAtom>) {
              scanLV(arg.rval, rs);
              return true;
            } else if constexpr (std::is_same_v<T, OpAtom>) {
              scanCoef(arg.coef, rs);
              scanLV(arg.rval, rs);
              return true;
            } else if constexpr (std::is_same_v<T, UnaryAtom>) {
              scanLV(arg.rval, rs);
              return true;
            } else if constexpr (std::is_same_v<T, CmpAtom>) {
              scanSelectVal(arg.lhs, rs);
              scanSelectVal(arg.rhs, rs);
              return true;
            } else if constexpr (std::is_same_v<T, CastAtom>) {
              if (auto lv = std::get_if<LValue>(&arg.src))
                scanLV(*lv, rs);
              else if (auto si = std::get_if<SymId>(&arg.src))
                rs.reads.insert(si->name);
              return true;
            } else if constexpr (std::is_same_v<T, SelectAtom>) {
              if (arg.cond && !scanCond(*arg.cond, rs))
                return false;
              if (arg.maskExpr && !scanExpr(*arg.maskExpr, rs))
                return false;
              scanSelectVal(arg.vtrue, rs);
              scanSelectVal(arg.vfalse, rs);
              return true;
            } else if constexpr (std::is_same_v<T, CallAtom>) {
              // Intrinsic calls are pure value functions; only their
              // argument reads matter. A non-intrinsic call could have
              // out-of-frame effects, so reject it.
              if (!arg.resolvedIntrinsic)
                return false;
              for (const auto &e: arg.args)
                if (e && !scanExpr(*e, rs))
                  return false;
              return true;
            } else if constexpr (std::is_same_v<T, AddrAtom>) {
              rs.mem = true;
              scanIndices(arg.lv, rs); // the address itself is state-free
              return true;
            } else if constexpr (std::is_same_v<T, LoadAtom>) {
              rs.mem = true;
              scanLV(arg.rval, rs);
              return true;
            } else if constexpr (std::is_same_v<T, PtrIndexAtom>) {
              rs.mem = true;
              scanLV(arg.rval, rs);
              if (auto id = std::get_if<LocalOrSymId>(&arg.index))
                std::visit([&](auto &&v) { rs.reads.insert(v.name); }, *id);
              return true;
            } else if constexpr (std::is_same_v<T, TensorAtom>) {
              for (const auto &operand: arg.operands)
                scanCoef(operand, rs);
              return true;
            } else {
              static_assert(std::is_same_v<T, PtrFieldAtom>);
              rs.mem = true;
              scanLV(arg.rval, rs);
              return true;
            }
          },
          a.v
      );
    }

    // --- profile value-tree helpers -------------------------------------

    using StateMap = std::unordered_map<std::string, const StateValue *>;

    StateMap toStateMap(const std::vector<std::pair<std::string, StateValue>> &vars) {
      StateMap m;
      for (const auto &kv: vars)
        m[kv.first] = &kv.second;
      return m;
    }

    // Navigate a value tree along `accs` (concrete accesses). Returns nullptr
    // if the shape doesn't match (e.g. field/index missing).
    const StateValue *navigate(const StateValue &root, const std::vector<Access> &accs) {
      const StateValue *cur = &root;
      for (const auto &acc: accs) {
        if (auto af = std::get_if<AccessField>(&acc)) {
          if (cur->kind != StateValue::Kind::Struct)
            return nullptr;
          const StateValue *next = nullptr;
          for (const auto &f: cur->fields)
            if (f.first == af->field) {
              next = &f.second;
              break;
            }
          if (!next)
            return nullptr;
          cur = next;
        } else {
          const auto &ai = std::get<AccessIndex>(acc);
          auto il = std::get_if<IntLit>(&ai.index);
          if (!il)
            return nullptr; // must be concrete by now
          if (cur->kind != StateValue::Kind::Array && cur->kind != StateValue::Kind::Vec)
            return nullptr;
          if (il->value < 0 || (std::size_t) il->value >= cur->elems.size())
            return nullptr;
          cur = &cur->elems[il->value];
        }
      }
      return cur;
    }

    // Canonical key for a (root, path) leaf so repeated writes dedup.
    struct LeafRef {
      std::string root;
      std::vector<Access> path;
      StateValue val;
      // Ptr leaves only: the static `ptr T` type of the cell, and the
      // lvalue whose address reproduces the captured pointer (unset for
      // null pointers).
      TypePtr ptrType;
      std::optional<LValue> ptrTarget;

      LValue lvalue() const { return LValue{LocalId{root, {}}, path, {}}; }

      bool isPtr() const { return val.kind == StateValue::Kind::Ptr; }
    };

    // The RHS that reproduces a pointer leaf: `addr <target>` or `null`.
    Atom ptrRhsAtom(const LeafRef &leaf) {
      if (leaf.ptrTarget)
        return buildAddrAtom(*leaf.ptrTarget);
      return buildCoefAtom(Coef{NullLit{}});
    }

    // --- twin planning ----------------------------------------------------

    // One state root the guard consumes, with its crossing strategy.
    struct GuardRoot {
      enum class Kind {
        Scalar, // by-value parameter of the root's own type
        Vec,    // one by-value parameter per lane
        Agg,    // by-address parameter (ptr [N] T / ptr @S), navigated inside
        Ptr,    // by-value ptr parameter, compared against an expected addr
      };
      std::string name;
      TypePtr type; // the root's static type in the entry function
      Kind kind;
      bool isParam = false;        // immutable in the entry function
      std::vector<LeafRef> leaves; // expected values from `s`
    };

    struct TwinPlan {
      std::vector<GuardRoot> guardRoots;     // the entire guardable state, in
                                             // profile (name-sorted) order
      std::vector<LeafRef> defs;             // per-leaf constant reconstruction of s'
      TraceBody body;                        // the region's flattened executed trace
      EntryState entry;                      // the profiled state, as the box's floor
      Box box;                               // how far the guard may open, per leaf
      std::vector<std::string> regionLabels; // the region's blocks, entry first
      std::vector<MiniRoot> roots;           // the profiled state, ready to re-run
      std::string exitLabel;                 // block the twin jumps to (region exit)
    };

    // Locate a root's declaration in the entry function.
    struct RootDecl {
      TypePtr type;
      bool isParam = false;
      bool isMutable = false;
      bool initialized = false; // param, or let with a non-undef declared init
    };

    std::optional<RootDecl> findRoot(const FunDecl &fn, const std::string &name) {
      for (const auto &p: fn.params)
        if (p.name.name == name)
          return RootDecl{p.type, true, false, true};
      for (const auto &l: fn.lets)
        if (l.name.name == name)
          return RootDecl{
              l.type, false, l.isMutable, l.init.has_value() && l.init->kind != InitVal::Kind::Undef
          };
      return std::nullopt;
    }

    // Fill a ptr leaf's static type and reconstruction target. Returns
    // false when the leaf cannot be reproduced or guarded; the cell it names
    // is recovered by reify/state_profile.hpp, which owns what a recorded
    // pointer means, and the null pointer is reproducible here even though it
    // names no cell.
    bool fillPtrLeaf(
        LeafRef &leaf, const FunDecl &fn, const StructMap &structs, const TypeLayout &layout,
        const TypePtr &rootType, const char **why = nullptr
    ) {
      TypePtr t = TypeUtils::accessPathType(rootType, leaf.path, structs);
      if (!t) {
        if (why)
          *why = "a leaf whose type does not follow the root's";
        return false;
      }
      if (!TypeUtils::isPtr(t)) {
        if (why)
          *why = "a pointer leaf of non-pointer type";
        return false;
      }
      leaf.ptrType = t;
      if (leaf.val.ptrNull)
        return true;
      auto target = resolvePointee(leaf.val, t, fn, structs, layout, true, why);
      if (!target)
        return false;
      leaf.ptrTarget = std::move(*target);
      return true;
    }

    // Collect the roots a terminator reads (branch condition / return value).
    // A region's twin skips its intermediate terminators, so their reads must
    // be guardable too; a single block's own terminator is preserved by the
    // graft, so Block scope does not scan it.
    bool scanTerm(const Terminator &t, ReadScan &rs) {
      if (auto br = std::get_if<BrTerm>(&t)) {
        if (br->isConditional && br->cond)
          return scanCond(*br->cond, rs);
        return true;
      }
      if (auto rt = std::get_if<RetTerm>(&t))
        return !rt->value || scanExpr(*rt->value, rs);
      return true; // UnreachableTerm reads nothing
    }

    // A region (one or more executed blocks with a single dominating entry)
    // is a twin candidate iff every block's instructions are free of
    // non-intrinsic calls, the whole read set is guardable, and the region's
    // net effect (the bit-exact state diff s -> s') is reproducible: scalar
    // leaves as literals, pointer leaves as `addr <target>` / `null`.
    // Memory-op regions additionally require the ENTIRE state to be guardable
    // — a load can observe any root through a pointer. `blocks` are the
    // skipped blocks (entry first); `scanTerms` also scans their terminators
    // (region scope, where the twin bypasses them). Fills `plan` with the
    // guard roots (all definitely-initialized state at entry, compared to
    // `s`) and the def leaves (the diff, from `sPrime` at the region exit).
    // Which locals the checker can see are initialized on entry to each
    // block: computed once per function, since it is a fixpoint over the CFG.
    using InitAtEntry = std::unordered_map<std::string, std::unordered_set<std::string>>;

    // The per-function analysis every planning step reads and none of them
    // changes. All of it is computed once for the function, so threading it as
    // one reference keeps the planners' own arguments to what actually varies
    // between candidates.
    struct PlanContext {
      const FunDecl &fn;
      const CFG &cfg;
      const DomTree &dt;
      const std::unordered_map<std::string, const Block *> &byLabel;
      const StructMap &structs;
      const TypeLayout &layout;
      const InitAtEntry &inited;
    };

    bool planRegion(
        const PlanContext &ctx, const std::vector<const Block *> &blocks, bool scanTerms,
        const std::vector<std::pair<std::string, StateValue>> &sVars,
        const std::vector<std::pair<std::string, StateValue>> &sPrimeVars, const StateMap &s,
        TwinPlan &plan, std::string *why = nullptr
    ) {
      const Block &b = *blocks.front(); // the region entry
      auto reject = [&](std::string r) {
        if (why)
          *why = std::move(r);
        return false;
      };
      // What the guard may read here. The guard call is spliced at the start
      // of this block and takes the live state as arguments, so a root it
      // reads has to be one the *checker* can see is initialized at this
      // point — otherwise the emitted program fails its own re-analysis. That
      // is a question the frontend's must-init analysis already answers, and
      // asking it is what admits a root assigned anywhere that dominates the
      // region rather than only in the function's entry block.
      auto initedHere = ctx.inited.find(b.label.name);
      auto isInitialized = [&](const std::string &nm) {
        return initedHere != ctx.inited.end() && initedHere->second.count(nm) > 0;
      };

      ReadScan rs;
      for (const Block *bp: blocks) {
        for (const auto &ins: bp->instrs) {
          bool ok = std::visit(
              [&](auto &&i) -> bool {
                using T = std::decay_t<decltype(i)>;
                if constexpr (std::is_same_v<T, AssignInstr>) {
                  if (!scanExpr(i.rhs, rs))
                    return false;
                  scanIndices(i.lhs, rs); // indices are read; the base is written
                  return true;
                } else if constexpr (std::is_same_v<T, AssumeInstr>) {
                  return scanCond(i.cond, rs);
                } else if constexpr (std::is_same_v<T, RequireInstr>) {
                  return scanCond(i.cond, rs);
                } else {
                  static_assert(std::is_same_v<T, StoreInstr>);
                  rs.mem = true;
                  return scanExpr(i.ptr, rs) && scanExpr(i.val, rs);
                }
              },
              ins
          );
          if (!ok)
            return reject("non-intrinsic call in " + bp->label.name);
        }
        if (scanTerms && !scanTerm(bp->term, rs))
          return reject("non-intrinsic call in terminator of " + bp->label.name);
      }

      // Guard roots: every definitely-initialized root of the entry state.
      // A root that cannot cross into the guard (undef leaves, opaque
      // pointers, immutable aggregate, vector nested in an aggregate) is
      // skipped when the block neither reads it nor touches memory —
      // soundness needs guard-set ⊇ read-set, and memory ops can read any
      // root — and rejects the block otherwise.
      std::unordered_set<std::string> guarded;
      for (const auto &[name, val]: sVars) {
        auto decl = findRoot(ctx.fn, name);
        // Why this root cannot cross into the guard, for the rejection
        // message: the name alone leaves a reader to guess which of half a
        // dozen reasons applied, and none of them is visible in the program.
        const char *why = nullptr;
        bool guardable = decl.has_value() && (decl->initialized || isInitialized(name));
        if (!guardable)
          why = decl.has_value() ? "not initialized at region entry" : "no declaration";
        GuardRoot::Kind kind = GuardRoot::Kind::Scalar;
        if (guardable) {
          if (TypeUtils::isPtr(decl->type))
            kind = GuardRoot::Kind::Ptr;
          else if (std::holds_alternative<VecType>(decl->type->v))
            kind = GuardRoot::Kind::Vec;
          else if (TypeUtils::asArray(decl->type) || TypeUtils::asStruct(decl->type)) {
            kind = GuardRoot::Kind::Agg;
            // `addr %root` needs a mutable root; vector lanes inside an
            // aggregate cannot be reached through a pointer at all.
            if (!decl->isMutable) {
              guardable = false;
              why = "an immutable aggregate, which `addr` cannot take";
            } else if (TypeUtils::containsVec(decl->type, ctx.structs)) {
              guardable = false;
              why = "a vector inside an aggregate, which no pointer reaches";
            }
          }
        }
        std::vector<StateLeaf> leaves;
        if (guardable) {
          bool hasPtr = false, hasUndef = false;
          enumStateLeaves(val, leaves, hasPtr, hasUndef);
          if (hasUndef) {
            guardable = false;
            why = "an undef leaf, which no guard can state";
          } else if (leaves.empty()) {
            guardable = false;
            why = "no leaves to compare";
          }
        }
        GuardRoot root;
        if (guardable) {
          root = GuardRoot{name, decl->type, kind, decl->isParam, {}};
          for (auto &lf: leaves) {
            LeafRef ref{name, std::move(lf.path), lf.val, {}, {}};
            if (ref.isPtr() &&
                !fillPtrLeaf(ref, ctx.fn, ctx.structs, ctx.layout, decl->type, &why)) {
              guardable = false;
              break;
            }
            root.leaves.push_back(std::move(ref));
          }
        }
        if (!guardable) {
          if (rs.reads.count(name) || rs.mem)
            // the region depends on state we cannot pin in the guard
            return reject(
                "unguardable state: " + name + " (" + (why ? why : "unknown reason") + ")"
            );
          continue;
        }
        plan.guardRoots.push_back(std::move(root));
      }
      if (plan.guardRoots.empty())
        return reject("no guardable live-in state");

      // Defs: the bit-exact diff s -> s' over every root, including roots
      // that first become initialized inside the region. This subsumes
      // write-site analysis: store-through-pointer effects surface as
      // diffs of the pointee root. The diff decides eligibility and scores
      // the region; it no longer has to be *reproducible*, since the twin
      // body replays the region's statements rather than rebuilding the
      // leaves they wrote.
      std::map<std::string, LeafRef> defMap;
      for (const auto &[name, val]: sPrimeVars) {
        auto decl = findRoot(ctx.fn, name);
        if (!decl)
          return reject("unknown exit root: " + name);
        std::vector<StateLeaf> leaves;
        bool hasPtr = false, hasUndef = false;
        enumStateLeaves(val, leaves, hasPtr, hasUndef);
        const StateValue *before = nullptr;
        if (auto it = s.find(name); it != s.end())
          before = it->second;
        for (auto &lf: leaves) {
          const StateValue *old = before ? navigate(*before, lf.path) : nullptr;
          if (old && bitExactEq(*old, lf.val))
            continue;
          if (decl->isParam || !decl->isMutable)
            return reject("immutable root changed: " + name);
          LeafRef ref{name, std::move(lf.path), lf.val, {}, {}};
          std::string key = leafKey(name, ref.path);
          defMap[key] = std::move(ref);
        }
      }
      if (defMap.empty())
        return reject("no state change over the region");
      for (auto &[k, v]: defMap)
        plan.defs.push_back(std::move(v));
      return true;
    }

    // --- guard-function synthesis ------------------------------------------

    std::string vecLaneParam(const std::string &root, std::int64_t lane) {
      return root + "__l" + std::to_string(lane);
    }

    std::int64_t laneOf(const LeafRef &leaf) {
      return std::get<IntLit>(std::get<AccessIndex>(leaf.path.front()).index).value;
    }

    // Build `fun @<name>(<state>) : i1` — a total, collision-free equality
    // check of the crossing state against the plan's expected values. Scalars
    // arrive by value, vectors per-lane, aggregates by address (navigated
    // with in-bounds ptrindex/ptrfield chains + load, so the body is UB-free
    // on EVERY input, matched or not).
    // A guard takes no parameter it never looks at. A free leaf is dropped
    // wherever the operand it needs is dropped with it: a scalar root becomes
    // its own operand, a vector lane has its own parameter, and an aggregate
    // is passed by address for all of its leaves at once — so an aggregate
    // only leaves the signature when every one of its leaves is free.
    bool leafIsFree(const Box &b, const struct GuardRoot &root, const struct LeafRef &leaf);
    bool rootIsFullyFree(const Box &b, const struct GuardRoot &root);

    LeafClass classOf(const Box &b, const std::string &key) {
      for (const auto &l: b.leaves)
        if (l.key == key)
          return l.cls;
      return LeafClass::Pinned;
    }

    Interval rangeOf(const Box &b, const std::string &key) {
      for (const auto &l: b.leaves)
        if (l.key == key)
          return l.range;
      return Interval{};
    }

    bool leafIsFree(const Box &b, const GuardRoot &root, const LeafRef &leaf) {
      // A pointer leaf is compared against a reconstructed expected pointer
      // and is never classified, so it is never free.
      return !leaf.isPtr() && classOf(b, leafKey(root.name, leaf.path)) == LeafClass::Free;
    }

    bool rootIsFullyFree(const Box &b, const GuardRoot &root) {
      if (root.leaves.empty() || root.kind == GuardRoot::Kind::Ptr)
        return false;
      for (const auto &leaf: root.leaves)
        if (!leafIsFree(b, root, leaf))
          return false;
      return true;
    }

    // Assembles the guard function for one plan. The parameter list, the lets
    // and the scratch pools are all built while walking the same leaves, so they
    // share state here instead of being threaded through arguments.
    struct GuardFunBuilder {
      GuardFunBuilder(const TwinPlan &p, const StructMap &s) : plan(p), structs(s) {}

      const TwinPlan &plan;
      const StructMap &structs;
      FunDecl g;
      Block e;
      std::vector<std::pair<TypePtr, std::string>> ptrScratch, loadScratch;
      int eIdx = 0;
      int kIdx = 0;

      void addLet(const std::string &nm, TypePtr ty, InitVal iv, bool mut) {
        LetDecl d;
        d.isMutable = mut;
        d.name = LocalId{nm, {}};
        d.type = std::move(ty);
        d.init = std::move(iv);
        g.lets.push_back(std::move(d));
      }

      static InitVal intInit(std::int64_t v) {
        return InitVal{InitVal::Kind::Int, IntLit{v, {}}, {}};
      }

      InitVal zeroInit(const TypePtr &ty) {
        if (TypeUtils::isPtr(ty))
          return InitVal{InitVal::Kind::Undef, IntLit{0, {}}, {}};
        if (TypeUtils::getFloatBitWidth(ty))
          return InitVal{InitVal::Kind::Float, FloatLit{0.0, {}}, {}};
        return intInit(0);
      }

      InitVal litInit(const StateValue &v) {
        if (v.kind == StateValue::Kind::Float)
          return InitVal{InitVal::Kind::Float, FloatLit{v.floatVal, {}}, {}};
        return intInit(v.intVal);
      }

      std::string getScratch(
          std::vector<std::pair<TypePtr, std::string>> &pool, const TypePtr &ty, const char *prefix,
          bool isPtr
      ) {
        for (const auto &[t, nm]: pool)
          if (TypeUtils::areTypesEqual(t, ty))
            return nm;
        std::string nm = std::string(prefix) + std::to_string(pool.size());
        pool.emplace_back(ty, nm);
        InitVal iv;
        if (isPtr)
          iv = InitVal{InitVal::Kind::Undef, IntLit{0, {}}, {}};
        else
          iv = zeroInit(ty);
        addLet(nm, ty, std::move(iv), /*mut=*/true);
        return nm;
      }

      void appendParams() {

        // Params, in guard-root order (the caller emits args the same way).
        // After each root's main parameter(s) come one expected-pointer
        // parameter per ptr leaf (`%__e<n>`): the caller reconstructs the
        // expected pointer with `addr` / `null` and the guard compares with
        // `==`, which is defined even across objects.
        for (const auto &root: plan.guardRoots) {
          if (rootIsFullyFree(plan.box, root))
            continue;
          switch (root.kind) {
            case GuardRoot::Kind::Scalar:
            case GuardRoot::Kind::Ptr:
              g.params.push_back({LocalId{root.name, {}}, root.type, {}});
              break;
            case GuardRoot::Kind::Vec: {
              const auto &vt = std::get<VecType>(root.type->v);
              for (const auto &leaf: root.leaves)
                if (!leafIsFree(plan.box, root, leaf))
                  g.params.push_back(
                      {LocalId{vecLaneParam(root.name, laneOf(leaf)), {}}, vt.elem, {}}
                  );
              break;
            }
            case GuardRoot::Kind::Agg:
              g.params.push_back({LocalId{root.name, {}}, makePtr(root.type), {}});
              break;
          }
          for (const auto &leaf: root.leaves)
            if (leaf.isPtr())
              g.params.push_back({LocalId{"%__e" + std::to_string(eIdx++), {}}, leaf.ptrType, {}});
        }
      }

      // Walk `root` down to `leaf`'s cell with ptrindex/ptrfield and load it.
      // The constant indices come from the state tree, which mirrors the static
      // type, so every step is in-bounds on any input. Returns the local
      // holding the loaded value and reports its type through `leafT`.
      std::string emitAggLeafLoad(const GuardRoot &root, const LeafRef &leaf, TypePtr &leafT) {
        std::string cur = root.name;
        TypePtr curT = root.type; // pointee of `cur`
        for (const auto &acc: leaf.path) {
          TypePtr nextT = TypeUtils::stepType(curT, acc, structs);
          std::string nxt = getScratch(ptrScratch, makePtr(nextT), "%__p", true);
          Atom nav =
              std::holds_alternative<AccessField>(acc)
                  ? Atom{PtrFieldAtom{buildLValue(cur), std::get<AccessField>(acc).field, {}}, {}}
                  : Atom{
                        PtrIndexAtom{
                            buildLValue(cur),
                            Index{std::get<IntLit>(std::get<AccessIndex>(acc).index)},
                            {}
                        },
                        {}
                    };
          e.instrs.push_back(buildAssign(nxt, buildExpr(std::move(nav))));
          cur = nxt;
          curT = nextT;
        }
        std::string operand = getScratch(loadScratch, curT, "%__v", false);
        e.instrs.push_back(
            buildAssign(operand, buildExpr(Atom{LoadAtom{buildLValue(cur), {}}, {}}))
        );
        leafT = curT;
        return operand;
      }

      void emitLeafChecks() {
        for (const auto &root: plan.guardRoots) {
          if (rootIsFullyFree(plan.box, root))
            continue;
          for (const auto &leaf: root.leaves) {
            // Decided before the operand is built: a free leaf needs no lane, no
            // navigation and no load, so none are emitted for it.
            if (leafIsFree(plan.box, root, leaf))
              continue;
            std::string operand;
            TypePtr leafT;
            switch (root.kind) {
              case GuardRoot::Kind::Scalar:
              case GuardRoot::Kind::Ptr:
                operand = root.name;
                leafT = root.type;
                break;
              case GuardRoot::Kind::Vec:
                operand = vecLaneParam(root.name, laneOf(leaf));
                leafT = std::get<VecType>(root.type->v).elem;
                break;
              case GuardRoot::Kind::Agg:
                operand = emitAggLeafLoad(root, leaf, leafT);
                break;
            }
            if (leaf.isPtr()) {
              // Pointer equality against the caller-reconstructed expected
              // pointer (defined across objects, so total on every input).
              e.instrs.push_back(cmpEqInstr("%__c", operand, "%__e" + std::to_string(eIdx++)));
              e.instrs.push_back(andInstr("%__acc", "%__c"));
              continue;
            }
            // What the box proved about this leaf decides what the guard says
            // about it. Comparisons never trap, so every form stays total.
            const LeafClass cls = classOf(plan.box, leafKey(root.name, leaf.path));
            if (cls == LeafClass::Ranged) {
              const Interval r = rangeOf(plan.box, leafKey(root.name, leaf.path));
              std::string klo = "%__k" + std::to_string(kIdx++);
              std::string khi = "%__k" + std::to_string(kIdx++);
              addLet(klo, leafT, intInit(r.lo), /*mut=*/false);
              addLet(khi, leafT, intInit(r.hi), /*mut=*/false);
              e.instrs.push_back(cmpRelInstr("%__c", operand, RelOp::GE, klo));
              e.instrs.push_back(andInstr("%__acc", "%__c"));
              e.instrs.push_back(cmpRelInstr("%__c", operand, RelOp::LE, khi));
              e.instrs.push_back(andInstr("%__acc", "%__c"));
              continue;
            }
            std::string k = "%__k" + std::to_string(kIdx++);
            addLet(k, leafT, litInit(leaf.val), /*mut=*/false);
            e.instrs.push_back(cmpEqInstr("%__c", operand, k));
            e.instrs.push_back(andInstr("%__acc", "%__c"));
          }
        }
      }

      FunDecl build(const std::string &name) {
        g.name = GlobalId{name, {}};
        g.retType = buildI1();
        appendParams();

        // i1 is a signed 1-bit type: true is all-ones (-1), so the neutral
        // AND accumulator starts at -1.
        addLet("%__acc", buildI1(), intInit(-1), /*mut=*/true);
        addLet("%__c", buildI1(), intInit(0), /*mut=*/true);

        e.label = BlockLabel{"^entry", {}};
        eIdx = 0;
        emitLeafChecks();
        e.term = Terminator{RetTerm{buildExpr(buildRValAtom(buildLValue("%__acc"))), {}}};
        g.blocks.push_back(std::move(e));
        return std::move(g);
      }
    };

    FunDecl buildGuardFun(const std::string &name, const TwinPlan &plan, const StructMap &structs) {
      GuardFunBuilder b(plan, structs);
      return b.build(name);
    }

    // The caller-side argument list matching buildGuardFun's parameters.
    std::vector<std::shared_ptr<Expr>> buildGuardArgs(const TwinPlan &plan) {
      std::vector<std::shared_ptr<Expr>> args;
      for (const auto &root: plan.guardRoots) {
        // Must skip exactly what buildGuardFun skipped, or the call would not
        // match the signature.
        if (rootIsFullyFree(plan.box, root))
          continue;
        switch (root.kind) {
          case GuardRoot::Kind::Scalar:
          case GuardRoot::Kind::Ptr:
            args.push_back(
                std::make_shared<Expr>(buildExpr(buildRValAtom(buildLValue(root.name))))
            );
            break;
          case GuardRoot::Kind::Vec:
            for (const auto &leaf: root.leaves) {
              if (leafIsFree(plan.box, root, leaf))
                continue;
              LValue lane = buildLValue(root.name);
              lane.accesses.push_back(leaf.path.front());
              args.push_back(std::make_shared<Expr>(buildExpr(buildRValAtom(std::move(lane)))));
            }
            break;
          case GuardRoot::Kind::Agg:
            args.push_back(
                std::make_shared<Expr>(buildExpr(buildAddrAtom(buildLValue(root.name))))
            );
            break;
        }
        // Expected pointers, mirroring buildGuardFun's `%__e<n>` params.
        for (const auto &leaf: root.leaves)
          if (leaf.isPtr())
            args.push_back(std::make_shared<Expr>(buildExpr(ptrRhsAtom(leaf))));
      }
      return args;
    }

    // --- candidate region planning + scoring -------------------------------

    // One planned candidate region rooted at trace point `t` (plan filled, no
    // solver body yet). `t`/`usedEnd` index the function's trace points.
    struct Cand {
      std::size_t t = 0, usedEnd = 0, nBlocks = 0;
      std::string label;
      TwinPlan plan;
      bool fellBack = false;
      // What the state-set pass made of the trace at the profiled state. It
      // does not gate the graft; it says how much room a guard has to widen.
      std::string verdict;
      // How far the guard may open: one class per integer leaf.
      Box box;
      // How much the body was rewritten before grafting.
      AntiOptReport disguised;
    };

    // What an intrinsic computes when the box pins every argument to one
    // value. The pass cannot say and must not guess — the interpreter is the
    // authority on intrinsic semantics, and it is already linked here — so
    // rytwin answers by asking it. A call whose arguments are not all single
    // values is left unknown, and one the interpreter reports as UB is not a
    // value at all: the trace traps for every state the box admits, so the
    // pass is told nothing and refuses on the arithmetic that reads it.
    IntrinsicFold intrinsicFold() {
      return [](const CallAtom &call,
                const std::vector<std::int64_t> &args) -> std::optional<std::int64_t> {
        const IntrinsicDecl *decl = call.resolvedIntrinsic;
        if (!decl || decl->params.size() != args.size())
          return std::nullopt;
        std::vector<RuntimeValue> rv;
        rv.reserve(args.size());
        for (std::size_t i = 0; i < args.size(); ++i) {
          auto bits = TypeUtils::getIntBitWidth(decl->params[i].type);
          if (!bits)
            return std::nullopt; // a float or aggregate argument: not this pass
          RuntimeValue v;
          v.kind = RuntimeValue::Kind::Int;
          v.intVal = args[i];
          v.bits = *bits;
          rv.push_back(std::move(v));
        }
        if (!TypeUtils::getIntBitWidth(decl->retType))
          return std::nullopt;
        try {
          RuntimeValue out = evalIntrinsic(*decl, rv);
          if (out.kind != RuntimeValue::Kind::Int)
            return std::nullopt;
          return out.intVal;
        } catch (...) {
          return std::nullopt; // UB, or an intrinsic the interpreter declines
        }
      };
    }

    // The profiled state, leaf by leaf: every integer pinned to the one value
    // it holds, every float to its exact value, every pointer to the cell its
    // recorded provenance names. This is the narrowest box there is — the point
    // guard — so the pass proving it is the floor, not an achievement.
    EntryState pointBox(
        const FunDecl &fn, const std::vector<std::pair<std::string, StateValue>> &vars,
        const StructMap &structs, const TypeLayout &layout
    ) {
      EntryState out;
      for (const auto &[name, val]: vars) {
        auto decl = findRoot(fn, name);
        std::vector<StateLeaf> leaves;
        bool hasPtr = false, hasUndef = false;
        enumStateLeaves(val, leaves, hasPtr, hasUndef);
        for (auto &lf: leaves) {
          const std::string key = leafKey(name, lf.path);
          switch (lf.val.kind) {
            case StateValue::Kind::Int:
              out.ints[key] = Interval{lf.val.intVal, lf.val.intVal, false, lf.val.bits, 0};
              break;
            case StateValue::Kind::Float:
              out.floats[key] = lf.val.floatVal;
              break;
            case StateValue::Kind::Ptr: {
              if (!decl)
                break;
              LeafRef ref{name, lf.path, lf.val, {}, {}};
              if (fillPtrLeaf(ref, fn, structs, layout, decl->type))
                out.ptrs[key] = ref.ptrTarget;
              break;
            }
            default:
              break;
          }
        }
      }
      return out;
    }

    // The guard's own roots carrying the profiled state — what it takes to
    // re-run the region, or the twin body, from a state near it.
    std::optional<std::vector<MiniRoot>> probeRoots(const TwinPlan &plan, const StateMap &s) {
      std::vector<MiniRoot> roots;
      roots.reserve(plan.guardRoots.size());
      for (const auto &r: plan.guardRoots) {
        auto si = s.find(r.name);
        if (si == s.end())
          return std::nullopt;
        MiniRoot g{r.name, r.type, r.isParam, *si->second, *si->second, {}};
        for (const auto &leaf: r.leaves)
          if (leaf.isPtr())
            g.ptrFixes.push_back(
                MiniPtrFix{leaf.path, leaf.ptrType, leaf.ptrTarget, leaf.ptrTarget}
            );
        roots.push_back(std::move(g));
      }
      return roots;
    }

    template<typename V>
    V *leafAtImpl(V &v, const std::vector<Access> &path, std::size_t i) {
      if (i == path.size())
        return &v;
      if (auto ai = std::get_if<AccessIndex>(&path[i])) {
        auto idx = std::get<IntLit>(ai->index).value;
        if (idx < 0 || (std::size_t) idx >= v.elems.size())
          return nullptr;
        return leafAtImpl(v.elems[(std::size_t) idx], path, i + 1);
      }
      const std::string &want = std::get<AccessField>(path[i]).field;
      for (auto &[nm, sub]: v.fields)
        if (nm == want)
          return leafAtImpl(sub, path, i + 1);
      return nullptr;
    }

    // Sample states the guard admits: every corner of every ranged leaf, then
    // interior points. Corners first, because a bound is where a box is most
    // likely to be wrong.
    std::vector<std::vector<MiniRoot>>
    sampleBox(const TwinPlan &plan, std::mt19937 &rng, std::size_t want) {
      std::vector<std::vector<MiniRoot>> out;

      struct Slot {
        std::size_t root;
        std::vector<Access> path;
        std::int64_t lo, hi;
      };

      std::vector<Slot> slots;
      for (std::size_t ri = 0; ri < plan.roots.size(); ++ri) {
        std::vector<StateLeaf> leaves;
        bool hasPtr = false, hasUndef = false;
        enumStateLeaves(plan.roots[ri].init, leaves, hasPtr, hasUndef);
        for (auto &lf: leaves) {
          if (lf.val.kind != StateValue::Kind::Int)
            continue;
          const std::string key = leafKey(plan.roots[ri].name, lf.path);
          for (const auto &bl: plan.box.leaves) {
            if (bl.key != key || bl.cls == LeafClass::Pinned)
              continue;
            const bool free = bl.cls == LeafClass::Free;
            std::int64_t lo = free ? lf.val.intVal : bl.range.lo;
            std::int64_t hi = free ? lf.val.intVal : bl.range.hi;
            if (free && lf.val.bits && lf.val.bits <= 64) {
              lo = lf.val.bits == 64 ? INT64_MIN : -(std::int64_t{1} << (lf.val.bits - 1));
              hi = lf.val.bits == 64 ? INT64_MAX : (std::int64_t{1} << (lf.val.bits - 1)) - 1;
            }
            slots.push_back({ri, lf.path, lo, hi});
          }
        }
      }
      if (slots.empty())
        return out;
      auto make = [&](const std::vector<std::int64_t> &vals) {
        std::vector<MiniRoot> st = plan.roots;
        for (std::size_t i = 0; i < slots.size(); ++i)
          if (StateValue *cell = leafAtImpl(st[slots[i].root].init, slots[i].path, 0))
            cell->intVal = vals[i];
        out.push_back(std::move(st));
      };
      std::vector<std::int64_t> vals;
      for (const auto &sl: slots)
        vals.push_back(sl.lo);
      make(vals);
      for (std::size_t i = 0; i < slots.size(); ++i)
        vals[i] = slots[i].hi;
      make(vals);
      while (out.size() < want) {
        for (std::size_t i = 0; i < slots.size(); ++i)
          vals[i] = std::uniform_int_distribution<std::int64_t>(slots[i].lo, slots[i].hi)(rng);
        make(vals);
      }
      return out;
    }

    std::size_t blockIndex(const CFG &cfg, const std::string &lbl) {
      auto it = cfg.indexOf.find(lbl);
      return it == cfg.indexOf.end() ? DomTree::kNone : it->second;
    }

    // Plan the region rooted at trace point `t` under the given claims (empty
    // when enumerating globally). The window extends while the entry dominates
    // the executed, same-frame, unclaimed block; the exit is the first block it
    // does not dominate, or the frame's returning block. Returns the candidate
    // or nullopt with a reason; a rejected region falls back to the one-block
    // window rooted at the same entry.
    std::optional<Cand> planCandidate(
        const PlanContext &ctx, const std::vector<const StatePoint *> &pts, std::size_t t,
        const std::unordered_set<std::string> &claims, std::string &why
    ) {
      const std::string &label = pts[t]->block;
      std::size_t tEnd = t + 1;
      {
        const std::size_t eIdx = blockIndex(ctx.cfg, label);
        std::size_t j = t + 1, last = t + 1;
        while (j < pts.size() && pts[j]->frame == pts[t]->frame) {
          const std::size_t bIdx = blockIndex(ctx.cfg, pts[j]->block);
          if (eIdx == DomTree::kNone || bIdx == DomTree::kNone || !ctx.dt.dominates(eIdx, bIdx) ||
              claims.count(pts[j]->block))
            break;
          last = j;
          ++j;
        }
        if (j < pts.size() && pts[j]->frame == pts[t]->frame)
          tEnd = j;
        else if (auto lb = ctx.byLabel.find(pts[last]->block);
                 lb != ctx.byLabel.end() && std::holds_alternative<RetTerm>(lb->second->term))
          tEnd = last;
      }
      std::string note;
      auto tryPlan = [&](std::size_t end, TwinPlan &plan, std::size_t &nBlocks) -> bool {
        std::vector<const Block *> blocks;
        std::unordered_set<std::string> seen;
        for (std::size_t k = t; k < end; ++k) {
          if (!seen.insert(pts[k]->block).second)
            continue;
          auto b2 = ctx.byLabel.find(pts[k]->block);
          if (b2 == ctx.byLabel.end()) {
            why = "block not found: " + pts[k]->block;
            return false;
          }
          blocks.push_back(b2->second);
        }
        nBlocks = blocks.size();
        if (blocks.empty()) {
          why = "empty window";
          return false;
        }
        plan.regionLabels.clear();
        for (const Block *bp: blocks)
          plan.regionLabels.push_back(bp->label.name);
        plan.exitLabel = pts[end]->block;
        if (!planRegion(
                ctx, blocks, /*scanTerms=*/true, pts[t]->vars, pts[end]->vars,
                toStateMap(pts[t]->vars), plan, &why
            ))
          return false;
        // The twin body is the executed trace, so it follows the window and
        // not the (deduplicated) block list: a block the loop ran three times
        // contributes its statements three times, in that order.
        std::vector<std::string> executed;
        executed.reserve(end - t);
        for (std::size_t k = t; k < end; ++k)
          executed.push_back(pts[k]->block);
        auto body = flattenTrace(executed, plan.exitLabel, ctx.byLabel, &why);
        if (!body)
          return false;
        EntryState es = pointBox(ctx.fn, pts[t]->vars, ctx.structs, ctx.layout);
        const IntrinsicFold fold = intrinsicFold();
        const StateSetVerdict iv = StateSetAnalysis::check(
            ctx.fn, ctx.structs, body->stmts, traceObligations(*body), es, &fold
        );
        note = iv.ok ? "ok" : iv.reason;
        plan.entry = std::move(es);
        plan.body = std::move(*body);
        if (auto rs = probeRoots(plan, toStateMap(pts[t]->vars)))
          plan.roots = std::move(*rs);
        return true;
      };

      Cand c;
      c.t = t;
      c.usedEnd = tEnd;
      c.label = label;
      if (!tryPlan(tEnd, c.plan, c.nBlocks)) {
        if (tEnd == t + 1)
          return std::nullopt;
        std::string whyRegion = why;
        c.plan = TwinPlan{};
        c.usedEnd = t + 1;
        if (!tryPlan(t + 1, c.plan, c.nBlocks)) {
          why = "region: " + whyRegion + "; block: " + why;
          return std::nullopt;
        }
        c.fellBack = true;
      }
      c.verdict = note;
      return c;
    }

    // How many leaves of each class the box ended up with, which is what a
    // reader of the log wants: a guard that frees leaves says more than one
    // that merely widens them.
    // Rewrite a twin body so it stops reading as the region it came from. The
    // obligation handed to the engine is the one the guard makes: the body has
    // to stay right for every state the box admits, not merely for the
    // profiled one — an identity whose intermediate overflows somewhere in the
    // box is fine at the profile and wrong in the guard.
    AntiOptReport antiOptimizeBody(
        FunDecl &fn, const StructMap &structs, TwinPlan &plan, const Box &box, std::mt19937 &rng
    ) {
      EntryState guarded = plan.entry;
      for (const auto &leaf: box.leaves)
        switch (leaf.cls) {
          case LeafClass::Free:
            guarded.ints.erase(leaf.key);
            break;
          case LeafClass::Ranged:
            guarded.ints[leaf.key] = leaf.range;
            break;
          default:
            break;
        }
      // Cells the rewriting declares as it goes are constants the box never
      // heard of: an operand on the right of `* / % & | ^ << >> >>>` must be an
      // lvalue (spec §5.3), so `%x << 3` is spelled with a literal local, and
      // the pass would otherwise have to call the shift amount unknown and
      // refuse every shift. An immutable local can never be assigned (§6.6), so
      // its initializer is its value for the whole function. Only names the box
      // does *not* classify are added: a leaf the box freed is one the guard
      // will not check, and the pass must not go on knowing what the guard has
      // stopped enforcing.
      std::unordered_set<std::string> boxed;
      for (const auto &leaf: box.leaves)
        boxed.insert(leaf.key);
      auto withConstantCells = [&](EntryState st) {
        for (const auto &l: fn.lets) {
          if (l.isMutable || !l.init || l.init->kind != InitVal::Kind::Int)
            continue;
          if (boxed.count(l.name.name) || st.ints.count(l.name.name))
            continue;
          auto bits = TypeUtils::getIntBitWidth(l.type);
          if (!bits)
            continue;
          const std::int64_t v = std::get<IntLit>(l.init->value).value;
          st.ints[l.name.name] = Interval{v, v, false, *bits, 0};
        }
        return st;
      };

      const IntrinsicFold fold = intrinsicFold();
      NameAllocator names(kAntiOptLocalPrefix);
      TraceBody &body = plan.body;
      auto probeOf = [&](const std::vector<Instr> &s) {
        TraceBody probe;
        probe.stmts = cloneInstrs(s);
        probe.exitLabel = body.exitLabel;
        // A check is move-only, so the body being judged gets its own copies.
        for (const auto &chk: body.checks)
          probe.checks.push_back(PathCheck{chk.afterStmt, cloneCond(chk.cond), chk.taken});
        return probe;
      };

      // What the rules that need a licence are told. It is the same forward
      // pass that certifies the guard, read for its annotations: a rule asking
      // "what can %x hold here" is asking about every state the guard admits,
      // which is exactly the question the box answers.
      class BoxFacts : public AntiOptFacts {
      public:
        BoxFacts(
            const FunDecl &fn, const StructMap &structs, const Box &box,
            std::function<TraceBody(const std::vector<Instr> &)> probe,
            std::function<EntryState()> state, IntrinsicFold fold
        ) :
            fn_(fn), structs_(structs), probe_(std::move(probe)), state_(std::move(state)),
            fold_(std::move(fold)) {
          for (const auto &leaf: box.leaves)
            if (leaf.cls == LeafClass::Free)
              free_.insert(leaf.key);
        }

        void refresh(const std::vector<Instr> &stmts) override {
          // The trace outlives the call: the obligations borrow its conditions.
          const TraceBody probed = probe_(stmts);
          snaps_ = StateSetAnalysis::snapshots(
              fn_, structs_, probed.stmts, traceObligations(probed), state_(), &fold_
          );
        }

        std::optional<ValueRange>
        rangeBefore(std::size_t at, const std::string &local) const override {
          if (at >= snaps_.size())
            return std::nullopt;
          auto it = snaps_[at].find(local);
          if (it == snaps_[at].end() || it->second.unknown)
            return std::nullopt;
          return ValueRange{it->second.lo, it->second.hi};
        }

        bool isFree(const std::string &local) const override { return free_.count(local) > 0; }

      private:
        const FunDecl &fn_;
        const StructMap &structs_;
        std::function<TraceBody(const std::vector<Instr> &)> probe_;
        std::function<EntryState()> state_;
        IntrinsicFold fold_;
        std::unordered_set<std::string> free_;
        std::vector<IntervalEnv> snaps_;
      };

      BoxFacts facts(fn, structs, box, probeOf, [&] { return withConstantCells(guarded); }, fold);
      AntiOptContext ctx{fn, structs, body.checks, names, fn.lets, rng, &facts};
      return antiOptimize(
          body.stmts, body.checks, ctx,
          [&](const std::vector<Instr> &s) {
            // Re-read the declarations every time: the engine adds cells as it
            // rewrites, and the body being judged may already use them.
            // The trace outlives the call: the obligations borrow its conditions.
            const TraceBody probed = probeOf(s);
            return StateSetAnalysis::check(
                       fn, structs, probed.stmts, traceObligations(probed),
                       withConstantCells(guarded), &fold
            )
                .ok;
          },
          rytwin::hp::kTwinRewriteRounds * rytwin::hp::kTwinRewritesPerRound
      );
    }

    std::string describeBox(const Box &b) {
      std::size_t nFree = 0, nRanged = 0, nPinned = 0;
      std::string widest;
      std::int64_t best = 0;
      for (const auto &l: b.leaves) {
        switch (l.cls) {
          case LeafClass::Free:
            ++nFree;
            break;
          case LeafClass::Ranged: {
            ++nRanged;
            const std::int64_t w = l.range.hi - l.range.lo;
            if (w > best) {
              best = w;
              widest = l.key;
            }
            break;
          }
          default:
            ++nPinned;
        }
      }
      std::string out = "box: " + std::to_string(nFree) + " free, " + std::to_string(nRanged) +
                        " ranged, " + std::to_string(nPinned) + " pinned";
      if (!widest.empty())
        out += " (widest " + widest + " +/-" + std::to_string(best / 2) + ")";
      return out + " [" + std::to_string(b.passes) + " passes]";
    }

    // The interestingness features of a candidate: loop iterations collapsed
    // (a repeated block in the window means a loop was swallowed), region
    // size, state-diff size, and the entry's fan-in.
    CandidateInfo candidateInfo(const Cand &c, const CFG &cfg) {
      CandidateInfo info;
      info.distinctBlocks = (long) c.nBlocks;
      info.loopItersCollapsed = (long) (c.usedEnd - c.t) - info.distinctBlocks;
      info.changedLeaves = (long) c.plan.defs.size();
      const std::size_t e = blockIndex(cfg, c.label);
      info.fanIn = e == DomTree::kNone ? 0 : (long) cfg.pred[e].size();
      return info;
    }

    class TwinTransform : public Transform {
    public:
      TwinTransform(SelectionPolicy select, bool spotCheck) :
          select_(std::move(select)), spotCheck_(spotCheck) {}

      std::string_view name() const override { return "TwinTransform"; }

      bool needsProfile() const override { return true; }

      TransformReport apply(Program &prog, TransformContext &ctx) override {
        TransformReport rep;
        std::size_t spotChecked = 0;
        std::uniform_real_distribution<double> coin(0.0, 1.0);
        auto vlog = [&](const std::string &m) {
          if (ctx.verbose)
            *ctx.verbose << "  twin " << m << "\n";
        };

        StructMap structs;
        for (const auto &sd: prog.structs)
          structs[sd.name.name] = &sd;
        const TypeLayout layout(prog);

        for (auto &[pfKey, profile]: ctx.profiles) {
          // Group block-entry points by executing function. Sidecar files
          // that predate frame capture have no per-point function; those
          // traces are single-frame by construction (the leaf entry).
          std::unordered_map<std::string, std::vector<const StatePoint *>> byFn;
          std::vector<std::string> fnOrder;
          for (const auto &pt: profile.trace) {
            if (pt.instr != -1)
              continue;
            const std::string &fnName = pt.func.empty() ? profile.func : pt.func;
            auto [it, inserted] = byFn.try_emplace(fnName);
            if (inserted)
              fnOrder.push_back(fnName);
            it->second.push_back(&pt);
          }

          // Guard functions must be declared before their (sole) caller;
          // insert them after all functions are processed so the indices
          // stay stable while grafting.
          std::vector<std::pair<std::string, std::vector<FunDecl>>> pendingGuards;

          auto isResidue = [](const std::string &l) {
            return l.find("__twin") != std::string::npos || l.find("__orig") != std::string::npos ||
                   l.find("__merge") != std::string::npos;
          };
          auto findFn = [&](const std::string &nm) -> FunDecl * {
            for (auto &f: prog.funs)
              if (f.name.name == nm)
                return &f;
            return nullptr;
          };
          auto byLabelOf = [](const FunDecl &fn) {
            std::unordered_map<std::string, const Block *> m;
            for (const auto &b: fn.blocks)
              m[b.label.name] = &b;
            return m;
          };

          // Log and record a chosen candidate. The body was built with the
          // plan (it is the trace itself), so there is nothing to synthesize.
          auto commit = [&](const std::string &fnName, Cand &c,
                            std::unordered_map<std::string, TwinPlan> &decided) {
            // Only regions actually being twinned pay for a box, or for the
            // rewriting that follows it: the box says which states the body
            // must stay right for, so it has to come first.
            // The trace's own length, which is what the region contributed;
            // the rewriting below adds to it, and says by how much itself.
            const std::size_t traced = c.plan.body.stmts.size();
            if (FunDecl *fnp = findFn(fnName)) {
              const IntrinsicFold fold = intrinsicFold();
              c.box = computeBox(*fnp, structs, c.plan.body, c.plan.entry, ctx.rng, &fold);
              c.disguised = antiOptimizeBody(*fnp, structs, c.plan, c.box, ctx.rng);
            }
            c.plan.box = c.box;
            vlog(
                fnName + " " + c.label + ": grafted region -> " + c.plan.exitLabel + " (" +
                std::to_string(c.nBlocks) + " blk, " + std::to_string(traced) + " stmts, " +
                std::to_string(c.plan.body.checks.size()) + " path cond, verdict " + c.verdict +
                ", " + describeBox(c.box) + ", " + std::to_string(c.disguised.applied) +
                " rewrites (" + std::to_string(c.disguised.rolledBack) + " undone" +
                (c.disguised.trapFreeOnly ? ", trap-free only" : "") +
                (c.disguised.byRule.empty() ? "" : "; " + describeRules(c.disguised)) + ")" +
                (c.fellBack ? " [window fell back to one block]" : "")
            );
            decided.emplace(c.label, std::move(c.plan));
          };

          // Expand a function's chosen `decided` map into guard/twin/orig
          // blocks and queue its guard functions.
          auto graftFunction = [&](FunDecl *fn, const std::string &fnName,
                                   std::unordered_map<std::string, TwinPlan> &decided) {
            if (decided.empty())
              return;
            const std::string fnStem =
                fnName.empty() || fnName[0] != '@' ? fnName : fnName.substr(1);
            std::vector<FunDecl> guardFuns;
            std::vector<Block> nb;
            nb.reserve(fn->blocks.size() + 4 * decided.size());
            for (auto &b: fn->blocks) {
              auto dit = decided.find(b.label.name);
              if (dit == decided.end()) {
                nb.push_back(std::move(b));
                continue;
              }
              std::string labelStem = b.label.name;
              if (!labelStem.empty() && labelStem[0] == '^')
                labelStem.erase(0, 1);
              std::string guardName = "@__twg_" + fnStem + "_" + labelStem;
              guardFuns.push_back(buildGuardFun(guardName, dit->second, structs));
              graftRegion(b, dit->second, guardName, nb);
              ++rep.sites;
            }
            fn->blocks = std::move(nb);
            pendingGuards.emplace_back(fnName, std::move(guardFuns));
          };

          // Selection is one loop over every eligible region program-wide:
          // the SelectionPolicy scores the candidates into per-region twin
          // probabilities, and each region is twinned by an independent draw.
          // Overlaps are resolved in enumeration (trace) order: the first
          // region drawn claims its blocks.
          struct Scored {
            std::string fnName;
            Cand cand;
            CandidateInfo info;
          };

          std::vector<Scored> pool;
          for (const auto &fnName: fnOrder) {
            FunDecl *fn = findFn(fnName);
            if (!fn)
              continue;
            const auto byLabel = byLabelOf(*fn);
            DiagBag diags;
            const CFG cfg = CFG::build(*fn, diags);
            const DomTree dt = DomTree::build(cfg);
            // One fixpoint per function, not per candidate region.
            const InitAtEntry inited = DefiniteInitAnalysis::initializedAtBlockEntry(*fn);
            const auto &pts = byFn[fnName];
            const std::unordered_set<std::string> noClaims;
            std::unordered_set<std::string> enumerated;
            for (std::size_t t = 0; t + 1 < pts.size(); ++t) {
              const std::string &label = pts[t]->block;
              if (isResidue(label) || pts[t]->frame != pts[t + 1]->frame ||
                  byLabel.find(label) == byLabel.end() || !enumerated.insert(label).second)
                continue;
              std::string why;
              const PlanContext ctx{*fn, cfg, dt, byLabel, structs, layout, inited};
              if (auto c = planCandidate(ctx, pts, t, noClaims, why)) {
                CandidateInfo info = candidateInfo(*c, cfg); // features before moving `c`
                pool.push_back({fnName, std::move(*c), info});
              } else
                vlog(fnName + " " + label + ": rejected (" + why + ")");
            }
          }
          // The policy turns the whole candidate set into per-region twin
          // probabilities (it owns any normalization and parameters).
          std::vector<CandidateInfo> infos;
          infos.reserve(pool.size());
          for (const auto &a: pool)
            infos.push_back(a.info);
          const std::vector<double> probs = select_ ? select_(infos) : std::vector<double>();

          std::unordered_set<std::string> claimed; // "<fn>#<block>"
          std::unordered_map<std::string, std::unordered_map<std::string, TwinPlan>> decidedByFn;
          for (std::size_t i = 0; i < pool.size(); ++i) {
            Scored &a = pool[i];
            const auto &pts = byFn[a.fnName];
            auto key = [&](std::size_t k) { return a.fnName + "#" + pts[k]->block; };
            bool overlap = false;
            for (std::size_t k = a.cand.t; k < a.cand.usedEnd && !overlap; ++k)
              overlap = claimed.count(key(k)) > 0;
            if (overlap) {
              vlog(a.fnName + " " + a.cand.label + ": skipped (overlaps a selected region)");
              continue;
            }
            const double p = i < probs.size() ? probs[i] : 0.0;
            if (coin(ctx.rng) >= p) {
              char buf[16];
              std::snprintf(buf, sizeof buf, "%.2f", p);
              vlog(a.fnName + " " + a.cand.label + ": skipped (twin p=" + buf + ")");
              continue;
            }
            for (std::size_t k = a.cand.t; k < a.cand.usedEnd; ++k)
              claimed.insert(key(k));
            commit(a.fnName, a.cand, decidedByFn[a.fnName]);
          }
          for (const auto &fnName: fnOrder) {
            FunDecl *fn = findFn(fnName);
            if (!fn)
              continue;
            auto it = decidedByFn.find(fnName);
            if (it == decidedByFn.end())
              continue;
            graftFunction(fn, fnName, it->second);
            if (!spotCheck_)
              continue;
            // The arms exist only now, so the check runs after the graft.
            for (const auto &[label, plan]: it->second) {
              auto agreed = spotCheck(prog, fnName, label, plan, ctx.rng);
              if (!agreed) {
                rep.ok = false;
                rep.message = "twin disagreed with its region on a state its guard admits (" +
                              fnName + " " + label + ")";
                return rep;
              }
              spotChecked += *agreed;
            }
          }

          if (spotCheck_ && rep.sites)
            rep.message =
                "spot-checked " + std::to_string(spotChecked) + " state(s) inside the guard boxes";

          for (auto &[fnName, guards]: pendingGuards) {
            for (std::size_t i = 0; i < prog.funs.size(); ++i)
              if (prog.funs[i].name.name == fnName) {
                prog.funs.insert(
                    prog.funs.begin() + i, std::make_move_iterator(guards.begin()),
                    std::make_move_iterator(guards.end())
                );
                break;
              }
          }
        }
        return rep;
      }

    private:
      // The guard block (label = base): branch on the guard-function call to
      // the twin or orig arm.
      static Block guardBlock(
          const std::string &base, TwinPlan &plan, const std::string &guardName,
          const std::string &twinL, const std::string &origL
      ) {
        Block guard;
        guard.label = BlockLabel{base, {}};
        CallAtom call;
        call.callee = GlobalId{guardName, {}};
        call.args = buildGuardArgs(plan);
        guard.term = brIfExpr(buildExpr(Atom{std::move(call), {}}), twinL, origL);
        return guard;
      }

      // guard / twin / orig. The twin replays the region's executed trace and
      // jumps straight to the observed exit, skipping every intermediate
      // block; orig keeps the entry block intact (instructions and
      // terminator) so the region runs normally when the guard misses.
      static void
      graftRegion(Block &b, TwinPlan &plan, const std::string &guardName, std::vector<Block> &out) {
        const std::string base = b.label.name;
        const std::string twinL = base + "__twin", origL = base + "__orig";

        out.push_back(guardBlock(base, plan, guardName, twinL, origL));

        Block twin;
        twin.label = BlockLabel{twinL, {}};
        twin.instrs = std::move(plan.body.stmts);
        twin.term = brTo(plan.exitLabel);
        out.push_back(std::move(twin));

        Block orig;
        orig.label = BlockLabel{origL, {}};
        orig.instrs = std::move(b.instrs);
        orig.term = std::move(b.term);
        out.push_back(std::move(orig));
      }

      // Run the region and the twin body from states the guard admits, and
      // compare. The box is a proof, so this proves nothing further — it is
      // there to catch a mistake in the proof, which is exactly the kind of
      // bug a single profiled input cannot see once a guard admits many
      // states. Returns how many states agreed, or nullopt on a disagreement.
      std::optional<std::size_t> spotCheck(
          const Program &prog, const std::string &fnName, const std::string &label,
          const TwinPlan &plan, std::mt19937 &rng
      ) {
        auto states = sampleBox(plan, rng, rytwin::hp::kTwinSpotChecks);
        if (states.empty())
          return std::size_t{0};
        // Post-graft the region begins at the orig arm; the twin body is one
        // block of its own.
        std::vector<std::string> origLabels{label + "__orig"};
        for (const auto &l: plan.regionLabels)
          if (l != label)
            origLabels.push_back(l);
        RegionProbe orig(prog, fnName, origLabels, plan.roots);
        RegionProbe twin(prog, fnName, {label + "__twin"}, plan.roots);
        if (!orig.valid() || !twin.valid())
          return std::size_t{0};
        std::size_t agreed = 0;
        for (const auto &st: states) {
          ProbeResult a = orig.run(st), b = twin.run(st);
          if (!a.ok || !b.ok || a.exitLabel != b.exitLabel)
            return std::nullopt;
          if (a.effect.size() != b.effect.size())
            return std::nullopt;
          for (std::size_t i = 0; i < a.effect.size(); ++i) {
            // Locals the rewriting introduced are the twin's own scratch: the
            // twin arm writes them, the orig arm leaves them at their
            // declaration, and nothing outside the block reads them. Comparing
            // them would report the arms as differing on something no caller
            // can observe.
            const std::string &nm = a.effect[i].first;
            if (nm.compare(0, std::strlen(kAntiOptLocalPrefix), kAntiOptLocalPrefix) == 0)
              continue;
            if (nm != b.effect[i].first || !bitExactEq(a.effect[i].second, b.effect[i].second)) {
              std::fprintf(
                  stderr, "[spot] %s orig=%lld twin=%lld | state:", nm.c_str(),
                  (long long) a.effect[i].second.intVal, (long long) b.effect[i].second.intVal
              );
              for (const auto &r: st) {
                std::vector<StateLeaf> lv;
                bool hp = false, hu = false;
                enumStateLeaves(r.init, lv, hp, hu);
                for (const auto &l: lv)
                  if (l.val.kind == StateValue::Kind::Int)
                    std::fprintf(
                        stderr, " %s=%lld", leafKey(r.name, l.path).c_str(),
                        (long long) l.val.intVal
                    );
              }
              std::fprintf(stderr, "\n");
              return std::nullopt;
            }
          }
          ++agreed;
        }
        return agreed;
      }

      SelectionPolicy select_;
      bool spotCheck_ = false;
    };

  } // namespace

  SelectionPolicy uniformPolicy(double pTwin) {
    return [pTwin](const std::vector<CandidateInfo> &cs) {
      return std::vector<double>(cs.size(), pTwin);
    };
  }

  SelectionPolicy interestingPolicy(double pTwin, double temp) {
    return [pTwin, temp](const std::vector<CandidateInfo> &cs) {
      std::vector<double> ps(cs.size(), 0.0);
      if (cs.empty())
        return ps;
      auto score = [](const CandidateInfo &c) {
        return 1000 * c.loopItersCollapsed + 10 * c.distinctBlocks + 5 * c.changedLeaves + c.fanIn;
      };
      long mn = score(cs[0]), mx = mn;
      for (const auto &c: cs) {
        const long s = score(c);
        mn = std::min(mn, s);
        mx = std::max(mx, s);
      }
      const double range = mx > mn ? (double) (mx - mn) : 0.0;
      for (std::size_t i = 0; i < cs.size(); ++i) {
        // Normalize the score to [0,1] so the temperature is scale-free, then
        // p = pTwin ^ exp((0.5 - norm)/temp): monotone in the score, 1 at
        // pTwin=1, 0 at pTwin=0, and -> the uniform pTwin coin as temp -> inf.
        const double norm = range > 0.0 ? (double) (score(cs[i]) - mn) / range : 0.5;
        ps[i] = std::pow(pTwin, std::exp((0.5 - norm) / temp));
      }
      return ps;
    };
  }

  std::unique_ptr<Transform> makeTwinTransform(SelectionPolicy select, bool spotCheck) {
    return std::make_unique<TwinTransform>(std::move(select), spotCheck);
  }

} // namespace refractir::reify
