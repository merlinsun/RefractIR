#include "frontend/typechecker.hpp"
#include <algorithm>
#include <functional>
#include "analysis/cfg.hpp"
#include "analysis/tensor.hpp"
#include "analysis/type_utils.hpp"

namespace refractir {

  refractir::PassResult TypeChecker::run(Program &prog, DiagBag &diags) {
    collectStructs(prog, diags);
    collectCallees(prog, diags);
    for (const auto &f: prog.funs) {
      TypeAnnotations ann;
      checkFunction(f, ann, diags);
    }
    // Contract well-formedness uses the callee registry.
    for (const auto &d: prog.extDecls) {
      if (d.contract) {
        TypeAnnotations ann;
        checkContract(d, ann, diags);
      }
    }
    checkNoRecursion(prog, diags);
    return diags.hasErrors() ? refractir::PassResult::Error : refractir::PassResult::Success;
  }

  // Recursive well-formedness check on type usages. Catches the
  // structural restrictions §3.2 and §6.9.1: no vector of non-scalar, no
  // ptr to vector, no vector with N<2, no vector as struct field, no array
  // of vector.
  static void validateTypeWF(
      const TypePtr &t, DiagBag &diags, bool insideVec = false, bool insideField = false,
      bool insideArray = false, bool insidePtr = false, TensorProfile profile = {}
  ) {
    if (!t)
      return;
    std::visit(
        [&](auto &&x) {
          using X = std::decay_t<decltype(x)>;
          if constexpr (std::is_same_v<X, VecType>) {
            if (insideVec)
              diags.error("Vectors cannot be nested (<N> <M> T)", t->span);
            if (insideField)
              diags.error("Vectors cannot appear as struct fields", t->span);
            if (insideArray)
              diags.error("Arrays of vectors are not supported", t->span);
            if (x.size < 2)
              diags.error("Vector lane count must be >= 2; got " + std::to_string(x.size), t->span);
            // The lane element must itself be scalar (Int or Float). Recurse
            // with insideVec=true so the recursion catches nested vectors;
            // PtrType / StructType / ArrayType / VecType inside a vector
            // is rejected by the recursive call (or by the explicit branches
            // below).
            if (x.elem && !std::holds_alternative<IntType>(x.elem->v) &&
                !std::holds_alternative<FloatType>(x.elem->v)) {
              diags.error("Vector lane type must be a scalar (iN, f32, f64)", t->span);
            }
            validateTypeWF(x.elem, diags, true, false, false, false, profile);
          } else if constexpr (std::is_same_v<X, TensorType>) {
            if (insideVec || insideField || insideArray || insidePtr)
              diags.error(
                  "tensor-type: tensors cannot be nested in vectors, aggregates, or pointers",
                  t->span
              );
            try {
              (void) tensorExecutionTileShape(x.shape, profile);
            } catch (const std::exception &e) {
              diags.error(e.what(), t->span);
            }
            const auto intBits = TypeUtils::getIntBitWidth(x.elem);
            const auto floatBits = TypeUtils::getFloatBitWidth(x.elem);
            if ((!intBits || *intBits != 32) && (!floatBits || *floatBits != 32))
              diags.error("tensor-type: tensor element type must be i32 or f32", t->span);
            validateTypeWF(x.elem, diags, false, false, false, false, profile);
          } else if constexpr (std::is_same_v<X, PtrType>) {
            // ptr <N> T is forbidden (§6.8.1).
            if (x.pointee && std::holds_alternative<VecType>(x.pointee->v))
              diags.error("Pointer to vector (ptr <N> T) is not supported", t->span);
            validateTypeWF(x.pointee, diags, false, false, false, true, profile);
          } else if constexpr (std::is_same_v<X, ArrayType>) {
            validateTypeWF(x.elem, diags, false, false, true, false, profile);
          }
          // StructType: validated separately in collectStructs.
        },
        t->v
    );
  }

  void TypeChecker::collectStructs(const Program &prog, DiagBag &diags) {
    for (const auto &sd: prog.structs) {
      StructInfo si;
      si.declSpan = sd.span;
      for (const auto &fd: sd.fields) {
        // Validate per-field type: in particular reject vector
        // fields explicitly with insideField=true.
        validateTypeWF(fd.type, diags, false, true, false, false, tensorProfile_);
        si.fields[fd.name] = fd.type;
        si.fieldList.push_back({fd.name, fd.type});
      }
      structs_[sd.name.name] = std::move(si);
    }
  }

  void TypeChecker::collectCallees(const Program &prog, DiagBag &diags) {
    // 1. Funs.
    for (const auto &f: prog.funs) {
      CalleeInfo ci;
      ci.kind = CalleeInfo::Kind::Fun;
      ci.retType = f.retType;
      ci.declSpan = f.span;
      ci.fun = &f;
      for (const auto &p: f.params)
        ci.paramTypes.push_back(p.type);
      callees_[f.name.name].push_back(std::move(ci));
    }
    // 2. Ext decls.
    for (const auto &d: prog.extDecls) {
      // Validate signatures.
      validateTypeWF(d.retType, diags, false, false, false, false, tensorProfile_);
      for (const auto &p: d.params)
        validateTypeWF(p.type, diags, false, false, false, false, tensorProfile_);
      auto it = callees_.find(d.name.name);
      if (it != callees_.end() && !it->second.empty()) {
        diags.error("`decl " + d.name.name + "` conflicts with a `fun` of the same name", d.span);
      }
      CalleeInfo ci;
      ci.kind = d.contract ? CalleeInfo::Kind::ExtContract : CalleeInfo::Kind::ExtLink;
      ci.retType = d.retType;
      ci.declSpan = d.span;
      ci.ext = &d;
      for (const auto &p: d.params)
        ci.paramTypes.push_back(p.type);
      if (it == callees_.end() || it->second.empty())
        callees_[d.name.name].push_back(std::move(ci));
    }
    // 3. Intrinsics.
    for (const auto &i: prog.intrinsics) {
      validateTypeWF(i.retType, diags, false, false, false, false, tensorProfile_);
      for (const auto &p: i.params)
        validateTypeWF(p.type, diags, false, false, false, false, tensorProfile_);
      // §12 Intrinsics are defined over scalar `iN` or `fN`
      // — except the v0.2.3 horizontal reductions (§12.4), whose parameter
      // is a vector `<N> T` of scalar lanes. This is the coarse gate;
      // per-intrinsic correctness (which iN/fN/`<N> T` combinations a given
      // @name accepts) is enforced by the semantic checker.
      auto isIntOrFp = [](const TypePtr &t) {
        return t &&
               (std::holds_alternative<IntType>(t->v) || std::holds_alternative<FloatType>(t->v));
      };
      auto isVecOfScalar = [&](const TypePtr &t) {
        auto vt = t ? std::get_if<VecType>(&t->v) : nullptr;
        return vt && isIntOrFp(vt->elem);
      };
      // The result of every intrinsic — reductions included — is scalar.
      if (!isIntOrFp(i.retType))
        diags.error(
            "`intrinsic " + i.name.name + "` return type must be `iN` or `fN` (§12)", i.span
        );
      for (const auto &p: i.params)
        if (!isIntOrFp(p.type) && !isVecOfScalar(p.type))
          diags.error(
              "`intrinsic " + i.name.name +
                  "` parameter type must be `iN`, `fN`, or a vector `<N> T` of these (§12)",
              i.span
          );
      auto it = callees_.find(i.name.name);
      bool hasNonIntrinsic = false;
      bool hasSameSig = false;
      if (it != callees_.end()) {
        // If any existing callee is NOT an intrinsic, reject (name collision
        // between intrinsic and fun/decl).
        for (const auto &existing: it->second) {
          if (existing.kind != CalleeInfo::Kind::Intrinsic) {
            hasNonIntrinsic = true;
            break;
          }
          // For intrinsic-intrinsic collision, check parameter types.
          if (existing.paramTypes.size() == i.params.size()) {
            bool same = true;
            for (size_t k = 0; k < i.params.size(); ++k) {
              if (!TypeUtils::areTypesEqual(existing.paramTypes[k], i.params[k].type)) {
                same = false;
                break;
              }
            }
            if (same) {
              hasSameSig = true;
              break;
            }
          }
        }
        if (hasNonIntrinsic) {
          diags.error(
              "`intrinsic " + i.name.name + "` conflicts with another declaration of the same name",
              i.span
          );
        } else if (hasSameSig) {
          diags.error(
              "`intrinsic " + i.name.name + "` conflicts with another declaration of the same name",
              i.span
          );
        }
      }
      CalleeInfo ci;
      ci.kind = CalleeInfo::Kind::Intrinsic;
      ci.retType = i.retType;
      ci.declSpan = i.span;
      ci.intr = &i;
      for (const auto &p: i.params)
        ci.paramTypes.push_back(p.type);
      // Add to registry unless it's a duplicate signature.
      if (!hasSameSig || it == callees_.end())
        callees_[i.name.name].push_back(std::move(ci));
    }
  }

  // §6.11 Contract well-formedness:
  //   - Each `pre` cond's free variables ⊆ params (no `ret`).
  //   - Each `post` cond's free variables ⊆ params ∪ {ret}.
  //   - `ret` has the declared return type.
  void TypeChecker::checkContract(const ExtDecl &d, TypeAnnotations &ann, DiagBag &diags) {
    if (!d.contract)
      return;
    std::unordered_map<std::string, VarInfo> vars;
    std::unordered_map<std::string, SymInfo> syms;
    for (const auto &p: d.params) {
      vars[p.name.name] = VarInfo{p.type, /*isMutable=*/false, /*isParam=*/true, p.span};
    }

    // Pre clauses: `ret` must not appear.
    for (const auto &pre: d.contract->pres) {
      // Check that `ret` isn't referenced in the cond.
      auto exprMentionsRet = [](const Expr &e) -> bool {
        // Walk atoms/coefs/lvalues looking for LocalId "ret".
        auto checkLV = [](const LValue &lv) { return lv.base.name == "ret"; };
        auto checkCoef = [](const Coef &c) {
          if (auto lsid = std::get_if<LocalOrSymId>(&c))
            if (auto lid = std::get_if<LocalId>(lsid))
              return lid->name == "ret";
          return false;
        };
        auto checkAtom = [&](const Atom &a) {
          return std::visit(
              [&](auto &&x) -> bool {
                using T = std::decay_t<decltype(x)>;
                if constexpr (std::is_same_v<T, CoefAtom>)
                  return checkCoef(x.coef);
                if constexpr (std::is_same_v<T, RValueAtom>)
                  return checkLV(x.rval);
                if constexpr (std::is_same_v<T, OpAtom>)
                  return checkCoef(x.coef) || checkLV(x.rval);
                if constexpr (std::is_same_v<T, UnaryAtom>)
                  return checkLV(x.rval);
                if constexpr (std::is_same_v<T, LoadAtom>)
                  return checkLV(x.rval);
                if constexpr (std::is_same_v<T, AddrAtom>)
                  return checkLV(x.lv);
                return false;
              },
              a.v
          );
        };
        if (checkAtom(e.first))
          return true;
        for (const auto &t: e.rest)
          if (checkAtom(t.atom))
            return true;
        return false;
      };
      if (exprMentionsRet(pre.cond.lhs) || exprMentionsRet(pre.cond.rhs)) {
        diags.error("`ret` may not appear in `pre` clauses (§6.11)", pre.span);
      }
      // §6.11 pre clauses must be scalar Cond; vector-typed
      // operands would produce a `<N> i1` mask, which is not a single
      // truth value. Type the LHS first (with null disallowed); if the
      // LHS is a vector, flag the error before delegating to checkCond
      // (which doesn't enforce scalar-only).
      {
        auto lt = typeOfExpr(pre.cond.lhs, vars, syms, ann, diags, std::nullopt);
        if (lt.isVec())
          diags.error("`pre` condition must be scalar, not vector (§6.11)", pre.span);
      }
      checkCond(pre.cond, vars, syms, ann, diags);
    }

    // Post clauses: `ret` is a synthetic immutable local of declared return type.
    vars["ret"] = VarInfo{d.retType, /*isMutable=*/false, /*isParam=*/true, d.span};
    for (const auto &post: d.contract->posts) {
      {
        auto lt = typeOfExpr(post.cond.lhs, vars, syms, ann, diags, std::nullopt);
        if (lt.isVec())
          diags.error("`post` condition must be scalar, not vector (§6.11)", post.span);
      }
      checkCond(post.cond, vars, syms, ann, diags);
    }
    vars.erase("ret");
  }

  // §9.7 No recursion. Build the call graph from fun bodies and
  // contract bodies (which can transitively `call` other functions via
  // `pre`/`post` clauses — though uncommon, the spec covers it). Tarjan's
  // SCC catches both self-edges and mutual cycles.
  void TypeChecker::checkNoRecursion(const Program &prog, DiagBag &diags) {
    // Build adjacency.
    std::unordered_map<std::string, std::vector<std::string>> adj;
    std::function<void(const std::string &, const Atom &)> walkAtom;
    walkAtom = [&](const std::string &caller, const Atom &a) {
      std::visit(
          [&](auto &&x) {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, CallAtom>) {
              adj[caller].push_back(x.callee.name);
              for (const auto &arg: x.args) {
                walkAtom(caller, arg->first);
                for (const auto &t: arg->rest)
                  walkAtom(caller, t.atom);
              }
            }
          },
          a.v
      );
    };
    auto addCalls = [&](const std::string &caller, const Expr &e) {
      walkAtom(caller, e.first);
      for (const auto &t: e.rest)
        walkAtom(caller, t.atom);
    };
    for (const auto &f: prog.funs) {
      adj[f.name.name]; // ensure node exists
      // Walk lets initializers, instructions, and terminators.
      for (const auto &l: f.lets) {
        if (l.init && l.init->kind == InitVal::Kind::Atom) {
          const auto &atom = *std::get<AtomPtr>(l.init->value);
          walkAtom(f.name.name, atom);
        }
      }
      for (const auto &b: f.blocks) {
        for (const auto &ins: b.instrs) {
          std::visit(
              [&](auto &&arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, AssignInstr>) {
                  addCalls(f.name.name, arg.rhs);
                } else if constexpr (std::is_same_v<T, AssumeInstr>) {
                  addCalls(f.name.name, arg.cond.lhs);
                  addCalls(f.name.name, arg.cond.rhs);
                } else if constexpr (std::is_same_v<T, RequireInstr>) {
                  addCalls(f.name.name, arg.cond.lhs);
                  addCalls(f.name.name, arg.cond.rhs);
                } else if constexpr (std::is_same_v<T, StoreInstr>) {
                  addCalls(f.name.name, arg.ptr);
                  addCalls(f.name.name, arg.val);
                }
              },
              ins
          );
        }
        std::visit(
            [&](auto &&arg) {
              using T = std::decay_t<decltype(arg)>;
              if constexpr (std::is_same_v<T, BrTerm>) {
                if (arg.cond) {
                  addCalls(f.name.name, arg.cond->lhs);
                  addCalls(f.name.name, arg.cond->rhs);
                }
              } else if constexpr (std::is_same_v<T, RetTerm>) {
                if (arg.value)
                  addCalls(f.name.name, *arg.value);
              }
            },
            b.term
        );
      }
    }
    for (const auto &d: prog.extDecls) {
      adj[d.name.name];
      if (d.contract) {
        for (const auto &pre: d.contract->pres) {
          addCalls(d.name.name, pre.cond.lhs);
          addCalls(d.name.name, pre.cond.rhs);
        }
        for (const auto &post: d.contract->posts) {
          addCalls(d.name.name, post.cond.lhs);
          addCalls(d.name.name, post.cond.rhs);
        }
      }
    }

    // Tarjan SCC.
    std::unordered_map<std::string, int> idx, low;
    std::unordered_map<std::string, bool> onStack;
    std::vector<std::string> stack;
    int counter = 0;
    std::function<void(const std::string &)> strongConnect = [&](const std::string &v) {
      idx[v] = counter;
      low[v] = counter;
      counter++;
      stack.push_back(v);
      onStack[v] = true;
      for (const auto &w: adj[v]) {
        if (!idx.count(w)) {
          strongConnect(w);
          low[v] = std::min(low[v], low[w]);
        } else if (onStack[w]) {
          low[v] = std::min(low[v], idx[w]);
        }
      }
      if (low[v] == idx[v]) {
        std::vector<std::string> comp;
        while (true) {
          std::string w = stack.back();
          stack.pop_back();
          onStack[w] = false;
          comp.push_back(w);
          if (w == v)
            break;
        }
        bool selfEdge = false;
        for (const auto &x: adj[v])
          if (x == v)
            selfEdge = true;
        if (comp.size() > 1 || selfEdge) {
          std::string cycle;
          for (size_t i = 0; i < comp.size(); ++i) {
            if (i)
              cycle += " -> ";
            cycle += comp[i];
          }
          auto cit = callees_.find(v);
          SourceSpan sp = (cit != callees_.end() && !cit->second.empty()) ? cit->second[0].declSpan
                                                                          : SourceSpan{};
          diags.error("Recursion is not supported in v0.2.2 (cycle: " + cycle + ")", sp);
        }
      }
    };
    for (const auto &kv: adj) {
      if (!idx.count(kv.first))
        strongConnect(kv.first);
    }
  }

  void TypeChecker::checkInitVal(
      const InitVal &iv, const TypePtr &targetType,
      const std::unordered_map<std::string, VarInfo> &vars,
      const std::unordered_map<std::string, SymInfo> &syms, DiagBag &diags
  ) {
    if (iv.kind == InitVal::Kind::Undef)
      return;

    if (iv.kind == InitVal::Kind::Null) {
      if (!targetType || !std::holds_alternative<PtrType>(targetType->v))
        diags.error("'null' can only initialize a pointer (ptr T) variable", iv.span);
      return;
    }

    auto at = TypeUtils::asArray(targetType);
    auto st = TypeUtils::asStruct(targetType);
    auto vt = TypeUtils::asVec(targetType);
    auto tt = TypeUtils::asTensor(targetType);

    if (iv.kind == InitVal::Kind::Aggregate) {
      const auto &elements = std::get<std::vector<InitValPtr>>(iv.value);
      if (at) {
        if (elements.size() != at->size) {
          diags.error(
              "Array initializer length mismatch: expected " + std::to_string(at->size) + ", got " +
                  std::to_string(elements.size()),
              iv.span
          );
          return;
        }
        for (const auto &elem: elements) {
          checkInitVal(*elem, at->elem, vars, syms, diags);
        }
      } else if (st) {
        auto sit = structs_.find(st->name.name);
        if (sit == structs_.end())
          return;
        if (elements.size() != sit->second.fieldList.size()) {
          diags.error(
              "Struct initializer field count mismatch: expected " +
                  std::to_string(sit->second.fieldList.size()) + ", got " +
                  std::to_string(elements.size()),
              iv.span
          );
          return;
        }
        for (size_t i = 0; i < elements.size(); ++i) {
          checkInitVal(*elements[i], sit->second.fieldList[i].second, vars, syms, diags);
        }
      } else if (vt) {
        // Vector brace init: exactly N lane initializers, each of
        // the lane scalar type. BraceElem restrictions (no atom-form inside
        // a brace) are enforced syntactically by the parser.
        if (elements.size() != vt->size) {
          diags.error(
              "Vector initializer lane count mismatch: expected " + std::to_string(vt->size) +
                  ", got " + std::to_string(elements.size()),
              iv.span
          );
          return;
        }
        for (const auto &elem: elements) {
          checkInitVal(*elem, vt->elem, vars, syms, diags);
        }
      } else if (tt) {
        std::uint32_t expectedCount = 0;
        try {
          expectedCount = tensorElementCount(tt->shape);
        } catch (const std::exception &e) {
          diags.error(e.what(), iv.span);
          return;
        }
        if (elements.size() != expectedCount) {
          diags.error(
              "tensor-shape: initializer element count mismatch: expected " +
                  std::to_string(expectedCount) + ", got " + std::to_string(elements.size()),
              iv.span
          );
          return;
        }
        for (const auto &elem: elements) {
          if (elem->kind == InitVal::Kind::Aggregate) {
            diags.error("tensor-shape: tensor initializer must be a flat list", elem->span);
            continue;
          }
          if (elem->kind == InitVal::Kind::Undef) {
            diags.error(
                "tensor-init: partially undefined tensor values are not supported", elem->span
            );
            continue;
          }
          checkInitVal(*elem, tt->elem, vars, syms, diags);
        }
      } else {
        diags.error("Aggregate initializer for non-aggregate type", iv.span);
      }
      return;
    }

    // Scalar Init: Int, Sym, Local
    // Broadcast check: scalar must be compatible with ALL leaf scalar elements.
    std::vector<TypePtr> targetLeaves;
    std::function<void(const TypePtr &)> collect = [&](const TypePtr &t) {
      if (auto inner_at = TypeUtils::asArray(t)) {
        collect(inner_at->elem);
      } else if (auto inner_st = TypeUtils::asStruct(t)) {
        auto sit = structs_.find(inner_st->name.name);
        if (sit != structs_.end()) {
          for (const auto &fld: sit->second.fieldList)
            collect(fld.second);
        }
      } else if (auto inner_vt = TypeUtils::asVec(t)) {
        // Vector broadcast: scalar init type must match the lane
        // scalar type. Recursing on `elem` lets the existing leaf-check
        // reject `let %v: <4> i32 = 1.5;` etc.
        collect(inner_vt->elem);
      } else if (auto inner_tt = TypeUtils::asTensor(t)) {
        targetLeaves.push_back(inner_tt->elem);
      } else {
        targetLeaves.push_back(t);
      }
    };
    collect(targetType);

    TypePtr initType = nullptr;
    if (iv.kind == InitVal::Kind::Int) {
      // Literals must fit in the target type's range
      for (const auto &leaf: targetLeaves) {
        auto bits = TypeUtils::getIntBitWidth(leaf);
        if (!bits) {
          diags.error("Cannot initialize non-integer with integer literal", iv.span);
          return;
        }
        checkLiteralRange(std::get<IntLit>(iv.value).value, *bits, iv.span, diags);
      }
      return;
    } else if (iv.kind == InitVal::Kind::Float) {
      for (const auto &leaf: targetLeaves) {
        if (!std::holds_alternative<FloatType>(leaf->v)) {
          diags.error("Cannot initialize non-float with float literal", iv.span);
          return;
        }
      }
      return;
    } else if (iv.kind == InitVal::Kind::Sym) {
      auto it = syms.find(std::get<SymId>(iv.value).name);
      if (it != syms.end())
        initType = it->second.type;
    } else if (iv.kind == InitVal::Kind::Local) {
      auto it = vars.find(std::get<LocalId>(iv.value).name);
      if (it != vars.end())
        initType = it->second.type;
      else {
        diags.error(
            "Undeclared local in initializer: " + std::get<LocalId>(iv.value).name, iv.span
        );
        return;
      }
    }

    if (initType) {
      // SPEC §3.4.2: whole-aggregate-copy via a LocalId/SymId is forbidden.
      // Arrays ([N] T) and structs (@S) are aggregates; copying them requires
      // explicit field-by-field assignment (see §6.6). The forbidden case is
      // when the *atom's type* is itself an aggregate — it means we are trying
      // to initialise an aggregate target with an aggregate-typed source
      // variable (whole-aggregate-copy).
      //
      // The *broadcast* exception is still valid: a scalar atom (initType not
      // aggregate) initialising an array target broadcasts the scalar value to
      // every element (§3.4.2, "Broadcast"). Scalars, pointers, and vectors
      // all map to a single SMT value term and are always safe to copy.
      if (TypeUtils::asArray(initType) || TypeUtils::asStruct(initType)) {
        diags.error(
            "Whole-aggregate-copy initializer is not allowed: use BraceInit "
            "or 'undef' to initialise array/struct types",
            iv.span
        );
        return;
      }
      if (TypeUtils::isTensor(initType)) {
        if (!TypeUtils::areTypesEqual(targetType, initType))
          diags.error("tensor-type: tensor initializer shape or element type mismatch", iv.span);
        return;
      }
      // Whole-vector / whole-pointer init: if initType matches the target type
      // exactly, accept without descending. This covers `let %w: <4> i32 = %v;`
      // and `let %p: ptr T = %q;`.
      if (TypeUtils::areTypesEqual(targetType, initType))
        return;
      for (const auto &leaf: targetLeaves) {
        if (!TypeUtils::areTypesEqual(leaf, initType)) {
          diags.error("Type mismatch in initializer", iv.span);
          return;
        }
      }
    }
  }

  void TypeChecker::checkFunction(const FunDecl &f, TypeAnnotations &ann, DiagBag &diags) {
    std::unordered_map<std::string, VarInfo> vars;
    std::unordered_map<std::string, SymInfo> syms;

    // Up-front type validation so structural errors fire before
    // any expression check that would also touch the type.
    validateTypeWF(f.retType, diags, false, false, false, false, tensorProfile_);
    for (const auto &p: f.params) {
      validateTypeWF(p.type, diags, false, false, false, false, tensorProfile_);
      vars[p.name.name] = VarInfo{p.type, false, true, p.span};
    }
    for (const auto &s: f.syms) {
      // v0.2.0 §3.4.1: `sym` declarations of pointer type are not allowed.
      // Pointer values must come from `addr` or parameters.
      if (s.type && std::holds_alternative<PtrType>(s.type->v)) {
        diags.error("sym of pointer type is not allowed in v0.2.0: " + s.name.name, s.span);
      }
      // Aggregate symbols are likewise rejected: the language
      // has no consumer syntax for them (whole-aggregate-copy init is
      // banned and syms take no element access), so a declarable
      // array/struct sym would be dead weight. Planned for a later
      // version (SPEC §13).
      if (s.type && (std::holds_alternative<ArrayType>(s.type->v) ||
                     std::holds_alternative<StructType>(s.type->v))) {
        diags.error("sym of array/struct type is not allowed: " + s.name.name, s.span);
      }
      validateTypeWF(s.type, diags, false, false, false, false, tensorProfile_);
      syms[s.name.name] = SymInfo{s.type, s.kind, s.span};
    }
    for (const auto &l: f.lets) {
      if (vars.count(l.name.name)) {
        diags.error("Duplicate name: " + l.name.name, l.span);
      }
      validateTypeWF(l.type, diags, false, false, false, false, tensorProfile_);
      vars[l.name.name] = VarInfo{l.type, l.isMutable, false, l.span};
      if (l.init) {
        checkInitVal(*l.init, l.type, vars, syms, diags);
      }
    }

    CFG::build(f, diags);

    auto retBits = TypeUtils::getIntBitWidth(f.retType);
    bool isRetFloat = f.retType && std::holds_alternative<FloatType>(f.retType->v);
    bool isRetPtr = f.retType && std::holds_alternative<PtrType>(f.retType->v);
    bool isRetVec = f.retType && std::holds_alternative<VecType>(f.retType->v);
    bool isRetTensor = TypeUtils::isTensor(f.retType);

    if (!retBits && !isRetFloat && !isRetPtr && !isRetVec && !isRetTensor)
      diags.error("Return type must be a scalar, pointer, vector, or tensor type", f.span);

    for (const auto &b: f.blocks) {
      for (const auto &ins: b.instrs) {
        std::visit(
            [&](auto &&arg) {
              using T = std::decay_t<decltype(arg)>;
              if constexpr (std::is_same_v<T, AssignInstr>) {
                auto it = vars.find(arg.lhs.base.name);
                if (it != vars.end() && !it->second.isMutable) {
                  diags.error("Assignment to immutable local: " + arg.lhs.base.name, arg.lhs.span);
                }
                auto lt = typeOfLValue(arg.lhs, vars, syms, diags);
                if (lt) {
                  if (std::holds_alternative<PtrType>(lt->v)) {
                    // Pointer assignment: RHS must be ptr T or null
                    Ty rhsTy = typeOfExpr(arg.rhs, vars, syms, ann, diags, std::nullopt, lt);
                    if (!rhsTy.isPtr())
                      diags.error(
                          "Expected pointer expression on RHS of pointer assignment", arg.rhs.span
                      );
                    else if (!TypeUtils::areTypesEqual(rhsTy.ptrType(), lt))
                      diags.error("Pointer type mismatch in assignment", arg.rhs.span);
                  } else if (auto lvt = TypeUtils::asVec(lt)) {
                    // Vector assignment: whole-vector copy or lane
                    // write. For lane write (LHS has a trailing [i] access
                    // and i resolves to a vector lane), the LHS lt is the
                    // lane element type — which is handled by the integer/
                    // float branch below. So this branch only fires when
                    // the LHS is a whole vector lvalue. RHS must be the
                    // same vector type (literal broadcast is handled by
                    // typeOfExpr's `expected` plumbing).
                    Ty rhsTy = typeOfExpr(arg.rhs, vars, syms, ann, diags, std::nullopt, lt);
                    if (!rhsTy.isVec())
                      diags.error("Expected vector expression on RHS", arg.rhs.span);
                    else if (!TypeUtils::areTypesEqual(rhsTy.vecType(), lt))
                      diags.error("Vector type mismatch in assignment", arg.rhs.span);
                    (void) lvt;
                  } else if (TypeUtils::isTensor(lt)) {
                    Ty rhsTy = typeOfExpr(arg.rhs, vars, syms, ann, diags, std::nullopt);
                    if (!rhsTy.isTensor())
                      diags.error(
                          "tensor-type: expected tensor expression on assignment RHS", arg.rhs.span
                      );
                    else if (!TypeUtils::areTypesEqual(rhsTy.tensorType(), lt))
                      diags.error(
                          "tensor-type: tensor shape or element type mismatch in assignment",
                          arg.rhs.span
                      );
                  } else if (std::holds_alternative<ArrayType>(lt->v) ||
                             std::holds_alternative<StructType>(lt->v)) {
                    diags.error("Aggregate-typed assignment is not supported", arg.lhs.span);
                  } else {
                    std::optional<uint32_t> expected;
                    bool isFloat = false;
                    if (auto lb = TypeUtils::getIntBitWidth(lt)) {
                      expected = *lb;
                    } else if (std::holds_alternative<FloatType>(lt->v)) {
                      auto &ft = std::get<FloatType>(lt->v);
                      expected = (ft.kind == FloatType::Kind::F32) ? 32 : 64;
                      isFloat = true;
                    } else {
                      diags.error(
                          "LHS of assignment must be a scalar or pointer type", arg.lhs.span
                      );
                    }
                    if (expected) {
                      Ty rhsTy = typeOfExpr(arg.rhs, vars, syms, ann, diags, expected);
                      if (isFloat) {
                        if (!rhsTy.isFloat())
                          diags.error("Expected float expression on RHS", arg.rhs.span);
                        else if (rhsTy.floatBits() != *expected)
                          diags.error("Float width mismatch in assignment", arg.rhs.span);
                      } else {
                        if (!rhsTy.isBV())
                          diags.error("Expected integer expression on RHS", arg.rhs.span);
                        else if (rhsTy.bvBits() != *expected)
                          diags.error("Bitwidth mismatch in assignment", arg.rhs.span);
                      }
                    }
                  }
                }
              } else if constexpr (std::is_same_v<T, AssumeInstr>) {
                checkCond(arg.cond, vars, syms, ann, diags);
              } else if constexpr (std::is_same_v<T, RequireInstr>) {
                checkCond(arg.cond, vars, syms, ann, diags);
              } else if constexpr (std::is_same_v<T, StoreInstr>) {
                // store <ptr>, <val>: ptr must be ptr T, val must be T
                Ty ptrTy = typeOfExpr(arg.ptr, vars, syms, ann, diags, std::nullopt);
                if (!ptrTy.isPtr()) {
                  diags.error("First operand of 'store' must be a pointer (ptr T)", arg.ptr.span);
                } else {
                  auto &pt = std::get<PtrType>(ptrTy.ptrType()->v);
                  TypePtr valueTy = pt.pointee;
                  if (auto vb = TypeUtils::getIntBitWidth(valueTy)) {
                    Ty valTy = typeOfExpr(arg.val, vars, syms, ann, diags, *vb);
                    if (!valTy.isBV() || valTy.bvBits() != *vb)
                      diags.error("'store' value type mismatch", arg.val.span);
                  } else if (valueTy && std::holds_alternative<FloatType>(valueTy->v)) {
                    auto &ft = std::get<FloatType>(valueTy->v);
                    uint32_t bits = (ft.kind == FloatType::Kind::F32) ? 32u : 64u;
                    Ty valTy = typeOfExpr(arg.val, vars, syms, ann, diags, bits);
                    if (!valTy.isFloat() || valTy.floatBits() != bits)
                      diags.error("'store' value float type mismatch", arg.val.span);
                  } else if (valueTy && std::holds_alternative<PtrType>(valueTy->v)) {
                    Ty valTy = typeOfExpr(arg.val, vars, syms, ann, diags, std::nullopt, valueTy);
                    if (!valTy.isPtr() || !TypeUtils::areTypesEqual(valTy.ptrType(), valueTy))
                      diags.error("'store' pointer value type mismatch", arg.val.span);
                  } else {
                    diags.error("'store' pointee type not supported", arg.span);
                  }
                }
              }
            },
            ins
        );
      }
      std::visit(
          [&](auto &&arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, BrTerm>) {
              if (arg.isConditional && arg.cond)
                checkCond(*arg.cond, vars, syms, ann, diags);
            } else if constexpr (std::is_same_v<T, RetTerm>) {
              if (arg.value) {
                Ty rhsTy;
                if (retBits) {
                  rhsTy = typeOfExpr(*arg.value, vars, syms, ann, diags, *retBits);
                  if (!rhsTy.isBV())
                    diags.error("Expected integer return value", arg.value->span);
                  else if (rhsTy.bvBits() != *retBits)
                    diags.error("Return bitwidth mismatch", arg.value->span);
                } else if (isRetFloat) {
                  auto const &ft = std::get<FloatType>(f.retType->v);
                  uint32_t bits = (ft.kind == FloatType::Kind::F32) ? 32 : 64;
                  rhsTy = typeOfExpr(*arg.value, vars, syms, ann, diags, bits);
                  if (!rhsTy.isFloat())
                    diags.error("Expected float return value", arg.value->span);
                  else if (rhsTy.floatBits() != bits)
                    diags.error("Return float width mismatch", arg.value->span);
                } else if (isRetPtr) {
                  rhsTy = typeOfExpr(*arg.value, vars, syms, ann, diags, std::nullopt, f.retType);
                  if (!rhsTy.isPtr())
                    diags.error("Expected pointer return value", arg.value->span);
                  else if (!TypeUtils::areTypesEqual(rhsTy.ptrType(), f.retType))
                    diags.error("Return pointer type mismatch", arg.value->span);
                } else if (isRetVec) {
                  // Vector return: RHS must be the same vector type.
                  rhsTy = typeOfExpr(*arg.value, vars, syms, ann, diags, std::nullopt, f.retType);
                  if (!rhsTy.isVec())
                    diags.error("Expected vector return value", arg.value->span);
                  else if (!TypeUtils::areTypesEqual(rhsTy.vecType(), f.retType))
                    diags.error("Return vector type mismatch", arg.value->span);
                } else if (isRetTensor) {
                  rhsTy = typeOfExpr(*arg.value, vars, syms, ann, diags, std::nullopt);
                  if (!rhsTy.isTensor())
                    diags.error("tensor-type: expected tensor return value", arg.value->span);
                  else if (!TypeUtils::areTypesEqual(rhsTy.tensorType(), f.retType))
                    diags.error(
                        "tensor-type: tensor return shape or element type mismatch", arg.value->span
                    );
                }
              } else if (!arg.value) {
                diags.error("Missing return value", arg.span);
              }
            }
          },
          b.term
      );
    }
  }

  TypePtr TypeChecker::typeOfLValue(
      const LValue &lv, const std::unordered_map<std::string, VarInfo> &vars,
      const std::unordered_map<std::string, SymInfo> &syms, DiagBag &diags
  ) {
    auto it = vars.find(lv.base.name);
    if (it == vars.end()) {
      diags.error("Undeclared local: " + lv.base.name, lv.base.span);
      return nullptr;
    }
    TypePtr cur = it->second.type;
    for (const auto &acc: lv.accesses) {
      if (TypeUtils::asTensor(cur)) {
        diags.error("tensor-type: tensor elements cannot be accessed as lvalues", lv.span);
        return nullptr;
      }
      if (auto ai = std::get_if<AccessIndex>(&acc)) {
        if (auto vt = TypeUtils::asVec(cur)) {
          // Vector lane access: lv[i] : T_elem. Bounds-check
          // constant index now (§7.6 rule 20 statically); dynamic index
          // bounds are runtime UB.
          if (auto il = std::get_if<IntLit>(&ai->index)) {
            if (il->value < 0 || static_cast<std::uint64_t>(il->value) >= vt->size) {
              diags.error(
                  "Vector lane index " + std::to_string(il->value) + " out of bounds for <" +
                      std::to_string(vt->size) + "> vector",
                  ai->span
              );
              return nullptr;
            }
          }
          checkIndex(ai->index, vars, syms, diags);
          cur = vt->elem;
          continue;
        }
        auto at = TypeUtils::asArray(cur);
        if (!at) {
          diags.error("Indexing non-array", ai->span);
          return nullptr;
        }
        checkIndex(ai->index, vars, syms, diags);
        cur = at->elem;
      } else if (auto af = std::get_if<AccessField>(&acc)) {
        auto st = TypeUtils::asStruct(cur);
        if (!st) {
          diags.error("Field access on non-struct", af->span);
          return nullptr;
        }
        auto sit = structs_.find(st->name.name);
        if (sit == structs_.end()) {
          diags.error("Unknown struct type: " + st->name.name, af->span);
          return nullptr;
        }
        auto fit = sit->second.fields.find(af->field);
        if (fit == sit->second.fields.end()) {
          diags.error("Unknown field '" + af->field + "' in struct " + st->name.name, af->span);
          return nullptr;
        }
        cur = fit->second;
      }
    }
    return cur;
  }

  void TypeChecker::checkIndex(
      const Index &idx, const std::unordered_map<std::string, VarInfo> &vars,
      const std::unordered_map<std::string, SymInfo> &syms, DiagBag &diags
  ) {
    if (std::holds_alternative<IntLit>(idx))
      return;
    auto id = std::get<LocalOrSymId>(idx);
    if (auto lid = std::get_if<LocalId>(&id)) {
      auto it = vars.find(lid->name);
      if (it == vars.end()) {
        diags.error("Undeclared local index: " + lid->name, lid->span);
        return;
      }
      if (!TypeUtils::getIntBitWidth(it->second.type))
        diags.error("Non-integer index", lid->span);
    } else {
      auto sid = std::get_if<SymId>(&id);
      auto it = syms.find(sid->name);
      if (it == syms.end()) {
        diags.error("Undeclared symbol index: " + sid->name, sid->span);
        return;
      }
      if (!TypeUtils::getIntBitWidth(it->second.type))
        diags.error("Non-integer symbol index", sid->span);
    }
  }

  Ty TypeChecker::typeOfExpr(
      const Expr &e, const std::unordered_map<std::string, VarInfo> &vars,
      const std::unordered_map<std::string, SymInfo> &syms, TypeAnnotations &ann, DiagBag &diags,
      std::optional<std::uint32_t> expectedBits, TypePtr ptrCtx
  ) {
    auto t = typeOfAtom(e.first, vars, syms, ann, diags, expectedBits, ptrCtx);
    for (const auto &tail: e.rest) {
      if (t.isPtr()) {
        // Pointer arithmetic: ptr T +/- iN → ptr T,  ptr T - ptr T → i64
        auto ti = typeOfAtom(tail.atom, vars, syms, ann, diags, 64);
        if (tail.op == AddOp::Minus && ti.isPtr()) {
          // ptr - ptr → i64
          if (!TypeUtils::areTypesEqual(t.ptrType(), ti.ptrType()))
            diags.error("Pointer subtraction requires same pointee type", tail.span);
          t = Ty{Ty::BVTy{64}};
        } else if (ti.isBV()) {
          // ptr +/- int → ptr (type unchanged)
        } else {
          diags.error("Pointer arithmetic requires integer offset or same-type pointer", tail.span);
        }
      } else {
        auto ti = typeOfAtom(
            tail.atom, vars, syms, ann, diags,
            t.isBV()      ? std::optional(t.bvBits())
            : t.isFloat() ? std::optional(t.floatBits())
                          : expectedBits
        );
        if (t.isTensor() || ti.isTensor()) {
          diags.error(
              "tensor-type: ordinary expression operators do not lift to tensors", tail.span
          );
          continue;
        }
        // Vector chain: both atoms must agree on vector type, but
        // a scalar literal atom (`CoefAtom` of IntLit/FloatLit) broadcasts
        // to the running vector type, mirroring §6.11.
        if (t.isVec() || ti.isVec()) {
          bool tIsLitScalar =
              (t.isBV() || t.isFloat()) && std::holds_alternative<CoefAtom>(e.first.v) &&
              (std::holds_alternative<IntLit>(std::get<CoefAtom>(e.first.v).coef) ||
               std::holds_alternative<FloatLit>(std::get<CoefAtom>(e.first.v).coef));
          bool tiIsLitScalar =
              (ti.isBV() || ti.isFloat()) && std::holds_alternative<CoefAtom>(tail.atom.v) &&
              (std::holds_alternative<IntLit>(std::get<CoefAtom>(tail.atom.v).coef) ||
               std::holds_alternative<FloatLit>(std::get<CoefAtom>(tail.atom.v).coef));
          if (t.isVec() && ti.isVec()) {
            if (!TypeUtils::areTypesEqual(t.vecType(), ti.vecType()))
              diags.error("Vector arithmetic type mismatch", tail.span);
          } else if (t.isVec() && tiIsLitScalar) {
            // Literal broadcasts; chain stays vector. Carry t forward.
          } else if (ti.isVec() && tIsLitScalar) {
            // Symmetric — but we'd need to convert t to the vector type;
            // simpler to require Coef on the left for literal broadcast.
            diags.error("Vector arithmetic: literal must appear on the left", tail.span);
          } else {
            diags.error("Vector arithmetic type mismatch", tail.span);
          }
          continue;
        }
        if (t.isBV() && ti.isBV() && t.bvBits() != ti.bvBits())
          diags.error("Bitwidth mismatch", tail.span);
        if (t.isFloat() && ti.isFloat() && t.floatBits() != ti.floatBits())
          diags.error("Float width mismatch", tail.span);
        if (t.isBV() != ti.isBV() || t.isFloat() != ti.isFloat())
          diags.error("Mixed integer/float arithmetic not allowed", tail.span);
      }
    }
    return t;
  }

  Ty TypeChecker::typeOfTensorAtom(
      const TensorAtom &atom, const std::unordered_map<std::string, VarInfo> &vars,
      const std::unordered_map<std::string, SymInfo> &syms, DiagBag &diags
  ) {
    auto requireOperandCount = [&](std::size_t count) {
      if (atom.operands.size() == count)
        return true;
      diags.error("tensor-type: wrong operand count for tensor operator", atom.span);
      return false;
    };
    auto namedType = [&](const Coef &operand) -> TypePtr {
      const auto id = std::get_if<LocalOrSymId>(&operand);
      if (!id)
        return nullptr;
      return std::visit(
          [&](const auto &name) -> TypePtr {
            if constexpr (std::is_same_v<std::decay_t<decltype(name)>, LocalId>) {
              const auto it = vars.find(name.name);
              return it == vars.end() ? nullptr : it->second.type;
            } else {
              const auto it = syms.find(name.name);
              return it == syms.end() ? nullptr : it->second.type;
            }
          },
          *id
      );
    };
    auto isTensorScalar = [](const TypePtr &type) {
      const auto bits = TypeUtils::getScalarBitWidth(type);
      return bits && *bits == 32;
    };
    auto resolveOperand = [&](std::size_t index, TypePtr expectedElem = nullptr) {
      const auto expectedBits = expectedElem ? TypeUtils::getScalarBitWidth(expectedElem)
                                             : std::optional<std::uint32_t>{};
      return typeOfCoef(atom.operands.at(index), vars, syms, diags, expectedBits);
    };
    auto makeResult = [&](const Shape &shape, const TypePtr &elementType) -> Ty {
      if (shape.empty()) {
        atom.resolvedType = elementType;
        if (const auto bits = TypeUtils::getIntBitWidth(elementType))
          return Ty{Ty::BVTy{*bits}};
        if (const auto bits = TypeUtils::getFloatBitWidth(elementType))
          return Ty{Ty::FloatTy{*bits}};
        return Ty{std::monostate{}};
      }
      TypePtr result = std::make_shared<Type>();
      result->v = TensorType{shape, elementType, atom.span};
      result->span = atom.span;
      atom.resolvedType = result;
      return Ty{Ty::TensorTy{std::move(result)}};
    };
    auto validateShape = [&](const Shape &shape) {
      try {
        (void) tensorExecutionTileShape(shape, tensorProfile_);
      } catch (const std::exception &error) {
        diags.error(error.what(), atom.span);
        return false;
      }
      return true;
    };
    auto sourceShapeAndElement = [&](const TypePtr &type, Shape &shape, TypePtr &element) {
      if (const auto tensor = TypeUtils::asTensor(type)) {
        shape = tensor->shape;
        element = tensor->elem;
        return true;
      }
      if (TypeUtils::isScalar(type)) {
        shape.clear();
        element = type;
        return true;
      }
      diags.error("tensor-type: operand must be a scalar or tensor", atom.span);
      return false;
    };
    auto operandWithElemHint = [&](std::size_t index, TypePtr hint) {
      TypePtr type = namedType(atom.operands.at(index));
      if (!type)
        type = resolveOperand(index, hint);
      return type;
    };
    auto typeFromElement = [&](const TypePtr &element) -> Ty {
      if (const auto bits = TypeUtils::getIntBitWidth(element))
        return Ty{Ty::BVTy{*bits}};
      if (const auto bits = TypeUtils::getFloatBitWidth(element))
        return Ty{Ty::FloatTy{*bits}};
      return Ty{std::monostate{}};
    };
    auto checkedAxis = [&](std::int64_t axis, std::uint32_t rank, std::uint32_t &normalized) {
      try {
        normalized = normalizeTensorAxis(axis, rank);
        return true;
      } catch (const std::exception &error) {
        diags.error(error.what(), atom.span);
        return false;
      }
    };

    if (atom.op == TensorOp::Iota) {
      if (!requireOperandCount(0) || !validateShape(atom.attribute))
        return Ty{std::monostate{}};
      std::uint32_t axis = 0;
      if (!checkedAxis(atom.axis, static_cast<std::uint32_t>(atom.attribute.size()), axis))
        return Ty{std::monostate{}};
      (void) axis;
      TypePtr i32 = std::make_shared<Type>();
      i32->v = IntType{IntType::Kind::I32, std::nullopt, atom.span};
      i32->span = atom.span;
      return makeResult(atom.attribute, i32);
    }

    if (atom.op == TensorOp::Extract || atom.op == TensorOp::Expand ||
        atom.op == TensorOp::Broadcast || atom.op == TensorOp::Transpose ||
        atom.op == TensorOp::Sum || atom.op == TensorOp::ReduceMin ||
        atom.op == TensorOp::ReduceMax || atom.op == TensorOp::Softmax ||
        atom.op == TensorOp::Exp) {
      if (!requireOperandCount(1))
        return Ty{std::monostate{}};
      TypePtr type = resolveOperand(0);
      Shape sourceShape;
      TypePtr elementType;
      if (!sourceShapeAndElement(type, sourceShape, elementType))
        return Ty{std::monostate{}};
      if (!isTensorScalar(elementType)) {
        diags.error("tensor-type: tensor operands require i32 or f32 elements", atom.span);
        return Ty{std::monostate{}};
      }

      if (atom.op == TensorOp::Extract) {
        const auto tensor = TypeUtils::asTensor(type);
        if (!tensor) {
          diags.error("tensor-type: extract requires a tensor operand", atom.span);
          return Ty{std::monostate{}};
        }
        if (atom.attribute.size() != sourceShape.size()) {
          diags.error("tensor-shape: extraction coordinate rank mismatch", atom.span);
          return Ty{std::monostate{}};
        }
        for (std::size_t i = 0; i < sourceShape.size(); ++i) {
          if (atom.attribute[i] >= sourceShape[i]) {
            diags.error("tensor-shape: extraction coordinate is out of bounds", atom.span);
            return Ty{std::monostate{}};
          }
        }
        atom.resolvedType = elementType;
        return typeFromElement(elementType);
      }

      if (atom.op == TensorOp::Expand) {
        Shape output = sourceShape;
        const auto outputRank = static_cast<std::uint32_t>(sourceShape.size() + 1);
        std::uint32_t axis = 0;
        if (!checkedAxis(atom.axis, outputRank, axis))
          return Ty{std::monostate{}};
        output.insert(output.begin() + axis, 1);
        if (!validateShape(output))
          return Ty{std::monostate{}};
        return makeResult(output, elementType);
      }

      if (atom.op == TensorOp::Broadcast) {
        if (!validateShape(atom.attribute))
          return Ty{std::monostate{}};
        if (atom.attribute.size() < sourceShape.size()) {
          diags.error("tensor-broadcast: explicit broadcast cannot reduce rank", atom.span);
          return Ty{std::monostate{}};
        }
        try {
          const auto inferred = tensorBroadcastShape(sourceShape, atom.attribute);
          if (inferred != atom.attribute) {
            diags.error(
                "tensor-broadcast: broadcast cannot shrink a non-singleton dimension", atom.span
            );
            return Ty{std::monostate{}};
          }
        } catch (const std::exception &error) {
          diags.error(error.what(), atom.span);
          return Ty{std::monostate{}};
        }
        return makeResult(atom.attribute, elementType);
      }

      if (atom.op == TensorOp::Transpose) {
        if (!TypeUtils::asTensor(type)) {
          diags.error("tensor-type: transpose requires a tensor operand", atom.span);
          return Ty{std::monostate{}};
        }
        if (atom.attribute.size() != sourceShape.size()) {
          diags.error("tensor-permutation: permutation rank mismatch", atom.span);
          return Ty{std::monostate{}};
        }
        std::vector<bool> seen(sourceShape.size(), false);
        Shape output(sourceShape.size());
        for (std::size_t i = 0; i < atom.attribute.size(); ++i) {
          const auto sourceAxis = atom.attribute[i];
          if (sourceAxis >= sourceShape.size() || seen[sourceAxis]) {
            diags.error("tensor-permutation: axes must form a permutation", atom.span);
            return Ty{std::monostate{}};
          }
          seen[sourceAxis] = true;
          output[i] = sourceShape[sourceAxis];
        }
        return makeResult(output, elementType);
      }

      if (atom.op == TensorOp::Sum || atom.op == TensorOp::ReduceMin ||
          atom.op == TensorOp::ReduceMax) {
        if (!TypeUtils::asTensor(type)) {
          diags.error("tensor-type: reduction requires a tensor operand", atom.span);
          return Ty{std::monostate{}};
        }
        std::uint32_t axis = 0;
        if (!checkedAxis(atom.axis, static_cast<std::uint32_t>(sourceShape.size()), axis))
          return Ty{std::monostate{}};
        (void) axis;
        Shape output;
        try {
          output = tensorReduceShape(sourceShape, atom.axis, atom.keepDims);
        } catch (const std::exception &error) {
          diags.error(error.what(), atom.span);
          return Ty{std::monostate{}};
        }
        return makeResult(output, elementType);
      }

      if (atom.op == TensorOp::Softmax) {
        if (!TypeUtils::asTensor(type)) {
          diags.error("tensor-type: softmax requires a tensor operand", atom.span);
          return Ty{std::monostate{}};
        }
        if (!TypeUtils::getFloatBitWidth(elementType)) {
          diags.error("tensor-type: softmax requires f32 elements", atom.span);
          return Ty{std::monostate{}};
        }
        std::uint32_t axis = 0;
        if (!checkedAxis(atom.axis, static_cast<std::uint32_t>(sourceShape.size()), axis))
          return Ty{std::monostate{}};
        (void) axis;
        return makeResult(sourceShape, elementType);
      }

      if (atom.op == TensorOp::Exp) {
        if (!TypeUtils::getFloatBitWidth(elementType)) {
          diags.error("tensor-type: exp requires f32 elements", atom.span);
          return Ty{std::monostate{}};
        }
        return makeResult(sourceShape, elementType);
      }
    }

    if (atom.op == TensorOp::Rem) {
      if (!requireOperandCount(2))
        return Ty{std::monostate{}};
      TypePtr leftType = resolveOperand(0);
      TypePtr rightType = resolveOperand(1);
      const auto left = TypeUtils::asTensor(leftType);
      const auto right = TypeUtils::asTensor(rightType);
      if (!left || !right) {
        diags.error("tensor-type: rem requires two tensor operands", atom.span);
        return Ty{std::monostate{}};
      }
      if (!TypeUtils::getFloatBitWidth(left->elem) ||
          TypeUtils::getFloatBitWidth(left->elem) != 32 ||
          TypeUtils::getFloatBitWidth(right->elem) != 32 || left->shape != right->shape) {
        diags.error("tensor-v4: rem requires equal-shaped f32 tensor operands", atom.span);
        return Ty{std::monostate{}};
      }
      if (!validateShape(left->shape))
        return Ty{std::monostate{}};
      return makeResult(left->shape, left->elem);
    }

    if (atom.op == TensorOp::Add || atom.op == TensorOp::Sub || atom.op == TensorOp::Mul ||
        atom.op == TensorOp::Div || atom.op == TensorOp::Min || atom.op == TensorOp::Max ||
        atom.op == TensorOp::And || atom.op == TensorOp::Or || atom.op == TensorOp::Xor) {
      if (!requireOperandCount(2))
        return Ty{std::monostate{}};
      TypePtr hint = namedType(atom.operands[0]);
      if (!TypeUtils::asTensor(hint))
        hint = namedType(atom.operands[1]);
      if (!TypeUtils::asTensor(hint)) {
        diags.error("tensor-type: at least one binary operand must be a tensor", atom.span);
        return Ty{std::monostate{}};
      }
      TypePtr hintedElement = TypeUtils::asTensor(hint)->elem;
      TypePtr leftType = operandWithElemHint(0, hintedElement);
      TypePtr rightType = operandWithElemHint(1, hintedElement);
      Shape leftShape;
      Shape rightShape;
      TypePtr leftElement;
      TypePtr rightElement;
      if (!sourceShapeAndElement(leftType, leftShape, leftElement) ||
          !sourceShapeAndElement(rightType, rightShape, rightElement))
        return Ty{std::monostate{}};
      if (!isTensorScalar(leftElement) || !isTensorScalar(rightElement)) {
        diags.error("tensor-type: tensor operands require i32 or f32 elements", atom.span);
        return Ty{std::monostate{}};
      }
      if (!TypeUtils::areTypesEqual(leftElement, rightElement)) {
        diags.error("tensor-type: binary operands must have the same element type", atom.span);
        return Ty{std::monostate{}};
      }
      const bool integer = TypeUtils::getIntBitWidth(leftElement).has_value();
      if ((atom.op == TensorOp::Div && integer) ||
          ((atom.op == TensorOp::And || atom.op == TensorOp::Or || atom.op == TensorOp::Xor) &&
           !integer)) {
        diags.error("tensor-type: operator is not defined for this element type", atom.span);
        return Ty{std::monostate{}};
      }
      Shape output;
      try {
        output = tensorBroadcastShape(leftShape, rightShape);
      } catch (const std::exception &error) {
        diags.error(error.what(), atom.span);
        return Ty{std::monostate{}};
      }
      if (output.empty()) {
        diags.error(
            "tensor-type: scalar-only binary operations use ordinary SIR operators", atom.span
        );
        return Ty{std::monostate{}};
      }
      if (!validateShape(output))
        return Ty{std::monostate{}};
      return makeResult(output, leftElement);
    }

    diags.error("tensor-type: unsupported tensor operator", atom.span);
    return Ty{std::monostate{}};
  }

  Ty TypeChecker::typeOfAtom(
      const Atom &a, const std::unordered_map<std::string, VarInfo> &vars,
      const std::unordered_map<std::string, SymInfo> &syms, TypeAnnotations &ann, DiagBag &diags,
      std::optional<std::uint32_t> expectedBits, TypePtr ptrCtx
  ) {
    return std::visit(
        [&](auto &&arg) -> Ty {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, OpAtom>) {
            auto rt = typeOfLValue(arg.rval, vars, syms, diags);
            auto rb = TypeUtils::getIntBitWidth(rt);

            // Vector OpAtom: Coef can be a vector LValue of the
            // same type, or a literal that broadcasts to the vector.
            if (auto vt = TypeUtils::asVec(rt)) {
              if (auto lsid = std::get_if<LocalOrSymId>(&arg.coef)) {
                // Vector coef must match rval's vector type exactly.
                std::string nm = std::visit([](auto &&v) { return v.name; }, *lsid);
                TypePtr ct;
                if (std::holds_alternative<LocalId>(*lsid)) {
                  auto vit = vars.find(nm);
                  if (vit != vars.end())
                    ct = vit->second.type;
                } else {
                  auto sit = syms.find(nm);
                  if (sit != syms.end())
                    ct = sit->second.type;
                }
                if (!TypeUtils::areTypesEqual(ct, rt))
                  diags.error("Vector op: coefficient type must match", arg.span);
              }
              // Literal coef (IntLit/FloatLit): broadcasts to vector. No
              // type check needed beyond elem-kind being plausible.
              return Ty{Ty::VecTy{rt}};
            }

            if (rb) {
              // Integer case
              auto targetBits = rb; // Use rval bits as truth
              auto ct = typeOfCoef(arg.coef, vars, syms, diags, targetBits);
              auto cb = TypeUtils::getIntBitWidth(ct);
              if (cb && rb && *cb != *rb)
                diags.error("Bitwidth mismatch in operation", arg.span);
              else if (!cb && ct)
                diags.error("Coefficient must be integer type in integer operation", arg.span);
              return Ty{Ty::BVTy{rb.value()}};
            } else if (rt && std::holds_alternative<FloatType>(rt->v)) {
              // Float case
              auto &ft = std::get<FloatType>(rt->v);
              uint32_t bits = (ft.kind == FloatType::Kind::F32) ? 32 : 64;

              if (arg.op != AtomOpKind::Mul && arg.op != AtomOpKind::Div &&
                  arg.op != AtomOpKind::Mod) {
                diags.error("Invalid operator for float type", arg.span);
              }

              auto ct = typeOfCoef(arg.coef, vars, syms, diags, bits);
              if (!ct || !std::holds_alternative<FloatType>(ct->v)) {
                diags.error("Coefficient must be float", arg.span);
              } else {
                auto &cft = std::get<FloatType>(ct->v);
                uint32_t cbits = (cft.kind == FloatType::Kind::F32) ? 32 : 64;
                if (cbits != bits)
                  diags.error("Float width mismatch in operation", arg.span);
              }
              return Ty{Ty::FloatTy{bits}};
            }
            return Ty{std::monostate{}};
          } else if constexpr (std::is_same_v<T, UnaryAtom>) {
            auto rt = typeOfLValue(arg.rval, vars, syms, diags);
            if (auto rb = TypeUtils::getIntBitWidth(rt)) {
              return Ty{Ty::BVTy{*rb}};
            }
            // Unary ~ on a vector of integers yields the same type.
            if (auto vt = TypeUtils::asVec(rt)) {
              if (!vt->elem || !std::holds_alternative<IntType>(vt->elem->v))
                diags.error("Unary ~ requires integer lane type", arg.span);
              return Ty{Ty::VecTy{rt}};
            }
            diags.error("Unary op not supported for float", arg.span);
            return Ty{std::monostate{}};
          } else if constexpr (std::is_same_v<T, SelectAtom>) {
            // Two forms. cond=Cond form; maskExpr=mask form.
            bool maskIsVec = false;
            if (arg.cond) {
              checkCond(*arg.cond, vars, syms, ann, diags);
            } else if (arg.maskExpr) {
              auto mt = typeOfExpr(*arg.maskExpr, vars, syms, ann, diags, std::nullopt);
              if (mt.isVec()) {
                // Mask must be <N> i1.
                auto vt = std::get_if<VecType>(&mt.vecType()->v);
                bool ok = vt && vt->elem && std::holds_alternative<IntType>(vt->elem->v) &&
                          (std::get<IntType>(vt->elem->v).kind == IntType::Kind::ICustom) &&
                          (std::get<IntType>(vt->elem->v).bits.value_or(0) == 1);
                if (!ok)
                  diags.error("Mask-based select requires <N> i1 mask", arg.span);
                maskIsVec = true;
              } else if (mt.isBV()) {
                if (mt.bvBits() != 1)
                  diags.error("Mask-based select scalar mask must be i1", arg.span);
              } else {
                diags.error("Mask-based select: mask must be i1 or <N> i1", arg.span);
              }
            }
            auto t1 = typeOfSelectVal(arg.vtrue, vars, syms, ann, diags, expectedBits, ptrCtx);
            auto t2 = typeOfSelectVal(
                arg.vfalse, vars, syms, ann, diags, expectedBits, t1.isPtr() ? t1.ptrType() : ptrCtx
            );
            // Mask-form with <N> i1: arms must be matching <N> T.
            if (maskIsVec) {
              if (!t1.isVec() || !t2.isVec() ||
                  !TypeUtils::areTypesEqual(t1.vecType(), t2.vecType()))
                diags.error(
                    "Mask-based select with <N> i1 mask requires matching <N> T arms", arg.span
                );
              return t1;
            }
            // Resolve null arm using sibling's type
            if (t1.isPtr() && !t2.isPtr())
              diags.error("Select arm type mismatch (pointer vs non-pointer)", arg.span);
            if (!t1.isPtr() && t2.isPtr())
              diags.error("Select arm type mismatch (non-pointer vs pointer)", arg.span);
            if (t1.isBV() && t2.isBV() && t1.bvBits() != t2.bvBits())
              diags.error("Select width mismatch", arg.span);
            if (t1.isFloat() && t2.isFloat() && t1.floatBits() != t2.floatBits())
              diags.error("Select float width mismatch", arg.span);
            if (t1.isVec() && t2.isVec() && !TypeUtils::areTypesEqual(t1.vecType(), t2.vecType()))
              diags.error("Select vector type mismatch", arg.span);
            if ((t1.isBV() || t1.isFloat()) &&
                (t2.isBV() != t1.isBV() || t2.isFloat() != t1.isFloat()))
              diags.error("Select type mismatch", arg.span);
            return t1;
          } else if constexpr (std::is_same_v<T, CoefAtom>) {
            auto ct = typeOfCoef(arg.coef, vars, syms, diags, expectedBits, ptrCtx);
            if (!ct)
              return Ty{std::monostate{}};
            if (auto cb = TypeUtils::getIntBitWidth(ct))
              return Ty{Ty::BVTy{*cb}};
            if (std::holds_alternative<FloatType>(ct->v))
              return Ty{
                  Ty::FloatTy{(std::get<FloatType>(ct->v).kind == FloatType::Kind::F32) ? 32u : 64u}
              };
            if (std::holds_alternative<PtrType>(ct->v))
              return Ty{Ty::PtrTy{ct}};
            // Sym of vector type: a CoefAtom referencing a
            // vector sym yields a vector value (used to stage sym
            // vectors into a local before lane subscripting).
            if (std::holds_alternative<VecType>(ct->v))
              return Ty{Ty::VecTy{ct}};
            if (std::holds_alternative<ArrayType>(ct->v))
              return Ty{Ty::ArrayTy{ct}};
            if (std::holds_alternative<StructType>(ct->v))
              return Ty{Ty::StructTy{ct}};
            if (std::holds_alternative<TensorType>(ct->v))
              return Ty{Ty::TensorTy{ct}};
            return Ty{std::monostate{}};
          } else if constexpr (std::is_same_v<T, RValueAtom>) {
            auto rt = typeOfLValue(arg.rval, vars, syms, diags);
            if (auto rb = TypeUtils::getIntBitWidth(rt))
              return Ty{Ty::BVTy{*rb}};
            if (rt && std::holds_alternative<FloatType>(rt->v))
              return Ty{
                  Ty::FloatTy{(std::get<FloatType>(rt->v).kind == FloatType::Kind::F32) ? 32u : 64u}
              };
            if (rt && std::holds_alternative<PtrType>(rt->v))
              return Ty{Ty::PtrTy{rt}};
            if (rt && std::holds_alternative<VecType>(rt->v))
              return Ty{Ty::VecTy{rt}};
            if (rt && std::holds_alternative<ArrayType>(rt->v))
              return Ty{Ty::ArrayTy{rt}};
            if (rt && std::holds_alternative<StructType>(rt->v))
              return Ty{Ty::StructTy{rt}};
            if (rt && std::holds_alternative<TensorType>(rt->v))
              return Ty{Ty::TensorTy{rt}};
            return Ty{std::monostate{}};
          } else if constexpr (std::is_same_v<T, CmpAtom>) {
            // cmp: result is i1 (scalar) or <N> i1 (vector).
            // Both operands must have the same type; we infer from lhs.
            auto t1 = typeOfSelectVal(arg.lhs, vars, syms, ann, diags, std::nullopt);
            // Sibling pointer context: if LHS is a pointer, pass it
            // as ptrCtx to RHS so that `cmp == %p, null` can infer null's type.
            // This mirrors the pattern in checkCond() (line ~1047).
            TypePtr cmpPtrCtx = t1.isPtr() ? t1.ptrType() : nullptr;
            auto t2 = typeOfSelectVal(arg.rhs, vars, syms, ann, diags, std::nullopt, cmpPtrCtx);
            if (t1.isVec()) {
              if (!t2.isVec() || !TypeUtils::areTypesEqual(t1.vecType(), t2.vecType())) {
                diags.error("cmp: vector operand type mismatch", arg.span);
                return Ty{std::monostate{}};
              }
              // Result type: <N> i1
              auto vt = std::get_if<VecType>(&t1.vecType()->v);
              auto i1 = std::make_shared<Type>();
              i1->v = IntType{IntType::Kind::ICustom, 1, {}};
              auto resVec = std::make_shared<Type>();
              resVec->v = VecType{vt->size, i1, {}};
              return Ty{Ty::VecTy{resVec}};
            }
            // Scalar cmp: both operands must agree (BV/Float/Ptr); result i1
            if (t1.isBV() && t2.isBV() && t1.bvBits() != t2.bvBits())
              diags.error("cmp: integer width mismatch", arg.span);
            else if (t1.isFloat() && t2.isFloat() && t1.floatBits() != t2.floatBits())
              diags.error("cmp: float width mismatch", arg.span);
            return Ty{Ty::BVTy{1}};
          } else if constexpr (std::is_same_v<T, CastAtom>) {
            // 'as' can cast between Int and Float or Int resizing
            TypePtr srcType = nullptr;
            if (auto lit = std::get_if<IntLit>(&arg.src)) {
              (void) lit; // IntLit -> infer from dstType
            } else if (auto flit = std::get_if<FloatLit>(&arg.src)) {
              (void) flit; // FloatLit
            } else if (auto sid = std::get_if<SymId>(&arg.src)) {
              auto it = syms.find(sid->name);
              if (it != syms.end())
                srcType = it->second.type;
            } else if (auto lv = std::get_if<LValue>(&arg.src)) {
              srcType = typeOfLValue(*lv, vars, syms, diags);
            }

            // Check destination
            if (auto it = std::get_if<IntType>(&arg.dstType->v)) {
              return Ty{Ty::BVTy{
                  static_cast<uint32_t>(it->bits.value_or(it->kind == IntType::Kind::I32 ? 32 : 64))
              }};
            } else if (auto ft = std::get_if<FloatType>(&arg.dstType->v)) {
              return Ty{Ty::FloatTy{ft->kind == FloatType::Kind::F32 ? 32u : 64u}};
            } else if (auto vt = std::get_if<VecType>(&arg.dstType->v)) {
              // Vector cast: src must be a vector of matching N.
              // The cast applies lane-wise; the elem-cast must be valid by
              // the scalar rules above (we don't re-check here).
              if (srcType) {
                auto svt = std::get_if<VecType>(&srcType->v);
                if (!svt || svt->size != vt->size)
                  diags.error("Vector cast: lane count must match", arg.span);
              }
              return Ty{Ty::VecTy{arg.dstType}};
            }
            diags.error("Destination of 'as' must be scalar or vector", arg.dstType->span);
            return Ty{std::monostate{}};
          } else if constexpr (std::is_same_v<T, AddrAtom>) {
            // addr <lv> : result is ptr T where T = type(lv)
            auto lvTy = typeOfLValue(arg.lv, vars, syms, diags);
            if (!lvTy)
              return Ty{std::monostate{}};
            // v0.2.0 §3.4.2: only a `let mut` local may have its address taken.
            // Parameters and immutable `let` locals are forbidden.
            auto it = vars.find(arg.lv.base.name);
            if (it != vars.end() && (it->second.isParam || !it->second.isMutable)) {
              diags.error(
                  "'addr' requires a 'let mut' local; '" + arg.lv.base.name + "' is " +
                      (it->second.isParam ? "a parameter" : "immutable"),
                  arg.span
              );
            }
            // §6.8.2: addr is forbidden on vector-typed lvalues
            // (whole vector or any lane). Walk the access chain looking for
            // any traversal through a vector.
            if (it != vars.end()) {
              TypePtr curT = it->second.type;
              for (const auto &acc: arg.lv.accesses) {
                if (curT && std::holds_alternative<VecType>(curT->v)) {
                  diags.error(
                      "addr is forbidden on a vector lane (vectors are not addressable)", arg.span
                  );
                  break;
                }
                if (auto ai_ = std::get_if<AccessIndex>(&acc)) {
                  (void) ai_;
                  if (auto at = TypeUtils::asArray(curT))
                    curT = at->elem;
                  else if (auto vt = TypeUtils::asVec(curT))
                    curT = vt->elem;
                  else
                    curT = nullptr;
                } else if (auto af_ = std::get_if<AccessField>(&acc)) {
                  if (auto st = TypeUtils::asStruct(curT)) {
                    auto sit = structs_.find(st->name.name);
                    if (sit != structs_.end()) {
                      auto fit = sit->second.fields.find(af_->field);
                      curT = (fit != sit->second.fields.end()) ? fit->second : nullptr;
                    }
                  } else
                    curT = nullptr;
                }
              }
              if (curT && std::holds_alternative<VecType>(curT->v))
                diags.error("addr is forbidden on a vector-typed lvalue (§2.11)", arg.span);
            }
            auto ptrNode = std::make_shared<Type>(PtrType{lvTy, arg.span}, arg.span);
            return Ty{Ty::PtrTy{ptrNode}};
          } else if constexpr (std::is_same_v<T, LoadAtom>) {
            // load <rval> : rval must be ptr T, result type is T
            auto rvTy = typeOfLValue(arg.rval, vars, syms, diags);
            if (!rvTy)
              return Ty{std::monostate{}};
            auto pt = std::get_if<PtrType>(&rvTy->v);
            if (!pt) {
              diags.error("'load' requires a pointer (ptr T) operand", arg.span);
              return Ty{std::monostate{}};
            }
            TypePtr pointee = pt->pointee;
            // §6.8.5: aggregate / vector pointees aren't loadable.
            if (std::holds_alternative<ArrayType>(pointee->v) ||
                std::holds_alternative<StructType>(pointee->v) ||
                std::holds_alternative<VecType>(pointee->v)) {
              diags.error(
                  "'load' pointee must be scalar or pointer (§6.8.5): aggregate / vector "
                  "pointees aren't loadable — use 'ptrindex' or 'ptrfield' to navigate first",
                  arg.span
              );
              return Ty{std::monostate{}};
            }
            if (auto pb = TypeUtils::getIntBitWidth(pointee))
              return Ty{Ty::BVTy{*pb}};
            if (std::holds_alternative<FloatType>(pointee->v)) {
              auto &ft = std::get<FloatType>(pointee->v);
              return Ty{Ty::FloatTy{ft.kind == FloatType::Kind::F32 ? 32u : 64u}};
            }
            if (std::holds_alternative<PtrType>(pointee->v))
              return Ty{Ty::PtrTy{pointee}};
            diags.error("'load' pointee type not supported", arg.span);
            return Ty{std::monostate{}};
          } else if constexpr (std::is_same_v<T, PtrIndexAtom>) {
            // §6.8.9: ptrindex <ptr>, <index> where ptr : ptr [N] T,
            // index : iN, result : ptr T.
            auto rvTy = typeOfLValue(arg.rval, vars, syms, diags);
            if (!rvTy)
              return Ty{std::monostate{}};
            auto pt = std::get_if<PtrType>(&rvTy->v);
            if (!pt) {
              diags.error("'ptrindex' requires a pointer (ptr [N] T) operand", arg.span);
              return Ty{std::monostate{}};
            }
            auto at = std::get_if<ArrayType>(&pt->pointee->v);
            if (!at) {
              diags.error(
                  "'ptrindex' requires a ptr [N] T operand; got pointer to non-array", arg.span
              );
              return Ty{std::monostate{}};
            }
            // index must be integer-typed (IntLit or LocalOrSymId of int).
            std::visit(
                [&](auto &&iv) {
                  using IV = std::decay_t<decltype(iv)>;
                  if constexpr (std::is_same_v<IV, IntLit>) {
                    // OK
                  } else {
                    std::visit(
                        [&](auto &&id) {
                          auto vit = vars.find(id.name);
                          if (vit != vars.end()) {
                            if (!vit->second.type ||
                                !std::holds_alternative<IntType>(vit->second.type->v))
                              diags.error("'ptrindex' index must be integer-typed", arg.span);
                            return;
                          }
                          auto sit = syms.find(id.name);
                          if (sit != syms.end() && sit->second.type &&
                              !std::holds_alternative<IntType>(sit->second.type->v))
                            diags.error("'ptrindex' index must be integer-typed", arg.span);
                        },
                        iv
                    );
                  }
                },
                arg.index
            );
            auto resultPtr = std::make_shared<Type>(PtrType{at->elem, arg.span}, arg.span);
            return Ty{Ty::PtrTy{resultPtr}};
          } else if constexpr (std::is_same_v<T, PtrFieldAtom>) {
            // §6.8.10: ptrfield <ptr>, <fld> where ptr : ptr @S,
            // result : ptr F where F is the declared type of @S.fld.
            auto rvTy = typeOfLValue(arg.rval, vars, syms, diags);
            if (!rvTy)
              return Ty{std::monostate{}};
            auto pt = std::get_if<PtrType>(&rvTy->v);
            if (!pt) {
              diags.error("'ptrfield' requires a pointer (ptr @S) operand", arg.span);
              return Ty{std::monostate{}};
            }
            auto st = std::get_if<StructType>(&pt->pointee->v);
            if (!st) {
              diags.error(
                  "'ptrfield' requires a ptr @S operand; got pointer to non-struct", arg.span
              );
              return Ty{std::monostate{}};
            }
            auto sit = structs_.find(st->name.name);
            if (sit == structs_.end()) {
              diags.error("'ptrfield': unknown struct " + st->name.name, arg.span);
              return Ty{std::monostate{}};
            }
            auto fit = sit->second.fields.find(arg.field);
            if (fit == sit->second.fields.end()) {
              diags.error(
                  "'ptrfield': struct " + st->name.name + " has no field '" + arg.field + "'",
                  arg.span
              );
              return Ty{std::monostate{}};
            }
            auto resultPtr = std::make_shared<Type>(PtrType{fit->second, arg.span}, arg.span);
            return Ty{Ty::PtrTy{resultPtr}};
          } else if constexpr (std::is_same_v<T, CallAtom>) {
            // §6.10 Call typing. Resolve callee against the
            // registry built in collectCallees, verify arity and exact
            // parameter-type match. When multiple callees share a name
            // (intrinsic overloading), select the one whose param types
            // match the argument types.
            auto it = callees_.find(arg.callee.name);
            if (it == callees_.end() || it->second.empty()) {
              diags.error("Call to undeclared function: " + arg.callee.name, arg.span);
              return Ty{std::monostate{}};
            }
            const auto &candidates = it->second;
            // Find a candidate with matching param count and compatible
            // param types. When multiple candidates have the same count,
            // peek at argument types (variable declarations, literal sizes)
            // to select the correct overload.
            const CalleeInfo *ci = nullptr;
            for (const auto &cand: candidates) {
              if (cand.paramTypes.size() != arg.args.size())
                continue;
              if (!ci) {
                ci = &cand;
                continue;
              }
              // Vector-parameter overloads (the horizontal
              // reductions) are disambiguated by the argument's *exact*
              // vector type — its lane count and element type. That is the
              // authoritative signal (a typed vector local, not an
              // ambiguous literal width), so it dominates the scalar
              // context / bitwidth heuristics below.
              auto vecArgMatch = [&](const CalleeInfo &c) -> bool {
                bool anyVec = false;
                for (size_t ai = 0; ai < c.paramTypes.size() && ai < arg.args.size(); ++ai) {
                  const TypePtr &pt = c.paramTypes[ai];
                  if (!pt || !std::holds_alternative<VecType>(pt->v))
                    continue;
                  anyVec = true;
                  DiagBag td;
                  Ty at = typeOfExpr(*arg.args[ai], vars, syms, ann, td, std::nullopt);
                  if (td.hasErrors() || !at.isVec() || !TypeUtils::areTypesEqual(at.vecType(), pt))
                    return false;
                }
                return anyVec;
              };
              bool candVec = vecArgMatch(cand);
              bool ciVec = vecArgMatch(*ci);
              if (candVec != ciVec) {
                if (candVec)
                  ci = &cand;
                continue;
              }
              // Multiple candidates with same arity — disambiguate by
              // probing each candidate's param bitwidth on every argument
              // and picking the candidate with the most matching arguments.
              auto paramBW = [](const TypePtr &t) -> std::optional<uint32_t> {
                if (auto b = TypeUtils::getIntBitWidth(t))
                  return b;
                if (t && std::holds_alternative<FloatType>(t->v))
                  return std::get<FloatType>(t->v).kind == FloatType::Kind::F64 ? 64u : 32u;
                return std::nullopt;
              };
              auto countArgMatches = [&](const CalleeInfo *c) -> int {
                int n = 0;
                for (size_t ai = 0; ai < std::min(c->paramTypes.size(), arg.args.size()); ++ai) {
                  auto pbw = paramBW(c->paramTypes[ai]);
                  if (!pbw)
                    continue;
                  DiagBag td;
                  Ty at = typeOfExpr(*arg.args[ai], vars, syms, ann, td, *pbw, c->paramTypes[ai]);
                  uint32_t r = at.isBV() ? at.bvBits() : at.isFloat() ? at.floatBits() : 0;
                  if (r == *pbw && !td.hasErrors())
                    ++n;
                }
                return n;
              };
              auto matchesContext = [&](const CalleeInfo &c) -> bool {
                if (ptrCtx) {
                  return TypeUtils::areTypesEqual(c.retType, ptrCtx);
                }
                if (expectedBits) {
                  if (auto bw = TypeUtils::getIntBitWidth(c.retType)) {
                    return *bw == *expectedBits;
                  }
                  if (c.retType && std::holds_alternative<FloatType>(c.retType->v)) {
                    auto &ft = std::get<FloatType>(c.retType->v);
                    uint32_t fwb = (ft.kind == FloatType::Kind::F32) ? 32u : 64u;
                    return fwb == *expectedBits;
                  }
                }
                return true;
              };
              bool ciMatch = matchesContext(*ci);
              bool candMatch = matchesContext(cand);
              if (candMatch && !ciMatch) {
                ci = &cand;
              } else if (ciMatch && !candMatch) {
                // Keep ci
              } else {
                int ciN = countArgMatches(ci);
                int candN = countArgMatches(&cand);
                if (candN > ciN) {
                  ci = &cand;
                } else if (candN == ciN) {
                  // Tie breaker: if the first argument is a literal, determine its natural width.
                  std::uint32_t argBW = 32;
                  if (!arg.args.empty() && arg.args[0]->rest.empty()) {
                    const auto &fa = arg.args[0]->first;
                    if (auto *ca = std::get_if<CoefAtom>(&fa.v)) {
                      if (auto lit = std::get_if<IntLit>(&ca->coef)) {
                        if (lit->value > 2147483647LL || lit->value < -2147483648LL)
                          argBW = 64;
                      }
                    }
                  }
                  auto pbw1 = paramBW(ci->paramTypes[0]);
                  auto pbw2 = paramBW(cand.paramTypes[0]);
                  if (pbw2 && *pbw2 == argBW && pbw1 && *pbw1 != argBW) {
                    ci = &cand;
                  }
                }
              }
            }
            if (!ci) {
              diags.error("Call to undeclared function: " + arg.callee.name, arg.span);
              return Ty{std::monostate{}};
            }
            // Pin the resolved overload onto the AST node so that
            // every downstream consumer (interpreter, C/WASM backends,
            // solver) honours the same decision instead of re-running
            // their own heuristics. Non-intrinsic callees leave it null.
            if (ci->kind == CalleeInfo::Kind::Intrinsic)
              arg.resolvedIntrinsic = ci->intr;
            if (arg.args.size() != ci->paramTypes.size()) {
              diags.error(
                  "Argument count mismatch in call to " + arg.callee.name + ": expected " +
                      std::to_string(ci->paramTypes.size()) + ", got " +
                      std::to_string(arg.args.size()),
                  arg.span
              );
              return Ty{std::monostate{}};
            }
            for (size_t k = 0; k < arg.args.size(); ++k) {
              const auto &paramT = ci->paramTypes[k];
              auto pbits = TypeUtils::getIntBitWidth(paramT);
              // Float param bitwidth isn't covered by getIntBitWidth (it
              // returns nullopt for non-int types) — feed it explicitly
              // so FloatLit args don't fall back to the f32 default.
              if (!pbits && paramT && std::holds_alternative<FloatType>(paramT->v)) {
                auto &ft = std::get<FloatType>(paramT->v);
                pbits = (ft.kind == FloatType::Kind::F32) ? 32u : 64u;
              }
              Ty argTy = typeOfExpr(*arg.args[k], vars, syms, ann, diags, pbits, paramT);
              // Check exact type match.
              bool ok = false;
              if (argTy.isPtr() && paramT && std::holds_alternative<PtrType>(paramT->v)) {
                ok = TypeUtils::areTypesEqual(argTy.ptrType(), paramT);
              } else if (argTy.isVec() && paramT && std::holds_alternative<VecType>(paramT->v)) {
                ok = TypeUtils::areTypesEqual(argTy.vecType(), paramT);
              } else if (argTy.isTensor() && paramT &&
                         std::holds_alternative<TensorType>(paramT->v)) {
                ok = TypeUtils::areTypesEqual(argTy.tensorType(), paramT);
              } else if (argTy.isBV() && paramT && std::holds_alternative<IntType>(paramT->v)) {
                auto pb = TypeUtils::getIntBitWidth(paramT);
                ok = pb && argTy.bvBits() == *pb;
              } else if (argTy.isFloat() && paramT &&
                         std::holds_alternative<FloatType>(paramT->v)) {
                auto &ft = std::get<FloatType>(paramT->v);
                uint32_t pb = (ft.kind == FloatType::Kind::F32) ? 32u : 64u;
                ok = argTy.floatBits() == pb;
              } else if (argTy.isStruct() && paramT &&
                         std::holds_alternative<StructType>(paramT->v)) {
                ok = TypeUtils::areTypesEqual(argTy.structType(), paramT);
              } else if (argTy.isArray() && paramT &&
                         std::holds_alternative<ArrayType>(paramT->v)) {
                ok = TypeUtils::areTypesEqual(argTy.arrayType(), paramT);
              }
              if (!ok) {
                diags.error(
                    "Argument " + std::to_string(k) + " type mismatch in call to " +
                        arg.callee.name,
                    arg.args[k]->span
                );
              }
            }
            // Result type.
            const auto &rt = ci->retType;
            if (!rt)
              return Ty{std::monostate{}};
            if (auto pb = TypeUtils::getIntBitWidth(rt))
              return Ty{Ty::BVTy{*pb}};
            if (std::holds_alternative<FloatType>(rt->v)) {
              auto &ft = std::get<FloatType>(rt->v);
              return Ty{Ty::FloatTy{ft.kind == FloatType::Kind::F32 ? 32u : 64u}};
            }
            if (std::holds_alternative<PtrType>(rt->v))
              return Ty{Ty::PtrTy{rt}};
            if (std::holds_alternative<VecType>(rt->v))
              return Ty{Ty::VecTy{rt}};
            if (std::holds_alternative<TensorType>(rt->v))
              return Ty{Ty::TensorTy{rt}};
            if (std::holds_alternative<StructType>(rt->v))
              return Ty{Ty::StructTy{rt}};
            if (std::holds_alternative<ArrayType>(rt->v))
              return Ty{Ty::ArrayTy{rt}};
            return Ty{std::monostate{}};
          } else if constexpr (std::is_same_v<T, TensorAtom>) {
            return typeOfTensorAtom(arg, vars, syms, diags);
          }
          return Ty{std::monostate{}};
        },
        a.v
    );
  }

  TypePtr TypeChecker::typeOfCoef(
      const Coef &c, const std::unordered_map<std::string, VarInfo> &vars,
      const std::unordered_map<std::string, SymInfo> &syms, DiagBag &diags,
      std::optional<std::uint32_t> expectedBits, TypePtr ptrCtx
  ) {
    if (auto lit = std::get_if<IntLit>(&c)) {
      uint32_t bits = expectedBits.value_or(32);
      checkLiteralRange(lit->value, bits, lit->span, diags);
      // Annotate the literal with its resolved bitwidth so the interpreter's
      // evalCoef can apply the correct signed-range bounds.  Without this,
      // evalCoef defaults to bits=64 and silently misses narrower overflows.
      lit->resolvedBits = bits;
      auto t = std::make_shared<Type>();
      IntType it;
      if (bits == 32)
        it.kind = IntType::Kind::I32;
      else if (bits == 64)
        it.kind = IntType::Kind::I64;
      else {
        it.kind = IntType::Kind::ICustom;
        it.bits = bits;
      }
      it.span = lit->span;
      t->v = it;
      t->span = lit->span;
      return t;
    }
    if (auto flit = std::get_if<FloatLit>(&c)) {
      uint32_t bits = expectedBits.value_or(32);
      // Annotate with resolved precision so evalCoef passes the correct bits
      // to checkFPResult; without this, evalCoef defaults to 64 and misses
      // f32 overflow (e.g. 1e38 + 1e38 is finite as f64 but +inf as f32).
      flit->resolvedBits = bits;
      auto t = std::make_shared<Type>();
      FloatType ft;
      ft.kind = (bits == 64) ? FloatType::Kind::F64 : FloatType::Kind::F32;
      ft.span = flit->span;
      t->v = ft;
      t->span = flit->span;
      return t;
    }
    if (auto nl = std::get_if<NullLit>(&c)) {
      if (ptrCtx && std::holds_alternative<PtrType>(ptrCtx->v))
        return ptrCtx;
      diags.error("'null' requires a pointer type context (ptr T)", nl->span);
      return nullptr;
    }
    auto id = std::get<LocalOrSymId>(c);
    if (auto lid = std::get_if<LocalId>(&id)) {
      auto it = vars.find(lid->name);
      return it != vars.end() ? it->second.type : nullptr;
    } else {
      auto sid = std::get_if<SymId>(&id);
      auto it = syms.find(sid->name);
      if (it == syms.end())
        return nullptr;
      return it->second.type;
    }
  }

  void TypeChecker::checkLiteralRange(int64_t val, uint32_t bits, SourceSpan sp, DiagBag &diags) {
    if (bits >= 64)
      return;
    // Spec §6.12 + §6.4: all iN are signed two's-complement
    // integers (including i1).  Literals must fit the *signed* range
    // [-2^(N-1), 2^(N-1) - 1] of the target type — no implicit
    // bit-truncating narrowing.  Previously the upper bound used the
    // unsigned max (2^N - 1) as a "convenience", which let `i8 == 255`
    // and `i1 == 1` pass the typecheck and then diverge between the
    // interpreter (cmp masking on i1, raw 64-bit cmp on i8) and the C
    // backend (sign-extending cast).  Tightening to the strict signed
    // max restores uniform iN handling: same rule for every width,
    // out-of-range literals are static errors.
    int64_t min = -(1LL << (bits - 1));
    int64_t max = (1LL << (bits - 1)) - 1;
    if (val < min || val > max) {
      diags.error(
          "Literal " + std::to_string(val) + " out of range for i" + std::to_string(bits) + " ([" +
              std::to_string(min) + ", " + std::to_string(max) + "])",
          sp
      );
    }
  }

  Ty TypeChecker::typeOfSelectVal(
      const SelectVal &sv, const std::unordered_map<std::string, VarInfo> &vars,
      const std::unordered_map<std::string, SymInfo> &syms, [[maybe_unused]] TypeAnnotations &ann,
      DiagBag &diags, std::optional<std::uint32_t> expectedBits, TypePtr ptrCtx
  ) {
    TypePtr t;
    if (auto rv = std::get_if<RValue>(&sv)) {
      t = typeOfLValue(*rv, vars, syms, diags);
    } else {
      t = typeOfCoef(std::get<Coef>(sv), vars, syms, diags, expectedBits, ptrCtx);
    }

    if (!t)
      return Ty{std::monostate{}};
    if (auto bits = TypeUtils::getIntBitWidth(t)) {
      return Ty{Ty::BVTy{*bits}};
    }
    if (std::holds_alternative<FloatType>(t->v)) {
      auto &ft = std::get<FloatType>(t->v);
      return Ty{Ty::FloatTy{ft.kind == FloatType::Kind::F32 ? 32u : 64u}};
    }
    if (std::holds_alternative<PtrType>(t->v))
      return Ty{Ty::PtrTy{t}};
    // Vector arms of `select` (Cond form or mask form).
    if (std::holds_alternative<VecType>(t->v))
      return Ty{Ty::VecTy{t}};
    if (std::holds_alternative<ArrayType>(t->v))
      return Ty{Ty::ArrayTy{t}};
    if (std::holds_alternative<StructType>(t->v))
      return Ty{Ty::StructTy{t}};
    if (std::holds_alternative<TensorType>(t->v))
      return Ty{Ty::TensorTy{t}};
    return Ty{std::monostate{}};
  }

  void TypeChecker::checkCond(
      const Cond &c, const std::unordered_map<std::string, VarInfo> &vars,
      const std::unordered_map<std::string, SymInfo> &syms, TypeAnnotations &ann, DiagBag &diags
  ) {
    auto t1 = typeOfExpr(c.lhs, vars, syms, ann, diags, std::nullopt);
    // If LHS is a pointer, pass it as ptrCtx for RHS (null inference)
    TypePtr rhsPtrCtx = t1.isPtr() ? t1.ptrType() : nullptr;
    auto t2 = typeOfExpr(
        c.rhs, vars, syms, ann, diags, t1.isBV() ? std::optional(t1.bvBits()) : std::nullopt,
        rhsPtrCtx
    );
    if (t1.isTensor() || t2.isTensor())
      diags.error("tensor-type: tensors cannot be used as scalar conditions", c.span);
    if (t1.isBV() && t2.isBV() && t1.bvBits() != t2.bvBits())
      diags.error("Bitwidth mismatch in condition", c.span);
    if (t1.isPtr() && !t2.isPtr())
      diags.error("Pointer condition requires both operands to be pointers", c.span);
    if (!t1.isPtr() && t2.isPtr())
      diags.error("Pointer condition requires both operands to be pointers", c.span);
    if (t1.isPtr() && t2.isPtr() && !TypeUtils::areTypesEqual(t1.ptrType(), t2.ptrType()))
      diags.error("Pointer comparison requires same pointee type", c.span);
  }

} // namespace refractir
