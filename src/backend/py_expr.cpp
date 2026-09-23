#include <stdexcept>
#include "backend/py_backend.hpp"
#include "py_internal.hpp"

namespace refractir {

  namespace {

    const VecType *asVec(const TypePtr &t) { return t ? std::get_if<VecType>(&t->v) : nullptr; }

    const char *relOpStr(RelOp op) {
      switch (op) {
        case RelOp::EQ:
          return "==";
        case RelOp::NE:
          return "!=";
        case RelOp::LT:
          return "<";
        case RelOp::LE:
          return "<=";
        case RelOp::GT:
          return ">";
        case RelOp::GE:
          return ">=";
      }
      return "==";
    }

    // Truncating (toward-zero) integer division as an inline Python
    // expression. Python `//` floors, so we flip the sign of the
    // magnitude when the operands disagree in sign. Operands are always
    // leaf references here, so the duplication is side-effect-free.
    std::string pyTruncDiv(const std::string &a, const std::string &b) {
      return "(" + a + " // " + b + " if (" + a + " < 0) == (" + b + " < 0) else -(-" + a + " // " +
             b + "))";
    }

  } // namespace

  // --- Binary-operator lowering (see py_backend.hpp) ---------------

  std::string PyBackend::binInt(
      const char *helper, char op, const std::string &a, const std::string &b, const std::string &n
  ) const {
    if (noUbGuards_)
      return "(" + a + " " + op + " " + b + ")";
    return std::string(helper) + "(" + a + ", " + b + ", " + n + ")";
  }

  std::string
  PyBackend::binIDiv(const std::string &a, const std::string &b, const std::string &n) const {
    if (noUbGuards_)
      return pyTruncDiv(a, b);
    return "_sdiv(" + a + ", " + b + ", " + n + ")";
  }

  std::string
  PyBackend::binIRem(const std::string &a, const std::string &b, const std::string &n) const {
    if (noUbGuards_)
      return "(" + a + " - " + b + " * " + pyTruncDiv(a, b) + ")";
    return "_srem(" + a + ", " + b + ", " + n + ")";
  }

  std::string PyBackend::binShift(
      const char *helper, const char *op, const std::string &a, const std::string &b,
      const std::string &n
  ) const {
    if (noUbGuards_)
      return "(" + a + " " + op + " " + b + ")";
    return std::string(helper) + "(" + a + ", " + b + ", " + n + ")";
  }

  std::string
  PyBackend::binLshr(const std::string &a, const std::string &b, const std::string &n) const {
    if (noUbGuards_)
      // Mask to n bits, unsigned-shift, then reinterpret as signed iN
      // via the retained _cast_int primitive.
      return "_cast_int((" + a + " & ((1 << " + n + ") - 1)) >> " + b + ", " + n + ")";
    return "_lshr(" + a + ", " + b + ", " + n + ")";
  }

  std::string PyBackend::binFloat(
      const char *helper, char op, const std::string &a, const std::string &b, const std::string &n
  ) const {
    if (noUbGuards_) {
      std::string e = a + " " + op + " " + b;
      return n == "32" ? "_f32(" + e + ")" : "(" + e + ")";
    }
    return std::string(helper) + "(" + a + ", " + b + ", " + n + ")";
  }

  std::string
  PyBackend::binFFmod(const std::string &a, const std::string &b, const std::string &n) const {
    if (noUbGuards_) {
      std::string e = "math.fmod(" + a + ", " + b + ")";
      return n == "32" ? "_f32(" + e + ")" : e;
    }
    return "_ffmod(" + a + ", " + b + ", " + n + ")";
  }

  std::string PyBackend::castF2I(const std::string &x, const std::string &n) const {
    // int() truncates toward zero, matching the finite-only float model.
    if (noUbGuards_)
      return "int(" + x + ")";
    return "_f2i(" + x + ", " + n + ")";
  }

  std::string PyBackend::exprStr(const Expr &expr) {
    std::string acc = atomStr(expr.first);
    if (expr.rest.empty())
      return acc;
    TypePtr t = getExprType(expr);
    if (const VecType *vt = asVec(t)) {
      // Lane-wise chain; scalar tail atoms broadcast.
      const std::string n = std::to_string(vt->size);
      const bool fp = floatWidth(vt->elem) != 0;
      const std::string w = std::to_string(fp ? floatWidth(vt->elem) : intWidth(vt->elem));
      for (const auto &tail: expr.rest) {
        std::string rhs = atomStr(tail.atom);
        if (!asVec(getAtomType(tail.atom)))
          rhs = "[" + rhs + "] * " + n;
        char op = tail.op == AddOp::Plus ? '+' : '-';
        std::string lane = fp ? binFloat(op == '+' ? "_fadd" : "_fsub", op, "x", "y", w)
                              : binInt(op == '+' ? "_iadd" : "_isub", op, "x", "y", w);
        acc = "[" + lane + " for x, y in zip(" + acc + ", " + rhs + ")]";
      }
      return acc;
    }
    if (std::uint32_t fb = floatWidth(t)) {
      for (const auto &tail: expr.rest) {
        char op = tail.op == AddOp::Plus ? '+' : '-';
        acc = binFloat(
            op == '+' ? "_fadd" : "_fsub", op, acc, atomStr(tail.atom), std::to_string(fb)
        );
      }
      return acc;
    }
    if (t && std::holds_alternative<PtrType>(t->v)) {
      // ptr ± iN walks elements; ptr - ptr yields the element distance
      // (i64), after which any remaining terms are integer arithmetic.
      bool isPtr = true;
      for (const auto &tail: expr.rest) {
        TypePtr tt = getAtomType(tail.atom);
        bool tailPtr = tt && std::holds_alternative<PtrType>(tt->v);
        if (isPtr && tailPtr && tail.op == AddOp::Minus) {
          acc = "_pdiff(" + acc + ", " + atomStr(tail.atom) + ")";
          isPtr = false;
        } else if (isPtr) {
          std::string n = atomStr(tail.atom);
          acc = "_padd(" + acc + ", " + (tail.op == AddOp::Plus ? n : "-(" + n + ")") + ")";
        } else {
          char op = tail.op == AddOp::Plus ? '+' : '-';
          acc = binInt(op == '+' ? "_iadd" : "_isub", op, acc, atomStr(tail.atom), "64");
        }
      }
      return acc;
    }
    std::uint32_t n = intWidth(t);
    if (n == 0)
      n = 32; // defensive: post-typecheck every non-ptr/float expr is integral
    for (const auto &tail: expr.rest) {
      char op = tail.op == AddOp::Plus ? '+' : '-';
      acc = binInt(op == '+' ? "_iadd" : "_isub", op, acc, atomStr(tail.atom), std::to_string(n));
    }
    return acc;
  }

  std::string PyBackend::atomStr(const Atom &atom) {
    return std::visit(
        [this](auto &&arg) -> std::string {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, OpAtom>) {
            return opAtomStr(arg);
          } else if constexpr (std::is_same_v<T, SelectAtom>) {
            return selectAtomStr(arg);
          } else if constexpr (std::is_same_v<T, CmpAtom>) {
            return cmpAtomStr(arg);
          } else if constexpr (std::is_same_v<T, CoefAtom>) {
            return coefStr(arg.coef);
          } else if constexpr (std::is_same_v<T, RValueAtom>) {
            return lvalueStr(arg.rval);
          } else if constexpr (std::is_same_v<T, UnaryAtom>) {
            if (asVec(getLValueType(arg.rval)))
              return "[(~x) for x in " + lvalueStr(arg.rval) + "]";
            return "(~" + lvalueStr(arg.rval) + ")";
          } else if constexpr (std::is_same_v<T, CastAtom>) {
            return castAtomStr(arg);
          } else if constexpr (std::is_same_v<T, CallAtom>) {
            return callAtomStr(arg);
          } else if constexpr (std::is_same_v<T, AddrAtom>) {
            return addrAtomStr(arg);
          } else if constexpr (std::is_same_v<T, LoadAtom>) {
            return loadAtomStr(arg);
          } else if constexpr (std::is_same_v<T, PtrIndexAtom>) {
            return ptrIndexAtomStr(arg);
          } else if constexpr (std::is_same_v<T, TensorAtom>) {
            throw std::runtime_error("python target does not support tensor atoms");
          } else {
            static_assert(std::is_same_v<T, PtrFieldAtom>);
            return ptrFieldAtomStr(arg);
          }
        },
        atom.v
    );
  }

  std::string PyBackend::opAtomStr(const OpAtom &arg) {
    TypePtr rt = getLValueType(arg.rval);
    TypePtr ct = getCoefType(arg.coef);
    const VecType *vt = asVec(rt);
    if (!vt)
      vt = asVec(ct);
    if (vt) {
      const std::uint64_t n = vt->size;
      const std::string C = vecCoefStr(arg.coef, n);
      std::string R = lvalueStr(arg.rval);
      if (!asVec(rt))
        R = "[" + R + "] * " + std::to_string(n);
      const bool fp = floatWidth(vt->elem) != 0;
      const std::string w = std::to_string(fp ? floatWidth(vt->elem) : intWidth(vt->elem));
      std::string lane;
      switch (arg.op) {
        case AtomOpKind::Mul:
          lane = fp ? binFloat("_fmul", '*', "x", "y", w) : binInt("_imul", '*', "x", "y", w);
          break;
        case AtomOpKind::Div:
          lane = fp ? binFloat("_fdiv", '/', "x", "y", w) : binIDiv("x", "y", w);
          break;
        case AtomOpKind::Mod:
          lane = fp ? binFFmod("x", "y", w) : binIRem("x", "y", w);
          break;
        case AtomOpKind::And:
          lane = "(x & y)";
          break;
        case AtomOpKind::Or:
          lane = "(x | y)";
          break;
        case AtomOpKind::Xor:
          lane = "(x ^ y)";
          break;
        case AtomOpKind::Shl:
          lane = binShift("_shl", "<<", "x", "y", w);
          break;
        case AtomOpKind::Shr:
          lane = binShift("_ashr", ">>", "x", "y", w);
          break;
        case AtomOpKind::LShr:
          lane = binLshr("x", "y", w);
          break;
      }
      return "[" + lane + " for x, y in zip(" + C + ", " + R + ")]";
    }
    const std::string c = coefStr(arg.coef);
    const std::string r = lvalueStr(arg.rval);
    TypePtr t = rt;
    if (std::uint32_t fb = floatWidth(t)) {
      const std::string b = std::to_string(fb);
      switch (arg.op) {
        case AtomOpKind::Mul:
          return binFloat("_fmul", '*', c, r, b);
        case AtomOpKind::Div:
          return binFloat("_fdiv", '/', c, r, b);
        case AtomOpKind::Mod:
          return binFFmod(c, r, b);
        default:
          throw std::runtime_error("python target: bitwise operator on float operands");
      }
    }
    std::uint32_t bits = intWidth(t);
    if (bits == 0)
      throw std::runtime_error("python target: atom operator on non-scalar operands");
    const std::string n = std::to_string(bits);
    switch (arg.op) {
      case AtomOpKind::Mul:
        return binInt("_imul", '*', c, r, n);
      case AtomOpKind::Div:
        return binIDiv(c, r, n);
      case AtomOpKind::Mod:
        return binIRem(c, r, n);
      case AtomOpKind::And:
        return "(" + c + " & " + r + ")";
      case AtomOpKind::Or:
        return "(" + c + " | " + r + ")";
      case AtomOpKind::Xor:
        return "(" + c + " ^ " + r + ")";
      case AtomOpKind::Shl:
        return binShift("_shl", "<<", c, r, n);
      case AtomOpKind::Shr:
        return binShift("_ashr", ">>", c, r, n);
      case AtomOpKind::LShr:
        return binLshr(c, r, n);
    }
    return "";
  }

  std::string PyBackend::selectAtomStr(const SelectAtom &arg) {
    const std::string vt = selectValStr(arg.vtrue);
    const std::string vf = selectValStr(arg.vfalse);
    if (arg.cond)
      return "(" + vt + " if " + condStr(*arg.cond) + " else " + vf + ")";
    auto maskTy = getExprType(*arg.maskExpr);
    if (const VecType *mv = asVec(maskTy)) {
      // Per-lane blend on a {0,-1} mask vector.
      const std::uint64_t n = mv->size;
      return "[(x if m != 0 else y) for m, x, y in zip(" + exprStr(*arg.maskExpr) + ", " +
             vecSelectValStr(arg.vtrue, n) + ", " + vecSelectValStr(arg.vfalse, n) + ")]";
    }
    // Mask form with a scalar i1 mask ({0,-1}): select on != 0.
    return "(" + vt + " if (" + exprStr(*arg.maskExpr) + ") != 0 else " + vf + ")";
  }

  namespace {

    // Pointer comparisons: equality is always legal (identity + offset);
    // relational compares go through _prel, which traps on cross-object
    // or null operands per spec rule 14.
    std::string ptrCompare(const std::string &l, const std::string &r, RelOp op) {
      switch (op) {
        case RelOp::EQ:
          return "_peq(" + l + ", " + r + ")";
        case RelOp::NE:
          return "not _peq(" + l + ", " + r + ")";
        default:
          return "_prel(" + l + ", " + r + ") " + relOpStr(op) + " 0";
      }
    }

  } // namespace

  std::string PyBackend::cmpAtomStr(const CmpAtom &arg) {
    auto lt = getSelectValType(arg.lhs);
    auto rt = getSelectValType(arg.rhs);
    const VecType *vt = asVec(lt);
    if (!vt)
      vt = asVec(rt);
    if (vt) {
      const std::uint64_t n = vt->size;
      return "[(-1 if x " + std::string(relOpStr(arg.op)) + " y else 0) for x, y in zip(" +
             vecSelectValStr(arg.lhs, n) + ", " + vecSelectValStr(arg.rhs, n) + ")]";
    }
    F32Guard ctx(f32Ctx_, !(containsF64(lt) || containsF64(rt)));
    if ((lt && std::holds_alternative<PtrType>(lt->v)) ||
        (rt && std::holds_alternative<PtrType>(rt->v)))
      return "(-1 if " + ptrCompare(selectValStr(arg.lhs), selectValStr(arg.rhs), arg.op) +
             " else 0)";
    return "(-1 if (" + selectValStr(arg.lhs) + ") " + relOpStr(arg.op) + " (" +
           selectValStr(arg.rhs) + ") else 0)";
  }

  std::string
  PyBackend::castValueStr(const std::string &src, const TypePtr &srcTy, const TypePtr &dstTy) {
    if (std::uint32_t m = intWidth(dstTy)) {
      if (floatWidth(srcTy))
        return castF2I(src, std::to_string(m));
      return "_cast_int(" + src + ", " + std::to_string(m) + ")";
    }
    std::uint32_t fb = floatWidth(dstTy);
    if (fb == 32)
      return "_f32(" + src + ")";
    if (intWidth(srcTy))
      return "float(" + src + ")";
    return src; // f32 -> f64 / f64 -> f64: exact
  }

  std::string PyBackend::castAtomStr(const CastAtom &arg) {
    // Vector cast: per-lane conversion of a vector lvalue.
    if (const VecType *dv = asVec(arg.dstType)) {
      auto lv = std::get_if<LValue>(&arg.src);
      if (!lv)
        throw std::runtime_error("python target: vector cast source must be an lvalue");
      const VecType *sv = asVec(getLValueType(*lv));
      if (!sv)
        throw std::runtime_error("python target: vector cast source must be a vector");
      return "[" + castValueStr("x", sv->elem, dv->elem) + " for x in " + lvalueStr(*lv) + "]";
    }
    std::string src;
    TypePtr srcTy;
    std::visit(
        [&](auto &&s) {
          using S = std::decay_t<decltype(s)>;
          if constexpr (std::is_same_v<S, IntLit>) {
            src = std::to_string(s.value);
            srcTy = getCoefType(Coef{s});
          } else if constexpr (std::is_same_v<S, FloatLit>) {
            src = formatFloatLit(s.value);
            // Literal precision follows the destination (mirrors the C
            // backend's dst-driven suffix in emitCastAtom).
            if (!containsF64(arg.dstType))
              src = "_f32(" + src + ")";
            srcTy = getCoefType(Coef{s});
          } else if constexpr (std::is_same_v<S, SymId>) {
            src = symCall(s.name);
            auto it = varTypes_.find(s.name);
            srcTy = it != varTypes_.end() ? it->second : nullptr;
          } else {
            src = lvalueStr(s);
            srcTy = getLValueType(s);
          }
        },
        arg.src
    );
    return castValueStr(src, srcTy, arg.dstType);
  }

  std::string PyBackend::vecCoefStr(const Coef &coef, std::uint64_t n) {
    if (asVec(getCoefType(coef)))
      return std::visit(
          [&](auto &&arg) -> std::string {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, LocalOrSymId>) {
              return std::visit(
                  [&](auto &&id) -> std::string {
                    if constexpr (std::is_same_v<std::decay_t<decltype(id)>, SymId>) {
                      return symCall(id.name); // provider list; lanes never undef
                    } else {
                      return lvalueStr(LValue{id, {}, {}});
                    }
                  },
                  arg
              );
            } else {
              throw std::runtime_error("python target: unexpected vector coefficient");
            }
          },
          coef
      );
    return "[" + coefStr(coef) + "] * " + std::to_string(n);
  }

  std::string PyBackend::vecSelectValStr(const SelectVal &sv, std::uint64_t n) {
    if (std::holds_alternative<RValue>(sv)) {
      const auto &lv = std::get<RValue>(sv);
      if (asVec(getLValueType(lv)))
        return lvalueStr(lv);
      return "[" + lvalueStr(lv) + "] * " + std::to_string(n);
    }
    return vecCoefStr(std::get<Coef>(sv), n);
  }

  std::string PyBackend::callAtomStr(const CallAtom &arg) {
    if (const IntrinsicDecl *intr = arg.resolvedIntrinsic) {
      std::vector<std::string> args;
      for (std::size_t i = 0; i < arg.args.size(); ++i) {
        F32Guard ctx(
            f32Ctx_, i < intr->params.size() ? !containsF64(intr->params[i].type) : f32Ctx_
        );
        args.push_back(exprStr(*arg.args[i]));
      }
      return callIntrinsic(*intr, args);
    }
    std::vector<TypePtr> paramTypes;
    for (const auto &fd: prog_->funs) {
      if (fd.name.name == arg.callee.name) {
        for (const auto &p: fd.params)
          paramTypes.push_back(p.type);
        break;
      }
    }
    if (paramTypes.empty()) {
      for (const auto &d: prog_->extDecls) {
        if (d.name.name == arg.callee.name) {
          for (const auto &p: d.params)
            paramTypes.push_back(p.type);
          break;
        }
      }
    }
    std::string s = mangleFun(arg.callee.name) + "(";
    for (std::size_t i = 0; i < arg.args.size(); ++i) {
      if (i)
        s += ", ";
      // The callee's i-th param type drives the literal context.
      F32Guard ctx(f32Ctx_, i < paramTypes.size() ? !containsF64(paramTypes[i]) : f32Ctx_);
      s += exprStr(*arg.args[i]);
    }
    return s + ")";
  }

  std::string PyBackend::condStr(const Cond &cond) {
    auto lt = getExprType(cond.lhs);
    auto rt = getExprType(cond.rhs);
    // Either operand's type decides the literal context (operands
    // share one type; the disjunction is defensive).
    F32Guard ctx(f32Ctx_, !(containsF64(lt) || containsF64(rt)));
    if ((lt && std::holds_alternative<PtrType>(lt->v)) &&
        (rt && std::holds_alternative<PtrType>(rt->v)))
      return ptrCompare(exprStr(cond.lhs), exprStr(cond.rhs), cond.op);
    return "(" + exprStr(cond.lhs) + ") " + relOpStr(cond.op) + " (" + exprStr(cond.rhs) + ")";
  }

  std::string PyBackend::coefStr(const Coef &coef) {
    return std::visit(
        [this](auto &&arg) -> std::string {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, IntLit>) {
            return std::to_string(arg.value);
          } else if constexpr (std::is_same_v<T, FloatLit>) {
            std::string s = formatFloatLit(arg.value);
            // Outside an f64 context a float literal rounds to single
            // precision, the way the C backend's `f` suffix does.
            if (f32Ctx_)
              return "_f32(" + s + ")";
            return s;
          } else if constexpr (std::is_same_v<T, NullLit>) {
            return "_NULL";
          } else {
            return std::visit(
                [this](auto &&id) -> std::string {
                  if constexpr (std::is_same_v<std::decay_t<decltype(id)>, SymId>) {
                    return symCall(id.name);
                  } else {
                    if (boxedRoots_.count(id.name)) {
                      return "_rd(" + pyLocal(id.name) + ", 0)";
                    }
                    return pyLocal(id.name);
                  }
                },
                arg
            );
          }
        },
        coef
    );
  }

  std::string PyBackend::selectValStr(const SelectVal &sv) {
    if (std::holds_alternative<RValue>(sv))
      return lvalueStr(std::get<RValue>(sv));
    return coefStr(std::get<Coef>(sv));
  }

} // namespace refractir
