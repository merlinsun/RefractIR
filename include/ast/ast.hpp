#pragma once

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>
#include "ast/tensor.hpp"

namespace refractir {

  /**
   * Represents a location in the source code.
   */
  struct SourcePos {
    std::size_t offset = 0; // byte offset in source
    int line = 1;           // 1-based
    int col = 1;            // 1-based
  };

  /**
   * Represents a span between two source positions.
   */
  struct SourceSpan {
    SourcePos begin;
    SourcePos end;
  };

  /**
   * A structured parse error with location information.
   */
  struct ParseError : std::runtime_error {
    SourceSpan span;

    explicit ParseError(const std::string &msg, SourceSpan sp) :
        std::runtime_error(msg), span(sp) {}
  };

  /**
   * A lexer error — a ParseError subclass so existing catch(ParseError)
   * sites still handle it, but the main driver can distinguish lex vs parse.
   */
  struct LexError : ParseError {
    explicit LexError(const std::string &msg, SourceSpan sp) : ParseError(msg, sp) {}
  };

  // ---------------------------
  // Node Base
  // ---------------------------

  using NodeId = std::uint32_t;

  /**
   * Base struct for all AST nodes containing common metadata.
   */
  struct NodeBase {
    NodeId id{};
    SourceSpan span{};
  };

  // ---------------------------
  // Identifier kinds (type-safe)
  // ---------------------------

  /**
   * Global identifier starting with '@', e.g., '@main'.
   */
  struct GlobalId {
    std::string name;
    SourceSpan span;
  };

  /**
   * Local identifier starting with '%', e.g., '%x'.
   */
  struct LocalId {
    std::string name;
    SourceSpan span;
  };

  /**
   * Symbolic identifier starting with '@?' or '%?', e.g., '%?v'.
   */
  struct SymId {
    std::string name;
    SourceSpan span;
  };

  /**
   * Block label identifier starting with '^', e.g., '^entry'.
   */
  struct BlockLabel {
    std::string name;
    SourceSpan span;
  };

  using AnyId = std::variant<GlobalId, LocalId, SymId, BlockLabel>;
  using LocalOrSymId = std::variant<LocalId, SymId>;

  // ---------------------------
  // Type system (AST-level)
  // ---------------------------

  /**
   * Represents integer types with specific bitwidths.
   */
  struct IntType {
    enum class Kind { I32, I64, ICustom } kind = Kind::I32;
    std::optional<int> bits; // bitwidth for ICustom
    SourceSpan span;
  };

  /**
   * Represents floating-point types.
   */
  struct FloatType {
    enum class Kind { F32, F64 } kind = Kind::F32;
    SourceSpan span;
  };

  /**
   * Represents user-defined struct types.
   */
  struct StructType {
    GlobalId name;
    SourceSpan span;
  };

  struct ArrayType;
  struct PtrType;
  struct VecType;
  struct Type;
  using TypePtr = std::shared_ptr<Type>;

  /**
   * Represents fixed-size array types.
   */
  struct ArrayType {
    std::uint64_t size = 0;
    TypePtr elem;
    SourceSpan span;
  };

  /**
   * Represents pointer types: ptr T.
   * In v0.2.0 the pointee must be a scalar (iN, f32, f64) or another ptr T.
   */
  struct PtrType {
    TypePtr pointee;
    SourceSpan span;
  };

  /**
   * Fixed-width SIMD vector type: <N> T.
   * T must be a scalar (iN, f32, f64). N >= 2. The bitwidth is
   * N * bitwidth(T); vectors are register-shaped value types and not
   * addressable (no `ptr <N> T`). Lane access uses LValue subscript.
   */
  struct VecType {
    std::uint64_t size = 0; // N (lane count)
    TypePtr elem;           // lane scalar type
    SourceSpan span;
  };

  /**
   * Fixed-rank tensor value with a statically-known ordered shape.
   */
  struct TensorType {
    Shape shape;
    TypePtr elem;
    SourceSpan span;
  };

  /**
   * Wrapper for all possible types in RefractIR.
   */
  struct Type {
    using Variant =
        std::variant<IntType, FloatType, StructType, ArrayType, PtrType, VecType, TensorType>;
    Variant v;
    SourceSpan span;

    Type() = default;

    Type(Variant value, SourceSpan sourceSpan = {})
        : v(std::move(value)), span(std::move(sourceSpan)) {}
  };

  // ---------------------------
  // AST: expressions
  // ---------------------------

  /**
   * Literal integer value.
   *
   * `resolvedBits` is zero until the type checker populates it with the
   * inferred bitwidth (from context, or the SPEC default of 32 for i32).
   * The interpreter reads this field in `evalCoef` to apply the correct
   * signed-overflow bounds; leaving it at the raw `int64_t` size (64) would
   * silently bypass overflow detection for narrower types.
   */
  struct IntLit {
    std::int64_t value = 0;
    SourceSpan span;
    // Populated by TypeChecker::typeOfCoef; 0 means "not yet resolved".
    mutable uint32_t resolvedBits = 0;
  };

  /**
   * Literal floating-point value.
   *
   * `resolvedBits` is zero until the type checker populates it with the
   * inferred bitwidth (32 for f32, 64 for f64; default per SPEC is 32).
   * The interpreter's `evalCoef` uses this so `checkFPResult` applies the
   * correct precision: with bits=64 it never truncates to float, so f32
   * overflow (e.g., `1e38 + 1e38`) would silently pass.
   */
  struct FloatLit {
    double value = 0.0;
    SourceSpan span;
    // Populated by TypeChecker::typeOfCoef; 0 means "not yet resolved".
    mutable uint32_t resolvedBits = 0;
  };

  /**
   * The null pointer constant, typed by context.
   */
  struct NullLit {
    SourceSpan span;
  };

  /**
   * A coefficient in an expression (literal, variable, or null pointer).
   */
  using Coef = std::variant<IntLit, FloatLit, LocalOrSymId, NullLit>;

  /**
   * An index for array access.
   */
  using Index = std::variant<IntLit, LocalOrSymId>;

  /**
   * Represents an array index access segment.
   */
  struct AccessIndex {
    Index index;
    SourceSpan span;
  };

  /**
   * Represents a struct field access segment.
   */
  struct AccessField {
    std::string field;
    SourceSpan span;
  };

  using Access = std::variant<AccessIndex, AccessField>;

  /**
   * Represents an addressable location (e.g., %x.y[0]).
   */
  struct LValue {
    LocalId base;
    std::vector<Access> accesses;
    SourceSpan span;
  };

  using RValue = LValue;

  /**
   * Relational operators for comparisons.
   */
  enum class RelOp { EQ, NE, LT, LE, GT, GE };

  struct Expr;
  struct Cond;

  using SelectVal = std::variant<RValue, Coef>;

  /**
   * Ternary select expression (lazy evaluation).
   *
   * Two forms (spec §5.3):
   *   - Cond form:  `select Cond, vt, vf` — scalar boolean predicate.
   *   - Mask form:  `select Expr, vt, vf` — Expr of type i1 or <N> i1.
   *
   * Exactly one of `cond` and `maskExpr` is non-null. The parser
   * disambiguates by lookahead: after parsing the first Expr, if the
   * next token is a RelOp the form is Cond; if it is `,` the form is
   * mask.
   */
  struct SelectAtom {
    std::unique_ptr<Cond> cond;     // Cond form
    std::unique_ptr<Expr> maskExpr; // mask form
    SelectVal vtrue;
    SelectVal vfalse;
    SourceSpan span;
  };

  /**
   * Binary operator kinds for atoms.
   */
  enum class AtomOpKind { Mul, Div, Mod, And, Or, Xor, Shl, Shr, LShr };

  /**
   * Binary operation atom.
   */
  struct OpAtom {
    AtomOpKind op;
    Coef coef;
    RValue rval;
    SourceSpan span;
  };

  enum class UnaryOpKind { Not };

  /**
   * Unary operation atom.
   */
  struct UnaryAtom {
    UnaryOpKind op;
    RValue rval;
    SourceSpan span;
  };

  /**
   * Constant or variable atom.
   */
  struct CoefAtom {
    Coef coef;
    SourceSpan span;
  };

  /**
   * Read from an LValue atom.
   */
  struct RValueAtom {
    RValue rval;
    SourceSpan span;
  };

  /**
   * Type cast atom.
   */
  struct CastAtom {
    using Variant = std::variant<IntLit, FloatLit, SymId, LValue>;
    Variant src;
    TypePtr dstType;
    SourceSpan span;
  };

  /**
   * Address-of atom: addr <lv>.  Result type is ptr T where T = type(lv).
   * The root of lv must be a let mut local.
   */
  struct AddrAtom {
    LValue lv;
    SourceSpan span;
  };

  /**
   * Load-through-pointer atom: load <rval>.
   * rval must have type ptr T; result type is T.
   */
  struct LoadAtom {
    RValue rval;
    SourceSpan span;
  };

  /**
   * Reified comparison: cmp <relop> <lhs>, <rhs>.
   * Produces i1 for scalar operands or <N> i1 for vector operands.
   * lhs/rhs are SelectVal (RValue | Coef) so literals are admitted.
   */
  struct CmpAtom {
    RelOp op;
    SelectVal lhs;
    SelectVal rhs;
    SourceSpan span;
  };

  /**
   * Aggregate-pointer navigation atoms (§6.8.9, §6.8.10).
   *
   * `ptrindex <ptr>, <index>` navigates from `ptr [N] T` to `ptr T` at
   * a runtime index. Strict UB rules: index in [0, N], non-null, non-
   * undef, non-one-past-end source.
   *
   * `ptrfield <ptr>, <fieldname>` navigates from `ptr @S` to
   * `ptr FieldType` at a statically-known field offset. Same source
   * UB checks (null / undef / one-past-end) as ptrindex.
   */
  struct PtrIndexAtom {
    RValue rval;
    Index index;
    SourceSpan span;
  };

  struct PtrFieldAtom {
    RValue rval;
    std::string field;
    SourceSpan span;
  };

  /**
   * Function call atom: `call @name(args...)`.
   * The callee is resolved by name at typecheck time against the visible
   * `fun`/`decl`/`intrinsic` declarations. Arguments are evaluated
   * left-to-right; side effects (memory mutation, PC/REQ updates) commit
   * unconditionally as part of evaluating the call.
   */
  struct IntrinsicDecl; // fwd

  struct CallAtom {
    GlobalId callee;
    std::vector<std::shared_ptr<Expr>> args;
    SourceSpan span;
    // Resolved overload for intrinsic calls. The type checker
    // pins the exact IntrinsicDecl chosen for this call site (using
    // arg types + return-type context). Non-intrinsic calls or
    // un-typechecked AST leave this null; consumers (interp, C/WASM
    // backends, solver) must consult this first to stay in lockstep
    // with the type checker's choice.
    mutable const IntrinsicDecl *resolvedIntrinsic = nullptr;
  };

  /**
   * Tensor operator atom. `operands` are the source scalar/tensor operands;
   * `attribute` stores a shape, coordinate, or axis permutation where the
   * operator grammar requires one. `axis` and `keepDims` store reduction and
   * axis-operation attributes.
   */
  struct TensorAtom {
    TensorOp op = TensorOp::Add;
    std::vector<Coef> operands;
    std::vector<std::uint32_t> attribute;
    std::int64_t axis = 0;
    bool keepDims = false;
    SourceSpan span;
    // Filled by the type checker so evaluators can consume an already-checked
    // tensor expression without repeating shape inference.
    mutable TypePtr resolvedType;
  };

  /**
   * The fundamental building block of expressions.
   */
  struct Atom {
    using Variant = std::variant<
        OpAtom, SelectAtom, CoefAtom, RValueAtom, CastAtom, UnaryAtom, AddrAtom, LoadAtom, CmpAtom,
        PtrIndexAtom, PtrFieldAtom, CallAtom, TensorAtom>;
    Variant v;
    SourceSpan span;
  };

  enum class AddOp { Plus, Minus };

  /**
   * Represents a linear expression of atoms.
   */
  struct Expr {
    Atom first;

    struct Tail {
      AddOp op;
      Atom atom;
      SourceSpan span;
    };

    std::vector<Tail> rest;
    SourceSpan span;
  };

  /**
   * A boolean condition (comparison of two expressions).
   */
  struct Cond {
    Expr lhs;
    RelOp op;
    Expr rhs;
    SourceSpan span;
  };

  // ---------------------------
  // AST: instructions / terminators
  // ---------------------------

  /**
   * Assignment instruction: lhs = rhs.
   */
  struct AssignInstr {
    LValue lhs;
    Expr rhs;
    SourceSpan span;
  };

  /**
   * Assume instruction: provides a constraint to the solver.
   */
  struct AssumeInstr {
    Cond cond;
    SourceSpan span;
  };

  /**
   * Require instruction: an assertion that must hold.
   */
  struct RequireInstr {
    Cond cond;
    std::optional<std::string> message;
    SourceSpan span;
  };

  /**
   * Store instruction: store <ptr>, <val>.
   * Writes val (type T) through ptr (type ptr T).
   */
  struct StoreInstr {
    Expr ptr; // must evaluate to ptr T
    Expr val; // must evaluate to T
    SourceSpan span;
  };

  using Instr = std::variant<AssignInstr, AssumeInstr, RequireInstr, StoreInstr>;

  /**
   * Branch terminator (conditional or unconditional).
   */
  struct BrTerm {
    std::optional<Cond> cond;
    BlockLabel dest;
    BlockLabel thenLabel;
    BlockLabel elseLabel;
    bool isConditional = false;
    SourceSpan span;
  };

  /**
   * Return terminator.
   */
  struct RetTerm {
    std::optional<Expr> value;
    SourceSpan span;
  };

  /**
   * Unreachable terminator.
   */
  struct UnreachableTerm {
    SourceSpan span;
  };

  using Terminator = std::variant<BrTerm, RetTerm, UnreachableTerm>;

  /**
   * A basic block containing instructions and ending with a terminator.
   */
  struct Block {
    BlockLabel label;
    std::vector<Instr> instrs;
    Terminator term;
    SourceSpan span;
  };

  // ---------------------------
  // AST: declarations
  // ---------------------------

  struct FieldDecl {
    std::string name;
    TypePtr type;
    SourceSpan span;
  };

  /**
   * User-defined struct declaration.
   */
  struct StructDecl {
    GlobalId name;
    std::vector<FieldDecl> fields;
    SourceSpan span;
  };

  enum class SymKind { Value, Coef, Index };

  struct DomainInterval {
    std::int64_t lo = 0;
    std::int64_t hi = 0;
    SourceSpan span;
  };

  struct DomainSet {
    std::vector<std::int64_t> values;
    SourceSpan span;
  };

  using Domain = std::variant<DomainInterval, DomainSet>;

  /**
   * Symbolic variable declaration.
   */
  struct SymDecl {
    SymId name;
    SymKind kind;
    TypePtr type;
    std::optional<Domain> domain;
    SourceSpan span;
  };

  struct InitVal;
  using InitValPtr = std::shared_ptr<InitVal>;

  // Forward-declare Atom so InitVal can carry one without dragging the
  // full Atom definition above this point in the header.
  struct Atom;
  using AtomPtr = std::shared_ptr<Atom>;

  /**
   * Initializer value for variables.
   *
   * §3.4.2: a non-aggregate target may use any `Atom` as its
   * initializer (e.g. `let %p: ptr i32 = addr %x;`,
   * `let %v: i32 = load %p;`, `let %m: i1 = cmp < %a, %b;`,
   * `let %lane: i32 = %v[0];`). For aggregate targets the spec still
   * restricts to BraceInit / literals / names / undef / null.
   */
  struct InitVal {
    enum class Kind {
      Int,
      Float,
      Sym,
      Local,
      Undef,
      Aggregate,
      Null,
      Atom, // atom-form init: load/addr/cmp/ptrindex/lvalue-with-accesses/etc.
    } kind;
    std::variant<IntLit, FloatLit, SymId, LocalId, std::vector<InitValPtr>, AtomPtr> value;
    SourceSpan span;
  };

  /**
   * Local variable declaration (mutable or immutable).
   */
  struct LetDecl {
    bool isMutable = false;
    LocalId name;
    TypePtr type;
    std::optional<InitVal> init;
    SourceSpan span;
  };

  /**
   * Function parameter declaration.
   */
  struct ParamDecl {
    LocalId name;
    TypePtr type;
    SourceSpan span;
  };

  /**
   * Function declaration.
   */
  struct FunDecl {
    GlobalId name;
    std::vector<ParamDecl> params;
    TypePtr retType;
    std::vector<SymDecl> syms;
    std::vector<LetDecl> lets;
    std::vector<Block> blocks;
    SourceSpan span;
    // Stem of the .sir file this fun came from (no extension,
    // no directory). Empty for the primary translation unit; populated
    // by the link resolver when the fun is moved in from an -I lib.
    // The C backend uses this to split per-source `<stem>.c` outputs
    // when --split-by-source is enabled.
    std::string sourceStem;

    // Backend-facing hints. These are not part of the RefractIR
    // surface syntax — the parser does not currently emit them. They
    // are set by upstream tooling that marks individual callees, and
    // consumed by each backend in whatever way the target language
    // supports. The C backend translates `noInline` / `noClone` into
    // the matching `__attribute__((noinline))` / `__attribute__((noclone))`
    // qualifier (combined when both are set); the WASM backend ignores
    // both because WAT has no portable inlining / cloning suppression
    // directive. Keeping these as a tagged struct (instead of a
    // free-form attribute list) means each backend interprets only
    // the hints it understands and ignores the rest by construction.
    struct Attributes {
      bool noInline = false;
      bool noClone = false;
    } attributes;
  };

  /**
   * A pre-clause inside a contract: `pre <cond>(, "msg")?;`.
   */
  struct PreClause {
    Cond cond;
    std::optional<std::string> message;
    SourceSpan span;
  };

  /**
   * A post-clause inside a contract: `post <cond>(, "msg")?;`.
   * `ret` may appear as a bareword identifier inside the cond, referring
   * to the callee's return value (handled by parser via InPostClause flag).
   */
  struct PostClause {
    Cond cond;
    std::optional<std::string> message;
    SourceSpan span;
  };

  /**
   * A behavioral contract on an external declaration:
   *   `{ pre... post... }` (zero or more pre, at least one post).
   */
  struct Contract {
    std::vector<PreClause> pres;
    std::vector<PostClause> posts;
    SourceSpan span;
  };

  /**
   * External function declaration.
   *
   * Two forms (mutually exclusive):
   *   - Link form: signature only (`contract` is std::nullopt).
   *     The body must be found in another `.sir` file via `-I`. The
   *     LinkResolver pass attaches `resolvedBody`.
   *   - Contract form: signature plus a `{ pre... post... }` block.
   *     The body is NEVER expected elsewhere; the contract IS the spec.
   *
   * `resolvedBody` is a non-owning pointer (the resolver owns the parsed
   * external Program). It is set only for link-form decls.
   */
  struct ExtDecl {
    GlobalId name;
    std::vector<ParamDecl> params;
    TypePtr retType;
    std::optional<Contract> contract;
    const FunDecl *resolvedBody = nullptr;
    SourceSpan span;
  };

  /**
   * A built-in intrinsic declaration. The toolchain owns the
   * semantics — no body, no contract, no `-I` resolution.
   */
  struct IntrinsicDecl {
    GlobalId name;
    std::vector<ParamDecl> params;
    TypePtr retType;
    SourceSpan span;
  };

  /**
   * Represents a complete RefractIR program.
   */
  struct Program {
    std::vector<StructDecl> structs;
    std::vector<FunDecl> funs;
    std::vector<ExtDecl> extDecls;
    std::vector<IntrinsicDecl> intrinsics;
    SourceSpan span;
  };

  // ---------------------------
  // Utilities
  // ---------------------------
  inline int64_t parseIntegerLiteral(const std::string &s) {
    // The literal might be:
    //   0x... (hexadecimal)
    //   0o... (octal)
    //   0b... (binary)
    //   ...or decimal by default.
    // Or with a leading '-'.
    if (s.size() >= 2) {
      size_t n = (s[0] == '-') ? 1 : 0;
      if (s.size() > n + 1 && s[n] == '0') {
        if (s[n + 1] == 'x' || s[n + 1] == 'X')
          return std::stoll(s, nullptr, 16);
        if (s[n + 1] == 'o' || s[n + 1] == 'O') {
          std::string octal = (n == 1 ? "-" : "") + s.substr(n + 2);
          return std::stoll(octal, nullptr, 8);
        }
        if (s[n + 1] == 'b' || s[n + 1] == 'B') {
          std::string binary = (n == 1 ? "-" : "") + s.substr(n + 2);
          return std::stoll(binary, nullptr, 2);
        }
      }
    }
    return static_cast<int64_t>(std::stoll(s, nullptr, 10));
  }

  // Bit-exact decimal serialization for finite doubles. Pairs with
  // parseFloatLiteral below: every formatDouble(d) string parses back
  // to exactly d, including subnormals and signed zero.
  //
  // RefractIR's finite-only FP model (±inf / NaN are UB) means we never
  // emit "inf" or "nan" here; if they ever leak through this still
  // produces a strtod-parseable string ("inf" / "-inf" / "nan").
  //
  // Format choice — std::to_chars(shortest) — gives the shortest
  // decimal string that round-trips to exactly d (Ryū/Grisu). That
  // means `1.0` stays `"1"`, `0.1` stays `"0.1"`, and edge cases like
  // subnormals get whatever minimal digits suffice. Bit-exact AND
  // readable.
  //
  // We post-process two cosmetic-but-load-bearing things the standard
  // doesn't give us:
  //   * If the output has no '.' or 'e', append `.0`. This protects
  //     consumers that dispatch int-vs-float on those characters (see
  //     parseNumberLiteral); without it, an integer-valued double like
  //     2.0 would print as `"2"` and get re-classified as i64 by some
  //     readers.
  //   * Signed zero: to_chars(0.0) → "0", to_chars(-0.0) → "-0". The
  //     "-0" form *does* round-trip through strtod (sign preserved),
  //     but the int-vs-float dispatch hazard above still applies, so
  //     the `.0` append handles both at once: "0.0" / "-0.0".
  inline std::string formatDouble(double d) {
    char buf[32];
    auto res = std::to_chars(buf, buf + sizeof(buf), d);
    if (res.ec != std::errc())
      return std::to_string(d); // fallback; should not happen for finite d
    std::string out(buf, res.ptr);
    if (out.find('.') == std::string::npos && out.find('e') == std::string::npos &&
        out.find('E') == std::string::npos && out.find('n') == std::string::npos &&
        out.find('i') == std::string::npos)
      out += ".0";
    return out;
  }

  inline double parseFloatLiteral(const std::string &s) {
    // strtod sets errno = ERANGE both on overflow (returns ±HUGE_VAL)
    // and on subnormal underflow (returns the denormal value it managed
    // to round to). std::stod conflates the two — libstdc++ throws
    // out_of_range on either — so a perfectly representable subnormal
    // like `-9.881…e-324` aborts the interpreter when forwarded as a
    // positional CLI arg. Use strtod directly and accept the underflow
    // case: the returned value is the correct denormal. Overflow still
    // surfaces as an exception so a true out-of-range literal isn't
    // silently saturated to inf.
    const char *p = s.c_str();
    char *end = nullptr;
    errno = 0;
    double v = std::strtod(p, &end);
    if (end == p)
      throw std::invalid_argument("parseFloatLiteral: not a number: " + s);
    if (errno == ERANGE && (v == HUGE_VAL || v == -HUGE_VAL))
      throw std::out_of_range("parseFloatLiteral: overflow: " + s);
    return v;
  }

  inline std::variant<int64_t, double> parseNumberLiteral(const std::string &s) {
    if (s.find('.') != std::string::npos || s.find('e') != std::string::npos ||
        s.find('E') != std::string::npos) {
      return parseFloatLiteral(s);
    }
    return parseIntegerLiteral(s);
  }
} // namespace refractir
