# RefractIR v0.2.3 Specification

**Status:** Draft — v0.2.3 is **in progress**; this document is the roadmap for the release

The [bounded tensor profiles](../src/analysis/tensor.cpp) support a separate extension for [tensor case generation](./reify.md#tensor-case-generation) and [Triton emission](./symirc.md#triton-tensor-target). They do not alter this release's scalar, vector, pointer, or aggregate contract or its shipped-feature status.

This document is the complete, standalone specification for RefractIR v0.2.3. It supersedes v0.2.2. It is written as a **follow-up of v0.2.2**: every feature the v0.2.2 specification slated for v0.2.3 (§13 there) appears here with its full design, and v0.2.3 is complete when every **[Planned]** item below has shipped. Sections and rules unchanged from earlier versions are included verbatim for self-containedness, keeping their historical markers (**[v0.2.2]**, **[v0.2.1]**); everything specific to this release is marked **[New in v0.2.3]** together with its implementation status:

- **[Shipped]** — implemented and tested on the v0.2.3 line.
- **[Planned]** — design is normative and final; implementation has not landed yet.
- **[Dropped]** — considered for v0.2.3 and rejected, with rationale recorded in §13.


## What's new in v0.2.3

### Roadmap at a glance

| # | Feature | Origin | Status |
|---|---------|--------|--------|
| T1 | Python compilation target | new in v0.2.3 | **Shipped** |
| T2 | WASM SIMD-128 vector lowering (`--vec-lowering`) | v0.2.2 §13 *WASM SIMD support* | **Shipped** |
| T3 | Intrinsic completion on WASM (checksum primitives) | v0.2.2 §12 Batch R WASM gap | **Shipped** |
| V1 | Horizontal vector reductions (`@reduce_*`) | v0.2.2 §13 | **Shipped** — §12.4 |
| V2 | Vector shuffles (`shuffle` atom) | v0.2.2 §13 | **Planned** — §5.3, §6.14 |
| V3 | Addressable vectors (`ptr <N> T`, whole-vector `load`/`store`) | v0.2.2 §13 | **Planned** — §2.8, §6.13, §7.8, §9.8 |
| V4 | Vectors in aggregates (struct fields, array elements) | v0.2.2 §13 | **Planned** — §3.2, §6.13 |
| F1 | Function attributes (`inline`, `noinline`, `pure`, `const`) | v0.2.2 §13 | **Planned** — §3.6, §6.15 |
| — | Relaxed SIMD | v0.2.2 §13 *WASM SIMD support* | **Dropped** — §13 |

T1–T3 changed only the compilation targets and their lowering guarantees. V1–V4 and F1 extend the language surface; until they land, the surface accepted by the shipped tools is exactly the v0.2.2 surface. V1–V4 are numbered in the intended implementation order, easiest first: V1 and V2 are register-only (no memory-model impact), V3 opens the memory model to vector cells, and V4 builds on V3.

### T1. Python compilation target **[Shipped]**

- **Third first-class target**. A RefractIR program may be lowered to a Python module. Concrete semantics (values, UB traps, symbol providers) are identical to the C and WASM targets.
- **Reducibility requirement**: the Python target accepts only **reducible** CFGs — its emission reconstructs genuine `while`/`if` control flow and cannot express irreducible regions; irreducible programs are rejected at compile time. The C and WASM targets accept any CFG. Reducibility is a per-program property of the CFG (every loop has a single entry, i.e., every retreating edge is a back edge to a dominator); see [`docs/reducibility.md`](./reducibility.md).

### T2. WASM SIMD-128 vector lowering **[Shipped]**

- Fulfils the v0.2.2 §13 plan: the WASM target now lowers vector locals to native `v128` registers by default (`i8x16`/`i16x8`/`i32x4`/`i64x2`/`f32x4`/`f64x2` lane shapes; shapes wider than 16 bytes split across registers, as §10.16 of the v0.2.1 spec permits). Lane-wise semantics (§2.11, §6.9, §7.6) are unchanged. The strategy is selectable per compilation via `--vec-lowering` (§11.6).
- Vector values now cross the WASM call boundary under a defined ABI: arguments spill to caller-owned frame memory and pass by address; returns arrive through a hidden trailing sret address parameter. The by-value semantics of §2.11 are preserved.
- The other half of the v0.2.2 *WASM SIMD support* bullet, the **Relaxed SIMD** extension, is **dropped**: its instructions have implementation-defined results, which is irreconcilable with RefractIR's bit-exact cross-target semantics and strict UB model (§13).

### T3. Intrinsic support completion **[Shipped]**

- The checksum primitives `@crc32_update` / `@check_chksum` (§12.7 of [`docs/intrinsics.md`](./intrinsics.md)) now lower on WASM. Every shipped intrinsic lowers on every compiled target; the WASM rejection layer of the intrinsic tiering is empty.

### V1. Horizontal vector reductions **[Shipped]**

- New intrinsics `@reduce_add`, `@reduce_min`, `@reduce_max` (integer and FP) and `@reduce_and`, `@reduce_or`, `@reduce_xor` (integer only) fold all lanes of a vector into one scalar (§12.4). The sequential lane-order fold is normative; `@reduce_mul` is rejected as solver-hostile (nonlinear BV multiplication chains).

### V2. Vector shuffles **[Planned]**

- New `shuffle` atom: `shuffle %v, %w, {i0, i1, …}` selects lanes from the concatenation of two same-typed vectors under a **static literal index list**, producing a new vector (§5.3, §6.14). A swizzle is a `shuffle` with both operands the same vector.
- Index validity is checked statically (a type error, never UB); lane values — including `undef` — propagate exactly as in whole-vector copy.

### V3. Addressable vectors **[Planned]**

Completes the design pre-committed in v0.2.1 §11 / v0.2.2 §13:

- `<N> T` joins the **loadable-type set** with `sizeof(<N> T) = N * sizeof(T)` (packed).
- `ptr <N> T` becomes a valid pointer type; `addr %v` on a `let mut` vector local yields one.
- `load p` / `store p, v` through `ptr <N> T` move the **whole vector** (LLVM-style).
- Lanes remain **not individually addressable**: `addr %v[i]` stays forbidden, and no pointer may point into the interior of a vector cell — a vector cell is an indivisible leaf of the memory model (§2.8, §7.8).

### V4. Vectors in aggregates **[Planned]**

Builds on V3:

- `<N> T` is admitted as a struct field type and as an array element type (`[M] <N> T`). Nested vectors (`<M> <N> T`) remain forbidden — the element type of a vector is always scalar.
- `ptrindex` / `ptrfield` navigate down to `ptr <N> T` leaves; navigation stops at the vector cell (§6.13).

### F1. Function attributes **[Planned]**

- `fun` and `decl` headers accept attributes after the return type: `inline`, `noinline` (pure lowering hints) and `pure`, `const` (statically enforced effect contracts that the solver exploits — a `pure`/`const` contract `decl` skips memory havoc; a `const` callee is memoizable). Design in §3.6, well-formedness in §6.15. `noreturn` is **dropped** (§13): `call` is an expression that must produce a value, so a no-return callee contradicts the expression-call model.


## 1. Notation and identifier classes

RefractIR uses sigils to make identifier categories immediately recognizable:

- `@name` — global identifiers (functions; global type names if desired).
- `%name` — local identifiers (parameters, locals).
- `@?name` — **global symbols** (solver-chosen unknowns).
- `%?name` — **local symbols** (solver-chosen unknowns).
- `^name` — basic block labels.

**Hard rule:** `?` is permitted **only** immediately after `@` or `%` to form `@?` / `%?`. It is forbidden in all other identifiers.


## 2. Key semantic commitments (v0.2.3)

### 2.1 Non-SSA and mutable store
RefractIR is not SSA. Locals declared with `let mut` denote **mutable storage cells**. Assignments update the store at the given lvalue location.

### 2.2 Path-based execution
Given a user-chosen path `π` (e.g., `^entry -> ^b1 -> ^b3 -> ^b1 -> ^exit`), the tool executes blocks along `π` in order. Only statements and terminators encountered on `π` contribute constraints.

### 2.3 `assume` vs `require`
- `assume <cond>;` adds **feasibility/admissibility** constraints (part of template semantics).
- `require <cond>;` adds **property/synthesis** constraints (must hold on the chosen path).

### 2.4 Expressions: flat, left-to-right, no parentheses
- Expressions contain **no parentheses**.
- Expressions are a left-to-right chain of atoms combined by `+` and `-` only.
- Evaluation order is **left-to-right** (no reassociation or reordering).

### 2.5 `div` / `mod` round toward 0
Both integer and floating-point division and modulo round toward 0 (C-like truncation semantics). For floats, `%` is `fmod`, not IEEE `remainder`. See §8.

### 2.6 Strict undefined behavior (UB)
RefractIR uses **strict UB** on the chosen path: if UB occurs during evaluation of any statement or condition on `π`, the path becomes **infeasible** and is pruned.

### 2.7 `select` expression (lazy)
`select` is supported as an atom:
- `select <cond>, <vtrue>, <vfalse>` — cond form, scalar predicate.
- `select <mask>, <vtrue>, <vfalse>` — mask form, `<N> i1` or `i1` per-lane blend.
- `select` is **lazy**: only the selected arm is evaluated.
- `vtrue` and `vfalse` are restricted to **scalar, pointer, or vector values** (`RValue`, constant `Coef`, or `null`). Both arms must have the same type.

### 2.8 Memory model
RefractIR uses a typed, stack-only memory model:

- **Stack-only**: all addressable storage is a `let mut` local within the current function. Heap allocation is not supported.
- **No cross-object aliasing**: a pointer derived from `addr %x` can never alias a pointer derived from `addr %y` for distinct locals `%x` and `%y`. Cross-object pointer arithmetic is **UB**.
- **Typed memory regions**: the memory is conceptually partitioned by pointee type. Each distinct pointee type `T` has its own independent memory region.
- **Pointer width**: all pointers are 64-bit values (BV64 in the SMT model), regardless of pointee type.
- **Vector cells [New in v0.2.3 — Planned, V3]**: `<N> T` joins the loadable-type set. A memory cell of vector type is an **indivisible leaf**: it is read and written only as a whole (`load`/`store` through `ptr <N> T`), its lanes have no independent addresses, and no pointer may point into its interior. Each vector shape `<N> T` has its own typed memory region `Mem[<N> T]`, disjoint from the scalar region `Mem[T]`. Until V3 lands, the shipped tools keep the v0.2.2 behavior: vectors are pure register-like value types with no `ptr <N> T` and no `addr` on vector locals.

### 2.9 Floating-point value model

RefractIR uses **finite IEEE 754-2008 semantics** for floating-point:

- **Domain**: the only valid floating-point values are **finite** IEEE 754 values. ±∞ and NaN are **not** RefractIR values. Any operation whose IEEE 754 result would be ±∞ or NaN is UB (see §7.4).
- **Signed zeros**: `+0.0` and `-0.0` are distinct bit patterns and both are valid values. They compare equal (`+0.0 == -0.0` is `true`) but are **not the same value** — `@signbit` and `@to_bits` tell them apart. Comparison and value identity are therefore distinct relations on floats: the `==` / `!=` operators and `cmp` are IEEE comparison (SMT `fp.eq`), while value identity is bit-exact (SMT `=`). The two agree everywhere except on `±0.0`. The language provides only the comparison form; an analysis needing value identity must express it as `@to_bits` equality or a bit-level comparison in its own layer. See [`docs/float.md`](./float.md) §2.1.
- **Subnormals**: subnormal (denormal) values are regular finite values. No flush-to-zero behavior.
- **Rounding mode**: all operations use a single fixed mode — **RNE (Round to Nearest, Ties to Even)**.
- **`%` for floats**: the `%` operator is **C's `fmod`** (truncated-quotient remainder), **not** IEEE 754 `remainder` (`fp.rem`).

SMT encoding: `f32` maps to `(_ FloatingPoint 8 24)` and `f64` maps to `(_ FloatingPoint 11 53)`. All FP operations use `roundNearestTiesToEven` except `%`, which encodes as `fp.sub(x, fp.mul(fp.roundToIntegral[RTZ](fp.div[RNE](x, y)), y))`.

**Bit-exact text serialization.** Floats survive every text boundary
(`.sir` source, descriptor JSON, SOLVED / PARAMS / RETURN headers,
model-dump files, CLI positional args) bit-exactly. The implementation
enforces this via one canonical pair:

- `formatDouble(d)` (`include/ast/ast.hpp`) — emits the shortest decimal
  string that round-trips, using `std::to_chars(…, chars_format::shortest)`,
  with `.0` appended when neither `.` nor an exponent is present (so a
  reader cannot misclassify an integer-valued double as an integer, and
  `-0.0` keeps its sign).
- `parseFloatLiteral(s)` (`include/ast/ast.hpp`) — wraps `std::strtod`
  directly; subnormal values are accepted, only true `±HUGE_VAL`
  overflow raises. **`std::stod` is forbidden in this codebase** —
  libstdc++ raises `out_of_range` on any `ERANGE`, including valid
  subnormals.

The C and WASM backends use their own bit-exact formatters because they
emit C / WAT literals respectively, with target-language-specific
suffixes; that divergence is intentional and is the only place where
`formatDouble` is not the canonical RefractIR-text formatter.

### 2.10 Pointer arithmetic

- `ptr T + n` advances the address by `n * sizeof(T)` bytes, where `n` is an integer operand.
- `ptr T - n` retreats by `n * sizeof(T)` bytes.
- `ptr T - ptr T` yields the element distance as `i64`.
- `integer + ptr T` is **not supported**.
- Arithmetic is valid only within the bounds of the originating object. Out-of-bounds is UB.

**`sizeof`**:

| Type | `sizeof` |
|------|---------|
| `iN` | `⌈N / 8⌉` bytes |
| `f32` | 4 bytes |
| `f64` | 8 bytes |
| `ptr T` | 8 bytes |
| `[N] T` | `N * sizeof(T)` bytes |
| `@S` | `Σ sizeof(field_i)` bytes (packed, no padding) |
| `<N> T` | `N * sizeof(T)` bytes (packed) **[New in v0.2.3 — Planned, V3]** |

### 2.11 Vector types

Vectors are fixed-width SIMD value types `<N> T` where `T` is a scalar type (`iN`, `f32`, `f64`) and `N ≥ 2`. Lane-wise arithmetic lifts all scalar operators. `cmp` produces `<N> i1` masks.

**[New in v0.2.3 — Planned]** v0.2.3 completes the vector story deferred from v0.2.1/v0.2.2:

- **V1 — horizontal reductions**: `@reduce_*` intrinsics fold all lanes into one scalar. See §12.4.
- **V2 — shuffles**: the `shuffle` atom rearranges lanes under a static index list. See §5.3, §6.14.
- **V3 — addressable vectors**: `addr %v` on a `let mut` vector local yields `ptr <N> T`; `load`/`store` move whole vectors. Lanes stay unaddressable (`addr %v[i]` remains forbidden). See §2.8, §6.13, §7.8, §9.8.
- **V4 — vectors in aggregates**: `<N> T` may be a struct field or array element; `ptrindex`/`ptrfield` navigate to vector-cell leaves. See §3.2, §6.13.

Until these land, the shipped tools accept exactly the v0.2.2 vector surface (register-only value types).

### 2.12 Function calls and interprocedural execution **[v0.2.2]**

- **`call @f(args...)`** evaluates arguments left-to-right, then transfers control to `@f`. The result is the callee's return value.
- **`fun` target**: a fresh callee context is created; parameters are bound to argument values. The callee's blocks execute along a sub-path. `Store`, `Mem`, `PC`, and `REQ` flow into and back from the callee. UB in the callee makes the calling path infeasible.
- **Contract `decl` target**: `pre` clauses are checked at the call site (violation → UB). `post` clauses are assumed true after the call, constraining the return value and post-call memory.
- **Link `decl` target**: the tool locates the `fun` body via `-I` and executes it as above. If no body is found, execution fails — the tool reports an error before producing any result.
- **`intrinsic` target**: the call is replaced by the intrinsic's built-in semantics (interpreter) or SMT encoding (solver) — see §12.
- **Calls in conditions and assumptions**: `call` is an atom and may appear inside `br`, `assume`, and `require` conditions. **Side effects (memory mutation, `PC`/`REQ` updates) commit unconditionally as part of evaluating the condition** — even if the resulting branch is not taken, even if an `assume` later renders the path infeasible. This is intentional: strict left-to-right evaluation order means the call's effects materialize at the point of evaluation.
- **`select` arms exclude `call`**: `select` arms (`SelectVal`) are restricted to `RValue` and `Coef` precisely because `select` is lazy. Allowing `call` in a lazy arm would make side-effect commitment depend on the predicate, breaking the strict left-to-right model.


## 3. Concrete syntax

### 3.1 Lexical
- `Ident` : `[A-Za-z_][A-Za-z0-9_]*`
- `Nat` : `[0-9]+`
- `IntLit` :
  - Decimal: `"-"? [0-9]+`
  - Hexadecimal: `"-"? "0x" [0-9A-Fa-f]+`
  - Octal: `"-"? "0o" [0-7]+`
  - Binary: `"-"? "0b" [01]+`
- `FloatLit` : standard floating point literal (e.g. `1.5`, `-0.2`, `1e-5`, `3.14E+2`).
- `StringLit` : double-quoted string (implementation-defined escapes)

**[v0.2.2]** New keywords: `call`, `decl`, `intrinsic`, `pre`, `post`, `ret`.

**[New in v0.2.3 — Planned]** New keywords: `shuffle` (V2), `inline`, `noinline`, `pure`, `const` (F1).

### 3.2 Types

```ebnf
Type        := IntType | FloatType | StructName | ArrayType | PtrType | VecType ;
IntType     := "i" Nat ;
FloatType   := "f32" | "f64" ;
ArrayType   := "[" Nat "]" Type ;
StructName  := GlobalId ;
PtrType     := "ptr" PointeeType ;
PointeeType := IntType | FloatType | PtrType | ArrayType | StructName
            | VecType ;                        (* [New in v0.2.3 — Planned, V3] *)
VecType     := "<" Nat ">" Type ;
```

The element type of a `VecType` must be scalar (`iN`, `f32`, `f64`); nested vectors are forbidden.

**[New in v0.2.3 — Planned, V4]** The v0.2.1 restriction that vectors cannot appear inside other types is lifted for aggregates: `<N> T` is now a valid struct field type and array element type (`[M] <N> T`). It remains invalid as a vector element type.

### 3.3 Program structure **[v0.2.2]**

```ebnf
Program     := (StructDecl | FunDecl | ExtDecl | IntrinsicDecl)* ;

FunDecl     := "fun" GlobalId "(" ParamList? ")" ":" Type Attr*
               "{" SymDecl* LetDecl* Block+ "}" ;

ExtDecl     := "decl" GlobalId "(" ParamList? ")" ":" Type Attr*
               ( Contract | ";" ) ;

Attr        := "inline" | "noinline" | "pure" | "const" ;   (* [New in v0.2.3 — Planned, F1] *)

Contract    := "{" PreClause* PostClause+ "}" ;

PreClause   := "pre" Cond ("," StringLit)? ";" ;
PostClause  := "post" Cond ("," StringLit)? ";" ;

IntrinsicDecl := "intrinsic" GlobalId "(" ParamList? ")" ":" Type ";" ;
```

**[v0.2.2]** A `decl` may be:
- **Link form**: `decl @name(...) : Type;` — signature only. The body must be found via `-I` paths.
- **Contract form**: `decl @name(...) : Type { pre ... post ... };` — a behavioral contract. The body is **not** expected elsewhere.

A `fun` **never** has a contract. The `fun` body is the ground truth for both interpretation and solving. If a `fun` body and a `decl` with contract share the same name across compilation units (found via `-I`), the tool reports an error — body and contract are mutually exclusive per function name.

**`fun` bodies vs `decl` contracts are mutually exclusive**: a given function name `@f` may be defined by exactly one `fun` (possibly in another `.sir` file found via `-I`) **or** by exactly one `decl` with a contract — never both.

Declarations must appear before any `call` that references them (no forward references to names not yet declared, except that link-form `decl` may reference a `fun` in a yet-to-be-scanned file — the name is available after the `decl` is parsed).

**Two-pass loading**: tools perform an initial scan of the primary file and all `-I` directories to collect every top-level declaration (`fun`, `decl`, `intrinsic`) before any body-checking, call-graph construction, or symbolic execution begins. This pre-scan is parse-only (signatures and contracts, no body checking) and is what makes link-form `decl` resolution and recursion detection possible. For large `-I` trees the cost is linear in the number of `.sir` files visible.

### 3.4 Contract semantics **[v0.2.2]**

- **`pre <cond>`**: a **precondition**. Evaluated in the caller's context at the call site with parameters bound to argument values. If any `pre` clause evaluates to `false`, the call is UB (path infeasible). `ret` must **not** appear in `pre` clauses.
- **`post <cond>`**: a **postcondition**. After the call returns, `<cond>` is assumed true. It may reference:
  - The function's parameters (which are immutable — their values equal the arguments passed).
  - The special identifier **`ret`**, representing the return value. `ret` has the return type of the `decl`.
  - Caller-side locals in scope at the call site.
- All pointer parameters are **modifiable** — the callee may write through any pointer parameter. A contract makes no implicit guarantee about what is or is not written. If the caller needs a specific post-state, it expresses it in a `post` clause or via caller-side temporaries (see below).
- A contract must contain at least one `post` clause. `pre` clauses are optional; a contract with no `pre` clauses imposes no preconditions.

**No `old()`**: to express a relationship between pre- and post-call state, the caller saves the pre-call value in a temporary and compares explicitly (§10.4).

**Contract-only `decl`**: a `decl` with a contract has no body anywhere. The interpreter rejects `call` on such a `decl` with an error — it has no concrete semantics to execute. The solver uses the contract as its behavioral model.

### 3.5 Declarations

#### 3.5.1 Symbols (`sym` implies immutable `let`)
A `sym` declaration introduces a solver-chosen unknown. Symbols are **immutable**.

```ebnf
SymDecl     := "sym" SymId ":" SymKind Type Domain? ";" ;
SymKind     := "value" | "coef" | "index" ;
Domain      := "in" Interval | "in" Set ;
Interval    := "[" IntLit "," IntLit "]" ;
Set         := "{" IntLit ("," IntLit)* "}" ;
```

**Restrictions:** `sym` of pointer type is not allowed. `sym` of aggregate (array / struct) type is likewise not allowed — the language has no consumer syntax for aggregate symbols (whole-aggregate-copy initializers are banned and symbols take no element access); see §13 for the plan. `sym` of vector type is allowed (per-lane independent symbols).

#### 3.5.2 Locals (`let` and `let mut`)
```ebnf
LetDecl     := "let" ("mut")? LocalId ":" Type ("=" InitVal)? ";" ;
InitVal     := ScalarInit | "null" | "undef" | BraceInit | AtomInit ;
ScalarInit  := IntLit | FloatLit | SymId | LocalId ;
BraceInit   := "{" InitVal ("," InitVal)* "}" ;
```

**[v0.2.2]** `AtomInit` extends the v0.2.1 atom-form initializer to include `call`:

```text
let %y: i32 = call @helper(%x);
```

`call` may appear as an initializer for any non-aggregate target. The referenced function must be declared earlier in the source.

Other initialization rules (broadcast, brace, `null`, `undef`) are unchanged from v0.2.1.

**`addr` restriction**: only a `let mut` local (or a sub-lvalue rooted at a `let mut` local) may appear as the operand of `addr`. Parameters and `let` (immutable) locals cannot have their address taken.

**[New in v0.2.3 — Planned, V3]** `addr %v` on a `let mut` vector local (or a vector-typed sub-lvalue such as `%s.f` or `%a[i]` rooted at a `let mut` local) is valid and yields `ptr <N> T`. `addr` on a vector **lane** (`addr %v[i]`) remains forbidden — lanes have no independent addresses.

### 3.6 Function attributes **[New in v0.2.3 — Planned, F1]**

Attributes appear between the return type and the body (`fun`) or the contract/semicolon (`decl`). They form two independent groups; at most one attribute of each group may be present:

- **Lowering hints** — `inline` | `noinline`. No semantic content: the interpreter and solver ignore them entirely. The C backend emits `static inline` / `__attribute__((noinline))`; the WASM and Python backends ignore them. Because they are semantics-free, they impose no checking obligations.
- **Effect contracts** — `pure` | `const`. Statically enforced restrictions on the callee's effects, checked by the semantic checker over the (acyclic) call graph:
  - **`pure`**: the body contains no `store` instruction and calls only `pure` or `const` callees. It may `load` — the result may depend on memory reachable from pointer parameters, but the call mutates nothing.
  - **`const`**: `pure`, and additionally the body contains no `load` — the result depends only on the argument values. `const` implies `pure`; writing both is an error.

**Enforcement.** For a `fun` (or a link-form `decl`, checked against its resolved body), the checker verifies the restriction syntactically over the body and transitively over callees; a violation is a semantic error at check time. For a contract-form `decl` there is no body: the attribute is a **trusted assumption**, exactly like its `post` clauses. All shipped intrinsics are treated as `const` for this check, except `@check_chksum` (impure — it aborts on mismatch) and `@observe` (impure — its C lowering performs an observable `volatile` write; its *value* is nonetheless the deterministic identity).

**Solver exploitation.**

- A `pure` or `const` callee writes no memory, so at contract-form call sites the memory havoc of §9.6.2 step 4 and the caller-`Store` invalidation of step 5 are **skipped** — pointer parameters are known unmodified.
- A `const` callee's result is a function of its arguments alone. The solver may **memoize**: two calls to the same `const` contract-form `decl` with syntactically identical argument terms share one `ret_sym` (instead of one fresh symbol per call site), and repeated `fun` expansions with identical arguments may share one expansion.

**`noreturn` is dropped** (it appeared in the v0.2.2 §13 sketch): every RefractIR function returns a value and `call` is an expression that must produce one, so a no-return callee has no coherent value semantics. Dead ends are expressed with the `unreachable` terminator. See §13.


## 4. CFG blocks and instructions

### 4.1 Blocks and terminators
```ebnf
Block       := BlockLabel ":" Instr* Terminator ;
BlockLabel  := "^" Ident ;

Terminator  := BrTerm | RetTerm | UnreachTerm ;

BrTerm      := "br" (Cond "," BlockLabel "," BlockLabel | BlockLabel) ";" ;
RetTerm     := "ret" Expr? ";" ;
UnreachTerm := "unreachable" ";" ;
```

### 4.2 Instructions
```ebnf
Instr       := AssignInstr | AssumeInstr | RequireInstr | StoreInstr ;

AssignInstr := LValue "=" Expr ";" ;
AssumeInstr := "assume" Cond ";" ;
RequireInstr:= "require" Cond ("," StringLit)? ";" ;
StoreInstr  := "store" RValue "," Expr ";" ;
```

**`store`**: `store <ptr>, <val>` writes `val` to the memory location addressed by `ptr`. `store` is a **statement**; it does not produce a value.


## 5. LValues, conditions, expressions

### 5.1 LValues and access paths
```ebnf
LValue      := Base Access* ;
Base        := LocalId ;
Access      := "[" Index "]" | "." Ident ;
Index       := IntLit | LocalId | SymId ;
```

### 5.2 Conditions
```ebnf
Cond        := Expr RelOp Expr ;
RelOp       := "==" | "!=" | "<" | "<=" | ">" | ">=" ;
```

### 5.3 Expressions (no parentheses, left-to-right) **[v0.2.2]**
```ebnf
Expr        := Atom (("+" | "-") Atom)* ;

Atom        := Coef "*" RValue
            | Coef "/" RValue
            | Coef "%" RValue
            | Coef "&" RValue
            | Coef "|" RValue
            | Coef "^" RValue
            | Coef "<<" RValue
            | Coef ">>" RValue
            | Coef ">>>" RValue
            | "~" RValue
            | Select
            | Cast
            | AddrOf
            | Load
            | Cmp
            | PtrIndex
            | PtrField
            | Call                         (* [v0.2.2] *)
            | Shuffle                      (* [New in v0.2.3 — Planned, V2] *)
            | Coef
            | RValue ;

Select      := "select" ( Cond | Expr ) "," SelectVal "," SelectVal ;
SelectVal   := RValue | Coef ;

Cast        := RValue "as" Type ;

AddrOf      := "addr" LValue ;
Load        := "load" RValue ;

Cmp         := "cmp" RelOp SelectVal "," SelectVal ;
PtrIndex    := "ptrindex" RValue "," Index ;
PtrField    := "ptrfield" RValue "," Ident ;

Call        := "call" GlobalId "(" ArgList ")" ;      (* [v0.2.2] *)
ArgList     := Expr ("," Expr)* | ε ;

Shuffle     := "shuffle" RValue "," RValue ","
               "{" IntLit ("," IntLit)* "}" ;         (* [New in v0.2.3 — Planned, V2] *)

Coef        := IntLit | FloatLit | LocalId | SymId | "null" ;
RValue      := LValue ;
```

**`call` [v0.2.2]**: invokes the function named by `GlobalId`. Arguments are evaluated left-to-right. The result type is the return type of the called function. `call` is an `Atom` and may appear wherever an `Atom` is expected. Zero-argument calls use empty parentheses: `call @get_value()`.

**`shuffle` [New in v0.2.3 — Planned, V2]**: `shuffle %v, %w, {i0, …, iK-1}` builds a new `<K> T` vector from two operands of the same type `<N> T`. Result lane `j` is lane `i_j` of the 2N-lane concatenation `%v ++ %w` (indices `0..N-1` select from `%v`, `N..2N-1` from `%w`). The index list is a **static literal** — every `i_j` is an `IntLit` in `[0, 2N)` and `K ≥ 2`; violations are type errors (§6.14), never UB. A swizzle is written by repeating the operand: `shuffle %v, %v, {3, 2, 1, 0}`. Lane values propagate exactly as in whole-vector copy: selecting an `undef` lane copies `undef` without UB — UB arises only when the resulting lane is later read (rule 22). Rationale for static-only indices: a symbolic shuffle mask degenerates into N nested selects per lane, which is solver-hostile; dynamic lane access already exists via `%v[%i]`.

### 5.4 `ret` in contract expressions **[v0.2.2]**

Inside `post` clauses of a contract-form `decl`, the identifier `ret` is a reserved name referring to the function's return value. It has the return type declared in the `decl` header. It behaves like an immutable local in scope only within `post` clauses of that specific `decl`.

`ret` is **not** a keyword outside `post` clauses. A local named `%ret` is a distinct identifier (different sigil) and is legal anywhere.

**Parser disambiguation**: the same token `ret` appears as (a) a terminator (`ret Expr?;` in §4.1) and (b) an expression identifier inside `post` clauses. These are unambiguous by context: terminators occur only at block-end positions; the `ret` identifier occurs only inside `post`-clause expressions. The parser distinguishes by the syntactic context in which the token is encountered.


## 6. Typing and well-formedness (v0.2.3)

### 6.1 Scalar arithmetic restriction
Arithmetic is defined over **scalar integer and floating-point leaves**. Pointers may only appear in arithmetic under §2.10.

### 6.2 LValue typing
- If `%x : T` then `%x` has type `T`.
- If `lv : [N] U` then `lv[i] : U`.
- If `lv : S` and struct `S` has field `f : U` then `lv.f : U`.
- If `lv : <N> T` then `lv[i] : T` (vector lane access).

### 6.3 `select` typing
`select c, a, b` is well-typed iff `a` and `b` have the same type and `c` is a boolean condition or `<N> i1` mask. Result has that type.

### 6.4 `as` typing
`rval as T` is well-typed iff both are scalar. Pointer/integer casts are not supported.

**Integer widening: signed sign-extension.** All `iN` types are *signed* N-bit two's-complement integers, including the boolean width `i1`. For `iN as iM` with `N < M`, the result sign-extends bit `N-1`. In particular, `(i1 = true) as iM = -1` for every `M > 1` (the one-bit value `1` sign-extends to all-ones), and `(i1 = false) as iM = 0`. The two representable `i1` values are therefore `{0, -1}` — interpretation as `{0, 1}` is unsupported and would diverge between the interpreter and the backends. `cmp` and predicate intrinsics (`@parity`, `@is_pow2`, `@signbit`, `@is_normal`, `@is_subnormal`) accordingly produce `-1` for true and `0` for false.

**Integer narrowing: truncation [Clarified].** For `iN as iM` with `N > M`, the result is the low `M` bits of the source, reinterpreted as a signed M-bit value (truncate mod `2^M`, then sign-extend bit `M-1`). Narrowing is always defined — never UB, regardless of the source value. This matches SMT `(extract (M-1) 0)` semantics and C's behavior for exact-width destination types; a backend whose storage type for `iM` is wider than `M` (e.g. `i20` stored in `int32_t`) must truncate and sign-extend explicitly.

### 6.5 Bitwise and shift typing
Both operands must be scalar integers of the same bit-width. Pointers not valid.

### 6.6 Mutability rules
- LHS of `=` must be an lvalue rooted at a `let mut` local.
- `sym` identifiers, `let` (immutable) locals, and parameters are immutable.

### 6.7 Floating-point arithmetic typing
- All atoms in the same `+`/`-` chain must have the exact same floating-point type.
- Mixed arithmetic between different float widths or integers/floats is forbidden without explicit `as` casts.

### 6.8 Pointer typing rules
Pointer typing rules (v0.2.1) carry forward. See the v0.2.1 specification for full details on `addr`, `load`, `store`, `null`, pointer arithmetic,-pointer comparison, `ptrindex`, `ptrfield`, and their type rules.

### 6.9 Vector typing rules
Vector typing rules (v0.2.1) carry forward. See v0.2.1 for `cmp`, lane access, whole-vector copy, and mask-based `select` typing. The v0.2.1 "not addressable" restriction is lifted by V3/V4 (§6.13); until they land, the shipped tools still enforce it.

### 6.10 Call typing **[v0.2.2]**

`call @f(e0, ..., eN-1)` is well-typed iff:

- `@f` is declared (via `fun`, `decl`, or `intrinsic`) earlier in the source.
- The number of arguments equals the number of parameters of `@f`.
- For each `i`, the type of `e_i` equals the type of parameter `i` of `@f` (exact match; no implicit conversions).
- If `@f` is a contract-form `decl`, the call target must already be declared. Recursion (direct or mutual) through `fun` or `decl` is forbidden and detected statically.

Result type: the return type of `@f`.

### 6.11 Contract well-formedness **[v0.2.2]**

A contract block `{ pre... post... }` on a `decl` is well-formed iff:

- Every `pre` clause is a well-typed `Cond` whose free variables are a subset of the function's parameters. `ret` must not appear.
- Every `post` clause is a well-typed `Cond` whose free variables are a subset of the function's parameters ∪ `{ret}`.
- `ret` has the return type declared in the `decl` header.
- At least one `post` clause is present.

### 6.12 Literal typing and inference
Literals are inferred from context. Integer default: `i32`. Float default: `f32`. `null` has no default; context must be available.

**Strict range-check [v0.2.2].** Once a literal is inferred at type `iN`, the typechecker rejects any value outside the *signed* range `[-2^(N-1), 2^(N-1)-1]` as a static error — there is no silent narrowing to fit. The check is uniform across decimal, hex, octal, and binary forms: `0x80` is the value 128 (per §3.1), not a "bit pattern" reinterpretation, so it is out of `i8` range and must be written as `-0x80` (= -128) if the author intended the signed-i8 minimum. For `i1`, this means the only representable literals are `0` and `-1`; `1` in `i1` context is rejected.

### 6.13 Vector memory typing **[New in v0.2.3 — Planned, V3/V4]**

- `addr lv` where `lv : <N> T` is an lvalue rooted at a `let mut` local (including a struct field `%s.f` or array element `%a[i]` of vector type) has type `ptr <N> T`.
- `load p` where `p : ptr <N> T` has type `<N> T`. `store p, e` requires `e : <N> T`.
- Pointer arithmetic on `ptr <N> T` follows §2.10 with element size `sizeof(<N> T) = N * sizeof(T)`: `ptr <N> T ± iM → ptr <N> T` strides whole vector cells; `ptr <N> T - ptr <N> T → i64` is a whole-cell element distance.
- `ptrindex p, i` where `p : ptr [M] <N> T` has type `ptr <N> T`. `ptrfield p, f` where field `f : <N> T` has type `ptr <N> T`. Navigation **stops** at the vector cell: there is no `ptrindex` into a `ptr <N> T` (a vector is not an array), and no access path traverses *through* a vector type inside an `addr` operand's lane position (`addr %v[i]` remains a type error).

### 6.14 `shuffle` typing **[New in v0.2.3 — Planned, V2]**

`shuffle a, b, {i0, …, iK-1}` is well-typed iff:

- `a` and `b` have the same vector type `<N> T`.
- `K ≥ 2`, and every index `i_j` is an integer literal with `0 ≤ i_j < 2N`.

Result type: `<K> T`. Index violations are static type errors, not UB.

### 6.15 Attribute well-formedness **[New in v0.2.3 — Planned, F1]**

- At most one of `inline` / `noinline`, and at most one of `pure` / `const`, per declaration; duplicates or conflicts are errors.
- A `pure` body (of a `fun`, or of the body resolved for a link-form `decl`) contains no `store` and calls only `pure`/`const` targets. A `const` body additionally contains no `load`. Checked transitively over the call graph (a DAG per §9.7); violations are semantic errors at check time.
- Attributes on a contract-form `decl` are trusted assumptions (no body exists to check).
- Attributes on `intrinsic` declarations are rejected — intrinsic effect classes are fixed by the toolchain (§3.6).


## 7. Strict UB rules (v0.2.3)

UB is checked during symbolic execution along the chosen path. Any UB makes the path infeasible.

**Static vs runtime.** Rules 1–22 are inherited from v0.2.1. Rules 23–25 are from v0.2.2. Rule 26 is new in v0.2.3 (§7.8, planned with V3), as is rule 27 (§7.5). Conditions caught at check time (undeclared call target, argument-parameter count/type mismatch, recursion cycles, contract-form `decl` call in the interpreter) are semantic errors, not runtime UB — they are rejected before execution begins.

### 7.1 Scalar UB

1. **Integer division/modulo by zero**: `a / b` or `a % b` where `b == 0` (both integer-typed) is UB.

2. **Out-of-bounds array access**: `a[i]` where `a : [N] T` and `i < 0` or `i >= N` is UB.

3. **Reading `undef`**: reading any leaf whose stored value is `undef` is UB. This covers uninitialised locals, uninitialised pointer values, and uninitialised vector lanes.

4. **Signed integer overflow**: `+`, `-`, `*`, `<<` that produce a value outside the representable range of the target bit-width. Also: `INT_MIN / -1`. For `<<` specifically: UB if the result `x * 2^n` is not representable in `width(x)` signed bits, OR if `x < 0`. RefractIR treats `<<` as signed integer arithmetic (aligning with `+`/`-`/`*`), not as a bit-vector shift — this keeps the overflow story consistent across all integer arithmetic operators.

5. **Overshift**: in `x << n`, `x >> n`, `x >>> n`, UB if `n < 0` or `n >= width(x)`. Separate from rule 4 — overshift is about the shift *amount*, not the shifted result.

### 7.2 `select` and strict UB (lazy)
For `select c, a, b`:
- Evaluate `c` first; UB in `c` makes the path infeasible.
- If `c` is true, evaluate only `a`; UB in `a` makes the path infeasible. `b` is not evaluated.
- If `c` is false, evaluate only `b`; UB in `b` makes the path infeasible. `a` is not evaluated.

Same for mask-based `select`: only lanes selected by the mask are evaluated per-lane.

### 7.3 Floating-point arithmetic semantics
See §2.9 for the full floating-point value model.

### 7.4 Floating-point UB

6. **FP overflow (±∞ result)**: any arithmetic operation (`+`, `-`, `*`, `/`) whose IEEE 754 result (under RNE) would be ±∞ is UB. This covers finite-operand overflow and division of any non-zero value by ±0.0.

7. **FP invalid operation (NaN result)**: any operation whose IEEE 754 result would be NaN is UB. This covers `±0.0 / ±0.0` and `x % ±0.0` for any `x`.

8. **Float-to-integer out-of-range**: `fN as iM` is UB if the float value, after truncation toward zero, is outside the representable range of `iM`.

### 7.5 Pointer UB

9. **Null pointer dereference**: `load <ptr>` or `store <ptr>, <val>` where `ptr` evaluates to `null` (address 0) is UB.

10. **Out-of-bounds pointer arithmetic**: every pointer carries a *provenance object* (the aggregate or scalar storage it was derived from — see rule 15 for how provenance is assigned). For `%p ± n`:
    - Let `base = addr(provenance)` and `size = sizeof(provenance)`.
    - UB if the resulting address falls outside `[base, base + size]`.
    - The "one-past-the-end" address `base + size` is a valid non-dereferenceable address (valid for arithmetic and equality; UB to `load`/`store`).

11. **Out-of-bounds load/store**: `load <ptr>` or `store <ptr>, <val>` where `ptr` points outside the bounds of the originating object (including the one-past-the-end address) is UB.

12. **Cross-object pointer arithmetic**: forming a pointer by arithmetic that crosses from one local variable's storage into another is UB. The memory regions of distinct local variables do not overlap.

13. **Uninitialized pointer dereference**: `load <ptr>` or `store <ptr>, <val>` where `ptr` itself is `undef` is UB (follows from rule 3 applied to the pointer value, kept as a separate rule for clarity).

14. **Cross-object pointer comparison**: comparing two pointers derived from different originating objects with `<`, `<=`, `>`, `>=` is UB. Equality (`==`, `!=`) between pointers of different objects is well-defined and always produces `false` / `true` respectively (distinct objects occupy disjoint address ranges per the non-overlap axioms).

15. **Aggregate-derived pointer provenance [revised in v0.2.1]**: every pointer derivation carries a *provenance object*:
    - **Top-level `addr %lv`** where `%lv : T` is a `let mut` local: provenance = `%lv` (the whole local).
    - **Field access — `addr lv.f` or `ptrfield <ptr>, f`**: provenance = the *immediate containing struct* of `f`.
    - **Index access — `addr lv[i]` or `ptrindex <ptr>, i`**: provenance = the *immediate containing array*.
    - **Pointer arithmetic — `<ptr> ± n`**: provenance is unchanged from `<ptr>`.

    Pointer arithmetic that walks outside the provenance object's storage range is UB (rule 10). One-past-the-end is valid for arithmetic and equality only.

15b. **Typed-access mismatch [v0.2.1]**: `load %p` or `store %p, v` through `%p : ptr T` is UB if the runtime address does **not** coincide with the start of a `T`-typed cell within the provenance object:
    - For an array `[N] U`: every offset `k * sizeof(U)` for `0 ≤ k < N` is a valid `U` cell.
    - For a struct `@S`: only the offsets of fields with declared type `T` are valid `T` cells. A `ptr i32` landing on the offset of an `i64` field is UB, even if arithmetic stayed within `@S`'s bounds.
    - For a top-level scalar local `%x : U`: offset `0` if `U == T`.

    Mid-cell, on a cell of a different type, or straddling cells: all UB. This rule does the real work for structs with mixed field types — rule 15 allows arithmetic within the struct, but the eventual deref must respect the field types.

16. **`ptrindex` out-of-bounds [v0.2.1]**: `ptrindex <ptr>, <index>` where `<ptr> : ptr [N] T` is UB if `index < 0` or `index > N`. Index `N` (one-past-the-end) produces a valid non-dereferenceable address. The provenance of the result is the array `<ptr>` points to (rule 15).

17. **Navigation through `null` [v0.2.1]**: `ptrindex <ptr>, <i>` or `ptrfield <ptr>, <f>` is UB if `<ptr>` evaluates to `null`. Declaring UB at the navigation site prunes the path immediately instead of waiting for a later `load`/`store`.

18. **Navigation through `undef` [v0.2.1]**: `ptrindex` and `ptrfield` count as reads of their pointer operand. A `<ptr>` operand whose value is `undef` is UB at the navigation site (consequence of rule 3, made explicit).

19. **Navigation from a one-past-the-end pointer [v0.2.1]**: `ptrindex <ptr>, <i>` or `ptrfield <ptr>, <f>` is UB if `<ptr>` is exactly the one-past-the-end address of its provenance object. That address is valid for arithmetic and equality (rule 10) but does not point to any element to navigate into.

27. **Dereferencing a pointer to a dead activation [New in v0.2.3]**: a `fun` activation's locals are provenance objects (rule 15) that exist only as long as the activation does. `load`, `store`, `ptrindex` and `ptrfield` through a pointer whose provenance object belonged to an activation that has returned are UB. Rules 10 and 11 are already stated against the provenance object; this rule says what happens when there is no longer one to state them against.

### 7.6 Vector UB [v0.2.1]

20. **Out-of-bounds vector lane access**: a lane read `lv[i]` or lane write `lv[i] = …` where `lv : <N> T` is UB if `i < 0` or `i >= N`.

21. **Lane-wise scalar UB**: all scalar UB rules (1–8) apply **per-lane** to vector operations. If any single lane would trigger UB in a vector operation, the entire path is infeasible. For example:
    - `%a / %b` where `%a, %b : <4> i32` is UB if any lane of `%b` is `0`.
    - `%a + %b` where `%a, %b : <4> i32` is UB if any lane overflows.

22. **Reading `undef` vector lane**: reading a lane whose value is `undef` (from a vector initialized with `undef`, or from a vector where the read lane has not yet been written by lane-write or whole-vector copy) is UB. This is the per-lane extension of rule 3.

### 7.7 Function call UB **[v0.2.2]**

23. **Contract precondition violation**: `call @f(...)` where `@f` is a contract-form `decl`, and any `pre` clause evaluates to `false` at the call site → path infeasible. Analogous to `assume` on the caller side: the caller must ensure preconditions hold on the chosen path, and if they don't, the path is pruned.

24. **Callee UB propagation**: UB encountered during execution of a `fun` callee makes the **caller's** path infeasible. UB is not a sandbox — it propagates across call boundaries. If any statement, condition, or nested `call` inside the callee triggers UB, the calling path is pruned.

25. **Intrinsic UB preconditions**: any intrinsic whose declared semantics requires a precondition treats violation as UB. This covers (i) results not representable in the declared return type (e.g., `@abs(INT_MIN)`, consistent with rule 4) and (ii) operand-domain restrictions (e.g., `@ctz`/`@clz` require non-zero input, `@ilog2` requires `x > 0`, `@div_euclid` requires non-zero divisor). Per-intrinsic preconditions are listed in §12 and consolidated in [`docs/undefined.md`](./undefined.md).

**Not UB (static checks).** The following are caught before execution and are **not** runtime UB:
- Call to an undeclared function (semantic error at check time).
- Argument-parameter count or type mismatch (type error at check time).
- Recursion cycle in the call graph (static error at check time, §9.7).
- Contract-form `decl` call in the interpreter (rejected before execution begins; the solver processes such calls via contract expansion, §9.6.2).
- Malformed `shuffle` index lists (type error at check time, §6.14) **[Planned, V2]**.
- `pure`/`const` attribute violations (semantic error at check time, §6.15) **[Planned, F1]**.

### 7.8 Vector memory UB **[New in v0.2.3 — Planned, V3/V4]**

Vector cells participate in the memory model as **leaf cells**: all pointer UB rules 9–19 apply unchanged with `<N> T` as the cell type (null/undef/one-past-the-end dereference, out-of-bounds arithmetic, cross-object rules, provenance through `ptrindex`/`ptrfield`).

26. **Vector cell indivisibility**: extending rule 15b, a `load`/`store` through `ptr <N> T` is UB unless the address coincides with the start of an `<N> T` cell within the provenance object, and a pointer of any *other* type (`ptr T`, `ptr <M> T` with `M != N`, …) landing anywhere inside an `<N> T` cell is UB to dereference. A vector cell is never accessed partially, per-lane, or under a different shape.

**Undef propagation (not UB).** Whole-vector `load` and `store` move lane values **including `undef`**, exactly like whole-vector copy `%v = %w;` (v0.2.1 §6.6): the move itself performs no lane reads. A vector cell never touched by a `store` is all-lanes-`undef`. UB arises only when an `undef` lane is subsequently *read* — used in arithmetic, `cmp`, a lane subscript, a reduction (§12.4), or a selected `shuffle` lane feeding a read (rule 22).


## 8. Division and modulo (round toward 0)

RefractIR uses **truncation toward zero** for both integer and floating-point `%`. The result sign matches the dividend sign.

### 8.1 Integer division and modulo
`Q = trunc(A / B)`, `R = A - Q*B`. `|R| < |B|`, sign of `R` matches `A`.

### 8.2 Floating-point modulo (`fmod` semantics)
C's `fmod`, **not** IEEE `remainder`. SMT encoding: `fp.sub(A, fp.mul(fp.roundToIntegral[RTZ](fp.div[RNE](A, B)), B))`.


## 9. Path-based symbolic execution and constraint extraction

### 9.1 Symbolic state
The executor maintains:
- `Store`: mapping from local lvalues to symbolic terms.
- `Mem[T]`: typed memory arrays, one per pointee type `T`.
- `PC`: feasibility constraints (conjunction).
- `REQ`: property constraints (conjunction).

### 9.2 Constraint sources (inherited)
- **Branches**: `br cond, ^t, ^f;` conjoins `cond` or `not(cond)` to `PC`.
- **Assumptions**: `assume c;` → `PC`.
- **Requirements**: `require c;` → `REQ`.
- **Strict UB**: any UB → `PC := false`.
- **`store`**: `Mem[T] := store(Mem[T], addr, val)`.

### 9.3 Solve goal
`DOM ∧ PC ∧ REQ` sent to the SMT solver.

### 9.4 SMT encoding of pointers (inherited)
Abstract address constants, non-overlap axioms, typed `Mem[T]` arrays. Full details in the v0.2.1 specification §9.4.

### 9.5 SMT encoding of vectors (inherited)
Per-lane symbolic terms, lane-wise SMT operations, `cmp` as per-lane boolean tuples. Full details in v0.2.1.

### 9.6 SMT encoding of function calls **[v0.2.2]**

#### 9.6.1 `call @f` where `@f` is a `fun` (defined body)

1. **Argument evaluation**: each argument is evaluated left-to-right in the caller's context. When multiple `call` atoms appear in the same expression (e.g., `call @f() + call @g()`), the leftmost call's side effects (on `Mem`, `PC`, `REQ`) commit before the next call's arguments are evaluated.
2. **Callee context**: a fresh context with parameters bound to argument values. **Symbol declarations are introduced once per program** — the first time a callee is symbolically entered, its `sym` declarations create symbolic constants that are then reused across all subsequent calls to the same `fun` on the path. (See "shared symbols" note below.) Locals get fresh storage on each call. `Mem[T]` and `PC`/`REQ` are carried forward.
3. **Callee execution**: blocks are symbolically executed along the sub-path through the callee. The caller's path `π` specifies which callee blocks are visited per call site.
4. **Return**: the return value is the symbolic term of the `ret` expression. `PC` and `REQ` reflect callee constraints. `Mem[T]` reflects callee `store`s. Callee-local `Store` is discarded. If callee `PC` becomes `false`, caller `PC` also becomes `false`.
5. **Caller `Store` coherence**: for any caller-side `let mut %x` whose address was passed (directly or transitively through `addr`, `ptrindex`, `ptrfield`) into the callee, the caller's cached `Store[%x]` is invalidated. Subsequent reads of `%x` re-fetch from `Mem[T]`. This keeps the local store consistent with possibly-mutated memory.
6. **Result**: the `call` atom evaluates to the return value.

**Shared symbols (v0.2.2).** A `sym` declared inside a `fun` body denotes a single solver-chosen value, even if the `fun` is called multiple times on the path. Rationale: fewer SMT variables → faster solving; semantically natural ("one hole, one value"). *Planned for a future version:* per-call-site fresh symbol instantiation, allowing the same `fun` to expose independent unknowns at each call site.

#### 9.6.2 `call @f` where `@f` is a contract-form `decl`

1. **Precondition check**: each `pre` clause is evaluated in the caller's context with parameters bound to arguments. Any `pre` evaluating to `false` → `PC := false` (UB per rule 23).
2. **Return value**: a **fresh symbolic constant** `ret_sym : RetType` is introduced for the return value.
3. **Postcondition assumption**: each `post` clause, with parameters bound to arguments and `ret` bound to `ret_sym`, is conjoined to `PC` as an assumption.
4. **Memory havoc**: all pointer parameters are potentially modifiable. For each pointer parameter `%p`, **every storage cell within `%p`'s provenance object is havoc'd** — replaced by fresh symbolic values in the appropriate `Mem[T_i]` arrays:
   - If `%p : ptr T` with provenance a scalar local `%x : T`: the single cell `Mem[T][addr(%x)]` is havoc'd.
   - If provenance is an array `[N] U`: all `N` cells `Mem[U][addr+k*sizeof(U)]` for `k ∈ [0, N)` are havoc'd.
   - If provenance is a struct `@S`: every field cell is havoc'd in its respective `Mem[FieldType]`.
   - If any cell in the provenance is itself a pointer (`ptr V`), it is havoc'd in `Mem[ptr V]`; transitively reachable storage **is not** further havoc'd — the contract is responsible for constraining nested-pointer behavior explicitly via `post` clauses.
   - The havoc'd values are constrained only by `post` clauses that reference them.

   > **[Partial in v0.2.3 — extended provenance forms are a non-goal, see §13.]**
   > The rule above is the target model. The shipped solver currently havocs only two
   > argument-expression forms — a direct `addr %x`, and a plain ptr local
   > `%p` whose provenance is known — and it does so by replacing the source
   > local's whole symbolic value with a fresh constant rather than havocing
   > per cell. Aggregate provenance (every cell of a `[N] U` / every field of
   > an `@S`), pointer arguments derived from `ptrindex` / `ptrfield` /
   > pointer arithmetic, and transitive nested-pointer cells are **not yet
   > havoc'd**. Callers passing those forms must constrain the post-state
   > explicitly via `post` clauses or caller-side asserts until the next
   > refinement lands. See the "Contract memory havoc — extended provenance
   > forms" bullet in §13.
5. **Caller `Store` coherence**: as in §9.6.1 step 5 — any caller-side `let mut` local whose address was reachable from an argument has its cached `Store` invalidated.
6. **Result**: the `call` atom evaluates to `ret_sym`.

#### 9.6.3 `call @f` where `@f` is an `intrinsic`

The solver applies the intrinsic's hard-coded SMT encoding. See §12 for per-intrinsic encodings.

#### 9.6.4 Path specification for interprocedural execution (non-goal for v0.2.3)

The user-chosen path `π` is extended to a **tree** of block visits:

```
@outer: ^entry -> ^b1 -> [call @inner: ^entry -> ^body -> ^exit] -> ^b2 -> ^exit
```

`[call @name: ...]` denotes the sub-path through the callee at that call site. Each call site on `π` may specify a different sub-path.

> **[Non-goal in v0.2.3 — surface syntax deferred, see §13.]** The
> `[call @name: ...]` tree notation above is **not yet a parseable surface
> form**, and `--path` accepts only the flat caller-block list. For each
> branchy callee the shipped solver instead **samples one random sub-path**
> per `solve()` invocation, seeded from `--seed`, bounding loops with a
> per-block visit cap (the cap raises an error if exceeded). Straight-line
> callees are therefore exact; branchy callees are not reproducible across
> the choice until the user-supplied sub-path syntax lands. See the
> "Callee sub-path syntax" bullet in §13.

### 9.7 No recursion **[v0.2.2]**

Recursion is **not supported** in v0.2.3:

- A `fun` body may not contain a `call` to itself (direct recursion).
- A `fun` may not participate in a cycle of calls — if `@a` calls `@b` and `@b` calls `@a`, that is mutual recursion and is forbidden.
- A contract-form `decl` may not contain a `call` to itself.

These conditions are detected statically by constructing the call graph from all `fun` and `decl` declarations visible in the program (including those resolved via `-I`). Any cycle in the call graph is reported as an error before execution begins.

This is a deliberate simplification: recursion introduces fixed-depth unrolling heuristics, complicates the SMT encoding with nested contexts, and provides limited value for the synthesis use cases v0.2.3 targets. Loops within a single `fun` (via CFG back-edges) remain the primary iteration mechanism.

### 9.8 SMT encoding of vector memory and shuffles **[New in v0.2.3 — Planned, V2/V3]**

**Vector memory (V3/V4).** Each vector shape gets its own typed region, encoded per-lane to reuse the existing per-lane vector machinery (§9.5):

- `Mem[<N> T]` is modeled as **`N` parallel arrays** `Mem[<N> T].lane_k : Array BV64 → sort(T)` for `k ∈ [0, N)`, all indexed by the same abstract cell address.
- `load p` produces the `N`-tuple `(select(Mem[<N> T].lane_0, p), …, select(Mem[<N> T].lane_{N-1}, p))` — one select per lane, mirroring how a register vector is already an `N`-tuple of lane terms.
- `store p, v` updates all `N` arrays at address `p` with `v`'s lane terms.
- Non-overlap axioms and provenance bookkeeping are unchanged from §9.4; the cell stride is `sizeof(<N> T)`. Rule 26 needs no dedicated encoding — typed regions are disjoint by construction, and mid-cell addresses are excluded by the same alignment side-conditions rule 15b already generates.
- `undef` lanes reuse the existing per-lane undef tracking; a `load`/`store` generates no read obligations (§7.8).

**Shuffle (V2).** `shuffle a, b, {i0, …, iK-1}` is pure lane-term rewiring: result lane `j` *is* the existing lane term `i_j` of `a ++ b`. No constraints, no fresh symbols — identical in cost to whole-vector copy (§9.5.7 of v0.2.1).


## 10. Examples

### 10.1 Simple function call

```text
fun @add_one(%x: i32) : i32 {
  let %one: i32 = 1;
^entry:
  ret %x + %one;
}

fun @use_add(%a: i32) : i32 {
  let mut %y: i32 = 0;
^entry:
  %y = call @add_one(%a);
  ret %y;
}
```

### 10.2 Link-form `decl` with `-I` resolution

File `lib.sir`:
```text
fun @sort3(%a: ptr i32, %b: ptr i32, %c: ptr i32) : i32 {
  ...
}
```

File `main.sir`:
```text
decl @sort3(%a: ptr i32, %b: ptr i32, %c: ptr i32) : i32;

fun @median(%x: ptr i32, %y: ptr i32, %z: ptr i32) : i32 {
  let mut %result: i32 = 0;
^entry:
  %result = call @sort3(%x, %y, %z);
  ret %result;
}
```

The interpreter resolves `decl @sort3` by scanning `-I` paths for a file containing `fun @sort3`. The solver inlines the body from the discovered file.

### 10.3 Contract-form `decl` (abstract model)

```text
decl @alloc(%size: i32) : ptr i32 {
  pre %size > 0, "size must be positive";
  post ret != null, "allocation never returns null";
};

fun @make_buffer(%n: i32) : ptr i32 {
  let mut %p: ptr i32 = null;
^entry:
  %p = call @alloc(%n);
  require %p != null, "buffer allocated";
  ret %p;
}
```

The solver checks `%size > 0` as a precondition and assumes `ret != null` after the call. The interpreter rejects this program — `@alloc` has no body.

### 10.4 Contract with caller-side pre/post comparison

```text
decl @increment(%p: ptr i32) : i32 {
  pre %p != null, "non-null pointer";
  post ret == load %p, "returns the new value";
};

fun @call_increment(%p: ptr i32) : i32 {
  let mut %before: i32 = 0;
  let mut %after: i32 = 0;
^entry:
  %before = load %p;
  %after = call @increment(%p);
  require %after == %before + 1, "incremented by 1";
  ret %after;
}
```

The caller saves `%before` to relate pre- and post-state. The contract's `post ret == load %p` constrains the return value in terms of the post-call memory. Since all pointer parameters are potentially modifiable, the solver knows `Mem[i32]` at `%p` may have changed.

### 10.5 Call chain mixing `fun`, `decl`, and `intrinsic`

```text
intrinsic @abs(%x: i32) : i32;

decl @validate(%data: ptr i32, %len: i32) : i1 {
  pre %len > 0, "non-empty buffer";
  pre %data != null, "non-null buffer";
  post ret == 0 || ret == -1, "boolean result";
};

fun @validated_abs(%data: ptr i32, %len: i32) : i32 {
  let mut %ok: i1 = 0;
  let mut %val: i32 = 0;
  let mut %result: i32 = 0;
  let %one: i1 = -1;
  let %zero: i32 = 0;
^entry:
  %ok = call @validate(%data, %len);
  br %ok == %one, ^do_abs, ^skip;

^do_abs:
  %val = load %data;
  %result = call @abs(%val);
  ret %result;

^skip:
  ret %zero;
}
```

On path `^entry -> ^do_abs -> (return)`: the solver expands `@validate`'s contract (`pre` checked, `ret ∈ {0,1}` assumed), the branch constrains `%ok == 1`, then `@abs` uses its built-in BV-theory encoding.

### 10.6 Synthesis with a contract

```text
decl @lookup(%key: i32) : i32 {
  pre %key >= 0, "non-negative key";
  post ret >= 0, "non-negative result";
};

fun @find_key(%target: i32) : i32 {
  sym %?k: value i32 in [0, 100];
  let mut %val: i32 = 0;
^entry:
  %val = call @lookup(%?k);
  require %val == %target, "found the target value";
  ret %?k;
}
```

The solver picks `%?k ∈ [0, 100]` and a symbolic return value `ret_sym` (constrained by `ret_sym >= 0`) such that `ret_sym == %target`. If `%target >= 0`, a solution exists.

### 10.7 Call graph across compilation units

With `-I lib/`, a link-form `decl` in `main.sir` resolves to `fun @helper` in `lib/util.sir`:

`lib/util.sir`:
```text
fun @helper(%x: i32) : i32 {
  let %two: i32 = 2;
^entry:
  ret %x * %two;
}
```

`main.sir`:
```text
decl @helper(%x: i32) : i32;

fun @main(%a: i32) : i32 {
  let mut %result: i32 = 0;
^entry:
  %result = call @helper(%a);
  ret %result;
}
```

The call graph is `@main → @helper` (acyclic). This is valid. If `lib/util.sir` contained `call @main`, the cycle `@main → @helper → @main` would be detected and reported as an error.

### 10.8 Addressable vectors and vectors in aggregates **[New in v0.2.3 — Planned, V3/V4]**

```text
struct @particle {
  pos: <4> f32;
  vel: <4> f32;
}

fun @advance(%p: ptr @particle) : f32 {
  let mut %pp: ptr <4> f32 = null;
  let mut %pv: ptr <4> f32 = null;
  let mut %pos: <4> f32 = 0.0;
  let mut %vel: <4> f32 = 0.0;
  let %dt: f32 = 0.5;
^entry:
  %pp = ptrfield %p, pos;            // ptr <4> f32 — navigation stops at the vector cell
  %pv = ptrfield %p, vel;
  %pos = load %pp;                   // whole-vector load from a struct field
  %vel = load %pv;
  %pos = %pos + %vel * %dt;          // lane-wise arithmetic (unchanged)
  store %pp, %pos;                   // whole-vector store back
  ret %pos[0];
}
```

`ptrfield %p, pos` has type `ptr <4> f32` (§6.13); the `load`/`store` move all four lanes at once. `addr %pos` inside the function would likewise yield `ptr <4> f32`.

### 10.9 Shuffle, reduction, and a `const` function **[New in v0.2.3 — Planned, V1/V2/F1]**

```text
intrinsic @reduce_add(%v: <4> i32) : i32;

fun @dot_reversed(%a: <4> i32, %b: <4> i32) : i32 const {
  let mut %r: <4> i32 = 0;
  let mut %prod: <4> i32 = 0;
^entry:
  %r = shuffle %b, %b, {3, 2, 1, 0};   // swizzle: reverse %b's lanes
  %prod = %a * %r;
  ret call @reduce_add(%prod);
}
```

The `const` attribute is verified statically: the body has no `load`/`store` and calls only the `const` intrinsic `@reduce_add` (§6.15). The reduction folds lanes 0→3 sequentially; any lane-sum overflow is UB (§12.4).


## 11. Toolchain updates

### 11.1 `-I` search path (all tools)

The `-I <path>` flag specifies a directory to search for `.sir` source files. Multiple `-I` flags may be specified; directories are searched in order. When a tool encounters a link-form `decl @name`, it scans each `-I` directory for `.sir` files. For each file found, it parses the top-level declarations only (function names + kinds). If exactly one `fun @name` is found, that body is used. If zero are found, the tool reports an error. If multiple are found, the tool reports an ambiguity error.

`-I` is orthogonal to the primary input file — the primary file is always fully processed, and `-I` provides additional resolution context.

### 11.2 `symirc` (compiler)

| Declaration | Lowering |
|---|---|
| `fun @f` | Compile body to target language |
| `decl @f` (link form) | Emit `extern` (C) / `import` (WASM), link at target level |
| `decl @f` (contract form) | Emit `extern` / `import` with contract as structured comment |
| `intrinsic @f` | Emit target-language built-in (§11.4) |
| `call @f` | Lower to direct function call in target language |

### 11.3 `symiri` (interpreter)

| Declaration | Behavior on `call @f` |
|---|---|
| `fun @f` | Interpret callee body in fresh context |
| `decl @f` (link form) | Resolve body via `-I`, interpret it. **Error** if no body found |
| `decl @f` (contract form) | **Error**: no body to execute. Interpreter rejects before starting |
| `intrinsic @f` | Execute built-in interpreter implementation |

### 11.4 `symirsolve` (solver)

| Declaration | Behavior on `call @f` |
|---|---|
| `fun @f` | Inline body; interprocedural symbolic execution (§9.6.1) |
| `decl @f` (link form) | Resolve body via `-I`, inline and symbolically execute |
| `decl @f` (contract form) | Expand contract (§9.6.2): preconditions → UB, postconditions → assumptions |
| `intrinsic @f` | Apply hard-coded SMT encoding (§12) |

### 11.5 Intrinsic lowering in `symirc`

Each intrinsic is declared once for `iN` (any `N ≥ 1`), not per concrete width. The backends use the same widening-and-mask strategy they already apply to all `iN` operations:

- Map `iN` to the smallest machine width `W` that fits: `W = 8` for `N ≤ 8`, `16` for `N ≤ 16`, `32` for `N ≤ 32`, `64` for `N ≤ 64`.
- Widen operands to `W` bits, perform the operation at width `W`, then truncate/mask the result to `N` bits.
- For `@clz`/`@ctz`: subtract `(W − N)` from the result so leading/trailing zeros in the widened operand are not counted.

| Intrinsic | C (generic over `iN`) | WASM (generic over `iN`) |
|---|---|---|
| `@abs` | Widen to `intW_t`. `x < 0 ? -x : x`. Mask to `N` bits. | `iW.abs`. Mask to `N` bits. |
| `@min` | Widen to `intW_t`. `x < y ? x : y`. Mask to `N` bits. | `iW.min_s`. Mask to `N` bits. |
| `@max` | Widen to `intW_t`. `x > y ? x : y`. Mask to `N` bits. | `iW.max_s`. Mask to `N` bits. |
| `@clz` | Widen to `uintW_t`. `__builtin_clz[ll](x) − (W−N)`. Mask to `N` bits. | `iW.clz − (W−N)`. Mask to `N` bits. |
| `@ctz` | Widen to `uintW_t`. `__builtin_ctz[ll](x) − (W−N)`. (Both `@clz` and `@ctz` require `%x != 0` per §12.2 — UB on zero, consistent with GCC builtins.) Mask to `N` bits. | `iW.ctz − (W−N)`. Mask to `N` bits. |
| `@popcount` | Widen to `uintW_t`. `__builtin_popcount[ll](x)`. Mask to `N` bits. | `iW.popcnt`. Mask to `N` bits. |

### 11.6 v0.2.3 target updates **[New in v0.2.3 — Shipped]**

- **Python target (T1)**: `symirc --target py` emits a Python module with semantics identical to the C and WASM targets. The lowering runs dominator-tree construction → reducibility check → loop forest → control tree → structured emission, reconstructing genuine `while`/`if` control flow; **irreducible CFGs are rejected** at compile time (see [`docs/reducibility.md`](./reducibility.md)). Symbols are provided by `func__symbol()` provider callables injected into the module globals by the embedding. The C backend offers the same structured reconstruction behind `--structured-lowering` (its default remains labels+`goto`, which accepts any CFG).
- **Structured WASM (T1b)**: `symirc --target wasm --structured-lowering` reconstructs genuine `block`/`loop`/`if` control flow in place of the default `$__pc`/`br_table` dispatch loop, and — like the C flag — implies `--require-reducible`. WASM's native multi-level `br N` consumes the **unlowered** control tree directly (no guard flags): each natural loop is a `(loop $__cont<h>)` continue target, each pending join a `(block $__jn<b>)`, and `Continue`/`Break`/`JumpJoin` become named `br`s. The dispatch loop remains the default and still accepts irreducible CFGs.
- **Vector lowering strategies (T2)**: `symirc --vec-lowering` selects the *storage form* of vector locals, per target: C accepts `vecext|scalars|array|structscalars|structarray`, WASM accepts `vecext|array|scalars` (`vecext` = native SIMD-128 `v128` registers, one per 16 bytes with wider shapes split), and Python accepts every C strategy except `vecext` (no native SIMD value type). Defaults: `vecext` (C, WASM) / `array` (Python). Every strategy is bit-exact per §2.11 semantics; the chosen strategy is stamped into the emitted module header. On WASM, vectors cross the call boundary the same way under every strategy: by address (caller-owned frame spill for arguments, hidden trailing sret parameter for returns). See [`docs/symirc.md`](./symirc.md) for the per-strategy details.
- **Checksum intrinsics on WASM (T3)**: `@crc32_update` / `@check_chksum` lower on WASM; `symirc --target wasm` no longer rejects programs containing them.

### 11.7 Lowering of planned v0.2.3 features **[New in v0.2.3 — Planned]**

| Feature | C | WASM | Python |
|---|---|---|---|
| `@reduce_*` (V1) | Unrolled sequential fold expression | Sequential lane extract + fold; order-insensitive members (`min`/`max`/`and`/`or`/`xor`) may use pairwise/hardware reductions (§12.4) | Sequential fold |
| `shuffle` (V2) | Lane-by-lane copy into a fresh temporary (reads complete before writes) | `vecext`: `i8x16.shuffle` when both operands are one `v128` (the mask is always static, §6.14), else lane extract/replace; other strategies: lane moves | Lane-list indexing |
| Vector `load`/`store` (V3/V4) | Copy the whole cell using the strategy's storage representation | `vecext`: `v128.load` / `v128.store` for ≤16-byte shapes, split for wider; other strategies: per-lane moves | Whole-cell lane-list copy |
| Attributes (F1) | `inline` → `static inline`; `noinline` → `__attribute__((noinline))`; `pure`/`const` → `__attribute__((pure))`/`__attribute__((const))` | Ignored | Ignored |

The interpreter (`symiri`) executes V1–V4 directly per §7.8/§12.4 semantics; attributes require no interpreter support beyond the static checks of §6.15.


## 12. Standard intrinsics **[v0.2.2]**

Intrinsics are built into the RefractIR toolchain. They require no body, no contract, and no `-I` resolution. In this section, `iN` denotes **any concrete integer type** (`i1`, `i8`, `i16`, `i32`, `i64`, or any other `i<Nat>`). Throughout §12.6 and onward, `fN` denotes **any concrete floating-point type** (`f32` or `f64`). The intrinsic is declared once per concrete width the program uses (e.g., `intrinsic @abs(%x: i32) : i32;` or `intrinsic @fabs(%x: f32) : f32;`), and the toolchain applies the same generic encoding regardless of `N`. Integer lowering follows the widening-and-mask strategy described in §11.5; floating-point lowering uses the native target precision directly (no widening, RNE-everywhere per §2.9).

### 12.1 Arithmetic intrinsics

#### `@abs`

```text
intrinsic @abs(%x: iN) : iN;
```

Returns the absolute value of `%x`. **UB if `%x == INT_MIN`** (the result `-INT_MIN` is not representable in `iN` — this is a signed integer overflow, consistent with rule 4 in §7.1). All intrinsics follow this general principle: any result not representable in the declared return type is UB.

**SMT encoding**: `ite(bvsge(x, (_ bv0 N)), x, bvneg(x))`, with a UB-precondition `x != INT_MIN` conjoined to `PC`.
**Interpreter**: assert `x != INT_MIN` (mark path infeasible if violated); otherwise `x < 0 ? -x : x`.

#### `@min`, `@max`

```text
intrinsic @min(%a: iN, %b: iN) : iN;
intrinsic @max(%a: iN, %b: iN) : iN;
```

Signed minimum / maximum.

**SMT encoding**: `ite(bvsle(a, b), a, b)` / `ite(bvsge(a, b), a, b)`
**Interpreter**: `a < b ? a : b` / `a > b ? a : b`

### 12.2 Bit-counting intrinsics

#### `@clz`, `@ctz`

```text
intrinsic @clz(%x: iN) : iN;
intrinsic @ctz(%x: iN) : iN;
```

Count leading / trailing zero bits. Following C/C++ semantics for `__builtin_clz`/`__builtin_ctz`, **`%x == 0` is UB** — the intrinsic implicitly requires `%x != 0`. Callers must ensure non-zero input on the chosen path.

**SMT encoding**: there is no native `bvclz` in all SMT solvers. For concrete `%x`, the tool computes the result directly and substitutes a constant. For symbolic `%x`, the tool introduces a fresh symbolic value constrained by:
- `result ∈ [0, N−1]` (since `x != 0`)
- `x != 0` is a UB-precondition that conjoins to `PC`
- For `clz`: `x[N-1 : N−result] == 0` and `x[N−result−1] == 1`
- For `ctz`: `x[result−1 : 0] == 0` and `x[result] == 1`

These constraints may be added as quantifier-free assertions where the solver supports bit-vector extraction, or the tool may bit-blast.

**Interpreter**: `__builtin_clz(x)` / `__builtin_ctz(x)` (or software fallback for widths not natively supported).

#### `@popcount`

```text
intrinsic @popcount(%x: iN) : iN;
```

Counts the number of 1 bits.

**SMT encoding**: for concrete `%x`, substitute the computed count. For symbolic `%x`, introduce a fresh symbolic value constrained by `result ∈ [0, N]` and `result == Σ bit_i(x)`. May use `bvadd` tree over bit extractions, or leave as an uninterpreted function with range bounds.

**Interpreter**: `__builtin_popcount(x)` (or software fallback for widths not natively supported).

### 12.3 Extensibility

Additional intrinsics may be added in future versions. Each new intrinsic must specify:
- Its declared signature (over `iN` or `fN`)
- Its SMT encoding (how the solver reasons about it)
- Its interpreter behavior (how `symiri` executes it)
- Its lowering pattern for each target (C, WASM), following the widening-and-mask strategy in §11.5 (for `iN`) or the native-precision strategy (for `fN`)

An intrinsic is accepted only if all four are defined. "Delegate to the target" is not acceptable — RefractIR owns the semantics.

**Priority taxonomy and rejection layers.** The six intrinsics defined above are the v0.2.2 baseline. The full design space — covering C `<math.h>`, WASM numeric instructions, and Rust `iN`/`fN` inherent methods — is classified into priority tiers **P0–P4** in [`docs/intrinsics.md`](./intrinsics.md). The tiering decides which intrinsics ship next, which are gated behind a feature flag, and which are deliberately rejected (at the frontend, by the solver, or by a specific backend). Solver and C support drive the priority; WASM is second-to-last; intrinsics with no efficient SMT encoding are last. v0.2.2 commits to shipping the **P0** tier.

**Shipped batches.** §12 ships in the following increments inside the
v0.2.2 line, each adding one solver-friendly group:

- Batch A — §12 *integer extras*: `@abs_diff`, `@signum`, `@clamp`,
  `@midpoint`.
- Batch B — §12 *bit-manipulation*: `@parity`, `@bswap`, `@bitreverse`,
  `@rotl`, `@rotr`, `@is_pow2`, `@ilog2`.
- Batch C — §12 *integer overflow-aware family (scalar-result subset)*:
  `@wrapping_{add,sub,mul,neg,shl,shr}`,
  `@saturating_{add,sub,mul,neg}`, `@div_euclid`, `@rem_euclid`.  See
  [`docs/intrinsics.md`](./intrinsics.md) §12.5 for the per-intrinsic
  signatures, SMT encodings, and UB conditions.  The tuple-returning
  members of this family (`@checked_*`, `@overflowing_*`) and the
  cross-width `@widening_mul` remain planned, gated on the multi-value
  return ABI.
- Batch D — §12 *floating-point basic IEEE family*: sign / bit ops
  (D.1), classification predicates (D.2), min / max (D.3),
  correctly-rounded math (D.4), and compositions (D.5) shipped (§12.6):
  `@fabs`, `@fneg`, `@copysign`, `@signbit`, `@to_bits`, `@from_bits`,
  `@is_normal`, `@is_subnormal`, `@fmin`, `@fmax`, `@sqrt`, `@floor`,
  `@ceil`, `@trunc`, `@fract`, `@recip`.  Shipping D.1 opens the
  type-restriction sentence at the top of §12 to `fN`.  This completes
  batch D.  The exponent-manipulation candidates (`@ldexp`, `@scalbn`,
  `@ilogb`, `@logb`) were **dropped** — symbolic `2^exp` scaling and
  exponent extraction have no clean QF_FP encoding, so they are not
  solver-friendly intrinsic targets.  See
  [`docs/intrinsics.md`](./intrinsics.md) §12.6 for the per-intrinsic
  signatures, SMT encodings, and UB conditions.
- Batch R — §12 *checksum primitives* (§12.7):
  `@crc32_update(state: i32, val: iN) : i32` (`N ∈ {8, 16, 24, 32, 40,
  48, 56, 64}`, byte-wise table-driven CRC32 update, reflected
  `0xEDB88320` polynomial, no initial / final XOR) and
  `@check_chksum(expected: i32, actual: i32) : i32` (returns `actual`
  on match, aborts in C / raises UB in symiri on mismatch). The C
  lowering carries function-local `static` tables and a
  `static __attribute__((noinline))` qualifier (new
  `CIntrinsic::linkageQualifier()` hook in
  `src/backend/intrinsics_c.cpp`) so the optimizer cannot fold the
  body or propagate the table contents into callers. See
  [`docs/intrinsics.md`](./intrinsics.md) §12.7 for the per-intrinsic
  specs. (Batch R shipped without a WASM lowering; that gap was closed
  by T3 in v0.2.3 — §11.6.)

### 12.4 Horizontal reduction intrinsics **[New in v0.2.3 — Shipped, V1]**

Reductions fold all lanes of a vector into one scalar. `T` ranges over the scalar element types; `<N> T` over the program's concrete vector shapes.

```text
intrinsic @reduce_add(%v: <N> T) : T;    // T ∈ iN, fN
intrinsic @reduce_min(%v: <N> T) : T;    // T ∈ iN, fN
intrinsic @reduce_max(%v: <N> T) : T;    // T ∈ iN, fN
intrinsic @reduce_and(%v: <N> T) : T;    // T ∈ iN only
intrinsic @reduce_or (%v: <N> T) : T;    // T ∈ iN only
intrinsic @reduce_xor(%v: <N> T) : T;    // T ∈ iN only
```

**Semantics**: the **sequential left-to-right fold over lanes `0 .. N-1`** is normative: `((v[0] ⊕ v[1]) ⊕ v[2]) ⊕ …`. A reduction reads **every** lane, so any `undef` lane is UB (rule 22).

**UB**: each intermediate step is an ordinary scalar operation and carries the ordinary scalar UB rules:
- `@reduce_add` over `iN`: any partial sum outside the `iN` signed range is UB (rule 4). Because UB depends on intermediate values, the fold order is semantically observable — no reassociation, in any component.
- `@reduce_add` over `fN`: each step rounds RNE; a ±∞ or NaN intermediate is UB (rules 6–7). FP addition is not associative, so the fold order is observable here too.
- `@reduce_min` / `@reduce_max` / bitwise reductions: no additional UB (finite-domain FP makes min/max total).

For floating-point `@reduce_min` / `@reduce_max`, equal-magnitude signed zeros are resolved with the IEEE 754 minNum / maxNum tie-break — identical to `@fmin` / `@fmax` (§12.6): a `-0.0` lane forces a `-0.0` minimum, a `+0.0` lane forces a `+0.0` maximum. This keeps the min/max result **independent of fold order** (a strict-comparison fold would make `±0.0` order-observable), which is what lets a backend use a pairwise or hardware reduction for these members. `@reduce_add` remains order-**dependent** (int overflow and FP non-associativity are observable), so its fold order is fixed left-to-right.

**SMT encoding**: the unrolled `N−1`-step fold over the vector's lane terms (`bvadd`/`fp.add`/`ite`-min-max/`bvand`/`bvor`/`bvxor`), with the per-step UB side-conditions of §7 conjoined to `PC`. Linear in `N`, no fresh symbols, no quantifiers.

**Interpreter**: direct sequential fold with per-step UB checks.

**Lowering**: see §11.7. Backends must reproduce the sequential fold bit-exactly; only the order-insensitive members (`min`/`max`/`and`/`or`/`xor`, whose results and UB behavior are order-independent) may use pairwise or hardware reduction instructions.

**Rejected member**: `@reduce_mul` is **not** provided — an `N−1`-deep chain of symbolic `bvmul`/`fp.mul` is solver-hostile (nonlinear), violating the SMT-friendliness goal. Write the product fold manually with lane subscripts if needed.


### 12.5 Observability beacon **[New in v0.2.3]**

```text
intrinsic @observe(%x: T) : T;      // T is any iN or fN
```

`@observe(v) = v` — the identity, at any integer or floating-point width. It is deterministic and oracle-consistent; the interpreter, solver, WASM, and Python all lower it to the identity. Its purpose is a **lowering** property rather than a value: the C backend emits an observable `volatile` write of `v`, a side effect the optimizer must preserve. This anchors a computation the compiler cannot prove dead — e.g. the body of a deliberately non-terminating loop — so the loop survives even under a forward-progress-assuming toolchain. Because that side effect is observable, `@observe` is treated as impure for the `const` restriction (§6.15); its identity *value* nonetheless keeps it deterministic and solver-friendly. Per-backend lowering: `docs/intrinsics.md` §12.7.


## 13. Non-goals for v0.2.3 (planned for later)

Every feature v0.2.2 §13 slated for v0.2.3 (WASM SIMD-128, addressable vectors, vectors in aggregates, shuffles, horizontal reductions, function attributes) is now **in scope** with its design in the body of this document — see the roadmap table in *What's new*. This section lists what v0.2.3 deliberately does **not** do.

**Dropped in v0.2.3** (considered and rejected, not merely postponed):

- **Relaxed SIMD**. The v0.2.2 *WASM SIMD support* bullet named both the SIMD-128 proposal (shipped, T2) and the Relaxed SIMD extension. Relaxed SIMD is dropped: its instructions have **implementation-defined results** (varying across engines and hardware), which is irreconcilable with RefractIR's bit-exact cross-target semantics, the interpreter/backend cross-validation methodology, and the strict UB model. A lowering whose output the spec cannot pin down bit-exactly has no place in the toolchain.
- **`noreturn` function attribute**. Appeared in the v0.2.2 attribute sketch alongside `pure`/`const`. Dropped because `call` is an expression that must produce a value — a callee that never returns has no coherent value semantics in the expression-call model. Dead ends are already expressible with the `unreachable` terminator (and, for contract `decl`s, a `pre 0 == 1`-style unsatisfiable precondition).
- **`@reduce_mul`**. See §12.4 — nonlinear symbolic multiplication chains are solver-hostile.

**Deferred to later versions:**

- **Defined lifetime model**. UB Rule 27 states locals' lifetime but we do not currently have a formal model. We may also refine our provenance model accordingly. For reference, this is [Nikita's proposal for LLVM](https://hackmd.io/@nikic/SJBt4mFCll). Since we do not allow casting between pointers and integers, ours should be simple.
- **Per-call-site fresh symbols**. A `sym` in a `fun` body denotes one solver-chosen value shared across all call sites on the path (§9.6.1). Per-call-site instantiation, exposing independent unknowns per call, is deferred.
- **Callee sub-path syntax** — planned. §9.6.4 promotes the user-chosen path `π` to a tree of block visits, but the surface syntax to specify each callee's sub-path (e.g., a nested `[call @inner: ^entry -> ^body -> ^exit]` form, a per-callee `--call-path @inner=^a,^b` CLI flag, or a JSON object) is deferred. The current solver picks one random path per callee per `solve()` invocation, seeded from `--seed`, with a per-block visit cap to bound loops. A future version will replace the random choice with an exact user-supplied sub-path so synthesis results are fully reproducible across branchy callees.
- **`@is_bitexact` — a value-identity predicate** — planned. `@is_bitexact(%a: T, %b: T) : i1`, true iff `a` and `b` are the *same* value. For integers it coincides with `==`; for floats it is the stricter relation of §2.9, differing from `==` only on `±0.0`. Today an analysis wanting value identity must open-code it (`@to_bits` equality at the source level, SMT `=` rather than `fp.eq` in the solver, a bit comparison in the interpreter), which is easy to get wrong in exactly one case and only in one direction. A first-class predicate would make the choice between the two relations explicit at the point of use. Lowers to `fp.eq` plus a signbit agreement check, or equivalently to `@to_bits` equality — both cheap and solver-friendly.
- **Char and string types** — planned. Add first-class support for `char` and string literals.
- **Aggregate symbols** — planned. `sym` of array / struct type is currently rejected (§3.4). Supporting them needs consumer syntax (element access on symbols or whole-aggregate-copy initialization), per-leaf solver encoding, and driver conventions in every backend; the interpreter's `--sym` binding format would extend to brace-initializer literals (e.g. `--sym '%?a={1,2,3}'`).
- **Recursion**. A `fun` body may not call itself (direct recursion) or participate in a mutual recursion cycle. The call graph must be a DAG. Loops within a single `fun` via CFG back-edges remain the primary iteration mechanism. Recursion introduces fixed-depth unrolling heuristics and complicates the SMT encoding with nested contexts — it provides limited value for the synthesis use cases v0.2.3 targets.
- **Indirect calls / function pointers**. `call` always targets a statically-named `GlobalId`. Function pointer types (`ptr fun(...)`) and indirect calls through computed addresses are not supported. All call targets are resolved by name at parse time.
- **`old()` in contracts**. Post-state only. Pre-state references require caller-side temporaries. Adding `old()` introduces a two-state logic into the SMT encoding, which is not yet justified.
- **Contracts on `fun` bodies**. A `fun` never has a contract — the body is the ground truth. Modular verification (prove the body satisfies a contract, then callers use the contract instead of inlining) is deferred to a future version.
- **Mutable pointee annotations (`mut ptr T`)**. All pointer parameters are modifiable — the callee may write through any pointer. Per-parameter mutability annotations with static enforcement (i.e., `store` through a non-`mut` parameter is a compile-time error) is deferred. The uniform "all modifiable" model is simpler and sufficient for v0.2.3 synthesis patterns.
- **Additional intrinsics — P0 tier (shipped in v0.2.2); a few members still deferred.** The P0 tier shipped in full across batches A–D: integer extras (`@abs_diff`, `@signum`, `@clamp`, `@midpoint` — `@umin`/`@umax` were dropped, RefractIR has no unsigned integer types), bit-manipulation (`@parity`, `@bswap`, `@bitreverse`, `@rotl`, `@rotr`, `@is_pow2`, `@ilog2`), the integer overflow-aware family scalar-result subset (`@wrapping_{add,sub,mul,neg,shl,shr}`, `@saturating_{add,sub,mul,neg}`, `@div_euclid`, `@rem_euclid`), and the floating-point basic IEEE family (`@fabs`, `@fneg`, `@copysign`, `@signbit`, `@to_bits`, `@from_bits`, `@is_normal`, `@is_subnormal`, `@fmin`, `@fmax`, `@sqrt`, `@floor`, `@ceil`, `@trunc`, `@fract`, `@recip`). Shipping batch D.1 opened the §12 type-restriction sentence to `fN`. **Still deferred:** the tuple-returning overflow members (`@checked_*`, `@overflowing_*`) and the cross-width `@widening_mul`, which wait on the multi-value return ABI. The exponent-manipulation candidates (`@ldexp`, `@scalbn`, `@ilogb`, `@logb`) were **dropped** — symbolic `2^exp` scaling and exponent extraction have no clean QF_FP encoding, so they are not solver-friendly intrinsic targets. See [`docs/intrinsics.md`](./intrinsics.md) for the per-batch scope.
- **Intrinsics in tiers P1–P4 — deferred to later versions.** Composed-WASM lowerings (P1: `@ffs`, `@next_pow2`, `@fmod`, `@remainder`, `@fdim`, `@modf`, `@frexp`, `@nextafter`, `@fpclassify`, `@total_cmp`, …); bounded SMT encodings behind a feature flag (P2: `@isqrt`, `@pow` with symbolic exponent, `@ilog10`, `@hypot`); libm-backed transcendentals where the solver prunes the path (P3: `@exp`/`@log`/`@pow`/`@sin`/`@cos`/`@tan`/`@erf`/`@tgamma`/… and friends); and stateful/impure or non-finite-producing intrinsics rejected at the frontend (P4: `@rand`, `@time`, `@nan`, `@inf`, I/O). See [`docs/intrinsics.md`](./intrinsics.md) for the full classification, the rejection layers, and the consistency rule between the interpreter and host libm.
- **Heap allocation and memory intrinsics** (`@memcpy`, `@memset`). Currently classified P4 in [`docs/intrinsics.md`](./intrinsics.md): their SMT encoding requires byte-level array reasoning with potentially symbolic sizes. Eligible for re-promotion when the solver gains byte-addressable memory. Track the alloc-free path, insert some free-after-use/double-free/... patterns in inaccessible paths; maybe there're benefits.
- **Contract memory havoc — extended provenance forms.** The solver now havocs the storage backing each pointer parameter at contract-form `decl` call sites (§9.6.2 step 4) so post-state pointee constraints are sound. The current implementation resolves two argument-expression forms — direct `addr %x` and a plain ptr local `%p` with a known provenance — by replacing the source local's symbolic value with a fresh constant. Aggregate provenance (havocing every cell of a `[N] T` or every field of an `@S`), pointer arguments derived from `ptrindex`/`ptrfield`/pointer arithmetic, and transitive nested-pointer cells are not yet havoc'd; callers passing those forms today should constrain the post-state via additional contract clauses or caller-side asserts until the next refinement lands.
- **Multiple return values**. Functions return exactly one typed value. Multi-result patterns can be expressed through pointer out-parameters.
- **`sym` of pointer type**. Pointer symbols require a richer address domain theory.
- **Heap allocation** (`malloc`/`free` or arena allocation). Pointers are stack-only.
- **Pointer/integer casts** (`ptr T as iN`, `iN as ptr T`).
- **Aliasing between distinct locals** (deliberately UB; explicit alias modeling deferred).
- **Parentheses and general expression trees**.
