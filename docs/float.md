# RefractIR floating-point

This document collects every floating-point commitment RefractIR makes, across the spec, the interpreter, the backends and the solver. Each section names the spec reference it derives from. Its siblings are [undefined.md](./undefined.md), which states the UB rules per tool, and [intrinsics.md](./intrinsics.md), which owns the intrinsic taxonomy.

One principle governs the design:

> Every FP operation produces the same bit pattern on every backend.

That is the only property worth defending, and everything below exists to keep it mechanical and checkable: the finite-only domain, RNE everywhere, `fmod` semantics, and the canonical serialization invariant. The matching rule for intrinsics, that the interpreter and every backend agree on every value, lives with the intrinsic taxonomy in [intrinsics.md](./intrinsics.md).

## 1. Value model (spec §2.9)

The types are `f32` (IEEE 754 binary32) and `f64` (IEEE 754 binary64). There is no `f16`, `f128`, `bfloat16`, x87 80-bit extended, or decimal float.

The domain is finite only. The valid RefractIR floating-point values are exactly the finite IEEE 754 values; `±∞` and NaN are not values of the language, and any operation whose IEEE result would be one of them is UB (spec §7.4 rules 6 and 7). A program that depends on infinity or NaN propagation is outside the language.

`+0.0` and `-0.0` are distinct bit patterns and both valid. They compare equal under `==`, and `@signbit` and `@to_bits` tell them apart. The canonical serializer preserves the sign bit across every text boundary.

Subnormal values are first-class. RefractIR does not flush to zero, and no backend may enable FTZ or DAZ. `parseFloatLiteral` accepts subnormals, rejecting only true overflow to `±HUGE_VAL`.

Every operation rounds to nearest, ties to even (IEEE `roundNearestTiesToEven`, SMT `RNE`). There is no alternate rounding mode, no `fenv` access, and no rounding-mode parameter on any intrinsic.

| RefractIR type | SMT sort |
|---|---|
| `f32` | `(_ FloatingPoint 8 24)` |
| `f64` | `(_ FloatingPoint 11 53)` |

## 2. Operations and rounding

Scalar operators are typed homogeneously: every atom in a `+` or `-` chain shares one FP type, and so do `*`, `/`, `%` and the atom-level coefficients. Mixing widths, or integers with floats, requires an explicit `as` cast (spec §6.7).

| Op | Semantics | Rounding | UB rule |
|---|---|---|---|
| `+` | IEEE add | RNE | §7.4-6,7 |
| `-` | IEEE sub | RNE | §7.4-6,7 |
| `*` | IEEE mul | RNE | §7.4-6,7 |
| `/` | IEEE div | RNE | §7.4-6,7 |
| `%` | C `fmod`, truncated-quotient remainder | RNE | §7.4-6,7 |

Each result must be finite, or the path is UB. The interpreter, the C backend and the Python backend check finiteness after every FP arithmetic operation, and the solver conjoins it to the path condition. Among the operators, the WASM backend guards only the `%` intermediate (§11.3); a non-finite result of a plain `+`, `-`, `*` or `/` is not trapped there, and a UB path silently computes with it ([symirc.md](./symirc.md#refinement-and-undefined-behaviour)). FP intrinsics are guarded on every backend, WASM included ([intrinsics.md](./intrinsics.md)).

Those five are the only built-in FP operators. Everything else, `sqrt`, `fabs`, `floor`, `fma` and the rest, lives in the intrinsic layer that [intrinsics.md](./intrinsics.md) §12.6 specifies.

### 2.1 Comparison

`cmp <relop>` lifts to FP transparently, lane-wise on vectors and scalar otherwise, giving `i1` or `<N> i1`.

| relop | Meaning |
|---|---|
| `==`, `!=` | IEEE numeric equality; `+0.0 == -0.0` is true |
| `<`, `<=` | IEEE ordered less-than, less-equal |
| `>`, `>=` | IEEE ordered greater-than, greater-equal |

NaN is UB, so the unordered cases of the IEEE relops never arise: both operands are finite by the time `cmp` runs, and every relop is total.

RefractIR keeps two notions of float equality apart, and conflating them is a real bug rather than a pedantic distinction:

| notion | applies to | `+0.0` vs `-0.0` | SMT |
|---|---|---|---|
| IEEE comparison | the `==` and `!=` operators, `cmp` | equal | `fp.eq` |
| Value identity | state recurrence, model equality, any "is this the same value" question | distinct | `=` |

The operator answers a numeric question and follows IEEE. Value identity asks whether two floats *are* the same RefractIR value, and since `+0.0` and `-0.0` are distinct values that `@signbit` and `@to_bits` tell apart, identity keeps them apart too.

Only the comparison form is in the language. Value identity has no operator, so an analysis that needs it reaches for it directly: `@to_bits` equality at the source level, SMT `=` rather than `fp.eq` in the solver, a raw bit comparison in the interpreter. Every whole-state question, whether a state recurred or whether two models agree, is an identity question, and answering it with `==` silently merges the two zeros. Spec §13 lists a dedicated `@is_bitexact` predicate as a candidate for a future revision.

### 2.2 select

`select cond, a, b` evaluates only the arm it takes, so UB in the other arm does not fire (spec §7.2). `select (y != 0.0), x/y, 0.0` is therefore a legal way to dodge divide-by-zero UB on the `y == 0` path.

## 3. The `%` operator is `fmod`, not `remainder`

RefractIR's `%` is C `fmod`, the truncated-quotient remainder, not IEEE 754 `remainder` (`fp.rem`):

> `x % y = x - trunc(x / y) * y`

with `trunc` rounding toward zero. The sign of the result follows the sign of `x`, which matches integer `%` (spec §2.5).

The two differ visibly in two places. `remainder` returns the result of smallest absolute value, which can be negative when `x > 0`, while `fmod` preserves `sign(x)`. And when `x/y` is exactly a half-integer, `remainder` rounds the quotient to even while `fmod` rounds it toward zero.

`x % 0.0`, for either signed zero, has NaN as its IEEE result and is UB under §7.4 rule 7. Overflow of the result itself cannot happen mathematically, since the true remainder has magnitude below `|y|`; it can only arrive spuriously through the SMT and WASM lowerings, which §11 covers.

An intrinsic needing IEEE `remainder` semantics would land as `@remainder`, distinct from an `@fmod` intrinsic that would duplicate `%` at the source level. Both sit at tier P1 in [intrinsics.md](./intrinsics.md).

## 4. Casts (`as`)

All four directions are well-typed under spec §6.4. Each cast rounds once under RNE, except the f32 to f64 widening, which is exact.

| Cast | Semantics | UB |
|---|---|---|
| `iN as fM` | Signed integer to FP under RNE | Never: with `N <= 64` (§11 lowering), every `iN` magnitude is far below `f32`'s finite maximum |
| `fN as iM` | Truncate toward zero, then check the fit | Out of range after truncation (§7.4-8) |
| `f32 as f64` | Exact widening: every f32 is an f64 | Never |
| `f64 as f32` | Round to f32 under RNE | Rounded result is `±∞` |

Vector casts apply per lane. There are no pointer-to-float casts (spec §13 non-goals).

## 5. UB rules

The three FP UB rules from spec §7.4 also appear in [undefined.md](./undefined.md#scalar-arithmetic-71-74), which is the per-tool enforcement reference:

* Rule 6, FP overflow: any `+`, `-`, `*`, `/` whose RNE result would be `±∞`.
* Rule 7, FP invalid: any operation whose result would be NaN, which covers `0/0` and `x % 0`.
* Rule 8, float-to-int out of range: `fN as iM` where the truncated mathematical value falls outside `iM`.

Rule 21 applies the per-lane analogues to vectors.

## 6. Vector FP (spec §2.11, §6.9)

`<N> f32` and `<N> f64` are first-class value types for `N >= 2`. Every scalar operation lifts to its lane-wise counterpart, with identical rounding, UB rules and `cmp` semantics applied per lane, so `cmp <relop> v, w` yields `<N> i1`.

The horizontal reductions `@reduce_add`, `@reduce_min` and `@reduce_max` fold an FP vector to a scalar; [intrinsics.md](./intrinsics.md) §12.8 specifies the fold order and the min/max tie-break, both of which matter for bit-exactness.

Vectors are not addressable: there is no `ptr <N> T` and no `addr` on a vector local (spec §13).

## 7. Literals and inference

A float literal is decimal, as in `1.5`, `-0.2`, `1e-5`, `3.14E+2`. There are no hex-float source literals, and no `inf`, `nan`, `INFINITY` or `NAN` literal forms, which follows directly from the finite-only domain: a program cannot construct a non-finite value.

A bare float literal is `f32` unless the surrounding context forces `f64`, alongside the integer default of `i32`. Every float literal parses through `refractir::parseFloatLiteral`, never `std::stod` (§9).

`undef` of FP type is allowed at the type level, as any leaf may be `undef`, and reading it is UB under §7.1 rule 3.

## 8. Why `fmod` and not `remainder`

Integer `%` is also truncated-quotient (`bvsrem`, as in C), and float `%` was chosen to match it, so one mental model covers both:

> `x % y` is the remainder when `x / y` is truncated toward zero.

The alternative introduces an asymmetry that trips up anyone writing `% 0.5` for a fractional-part trick, where `remainder` returns a negative value for positive `x`.

## 9. Canonical serialization

RefractIR carries `f32` and `f64` values bit-exactly across every text boundary: `.sir` source, descriptor JSON, the `SOLVED` / `PARAMS` / `RETURN` headers, model-dump files, and CLI positional arguments. Two entry points in [include/ast/ast.hpp](../include/ast/ast.hpp) own this, and every producer and consumer goes through them.

`refractir::formatDouble(double)` emits the shortest decimal string that round-trips via `std::to_chars(…, std::chars_format::shortest)`, appending `.0` when neither a `.` nor an exponent appears, so an integer-valued literal still tokenises as a float and signed zero survives.

`refractir::parseFloatLiteral(std::string)` wraps `std::strtod`. It accepts subnormals and raises only on true overflow to `±HUGE_VAL`. Never use `std::stod`: libstdc++ throws `out_of_range` on any `ERANGE`, valid subnormals included, so a perfectly representable denormal would abort the interpreter.

Two sites diverge on purpose, because they emit a different grammar. [src/backend/c_backend.cpp](../src/backend/c_backend.cpp) emits C floats with the `f` suffix, and [src/backend/wasm_backend.cpp](../src/backend/wasm_backend.cpp) emits WAT floats with `f32.const` / `f64.const` syntax. Each carries its own bit-exact formatter and a comment pointing back to `formatDouble`.

Before writing `std::stod`, `std::to_string(double)`, an `std::ostringstream` with `precision(17)`, `printf("%.17g", …)` or `printf("%f", …)` anywhere in RefractIR, use the canonical pair instead.

The invariant covers the SMT boundary too, because handing a float to a solver is a value crossing and a decimal rendering loses at both ends. `std::to_string` formats with `%f` and keeps six fraction digits, so `1e-7` becomes `"0.000000"`, flushing a positive value to zero. And a decimal string parses through a real, which has no signed zero, so `-0.0` comes back `+0.0`. Build FP constants from the `double` itself through `ISolver::make_fp_value_from_real`, which reaches `Z3_mk_fpa_numeral_double` on the Alive backend and Bitwuzla's sign, exponent and significand constructor. The decimal-string `make_fp_value` is not for a value that started life as a `double`.

The interpreter's `Result:` line is the one channel in hex-float form, via `printf("%a", …)`. Hex float is parseable by `strtod` and bit-exact by construction, which is what lets the cross-validation harness compare it against the C side's `printf("Result: %a\n", …)` byte for byte. It is a serialization channel only; hex float is not a source-level literal.

## 10. SMT encoding (spec §9, §2.9)

All FP reasoning happens in QF_FP, which CVC5, Z3 and Bitwuzla all support, and RefractIR requires no solver-specific extension. The sorts are `(_ FloatingPoint 8 24)` and `(_ FloatingPoint 11 53)`.

A single `roundNearestTiesToEven` constant is created once per solver and threaded through every FP operation.

After every FP `+`, `-`, `*` and `/`, the encoder conjoins `(not (fp.isInfinite t))` and `(not (fp.isNaN t))` to the path condition; `assertFPFinite` in [src/solver/internal.hpp](../src/solver/internal.hpp) centralizes it.

A `sym %?x: value f32` encodes as a fresh FP constant of the matching sort. Vector FP symbols are allowed, their lanes independent.

`%` uses IEEE `fp.rem` as a base and corrects a nonzero remainder whose sign differs from the dividend by adding the dividend-signed `|y|`. This maps the nearest-quotient result to C `fmod` semantics while preserving signed zero. Casts encode through `fp.to_fp` and `fp.to_sbv`, with RNE on the conversion and an explicit BV range check for `fN as iM`.

## 11. Lowering and the cross-backend contract

Bit-exactness partitions into one responsibility per tool.

### 11.1 Interpreter

The interpreter sets `std::fesetround(FE_TONEAREST)` at startup and, on x86, clears MXCSR FTZ and DAZ, in case a parent process or an upstream library left them set.

It stores f32 values as `double` and, after every f32 operation, narrows via `static_cast<double>(static_cast<float>(v))` and runs the finiteness checks in `checkFPResult`. That narrow-then-check pattern is bit-exact with single-rounded f32 arithmetic: computing in precision `q` and then rounding to precision `p` matches single-rounded precision-`p` arithmetic for every RNE-rounded binary operation among `+ - * / sqrt` when `q >= 2p + 1` (Boldo and Melquiond, *When Double Rounding is Odd*, 2005), and RefractIR has `p = 24` and `q = 53`.

Parameter binding rounds an f32 argument to f32 precision before the body runs, and a stored f32 local truncates its held `double` to 4-byte storage, so a re-read agrees with the C backend.

`%` calls `std::fmod` or `std::fmodf`, preceded by an explicit finiteness check on the intermediate `x/y`, which enforces the §2.9 operand-precision overflow rule consistently with the WASM and solver paths. A cast to integer bounds-checks against `[-2^(bits-1), 2^(bits-1))` before the `static_cast<int64_t>`.

### 11.2 C backend

The C backend emits `float` and `double`, and float literals through its own bit-exact formatter.

`%` emits `fmodf` or `fmod` wrapped in a GCC statement expression that pre-evaluates `x/y` and traps on a non-finite intermediate, matching the interpreter and WASM. A float vector `%` is forced to lane-unroll through the lane-wise fmod helper, since GCC vector extensions do not define `%` for floats and the inline path would emit invalid C.

Explicit `isinf` and `isnan` checks follow every FP arithmetic operation, because `-fsanitize=float-divide-by-zero` does not catch finite overflow to infinity by itself.

FP behaviour is pinned at the source level rather than through harness compile flags. Every emitted compilation unit carries three guards via `common.h`:

```c
#if !defined(__STDC_IEC_559__) || __STDC_IEC_559__ != 1
# error "RefractIR-lowered C requires an IEC 60559 / IEEE 754 conforming implementation"
#endif
#if !defined(FLT_EVAL_METHOD) || FLT_EVAL_METHOD != 0
# error "RefractIR-lowered C requires an implementation with FLT_EVAL_METHOD == 0"
#endif
#pragma STDC FP_CONTRACT OFF
```

The C standard requires the pragma to override any `-ffp-contract` flag the caller passes, and `FLT_EVAL_METHOD == 0` rules out x87 and `long double` excess-precision evaluation, so no harness has to pass an FP flag.

Cross-validation compiles with `-fsanitize=undefined -fno-sanitize-recover=all -lm`. `-Ofast` and `-ffast-math` are forbidden, since they bypass the source-level pragma.

### 11.3 WASM backend

The WASM backend emits `f32.const` and `f64.const` through its own bit-exact formatter, and `f32.{add,sub,mul,div}` / `f64.{add,sub,mul,div}` directly. f32 and f64 conversion uses `f64.promote_f32` and `f32.demote_f64`.

WASM MVP has no `fN.rem`, so `%` is composed inline as the §2.9 encoding `x - fN.trunc(x / y) * y`, with an explicit finiteness trap on the intermediate `x/y`: the quotient is spilled to a scratch local and a non-finite value hits `unreachable`, enforcing §7.4 rule 6 on the encoding's inner division exactly as the interpreter and the C backend do. `--no-ub-guards` elides the whole stack-neutral check. This is the one FP trap the WASM backend emits for the operators; plain `+`, `-`, `*` and `/` results carry no finiteness guard (§2).

### 11.4 Cross-validation

The cross-validation harness solves a path, reifies the model into a concrete `.sir`, runs the interpreter, compiles the C output, runs that, and diffs the `Result:` line byte for byte. Both sides print through `%a`, so every bit is observable.

It compiles with `gcc … -fsanitize=undefined -fno-sanitize-recover=all -lm` and no explicit optimization level. It deliberately passes no `-ffp-contract` or `-fexcess-precision` flag, because the C backend pins both concerns at the source level (§11.2).

## 12. What is intentionally excluded

These are not gaps. Each one simplifies the cross-backend contract.

| Excluded | Why |
|---|---|
| `±∞` and NaN as values | One domain, one source of truth. No quiet/signaling distinction, no payloads, no NaN comparison corner cases. |
| Alternate rounding modes (`RTZ`, `RU`, `RD`, `RMM`) | RNE is universal across SSE2, AArch64, WASM and SMT. Anything else needs fenv plumbing or a per-operation rounding parameter. |
| `fenv` access | Process-level mutable state, impossible to reproduce in WASM, and it breaks SMT semantics. |
| Decimal floats, `f16`, `f128`, bfloat16, x87 80-bit | None is universally available across the backends. |
| Hex-float source literals (`0x1.8p+3`) | Decimal literals plus the canonical serializer already round-trip every value; hex float stays a serialization form. |
| Subnormal flush-to-zero | Subnormals are first-class, so no backend may enable FTZ or DAZ. |
| FMA contraction | A `*` followed by a `+` is two operations with two finiteness checks. Contracting them silently changes the rounding, which is why the C backend emits `#pragma STDC FP_CONTRACT OFF`. |
| Pointer to float casts | Spec §13 non-goal. |
