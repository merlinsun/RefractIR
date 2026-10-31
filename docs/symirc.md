# symirc

`symirc` translates a RefractIR (`.sir`) program into C, WebAssembly text, Python, or a Triton JIT module for the bounded tensor profile. It solves nothing: a concrete program lowers to concrete target code, and a symbolic program lowers with each symbol left as an external hook the embedding provides. The scalar targets are meant to agree on every program that is free of undefined behaviour, and cross-validation against `symiri` is what holds them to it.

## Usage

```bash
symirc <input.sir> --target <c|wasm|python|triton> [-o output]
```

The full option list is `symirc --help`, declared in [src/symirc.cpp](../src/symirc.cpp). The target defaults to `c`. C, WebAssembly, and Python output defaults to stdout; Triton output defaults to the input path with a `.py` extension.

```bash
symirc prog.sir --target c      -o prog.c
symirc prog.sir --target wasm   -o prog.wat
symirc prog.sir --target python -o prog.py
```

## Triton tensor target

`--target triton` emits a JIT module under the profile selected by `--tensor-profile-version v1|v2|v3|v4` (default `v1`). The selected entry defaults to `@kernel`; `--tensor-softmax-lowering direct|decomposed` selects one of the two supported softmax implementations. `--tensor-loop-bound runtime` passes each counted loop's bound as a kernel parameter (`<induction>_bound`, after `out_ptr`) whose default is the SIR literal and which is excluded from Triton's specialization, so the compiler sees an unknown trip count, even for one trip, while an unchanged launch computes the literal program; `literal` (the default) emits the bound as a constant. `--tensor-loop-hint num_stages=N|loop_unroll_factor=N|flatten` (N from 1 to 8) adds that keyword to every counted loop's `tl.range`; it changes how Triton schedules the loop, not what the loop computes.

```bash
symirc tensor.sir --target triton --entry kernel -o tensor_kernel.py
symirc tensor.sir --target triton --entry kernel --tensor-softmax-lowering decomposed -o tensor_kernel.py
```

The target enforces the selected bounded profile and emits one program instance. V3 masks padded lanes according to their logical extents. Buffers are disjoint and ordered by the entry signature. Contiguous row-major storage is the default; `--tensor-layout-manifest case.json` supplies per-buffer positive strides and offsets from a validated case manifest. The [tensor runner](../test/lib/run_triton_tests.py) supplies the launch adapter and independent reference check. C, WebAssembly, and Python emission reject tensor modules before creating or replacing an output file. [Tensor capability analysis](../src/analysis/tensor.cpp) enforces the profile, and [the Triton backend](../src/backend/triton_backend.cpp) defines the operator mappings.

### Tensor window

`tensor window %source, %index, <fill>, [rows, stride, active]` reads a tile of `rows` rows starting at `index * stride` from a tensor parameter and produces `[rows, d1 … dn]` with the source's element type. The emitter computes the base in 64-bit arithmetic and masks each lane with the conjunction of the result's shape validity and `base < active`, so an inactive logical position and an execution-tile pad both read the literal `fill` through `other=`. The fill is a defined value that participates in every consumer. Windows are admitted under the v2 and v3 profiles only; a negative index is undefined behaviour.

### Tensor dot

`tensor dot %a, %b` multiplies a `[M, K]` tensor by a `[K, N]` tensor of the same element type and produces `[M, N]`. An `i32` result is the exact inner product; every operand lane must lie in the int8 range, and a lane outside it is undefined behaviour, so the emitter can pass `int8` operands to `tl.dot` with an `int32` accumulator. An `f32` result is the exact inner product rounded once, like `sum`; the emitter asks for `input_precision="ieee"`, so Triton does not substitute TF32. Dot is admitted under the v2 and v3 profiles, with `M`, `K`, and `N` powers of two of at least 16, the smallest tile `tl.dot` accepts; no lane is padded.

### Tensor narrow

`tensor narrow f16 %x` and `tensor narrow bf16 %x` round every lane of an `f32` value to the named format, to nearest with ties to even, and keep the result as an `f32` value representable in that format; subnormals of the format are kept, and the sign of zero is preserved. A lane that rounds past the format's largest finite value is undefined behaviour. The emitter converts and widens back with `.to(tl.float16, fp_downcast_rounding="rtne").to(tl.float32)` (or `tl.bfloat16`). Narrowing is admitted under the v2 and v3 profiles.

`-I <path>` adds a search path for resolving link-form `decl`s, and may repeat. `--split-by-source` (C only) emits one `<stem>.c` per source file plus a shared `common.h` into the directory given by `-o`.

## Name mangling

Function names are mangled to `refractir_<name>` on every target, so `@sum` becomes `refractir_sum`. `--emit-main` leaves `@main` unmangled, which is what makes an emitted program directly runnable.

A symbol becomes an external provider named `<func>__<sym>`, both parts with sigils removed: `@f0` with `@?c4` gives `f0__c4`, and `@f0` with `%?k` gives `f0__k`. Global and local symbols are treated alike. The scheme is deterministic, so an embedding can name its providers ahead of time.

## Symbolic programs

Each target expresses the provider in its own idiom. In C a symbol is a zero-argument extern:

```c
extern int32_t f0__c4(void);
```

In WASM it is an import:

```wat
(import "f0" "c4" (func $f0__c4 (result i32)))
```

In Python it is a bare call to a name the module never defines, which the embedding injects into module globals before calling the entry:

```python
import prog
prog.f0__c4 = lambda: 7
print(prog.refractir_sum(10))
```

A vector-typed symbol provides one value per lane, in each target's own shape: the C extern returns the whole vector (the vecext ABI, copied per lane under the `array` / `scalars` strategies), WASM imports one scalar provider per lane (`f0__v__0`, `f0__v__1`, …), and the Python provider returns a list with one element per lane.

## Control flow per target

The C backend emits labels and `goto` and accepts any CFG. The WASM backend encodes any CFG as a `$__pc` + `br_table` dispatch loop. The Python backend has neither `goto` nor labeled `break`, so it reconstructs genuine `while` / `if` control flow, and therefore accepts only reducible CFGs; an irreducible function is a static error naming the offending branch (exit code 4).

`--structured-lowering` asks the C and WASM backends for the same reconstruction, and implies `--require-reducible`; on the Python target, which is already structured, it is a no-op. `--require-reducible` on its own applies the check to any target. [reducibility.md](./reducibility.md) owns the analyses behind all of this: the dominator tree, the reducibility test, the loop forest, the control-tree builder, and the structured lowering pass, along with the `--dump-domtree` / `--dump-loops` / `--dump-control-tree` / `--dump-lowered-tree` inspection flags.

The three targets consume that pipeline at different depths. Python and structured C take the *lowered* control tree, whose multi-level transfers have been rewritten into one-shot guard flags and single-level breaks. Structured WASM takes the *unlowered* tree, because its native multi-level `br` expresses every transfer directly and needs no flags at all.

Take this program:

```sir
fun @sum(%n: i32) : i32 {
  sym %?step : value i32;
  let mut %i: i32 = 0;
  let mut %acc: i32 = 0;
^head:
  br %i < %n, ^body, ^done;
^body:
  %acc = %acc + %?step;
  %i = %i + 1;
  br ^head;
^done:
  ret %acc;
}
```

`--target python` emits, after the runtime preamble:

```python
def refractir_sum(n):
    i = 0
    acc = 0
    while (i) < (n):
        # ^body
        acc = _iadd(acc, sum__step(), 32)
        i = _iadd(i, 1, 32)
    return acc
```

`--target c --structured-lowering`:

```c
int32_t refractir_sum(int32_t refractir_n) {
  int32_t refractir_i = 0;
  int32_t refractir_acc = 0;
  while (((refractir_i)) < ((refractir_n))) {
    // ^body
    refractir_acc = ((refractir_acc) + (sum__step()));
    refractir_i = ((refractir_i) + (1));
  }
  return ((refractir_acc));
}
```

`--target wasm --structured-lowering`, where each natural loop becomes a `(loop $__cont<h>)` continue target and each pending join a `(block $__jn<b>)` that ends just before the join's code, so a `Continue` is `br $__cont<h>` and a `Break` or `JumpJoin` is `br $__jn<b>`:

```wat
(func $sum (param $n i32) (result i32) ...
  (block $__jn2
    (loop $__cont0
      local.get $i
      local.get $n
      i32.lt_s
      if
        ;; ^body
        local.get $acc
        call $sum__step
        i32.add
        local.set $acc
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $__cont0
      else
        br $__jn2
      end
    )
  )
  local.get $acc
  return
  unreachable)
```

Block labels survive as comments on every target. Structured lowering changes only the shape of a function body: expression emission, types, intrinsics, vector lowering, symbols, `--split-by-source` and the UB story are the same as the default emission, with one deliberate refinement, that an executed `unreachable` traps under structured C (`__builtin_trap()`) instead of falling through. Branches that leave several nested loops at once cannot use a native statement in C or Python, so they lower to one-shot guard flags with cascaded breaks (`_brk_*`, `_cnt_*`, `_go_*`) hoisted to function entry; single-level breaks and early returns emit none.

## The Python runtime preamble

Python needs a preamble because its arithmetic does not match RefractIR's. It is emitted once per module and checks strict UB eagerly, at runtime:

* `class RefractIRTrap(Exception)`, raised by every UB event, so an executed UB path exits non-zero with a traceback.
* `_iadd` / `_isub` / `_imul`, which trap when a result leaves `iN` (Python integers are unbounded and would otherwise grow).
* `_sdiv` / `_srem`, which truncate toward zero and trap on division by zero and on `INT_MIN % -1`. Python's flooring `//` and `%` are never used raw.
* Shift-amount range checks, per spec §7.1.
* `f32` arithmetic round-tripped through `struct.pack/unpack('<f', …)` for RNE, and a finiteness check after every FP operation.
* The spec's canonical `i1` values `{0, -1}`, with vector `cmp` yielding a mask list over the same two values.

Aggregates and address-taken scalars lower to a flat byte-indexed buffer addressed by a provenance-tracking `_Ptr(buf, off, stride, lo, hi)`, which traps on null, uninitialized, and out-of-bounds dereference, on cross-object arithmetic, and on cross-object relational comparison. An `undef` slot holds a sentinel that traps on read. Scalars whose address is never taken stay plain Python variables.

The buffer is indexed by packed byte offset, the same scale the interpreter, the C backend and the solver use, so an object's extent is `sizeof(@S) = Σ sizeof(field_i)` rather than a count of leaves. Each leaf owns the first of the bytes it spans and the rest hold a `_PAD` sentinel that traps if read, which is what an interior misaligned access amounts to. The distinction is observable, because spec §7.5 rule 15 gives a field pointer the whole enclosing struct as provenance and arithmetic may legally roam across sibling fields: for `struct @S { f0: [2] i8; f1: i64; }` a `ptr [2] i8` advanced by 4 sits at byte 8, inside the 10-byte object, but at leaf 8 of only 3.

## Vector lowering

`--vec-lowering` chooses the storage form of vector locals; the compute is lane-by-lane on every target and every strategy, and the chosen strategy is stamped into the emitted module as a comment.

| Strategy | Storage |
|---|---|
| `vecext` | Native SIMD registers: GCC vector extensions in C, `v128` in WASM |
| `array` | A lane array in C and Python, shadow-stack frame memory in WASM |
| `scalars` | One variable or WASM local per lane |
| `structarray`, `structscalars` | A per-shape helper struct or class wrapping the two forms above |

C accepts all five and defaults to `vecext`. WASM accepts `vecext`, `array` and `scalars`, and defaults to `vecext`, where a shape wider than 16 bytes splits across registers and lanes move through `extract_lane` / `replace_lane`. Python has no native SIMD value type, so it rejects `vecext` and defaults to `array`; the others map onto lane lists and per-shape helper classes.

`scalars` and `structscalars` reject a dynamic lane index at emission time, since there is no variable to select. WASM's `vecext` accepts one, lowering it to a bounds-trapped per-lane select chain.

The WASM call boundary is the same under every strategy: a vector argument is spilled to caller-owned frame memory and passed by address, and a vector return arrives through a hidden trailing sret address parameter.

## Omitting UB guards

By default the C, WASM and Python backends emit dynamic guards that trap on RefractIR's undefined behaviours: null and out-of-bounds pointer dereference, integer division and remainder by zero, non-finite floating-point results, out-of-range intrinsic arguments. They earn their cost when it is unknown whether a program can trigger UB, and cost without earning it when the program is known to be UB-free, as a `rysmith` program in its default mode is.

`--no-ub-guards` drops them. It is sound exactly when the program never triggers UB, because on such a program the guards never fire and the emitted code is behaviourally identical with or without them. That equivalence is the whole invariant, and it is cross-validated over every UB-free program in `test/xval` (`make cross-validation`).

Value semantics are untouched: truncating division and remainder, logical-shift masking, per-operation `f32` rounding and integer-cast wrapping all survive. Only the traps go. The option is orthogonal to `--no-require`, which governs the separate `require` property assertions, and the two combine.

What `--no-ub-guards` never drops are the non-UB divergences. A `@check_chksum` mismatch traps in every guard mode on all three targets (C aborts unconditionally, Python raises `RefractIRTrap` directly, WASM keeps an unconditional `unreachable`), as do `require` violations (governed separately by `--no-require`) and calls to contract-form declarations.

Each target drops a little differently. C elides the pointer, FP and intrinsic traps, reverts integer division and remainder to plain `/` and `%`, and lowers `unreachable` to `__builtin_unreachable()` rather than `__builtin_trap()`. WASM elides the guard `unreachable` instructions while preserving the surrounding stack shape, and keeps the `unreachable` terminator, since WASM has no no-op hint form. Python lowers arithmetic to inline expressions instead of the checked helpers, and the preamble's `_trap` becomes a no-op while non-UB failures keep raising `RefractIRTrap`.

The reify tools set this automatically for their UB-free output, so it seldom needs passing by hand; see [reify.md](./reify.md).

## What lowers, and what does not

Every scalar type (`i1` through `i64`, `f32`, `f64`), every aggregate, and pointers lower to all three targets, as do function calls, link-form `decl` resolution and the standard intrinsics, including the reify checksum intrinsics `@crc32_update` and `@check_chksum` ([intrinsics.md](./intrinsics.md) §12.7).

Pointers are native C pointers in C, 32-bit linear-memory addresses in WASM, and provenance-tracked `_Ptr` objects in Python. Pointer arithmetic and `ptr - ptr` element distance work on all three, with cross-object arithmetic remaining UB per spec §7.5. Heap allocation is out of scope everywhere: a pointer always refers to a stack-resident `let mut` local (spec §2.8).

There are no optimization passes. The emitted code follows the source closely, which is what makes it useful as a compiler's input rather than its output.

`make cross-validation` covers the C target in both emission modes. The Python target is covered by `make test-backends`, which checks exit-code semantics over the compile corpus; bit-exact `Result:` line parity between `printf %a` and `float.hex()` is what a Python xval phase would still need.

## Refinement and undefined behaviour

Translation is a semantic refinement: the emitted C, WASM or Python must not exhibit a behaviour the source program did not allow. That guarantee holds only for a UB-free input. Once an executed path contains UB, the source program has no defined behaviour to refine, and the target program may do anything at all.

The three targets differ in how much they try to preserve RefractIR's trapping.

Python is the strictest, because its preamble checks every UB-capable operation eagerly and raises `RefractIRTrap` the moment UB executes, with no external tooling: signed overflow, division and remainder by zero, shift range, non-finite FP, out-of-bounds indices and lanes, `undef` reads, and every null, out-of-bounds or cross-object pointer operation.

C preserves trapping as far as C allows. Many RefractIR undefined behaviours map onto native C ones, so compiling the emitted code with sanitizers (`-fsanitize=address,undefined,float-cast-overflow,pointer-compare,pointer-subtract`) catches them, and the backend emits explicit guards for the rest.

WASM makes no such effort for arithmetic UB. Its dynamic guards cover pointers, intrinsic preconditions and the FP `%` intermediate, but signed overflow, over-shift and a non-finite result of a plain FP operation are lowered to bare WASM instructions, so an executed arithmetic UB follows whatever the instruction does. This program is UB under spec §7.1 rule 4:

```sir
fun @main() : i32 {
  let mut %min: i32 = -2147483648;
  let mut %neg1: i32 = -1;
  let mut %res: i32 = 0;
^entry:
  %res = %min % %neg1;
  ret %res;
}
```

The interpreter and the solver both treat that path as invalid. WASM lowers `%` to `i32.rem_s`, which, unlike `i32.div_s`, does not trap on `INT_MIN` and `-1`: it returns `0`, and the program exits successfully. That is the refinement guarantee expiring, exactly as documented.
