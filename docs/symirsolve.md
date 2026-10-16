# symirsolve

`symirsolve` turns a symbolic RefractIR program into a concrete one. It executes a chosen path symbolically, hands the resulting bit-vector constraints to an SMT solver, and substitutes the model back into the program, so every `@?x` / `%?y` becomes a literal.

The output is an ordinary `.sir` program: `symiri` runs it and `symirc` translates it. `symirc` also translates a symbolic program directly, emitting each symbol as an external provider hook, but only `symirsolve` decides what the symbols are.

## Usage

```bash
symirsolve <input.sir> [--path <labels> | --sample <n>] [options]
```

The full option list is `symirsolve --help`, declared in [src/symirsolve.cpp](../src/symirsolve.cpp). The entry function defaults to `@main`.

Solve along one explicit path, and let the sampler find a feasible one:

```bash
symirsolve template.sir --path '^entry,^b1,^b3,^b1,^b2,^exit' -o concrete.sir
symirsolve template.sir --sample 100 --require-terminal -o concrete.sir
```

Pin a symbol before solving, and record the model alongside the program:

```bash
symirsolve template.sir --path '^entry,^b1,^exit' --sym %?c4=3 -o concrete.sir
symirsolve template.sir --path '^entry,^b1,^exit' --emit-model model.json -o concrete.sir
```

## Where the constraints come from

Four sources feed the path condition:

* Symbol declarations, including a declared domain (`in [lo,hi]` or `in {…}`).
* The chosen path. A `br cond, ^t, ^f;` contributes `cond` when the path continues at `^t` and `not(cond)` when it continues at `^f`.
* `assume` clauses, which constrain feasibility.
* `require` clauses, which constrain the property being solved for.

On top of those, every operation that can be undefined contributes a safety guard: an SMT term true exactly when that operation is well defined on this path (division by zero, signed overflow, over-shift, out-of-bounds array or lane access, `undef` read, null / out-of-bounds / cross-object pointer use, FP overflow or NaN, float-to-int out of range, and the rest of [undefined.md](./undefined.md)). What the solver does with those guards is the mode.

## Solving modes

The default mode asserts every safety guard true. A returned model runs the whole path without triggering any UB; a path whose guards cannot hold together is UNSAT, which is how an infeasible path reports.

`--require-ub` negates the conjunction of the guards, `not (and g_1 … g_n)`, so a returned model violates at least one of them and the concretized program is guaranteed to trigger UB somewhere on the path. Branch conditions, `assume` and `require` are still asserted normally, so the interpreter follows the same path to the same trap. A path with no UB-capable operation has an empty guard set and reports UNSAT. Because the interpreter stops at the first UB it reaches, such a program traps instead of returning a value.

`--require-nonterm` solves for divergence. The path must be a lasso: its final block is a loop header that appears earlier on the path, and the segment between the two visits is one lap of the cycle. Guards are asserted true as in the default mode, and the complete mutable state at the header is additionally constrained to be bit-identical on the first and last visit, leaf by leaf over every `let mut` (parameters and symbols are immutable, so they need no constraint). A deterministic lap that returns to its own entry state replays forever, so a model that satisfies this diverges UB-free, certified by a finite one-lap witness rather than by running to completion. A path that does not revisit its final block reports UNSAT. Such a program has no `ret`, so a solved lasso carries no `ret=` in its `SOLVED` header.

The state equality here is value identity, not IEEE `==`: the two disagree on `+0.0` versus `-0.0`, and identity is the one that keeps a recurrence honest ([float.md](./float.md) §2.1).

## Paths and sampling

A path is a comma-separated sequence of block labels, repeats included:

```text
^entry,^b1,^b3,^b1,^b2,^exit
```

The sequence must follow `br` edges in the CFG. `symirsolve` reads it as the choice made at each conditional branch.

`--sample N` walks the CFG randomly from the entry, up to `N` times, and stops at the first path that solves. With `--path` also given, the path is a mandatory prefix of every sampled walk. A walk that reaches `--max-path-len` without hitting a `ret` is discarded, unless `--require-terminal` is set, in which case it is completed by the shortest route to a `ret` block.

`-j N` runs `N` sampling threads, each with its own solver instance and its own seed derived from `--seed`, and the first SAT result stops the rest; `-j 0` uses `std::thread::hardware_concurrency()`. This is worth setting when a run samples many paths or the search space is wide. It applies to the Bitwuzla backend only: Z3's global context is not thread-safe, so the AliveSMT backend warns and falls back to one thread. `--num-smt-threads N` is the other axis, parallelism inside a single solver instance (Bitwuzla's `NTHREADS`, Z3's `sat.threads`); the two combine.

`--timeout-ms` bounds each `check-sat` call rather than the whole run, so it caps how long one query may take, not how long `symirsolve` may run.

## Outputs

A solve reports SAT, UNSAT (no assignment satisfies the constraints on this path) or UNKNOWN (the solver reached its timeout or gave up). The status is always reported with a diagnostic.

On SAT, `-o <file>` writes the concrete `.sir` with every symbol replaced by a literal, `--emit-model <file>` writes the symbol assignments as nested JSON, and `--dump-ast` prints the concretized AST. Floating-point values cross all three boundaries through the canonical formatter ([float.md](./float.md) §9).

`--tensor-profile-version v1|v2|v3|v4` selects the tensor profile the program is checked against before it is solved, through the same capability analysis every other consumer uses. An out-of-profile tensor program is a static error, not an UNSAT result.

A tensor **parameter** is a runtime argument the solver chooses, exactly as a symbol is. Its witnesses are exported beside the symbols rather than substituted into the program: `--emit-model` adds one record per tensor parameter, `{"shape": [...], "element": "i32"|"f32", "leaves": [...]}`, to the same per-function object, and `--emit-tensor-params <file>` writes those records on their own. Leaves are row-major logical elements, and an `f32` leaf is the exact widening of its 32-bit pattern, so narrowing it back reproduces the model bit for bit. The record is written only on SAT; its absence means no witness, not an empty one.

`--tensor-param-domain <lo>,<hi>` bounds every integer leaf of every entry parameter, scalar parameters and tensor elements alike, so a consumer can keep a witness inside a domain its reference can execute. It is rejected for a parameter with a non-integer leaf or when `lo > hi`; absent, parameters stay unbounded.

## Bit-vector encoding

Integer types map to BV sorts, `iN` to `(_ BitVec N)`, with `/` encoded as `bvsdiv` and `%` as `bvsrem` to match the language's truncating division (spec §8). Floating-point uses QF_FP; [float.md](./float.md) §10 states the encoding.

The bounded exact tensor subset is expanded into scalar solver terms, with shape correspondence checked before solving. Tensor f32 arithmetic and reductions, `exp`, `softmax`, and tensor-v4 remainder have no numerical SMT encoding and report `unsupported_theory`. Bit-preserving tensor operations retain their supported encodings. [Tensor capability analysis](../src/analysis/tensor.cpp) checks the profile; [tensor symbolic execution](../src/solver/tensor.cpp) defines the supported encodings.

## Pointer encoding

A pointer is a 64-bit BV tag identifying the storage it addresses, with tag `0` reserved for `null`. The base tag is the FNV-1a hash of the local's name, and sub-object addressing advances it by a leaf-unit offset: each scalar or pointer leaf counts as one unit, arrays and structs sum their leaves (`sizeofTagUnits`). This is not a byte offset. `addr`, `load` and `store` dispatch through an `ite` chain over the candidate targets of matching pointee type, so a load or store through a symbolic pointer resolves to the right cell without SMT array theory.

The encoding covers `addr` of a whole `let mut` local and of an array element or struct field, `ptrindex` / `ptrfield` navigation including one-past-the-end, pointer arithmetic with in-bounds and cross-object UB enforced per provenance, `load` / `store` through any `ptr T` chain down to a `T`-typed cell, pointer equality and inequality on the tags, relational comparison as UB across distinct objects, and pointer parameters threaded through interprocedural `call` with caller-store coherence on callee stores. The provenance model is the spec's: base tag plus leaf-unit offset, provenance being the immediate containing aggregate, with the typed-access check applied at the dereference (spec §7.5 rules 10 to 19, §9.4).

Three things the abstract model in spec §9.4 allows are outside the encoding, and each is a spec §13 non-goal:

* `sym` of pointer type, which would need a richer address-domain theory.
* Contract-form `decl` memory havoc beyond the `addr %x` and plain-pointer-local argument forms. A caller passing an aggregate, a `ptrindex` / `ptrfield` derivation, or a nested pointer must constrain the post-state explicitly (spec §9.6.2 step 4).
* A user-supplied sub-path through a branchy callee; those are sampled with a seeded random walk instead (spec §9.6.4).

Byte-granular reasoning is out of reach by construction, since a scalar leaf is a single addressable unit rather than a run of bytes, which is why `@memcpy` and `@memset` stay a non-goal. What the model buys for that price is aliasing and pointer arithmetic resolved by `ite` chains over candidate cells, with no `Mem[T]` array and no quantifiers.
