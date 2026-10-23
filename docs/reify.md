# Semantic reification and the reify tools

Reify generates random programs by a technique called *semantic reification*. Syntactic reification works on program text; semantic reification works on program meaning, and separates two kinds of it: compile-time semantics, what a program *can* do, and runtime semantics, what a program *actually does* on one input.

Given an arbitrary control flow graph `g` and an arbitrary entry-to-exit path `pi` through it, reify produces a program `P`, an input `i`, and an output `o` such that:

1. `P` is syntactically and semantically correct for `i`;
2. `g` is the CFG of `P`;
3. `P(i)` deterministically follows `pi` and produces `o`.

The `g` fixes the compile-time semantics and the `pi` fixes the runtime semantics, so both are chosen rather than discovered.

That combination is what makes the output useful for compiler testing. A compiler must reason about every possible execution when it optimizes, even though the runtime semantics of one input are fixed, and semantic reification exposes errors in that reasoning while keeping each generated program deterministic and free of undefined behaviour on its stated input. Because `g` and `pi` are arbitrary, the generator reaches control structures and data flows that a grammar-driven generator does not: unbounded loops, irreducible regions, deep type mixing. And because `o` is known before the program is ever compiled, a miscompilation shows up as a wrong answer rather than as a disagreement between two compilers, so no pseudo-oracle is needed.

Reify separates *leaf function generation*, compact functions with no calls, from *whole-program generation*, which composes leaf functions under an arbitrary call graph. `rysmith` implements the first, `rylink` the second, and `rytwin` transforms either into an equivalent variant.

## Leaf function generation

Two of the clauses hold by construction: the program is built over `g` itself, so `g` is its CFG, and its statements are typed as they are generated, so it is syntactically correct. What is left is that `P` runs `pi` without UB and produces `o` on `i`, and that is a question about values.

Let `T` be that program with the values that matter left as symbols, and `T[x]` the program obtained by assigning `x` to them. The task is to find one `x`:

```
∃x. follows(T[x], pi) ∧ safe(T[x], pi) ∧ interesting(x)
```

`follows` is the conjunction of the branch conditions `pi` decides, and `safe` the UB guard of every operation `pi` executes; between them they carry what construction could not. `interesting` answers to no clause at all; it excludes degenerate values, and its job is to make the program worth compiling. Every conjunct is a bit-vector formula over `x`, so this is a first-order existential over values: decidable, and settled by a single query. A model is `P`, the values it gives the parameters are `i`, and running it produces `o`.

Which conjuncts are asserted decides what kind of program comes out. `safe` as stated gives a clean run, `¬safe` one that triggers UB, and a `pi` that returns to a loop header in the state it left one that diverges.

`rysmith` implements this:

```
S1. CFG generation   - random control-flow skeleton
S2. Path sampling    - random entry-to-exit walk through the CFG
S3. Program seeding  - populate all blocks with typed statements
S4. Concretization   - solve the symbols along the path via SMT
S5. Lowering         - emit concrete RefractIR, then C / WASM / Python
S6. Validation       - execute and compare the output against o
```

### S1: CFG generation

A random CFG starts as a spanning chain from entry through the interior blocks to exit, and then stochastically gains branch edges (a second successor pointing forward) and back edges (producing loops). The result is always connected and always has a path to exit.

A back edge may land past a loop header, which makes the CFG irreducible. When reducible CFGs are required, the CFG is repaired instead of resampled: a retreating edge whose target does not dominate its source is deleted, one per re-analysis pass, so every valid loop survives and only irreducible cycles are broken.

### S2: Path sampling

The execution path is a random walk from entry to exit, counting each back-edge traversal so loop iteration counts stay bounded. A walk that gets stuck escapes to exit along the shortest BFS route. The result is a sequence of block labels:

```
^entry -> ^b0 -> ^b3 -> ^b0 -> ^b4 -> ^exit
```

One CFG yields many distinct paths, differing in which branches are taken and how often each loop runs.

### S3: Program seeding

Every block gets typed statements, and a block's role decides what kind.

An on-path block, one that appears in `pi`, uses *symbolic* variables whose values the solver picks. Symbols are declared with a domain and a kind annotation (`coef`, `value`, `index`). Interest constraints, `require` statements excluding degenerate coefficients such as 0, 1 and -1, push the solver toward programs that are worth compiling.

An off-path block uses concrete random literals. It never executes, because the solver pins every on-path branch, so control never reaches an off-path successor. Off-path code is therefore left unconstrained and may contain UB: division by a variable that could be zero, signed overflow from a wide literal, an index that could leave its array. None of it reaches the differential oracle, and all of it is surface for the optimizer's dead-code elimination, alias analysis and vectorization to work over.

Off-path volume costs the solver nothing, which is why the volume knobs (`--n-stmts`, `--min-atoms`, `--max-atoms`) describe on-path blocks and off-path blocks scale them by `--off-path-multiplier`. On-path volume is the solver's bottleneck and tunes independently.

#### Types

Generation draws from the full RefractIR type lattice, each variable choosing independently: integer scalars `i8` through `i64` and arbitrary `iN`, floating-point `f32` and `f64`, arrays `[N] T` and structs `@Name { … }` up to a bounded nesting depth, vectors `<N> T`, and pointers `ptr T` including `ptr ptr T` chains. Mixed types appear within one function, and a scalar type boundary is crossed by an explicit cast atom, which is what exercises a compiler's promotion and narrowing paths.

A floating-point variable is initialized on-path by casting from an integer symbol, `(f32) %?s0`, which keeps the SMT problem in bit-vector theory. Off-path float code uses concrete literals.

#### Expressions

Expressions are generated type-directedly: given a target type `T`, the generator produces an expression of type `T`, and every atom in one expression shares that type. The repertoire covers linear terms with a symbolic coefficient (`coef_sym * var`), the bitwise and shift operators, `~var`, explicit casts, `load` through a pointer variable, `addr` of a local, a one-level `select`, and division or modulo by a concrete non-zero denominator.

Division and modulo keep a concrete denominator on-path, as in `%?s3 / 7`, which yields the divide-by-constant patterns that stress strength reduction. Off-path division uses any literal, zero included.

A symbolic coefficient is typed to match its expression context, so an `i64` expression takes a `coef i64` symbol and an `i32` expression a `coef i32` one. That reads as natural code, a 64-bit multiply against a 64-bit coefficient, and reaches type-specific optimization patterns.

#### Pointer initialization

`addr lv` is an expression atom, not a valid `let` initializer, so a pointer variable is declared `undef` and assigned in the entry block before anything else is generated:

```sir
fun @func0() : i32 {
  let mut %v0: i32 = %?s0;           // integer var, init from input sym
  let mut %p0: ptr i32 = undef;      // pointer var, init deferred
  let mut %pp0: ptr ptr i32 = undef; // depth-2 pointer, init deferred
^entry:
  %p0 = addr %v0;
  %pp0 = addr %p0;
  require %?s0 != 0, "nonzero input";
  ...
```

`^entry` is the first block on every path, so every pointer is definitely initialized whichever path is sampled.

### S4: Concretization

`symirsolve`, or the in-process symbolic executor when running under `rysmith`, executes `pi` symbolically: it collects the path conditions from branch terminators, the `require` constraints (interest constraints and UB guards), and the computed value of each assignment, encodes them as bit-vector constraints, and asks Bitwuzla for a satisfying assignment. The model is substituted back through `SIRPrinter` to give a fully concrete `.sir`. Off-path blocks pass through untouched, their literals needing no solving.

Solving the same symbolic template again, with a different solver or RNG seed, gives a structurally similar program with different numbers, so one control-flow structure yields many distinct optimization problems.

### S5: Lowering

`symirc` lowers the concrete `.sir` to C, WASM or Python:

```
rysmith -> concrete .sir -> symirc -t c      -> .c   -> gcc / clang (link -lm)
                         -> symirc -t wasm   -> .wat / .wasm
                         -> symirc -t python -> .py
```

The expected output `o` is the function's return value, a checksum over every live variable at exit.

### S6: Validation

The program is compiled and executed under `i`. An output other than `o` is a potential miscompilation:

```
Expected:        func0() = -847
Compiled (-O3):  func0() = -846   -> POTENTIAL BUG
```

Differential testing across compiler versions or optimization levels works the same way.

## Whole-program generation

S1 to S6 produce independent functions. To build a whole program, reify generates a random call graph and realizes each of its edges as a real call, without disturbing what the caller computes.

The leaves arrive already satisfying the three clauses, and composition adds a fourth: the call graph of `P` is the sampled DAG `h`. An edge of `h` is realized at a point where the caller already computes some value `v`, by routing that value through the callee. Writing `S` for the states the profiled run reaches that point in, the task is to find an argument expression `e1` and a result expression `e2` such that

```
∃e1. ∃e2. ∀s ∈ S. e2(f(e1(s))) = v(s)
```

`e1` builds the callee's parameters out of variables the caller has in scope, and `e2` turns the callee's return value back into the variable or constant the caller wanted. Neither may trap, and neither may the call between them, since the program has to stay UB-free on `i`. Both are unknown functions, so this is a second-order synthesis problem.

What collapses it is that `f` is known at the inputs it was concretized on and nowhere else. A non-constant `e1` would demand `f`'s behaviour at every state in `S`; behaviour nobody has computed, on inputs the callee was never proven UB-free for.

In `rylink`, `e1` is therefore *value*-constant, pinned to an input `f` was concretized on. Then `f(e1(s))` is the known `o` whatever `s` is, the callee runs on an input it was proven safe for, and `e2` is left alone and first-order: for a constant `v = c`, it is `+ (c - o)`.

Being pinned in value does not make `e1` a literal. It is drawn from a catalog of constructions, each of which reaches the same `i` from whatever the caller holds where the call lands, writing `σ₁…σₘ` for the distinct states the profiled run passes that point in:

```
literal-or-bias   e1 = %x + (i - x)                  %x steady at x across every σⱼ
xor-mask          e1 = i ^ ⋀ⱼ (g ^ gⱼ)               g a chosen XOR-fold, gⱼ its value at σⱼ
prime-interp      e1 = P(vars) - (i mod p) + i       P(σⱼ) = i mod p, P(t) ≠ i mod p off S
```

The first holds only where a variable does; the other two answer to every `σⱼ` at once, so the variables they read are free to move between visits.

`rylink` implements this:

```
W1. Pool ingest       - load a directory of rysmith (.sir + .json) pairs
W2. CG generation     - pick K functions, build a DAG call graph over them
W3. Bundle merge      - parse each .sir, union into one Program (dedup structs by name)
W4. Peephole rewrite  - per (caller, callee) edge, splice `call @callee(args) + (c - o)`
W5. Anti-optimization - rewrite every block by identities that cannot trap
W6. Lowering          - program.sir plus optional C / WASM / Python
W7. Validation        - run the bundled entry and check its outcome
```

Which realization a leaf brings is a choice among the `--n-concretes` concretizations `rysmith` emitted for it, and it fixes the `i` and the `o` the splice is built from. `CallRealizeTransform` consumes each rewrite site at most once across the whole program: composing two rewrites on one literal would build a left-to-right call chain, `f1() + f2() + …`, whose prefix sums can wrap even though each rewrite is individually sound in bit-vector arithmetic.

Every function in a bundle comes out of the same statement generator, so they read alike. W5 breaks that up by rewriting them with identities applied in the direction a compiler does not take: reversed peepholes, arithmetic and bitwise crossings, restructuring. The rule families live in [src/reify/antiopt](../src/reify/antiopt), driven by [include/reify/antiopt.hpp](../include/reify/antiopt.hpp). A bundled program is concrete, with no set of states to prove anything over, so only rules that cannot introduce a trapping operation apply; those hold whatever the state does, and so does any composition of them.

## Twin-program generation

A twin program is an equivalent variant: `f2(i) == f1(i)` for every input, with the same UB outcome. `rytwin` builds one from a program and the input that concretizes it, so the execution is deterministic and known, and no solver is involved anywhere.

Its unit is a *region*: the maximal single-entry region rooted at an executed block, covering every later block that the entry dominates on the executed path. For a region `R` and the entry state `s0` the profiled run reaches it in, the task is to find a guard `G` and a twin body `B` such that

```
∃G. ∃B. G(s0) ∧ ∀s. G(s) ⇒ R(s) ≡ B(s)
```

where `R(s) ≡ B(s)` holds when, started in `s` or `s0`, the region and the twin leave at the same block, neither hits UB on the way, and the states they leave in are bit-identical.

`G` and `B` are unknown *functions*, so this is a challenging, second-order synthesis problem under one ∀.

In `rytwin`, neither `G` nor `B` is searched for over a space of functions: T3 witnesses `B` by construction, and T4 draws `G` from a single template, a box around `s0`, so `G(s0)` holds for free. What is left of the ∀ is a first-order question about a fixed body and a fixed set of states.

`rytwin` implements this:

```
T1. State profile     - every initialized local at each on-path point, from the
                        .state.json sidecar (rysmith --emit-state) or by interpreting
T2. Region planning   - pick a dominance region whose live-in state a guard can state
T3. Trace flattening  - the statements the run executed, branches dropped, loops laid out
T4. Guard box         - a state-set pass opens each leaf as far as it can prove
T5. Anti-optimization - the W5 engine again, here licensed by the box
T6. Graft             - splice the guarded diamond; --validate spot-checks the box
```

The graft is a diamond at the region entry:

```
^X:           br call @__twg_<fn>_<X>(<state>) != 0  ->  ^X__twin  else  ^X__orig
^X__twin:     R'                            ->  br ^<exit>
^X__orig:     the region's entry block, unchanged - the region runs on from here
```

The guard takes over the region entry's own label, so no predecessor edge is rewritten, and the two arms rejoin at the exit the region left to.

The twin body is the executed trace, which makes it correct for every state that follows the same path UB-free, a far larger set than the one profiled state. The guard therefore states per leaf what the state-set pass proves: nothing at all (free), `lo <= x <= hi` (ranged), or `x == v` (pinned). Every comparison the guard makes is total, so the guard itself cannot trap.

Having a proof also buys the rewriting more than W5 gets. Rules that can trap are kept wherever the box clears them, and a family of rules reads the box's facts directly.

A region holding a non-intrinsic call is not twinned, since a callee could mutate state the frame diff does not see.

## rysmith

`rysmith` runs S1 to S5 in one process. It builds the program AST in memory, calls the symbolic executor in-process rather than as a subprocess, and writes concrete `.sir` through `SIRPrinter`. It can invoke `symiri` for S6 under `--validate`. Its subject is function generation; it does not test compilers itself.

```
rysmith [OPTIONS]
```

The full option list is `rysmith --help`, declared in [src/rysmith.cpp](../src/rysmith.cpp). The knobs group into type control (`--no-fp`, `--max-ptr-depth`, `--max-agg-nest`, …), generation volume (`--n-vars`, `--n-stmts`, `--min-atoms`, `--off-path-multiplier`), operator repertoire (`--no-divmod`, `--no-select`, `--no-intrinsics`, …), CFG shape (`--n-bbls`, `--p-branch`, `--p-backedge`), solver control (`--timeout`, `--seed`, `--max-retries`), and output (`-n`, `--n-concretes`, `--n-examples`, `-o`, `--target`, the `--emit-*` family). The sections below cover the modes whose semantics are not evident from the flag name.

```sh
# 10 functions, 3 concretizations each, all validated
rysmith -n 10 --n-concretes 3 --validate -o out/

# 10 functions, two distinct solved examples per concretization
rysmith -n 10 --n-examples 2 -o out/

# stress pointers and mixed types, no floats
rysmith -n 20 --no-fp --max-ptr-depth 2 --max-agg-nest 2 -o out/

# reproduce a run
rysmith -n 30 --seed 42 -o out/
```

### Tensor case generation

`--profile tensor` selects the separate property-directed tensor generator. Choose a dtype and prescribed property explicitly; `--tensor-synthesis construction|exact` selects the candidate route. The generator validates the property against the independent CPU reference before it accepts a case and writes `program.sir`, typed input and expected buffers, and a `case.json` manifest for the tensor runner.

```sh
rysmith --profile tensor --tensor-dtype i32 --tensor-property coordinates --tensor-synthesis exact --seed 0 -n 1 -o tensor_cases/
rysmith --profile tensor --tensor-dtype f32 --tensor-property softmax-dominance --tensor-synthesis construction --seed 0 -n 1 -o tensor_cases/
```

`--max-retries` bounds candidate attempts. A requested case shortfall is reported with a nonzero exit status. Scalar-only controls such as `--n-examples` are rejected in this profile. The optional `cvc5` route requires a matching recorded feasibility result; a failed feasibility gate leaves that route unavailable. [Tensor capability analysis](../src/analysis/tensor.cpp) enforces the selected profile, and [the generator](../src/reify/tensor_gen.cpp) defines its property families.

`--tensor-generator compositional` selects the typed dependency-graph builder for tensor-v2 and tensor-v3 instead of the per-stratum constructors. It draws a graph from two or three runtime tensor inputs, admits add, sub, mul against a small literal, transpose, expand, broadcast, sum, and whole-value copy, guarantees a shared intermediate and (within the default operator budget) two reductions with different consumers, and prunes every node outside the returned value's backward slice. Operators are placed between the required structures rather than after them, so a reduction can consume a transposed, expanded, or broadcast operand. The operator, shape, and axis records in the manifest come from walking the checked program, not from a second graph description. `--tensor-dtype` and `--tensor-profile-version` keep their meaning; v1 and v4 are rejected rather than partially supported, and `--tensor-input-pattern` is rejected because this route selects values through its input policy instead.

#### Loop regions

The tensor-v2 `loop-reduction` stratum can build a bounded two-state loop region instead of one of its single loop forms. Four options choose that region, and naming any one of them asks for it:

| Option | Values | Domain |
|---|---|---|
| `--tensor-region-family` | `fold`, `orthogonal-reductions`, `reduction-arithmetic`, `cross-state`, `mixed-depth`, `broadcast-hoist` | the skeleton connecting the two states to the named results |
| `--tensor-region-shape rows,columns` | two extents | powers of two of at least two, holding at most the profile's 512 elements |
| `--tensor-region-iterations N` | one to eight | the counted trips of the region's bounded loop |
| `--tensor-region-arithmetic` | `additive`, `subtraction`, `multiplication`, `mixed` | the operation the region's root placements use |

```sh
# a four-trip cross-state region over a 32x16 state, with mixed arithmetic
rysmith --profile tensor --tensor-dtype f32 --tensor-profile-version v2 \
  --tensor-region-family cross-state --tensor-region-shape 32,16 \
  --tensor-region-iterations 4 --tensor-region-arithmetic mixed \
  --seed 0 -n 4 -o tensor_cases/
```

`broadcast-hoist` states a connection a layout pass can act on: the state carrying the runtime input is reduced, the one-extent result is broadcast back over the state shape, and that view is consumed twice - by a second reduction on the other axis and by the elementwise path into the other root - so a copy that takes one consumer over would leave the original with the other. It needs two reducible axes, and a state shape with a singleton axis is refused rather than served by repeating one reduction axis.

The region's two states are supplied in a fixed order: the first carries the runtime input and the second starts at zero. The shared producer is additive in every mode, because a product there is zero on every trip and a difference cancels one state out of the update; the arithmetic mode decides the operation each root placement uses. A placement that would multiply the zero-initialized state adds instead, and a multiplicative placement joins the state that carries the input, so neither state's update is annihilated and the mode stays visible in the body.

A case generated from a named request records the values it used under `generator.region` in its manifest, with the shape, the trip count, the family, and the operation mode. Naming no region option keeps the region policy's own 2x2 body, its two trips, and additive arithmetic, and writes no `generator.region`.

A value outside its domain, such as `3,3` or nine trips, is refused with the rule it breaks. A route that cannot build a region - the compositional builder, a profile other than tensor-v2, or a stratum other than `loop-reduction` - is refused rather than emitting a case that ignores the request.

#### Free-form window region assembly

The free-form builder grows one typed DAG and the assembler wraps it in a counted loop. A loop names its runtime parameters in two roles: a region input is a declared local copied from its parameter in `^entry`, while a window backing parameter (`FreeFormLoop::windowSources`) is an assembly-only tensor that only the window production reads. Backing parameters are emitted before the runtime parameters, never enter the region value pool, and never receive a local copy, so a `[20,4]` buffer can supply `[4,4]` tiles without materializing the whole buffer. A window source names either a runtime parameter (the original route) or a backing parameter; its witness is computed from the source's own extent.

The test exporter exposes the study's construction requests:

| Option | Meaning |
|---|---|
| `--window rows,stride,active` | adds the `%tile` window invariant; needs a source |
| `--window-source %x` | uses a runtime state parameter as the backing source |
| `--window-buffer-rows P` | adds `%window_buffer: [P,C]`; requires `--window`, conflicts with `--window-source` |
| `--window-requirement reduction\|shared\|extrema-transpose\|extrema-expand` | adds the bounded live relationships below; requires `--window` |
| `--max-attempts N`, `--max-multiplications N` | set the builder's internal limits instead of only recording them |

A requirement is a bounded live-graph relationship, not a graph constraint language: a named input must reach every listed result root; the connecting path must contain one operation from `pathOps` and, when `pathShapeOps` is nonempty, one from that set too; and `distinctConsumers` requires a value derived from the input with two distinct live consuming nodes whose branches reach two distinct listed roots. Requirements bias growth, are re-validated on the pruned live graph, and emit witnesses (source/result names, path values and edges, live operation IDs, consumer nodes and their roots) plus a normalized live-DAG record with a connectivity signature and typed fingerprint. A path counts only if it is live; operations on separate branches do not combine, and an input selected as a root is not a consumer. Path requirements reserve three bridge nodes per listed result, spent through the same legal-action enumerator when growth leaves a path unfinished. Refusals name `required_input_unreachable`, `required_path_missing`, `distinct_consumers_missing`, or `node_budget_exhausted`.

#### Runtime input selection

`--tensor-input-policy` chooses how a compositional case's runtime inputs are proposed for that frozen program, and the program does not depend on the choice:

```sh
rysmith --profile tensor --tensor-dtype i32 --tensor-profile-version v2 \
  --tensor-generator compositional --tensor-input-policy boundary --seed 0 -n 4 -o tensor_cases/
rysmith --profile tensor --tensor-dtype i32 --tensor-profile-version v2 \
  --tensor-generator compositional --tensor-input-policy exact \
  --tensor-property-target 18 --seed 0 -n 1 -o tensor_cases/
rysmith --profile tensor --tensor-dtype i32 --tensor-profile-version v2 \
  --tensor-generator compositional --tensor-input-policy exact \
  --tensor-output-element 0,1=-1 --tensor-output-element 1,0=3 --seed 0 -n 1 -o tensor_cases/
```

`ordinary` draws seeded values from `[1, 2]`. `boundary` draws each element from `{-1, 0, 1, 2}` using a deterministic stream keyed by the seed, parameter name, dtype, and shape. Equal-shaped parameters have separate streams; repeated values remain possible. Boundary cases record `boundary-splitmix64-v1` in `generator.input_sampler_version` and `coverage.input_behavior.sampler_version`, and the loader requires the two fields to agree. `exact` states the property before any input is accepted: `--tensor-property-target` names the scalar return value or the element sum of a rank-one tensor return, the generator builds a property-constrained clone of the same computation, checks it under the selected profile, and solves it for at most 64 symbolic input elements within a five-second budget. The clone is solved inside `[-1, 2]`, the union of what the samplers propose, so a target either sampler can witness is never out of the exact route's domain. The three policies do not use identical value sets: ordinary's `[1, 2]` is narrower than boundary's and exact's `[-1, 2]`, and each case records the interval it used. A clone whose digest differs from the frozen program once query constraints are removed is rejected. Unsupported graphs, an exceeded budget, an unsatisfiable target, and an undecided query are separate outcomes in `generation.json`; none of them is reported as an accepted case.

The value is one complete literal: decimal, or `0x`/`0o`/`0b` prefixed, with an optional sign (`+0x10` is 16, and `1.5`, `1junk`, and `+0x80000000` are rejected rather than read as prefixes). Repeatable `--tensor-output-element <coordinate>,...=<i32>` states one required value per logical output coordinate, for tensor-v2 i32 outputs only, and is mutually exclusive with `--tensor-property-target`. Coordinates must state every output dimension, stay in bounds, and not repeat; the accepted list is sorted row-major so the same request always serializes the same way. A request that names every output coordinate is a complete target and is recorded as such; a shorter one is partial, and the coordinates it does not name take whatever the reference computes from the witness, so the unconstrained output is never taken from the request. Each constraint becomes one `tensor extract` and one scalar `require` in the query clone, and the source keeps the original computation. `property.constraint` in the manifest carries the typed request, and both the generator and the loader check it against the case's own reference output: a permutation with the same element total is rejected because elements are compared, not totals. Solver completion depends on the graph and property. An undecided query reports `input_policy_unknown`; a complete target does not guarantee completion within the budget.

Every accepted case states its selection provenance: the policy that proposed the inputs, the inclusive leaf domain they respect, and the output target they were asked to satisfy, in `generator` and in `coverage.input_behavior`. A case with a target records `exact_smt` evidence, and the target is checked against the case's own reference output both when the case is written and when it is loaded, so a witness that does not satisfy its stated property is rejected rather than admitted. The `coverage.input_behavior.pattern` field names the sampler that ran (`seeded-random`, `boundary-value`, or `smt-selected`), not a value family the route never built.

### Output format

Each concrete `.sir` holds one function `@funcN` with every variable initialized to a concrete value. The `^exit` block folds every scalar leaf of every local and parameter, recursing through nested arrays, structs and vector lanes, into a running CRC32 state and returns it:

```sir
intrinsic @crc32_update(%state: i32, %val: i32) : i32;

fun @func0(%pa0: i32) : i32 {
  let mut %v0: i32 = 7;
  let mut %v1: i64 = -3;
  let mut %p0: ptr i32 = undef;
  let mut %_chk: i32 = 0;
^entry:
  %p0 = addr %v0;
  ...
^exit:
  %_chk = 0;
  %_chk = call @crc32_update(%_chk, %v0);
  %_chk = call @crc32_update(%_chk, %v1 as i32);
  %_chk = call @crc32_update(%_chk, %pa0);
  ret %_chk;
}
```

That return value is the expected output `o`. The solver never sees the CRC32 recurrence: it is asked for the cheaper sum-form contract, `%_chk = %_chk + atom`, and a post-solve rewriter replaces each accumulator step with a `@crc32_update` call before the file is written. After lowering to C, the function returns `o` whatever the compiler and optimization level, because the helper carries a function-local `static` table and a `static __attribute__((noinline))` qualifier that stop the optimizer folding the chain ([intrinsics.md](./intrinsics.md) §12.7).

`--emit-main` appends a `@main()` that calls the entry with the solver's parameter values and asserts the return against the captured checksum through `@check_chksum(EXPECTED, %r);`. The C lowering of `@check_chksum` aborts on mismatch, and that externally visible side effect anchors the whole call chain against interprocedural constant propagation, so the body survives `-O3 -flto`.

`--n-examples N` asks for `N` distinct input/output examples per concretized `.sir`. The first is the concretized solve; each further one comes from re-solving a clone of the same symbolic template armed with two splices into the entry block: one `require %?s == v` per solved sym pinning it to the value the emitted program embeds, and one exclusion require per collected input, `%paK != v` on a single parameter coordinate. Only the first example is consumed by `rylink` and `rytwin`.

### Generating UB-triggering programs

By default the solver asserts each operation's safety guard, so the program executes cleanly and returns its checksum. `--require-ub` asks it to negate the conjunction of those guards instead, delegating to `symirsolve`'s RequireUB mode ([symirsolve.md](./symirsolve.md)), so the concretization triggers at least one UB on the sampled path. That is what exercises a downstream tool's UB detection.

`--require-ub` implies `--no-crc32`, and the reason matters. The solver reasons about the sum-form checksum, and one legitimate way to satisfy "at least one UB on this path" is to overflow that signed accumulator. The post-solve CRC32 rewriter would then replace `%_chk = %_chk + <leaf>` with a total `@crc32_update` call, which cannot overflow, deleting the very UB the solver just proved and leaving an emitted program that is UB-free. Keeping the sum form makes the emitted program byte-identical to the solved one, so a solver-found UB is guaranteed to trap under the interpreter. It costs nothing, because a UB-triggering program aborts before reaching a clean `ret` and its return-value oracle is vestigial anyway.

### Generating non-terminating programs

`--require-nonterm` generates programs that are UB-free and diverge on their input. The witness is a lasso: a stem from `^entry` to a loop header `^h`, then a cycle closing back at `^h`, carried as the finite prefix stem followed by lap. The mode implies `--require-reducible`, so every sampled back edge has a header that dominates it.

The certificate is a state fixed point. The mutable state at `^h` must be bit-identical on the first arrival and on the revisit, alongside the UB guards of the stem and the lap. A deterministic lap that begins and ends in the same state replays forever, so one finite SMT query certifies an infinite UB-free run. [symirsolve.md](./symirsolve.md) owns that query, including why its equality is value identity rather than IEEE `==`.

A random cycle almost never admits a fixed point on its own, so the cycle carries one additive correction symbol per mutable leaf it touches, `leaf = leaf + %?ntK`. The closer is uniform because `+` is the invertible operation in every domain:

| leaf type | symbol | why it closes |
|---|---|---|
| `iN` | `iN` | wrapping add is a bijection, so some value spans the gap |
| `fN` | `fN` | reaches the header value whenever the difference is representable |
| `ptr T` | `i64` | shifts the offset inside the pointee's object |

A shape that cannot close, a float too far from its header value or a pointer that changed object, comes back UNSAT and `rysmith` resamples. About 70% of seeds solve over the full type lattice, against 85% over integers alone; `--no-fp --max-ptr-depth 0` restores the higher rate.

`--max-lasso-period N` asks for a state that recurs only after `k` laps, with `k` drawn per attempt. Two constraints keep the orbit primitive: the state at lap `k` equals the header state, and the state at every earlier arrival does not. The corrections cannot supply a `k`-cycle by themselves, since `+` traps on overflow and the solver would answer `c = 0`, so a modular counter goes in the latch instead. Yield falls from about 70% at `k = 1` to 60% at `N = 4`.

The mode implies `--no-crc32`, since a diverging program has no return value and the post-solve oracle capture would hang waiting for one. Validation is a bounded replay instead: the interpreter runs the stem and the laps under a fuel bound and checks that two arrivals one orbit apart carry bit-identical state, that no UB fires, and that no `ret` is reached. `--emit-main` still applies, asserting a random `EXPECTED` the program never reaches, so a correct compilation hangs and a miscompiled one aborts. The descriptor records `outcome: diverge`, which `rylink` composes homogeneously and `rytwin` refuses.

C needs help to preserve divergence, because C11 §6.8.5p6 lets a compiler delete a side-effect-free loop, and gcc and clang both do so non-deterministically at `-O2`. `rysmith` plants an `@observe` beacon in the cycle, the identity on values and an observable `volatile` write in the C lowering ([intrinsics.md](./intrinsics.md) §12.7), so the loop survives at `-O0` and `-O2` under both compilers. WASM and Python have no forward-progress assumption and need no beacon.

### Dropping UB guards

The backends emit dynamic UB guards, and `symirc --no-ub-guards` drops them ([symirc.md](./symirc.md#omitting-ub-guards)). Those guards only ever fire on a UB path, so on a program the reify pipeline proves UB-free they are dead weight, and the tools drop them automatically rather than exposing the decision as a flag.

`rysmith` drops them whenever it is not in `--require-ub` mode, and records `has_ub` in each descriptor accordingly. `rylink` drops them only when every selected pool leaf has `has_ub: false`, since a bundle is UB-free exactly when all its leaves are; a descriptor without the field parses as `has_ub: true` and its guards are kept. `rytwin` drops them unconditionally, because a twin preserves equivalence over UB-free input and the interpreter it profiles with would have failed on any UB.

Each tool takes `--keep-ub-guards` to force them back on, which catches a program mislabeled UB-free that does trigger UB: it traps at runtime instead of misbehaving quietly.

## rylink

`rylink` reads a `rysmith` function pool, builds whole programs over it through W1 to W7, and optionally compiles and validates each one.

```
rylink [OPTIONS]
```

The full option list is `rylink --help`, declared in [src/rylink.cpp](../src/rylink.cpp).

### Pool homogeneity

Every whole program `rylink` builds has one well-defined behaviour, which requires the pool to be homogeneous: every leaf descriptor's `outcome` must agree, all `return`, all `trap`, or all `diverge`. A mixed pool has no well-defined fused behaviour, a returning caller splicing a call to a trapping or non-terminating callee, so `rylink` rejects it with an error. The common outcome becomes the program's outcome and decides what `--validate` asserts:

| Pool | Fused program | `--validate` asserts |
|---|---|---|
| `return` (default `rysmith`) | returns the entry's value; the peephole `call + (c - o)` preserves each rewritten literal | the entry returns its descriptor's solved `ret` |
| `trap` (`--require-ub`) | triggers UB, in the entry or in a spliced trapping callee | the program traps under `symiri` |
| `diverge` (`--require-nonterm`) | diverges; the entry is the unmodified diverging leaf, and its empty `ret` means no value-preserving call splices, so the callees ride along as compiler surface | the program diverges, by bounded replay on the entry's lasso header from the descriptor path |

UB guards are dropped for `return` and `diverge` bundles, both UB-free, and kept for `trap`. Under `--emit-main`, a `diverge` entry's `@main` asserts a random `i32` checksum: the entry never returns, so the check is unreachable, but the compiler still has to keep the computation.

### Reducible pools

When `--structured-lowering` is `true` or `random`, or the target is `python`, the composed program has to be reducible. Pool seeds may not be, so `rylink` discards every seed whose descriptor `reducible` flag is false before generating, a descriptor predating the flag parsing as false. If no reducible seed remains it aborts, pointing at `rysmith --require-reducible`. What survives composes into a reducible program by construction, since every inlined seed is reducible and the generated `@main` wrapper's CFG is trivial.

### Output layout

Each program lives in its own subdirectory:

```
rylink_out/
  prog_<id>_0/
    program.sir        # bundled RefractIR (header comments: ENTRY, CG, PARAMS, RETURN)
    common.h           # symirc --split-by-source artefacts, under --target c
    program.c
  prog_<id>_1/
    ...
```

The bundled `.sir` is the source of truth for every downstream consumer. Its header comments record the entry function, the call graph, the entry's solved parameter values and its expected return, which makes a bundle reproducible without the descriptor JSON.

```sh
# 1. build a pool of 200 leaf functions with descriptors
rysmith -n 200 --emit-desc -o pool/

# 2. 10 whole programs of about 4 functions each, all validated
rylink -n 10 --n-nodes 4 --validate -i pool/ -o progs/

# 3. C target with require checks kept
rylink -n 5 --target c --keep-require -i pool/ -o progs/

# 4. structured (goto-free) C over a reducible pool
rysmith -n 200 --emit-desc --require-reducible -o pool/
rylink -n 5 --target c --structured-lowering random -i pool/ -o progs/
```

## rytwin

`rytwin` takes a generated program `f1`, a `rysmith` leaf or a `rylink` whole program, and emits an equivalent `f2`: same result and same undefined-behaviour outcome for every input. A whole program is profiled from `@main` and twins are grafted into any function along the executed trace; the state capture is frame-aware, so a state is attributed to the right activation even when block labels repeat across functions.

```sh
rytwin <f1.sir> [OPTIONS]
```

The full option list is `rytwin --help`, declared in [src/rytwin.cpp](../src/rytwin.cpp). Only `f1` is passed positionally: the descriptor (`func_<id>_<i>.json`) and, when present, the state profile (`<stem>.state.json`) are read from `f1`'s directory following `rysmith`'s naming. Without a sidecar the profile is computed in-process, by interpreting `f1` on its solved input, taken from the descriptor realization or from `f1`'s own `// SOLVED:` header.

`rytwin` transforms UB-free terminating programs only, because it profiles `f1` by interpreting it. With a descriptor present, a `trap` or `diverge` input is rejected up front with a clear message, since profiling one would trap and the other would hang. Without a descriptor the profiling run is bounded by a block-step cap (`kNoDescProfileStepCap`, 3200): a terminating program finishes well inside it, a trapping one throws UB, and a non-terminating one hits the cap, all three reported as a clean failure rather than a hang.

### The twin unit

The twin unit is the maximal single-entry region rooted at an executed block: every later block the entry dominates on the executed path, up to the first block it does not, or the function's return.

The guard fires on the region's entry state, and the twin reproduces the region's net effect and jumps straight to the region exit, skipping every intermediate block and every loop iteration in between. A whole loop collapses when its header is the region entry. A straight-line run collapses to one block, which is the degenerate case of the same rule rather than a second mode.

A region is twinned only when its entry state is fully guardable and every covered block is free of non-intrinsic calls. Otherwise the window falls back to the single block at its entry.

### Region selection

`--twin-select` names a policy that assigns each eligible region a twin probability; every region is then twinned by an independent draw. A single block is the degenerate one-block region, so one rule covers both. The policies differ only in that probability.

`random`, the default, gives every region probability `--p-twin`.

`interesting` tilts the probability by how hard the region's twin is to prove equivalent. Each region scores `1000*(loop iterations collapsed) + 10*(distinct blocks) + 5*(changed leaves) + (entry fan-in)`, so collapsing a whole loop dominates. The score is normalized program-wide to `norm` in `[0,1]`, and the twin probability is `p = pTwin ^ exp((0.5 - norm) / T)` at a fixed softmax temperature `T = 0.5`. That is monotone in the score, and it is `1` at `--p-twin 1` and `0` at `--p-twin 0`, so `--p-twin` still sets the overall rate while the score biases which regions win it.

Overlapping regions resolve in trace order: the first region drawn claims its blocks, and a later region covering any claimed block is dropped. Twin bodies are synthesized only for the regions actually chosen. Selection spans the whole program, every function in the profiled trace, so on a `rylink` program `interesting` concentrates twins on the hardest regions across functions instead of scattering them uniformly.

```sh
# 1. generate a program (pointer-free, so more blocks are twin-eligible)
rysmith -n 1 --emit-desc --emit-main --max-ptr-depth 0 -o out/

# 2. emit an equivalent twin, twinning every eligible region, and self-validate
rytwin out/func_<id>_0.sir --p-twin 1.0 --validate -o out/twin.sir

# 3. differential test: compile both and compare
rytwin out/func_<id>_0.sir --p-twin 1.0 --target c --emit-main -o out/twin.sir
```

`--validate` runs `symiri` on `f1` and `f2` with the profiled input, asserts they agree and that at least one twin executed, and then spot-checks states sampled inside each guard's box, comparing the region against the twin body bit-exactly.

## Generation throughput

Three generator choices dominate the cost of a `rysmith` run: no trivially constant `lit op lit` atoms, store statements not counting toward `--n-stmts`, and indirect loads and stores. Each raises the solver work per function, and relaxing any of them would trade output quality for throughput. [.agents/notes/implementation/2026-08-05-rysmith-throughput.md](../.agents/notes/implementation/2026-08-05-rysmith-throughput.md) records what they cost together and what a per-change attribution would take.
