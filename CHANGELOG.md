# Changelog

Notable changes to RefractIR (internally SymIR), newest first. Language-level changes are defined normatively by the matching `docs/SPEC_v*.md`; entries here summarize them alongside toolchain milestones.

## [v0.2.3] - in progress

Spec: [docs/SPEC_v0.2.3.md](./docs/SPEC_v0.2.3.md), which doubles as the release roadmap and marks each feature `[Shipped]` or `[Planned]`.

### Added

- Bounded tensor execution, numerical references, Triton emission, property-directed generation, and regression tooling.

- Compositional tensor generation (`rysmith --profile tensor --tensor-generator compositional`): a typed dependency graph built forward from two or three runtime tensor inputs, with operator and shape metadata derived from the checked program rather than a second graph description, guaranteed shared intermediates and reductions with distinct consumers, and only nodes in the returned value's backward slice. Operators are placed between the required structures, so a reduction can consume a reshaped operand. Opt-in beside the existing per-stratum constructors.
- Interchangeable tensor input policies (`--tensor-input-policy ordinary|boundary|exact`): one frozen compositional program accepts sampled, boundary, or solver-selected inputs. The exact route states its output target before selection, solves a property-constrained clone of the same computation at up to 64 symbolic i32 elements inside `[-1, 2]`, the union of what the samplers propose, rejects a clone whose query-free digest differs, and records unsupported, over-budget, unsatisfiable, and undecided outcomes separately. Every accepted case records its policy, leaf domain, and stated target; the loader requires every recorded copy of the target to agree and checks it against the case's own reference output, bounds integer and floating input leaves by the recorded domain, and the generator makes the same target check before writing the case.
- Tensor parameter witnesses (`symirsolve --emit-tensor-params`, plus one record per tensor parameter in `--emit-model`) carry declared shape, element kind, and row-major logical leaves, with `f32` leaves as the exact widening of their 32-bit pattern. `--tensor-profile-version` checks a tensor program against the selected profile before solving, and `--tensor-param-domain <lo>,<hi>` bounds every integer entry-parameter leaf.
- `make test-triton` smoke case `scalar_splat_sum`, which pins a scalar argument being materialized at the declared tensor rank before a reduction.
- Tensor element arguments parse strictly: `--tensor-output-element` requires a complete coordinate and i32 value (`0=1.5`, `0=1junk`, and `4294967296=1` are rejected rather than read as `1` or `[0]`), each occurrence is read whole so a malformed argument cannot be joined with the next, and the dependency audit's unlabelled topology no longer distinguishes a whole-value copy from any other operation (recomputed: 60 v2 topologies of 253 cases, 58 v3 of 256).
- Sampling resolves distinct reported causes by a documented precedence (interrupted, then timeout, then resource limit) instead of by completion order, so the aggregate reason is the same however the attempts interleave, and a cause reported by any attempt still outranks one that reports none.
- Sampling aggregates reasons by one rule at every level: within a worker, between workers, and single-threaded, a reported cause survives the attempts that report none, so one timeout among undecided samples is not lost. `symirsolve --emit-outcome` flushes and closes the record before checking the stream and before printing the result, so what a full or unwritable device discards is reported instead of looking like success.
- Solver termination reasons survive every path that carries a result: a sampling worker that reaches no query no longer publishes its default over a reported reason, the Z3 backend maps only complete documented diagnostics (`unknown timeout configuration` and `not interrupted` stay unspecified), `symirsolve --emit-outcome` writes `{status, reason}` for machine consumers while the console text and exit codes are unchanged, the attempt record keeps the candidate and query digests and the selected path, and the tensor generator takes an injected backend so the refusal paths are tested without waiting for a real query to be undecided. The AliveSMT configuration builds and runs.
- A selection response's interval has to be whole numbers, so `-1junk` is malformed rather than -1, and a refusal keeps the symbolic size the request declared instead of a count the worker reports, while an acceptance has to agree with that size.
- A malformed domain in a selection response is refused instead of skipped, a refusal keeps the identity, interval, and size the request stated rather than the worker's claims about them, and provenance is recorded before any path can return early, so a missing worker or an unusable working directory is still attributable.
- The supervised selection decides its domain before the worker starts and consults the clock immediately before accepting: the effective interval comes from the policy default when the caller states none, so a response cannot widen it; every profile counts dependence evidence before indexing it; and a timeout, a failed worker, or a rejected response still names the graph, domain, and symbolic size it concerned.
- The supervised selection holds its boundary strictly: a response observed at or after the deadline is late however promptly the worker produced it, the worker has to exit cleanly for its answer to count, the owned process group is signalled on every terminal path so no descendant outlives the selection, the worker checks the program under the requested tensor profile, every returned leaf is verified against the domain the request stated, a refusal keeps the identity, domain, and symbolic size it refused, the counterfactual dependence check runs inside the worker rather than after the deadline, and a backend that cannot report why it stopped is given no timer of its own so the deadline this project owns is the one that fires.
- One deadline covers a supervised tensor selection: the request, worker startup, query, materialized inputs, independent replay, property check, and response validation all run inside the interval `rysmith --tensor-selection-timeout-ms` sets (5000 ms by default). The worker is a child process group the parent terminates on expiry, so no child work outlives a returned timeout; expiry reports `timeout_source=selection_deadline` with the stage the worker last reported, cleanup latency is measured separately, and a late or partial result is never an accepted case. The parent validates the response against the program it sent (identity digest, parameter identity and lengths, path, output length, and the stated property) without replaying again. `make test-solver-outcomes` covers the deadline cases through a controllable worker fixture.
- Solver termination reasons are typed and preserved: `smt::UnknownReason` (unspecified, timeout, resource limit, interrupted) carries a backend's own report through symbolic execution, the Z3 backend maps the diagnostic it receives, Bitwuzla keeps the unspecified default because its pinned API reports no reason, an undecided tensor input selection with a reported timeout becomes `timeout` with `timeout_source=solver`, and every unsuccessful attempt records its seed, property, policy, domain, symbolic size, status, and reason. `make test-solver-outcomes` runs the classification tests against a scripted backend.
- Tensor element values convert from the sign and base the argument states, so a plus sign before a base prefix reads as its value (`0=+0x10` is 16, `0=+0b11` is 3) and `0=+0x80000000` is rejected as outside i32 instead of becoming zero. Element occurrences are read from cxxopts' sequential argument list, which keeps a value belonging to another option and anything after `--` out of the element set.
- Tracked tensor campaign runner and baseline fixture: `test/tensor/campaigns/run_matched_tensor_evaluation.py` and `v1-baseline.json` are the two files the committed evaluation and coverage tests load, so the tracked tree can run them. `libsymir.a` now carries the tensor interpreter units, and `make test-library` links and runs a small consumer that interprets a scalar, an exact tensor, and a numerical tensor program against the archive.
- Tensor output-element properties (`rysmith --tensor-output-element <coordinate>,...=<i32>`, repeatable): a tensor-v2 i32 case can require named output elements instead of one scalar target. The request is canonicalized row-major, recorded as a typed `property.constraint` in the case manifest, checked against the case's own reference output when the case is written and again when it is loaded, and rejected when it names a repeated or out-of-bounds coordinate or a complete target that skips a coordinate. Elements are compared individually, so a permutation with the same total fails.
- Tensor dependency audit (`test/lib/tensor_graph_audit.py`, `make test-tensor-graph-audit`): a deterministic 256-seed-per-profile measurement that separates source bytes, ordered dependency topology, and the operator-labelled graph, and reports dimensions, literals, and axes separately. The compositional builder now opens each graph with bounded operations on two separate runtime inputs that later join, a structure it previously could not reach.

- Python compilation target (`symirc --target python`): genuine `while`/`if` control flow, a boxed pointer and aggregate memory model with runtime UB traps, lane-list vectors, and symbol providers through module globals. Accepts reducible CFGs only.
- CFG structuring analyses: dominator trees, reducibility check, loop forest, control-tree builder, and structured lowering with `while`/`do-while` peepholes and header-test loop rotation ([docs/reducibility.md](./docs/reducibility.md)).
- `symirc --structured-lowering`: reconstructed `while`/`do-while`/`if` on the C target, and `block`/`loop`/`if` on the WASM target, where the native multi-level `br` consumes the unlowered control tree and needs no guard flags. The `$__pc`/`br_table` dispatch loop stays the WASM default and still accepts irreducible CFGs.
- WASM SIMD-128 vector lowering: vector locals live in native `v128` registers by default, with shapes wider than 16 bytes split across registers.
- `symirc --vec-lowering` strategy families: C `vecext|scalars|array|structscalars|structarray`, Python all but `vecext`, WASM `vecext|array|scalars`; the chosen strategy is stamped into the emitted module.
- `symirc --no-ub-guards`: omit the backends' dynamic UB guards on all three targets. Sound only for UB-free programs, where the guards never fire; value semantics are unchanged. The reify tools drive it from the descriptor's `has_ub` field, and `--keep-ub-guards` forces it back on.
- WASM checksum intrinsics `@crc32_update` (table-free LFSR loop) and `@check_chksum`, with no host imports, so every shipped intrinsic lowers on every compiled target.
- Horizontal vector reductions `@reduce_add`/`min`/`max`/`and`/`or`/`xor` (spec §12.4): a sequential per-step-UB fold from `<N> T` to `T`, order-independent for min/max via the `@fmin`/`@fmax` tie-break, across the interpreter, the solver and all three targets. `@reduce_mul` is intentionally absent, being nonlinear.
- Non-terminating program generation (`symirsolve --require-nonterm`, `rysmith --require-nonterm`): UB-free diverging functions from a lasso witness whose header state is a fixed point, closed by a uniform `leaf = leaf + %?ntK` correction over integer, float and pointer leaves. `rylink` fuses a homogeneous pool into a whole program of the same outcome, and `rytwin` refuses one. Adds the descriptor `outcome` field generalizing `has_ub`, the `@observe` beacon intrinsic, and `symiri --max-bbl-steps`.
- UB-directed generation: `symirsolve --require-ub` and `rysmith --require-ub` solve for symbol values that trigger UB on the chosen path; `rysmith --no-crc32` skips the checksum oracle.
- `rytwin` (new tool): emits a semantically equivalent variant of a generated program by grafting guarded twin regions, with no SMT solver involved. The twin unit is a whole dominance region, collapsing straight-line runs and entire loops into one guarded block that jumps to the region exit, and its body is the region's own executed trace. An interval pass classifies each guard leaf free, ranged or pinned, and `--validate` spot-checks states sampled inside the resulting box.
- Anti-optimization engine ([include/reify/antiopt.hpp](./include/reify/antiopt.hpp)): identities applied in the opposite direction to a compiler's, re-checked over the states a guard admits and rolled back otherwise, so a rule that can trap is kept exactly where the box proves it cannot. Used by `rylink` on bundled programs and by `rytwin` on twin bodies.
- `rytwin --twin-select interesting`: a selection policy that softmax-tilts each region's twin probability by how hard its twin is to prove equivalent, concentrating twins on the hardest regions.
- Per-lane vector symbol binding, `symiri --sym '%?v=1,2,3,4'`.
- Reify pipeline support for the Python target, structured lowering with per-program strategy sweeps, and reducible CFG generation.
- `rysmith --n-examples`: more than one input/output example per concretized `.sir`.
- `make install` target.
- Compiler-behavior observation for the Triton tracks: loop retention is decided per required stage, so a loop that survives only in TTIR cannot satisfy a later-stage requirement; layout conversions are matched across pass boundaries by resolved encoding definitions, source line, and enclosing region, so renamed aliases and renumbered values stay one fragment and a conversion that moves into a loop is reported as moved at equal counts; rematerialization is confirmed only when a recomputed pure value takes over a use the original no longer has, with untransferred duplicates reported as unattributed work; and a named target pass reports its own transition, or `unavailable` when the dump holds no measurable one. Worker compilations can capture the raw pass snapshots (`REFRACTIR_TRITON_PASS_DUMP=1`) into one file per compilation.
- Qualified GPU compiler identity: the Gluon runners accept the pinned development revision (`3.9.0` at `32f06df4`), recorded with its environment probe and a compile-and-run qualification, instead of the version string alone. `make test-tensor-runner` now also runs the previously unreferenced worker timing tests.
- Configurable bounded loop regions (`tensor_graph_region_export --family`): beside the original shared-producer-then-folds skeleton, the native builder emits orthogonal axis reductions with separate broadcasts, a reduction feeding bounded arithmetic, cross-state roots that both read both states, and a shared producer with consumers at different depths. The exporter's volume cap now comes from the tensor profile (512 elements) instead of a hardcoded 64, and `--max-operators` exposes the node budget. `test/lib/tensor_region_generation.py` states a region as an explicit configuration (seed, shape, bounds, family, arithmetic, budget), reproduces it from the seed, and replays it; generated cases record that configuration, and each admitted long-axis configuration carries a full exact certificate while an overflowing recurrence is rejected.
- Rank-two logical tails in the compositional tensor generator (v3): the working-shape policy admits the six published tail pairs (31x8, 33x8, 8x31, 8x33, 63x4, 4x65) whose padded tile stays inside the 512-element profile volume, so a reduction, broadcast, or join meets a masked axis beside a full one. `make test-tensor-tail-shapes` checks reachability, next-power-of-two padding, complete logical replay, and the masked Triton tile the backend emits.
- Guarded physical storage (`test/lib/tensor_layout.py`): an allocation envelope records prefix and suffix guards around the storage span, staging fills guards and internal gaps with a chosen finite padding pattern, and a changed slot is named by role (`prefix-guard-N`, `suffix-guard-N`, `gap-N`, `input-N`) so an unwanted write is distinguishable from a logical value. Two padding patterns over identical logical input leave the logical output unchanged while remaining visible outside the logical mapping, and an intentional in-allocation control difference is classified as a control variant rather than as a compiler finding.
- Explicit Gluon layout plans (`test/lib/tensor_gluon.py`, `render_gluon_kernel(..., plan=...)`): a plan states the load, compute, reduction, loop-state, and store roles plus a layout per consumer view, and the emitter converts into a fresh view instead of rebinding the producer, so one consumer's layout never changes what another consumer reads. Views asking for a layout a value already has are shared rather than recomputed, and every layout the body names is a constexpr parameter of the generated kernel. `effective_mapping` reports which warp, lane, and register owns each element under a blocked layout, and `classify_layout_pair` separates an identical plan, an equivalent mapping on a small tensor, a lane change, and a warp change, so an equivalent parameter tuple is not counted as cross-lane work.

- First device evidence for the three compiler-behavior tracks (`test/tensor/campaigns/results/2026-10-11-targeted-compiler-behavior.md`): four configured long-axis families execute against their complete exact references with 8-15 layout conversions removed by `TritonGPURemoveLayoutConversions` and 0-10 PTX warp shuffles; fifteen v3 tail cases (rank-one `17`/`31`/`33` and rank-two `[8,31]`/`[63,4]`) execute with the logical extent surviving into the kernel predicate and nonzero stride offsets; and four Gluon plan families execute with surviving conversions that lower to 2-33 shared-memory transfers. The counted loops in these small cases are unrolled before the device, so no retained-device-loop or rematerialization claim is made.

- Loop retention is decided after TTGIR, not inside it: loop detection now recognises a lowered loop by its backedge (PTX `bra`/`bra.uni`/predicated forms and LLVM IR `br label`), so a stage that lowered `scf.for` into branches is no longer reported as loop-free. With that correction the counted loop survives TTIR and TTGIR everywhere, is unrolled by the LLVM pipeline for small constant trip counts, and reaches PTX with a real backedge at the 512-element profile volume (two `mixed-depth` `[32, 16]` four-trip cases, built through the exact builder and executed against their complete references, are recorded in `test/tensor/campaigns/results/`).

- Guarded allocations reach the device: a guarded tensor is handed to the kernel as a view that starts at the guard prefix, so the case's recorded addressing stays correct while the worker still validates the whole allocation. Three admitted cases (two padding patterns and an unguarded control) pass on the V100 against their complete references.
- Guarded cases reach the device route: the worker stages inputs inside an allocated envelope, fills guards and internal gaps with the case's padding, allocates the output the same way, compares the whole input allocation, rejects any change to a guard, gap, or logical slot by name, gathers the logical result, and records what it staged (`worker-staging.json`). The exporter writes `execution.guard_envelope` and `execution.padding_by_dtype` when a case asks for guards, and an explicit zero envelope stages byte-identical bytes to no envelope.

- All six Gluon layout plan families ran through the runner on the V100 at one and four warps: twelve runs, every one matching the complete 64-word reference, with 38 of 114 requested conversions surviving, 76 eliminated, none folded or unattributed. No pass adds a conversion; every survivor lowers in `ConvertTritonGPUToLLVM` and is a cross-lane change at one warp and a cross-warp change at four. The sweep exposed two plan-side gaps, now fixed: a family whose views were the compute layout by construction, and a declared view that no consumer queried (emission now refuses it).

- The LLVM backedge detector matches printed optimized IR: numeric labels count, and every `label %x` operand of a branch is read, so a bottom-tested loop whose exit block is listed first is no longer missed. The Track A matrix (72 configurations, 144 device runs) agrees with the PTX backedge in every run under that reading.

- Guarded storage works on logical tails: slot classification carries the tensor profile through, so a guarded v3 tail no longer fails address validation, and the worker saves the observed bytes before judging storage, checks only nonlogical slots for guard writes, and leaves a wrong logical value to the numerical comparator. Six guarded tail cases (three shapes, two padding patterns) pass on the V100 with identical logical output per shape.

- Physical layouts and launch settings were varied independently on one admitted retained-loop region: seven input/output layout combinations (row-major, column-major, padded with offsets) and four warp counts all match the complete reference, with the TTGIR loop present in every run and a PTX loop in all but the eight-warp and padded/padded variants.

### Changed

- `rysmith --n-inits` renamed to `--n-concretes`; the old flag no longer parses.
- Internal modularization: the interpreter, the solver and the C/WASM backends split into cohesive translation units with extracted collaborators (`TypeLayout`, `Memory`, `Provenance`) and decomposed visitor dispatches; backend files renamed to a target-prefix convention; C vector lowering made target-specific. The shared mechanism the reify tools had accumulated moved to the components that own it: the parse-and-check pipeline, the type and access-path queries, the byte-offset walk and pointer provenance, AST cloning and construction, the emit driver, declaration identity, the solved-program header, and the state-set analysis.

### Fixed

- Tensor initializer lowering: a scalar initializer materializes at the declared tensor shape for every profile before a shape-sensitive operator, and a tensor initializer copies its value. The previous Triton emission either lost the rank or passed a block tensor as `tl.full`'s scalar fill value, which Triton rejects. `docs/SPEC_TENSOR_v0.1.md` §3 states the rule.
- Linking a library with `-I` merged its declarations by name. An intrinsic overload the primary also declared under a different signature was dropped, so the library body stopped typechecking; a struct the primary declared differently displaced the library's, and when the two agreed on field types but not their order the result still typechecked and read fields at the wrong offsets. Both are keyed by the declaration's identity now, and a struct conflict is rejected.
- WASM vector call boundary: arguments were passed as a garbage byte-load rather than an address, and vector returns emitted invalid WASM. Both use a defined memory ABI (caller-owned spill slots plus a hidden sret parameter) preserving by-value semantics.
- Object extents in the solver were measured in scalar leaves rather than packed bytes, a scale that agrees with the interpreter only when every scalar in an object has the same width. Mixed-width structs both hid real out-of-bounds pointer arithmetic and rejected valid programs, and the `--emit-main` checksum oracle decoded exit-time pointer offsets on a third differently-scaled copy of the model, with the Python backend's leaf-slot memory a fourth. All four measure with `TypeUtils::packedSizeof`, and Python's buffer is byte-indexed with a `_PAD` sentinel that traps on read of a wider leaf's interior bytes.
- Float literals reached the solver through a decimal rendering, which dropped the sign of `-0.0` and truncated every value to six fraction digits. All four construction sites and both solver backends build FP constants from the `double` itself, bit-exactly.
- `rylink` composed programs whose cost was exponential in the call graph's depth: an edge could be spliced at several sites on the caller's executed path, and the per-match acceptance coin was tuned to keep the expected count in [1, 2), at or above the branching factor from which a run never terminates in practice. An edge now takes at most one executed site, chosen with probability `--p-onpath-call`, and is realized off-path regardless, so the emitted call graph is the sampled DAG whatever the coin says. A 64-node bundle went from ~10⁹ calls (hanging `--validate`) to a few thousand.
- New UB rule 27 (spec §7.5): dereferencing a pointer to a returned activation's local. The solver tagged a pointer by the local's name alone, so a dead frame's pointer resolved against any live cell that shared the name, a load through it read the wrong object rather than being ruled out; tags now name the activation too. `symirc` does not enforce the rule. The interpreter kept every activation's provenance objects for the life of the run, so a dead frame's locals stayed readable and, since provenance lookup is a linear scan, a program's cost grew quadratically in its own call count. An activation now releases its objects on return, keeping only the provenance ids reserved.
- Rule 3 (read of `undef`) enforced on a load through a pointer to an uninitialized cell, in the interpreter and the solver.
- Symbolic floating-point inputs constrained finite by the solver, per the finite-only FP domain (spec §2.9).
- Literal bit-width inference propagated, so UB overflow detection sees the resolved width.
- C backend: inline `cmp`-atom masks and vector symbol initialization.
- `symirc --no-ub-guards` no longer affects `@check_chksum` and `require`.

## [v0.2.2] - 2026-06-29

Spec: [docs/SPEC_v0.2.2.md](./docs/SPEC_v0.2.2.md).

### Added: language

- Function calls: the `call` atom with left-to-right argument evaluation and interprocedural execution, threading path conditions, store and memory through callees, with callee UB pruning the calling path. Recursion, indirect calls and variadics are rejected.
- External declarations (`decl`) in two mutually exclusive forms: contract (`pre`/`post`/`ret` clauses as the callee's specification for solver reasoning, with pointer-argument memory havoc) and link (signature resolved to a body in another `.sir` file via `-I`).
- Intrinsics (`intrinsic`): toolchain-defined built-ins with fixed interpreter, SMT and backend lowerings, shipping the §12 baseline and the full P0 tier (integer extras, bit-manipulation, the overflow-aware family, the FP IEEE family) with per-width overload resolution ([docs/intrinsics.md](./docs/intrinsics.md)).
- Checksum primitives `@crc32_update` and `@check_chksum` for the reify pipeline's opaque return-value oracle, lowering to C at this point and to WASM in v0.2.3.
- Signed `i1 = {0, -1}` value convention, and strict signed-range literal checking with no silent narrowing.

### Added: toolchain

- `rylink` (new tool): whole-program generator composing `rysmith` leaf functions, with per-artifact output, `--split-by-source`, and differential cross-validation batches.
- `-I` link resolution across `symiri` / `symirc` / `symirsolve`, the `test/lib/std` stdlib, entry-point positional arguments, the bit-exact `SOLVED`/`PARAMS`/`RETURN` headers, and `--emit-main`.
- Solver: contract-form `decl` expansion, random-path sampling for branchy callees, and contract memory havoc for direct pointer arguments.
- Reify: intrinsic generation, the opaque checksum rewrite, and a broad set of generator controls (`--min/max-atoms`, `--large-coef`, `--off-path-multiplier`, noinline and noclone probabilities).

## [v0.2.1] - 2026-05-27

Spec: [docs/SPEC_v0.2.1.md](./docs/SPEC_v0.2.1.md).

### Added: language

- SIMD vector types `<N> T`: lane-wise arithmetic, lane access by subscript, whole-vector copy, and per-lane independent vector symbols. Vectors are pure value types and are not addressable.
- Reified comparisons (`cmp <relop>`): `i1` for scalars and `<N> i1` masks for vectors, with mask-based `select` for per-lane blends.
- Aggregate pointers `ptr [N] T` and `ptr @S` with `ptrindex` / `ptrfield` navigation, packed struct layout, UB rules 14 to 19 for provenance and typed-access mismatch, and the vector UB rules.
- Atom-form initializers for non-aggregate locals, and the finalized floating-point value model.

### Added: toolchain

- C backend vector-lowering strategies (`vecext`, `array`, `scalars`, `structarray`, `structscalars`), full v0.2.1 support in the WASM backend, and per-lane vector SMT encoding in the solver.
- Reify: vector and aggregate-pointer generation with a random per-program vec-lowering strategy.

## [v0.2.0] - 2026-05-23

Spec: [docs/SPEC_v0.2.0.md](./docs/SPEC_v0.2.0.md).

### Added: language

- Pointers: the `ptr T` type, `addr` requiring a `let mut` root, `load` / `store`, context-typed `null`, pointer arithmetic (`ptr T ± iN` and `ptr T - ptr T` element distance), and strict provenance UB rules including rule 15 for struct-field provenance.
- Float `%` redefined from IEEE remainder to C `fmod` truncated-quotient semantics, and `shl` result overflow classified as signed-overflow UB.

### Added: toolchain

- `rysmith` (new tool): the C++ reify pipeline generating random leaf functions with pointer support, AoS/SoA type diversity, and `--target c/wasm` compilation of concrete output.
- Solver pointer support through a tagged BV64 encoding, typed process exit codes with `EXPECT: FAIL:<subtype>` test-framework support, and `symirc --no-require`.

## [v0.1.0] - 2026-05-18

Spec: [docs/SPEC_v0.1.0.md](./docs/SPEC_v0.1.0.md), the draft v0 spec renamed under semantic versioning. First complete implementation of the language and toolchain.

### Added

- Language core: a non-SSA CFG-based IR with mutable locals (`let mut`), explicit symbols, flat left-to-right expressions, lazy `select`, strict UB, `as` casts, bitwise operators, hex/octal/binary literals, multidimensional arrays with brace initialization, and finite-only `f32`/`f64` floating-point.
- `symiri` reference interpreter with strict UB checks and execution tracing.
- `symirc` compiler with C and WebAssembly (WAT) backends.
- `symirsolve` SMT concretizer: path-based symbolic execution, an abstract solver interface with Bitwuzla and AliveSMT (Z3) backends, and random path sampling with multi-threading.
- Frontend stack: lexer, recursive-descent parser, CFG builder, BV-aware typechecker, semantic checker with definite-initialization analysis, pass manager and dataflow framework, and clang-style diagnostics.
- Examples (ciphers, sorts, a robot navigator), the `.sir` VS Code extension, and the automated test suite.

## [v0.0.1] - 2026-01-23

Repository bootstrap: the draft v0 language specification (`SPEC_v0.md`, later `SPEC_v0.1.0.md`), design documents for `symirc` / `symiri` / `symirsolve`, and the project guides. No implementation.
