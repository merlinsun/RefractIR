# Generation without an expected output: a generator aimed at the relations

Prepared 2026-11-03, branch `free-form-regions` at `b890ea73`. Follows the seed-export plan (`.planning/2026-11-01-seed-export/`, decision "Generation without an expected output") and the calibration spike (`output/calibration-spike-2026-11-02/REPORT.md`). Its kernels leave RefractIR through the seed-export exporter (E1–E4) and are judged by layout-mutator's relation stage (`docs/superpowers/plans/2026-11-01-exact-relations.md`). Nothing here is implemented.

## Part I. Why and what

### Decision

Reification stays in RefractIR as a generator. It no longer certifies an expected output: the relations judge every value, so a generated kernel needs only to be valid on its chosen input. The split is by type:

- **Integers decide the path and the addresses.** Loop bounds and steps, trip counts, branch selectors, extents, offsets and strides are scalars. The generator picks a target path and the solver chooses scalars that take it, steered toward corner paths. This is the part of reification that survives.
- **Floats only flow through the data.** No float value is constrained, bounded or predicted. What a run produces decides what is compared (run, then classify): the sidecar's class per output, the relation guards, and the interpreter admission.
- **Validity by construction.** Every load and store is masked by its full in-range predicate, every loop's trip count is bounded by its declared scalar ranges, and index arithmetic stays inside i32. Float overflow, NaN and division by zero are not reasons to refuse: SIR calls them UB, Triton defines them, and the relation guards skip such lanes.
- **Structured control only**, as Triton's frontend requires: counted `for` loops with runtime bounds and any nonzero literal step, and `if` diamonds. No early exits, no irreducible flow.

P0 keeps two roles. It checks the integer side: a solved path must be the path P0 takes. Where P0 is defined on the floats as well, its output stays an absolute check (seed-export E3); where it is not, the seed is still exported.

### What the spike says to aim at

The calibration spike (two runs on the V100, 2026-11-02) showed which relation sees which kind of fault. The generator should produce the kernel features each relation needs.

| Relation | What it caught in the spike | What the generator must produce |
|---|---|---|
| Launch, tile size | a tile fault at one `BLOCK` (`tail_mask`, `drop_last_column`) | grid-tiled kernels with a `BLOCK` constexpr and a runtime extent that leaves a partial last tile |
| Code, loop unrolling | F2 at `0861b90`, on three empty ranges | runtime loop bounds, empty ranges, negative steps |
| Values (uniform factors) | a comparison constant folded wrong (`threshold`) | literals mixed with scaled data: comparison thresholds, `max(x, 0)`, multiplicative and additive constants |
| Values with sign (factor −1) | f16 rounding toward zero for negatives (`rtz_negative`) | conversions: narrowing inside the body and half-precision storage at the boundary |
| Positions, per-element factors | a misrouted lane (`lane_swap`) | pointwise regions with a parallel axis |

### What exists today

- **Builder** (`include/reify/tensor_free_form.hpp`, `src/reify/tensor_free_form.cpp`). A typed DAG over a single element type (`validateRequest`, cpp:1382), grown from runtime inputs only: there are no literal leaves. Actions (cpp:567–825): add, sub (both orders), mul (f32, at most `maxMultiplications`), and behind feature flags min, max, where (lt, le, eq), cumsum, narrow f16/bf16 (an f32-to-f32 round trip), dot (power-of-two extents of at least 16), sum, rmin, rmax, transpose, expand, broadcast. Limits (hpp:97–103): 12 nodes, rank 2, 512 elements, 2 multiplications.
- **Assembler** (`assembleCountedLoop`, cpp:2034). One counted loop `%i` from 0 by +1 to a literal or to a trip parameter assumed in [0, `tripCount`]; an optional branch parameter choosing between keeping and rotating the states; an optional outer loop, capped at eight body executions in all; window productions that read a backing tensor at row `%i` with an active-row mask (a partial tile within one program).
- **Profile** (`src/analysis/tensor.cpp`). v3 admits the builder's operations plus window (`v3SupportsTensorOperation`, :359) and refuses div, exp, rem and softmax. Loops (`analyzeTensorLoops`, :877–1011) need a `<` header, a literal start, a +1 latch, one to eight trips, and at most eight body executions in all; a parameter bound must be assumed with a range starting at or above the start.
- **Emitter** (`src/backend/triton_backend.cpp`). One program, no `program_id`, whole-tensor tiles with v3 padding masks (`emitOffsets`, :410). Loops become `for i in tl.range(start, bound[, hint])` (:1158), with the bound a literal, a runtime argument under `do_not_specialize`, or a scalar loaded from a pointer; after the loop the induction variable is set to the bound (:1189), which is right only for step +1. Hints `loop_unroll_factor`, `num_stages` and `flatten` exist (:170). Div, rem, exp and softmax already lower (:843–924). Buffer layouts carry literal strides and an element offset.
- **Solver.** `symirsolve --path <labels>` solves a scalar program for a chosen path (`docs/symirsolve.md`).

### P0-driven restrictions and which task removes each

| Restriction | Where | Why it existed | Removed by |
|---|---|---|---|
| every value within 2^24 for all inputs | `--exact-input-bound`, `FreeFormExactContract` (hpp:422–453) | P0 exact and order-free | G9 |
| at most two multiplications | `maxMultiplications` (hpp:102; cpp:618, 707) | magnitude growth under that bound | G9 |
| inputs drawn from [1, 2] or {−1, 0, 1, 2} | input policies (`docs/reify.md`) | exactness | G9 |
| no div, exp or rem | `v3SupportsTensorOperation` (tensor.cpp:359) | P0 cannot predict the GPU's approximate last bit | G5 |
| loops start at a literal, step +1, run 1–8 times, 8 executions in all; no empty literal range | `analyzeTensorLoops` (tensor.cpp:877–1011), profile (tensor.cpp:24–26), assembler (cpp:2037–2042) | bounded unrolling for the solver and the magnitude bound | G1, G2 |
| at most 12 nodes | `maxNodes` (hpp:98) | certification cost | G9 |
| at most 512 elements, one program | profile (tensor.cpp:26), `maxElements` (hpp:100), emitter | interpreter and solver cost | G6 |

Three further gaps are not P0's doing: the builder has no literals (G4), strides are compiled in as literals (G7), and no policy favours pointwise regions (G8). The v3 profile and the certification route stay as they are; everything here lives in a new profile used only by this route (G0).

## Part II. Contract additions

The relations sidecar (`relations.json`, schema `layout-mutator.relations-sidecar.v1`, both plans) gains three optional fields. They are additive, unknown fields are already ignored, and G10 updates the contract text in the exact-relations plan (section 1) and the seed-export plan (Part II) before any seed carries them.

```json
{
  "launch": {
    "tiles":   { "BLOCK": { "extent_arg": "N", "grid_axis": 0 } },
    "strides": { "<arg>": ["<stride arg>", "<stride arg>"] },
    "offsets": { "<arg>": "<offset arg>" }
  },
  "literals": [ { "value": 0.0009765625, "role": "where_threshold", "lifted_to": "c0" | null } ],
  "path":     { "target": "empty_negative_step", "scalars": { "lo": -3, "hi": -1 }, "step": -2 }
}
```

- `launch.tiles` lets the launch relation recompute the grid exactly (`cdiv(N, BLOCK')`) instead of guessing it, which is where the spike's admission bug lived.
- `launch.strides` and `launch.offsets` name the scalar arguments that carry a buffer's layout, so the layout and alignment variants know what to change.
- `literals` records each constant and, for a lifted twin, the parameter that replaced it, so the values relation scales it.
- `path.target` names the corner path the scalars were solved for, so reports can count findings per target.

## Part III. Work packages [LOCAL unless marked]

Each task is test-first under the repository's rules (five failing tests run one by one, then green, then the smallest edge test); estimates are working days. Order: G0–G3 first (the schedule relations found F2 and both tile faults), then G4–G8 (G6 before G8, whose requirement reuses G6's analysis), then G9–G10, then the remote runbooks.

### G0. A generation profile (1 day)

`tensor-v5` in `src/analysis/tensor.cpp` and `include/analysis/tensor.hpp`: v3's operations plus div, exp and rem; rank 2; elements up to 65,536; loop executions up to 64 in all; consumers Reference and Triton only, so `ExactSolver` refuses it with a clear message. The spec's tensor section and `docs/reify.md` state the profile. **Removes:** nothing yet; it is the place the later tasks relax. Tests: a v3-legal program is v5-legal; div and exp pass v5 and still fail v3; the exact solver refuses v5; 65 loop executions fail.

### G1. Counted loops with runtime bounds and any step (4 days)

Profile and emitter accept `for i in range(lo, hi, s)`: a header `%i < %hi` for a positive literal step or `%i > %hi` for a negative one, a latch `%i = %i + s` with `s` a nonzero literal, and `lo`, `hi` each a literal or an i32 parameter with an assumed range. Empty ranges are legal. The trip cap is computed from the ranges. The emitter writes `tl.range(lo, hi, s)` with runtime bounds passed by value under `do_not_specialize`, which is F2's shape, and sets the induction variable after the loop to `lo + k·s` with `k` the trip count, not to the bound. **Removes:** the literal start, the +1 step, the one-trip minimum and the eight-execution cap (G0's budget). Tests: F2's four bound sets `(−3, −1, −2)`, `(0, 2, −2)`, `(2, 0, 2)`, `(0, 10, 1)` give P0 trip counts 0, 0, 0, 10 and the emitted text names `tl.range(lo, hi, -2)`; the exit value matches Python's `range` for a table of 30 (lo, hi, s) cases; a zero step is refused; a range whose cap exceeds the budget is refused; v3 still refuses a negative step.

### G2. The assembler on general loops (3 days)

`FreeFormLoop` gains `lo`, `hi` and `step` (each a literal or a named parameter with a range) in place of `tripCount`/`tripParameter`, with the old fields kept as shorthand. Nested loops take their own bounds; the outer cap follows G0's budget. Windows index rows by the induction variable at any sign, so the window mask gains a lower-bound term (`row ≥ 0`) beside `row < rows` and `linear < active`; any integer is then a valid address and out-of-range rows read the fill. **Removes:** the assembler's trip and outer caps (cpp:2037–2042). Tests: a negative-step loop over a window reads rows in reverse; a row index of −1 reads the fill; nested loops with runtime bounds assemble and pass G0; the canonical two-state loop is byte-identical to today's.

### G3. Steering the integers (4 days)

`src/reify/tensor_path_steering.cpp` with a test driver: from an assembled kernel, extract the scalar skeleton (the CFG, the integer locals, the assumes; tensor statements removed; the integer parameters turned into symbols such as `%?lo`), enumerate target paths, and solve each with `symirsolve --path`. Targets: each loop at 0 trips (both empty forms, `lo ≥ hi` with a positive step and `lo ≤ hi` with a negative one), at 1 trip, at its cap; each branch arm; a window whose rows straddle the backing tensor's start or end; for tiled kernels (G6), `N mod BLOCK` at 1 and `BLOCK − 1`, and `N < BLOCK`. Each solved assignment is replayed in P0 (`symiri --dump-trace` lists the executed blocks), and a path that differs from the target is a defect, not a refusal. **Removes:** the need for P0 to certify values; the solver now only chooses paths. Tests: each target is reached in P0; F2's empty negative-step range is produced from the target alone; an unreachable target (a cap below one trip) is reported as unsatisfiable; the skeleton of a kernel without loops has one path.

### G4. Literals in the builder (3 days)

Literal leaves enter the pool behind a feature flag: multiplicative constants (2.0, 0.5, 3.0, −1.0), additive constants, thresholds in `where` against a value, and `max`/`min` against 0 (relu and its kind). A power-of-two or zero literal keeps the values relation applicable; any other literal joins the scaled data, so the exporter can also write a twin seed with that literal lifted to a parameter (`literals[].lifted_to`). **Removes:** nothing P0-driven; it closes the gap the `threshold` fault exposed. Tests: `where(x > 2^-10, x, 0)` and `max(x, 0)` build and emit; the sidecar marks an additive literal's output as not scalable unless lifted; a lifted twin's sidecar scales the new parameter; with the flag off, the frozen corpus is unchanged.

### G5. Division, exp and remainder in the builder (2 days)

Builder actions for `div` (by a value or a literal), `exp` and `rem` under G0's profile; no guard against a zero divisor, because a float division by zero is defined on the GPU and the relation guards skip non-finite lanes. **Removes:** the v3 exclusion of div, exp and rem. Tests: each operation builds, passes v5 and emits; a body with `div` exports with P0 `expected` null when P0 reports UB, and with its other outputs still classified; with the flags off, the corpus is unchanged.

### G6. Grid-tiled emission (4 days)

`parallelAxes(region)`: the axes that no live reduction, transpose, dot or scan mixes (G8's requirement uses the same analysis). An emitter mode for kernels whose leading axis is parallel: `pid = tl.program_id(0)`, a `BLOCK` constexpr, the axis extent `N` passed as a runtime scalar, every access on that axis masked by `offs < N`, and grid `cdiv(N, BLOCK)`. Tensors without that axis are loaded whole by every program; windows are refused in this mode for now. The axis extent is free of the 512 cap (G0's 65,536), and `N` is chosen by G3 to leave a partial last tile. The exporter records `launch.tiles`. **Removes:** the one-program, 512-element limit for this route. Tests: a pointwise body emits `program_id`, the mask and the constexpr; a body with a reduction across the leading axis is refused for this mode; `N = 1000, BLOCK = 256` gives grid 4 in the launch record; the sidecar's `launch.tiles` names `N` and axis 0.

### G7. Strides and offsets as arguments (2 days)

An emitter option passes each buffer's strides and element offset as runtime scalars instead of literals; the generator chooses layouts (padded rows, a transposed view, an offset into a larger buffer, a base offset of one element) and G3 may steer offsets toward the buffer's end. The exporter records `launch.strides` and `launch.offsets`. **Removes:** nothing P0-driven; it lets the layout and alignment variants apply to generated kernels. Tests: a transposed layout emits stride arguments and P0 is unchanged; an offset of one element is recorded; the default literal-stride emission is byte-identical to today's.

### G8. Pointwise regions and boundary dtypes (3 days)

A `Pointwise` policy biases the builder toward elementwise operations, and a `parallelAxis` requirement keeps one axis in G6's `parallelAxes`, so G6 can tile it and the sidecar's `permutable_axes` and `scale_axes` cover it. Conversions at the boundary: an input may be loaded as f16 or bf16 and widened, and an output that is the result of `narrow f16` or `narrow bf16` may be stored in that type, with the SIR program unchanged: the exporter rounds such an input to the half type before P0 sees it, and a stored output is a narrowing result, so every value crossing the boundary is exactly representable. Narrowing inside the body gets a higher weight under this policy. **Removes:** nothing P0-driven. Tests: under `Pointwise`, every live node keeps axis 0 parallel; a refused `parallelAxis` request names the operation; an f16 output's emitted store has `.to(tl.float16)` and the launch record's dtype is `float16`; P0's output, narrowed, equals the stored bytes.

### G9. The generation route end to end (2 days)

A request preset for this route: profile v5; no exact-input bound or contract; multiplications and nodes up to the compiler's limit (default 32 nodes); inputs uniform in [−1, 1] with the corner recipes I4–I8; G3's targets as the inputs' scalar part. `export_seed_block.py` (seed-export E4) gains a `--generation` block type that uses it. **Removes:** the 2^24 bound, the two-multiplication cap, the small-integer inputs and the 12-node cap. Tests: a block of 8 seeds exports 8 directories with sidecars and `path.target`; a builder refusal is logged, not exported; nothing in the block depends on `freeFormWithinExactContract`.

### G10. Contract and exporter updates (2 days)

Implement Part II in `test/lib/tensor_relations_sidecar.py` and `test/lib/tensor_seed_export.py` (seed-export E1–E2), and update the contract text in both plans in the same change. Tests: each new field round-trips through layout-mutator's seed loader; a seed without the new fields loads as before.

### R1. Emit-and-compile check [REMOTE-HOST, then REMOTE-GPU, one GPU, 30 minutes]

Runbook `.planning/2026-11-03-generation-without-expected-output/R1-RUNBOOK.md`, written when G0–G9 are done, for the host agent. Environment as in the spike: the `great_rhodes` container on the V100 server, `TRITON_PTXAS_PATH` set to the CUDA 12.9 ptxas, Triton `0861b90f`. For each feature (general loops, literals, div/exp/rem, grid tiling, stride arguments, boundary dtypes), ten generated kernels must compile, and their TTGIR must show the feature (`scf.for` with a negative step, `program_id` and the mask, `div.full` and `ex2.approx`, stride arguments, f16 stores). Under the interpreter, every kernel must take the path G3 solved for, and wherever P0 is defined it must agree with the interpreter bitwise. Any disagreement is a defect in G1–G8 before it is anything else. Bring back the per-feature compile table and the disagreements.

### R2. Relations on generated kernels [REMOTE-GPU, declared, two GPUs, 90 minutes]

Runbook `R2-RUNBOOK.md`, after R1 passes. One `--generation` block per feature, 50 seeds each, judged by layout-mutator's relation stage if phase 1 has landed, otherwise by the spike's driver (`output/calibration-spike-2026-11-02/spike/spike.py`), which already reads seeds and runs the values, positions, launch and code relations. Declared before the run: counts per feature and per `path.target` of admitted and violated relations, every violation rerun once for determinism, and no finding claimed until a reduced reproducer runs on a fresh checkout of trunk and of current main. Outcome rule: if a feature's block produces a deterministic violation, that feature gets the next GPU time and a reduction; if no block does, the per-target admission table is the result and the next step is chosen with Merlin.

## Part IV. Relation to the other plans

- **Seed-export plan.** G9 and G10 extend E1–E4; E5 and E6 are unchanged. Its decision text already allows div, exp, rem and zero-trip loops in generated bodies; G0, G1 and G5 make the profile accept them.
- **Exact-relations plan.** The launch relation's tile variant (task 5) should read `launch.tiles` when present instead of guessing the grid; its layout and alignment variants (task 5) should read `launch.strides` and `launch.offsets`. Both are optional reads of the new fields.
- **31 October plan.** Waves 1 and 2 and R0 are unaffected; they run on the v3 route.

## Part V. Record

- 2026-11-03. Plan written after the calibration spike; based on the builder, profile and emitter at `b890ea73`. Nothing implemented.
