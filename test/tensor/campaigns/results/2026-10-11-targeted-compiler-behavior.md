# Targeted compiler behavior: first evidence runs

Environment: Tesla V100-SXM2-32GB (sm_70), driver 535.274.02, Python 3.12.15,
torch 2.5.1+cu121, Triton `3.9.0` imported from the pinned checkout at
`32f06df4bf1c30885ff61ee3488836ca81ed65f0` (`test/tensor/environments/v100-triton-3.9.0-32f06df4`).
Every run below records `campaign_valid: true` (source revision, compiler
digest, environment probe, and package lock present).

Outcomes separate execution from observation. An execution result is pass,
rejection, or failure; an observation result is "behaved as stated", "not
observed", or "unavailable". Nothing here is a compiler discovery.

## Track A: configured long-axis regions

Four generated families at `[4, 8]`, bounds 2–3, admitted by the exact f32
reference (nonnegative integer inputs, positive-zero obligations), compiled by
`TritonGPURemoveLayoutConversions` and friends, and executed on the device.
Each case matches its complete 128-byte reference bit-for-bit.

| Case | Execution | Target pass evidence | Effective conversions removed by that pass | PTX warp shuffles |
|---|---|---|---|---|
| cross-state | pass | `RemoveLayoutConversions` runs 4× per compilation | 8 | 0 |
| reduction-arithmetic | pass | same, 4× | 11 | 4 |
| orthogonal-reductions | pass | same, 4× | 15 | 10 |
| mixed-depth | pass | same, 4× | 11 | 6 |

Observations:

* The four families reach different conversion counts under one pipeline, so
  the connection skeleton changes what the layout passes see.
* These small cases still hold the loop in TTGIR, but LLVM unrolls it before
  PTX, so this table alone does **not** show a layout decision that reaches a
  device loop; the retained-loop run below does.
* `TritonGPUReduceDataDuplication`, `TritonGPUOptimizeThreadLocality`, and
  `TritonLoopInvariantCodeMotion` each report one measurable transition per
  compilation (`target_pass_evidence` in `track-a-evidence.json`).
* No rematerialization is attributable to any pass. An earlier reading of this
  record reported matched recomputation in `InlinerPass`, `CSEPass`, and
  `SCCPPass`; those were SSA renames that the matcher mistook for redirected
  uses, and the corrected matcher confirms none. The record has been
  regenerated: the conversion removals above stand, and every
  `confirmed_rematerialization_passes` entry is now empty.
* In these small cases the counted loop survives TTIR and TTGIR (one `scf.for`
  in each, `mlir-construct` evidence) and is gone by LLIR: `SCFToControlFlow`
  lowers it and the LLVM pipeline then fully unrolls it, so no branch cycle
  remains in LLIR or PTX. An earlier reading said the loop was already removed
  before TTGIR; that was an observer defect (loop detection only recognised
  structured MLIR operations, so any lowered stage counted as loop-free) and is
  corrected here. The next section shows a configuration that keeps a cycle all
  the way to PTX.

## Track A, second run: a retained loop with a layout decision inside it

Configurations at the profile volume retain the loop. The sweep
(`sweep-retained.json`, `sweep2-retained.json`) shows one `scf.for` in TTGIR at
`[256, 2]` and `[128, 4]` for 2–8 iterations and at `[32, 16]` for 4–8, for both
the default and the orthogonal-reduction families, at one and four warps. Those
two files record the structured-loop count only: their PTX column came from an
earlier branch heuristic and has been removed rather than corrected, because the
sweep's raw IR was not archived. The cycle-based records are
`track-a-retained-evidence.json` and `track-a-layout-warp-evidence.json`.
Among these, `mixed-depth` at `[32, 16]` with four iterations is also admitted
by the exact f32 reference, so it can carry a full certificate.

Two such cases (`num_warps` 1 and 4) were built through the exact builder with
the region configuration recorded, executed on the device, and compared with
their complete 2048-byte references: both pass, both `campaign_valid`
(`track-a-retained-evidence.json`).

| Observation | Result |
|---|---|
| TTGIR / TTIR loop constructs | 1 / 1 (`mlir-construct` evidence) |
| `RemoveLayoutConversions` occurrences | 4, loop present before and after every one |
| Conversions removed by that pass | 12, with `loop_construct_delta` 0 |
| Loop removal from MLIR | `SCFToControlFlowPass`, `loop_construct_delta` -1 (a lowering; the cycle then lives in LLIR and PTX) |
| PTX loop | one branch cycle, `bra`/`bra.uni` back to the loop header (`branch-cycle`) |
| PTX shuffles / shared transfers | 12 / 0 (one warp), 20 / 2 (four warps) |
| Rematerialization | none attributable to any pass (the earlier "matches" in the inliner, CSE, and SCCP were renames) |

So a layout decision is eliminated inside a loop that survives the layout
passes and reaches PTX as a genuine backedge. The structured loop disappears at
the `scf`-to-`cf` boundary, which is after TTGIR and before LLIR.

## Retained-loop configurations and exact admission

The exact reference admits a recurrence only while every materialized value and
reduction partial stays inside 2^24. Reduction width and trip count multiply, so
a 512-element shape with 2–4 trips is admitted only when the reduced axis is
narrow (`mixed-depth` at `[32, 16]` reduces a 32-wide axis; `[2, 256]` reduces a
2-wide axis and is admitted to eight trips) or the domain is small. Rejected
configurations are recorded with their reason in `sweep2-retained.json` rather
than relaxed: a 256-wide reduction with four trips over `[0, 2]` is refused.

## Track A, fourth run: physical layouts and launch settings

The retained-loop case was repacked under independent input and output layouts
and re-run at four warp counts (`track-a-layout-warp-evidence.json`). The
logical program, inputs, and reference are identical in every run; only the
recorded physical layout or the launch setting changes.

| Variant | Execution | TTGIR loop | PTX loop cycles | PTX shuffles |
|---|---|---|---|---|
| row-major / row-major | pass | 1 | 1 | 12 |
| row-major / column-major | pass | 1 | 1 | 12 |
| column-major / row-major | pass | 1 | 1 | 12 |
| column-major / column-major | pass | 1 | 1 | 12 |
| padded (stride 18, offset 3) / row-major | pass | 1 | 1 | 12 |
| row-major / padded (stride 20, offset 1) | pass | 1 | 1 | 12 |
| padded / padded | pass | 1 | 0 | 4 |
| one warp | pass | 1 | 1 | 12 |
| two warps | pass | 1 | 1 | 16 |
| four warps | pass | 1 | 1 | 20 |
| eight warps | pass | 1 | 0 | 40 |

Every run matches the complete reference, so the emitted addressing follows
the recorded layout rather than the identity mapping. Retention is not
monotone in the launch setting: the loop survives TTGIR everywhere, and LLVM
keeps a PTX loop at one to four warps and for six of the seven layout
combinations, but unrolls it at eight warps and for padded/padded. That is an
observation about the toolchain under these settings, not a defect.

## Track B: tails and physical layouts

Nine compositional v3 rank-two tails (`[8, 31]`, `[63, 4]`; padded tiles
`[8, 32]`, `[64, 4]`) and six rank-one v3 stratum tails (`17`, `31`, `33`)
compiled and executed. All fifteen match their complete logical references.

Observations (the rank-one runs are in `track-b-evidence.json` records of the
run logs; the rank-two runs are in `track-b2-evidence.json`):

* The logical extent survives into the emitted kernel predicate: masks compare
  the padded arange against `17`, `31`, `33`, or `63` while the padded tile is
  `32`, `32`, `64`, or `64`.
* Inputs carry positive strides with nonzero offsets (`[32, 1]` offset 1,
  `[1, 11]` offset 1), so addressing is not the identity mapping.
* Every tail case also reports `loop_status: unrolled_or_removed`; the tail
  evidence is the predicate and address relation, not loop retention.

## Track C: explicit Gluon layout plans

One admitted orthogonal-reduction region `[8, 8]`, one bound, four plan families
emitted by `render_gluon_kernel(..., plan=...)` and run on the device with the
qualified compiler. All four match the complete 64-word reference.

| Family | Execution | TTGIR conversions (effective) | PTX shuffles | PTX shared transfers |
|---|---|---|---|---|
| all-same-layout | pass | 8 (0) | 8 | 2 |
| load-to-compute | pass | 8 (1) | 8 | 8 |
| split-reduction-consumers | pass | 10 (5) | 10 | 19 |
| loop-state-layout | pass | 14 (7) | 8 | 33 |

Observations:

* Requested conversions survive to TTGIR: the baseline has none that change a
  layout, and each family adds the ones it asked for.
* Surviving conversions lower to real communication: shared-memory transfers
  grow from 2 to 33 across the families, and warp shuffles appear in every run.
* The loop-state family required a real correction during qualification: a
  loop-carried value must keep one layout for the whole loop, so the state
  layout is established before the loop and the assigned roots are converted
  into it, and the broadcast reference needed its own view. Without that, Gluon
  refuses the kernel, which is why the plan asks for consistent loop-state
  layouts across the backedge.

## Track B, second run: guarded allocations on the device

Three cases of the admitted `mixed-depth` `[32, 16]` four-trip region were built
through the exact builder: one with a `(3, 5)` guard envelope and padding
`55555555`, one with the same envelope and padding `aaaaaaaa`, and one with no
envelope as a control. All three pass on the device against their complete
references with `campaign_valid` (`track-b2-guarded-evidence.json`); the
per-case `worker-staging.json` records the guards, the padding, and a 2080-byte
physical allocation around a 2048-byte logical tensor.

The first device attempt of this run failed: with the whole guarded allocation
handed to the kernel, the recorded element offsets addressed the tensor as if
it started at slot 0, so the kernel read three slots early. Guarded inputs and
outputs are now handed over as views that start at the guard prefix while
sharing storage with the whole allocation, which keeps the record's addressing
correct and still lets the worker refuse a write into a guard, gap, or logical
slot by name.

## Track C, second run: all six families through the runner

Twelve device runs (`summary-gluon-plan-sweep-<family>-w<warps>.json`) cover the
six families at one and four warps on the same admitted `[8, 8]` region, one
bound. Every run matched the complete 64-word reference, and both calibration
controls behaved as declared.

| Family | TTGIR conversions (total/effective), w1 | w4 | PTX shuffles / shared, w1 | w4 |
|---|---|---|---|---|
| all-same-layout | 8 / 0 | 8 / 0 | 24 / 0 | 8 / 2 |
| load-to-compute | 8 / 1 | 8 / 1 | 24 / 16 | 8 / 6 |
| reduction-result-expand | 8 / 2 | 8 / 2 | 21 / 12 | 24 / 6 |
| shared-value-reused | 9 / 4 | 9 / 4 | 24 / 38 | 12 / 13 |
| split-reduction-consumers | 10 / 5 | 10 / 5 | 12 / 49 | 12 / 19 |
| loop-state-layout | 14 / 7 | 14 / 7 | 24 / 77 | 8 / 43 |

Of 114 requested conversions, 38 survive, 76 are eliminated, none are folded or
unattributed. Every survivor is a cross-lane change at one warp and a cross-warp
change at four, and all lower in `ConvertTritonGPUToLLVM`; no pass adds a
conversion, which is why nothing is unattributed. Registers per thread stay at
or below 58 with no spills.

Two gaps this sweep exposed, both in the plan rather than in the compiler, are
now fixed:

* `reduction-result-expand` asked for views that were the compute layout by
  construction, and named one consumer the emitter never queries, so its run
  reduced to load-to-compute plus compute-to-store. The family now gives the two
  broadcast consumers a view that differs from the compute layout.
* A declared consumer view that no consumer ever queries silently did nothing.
  Emission now refuses such a plan, and a test checks that every family's
  difference from the baseline reaches the generated kernel.

## Track A, third run: the generation matrix

Seventy-two configurations (six shapes, four families, bounds 2–4) were
generated, compiled, and launched at one and four warps
(`.planning/2026-10-11-targeted-compiler-behavior/track-a-matrix.md`, raw
records under `/host/track-a-matrix/`):

| Count | Value |
|---|---|
| configurations generated / out of profile | 72 / 0 |
| admitted / rejected by the exact proof | 51 / 21 |
| Triton runs compiled and launched | 144 / 144 |
| admitted runs matching the reference bits | 102 / 102 |
| configurations with a TTGIR loop | 51 / 51 admitted |
| configurations with a PTX backedge | 23 (11 of them admitted) |

Every rejection carries the same reason (`materialized value or reduction
partial sum exceeds exact range`) and follows the reduced-axis width times the
trip count. Retention does not follow admission: some configurations keep a
device loop only where the proof rejects them, and `cross-state`, which has no
reduction, never keeps a backedge.

Zero layout conversions survive to the final IR in all 144 runs; the pass-dump
sub-study shows `TritonGPURemoveLayoutConversions` removing everything present
on its first occurrence with `loop_construct_delta` 0, and the three reduction
families reaching PTX with shuffles and shared-memory transfers (928/114,
738/118, 876/122) while `cross-state` reports none.

This run also caught a second observer defect: optimized LLVM IR prints numeric
labels and lists a bottom-tested loop's exit block first, so the LLVM backedge
detector missed real loops. A third correction followed from the review: a
branch to an earlier *printed* block is not a cycle, so the detector now builds
a per-function branch graph (including PTX fall-through edges) and reports the
branches that participate in a genuine cycle; a stage whose branches cannot be
read stays unavailable rather than loop-free. Under that reading the matrix
device-loop claim holds unchanged (39 runs, 23 configurations, 11 admitted, and
7 at both warp counts).

## Track B, third run: guarded v3 tails

Six cases (three rank-two tail shapes, each with two padding patterns and a
`(3, 5)` guard envelope) pass on the device against their complete references
with `campaign_valid` (`track-b3-tail-guard-evidence.json`). For each shape the
two padding patterns produce byte-identical logical outputs, which is the
output-independence check the guard work is for. This closes the combination
the first guarded run missed: the earlier examples were power-of-two v2 cases,
so the v3 tail path through slot classification was never exercised.

## Track A, fifth run: a counted region against its unrolled twin

The plan asks for paired counted and unrolled forms of one representative, on
the device, with complete references (`track-a-paired-unrolled-device.json`).
The representative is a `mixed-depth` region over a `[32, 16]` state with four
trips (seed 9, additive, input domain `[0, 8]`), certified through a concrete
in-domain input whose output becomes the case's target.

| Form | Program | TTIR/TTGIR | LLIR/PTX | Execution | Output |
|---|---|---|---|---|---|
| counted | `638322a9…` | one `scf.for` | one branch cycle | pass, 0 mismatches | `f5826a57…` |
| unrolled | `359b755b…` | none | none | pass, 0 mismatches | `f5826a57…` |

The unrolled companion is built by `test.lib.tensor_unrolled_companion`: the
counted program is unrolled, its Triton kernel is re-emitted for the
straight-line program, and its reference is an interpreter replay of the
unrolled program over the counted inputs, which must equal the counted
`expected.bin` at 128 and 256 reference bits. Copying the counted case and
rewriting `program.sir` is not enough: the loader refuses that with "generated
region writes an undeclared or loop-carried value", because the proof, the
selection artifacts, and the kernel still describe the counted program.

Two readings. The same computation keeps a loop in every stage when it is
counted and has none when it is unrolled, so a loop-targeting observation on
this program is attributable to the loop rather than to the arithmetic. And both
forms produce the identical complete output on the device, which is the
equivalence the pair exists to show.

Both arms are campaign-valid: each run names the source revision `1c09c9a5…`,
the compiler flags, the Triton source revision, and the package lock, which is
what the runner derives that flag from. The companion's reference is an
interpreter replay of the unrolled program rather than a region certificate,
because no proof family accepts an unrolled body; that is a limitation of the
reference kind and is stated separately from qualification.

The pair was rebuilt from the counted configuration and the committed companion
builder under the tightened derivation check and re-run on the device: both arms
pass with `actual == expected`, and the rebuilt artifacts reproduce the reviewed
hashes exactly (`f5826a57…` for the reference, `f60bb874…` for the inputs,
`359b755b…` for the unrolled program, `638322a9…` for the carried counted
program). Changing an inherited input and refreshing its digest is refused at
load time, so a stale reference cannot be dressed up as a companion. One
environment note: a rebuild raises the builder's per-stage solve budget, because
this 512-lane integer-twin query needs about six minutes and the library's own
cap is sixty seconds.

The two comparison policies agree on signed zeros: the counted case compares
exactly, and the companion states unsigned-zero exactness and earns it through a
derivation checked against the counted case - its identity, its reference
digest, the unrolled source digest, the trip accounting, the inherited inputs,
and one replay per precision that reproduces the reference. Flipping the sign of
one zero lane makes both policies report `exact_mismatch`; the genuine bytes
pass.

## Gaps and controls

* Guard *detection* is verified on the host (a mutated allocation is refused by
  slot name) but not on the device: no run has yet executed a kernel that
  deliberately writes outside its tensor.
* Track C ran one bound, so every device run was unrolled and no cross-backedge
  loop-state conversion was exercised on the device; no rematerialization is
  attributed to any pass.
* The Track C intersection with Track B (padded logical tails under explicit
  layouts) is not implemented: the Gluon emitter still rejects non-power-of-two
  dimensions.
* Ranks, warps, and layouts beyond the runs above are untouched; the run
  counts here are far below the plan's per-track budgets.
* No new compiler failure was found. The differences reported above are
  compiler behavior, not defects.
