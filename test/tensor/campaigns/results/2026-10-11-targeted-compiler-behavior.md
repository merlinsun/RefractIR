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
* No rematerialization is attributable to a layout pass. Matched recomputation
  with redirected uses appears only in `InlinerPass`, `CSEPass`, and `SCCPPass`.
  Per the plan, a layout result without rematerialization is useful evidence but
  does not satisfy a rematerialization claim.
* In these small cases the counted loop survives TTIR and TTGIR (one `scf.for`
  in each) and is gone by LLIR: `SCFToControlFlow` lowers it and the LLVM
  pipeline then fully unrolls it, so the PTX has no backedge. An earlier
  reading of this record said the loop was already removed before TTGIR; that
  was an observer defect (loop detection only recognised structured MLIR
  operations, so any lowered stage counted as loop-free) and is corrected here.
  The next section shows a configuration that reaches PTX with a backedge.

## Track A, second run: a retained loop with a layout decision inside it

Configurations at the profile volume retain the loop. The sweep
(`sweep-retained.json`, `sweep2-retained.json`) shows one `scf.for` in TTGIR at
`[256, 2]` and `[128, 4]` for 2–8 iterations and at `[32, 16]` for 4–8, for both
the default and the orthogonal-reduction families, at one and four warps.
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
| Loop removal | `SCFToControlFlowPass`, `loop_construct_delta` -1 |
| PTX backedge | one, `bra $L__BB0_1` to a label defined 132 lines earlier |
| PTX shuffles / shared transfers | 12 / 0 (one warp), 20 / 2 (four warps) |
| Rematerialization | none attributable to a layout pass |

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
detector missed real loops. Counting every `label %x` operand with numeric
labels agrees with the PTX backedge in 144/144 runs. Both the label patterns
and the branch scanning are fixed, and the helper now reports the backedge where
it previously reported a null.

## Track B, third run: guarded v3 tails

Six cases (three rank-two tail shapes, each with two padding patterns and a
`(3, 5)` guard envelope) pass on the device against their complete references
with `campaign_valid` (`track-b3-tail-guard-evidence.json`). For each shape the
two padding patterns produce byte-identical logical outputs, which is the
output-independence check the guard work is for. This closes the combination
the first guarded run missed: the earlier examples were power-of-two v2 cases,
so the v3 tail path through slot classification was never exercised.

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
