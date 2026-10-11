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
* `TritonGPUReduceDataDuplication`, `TritonGPUOptimizeThreadLocality`, and
  `TritonLoopInvariantCodeMotion` each report one measurable transition per
  compilation (`target_pass_evidence` in `track-a-evidence.json`).
* No rematerialization is attributable to a layout pass. Matched recomputation
  with redirected uses appears only in `InlinerPass`, `CSEPass`, and `SCCPPass`.
  Per the plan, a layout result without rematerialization is useful evidence but
  does not satisfy a rematerialization claim.
* The counted loop is **not** retained: `loop_status` is `unrolled_or_removed`
  for the required TTGIR stage and the final snapshot holds no loop construct.
  Two-to-three-trip loops over 32 elements are unrolled before the device. A
  retained-device-loop claim needs larger trip counts or volumes, which is a
  separate extension rather than a result of this run.

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

## Gaps and controls

* Two padding patterns and allocated guard envelopes are implemented and
  host-tested; no device run used them yet, so no guard-write observation is
  claimed.
* Ranks, warps, and layouts beyond the runs above are untouched; the run
  counts here are far below the plan's per-track budgets.
* No new compiler failure was found. The differences reported above are
  compiler behavior, not defects.
