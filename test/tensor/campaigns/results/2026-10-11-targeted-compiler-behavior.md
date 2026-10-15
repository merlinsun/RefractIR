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

| Case | Execution | Target pass evidence | Conversions removed by that pass (total) | PTX warp shuffles |
|---|---|---|---|---|
| cross-state | pass | `RemoveLayoutConversions` runs 4× per compilation | 8 | 0 |
| reduction-arithmetic | pass | same, 4× | 11 | 4 |
| orthogonal-reductions | pass | same, 4× | 15 | 10 |
| mixed-depth | pass | same, 4× | 11 | 6 |

These are the conversions the pass removes, not an effective count: three per
family are same-encoding no-ops, so 5, 8, 12, and 8 respectively change an
encoding. The named-pass section below attributes each one.

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
  `confirmed_rematerialization_passes` entry is now empty. The named-pass
  section below settles this with per-transition witnesses and records the
  category as explicitly unreached.
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

Of 114 requested conversions, 38 change an encoding and 76 are
equivalent-encoding no-ops; all 114 survive to the tensor-dialect boundary, and
none is eliminated, folded by a recomputed replacement, or unattributed. Every
survivor is a cross-lane change at one warp and a cross-warp change at four, and
all lower in `ConvertTritonGPUToLLVM`; no pass adds a conversion, which is why
nothing is unattributed. Registers per thread stay at or below 58 with no
spills.

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
builder under the tightened derivation check: both arms pass with `actual ==
expected`, and the rebuilt artifacts reproduce the reviewed hashes exactly
(`f5826a57…` for the reference, `f60bb874…` for the inputs, `359b755b…` for the
unrolled program, `638322a9…` for the carried counted program). Changing an
inherited input and refreshing its digest is refused at load time, so a stale
reference cannot be dressed up as a companion.

The rebuild was first checked under Triton 3.7.1 and Python 3.10.12 with no
source revision and no compiler flags recorded, so both of its arms were
campaign-invalid: matching program, input, and output hashes show that the
companion computes the same thing, not that the same compiler produced it.

A second re-run in the intended environment had a subtler problem: its driver
imported the loader and companion builder from a tree built before the
derivation repair, so the companion it produced carried no counted artifacts and
the repaired loader refuses it. That run is kept in the evidence as superseded.

The qualified re-run uses the repaired counted case with a companion rebuilt by
the committed builder, and it is verified rather than asserted: the imported
`test/lib/tensor_case.py` digests as `67643011…` and
`test/lib/tensor_unrolled_companion.py` as `f9aacfc9…`, both equal to the
committed blobs, the companion carries seven counted artifacts, and both arms
pass with campaign-valid provenance - Triton `3.9.0` at `32f06df4…`, Python
3.12.15, revision `841ef12`, compiler flags recorded - on the same
`f5826a578a9ae11e` reference and program digests. Each run in the evidence
carries its own environment, its own qualification, and, where it matters, the
digest of the library it actually imported. One
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

## Track C, third run: layout conversions across a retained backedge

Six device runs (`track-c-retained-backedge.json`) emit the counted region of the
Track A fifth run - same `program.sir`, digest `638322a9…` - through the
plan-mode Gluon emitter. The earlier Track C sweep ran a one-trip region, so
every one of its kernels unrolled; this program has four trips, and every run
kept them.

| run | execution | loop at TTGIR / LLIR / PTX | requested | in loop body | eliminated | surviving | surviving in loop body | surviving that feed the backedge |
|---|---|---|---|---|---|---|---|---|
| all-same-layout w1 | pass | `scf.for` / cycle / `bra` | 8 | 6 | 8 | 0 | 0 | 0 |
| all-same-layout w4 | pass | `scf.for` / cycle / `bra` | 8 | 6 | 8 | 0 | 0 | 0 |
| load-to-compute w1 / w4 | pass | `scf.for` / cycle / `bra` | 8 | 6 | 7 | 1 | 0 | 0 |
| loop-state-layout w1 | pass | `scf.for` / cycle / `bra` | 15 | 11 | 6 | 9 | 6 | 2 |
| loop-state-layout w4 | pass | `scf.for` / cycle / `bra` | 15 | 11 | 6 | 9 | 6 | 2 |

Every run matched the complete 512-word exact f32 reference bit for bit, and each
one's `add-one-output-mutant` calibration was detected. The retained loop is
structural evidence, not a count: TTGIR holds
`scf.for %_ = %c0_i32 to %c4_i32 … iter_args(%acc_x, %acc_y)`, LLIR closes a
branch cycle with `br i1 %exitcond.not, label %566, label %177`, and PTX closes
the body with `@%p2 bra $L__BB0_1;`.

What the conversions do across the backedge, at one warp:

* 9 of the 15 requested conversions survive to the final TTGIR; 6 are
  `identical` no-ops the emitter writes after each arithmetic node.
* 2 survivors are the state conversions whose results are the `scf.yield`
  operands, and 2 more read the loop's block arguments. All nine classify
  `cross_lane` at one warp; at four warps six are `cross_warp` and three stay
  `cross_lane`.
* No pass removes a requested conversion before `ConvertTritonGPUToLLVM`; no
  pass adds one anywhere, so nothing is unattributed and nothing is folded.
  `SCFToControlFlowPass` turns the structured loop into the branch cycle with all
  15 matched: 11 of them change enclosing region when the loop it sat in is
  lowered away (`preserved` 4, `moved` 11 in the corrected record), and the four
  that stay are the ones outside the loop.
* The in-loop survivors lower to shared-memory transfers with barriers: 164
  transfers and 5 barriers inside the loop body at one warp (157 and 14 at four),
  where the two same-graph control families have 0 and 0. PTX `.loc` attributes
  every transfer to a conversion line - 36 each to the two loop-carried reads
  and to the two backedge writes, 20 to the reduction view - so this is
  per-conversion evidence, not only a whole-kernel delta.
* The plan costs registers: 255 per thread with 16 spills at one warp (63 and 0
  for the baseline); 138 and 0 at four.

`loop-state-layout` is the only one of the six plan families that can address
this graph together with the two controls; `shared-value-reused` and
`split-reduction-consumers` are refused with `no value is read by two reduction
nodes` and `reduction-result-expand` with `fewer than two reductions feed a
broadcast`. Nothing refused a plan for the loop's state.

The region-map limit recorded in the first draft of this section is fixed in the
working tree. `observe_compiler_ir`'s `enclosing_region` now names a loop whose
header carries a counted result list (`%acc_y:2 = scf.for … {`) and an operation
printed in generic form (`"tt.reduce"(…) ({`), so the body of this loop reads
`scf.for` instead of `tt.func`/`module`: 11 of the 15 requested conversions
change reading in the final TTGIR, and
`effective_conversions_inside_loop_region` is 6 (the six effective in-loop
survivors, kernel lines 22, 23, 27, 28, 40, 41), not 0. The control on the
module's own parser still holds: a single-result header is reported as
`scf.for`. Loop membership in the record still comes from a direct parse of the
loop header, its `iter_args`, and its `scf.yield`, and the corrected field now
agrees with that parse. `GluonInline`'s seven "moved" fragments are gone with
the defect: the transition now reports `moved` 0 and `preserved` 15, all fifteen
inside `scf.for` in both snapshots. The movement the run does report is
`SCFToControlFlowPass`'s (`moved` 11, `scf.for` → `tt.func`), which is the loop
region being lowered rather than a conversion leaving a retained loop.

The ordinary Triton emitter of the same program shows the contrast: its 12
conversions are all removed by `TritonGPURemoveLayoutConversions` before the
device, so its retained loop carries none. The Track C x Track B intersection
(padded tails under explicit layouts) is still not attempted; beyond the
power-of-two guard it needs mask-aware emission and layout validation for
non-power-of-two extents.

## Track A, sixth run: recorded input proposals

The plan asks for four recorded proposals per admitted kernel - independent small
values, an asymmetric coordinate pattern, and two sparse patterns that exercise
different groups or the final valid lanes - each with its own complete reference
and a compiled-mode device run (`track-a-input-proposals.json`). The rules live
in `test/lib/tensor_input_proposals.py` and are a pure function of the case:

| Proposal | Rule (all leaves integer, in `[0, 8]`, positive zero) |
|---|---|
| `independent-small` | per-lane `sha256(rule:source:seed:lane)[0:8] mod 4` |
| `asymmetric-coordinates` | `(3*row + 5*column + 1) mod 9` |
| `sparse-distinct-groups` | first lane of every group on the first reduction axis cycles `1,2,3` |
| `sparse-final-lanes` | final lane of every group on every reduction axis holds `1`; the final element holds `8` |

`test/tensor/campaigns/run_input_proposals.py` regenerates the region source
through the native exporter, admits it over `[0, 8]`, writes one derived case per
proposal, and runs the suite through the ordinary runner with
`REFRACTIR_TRITON_PASS_DUMP=1`. Each derived case carries the same `program.sir`
as the counted case, its own `inputs.bin`, `expected.bin`, `certificate.json`,
and its interpreter replays at 128 and 256 reference bits.

The counted representative was rebuilt first and reproduces exactly: target
`12005000` from the certificate of the all-domain-bound input, program
`638322a9…`, inputs `f60bb874…`, reference `f5826a57…`, kernel `2b5e96b0…`, all
four equal to the recorded identity. It ran in the same suite in exact mode and
passed. At this configuration (`[32, 16]`, four trips, seed 9, additive,
`num_warps` 4, row-major) two of the four Track A families are refused by the
exact reference - `reduction-arithmetic` and `orthogonal-reductions` report
"materialized value or reduction partial sum exceeds exact range" - so they carry
no proposals. A diagnostic sweep says the largest admitted domain upper bound
would be 2 for `orthogonal-reductions` and 0 for `reduction-arithmetic`; no case
was built from those bounds and no domain was changed to obtain admission.

| Kernel | Proposal | Input | Nonzero | Admission | Reference | Execution | Comparison | Valid | Word-exact |
|---|---|---|---|---|---|---|---|---|---|
| mixed-depth | independent-small | `8f04b156…` | 396 | certificate | certificate `b9bd16f2…` | pass | pass | yes | yes, 0 zero lanes |
| mixed-depth | asymmetric-coordinates | `a14682df…` | 459 | certificate | certificate `b9bd16f2…` | pass | pass | yes | yes, 0 zero lanes |
| mixed-depth | sparse-distinct-groups | `a2e3ecd6…` | 16 | certificate | certificate `b9bd16f2…` | pass | pass | yes | yes, 0 zero lanes |
| mixed-depth | sparse-final-lanes | `97b76598…` | 16 | certificate | certificate `b9bd16f2…` | pass | pass | yes | yes, 0 zero lanes |
| cross-state | independent-small | `a9456f78…` | 380 | certificate | certificate `7d1556be…` | pass | pass | yes | yes, 132 zero lanes |
| cross-state | asymmetric-coordinates | `a14682df…` | 459 | certificate | certificate `7d1556be…` | pass | pass | yes | yes, 53 zero lanes |
| cross-state | sparse-distinct-groups | `a2e3ecd6…` | 16 | certificate | certificate `7d1556be…` | pass | pass | yes | yes, 496 zero lanes |
| cross-state | sparse-final-lanes | `97b76598…` | 16 | certificate | certificate `7d1556be…` | pass | pass | yes | yes, 496 zero lanes |

The reference is the certificate's complete output wherever the exact reference
admits the proposal, which is all eight; the interpreter replays at both
precisions reproduce those bytes, and the recorded comparator is zero-tolerance
numeric (`atol` 0, `rtol` 0). The exact comparator is not available to these
cases: the loader grants it only with solver-selection evidence whose model is
the runtime input, and a rule-chosen proposal has none. The comparison is
therefore backed by an additional word-for-word check of `actual.bin` against
`expected.bin`, which is what the "word-exact" column reports; it matters for
`cross-state`, whose references contain up to 496 zero lanes, because zero
tolerance alone cannot tell `+0.0` from `-0.0`. No proposal duplicates another
(six pairwise comparisons per kernel, all distinct) and none coincides with the
counted solver witness; the asymmetric and sparse bytes do coincide across the
two kernels, since those rules depend on shape, domain, and axes rather than on
the source, which the record states as `cross_kernel_duplicate_inputs`; and no
reference output is constant, though the all-zero probe input does produce a
constant all-zero output, which is recorded separately as the zero probe (both
families' outputs do depend on their inputs).

| Kernel | TTIR | TTGIR | LLIR | PTX | Loop status | PTX shuffles / shared |
|---|---|---|---|---|---|---|
| mixed-depth | 1 `scf.for` | 1 `scf.for` | 1 cycle | 1 cycle | retained | 20 / 2 |
| cross-state | 1 `scf.for` | 1 `scf.for` | none | none | unrolled by LLIR | 0 / 0 |

| Pass (transition index) | mixed-depth | cross-state |
|---|---|---|
| `ConvertTritonToTritonGPU` conversions added / effective | 8 / 8 | 5 / 5 |
| `TritonGPUCoalesce` added / preserved | 4 / 8 | 4 / 5 |
| `TritonGPURemoveLayoutConversions` occurrences | 4 (loop present before and after each) | 4 (same) |
| conversions removed by its first occurrence / effective | 12 / 12 | 9 / 9 |
| `SCFToControlFlowPass` `loop_construct_delta` | -1 | -1 |
| `TritonGPUOptimizeThreadLocality`, `ReduceDataDuplication`, `LoopInvariantCodeMotion` | 1 occurrence each, no conversion or loop delta | same |
| rematerialization attributable to any pass | none | none |

The four proposals of each kernel compile to a byte-identical emitted kernel
(`2b5e96b0…` for mixed-depth, equal to the counted case's kernel; `60fe7360…` for
cross-state), and their TTIR, TTGIR, and LLIR are identical once the case path
the compiler embeds in source locations is removed. PTX differs only in the
length-prefixed path string of its debug section. So in this environment the
input data does not change the compiled program or its layout decisions: the
same 12 conversions are removed inside the same retained loop for every
mixed-depth proposal, which is why these runs extend numerical and comparison
coverage rather than layout coverage.

What the proposals do change is the data path: the two sparse patterns leave 496
of 512 lanes at zero in `cross-state` and the device still reproduces the
reference word for word, and the same computation loses its loop by LLIR in
`cross-state` while keeping it to PTX in `mixed-depth` - a difference between the
two logical programs, not between their inputs.

Provenance: Tesla V100-SXM2-32GB, driver 535.274.02, Triton `3.9.0` at
`32f06df4…`, Python 3.12.15, torch 2.5.1+cu121, runner revision label
`81a55d3…` (the exported tree whose modules were imported; 47 of the 48 recorded
`test/lib` modules equal their committed blobs and the 48th is the new proposal
module), native tools by digest (`symirc` `5915fbda…`), and the driver by digest
`1639b9e8…`. Raw pass dumps and case directories are archived under
`test/tensor/campaigns/archive/2026-10-16-input-proposals/`.

## Named-pass attribution of every conversion

The sections above report conversion *counts* per pass and stage. This run
attributes each individual conversion: which named pass created it, which one
removed it, whether it crossed a loop boundary, and which ones survived. The
evidence is `track-a-named-passes.json` (per-conversion, per-pass, per-stage
records for the eight device cases), `track-c-named-passes.json` (the same
reading of the twelve archived Track C dumps), and the reports
`.planning/2026-10-11-targeted-compiler-behavior/named-passes.md` and
`named-passes-driver/`.

Eight device cases were run again with `REFRACTIR_TRITON_PASS_DUMP=1` and
`MLIR_DUMP_PATH` set by `test.lib.triton_case_worker.configure_pass_dumps`, each
with a fresh Triton cache: the four admitted `[4, 8]` families, the two
retained-loop `[32, 16]` four-trip cases, the counted arm of the recorded pair,
and its mechanically unrolled twin. Each run re-emitted its kernel with the
recorded `symirc` digest (`5915fbda…`), passed, and reproduced the archived
device output bytes exactly — `14a7cffd…`, `8102938a…`, `ad3ba996…`,
`f12aa932…` for the four `[4,8]` families, `f5d5a7f8…` for both retained cases —
and the pair reproduced the reviewed digests exactly (`638322a9…` counted,
`359b755b…` unrolled, `f5826a57…` reference). The Track C dumps were re-read
from the archive, with no new compilation.

Classification, per conversion followed across every snapshot: **surviving** —
still present in the last snapshot of the tensor dialects and lowered by the
transition that converts tensor operations to LLVM; **eliminated** — removed by
a named pass while tensor operations remain, with no equivalent definition
taking over its consumers; **folded** — it carries no layout change (source and
target encoding resolve equal) or its transition recomputed the value for its
former consumers; **unattributed** — it appears before the first snapshot,
vanishes without a matched transition, or its transition has no named pass.

| Case | Conversions | Created by | Removed by | Eliminated | Folded (equal encoding) | Surviving | Moved into / out of a loop |
|---|---|---|---|---|---|---|---|
| `cross-state` `[4,8]` | 8 | `ConvertTritonToTritonGPU` 5, `TritonGPUCoalesce` 3 | `RemoveLayoutConversions` ×1 | 5 | 3 | 0 | 0 / 0 |
| `reduction-arithmetic` `[4,8]` | 11 | 8, 3 | same | 8 | 3 | 0 | 0 / 0 |
| `orthogonal-reductions` `[4,8]` | 15 | 12, 3 | same | 12 | 3 | 0 | 0 / 0 |
| `mixed-depth` `[4,8]` | 11 | 8, 3 | same | 8 | 3 | 0 | 0 / 0 |
| `mixed-depth` `[32,16]` it4 w1 | 12 | 8, 4 | same | 12 | 0 | 0 | 0 / 0 |
| `mixed-depth` `[32,16]` it4 w4 | 12 | 8, 4 | same | 12 | 0 | 0 | 0 / 0 |
| counted pair | 12 | 8, 4 | same | 12 | 0 | 0 | 0 / 0 |
| unrolled twin | 21 | 17, 4 | same | 21 | 0 | 0 | 0 / 0 |

Every one of the 102 Track A conversions is created at the TTIR-to-TTGIR
boundary and removed by the **first** occurrence of
`TritonGPURemoveLayoutConversions` (transition 12 of 65, before
`OptimizeCTALocality`); the three later occurrences of that pass change nothing.
Counts per stage: 0 in TTIR, 102 born at the boundary, 90 eliminated and 12
folded in TTGIR, 0 lowered, 0 in LLIR or PTX. The final TTGIR artifact holds no
`convert_layout` operation in any of the eight cases, which is the independent
check on the zero-survivor verdict.

No Track A conversion enters or leaves a loop. Conversions created *inside* the
`scf.for` region are removed inside it: `orthogonal-reductions` 7 of 15,
`mixed-depth` and `reduction-arithmetic` 3 each, the retained and counted cases 3
of 12 each, and `cross-state` none — every conversion's birth and death region
are the same. The loop
itself is untouched until `SCFToControlFlowPass` (snapshot 45 to 46) turns the
structured loop into a branch cycle — `loop_construct_delta` −1, and the cycle
continues in LLIR and PTX (one backedge in the counted and retained cases). The
unrolled twin has no loop from the first inlined canonicalize onwards: its
source keeps a one-trip header (`br %i < 1, ^body, ^exit;` with four body
copies), which `InlinerPass` folds at transition 0, so its 21 conversions are
all born and removed outside any loop.

Track C, re-read from the twelve archived dumps: all **114** requested
conversions are present at the tensor-dialect boundary and are lowered by
`ConvertTritonGPUToLLVM`; none is removed by an earlier pass and none is
unattributed. 38 of them change the encoding and 76 are equal-encoding no-ops
the emitter writes after each arithmetic node.

| Family (w1 / w4) | Conversions | Effective (encoding change) | Equivalent | Removed before the boundary | Lowered by | Unattributed | PTX shuffles | PTX shared transfers | Registers/thread |
|---|---|---|---|---|---|---|---|---|---|
| all-same-layout | 8 / 8 | 0 / 0 | 8 / 8 | 0 / 0 | `ConvertTritonGPUToLLVM` | 0 / 0 | 24 / 8 | 0 / 2 | 30 / 19 |
| load-to-compute | 8 / 8 | 1 / 1 | 7 / 7 | 0 / 0 | same | 0 / 0 | 24 / 8 | 16 / 6 | 31 / 21 |
| reduction-result-expand | 8 / 8 | 2 / 2 | 6 / 6 | 0 / 0 | same | 0 / 0 | 21 / 24 | 12 / 6 | 30 / 32 |
| shared-value-reused | 9 / 9 | 4 / 4 | 5 / 5 | 0 / 0 | same | 0 / 0 | 24 / 12 | 38 / 13 | 41 / 32 |
| split-reduction-consumers | 10 / 10 | 5 / 5 | 5 / 5 | 0 / 0 | same | 0 / 0 | 12 / 12 | 49 / 19 | 40 / 32 |
| loop-state-layout | 14 / 14 | 7 / 7 | 7 / 7 | 0 / 0 | same | 0 / 0 | 24 / 8 | 77 / 43 | 58 / 46 |

This is the same 38/76 split the second Track C run reports, with the vocabulary
made explicit: the earlier table's "38 effective / 76 equivalent-encoding no-ops"
is "38 effective survivors / 76 equivalent-layout conversions that also reach the
boundary"; under the four-class scheme the 76 are *folded* (no layout change), not
eliminated, and nothing is eliminated or unattributed in that sweep. Every
family's effective count matches the archived runner's own
`ttgir_layout_conversion_count` exactly (0, 1, 2, 4, 5, 7 at both warp counts),
and registers stay at or below 58 with no spills.

The 86 "moved" fragments the library reports for Track C - the family compilation
of each of the twelve sweep runs - are not conversions crossing into or out of a
live loop. They are the single-trip `scf.for` of the Gluon region being folded
away by `GluonInline` at transition 0 (12 of 12 runs, `loop_construct_delta`
−1); the conversions were inside that loop and are outside it afterwards because
the loop is gone, not because a pass hoisted them out of a retained one. The
figure came from the named-pass driver's own region tracker; the corrected
library now reports the same 86 for this aggregation and names the two regions
`scf.for` (before) and `tt.func` (after), where the uncorrected library reported
80 of the same fragments as `tt.func` → `module`. The retained-backedge run above
is the case where the loop stays.

Local lowering evidence for survivors, and its limit. The Track C surviving
conversions are lowered by `ConvertTritonGPUToLLVM`, and the kernels carry the
lowering: at one warp `loop-state-layout` emits 24 `shfl.sync.bfly.b32` and 77
`st.shared`/`ld.shared` operations (`shfl.sync.bfly.b32 %r19, %r1, 4, 31, -1;`,
`st.shared.b32 [%r26], %r1;`), with LLIR holding
`tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(…)` and
`@global_smem = external local_unnamed_addr addrspace(3) global [0 x i8]`. Those
whole-kernel counters rise with the effective conversion count across the six
families (0, 1, 2, 4, 5, 7 effective against 0, 16, 12, 38, 49, 77 shared
transfers at one warp), which supports the reading but is not per-conversion
attribution; the per-conversion `.loc` attribution is in the retained-backedge
run above. For Track A the opposite statement holds and is worth stating
plainly: **zero** conversions survive to the final TTGIR, so the PTX shuffle and
shared-transfer counters in those runs are not evidence of conversion lowering.
What they track is the reductions — `cross-state`, the one family with no
`tt.reduce`, reports 0 shuffles, and the counted/unrolled pair reports 20/80
shuffles with 2/8 reductions — and PTX register counts (12–50 per thread, no
spills) are likewise whole-kernel facts, not conversion evidence.

Rematerialization is **explicitly unreached**, not merely unobserved. The
corrected matcher ran over 174 candidate transitions in the eight Track A
compilations and 24 in the twelve Track C compilations: `confirmed` is 0
everywhere. Fourteen positive-delta transitions were inspected as witnesses, and
every candidate is one of two things that are not recomputation. First,
`SCFToControlFlowPass` re-prints a computation whose operands were region-scoped
block arguments: before the pass the exit block reads
`%_sir_out = arith.addf %_sir_acc_x#1, %_sir_acc_x#0` and the body reads
`%_sir_t0 = arith.addf %_sir_acc_x_8, %_sir_acc_y`; after it both read
`arith.addf %_sir_acc_x, %_sir_acc_y`, so one canonical expression goes 0 to 2
while the function still holds exactly two `arith.addf` operations and no
surviving original loses a consumer. Second, `ConvertTritonGPUToLLVM` introduces
three to six `ttg.warp_id {omitUniformHint}` reads — a no-operand hardware query,
not a recomputed value. The earlier withdrawn reading does not reappear: in
these dumps the `InlinerPass`, `CSEPass`, and `SCCPPass` transitions have a
repeated-operation delta of zero or negative, so they add no repeated work at
all. The candidate list, the witnesses with their before/after definitions, and
the proposed generation change that would aim at a real instance are in
`track-a-named-passes.json` and the report.

Two observer limits were found while reading these dumps, and both are recorded
rather than silently repaired. The first was `_TENSOR_ENCODING`, the regular
expression the library used to read the encoding of a `convert_layout`: it
needed an alias immediately before the `>` that closes the tensor type, so
`tensor<4xi32, #blocked2> -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>`
was reported as source equal to target and excluded from the effective count, and
in the four `[4,8]` dumps the library read 4 of 11 conversions as effective while
8 changed encoding. That recogniser has since been replaced by a
bracket-matching scan (`_tensor_encodings`, resolved through
`_resolve_encoding_text` and `_encoding_identity`) that reads an alias, an
encoding printed inline in parameterized form at any nesting depth, and the
generic quoted spelling. It now agrees with the driver's own parser on all 102
Track A conversion occurrences, so the affected counts in this record were
corrected: 46 of the 102 conversions moved from ineffective to effective, the
library-effective total went from 44 to 90 (equal to the record's own 90
eliminated conversions), and the "effective" columns of the fifth-run proposal
table above are the library's own reading, which now agrees with the driver's on
every one of those proposals. The "effective conversions removed by that pass"
column of the first Track A table above is still a total conversion count, and is
still relabelled here as the total conversion count, but no longer because the
library cannot read an inline encoding. `_region_map` could not open a region
whose header carries a counted result list (`%acc_y:2 = scf.for … {`) or a quoted
generic name (`"tt.reduce"(…) ({`), so values inside a function reported `module`
and repackaged regions looked like moves; a fix for that is now in the working
tree from parallel work, and this driver's own region tracker agrees with it on
672 of 672 Track C and 41 of 41 `[4,8]` conversion occurrences. The driver, its
digests, and the exact counterexamples are in
`.planning/2026-10-11-targeted-compiler-behavior/named-passes-driver/`.

## Gaps and controls

* Guard *detection* is verified on the host (a mutated allocation is refused by
  slot name) but not on the device: no run has yet executed a kernel that
  deliberately writes outside its tensor.
* Track C now has device evidence for conversions across a retained backedge
  (`track-c-retained-backedge.json`); no rematerialization is attributed to any
  pass, and no `subsumed` or `register_only` layout pair was reached, so the
  register-only limb of the surviving-conversion evidence still rests on
  per-run register counts.
* The Track C intersection with Track B (padded logical tails under explicit
  layouts) is not implemented: the Gluon emitter still rejects non-power-of-two
  dimensions, and the intersection needs more than that guard (mask-aware
  emission, logical-versus-padded layout validation, physical stride and offset
  addressing).
* Ranks, warps, and layouts beyond the runs above are untouched; the run
  counts here are far below the plan's per-track budgets.
* Two of the four Track A families carry no input proposal at the
  representative's configuration: `reduction-arithmetic` and
  `orthogonal-reductions` are refused by the exact reference over `[0, 8]` with
  four trips, and their proposals are recorded as refusals rather than obtained
  by narrowing the domain.
* The proposal input is chosen by a rule, so its case carries no solver
  selection and the loader's exact comparator is unavailable to it; the
  comparison is zero-tolerance numeric plus a driver-side word-for-word check.
  An exact-mode proposal case would need the loader to accept a non-solver
  witness, which is a library change and was not made here.
* The four proposals of one kernel compile to identical IR, so they add
  numerical and comparison coverage, not layout coverage; the layout and
  rematerialization gaps above are unchanged by this run.
* No new compiler failure was found. The differences reported above are
  compiler behavior, not defects.
