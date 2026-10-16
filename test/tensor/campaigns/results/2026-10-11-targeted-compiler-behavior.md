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

## Track C × Track B: a padded logical tail on an explicit layout tile

The intersection the plan asks for - padded logical tails and physical strides,
with layouts assigned to the power-of-two execution tile - was first established
as blocked, then implemented as mask-aware emission and run on the device. One
logical extent carries the whole run: the rank-two tail `[8, 31]` from the
Track B evidence, on its `[8, 32]` execution tile, with the `fold` region graph
(elementwise fold, reduction over the tailed axis, broadcast of that reduction)
in the bounded-region grammar the Gluon emitter consumes.

### What blocked the combination

Four refusals, recorded verbatim in
`track-c-padded-tail-gluon-evidence.json` (`emitter_refusals`):

| Entry point | Refusal |
|---|---|
| `render_gluon_kernel(source)` | `ValueError: Gluon region lowering requires power-of-two dimensions` |
| `render_gluon_kernel(source, plan)` without a tile | same |
| `BlockedLayoutPlan.validate((8, 31))` | `ValueError: shape dimensions must be positive powers of two` |
| `generate_layout_plans((8, 31))` | `ValueError: shape dimensions must be positive powers of two` |

Two further refusals separate the tail *case* from the tail *shape*:

* Track B's admitted tails are compositional leaf functions, not bounded
  regions. A freshly generated `[8, 31]` case from the committed v3 sampler
  (`rysmith --tensor-profile-version v3 --tensor-generator compositional`, case
  `program.sha256 c00aa0d9…`) is refused by the emitter with
  `ValueError: source is outside the bounded generated-region signature`: the
  plan path can only lower the region grammar, so no Track B tail case can be
  fed to it unchanged.
* The exact-selection arm cannot certify a tailed region either:
  `symirsolve --tensor-profile-version v3` refuses the selection twin
  (`selection-query.sha256 89c3685f…`) with
  `error: tensor-profile: unsupported tensor operation in tensor-v3 profile`,
  because the query selects its target with `tensor extract`, which the v3
  operation set (`add`, `sub`, `mul`, `expand`, `broadcast`, `transpose`,
  `sum`) excludes. The tailed case is therefore admitted by a declared
  in-domain witness instead of a solver selection: the reference is still the
  exact obligation proof of that input (`certify`) replayed by `symiri`, the
  evidence kind is `construction`, and the comparison is zero-tolerance
  numeric. Reading the recorded bytes closes the gap the comparator leaves: the
  run's `actual.bin` equals `expected.bin` byte for byte.

### Mask-aware emission

`render_gluon_kernel(source, *, plan=None, tile=None)` now takes the execution
tile explicitly and keeps it while honouring the logical extent. The tile is
required whenever an extent is not a power of two, so a tailed shape without a
declared tile is still refused; a plan states the tile it assigns its layouts
to (`GluonRegionPlan.tile`), and `BlockedLayoutPlan.validate`,
`generate_layout_plans`, `select_distinct_layouts`, and
`region_plan_families` validate against that tile rather than the logical
extent. Five host tests cover the entry point, the reduction mask, the stored
output, a plan preserved on the tile, and the untiled refusal, with edge tests
for a tile that equals the logical extent (no mask is emitted at all) and a
one-element tail.

For `[8, 31]` on `[8, 32]` the emitter writes one predicate per layout that
reads it, and no padded lane can reach a value:

```
valid_load_layout = (load_rows[:, None] < 8) & (load_columns[None, :] < 31)
acc_x = gl.convert_layout(gl.load(input_ptr + load_offsets, mask=valid_load_layout, other=0.0), compute_layout)
...
v0_reduction_input_1_masked = gl.where(valid_reduction_layout, v0_reduction_input_1, 0.0)
v1 = gl.convert_layout(gl.expand_dims(gl.sum(v0_reduction_input_1_masked, axis=1), 1), compute_layout)
...
gl.store(output_ptr + store_offsets, output, mask=valid_store_layout)
```

The addressing stays on the logical buffer (`load_offsets = load_rows[:, None] *
31 + load_columns[None, :]`), so a padded lane computes an address it never
dereferences. That is the same shape as the counted emitter's v3 kernel
(`tl.load(..., mask=..., other=0.0)`, `tl.sum(tl.where(valid, x, 0.0), …)`,
`tl.store(..., mask=...)`), which is what makes the two arms comparable.

### Device runs

The counted arm is the region case above at `tensor-v3` with a padded physical
layout (row stride 32 at element offset 1, storage span 256) inside a `(3, 5)`
guard envelope, launched through `test.lib.run_triton_tests --mode compiled`.
Its runtime input is the committed proposal `sparse-final-lanes`
(sha256 `775b2ea6…`) from `test.lib.tensor_input_proposals`; the other three
proposals and their hashes are recorded in the JSON.
The plan arm runs the same program through
`test/tensor/gluon/run_generated_region.py` with `--tile 8,32`.

| Arm | Variant | Result | Mask predicate reaching the kernel | Loop | Effective conversions |
|---|---|---|---|---|---|
| counted | padding pattern A | pass, `campaign_valid`, output bytes == reference | `_sir_x_valid = (_sir_x_d0 < 8) & (_sir_x_d1 < 31)` | TTIR/TTGIR hold `scf.for`; LLIR/PTX none | 0 |
| counted | padding pattern B | pass, `campaign_valid`, identical output hash | same | same | 0 |
| counted | PTX reading | — | `setp.ne.b32 %p1, %r7, 31`, 4 predicated ops | 20 warp shuffles | — |
| plan | `no-plan` (sampled layouts) | pass, complete output bits | `valid_load_layout`, `valid_reduction_layout`, `valid_store_layout` | retained in TTGIR, LLIR, PTX | 2 (18 operations) |
| plan | `all-same-layout` | pass | one predicate shared by load, reduction, and store | retained | 0 (8 same-encoding ops) |
| plan | `load-to-compute` | pass | load mask + reduction/store mask | retained | 1 (`load_to_compute`) |
| plan | canary allocation | pass | — | — | — |

Readings:

* Both padding patterns produce the same 992-byte logical output as the
  certified witness output (`e523c11e…`), and the recorded `actual.bin` equals
  `expected.bin` word for word. The guarded allocation (`prefix` 3, `suffix` 5,
  padding `3f800001` / `3f800002` by pattern) is what makes a padded lane that
  leaked into the arithmetic or the store visible.
* The plan-mode kernel was also launched against a canary allocation on both
  sides: the logical output equals the reference bits, the 72 slots past the
  logical extent are unchanged on both buffers, and the input allocation is
  unchanged. A store that ignored the mask would have overwritten canary slots
  and a load that ignored it would have read them.
* The counted arm's loop is a two-trip `scf.for` in TTIR and TTGIR and is gone
  from LLIR and PTX, so its mask is observed against an unrolled body; the plan
  arm keeps a structured loop in TTGIR and a real branch cycle in PTX, which is
  the loop the plan's `all-same-layout` family states a single layout on.
* The plan families keep 47–48 registers per thread with no spills; the wider
  sampled layout in the `no-plan` run reaches 123, so the sampled pool is not
  uniformly cheap at one warp even though nothing spills.

Driver, provenance, and artifacts:
`.planning/2026-10-11-targeted-compiler-behavior/padded-tails/` holds the three
device drivers, the emitted kernels, the region source, and the imported-module
and binary digests; the record is
`test/tensor/campaigns/results/track-c-padded-tail-gluon-evidence.json`
(export revision `43d9e163…` plus three staged `test/lib` overrides, no device
build performed in `/host/refractir-targeted-20261011`).

## Track C × Track B, second run: recorded physical strides in the plan-mode addressing

The padded-tail run above left one thing open: the counted arm staged the tail at
the row stride and element offset its case declared, while the plan arm
addressed a dense logical row-major buffer, so no plan-mode kernel could run
against a padded allocation. A review of that run stated the gap as "ordinary
Triton stride evidence does not cover Gluon addressing". This run records what
the gap costs on the device and closes it: the emitter takes the layout record
the case declares, both arms address one physical allocation inside one guard
envelope, and the two gathered outputs are the same bytes.

### The gap, recorded before any change

The plan-mode kernel of the `strided-offset` case, emitted with its
`all-same-layout` family and without a layout record, addresses the dense
logical buffer:

```
load_offsets = load_rows[:, None] * 31 + load_columns[None, :]
```

The counted kernel `symirc` emitted for the same case addresses the layout the
manifest declares, a row stride of 33 at element offset 1:

```
_sir_x_offsets = 1 + _sir_x_d0 * 33 + _sir_x_d1
```

Both kernels were launched through `test.lib.triton_case_worker` against the
same physical allocation (`(3, 5)` guard envelope, storage span 263). The dense
plan-mode kernel reads padding and writes outside the logical mapping, so the
worker refuses it:

```
RuntimeError: kernel wrote outside its logical output mapping: guards and gaps
differ at ['gap-0', 'gap-32', 'gap-33', 'gap-65', 'gap-66', 'gap-98', ...]
```

Its gathered output (`1c927edd…`) matches the complete reference
(`e523c11e…`) on 233 of the 248 logical slots. The other two layouts fail the
same way, with 8 and 14 changed gap slots and 240 and 234 matching logical
slots. A wrong result alone would be a numerical finding; the guard and gap
writes are what make it a refusal, because the allocation states where the
tensor is.

### Stride-aware addressing

`render_gluon_kernel(source, *, plan=None, tile=None, layout=None)` takes the
case's layout record: the element strides, element offset, storage span, and
axis order that `test.lib.tensor_layout` validates and stages. Every address
comes from `tensor_layout.storage_addressing(layout)`, one entry point that
validates the record through `logical_addresses` and restates the addressing it
declares, so the emitter and the stager cannot disagree about what the record
says. The axis order is checked against the strides there rather than being a
second addressing input: a record whose order contradicts its strides is
refused, as is a record whose farthest logical address leaves its storage span.

The same plan family emits the recorded addressing for all three layouts:

| Case | Record | Emitted load addressing |
|---|---|---|
| `identity-strides-offset` | strides `[32, 1]`, offset 1, span 256 | `load_offsets = 1 + load_rows[:, None] * 32 + load_columns[None, :]` |
| `strided-zero-offset` | strides `[33, 1]`, offset 0, span 262 | `load_offsets = load_rows[:, None] * 33 + load_columns[None, :]` |
| `strided-offset` | strides `[33, 1]`, offset 1, span 263 | `load_offsets = 1 + load_rows[:, None] * 33 + load_columns[None, :]` |

The store states the same addressing as the load. A stride of one states no
multiplication and a zero offset states no addition, so a record that is already
the dense logical row-major layout emits exactly the kernel the emitter writes
without one. The masks are untouched by the record: the load predicate is still
`(load_rows[:, None] < 8) & (load_columns[None, :] < 31)`, so a padded lane
still computes an address it never dereferences.

The CPU replay of the emitter's unit suite stages the logical input at the
recorded addresses through `tensor_layout.stage_buffer` (or `stage_envelope`
with guards), hands the kernel the same view the device worker hands it, and
gathers the stored bytes back into logical order, so the tests prove addressing
by execution rather than by reading the emitted text.

### Device runs

One region (`test/tensor/gluon/programs/padded-tail-8x31-region-v1.sir`, the
`fold` graph at `[8, 31]`, two trips), one input (the committed proposal
`sparse-final-lanes`, sha256 `775b2ea6…`), one guard envelope `(3, 5)` with
padding `3f800001`, and one launch shape (`tensor-v3`, tile `[8, 32]`,
`num_warps` 4, `enable_fp_fusion` false). Both arms run the same `case.json`
through the same worker, so the staging, the gathering, and the guard and gap
checks are one implementation.

| Case | Arm | Exit | Gathered output | Logical slots matching the reference | Guard or gap slots changed |
|---|---|---|---|---|---|
| `identity-strides-offset` | counted | 0 | `e523c11e…` | 248 of 248 | 0 |
| | plan, record `[32, 1]` offset 1 | 0 | `e523c11e…` | 248 of 248 | 0 |
| | plan, no record (control) | 1 | `1aa5f41b…` | 240 of 248 | 8 |
| `strided-zero-offset` | counted | 0 | `e523c11e…` | 248 of 248 | 0 |
| | plan, record `[33, 1]` offset 0 | 0 | `e523c11e…` | 248 of 248 | 0 |
| | plan, no record (control) | 1 | `4f6940c9…` | 234 of 248 | 14 |
| `strided-offset` | counted | 0 | `e523c11e…` | 248 of 248 | 0 |
| | plan, record `[33, 1]` offset 1 | 0 | `e523c11e…` | 248 of 248 | 0 |
| | plan, no record (control) | 1 | `1c927edd…` | 233 of 248 | 15 |

Readings:

* Each plan arm's gathered logical bytes hash to `e523c11e…`, the same digest
  as the counted arm's `actual.bin`, the case's `expected.bin`, the exact
  reference `certify` derives from the same input, and the counted arm of the
  padded-tail run above. The identity-strides case at element offset 1 is the
  layout of that earlier counted case, so the record changes nothing that was
  already right.
* Every arm wrote all 248 logical slots and no guard or gap slot. The dense
  controls wrote 8, 14, and 15 gap slots and left 8, 14, and 15 logical slots at
  the canary, which reproduces the gap record under the change rather than
  quoting it.
* The plan arm's compiled artifacts come from the same worker dump as the
  counted arm's (`compiler-ttir.ir`, `compiler-ttgir.ir`, `compiler-glir.ir`,
  `compiler-llir.ir`, `compiler-ptx.ir`), so the stride reaches the compiled
  kernel rather than only the emitted Python.

### Test discipline

Five tests were written first and run one by one; each failed on the missing
`layout` keyword. After the change each passes on its own.

| Test | First run | After the change |
|---|---|---|
| `test_a_non_identity_stride_layout_addresses_the_recorded_elements` | red: `TypeError: render_gluon_kernel() got an unexpected keyword argument 'layout'` | green |
| `test_a_nonzero_element_offset_is_honoured` | red: same | green |
| `test_a_padded_allocation_keeps_its_guards_and_gaps` | red: same | green |
| `test_the_gathered_plan_mode_result_equals_the_counted_runs_bytes` | red: same | green |
| `test_a_layout_whose_strides_do_not_fit_its_storage_span_is_refused` | red: same | green |

The fourth test binds the host suite to this device record: it replays the
committed program with the `sparse-final-lanes` witness, the counted case's
layout record, and its guard envelope, and requires the gathered bytes to hash
to the counted arm's recorded `e523c11e…`.

The smallest edge case then followed: an identity layout with a zero offset on a
power-of-two region, which emits the kernel the emitter writes without a record
and replays to the same logical output.

### What is reached and what is not

Reached: a plan-mode kernel addresses the element strides and element offset its
case records, runs against the same guarded physical allocation as the counted
arm, reproduces that arm's complete logical bytes for three layouts (identity
strides at offset 1, non-identity strides at offset 0, and both together), and
leaves every guard and internal gap at its canary.

Not reached, with the reason:

* One logical tile shape (`[8, 31]` on `[8, 32]`), one warp count (4), one plan
  family (`all-same-layout`), and one padding sentinel. Other tail pairs, warp
  counts, families, and the second padding pattern of the first run are untested
  here.
* One record per case rather than one per buffer. Both arms address the input
  and the output with the layout the case builder states for both.
* Ranks other than two, and a nonzero `tile_origin`. The region grammar is rank
  two and the case builder states the origin at `[0, 0]`.
* The compositional tail cases of Track B under the plan path, and an
  `exact_smt` selection for a tailed case, both unchanged from the first run.
* Addressing is proved by the gathered bytes and the canary verdicts, not by
  disassembling executed addresses: a kernel that read a guard value and dropped
  it before the sum would still agree.

### Artifacts

`test/tensor/campaigns/results/track-c-physical-strides-evidence.json` holds the
three cases, the nine arm runs with their digests, the emitted addressing lines,
and the digest of every imported `test/lib` module and binary. The drivers, the
emitted plan and control kernels, and the frozen case records are under
`.planning/2026-10-11-targeted-compiler-behavior/physical-strides/`, with the
report at [gluon-physical-strides.md](../../../../.planning/2026-10-11-targeted-compiler-behavior/gluon-physical-strides.md).
The run imported Triton 3.9.0 at `32f06df4…` from `/root/triton-latest/python`,
used `symiri` `4674cc7d…` and `symirc` `42e951d3…` rebuilt in
`/host/refractir-strides-20261016`, and performed no build in
`/host/refractir-targeted-20261011`.

## Track C discovery batch: 128 fresh proposals across eight frozen configurations

The plan's bounded discovery batch for Track C ran on 2026-10-15, 12:13:26Z to
12:33:01Z. The manifest was frozen at 12:00:08Z, before any proposal was drawn,
with digest `f5591654…` and file `sha256 868d65fe…`. It declares eight
configurations (four established strata, four underexplored legal
combinations), a seed formula, the four-input derivation, the environment, and
a retention policy; the batch record carries the manifest digest, so the
allocation cannot have moved after a result was seen.

| # | stratum | plan family | graph family | shape × trips | warps | the evidence it starts from |
|---|---|---|---|---|---|---|
| c01 | established | `all-same-layout` | `fold` | [8, 8] × 1 | 1 | plan sweep at 1 warp: 8 requested, 8 eliminated, 0 effective |
| c02 | established | `load-to-compute` | `fold` | [8, 8] × 1 | 4 | plan sweep at 4 warps: the `load_to_compute` survivor is `cross_warp` |
| c03 | established | `loop-state-layout` | `mixed-depth` | [32, 16] × 4 | 1 | retained backedge at 1 warp: 15 requested, 9 surviving, 2 `scf.yield` operands |
| c04 | established | `reduction-result-expand` | `orthogonal-reductions` | [8, 8] × 1 | 1 | plan sweep at 1 warp: 2 surviving on the graph whose two reductions feed broadcasts |
| c05 | underexplored | `loop-state-layout` | `mixed-depth` | [32, 16] × 4 | 8 | the same plan at a warp count with no evidence (1 and 4 measured) |
| c06 | underexplored | `load-to-compute` | padded tail, authored region | [8, 31] on tile [8, 32] × 2 | 4 | the `--tile` plan arm, measured at one warp only |
| c07 | underexplored | `split-reduction-consumers` | `orthogonal-reductions` | [16, 32] × 2 | 2 | a family earlier runs refused on graph grounds, on a graph that admits it, at an unmeasured warp count |
| c08 | underexplored | `reduction-result-expand` | `orthogonal-reductions` | [16, 32] × 2 | 4 | the family whose declared views were never reached, on a loop-composed region at another trip and warp count |

Each configuration drew 16 proposals, one per selected seed
(`region_seed = 20261016 + 1000·ordinal + index`, `layout_seed = 7680000 +
1000·ordinal + index`), and `c06` drew its 16 on the authored tailed program
with a fresh layout seed each, because the region exporter refuses a tailed
logical extent (`multi-output region input shape is outside the limits`,
recorded verbatim in the manifest's `axis_probe`).

**Denominator.** 128 proposals drawn, **128 admitted, 0 rejected**; 512
executions attempted, **384 pass** and **128 refusals by the compiler**. Every
passing run matched the complete exact f32 reference word for word
(`proof_bound` 16777216, 12–36 obligations), and both calibration controls
behaved as declared in all 384 (`correct-layout-baseline` not detected,
`add-one-output-mutant` detected). Compilations: 512 family kernels attempted
(384 completed, 128 refused before IR) plus 384 calibration mutants completed.
No configuration had all its proposals rejected; no proposal was replaced, and
none needed to be, because the eight configurations were chosen from the
pre-freeze probe of 120 (family, shape, trip) combinations, which recorded the
graph-ground refusals the batch therefore did not draw: `fold`, `mixed-depth`,
`reduction-arithmetic` and the padded tail refuse the three multi-reduction
families (`no value is read by two reduction nodes`, `fewer than two reductions
feed a broadcast`), `cross-state` has no full-shape reduction at all, and
`fold` [32, 16] × 4 and `orthogonal-reductions` [16, 32] × 4 fail exact
admission. Controls and smoke tests are listed in the record and excluded: the
pre-freeze probe, the one-run timing calibration, the tail-extent axis probe,
the per-run `add-one-output-mutant` calibration compilation, and a
four-configuration smoke run whose manifest draws seeds outside the batch's
blocks.

**Program diversity.** The 128 proposals drew 79 distinct
(configuration, program) pairs and 66 distinct program sources. Fresh seeds are
not fresh graphs: `fold` at [8, 8] collapsed 16 seeds onto 6 programs at `c01`
and 4 at `c02`, while `mixed-depth` at [32, 16] × 4 drew 16 distinct programs at
`c05` and 13 at `c03`. `all-same-layout` is also seed-independent by
construction: its 16 proposals realize **one** plan, because the family places
the same baseline layout on every role. Every other configuration realized 16
distinct plans, from 452 distinct sampled layouts and 162 distinct layout
parameter tuples.

| # | executions | loop (TTGIR/LLIR/PTX) | requested | surviving | in loop | folded | unattributed | surviving mapping classes | in-loop lowering (shared/shuffle/barrier) | registers/thread | spills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c01 | 64 pass | none | 596 | 0 | 0 | 0 | 0 | all `identical` | – | 38–39 | 0 |
| c02 | 64 pass | none | 608 | 64 | 0 | 0 | 0 | 28 `cross_lane`, 28 `cross_warp`, 8 `subsumed` | – | 16–52 | 0 |
| c03 | 64 pass | 64/64/64 | 960 | 576 | 384 | 0 | 0 | 540 `cross_lane`, 36 `subsumed` | 13024 / 3568 / 296 | 64–255 | 0–1438 |
| c04 | 64 refused | no IR | – | – | – | – | – | – | – | – | – |
| c05 | 64 pass | 64/60/60 | 960 | 576 | 384 | 0 | 0 | 440 `cross_warp`, 112 `cross_lane`, 24 `subsumed` | 11672 / 300 / 776 | 32–186 | 0–2190 |
| c06 | 64 pass | 64/0/0 | 512 | 64 | 0 | 0 | 0 | 32 `cross_warp`, 24 `cross_lane`, 8 `subsumed` | – | 20–39 | 0 |
| c07 | 64 pass | 64/64/64 | 704 | 320 | 256 | 0 | 0 | 232 `cross_warp`, 80 `cross_lane`, 8 `subsumed` | 4268 / 2208 / 624 | 63–110 | 0 |
| c08 | 64 refused | no IR | – | – | – | – | – | – | – | – | – |

Across the batch: 4340 requested conversions, 1600 surviving, 1024 of those
inside a loop, **0 folded and 0 unattributed**. Only three passes touch a
conversion, and none adds one:

| pass | what it does to the batch's conversions |
|---|---|
| `GluonInline` | moves 468 conversions at `c01` and 480 at `c02` while lowering one loop construct per run; removes and adds none |
| `SCFToControlFlowPass` | moves 704 (`c03`, `c05`), 576 (`c07`) and 384 (`c06`) conversions and lowers the structured loop to a branch cycle; removes and adds none |
| `ConvertTritonGPUToLLVM` | removes every conversion once (596/608/960/512/704 over the configurations) with an effective delta equal to the surviving count: 0, 64, 576, 576, 64, 320 |

The per-conversion readings the record keeps for every run locate the work:

* `loop-state-layout` (c03 at 1 warp, c05 at 8): `acc_x_to_state_layout` and
  `acc_y_to_state_layout` appear twice per kernel, both survive, and the
  copy inside the body is a backedge conversion; the in-loop copies lower to
  4680 and 4296 shared transfers with 176 barriers each, and the state views
  (`region_add_0_left_layout`, `region_add_0_right_layout`,
  `region_sum_1_to_reduction`) add 4864 + 4864 + 5136 shared transfers. At 1 warp
  540 of the 576 survivors are `cross_lane` and 36 are mapping-free `subsumed`,
  with none `cross_warp`; at 8 warps 440 of 576 are `cross_warp`. This reproduces
  the retained-backedge result on 32 fresh programs and extends it to a warp
  count nobody had measured.
* `split-reduction-consumers` at 2 warps (c07) reaches the behaviour the family
  is named for: `region_sum_1_to_reduction` and `region_sum_2_to_reduction` both
  survive inside the retained loop and lower to 1792 and 1968 in-loop shared
  transfers with 176 and 184 barriers, while the two result conversions add
  632 + 1020 shuffles. The family that the six-node graphs refused on graph
  grounds runs on a graph that admits it, at a warp count with no prior
  evidence.
* `load-to-compute` is the only survivor in c02 and c06 (one per kernel) and it
  never sits in a loop in either: `c02`'s trip count is 1, and `c06`'s two-trip
  tailed loop is present in TTGIR (64/64) but gone from LLIR and PTX. All 64 of
  c06's runs match the complete exact reference, so its masked load
  (`other=0.0`), masked reduction operand and masked store hold at 4 warps on 16
  fresh layout seeds; the two padding patterns the plan gives a tail program
  belong to the physical allocation, which this route cannot stage because it
  addresses the dense logical buffer, so they are measured on the counted route
  instead (512 pattern pairs, all 512 agreeing, in
  `track-b-discovery-batch.json`) and the record states the split in its
  `provenance.tail_padding_patterns` block.
* 384 in-loop shared transfers sit under `.loc 1 0 0` and are reported as
  unattributed to a conversion line rather than assigned to one; the rest of the
  in-loop traffic is attributed line by line.

### The batch's headline: `reduction-result-expand` no longer compiles

Both configurations that ask for `reduction-result-expand` fail **all 128
executions** (32 programs × 4 inputs) the same way, and the failure is the same
one a four-configuration smoke run produced on this driver before the batch:

```
triton.compiler.errors.CompilationError: at 22:16:
    v1_view_blocked_0180 = gl.convert_layout(v1, blocked_0180)
    v3, _ = gl.broadcast(v1_view_blocked_0180, acc_x)
            ^
Layout mismatch in broadcast: BlockedLayout(size_per_thread=[8, 8], threads_per_warp=[1, 32], warps_per_cta=[1, 1], order=[0, 1], cga_layout=[]) vs BlockedLayout(size_per_thread=[4, 2], threads_per_warp=[4, 8], warps_per_cta=[1, 1], order=[0, 1], cga_layout=[])
```

The family's consumer views are keyed on `(value, consumer node)` and land on
the *first operand* of `gl.broadcast`, while the second operand keeps the
compute layout; Gluon requires the two operands of a block-level broadcast to
carry the same layout (`triton/experimental/gluon/language/_semantic.py:265`),
so the frontend refuses the kernel before any IR exists. The plan sweep did not
see this because its program's view keys never matched a queried pair, which the
sweep recorded as an unreached family feature; on 32 fresh graphs the views do
match, and the emitted kernel is invalid. This is a defect in the experimental
emitter route (`test/lib/tensor_gluon.py`'s family definition and the emitter's
per-consumer view placement), **not** a compiler defect: the compiler is right
to reject the program. The report records the reproduction, the mechanism, and
the two candidate fixes (state the view on the broadcast result rather than on
one operand, or refuse the family when the consuming node is a broadcast).

Artifacts: `test/tensor/campaigns/results/track-c-discovery-batch.json` (the
compact record; the full 21.6 MB record `sha256 a72f328d…`, the proposal and run
ledgers, and every run's generated kernel, per-stage IR and MLIR pass dump stay
under `/host/trackc-discovery/out`), the manifest at
`.planning/2026-10-11-targeted-compiler-behavior/discovery-track-c/manifest.json`,
the six drivers beside it, and the report at
`.planning/2026-10-11-targeted-compiler-behavior/discovery-track-c.md`. Storage:
387 MB on the host after gzipping each raw IR file and hard-linking identical
copies, against the 2 GB the manifest declared.

## Track B discovery batch: 128 fresh proposals across eight frozen configurations

The plan's bounded discovery batch for Track B froze eight configurations - four
established connections and four underexplored legal combinations - with sixteen
fresh seeds each, and drew all 128 proposals. The manifest is
`test/tensor/campaigns/results/track-b-discovery-batch-manifest.json`
(`9396a530…`, frozen 2026-10-15T12:09:39Z; the driver recomputes the digest and
records that it was unchanged during the run), and the record is
`test/tensor/campaigns/results/track-b-discovery-batch.json`.

Each proposal is a freshly seeded rysmith tensor-v3 constructor run; its program
is the case at the configuration's scheduled case index, so the schedules choose
the rank-one tails `17/31/33/63/65` and the ten rank-two pairs whose padded
tiles are exactly the 512-element profile volume boundary. Admission is the
case's own complete interpreter reference reproduced bit-identically by two
native replays at 128 and 256 reference bits. The runtime element is i32, so the
on-device comparison is exact and no rounding tolerance can excuse a misplaced
lane.

| Configuration | Class | Connection | Shapes | Admitted | Executions | Loop | Layout-changed conversions |
|---|---|---|---|---|---|---|---|
| `est-tail-sum-rank2` | established | reduction | 10 boundary pairs | 16/16 | 128 pass | none | 23 (1-2 per program) |
| `est-tail-broadcast-rank2` | established | broadcast | 10 narrow `[1,n]`/`[n,1]` | 16/16 | 128 pass | none | 9 (0-1) |
| `est-tail-join-rank1` | established | join | 5 rank-one tails | 16/16 | 128 pass | none | 0 |
| `est-tail-layout-rank2` | established | transpose | 10 boundary pairs | 16/16 | 128 pass | none | 16 (exactly 1) |
| `und-tail-loop-rank2` | underexplored | tail + loop | 10 boundary pairs | 16/16 | 128 pass | TTGIR 16/16, PTX 4/16 | 23 |
| `und-tail-join-rank2` | underexplored | join on boundary tails | 10 boundary pairs | 16/16 | 128 pass | none | 0 |
| `und-tail-layout-rank1` | underexplored | copy on a rank-one tail | 5 rank-one tails | 16/16 | 128 pass | none | 0 |
| `und-envelope-placement-b` | underexplored | second envelope placement | 10 boundary pairs | 16/16 | 128 pass | none | 23 |

Budget spent: 128 proposals drawn, 128 admitted, 0 refused, so no configuration
lost all its proposals; 512 input variants admitted and 0 refused; 1024 run
directories built and 1024 executions attempted, completed, and passed with
`campaign_valid`; 1024 compilations, because every run emits and compiles its own
kernel from an empty Triton cache; one control execution. The executions are 128
programs by four input patterns by two padding patterns; 896 carry the `(3,5)`
envelope and 128 the `(5,3)` one.

Readings:

* Every logical output equals its own complete reference (396,400 logical slots
  compared over the batch), and the two padding patterns produce byte-identical
  logical outputs for all 512 program/input pairs.
* Guards and gaps held in all 1024 executions. The verdicts were re-derived from
  the archived `actual-physical.bin`, and 123,504 nonlogical slots - 16,384 guard
  slots outside the storage span and 107,120 internal gap slots inside it -
  stayed at their canary in every run; 1000 of the 1024 runs had at least one
  internal gap slot.
* The seeds, not the schedules, carry the storage variety: the constructor source
  is a function of the stratum and the case index, so the 128 proposals hold 60
  distinct programs but 124 distinct `(program, input strides/offset, output
  strides/offset, num_warps)` configurations. Per configuration the batch covers
  7-16 distinct input layouts and 6-16 distinct output layouts over all four warp
  counts (1: 29 programs, 2: 38, 4: 34, 8: 27), and 86 of the 128 inputs carry a
  nonzero element offset.
* The only loop in the batch is `und-tail-loop-rank2`'s three-trip counted loop:
  present in TTIR and TTGIR for all 16 programs, and still a branch cycle in LLIR
  and PTX for four of them (`[16,17]` at 4 warps twice, `[8,63]` at 2 warps,
  `[4,65]` at 1 warp). The other twelve are straight-line by LLIR.
* 94 layout-changed conversions survive to TTGIR across the batch, all inside
  `tt.func`, none in a loop body, and none visible in TTIR, LLIR, or PTX: 48 move
  a slice layout to a blocked layout and 21 move back, 17 and 8 move between
  blocked layouts. The reduction, transpose, loop, and placement-B
  configurations carry them (23, 16, 23, 23); the arithmetic joins and the
  rank-one copy carry none.
* The intentionally incorrect in-allocation control - one element written inside
  the output prefix guard of a copied passing run - is refused by the worker with
  `prefix-guard-0` named, and is excluded from the denominator.
* No new compiler failure: zero output mismatches, zero storage violations, zero
  refusals, and no configuration whose proposals were all rejected. Three runs
  (a transposed tail with padding B, a loop-stratum program, and a
  second-placement program) were re-checked from the artifacts alone - record
  digests recomputed from disk, the guard and gap verdict re-derived from
  `actual-physical.bin`, and both reference precisions replayed from the
  interpreter - and all three agree with the record.

Two limits of the route were recorded rather than worked around. A rank-one
reduction returns a scalar, and the guarded storage route cannot stage a scalar
output: `_initial_output` allocates `storage_span` elements for an empty output
shape and then hands the kernel a view at the guard prefix. The pre-batch
qualification reproduced the illegal memory access under the envelope beside the
same program passing unguarded, so no configuration in the batch produces a
scalar output and rank-one reduction tails remain a route gap. Input immutability
is enforced by the worker on every run and no run refused, but the worker
archives only the output allocation, so the record shows the check rather than
the input bytes.

Driver, manifest, and raw record:
`.planning/2026-10-11-targeted-compiler-behavior/padded-tails/drivers/` holds the
batch driver, the qualification probe, the conversion observer, and the reduction
that writes the review record; the unfiltered driver output (9.5 MB, `5c6dc2cc…`),
the control record, the generation summary, and the batch log are archived under
`test/tensor/campaigns/archive/2026-10-17-track-b-discovery/`. The export is the
committed revision `47d7ceb7…`, built in its own copy under
`/host/track-b-disc-20261017` and never in `/host/refractir-targeted-20261011`,
with Triton `3.9.0+git32f06df4` from `/root/triton-latest/python` on eight
V100-SXM2-32GB devices. The record carries the compiler flags, the four native
tool digests, and the sha256 of every `test/lib` module of the import root.

## Track A discovery batch: 128 fresh proposals across eight frozen configurations

The plan's bounded discovery batch for Track A froze its allocation on
2026-10-15 at 11:57:31Z, digest `68880da39fd718e9cb93930a5de4e90bdd19518ba25af028391ed5ca7a128cc3`,
before any proposal was generated, and drew all 128. The record is
`test/tensor/campaigns/results/track-a-discovery-batch.json` (`sha256 2456e787…`,
7.5 MB compact; the full 17.5 MB record `sha256 eec15e0d…` is at
`/host/track-a-fix-089ee0c/work/`), the frozen
manifest is `.planning/2026-10-11-targeted-compiler-behavior/discovery-track-a/manifest.json`,
the driver is `test/tensor/campaigns/run_track_a_discovery.py` (`eb173081…`), and
the report is `.planning/2026-10-11-targeted-compiler-behavior/discovery-track-a.md`.

**Correction (2026-10-16).** The record above was re-derived from the same 512
saved case directories by a corrected driver, which now attributes every removed
conversion to the loop that enclosed *it* instead of to any loop present in the
module, and which merges a certified case's word-for-word byte check into its
execution and comparison verdict. The record carries a `corrections` block with
the superseded numbers and both driver digests. The runs themselves are
unchanged — same cases, same directory tree, same 512 device executions, all 512
still pass — but the batch's observation rate is not what was first published:
**384 of 512** runs observe the target behavior, not 512, and the 128 runs of the
two `cross-state` configurations observe none. The superseded figure was
`sha256 e5665877…` of the record, produced by driver `94a08acb…`, reporting
512/512. Where the two numbers differ below, the superseded one is stated as
such.

**Second correction (2026-10-16, record unchanged).** A review of the same driver
closed two defects in the byte verdict. The verdict reads each case's comparison
contract from the case's own certificate rather than from the build step's
summary of it, so a case an interrupted build reused reaches the same verdict as
one built fresh. A case with no output buffer keeps its execution failure class
(`compile_failure`, `runtime_failure`, `runtime_timeout`) with the comparison
`unavailable`, instead of restating every nonpassing byte check as a comparison
failure. Neither defect is exercised by this batch: all 512 inputs were built in
this run, none reused, and all 512 produced an output whose byte check passed, so
every recorded value stands and the record was not regenerated.

| # | half | family | shape × trips | mode | domain | the evidence it starts from |
|---|---|---|---|---|---|---|
| E1 | established | `mixed-depth` | [4, 8] × 3 | `additive` | [0, 2] | first run and named-pass study: bound 3, four warps, exact admission |
| E2 | established | `cross-state` | [4, 8] × 2 | `additive` | [0, 2] | same evidence: bound 2 |
| E3 | established | `orthogonal-reductions` | [4, 8] × 2 | `additive` | [0, 2] | same evidence: bound 2 |
| E4 | established | `reduction-arithmetic` | [4, 8] × 2 | `additive` | [0, 2] | same evidence: bound 2 |
| U1 | underexplored | `mixed-depth` | [4, 8] × 2 | `subtraction` | [0, 2] | an arithmetic mode no recorded Track A run uses |
| U2 | underexplored | `cross-state` | [4, 8] × **8** | `additive` | [0, 2] | a trip count at the profile maximum, on the one family with no reduction |
| U3 | underexplored | `orthogonal-reductions` | [32, 16] × 3 | `additive` | [0, 2] | a shape at the 512-element cap, re-drawn from 16 fresh seeds |
| U4 | underexplored | `mixed-depth` | [128, 4] × 2 | `additive` | [0, 2] | a family at a shape every other recorded run of it avoids |

Each configuration drew 16 proposals, one per seed
(`seed = 100000 + 1000·configuration_index + offset`), all disjoint from the
recorded Track A seeds 0–9. Launch is held at `num_warps` 4, `num_stages` 1,
`enable_fp_fusion` false, grid `[1, 1, 1]`, so a difference between the
configurations is a difference of generated program rather than of launch
setting.

**Denominator.** 128 proposals drawn, **128 admitted, 0 rejected** — every one
by the same probe the recorded family runs use, the exact f32 certificate over
the whole frozen domain. No proposal was replaced and none needed to be. 512
inputs derived and built, 0 refused; 512 compiled-mode executions attempted, all
512 passed with `campaign_valid` and a word-for-word byte check, and no
reference output is constant. 512 compilations, each from its own empty Triton
cache, so no cache reuse is visible. Stage wall clock: propose 37.5 s, build
167.1 s, run 420.2 s over eight shards. The batch was kept bounded by keeping
solver selection off the path entirely — admission is the pure-Python exact
bound check, so the six-minute 512-lane solve is never incurred — by putting 96
of the 128 proposals on 32-element shapes and only 32 on the two 512-element
configurations, and by raising the interpreter replay budget in the driver only
(`build_tensor_exact_loop_case.TIMEOUT_S = 900 s`). Controls and smoke tests are
recorded separately and excluded: the two seed-9 programs at E1 and U3 that the
first run and the 512-element matrix already carry, re-derived through the same
path, which reproduce their recorded loop behavior (E1 unrolled by LLIR, U3
retained with one PTX backedge) in all eight control executions.

**Observations.** 384 of the 512 cases are `target-behavior-observed`, and all
for the same reason: `TritonGPURemoveLayoutConversions` removes layout
conversions *from inside the loop*, while exactly one loop construct is present
before and after it. The published figure of 512/512 counted every removal the
pass made while any loop existed in the module; re-attributing each removal to
its own enclosing region moves 4,224 of the 6,032 removals to function scope, and
in both `cross-state` configurations every one of the eight removals per
compilation is at function scope, so their 128 runs observe nothing. The removal
counts, the part of them that came from inside the loop, and what survives them
(the published 512/512 is superseded; the "in loop" column is new):

| # | removed by `RemoveLayoutConversions` (median, [min, max]) | of which inside the loop (median, [min, max]) | effective | observing runs | surviving to final TTGIR | TTIR/TTGIR loops | device loop (LLIR+PTX backedge) | PTX shuffles / shared |
|---|---|---|---|---|---|---|---|---|
| E1 | 12 [11, 12] | 4 [3, 4] | 9 | 64/64 | 0 | 64/64 each | 0/64 | 9 / 0 |
| E2 | 8 [8, 8] | **0 [0, 0]** | 5 | **0/64** | 0 | 64/64 each | 0/64 | 0 / 0 |
| E3 | 15 [15, 15] | 7 [7, 7] | 12 | 64/64 | 0 | 64/64 each | 0/64 | 10 / 0 |
| E4 | 11 [11, 13] | 3 [3, 5] | 8 | 64/64 | 0 | 64/64 each | 0/64 | 4 / 0 |
| U1 | 12 [11, 12] | 4 [3, 4] | 9 | 64/64 | 0 | 64/64 each | 0/64 | 6 / 0 |
| U2 | 8 [8, 8] | **0 [0, 0]** | 5 | **0/64** | 0 | 64/64 each | 0/64 | 0 / 0 |
| U3 | 16 [16, 16] | 7 [7, 7] | 16 | 64/64 | 0 | 64/64 each | **64/64** | 22 / 2 |
| U4 | 13 [12, 13] | 4 [3, 4] | 13 | 64/64 | 0 | 64/64 each | 0/64 | 56 / 4 |

Published, every configuration stood at 64/64; corrected, six still do (E1, E3,
E4, U1, U3, U4) and E2 and U2 are 0/64. The record keeps both readings: the
superseded counts sit in `observation_outcomes_superseded` on each configuration
that moved, and the top-level `corrections` block names the record and the driver
digest behind each number.

Four named passes touch a conversion anywhere in the batch, and none adds one
after the first two: `ConvertTritonToTritonGPU` added 4368 (8.5 per
compilation), `TritonGPUCoalesce` added 1664 and preserved all 4368,
`TritonGPURemoveLayoutConversions` removed 6032 at its first occurrence
(11.8 per compilation) with `loop_construct_delta` 0 in all four of its
occurrences, and `SCFToControlFlowPass` lowered one loop construct per
compilation (512 in total) while moving no conversion. Of those 6032 removals,
**1808 (3.5 per compilation) were inside the loop that enclosed them and 4224
were at function scope** — the split the corrected driver records per pass
occurrence as `removed_in_loop_regions` and `removed_outside_a_loop`. The
difference between "removed" and "effective" is the same-encoding no-op the
named-pass study identified: at `[4, 8]` three conversions per compilation carry
no layout change, which is why E1 removes 12 and effectively removes 9; U3 and
U4 remove only conversions that change an encoding.

The other three Track A categories are zero in all 512 cases: no conversion
survives to the final TTGIR, none changes its enclosing region across a pass,
and no pass confirms a rematerialization. For 384 of the 512 runs the in-loop
removal above is therefore the whole observation; for the other 128 — every run
of E2 and U2 — the batch now records `not-observed`, with the reason "8 layout
conversion(s) removed by `TritonGPURemoveLayoutConversions`, none from inside a
loop". Nothing else about the batch changes: the executions still pass, and the
in-loop category is still the only one reached. The new reading is U3:
`orthogonal-reductions` at [32, 16] with three trips and four warps keeps a
device loop in 64 of 64 fresh cases, so the "layout decision removed inside a
loop that reaches the hardware" configuration is reachable from a second family,
not only from the `mixed-depth` representative the earlier runs used.

**Program diversity, measured.** The 128 proposals drew **91 distinct
programs**, and the repeats are not uniform: E1 14, E2 10, E3 12, E4 13, U2 10,
U3 12, U4 16, and **U1 only 4 of 16**, where one program is drawn five times.
The subtraction mode at this shape and trip count collapses the graph space, so
a batch that wants diversity from that mode has to select graphs rather than
seeds. The four inputs of one proposal are four numerical runs of one program,
measured rather than assumed: for all 128 proposals the path-normalized TTIR,
TTGIR, and LLIR digests are identical across the four inputs, and only PTX
differs, because its debug section encodes the length of the embedded case path.
The batch's layout coverage is therefore 128 programs (91 distinct), not 512
compilations.

**Provenance.** RefractIR revision `47d7ceb7…`, exported with
`git checkout-index -a --prefix=<dir>/`, built in its own copy at
`/host/track-a-discovery-20261015/refractir` and never in
`/host/refractir-targeted-20261011`; the resulting `symirc` digests to
`5915fbda…`, the same binary the earlier Track A and Track C runs recorded.
Triton `3.9.0` at `32f06df4bf1c30885ff61ee3488836ca81ed65f0` from
`/root/triton-latest/python`, Python 3.12.15, torch 2.5.1+cu121, driver
535.274.02, CUDA 12.1, eight Tesla V100-SXM2-32GB. The record carries the
compiler flags, the three native tool digests, and the sha256 of every imported
`test/lib` module — `tensor_case.py` `67643011…`, the same digest the recorded
runs imported — plus the digest of the driver. The corrected record differs from
the published one in exactly one imported module: `tensor_compiler_observation.py`
`fe1cb5ea…` (superseded `9e3485a1…`), the module that now attributes each removal
to its own enclosing region. The driver digest is `eb173081…` (superseded
`94a08acb…`); the native tools, the frozen manifest, the cases, and the controls
are unchanged.

**No new compiler failure.** 512 executions, 512 passes, 0 mismatches, 0
compile failures, 0 refusals, and no configuration whose proposals were all
rejected. The batch extends the established behavior over 128 fresh programs and
four underexplored legal combinations; it does not add a defect, and it adds no
evidence to the surviving-conversion, movement, or rematerialization categories.
The byte-check verdict is authoritative for these certified inputs, and is read
from each case's own certificate: all 512 cases carry `byte_check_applied` true
and `byte_check_status` `pass`, and a signed-zero or any other word-level
difference would fail the execution and the comparison instead of passing them
(no such case is present in this batch). A case that produced no output would be
read differently: an execution that already failed keeps its class
(`compile_failure`, `runtime_failure`, `runtime_timeout`) with the comparison
`unavailable`, and a run that reports success without an output is a failure of
the comparison; no case of either kind is present in this batch.

## A broadcast-hoist family: the rematerialization candidate, measured

The named-pass run left rematerialization unreached and proposed a generation
change: place a broadcast of a loop-carried value in the retained loop body,
give its result two consumers so the original survives one of them, and let the
other ask for a second layout. That change is implemented as the sixth
loop-region family, `broadcast-hoist` (`rysmith --tensor-region-family
broadcast-hoist`), and this run measures it on the device.

The family's body is fixed: the shared two-state producer, a reduction of the
state that carries the runtime input, the broadcast of that one-extent result, a
second reduction of the view on the other axis, the re-broadcast, and the two
root joins. The view therefore has exactly two consumers, the second reduction
and the elementwise path into a root. That is what the pass's two
rematerialization entry points need to see at the conversion they act on, and the
TTGIR below confirms the two consumers; what it does not confirm is a conversion
*on* that broadcast.

Evidence: `broadcast-hoist-rematerialization.json`, produced by
`.planning/2026-10-11-targeted-compiler-behavior/remat-driver/`.

**Correction (2026-10-16).** The consumer counts first published here were
re-derived and one claim was withdrawn. The driver counted every SSA name on a
line, including the defining left-hand side, so each conversion's source was
reported with one consumer too many: all six opportunities per family arm read
`source_consumers: 2` and all six have exactly **one** consuming operation. The
broadcast first cited as the family's pattern
(`tt.broadcast %_sir_x_offsets_5`) is an `i32` address computation with one
consumer, not the generated `f32` state view. The intended view is now located
in TTGIR by walking the operand spine down to its floating-point reduction, and
the record states it per boundary. The superseded counts and the superseded
comparison are listed in the record's `superseded` block, which now also keeps
the per-arm compiler, environment, qualification, and dataset digests and the
certificate refusal as a first-class arm.

| Case | Selection | Execution | Conversions created / removed by the first `RemoveLayoutConversions` | Conversions on an extend or broadcast at that boundary | Generated state view at that boundary | Matcher | Structural recomputations |
|---|---|---|---|---|---|---|---|
| `broadcast-hoist` `[4,8]` × 4 | solver | pass | 12 + 3 created, 15 removed | 6 | present: `tt.broadcast %_sir_t2` (f32), 2 consumers, no conversion reads it | 0 confirmed | 0 |
| `broadcast-hoist` `[32,16]` × 2 | declared witness | pass | 12 + 4 created, 16 removed | 6 | present: `tt.broadcast %_sir_t2` (f32), 2 consumers, no conversion reads it | 0 confirmed | 0 |
| `mixed-depth` `[32,16]` × 4 (control) | declared witness | pass | 8 + 4 created, 12 removed | 4 | present: `tt.broadcast %_sir_t3` (f32), 1 consumer, no conversion reads it | 0 confirmed | 0 |
| `broadcast-hoist` `[32,16]` × 4 | — | refused | — | — | — | — | — |

All three executed arms hold one `scf.for` through every layout pass and lose it
at `SCFToControlFlowPass`, so the layout decision is made while a loop exists.
The control's program is byte-identical to the archived counted case
(`638322a9be95072a`); the two family arms are new programs
(`206f869cbddffdec`, `b9b37942fca352f1`).

The requested configuration, `[32,16]` over four trips, is refused by the
family's exact certificate: feeding a doubly reduced value back into a loop
state multiplies the state by the product of both extents on every trip, so the
bounds proof admits one and two trips at that shape (largest partial sum
2,101,248 at domain `[0,8]`) and refuses three and four (`exceeds exact range`).
The four-trip arm therefore runs at `[4,8]`, the shape the archived family cases
use, and the refusal has its own arm in the record
(`broadcast-hoist-32x16-it4`, `status: certificate_refused`, reason
`materialized value or reduction partial sum exceeds exact range`) rather than
being inferred from a missing case; the record's `superseded` block names it as
an arm the first summary did not carry. The control admits the requested
configuration (10,976,000 against the 16,777,216 exactness bound).

What the pass did with the pattern:

* The family's own TTGIR contains the view before the pass, and the view is not
  the broadcast first cited here. In the `[4,8]` arm the generated `f32` state
  view is `%_sir_t2_18 = tt.broadcast %_sir_t2 : tensor<1x8xf32, #blocked> ->
  tensor<4x8xf32, #blocked>`, and its two consumers are exactly the second
  reduction (`"tt.reduce"(%_sir_t2_18) <{axis = 1 : i32}>`) and the shallow root
  join (`%_sir_next_y = arith.addf %_sir_acc_y_13, %_sir_t2_18`); the `[32,16]` × 2
  arm is the same shape. **No conversion reads that broadcast.** The conversion
  the pipeline inserts for this path sits one step below it, on the view's
  operand: `%_sir_t1_17 = tt.expand_dims %_sir_t1_16`, whose only consumer is
  `%_sir_t2 = ttg.convert_layout %_sir_t1_17 : #blocked4 -> #blocked`. So the
  candidate the hoist path sees is a conversion on the one-extent expansion, not
  on the two-consumer broadcast the proposal asked for. The superseded sentence
  ("a broadcast result with two consumers feeds a conversion", citing
  `%_sir_x_offsets_6`) counted the `i32` broadcast's own definition as its second
  consumer; that broadcast has one consumer, the conversion reading it, and it is
  part of the address chain.
* Six conversions on an extend or broadcast exist at the boundary in each family
  arm and four in the control, as before; in the family arms three of the six are
  in the family's own chains (the two expansions `%_sir_t2`, `%_sir_t4` and the
  re-broadcast feeding `%_sir_next_x`) and three are the `i32` offset
  conversions. In the control, the state broadcast (`tt.broadcast %_sir_t3`,
  1 consumer) has no conversion either.

* `TritonGPURemoveLayoutConversions` removes every one of them in its first
  occurrence (15, 16, and 12 conversions), with a duplicate-expression delta of
  zero and no repeated definition anywhere in the transition. The pass's first
  step is forward layout propagation from the load anchor, which rewrites the
  whole loop into the anchor's layout and erases those conversions in place. No
  value is recomputed, so no candidate transition has a repeated definition to
  classify: the structural reading (one operation recomputed on the same
  operands or on conversions of them, original surviving, a served consumer
  taken over by the copy) finds nothing in any arm.
* The matcher confirms nothing: 0 confirmed in all three arms across 19
  candidate transitions each, and 0 unattributed duplicates at the boundary in
  question.

Two control measurements fix what a confirmable event would need, and neither is
device evidence:

* The matcher's own criterion, read on a synthetic pair with the committed
  library (`test/lib/tensor_compiler_observation.py` `08c4772a…`): a broadcast
  that survives for one consumer while a copy takes the converted consumer over
  is confirmed both when the copy keeps the original's encoding
  (`same_encoding_copy`, `confirmed_only` 1, `encoding_change.changed` false) and
  when it carries the encoding the conversion asked for
  (`layout_rematerialization`, `confirmed_only` 1, `encoding_change.changed`
  true). The structural reading finds the recomputation in the second pair (a
  cloned `tt.broadcast` with the same operand, the original losing the conversion
  and its consumer, that consumer reading the copy), so the layout
  rematerialization shape is now confirmable and the obstacle that remains is the
  graph rather than the matcher's equivalence. The first published reading of
  that pair, taken with the library the runs imported (`fe1cb5ea…`), refused it
  (`none`, 0); the record keeps both values in its `superseded` block and states
  in `matcher_rederivation` that the three device arms were re-read with the
  committed library without one matcher or structural field moving (0 confirmed,
  0 structural recomputations, 19 candidate transitions per arm, as before).
* Ten hand-written kernels in the family's operation set, compiled on the same
  device with pass dumps, found no configuration in which the pass recomputes
  anything. The two kernels whose load and store anchor different layouts
  (`store-transposed`, `load-transposed`) keep exactly one conversion alive past
  the pass, `%out` from the loop result, and the pass leaves it alone:
  `getConvertBackwardSlice` refuses to propagate through an `scf.for` result, so
  a conversion whose source is a loop result is out of the rematerialization
  path by construction.

The corpus is additive: 2,800 configurations over the five families that existed
before the change (seeds 0, 1, 5, 9, 13 by seven shapes by four bounds by four
arithmetic modes) are byte-identical between an exporter built from the tree
before the change and one built after it, and the fold bodies are among them.

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
  layouts) is now implemented and run for one shape: the emitter takes a
  declared power-of-two tile, masks the load, the reduction, and the store, and
  both arms compute the complete logical reference on the device. What is *not*
  reached: the plan path consumes the bounded-region grammar only, so Track B's
  compositional tail cases still cannot be lowered by it; the exact-selection
  arm cannot certify a tailed region, because tensor-v3 excludes the
  `tensor extract` the selection query uses, so the tailed case is admitted by a
  declared witness with a zero-tolerance numeric comparison and a driver-side
  byte check rather than by `exact_smt` selection; the plan arm runs against a
  dense logical allocation, so the *physical* strides and offsets that the
  counted arm uses are not yet stated to the Gluon emitter, whose addressing
  stays on the logical row-major buffer; and one shape at one warp does not
  cover the six planned tail pairs, other warp counts, or the plan families that
  need more than two layouts.
* Ranks, warps, and layouts beyond the runs above are untouched; the run
  counts here are far below the plan's per-track budgets.
* The Track A discovery batch adds no failure and no new layout behavior: its
  512 executions all pass, and surviving conversions, conversion movement, and
  rematerialization are zero in all of them. It observes the target behavior in
  384 runs, not 512: in the two `cross-state` configurations (E2 and U2) the
  pass removes eight conversions per compilation and every one of them is at
  function scope, so those 128 runs are recorded `not-observed`. The published
  512/512 rate is superseded, and the batch's headline is 384/512. The
  rematerialization question is still unreached, now over 128 more programs.
* Track A's batch drew 91 distinct programs from 128 seeds, and one
  configuration (`mixed-depth` with `subtraction` at [4, 8] and two trips) drew
  only 4, so its 16 proposals are largely the same program re-run. Only one
  arithmetic mode and one trip count were frozen for that family, so the run
  cannot separate the mode from the shape and bound it was drawn at.
* Track A's four inputs per program compile to the same structure: the TTIR,
  TTGIR, and LLIR digests are identical across the four inputs of all 128
  proposals (PTX differs only in its embedded path string). That is repeated
  compiled structure, not cache reuse - the batch records 512 compilations from
  512 caches that were empty at the start - so the layout coverage the batch can
  attribute is 91 distinct programs, while the remaining executions are
  numerical and comparison coverage.
* Two of the four Track A families carry no input proposal at the
  representative's configuration: `reduction-arithmetic` and
  `orthogonal-reductions` are refused by the exact reference over `[0, 8]` with
  four trips, and their proposals are recorded as refusals rather than obtained
  by narrowing the domain.
* The proposal input is chosen by a rule, so its case carries no solver
  selection and the loader's exact comparator is unavailable to it; the
  comparison is zero-tolerance numeric plus a driver-side word-for-word check.
  An exact-mode proposal case would need the loader to accept a non-solver
  witness, which is a library change and was not made here. The padded-tail run
  above did make the narrower version of that change for region cases
  (`build_tensor_exact_region_case.build_case(selection="witness")`), and it
  uses the same zero-tolerance numeric comparison plus a byte check.
* The four proposals of one kernel compile to identical IR, so they add
  numerical and comparison coverage, not layout coverage; the layout and
  rematerialization gaps above are unchanged by this run.
* No new compiler failure was found. The differences reported above are
  compiler behavior, not defects.
* The discovery batch added no compiler failure either: its 128 refusals are
  the Gluon frontend rejecting an invalid kernel the `reduction-result-expand`
  family emits, which is a defect of the experimental emitter route. Until that
  family is fixed, the batch carries no device evidence for it at all, so the
  plan family has no surviving-conversion reading in any configuration.
* Rematerialization is still unreached: the batch's pass transitions move and
  remove conversions but never recompute one, and no `register_only` or
  `equivalent` pair appears in the 1600 surviving conversions.
* The batch's four inputs per program are four numerical runs of one
  compilation: 96 of 96 programs with passing runs produced byte-identical IR
  for all four inputs, so the batch's layout coverage is 79
  (configuration, program) pairs, not 512 compilations.
* Fresh seeds are not fresh graphs — `fold` at [8, 8] drew 6 and 4 distinct
  programs from 16 seeds each — so a future batch that wants graph diversity
  should measure it rather than assume it from the seed count.
* Cross-lane and cross-warp attribution inside a loop is line-granular: 384
  in-loop shared transfers in the batch's `loop-state-layout` runs sit under
  `.loc 1 0 0` and stay unattributed to a conversion. An instruction-level
  attribution would need liveness or disassembly evidence the runner does not
  collect.
* The Track B batch's 1024 executions compile 119 distinct kernels, not 1024:
  every one of a program's eight runs (four inputs by two padding patterns)
  emits the same kernel and is compiled again from an empty cache, so the
  `compilations` count measures the route's work rather than distinct code.
  The same effect appears one level up: 128 fresh seeds drew 60 distinct
  programs, because the v3 constructor's source is a function of the stratum and
  the case index, and the physical layout is what the seed varies.
* The storage variety the batch bought is real and separable: 124 of the 128
  proposals carry a distinct `(program, input strides/offset, output
  strides/offset, num_warps)` configuration, and 86 inputs carry a nonzero
  element offset. A batch that wants layout coverage should therefore freeze
  layouts directly rather than sampling them through the generation seed.
* Rank-one reduction tails are unreachable through the guarded storage route:
  the reduction returns a scalar, and `triton_case_worker._initial_output`
  allocates `storage_span` elements for an empty output shape before handing the
  kernel a view at the guard prefix, which the qualification turned into an
  illegal memory access beside the same program passing unguarded. Either the
  worker's scalar path or the configuration space has to change before those
  tails can carry guarded evidence.
* Input immutability is checked on every run and never refused, but it is not
  re-derivable from the archived bytes: the worker writes `actual-physical.bin`
  for the output allocation only. A future storage batch that wants the input
  side in the record needs the worker to archive the staged input allocation
  too; until then the record shows the enforcement, not the bytes.
