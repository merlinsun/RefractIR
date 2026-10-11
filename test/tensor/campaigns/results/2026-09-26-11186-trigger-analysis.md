# PR #11186 trigger analysis

The corrected source snapshot passed a clean Bitwuzla `0.9.1-dev` build and `make TENSOR_REF=1 BOOST_CPPFLAGS=-I/tmp/boost_1_89 test-tensor`. The gate ran 305 test executions and reported `campaign_valid: true`. Its log SHA-256 is `7978ea9ba4042143b044a12e64d8d59d606e98b566d8fc06fd908dd1ee081c2b`; the corrected snapshot and exact log are listed in the [campaign archive](../archive/2026-09-26/README.md).

The trigger sweep started from the authored f32 `[256,2]` reduction, physical input strides `[1,256]`, one program, one warp, one two-iteration counted loop, and `x[row,col] = row`. Each case was checked against its independently generated SIR reference on the affected full build `446f56ffd12431b1760ed504b7bcfdb6bb7b93c9` and corrected full build `e2e356723ce35f46b5144373b145c55c4d712237`. Every executed case was campaign-valid on both builds, and the corrected build passed all 11.

| Case | Change from the authored fixture | Affected build | Affected TTGIR contains `allow_reorder` |
|---|---|---|---|
| Baseline | None | 256 mismatches; max absolute error 256 | Yes |
| `dtype-i32` | f32 to i32 | Pass | No |
| `rows-128` | 256 rows to 128 | Pass | Yes |
| `layout-contiguous` | Strides `[1,256]` to `[2,1]` | Pass | Yes |
| `warps-2` | One warp to two | Pass | Yes |
| `warps-4` | One warp to four | Pass | No |
| `warps-8` | One warp to eight | Pass | No |
| `loop-unrolled` | Remove the counted loop and duplicate its body | Pass | No |
| `input-row-plus-col` | `x[row,col] = row + col` | 256 mismatches; max absolute error 254 | Yes |
| `interaction-row128-contiguous` | 128 rows and contiguous strides | Pass | Yes |
| `interaction-row128-warps2` | 128 rows and two warps | Pass | No |

The affected baseline TTGIR reshapes `tensor<256x2xf32>` to `tensor<256x1x2xf32>` with `allow_reorder`, then reduces axis 2 without restoring the elements outside the reduction axis. The corrected TTGIR inserts a transpose and reshape before that reduction. This matches the recorded OptimizeThreadLocality root cause for PR #11186.

In this authored neighborhood, changing f32 to i32 removes the relevant reshape/reorder path, and removing the counted loop also removes it. The 128-row, contiguous-layout, and two-warp variants retain the `allow_reorder` reshape but return correct outputs. Four and eight warps use different optimized IR and pass. Changing the input pattern alone does not hide the error. The wrong result therefore depends on more than the presence of `allow_reorder`; this sweep does not establish that any single factor is universally necessary for the compiler defect.

The 128 fresh rediscovery cases were all i32. The paired dtype ablation shows that this choice can keep those cases out of the faulty f32 reshape/reorder path, which explains why that campaign did not reach the calibrated behavior. This is a targeted explanation for the observed campaign, not evidence that f32 is required for every form of the defect.

A `[512,2]` shape was rejected by the v2 profile because it exceeds the 512-element tensor limit. It was excluded before compilation and is not counted as an ablation result.

One initial baseline attempt stopped before compilation because `TRITON_PTXAS_PATH` was unset. The retry used the frozen CUDA 12.8 `ptxas` path; its SHA-256 matched the pinned value `c960a4f238b17d5c5d3c01ad2bbc1ebd2c5aecc459cb4d223bff10b45f9b8fca`.

The original directed-selection records contain 38 of 100 fixed-count graphs and 395 of 849 fixed-time graphs with no applicable model in the then-recorded catalogue. No graph exhausted its budget with an applicable model. Extending the model set with a defined, same-shape reversal of output reduction groups applies to all 38 fixed-count and 395 fixed-time shortfalls. Applied retrospectively to their saved reference outputs, it distinguishes 35 and 358 cases, respectively. Across the saved ordinary corpus, it applies to all 100 fixed-count and 516 fixed-time graphs and distinguishes 95 and 448 ordinary inputs.

This retrospective check compares a deterministic output-group permutation with saved reference bytes under each case's existing comparison policy. It does not rerun P/D selection or establish that a new proposal search is more effective. It shows that the old shortfall count was largely a model-applicability gap, while the remaining symmetric outputs need other fault models or input patterns; increasing proposal limits alone is still unsupported. Recompute these counts with the [audit script](../archive/2026-09-26/audit_directed_group_model.py).

The corrected selector passes 24 focused host tests; the evaluation and coverage suites pass another 59 tests, and Ruff checks pass. The corrected source snapshot has also passed the remote tensor gate. A local C++ build is unavailable because the workstation lacks `pkg-config` and Bitwuzla.

The corrected selector pilot completed on 100 fresh i32 graphs but did not compile Triton kernels. It recorded one fixed-count rescue and five fixed-time rescues across the selected fault models; the fixed-time directed arm completed fewer graph/model tasks than the ordinary arm. These results measure reference-model selection only.

The prospective f32 follow-up and paired GPU sample are recorded below. They rediscover the known behavior under the targeted one-warp setting; they do not establish generalization to new source structures or a new compiler defect. No independent holdout or cvc5 run is claimed.

## Prospective f32 selector and GPU sample

The fixed-count selector run generated 100 accepted f32 loop-reduction graphs for seeds `4200000`–`4200099`. The only target was `swapped_reduction_groups`; every graph had an applicable model, and all 100 ordinary inputs already satisfied the distinguishing objective. Directed selection accepted the same input bytes as ordinary selection in all 100 pairs. There were zero rescues, proposal exhaustions, or inapplicable models. Each arm used one property reference evaluation per graph and no model reference evaluations. Mean selection elapsed time was 23.9 ms ordinary and 21.2 ms directed; mean reference time was 13.1 ms and 12.8 ms, respectively. Since there were no rescues, no per-rescue cost is defined.

The 100 graphs contain only two source signatures: 50 nested-loop graphs with bounds `[2,2]` and 50 single-loop graphs with bounds `[2]`. The generator's deterministic launch mapping assigns one or four warps to the nested form and two or eight warps to the single-loop form. Thus the generated single-loop cases do not naturally use the one-warp configuration that triggers the known defect.

Eight preselected single-loop seeds (`4200001`, `4200003`, `4200005`, `4200007`, `4200009`, `4200011`, `4200013`, `4200015`) were compiled on eight V100s against both pinned full builds. Their launch metadata was changed from the generated warp count to one warp; each change is recorded as `targeted-calibration`. In every ordinary/directed pair, the SIR, input bytes, and expected bytes were identical, so each shared input was compiled once. All 8 affected runs were campaign-valid and reported `numeric_mismatch` (176–202 mismatching output elements; maximum absolute error 6). All 8 corrected runs were campaign-valid and passed.

The selected cases use the same SIR source hash as the authored calibration fixture. All eight affected TTGIR files retain one `scf.for` and the `tt.reshape ... allow_reorder` sequence; all eight corrected TTGIR files retain the loop and insert `tt.trans` before reduction. Same-input controls for seeds `4200001` and `4200003` at their generated two- and eight-warp settings pass on both builds. The replay wrapper exits 1 for these controls because the expected affected-build mismatch is absent; each per-build run is campaign-valid and reports `pass`. The evidence therefore isolates the launch setting within this tested neighborhood, while the fixture and fresh cases still share one source structure.

The outcome is a targeted rediscovery of known PR #11186 behavior, not eight defects and not an independent generalization result. It shows that the current search missed the one-warp execution setting for the single-loop form. More seeds alone will not fix the issue: the 100-case selector run had only two source signatures and found no input that ordinary selection failed to distinguish. An overnight run with this frozen generator would repeat the same structures and is deferred until launch choices are decoupled from loop form or the generator produces meaningfully different loop-reduction programs.

The compact archive `f32-loop-followup-20260926.tar.gz` contains the 100 selection ledgers/cases, paired compiler artifacts, same-input warp controls, exact drivers, and no Triton cache directories. Its SHA-256 is `6a23d1536b2e5820d21cef5eed70804b85f396c3aef2d9dd48e22886c7c4b2db`. The local and remote archive hashes matched, and the remote originals remain intact. See the [archive index](../archive/2026-09-26/README.md).
