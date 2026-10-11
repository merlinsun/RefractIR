# Joint coverage audit of the 2026-09-26 campaign

## Evidence and method

This audit reads the archived development and release cases from [the verified campaign archive](../archive/2026-09-26/generator-discovery-v2-20260926-seed-20260927.tar.gz). It joins each case's `case.json`, `program.sir`, compiled TTGIR loop summary, and compiled `result.json`. Source loop forms below are classified from the SIR block bodies and self-updates; loop survival is taken from the normalized TTGIR coverage record. This is a structural coverage description, not a claim that every distinct IR fingerprint represents a distinct optimization behavior.

## Campaign outcome and stratum coverage

| Primary stratum | Cases | Source signatures | Runtime inputs | Source loops | TTGIR loop survives | Development | Release |
|---|---:|---:|---:|---:|---:|---|---|
| Bit-preserving f32 layout | 1,639 | 12 | 1 | 0 | 0 | 1,639 pass | 1,639 pass |
| Exact f32 reduction | 1,639 | 7 | 1 | 0 | 0 | 1,639 pass | 1,639 pass |
| Loop reduction | 1,638 | 45 | 1 | 1,638 | 1,638 | 1,638 pass | 1,568 pass, 70 mismatches |
| Multi-input composition | 2,457 | 2,454 | 2 | 0 | 0 | 2,457 pass | 2,457 pass |
| Untargeted exploration | 819 | 34 | 1 | 0 | 0 | 819 pass | 819 pass |
| **Total** | **8,192** | **2,552 overall** | — | **1,638** | **1,638** | **8,192 pass** | **8,122 pass, 70 mismatches** |

All 16,384 compiled results have valid campaign evidence. Development numerically passes all 8,192 cases. The 70 release mismatches are one known #11186 family; all are paired with identical source, input, and expected-output bytes and pass on development.

## Loop structure combinations

Every loop case has one runtime input, one self-updated tensor accumulator, a single program instance, f32 elements, and a noncontiguous input layout with element strides `[1, row_count]`. All five source forms appear with 1, 2, 4, and 8 warps. Every source loop has a qualifying tensor-carried loop in optimized TTGIR and its body contributes to the returned value.

| Source form | Cases | Warps 1 / 2 / 4 / 8 | Reduction placement and axes | Input shapes | Release result |
|---|---:|---|---|---|---|
| Reduction before loop; loop adds its result | 353 | 89 / 94 / 83 / 87 | One axis-1 reduction before loop | `[64,2]`, `[64,4]`, `[128,4]`, `[256,2]` | All pass |
| Repeated update to the same accumulator | 344 | 84 / 94 / 80 / 86 | One axis-1 reduction in loop; two adds to the same accumulator | Same four shapes | All pass |
| Nested counted loops | 324 | 77 / 90 / 66 / 91 | One axis-1 reduction in inner loop | Same four shapes | All pass |
| Transpose/reduce/expand/broadcast/reduce | 310 | 71 / 77 / 82 / 80 | Two reductions in loop, axes 0 and 1 | Same four shapes | All pass |
| Single axis-1 reduction in the loop | 307 | 70 / 78 / 80 / 79 | One axis-1 reduction in loop | `[256,2]` | 237 pass, 70 known mismatches |

The last form is the calibrated #11186 neighborhood: the 70 mismatches occur at one warp, two iterations, shape `[256,2]`, and strides `[1,256]`. The other warp counts for that form pass.

## The measured gap

The campaign successfully varied loop form, shape, trip count, warp count, and reduction placement. It already includes a loop with two in-loop reductions on different axes, so adding that same one-input form again is not the clearest next step.

Two structures remain disjoint:

- All 1,638 loop cases have exactly one runtime input and one loop-carried tensor accumulator. No loop case has two dependent tensor-carried values.
- All 2,457 two-input cases are straight-line compositions with no counted loop.

Therefore, **multi-input loop reductions with dependent loop-carried tensor state** are absent. The next generator expansion should combine those structures, with different physical strides for the two inputs so the reductions exercise distinct memory-axis orders. Keep the case within one program instance and the existing tensor-v2 semantics. Use bounded integer-valued inputs and exact i32 cases for a decisive baseline; include f32 cases only under the declared numerical comparison policy unless whole-output exactness is proved.

## Pilot gate

Before a broad campaign, the pilot should establish all of the following for the new stratum:

1. The generated SIR has two runtime tensor inputs, two dependent tensor loop-carried values, and reductions on the declared axes.
2. Both inputs' different recorded physical layouts reach the exported Triton kernel correctly.
3. Optimized TTGIR retains the intended loop state, reductions, and data dependency; source syntax alone does not count.
4. Counterfactual checks show that each input and both loop-carried updates influence the returned tensor.
5. Exact i32 reference comparisons pass for the baseline cases. Any f32 cases use the existing numeric policy unless exactness covers the complete returned computation.

A new defect is not required for pilot acceptance. Keep #11186 runs in the separate calibration set and keep masked tails deferred unless the joint audit later shows the current profile is already well exercised.
