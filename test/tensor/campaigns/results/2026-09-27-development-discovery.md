# Development discovery campaign: 2026-09-27

## Frozen protocol

This was an ordinary-input discovery run on the pinned development build. The
plan was frozen before generation. It requested 8,192 tensor-v2 cases with
seed `20260929`, construction-based inputs, the v2 stratum quotas, one fresh
Triton cache per case, and no fault-directed input selection. The exact source
snapshot digest was
`59653cce635c50a15c1bab5157053ea0b1cb37661bb948ba2a068b4502760799`.

The target was Triton 3.8.0 at
`e2e356723ce35f46b5144373b145c55c4d712237`, running on eight Tesla V100-SXM2-32GB
GPUs with CUDA 12.8 and driver 535.274.02. The end-to-end run, including
generation, reference validation, interpretation, compilation, GPU execution,
and coverage indexing, ran from 09:57:20 UTC to 13:01:55 UTC (3h 04m 35s).

The generator emitted all 8,192 requested cases after 1,331 rejected proposals
for failed counterfactual input-dependency checks. The resulting acceptance
rate was 86.0% over 9,523 proposals.

## Results

| Outcome | Result |
|---|---:|
| Generated cases | 8,192 / 8,192 |
| Interpreter | 8,192 valid passes |
| Compiled Triton | 8,192 valid passes |
| Valid result records | 16,384 / 16,384 |
| Wrong-result candidates | 0 |
| Source signatures | 2,556 |
| TTIR / TTGIR signatures | 2,855 / 2,853 |
| LLVM / PTX signatures | 8,192 / 8,192 |
| End-to-end paired-case throughput | about 2,663 cases/hour |

The signature counts describe recorded structural diversity. They do not prove
that every signature exercises a distinct compiler optimization.

| Primary stratum | Cases | Comparison policy | Result |
|---|---:|---|---|
| Bit-preserving f32 layout | 1,639 | Exact | 1,639 pass |
| Exact f32 reduction | 1,639 | Exact | 1,639 pass |
| Loop reduction | 1,638 | Numeric | 1,638 pass |
| Multi-input composition | 2,457 | Numeric | 2,457 pass |
| Untargeted exploration | 819 | Numeric | 819 pass |

All cases used f32. The campaign found no new compiler defect. Its separate
#11186 affected/fixed replay remains calibration evidence and is not included
in the discovery counts. There was no independently verified holdout in this
experiment, and the run makes no generator-generalization or directed-selection
claim.

## Joint source and compiled coverage

| Source form | Cases | Source reductions | TTGIR loops | TTGIR tensor loop states |
|---|---:|---|---:|---:|
| Reduction before loop | 263 | Before loop, axis 1 | 1 each | 1 each |
| Repeated accumulator update | 279 | In loop, axis 1 | 1 each | 1 each |
| Single-accumulator reduction loop | 274 | In loop, axis 1 | 1 each | 1 each |
| Nested loop | 266 | In inner loop, axis 1 | 2 each | 1 each |
| Transpose/broadcast loop | 286 | In loop, axes 0 and 1 | 1 each | 1 each |
| Two-input/two-accumulator loop | 270 | In loop, axes 0 and 1 | 1 each | 2 each |
| **Total loop cases** | **1,638** | 1,375 have source reductions in a loop | **1,904 regions** | **1,638 cases** |

Every source loop survived optimized TTGIR: 1,638/1,638 cases. However, **zero
TTGIR loop regions contain a reduction**. Triton hoisted every source-level
in-loop reduction out of its loop. In particular, each of the 270
two-input/two-accumulator cases reduces the original input tensors, which do
not change between iterations. Their two tensor states and dependent updates
survive, but the reduction work is loop invariant.

The two-input loop cases all use logical shape `[8,8]`, with input `x` laid out
at strides `[1,8]` and axis order `[0,1]`, and input `y` at strides `[8,1]` and
axis order `[1,0]`. They span launch warps 1/2/4/8 with counts 56/68/75/71 and
trip counts 2/3/4 with counts 90/95/85. The other loop shapes were `[64,2]`
(272 cases), `[64,4]` (276), `[128,4]` (307), and `[256,2]` (513); their single
input uses strides `[1,row_count]`.

Across all cases, launch warp counts were balanced: 1: 2,035; 2: 2,058; 4:
2,078; 8: 2,021. The 1,638 loop cases used 1: 386; 2: 417; 4: 446; 8: 389.
All source loops retained a tensor-carried state; 270 retained two states and
1,368 retained one.

## Interpretation and next experiment

The campaign reaches more source structures and retains the intended loops,
including multi-input loop state, but its reductions do not execute inside the
optimized loops. The next focused addition should reduce loop-carried tensor
state whose value changes on every iteration. A short pilot should require an
optimized TTGIR reduction inside the loop, retained state dependencies, and
reference-checked output influence before another broad run.

Use small exact i32 recurrences first where the current property contract
supports them; keep f32 cases under the current numeric policy unless exactness
is proved for the complete returned computation. Continue ordinary input
selection as the discovery arm, and keep any directed-input sensitivity test
separate.

## Reproduction and archive

Generation command:

```sh
./rysmith --profile tensor --target sir --tensor-profile-version v2 \
  --tensor-dtype auto --tensor-synthesis construction --seed 20260929 \
  -n 8192 --output-dir CASES
```

The joined audit script is
[`audit_development_discovery.py`](../archive/2026-09-27/audit_development_discovery.py).
The [campaign bundle](../archive/2026-09-27/development-discovery-v2-20260927-seed-20260929.tar.gz)
contains the per-case source, inputs, expected and actual outputs, mode-specific
run artifacts, environment records, compiler IR, campaign plan and ledgers,
coverage, joined audit, calibration replay, and the exact source snapshot. Its
SHA-256 is
`6b451c2a37fd545d1de6554d8f55bdc9fc3745c202b3f01dca8ad1240f85690d`.
The [file manifest](../archive/2026-09-27/development-discovery-v2-20260927-seed-20260929.manifest.jsonl)
has SHA-256
`c86338defb660416103cd867efadfd3b3cda02392b32e74cb11b6af92745b349`.
All 295,140 manifest paths were verified against the extracted local archive;
Triton caches and redundant shard symlinks were excluded. Remote raw campaign
and calibration originals remain intact.
