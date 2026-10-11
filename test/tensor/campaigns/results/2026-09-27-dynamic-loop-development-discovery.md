# Dynamic loop development discovery

## Frozen experiment

This was a fresh ordinary-input campaign against the pinned Triton 3.8.0
development build, commit
e2e356723ce35f46b5144373b145c55c4d712237. The exact RefractIR source snapshot
has SHA-256
**dac691c4b39756169db4bee56ed596d2ae756aa8f61d5a4190aae27780fd669e**.
The frozen plan records the generator, worker, audit, source, and ptxas
hashes, along with the seed, budgets, and execution policy.

The campaign generated 8,192 accepted tensor-v2 cases with seed 20261001,
using deterministic construction inputs and no fault-directed selection. Its
primary quotas were 20% bit-preserving f32 layout, 20% exact f32 reduction,
20% loop reduction, 30% multi-input composition, and 10% untargeted
exploration. All quotas were met; generation recorded 1,270 rejected
proposals before reaching the requested count.

Each case ran independently in interpreter and compiled modes, checked
against its recorded reference and comparison policy. Compiled runs used a
fresh Triton cache per case. The campaign budget included generation,
reference validation, compilation, execution, and audit. Seven V100 devices
(IDs 1–7) ran the cases; device 0 was excluded because it had a pre-existing
369 MiB allocation. No process on that device was changed.

The environment was Python 3.10.12, CUDA 12.8, driver 535.274.02, and
Tesla V100-SXM2-32GB (SM70). The pinned ptxas SHA-256 is
**c960a4f238b17d5c5d3c01ad2bbc1ebd2c5aecc459cb4d223bff10b45f9b8fca**.

## Outcomes

All 8,192 interpreter results and all 8,192 compiled results are campaign-valid
passes. Of the cases, 3,278 use exact comparison and 4,914 use the declared
numerical comparison. There were no wrong-result candidates, invalid
evidence records, or quota shortfalls.

The run took 12,519 seconds (3 h 28 min 39 s), from 2026-09-27 14:47:50 UTC to
18:16:29 UTC. End-to-end throughput was about 2,356 accepted cases per hour
across seven devices. This includes generation and the final audit; it is not
GPU-only execution throughput.

## Compiler-behavior coverage

The joined audit reports 2,566 source signatures; compiled signatures were
2,889 TTIR, 2,887 TTGIR, and 8,192 each for LLVM and PTX. These signatures
are coverage diagnostics, not counts of distinct optimization behaviors.
The launch distribution was 1,994 / 2,055 / 2,068 / 2,075 cases at 1 / 2 / 4
/ 8 warps.

All 1,638 source cases containing counted loops retained a loop in optimized
TTGIR. The 8,192 generated cases had these source forms:

| Source form | Cases |
|---|---:|
| Straight-line layout/composition | 4,915 |
| Straight-line reduction | 1,639 |
| Two-input, two-accumulator loop | 474 |
| Transpose/broadcast loop | 242 |
| Pre-loop reduction | 245 |
| Nested loop | 223 |
| Single-accumulator reduction loop | 219 |
| Repeated accumulator-update loop | 235 |

The focused target was a loop whose reductions consume evolving loop-carried
tensors. Of the 474 two-input, two-accumulator loop cases, 249 retained four
reductions inside one optimized TTGIR loop. Each of those 249 cases carries
two tensor values, reduces on axes 0 and 1, uses two runtime inputs, and has
counterfactual witnesses showing that either input can change the returned
value. The other 225 cases retained the loop but had no reductions inside it
after optimization, so they do not count as exercising this target behavior.

The 249 qualifying cases cover logical input shapes [2,4], [4,2], and [4,4];
trip counts 2, 3, and 4; all four warp counts; and three pairs of physical
input layouts. They use f32 and the numerical comparison policy. All passed
against the independent reference. This confirms that ordinary generation
reaches the intended optimized structure in a portion of the selected source
family. It does not establish how often this structure leads to a compiler
bug.

## Interpretation and limits

This campaign found no new Triton defect. It ran only on the pinned
development build; the full campaign was not paired with an affected release.
The separate #11186 calibration replay and the fresh paired behavior control
remain separate evidence. The run used ordinary construction inputs, so it
makes no claim that directed input selection improves discovery. It also
provides no independent historical holdout or generalization result.

The source and IR signature counts indicate structural and compiled diversity,
but they do not show that each signature exercises a unique optimization.
Likewise, successful coverage of the loop-reduction target is evidence of
reachability, not evidence of defect discovery.

## Reproduction and archived evidence

The generator command was:

    ./rysmith --profile tensor --target sir --tensor-profile-version v2 \
      --tensor-dtype auto --tensor-synthesis construction --seed 20261001 \
      -n 8192 --output-dir CASES

The [frozen plan](../archive/2026-09-27/dynamic-loop-development-discovery-plan-20261001.json)
and [campaign archive](../archive/2026-09-27/refractir-dynamic-loop-development-discovery-20260927-seed-20261001.tar.gz)
preserve the exact source snapshot, generated cases, inputs, references,
results, compiler IR, environment, logs, and scripts. The archive SHA-256 is
**bacf7112ffdaea64e22881bb55e4a1666ab86f73aa755d0e57842e75e83bf44a**;
the 295,282-entry
[sidecar manifest](../archive/2026-09-27/refractir-dynamic-loop-development-discovery-20260927-seed-20261001.manifest.jsonl)
has SHA-256
**f147fef1d01d5ef3c2d55e60b00fa15ac52e9dff785df2e1fc497cc8d4ebeb68**.

Both hashes match the remote originals. Every archive path, size, and content
hash was checked locally by streaming the archive: 278,882 regular-file
payloads were hashed, and 16,400 hardlink entries were checked against their
manifest targets. The expanded tree was not materialized locally. Raw remote
campaign files remain intact.

Earlier #11186 calibration is summarized in the
[development discovery report](2026-09-27-development-discovery.md); the
[dynamic-loop paired control](2026-09-27-dynamic-loop-revision-control.md)
records a separate fresh case that passed on both affected and corrected
builds.
