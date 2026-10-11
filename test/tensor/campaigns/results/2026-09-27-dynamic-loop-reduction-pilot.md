# Dynamic loop-carried reduction pilot

## Snapshot and procedure

The final formatted generator/runtime/test snapshot is archived with SHA-256
`dac691c4b39756169db4bee56ed596d2ae756aa8f61d5a4190aae27780fd669e`. The
remote Bitwuzla build passed
`make -j4 TENSOR_REF=1 BOOST_CPPFLAGS=-I/tmp/boost_1_89 test-tensor` on this
snapshot: 182 tensor profile tests and six tensor build tests passed, with no
compiler warnings. The six focused generator tests also passed; the final one
checks that the f32 form remains numerical and is generated across all warp
counts.

With seed `20260930`, the generator emitted all 900 requested v2 loop-reduction
cases. 133 matched the new single-loop recurrence form. We selected 100 exact
i32 cases, balanced across 1, 2, 4, and 8 warps (25 each), with logical shapes
`[2,4]`, `[4,2]`, and `[4,4]`, and trip counts 2–4. Re-running generation from
the final formatted snapshot reproduced the selected SIR, input, expected
output, and manifest hashes exactly.

The cases ran in interpreter and compiled modes on V100 devices 1–7 using
Triton 3.8.0 at revision
`e2e356723ce35f46b5144373b145c55c4d712237`, CUDA 12.8, and driver 535.274.02.
GPU 0 was left unused because it had an existing 369 MiB allocation. The
per-device interpreter phase took 50.0–54.6 seconds and the compiled phase
96.4–103.7 seconds.

## Results

All 100 interpreter results and all 100 compiled results are campaign-valid
exact passes. Each actual output matches `expected.bin` byte-for-byte, and the
runner's artifact hashes match every retained artifact.

All 100 optimized TTGIR modules retain the target behavior: a counted loop
carries two tensor values and contains both reductions. The reductions consume
the evolving accumulator tensors, then expand and broadcast their results into
cross-dependent updates. Each case also has two runtime inputs with distinct
physical layouts: `x` uses `[1, rows]` element strides and `y` uses
`[columns, 1]`. Counterfactual witnesses show that changing either input
changes the returned value.

The nine source signatures reflect shape and trip-count constants within this
one recurrence structure; they do not represent nine different compositions.
The pilot found no new compiler defect. It establishes that fresh generated
programs reach the previously missing optimized loop-reduction behavior with
an exact, independently checked output.

Before scaling, one fresh case was also paired across the pinned #11186
affected and corrected builds. Both builds pass this case with valid exact
evidence and retain the loop reductions in TTGIR. This is a behavior control,
not a #11186 rediscovery; the separate historical fixture remains the
affected-mismatch/corrected-pass calibration.

## Reproduction and archive

Generator command:

```sh
./rysmith --profile tensor --target sir -n 900 --tensor-dtype i32 \
  --tensor-profile-version v2 --tensor-stratum loop-reduction \
  --seed 20260930 --output-dir GENERATED
```

The compact bundle contains all 900 generated candidates, the selected 100
case inputs and results, per-case logs and compiler IR, the final source
snapshot, and the exact environment record. Its SHA-256 is
`4a49fe1c8a8e01f5246373bf15bc0ee7c784e7c3a80b1afe354e57455779c570`. The
8,147-entry sidecar manifest has SHA-256
`59a01dc7f33559cd0d819f63037aa97126c67560d72ffab0a663a36851ef9c34`; all
8,147 extracted file paths, sizes, and hashes verified locally. Temporary
absolute shard symlinks are excluded. Remote raw cases and generated files
remain intact.

See the [pilot bundle](../archive/2026-09-27/dynamic-loop-reduction-pilot-20260927.tar.gz)
and its [file manifest](../archive/2026-09-27/dynamic-loop-reduction-pilot-20260927.manifest.jsonl).
The [paired revision control](2026-09-27-dynamic-loop-revision-control.md)
records the per-build `ptxas` setup and outcomes.
