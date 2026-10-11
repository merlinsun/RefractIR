# Multi-input loop-reduction pilot

## Snapshot and procedure

The pilot used the updated generator source snapshot with tree digest
`59653cce635c50a15c1bab5157053ea0b1cb37661bb948ba2a068b4502760799`. The
remote Bitwuzla build passed
`make -j4 TENSOR_REF=1 BOOST_CPPFLAGS=-I/tmp/boost_1_89 test-tensor`: 176 tensor
profile tests and six build tests passed. The C++ rebuild emitted no warnings.

The command generated 100 fresh `loop-reduction` cases with seed `20260928`,
dtype `i32`, and construction inputs. It accepted all 100 without rejected
proposals. Cases ran across all eight idle V100s on Triton
`e2e356723ce35f46b5144373b145c55c4d712237` (3.8.0), CUDA 12.8, and driver
535.274.02. Each case used an empty per-case Triton cache. The existing pilot
driver retained source, input, expected output, environment, worker, and compiler
IR evidence.

## Results

| Measure | Result |
|---|---:|
| Generated cases | 100/100 accepted |
| Interpreter | 100 valid passes |
| Compiled Triton | 100 valid passes |
| Source loops / loops retained in compiled IR | 100 / 100 |
| Source signatures | 34 |
| Compiled signatures | 64 TTIR, 62 TTGIR, 100 LLVM, 100 PTX |
| Generated launch warps | 1: 29, 2: 20, 4: 27, 8: 24 |

Eighteen cases used the new two-input loop form, spanning all warp counts
(1: 6, 2: 2, 4: 6, 8: 4) and trip counts 2, 3, and 4. All used logical shape
`[8,8]`, with input `x` at strides `[1,8]` / axis order `[0,1]` and input `y`
at strides `[8,1]` / axis order `[1,0]`. The exported Triton kernels and
optimized TTGIR use both recorded layouts.

All 18 targeted TTGIR records contain one `scf.for` with two tensor loop results,
reductions on axes 0 and 1, and a loop-body update where the `y` accumulator
consumes the updated `x` accumulator. The input reductions are loop invariant,
so Triton hoisted both reductions before the loop. The loop-carried dependency
remains in the loop body. The return uses both final loop states; eight cases
insert a layout conversion for one state before the final add. Every targeted
case also records counterfactual witnesses showing that changing either input
changes the reference output. All cases use exact i32 comparison and pass.

This pilot establishes that the new source/configuration combination reaches
the intended optimized behavior and affects checked outputs. It found no new
compiler defect. The #11186 cases remain a separate calibration family and were
not included in this pilot.

## Reproduction and archives

Generator command:

```sh
./rysmith --profile tensor --target sir --seed 20260928 \
  --output-dir CASES -n 100 --tensor-dtype i32 \
  --tensor-profile-version v2 --tensor-stratum loop-reduction
```

The case results and retained IR are in
[the pilot artifact bundle](../archive/2026-09-27/multi-input-loop-pilot-20260927.tar.gz)
(SHA-256 `c9a27ee45b5dd23691f6fb60d010b2d45b1d4c490c97b5f750f392b419448f33`).
The exact generator/runtime source snapshot is in
[the source bundle](../archive/2026-09-27/multi-input-loop-source-20260927.tar.gz)
(SHA-256 `6a799167366dc4b96729b467e80215cfe3f4eef2ff36797233ce21a3cbc82326`);
its per-file hashes are in
[the source manifest](../archive/2026-09-27/multi-input-loop-source-20260927.files.sha256)
(SHA-256 `59653cce635c50a15c1bab5157053ea0b1cb37661bb948ba2a068b4502760799`).
The source-tree manifest digest is the `source_snapshot_sha256` recorded by the
pilot runner. Remote originals were left intact.
