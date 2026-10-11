# Tensor generator refresh results

## Prospective pilot

The generator refresh expands loop-reduction source forms, varies bounded shapes and trip counts, and samples warp count independently from source form. The final source snapshot is archived at [generator-refresh-source-20260926-final.tar.gz](../archive/2026-09-26/generator-refresh-source-20260926-final.tar.gz), SHA-256 `a8c83a439f5b5aedb53f11560d16ef2f837c99bba1603c8ae238ba42b18dc18f`.

Seed `20260926` produced 100 accepted cases after 12 rejected proposals. The accepted quotas were 20 bit-preserving f32 layout, 20 exact f32 reduction, 20 loop reduction, 30 multi-input composition, and 10 untargeted cases. The five strata produced 10, 7, 15, 30, and 6 distinct source signatures, respectively. Eighteen loop cases use one counted loop and two use nested loops. Single-loop f32 cases cover 1, 2, 4, and 8 warps.

All 100 interpreter checks, 100 selected compiled checks, and 72 warp-probe checks passed with valid evidence. The retained compiler outputs contain 590 normalized fingerprints across TTIR, TTGIR, LLVM, and PTX. One fresh generated case, `case_000032`, reproduces known PR #11186 behavior on the affected build and passes on the corrected development build. It mismatches all 256 outputs on the affected build, with maximum absolute error 272. This is calibration evidence, not a new defect.

The remote Bitwuzla gate passed with `make TENSOR_REF=1 BOOST_CPPFLAGS=-I/tmp/boost_1_89 test-tensor`: 170 tensor profile tests and six build tests passed on the exact archived source snapshot. The final log is [generator-refresh-test-tensor.log](../archive/2026-09-26/generator-refresh-test-tensor.log), SHA-256 `da32b139962caad2493860612656af262c8fd2a47ac37367cbaf84733aba96db`.

## Matched discovery campaign

The campaign generates 8,192 cases with seed `20260927`, uses ordinary deterministic inputs, and runs batches of 256 against Triton development revision `e2e356723ce35f46b5144373b145c55c4d712237` and Triton 3.4.0 release revision `2c59df5b2606256842fde9799cde5c37a40fec81`. Each generated program, input, expected result, and launch configuration is paired across targets. Development runs on GPUs 0–3 and release runs on GPUs 4–7. Each case starts with an isolated empty Triton cache. The eight-hour budget includes generation, case preparation, reference execution, compilation, GPU execution, and coverage summaries.

The 8,192 generated manifests meet the same proportional quotas as the pilot: 1,639 exact-f32 reductions, 1,639 bit-preserving layouts, 1,638 loop reductions, 2,457 multi-input compositions, and 819 untargeted cases. They contain 2,552 source signatures overall, with 7, 12, 45, 2,454, and 34 signatures in those strata, respectively. All inputs use f32 across 17 shapes. Warp counts 1, 2, 4, and 8 occur 2,051, 1,982, 2,024, and 2,135 times. Loop-bound patterns are `[]`, `[2]`, `[3]`, `[4]`, `[2,2]`, and `[2,3]`.

The final campaign driver is [run_generator_discovery.sh](../archive/2026-09-26/run_generator_discovery.sh), SHA-256 `8515a56233b377055ae6728b08b79eb22c12a2432b65c2d146596bf333c4c624`. An eight-case smoke run with seed `20260930` verified the driver: each GPU worker received two disjoint cases; all 16 interpreter and compiled worker results were campaign-valid and passed; both targets produced coverage files. A first smoke with overlapping shards is excluded from evidence.

The full campaign was launched on `great_rhodes` at `/tmp/refractir-discovery-v2-20260926-seed-20260927`, under an eight-hour timeout. The remote campaign plan records the target pins, cache policy, pairing rule, seed, and budget. Release mismatches matching the known #11186 behavior are calibration duplicates and will be reported separately from new candidates.

At 4,096 paired cases, development results are all valid passes. Release has 4,061 passes and 35 valid numeric mismatches. All 35 have byte-identical source, input, and expected output artifacts on development, pass on development, and share the calibrated one-warp f32 `[256,2]` loop-reduction signature with release `allow_reorder` TTGIR.

## Final campaign results

The run completed on 2026-09-26 at 22:46:16 UTC, 6 hours 15 minutes 26 seconds after launch and before its eight-hour limit. The campaign recorded 32 batches and 256 GPU worker records, assigning all 8,192 generated cases to each target.

All 8,192 interpreter results passed with `campaign_valid: true` on both targets. Development compiled all 8,192 cases successfully. Release produced 8,122 valid passes and 70 valid numeric mismatches; all 8,192 release results have valid evidence.

All 70 mismatches have byte-identical source, input, and expected output artifacts to their paired development cases, which pass. They share source signature `41f164cbb1a9e2c4a8f193b244d38ca57e6d7972aaca1914fc68ec88bf8a820a` and the same f32 input shape `[256,2]`, element strides `[1,256]`, reduction axis 1, loop bound 2, and one-warp launch. Every mismatching release TTGIR contains `allow_reorder`. These are 70 fresh ordinary-input reproductions of the known #11186 defect family and no new defect.

Both targets recorded 2,552 source signatures and 1,638 loop cases whose loop structure survived compiled IR. Their compiled fingerprints include 2,848 TTIR signatures, 2,846 TTGIR signatures, and 8,192 each at LLVM and PTX. The generated set contains 17 input shapes, all f32, and warp counts 1, 2, 4, and 8.

The completed campaign archive is [generator-discovery-v2-20260926-seed-20260927.tar.gz](../archive/2026-09-26/generator-discovery-v2-20260926-seed-20260927.tar.gz), SHA-256 `8dceb98f8d2bfe03dd8b011de2df5f92ff450ae7e7c5f60e58fad5ab3b477dfb`. Its [member manifest](../archive/2026-09-26/generator-discovery-v2-20260926-seed-20260927.files.jsonl) records sizes and SHA-256 hashes for 623,884 files. The archive omits Triton cache contents and shard symlinks. Every extracted file matched its manifest entry; no remote originals were removed.
