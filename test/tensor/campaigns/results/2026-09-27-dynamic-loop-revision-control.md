# Dynamic loop-reduction revision control

The freshly generated exact-i32 case `case_000024` (shape `[4,2]`, trip count
3, one warp) was replayed with the same SIR, inputs, expected output, and layout
manifest on Triton #11186's affected commit
`446f56ffd12431b1760ed504b7bcfdb6bb7b93c9` and the corrected development
commit `e2e356723ce35f46b5144373b145c55c4d712237`.

An initial attempt omitted `TRITON_PTXAS_PATH`. It produced `Cannot find
ptxas` on the affected build and is retained as an environment setup failure;
it is not scored as a compiler result. The corrected rerun supplied each
revision's own `ptxas` binary. Both executions then produced campaign-valid
exact passes matching `expected.bin`. Both optimized TTGIR records contain the
counted loop and its two reductions. There was no mismatch, so this case is a
paired behavior control, not a compiler defect or #11186 rediscovery.

The replay bundle includes both attempts and the exact `ptxas` hashes:
[paired-control archive](../archive/2026-09-27/dynamic-loop-paired-control-20260927.tar.gz)
(SHA-256 `06e0102e58c38941b83bd8f631957614d1b29f2cb2de9735278efd41ea346277`),
[file manifest](../archive/2026-09-27/dynamic-loop-paired-control-20260927.manifest.jsonl)
(SHA-256 `fa204169b39f6b42057ac2c55fd7484c94822ed86c398588c9be129fd037b969`).
All 142 files verified after local extraction.
