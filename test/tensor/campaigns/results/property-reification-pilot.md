# Tensor property reification pilot

## Protocol

The prospective selection pilot compared ordinary rejection sampling, per-coordinate boundary-value rejection sampling, simple construction, and Bitwuzla selection on the same frozen 120 graph/target problems over three paired trials. It scheduled 1,440 attempts with a five-second selection deadline per attempt. The evaluation workload used seed `281`; a disjoint 12-problem development workload used seed `282`.

Each accepted input was replayed independently against the complete requested output target. Evaluation targets were frozen before selection and were not derived from sampled witnesses. The domain was `[-8, 8]`, with 16, 32, or 64 input elements. The workload contains independent sums, affine sums, layout-plus-sum graphs, shared reductions, and bounded nonlinear graphs. Its SHA-256 is `9091cf3905a2a5255d923114f8fb81db6cac24d682f26917b3f918cb5f5babeb`; the manifest SHA-256 is `6dbda0d543a123e6560f36b5564de3b61abf2759b50f46be661e5ab9f8b9e958`.

Run 002 is the prospective result. It uses selector method-code SHA-256 `10ccf0dd1a68744aaf2224961d4bd606a8bf44bfbad4315021cd90d0a6467533`, schedule SHA-256 `bdd481b789c4fc0c864a62c27ca8781722a08c917f17b486c7205ee6762b6325`, and run-identity SHA-256 `3c5dc875ca948241d906222c5ce12d25626e91c81a64a4a0b3e63d07c7218e30`.

## Selection results

All 1,440 attempts reached a terminal status. The ledger recorded zero witness-validation or contradictory-evidence incidents. The feasibility audit classified 78 problems as confirmed feasible, 33 as solver-proved infeasible with no contradictory witness, and 9 as unresolved.

| Method | Validated | Other terminal outcomes | Distinct problems with a witness in any trial |
|---|---:|---|---:|
| Ordinary sampling | 42/360 | 318 budget exhausted | 17/120 |
| Boundary sampling | 51/360 | 309 budget exhausted | 22/120 |
| Simple construction | 72 validated among 72 supported; 288 unsupported | unsupported construction family | 24/120 |
| SMT | 222/360 | 99 UNSAT; 39 budget exhausted | 74/120 |

The constructor supports independent-sum cases only. Its 72 validated attempts cover its supported subset; unsupported cases are not sampling failures. In each trial, SMT validated 74 problems. Ordinary sampling validated 13, 16, and 13; boundary sampling validated 17, 20, and 14; construction validated its 24 supported problems. SMT therefore validated 61, 58, and 61 more problem/trial pairs than ordinary sampling, and 57, 54, and 60 more than boundary sampling.

Summed end-to-end driver time, including unsuccessful attempts, was 1,651.124 seconds for ordinary sampling, 1,622.670 seconds for boundary sampling, 2.782 seconds for construction, and 345.218 seconds for SMT. This is 0.025, 0.031, and 0.643 validated inputs per second for ordinary, boundary, and SMT respectively. These are engineering measurements on the frozen workload, not statistical significance claims.

The corrected boundary selector makes an independent 50/50 choice for every input coordinate between a uniform `[-8, 8]` value and the declared boundary menu. Run 001 remains archived as historical evidence, but its boundary selector used one choice for the whole tensor and does not measure the frozen per-coordinate policy. Run 002 supersedes run 001 for method comparisons; the earlier archive was preserved unchanged.

The pilot met the predeclared expansion gate: SMT exceeded both universal samplers in each paired trial, found witnesses in multiple composed graph families, and increased validated-input throughput. This supports expanding the SMT selector for composed exact-target families. It does not establish that SMT selection improves Triton compiler bug discovery.

## V100 replay

The run-002 corpus contains 95 cases selected from independently validated runtime witnesses: 17 ordinary, 22 boundary, 24 construction, and 32 SMT. The strict tensor-v1 loader accepted every case before execution. All outputs use exact complete i32 comparison.

| Triton mode | Cases | Pass | Other outcomes | Campaign-valid |
|---|---:|---:|---:|---:|
| Interpreter | 95 | 95 | 0 | 95 |
| Compiled | 95 | 95 | 0 | 95 |

Compiled replay ran serially on device 0, a Tesla V100-SXM2-32GB (SM70), with Triton 3.8.0 at commit `e2e356723ce35f46b5144373b145c55c4d712237`, CUDA 12.8, and driver 535.274.02. The package-lock SHA-256 was `86ee249201768e82cf8c5d1d122bf378bf11d7f239e62dd78a4d87396e61cb44`. Every compiled case began with an empty per-case Triton cache. The frontend binary SHA-256 was `6c4f2adc11f1e54a706b0627139bfaf8aea33a993843e4d214f354b39aed4ef4`; the recorded source snapshot identity was `edf43d8022c5cb9b9bf4b4c3fcec608b38ea769fad35c3f8aedaab6970c41db7`. The replay found no mismatch candidate.

## Validation and interpretation

On the isolated remote snapshot with Bitwuzla and `TENSOR_REF=1`, the property-workload, input-selection, pilot-driver, and case-export modules passed 39 tests, and the full `make TENSOR_REF=1 test-tensor` gate passed. The remote renderer check accepted 396 SIR forms across 132 target pairs. Local focused pilot/workload tests, selector tests using the available interpreter, Ruff checks, Python compilation, and `git diff --check` passed. The local environment lacks matching `symirsolve` and current `symiri` binaries, so the full selector module was validated remotely.

The result shows that this adapter can select runtime inputs for frozen tensor outputs and that the selected corpus replays correctly through Triton's interpreter and compiled paths. It supports expanding SMT selection under this measured profile. The experiment measured exact-target input selection, not compiler defect discovery, and found no new compiler defect.

## Reproduction and evidence

The corrected run-002 selection ledger and attempt artifacts are in [`run-002-selection.tar.gz`](../archive/property-reification-pilot/run-002-selection.tar.gz), SHA-256 `6574f2be4ce985080ae06be91f0e04a34036c191d67a15056cf201eb4a28462b`. The compact 95-case replay corpus, including both mode reports, outputs, per-case environments, package locks, and retained compiler IR, is [`corpus-002-compact.tar.gz`](../archive/property-reification-pilot/corpus-002-compact.tar.gz), SHA-256 `58c808390cf1ccfd6ae8a8b9782a8dd632598ce2e4013601a4534fd1e0db4a37`; its embedded and separately available file manifest is [`corpus-002-files.json`](../archive/property-reification-pilot/corpus-002-files.json), SHA-256 `170523af7ffc40bd5999e3a98955bd4c0b31e34b2bb6e292dfe9a660f1b3491e`.

The original run-001 selection and corpus archives remain available for historical comparison. The current dirty source/test snapshot and file manifest are [`source-snapshot.tar.gz`](../archive/property-reification-pilot/source-snapshot.tar.gz) and [`source-snapshot-files.json`](../archive/property-reification-pilot/source-snapshot-files.json). Their predecessors are preserved as `source-snapshot-run-001.tar.gz` and `source-snapshot-run-001-files.json`. SHA-256 values for all archives and manifests are listed in [`SHA256SUMS`](../archive/property-reification-pilot/SHA256SUMS).
