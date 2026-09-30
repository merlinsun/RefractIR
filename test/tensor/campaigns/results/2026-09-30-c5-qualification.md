# C5 qualification of the tensor foundation batch

Snapshot: tracked-only export of revision `2bd7ab3c48afc4a2f4170d7e5f22c31e8b65fb1d`,
archive sha256 `f1cf2fad2d518a6ad73fdc0f002f979994832db70399aa3f8b9aa6802c18a9a6`.
The export carries no `.git` directory, so the runner's source metadata is set
explicitly (`REFRACTIR_REVISION`, `REFRACTIR_WORKTREE_DIRTY=false`) rather than
inferred. Host: Docker container `great_rhodes` on `root@210.28.133.13:22230`,
8x Tesla V100-SXM2-32GB, Triton checkout `e2e356723ce35f46b5144373b145c55c4d712237`,
Python 3.10, PyTorch 2.10.0+cu128, solver Bitwuzla.

## Host gates

| Gate | Result |
|---|---|
| `make -j16 all SOLVER=bitwuzla TENSOR_REF=1` | exit 0, no warnings |
| `make -j16 test-tensor` | exit 0 (257, 279, and 24 tests) |
| `make -j16 test-solver` | exit 0 |
| `make -j16 cross-validation` | exit 0 |
| `make test-solver-outcomes` | exit 0 (34 tests: C2 outcomes and C3 deadlines) |
| `make test-tensor-graph-audit` | exit 0 (7 tests) |
| `make test-tensor-property-pilot` | exit 0 (44 tests) |
| `make test-triton` | exit 0, both modes, warp preflight for 1/2/4/8 |
| `make test-unit` | exit 0 (`rytwin_tests` 344/344 last) |
| `make test` | exit 2, with the five classified exceptions below |

## Generated corpus on the pinned GPU

Seven predeclared targets, executed in both Triton modes; every accepted case
ran as both an interpreted and a compiled kernel.

| Target | Generation | interpret | compiled |
|---|---|---|---|
| v2 i32 ordinary, seed 1 | accepted | pass | pass |
| v2 i32 ordinary, seed 2 | accepted | pass | pass |
| v2 f32 ordinary, seed 1 | accepted | pass | pass |
| v2 f32 ordinary, seed 2 | accepted | pass | pass |
| v3 i32 ordinary, seed 1 | accepted | pass | pass |
| v2 i32 exact, target 3, seed 2 | accepted | pass | pass |
| v2 i32 exact, target 3, seed 5 | refused, `input_policy_unsat` | not run | not run |

Twelve executions, twelve passes. The refusal is reported rather than
substituted: seed 5's graph admits no input whose output equals 3, and the
attempt record says so with no case directory.

## Full-suite exceptions

`make test` fails on ten cases, all in the compiler group
(`compiler_tests: 2287/2678 passed, 386 skipped, 5 failed`):

| Failure | Count | Classification |
|---|---|---|
| `test/interp/v023_interp_frame_dangling_{deep,field,load,sibling,store}_1.sir` | 5 | pre-existing, reproduced on the pristine baseline revision |
| `Expected runtime failure (FAIL:UndefinedBehavior) but exited 0` | 5 | pre-existing, same group and the same repetition |

Both classes were reproduced byte-identically from an untouched export of the
baseline revision `43b0859` before this batch began, so they are exceptions to
this qualification, not results of it. No other group failed: reify, rylink,
rytwin, interp, and reducibility report full pass counts in the same run.

## Requirements from the original batch, clause by clause

| Clause | Evidence |
|---|---|
| Task 0: a clean source checkout reproduces the build and tensor gate | This export: `make all` and `make test-tensor` from the tracked-only tree |
| Task 0: every full-suite failure is reproduced and classified | `make test` on this snapshot, below |
| Task 0: no dependency on an untracked local helper remains | `test/tensor/campaigns/run_matched_tensor_evaluation.py` and `v1-baseline.json` are tracked; the evaluation import and the coverage tests pass from the export |
| Task 1: the emitted value has the declared rank before a shape-sensitive operator | `make test-tensor` build group (24 tests) plus `test/unit/test_tensor_emit.py` |
| Task 1: a tensor is never supplied as the scalar fill value to `tl.full` | the same tests assert the fill argument is a scalar for every profile |
| Task 2: chains, shared subexpressions, diamonds, and multiple reductions in one generator | `make test-tensor-graph-audit`, and the 256-seed report: v2 253 accepted with 60 topologies and 214 operator graphs, v3 256 accepted with 58 and 223 |
| Task 2: the report separates topology from dimensions, constants, axes, and names | the audit records source bytes, ordered dependency topology, operator-labelled graph, dimensions, literals, and axes as separate counts |
| Task 2: existing constructor regressions retain their outputs | `make test-reify`-class constructor tests inside `make test` |
| Task 3: one generated i32 graph runs with ordinary and SMT-selected inputs | the exact and ordinary element cases in `test/unit/test_tensor_generation.py`, and the corpus above (v2 i32 ordinary seed 1 and v2 i32 exact seed 2) |
| Task 3: tensor parameters are not rewritten into a separate semantic implementation | the exact route emits a property-constrained clone whose computation digest matches the original once query constraints are removed |
| Task 3: a loop witness records repeated block visits | the C3 worker compares the ordered replay labels with the selected path, repeats included |
| Task 3: a v3 witness models logical elements only | `TensorCoverageTests` and the v3 corpus target |
| Task 3: larger graphs can use other validated input policies when the SMT budget is exceeded | `budget_exceeded` for the exact policy above 64 symbolic elements, with the sampling policies recorded as the alternative |

## Explicit limits

The completion report claims no new Triton bug discovery and no improved SMT
effectiveness. The C2 channel reports a typed reason for Z3 only; Bitwuzla's
pinned API exposes no termination reason, so the default build reports
`unspecified` unless a deadline this project owns fires. No real Z3 timeout was
reproduced during qualification, because the fixtures tried decide inside a
one-millisecond limit; the timeout mapping rests on the scripted backend.
