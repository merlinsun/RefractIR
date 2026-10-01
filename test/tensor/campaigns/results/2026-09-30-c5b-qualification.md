# Qualification of the selection-boundary fixes

Snapshot: tracked-only export of revision
`d1580adddc4d955533f599b5ff6ae00899f0927f`, archive sha256
`146d4921fccbbf72c22163941917f85347c84ac822e7beb5d687990c5aa0661e`, extracted
without a `.git` directory so the runner metadata is stated rather than
inferred. This supersedes the qualification of `2bd7ab3` for the supervised
selection path; the initializer, generator, and audit results are unchanged.

## Host gates

| Gate | Result |
|---|---|
| `make -j16 all SOLVER=bitwuzla TENSOR_REF=1` | exit 0, no warnings |
| `make -j16 test-tensor` | exit 0 (257, 279, and 24 tests) |
| `make -j16 test-solver` | exit 0 |
| `make -j16 cross-validation` | exit 0 |
| `make test-solver-outcomes` | exit 0 (41 tests) |
| `make test-tensor-graph-audit` | exit 0 (7 tests) |
| `make test-tensor-property-pilot` | exit 0 (44 tests) |
| `make -j16 test-unit` | exit 0 |
| `make test-triton` | exit 0 on the second run |

The first `make test-triton` run reported one case with `campaign_valid:
false` while its comparison passed. The case's environment record showed
`campaign_available: false`, an environment-probe result rather than a property
of the case; the case re-ran clean, and a fresh run of the whole gate reports
exit 0 with no invalid campaign. It is recorded here as a transient probe
failure, since a single clean re-run does not prove the probe is reliable.

## Generated corpus on the pinned GPU

Seven targets, both Triton modes, the same list as the earlier run.

| Target | Generation | interpret | compiled |
|---|---|---|---|
| v2 i32 ordinary, seed 1 | accepted | pass | pass |
| v2 i32 ordinary, seed 2 | accepted | pass | pass |
| v2 f32 ordinary, seed 1 | accepted | pass | pass |
| v2 f32 ordinary, seed 2 | accepted | pass | pass |
| v3 i32 ordinary, seed 1 | accepted | pass | pass |
| v2 i32 exact, target 3, seed 2 | accepted | pass | pass |
| v2 i32 exact, target 3, seed 5 | refused, `input_policy_unsat` | not run | not run |

Twelve executions, twelve passes, and the refusal is reported rather than
substituted.

## Not re-run on this snapshot

`make test` was qualified on `2bd7ab3` with ten classified exceptions: five
frame-dangling interpreter cases and five cases whose expected runtime failure
does not occur, both reproduced from an untouched baseline export before this
batch. It was not repeated here. Nothing in the selection-boundary fixes
touches the interpreter or the backends that group covers, so the expectation
is that the same ten failures remain the only ones; that expectation is not
evidence, and the gate is outstanding for this revision.
