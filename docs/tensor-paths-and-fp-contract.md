# Tensor kernel paths and the floating-point comparison contract

A generated Triton kernel is compared on two paths: P1, Triton's interpreter, and P2, the GPU. P0, the SIR interpreter, is the third. This document defines what varies between runs of one kernel (its path) and which floating-point results the paths must agree on bit for bit.

## Paths

A tensor kernel has no irreducible control flow; its path is the set of runtime-decided quantities the compiler cannot fold.

| Path component | How a generated kernel sets it |
|---|---|
| Trip count of each counted loop | a literal in SIR, or an `i32` entry parameter whose range `assume` statements give; `symirc --tensor-loop-bound runtime` passes a literal as an unspecialized kernel parameter whose default is the literal |
| Lanes masked at a window boundary | the window's rows, stride, and active bound |
| Arm of each per-lane `where` | the inputs, through the comparison each lane makes |
| Scalar branches | a diamond on a comparison of `i32` scalars; generated as the `%sel` selector, assumed 0 or 1, which each case draws from its seed |
| Nesting | an outer counted loop around the window loop, within eight body executions in all |

A literal bound gives one to eight trips ([tensor capability analysis](../src/analysis/tensor.cpp)). A parameter bound `%n` needs `assume %n >= lo;` and `assume %n <= hi;` in `^entry`, with `lo` at or above the loop's start and `hi` allowing one to eight trips, so the loop may run zero times; P0 runs the trips the argument gives and refuses an argument outside the range, and the emitted kernel loads the argument once and loops to it. A scalar parameter travels in a bundle as a one-element input of shape `[]`. A runtime bound with its literal default computes the literal program, so P0 needs no change to compare it. Excluding the parameter from specialization matters because Triton turns an integer argument equal to one into a constant.

## Comparison classes

SIR's domain is finite: NaN and infinities are undefined behaviour in P0 ([float.md](./float.md)). Every class below therefore compares finite outputs only.

| Class | Operations | Comparison |
|---|---|---|
| Specified | `add`, `sub`, `mul`, `min`, `max`, `rmin`, `rmax`, `narrow f16`, `narrow bf16`, and data movement (`window`, `transpose`, `expand`, `broadcast`, `where`) | bit-exact against P0 at any finite value |
| Pinned by a launch option | elementwise operations under `enable_fp_fusion=False`, which forbids contracting a multiply and an add into one rounding | bit-exact once pinned |
| Unspecified | `sum`, f32 `dot`, and f32 `cumsum`, whose accumulation order the backend chooses; `div`, `exp`, and `rem`, whose lowerings are not one correctly rounded operation | enclosure, or excluded from the comparison |

`min` and `max` resolve a tie between zeros of opposite sign by the IEEE minNum/maxNum rule; Triton leaves that tie open, so the emitter computes it explicitly ([the Triton backend](../src/backend/triton_backend.cpp)). An `i32` `dot` is exact and compared like any integer result. SIR has no `sqrt` or low-precision floating-point type; `narrow` rounds an f32 value to f16 or bf16 and keeps it in f32. On the V100 (sm_70, Triton `0861b90`, 2026-11-02), f16 narrowing is one `cvt.rn.f16.f32`. bf16 has no native conversion below sm_80, so Triton emits a round-to-nearest-even on bits instead: it adds `0x7FFF` plus bit 16, quiets NaNs, and clears the low 16 bits. Simulated instruction by instruction, that sequence agrees with exact rounding on 212,288 bit patterns, so both rows are specified on that host. The rows are re-recorded on any other host before a sweep there.

On the Triton build of the 2026-10-29 server run, `/` lowers to `div.full.f32` (approximate), `exp` to `ex2.approx.f32`, and `rem` to `div.rn.f32`, `cvt.rzi.f32.f32`, and `fma.rn.f32`, which compute a - trunc(a / b) * b. That formula is not IEEE `fmod`: when a / b rounds up to an integer, the truncated quotient is one too large. Every other operation in the table above lowers to one `.rn` instruction, `min.f32`, or `max.f32`, plus the exact signed-zero tie computation.

## Unspecified results

A `sum` of n terms x_i, computed in any order with round-to-nearest-even, lies within gamma_(n-1) * sum|x_i| of the exact sum, where gamma_k = k*u / (1 - k*u) and u = 2^-24. When every term is a multiple of one power of two g and sum|x_i| <= 2^24 * g, every partial sum is representable, so every order gives the exact sum. An f32 `dot` lane is a sum of K exact products a_ik * b_kj. Accumulating them with K fused multiply-adds, in any order, rounds K times, so the lane lies within gamma_K * sum|a_ik * b_kj| of the exact inner product; an f32 `cumsum` prefix of k terms is enclosed like a `sum` of those k terms. On the V100 the f32 `dot` lowers to `fma.rn.f32` and `cumsum` to `add.rn.f32` only, which these bounds cover; on any other host the PTX is re-recorded before a sweep. [tensor_enclosure.py](../test/lib/tensor_enclosure.py) evaluates a body over these bounds; an output lane whose enclosure is a single value is compared bit-exactly.

The moving-window sweep compares a `sum` body's outputs on the exact window probes I0 to I3, where the exporter's reduction contract keeps every sum within 2^23 for inputs of magnitude at most 5. On the corner probes and steered inputs, which that contract does not bound, each path output of a body containing `sum` is checked lane by lane against that input's enclosure: a lane outside it, or a single-value lane with other bits, is a candidate, and two paths that differ inside it are not. A single-value lane of zero accepts either sign, since the sign of a sum of zeros depends on where the sum starts. When the enclosure cannot be computed, the outputs are not compared.

## Evidence

[fp_contract_evidence.py](../test/tensor/campaigns/fp_contract_evidence.py) emits each floating-point tensor operation alone, compiles it without a launch at one warp and one stage with fusion off, and lists the PTX f32 instructions. An operation belongs to the specified class only if none of its instructions is approximate (`.approx`, `div.full`), flushes subnormals (`.ftz`), or is a fused multiply-add. Fusion is off, so an `fma` in a one-operation kernel means the operation is computed by a formula. Run it where Triton and the CUDA toolchain are installed:

```bash
python test/tensor/campaigns/fp_contract_evidence.py --out fp-contract-evidence
```
