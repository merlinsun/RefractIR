# symiri

`symiri` is the reference interpreter for RefractIR. It executes a `.sir` program directly, and its behaviour is the definition of correct execution: the compiler backends and the solver are validated against it.

## Usage

```bash
symiri <input.sir> [args...] [--main <function>] [--sym name=value ...] [options]
```

Entry defaults to `@main`. Positional arguments after the input file bind the entry function's parameters in declaration order. The full option list is `symiri --help`, declared in [src/symiri.cpp](../src/symiri.cpp).

Interpret a concrete program, then a specific entry function:

```bash
symiri prog.sir
symiri prog.sir --main @f0
```

Interpret a symbolic program, binding every symbol it declares:

```bash
symiri prog.sir --main @f0 --sym @?c4=3 --sym %?k=10
```

## Symbol binding

A program that declares symbols runs only when every symbol is bound; a missing or unrecognised binding is an error. Bound values are immutable for the whole run, exactly as a solver-chosen value would be.

A value is parsed as a decimal literal against the symbol's declared type: integers against the declared bit width, floats through `refractir::parseFloatLiteral` ([float.md](./float.md) §9).

A vector symbol takes one value per lane, `--sym '%?v=1,2,3,4'`, and the count must equal the lane count. A single value splats across all lanes.

## Execution

Expressions evaluate left to right, `select` evaluates only the arm it takes, and division and modulo round toward zero (spec §2.4, §2.7, §8).

Undefined behaviour aborts the run with a diagnostic naming the rule and the source span, and the process exits non-zero. [undefined.md](./undefined.md) states each rule and how the interpreter enforces it.

`--check` stops after the frontend, reporting type and semantic errors without executing. `--dump-trace` prints each executed block and every variable update, in typed tensor runs too. `--max-bbl-steps <n>` aborts after entering `n` blocks, which bounds a program that may not terminate. `--dump-call` prints each function calls.

## Typed tensor execution

`--tensor-profile-version v1|v2|v3|v4` selects the bounded tensor profile (default `v1`). `--tensor-describe --entry @kernel` prints the checked entry signature and its little-endian buffer layout. `--entry @kernel --tensor-input inputs.bin --tensor-output output.bin` runs that entry with input buffers in parameter order and writes the complete result buffer. The input and output options must be supplied together. [Tensor capability analysis](../src/analysis/tensor.cpp) enforces the profile; [tensor I/O](../src/interp/tensor_io.cpp) implements the binary format.

A tensor window reads row `r` of its tensor parameter at address `index * stride + r`, computed in 64-bit arithmetic, while that address is below `active`, and reads its literal fill otherwise. A position is active when its address lies in `[0, active)`; a position below zero or at or above `active` takes the fill without accessing the source. Windows are admitted under the v2, v3, and v5 profiles.

Numerical tensor atoms use the Boost.Multiprecision reference when built with `TENSOR_REF=1`, which is the default. `--tensor-reference-bits 128|256` selects its precision, and `--tensor-reference-stability` compares each atom at both precisions and rejects an unstable result. A `TENSOR_REF=0` build retains exact tensor execution and reports numerical reference requests as unavailable.

## Result reporting

A terminating run prints the entry function's return value on a `Result:` line: a decimal integer, a hex float, `ptr(0x…)`, or `void`. Hex-float form (`printf("%a", …)`) is bit-exact by construction, so cross-validation compares the line against the C backend's output byte for byte ([float.md](./float.md) §11.4).
