# RefractIR Testing Infrastructure

This directory contains the regression test suite for the RefractIR frontend, CFG reducibility analyses, reference interpreter, compiler backends, and symbolic solver.

## Unified Testing Framework

The RefractIR test suite is managed by a unified Python-based testing framework located in `test/lib/`. It orchestrates the following tools and scripts:

- **`symiri --check`**: Validates static semantics (lexing, parsing, duplicate checks, type checking, and dataflow analysis).
- **`symiri`**: Executes RefractIR programs using the reference interpreter.
- **`symirc`**: Compiles RefractIR programs to target code (C, WASM, Python) and verifies them by linking with a test harness.
- **`symirsolve`**: Solves symbolic RefractIR programs into concrete RefractIR programs given a path.
- **Cross-Validation (`test/lib/run_xval_tests.py`)**: Runs programs through the interpreter *and* the compiled C backend, cross-checking that their execution outputs and UB traps match exactly.
- **Reify Differential Tests (`test/lib/run_reify_diff_tests.py`)**: Executes differential random-testing of generated leaf functions (`rysmith`), composed programs (`rylink`), and twins (`rytwin`).

### Framework Features
- **Automatic Discovery**: Recursively finds all `.sir` files in a given directory.
- **Colored Output**: Uses ANSI colors for clear status reporting (**Green for OK**, **Red for FAIL**, **Yellow for TIMEOUT/SKIP**).
- **Timeouts**: Enforces execution limits (default 5s for tools, 1s for compiled binaries) to detect infinite loops.
- **Sanitization**: Compiler tests are linked with AddressSanitizer and UB Sanitizer to catch memory and semantic errors.

### Metadata Tags

Each `.sir` file should contain metadata tags in the first few lines to guide the runner:

| Tag | Description |
|---|---|
| `// EXPECT: PASS` | The tool is expected to succeed (exit code 0). |
| `// EXPECT: FAIL` | The tool is expected to report an error (non-zero exit code). |
| `// EXPECT: FAIL:<Type>` | The tool must return a specific exit code. Supported: `FAIL:LexError` (2), `FAIL:ParseError` (3), `FAIL:StaticError` (4). |
| `// COMPILER_ARGS: <args>` | CLI arguments passed to `symirc`. |
| `// INTERP_ARGS: <args>` | CLI arguments passed to `symiri`. |
| `// SOLVER_ARGS: <args>` | CLI arguments passed to `symirsolve` (e.g., `--path '^entry,^exit'`). |
| `// SKIP: <TOOL>` | Skip this test for a specific tool or backend target (`INTERPRETER`, `COMPILER`, `SOLVER`, `REDUCIBILITY`, `WASM`, `PYTHON`, `STRUCTURED`, or `ALL`). |

## Writing Tests

### 1. Frontend Static Validation Tests
These tests validate that the frontend correctly identifies statically valid or invalid code (lexing, parsing, CFG structure, type checking, and semantic checking). They are run using the reference interpreter in checking mode: `symiri --check`.

**Example: `test/typechecker/fail_mismatch.sir`**
```sir
// EXPECT: FAIL
fun @main() : i32 {
  let %a: i32 = 1;
  let %b: i64 = 2;
^entry:
  ret %a + %b; // Should fail due to bitwidth mismatch
}
```

### 2. Reducibility & CFG Analysis Tests
These tests verify structural compiler analyses (dominator trees, loop information, reducibility, and structured control trees). They are run using `symirc` with dump arguments (e.g., `--dump-domtree`, `--dump-loops`, `--dump-control-tree`) and match the compiler's stdout against a sibling `<name>.sir.expected` file in `test/reducibility/`.

### 3. Interpreter Dynamic Execution Tests
These tests validate the runtime evaluation and execution semantics of concrete RefractIR programs. They are run using the reference interpreter in execution mode *without* `--check` (`symiri`). To perform runtime assertions, use the `require` instruction.

### 4. Compiler & Structured Backend Tests
These tests validate target code generation (C, WASM, Python) by compiling `.sir` files and executing them.
- **C Target**: Compiles via `symirc --target c`, links with a test driver, and executes with ASan and UBSan enabled. A second run adds `--structured-lowering` (runner flag `--structured`), emitting while/do-while loops instead of goto; irreducible fixtures carry `// SKIP: STRUCTURED`.
- **WASM Target**: Compiles to WebAssembly Text (`symirc --target wasm`) and executes using local WebAssembly runtimes (`wasmtime` or `wasmer`).
- **Python Target**: Compiles to Python scripts (`symirc --target python`) and executes using `python3`.

The core structured backend tests reside in `test/sbackend/` and verify loops, vector lanes, struct arrays, pointer arithmetic, and UB boundaries.

Tensor-only source fixtures use the `.sir.in` suffix under `test/tensor/`. The
recursive scalar compiler and interpreter runners discover `.sir` files, so a
tensor fixture must be copied to a temporary `.sir` path by the tensor test
harness before validation. The smoke-case builder copies these fixtures into
`build/tensor_cases/hand/` and adds binary inputs and references there. This
keeps generated `.sir` files out of recursive scalar discovery; only the
opt-in `test-triton` runner consumes the compiled case bundles.

`make test-tensor` runs the host-only tensor checks. `make test-triton` runs the
case bundles serially in interpreter and compiled modes and requires the
locked NVIDIA/Triton environment. When the source is copied to a remote
directory without `.git` metadata, pass `TENSOR_SOURCE_REVISION` and
`TENSOR_WORKTREE_DIRTY` to Make so each run record identifies the source build.

### 5. Solver Tests
These tests validate the symbolic execution and constraint generation of the `symirsolve` tool, checking that path assumptions and required properties compile down to correct SMT assertions.

### 6. Cross-Validation (xval) Tests
Located in `test/xval/`, these tests compare the output of the interpreter (`symiri`) with the compiled C binary (`symirc`), once with the default goto emission and once with `--structured-lowering`. Any difference in return values, outputs, or UB detection constitutes a cross-validation failure.

### 7. Unit & CLI Tests
Located in `test/unit/`, these are Python unit tests validating parameter formats, rysmith/rylink generation CLI logic, twin passes, and JSON descriptor files.

## Running Tests

The recommended way to run tests is via the `Makefile` targets:

```bash
# Run all tests (Frontend/Reducibility, Interpreter, Compiler, Solver, XVal, and Reify)
make test

# Run specific suites
make test-unit               # Run unit tests only
make test-frontend           # Run frontend checks and CFG reducibility tests
make test-interp             # Run reference interpreter tests
make test-backends           # Run target backend C/WASM/Python codegen and execution
cross-validation             # Run interpreter vs compiled C cross-validation
make test-solver             # Run solver execution and curated SMT examples
make test-reify              # Run differential random reification testing
```

Alternatively, you can run specific suites using the Python module syntax:

```bash
# Run only the interpreter tests
python3 -m test.lib.run_interp_tests test/interp ./symiri

# Run only the compiler tests
python3 -m test.lib.run_compiler_tests test/ ./symirc

# Run only the solver tests
python3 -m test.lib.run_solver_tests test/solver ./symirsolve

# Run only cross-validation tests
python3 -m test.lib.run_xval_tests test/xval ./symiri ./symirc

# Run only CFG reducibility analysis tests
python3 -m test.lib.run_reducibility_tests test/reducibility ./symirc

# Run only a specific frontend check directory
python3 -m test.lib.run_interp_tests test/typechecker ./symiri --check
```
