# R1 runbook: emit-and-compile check of generated kernels on the V100

For the host agent. Generation plan R1: every generation block's kernels must compile on the V100, show their feature in TTGIR or PTX, and, under Triton's interpreter, agree bitwise with P0 wherever P0 is defined. Any disagreement is a defect in G1–G8 before it is anything else. Nothing here is a bug report, and nothing is filed.

**Budget.** One idle GPU, 30 minutes of GPU time. Touch no other process. File nothing.

**Approval.** Merlin approves this run before it starts, and names the commit to sync.

## 0. Environment (as in the calibration spike)

The V100 server's paths exist inside the `great_rhodes` Docker container; start it for the run and stop it afterwards.

```bash
PY=/root/triton-latest/.venv/bin/python
export TRITON_PTXAS_PATH=/root/.triton/nvidia/nvcc/cuda_nvcc-linux-x86_64-12.9.86-archive/bin/ptxas
git -C /root/triton-latest rev-parse HEAD                      # expect 0861b90f...
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
W=/host/ffr/r1-generation
test ! -e $W && mkdir -p $W
```

Record the facts, pick one GPU with no processes as `DEV`, and stop if the Triton revision differs.

## 1. Sources and build

Sync the RefractIR checkout at the commit Merlin names to `$W/refractir` and layout-mutator's `stage-4` checkout to `$W/layout-mutator` (exclude `._*`, `.DS_Store`, `.venv`, `output`). Record both commits.

```bash
cd $W/refractir
make -j"$(nproc)" all CXX=clang++ SOLVER=bitwuzla > $W/build.log 2>&1; echo $? > $W/build.rc
make build/bin/tensor_free_form_export build/bin/tensor_free_form_api build/bin/tensor_path_steering \
  CXX=clang++ SOLVER=bitwuzla >> $W/build.log 2>&1; echo $? >> $W/build.rc
```

Use the same Boost include flag R0 used if the build asks for one. Stop if either return code is not 0.

## 2. Local gate on the host (no GPU)

```bash
cd $W/refractir
make -k test-tensor-emit test-tensor-free-form PY=$PY CXX=clang++ SOLVER=bitwuzla > $W/gate.log 2>&1; echo $? > $W/gate.rc
```

Expected: both targets pass. The Triton-interpreter tests that skip on the Mac run here. Report every failure, and stop if any test whose name contains `general_loops`, `grid_tiles`, `stride_arguments`, `pointwise_dtypes`, `path_steering` or `generation_block` fails.

## 3. Export the generation blocks (no GPU)

```bash
cd $W/refractir
for kind in loops literals approximate tiled strides dtypes; do
  $PY test/tensor/campaigns/export_seed_block.py generation-$kind \
    --seed-start 61000000 --seed-count 10 --out $W/blocks/$kind > $W/blocks-$kind.log 2>&1
  echo "$kind $?" >> $W/blocks.rc
done
```

Each block writes `BLOCK.json`. Expected: at least 8 of 10 bodies exported per block, every entry's status `exported` or `refused` (a builder refusal), and no entry `steering_missed`. A `steering_missed` entry is a defect; record it and continue.

## 4. Compile and run every seed and variant (GPU `DEV`)

For every directory that holds a `kernel.py` (seeds and `variants/*`), run layout-mutator's child twice: once compiled on the GPU, which also saves `ttgir.mlir` and `ptx`, and once under the interpreter.

```bash
cd $W/layout-mutator
export PYTHONPATH=$W/layout-mutator
find $W/blocks -name kernel.py -printf '%h\n' | sort > $W/programs.txt
while read program; do
  rel=${program#$W/blocks/}
  CUDA_VISIBLE_DEVICES=$DEV $PY -m layout_mutator.execute.child --program $program --mode gpu \
    --out $W/runs/$rel/gpu > /dev/null 2>&1
  TRITON_INTERPRET=1 $PY -m layout_mutator.execute.child --program $program --mode interp \
    --out $W/runs/$rel/interp > /dev/null 2>&1
done < $W/programs.txt
```

## 5. Check each block's feature and the interpreter against P0 (no GPU)

Write `$W/check.py` and run it with `$PY $W/check.py > $W/R1-TABLE.md`. For every program it must:

1. read `runs/<rel>/gpu/result.json` and `runs/<rel>/interp/result.json` and record both statuses;
2. when the program's `relations.json` has `expected.p0_file`, compare the interpreter's `tensors/out_ptr.bin` with that file byte for byte, and record `agree`, `differ` (with the first differing element), or `no_p0`;
3. check the block's feature in the compiled artifacts and record `present` or `absent`:

| Block | Feature check |
|---|---|
| loops | `kernel.py` has `tl.range(` with a third argument, and `ttgir.mlir` has `scf.for` or `scf.while` |
| literals | `relations.json` `literals` is nonempty |
| approximate | `ptx` has `div.full.f32`, `ex2.approx`, or (for `rem`) a `cvt.rzi`; the matching operation is in `kernel.py` |
| tiled | `ttgir.mlir` has `tt.get_program_id`, and `launch.json` names `N` and `BLOCK` |
| strides | `launch.json` names `x_stride0` and `x_offset` |
| dtypes | `launch.json` gives `x_ptr` dtype `float16`, and `ptx` has `cvt.f32.f16` |

The table lists one row per program: block, seed, target, GPU status, interpreter status, P0 comparison, feature check. Below it, list the count per block of programs that compiled, that show their feature, and whose interpreter agrees with P0.

## 6. Bring back

`R1-TABLE.md`, `BLOCK.json` of every block, `build.rc`, `gate.log`, `gate.rc`, `blocks.rc`, the step-0 facts, both commits, and for every program with a compile failure, an interpreter disagreement, or an absent feature: its directory (seed, `runs/`) as a tarball. Stop the container.

## Outcome

- **Pass.** Every exported program compiles. Every block shows its feature in at least 8 of its programs. No interpreter output differs from P0. R2 (the relations run) is then prepared.
- **Fail.** Any difference from P0, a block without its feature, or a `steering_missed` entry goes back to RefractIR as a defect in the named task (G1 loops, G4 literals, G5 operations, G6 tiling, G7 strides, G8 dtypes, G3 steering) before R2.
