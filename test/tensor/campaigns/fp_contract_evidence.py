"""PTX evidence for the floating-point comparison contract (plan W3).

Each SIR floating-point tensor operation is emitted alone as a Triton kernel
and compiled, without a launch, at one warp and one stage with fusion off. The
PTX f32 instructions the kernel uses then show which operations lower exactly:

  approximate  `.approx` or `div.full`: not correctly rounded.
  ftz          subnormal inputs or results are flushed to zero.
  fma          a fused multiply-add, which rounds once for two operations.

An operation is specified only if none of the three appears.

The table backs the specified / pinned / unspecified classes of
`docs/tensor-paths-and-fp-contract.md`. Compiling needs Triton and the CUDA
toolchain; emitting and classifying do not.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

SCHEMA = "refractir.fp-contract-evidence.v1"

_BINARY = """\
fun @kernel(%x: tensor [8] f32, %y: tensor [8] f32) : tensor [8] f32 {{
  let mut %out: tensor [8] f32 = undef;
^entry:
  %out = tensor {op} %x, %y;
  ret %out;
}}
"""
_UNARY = """\
fun @kernel(%x: tensor [8] f32) : tensor [8] f32 {{
  let mut %out: tensor [8] f32 = undef;
^entry:
  %out = tensor {op} %x;
  ret %out;
}}
"""
_REDUCTION = """\
fun @kernel(%x: tensor [4, 8] f32) : tensor [4, 1] f32 {{
  let mut %out: tensor [4, 1] f32 = undef;
^entry:
  %out = tensor {op} %x, 1, keep;
  ret %out;
}}
"""
_SCAN = """\
fun @kernel(%x: tensor [4, 8] f32) : tensor [4, 8] f32 {
  let mut %out: tensor [4, 8] f32 = undef;
^entry:
  %out = tensor cumsum %x, 1;
  ret %out;
}
"""
# `tl.dot` needs every extent to be at least 16.
_DOT = """\
fun @kernel(%x: tensor [16, 16] f32, %y: tensor [16, 16] f32) : tensor [16, 16] f32 {
  let mut %out: tensor [16, 16] f32 = undef;
^entry:
  %out = tensor dot %x, %y;
  ret %out;
}
"""
# Each operation with the lowest tensor profile that accepts it: v3 has no
# `div` or `exp`, and only v4 has `rem`.
OPS = {
  **{op: ("v3", _BINARY.format(op=op)) for op in ("add", "sub", "mul")},
  "div": ("v2", _BINARY.format(op="div")),
  "rem": ("v4", _BINARY.format(op="rem")),
  **{op: ("v3", _BINARY.format(op=op)) for op in ("min", "max")},
  "exp": ("v2", _UNARY.format(op="exp")),
  **{op: ("v3", _REDUCTION.format(op=op)) for op in ("sum", "rmin", "rmax")},
  "dot": ("v3", _DOT),
  "cumsum": ("v3", _SCAN),
  **{
    f"narrow-{fmt}": ("v3", _UNARY.format(op=f"narrow {fmt}"))
    for fmt in ("f16", "bf16")
  },
}

_F32 = re.compile(
  r"\b((?:add|sub|mul|div|fma|rcp|sqrt|rsqrt|ex2|lg2|max|min|neg|abs|cvt)(?:\.[a-z0-9]+)*\.f32)\b"
)


def classify_ptx(text):
  """The distinct f32 instructions in PTX text and what they imply."""
  instructions = sorted(set(_F32.findall(text)))
  found = {
    "instructions": instructions,
    "approximate": any(
      ".approx" in name or name.startswith("div.full") for name in instructions
    ),
    "ftz": any(".ftz" in name for name in instructions),
    "fma": any(name.startswith("fma") for name in instructions),
  }
  # Fusion is off, so an fma in a one-operation kernel means the operation is
  # computed by a formula (rem: a - trunc(a / b) * b), not one IEEE operation.
  # With no f32 instruction at all (a conversion done in integer arithmetic,
  # as bf16 is below sm_80) the table cannot vouch for the operation.
  found["specified"] = bool(instructions) and not (
    found["approximate"] or found["ftz"] or found["fma"]
  )
  return found


def _compile_matrix():
  path = Path(__file__).with_name("compile_kernel_stages.py")
  spec = importlib.util.spec_from_file_location("compile_kernel_stages", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module.compile_matrix


def _emit(profile, source, directory, symirc):
  directory.mkdir(parents=True, exist_ok=True)
  (directory / "program.sir").write_text(source, encoding="utf-8")
  kernel = directory / "kernel.py"
  completed = subprocess.run(
    [
      str(symirc),
      "--target",
      "triton",
      "--entry",
      "kernel",
      "--tensor-profile-version",
      profile,
      "-o",
      str(kernel),
      str(directory / "program.sir"),
    ],
    capture_output=True,
    text=True,
    check=False,
    timeout=60,
  )
  return kernel if completed.returncode == 0 else completed.stdout + completed.stderr


def collect(out, *, symirc, compile_matrix=None):
  """Emit, compile, and classify every operation; write `evidence.json`."""
  compile_matrix = compile_matrix or _compile_matrix()
  out = Path(out)
  rows = []
  for op, (profile, source) in OPS.items():
    directory = out / op
    kernel = _emit(profile, source, directory, symirc)
    if not isinstance(kernel, Path):
      rows.append(
        {
          "op": op,
          "profile": profile,
          "status": "emission_failed",
          "error": kernel[-2000:],
        }
      )
      continue
    table = compile_matrix(kernel, warps=[1], stages=[1], out=directory)
    setting = table["settings"][0]
    if setting["status"] != "compiled":
      rows.append(
        {
          "op": op,
          "profile": profile,
          "status": "failed",
          "error": setting.get("error", ""),
        }
      )
      continue
    ptx = (directory / "warps-1-stages-1" / "ptx.ir").read_text(encoding="utf-8")
    rows.append(
      {"op": op, "profile": profile, "status": "compiled", **classify_ptx(ptx)}
    )
  table = {"schema": SCHEMA, "enable_fp_fusion": False, "ops": rows}
  out.mkdir(parents=True, exist_ok=True)
  (out / "evidence.json").write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
  return table


def main(argv=None):
  import argparse

  root = Path(__file__).resolve().parents[3]
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--symirc", type=Path, default=root / "symirc")
  arguments = parser.parse_args(argv)
  table = collect(arguments.out, symirc=arguments.symirc.resolve())
  for row in table["ops"]:
    flags = [flag for flag in ("approximate", "ftz", "fma") if row.get(flag)]
    flags = flags or (["specified"] if row.get("specified") else [])
    detail = ", ".join(row.get("instructions", [])) or row.get("error", "")[:80]
    print(f"{row['op']:<6} {row['status']:<15} {' '.join(flags) or '-':<22} {detail}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
