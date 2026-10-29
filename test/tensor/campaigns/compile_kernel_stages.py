"""Compile one emitted Triton kernel at several launch settings, without launching.

The paired runner has no compile-only mode, so the toolchain capability check
(plan section 6.5, H2a) compiles the kernel file directly with
`triton.compile` and keeps each setting's TTIR, TTGIR, LLIR, and PTX. Every
kernel parameter is a pointer of one declared type, as in the moving-window
emissions and a standalone dot kernel written the same way, except a runtime
loop bound, which is an i32.
A setting that fails to compile is recorded with its error, and the others
still run. The table states, per warp count, whether TTGIR differs between
`num_stages` values, which decides whether that axis is kept.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import runpy
import sys
from pathlib import Path

SCHEMA = "refractir.compile-kernel-stages.v1"
STAGES = ("ttir", "ttgir", "llir", "ptx")


def _signature(kernel, arg_type):
  """Pointers of the declared type; an integer-defaulted parameter is an i32.

  `symirc --tensor-loop-bound runtime` passes each loop bound as a parameter
  defaulting to its literal, so it compiles as a runtime integer.
  """
  defaults = inspect.signature(kernel.fn).parameters
  return {
    name: "i32" if type(defaults[name].default) is int else arg_type
    for name in kernel.arg_names
  }


def _compile(kernel, *, arg_type, num_warps, num_stages):
  import triton
  from triton.compiler import ASTSource

  source = ASTSource(fn=kernel, signature=_signature(kernel, arg_type))
  return triton.compile(
    source,
    options={
      "num_warps": num_warps,
      "num_stages": num_stages,
      "enable_fp_fusion": False,
    },
  )


def compile_matrix(kernel_path, *, warps, stages, out, arg_type="*fp32"):
  """Compile every (warps, stages) setting and write the stages and a table."""
  kernel_path = Path(kernel_path)
  out = Path(out)
  kernel = runpy.run_path(str(kernel_path))["kernel"]
  settings = []
  for num_warps in warps:
    for num_stages in stages:
      row = {"num_warps": num_warps, "num_stages": num_stages}
      try:
        compiled = _compile(
          kernel, arg_type=arg_type, num_warps=num_warps, num_stages=num_stages
        )
      except Exception as error:
        settings.append(
          {**row, "status": "failed", "error": f"{type(error).__name__}: {error}"}
        )
        continue
      directory = out / f"warps-{num_warps}-stages-{num_stages}"
      directory.mkdir(parents=True, exist_ok=True)
      hashes = {}
      for stage in STAGES:
        text = compiled.asm.get(stage)
        if isinstance(text, str):
          (directory / f"{stage}.ir").write_text(text, encoding="utf-8")
          hashes[stage] = hashlib.sha256(text.encode("utf-8")).hexdigest()
      settings.append({**row, "status": "compiled", "sha256": hashes})
  table = {
    "schema": SCHEMA,
    "kernel": str(kernel_path),
    "kernel_sha256": hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
    "arg_type": arg_type,
    "settings": settings,
    "ttgir_changes_with_stages": _stage_effect(settings, warps),
  }
  out.mkdir(parents=True, exist_ok=True)
  (out / "table.json").write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
  return table


def _stage_effect(settings, warps):
  """Per warp count: whether TTGIR differs across stages; None if any failed."""
  effect = {}
  for num_warps in warps:
    rows = [row for row in settings if row["num_warps"] == num_warps]
    if any(row["status"] != "compiled" for row in rows):
      effect[str(num_warps)] = None
      continue
    effect[str(num_warps)] = len({row["sha256"].get("ttgir") for row in rows}) > 1
  return effect


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("kernel", type=Path)
  parser.add_argument("--warps", type=int, nargs="+", default=[1, 2, 4, 8])
  parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3])
  parser.add_argument("--arg-type", default="*fp32")
  parser.add_argument("--out", type=Path, required=True)
  arguments = parser.parse_args(argv)
  table = compile_matrix(
    arguments.kernel,
    warps=arguments.warps,
    stages=arguments.stages,
    out=arguments.out,
    arg_type=arguments.arg_type,
  )
  json.dump(table, sys.stdout, indent=1, sort_keys=True)
  print()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
