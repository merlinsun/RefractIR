"""Route free-form bodies to the Gluon layout runner (feature-ranked plan, L5).

A two-state free-form body without a window has a canonical adapter in the
graph-region signature (`fun @kernel(%x: ...)`), which the Gluon lowering in
`test/lib/tensor_gluon_lowering.py` reads. For each seed this exports the body
with `--canonical-out`, renders the canonical program once without layout
plans, and keeps it when the lowering accepts it. The kept programs are the
input of `test/tensor/gluon/run_generated_region.py --program`, which compiles
them under several layout plans on a GPU and compares each run with P0.

Rendering needs no Triton; compiling does, so this runs on any host.

Usage:
  python build_gluon_corpus.py --exporter EXPORTER --seed-start N --seed-count K \
    --shape 2,8 --out DIR
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from test.lib.tensor_gluon_lowering import render_gluon_kernel  # noqa: E402

SCHEMA = "refractir.gluon-corpus.v1"


def sha256(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _entry(exporter, seed, shape, out):
  directory = out / f"seed-{seed}"
  completed = subprocess.run(
    [
      str(exporter),
      "--policy",
      "unguided",
      "--seed",
      str(seed),
      "--shape",
      f"{shape[0]},{shape[1]}",
      "--states",
      "2",
      "--profile",
      "v3",
      "--canonical-out",
      "--out",
      str(directory),
    ],
    capture_output=True,
    text=True,
    check=False,
    timeout=120,
  )
  if completed.returncode != 0:
    reason = (completed.stderr or completed.stdout).strip()[-500:]
    return {"seed": seed, "status": "refused", "reason": reason}
  program = directory / "canonical.sir"
  if not program.exists():
    region = json.loads((directory / "region.json").read_text(encoding="utf-8"))
    return {
      "seed": seed,
      "status": "no_canonical",
      "reason": region["canonical"]["refusal"] or "no canonical adapter",
    }
  try:
    kernel = render_gluon_kernel(program.read_text(encoding="utf-8"))
  except ValueError as error:
    return {"seed": seed, "status": "unsupported", "reason": str(error)}
  kernel_path = directory / "gluon_kernel.py"
  kernel_path.write_text(kernel.source, encoding="utf-8")
  return {
    "seed": seed,
    "status": "rendered",
    "program": str(program.relative_to(out)),
    "program_sha256": sha256(program),
    "kernel": str(kernel_path.relative_to(out)),
    "kernel_sha256": sha256(kernel_path),
  }


def build(exporter, *, seeds, shape, out):
  """Export, render, and record every seed; write `CORPUS.json` into `out`."""
  out = Path(out)
  out.mkdir(parents=True, exist_ok=True)
  entries = [_entry(exporter, seed, shape, out) for seed in seeds]
  manifest = {
    "schema": SCHEMA,
    "shape": list(shape),
    "exporter_sha256": sha256(exporter),
    "entries": entries,
    "rendered": sum(entry["status"] == "rendered" for entry in entries),
  }
  (out / "CORPUS.json").write_text(
    json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
  )
  return manifest


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--exporter", type=Path, required=True)
  parser.add_argument("--seed-start", type=int, required=True)
  parser.add_argument("--seed-count", type=int, required=True)
  parser.add_argument("--shape", default="2,8")
  parser.add_argument("--out", type=Path, required=True)
  arguments = parser.parse_args(argv)
  shape = tuple(int(part) for part in arguments.shape.split(","))
  seeds = range(arguments.seed_start, arguments.seed_start + arguments.seed_count)
  manifest = build(arguments.exporter, seeds=seeds, shape=shape, out=arguments.out)
  print(
    json.dumps({"rendered": manifest["rendered"], "seeds": len(manifest["entries"])})
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
