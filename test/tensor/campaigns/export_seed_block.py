"""Export a block of generated bodies as layout-mutator seeds (seed-export plan, E4).

Each request of the block is built as the moving-window sweep would build it
(its seed, column, policy, trip count and branch selector), but with every
admission rule that exists only to certify P0's output switched off: no
exact-input bound and no exact contract, no motif requirement, and the node
and multiplication caps raised. Each body becomes one seed directory whose own
inputs are I0; the other probes are named variants under `variants/`, each a
complete seed directory with the same kernel. A body the builder refuses is
logged in `BLOCK.json` and not exported. The block's digest covers every file
it wrote.

Blocks follow the census order: dot, narrow, where, nested, scan.

Usage:
  python export_seed_block.py dot --seed-start N --seed-count K --out DIR \\
    [--exporter PATH] [--symirc PATH] [--symiri PATH] [--narrow-bf16 specified]
"""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from test.lib import (  # noqa: E402
  tensor_free_form_case,
  tensor_free_form_inputs,
  tensor_seed_export,
)

_spec = importlib.util.spec_from_file_location(
  "moving_window_sweep", Path(__file__).with_name("run_moving_window_sweep.py")
)
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

SCHEMA = "refractir.seed-block.v1"
# The sweep caps bodies at 16 nodes; without the exact contract a larger body
# needs no extra certification, so the export doubles it. There is no stated
# compiler limit; this is a choice, recorded in every block.
EXPORT_MAX_NODES = 32
# I0 is the seed's own input; the rest are its named variants.
VARIANTS = ("i0-base", "i3-unused-rows", *sweep.CORNER_PROBES)
STUDY = ["expand", "extrema", "transpose"]
BLOCKS = {
  "dot": {"columns": sweep.DOT_COLUMNS, "features": ["dot", *STUDY]},
  "narrow": {"columns": sweep.COLUMNS, "features": ["narrow", *STUDY]},
  "where": {"columns": sweep.COLUMNS, "features": ["where"]},
  "nested": {
    "columns": sweep.NESTED_COLUMNS,
    "features": STUDY,
    "branch_parameter": True,
    "outer_trips": 2,
  },
  "scan": {"columns": sweep.COLUMNS, "features": ["scan", *STUDY]},
}


def _spec_for(feature, seed_start, seed_count=1):
  options = dict(BLOCKS[feature])
  return sweep.sweep_spec(
    seed_start=seed_start,
    seed_count=seed_count,
    configurations=[{"num_warps": 1, "num_stages": 1}],
    sample_every=1,
    gpu_workers=1,
    wall_clock_s=1.0,
    columns=options.pop("columns"),
    trip_parameter=True,
    **options,
  )


def block_request(feature, seed_start, index, spec=None):
  """The sweep's request for one index, with the certification rules off."""
  request = sweep.kernel_request(spec or _spec_for(feature, seed_start), index)
  request.update(
    requirement=None,
    max_nodes=EXPORT_MAX_NODES,
    max_multiplications=EXPORT_MAX_NODES,
    exact_input_bound=None,
    exact_contract=None,
  )
  return request


def export_command(exporter, request, out):
  return sweep._export_command(exporter, request, out)


def _probes(region, request):
  column = request["column"]["column"]
  group = sweep.GROUPS.index(request["group"]) % 3
  path = {"trips": request.get("trips"), "branch": request.get("branch")}
  probes = tensor_free_form_inputs.window_probes(
    region, group=group, column=column, **path
  ) + tensor_free_form_inputs.corner_probes(region, group=group, column=column, **path)
  by_name = {probe.probe: probe for probe in probes}
  return [(name, by_name[name]) for name in VARIANTS if name in by_name]


def _export_body(request, out, **tools):
  """Export one body, or return why not; a seed enters `out` only whole."""
  with tempfile.TemporaryDirectory() as scratch:
    scratch = Path(scratch)
    completed = subprocess.run(
      export_command(tools["exporter"], request, scratch / "export"),
      capture_output=True,
      text=True,
      check=False,
      timeout=120,
    )
    if completed.returncode != 0:
      reason = (completed.stderr or completed.stdout).strip()[-500:]
      return {"status": "refused", "reason": reason or "the exporter failed"}
    region = json.loads((scratch / "export" / "region.json").read_text("utf-8"))
    source = (scratch / "export" / "program.sir").read_text("utf-8")
    try:
      staged = _stage_variants(request, region, source, scratch, tools)
    except tensor_seed_export.ExportError as error:
      return {"status": "export_failed", "reason": str(error)}
    seed_dir = out / f"seed-{request['seed']}"
    shutil.move(str(staged), str(seed_dir))
    return {"status": "exported", "directory": seed_dir.name}


def _stage_variants(request, region, source, scratch, tools):
  """The seed directory with I0 as its own input and the rest as variants."""
  staged = scratch / "staged"
  for position, (name, probe) in enumerate(_probes(region, request)):
    case = tensor_free_form_case.build_bundle(
      scratch / "bundles" / name, region=region, source=source, proposal=probe
    )
    exported = tensor_seed_export.export_seed(
      case.path,
      region,
      scratch / "seeds" / name,
      symirc=tools["symirc"],
      symiri=tools["symiri"],
      narrow_bf16=tools["narrow_bf16"],
      provenance={
        "seed": request["seed"],
        "input": name,
        "column": request["column"]["column"],
        "policy": request["policy"],
        "exporter_sha256": tensor_seed_export._sha256(
          Path(tools["exporter"]).read_bytes()
        ),
      },
    )
    target = staged if position == 0 else staged / "variants" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    exported.rename(target)
  return staged


def _digest(out):
  lines = [
    f"{path.relative_to(out)} {hashlib.sha256(path.read_bytes()).hexdigest()}"
    for path in sorted(out.rglob("*"))
    if path.is_file() and path.name != "BLOCK.json"
  ]
  return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def export_block(
  feature,
  *,
  seed_start,
  seed_count,
  out,
  exporter,
  symirc,
  symiri,
  narrow_bf16="excluded",
):
  """Export every request of the block; write and return `BLOCK.json`."""
  out = Path(out)
  out.mkdir(parents=True, exist_ok=True)
  spec = _spec_for(feature, seed_start, seed_count)
  entries = []
  for index in range(seed_count):
    request = block_request(feature, seed_start, index, spec)
    outcome = _export_body(
      request,
      out,
      exporter=exporter,
      symirc=symirc,
      symiri=symiri,
      narrow_bf16=narrow_bf16,
    )
    entries.append({"seed": request["seed"], **outcome})
  record = {
    "schema": SCHEMA,
    "feature": feature,
    "seed_start": seed_start,
    "seed_count": seed_count,
    "max_nodes": EXPORT_MAX_NODES,
    "narrow_bf16": narrow_bf16,
    "variants": list(VARIANTS),
    "entries": entries,
    "exported": sum(entry["status"] == "exported" for entry in entries),
    "digest": _digest(out),
  }
  (out / "BLOCK.json").write_text(
    json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8"
  )
  return record


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("feature", choices=sorted(BLOCKS))
  parser.add_argument("--seed-start", type=int, required=True)
  parser.add_argument("--seed-count", type=int, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument(
    "--exporter", type=Path, default=ROOT / "build" / "bin" / "tensor_free_form_export"
  )
  parser.add_argument("--symirc", type=Path, default=ROOT / "symirc")
  parser.add_argument("--symiri", type=Path, default=ROOT / "symiri")
  parser.add_argument(
    "--narrow-bf16", choices=("specified", "excluded"), default="excluded"
  )
  arguments = parser.parse_args(argv)
  record = export_block(
    arguments.feature,
    seed_start=arguments.seed_start,
    seed_count=arguments.seed_count,
    out=arguments.out,
    exporter=arguments.exporter,
    symirc=arguments.symirc,
    symiri=arguments.symiri,
    narrow_bf16=arguments.narrow_bf16,
  )
  print(json.dumps({"exported": record["exported"], "digest": record["digest"]}))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
