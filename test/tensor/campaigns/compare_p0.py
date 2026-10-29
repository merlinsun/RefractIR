"""Compare retained moving-window sweep runs with P0, the SIR interpreter.

A sweep keeps each admitted kernel's SIR, bundle, and inputs, and records the
SHA-256 of every P1 (Triton interpreter) and P2 (GPU) output in its run log.
This tool runs `symiri` once per body and input over the retained bundle and
compares P0's output hash with each recorded path hash. Nothing in the shard
is modified; P0 outputs are written to a scratch directory.

Each recorded path output is classed as:

  agree        P0 produced the same bytes.
  differ       P0 completed with other bytes.
  restricted   not compared: the body contains an operation whose result
               order or precision the backend chooses (`sum`), on a corner
               probe whose values the exact-input contract does not bound.
  p0_ub        P0 stopped on undefined behaviour, such as a non-finite value.
  unavailable  P0 or the path left nothing to compare, or the bundle is gone.

A disagreement names the body's operations, the path, and the output; when
the sweep retained the raw output, it also lists the differing words.
"""

from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from test.lib import tensor_target_case, tensor_target_reference  # noqa: E402

_spec = importlib.util.spec_from_file_location(
  "moving_window_sweep", Path(__file__).with_name("run_moving_window_sweep.py")
)
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

SCHEMA = "refractir.moving-window-p0-comparison.v1"
UNSPECIFIED = sweep.UNSPECIFIED
MAX_WORDS = 16
_MODES = {"p1": "interpret", "p2": "compiled"}


def _sha(data):
  return hashlib.sha256(data).hexdigest()


def _kernels(shard):
  """The run log's admitted kernel lines."""
  with (Path(shard) / "run-log.jsonl").open(encoding="utf-8") as log:
    for line in log:
      record = json.loads(line)
      if record.get("event") == "kernel" and record.get("status") != "refused":
        yield record


def _operations(root):
  try:
    region = json.loads((root / "export" / "region.json").read_text(encoding="utf-8"))
  except (OSError, ValueError):
    return None
  return list((region.get("live_dag") or {}).get("ops") or [])


def _bundle(root, pair):
  return (
    root / "pairs" / sweep._pair_name(pair["configuration"], pair["input"]) / "bundle"
  )


def _observe(bundle, *, symiri, timeout_s, output_path):
  """P0 over one retained bundle, or an unavailable observation."""
  try:
    case = tensor_target_case.load(bundle, symiri=symiri)
  except (OSError, tensor_target_case.TargetCaseError) as error:
    return tensor_target_reference.ReferenceObservation(
      tensor_target_reference.UNAVAILABLE, reason=f"bundle: {error}"
    )
  return tensor_target_reference.observe(
    case, symiri=symiri, timeout_s=timeout_s, output_path=output_path
  )


def _words(expected, bundle, path, recorded):
  """Differing words against the retained raw output, if it was kept."""
  try:
    actual = (bundle / "target-runs" / _MODES[path] / "actual.bin").read_bytes()
  except OSError:
    return None
  if _sha(actual) != recorded or len(actual) != len(expected):
    return None
  return [
    {
      "index": index // 4,
      "p0_bits": expected[index : index + 4][::-1].hex(),
      "path_bits": actual[index : index + 4][::-1].hex(),
    }
    for index in range(0, len(expected), 4)
    if expected[index : index + 4] != actual[index : index + 4]
  ][:MAX_WORDS]


def _classify(observation, recorded, restricted):
  if restricted:
    return "restricted"
  if observation.status == tensor_target_reference.UB:
    return "p0_ub"
  if not observation.completed or recorded is None:
    return "unavailable"
  return "agree" if _sha(observation.output) == recorded else "differ"


def compare(shards, *, symiri, scratch, timeout_s=120.0):
  """Compare every retained path output in the shards with P0."""
  scratch = Path(scratch)
  by_probe = collections.defaultdict(collections.Counter)
  bodies = collections.Counter()
  disagreements = []
  kernels = 0
  p0_runs = 0
  for shard in shards:
    shard = Path(shard)
    for line in _kernels(shard):
      kernels += 1
      seed = line["seed"]
      root = shard / "kernels" / f"seed-{seed}"
      operations = _operations(root)
      unspecified = operations is None or bool(UNSPECIFIED & set(operations))
      observations = {}
      verdicts = collections.Counter()
      for pair in line.get("pairs", []):
        name = pair["input"]
        restricted = name in sweep.CORNER_PROBES and unspecified
        bundle = _bundle(root, pair)
        if name not in observations or not observations[name].completed:
          if not restricted:
            p0_runs += 1
          observations[name] = (
            tensor_target_reference.ReferenceObservation(
              tensor_target_reference.NOT_RUN
            )
            if restricted
            else _observe(
              bundle,
              symiri=symiri,
              timeout_s=timeout_s,
              output_path=scratch / shard.name / f"seed-{seed}" / f"{name}.bin",
            )
          )
        observation = observations[name]
        for path in ("p1", "p2"):
          recorded = (pair.get(path) or {}).get("output_sha256")
          verdict = _classify(observation, recorded, restricted)
          by_probe[name][verdict] += 1
          verdicts[verdict] += 1
          if verdict == "differ":
            disagreements.append(
              {
                "shard": str(shard),
                "seed": seed,
                "live_nodes": line.get("live_nodes"),
                "operations": sorted(set(operations or [])),
                "configuration": pair["configuration"],
                "input": name,
                "path": path,
                "output": "return",
                "p0_sha256": _sha(observation.output),
                "path_sha256": recorded,
                "words": _words(observation.output, bundle, path, recorded),
              }
            )
      bodies[
        "differ"
        if verdicts["differ"]
        else "agree"
        if set(verdicts) == {"agree"}
        else "other"
      ] += 1
  return {
    "schema": SCHEMA,
    "shards": [str(Path(shard)) for shard in shards],
    "symiri_sha256": _sha(Path(symiri).read_bytes()),
    "unspecified_operations": sorted(UNSPECIFIED),
    "kernels": kernels,
    "p0_runs": p0_runs,
    "by_probe": {name: dict(counts) for name, counts in sorted(by_probe.items())},
    "bodies": dict(bodies),
    "disagreements": disagreements,
  }


def main(argv=None):
  import argparse
  import tempfile

  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("shards", type=Path, nargs="+")
  parser.add_argument("--symiri", type=Path, default=_ROOT / "symiri")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--timeout", type=float, default=120.0)
  arguments = parser.parse_args(argv)
  with tempfile.TemporaryDirectory() as scratch:
    report = compare(
      arguments.shards,
      symiri=arguments.symiri.resolve(),
      scratch=scratch,
      timeout_s=arguments.timeout,
    )
  arguments.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
  print(f"{'input':<18} " + " ".join(f"{name:>11}" for name in _COLUMNS))
  for name, counts in report["by_probe"].items():
    print(f"{name:<18} " + " ".join(f"{counts.get(key, 0):>11}" for key in _COLUMNS))
  print(f"bodies {report['bodies']}; disagreements {len(report['disagreements'])}")
  return 0


_COLUMNS = ("agree", "differ", "restricted", "p0_ub", "unavailable")


if __name__ == "__main__":
  raise SystemExit(main())
