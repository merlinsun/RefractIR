"""Enclosure widths over retained moving-window sweep bodies (plan W2, rule D2).

For each admitted kernel and input in the shards, the retained bundle's body is
evaluated by `tensor_enclosure` on its recorded inputs. A body's width is the
widest output lane over its inputs, in f32 steps. D2 drops the magnitude bound
and the `sum` restriction if the median body width is at most 4 ulp and the
widest body at most 64 ulp. Bodies that leave the finite range, or use an
operation the evaluator does not model, are counted apart, not in the widths.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from test.lib import tensor_enclosure  # noqa: E402

_spec = importlib.util.spec_from_file_location(
  "compare_p0", Path(__file__).with_name("compare_p0.py")
)
_compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_compare)
sweep = _compare.sweep

SCHEMA = "refractir.moving-window-enclosure.v1"
D2_MEDIAN_ULPS = 4
D2_MAX_ULPS = 64


def _summary(widths):
  return {
    "bodies": len(widths),
    "median": statistics.median(widths),
    "max": max(widths),
  }


def enclose_shards(shards):
  """Every retained body's enclosure width, per input, and the D2 decision."""
  bodies = []
  by_input = collections.defaultdict(list)
  counts = collections.Counter()
  for shard in shards:
    shard = Path(shard)
    for line in _compare._kernels(shard):
      root = shard / "kernels" / f"seed-{line['seed']}"
      widths = {}
      for pair in line.get("pairs", []):
        if _compare._skipped(pair):
          continue
        name = pair["input"]
        if name in widths:
          continue
        try:
          widths[name] = max(
            tensor_enclosure.enclose_bundle(_compare._bundle(root, pair)).widths()
          )
        except tensor_enclosure.NonFinite:
          counts["non_finite"] += 1
        except tensor_enclosure.Unsupported:
          counts["unsupported"] += 1
          widths = None
          break
        except OSError:
          counts["missing"] += 1
      if not widths:
        continue
      for name, width in widths.items():
        by_input[name].append(width)
      bodies.append(
        {
          "shard": str(shard),
          "seed": line["seed"],
          "live_nodes": line.get("live_nodes"),
          "max_width": max(widths.values()),
          "widths_by_input": widths,
        }
      )
  body_widths = [body["max_width"] for body in bodies]
  median = statistics.median(body_widths) if body_widths else None
  widest = max(body_widths) if body_widths else None
  return {
    "schema": SCHEMA,
    "shards": [str(Path(shard)) for shard in shards],
    "bodies": bodies,
    "by_input": {name: _summary(widths) for name, widths in sorted(by_input.items())},
    "non_finite": counts["non_finite"],
    "unsupported": counts["unsupported"],
    "missing": counts["missing"],
    "d2": {
      "rule": f"median <= {D2_MEDIAN_ULPS} ulp and max <= {D2_MAX_ULPS} ulp",
      "median_body_width": median,
      "max_body_width": widest,
      "passes": bool(body_widths)
      and median <= D2_MEDIAN_ULPS
      and widest <= D2_MAX_ULPS,
    },
  }


def main(argv=None):
  import argparse

  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("shards", type=Path, nargs="+")
  parser.add_argument("--out", type=Path, required=True)
  arguments = parser.parse_args(argv)
  report = enclose_shards(arguments.shards)
  arguments.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
  for name, summary in report["by_input"].items():
    print(
      f"{name:<18} bodies {summary['bodies']:>5}  median {summary['median']:>8}  max {summary['max']:>10}"
    )
  d2 = report["d2"]
  print(
    f"D2 ({d2['rule']}): median {d2['median_body_width']}, max {d2['max_body_width']}, "
    f"{'passes' if d2['passes'] else 'fails'}; non-finite {report['non_finite']}, "
    f"unsupported {report['unsupported']}, missing {report['missing']}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
