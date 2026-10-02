"""Run the authored exact loop from solver selection through GPU execution."""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

from test.lib import build_tensor_exact_loop_case, run_triton_tests, tensor_case

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SYMIRI = ROOT / "build/bin/symiri"
DEFAULT_SYMIRC = ROOT / "build/bin/symirc"
DEFAULT_SYMIRSOLVE = ROOT / "build/bin/symirsolve"


def run_integration(
  output_dir: str | Path,
  *,
  symiri: str | Path = DEFAULT_SYMIRI,
  symirc: str | Path = DEFAULT_SYMIRC,
  symirsolve: str | Path = DEFAULT_SYMIRSOLVE,
  device_id: int = 0,
):
  """Build a case from an actual SMT witness, validate it, and execute Triton."""
  case_dir = build_tensor_exact_loop_case.build_case(
    output_dir, symiri=symiri, symirc=symirc, solver=symirsolve
  )
  case = tensor_case.load_case(case_dir, symiri=symiri)
  model = json.loads((case_dir / "selection-model.json").read_text(encoding="utf-8"))
  selected = model["@kernel"]["%?x"]
  if not isinstance(selected, list) or len(selected) != 4:
    raise ValueError("solver model does not contain four selected input leaves")
  selected_bits = struct.pack("<4f", *selected)
  if selected_bits != case.inputs:
    raise ValueError("runtime input bytes differ from the exported solver witness")

  report = run_triton_tests.run_case(
    case_dir,
    "compiled",
    symiri=symiri,
    symirc=symirc,
    device_id=device_id,
  )
  if report.get("status") != "pass" or report.get("campaign_valid") is not True:
    raise RuntimeError(
      "compiled exact-loop integration failed: "
      + json.dumps(report, ensure_ascii=False, sort_keys=True)
    )
  return case_dir, report


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument(
    "--symiri", default=os.environ.get("REFRACTIR_SYMIRI", DEFAULT_SYMIRI)
  )
  parser.add_argument(
    "--symirc", default=os.environ.get("REFRACTIR_SYMIRC", DEFAULT_SYMIRC)
  )
  parser.add_argument(
    "--symirsolve",
    default=os.environ.get("REFRACTIR_SYMIR_SOLVE", DEFAULT_SYMIRSOLVE),
  )
  parser.add_argument("--device", type=int, default=0)
  args = parser.parse_args(argv)
  case_dir, report = run_integration(
    args.output_dir,
    symiri=args.symiri,
    symirc=args.symirc,
    symirsolve=args.symirsolve,
    device_id=args.device,
  )
  print(
    json.dumps(
      {
        "case": str(case_dir),
        "status": report["status"],
        "comparison": report["comparison"],
        "actual_sha256": report["actual_sha256"],
        "result_file": str(case_dir / "result-compiled.json"),
      },
      ensure_ascii=False,
      sort_keys=True,
    )
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
