"""Solve a kernel's integers for corner paths and confirm them in P0 (generation plan G3).

`tensor_path_steering` (the C++ driver) lists a kernel's targets: a block path
through its scalar skeleton and the skeleton itself, with `require` lines that
pin the corner. This module solves each skeleton with `symirsolve --path`,
replays the real kernel in P0 with the solved scalars and zero tensors, and
compares the executed blocks with the target's path. A replay that leaves the
path is a defect in the skeleton, never a refusal.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEERING = Path(
  os.environ.get(
    "REFRACTIR_TENSOR_PATH_STEERING", ROOT / "build/bin/tensor_path_steering"
  )
).resolve()
SYMIRI = Path(os.environ.get("REFRACTIR_SYMIRI", ROOT / "symiri")).resolve()
SYMIRSOLVE = Path(os.environ.get("REFRACTIR_SYMIRSOLVE", ROOT / "symirsolve")).resolve()
_FORMATS = {"i32": "i", "f32": "f"}


def _run(command, timeout=60):
  return subprocess.run(
    command, capture_output=True, text=True, check=False, timeout=timeout
  )


def targets(source: str, *, entry: str = "kernel", profile: str = "v5") -> list[dict]:
  """The kernel's targets: `name`, `path` (block labels) and `skeleton` (SIR text)."""
  with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "kernel.sir"
    path.write_text(source, encoding="utf-8")
    completed = _run([str(STEERING), str(path), "--entry", entry, "--profile", profile])
  if completed.returncode != 0:
    raise RuntimeError(completed.stderr or completed.stdout)
  return json.loads(completed.stdout)["targets"]


def solve(target: dict) -> dict[str, int] | None:
  """The scalars the target's skeleton is solved for, by parameter name; None if UNSAT."""
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "skeleton.sir").write_text(target["skeleton"], encoding="utf-8")
    completed = _run(
      [
        str(SYMIRSOLVE),
        "--main",
        "@kernel",
        "--path",
        ",".join(target["path"]),
        "--emit-model",
        str(root / "model.json"),
        "-o",
        str(root / "solved.sir"),
        str(root / "skeleton.sir"),
      ]
    )
    if completed.returncode != 0 or not (root / "model.json").exists():
      if "UNSAT" in completed.stdout + completed.stderr:
        return None
      raise RuntimeError(completed.stdout + completed.stderr)
    model = json.loads((root / "model.json").read_text(encoding="utf-8"))["@kernel"]
  return {name.removeprefix("%?"): int(value) for name, value in model.items()}


def executed_blocks(trace: str, entry: str = "kernel") -> list[str]:
  """The block labels of the entry function, in order, from `symiri --dump-trace`."""
  prefix = f"@{entry} ^"
  return [
    line[len(prefix) - 1 : -1]
    for line in trace.splitlines()
    if line.startswith(prefix) and line.endswith(":")
  ]


def _describe(source_path: Path, entry: str, profile: str) -> dict:
  completed = _run(
    [
      str(SYMIRI),
      "--tensor-profile-version",
      profile,
      "--entry",
      entry,
      "--tensor-describe",
      str(source_path),
    ]
  )
  if completed.returncode != 0:
    raise RuntimeError(completed.stdout + completed.stderr)
  return json.loads(completed.stdout)


def replay(
  source: str, scalars: dict[str, int], *, entry: str = "kernel", profile: str = "v5"
):
  """P0's executed blocks for the scalars and zero tensors, or the failure text."""
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source_path = root / "kernel.sir"
    source_path.write_text(source, encoding="utf-8")
    data = b""
    for parameter in _describe(source_path, entry, profile)["inputs"]:
      code = _FORMATS[parameter["dtype"]]
      if parameter["shape"]:
        count = 1
        for extent in parameter["shape"]:
          count *= extent
        data += struct.pack(f"<{count}{code}", *([0] * count))
      else:
        data += struct.pack(f"<{code}", scalars.get(parameter["name"], 0))
    (root / "in.bin").write_bytes(data)
    completed = _run(
      [
        str(SYMIRI),
        "--tensor-profile-version",
        profile,
        "--dump-trace",
        "--entry",
        entry,
        "--tensor-input",
        str(root / "in.bin"),
        "--tensor-output",
        str(root / "out.bin"),
        str(source_path),
      ]
    )
  if completed.returncode != 0:
    return completed.stdout + completed.stderr
  return executed_blocks(completed.stdout, entry)


def steer(source: str, *, entry: str = "kernel", profile: str = "v5") -> list[dict]:
  """Every target with its status: `reached`, `unsatisfiable`, or `missed`."""
  results = []
  for target in targets(source, entry=entry, profile=profile):
    scalars = solve(target)
    result = {"name": target["name"], "path": target["path"], "scalars": scalars}
    if scalars is None:
      result["status"] = "unsatisfiable"
    else:
      executed = replay(source, scalars, entry=entry, profile=profile)
      result["executed"] = executed
      result["status"] = "reached" if executed == target["path"] else "missed"
    results.append(result)
  return results
