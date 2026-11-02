# Vendored for tests from layout-mutator d03ba58aeaaafc5d90ab31f4de52c52cc71ef72f,
# layout_mutator/seeds/format.py, reformatted by this repository's Ruff settings and
# otherwise unchanged. RefractIR does not depend on layout-mutator at runtime; this
# copy checks that exported seeds load through the harness's own reader.
# layout_mutator/seeds/format.py
"""The on-disk seed and mutant format.

A directory holds kernel.py, launch.json, seed.json, and inputs/. Arguments are
positional in kernel parameter order; each carries a kind so the executor can
rebuild it without inspecting the kernel.

Containers: JSON has no tuple, so a tuple value is written as a list. A constexpr value is
always read back with every list (nested too) turned into a tuple: Triton hashes constexpr
values into its cache key, so a launch with a list-valued constexpr cannot have run and every
list in a constexpr's JSON stands for a tuple. (Launch options get the same rule in the child.)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KERNEL_FILE = "kernel.py"
LAUNCH_FILE = "launch.json"
SEED_FILE = "seed.json"
INPUTS_DIR = "inputs"

TRITON = "triton"
GLUON = "gluon"
# seed.json keys that make a program a translated Gluon program; every program derived from it
# (chain steps, mutants, controls) carries them, so it is typed, mutated and classified as Gluon.
GLUON_PROVENANCE = ("gluon", "translated_from", "triton_digest")


@dataclass
class ScalarArg:
  value: int | float | bool | None
  type: str  # "int", "float", "bool", or "none" (an omitted optional pointer)
  kind: str = "scalar"


def as_tuples(value):
  """`value` with every list, nested ones too, made a tuple (dict values are converted, the
  dicts kept). The inverse of JSON's tuple-to-list."""
  if isinstance(value, (list, tuple)):
    return tuple(as_tuples(v) for v in value)
  if isinstance(value, dict):
    return {k: as_tuples(v) for k, v in value.items()}
  return value


@dataclass
class ConstexprArg:
  value: Any  # lists become tuples (module docstring)
  kind: str = "constexpr"

  def __post_init__(self):
    self.value = as_tuples(self.value)


@dataclass
class TensorArg:
  shape: tuple[int, ...]
  dtype: str  # torch dtype name without the "torch." prefix
  strides: tuple[int, ...]
  fill: dict | None = None
  raw: str | None = None  # file name under inputs/
  kind: str = "tensor"
  # Elements of storage before the first element: the view starts `offset` elements into its
  # buffer (layout and alignment variants). Written to launch.json only when nonzero.
  offset: int = 0

  def __post_init__(self):
    self.shape = tuple(int(x) for x in self.shape)
    self.strides = tuple(int(x) for x in self.strides)
    self.offset = int(self.offset)
    if (self.fill is None) == (self.raw is None):
      raise ValueError("a tensor argument needs exactly one of fill or raw")


Arg = ScalarArg | ConstexprArg | TensorArg
_KINDS = {"scalar": ScalarArg, "constexpr": ConstexprArg, "tensor": TensorArg}


@dataclass
class Launch:
  kernel: str
  grid: tuple[int, ...]
  options: dict
  names: list[str]
  args: list[Arg]

  def __post_init__(self):
    self.grid = tuple(int(x) for x in self.grid)
    if len(self.names) != len(self.args):
      raise ValueError("names and args must have the same length")


@dataclass
class Seed:
  kernel_source: str
  launch: Launch
  provenance: dict = field(default_factory=dict)


def language(provenance: dict) -> str:
  """A program's language from its seed.json: GLUON when `"gluon": true`, else TRITON."""
  return GLUON if provenance.get("gluon") is True else TRITON


def carried_provenance(provenance: dict) -> dict:
  """The GLUON_PROVENANCE keys of `provenance` that are present, for a derived program's seed.json."""
  return {key: provenance[key] for key in GLUON_PROVENANCE if key in provenance}


def _arg_to_json(arg: Arg) -> dict:
  data = dataclasses.asdict(arg)
  if data.get("offset") == 0:
    del data["offset"]
  return {"kind": data.pop("kind"), **data}


def _arg_from_json(data: dict) -> Arg:
  data = dict(data)
  cls = _KINDS[data.pop("kind")]
  return cls(**data)


def launch_to_json(launch: Launch) -> dict:
  return {
    "kernel": launch.kernel,
    "grid": list(launch.grid),
    "options": launch.options,
    "names": launch.names,
    "args": [_arg_to_json(a) for a in launch.args],
  }


def launch_from_json(data: dict) -> Launch:
  return Launch(
    kernel=data["kernel"],
    grid=tuple(data["grid"]),
    options=dict(data["options"]),
    names=list(data["names"]),
    args=[_arg_from_json(a) for a in data["args"]],
  )


def write_seed(directory: Path | str, seed: Seed, raw: dict[str, bytes]) -> Path:
  directory = Path(directory)
  directory.mkdir(parents=True, exist_ok=True)
  (directory / KERNEL_FILE).write_text(seed.kernel_source, encoding="utf-8")
  (directory / LAUNCH_FILE).write_text(
    json.dumps(launch_to_json(seed.launch), indent=2, sort_keys=False) + "\n",
    encoding="utf-8",
  )
  (directory / SEED_FILE).write_text(
    json.dumps(seed.provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  inputs = directory / INPUTS_DIR
  for name, data in raw.items():
    inputs.mkdir(exist_ok=True)
    (inputs / name).write_bytes(data)
  return directory


def read_seed(directory: Path | str) -> Seed:
  directory = Path(directory)
  return Seed(
    kernel_source=(directory / KERNEL_FILE).read_text(encoding="utf-8"),
    launch=launch_from_json(
      json.loads((directory / LAUNCH_FILE).read_text(encoding="utf-8"))
    ),
    provenance=json.loads((directory / SEED_FILE).read_text(encoding="utf-8")),
  )


def launch_signature(launch: Launch) -> str:
  """Types, constexpr values, grid, and options; fills and raw bytes excluded."""
  parts = [launch.kernel, "grid=" + ",".join(map(str, launch.grid))]
  for key in sorted(launch.options):
    parts.append(f"{key}={launch.options[key]!r}")
  for name, arg in zip(launch.names, launch.args):
    if isinstance(arg, TensorArg):
      parts.append(
        f"{name}:tensor<{','.join(map(str, arg.shape))}x{arg.dtype}>"
        f"@{','.join(map(str, arg.strides))}" + (f"+{arg.offset}" if arg.offset else "")
      )
    elif isinstance(arg, ConstexprArg):
      parts.append(f"{name}:constexpr={arg.value!r}")
    else:
      parts.append(f"{name}:{arg.type}")
  return ";".join(parts)


def source_digest(source: str) -> str:
  return hashlib.sha256(source.encode("utf-8")).hexdigest()


def seed_digest(seed: Seed) -> str:
  text = source_digest(seed.kernel_source) + "\n" + launch_signature(seed.launch)
  return hashlib.sha256(text.encode("utf-8")).hexdigest()
