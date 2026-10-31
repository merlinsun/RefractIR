"""High-volume moving-window interpreter-versus-GPU sweep (plan section 6.5).

Every admitted body in a declared seed block runs at every declared launch
configuration on inputs I0 and I3, each (kernel, configuration, input) as one
paired interpreter/GPU run. The oracles need no floating-point tolerance: the
two paths must agree byte for byte, I3 must equal I0 on each path, and every
path must complete with finite output. A violation is quarantined as a
candidate and the sweep continues.

Accounting is one run budget rather than the controlled study's per-slot
ledgers: GPU worker dispatches across all sessions, wall-clock time per
session, the disk guard, and an infrastructure-failure rate. A dispatch line
is written before each paired run, so an interrupted session still charges
what it started. The builder and paired worker are injected so host tests can
script them; `device_operations` binds the exporter and `tensor_target_run`.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import random
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from test.lib import tensor_execution  # noqa: E402
from test.lib.tensor_free_form_inputs import CORNER_PROBES  # noqa: E402
from test.lib.tensor_target_case import LOOP_HINT  # noqa: E402

SCHEMA = "refractir.moving-window-sweep.v1"
INPUTS = ("i0-base", "i3-unused-rows")
GROUPS = ("G0", "G1", "G2", "unconstrained")
ORDER = "policy-stratified-seeded-permutation-v1"
MAX_NODES = 16
MAX_MULTIPLICATIONS = 1
# The largest input magnitude the window probe recipe writes (the late row's
# +-5). The exporter refuses a body whose values could then leave the exact
# f32 integer range, so tolerance-free oracles stay exact.
EXACT_INPUT_BOUND = 5
# Elementwise f32 operations round identically on both paths with fusion off,
# so only `sum`, whose lane order a backend chooses, must stay exact.
EXACT_CONTRACT = "reduction"
# Keep the most grown nodes alive when choosing result roots: larger live
# bodies are where H2b observed layout conversions.
ROOT_SELECTION = "most-live"
# Operations whose result order or precision a backend may choose: `sum`
# accumulates in a lane order the backend picks. Exact under the reduction
# contract on I0-I3, but not on a corner probe, where a path difference in a
# body that contains one is restricted rather than quarantined.
UNSPECIFIED = frozenset({"sum", "dot"})
# Literal bounds are constants Triton folds; runtime bounds are unspecialized
# kernel parameters with the same values (`symirc --tensor-loop-bound`).
LOOP_BOUNDS = ("literal", "runtime")
INFRASTRUCTURE_WINDOW = 100
INFRASTRUCTURE_MAX = 5
COMPLETED = "completed"
# A path that never produced an observation for reasons outside the kernel.
INFRASTRUCTURE_STATUSES = frozenset(
  {"environment_unavailable", "not_run", "infrastructure_error"}
)
MAX_POSITIONS = 16
# A pair the driver chose not to dispatch; it is neither a failure nor evidence.
SKIPPED = "skipped"


class SweepRefused(RuntimeError):
  """The work directory cannot continue under the requested declaration."""


def _part_one():
  """The controlled-study driver, reused read-only for its shared constants."""
  import importlib.util

  path = Path(__file__).with_name("run_moving_window_discovery.py")
  spec = importlib.util.spec_from_file_location("moving_window_discovery", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


_STUDY = _part_one()
COLUMNS = _STUDY.COLUMNS
# One-trip windows, interior and tail. Every study column runs four or more
# trips, so Triton never compiled a loop whose body runs once. A zero-trip
# loop is outside the tensor profile, which requires one to eight trips.
ONE_TRIP_COLUMNS = (
  {
    "column": 4,
    "shape": [2, 8],
    "trips": 1,
    "stride": 1,
    "backing_rows": 5,
    "active": 2,
    "regime": "interior",
  },
  {
    "column": 5,
    "shape": [3, 4],
    "trips": 1,
    "stride": 1,
    "backing_rows": 5,
    "active": 2,
    "regime": "tail",
  },
)
# Square windows of 16 rows, the smallest tile `tl.dot` accepts, whose backing
# stays within the profile's 512 elements: interior, and a tail whose second
# trip reads past the active bound.
DOT_COLUMNS = (
  {
    "column": 0,
    "shape": [16, 16],
    "trips": 2,
    "stride": 8,
    "backing_rows": 32,
    "active": 24,
    "regime": "interior",
  },
  {
    "column": 1,
    "shape": [16, 16],
    "trips": 2,
    "stride": 8,
    "backing_rows": 32,
    "active": 20,
    "regime": "tail",
  },
)
# Two-trip windows for nested loops: two outer trips of two inner trips keep
# within the profile's eight body executions (2 + 2 x 2 = 6).
NESTED_COLUMNS = (
  {
    "column": 0,
    "shape": [2, 8],
    "trips": 2,
    "stride": 1,
    "backing_rows": 5,
    "active": 3,
    "regime": "interior",
  },
  {
    "column": 1,
    "shape": [3, 4],
    "trips": 2,
    "stride": 2,
    "backing_rows": 6,
    "active": 4,
    "regime": "tail",
  },
)
# Exporter features; a declaration that names none keeps the study's set.
DEFAULT_FEATURES = ("extrema", "transpose", "expand")
KNOWN_FEATURES = frozenset(DEFAULT_FEATURES + ("dot", "narrow", "where"))
POLICIES = _STUDY.POLICIES
PROBES = _STUDY.PROBES
# Inputs a declaration may name: the window probes, whose values stay within
# the exact-input bound, and the corner-value probes, which do not.
# Inputs steered toward a floating-point property per body (`tensor_steering`).
# A body yields a steered input only when P0 confirms the property, so a
# declared steered input is skipped for the bodies that yield none.
STEERED_INPUTS = ("steer-signed-zero", "steer-cancel", "steer-tie")
KNOWN_INPUTS = PROBES + CORNER_PROBES + STEERED_INPUTS
# Inputs whose values the exact-input contract does not bound.
UNBOUNDED_INPUTS = frozenset(CORNER_PROBES + STEERED_INPUTS)
# Steering per body: Z3 seconds, lanes tried per property, linearization bases.
STEERING = {"budget_s": 60.0, "lanes": 4, "bases": ["i0-base", "i5-denormal"]}
MIN_FREE_BYTES = _STUDY.MIN_FREE_BYTES


def _canonical(value):
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value):
  return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _positive(value, name, kind=int):
  if type(value) is not kind or value <= 0:
    raise SweepRefused(f"{name} must be a positive {kind.__name__}")


def sweep_spec(
  *,
  seed_start,
  seed_count,
  configurations,
  sample_every,
  gpu_workers,
  wall_clock_s,
  columns=COLUMNS,
  inputs=INPUTS,
  loop_bound="literal",
  trip_parameter=False,
  features=None,
  required_operation=None,
  branch_parameter=False,
  outer_trips=None,
):
  """The declaration frozen before the first builder call.

  `features` and `required_operation` are recorded only when given, so a
  declaration without them keeps its digest.
  """
  _positive(seed_start, "seed_start")
  _positive(seed_count, "seed_count")
  _positive(sample_every, "sample_every")
  _positive(gpu_workers, "gpu_workers")
  _positive(float(wall_clock_s), "wall_clock_s", float)
  configurations = [dict(configuration) for configuration in configurations]
  if not configurations:
    raise SweepRefused("at least one launch configuration is required")
  inputs = list(inputs)
  if (
    not inputs
    or len(set(inputs)) != len(inputs)
    or not set(inputs) <= set(KNOWN_INPUTS)
  ):
    raise SweepRefused(f"inputs are distinct names from {list(KNOWN_INPUTS)}")
  columns = [dict(column) for column in columns]
  if [column["column"] for column in columns] != list(range(len(columns))):
    raise SweepRefused("columns are numbered in order from zero")
  for configuration in configurations:
    if set(configuration) - {"loop_hint"} != {"num_warps", "num_stages"}:
      raise SweepRefused(
        "a configuration states num_warps, num_stages, and at most a loop_hint"
      )
    _positive(configuration["num_warps"], "num_warps")
    _positive(configuration["num_stages"], "num_stages")
    hint = configuration.get("loop_hint")
    if hint is not None and (type(hint) is not str or not LOOP_HINT.fullmatch(hint)):
      raise SweepRefused(f"unknown loop hint {hint!r}")
  if loop_bound not in LOOP_BOUNDS:
    raise SweepRefused(f"loop_bound must be one of {list(LOOP_BOUNDS)}")
  if features is not None:
    features = sorted(set(features))
    if not features or not set(features) <= KNOWN_FEATURES:
      raise SweepRefused(f"features are names from {sorted(KNOWN_FEATURES)}")
  if required_operation is not None and type(required_operation) is not str:
    raise SweepRefused("required_operation names one operation")
  if outer_trips is not None and (type(outer_trips) is not int or outer_trips < 2):
    raise SweepRefused("outer_trips must be an integer of at least two")
  steering = (
    {} if not set(inputs) & set(STEERED_INPUTS) else {"steering": dict(STEERING)}
  )
  return {
    "schema": SCHEMA,
    "seed_start": seed_start,
    "seed_count": seed_count,
    "configurations": configurations,
    "inputs": inputs,
    "columns": columns,
    "max_nodes": MAX_NODES,
    "max_multiplications": MAX_MULTIPLICATIONS,
    "exact_input_bound": EXACT_INPUT_BOUND,
    "exact_contract": EXACT_CONTRACT,
    "root_selection": ROOT_SELECTION,
    "loop_bound": loop_bound,
    **steering,
    **({"trip_parameter": True} if trip_parameter else {}),
    **({"features": features} if features is not None else {}),
    **({"required_operation": required_operation} if required_operation else {}),
    **({"branch_parameter": True} if branch_parameter else {}),
    **({"outer_trips": outer_trips} if outer_trips is not None else {}),
    "enable_fp_fusion": False,
    "order": ORDER,
    "sample_every": sample_every,
    "budget": {
      "gpu_workers": gpu_workers,
      "wall_clock_s_per_session": float(wall_clock_s),
      "min_free_bytes": MIN_FREE_BYTES,
      "infrastructure_window": INFRASTRUCTURE_WINDOW,
      "infrastructure_max": INFRASTRUCTURE_MAX,
    },
  }


@functools.lru_cache(maxsize=8)
def _block_cells(seed_start, block, column_count):
  """One block's (group, column, policy) cells, each exactly once.

  Policies take turns in a seeded order, so every prefix holds each policy
  within one of the others; each policy visits all group/column pairs in its
  own seeded order. A sweep stopped by its budget therefore keeps a balanced
  policy mix, and roughly balanced groups and columns.
  """
  rng = random.Random(f"{SCHEMA}:{ORDER}:{seed_start}:{block}")
  pairs = [
    (group, column) for group in range(len(GROUPS)) for column in range(column_count)
  ]
  policies = rng.sample(range(len(POLICIES)), len(POLICIES))
  visits = {policy: rng.sample(pairs, len(pairs)) for policy in policies}
  return tuple(
    (*visits[policy][turn], policy) for turn in range(len(pairs)) for policy in policies
  )


def _trips(spec, index, column):
  """The case's trip count: seeded, uniform over zero to the column's trips."""
  rng = random.Random(f"{SCHEMA}:trips:{spec['seed_start']}:{index}")
  return rng.randint(0, column["trips"])


def _branch(spec, index):
  """The case's branch selector: seeded, 0 or 1 with equal odds."""
  rng = random.Random(f"{SCHEMA}:branch:{spec['seed_start']}:{index}")
  return rng.randint(0, 1)


def kernel_request(spec, index):
  """Map a seed-block index to its group, shape column, and policy."""
  columns = spec["columns"]
  block, offset = divmod(index, len(GROUPS) * len(columns) * len(POLICIES))
  group_index, column_index, policy_index = _block_cells(
    spec["seed_start"], block, len(columns)
  )[offset]
  column = columns[column_index]
  group = GROUPS[group_index]
  requirement = {
    "G0": "reduction",
    "G1": "shared",
    "G2": "extrema-transpose" if column["column"] % 2 == 0 else "extrema-expand",
    "unconstrained": None,
  }[group]
  return {
    "seed": spec["seed_start"] + index,
    "index": index,
    "group": group,
    "requirement": requirement,
    "policy": POLICIES[policy_index],
    "padded": (group_index + column["column"]) % 2 == 1,
    "column": dict(column),
    "max_nodes": spec["max_nodes"],
    "max_multiplications": spec["max_multiplications"],
    "exact_input_bound": spec["exact_input_bound"],
    "exact_contract": spec["exact_contract"],
    "root_selection": spec["root_selection"],
    "loop_bound": spec["loop_bound"],
    "features": list(spec.get("features", DEFAULT_FEATURES)),
    **(
      {
        "steer": [
          name[len("steer-") :] for name in spec["inputs"] if name in STEERED_INPUTS
        ],
        "steering": spec["steering"],
      }
      if "steering" in spec
      else {}
    ),
    **({"trips": _trips(spec, index, column)} if spec.get("trip_parameter") else {}),
    **({"branch": _branch(spec, index)} if spec.get("branch_parameter") else {}),
    **({"outer_trips": spec["outer_trips"]} if "outer_trips" in spec else {}),
  }


class RunLog:
  """Append-only JSON lines: the budget, tool identities, one line per kernel."""

  def __init__(self, path):
    self.path = Path(path)

  def read(self):
    if not self.path.is_file():
      return []
    return [json.loads(line) for line in self.path.read_text().splitlines() if line]

  def append(self, event, **fields):
    line = {"event": event, "time": time.time(), **fields}
    with self.path.open("a", encoding="utf-8") as handle:
      handle.write(_canonical(line) + "\n")
    return line


def _freeze(work, spec, toolchain):
  """Write the declaration once; a restart must present the same one."""
  frozen = {"spec": spec, "spec_sha256": _digest(spec), "toolchain": toolchain}
  path = work / "sweep.json"
  if not path.is_file():
    tensor_execution.write_atomic(path, (_canonical(frozen) + "\n").encode("utf-8"))
    return frozen
  existing = json.loads(path.read_text(encoding="utf-8"))
  if existing["spec_sha256"] != frozen["spec_sha256"]:
    raise SweepRefused("the declaration differs from the frozen sweep")
  if existing["toolchain"] != toolchain:
    raise SweepRefused("the tool identities differ from the frozen sweep")
  return existing


def _finite(output):
  if len(output) % 4 != 0:
    return False
  return all(math.isfinite(value) for (value,) in struct.iter_unpack("<f", output))


def _positions(first, second):
  """Word positions where two outputs differ, capped for the record."""
  if len(first) != len(second):
    return ["length"]
  found = [
    index // 4
    for index in range(0, len(first), 4)
    if first[index : index + 4] != second[index : index + 4]
  ]
  return found[:MAX_POSITIONS]


def _normalized(reason):
  """The last line of an error, without addresses, numbers, or paths."""
  text = [line for line in str(reason or "").strip().splitlines() if line.strip()]
  text = text[-1] if text else ""
  text = re.sub(r"\S*/\S+", "<path>", text)
  return re.sub(r"0x[0-9a-fA-F]+|\d+", "N", text)


def signature(candidate):
  """Group candidates by failing stage and error, or by differing positions."""
  keys = ("kind", "path", "status", "positions")
  fields = {key: candidate.get(key) for key in keys}
  fields["reason"] = _normalized(candidate.get("reason"))
  return _digest(fields)[:16]


def _sha(output):
  return None if output is None else hashlib.sha256(output).hexdigest()


def _enclosure_check(pair, enclosure):
  """Check both paths' lanes against the enclosure; record and return violations.

  The pair line keeps the lane counts and each path's violation count.
  """
  from test.lib import tensor_enclosure

  candidates, summary = [], {"violations": {}}
  where = {"configuration": pair["configuration"], "input": pair["input"]}
  for path in ("p1", "p2"):
    report = tensor_enclosure.check_lanes(enclosure, pair[path]["output"])
    summary.update(
      point_lanes=report["point_lanes"],
      widened_lanes=report["widened_lanes"],
      widest_ulps=report["widest_ulps"],
    )
    summary["violations"][path] = len(report["violations"])
    for kind in ("point_mismatch", "outside"):
      positions = [v["index"] for v in report["violations"] if v["class"] == kind]
      if positions:
        candidates.append(
          {
            "kind": f"enclosure_{kind}",
            "path": path,
            "positions": positions[:MAX_POSITIONS],
            **where,
          }
        )
  pair["enclosure"] = summary
  return candidates


def evaluate(pairs, *, operations=None, enclosures=None):
  """Apply the three oracles to one kernel's paired runs.

  A corner-value input is not bounded, so its values may leave the finite
  range SIR defines. Its non-finite runs are skipped rather than candidates;
  only a disagreement over whether the output is finite remains one. In a
  body with an `UNSPECIFIED` operation, its finite outputs are checked lane by
  lane against the input's enclosure when the builder computed one: a lane
  outside it, or a point lane with other bits, is a candidate, while two
  paths that differ inside it are not. Without an enclosure they are not
  compared.
  """
  restricted = operations is None or bool(UNSPECIFIED & set(operations))
  candidates = []
  infrastructure = False
  for pair in pairs:
    where = {"configuration": pair["configuration"], "input": pair["input"]}
    corner = pair["input"] in UNBOUNDED_INPUTS
    for path in ("p1", "p2"):
      run = pair[path]
      if run["status"] == SKIPPED:
        continue
      if run["status"] in INFRASTRUCTURE_STATUSES:
        infrastructure = True
      elif run["status"] != COMPLETED:
        candidates.append(
          {
            "kind": "execution_status",
            "path": path,
            "status": run["status"],
            "reason": run.get("reason", ""),
            **where,
          }
        )
      elif not _finite(run["output"]) and not corner:
        candidates.append({"kind": "non_finite", "path": path, **where})
    first, second = pair["p1"].get("output"), pair["p2"].get("output")
    both_completed = pair["p1"]["status"] == pair["p2"]["status"] == COMPLETED
    if both_completed and corner and not (_finite(first) or _finite(second)):
      continue
    if both_completed and corner and restricted and _finite(first) and _finite(second):
      enclosure = (enclosures or {}).get(pair["input"])
      if enclosure is not None and enclosure != "unavailable":
        if len(first) == len(second) == 4 * len(list(enclosure.lo.flat)):
          candidates.extend(_enclosure_check(pair, enclosure))
          continue
        # A wrong output length is the path oracle's to report.
        pair["enclosure"] = "unavailable"
      else:
        continue
    if both_completed and first != second:
      # Non-finite values are undefined in SIR, so a difference among them is
      # grouped apart from a difference between defined outputs.
      kind = (
        "path_difference"
        if _finite(first) and _finite(second)
        else "non_finite_path_difference"
      )
      candidates.append({"kind": kind, "positions": _positions(first, second), **where})
  for configuration in {_canonical(pair["configuration"]) for pair in pairs}:
    runs = {
      pair["input"]: pair
      for pair in pairs
      if _canonical(pair["configuration"]) == configuration
    }
    base, unused = runs.get(INPUTS[0]), runs.get(INPUTS[1])
    if base is None or unused is None:
      continue
    for path in ("p1", "p2"):
      if base[path]["status"] == unused[path]["status"] == COMPLETED and (
        base[path]["output"] != unused[path]["output"]
      ):
        candidates.append(
          {
            "kind": "unused_row_difference",
            "path": path,
            "configuration": json.loads(configuration),
            "positions": _positions(base[path]["output"], unused[path]["output"]),
          }
        )
  return candidates, infrastructure


def _pair_line(pair):
  """The every-kernel record of one paired run: statuses and digests."""
  return {
    "configuration": pair["configuration"],
    "input": pair["input"],
    "kernel_sha256": pair.get("kernel_sha256"),
    "inputs_sha256": pair.get("inputs_sha256"),
    "compiler": pair.get("compiler"),
    **({"enclosure": pair["enclosure"]} if "enclosure" in pair else {}),
    **{
      path: {
        "status": pair[path]["status"],
        "reason": pair[path].get("reason", ""),
        "output_sha256": _sha(pair[path].get("output")),
        # Recorded because a non-finite corner run is skipped, not quarantined.
        "finite": (
          _finite(pair[path]["output"]) if pair[path]["status"] == COMPLETED else None
        ),
      }
      for path in ("p1", "p2")
    },
  }


def _prune(pairs):
  """Drop raw outputs and worker logs; SIR, kernel, and inputs stay."""
  for pair in pairs:
    for raw in pair.get("raw") or ():
      path = Path(raw)
      if path.is_dir():
        shutil.rmtree(path)
      elif path.is_file():
        path.unlink()


class _Session:
  """One invocation over the frozen seed block."""

  def __init__(self, work, spec, operations, clock, free_bytes):
    self.work = work
    self.spec = spec
    self.operations = operations
    self.clock = clock
    self.free_bytes = free_bytes
    self.log = RunLog(work / "run-log.jsonl")
    history = self.log.read()
    kernels = [line for line in history if line["event"] == "kernel"]
    self.done = {line["seed"] for line in kernels}
    self.dispatched = sum(line["event"] == "dispatch" for line in history)
    self.admitted = sum(line["status"] != "refused" for line in kernels)
    self.infrastructure = [
      line["infrastructure"] for line in kernels if line["status"] != "refused"
    ]
    self.restarted = bool(history)
    self.pairs_per_kernel = len(spec["configurations"]) * len(spec["inputs"])

  def stop_reason(self, started, now):
    budget = self.spec["budget"]
    if now - started >= budget["wall_clock_s_per_session"]:
      return "wall_clock"
    if self.dispatched + self.pairs_per_kernel > budget["gpu_workers"]:
      return "gpu_budget"
    if self.free_bytes(self.work) < budget["min_free_bytes"]:
      return "disk_guard"
    # Checked before each kernel, so a restart after this stop dispatches nothing.
    if self.infrastructure_exceeded():
      return "infrastructure_failure_rate"
    return None

  def kernel_directory(self, seed):
    """A fresh directory; a partial one from an interrupted session is kept."""
    directory = self.work / "kernels" / f"seed-{seed}"
    if directory.exists():
      attempt = 1
      while directory.with_name(f"{directory.name}.interrupted-{attempt}").exists():
        attempt += 1
      kept = directory.with_name(f"{directory.name}.interrupted-{attempt}")
      directory.rename(kept)
      self.log.append("interrupted", seed=seed, kept=str(kept))
    directory.mkdir(parents=True)
    return directory

  def run_kernel(self, index):
    """Build and run one kernel; return its status fields and directory."""
    request = kernel_request(self.spec, index)
    directory = self.kernel_directory(request["seed"])
    built = self.operations["build"]({**request, "directory": str(directory)})
    required = self.spec.get("required_operation")
    # An operation with formats (`narrow f16`) is required by its first word.
    if (
      built["kind"] == "admitted"
      and required
      and required not in {op.split()[0] for op in built["operations"]}
    ):
      built = {"kind": "refused", "reason": f"the body has no {required}"}
    if built["kind"] != "admitted":
      return directory, dict(
        seed=request["seed"],
        request=request,
        status="refused",
        reason=built.get("reason", ""),
      )
    self.admitted += 1
    sampled = self.admitted % self.spec["sample_every"] == 0
    kernel = {**built, "seed": request["seed"], "request": request}
    pairs = []
    for configuration in self.spec["configurations"]:
      skip = None
      for name in self.spec["inputs"]:
        if name in STEERED_INPUTS and name not in built.get("steered", ()):
          pairs.append(
            _skipped_pair(configuration, name, "the body yielded no steered input")
          )
          continue
        if skip is not None:
          pairs.append(_skipped_pair(configuration, name, skip))
          continue
        self.log.append(
          "dispatch", seed=request["seed"], configuration=configuration, input=name
        )
        self.dispatched += 1
        pair_directory = directory / "pairs" / _pair_name(configuration, name)
        result = self.operations["run_pair"](
          {
            "kernel": kernel,
            "configuration": configuration,
            "input": name,
            "directory": str(pair_directory),
          }
        )
        pairs.append({**result, "configuration": configuration, "input": name})
        if _compilation_timeout(result):
          # Every input at this setting compiles the same kernel, so the rest
          # would only repeat the timeout.
          skip = f"P2 compilation timed out on {name} at this configuration"
    candidates, infrastructure = evaluate(
      pairs, operations=built.get("operations"), enclosures=built.get("enclosures")
    )
    exposure = self.operations["screen"](kernel) if sampled else None
    if not (candidates or sampled):
      _prune(pairs)
    for candidate in candidates:
      self._quarantine(request["seed"], directory, candidate)
    self.infrastructure.append(infrastructure)
    status = (
      "candidate" if candidates else "infrastructure" if infrastructure else "agreement"
    )
    return directory, dict(
      seed=request["seed"],
      request=request,
      status=status,
      infrastructure=infrastructure,
      sampled=sampled,
      typed_fingerprint=built.get("typed_fingerprint"),
      connectivity_signature=built.get("connectivity_signature"),
      live_nodes=built.get("live_nodes"),
      operations=built.get("operations"),
      source_sha256=built.get("source_sha256"),
      pairs=[_pair_line(pair) for pair in pairs],
      candidates=[signature(candidate) for candidate in candidates],
      exposure_sample=exposure,
    )

  def _quarantine(self, seed, directory, candidate):
    """Record the candidate beside the payloads it needs; nothing is pruned."""
    if "input" in candidate:
      directory = (
        directory / "pairs" / _pair_name(candidate["configuration"], candidate["input"])
      )
    line = {
      **candidate,
      "seed": seed,
      "directory": str(directory),
      "signature": signature(candidate),
    }
    with (self.work / "candidates.jsonl").open("a", encoding="utf-8") as handle:
      handle.write(_canonical(line) + "\n")

  def infrastructure_exceeded(self):
    return sum(self.infrastructure[-INFRASTRUCTURE_WINDOW:]) > INFRASTRUCTURE_MAX

  def run(self):
    self.log.append(
      "restart" if self.restarted else "start",
      spec_sha256=_digest(self.spec),
      budget=self.spec["budget"],
      toolchain=self.operations["toolchain"],
      device=self.operations.get("device"),
      timeout_s=self.operations.get("timeout_s"),
      gpu_workers_dispatched=self.dispatched,
    )
    started = self.clock()
    reason = None
    for index in range(self.spec["seed_count"]):
      if self.spec["seed_start"] + index in self.done:
        continue
      now = self.clock()
      reason = self.stop_reason(started, now)
      if reason is not None:
        break
      directory, fields = self.run_kernel(index)
      self.log.append(
        "kernel",
        **fields,
        elapsed_s=self.clock() - now,
        disk_bytes=_disk_bytes(directory),
      )
    reason = reason or "seed_block_complete"
    self.log.append("stop", reason=reason, gpu_workers_dispatched=self.dispatched)
    return report(self.work, reason)


def _compilation_timeout(result):
  compiled = result.get("p2") or {}
  return (
    compiled.get("status") == "timed_out" and compiled.get("phase") == "compilation"
  )


def _skipped_pair(configuration, name, reason):
  path = {"status": SKIPPED, "reason": reason, "output": None}
  return {
    "configuration": configuration,
    "input": name,
    "p1": dict(path),
    "p2": dict(path),
  }


def _disk_bytes(directory):
  return sum(
    path.stat().st_size for path in Path(directory).rglob("*") if path.is_file()
  )


def _pair_name(configuration, name):
  hint = configuration.get("loop_hint")
  hint = "" if hint is None else hint.replace("=", "-") + "-"
  return f"warps-{configuration['num_warps']}-stages-{configuration['num_stages']}-{hint}{name}"


def _histogram(values):
  counts = {}
  for value in values:
    counts[str(value)] = counts.get(str(value), 0) + 1
  return counts


def _exposure(admitted):
  """Section 6.6 statistics over every paired run's compiler summary."""
  compilers = [
    pair.get("compiler") or {} for line in admitted for pair in line["pairs"]
  ]
  observed = [
    compiler for compiler in compilers if compiler.get("status") == "observed"
  ]
  ttgir = [(compiler.get("stages") or {}).get("ttgir") or {} for compiler in observed]
  kinds = {}
  for compiler in observed:
    for kind in sorted(set(compiler.get("encodings") or ())):
      kinds[kind] = kinds.get(kind, 0) + 1
  return {
    "observed": len(observed),
    "unavailable": len(compilers) - len(observed),
    "ttgir_layout_conversions": _histogram(
      stage.get("layout_conversion_count") for stage in ttgir
    ),
    "ttgir_loop_retained": sum(
      (stage.get("loop_construct_count") or 0) > 0 for stage in ttgir
    ),
    "encoding_kinds": kinds,
  }


def measurements(kernels):
  """The pilot's acceptance measurements from the kernel status lines."""
  admitted = [line for line in kernels if line["status"] != "refused"]
  elapsed = sum(line.get("elapsed_s", 0) for line in kernels)
  disk = [line.get("disk_bytes", 0) for line in admitted]
  failures = sum(line["infrastructure"] for line in admitted)
  return {
    "elapsed_s": elapsed,
    "kernels_per_hour": len(admitted) * 3600 / elapsed if elapsed else None,
    "disk_bytes_per_kernel": {
      "mean": sum(disk) / len(disk) if disk else None,
      "max": max(disk, default=None),
      "total": sum(disk),
    },
    "infrastructure": {
      "kernels": failures,
      "admitted": len(admitted),
      "rate": failures / len(admitted) if admitted else None,
    },
    "live_nodes": _histogram(line.get("live_nodes") for line in admitted),
    "exposure": _exposure(admitted),
  }


def report(work, reason=None):
  """Counts, budget use, and candidate signatures from the retained records."""
  work = Path(work)
  history = RunLog(work / "run-log.jsonl").read()
  kernels = [line for line in history if line["event"] == "kernel"]
  counts = {
    status: sum(line["status"] == status for line in kernels)
    for status in ("refused", "agreement", "candidate", "infrastructure")
  }
  signatures = {}
  candidates = work / "candidates.jsonl"
  if candidates.is_file():
    for text in candidates.read_text(encoding="utf-8").splitlines():
      candidate = json.loads(text)
      entry = signatures.setdefault(
        candidate["signature"],
        {"kind": candidate["kind"], "count": 0, "first_seed": candidate["seed"]},
      )
      entry["count"] += 1
  result = {
    "schema": f"{SCHEMA}.report",
    "stop_reason": reason,
    "kernels": len(kernels),
    "counts": counts,
    "gpu_workers_dispatched": sum(line["event"] == "dispatch" for line in history),
    "sessions": sum(line["event"] in ("start", "restart") for line in history),
    "signatures": signatures,
    "measurements": measurements(kernels),
  }
  tensor_execution.write_atomic(
    work / "report.json", (_canonical(result) + "\n").encode("utf-8")
  )
  return result


def _free_bytes(path):
  return shutil.disk_usage(path).free


def run_sweep(work, spec, operations, *, clock=time.monotonic, free_bytes=_free_bytes):
  """Run, or continue, the declared seed block until a stop condition."""
  work = Path(work)
  work.mkdir(parents=True, exist_ok=True)
  _freeze(work, spec, operations["toolchain"])
  return _Session(work, spec, operations, clock, free_bytes).run()


def _export_command(exporter, request, out):
  """The controlled study's exporter command, with an optional requirement."""
  column = request["column"]
  rows, columns = column["shape"]
  command = [
    str(exporter),
    "--policy",
    request["policy"],
    "--seed",
    str(request["seed"]),
    "--shape",
    f"{rows},{columns}",
    "--states",
    "2",
    "--trip-count",
    str(column["trips"]),
    "--window",
    f"{rows},{column['stride']},{column['active']}",
    "--window-buffer-rows",
    str(column["backing_rows"]),
    "--profile",
    "v3",
    "--features",
    ",".join(request["features"]),
  ]
  if request["requirement"] is not None:
    command += ["--window-requirement", request["requirement"]]
  if "trips" in request:
    command += ["--trip-parameter"]
  if "branch" in request:
    command += ["--branch-parameter"]
  if "outer_trips" in request:
    command += ["--outer-trips", str(request["outer_trips"])]
  return command + [
    "--max-attempts",
    "1",
    "--max-nodes",
    str(request["max_nodes"]),
    "--max-multiplications",
    str(request["max_multiplications"]),
    "--exact-input-bound",
    str(request["exact_input_bound"]),
    "--exact-contract",
    request["exact_contract"],
    "--root-selection",
    request["root_selection"],
    "--require-computed-roots",
    "--num-warps",
    "1",
    "--out",
    str(out),
  ]


def _mode_result(run):
  return {
    "status": run.status,
    "reason": run.reason or run.stderr[-2000:],
    "output": run.actual,
    "phase": run.phase,
  }


def ttgir_encodings(text):
  """The kind of each named layout encoding defined at the top of TTGIR."""
  return [
    match.group(1)
    for match in re.finditer(r"^#\w+ = #ttg\.(\w+)<", text, flags=re.MULTILINE)
  ]


def ttgir_loop_bounds(text):
  """`scf.for` loops by upper bound: an `arith.constant`, or anything else.

  A dynamic bound is what a runtime loop-bound kernel should leave in TTGIR;
  a constant one means Triton folded the trip count.
  """
  constants = set(
    re.findall(r"^\s*(%[\w.]+) = arith\.constant\b", text, flags=re.MULTILINE)
  )
  bounds = re.findall(r"\bscf\.for %[\w.]+ = %[\w.]+ to (%[\w.]+) step\b", text)
  dynamic = sum(bound not in constants for bound in bounds)
  return {"constant": len(bounds) - dynamic, "dynamic": dynamic}


def pair_result(run, bundle):
  """A `tensor_target_run.TargetRun` as the driver's per-path observation."""
  artifacts = Path(bundle) / "target-runs"
  return {
    "kernel_sha256": run.emission.kernel_sha256 if run.emission is not None else None,
    "p1": _mode_result(run.interpreter),
    "p2": _mode_result(run.compiled),
    "raw": [str(artifacts / "interpret"), str(artifacts / "compiled")],
  }


def pair_failure(error):
  """A paired run that raised before producing an observation."""
  path = {
    "status": "infrastructure_error",
    "reason": f"{type(error).__name__}: {error}",
    "output": None,
  }
  return {"kernel_sha256": None, "p1": dict(path), "p2": dict(path), "raw": []}


def _toolchain(exporter, symirc, symiri):
  """Frozen identities, as recorded by the controlled study."""
  from test.lib import native_triton_family, probe_triton_env

  identity = native_triton_family.triton_identity(
    probe_triton_env.collect_environment()
  )
  try:
    import triton

    library = Path(triton.__file__).parent / "_C" / "libtriton.so"
    library_sha256 = _STUDY._file_sha256(library)
  except Exception:
    library_sha256 = None
  return {
    "exporter_sha256": _STUDY._file_sha256(exporter),
    "symirc_sha256": _STUDY._file_sha256(symirc),
    "symiri_sha256": _STUDY._file_sha256(symiri),
    "triton": identity,
    "triton_library_sha256": library_sha256,
    "python": sys.version.split()[0],
    "z3": z3_identity(),
  }


def screen_variants(kernel):
  """The interpreter-only exposure runs for a sampled kernel.

  The one-trip-shorter run edits the literal trip count, so a kernel whose
  loop runs to a trip parameter has none.
  """
  variants = {
    "i0-base": ("i0-base", None),
    "i1-read-plus-one": ("i1-read-plus-one", None),
    "i2-late-row": ("i2-late-row", None),
  }
  if not kernel["region"].get("trip_parameter"):
    trips = kernel["region"]["trip_count"]
    variants["i0-base-tminus"] = (
      "i0-base",
      _STUDY._trip_variant(kernel["source"], trips - 1),
    )
  return variants


def _enclosures(source, region, probe_values):
  """Each unbounded probe's output enclosure, or `unavailable`.

  Kept with the kernel for `evaluate`; it is never written to the run log.
  """
  from test.lib import tensor_enclosure, tensor_free_form_inputs

  parameters = tensor_free_form_inputs.parameters(region)
  enclosures = {}
  for name, values in probe_values.items():
    if name not in UNBOUNDED_INPUTS:
      continue
    try:
      enclosures[name] = tensor_enclosure.enclose_values(
        source, parameters, bytes(values)
      )
    except (tensor_enclosure.NonFinite, tensor_enclosure.Unsupported):
      enclosures[name] = "unavailable"
  return enclosures


def z3_identity():
  """The `z3` binary steering runs, or None when it is not on the path."""
  path = shutil.which("z3")
  if path is None:
    return None
  version = subprocess.run([path, "--version"], capture_output=True, text=True).stdout
  return {"path": path, "sha256": _STUDY._file_sha256(path), "version": version.strip()}


def _steered(request, region, source, probes, symiri, directory):
  """Steer the declared properties; the accepted inputs and a summary."""
  from test.lib import (
    tensor_free_form_case,
    tensor_free_form_inputs,
    tensor_steering,
    tensor_target_reference,
  )

  parameters = tensor_free_form_inputs.parameters(region)
  shapes = [(record["name"], record["shape"]) for record in parameters]
  runs = []

  def p0(values):
    runs.append(None)
    proposal = tensor_free_form_inputs.Proposal(
      schema=tensor_free_form_inputs.WINDOW_PROBE_SCHEMA,
      pattern="steered",
      seed=0,
      names=tuple(record["name"] for record in parameters),
      shapes=tuple(tuple(record["shape"]) for record in parameters),
      dtype="f32",
      values=(),
      data=tensor_steering.pack(shapes, values),
    )
    case = tensor_free_form_case.build_bundle(
      directory / f"p0-{len(runs)}",
      region={**region, "num_warps": 1},
      source=source,
      proposal=proposal,
    )
    observation = tensor_target_reference.observe(case, symiri=symiri)
    return observation.output if observation.completed else None

  steering = request["steering"]
  bases = {
    name: tensor_steering.unpack(shapes, bytes(probes[name]))
    for name in steering["bases"]
  }
  search = tensor_steering.steer_body(
    source,
    shapes,
    bases,
    p0=p0,
    scratch=directory / "smt",
    properties=tuple(request["steer"]),
    lanes=steering["lanes"],
    budget_s=steering["budget_s"],
  )
  values = {
    f"steer-{name}": list(tensor_steering.pack(shapes, result.values))
    for name, result in search.accepted.items()
  }
  summary = {
    "z3_seconds": round(search.z3_seconds, 3),
    "trials": len(search.trials),
    "accepted": sorted(search.accepted),
  }
  return values, summary


def device_operations(
  *, exporter, symirc, symiri, timeout_s=300.0, toolchain=None, device_id=0
):
  """Bind the sweep to the exporter, the interpreter, and the paired runner.

  `device_id` selects the GPU for compiled runs; several shards on one host
  each take their own device and seed block.
  """
  from test.lib import (
    tensor_free_form_case,
    tensor_free_form_inputs,
    tensor_target_run,
  )

  exporter, symirc, symiri = (
    Path(path).resolve() for path in (exporter, symirc, symiri)
  )
  if toolchain is None:
    toolchain = _toolchain(exporter, symirc, symiri)
  identity = toolchain.get("triton")

  def build(request):
    directory = Path(request["directory"])
    column = request["column"]
    command = _export_command(exporter, request, directory / "export")
    try:
      completed = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=timeout_s
      )
    except subprocess.TimeoutExpired:
      return {"kind": "refused", "reason": f"the exporter exceeded {timeout_s} s"}
    if completed.returncode != 0:
      return {
        "kind": "refused",
        "reason": (completed.stderr or completed.stdout).strip()[-2000:],
      }
    region = json.loads(
      (directory / "export" / "region.json").read_text(encoding="utf-8")
    )
    if region.get("max_nodes") != request["max_nodes"]:
      raise SweepRefused("the exporter node limit differs from the declaration")
    nodes = (region.get("structure") or {}).get("nodes")
    if type(nodes) is not int or not 0 <= nodes <= request["max_nodes"]:
      raise SweepRefused("the exported live node count exceeds the declared limit")
    source = (directory / "export" / "program.sir").read_text(encoding="utf-8")
    # The unconstrained group reuses the G0 probe recipe for its column.
    group = GROUPS.index(request["group"]) % 3
    try:
      path = {"trips": request.get("trips"), "branch": request.get("branch")}
      probes = tensor_free_form_inputs.window_probes(
        region, group=group, column=column["column"], **path
      ) + tensor_free_form_inputs.corner_probes(
        region, group=group, column=column["column"], **path
      )
    except tensor_free_form_inputs.SpecError as error:
      return {"kind": "refused", "reason": f"window probes: {error}"}
    layouts = {}
    if request["padded"]:
      layouts["%window_buffer"] = tensor_free_form_case.strided_output(
        [column["backing_rows"], column["shape"][1]], row_padding=3, element_offset=1
      )
    live = region.get("live_dag") or {}
    probe_values = {probe.probe: list(probe.data) for probe in probes}
    steered = {}
    if request.get("steer"):
      steered, summary = _steered(
        request, region, source, probe_values, symiri, directory / "steering"
      )
      probe_values.update(steered)
    enclosures = (
      _enclosures(source, region, probe_values)
      if UNSPECIFIED & set(live.get("ops") or [])
      else {}
    )
    return {
      "kind": "admitted",
      "directory": str(directory),
      "region": region,
      "source": source,
      "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
      "probe_values": probe_values,
      "steered": sorted(steered),
      "enclosures": enclosures,
      **({"steering": summary} if request.get("steer") else {}),
      "layouts": layouts,
      "typed_fingerprint": live.get("typed_fingerprint"),
      "connectivity_signature": live.get("connectivity_signature"),
      "live_nodes": nodes,
      "operations": list(live.get("ops") or []),
    }

  def bundle(
    kernel, directory, name, *, num_warps, num_stages, loop_hint=None, source=None
  ):
    region = kernel["region"]
    parameters = tensor_free_form_inputs.parameters(region)
    data = bytes(kernel["probe_values"][name])
    proposal = tensor_free_form_inputs.Proposal(
      schema=tensor_free_form_inputs.WINDOW_PROBE_SCHEMA,
      pattern=name,
      seed=0,
      names=tuple(record["name"] for record in parameters),
      shapes=tuple(tuple(record["shape"]) for record in parameters),
      dtype="f32",
      values=(),
      data=data,
    )
    tensor_free_form_case.build_bundle(
      directory,
      region={**region, "num_warps": num_warps},
      source=source if source is not None else kernel["source"],
      proposal=proposal,
      input_layouts=kernel["layouts"] or None,
      output_row_padding=1,
      output_offset=1,
      num_stages=num_stages,
      loop_bound=kernel["request"].get("loop_bound", "literal"),
      loop_hint=loop_hint,
    )
    return data

  def run_pair(request):
    configuration = request["configuration"]
    directory = Path(request["directory"]) / "bundle"
    try:
      data = bundle(request["kernel"], directory, request["input"], **configuration)
      run = tensor_target_run.run(
        directory,
        symirc=symirc,
        symiri=symiri,
        timeout_s=timeout_s,
        compiler_identity=identity,
        device_id=device_id,
      )
    except Exception as error:
      # Recorded as infrastructure: it counts against the failure-rate stop
      # rather than disappearing, and the bundle stays for diagnosis.
      return pair_failure(error)
    result = pair_result(run, directory)
    result["inputs_sha256"] = hashlib.sha256(data).hexdigest()
    compiled = directory / "target-runs" / "compiled"
    result["compiler"] = _STUDY.compiler_stage_observation(compiled)
    ttgir = compiled / "compiler-ttgir.ir"
    if ttgir.is_file():
      text = ttgir.read_text(encoding="utf-8")
      result["compiler"]["encodings"] = ttgir_encodings(text)
      result["compiler"]["loop_bounds"] = ttgir_loop_bounds(text)
    return result

  def screen(kernel):
    """Interpreter-only exposure statistics for a sampled kernel."""
    root = Path(kernel["directory"]) / "exposure"
    variants = screen_variants(kernel)
    outputs = {}
    for key, (name, source) in variants.items():
      directory = root / key
      try:
        bundle(kernel, directory, name, num_warps=1, num_stages=1, source=source)
        run = tensor_target_run.run_interpreter(
          directory,
          symirc=symirc,
          symiri=symiri,
          timeout_s=timeout_s,
          compiler_identity=identity,
        )
        outputs[key] = (run.status, run.interpreter.actual)
      except Exception as error:
        outputs[key] = (f"{type(error).__name__}: {error}", None)
    base_status, base = outputs["i0-base"]

    def compare(key):
      if key not in outputs:
        return "not_applicable"
      status, actual = outputs[key]
      if base is None or actual is None:
        return {"status": "unavailable", "base": base_status, "variant": status}
      return "unchanged" if actual == base else "changed"

    return {
      "i1-read-plus-one": compare("i1-read-plus-one"),
      "i2-late-row": compare("i2-late-row"),
      "trip_count_minus_one": compare("i0-base-tminus"),
    }

  return {
    "build": build,
    "run_pair": run_pair,
    "screen": screen,
    "toolchain": toolchain,
    "device": device_id,
    "timeout_s": timeout_s,
  }


def main(argv=None):
  import argparse
  import itertools

  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--work", type=Path, required=True)
  parser.add_argument("--seed-start", type=int, required=True)
  parser.add_argument("--seed-count", type=int, required=True)
  parser.add_argument("--warps", type=int, nargs="+", default=[1, 2, 4, 8])
  parser.add_argument("--stages", type=int, nargs="+", default=[1])
  parser.add_argument("--sample-every", type=int, default=20)
  parser.add_argument("--gpu-workers", type=int, required=True)
  parser.add_argument("--wall-clock-hours", type=float, required=True)
  parser.add_argument("--exporter", type=Path, required=True)
  parser.add_argument("--symirc", type=Path, default=_ROOT / "symirc")
  parser.add_argument("--symiri", type=Path, default=_ROOT / "symiri")
  parser.add_argument("--timeout", type=float, default=300.0)
  parser.add_argument("--device", type=int, default=0, help="GPU for compiled runs")
  parser.add_argument(
    "--one-trip-columns", action="store_true", help="add the two one-trip columns"
  )
  parser.add_argument(
    "--dot-columns",
    action="store_true",
    help="use the two 16 x 16 dot columns instead of the study's columns",
  )
  parser.add_argument(
    "--nested-columns",
    action="store_true",
    help="use the two two-trip columns that leave room for outer trips",
  )
  parser.add_argument("--features", nargs="+", choices=sorted(KNOWN_FEATURES))
  parser.add_argument("--required-operation", help="refuse bodies without it")
  parser.add_argument(
    "--branch-parameter",
    action="store_true",
    help="choose each case's branch arm from its seed",
  )
  parser.add_argument(
    "--outer-trips", type=int, help="nest the counted loop this often"
  )
  parser.add_argument("--inputs", nargs="+", default=list(INPUTS), choices=KNOWN_INPUTS)
  parser.add_argument("--loop-bound", choices=LOOP_BOUNDS, default="literal")
  parser.add_argument(
    "--trip-parameter",
    action="store_true",
    help="run each loop to a trip count chosen per case, zero included",
  )
  parser.add_argument(
    "--loop-hints",
    nargs="+",
    default=["none"],
    help="tl.range hints crossed with warps and stages; none adds no hint",
  )
  arguments = parser.parse_args(argv)
  declared = sweep_spec(
    seed_start=arguments.seed_start,
    seed_count=arguments.seed_count,
    configurations=[
      {"num_warps": warps, "num_stages": stages}
      | ({} if hint == "none" else {"loop_hint": hint})
      for warps, stages, hint in itertools.product(
        arguments.warps, arguments.stages, arguments.loop_hints
      )
    ],
    sample_every=arguments.sample_every,
    gpu_workers=arguments.gpu_workers,
    wall_clock_s=arguments.wall_clock_hours * 3600.0,
    columns=DOT_COLUMNS
    if arguments.dot_columns
    else NESTED_COLUMNS
    if arguments.nested_columns
    else COLUMNS + ONE_TRIP_COLUMNS
    if arguments.one_trip_columns
    else COLUMNS,
    inputs=arguments.inputs,
    loop_bound=arguments.loop_bound,
    trip_parameter=arguments.trip_parameter,
    features=arguments.features,
    required_operation=arguments.required_operation,
    branch_parameter=arguments.branch_parameter,
    outer_trips=arguments.outer_trips,
  )
  operations = device_operations(
    exporter=arguments.exporter,
    symirc=arguments.symirc,
    symiri=arguments.symiri,
    timeout_s=arguments.timeout,
    device_id=arguments.device,
  )
  try:
    result = run_sweep(arguments.work, declared, operations)
  except SweepRefused as error:
    print(f"refused: {error}", file=sys.stderr)
    return 2
  print(json.dumps(result, indent=1, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
