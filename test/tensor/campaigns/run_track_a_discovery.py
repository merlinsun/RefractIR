"""Bounded discovery batch for Track A: bounded long-axis regions.

The plan asks for a bounded batch of 128 fresh generation proposals per track,
with at most eight predeclared configurations and four inputs per admitted
program, and with the allocation frozen before any proposal is generated. This
driver is that batch for Track A.

It works in five separable steps, each of which leaves a record on disk so a
dropped connection costs one step rather than the batch:

1. `freeze` writes the manifest of eight configurations and their proposal
   allocation, with a digest and a timestamp, and refuses to overwrite a
   different one. Nothing is generated in this step.
2. `propose` draws one fresh seed per proposal, generates the region source
   through the native exporter, and admits it over the configuration's frozen
   input domain with the same exact-certificate probe the recorded family runs
   use. Every proposal is recorded, admitted or refused, with its reason.
3. `build` derives the four recorded input proposals of every admitted program
   (`test.lib.tensor_input_proposals`) and writes one complete case per input,
   exactly as `test/tensor/campaigns/run_input_proposals.py` does: the exact
   certificate's output where the domain admits the rule-chosen input, otherwise
   a declared in-domain native RefractIR replay, with both reference precisions
   replayed and compared.
4. `run` executes the cases in compiled mode on the device through
   `test.lib.run_triton_tests.run_case`, sharded over the visible GPUs, with a
   fresh Triton cache and the named-pass dump enabled.
5. `report` assembles the machine-readable deliverable: the frozen manifest,
   every proposal with its outcome, every execution, the observations, and the
   budget actually spent.

A rejected proposal is never replaced to reach a count. Controls and smoke
tests run through their own steps and are excluded from the discovery
denominator, which is stated in the record.

Storage policy for this batch (declared before the run): every case keeps its
own directory (program, inputs, reference, certificate, emitted kernel, IR, and
pass dump) under the work tree, bounded by the batch size; the assembled JSON
keeps per-conversion detail for every case that shows target behavior and
aggregates for the rest, with the raw dumps left on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TREE = Path(__file__).resolve().parents[3]
if str(TREE) not in sys.path:
  sys.path.insert(0, str(TREE))

from test.lib import (  # noqa: E402
  probe_triton_env,
  tensor_case,
  tensor_exact_f32,
  tensor_input_proposals,
  tensor_region_generation,
)
from test.tensor.campaigns import run_input_proposals as proposals  # noqa: E402

SCHEMA = "refractir.track-a-discovery-batch.v1"
MANIFEST_SCHEMA = "refractir.track-a-discovery-manifest.v1"
MANIFEST_FILE = "frozen-manifest.json"
STEPS = ("freeze", "propose", "build", "run", "report")
DEFAULT_SHARDS = 8
PROPOSALS_PER_CONFIGURATION = 16
PROPOSAL_TOTAL = 128
SEED_BASE = 100000
SEED_STRIDE = 1000
INPUTS_PER_ADMITTED_PROGRAM = 4
LAUNCH = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}
MAX_OPERATORS = 10
DOMAIN = (0, 2)

# The frozen allocation. Four configurations are the established behavior
# strata: the bounded region families that already carry evidence, at the shape
# and bounds the recorded Track A first run and the named-pass study measured
# (`[4, 8]`, bounds 2 for cross-state, orthogonal-reductions, and
# reduction-arithmetic, bounds 3 for mixed-depth, domain `[0, 2]`, four warps).
# The other four are underexplored legal combinations of the same generator:
# an arithmetic mode, a larger trip count, a shape at the profile's 512-element
# cap, and a family at a shape it has not been run at. Each entry states which
# half it belongs to.
CONFIGURATIONS = (
  {
    "id": "E1-mixed-depth-4x8-it3",
    "half": "established",
    "family": "mixed-depth",
    "shape": [4, 8],
    "iterations": 3,
    "operation_mode": "additive",
    "domain": [0, 2],
    "elements": 32,
    "category": "recorded behavior stratum",
    "evidence": (
      "Track A first run and the named-pass study: `mixed-depth` at [4, 8], "
      "bound 3, four warps, exact admission over [0, 2]"
    ),
  },
  {
    "id": "E2-cross-state-4x8-it2",
    "half": "established",
    "family": "cross-state",
    "shape": [4, 8],
    "iterations": 2,
    "operation_mode": "additive",
    "domain": [0, 2],
    "elements": 32,
    "category": "recorded behavior stratum",
    "evidence": (
      "Track A first run and the named-pass study: `cross-state` at [4, 8], "
      "bound 2, four warps, exact admission over [0, 2]"
    ),
  },
  {
    "id": "E3-orthogonal-reductions-4x8-it2",
    "half": "established",
    "family": "orthogonal-reductions",
    "shape": [4, 8],
    "iterations": 2,
    "operation_mode": "additive",
    "domain": [0, 2],
    "elements": 32,
    "category": "recorded behavior stratum",
    "evidence": (
      "Track A first run and the named-pass study: `orthogonal-reductions` at "
      "[4, 8], bound 2, four warps, exact admission over [0, 2]"
    ),
  },
  {
    "id": "E4-reduction-arithmetic-4x8-it2",
    "half": "established",
    "family": "reduction-arithmetic",
    "shape": [4, 8],
    "iterations": 2,
    "operation_mode": "additive",
    "domain": [0, 2],
    "elements": 32,
    "category": "recorded behavior stratum",
    "evidence": (
      "Track A first run and the named-pass study: `reduction-arithmetic` at "
      "[4, 8], bound 2, four warps, exact admission over [0, 2]"
    ),
  },
  {
    "id": "U1-mixed-depth-subtraction-4x8-it2",
    "half": "underexplored",
    "family": "mixed-depth",
    "shape": [4, 8],
    "iterations": 2,
    "operation_mode": "subtraction",
    "domain": [0, 2],
    "elements": 32,
    "category": "arithmetic mode",
    "evidence": (
      "Every recorded Track A run is `additive`; the generator's subtractive "
      "placement is the mode it states keeps the recurrence inside the "
      "nonnegative domain, and it has not been run"
    ),
  },
  {
    "id": "U2-cross-state-4x8-it8",
    "half": "underexplored",
    "family": "cross-state",
    "shape": [4, 8],
    "iterations": 8,
    "operation_mode": "additive",
    "domain": [0, 2],
    "elements": 32,
    "category": "larger trip count",
    "evidence": (
      "Recorded bounds are 2, 3, and 4; 8 is the profile maximum the "
      "configuration admits, and `cross-state` carries no reduction, so the "
      "trip count is the only growth term"
    ),
  },
  {
    "id": "U3-orthogonal-reductions-32x16-it3",
    "half": "underexplored",
    "family": "orthogonal-reductions",
    "shape": [32, 16],
    "iterations": 3,
    "operation_mode": "additive",
    "domain": [0, 2],
    "elements": 512,
    "category": "shape at the profile's element cap",
    "evidence": (
      "The 512-element matrix admitted `orthogonal-reductions` at [32, 16] "
      "bounds 2-4 over [0, 2] at its one recorded seed; this half re-draws it"
    ),
  },
  {
    "id": "U4-mixed-depth-128x4-it2",
    "half": "underexplored",
    "family": "mixed-depth",
    "shape": [128, 4],
    "iterations": 2,
    "operation_mode": "additive",
    "domain": [0, 2],
    "elements": 512,
    "category": "family at a shape it has not been run at",
    "evidence": (
      "The 512-element matrix admitted `mixed-depth` at [128, 4] bound 2 over "
      "[0, 2]; every other recorded Track A run of this family is at [4, 8] or "
      "[32, 16]"
    ),
  },
)


def _canonical(value):
  return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(data: bytes) -> dict:
  return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _write_json(path, value) -> None:
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
    json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
  )


def _read_json(path):
  return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def configuration_records():
  """The eight frozen configurations with their proposal allocation and seeds."""
  records = []
  for index, entry in enumerate(CONFIGURATIONS):
    record = dict(entry)
    record["index"] = index
    record["proposals"] = PROPOSALS_PER_CONFIGURATION
    record["launch"] = dict(LAUNCH)
    record["max_operators"] = MAX_OPERATORS
    record["seeds"] = [
      SEED_BASE + SEED_STRIDE * index + offset
      for offset in range(PROPOSALS_PER_CONFIGURATION)
    ]
    record["seed_derivation"] = (
      f"{SEED_BASE} + {SEED_STRIDE}*configuration_index + proposal_offset"
    )
    records.append(record)
  return records


def build_manifest(*, frozen_unix, frozen_utc):
  """The frozen manifest body, without its own digest."""
  configurations = configuration_records()
  return {
    "schema": MANIFEST_SCHEMA,
    "what": (
      "The predeclared allocation of the Track A bounded discovery batch: eight "
      "configurations, 128 fresh generation proposals, four recorded inputs per "
      "admitted program."
    ),
    "frozen_at_unix": frozen_unix,
    "frozen_at_utc": frozen_utc,
    "frozen_before_generation": True,
    "frozen_rule": (
      "This file is written before any proposal is generated. The batch steps "
      "refuse to run unless the file on disk digests to the value recorded "
      "here, and the file is never rewritten while the batch runs."
    ),
    "proposal_total": PROPOSAL_TOTAL,
    "proposals_per_configuration": PROPOSALS_PER_CONFIGURATION,
    "inputs_per_admitted_program": INPUTS_PER_ADMITTED_PROGRAM,
    "configuration_count": len(configurations),
    "half_allocation": {
      "established": sum(
        1 for entry in configurations if entry["half"] == "established"
      ),
      "underexplored": sum(
        1 for entry in configurations if entry["half"] == "underexplored"
      ),
      "note": (
        "Four configuration slots are established behavior strata and four are "
        "underexplored legal combinations of the same generator. Every proposal "
        "count is predeclared per configuration; a rejected proposal is kept and "
        "reported rather than replaced."
      ),
    },
    "seed_freshness": (
      "Proposal seeds are 100000 or greater and are disjoint from every seed in "
      "the recorded Track A evidence (seeds 0-9). Each proposal is a new seed."
    ),
    "held_fixed": {
      "launch": dict(LAUNCH),
      "max_operators": MAX_OPERATORS,
      "grid": [1, 1, 1],
      "note": (
        "The batch varies the logical program (family, shape, trip count, "
        "arithmetic mode) and holds the launch setting at the recorded Track A "
        "value, so a difference between configurations is a difference of "
        "generated program rather than of launch configuration."
      ),
    },
    "budget_bound": {
      "admissions": (
        "Every proposal is admitted by the exact certificate over the frozen "
        "domain, or refused with its reason. Admission cost is the pure-Python "
        "exact bound check, not an SMT selection."
      ),
      "executions": (
        "At most four compiled-mode device executions per admitted program, the "
        "four rules of `test.lib.tensor_input_proposals`."
      ),
      "time": (
        "No solver selection is used, so the six-minute 512-lane exact solve the "
        "counted builder needs is not on this path. The 512-element "
        "configurations (U3, U4) carry 32 of the 128 proposals; the other 96 "
        "proposals are 32-element shapes. Device runs are sharded over the "
        "visible GPUs, each case with a fresh Triton cache."
      ),
    },
    "storage_policy": {
      "declared_before_the_run": True,
      "raw": (
        "Every case directory keeps program, inputs, reference, certificate, "
        "emitted kernel, IR stages, and pass dump under the work tree; the "
        "archive is bounded by the batch size (at most 512 cases)."
      ),
      "assembled": (
        "The deliverable keeps per-conversion records for cases that show "
        "target behavior and aggregates elsewhere."
      ),
    },
    "configurations": configurations,
  }


def step_freeze(work, args, state):
  """Write the frozen manifest, or verify the one already on disk."""
  path = Path(work) / MANIFEST_FILE
  body = build_manifest(
    frozen_unix=args.frozen_unix, frozen_utc=args.frozen_utc
  )
  digest = _sha256(_canonical(body).encode("utf-8"))
  manifest = {**body, "digest": digest}
  if path.is_file():
    existing = _read_json(path)
    if existing.get("digest") != digest:
      if not args.force:
        raise SystemExit(
          f"{path} was frozen as {existing.get('digest')} and this build states "
          f"{digest}; the manifest is not rewritten while a batch runs"
        )
      _write_json(path, manifest)
      return {"status": "rewritten", "path": str(path), "digest": digest}
    return {"status": "verified", "path": str(path), "digest": digest}
  _write_json(path, manifest)
  return {"status": "written", "path": str(path), "digest": digest}


def frozen_manifest(work):
  """The frozen manifest on disk, checked against its own digest."""
  path = Path(work) / MANIFEST_FILE
  manifest = _read_json(path)
  body = {key: value for key, value in manifest.items() if key != "digest"}
  digest = _sha256(_canonical(body).encode("utf-8"))
  if digest != manifest.get("digest"):
    raise SystemExit(
      f"the frozen manifest at {path} does not match its recorded digest; "
      "refusing to generate proposals against a changed allocation"
    )
  return manifest


def proposal_id(configuration, seed):
  return f"{configuration['id']}--seed{seed}"


def _configuration(configuration):
  return tensor_region_generation.RegionConfiguration(
    seed=configuration["seed"],
    shape=tuple(configuration["shape"]),
    iterations=configuration["iterations"],
    family=configuration["family"],
    operation_mode=configuration["operation_mode"],
    max_operators=configuration["max_operators"],
  )


def step_propose(work, args, state):
  """Draw one fresh proposal per predeclared seed and admit it over the domain."""
  work = Path(work)
  manifest = frozen_manifest(work)
  record_path = work / "proposals.json"
  records = _read_json(record_path) if record_path.is_file() else {}
  drawn = 0
  for configuration in manifest["configurations"]:
    for seed in configuration["seeds"]:
      identifier = proposal_id(configuration, seed)
      if identifier in records and not args.force:
        continue
      entry = {
        "id": identifier,
        "configuration_id": configuration["id"],
        "half": configuration["half"],
        "category": configuration["category"],
        "seed": seed,
        "configuration": {
          key: configuration[key]
          for key in (
            "family",
            "shape",
            "iterations",
            "operation_mode",
            "domain",
            "max_operators",
            "launch",
          )
        },
        "generation": None,
        "admission": None,
      }
      configuration_record = {
        key: configuration[key]
        for key in (
          "family",
          "shape",
          "iterations",
          "operation_mode",
          "domain",
          "max_operators",
        )
      }
      configuration_record["seed"] = seed
      request = _configuration({**configuration, "seed": seed})
      entry["region"] = request.as_record()
      entry["region_identity"] = request.identity
      try:
        source = tensor_region_generation.generate_source(
          request, exporter=args.exporter, timeout_s=args.stage_timeout
        )
      except tensor_region_generation.RegionGenerationError as error:
        entry["generation"] = {"status": "refused", "reason": str(error)}
        entry["admission"] = {
          "status": "not_probed",
          "reason": "the exporter refused the configuration",
        }
        records[identifier] = entry
        drawn += 1
        continue
      entry["generation"] = {
        "status": "generated",
        "source": _digest(source.encode("utf-8")),
        "lines": len(source.splitlines()),
        "operations": tensor_exact_f32.generated_region_operations(source),
        "reduction_axes": list(
          tensor_region_generation.reduction_axes(source)
        ),
        "root_join_patterns": [
          [operation, list(operands)]
          for operation, operands in tensor_region_generation.root_join_patterns(source)
        ],
      }
      domain = tuple(configuration["domain"])
      entry["admission"] = proposals._probe_kernel(source, domain)
      records[identifier] = entry
      drawn += 1
      _write_json(record_path, records)
  _write_json(record_path, records)
  return {"drawn_this_step": drawn, "records": len(records), "path": str(record_path)}


def _template(configuration, source, seed):
  """The counted-manifest fields a derived proposal case inherits.

  `run_input_proposals.build_proposal_case` states one derived case per recorded
  input proposal and takes the profile, entry, launch, and execution envelope
  from the admitted program it derives from. Here that program is the proposal
  itself, so the template is built from the proposal's own emitted source: the
  tensor shapes of its graph, its trip count, its region record, and the frozen
  launch setting. No solver selection and no counterfactual anchor is involved.
  """
  shape, iterations, tensor_types, _nodes, _updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  shapes = [
    list(value)
    for value in sorted(set(tensor_types.values()), key=lambda item: (len(item), item))
  ]
  return {
    "schema": "refractir.tensor-case.v2",
    "profile": "tensor-v2",
    "entry": "kernel",
    "source": {"seed": seed, "sha256": _sha256(source.encode("utf-8"))},
    "grid": [1, 1, 1],
    "softmax_lowering": "direct",
    "launch": dict(configuration["launch"]),
    "execution": {
      "full_shapes": shapes,
      "range_extents": shapes,
      "warp_probe": {
        "status": "not_probed",
        "num_warps": configuration["launch"]["num_warps"],
      },
      "loop_bounds": [iterations],
      "total_body_executions": iterations,
      "gap_sentinel_by_dtype": {"i32": "55555555", "f32": "3f000001"},
      "output_canary_by_dtype": {"i32": "aaaaaaaa", "f32": "3f000002"},
    },
    "coverage": {
      "source_signature": None,
      "loop_structure": [{"bound": iterations}],
    },
    "generator": {
      "region": {**configuration["region"], "shape": list(shape)},
      "node_limit": configuration["max_operators"],
    },
    "shape_axes": [],
    "comparison": {"mode": "numeric", "atol": [0, 1], "rtol": [0, 1]},
  }


def _provenance(configuration, region, source, admission, domain):
  """What one recorded input is derived from, stated inside the case.

  `region` is the proposal's own region request, passed separately from the
  frozen configuration so a caller cannot describe a case with a region the
  proposal did not ask for.
  """
  return {
    "family": configuration["family"],
    "program_sha256": _sha256(source.encode("utf-8")),
    "region": region,
    "rules": "test.lib.tensor_input_proposals",
    "input_domain": {"lo": domain[0], "hi": domain[1]},
    "reference_case": {
      "case": "self",
      "note": (
        "the proposal is its own anchor: its generated source is admitted over "
        "the frozen domain by the exact certificate, and each recorded input is "
        "certified or replayed against that same source"
      ),
      "admission": admission.get("status"),
      "probe": admission.get("probe"),
      "probe_input_sha256": admission.get("probe_input_sha256"),
      "target": admission.get("target"),
      "reference_sha256": admission.get("reference_sha256"),
    },
  }


def _capped_proposal_record(proposal):
  """A proposal's derivation record with its lane lists stated as digests.

  Every value is decided by the recorded rule, the seed, the shape, and the
  domain, and the bytes are kept beside the case with their sha256, so the lane
  lists are redundant in the assembled record. The digest keeps the statement
  checkable without a megabyte of repeated integers.
  """
  record = proposal.record()
  values = record.pop("values")
  lanes = record.pop("nonzero_lanes")
  record["values_sha256"] = _sha256(",".join(str(value) for value in values).encode())
  record["nonzero_lane_count"] = len(lanes)
  record["lane_count"] = len(values)
  return record


def step_build(work, args, state):
  """Derive the four recorded inputs of every admitted proposal and build cases."""
  work = Path(work)
  manifest = frozen_manifest(work)
  # This driver raises the interpreter replay budget rather than the library's
  # own default, and only for its own stages.
  from test.lib import build_tensor_exact_loop_case

  build_tensor_exact_loop_case.TIMEOUT_S = args.stage_timeout
  records = _read_json(work / "proposals.json")
  build_path = work / "build-records.json"
  builds = _read_json(build_path) if build_path.is_file() else {}
  built = 0
  for configuration in manifest["configurations"]:
    domain = tuple(configuration["domain"])
    for seed in configuration["seeds"]:
      identifier = proposal_id(configuration, seed)
      entry = records[identifier]
      if entry["admission"].get("status") != "admitted":
        builds[identifier] = {
          "status": "not_admitted",
          "admission": entry["admission"],
          "inputs": [],
        }
        continue
      request = _configuration({**configuration, "seed": seed})
      source = tensor_region_generation.generate_source(
        request, exporter=args.exporter, timeout_s=args.stage_timeout
      )
      proposals_list = tensor_input_proposals.derive_proposals(
        source, domain=domain, seed=seed
      )
      derived = {
        "status": "built",
        "duplicate_input_groups": tensor_input_proposals.duplicate_groups(
          proposals_list
        ),
        "inputs": [],
      }
      existing = builds.get(identifier)
      if existing and existing.get("status") == "built" and not args.force:
        continue
      template = _template(
        {
          **configuration,
          "region": entry["region"],
          "launch": configuration["launch"],
          "max_operators": configuration["max_operators"],
        },
        source,
        seed,
      )
      provenance = _provenance(
        configuration, entry["region"], source, entry["admission"], domain
      )
      for proposal in proposals_list:
        directory = (
          work / "suite" / identifier / proposal.name
        )
        if directory.is_dir() and not args.force:
          derived["inputs"].append(
            {
              "name": proposal.name,
              "status": "reused",
              "directory": str(directory),
            }
          )
          continue
        if directory.is_dir():
          shutil.rmtree(directory)
        deadline = time.monotonic() + args.stage_timeout
        ledger = []
        try:
          proposals.build_proposal_case(
            directory,
            source=source,
            proposal=proposal,
            domain=domain,
            counted_manifest=template,
            provenance=provenance,
            symiri=args.symiri,
            ledger=ledger,
            deadline=deadline,
          )
        except (ValueError, tensor_case.CaseValidationError) as error:
          derived["inputs"].append(
            {"name": proposal.name, "status": "refused", "reason": str(error)}
          )
          continue
        card = _read_json(directory / "proposal-record.json")
        derived["inputs"].append(
          {
            "name": proposal.name,
            "status": "built",
            "directory": str(directory),
            "admission": card["admission"],
            "reference_kind": card["reference_kind"],
            "expected_sha256": card["expected_sha256"],
            "input_bytes_sha256": card["input_bytes_sha256"],
          }
        )
        with (work / "replay-ledger.jsonl").open("a", encoding="utf-8") as stream:
          for replay in card["replays"]:
            stream.write(
              _canonical(
                {
                  "proposal": identifier,
                  "input": proposal.name,
                  "reference_bits": replay["reference_bits"],
                  "sha256": replay["sha256"],
                  "matches_expected": replay["matches_expected"],
                  "command": replay["command"],
                }
              )
              + "\n"
            )
        built += 1
      builds[identifier] = derived
      _write_json(build_path, builds)
  _write_json(build_path, builds)
  return {"built_this_step": built, "records": len(builds), "path": str(build_path)}


def case_directories(work):
  """Every built case directory of the batch, in a stable order."""
  builds = _read_json(Path(work) / "build-records.json")
  directories = []
  for identifier in sorted(builds):
    for entry in builds[identifier].get("inputs", []):
      if entry.get("status") in {"built", "reused"}:
        directories.append((identifier, entry["name"], entry["directory"]))
  return directories


def merged_run_records(work):
  """The run records of every shard, merged into one mapping."""
  work = Path(work)
  merged = {}
  paths = sorted(work.glob("run-records-*.json"))
  legacy = work / "run-records.json"
  if legacy.is_file():
    paths.append(legacy)
  for path in paths:
    try:
      merged.update(_read_json(path))
    except (OSError, json.JSONDecodeError):
      continue
  return merged


def step_run(work, args, state):
  """Execute every built case in compiled mode, sharded over the GPUs."""
  work = Path(work)
  cases = case_directories(work)
  if args.shard is not None:
    return run_shard(work, args, cases)
  results_path = work / "run-records.json"
  results = merged_run_records(work)
  pending = [
    item
    for item in cases
    if args.force or f"{item[0]}/{item[1]}" not in results
  ]
  shards = max(1, min(args.shards, args.devices))
  (work / "logs").mkdir(parents=True, exist_ok=True)
  started = time.monotonic()
  children = []
  for shard in range(shards):
    command = [
      sys.executable,
      str(Path(__file__).resolve()),
      "--work",
      str(work),
      "--step",
      "run",
      "--shard",
      str(shard),
      "--shards",
      str(shards),
      "--symiri",
      args.symiri,
      "--symirc",
      args.symirc,
      "--run-timeout",
      str(args.run_timeout),
      "--stage-timeout",
      str(args.stage_timeout),
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(shard)
    environment.update(args.environment)
    if args.pass_dump:
      environment["REFRACTIR_TRITON_PASS_DUMP"] = "1"
    log = (work / "logs" / f"shard-{shard}.log").open("w", encoding="utf-8")
    children.append(
      (
        shard,
        subprocess.Popen(
          command, cwd=str(TREE), env=environment, stdout=log, stderr=log, text=True
        ),
        log,
      )
    )
  for shard, child, log in children:
    code = child.wait()
    log.close()
    if code != 0:
      state.setdefault("shard_failures", []).append({"shard": shard, "code": code})
  return {
    "cases": len(cases),
    "pending_at_start": len(pending),
    "seconds": round(time.monotonic() - started, 3),
    "shards": shards,
    "shard_failures": state.get("shard_failures", []),
  }


def run_shard(work, args, cases):
  """One shard: run its share of the cases sequentially on its own device.

  Each shard writes its own result file, so concurrent shards never merge into
  one JSON document from several processes.
  """
  from test.lib import run_triton_tests

  results_path = Path(work) / f"run-records-{args.shard}.json"
  mine = [item for index, item in enumerate(cases) if index % args.shards == args.shard]
  results = _read_json(results_path) if results_path.is_file() else {}
  for identifier, name, directory in mine:
    key = f"{identifier}/{name}"
    if key in results and not args.force:
      continue
    started = time.monotonic()
    record = {"identifier": identifier, "input": name, "directory": directory}
    try:
      report = run_triton_tests.run_case(
        directory,
        "compiled",
        timeout_s=args.run_timeout,
        symiri=args.symiri,
        symirc=args.symirc,
        gpu_probe=lambda: True,
        environment_probe=probe_triton_env.collect_environment,
        device_id=0,
      )
      record["report"] = {
        key_: report.get(key_)
        for key_ in (
          "status",
          "mode",
          "reason",
          "exit_code",
          "comparison",
          "campaign_valid",
          "artifact_digests",
          "artifacts",
        )
      }
    except Exception as error:  # a shard must keep going through one bad case
      record["report"] = {
        "status": "driver_error",
        "reason": f"{type(error).__name__}: {error}",
      }
    record["seconds"] = round(time.monotonic() - started, 3)
    results[key] = record
    _write_json(results_path, results)
  return {"shard": args.shard, "cases": len(mine), "path": str(results_path)}


def step_smoke(work, args, state):
  """Calibrate the derivation path on recorded programs, outside the denominator.

  The controls are the seed-9 programs the recorded Track A first run and the
  512-element matrix already carry evidence for. They are re-derived through the
  same build and run path the batch uses, so a defect in the derivation shows up
  before the batch spends its budget, and the per-case cost they measure bounds
  the batch. They are recorded separately and never counted as proposals.
  """
  work = Path(work)
  manifest = frozen_manifest(work)
  from test.lib import build_tensor_exact_loop_case, run_triton_tests

  build_tensor_exact_loop_case.TIMEOUT_S = args.stage_timeout
  table = {entry["id"]: entry for entry in manifest["configurations"]}
  controls = []
  for identifier in args.controls.split(","):
    configuration = table.get(identifier.strip())
    if configuration is None:
      raise SystemExit(f"unknown control configuration {identifier!r}")
    domain = tuple(configuration["domain"])
    request = _configuration({**configuration, "seed": args.control_seed})
    source = tensor_region_generation.generate_source(
      request, exporter=args.exporter, timeout_s=args.stage_timeout
    )
    admission = proposals._probe_kernel(source, domain)
    record = {
      "control": f"{configuration['id']}--seed{args.control_seed}",
      "note": (
        "recorded program, not a fresh proposal; excluded from the discovery "
        "denominator"
      ),
      "configuration": configuration,
      "region": request.as_record(),
      "source": _digest(source.encode("utf-8")),
      "admission": admission,
      "inputs": [],
    }
    if admission.get("status") != "admitted":
      record["status"] = "refused"
      controls.append(record)
      continue
    template = _template(
      {
        **configuration,
        "region": request.as_record(),
        "launch": configuration["launch"],
        "max_operators": configuration["max_operators"],
      },
      source,
      args.control_seed,
    )
    provenance = _provenance(
      configuration, request.as_record(), source, admission, domain
    )
    root = work / "controls" / record["control"]
    started = time.monotonic()
    for proposal in tensor_input_proposals.derive_proposals(
      source, domain=domain, seed=args.control_seed
    ):
      directory = root / proposal.name
      if directory.is_dir():
        shutil.rmtree(directory)
      deadline = time.monotonic() + args.stage_timeout
      try:
        proposals.build_proposal_case(
          directory,
          source=source,
          proposal=proposal,
          domain=domain,
          counted_manifest=template,
          provenance=provenance,
          symiri=args.symiri,
          ledger=[],
          deadline=deadline,
        )
      except (ValueError, tensor_case.CaseValidationError) as error:
        record["inputs"].append(
          {"name": proposal.name, "status": "refused", "reason": str(error)}
        )
        continue
      card = _read_json(directory / "proposal-record.json")
      entry = {
        "name": proposal.name,
        "status": "built",
        "directory": str(directory),
        "admission": card["admission"],
        "reference_kind": card["reference_kind"],
        "input_bytes_sha256": card["input_bytes_sha256"],
        "expected_sha256": card["expected_sha256"],
      }
      if args.smoke_devices:
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        environment.update(args.environment)
        environment["REFRACTIR_TRITON_PASS_DUMP"] = "1"
        child = subprocess.run(
          [
            sys.executable,
            str(Path(__file__).resolve()),
            "--work",
            str(work),
            "--step",
            "smoke-run",
            "--case",
            str(directory),
            "--symiri",
            args.symiri,
            "--symirc",
            args.symirc,
            "--run-timeout",
            str(args.run_timeout),
          ],
          cwd=str(TREE),
          env=environment,
          capture_output=True,
          text=True,
          check=False,
        )
        entry["run"] = _case_run_summary(directory)
        entry["run"]["exit_code"] = child.returncode
        if child.returncode != 0:
          entry["run"]["stderr"] = child.stderr[-2000:]
      record["inputs"].append(entry)
    record["seconds"] = round(time.monotonic() - started, 3)
    record["status"] = "built"
    controls.append(record)
  state["controls"] = controls
  return {"controls": [entry["control"] for entry in controls]}


def step_smoke_run(work, args, state):
  """Run one control case in compiled mode on device 0."""
  from test.lib import run_triton_tests

  directory = Path(args.case)
  started = time.monotonic()
  report = run_triton_tests.run_case(
    directory,
    "compiled",
    timeout_s=args.run_timeout,
    symiri=args.symiri,
    symirc=args.symirc,
    gpu_probe=lambda: True,
    environment_probe=probe_triton_env.collect_environment,
    device_id=0,
  )
  observations = _summarize_observations(directory)
  observations.pop("passes", None)
  result = {
    "status": report.get("status"),
    "comparison": report.get("comparison"),
    "campaign_valid": report.get("campaign_valid"),
    "exit_code": report.get("exit_code"),
    "seconds": round(time.monotonic() - started, 3),
    "loop_status": observations.get("loop_status"),
    "effective_layout_conversions_by_stage": observations.get(
      "effective_layout_conversions_by_stage"
    ),
    "surviving_layout_conversions": observations.get(
      "surviving_layout_conversions"
    ),
    "backedges_by_stage": observations.get("backedges_by_stage"),
  }
  return result


def _case_run_summary(directory):
  """The device-run reading of one case, taken from the case's own record."""
  # `run_triton_tests.run_case` writes the mode-suffixed result beside the case
  # manifest; the IR and the device bytes stay under `runs/compiled/`.
  result_path = Path(directory) / "result-compiled.json"
  if not result_path.is_file():
    return {"status": "not_run"}
  result = _read_json(result_path)
  observations = _summarize_observations(directory)
  observations.pop("passes", None)
  return {
    "status": result.get("status"),
    "comparison": result.get("comparison"),
    "campaign_valid": result.get("campaign_valid"),
    "exit_code": result.get("exit_code"),
    "loop_status": observations.get("loop_status"),
    "effective_layout_conversions_by_stage": observations.get(
      "effective_layout_conversions_by_stage"
    ),
    "surviving_layout_conversions": observations.get(
      "surviving_layout_conversions"
    ),
    "backedges_by_stage": observations.get("backedges_by_stage"),
    "artifact_digests": result.get("artifact_digests"),
  }


def observation_outcome(entry):
  """The observation reading of one executed case, kept apart from execution.

  Track A's target behavior is a layout decision in or across a loop, so this
  reads the four ways the record allows such a decision to appear: a conversion
  that survives to the final tensor IR, a conversion whose enclosing region
  changes across a pass, a rematerialization attributed to a pass, and a named
  pass that removes layout conversions while a loop construct is present before
  and after. A device backedge is reported beside the outcome because a removal
  inside a TTGIR loop and a removal inside a device loop are different claims.
  A case that executed and shows none of those is `not-observed`; a case whose
  IR could not be read is `unavailable`.
  """
  observations = entry.get("observations") or {}
  if observations.get("status") != "observed":
    return {
      "status": "unavailable",
      "reason": observations.get("reason") or "no archived compiler IR",
    }
  passes = entry.get("passes") or {}
  surviving = int(observations.get("surviving_layout_conversions") or 0)
  effective = observations.get("effective_layout_conversions_by_stage") or {}
  moved = int(passes.get("conversion_movement_total") or 0)
  rematerialized = int(
    (passes.get("rematerialization") or {}).get("confirmed_total") or 0
  )
  in_loop = passes.get("in_loop_removals") or []
  removed_in_loop = sum(int(entry_["conversions_removed"]) for entry_ in in_loop)
  device_loop = bool(observations.get("device_backedge"))
  reasons = []
  if surviving:
    reasons.append(f"{surviving} layout conversion(s) survive to the final TTGIR")
  if moved:
    reasons.append(f"{moved} conversion(s) change enclosing region across a pass")
  if rematerialized:
    reasons.append(f"{rematerialized} rematerialization(s) attributable to a pass")
  if removed_in_loop:
    names = sorted({entry_["pass"] for entry_ in in_loop})
    reasons.append(
      f"{removed_in_loop} layout conversion(s) removed by {', '.join(names)} "
      "while a loop construct is present before and after"
    )
  # A device backedge is reported beside the outcome rather than as a reason:
  # a loop that survives to PTX with no layout decision in it is not the
  # layout behavior this track is looking for.
  if reasons:
    return {
      "status": "target-behavior-observed",
      "reasons": reasons,
      "device_loop": device_loop,
      "removed_inside_a_loop": removed_in_loop,
      "surviving_conversions": surviving,
    }
  if any(effective.values()):
    return {
      "status": "not-observed",
      "reasons": [
        "conversions exist but every one is removed before the final tensor IR, "
        "outside any loop"
      ],
      "device_loop": device_loop,
    }
  return {
    "status": "not-observed",
    "reasons": ["no layout conversion in any stage"],
    "device_loop": device_loop,
  }


def _summarize_observations(case_dir):
  """Loop presence, conversions per stage, and the named passes of one case."""
  observations = proposals.loop_and_conversion_observations(case_dir)
  summary = {"status": observations.get("status")}
  if observations.get("status") != "observed":
    summary["reason"] = observations.get("reason")
    return summary
  stages = observations["stages"]
  summary.update(
    {
      "loop_status": observations["loop_status"],
      "loop_presence": {
        stage: record["loop_presence"] for stage, record in stages.items()
      },
      "loop_constructs_by_stage": observations["loop_constructs_by_stage"],
      "backedges_by_stage": observations["backedges_by_stage"],
      "layout_conversions_by_stage": {
        stage: record["layout_conversion_operations"]
        for stage, record in stages.items()
      },
      "effective_layout_conversions_by_stage": observations[
        "effective_layout_conversions_by_stage"
      ],
      "ptx": {
        stage: {
          "warp_shuffle_count": record.get("warp_shuffle_count"),
          "shared_memory_transfer_count": record.get(
            "shared_memory_transfer_count"
          ),
        }
        for stage, record in stages.items()
        if stage.lower() in {"ptx", "sass"}
      },
      "structural_ir_digests": observations["structural_ir_digests"],
      "full_ir_digests": observations["ir_digests"],
      "surviving_layout_conversions": (
        stages.get("ttgir", {}).get("layout_conversion_operations", 0)
      ),
      "conversions_in_final_ttgir": [
        entry
        for entry in observations["layout_conversions"].get("ttgir", [])
      ],
      "device_backedge": bool(
        observations["backedges_by_stage"].get("ptx")
        or observations["backedges_by_stage"].get("llir")
      ),
      "loop_reaches": {
        "ttgir": bool(
          observations["loop_constructs_by_stage"].get("ttgir")
          or observations["backedges_by_stage"].get("ttgir")
        ),
        "llir": bool(observations["backedges_by_stage"].get("llir")),
        "ptx": bool(observations["backedges_by_stage"].get("ptx")),
      },
    }
  )
  passes = proposals.pass_dump_observations(case_dir)
  if passes.get("status") != "observed" or not passes.get("compilations"):
    summary["passes"] = {
      "status": passes.get("status"),
      "reason": passes.get("reason") or "no compilation in the pass dump",
    }
    return summary
  compilation = passes["compilations"][0]
  transitions = compilation["transitions"]
  interesting = [
    transition
    for transition in transitions
    if (
      transition["loop_construct_delta"]
      or transition["conversion_match"]["added"]
      or transition["conversion_match"]["removed"]
      or transition["conversion_match"]["moved"]
      or (transition["rematerialization"] or {}).get("confirmed_count")
    )
  ]
  # A named pass that removes layout conversions while a loop construct is
  # present before and after is layout work inside a retained loop: the loop is
  # still a structured construct in the compiler's view while the removal
  # happens, whatever LLVM does to it later.
  in_loop_removals = []
  for pass_name, occurrences in compilation["named_pass_occurrences"].items():
    for occurrence in occurrences:
      if (
        occurrence["conversion_match"]["removed"]
        and occurrence["loop_constructs_before"]
        and occurrence["loop_constructs_after"]
      ):
        in_loop_removals.append(
          {
            "pass": pass_name,
            "transition_index": occurrence["transition_index"],
            "loop_constructs_before": occurrence["loop_constructs_before"],
            "loop_constructs_after": occurrence["loop_constructs_after"],
            "conversions_removed": occurrence["conversion_match"]["removed"],
            "effective_removed": occurrence.get(
              "effective_layout_conversion_delta"
            ),
          }
        )
  # The named-pass attribution of this case, stated once: what each pass that
  # changed anything added, preserved, removed, or moved, and what it did to the
  # loop constructs it saw.
  pass_totals = {}
  for transition in interesting:
    totals = pass_totals.setdefault(
      transition["pass"],
      {
        "transitions": 0,
        "conversions_added": 0,
        "conversions_preserved": 0,
        "conversions_removed": 0,
        "conversions_moved": 0,
        "loop_construct_delta": 0,
        "rematerializations": 0,
      },
    )
    match = transition["conversion_match"]
    totals["transitions"] += 1
    totals["conversions_added"] += match["added"]
    totals["conversions_preserved"] += match["preserved"]
    totals["conversions_removed"] += match["removed"]
    totals["conversions_moved"] += match["moved"]
    totals["loop_construct_delta"] += transition["loop_construct_delta"]
    totals["rematerializations"] += (
      transition["rematerialization"] or {}
    ).get("confirmed_count") or 0
  summary["passes"] = {
    "status": "observed",
    "compilation_count": passes["compilation_count"],
    "snapshot_count": compilation["snapshot_count"],
    "terminal_pass": compilation["terminal_pass"],
    "final_loop_construct_count": compilation["final_loop_construct_count"],
    "target_pass_evidence": compilation["target_pass_evidence"],
    "named_pass_occurrences": compilation["named_pass_occurrences"],
    "changing_transitions": interesting,
    "pass_totals": pass_totals,
    "in_loop_removals": in_loop_removals,
    "rematerialization": {
      "confirmed_total": sum(
        (transition["rematerialization"] or {}).get("confirmed_count") or 0
        for transition in transitions
      ),
      "unattributed_total": sum(
        (transition["rematerialization"] or {}).get("unattributed_count") or 0
        for transition in transitions
      ),
      "confirmed_passes": sorted(
        {
          transition["pass"]
          for transition in transitions
          if (transition["rematerialization"] or {}).get("confirmed_count")
        }
      ),
    },
    "conversion_movement_total": sum(
      transition["conversion_movement"] for transition in transitions
    ),
    "pass_dump": passes["pass_dump"],
    "interpretation_limit": passes["interpretation_limit"],
  }
  return summary


def step_report(work, args, state):
  """Assemble the deliverable: manifest, proposals, executions, and budget."""
  work = Path(work)
  manifest = frozen_manifest(work)
  proposal_records = _read_json(work / "proposals.json")
  build_records = _read_json(work / "build-records.json")
  run_records = merged_run_records(work)
  environment = probe_triton_env.collect_environment()
  report = {
    "schema": SCHEMA,
    "what": (
      "The Track A bounded discovery batch: 128 fresh generation proposals "
      "across eight predeclared configurations, four recorded inputs per "
      "admitted program, executed in compiled mode on the device with the loop "
      "and layout-conversion observations of every case."
    ),
    "denominator": {
      "discovery_proposals": PROPOSAL_TOTAL,
      "note": (
        "The discovery denominator is exactly the 128 predeclared generation "
        "proposals. Controls and smoke tests are recorded separately under "
        "`controls` and are excluded from every rate below."
      ),
      "controls_excluded": [
        "the seed-9 family configurations already recorded by the Track A first "
        "run, used to time the batch and to check the derivation path",
      ],
    },
    "frozen_manifest": manifest,
    "environment": {
      "gpus": environment["gpus"],
      "driver_version": environment["driver_version"],
      "cuda_version": environment["cuda_version"],
      "python_version": environment["python_version"],
      "torch_version": environment["torch_version"],
      "triton_version": environment["triton_version"],
      "triton_source_revision": environment["triton_source_revision"],
      "campaign_available": environment["campaign_available"],
      "failure_reasons": environment["failure_reasons"],
      "package_lock_sha256": environment["package_lock_sha256"],
      "import_root": state.get("import_root"),
      "refractir_revision": args.revision,
      "compiler_flags": args.compiler_flags,
      "runner_environment": state.get("run"),
    },
    "imports": {
      "imported_module_sha256": proposals.imported_modules(),
      "test_library_inventory_sha256": proposals.library_inventory(args.import_root),
      "note": (
        "the digest of every `test.lib` module this driver imported and of every "
        "module in the imported test library"
      ),
    },
    "tools": {
      "symiri": args.symiri,
      "symirc": args.symirc,
      "exporter": args.exporter,
      "digests": {
        name: (
          _digest(Path(path).read_bytes()) if Path(path).is_file() else None
        )
        for name, path in (
          ("symiri", args.symiri),
          ("symirc", args.symirc),
          ("exporter", args.exporter),
        )
      },
    },
    "proposals": [],
    "controls": state.get("controls", []),
    "ledger": {
      "driver": str(Path(__file__).resolve()),
      "driver_sha256": _sha256(Path(__file__).read_bytes()),
      "invocation": {
        "work": str(work),
        "steps": args.steps,
        "shards": args.shards,
        "domain_default": list(DOMAIN),
        "exporter": args.exporter,
        "symiri": args.symiri,
        "symirc": args.symirc,
        "pass_dump": args.pass_dump,
      },
      "reference_replay_commands": _replay_commands(work),
      "note": (
        "Every case carries its own replay ledger in `proposal-record.json` and "
        "every command is written once to `replay-ledger.jsonl` under the work "
        "tree; the entries above are the distinct command shapes."
      ),
    },
  }
  # Proposals in frozen order, so the assembled order is the allocation order.
  ordered = []
  for configuration in manifest["configurations"]:
    for seed in configuration["seeds"]:
      ordered.append(proposal_id(configuration, seed))
  for identifier in ordered:
    entry = proposal_records.get(identifier)
    if entry is None:
      report["proposals"].append(
        {
          "id": identifier,
          "outcome": "not_drawn",
          "inputs": [],
          "rejection_reason": "the propose step did not reach this proposal",
        }
      )
      continue
    build = build_records.get(identifier) or {"status": "not_built", "inputs": []}
    record = {
      "id": identifier,
      "configuration_id": entry["configuration_id"],
      "half": entry["half"],
      "category": entry["category"],
      "seed": entry["seed"],
      "region": entry["region"],
      "region_identity": entry["region_identity"],
      "generation": entry["generation"],
      "admission": entry["admission"],
      "inputs": [],
    }
    if entry["admission"].get("status") != "admitted":
      record["outcome"] = "rejected"
      record["rejection_reason"] = (
        entry["admission"].get("reason")
        or entry["generation"].get("reason")
        or "unknown refusal"
      )
      report["proposals"].append(record)
      continue
    admitted_inputs = 0
    for item in build.get("inputs", []):
      case = {
        "name": item["name"],
        "status": item["status"],
        "directory": item.get("directory"),
        "admission": item.get("admission"),
        "reason": item.get("reason"),
        "reference_kind": item.get("reference_kind"),
        "input_bytes_sha256": item.get("input_bytes_sha256"),
        "expected_sha256": item.get("expected_sha256"),
        "execution": {"status": "not_run"},
        "observation": {"status": "unavailable"},
      }
      if item["status"] not in {"built", "reused"}:
        record["inputs"].append(case)
        continue
      admitted_inputs += 1
      directory = Path(item["directory"])
      card = _read_json(directory / "proposal-record.json")
      case["proposal"] = _capped_proposal_record_from_record(card["proposal"])
      case["certificate"] = card["certificate"]
      case["replays"] = card["replays"]
      case["loader"] = card["loader"]
      expected = (directory / "expected.bin").read_bytes()
      case["output"] = tensor_input_proposals.constant_output(expected)
      result = run_records.get(f"{identifier}/{item['name']}")
      if result is not None:
        case["execution"] = {
          "status": result["report"].get("status"),
          "comparison": result["report"].get("comparison"),
          "campaign_valid": result["report"].get("campaign_valid"),
          "exit_code": result["report"].get("exit_code"),
          "reason": result["report"].get("reason"),
          "seconds": result.get("seconds"),
          "artifact_digests": result["report"].get("artifact_digests"),
        }
        case["bit_level_check"] = proposals._bit_level_check(directory, expected)
        observations = _summarize_observations(directory)
        # The pass record is one record, not two: it is lifted out of the
        # observation reading rather than duplicated beside it.
        case["passes"] = observations.pop("passes", None)
        case["observations"] = observations
        case["observation"] = observation_outcome(case)
      record["inputs"].append(case)
    record["outcome"] = (
      "admitted" if admitted_inputs else "admitted_without_a_built_input"
    )
    record["admitted_inputs"] = admitted_inputs
    report["proposals"].append(record)
  report["executions"] = [
    {
      "proposal": proposal["id"],
      "configuration_id": proposal["configuration_id"],
      "input": case["name"],
      "status": case["execution"].get("status"),
      "comparison": (case["execution"].get("comparison") or {}).get("status"),
      "campaign_valid": case["execution"].get("campaign_valid"),
      "word_exact": (case.get("bit_level_check") or {}).get("word_exact"),
      "observation": case["observation"].get("status"),
      "loop_status": (case.get("observations") or {}).get("loop_status"),
      "seconds": case["execution"].get("seconds"),
    }
    for proposal in report["proposals"]
    for case in proposal["inputs"]
    if case["execution"].get("status") != "not_run"
  ]
  report["configurations"] = _configuration_summary(report)
  report["graph_diversity"] = _graph_diversity(report)
  report["budget"] = _budget(report, state, args)
  full_record = Path(args.full_output) if args.full_output else Path(args.output).with_name(
    "track-a-discovery-batch-full.json"
  )
  _write_json(full_record, report)
  _write_json(
    Path(args.output), compact_report(report, full_record=str(full_record))
  )
  return report


def compact_report(report, *, full_record):
  """The deliverable-sized view of the assembled report.

  The declared storage policy keeps per-conversion records for cases that show
  target behavior and aggregates elsewhere, with the raw dumps and the full
  assembled record left on disk. This applies that policy, and it uses a
  measured identity rather than a quota: the four recorded inputs of one
  proposal compile to byte-identical TTIR, TTGIR, and LLIR (measured on all 128
  proposals of this batch), so one input of each proposal carries the full
  per-transition and per-conversion record and its three siblings carry the
  counts, the named-pass totals, the loop readings, and an explicit pointer
  stating what was proven equal. No measurement is dropped: every count, every
  named pass, every loop reading, every admission, and every execution outcome
  of the full record is still present; what goes is duplication.
  """
  compact = {key: value for key, value in report.items() if key != "proposals"}
  compact["compaction"] = {
    "policy": report["frozen_manifest"]["storage_policy"],
    "full_record": full_record,
    "full_record_note": (
      "the full assembled record, including every per-conversion record of every "
      "case, is kept beside the batch on the work tree"
    ),
    "dropped": [
      "the copy of the pass record nested inside the observation record",
      "per-stage raw IR digests (the path-normalized structural digests stay)",
      "per-occurrence conversion-match objects inside `target_pass_evidence` and "
      "the redundant `named_pass_occurrences` list (each pass's totals and the "
      "in-loop removals stay)",
      "interpreter replay command lines (the digest, the precision, and the "
      "agreement of every replay stay; `ledger.reference_replay_commands` "
      "states the command shapes)",
      "the artifact digest map of each execution (the emitted kernel digest and "
      "the actual-bytes digest stay)",
      "the bound `blocks` list of each certificate (its obligation count, its "
      "maxima, and its digest stay)",
      "the per-transition list of the three sibling inputs of each proposal, "
      "which is carried once per proposal and stated equal by the structural IR "
      "digests of the three tensor stages",
    ],
  }
  proposals = []
  for proposal in report["proposals"]:
    entry = {key: value for key, value in proposal.items() if key != "inputs"}
    entry["inputs"] = []
    inputs = proposal["inputs"]
    reference = None
    for case in inputs:
      if (case.get("observations") or {}).get("status") == "observed":
        reference = case
        break
    reference_name = (reference or {}).get("name")
    reference_stages = (
      ((reference or {}).get("observations") or {}).get("structural_ir_digests")
      or {}
    )
    equal_siblings = []
    for case in inputs:
      stages = (case.get("observations") or {}).get("structural_ir_digests") or {}
      if case is not reference and all(
        (stages.get(stage) or {}).get("sha256")
        == (reference_stages.get(stage) or {}).get("sha256")
        for stage in ("ttir", "ttgir", "llir")
      ):
        equal_siblings.append(case["name"])
    entry["input_identity"] = {
      "reference_input": reference_name,
      "siblings_with_identical_tensor_ir": equal_siblings,
      "basis": (
        "path-normalized TTIR, TTGIR, and LLIR digests of the sibling input equal "
        "the reference input's"
      ),
      "note": (
        "PTX is excluded: its debug section encodes the length of the embedded "
        "case path, so it differs between inputs of the same program"
      ),
    }
    for case in inputs:
      item = {
        key: value
        for key, value in case.items()
        if key not in {"certificate", "observations", "passes", "replays"}
      }
      if case.get("replays"):
        item["replays"] = [
          {
            "reference_bits": replay.get("reference_bits"),
            "sha256": replay.get("sha256"),
            "matches_expected": replay.get("matches_expected"),
          }
          for replay in case["replays"]
        ]
      execution = item.get("execution") or {}
      digests = execution.get("artifact_digests") or {}
      if digests:
        item["execution"] = {
          key: value for key, value in execution.items() if key != "artifact_digests"
        }
        item["execution"]["kernel_sha256"] = (
          digests.get("runs/compiled/kernel.py") or {}
        ).get("sha256")
        item["execution"]["actual_sha256"] = (
          digests.get("runs/compiled/actual.bin") or {}
        ).get("sha256")
      certificate = case.get("certificate") or {}
      if certificate:
        summary = dict(certificate.get("proof_summary") or {})
        summary.pop("blocks", None)
        item["certificate"] = {
          key: certificate.get(key)
          for key in (
            "admission",
            "reference_kind",
            "refusal_reason",
            "expected_sha256",
            "certificate_sha256",
            "replay_matches_certificate",
          )
        }
        item["certificate"]["proof_summary"] = summary
      observations = case.get("observations") or {}
      if observations:
        item["observations"] = {
          key: value
          for key, value in observations.items()
          if key not in {"full_ir_digests", "conversions_in_final_ttgir"}
        }
        if observations.get("conversions_in_final_ttgir"):
          item["observations"]["conversions_in_final_ttgir"] = observations[
            "conversions_in_final_ttgir"
          ]
      passes = case.get("passes") or {}
      if passes:
        item["passes"] = {
          key: value
          for key, value in passes.items()
          if key
          not in {
            "changing_transitions",
            "named_pass_occurrences",
            "pass_dump",
          }
        }
        item["passes"]["pass_dump"] = passes.get("pass_dump")
        item["passes"]["target_pass_evidence"] = {
          name: {
            "status": evidence.get("status"),
            "transition_indices": evidence.get("transition_indices"),
            "occurrence_count": len(evidence.get("occurrences") or []),
            "loop_present_before_and_after": sum(
              1
              for occurrence in (evidence.get("occurrences") or [])
              if occurrence.get("loop_before") and occurrence.get("loop_after")
            ),
          }
          for name, evidence in (passes.get("target_pass_evidence") or {}).items()
        }
        if case is reference or case["name"] not in equal_siblings:
          item["passes"]["changing_transitions"] = [
            {
              "pass": transition.get("pass"),
              "loop_construct_delta": transition.get("loop_construct_delta"),
              "layout_conversion_delta": transition.get("layout_conversion_delta"),
              "effective_layout_conversion_delta": transition.get(
                "effective_layout_conversion_delta"
              ),
              "conversion_match": transition.get("conversion_match"),
              "conversion_movement": transition.get("conversion_movement"),
              "rematerialization": transition.get("rematerialization"),
            }
            for transition in (passes.get("changing_transitions") or [])
          ]
        else:
          item["passes"]["changing_transitions_detail"] = {
            "kind": "identical_to_sibling_input",
            "of": reference_name,
            "basis": entry["input_identity"]["basis"],
          }
      entry["inputs"].append(item)
    proposals.append(entry)
  compact["proposals"] = proposals
  return compact


def _replay_commands(work):
  """The distinct interpreter replay command shapes of the batch."""
  path = Path(work) / "replay-ledger.jsonl"
  if not path.is_file():
    return {"entries": 0, "shapes": []}
  entries = 0
  shapes = {}
  for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    entries += 1
    record = json.loads(line)
    shape = [str(part) for part in record["command"]]
    # The case paths are the only per-case part; the rest is the command shape.
    key = json.dumps(
      [
        Path(part).name if part.endswith((".sir", ".bin")) else part
        for part in shape
      ],
      ensure_ascii=False,
    )
    shapes.setdefault(key, {"count": 0, "example": record["command"]})
    shapes[key]["count"] += 1
  return {
    "entries": entries,
    "shapes": [value for _key, value in sorted(shapes.items())],
  }


def _capped_proposal_record_from_record(record):
  values = record.pop("values", None)
  lanes = record.pop("nonzero_lanes", None)
  if values is not None:
    record["values_sha256"] = _sha256(",".join(str(value) for value in values).encode())
    record["lane_count"] = len(values)
  if lanes is not None:
    record["nonzero_lane_count"] = len(lanes)
  return record


def _configuration_summary(report):
  """Per-configuration accounting, including the all-rejected case."""
  summaries = []
  for configuration in report["frozen_manifest"]["configurations"]:
    entries = [
      entry
      for entry in report["proposals"]
      if entry["configuration_id"] == configuration["id"]
    ]
    admitted = [entry for entry in entries if entry["outcome"] == "admitted"]
    rejected = [entry for entry in entries if entry["outcome"] == "rejected"]
    reasons = {}
    for entry in rejected:
      reasons.setdefault(entry["rejection_reason"], []).append(entry["id"])
    cases = [case for entry in admitted for case in entry["inputs"]]
    executed = [
      case for case in cases if case["execution"].get("status") != "not_run"
    ]
    executed_ok = [case for case in executed if case["execution"]["status"] == "pass"]
    summaries.append(
      {
        "id": configuration["id"],
        "half": configuration["half"],
        "category": configuration["category"],
        "family": configuration["family"],
        "shape": configuration["shape"],
        "iterations": configuration["iterations"],
        "operation_mode": configuration["operation_mode"],
        "domain": configuration["domain"],
        "elements": configuration["elements"],
        "launch": configuration["launch"],
        "proposals": len(entries),
        "admitted": len(admitted),
        "rejected": len(rejected),
        "rejection_reasons": {
          reason: len(ids) for reason, ids in sorted(reasons.items())
        },
        "all_proposals_rejected": len(admitted) == 0,
        "built_inputs": len(cases),
        "executions_attempted": len(executed),
        "executions_passed": len(executed_ok),
        "distinct_programs": len(
          {
            ((entry.get("generation") or {}).get("source") or {}).get("sha256")
            for entry in admitted
          }
          - {None}
        ),
        "observation_outcomes": _count(
          case["observation"].get("status") for case in executed
        ),
      }
    )
  return summaries


def _graph_diversity(report):
  """Distinct generated programs per configuration, and cross-configuration ties."""
  digests = {}
  for entry in report["proposals"]:
    generation = entry.get("generation") or {}
    digest = (generation.get("source") or {}).get("sha256")
    if digest:
      digests.setdefault(digest, []).append(entry["id"])
  repeated = {
    digest: ids for digest, ids in digests.items() if len(ids) > 1
  }
  return {
    "distinct_programs": len(digests),
    "proposals_with_a_source": sum(len(ids) for ids in digests.values()),
    "repeated_programs": [
      {"source_sha256": digest, "proposals": ids}
      for digest, ids in sorted(repeated.items())
    ],
    "note": (
      "A fresh seed is not by itself graph diversity: this counts the distinct "
      "generated sources the batch actually drew."
    ),
  }


def _count(values):
  counts = {}
  for value in values:
    counts[value] = counts.get(value, 0) + 1
  return counts


def _budget(report, state, args):
  """The budget actually spent, separated by kind."""
  proposals_drawn = len(report["proposals"])
  admitted = [entry for entry in report["proposals"] if entry["outcome"] == "admitted"]
  rejected = [entry for entry in report["proposals"] if entry["outcome"] == "rejected"]
  reasons = {}
  for entry in rejected:
    reasons.setdefault(entry["rejection_reason"], []).append(entry["id"])
  cases = [case for entry in admitted for case in entry["inputs"]]
  built = [case for case in cases if case["status"] in {"built", "reused"}]
  executed = [case for case in cases if case["execution"].get("status") != "not_run"]
  passes = _count(case["execution"].get("status") for case in executed)
  comparisons = _count(
    (case["execution"].get("comparison") or {}).get("status") for case in executed
  )
  observations = _count(case["observation"].get("status") for case in executed)
  cache_flags = []
  for entry in admitted:
    for case in entry["inputs"]:
      directory = case.get("directory")
      if not directory:
        continue
      run_record = Path(directory) / "runs" / "compiled" / "run.json"
      if run_record.is_file():
        record = _read_json(run_record)
        cache = record.get("compiler_cache") or {}
        cache_flags.append(
          {
            "case": f"{entry['id']}/{case['name']}",
            "cache_id": cache.get("cache_id"),
            "empty_at_start": cache.get("empty_at_start"),
          }
        )
  return {
    "proposals_drawn": proposals_drawn,
    "proposals_admitted": len(admitted),
    "proposals_rejected": len(rejected),
    "rejections_by_reason": {
      reason: {"count": len(ids), "proposals": ids}
      for reason, ids in sorted(reasons.items())
    },
    "configurations_with_every_proposal_rejected": [
      summary["id"]
      for summary in report["configurations"]
      if summary["all_proposals_rejected"]
    ],
    "inputs_derived": len(cases),
    "inputs_built": len(built),
    "input_build_refusals": [
      {"proposal": entry["id"], "input": case["name"], "reason": case.get("reason")}
      for entry in admitted
      for case in entry["inputs"]
      if case["status"] not in {"built", "reused"}
    ],
    "executions_attempted": len(executed),
    "execution_outcomes": passes,
    "comparison_outcomes": comparisons,
    "observation_outcomes": observations,
    "compilations": {
      "cases_with_a_run_record": len(cache_flags),
      "distinct_cache_ids": len({flag["cache_id"] for flag in cache_flags}),
      "caches_empty_at_start": sum(1 for flag in cache_flags if flag["empty_at_start"]),
      "cache_reuse_visible": False,
      "note": (
        "`run_triton_tests.run_case` gives every case its own fresh Triton cache "
        "and records it empty at start, so no compilation in this batch reuses "
        "another case's cache entry; the compilation count equals the number of "
        "executed cases."
      ),
    },
    "seconds": {
      "propose": state.get("seconds", {}).get("propose"),
      "build": state.get("seconds", {}).get("build"),
      "run": state.get("seconds", {}).get("run"),
      "total_measured": sum(
        value
        for value in (
          state.get("seconds", {}).get("propose"),
          state.get("seconds", {}).get("build"),
          state.get("seconds", {}).get("run"),
        )
        if value
      ),
    },
    "bounding": {
      "shards": args.shards,
      "stage_timeout_s": args.stage_timeout,
      "run_timeout_s": args.run_timeout,
      "note": (
        "The batch is bounded by the frozen allocation: 128 proposals, at most "
        "four executions each, at most eight configurations, and no solver "
        "selection anywhere on the path. The bulk of the proposals (96 of 128) "
        "are 32-element shapes; the two 512-element configurations carry the "
        "remaining 32. The interpreter replay budget is raised in this driver "
        "only (`build_tensor_exact_loop_case.TIMEOUT_S`), not in the library."
      ),
    },
  }


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--work", required=True)
  parser.add_argument("--output")
  parser.add_argument(
    "--full-output",
    default=None,
    help="where the uncompacted assembled record is kept",
  )
  parser.add_argument("--steps", default="freeze,propose,build,run,report")
  parser.add_argument("--exporter", default=str(tensor_region_generation.EXPORTER))
  parser.add_argument("--symiri", default=str(tensor_case.DEFAULT_SYMIRI))
  parser.add_argument(
    "--symirc", default=str(os.environ.get("REFRACTIR_SYMIRC", TREE / "symirc"))
  )
  parser.add_argument("--stage-timeout", type=float, default=900.0)
  parser.add_argument("--run-timeout", type=float, default=600.0)
  parser.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
  parser.add_argument("--devices", type=int, default=DEFAULT_SHARDS)
  parser.add_argument("--shard", type=int, default=None)
  parser.add_argument(
    "--controls",
    default="E1-mixed-depth-4x8-it3,U3-orthogonal-reductions-32x16-it3",
  )
  parser.add_argument("--control-seed", type=int, default=9)
  parser.add_argument("--case", default=None)
  parser.add_argument("--smoke-devices", action="store_true", default=True)
  parser.add_argument("--revision", default=os.environ.get("REFRACTIR_REVISION"))
  parser.add_argument(
    "--compiler-flags", default=os.environ.get("REFRACTIR_COMPILER_FLAGS")
  )
  parser.add_argument("--import-root", default=None)
  parser.add_argument("--pass-dump", action="store_true", default=True)
  parser.add_argument("--frozen-unix", type=float, default=None)
  parser.add_argument("--frozen-utc", default=None)
  parser.add_argument("--force", action="store_true")
  parser.add_argument(
    "--environment",
    action="append",
    default=[],
    help="KEY=VALUE provenance the runner records; may repeat",
  )
  args = parser.parse_args(argv)
  args.environment = dict(
    item.split("=", 1) for item in args.environment if "=" in item
  )
  if args.frozen_unix is None:
    args.frozen_unix = time.time()
  if args.frozen_utc is None:
    args.frozen_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(args.frozen_unix))
  if args.import_root is None:
    args.import_root = str(TREE)
  steps = [step.strip() for step in args.steps.split(",") if step.strip()]
  work = Path(args.work)
  work.mkdir(parents=True, exist_ok=True)
  # A shard subprocess owns only its own result file; it must not write the
  # shared driver state while its siblings are running.
  if args.shard is not None:
    state = {}
    for step in steps:
      if step != "run":
        raise SystemExit("a shard subprocess may only run the run step")
      step_run(work, args, state)
    return 0
  state_path = work / "driver-state.json"
  state = _read_json(state_path) if state_path.is_file() else {}
  state.setdefault("seconds", {})
  state["import_root"] = args.import_root
  for step in steps:
    started = time.monotonic()
    if step == "freeze":
      state["freeze"] = step_freeze(work, args, state)
    elif step == "propose":
      state["propose"] = step_propose(work, args, state)
    elif step == "build":
      state["build"] = step_build(work, args, state)
    elif step == "run":
      state["run"] = step_run(work, args, state)
    elif step == "smoke":
      state["smoke"] = step_smoke(work, args, state)
    elif step == "smoke-run":
      state["smoke_run"] = step_smoke_run(work, args, state)
    elif step == "report":
      state["report"] = {"output": args.output}
      step_report(work, args, state)
    else:
      raise SystemExit(f"unknown step {step!r}")
    state["seconds"][step] = state["seconds"].get(step, 0.0) + round(
      time.monotonic() - started, 3
    )
    _write_json(state_path, state)
  _write_json(state_path, state)
  print(json.dumps({"work": str(work), "steps": steps, "status": "done"}))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
