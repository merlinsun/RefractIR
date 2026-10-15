"""Build, run, and observe the recorded input proposals for Track A regions.

The representative is the counted `mixed-depth` region at `[32, 16]` with four
trips, seed 9, additive arithmetic, and input domain `[0, 8]`. This driver:

1. regenerates the region source through the native exporter and checks it is the
   counted program the evidence records;
2. derives the four recorded proposals from that case
   (`test.lib.tensor_input_proposals`), each with its own complete reference --
   the exact f32 certificate where the domain admits the proposal, otherwise a
   native RefractIR replay -- and writes one derived case per proposal;
3. optionally rebuilds the counted case itself, through the committed builder, to
   confirm the representative reproduces;
4. runs the whole suite in compiled mode with the ordinary runner, so every
   proposal is compared against its own reference with the recorded comparator
   and carries the runner's campaign validity;
5. extracts the loop observations, the layout-conversion counts, and the named
   passes from the archived IR and the pass dump.

Nothing here relaxes a comparator, widens a domain, or supplies a witness: a
proposal the exact reference refuses is recorded as a refusal and falls back to
the interpreter replay the plan names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

from test.lib import (
  probe_triton_env,
  tensor_case,
  tensor_compiler_observation,
  tensor_exact_f32,
  tensor_input_proposals,
  tensor_region_generation,
)
from test.lib.build_tensor_exact_loop_case import _record, _run

DOMAIN = (0, 8)
REFERENCE_PRECISIONS = (128, 256)
SCHEMA = "refractir.track-a-input-proposals.v1"
TREE = Path(__file__).resolve().parents[3]
REPRESENTATIVE = "mixed-depth"
FAMILIES = (
  "mixed-depth",
  "cross-state",
  "reduction-arithmetic",
  "orthogonal-reductions",
)
# The recorded identity of the counted representative, from
# `test/tensor/campaigns/results/track-a-paired-unrolled-device.json` and the
# counted case its qualified re-run carried.
COUNTED_IDENTITY = {
  "program.sir": "638322a9be95072a6d59bcea366d0f4d9d4445a3aa63305765a44637c77dedcd",
  "inputs.bin": "f60bb874d66083b2d5d823eba8362711995682c8f18cc3f552be9cd380ad524e",
  "expected.bin": "f5826a578a9ae11e5a2712ed690b41a0e5af93947ecd50bac4f87660432233de",
  "kernel.py": "2b5e96b0fd011a79ad44dceb3e4b7e10872d719b639006293ce7bd1fcda2f38a",
}
# Passes whose own before/after transition the pass-dump observer reports.
TARGET_PASSES = (
  "TritonGPURemoveLayoutConversions",
  "TritonGPUReduceDataDuplication",
  "TritonGPUOptimizeThreadLocality",
  "TritonLoopInvariantCodeMotion",
  "SCFToControlFlowPass",
)
CASE_FILES = (
  "program.sir",
  "inputs.bin",
  "expected.bin",
  "certificate.json",
  "reference-128.bin",
  "reference-256.bin",
)
# The archived stages that are IR; `compiled.asm` may carry others, such as the
# emitted Python kernel, which no loop or layout observation may read.
IR_STAGES = ("ttir", "ttgir", "llir", "ptx", "sass")


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


def imported_modules() -> dict:
  """The digest of every `test.lib` module this process actually imported."""
  modules = {}
  for name, module in sorted(sys.modules.items()):
    if not name.startswith("test.lib"):
      continue
    path = getattr(module, "__file__", None)
    if path is None:
      continue
    try:
      data = Path(path).read_bytes()
    except OSError:
      continue
    modules[name] = hashlib.sha256(data).hexdigest()
  return modules


def library_inventory(root=TREE) -> dict:
  """The digest of every Python module in the imported test library."""
  library = Path(root) / "test" / "lib"
  return {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(library.glob("*.py"))
  }


def _probe_kernel(source, domain):
  """Admit the region over the domain and read a reachable target from it."""
  shape, _iterations, _types, _nodes, _updates = (
    tensor_exact_f32._generated_region_program(source)
  )
  count = math.prod(shape)
  probe_input = struct.pack(f"<{count}f", *([float(domain[1])] * count))
  try:
    proof, expected = tensor_exact_f32.certify(source, probe_input, domain)
  except ValueError as error:
    return {
      "status": "refused",
      "reason": str(error),
      "probe": "all-lanes-at-domain-upper-bound",
      "probe_input_sha256": hashlib.sha256(probe_input).hexdigest(),
      "domain_headroom": _domain_headroom(source, domain, count),
    }
  zero_input = struct.pack(f"<{count}f", *([0.0] * count))
  zero_proof, zero_expected = tensor_exact_f32.certify(source, zero_input, domain)
  return {
    "status": "admitted",
    "probe": "all-lanes-at-domain-upper-bound",
    "probe_input_sha256": hashlib.sha256(probe_input).hexdigest(),
    "target": int(struct.unpack(f"<{count}f", expected)[0]),
    "reference_sha256": hashlib.sha256(expected).hexdigest(),
    "zero_probe": {
      "input": "all-lanes-at-domain-lower-bound",
      "input_sha256": hashlib.sha256(zero_input).hexdigest(),
      "output_sha256": hashlib.sha256(zero_expected).hexdigest(),
      "output": tensor_input_proposals.constant_output(zero_expected),
      "obligation_count": len(zero_proof.get("bounds") or []),
    },
    # A family whose output cannot change with its input would not carry a
    # behavior experiment, so the two extreme probes are compared here.
    "output_depends_on_input": zero_expected != expected,
    "proof": proof_summary(proof),
  }


def _domain_headroom(source, domain, count):
  """Diagnostic: which domain upper bounds would admit this region.

  It quantifies a refusal without producing a case: no proposal is built from
  these bounds, and the frozen configuration of the experiment is unchanged.
  """
  admitted = []
  for high in range(domain[0], domain[1] + 1):
    probe = struct.pack(f"<{count}f", *([float(high)] * count))
    try:
      tensor_exact_f32.certify(source, probe, (domain[0], high))
    except ValueError:
      continue
    admitted.append(high)
  return {
    "admitted_upper_bounds": admitted,
    "largest_admitted_upper_bound": max(admitted) if admitted else None,
    "note": "diagnostic only; no proposal case was built from these bounds",
  }


def proof_summary(proof):
  """The bound obligations of an exact certificate, without its per-lane list."""
  bounds = proof.get("bounds") or []
  maxima = [
    entry["max_value"] for entry in bounds if type(entry.get("max_value")) is int
  ]
  partials = [
    entry["max_partial_sum"]
    for entry in bounds
    if type(entry.get("max_partial_sum")) is int
  ]
  return {
    "kind": proof.get("kind"),
    "complete_output": proof.get("complete_output"),
    "input_domain": proof.get("input_domain"),
    "integer_exactness_bound": proof.get("integer_exactness_bound"),
    "signed_zero": proof.get("signed_zero"),
    "allowed_operations": proof.get("allowed_operations"),
    "bound_method": proof.get("bound_method"),
    "path": proof.get("path"),
    "state_update_order": proof.get("state_update_order"),
    "obligation_count": len(bounds),
    "max_materialized_value": max(maxima) if maxima else None,
    "max_partial_sum": max(partials) if partials else None,
    "blocks": sorted({entry.get("block") for entry in bounds}),
    "certificate_sha256": hashlib.sha256(_canonical(proof).encode()).hexdigest(),
  }


def _replay(directory, symiri, bits, deadline, ledger):
  output = Path(directory) / f"reference-{bits}.bin"
  command = [
    str(symiri),
    "--tensor-profile-version",
    "v2",
    "--tensor-reference-bits",
    str(bits),
    "--entry",
    "kernel",
    "--tensor-input",
    str(Path(directory) / "inputs.bin"),
    "--tensor-output",
    str(output),
    str(Path(directory) / "program.sir"),
  ]
  result = _run("reference", command, deadline=deadline, ledger=ledger)
  return {
    "reference_bits": bits,
    "file": output.name,
    "command": command,
    "exit_code": result.returncode,
    "data": output.read_bytes(),
  }


def _proposal_manifest(
  *,
  source,
  proposal,
  input_bytes,
  expected,
  certificate,
  counted_manifest,
  provenance,
):
  shape = list(tensor_exact_f32._generated_region_program(source)[0])
  files = {
    name: (Path(certificate["directory"]) / name).read_bytes() for name in CASE_FILES
  }
  program = source.encode("utf-8")
  files["program.sir"] = program
  files["inputs.bin"] = input_bytes
  files["expected.bin"] = expected
  input_record = _record("x", "f32", shape, "inputs.bin", len(input_bytes))
  output_record = _record("return", "f32", shape, "expected.bin", len(expected))
  domain = {"lo": certificate["domain"][0], "hi": certificate["domain"][1]}
  return {
    "schema": counted_manifest["schema"],
    "profile": counted_manifest["profile"],
    "entry": counted_manifest["entry"],
    "source": {
      "file": "program.sir",
      "sha256": hashlib.sha256(program).hexdigest(),
      "generator": "generated-bounded-region-input-proposal",
      "seed": counted_manifest["source"]["seed"],
    },
    "inputs": [input_record],
    "output": output_record,
    "grid": list(counted_manifest["grid"]),
    "softmax_lowering": counted_manifest["softmax_lowering"],
    "launch": dict(counted_manifest["launch"]),
    "property": {
      "id": f"input-proposal-{proposal.name}-complete-reference",
      "source": (
        f"Run the counted logical region on the deterministic {proposal.name} "
        "proposal and compare the complete output with the reference this "
        "proposal carries."
      ),
    },
    "evidence": {
      "kind": "numerical_reference",
      "status": "numerical_reference_checked",
      "artifacts": sorted(files),
    },
    # The exact comparator needs solver selection evidence whose model is the
    # runtime input, and a proposal is chosen by a rule rather than by the
    # solver. The reference is therefore the certificate's own bytes, compared
    # with zero tolerance here and checked word-for-word in the driver.
    "comparison": {"mode": "numeric", "atol": [0, 1], "rtol": [0, 1]},
    "reference": {
      "defined": True,
      "stable": True,
      "precisions": list(REFERENCE_PRECISIONS),
      "exactness": None,
      "source": certificate["reference_kind"],
      "note": (
        "Expected bytes are the exact f32 certificate's complete output where "
        "the reference admits the proposal, otherwise a native RefractIR replay; "
        "the replays at both reference precisions have to reproduce the expected "
        "bytes for the case to be admitted. An exact comparator would need "
        "solver-selection evidence for this input, which a rule-chosen proposal "
        "cannot have."
      ),
    },
    "execution": dict(counted_manifest["execution"]),
    "coverage": {
      "primary_stratum": "generated-bounded-region-input-proposal",
      "source_signature": counted_manifest["coverage"]["source_signature"],
      "loop_structure": counted_manifest["coverage"]["loop_structure"],
      "input_behavior": {
        "policy": "exact-domain-proposal",
        "input_domain": domain,
        "input_mode": "deterministic-input-proposal",
        "pattern": proposal.name,
        "proposal": proposal.record(),
      },
    },
    "environment": None,
    "operator_coverage": tensor_exact_f32.generated_region_operations(source),
    "shape_axes": list(counted_manifest.get("shape_axes", [])),
    "generator": {
      "version": "generated-bounded-region-input-proposal-v1",
      "seed": counted_manifest["source"]["seed"],
      "region": counted_manifest["generator"]["region"],
      "input_mode": "deterministic-input-proposal",
      "input_policy": "exact-domain-proposal",
      "input_domain": domain,
      "input_pattern": proposal.name,
      "synthesis": "deterministic-derivation-from-case",
      "primary_stratum": "generated-bounded-region-input-proposal",
      "node_limit": counted_manifest["generator"]["node_limit"],
      "base_case": provenance,
    },
    "input_proposal": {**proposal.record(), "provenance": provenance},
    "exact_cross_check": certificate["record"],
    "files": {name: _digest(data) for name, data in sorted(files.items())},
  }


def _provenance(family, source, generation, reference_case, domain):
  """What a proposal is derived from, stated inside the case that carries it."""
  return {
    "family": family,
    "program_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    "region": generation["configuration"],
    "rules": "test.lib.tensor_input_proposals",
    "input_domain": {"lo": domain[0], "hi": domain[1]},
    "reference_case": reference_case,
  }


def _reference_case(counted_manifest, counted_dir):
  """The counted case this campaign treats as the admitted anchor."""
  counted_dir = Path(counted_dir)
  inputs = counted_dir / "inputs.bin"
  expected = counted_dir / "expected.bin"
  return {
    "case": str(counted_dir),
    "program_sha256": counted_manifest["source"]["sha256"],
    "region": counted_manifest["generator"]["region"],
    "comparison": counted_manifest["comparison"],
    "inputs_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest()
    if inputs.is_file()
    else None,
    "expected_sha256": hashlib.sha256(expected.read_bytes()).hexdigest()
    if expected.is_file()
    else None,
  }


def build_proposal_case(
  directory,
  *,
  source,
  proposal,
  domain,
  counted_manifest,
  provenance,
  symiri,
  ledger,
  deadline,
):
  """Write one derived proposal case with its own complete reference."""
  directory = Path(directory)
  directory.mkdir(parents=True, exist_ok=False)
  (directory / "program.sir").write_text(source, encoding="utf-8")
  input_bytes = tensor_input_proposals.pack(proposal, domain=domain)
  (directory / "inputs.bin").write_bytes(input_bytes)

  certificate = {
    "proposal": proposal.name,
    "domain": list(domain),
    "admission": None,
    "reference_kind": None,
    "directory": str(directory),
    "record": None,
  }
  proof = None
  expected = None
  try:
    proof, expected = tensor_exact_f32.certify(source, input_bytes, domain)
    certificate["admission"] = "admitted-exact-certificate"
    certificate["reference_kind"] = "exact-certificate"
  except ValueError as error:
    certificate["admission"] = "refused-exact-certificate"
    certificate["refusal_reason"] = str(error)
    certificate["reference_kind"] = "native-refractir-replay"

  replays = [
    _replay(directory, symiri, bits, deadline, ledger) for bits in REFERENCE_PRECISIONS
  ]
  replay_bytes = {record["reference_bits"]: record["data"] for record in replays}
  if proof is not None:
    if expected != replay_bytes[128] or expected != replay_bytes[256]:
      raise ValueError(
        f"proposal {proposal.name}: the certificate and the interpreter replays "
        "disagree, so no reference can be admitted"
      )
    tensor_exact_f32.validate(source, input_bytes, expected, proof)
  else:
    if replay_bytes[128] != replay_bytes[256]:
      raise ValueError(
        f"proposal {proposal.name}: the two reference precisions disagree"
      )
    expected = replay_bytes[128]
  (directory / "expected.bin").write_bytes(expected)

  certificate["expected_sha256"] = hashlib.sha256(expected).hexdigest()
  certificate["replays"] = [
    {
      "reference_bits": record["reference_bits"],
      "file": record["file"],
      "sha256": hashlib.sha256(record["data"]).hexdigest(),
      "command": record["command"],
      "matches_expected": record["data"] == expected,
    }
    for record in replays
  ]
  certificate["record"] = {
    "admission": certificate["admission"],
    "refusal_reason": certificate.get("refusal_reason"),
    "reference_kind": certificate["reference_kind"],
    "expected_sha256": certificate["expected_sha256"],
    "certificate_file": "certificate.json",
    "certificate_sha256": hashlib.sha256(_canonical(proof).encode()).hexdigest()
    if proof is not None
    else None,
    "proof_summary": proof_summary(proof) if proof is not None else None,
    "replays": certificate["replays"],
    "replay_matches_certificate": proof is not None,
    "why_not_exact_comparator": (
      "an exact comparator requires the case loader's solver-selection evidence, "
      "whose model must equal the runtime input bytes; this proposal is chosen by "
      "a rule, so its reference is compared with zero tolerance instead"
    ),
  }
  # The certificate is written once, before the manifest digests it: the loader
  # checks every declared file against the manifest's own record.
  _write_json(directory / "certificate.json", {"proof": proof, **certificate["record"]})

  manifest = _proposal_manifest(
    source=source,
    proposal=proposal,
    input_bytes=input_bytes,
    expected=expected,
    certificate=certificate,
    counted_manifest=counted_manifest,
    provenance=provenance,
  )
  _write_json(directory / "case.json", manifest)
  loaded = tensor_case.load_case(directory, symiri=symiri)
  loader = {
    "status": "admitted",
    "inputs_sha256": hashlib.sha256(loaded.inputs).hexdigest(),
    "expected_sha256": hashlib.sha256(loaded.expected).hexdigest(),
  }
  _write_json(
    directory / "proposal-record.json",
    {
      "name": proposal.name,
      "proposal": proposal.record(),
      "certificate": certificate["record"],
      "admission": certificate["admission"],
      "reference_kind": certificate["reference_kind"],
      "expected_sha256": certificate["expected_sha256"],
      "input_bytes_sha256": hashlib.sha256(input_bytes).hexdigest(),
      "input_bytes": len(input_bytes),
      "loader": loader,
      "replays": certificate["replays"],
      "files": {name: _digest((directory / name).read_bytes()) for name in CASE_FILES},
    },
  )
  return directory


def build_family(work, args, family):
  """Generate one region and admit it over the declared domain."""
  source_path = Path(work) / "sources" / f"{family}.sir"
  source_path.parent.mkdir(parents=True, exist_ok=True)
  configuration = tensor_region_generation.RegionConfiguration(
    seed=args.seed,
    shape=tuple(int(value) for value in args.shape.split(",")),
    iterations=args.iterations,
    family=family,
    operation_mode=args.operation_mode,
    max_operators=args.max_operators,
  )
  if source_path.is_file() and not args.force:
    source = source_path.read_text(encoding="utf-8")
    generation = {"status": "reused", "path": str(source_path)}
  else:
    source = tensor_region_generation.generate_source(
      configuration, exporter=args.exporter
    )
    source_path.write_text(source, encoding="utf-8")
    generation = {"status": "generated", "path": str(source_path)}
  generation["configuration"] = configuration.as_record()
  generation["identity"] = configuration.identity
  generation["source"] = _digest(source.encode("utf-8"))
  generation["admission"] = _probe_kernel(
    source, tuple(int(value) for value in args.domain.split(","))
  )
  generation["counts_as_counted_program"] = (
    generation["source"]["sha256"] == COUNTED_IDENTITY["program.sir"]
    if family == args.representative
    else None
  )
  return source, configuration, generation


def step_build(work, args, state):
  """Derive and write the proposal cases for every requested family."""
  built = {}
  counted_manifest = _read_json(args.counted_manifest)
  reference_case = _reference_case(counted_manifest, args.counted_case)
  for family in args.families.split(","):
    source, _configuration, generation = build_family(work, args, family)
    record = {"generation": generation, "proposals": {}, "duplicate_inputs": []}
    built[family] = record
    if generation["admission"]["status"] != "admitted":
      record["status"] = "kernel_refused"
      continue
    domain = tuple(int(value) for value in args.domain.split(","))
    provenance = _provenance(family, source, generation, reference_case, domain)
    proposals = tensor_input_proposals.derive_proposals(
      source, domain=domain, seed=args.seed
    )
    record["duplicate_inputs"] = tensor_input_proposals.duplicate_groups(proposals)
    for proposal in proposals:
      directory = Path(work) / "suite" / f"{family}--{proposal.name}"
      if directory.is_dir():
        if not args.force:
          # A case that is already on disk keeps its own record; the report
          # reads the same fields from a reused and a freshly built case.
          proposal_record = _read_json(directory / "proposal-record.json")
          record["proposals"][proposal.name] = {
            "status": "reused",
            "directory": str(directory),
            "admission": proposal_record["admission"],
            "reference_kind": proposal_record["reference_kind"],
            "expected_sha256": proposal_record["expected_sha256"],
            "input_bytes_sha256": proposal_record["input_bytes_sha256"],
            "loader": proposal_record["loader"],
          }
          continue
        shutil.rmtree(directory)
      deadline = time.monotonic() + args.stage_timeout
      try:
        build_proposal_case(
          directory,
          source=source,
          proposal=proposal,
          domain=domain,
          counted_manifest=counted_manifest,
          provenance=provenance,
          symiri=args.symiri,
          ledger=state.setdefault("ledger", []),
          deadline=deadline,
        )
      except (ValueError, tensor_case.CaseValidationError) as error:
        record["proposals"][proposal.name] = {
          "status": "refused",
          "reason": str(error),
        }
        continue
      proposal_record = _read_json(directory / "proposal-record.json")
      record["proposals"][proposal.name] = {
        "status": "built",
        "directory": str(directory),
        "admission": proposal_record["admission"],
        "reference_kind": proposal_record["reference_kind"],
        "expected_sha256": proposal_record["expected_sha256"],
        "input_bytes_sha256": proposal_record["input_bytes_sha256"],
        "loader": proposal_record["loader"],
      }
    record["status"] = (
      "admitted"
      if any(
        entry.get("status") in {"built", "reused"}
        for entry in record["proposals"].values()
      )
      else "no_proposal_admitted"
    )
  _write_json(Path(work) / "build-records.json", built)
  return built


def step_rebuild(work, args, state):
  """Rebuild the counted representative and check its recorded identity."""
  from test.lib import build_tensor_exact_loop_case, build_tensor_exact_region_case

  # This 512-lane integer-twin query needs minutes of solver time; the library
  # default stays untouched and only this driver's stages are raised.
  build_tensor_exact_loop_case.TIMEOUT_S = args.solver_timeout
  source, configuration, generation = build_family(work, args, args.representative)
  admission = generation["admission"]
  if admission["status"] != "admitted":
    return {"status": "refused", "reason": admission.get("reason")}
  directory = Path(work) / "suite" / f"counted-{args.representative}"
  if directory.exists() and not args.force:
    digests = {
      name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
      for name in COUNTED_IDENTITY
      if (directory / name).is_file()
    }
    return {
      "status": "already_present",
      "directory": str(directory),
      "digests": digests,
      "matches_recorded_identity": {
        name: digests.get(name) == digest for name, digest in COUNTED_IDENTITY.items()
      },
    }
  if directory.exists():
    shutil.rmtree(directory)
  build_tensor_exact_region_case.build_case(
    directory,
    source=source,
    target=admission["target"],
    symiri=args.symiri,
    symirc=args.symirc,
    solver=args.symirsolve,
    domain=tuple(int(value) for value in args.domain.split(",")),
    timeout_s=args.solver_timeout,
    configuration=configuration,
  )
  digests = {
    name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
    for name in COUNTED_IDENTITY
    if (directory / name).is_file()
  }
  return {
    "status": "rebuilt",
    "directory": str(directory),
    "target": admission["target"],
    "seconds": None,
    "digests": digests,
    "matches_recorded_identity": {
      name: digests.get(name) == digest for name, digest in COUNTED_IDENTITY.items()
    },
  }


def step_run(work, args, state):
  """Run the whole suite in compiled mode through the ordinary runner."""
  environment = dict(os.environ)
  environment.update(args.environment)
  if args.pass_dump:
    environment["REFRACTIR_TRITON_PASS_DUMP"] = "1"
  suite = Path(work) / "suite"
  logs = Path(work) / "logs"
  logs.mkdir(parents=True, exist_ok=True)
  command = [
    sys.executable,
    "-m",
    "test.lib.run_triton_tests",
    "--suite",
    str(suite),
    "--mode",
    "compiled",
    "--timeout",
    str(args.run_timeout),
  ]
  started = time.monotonic()
  result = subprocess.run(
    command,
    cwd=str(TREE),
    env=environment,
    capture_output=True,
    text=True,
    check=False,
  )
  (logs / "runner.stdout.log").write_text(result.stdout, encoding="utf-8")
  (logs / "runner.stderr.log").write_text(result.stderr, encoding="utf-8")
  summary = None
  for line in reversed(result.stdout.splitlines()):
    line = line.strip()
    if line.startswith("{"):
      try:
        summary = json.loads(line)
      except json.JSONDecodeError:
        continue
      break
  return {
    "command": command,
    "exit_code": result.returncode,
    "seconds": round(time.monotonic() - started, 3),
    "summary": summary,
    "environment": {
      key: environment.get(key)
      for key in (
        "REFRACTIR_REVISION",
        "REFRACTIR_COMPILER_FLAGS",
        "REFRACTIR_WORKTREE_DIRTY",
        "REFRACTIR_TRITON_PASS_DUMP",
        "PYTHONPATH",
      )
    },
  }


def ir_artifacts(case_dir):
  """The archived IR stages, keyed by stage name.

  `compiled.asm` also carries the emitted Python kernel under `source`, which is
  not IR: its `for` statements are not loop constructs, so it is kept apart from
  the loop verdict.
  """
  run_dir = Path(case_dir) / "runs" / "compiled"
  artifacts = {}
  other = {}
  for path in sorted(run_dir.glob("compiler-*.ir")):
    stage = path.name[len("compiler-") : -len(".ir")]
    text = path.read_text(encoding="utf-8")
    if stage in IR_STAGES:
      artifacts[stage] = text
    else:
      other[stage] = _digest(text.encode("utf-8"))
  return artifacts, other


def _structural_digest(text, case_dir):
  """The IR digest with the case's own paths removed.

  Every case lives in its own directory and the compiler embeds that path in
  source locations and debug metadata, so two identical compilations of the same
  emitted kernel differ in that string alone.
  """
  root = str(Path(case_dir).resolve())
  normalized = text.replace(f"{root}/runs/compiled/kernel.py", "<kernel>").replace(
    root, "<case>"
  )
  return {
    "bytes": len(normalized),
    "sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
  }


def loop_and_conversion_observations(case_dir):
  """The loop form and the layout conversions of every archived IR stage."""
  artifacts, other = ir_artifacts(case_dir)
  if not artifacts:
    return {"status": "unavailable", "reason": "no archived compiler IR"}
  observed = tensor_compiler_observation.observe_compiler_ir(
    artifacts, expected_loop=True, loop_required_stages=tuple(artifacts)
  )
  stages = {}
  conversions = {}
  for stage, record in observed["stages"].items():
    stages[stage] = {
      "defined_value_count": record["defined_value_count"],
      "operation_counts": record["operation_counts"],
      "loop_construct_count": record["loop_construct_count"],
      "loop_constructs": record["loop_constructs"],
      "backedge_count": record["backedge_count"],
      "unparsed_branch_count": record["unparsed_branch_count"],
      "layout_conversion_operations": len(record["layout_conversions"]),
      "effective_layout_conversions": sum(
        entry["layout_changed"] for entry in record["layout_conversions"]
      ),
      "duplicate_expression_groups": len(record["duplicate_expressions"]),
      "loop_evidence": observed["loop_evidence"].get(stage),
      "loop_presence": observed["loop_presence"].get(stage),
    }
    if stage.lower() in {"ptx", "sass"}:
      stages[stage]["warp_shuffle_count"] = record.get("warp_shuffle_count")
      stages[stage]["shared_memory_transfer_count"] = record.get(
        "shared_memory_transfer_count"
      )
    conversions[stage] = [
      {
        "result": entry["result"],
        "source_value": entry["source_value"],
        "source_operation": entry["source_operation"],
        "source_layout": entry["source_layout"],
        "target_layout": entry["target_layout"],
        "layout_changed": entry["layout_changed"],
        "enclosing_region": entry["enclosing_region"],
        "consumers": entry["consumers"],
        "ir_line": entry["ir_line"],
      }
      for entry in record["layout_conversions"]
    ]
  return {
    "status": "observed",
    "loop_status": observed["loop_status"],
    "loop_required_stages": observed["loop_required_stages"],
    "loop_constructs_by_stage": {
      stage: record["loop_construct_count"] for stage, record in stages.items()
    },
    "backedges_by_stage": {
      stage: record["backedge_count"] for stage, record in stages.items()
    },
    "effective_layout_conversions_by_stage": {
      stage: record["effective_layout_conversions"] for stage, record in stages.items()
    },
    "stages": stages,
    "layout_conversions": conversions,
    "ir_digests": {
      stage: _digest(text.encode("utf-8")) for stage, text in sorted(artifacts.items())
    },
    "structural_ir_digests": {
      stage: _structural_digest(text, case_dir)
      for stage, text in sorted(artifacts.items())
    },
    "non_ir_asm_keys": other,
    "non_ir_asm_note": (
      "compiled.asm also carries the emitted Python kernel; it is not IR and is "
      "excluded from the loop verdict"
    ),
  }


def pass_dump_observations(case_dir):
  """Named-pass transitions of the pass dump, when one was archived."""
  path = Path(case_dir) / "runs" / "compiled" / "compiler-pass-dump.txt"
  if not path.is_file():
    return {"status": "unavailable", "reason": "no pass dump was archived"}
  try:
    observed = tensor_compiler_observation.observe_pass_dumps(
      path.read_text(encoding="utf-8", errors="replace"),
      target_passes=TARGET_PASSES,
      rematerialization=True,
    )
  except ValueError as error:
    return {"status": "unavailable", "reason": str(error)}
  compilations = []
  for compilation in observed["compilations"]:
    transitions = []
    occurrences = {name: [] for name in TARGET_PASSES}
    for index, transition in enumerate(compilation["transitions"]):
      materialization = transition.get("rematerialization") or {}
      summary = {
        "pass": transition["pass"],
        "pipeline": transition["pipeline"],
        "next_snapshot_pass": transition["next_snapshot_pass"],
        "layout_conversions_before": transition["before"]["layout_conversion_count"],
        "layout_conversions_after": transition["after"]["layout_conversion_count"],
        "layout_conversion_delta": transition["layout_conversion_delta"],
        "effective_layout_conversion_delta": transition[
          "effective_layout_conversion_delta"
        ],
        "effective_conversion_operation_deltas": transition[
          "effective_conversion_operation_deltas"
        ],
        "conversion_match": transition["conversion_match"],
        "conversion_movement": len(transition["conversion_movement"]),
        "loop_construct_delta": transition["loop_construct_delta"],
        "rematerialization": {
          "status": materialization.get("status"),
          "confirmed_count": materialization.get("confirmed_count"),
          "unattributed_count": materialization.get("unattributed_count"),
        },
      }
      if transition["pass"] in occurrences:
        occurrences[transition["pass"]].append(
          {
            "transition_index": index,
            "loop_constructs_before": transition["before"]["loop_construct_count"],
            "loop_constructs_after": transition["after"]["loop_construct_count"],
            **{
              key: summary[key]
              for key in (
                "layout_conversion_delta",
                "effective_layout_conversion_delta",
                "effective_conversion_operation_deltas",
                "conversion_match",
                "conversion_movement",
                "loop_construct_delta",
              )
            },
          }
        )
      interesting = any(
        (
          transition["layout_conversion_delta"],
          transition["effective_layout_conversion_delta"],
          transition["loop_construct_delta"],
          transition["conversion_match"]["removed"],
          transition["conversion_match"]["added"],
          transition["conversion_match"]["moved"],
          materialization.get("confirmed_count", 0),
        )
      )
      if not interesting and transition["pass"] not in occurrences:
        continue
      transitions.append(summary)
    compilations.append(
      {
        "snapshot_count": compilation["snapshot_count"],
        "terminal_pass": compilation["terminal_pass"],
        "final_loop_construct_count": compilation["final_loop_construct_count"],
        "target_pass_evidence": {
          name: {
            "status": evidence["status"],
            "transition_indices": evidence["transition_indices"],
            "occurrences": evidence["occurrences"],
          }
          for name, evidence in compilation["target_pass_evidence"].items()
        },
        "named_pass_occurrences": occurrences,
        "transitions": transitions,
      }
    )
  return {
    "status": "observed",
    "compilation_count": observed["compilation_count"],
    "target_passes": list(TARGET_PASSES),
    "compilations": compilations,
    "pass_dump": _digest(path.read_bytes()),
    "interpretation_limit": observed["interpretation_limit"],
  }


def _emitted_kernel_identity(proposals):
  """Whether the four proposals compiled the same emitted kernel and IR.

  A proposal changes runtime data, not the program, so the emitted kernel and
  the structural IR should be identical across proposals; only the embedded case
  path differs. Recording the comparison keeps that claim checkable.
  """
  kernels = {
    entry["execution"]["artifact_digests"]["runs/compiled/kernel.py"]["sha256"]
    for entry in proposals
    if entry.get("execution", {}).get("artifact_digests")
  }
  stages = {}
  for entry in proposals:
    observations = entry.get("observations") or {}
    for stage, digest in (observations.get("structural_ir_digests") or {}).items():
      stages.setdefault(stage, set()).add(digest["sha256"])
  return {
    "emitted_kernel_sha256": sorted(kernels),
    "identical_across_proposals": len(kernels) == 1,
    "matches_counted_case_kernel": kernels == {COUNTED_IDENTITY["kernel.py"]},
    "structural_ir_identical_across_proposals": {
      stage: len(digests) == 1 for stage, digests in sorted(stages.items())
    },
    "structural_ir_note": (
      "structural digests replace each case's own kernel path, which the compiler "
      "embeds in source locations; PTX additionally encodes that path length in "
      "its debug section, so its structural digest is not expected to agree"
    ),
  }


def _input_comparison(proposals, counted_inputs_sha256):
  """Every pairwise input comparison, so a coincidence cannot go unrecorded."""
  digests = {
    entry["name"]: (entry.get("reference") or {}).get("input_bytes_sha256")
    for entry in proposals
  }
  names = [entry["name"] for entry in proposals]
  pairs = [
    [left, right]
    for index, left in enumerate(names)
    for right in names[index + 1 :]
    if digests.get(left) and digests.get(right)
  ]
  identical = [pair for pair in pairs if digests[pair[0]] == digests[pair[1]]]
  return {
    "per_proposal_input_sha256": digests,
    "proposals_with_inputs": sum(1 for digest in digests.values() if digest),
    "pairs_compared": len(pairs),
    "identical_pairs": identical,
    "identical_to_counted_witness": sorted(
      name
      for name, digest in digests.items()
      if digest and digest == counted_inputs_sha256
    ),
    "counted_witness_sha256": counted_inputs_sha256,
  }


def _bit_level_check(case_dir, expected):
  """Compare the device bytes with the reference word for word."""
  actual_path = Path(case_dir) / "runs" / "compiled" / "actual.bin"
  if not actual_path.is_file():
    return {"status": "unavailable"}
  actual = actual_path.read_bytes()
  if len(actual) != len(expected):
    return {"status": "length_mismatch", "actual_bytes": len(actual)}
  actual_words = struct.unpack(f"<{len(actual) // 4}I", actual)
  expected_words = struct.unpack(f"<{len(expected) // 4}I", expected)
  mismatches = [
    index
    for index, (left, right) in enumerate(zip(actual_words, expected_words))
    if left != right
  ]
  zero_indices = [index for index, word in enumerate(expected_words) if word == 0]
  return {
    "status": "pass" if not mismatches else "exact_mismatch",
    "mismatching_indices": mismatches,
    "word_exact": not mismatches,
    # Zero tolerance alone treats +0.0 and -0.0 as equal, so the zero lanes are
    # compared by word as a separate observation about the same comparison.
    "zero_lanes": len(zero_indices),
    "signed_zero_exact": all(actual_words[index] == 0 for index in zero_indices),
    "actual_sha256": hashlib.sha256(actual).hexdigest(),
  }


def step_report(work, args, state):
  """Assemble the machine-readable proposal records."""
  work = Path(work)
  build_records = _read_json(work / "build-records.json")
  counted_inputs = Path(args.counted_case) / "inputs.bin"
  counted_inputs_sha256 = (
    hashlib.sha256(counted_inputs.read_bytes()).hexdigest()
    if counted_inputs.is_file()
    else None
  )
  environment = probe_triton_env.collect_environment()
  report = {
    "schema": SCHEMA,
    "what": (
      "the recorded input proposals for the admitted Track A regions, each with "
      "its own complete reference and a compiled-mode device run"
    ),
    "proposal_rules": {
      "names": list(tensor_input_proposals.PROPOSAL_NAMES),
      "derivation": tensor_input_proposals.DERIVATION_VERSION,
      "domain": args.domain,
      "seed": args.seed,
      "duplicate_policy": (
        "coincident proposals are grouped and reported, never dropped"
      ),
      "constant_output_policy": (
        "a constant reference output is reported per proposal"
      ),
    },
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
      # The probe reads `git rev-parse HEAD` in the working directory; the
      # exported tree carries no git metadata, so the runner's own environment
      # label is the operative revision of the imported modules.
      "refractir_revision": (state.get("run", {}).get("environment") or {}).get(
        "REFRACTIR_REVISION"
      ),
      "refractir_revision_probe": environment["refractir_revision"],
      "package_lock_sha256": environment["package_lock_sha256"],
      "runner_environment": state.get("run", {}).get("environment"),
      "tool_paths": {
        "symiri": args.symiri,
        "symirc": args.symirc,
        "symirsolve": args.symirsolve,
        "exporter": args.exporter,
      },
      "tool_digests": {
        name: _digest(Path(path).read_bytes()) if Path(path).is_file() else None
        for name, path in (
          ("symiri", args.symiri),
          ("symirc", args.symirc),
          ("symirsolve", args.symirsolve),
          ("exporter", args.exporter),
        )
      },
      "driver_digest": _digest(Path(__file__).read_bytes()),
    },
    "imports": {
      "imported_module_sha256": imported_modules(),
      "test_library_inventory_sha256": library_inventory(),
      "note": (
        "every test.lib module this driver imported, and the digest of every "
        "module in the imported test library"
      ),
    },
    "runner": state.get("run"),
    "counted_rebuild": state.get("rebuild"),
    "counted_inputs_sha256": counted_inputs_sha256,
    "kernels": [],
  }
  for family, build_record in build_records.items():
    kernel = {
      "family": family,
      "configuration": build_record["generation"]["configuration"],
      "identity": build_record["generation"]["identity"],
      "source": build_record["generation"]["source"],
      "counts_as_counted_program": build_record["generation"].get(
        "counts_as_counted_program"
      ),
      "admission": build_record["generation"]["admission"],
      "duplicate_inputs": build_record.get("duplicate_inputs", []),
      "proposals": [],
    }
    for name in tensor_input_proposals.PROPOSAL_NAMES:
      entry = build_record["proposals"].get(name)
      if entry is None or entry.get("status") not in {"built", "reused"}:
        kernel["proposals"].append(
          {
            "name": name,
            "admission": (entry or {}).get("admission", "not_built"),
            "reason": (entry or {}).get("reason"),
            "execution": {"status": "not_run"},
          }
        )
        continue
      directory = Path(entry["directory"])
      proposal_record = _read_json(directory / "proposal-record.json")
      manifest = _read_json(directory / "case.json")
      expected = (directory / "expected.bin").read_bytes()
      result = (
        _read_json(directory / "result-compiled.json")
        if (directory / "result-compiled.json").is_file()
        else None
      )
      record = {
        "name": name,
        "directory": str(directory),
        "proposal": proposal_record["proposal"],
        "admission": proposal_record["admission"],
        "reference": {
          "kind": proposal_record["reference_kind"],
          "expected_sha256": proposal_record["expected_sha256"],
          "expected_matches_case": hashlib.sha256(expected).hexdigest()
          == proposal_record["expected_sha256"],
          "input_bytes_sha256": proposal_record["input_bytes_sha256"],
          "loader": proposal_record["loader"],
          "replays": proposal_record["replays"],
          "certificate": proposal_record["certificate"],
        },
        "comparison": manifest["comparison"],
        "output": tensor_input_proposals.constant_output(expected),
        "duplicate_inputs": {
          "proposal_groups": [
            group for group in build_record.get("duplicate_inputs", []) if name in group
          ],
          "identical_to_counted_witness": proposal_record["input_bytes_sha256"]
          == counted_inputs_sha256,
          "counted_witness_sha256": counted_inputs_sha256,
        },
        "execution": {"status": "not_run"},
      }
      if result is not None:
        record["execution"] = {
          "status": result.get("status"),
          "comparison": result.get("comparison"),
          "campaign_valid": result.get("campaign_valid"),
          "exit_code": result.get("exit_code"),
          "artifact_digests": result.get("artifact_digests"),
        }
        record["bit_level_check"] = _bit_level_check(directory, expected)
        record["observations"] = loop_and_conversion_observations(directory)
        record["pass_dump"] = pass_dump_observations(directory)
      kernel["proposals"].append(record)
    kernel["constant_outputs"] = [
      {"name": entry["name"], "output": entry["output"]}
      for entry in kernel["proposals"]
      if entry.get("output", {}).get("constant")
    ]
    kernel["emitted_kernel_identity"] = _emitted_kernel_identity(kernel["proposals"])
    kernel["input_comparison"] = _input_comparison(
      kernel["proposals"], counted_inputs_sha256
    )
    report["kernels"].append(kernel)
  # The rebuilt counted case is the anchor control: same program, same
  # reference, chosen by the solver and compared exactly.
  rebuild = state.get("rebuild") or {}
  if rebuild.get("directory"):
    directory = Path(rebuild["directory"])
    result = (
      _read_json(directory / "result-compiled.json")
      if (directory / "result-compiled.json").is_file()
      else None
    )
    report["counted_control"] = {
      "directory": str(directory),
      "identity": rebuild,
      "execution": {
        "status": result.get("status") if result else "not_run",
        "comparison": result.get("comparison") if result else None,
        "campaign_valid": result.get("campaign_valid") if result else None,
      },
      "observations": loop_and_conversion_observations(directory) if result else None,
      "pass_dump": pass_dump_observations(directory) if result else None,
    }
  report["ledger"] = _provenance_ledger(report, state, args)
  report["summary"] = _summary(report)
  _write_json(Path(args.output), report)
  return report


def _provenance_ledger(report, state, args):
  """The commands that produced the recorded artifacts, gathered from them.

  Every reference replay is named by the case that carries its record, and the
  counted case's own build stages come from its attempt ledger.
  """
  replays = []
  for kernel in report["kernels"]:
    for entry in kernel["proposals"]:
      reference = entry.get("reference") or {}
      for replay in reference.get("replays") or []:
        replays.append(
          {
            "family": kernel["family"],
            "proposal": entry["name"],
            "reference_bits": replay["reference_bits"],
            "sha256": replay["sha256"],
            "matches_expected": replay["matches_expected"],
            "command": replay["command"],
          }
        )
  counted = {}
  counted_case = state.get("rebuild", {}).get("directory") or args.counted_case
  attempt = Path(counted_case) / "attempt.json"
  if attempt.is_file():
    counted = _read_json(attempt)
  return {
    "reference_replays": replays,
    "counted_case_stages": counted,
    "runner_command": (report.get("runner") or {}).get("command"),
    "driver_invocation": {
      "steps": args.steps,
      "families": args.families,
      "seed": args.seed,
      "shape": args.shape,
      "iterations": args.iterations,
      "operation_mode": args.operation_mode,
      "domain": args.domain,
      "pass_dump": args.pass_dump,
      "work": args.work,
    },
    "note": (
      "the driver's own stage ledger is a record of the same commands; this "
      "ledger is gathered from the artifacts each command left behind"
    ),
  }


def _summary(report):
  """A scannable index of outcomes, refusals, duplicates, and constants."""
  rows = []
  refusals = []
  duplicates = []
  constants = []
  not_run = []
  kernel_refusals = []
  for kernel in report["kernels"]:
    if kernel["admission"]["status"] != "admitted":
      kernel_refusals.append(
        {
          "family": kernel["family"],
          "reason": kernel["admission"].get("reason"),
          "domain_headroom": kernel["admission"].get("domain_headroom"),
          "proposals_built": 0,
        }
      )
    for entry in kernel["proposals"]:
      execution = entry.get("execution") or {}
      comparison = execution.get("comparison") or {}
      row = {
        "family": kernel["family"],
        "proposal": entry["name"],
        "admission": entry.get("admission"),
        "reference_kind": (entry.get("reference") or {}).get("kind"),
        "execution_status": execution.get("status"),
        "comparison_status": comparison.get("status"),
        "campaign_valid": execution.get("campaign_valid"),
        "bit_exact": (entry.get("bit_level_check") or {}).get("word_exact"),
        "output_constant": (entry.get("output") or {}).get("constant"),
      }
      rows.append(row)
      if row["admission"] not in {None, "admitted-exact-certificate"}:
        refusals.append({**row, "reason": entry.get("reason")})
      if execution.get("status") != "pass":
        not_run.append(row)
      if entry.get("duplicate_inputs", {}).get("proposal_groups"):
        duplicates.append(
          {
            "family": kernel["family"],
            "proposal": entry["name"],
            "groups": entry["duplicate_inputs"]["proposal_groups"],
          }
        )
      if (entry.get("output") or {}).get("constant"):
        constants.append(
          {
            "family": kernel["family"],
            "proposal": entry["name"],
            "output": entry["output"],
          }
        )
  return {
    "rows": rows,
    "kernel_refusals": kernel_refusals,
    "proposal_refusals": refusals,
    "not_pass": not_run,
    "duplicate_inputs": duplicates,
    "cross_kernel_duplicate_inputs": _cross_kernel_inputs(report),
    "constant_outputs": constants,
    "counted_control": (report.get("counted_control") or {}).get("execution"),
  }


def _cross_kernel_inputs(report):
  """Input bytes that more than one kernel's proposal carries.

  The asymmetric and sparse rules depend on the shape, domain, seed, and the
  region's reduction axes only, so two kernels of the same shape and domain
  derive the same bytes for them. That is a recorded coincidence, not an
  omission: the independent-small rule mixes in the source digest, so it
  differs per kernel.
  """
  groups = {}
  for kernel in report["kernels"]:
    for entry in kernel["proposals"]:
      digest = (entry.get("reference") or {}).get("input_bytes_sha256")
      if not digest:
        continue
      groups.setdefault(digest, []).append(
        {"family": kernel["family"], "proposal": entry["name"]}
      )
  return [
    {"sha256": digest, "proposals": entries}
    for digest, entries in sorted(groups.items())
    if len(entries) > 1
  ]


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--work", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--steps", default="build,run,report")
  parser.add_argument("--families", default=",".join(FAMILIES))
  parser.add_argument("--representative", default=REPRESENTATIVE)
  parser.add_argument("--seed", type=int, default=9)
  parser.add_argument("--shape", default="32,16")
  parser.add_argument("--iterations", type=int, default=4)
  parser.add_argument("--operation-mode", default="additive")
  parser.add_argument("--max-operators", type=int, default=10)
  parser.add_argument("--domain", default="0,8")
  parser.add_argument("--exporter", default=str(tensor_region_generation.EXPORTER))
  parser.add_argument("--symiri", default=str(tensor_case.DEFAULT_SYMIRI))
  parser.add_argument(
    "--symirc",
    default=str(os.environ.get("REFRACTIR_SYMIRC", TREE / "symirc")),
  )
  parser.add_argument(
    "--symirsolve",
    default=str(os.environ.get("REFRACTIR_SYMIRSOLVE", TREE / "symirsolve")),
  )
  parser.add_argument("--counted-case", required=True)
  parser.add_argument("--counted-manifest", required=True)
  parser.add_argument("--stage-timeout", type=float, default=120.0)
  parser.add_argument("--solver-timeout", type=float, default=1200.0)
  parser.add_argument("--run-timeout", type=float, default=300.0)
  parser.add_argument("--pass-dump", action="store_true")
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
  steps = [step.strip() for step in args.steps.split(",") if step.strip()]
  work = Path(args.work)
  work.mkdir(parents=True, exist_ok=True)
  state = {}
  # A later step run has to see what the earlier ones recorded: `report` reads
  # the rebuild and runner records, and those live in the work directory.
  state_path = work / "driver-state.json"
  if state_path.is_file():
    try:
      previous = _read_json(state_path)
    except (OSError, json.JSONDecodeError):
      previous = None
    if isinstance(previous, dict):
      state.update(previous)
  if "rebuild" in steps:
    state["rebuild"] = step_rebuild(work, args, state)
  if "build" in steps:
    state["build"] = step_build(work, args, state)
  if "run" in steps:
    state["run"] = step_run(work, args, state)
  if "report" in steps:
    step_report(work, args, state)
  _write_json(work / "driver-state.json", state)
  print(json.dumps({"work": str(work), "steps": steps, "status": "done"}))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
