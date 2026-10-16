"""Assemble the broadcast-hoist rematerialization record from raw records.

The summary is rebuilt from the build, run, and case records the device wrote
rather than from a summarized intermediate, so each arm keeps the identity of the
compiler that produced it, the environment it ran in, the qualification the case
builder recorded, the input, reference, and observed digests, and every refusal
the build recorded.

A refused arm is an arm: it carries the configuration it asked for, the reason the
certificate gave, and the steps that ran, with no execution record. An executed
arm carries the run's own identity even where that identity is *unavailable* - a
record that says so is evidence about the run, and a summary that drops the field
cannot make that claim at all.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from test.lib import tensor_compiler_observation

SCHEMA = "refractir.tensor-rematerialization-broadcast-hoist.v2"

_QUESTION = (
  "does the corrected rematerialization matcher confirm a recomputed value with "
  "redirected consumers on the new broadcast-hoist region family at the retained "
  "[32,16] loop configuration"
)

_PROBE_NOTE = (
  "hand-written Triton kernels in the family's operation set, compiled on the same "
  "device with pass dumps; supporting evidence for what the pass does with a "
  "surviving conversion, not campaign cases"
)

_STAGE_ARTIFACTS = (
  "compiler-ttir.ir",
  "compiler-ttgir.ir",
  "compiler-llir.ir",
  "compiler-ptx.ir",
)

_UNRECORDED = (None, "unavailable")

# The view a broadcast-hoist body leaves: a broadcast or expansion whose operand
# spine reaches a floating-point reduction through the layout-transparent ops the
# pipeline interleaves. MLIR prints either the custom or the quoted generic
# spelling, so both are read.
_FLOAT_TYPE = re.compile(r"f(?:16|32|64)\b")
_VIEW_OPERATION = re.compile(r'^"?(?:tt)\.(broadcast|expand_dims)"?[\s(]')
_REDUCTION_OPERATION = re.compile(r'^"?tt\.(?:reduce|sum)"?[\s(<]')
_SPINE_OPERATION = re.compile(
  r'^"?(?:tt\.(?:broadcast|expand_dims|reduce|sum)|ttg\.convert_layout)"?[\s(<]'
)
_VIEW_SPINE_DEPTH = 6


def consuming_operations(text, value):
  """The distinct operations that read a value as an operand.

  A definition is not its own consumer: the reading covers operand uses only, so
  a value nothing reads has none. One operation is one consumer however often it
  prints the value, which a generic op carrying its own region does whenever the
  operand is also named inside that region. The scan itself is the shared one
  (`tensor_compiler_observation._definitions_and_consumers`), which reads operand
  uses and never the defining left-hand side.
  """
  _definitions, consumers, _lines = (
    tensor_compiler_observation._definitions_and_consumers(text)
  )
  return tuple(sorted(set(consumers.get(value, ()))))


def broadcast_state_views(text):
  """The broadcasts of a reduced state a body leaves in the printed IR.

  The `broadcast-hoist` family's view is a broadcast of a one-extent reduction of
  the loop-carried state. Triton prints the reduction in generic form and the
  re-expansion as an `tt.expand_dims` of its one-extent result followed by a
  `tt.broadcast` back to the state shape, with layout conversions interleaved by
  the pipeline, so a view is identified by walking that operand spine down to the
  reduction rather than by taking whichever broadcast happens to be nearby. Only
  a broadcast or expansion of a floating-point value whose spine reaches a
  reduction counts.

  An IR whose spines never reach a reduction has no view, and that reading is
  reported as a status with its reason: a family body that lost its shape by this
  stage is a result about the stage rather than a parsing failure.
  """
  definitions, consumers, _lines = (
    tensor_compiler_observation._definitions_and_consumers(text)
  )
  views = []
  for name, definition in sorted(definitions.items()):
    operation = _VIEW_OPERATION.match(definition)
    if operation is None or not _FLOAT_TYPE.search(definition):
      continue
    operands = tensor_compiler_observation._SSA.findall(definition)
    if not operands:
      continue
    spine, reduction, reduction_definition = _reduction_spine(operands[0], definitions)
    if reduction is None:
      continue
    reads = tuple(sorted(set(consumers.get(name, ()))))
    views.append(
      {
        "value": name,
        "operation": operation[1],
        "definition": definition,
        "source": operands[0],
        "spine": list(spine),
        "reduction": reduction,
        "reduction_definition": reduction_definition,
        "consumers": reads,
        "consumer_count": len(reads),
      }
    )
  if views:
    return {"status": "view_present", "views": views}
  return {
    "status": "no_view",
    "views": [],
    "reason": (
      "no broadcast or expansion in this IR reaches a floating-point reduction "
      "over its operand spine"
    ),
  }


def _reduction_spine(value, definitions, depth=_VIEW_SPINE_DEPTH):
  """Walk a view's layout-transparent operand chain down to its reduction."""
  spine = []
  current = value
  for _step in range(depth):
    definition = definitions.get(current)
    if definition is None:
      return spine, None, ""
    spine.append(current)
    if _REDUCTION_OPERATION.match(definition):
      return spine, current, definition
    if not _SPINE_OPERATION.match(definition):
      return spine, None, ""
    operands = tensor_compiler_observation._SSA.findall(definition)
    if not operands:
      return spine, None, ""
    current = operands[0]
  return spine, None, ""


def _read(path):
  path = Path(path) if path else None
  if path is None or not path.is_file():
    return None
  return json.loads(path.read_text(encoding="utf-8"))


def _device_summary(environment):
  """Name the devices a run saw, without pretending to more than it recorded."""
  gpus = (environment or {}).get("gpus") or []
  if not gpus:
    return None
  first = gpus[0]
  capability = first.get("compute_capability")
  name = first.get("name")
  label = f"{name} (sm_{capability.replace('.', '')})" if capability else name
  return f"{label} x{len(gpus)}"


def _stage_digests(artifacts):
  """The compiler stage artifacts a run left, keyed by their own name."""
  return {
    name: artifacts[key]
    for key in sorted(artifacts)
    for name in _STAGE_ARTIFACTS
    if key.rsplit("/", 1)[-1] == name
  }


def _campaign_validity(run_metadata, environment):
  """Why the run's own validity rule holds or fails, field by field.

  `run_triton_tests.run_case` marks a run campaign-valid only when the
  environment, the RefractIR revision, the compiler flags and digest, the Triton
  source revision, and the package lock are all recorded. A summary that reports
  the verdict without the fields cannot say which requirement was missing.
  """
  metadata = run_metadata or {}
  compiler = metadata.get("compiler") or {}
  env = environment or {}
  missing = []
  if not env.get("campaign_available"):
    missing.append("device campaign availability")
  if metadata.get("refractir_revision") in _UNRECORDED:
    missing.append("RefractIR revision")
  if compiler.get("flags") in _UNRECORDED:
    missing.append("compiler flags")
  if not compiler.get("sha256"):
    missing.append("compiler digest")
  if env.get("triton_source_revision") in _UNRECORDED:
    missing.append("Triton source revision")
  if not env.get("package_lock_sha256"):
    missing.append("package lock digest")
  valid = not missing
  return valid, (None if valid else "unrecorded: " + ", ".join(missing))


def arm_record(case_id, build_entry, run_entry=None, case_dir=None, import_root=None):
  """One arm: what was built, what ran, what qualified it, in which environment."""
  record = {
    "case": case_id,
    "configuration": build_entry.get("configuration"),
    "selection": build_entry.get("selection", "solver"),
    "status": build_entry.get("status"),
    "build_reason": build_entry.get("reason"),
    "bounds": build_entry.get("bounds"),
    "region_pattern": build_entry.get("region_pattern"),
    "source_sha256": build_entry.get("source_sha256"),
    "program_sha256": build_entry.get("program_sha256"),
    "kernel_sha256": build_entry.get("kernel_sha256"),
    "run_status": None,
    "execution": None,
    "qualification": None,
    "digests": {},
    "provenance": {
      "import_root": import_root,
      "recorded_by": {"build": "test.lib.build_tensor_exact_region_case"},
      "refractir": None,
      "compiler": None,
      "triton": None,
      "environment": None,
      "campaign_valid": None,
      "campaign_valid_reason": None,
      "run_record": False,
    },
    "refusal": None,
  }
  if run_entry is None:
    if build_entry.get("status") != "built":
      record["refusal"] = {
        "status": build_entry.get("status"),
        "reason": build_entry.get("reason"),
        "steps": build_entry.get("steps"),
        "configuration": build_entry.get("configuration"),
      }
    return record

  report = run_entry.get("report") or {}
  case = _read(case_dir / "case.json") if case_dir else None
  metadata = _read(case_dir / "runs" / "compiled" / "run.json") if case_dir else None
  environment = (
    _read(case_dir / "runs" / "compiled" / "environment.json") if case_dir else None
  )
  case = case or {}
  metadata = metadata or {}
  environment = environment or {}
  files = case.get("files") or {}
  artifacts = report.get("artifact_digests") or {}
  campaign_valid, reason = _campaign_validity(metadata, environment)

  record["run_status"] = run_entry.get("status")
  record["execution"] = {
    "status": report.get("status", run_entry.get("status")),
    "mode": report.get("mode"),
    "exit_code": report.get("exit_code"),
    "comparison": report.get("comparison"),
    "campaign_valid": report.get("campaign_valid"),
    "started_at": metadata.get("started_at"),
  }
  record["digests"] = {
    "program": {"sha256": run_entry.get("program_sha256")},
    "kernel": {"sha256": run_entry.get("kernel_sha256")},
    "pass_dump": run_entry.get("pass_dump"),
    "inputs": files.get("inputs.bin") or artifacts.get("inputs.bin"),
    "reference": files.get("expected.bin") or artifacts.get("expected.bin"),
    "observed": {
      "sha256": report.get("actual_sha256"),
      "bytes": report.get("actual_bytes"),
    },
    "artifacts": _stage_digests(artifacts),
  }
  record["qualification"] = {
    "evidence": case.get("evidence"),
    "property": case.get("property"),
    "comparison": case.get("comparison"),
    "coverage": case.get("coverage"),
    "reference_defined": (case.get("reference") or {}).get("defined"),
    "exactness": (case.get("reference") or {}).get("exactness"),
    "source": case.get("source"),
    "inputs": case.get("inputs"),
    "output": case.get("output"),
  }
  record["provenance"].update(
    {
      "recorded_by": {
        "build": "test.lib.build_tensor_exact_region_case",
        "run": "test.lib.run_triton_tests.run_case",
      },
      "refractir": {
        "revision": metadata.get("refractir_revision"),
        "tree": import_root,
        "working_tree_dirty": metadata.get("working_tree_dirty"),
      },
      "compiler": metadata.get("compiler"),
      "triton": {
        "version": environment.get("triton_version"),
        "source_revision": environment.get("triton_source_revision"),
      },
      "environment": {
        "status": environment.get("status"),
        "device": _device_summary(environment),
        "driver": environment.get("driver_version"),
        "cuda": environment.get("cuda_version"),
        "python": environment.get("python_version"),
        "torch": environment.get("torch_version"),
        "gpus": len(environment.get("gpus") or []),
      },
      "campaign_valid": campaign_valid,
      "campaign_valid_reason": reason,
      "run_record": True,
    }
  )
  return record


def superseded_readings(previous, cases, provenance, matcher_boundary_probe=None):
  """Every value the previous record stated that this regeneration changes.

  A correction that cannot be compared with what it replaces is a rewrite, so
  each changed reading is listed with the value the previous record held: the
  consumer counts that counted a definition as its own consumer, the imported
  module digest that named another tree's copy, the synthetic boundary probe a
  later matcher reads differently, and the fields the previous summary dropped.
  """
  entries = []
  previous_cases = previous.get("cases") or {}
  for case_id, arm in sorted(cases.items()):
    previous_arm = previous_cases.get(case_id)
    if previous_arm is None:
      entries.append(
        {
          "path": f"cases[{case_id}]",
          "previous": None,
          "corrected": arm.get("status"),
          "reason": (
            "the previous summary held only the executed arms; every arm the build "
            "recorded is now present, refusals included"
          ),
        }
      )
      continue
    added = [
      key
      for key in ("provenance", "execution", "qualification", "digests")
      if arm.get(key) and not previous_arm.get(key)
    ]
    if added:
      entries.append(
        {
          "path": f"cases[{case_id}]",
          "previous": "absent",
          "corrected": sorted(added),
          "reason": (
            "the previous summary kept run_status and dropped the run's identity, "
            "the qualification record, and the dataset digests"
          ),
        }
      )
    entries.extend(_consumer_corrections(case_id, previous_arm, arm))
  for path, previous_digest in sorted((previous.get("module_digests") or {}).items()):
    corrected = (provenance.get("imported_modules") or {}).get("sha256", {}).get(path)
    if corrected and corrected != previous_digest:
      entries.append(
        {
          "path": f"imported_modules[{path}]",
          "previous": previous_digest,
          "corrected": corrected,
          "reason": (
            "the previous digest named the red-run tree's copy of the module, whose "
            "only difference from the imported copy is a formatting reflow"
          ),
        }
      )
  entries.extend(_probe_corrections(previous, matcher_boundary_probe))
  return entries


def _probe_corrections(previous, matcher_boundary_probe):
  """The synthetic boundary probe's readings that moved under the new matcher."""
  entries = []
  previous_probe = previous.get("matcher_boundary_probe") or {}
  for variant in ("same_encoding_copy", "layout_rematerialization"):
    old = previous_probe.get(variant)
    new = (matcher_boundary_probe or {}).get(variant)
    if not isinstance(old, dict) or not isinstance(new, dict):
      continue
    for field in ("status", "confirmed_count", "unattributed_count"):
      if old.get(field) == new.get(field):
        continue
      entries.append(
        {
          "path": f"matcher_boundary_probe[{variant}].{field}",
          "previous": old.get(field),
          "corrected": new.get(field),
          "reason": (
            "the probe was re-read with this library, which compares the canonical "
            "expression without regard to the encoding the copy was created to "
            "change; the previous reading came from the library the runs imported"
          ),
        }
      )
  return entries


_MATCHER_FIELDS = (
  "matcher",
  "structural_recomputations",
  "layout_conversion_delta",
  "duplicate_expression_group_delta",
  "duplicate_operation_deltas",
  "loop_construct_delta",
)


def matcher_drift(previous, cases):
  """What re-reading the device dumps moved, field by field against the record.

  A reader has to be able to tell "the device readings are unchanged under the new
  matcher" from "the record was regenerated and nobody compared". Every arm the
  two records share is named in `arms_compared`, and each matcher-derived field
  either matches or is listed with both values.
  """
  compared = []
  changed = []
  previous_cases = previous.get("cases") or {}
  for case_id, arm in sorted(cases.items()):
    transitions = (arm.get("analysis") or {}).get("transitions")
    previous_arm = previous_cases.get(case_id) or {}
    previous_transitions = (previous_arm.get("analysis") or {}).get("transitions")
    if transitions is None or previous_transitions is None:
      continue
    compared.append(case_id)
    for index, transition in enumerate(transitions):
      if index >= len(previous_transitions):
        continue
      old = previous_transitions[index]
      for field in _MATCHER_FIELDS:
        if field not in old and field not in transition:
          continue
        if old.get(field) != transition.get(field):
          changed.append(
            {
              "case": case_id,
              "transition": transition.get("index"),
              "field": field,
              "previous": old.get(field),
              "corrected": transition.get(field),
            }
          )
  return {"arms_compared": compared, "changed": changed}


def _consumer_corrections(case_id, previous_arm, arm):
  """The opportunity counts that moved from uses to consuming operations."""
  entries = []
  previous_transitions = (previous_arm.get("analysis") or {}).get("transitions") or []
  for index, transition in enumerate(
    (arm.get("analysis") or {}).get("transitions") or []
  ):
    if index >= len(previous_transitions):
      continue
    previous_by_conversion = {
      entry.get("conversion"): entry
      for entry in previous_transitions[index].get("opportunities_before") or []
    }
    for entry in transition.get("opportunities_before") or []:
      old = previous_by_conversion.get(entry.get("conversion"))
      if old is None or old.get("source_consumers") == entry.get(
        "source_consumer_count"
      ):
        continue
      entries.append(
        {
          "path": (
            f"cases[{case_id}].analysis.transitions[{transition.get('index')}]"
            f".opportunities_before[{entry.get('conversion')}].source_consumers"
          ),
          "previous": old.get("source_consumers"),
          "corrected": entry.get("source_consumer_count"),
          "reason": (
            "the previous count scanned every SSA name on the line, so a value's "
            "own definition counted as one of its consumers"
          ),
        }
      )
  return entries


def assemble(
  *,
  analysis,
  build,
  runs,
  case_dirs,
  frozen_corpus,
  probes,
  provenance,
  previous=None,
):
  """The deliverable record: one arm per build record, each with its evidence."""
  import_root = provenance.get("import_root")
  cases = {}
  for case_id, build_entry in sorted(build.items()):
    run_entry = runs.get(case_id)
    arm = arm_record(
      case_id, build_entry, run_entry, case_dirs.get(case_id), import_root
    )
    analysed = (analysis.get("cases") or {}).get(case_id) or {}
    if analysed:
      arm["source_pattern"] = analysed.get("source_pattern")
      arm["reduction_axes"] = analysed.get("reduction_axes")
      arm["pass_dump_sha256"] = analysed.get("pass_dump_sha256")
      arm["analysis"] = analysed.get("analysis")
    arm["manifest"] = {
      "build": build_entry.get("manifest"),
      "run": (run_entry or {}).get("manifest"),
    }
    cases[case_id] = arm
  record = {
    "schema": SCHEMA,
    "question": _QUESTION,
    "matcher": analysis.get("matcher"),
    "structural_reading": analysis.get("structural_reading"),
    "interpretation_limit": analysis.get("interpretation_limit"),
    "cases": cases,
    "matcher_boundary_probe": analysis.get("matcher_boundary_probe"),
    "probe_kernels": {"note": _PROBE_NOTE, "kernels": probes},
    "frozen_corpus": frozen_corpus,
    "provenance": provenance,
    "superseded": (
      superseded_readings(
        previous, cases, provenance, analysis.get("matcher_boundary_probe")
      )
      if previous
      else []
    ),
  }
  if previous:
    record["matcher_rederivation"] = matcher_drift(previous, cases)
  return record
