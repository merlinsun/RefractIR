#!/usr/bin/env python3
"""Execute the predeclared tensor-v1/v2 Triton comparison protocols."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CORRECTED_TRITON = Path("/root/triton-latest/python")
PTXAS = Path(
  "/root/.triton/.triton/nvidia/nvcc/cuda_nvcc-linux-x86_64-12.8.93-archive/bin/ptxas"
)
REFRACTIR_REVISION = "7b43faa83e76d5ab17a8ec3fb62634746e28d104"
COMPILER_FLAGS = (
  "g++ -std=c++20 -Wall -Wextra -O2 -DREFRACTIR_TENSOR_REF "
  "-DUSE_BITWUZLA -I/tmp/boost_1_89"
)
TIME_FIELDS = (
  "generation_s",
  "selection_s",
  "input_materialization_s",
  "reference_validation_s",
  "compilation_s",
  "gpu_execution_s",
)
RUN_TIMEOUT_S = 60
GENERATION_TIMEOUT_S = 60
SELECTION_TIMEOUT_S = 120
FIXED_TIME_START_RESERVE_S = 300


def _json_write(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(
    json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    encoding="utf-8",
  )
  os.replace(temporary, path)


def _sha256(data):
  return hashlib.sha256(data).hexdigest()


def _sha256_file(path):
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _now_iso():
  return datetime.now(timezone.utc).isoformat()


def _init_environment(config):
  env = config["environment"]
  pins = config["refractir"]
  os.environ["REFRACTIR_SYMIRI"] = str(ROOT / "symiri")
  os.environ["REFRACTIR_SYMIRC"] = str(ROOT / "symirc")
  os.environ["REFRACTIR_REVISION"] = REFRACTIR_REVISION
  os.environ["REFRACTIR_COMPILER_FLAGS"] = COMPILER_FLAGS
  os.environ["REFRACTIR_WORKTREE_DIRTY"] = "true"
  os.environ["TRITON_PTXAS_PATH"] = str(PTXAS)
  pythonpath = [str(ROOT), str(CORRECTED_TRITON)]
  pythonpath.extend(os.environ.get("PYTHONPATH", "").split(os.pathsep))
  os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(x for x in pythonpath if x))
  sys.path.insert(0, str(ROOT))
  sys.path.insert(1, str(CORRECTED_TRITON))
  if env["triton_revision"] != "e2e356723ce35f46b5144373b145c55c4d712237":
    raise ValueError("configuration does not pin the corrected Triton build")
  if not Path(os.environ["REFRACTIR_SYMIRI"]).is_file():
    raise FileNotFoundError(os.environ["REFRACTIR_SYMIRI"])
  if not Path(os.environ["REFRACTIR_SYMIRC"]).is_file():
    raise FileNotFoundError(os.environ["REFRACTIR_SYMIRC"])
  if not PTXAS.is_file():
    raise FileNotFoundError(PTXAS)
  paths = {
    "rysmith_sha256": ROOT / "rysmith",
    "symiri_sha256": ROOT / "symiri",
    "symirc_sha256": ROOT / "symirc",
    "tensor_gen_cpp_sha256": ROOT / "src/reify/tensor_gen.cpp",
    "tensor_directed_py_sha256": ROOT / "test/lib/tensor_directed.py",
    "tensor_case_py_sha256": ROOT / "test/lib/tensor_case.py",
    "tensor_coverage_py_sha256": ROOT / "test/lib/tensor_coverage.py",
    "probe_triton_env_py_sha256": ROOT / "test/lib/probe_triton_env.py",
    "tensor_evaluation_py_sha256": ROOT / "test/lib/tensor_evaluation.py",
    "run_triton_tests_py_sha256": ROOT / "test/lib/run_triton_tests.py",
    "triton_case_worker_py_sha256": ROOT / "test/lib/triton_case_worker.py",
    "campaign_driver_py_sha256": Path(__file__).resolve(),
  }
  for pin, path in paths.items():
    if _sha256_file(path) != pins.get(pin):
      raise ValueError(f"pinned input differs from configuration: {pin}")
  if _sha256_file(PTXAS) != env["ptxas_sha256"]:
    raise ValueError("pinned ptxas binary differs from configuration")


class TimingHooks:
  """Attribute reference, selection, code emission, and worker stage costs."""

  def __init__(self, case_runner, tensor_case, tensor_directed):
    self.runner = case_runner
    self.tensor_case = tensor_case
    self.tensor_directed = tensor_directed
    self.active_run = None
    self.active_selection = None
    self.originals = {}

  def install(self, pinned_environment):
    self.originals["load_case"] = self.tensor_case.load_case
    self.originals["run_reference"] = self.tensor_directed._run_reference
    self.originals["prepare_case_arm"] = self.tensor_directed.prepare_case_arm
    self.originals["emit_kernel"] = self.runner._emit_kernel
    self.originals["collect_environment"] = (
      self.runner.probe_triton_env.collect_environment
    )

    def timed_load_case(*args, **kwargs):
      metrics = self.active_selection or self.active_run
      started = time.monotonic()
      try:
        return self.originals["load_case"](*args, **kwargs)
      finally:
        if metrics is not None:
          metrics["reference_validation_s"] += time.monotonic() - started

    def timed_reference(*args, **kwargs):
      metrics = self.active_selection
      started = time.monotonic()
      try:
        return self.originals["run_reference"](*args, **kwargs)
      finally:
        if metrics is not None:
          metrics["reference_validation_s"] += time.monotonic() - started

    def timed_prepare_case_arm(case_path, *, arm, **kwargs):
      metrics = self.selection_metrics[arm]
      reference_before = metrics["reference_validation_s"]
      self.active_selection = metrics
      started = time.monotonic()
      try:
        return self.originals["prepare_case_arm"](case_path, arm=arm, **kwargs)
      finally:
        elapsed = time.monotonic() - started
        reference_time = metrics["reference_validation_s"] - reference_before
        metrics["selection_s"] += max(0.0, elapsed - reference_time)
        self.active_selection = None

    def timed_emit(*args, **kwargs):
      metrics = self.active_run
      started = time.monotonic()
      try:
        return self.originals["emit_kernel"](*args, **kwargs)
      finally:
        if metrics is not None:
          metrics["symirc_emission_s"] += time.monotonic() - started

    def collect_pinned_environment():
      record = self.originals["collect_environment"]()
      expected_values = {
        "triton_source_revision": pinned_environment["triton_revision"],
        "triton_version": pinned_environment["triton_version"],
        "package_lock_sha256": pinned_environment["package_lock_sha256"],
        "python_version": pinned_environment["python"],
        "torch_version": pinned_environment["pytorch"],
        "cuda_version": pinned_environment["cuda"],
      }
      errors = [
        f"{field}={record.get(field)!r} does not match {expected!r}"
        for field, expected in expected_values.items()
        if record.get(field) != expected
      ]
      gpus = record.get("gpus", [])
      if len(gpus) != pinned_environment.get("gpu_count", len(gpus)):
        errors.append(f"GPU count {len(gpus)} differs from the frozen environment")
      for index, gpu in enumerate(gpus):
        for field, expected in (
          ("name", pinned_environment["device_class"]),
          ("compute_capability", pinned_environment["compute_capability"]),
          ("driver_version", pinned_environment["driver"]),
        ):
          if gpu.get(field) != expected:
            errors.append(
              f"GPU {index} {field}={gpu.get(field)!r} differs from {expected!r}"
            )
      if errors:
        record["campaign_available"] = False
        record["status"] = "environment_unavailable"
        record["failure_reasons"] = list(record.get("failure_reasons", [])) + errors
      return record

    self.tensor_case.load_case = timed_load_case
    self.tensor_directed._run_reference = timed_reference
    self.tensor_directed.prepare_case_arm = timed_prepare_case_arm
    self.runner._emit_kernel = timed_emit
    self.runner.probe_triton_env.collect_environment = collect_pinned_environment

  def restore(self):
    if not self.originals:
      return
    self.tensor_case.load_case = self.originals["load_case"]
    self.tensor_directed._run_reference = self.originals["run_reference"]
    self.tensor_directed.prepare_case_arm = self.originals["prepare_case_arm"]
    self.runner._emit_kernel = self.originals["emit_kernel"]
    self.runner.probe_triton_env.collect_environment = self.originals[
      "collect_environment"
    ]

  def begin_selection(self, metrics_by_arm):
    self.selection_metrics = metrics_by_arm

  def run_case(self, case_path, metrics, *, device_id):
    reference_before = metrics["reference_validation_s"]
    emission_before = metrics["symirc_emission_s"]
    started = time.monotonic()
    self.active_run = metrics
    try:
      report = self.runner.run_case(
        case_path,
        "compiled",
        timeout_s=RUN_TIMEOUT_S,
        device_id=device_id,
      )
    finally:
      self.active_run = None
    elapsed = time.monotonic() - started
    artifact_dir = Path(case_path) / "runs" / "compiled"
    timing_path = artifact_dir / "worker-timing.json"
    worker_timing = {}
    if timing_path.is_file():
      worker_timing = json.loads(timing_path.read_text(encoding="utf-8"))
    reference_time = metrics["reference_validation_s"] - reference_before
    emission_time = metrics["symirc_emission_s"] - emission_before
    worker_compile = float(worker_timing.get("compilation_s", 0.0))
    worker_input = float(worker_timing.get("input_staging_s", 0.0))
    worker_gpu = float(worker_timing.get("gpu_execution_s", 0.0))
    metrics["input_materialization_s"] += worker_input
    metrics["compilation_s"] += emission_time + worker_compile
    metrics["gpu_execution_s"] += worker_gpu
    metrics["worker_output_materialization_s"] += float(
      worker_timing.get("output_materialization_s", 0.0)
    )
    known = (
      reference_time
      + emission_time
      + worker_input
      + worker_compile
      + worker_gpu
      + float(worker_timing.get("output_materialization_s", 0.0))
    )
    metrics["runner_overhead_s"] += max(0.0, elapsed - known)
    metrics["case_runner_wall_s"] += elapsed
    report["worker_timing"] = worker_timing
    return report


def _empty_metrics():
  return {
    **{field: 0.0 for field in TIME_FIELDS},
    "symirc_emission_s": 0.0,
    "worker_output_materialization_s": 0.0,
    "runner_overhead_s": 0.0,
    "case_runner_wall_s": 0.0,
  }


def _generate_case(seed, out_dir, *, version, dtype, property_name, stratum):
  out_dir = Path(out_dir)
  command = [
    str(ROOT / "rysmith"),
    "--profile",
    "tensor",
    "--target",
    "sir",
    "--tensor-profile-version",
    version,
    "--tensor-dtype",
    dtype,
    "--tensor-synthesis",
    "construction",
    "--max-retries",
    "100",
    "--seed",
    str(seed),
    "--output-dir",
    str(out_dir),
  ]
  if property_name:
    command.extend(["--tensor-property", property_name])
  if stratum:
    command.extend(["--tensor-stratum", stratum])
  started = time.monotonic()
  try:
    result = subprocess.run(
      command,
      cwd=ROOT,
      capture_output=True,
      text=True,
      timeout=GENERATION_TIMEOUT_S,
      check=False,
    )
    status = "complete" if result.returncode == 0 else "generation_rejected"
    stdout, stderr = result.stdout, result.stderr
    returncode = result.returncode
  except subprocess.TimeoutExpired as error:
    status, stdout, stderr, returncode = (
      "generation_timeout",
      error.stdout or "",
      error.stderr or "",
      124,
    )
  generation_s = time.monotonic() - started
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "generation.stdout.log").write_text(stdout, encoding="utf-8")
  (out_dir / "generation.stderr.log").write_text(stderr, encoding="utf-8")
  summary_path = out_dir / "generation.json"
  generation = (
    json.loads(summary_path.read_text(encoding="utf-8"))
    if summary_path.is_file()
    else {}
  )
  case_path = out_dir / "case_000000"
  accepted = (
    status == "complete"
    and generation.get("status") == "complete"
    and generation.get("emitted") == 1
    and case_path.is_dir()
  )
  return {
    "case_path": case_path if accepted else None,
    "generation_s": generation_s,
    "status": "accepted" if accepted else status,
    "exit_code": returncode,
    "generation": generation,
    "command": command,
  }


def _selection_timeout(seconds):
  def expired(_signum, _frame):
    raise TimeoutError(f"input selection exceeded {seconds} seconds")

  previous = signal.signal(signal.SIGALRM, expired)
  signal.setitimer(signal.ITIMER_REAL, seconds)
  return previous


def _run_selection(function, *, timeout_s=SELECTION_TIMEOUT_S):
  previous = _selection_timeout(timeout_s)
  try:
    return function()
  finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, previous)


def _load_manifest(case_path):
  return json.loads((Path(case_path) / "case.json").read_text(encoding="utf-8"))


def _case_record(case_path, *, arm, seed, status, metrics, selection=None, report=None):
  manifest = _load_manifest(case_path) if case_path and Path(case_path).is_dir() else {}
  source = (
    (Path(case_path) / "program.sir").read_text(encoding="utf-8") if manifest else ""
  )
  selection = selection or {}
  record = {
    "arm": arm,
    "seed": seed,
    "trial": str(seed),
    "status": status,
    "completed_target_runs": int(
      status in {"pass", "exact_mismatch", "numeric_mismatch"}
    ),
    "source_sha256": _sha256(source.encode("utf-8")) if source else None,
    "source_signature": None,
    "source_operators": re.findall(r"\btensor\s+([A-Za-z_][A-Za-z_0-9]*)", source),
    "inputs_sha256": _sha256_file(Path(case_path) / "inputs.bin") if manifest else None,
    "shape": manifest.get("output", {}).get("shape"),
    "layout": {
      "inputs": [item.get("element_strides") for item in manifest.get("inputs", [])],
      "output": manifest.get("output", {}).get("element_strides"),
    },
    "num_warps": manifest.get("launch", {}).get("num_warps"),
    "primary_stratum": manifest.get("generator", {}).get("primary_stratum"),
    "property_passed": selection.get("property_passed"),
    "distinguished": selection.get("distinguished"),
    "P": selection.get("P"),
    "D": selection.get("D"),
    "target_model": selection.get("target_model"),
    "objective_applicable": selection.get("objective_applicable"),
    "input_domain_id": selection.get("input_domain_id"),
    "proposal_mechanism_id": selection.get("proposal_mechanism_id"),
    "seed_schedule_id": selection.get("seed_schedule_id"),
    "proposal_budget": selection.get("proposal_budget"),
    "proposal_count": selection.get("proposal_count"),
    "selection_status": selection.get("status"),
    "selection_cost": selection.get("cost", {}),
    "stages": {field: metrics.get(field, 0.0) for field in TIME_FIELDS},
    "worker_output_materialization_s": metrics.get(
      "worker_output_materialization_s", 0.0
    ),
    "runner_overhead_s": metrics.get("runner_overhead_s", 0.0),
  }
  if manifest:
    from test.lib import tensor_coverage

    record["source_signature"] = tensor_coverage.source_signature(source)
    record["case_directory"] = str(Path(case_path).resolve())
  if report is not None:
    record["runner_status"] = report.get("status")
    record["campaign_valid"] = report.get("campaign_valid")
    record["result_artifacts"] = report.get("artifacts", [])
    record["worker_timing"] = report.get("worker_timing", {})
    record["reason"] = report.get("reason")
  return record


def _selection_outcome_status(arm, selection):
  """Keep inapplicability distinct from a valid directed-search shortfall."""
  status = selection.get("status")
  if status in {
    "accepted",
    "no_applicable_model",
    "model_invalid_for_proposals",
    "property_exhausted",
  }:
    return status
  if arm == "directed" and selection.get("property_passed") is True:
    return "directed_shortfall"
  return "selection_rejected"


def _validate_model_objective_pairs(config):
  """Require each ordinary/directed protocol pair to test the same D."""
  for protocol_name in ("fixed_count", "fixed_time"):
    protocol = config.get(protocol_name)
    arms = protocol.get("arms", {}) if isinstance(protocol, dict) else {}
    selections = {}
    for arm_name, arm in arms.items():
      if not isinstance(arm, dict) or arm.get("profile") != "tensor-v2":
        continue
      selection = arm.get("selection")
      if not isinstance(selection, dict):
        continue
      kind = selection.get("arm")
      if kind in {"ordinary", "directed"}:
        selections.setdefault(kind, []).append((arm_name, selection))
    if not selections:
      continue
    if set(selections) != {"ordinary", "directed"}:
      raise ValueError(f"{protocol_name} must configure both P/D arms")
    ordinary_entries, directed_entries = selections["ordinary"], selections["directed"]
    if len(ordinary_entries) != len(directed_entries):
      raise ValueError(f"{protocol_name} must pair P/D objectives one-to-one")
    directed_by_name = dict(directed_entries)
    for ordinary_name, ordinary in ordinary_entries:
      directed_name = ordinary_name.replace("ordinary", "directed")
      if directed_name not in directed_by_name:
        if len(directed_entries) != 1:
          raise ValueError(f"{protocol_name} cannot match the P/D arm names")
        directed = directed_entries[0][1]
      else:
        directed = directed_by_name[directed_name]
      ordinary_config = arms[ordinary_name]
      directed_config = (
        arms[directed_name] if directed_name in arms else arms[directed_entries[0][0]]
      )
      for field in (
        "profile",
        "stratum",
        "dtype",
        "synthesis",
        "property",
        "graph_seed_schedule",
        "seed_schedule",
      ):
        if ordinary_config.get(field) != directed_config.get(field):
          raise ValueError(f"{protocol_name} paired arms disagree on {field}")
      if ordinary.get("target_model") != directed.get("target_model"):
        raise ValueError(f"{protocol_name} paired arms must use the same target model")
      for field in (
        "P",
        "D",
        "domain",
        "proposal_mechanism",
        "proposal_budget",
      ):
        if ordinary.get(field) != directed.get(field):
          raise ValueError(f"{protocol_name} paired arms disagree on {field}")


def _cache_record(case_path, *, arm, seed):
  path = Path(case_path) / "runs" / "compiled" / "run.json"
  if not path.is_file():
    return None
  run = json.loads(path.read_text(encoding="utf-8"))
  cache = run.get("compiler_cache")
  if not isinstance(cache, dict):
    return None
  return {
    "arm": arm,
    "trial": str(seed),
    "cache_id": cache.get("cache_id"),
    "cache_empty_at_start": cache.get("empty_at_start"),
    "path": cache.get("path"),
  }


def _execute_one(hooks, case_path, metrics, *, device_id):
  return hooks.run_case(case_path, metrics, device_id=device_id)


def _write_progress(ledger_path, summary_path, ledger, summarizer):
  _json_write(ledger_path, ledger)
  try:
    summary = summarizer(ledger)
  except (ValueError, KeyError) as error:
    summary = {"status": "summary_pending", "reason": str(error)}
  _json_write(summary_path, summary)


def _make_generated_record(arm, seed, generated):
  return {
    "arm": arm,
    "seed": seed,
    "generation_status": generated["status"],
    "generation_s": generated["generation_s"],
    "generation": generated.get("generation", {}),
    "command": generated.get("command"),
  }


def run_fixed_count(
  config,
  output,
  hooks,
  tensor_case,
  tensor_coverage,
  tensor_directed,
  tensor_evaluation,
):
  protocol = config["fixed_count"]
  root = Path(output) / "fixed-count"
  root.mkdir(parents=True, exist_ok=True)
  ledger_path, summary_path = root / "ledger.json", root / "summary.json"
  target = protocol["target_completed_runs_per_arm"]
  arms = protocol["arms"]
  ledger = {
    "schema": "refractir.tensor-matched-campaign-ledger.v1",
    "protocol": "fixed-count",
    "status": "in_progress",
    "target_count": target,
    "configuration": str(Path(output) / "manifest.json"),
    "started_at": _now_iso(),
    "records": [],
    "cases": [],
    "cache_records": [],
    "failures": [],
  }
  counts = {name: 0 for name in arms}
  case_index = 0
  for arm_name, arm in arms.items():
    if arm_name == "tensor-v2-compositional-directed":
      continue
    schedule = arm.get("graph_seed_schedule", arm.get("seed_schedule"))
    start = schedule["start"]
    count = schedule["count"]
    for offset in range(count):
      seed = start + offset
      attempt = root / "cases" / arm_name / f"seed_{seed}"
      generated = _generate_case(
        seed,
        attempt / "generated",
        version=arm["profile"].removeprefix("tensor-"),
        dtype=arm["dtype"],
        property_name=arm.get("property"),
        stratum=arm.get("stratum"),
      )
      attempt_record = _make_generated_record(arm_name, seed, generated)
      if generated["case_path"] is None:
        ledger["cases"].append({**attempt_record, "status": generated["status"]})
        _write_progress(
          ledger_path, summary_path, ledger, tensor_evaluation.summarize_ledger
        )
        continue
      generated_case = generated["case_path"]
      if arm["profile"] == "tensor-v2":
        pair_device_id = offset % len(protocol["device_pool"])
        pair_path = attempt / "pair"
        metrics_by_arm = {name: _empty_metrics() for name in ("ordinary", "directed")}
        for metrics in metrics_by_arm.values():
          metrics["generation_s"] = generated["generation_s"] / 2
        hooks.begin_selection(metrics_by_arm)
        selection_started = time.monotonic()
        try:
          pair = _run_selection(
            lambda: tensor_evaluation.prepare_matched_pair(
              generated_case,
              pair_path,
              seed=seed,
              proposal_budget=arm["selection"]["proposal_budget"],
              compiler_revision=protocol["compiler_revision"],
              symiri=tensor_case.DEFAULT_SYMIRI,
              target_model=arm["selection"].get("target_model"),
            )
          )
        except Exception as error:
          failure = {
            "arm": arm_name,
            "seed": seed,
            "stage": "selection",
            "reason": str(error),
          }
          ledger["failures"].append(failure)
          ledger["cases"].append(
            {**attempt_record, "status": "selection_failure", "reason": str(error)}
          )
          _write_progress(
            ledger_path, summary_path, ledger, tensor_evaluation.summarize_ledger
          )
          break
        pair_s = time.monotonic() - selection_started
        per_arm = {item["arm"]: item for item in pair["arms"]}
        accepted_any = False
        for local_index, local_arm in enumerate(("ordinary", "directed")):
          selected = per_arm[local_arm]
          case_path = pair_path / local_arm
          metrics = metrics_by_arm[local_arm]
          if local_index == 0:
            # Pair setup outside the two arm selectors is shared graph overhead.
            metrics["selection_s"] += max(
              0.0,
              pair_s
              - sum(
                item["selection_s"] + item["reference_validation_s"]
                for item in metrics_by_arm.values()
              ),
            )
          eligible = (
            selected.get("status") == "accepted"
            and selected.get("property_passed") is True
            and (local_arm != "directed" or selected.get("distinguished") is True)
          )
          status = _selection_outcome_status(local_arm, selected)
          report = None
          if eligible:
            accepted_any = True
            report = _execute_one(
              hooks,
              case_path,
              metrics,
              device_id=pair_device_id,
            )
            status = report.get("status", "runner_failure")
            cache = _cache_record(
              case_path,
              arm=f"tensor-v2-compositional-{local_arm}",
              seed=seed,
            )
            if cache:
              ledger["cache_records"].append(cache)
            if status in {"pass", "exact_mismatch", "numeric_mismatch"}:
              counts[
                arm_name
                if arm_name.startswith("tensor-v1")
                else f"tensor-v2-compositional-{local_arm}"
              ] += 1
            if status != "pass" or report.get("campaign_valid") is not True:
              ledger["failures"].append(
                {
                  "arm": local_arm,
                  "seed": seed,
                  "stage": "compiled-run",
                  "status": status,
                  "reason": report.get("reason"),
                }
              )
          case_record = _case_record(
            case_path,
            arm=(
              arm_name
              if arm_name.startswith("tensor-v1")
              else f"tensor-v2-compositional-{local_arm}"
            ),
            seed=seed,
            status=status,
            metrics=metrics,
            selection=selected,
            report=report,
          )
          ledger["cases"].append({**attempt_record, **case_record})
          if report is not None:
            case_index += 1
          _write_progress(
            ledger_path, summary_path, ledger, tensor_evaluation.summarize_ledger
          )
          if report is not None and (
            status != "pass" or report.get("campaign_valid") is not True
          ):
            break
        if ledger["failures"]:
          break
        if not accepted_any:
          ledger["cases"].append({**attempt_record, "status": "both_arms_shortfall"})
      else:
        metrics = _empty_metrics()
        report = _execute_one(
          hooks,
          generated_case,
          metrics,
          device_id=case_index % len(protocol["device_pool"]),
        )
        status = report.get("status", "runner_failure")
        counts[arm_name] += int(
          status in {"pass", "exact_mismatch", "numeric_mismatch"}
        )
        cache = _cache_record(generated_case, arm=arm_name, seed=seed)
        if cache:
          ledger["cache_records"].append(cache)
        case_record = _case_record(
          generated_case,
          arm=arm_name,
          seed=seed,
          status=status,
          metrics=metrics,
          report=report,
        )
        ledger["cases"].append({**attempt_record, **case_record})
        case_index += 1
        if status != "pass" or report.get("campaign_valid") is not True:
          ledger["failures"].append(
            {
              "arm": arm_name,
              "seed": seed,
              "stage": "compiled-run",
              "status": status,
              "reason": report.get("reason"),
            }
          )
        _write_progress(
          ledger_path, summary_path, ledger, tensor_evaluation.summarize_ledger
        )
        if ledger["failures"]:
          break
    if ledger["failures"]:
      break
  ledger["records"] = [
    {"arm": arm, "completed_target_runs": count} for arm, count in counts.items()
  ]
  ledger["finished_at"] = _now_iso()
  ledger["status"] = "stopped_on_failure" if ledger["failures"] else "finished"
  case_dirs = [
    Path(record["case_directory"])
    for record in ledger["cases"]
    if record.get("case_directory") and Path(record["case_directory"]).is_dir()
  ]
  if case_dirs:
    ledger["coverage"] = tensor_coverage.index_cases(case_dirs)
  _write_progress(ledger_path, summary_path, ledger, tensor_evaluation.summarize_ledger)
  return ledger


def _fixed_time_arm(config, arm_name, arm, root, hooks, tensor_case, tensor_directed):
  arm_started = time.monotonic()
  deadline = arm_started + config["wall_clock_budget_s"]
  records, cases, caches = [], [], []
  metrics_total = {field: 0.0 for field in TIME_FIELDS}
  metrics_other = {
    field: 0.0 for field in ("worker_output_materialization_s", "runner_overhead_s")
  }
  seed_schedule = arm.get("graph_seed_schedule", arm.get("seed_schedule"))
  seed = seed_schedule["start"]
  completed = 0
  failure = None
  case_index = 0
  progress_path = root / "progress" / f"{arm_name}.jsonl"

  def append_progress(record):
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as stream:
      stream.write(json.dumps(record, sort_keys=True) + "\n")
      stream.flush()
      os.fsync(stream.fileno())

  def accumulate(metrics):
    for field in TIME_FIELDS:
      metrics_total[field] += metrics[field]
    for field in metrics_other:
      metrics_other[field] += metrics[field]

  while time.monotonic() < deadline - FIXED_TIME_START_RESERVE_S:
    case_root = root / "cases" / arm_name / f"seed_{seed}"
    generated = _generate_case(
      seed,
      case_root / "generated",
      version=arm["profile"].removeprefix("tensor-"),
      dtype=arm["dtype"],
      property_name=arm.get("property"),
      stratum=arm.get("stratum"),
    )
    metrics = _empty_metrics()
    metrics["generation_s"] = generated["generation_s"]
    base_record = _make_generated_record(arm_name, seed, generated)
    if generated["case_path"] is None:
      candidate = {
        **base_record,
        "status": generated["status"],
        "stages": {field: metrics[field] for field in TIME_FIELDS},
      }
      cases.append(candidate)
      records.append({"seed": seed, "status": generated["status"]})
      accumulate(metrics)
      append_progress(candidate)
      print(
        f"fixed-time arm={arm_name} seed={seed} status={generated['status']} "
        f"elapsed={time.monotonic() - arm_started:.1f}s",
        flush=True,
      )
      seed += 1
      continue
    case_path = generated["case_path"]
    selection = None
    if arm["profile"] == "tensor-v2":
      hooks.begin_selection({arm["selection"]["arm"]: metrics})
      try:
        selection = _run_selection(
          lambda: tensor_directed.prepare_case_arm(
            case_path,
            arm=arm["selection"]["arm"],
            seed=seed,
            proposal_limit=arm["selection"]["proposal_budget"],
            symiri=tensor_case.DEFAULT_SYMIRI,
            target_model=arm["selection"].get("target_model"),
          )
        )
      except Exception as error:
        failure = {"seed": seed, "stage": "selection", "reason": str(error)}
        candidate = {
          **base_record,
          "status": "selection_failure",
          "reason": str(error),
          "stages": {field: metrics[field] for field in TIME_FIELDS},
        }
        cases.append(candidate)
        accumulate(metrics)
        append_progress(candidate)
        break
      eligible = (
        selection.get("status") == "accepted"
        and selection.get("property_passed") is True
        and (
          arm["selection"]["arm"] != "directed"
          or selection.get("distinguished") is True
        )
      )
      if not eligible:
        status = _selection_outcome_status(arm["selection"]["arm"], selection)
        candidate = _case_record(
          case_path,
          arm=arm_name,
          seed=seed,
          status=status,
          metrics=metrics,
          selection=selection,
        )
        cases.append(candidate)
        records.append(
          {
            "seed": seed,
            "status": status,
            "proposal_count": selection.get("proposal_count"),
          }
        )
        accumulate(metrics)
        append_progress(candidate)
        print(
          f"fixed-time arm={arm_name} seed={seed} status={status} "
          f"elapsed={time.monotonic() - arm_started:.1f}s",
          flush=True,
        )
        seed += 1
        continue
    report = _execute_one(
      hooks,
      case_path,
      metrics,
      device_id=config["device_pool"][0],
    )
    status = report.get("status", "runner_failure")
    completed += int(status in {"pass", "exact_mismatch", "numeric_mismatch"})
    cache = _cache_record(case_path, arm=arm_name, seed=seed)
    if cache:
      caches.append(cache)
    case_record = _case_record(
      case_path,
      arm=arm_name,
      seed=seed,
      status=status,
      metrics=metrics,
      selection=selection,
      report=report,
    )
    cases.append({**base_record, **case_record})
    records.append(
      {
        "seed": seed,
        "status": status,
        "proposal_count": selection.get("proposal_count") if selection else None,
      }
    )
    accumulate(metrics)
    append_progress(cases[-1])
    case_index += 1
    print(
      f"fixed-time arm={arm_name} seed={seed} status={status} "
      f"completed={completed} elapsed={time.monotonic() - arm_started:.1f}s",
      flush=True,
    )
    if status != "pass" or report.get("campaign_valid") is not True:
      failure = {
        "seed": seed,
        "stage": "compiled-run",
        "status": status,
        "reason": report.get("reason"),
      }
      break
    seed += 1
    if shutil.disk_usage(root).free < 1_500_000_000:
      failure = {
        "seed": seed,
        "stage": "storage_guard",
        "reason": "less than 1.5 GB free; stopped before filling the artifact volume",
      }
      break
  stopped = time.monotonic()
  idle_s = max(0.0, deadline - stopped) if failure is None else 0.0
  if idle_s:
    time.sleep(idle_s)
  elapsed = time.monotonic() - arm_started
  record = {
    "arm": arm_name,
    "budget_s": float(config["wall_clock_budget_s"]),
    "elapsed_s": min(elapsed, float(config["wall_clock_budget_s"]))
    if failure is None
    else elapsed,
    "completed_target_runs": completed,
    **metrics_total,
    "worker_output_materialization_s": metrics_other["worker_output_materialization_s"],
    "runner_overhead_s": metrics_other["runner_overhead_s"],
    "idle_reserve_s": idle_s,
  }
  return {
    "record": record,
    "cases": cases,
    "attempts": records,
    "cache_records": caches,
    "failure": failure,
    "next_seed": seed,
  }


def _fixed_time_arm_config(protocol):
  """Translate the manifest's per-arm budget to the arm runner's field."""
  key = "wall_clock_budget_s_per_arm"
  budget = protocol.get(key) if isinstance(protocol, dict) else None
  if (
    isinstance(budget, bool)
    or not isinstance(budget, (int, float))
    or not math.isfinite(budget)
    or budget <= 0
  ):
    raise ValueError(f"{key} must be a finite positive number")
  return {**protocol, "wall_clock_budget_s": float(budget)}


def run_fixed_time(
  config,
  output,
  hooks,
  tensor_case,
  tensor_coverage,
  tensor_directed,
  tensor_evaluation,
):
  protocol = config["fixed_time"]
  arm_config = _fixed_time_arm_config(protocol)
  root = Path(output) / "fixed-time"
  root.mkdir(parents=True, exist_ok=True)
  ledger_path, summary_path = root / "ledger.json", root / "summary.json"
  ledger = {
    "schema": "refractir.tensor-matched-campaign-ledger.v1",
    "protocol": "fixed-time",
    "status": "in_progress",
    "budget_s": protocol["wall_clock_budget_s_per_arm"],
    "started_at": _now_iso(),
    "records": [],
    "cases": [],
    "cache_records": [],
    "failures": [],
  }
  for arm_name, arm in protocol["arms"].items():
    result = _fixed_time_arm(
      arm_config,
      arm_name,
      arm,
      root,
      hooks,
      tensor_case,
      tensor_directed,
    )
    ledger["records"].append(result["record"])
    ledger["cases"].extend(result["cases"])
    ledger["cache_records"].extend(result["cache_records"])
    if result["failure"]:
      ledger["failures"].append({"arm": arm_name, **result["failure"]})
      break
    _json_write(ledger_path, ledger)
  ledger["finished_at"] = _now_iso()
  ledger["status"] = "stopped_on_failure" if ledger["failures"] else "finished"
  case_dirs = [
    Path(record["case_directory"])
    for record in ledger["cases"]
    if record.get("case_directory") and Path(record["case_directory"]).is_dir()
  ]
  if case_dirs:
    ledger["coverage"] = tensor_coverage.index_cases(case_dirs)
  _json_write(ledger_path, ledger)
  if len(ledger["records"]) == len(protocol["arms"]) and not ledger["failures"]:
    _json_write(summary_path, tensor_evaluation.summarize_ledger(ledger))
  else:
    _json_write(
      summary_path,
      {
        "protocol": "fixed-time",
        "status": "aborted_before_equal-budget-completion",
        "failures": ledger["failures"],
        "completed_arms": len(ledger["records"]),
        "predeclared_arms": len(protocol["arms"]),
      },
    )
  return ledger


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument(
    "--protocol", choices=("fixed-count", "fixed-time", "both"), default="both"
  )
  args = parser.parse_args(argv)
  config = json.loads(args.config.read_text(encoding="utf-8"))
  _validate_model_objective_pairs(config)
  if (
    config.get("status") != "predeclared_not_run"
    or config.get("configuration_frozen") is not True
  ):
    raise ValueError("campaign configuration must be frozen before execution")
  if args.output.exists() and any(args.output.iterdir()):
    raise FileExistsError(f"campaign output directory is not empty: {args.output}")
  args.output.mkdir(parents=True, exist_ok=True)
  shutil.copy2(args.config, args.output / "manifest.json")
  driver_copy = args.output / "driver.py"
  if Path(__file__).resolve() != driver_copy.resolve():
    shutil.copy2(__file__, driver_copy)
  _init_environment(config)
  from test.lib import (
    run_triton_tests,
    tensor_case,
    tensor_coverage,
    tensor_directed,
    tensor_evaluation,
  )

  hooks = TimingHooks(run_triton_tests, tensor_case, tensor_directed)
  hooks.install(config["environment"])
  overall_failure = False
  try:
    if args.protocol in {"fixed-count", "both"}:
      fixed_count = run_fixed_count(
        config,
        args.output,
        hooks,
        tensor_case,
        tensor_coverage,
        tensor_directed,
        tensor_evaluation,
      )
      overall_failure = bool(fixed_count["failures"])
    if args.protocol in {"fixed-time", "both"} and not overall_failure:
      fixed_time = run_fixed_time(
        config,
        args.output,
        hooks,
        tensor_case,
        tensor_coverage,
        tensor_directed,
        tensor_evaluation,
      )
      overall_failure = bool(fixed_time["failures"])
  finally:
    hooks.restore()
  return 1 if overall_failure else 0


if __name__ == "__main__":
  sys.exit(main())
