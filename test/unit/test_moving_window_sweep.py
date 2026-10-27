"""High-volume sweep contracts (section 6.5, H1) with scripted workers."""

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

DRIVER_PATH = (
  Path(__file__).resolve().parents[2]
  / "test"
  / "tensor"
  / "campaigns"
  / "run_moving_window_sweep.py"
)
_spec = importlib.util.spec_from_file_location("moving_window_sweep", DRIVER_PATH)
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

S0 = 30000000
ONE = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
TWO = struct.pack("<4f", 1.0, 2.0, 3.0, 5.0)
NAN = struct.pack("<4f", 1.0, float("nan"), 3.0, 4.0)
# The same positions, but a different NaN payload in the second word.
OTHER_NAN = (
  struct.pack("<f", 1.0) + bytes.fromhex("0100c07f") + struct.pack("<2f", 3.0, 4.0)
)
CONFIGURATIONS = ({"num_warps": 1, "num_stages": 1}, {"num_warps": 4, "num_stages": 1})
I0, I3 = "i0-base", "i3-unused-rows"


def spec(count, *, gpu_workers=1000, wall_clock_s=3600.0, sample_every=20):
  return sweep.sweep_spec(
    seed_start=S0,
    seed_count=count,
    configurations=CONFIGURATIONS,
    sample_every=sample_every,
    gpu_workers=gpu_workers,
    wall_clock_s=wall_clock_s,
  )


def lines(path):
  path = Path(path)
  if not path.is_file():
    return []
  return [json.loads(line) for line in path.read_text().splitlines() if line]


class Script:
  """Scripted builder and paired worker; every unscripted path returns ONE."""

  def __init__(
    self,
    *,
    refused=(),
    outputs=None,
    statuses=None,
    interrupt=None,
    live_nodes=None,
    compiler=None,
  ):
    self.refused = set(refused)
    self.live_nodes = live_nodes or {}
    self.compiler = compiler or {}
    self.outputs = outputs or {}
    self.statuses = statuses or {}
    self.interrupt = interrupt
    self.toolchain = {"exporter_sha256": "exporter", "triton": "32f06df4"}
    self.builds = []
    self.pairs = []
    self.screens = []

  def build(self, request):
    seed = request["seed"]
    self.builds.append(seed)
    if seed in self.refused:
      return {"kind": "refused", "reason": "no live window witness"}
    directory = Path(request["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "program.sir").write_text(f"sir {seed}\n")
    return {
      "kind": "admitted",
      "directory": str(directory),
      "source_sha256": f"sir-{seed}",
      "typed_fingerprint": f"fp-{seed}",
      "connectivity_signature": "c0",
      "live_nodes": self.live_nodes.get(seed, 9),
    }

  def run_pair(self, request):
    seed = request["kernel"]["seed"]
    key = (seed, request["configuration"]["num_warps"], request["input"])
    if self.interrupt == key:
      self.interrupt = None
      raise KeyboardInterrupt
    self.pairs.append(key)
    raw = Path(request["directory"]) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "worker.log").write_text("worker log\n")
    result = {
      "kernel_sha256": f"kernel-{seed}",
      "inputs_sha256": f"inputs-{request['input']}",
      "raw": [str(raw)],
      "compiler": self.compiler.get(key, {"status": "unavailable"}),
    }
    for path in ("p1", "p2"):
      status, reason, *phase = self.statuses.get((*key, path), ("completed", ""))
      output = self.outputs.get((*key, path), ONE) if status == "completed" else None
      result[path] = {
        "status": status,
        "reason": reason,
        "output": output,
        "phase": phase[0] if phase else "runtime",
      }
    return result

  def screen(self, kernel):
    self.screens.append(kernel["seed"])
    return {"i1-read-plus-one": "changed", "i2-late-row": "unchanged"}

  @property
  def operations(self):
    return {
      "build": self.build,
      "run_pair": self.run_pair,
      "screen": self.screen,
      "toolchain": self.toolchain,
    }


class SweepFixture(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.work = Path(temporary.name) / "sweep"

  def run_sweep(self, declared, script, **kwargs):
    kwargs.setdefault("free_bytes", lambda path: 10**12)
    return sweep.run_sweep(self.work, declared, script.operations, **kwargs)

  def kernels(self):
    return [
      line for line in lines(self.work / "run-log.jsonl") if line["event"] == "kernel"
    ]

  def candidates(self):
    return lines(self.work / "candidates.jsonl")


class SweepContractTests(SweepFixture):
  def test_a_path_difference_is_quarantined_and_the_sweep_continues(self):
    # P2 differs on both inputs, so I3 still equals I0 on each path and the
    # path difference is the only candidate.
    script = Script(outputs={(S0, 1, I0, "p2"): TWO, (S0, 1, I3, "p2"): TWO})
    report = self.run_sweep(spec(3), script)
    candidates = self.candidates()
    self.assertEqual(
      [(c["seed"], c["kind"], c["input"]) for c in candidates],
      [(S0, "path_difference", I0), (S0, "path_difference", I3)],
    )
    self.assertEqual(candidates[0]["positions"], [3])
    self.assertEqual(candidates[0]["signature"], candidates[1]["signature"])
    self.assertTrue(candidates[0]["signature"])
    self.assertEqual(
      [k["status"] for k in self.kernels()], ["candidate", "agreement", "agreement"]
    )
    self.assertEqual(len(script.pairs), 12)
    self.assertTrue((Path(candidates[0]["directory"]) / "raw" / "worker.log").is_file())
    self.assertEqual(report["stop_reason"], "seed_block_complete")

  def test_an_unused_row_difference_on_either_path_is_a_candidate(self):
    script = Script(
      outputs={
        (S0, 1, I3, "p1"): TWO,
        (S0, 1, I3, "p2"): TWO,
        (S0 + 1, 4, I3, "p2"): TWO,
      }
    )
    self.run_sweep(spec(2), script)
    found = {(c["seed"], c["kind"], c.get("path")) for c in self.candidates()}
    self.assertIn((S0, "unused_row_difference", "p1"), found)
    self.assertIn((S0, "unused_row_difference", "p2"), found)
    self.assertNotIn((S0, "path_difference", None), found)
    self.assertIn((S0 + 1, "unused_row_difference", "p2"), found)
    self.assertNotIn((S0 + 1, "unused_row_difference", "p1"), found)
    self.assertIn((S0 + 1, "path_difference", None), found)
    self.assertEqual([k["status"] for k in self.kernels()], ["candidate", "candidate"])

  def test_a_failed_path_is_a_candidate_with_its_status_and_never_agreement(self):
    script = Script(
      statuses={
        (S0, 1, I0, "p2"): ("compile_failure", "error: unsupported layout"),
        (S0 + 1, 4, I3, "p1"): ("timed_out", "worker exceeded 300 s"),
        (S0 + 2, 1, I0, "p2"): ("runtime_failure", "CUDA error: illegal address"),
      },
      outputs={(S0 + 3, 1, I0, "p1"): NAN, (S0 + 3, 1, I0, "p2"): NAN},
    )
    report = self.run_sweep(spec(4), script)
    found = {(c["seed"], c["kind"], c.get("status")) for c in self.candidates()}
    self.assertIn((S0, "execution_status", "compile_failure"), found)
    self.assertIn((S0 + 1, "execution_status", "timed_out"), found)
    self.assertIn((S0 + 2, "execution_status", "runtime_failure"), found)
    self.assertIn((S0 + 3, "non_finite", None), found)
    self.assertEqual([k["status"] for k in self.kernels()], ["candidate"] * 4)
    self.assertEqual(report["counts"]["agreement"], 0)
    failed = self.kernels()[0]["pairs"][0]["p2"]
    self.assertEqual(
      (failed["status"], failed["output_sha256"]), ("compile_failure", None)
    )

  def test_a_restart_continues_the_seed_block_without_rerunning_completed_kernels(self):
    script = Script(interrupt=(S0 + 2, 1, I0))
    with self.assertRaises(KeyboardInterrupt):
      self.run_sweep(spec(4), script)
    report = self.run_sweep(spec(4), script)
    self.assertEqual(script.builds, [S0, S0 + 1, S0 + 2, S0 + 2, S0 + 3])
    self.assertEqual([k["seed"] for k in self.kernels()], [S0, S0 + 1, S0 + 2, S0 + 3])
    events = [line["event"] for line in lines(self.work / "run-log.jsonl")]
    self.assertEqual(events.count("restart"), 1)
    self.assertIn("interrupted", events)
    self.assertTrue(
      any(
        path.name.startswith(f"seed-{S0 + 2}.interrupted")
        for path in (self.work / "kernels").iterdir()
      )
    )
    self.assertEqual(report["gpu_workers_dispatched"], 17)
    self.assertEqual(report["stop_reason"], "seed_block_complete")

  def test_the_budget_and_the_infrastructure_threshold_stop_dispatch(self):
    with self.subTest("gpu budget"):
      script = Script()
      report = self.run_sweep(spec(5, gpu_workers=9), script)
      self.assertEqual(len(script.pairs), 8)
      self.assertEqual(report["stop_reason"], "gpu_budget")
    with self.subTest("infrastructure rate"):
      self.work = self.work.with_name("infrastructure")
      script = Script(
        statuses={
          (S0 + index, 1, I0, "p2"): ("environment_unavailable", "CUDA device 0")
          for index in range(10)
        }
      )
      report = self.run_sweep(spec(10), script)
      self.assertEqual(len(self.kernels()), 6)
      self.assertEqual({k["status"] for k in self.kernels()}, {"infrastructure"})
      self.assertEqual(report["stop_reason"], "infrastructure_failure_rate")
    with self.subTest("wall clock"):
      self.work = self.work.with_name("clock")
      ticks = iter(range(0, 100000, 1000))
      script = Script()
      report = self.run_sweep(
        spec(10, wall_clock_s=2500.0), script, clock=lambda: next(ticks)
      )
      self.assertEqual(report["stop_reason"], "wall_clock")
      self.assertLess(len(self.kernels()), 10)
    with self.subTest("disk guard"):
      self.work = self.work.with_name("disk")
      script = Script()
      report = self.run_sweep(spec(3), script, free_bytes=lambda path: 1024)
      self.assertEqual(script.builds, [])
      self.assertEqual(report["stop_reason"], "disk_guard")


def observed(conversions, *, loop=1, encodings=("blocked",)):
  return {
    "status": "observed",
    "stages": {
      "ttgir": {"layout_conversion_count": conversions, "loop_construct_count": loop}
    },
    "encodings": list(encodings),
  }


def ticking(step):
  """A session clock that advances `step` seconds on every reading."""
  readings = iter(range(0, 10**9, step))
  return lambda: next(readings)


def cell(request):
  return (request["group"], request["column"]["column"], request["policy"])


class SeedOrderTests(unittest.TestCase):
  """Each block of 96 seeds is a seeded, policy-stratified permutation."""

  def requests(self, start=S0, count=192):
    declared = sweep.sweep_spec(
      seed_start=start,
      seed_count=count,
      configurations=CONFIGURATIONS,
      sample_every=20,
      gpu_workers=1000,
      wall_clock_s=3600.0,
    )
    return [sweep.kernel_request(declared, index) for index in range(count)]

  def test_every_block_covers_each_cell_once(self):
    requests = self.requests()
    for block in (requests[:96], requests[96:]):
      self.assertEqual(len({cell(request) for request in block}), 96)
    self.assertEqual([r["seed"] for r in requests], list(range(S0, S0 + 192)))

  def test_policies_take_turns_in_every_prefix(self):
    counts = dict.fromkeys(sweep.POLICIES, 0)
    for request in self.requests(count=96):
      counts[request["policy"]] += 1
      self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

  def test_the_order_is_seeded_and_differs_between_blocks(self):
    first = [cell(request) for request in self.requests()]
    self.assertEqual(first, [cell(request) for request in self.requests()])
    self.assertNotEqual(first[:96], first[96:])
    other = [cell(request) for request in self.requests(start=S0 + 7)]
    self.assertNotEqual(first[:96], other[:96])

  def test_the_declaration_names_the_order(self):
    self.assertEqual(spec(4)["order"], sweep.ORDER)

  def test_requirement_and_padding_follow_group_and_column(self):
    requirements = {"G0": "reduction", "G1": "shared", "unconstrained": None}
    for request in self.requests(count=96):
      group, column = request["group"], request["column"]["column"]
      expected = (
        requirements.get(group)
        if group != "G2"
        else ("extrema-transpose" if column % 2 == 0 else "extrema-expand")
      )
      self.assertEqual(request["requirement"], expected)
      self.assertEqual(request["padded"], (sweep.GROUPS.index(group) + column) % 2 == 1)


class SweepReportTests(SweepFixture):
  """The pilot's acceptance measurements, computed from the retained records."""

  def measurements(self, *args, **kwargs):
    return self.run_sweep(*args, **kwargs)["measurements"]

  def test_kernels_per_hour_is_measured_on_the_session_clock(self):
    script = Script(refused={S0 + 1})
    measured = self.measurements(
      spec(3, wall_clock_s=10**6), script, clock=ticking(600)
    )
    self.assertEqual([k["elapsed_s"] for k in self.kernels()], [600, 600, 600])
    self.assertEqual(measured["elapsed_s"], 1800)
    self.assertEqual(measured["kernels_per_hour"], 4.0)

  def test_disk_use_per_admitted_kernel_is_measured_after_pruning(self):
    script = Script()
    measured = self.measurements(spec(2, sample_every=2), script)

    def size(seed):
      root = self.work / "kernels" / f"seed-{seed}"
      return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())

    sizes = [size(S0), size(S0 + 1)]
    self.assertLess(sizes[0], sizes[1])
    self.assertEqual([k["disk_bytes"] for k in self.kernels()], sizes)
    self.assertEqual(
      measured["disk_bytes_per_kernel"],
      {"mean": sum(sizes) / 2, "max": sizes[1], "total": sum(sizes)},
    )

  def test_the_infrastructure_rate_is_over_admitted_kernels(self):
    script = Script(
      refused={S0 + 4},
      statuses={(S0 + 1, 4, I3, "p2"): ("environment_unavailable", "CUDA device 0")},
    )
    measured = self.measurements(spec(5), script)
    self.assertEqual(
      measured["infrastructure"], {"kernels": 1, "admitted": 4, "rate": 0.25}
    )

  def test_the_live_node_histogram_counts_admitted_kernels_only(self):
    script = Script(refused={S0 + 2}, live_nodes={S0: 5, S0 + 1: 12, S0 + 3: 5})
    measured = self.measurements(spec(4), script)
    self.assertEqual(measured["live_nodes"], {"5": 2, "12": 1})

  def test_exposure_statistics_summarize_ttgir_per_paired_run(self):
    script = Script(
      compiler={
        (S0, 1, I0): observed(0),
        (S0, 1, I3): observed(0),
        (S0, 4, I0): observed(2, encodings=("blocked", "slice")),
        (S0, 4, I3): observed(2, loop=0, encodings=("blocked", "slice")),
      }
    )
    exposure = self.measurements(spec(2), script)["exposure"]
    self.assertEqual((exposure["observed"], exposure["unavailable"]), (4, 4))
    self.assertEqual(exposure["ttgir_layout_conversions"], {"0": 2, "2": 2})
    self.assertEqual(exposure["ttgir_loop_retained"], 3)
    self.assertEqual(exposure["encoding_kinds"], {"blocked": 4, "slice": 2})


class CompileTimeoutTests(SweepFixture):
  """A compiled-path timeout during compilation is not paid twice per setting."""

  def test_a_compilation_timeout_skips_the_rest_of_its_configuration(self):
    script = Script(
      statuses={(S0, 1, I0, "p2"): ("timed_out", "worker exceeded", "compilation")}
    )
    report = self.run_sweep(spec(1), script)
    self.assertEqual(script.pairs, [(S0, 1, I0), (S0, 4, I0), (S0, 4, I3)])
    self.assertEqual(report["gpu_workers_dispatched"], 3)
    kernel = self.kernels()[0]
    skipped = [pair for pair in kernel["pairs"] if pair["p1"]["status"] == "skipped"]
    self.assertEqual(
      [(pair["configuration"]["num_warps"], pair["input"]) for pair in skipped],
      [(1, I3)],
    )
    self.assertIn("compilation timed out", skipped[0]["p2"]["reason"])
    self.assertEqual((kernel["status"], kernel["infrastructure"]), ("candidate", False))
    kinds = {c["kind"] for c in self.candidates()}
    self.assertEqual(kinds, {"execution_status"})

  def test_a_runtime_timeout_still_runs_the_other_input(self):
    script = Script(
      statuses={(S0, 1, I0, "p2"): ("timed_out", "worker exceeded", "runtime")}
    )
    self.run_sweep(spec(1), script)
    self.assertEqual(len(script.pairs), 4)

  def test_each_session_records_its_timeout(self):
    script = Script()
    operations = {**script.operations, "timeout_s": 120.0}
    sweep.run_sweep(self.work, spec(1), operations, free_bytes=lambda path: 10**12)
    starts = [
      line for line in lines(self.work / "run-log.jsonl") if line["event"] == "start"
    ]
    self.assertEqual(starts[0]["timeout_s"], 120.0)

  def test_a_path_difference_in_non_finite_output_is_tagged(self):
    script = Script(
      outputs={
        (S0, 1, I0, "p1"): NAN,
        (S0, 1, I0, "p2"): OTHER_NAN,
        (S0, 1, I3, "p1"): NAN,
        (S0, 1, I3, "p2"): OTHER_NAN,
      }
    )
    self.run_sweep(spec(1), script)
    kinds = {c["kind"] for c in self.candidates()}
    self.assertIn("non_finite_path_difference", kinds)
    self.assertNotIn("path_difference", kinds)


class SweepEdgeTests(SweepFixture):
  def test_a_restart_after_an_infrastructure_stop_dispatches_nothing(self):
    script = Script(
      statuses={
        (S0 + index, 1, I0, "p2"): ("environment_unavailable", "CUDA device 0")
        for index in range(10)
      }
    )
    self.run_sweep(spec(10), script)
    dispatched = len(script.pairs)
    report = self.run_sweep(spec(10), script)
    self.assertEqual(len(script.pairs), dispatched)
    self.assertEqual(report["stop_reason"], "infrastructure_failure_rate")

  def test_each_session_records_its_device(self):
    script = Script()
    operations = {**script.operations, "device": 6}
    sweep.run_sweep(self.work, spec(1), operations, free_bytes=lambda path: 10**12)
    sessions = [
      line for line in lines(self.work / "run-log.jsonl") if line["event"] == "start"
    ]
    self.assertEqual(sessions[0]["device"], 6)

  def test_a_builder_refusal_is_recorded_and_consumes_its_seed(self):
    script = Script(refused={S0 + 1})
    self.run_sweep(spec(3), script)
    self.run_sweep(spec(3), script)
    kernels = self.kernels()
    self.assertEqual(
      [k["status"] for k in kernels], ["agreement", "refused", "agreement"]
    )
    self.assertEqual(kernels[1]["reason"], "no live window witness")
    self.assertEqual(script.builds, [S0, S0 + 1, S0 + 2])
    self.assertNotIn(S0 + 1, {pair[0] for pair in script.pairs})

  def test_only_sampled_and_candidate_kernels_keep_raw_payloads(self):
    script = Script(outputs={(S0, 1, I0, "p2"): TWO, (S0, 1, I3, "p2"): TWO})
    self.run_sweep(spec(4, sample_every=2), script)
    kernels = self.kernels()
    self.assertEqual([k["sampled"] for k in kernels], [False, True, False, True])
    self.assertEqual(script.screens, [S0 + 1, S0 + 3])
    self.assertEqual(kernels[1]["exposure_sample"]["i1-read-plus-one"], "changed")
    kept = {
      int(path.parts[-5].split("-")[1])
      for path in (self.work / "kernels").glob("*/pairs/*/raw/worker.log")
    }
    self.assertEqual(kept, {S0, S0 + 1, S0 + 3})
    self.assertTrue(all(pair["p1"]["output_sha256"] for pair in kernels[2]["pairs"]))

  def test_a_restart_refuses_a_changed_declaration_or_toolchain(self):
    script = Script()
    self.run_sweep(spec(2), script)
    with self.assertRaises(sweep.SweepRefused):
      self.run_sweep(spec(3), script)
    script.toolchain = {**script.toolchain, "triton": "other"}
    with self.assertRaises(sweep.SweepRefused):
      self.run_sweep(spec(2), script)


FAKE_EXPORTER = """\
import json
import pathlib
import sys

arguments = sys.argv[1:]
out = pathlib.Path(arguments[arguments.index("--out") + 1])
out.mkdir(parents=True)
(out.parent / "argv.json").write_text(json.dumps(arguments))
region = json.loads(pathlib.Path(sys.argv[0]).with_name("region.json").read_text())
(out / "region.json").write_text(json.dumps(region))
source = pathlib.Path(sys.argv[0]).with_name("program.sir").read_text()
(out / "program.sir").write_text(source)
"""


class DeviceBindingTests(unittest.TestCase):
  def setUp(self):
    from test.unit import test_tensor_free_form_case as fixtures

    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name)
    region = {
      **fixtures.WINDOW_REGION,
      "num_warps": 1,
      "max_nodes": 16,
      "structure": {"nodes": 9},
      "live_dag": {"typed_fingerprint": "fp", "connectivity_signature": "cs"},
    }
    (self.root / "region.json").write_text(json.dumps(region))
    (self.root / "program.sir").write_text(fixtures.WINDOW_SOURCE)
    self.exporter = self.root / "exporter.py"
    self.exporter.write_text(f"#!{sys.executable}\n" + FAKE_EXPORTER)
    self.exporter.chmod(0o755)
    self.operations = sweep.device_operations(
      exporter=self.exporter,
      symirc=self.root / "symirc",
      symiri=self.root / "symiri",
      toolchain={"exporter_sha256": "fake"},
    )

  def index_of(self, group):
    """The first index whose cell matches the column-0 window fixture."""
    declared = spec(96)
    for index in range(96):
      request = sweep.kernel_request(declared, index)
      if request["group"] == group and request["column"]["column"] == 0:
        return index
    raise AssertionError(f"no column-0 cell for {group}")

  def build(self, index):
    request = sweep.kernel_request(spec(96), index)
    directory = self.root / f"kernel-{index}"
    built = self.operations["build"]({**request, "directory": str(directory)})
    argv = json.loads((directory / "argv.json").read_text())
    return request, built, argv

  def test_the_builder_exports_the_declared_request(self):
    request, built, argv = self.build(self.index_of("G0"))
    self.assertEqual(request["requirement"], "reduction")
    self.assertEqual(argv[argv.index("--window-requirement") + 1], "reduction")
    self.assertEqual(argv[argv.index("--max-nodes") + 1], "16")
    self.assertEqual(argv[argv.index("--max-multiplications") + 1], "1")
    self.assertEqual(argv[argv.index("--seed") + 1], str(request["seed"]))
    self.assertEqual(built["kind"], "admitted")
    self.assertEqual(
      (
        built["typed_fingerprint"],
        built["connectivity_signature"],
        built["live_nodes"],
      ),
      ("fp", "cs", 9),
    )
    self.assertEqual(set(built["probe_values"]), set(sweep.PROBES))

  def test_the_unconstrained_group_exports_without_a_requirement(self):
    request, built, argv = self.build(self.index_of("unconstrained"))
    self.assertEqual(request["group"], "unconstrained")
    self.assertNotIn("--window-requirement", argv)
    self.assertEqual(built["kind"], "admitted")

  def test_an_exporter_failure_or_timeout_is_a_refusal(self):
    request = sweep.kernel_request(spec(4), 0)
    for name, body, timeout in (
      ("failure", "import sys\nsys.exit('no witness within one attempt')\n", 30.0),
      ("timeout", "import time\ntime.sleep(30)\n", 0.5),
    ):
      with self.subTest(name):
        exporter = self.root / f"{name}.py"
        exporter.write_text(f"#!{sys.executable}\n" + body)
        exporter.chmod(0o755)
        operations = sweep.device_operations(
          exporter=exporter,
          symirc=self.root / "symirc",
          symiri=self.root / "symiri",
          timeout_s=timeout,
          toolchain={"exporter_sha256": "fake"},
        )
        built = operations["build"]({**request, "directory": str(self.root / name)})
        self.assertEqual(built["kind"], "refused")
        self.assertTrue(built["reason"])

  def test_ttgir_encoding_kinds_are_read_from_their_definitions(self):
    text = (
      "#blocked = #ttg.blocked<{sizePerThread = [1, 1], warpsPerCTA = [4, 1]}>\n"
      "#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], warpsPerCTA = [1, 4]}>\n"
      "#loc = loc(unknown)\n"
      "#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1}>\n"
      "module attributes {} {\n"
      "  %0 = ttg.convert_layout %x : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>>\n"
      "}\n"
    )
    self.assertEqual(
      sweep.ttgir_encodings(text), ["blocked", "blocked", "swizzled_shared"]
    )

  def test_the_paired_run_uses_the_declared_device(self):
    from unittest import mock

    from test.lib import tensor_target_run as target

    operations = sweep.device_operations(
      exporter=self.exporter,
      symirc=self.root / "symirc",
      symiri=self.root / "symiri",
      toolchain={"exporter_sha256": "fake"},
      device_id=3,
    )
    self.assertEqual(operations["device"], 3)
    request, built, _ = self.build(self.index_of("G0"))
    kernel = {**built, "seed": request["seed"], "request": request}
    finished = target.TargetRun(
      bundle=self.root,
      status=target.TARGET_AGREEMENT,
      emission=None,
      interpreter=target.ModeRun(
        "interpret", target.COMPLETED, 0, "runtime", ONE, "", ""
      ),
      compiled=target.ModeRun("compiled", target.COMPLETED, 0, "runtime", ONE, "", ""),
    )
    with mock.patch.object(target, "run", return_value=finished) as run:
      operations["run_pair"](
        {
          "kernel": kernel,
          "configuration": {"num_warps": 1, "num_stages": 1},
          "input": I0,
          "directory": str(self.root / "pair"),
        }
      )
    self.assertEqual(run.call_args.kwargs["device_id"], 3)

  def test_the_command_line_passes_the_device(self):
    import contextlib
    import io
    from unittest import mock

    with (
      mock.patch.object(sweep, "device_operations", return_value={}) as bind,
      mock.patch.object(sweep, "run_sweep", return_value={}),
      contextlib.redirect_stdout(io.StringIO()),
    ):
      arguments = ["--work", str(self.root / "w"), "--seed-start", str(S0)]
      arguments += ["--seed-count", "4", "--gpu-workers", "8"]
      arguments += ["--wall-clock-hours", "1", "--exporter", str(self.exporter)]
      sweep.main(arguments + ["--device", "5"])
      self.assertEqual(bind.call_args.kwargs["device_id"], 5)
      sweep.main(arguments)
      self.assertEqual(bind.call_args.kwargs["device_id"], 0)

  def test_a_target_run_becomes_path_statuses_outputs_and_raw_paths(self):
    from test.lib import tensor_target_run as target

    def mode(name, status, actual):
      return target.ModeRun(name, status, 0, "runtime", actual, "", "", reason="why")

    bundle = self.root / "bundle"
    run = target.TargetRun(
      bundle=bundle,
      status=target.COMPILE_FAILURE,
      emission=None,
      interpreter=mode("interpret", target.COMPLETED, ONE),
      compiled=mode("compiled", target.COMPILE_FAILURE, None),
    )
    result = sweep.pair_result(run, bundle)
    self.assertEqual(
      result["p1"],
      {"status": "completed", "reason": "why", "output": ONE, "phase": "runtime"},
    )
    self.assertEqual(result["p2"]["status"], "compile_failure")
    self.assertIsNone(result["p2"]["output"])
    self.assertEqual(
      result["raw"],
      [
        str(bundle / "target-runs" / "interpret"),
        str(bundle / "target-runs" / "compiled"),
      ],
    )
    failure = sweep.pair_failure(RuntimeError("bundle was not executable"))
    self.assertEqual(
      {failure["p1"]["status"], failure["p2"]["status"]}, {"infrastructure_error"}
    )


COMPILE_PATH = DRIVER_PATH.with_name("compile_kernel_stages.py")

KERNEL = """\
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, out_ptr):
    pass
"""


class FakeTriton:
  """Just enough of the Triton API for a compile-only run without a GPU."""

  def __init__(self, fail=None):
    import types

    self.calls = []
    self.fail = fail
    triton = types.ModuleType("triton")
    compiler = types.ModuleType("triton.compiler")
    language = types.ModuleType("triton.language")

    class JITFunction:
      def __init__(self, fn):
        self.fn = fn
        self.arg_names = list(fn.__code__.co_varnames[: fn.__code__.co_argcount])

    class ASTSource:
      def __init__(self, fn, signature, constexprs=None, attrs=None):
        self.fn, self.signature = fn, signature

    def compile(source, target=None, options=None):
      self.calls.append((source.signature, dict(options)))
      warps, stages = options["num_warps"], options["num_stages"]
      if (warps, stages) == self.fail:
        raise RuntimeError("PassManager::run failed")
      return types.SimpleNamespace(
        asm={
          "ttir": "ttir\n",
          "ttgir": f"ttgir warps={warps} pipelined={stages > 1}\n",
          "llir": "llir\n",
          "ptx": f"ptx {warps} {stages}\n",
          "cubin": b"\x00",
        }
      )

    triton.jit = JITFunction
    triton.compile = compile
    triton.compiler = compiler
    triton.language = language
    compiler.ASTSource = ASTSource
    self.modules = {
      "triton": triton,
      "triton.compiler": compiler,
      "triton.language": language,
    }


class CompileOnlyTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name)
    self.kernel = self.root / "kernel.py"
    self.kernel.write_text(KERNEL)
    spec = importlib.util.spec_from_file_location("compile_kernel_stages", COMPILE_PATH)
    self.stages = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(self.stages)

  def compile(self, fake):
    from unittest import mock

    with mock.patch.dict(sys.modules, fake.modules):
      return self.stages.compile_matrix(
        self.kernel, warps=[1, 4], stages=[1, 2], out=self.root / "out"
      )

  def test_each_setting_is_compiled_once_and_its_stages_are_saved(self):
    fake = FakeTriton()
    table = self.compile(fake)
    self.assertEqual(len(fake.calls), 4)
    signature, options = fake.calls[0]
    self.assertEqual(signature, {"x_ptr": "*fp32", "out_ptr": "*fp32"})
    self.assertEqual(
      options, {"num_warps": 1, "num_stages": 1, "enable_fp_fusion": False}
    )
    self.assertEqual(
      sorted(path.name for path in (self.root / "out" / "warps-4-stages-2").iterdir()),
      ["llir.ir", "ptx.ir", "ttgir.ir", "ttir.ir"],
    )
    self.assertEqual({row["status"] for row in table["settings"]}, {"compiled"})
    self.assertEqual(table["ttgir_changes_with_stages"], {"1": True, "4": True})

  def test_a_failed_setting_is_recorded_and_the_others_still_compile(self):
    table = self.compile(FakeTriton(fail=(4, 2)))
    rows = {(row["num_warps"], row["num_stages"]): row for row in table["settings"]}
    self.assertEqual(rows[(4, 2)]["status"], "failed")
    self.assertIn("PassManager::run failed", rows[(4, 2)]["error"])
    self.assertEqual(rows[(1, 2)]["status"], "compiled")
    self.assertIsNone(table["ttgir_changes_with_stages"]["4"])
    self.assertTrue((self.root / "out" / "table.json").is_file())


if __name__ == "__main__":
  unittest.main()
