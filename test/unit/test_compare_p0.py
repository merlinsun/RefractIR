"""Retained sweep runs compared with P0, the SIR interpreter (plan W1e)."""

import contextlib
import hashlib
import importlib.util
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

from test.lib import tensor_free_form_case, tensor_free_form_inputs
from test.unit.tensor_support import SYMIRI
from test.unit.test_tensor_free_form_case import ONE_TRIP_REGION, ONE_TRIP_SOURCE

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
  "compare_p0", _ROOT / "test" / "tensor" / "campaigns" / "compare_p0.py"
)
compare_p0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare_p0)

S0 = 30000000
WARPS_1 = {"num_warps": 1, "num_stages": 1}
WARPS_4 = {"num_warps": 4, "num_stages": 1}
I0, I6 = "i0-base", "i6-cancel"


def sha(data):
  return hashlib.sha256(data).hexdigest()


@unittest.skipUnless(SYMIRI.is_file(), "symiri is unavailable")
class CompareP0Tests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.shard = Path(temporary.name) / "shard-0"
    self.scratch = Path(temporary.name) / "scratch"
    self.kernel_lines = []

  def probe(self, name):
    probes = tensor_free_form_inputs.window_probes(ONE_TRIP_REGION, group=0, column=5)
    probes += tensor_free_form_inputs.corner_probes(ONE_TRIP_REGION, group=0, column=5)
    return next(probe for probe in probes if probe.probe == name)

  def p0_output(self, name):
    """P0's logical output for the fixture body: x plus the one padded tile."""
    probe = self.probe(name)
    backing, x = probe.values[0], probe.values[1]
    values = [
      x[r][c] + (backing[r][c] if r < 2 else 0.0) for r in range(3) for c in range(4)
    ]
    return struct.pack("<12f", *values)

  def kernel(self, seed, pairs, *, ops=("add",), status="agreement", raw=None):
    """One retained kernel: its region, a bundle per pair, and its run-log line."""
    root = self.shard / "kernels" / f"seed-{seed}"
    (root / "export").mkdir(parents=True)
    (root / "export" / "region.json").write_text(
      json.dumps({"live_dag": {"ops": list(ops)}, "structure": {"nodes": len(ops)}})
    )
    lines = []
    for configuration, name, p1, p2 in pairs:
      bundle = (
        root / "pairs" / compare_p0.sweep._pair_name(configuration, name) / "bundle"
      )
      tensor_free_form_case.build_bundle(
        bundle,
        region={**ONE_TRIP_REGION, "num_warps": configuration["num_warps"]},
        source=ONE_TRIP_SOURCE,
        proposal=self.probe(name),
      )
      for mode, output in (
        (raw or {}).get((configuration["num_warps"], name), {}).items()
      ):
        directory = bundle / "target-runs" / mode
        directory.mkdir(parents=True)
        (directory / "actual.bin").write_bytes(output)
      lines.append(
        {
          "configuration": configuration,
          "input": name,
          "p1": {"status": "completed", "output_sha256": p1},
          "p2": {"status": "completed", "output_sha256": p2},
        }
      )
    self.kernel_lines.append(
      {
        "event": "kernel",
        "seed": seed,
        "status": status,
        "live_nodes": len(ops),
        "pairs": lines,
      }
    )

  def compare(self):
    self.shard.mkdir(parents=True, exist_ok=True)
    with (self.shard / "run-log.jsonl").open("w", encoding="utf-8") as log:
      log.write(json.dumps({"event": "start"}) + "\n")
      for line in self.kernel_lines:
        log.write(json.dumps(line) + "\n")
    return compare_p0.compare([self.shard], symiri=SYMIRI, scratch=self.scratch)

  def test_agreement_is_reported_per_body_and_per_probe(self):
    good = sha(self.p0_output(I0))
    self.kernel(S0, [(WARPS_1, I0, good, good), (WARPS_4, I0, good, good)])
    report = self.compare()
    self.assertEqual(report["by_probe"][I0], {"agree": 4})
    self.assertEqual(report["bodies"], {"agree": 1})
    self.assertEqual(report["disagreements"], [])

  def test_p0_runs_once_per_body_and_input(self):
    good = sha(self.p0_output(I0))
    self.kernel(S0, [(WARPS_1, I0, good, good), (WARPS_4, I0, good, good)])
    self.assertEqual(self.compare()["p0_runs"], 1)

  def test_a_mismatch_names_the_path_output_and_differing_words(self):
    expected = self.p0_output(I0)
    wrong = expected[:8] + struct.pack("<f", 99.0) + expected[12:]
    self.kernel(
      S0,
      [(WARPS_1, I0, sha(expected), sha(wrong))],
      ops=("add", "max"),
      status="candidate",
      raw={(1, I0): {"compiled": wrong}},
    )
    report = self.compare()
    self.assertEqual(report["by_probe"][I0], {"agree": 1, "differ": 1})
    self.assertEqual(report["bodies"], {"differ": 1})
    [mismatch] = report["disagreements"]
    self.assertEqual(
      (mismatch["seed"], mismatch["path"], mismatch["output"], mismatch["input"]),
      (S0, "p2", "return", I0),
    )
    self.assertEqual(mismatch["operations"], ["add", "max"])
    self.assertEqual(
      mismatch["words"],
      [
        {
          "index": 2,
          "p0_bits": expected[8:12][::-1].hex(),
          "path_bits": struct.pack("<f", 99.0)[::-1].hex(),
        }
      ],
    )

  def test_a_mismatch_without_retained_output_has_no_words(self):
    self.kernel(S0, [(WARPS_1, I0, sha(self.p0_output(I0)), "0" * 64)])
    [mismatch] = self.compare()["disagreements"]
    self.assertIsNone(mismatch["words"])

  def test_a_sum_body_on_a_corner_probe_is_restricted(self):
    self.kernel(S0, [(WARPS_1, I6, "0" * 64, "1" * 64)], ops=("sum", "add"))
    report = self.compare()
    self.assertEqual(report["by_probe"][I6], {"restricted": 2})
    self.assertEqual(report["disagreements"], [])

  def test_a_sum_body_on_an_exact_probe_is_compared(self):
    self.kernel(S0, [(WARPS_1, I0, "0" * 64, sha(self.p0_output(I0)))], ops=("sum",))
    self.assertEqual(self.compare()["by_probe"][I0], {"differ": 1, "agree": 1})

  def test_refused_kernels_are_skipped_and_missing_bundles_unavailable(self):
    self.kernel_lines.append({"event": "kernel", "seed": S0 + 1, "status": "refused"})
    good = sha(self.p0_output(I0))
    self.kernel(S0, [(WARPS_1, I0, good, good)])
    # No bundle was retained for I3, so no configuration can supply P0 for it.
    missing = self.kernel_lines[-1]["pairs"][0]
    self.kernel_lines[-1]["pairs"].append({**missing, "input": "i3-unused-rows"})
    report = self.compare()
    self.assertEqual(report["kernels"], 1)
    self.assertEqual(report["by_probe"][I0], {"agree": 2})
    self.assertEqual(report["by_probe"]["i3-unused-rows"], {"unavailable": 2})

  def test_p0_undefined_behaviour_is_its_own_class(self):
    from test.lib import tensor_target_reference as reference

    ub = reference.ReferenceObservation(reference.UB, reason="non-finite value")
    self.assertEqual(compare_p0._classify(ub, "0" * 64, False), "p0_ub")
    self.assertEqual(compare_p0._classify(ub, "0" * 64, True), "restricted")

  def test_the_command_line_writes_the_report(self):
    good = sha(self.p0_output(I0))
    self.kernel(S0, [(WARPS_1, I0, good, good)])
    self.compare()
    out = self.scratch.parent / "P0.json"
    with contextlib.redirect_stdout(io.StringIO()):
      code = compare_p0.main(
        [str(self.shard), "--symiri", str(SYMIRI), "--out", str(out)]
      )
    self.assertEqual(code, 0)
    written = json.loads(out.read_text(encoding="utf-8"))
    self.assertEqual(written["schema"], compare_p0.SCHEMA)
    self.assertEqual(written["symiri_sha256"], sha(SYMIRI.read_bytes()))


if __name__ == "__main__":
  unittest.main()
