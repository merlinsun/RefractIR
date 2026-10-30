"""Enclosure widths over retained sweep bodies and the D2 decision (plan W2)."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from test.lib import tensor_free_form_case

try:
  import numpy  # noqa: F401

  from test.unit.test_tensor_enclosure import REGION, SPECIFIED, SUMMED, proposals
except ImportError:  # pragma: no cover - numpy is a test-environment dependency
  numpy = None

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
  "enclose_sweep_bodies",
  _ROOT / "test" / "tensor" / "campaigns" / "enclose_sweep_bodies.py",
)

S0 = 30000000
WARPS_1 = {"num_warps": 1, "num_stages": 1}


@unittest.skipUnless(numpy is not None, "numpy is unavailable")
class EncloseSweepBodiesTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.shard = Path(temporary.name) / "shard-0"
    self.lines = []
    self.module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(self.module)

  def kernel(self, seed, source, names):
    by_name = {proposal.probe: proposal for proposal in proposals()}
    root = self.shard / "kernels" / f"seed-{seed}"
    pairs = []
    for name in names:
      bundle = root / "pairs" / self.module.sweep._pair_name(WARPS_1, name) / "bundle"
      tensor_free_form_case.build_bundle(
        bundle, region=REGION, source=source, proposal=by_name[name]
      )
      pairs.append({"configuration": WARPS_1, "input": name})
    self.lines.append(
      {"event": "kernel", "seed": seed, "status": "agreement", "pairs": pairs}
    )

  def report(self):
    self.shard.mkdir(parents=True, exist_ok=True)
    (self.shard / "run-log.jsonl").write_text(
      "".join(json.dumps(line) + "\n" for line in self.lines)
    )
    return self.module.enclose_shards([self.shard])

  def test_integer_bodies_have_width_zero_and_pass_d2(self):
    self.kernel(S0, SUMMED, ["i0-base"])
    self.kernel(S0 + 1, SPECIFIED, ["i0-base"])
    report = self.report()
    self.assertEqual([body["max_width"] for body in report["bodies"]], [0, 0])
    self.assertEqual(report["d2"]["median_body_width"], 0)
    self.assertTrue(report["d2"]["passes"])

  def test_a_cancelling_sum_widens_its_body_and_is_reported_per_input(self):
    self.kernel(S0, SUMMED, ["i0-base", "i6-cancel"])
    report = self.report()
    [body] = report["bodies"]
    self.assertGreater(body["max_width"], 0)
    self.assertEqual(body["widths_by_input"]["i0-base"], 0)
    self.assertEqual(body["max_width"], body["widths_by_input"]["i6-cancel"])
    self.assertEqual(report["by_input"]["i0-base"]["max"], 0)

  def test_the_d2_rule_fails_on_a_wide_body(self):
    # Cancellation leaves results near zero, where a few units of reordering
    # error span many f32 steps.
    self.kernel(S0, SUMMED, ["i6-cancel"])
    report = self.report()
    self.assertGreater(report["d2"]["max_body_width"], 64)
    self.assertFalse(report["d2"]["passes"])
    self.assertEqual(report["d2"]["rule"], "median <= 4 ulp and max <= 64 ulp")

  def test_an_unsupported_body_is_counted_apart(self):
    self.kernel(S0, SUMMED.replace("tensor sum", "tensor rmax"), ["i0-base"])
    self.kernel(S0 + 1, SPECIFIED.replace("tensor sub", "tensor div"), ["i0-base"])
    report = self.report()
    self.assertEqual(report["unsupported"], 1)
    self.assertEqual(len(report["bodies"]), 1)

  def test_a_skipped_pair_is_not_missing(self):
    self.kernel(S0, SUMMED, ["i0-base"])
    self.lines[-1]["pairs"].append(
      {
        "configuration": WARPS_1,
        "input": "steer-tie",
        "p1": {"status": "skipped"},
        "p2": {"status": "skipped"},
      }
    )
    report = self.report()
    self.assertEqual(report["missing"], 0)
    self.assertEqual(report["bodies"][0]["widths_by_input"], {"i0-base": 0})

  def test_the_command_line_writes_the_report(self):
    self.kernel(S0, SUMMED, ["i0-base"])
    self.report()
    out = self.shard.parent / "ENCLOSURE.json"
    with contextlib.redirect_stdout(io.StringIO()):
      code = self.module.main([str(self.shard), "--out", str(out)])
    self.assertEqual(code, 0)
    self.assertEqual(json.loads(out.read_text())["schema"], self.module.SCHEMA)


if __name__ == "__main__":
  unittest.main()
