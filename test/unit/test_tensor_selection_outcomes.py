"""Solver termination reasons: reported causes stay distinct from UNKNOWN.

The classification is driven by a scripted backend rather than by a real query.
A query slow enough to time out on one machine is fast on another, so timing a
real solver would make the regression depend on the host.
"""

import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROBE = Path(
  os.environ.get(
    "REFRACTIR_SOLVER_OUTCOME_PROBE", ROOT / "build" / "bin" / "solver_outcome_probe"
  )
).resolve()


def probe(*arguments):
  return subprocess.run(
    [str(PROBE), *arguments],
    capture_output=True,
    text=True,
    check=False,
    timeout=60,
  )


class SolverOutcomeTests(unittest.TestCase):
  def setUp(self):
    self.assertTrue(PROBE.is_file(), f"solver outcome probe not found: {PROBE}")

  def solved(self, scripted, reason):
    result = probe(scripted, reason)
    self.assertEqual(result.returncode, 0, result.stderr)
    return result.stdout.strip()

  def test_reported_timeout_is_preserved(self):
    self.assertEqual(self.solved("unknown", "timeout"), "status=unknown reason=timeout")

  def test_unexplained_unknown_stays_unspecified(self):
    self.assertEqual(
      self.solved("unknown", "unspecified"), "status=unknown reason=unspecified"
    )

  def test_resource_limit_is_preserved(self):
    self.assertEqual(
      self.solved("unknown", "resource_limit"), "status=unknown reason=resource_limit"
    )

  def test_interruption_is_preserved(self):
    self.assertEqual(
      self.solved("unknown", "interrupted"), "status=unknown reason=interrupted"
    )

  def test_decided_results_carry_no_stale_reason(self):
    # The backend reports a timeout, but this query was decided, so nothing may
    # leak into the result.
    self.assertEqual(self.solved("sat", "timeout"), "status=sat reason=unspecified")
    self.assertEqual(self.solved("unsat", "timeout"), "status=unsat reason=unspecified")

  def test_unknown_diagnostic_string_stays_unspecified(self):
    result = probe(
      "--classify",
      "timeout",
      "canceled",
      "resource limits",
      "max. memory out",
      "unknown",
      "",
    )

    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(
      result.stdout.splitlines(),
      [
        "classify timeout -> timeout",
        "classify canceled -> interrupted",
        "classify resource limits -> resource_limit",
        "classify max. memory out -> resource_limit",
        # A diagnostic the mapping does not recognise is not a cause.
        "classify unknown -> unspecified",
        "classify  -> unspecified",
      ],
    )


class TensorProposalOutcomeTests(unittest.TestCase):
  """The same reasons reach the input policy's machine-readable outcome."""

  def setUp(self):
    self.assertTrue(PROBE.is_file(), f"solver outcome probe not found: {PROBE}")

  def proposed(self, scripted, reason):
    result = probe("--propose", scripted, reason)
    self.assertEqual(result.returncode, 0, result.stderr)
    return result.stdout.strip()

  def test_solver_timeout_becomes_a_sourced_timeout(self):
    self.assertEqual(
      self.proposed("unknown", "timeout"),
      "outcome=timeout timeout_source=solver unknown_reason=timeout",
    )

  def test_unexplained_unknown_is_not_a_timeout(self):
    self.assertEqual(
      self.proposed("unknown", "unspecified"),
      "outcome=unknown timeout_source=none unknown_reason=unspecified",
    )

  def test_resource_limit_and_interruption_stay_unknown_with_detail(self):
    self.assertEqual(
      self.proposed("unknown", "resource_limit"),
      "outcome=unknown timeout_source=none unknown_reason=resource_limit",
    )
    self.assertEqual(
      self.proposed("unknown", "interrupted"),
      "outcome=unknown timeout_source=none unknown_reason=interrupted",
    )

  def test_unsatisfiable_query_is_not_an_unknown(self):
    self.assertEqual(
      self.proposed("unsat", "timeout"),
      "outcome=unsat timeout_source=none unknown_reason=none",
    )


if __name__ == "__main__":
  unittest.main()
