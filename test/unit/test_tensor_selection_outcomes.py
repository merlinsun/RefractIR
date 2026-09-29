"""Solver termination reasons: reported causes stay distinct from UNKNOWN.

The classification is driven by a scripted backend rather than by a real query.
A query slow enough to time out on one machine is fast on another, so timing a
real solver would make the regression depend on the host.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROBE = Path(
  os.environ.get(
    "REFRACTIR_SOLVER_OUTCOME_PROBE", ROOT / "build" / "bin" / "solver_outcome_probe"
  )
).resolve()


def read_summary(directory):
  return json.loads((directory / "generation.json").read_text())


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

  def test_a_diagnostic_that_merely_mentions_a_cause_is_not_one(self):
    """Only a complete documented diagnostic names a cause.

    A backend report is specific. Text that happens to contain a word is not a
    statement that the event occurred, so it stays unspecified.
    """
    result = probe(
      "--classify",
      "unknown timeout configuration",
      "not interrupted",
      "resource limit",
      "Canceled",
      "timeout ",
      "cancel",
    )

    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertTrue(
      all(line.endswith("-> unspecified") for line in result.stdout.splitlines()),
      result.stdout,
    )

  def test_sampling_keeps_a_reported_reason(self):
    """A worker that runs no query must not publish its default result.

    One sample across four threads leaves three workers with nothing to
    report; if one of them published anyway it would replace the reason the
    fourth reported, depending on the interleaving.
    """
    for _ in range(5):
      result = probe("--sample", "4", "1", "timeout")

      self.assertEqual(result.returncode, 0, result.stderr)
      self.assertEqual(result.stdout.strip(), "reason=timeout")


class SymirsolveOutcomeRecordTests(unittest.TestCase):
  """The CLI keeps the reason available to a consumer that cannot read prose.

  The console text and the exit code are unchanged; the record is a separate
  output path, so nothing that already parses them has to change.
  """

  SYMIRSOLVE = Path(
    os.environ.get("REFRACTIR_SYMIRSOLVE", ROOT / "symirsolve")
  ).resolve()

  def solve(self, source, *arguments):
    directory = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, directory, True)
    program = Path(directory) / "probe.sir"
    program.write_text(source)
    record = Path(directory) / "outcome.json"
    result = subprocess.run(
      [
        str(self.SYMIRSOLVE),
        str(program),
        "--main",
        "@main",
        "--path",
        "^entry",
        "--emit-outcome",
        str(record),
        *arguments,
      ],
      capture_output=True,
      text=True,
      check=False,
      timeout=120,
    )
    return result, record

  def setUp(self):
    self.assertTrue(
      self.SYMIRSOLVE.is_file(), f"symirsolve not found: {self.SYMIRSOLVE}"
    )

  def test_a_satisfiable_query_records_sat(self):
    result, record = self.solve("""fun @main() : i32 {
  sym %?x : value i32 in [10, 20];
^entry:
  require %?x == 15;
  ret 0;
}
""")

    self.assertEqual(result.stdout.strip(), "SAT", result.stderr)
    self.assertEqual(json.loads(record.read_text()), {"status": "sat"})

  def test_an_unsatisfiable_query_records_unsat(self):
    result, record = self.solve("""fun @main() : i32 {
  sym %?x : value i32 in [10, 20];
^entry:
  require %?x == 5;
  ret 0;
}
""")

    self.assertEqual(result.stdout.strip(), "UNSAT", result.stderr)
    self.assertEqual(json.loads(record.read_text()), {"status": "unsat"})

  def test_an_undecided_query_carries_a_typed_reason(self):
    """A one-millisecond limit may or may not stop this query, so the test
    asserts the implication: an UNKNOWN record always carries a reason from the
    closed set, and a decided record agrees with the console."""
    result, record = self.solve(
      """fun @main() : i32 {
  sym %?x : value i32 in [10, 20];
^entry:
  require %?x == 15;
  ret 0;
}
""",
      "--timeout-ms",
      "1",
    )
    payload = json.loads(record.read_text())

    if payload["status"] == "unknown":
      self.assertEqual(result.stdout.strip(), "UNKNOWN")
      self.assertIn(
        payload["reason"],
        {"unspecified", "timeout", "resource_limit", "interrupted"},
      )
    else:
      self.assertEqual(payload["status"], result.stdout.strip().lower())
      self.assertNotIn("reason", payload)

  def test_console_and_record_agree(self):
    result, record = self.solve("""fun @main() : i32 {
  sym %?x : value i32 in [10, 20];
^entry:
  require %?x == 5;
  ret 0;
}
""")
    payload = json.loads(record.read_text())

    self.assertEqual(result.stdout.strip().lower(), payload["status"])


class TensorAttemptRecordTests(unittest.TestCase):
  """The record a refused attempt leaves behind, driven by a scripted backend.

  Waiting for a real query to be undecided would make the regression depend on
  solver speed, so the generator takes an injected backend and the outcome is
  chosen rather than waited for.
  """

  def setUp(self):
    self.assertTrue(PROBE.is_file(), f"solver outcome probe not found: {PROBE}")

  def attempt_record(self, scripted, reason, seed="2"):
    directory = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, directory, True)
    result = probe("--attempt-record", scripted, reason, directory, seed)
    self.assertEqual(result.returncode, 0, result.stderr)
    return result.stdout.strip(), read_summary(Path(directory))

  def test_a_reported_timeout_is_recorded_with_its_source(self):
    status, summary = self.attempt_record("unknown", "timeout")

    self.assertEqual(status, "status=input_policy_timeout")
    self.assertEqual(summary["status"], "input_policy_timeout")
    self.assertEqual(len(summary["attempts"]), 1)
    attempt = summary["attempts"][0]
    self.assertEqual(attempt["timeout_source"], "solver")
    self.assertEqual(attempt["unknown_reason"], "timeout")
    self.assertEqual(attempt["outcome"], "input_policy_timeout")
    self.assertEqual(attempt["policy"], "exact")

  def test_a_refused_attempt_names_the_graph_it_refused(self):
    """A seed names the graph for one builder revision; the digests name it
    across revisions, and the path says what was executed."""
    _, summary = self.attempt_record("unknown", "timeout")
    attempt = summary["attempts"][0]

    self.assertEqual(len(attempt["candidate_digest"]), 16)
    self.assertEqual(attempt["path"], "^entry")
    self.assertEqual(attempt["domain"], "-1,2")
    self.assertGreater(attempt["symbolic_elements"], 0)
    self.assertEqual(attempt["seed"], 2)

  def test_an_unexplained_unknown_is_not_a_timeout(self):
    status, summary = self.attempt_record("unknown", "unspecified")

    self.assertEqual(status, "status=input_policy_unknown")
    attempt = summary["attempts"][0]
    self.assertEqual(attempt["unknown_reason"], "unspecified")
    self.assertNotIn("timeout_source", attempt)

  def test_an_unsatisfiable_query_stays_unsat(self):
    status, summary = self.attempt_record("unsat", "timeout")

    self.assertEqual(status, "status=input_policy_unsat")
    attempt = summary["attempts"][0]
    self.assertNotIn("timeout_source", attempt)
    self.assertNotIn("unknown_reason", attempt)


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
