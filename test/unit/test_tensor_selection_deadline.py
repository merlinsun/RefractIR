"""One deadline covers selection, replay, and response validation.

The supervisor is driven with a controllable worker, so every terminal outcome
is reached deliberately rather than by waiting for a solver to be slow. The
fixture reports the stage it reached through the same file the real worker
writes, and delegates to the real worker when a test needs an honest result.
"""

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROBE = Path(
  os.environ.get(
    "REFRACTIR_SOLVER_OUTCOME_PROBE", ROOT / "build" / "bin" / "solver_outcome_probe"
  )
).resolve()
FIXTURE = (ROOT / "test" / "unit" / "fixtures" / "selection_worker.sh").resolve()

# The supervisor returns as soon as its deadline expires; the margin covers
# process startup and cleanup on a loaded machine.
OUTER_MARGIN_MS = 10_000


def parse(line):
  fields = {}
  for item in line.split():
    key, _, value = item.partition("=")
    fields[key] = value
  return fields


def select(worker, budget_ms, mode="valid", stall_s=30, extra=()):
  env = dict(os.environ)
  env["FIXTURE_MODE"] = mode
  env["FIXTURE_STALL_S"] = str(stall_s)
  env["FIXTURE_REAL"] = str(PROBE)
  result = subprocess.run(
    [str(PROBE), "--select", str(worker), str(budget_ms), *extra],
    capture_output=True,
    text=True,
    check=False,
    env=env,
    timeout=(budget_ms + OUTER_MARGIN_MS) / 1000 + 60,
  )
  if result.returncode != 0:
    raise AssertionError(
      f"selection probe failed: {result.returncode}: {result.stderr}"
    )
  return parse(result.stdout.strip())


def select_twice(modes, first_worker, first_budget, second_worker, second_budget):
  """Run two selections in one process, each child with its own behaviour."""
  with tempfile.TemporaryDirectory() as directory:
    modes_file = Path(directory) / "modes.txt"
    modes_file.write_text("\n".join(modes) + "\n")
    env = dict(os.environ)
    env["FIXTURE_MODES_FILE"] = str(modes_file)
    env["FIXTURE_STALL_S"] = "30"
    env["FIXTURE_REAL"] = str(PROBE)
    result = subprocess.run(
      [
        str(PROBE),
        "--select-twice",
        str(first_worker),
        str(first_budget),
        str(second_worker),
        str(second_budget),
      ],
      capture_output=True,
      text=True,
      check=False,
      env=env,
      timeout=(first_budget + second_budget + OUTER_MARGIN_MS) / 1000 + 60,
    )
  if result.returncode != 0:
    raise AssertionError(
      f"selection probe failed: {result.returncode}: {result.stderr}"
    )
  lines = result.stdout.strip().splitlines()
  return parse(lines[0][len("first ") :]), parse(lines[1][len("second ") :])


class SelectionDeadlineTests(unittest.TestCase):
  def setUp(self):
    self.assertTrue(PROBE.is_file(), f"solver outcome probe not found: {PROBE}")
    self.assertTrue(FIXTURE.is_file(), f"selection worker fixture not found: {FIXTURE}")

  def timed_out(self, mode, budget_ms=300):
    result = select(FIXTURE, budget_ms, mode=mode)

    self.assertEqual(result["outcome"], "timeout")
    self.assertEqual(result["timeout_source"], "selection_deadline")
    self.assertEqual(result["replay"], "no")
    self.assertEqual(result["expected_bytes"], "0")
    self.assertGreaterEqual(int(result["elapsed_ms"]), budget_ms)
    self.assertLessEqual(int(result["elapsed_ms"]), budget_ms + OUTER_MARGIN_MS)
    # Cleanup is measured, not assumed.
    self.assertGreaterEqual(int(result["cleanup_us"]), 0)
    return result

  def test_a_worker_stalled_before_solving_reports_no_stage(self):
    result = self.timed_out("stall_before")

    self.assertEqual(result["stage"], "unknown_stage")

  def test_a_worker_stalled_in_the_solver_reports_that_stage(self):
    result = self.timed_out("stall_solver")

    self.assertEqual(result["stage"], "solver")

  def test_a_worker_stalled_during_replay_reports_that_stage(self):
    result = self.timed_out("stall_replay")

    self.assertEqual(result["stage"], "replay")

  def test_a_result_finishing_after_the_deadline_is_not_accepted(self):
    # The worker would produce a real result, but only after the deadline.
    result = select(FIXTURE, 300, mode="slow_valid", stall_s=3)

    self.assertEqual(result["outcome"], "timeout")
    self.assertEqual(result["timeout_source"], "selection_deadline")
    self.assertFalse(result["replay"] == "yes")

  def test_a_valid_result_inside_the_deadline_is_accepted(self):
    result = select(FIXTURE, 20_000, mode="valid")

    self.assertEqual(result["outcome"], "accepted")
    self.assertEqual(result["timeout_source"], "none")
    self.assertEqual(result["replay"], "yes")
    self.assertGreater(int(result["expected_bytes"]), 0)

  def test_a_crashed_worker_leaves_no_accepted_case(self):
    result = select(FIXTURE, 20_000, mode="crash_partial")

    self.assertEqual(result["outcome"], "invalid")
    self.assertEqual(result["replay"], "no")
    self.assertEqual(result["expected_bytes"], "0")

  def test_a_widened_domain_does_not_admit_the_witness(self):
    """The effective domain is the policy's, not the one the response claims.

    A response that widens the interval can make its own leaves look admissible;
    what matters is the interval the caller's policy fixes.
    """
    # The domain is left to the policy, as production composition does: an
    # explicit interval would be caught by the older agreement rule alone.
    result = select(FIXTURE, 20_000, mode="domain_widened", extra=("implicit-domain",))

    self.assertEqual(result["outcome"], "invalid")

  def test_an_interval_with_trailing_text_is_malformed(self):
    """A number followed by text is not a number.

    `-1junk` reads as -1 to a prefix parser, which would certify an interval the
    response never stated.
    """
    result = select(FIXTURE, 20_000, mode="domain_trailing")

    self.assertEqual(result["outcome"], "invalid")

  def test_a_refusal_keeps_the_size_the_request_declared(self):
    """A refusal reports the request's symbolic size, not the worker's claim.

    A worker that refused before solving has no search to describe, and one
    that reports a different number is describing another request.
    """
    result = select(FIXTURE, 20_000, mode="refusal_wrong_size")

    self.assertEqual(result["outcome"], "unsat")
    self.assertNotEqual(result["symbolic"], "999")
    self.assertEqual(result["symbolic"], "2")

  def test_dependence_evidence_for_more_inputs_than_exist_is_rejected(self):
    """The evidence is counted before it is indexed.

    A single-input program that receives two dependence records must be
    refused rather than reading past the signature it is checked against.
    """
    result = select(FIXTURE, 20_000, mode="excess_dependence")

    self.assertEqual(result["outcome"], "invalid")

  def test_a_timeout_still_names_the_graph_it_abandoned(self):
    """Expiry is an outcome about a known request, not an empty refusal."""
    result = self.timed_out("stall_solver")

    self.assertNotEqual(result["digest"], "none")
    self.assertEqual(result["domain"], "-1,2")
    self.assertGreater(int(result["symbolic"]), 0)

  def test_a_rejected_response_still_names_the_graph_it_refused(self):
    result = select(FIXTURE, 20_000, mode="domain_widened")

    self.assertNotEqual(result["digest"], "none")
    self.assertEqual(result["domain"], "-1,2")

  def test_a_worker_that_fails_after_answering_is_not_accepted(self):
    """A complete response is not enough: the worker has to exit cleanly.

    A worker that crashed after writing its answer may have written it from a
    state it could not finish from, so the answer is not evidence.
    """
    result = select(FIXTURE, 20_000, mode="exit_13")

    self.assertEqual(result["outcome"], "invalid")

  def test_a_leaf_outside_the_stated_domain_is_rejected(self):
    """The request states the domain, so the parent checks the leaves itself.

    A response can describe any domain it likes; what makes it a witness is
    that every leaf lies in the interval the caller asked about.
    """
    result = select(FIXTURE, 20_000, mode="domain_violation")

    self.assertEqual(result["outcome"], "invalid")

  def test_terminating_a_worker_also_ends_its_descendants(self):
    """The owned process group is signalled, not only the worker itself.

    The helper writes a marker well after the deadline; if the group survived
    the worker, the marker appears.
    """
    with tempfile.TemporaryDirectory() as directory:
      marker = Path(directory) / "descendant-survived"
      env = dict(os.environ)
      env["FIXTURE_MODE"] = "spawn_descendant"
      env["FIXTURE_STALL_S"] = "30"
      env["FIXTURE_REAL"] = str(PROBE)
      env["FIXTURE_DESCENDANT"] = str(marker)
      completed = subprocess.run(
        [str(PROBE), "--select", str(FIXTURE), "300"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=(300 + OUTER_MARGIN_MS) / 1000 + 60,
      )
      self.assertEqual(completed.returncode, 0, completed.stderr)
      self.assertEqual(parse(completed.stdout.strip())["outcome"], "timeout")

      # The helper would write the marker five seconds after it started.
      time.sleep(6)
      self.assertFalse(marker.exists(), "a descendant outlived the selection deadline")

  def test_an_unavailable_worker_is_rejected(self):
    result = select(ROOT / "test" / "unit" / "fixtures" / "does-not-exist.sh", 20_000)

    self.assertEqual(result["outcome"], "invalid")
    self.assertEqual(result["replay"], "no")

  def test_a_timed_out_child_does_not_affect_the_next_selection(self):
    """Terminating one child must not touch the parent or the next child."""
    first, second = select_twice(
      ["stall_solver", "valid"], FIXTURE, 300, FIXTURE, 20_000
    )

    self.assertEqual(first["outcome"], "timeout")
    self.assertEqual(first["timeout_source"], "selection_deadline")
    self.assertEqual(second["outcome"], "accepted")
    self.assertEqual(second["replay"], "yes")
    self.assertGreater(int(second["expected_bytes"]), 0)


if __name__ == "__main__":
  unittest.main()
