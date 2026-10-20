"""Three-path reporting: one reference, one emission, two targets.

The tests drive `test.lib.tensor_three_way` with fake runners, so every matrix
row is exercised without Triton, torch, or a GPU. The reference side is a
`ReferenceResult` from `test.lib.tensor_reference`, built here directly so the
tests pin what the reporting layer does with each classification. A path runner
is any callable taking the shared emission and returning a `PathRun`.
"""

import unittest

from test.lib import tensor_reference, tensor_three_way


def _case():
  return object()


def _reference(classification="reference_ok", data=b"\x00\x00\x80\x3f" * 2):
  """A ReferenceResult with only the fields the reporting layer reads."""
  return tensor_reference.ReferenceResult(
    case=None,
    profile="v2",
    entry="kernel",
    mode="numeric",
    directory=None,
    classification=classification,
    reason="test",
    replays=(),
    expected_sha256="0" * 64,
    reference=data if classification == tensor_reference.REFERENCE_OK else None,
  )


def _run(status="completed", output=None, **detail):
  return tensor_three_way.PathRun(status=status, output=output, detail=dict(detail))


class ThreeWayAgreementTests(unittest.TestCase):
  def test_three_paths_that_agree_report_agreement_and_completed_statuses(self):
    # R, T and G all produce the same bytes: nothing to diagnose, and every
    # path ran.
    reference = _reference()
    data = reference.reference
    result = tensor_three_way.run(
      case=_case(),
      reference=reference,
      interpreter=lambda emission: _run(output=data),
      compiled=lambda emission: _run(output=data),
    )
    self.assertEqual(result.execution, {"P0": "completed", "P1": "completed", "P2": "completed"})
    self.assertEqual(result.observations["R-T"], "equal")
    self.assertEqual(result.observations["R-G"], "equal")
    self.assertEqual(result.observations["T-G"], "equal")
    self.assertEqual(result.diagnosis, "agreement")
    self.assertTrue(result.interpretation_available)

  def test_a_reference_that_is_not_accepted_suppresses_both_targets(self):
    # P0 did not establish an accepted reference, so P1 and P2 are not_run -
    # not failed, and not unsupported.
    called = []
    result = tensor_three_way.run(
      case=_case(),
      reference=_reference(classification=tensor_reference.REFERENCE_MISMATCH),
      interpreter=lambda emission: called.append("P1") or _run(output=b"\x01"),
      compiled=lambda emission: called.append("P2") or _run(output=b"\x01"),
    )
    self.assertEqual(called, [])
    self.assertEqual(result.execution, {"P0": "reference_mismatch", "P1": "not_run", "P2": "not_run"})
    self.assertEqual(result.observations, {"R-T": "unavailable", "R-G": "unavailable", "T-G": "unavailable"})
    self.assertEqual(result.diagnosis, "reference drift")


class ThreeWayTargetTests(unittest.TestCase):
  def test_an_unsupported_interpreter_does_not_suppress_the_compiled_target(self):
    # The plan requires G to run even when T is unsupported, and to report the
    # two-way outcome when G satisfies its contract.
    reference = _reference()
    result = tensor_three_way.run(
      case=_case(),
      reference=reference,
      interpreter=lambda emission: _run(status="unsupported", detail={"reason": "no interpreter"}),
      compiled=lambda emission: _run(output=reference.reference),
    )
    self.assertEqual(result.execution["P1"], "unsupported")
    self.assertEqual(result.execution["P2"], "completed")
    self.assertEqual(result.observations["R-G"], "equal")
    self.assertEqual(result.observations["T-G"], "unavailable")
    self.assertEqual(result.diagnosis, "two_way_passed_interpreter_unavailable")

  def test_an_interpreter_disagreement_does_not_suppress_the_compiled_target(self):
    # T differs from R; G agrees. That is an interpreter candidate, and G must
    # still have run.
    reference = _reference()
    result = tensor_three_way.run(
      case=_case(),
      reference=reference,
      interpreter=lambda emission: _run(output=b"\x02\x00\x00\x00"),
      compiled=lambda emission: _run(output=reference.reference),
    )
    self.assertEqual(result.execution["P2"], "completed")
    self.assertEqual(result.observations["R-T"], "different")
    self.assertEqual(result.observations["R-G"], "equal")
    self.assertEqual(result.diagnosis, "interpreter candidate")
    self.assertTrue(result.detail["P1"]["output_sha256"])

  def test_a_storage_violation_stays_a_candidate_outcome(self):
    # A guard/gap write is a candidate, not a plain failure, and it keeps the
    # slots it names.
    reference = _reference()
    result = tensor_three_way.run(
      case=_case(),
      reference=reference,
      interpreter=lambda emission: _run(output=reference.reference),
      compiled=lambda emission: _run(
        status="failed",
        output=None,
        storage_violation=["gap-0", "suffix-guard-1"],
        reason="wrote outside the logical output mapping",
      ),
    )
    self.assertEqual(result.execution["P2"], "failed")
    self.assertEqual(result.diagnosis, "compiled-path candidate")
    self.assertEqual(result.candidates[0]["kind"], "storage_violation")
    self.assertEqual(result.candidates[0]["slots"], ["gap-0", "suffix-guard-1"])


class ThreeWayNumericalTests(unittest.TestCase):
  def test_target_disagreement_accepted_only_by_screening_is_unresolved(self):
    # T and G differ from each other while each is inside the R-based screening
    # tolerance. The plan forbids calling that permitted: two acceptable errors
    # can point in opposite directions.
    reference = _reference()
    result = tensor_three_way.run(
      case=_case(),
      reference=reference,
      interpreter=lambda emission: _run(output=b"\x00\x00\xc0\x3f" * 2),
      compiled=lambda emission: _run(output=b"\x9a\x99\x19\x3f" * 2),
      atol=(1, 1),
      rtol=(1, 1),
    )
    self.assertEqual(result.observations["T-G"], "different")
    self.assertEqual(result.contract["R-T"], "accepted")
    self.assertEqual(result.contract["R-G"], "accepted")
    self.assertEqual(result.contract["T-G"], "not_applicable")
    self.assertEqual(result.diagnosis, "unresolved numerical difference")
    self.assertFalse(result.permitted)

  def test_an_exact_contract_mismatch_is_not_permitted(self):
    # Under an exact contract a differing G is a candidate, never a permitted
    # numerical difference.
    reference = _reference()
    result = tensor_three_way.run(
      case=_case(),
      reference=reference,
      interpreter=lambda emission: _run(output=reference.reference),
      compiled=lambda emission: _run(output=b"\x00\x00\x00\x80"),
      exact=True,
    )
    self.assertEqual(result.contract["R-G"], "rejected")
    self.assertEqual(result.diagnosis, "compiled-path candidate")
    self.assertFalse(result.permitted)


if __name__ == "__main__":
  unittest.main()
