"""FP diagnostics: model assumptions, allowed words, and outcome classification."""

import dataclasses
import unittest

from test.lib import tensor_fp_diagnostics as diagnostics


class FpModelTests(unittest.TestCase):
  def test_records_model_applicability_and_assumptions(self):
    # Every frozen analysis states a model that applies to its case and the
    # assumptions that make the allowed-result set meaningful.
    for case in (
      diagnostics.fp1_case(),
      diagnostics.fp2_case(),
      diagnostics.fp3_case(),
      diagnostics.fp4_case(),
    ):
      record = case.analyse(case.allowed_results[0]).record()
      model = record["model"]
      self.assertTrue(model["applicable"], msg=record["case"])
      self.assertIsNone(model["applicability_reason"], msg=record["case"])
      self.assertTrue(model["fully_justified"], msg=record["case"])
      self.assertTrue(model["name"], msg=record["case"])
      self.assertTrue(model["assumptions"], msg=record["case"])
      self.assertTrue(
        all(isinstance(item, str) and item for item in model["assumptions"]),
        msg=record["case"],
      )
      self.assertEqual(
        len(model["assumptions"]), len(set(model["assumptions"])), msg=record["case"]
      )

    # A non-finite operand cannot enter the reduction model, so applicability and
    # the reason are recorded instead of an allowed-result set.
    case = diagnostics.reduction_case(
      name="non-finite",
      program="reduction with a non-finite operand",
      words=(0x7F800000, 0x3F800000),
    )
    record = case.analyse((0x7F800000,)).record()
    self.assertFalse(record["model"]["applicable"])
    self.assertTrue(record["model"]["applicability_reason"])
    self.assertEqual(record["allowed_results"], [])
    self.assertEqual(record["classification"], diagnostics.NUMERIC_UNRESOLVED)


class FpAllowedResultTests(unittest.TestCase):
  def test_records_input_words_and_allowed_results(self):
    fp1 = diagnostics.fp1_case().analyse((0x00000000,)).record()
    self.assertEqual(fp1["input_words"], list(diagnostics.FP1_WORDS))
    self.assertEqual(fp1["allowed_results"], [[0x00000000], [0x3F800000]])
    self.assertGreater(len(fp1["allowed_results"]), 1)

    # The exact-result control has one modeled outcome.
    fp2 = diagnostics.fp2_case().analyse((0x00000000,)).record()
    self.assertEqual(fp2["allowed_results"], [[0x00000000]])
    self.assertEqual(len(fp2["allowed_results"]), 1)

    # Eight operands still enumerate tractably: 135135 unordered binary trees,
    # memoized by operand subset, produce five allowed sums for this input.
    fp3_case = diagnostics.fp3_case()
    fp3 = fp3_case.analyse(fp3_case.allowed_results[0]).record()
    self.assertEqual(fp3["input_words"], list(fp3_case.input_words))
    self.assertEqual(len(fp3["input_words"]), 8)
    self.assertEqual(
      [result[0] for result in fp3["allowed_results"]],
      [0x00000000, 0x3F800000, 0x40000000, 0x40400000, 0x40800000],
    )

    # The bit-preserving transpose records its exact output words, including both
    # signed zeros.
    fp4_case = diagnostics.fp4_case()
    fp4 = fp4_case.analyse(fp4_case.allowed_results[0]).record()
    self.assertEqual(fp4["input_words"], list(fp4_case.input_words))
    self.assertEqual(fp4["allowed_results"], [list(fp4_case.allowed_results[0])])
    self.assertIn(0x80000000, fp4["allowed_results"][0])
    self.assertIn(0x00000000, fp4["allowed_results"][0])


class FpClassificationTests(unittest.TestCase):
  def test_records_observed_result_and_classification(self):
    allowed = diagnostics.fp1_case().analyse((0x3F800000,)).record()
    self.assertEqual(allowed["observed_result"], [0x3F800000])
    self.assertEqual(allowed["classification"], diagnostics.IN_JUSTIFIED_SET)

    wrong = diagnostics.fp1_case().analyse((0x4B800000,)).record()
    self.assertEqual(wrong["observed_result"], [0x4B800000])
    self.assertEqual(wrong["classification"], diagnostics.OUTSIDE_JUSTIFIED_SET)
    self.assertIn(wrong["classification"], diagnostics.CLASSIFICATIONS)

    # The transpose model is exact, so a reversed output sequence is outside it.
    fp4_case = diagnostics.fp4_case()
    reversed_words = tuple(reversed(fp4_case.allowed_results[0]))
    self.assertEqual(
      fp4_case.analyse(reversed_words).classification,
      diagnostics.OUTSIDE_JUSTIFIED_SET,
    )

  def test_distinguishes_justified_set_failure_from_model_relative_unresolved(self):
    wrong = (0x4B800000,)
    case = diagnostics.fp1_case()
    justified = case.analyse(wrong).record()
    self.assertTrue(justified["model"]["fully_justified"])
    self.assertEqual(justified["classification"], diagnostics.OUTSIDE_JUSTIFIED_SET)

    # The same allowed-result set under a model whose assumptions are not fully
    # justified cannot turn an outside output into a strong candidate.
    incomplete_model = dataclasses.replace(
      case.model,
      name="all binary trees with an unproven finite-value assumption",
      assumptions=case.model.assumptions[:-1],
      fully_justified=False,
    )
    incomplete_case = dataclasses.replace(case, model=incomplete_model)
    unresolved = incomplete_case.analyse(wrong).record()
    self.assertFalse(unresolved["model"]["fully_justified"])
    self.assertEqual(unresolved["classification"], diagnostics.NUMERIC_UNRESOLVED)

    # Membership is consistency with the model, not permission: an allowed word
    # stays in the set while the model's justification is reported separately.
    member = incomplete_case.analyse((0x00000000,)).record()
    self.assertFalse(member["model"]["fully_justified"])
    self.assertEqual(member["classification"], diagnostics.IN_JUSTIFIED_SET)


class FpEdgeTests(unittest.TestCase):
  def test_signed_zero_signs_are_part_of_the_exact_transpose_result(self):
    case = diagnostics.fp4_case()
    expected = case.allowed_results[0]
    self.assertIn(0x80000000, expected)
    self.assertIn(0x00000000, expected)

    # Flipping both zero signs keeps the numerical values but changes the words,
    # so the bit-preserving model must reject the sequence.
    flipped = tuple(
      0x00000000 if word == 0x80000000 else 0x80000000 if word == 0x00000000 else word
      for word in expected
    )
    self.assertNotEqual(flipped, expected)
    self.assertEqual(
      case.analyse(expected).classification, diagnostics.IN_JUSTIFIED_SET
    )
    self.assertEqual(
      case.analyse(flipped).classification,
      diagnostics.OUTSIDE_JUSTIFIED_SET,
    )


class FpReportTests(unittest.TestCase):
  def test_records_tg_raw_difference_separately_from_screening_acceptance(self):
    # T and G differ while both pass the configured screening threshold. The raw
    # equality and the screening verdict are supplied and stored as a pair; the
    # record does not derive either from the other.
    disagree_but_accepted = (
      diagnostics.fp1_case()
      .analyse(
        (0x00000000,),
        target_evidence=diagnostics.TargetEvidence(
          raw_equal=False, screening_acceptance="accepted"
        ),
      )
      .record()
    )
    self.assertEqual(
      disagree_but_accepted["target_evidence"],
      {"raw_equal": False, "screening_acceptance": "accepted"},
    )
    self.assertEqual(
      disagree_but_accepted["classification"], diagnostics.IN_JUSTIFIED_SET
    )

    # The contrary pair is recorded unchanged as well. Acceptance is not
    # reconstructed from equality.
    equal_but_rejected = (
      diagnostics.fp1_case()
      .analyse(
        (0x00000000,),
        target_evidence=diagnostics.TargetEvidence(
          raw_equal=True, screening_acceptance="rejected"
        ),
      )
      .record()
    )
    self.assertEqual(
      equal_but_rejected["target_evidence"],
      {"raw_equal": True, "screening_acceptance": "rejected"},
    )


if __name__ == "__main__":
  unittest.main()
