"""Allowed-result FP diagnostics at the report boundary.

Each analysis freezes a program's input words, states a model of the outputs the
program may produce, enumerates that model's allowed results as binary32 words,
and classifies a caller-supplied observed result. The record keeps the model and
its assumptions beside the observation, so a report can state which claim the
observation was tested against.

Membership means "consistent with this model", not absolute permission. An
observed result inside the allowed set is `in_justified_set` even when the model
is not fully justified; `fully_justified` says whether the assumptions cover
identity, rounding, subnormal, and exceptional-value behaviour for this input.
An observed result outside a fully justified model is `outside_justified_set`, a
strong candidate. Outside an incompletely justified model it is
`numeric_unresolved`.

The four frozen cases are FP1 to FP4 from the three-path discovery plan. FP1 to
FP3 enumerate every binary tree over the reduction's frozen input words with
`tensor_fp32_reference`, which rounds each addition once to binary32
round-to-nearest-even. The tree's child order is immaterial because addition is
commutative for finite operands, so the model also covers parenthesizations of
permuted operands. FP4 compares a transpose's words exactly and treats the sign
of zero as data.

This module is not a comparator. The T-G raw equality and the configured
screening verdict arrive as separate `TargetEvidence` inputs, are recorded side
by side, and neither is derived from the other.
"""

from dataclasses import dataclass

from test.lib.tensor_fp32_reference import reduction_f32_rne_outcomes

IN_JUSTIFIED_SET = "in_justified_set"
OUTSIDE_JUSTIFIED_SET = "outside_justified_set"
NUMERIC_UNRESOLVED = "numeric_unresolved"
CLASSIFICATIONS = (IN_JUSTIFIED_SET, OUTSIDE_JUSTIFIED_SET, NUMERIC_UNRESOLVED)

SCREENING_ACCEPTANCE = ("accepted", "rejected", "unavailable")

# The plan's rounding-sensitive four-word cancellation sum:
# [16777216.0, 1.0, -16777216.0, 0.0]. Adding the large terms first yields 0.0;
# cancelling them first yields 1.0.
FP1_WORDS = (0x4B800000, 0x3F800000, 0xCB800000, 0x00000000)

# An exact-result control, [1.0, -1.0, 1.0, -1.0]: every partial sum is a small
# integer, so every tree produces positive zero.
FP2_WORDS = (0x3F800000, 0xBF800000, 0x3F800000, 0xBF800000)

# A repeated and mixed eight-word sum. Eight operands admit (2*8-3)!! = 135135
# unordered binary trees, and memoizing by operand subset leaves at most 2^8
# subset states.
FP3_WORDS = (
  0x4B800000,
  0x3F800000,
  0xCB800000,
  0x3F800000,
  0x3F800000,
  0xCB800000,
  0x4B800000,
  0x00000000,
)

# A 2x2 row-major transpose whose input and output both carry a negative and a
# positive zero, so a canonicalising copy changes the output words. Input words
# are [-0.0, 1.0, -1.0, 0.0]; transposed output words are
# [-0.0, -1.0, 1.0, 0.0].
FP4_INPUT_WORDS = (0x80000000, 0x3F800000, 0xBF800000, 0x00000000)
FP4_TRANSPOSED_WORDS = (0x80000000, 0xBF800000, 0x3F800000, 0x00000000)

_REDUCTION_ASSUMPTIONS = (
  "the program computes one binary32 scalar sum of exactly the frozen input words",
  "each addition rounds its exact sum once to binary32 round-to-nearest-even",
  (
    "every binary tree over the frozen words is legal, including non-contiguous "
    "pairings; child order is immaterial because addition is commutative"
  ),
  (
    "operands and partial sums are finite; subnormal results are not flushed and "
    "signed zero follows round-to-nearest-even"
  ),
)

_EXACT_CANCELLATION_ASSUMPTIONS = (
  (
    "every partial sum is an integer of magnitude at most two, exactly "
    "representable in binary32"
  ),
  (
    "exact cancellation of opposite-signed finite operands produces positive "
    "zero under round-to-nearest-even"
  ),
)

_EIGHT_OPERAND_ASSUMPTIONS = (
  (
    "eight operands admit (2*8-3)!! = 135135 unordered binary trees; memoizing "
    "the enumeration by operand subset leaves at most 2^8 subset states"
  ),
  (
    "a repeated value is still one recorded operand; repetition does not add or "
    "remove a legal tree, and position-distinct subsets are memoized separately"
  ),
)

_EXACT_TRANSPOSE_ASSUMPTIONS = (
  (
    "the program copies or transposes words without arithmetic, conversion, or "
    "canonicalisation"
  ),
  "signed zero is data: 0x00000000 and 0x80000000 are distinct output words",
  "the output holds the same number of binary32 words as the input",
  "the frozen 2x2 transpose map assigns each input word to one output position",
)


def _check_word(word):
  """Refuse a value that is not an unsigned binary32 word."""
  if type(word) is not int or not 0 <= word <= 0xFFFFFFFF:
    raise ValueError("a diagnostic word must be an unsigned 32-bit integer")


def _result_words(value):
  """The observed output as a nonempty tuple of binary32 words."""
  if type(value) is int:
    words = (value,)
  elif isinstance(value, (tuple, list)) and all(type(word) is int for word in value):
    words = tuple(value)
  else:
    raise TypeError("an observed result must be a word or a sequence of words")
  if not words:
    raise ValueError("an observed result must contain at least one word")
  for word in words:
    _check_word(word)
  return words


def _finite_words(words):
  """Whether every input word is a finite binary32 value, and why not."""
  for word in words:
    _check_word(word)
    if (word >> 23) & 0xFF == 0xFF:
      return False, f"input word 0x{word:08x} is not a finite binary32 value"
  return True, None


@dataclass(frozen=True)
class Model:
  """A stated model of the outputs a frozen program may produce.

  `fully_justified` is a claim about the assumptions, not about an observation:
  it is true only when identity, rounding, subnormal, and exceptional-value
  behaviour are all covered for this input.
  """

  name: str
  assumptions: tuple[str, ...]
  fully_justified: bool

  def __post_init__(self):
    if not isinstance(self.name, str) or not self.name:
      raise ValueError("a model needs a nonempty name")
    if not isinstance(self.assumptions, tuple) or not self.assumptions:
      raise ValueError("a model needs at least one assumption")
    if any(not isinstance(item, str) or not item for item in self.assumptions):
      raise ValueError("model assumptions must be nonempty strings")
    if type(self.fully_justified) is not bool:
      raise TypeError("fully_justified must be a bool")


@dataclass(frozen=True)
class TargetEvidence:
  """T-G raw equality and the configured screening verdict, kept independent.

  `raw_equal` is raw byte equality of the two targets. `screening_acceptance` is
  the configured comparator's verdict for the targets against the reference; it
  is not a verdict about the target pair. Both values are stored exactly as the
  caller supplied them.
  """

  raw_equal: bool
  screening_acceptance: str

  def __post_init__(self):
    if type(self.raw_equal) is not bool:
      raise TypeError("raw target equality must be a bool")
    if self.screening_acceptance not in SCREENING_ACCEPTANCE:
      raise ValueError(
        "screening acceptance must be one of " + ", ".join(SCREENING_ACCEPTANCE)
      )

  def record(self):
    return {
      "raw_equal": self.raw_equal,
      "screening_acceptance": self.screening_acceptance,
    }


@dataclass(frozen=True)
class FrozenCase:
  """One frozen FP program, its model, and the model's allowed output words."""

  name: str
  program: str
  input_words: tuple[int, ...]
  allowed_results: tuple[tuple[int, ...], ...]
  model: Model
  applicable: bool = True
  applicability_reason: str | None = None

  def __post_init__(self):
    if not isinstance(self.name, str) or not self.name:
      raise ValueError("a frozen case needs a nonempty name")
    if not isinstance(self.program, str) or not self.program:
      raise ValueError("a frozen case needs a nonempty program description")
    if not isinstance(self.input_words, tuple) or not self.input_words:
      raise ValueError("a frozen case needs at least one input word")
    if not isinstance(self.model, Model):
      raise TypeError("a frozen case needs a Model record")
    for word in self.input_words:
      _check_word(word)
    if not isinstance(self.allowed_results, tuple):
      raise TypeError("allowed results must be a tuple of word sequences")
    for result in self.allowed_results:
      if not isinstance(result, tuple) or not result:
        raise ValueError("an allowed result must be a nonempty word sequence")
      for word in result:
        _check_word(word)
    if type(self.applicable) is not bool:
      raise TypeError("applicability must be a bool")
    if self.applicable:
      if not self.allowed_results:
        raise ValueError("an applicable model needs at least one allowed result")
      if self.applicability_reason is not None:
        raise ValueError("an applicable model cannot carry an applicability reason")
    elif not self.applicability_reason:
      raise ValueError("an inapplicable model needs an applicability reason")

  def analyse(self, observed_result, *, target_evidence=None):
    """Classify one observed output against this case's model."""
    observed = _result_words(observed_result)
    if not self.applicable:
      classification = NUMERIC_UNRESOLVED
    elif observed in self.allowed_results:
      classification = IN_JUSTIFIED_SET
    elif self.model.fully_justified:
      # The output is outside every result the justified model permits.
      classification = OUTSIDE_JUSTIFIED_SET
    else:
      # The model itself is incomplete, so the output does not yet establish a
      # defect.
      classification = NUMERIC_UNRESOLVED
    return Analysis(
      case=self,
      observed_result=observed,
      classification=classification,
      target_evidence=target_evidence,
    )


@dataclass(frozen=True)
class Analysis:
  """One observed result classified against one frozen case's model."""

  case: FrozenCase
  observed_result: tuple[int, ...]
  classification: str
  target_evidence: TargetEvidence | None = None

  def __post_init__(self):
    if not isinstance(self.case, FrozenCase):
      raise TypeError("an analysis needs a FrozenCase")
    if self.classification not in CLASSIFICATIONS:
      raise ValueError(f"classification must be one of {', '.join(CLASSIFICATIONS)}")
    if self.target_evidence is not None and not isinstance(
      self.target_evidence, TargetEvidence
    ):
      raise TypeError("target evidence must be a TargetEvidence record")

  def record(self):
    """A JSON-able statement of the model, its allowed words, and the observation."""
    return {
      "case": self.case.name,
      "program": self.case.program,
      "model": {
        "name": self.case.model.name,
        "assumptions": list(self.case.model.assumptions),
        "fully_justified": self.case.model.fully_justified,
        "applicable": self.case.applicable,
        "applicability_reason": self.case.applicability_reason,
      },
      "input_words": list(self.case.input_words),
      "allowed_results": sorted(list(result) for result in self.case.allowed_results),
      "allowed_result_count": len(self.case.allowed_results),
      "observed_result": list(self.observed_result),
      "classification": self.classification,
      "target_evidence": (
        None if self.target_evidence is None else self.target_evidence.record()
      ),
    }


def reduction_case(*, name, program, words, model=None):
  """One reduction case whose allowed results are every binary tree.

  A non-finite operand or an overflowing partial sum leaves the model
  inapplicable: the analysis then records the reason and classifies as
  `numeric_unresolved` instead of claiming an allowed-result set.
  """
  words = tuple(words)
  for word in words:
    _check_word(word)
  if model is None:
    model = Model(
      name=f"all binary trees over {len(words)} frozen words",
      assumptions=_REDUCTION_ASSUMPTIONS,
      fully_justified=True,
    )
  try:
    outcomes = reduction_f32_rne_outcomes(words)
  except (ValueError, OverflowError) as error:
    return FrozenCase(
      name=name,
      program=program,
      input_words=words,
      allowed_results=(),
      model=model,
      applicable=False,
      applicability_reason=str(error),
    )
  return FrozenCase(
    name=name,
    program=program,
    input_words=words,
    allowed_results=tuple(sorted((outcome,) for outcome in outcomes)),
    model=model,
  )


def fp1_case():
  """The plan's rounding-sensitive four-word cancellation sum."""
  return reduction_case(
    name="FP1",
    program="rounding-sensitive four-word cancellation sum",
    words=FP1_WORDS,
    model=Model(
      name="all binary trees over the four frozen words",
      assumptions=_REDUCTION_ASSUMPTIONS,
      fully_justified=True,
    ),
  )


def fp2_case():
  """The exact-result cancellation control with one modeled outcome."""
  return reduction_case(
    name="FP2",
    program="exact-result cancellation control",
    words=FP2_WORDS,
    model=Model(
      name="exact-result cancellation over every binary tree",
      assumptions=_REDUCTION_ASSUMPTIONS + _EXACT_CANCELLATION_ASSUMPTIONS,
      fully_justified=True,
    ),
  )


def fp3_case():
  """The eight-value repeated and mixed sum."""
  return reduction_case(
    name="FP3",
    program="eight-value repeated and mixed sum",
    words=FP3_WORDS,
    model=Model(
      name="all binary trees over the eight frozen words, memoized by operand subset",
      assumptions=_REDUCTION_ASSUMPTIONS + _EIGHT_OPERAND_ASSUMPTIONS,
      fully_justified=True,
    ),
  )


def fp4_case():
  """The 2x2 transpose that preserves both signed-zero words."""
  applicable, reason = _finite_words(FP4_INPUT_WORDS)
  return FrozenCase(
    name="FP4",
    program="2x2 transpose with signed zero",
    input_words=FP4_INPUT_WORDS,
    allowed_results=(FP4_TRANSPOSED_WORDS,),
    model=Model(
      name="bit-preserving exact transpose",
      assumptions=_EXACT_TRANSPOSE_ASSUMPTIONS,
      fully_justified=True,
    ),
    applicable=applicable,
    applicability_reason=reason,
  )
