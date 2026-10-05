"""Independent exact-rational rounding helpers for elementary binary32 tests."""

from fractions import Fraction
from functools import lru_cache

_SIGN_MASK = 1 << 31
_EXPONENT_MASK = 0xFF
_FRACTION_MASK = (1 << 23) - 1
_MIN_NORMAL_EXPONENT = -126
_MIN_SUBNORMAL_EXPONENT = -149


def _checked_word(word):
  if type(word) is not int or not 0 <= word <= 0xFFFFFFFF:
    raise ValueError("binary32 operand must be an unsigned 32-bit word")
  if (word >> 23) & _EXPONENT_MASK == _EXPONENT_MASK:
    raise ValueError("binary32 reference requires finite operands")
  return word


def _word_fraction(word):
  sign = -1 if word & _SIGN_MASK else 1
  exponent = (word >> 23) & _EXPONENT_MASK
  fraction = word & _FRACTION_MASK
  if exponent == 0:
    return Fraction(sign * fraction, 1 << -_MIN_SUBNORMAL_EXPONENT)
  significand = (1 << 23) | fraction
  power = exponent - 127 - 23
  if power >= 0:
    return Fraction(sign * significand * (1 << power), 1)
  return Fraction(sign * significand, 1 << -power)


def _round_positive_ratio_to_even(numerator, denominator):
  quotient, remainder = divmod(numerator, denominator)
  twice_remainder = remainder * 2
  if twice_remainder > denominator or (twice_remainder == denominator and quotient & 1):
    return quotient + 1
  return quotient


def _round_fraction_scaled(value, power):
  numerator = value.numerator
  denominator = value.denominator
  if power >= 0:
    numerator <<= power
  else:
    denominator <<= -power
  return _round_positive_ratio_to_even(numerator, denominator)


def round_fraction_to_f32_bits(value):
  """Round a finite exact rational to binary32 using round-to-nearest-even."""
  if not isinstance(value, Fraction):
    value = Fraction(value)
  if value == 0:
    return 0
  sign_word = _SIGN_MASK if value < 0 else 0
  magnitude = abs(value)
  numerator = magnitude.numerator
  denominator = magnitude.denominator
  exponent = numerator.bit_length() - denominator.bit_length()
  if exponent >= 0:
    if numerator < denominator << exponent:
      exponent -= 1
  elif numerator << -exponent < denominator:
    exponent -= 1

  if exponent < _MIN_NORMAL_EXPONENT:
    significand = _round_fraction_scaled(magnitude, -_MIN_SUBNORMAL_EXPONENT)
    if significand == 0:
      return sign_word
    if significand >= 1 << 23:
      return sign_word | (1 << 23)
    return sign_word | significand

  significand = _round_fraction_scaled(magnitude, 23 - exponent)
  if significand == 1 << 24:
    exponent += 1
    significand >>= 1
  if exponent > 127:
    raise OverflowError("binary32 overflow under round-to-nearest-even")
  exponent_field = exponent + 127
  return sign_word | (exponent_field << 23) | (significand - (1 << 23))


def add_f32_rne_bits(left_word, right_word):
  """Add two finite binary32 words, rounding the exact sum once to RNE."""
  left_word = _checked_word(left_word)
  right_word = _checked_word(right_word)
  left = _word_fraction(left_word)
  right = _word_fraction(right_word)
  total = left + right
  if total == 0:
    if left_word == _SIGN_MASK and right_word == _SIGN_MASK:
      return _SIGN_MASK
    return 0
  return round_fraction_to_f32_bits(total)


def sub_f32_rne_bits(left_word, right_word):
  """Subtract two finite binary32 words, rounding the exact difference once."""
  left_word = _checked_word(left_word)
  right_word = _checked_word(right_word)
  left = _word_fraction(left_word)
  right = _word_fraction(right_word)
  difference = left - right
  if difference == 0:
    if left_word == _SIGN_MASK and right_word == 0:
      return _SIGN_MASK
    return 0
  return round_fraction_to_f32_bits(difference)


def mul_f32_rne_bits(left_word, right_word):
  """Multiply two finite binary32 words, rounding the exact product once."""
  left_word = _checked_word(left_word)
  right_word = _checked_word(right_word)
  sign = (left_word ^ right_word) & _SIGN_MASK
  product = _word_fraction(left_word) * _word_fraction(right_word)
  if product == 0:
    return sign
  return round_fraction_to_f32_bits(product)


def fma_f32_rne_bits(left_word, right_word, addend_word):
  """Evaluate a*b+c with one binary32 round-to-nearest-even step."""
  left_word = _checked_word(left_word)
  right_word = _checked_word(right_word)
  addend_word = _checked_word(addend_word)
  product = _word_fraction(left_word) * _word_fraction(right_word)
  addend = _word_fraction(addend_word)
  total = product + addend
  if total == 0:
    product_sign = (left_word ^ right_word) & _SIGN_MASK
    addend_sign = addend_word & _SIGN_MASK
    if product == 0 and product_sign == addend_sign == _SIGN_MASK:
      return _SIGN_MASK
    return 0
  return round_fraction_to_f32_bits(total)


def mul_add_separate_f32_bits(left_word, right_word, addend_word):
  """Evaluate a*b+c with binary32 rounding after multiplication and addition."""
  product_word = mul_f32_rne_bits(left_word, right_word)
  return add_f32_rne_bits(product_word, addend_word)


def reduction_f32_rne_outcomes(words):
  """Return results of every binary tree over up to eight finite input words.

  This deliberately models the legal arithmetic variability of an unordered
  reduction without claiming that a GPU chooses one particular tree.
  """
  words = tuple(_checked_word(word) for word in words)
  if not words or len(words) > 8:
    raise ValueError("binary32 reduction oracle requires one to eight operands")

  @lru_cache(maxsize=None)
  def outcomes(mask):
    if mask & (mask - 1) == 0:
      index = mask.bit_length() - 1
      return frozenset((words[index],))

    result = set()
    anchor = mask & -mask
    left_mask = (mask - 1) & mask
    while left_mask:
      right_mask = mask ^ left_mask
      if left_mask & anchor and right_mask:
        for left_word in outcomes(left_mask):
          for right_word in outcomes(right_mask):
            result.add(add_f32_rne_bits(left_word, right_word))
      left_mask = (left_mask - 1) & mask
    return frozenset(result)

  return outcomes((1 << len(words)) - 1)
