"""Independent exact-domain obligations for the authored two-state loop."""

import copy
import importlib
import struct
import unittest

SOURCE = """\
fun @kernel(%x: tensor [2, 2] f32) : tensor [2, 2] f32 {
  let mut %i: i32 = undef;
  let mut %a: tensor [2, 2] f32 = undef;
  let mut %b: tensor [2, 2] f32 = {0.0, 0.0, 0.0, 0.0};
  let mut %v: tensor [2, 2] f32 = undef;
  let mut %r: tensor [2, 1] f32 = undef;
  let mut %c: tensor [1, 2] f32 = undef;
  let mut %rb: tensor [2, 2] f32 = undef;
  let mut %cb: tensor [2, 2] f32 = undef;
  let mut %a_next: tensor [2, 2] f32 = undef;
  let mut %b_next: tensor [2, 2] f32 = undef;
  let mut %out: tensor [2, 2] f32 = undef;
^entry:
  %a = %x;
  %i = 0;
  br ^header;
^header:
  br %i < 2, ^body, ^exit;
^body:
  %v = tensor add %a, %b;
  %r = tensor sum %v, 1, keep;
  %c = tensor sum %v, 0, keep;
  %rb = tensor broadcast %r, [2, 2];
  %cb = tensor broadcast %c, [2, 2];
  %a_next = tensor add %a, %rb;
  %b_next = tensor add %b, %cb;
  %a = %a_next;
  %b = %b_next;
  %i = %i + 1;
  br ^header;
^exit:
  %out = tensor add %a, %b;
  ret %out;
}
"""


class ComposedExactTests(unittest.TestCase):
  def api(self):
    return importlib.import_module("test.lib.tensor_exact_f32")

  def certify(self, values=(1, 2, 4, 8), domain=(0, 8), source=SOURCE):
    return self.api().certify(source, struct.pack("<4f", *values), domain)

  def test_composed_addition_reduction_is_exact(self):
    proof, output = self.certify()
    self.assertEqual(struct.unpack("<4f", output), (63, 84, 102, 126))
    self.assertTrue(proof["complete_output"])
    self.assertEqual(self.api().render_program(), SOURCE)

  def test_bounds_cover_both_loop_iterations(self):
    proof, _ = self.certify()
    self.assertEqual(proof["path"].count("^body"), 2)
    self.assertEqual(len(proof["bounds"]), 18)
    self.assertEqual(proof["bounds"][-1]["max_value"], 200)
    sums = [row for row in proof["bounds"] if row["operation"] == "sum"]
    self.assertEqual([row["max_partial_sum"] for row in sums], [16, 16, 80, 80])

  def test_reject_partial_sum_outside_exact_range(self):
    self.certify()
    with self.assertRaisesRegex(ValueError, "exact range"):
      self.certify((2**24, 0, 0, 0), (0, 2**24))

  def test_reject_reassociation_sensitive_signed_inputs(self):
    self.certify()
    with self.assertRaisesRegex(ValueError, "nonnegative"):
      self.certify((1, 2**-24, -1, 0), (-1, 8))

  def test_reject_missing_output_dependency_evidence(self):
    proof, output = self.certify()
    incomplete = copy.deepcopy(proof)
    incomplete["bounds"] = [
      row for row in incomplete["bounds"] if row["destination"] != "%b_next"
    ]
    with self.assertRaisesRegex(ValueError, "evidence"):
      self.api().validate(SOURCE, struct.pack("<4f", 1, 2, 4, 8), output, incomplete)

  def test_singleton_axis_and_positive_zero(self):
    api = self.api()
    source = api.render_program((1, 2))
    proof, output = api.certify(source, struct.pack("<2f", 0, 1))
    self.assertEqual(struct.unpack("<2f", output), (6, 10))
    self.assertEqual(
      api.validate(source, struct.pack("<2f", 0, 1), output, proof), output
    )


if __name__ == "__main__":
  unittest.main()
