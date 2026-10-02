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


def generated_region_source(axis=1, iterations=2):
  reduced_shape = "[2, 1]" if axis == 1 else "[1, 2]"
  return f"""\
fun @kernel(%x: tensor [2, 2] f32) : tensor [2, 2] f32 {{
  let mut %i: i32 = undef;
  let mut %acc_x: tensor [2, 2] f32 = undef;
  let mut %acc_y: tensor [2, 2] f32 = {{0.0, 0.0, 0.0, 0.0}};
  let mut %t0: tensor [2, 2] f32 = undef;
  let mut %t1: tensor {reduced_shape} f32 = undef;
  let mut %t2: tensor [2, 2] f32 = undef;
  let mut %t3: tensor [2, 2] f32 = undef;
  let mut %next_x: tensor [2, 2] f32 = undef;
  let mut %next_y: tensor [2, 2] f32 = undef;
  let mut %out: tensor [2, 2] f32 = undef;
^entry:
  %acc_x = %x;
  %i = 0;
  br ^header;
^header:
  br %i < {iterations}, ^body, ^exit;
^body:
  %t0 = tensor add %acc_x, %acc_y;
  %t1 = tensor sum %t0, {axis}, keep;
  %t2 = tensor broadcast %t1, [2, 2];
  %t3 = tensor add %t0, %t2;
  %next_x = tensor add %acc_x, %t3;
  %next_y = tensor add %acc_y, %t3;
  %acc_x = %next_x;
  %acc_y = %next_y;
  %i = %i + 1;
  br ^header;
^exit:
  %out = tensor add %acc_x, %acc_y;
  ret %out;
}}
"""


class GeneratedRegionExactTests(unittest.TestCase):
  """Exact bounds for the restricted generated multi-output region subset."""

  def setUp(self):
    self.api = importlib.import_module("test.lib.tensor_exact_f32")
    self.source = generated_region_source()
    self.input_bytes = struct.pack("<4f", 1, 2, 4, 8)

  def test_certifies_generated_region_and_complete_output(self):
    proof, output = self.api.certify(self.source, self.input_bytes, (0, 8))
    self.assertEqual(struct.unpack("<4f", output), (69, 78, 276, 312))
    self.assertTrue(proof["complete_output"])
    self.assertEqual(proof["kind"], "bounded-region-add-sum-f32-v1")

  def test_bounds_include_each_visit_and_reduction_partials(self):
    proof, _ = self.api.certify(self.source, self.input_bytes, (0, 8))
    self.assertEqual(proof["path"].count("^body"), 2)
    sums = [item for item in proof["bounds"] if item["operation"] == "sum"]
    self.assertEqual([item["max_partial_sum"] for item in sums], [16, 112])
    self.assertTrue(
      all(item["max_value"] <= self.api.EXACT_BOUND for item in proof["bounds"])
    )

  def test_supports_column_reduction_and_single_visit(self):
    source = generated_region_source(axis=0, iterations=1)
    proof, output = self.api.certify(source, self.input_bytes, (0, 8))
    self.assertEqual(struct.unpack("<4f", output), (13, 26, 22, 44))
    sums = [item for item in proof["bounds"] if item["operation"] == "sum"]
    self.assertEqual(len(sums), 1)
    self.assertEqual(sums[0]["max_partial_sum"], 16)

  def test_validator_recomputes_generated_region_evidence(self):
    proof, output = self.api.certify(self.source, self.input_bytes, (0, 8))
    self.assertEqual(
      self.api.validate(self.source, self.input_bytes, output, proof), output
    )
    proof["bounds"][-1]["max_value"] += 1
    with self.assertRaisesRegex(ValueError, "differs from derived obligations"):
      self.api.validate(self.source, self.input_bytes, output, proof)

  def test_rejects_nonpositive_zero_and_out_of_domain_inputs(self):
    with self.assertRaisesRegex(ValueError, "positive zero"):
      self.api.certify(self.source, struct.pack("<4f", -0.0, 1, 2, 3), (0, 8))
    with self.assertRaisesRegex(ValueError, "positive zero"):
      self.api.certify(self.source, struct.pack("<4f", -1, 1, 2, 3), (0, 8))

  def test_rejects_operations_outside_the_generated_subset(self):
    unsupported = self.source.replace(
      "%t3 = tensor add %t0, %t2;", "%t3 = tensor sub %t0, %t2;"
    )
    with self.assertRaisesRegex(ValueError, "unsupported tensor operation"):
      self.api.certify(unsupported, self.input_bytes, (0, 8))


if __name__ == "__main__":
  unittest.main()
