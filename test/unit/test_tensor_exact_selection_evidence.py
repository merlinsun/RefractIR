"""Selected witness and solver transcript must match the exported case."""

import copy
import hashlib
import importlib
import json
import struct
import unittest


class SelectionEvidenceTests(unittest.TestCase):
  def fixture(self):
    proof_api = importlib.import_module("test.lib.tensor_exact_f32")
    builder = importlib.import_module("test.lib.build_tensor_exact_loop_case")
    source = proof_api.render_program()
    values = [1, 2, 4, 8]
    inputs = struct.pack("<4f", *values)
    proof, output = proof_api.certify(source, inputs, (0, 8))
    query = builder.render_query()
    model = json.dumps(
      {"@kernel": {"%?x": {"shape": [2, 2], "element": "i32", "leaves": values}}}
    ).encode()
    path = builder.PATH
    command = [
      "symirsolve",
      "--main",
      "@kernel",
      "--path",
      ",".join(path),
      "--tensor-param-domain",
      "0,8",
      "query.sir",
    ]
    result = json.dumps(
      {"status": "SAT", "exit_code": 0, "stdout": "SAT\n", "command": command}
    ).encode()
    artifacts = {
      "selection-query.sir": query.encode(),
      "selection-model.json": model,
      "selection-result.json": result,
    }
    selection = {
      "method": "bounded-integer-twin-smt",
      "domain": [0, 8],
      "target": {"index": [0, 0], "value": 63},
      "path": path,
      "query_file": "selection-query.sir",
      "query_sha256": hashlib.sha256(artifacts["selection-query.sir"]).hexdigest(),
      "model_file": "selection-model.json",
      "model_sha256": hashlib.sha256(model).hexdigest(),
      "result_file": "selection-result.json",
      "result_sha256": hashlib.sha256(result).hexdigest(),
      "replay": "complete-f32-output-bitwise",
    }
    return source, inputs, output, selection, artifacts

  def validate(self, fixture):
    api = importlib.import_module("test.lib.tensor_exact_f32")
    return api.validate_selection(*fixture)

  def test_accepts_query_model_and_full_output_witness(self):
    self.validate(self.fixture())

  def test_rejects_selection_target_that_output_does_not_satisfy(self):
    fixture = list(self.fixture())
    fixture[3] = copy.deepcopy(fixture[3])
    fixture[3]["target"]["value"] = 64
    with self.assertRaisesRegex(ValueError, "target"):
      self.validate(fixture)

  def test_rejects_solver_domain_mismatch(self):
    fixture = list(self.fixture())
    fixture[3] = copy.deepcopy(fixture[3])
    fixture[3]["domain"] = [0, 9]
    with self.assertRaisesRegex(ValueError, "domain"):
      self.validate(fixture)

  def test_rejects_query_changed_after_selection(self):
    fixture = list(self.fixture())
    fixture[4] = dict(fixture[4])
    fixture[4]["selection-query.sir"] += b"// altered\n"
    with self.assertRaisesRegex(ValueError, "query"):
      self.validate(fixture)

  def test_rejects_model_different_from_exported_input_bits(self):
    fixture = list(self.fixture())
    fixture[4] = dict(fixture[4])
    model = json.loads(fixture[4]["selection-model.json"])
    model["@kernel"]["%?x"]["leaves"][0] = 2
    fixture[4]["selection-model.json"] = json.dumps(model).encode()
    with self.assertRaisesRegex(ValueError, "model"):
      self.validate(fixture)


if __name__ == "__main__":
  unittest.main()
