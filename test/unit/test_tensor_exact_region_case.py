"""Generated bounded regions must cross the ordinary case boundary exactly."""

import hashlib
import importlib
import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test.unit.test_tensor_exact_composed import generated_region_source

ROOT = Path(__file__).resolve().parents[2]


def _tool(environment_name, binary_name):
  value = os.environ.get(environment_name)
  return Path(value).resolve() if value else ROOT / "build/bin" / binary_name


SYMIRI = _tool("REFRACTIR_SYMIRI", "symiri")
SYMIRC = _tool("REFRACTIR_SYMIRC", "symirc")
SYMIR_SOLVE = _tool("REFRACTIR_SYMIR_SOLVE", "symirsolve")
TOOLS_AVAILABLE = all(path.is_file() for path in (SYMIRI, SYMIRC, SYMIR_SOLVE))


def generated_region_source_for_shape(rows, columns, iterations=2):
  zeros = ", ".join(["0.0"] * (rows * columns))
  return f"""\
fun @kernel(%x: tensor [{rows}, {columns}] f32) : tensor [{rows}, {columns}] f32 {{
  let mut %i: i32 = undef;
  let mut %acc_x: tensor [{rows}, {columns}] f32 = undef;
  let mut %acc_y: tensor [{rows}, {columns}] f32 = {{{zeros}}};
  let mut %t0: tensor [{rows}, {columns}] f32 = undef;
  let mut %t1: tensor [{rows}, 1] f32 = undef;
  let mut %t2: tensor [{rows}, {columns}] f32 = undef;
  let mut %t3: tensor [{rows}, {columns}] f32 = undef;
  let mut %next_x: tensor [{rows}, {columns}] f32 = undef;
  let mut %next_y: tensor [{rows}, {columns}] f32 = undef;
  let mut %out: tensor [{rows}, {columns}] f32 = undef;
^entry:
  %acc_x = %x;
  %i = 0;
  br ^header;
^header:
  br %i < {iterations}, ^body, ^exit;
^body:
  %t0 = tensor add %acc_x, %acc_y;
  %t1 = tensor sum %t0, 1, keep;
  %t2 = tensor broadcast %t1, [{rows}, {columns}];
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


class GeneratedRegionCaseTests(unittest.TestCase):
  def setUp(self):
    self.exact = importlib.import_module("test.lib.tensor_exact_f32")
    self.builder = importlib.import_module("test.lib.build_tensor_exact_region_case")
    self.source = generated_region_source()
    self.shape = (2, 2)
    self.values = (1, 2, 4, 8)
    self.domain = (0, 8)
    self.input_bytes = struct.pack("<4f", 1, 2, 4, 8)
    self.expected = self.exact.certify(self.source, self.input_bytes, self.domain)[1]

  def _tools(self, command, **kwargs):
    if "--emit-model" in command:
      Path(command[command.index("--emit-model") + 1]).write_text(
        json.dumps({"@kernel": {"%?x": self.values}})
      )
    if "--output" in command:
      Path(command[command.index("--output") + 1]).write_text(
        f"fun @kernel() : tensor {list(self.shape)} i32 {{ ret %out; }}\n"
      )
    if "--tensor-output" in command:
      Path(command[command.index("--tensor-output") + 1]).write_bytes(self.expected)
    if "--target" in command:
      Path(command[command.index("-o") + 1]).write_text("# generated kernel\n")
    return subprocess.CompletedProcess(command, 0, "SAT\n", "")

  def _build(
    self,
    root,
    side_effect=None,
    *,
    source=None,
    target=69,
    domain=(0, 8),
    values=(1, 2, 4, 8),
    shape=(2, 2),
    guard_envelope=None,
    padding=None,
    profile="tensor-v2",
    row_padding=0,
    element_offset=0,
    commands=None,
    selection="solver",
    witness=None,
  ):
    if source is not None:
      self.source = source
    self.shape = shape
    self.values = tuple(values)
    self.domain = domain
    self.input_bytes = struct.pack(f"<{len(self.values)}f", *self.values)
    self.expected = self.exact.certify(self.source, self.input_bytes, domain)[1]
    helper = importlib.import_module("test.lib.build_tensor_exact_loop_case")
    tools = side_effect or self._tools

    def record(command, **kwargs):
      if commands is not None:
        commands.append(list(command))
      return tools(command, **kwargs)

    with mock.patch.object(helper.subprocess, "run", side_effect=record):
      return self.builder.build_case(
        Path(root) / "case",
        source=self.source,
        target=target,
        symiri="symiri",
        symirc="symirc",
        solver="symirsolve",
        domain=domain,
        timeout_s=0.1,
        guard_envelope=guard_envelope,
        padding=padding,
        profile=profile,
        row_padding=row_padding,
        element_offset=element_offset,
        selection=selection,
        witness=witness,
      )

  def _load(self, case):
    tensor_case = importlib.import_module("test.lib.tensor_case")
    manifest = json.loads((case / "case.json").read_text())
    input_shape = manifest["inputs"][0]["shape"]
    output_shape = manifest["output"]["shape"]
    signature = {
      "entry": "kernel",
      "inputs": [
        {
          "name": "x",
          "dtype": "f32",
          "shape": input_shape,
          "byte_length": len(self.input_bytes),
        }
      ],
      "output": {
        "name": "return",
        "dtype": "f32",
        "shape": output_shape,
        "byte_length": len(self.expected),
      },
    }
    with mock.patch.object(tensor_case, "_describe", return_value=signature):
      return tensor_case.load_case(case)

  def _build_and_load(self, *, source, shape, values, domain):
    input_bytes = struct.pack(f"<{len(values)}f", *values)
    expected = self.exact.certify(source, input_bytes, domain)[1]
    target = int(struct.unpack(f"<{len(values)}f", expected)[0])
    with tempfile.TemporaryDirectory() as root:
      case = self._build(
        root,
        source=source,
        target=target,
        domain=domain,
        values=values,
        shape=shape,
      )
      return self._load(case), input_bytes, expected

  def _rewrite_file(self, case, name, data):
    path = case / name
    path.write_bytes(data)
    manifest_path = case / "case.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][name] = {
      "bytes": len(data),
      "sha256": hashlib.sha256(data).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

  def test_exports_and_loads_a_complete_bitwise_generated_case(self):
    with tempfile.TemporaryDirectory() as root:
      case = self._build(root)
      loaded = self._load(case)
      model = json.loads((case / "selection-model.json").read_text())
      self.assertEqual(loaded.inputs, struct.pack("<4f", *model["@kernel"]["%?x"]))
      self.assertEqual(loaded.expected, self.expected)
      self.assertEqual(loaded.manifest["comparison"]["mode"], "exact")
      self.assertEqual(
        loaded.manifest["reference"]["exactness"]["kind"],
        "bounded-region-add-sum-f32-v1",
      )

  def test_exports_and_loads_an_eight_element_generated_region(self):
    shape = (2, 4)
    values = (1,) * 8
    loaded, input_bytes, expected = self._build_and_load(
      source=generated_region_source_for_shape(*shape),
      shape=shape,
      values=values,
      domain=(0, 8),
    )

    self.assertEqual(loaded.inputs, input_bytes)
    self.assertEqual(loaded.expected, expected)
    self.assertEqual(loaded.manifest["output"]["shape"], [2, 4])

  def test_exports_a_domain_with_positive_lower_bound(self):
    values = (1, 1, 1, 1)
    loaded, input_bytes, expected = self._build_and_load(
      source=self.source, shape=(2, 2), values=values, domain=(1, 8)
    )

    self.assertEqual(loaded.inputs, input_bytes)
    self.assertEqual(loaded.expected, expected)
    self.assertEqual(loaded.manifest["reference"]["exactness"]["input_domain"], [1, 8])

  def test_exports_a_positive_subdomain_witness(self):
    values = (3, 3, 3, 3)
    loaded, input_bytes, expected = self._build_and_load(
      source=self.source, shape=(2, 2), values=values, domain=(3, 8)
    )

    self.assertEqual(loaded.inputs, input_bytes)
    self.assertEqual(loaded.expected, expected)

  def test_exports_a_positive_singleton_domain(self):
    values = (1, 1, 1, 1)
    loaded, input_bytes, expected = self._build_and_load(
      source=self.source, shape=(2, 2), values=values, domain=(1, 1)
    )

    self.assertEqual(loaded.inputs, input_bytes)
    self.assertEqual(loaded.expected, expected)

  def test_exports_an_upper_singleton_domain(self):
    values = (8, 8, 8, 8)
    loaded, input_bytes, expected = self._build_and_load(
      source=self.source, shape=(2, 2), values=values, domain=(8, 8)
    )

    self.assertEqual(loaded.inputs, input_bytes)
    self.assertEqual(loaded.expected, expected)

  def test_rejects_a_changed_complete_output_after_digest_refresh(self):
    with tempfile.TemporaryDirectory() as root:
      case = self._build(root)
      changed = bytearray(self.expected)
      changed[-1] ^= 1
      self._rewrite_file(case, "expected.bin", bytes(changed))
      with self.assertRaisesRegex(ValueError, "complete output bits"):
        self._load(case)

  def test_rejects_bounds_that_do_not_match_recomputation(self):
    with tempfile.TemporaryDirectory() as root:
      case = self._build(root)
      manifest_path = case / "case.json"
      manifest = json.loads(manifest_path.read_text())
      manifest["reference"]["exactness"]["bounds"][0]["max_value"] += 1
      manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
      with self.assertRaisesRegex(ValueError, "derived obligations"):
        self._load(case)

  def test_rejects_solver_query_changed_consistently_with_digests(self):
    with tempfile.TemporaryDirectory() as root:
      case = self._build(root)
      query_path = case / "selection-query.sir"
      query = query_path.read_bytes() + b"// changed\n"
      query_path.write_bytes(query)
      manifest_path = case / "case.json"
      manifest = json.loads(manifest_path.read_text())
      digest = hashlib.sha256(query).hexdigest()
      manifest["files"]["selection-query.sir"] = {"bytes": len(query), "sha256": digest}
      selection = manifest["reference"]["exactness"]["selection"]
      selection["query_sha256"] = digest
      manifest["solver_selection"]["query_sha256"] = digest
      manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
      with self.assertRaisesRegex(ValueError, "query differs"):
        self._load(case)

  def test_rejects_models_outside_the_proved_domain(self):
    def invalid_model(command, **kwargs):
      result = self._tools(command, **kwargs)
      if "--emit-model" in command:
        model_path = Path(command[command.index("--emit-model") + 1])
        model = json.loads(model_path.read_text())
        model["@kernel"]["%?x"][0] = 9
        model_path.write_text(json.dumps(model))
      return result

    with tempfile.TemporaryDirectory() as root:
      with self.assertRaisesRegex(ValueError, "outside the declared domain"):
        self._build(root, invalid_model)
      self.assertFalse((Path(root) / "case/case.json").exists())

  def test_single_visit_query_uses_its_derived_path_and_domain(self):
    source = generated_region_source(iterations=1)
    input_bytes = struct.pack("<4f", 1, 2, 3, 4)
    _, expected = self.exact.certify(source, input_bytes, (0, 4))
    target = struct.unpack("<4f", expected)[0]
    query = self.exact.render_generated_region_query(source, (0, 4), int(target))
    self.assertIn("in [0, 4]", query)
    self.assertIn(f"require %__exact_selected == {int(target)};", query)
    self.assertEqual(
      self.exact.generated_region_path(1),
      ["^entry", "^header", "^body", "^header", "^exit"],
    )

  def test_integer_twin_rewrites_positive_zero_initializers(self):
    query = self.exact.render_generated_region_query(self.source, (0, 8), 69)
    self.assertNotIn("0.0", query)
    self.assertIn("= {0, 0, 0, 0};", query)

  def test_export_records_requested_guard_envelope_and_padding(self):
    padding = {"i32": "55555555", "f32": "3f800001"}
    with tempfile.TemporaryDirectory() as root:
      case = self._build(root, guard_envelope=(2, 3), padding=padding)
      manifest = json.loads((case / "case.json").read_text())
      unguarded = self._build(Path(root) / "plain")
      plain = json.loads((unguarded / "case.json").read_text())

    self.assertEqual(
      manifest["execution"]["guard_envelope"], {"prefix": 2, "suffix": 3}
    )
    self.assertEqual(manifest["execution"]["padding_by_dtype"], padding)
    self.assertNotIn("guard_envelope", plain["execution"])
    self.assertEqual(manifest["inputs"], plain["inputs"])
    self.assertEqual(manifest["output"], plain["output"])

  def test_a_tensor_v3_tail_case_records_the_padded_tile_and_strides(self):
    source = generated_region_source_for_shape(4, 3)
    values = (1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4)
    self.source = source
    self.shape = (4, 3)
    self.input_bytes = struct.pack(f"<{len(values)}f", *values)
    expected = self.exact.certify(source, self.input_bytes, (0, 8))[1]
    target = int(struct.unpack(f"<{len(values)}f", expected)[0])
    commands = []
    with tempfile.TemporaryDirectory() as root:
      case = self._build(
        root,
        source=source,
        shape=(4, 3),
        values=values,
        target=target,
        profile="tensor-v3",
        row_padding=2,
        element_offset=1,
        guard_envelope=(3, 5),
        commands=commands,
      )
      manifest = json.loads((case / "case.json").read_text())
      record = manifest["inputs"][0]
      loaded = self._load(case)

    self.assertEqual(manifest["profile"], "tensor-v3")
    self.assertEqual(record["shape"], [4, 3])
    self.assertEqual(record["tile_shape"], [4, 4])
    self.assertEqual(record["element_strides"], [6, 1])
    self.assertEqual(record["element_offset"], 1)
    self.assertEqual(record["storage_span"], 1 + 3 * 6 + 2 * 1 + 1)
    self.assertEqual(manifest["output"]["tile_shape"], [4, 4])
    self.assertEqual(manifest["output"]["element_strides"], [6, 1])
    self.assertEqual(loaded.expected, self.expected)
    versions = {
      command[command.index("--tensor-profile-version") + 1]
      for command in commands
      if "--tensor-profile-version" in command
    }
    self.assertEqual(versions, {"v3"})
    compile_command = next(command for command in commands if "--target" in command)
    self.assertIn("--tensor-layout-manifest", compile_command)

  def test_a_witness_admitted_tail_case_loads_without_a_solver_selection(self):
    source = generated_region_source_for_shape(4, 3)
    witness = (2,) * 12
    commands = []
    with tempfile.TemporaryDirectory() as root:
      case = self._build(
        root,
        source=source,
        shape=(4, 3),
        values=witness,
        target=0,
        profile="tensor-v3",
        selection="witness",
        witness=witness,
        commands=commands,
      )
      manifest = json.loads((case / "case.json").read_text())
      loaded = self._load(case)

    self.assertEqual(manifest["evidence"]["kind"], "construction")
    self.assertEqual(manifest["evidence"]["status"], "construction_checked")
    self.assertEqual(manifest["comparison"]["mode"], "numeric")
    self.assertEqual(manifest["comparison"]["atol"], [0, 1])
    self.assertEqual(manifest["comparison"]["rtol"], [0, 1])
    self.assertEqual(manifest["reference"]["precisions"], [128, 256])
    self.assertIsNone(manifest["reference"]["exactness"])
    self.assertNotIn("solver_selection", manifest)
    self.assertNotIn("selection-query.sir", manifest["files"])
    self.assertNotIn("target", manifest["property"])
    self.assertEqual(
      manifest["coverage"]["input_behavior"]["pattern"], "repeated-value"
    )
    self.assertEqual(
      manifest["generator"]["input_pattern"],
      manifest["coverage"]["input_behavior"]["pattern"],
    )
    self.assertEqual(loaded.inputs, struct.pack(f"<{len(witness)}f", *witness))
    self.assertEqual(loaded.expected, self.expected)
    self.assertFalse([command for command in commands if "symirsolve" in command[0]])

  def test_a_witness_outside_the_declared_domain_is_refused(self):
    source = generated_region_source_for_shape(4, 3)
    with tempfile.TemporaryDirectory() as root:
      with self.assertRaisesRegex(
        importlib.import_module("test.lib.build_tensor_exact_loop_case").CaseBuildError,
        "domain",
      ):
        self._build(
          root,
          source=source,
          shape=(4, 3),
          values=(1,) * 12,
          target=0,
          profile="tensor-v3",
          selection="witness",
          witness=(9,) + (1,) * 11,
        )

  def test_a_tensor_v2_case_keeps_its_dense_logical_tile(self):
    with tempfile.TemporaryDirectory() as root:
      case = self._build(root)
      manifest = json.loads((case / "case.json").read_text())
      record = manifest["inputs"][0]

    self.assertEqual(manifest["profile"], "tensor-v2")
    self.assertEqual(record["tile_shape"], [2, 2])
    self.assertEqual(record["element_strides"], [2, 1])
    self.assertEqual(record["element_offset"], 0)
    self.assertEqual(record["storage_span"], 4)

  @unittest.skipUnless(
    TOOLS_AVAILABLE, "built symiri, symirc and symirsolve are required"
  )
  def test_native_solver_replay_and_case_loader(self):
    with tempfile.TemporaryDirectory() as root:
      case = self.builder.build_case(
        Path(root) / "case",
        source=self.source,
        target=69,
        symiri=SYMIRI,
        symirc=SYMIRC,
        solver=SYMIR_SOLVE,
      )
      loaded = importlib.import_module("test.lib.tensor_case").load_case(
        case, symiri=SYMIRI
      )
      _, independently_expected = self.exact.certify(self.source, loaded.inputs, (0, 8))
      self.assertEqual(loaded.manifest["comparison"]["mode"], "exact")
      self.assertEqual(loaded.expected, independently_expected)
      self.assertEqual(struct.unpack("<f", loaded.expected[:4])[0], 69.0)

  @unittest.skipUnless(
    TOOLS_AVAILABLE, "built symiri, symirc and symirsolve are required"
  )
  def test_native_solver_replays_larger_shape_with_positive_domain(self):
    shape = (2, 4)
    source = generated_region_source_for_shape(*shape)
    values = (1,) * 8
    domain = (1, 8)
    input_bytes = struct.pack("<8f", *values)
    _, anchor_expected = self.exact.certify(source, input_bytes, domain)
    target = int(struct.unpack("<f", anchor_expected[:4])[0])

    with tempfile.TemporaryDirectory() as root:
      case = self.builder.build_case(
        Path(root) / "case",
        source=source,
        target=target,
        symiri=SYMIRI,
        symirc=SYMIRC,
        solver=SYMIR_SOLVE,
        domain=domain,
      )
      loaded = importlib.import_module("test.lib.tensor_case").load_case(
        case, symiri=SYMIRI
      )

    _, independently_expected = self.exact.certify(source, loaded.inputs, domain)
    self.assertEqual(loaded.manifest["inputs"][0]["shape"], [2, 4])
    self.assertEqual(loaded.manifest["reference"]["exactness"]["input_domain"], [1, 8])
    self.assertEqual(loaded.expected, independently_expected)
    self.assertEqual(struct.unpack("<f", loaded.expected[:4])[0], target)


if __name__ == "__main__":
  unittest.main()
