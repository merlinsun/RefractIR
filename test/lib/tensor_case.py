"""Strict loader for reproducible tensor case directories.

A companion of a counted loop region earns its bit-exact comparison here, so
this loader depends on `test.lib.tensor_region_generation.unroll_source`: the
mechanical unrolling of the counted program is the only companion source the
derivation check admits.
"""

import hashlib
import json
import math
import os
import re
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from test.lib import tensor_layout

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYMIRI = Path(os.environ.get("REFRACTIR_SYMIRI", ROOT / "symiri")).resolve()
SCHEMA_V1 = "refractir.tensor-case.v1"
SCHEMA_V2 = "refractir.tensor-case.v2"
SCHEMA_V3 = "refractir.tensor-case.v3"
SCHEMA_V4 = "refractir.tensor-case.v4"
PROFILE_V1 = "tensor-v1"
PROFILE_V2 = "tensor-v2"
PROFILE_V3 = "tensor-v3"
PROFILE_V4 = "tensor-v4"
# The one transformation a companion's derivation may name, and the name the
# carried copy of the counted program is written under.
COMPANION_TRANSFORMATION = "test.lib.tensor_region_generation.unroll_source"
COUNTED_PROGRAM = "counted/program.sir"
_DTYPE_SIZE = {"i32": 4, "f32": 4}
# The reference-interpreter precisions every numerical reference is replayed at.
NUMERIC_PRECISIONS = [128, 256]


class CaseValidationError(ValueError):
  """A case is not a valid reproducible input to the runner."""

  def __init__(self, message, status="invalid_case"):
    super().__init__(message)
    self.status = status


@dataclass(frozen=True)
class TensorCase:
  path: Path
  manifest: dict
  program: bytes
  inputs: bytes
  expected: bytes
  signature: dict


def _require(condition, message):
  if not condition:
    raise CaseValidationError(message)


def _fnv1a64(data):
  value = 14695981039346656037
  for byte in data:
    value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
  return f"{value:016x}"


def _integer(value, where, minimum=0):
  _require(
    type(value) is int and value >= minimum, f"{where} must be an integer >= {minimum}"
  )
  return value


def _shape(value, where, allow_scalar=True, *, profile=PROFILE_V1):
  _require(isinstance(value, list), f"{where} shape must be a list")
  if not value:
    _require(allow_scalar, f"{where} cannot be scalar")
    return []
  rank_limit = 1 if profile == PROFILE_V4 else 2 if profile == PROFILE_V3 else 3
  _require(1 <= len(value) <= rank_limit, f"{where} rank must be in [1, {rank_limit}]")
  count = 1
  for axis, dimension in enumerate(value):
    dimension = _integer(dimension, f"{where} dimension {axis}", 1)
    if profile != PROFILE_V3:
      _require(
        dimension & (dimension - 1) == 0, f"{where} dimensions must be powers of two"
      )
    count *= dimension
  if profile == PROFILE_V3:
    try:
      tile = tensor_layout.execution_tile_shape(value, profile="v3")
    except tensor_layout.LayoutError as error:
      raise CaseValidationError(
        f"{where} has an invalid execution tile: {error}"
      ) from error
    _require(
      _elements(tile) <= 512,
      f"{where} padded tile volume exceeds 512 elements",
    )
  else:
    limit = 256 if profile == PROFILE_V1 else 512
    _require(count <= limit, f"{where} exceeds the tensor element limit {limit}")
  return value


def _elements(shape):
  count = 1
  for dimension in shape:
    count *= dimension
  return count


def _row_major_strides(shape):
  stride = 1
  result = [0] * len(shape)
  for index in range(len(shape) - 1, -1, -1):
    result[index] = stride
    stride *= shape[index]
  return result


def _artifact_path(root, name):
  _require(isinstance(name, str) and name, "artifact path must be a nonempty string")
  relative = PurePosixPath(name)
  _require(
    not relative.is_absolute() and ".." not in relative.parts,
    "artifact path escapes the case directory",
  )
  path = root.joinpath(*relative.parts)
  _require(path.is_file(), f"artifact is missing: {name}")
  return path


def _read_artifacts(root, manifest, *, schema):
  records = manifest.get("files")
  _require(isinstance(records, dict), "files must be an artifact map")
  artifacts = {}
  for name, record in records.items():
    path = _artifact_path(root, name)
    _require(isinstance(record, dict), f"file record for {name} must be an object")
    data = path.read_bytes()
    _require(record.get("bytes") == len(data), f"byte length mismatch for {name}")
    if schema == SCHEMA_V1:
      digest = record.get("fnv1a64")
      _require(
        isinstance(digest, str) and digest == _fnv1a64(data),
        f"FNV-1a digest mismatch for {name}",
      )
    else:
      digest = record.get("sha256")
      _require(
        isinstance(digest, str)
        and len(digest) == 64
        and digest == hashlib.sha256(data).hexdigest(),
        f"SHA-256 digest mismatch for {name}",
      )
    artifacts[name] = data
  for required in ("program.sir", "inputs.bin", "expected.bin"):
    _require(
      required in artifacts, f"files does not describe required artifact {required}"
    )
  return artifacts


def _validate_rational_pair(pair, where):
  _require(
    isinstance(pair, list) and len(pair) == 2, f"{where} must be a rational pair"
  )
  numerator = _integer(pair[0], f"{where} numerator")
  denominator = _integer(pair[1], f"{where} denominator", 1)
  return numerator, denominator


def _validate_buffer(record, where, artifacts, *, profile):
  _require(isinstance(record, dict), f"{where} must be an object")
  _require(
    isinstance(record.get("name"), str) and record["name"], f"{where} name is missing"
  )
  dtype = record.get("dtype")
  _require(dtype in _DTYPE_SIZE, f"{where} has unsupported dtype {dtype!r}")
  shape = _shape(record.get("shape"), where, profile=profile)
  if profile == PROFILE_V4:
    _require(bool(shape), f"{where} cannot be scalar under tensor-v4")
    _require(dtype == "f32", f"{where} must use f32 under tensor-v4")
  file_name = record.get("file")
  _require(
    file_name in artifacts, f"{where} file is not covered by the artifact digests"
  )
  offset = _integer(record.get("byte_offset"), f"{where} byte_offset")
  length = _integer(record.get("byte_length"), f"{where} byte_length")
  expected_length = _elements(shape) * _DTYPE_SIZE[dtype]
  _require(length == expected_length, f"{where} byte length does not match its shape")
  _require(
    offset + length <= len(artifacts[file_name]), f"{where} range exceeds its file"
  )
  strides = record.get("element_strides")
  layout_profile = "v2" if profile == PROFILE_V4 else profile.removeprefix("tensor-")
  if profile == PROFILE_V1:
    _require(
      strides == _row_major_strides(shape), f"{where} element strides are not row-major"
    )
    _require(
      _integer(record.get("element_offset"), f"{where} element_offset") == 0,
      f"{where} must begin at element offset zero",
    )
  else:
    if shape:
      _require(
        isinstance(strides, list) and len(strides) == len(shape),
        f"{where} element strides must match shape rank",
      )
      layout = {
        "shape": shape,
        "element_strides": strides,
        "element_offset": record.get("element_offset"),
        "storage_span": record.get("storage_span"),
        "axis_order": record.get("axis_order"),
      }
      try:
        tensor_layout.logical_addresses(layout, profile=layout_profile)
      except (tensor_layout.LayoutError, TypeError) as error:
        raise CaseValidationError(f"{where} layout is invalid: {error}") from error
      if profile == PROFILE_V4:
        _require(
          strides == _row_major_strides(shape)
          and record.get("element_offset") == 0
          and record.get("storage_span") == _elements(shape)
          and record.get("axis_order") == [0],
          f"{where} must use a contiguous tensor-v4 layout",
        )
    else:
      _require(
        strides == []
        and record.get("axis_order") == []
        and record.get("element_offset") == 0
        and record.get("storage_span") == 1,
        f"{where} scalar storage must use one contiguous element",
      )
  expected_tile = (
    shape[:]
    if profile == PROFILE_V4
    else tensor_layout.execution_tile_shape(shape, profile=layout_profile)
    if shape
    else []
  )
  _require(
    record.get("tile_shape") == expected_tile,
    f"{where} tile shape does not match its profile",
  )
  _require(
    record.get("tile_origin") == [0] * len(shape), f"{where} tile origin must be zero"
  )
  return {
    "name": record["name"],
    "dtype": dtype,
    "shape": shape,
    "byte_length": length,
    "file": file_name,
    "byte_offset": offset,
  }


def _validate_v4_remainder_inputs(artifacts, inputs, output):
  _require(
    len(inputs) == 2 and all(item["dtype"] == "f32" for item in inputs),
    "tensor-v4 requires two runtime f32 inputs",
  )
  _require(
    inputs[0]["shape"] == inputs[1]["shape"] == output["shape"]
    and output["dtype"] == "f32",
    "tensor-v4 input and output shapes must match",
  )

  def values_for(record):
    start = record["byte_offset"]
    end = start + record["byte_length"]
    return struct.unpack(
      f"<{record['byte_length'] // 4}f", artifacts[record["file"]][start:end]
    )

  def bounded_normal(value):
    return value == 0.0 or (math.isfinite(value) and 2.0**-64 <= abs(value) <= 2.0**16)

  lhs = values_for(inputs[0])
  rhs = values_for(inputs[1])
  expected = values_for(output)
  for index, (a, b, result) in enumerate(zip(lhs, rhs, expected)):
    _require(
      math.isfinite(a) and math.isfinite(b),
      f"tensor-v4 input lane {index} is non-finite",
    )
    _require(b != 0.0, f"tensor-v4 divisor lane {index} is zero")
    _require(
      bounded_normal(a) and bounded_normal(b),
      f"tensor-v4 input lane {index} is outside the normal-value bounds",
    )
    quotient = a / b
    try:
      quotient = struct.unpack("<f", struct.pack("<f", quotient))[0]
    except OverflowError as error:
      raise CaseValidationError(
        f"tensor-v4 quotient lane {index} overflows f32"
      ) from error
    _require(math.isfinite(quotient), f"tensor-v4 quotient lane {index} is non-finite")
    _require(
      bounded_normal(quotient),
      f"tensor-v4 quotient lane {index} is outside the normal-value bounds",
    )
    _require(
      math.isfinite(result) and bounded_normal(result),
      f"tensor-v4 reference lane {index} is outside the normal-value bounds",
    )


def _check_finite(data, dtype, where):
  if dtype != "f32":
    return
  values = struct.unpack(f"<{len(data) // 4}f", data)
  _require(
    all(math.isfinite(value) for value in values),
    f"{where} contains non-finite f32 values",
  )


def _recorded_targets(manifest):
  """Every place a case states its output target, with the field that says so.

  A generated case writes the same target into the input record, the generator
  block, and the property, so a reader has to see them agree before any one of
  them can be checked against the reference output.
  """
  behavior = manifest.get("coverage", {}).get("input_behavior", {})
  recorded = []
  if "property_target" in behavior:
    recorded.append(
      ("coverage.input_behavior.property_target", behavior["property_target"])
    )
  if "property_target" in manifest.get("generator", {}):
    recorded.append(
      ("generator.property_target", manifest["generator"]["property_target"])
    )
  if "target" in manifest.get("property", {}):
    recorded.append(("property.target", manifest["property"]["target"]))
  constraint = manifest.get("property", {}).get("constraint")
  if isinstance(constraint, dict) and "target" in constraint:
    recorded.append(("property.constraint.target", constraint["target"]))
  return recorded


def _check_property_constraint(constraint, output, expected):
  """The typed property has to hold on the case's own reference output.

  This is the same comparison the generator makes before writing a case, so a
  manifest edited after generation, or a case whose expected bytes were
  replaced, is rejected here rather than admitted as a witness.
  """
  _require(isinstance(constraint, dict), "the property constraint must be an object")
  kind = constraint.get("kind")
  _require(
    kind in {"output_equals", "output_sum_equals", "elements_equal"},
    "unknown property constraint kind",
  )
  _require(output["dtype"] == "i32", "a property constraint requires an integer output")
  values = struct.unpack(f"<{len(expected) // 4}i", expected)
  shape = output["shape"]
  if kind == "elements_equal":
    elements = constraint.get("elements")
    _require(
      isinstance(elements, list) and elements,
      "an element constraint needs at least one element",
    )
    complete = constraint.get("complete")
    _require(isinstance(complete, bool), "an element constraint states completeness")
    seen = set()
    for record in elements:
      _require(isinstance(record, dict), "an element record must be an object")
      index = record.get("index")
      value = record.get("value")
      _require(
        isinstance(index, list)
        and len(index) == len(shape)
        and all(isinstance(axis, int) and not isinstance(axis, bool) for axis in index),
        "an element coordinate must state every output dimension",
      )
      _require(
        isinstance(value, int) and not isinstance(value, bool),
        "an element value must be an integer",
      )
      offset = 0
      for axis, coordinate in enumerate(index):
        _require(
          0 <= coordinate < shape[axis],
          "an element coordinate is outside the output shape",
        )
        offset = offset * shape[axis] + coordinate
      _require(tuple(index) not in seen, "an element coordinate is repeated")
      seen.add(tuple(index))
      _require(
        values[offset] == value,
        f"output element {index} is {values[offset]}, not the required {value}",
      )
    if complete:
      total = 1
      for dimension in shape:
        total *= dimension
      _require(
        len(seen) == total,
        "a complete element target must name every output coordinate",
      )
    return
  target = constraint.get("target")
  _require(
    isinstance(target, int) and not isinstance(target, bool),
    "a constraint target must be an integer",
  )
  observed = values[0] if kind == "output_equals" or not shape else sum(values)
  _require(
    observed == target,
    f"the reference output {observed} does not satisfy the stated {kind} of {target}",
  )


def _check_selection_provenance(
  manifest, described_inputs, input_bytes, output, expected
):
  """Cross-check a case's recorded input selection against its own bytes.

  A case that names the policy which proposed its runtime inputs, the leaf
  domain it respected, or the output target it was asked to satisfy has to be
  consistent with what it carries: every copy of the domain and target must
  agree, the domain must contain every input leaf, and the reference output must
  satisfy the stated target. Otherwise the case is a witness for a claim its own
  bytes do not support.
  """
  behavior = manifest.get("coverage", {}).get("input_behavior", {})
  generator = manifest.get("generator", {})
  policy = behavior.get("policy")
  _require(
    policy is None or generator.get("input_policy", policy) == policy,
    "recorded input policy labels disagree",
  )
  sampler_version = behavior.get("sampler_version")
  generator_sampler_version = generator.get("input_sampler_version")
  _require(
    sampler_version == generator_sampler_version,
    "recorded input sampler versions disagree",
  )
  _require(
    sampler_version is None
    or (isinstance(sampler_version, str) and bool(sampler_version)),
    "input sampler version must be a nonempty string",
  )
  domain = behavior.get("input_domain")
  _require(
    domain is None or generator.get("input_domain", domain) == domain,
    "recorded input domains disagree",
  )
  if domain is not None:
    _require(
      isinstance(domain, dict)
      and isinstance(domain.get("lo"), int)
      and isinstance(domain.get("hi"), int)
      and not isinstance(domain["lo"], bool)
      and not isinstance(domain["hi"], bool),
      "input domain must be an integer interval",
    )
    _require(domain["lo"] <= domain["hi"], "input domain is empty")
    for item in described_inputs:
      # The recorded interval bounds every element the policy may propose, so
      # a floating leaf is bound by it as well; only the element width in the
      # buffer differs.
      format_code = {"i32": "i", "f32": "f"}.get(item["dtype"])
      if format_code is None:
        continue
      begin = item["byte_offset"]
      end = begin + item["byte_length"]
      for value in struct.unpack(
        f"<{max(0, end - begin) // 4}{format_code}", input_bytes[begin:end]
      ):
        _require(
          domain["lo"] <= value <= domain["hi"],
          f"input {item['name']} carries {value}, outside the recorded domain",
        )
  recorded = _recorded_targets(manifest)
  constraint = manifest.get("property", {}).get("constraint")
  if constraint is not None:
    _check_property_constraint(constraint, output, expected)
  if not recorded:
    return
  for field, value in recorded:
    _require(
      isinstance(value, int) and not isinstance(value, bool),
      f"{field} must be an integer",
    )
  target = recorded[0][1]
  for field, value in recorded[1:]:
    _require(
      value == target,
      f"{field} states {value} where {recorded[0][0]} states {target}",
    )
  _require(
    output["dtype"] == "i32", "a stated output target requires an integer output"
  )
  values = struct.unpack(f"<{len(expected) // 4}i", expected)
  observed = values[0] if not output["shape"] else sum(values)
  _require(
    observed == target,
    f"expected output {observed} does not satisfy the stated target {target}",
  )


def _describe(source_path, entry, symiri, *, profile):
  command = [str(symiri), "--entry", entry, "--tensor-describe"]
  if profile in {PROFILE_V2, PROFILE_V3, PROFILE_V4}:
    version = {
      PROFILE_V2: "v2",
      PROFILE_V3: "v3",
      PROFILE_V4: "v4",
    }[profile]
    command.extend(["--tensor-profile-version", version])
  command.append(str(source_path))
  try:
    result = subprocess.run(
      command,
      capture_output=True,
      text=True,
      check=False,
      timeout=10,
    )
  except (OSError, subprocess.TimeoutExpired) as error:
    raise CaseValidationError(f"cannot describe case signature: {error}") from error
  if result.returncode != 0:
    raise CaseValidationError(
      f"source signature validation failed: {result.stderr.strip()}"
    )
  try:
    return json.loads(result.stdout)
  except json.JSONDecodeError as error:
    raise CaseValidationError(
      "symiri returned invalid tensor signature JSON"
    ) from error


def _signature_record(record):
  return {key: record.get(key) for key in ("name", "dtype", "shape", "byte_length")}


def _certificate_files(value):
  """Counted artifact names a certificate's records refer to.

  The exact-f32 validator reads the counted runtime buffers through the records
  its caller resolves, and the recorded solver selection by name, so the files a
  certificate depends on are exactly the names those records state.
  """
  names = set()

  def walk(item):
    if isinstance(item, dict):
      for key, entry in item.items():
        if key.endswith("_file"):
          names.add(entry)
        elif not isinstance(entry, str):
          walk(entry)
    elif isinstance(item, list):
      for entry in item:
        walk(entry)

  walk(value)
  return names


def _counted_derivation_files(certificate):
  """The counted artifact names a companion has to carry for its derivation.

  The counted program proves the certificate was written for the program the
  companion unrolls, the buffers the certificate is checked against are the
  counted runtime bytes, and the selection artifacts are what a checked solver
  witness is made of. Requiring them as digested artifacts of the companion's own
  case leaves the relationship checkable without the counted directory.
  """
  return sorted(
    {"program.sir", "inputs.bin", "expected.bin"} | _certificate_files(certificate)
  )


def _verified_counted_artifacts(manifest, artifacts, derivation):
  """Resolve the carried counted artifacts and check them against their records.

  Every carried artifact has to be declared by the companion's own file map, so
  the bytes this check reads are the digested bytes the loader verified.
  """
  name_map = derivation["of"].get("files")
  _require(
    isinstance(name_map, dict)
    and bool(name_map)
    and all(isinstance(key, str) and key for key in name_map)
    and all(isinstance(value, str) and value for value in name_map.values()),
    "the derivation's counted artifacts must map counted names to carried files",
  )
  certificate = derivation["of"].get("certificate")
  required = _counted_derivation_files(
    certificate if isinstance(certificate, dict) else {}
  )
  _require(
    set(name_map) == set(required),
    "the derivation must carry the counted artifacts it depends on, and only "
    "those: " + ", ".join(required),
  )
  carried = {}
  for name in required:
    path = name_map[name]
    _require(
      path in artifacts,
      f"the derivation's counted artifacts must carry {name} as a file of the case",
    )
    record = manifest["files"].get(path)
    _require(
      isinstance(record, dict)
      and record.get("bytes") == len(artifacts[path])
      and record.get("sha256") == hashlib.sha256(artifacts[path]).hexdigest(),
      f"the derivation's counted artifact {name} does not match its file record",
    )
    carried[name] = artifacts[path]
  return carried


def _compare_counted_bytes(carried, artifacts, manifest):
  """Require the companion's inherited runtime bytes to be the counted ones.

  `inputs` is the companion's whole runtime input record, checked here against
  the carried counted bytes; `load_case` has already made it a single contiguous
  buffer that spans `inputs.bin`, so comparing the file compares what a run
  stages.
  """
  _require(
    carried["inputs.bin"] == artifacts["inputs.bin"],
    "the companion's runtime inputs are not the counted runtime inputs",
  )
  _require(
    carried["expected.bin"] == artifacts["expected.bin"],
    "the companion's reference is not the counted reference",
  )
  inputs = manifest.get("inputs")
  _require(
    isinstance(inputs, list)
    and len(inputs) == 1
    and inputs[0].get("file") == "inputs.bin",
    "the companion must stage the counted runtime input as its one input buffer",
  )


def _validate_mechanical_unrolling(derivation, carried):
  """Check the companion source is the supported unrolling of the counted one."""
  from test.lib import tensor_region_generation

  try:
    counted_source = carried["program.sir"].decode("utf-8")
    expected = tensor_region_generation.unroll_source(counted_source)
  except (UnicodeDecodeError, ValueError) as error:
    raise CaseValidationError(
      f"the counted source does not support the derivation's unrolling: {error}"
    ) from error
  _require(
    derivation["unrolled_source_sha256"]
    == hashlib.sha256(expected.encode("utf-8")).hexdigest(),
    "the companion source is not the mechanical unrolling of the counted source",
  )


def _validate_counted_certificate(manifest, derivation, carried, artifacts, output):
  """Validate the carried counted certificate the way the counted path does.

  The counted loader proves a counted case by checking the whole-output
  certificate against the counted source, its runtime input bytes and its
  reference; a companion's derivation stands only if the certificate it carries
  passes that same check. The view below is the counted case that derivation
  describes: the counted program, the companion's input and reference bytes,
  which the byte comparison has already shown to be the counted ones.
  """
  certificate = derivation["of"].get("certificate")
  _require(
    isinstance(certificate, dict),
    "a bit-exact comparison requires the counted certificate it derives from",
  )
  _require(
    manifest["comparison"].get("policy") == "derived-counted-certificate-bit-exact",
    "a bit-exact companion derivation must record its comparison policy",
  )
  mirror = {"program.sir": carried["program.sir"]}
  for name in _certificate_files(certificate):
    mirror[name] = carried[name]
  mirror["inputs.bin"] = artifacts["inputs.bin"]
  mirror["expected.bin"] = artifacts["expected.bin"]
  # The counted source's operations are the certificate's own claim about the
  # program it certifies, so they are carried beside the certificate rather than
  # taken from the unrolled source, whose operations need not be listed the same
  # way.
  counted_view = {
    "reference": {"exactness": certificate},
    "operator_coverage": derivation["of"].get("operator_coverage"),
  }
  try:
    _validate_exact_f32_output_proof(
      carried["program.sir"].decode("utf-8"),
      counted_view,
      output,
      "exact",
      mirror,
      manifest["inputs"],
    )
  except (UnicodeDecodeError, ValueError) as error:
    raise CaseValidationError(
      f"the counted certificate is not valid for the counted source: {error}"
    ) from error


def _validate_exact_f32_output_proof(
  source, manifest, output, mode, artifacts, described_inputs
):
  reference = manifest["reference"]
  proof = reference.get("exactness")
  if output["dtype"] != "f32" or mode != "exact":
    _require(proof is None, "exactness evidence is only valid for exact f32 outputs")
    return

  _require(isinstance(proof, dict), "exact f32 output requires a whole-output proof")
  _require(
    proof.get("complete_output") is True,
    "exactness proof must cover the complete output",
  )
  if proof.get("kind") in {
    "composed-add-sum-f32-v1",
    "bounded-region-add-sum-f32-v1",
  }:
    from test.lib import tensor_exact_f32

    expected_operators = (
      tensor_exact_f32.OPERATIONS
      if proof.get("kind") == tensor_exact_f32.KIND
      else tensor_exact_f32.generated_region_operations(source)
    )
    _require(
      len(described_inputs) == 1
      and described_inputs[0]["name"] == "x"
      and described_inputs[0]["dtype"] == "f32"
      and described_inputs[0]["shape"] == output["shape"],
      "composed proof requires one shape-matched runtime f32 input",
    )
    _require(
      manifest.get("operator_coverage") == expected_operators,
      "composed exactness proof disagrees with source operations",
    )
    selection = proof.get("selection")
    _require(
      selection is None or isinstance(selection, dict),
      "composed exactness proof has malformed solver selection evidence",
    )
    if selection is None:
      # A rule-chosen input has no solver witness. The generated-region
      # certificate is self-contained: it recertifies from the source and the
      # recorded domain, and the complete expected bytes are recomputed. The
      # authored composed loop keeps its selection, which is its only statement
      # of how the runtime input was chosen.
      _require(
        proof.get("kind") == tensor_exact_f32.REGION_KIND,
        "composed exactness proof requires checked solver selection evidence",
      )
      recorded_domain = (
        manifest.get("coverage", {}).get("input_behavior", {}).get("input_domain")
      )
      _require(
        recorded_domain is None
        or proof.get("input_domain")
        == [recorded_domain.get("lo"), recorded_domain.get("hi")],
        "exactness proof domain disagrees with the recorded input domain",
      )
    input_record = described_inputs[0]
    input_bytes = artifacts[input_record["file"]][
      input_record["byte_offset"] : input_record["byte_offset"]
      + input_record["byte_length"]
    ]
    expected_bytes = artifacts[output["file"]][
      output["byte_offset"] : output["byte_offset"] + output["byte_length"]
    ]
    try:
      tensor_exact_f32.validate(
        source, input_bytes, expected_bytes, proof, artifacts=artifacts
      )
    except ValueError as error:
      raise CaseValidationError(str(error)) from error
    return
  _require(
    proof.get("kind") in {"bit-preserving-f32-layout", "integer-valued-f32-reduction"},
    "unknown exact f32 proof family",
  )
  operations = re.findall(
    r"^\s*%([A-Za-z_][A-Za-z_0-9]*)\s*=\s*tensor\s+([A-Za-z_][A-Za-z_0-9]*)\b",
    source,
    re.MULTILINE,
  )
  returned = re.findall(
    r"^\s*ret\s+(%[A-Za-z_][A-Za-z_0-9]*)\s*;", source, re.MULTILINE
  )
  _require(
    len(returned) == 1 and len(operations) == 1,
    "exact f32 proof requires one returned operation",
  )
  destination, operation = operations[0]
  _require(
    returned[0] == f"%{destination}",
    "exact f32 proof operation must directly produce the return",
  )
  declared_operations = manifest.get("operator_coverage")
  _require(
    declared_operations == [operation],
    "exact f32 proof disagrees with source operations",
  )

  if proof["kind"] == "bit-preserving-f32-layout":
    _require(operation == "transpose", "layout proof permits only transpose")
    _require(
      len(described_inputs) == 1 and described_inputs[0]["dtype"] == "f32",
      "layout proof requires one runtime f32 input",
    )
    _require(
      proof.get("allowed_operations") == ["transpose"]
      and proof.get("signed_zero") == "preserve-bits",
      "bit-preserving layout evidence is incomplete",
    )
    return

  _require(
    operation == "sum", "exact reduction proof must return the reduction directly"
  )
  _require(
    proof.get("leaf_values") == "finite-nonzero-integer-valued-f32",
    "exact reduction proof requires finite, nonzero, integer-valued f32 leaves",
  )
  _require(
    type(proof.get("max_abs_leaf_sum")) is int
    and 0 < proof["max_abs_leaf_sum"] <= 2**24
    and proof.get("integer_exactness_bound") == 2**24,
    "exact reduction leaf-sum bound is invalid",
  )
  _require(
    proof.get("before_operations") == ["input"]
    and proof.get("after_operations") == []
    and proof.get("signed_zero") == "positive-zero-only",
    "exact reduction proof does not establish the full output policy",
  )
  _require(
    len(described_inputs) == 1 and described_inputs[0]["dtype"] == "f32",
    "exact reduction proof requires one runtime f32 input",
  )
  input_record = described_inputs[0]
  input_bytes = artifacts[input_record["file"]][
    input_record["byte_offset"] : input_record["byte_offset"]
    + input_record["byte_length"]
  ]
  leaf_values = struct.unpack(f"<{len(input_bytes) // 4}f", input_bytes)
  _require(
    all(
      math.isfinite(value) and value != 0.0 and value.is_integer()
      for value in leaf_values
    ),
    "exact reduction input contains a non-finite, zero, or non-integer leaf",
  )
  actual_leaf_bound = sum(abs(int(value)) for value in leaf_values)
  _require(
    actual_leaf_bound == proof["max_abs_leaf_sum"],
    "exact reduction proof bound does not match runtime input bytes",
  )
  expected_bytes = artifacts[output["file"]][
    output["byte_offset"] : output["byte_offset"] + output["byte_length"]
  ]
  expected_words = struct.unpack(f"<{len(expected_bytes) // 4}I", expected_bytes)
  expected_values = struct.unpack(f"<{len(expected_bytes) // 4}f", expected_bytes)
  _require(
    all(
      value != 0.0 or word == 0 for value, word in zip(expected_values, expected_words)
    ),
    "exact reduction output contains negative zero",
  )


def _validate_companion_derivation(manifest, artifacts, output):
  """Check the derivation a bit-exact companion states instead of a proof.

  The unrolled body has no whole-output exactness proof of its own, so the
  companion's bit-exact contract is only as good as its derivation from the
  counted case it unrolls. The derivation has to be checkable from the companion
  alone, which is why the companion carries the counted program, its runtime
  bytes, and its certificate as digested artifacts of its own case. Four
  relationships are then established here: those artifacts match their records,
  the inherited runtime input and reference bytes are the counted ones, the
  companion source is the supported mechanical unrolling of the counted program,
  and the counted certificate validates against the counted source the way the
  counted loader validates it. A derivation that cannot show all four is refused,
  and the bit-exact comparison is not granted.

  The counted program and manifest digests are the identity of the counted case;
  a standalone companion cannot recompute the manifest digest, so it is checked
  for syntax and cross-checked through the artifacts above.
  """
  derivation = manifest.get("companion")
  _require(
    isinstance(derivation, dict),
    "a bit-exact comparison requires a companion derivation record",
  )
  counted = derivation.get("of")
  _require(isinstance(counted, dict), "the derivation must name its counted case")
  _require(
    counted.get("entry") == manifest["entry"]
    and counted.get("profile") == manifest["profile"],
    "the derivation names a counted case with another entry or profile",
  )
  for field in ("program_sha256", "manifest_sha256", "certificate_sha256"):
    digest = counted.get(field)
    _require(
      isinstance(digest, str)
      and len(digest) == 64
      and all(char in "0123456789abcdef" for char in digest),
      f"the derivation's counted {field} must be a SHA-256 digest",
    )
  _require(
    counted.get("reference_sha256")
    == hashlib.sha256(artifacts["expected.bin"]).hexdigest(),
    "the derived reference is not the counted reference it claims",
  )
  transformation = derivation.get("transformation")
  _require(
    transformation == COMPANION_TRANSFORMATION,
    "the derivation must name the supported mechanical unrolling",
  )
  _require(
    derivation.get("unrolled_source_sha256")
    == hashlib.sha256(artifacts["program.sir"]).hexdigest(),
    "the derivation does not digest the unrolled source it carries",
  )
  for field in ("counted_trips", "unrolled_trips", "unrolled_body_copies"):
    _integer(derivation.get(field), f"derivation {field}", 1)
  _require(
    derivation["unrolled_body_copies"] == derivation["counted_trips"]
    and derivation["unrolled_trips"] == 1
    and derivation["counted_trips"] >= 2,
    "the derivation's trip counts do not describe a counted loop unrolled once",
  )
  inherited = derivation.get("inherited_files")
  _require(
    isinstance(inherited, list)
    and "inputs.bin" in inherited
    and all(name in artifacts for name in inherited),
    "the derivation must inherit the counted runtime inputs this case carries",
  )
  carried = _verified_counted_artifacts(manifest, artifacts, derivation)
  _compare_counted_bytes(carried, artifacts, manifest)
  _validate_mechanical_unrolling(derivation, carried)
  _validate_counted_certificate(manifest, derivation, carried, artifacts, output)
  replays = derivation.get("replays")
  _require(
    isinstance(replays, list) and all(isinstance(replay, dict) for replay in replays),
    "the derivation must record one object per replay",
  )
  _require(
    [replay.get("reference_bits") for replay in replays] == NUMERIC_PRECISIONS,
    "the derivation must record one replay per reference precision",
  )
  for index, replay in enumerate(replays):
    name = replay.get("file")
    _require(
      isinstance(name, str) and name in artifacts,
      f"replay {index} names a file the case does not carry",
    )
    _require(
      replay.get("sha256") == hashlib.sha256(artifacts[name]).hexdigest(),
      f"replay {index} does not match the file it names",
    )
    _require(
      artifacts[name] == artifacts["expected.bin"],
      f"replay {index} does not reproduce the reference",
    )
    command = replay.get("command")
    _require(
      isinstance(command, list)
      and bool(command)
      and all(isinstance(part, str) for part in command),
      f"replay {index} must record the command that produced it",
    )


def _validate_bit_exact_companion_comparison(
  manifest, artifacts, mode, comparison, output
):
  """Admit a signed-zero-exact comparison only for a derived companion.

  An ordinary tensor-v2 case compares numerically, and a zero tolerance does not
  make that bit-exact: `+0.0` and `-0.0` differ in bits but not in value, so the
  comparator's signed-zero rule is what turns a zero-tolerance comparison into a
  bit-exact contract -- every other pair of distinct f32 bit patterns is a pair
  of distinct values, and the rule compares the two zeros by word. A companion's
  unrolled body has no whole-output exactness proof, so it earns the rule the
  other way round, by recording a derivation from a counted case's certificate.
  """
  _require(
    mode == "numeric" and comparison.get("zero_sign_exact") is True,
    "zero_sign_exact is reserved for tensor-v4 or a companion derivation",
  )
  _require(
    comparison["atol"] == [0, 1] and comparison["rtol"] == [0, 1],
    "a companion derivation must compare at zero tolerances",
  )
  _validate_companion_derivation(manifest, artifacts, output)


def load_case(case_path, *, symiri=DEFAULT_SYMIRI):
  """Load and cross-check a tensor case before any target process starts."""
  root = Path(case_path).resolve()
  manifest_path = root / "case.json"
  try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise CaseValidationError(f"cannot read case.json: {error}") from error
  _require(isinstance(manifest, dict), "case manifest must be an object")
  schema = manifest.get("schema")
  profile = manifest.get("profile")
  _require(
    (schema, profile)
    in {
      (SCHEMA_V1, PROFILE_V1),
      (SCHEMA_V2, PROFILE_V2),
      (SCHEMA_V3, PROFILE_V3),
      (SCHEMA_V4, PROFILE_V4),
    },
    "unsupported tensor case schema or profile",
  )
  entry = manifest.get("entry")
  _require(isinstance(entry, str) and entry, "entry is missing")
  artifacts = _read_artifacts(root, manifest, schema=schema)

  inputs = manifest.get("inputs")
  _require(isinstance(inputs, list), "inputs must be an ordered list")
  described_inputs = [
    _validate_buffer(item, f"input {index}", artifacts, profile=profile)
    for index, item in enumerate(inputs)
  ]
  output = _validate_buffer(
    manifest.get("output"), "output", artifacts, profile=profile
  )
  _require(output["name"] == "return", "output name must be 'return'")
  for item in described_inputs:
    if item["dtype"] == "f32":
      begin = item["byte_offset"]
      end = begin + item["byte_length"]
      _check_finite(artifacts[item["file"]][begin:end], "f32", f"input {item['name']}")
  if output["dtype"] == "f32":
    _check_finite(
      artifacts[output["file"]][
        output["byte_offset"] : output["byte_offset"] + output["byte_length"]
      ],
      "f32",
      "expected output",
    )

  _check_selection_provenance(
    manifest,
    described_inputs,
    artifacts["inputs.bin"] if inputs else b"",
    output,
    artifacts[output["file"]][
      output["byte_offset"] : output["byte_offset"] + output["byte_length"]
    ],
  )

  if inputs:
    _require(
      all(item["file"] == "inputs.bin" for item in described_inputs),
      "runtime parameters must use inputs.bin",
    )
    expected_offset = 0
    for item in described_inputs:
      _require(
        item["byte_offset"] == expected_offset,
        "input records must be concatenated in declaration order",
      )
      expected_offset += item["byte_length"]
    _require(
      expected_offset == len(artifacts["inputs.bin"]),
      "inputs.bin has trailing or missing bytes",
    )
  else:
    _require(
      len(artifacts["inputs.bin"]) == 0,
      "inputs.bin must be empty for a zero-parameter entry",
    )
  _require(
    output["file"] == "expected.bin" and output["byte_offset"] == 0,
    "expected output must begin at expected.bin offset zero",
  )
  _require(
    len(artifacts["expected.bin"]) == output["byte_length"],
    "expected.bin has trailing or missing bytes",
  )

  grid = manifest.get("grid")
  _require(grid == [1, 1, 1], "tensor profiles require a unit three-dimensional grid")
  lowering = manifest.get("softmax_lowering")
  _require(
    lowering in {"direct", "decomposed"},
    "softmax_lowering must be direct or decomposed",
  )
  launch = manifest.get("launch")
  if profile == PROFILE_V1:
    _require(
      launch == {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False},
      "launch settings do not match tensor-v1",
    )
  else:
    _require(
      isinstance(launch, dict)
      and type(launch.get("num_warps")) is int
      and launch.get("num_warps") in {1, 2, 4, 8}
      and launch.get("num_stages") == 1
      and launch.get("enable_fp_fusion") is False,
      f"launch settings do not match {profile}",
    )
    if profile == PROFILE_V4:
      _require(launch["num_warps"] == 1, "tensor-v4 calibration requires num_warps=1")
    execution = manifest.get("execution")
    _require(isinstance(execution, dict), f"{profile} execution record is required")
    for key in ("gap_sentinel_by_dtype", "output_canary_by_dtype"):
      values = execution.get(key)
      _require(isinstance(values, dict), f"execution {key} is required")
      for dtype in _DTYPE_SIZE:
        value = values.get(dtype)
        _require(
          isinstance(value, str)
          and len(value) == 8
          and all(char in "0123456789abcdefABCDEF" for char in value),
          f"execution {key} must contain four-byte hex values for {dtype}",
        )
    _require(
      execution["gap_sentinel_by_dtype"]["i32"]
      != execution["output_canary_by_dtype"]["i32"],
      "input gap sentinel and output canary must differ",
    )
    _require(
      execution["gap_sentinel_by_dtype"]["f32"]
      != execution["output_canary_by_dtype"]["f32"],
      "input gap sentinel and output canary must differ",
    )
    warp_probe = execution.get("warp_probe")
    _require(isinstance(warp_probe, dict), f"{profile} warp probe record is required")
    _require(
      warp_probe.get("status") in {"not_probed", "probed"}
      and type(warp_probe.get("num_warps")) is int
      and warp_probe["num_warps"] == launch["num_warps"],
      "warp probe status or selected num_warps is invalid",
    )
    if warp_probe["status"] == "not_probed":
      _require(
        "results" not in warp_probe,
        "unprobed warp metadata cannot contain results",
      )
    else:
      results = warp_probe.get("results")
      _require(isinstance(results, list), "probed warp settings require results")
      classifications = {
        "pass",
        "candidate",
        "resource_limit",
        "environment_unavailable",
        "unsupported_configuration",
        "invalid_case",
      }
      for index, result in enumerate(results):
        _require(isinstance(result, dict), f"warp result {index} must be an object")
        count = result.get("num_warps")
        _require(
          type(count) is int and count > 0,
          f"warp result {index} has an invalid num_warps",
        )
        _require(
          result.get("classification") in classifications,
          f"warp result {index} has an invalid classification",
        )
        _require(
          isinstance(result.get("status"), str),
          f"warp result {index} has no raw status",
        )
        _require(
          result.get("case_dir") is None or isinstance(result.get("case_dir"), str),
          f"warp result {index} has an invalid case directory",
        )
  prop = manifest.get("property")
  _require(
    isinstance(prop, dict)
    and isinstance(prop.get("id"), str)
    and isinstance(prop.get("source"), str),
    "property id/source are required",
  )
  evidence = manifest.get("evidence")
  _require(
    isinstance(evidence, dict)
    and isinstance(evidence.get("kind"), str)
    and isinstance(evidence.get("status"), str)
    and isinstance(evidence.get("artifacts"), list),
    "evidence kind/status/artifacts are required",
  )
  allowed_evidence_statuses = {
    "construction": {"validated", "construction_checked"},
    "numerical_reference": {"validated", "numerical_reference_checked"},
    "exact_smt": {"validated", "exact_smt_checked"},
    "error_directed": {"error_directed_checked"},
    "handwritten": {"validated"},
    "conformance_matrix": {"validated"},
    "historical_regression": {"validated"},
    "authored_calibration": {"validated"},
  }
  _require(
    evidence["kind"] in allowed_evidence_statuses
    and evidence["status"] in allowed_evidence_statuses[evidence["kind"]],
    "evidence status does not represent an accepted case",
  )
  if profile == PROFILE_V3:
    pattern = manifest.get("coverage", {}).get("input_behavior", {}).get("pattern")
    _require(
      pattern
      in {
        "coordinate-distinct",
        "mixed-sign",
        "cancellation",
        "repeated-value",
        "distinct-groups",
        "seeded-random",
        # Sampler labels: values drawn by the boundary walk or chosen by the
        # exact input policy rather than built from a named value family.
        "boundary-value",
        "smt-selected",
      },
      "tensor-v3 input pattern is missing or unknown",
    )
    _require(
      manifest.get("generator", {}).get("input_pattern") == pattern,
      "tensor-v3 input pattern labels disagree",
    )
  comparison = manifest.get("comparison")
  _require(isinstance(comparison, dict), "comparison policy is required")
  mode = comparison.get("mode")
  _require(mode in {"exact", "numeric"}, "comparison mode must be exact or numeric")
  _validate_rational_pair(comparison.get("atol"), "atol")
  _validate_rational_pair(comparison.get("rtol"), "rtol")
  _require(
    mode != "numeric" or output["dtype"] == "f32",
    "numeric comparison requires f32 output",
  )
  _require(
    mode != "exact" or (comparison["atol"] == [0, 1] and comparison["rtol"] == [0, 1]),
    "exact comparison must use zero tolerances",
  )
  if profile == PROFILE_V4:
    _require(
      comparison.get("zero_sign_exact") is True,
      "tensor-v4 requires exact signed-zero handling in both comparison modes",
    )
    _validate_v4_remainder_inputs(artifacts, described_inputs, output)
  elif "zero_sign_exact" in comparison:
    _validate_bit_exact_companion_comparison(
      manifest, artifacts, mode, comparison, output
    )
  else:
    _require(
      "companion" not in manifest,
      "a companion record needs the bit-exact comparison its derivation grants",
    )
  reference = manifest.get("reference")
  _require(
    isinstance(reference, dict) and type(reference.get("stable")) is bool,
    "reference stability metadata is required",
  )
  _require(reference["stable"] is True, "unstable references cannot be executed")
  expected_precisions = NUMERIC_PRECISIONS if mode == "numeric" else []
  _require(
    reference.get("precisions") == expected_precisions,
    "reference precision metadata does not match comparison mode",
  )
  if profile in {PROFILE_V2, PROFILE_V3}:
    _validate_exact_f32_output_proof(
      artifacts["program.sir"].decode("utf-8"),
      manifest,
      output,
      mode,
      artifacts,
      described_inputs,
    )
  if profile == PROFILE_V3:
    execution = manifest["execution"]
    canary = bytes.fromhex(execution["output_canary_by_dtype"][output["dtype"]])
    expected = artifacts[output["file"]][
      output["byte_offset"] : output["byte_offset"] + output["byte_length"]
    ]
    _require(
      all(
        expected[index : index + len(canary)] != canary
        for index in range(0, len(expected), len(canary))
      ),
      "output canary must differ bitwise from every expected logical output value",
    )
  environment = manifest.get("environment")
  _require(
    environment is None or isinstance(environment, dict),
    "environment must be null or an artifact reference",
  )

  signature = _describe(root / "program.sir", entry, symiri, profile=profile)
  _require(signature.get("entry") == entry, "described entry does not match manifest")
  actual_inputs = signature.get("inputs")
  _require(isinstance(actual_inputs, list), "described input signature is missing")
  _require(
    [_signature_record(item) for item in described_inputs]
    == [_signature_record(item) for item in actual_inputs],
    "manifest input shape/type/order differs from checked source signature",
  )
  _require(
    _signature_record(output) == _signature_record(signature.get("output", {})),
    "manifest output shape/type differs from checked source signature",
  )

  intervals = {}
  for item in (*described_inputs, output):
    start = item["byte_offset"]
    end = start + item["byte_length"]
    for previous in intervals.setdefault(item["file"], []):
      _require(
        end <= previous[0] or start >= previous[1], "input/output buffers overlap"
      )
    intervals[item["file"]].append((start, end))

  return TensorCase(
    root,
    manifest,
    artifacts["program.sir"],
    artifacts["inputs.bin"],
    artifacts["expected.bin"],
    signature,
  )
