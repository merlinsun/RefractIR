"""Load, validate, and write target-only tensor bundles.

A bundle states one kernel, the bytes it consumes, and the physical layout and
launch settings a target run needs. It carries no expected bytes and no
comparison policy, so a consumer of this format has nothing to compare against
and cannot report an agreement it did not establish. `load` refuses a bundle
that declares either claim, that is missing a declared artifact, or whose
declared digests do not match the bytes on disk. It also runs the native
frontend, because a bundle is executable only when the interpreter accepts its
source.
"""

import copy
import hashlib
import itertools
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "refractir.tensor-target-case.v1"
KIND = "target-only"
ENTRY = "kernel"
CASE_FILE = "case.json"
PROGRAM_FILE = "program.sir"
INPUTS_FILE = "inputs.bin"
REFERENCE_FILE = "expected.bin"
PROFILE_V2 = "tensor-v2"
PROFILE_V3 = "tensor-v3"
PROFILES = {PROFILE_V2: "v2", PROFILE_V3: "v3"}
DTYPES = ("f32", "i32")
ELEMENT_SIZE = 4
WARP_COUNTS = (1, 2, 4, 8)

# Refusal statuses are stable, so a consumer classifies a refusal without
# parsing the reason text.
INVALID_CASE = "invalid_case"
DIGEST_MISMATCH = "digest_mismatch"
INVALID_LAYOUT = "invalid_layout"
UNSUPPORTED_GRID = "unsupported_grid"
INVALID_SOURCE = "invalid_source"
FRONTEND_UNAVAILABLE = "frontend_unavailable"
REFERENCE_NOT_ALLOWED = "reference_not_allowed"
SIGNATURE_MISMATCH = "signature_mismatch"

_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX_8 = re.compile(r"[0-9a-fA-F]{8}\Z")
_INPUT_FIELDS = (
  "name",
  "dtype",
  "shape",
  "element_strides",
  "element_offset",
  "storage_span",
  "axis_order",
)
_OUTPUT_FIELDS = _INPUT_FIELDS


class TargetCaseError(ValueError):
  """A target-only bundle is not an executable input to a target run."""

  def __init__(self, status, reason):
    super().__init__(reason)
    self.status = status
    self.reason = reason


@dataclass(frozen=True)
class TargetCase:
  """One validated target-only bundle.

  `record` is the parsed case.json, `source` is the text of program.sir,
  `inputs` is the bytes of inputs.bin, and `path` is the bundle directory. No
  expected bytes are carried, because this format has no reference.
  """

  path: Path
  record: dict
  source: str
  inputs: bytes
  signature: dict | None = None

  @property
  def manifest(self):
    """The parsed record under the name the shared emission helper reads."""
    return self.record

  def worker_manifest(self):
    """The manifest the shared worker reads for this bundle.

    The records restate the bundle's own layout, launch, and execution fields.
    The schema travels with them so a recorded worker run can be traced to the
    format that produced it; the worker does not read the schema. The returned
    dict is a copy, so a caller cannot change the case through it.
    """
    return {
      "schema": SCHEMA,
      "kind": KIND,
      "profile": self.record["profile"],
      "entry": self.record["entry"],
      "inputs": [dict(record) for record in self.record["inputs"]],
      "output": dict(self.record["output"]),
      "grid": list(self.record["grid"]),
      "launch": dict(self.record["launch"]),
      "execution": copy.deepcopy(self.record["execution"]),
    }


def load(bundle_dir, *, symiri=None):
  """Load and validate one target-only bundle directory.

  Structural validity, artifact digests, and frontend acceptance are all
  required, so a returned case carries the source and bytes its record
  declares. A refusal names the first failed contract and carries a stable
  status.
  """
  root = Path(bundle_dir).resolve()
  manifest = _read_manifest(root)
  source_bytes = _read_artifact(root, PROGRAM_FILE)
  inputs_bytes = _read_artifact(root, INPUTS_FILE)
  _validate_bundle(
    manifest, source_bytes=source_bytes, inputs_bytes=inputs_bytes, root=root
  )
  try:
    source = source_bytes.decode("utf-8")
  except UnicodeDecodeError as error:
    raise TargetCaseError(
      INVALID_SOURCE, f"program.sir is not UTF-8 text: {error}"
    ) from error
  signature = _check_frontend(manifest["profile"], root=root, symiri=symiri)
  _validate_signature(manifest, signature)
  return TargetCase(
    path=root,
    record=manifest,
    source=source,
    inputs=inputs_bytes,
    signature=signature,
  )


def write(
  bundle_dir,
  *,
  source,
  inputs,
  output,
  launch,
  execution,
  profile=PROFILE_V2,
):
  """Write one target-only bundle and return the case it declares.

  Each input record carries the bytes it occupies under `data`; the writer
  concatenates those bytes into inputs.bin and assigns the placement and digest
  fields, so a caller never supplies either. The frozen entry, softmax lowering,
  and one-tile grid are written as the schema states them. The record is
  validated before anything is written. Frontend acceptance is left to `load`,
  which owns the native interpreter.
  """
  _require(isinstance(source, str), INVALID_CASE, "source must be text")
  source_bytes = source.encode("utf-8")
  _require(isinstance(inputs, (list, tuple)), INVALID_CASE, "inputs must be a list")
  records = []
  chunks = []
  offset = 0
  for index, spec in enumerate(inputs):
    where = f"input {index}"
    record = _layout_record(spec, _INPUT_FIELDS, where)
    _require("data" in spec, INVALID_CASE, f"{where} data is missing")
    data = spec["data"]
    _require(
      isinstance(data, (bytes, bytearray)),
      INVALID_CASE,
      f"{where} data must be bytes",
    )
    data = bytes(data)
    record["file"] = INPUTS_FILE
    record["byte_offset"] = offset
    record["byte_length"] = len(data)
    records.append(record)
    chunks.append(data)
    offset += len(data)
  inputs_bytes = b"".join(chunks)

  output_record = _layout_record(output, _OUTPUT_FIELDS, "output")
  _validate_layout(output_record, "output")
  output_record["byte_length"] = _element_count(output_record["shape"]) * ELEMENT_SIZE

  manifest = {
    "schema": SCHEMA,
    "kind": KIND,
    "profile": profile,
    "entry": ENTRY,
    "source": {"file": PROGRAM_FILE, "sha256": _sha256(source_bytes)},
    "inputs": records,
    "inputs_bin": {
      "file": INPUTS_FILE,
      "sha256": _sha256(inputs_bytes),
      "byte_length": len(inputs_bytes),
    },
    "output": output_record,
    "grid": [1, 1, 1],
    "softmax_lowering": "direct",
    "launch": copy.deepcopy(launch),
    "execution": copy.deepcopy(execution),
    "reference": {"kind": "none"},
    "comparison": {"kind": "none"},
  }
  _validate_bundle(manifest, source_bytes=source_bytes, inputs_bytes=inputs_bytes)
  try:
    case_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
  except (TypeError, ValueError) as error:
    raise TargetCaseError(
      INVALID_CASE, f"the bundle record is not JSON-serializable: {error}"
    ) from error

  root = Path(bundle_dir)
  root.mkdir(parents=True, exist_ok=True)
  (root / PROGRAM_FILE).write_bytes(source_bytes)
  (root / INPUTS_FILE).write_bytes(inputs_bytes)
  (root / CASE_FILE).write_text(case_text, encoding="utf-8")
  return TargetCase(
    path=root.resolve(), record=manifest, source=source, inputs=inputs_bytes
  )


def _read_manifest(root):
  """Read case.json, refusing a missing or undecodable manifest."""
  try:
    text = (root / CASE_FILE).read_text(encoding="utf-8")
  except OSError as error:
    raise TargetCaseError(INVALID_CASE, f"cannot read {CASE_FILE}: {error}") from error
  try:
    return json.loads(text)
  except json.JSONDecodeError as error:
    raise TargetCaseError(
      INVALID_CASE, f"{CASE_FILE} is not valid JSON: {error}"
    ) from error


def _read_artifact(root, name):
  """Read one artifact the manifest declares, refusing a missing file."""
  try:
    return (root / name).read_bytes()
  except OSError as error:
    raise TargetCaseError(INVALID_CASE, f"cannot read {name}: {error}") from error


def _validate_bundle(manifest, *, source_bytes, inputs_bytes, root=None):
  """Check every contract one target-only bundle states."""
  _require(isinstance(manifest, dict), INVALID_CASE, "case.json must be an object")
  _require(
    manifest.get("schema") == SCHEMA,
    INVALID_CASE,
    f"schema must be {SCHEMA}",
  )
  _require(manifest.get("kind") == KIND, INVALID_CASE, f"kind must be {KIND}")
  _require(
    isinstance(manifest.get("profile"), str) and manifest["profile"] in PROFILES,
    INVALID_CASE,
    f"profile must be one of {sorted(PROFILES)}",
  )
  _require(manifest.get("entry") == ENTRY, INVALID_CASE, f"entry must be {ENTRY}")
  _require(
    manifest.get("softmax_lowering") == "direct",
    INVALID_CASE,
    "softmax_lowering must be direct",
  )
  if root is not None and (Path(root) / REFERENCE_FILE).exists():
    raise TargetCaseError(
      REFERENCE_NOT_ALLOWED, f"a target-only bundle carries no {REFERENCE_FILE}"
    )
  _reject_claim(manifest, "reference")
  _reject_claim(manifest, "comparison")

  source = _object(manifest.get("source"), "source")
  _require(
    source.get("file") == PROGRAM_FILE,
    INVALID_CASE,
    f"source.file must be {PROGRAM_FILE}",
  )
  _require(
    _is_digest(source.get("sha256")),
    INVALID_CASE,
    "source.sha256 must be 64 lowercase hex digits",
  )
  _require(
    _sha256(source_bytes) == source["sha256"],
    DIGEST_MISMATCH,
    "program.sir SHA-256 does not match source.sha256",
  )

  inputs_bin = _object(manifest.get("inputs_bin"), "inputs_bin")
  _require(
    inputs_bin.get("file") == INPUTS_FILE,
    INVALID_CASE,
    f"inputs_bin.file must be {INPUTS_FILE}",
  )
  _require(
    _is_digest(inputs_bin.get("sha256")),
    INVALID_CASE,
    "inputs_bin.sha256 must be 64 lowercase hex digits",
  )
  _require(
    _is_integer(inputs_bin.get("byte_length")),
    INVALID_CASE,
    "inputs_bin.byte_length must be a nonnegative integer",
  )
  _require(
    inputs_bin["byte_length"] == len(inputs_bytes),
    DIGEST_MISMATCH,
    f"inputs.bin is {len(inputs_bytes)} bytes, inputs_bin.byte_length is "
    f"{inputs_bin['byte_length']}",
  )
  _require(
    _sha256(inputs_bytes) == inputs_bin["sha256"],
    DIGEST_MISMATCH,
    "inputs.bin SHA-256 does not match inputs_bin.sha256",
  )

  inputs = manifest.get("inputs")
  _require(isinstance(inputs, list), INVALID_CASE, "inputs must be a list")
  for index, record in enumerate(inputs):
    _validate_input(record, f"input {index}", len(inputs_bytes))

  _validate_output(manifest.get("output"), "output")

  _require("grid" in manifest, INVALID_CASE, "grid is missing")
  grid = manifest["grid"]
  _require(
    isinstance(grid, list)
    and len(grid) == 3
    and all(type(dimension) is int and dimension == 1 for dimension in grid),
    UNSUPPORTED_GRID,
    f"grid must be [1, 1, 1], got {grid!r}",
  )

  _validate_launch(manifest.get("launch"))
  _validate_execution(manifest.get("execution"), launch=manifest["launch"])


def _reject_claim(manifest, field):
  """Refuse any reference or comparison statement beyond the frozen none record."""
  _require(field in manifest, INVALID_CASE, f"{field} is missing")
  claim = manifest[field]
  _require(isinstance(claim, dict), INVALID_CASE, f"{field} must be an object")
  if claim != {"kind": "none"}:
    raise TargetCaseError(
      REFERENCE_NOT_ALLOWED,
      f'a target-only bundle states {field} as {{"kind": "none"}}',
    )


def _validate_input(record, where, inputs_length):
  """Check one input record against its layout and its slice of inputs.bin."""
  _object(record, where)
  _require(
    isinstance(record.get("name"), str) and bool(record["name"]),
    INVALID_CASE,
    f"{where} name is missing",
  )
  _validate_dtype(record.get("dtype"), where)
  _validate_layout(record, where)
  _require(
    record.get("file") == INPUTS_FILE,
    INVALID_CASE,
    f"{where} file must be {INPUTS_FILE}",
  )
  offset = record.get("byte_offset")
  _require(
    _is_integer(offset),
    INVALID_LAYOUT,
    f"{where} byte_offset must be a nonnegative integer",
  )
  length = record.get("byte_length")
  expected = _element_count(record["shape"]) * ELEMENT_SIZE
  _require(
    _is_integer(length) and length == expected,
    INVALID_LAYOUT,
    f"{where} byte_length must be the {expected} logical bytes of its shape, "
    f"got {length!r}",
  )
  _require(
    offset + length <= inputs_length,
    INVALID_LAYOUT,
    f"{where} byte range {offset}..{offset + length} exceeds inputs.bin "
    f"({inputs_length} bytes)",
  )


def _validate_output(record, where):
  """Check the single output record the worker returns."""
  _object(record, where)
  _require(
    record.get("name") == "return",
    INVALID_CASE,
    f'{where} name must be "return"',
  )
  _validate_dtype(record.get("dtype"), where)
  _validate_layout(record, where)
  length = record.get("byte_length")
  expected = _element_count(record["shape"]) * ELEMENT_SIZE
  _require(
    _is_integer(length) and length == expected,
    INVALID_LAYOUT,
    f"{where} byte_length must be the {expected} logical bytes of its shape, "
    f"got {length!r}",
  )


def _validate_dtype(dtype, where):
  _require(
    isinstance(dtype, str) and dtype in DTYPES,
    INVALID_CASE,
    f"{where} dtype must be one of {list(DTYPES)}",
  )


def _validate_layout(record, where):
  """Check one layout record and return the storage span it requires.

  `axis_order` lists axes from fastest to slowest stride, which is the order the
  shared layout helper reads. A bundle that states another order would reach the
  worker and be refused there, so it is refused at the boundary instead.
  """
  shape = record.get("shape")
  _require(isinstance(shape, list), INVALID_LAYOUT, f"{where} shape must be a list")
  for axis, dimension in enumerate(shape):
    _require(
      _is_integer(dimension, 1),
      INVALID_LAYOUT,
      f"{where} shape dimension {axis} must be a positive integer",
    )
  strides = record.get("element_strides")
  _require(
    isinstance(strides, list) and len(strides) == len(shape),
    INVALID_LAYOUT,
    f"{where} element_strides must match the shape rank",
  )
  for axis, stride in enumerate(strides):
    _require(
      _is_integer(stride, 1),
      INVALID_LAYOUT,
      f"{where} element stride {axis} must be a positive integer",
    )
  offset = record.get("element_offset")
  _require(
    _is_integer(offset),
    INVALID_LAYOUT,
    f"{where} element_offset must be a nonnegative integer",
  )
  order = record.get("axis_order")
  _require(
    isinstance(order, list)
    and len(order) == len(shape)
    and all(_is_integer(axis) for axis in order)
    and sorted(order) == list(range(len(shape))),
    INVALID_LAYOUT,
    f"{where} axis_order must be a permutation of the tensor axes",
  )
  stride_order = sorted(range(len(shape)), key=lambda axis: (strides[axis], axis))
  _require(
    order == stride_order,
    INVALID_LAYOUT,
    f"{where} axis_order must list axes from fastest to slowest stride "
    f"({stride_order}), got {order}",
  )
  span = record.get("storage_span")
  required = _required_span(shape, strides, offset)
  _require(
    _is_integer(span) and span == required,
    INVALID_LAYOUT,
    f"{where} storage_span must be {required} for its strides, got {span!r}",
  )
  addresses = [
    offset + sum(coordinate * stride for coordinate, stride in zip(index, strides))
    for index in itertools.product(*(range(dimension) for dimension in shape))
  ]
  _require(
    len(set(addresses)) == len(addresses),
    INVALID_LAYOUT,
    f"{where} element strides overlap: two logical coordinates share an address",
  )
  return span


def _element_count(shape):
  """The logical element count of one tensor shape."""
  count = 1
  for dimension in shape:
    count *= dimension
  return count


def _required_span(shape, strides, offset):
  """The physical element count one layout needs: its last address plus one.

  This is the span formula the native exporter writes into a v2 buffer record,
  so a bundle and a native case describing one layout agree on it. A
  layout-free allocation would count elements instead; this schema names only
  layout profiles.
  """
  return (
    offset
    + sum((dimension - 1) * stride for dimension, stride in zip(shape, strides))
    + 1
  )


def _validate_launch(launch):
  """Check the launch settings the shared worker passes to Triton."""
  _object(launch, "launch")
  warps = launch.get("num_warps")
  _require(
    _is_integer(warps) and warps in WARP_COUNTS,
    INVALID_CASE,
    f"launch.num_warps must be one of {list(WARP_COUNTS)}",
  )
  _require(
    _is_integer(launch.get("num_stages"), 1),
    INVALID_CASE,
    "launch.num_stages must be a positive integer",
  )
  _require(
    type(launch.get("enable_fp_fusion")) is bool,
    INVALID_CASE,
    "launch.enable_fp_fusion must be a boolean",
  )


def _validate_execution(execution, *, launch):
  """Check the staging and probe fields the worker reads from execution."""
  _object(execution, "execution")
  envelope = _object(execution.get("guard_envelope"), "execution.guard_envelope")
  _require(
    _is_integer(envelope.get("prefix")),
    INVALID_CASE,
    "execution.guard_envelope.prefix must be a nonnegative integer",
  )
  _require(
    _is_integer(envelope.get("suffix")),
    INVALID_CASE,
    "execution.guard_envelope.suffix must be a nonnegative integer",
  )
  for field in (
    "gap_sentinel_by_dtype",
    "output_canary_by_dtype",
    "padding_by_dtype",
  ):
    sentinels = _object(execution.get(field), f"execution.{field}")
    for dtype in DTYPES:
      value = sentinels.get(dtype)
      _require(
        isinstance(value, str) and _HEX_8.fullmatch(value) is not None,
        INVALID_CASE,
        f"execution.{field}.{dtype} must be 8 hex digits",
      )
  probe = _object(execution.get("warp_probe"), "execution.warp_probe")
  _require(
    probe.get("status") == "not_probed",
    INVALID_CASE,
    "execution.warp_probe.status must be not_probed",
  )
  _require(
    type(probe.get("num_warps")) is int and probe["num_warps"] == launch["num_warps"],
    INVALID_CASE,
    "execution.warp_probe.num_warps must match launch.num_warps",
  )


def _check_frontend(profile, *, root, symiri):
  """Check program.sir and return the signature the frontend describes.

  One `--tensor-describe` call both validates the source and states the
  parameter and return types, so the manifest can be compared with what the
  source actually declares rather than with a record that happens to be
  internally consistent.
  """
  executable = _frontend_path(symiri).resolve()
  _require(
    executable.is_file(),
    FRONTEND_UNAVAILABLE,
    f"native frontend not found: {executable}",
  )
  try:
    completed = subprocess.run(
      [
        str(executable),
        "--entry",
        ENTRY,
        "--tensor-describe",
        "--tensor-profile-version",
        PROFILES[profile],
        PROGRAM_FILE,
      ],
      cwd=root,
      capture_output=True,
      text=True,
      errors="replace",
      timeout=60,
      check=False,
    )
  except (OSError, subprocess.SubprocessError) as error:
    raise TargetCaseError(
      FRONTEND_UNAVAILABLE, f"cannot run the native frontend: {error}"
    ) from error
  if completed.returncode != 0:
    diagnostic = completed.stderr.strip() or completed.stdout.strip()
    if not diagnostic:
      diagnostic = f"exit status {completed.returncode}"
    raise TargetCaseError(
      INVALID_SOURCE, f"the native frontend rejected program.sir: {diagnostic}"
    )
  try:
    signature = json.loads(completed.stdout)
  except json.JSONDecodeError as error:
    raise TargetCaseError(
      INVALID_SOURCE, f"the native frontend did not describe program.sir: {error}"
    ) from error
  if (
    not isinstance(signature, dict)
    or not isinstance(signature.get("inputs"), list)
    or not isinstance(signature.get("output"), dict)
  ):
    raise TargetCaseError(
      INVALID_SOURCE, "the native frontend described no input and output signature"
    )
  return signature


def _validate_signature(manifest, signature):
  """Require the manifest to state the signature the source declares."""
  _require(
    signature.get("entry") == manifest["entry"],
    SIGNATURE_MISMATCH,
    f"source entry {signature.get('entry')!r} differs from the manifest entry "
    f"{manifest['entry']!r}",
  )
  described = signature["inputs"]
  records = manifest["inputs"]
  _require(
    len(described) == len(records),
    SIGNATURE_MISMATCH,
    f"source declares {len(described)} parameters, the manifest states {len(records)} inputs",
  )
  for index, (parameter, record) in enumerate(zip(described, records)):
    _object(parameter, f"described parameter {index}")
    _require(
      _plain_name(parameter.get("name")) == _plain_name(record["name"]),
      SIGNATURE_MISMATCH,
      f"parameter {index} is named {parameter.get('name')!r} in the source and "
      f"{record['name']!r} in the manifest",
    )
    _require(
      parameter.get("dtype") == record["dtype"],
      SIGNATURE_MISMATCH,
      f"parameter {index} dtype {parameter.get('dtype')!r} differs from the manifest "
      f"{record['dtype']!r}",
    )
    _require(
      list(parameter.get("shape") or []) == list(record["shape"]),
      SIGNATURE_MISMATCH,
      f"parameter {index} shape {parameter.get('shape')!r} differs from the manifest "
      f"{record['shape']!r}",
    )
  output = signature["output"]
  _object(output, "described output")
  _require(
    output.get("name") == "return",
    SIGNATURE_MISMATCH,
    f"source return is named {output.get('name')!r}",
  )
  _require(
    output.get("dtype") == manifest["output"]["dtype"],
    SIGNATURE_MISMATCH,
    f"return dtype {output.get('dtype')!r} differs from the manifest "
    f"{manifest['output']['dtype']!r}",
  )
  _require(
    list(output.get("shape") or []) == list(manifest["output"]["shape"]),
    SIGNATURE_MISMATCH,
    f"return shape {output.get('shape')!r} differs from the manifest "
    f"{manifest['output']['shape']!r}",
  )


def _plain_name(name):
  """A local name without its sigil, as the described signature spells it."""
  if not isinstance(name, str):
    return name
  return name[1:] if name.startswith(("%", "@", "^")) else name


def _frontend_path(symiri):
  """Resolve the frontend from the argument, REFRACTIR_SYMIRI, or ./symiri."""
  if symiri is not None:
    return Path(symiri)
  configured = os.environ.get("REFRACTIR_SYMIRI")
  if configured:
    return Path(configured)
  return Path("./symiri")


def _layout_record(spec, fields, where):
  """Copy the frozen layout fields out of one writer record."""
  _require(isinstance(spec, dict), INVALID_CASE, f"{where} must be an object")
  missing = [name for name in fields if name not in spec]
  _require(not missing, INVALID_CASE, f"{where} is missing {', '.join(missing)}")
  record = {}
  for name in fields:
    value = spec[name]
    if name in {"shape", "element_strides", "axis_order"} and isinstance(
      value, (list, tuple)
    ):
      value = list(value)
    record[name] = value
  return record


def _object(value, where):
  _require(isinstance(value, dict), INVALID_CASE, f"{where} must be an object")
  return value


def _is_digest(value):
  return isinstance(value, str) and _HEX_64.fullmatch(value) is not None


def _is_integer(value, minimum=0):
  return type(value) is int and value >= minimum


def _sha256(data):
  return hashlib.sha256(data).hexdigest()


def _require(condition, status, reason):
  if not condition:
    raise TargetCaseError(status, reason)
