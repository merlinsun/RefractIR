"""Export a validated unrolled companion of a counted loop-region case.

A counted region case describes a bounded loop. Its companion describes the same
program mechanically unrolled: one trip whose body holds every original body
copy. Everything that depends on the program text is derived again -- the
emitted Triton kernel, the loop and shape metadata, and the reference output --
so a companion is a case in its own right instead of a counted case with a
replaced source.

The companion's reference is a reference-interpreter replay of the unrolled
program over the counted case's inputs, checked at both supported reference
precisions and against the counted case's own expected bytes. It carries no
region certificate: the exact-region recogniser describes the counted loop, and
a mechanically unrolled body writes each temporary once per copy, which that
recogniser refuses by design. Instead of proving the unrolled body exact, the
manifest records how the reference was derived -- the counted case, its
certificate, the unrolled source digest, and the replays that had to agree -- and
the loader grants a bit-exact comparison only to a reference derived that way.

That derivation is a relationship, so the companion carries the counted artifacts
the loader checks it against: the counted program, its runtime input and
reference, and the certificate's own solver artifacts, each a digested file of
the companion's case. The loader compares the inherited input and reference bytes
with them, reproduces the unrolled source with
`test.lib.tensor_region_generation.unroll_source`, and validates the counted
certificate the way the counted path validates it. Zero tolerance alone is not
bit-exact: ``+0.0`` and ``-0.0`` differ in bits but compare equal, so the
comparison states signed-zero exactness as well, which compares every zero lane
by word.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

from test.lib import tensor_case, tensor_region_generation
from test.lib.build_tensor_exact_loop_case import (
  CaseBuildError,
  _digest,
  _run,
  _write_attempt,
)
from test.lib.run_triton_tests import _emit_kernel
from test.lib.tensor_coverage import source_signature

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYMIRI = Path(os.environ.get("REFRACTIR_SYMIRI", ROOT / "symiri")).resolve()
DEFAULT_SYMIRC = Path(os.environ.get("REFRACTIR_SYMIRC", ROOT / "symirc")).resolve()
# The reference interpreter evaluates f32 tensor atoms at one precision per run;
# both supported precisions must reproduce the counted reference bytes.
REFERENCE_PRECISIONS = (128, 256)
TIMEOUT_S = 60.0
UNROLLER = "test.lib.tensor_region_generation.unroll_source"
PROFILE_VERSION = {"tensor-v2": "v2"}
ARTIFACT_NAMES = (
  "program.sir",
  "inputs.bin",
  "expected.bin",
  "kernel.py",
  "tensor-layout.json",
  *(f"reference-{bits}.bin" for bits in REFERENCE_PRECISIONS),
)
_LOOP_HEADER = re.compile(r"(?m)^\s*br %i < (\d+), \^body, \^exit;\s*$")
_TENSOR_DECLARATION = re.compile(
  r"(?m)^\s*let mut %[A-Za-z_][A-Za-z_0-9]*\s*:\s*tensor\s+\[([^\]]+)\]\s+f32\s*="
)
_TENSOR_OPERATION = re.compile(
  r"(?m)^\s*%[A-Za-z_][A-Za-z_0-9]*\s*=\s*tensor\s+([A-Za-z_][A-Za-z_0-9]*)\b"
)


class CompanionError(ValueError):
  """A counted case this library refuses to turn into a companion."""

  def __init__(self, status: str, message: str):
    super().__init__(message)
    self.status = status


def _load(directory, symiri, status):
  """Load a case through the loader every target run uses, naming the side."""
  try:
    return tensor_case.load_case(directory, symiri=symiri)
  except tensor_case.CaseValidationError as error:
    raise CompanionError(status, str(error)) from error


def _loop_bound(source, where):
  """The bound of `source`'s canonical counted loop header."""
  match = _LOOP_HEADER.search(source)
  if match is None:
    raise CompanionError(
      "unsupported_region", f"{where} has no canonical counted loop header"
    )
  return int(match.group(1))


def _unrolled(source, iterations):
  """The mechanically unrolled source, or a refusal naming why there is none."""
  try:
    unrolled = tensor_region_generation.unroll_source(source, iterations)
  except ValueError as error:
    raise CompanionError("unsupported_region", str(error)) from error
  if unrolled == source:
    raise CompanionError(
      "unroll_is_identity",
      "the mechanically unrolled form of this source is the counted source "
      "itself, so there is no distinct form to compare",
    )
  return unrolled


def _profile_version(manifest):
  profile = manifest.get("profile")
  try:
    return PROFILE_VERSION[profile]
  except KeyError as error:
    raise CompanionError(
      "unsupported_profile",
      f"unrolling is defined for tensor-v2 counted regions, not {profile!r}",
    ) from error


def _canonical_digest(value):
  """Digest a manifest fragment the way the case writer serialises it."""
  canonical = json.dumps(
    value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
  )
  return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _counted_certificate(case):
  """The counted case's whole-output certificate, or a refusal naming why none.

  A companion earns its bit-exact comparison by deriving it from the counted
  case's certificate, so a counted case without one leaves nothing to derive
  from.
  """
  certificate = case.manifest.get("reference", {}).get("exactness")
  if case.manifest.get("comparison", {}).get("mode") != "exact" or not isinstance(
    certificate, dict
  ):
    raise CompanionError(
      "uncertified_counted_case",
      "a bit-exact companion is derived from a counted exactness certificate, "
      "and this counted case carries none",
    )
  return certificate


def _emit_companion_kernel(directory, manifest, symirc):
  """Emit the companion's kernel through the runner's own emission route.

  `run_triton_tests._emit_kernel` owns the argument set, so the case directory
  and every target run compile the same program the same way, layout records
  included.
  """
  staged = SimpleNamespace(path=directory, manifest=manifest)
  kernel, stderr, stdout = _emit_kernel(staged, symirc, directory)
  if kernel is None:
    raise CompanionError("compile_failure", (stderr or stdout).strip())
  return kernel


def _reference_replays(directory, case, version, symiri, deadline, ledger):
  """Replay the unrolled program at every supported reference precision."""
  records = []
  for bits in REFERENCE_PRECISIONS:
    output = directory / f"reference-{bits}.bin"
    command = [
      str(symiri),
      "--tensor-profile-version",
      version,
      "--tensor-reference-bits",
      str(bits),
      "--entry",
      case.manifest["entry"],
      "--tensor-input",
      str(directory / "inputs.bin"),
      "--tensor-output",
      str(output),
      str(directory / "program.sir"),
    ]
    try:
      _run("reference", command, deadline=deadline, ledger=ledger)
    except CaseBuildError as error:
      raise CompanionError(error.status, str(error)) from error
    records.append(
      {
        "reference_bits": bits,
        "file": output.name,
        "command": command,
        "data": output.read_bytes(),
      }
    )
  return records


def _tensor_shapes(source):
  shapes = set()
  for match in _TENSOR_DECLARATION.finditer(source):
    shapes.add(
      tuple(int(value) for value in match.group(1).replace(" ", "").split(","))
    )
  return [list(shape) for shape in sorted(shapes, key=lambda item: (len(item), item))]


def _tensor_operations(source):
  """The unrolled body's tensor operations, in emission order."""
  return _TENSOR_OPERATION.findall(source)


def _manifest_records(manifest, names):
  return {name: manifest[name] for name in names if name in manifest}


def _counted_files(certificate):
  """Counted artifact names the certificate's own records refer to."""
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

  walk(certificate)
  return names


def _carry_counted_files(counted_dir, directory, certificate):
  """Copy the counted artifacts a derivation depends on into the companion.

  A companion has to be checkable on its own, so the counted program, the
  certificate's solver artifacts and the counted runtime bytes are digested
  artifacts of the companion's own case. The counted program keeps its own name
  under ``counted/``; the certificate's selection artifacts keep theirs, so the
  certificate's records resolve against the carried copies unchanged.
  """
  counted_dir = Path(counted_dir)
  directory = Path(directory)
  names = {"program.sir", "inputs.bin", "expected.bin"} | _counted_files(certificate)
  carried = {}
  for name in sorted(names):
    source = counted_dir / name
    if not source.is_file():
      raise CompanionError(
        "missing_counted_artifact",
        f"the counted case does not carry {name}, which the derivation needs",
      )
    target = directory / "counted" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    carried[name] = str(target.relative_to(directory))
  return carried


def _companion_manifest(
  case,
  counted_dir,
  unrolled,
  counted_bound,
  files,
  replays,
  certificate,
  carried,
):
  counted = case.manifest
  counted_behavior = counted.get("coverage", {}).get("input_behavior", {})
  inherited_execution = _manifest_records(
    counted["execution"], ("guard_envelope", "padding_by_dtype")
  )
  bound = _loop_bound(unrolled, "the unrolled source")
  shapes = _tensor_shapes(unrolled)
  return {
    "schema": "refractir.tensor-case.v2",
    "profile": counted["profile"],
    "entry": counted["entry"],
    "source": {
      "file": "program.sir",
      "sha256": _digest(files["program.sir"])["sha256"],
      "generator": "unrolled-companion",
      "seed": counted.get("source", {}).get("seed", 0),
    },
    # Everything below states the derivation the bit-exact comparison rests on:
    # what the companion was derived from, the counted certificate itself, the
    # carried copies of the counted artifacts that make it checkable, and the
    # replays that had to agree with the counted reference for the derivation to
    # hold.
    "companion": {
      "of": {
        "entry": counted["entry"],
        "profile": counted["profile"],
        "program_sha256": hashlib.sha256(case.program).hexdigest(),
        "manifest_sha256": hashlib.sha256(
          (counted_dir / "case.json").read_bytes()
        ).hexdigest(),
        "certificate_sha256": _canonical_digest(certificate),
        "reference_sha256": hashlib.sha256(case.expected).hexdigest(),
        "certificate": certificate,
        # The counted case's own operation list: the certificate was written for
        # that program, and an unrolled source need not list the same operations
        # the same way, so the derivation carries the counted claim itself.
        "operator_coverage": counted.get("operator_coverage"),
        "files": carried,
      },
      "transformation": UNROLLER,
      "counted_trips": counted_bound,
      "unrolled_trips": bound,
      "unrolled_body_copies": counted_bound,
      "unrolled_source_sha256": _digest(files["program.sir"])["sha256"],
      "inherited_files": ["inputs.bin"],
      "reference": "interpreter-replay-unrolled",
      "replays": [
        {
          "reference_bits": record["reference_bits"],
          "file": record["file"],
          "sha256": _digest(record["data"])["sha256"],
          "command": record["command"],
        }
        for record in replays
      ],
    },
    # Runtime inputs are the counted case's own bytes; the records that describe
    # their staging are program-independent.
    "inputs": counted["inputs"],
    "output": counted["output"],
    "grid": counted["grid"],
    "softmax_lowering": counted["softmax_lowering"],
    "launch": counted["launch"],
    "property": counted["property"],
    "evidence": {
      "kind": "numerical_reference",
      "status": "numerical_reference_checked",
      "artifacts": sorted(files),
    },
    # A companion holds no exactness proof of its own, so it compares
    # numerically; the derivation above is what makes that comparison bit-exact,
    # which is why zero tolerance comes with signed-zero exactness.
    "comparison": {
      "mode": "numeric",
      "policy": "derived-counted-certificate-bit-exact",
      "atol": [0, 1],
      "rtol": [0, 1],
      "zero_sign_exact": True,
    },
    "reference": {
      "defined": True,
      "stable": True,
      "precisions": list(REFERENCE_PRECISIONS),
      "exactness": None,
      "source": "interpreter-replay-unrolled",
      "note": (
        "Expected bytes are a reference-interpreter replay of the unrolled "
        "program over the counted case's inputs, replayed at every supported "
        "reference precision and required to reproduce the counted "
        "certificate's reference bytes. The companion record names the counted "
        "case, its certificate, the unrolled source, and those replays; the "
        "comparison is bit-exact because zero lanes are compared by word."
      ),
    },
    "execution": {
      "full_shapes": shapes,
      "range_extents": shapes,
      "warp_probe": counted["execution"]["warp_probe"],
      "loop_bounds": [bound],
      "total_body_executions": bound,
      "gap_sentinel_by_dtype": counted["execution"]["gap_sentinel_by_dtype"],
      "output_canary_by_dtype": counted["execution"]["output_canary_by_dtype"],
      **inherited_execution,
    },
    "coverage": {
      "primary_stratum": "generated-bounded-region-unrolled",
      "source_signature": source_signature(unrolled),
      "loop_structure": [{"bound": bound, "counted_bound": counted_bound}],
      "input_behavior": {
        **_manifest_records(
          counted_behavior,
          (
            "property_passed",
            "pattern",
            "input_mode",
            "policy",
            "input_domain",
            "distinguishing_objective",
          ),
        ),
        "source": "counted-case-selection",
      },
    },
    "environment": None,
    "operator_coverage": _tensor_operations(unrolled),
    "shape_axes": counted.get("shape_axes", []),
    "files": {name: _digest(data) for name, data in files.items()},
  }


def build_companion(
  counted_case_dir: str | Path,
  destination: str | Path,
  *,
  symiri: str | Path = DEFAULT_SYMIRI,
  symirc: str | Path = DEFAULT_SYMIRC,
  iterations: int | None = None,
  timeout_s: float = TIMEOUT_S,
) -> Path:
  """Write a complete unrolled companion case of a counted region case."""
  counted_dir = Path(counted_case_dir).resolve()
  directory = Path(destination).resolve()
  case = _load(counted_dir, symiri, "invalid_counted_case")
  certificate = _counted_certificate(case)
  source = case.program.decode("utf-8")
  counted_bound = _loop_bound(source, "the counted source")
  if iterations is not None and iterations != counted_bound:
    raise CompanionError(
      "mismatched_iterations",
      f"unrolling must keep the counted {counted_bound} trips, not {iterations}",
    )
  unrolled = _unrolled(source, iterations)
  version = _profile_version(case.manifest)
  directory.mkdir(parents=True, exist_ok=False)
  ledger: list[dict] = []
  deadline = time.monotonic() + timeout_s
  try:
    (directory / "program.sir").write_text(unrolled, encoding="utf-8")
    (directory / "inputs.bin").write_bytes(case.inputs)
    _emit_companion_kernel(directory, case.manifest, symirc)
    replays = _reference_replays(directory, case, version, symiri, deadline, ledger)
    for record in replays:
      if record["data"] != case.expected:
        raise CompanionError(
          "replay_mismatch",
          f"the unrolled program replayed at {record['reference_bits']} reference "
          "bits does not reproduce the counted expected bytes",
        )
    (directory / "expected.bin").write_bytes(replays[0]["data"])
    carried = _carry_counted_files(counted_dir, directory, certificate)
    files = {
      name: (directory / Path(name)).read_bytes()
      for name in (*ARTIFACT_NAMES, *carried.values())
    }
    manifest = _companion_manifest(
      case,
      counted_dir,
      unrolled,
      counted_bound,
      files,
      replays,
      certificate,
      carried,
    )
    (directory / "case.json").write_text(
      json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
      encoding="utf-8",
    )
    # A companion is only useful if the loader every target run uses admits it.
    _load(directory, symiri, "invalid_companion")
    _write_attempt(directory, "accepted", ledger, None)
  except CompanionError as error:
    _write_attempt(directory, error.status, ledger, str(error))
    raise
  except BaseException as error:
    _write_attempt(directory, "failed", ledger, f"{type(error).__name__}: {error}")
    raise
  return directory


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--counted", required=True)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--symiri", default=str(DEFAULT_SYMIRI))
  parser.add_argument("--symirc", default=str(DEFAULT_SYMIRC))
  parser.add_argument("--timeout", type=float, default=TIMEOUT_S)
  args = parser.parse_args(argv)
  companion = build_companion(
    args.counted,
    args.output_dir,
    symiri=args.symiri,
    symirc=args.symirc,
    timeout_s=args.timeout,
  )
  print(json.dumps({"companion": str(companion), "status": "accepted"}))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
