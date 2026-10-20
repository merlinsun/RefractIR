"""One recorded emission shared by the interpreter and compiled execution paths.

P1 and P2 must consume the same kernel bytes, layout, and launch settings, so an
emission is rendered once and recorded with the digests that bind it to its SIR
case. A consumer either reuses exactly that emission or refuses it; the emitter
is never called a second time for one record.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from test.lib.tensor_gluon_lowering import GeneratedGluonKernel

SCHEMA = "refractir.tensor-emission.v1"
EMITTER_MODULE = "test.lib.tensor_gluon_lowering"
EMITTER_SOURCE = Path(__file__).with_name("tensor_gluon_lowering.py")
_DIGEST = re.compile(r"[0-9a-f]{64}")


class EmissionRefused(ValueError):
  """The record does not describe the emission a consumer asked for."""


@dataclass(frozen=True)
class EmitterIdentity:
  """The emitter source and the render options that changed the kernel.

  `options` holds the render keyword arguments whose values reach the emitted
  text. An option that leaves the kernel unchanged leaves this identity
  unchanged, so the identity names the code that chose the bytes rather than the
  caller that asked for them.
  """

  module: str
  source_sha256: str
  options: tuple[tuple[str, Any], ...]

  @property
  def digest(self):
    """The source digest combined with the options that reached the kernel."""
    payload = _canonical(dict(self.options))
    material = self.source_sha256.encode("utf-8") + b"\n" + payload.encode("utf-8")
    return hashlib.sha256(material).hexdigest()


@dataclass(frozen=True)
class EmissionRecord:
  """One rendered kernel bound to the SIR, layout, and launch it belongs to."""

  sir_sha256: str
  kernel_sha256: str
  layout_sha256: str
  emitter: EmitterIdentity
  profile: str
  entry: str
  launch: tuple[tuple[str, Any], ...]
  kernel: GeneratedGluonKernel

  def as_record(self):
    """The JSON-able statement of this record, without the kernel text."""
    return {
      "schema": SCHEMA,
      "sir_sha256": self.sir_sha256,
      "kernel_sha256": self.kernel_sha256,
      "kernel_bytes": len(self.kernel.source.encode("utf-8")),
      "layout_sha256": self.layout_sha256,
      "emitter": {
        "module": self.emitter.module,
        "source_sha256": self.emitter.source_sha256,
        "options": dict(self.emitter.options),
        "digest": self.emitter.digest,
      },
      "profile": self.profile,
      "entry": self.entry,
      "launch": dict(self.launch),
    }


def _canonical(value):
  """One JSON statement of a record payload, independent of key insertion order."""
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest_text(text):
  return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest_value(value):
  return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _pairs(mapping, *, what):
  """One mapping as sorted pairs, refused when it has no canonical JSON form."""
  if not isinstance(mapping, dict) or any(not isinstance(key, str) for key in mapping):
    raise EmissionRefused(f"{what} must be a mapping with string keys")
  try:
    _canonical(mapping)
  except (TypeError, ValueError) as error:
    raise EmissionRefused(f"{what} is not canonical JSON: {error}") from error
  return tuple(sorted(mapping.items()))


def _emitter_identity(options):
  """The current emitter source, hashed with the options that reached the kernel."""
  if not EMITTER_SOURCE.is_file():
    raise EmissionRefused(f"emitter source is missing: {EMITTER_SOURCE}")
  return EmitterIdentity(
    module=EMITTER_MODULE,
    source_sha256=hashlib.sha256(EMITTER_SOURCE.read_bytes()).hexdigest(),
    options=_pairs(options, what="emitter options"),
  )


def _kernel_source(kernel):
  """The rendered text of an emission, refused when there is none.

  The record accepts the emitter's own kernel object rather than requiring one
  class identity, so a test harness that reloads the emitter module still reads
  the bytes it rendered.
  """
  source = getattr(kernel, "source", None)
  if not isinstance(source, str) or not source:
    raise EmissionRefused("the emitter did not return a generated Gluon kernel")
  return source


def build(*, sir, render, layout, profile, entry, launch, options):
  """Render one kernel and bind it to the case that asked for it.

  `render` is called exactly once and its result is stored in the record, so a
  later consumer reads the kernel instead of emitting it again. The digests
  cover the SIR text, the rendered kernel text, and the layout payload. The
  kernel digest is recorded explicitly rather than derived from either of the
  other two.
  """
  if not isinstance(sir, str) or not sir:
    raise EmissionRefused("the SIR source must be nonempty text")
  if not isinstance(profile, str) or not profile:
    raise EmissionRefused("the profile must be a nonempty string")
  if not isinstance(entry, str) or not entry:
    raise EmissionRefused("the entry must be a nonempty string")
  kernel = render()
  kernel_source = _kernel_source(kernel)
  return EmissionRecord(
    sir_sha256=_digest_text(sir),
    kernel_sha256=_digest_text(kernel_source),
    layout_sha256=_digest_value(layout),
    emitter=_emitter_identity(options),
    profile=profile,
    entry=entry,
    launch=_pairs(launch, what="launch settings"),
    kernel=kernel,
  )


def _require_record(record):
  """Refuse a missing or malformed record before any digest is compared."""
  if record is None:
    raise EmissionRefused("emission record is missing")
  if not isinstance(record, EmissionRecord):
    raise EmissionRefused(
      "emission record is garbled: expected an emission record, got "
      f"{type(record).__name__}"
    )
  for name in ("sir_sha256", "kernel_sha256", "layout_sha256"):
    value = getattr(record, name)
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
      raise EmissionRefused(
        f"emission record is garbled: {name} is not a sha256 digest"
      )
  if not isinstance(record.emitter, EmitterIdentity):
    raise EmissionRefused("emission record is garbled: emitter identity is missing")
  if _DIGEST.fullmatch(record.emitter.source_sha256 or "") is None:
    raise EmissionRefused(
      "emission record is garbled: emitter source digest is missing"
    )
  if not isinstance(getattr(record.kernel, "source", None), str):
    raise EmissionRefused("emission record is garbled: kernel is missing")
  if not isinstance(record.profile, str) or not record.profile:
    raise EmissionRefused("emission record is garbled: profile is missing")
  if not isinstance(record.entry, str) or not record.entry:
    raise EmissionRefused("emission record is garbled: entry is missing")
  if not isinstance(record.launch, tuple) or any(
    not isinstance(pair, tuple) or len(pair) != 2 for pair in record.launch
  ):
    raise EmissionRefused("emission record is garbled: launch settings are missing")


def verify(record, *, sir, layout, profile, entry, launch, options):
  """Return `record` once it matches the emission the caller is about to use.

  A record from another SIR case, a kernel whose bytes no longer hash to the
  recorded digest, a layout digest for another layout, and an emitter source or
  option set that no longer matches are each refused with the field that failed.
  """
  _require_record(record)
  if not isinstance(sir, str) or not sir:
    raise EmissionRefused("the SIR source must be nonempty text")
  actual_sir = _digest_text(sir)
  if record.sir_sha256 != actual_sir:
    raise EmissionRefused(
      "emission record belongs to another SIR case: record "
      f"{record.sir_sha256[:12]} does not hash the supplied SIR {actual_sir[:12]}"
    )
  actual_kernel = _digest_text(record.kernel.source)
  if record.kernel_sha256 != actual_kernel:
    raise EmissionRefused(
      "emission record kernel was tampered with: record "
      f"{record.kernel_sha256[:12]} does not hash the recorded kernel "
      f"{actual_kernel[:12]}"
    )
  actual_layout = _digest_value(layout)
  if record.layout_sha256 != actual_layout:
    raise EmissionRefused(
      "emission record layout digest does not match the layout in use: record "
      f"{record.layout_sha256[:12]} does not hash the supplied layout "
      f"{actual_layout[:12]}"
    )
  actual_emitter = _emitter_identity(options)
  if record.emitter != actual_emitter:
    raise EmissionRefused(
      "emission record emitter identity does not match the emitter in use: "
      f"record {record.emitter.digest[:12]} != {actual_emitter.digest[:12]}"
    )
  if record.profile != profile:
    raise EmissionRefused(
      f"emission record profile differs: {record.profile!r} != {profile!r}"
    )
  if record.entry != entry:
    raise EmissionRefused(
      f"emission record entry differs: {record.entry!r} != {entry!r}"
    )
  actual_launch = _pairs(launch, what="launch settings")
  if record.launch != actual_launch:
    raise EmissionRefused(
      "emission record launch settings differ from the launch in use"
    )
  return record


def reuse(record, *, sir, layout, profile, entry, launch, options):
  """The recorded kernel, verified against the case; no second render happens."""
  return verify(
    record,
    sir=sir,
    layout=layout,
    profile=profile,
    entry=entry,
    launch=launch,
    options=options,
  ).kernel
