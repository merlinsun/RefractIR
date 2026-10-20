"""One recorded emission shared by the interpreter and compiled execution paths.

P1 and P2 must consume the same kernel bytes, layout, and launch settings, so an
emission is rendered once and recorded with the digests that bind it to its SIR
case. A consumer either reuses exactly that emission or refuses it; the emitter
is never called a second time for one record.

The record names the emitter, not one module. A Gluon emission states the
lowering module and hashes its Python source; an ordinary emission states the
`symirc` executable and hashes its bytes. Both share the record, the option
canonicalization, and the verify/reuse refusal.
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
GLUON = "gluon"
SYMIRC = "symirc"
TARGET = "triton"
_DIGEST = re.compile(r"[0-9a-f]{64}")


class EmissionRefused(ValueError):
  """The record does not describe the emission a consumer asked for."""


@dataclass(frozen=True)
class EmittedKernel:
  """The text of a kernel whose emitter writes a file instead of returning one."""

  source: str


@dataclass(frozen=True)
class EmitterIdentity:
  """The code and settings that produced one kernel.

  `kind` names the emission route. A Gluon identity states its lowering module
  and the digest of that module's Python source. An ordinary identity states the
  `symirc` executable path and the digest of the executable bytes. `options`
  holds every setting whose value reaches the emitted text, so two calls that
  differ only in a setting that leaves the text unchanged share an identity.
  """

  kind: str
  options: tuple[tuple[str, Any], ...]
  module: str | None = None
  source_sha256: str | None = None
  executable: str | None = None
  binary_sha256: str | None = None

  @property
  def digest(self):
    """The kind, the emitter source or binary digest, and the canonical options."""
    material = {"kind": self.kind, "options": dict(self.options)}
    if self.kind == SYMIRC:
      material["binary_sha256"] = self.binary_sha256
    else:
      material["source_sha256"] = self.source_sha256
    return _digest_value(material)

  def as_record(self):
    """The JSON-able statement of this identity."""
    record = {"kind": self.kind}
    if self.kind == SYMIRC:
      record["executable"] = self.executable
      record["binary_sha256"] = self.binary_sha256
    else:
      record["module"] = self.module
      record["source_sha256"] = self.source_sha256
    record["options"] = dict(self.options)
    record["digest"] = self.digest
    return record


@dataclass(frozen=True)
class EmissionRecord:
  """One rendered kernel bound to the SIR, layout, and launch it belongs to.

  `layout_manifest_sha256` and `softmax_lowering` are stated only by an ordinary
  emission, where the layout manifest and the lowering are inputs to the
  emitted text. A Gluon record carries neither.
  """

  sir_sha256: str
  kernel_sha256: str
  layout_sha256: str
  emitter: EmitterIdentity
  profile: str
  entry: str
  launch: tuple[tuple[str, Any], ...]
  kernel: GeneratedGluonKernel | EmittedKernel
  target: str = TARGET
  layout_manifest_sha256: str | None = None
  softmax_lowering: str | None = None

  def as_record(self):
    """The JSON-able statement of this record, without the kernel text."""
    return {
      "schema": SCHEMA,
      "kind": self.emitter.kind,
      "sir_sha256": self.sir_sha256,
      "kernel_sha256": self.kernel_sha256,
      "kernel_bytes": len(self.kernel.source.encode("utf-8")),
      "layout_sha256": self.layout_sha256,
      "layout_manifest_sha256": self.layout_manifest_sha256,
      "emitter": self.emitter.as_record(),
      "profile": self.profile,
      "entry": self.entry,
      "target": self.target,
      "softmax_lowering": self.softmax_lowering,
      "launch": dict(self.launch),
    }


def _canonical(value):
  """One JSON statement of a record payload, independent of key insertion order."""
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest_text(text):
  return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest_value(value):
  return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _digest_bytes(data):
  return hashlib.sha256(data).hexdigest()


def _pairs(mapping, *, what):
  """One mapping as sorted pairs, refused when it has no canonical JSON form."""
  if not isinstance(mapping, dict) or any(not isinstance(key, str) for key in mapping):
    raise EmissionRefused(f"{what} must be a mapping with string keys")
  try:
    _canonical(mapping)
  except (TypeError, ValueError) as error:
    raise EmissionRefused(f"{what} is not canonical JSON: {error}") from error
  return tuple(sorted(mapping.items()))


def _executable_path(executable):
  """The stated executable, refused when it is absent or unreadable."""
  try:
    path = Path(executable)
  except TypeError as error:
    raise EmissionRefused("the symirc executable path must be stated") from error
  if not path.is_file():
    raise EmissionRefused(f"the symirc executable is missing: {path}")
  return path


def _emitter_identity(kind, options, executable=None):
  """The emitter in use, hashed with the options that reached the kernel."""
  pairs = _pairs(options, what="emitter options")
  if kind == GLUON:
    if executable is not None:
      raise EmissionRefused("a Gluon emission records no executable")
    if not EMITTER_SOURCE.is_file():
      raise EmissionRefused(f"emitter source is missing: {EMITTER_SOURCE}")
    return EmitterIdentity(
      kind=GLUON,
      module=EMITTER_MODULE,
      source_sha256=_digest_bytes(EMITTER_SOURCE.read_bytes()),
      options=pairs,
    )
  if kind == SYMIRC:
    path = _executable_path(executable)
    try:
      data = path.read_bytes()
    except OSError as error:
      raise EmissionRefused(
        f"the symirc executable cannot be read: {path}: {error}"
      ) from error
    return EmitterIdentity(
      kind=SYMIRC,
      executable=str(path),
      binary_sha256=_digest_bytes(data),
      options=pairs,
    )
  raise EmissionRefused(f"emitter kind must be {GLUON!r} or {SYMIRC!r}, got {kind!r}")


def _kernel_source(kernel):
  """The rendered text of an emission, refused when there is none.

  The record accepts the emitter's own kernel object rather than requiring one
  class identity, so a test harness that reloads the emitter module still reads
  the bytes it rendered. An ordinary emitter returns text, which `build_ordinary`
  wraps in `EmittedKernel`.
  """
  source = getattr(kernel, "source", None)
  if not isinstance(source, str) or not source:
    raise EmissionRefused("the emitter did not return kernel source text")
  return source


def _ordinary_kernel(kernel):
  """The kernel text from a string, a path, or an object that carries a source."""
  if isinstance(kernel, str):
    return EmittedKernel(source=kernel)
  if isinstance(kernel, Path):
    try:
      return EmittedKernel(source=kernel.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as error:
      raise EmissionRefused(f"the emitted kernel cannot be read: {error}") from error
  return kernel


def build(
  *,
  sir,
  render,
  layout,
  profile,
  entry,
  launch,
  options,
  kind=GLUON,
  executable=None,
  layout_manifest=None,
  softmax_lowering=None,
  target=TARGET,
):
  """Render one kernel and bind it to the case that asked for it.

  `render` is called exactly once and its result is stored in the record, so a
  later consumer reads the kernel instead of emitting it again. The digests
  cover the SIR text, the rendered kernel text, and the layout payload. The
  kernel digest is recorded explicitly rather than derived from either of the
  other two.

  `kind` selects the emitter identity and defaults to a Gluon emission. An
  ordinary emission also states `executable`, `layout_manifest`, and
  `softmax_lowering`; a Gluon emission states none of them.
  """
  if not isinstance(sir, str) or not sir:
    raise EmissionRefused("the SIR source must be nonempty text")
  if not isinstance(profile, str) or not profile:
    raise EmissionRefused("the profile must be a nonempty string")
  if not isinstance(entry, str) or not entry:
    raise EmissionRefused("the entry must be a nonempty string")
  if not isinstance(target, str) or not target:
    raise EmissionRefused("the target must be a nonempty string")
  if kind == GLUON:
    if layout_manifest is not None or softmax_lowering is not None:
      raise EmissionRefused(
        "a Gluon emission records no ordinary layout manifest or softmax lowering"
      )
  elif kind == SYMIRC:
    if layout_manifest is None:
      raise EmissionRefused("an ordinary emission must state its layout manifest")
    if not isinstance(softmax_lowering, str) or not softmax_lowering:
      raise EmissionRefused("an ordinary emission must state its softmax lowering")
  else:
    raise EmissionRefused(f"emitter kind must be {GLUON!r} or {SYMIRC!r}, got {kind!r}")
  manifest_sha256 = None if layout_manifest is None else _digest_value(layout_manifest)
  kernel = render()
  if isinstance(kernel, str):
    kernel = EmittedKernel(source=kernel)
  kernel_source = _kernel_source(kernel)
  return EmissionRecord(
    sir_sha256=_digest_text(sir),
    kernel_sha256=_digest_text(kernel_source),
    layout_sha256=_digest_value(layout),
    emitter=_emitter_identity(kind, options, executable),
    profile=profile,
    entry=entry,
    launch=_pairs(launch, what="launch settings"),
    kernel=kernel,
    target=target,
    layout_manifest_sha256=manifest_sha256,
    softmax_lowering=softmax_lowering,
  )


def build_ordinary(
  *,
  sir,
  executable,
  layout,
  layout_manifest,
  profile,
  entry,
  launch,
  softmax_lowering,
  options,
  kernel=None,
  render=None,
  target=TARGET,
):
  """Record one ordinary `symirc` emission and the executable that produced it.

  Exactly one of `kernel` and `render` is stated. `kernel` is the emitted Triton
  source text, a path to the emitted file, or an object with a `source`
  attribute; `render` is a zero-argument callable returning the same. The
  identity hashes the executable bytes and the options that reached the text,
  so a changed option or a changed binary refuses the record.
  """
  if (kernel is None) == (render is None):
    raise EmissionRefused("an ordinary emission states exactly one of kernel or render")
  if render is None:

    def render():
      return _ordinary_kernel(kernel)

  return build(
    kind=SYMIRC,
    sir=sir,
    render=render,
    layout=layout,
    layout_manifest=layout_manifest,
    profile=profile,
    entry=entry,
    launch=launch,
    softmax_lowering=softmax_lowering,
    options=options,
    executable=executable,
    target=target,
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
  if record.emitter.kind == GLUON:
    if _DIGEST.fullmatch(record.emitter.source_sha256 or "") is None:
      raise EmissionRefused(
        "emission record is garbled: emitter source digest is missing"
      )
    if not isinstance(record.emitter.module, str) or not record.emitter.module:
      raise EmissionRefused("emission record is garbled: emitter module is missing")
  elif record.emitter.kind == SYMIRC:
    if _DIGEST.fullmatch(record.emitter.binary_sha256 or "") is None:
      raise EmissionRefused(
        "emission record is garbled: emitter binary digest is missing"
      )
    if not isinstance(record.emitter.executable, str) or not record.emitter.executable:
      raise EmissionRefused("emission record is garbled: emitter executable is missing")
    if _DIGEST.fullmatch(record.layout_manifest_sha256 or "") is None:
      raise EmissionRefused(
        "emission record is garbled: layout manifest digest is missing"
      )
    if not isinstance(record.softmax_lowering, str) or not record.softmax_lowering:
      raise EmissionRefused("emission record is garbled: softmax lowering is missing")
  else:
    raise EmissionRefused("emission record is garbled: emitter kind is missing")
  if not isinstance(getattr(record.kernel, "source", None), str):
    raise EmissionRefused("emission record is garbled: kernel is missing")
  if not isinstance(record.profile, str) or not record.profile:
    raise EmissionRefused("emission record is garbled: profile is missing")
  if not isinstance(record.entry, str) or not record.entry:
    raise EmissionRefused("emission record is garbled: entry is missing")
  if not isinstance(record.target, str) or not record.target:
    raise EmissionRefused("emission record is garbled: target is missing")
  if not isinstance(record.launch, tuple) or any(
    not isinstance(pair, tuple) or len(pair) != 2 for pair in record.launch
  ):
    raise EmissionRefused("emission record is garbled: launch settings are missing")


def _option_difference(recorded, actual):
  """The option names whose presence or value differs between two pair lists."""
  recorded_map = dict(recorded)
  actual_map = dict(actual)
  return sorted(
    key
    for key in set(recorded_map) | set(actual_map)
    if (key in recorded_map, recorded_map.get(key))
    != (
      key in actual_map,
      actual_map.get(key),
    )
  )


def _compare_emitter(recorded, actual):
  """Refuse an emitter whose kind, digest, or options differ from the record."""
  if recorded.kind != actual.kind:
    raise EmissionRefused(
      "emission record emitter kind does not match the emitter in use: "
      f"record {recorded.kind!r} != {actual.kind!r}"
    )
  if recorded.source_sha256 != actual.source_sha256:
    raise EmissionRefused(
      "emission record emitter source digest does not match the emitter in use: "
      f"record {recorded.source_sha256[:12]} != {actual.source_sha256[:12]}"
    )
  if recorded.binary_sha256 != actual.binary_sha256:
    raise EmissionRefused(
      "emission record emitter executable digest does not match the executable "
      f"in use: record {recorded.binary_sha256[:12]} != "
      f"{actual.binary_sha256[:12]}"
    )
  changed = _option_difference(recorded.options, actual.options)
  if changed:
    name = changed[0]
    raise EmissionRefused(
      f"emission record emitter option {name!r} differs from the option in use: "
      f"record {dict(recorded.options).get(name)!r} != "
      f"{dict(actual.options).get(name)!r}"
    )


def _requested_kind(executable, layout_manifest, softmax_lowering):
  """The emission route the caller's arguments describe."""
  if any(
    value is not None for value in (executable, layout_manifest, softmax_lowering)
  ):
    return SYMIRC
  return GLUON


def verify(
  record,
  *,
  sir,
  layout,
  profile,
  entry,
  launch,
  options,
  executable=None,
  layout_manifest=None,
  softmax_lowering=None,
  target=None,
):
  """Return `record` once it matches the emission the caller is about to use.

  A record from another SIR case, a kernel whose bytes no longer hash to the
  recorded digest, a layout digest for another layout, a layout manifest for
  another manifest, another `symirc` binary, a changed emitter option, and an
  emitter of the other kind are each refused with the field that failed. The
  arguments that an ordinary emission needs also state that the caller is on the
  ordinary route; omitting all of them states the Gluon route.
  """
  _require_record(record)
  requested = _requested_kind(executable, layout_manifest, softmax_lowering)
  if record.emitter.kind != requested:
    raise EmissionRefused(
      "emission record emitter kind does not match the emitter in use: "
      f"record {record.emitter.kind!r} != {requested!r}"
    )
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
  if requested == SYMIRC:
    if executable is None:
      raise EmissionRefused("an ordinary emission must state its executable")
    if layout_manifest is None:
      raise EmissionRefused("an ordinary emission must state its layout manifest")
    if softmax_lowering is None:
      raise EmissionRefused("an ordinary emission must state its softmax lowering")
    actual_manifest = _digest_value(layout_manifest)
    if record.layout_manifest_sha256 != actual_manifest:
      raise EmissionRefused(
        "emission record layout manifest digest does not match the manifest in "
        f"use: record {record.layout_manifest_sha256[:12]} does not hash the "
        f"supplied manifest {actual_manifest[:12]}"
      )
    if record.softmax_lowering != softmax_lowering:
      raise EmissionRefused(
        "emission record softmax lowering differs: "
        f"{record.softmax_lowering!r} != {softmax_lowering!r}"
      )
    _compare_emitter(record.emitter, _emitter_identity(SYMIRC, options, executable))
  else:
    _compare_emitter(record.emitter, _emitter_identity(GLUON, options))
  expected_target = TARGET if target is None else target
  if record.target != expected_target:
    raise EmissionRefused(
      f"emission record target differs: {record.target!r} != {expected_target!r}"
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


def reuse(
  record,
  *,
  sir,
  layout,
  profile,
  entry,
  launch,
  options,
  executable=None,
  layout_manifest=None,
  softmax_lowering=None,
  target=None,
):
  """The recorded kernel, verified against the case; no second render happens."""
  return verify(
    record,
    sir=sir,
    layout=layout,
    profile=profile,
    entry=entry,
    launch=launch,
    options=options,
    executable=executable,
    layout_manifest=layout_manifest,
    softmax_lowering=softmax_lowering,
    target=target,
  ).kernel
