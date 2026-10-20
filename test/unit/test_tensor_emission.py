"""One recorded emission, Gluon or ordinary, is what both execution paths consume."""

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from test.lib.build_tensor_exact_region_case import physical_layout
from test.lib.tensor_gluon_lowering import render_gluon_kernel
from test.unit.test_tensor_generated_region_runner import (
  NATIVE_SEED_7_SOURCE,
  ORTHOGONAL_REGION_SOURCE,
  _run_case_directory_runner,
  _run_generated_region_program,
)

PROFILE = "tensor-v2"
ENTRY = "kernel"
LAUNCH = {"grid": (1,), "num_warps": 1, "enable_fp_fusion": False}
LAYOUT = {"shape": [2, 2], **physical_layout((2, 2))}
OTHER_LAYOUT = {"shape": [2, 2], **physical_layout((2, 2), row_padding=1)}
STUB_BYTES = b"#!/bin/sh\nexit 0\n"
ORDINARY_KERNEL = "def kernel(*args, **kwargs):\n  return args\n"
LAYOUT_MANIFEST = {
  "buffers": [
    {
      "name": "x",
      "role": "input",
      "element_strides": [2, 1],
      "element_offset": 0,
      "storage_span": 4,
      "axis_order": [1, 0],
    }
  ]
}


def _emission():
  """The emission-record module under test."""
  from test.lib import tensor_emission

  return tensor_emission


def _digest(text):
  return hashlib.sha256(text.encode()).hexdigest()


def _record(emission, *, sir=NATIVE_SEED_7_SOURCE, layout=LAYOUT):
  return emission.build(
    sir=sir,
    render=lambda: render_gluon_kernel(sir, tile=None),
    layout=layout,
    profile=PROFILE,
    entry=ENTRY,
    launch=LAUNCH,
    options={"tile": None},
  )


class SharedEmissionTests(unittest.TestCase):
  """A record binds one rendered kernel to the case that asked for it."""

  def test_one_record_serves_both_paths_with_identical_digests(self):
    emission = _emission()
    shared = _record(emission)
    expected = (
      shared.sir_sha256,
      shared.kernel_sha256,
      shared.layout_sha256,
      shared.kernel.source,
    )

    def consume(path):
      # The two paths read one record; neither path renders a kernel of its own.
      self.assertIn(path, {"P1", "P2"})
      kernel = emission.reuse(
        shared,
        sir=NATIVE_SEED_7_SOURCE,
        layout=LAYOUT,
        profile=PROFILE,
        entry=ENTRY,
        launch=LAUNCH,
        options={"tile": None},
      )
      return (
        shared.sir_sha256,
        shared.kernel_sha256,
        shared.layout_sha256,
        kernel.source,
      )

    self.assertEqual(consume("P1"), expected)
    self.assertEqual(consume("P2"), expected)
    self.assertEqual(shared.sir_sha256, _digest(NATIVE_SEED_7_SOURCE))
    self.assertEqual(shared.kernel_sha256, _digest(shared.kernel.source))
    self.assertEqual(len(shared.layout_sha256), 64)

  def test_a_record_from_another_sir_case_is_refused(self):
    emission = _emission()
    shared = _record(emission)

    with self.assertRaisesRegex(emission.EmissionRefused, "SIR"):
      emission.reuse(
        shared,
        sir=ORTHOGONAL_REGION_SOURCE,
        layout=LAYOUT,
        profile=PROFILE,
        entry=ENTRY,
        launch=LAUNCH,
        options={"tile": None},
      )

  def test_a_tampered_kernel_is_refused(self):
    emission = _emission()
    shared = _record(emission)
    tampered_kernel = replace(
      shared.kernel, source=shared.kernel.source + "# tampered\n"
    )
    tampered = replace(shared, kernel=tampered_kernel)

    with self.assertRaisesRegex(emission.EmissionRefused, "kernel"):
      emission.reuse(
        tampered,
        sir=NATIVE_SEED_7_SOURCE,
        layout=LAYOUT,
        profile=PROFILE,
        entry=ENTRY,
        launch=LAUNCH,
        options={"tile": None},
      )

  def test_a_valid_record_is_reused_without_rerendering(self):
    emission = _emission()
    render = mock.Mock(wraps=render_gluon_kernel)
    shared = emission.build(
      sir=NATIVE_SEED_7_SOURCE,
      render=lambda: render(NATIVE_SEED_7_SOURCE, tile=None),
      layout=LAYOUT,
      profile=PROFILE,
      entry=ENTRY,
      launch=LAUNCH,
      options={"tile": None},
    )
    self.assertEqual(render.call_count, 1)

    for _ in range(2):
      emission.reuse(
        shared,
        sir=NATIVE_SEED_7_SOURCE,
        layout=LAYOUT,
        profile=PROFILE,
        entry=ENTRY,
        launch=LAUNCH,
        options={"tile": None},
      )
    self.assertEqual(render.call_count, 1)

    # The runner consumes the record instead of its own emitter as well.
    first = _run_generated_region_program(layout_count=1)
    record = first["result"].get("_emission_record")
    self.assertIsNotNone(record)
    self.assertEqual(first["render"].call_count, 1)
    second = _run_generated_region_program(layout_count=1, record=record)
    self.assertEqual(second["render"].call_count, 0)
    self.assertEqual(second["kernels"], first["kernels"])

  def test_a_missing_or_garbled_record_is_refused(self):
    emission = _emission()
    expected = {
      "sir": NATIVE_SEED_7_SOURCE,
      "layout": LAYOUT,
      "profile": PROFILE,
      "entry": ENTRY,
      "launch": LAUNCH,
      "options": {"tile": None},
    }

    with self.assertRaisesRegex(emission.EmissionRefused, "missing"):
      emission.verify(None, **expected)
    with self.assertRaisesRegex(emission.EmissionRefused, "garbled"):
      emission.verify("not a record", **expected)
    with self.assertRaisesRegex(emission.EmissionRefused, "garbled"):
      emission.verify(replace(_record(emission), kernel=None), **expected)

  def test_the_case_route_reuses_a_record_without_rerendering(self):
    first = _run_case_directory_runner(source=NATIVE_SEED_7_SOURCE, record=None)
    self.assertIsNone(first["refusal"])
    record = first["report"]["_emission_record"]
    self.assertEqual(first["render"].call_count, 1)

    second = _run_case_directory_runner(
      source=NATIVE_SEED_7_SOURCE, record=None, emission_record=record
    )
    self.assertIsNone(second["refusal"])
    self.assertEqual(second["render"].call_count, 0)
    self.assertEqual(
      second["report"]["emission"]["record"], first["report"]["emission"]["record"]
    )

  def test_two_layouts_change_only_the_layout_digest(self):
    emission = _emission()
    first = emission.build(
      sir=NATIVE_SEED_7_SOURCE,
      render=lambda: render_gluon_kernel(NATIVE_SEED_7_SOURCE, layout=LAYOUT),
      layout=LAYOUT,
      profile=PROFILE,
      entry=ENTRY,
      launch=LAUNCH,
      options={"layout": LAYOUT},
    )
    second = emission.build(
      sir=NATIVE_SEED_7_SOURCE,
      render=lambda: render_gluon_kernel(NATIVE_SEED_7_SOURCE, layout=OTHER_LAYOUT),
      layout=OTHER_LAYOUT,
      profile=PROFILE,
      entry=ENTRY,
      launch=LAUNCH,
      options={"layout": OTHER_LAYOUT},
    )

    self.assertEqual(first.sir_sha256, second.sir_sha256)
    self.assertNotEqual(first.layout_sha256, second.layout_sha256)
    self.assertNotEqual(first.kernel_sha256, second.kernel_sha256)
    with self.assertRaisesRegex(emission.EmissionRefused, "layout"):
      emission.reuse(
        first,
        sir=NATIVE_SEED_7_SOURCE,
        layout=OTHER_LAYOUT,
        profile=PROFILE,
        entry=ENTRY,
        launch=LAUNCH,
        options={"layout": LAYOUT},
      )

  def test_the_no_record_call_shape_is_unchanged(self):
    observation = _run_generated_region_program(layout_count=1)
    emitted = render_gluon_kernel(NATIVE_SEED_7_SOURCE, tile=None)

    self.assertEqual(observation["render"].call_count, 1)
    self.assertEqual(
      observation["render"].call_args, mock.call(NATIVE_SEED_7_SOURCE, tile=None)
    )
    self.assertEqual(
      observation["result"]["emission"]["sha256"], _digest(emitted.source)
    )
    self.assertEqual(observation["kernels"]["generated-region-kernel"], emitted.source)
    self.assertNotIn("_emission_record", observation["report"])


def _digest_bytes(data):
  return hashlib.sha256(data).hexdigest()


def _value_digest(value):
  payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
  return _digest(payload)


def _stub(directory, *, name="symirc", body=STUB_BYTES):
  path = Path(directory) / name
  path.write_bytes(body)
  return path


def _gluon_identity_arguments():
  return {
    "sir": NATIVE_SEED_7_SOURCE,
    "layout": LAYOUT,
    "profile": PROFILE,
    "entry": ENTRY,
    "launch": LAUNCH,
    "options": {"tile": None},
  }


def _ordinary_identity_arguments(*, options=None, **overrides):
  arguments = {
    "sir": NATIVE_SEED_7_SOURCE,
    "layout": LAYOUT,
    "layout_manifest": LAYOUT_MANIFEST,
    "profile": PROFILE,
    "entry": ENTRY,
    "launch": LAUNCH,
    "softmax_lowering": "direct",
    "options": {"tensor_profile_version": "v2"} if options is None else options,
  }
  arguments.update(overrides)
  return arguments


def _ordinary_record(executable, *, kernel=ORDINARY_KERNEL, **overrides):
  return _emission().build_ordinary(
    executable=executable,
    kernel=kernel,
    **_ordinary_identity_arguments(**overrides),
  )


class OrdinaryEmissionTests(unittest.TestCase):
  """An ordinary symirc emission is identified by its binary and options."""

  def test_an_ordinary_record_states_its_emission_identity(self):
    emission = _emission()
    with tempfile.TemporaryDirectory() as directory:
      executable = _stub(directory)
      record = _ordinary_record(executable)

      self.assertEqual(record.emitter.kind, "symirc")
      self.assertEqual(record.emitter.executable, str(executable))
      self.assertEqual(record.emitter.binary_sha256, _digest_bytes(STUB_BYTES))
      self.assertEqual(record.target, "triton")
      self.assertEqual(record.entry, ENTRY)
      self.assertEqual(record.profile, PROFILE)
      self.assertEqual(record.layout_manifest_sha256, _value_digest(LAYOUT_MANIFEST))
      self.assertEqual(record.softmax_lowering, "direct")
      self.assertEqual(record.emitter.options, (("tensor_profile_version", "v2"),))
      self.assertEqual(record.kernel_sha256, _digest(ORDINARY_KERNEL))
      self.assertEqual(record.kernel.source, ORDINARY_KERNEL)
      self.assertEqual(record.sir_sha256, _digest(NATIVE_SEED_7_SOURCE))
      self.assertEqual(record.layout_sha256, _value_digest(LAYOUT))

      stated = record.as_record()
      self.assertEqual(stated["schema"], emission.SCHEMA)
      self.assertEqual(stated["kind"], "symirc")
      self.assertEqual(stated["target"], "triton")
      self.assertEqual(stated["layout_manifest_sha256"], _value_digest(LAYOUT_MANIFEST))
      self.assertEqual(stated["softmax_lowering"], "direct")
      self.assertEqual(stated["emitter"]["executable"], str(executable))
      self.assertEqual(stated["emitter"]["binary_sha256"], _digest_bytes(STUB_BYTES))
      self.assertEqual(stated["emitter"]["options"], {"tensor_profile_version": "v2"})

      kernel = emission.reuse(
        record, executable=executable, **_ordinary_identity_arguments()
      )
      self.assertEqual(kernel.source, ORDINARY_KERNEL)

  def test_changing_an_emitter_option_is_refused(self):
    emission = _emission()
    with tempfile.TemporaryDirectory() as directory:
      executable = _stub(directory)
      first = _ordinary_record(executable)
      second = _ordinary_record(executable, options={"tensor_profile_version": "v3"})

      self.assertNotEqual(first.emitter.digest, second.emitter.digest)
      with self.assertRaisesRegex(emission.EmissionRefused, "option"):
        emission.reuse(
          first,
          executable=executable,
          **_ordinary_identity_arguments(options={"tensor_profile_version": "v3"}),
        )

  def test_an_ordinary_record_from_another_sir_case_is_refused(self):
    emission = _emission()
    with tempfile.TemporaryDirectory() as directory:
      executable = _stub(directory)
      record = _ordinary_record(executable)

      with self.assertRaisesRegex(emission.EmissionRefused, "SIR"):
        emission.reuse(
          record,
          executable=executable,
          **_ordinary_identity_arguments(sir=ORTHOGONAL_REGION_SOURCE),
        )

  def test_a_changed_layout_digest_is_refused(self):
    emission = _emission()
    with tempfile.TemporaryDirectory() as directory:
      executable = _stub(directory)
      record = _ordinary_record(executable)

      with self.assertRaisesRegex(emission.EmissionRefused, "layout"):
        emission.reuse(
          record,
          executable=executable,
          **_ordinary_identity_arguments(layout=OTHER_LAYOUT),
        )

  def test_a_changed_kernel_byte_is_refused(self):
    emission = _emission()
    with tempfile.TemporaryDirectory() as directory:
      executable = _stub(directory)
      record = _ordinary_record(executable)
      tampered = replace(
        record,
        kernel=emission.EmittedKernel(source=record.kernel.source + "# tampered\n"),
      )

      with self.assertRaisesRegex(emission.EmissionRefused, "kernel"):
        emission.reuse(
          tampered,
          executable=executable,
          **_ordinary_identity_arguments(),
        )

  def test_changing_the_symirc_executable_is_an_emitter_change(self):
    emission = _emission()
    with tempfile.TemporaryDirectory() as directory:
      executable = _stub(directory)
      other = _stub(directory, name="other-symirc", body=b"#!/bin/sh\nexit 1\n")
      record = _ordinary_record(executable)

      with self.assertRaisesRegex(emission.EmissionRefused, "emitter executable"):
        emission.reuse(record, executable=other, **_ordinary_identity_arguments())

  def test_build_states_the_ordinary_kind_directly(self):
    emission = _emission()
    with tempfile.TemporaryDirectory() as directory:
      executable = _stub(directory)
      record = emission.build(
        render=lambda: ORDINARY_KERNEL,
        kind="symirc",
        executable=executable,
        **_ordinary_identity_arguments(),
      )

      self.assertEqual(record.emitter.kind, "symirc")
      self.assertEqual(record.kernel.source, ORDINARY_KERNEL)
      self.assertEqual(
        emission.reuse(
          record, executable=executable, **_ordinary_identity_arguments()
        ).source,
        ORDINARY_KERNEL,
      )

  def test_gluon_and_symirc_records_cannot_be_reused_across_kinds(self):
    emission = _emission()
    gluon = _record(emission)
    with tempfile.TemporaryDirectory() as directory:
      executable = _stub(directory)
      ordinary = _ordinary_record(executable)

      self.assertEqual(gluon.sir_sha256, ordinary.sir_sha256)
      self.assertEqual(gluon.emitter.kind, "gluon")
      self.assertNotEqual(gluon.emitter.kind, ordinary.emitter.kind)
      with self.assertRaisesRegex(emission.EmissionRefused, "kind"):
        emission.reuse(gluon, executable=executable, **_ordinary_identity_arguments())
      with self.assertRaisesRegex(emission.EmissionRefused, "kind"):
        emission.reuse(ordinary, **_gluon_identity_arguments())


if __name__ == "__main__":
  unittest.main()
