"""One recorded Gluon emission is the kernel both execution paths consume."""

import hashlib
import unittest
from dataclasses import replace
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


if __name__ == "__main__":
  unittest.main()
