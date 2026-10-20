"""Study records carry the compiler observer's target evidence without restating it.

The observations these tests assemble are produced by the real observer entry
points, so each fixture is the shape a case archive yields. The tests assert the
record copies the observer's own fields rather than re-deriving them.
"""

import unittest

from test.lib import tensor_compiler_observation, tensor_target_evidence

_LAYOUTS = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 0, parent = #blocked}>
#loc1 = loc("kernel.py":41:12)
#loc2 = loc("kernel.py":42:12)
"""

_LOOP_BODY = """\
module {
  tt.func public @k(%x: tensor<4xf32, #blocked>) {
    %acc = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %x) -> (tensor<4xf32>) {
      %0 = ttg.convert_layout %a : tensor<4xf32, #blocked> -> tensor<4xf32, #slice> loc(#loc1)
      scf.yield %a : tensor<4xf32>
    }
  }
}
"""

_STRAIGHT_LINE_BODY = """\
module {
  tt.func public @k(%x: tensor<4xf32, #blocked>) {
    %a = arith.addf %x, %x : tensor<4xf32, #blocked>
  }
}
"""

_PTX_LOOP = """\
//
.version 8.8
.target sm_70
.visible .entry kernel() {
$L__BB0_1:
  ld.shared.b32 %r1, [%r2];
  shfl.sync.bfly.b32 %r3, %r4, 1, 31, -1;
  @%p1 bra $L__BB0_1;
  ret;
}
"""

_PTX_STRAIGHT_LINE = """\
//
.version 8.8
.target sm_70
.visible .entry kernel() {
  ld.shared.b32 %r1, [%r2];
  ret;
}
"""

_PTX_QUIET = """\
//
.version 8.8
.target sm_70
.visible .entry kernel() {
  ld.global.b32 %r1, [%rd1];
  ret;
}
"""

_PTX_UNPARSED_BRANCH = """\
//
.version 8.8
.target sm_70
.visible .entry kernel() {
  bra $L__BB0_9;
  ret;
}
"""


def _pass_snapshot(name, ir):
  return (
    f"// -----// IR Dump Before {name}: {name.lower()} "
    "('builtin.module' operation) //----- //\n"
    f"{ir}\n"
  )


def _compiler_observation(*, ptx=_PTX_LOOP, ttgir=_LOOP_BODY):
  artifacts = {"ttgir": ttgir}
  if ptx is not None:
    artifacts["ptx"] = ptx
  return tensor_compiler_observation.observe_compiler_ir(
    artifacts, expected_loop=True, loop_required_stages=("ptx",)
  )


_IN_LOOP_DUMP = "".join(
  (
    _pass_snapshot(
      "TritonLoopInvariantCodeMotion",
      _LAYOUTS
      + """\
module {
  tt.func public @k(%x: tensor<4xf32, #blocked>, %y: tensor<4xf32, #blocked>) {
    %f = ttg.convert_layout %x : tensor<4xf32, #blocked> -> tensor<4xf32, #slice> loc(#loc1)
    %g = ttg.convert_layout %y : tensor<4xf32, #blocked> -> tensor<4xf32, #blocked_alt> loc(#loc2)
  }
}
""",
    ),
    _pass_snapshot(
      "TritonGPURemoveLayoutConversions",
      _LAYOUTS
      + """\
module {
  tt.func public @k(%x: tensor<4xf32, #blocked>, %y: tensor<4xf32, #blocked>) {
    %acc = scf.for %i = %c0 to %c4 step %c1 iter_args(%v = %x) -> (tensor<4xf32>) {
      %7 = ttg.convert_layout %x : tensor<4xf32, #blocked> -> tensor<4xf32, #slice> loc(#loc1)
      scf.yield %v : tensor<4xf32>
    }
  }
}
""",
    ),
    _pass_snapshot(
      "LLVMDIScope",
      _LAYOUTS
      + """\
module {
  tt.func public @k(%x: tensor<4xf32, #blocked>, %y: tensor<4xf32, #blocked>) {
    %acc = scf.for %i = %c0 to %c4 step %c1 iter_args(%v = %x) -> (tensor<4xf32>) {
      scf.yield %v : tensor<4xf32>
    }
  }
}
""",
    ),
  )
)

_REMATERIALIZATION_DUMP = "".join(
  (
    _pass_snapshot(
      "TritonGPURemoveLayoutConversions",
      _LAYOUTS
      + """\
module {
  %b = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>
  %d = arith.addf %b, %b : tensor<32x16xf32, #blocked>
  %e = arith.addf %b, %b : tensor<32x16xf32, #blocked>
}
""",
    ),
    _pass_snapshot(
      "LLVMDIScope",
      _LAYOUTS
      + """\
module {
  %b = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>
  %b2 = tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked_alt>
  %d = arith.addf %b2, %b2 : tensor<32x16xf32, #blocked_alt>
  %e = arith.addf %b, %b : tensor<32x16xf32, #blocked>
}
""",
    ),
  )
)

_IN_LOOP_PASSES = (
  "TritonLoopInvariantCodeMotion",
  "TritonGPURemoveLayoutConversions",
)


def _in_loop_pass_observation(dump=_IN_LOOP_DUMP):
  """Two named passes: one moves a conversion into the loop, one removes it."""
  return tensor_compiler_observation.observe_pass_dumps(
    dump,
    source_lines={41: "load_to_compute", 42: "input_to_compute"},
    target_passes=_IN_LOOP_PASSES,
    rematerialization=True,
  )


def _reached_pass_observation():
  """The in-loop passes plus the pass that confirms rematerialization."""
  return _in_loop_pass_observation(_IN_LOOP_DUMP + _REMATERIALIZATION_DUMP)


def _rematerialization_pass_observation():
  """One pass computes the same broadcast in another encoding for its consumer."""
  return tensor_compiler_observation.observe_pass_dumps(
    _REMATERIALIZATION_DUMP,
    target_passes=("TritonGPURemoveLayoutConversions",),
    rematerialization=True,
  )


def _no_change_pass_observation():
  dump = "".join(
    (
      _pass_snapshot("CanonicalizerPass", _LAYOUTS + _STRAIGHT_LINE_BODY),
      _pass_snapshot("LLVMDIScope", _LAYOUTS + _STRAIGHT_LINE_BODY),
    )
  )
  return tensor_compiler_observation.observe_pass_dumps(
    dump,
    target_passes=("CanonicalizerPass",),
    rematerialization=True,
  )


class TensorTargetEvidenceTests(unittest.TestCase):
  def test_the_record_carries_the_ptx_backedge_observation(self):
    observation = _compiler_observation()

    record = tensor_target_evidence.assemble_study_record(
      "case-ptx",
      compiler_observation=observation,
      pass_observation=None,
      candidate_validity=tensor_target_evidence.CandidateValidity(valid=True),
    )

    ptx = record["ptx_loop"]
    observed = observation["stages"]["ptx"]
    self.assertEqual(ptx["status"], "observed")
    self.assertEqual(ptx["stage"], "ptx")
    self.assertEqual(ptx["backedge_count"], observed["backedge_count"])
    self.assertEqual(ptx["backedge_count"], 1)
    self.assertEqual(ptx["loop_construct_count"], observed["loop_construct_count"])
    self.assertEqual(ptx["unparsed_branch_count"], observed["unparsed_branch_count"])
    self.assertEqual(ptx["loop_evidence"], observation["loop_evidence"]["ptx"])
    self.assertEqual(ptx["loop_evidence"], "branch-cycle")
    self.assertIs(ptx["loop_presence"], observation["loop_presence"]["ptx"])
    self.assertIs(ptx["loop_presence"], True)
    self.assertEqual(ptx["loop_status"], observation["loop_status"])

  def test_the_record_carries_in_loop_conversion_attribution(self):
    observation = _in_loop_pass_observation()

    record = tensor_target_evidence.assemble_study_record(
      "case-in-loop",
      compiler_observation=None,
      pass_observation=observation,
      candidate_validity=tensor_target_evidence.CandidateValidity(valid=True),
    )

    section = record["in_loop_conversions"]
    self.assertEqual(section["status"], "observed")
    self.assertEqual(
      section["named_passes"],
      ["TritonGPURemoveLayoutConversions", "TritonLoopInvariantCodeMotion"],
    )
    passes = {entry["pass"]: entry for entry in section["passes"]}
    removed = passes["TritonGPURemoveLayoutConversions"]
    self.assertEqual(removed["removed_in_loop_regions"], {"scf.for": 1})
    self.assertEqual(removed["removed_outside_a_loop"], 0)
    self.assertEqual(removed["loop_constructs_before"], 1)
    self.assertEqual(removed["loop_constructs_after"], 1)
    self.assertEqual(removed["conversion_match"]["removed"], 1)
    self.assertEqual(
      removed["effective_conversion_operation_deltas"], {"load_to_compute": -1}
    )
    moved = passes["TritonLoopInvariantCodeMotion"]
    self.assertEqual(moved["removed_in_loop_regions"], {})
    self.assertEqual(moved["removed_outside_a_loop"], 1)
    self.assertEqual(len(moved["conversion_movement"]), 1)
    self.assertIsNone(moved["conversion_movement"][0]["enclosing_loop_before"])
    self.assertEqual(moved["conversion_movement"][0]["enclosing_loop_after"], "scf.for")
    self.assertEqual(section["in_loop_removed_count"], 1)
    self.assertEqual(
      record["research_objectives"]["in_loop_conversion_change"]["status"], "reached"
    )

  def test_the_record_carries_every_field_of_a_confirmed_rematerialization(self):
    observation = _rematerialization_pass_observation()

    record = tensor_target_evidence.assemble_study_record(
      "case-rematerialized",
      compiler_observation=None,
      pass_observation=observation,
      candidate_validity=tensor_target_evidence.CandidateValidity(valid=True),
    )

    observed = observation["compilations"][0]["transitions"][0]["rematerialization"]
    section = record["rematerialization"]
    self.assertEqual(section["status"], "observed")
    self.assertEqual(section["confirmed_count"], 1)
    self.assertEqual(section["observations"][0]["status"], observed["status"])
    entry = section["observations"][0]["confirmed"][0]
    self.assertIsInstance(entry, dict)
    self.assertEqual(set(entry), set(observed["confirmed"][0]))
    self.assertEqual(entry, observed["confirmed"][0])
    self.assertEqual(entry["original_value"], "%b")
    self.assertEqual(
      entry["original_expression"],
      "tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked>",
    )
    self.assertEqual(entry["recomputed_values"], ["%b2"])
    self.assertEqual(
      entry["expression"],
      "tt.broadcast %x : tensor<1x16xf32, #blocked> -> tensor<32x16xf32, #blocked_alt>",
    )
    self.assertEqual(entry["redirected_consumers"], ["%d"])
    self.assertEqual(entry["remaining_consumers"], ["%e"])
    self.assertTrue(entry["encoding_change"]["changed"])
    self.assertEqual(
      entry["encoding_change"]["recomputed"][1],
      "#ttg.blocked<{sizePerThread=[1,4],threadsPerWarp=[8,4],"
      "warpsPerCTA=[1,1],order=[1,0]}>",
    )
    self.assertIsNone(entry["source_line"])
    self.assertEqual(
      section["observations"][0]["interpretation_limit"],
      observed["interpretation_limit"],
    )

  def test_a_missing_stage_or_pass_is_unavailable_with_its_reason(self):
    compiler = _compiler_observation(ptx=None)
    no_pass_dump = {
      "status": "unavailable",
      "reason": "no pass dump was archived",
    }

    record = tensor_target_evidence.assemble_study_record(
      "case-missing",
      compiler_observation=compiler,
      pass_observation=no_pass_dump,
      candidate_validity=tensor_target_evidence.CandidateValidity(valid=True),
    )

    ptx = record["ptx_loop"]
    self.assertEqual(ptx["status"], "unavailable")
    self.assertIn("ptx", ptx["reason"])
    self.assertIsNone(ptx["backedge_count"])
    for name in ("in_loop_conversions", "rematerialization"):
      section = record[name]
      self.assertEqual(section["status"], "unavailable")
      self.assertEqual(section["reason"], "no pass dump was archived")
    objectives = record["research_objectives"]
    for name in (
      "retained_ptx_loop",
      "in_loop_conversion_change",
      "shared_memory_or_shuffle_communication",
      "confirmed_rematerialization",
    ):
      self.assertEqual(objectives[name]["status"], "unavailable")
      self.assertTrue(objectives[name]["reason"])
    for name in (
      "mask_and_stride_evidence",
      "gluon_effective_mapping_and_conversion_survival",
    ):
      self.assertEqual(objectives[name]["status"], "unavailable")
      self.assertTrue(objectives[name]["reason"])

    without_any = tensor_target_evidence.assemble_study_record(
      "case-missing",
      compiler_observation=None,
      pass_observation=None,
      candidate_validity=tensor_target_evidence.CandidateValidity(valid=True),
    )
    self.assertEqual(without_any["ptx_loop"]["status"], "unavailable")
    self.assertIn("supplied", without_any["ptx_loop"]["reason"])
    self.assertEqual(without_any["in_loop_conversions"]["status"], "unavailable")
    self.assertIn("supplied", without_any["in_loop_conversions"]["reason"])
    self.assertEqual(without_any["rematerialization"]["status"], "unavailable")
    self.assertIn("supplied", without_any["rematerialization"]["reason"])

  def test_an_unreached_objective_leaves_candidate_validity_unchanged(self):
    validity = tensor_target_evidence.CandidateValidity(valid=True)
    reached = tensor_target_evidence.assemble_study_record(
      "case-valid",
      compiler_observation=_compiler_observation(),
      pass_observation=_reached_pass_observation(),
      candidate_validity=validity,
      objective_evidence={
        "mask_and_stride_evidence": {
          "status": "reached",
          "evidence": {"mask_words": ["0xffffffff"], "stride": 2},
        },
        "gluon_effective_mapping_and_conversion_survival": {
          "status": "reached",
          "evidence": {"mapping": [[0, 1], [1, 0]]},
        },
      },
    )
    unreached = tensor_target_evidence.assemble_study_record(
      "case-valid",
      compiler_observation=_compiler_observation(
        ptx=_PTX_QUIET, ttgir=_STRAIGHT_LINE_BODY
      ),
      pass_observation=_no_change_pass_observation(),
      candidate_validity=validity,
    )

    self.assertEqual(reached["candidate_validity"], {"valid": True, "reason": None})
    self.assertEqual(unreached["candidate_validity"], reached["candidate_validity"])
    statuses = {
      name: entry["status"] for name, entry in reached["research_objectives"].items()
    }
    self.assertEqual(set(statuses.values()), {"reached"})
    other = {
      name: entry["status"] for name, entry in unreached["research_objectives"].items()
    }
    self.assertEqual(other["retained_ptx_loop"], "not_reached")
    self.assertEqual(other["in_loop_conversion_change"], "not_reached")
    self.assertEqual(other["shared_memory_or_shuffle_communication"], "not_reached")
    self.assertEqual(other["confirmed_rematerialization"], "not_reached")
    self.assertNotEqual(statuses, other)

    invalid = tensor_target_evidence.assemble_study_record(
      "case-invalid",
      compiler_observation=_compiler_observation(),
      pass_observation=_reached_pass_observation(),
      candidate_validity=tensor_target_evidence.CandidateValidity(
        valid=False, reason="output guard write"
      ),
    )
    self.assertEqual(
      invalid["candidate_validity"],
      {"valid": False, "reason": "output guard write"},
    )


class TensorTargetEvidenceEdgeTests(unittest.TestCase):
  def test_an_unclassified_ptx_branch_is_unavailable_not_loop_free(self):
    observation = _compiler_observation(ptx=_PTX_UNPARSED_BRANCH)

    record = tensor_target_evidence.assemble_study_record(
      "case-unparsed-branch",
      compiler_observation=observation,
      pass_observation=None,
      candidate_validity=tensor_target_evidence.CandidateValidity(valid=True),
    )

    ptx = record["ptx_loop"]
    self.assertEqual(ptx["status"], "unavailable")
    self.assertIn("could not classify", ptx["reason"])
    self.assertEqual(ptx["backedge_count"], 0)
    self.assertEqual(ptx["unparsed_branch_count"], 1)
    self.assertEqual(
      record["research_objectives"]["retained_ptx_loop"]["status"], "unavailable"
    )
    self.assertEqual(record["candidate_validity"], {"valid": True, "reason": None})

  def test_a_pass_dump_without_a_transition_marks_the_pass_evidence_unavailable(self):
    observation = tensor_compiler_observation.observe_pass_dumps(
      _pass_snapshot("LLVMDIScope", _LOOP_BODY), rematerialization=True
    )
    self.assertEqual(observation["compilations"][0]["transitions"], [])

    record = tensor_target_evidence.assemble_study_record(
      "case-single-snapshot",
      compiler_observation=None,
      pass_observation=observation,
      candidate_validity=tensor_target_evidence.CandidateValidity(valid=True),
    )

    self.assertEqual(record["in_loop_conversions"]["status"], "unavailable")
    self.assertIn("transition", record["in_loop_conversions"]["reason"])
    self.assertEqual(record["rematerialization"]["status"], "unavailable")
    self.assertIn("transition", record["rematerialization"]["reason"])
    self.assertEqual(
      record["research_objectives"]["in_loop_conversion_change"]["status"],
      "unavailable",
    )
    self.assertEqual(
      record["research_objectives"]["confirmed_rematerialization"]["status"],
      "unavailable",
    )
    self.assertEqual(record["candidate_validity"], {"valid": True, "reason": None})


if __name__ == "__main__":
  unittest.main()
