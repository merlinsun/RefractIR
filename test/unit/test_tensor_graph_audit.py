"""Tests for the compositional tensor graph dependency audit.

The audit keeps three identities apart: the source bytes, the ordered
dependency topology, and the operator-labelled graph. These tests pin the
invariance rules that keep dimensions, constants, names, and axes from
inflating the structural claim, and they require a witness for the
independent-branch join the builder has to reach.
"""

import re
import tempfile
import unittest

from test.lib.tensor_graph_audit import (
  PROFILES,
  RYSMITH,
  audit_profile,
  feature_witnesses,
  generate_case,
  literal_signature,
  operator_digest,
  parse_graph,
  return_slice,
  shape_signature,
  topology_digest,
)

# Enough seeds to witness a scheduled feature while keeping one audit cheap.
SEEDS = range(16)

# Seeded sources are regenerated per process when a test runs alone.
_SOURCES = {}
_REPORTS = {}

# One shared intermediate feeds both branches, which meet again at the join.
SHARED_INTERMEDIATE = """fun @kernel(%a: tensor [2, 2] i32, %b: tensor [2, 2] i32) \
: tensor [2, 2] i32 {
  let mut %t: tensor [2, 2] i32 = undef;
  let mut %left: tensor [2, 2] i32 = undef;
  let mut %right: tensor [2, 2] i32 = undef;
  let mut %out: tensor [2, 2] i32 = undef;
^entry:
  %t = tensor add %a, %b;
  %left = tensor add %t, 1;
  %right = tensor add %t, 2;
  %out = tensor add %left, %right;
  ret %out;
}
"""

# The two branches carry identical computations over the same inputs, but they
# are separate values, so neither branch is shared with the other.
DUPLICATED_COMPUTATION = """fun @kernel(%a: tensor [2, 2] i32, %b: tensor [2, 2] i32) \
: tensor [2, 2] i32 {
  let mut %t0: tensor [2, 2] i32 = undef;
  let mut %t1: tensor [2, 2] i32 = undef;
  let mut %left: tensor [2, 2] i32 = undef;
  let mut %right: tensor [2, 2] i32 = undef;
  let mut %out: tensor [2, 2] i32 = undef;
^entry:
  %t0 = tensor add %a, %b;
  %t1 = tensor add %a, %b;
  %left = tensor add %t0, 1;
  %right = tensor add %t1, 2;
  %out = tensor add %left, %right;
  ret %out;
}
"""


def compositional_source(*, profile, seed):
  """Return one generated compositional source, generated at most once."""
  key = (profile, seed)
  if key not in _SOURCES:
    with tempfile.TemporaryDirectory() as directory:
      generated = generate_case(profile, seed, root=directory)
    if not generated.accepted:
      raise AssertionError(
        f"{profile} seed {seed} did not generate a case: {generated.reason}"
      )
    _SOURCES[key] = generated.source
  return _SOURCES[key]


def audit_report(profile, seeds=SEEDS):
  """Return one profile audit, computed at most once per process."""
  key = (profile, tuple(seeds))
  if key not in _REPORTS:
    _REPORTS[key] = audit_profile(profile, seeds=seeds)
  return _REPORTS[key]


def rename_locals(source, mapping):
  """Rename every local and parameter, keeping the program structure."""
  return re.sub(
    r"%[A-Za-z_][\w.$-]*",
    lambda match: mapping.get(match.group(), match.group()),
    source,
  )


_OPERAND_LITERAL = re.compile(
  r"(?m)^(\s*%[A-Za-z_][\w.$-]* = tensor (?:add|sub|mul|div|rem|min|max) "
  r"%[A-Za-z_][\w.$-]*, )(-?\d+)(;)$"
)
_TYPE_SHAPE = re.compile(r"tensor \[([0-9, ]+)\]")


def rewrite_constants_and_shapes(source):
  """Change every literal operand and every declared tensor shape."""

  def replace_literal(match):
    return f"{match.group(1)}{int(match.group(2)) + 3}{match.group(3)}"

  def replace_shape(match):
    dimensions = [int(value) for value in match.group(1).split(",")]
    return "tensor [" + ", ".join(str(value + 1) for value in dimensions) + "]"

  rewritten = _OPERAND_LITERAL.sub(replace_literal, source)
  return _TYPE_SHAPE.sub(replace_shape, rewritten)


class TensorGraphAuditTests(unittest.TestCase):
  def setUp(self):
    self.assertTrue(RYSMITH.is_file(), f"rysmith executable not found: {RYSMITH}")

  def test_renaming_locals_preserves_topology_identity(self):
    for profile in PROFILES:
      with self.subTest(profile=profile):
        source = compositional_source(profile=profile, seed=3)
        graph = parse_graph(source)
        names = sorted({value.name for value in graph.values})
        mapping = {name: f"%renamed{index}" for index, name in enumerate(names)}
        renamed = rename_locals(source, mapping)

        self.assertNotEqual(source, renamed)
        self.assertEqual(graph.values[0].name, "%a")
        self.assertEqual(parse_graph(renamed).values[0].name, "%renamed0")
        self.assertEqual(topology_digest(graph), topology_digest(parse_graph(renamed)))
        self.assertEqual(operator_digest(graph), operator_digest(parse_graph(renamed)))

  def test_changing_constants_and_shapes_preserves_topology_identity(self):
    for profile in PROFILES:
      with self.subTest(profile=profile):
        source = compositional_source(profile=profile, seed=4)
        graph = parse_graph(source)
        changed = rewrite_constants_and_shapes(source)
        other = parse_graph(changed)

        self.assertNotEqual(source, changed)
        self.assertNotEqual(literal_signature(graph), literal_signature(other))
        self.assertNotEqual(shape_signature(graph), shape_signature(other))
        self.assertEqual(topology_digest(graph), topology_digest(other))
        self.assertEqual(operator_digest(graph), operator_digest(other))

  def test_shared_intermediate_and_duplicated_computation_differ(self):
    shared = parse_graph(SHARED_INTERMEDIATE)
    duplicated = parse_graph(DUPLICATED_COMPUTATION)

    self.assertEqual(
      sorted(value.name for value in shared.computations),
      ["%left", "%out", "%right", "%t"],
    )
    self.assertEqual(
      sorted(value.name for value in duplicated.computations),
      ["%left", "%out", "%right", "%t0", "%t1"],
    )
    self.assertEqual(
      {value.shape for value in shared.values},
      {value.shape for value in duplicated.values},
    )
    self.assertNotEqual(topology_digest(shared), topology_digest(duplicated))
    self.assertNotEqual(operator_digest(shared), operator_digest(duplicated))
    self.assertIn("shared-intermediate-diamond", feature_witnesses(shared))
    self.assertNotIn("shared-intermediate-diamond", feature_witnesses(duplicated))
    self.assertNotIn("independent-branch-join", feature_witnesses(duplicated))

  def test_independent_branch_join_is_recognized_as_a_feature(self):
    for profile in PROFILES:
      with self.subTest(profile=profile):
        report = audit_report(profile)
        self.assertEqual(report.attempted, len(SEEDS))
        self.assertGreaterEqual(
          report.feature_counts.get("independent-branch-join", 0), 1
        )
        witness = report.feature_witnesses["independent-branch-join"]
        self.assertIn("seed", witness)
        self.assertIn("tensor ", witness["witness"])
        self.assertIn("%", witness["source"])

  def test_every_declared_computation_lies_in_the_return_slice(self):
    for profile in PROFILES:
      with self.subTest(profile=profile):
        report = audit_report(profile)
        # Refused seeds stay in the denominator, so the count is a floor, not a
        # promise that every seed generates. A dead computation is its own
        # refusal reason and must never appear.
        self.assertEqual(report.refusals.get("dead_computation", 0), 0)
        self.assertGreaterEqual(report.accepted, len(SEEDS) - 1)
        for case in report.cases:
          if not case.accepted:
            continue
          graph = parse_graph(case.source)
          declared = {value.name for value in graph.values if not value.is_input}
          self.assertTrue(declared)
          self.assertIn(graph.result, return_slice(graph))
          self.assertEqual(declared - return_slice(graph), set())

  def test_scalar_reduction_followed_by_expand_is_recognized(self):
    for profile in PROFILES:
      with self.subTest(profile=profile):
        report = audit_report(profile)
        self.assertGreaterEqual(
          report.feature_counts.get("scalar-reduction-then-expand", 0), 1
        )
        witness = report.feature_witnesses["scalar-reduction-then-expand"]
        self.assertIn("tensor sum", witness["witness"])
        self.assertIn("tensor expand", witness["witness"])
        self.assertIn("tensor ", witness["source"])


if __name__ == "__main__":
  unittest.main()
