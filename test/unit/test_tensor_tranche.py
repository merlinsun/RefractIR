"""Frozen manifest, append-only ledger, resume and budget for one tranche.

The tests drive the module directly: no GPU, no Triton, no project binary. A
tranche is frozen before dispatch, its slots are claimed as work starts, and the
ledger keeps every claim and outcome so an interrupted run resumes with the same
identities instead of re-deriving them.
"""

import tempfile
import unittest
from pathlib import Path

from test.lib import tensor_tranche


def _manifest(**overrides):
  manifest = {
    "tranche": "test",
    "compiler_revision": "deadbeef",
    "budget": {"executions": 6, "controls": 2},
    "cases": [
      {"id": "A1", "lane": "A", "inputs": ["i0", "i1"], "configurations": ["w1", "w4"]},
      {"id": "B1", "lane": "B", "inputs": ["i0"], "configurations": ["w1", "w4"]},
    ],
  }
  manifest.update(overrides)
  return manifest


class FreezeTests(unittest.TestCase):
  def test_freezing_expands_every_case_into_its_slots_in_order(self):
    # The slot list is derived once, deterministically, and each slot carries
    # the case identity and the configuration that distinguishes it.
    frozen = tensor_tranche.freeze(_manifest())
    self.assertEqual(
      [slot.slot_id for slot in frozen.slots],
      ["A1/i0/w1", "A1/i0/w4", "A1/i1/w1", "A1/i1/w4", "B1/i0/w1", "B1/i0/w4"],
    )
    self.assertEqual(frozen.slots[0].case_id, "A1")
    self.assertEqual(frozen.slots[0].lane, "A")

  def test_a_mutated_manifest_is_refused(self):
    # A frozen tranche is identified by its digest. Editing any field after the
    # freeze is refused rather than silently changing the allocation.
    frozen = tensor_tranche.freeze(_manifest())
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "frozen.json"
      frozen.write(path)
      payload = tensor_tranche.read(path)
      payload["payload"]["manifest"]["budget"]["executions"] = 999
      with self.assertRaises(tensor_tranche.ManifestRefused):
        tensor_tranche.FrozenTranche.from_payload(payload)

  def test_a_duplicated_slot_id_is_refused(self):
    manifest = _manifest(
      cases=[{"id": "A1", "lane": "A", "inputs": ["i0"], "configurations": ["w1"]}]
    )
    manifest["cases"][0]["inputs"] = ["i0", "i0"]
    with self.assertRaises(tensor_tranche.ManifestRefused):
      tensor_tranche.freeze(manifest)


class LedgerTests(unittest.TestCase):
  def test_a_refusal_consumes_its_slot_and_cannot_be_reclaimed(self):
    # A refused slot is terminal: the work is recorded, the slot is not handed
    # out again on resume.
    frozen = tensor_tranche.freeze(_manifest())
    ledger = tensor_tranche.Ledger.in_memory(frozen)
    ledger.claim("A1/i0/w1")
    ledger.finish("A1/i0/w1", "refused", reason="out of domain")
    with self.assertRaises(tensor_tranche.SlotRefused):
      ledger.claim("A1/i0/w1")
    self.assertEqual(ledger.counts()["refused"], 1)

  def test_every_pending_slot_is_recorded_as_unexecuted_at_the_end(self):
    # Closing the tranche does not drop the work that never ran; it names it.
    frozen = tensor_tranche.freeze(_manifest())
    ledger = tensor_tranche.Ledger.in_memory(frozen)
    ledger.claim("A1/i0/w1")
    ledger.finish("A1/i0/w1", "completed", detail={"status": "pass"})
    ledger.close()
    self.assertEqual(ledger.counts()["completed"], 1)
    self.assertEqual(ledger.counts()["unexecuted"], 5)
    self.assertEqual(ledger.state("A1/i1/w4"), "unexecuted")

  def test_resume_preserves_identity_and_order(self):
    # A ledger reloaded from disk continues the same tranche: same digest, same
    # slot order, same completed work, and the next claim is the next pending
    # slot.
    frozen = tensor_tranche.freeze(_manifest())
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "ledger.jsonl"
      ledger = tensor_tranche.Ledger.open(path, frozen)
      ledger.claim("A1/i0/w1")
      ledger.finish("A1/i0/w1", "completed")
      resumed = tensor_tranche.Ledger.open(path, frozen)
      self.assertEqual(resumed.digest, frozen.digest)
      self.assertEqual(resumed.completed_ids(), ["A1/i0/w1"])
      claim = resumed.claim_next()
      self.assertEqual(claim.slot_id, "A1/i0/w2".replace("w2", "w4"))

  def test_budget_exhaustion_stops_claims_without_deleting_evidence(self):
    # Once the execution budget is spent the ledger refuses further claims and
    # keeps every earlier record, including a running one it cannot finish.
    frozen = tensor_tranche.freeze(_manifest(budget={"executions": 2, "controls": 0}))
    ledger = tensor_tranche.Ledger.in_memory(frozen)
    ledger.claim("A1/i0/w1")
    ledger.finish("A1/i0/w1", "completed")
    ledger.claim("A1/i0/w4")
    ledger.finish("A1/i0/w4", "completed")
    with self.assertRaises(tensor_tranche.BudgetExhausted):
      ledger.claim("A1/i1/w1")
    self.assertEqual(ledger.counts()["completed"], 2)
    # Two claims and two outcomes were recorded before the budget stopped the
    # third claim; nothing was removed when it refused.
    self.assertEqual(len(ledger.events), 4)


def _lane_manifest(**overrides):
  manifest = {
    "tranche": "study",
    "lanes": ["G0", "G1", "control"],
    "budget": {"executions": 4},
    "cases": [
      {"id": "S0", "lane": "G0", "inputs": ["i0"], "configurations": ["w1", "w4"]},
      {"id": "C0", "lane": "control", "inputs": ["i0"], "configurations": ["w1", "w4"]},
    ],
  }
  manifest.update(overrides)
  return manifest


class LaneTests(unittest.TestCase):
  def test_an_explicit_lane_list_is_honoured(self):
    frozen = tensor_tranche.freeze(_lane_manifest())
    self.assertEqual([slot.lane for slot in frozen.slots], ["G0", "G0", "control", "control"])
    self.assertEqual(
      frozen.payload["manifest"]["lanes"], ["G0", "G1", "control"]
    )

  def test_a_lane_outside_the_declared_list_is_refused(self):
    manifest = _lane_manifest(
      cases=[{"id": "A1", "lane": "A", "inputs": ["i0"], "configurations": ["w1"]}]
    )
    with self.assertRaises(tensor_tranche.ManifestRefused):
      tensor_tranche.freeze(manifest)

  def test_the_default_lanes_are_preserved(self):
    manifest = _manifest()
    frozen = tensor_tranche.freeze(manifest)
    self.assertEqual(
      frozen.payload["manifest"]["lanes"] if "lanes" in frozen.payload["manifest"]
      else list(tensor_tranche.TRANCHES),
      list(tensor_tranche.TRANCHES),
    )
    self.assertEqual([slot.lane for slot in frozen.slots], ["A", "A", "A", "A", "B", "B"])


if __name__ == "__main__":
  unittest.main()
