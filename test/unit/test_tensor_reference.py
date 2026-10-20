"""Fresh reference replay takes identity from the case and classifies it.

The replay contract is exercised with shell scripts standing in for `symiri`,
so no GPU, Triton, or real interpreter build is needed. The scripts parse the
same flags the module passes and decide their own bytes, output, and status.
"""

from __future__ import annotations

import hashlib
import shlex
import tempfile
import unittest
from pathlib import Path

from test.lib import tensor_case, tensor_reference

# Flags the reference replay may use. Each fake interpreter parses the ones it
# needs and ignores the rest, so a test asserts the command through behaviour as
# well as through the recorded command.
_FAKE_HEADER = """\
#!/bin/sh
set -u
out=
bits=
profile=
entry=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --tensor-output) out=$2; shift 2 ;;
    --tensor-reference-bits) bits=$2; shift 2 ;;
    --tensor-profile-version) profile=$2; shift 2 ;;
    --entry) entry=$2; shift 2 ;;
    *) shift ;;
  esac
done
"""


def _case(
  root,
  *,
  profile="tensor-v2",
  entry="kernel",
  mode="exact",
  expected=b"\x01\x02\x03\x04",
  program=b"fun @kernel() : tensor [1] i32 { ret tensor [1] i32 [0]; }\n",
  inputs=b"",
):
  """A `TensorCase` as `tensor_case.load_case` returns it.

  The loader's dataclass is the replay boundary. These tests do not exercise
  the loader's manifest cross-checks, so they build its value directly instead
  of describing a source through a real `symiri`.
  """
  directory = Path(root) / "case"
  directory.mkdir(parents=True)
  (directory / "program.sir").write_bytes(program)
  (directory / "inputs.bin").write_bytes(inputs)
  (directory / "expected.bin").write_bytes(expected)
  manifest = {
    "profile": profile,
    "entry": entry,
    "comparison": {"mode": mode},
    "reference": {
      "precisions": [128, 256] if mode == "numeric" else [],
      "stable": True,
    },
  }
  return tensor_case.TensorCase(
    directory, manifest, program, inputs, expected, {"entry": entry}
  )


class TensorReferenceReplayTests(unittest.TestCase):
  def fake(self, root, *, body="", stdout="", stderr="", exit_code=0):
    script = Path(root) / "fake-symiri"
    lines = [_FAKE_HEADER]
    if body:
      lines.append(body)
    if stdout:
      lines.append(f"printf '%s\\n' {shlex.quote(stdout)}\n")
    if stderr:
      lines.append(f"printf '%s\\n' {shlex.quote(stderr)} >&2\n")
    lines.append(f"exit {exit_code}\n")
    script.write_text("".join(lines), encoding="utf-8")
    script.chmod(0o755)
    return script

  def bytes_body(self, root, by_bits):
    """Shell that copies one fixture per reference precision to the output."""
    lines = []
    for bits, data in by_bits.items():
      name = "exact" if bits is None else str(bits)
      path = Path(root) / f"bytes-{name}.bin"
      path.write_bytes(data)
      condition = '[ -z "$bits" ]' if bits is None else f'[ "$bits" = "{bits}" ]'
      lines.append(f'if {condition}; then cp {shlex.quote(str(path))} "$out"; fi\n')
    return "".join(lines)

  def test_derives_profile_and_entry_from_the_case(self):
    with tempfile.TemporaryDirectory() as root:
      expected = b"\x00\x00\x80\x3f"
      case = _case(root, profile="tensor-v3", entry="scale", expected=expected)
      body = (
        '[ "$profile" = "v3" ] || { echo "profile=$profile" >&2; exit 71; }\n'
        '[ "$entry" = "scale" ] || { echo "entry=$entry" >&2; exit 72; }\n'
      ) + self.bytes_body(root, {None: expected})
      result = tensor_reference.replay(
        case,
        directory=Path(root) / "out",
        symiri=self.fake(root, body=body),
      )

      self.assertEqual(result.classification, "reference_ok", result.reason)
      command = result.replays[0].command
      self.assertEqual(
        command[command.index("--tensor-profile-version") + 1],
        "v3",
        "the replay must pass the profile the case records, not a fixed one",
      )
      self.assertEqual(
        command[command.index("--entry") + 1],
        "scale",
        "the replay must pass the entry the case records, not a fixed one",
      )

  def test_exact_case_replays_once_and_matches_expected(self):
    with tempfile.TemporaryDirectory() as root:
      expected = b"\x01\x02\x03\x04"
      case = _case(root, mode="exact", expected=expected)
      result = tensor_reference.replay(
        case,
        directory=Path(root) / "out",
        symiri=self.fake(
          root,
          body=self.bytes_body(root, {None: expected}),
          stdout="replayed",
          stderr="reference note",
        ),
      )

      self.assertEqual(result.classification, "reference_ok", result.reason)
      self.assertEqual(len(result.replays), 1)
      replay = result.replays[0]
      self.assertIsNone(replay.reference_bits)
      self.assertNotIn("--tensor-reference-bits", replay.command)
      self.assertEqual(replay.exit_code, 0)
      self.assertEqual(replay.stdout, "replayed\n")
      self.assertEqual(replay.stderr, "reference note\n")
      self.assertEqual(replay.output, expected)
      self.assertEqual(replay.sha256, hashlib.sha256(expected).hexdigest())
      self.assertEqual(result.reference, expected)
      record = result.record()["replays"][0]
      self.assertEqual(record["output_bytes"], len(expected))
      self.assertEqual(record["output_sha256"], replay.sha256)

  def test_numerical_case_replays_at_both_precisions(self):
    with tempfile.TemporaryDirectory() as root:
      expected = b"\x0a\x0b\x0c\x0d"
      case = _case(root, mode="numeric", expected=expected)
      result = tensor_reference.replay(
        case,
        directory=Path(root) / "stable",
        symiri=self.fake(
          root, body=self.bytes_body(root, {128: expected, 256: expected})
        ),
      )
      self.assertEqual([replay.reference_bits for replay in result.replays], [128, 256])
      for replay in result.replays:
        command = replay.command
        self.assertEqual(
          command[command.index("--tensor-reference-bits") + 1],
          str(replay.reference_bits),
        )
      self.assertEqual(result.classification, "reference_ok", result.reason)
      self.assertEqual(result.reference, expected)

    with tempfile.TemporaryDirectory() as root:
      expected = b"\x0a\x0b\x0c\x0d"
      case = _case(root, mode="numeric", expected=expected)
      result = tensor_reference.replay(
        case,
        directory=Path(root) / "unstable",
        symiri=self.fake(
          root,
          body=self.bytes_body(
            root, {128: b"\x0a\x0b\x0c\x0d", 256: b"\x0a\x0b\x0c\x0e"}
          ),
        ),
      )
      self.assertEqual(len(result.replays), 2)
      self.assertEqual(result.classification, "reference_unstable", result.reason)
      self.assertIsNone(result.reference)

  def test_bytes_differing_from_expected_are_a_mismatch(self):
    with tempfile.TemporaryDirectory() as root:
      expected = b"good"
      case = _case(root, mode="exact", expected=expected)
      result = tensor_reference.replay(
        case,
        directory=Path(root) / "out",
        symiri=self.fake(root, body=self.bytes_body(root, {None: b"bad!"})),
      )

      self.assertEqual(result.classification, "reference_mismatch", result.reason)
      self.assertIn("expected.bin", result.reason)
      self.assertIsNone(result.reference)
      self.assertEqual(result.replays[0].output, b"bad!")

  def test_undefined_behavior_and_unrunnable_interpreters(self):
    for exit_code, stderr, signal in (
      (5, "", "the interpreter's undefined-behaviour exit code"),
      (1, "UB: Reading undef tensor element", "a UB diagnostic line"),
    ):
      with self.subTest(signal=signal), tempfile.TemporaryDirectory() as root:
        case = _case(root)
        result = tensor_reference.replay(
          case,
          directory=Path(root) / "ub",
          symiri=self.fake(root, stderr=stderr, exit_code=exit_code),
        )
        self.assertEqual(result.classification, "sir_ub", result.reason)
        self.assertIsNone(result.reference)

    with tempfile.TemporaryDirectory() as root:
      case = _case(root)
      result = tensor_reference.replay(
        case,
        directory=Path(root) / "missing",
        symiri=Path(root) / "absent-symiri",
      )
      self.assertEqual(result.classification, "reference_unavailable", result.reason)
      self.assertIsNone(result.replays[0].exit_code)
      self.assertIn("absent-symiri", result.reason)

    with tempfile.TemporaryDirectory() as root:
      case = _case(root, profile="tensor-v9")
      result = tensor_reference.replay(
        case,
        directory=Path(root) / "unsupported",
        symiri=Path(root) / "unused-symiri",
      )
      self.assertEqual(result.classification, "reference_unavailable", result.reason)
      self.assertIn("unsupported", result.reason)
      self.assertEqual(result.replays, ())

  def test_numerical_case_with_one_precision_missing_is_unavailable(self):
    """A missing precision is unavailable, not unstable.

    Instability is an observed disagreement between two outputs. A replay that
    never completed has no output to disagree with, and the case declares a
    dual-precision reference, so the reference cannot be established at all.
    """
    with tempfile.TemporaryDirectory() as root:
      expected = b"\x0a\x0b\x0c\x0d"
      case = _case(root, mode="numeric", expected=expected)
      body = 'if [ "$bits" = "256" ]; then exec sleep 3; fi\n'
      body += self.bytes_body(root, {128: expected})
      result = tensor_reference.replay(
        case,
        directory=Path(root) / "out",
        symiri=self.fake(root, body=body),
        timeout=1.0,
      )

      self.assertEqual(result.classification, "reference_unavailable", result.reason)
      self.assertIn("256", result.reason)
      self.assertTrue(result.replays[1].timed_out)
      self.assertIsNone(result.replays[1].output)
      self.assertIsNone(result.reference)


if __name__ == "__main__":
  unittest.main()
