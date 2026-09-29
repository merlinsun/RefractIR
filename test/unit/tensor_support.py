"""Source and subprocess helpers for proposed tensor frontend tests."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYMIRI = Path(os.environ.get("REFRACTIR_SYMIRI", ROOT / "symiri")).resolve()
SYMIRC = Path(os.environ.get("REFRACTIR_SYMIRC", ROOT / "symirc")).resolve()
SYMIR_SOLVE = Path(
  os.environ.get("REFRACTIR_SYMIR_SOLVE", ROOT / "symirsolve")
).resolve()
EMIT_API = Path(
  os.environ.get("REFRACTIR_EMIT_API", ROOT / "build" / "bin" / "tensor_emit_api")
).resolve()


def sir(params, result_type, declarations, statements, result="%out"):
  return (
    f"fun @kernel({params}) : {result_type} {{\n"
    + declarations
    + "\n^entry:\n"
    + statements
    + f"\n  ret {result};\n}}\n"
  )


def check_source(source, compiler=SYMIRI, *, tensor_profile=None):
  with tempfile.TemporaryDirectory() as directory:
    source_path = Path(directory) / "case.sir"
    source_path.write_text(source, encoding="utf-8")
    command = [str(compiler), "--check"]
    if tensor_profile is not None:
      command.extend(["--tensor-profile-version", tensor_profile])
    command.append(str(source_path))
    return subprocess.run(
      command,
      capture_output=True,
      text=True,
      check=False,
      timeout=10,
    )


def compile_source(source, target, output_path, compiler=SYMIRC):
  with tempfile.TemporaryDirectory() as directory:
    source_path = Path(directory) / "case.sir"
    source_path.write_text(source, encoding="utf-8")
    return subprocess.run(
      [str(compiler), "--target", target, "-o", str(output_path), str(source_path)],
      capture_output=True,
      text=True,
      check=False,
      timeout=10,
    )


def compile_via_api(source, target, output_path, emitter=EMIT_API):
  with tempfile.TemporaryDirectory() as directory:
    source_path = Path(directory) / "case.sir"
    source_path.write_text(source, encoding="utf-8")
    return subprocess.run(
      [str(emitter), target, str(source_path), str(output_path)],
      capture_output=True,
      text=True,
      check=False,
      timeout=10,
    )


def run_typed(
  source,
  input_bytes,
  *,
  input_path=None,
  entry="kernel",
  reference_bits=None,
  reference_stability=False,
  tensor_profile=None,
):
  """Run the typed tensor CLI and return process output plus result bytes."""
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source_path = root / "case.sir"
    source_path.write_text(source, encoding="utf-8")
    if input_path is None:
      input_file = root / "inputs.bin"
      input_file.write_bytes(input_bytes)
    else:
      input_file = Path(input_path)
    output_file = root / "result.bin"
    command = [str(SYMIRI)]
    if tensor_profile is not None:
      command.extend(["--tensor-profile-version", tensor_profile])
    if reference_stability:
      command.append("--tensor-reference-stability")
    elif reference_bits is not None:
      command.extend(["--tensor-reference-bits", str(reference_bits)])
    command.extend(
      [
        "--entry",
        entry,
        "--tensor-input",
        str(input_file),
        "--tensor-output",
        str(output_file),
        str(source_path),
      ]
    )
    result = subprocess.run(
      command,
      capture_output=True,
      text=True,
      check=False,
      timeout=10,
    )
    output = output_file.read_bytes() if output_file.exists() else None
    return result, output


def describe_entry(source, entry="kernel", compiler=SYMIRI):
  with tempfile.TemporaryDirectory() as directory:
    source_path = Path(directory) / "case.sir"
    source_path.write_text(source, encoding="utf-8")
    result = subprocess.run(
      [str(compiler), "--entry", entry, "--tensor-describe", str(source_path)],
      capture_output=True,
      text=True,
      check=False,
      timeout=10,
    )
    try:
      description = json.loads(result.stdout)
    except json.JSONDecodeError:
      description = None
    return result, description


def solve_source(source, *, entry="kernel", solver=SYMIR_SOLVE):
  """Solve one complete straight-line tensor function and return its artifacts."""
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source_path = root / "case.sir"
    output_path = root / "solved.sir"
    model_path = root / "model.json"
    source_path.write_text(source, encoding="utf-8")
    result = subprocess.run(
      [
        str(solver),
        "--main",
        f"@{entry}" if not entry.startswith("@") else entry,
        "--path",
        "^entry",
        "--output",
        str(output_path),
        "--emit-model",
        str(model_path),
        str(source_path),
      ],
      capture_output=True,
      text=True,
      check=False,
      timeout=20,
    )
    output = output_path.read_text(encoding="utf-8") if output_path.exists() else None
    try:
      model = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      model = None
    return result, output, model
