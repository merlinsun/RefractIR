"""Replay the Triton #11406 softmax regression with pinned source overlays."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

from test.lib import triton_case_worker
from test.lib.run_triton_tests import compare_outputs

_REVISIONS = {
  "affected": "f4fca2c0bc0e19930d2be76bb06ee7b232b60e86",
  "fixed": "31b1bf9aed2714af9a816d023b405499e8b27883",
}
ROOT = Path(__file__).resolve().parents[2]


def _canonical_json(value):
  return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _source_at_revision(triton_root, revision):
  result = subprocess.run(
    [
      "git",
      "-C",
      str(triton_root),
      "show",
      f"{revision}:python/triton/language/standard.py",
    ],
    capture_output=True,
    check=False,
  )
  if result.returncode:
    raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
  return result.stdout


def _install_softmax_overlay(source, revision, target):
  import triton.language as tl
  import triton.language.standard as standard

  decoded = source.decode("utf-8")
  module_ast = ast.parse(decoded, filename=f"{revision}/standard.py")
  function = next(
    node
    for node in module_ast.body
    if isinstance(node, ast.FunctionDef) and node.name == "softmax"
  )
  lines = decoded.splitlines(keepends=True)
  first = min(decorator.lineno for decorator in function.decorator_list)
  function_source = "".join(lines[first - 1 : function.end_lineno])
  module_source = (
    "import triton\n"
    "import triton.language as tl\n"
    "from triton.language import core, math\n"
    "jit = triton.jit\n"
    "max = tl.max\n"
    "sum = tl.sum\n\n" + function_source
  )
  module_path = target / "legacy_standard.py"
  module_path.write_text(module_source, encoding="utf-8")
  module_name = f"triton_softmax_{revision[:12]}"
  spec = importlib.util.spec_from_file_location(module_name, module_path)
  if spec is None or spec.loader is None:
    raise RuntimeError("cannot create the pinned softmax source module")
  overlay_module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = overlay_module
  spec.loader.exec_module(overlay_module)
  standard.softmax = overlay_module.softmax
  tl.softmax = overlay_module.softmax
  return module_path


def _variant_call(source, variant):
  pattern = r"tl\.softmax\(([^,]+),\s*dim=1\)"
  match = re.search(pattern, source)
  if not match:
    raise ValueError("historical fixture kernel must call softmax on axis one")
  if variant == "affected":
    replacement = f"tl.softmax({match.group(1)}, dim=1, keep_dims=False)"
    note = "explicit legacy keep_dims=False invocation"
  else:
    replacement = f"tl.softmax({match.group(1)}, dim=1)"
    note = "omit removed keep_dims keyword; operation and tolerance are unchanged"
  mutated = source[: match.start()] + replacement + source[match.end() :]
  return mutated, note


def _validate_source_revision(source, revision, variant):
  module = ast.parse(source.decode("utf-8"), filename=f"{revision}/standard.py")
  functions = [
    node
    for node in module.body
    if isinstance(node, ast.FunctionDef) and node.name == "softmax"
  ]
  if len(functions) != 1:
    raise RuntimeError("pinned standard.py must contain exactly one softmax function")
  has_legacy_keyword = any(
    argument.arg == "keep_dims" for argument in functions[0].args.args
  )
  if has_legacy_keyword != (variant == "affected"):
    raise RuntimeError("pinned softmax signature does not match the selected revision")


def replay(case_path, triton_root, output_root, variant):
  if variant not in _REVISIONS:
    raise ValueError("variant must be affected or fixed")
  case_path = Path(case_path).resolve()
  triton_root = Path(triton_root).resolve()
  target = Path(output_root).resolve() / variant
  target.mkdir(parents=True, exist_ok=False)
  for name in ("case.json", "program.sir", "inputs.bin", "expected.bin", "kernel.py"):
    (target / name).write_bytes((case_path / name).read_bytes())

  manifest_path = target / "case.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  kernel_path = target / "kernel.py"
  kernel, adapter = _variant_call(kernel_path.read_text(encoding="utf-8"), variant)
  kernel_path.write_text(kernel, encoding="utf-8")

  revision = _REVISIONS[variant]
  standard_source = _source_at_revision(triton_root, revision)
  _validate_source_revision(standard_source, revision, variant)
  overlay_path = None
  if variant == "affected":
    overlay_path = _install_softmax_overlay(standard_source, revision, target)
  standard_sha256 = hashlib.sha256(standard_source).hexdigest()

  base_environment_path = case_path / "environment.json"
  environment = json.loads(base_environment_path.read_text(encoding="utf-8"))
  relative_lock = os.path.relpath(
    ROOT / "test/tensor/environments/v100-triton-3.8.0/package-lock.json", target
  )
  environment.update(
    {
      "package_lock_file": relative_lock,
      "runtime_triton_source_revision": environment.get("triton_source_revision"),
      "softmax_source_revision": revision,
      "softmax_source_sha256": standard_sha256,
      "softmax_overlay": (
        "exact softmax function extracted into an isolated namespace; other modules use the locked runtime"
        if variant == "affected"
        else "corrected softmax runs in the locked post-fix Triton runtime"
      ),
      "softmax_overlay_file": overlay_path.name if overlay_path else None,
      "invocation_adapter": adapter,
    }
  )
  (target / "environment.json").write_text(
    _canonical_json(environment), encoding="utf-8"
  )
  manifest["environment"] = {
    "file": "environment.json",
    "package_lock": relative_lock,
    "package_lock_sha256": environment["package_lock_sha256"],
  }
  manifest["historical_variant"] = variant
  manifest_path.write_text(_canonical_json(manifest), encoding="utf-8")

  inputs = (target / "inputs.bin").read_bytes()
  expected = (target / "expected.bin").read_bytes()
  (target / "run.json").write_text(
    _canonical_json(
      {
        "schema": "refractir.tensor-historical-run.v1",
        "mode": "compiled",
        "variant": variant,
        "softmax_source_revision": revision,
        "softmax_source_sha256": standard_sha256,
        "invocation_adapter": adapter,
        "source_sha256": hashlib.sha256(
          (target / "program.sir").read_bytes()
        ).hexdigest(),
        "inputs_sha256": hashlib.sha256(inputs).hexdigest(),
        "expected_sha256": hashlib.sha256(expected).hexdigest(),
        "kernel_sha256": hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
        "launch": manifest["launch"],
        "grid": manifest["grid"],
      }
    ),
    encoding="utf-8",
  )

  try:
    triton_case_worker.execute(target, "compiled")
    stdout = ""
    stderr = ""
  except Exception:
    stdout = ""
    stderr = traceback.format_exc()
  (target / "stdout.log").write_text(stdout, encoding="utf-8")
  (target / "stderr.log").write_text(stderr, encoding="utf-8")

  actual_path = target / "actual.bin"
  if not actual_path.is_file():
    try:
      phase = json.loads((target / "worker-phase.json").read_text()).get(
        "phase", "runtime"
      )
    except (OSError, json.JSONDecodeError):
      phase = "runtime"
    status = "compile_failure" if phase == "compilation" else "runtime_failure"
    comparison = None
    actual_sha256 = None
  else:
    actual = actual_path.read_bytes()
    output = manifest["output"]
    comparison = compare_outputs(
      actual,
      expected,
      output["dtype"],
      output["shape"],
      manifest["comparison"],
    )
    status = comparison["status"]
    actual_sha256 = hashlib.sha256(actual).hexdigest()
  result = {
    "variant": variant,
    "mode": "compiled",
    "status": status,
    "campaign_valid": bool(environment.get("campaign_available")),
    "softmax_source_revision": revision,
    "softmax_source_sha256": standard_sha256,
    "runtime_triton_source_revision": environment.get("runtime_triton_source_revision"),
    "invocation_adapter": adapter,
    "expected_sha256": hashlib.sha256(expected).hexdigest(),
    "actual_sha256": actual_sha256,
    "comparison": comparison,
    "artifacts": sorted(path.name for path in target.iterdir() if path.is_file()),
  }
  (target / "result.json").write_text(_canonical_json(result), encoding="utf-8")
  return result


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--case", required=True)
  parser.add_argument("--triton-root", required=True)
  parser.add_argument("--output-root", required=True)
  parser.add_argument("--variant", choices=tuple(_REVISIONS), required=True)
  args = parser.parse_args(argv)
  try:
    result = replay(args.case, args.triton_root, args.output_root, args.variant)
  except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
    print(f"error: {error}", file=sys.stderr)
    return 2
  print(_canonical_json(result), end="")
  expected_status = "numeric_mismatch" if args.variant == "affected" else "pass"
  return 0 if result["status"] == expected_status else 1


if __name__ == "__main__":
  sys.exit(main())
