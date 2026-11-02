"""Shared execution, staging, and compiler-emission helpers for tensor runners.

Every tensor runner crosses the same boundaries: it writes artifacts atomically,
launches a fresh interpreter or GPU process with an explicit environment, and
emits one ordinary Triton kernel through `symirc`. Keeping those mechanics here
means a second runner exercises the same implementation instead of a copy.
"""

import ast
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

class DispatchRefused(RuntimeError):
  """A resource refusal issued before a subprocess was spawned."""


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMEOUT_S = 60
EMIT_TIMEOUT_S = 10


def canonical_json(value):
  """One deterministic JSON document, indented and newline terminated."""
  return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def write_atomic(path, data):
  """Replace `path` with `data` so a reader sees the old or the new bytes."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
  try:
    with os.fdopen(descriptor, "wb") as output:
      output.write(data)
    os.replace(temporary, path)
  finally:
    if os.path.exists(temporary):
      os.unlink(temporary)


def run_child(
  command,
  mode,
  timeout_s=DEFAULT_TIMEOUT_S,
  *,
  cwd=ROOT,
  cache_dir=None,
  device_id=None,
  on_process=None,
):
  """Run one mode in a fresh process with an explicit interpreter setting."""
  if mode not in {"compiled", "interpret"}:
    raise ValueError("mode must be compiled or interpret")
  if device_id is not None and (type(device_id) is not int or device_id < 0):
    raise ValueError("device_id must be a nonnegative integer")
  if mode == "interpret" and device_id is not None:
    raise ValueError("device_id is available only in compiled mode")
  env = dict(os.environ)
  env.pop("TRITON_INTERPRET", None)
  env.pop("TRITON_CACHE_DIR", None)
  if mode == "interpret":
    env["TRITON_INTERPRET"] = "1"
  elif cache_dir is not None:
    cache_path = Path(cache_dir).resolve()
    if cache_path.exists():
      shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True)
    env["TRITON_CACHE_DIR"] = str(cache_path)
  if mode == "compiled" and device_id is not None:
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
  existing_pythonpath = env.get("PYTHONPATH")
  env["PYTHONPATH"] = str(ROOT) + (
    os.pathsep + existing_pythonpath if existing_pythonpath else ""
  )
  if on_process is not None:
    on_process({"event": "reserved", "mode": mode, "command": list(command)})
  try:
    process = subprocess.Popen(
      command,
      cwd=cwd,
      env=env,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
    )
  except OSError as error:
    if on_process is not None:
      on_process(
        {"event": "finish", "mode": mode, "status": "spawn_failed", "reason": str(error)}
      )
    raise
  try:
    if on_process is not None:
      on_process({"event": "start", "mode": mode, "pid": process.pid})
    stdout, stderr = process.communicate(timeout=timeout_s)
  except subprocess.TimeoutExpired:
    process.kill()
    process.communicate()
    try:
      if on_process is not None:
        on_process({"event": "finish", "mode": mode, "status": "timed_out"})
    except Exception:
      pass
    raise
  except BaseException:
    if process.poll() is None:
      process.kill()
    try:
      process.communicate(timeout=5)
    except Exception:
      pass
    try:
      if on_process is not None:
        on_process({"event": "finish", "mode": mode, "status": "interrupted"})
    except Exception:
      pass
    raise
  completed = subprocess.CompletedProcess(
    command, process.returncode, stdout, stderr
  )
  if on_process is not None:
    on_process(
      {
        "event": "finish",
        "mode": mode,
        "status": "completed" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
      }
    )
  return completed


def gpu_available(device_id=0):
  """Whether a CUDA device with this index is visible to the process."""
  try:
    import torch

    return bool(torch.cuda.is_available() and device_id < torch.cuda.device_count())
  except Exception:
    return False


def layout_manifest(case):
  """The per-buffer layout manifest a v2/v3 worker stages for this case.

  Both the ordinary emitter and a frozen native kernel use the same staging, so
  both build the manifest here; the two calls agree byte for byte for one case.
  """
  buffers = []
  for role, records in (
    ("input", case.manifest["inputs"]),
    ("output", (case.manifest["output"],)),
  ):
    for record in records:
      if not record["shape"]:
        continue
      buffers.append(
        {
          "name": record["name"],
          "role": role,
          "element_strides": record["element_strides"],
          "element_offset": record["element_offset"],
          "storage_span": record["storage_span"],
          "axis_order": record["axis_order"],
        }
      )
  return {"buffers": buffers}


def emit_kernel(case, symirc, artifact_dir, extra_args=()):
  """Emit one ordinary Triton kernel and return its path with both logs.

  The command line is the ordinary `symirc` invocation for the case's profile:
  v2 and v3 carry the layout manifest, v4 carries only the profile version, and
  a bundle's `loop_bound` and `loop_hint` add `--tensor-loop-bound` and
  `--tensor-loop-hint`. v5 carries the layout manifest like v3, and
  `extra_args` (grid tiling, stride arguments, boundary dtypes, a launch record)
  are appended for a generation export. The
  kernel is written to a temporary name, parsed, then moved into place, so a
  failed emission leaves no kernel artifact. A failure returns a missing path
  with the compiler's stderr and stdout; callers turn that into a compile
  failure status.
  """
  artifact_dir.mkdir(parents=True, exist_ok=True)
  target = artifact_dir / "kernel.py"
  descriptor, temporary_name = tempfile.mkstemp(
    prefix=".kernel.", suffix=".py", dir=artifact_dir
  )
  os.close(descriptor)
  temporary = Path(temporary_name)
  temporary.unlink()
  command = [
    str(symirc),
    "--target",
    "triton",
    "--entry",
    case.manifest["entry"],
    "--tensor-softmax-lowering",
    case.manifest["softmax_lowering"],
    "-o",
    str(temporary),
    str(case.path / "program.sir"),
  ]
  command[-3:-3] = list(extra_args)
  if case.manifest["profile"] in {"tensor-v2", "tensor-v3", "tensor-v5"}:
    staged_layout = layout_manifest(case)
    layout_path = artifact_dir / "tensor-layout.json"
    layout_path.write_text(
      canonical_json(staged_layout) + "\n",
      encoding="utf-8",
    )
    version = {"tensor-v2": "v2", "tensor-v3": "v3", "tensor-v5": "v5"}[
      case.manifest["profile"]
    ]
    command[1:1] = ["--tensor-profile-version", version]
    command[1:1] = ["--tensor-layout-manifest", str(layout_path)]
  elif case.manifest["profile"] == "tensor-v4":
    command[1:1] = ["--tensor-profile-version", "v4"]
  if "loop_bound" in case.manifest:
    command[1:1] = ["--tensor-loop-bound", case.manifest["loop_bound"]]
  if "loop_hint" in case.manifest:
    command[1:1] = ["--tensor-loop-hint", case.manifest["loop_hint"]]
  try:
    result = subprocess.run(
      command,
      cwd=ROOT,
      capture_output=True,
      text=True,
      timeout=EMIT_TIMEOUT_S,
      check=False,
    )
  except (OSError, subprocess.TimeoutExpired) as error:
    return None, str(error), ""
  if result.returncode != 0:
    temporary.unlink(missing_ok=True)
    return None, result.stderr, result.stdout
  try:
    text = temporary.read_text(encoding="utf-8")
    ast.parse(text, filename=str(target))
    os.replace(temporary, target)
  except (OSError, SyntaxError, UnicodeError) as error:
    temporary.unlink(missing_ok=True)
    return None, str(error), result.stdout
  return target, result.stderr, result.stdout
