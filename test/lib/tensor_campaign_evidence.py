"""Copy and verify moving-window evidence without executing any recorded command.

The manifest inventories exact bytes. Process commands identify required raw
bundles, including rejected screening attempts; saved run hashes bind outputs
and kernels. An archive can move while the original records stay byte-exact.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

NAME = "evidence-manifest.json"
SCHEMA = "refractir.moving-window-evidence.v1"


def _sha(path):
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def inventory(work, external_roots=()):
  work = Path(work).resolve()
  files = []
  for path in sorted(work.rglob("*")):
    if path.is_symlink():
      raise ValueError(f"evidence symlink is not portable: {path}")
    if path.is_file() and path != work / NAME:
      files.append(
        {
          "path": path.relative_to(work).as_posix(),
          "size": path.stat().st_size,
          "sha256": _sha(path),
        }
      )
  mounts = [
    {"origin": str(Path(path).resolve()), "prefix": f"external/{index:03d}"}
    for index, path in enumerate(external_roots)
  ]
  if mounts and (work / "external").exists():
    raise ValueError("external is reserved for delivered control payloads")
  external = set()
  ledger = work / "path-ledger.jsonl"
  for line in ledger.read_text().splitlines():
    if not line.strip():
      continue
    event = json.loads(line)
    command = event.get("command") or []
    if event.get("kind") != "process" or not command:
      continue
    if "--case" in command:
      bundle = Path(command[command.index("--case") + 1])
      references = [
        bundle / name for name in ("program.sir", "inputs.bin", "case.json")
      ]
    elif "--tensor-input" in command:
      references = [
        Path(command[command.index("--tensor-input") + 1]),
        Path(command[-1]),
      ]
    else:
      continue
    external.update(
      path.resolve()
      for path in references
      if path.is_absolute() and not path.resolve().is_relative_to(work)
    )
  for path in sorted(external):
    mount = next(
      (item for item in mounts if path.is_relative_to(Path(item["origin"]))), None
    )
    if mount is None:
      raise ValueError(f"external evidence needs an explicit root: {path}")
    if not path.is_file():
      raise ValueError(f"missing external evidence: {path}")
    files.append(
      {
        "path": str(Path(mount["prefix"]) / path.relative_to(mount["origin"])),
        "size": path.stat().st_size,
        "sha256": _sha(path),
      }
    )
  return {
    "external_roots": mounts,
    "schema": SCHEMA,
    "study": json.loads((work / "study.json").read_text())["digest"],
    "origin": str(work),
    "files": files,
  }


def verify(work, *, manifest=None, study_digest=None):
  root = Path(work).resolve()
  record = manifest if manifest is not None else json.loads((root / NAME).read_text())
  if record.get("schema") != SCHEMA or not isinstance(record.get("files"), list):
    raise ValueError("unsupported evidence manifest")
  if study_digest is None:
    study_digest = json.loads((root / "study.json").read_text())["digest"]
  if record.get("study") != study_digest:
    raise ValueError("evidence belongs to another study")
  origin = Path(record.get("origin", ""))
  if not origin.is_absolute():
    raise ValueError("evidence origin must be absolute")
  mounts = record.get("external_roots", [])
  if not isinstance(mounts, list):
    raise ValueError("invalid external-root mapping")
  seen = set()
  for mount in mounts:
    prefix = Path(mount.get("prefix", ""))
    if (
      not Path(mount.get("origin", "")).is_absolute()
      or prefix.is_absolute()
      or ".." in prefix.parts
      or len(prefix.parts) != 2
      or prefix.parts[0] != "external"
      or str(prefix) in seen
    ):
      raise ValueError("invalid external-root mapping")
    seen.add(str(prefix))
  indexed = {}
  for item in record["files"]:
    name = item.get("path", "")
    relative = Path(name)
    if not name or relative.is_absolute() or ".." in relative.parts or name in indexed:
      raise ValueError("invalid or duplicate evidence path")
    path = root / relative
    if (
      path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file()
    ):
      raise ValueError(f"missing evidence: {name}")
    if path.stat().st_size != item.get("size") or _sha(path) != item.get("sha256"):
      raise ValueError(f"evidence digest changed: {name}")
    indexed[name] = item

  def relative(path):
    path = Path(path)
    if path.is_absolute():
      if path.is_relative_to(origin):
        path = path.relative_to(origin)
      elif path.resolve().is_relative_to(origin.resolve()):
        path = path.resolve().relative_to(origin.resolve())
      else:
        mapped = None
        for mount in mounts:
          external = Path(mount["origin"])
          # Resolve aliases only while packaging on the original host. On a
          # different host the lexical original prefix still maps unchanged.
          candidate = path if path.is_relative_to(external) else path.resolve()
          if candidate.is_relative_to(external):
            mapped = Path(mount["prefix"]) / candidate.relative_to(external)
            break
        if mapped is None:
          raise ValueError(f"raw artifact is outside the archived origin: {path}")
        path = mapped
    if ".." in path.parts:
      raise ValueError("raw artifact escapes the archive")
    return path.as_posix()

  def need(path, expected=None):
    name = relative(path)
    if name not in indexed:
      raise ValueError(f"required raw artifact is not delivered: {name}")
    if expected is not None and indexed[name]["sha256"] != expected:
      raise ValueError(f"saved digest differs from raw artifact: {name}")
    return root / name

  for name in [
    "study.json",
    "path-ledger.jsonl",
    "tranche-ledger.jsonl",
    "corpus-frozen.json",
  ]:
    need(name)
  events = [
    json.loads(line)
    for line in (root / "path-ledger.jsonl").read_text().splitlines()
    if line.strip()
  ]
  processes = {}
  for event in events:
    if event.get("kind") == "process":
      identity = event.get("execution_id")
      if identity is None:
        raise ValueError("raw verification needs identified process events")
      processes.setdefault(identity, []).append(event)
  for history in processes.values():
    if not any(event.get("event") == "start" for event in history):
      continue
    command = next(
      (event.get("command") for event in history if event.get("command")), None
    )
    if not command:
      raise ValueError("a started process has no recorded command")
    finished = next(
      (event for event in reversed(history) if event.get("event") == "finish"), None
    )

    def arg(flag):
      try:
        return Path(command[command.index(flag) + 1])
      except (ValueError, IndexError):
        raise ValueError(f"recorded process is missing {flag}") from None

    if history[0].get("path") == "p0":
      need(arg("--tensor-input"))
      need(command[-1])
      if finished and finished.get("status") == "completed":
        need(arg("--tensor-output"))
      continue
    bundle = arg("--case")
    artifacts = arg("--artifacts-dir")
    for name in ["program.sir", "inputs.bin", "case.json"]:
      need(bundle / name)
    need(artifacts / "kernel.py")
    if finished:
      need(artifacts / "worker.log")
      names = [
        artifacts.parent / "target-run.json",
        artifacts.parent / "interpreter-run.json",
      ]
      delivered = [name for name in names if relative(name) in indexed]
      if not delivered:
        raise ValueError(f"worker result record is missing: {artifacts}")
      result = json.loads(need(delivered[0]).read_text())
      first, second = result.get("interpreter") or {}, result.get("compiled") or {}
      if first.get("status") == second.get("status") == "completed":
        left = need(
          artifacts.parent / "interpret/actual.bin", first.get("actual_sha256")
        )
        right = need(
          artifacts.parent / "compiled/actual.bin", second.get("actual_sha256")
        )
        equal = left.read_bytes() == right.read_bytes()
        expected = (result.get("agreement") or {}).get("kind")
        if (expected == "identical-bytes" and not equal) or (
          expected == "differ" and equal
        ):
          raise ValueError("saved agreement differs from raw output comparison")
        if result.get("status") == "target_agreement" and not equal:
          raise ValueError("saved target agreement differs from raw outputs")
      mode = "compiled" if history[0].get("path") == "p2" else "interpreter"
      observation = result.get(mode) or {}
      emission = result.get("emission") or {}
      if emission.get("sir_sha256"):
        need(bundle / "program.sir", emission["sir_sha256"])
      if result.get("inputs_sha256"):
        need(bundle / "inputs.bin", result["inputs_sha256"])
      kernel = emission.get("kernel_sha256")
      if kernel:
        need(artifacts / "kernel.py", kernel)
      actual = observation.get("actual_sha256")
      if actual:
        need(artifacts / "actual.bin", actual)
      elif observation.get("status") == "completed":
        raise ValueError(
          f"completed output has no saved digest: {artifacts}/actual.bin"
        )

  # Every claimed compiler artifact must be present with its recorded bytes.
  # Identical stages may be shared by several runs; paths need not be rewritten.
  def compiler_hashes(value):
    if isinstance(value, dict):
      hashes = value.get("artifact_sha256")
      if isinstance(hashes, dict):
        for stage, digest in hashes.items():
          name = f"compiler-{stage}.ir"
          if not any(
            Path(path).name == name and item["sha256"] == digest
            for path, item in indexed.items()
          ):
            raise ValueError(f"claimed compiler artifact is not delivered: {name}")
      for child in value.values():
        compiler_hashes(child)
    elif isinstance(value, list):
      for child in value:
        compiler_hashes(child)

  compiler_hashes(events)
  if (root / "qualification.json").is_file():
    need("qualification.json")
    compiler_hashes(json.loads((root / "qualification.json").read_text()))
  return record


def package(work, destination, *, external_roots=()):
  work, destination = Path(work).resolve(), Path(destination).resolve()
  if destination.exists() or destination.is_relative_to(work):
    raise ValueError("evidence destination must be new and outside the source")
  manifest = inventory(work, external_roots)
  destination.mkdir(parents=True)
  for item in manifest["files"]:
    target = destination / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    source = work / item["path"]
    for mount in manifest["external_roots"]:
      if Path(item["path"]).is_relative_to(mount["prefix"]):
        source = Path(mount["origin"]) / Path(item["path"]).relative_to(mount["prefix"])
        break
    shutil.copy2(source, target)
  verify(destination, manifest=manifest)
  # This marker is written only after the copied bytes and required artifacts pass.
  (destination / NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  return manifest


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("mode", choices=("pack", "verify"))
  parser.add_argument("--work", required=True, type=Path)
  parser.add_argument("--destination", type=Path)
  parser.add_argument("--external-root", type=Path, action="append", default=[])
  args = parser.parse_args(argv)
  if args.mode == "pack":
    if args.destination is None:
      parser.error("pack requires --destination")
    result = package(args.work, args.destination, external_roots=args.external_root)
  else:
    result = verify(args.work)
  print(json.dumps({"study": result["study"], "files": len(result["files"])}))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
