"""Build source, compiler, and input-behavior coverage records for tensor cases."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

_TOKEN = re.compile(
  r"(?P<space>\s+)|(?P<comment>//[^\n]*)|"
  r'(?P<string>"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|'
  r"(?P<sigil>[@%^](?:\?[A-Za-z_][\w.$-]*|[A-Za-z_][\w.$-]*|\d+))|"
  r"(?P<number>0[xX][0-9a-fA-F]+|(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+)|"
  r"(?P<operator>>>>|>>|<<|<=|>=|==|!=|&&|\|\||->|::|[+*/%<>=!&|^~-])|"
  r"(?P<punct>[{}()\[\],.:;])|(?P<word>[A-Za-z_][\w.$-]*)|(?P<other>.)"
)
_SIR_TOKEN = re.compile(
  r"(?P<space>\s+)|(?P<line_comment>//[^\n]*)|(?P<block_comment>/\*.*?\*/)|"
  r'(?P<string>"(?:\\.|[^"\\])*")|'
  r"(?P<sigil>@\?[A-Za-z_][A-Za-z0-9_]*|%\?[A-Za-z_][A-Za-z0-9_]*|"
  r"@[A-Za-z_][A-Za-z0-9_]*|%[A-Za-z_][A-Za-z0-9_]*|\^[A-Za-z_][A-Za-z0-9_]*)|"
  r"(?P<number>-?(?:0[xX][0-9a-fA-F]+|0[oO][0-7]+|0[bB][01]+|"
  r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?))|"
  r"(?P<operator>>>>|>>|<<|<=|>=|==|!=|[+*/%<>=!&|^~-])|"
  r"(?P<punct>[{}()\[\],.:;])|(?P<word>[A-Za-z_][A-Za-z0-9_]*)|(?P<other>.)",
  re.DOTALL,
)
_SIR_KEYWORDS = {
  "struct": 38,
  "fun": 39,
  "sym": 40,
  "let": 41,
  "mut": 42,
  "assume": 43,
  "require": 44,
  "br": 45,
  "ret": 46,
  "unreachable": 47,
  "in": 48,
  "select": 49,
  "undef": 52,
  "as": 53,
  "ptr": 54,
  "addr": 55,
  "load": 56,
  "store": 57,
  "null": 58,
  "cmp": 59,
  "ptrindex": 60,
  "ptrfield": 61,
  "call": 62,
  "decl": 63,
  "intrinsic": 64,
  "pre": 65,
  "post": 66,
}
_SIR_OPERATORS = {
  "{": 9,
  "}": 10,
  "(": 11,
  ")": 12,
  "[": 13,
  "]": 14,
  ":": 15,
  ";": 16,
  ",": 17,
  ".": 18,
  "+": 19,
  "-": 20,
  "*": 21,
  "/": 22,
  "%": 23,
  "&": 24,
  "|": 25,
  "^": 26,
  "~": 27,
  "<<": 28,
  ">>": 29,
  ">>>": 30,
  "=": 31,
  "==": 32,
  "!=": 33,
  "<": 34,
  "<=": 35,
  ">": 36,
  ">=": 37,
}
_DEBUG_LINE = re.compile(
  r"(?m)^\s*(?:!llvm\.(?:dbg|ident)|#loc\d*\s*=|debug_scope|\.file\b|\.loc\b).*?(?:\n|$)"
)
_LOCATION = re.compile(r"\bloc\([^)]*\)")
_DEBUG_ATTACHMENT = re.compile(r",?\s*!dbg\s+![\w.-]+")
_ARTIFACT_STAGE = {
  "ttir": "ttir",
  "ttgir": "ttgir",
  "llir": "llvm",
  "llvm": "llvm",
  "ptx": "ptx",
}


def _canonical_tokens(text):
  identifiers = {}
  tokens = []
  for match in _TOKEN.finditer(text):
    kind = match.lastgroup
    value = match.group()
    if kind in {"space", "comment"}:
      continue
    if kind == "sigil":
      value = identifiers.setdefault(value[0], {}).setdefault(
        value, f"{value[0]}{len(identifiers.setdefault(value[0], {}))}"
      )
    tokens.append(value)
  return " ".join(tokens)


def _digest(value):
  return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_signature(source):
  """Hash lexer-compatible source tokens with identifiers alpha-renamed."""
  if not isinstance(source, str):
    raise TypeError("source must be text")
  names = {}
  next_name = {}
  canonical = "sir-source-v3\n"
  for match in _SIR_TOKEN.finditer(source):
    kind = match.lastgroup
    if kind in {"space", "line_comment", "block_comment"}:
      continue
    value = match.group()
    if kind == "sigil":
      if value.startswith("^"):
        token_kind = 4
      elif value.startswith("@?") or value.startswith("%?"):
        token_kind = 3
      else:
        token_kind = 1 if value.startswith("@") else 2
      scope = names.setdefault(token_kind, {})
      if value not in scope:
        scope[value] = next_name.get(token_kind, 0)
        next_name[token_kind] = scope[value] + 1
      value = f"id{scope[value]}"
    elif kind == "string":
      raw = value[1:-1]
      decoded = []
      index = 0
      escapes = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}
      while index < len(raw):
        if raw[index] == "\\" and index + 1 < len(raw):
          index += 1
          decoded.append(escapes.get(raw[index], raw[index]))
        else:
          decoded.append(raw[index])
        index += 1
      value = "".join(decoded)
      token_kind = 8
    elif kind == "number":
      token_kind = (
        6
        if value.lower().startswith(("0x", "-0x", "0o", "-0o", "0b", "-0b"))
        else (7 if "." in value or "e" in value.lower() else 6)
      )
    elif kind == "word":
      token_kind = _SIR_KEYWORDS.get(value)
      if token_kind is None:
        if len(value) >= 2 and value[0] == "i" and value[1:].isdigit():
          token_kind = 50
        elif value in {"f32", "f64"}:
          token_kind = 51
        else:
          token_kind = 5
    else:
      token_kind = _SIR_OPERATORS.get(value)
      if token_kind is None:
        raise ValueError(f"invalid RefractIR token: {value!r}")
    byte_length = len(value.encode("utf-8"))
    canonical += f"{token_kind}:{byte_length}:{value}\n"
  return _digest(canonical)


def normalize_triton_ir(stage, text):
  """Remove unstable names and debug locations while retaining code attributes."""
  if stage not in {"ttir", "ttgir", "llvm", "ptx"}:
    raise ValueError(f"unknown compiler IR stage: {stage}")
  if not isinstance(text, str):
    raise TypeError("compiler IR must be text")
  source = _DEBUG_LINE.sub("", text)
  source = _LOCATION.sub("", source)
  source = _DEBUG_ATTACHMENT.sub("", source)
  return _canonical_tokens(source)


def compiled_signature(stage, text, *, context=None):
  """Return a stage-specific digest of normalized Triton compiler output."""
  normalized = normalize_triton_ir(stage, text)
  canonical_context = json.dumps(context or {}, sort_keys=True, separators=(",", ":"))
  return _digest(f"triton-{stage}-v1\n{canonical_context}\n{normalized}")


def _ttgir_loop_coverage(text):
  """Require optimized loop IR, tensor-carried state, and relevant work."""
  loop_tokens = list(re.finditer(r"\bscf\.(?:for|while)\b", text))
  loop_headers = re.finditer(
    r"\bscf\.(?:for|while)\b.*?->\s*\([^)]*\)\s*:\s*i\d+\s*\{",
    text,
    re.DOTALL,
  )
  tensor_carried_loop = False
  loop_body_has_work = False
  for header in loop_headers:
    header_text = header.group()
    if "iter_args" not in header_text or not re.search(
      r"->\s*\(\s*tensor<", header_text
    ):
      continue
    tensor_carried_loop = True
    depth = 1
    end = header.end()
    while end < len(text) and depth:
      if text[end] == "{":
        depth += 1
      elif text[end] == "}":
        depth -= 1
      end += 1
    body = text[header.end() : end - 1]
    has_update = re.search(
      r"\b(?:arith\.(?:add|sub|mul)[if]|tt\.(?:reduce|gather|load))\b", body
    )
    if has_update and "scf.yield" in body:
      loop_body_has_work = True
  reduction_count = len(re.findall(r"(?:\btt\.reduce\b|\"tt\.reduce\")", text))
  return {
    "loop_count": len(loop_tokens),
    "tensor_carried_loop": tensor_carried_loop,
    "reduction_count": reduction_count,
    "loop_body_has_work": loop_body_has_work,
    "covered": tensor_carried_loop and loop_body_has_work and reduction_count > 0,
  }


def _sha256_file(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def case_coverage(case_directory, *, unavailable_stages=None):
  """Record source identity separately from generation and input evidence."""
  case_path = Path(case_directory)
  manifest = json.loads((case_path / "case.json").read_text(encoding="utf-8"))
  source_path = case_path / "program.sir"
  source = source_path.read_text(encoding="utf-8")
  inputs_path = case_path / "inputs.bin"
  generator = manifest.get("generator", {})
  evidence = manifest.get("evidence", {})
  comparison = manifest.get("comparison", {})
  property_record = manifest.get("property", {})
  coverage = manifest.get("coverage", {})
  if not isinstance(coverage, dict):
    coverage = {}
  environment = manifest.get("environment")
  compiler_context = {
    "grid": manifest.get("grid"),
    "launch": manifest.get("launch"),
    "target": environment.get("target_arch") if isinstance(environment, dict) else None,
  }
  raw_artifacts = {}
  compiled = {}
  run_raw_artifacts = {}
  compiled_runs = {}
  compiled_loop_coverage = {}
  for path in sorted(case_path.rglob("compiler-*.ir")):
    raw_stage = path.stem.removeprefix("compiler-")
    stage = _ARTIFACT_STAGE.get(raw_stage)
    if stage is None:
      continue
    raw_text = path.read_text(encoding="utf-8")
    if path.parent == case_path:
      raw_artifacts[stage] = _sha256_file(path)
      compiled[stage] = compiled_signature(stage, raw_text, context=compiler_context)
      if stage == "ttgir":
        compiled_loop_coverage["case-artifacts"] = _ttgir_loop_coverage(raw_text)
      continue

    run_id = path.parent.relative_to(case_path).as_posix()
    context = dict(compiler_context)
    run_metadata = {}
    try:
      run_metadata = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      pass
    try:
      run_environment = json.loads(
        (path.parent / "environment.json").read_text(encoding="utf-8")
      )
    except (OSError, json.JSONDecodeError):
      run_environment = {}
    context.update(
      {
        "launch": run_metadata.get("launch", context["launch"]),
        "grid": run_metadata.get("grid", context["grid"]),
        "target": run_environment.get("target_arch", context["target"]),
        "triton_source_revision": run_environment.get("triton_source_revision"),
        "compiler": run_metadata.get("compiler", {}).get("sha256"),
      }
    )
    run_raw_artifacts.setdefault(run_id, {})[stage] = _sha256_file(path)
    compiled_runs.setdefault(run_id, {})[stage] = compiled_signature(
      stage, raw_text, context=context
    )
    if stage == "ttgir":
      compiled_loop_coverage[run_id] = _ttgir_loop_coverage(raw_text)

  primary_stratum = generator.get(
    "primary_stratum",
    coverage.get(
      "primary_stratum", manifest.get("primary_stratum", "legacy-unclassified")
    ),
  )
  exactness_evidence = generator.get(
    "exactness_evidence",
    manifest.get("exactness_evidence", coverage.get("exactness_evidence")),
  )
  execution = manifest.get("execution", {})
  loop_bounds = execution.get("loop_bounds", []) if isinstance(execution, dict) else []
  source_loop_present = bool(loop_bounds)

  return {
    "case_id": case_path.name,
    "source_signature": source_signature(source),
    "compiled_signatures": compiled,
    "compiled_runs": compiled_runs,
    "raw_artifact_sha256": raw_artifacts,
    "run_raw_artifact_sha256": run_raw_artifacts,
    "unavailable_stages": dict(unavailable_stages or {}),
    "execution_coverage": {
      "source_loop_present": source_loop_present,
      "loop_bounds": loop_bounds,
      "total_body_executions": execution.get("total_body_executions", 0)
      if isinstance(execution, dict)
      else 0,
      "compiled_loop_coverage": compiled_loop_coverage,
    },
    "input_behavior": {
      "inputs_sha256": _sha256_file(inputs_path) if inputs_path.is_file() else None,
      "primary_stratum": primary_stratum,
      "exactness_evidence": exactness_evidence,
      "comparison_mode": comparison.get("mode"),
      "property_id": property_record.get("id"),
      "evidence_kind": evidence.get("kind"),
      "evidence_status": evidence.get("status"),
      **coverage.get("input_behavior", {}),
    },
  }


def index_cases(case_directories):
  """Retain all cases and report structure counts without deduplicating inputs."""
  cases = [case_coverage(path) for path in case_directories]
  source_signatures = {case["source_signature"] for case in cases}
  compiled_by_stage = {}
  for case in cases:
    signatures = [case["compiled_signatures"]]
    signatures.extend(case["compiled_runs"].values())
    for run_signatures in signatures:
      for stage, signature in run_signatures.items():
        compiled_by_stage.setdefault(stage, set()).add(signature)
  source_loop_cases = sum(
    case["execution_coverage"]["source_loop_present"] for case in cases
  )
  compiled_loop_cases = sum(
    any(
      loop_record.get("covered") is True
      for loop_record in case["execution_coverage"]["compiled_loop_coverage"].values()
    )
    for case in cases
  )
  return {
    "schema": "refractir.tensor-coverage.v1",
    "case_count": len(cases),
    "unique_source_signatures": len(source_signatures),
    "unique_compiled_signatures": {
      stage: len(signatures) for stage, signatures in sorted(compiled_by_stage.items())
    },
    "source_loop_cases": source_loop_cases,
    "compiled_loop_cases": compiled_loop_cases,
    "cases": cases,
  }


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--suite", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args(argv)
  suite = args.suite.resolve()
  cases = sorted(path.parent for path in suite.rglob("case.json"))
  report = index_cases(cases)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
    f"indexed {report['case_count']} cases ({report['unique_source_signatures']} source structures)"
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
