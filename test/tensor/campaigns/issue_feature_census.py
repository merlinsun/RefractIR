"""Feature census of Triton issue reproducers (plan L0, rule D7).

Each seed reproducer is parsed with `ast` and tagged with the features it
uses. Its symptom is inferred locally from two signals kept side by side: the
manifest's family label, and whether the file checks a value. The label wins
when it names a symptom, because it summarises the issue; the file check
decides otherwise. A file `ast` cannot parse (an excerpt with ellipses) is
tagged by call names alone, so it never contributes control-flow features.

D7 ranks the five candidate features by wrong-result issues, ties broken by
density (wrong-result issues per file using the feature). A feature outside
the five with more wrong-result issues than the fifth-ranked one is added at
its rank; math functions are excluded from promotion.

Usage:
  python issue_feature_census.py SEED_DIR OUT_DIR
"""

import argparse
import ast
import json
import re
from pathlib import Path

CANDIDATES = ("gluon", "conversion", "dot", "scan", "nested_control")
# Math functions are tallied but never promoted: their lowerings are
# unspecified, so the comparison contract excludes them (plan Part IV).
OUTSIDE = ("where", "atomic", "descriptor", "warp_specialize")

# Calls of `tl.<name>` (or `gl.<name>`) and the feature each one marks.
_CALLS = {
  "dot": {"dot", "dot_scaled"},
  "conversion": {"cast"},
  "reduction": {"sum", "max", "min", "argmax", "argmin", "reduce", "xor_sum"},
  "shape": {
    "trans",
    "permute",
    "reshape",
    "view",
    "join",
    "split",
    "interleave",
    "expand_dims",
    "broadcast_to",
  },
  "math": {
    "exp",
    "exp2",
    "log",
    "log2",
    "sqrt",
    "rsqrt",
    "sin",
    "cos",
    "erf",
    "fdiv",
    "div_rn",
    "sqrt_rn",
    "fma",
    "sigmoid",
    "softmax",
  },
  "where": {"where"},
  "scan": {"cumsum", "cumprod", "associative_scan"},
  "atomic": {
    "atomic_add",
    "atomic_max",
    "atomic_min",
    "atomic_and",
    "atomic_or",
    "atomic_xor",
    "atomic_xchg",
    "atomic_cas",
  },
  "descriptor": {"make_tensor_descriptor", "load_tensor_descriptor"},
  "block_ptr": {"make_block_ptr"},
  "warp_specialize": {"warp_specialize"},
}
_DTYPES = {
  "fp16": {"float16"},
  "bf16": {"bfloat16"},
  "fp8": {
    "float8e4nv",
    "float8e5",
    "float8e4b15",
    "float8e4b8",
    "float8e5b16",
    "float8_e4m3fn",
    "float8_e5m2",
  },
}
_LANGUAGE = {"tl", "gl", "language", "libdevice", "math"}

_LABEL_WORDS = (
  ("hang", ("hang", "hangs", "timeout", "deadlock")),
  (
    "wrong_result",
    (
      "wrong",
      "incorrect",
      "mismatch",
      "nan",
      "precision",
      "rounding",
      "garbage",
      "result",
      "value",
    ),
  ),
  ("crash", ("assert", "segfault", "crash", "error", "compile", "verifier")),
  ("other", ("performance", "slow", "perf")),
)
_CHECK_CALLS = {"allclose", "assert_close", "assert_allclose", "equal", "isclose"}
_EXCERPT = re.compile(r"excerpt|fragment", re.IGNORECASE)


def _is_kernel(node):
  """A function decorated with any `jit` (triton.jit, gluon.jit, aliases)."""
  for decorator in node.decorator_list:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    name = (
      target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
    )
    if name.endswith("jit"):
      return True
  return False


def _loop_depth(node, depth=0):
  loops = (ast.For, ast.While)
  deepest = depth
  for child in ast.iter_child_nodes(node):
    inner = depth + 1 if isinstance(child, loops) else depth
    deepest = max(deepest, _loop_depth(child, inner))
  return deepest


def _kernel_features(kernel):
  """Control flow, and `.to` (in a kernel always a Triton conversion)."""
  found = set()
  if _loop_depth(kernel) >= 2:
    found.add("nested_loop")
  for node in ast.walk(kernel):
    if (
      isinstance(node, ast.Call)
      and isinstance(node.func, ast.Attribute)
      and node.func.attr == "to"
    ):
      found.add("conversion")
    elif isinstance(node, ast.While):
      found.add("while")
    elif isinstance(node, ast.If):
      found.add("if")
  return found


def _call_features(names):
  return {feature for feature, calls in _CALLS.items() if names & calls}


def _tree_features(tree):
  found, called = set(), set()
  for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_kernel(node):
      found |= _kernel_features(node)
      for decorator in node.decorator_list:
        if "gluon" in ast.unparse(decorator):
          found.add("gluon")
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
      if "gluon" in ast.unparse(node):
        found.add("gluon")
    elif isinstance(node, ast.Name) and node.id == "TensorDescriptor":
      found.add("descriptor")
    elif isinstance(node, ast.Attribute):
      for dtype, names in _DTYPES.items():
        if node.attr in names:
          found.add(dtype)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
      owner = node.func.value
      if isinstance(owner, ast.Name) and owner.id in _LANGUAGE:
        called.add(node.func.attr)
      if node.func.attr == "to" and any(
        isinstance(argument, ast.Attribute)
        and isinstance(argument.value, ast.Name)
        and argument.value.id in {"tl", "gl"}
        for argument in node.args
      ):
        found.add("conversion")
      if node.func.attr == "range" and node.keywords:
        found.add("range_hint")
  return found | _call_features(called)


def _text_features(text):
  called = set(re.findall(r"\b(?:tl|gl|libdevice)\.(\w+)\s*\(", text))
  found = _call_features(called)
  if re.search(r"\.to\(\s*(?:tl|gl)\.", text):
    found.add("conversion")
  if "gluon" in text:
    found.add("gluon")
  if "TensorDescriptor" in text:
    found.add("descriptor")
  for dtype, names in _DTYPES.items():
    if any(f".{name}" in text for name in names):
      found.add(dtype)
  return found


def features(text):
  """The feature names a reproducer uses; derived features included."""
  try:
    found = _tree_features(ast.parse(text))
  except SyntaxError:
    found = _text_features(text)
  if found & {"nested_loop", "while"}:
    found.add("nested_control")
  return found


def _checks_value(text):
  try:
    tree = ast.parse(text)
  except SyntaxError:
    return bool(re.search(r"\bassert\b|allclose|assert_close|\bexpected\b", text))
  for node in ast.walk(tree):
    if isinstance(node, ast.Assert):
      return True
    if isinstance(node, ast.Name) and node.id == "expected":
      return True
    if isinstance(node, ast.Call):
      func = node.func
      name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
      if name in _CHECK_CALLS:
        return True
  return False


def _label_signal(label):
  words = set(re.findall(r"[a-z0-9]+", (label or "").lower()))
  for signal, vocabulary in _LABEL_WORDS:
    if words & set(vocabulary):
      return signal
  return None


def symptom(label, text):
  """The inferred symptom with both of its signals."""
  label_signal = _label_signal(label)
  checks = _checks_value(text)
  inferred = label_signal or ("wrong_result" if checks else "crash")
  return {"symptom": inferred, "label_signal": label_signal, "checks_value": checks}


def census(seeds):
  """One row per manifest seed, in manifest order."""
  seeds = Path(seeds)
  manifest = json.loads((seeds / "manifest.json").read_text(encoding="utf-8"))
  rows = []
  for entry in manifest["seeds"]:
    text = (seeds / entry["seed_path"]).read_text(encoding="utf-8")
    try:
      ast.parse(text)
      parsed = True
    except SyntaxError:
      parsed = False
    label = entry.get("title")
    rows.append(
      {
        "issue": entry["issue"],
        "seed_path": entry["seed_path"],
        "label": label,
        "status": entry.get("status"),
        "fix_status": entry.get("fix_status"),
        "excerpt": bool(_EXCERPT.search(entry["seed_path"]))
        or bool(_EXCERPT.search("\n".join(text.splitlines()[:6]))),
        "parsed": parsed,
        "features": sorted(features(text)),
        **symptom(label, text),
      }
    )
  return rows


def _tally(rows, feature):
  using = [row for row in rows if feature in row["features"]]
  wrong = sum(row["symptom"] == "wrong_result" for row in using)
  return {
    "feature": feature,
    "files": len(using),
    "wrong_result": wrong,
    "crash": sum(row["symptom"] == "crash" for row in using),
    "hang": sum(row["symptom"] == "hang" for row in using),
    "density": wrong / len(using) if using else 0.0,
  }


def _order(entries):
  return sorted(entries, key=lambda e: (-e["wrong_result"], -e["density"]))


def rank(rows):
  """D7: the candidates in rank order, with outside features promoted."""
  candidates = _order([_tally(rows, feature) for feature in CANDIDATES])
  outside = _order([_tally(rows, feature) for feature in OUTSIDE])
  threshold = candidates[-1]["wrong_result"]
  promoted = [entry for entry in outside if entry["wrong_result"] > threshold]
  return {
    "order": _order(candidates + promoted),
    "promoted": [entry["feature"] for entry in promoted],
    "outside": outside,
  }


def _all_features(rows):
  return sorted({feature for row in rows for feature in row["features"]})


def _markdown(rows, ranking):
  lines = [
    "# Feature census of the issue reproducers",
    "",
    f"{len(rows)} seeds; {sum(not row['parsed'] for row in rows)} tagged by call"
    " names only (not valid Python).",
    "",
    "| Feature | Files | Wrong result | Crash | Hang | Density |",
    "|---|---:|---:|---:|---:|---:|",
  ]
  for feature in _all_features(rows):
    entry = _tally(rows, feature)
    lines.append(
      f"| `{feature}` | {entry['files']} | {entry['wrong_result']} | {entry['crash']}"
      f" | {entry['hang']} | {entry['density']:.2f} |"
    )
  lines += ["", "## D7 order", ""]
  for position, entry in enumerate(ranking["order"], 1):
    lines.append(
      f"{position}. `{entry['feature']}`: {entry['wrong_result']} wrong-result"
      f" issues of {entry['files']} (density {entry['density']:.2f})"
    )
  promoted = ", ".join(f"`{name}`" for name in ranking["promoted"]) or "none"
  lines += ["", f"Promoted from outside the five: {promoted}.", ""]
  return "\n".join(lines)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("seeds", type=Path)
  parser.add_argument("out", type=Path)
  arguments = parser.parse_args(argv)
  rows = census(arguments.seeds)
  ranking = rank(rows)
  totals = {feature: _tally(rows, feature) for feature in _all_features(rows)}
  symptoms = {}
  for row in rows:
    symptoms[row["symptom"]] = symptoms.get(row["symptom"], 0) + 1
  arguments.out.mkdir(parents=True, exist_ok=True)
  result = {
    "schema": "refractir.issue-feature-census.v1",
    "seeds": str(arguments.seeds),
    "symptoms": symptoms,
    "features": totals,
    "d7": ranking,
    "rows": rows,
  }
  (arguments.out / "CENSUS.json").write_text(
    json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8"
  )
  (arguments.out / "CENSUS.md").write_text(_markdown(rows, ranking), encoding="utf-8")
  print(json.dumps({"symptoms": symptoms, "order": ranking["order"]}, indent=1))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
