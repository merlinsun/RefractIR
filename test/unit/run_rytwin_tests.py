#!/usr/bin/env python3
"""Unit tests for rytwin (equivalence-preserving RefractIR transformer).

Scaffold-level coverage: the CLI loads p1 + descriptor + state-profile
sidecar, runs the (currently empty) Pass pipeline, and emits an equivalent
p2. The headline invariant checked here is p1(i) == p2(i): symiri must
return the same value for the emitted program as for the input.

Usage: python3 -m test.unit.run_rytwin_tests <rytwin> <rysmith> <symiri>
"""

import os
import re
import subprocess
import sys
import tempfile

GREEN = "\033[92m"
RED = "\033[91m"
GRAY = "\033[90m"
NC = "\033[0m"

results = []


def run(cmd, **kw):
  print(f"  {GRAY}[RUN>]{NC} " + " ".join(str(c) for c in cmd))
  return subprocess.run(cmd, capture_output=True, text=True, timeout=120, **kw)


def check(name, ok, detail=""):
  results.append((name, ok, detail))
  color = GREEN if ok else RED
  tag = "PASS" if ok else "FAIL"
  print(f"  [{color}{tag}{NC}] {name}" + (f" — {detail}" if detail and not ok else ""))


def gen_p1(rysmith, d, seed="5", nparams="2"):
  """Generate one rysmith leaf with a state sidecar + descriptor. Returns
  (p1_path, desc_path, entry_func, [param_values]) or None on failure."""
  r = run(
    [
      rysmith,
      "--emit-state",
      "pbb",
      "--emit-desc",
      "--n-funcs",
      "1",
      "--seed",
      seed,
      "--n-params",
      nparams,
      "-o",
      d,
    ]
  )
  if r.returncode != 0:
    return None
  sirs = [f for f in os.listdir(d) if f.endswith(".sir") and "_sym" not in f]
  descs = [
    f for f in os.listdir(d) if f.endswith(".json") and not f.endswith(".state.json")
  ]
  if not sirs or not descs:
    return None
  p1 = os.path.join(d, sorted(sirs)[0])
  desc = os.path.join(d, sorted(descs)[0])
  src = open(p1).read()
  fm = re.search(r"fun\s+(@\w+)\s*\(([^)]*)\)", src)
  entry = fm.group(1)
  pnames = [p.split(":")[0].strip() for p in fm.group(2).split(",") if p.strip()]
  hdr = re.search(r"//\s*SOLVED:\s*(.*)", src)
  kv = {}
  if hdr:
    for part in hdr.group(1).split(","):
      if "=" in part:
        k, v = part.strip().split("=", 1)
        kv[k.strip()] = v.strip()
  args = [kv[p] for p in pnames if p in kv]
  return p1, desc, entry, args


def gen_pool(rysmith, d, n="12", seed="202", emit_state=False, extra=None):
  """Generate a pool of pointer-free leaves (guard/vector/aggregate feature
  assertions are cleanest without memory ops; gen_ptr_pool covers those).
  Returns the sorted list of concrete .sir paths, or None on failure."""
  cmd = [rysmith, "--emit-desc", "--n-funcs", n, "--seed", seed]
  cmd += ["--n-params", "2", "--n-stmts", "5", "--max-ptr-depth", "0"]
  if emit_state:
    cmd += ["--emit-state", "pbb"]
  cmd += (extra or []) + ["-o", d]
  r = run(cmd)
  if r.returncode != 0:
    return None
  sirs = [f for f in os.listdir(d) if f.endswith(".sir") and "_sym" not in f]
  return [os.path.join(d, s) for s in sorted(sirs)]


def first_twinned(rytwin, sirs, extra=None):
  """Run rytwin --p-twin 1.0 over the pool; return (p1, p2, stdout+stderr) of
  the first program that grafts, or None if none does."""
  for p1 in sirs:
    p2 = p1[:-4] + ".p2.sir"
    r = run([rytwin, p1, "--p-twin", "1.0", "--seed", "3"] + (extra or []) + ["-o", p2])
    if r.returncode == 0 and os.path.exists(p2):
      return p1, p2, r.stdout + r.stderr
  return None


def symiri_result(symiri, path, entry, args):
  r = run([symiri, "--main", entry, path, "--"] + args)
  out = r.stdout + r.stderr
  m = re.search(r"Result:\s*(\S+)", out)
  return (
    r.returncode,
    m.group(1) if m else None,
    out.strip().splitlines()[-1:] if out else [],
  )


def test_no_twins_errors(rytwin, rysmith):
  """When no twin is grafted (forced here via --p-twin 0), rytwin reports an
  error and writes no output rather than emitting an unchanged copy of p1."""
  with tempfile.TemporaryDirectory() as d:
    g = gen_p1(rysmith, d)
    if not g:
      check("no-twins setup (rysmith gen)", False, "generation failed")
      return
    p1, _, _, _ = g
    p2 = os.path.join(d, "p2.sir")
    r = run([rytwin, p1, "--p-twin", "0", "-o", p2])
    check("rytwin errors when no twin grafted", r.returncode != 0, f"rc={r.returncode}")
    check("no output written when no twin", not os.path.exists(p2), "p2 exists")
    check(
      "error explains the missing twin",
      "no twin" in r.stderr or "nothing written" in r.stderr,
      r.stderr[:160],
    )


def test_no_sidecar_needed(rytwin, rysmith):
  """[Stage 1] rytwin profiles p1 in-process: a p1 generated WITHOUT
  --emit-state (no .state.json on disk) still twins successfully."""
  with tempfile.TemporaryDirectory() as d:
    sirs = gen_pool(rysmith, d, emit_state=False)
    if not sirs:
      check("no-sidecar setup (rysmith gen)", False, "generation failed")
      return
    assert not any(f.endswith(".state.json") for f in os.listdir(d))
    got = first_twinned(rytwin, sirs)
    check("rytwin twins without a .state.json sidecar", got is not None, "")


def test_corrupt_sidecar_falls_back(rytwin, rysmith):
  """[Stage 1] A stale/corrupt .state.json next to p1 does not kill the run:
  rytwin warns and falls back to in-process profiling."""
  with tempfile.TemporaryDirectory() as d:
    sirs = gen_pool(rysmith, d, emit_state=True)
    if not sirs:
      check("corrupt-sidecar setup (rysmith gen)", False, "generation failed")
      return
    states = [f for f in os.listdir(d) if f.endswith(".state.json")]
    for f in states:
      open(os.path.join(d, f), "w").write("garbage{not json")
    got = first_twinned(rytwin, sirs)
    check("corrupt sidecar tolerated (twinning still works)", got is not None, "")


def strip_solved_header(path):
  src = open(path).read()
  open(path, "w").write(
    "\n".join(ln for ln in src.splitlines() if not ln.startswith("// SOLVED:")) + "\n"
  )


def test_sidecar_preferred(rytwin, rysmith):
  """[Stage 1] When a valid .state.json is present, rytwin loads it instead
  of interpreting. Discriminator: with the descriptor and SOLVED header
  removed, in-process profiling would run at all-zero args and trap on the
  interest requires — only the sidecar path can succeed."""
  with tempfile.TemporaryDirectory() as d:
    sirs = gen_pool(rysmith, d, emit_state=True)
    if not sirs:
      check("sidecar-preferred setup (rysmith gen)", False, "generation failed")
      return
    for f in os.listdir(d):
      if f.endswith(".json") and not f.endswith(".state.json"):
        os.remove(os.path.join(d, f))
    for p1 in sirs:
      strip_solved_header(p1)
    rescued = 0
    for p1 in sirs:
      state = p1[:-4] + ".state.json"
      if not os.path.exists(state):
        continue
      # Control: without the sidecar this program must fail (in-process
      # profiling at zero args traps); skip programs that pass at zeros.
      hidden = state + ".hidden"
      os.rename(state, hidden)
      r_ctl = run([rytwin, p1, "--p-twin", "1.0", "--seed", "3", "-o", p1 + ".ctl.sir"])
      os.rename(hidden, state)
      if r_ctl.returncode == 0:
        continue
      r = run([rytwin, p1, "--p-twin", "1.0", "--seed", "3", "-o", p1 + ".p2.sir"])
      if r.returncode == 0:
        rescued += 1
        break
    check("valid sidecar loaded in preference to interpreting", rescued > 0, "")


GUARD_FN_RE = re.compile(r"fun\s+(@__twg_\w+)\s*\(([^)]*)\)")
GUARD_CALL_RE = re.compile(r"br\s+call\s+@__twg_\w+\s*\([^)]*\)\s*!=\s*0")


def guard_fun_bodies(src):
  """Return {guard_fn_name: body_text} for every @__twg_ function in src."""
  out = {}
  for m in GUARD_FN_RE.finditer(src):
    start = src.index("{", m.end())
    depth, i = 1, start + 1
    while depth and i < len(src):
      depth += {"{": 1, "}": -1}.get(src[i], 0)
      i += 1
    out[m.group(1)] = src[start:i]
  return out


def test_guard_is_function(rytwin, rysmith):
  """[Stage 2] The guard lives in a dedicated `fun @__twg_... : i1`; the
  twinned block's terminator is `br call @__twg_...(...) != 0, ...` and the
  old in-block scratch locals (%__twg/%__twa/...) are gone."""
  with tempfile.TemporaryDirectory() as d:
    sirs = gen_pool(rysmith, d, emit_state=False)
    if not sirs:
      check("guard-fn setup (rysmith gen)", False, "generation failed")
      return
    got = first_twinned(rytwin, sirs)
    if not got:
      check("guard-fn setup (twinnable program)", False, "no twin grafted")
      return
    src = open(got[1]).read()
    check("p2 declares a fun @__twg_", "fun @__twg_" in src, "")
    check("twinned block branches on call @__twg_", bool(GUARD_CALL_RE.search(src)), "")
    check(
      "no in-block guard scratch locals remain",
      "%__twg" not in src and "%__twa" not in src and "%__twfa" not in src,
      "",
    )


def test_guard_omits_state_the_twin_ignores(rytwin):
  """The guard covers the live-in state the twin actually depends on. %b is
  never read by the region, so the pass proves it free and the guard does not
  mention it — the earlier design pinned it anyway, to "maximize
  discrimination", which is exactly the over-fitting that made a twin
  recognizable."""
  fixture = """// SOLVED: %pa0=7
fun @guardfix(%pa0: i32) : i32 {
  let mut %a: i32 = 3;
  let mut %b: i32 = 1234567;
  ^entry:
    br ^work;
  ^work:
    %a = 2 * %a + %pa0;
    br ^exit;
  ^exit:
    ret %a;
}
"""
  with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "guardfix.sir")
    open(p1, "w").write(fixture)
    p2 = os.path.join(d, "p2.sir")
    r = run([rytwin, p1, "--p-twin", "1.0", "--seed", "3", "--validate", "-o", p2])
    check("fixture twinned", r.returncode == 0, r.stderr[:200])
    if r.returncode != 0:
      return
    bodies = guard_fun_bodies(open(p2).read())
    check(
      "the guard drops the unread variable %b",
      bodies and not any("1234567" in b for b in bodies.values()),
      f"guards={list(bodies)}",
    )
    check("and --validate still agrees", "validated: OK" in r.stdout, r.stdout[:160])


def test_guard_unique_names(rytwin, rysmith):
  """[Stage 2] One guard function per twin site, names unique; the rewritten
  program re-analyzes (rytwin exits 0). A region often swallows a whole leaf,
  so this checks every twinned program in the pool rather than hunting for
  one that happens to carry two sites."""
  with tempfile.TemporaryDirectory() as d:
    sirs = gen_pool(rysmith, d, emit_state=False)
    if not sirs:
      check("guard-names setup (rysmith gen)", False, "generation failed")
      return
    checked = bad = 0
    for p1 in sirs:
      p2 = p1[:-4] + ".p2.sir"
      r = run([rytwin, p1, "--p-twin", "1.0", "--seed", "3", "-o", p2])
      m = re.search(r"\((\d+) twin", r.stdout)
      if r.returncode != 0 or not m:
        continue
      checked += 1
      n = int(m.group(1))
      names = GUARD_FN_RE.findall(open(p2).read())
      if len(names) != n or len({nm for nm, _ in names}) != n:
        bad += 1
    check("guard-names setup (twinned programs)", checked > 0, "none twinned")
    check("guard fn per site, all names distinct", bad == 0, f"{bad}/{checked} bad")


def test_guard_aggregates_and_vectors(rytwin, rysmith, symiri):
  """[Stage 2] Aggregate state roots cross into the guard by address
  (`ptr [N] T` / `ptr @S` params, `addr %root` args, ptrindex/ptrfield+load
  navigation inside); vector roots cross per-lane (`%v[i]` args). The twins
  stay equivalent on the profiled input."""
  with tempfile.TemporaryDirectory() as d:
    sirs = gen_pool(rysmith, d, emit_state=False)
    if not sirs:
      check("agg/vec guard setup (rysmith gen)", False, "generation failed")
      return
    saw_ptr_param = saw_addr_arg = saw_lane_arg = False
    bad = 0
    for p1 in sirs:
      p2 = p1[:-4] + ".p2.sir"
      r = run([rytwin, p1, "--p-twin", "1.0", "--seed", "3", "-o", p2])
      if r.returncode != 0:
        continue
      src = open(p2).read()
      for _, params in GUARD_FN_RE.findall(src):
        if "ptr [" in params or "ptr @" in params:
          saw_ptr_param = True
      for m in re.finditer(r"br\s+call\s+@__twg_\w+\s*\(([^)]*)\)", src):
        if "addr %" in m.group(1):
          saw_addr_arg = True
        if re.search(r"%\w+\[\d+\]", m.group(1)):
          saw_lane_arg = True
      fn, pnames, _, iargs = parse_entry(open(p1).read())
      if (
        symiri_result(symiri, p1, fn, iargs)[1:]
        != symiri_result(symiri, p2, fn, iargs)[1:]
      ):
        bad += 1
    check("some guard takes an aggregate by pointer", saw_ptr_param, "")
    check("some caller passes addr %root", saw_addr_arg, "")
    check("some caller passes vector lanes %v[i]", saw_lane_arg, "")
    check("agg/vec twins preserve the profiled result", bad == 0, f"{bad} mismatch(es)")


def test_guard_compiles_c_and_wasm(rytwin, rysmith):
  """[Stage 2] Guard functions survive both backends: p1 and p2 compile via
  --target c --emit-main, the C binaries agree, and --target wasm emits."""
  import shutil

  with tempfile.TemporaryDirectory() as d:
    sirs = gen_pool(rysmith, d, emit_state=False, extra=["--emit-main"])
    if not sirs:
      check("backend setup (rysmith gen)", False, "generation failed")
      return
    got = first_twinned(rytwin, sirs, extra=["--target", "c", "--emit-main"])
    if not got:
      check("backend setup (twinnable program)", False, "no twin grafted")
      return
    p1, p2, _ = got
    check("compiled p2 uses guard functions", "fun @__twg_" in open(p2).read(), "")
    p2c = p2[:-4] + ".c"
    check("p2.c emitted", os.path.exists(p2c), "")
    r = run(
      [
        rytwin,
        p1,
        "--p-twin",
        "1.0",
        "--seed",
        "3",
        "--target",
        "wasm",
        "--emit-main",
        "-o",
        p2[:-4] + ".w.sir",
      ]
    )
    check(
      "p2 compiles to wasm",
      r.returncode == 0 and os.path.exists(p2[:-4] + ".w.wat"),
      r.stderr[:160],
    )
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc or not os.path.exists(p2c):
      return
    # p2's @main asserts the profiled checksum via @check_chksum, so a clean
    # run of the compiled binary proves the twinned program still computes
    # p1's result through the C backend.
    exe = os.path.join(d, "p2.bin")
    rc = run([cc, "-O1", "-o", exe, p2c, "-lm"])
    check("p2.c compiles", rc.returncode == 0, rc.stderr[:200])
    if rc.returncode == 0:
      rr = run([exe])
      check(
        "p2 binary runs clean (checksum assert passes)",
        rr.returncode == 0,
        f"rc={rr.returncode} out={rr.stdout[:120]}",
      )


# --- region twins (--twin-scope region) ---------------------------------
#
# The region scope generalizes the twin unit from one block to the maximal
# dominance region rooted at the entry: the guard fires on the entry state,
# and the twin reproduces the region's net effect and jumps straight to the
# region exit, skipping every block (and every loop iteration) in between.
# Sound by determinism — the full entry state fixes the whole continuation.

# A diamond so ^x has a predecessor outside ^e's region (^entry), making it
# the region exit rather than another dominated block. On %p0!=0 the trace
# is ^entry -> ^e -> ^b1 -> ^x, so the region rooted at ^e covers {^e,^b1}.
SEQ_FIXTURE = """// SOLVED: %p0=3
fun @seqreg(%p0: i32) : i32 {
  let mut %a: i32 = 1;
  let mut %b: i32 = 2;
  ^entry:
    br %p0 != 0, ^e, ^x;
  ^e:
    %a = %a + %p0;
    br ^b1;
  ^b1:
    %a = %a + %b;
    br ^x;
  ^x:
    ret %a;
}
"""

# A counting loop; the whole loop is dominated by ^head and everything runs
# to ^done (ret), so the region collapses all iterations into one twin that
# jumps to ^done.
LOOP_FIXTURE = """// SOLVED: %n=3
fun @loopreg(%n: i32) : i32 {
  let mut %i: i32 = 0;
  let mut %s: i32 = 0;
  ^entry:
    br ^head;
  ^head:
    br %i < %n, ^body, ^done;
  ^body:
    %s = %s + %i;
    %i = %i + 1;
    br ^head;
  ^done:
    ret %s;
}
"""


def test_removed_flags_rejected(rytwin):
  """The twin body is the region's own executed trace, so the flags that
  configured the solver-driven generator — and the block/region choice it
  was tuned per — no longer exist. Each must be rejected rather than
  silently ignored."""
  with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "seqreg.sir")
    open(p1, "w").write(SEQ_FIXTURE)
    p2 = os.path.join(d, "p2.sir")
    for flag in (
      ["--twin-scope", "region"],
      ["--twin-scope", "block"],
      ["--no-twin-smith"],
      ["--twin-retries", "2"],
      ["--twin-stmts", "4"],
    ):
      r = run([rytwin, p1, *flag, "-o", p2])
      check(f"{flag[0]} rejected", r.returncode != 0, f"rc={r.returncode}")


def test_no_solver_linked(rytwin):
  """Deciding a twin body no longer asks the solver anything, so rytwin
  must not link an SMT backend at all. The backend is linked statically
  here, so its symbols — not ldd — are the evidence."""
  r = subprocess.run(["nm", "-C", rytwin], capture_output=True, text=True)
  if r.returncode != 0:
    check("nm unavailable — skipped", True, "")
    return
  hits = [
    ln
    for ln in r.stdout.splitlines()
    if "bitwuzla" in ln.lower() or "::smt::" in ln.lower()
  ]
  check("rytwin carries no SMT backend symbols", not hits, f"{len(hits)} symbols")


def test_trace_body_unrolls_loop(rytwin):
  """The twin body is the executed trace laid straight: a loop that ran three
  times contributes its body three times, with no branch in between. A
  memoized constant (or a solved one-liner) would contribute one statement or
  none. The count comes off the graft log rather than the emitted text,
  because the disguise pass that follows rewrites the statements themselves —
  and is free to reorder them, so their order is no longer the claim."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, _, p2 = graft_log(rytwin, d, LOOP_FIXTURE, "loopreg", ["--validate"])
    check("loop fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    check(
      "the trace holds the loop body three times", trace_stmts(lines[0]) == 6, lines[0]
    )
    # twin_block_bodies keeps each block's terminator, which is the single jump
    # to the region exit; no branch may appear before it.
    inner = [
      ln for b in twin_block_bodies(open(p2).read()) for ln in b.splitlines()[:-1]
    ]
    check(
      "the unrolled body is straight-line",
      not [ln for ln in inner if ln.startswith("br ")],
      str(inner)[:200],
    )
    check("and it still validates", "validated: OK" in r.stdout, r.stdout[:200])


def test_trace_body_replays_region_stmts(rytwin):
  """A multi-block region contributes every block's statements: the sequence
  fixture holds one in each of its two blocks, and the trace holds both — one
  reads %p0 and the other %b, so both operands survive into the twin."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, _, p2 = graft_log(rytwin, d, SEQ_FIXTURE, "seqreg", ["--validate"])
    check("sequence fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    check(
      "the trace holds both blocks' statements", trace_stmts(lines[0]) == 2, lines[0]
    )
    body = "".join(twin_block_bodies(open(p2).read()))
    check("both blocks' operands are read", "%p0" in body and "%b" in body, body[:200])
    check("and it still validates", "validated: OK" in r.stdout, r.stdout[:200])


# A chain of unconditional branches: nothing about the path is in question,
# so the trace assumes no conditions at all.
CHAIN_FIXTURE = """// SOLVED: %p0=3
fun @chain(%p0: i32) : i32 {
  let mut %a: i32 = 1;
^entry:
  %a = %a + %p0;
  br ^b1;
^b1:
  %a = 2 * %a;
  br ^b2;
^b2:
  %a = %a - 3;
  br ^done;
^done:
  ret %a;
}
"""

# A branch that opens and rejoins inside the region: the twin replays the
# taken side only, so the guard must keep the state on that side.
DIAMOND_FIXTURE = """// SOLVED: %p0=3
fun @diamond(%p0: i32) : i32 {
  let mut %a: i32 = 1;
^entry:
  br ^head;
^head:
  br %p0 > 0, ^pos, ^neg;
^pos:
  %a = %a + 10;
  br ^join;
^neg:
  %a = 7 * %a;
  br ^join;
^join:
  %a = 3 * %a;
  br ^done;
^done:
  ret %a;
}
"""


def graft_log(rytwin, d, fixture, name, args):
  """Twin `fixture` with --verbose and return (result, the graft log lines)."""
  p1 = os.path.join(d, name + ".sir")
  open(p1, "w").write(fixture)
  p2 = os.path.join(d, name + ".p2.sir")
  r = run([rytwin, p1, "--p-twin", "1.0", "--seed", "3", "-v", *args, "-o", p2])
  lines = [ln for ln in r.stderr.splitlines() if "grafted" in ln]
  return r, lines, p1, p2


def path_cond_count(line):
  m = re.search(r"(\d+) path cond", line)
  return int(m.group(1)) if m else None


def test_trace_records_path_conditions(rytwin):
  """Flattening drops the branches, so the trace has to carry the conditions
  it assumed and the way each one went — otherwise nothing records that the
  twin is valid only while the state keeps taking that path."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, _, _ = graft_log(rytwin, d, LOOP_FIXTURE, "loopreg", [])
    check("loop fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if lines:
      check(
        "a loop region carries one condition per header visit",
        path_cond_count(lines[0]) == 4,
        lines[0],
      )
  with tempfile.TemporaryDirectory() as d:
    r, lines, _, _ = graft_log(rytwin, d, CHAIN_FIXTURE, "chain", [])
    check("chain fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if lines:
      check(
        "an all-unconditional region carries none",
        path_cond_count(lines[0]) == 0,
        lines[0],
      )
  with tempfile.TemporaryDirectory() as d:
    r, lines, _, _ = graft_log(rytwin, d, SEQ_FIXTURE, "seqreg", [])
    check("sequence fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if lines:
      check(
        "the region's own entry branch counts too",
        path_cond_count(lines[0]) == 1,
        lines[0],
      )


def test_trace_records_diamond_condition(rytwin, symiri):
  """A branch taken inside the region contributes exactly one condition, and
  the twin replays only the side that ran."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, p1, p2 = graft_log(rytwin, d, DIAMOND_FIXTURE, "diamond", ["--validate"])
    check("diamond fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    check(
      "the diamond contributes one condition", path_cond_count(lines[0]) == 1, lines[0]
    )
    body = "\n".join(twin_block_bodies(open(p2).read()))
    # The body is rewritten after flattening, so the taken side cannot be
    # recognized by its spelling. The untaken side's `7 *` is a shape no rule
    # introduces, which makes its absence the thing worth asserting.
    check(
      "the twin replays the taken side only",
      "7 * %a" not in body,
      body[:200],
    )
    r1 = symiri_result(symiri, p1, "@diamond", ["3"])
    r2 = symiri_result(symiri, p2, "@diamond", ["3"])
    check("diamond program equivalent", r1[1:] == r2[1:], f"{r1} vs {r2}")


# A value that only exists in memory: the interval domain does not track it,
# so a division by it cannot be proven safe for a range of entry states.
LOAD_DIV_FIXTURE = """// SOLVED: %p0=3
fun @loaddiv(%p0: i32) : i32 {
  let mut %a: [2] i32 = {7, 5};
  let mut %p: ptr i32 = undef;
  let mut %d: i32 = 0;
  let mut %r: i32 = 0;
^entry:
  %p = addr %a[1];
  br ^work;
^work:
  store %p, 5;
  %d = load %p;
  %r = %p0 / %d;
  br ^done;
^done:
  ret %r;
}
"""


# Walking an array through a pointer: each step lands on a cell whose name
# is known, so nothing is lost by going through memory.
PTR_WALK_FIXTURE = """// SOLVED: %p0=3
fun @ptrwalk(%p0: i32) : i32 {
  let mut %a: [3] i32 = {4, 5, 6};
  let mut %p: ptr i32 = undef;
  let mut %one: i32 = 1;
  let mut %r: i32 = 0;
^entry:
  %p = addr %a[0];
  br ^work;
^work:
  store %p, %p0;
  %p = %p + %one;
  %r = load %p;
  %r = %r + %p0;
  br ^done;
^done:
  ret %r;
}
"""

# A pointer read back out of memory: its target is whatever was stored
# there, which the domain does not follow.
PTR_PTR_FIXTURE = """// SOLVED: %p0=3
fun @ptrptr(%p0: i32) : i32 {
  let mut %a: [2] i32 = {7, 9};
  let mut %p: ptr i32 = undef;
  let mut %q: ptr i32 = undef;
  let mut %pp: ptr ptr i32 = undef;
  let mut %d: i32 = 1;
  let mut %r: i32 = 0;
^entry:
  %p = addr %a[1];
  %pp = addr %p;
  br ^work;
^work:
  %q = load %pp;
  %d = load %q;
  %r = %p0 / %d;
  br ^done;
^done:
  ret %r;
}
"""


# The pointer is set up before the region begins (^entry is ineligible — it
# calls out), so the region's own statements never mention `addr`. Only the
# provenance the profile recorded says what the load reads.
SEED_PTR_FIXTURE = """// SOLVED: %p0=3
fun @side() : i32 {
^e:
  ret 7;
}

fun @seedptr(%p0: i32) : i32 {
  let mut %a: [2] i32 = {3, 4};
  let mut %p: ptr i32 = undef;
  let mut %d: i32 = 0;
  let mut %r: i32 = 0;
^entry:
  %p = addr %a[1];
  %d = call @side();
  br ^work;
^work:
  %r = load %p;
  %r = %r + %p0;
  br ^done;
^done:
  ret %r;
}

fun @main() : i32 {
  let mut %r: i32 = undef;
^entry:
  %r = call @seedptr(3);
  ret %r;
}
"""


# Floating-point work: a pinned float is a constant, so the pass can follow
# it exactly — through arithmetic, a branch on a float, and a cast back to an
# integer.
FLOAT_FIXTURE = """// SOLVED: %p0=3
fun @floats(%p0: i32) : i32 {
  let mut %f: f64 = 1.5;
  let mut %g: f64 = 0.0;
  let mut %r: f64 = 0.0;
  let mut %i: i32 = 0;
^entry:
  %g = 2.0 * %f;
  br ^work;
^work:
  %r = %g + 1.25;
  %r = %r / %f;
  br %r > 0.0, ^pos, ^neg;
^pos:
  %i = %r as i32;
  br ^done;
^neg:
  %i = 0 - %p0;
  br ^done;
^done:
  ret %i;
}
"""


# Only lane 0 is ever read, so lane 1 is free and has no business being a
# parameter of the guard.
VEC_LANE_FIXTURE = """// SOLVED: %p0=3
fun @veclanes(%p0: i32) : i32 {
  let mut %v: <2> i32 = {5, 6};
  let mut %r: i32 = 0;
^entry:
  %r = %v[0] + %p0;
  br ^work;
^work:
  %r = 2 * %r;
  br ^done;
^done:
  ret %r;
}
"""


def guard_signature(src):
  m = re.search(r"fun @__twg_\S+\(([^)]*)\)", src)
  return m.group(1) if m else ""


def box_counts(line):
  """(free, ranged, pinned) from a graft log line, or None."""
  m = re.search(r"box: (\d+) free, (\d+) ranged, (\d+) pinned", line)
  return tuple(int(g) for g in m.groups()) if m else None


def box_widest(line):
  """(leaf, radius) of the widest ranged leaf, or None."""
  m = re.search(r"widest (\S+) \+/-(\d+)", line)
  return (m.group(1), int(m.group(2))) if m else None


def trace_stmts(line):
  """How many statements the flattened trace held, before any rewriting."""
  m = re.search(r"(\d+) stmts", line)
  return int(m.group(1)) if m else None


def verdict_note(line):
  m = re.search(r"verdict (ok|[^)]+)", line)
  return m.group(1) if m else None


def test_state_set_pass_proves_scalar_traces(rytwin):
  """The pass must prove the profiled state itself: every leaf is a single
  value there, so anything it cannot prove is a gap in the domain rather than
  a genuinely unprovable trace."""
  with tempfile.TemporaryDirectory() as d:
    for fixture, name in (
      (LOOP_FIXTURE, "loopreg"),
      (CHAIN_FIXTURE, "chain"),
      (DIAMOND_FIXTURE, "diamond"),
      (SEQ_FIXTURE, "seqreg"),
    ):
      r, lines, _, _ = graft_log(rytwin, d, fixture, name, [])
      if not lines:
        check(f"{name} twinned", False, r.stderr[:160])
        continue
      check(
        f"{name}: proven at the profiled state",
        verdict_note(lines[0]) == "ok",
        lines[0],
      )


def test_state_set_pass_follows_memory(rytwin, symiri):
  """A value that round-trips through memory is still a value: the pass
  follows `addr` provenance, so a store and the load that reads it back
  connect, and a division by the loaded value is provable."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, p1, p2 = graft_log(rytwin, d, LOAD_DIV_FIXTURE, "loaddiv", ["--validate"])
    check("load/div fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    check("the loaded divisor is proven", verdict_note(lines[0]) == "ok", lines[0])
    r1 = symiri_result(symiri, p1, "@loaddiv", ["3"])
    r2 = symiri_result(symiri, p2, "@loaddiv", ["3"])
    check("memory-bearing twin still equivalent", r1[1:] == r2[1:], f"{r1} vs {r2}")


def test_box_pins_control_and_opens_data(rytwin):
  """The shape of a loop is pinned and its data is not: moving the trip count
  or the induction variable takes another path, so the failing check names
  them and they stop; the accumulator has nothing to stop it but overflow."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, _, _ = graft_log(rytwin, d, LOOP_FIXTURE, "loopreg", [])
    check("loop fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    counts = box_counts(lines[0])
    check("the loop's shape stays pinned", counts and counts[2] == 2, str(counts))
    check("exactly one leaf opens", counts and counts[1] == 1, str(counts))
    widest = box_widest(lines[0])
    check(
      "the accumulator is the one that opens", widest and widest[0] == "%s", str(widest)
    )
    check(
      "it opens far past anything probing could enumerate",
      widest and widest[1] > 100000,
      str(widest),
    )


def test_box_frees_a_leaf_no_sampling_could(rytwin):
  """A leaf the trace provably does not depend on is dropped, not widened —
  no amount of probing could establish that, since it covers every value."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, _, _ = graft_log(rytwin, d, FLOAT_FIXTURE, "floats", [])
    check("float fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    counts = box_counts(lines[0])
    check("leaves come back free", counts and counts[0] > 0, str(counts))


def test_guard_states_ranges_and_drops_free_leaves(rytwin):
  """The guard says what the box found: a ranged leaf becomes a pair of
  bounds, a free leaf is not mentioned at all, and only pinned leaves are
  compared for equality."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, _, p2 = graft_log(rytwin, d, LOOP_FIXTURE, "loopreg", ["--validate"])
    check("loop fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    guard = "".join(guard_fun_bodies(open(p2).read()).values())
    check(
      "the ranged leaf is bounded, not pinned",
      "cmp >=" in guard and "cmp <=" in guard,
      guard[:300],
    )
    check("pinned leaves keep their equality", "cmp ==" in guard, guard[:300])
  with tempfile.TemporaryDirectory() as d:
    r, lines, _, p2 = graft_log(rytwin, d, FLOAT_FIXTURE, "floats", ["--validate"])
    if not lines:
      check("float fixture twinned", False, r.stderr[:160])
      return
    counts = box_counts(lines[0])
    src = open(p2).read()
    guard = "".join(guard_fun_bodies(src).values())
    sig = re.search(r"fun (@__twg_\S+)\(([^)]*)\)", src)
    check(
      "free leaves are not compared",
      counts
      and counts[0] > 0
      and "cmp == %i," not in guard
      and "cmp == %p0," not in guard,
      f"{counts} guard={guard[:200]}",
    )
    check(
      "and not even passed in",
      sig and "%i:" not in sig.group(2) and "%p0:" not in sig.group(2),
      sig.group(2) if sig else "",
    )
    check(
      "leaves the box does not classify stay pinned",
      "cmp == %f," in guard,
      guard[:200],
    )


def test_guard_drops_free_vector_lanes(rytwin, symiri):
  """A guard should not take a parameter it never looks at. A vector lane the
  region never reads is free, so its lane parameter goes with it."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, p1, p2 = graft_log(
      rytwin, d, VEC_LANE_FIXTURE, "veclanes", ["--validate"]
    )
    check("vector fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    sig = guard_signature(open(p2).read())
    check("the unread lane is not a parameter", "%v__l1" not in sig, sig)
    check("the read lane still is", "%v__l0" in sig, sig)
    r1 = symiri_result(symiri, p1, "@veclanes", ["3"])
    r2 = symiri_result(symiri, p2, "@veclanes", ["3"])
    check("vector program equivalent", r1[1:] == r2[1:], f"{r1} vs {r2}")


def test_widened_guard_fires_on_another_input(rytwin, symiri):
  """The point of widening: an input the twin was never profiled on still
  lands inside the guard, so the twin arm runs and the program agrees."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, p1, p2 = graft_log(rytwin, d, CHAIN_FIXTURE, "chain", ["--validate"])
    check("chain fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    counts = box_counts(lines[0])
    check("something opened", counts and (counts[0] or counts[1]), str(counts))
    # profiled on %p0 = 3; try a neighbour it never saw
    trace = dump_trace(symiri, p2, "@chain", ["9"])
    check("the twin runs on an unprofiled input", "__twin" in trace, trace[-200:])
    r1 = symiri_result(symiri, p1, "@chain", ["9"])
    r2 = symiri_result(symiri, p2, "@chain", ["9"])
    check("and agrees with the original there", r1[1:] == r2[1:], f"{r1} vs {r2}")


def test_validate_spot_checks_the_box(rytwin):
  """--validate no longer covers a guard: it runs the profiled input, and the
  guard now admits many states. So it also samples states inside the box and
  runs the region against the twin body on each, reporting how many."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, _, _ = graft_log(rytwin, d, LOOP_FIXTURE, "loopreg", ["--validate"])
    check("loop fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    out = r.stdout + r.stderr
    m = re.search(r"spot-checked (\d+) state", out)
    check("--validate reports spot checks", m is not None, out[:300])
    if m:
      check("it checked more than the profiled state", int(m.group(1)) > 1, m.group(0))


def test_box_computes_ceilings_before_searching(rytwin):
  """Pushing each check's requirement backward settles the leaves a branch
  holds still — a loop's trip count and induction variable — before any trial
  runs. They then take no part in the lockstep rounds, so a refusal that is
  really about them no longer costs the data leaf its width."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, _, _ = graft_log(rytwin, d, LOOP_FIXTURE, "loopreg", [])
    check("loop fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    counts = box_counts(lines[0])
    check("the loop's shape is still pinned", counts == (0, 1, 2), str(counts))
    widest = box_widest(lines[0])
    # Without ceilings the control leaves joined every round and their refusals
    # froze the accumulator at half this width.
    check(
      "the accumulator is no longer dragged down with them",
      widest and widest[1] > 10000000,
      str(widest),
    )


def test_disguised_body_is_not_a_copy(rytwin, symiri):
  """A flattened trace is the region's statements back in order, which is a
  copy however wide the guard is. After rewriting it should no longer be one,
  and it must still agree with the region."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, p1, p2 = graft_log(rytwin, d, LOOP_FIXTURE, "loopreg", ["--validate"])
    check("loop fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    body = "".join(twin_block_bodies(open(p2).read()))
    m = re.search(r"(\d+) rewrites", lines[0])
    check("the body was rewritten", m and int(m.group(1)) > 0, lines[0])
    # Which statements get rewritten depends on the draw, so the claim worth
    # asserting is that the body is no longer the trace as flattened — it has
    # more statements than the region has, or names locals only the rewriting
    # introduces.
    region_stmts = LOOP_FIXTURE.count(";") - LOOP_FIXTURE.count("ret")
    check(
      "and no longer reads as the region",
      "%__ao" in body
      or len([ln for ln in body.splitlines() if ln.strip()]) > region_stmts,
      body[:300],
    )
    check("--validate still agrees", "validated: OK" in r.stdout, r.stdout[:200])
    r1 = symiri_result(symiri, p1, "@loopreg", ["3"])
    r2 = symiri_result(symiri, p2, "@loopreg", ["3"])
    check("disguised program equivalent", r1[1:] == r2[1:], f"{r1} vs {r2}")


def test_disguise_rolls_back_what_it_cannot_prove(rytwin):
  """A rule is sound on its own; two of them together need not be. The
  state-set pass re-checks after every application and the body is restored
  when it no longer proves — so rewrites being undone is the mechanism
  working, not a fault."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, _, _ = graft_log(rytwin, d, LOOP_FIXTURE, "loopreg", ["--validate"])
    if not lines:
      check("loop fixture twinned", False, r.stderr[:160])
      return
    m = re.search(r"(\d+) rewrites \((\d+) undone", lines[0])
    check("the report distinguishes kept from undone", m is not None, lines[0])
    if m:
      check("some were undone", int(m.group(2)) > 0, m.group(0))
      check("and some were kept", int(m.group(1)) > 0, m.group(0))
    check("the result still validates", "validated: OK" in r.stdout, r.stdout[:200])


def test_every_attempt_lands_on_a_live_candidate(rytwin):
  """The engine scans for candidates once per attempt rather than once per
  round. A list built once goes stale the moment a rewrite splices statements
  in — every position after it means something else — so half the budget used
  to be spent skipping positions that no longer matched, and the picks that
  survived were whatever sat early in the body."""
  for seed in ("1", "3"):
    with tempfile.TemporaryDirectory() as d:
      p1 = os.path.join(d, "chain.sir")
      open(p1, "w").write(CHAIN_FIXTURE)
      p2 = os.path.join(d, "chain.p2.sir")
      r = run([rytwin, p1, "--p-twin", "1.0", "--seed", seed, "-v", "-o", p2])
      lines = [ln for ln in r.stderr.splitlines() if "grafted" in ln]
      if not lines:
        check(f"chain fixture twinned at seed {seed}", False, r.stderr[:200])
        continue
      m = re.search(r"(\d+) rewrites \((\d+) undone", lines[0])
      spent = int(m.group(1)) + int(m.group(2)) if m else 0
      # The budget is kTwinRewriteRounds * kTwinRewritesPerRound = 24; a rule
      # that declines after matching is the only thing that should cost less.
      check(
        f"the budget is spent on live candidates at seed {seed}",
        spent >= 23,
        str(spent),
      )


def test_disguise_varies_with_the_seed(rytwin):
  """The rewriting is seeded, so two seeds give two different bodies for the
  same region — otherwise every twin of a given region would look alike."""
  bodies = []
  for seed in ("3", "9"):
    with tempfile.TemporaryDirectory() as d:
      p1 = os.path.join(d, "loopreg.sir")
      open(p1, "w").write(LOOP_FIXTURE)
      p2 = os.path.join(d, "p2.sir")
      rr = run([rytwin, p1, "--p-twin", "1.0", "--seed", seed, "-o", p2])
      if rr.returncode == 0:
        bodies.append("".join(twin_block_bodies(open(p2).read())))
  check("both seeds twinned", len(bodies) == 2, str(len(bodies)))
  if len(bodies) == 2:
    check("different seeds, different bodies", bodies[0] != bodies[1], bodies[0][:150])


def test_rewrite_rules_selftest(rytwin):
  """Every anti-optimization rule claims that two spellings compute the same
  thing.
  The claim is settled by *running* both through the interpreter, since the
  authority on what a RefractIR statement does is the interpreter and not a
  restatement of it in the checking code — and a restatement could not cover
  a rule with no closed form at all, which was half the catalog."""
  r = run([rytwin, "--selftest-antiopt", "-v"])
  check("rewrite selftest passes", r.returncode == 0, (r.stderr + r.stdout)[:400])
  check("the interpreter is the oracle", "interpreter" in r.stdout, r.stdout[:200])
  m = re.search(r"(\d+) rule", r.stdout)
  check(
    "it checked the catalog, not a corner of it",
    m and int(m.group(1)) >= 25,
    r.stdout[:200],
  )
  checked = set(re.findall(r"checked ([a-z0-9-]+)", r.stdout))
  for structural in ("swap-adjacent", "memory-routing", "un-cse", "select-to-mask"):
    check(
      f"{structural} is checked too", structural in checked, str(sorted(checked))[:300]
    )


# Each family-A fixture puts one shape in front of the rules and bounds the
# leaf that shape needs bounded: a rewrite the state-set pass cannot re-prove
# over the whole box is rolled back, so an unbounded leaf would leave the
# rule sound, applicable, and never kept.

MUL_FIXTURE = """// SOLVED: %p0=3
fun @mulshift(%p0: i32) : i32 {
  let mut %a: i32 = 0;
  let mut %r: i32 = 0;
^entry:
  %a = %p0;
  br ^head;
^head:
  br %a > 0, ^work, ^done;
^work:
  %r = 8 * %a;
  br ^done;
^done:
  ret %r;
}
"""

NOT_FIXTURE = """// SOLVED: %p0=3
fun @comp(%p0: i32) : i32 {
  let mut %r: i32 = 0;
^entry:
  br %p0 > 0, ^work, ^done;
^work:
  %r = ~%p0;
  %r = %r + %p0;
  br ^done;
^done:
  ret %r;
}
"""

SUB_FIXTURE = """// SOLVED: %p0=3
fun @subadd(%p0: i32) : i32 {
  let mut %b: i32 = 5;
  let mut %r: i32 = 0;
^entry:
  br %p0 > 0, ^work, ^done;
^work:
  %r = %p0 - %b;
  br ^done;
^done:
  ret %r;
}
"""

DOUBLE_FIXTURE = """// SOLVED: %p0=3
fun @dbl(%p0: i32) : i32 {
  let mut %r: i32 = 0;
^entry:
  br %p0 > 0, ^work, ^done;
^work:
  %r = %p0 + %p0;
  br ^done;
^done:
  ret %r;
}
"""

CMP_FIXTURE = """// SOLVED: %p0=3
fun @cmpswap(%p0: i32) : i32 {
  let mut %b: i32 = 5;
  let mut %c: i1 = 0;
  let mut %r: i32 = 0;
^entry:
  br %p0 > 0, ^work, ^done;
^work:
  %c = cmp < %p0, %b;
  %r = %c as i32;
  br ^done;
^done:
  ret %r;
}
"""


def rules_kept(rytwin, fixture, name, seeds=("1", "2", "3", "4", "5", "6")):
  """Twin `fixture` once per seed and return (rule names the log reports as
  kept over all of them, every run validated). Which rules a body draws is
  seeded, so a rule is looked for across several seeds rather than one."""
  names, ok = set(), True
  for seed in seeds:
    with tempfile.TemporaryDirectory() as d:
      p1 = os.path.join(d, name + ".sir")
      open(p1, "w").write(fixture)
      p2 = os.path.join(d, name + ".p2.sir")
      r = run(
        [rytwin, p1, "--p-twin", "1.0", "--seed", seed, "-v", "--validate", "-o", p2]
      )
      if r.returncode != 0 or "validated: OK" not in r.stdout:
        ok = False
        continue
      for ln in r.stderr.splitlines():
        m = re.search(r"rewrites \(\d+ undone; ([^)]*)\)", ln)
        if m:
          names.update(re.findall(r"([a-z0-9-]+) x\d+", m.group(1)))
  return names, ok


def test_mask_stays_inside_its_own_type(rytwin):
  """An inserted mask is a literal of the statement's type, and i1 holds only
  0 and -1. Drawing it from a positive range emitted `let %k: i1 = 1;` — a
  program the checker rejects, so any region holding an i1 local aborted."""
  for seed in ("1", "2", "3"):
    with tempfile.TemporaryDirectory() as d:
      p1 = os.path.join(d, "cmpswap.sir")
      open(p1, "w").write(CMP_FIXTURE)
      p2 = os.path.join(d, "cmpswap.p2.sir")
      r = run([rytwin, p1, "--p-twin", "1.0", "--seed", seed, "--validate", "-o", p2])
      ok = r.returncode == 0 and "validated: OK" in r.stdout
      check(
        f"a region with an i1 local twins at seed {seed}",
        ok,
        (r.stderr + r.stdout)[:200],
      )


def test_negation_is_unfolded(rytwin):
  """`%a - 3` and `%a + -3` are the same value; a compiler prefers the first,
  so the rules prefer the second."""
  kept, ok = rules_kept(rytwin, CHAIN_FIXTURE, "chain")
  check(
    "fold-negation fires on a literal subtraction", "fold-negation" in kept, str(kept)
  )
  check("and the twins validate", ok)


def test_multiplication_becomes_a_shift(rytwin):
  """`8 * %a` is `%a << 3` only where the guard proves %a >= 0 — RefractIR's
  shift is arithmetic and traps on a negative operand."""
  kept, ok = rules_kept(rytwin, MUL_FIXTURE, "mulshift")
  check(
    "mul-to-shift fires where the box proves the sign",
    "mul-to-shift" in kept,
    str(kept),
  )
  check("and the twins validate", ok)


UNBOUNDED_MUL_FIXTURE = """// SOLVED: %p0=3
fun @mulfree(%p0: i32) : i32 {
  let mut %a: i32 = 0;
  let mut %r: i32 = 0;
^entry:
  %a = %p0;
  br ^work;
^work:
  %r = 8 * %a;
  br ^done;
^done:
  ret %r;
}
"""


def test_shift_refuses_what_the_box_does_not_prove(rytwin):
  """The same multiplication, with nothing holding %a above zero: the box
  admits negative states, `<<` traps on those, and the rewrite is undone.
  A rule is offered everywhere and kept only where it is proven."""
  kept, ok = rules_kept(rytwin, UNBOUNDED_MUL_FIXTURE, "mulfree")
  check(
    "mul-to-shift is not kept without the sign", "mul-to-shift" not in kept, str(kept)
  )
  check("and the twins still validate", ok)


def test_complement_becomes_arithmetic(rytwin):
  """`~%x` is `0 - %x - 1`, which crosses out of the bitwise domain and costs
  a reader the identity to get back."""
  kept, ok = rules_kept(rytwin, NOT_FIXTURE, "comp")
  check("not-complement fires on a complement", "not-complement" in kept, str(kept))
  check("and the twins validate", ok)


def test_subtraction_becomes_addition(rytwin):
  """`%x - %y` is `%x + ~%y + 1`, the two's-complement definition written
  out."""
  kept, ok = rules_kept(rytwin, SUB_FIXTURE, "subadd")
  check("sub-as-add fires on a subtraction", "sub-as-add" in kept, str(kept))
  check("and the twins validate", ok)


def test_doubling_becomes_multiplication(rytwin):
  """`%x + %x` is `2 * %x` — strength reduction run backwards, so the
  optimizer has to redo work it already did on the way in."""
  kept, ok = rules_kept(rytwin, DOUBLE_FIXTURE, "dbl")
  check("double-as-mul fires on a self-addition", "double-as-mul" in kept, str(kept))
  check("and the twins validate", ok)


def test_comparison_operands_swap(rytwin):
  """`cmp < %x, %y` is `cmp > %y, %x`. Trap-free, so it fires wherever a
  comparison appears."""
  kept, ok = rules_kept(rytwin, CMP_FIXTURE, "cmpswap")
  check("swap-compare fires on a comparison", "swap-compare" in kept, str(kept))
  check("and the twins validate", ok)


def test_values_round_trip_through_a_wider_type(rytwin):
  """Widening then narrowing back is the identity, and it puts a cast pair in
  front of anything reading the body."""
  kept, ok = rules_kept(rytwin, CHAIN_FIXTURE, "chain")
  check("cast-roundtrip fires", "cast-roundtrip" in kept, str(kept))
  check("and the twins validate", ok)


# Two constants the box has every reason to free: nothing about the trace
# depends on their *entry* values being what the profile saw, and a leaf the
# trace proves without is free by definition.
FREED_CONST_FIXTURE = """// SOLVED: %p0=3
fun @freedconst(%p0: i32) : i32 {
  let %a: i32 = 12;
  let %b: i32 = 10;
  let mut %r: i32 = 0;
^entry:
  br %p0 > 0, ^work, ^done;
^work:
  %r = %a & %b;
  %r = %r + %p0;
  br ^done;
^done:
  ret %r;
}
"""


def test_a_freed_leaf_stays_unknown(rytwin):
  """Freeing a leaf is the box saying the guard will not check it, so the pass
  may not go on knowing its value — not even when the declaration says it is
  an immutable constant. Assuming it anyway freed leaves the body did depend
  on, and the twin then disagreed with its region on a state its own guard
  admitted."""
  for seed in ("1", "2", "3"):
    with tempfile.TemporaryDirectory() as d:
      p1 = os.path.join(d, "freedconst.sir")
      open(p1, "w").write(FREED_CONST_FIXTURE)
      p2 = os.path.join(d, "freedconst.p2.sir")
      r = run([rytwin, p1, "--p-twin", "1.0", "--seed", seed, "--validate", "-o", p2])
      ok = r.returncode == 0 and "validated: OK" in r.stdout
      check(
        f"the twin holds on every admitted state at seed {seed}",
        ok,
        (r.stderr + r.stdout)[:200],
      )


# A shifted or masked value feeding arithmetic: the sum is what forces the
# pass to say something about it, since an unknown operand cannot be proven
# not to overflow and the leaf that fed it then has to stay pinned. The
# shift amount and the mask are set *inside* the region so they are constants
# of the trace rather than leaves the box would widen alongside %p0.
SHIFT_WIDEN_FIXTURE = """// SOLVED: %p0=-8
fun @shiftbox(%p0: i32) : i32 {
  let mut %k: i32 = 0;
  let mut %r: i32 = 0;
^entry:
  %k = 2;
  br ^work;
^work:
  %r = %p0 >> %k;
  %r = %r + %p0;
  br ^done;
^done:
  ret %r;
}
"""

# The negative side is where a logical shift and a mask stop agreeing with
# the arithmetic ones. %p0 is profiled negative and nothing holds it there,
# so the box widens it across zero and the bound has to come from the
# operator itself.
LSHIFT_WIDEN_FIXTURE = SHIFT_WIDEN_FIXTURE.replace("@shiftbox", "@lshiftbox").replace(
  ">> %k", ">>> %k"
)

MASK_WIDEN_FIXTURE = """// SOLVED: %p0=-8
fun @maskbox(%p0: i32) : i32 {
  let mut %m: i32 = 0;
  let mut %r: i32 = 0;
^entry:
  %m = 255;
  br ^work;
^work:
  %r = %p0 & %m;
  %r = %r + %p0;
  br ^done;
^done:
  ret %r;
}
"""

# The same shape with a mask that keeps the sign bit: `x & -8` says nothing
# about how large x is, and the pass must not pretend otherwise.
NEG_MASK_FIXTURE = MASK_WIDEN_FIXTURE.replace("@maskbox", "@negmaskbox").replace(
  "= 255", "= -8"
)


def widened_leaf(rytwin, fixture, name):
  """Twin `fixture` and return ((free, ranged, pinned), widest radius) off the
  graft log, or (None, None)."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, _, _ = graft_log(rytwin, d, fixture, name, ["--validate"])
    if not lines:
      check(f"{name} twinned", False, r.stderr[:200])
      return None, None
    widest = box_widest(lines[0])
    return box_counts(lines[0]), (widest[1] if widest else 0)


def test_state_set_pass_bounds_a_shift(rytwin):
  """`%x >> 2` is `%x` divided by four rounding down — monotone, so the range
  shifts with it. Calling it unknown poisoned the addition that read it, and
  the leaf behind it had to stay pinned."""
  counts, radius = widened_leaf(rytwin, SHIFT_WIDEN_FIXTURE, "shiftbox")
  if counts is None:
    return
  check("the shifted trace opens a range", counts[1] >= 1, str(counts))
  check("and it is a real one", radius > 1000000, str(radius))


def test_state_set_pass_bounds_a_logical_shift(rytwin):
  """`%x >>> 2` is not monotone across zero — `-1 >>> 1` is the largest value
  there is — but the shifted-out bits are gone either way, so even a value the
  guard admits negative lands well inside the type."""
  counts, radius = widened_leaf(rytwin, LSHIFT_WIDEN_FIXTURE, "lshiftbox")
  if counts is None:
    return
  check("the logically shifted trace opens a range", counts[1] >= 1, str(counts))
  check("and it is a real one", radius > 1000000, str(radius))


def test_state_set_pass_bounds_a_mask(rytwin):
  """The bits of `%x & 255` are a subset of 255's, so the result is in
  [0, 255] whatever %x holds — including the negatives this guard admits."""
  counts, radius = widened_leaf(rytwin, MASK_WIDEN_FIXTURE, "maskbox")
  if counts is None:
    return
  check("the masked trace opens a range", counts[1] >= 1, str(counts))
  check("and it is a real one", radius > 1000000, str(radius))


def test_state_set_pass_declines_a_signed_mask(rytwin):
  """`%x & -8` keeps the sign bit, and with %x negative too there is nothing
  to bound the result by. The pass must decline rather than invent a bound,
  so the leaf behind it stays pinned."""
  counts, _ = widened_leaf(rytwin, NEG_MASK_FIXTURE, "negmaskbox")
  if counts is None:
    return
  check("nothing is claimed about a signed mask", counts[1] == 0, str(counts))


def test_shifted_program_still_agrees(rytwin, symiri):
  """The widened guard admits far more states than the profiled one, so the
  twin has to agree with the region on an input the profile never saw."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, p1, p2 = graft_log(
      rytwin, d, SHIFT_WIDEN_FIXTURE, "shiftbox", ["--validate"]
    )
    check("shift fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    check("--validate agrees", "validated: OK" in r.stdout, r.stdout[:200])
    for arg in ("8", "-37", "1000003"):
      r1 = symiri_result(symiri, p1, "@shiftbox", [arg])
      r2 = symiri_result(symiri, p2, "@shiftbox", [arg])
      check(f"same answer for {arg}", r1[1:] == r2[1:], f"{r1} vs {r2}")


# The crossings that leave the bitwise domain need their operands *known*,
# not merely bounded: the interval domain widens `& | ^` of two ranges to
# unknown, and arithmetic over an unknown cannot be proven not to overflow.
# Assigning the operands inside the region makes them constants of the trace
# rather than leaves the box would widen. One operator per fixture, so a rule
# that fails to fire is a rule that could not rather than one that lost a
# draw.
def bitwise_fixture(name, op):
  """One operator, three times over: several rules match the same shape and
  only one of them can have a given site, so a single site would test the
  draw rather than the rule."""
  return f"""// SOLVED: %p0=3
fun @{name}(%p0: i32) : i32 {{
  let mut %a: i32 = 0;
  let mut %b: i32 = 0;
  let mut %r: i32 = 0;
  let mut %s: i32 = 0;
  let mut %t: i32 = 0;
^entry:
  %a = 12;
  %b = 10;
  br ^work;
^work:
  %r = %a {op} %b;
  %s = %a {op} %b;
  %t = %a {op} %b;
  %r = %r + %s + %t + %p0;
  br ^done;
^done:
  ret %r;
}}
"""


AND_FIXTURE = bitwise_fixture("bitand", "&")
OR_FIXTURE = bitwise_fixture("bitor", "|")
SUB_CONST_FIXTURE = bitwise_fixture("constsub", "-")

# The same three operators with nothing pinning the operands: the trap-free
# crossings still apply here, because they never leave the bitwise domain.
OPEN_BITWISE_FIXTURE = """// SOLVED: %p0=3
fun @openbits(%p0: i32) : i32 {
  let mut %m: i32 = 6;
  let mut %r: i32 = 0;
^entry:
  br %p0 > 0, ^work, ^done;
^work:
  %r = %p0 ^ %m;
  %m = %p0 & %m;
  br ^done;
^done:
  ret %r;
}
"""

SPLIT_FIXTURE = """// SOLVED: %p0=3
fun @bitsplit(%p0: i32) : i32 {
  let mut %r: i32 = 0;
^entry:
  br %p0 > 0, ^work, ^done;
^work:
  %r = %p0;
  %r = %r + %p0;
  br ^done;
^done:
  ret %r;
}
"""


def test_and_crosses_into_arithmetic(rytwin):
  """`%x & %y` is `%x - (%x & ~%y)` and `(%x | %y) - (%x ^ %y)`. Both leave
  the bitwise domain, which is what an optimizer will not follow back."""
  kept, ok = rules_kept(rytwin, AND_FIXTURE, "bitand")
  check("mba-and-sub fires on a conjunction", "mba-and-sub" in kept, str(kept))
  check("mba-and-diff fires on a conjunction", "mba-and-diff" in kept, str(kept))
  check("and the twins validate", ok)


def test_or_crosses_into_arithmetic(rytwin):
  """`%x | %y` is `%x + %y - (%x & %y)` — inclusion-exclusion, spelled out."""
  kept, ok = rules_kept(rytwin, OR_FIXTURE, "bitor")
  check("mba-or-add fires on a disjunction", "mba-or-add" in kept, str(kept))
  check("and so does the older crossing", "mba-or" in kept, str(kept))
  check("and the twins validate", ok)


def test_subtraction_crosses_into_bitwise(rytwin):
  """`%x - %y` is `(%x ^ %y) - 2 * (~%x & %y)`, the dual of the carry
  identity family B already carries for `+`."""
  kept, ok = rules_kept(rytwin, SUB_CONST_FIXTURE, "constsub")
  check("mba-sub fires on a subtraction", "mba-sub" in kept, str(kept))
  check("and the twins validate", ok)


def test_xor_stays_inside_the_bitwise_domain(rytwin):
  """`%x ^ %y` is `(%x | %y) & ~(%x & %y)` — the one crossing with no
  arithmetic in it, so unlike the rest it needs nothing proven and fires
  where the operands are merely values."""
  kept, ok = rules_kept(rytwin, OPEN_BITWISE_FIXTURE, "openbits")
  check("mba-xor-nand fires without a license", "mba-xor-nand" in kept, str(kept))
  check("and the twins validate", ok)


def test_de_morgan_doubles_the_complements(rytwin):
  """`%x & %y` is `~(~%x | ~%y)` and `%x | %y` is `~(~%x & ~%y)`. Trap-free
  both ways round."""
  kept, ok = rules_kept(rytwin, OPEN_BITWISE_FIXTURE, "openbits")
  check("mba-demorgan fires on a conjunction", "mba-demorgan-and" in kept, str(kept))
  # The OR fixture also matches several arithmetic crossings. Sample a fixed,
  # wider seed set so this checks the De Morgan rule's availability rather than
  # whether six random candidate draws happened to select it.
  demorgan_or_seeds = tuple(str(seed) for seed in range(1, 129))
  kept2, ok2 = rules_kept(rytwin, OR_FIXTURE, "bitor", seeds=demorgan_or_seeds)
  check("and on a disjunction", "mba-demorgan-or" in kept2, str(kept2))
  check("and the twins validate", ok and ok2)


def test_value_splits_at_a_bit(rytwin):
  """A non-negative value is its high part plus its low part:
  `%x == (%x >>> k << k) + (%x & (2^k - 1))`. Licensed by the sign, since
  `<<` traps on a negative operand."""
  kept, ok = rules_kept(rytwin, SPLIT_FIXTURE, "bitsplit")
  check(
    "mba-bit-split fires where the sign is proven", "mba-bit-split" in kept, str(kept)
  )
  check("and the twins validate", ok)


RECOMPUTE_FIXTURE = """// SOLVED: %p0=3
fun @recompute(%p0: i32) : i32 {
  let mut %t: i32 = 0;
  let mut %r: i32 = 0;
  let mut %s: i32 = 0;
^entry:
  %t = %p0 + 1;
  br ^work;
^work:
  %r = %t + %t;
  %s = %t + %r;
  br ^done;
^done:
  ret %s;
}
"""


def twin_sources(rytwin, fixture, name, seeds=("1", "2", "3", "4", "5", "6")):
  """The twinned program `fixture` produces, one source string per seed."""
  out = []
  for seed in seeds:
    with tempfile.TemporaryDirectory() as d:
      p1 = os.path.join(d, name + ".sir")
      open(p1, "w").write(fixture)
      p2 = os.path.join(d, name + ".p2.sir")
      r = run([rytwin, p1, "--p-twin", "1.0", "--seed", seed, "-o", p2])
      if r.returncode == 0 and os.path.exists(p2):
        out.append(open(p2).read())
  return out


def twin_text(rytwin, fixture, name, seeds=("1", "2", "3", "4", "5", "6")):
  """The twin bodies `fixture` produces, one string per seed."""
  return [
    "".join(twin_block_bodies(src))
    for src in twin_sources(rytwin, fixture, name, seeds)
  ]


def twin_declarations(rytwin, fixture, name, seeds=("1", "2", "3", "4", "5", "6")):
  """The `let` declarations of every twinned program, one string per seed."""
  out = []
  for src in twin_sources(rytwin, fixture, name, seeds):
    lets = [ln for ln in src.splitlines() if ln.strip().startswith("let ")]
    out.append("\n".join(lets))
  return out


def test_a_value_is_recomputed_instead_of_reused(rytwin):
  """A temp holding a value every later statement reads is a common
  subexpression an optimizer already found. Recomputing it instead lets the
  two copies take different rewrites and stop looking like one value."""
  kept, ok = rules_kept(rytwin, RECOMPUTE_FIXTURE, "recompute")
  check("un-cse fires where a temp is reused", "un-cse" in kept, str(kept))
  check("and the twins validate", ok)


def test_dead_statements_read_live_locals(rytwin):
  """Statements whose result nothing reads, over locals that are live: they
  have to be generated the way the program's own statements were, or they
  read as padding rather than as work."""
  kept, ok = rules_kept(rytwin, CHAIN_FIXTURE, "chain")
  check("dead-statement fires", "dead-statement" in kept, str(kept))
  check("and the twins validate", ok)


def test_a_value_is_routed_through_memory(rytwin, symiri):
  """Storing a value and loading it back turns dataflow into memory flow,
  which an optimizer has to alias-analyse to undo."""
  kept, ok = rules_kept(rytwin, CHAIN_FIXTURE, "chain")
  check("memory-routing fires", "memory-routing" in kept, str(kept))
  check("and the twins validate", ok)
  bodies = [b for b in twin_text(rytwin, CHAIN_FIXTURE, "chain") if "store" in b]
  check("some body goes through memory", bodies, "")
  if bodies:
    check("and reads it back", "load" in bodies[0], bodies[0][:200])


def test_scratch_declarations_are_shuffled(rytwin):
  """The engine allocates its scratch in the order it rewrites, so declaring
  them in that order hands a reader the order the rewrites happened in.
  (Renaming them is not on offer: the caller tells its own scratch from the
  program's state by the prefix.)"""
  orders = []
  for body in twin_declarations(rytwin, RECOMPUTE_FIXTURE, "recompute"):
    nums = [int(n) for n in re.findall(r"%__aok?(\d+)", body)]
    if len(nums) >= 3:
      orders.append(nums)
  check("some twin declares three or more scratch locals", orders, "")
  check(
    "and at least one does not declare them in allocation order",
    any(ns != sorted(ns) for ns in orders),
    str(orders[:2]),
  )


# Family D reads the same state-set table that certified the guard, so these
# fixtures are about what the box *knows*: a local pinned to a value, a value
# the box bounds, and a leaf the guard never mentions at all.
LICENSED_FIXTURE = """// SOLVED: %p0=3
fun @licensed(%p0: i32) : i32 {
  let mut %k: i32 = 0;
  let mut %m: i32 = 0;
  let mut %r: i32 = 0;
  let mut %s: i32 = 0;
^entry:
  %k = 7;
  %m = 255;
  br ^work;
^work:
  %r = %p0 & %m;
  %s = %r + 7;
  %s = 7 * %s;
  %s = %s + 7;
  %s = %s + %k;
  br ^done;
^done:
  ret %s;
}
"""

# A literal no local is pinned to, so the only way to spell it is out of a
# bound: %m is 255 on the box, and 255 >>> 5 is 7 for every state in it.
BOUND_FIXTURE = """// SOLVED: %p0=3
fun @bounded(%p0: i32) : i32 {
  let mut %m: i32 = 0;
  let mut %s: i32 = 0;
^entry:
  %m = 255;
  br ^work;
^work:
  %s = %p0 + 7;
  %s = %s + 7;
  %s = %s + %m;
  br ^done;
^done:
  ret %s;
}
"""

# %q is read by nothing the trace does, so the box frees it and the guard
# never mentions it — which is what makes a dependency on it convincing.
FREE_LEAF_FIXTURE = """// SOLVED: %p0=3,%q=5
fun @freeleaf(%p0: i32, %q: i32) : i32 {
  let mut %r: i32 = 0;
^entry:
  br %p0 > 0, ^work, ^done;
^work:
  %r = %p0 + 1;
  br ^done;
^done:
  ret %r;
}
"""


def test_a_pinned_value_stands_in_for_a_literal(rytwin):
  """A value the table pins to one number is that number on every state the
  guard admits, so a literal can be spelled as a read of it — and the body
  starts looking state-dependent where it is not."""
  kept, ok = rules_kept(rytwin, LICENSED_FIXTURE, "licensed")
  check("known-constant fires", "known-constant" in kept, str(kept))
  check("and the twins validate", ok)


def test_a_bounded_value_licenses_a_mask(rytwin):
  """`%x & 255` is the identity exactly where the table says %x is in
  [0, 255]. Nothing but the guard makes that true."""
  kept, ok = rules_kept(rytwin, LICENSED_FIXTURE, "licensed")
  check("range-mask fires", "range-mask" in kept, str(kept))
  check("and the twins validate", ok)


def test_a_true_condition_hides_an_arm(rytwin):
  """A comparison the table settles over the whole box wraps a value in a
  select whose other arm never runs — so the arm can hold anything, and a
  reader has to prove the condition to know that."""
  kept, ok = rules_kept(rytwin, LICENSED_FIXTURE, "licensed")
  check("true-select fires", "true-select" in kept, str(kept))
  check("and the twins validate", ok)


def test_a_free_leaf_is_faked_into_the_body(rytwin):
  """The guard does not mention a free leaf at all, so a body that reads one
  reads as an under-fitted defensive check rather than a memo — while the
  value it computes cannot depend on it."""
  kept, ok = rules_kept(rytwin, FREE_LEAF_FIXTURE, "freeleaf")
  check("free-leaf-mix fires", "free-leaf-mix" in kept, str(kept))
  check("and the twins validate", ok)


def test_a_bound_makes_a_constant_out_of_a_range(rytwin):
  """A value the box bounds tightly enough is constant under a shift: every
  state the guard admits sends %x >>> k to the same number, so a literal can
  be spelled that way and is constant only *because* of the guard."""
  kept, ok = rules_kept(rytwin, BOUND_FIXTURE, "bounded")
  check("bound-derived fires", "bound-derived" in kept, str(kept))
  check("and the twins validate", ok)


# A select whose condition and arms are all spellable as `cmp` operands, so
# the branchless form has somewhere to put them.
SELECT_FIXTURE = """// SOLVED: %p0=3
fun @sel(%p0: i32) : i32 {
  let mut %a: i32 = 0;
  let mut %b: i32 = 0;
  let mut %r: i32 = 0;
^entry:
  %a = 10;
  %b = 20;
  br ^work;
^work:
  %r = select %p0 > 0, %a, %b;
  %r = %r + %p0;
  br ^done;
^done:
  ret %r;
}
"""

INTRINSIC_FIXTURE = """// SOLVED: %p0=3
intrinsic @min(%a: i32, %b: i32) : i32;

fun @intr(%p0: i32) : i32 {
  let mut %b: i32 = 0;
  let mut %r: i32 = 0;
^entry:
  %b = 20;
  br ^work;
^work:
  %r = call @min(%p0, %b);
  %b = %b + %p0;
  br ^done;
^done:
  ret %r;
}
"""


def test_a_select_becomes_branchless(rytwin):
  """`select c, a, b` is `b + ((a - b) & (c as iN))` — i1 true is all-ones
  (spec §6.4), so the cast is a ready-made mask. The choice stops being a
  choice and becomes arithmetic."""
  kept, ok = rules_kept(rytwin, SELECT_FIXTURE, "sel")
  check("select-to-mask fires", "select-to-mask" in kept, str(kept))
  check("and the twins validate", ok)


# Three comparisons, since `swap-compare`, `split-compare` and
# `compare-to-diff` all want the same statement and only one can have it. The
# trailing sum is what keeps the operands bounded: comparisons never trap, so
# a body of nothing but comparisons frees every leaf and licenses nothing.
CMP3_FIXTURE = """// SOLVED: %p0=3
fun @cmp3(%p0: i32) : i32 {
  let mut %b: i32 = 0;
  let mut %c: i1 = 0;
  let mut %d: i1 = 0;
  let mut %e: i1 = 0;
  let mut %r: i32 = 0;
^entry:
  %b = 5;
  br ^work;
^work:
  %c = cmp < %p0, %b;
  %d = cmp > %p0, %b;
  %e = cmp != %p0, %b;
  %r = %p0 + %b;
  br ^done;
^done:
  ret %r;
}
"""


def test_a_comparison_splits_in_two(rytwin):
  """`a < b` is `(a <= b) & (a != b)`, and every relation has such a pair.
  Trap-free, since nothing but `cmp` and a bitwise op is introduced."""
  kept, ok = rules_kept(rytwin, CMP3_FIXTURE, "cmp3")
  check("split-compare fires", "split-compare" in kept, str(kept))
  check("and the twins validate", ok)


def test_a_comparison_becomes_a_difference(rytwin):
  """`a < b` is `a - b < 0` exactly where the subtraction does not overflow,
  which is a fact about the box rather than about the operands."""
  kept, ok = rules_kept(rytwin, CMP3_FIXTURE, "cmp3")
  check("compare-to-diff fires", "compare-to-diff" in kept, str(kept))
  check("and the twins validate", ok)


def test_an_intrinsic_is_written_out(rytwin, symiri):
  """An intrinsic is a name for something the program could have spelled
  itself. Spelling it out removes the name the reader was going to match on —
  and the identity is the intrinsic's own documented one."""
  kept, ok = rules_kept(rytwin, INTRINSIC_FIXTURE, "intr")
  check("intrinsic-expand fires", "intrinsic-expand" in kept, str(kept))
  check("and the twins validate", ok)


# A pointer read back out of memory has no target the pass can name, so what
# it addresses is unknown and the division that reads it cannot be proven.
# The body is still the region's own statements, and rules that introduce no
# trapping operation are still identities.
UNPROVEN_FIXTURE = PTR_PTR_FIXTURE.replace("@ptrptr", "@unproven")


# %a holds an undef leaf, so no guard can state what it is — and the region
# reads it, so the region cannot be twinned. Saying *that* is the point: the
# bare name told a reader nothing about which of the several reasons applied.
UNDEF_LEAF_FIXTURE = """// SOLVED: %p0=3
fun @undefleaf(%p0: i32) : i32 {
  let mut %a: [2] i32 = {undef, 7};
  let mut %r: i32 = 0;
^entry:
  br ^work;
^work:
  %r = %p0 + %a[1];
  br ^done;
^done:
  ret %r;
}
"""


def test_an_unproven_body_still_gets_trap_free_rewrites(rytwin):
  """A body the state-set pass cannot prove used to be left exactly as the
  region wrote it — the worst case for a twin, since it is then a copy. The
  rules that introduce no trapping operation need no proof, so those still
  apply and the rest stand down."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, _, _ = graft_log(rytwin, d, UNPROVEN_FIXTURE, "unproven", ["--validate"])
    check("unproven fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    check("the pass declines the trace", verdict_note(lines[0]) != "ok", lines[0])
    m = re.search(r"(\d+) rewrites \((\d+) undone", lines[0])
    check("it is still rewritten", m and int(m.group(1)) > 0, lines[0])
    check("the log says why the choice was limited", "trap-free" in lines[0], lines[0])
    # Nothing that could introduce a trap: no proof exists to clear one.
    for tier1 in ("memory-routing", "dead-statement", "mba-bit-split", "mul-to-shift"):
      check(f"{tier1} stands down", tier1 not in lines[0], lines[0])
    check("and it validates", "validated: OK" in r.stdout, r.stdout[:200])


def test_a_rejection_says_what_it_could_not_guard(rytwin):
  """`unguardable state: %a` names the root but not the reason, and there are
  several — undef leaves, opaque pointers, an immutable aggregate. The reason
  is the part a reader cannot work out from the program alone."""
  with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "undefleaf.sir")
    open(p1, "w").write(UNDEF_LEAF_FIXTURE)
    p2 = os.path.join(d, "undefleaf.p2.sir")
    r = run([rytwin, p1, "--p-twin", "1.0", "--seed", "3", "-v", "-o", p2])
    out = r.stderr + r.stdout
    check("the root is named", "unguardable state: %a" in out, out[:300])
    m = re.search(r"unguardable state: %a \(([^)]*)\)", out)
    check("and the reason with it", m is not None, out[:300])
    if m:
      check("which is the undef leaf", "undef" in m.group(1), m.group(1))


# The same three comparisons at a width that is not i32. A literal written
# straight into a `cmp` is an i32 literal, so a rule that compares against a
# bare zero emits a program the checker rejects at every other width — and
# rytwin aborts on its own output rather than producing anything.
NARROW_CMP_FIXTURE = """// SOLVED: %p0=3
fun @cmpnarrow(%p0: i16) : i16 {
  let mut %b: i16 = 0;
  let mut %c: i1 = 0;
  let mut %d: i1 = 0;
  let mut %e: i1 = 0;
  let mut %r: i16 = 0;
^entry:
  %b = 5;
  br ^work;
^work:
  %c = cmp < %p0, %b;
  %d = cmp > %p0, %b;
  %e = cmp != %p0, %b;
  %r = %p0 + %b;
  br ^done;
^done:
  ret %r;
}
"""


def test_comparisons_rewrite_at_any_width(rytwin):
  """Every operand of a `cmp` shares one width (spec §6.5), and a literal
  written into one is i32 whatever it is compared against. A rule that needs
  a zero has to give it the compared type."""
  for seed in ("1", "2", "3", "4", "5", "6"):
    with tempfile.TemporaryDirectory() as d:
      p1 = os.path.join(d, "cmpnarrow.sir")
      open(p1, "w").write(NARROW_CMP_FIXTURE)
      p2 = os.path.join(d, "cmpnarrow.p2.sir")
      r = run([rytwin, p1, "--p-twin", "1.0", "--seed", seed, "--validate", "-o", p2])
      ok = r.returncode == 0 and "validated: OK" in r.stdout
      check(f"an i16 comparison twins at seed {seed}", ok, (r.stderr + r.stdout)[:200])


# Vectors were unknown from the start, so a region touching one proved
# nothing and its guard could not open: every leaf pinned, the twin admitted
# one state. A vector statement is N scalar statements, one per lane, and
# that is all the pass needs to follow it.
VEC_FIXTURE = """// SOLVED: %p0=3
fun @vecadd(%p0: i32) : i32 {
  let mut %v: <4> i32 = {1, 2, 3, 4};
  let mut %w: <4> i32 = {5, 6, 7, 8};
  let mut %r: i32 = 0;
^entry:
  br ^work;
^work:
  %v = %v + %w;
  %r = %v[1];
  %r = %r + %p0;
  br ^done;
^done:
  ret %r;
}
"""

VEC_WRITE_FIXTURE = """// SOLVED: %p0=3
fun @veclane(%p0: i32) : i32 {
  let mut %v: <4> i32 = {1, 2, 3, 4};
  let mut %r: i32 = 0;
^entry:
  br ^work;
^work:
  %v[2] = %p0;
  %v = 2 * %v;
  %r = %v[2];
  %r = %r + %p0;
  br ^done;
^done:
  ret %r;
}
"""

# The lane written is not known here, so the write could have landed on any
# of them and nothing about the vector survives it.
VEC_DYN_FIXTURE = """// SOLVED: %p0=3
intrinsic @min(%a: i32, %b: i32) : i32;

fun @vecdyn(%p0: i32) : i32 {
  let mut %v: <4> i32 = {1, 2, 3, 4};
  let mut %i: i32 = 0;
  let mut %r: i32 = 0;
^entry:
  %i = call @min(%p0, 3);
  br ^work;
^work:
  %v[%i] = %p0;
  %r = %v[0];
  %r = %r + %p0;
  br ^done;
^done:
  ret %r;
}
"""


def test_state_set_pass_follows_vector_lanes(rytwin):
  """Lane-wise arithmetic is arithmetic: `%v = %v + %w` is one addition per
  lane, each with the same overflow question as a scalar one. Following it
  is what lets a region that touches a vector prove anything at all."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, _, _ = graft_log(rytwin, d, VEC_FIXTURE, "vecadd", ["--validate"])
    check("vector fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    check("the vector trace is proven", verdict_note(lines[0]) == "ok", lines[0])
    counts = box_counts(lines[0])
    check("and the guard opens", counts and counts[0] + counts[1] > 0, str(counts))


def test_state_set_pass_follows_a_written_lane(rytwin):
  """A lane written by name is a cell like any other, and the vector it sits
  in keeps its other lanes."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, _, _ = graft_log(rytwin, d, VEC_WRITE_FIXTURE, "veclane", ["--validate"])
    check("lane fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    check("the lane write is followed", verdict_note(lines[0]) == "ok", lines[0])
    counts = box_counts(lines[0])
    check("and the guard opens", counts and counts[0] + counts[1] > 0, str(counts))


def test_an_unknown_lane_index_claims_nothing(rytwin):
  """A write through a lane index the pass cannot pin could have landed on
  any lane, so nothing about the vector survives it. Declining is the point:
  the alternative is a guard that admits states the twin gets wrong."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, _, _ = graft_log(rytwin, d, VEC_DYN_FIXTURE, "vecdyn", ["--validate"])
    check("dynamic-lane fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    check("the pass declines", verdict_note(lines[0]) != "ok", lines[0])


def test_vector_twin_agrees_off_the_profile(rytwin, symiri):
  """The widened guard admits inputs the profile never saw, and the twin has
  to agree with the region on those too."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, p1, p2 = graft_log(rytwin, d, VEC_FIXTURE, "vecadd", ["--validate"])
    check("vector fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    check("--validate agrees", "validated: OK" in r.stdout, r.stdout[:200])
    for arg in ("3", "-11", "40000"):
      r1 = symiri_result(symiri, p1, "@vecadd", [arg])
      r2 = symiri_result(symiri, p2, "@vecadd", [arg])
      check(f"same answer for {arg}", r1[1:] == r2[1:], f"{r1} vs {r2}")


# `& | ^ / %` of two values the pass merely bounds used to widen to unknown,
# and an unknown poisons every addition downstream of it — which is what most
# pool regions were actually failing on. Each fixture puts one operator
# between a bounded value and the sum that reads it.
def scalar_bound_fixture(name, setup, op):
  return f"""// SOLVED: %p0=8
fun @{name}(%p0: i32) : i32 {{
  let mut %m: i32 = 0;
  let mut %d: i32 = 0;
  let mut %a: i32 = 0;
  let mut %r: i32 = 0;
^entry:
  %m = 255;
  %d = 3;
  br ^work;
^work:
{setup}
  %r = %a {op};
  %r = %r + %p0;
  br ^done;
^done:
  ret %r;
}}
"""


# %a is in [0, 255] by the mask, so `%a | %m` and `%a ^ %m` cannot exceed the
# next power of two and cannot go negative.
OR_BOUND_FIXTURE = scalar_bound_fixture("orbound", "  %a = %p0 & %m;", "| %m")
XOR_BOUND_FIXTURE = scalar_bound_fixture("xorbound", "  %a = %p0 & %m;", "^ %m")
# A quotient is no larger in magnitude than its dividend, and a remainder is
# smaller than its divisor.
DIV_BOUND_FIXTURE = scalar_bound_fixture("divbound", "  %a = %p0;", "/ %d")
MOD_BOUND_FIXTURE = scalar_bound_fixture("modbound", "  %a = %p0;", "% %d")

# The same disjunction on a value the guard would have to admit negative:
# `x | y` says nothing about magnitude once a sign bit is in play, so the
# leaf cannot widen at all.
OPEN_OR_FIXTURE = scalar_bound_fixture("openor", "  %a = %p0;", "| %m").replace(
  "%p0=8", "%p0=-8"
)


def bounded_opens(rytwin, fixture, name):
  """(ranged leaves, widest radius) for `fixture`. At the point box every
  value is a constant and folds, so what a transfer function buys is not a
  proof *there* — it is whether the leaf can be widened away from it."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, _, _ = graft_log(rytwin, d, fixture, name, ["--validate"])
    if not lines:
      check(f"{name} twinned", False, r.stderr[:200])
      return None, None
    counts = box_counts(lines[0])
    widest = box_widest(lines[0])
    return (counts[1] if counts else None), (widest[1] if widest else 0)


def test_state_set_pass_bounds_a_disjunction(rytwin):
  """`x | y` only sets bits, so for two non-negative ranges it is at least
  the larger operand and at most the next power of two."""
  ranged, radius = bounded_opens(rytwin, OR_BOUND_FIXTURE, "orbound")
  if ranged is None:
    return
  check("the disjunction lets a leaf widen", ranged >= 1, str(ranged))
  check("and it widens far", radius > 1000, str(radius))


def test_state_set_pass_bounds_an_exclusive_or(rytwin):
  """`x ^ y` cannot set a bit neither operand has, so the same ceiling holds
  and the result stays non-negative."""
  ranged, radius = bounded_opens(rytwin, XOR_BOUND_FIXTURE, "xorbound")
  if ranged is None:
    return
  check("the exclusive or lets a leaf widen", ranged >= 1, str(ranged))
  check("and it widens far", radius > 1000, str(radius))


def test_state_set_pass_bounds_a_quotient(rytwin):
  """Dividing cannot grow a value: with the divisor away from zero, the
  quotient is bounded by the dividend over the smallest divisor there is."""
  ranged, radius = bounded_opens(rytwin, DIV_BOUND_FIXTURE, "divbound")
  if ranged is None:
    return
  check("the quotient lets a leaf widen", ranged >= 1, str(ranged))
  check("and it widens far", radius > 1000, str(radius))


def test_state_set_pass_bounds_a_remainder(rytwin):
  """A remainder is smaller than its divisor and no larger than its dividend,
  and it takes the dividend's sign."""
  ranged, radius = bounded_opens(rytwin, MOD_BOUND_FIXTURE, "modbound")
  if ranged is None:
    return
  check("the remainder lets a leaf widen", ranged >= 1, str(ranged))
  check("and it widens far", radius > 1000, str(radius))


def test_a_signed_disjunction_claims_nothing(rytwin):
  """With a sign bit possibly in play there is no ceiling to claim, and the
  pass must decline rather than invent one."""
  ranged, _ = bounded_opens(rytwin, OPEN_OR_FIXTURE, "openor")
  if ranged is None:
    return
  check("nothing is claimed about a signed disjunction", ranged == 0, str(ranged))


# An intrinsic result was unknown to the pass, and an unknown poisons every
# sum that reads it. Its arguments are often pinned, though, and then the
# result is one value — one the interpreter can say, since it is the authority
# on what an intrinsic computes.
INTRINSIC_FOLD_FIXTURE = """// SOLVED: %p0=8
intrinsic @min(%a: i32, %b: i32) : i32;

fun @foldmin(%p0: i32) : i32 {
  let mut %k: i32 = 0;
  let mut %j: i32 = 0;
  let mut %r: i32 = 0;
^entry:
  %k = 20;
  %j = 30;
  br ^work;
^work:
  %r = call @min(%k, %j);
  %r = %r + %p0;
  br ^done;
^done:
  ret %r;
}
"""

# The same call with an argument the box will widen: no single value to fold
# to, so the pass must go back to knowing nothing about it.
INTRINSIC_OPEN_FIXTURE = INTRINSIC_FOLD_FIXTURE.replace("@foldmin", "@openmin").replace(
  "%r = call @min(%k, %j);", "%r = call @min(%p0, %j);"
)


def test_a_pinned_intrinsic_is_folded(rytwin):
  """With every argument pinned, an intrinsic call has one result, and the
  interpreter is asked for it rather than the pass guessing. Without that the
  sum reading it cannot be proven and the guard cannot open."""
  ranged, radius = bounded_opens(rytwin, INTRINSIC_FOLD_FIXTURE, "foldmin")
  if ranged is None:
    return
  check("the folded call lets a leaf widen", ranged >= 1, str(ranged))
  check("and it widens far", radius > 1000, str(radius))


def test_an_open_intrinsic_argument_folds_nothing(rytwin):
  """An argument the box widens has no single value, so neither has the
  result. Declining is the point."""
  ranged, _ = bounded_opens(rytwin, INTRINSIC_OPEN_FIXTURE, "openmin")
  if ranged is None:
    return
  check("nothing is claimed about an unpinned call", ranged == 0, str(ranged))


def test_a_folded_call_agrees_off_the_profile(rytwin, symiri):
  """The guard the fold opens admits inputs the profile never saw, and the
  twin has to agree with the region on those too."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, p1, p2 = graft_log(
      rytwin, d, INTRINSIC_FOLD_FIXTURE, "foldmin", ["--validate"]
    )
    check("folding fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    check("--validate agrees", "validated: OK" in r.stdout, r.stdout[:200])
    for arg in ("8", "-4000", "123456"):
      r1 = symiri_result(symiri, p1, "@foldmin", [arg])
      r2 = symiri_result(symiri, p2, "@foldmin", [arg])
      check(f"same answer for {arg}", r1[1:] == r2[1:], f"{r1} vs {r2}")


# %p is assigned in ^setup, not in the function's entry block — the shape
# rysmith emits whenever a pointer is set up where it is used rather than up
# front. Every path to ^work assigns it, so a guard there may read it; a test
# that only looks at the entry block cannot tell.
LATE_INIT_FIXTURE = """// SOLVED: %p0=3
fun @lateinit(%p0: i32) : i32 {
  let mut %a: [2] i32 = {5, 7};
  let mut %p: ptr i32 = undef;
  let mut %r: i32 = 0;
^entry:
  br ^setup;
^setup:
  %p = addr %a[1];
  br ^work;
^work:
  %r = load %p;
  %r = %r + %p0;
  br ^done;
^done:
  ret %r;
}
"""

# The same pointer assigned on only one side of a branch: the region below
# the join is reached with it uninitialized, and a guard reading it there
# would be a program the checker rejects.
MAYBE_INIT_FIXTURE = """// SOLVED: %p0=3
fun @maybeinit(%p0: i32) : i32 {
  let mut %a: [2] i32 = {5, 7};
  let mut %p: ptr i32 = undef;
  let mut %r: i32 = 0;
^entry:
  br %p0 > 0, ^setup, ^skip;
^setup:
  %p = addr %a[1];
  br ^work;
^skip:
  %r = 1;
  br ^work;
^work:
  %r = %r + %p0;
  br ^done;
^done:
  ret %r;
}
"""


def test_a_root_initialized_before_the_region_is_guardable(rytwin, symiri):
  """A root assigned anywhere that dominates the region can be read by the
  guard — which is what the frontend's own must-init analysis says, and the
  guard has to agree with it exactly or it emits programs the checker
  rejects. Testing only the function's entry block turned that into "assigned
  early or not at all"."""
  with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "lateinit.sir")
    open(p1, "w").write(LATE_INIT_FIXTURE)
    p2 = os.path.join(d, "lateinit.p2.sir")
    r = run(
      [rytwin, p1, "--p-twin", "1.0", "--seed", "3", "-v", "--validate", "-o", p2]
    )
    check("the fixture twins", r.returncode == 0, (r.stderr + r.stdout)[:200])
    if r.returncode != 0:
      return
    check(
      "the region below the assignment is not rejected",
      "unguardable state: %p" not in r.stderr,
      [ln for ln in r.stderr.splitlines() if "unguardable" in ln][:1],
    )
    check("and it validates", "validated: OK" in r.stdout, r.stdout[:200])
    for arg in ("3", "-9"):
      r1 = symiri_result(symiri, p1, "@lateinit", [arg])
      r2 = symiri_result(symiri, p2, "@lateinit", [arg])
      check(f"same answer for {arg}", r1[1:] == r2[1:], f"{r1} vs {r2}")


def test_a_root_initialized_on_one_path_is_not(rytwin):
  """Assigned on one side of a branch is not assigned: below the join the
  checker cannot prove a read is legal, so neither may the guard."""
  with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "maybeinit.sir")
    open(p1, "w").write(MAYBE_INIT_FIXTURE)
    p2 = os.path.join(d, "maybeinit.p2.sir")
    r = run(
      [rytwin, p1, "--p-twin", "1.0", "--seed", "3", "-v", "--validate", "-o", p2]
    )
    log = r.stderr
    joined = [ln for ln in log.splitlines() if "^work" in ln]
    check("the join region is considered", joined, log[:200])
    if joined:
      check(
        "and refused for the root only one path assigns",
        "unguardable state: %p" in joined[0] or "skipped" in joined[0],
        joined[0],
      )
    if r.returncode == 0:
      check(
        "what is emitted still validates", "validated: OK" in r.stdout, r.stdout[:200]
      )


# %q is multiplied by 100000 before the sum, so a step of its own costs the
# sum a hundred thousand times what a step of %p0 costs. Lockstep advances
# them together and a refusal freezes everything that failure depended on, so
# %q hitting its limit takes %p0 with it — at a fraction of the width %p0
# could have had on its own.
LOCKSTEP_FIXTURE = """// SOLVED: %p0=3,%q=2
fun @lockstep(%p0: i32, %q: i32) : i32 {
  let mut %k: i32 = 0;
  let mut %big: i32 = 0;
  let mut %r: i32 = 0;
^entry:
  %k = 100000;
  br ^work;
^work:
  %big = %q * %k;
  %r = %big + %p0;
  br ^done;
^done:
  ret %r;
}
"""


def test_a_leaf_is_not_held_back_by_another(rytwin, symiri):
  """A leaf frozen for someone else's refusal keeps whatever width the
  lockstep had reached, which can be orders of magnitude short of what it can
  prove alone. Growing each leaf on its own afterwards recovers that, and
  every step of it is proved rather than argued."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, p1, p2 = graft_log(
      rytwin, d, LOCKSTEP_FIXTURE, "lockstep", ["--validate"]
    )
    check("lockstep fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    widest = box_widest(lines[0])
    check(
      "the independent leaf is the widest", widest and widest[0] == "%p0", str(widest)
    )
    check(
      "and it reaches what it can prove alone",
      widest and widest[1] > 100000000,
      str(widest),
    )
    check("the guard still validates", "validated: OK" in r.stdout, r.stdout[:200])
    for arg in ("3", "-2000000"):
      r1 = symiri_result(symiri, p1, "@lockstep", [arg, "2"])
      r2 = symiri_result(symiri, p2, "@lockstep", [arg, "2"])
      check(f"same answer for {arg}", r1[1:] == r2[1:], f"{r1} vs {r2}")


def test_box_is_deterministic_for_a_seed(rytwin):
  """The search is seeded, so two runs agree — the trim that keeps guards
  from being identical is drawn from the same stream, not from chance."""
  with tempfile.TemporaryDirectory() as d:
    first = graft_log(rytwin, d, CHAIN_FIXTURE, "chain", [])[1]
  with tempfile.TemporaryDirectory() as d:
    second = graft_log(rytwin, d, CHAIN_FIXTURE, "chain", [])[1]
    check("both runs twinned", first and second, "")
    if first and second:
      check(
        "same seed, same box",
        box_counts(first[0]) == box_counts(second[0])
        and box_widest(first[0]) == box_widest(second[0]),
        f"{first[0]} vs {second[0]}",
      )


def test_state_set_pass_follows_floats(rytwin, symiri):
  """A float leaf the guard pins is a constant, so float arithmetic, a branch
  on a float, and a cast back to an integer are all followable — none of them
  should stop a region being proven."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, p1, p2 = graft_log(rytwin, d, FLOAT_FIXTURE, "floats", ["--validate"])
    check("float fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    check("float work is proven", verdict_note(lines[0]) == "ok", lines[0])
    check("the float branch is settled", "1 path cond" in lines[0], lines[0])
    r1 = symiri_result(symiri, p1, "@floats", ["3"])
    r2 = symiri_result(symiri, p2, "@floats", ["3"])
    check("float program equivalent", r1[1:] == r2[1:], f"{r1} vs {r2}")


def test_state_set_pass_seeds_pointers_from_the_profile(rytwin):
  """A pointer set up before the region has no `addr` inside it to follow.
  The profile recorded where it points, so the load through it is still
  known."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, _, _ = graft_log(rytwin, d, SEED_PTR_FIXTURE, "seedptr", ["--validate"])
    check("seeded-pointer fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if not lines:
      return
    check(
      "the region begins after the pointer setup",
      any("^work" in ln for ln in lines),
      str(lines),
    )
    hit = [ln for ln in lines if "^work" in ln]
    if hit:
      check(
        "a pointer from the profile is followed", verdict_note(hit[0]) == "ok", hit[0]
      )


def test_state_set_pass_follows_pointer_arithmetic(rytwin):
  """Pointer arithmetic by a known amount lands on a known cell, so the
  region that walks an array is provable too."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, _, _ = graft_log(rytwin, d, PTR_WALK_FIXTURE, "ptrwalk", ["--validate"])
    check("pointer-walk fixture twinned", r.returncode == 0 and lines, r.stderr[:200])
    if lines:
      check("a walked pointer is proven", verdict_note(lines[0]) == "ok", lines[0])


def test_state_set_pass_reports_what_it_cannot_prove(rytwin, symiri):
  """A pointer that itself comes out of memory has no known target, so what
  it addresses stays unknown — and saying so is the point, since the guard
  cannot widen past it. The graft is unaffected either way."""
  with tempfile.TemporaryDirectory() as d:
    r, lines, p1, p2 = graft_log(rytwin, d, PTR_PTR_FIXTURE, "ptrptr", ["--validate"])
    check(
      "indirect-pointer fixture twinned", r.returncode == 0 and lines, r.stderr[:200]
    )
    if not lines:
      return
    check("the pass declines", verdict_note(lines[0]) != "ok", lines[0])
    r1 = symiri_result(symiri, p1, "@ptrptr", ["3"])
    r2 = symiri_result(symiri, p2, "@ptrptr", ["3"])
    check(
      "declining changes nothing about the graft", r1[1:] == r2[1:], f"{r1} vs {r2}"
    )


def test_default_scope_is_region(rytwin):
  """Region is the only unit now, so the ^e twin jumps straight to the
  region exit ^x without being asked to."""
  with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "seqreg.sir")
    open(p1, "w").write(SEQ_FIXTURE)
    p2 = os.path.join(d, "p2.sir")
    r = run([rytwin, p1, "--p-twin", "1.0", "--seed", "3", "-o", p2])
    check("default fixture twinned", r.returncode == 0, r.stderr[:160])
    if r.returncode != 0:
      return
    bodies = twin_block_bodies(open(p2).read())
    check(
      "a twin collapses ^e->^x by default",
      any("br ^x" in b for b in bodies),
      str(bodies)[:300],
    )


def test_scope_region_sequence(rytwin, symiri):
  """Region scope collapses the ^e->^b1 chain: the ^e twin branches straight
  to ^x, ^b1 stays put (for the orig path), and p1 === p2."""
  with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "seqreg.sir")
    open(p1, "w").write(SEQ_FIXTURE)
    p2 = os.path.join(d, "p2.sir")
    r = run(
      [
        rytwin,
        p1,
        "--p-twin",
        "1.0",
        "--seed",
        "3",
        "--validate",
        "-o",
        p2,
      ]
    )
    check("sequence region twinned + validated", r.returncode == 0, r.stderr[:200])
    if r.returncode != 0:
      return
    src = open(p2).read()
    bodies = twin_block_bodies(src)
    check(
      "a twin jumps straight to ^x (skips ^b1)",
      any("br ^x" in b for b in bodies),
      str(bodies),
    )
    check("skipped block ^b1 is preserved for the orig path", "^b1:" in src, "")
    check(
      "sequence p1 === p2",
      symiri_result(symiri, p1, "@seqreg", ["3"])[1:]
      == symiri_result(symiri, p2, "@seqreg", ["3"])[1:],
      "",
    )


def test_scope_region_loop(rytwin, symiri):
  """Region scope collapses an entire loop: a twin branches straight to
  ^done, the loop body ^body is preserved, the twin actually executes (so the
  loop was skipped on the profiled run), and p1 === p2."""
  with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "loopreg.sir")
    open(p1, "w").write(LOOP_FIXTURE)
    p2 = os.path.join(d, "p2.sir")
    r = run(
      [
        rytwin,
        p1,
        "--p-twin",
        "1.0",
        "--seed",
        "3",
        "--validate",
        "-o",
        p2,
      ]
    )
    check("loop region twinned + validated", r.returncode == 0, r.stderr[:200])
    if r.returncode != 0:
      return
    src = open(p2).read()
    bodies = twin_block_bodies(src)
    check(
      "a twin jumps straight to ^done (loop collapsed)",
      any("br ^done" in b for b in bodies),
      str(bodies),
    )
    check("loop body ^body is preserved for the orig path", "^body:" in src, "")
    check(
      "loop twin executes on profiled run",
      "__twin" in dump_trace(symiri, p2, "@loopreg", ["3"]),
      "",
    )
    check(
      "loop p1 === p2",
      symiri_result(symiri, p1, "@loopreg", ["3"])[1:]
      == symiri_result(symiri, p2, "@loopreg", ["3"])[1:],
      "",
    )


def test_scope_default_is_block(rytwin):
  """Without --twin-scope the default is single-block: the ^e twin branches
  to ^e's immediate successor ^b1, never straight to ^x."""
  with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "seqreg.sir")
    open(p1, "w").write(SEQ_FIXTURE)
    p2 = os.path.join(d, "p2.sir")
    r = run([rytwin, p1, "--p-twin", "1.0", "--seed", "3", "-o", p2])
    check("default (block) fixture twinned", r.returncode == 0, r.stderr[:160])
    if r.returncode != 0:
      return
    bodies = twin_block_bodies(open(p2).read())
    check(
      "no twin collapses ^e->^x under block scope",
      not any("br ^x" in b for b in bodies),
      str(bodies),
    )


def test_scope_region_equivalence(rytwin, rysmith, symiri):
  """Region scope preserves p1 === p2 on the profiled input AND other inputs
  across a random pool — the guard still fires only on the entry state."""
  with tempfile.TemporaryDirectory() as d:
    sirs = gen_pool(rysmith, d, n="12", seed="202")
    if not sirs:
      check("region-equivalence setup (rysmith gen)", False, "generation failed")
      return
    equivalence_over_pool(rytwin, symiri, d, sirs, "region", extra=None)


# --- interesting-region selection (--twin-select) -----------------------
#
# `interesting` scores every eligible region (loop iterations collapsed
# dominate, then size / state-diff / fan-in) and grafts the highest-scoring
# non-overlapping ones. Two sequential loops under one dominating entry: the
# entry's whole-function region subsumes both and outscores every inner
# candidate, so it wins the overlap.
TWO_LOOP_FIXTURE = """// SOLVED: %p=1
fun @twoloop(%p: i32) : i32 {
  let mut %i: i32 = 0;
  let mut %j: i32 = 0;
  let mut %s: i32 = 0;
  ^entry:
    br %p != 0, ^a_head, ^mid;
  ^a_head:
    br %i < 4, ^a_body, ^mid;
  ^a_body:
    %s = %s + %i;
    %i = %i + 1;
    br ^a_head;
  ^mid:
    br ^b_head;
  ^b_head:
    br %j < 2, ^b_body, ^done;
  ^b_body:
    %s = %s + %j;
    %j = %j + 1;
    br ^b_head;
  ^done:
    ret %s;
}
"""


def test_select_flag(rytwin):
  """--twin-select interesting is accepted; an unknown value exits non-zero."""
  with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "twoloop.sir")
    open(p1, "w").write(TWO_LOOP_FIXTURE)
    p2 = os.path.join(d, "p2.sir")
    r = run(
      [
        rytwin,
        p1,
        "--p-twin",
        "1.0",
        "--seed",
        "3",
        "--twin-select",
        "interesting",
        "-o",
        p2,
      ]
    )
    check("rytwin accepts --twin-select interesting", r.returncode == 0, r.stderr[:160])
    rb = run([rytwin, p1, "--twin-select", "bogus", "-o", p2 + ".x"])
    check("invalid --twin-select rejected", rb.returncode != 0, f"rc={rb.returncode}")


def test_select_interesting_wins_overlap(rytwin, symiri):
  """The highest-scored region wins the overlap: the entry's whole-function
  region (both loops) is grafted and jumps straight to ^done, the losing
  overlapping candidates are dropped, and p1 === p2."""
  with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "twoloop.sir")
    open(p1, "w").write(TWO_LOOP_FIXTURE)
    p2 = os.path.join(d, "p2.sir")
    r = run(
      [
        rytwin,
        p1,
        "--p-twin",
        "1.0",
        "--seed",
        "3",
        "--twin-select",
        "interesting",
        "--verbose",
        "--validate",
        "-o",
        p2,
      ]
    )
    check(
      "interesting selection twinned + validated", r.returncode == 0, r.stderr[:200]
    )
    if r.returncode != 0:
      return
    log = r.stdout + r.stderr
    check(
      "losing candidates report the overlap",
      "overlaps a selected region" in log,
      log[-300:],
    )
    bodies = twin_block_bodies(open(p2).read())
    check(
      "winning twin collapses both loops to ^done",
      any("br ^done" in b for b in bodies),
      str(bodies),
    )
    check(
      "interesting p1 === p2",
      symiri_result(symiri, p1, "@twoloop", ["1"])[1:]
      == symiri_result(symiri, p2, "@twoloop", ["1"])[1:],
      "",
    )


def test_select_interesting_equivalence(rytwin, rysmith, symiri):
  """Interesting selection preserves p1 === p2 on the profiled input AND
  other inputs across a random pool."""
  with tempfile.TemporaryDirectory() as d:
    sirs = gen_pool(rysmith, d, n="12", seed="202")
    if not sirs:
      check("interesting-equivalence setup (rysmith gen)", False, "generation failed")
      return
    equivalence_over_pool(
      rytwin,
      symiri,
      d,
      sirs,
      "interesting",
      extra=["--twin-select", "interesting"],
    )


def test_select_softmax_probability(rytwin):
  """The per-region twin probability rises with interestingness: at a low
  --p-twin the whole-function region (highest score) gets a strictly higher
  twin probability than a plain one-block region."""
  with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "twoloop.sir")
    open(p1, "w").write(TWO_LOOP_FIXTURE)
    p2 = os.path.join(d, "p2.sir")
    # The probability is deterministic given score/p-twin; the seed only
    # drives the draw. Scan seeds for a run where both regions declined (so
    # both probabilities are printed) and compare them.
    entry_p = body_p = None
    for s in range(1, 60):
      r = run(
        [
          rytwin,
          p1,
          "--p-twin",
          "0.15",
          "--seed",
          str(s),
          "--twin-select",
          "interesting",
          "--verbose",
          "-o",
          p2,
        ]
      )
      log = r.stdout + r.stderr
      me = re.search(r"\^entry: skipped \(twin p=([\d.]+)\)", log)
      mb = re.search(r"\^a_body: skipped \(twin p=([\d.]+)\)", log)
      if me and mb:
        entry_p, body_p = float(me.group(1)), float(mb.group(1))
        break
    check("captured per-region twin probabilities", entry_p is not None, "")
    if entry_p is not None:
      check(
        "harder region gets a higher twin probability",
        entry_p > body_p,
        f"entry={entry_p} body={body_p}",
      )


def twin_block_bodies(src):
  """Return the instruction text of every ^..__twin block in src."""
  bodies = []
  cur = None
  for line in src.splitlines():
    stripped = line.strip()
    if stripped.startswith("^") and stripped.endswith(":"):
      if cur is not None:
        bodies.append("\n".join(cur))
        cur = None
      if stripped.rstrip(":").endswith("__twin"):
        cur = []
      continue
    if cur is not None:
      if stripped.startswith("}"):
        bodies.append("\n".join(cur))
        cur = None
      else:
        cur.append(stripped)
  if cur is not None:
    bodies.append("\n".join(cur))
  return bodies


def has_computed_rhs(body):
  """True if the body computes something from the live-in state rather than
  restating constants: an assignment or a store whose value references a
  local. `addr`/`null` RHSs are pointer reconstruction and don't count."""
  for line in body.splitlines():
    if line.startswith("store ") and "," in line:
      if "%" in line.split(",", 1)[1]:
        return True
      continue
    if " = " in line and not line.startswith("require") and not line.startswith("br"):
      rhs = line.split(" = ", 1)[1]
      if rhs.startswith("addr ") or rhs.startswith("null"):
        continue
      if "%" in rhs:
        return True
  return False


def test_twin_is_generated(rytwin, rysmith):
  """[Stage 3] By default twin blocks are
  solver-generated computations, not bare constant reconstructions: some
  twin RHS references a variable."""
  with tempfile.TemporaryDirectory() as d:
    sirs = gen_pool(rysmith, d, emit_state=False)
    if not sirs:
      check("twin-gen setup (rysmith gen)", False, "generation failed")
      return
    computed = total = 0
    for p1 in sirs:
      p2 = p1[:-4] + ".p2.sir"
      r = run([rytwin, p1, "--p-twin", "1.0", "--seed", "3", "-o", p2])
      if r.returncode != 0:
        continue
      for b in twin_block_bodies(open(p2).read()):
        total += 1
        if has_computed_rhs(b):
          computed += 1
    check(
      "some twin blocks are generated computations",
      computed > 0,
      f"{computed}/{total} twin blocks computed",
    )


def test_twin_fully_concrete(rytwin, rysmith):
  """[Stage 3] Generated twins are concretized before grafting: no %?
  symbol survives into p2."""
  with tempfile.TemporaryDirectory() as d:
    sirs = gen_pool(rysmith, d, emit_state=False)
    if not sirs:
      check("twin-concrete setup (rysmith gen)", False, "generation failed")
      return
    got = first_twinned(rytwin, sirs, extra=None)
    if not got:
      check("twin-concrete setup (twinnable program)", False, "no twin grafted")
      return
    check("no %? symbol left in p2", "%?" not in open(got[1]).read(), "")


def test_twin_body_equivalence(rytwin, rysmith, symiri):
  """[Stage 3] The full equivalence sweep holds with generated twins."""
  with tempfile.TemporaryDirectory() as d:
    sirs = gen_pool(rysmith, d, seed="303", emit_state=False)
    if sirs is None:
      check("twin-gen equivalence setup", False, "generation failed")
      return
    equivalence_over_pool(rytwin, symiri, d, sirs, "twin-gen", extra=None)


def test_twin_requires_stripped(rytwin, rysmith):
  """[Stage 3] The equality requires that pin the generated twin to s' are
  solver-side scaffolding; they are stripped from the graft. UB-safety
  requires may remain and must survive --keep-require compilation."""
  with tempfile.TemporaryDirectory() as d:
    sirs = gen_pool(rysmith, d, emit_state=False, extra=["--emit-main"])
    if not sirs:
      check("requires-stripped setup", False, "generation failed")
      return
    got = first_twinned(
      rytwin,
      sirs,
      extra=["--keep-require", "--target", "c", "--emit-main"],
    )
    if not got:
      check("requires-stripped setup (twinnable program)", False, "no twin")
      return
    bodies = twin_block_bodies(open(got[1]).read())
    eq_requires = [
      ln
      for b in bodies
      for ln in b.splitlines()
      if ln.startswith("require") and "==" in ln
    ]
    check(
      "no equality require left in twin blocks", not eq_requires, str(eq_requires[:3])
    )
    check("--keep-require --target c compiles", os.path.exists(got[1][:-4] + ".c"), "")


def gen_ptr_pool(rysmith, d, n="12", seed="404", emit_state=False):
  """[Stage 4] A pool generated WITH pointers and memory ops (default
  --max-ptr-depth)."""
  cmd = [rysmith, "--emit-desc", "--n-funcs", n, "--seed", seed]
  cmd += ["--n-params", "2", "--n-stmts", "5"]
  if emit_state:
    cmd += ["--emit-state", "pbb"]
  cmd += ["-o", d]
  r = run(cmd)
  if r.returncode != 0:
    return None
  sirs = [f for f in os.listdir(d) if f.endswith(".sir") and "_sym" not in f]
  return [os.path.join(d, s) for s in sorted(sirs)]


def test_ptr_state_json_provenance(rysmith):
  """[Stage 4] The state sidecar records pointer provenance (root + path)
  instead of an opaque ptr marker, and round-trips through the reader."""
  with tempfile.TemporaryDirectory() as d:
    sirs = gen_ptr_pool(rysmith, d, n="4", emit_state=True)
    if not sirs:
      check("ptr-provenance setup (rysmith gen)", False, "generation failed")
      return
    states = [f for f in os.listdir(d) if f.endswith(".state.json")]
    joined = ""
    for f in states:
      joined += open(os.path.join(d, f)).read()
    check("sidecar has ptr entries", '"k":"ptr"' in joined, "")
    check('ptr entries carry provenance ("root")', '"root"' in joined, "")


STORE_FIXTURE = """// SOLVED: %pa0=9
fun @stfix(%pa0: i32) : i32 {
  let mut %a: [3] i32 = {1, 2, 3};
  let mut %p: ptr i32 = undef;
^entry:
  %p = addr %a[1];
  br ^work;
^work:
  store %p, 7 * %pa0;
  %p = addr %a[2];
  br ^exit;
^exit:
  ret %a[1] + %a[0];
}

fun @main() : i32 {
  let mut %r: i32 = undef;
^entry:
  %r = call @stfix(9);
  ret %r;
}
"""


def test_store_block_twinnable(rytwin, symiri):
  """[Stage 4] A block containing a store (and a pointer reassignment) is a
  twin candidate: its effect is the state diff, pointers reconstruct via
  addr, and the twin executes."""
  with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "stfix.sir")
    open(p1, "w").write(STORE_FIXTURE)
    p2 = os.path.join(d, "p2.sir")
    r = run([rytwin, p1, "--p-twin", "1.0", "--seed", "3", "--validate", "-o", p2])
    check("store-bearing fixture twinned", r.returncode == 0, r.stderr[:200])
    if r.returncode != 0:
      return
    src = open(p2).read()
    check("store region grafted", "^entry__twin" in src, "")
    check(
      "the store is replayed in the twin",
      any("store %p," in b for b in twin_block_bodies(src)),
      "",
    )
    check(
      "store twin validated + fired",
      "validated: OK" in r.stdout and "0 twin exec" not in r.stdout,
      r.stdout[:160],
    )


# A pointer that ends the region one past the end of its object. Forming it is
# legal (only dereferencing is not), but there is no `addr <lv>` that names it,
# so the old constant reconstruction could not rebuild the leaf and the region
# was rejected outright. Replaying the trace re-derives it by running the same
# arithmetic, so the region is eligible like any other.
END_PTR_FIXTURE = """// SOLVED: %pa0=9
fun @endptr(%pa0: i32) : i32 {
  let mut %a: [3] i32 = {1, 2, 3};
  let mut %p: ptr i32 = undef;
  let mut %one: i32 = 1;
^entry:
  %p = addr %a[2];
  br ^work;
^work:
  store %p, 7 * %pa0;
  %p = %p + %one;
  br ^exit;
^exit:
  ret %a[2];
}

fun @main() : i32 {
  let mut %r: i32 = undef;
^entry:
  %r = call @endptr(9);
  ret %r;
}
"""


def test_one_past_end_pointer_region(rytwin, symiri):
  """A region whose net effect leaves a pointer one past the end of its
  object is twinnable: the body replays the arithmetic instead of naming
  the address, so nothing has to reconstruct it."""
  with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "endptr.sir")
    open(p1, "w").write(END_PTR_FIXTURE)
    p2 = os.path.join(d, "p2.sir")
    r = run(
      [rytwin, p1, "--p-twin", "1.0", "--seed", "3", "-v", "--validate", "-o", p2]
    )
    check("one-past-end fixture twinned", r.returncode == 0, r.stderr[:200])
    if r.returncode != 0:
      return
    log = r.stderr
    check(
      "no region rejected for an unreconstructable pointer",
      "unreconstructable pointer" not in log,
      [ln for ln in log.splitlines() if "unreconstruct" in ln][:1],
    )
    check(
      "the region spans both blocks (no fallback to one)",
      "2 blk" in log and "fell back" not in log,
      [ln for ln in log.splitlines() if "grafted" in ln][:2],
    )
    body = "\n".join(twin_block_bodies(open(p2).read()))
    check("the twin replays the store", "store %p," in body, body[:200])
    # The bump is replayed rather than the address named; which local holds
    # the 1 is the disguise's business, so only the arithmetic on %p is the
    # claim here.
    check("the twin replays the pointer bump", "%p = %p + " in body, body[:200])
    r1 = symiri_result(symiri, p1, "@main", [])
    r2 = symiri_result(symiri, p2, "@main", [])
    check("one-past-end program equivalent", r1[1:] == r2[1:], f"{r1} vs {r2}")


MEM_OP_RE = re.compile(r"\b(load|store|addr|ptrindex|ptrfield)\b")


def orig_block_bodies(src):
  """Instruction text of every ^..__orig block in src."""
  bodies = []
  cur = None
  for line in src.splitlines():
    stripped = line.strip()
    if stripped.startswith("^") and stripped.endswith(":"):
      if cur is not None:
        bodies.append("\n".join(cur))
        cur = None
      if stripped.rstrip(":").endswith("__orig"):
        cur = []
      continue
    if cur is not None:
      if stripped.startswith("}"):
        bodies.append("\n".join(cur))
        cur = None
      else:
        cur.append(stripped)
  if cur is not None:
    bodies.append("\n".join(cur))
  return bodies


def test_equivalence_with_pointers(rytwin, rysmith, symiri):
  """[Stage 4] The full equivalence sweep holds on programs generated WITH
  pointers and memory operations, and memory-op blocks are actually being
  twinned (not just the pointer-free remnant)."""
  with tempfile.TemporaryDirectory() as d:
    sirs = gen_ptr_pool(rysmith, d)
    if sirs is None:
      check("ptr equivalence setup", False, "generation failed")
      return
    equivalence_over_pool(rytwin, symiri, d, sirs, "ptr")
    mem_twins = 0
    for p1 in sirs:
      p2 = os.path.join(d, os.path.basename(p1)[:-4] + ".p2.sir")
      if not os.path.exists(p2):
        continue
      for b in orig_block_bodies(open(p2).read()):
        if MEM_OP_RE.search(b):
          mem_twins += 1
    check("memory-op blocks get twinned", mem_twins > 0, f"mem twins={mem_twins}")


def test_guard_covers_ptr_leaves(rytwin, rysmith):
  """[Stage 4] Pointer-valued state crosses into the guard: the caller
  passes the pointer and its addr-reconstructed expected value, compared
  with `==` (defined cross-object)."""
  with tempfile.TemporaryDirectory() as d:
    sirs = gen_ptr_pool(rysmith, d)
    if not sirs:
      check("ptr-guard setup (rysmith gen)", False, "generation failed")
      return
    saw_expected_ptr = False
    for p1 in sirs:
      p2 = p1[:-4] + ".p2.sir"
      r = run([rytwin, p1, "--p-twin", "1.0", "--seed", "3", "-o", p2])
      if r.returncode != 0:
        continue
      # Expected-pointer params are named %__e<k> by the guard builder.
      for _, params in GUARD_FN_RE.findall(open(p2).read()):
        if "%__e" in params:
          saw_expected_ptr = True
    check("guard takes addr-reconstructed expected-ptr params", saw_expected_ptr, "")


def test_ptr_program_targets(rytwin, symiri):
  """[Stage 4] The store-bearing fixture survives the backends: its twin
  compiles to C and wasm, and the C binary computes p1's result."""
  import shutil

  with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "stfix.sir")
    open(p1, "w").write(STORE_FIXTURE)
    p2 = os.path.join(d, "p2.sir")
    r = run(
      [
        rytwin,
        p1,
        "--p-twin",
        "1.0",
        "--seed",
        "3",
        "--target",
        "c",
        "--emit-main",
        "-o",
        p2,
      ]
    )
    check(
      "store fixture compiles to C",
      r.returncode == 0 and os.path.exists(p2[:-4] + ".c"),
      r.stderr[:200],
    )
    r = run(
      [
        rytwin,
        p1,
        "--p-twin",
        "1.0",
        "--seed",
        "3",
        "--target",
        "wasm",
        "--emit-main",
        "-o",
        p2[:-4] + ".w.sir",
      ]
    )
    check(
      "store fixture compiles to wasm",
      r.returncode == 0 and os.path.exists(p2[:-4] + ".w.wat"),
      r.stderr[:160],
    )
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc or not os.path.exists(p2[:-4] + ".c"):
      return
    exe = os.path.join(d, "p2.bin")
    rc = run([cc, "-O1", "-o", exe, p2[:-4] + ".c", "-lm"])
    check("store fixture p2.c compiles", rc.returncode == 0, rc.stderr[:200])
    if rc.returncode == 0:
      rr = run([exe])
      want = symiri_result(symiri, p1, "@stfix", ["9"])[1]
      # @main returns the result; the process exit code carries its low
      # 8 bits.
      ok = want is not None and rr.returncode == int(want) % 256
      check(
        "store fixture binary matches symiri",
        ok,
        f"rc={rr.returncode} want={want}",
      )


def test_ptr_state_twin_generated(rytwin, symiri):
  """[Stage 4b] Pointer-bearing state no longer disables twin generation:
  the store fixture's twin is a computed body (with its final ptr
  reconstruction), not a constant replication."""
  with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "stfix.sir")
    open(p1, "w").write(STORE_FIXTURE)
    p2 = os.path.join(d, "p2.sir")
    r = run([rytwin, p1, "--p-twin", "1.0", "--seed", "3", "--validate", "-o", p2])
    check("gen store fixture twinned + validated", r.returncode == 0, r.stderr[:200])
    if r.returncode != 0:
      return
    bodies = twin_block_bodies(open(p2).read())
    check(
      "store-fixture twin is a generated computation",
      any(has_computed_rhs(b) for b in bodies),
      "\n---\n".join(bodies)[:300],
    )


def test_rylink_generated_twins(rytwin, symiri):
  """[Stage 4b] The real rylink program gets generated (computed) twin
  bodies, not 100% constant replication."""
  fixture = os.path.join(os.path.dirname(__file__), "fixtures", "rylink_twin_p1.sir")
  with tempfile.TemporaryDirectory() as d:
    p2 = os.path.join(d, "p2.sir")
    r = run([rytwin, fixture, "--p-twin", "1.0", "--seed", "3", "--validate", "-o", p2])
    check("rylink fixture twinned + validated (gen)", r.returncode == 0, r.stderr[:200])
    if r.returncode != 0:
      return
    bodies = twin_block_bodies(open(p2).read())
    computed = sum(1 for b in bodies if has_computed_rhs(b))
    check(
      "some rylink twins are generated computations",
      computed > 0,
      f"{computed}/{len(bodies)} computed",
    )
    r1 = symiri_result(symiri, fixture, "@main", [])
    r2 = symiri_result(symiri, p2, "@main", [])
    check("rylink generated twins stay equivalent", r1[1:] == r2[1:], f"{r1} vs {r2}")


def test_ptr_pool_generated(rytwin, rysmith):
  """[Stage 4b] Across a pointered pool, generation produces computed twin
  bodies and every one is fully concretized (no %? survives)."""
  with tempfile.TemporaryDirectory() as d:
    sirs = gen_ptr_pool(rysmith, d)
    if not sirs:
      check("ptr-pool gen setup", False, "generation failed")
      return
    computed = total = 0
    leaked = False
    for p1 in sirs:
      p2 = p1[:-4] + ".p2.sir"
      r = run([rytwin, p1, "--p-twin", "1.0", "--seed", "3", "-o", p2])
      if r.returncode != 0:
        continue
      src = open(p2).read()
      if "%?" in src:
        leaked = True
      for b in twin_block_bodies(src):
        total += 1
        if has_computed_rhs(b):
          computed += 1
    check(
      "pointered pool has generated twins",
      computed > 0,
      f"{computed}/{total} computed",
    )
    check("no %? leaks into pointered p2s", not leaked, "")


WHOLE_PROG_FIXTURE = """fun @leaf(%pa0: i32) : i32 {
  let mut %a: i32 = 3;
^entry:
  br ^work;
^work:
  %a = 2 * %a + %pa0;
  br ^exit;
^exit:
  ret %a;
}

fun @main() : i32 {
  let mut %r: i32 = undef;
^entry:
  %r = call @leaf(41);
  ret %r;
}
"""


def dump_trace(symiri, path, entry="@main", args=None):
  r = run([symiri, "--dump-trace", "--main", entry, path, "--"] + (args or []))
  return r.stdout + r.stderr


def test_whole_program_twin_fires(rytwin, symiri):
  """[whole-program] The program's entry is @main; the twin lives in a
  callee. The guard must be keyed on the state the callee actually sees at
  runtime (args from @main's call site), so the twin executes."""
  with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "prog.sir")
    open(p1, "w").write(WHOLE_PROG_FIXTURE)
    p2 = os.path.join(d, "p2.sir")
    r = run([rytwin, p1, "--p-twin", "1.0", "--seed", "3", "-o", p2])
    check("whole-program p1 twinned", r.returncode == 0, r.stderr[:200])
    if r.returncode != 0:
      return
    check(
      "twin block executes on the real input",
      "__twin" in dump_trace(symiri, p2),
      "",
    )


def test_validate_asserts_twin_fires(rytwin, symiri):
  """[whole-program] --validate interprets p2 on the profiled input and
  confirms at least one twin actually executed — a dead twin is a failure,
  not a silent success."""
  with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "prog.sir")
    open(p1, "w").write(WHOLE_PROG_FIXTURE)
    p2 = os.path.join(d, "p2.sir")
    r = run([rytwin, p1, "--p-twin", "1.0", "--seed", "3", "--validate", "-o", p2])
    check("--validate exits 0 on whole program", r.returncode == 0, r.stderr[:200])
    check(
      "--validate reports twin execution",
      re.search(r"twin exec", r.stdout) is not None,
      r.stdout[:200],
    )


LABEL_COLLIDE_FIXTURE = """fun @leafa(%pa0: i32) : i32 {
  let mut %a: i32 = 3;
^entry:
  br ^work;
^work:
  %a = 2 * %a + %pa0;
  br ^exit;
^exit:
  ret %a;
}

fun @leafb(%pa0: i32) : i32 {
  let mut %b: i32 = 5;
^entry:
  br ^work;
^work:
  %b = 3 * %b + %pa0;
  br ^exit;
^exit:
  ret %b;
}

fun @main() : i32 {
  let mut %r: i32 = undef;
  let mut %s: i32 = undef;
^entry:
  %r = call @leafa(41);
  %s = call @leafb(17);
  %r = %r + %s;
  ret %r;
}
"""


def test_label_collision_across_functions(rytwin, symiri):
  """[whole-program] Two callees share the label ^work. Each must get its
  own guard keyed on its own frame state, and both twins must execute."""
  with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "prog.sir")
    open(p1, "w").write(LABEL_COLLIDE_FIXTURE)
    p2 = os.path.join(d, "p2.sir")
    r = run([rytwin, p1, "--p-twin", "1.0", "--seed", "3", "-o", p2])
    check("collision p1 twinned", r.returncode == 0, r.stderr[:200])
    if r.returncode != 0:
      return
    src = open(p2).read()
    names = {nm for nm, _ in GUARD_FN_RE.findall(src)}
    check(
      "one guard per function, frame-scoped names",
      len([n for n in names if n.startswith("@__twg_leafa_")]) == 1
      and len([n for n in names if n.startswith("@__twg_leafb_")]) == 1,
      str(names),
    )
    check(
      "both twins execute",
      dump_trace(symiri, p2).count("__twin") >= 2,
      "",
    )
    r1 = symiri_result(symiri, p1, "@main", [])
    r2 = symiri_result(symiri, p2, "@main", [])
    check("collision program equivalent", r1[1:] == r2[1:], f"{r1} vs {r2}")


TWICE_CALLED_FIXTURE = """fun @leaf(%pa0: i32) : i32 {
  let mut %a: i32 = 3;
^entry:
  br ^work;
^work:
  %a = 2 * %a + %pa0;
  br ^exit;
^exit:
  ret %a;
}

fun @main() : i32 {
  let mut %r: i32 = undef;
  let mut %s: i32 = undef;
^entry:
  %r = call @leaf(41);
  %s = call @leaf(7);
  %r = %r + %s;
  ret %r;
}
"""


def test_widened_guard_serves_both_activations(rytwin, symiri):
  """[whole-program] A callee invoked twice with different args. The twin is
  planned from the first activation, but the box proves the region behaves the
  same across a range, so the guard admits the second call too — one twin
  serving both is the point of widening."""
  with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, "prog.sir")
    open(p1, "w").write(TWICE_CALLED_FIXTURE)
    p2 = os.path.join(d, "p2.sir")
    r = run([rytwin, p1, "--p-twin", "1.0", "--seed", "3", "-o", p2])
    check("twice-called p1 twinned", r.returncode == 0, r.stderr[:200])
    if r.returncode != 0:
      return
    trace = dump_trace(symiri, p2)
    check(
      "the twin serves both activations",
      trace.count("__twin:") == 2,
      f"count={trace.count('__twin:')}",
    )
    r1 = symiri_result(symiri, p1, "@main", [])
    r2 = symiri_result(symiri, p2, "@main", [])
    check("twice-called program equivalent", r1[1:] == r2[1:], f"{r1} vs {r2}")


def test_real_rylink_regression(rytwin, symiri):
  """[whole-program] The checked-in rylink program (structs, vectors, odd
  widths, 8 callees + @main) twins and the twin executes at runtime."""
  fixture = os.path.join(os.path.dirname(__file__), "fixtures", "rylink_twin_p1.sir")
  with tempfile.TemporaryDirectory() as d:
    p2 = os.path.join(d, "p2.sir")
    r = run([rytwin, fixture, "--p-twin", "1.0", "--seed", "3", "-o", p2])
    check("rylink fixture twinned", r.returncode == 0, r.stderr[:200])
    if r.returncode != 0:
      return
    check("rylink twin executes", "__twin" in dump_trace(symiri, p2), "")
    r1 = symiri_result(symiri, fixture, "@main", [])
    r2 = symiri_result(symiri, p2, "@main", [])
    check("rylink program equivalent", r1[1:] == r2[1:], f"{r1} vs {r2}")


def test_solved_header_fallback(rytwin, rysmith, symiri):
  """[Stage 1] With no descriptor AND no sidecar, rytwin recovers the
  profiled input from p1's `// SOLVED:` header (a wrong input would trap on
  the interest requires during in-process profiling)."""
  with tempfile.TemporaryDirectory() as d:
    sirs = gen_pool(rysmith, d, emit_state=False)
    if not sirs:
      check("solved-header setup (rysmith gen)", False, "generation failed")
      return
    for f in os.listdir(d):
      if f.endswith(".json"):
        os.remove(os.path.join(d, f))
    got = first_twinned(rytwin, sirs, extra=["--validate"])
    check("rytwin twins from the SOLVED header alone", got is not None, "")
    if got:
      check("validated: OK without descriptor", "validated: OK" in got[2], got[2][:160])


def test_validate_without_sidecar(rytwin, rysmith):
  """[Stage 1] --validate works on a sidecar-free p1 (profiled input comes
  from the descriptor realization)."""
  with tempfile.TemporaryDirectory() as d:
    sirs = gen_pool(rysmith, d, emit_state=False)
    if not sirs:
      check("validate-no-sidecar setup (rysmith gen)", False, "generation failed")
      return
    got = first_twinned(rytwin, sirs, extra=["--validate"])
    check("--validate succeeds without a sidecar", got is not None, "")
    if got:
      check("validated: OK without sidecar", "validated: OK" in got[2], got[2][:160])


def test_bad_guard_rejected(rytwin, rysmith):
  """[Stage 2] The --guard flag is gone (the full-state guard function is
  the only guard); passing it is an unknown-option error."""
  with tempfile.TemporaryDirectory() as d:
    g = gen_p1(rysmith, d)
    if not g:
      check("rytwin bad-guard setup", False, "generation failed")
      return
    p1, desc, _, _ = g
    p2 = os.path.join(d, "p2.sir")
    r = run([rytwin, p1, "--guard", "bogus", "-o", p2])
    check("removed --guard rejected (rc != 0)", r.returncode != 0, f"rc={r.returncode}")


def test_missing_args_usage(rytwin):
  """No input / no output prints usage with a non-zero exit."""
  r = run([rytwin])
  check("missing input/output → non-zero exit", r.returncode != 0, f"rc={r.returncode}")


def parse_entry(src):
  fm = re.search(r"fun\s+(@\w+)\s*\(([^)]*)\)", src)
  params = [p.split(":") for p in fm.group(2).split(",") if p.strip()]
  pnames = [p[0].strip() for p in params]
  ptypes = [p[1].strip() for p in params]
  hdr = re.search(r"//\s*SOLVED:\s*(.*)", src)
  kv = {}
  if hdr:
    for part in hdr.group(1).split(","):
      if "=" in part:
        k, v = part.strip().split("=", 1)
        kv[k.strip()] = v.strip()
  return fm.group(1), pnames, ptypes, [kv.get(p, "0") for p in pnames]


def equivalence_over_pool(rytwin, symiri, d, sirs, tag, extra=None):
  """Twin every program in the pool with --p-twin 1.0 and assert p1 === p2 on
  the profiled input AND on other inputs (where the guard does not fire)."""
  import random

  rng = random.Random(1)
  twinned = prof_bad = other_bad = other_checks = 0
  for p1 in sirs:
    stem = os.path.basename(p1)[:-4]
    desc = os.path.join(d, re.sub(r"[a-z]$", "", stem) + ".json")
    if not os.path.exists(desc):
      continue
    fn, pnames, ptypes, iargs = parse_entry(open(p1).read())
    p2 = os.path.join(d, stem + ".p2.sir")
    rr = run(
      [rytwin, p1, "--p-twin", "1.0", "--seed", "3"] + (extra or []) + ["-o", p2]
    )
    if rr.returncode != 0:
      # A block-free-of-eligible-scalars program now exits non-zero with a
      # "no twin" message — expected, skip it. Anything else is a failure.
      if "no twin" in rr.stderr or "nothing written" in rr.stderr:
        continue
      check(f"rytwin ran on {stem} [{tag}]", False, rr.stderr[:160])
      return
    twinned += 1
    if (
      symiri_result(symiri, p1, fn, iargs)[1:]
      != symiri_result(symiri, p2, fn, iargs)[1:]
    ):
      prof_bad += 1
    # Other inputs (only when all params are integer) — the exact guard
    # must keep the equivalence even where the guard does not fire.
    if all("f" not in t for t in ptypes):
      for _ in range(4):
        a = [str(rng.randint(-1_000_000, 1_000_000)) for _ in pnames]
        other_checks += 1
        if symiri_result(symiri, p1, fn, a)[1:] != symiri_result(symiri, p2, fn, a)[1:]:
          other_bad += 1
  check(
    f"TwinTransform grafted at least one twin [{tag}]",
    twinned > 0,
    f"twinned={twinned}",
  )
  check(
    f"every twin preserves the profiled result [{tag}]",
    prof_bad == 0,
    f"{prof_bad} mismatch(es)",
  )
  check(
    f"every twin preserves results on other inputs [{tag}]",
    other_bad == 0,
    f"{other_bad}/{other_checks} mismatch(es)",
  )


def test_twinpass_grafts_and_preserves_equivalence(rytwin, rysmith, symiri):
  """On pointer-free programs (scalars, structs, arrays, vectors and pure
  intrinsic calls — everything eligibility now covers), TwinTransform grafts twins
  whose exact guard keeps p1 === p2: identical results on the profiled input
  AND on other inputs."""
  with tempfile.TemporaryDirectory() as d:
    # Pointers/memory aren't twin candidates yet, so disable them; keep
    # aggregates, vectors and intrinsic calls to exercise the lifted
    # eligibility.
    sirs = gen_pool(rysmith, d, emit_state=True)
    if sirs is None:
      check("twinpass setup (scalar-only gen)", False, "generation failed")
      return
    equivalence_over_pool(rytwin, symiri, d, sirs, "sidecar")


def test_equivalence_without_sidecar(rytwin, rysmith, symiri):
  """[Stage 1] The full equivalence suite holds when p1 is generated without
  --emit-state — the profile rytwin keys its guards on is computed
  in-process."""
  with tempfile.TemporaryDirectory() as d:
    sirs = gen_pool(rysmith, d, emit_state=False)
    if sirs is None:
      check("no-sidecar equivalence setup", False, "generation failed")
      return
    equivalence_over_pool(rytwin, symiri, d, sirs, "no-sidecar")


def test_validate_and_target(rytwin, rysmith, symiri):
  """--validate asserts p1 === p2 in-process, and --target c compiles p2."""
  with tempfile.TemporaryDirectory() as d:
    r = run(
      [
        rysmith,
        "--emit-state",
        "pbb",
        "--emit-desc",
        "--emit-main",
        "--n-funcs",
        "12",
        "--seed",
        "100",
        "--n-params",
        "2",
        "--n-stmts",
        "4",
        "--max-ptr-depth",
        "0",
        "--no-vec",
        "--no-agg-ptr",
        "--no-intrinsics",
        "-o",
        d,
      ]
    )
    if r.returncode != 0:
      check("validate/target setup", False, r.stderr[:200])
      return
    # Find a program that actually grafts a twin.
    picked = None
    for s in sorted(f for f in os.listdir(d) if f.endswith(".sir") and "_sym" not in f):
      p1 = os.path.join(d, s)
      p2 = os.path.join(d, s[:-4] + ".p2.sir")
      rr = run([rytwin, p1, "--p-twin", "1.0", "--seed", "3", "-o", p2])
      m = re.search(r"\((\d+) twin", rr.stdout)
      if rr.returncode == 0 and m and int(m.group(1)) > 0:
        picked = p1
        break
    check("found a twinnable program", picked is not None, "")
    if not picked:
      return
    p2 = os.path.join(d, "pv.sir")
    r = run(
      [
        rytwin,
        picked,
        "--p-twin",
        "1.0",
        "--seed",
        "3",
        "--validate",
        "--target",
        "c",
        "--emit-main",
        "-o",
        p2,
      ]
    )
    check("rytwin --validate --target c exits 0", r.returncode == 0, r.stderr[:200])
    check("rytwin reports validated: OK", "validated: OK" in r.stdout, r.stdout[:200])
    check("rytwin emitted p2.c", os.path.exists(os.path.join(d, "pv.c")), "")


def main():
  if len(sys.argv) not in (4, 5):
    print(
      "Usage: python3 -m test.unit.run_rytwin_tests <rytwin> <rysmith> <symiri>"
      " [test-name-substring]"
    )
    sys.exit(2)
  rytwin, rysmith, symiri = sys.argv[1:4]
  only = sys.argv[4] if len(sys.argv) == 5 else ""

  tests = [
    (
      "rytwin: no twin grafted -> error, no output",
      lambda: test_no_twins_errors(rytwin, rysmith),
    ),
    (
      "rytwin: no sidecar needed (in-process profiling)",
      lambda: test_no_sidecar_needed(rytwin, rysmith),
    ),
    (
      "rytwin: corrupt sidecar falls back to in-process profiling",
      lambda: test_corrupt_sidecar_falls_back(rytwin, rysmith),
    ),
    (
      "rytwin: valid sidecar preferred over interpreting",
      lambda: test_sidecar_preferred(rytwin, rysmith),
    ),
    (
      "rytwin: SOLVED-header fallback",
      lambda: test_solved_header_fallback(rytwin, rysmith, symiri),
    ),
    (
      "rytwin: --validate without sidecar",
      lambda: test_validate_without_sidecar(rytwin, rysmith),
    ),
    (
      "rytwin: invalid --guard rejected",
      lambda: test_bad_guard_rejected(rytwin, rysmith),
    ),
    (
      "TwinTransform: guard is a function",
      lambda: test_guard_is_function(rytwin, rysmith),
    ),
    (
      "TwinTransform: guard covers the full state",
      lambda: test_guard_omits_state_the_twin_ignores(rytwin),
    ),
    (
      "TwinTransform: guard function names unique per site",
      lambda: test_guard_unique_names(rytwin, rysmith),
    ),
    (
      "TwinTransform: aggregates by pointer, vectors per-lane",
      lambda: test_guard_aggregates_and_vectors(rytwin, rysmith, symiri),
    ),
    (
      "TwinTransform: guard functions compile (C binary + wasm)",
      lambda: test_guard_compiles_c_and_wasm(rytwin, rysmith),
    ),
    (
      "twin body: flags of the removed solver generator are rejected",
      lambda: test_removed_flags_rejected(rytwin),
    ),
    (
      "twin body: rytwin links no SMT backend",
      lambda: test_no_solver_linked(rytwin),
    ),
    (
      "state-set: the profiled state is provable",
      lambda: test_state_set_pass_proves_scalar_traces(rytwin),
    ),
    (
      "state-set: values are followed through memory",
      lambda: test_state_set_pass_follows_memory(rytwin, symiri),
    ),
    (
      "box: a loop pins its shape and opens its data",
      lambda: test_box_pins_control_and_opens_data(rytwin),
    ),
    (
      "box: a leaf the trace ignores is freed",
      lambda: test_box_frees_a_leaf_no_sampling_could(rytwin),
    ),
    (
      "guard: ranges are stated and free leaves dropped",
      lambda: test_guard_states_ranges_and_drops_free_leaves(rytwin),
    ),
    (
      "guard: free vector lanes leave the signature",
      lambda: test_guard_drops_free_vector_lanes(rytwin, symiri),
    ),
    (
      "guard: a widened guard fires on an unprofiled input",
      lambda: test_widened_guard_fires_on_another_input(rytwin, symiri),
    ),
    (
      "validate: states inside the box are spot-checked",
      lambda: test_validate_spot_checks_the_box(rytwin),
    ),
    (
      "box: ceilings bound the search before it starts",
      lambda: test_box_computes_ceilings_before_searching(rytwin),
    ),
    (
      "rewrite: a disguised body is no longer a copy",
      lambda: test_disguised_body_is_not_a_copy(rytwin, symiri),
    ),
    (
      "rewrite: what cannot be re-proved is rolled back",
      lambda: test_disguise_rolls_back_what_it_cannot_prove(rytwin),
    ),
    (
      "rewrite: every attempt lands on a live candidate",
      lambda: test_every_attempt_lands_on_a_live_candidate(rytwin),
    ),
    (
      "rewrite: bodies vary with the seed",
      lambda: test_disguise_varies_with_the_seed(rytwin),
    ),
    (
      "rewrite: every rule's identity holds, per the interpreter",
      lambda: test_rewrite_rules_selftest(rytwin),
    ),
    (
      "rewrite: an inserted mask stays inside its own type",
      lambda: test_mask_stays_inside_its_own_type(rytwin),
    ),
    (
      "rewrite: a literal subtraction becomes a negative addition",
      lambda: test_negation_is_unfolded(rytwin),
    ),
    (
      "rewrite: a multiplication becomes a shift where the sign is proven",
      lambda: test_multiplication_becomes_a_shift(rytwin),
    ),
    (
      "rewrite: a shift is refused where the sign is not proven",
      lambda: test_shift_refuses_what_the_box_does_not_prove(rytwin),
    ),
    (
      "rewrite: a complement becomes arithmetic",
      lambda: test_complement_becomes_arithmetic(rytwin),
    ),
    (
      "rewrite: a subtraction becomes an addition",
      lambda: test_subtraction_becomes_addition(rytwin),
    ),
    (
      "rewrite: a self-addition becomes a multiplication",
      lambda: test_doubling_becomes_multiplication(rytwin),
    ),
    (
      "rewrite: a comparison swaps its operands",
      lambda: test_comparison_operands_swap(rytwin),
    ),
    (
      "rewrite: a value round-trips through a wider type",
      lambda: test_values_round_trip_through_a_wider_type(rytwin),
    ),
    (
      "rewrite: a reused temp is recomputed instead",
      lambda: test_a_value_is_recomputed_instead_of_reused(rytwin),
    ),
    (
      "rewrite: dead statements read live locals",
      lambda: test_dead_statements_read_live_locals(rytwin),
    ),
    (
      "rewrite: a value is routed through memory",
      lambda: test_a_value_is_routed_through_memory(rytwin, symiri),
    ),
    (
      "rewrite: scratch declarations are shuffled",
      lambda: test_scratch_declarations_are_shuffled(rytwin),
    ),
    (
      "rewrite: a pinned value stands in for a literal",
      lambda: test_a_pinned_value_stands_in_for_a_literal(rytwin),
    ),
    (
      "rewrite: a bounded value licenses a mask",
      lambda: test_a_bounded_value_licenses_a_mask(rytwin),
    ),
    (
      "rewrite: a true condition hides an arm",
      lambda: test_a_true_condition_hides_an_arm(rytwin),
    ),
    (
      "rewrite: a free leaf is faked into the body",
      lambda: test_a_free_leaf_is_faked_into_the_body(rytwin),
    ),
    (
      "rewrite: a bound makes a constant out of a range",
      lambda: test_a_bound_makes_a_constant_out_of_a_range(rytwin),
    ),
    (
      "rewrite: a select becomes branchless",
      lambda: test_a_select_becomes_branchless(rytwin),
    ),
    (
      "rewrite: a comparison splits in two",
      lambda: test_a_comparison_splits_in_two(rytwin),
    ),
    (
      "rewrite: a comparison becomes a difference",
      lambda: test_a_comparison_becomes_a_difference(rytwin),
    ),
    (
      "rewrite: an intrinsic is written out",
      lambda: test_an_intrinsic_is_written_out(rytwin, symiri),
    ),
    (
      "state-set: a leaf the box freed stays unknown",
      lambda: test_a_freed_leaf_stays_unknown(rytwin),
    ),
    (
      "state-set: a shift by a known amount is bounded",
      lambda: test_state_set_pass_bounds_a_shift(rytwin),
    ),
    (
      "state-set: a logical shift is bounded across zero",
      lambda: test_state_set_pass_bounds_a_logical_shift(rytwin),
    ),
    (
      "state-set: a mask by a non-negative constant is bounded",
      lambda: test_state_set_pass_bounds_a_mask(rytwin),
    ),
    (
      "state-set: a signed mask claims nothing",
      lambda: test_state_set_pass_declines_a_signed_mask(rytwin),
    ),
    (
      "state-set: a widened shift guard still agrees",
      lambda: test_shifted_program_still_agrees(rytwin, symiri),
    ),
    (
      "rewrite: a conjunction crosses into arithmetic",
      lambda: test_and_crosses_into_arithmetic(rytwin),
    ),
    (
      "rewrite: a disjunction crosses into arithmetic",
      lambda: test_or_crosses_into_arithmetic(rytwin),
    ),
    (
      "rewrite: a subtraction crosses into the bitwise domain",
      lambda: test_subtraction_crosses_into_bitwise(rytwin),
    ),
    (
      "rewrite: an exclusive or stays bitwise, unlicensed",
      lambda: test_xor_stays_inside_the_bitwise_domain(rytwin),
    ),
    (
      "rewrite: De Morgan doubles the complements",
      lambda: test_de_morgan_doubles_the_complements(rytwin),
    ),
    (
      "rewrite: a non-negative value splits at a bit",
      lambda: test_value_splits_at_a_bit(rytwin),
    ),
    (
      "rewrite: an unproven body still gets trap-free rewrites",
      lambda: test_an_unproven_body_still_gets_trap_free_rewrites(rytwin),
    ),
    (
      "twin: a rejection says what it could not guard",
      lambda: test_a_rejection_says_what_it_could_not_guard(rytwin),
    ),
    (
      "rewrite: comparisons rewrite at any width",
      lambda: test_comparisons_rewrite_at_any_width(rytwin),
    ),
    (
      "state-set: vector lanes are followed",
      lambda: test_state_set_pass_follows_vector_lanes(rytwin),
    ),
    (
      "state-set: a written lane is followed",
      lambda: test_state_set_pass_follows_a_written_lane(rytwin),
    ),
    (
      "state-set: an unknown lane index claims nothing",
      lambda: test_an_unknown_lane_index_claims_nothing(rytwin),
    ),
    (
      "state-set: a vector twin agrees off the profile",
      lambda: test_vector_twin_agrees_off_the_profile(rytwin, symiri),
    ),
    (
      "state-set: a disjunction is bounded",
      lambda: test_state_set_pass_bounds_a_disjunction(rytwin),
    ),
    (
      "state-set: an exclusive or is bounded",
      lambda: test_state_set_pass_bounds_an_exclusive_or(rytwin),
    ),
    (
      "state-set: a quotient is bounded",
      lambda: test_state_set_pass_bounds_a_quotient(rytwin),
    ),
    (
      "state-set: a remainder is bounded",
      lambda: test_state_set_pass_bounds_a_remainder(rytwin),
    ),
    (
      "state-set: a signed disjunction claims nothing",
      lambda: test_a_signed_disjunction_claims_nothing(rytwin),
    ),
    (
      "state-set: a pinned intrinsic call is folded",
      lambda: test_a_pinned_intrinsic_is_folded(rytwin),
    ),
    (
      "state-set: an open intrinsic argument folds nothing",
      lambda: test_an_open_intrinsic_argument_folds_nothing(rytwin),
    ),
    (
      "state-set: a folded call agrees off the profile",
      lambda: test_a_folded_call_agrees_off_the_profile(rytwin, symiri),
    ),
    (
      "guard: a root initialized before the region is guardable",
      lambda: test_a_root_initialized_before_the_region_is_guardable(rytwin, symiri),
    ),
    (
      "guard: a root initialized on one path is not",
      lambda: test_a_root_initialized_on_one_path_is_not(rytwin),
    ),
    (
      "box: a leaf is not held back by another",
      lambda: test_a_leaf_is_not_held_back_by_another(rytwin, symiri),
    ),
    (
      "box: the search is seeded, not chancy",
      lambda: test_box_is_deterministic_for_a_seed(rytwin),
    ),
    (
      "state-set: float values are followed exactly",
      lambda: test_state_set_pass_follows_floats(rytwin, symiri),
    ),
    (
      "state-set: pointers set up before the region are known",
      lambda: test_state_set_pass_seeds_pointers_from_the_profile(rytwin),
    ),
    (
      "state-set: pointer arithmetic lands on a known cell",
      lambda: test_state_set_pass_follows_pointer_arithmetic(rytwin),
    ),
    (
      "state-set: an unprovable trace is reported, not hidden",
      lambda: test_state_set_pass_reports_what_it_cannot_prove(rytwin, symiri),
    ),
    (
      "twin body: the trace records the conditions it assumed",
      lambda: test_trace_records_path_conditions(rytwin),
    ),
    (
      "twin body: a branch inside the region is one condition",
      lambda: test_trace_records_diamond_condition(rytwin, symiri),
    ),
    (
      "twin body: a loop is unrolled into the trace",
      lambda: test_trace_body_unrolls_loop(rytwin),
    ),
    (
      "twin body: every block of the region is replayed",
      lambda: test_trace_body_replays_region_stmts(rytwin),
    ),
    (
      "region scope: collapses a block sequence to its exit",
      lambda: test_scope_region_sequence(rytwin, symiri),
    ),
    (
      "region scope: collapses a whole loop to its exit",
      lambda: test_scope_region_loop(rytwin, symiri),
    ),
    (
      "region scope: region is the default unit",
      lambda: test_default_scope_is_region(rytwin),
    ),
    (
      "region scope: equivalence sweep (profiled + other)",
      lambda: test_scope_region_equivalence(rytwin, rysmith, symiri),
    ),
    (
      "select: --twin-select option accepted/rejected",
      lambda: test_select_flag(rytwin),
    ),
    (
      "select: interesting wins the overlap (whole-function region)",
      lambda: test_select_interesting_wins_overlap(rytwin, symiri),
    ),
    (
      "select: interesting-selection equivalence sweep",
      lambda: test_select_interesting_equivalence(rytwin, rysmith, symiri),
    ),
    (
      "select: softmax per-region probability",
      lambda: test_select_softmax_probability(rytwin),
    ),
    (
      "twin body: twins compute from the live-in state",
      lambda: test_twin_is_generated(rytwin, rysmith),
    ),
    (
      "twin body: twins fully concrete",
      lambda: test_twin_fully_concrete(rytwin, rysmith),
    ),
    (
      "twin body: equivalence sweep",
      lambda: test_twin_body_equivalence(rytwin, rysmith, symiri),
    ),
    (
      "twin body: no equality requires in the graft",
      lambda: test_twin_requires_stripped(rytwin, rysmith),
    ),
    (
      "whole-program: twin fires via @main",
      lambda: test_whole_program_twin_fires(rytwin, symiri),
    ),
    (
      "whole-program: --validate asserts twin execution",
      lambda: test_validate_asserts_twin_fires(rytwin, symiri),
    ),
    (
      "whole-program: label collision across functions",
      lambda: test_label_collision_across_functions(rytwin, symiri),
    ),
    (
      "whole-program: first visit across activations",
      lambda: test_widened_guard_serves_both_activations(rytwin, symiri),
    ),
    (
      "whole-program: real rylink regression",
      lambda: test_real_rylink_regression(rytwin, symiri),
    ),
    (
      "pointers: sidecar records provenance",
      lambda: test_ptr_state_json_provenance(rysmith),
    ),
    (
      "pointers: store-bearing block twinnable",
      lambda: test_store_block_twinnable(rytwin, symiri),
    ),
    (
      "pointers: equivalence with memory ops",
      lambda: test_equivalence_with_pointers(rytwin, rysmith, symiri),
    ),
    (
      "pointers: guard covers ptr leaves",
      lambda: test_guard_covers_ptr_leaves(rytwin, rysmith),
    ),
    (
      "pointers: a one-past-the-end pointer leaf is twinnable",
      lambda: test_one_past_end_pointer_region(rytwin, symiri),
    ),
    (
      "pointers: backends on pointered twins",
      lambda: test_ptr_program_targets(rytwin, symiri),
    ),
    (
      "twin body+ptr: store fixture twin computes",
      lambda: test_ptr_state_twin_generated(rytwin, symiri),
    ),
    (
      "twin body+ptr: rylink program gets computing twins",
      lambda: test_rylink_generated_twins(rytwin, symiri),
    ),
    (
      "twin body+ptr: pointered pool twins are concrete",
      lambda: test_ptr_pool_generated(rytwin, rysmith),
    ),
    ("rytwin: missing args usage", lambda: test_missing_args_usage(rytwin)),
    (
      "TwinTransform: grafts twins, preserves equivalence (profiled + other inputs)",
      lambda: test_twinpass_grafts_and_preserves_equivalence(rytwin, rysmith, symiri),
    ),
    (
      "TwinTransform: equivalence without sidecar",
      lambda: test_equivalence_without_sidecar(rytwin, rysmith, symiri),
    ),
    (
      "rytwin: --validate and --target c",
      lambda: test_validate_and_target(rytwin, rysmith, symiri),
    ),
  ]
  for title, fn in tests:
    if only and only not in title and only not in fn.__code__.co_names[0]:
      continue
    print(f"=== {title} ===")
    fn()

  passed = sum(1 for _, ok, _ in results if ok)
  total = len(results)
  print(f"\nSummary (rytwin_tests): {passed}/{total} passed.\n")
  sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
  main()
