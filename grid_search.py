#!/usr/bin/env python3
# Grid search over --sep-mult x --pert-mult to maximise simplex quality
# (worst-case triangle altitude / longest edge over K).
# Usage: python3 grid_search.py [surface] [L]
import subprocess, sys, re

surface = sys.argv[1] if len(sys.argv) > 1 else "torus"
L       = sys.argv[2] if len(sys.argv) > 2 else "0.08"

SEP  = [2, 5, 8, 10, 12, 15, 20]
PERT = [25, 50, 75, 100, 150, 200]

REF_VERTS = None          # first valid run sets the reference vertex count
rx_q = re.compile(r"min altitude/longest-edge over K : ([0-9.eE+-]+) \(mean ([0-9.eE+-]+)\)")
rx_v = re.compile(r"K: (\d+) vertices")

def run(sep, pert):
    out = subprocess.run(
        ["./test1", "--surface", surface, "--L", L,
         "--sep-mult", str(sep), "--pert-mult", str(pert), "--out", "/tmp/gs.obj"],
        capture_output=True, text=True).stdout
    q = rx_q.search(out); v = rx_v.search(out)
    if not q or not v:
        return None
    return float(q.group(1)), float(q.group(2)), int(v.group(1))

results = []
for pert in PERT:
    for sep in SEP:
        r = run(sep, pert)
        if r is None:
            print(f"sep={sep:<3} pert={pert:<4} FAILED"); continue
        mn, mean, verts = r
        if REF_VERTS is None:
            REF_VERTS = verts
        valid = "ok" if verts == REF_VERTS else f"BROKEN({verts})"
        results.append((mn, mean, sep, pert, valid))
        print(f"sep={sep:<3} pert={pert:<4} min_alt={mn:.4e} mean={mean:.4e} {valid}")

print("\n=== top 5 by worst-case altitude (valid only) ===")
for mn, mean, sep, pert, valid in sorted(
        [r for r in results if r[4] == "ok"], reverse=True)[:5]:
    print(f"sep={sep:<3} pert={pert:<4} min_alt={mn:.4e} mean={mean:.4e}")
