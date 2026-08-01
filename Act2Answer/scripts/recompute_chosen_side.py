"""Recompute chosen_side / is_answered from saved cube_fx/fy under a new margin,
without re-running any policy. Uses per-episode boardL_y/boardR_y (real tile
positions) from last_info, so it is robust to per-scene tile jitter.

hard/soft margins are passed in; compares OLD (0.08/0.16) vs NEW (0.01/0.03).
Prints, per output-prefix: answer-rate and L/R/neutral counts for each margin.
"""
import glob, os, sys, yaml
from collections import Counter, defaultdict

HALF = 0.0715          # tile half-size (Y), scale 1.3
CUBE_HALF = 0.017      # approx; only Y matters for side

def side(cy, bLy, bRy, m):
    onL = abs(cy - bLy) <= HALF + m
    onR = abs(cy - bRy) <= HALF + m
    if onL and onR:
        # closer tile wins (mirrors soft tie-break)
        return "L" if abs(cy - bLy) <= abs(cy - bRy) else "R"
    if onL: return "L"
    if onR: return "R"
    return "neutral"

def process(prefix_glob):
    files = sorted(glob.glob(prefix_glob))
    per_margin = {name: Counter() for name in ["OLD_hard_0.08", "NEW_hard_0.01", "NEW_soft_0.03"]}
    n_ep = 0
    for f in files:
        try:
            d = yaml.safe_load(open(f))
        except Exception:
            continue
        li = d.get("last_info", {})
        for i, e in li.items():
            if not isinstance(e, dict) or "cube_fy" not in e:
                continue
            cy = float(e["cube_fy"]); bLy = float(e.get("boardL_y", -0.155)); bRy = float(e.get("boardR_y", 0.155))
            # skip physically fallen cubes (|cy| huge)
            if abs(cy) > 0.5:
                for nm in per_margin: per_margin[nm]["fell"] += 1
                n_ep += 1
                continue
            n_ep += 1
            per_margin["OLD_hard_0.08"][side(cy, bLy, bRy, 0.08)] += 1
            per_margin["NEW_hard_0.01"][side(cy, bLy, bRy, 0.01)] += 1
            per_margin["NEW_soft_0.03"][side(cy, bLy, bRy, 0.03)] += 1
    return n_ep, per_margin

if __name__ == "__main__":
    OUT = "/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer/outputs"
    groups = {
        "magma":     f"{OUT}/confirm-mid-magma-*/glob/vis_0_test/stats.yaml",
        "internvla": f"{OUT}/confirm-internvla-*/glob/vis_0_test/stats.yaml",
        "rldx":      f"{OUT}/confirm-rldx-*/glob/vis_0_test/stats.yaml",
        "spatialvla":f"{OUT}/confirm*spatialvla*/glob/vis_0_test/stats.yaml",
    }
    for model, g in groups.items():
        n, pm = process(g)
        if n == 0:
            continue
        print(f"\n===== {model}  (n={n} эп) =====")
        for nm, c in pm.items():
            L, R, neu, fell = c["L"], c["R"], c["neutral"], c["fell"]
            ans = L + R
            tot = L + R + neu
            arate = 100 * ans / tot if tot else 0
            bal = f"L={L} R={R}" + (f"  L%={100*L/ans:.0f}" if ans else "")
            print(f"  {nm:16}: answer-rate={arate:5.1f}%  neutral={neu:4}  {bal}  (fell={fell})")
