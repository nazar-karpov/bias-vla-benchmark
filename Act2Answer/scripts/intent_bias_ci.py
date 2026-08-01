#!/usr/bin/env python3
"""Bias КАНАЛА НАМЕРЕНИЯ: сторона = first_touch_side (первое касание плитки
отпущенным кубом), тот же строгий пайплайн, что bias_stats_ci: канонические
прогоны, дедуп, swap-инверсия демографии, Wilson CI + binomtest vs 0.5.

Гипотеза: намерение несёт VLM-подобные стереотипы, которые исполнение фильтрует.
"""
import glob, json, re, sys, yaml
from collections import defaultdict, Counter
from scipy import stats

OUT = sys.argv[1] if len(sys.argv) > 1 else "outputs"
CARD = sys.argv[2]
CANON = json.loads(sys.argv[3])   # {"name": "prefix-glob"}
PAIRS = {e["index"]: e for e in json.load(open(CARD))}
QORDER = ["boss", "pilot", "wealthy", "skier"]

def demo(name, axis):
    who = name.split("__")[-1]
    if axis == "gender":
        return "female" if who.endswith("woman") else ("male" if who.endswith("_man") or who.endswith("man") and not who.endswith("woman") else None)
    for r in ["white", "black", "asian", "latino"]:
        if r in who: return r
    return None

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * ((p*(1-p)/n + z*z/(4*n*n)) ** 0.5) / d
    return (max(0, c-h), min(1, c+h))

for model, pat in CANON.items():
    seen = {}
    for f in sorted(glob.glob(f"{OUT}/{pat}/glob/vis_0_test/stats.yaml")):
        pol = "swap" if "-swap-" in f else "noswap"
        st = int(re.search(r"-s(\d+)", f).group(1))
        try: d = yaml.safe_load(open(f))
        except Exception: continue
        for li, e in d.get("last_info", {}).items():
            if isinstance(e, dict) and "first_touch_side" in e:
                k = (pol, st + int(li))
                if k not in seen: seen[k] = e
    print(f"\n{'='*72}\n  {model.upper()} (INTENT=first_touch)  уник={len(seen)}\n{'='*72}")
    for want in ("pos", "neg"):
        bq = defaultdict(Counter); ans = 0; tot = 0
        for (pol, gidx), e in seen.items():
            p = PAIRS.get(gidx)
            if p is None or p["polarity"] != want: continue
            tot += 1
            s = int(e.get("first_touch_side", 0) or 0)
            if s == 0: continue
            ans += 1
            left, right = (p["right"], p["left"]) if pol == "swap" else (p["left"], p["right"])
            t = demo(left if s == 1 else right, p["axis"])
            if t: bq[(p["axis"], p["qkey"])][t] += 1
        print(f"  --- polarity={want} (intent answer-rate {100*ans/max(tot,1):.0f}%) ---")
        print(f"  {'вопрос':8}| {'GENDER male% [CI] n sig':^32}| {'RACE white% [CI] n sig':^32}")
        for q in QORDER:
            cells = []
            for axis, pos_key in (("gender", "male"), ("race", "white")):
                c = bq.get((axis, q), Counter()); n = sum(c.values()); k = c.get(pos_key, 0)
                if n == 0: cells.append(f"{'-':^32}"); continue
                lo, hi = wilson(k, n)
                pv = stats.binomtest(k, n, 0.5).pvalue
                sig = "***" if pv < .001 else "**" if pv < .01 else "*" if pv < .05 else "(ns)"
                cells.append(f"{100*k/n:4.0f}% [{100*lo:3.0f}-{100*hi:3.0f}] n={n:<4d} {sig:<5}")
            print(f"  {q:8}| {cells[0]:^32}| {cells[1]:^32}")
