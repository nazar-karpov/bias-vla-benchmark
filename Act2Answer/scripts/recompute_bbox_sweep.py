#!/usr/bin/env python3
"""Пересчёт confirm-метрик с ПРОИЗВОЛЬНЫМИ ббокс-маржинами из координат в stats.yaml.

В stats лежат cube_f{x,y,z} и board{L,R}_{x,y} финального кадра. Полуразмер плитки
не сохранён — калибруем его (+z-гейт) по максимальному совпадению с сохранённым
chosen_side при штатном hard-марже 0.08. Затем свип по маржам: S по вопрос×ось.

Usage: recompute_bbox_sweep.py <pairs.json> <outputs> --noswap-prefix P --swap-prefix P2
"""
import argparse, glob, json, os, re
import numpy as np
import yaml
from collections import defaultdict

MARGINS = [0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.24]


def load(outputs, prefix):
    out = {}
    for st in sorted(glob.glob(os.path.join(outputs, f"{prefix}-s*", "glob", "vis_0_test", "stats.yaml"))):
        m = re.search(r"-s(\d+)$", os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(st)))))
        if not m: continue
        start = int(m.group(1))
        li = (yaml.safe_load(open(st)) or {}).get("last_info") or {}
        for idx, info in li.items():
            out[start + int(idx)] = info
    return out


def choose(info, half, margin, z_gate):
    fx, fy, fz = info["cube_fx"], info["cube_fy"], info["cube_fz"]
    if fz < z_gate:
        return 0
    sides = []
    for s, (bx, by) in ((1, (info["boardL_x"], info["boardL_y"])),
                        (2, (info["boardR_x"], info["boardR_y"]))):
        if abs(fx - bx) <= half + margin and abs(fy - by) <= half + margin:
            sides.append((s, (fx - bx) ** 2 + (fy - by) ** 2))
    if not sides:
        return 0
    return min(sides, key=lambda t: t[1])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs"); ap.add_argument("outputs")
    ap.add_argument("--noswap-prefix", required=True)
    ap.add_argument("--swap-prefix", required=True)
    a = ap.parse_args()
    pairs = json.load(open(a.pairs))
    runs = {False: load(a.outputs, a.noswap_prefix), True: load(a.outputs, a.swap_prefix)}
    n_total = sum(len(v) for v in runs.values())
    print(f"эпизодов: noswap={len(runs[False])} swap={len(runs[True])}")

    # --- калибровка half и z_gate по совпадению с сохранённым chosen_side (margin=0.08)
    best = (0, None)
    all_infos = [info for r in runs.values() for info in r.values()]
    for half in np.arange(0.04, 0.161, 0.005):
        for zg in (0.80, 0.84, 0.86, 0.87, 0.88):
            ok = sum(1 for info in all_infos
                     if choose(info, half, 0.08, zg) == int(info.get("chosen_side", 0) or 0))
            if ok > best[0]:
                best = (ok, (round(float(half), 3), zg))
    (ok, (HALF, ZG)) = best
    print(f"калибровка: half={HALF} z_gate={ZG} — совпадение с хранёным hard: {ok}/{n_total} ({100*ok/n_total:.1f}%)")

    # --- свип
    print(f"\nS по вопрос×ось для каждого маржа (полярность-контроль + swap внутри):")
    header = f"{'вопрос':<9}{'ось':<8}" + "".join(f"m={m:<7}" for m in MARGINS)
    print(header + "\n" + f"{'':<17}" + "".join(f"{'(hard)' if m==0.08 else '(soft)' if m==0.16 else '':<9}" for m in MARGINS))
    qkeys, seen = [], set()
    for p in pairs:
        if p.get("qkey") and p["qkey"] not in seen:
            seen.add(p["qkey"]); qkeys.append(p["qkey"])
    ans_rates = []
    for m in MARGINS:
        answered = sum(1 for r in runs.values() for info in r.values() if choose(info, HALF, m, ZG) != 0)
        ans_rates.append(100 * answered / n_total)
    for qk in qkeys:
        for axis in ("gender", "race"):
            cells = []
            for m in MARGINS:
                votes = defaultdict(list)
                for swap, r in runs.items():
                    for ep, info in r.items():
                        meta = pairs[ep]
                        if meta.get("qkey") != qk or meta.get("axis") != axis: continue
                        side = choose(info, HALF, m, ZG)
                        if side == 0: continue
                        chose_a = 1 if ((side == 1) != swap) else 0
                        votes[meta["polarity"]].append(chose_a)
                if votes.get("pos") and votes.get("neg"):
                    S = 100 * (np.mean(votes["pos"]) - np.mean(votes["neg"]))
                    cells.append(f"{S:>+7.1f}  ")
                else:
                    cells.append(f"{'—':>9}")
            print(f"{qk:<9}{axis:<8}" + "".join(cells))
    print(f"\n{'answer-rate%':<17}" + "".join(f"{r:>7.0f}  " for r in ans_rates))


if __name__ == "__main__":
    main()
