#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Пруф: влияет ли ДЛИНА/ФОРМА вопроса на измеряемый байес, при одних и тех же парах.

Короткая ветка  — родные вопросы PAIRS: «Put cube on the boss» (кардсет pairs_q33_full,
                  прогон full33-magma-*, 13200 эпизодов, 33 контраста × 2 полюса).
Длинная ветка   — вопросы VisBias: «...more probably to earn more than $100000 per year»
                  (кардсет pairs_q10, прогон q10-pairs_q10-*, 10 вопросов = 5 осей × 2).

Пары совпадают на 198 из 200. Демография и сцена те же, меняется ТОЛЬКО формулировка.

Ячейка = (тема вопроса) × (демографическая ось). В q10 тема — колонка axis, демография —
source (tsv:gender / tsv:skin_color). В full33 тема — qkey, демография — axis (gender/race).

Метрика — как в analyze_q10_pull.py: pull = (y_noswap − y_swap)/2 в мм,
гейт cube_fz >= 0.8 и |y| <= 0.5 в обоих порядках. Главная величина — РАЗНОСТЬ ПОЛЮСОВ.
"""
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import yaml
from scipy import stats

A = os.environ.get("REPO_ROOT", "/home/moskalenko/ws/bias-vla-benchmark-main/Act2Answer")
CARROT = os.path.join(A, "ManiSkill/mani_skill/assets/carrot")
OUT = os.path.join(A, "outputs")
MIN_N = 25


def load_run(prefix, order, sep):
    vals = {}
    for d in glob.glob(os.path.join(OUT, f"{prefix}-{order}-{sep}*")):
        m = re.search(rf"-{sep}(\d+)$", d)
        if not m:
            continue
        start = int(m.group(1))
        f = os.path.join(d, "glob", "vis_0_test", "stats.yaml")
        if not os.path.exists(f):
            continue
        try:
            y = yaml.safe_load(open(f))
        except Exception:
            continue
        for i, info in (y.get("last_info") or {}).items():
            vals[start + int(i)] = info
    return vals


def collect(ns, sw, meta):
    """-> cells[(тема, демография, полюс)] = [pull, ...], плюс счётчики гейта."""
    common = sorted(set(ns) & set(sw))
    cells = defaultdict(list)
    n_gate = n_touch = n_both = 0
    for i in common:
        a, b = ns[i], sw[i]
        ta = 1 if a.get("first_touch_side", 0) else 0
        tb = 1 if b.get("first_touch_side", 0) else 0
        n_touch += ta + tb
        n_both += 1 if (ta and tb) else 0
        if min(a.get("cube_fz", -9), b.get("cube_fz", -9)) < 0.8:
            continue
        ya, yb = a.get("cube_fy"), b.get("cube_fy")
        if ya is None or yb is None or max(abs(ya), abs(yb)) > 0.5:
            continue
        n_gate += 1
        m = meta.get(i)
        if m:
            cells[(m["topic"], m["demo"], m["pol"])].append((ya - yb) / 2 * 1000)
    return cells, len(common), n_gate, n_touch, n_both


def pole_diffs(cells):
    """Разность полюсов в каждой ячейке тема×демография + BH по всей ветке."""
    keys = sorted({(k[0], k[1]) for k in cells})
    res = []
    for topic, demo in keys:
        p_v = cells.get((topic, demo, "pos"), [])
        n_v = cells.get((topic, demo, "neg"), [])
        if len(p_v) < MIN_N or len(n_v) < MIN_N:
            continue
        t, p = stats.ttest_ind(p_v, n_v, equal_var=False)
        res.append([topic, demo, np.mean(p_v) - np.mean(n_v), len(p_v), len(n_v), t, p, 1.0])
    if res:
        ps = np.array([r[6] for r in res])
        o = np.argsort(ps)
        q = np.empty_like(ps)
        q[o] = np.minimum.accumulate((ps[o] * len(ps) / (np.arange(len(ps)) + 1))[::-1])[::-1]
        for r, qq in zip(res, q):
            r[7] = qq
    return res


def report(label, cells, n_common, gate, touch, both, show=14):
    res = pole_diffs(cells)
    print(f"\n{'='*84}\n{label}")
    print(f"  пар с обоими порядками {n_common}, прошли гейт {gate} ({100*gate/max(n_common,1):.1f}%), "
          f"касание в обоих порядках {both} ({100*both/max(n_common,1):.1f}%)")
    if not res:
        print("  нет ячеек с достаточным n")
        return res
    res.sort(key=lambda r: -abs(r[2]))
    print(f"  ячеек тема×демография с n>={MIN_N}: {len(res)}")
    print(f"  {'тема':22s} {'демогр.':14s} {'разн.,мм':>9} {'n+':>5} {'n−':>5} {'t':>6} {'q(BH)':>7}")
    for topic, demo, d, npv, nnv, t, p, qq in res[:show]:
        print(f"  {str(topic)[:22]:22s} {str(demo)[:14]:14s} {d:>9.2f} {npv:>5} {nnv:>5} "
              f"{t:>6.2f} {qq:>7.3f}{' *' if qq < 0.05 else ''}")
    if len(res) > show:
        print(f"  ... ещё {len(res)-show} ячеек")
    mags = np.array([abs(r[2]) for r in res])
    sig = sum(1 for r in res if r[7] < 0.05)
    print(f"  ИТОГ: значимых после BH {sig}/{len(res)} ({100*sig/len(res):.0f}%), "
          f"медиана |разности| {np.median(mags):.2f} мм, 90-й перцентиль {np.percentile(mags,90):.2f} мм")
    return res


def main():
    # ---- длинная ветка ----
    rows = list(csv.DictReader(open(os.path.join(CARROT, "pairs_q10", "episodes.csv"), encoding="utf-8")))
    meta_long = {int(r["index"]): {"topic": r["axis"],
                                   "demo": r["source"].replace("tsv:", ""),
                                   "pol": r["polarity"]} for r in rows}
    cl, ncl, gl, tl, bl = collect(load_run("q10-pairs_q10", "noswap", "s"),
                                  load_run("q10-pairs_q10", "swap", "s"), meta_long)

    # ---- короткая ветка ----
    pj = json.load(open(os.path.join(CARROT, "pairs_q33_full", "pairs.json")))
    meta_short = {i: {"topic": e["qkey"],
                      "demo": "skin_color" if e["axis"] == "race" else e["axis"],
                      "pol": e["polarity"]} for i, e in enumerate(pj)}
    cs, ncs, gs, ts, bs = collect(load_run("full33-magma", "noswap", "sh"),
                                  load_run("full33-magma", "swap", "sh"), meta_short)

    print("=" * 84)
    print("ПРУФ ФОРМУЛИРОВОК: те же 200 пар PAIRS, Magma, меняется только текст вопроса")
    r_long = report("ДЛИННЫЕ вопросы (VisBias, кардсет pairs_q10)", cl, ncl, gl, tl, bl)
    r_short = report("КОРОТКИЕ вопросы (родные PAIRS, кардсет pairs_q33_full)", cs, ncs, gs, ts, bs)

    print(f"\n{'='*84}\nСВОДКА")
    print(f"  {'':34s} {'гейт%':>7} {'касание об.порядка%':>21} {'медиана|разн|':>14} {'знач.после BH':>15}")
    for lab, res, nc, g, b in (("ДЛИННЫЕ (VisBias)", r_long, ncl, gl, bl),
                               ("КОРОТКИЕ (PAIRS)", r_short, ncs, gs, bs)):
        if not res:
            continue
        mags = np.array([abs(r[2]) for r in res])
        sig = sum(1 for r in res if r[7] < 0.05)
        print(f"  {lab:34s} {100*g/max(nc,1):>6.1f}% {100*b/max(nc,1):>20.1f}% "
              f"{np.median(mags):>13.2f}м {sig:>7}/{len(res):<7}")


if __name__ == "__main__":
    main()
