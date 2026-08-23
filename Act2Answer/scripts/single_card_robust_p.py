#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка надёжности p-value в single-card дизайне.

Наивный t-тест по 200 парным разностям предполагает их независимость. В нашем
кардсете это НЕ так: 50 сцен × 4 демо-варианта, и две гендерные разности из
одной сцены (white_m−white_w и black_m−black_w) делят фон, композицию и
профессию. Кластеризация занижает SE => завышает t => занижает p.

Считаем четыре версии p для каждой ячейки:
  1) naive     — обычный парный t-тест (то, что было в отчёте);
  2) cluster   — кластерные SE по СЦЕНЕ (агрегируем разности внутри сцены,
                 t-тест по 50 сценам): консервативная и правильная версия;
  3) perm      — пермутационный тест: случайно переворачиваем знак каждой
                 разности (H0: распределение симметрично относительно нуля),
                 20000 итераций; свободен от предположения нормальности;
  4) wilcoxon  — знаково-ранговый тест, непараметрика.

Плюс ICC (внутриклассовая корреляция по сценам) — она показывает, насколько
сильна кластеризация и во сколько раз раздут эффективный размер выборки
(design effect = 1 + (m−1)·ICC, m = наблюдений на кластер).

Запуск:
  python3 single_card_robust_p.py --runs 'outputs/single-top6-magma-*' \
      --pairs ManiSkill/.../pairs_single_top6/pairs.json --qkey athlete
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

try:
    from scipy import stats as sps
except Exception:
    sps = None

CARD_X = -0.25
CARD_Y = {"noswap": -0.155, "swap": +0.155}
# Центральный дизайн (одна фиксированная точка, папки помечены -center-):
# координата задаётся ключами --card-x/--card-y, слот один.
CARD_XY_CENTER = [-0.25, 0.05]
SLOTS = ["noswap", "swap"]


def load_assent(patterns, window="all", channel="cube"):
    key = "cube_xyz" if channel == "cube" else "tcp_xyz"
    files = []
    for pat in patterns:
        files += sorted(glob.glob(os.path.join(pat, "**", "traj.npz"), recursive=True))
    out = {}
    for f in files:
        low = f.lower()
        if "-center-" in low:
            slot, card = "center", np.array(CARD_XY_CENTER, dtype=float)
        elif "-noswap-" in low or "-swap-" in low:
            slot = "noswap" if "-noswap-" in low else "swap"
            card = np.array([CARD_X, CARD_Y[slot]])
        else:
            continue
        z = np.load(f)
        arr, ep_ids = z[key], z["ep_ids"]
        b, T, _ = arr.shape
        lo, hi = (0, T) if window == "all" else (int(np.ceil(T * 2 / 3)), T)
        d = np.linalg.norm(arr[:, :, :2] - card[None, None, :], axis=2)
        offtab = (np.abs(arr[:, :, 0]) > 0.5) | (np.abs(arr[:, :, 1]) > 0.5)
        dm = np.where(offtab, np.nan, d)
        d0 = dm[:, 0]
        with np.errstate(invalid="ignore"):
            dwin = np.nanmean(dm[:, lo:hi], axis=1)
        for i in range(b):
            if np.isfinite(d0[i]) and np.isfinite(dwin[i]):
                out[(int(ep_ids[i]), slot)] = float((d0[i] - dwin[i]) * 1000.0)
    return out


def diffs_with_scene(rows, data, var_key, la, lb, strat_keys):
    """-> (массив разностей, массив имён сцен-кластеров)"""
    groups = {}
    for r in rows:
        for slot in SLOTS:
            v = data.get((r["index"], slot))
            if v is None:
                continue
            strat = tuple(r[k] for k in strat_keys) + (slot,)
            groups.setdefault(strat, {})[r[var_key]] = (v, r["scene"])
    d, cl = [], []
    for g in groups.values():
        if la in g and lb in g:
            d.append(g[la][0] - g[lb][0])
            cl.append(g[la][1])
    return np.array(d), np.array(cl)


def t_test(x):
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), n
    se = x.std(ddof=1) / np.sqrt(n)
    t = x.mean() / se if se > 0 else np.nan
    p = 2 * sps.t.sf(abs(t), n - 1) if sps else np.nan
    return t, p, n


def icc_oneway(x, cl):
    """Внутриклассовая корреляция (однофакторная модель)."""
    ks = np.unique(cl)
    k = len(ks)
    if k < 2:
        return float("nan"), float("nan")
    ns = np.array([(cl == c).sum() for c in ks])
    grand = x.mean()
    msb = sum(n * (x[cl == c].mean() - grand) ** 2 for c, n in zip(ks, ns)) / (k - 1)
    within = sum(((x[cl == c] - x[cl == c].mean()) ** 2).sum() for c in ks)
    dfw = len(x) - k
    msw = within / dfw if dfw > 0 else np.nan
    m = ns.mean()
    icc = (msb - msw) / (msb + (m - 1) * msw) if (msb + (m - 1) * msw) > 0 else 0.0
    icc = max(0.0, min(1.0, icc))
    deff = 1 + (m - 1) * icc
    return icc, deff


def perm_p(x, B=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = abs(x.mean())
    signs = rng.choice([-1.0, 1.0], size=(B, len(x)))
    null = np.abs((signs * x).mean(axis=1))
    return (1 + (null >= obs).sum()) / (B + 1)


def analyse(name, d, cl):
    t, p, n = t_test(d)
    # кластерная версия: одно наблюдение на сцену
    ks = np.unique(cl)
    dc = np.array([d[cl == c].mean() for c in ks])
    tc, pc, nc = t_test(dc)
    icc, deff = icc_oneway(d, cl)
    pp = perm_p(d)
    pw = sps.wilcoxon(d).pvalue if sps and len(d) > 5 else float("nan")
    print(f"  {name}")
    print(f"    эффект      {d.mean():+7.1f} мм   (n пар = {n}, сцен = {nc})")
    print(f"    naive  t={t:6.2f}  p={p:.2g}")
    print(f"    cluster t={tc:6.2f}  p={pc:.2g}   <- по сценам, df={nc-1}")
    print(f"    perm            p={pp:.2g}")
    print(f"    wilcoxon        p={pw:.2g}")
    print(f"    ICC={icc:.3f}  design effect={deff:.2f}  "
          f"(эффективное n ≈ {n/deff:.0f} вместо {n})")
    return dict(effect=d.mean(), p_naive=p, p_cluster=pc, p_perm=pp,
                p_wilcox=pw, icc=icc, deff=deff, n=n, n_clusters=nc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--qkey", required=True)
    ap.add_argument("--window", default="all")
    ap.add_argument("--card-x", type=float, default=None,
                    help="центральный дизайн: x карточки (папки -center-)")
    ap.add_argument("--card-y", type=float, default=None,
                    help="центральный дизайн: y карточки")
    args = ap.parse_args()

    global SLOTS, CARD_XY_CENTER
    if args.card_x is not None or args.card_y is not None:
        CARD_XY_CENTER = [args.card_x if args.card_x is not None else -0.25,
                          args.card_y if args.card_y is not None else 0.0]
        SLOTS = ["center"]

    rows = [r for r in json.loads(open(args.pairs).read()) if r["qkey"] == args.qkey]
    data = load_assent(args.runs, args.window)

    print(f"### {args.qkey} (окно={args.window})")
    res = {}
    for pol in ("pos", "neg"):
        sub = [r for r in rows if r["polarity"] == pol]
        if not sub:
            continue
        print(f"\n{pol}: {sub[0]['question']}")
        d, cl = diffs_with_scene(sub, data, "gender", "man", "woman", ("scene", "race"))
        if len(d) > 5:
            res[(pol, "gender")] = analyse("ГЕНДЕР man-woman", d, cl)
        d, cl = diffs_with_scene(sub, data, "race", "white", "black", ("scene", "gender"))
        if len(d) > 5:
            res[(pol, "race")] = analyse("РАСА white-black", d, cl)
    return res


if __name__ == "__main__":
    main()
