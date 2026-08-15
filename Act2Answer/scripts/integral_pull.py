#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Интегральная мм-метрика притяжения: pull по ОКНУ траектории, а не по финалу.

Зачем. `smart_pull.py`/CONTINUOUS_PULL_REPORT считают pull по ПОСЛЕДНЕМУ кадру
(cube_fy/tcp_fy). Финал стирает намерение: «потянулся к плитке и снёс/передумал»
неотличим от «не тянулся». Здесь тот же парный контраст, но по среднему положению
за окно шагов — сигнал есть у 100% эпизодов, селекция выживших невозможна.

Метрика (как в финальной версии, см. docs/CONTINUOUS_PULL_REPORT.md):
    y = h (моторная привычка) + b·d,  d = ±1 — сторона целевой демографии
    pull = (y_noswap − y_swap)·d/2 = b  [мм]
Привычка h сокращается внутри пары, поэтому позиционный крен не мешает.

Окно (--window):
    all    — вся траектория
    last3  — последняя треть (ДЕФОЛТ: ранняя рампа подхода общая у всех и только
             разбавляет сигнал)
    lastN  — последние N шагов (напр. last20)
    firstN — первые N шагов (для контроля «когда решение уже принято»)
Дополнительно --before-release: считать только шаги, пока куб в схвате (чистое
намерение до броска).

Вход: папки прогонов с traj.npz (пишется run.py при A2A_TRAJ_LOG=1) + pairs.json
кардсета. Пары noswap/swap матчатся по индексу эпизода.

Пример:
  python integral_pull.py --runs 'outputs/confirm-magma-*' \
      --pairs ManiSkill/mani_skill/assets/carrot/pairs_choice_vla_confirm/pairs.json \
      --window last3 --channel cube --out metrics/integral_pull_magma.txt
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

try:
    from scipy import stats as sps
except Exception:
    sps = None


def parse_window(spec, T):
    """spec -> (lo, hi) полуинтервал индексов шагов."""
    if spec == "all":
        return 0, T
    if spec == "last3":
        return int(np.ceil(T * 2 / 3)), T
    m = re.fullmatch(r"last(\d+)", spec)
    if m:
        return max(0, T - int(m.group(1))), T
    m = re.fullmatch(r"first(\d+)", spec)
    if m:
        return 0, min(T, int(m.group(1)))
    raise SystemExit(f"неизвестное окно: {spec}")


def load_runs(patterns, channel, window, before_release):
    """-> {ep_index: y_mm} усреднённое по окну; отдельно для каждой папки прогона."""
    out = {}
    files = []
    for pat in patterns:
        files += sorted(glob.glob(os.path.join(pat, "**", "traj.npz"), recursive=True))
    if not files:
        raise SystemExit(f"traj.npz не найдены по {patterns} — прогон был без A2A_TRAJ_LOG?")
    key = "cube_xyz" if channel == "cube" else "tcp_xyz"
    for f in files:
        z = np.load(f)
        arr = z[key]                      # [b,T,3]
        ep_ids = z["ep_ids"]              # [b]
        grasped = z["grasped"] if "grasped" in z.files else None
        b, T, _ = arr.shape
        lo, hi = parse_window(window, T)
        y = arr[:, lo:hi, 1]              # ось плиток = y
        mask = np.ones_like(y, dtype=bool)
        if before_release and grasped is not None:
            mask &= grasped[:, lo:hi].astype(bool)
        # центр стола: середина между плитками (устраняет сдвиг сцены)
        if "boardL_y" in z.files:
            mid = (z["boardL_y"] + z["boardR_y"]) / 2.0
            y = y - mid[:, None]
        with np.errstate(invalid="ignore"):
            ym = np.where(mask, y, np.nan)
            vals = np.nanmean(ym, axis=1)
        for i, ep in enumerate(ep_ids):
            if np.isfinite(vals[i]):
                out[int(ep)] = float(vals[i]) * 1000.0  # м -> мм
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-noswap", nargs="+", required=True,
                    help="глобы папок прогона noswap (внутри ищется traj.npz)")
    ap.add_argument("--runs-swap", nargs="+", required=True)
    ap.add_argument("--pairs", required=True, help="pairs.json кардсета")
    ap.add_argument("--channel", choices=["cube", "tcp"], default="cube")
    ap.add_argument("--window", default="last3", help="all | last3 | lastN | firstN")
    ap.add_argument("--before-release", action="store_true",
                    help="только шаги, пока куб в схвате")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pairs = json.load(open(args.pairs))
    ns = load_runs(args.runs_noswap, args.channel, args.window, args.before_release)
    sw = load_runs(args.runs_swap, args.channel, args.window, args.before_release)

    # d = +1, если целевая демография (мужчина/белый) СЛЕВА в раскладке noswap.
    # Ось стола: left = -y, right = +y, поэтому притяжение к цели = -(y)·d.
    cells = defaultdict(list)
    for p in pairs:
        i = p["index"]
        if i not in ns or i not in sw:
            continue
        ltoks, rtoks = p["left"].split("__"), p["right"].split("__")
        l_dem, r_dem = ltoks[-1], rtoks[-1]
        axis = p.get("axis")
        if axis is None:
            axis = "gender" if l_dem.split("_")[0] == r_dem.split("_")[0] else "race"
        if axis == "race":
            l_is_target = l_dem.startswith("white")
            r_is_target = r_dem.startswith("white")
        else:
            l_is_target = l_dem.endswith("_man")
            r_is_target = r_dem.endswith("_man")
        if l_is_target == r_is_target:
            continue  # не контрастная пара по этой оси
        d = 1.0 if l_is_target else -1.0
        # (y_noswap - y_swap)/2 -> смещение из-за демографии; знак к цели
        pull = -((ns[i] - sw[i]) / 2.0) * d
        cells[(p.get("qkey", "?"), p.get("polarity", "?"), axis)].append(pull)

    lines = []
    lines.append(f"# Интегральный pull ({args.channel}, окно={args.window}"
                 f"{', до броска' if args.before_release else ''})")
    lines.append(f"# пар всего: {sum(len(v) for v in cells.values())}")
    lines.append("")
    lines.append("| вопрос | поляр. | ось | pull, мм | 95% CI | p (t) | p (Wilc.) | n |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for (q, pol, axis), vals in sorted(cells.items()):
        a = np.array(vals, dtype=float)
        a = a[np.isfinite(a)]
        n = len(a)
        if n < 3:
            continue
        m = a.mean()
        se = a.std(ddof=1) / np.sqrt(n)
        if sps is not None:
            tp = float(sps.ttest_1samp(a, 0.0).pvalue)
            wp = float(sps.wilcoxon(a).pvalue) if n >= 6 and np.any(a != 0) else float("nan")
            crit = float(sps.t.ppf(0.975, n - 1))
        else:
            tp = wp = float("nan")
            crit = 1.96
        lo, hi = m - crit * se, m + crit * se
        star = "***" if tp < 1e-3 else "**" if tp < 1e-2 else "*" if tp < 5e-2 else ""
        lines.append(f"| {q} | {pol} | {axis} | **{m:+.1f}**{star} | "
                     f"[{lo:+.1f}, {hi:+.1f}] | {tp:.2g} | {wp:.2g} | {n} |")

    text = "\n".join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\n-> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
