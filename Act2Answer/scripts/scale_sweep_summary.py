#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сводка сетки масштабов: где размер плиток оптимален.

Критерий — НЕ величина эффекта, а отношение сигнал/шум на задаче с ИЗВЕСТНЫМ
ответом (ceiling): при каком масштабе модель чаще всего кладёт куб на нужную
плитку. Плюс позиционный крен на тех же данных и данные о подрезке
(metrics/tile_visibility.txt) — масштаб дисквалифицируется, если плитки
режутся краем несимметрично.

Колонки:
  acc        — доля эпизодов с правильной плиткой (ceiling: ответ известен);
  Wilson CI  — интервал для acc;
  AR         — answer-rate (доехал вообще);
  left%      — доля выбора ЛЕВОЙ среди ответивших = позиционный крен (50 = нет);
  vis/asym   — из tile_visibility.txt: подрезка и её асимметрия.

Запуск:
  python scale_sweep_summary.py --glob 'outputs/scalesweep-color-magma-s*'
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import yaml

R = "/workspace/moskalenko/bias-vla-benchmark-main"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def load_visibility(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        m = re.match(r"scale=([\d.]+):.*vis_L=([\d.]+) vis_R=([\d.]+) asym=([\d.]+)", line)
        if m:
            out[m.group(1)] = {"vis_L": float(m.group(2)), "vis_R": float(m.group(3)),
                               "asym": float(m.group(4))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="Act2Answer/outputs/scalesweep-color-magma-s*")
    ap.add_argument("--pairs", default="Act2Answer/ManiSkill/mani_skill/assets/"
                                       "carrot/ceiling_color/pairs.json")
    ap.add_argument("--vis", default="metrics/tile_visibility.txt")
    ap.add_argument("--out", default="metrics/scale_sweep.md")
    args = ap.parse_args()

    pairs = json.load(open(os.path.join(R, args.pairs)))
    ans = {p["index"]: p["answer"] for p in pairs}

    # scale -> счётчики
    agg = defaultdict(lambda: {"ok": 0, "ans": 0, "n": 0, "left": 0})
    for d in sorted(glob.glob(os.path.join(R, args.glob))):
        m = re.search(r"-s([\d.]+)-(noswap|swap)$", d)
        if not m:
            continue
        sc, layout = m.group(1), m.group(2)
        st = os.path.join(d, "glob", "vis_0_test", "stats.yaml")
        if not os.path.exists(st):
            continue
        li = (yaml.safe_load(open(st)) or {}).get("last_info") or {}
        for idx, info in li.items():
            ep = int(idx)
            side = int(info.get("chosen_side", 0) or 0)   # 1=left 2=right
            a = agg[sc]
            a["n"] += 1
            if side == 0:
                continue
            a["ans"] += 1
            if side == 1:
                a["left"] += 1
            # swap физически меняет плитки местами -> правильная сторона инвертируется
            correct = ans.get(ep, "Left")
            want = 1 if correct == "Left" else 2
            if layout == "swap":
                want = 3 - want
            if side == want:
                a["ok"] += 1

    vis = load_visibility(os.path.join(R, args.vis))

    lines = ["# Сетка масштабов плиток (ceiling_color, Magma)", "",
             "acc — доля правильной плитки среди ОТВЕТИВШИХ (ответ известен);",
             "left% — позиционный крен (50 = нет); asym — асимметрия подрезки.", "",
             "| масштаб | acc | 95% CI | AR | left% | vis_L | vis_R | asym |",
             "|---|---|---|---|---|---|---|---|"]
    best = None
    for sc in sorted(agg, key=float):
        a = agg[sc]
        if a["ans"] == 0:
            continue
        acc = a["ok"] / a["ans"]
        lo, hi = wilson(a["ok"], a["ans"])
        ar = a["ans"] / max(a["n"], 1)
        left = a["left"] / a["ans"] * 100
        v = vis.get(sc, {})
        lines.append(f"| {sc} | **{acc:.3f}** | [{lo:.3f}, {hi:.3f}] | {ar:.2f} | "
                     f"{left:.1f}% | {v.get('vis_L', float('nan')):.3f} | "
                     f"{v.get('vis_R', float('nan')):.3f} | "
                     f"{v.get('asym', float('nan')):.3f} |")
        # кандидат: максимум acc среди «чистых» по геометрии
        clean = v.get("asym", 1.0) <= 0.02 and v.get("vis_R", 0) >= 0.98
        if clean and (best is None or acc > best[1]):
            best = (sc, acc)
    lines.append("")
    if best:
        lines.append(f"**Оптимум по acc среди геометрически чистых (asym≤0.02, "
                     f"vis_R≥0.98): масштаб {best[0]}, acc={best[1]:.3f}**")
    else:
        lines.append("_Ни один масштаб не прошёл фильтр чистой геометрии._")

    # ВАЖНО: acc упирается в потолок (Magma ~1.0 везде) и потому НЕ различает
    # масштабы. Различают AR (доехал ли) и крен. Ищем компромисс:
    # максимум AR при минимальном |крен-50| среди чистых по геометрии.
    lines.append("")
    lines.append("## Что реально различает масштабы")
    lines.append("")
    lines.append("acc у Magma ~1.0 на ВСЕХ масштабах (семантику цвета читает всегда) —")
    lines.append("значит выбирать по acc нельзя. Различают **AR** (доезжает ли вообще)")
    lines.append("и **позиционный крен**:")
    lines.append("")
    lines.append("| масштаб | AR | |крен−50| | подрезка asym |")
    lines.append("|---|---|---|---|")
    for sc in sorted(agg, key=float):
        a = agg[sc]
        if a["ans"] == 0:
            continue
        ar = a["ans"] / max(a["n"], 1)
        skew = abs(a["left"] / a["ans"] * 100 - 50)
        asym = vis.get(sc, {}).get("asym", float("nan"))
        lines.append(f"| {sc} | {ar:.2f} | {skew:.1f} пп | {asym:.3f} |")

    txt = "\n".join(lines)
    print(txt)
    op = os.path.join(R, args.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    open(op, "w", encoding="utf-8").write(txt + "\n")
    print(f"\n-> {op}")


if __name__ == "__main__":
    main()
