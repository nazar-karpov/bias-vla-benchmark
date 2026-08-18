#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сетка масштабов на НЕЙТРАЛЬНОМ кардсете: чистый позиционный крен по моделям.

Кардсет neutral_colors_big: плитки одноцветные, вопрос нейтральный, правильного
ответа нет => любое отклонение доли LEFT от 50% есть ЧИСТЫЙ крен (моторика +
геометрия сцены), без вклада семантики картинки.

Считает по каждому масштабу и модели:
  AR      — answer-rate: доля эпизодов, где куб доехал до какой-либо плитки;
  left%   — доля LEFT среди ответивших = крен (50% = крена нет);
  95% CI  — интервал Уилсона для left%;
  p       — двусторонний биномиальный тест против 50%.

Данные берутся из episode_stats в stats.yaml: chosen_side (1=left, 2=right).
noswap и swap объединяются: на нейтральных плитках содержимое неразличимо,
поэтому раскладка не меняет смысла, а объём удваивается.

Запуск:
  python3 scale_sweep_neutral.py --glob 'outputs/scalepar-neutral_colors_big-*'
"""
import argparse
import glob
import math
import os
import re
from collections import defaultdict

import yaml

try:
    from scipy import stats as sps
except Exception:
    sps = None


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/scalepar-neutral_colors_big-*")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # scalepar-neutral_colors_big-<model>-s<scale>-<lay>-sh<N>
    pat = re.compile(r"-(magma|internvla)-s([\d.]+)-(noswap|swap)-sh(\d+)$")
    acc = defaultdict(lambda: {"left": 0, "right": 0, "none": 0, "n": 0})

    for d in sorted(glob.glob(args.glob)):
        m = pat.search(os.path.basename(d))
        if not m:
            continue
        model, scale = m.group(1), float(m.group(2))
        f = os.path.join(d, "glob", "vis_0_test", "stats.yaml")
        if not os.path.exists(f):
            continue
        y = yaml.safe_load(open(f))
        # поэпизодные значения лежат в last_info: {ep_idx: {...,'chosen_side':x}}
        li = y.get("last_info", {})
        vals = [ep["chosen_side"] for ep in li.values()
                if isinstance(ep, dict) and "chosen_side" in ep]
        if not vals:
            continue
        a = acc[(model, scale)]
        for v in vals:
            v = int(round(float(v)))
            a["n"] += 1
            if v == 1:
                a["left"] += 1
            elif v == 2:
                a["right"] += 1
            else:
                a["none"] += 1

    lines = []
    say = lines.append
    say("# Сетка масштабов на нейтральном кардсете: чистый позиционный крен")
    say("")
    say("neutral_colors_big: плитки без смысла, вопрос нейтральный, правильного")
    say("ответа нет => отклонение LEFT от 50% = чистый крен (моторика+геометрия).")
    say("noswap+swap объединены (плитки неразличимы, раскладка не меняет смысла).")
    say("")
    for model in ("magma", "internvla"):
        keys = sorted([k for k in acc if k[0] == model], key=lambda z: z[1])
        if not keys:
            continue
        say(f"## {model}")
        say("")
        say("| масштаб | эпизодов | ответили (AR) | LEFT | RIGHT | left% | 95% CI | p vs 50% |")
        say("|---|---|---|---|---|---|---|---|")
        for k in keys:
            a = acc[k]
            ans = a["left"] + a["right"]
            if ans == 0:
                say(f"| {k[1]} | {a['n']} | 0 (0%) | 0 | 0 | — | — | — |")
                continue
            lp = 100 * a["left"] / ans
            lo, hi = wilson(a["left"], ans)
            p = sps.binomtest(a["left"], ans, 0.5).pvalue if sps else float("nan")
            star = "**" if p < 0.05 else ""
            say(f"| {k[1]} | {a['n']} | {ans} ({100*ans/a['n']:.0f}%) | {a['left']} | "
                f"{a['right']} | {star}{lp:.1f}%{star} | [{lo:.1f}, {hi:.1f}] | {p:.2g} |")
        say("")

    text = "\n".join(lines)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
