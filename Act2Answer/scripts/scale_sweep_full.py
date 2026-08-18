#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ПОЛНАЯ сводка по масштабам плиток: ceiling + нейтральный кардсет в одной таблице.

Раньше цифры жили в двух местах и с дырами: ceiling (задача с ИЗВЕСТНЫМ ответом,
кардсет ceiling_color) разбирался только на краях сетки, нейтральный кардсет
(без семантики) — только на 1.3/1.5. Здесь оба источника считаются по всем
доступным масштабам и печатаются рядом.

Два замера отвечают на РАЗНЫЕ вопросы и потому расходятся:
  ceiling  — цель названа («положи на красную»); крен показывает, насколько
             семантика перебивает моторный дефолт;
  neutral  — цели нет («положи на плитку»); крен показывает чистую сумму
             моторной привычки и геометрического перекоса сцены.

Колонки:
  acc     — доля правильной плитки среди ответивших (только ceiling);
  AR      — answer-rate: доехал ли куб вообще;
  left%   — доля LEFT среди ответивших = позиционный крен (50 = нет);
  CI      — интервал Уилсона.

  python3 scale_sweep_full.py --out ../metrics/scale_sweep_full.md
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


def collect(pattern, regex):
    """-> {(model, scale): {'left','right','none','n','correct','answered'}}"""
    acc = defaultdict(lambda: defaultdict(int))
    for d in sorted(glob.glob(pattern)):
        m = regex.search(os.path.basename(d))
        if not m:
            continue
        model, scale = m.group("model"), float(m.group("scale"))
        f = os.path.join(d, "glob", "vis_0_test", "stats.yaml")
        if not os.path.exists(f):
            continue
        y = yaml.safe_load(open(f))
        for ep in y.get("last_info", {}).values():
            if not isinstance(ep, dict) or "chosen_side" not in ep:
                continue
            a = acc[(model, scale)]
            a["n"] += 1
            side = int(round(float(ep["chosen_side"])))
            if side == 1:
                a["left"] += 1
            elif side == 2:
                a["right"] += 1
            if side in (1, 2):
                a["answered"] += 1
                # в поэпизодных данных ключ называется success (не is_success)
                sk = "success" if "success" in ep else "is_success"
                if sk in ep:
                    a["correct"] += int(round(float(ep[sk])))
    return acc


def render(acc, title, with_acc, say):
    models = sorted({k[0] for k in acc})
    for model in models:
        keys = sorted([k for k in acc if k[0] == model], key=lambda z: z[1])
        if not keys:
            continue
        say(f"### {title} — {model}")
        say("")
        head = "| масштаб | эпизодов | AR | LEFT | RIGHT | left% | 95% CI | p vs 50% |"
        if with_acc:
            head = head[:-1] + " acc |"
        say(head)
        say("|---" * (9 if with_acc else 8) + "|")
        for k in keys:
            a = acc[k]
            ans = a["answered"]
            if ans == 0:
                say(f"| {k[1]} | {a['n']} | 0% | 0 | 0 | — | — | — |"
                    + (" — |" if with_acc else ""))
                continue
            lp = 100 * a["left"] / ans
            lo, hi = wilson(a["left"], ans)
            p = sps.binomtest(a["left"], ans, 0.5).pvalue if sps else float("nan")
            star = "**" if p < 0.05 else ""
            row = (f"| {k[1]} | {a['n']} | {100*ans/a['n']:.0f}% | {a['left']} | "
                   f"{a['right']} | {star}{lp:.1f}%{star} | [{lo:.1f}, {hi:.1f}] | {p:.2g} |")
            if with_acc:
                row += f" {a['correct']/ans:.2f} |"
            say(row)
        say("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lines = []
    say = lines.append
    say("# Масштаб плиток: полная сетка (ceiling + нейтральный кардсет)")
    say("")
    say("**ceiling** (`ceiling_color`) — цель НАЗВАНА, у задачи есть правильный ответ;")
    say("крен показывает, насколько семантика перебивает моторный дефолт.")
    say("**neutral** (`neutral_colors_big`) — цели НЕТ, правильного ответа нет;")
    say("крен = чистая моторная привычка + геометрический перекос сцены.")
    say("")

    ceil = collect("outputs/scalesweep-color-*-s*",
                   re.compile(r"scalesweep-color-(?P<model>[a-z0-9]+)-s(?P<scale>[\d.]+)-"))
    neut = collect("outputs/scalepar-neutral_colors_big-*-s*",
                   re.compile(r"-(?P<model>magma|internvla)-s(?P<scale>[\d.]+)-"))

    render(ceil, "CEILING (цель названа)", True, say)
    render(neut, "NEUTRAL (без семантики)", False, say)

    text = "\n".join(lines)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
