#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Согласие слотового и центрального single-card дизайна (те же 28 ячеек).

Зачем: в центре на каждую ячейку приходится n=100 пар вместо 200 (контрбаланс
слотов не нужен, но и второго замера той же карточки нет), поэтому t падает в
~√2 и после FDR по 28 тестам не выживает ничего. Вопрос «дизайн сломался или
просто потерял мощность» решается не по одной ячейке, а по согласию оценок:

1. знаковый тест по всем ячейкам (биномиальный, p=0.5);
2. то же по ячейкам, где слотовый дизайн видел эффект (|t|>2) — там знак вообще
   определён, а не шум вокруг нуля;
3. корреляция величин (Пирсон/Спирмен);
4. отношение величин center/slot — если центральный меряет то же, отношение
   должно держаться около 1 (обе оценки одной и той же тяги в мм).

Слотовые числа берутся из `single_card_fdr.py::TESTS` (эксп. 37+39, уже в гите),
центральные считаются здесь же из traj.npz теми же функциями, что и в анализе.

  python3 design_agreement_center.py \\
     --sets 'outputs/center-top6-magma-center-*::.../pairs_single_top6/pairs.json' \\
            'outputs/center-pilot-magma-center-*::.../pairs_single_pilot/pairs.json'
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import single_card_assent as sca          # noqa: E402
from single_card_fdr import TESTS as SLOT_TESTS   # noqa: E402

try:
    from scipy import stats as sps
except Exception:
    sps = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True,
                    help="'glob_прогонов::pairs.json' — по одному на кардсет "
                         "(индексы эпизодов у кардсетов пересекаются, мешать нельзя)")
    ap.add_argument("--card-x", type=float, default=-0.25)
    ap.add_argument("--card-y", type=float, default=0.05)
    ap.add_argument("--window", default="all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sca.CARD_XY_CENTER = [args.card_x, args.card_y]
    sca.SLOTS = ["center"]

    center = {}
    for spec in args.sets:
        runs_glob, pairs_path = spec.split("::")
        rows_all = json.loads(open(pairs_path).read())
        data = sca.load_assent(runs_glob.split(), "cube", args.window)
        for q in sorted({r["qkey"] for r in rows_all}):
            rows = [r for r in rows_all if r["qkey"] == q]
            for pol in ("pos", "neg"):
                sub = [r for r in rows if r["polarity"] == pol]
                m, n, t, p = sca.contrast(sub, data, ("scene", "race"),
                                          "gender", "man", "woman")
                center[(q, pol, "gender")] = (m, t, p, n)
                m, n, t, p = sca.contrast(sub, data, ("scene", "gender"),
                                          "race", "white", "black")
                center[(q, pol, "race")] = (m, t, p, n)

    lines = []
    say = lines.append
    say("# Согласие дизайнов: слот (эксп. 37+39, n=200) vs центр (n=100)")
    say(f"{'вопрос':<11}{'пол':<5}{'ось':<7}{'slot мм':>9}{'t':>7}"
        f"{'center мм':>11}{'t':>7}{'знак':>6}{'отнош.':>8}")
    pairs = []
    for q, pol, axis, mm_s, t_s, p_s in SLOT_TESTS:
        c = center.get((q, pol, axis))
        if c is None:
            continue
        mm_c, t_c, p_c, n_c = c
        same = np.sign(mm_s) == np.sign(mm_c)
        ratio = mm_c / mm_s if abs(mm_s) > 1e-9 else float("nan")
        pairs.append((q, pol, axis, mm_s, t_s, mm_c, t_c, same, ratio))
        say(f"{q:<11}{pol:<5}{axis:<7}{mm_s:>9.1f}{t_s:>7.2f}"
            f"{mm_c:>11.1f}{t_c:>7.2f}{'=' if same else 'X':>6}{ratio:>8.2f}")

    def sign_block(sel, title):
        k = sum(1 for x in sel if x[7])
        n = len(sel)
        p = sps.binomtest(k, n, 0.5).pvalue if sps and n else float("nan")
        say(f"\n{title}: знак совпал {k}/{n} (биномиальный p={p:.3g})")
        return k, n, p

    sign_block(pairs, "Все ячейки")
    strong = [x for x in pairs if abs(x[4]) > 2]
    sign_block(strong, "Только там, где слотовый дизайн видел эффект (|t|>2)")

    s = np.array([x[3] for x in pairs], float)
    c = np.array([x[5] for x in pairs], float)
    if sps is not None and len(s) > 3:
        r, pr = sps.pearsonr(s, c)
        rho, prho = sps.spearmanr(s, c)
        say(f"\nКорреляция величин: Пирсон r={r:+.2f} (p={pr:.3g}), "
            f"Спирмен ρ={rho:+.2f} (p={prho:.3g}), точек {len(s)}")
    ratios = np.array([x[8] for x in strong], float)
    ratios = ratios[np.isfinite(ratios)]
    if len(ratios):
        say(f"Отношение center/slot на сильных ячейках: медиана {np.median(ratios):.2f}, "
            f"разброс [{ratios.min():.2f}, {ratios.max():.2f}], n={len(ratios)}")
        say("  (~1 = обе оценки меряют одну и ту же тягу в мм; <1 = центр систематически слабее)")

    text = "\n".join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"\n-> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
