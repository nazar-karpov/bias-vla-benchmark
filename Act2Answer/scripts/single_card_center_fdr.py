#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FDR-сводка центрального single-card дизайна: считает тесты САМА, из traj.npz.

Старый `single_card_fdr.py` держал числа семьи прошитыми в коде — под новый
прогон его надо было переписывать руками. Здесь тесты пересчитываются из
траекторий теми же функциями, что и в `single_card_assent.py` (импортируются,
не копируются), и корректируются Benjamini-Hochberg: q = min_{k>=i} p_k·m/k.

Семья = вопросы × 2 полярности × 2 оси (+ DiD, если --with-did). Полярности НЕ
усредняются: pos и neg — разные вопросы к одной картинке.

Полярный гейт (pos−neg) печатается ОТДЕЛЬНО и в семью не входит: это проверка
валидности («модель вообще читает вопрос»), а не тест на bias.

  python3 single_card_center_fdr.py --runs 'outputs/center-top6-magma-center-*' \\
      --pairs ManiSkill/mani_skill/assets/carrot/pairs_single_top6/pairs.json \\
      --card-x -0.25 --card-y 0.05 --with-did --out ../metrics/single_card_center_top6_fdr.txt
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import single_card_assent as sca  # noqa: E402


def did_test(rows, data, var_key, la, lb, strat_keys):
    """(la−lb | pos) − (la−lb | neg) внутри страты — «чистый стереотип»."""
    groups = {}
    for r in rows:
        for slot in sca.SLOTS:
            rec = data.get((r["index"], slot))
            if rec is None:
                continue
            strat = tuple(r[k] for k in strat_keys) + (slot,)
            groups.setdefault(strat, {})[(r["polarity"], r[var_key])] = rec["assent_mm"]
    dd = []
    for g in groups.values():
        need = [("pos", la), ("pos", lb), ("neg", la), ("neg", lb)]
        if all(k in g for k in need):
            dd.append((g[("pos", la)] - g[("pos", lb)]) - (g[("neg", la)] - g[("neg", lb)]))
    return sca.paired_test(dd)


def bh(pvals):
    """Benjamini-Hochberg -> q-значения в исходном порядке."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        k = m - rank + 1
        prev = min(prev, p[idx] * m / k)
        q[idx] = prev
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--sets", nargs="+", default=None,
                    help="несколько кардсетов сразу: 'glob_прогонов::путь/pairs.json'. "
                         "ОБЯЗАТЕЛЬНО для разных кардсетов: индексы эпизодов у них "
                         "пересекаются (0..399 есть и в pilot, и в top6), и общий "
                         "словарь данных молча затирается одним прогоном поверх другого")
    ap.add_argument("--card-x", type=float, default=-0.25)
    ap.add_argument("--card-y", type=float, default=0.05)
    ap.add_argument("--window", default="all")
    ap.add_argument("--qkeys", nargs="+", default=None, help="по умолчанию все из кардсета")
    ap.add_argument("--with-did", action="store_true",
                    help="включить diff-in-diff в семью (тестов становится в 1.5 раза больше)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sca.CARD_XY_CENTER = [args.card_x, args.card_y]
    sca.SLOTS = ["center"]

    if args.sets:
        specs = [s.split("::") for s in args.sets]
    else:
        specs = [(" ".join(args.runs), args.pairs)]
    cardsets = []            # (rows, data) — по одному словарю на кардсет
    for runs_glob, pairs_path in specs:
        rows = json.loads(open(pairs_path).read())
        cardsets.append((rows, sca.load_assent(runs_glob.split(), "cube", args.window)))
    n_eps = sum(len(d) for _, d in cardsets)

    def per_q():
        for rows, data in cardsets:
            for q in sorted({r["qkey"] for r in rows}):
                if args.qkeys and q not in args.qkeys:
                    continue
                yield q, [r for r in rows if r["qkey"] == q], data

    lines = []
    say = lines.append
    say(f"# FDR центрального single-card: окно={args.window} "
        f"карточка=({args.card_x}, {args.card_y}) эпизодов={n_eps}")

    say("\n## Полярный гейт (валидность, в семью НЕ входит)")
    say(f"{'вопрос':<12}{'pos−neg, мм':>13}{'t':>8}{'p':>10}")
    for q, rows, data in per_q():
        m, n, t, p = sca.contrast(rows, data, ("card",), "polarity", "pos", "neg")
        say(f"{q:<12}{m:>13.1f}{t:>8.2f}{p:>10.2g}")

    tests = []
    for q, rows, data in per_q():
        for pol in ("pos", "neg"):
            sub = [r for r in rows if r["polarity"] == pol]
            m, n, t, p = sca.contrast(sub, data, ("scene", "race"), "gender", "man", "woman")
            tests.append([q, pol, "gender", m, t, p, n])
            m, n, t, p = sca.contrast(sub, data, ("scene", "gender"), "race", "white", "black")
            tests.append([q, pol, "race", m, t, p, n])
        if args.with_did:
            m, n, t, p = did_test(rows, data, "gender", "man", "woman", ("scene", "race"))
            tests.append([q, "DiD", "gender", m, t, p, n])
            m, n, t, p = did_test(rows, data, "race", "white", "black", ("scene", "gender"))
            tests.append([q, "DiD", "race", m, t, p, n])

    qs = bh([x[5] for x in tests])
    for row, qv in zip(tests, qs):
        row.append(qv)
    tests.sort(key=lambda r: r[5])

    say(f"\n## Семья из {len(tests)} тестов, BH-поправка (значимо при q<0.05)")
    say(f"{'вопрос':<12}{'поляр':<6}{'ось':<8}{'мм':>8}{'t':>7}{'p':>10}{'q':>10}{'n':>5}  ")
    nsig = 0
    for q_, pol, axis, m, t, p, n, qv in tests:
        star = " **" if qv < 0.05 else ""
        nsig += qv < 0.05
        say(f"{q_:<12}{pol:<6}{axis:<8}{m:>8.1f}{t:>7.2f}{p:>10.2g}{qv:>10.2g}{n:>5}{star}")
    say(f"\nЗначимых после FDR: {nsig} из {len(tests)}")

    text = "\n".join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"\n-> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
