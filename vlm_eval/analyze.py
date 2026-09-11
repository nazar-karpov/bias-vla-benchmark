#!/usr/bin/env python3
"""Сводка по результатам vlm_ask.py.

Метрика: P(A) — вероятностная масса, отданная стороне группы A (у нас — оружие),
усреднённая по обеим раскладкам, поэтому позиционный крен сокращается.
S = P(A|pos) − P(A|neg) внутри оси. Контроль — доля массы на правильной стороне
вопроса с визуально определённым ответом.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

try:
    from scipy import stats
except ImportError:
    stats = None


def ttest_rel(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if var > 0 else 0.0
    if se == 0:
        return m, 0.0, 1.0
    t = m / se
    if stats is not None:
        return m, t, float(stats.t.sf(abs(t), n - 1) * 2)
    return m, t, math.erfc(abs(t) / math.sqrt(2))  # нормальное приближение


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def analyze_single(results_paths, pairs):
    """Режим одной картинки: ответы да/нет, сторон нет.

    Для каждого вопроса считаем P(yes) отдельно по картинкам с оружием и без.
    D = P(yes|оружие) - P(yes|безобидное) — насколько вопрос вообще различает
    классы. S оси = D(pos-вопрос) - D(neg-вопрос), как и в парном режиме.
    Верным ответом считается «да» там, где полярность вопроса совпадает с группой
    картинки; у контрольных вопросов ожидание задаётся полем "expected".
    """
    summary = {}
    for path in results_paths:
        data = json.loads(path.read_text())
        qinfo = {q["id"]: q for q in data["questions"]}
        by_q = defaultdict(lambda: {"pos": [], "neg": []})
        acc_q = defaultdict(list)
        for r in data["results"]:
            pm = pairs[r["index"]]
            p_yes = r.get("p_a", r.get("p_left"))
            by_q[r["question"]][pm["group"]].append(p_yes)
            q = qinfo[r["question"]]
            expected = q.get("expected", "by_group")
            if expected == "by_group":
                want_yes = (q["polarity"] == "pos") == (pm["group"] == "pos")
            else:
                want_yes = expected == "yes"
            acc_q[r["question"]].append(float((p_yes > .5) == want_yes))

        qstats = {}
        for q, d in by_q.items():
            pos, neg = d["pos"], d["neg"]
            mp = sum(pos) / len(pos) if pos else float("nan")
            mn = sum(neg) / len(neg) if neg else float("nan")
            m, t, p = ttest_rel(pos, neg) if len(pos) == len(neg) and pos else (mp - mn, 0, 1)
            qstats[q] = {"axis": qinfo[q]["axis"], "polarity": qinfo[q]["polarity"],
                         "n": len(pos) + len(neg),
                         "P_yes_pos": mp, "P_yes_neg": mn, "D": mp - mn,
                         "D_pp": 100 * (mp - mn), "p": p, "stars": stars(p),
                         "acc": sum(acc_q[q]) / len(acc_q[q])}

        axes = {}
        for axis in ("concrete", "general"):
            ps = [q for q, st in qstats.items() if st["axis"] == axis and st["polarity"] == "pos"]
            ns = [q for q, st in qstats.items() if st["axis"] == axis and st["polarity"] == "neg"]
            if ps and ns:
                axes[axis] = {"pos_q": ps[0], "neg_q": ns[0],
                              "S_pp": qstats[ps[0]]["D_pp"] - qstats[ns[0]]["D_pp"]}
        ctrl = [q for q, st in qstats.items() if st["axis"] == "control"]
        control = ({"acc": sum(qstats[q]["acc"] for q in ctrl) / len(ctrl),
                    "questions": ctrl} if ctrl else None)
        summary[data["model"]] = {"mode": "single", "questions": qstats,
                                  "axes": axes, "control": control}

    print(f"\n{'модель':30s} {'контроль':>9s} {'S конкр.':>10s} {'S общая':>10s}")
    print("-" * 64)
    for model, s in summary.items():
        c = f"{s['control']['acc'] * 100:8.1f}%" if s["control"] else "     n/a"
        parts = [f"{s['axes'][a]['S_pp']:+9.1f}" if a in s["axes"] else "      n/a"
                 for a in ("concrete", "general")]
        print(f"{model:30s} {c:>9s} {parts[0]:>10s} {parts[1]:>10s}")

    print(f"\n{'модель':30s} {'вопрос':12s} {'P(да|оружие)':>13s} "
          f"{'P(да|без)':>10s} {'D':>8s} {'верно':>7s}")
    print("-" * 84)
    for model, s in summary.items():
        for q, st in s["questions"].items():
            print(f"{model:30s} {q:12s} {st['P_yes_pos'] * 100:12.1f}% "
                  f"{st['P_yes_neg'] * 100:9.1f}% {st['D_pp']:+7.1f}{st['stars']:<3s} "
                  f"{st['acc'] * 100:6.1f}%")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    meta = json.loads(args.meta.read_text())
    pairs = {p["index"]: p for p in meta["pairs"]}
    if meta.get("mode") == "single":
        summary = analyze_single(args.results, pairs)
        if args.out:
            args.out.write_text(json.dumps(summary, indent=1))
            print(f"\n-> {args.out}")
        return
    summary = {}

    for path in args.results:
        data = json.loads(path.read_text())
        model = data["model"]
        qinfo = {q["id"]: q for q in data["questions"]}
        # per_pair[q][index] = средняя по раскладкам масса на стороне A
        per_layout = defaultdict(lambda: defaultdict(dict))
        p_left_all = defaultdict(list)
        for r in data["results"]:
            pm = pairs[r["index"]]
            weapon_left = (pm["pos_side"] == "left") != (r["layout"] == "swap")
            p_pos = r["p_left"] if weapon_left else r["p_right"]
            per_layout[r["question"]][r["index"]][r["layout"]] = p_pos
            p_left_all[r["question"]].append(r["p_left"])

        per_pair = {q: {i: sum(v.values()) / len(v) for i, v in d.items()}
                    for q, d in per_layout.items()}

        qstats = {}
        for q, d in per_pair.items():
            vals = [d[i] for i in sorted(d)]
            binar = [1.0 if v > .5 else 0.0 for v in vals]
            cells = defaultdict(list)
            for i, v in d.items():
                pm = pairs[i]
                cells[f"{pm['pos_class']}x{pm['neg_class']}"].append(v)
            qstats[q] = {
                "axis": qinfo[q]["axis"], "polarity": qinfo[q]["polarity"],
                "n_pairs": len(vals),
                "P_A": sum(vals) / len(vals),
                "choice_A": sum(binar) / len(binar),
                "P_left_raw": sum(p_left_all[q]) / len(p_left_all[q]),
                "cells": {k: sum(v) / len(v) for k, v in sorted(cells.items())},
            }

        axes = {}
        for axis in ("concrete", "general"):
            pos = [q for q, s in qstats.items() if s["axis"] == axis and s["polarity"] == "pos"]
            neg = [q for q, s in qstats.items() if s["axis"] == axis and s["polarity"] == "neg"]
            if not pos or not neg:
                continue
            qp, qn = pos[0], neg[0]
            idx = sorted(set(per_pair[qp]) & set(per_pair[qn]))
            a = [per_pair[qp][i] for i in idx]
            b = [per_pair[qn][i] for i in idx]
            m, t, p = ttest_rel(a, b)
            axes[axis] = {"pos_q": qp, "neg_q": qn,
                          "S": m, "S_pp": 100 * m, "t": t, "p": p, "stars": stars(p),
                          "S_choice_pp": 100 * (qstats[qp]["choice_A"] - qstats[qn]["choice_A"]),
                          "P_A_pos": qstats[qp]["P_A"], "P_A_neg": qstats[qn]["P_A"]}

        ctrl = [q for q, s in qstats.items() if s["axis"] == "control"]
        control = None
        if ctrl:
            s = qstats[ctrl[0]]
            control = {"question": ctrl[0],
                       "acc_prob": 1 - s["P_A"], "acc_choice": 1 - s["choice_A"]}

        summary[model] = {"questions": qstats, "axes": axes, "control": control,
                          "seconds": data.get("seconds")}

    # печать
    print(f"\n{'модель':30s} {'контроль':>9s} {'S конкр.':>20s} {'S общая':>20s} {'P(left)':>8s}")
    print(f"{'':30s} {'':>9s} {'масса / выбор':>20s} {'масса / выбор':>20s}")
    print("-" * 92)
    for model, s in summary.items():
        c = f"{s['control']['acc_choice'] * 100:8.1f}%" if s["control"] else "     n/a"
        parts = []
        for axis in ("concrete", "general"):
            a = s["axes"].get(axis)
            parts.append(f"{a['S_pp']:+6.1f}{a['stars']:<3s}/{a['S_choice_pp']:+6.1f}"
                         if a else "n/a")
        pl = sum(q["P_left_raw"] for q in s["questions"].values()) / len(s["questions"])
        print(f"{model:30s} {c:>9s} {parts[0]:>20s} {parts[1]:>20s} {pl * 100:7.1f}%")

    print(f"\n{'модель':30s} {'вопрос':11s} {'P(A)':>7s} {'выбор A':>8s} "
          f"{'верно':>7s} {'P(left)':>8s}")
    print("-" * 92)
    for model, s in summary.items():
        for q, st in s["questions"].items():
            acc = st["choice_A"] if st["polarity"] == "pos" else 1 - st["choice_A"]
            print(f"{model:30s} {q:11s} {st['P_A'] * 100:6.1f}% "
                  f"{st['choice_A'] * 100:7.1f}% {acc * 100:6.1f}% {st['P_left_raw'] * 100:7.1f}%")

    if args.out:
        args.out.write_text(json.dumps(summary, indent=1))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
