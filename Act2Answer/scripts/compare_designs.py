#!/usr/bin/env python3
"""Сводная таблица: ОДНА картинка (yes/no) vs ПАРА картинок (парный выбор).

Кладёт рядом эффект одного и того же вопроса, померянный двумя дизайнами, чтобы было
видно, что даёт формат подачи. Пишет metrics_comparison.csv.

S_yesno  — одна картинка: [P(yes|поз,A) − P(yes|поз,B)] − [P(yes|нег,A) − P(yes|нег,B)]
S_choice — пара рядом:    P(выбрал A|поз) − P(выбрал A|нег), усреднено по порядкам ab/ba
ratio    — во сколько раз парный дизайн больше (по модулю)
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load(p):
    return list(csv.DictReader(p.open(encoding="utf-8"))) if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path.home() / "bias_benchmark/metrics")
    args = ap.parse_args()

    yn = load(args.dir / "metrics_yesno.csv")
    ch = load(args.dir / "metrics_choice.csv")

    # yes/no: одна строка на (model, question, axis)
    Y = {(r["model"], r["question_pos"], r["axis"]): r for r in yn}

    # choice: две строки (две пары) на (model, question, axis) -> усредняем
    agg = defaultdict(list)
    for r in ch:
        agg[(r["model"], r["question_pos"], r["axis"])].append(r)

    rows = []
    for k, v in agg.items():
        model, q, axis = k
        S_ch = sum(float(x["S_pp"]) for x in v) / len(v)
        t_ch = sum(float(x["t"]) for x in v) / len(v)
        n_ch = sum(int(x["n"]) for x in v)
        w_ch = sum(int(x["scenes_S_pos"]) for x in v)
        y = Y.get(k)
        S_yn = float(y["S_pp"]) if y else None
        t_yn = float(y["t"]) if y else None
        ratio = (abs(S_ch) / abs(S_yn)) if (S_yn not in (None, 0)) else None
        rows.append(dict(
            model=model, category=v[0]["category"], question_pos=q,
            question_neg=v[0]["question_neg"], axis=axis,
            S_yesno_pp=round(S_yn, 2) if S_yn is not None else "",
            t_yesno=round(t_yn, 2) if t_yn is not None else "",
            n_yesno=y["n"] if y else "",
            S_choice_pp=round(S_ch, 2), t_choice=round(t_ch, 2),
            n_choice=n_ch, scenes_pos_choice=f"{w_ch}/{n_ch}",
            ratio_choice_over_yesno=round(ratio, 1) if ratio else "",
            same_sign=("да" if (S_yn is not None and S_yn * S_ch > 0) else
                       ("нет" if S_yn is not None else "")),
        ))
    rows.sort(key=lambda r: (r["model"], -abs(r["S_choice_pp"])))
    out = args.dir / "metrics_comparison.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{out}: {len(rows)} строк")

    # краткая сводка в консоль
    for model in ("magma", "paligemma", "qwenbase"):
        sub = [r for r in rows if r["model"] == model and r["S_yesno_pp"] != ""]
        if not sub:
            continue
        big = [r for r in sub if abs(r["S_choice_pp"]) >= 5]
        agree = sum(1 for r in big if r["same_sign"] == "да")
        med = sorted(r["ratio_choice_over_yesno"] for r in big
                     if r["ratio_choice_over_yesno"] != "")
        print(f"  {model:11} эффектов≥5пп(пара): {len(big):3d}  знак совпал с одной "
              f"картинкой: {agree}/{len(big)}  медиана усиления: "
              f"×{med[len(med)//2] if med else '—'}")


if __name__ == "__main__":
    main()
