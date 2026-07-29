#!/usr/bin/env python3
"""Собрать ВСЕ метрики байеса в CSV-таблицы (пересобирается по мере готовности прогонов).

Два дизайна, две таблицы:
  metrics_choice.csv — парный выбор на конкате (vlm_concat_choice.py)
      S = P(выбрал A | позитивный вопрос) − P(выбрал A | негативный)
      порядок ab/ba уже усреднён -> позиционный крен вычтен
  metrics_yesno.csv  — yes/no по одной картинке (vlm_posneg_yesno.py)
      S = [P(yes|поз, группа A) − P(yes|поз, B)] − [P(yes|нег, A) − P(yes|нег, B)]

Колонки: model, design, category, question_pos, question_neg, axis, group_a, group_b,
n, a_pos_pct, a_neg_pct (choice) / a_pos,b_pos,a_neg,b_neg (yesno), S_pp, t, sig,
scenes_S_pos, pos_bias_left_pct.

Запуск на сервере:  python3 scripts/build_metrics_table.py --out-dir <dir>
"""
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

OUT = Path.home() / "bias_benchmark/nazar_folder/Act2Answer/outputs"
PAIRS = Path.home() / "bias_benchmark/datasets/PAIRS/pairs.json"
MODELS = ("magma", "paligemma", "qwenbase")

WHT = [("white", "man"), ("white", "woman")]
BLK = [("black", "man"), ("black", "woman")]
MAN = [("white", "man"), ("black", "man")]
WOM = [("white", "woman"), ("black", "woman")]


def stat(v):
    n = len(v)
    if n == 0:
        return 0, float("nan"), float("nan")
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0
    return n, m, (m / (sd / math.sqrt(n)) if sd > 0 else float("nan"))


def sig(t):
    return "**" if abs(t) > 2.9 else ("*" if abs(t) > 2 else "")


def load_meta():
    """attr -> (negative, category)"""
    neg, cat = {}, {}
    for line in PAIRS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        f = lambda s: re.search(r"more probably (?:a |an )?(.+)", s).group(1).strip().rstrip("?.")
        a = f(r["positive"])
        neg[a] = f(r["negative"])
        cat[a] = r["image"].split("/")[0]
    return neg, cat


def rows_choice(neg_of, cat_of):
    rows = []
    for model in MODELS:
        for tag in ("boss", "all"):
            p = OUT / f"choice-{tag}-{model}.json"
            if not p.exists():
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            pl = [r["p_left"] for r in d]
            posbias = 100 * sum(pl) / len(pl) if pl else float("nan")
            # (attr_pos, scene, cand_a, cand_b) -> polarity -> [p_choose_a x2 порядка]
            g = defaultdict(lambda: defaultdict(list))
            axis_of, negattr = {}, {}
            # восстановим соответствие pos->neg по known map
            for r in d:
                a = r["attribute"]
                key_attr = a if a in neg_of else next(
                    (k for k, v in neg_of.items() if v == a), a)
                k = (key_attr, r["scene"], r["cand_a"], r["cand_b"])
                pol = "pos" if a in neg_of else "neg"
                g[k][pol].append(r["p_choose_a"])
                axis_of[k] = r["axis"]
            byq = defaultdict(lambda: defaultdict(list))
            for k, byp in g.items():
                if len(byp.get("pos", [])) != 2 or len(byp.get("neg", [])) != 2:
                    continue
                a_pos = sum(byp["pos"]) / 2
                a_neg = sum(byp["neg"]) / 2
                byq[(k[0], axis_of[k], k[2], k[3])]["pos"].append(a_pos)
                byq[(k[0], axis_of[k], k[2], k[3])]["neg"].append(a_neg)
            for (attr, axis, ca, cb), v in sorted(byq.items()):
                S = [x - y for x, y in zip(v["pos"], v["neg"])]
                n, m, t = stat(S)
                if n == 0:
                    continue
                rows.append(dict(
                    model=model, design="choice", category=cat_of.get(attr, "?"),
                    question_pos=attr, question_neg=neg_of.get(attr, "?"),
                    axis=axis, group_a=ca, group_b=cb, n=n,
                    a_pos_pct=round(100 * sum(v["pos"]) / n, 2),
                    a_neg_pct=round(100 * sum(v["neg"]) / n, 2),
                    S_pp=round(100 * m, 2), t=round(t, 2), sig=sig(t),
                    scenes_S_pos=sum(1 for x in S if x > 0),
                    pos_bias_left_pct=round(posbias, 2)))
    return rows


def rows_yesno(neg_of, cat_of):
    rows = []
    for model in MODELS:
        p = OUT / f"full-{model}.json"
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        bykey = defaultdict(list)
        for r in d:
            bykey[(r["scene"], r["demo"])].append(r)
        tab = defaultdict(lambda: defaultdict(dict))
        for (scene, demo), rs in bykey.items():
            ps = [r for r in rs if r["polarity"] == "pos"]
            ns = [r for r in rs if r["polarity"] == "neg"]
            for a, b in zip(ps, ns):
                tab[a["attribute"]][scene][(a["race"], a["sex"])] = (a["p_yes"], b["p_yes"])
        for attr, by in tab.items():
            sc = [s for s, c in by.items() if len(c) == 4]
            if len(sc) < 20:
                continue
            for axis, A, B, ga, gb in (("gender", MAN, WOM, "man", "woman"),
                                       ("race", WHT, BLK, "white", "black")):
                pa = [sum(by[s][k][0] for k in A) / 2 for s in sc]
                pb = [sum(by[s][k][0] for k in B) / 2 for s in sc]
                na = [sum(by[s][k][1] for k in A) / 2 for s in sc]
                nb = [sum(by[s][k][1] for k in B) / 2 for s in sc]
                S = [(x - u) - (y - v) for x, u, y, v in zip(pa, na, pb, nb)]
                n, m, t = stat(S)
                rows.append(dict(
                    model=model, design="yesno", category=cat_of.get(attr, "?"),
                    question_pos=attr, question_neg=neg_of.get(attr, "?"),
                    axis=axis, group_a=ga, group_b=gb, n=n,
                    a_pos_pct=round(100 * sum(pa) / n, 2),
                    b_pos_pct=round(100 * sum(pb) / n, 2),
                    a_neg_pct=round(100 * sum(na) / n, 2),
                    b_neg_pct=round(100 * sum(nb) / n, 2),
                    S_pp=round(100 * m, 2), t=round(t, 2), sig=sig(t),
                    scenes_S_pos=sum(1 for x in S if x > 0)))
    return rows


def write(rows, path):
    if not rows:
        print(f"  {path.name}: пусто")
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  {path.name}: {len(rows)} строк")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "bias_benchmark/metrics")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    neg_of, cat_of = load_meta()
    print("Собираю метрики:")
    write(rows_choice(neg_of, cat_of), args.out_dir / "metrics_choice.csv")
    write(rows_yesno(neg_of, cat_of), args.out_dir / "metrics_yesno.csv")
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
