#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Кардсет для VLA-прогона: пары существующего кардсета × выбранные вопросы из questions.tsv.

Симулятор берёт текст инструкции из pairs.json (поле question), поэтому одна запись =
один эпизод = (пара × вопрос). Плитки (shapes) и model_db переиспользуются из исходного
кардсета симлинком — они не зависят от вопроса.

Порядок записей: сначала все пары первого вопроса, потом второго и т.д. Это позволяет
шардировать прогон по вопросам через --start-id/--count.

  python gen_question_cardset.py --src focus_pairs --out focus_q10 \
      --questions ~/ws/datasets/gdrive_meta/questions.tsv --question-ids-file q10.txt \
      --exclude-source profession
"""
import argparse
import csv
import json
import os
from pathlib import Path

# ось и полярность для наших десяти вопросов: нужны анализу (полярный контроль)
AXIS = {
    "visbias_mcq_income_gt_100000": ("income", "pos"),
    "visbias_mcq_income_lt_20000": ("income", "neg"),
    "visbias_mcq_education_doctorate": ("education", "pos"),
    "visbias_form_education_no_education": ("education", "neg"),
    "visbias_form_criminal_record_no": ("criminal", "pos"),
    "visbias_form_criminal_record_yes": ("criminal", "neg"),
    "visbias_form_disability_status_none": ("health", "pos"),
    "visbias_form_disability_status_mental_disorder": ("health", "neg"),
    "visbias_form_occupation_doctor": ("employment", "pos"),
    "visbias_form_occupation_unemployed": ("employment", "neg"),
    # g10 (12.09.2026): гендерная программа на 5 моделей — 3 пары в короткой форме PAIRS
    # (на ней подтверждены эффекты) + 2 категории VisBias в формулировке команды
    # (чтобы данные Magma q10 по ним переиспользовались). Файл questions_g10.tsv.
    "pairs_pilot": ("pilot", "pos"),
    "pairs_flight_attendant": ("pilot", "neg"),
    "pairs_janitor": ("janitor", "pos"),
    "pairs_stay_at_home_parent": ("janitor", "neg"),
    "pairs_wealthy": ("wealthy", "pos"),
    "pairs_poor": ("wealthy", "neg"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carrot", type=Path,
                    default=Path(os.environ.get("REPO_ROOT", ".")) / "ManiSkill/mani_skill/assets/carrot")
    ap.add_argument("--src", required=True, help="исходный кардсет (pairs_meta.json + shapes)")
    ap.add_argument("--out", required=True, help="имя нового кардсета")
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--question-ids", nargs="+", required=True)
    ap.add_argument("--exclude-source", default="", help="подстрока в source: такие пары пропускать")
    ap.add_argument("--include-source", default="",
                    help="подстрока в source: оставить ТОЛЬКО такие пары (напр. tsv:gender)")
    ap.add_argument("--contrasts", default="",
                    help="e10: оставить пары только этих контрастов, напр. white-black,white-asian "
                         "(по attrs ethnicity_1/2 или skin_color_1/2, порядок внутри пары любой)")
    args = ap.parse_args()

    src, out = args.carrot / args.src, args.carrot / args.out
    meta = json.loads((src / "pairs_meta.json").read_text(encoding="utf-8"))
    if args.exclude_source:
        meta = [m for m in meta if args.exclude_source not in m["source"]]
    if args.include_source:
        meta = [m for m in meta if args.include_source in m["source"]]
    if args.contrasts:
        want = {frozenset(c.split("-")) for c in args.contrasts.split(",")}
        def _pair_set(m):
            a = m["attrs"]
            return frozenset((a.get("ethnicity_1") or a.get("skin_color_1"), a.get("ethnicity_2") or a.get("skin_color_2")))
        meta = [m for m in meta if _pair_set(m) in want]
    if not meta:
        raise SystemExit("после фильтров не осталось ни одной пары")

    qrows = {r["question_id"]: r for r in
             csv.DictReader(args.questions.open(encoding="utf-8-sig", newline=""), delimiter="\t")}
    missing = [q for q in args.question_ids if q not in qrows]
    if missing:
        raise SystemExit(f"нет таких вопросов в {args.questions}: {missing}")

    out.mkdir(parents=True, exist_ok=True)
    # плитки не зависят от вопроса — переиспользуем
    for name in ("shapes", "model_db.json"):
        link = out / name
        if not link.exists():
            link.symlink_to(os.path.relpath(src / name, out))

    # у разных таблиц пар разный набор колонок (gender.tsv не знает про ethnicity_1 и т.д.)
    attr_keys = sorted({k for m in meta for k in m["attrs"]})
    entries, rows = [], []
    for qid in args.question_ids:
        q = qrows[qid]
        axis, pol = AXIS.get(qid, (q["attribute"], ""))
        for m in meta:
            i = len(entries)
            entries.append({"index": i, "left": m["left"], "right": m["right"],
                            "question": q["question_vla"], "answer": "Left",
                            "qkey": qid, "polarity": pol, "axis": axis})
            rows.append({"index": i, "pair_id": m["uid_base"], "question_id": qid,
                         "axis": axis, "polarity": pol, "source": m["source"],
                         "left_image": m["left_image"], "right_image": m["right_image"],
                         "same_scene": str(Path(m["left_image"]).parent == Path(m["right_image"]).parent).lower(),
                         **{"attr_" + k: m["attrs"].get(k, "") for k in attr_keys}})

    (out / "pairs.json").write_text(json.dumps(entries, indent=1, ensure_ascii=False))
    with (out / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{args.out}: пар {len(meta)} × вопросов {len(args.question_ids)} = {len(entries)} эпизодов "
          f"(×2 порядка = {2*len(entries)}) -> {out}")


if __name__ == "__main__":
    main()
