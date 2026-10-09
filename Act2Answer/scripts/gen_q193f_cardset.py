#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""q193f: банк FairACT (193 вопроса, questions_q193.tsv) на ВСЕХ парах FOCUS / VisBias — gender 500 + ethnicity 2500
(10 контрастов × 250), без profession — как в Table IV статьи. 193 × 3000 = 579 000 эпизодов на порядок.

Порядок «раундами», чтобы любой готовый префикс прогона был сбалансированной подвыборкой (на одной ноде полный
объём — недели, а ноду могут забрать раньше):
  раунд r < 50:  на каждый вопрос 10 пар gender (по перестановке) + 10 пар ethnicity (по одной на каждый контраст);
  раунд r ≥ 50:  только ethnicity, 10 пар (по одной на контраст);
  всего 250 раундов. После раунда r у каждого вопроса 10·min(r,50) пар gender и 10·r пар ethnicity (r — число
  готовых раундов). Внутри раунда — блоки по вопросам в порядке questions_q193.tsv.
Перестановки фиксированы (--seed), колонка round в episodes.csv.

  REPO_ROOT=… python gen_q193f_cardset.py --src focus_pairs --out focus_q193f
  REPO_ROOT=… python gen_q193f_cardset.py --src visbias_frames --out visbias_q193f
"""
import argparse
import csv
import json
import os
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carrot", type=Path,
                    default=Path(os.environ.get("REPO_ROOT", ".")) / "ManiSkill/mani_skill/assets/carrot")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--questions", type=Path, default=Path(__file__).resolve().parent / "questions_q193.tsv")
    ap.add_argument("--seed", type=int, default=193)
    ap.add_argument("--per-round", type=int, default=10, help="пар gender на вопрос за раунд (ethnicity — по 1 на контраст)")
    ap.add_argument("--part", default="0/1",
                    help="K/M — взять вопросы с позициями i %% M == K (две ноды по 4 карты: 0/2 и 1/2; чередование "
                         "сохраняет в каждой части все 7 категорий; перестановки пар общие для всех частей)")
    args = ap.parse_args()
    part_k, part_m = (int(x) for x in args.part.split("/"))

    src, out = args.carrot / args.src, args.carrot / args.out
    meta = json.loads((src / "pairs_meta.json").read_text(encoding="utf-8"))
    gender = [m for m in meta if m["source"] == "tsv:gender"]
    eth = {}
    for m in meta:
        if m["source"] == "tsv:ethnicity":
            a = m["attrs"]
            eth.setdefault(tuple(sorted((a["ethnicity_1"], a["ethnicity_2"]))), []).append(m)
    contrasts = sorted(eth)
    rng = random.Random(args.seed)
    rng.shuffle(gender)
    for c in contrasts:
        rng.shuffle(eth[c])
    n_eth_rounds = min(len(v) for v in eth.values())
    assert all(len(v) == n_eth_rounds for v in eth.values()), "контрасты разного размера"
    k = args.per_round
    n_rounds = max(n_eth_rounds, (len(gender) + k - 1) // k)

    questions = list(csv.DictReader(args.questions.open(encoding="utf-8-sig", newline=""), delimiter="\t"))
    questions = [q for i, q in enumerate(questions) if i % part_m == part_k]
    out.mkdir(parents=True, exist_ok=True)
    for name in ("shapes", "model_db.json"):
        link = out / name
        if not link.exists():
            link.symlink_to(os.path.relpath(src / name, out))

    attr_keys = sorted({kk for m in gender + [x for v in eth.values() for x in v] for kk in m["attrs"]})
    entries, rows = [], []
    for r in range(n_rounds):
        batch = gender[r * k:(r + 1) * k] + [eth[c][r] for c in contrasts if r < len(eth[c])]
        for q in questions:
            for m in batch:
                i = len(entries)
                entries.append({"index": i, "left": m["left"], "right": m["right"], "question": q["question_vla"],
                                "answer": "Left", "qkey": q["question_id"], "polarity": q.get("polarity") or "",
                                "axis": q["attribute"]})
                rows.append({"index": i, "pair_id": m["uid_base"], "question_id": q["question_id"],
                             "axis": q["attribute"], "polarity": q.get("polarity") or "", "source": m["source"],
                             "left_image": m["left_image"], "right_image": m["right_image"],
                             "same_scene": str(Path(m["left_image"]).parent == Path(m["right_image"]).parent).lower(),
                             **{"attr_" + kk: m["attrs"].get(kk, "") for kk in attr_keys},
                             "stable_id": q.get("stable_id", ""), "category": q.get("category", ""), "round": r})
    (out / "pairs.json").write_text(json.dumps(entries, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    with (out / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{args.out}: gender {len(gender)}, ethnicity {sum(len(v) for v in eth.values())} ({len(contrasts)} контрастов), "
          f"вопросов {len(questions)}, раундов {n_rounds} → {len(entries)} эп. на порядок; "
          f"раунд 0..{(len(gender) + k - 1) // k - 1}: {len(questions) * (k + len(contrasts))} эп., дальше "
          f"{len(questions) * len(contrasts)} эп.")


if __name__ == "__main__":
    main()
