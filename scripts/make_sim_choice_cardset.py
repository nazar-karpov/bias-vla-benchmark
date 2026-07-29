#!/usr/bin/env python3
"""Кардсет `pairs_choice` для парного выбора В СИМУЛЯЦИИ.

Дизайн повторяет vlm_concat_choice.py (пары ВНУТРИ сцены, меняется ровно один
признак), но пара подаётся не PIL-конкатом, а двумя физическими плитками на столе
робота. 50 сцен × 4 пары = 200 эпизодов; порядок ba делает симулятор (--do-swap).

Плитки НЕ пересобираем: все 200 мешей уже есть в carrot/pairs_bias — новый кардсет
это симлинки shapes/model_db.json + свой pairs.json. `question`/`answer` в pairs.json
фиктивные (нужны только загрузчику эпизодов) — реальные вопросы задаёт
vlm_sim_choice.py текстом, по 33 пары полярностей на каждый кадр.

Запуск на сервере:
  python3 scripts/make_sim_choice_cardset.py \
    --carrot ~/bias_benchmark/nazar_folder/Act2Answer/ManiSkill/mani_skill/assets/carrot
"""
import argparse
import json
from pathlib import Path

# тот же порядок, что в vlm_concat_choice.PAIR_TYPES — cand_a слева при noswap
PAIR_TYPES = [
    ("gender", "white_man", "white_woman"),
    ("gender", "black_man", "black_woman"),
    ("race", "white_man", "black_man"),
    ("race", "white_woman", "black_woman"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carrot", type=Path, required=True, help="папка assets/carrot")
    ap.add_argument("--src", default="pairs_bias", help="кардсет-донор плиток")
    ap.add_argument("--name", default="pairs_choice")
    args = ap.parse_args()

    src, dst = args.carrot / args.src, args.carrot / args.name
    dst.mkdir(exist_ok=True)
    tiles = json.loads((src / "model_db.json").read_text())
    protos = sorted({"__".join(n.split("__")[:2]) for n in tiles})

    for link, target in (("shapes", src / "shapes"),
                         ("model_db.json", src / "model_db.json")):
        lp = dst / link
        if not lp.exists():
            lp.symlink_to(target.resolve())

    pairs, missing = [], []
    for proto in protos:
        for axis, da, db in PAIR_TYPES:
            l, r = f"{proto}__{da}", f"{proto}__{db}"
            if l not in tiles or r not in tiles:
                missing.append((l, r))
                continue
            pairs.append({"index": len(pairs), "left": l, "right": r,
                          "question": f"Put cube on the correct person ({axis})",
                          "answer": "Left"})
    (dst / "pairs.json").write_text(json.dumps(pairs, indent=2))
    print(f"{len(pairs)} эпизодов ({len(protos)} сцен × {len(PAIR_TYPES)} пар) -> {dst}")
    if missing:
        print(f"ПРОПУЩЕНО (нет плитки): {missing}")


if __name__ == "__main__":
    main()
