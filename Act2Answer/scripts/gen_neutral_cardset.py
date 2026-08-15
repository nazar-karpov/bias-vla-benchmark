#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Нейтральный кардсет: ОБЕ плитки визуально одинаковые, вопрос без семантики.

Зачем. Позиционный крен мы всегда мерили ВНУТРИ задачи с семантикой и не могли
отделить моторную привычку от влияния картинки. Здесь семантики нет: обе плитки
— одна текстура, вопрос нейтральный. Отклонение от 50/50 = ЧИСТЫЙ крен.

⚠️ ГЛАВНАЯ ГРАБЛЯ (обожглись 15.08): env держит плитки в словаре objs_board ПО
ИМЕНИ. Если в паре left==right (одно имя), это ФИЗИЧЕСКИ ОДИН объект: вторая
позиция затирает первую, boardL_y==boardR_y, все эпизоды одинаковые и swap
ничего не меняет. Поэтому здесь для каждого цвета создаются ДВА имени-клона
(`white__a`, `white__b`) — разные акторы с одной и той же текстурой (симлинки
на один меш + записи в model_db).

Режимы (--mode):
  plain   — обе плитки одного цвета (клоны). Базовая линия крена.
  colors  — слева и справа РАЗНЫЕ цвета, вопрос по-прежнему нейтральный:
            ловит визуальное притяжение к цвету/яркости.

Запуск:
  python gen_neutral_cardset.py --name neutral_plain --mode plain --n 200
"""
import argparse
import json
import os
from pathlib import Path

CARROT = Path("/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer/"
              "ManiSkill/mani_skill/assets/carrot")
SRC = CARROT / "ceiling_color"

QUESTIONS = [
    "Put cube on the tile",
    "Put cube on a tile",
    "Place the cube on the plate",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["plain", "colors"], default="plain")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--colors", nargs="+",
                    default=["white", "green", "blue", "red", "yellow",
                             "orange", "purple", "black"])
    args = ap.parse_args()

    src_db = json.loads((SRC / "model_db.json").read_text())
    cols = [c for c in args.colors if c in src_db]
    assert cols, f"нет цветов {args.colors} в {SRC}/model_db.json"

    dst = CARROT / args.name
    (dst / "shapes").mkdir(parents=True, exist_ok=True)

    # два имени-клона на каждый цвет: разные акторы, одинаковая текстура
    db = {}
    for c in cols:
        for suf in ("a", "b"):
            nm = f"{c}__{suf}"
            rec = dict(src_db[c])
            rec["name"] = nm
            rec["sign"] = nm
            db[nm] = rec
            link = dst / "shapes" / nm
            if not link.exists():
                os.symlink(SRC / "shapes" / c, link)

    pairs = []
    for i in range(args.n):
        if args.mode == "colors":
            lc = cols[i % len(cols)]
            rc = cols[(i // len(cols) + i + 1) % len(cols)]
            if rc == lc:
                rc = cols[(cols.index(lc) + 1) % len(cols)]
        else:
            lc = rc = cols[i % len(cols)]
        pairs.append({
            "index": i,
            "left": f"{lc}__a",
            "right": f"{rc}__b",
            "question": QUESTIONS[i % len(QUESTIONS)],
            # правильного ответа нет — поле формальное, success по нему НЕ считать
            "answer": "Left" if i % 2 == 0 else "Right",
            "qkey": "neutral",
            "polarity": "pos",
            "axis": "position",
        })

    (dst / "pairs.json").write_text(json.dumps(pairs, indent=1, ensure_ascii=False))
    (dst / "model_db.json").write_text(json.dumps(db, indent=1))
    print(f"{args.name}: {len(pairs)} пар, режим={args.mode}, "
          f"имён плиток={len(db)} (клоны a/b)")
    print(f"  пример: {pairs[0]['left']} | {pairs[0]['right']}")
    print(f"  -> {dst}")


if __name__ == "__main__":
    main()
