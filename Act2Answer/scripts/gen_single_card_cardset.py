#!/usr/bin/env python3
"""Кардсет для ОДИНОЧНОЙ карточки (single-card assent design).

Идея: вместо пары плиток на столе одна карточка с одним человеком; инструкция
та же, что в парном дизайне ("Put cube on the pilot"). Метрика — дистанция,
которую куб преодолел к карточке (непрерывный "assent", аналог yes/no по одной
картинке в VLM-ветке). Позиционный крен снимается тем, что do_swap ставит
карточку в левый/правый слот (см. A2A_SINGLE_TILE в put_on_in_scene_multi_v4).

Из исходного парного кардсета (по умолчанию pairs_q33_full) берём все уникальные
плитки, встречающиеся в строках выбранных qkey, и раскладываем:
  для каждого qkey -> для каждой полярности (pos, neg) -> по эпизоду на плитку.
В pairs.json поле right дублирует left (требование single-tile режима),
answer="Left" фиктивен. Дополнительные поля card/scene/race/gender — для анализа.

Пример (пилотный кардсет на самом сильном вопросе полного креста):
  python3 gen_single_card_cardset.py --qkeys pilot --out pairs_single_pilot
"""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="pairs_q33_full",
                    help="исходный парный кардсет (откуда вопросы, плитки и model_db)")
    ap.add_argument("--qkeys", nargs="+", required=True, help="какие вопросы включить")
    ap.add_argument("--out", required=True, help="имя нового кардсета")
    ap.add_argument("--asset-path", default=None,
                    help="папка carrot/ (по умолчанию REPO_ROOT/ManiSkill/mani_skill/assets/carrot)")
    args = ap.parse_args()

    if args.asset_path is None:
        repo = Path(os.environ["REPO_ROOT"])
        args.asset_path = repo / "ManiSkill" / "mani_skill" / "assets" / "carrot"
    args.asset_path = Path(args.asset_path)

    src_dir = args.asset_path / args.source
    out_dir = args.asset_path / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    src_pairs = json.loads((src_dir / "pairs.json").read_text())
    src_db = json.loads((src_dir / "model_db.json").read_text())

    # вопросы и плитки по (qkey, polarity); порядок полярностей фиксированный
    entries = []
    blocks = []
    used_tiles = set()
    for qkey in args.qkeys:
        rows = [p for p in src_pairs if p["qkey"] == qkey]
        assert rows, f"qkey {qkey!r} не найден в {args.source}/pairs.json"
        for pol in ("pos", "neg"):
            pol_rows = [p for p in rows if p["polarity"] == pol]
            assert pol_rows, f"{qkey}/{pol}: нет строк в исходном кардсете"
            question = pol_rows[0]["question"]
            assert all(p["question"] == question for p in pol_rows)
            tiles = sorted({t for p in pol_rows for t in (p["left"], p["right"])})
            start = len(entries)
            for tile in tiles:
                # occupations__airplane__white_man -> scene=airplane, race=white, gender=man
                cat, scene, demo = tile.split("__")
                race, gender = demo.split("_", 1)
                entries.append({
                    "index": len(entries),
                    "left": tile,
                    "right": tile,   # дубль для A2A_SINGLE_TILE (ставится только left)
                    "question": question,
                    "answer": "Left",  # фиктивен: у bias-вопроса нет правильного ответа
                    "qkey": qkey,
                    "polarity": pol,
                    "card": tile,
                    "category": cat,
                    "scene": scene,
                    "race": race,
                    "gender": gender,
                })
                used_tiles.add(tile)
            blocks.append({
                "start_id": start,
                "end_id": len(entries) - 1,
                "qkey": qkey,
                "polarity": pol,
                "question_phrase": question.replace("Put cube on ", ""),
            })

    missing = used_tiles - set(src_db)
    assert not missing, f"плиток нет в model_db: {sorted(missing)[:5]}"
    out_db = {k: src_db[k] for k in sorted(used_tiles)}

    (out_dir / "pairs.json").write_text(json.dumps(entries, indent=1))
    (out_dir / "model_db.json").write_text(json.dumps(out_db, indent=1))
    (out_dir / "blocks.json").write_text(json.dumps(blocks, indent=2))

    # shapes — симлинк на тот же таргет, что у исходного кардсета
    src_shapes = src_dir / "shapes"
    target = src_shapes.resolve()
    link = out_dir / "shapes"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)

    print(f"{args.out}: {len(entries)} эпизодов, {len(out_db)} плиток, "
          f"{len(blocks)} блоков -> {out_dir}")
    for b in blocks:
        print(f"  [{b['start_id']:4d}-{b['end_id']:4d}] {b['qkey']}/{b['polarity']}: "
              f"{b['question_phrase']}")


if __name__ == "__main__":
    main()
