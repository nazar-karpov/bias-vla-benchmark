#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Кардсет Act2Answer из FOCUS (REFLECT, face-only counterfactuals) по манифесту команды.

Вход: focus_two_image_selection.csv (uid, question_vla, left_image, right_image, occupation,
left_group, right_group, attribute) — 12960 строк = 2160 пар × ab/ba × 3 атрибута.
Кадр симуляции зависит только от пары картинок и порядка, поэтому кардсет = 2160 уникальных
пар в ориентации «ab» (left_image слева); порядок «ba» даёт do_swap. Вопрос в pairs.json
фиктивный (первый атрибут), answer фиктивный — у пары нет правильного ответа.

Плитки строятся make_cardset.py (геометрия 14.5 см, model_db scale 1.0 → эффективный
масштаб = BOARD_XY_SCALE). Имя плитки = путь картинки без расширения через '_'
(original/focus/ceo/1/Asian_man.jpg → focus_ceo_1_Asian_man).

  python gen_focus_cardset.py --manifest .../focus_two_image_selection.csv \
      --images-root ~/ws/datasets/focus_reflect --out .../carrot/focus_pairs
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_cardset import build_box_mesh, build_collision_obj, model_db_entry  # noqa: E402


def tile_name(rel: str) -> str:
    p = Path(rel)
    return "focus_" + "_".join(p.with_suffix("").parts[-3:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--images-root", required=True, type=Path,
                    help="папка, относительно которой заданы image-пути манифеста (содержит original/focus)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8", newline="")))
    # уникальные пары в ориентации ab; uid_base = uid без _<attr>_<ab|ba>
    pairs, seen = [], {}
    for r in rows:
        if not r["uid"].endswith("_ab"):
            continue
        key = (r["left_image"], r["right_image"])
        if key in seen:
            continue
        base = r["uid"].rsplit("_", 2)[0]          # focus_pair_ceo_01_00
        seen[key] = len(pairs)
        pairs.append({"index": len(pairs), "uid_base": base,
                      "left_image": r["left_image"], "right_image": r["right_image"],
                      "left": tile_name(r["left_image"]), "right": tile_name(r["right_image"]),
                      "occupation": r["occupation"], "left_group": r["left_group"],
                      "right_group": r["right_group"], "question": r["question_vla"]})
    # проверка: каждая ba-строка — зеркало известной ab-пары
    for r in rows:
        if r["uid"].endswith("_ba"):
            assert (r["right_image"], r["left_image"]) in seen, r["uid"]
    print(f"уникальных пар: {len(pairs)} из {len(rows)} строк")

    shapes = args.out / "shapes"
    shapes.mkdir(parents=True, exist_ok=True)
    model_db = {}
    tiles = {}
    for p in pairs:
        for side in ("left", "right"):
            tiles[p[side]] = args.images_root / p[f"{side}_image"]
    for name, img in sorted(tiles.items()):
        assert img.exists(), img
        tdir = shapes / name
        if not (tdir / "textured.glb").exists():
            tdir.mkdir(exist_ok=True)
            build_box_mesh(img).export(tdir / "textured.glb")
            build_collision_obj(tdir / "collision.obj")
        model_db[name] = model_db_entry(name)
    (args.out / "model_db.json").write_text(json.dumps(model_db, indent=2))
    (args.out / "pairs.json").write_text(json.dumps(
        [{"index": p["index"], "left": p["left"], "right": p["right"],
          "question": p["question"], "answer": "Left"} for p in pairs], indent=2))
    (args.out / "pairs_meta.json").write_text(json.dumps(pairs, indent=1))
    print(f"плиток: {len(tiles)}; -> {args.out}")


if __name__ == "__main__":
    main()
