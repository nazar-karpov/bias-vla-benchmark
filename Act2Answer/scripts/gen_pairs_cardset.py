#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Кардсет Act2Answer из таблиц пар датасета с общего Drive (формат команды).

Источники пар (объединяются, дубликаты по неупорядоченной паре картинок схлопываются):
  --pairs  *.tsv   актуальные таблицы: pair_id, image_1, image_2 (+атрибуты) — ab = (image_1, image_2)
  --deprecated *.csv  старые VLA-манифесты: uid, left_image, right_image — берутся только пары,
                      которых нет в tsv; uid_base = uid без _<attr>_<ab|ba> хвоста
Картинки берутся из --images-root/<путь из таблицы> (расширение подменяется на .jpg, если
квадратные копии сохранены так). Имя плитки = путь без расширения через '_'.

pairs_meta.json: index, uid_base, left_image, right_image, left, right, source (tsv|deprecated),
attrs (все колонки строки tsv). Вопрос/answer в pairs.json фиктивные.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_cardset import build_box_mesh, build_collision_obj, model_db_entry  # noqa: E402


def tile_name(rel: str, prefix: str) -> str:
    p = Path(rel).with_suffix("")
    return prefix + "_" + "_".join(x for x in p.parts if x not in ("original", "data", "source", "images"))


def resolve(root: Path, rel: str) -> Path:
    p = root / rel
    if p.exists():
        return p
    q = p.with_suffix(".jpg")
    return q if q.exists() else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="имя кардсета / префикс плиток")
    ap.add_argument("--pairs", nargs="*", default=[], type=Path)
    ap.add_argument("--deprecated", nargs="*", default=[], type=Path)
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    pairs, seen, skipped = [], {}, []

    def add(a, b, uid_base, source, attrs):
        key = tuple(sorted((a, b)))
        if key in seen:
            return
        pa, pb = resolve(args.images_root, a), resolve(args.images_root, b)
        if not (pa.exists() and pb.exists()):
            skipped.append({"uid_base": uid_base, "missing": str(pa if not pa.exists() else pb)})
            return
        seen[key] = len(pairs)
        pairs.append({"index": len(pairs), "uid_base": uid_base, "left_image": a, "right_image": b,
                      "left": tile_name(a, args.name), "right": tile_name(b, args.name),
                      "source": source, "attrs": attrs})

    for t in args.pairs:
        with t.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                r = {k.strip(): (v or "").strip() for k, v in r.items()}
                add(r["image_1"], r["image_2"], r["pair_id"], f"tsv:{t.stem}",
                    {k: v for k, v in r.items() if k not in ("pair_id", "image_1", "image_2")})
    n_tsv = len(pairs)
    for c in args.deprecated:
        with c.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                uid = r["uid"]
                if uid.endswith("_ba"):
                    continue                      # зеркало ab-строки
                if uid.endswith("_ab"):           # FOCUS/VERI: uid_<attr>_<ab|ba>
                    base = uid.rsplit("_", 2)[0]
                else:                             # VisBias: uid_<attr> без порядка; пара одна на все атрибуты
                    attr = (r.get("attribute") or "").strip()
                    base = uid[: -len(attr) - 1] if attr and uid.endswith("_" + attr) else uid
                add(r["left_image"], r["right_image"], base, f"deprecated:{c.stem}",
                    {k: v for k, v in r.items() if k not in ("uid", "question_vla", "left_image", "right_image")})
    print(f"пар: {len(pairs)} (из tsv {n_tsv}, из deprecated {len(pairs) - n_tsv}); пропущено {len(skipped)}")

    shapes = args.out / "shapes"
    shapes.mkdir(parents=True, exist_ok=True)
    tiles = {}
    for p in pairs:
        tiles[p["left"]] = resolve(args.images_root, p["left_image"])
        tiles[p["right"]] = resolve(args.images_root, p["right_image"])
    model_db = {}
    for name, img in sorted(tiles.items()):
        tdir = shapes / name
        if not (tdir / "textured.glb").exists():
            tdir.mkdir(exist_ok=True)
            build_box_mesh(img).export(tdir / "textured.glb")
            build_collision_obj(tdir / "collision.obj")
        model_db[name] = model_db_entry(name)
    (args.out / "model_db.json").write_text(json.dumps(model_db, indent=2))
    (args.out / "pairs.json").write_text(json.dumps(
        [{"index": p["index"], "left": p["left"], "right": p["right"],
          "question": "Put cube on the left picture", "answer": "Left"} for p in pairs], indent=2))
    (args.out / "pairs_meta.json").write_text(json.dumps(pairs, indent=1, ensure_ascii=False))
    (args.out / "skipped.json").write_text(json.dumps(skipped, indent=1))
    print(f"плиток: {len(tiles)} -> {args.out}")


if __name__ == "__main__":
    main()
