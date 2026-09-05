#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сводный манифест кадров для кардсета из gen_pairs_cardset.py.

manifest.csv — строка на (pair uid_base × порядок ab/ba × конфиг): pair_id/uid_base, source,
config, frame, left_image/right_image (что НА КАДРЕ), order, board_xy_scale, tile_y + все
атрибуты строки таблицы пар (attr_*). Вопросы к парам — в questions.tsv команды (колонка
source_dataset); для deprecated-пар их uid-вопросы в manifest_deprecated.csv
(uid × конфиг → кадр, question_vla, question_vlm), если переданы --deprecated-vla/--deprecated-vlm.
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-dir", type=Path, required=True, help="carrot/<name> (pairs_meta.json)")
    ap.add_argument("--frames-root", type=Path, required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--deprecated-vla", type=Path, nargs="*", default=[])
    ap.add_argument("--deprecated-vlm", type=Path, nargs="*", default=[])
    args = ap.parse_args()

    meta = json.loads((args.assets_dir / "pairs_meta.json").read_text(encoding="utf-8"))
    attr_keys = sorted({k for m in meta for k in m["attrs"]})
    rows, missing = [], 0
    frames = {}
    for cfg in args.configs:
        for r in csv.DictReader((args.frames_root / cfg / "frames.csv").open(encoding="utf-8", newline="")):
            frames[(cfg, r["uid_base"], r["order"])] = r
    for cfg in args.configs:
        for m in meta:
            for order in ("ab", "ba"):
                f = frames.get((cfg, m["uid_base"], order))
                if f is None or not (args.frames_root / f["frame"]).exists():
                    missing += 1
                    continue
                row = {"pair_id": m["uid_base"], "source": m["source"], "config": cfg, "order": order,
                       "frame": f["frame"], "left_image": f["left_image"], "right_image": f["right_image"],
                       "board_xy_scale": f["board_xy_scale"], "tile_y": f["tile_y"]}
                for k in attr_keys:
                    row["attr_" + k] = m["attrs"].get(k, "")
                rows.append(row)
    p = args.frames_root / "manifest.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    stats = {"rows": len(rows), "pairs": len(meta), "configs": args.configs, "missing": missing,
             "expected": len(meta) * 2 * len(args.configs)}

    # deprecated uid-манифесты команды → кадр по паре картинок (в любом порядке)
    if args.deprecated_vla:
        by_imgs = {}
        for cfg in args.configs:
            for m in meta:
                for order in ("ab", "ba"):
                    f = frames.get((cfg, m["uid_base"], order))
                    if f:
                        by_imgs[(cfg, f["left_image"], f["right_image"])] = f["frame"]
        vlm = {}
        for v in args.deprecated_vlm:
            for r in csv.DictReader(v.open(encoding="utf-8", newline="")):
                vlm[r["uid"]] = r["question_vlm"]
        drows, dmiss = [], 0
        for c in args.deprecated_vla:
            for r in csv.DictReader(c.open(encoding="utf-8", newline="")):
                for cfg in args.configs:
                    fr = by_imgs.get((cfg, r["left_image"], r["right_image"]))
                    if fr is None:
                        dmiss += 1
                        continue
                    drows.append({"uid": r["uid"], "config": cfg, "frame": fr,
                                  "question_vla": r["question_vla"], "question_vlm": vlm.get(r["uid"], ""),
                                  "left_image": r["left_image"], "right_image": r["right_image"],
                                  **{k: v for k, v in r.items() if k not in
                                     ("uid", "question_vla", "left_image", "right_image")}})
        pd = args.frames_root / "manifest_deprecated.csv"
        with pd.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(drows[0].keys())); w.writeheader(); w.writerows(drows)
        stats.update({"deprecated_rows": len(drows), "deprecated_missing": dmiss})
    (args.frames_root / "manifest_stats.json").write_text(json.dumps(stats, indent=1))
    print(stats, "->", p)


if __name__ == "__main__":
    main()
