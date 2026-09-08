#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сводный манифест кадров для кардсета из gen_pairs_cardset.py.

manifest.csv — строка на (pair uid_base × порядок ab/ba × конфиг): pair_id/uid_base, source,
config, frame, left_image/right_image (что НА КАДРЕ), order, board_xy_scale, tile_y + все
атрибуты строки таблицы пар (attr_*). Вопросы к парам — в questions.tsv команды (колонка
source_dataset).

Deprecated-манифесты команды здесь больше не участвуют (08.09.2026): раньше по ним писался
отдельный manifest_deprecated.csv, но кардсет теперь строится только из актуальных tsv, и такой
файл либо дублировал те же пары (VERI), либо тянул устаревшие (VisBias).
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

    (args.frames_root / "manifest_stats.json").write_text(json.dumps(stats, indent=1))
    print(stats, "->", p)


if __name__ == "__main__":
    main()
