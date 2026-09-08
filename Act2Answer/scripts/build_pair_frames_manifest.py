#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сводный манифест кадров для кардсета из gen_pairs_cardset.py.

manifest.csv — строка на (pair uid_base × порядок ab/ba × конфиг): pair_id/uid_base, source,
config, frame, left_image/right_image (что НА КАДРЕ), order, board_xy_scale, tile_y + все
атрибуты строки таблицы пар (attr_*).

Вопросы к парам лежат в questions.tsv команды и задаются ВСЕ ко ВСЕМ парам всех датасетов.
Манифест их НЕ содержит: кадр от вопроса не зависит, а полный крест (155 вопросов) дал бы
~12 млн строк на четыре датасета. questions.tsv кладётся рядом с кадрами, крест делает
потребитель. Флаг `--questions` разворачивает вопросы в строки, если это всё же нужно.

`--same-scene-column` добавляет производную колонку same_scene: лежат ли обе картинки пары в
одном каталоге, т.е. параллельная ли это пара (тот же кадр, подменено лицо) или снимки из
разных сцен. Нужна для FOCUS, где актуальные таблицы смешивают оба типа: у gender/ethnicity
это ровно same_identity, а у profession колонки same_identity нет вовсе (все пары кросс-сценовые).

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
    ap.add_argument("--questions", type=Path, nargs="*", default=[], help="questions.tsv команды")
    ap.add_argument("--same-scene-column", action="store_true",
                    help="добавить производную колонку same_scene (обе картинки из одного каталога)")
    args = ap.parse_args()

    # Вопросы задаются ВСЕ ко ВСЕМ парам всех датасетов. Колонки source_dataset и source_format
    # в questions.tsv — только происхождение вопроса (из какой задачи исходного датасета он вырос),
    # НЕ фильтр: команда все 150 уже переписала под парный дизайн («Put the cube…» / «A or B»).
    questions = []
    for qf in args.questions:
        with qf.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                questions.append({"question_id": r["question_id"],
                                  "question_source_dataset": r["source_dataset"],
                                  "question_source_format": r["source_format"],
                                  "question_attribute": r["attribute"],
                                  "question_vla": r["question_vla"], "question_vlm": r["question_vlm"]})

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
                if args.same_scene_column:
                    a, b = m["left_image"], m["right_image"]
                    row["same_scene"] = str(a.rsplit("/", 1)[0] == b.rsplit("/", 1)[0]).lower()
                for k in attr_keys:
                    row["attr_" + k] = m["attrs"].get(k, "")
                if questions:
                    rows.extend({**row, **q} for q in questions)
                else:
                    rows.append(row)
    p = args.frames_root / "manifest.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    stats = {"rows": len(rows), "pairs": len(meta), "configs": args.configs, "missing": missing,
             "frames_expected": len(meta) * 2 * len(args.configs),
             "questions": len(questions) or None,
             "expected": len(meta) * 2 * len(args.configs) * max(len(questions), 1)}

    (args.frames_root / "manifest_stats.json").write_text(json.dumps(stats, indent=1))
    print(stats, "->", p)


if __name__ == "__main__":
    main()
