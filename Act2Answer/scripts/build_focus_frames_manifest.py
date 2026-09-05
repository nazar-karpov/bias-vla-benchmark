#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сводный манифест кадров FOCUS: строка на (uid манифеста команды × конфиг раскладки).

Склеивает focus_two_image_selection.csv (VLA-вопрос, порядок ab/ba, атрибут) и
focus_vlm_parallel_two_image_selection.csv (VLM-вопрос «A или B») с frames.csv каждого
конфига. uid тот же, что в манифестах команды → коллега с VLM подставляет кадр по uid.

Проверки: для каждого uid картинка слева на кадре == left_image манифеста (и справа),
все кадры существуют.
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", type=Path, required=True, help="outputs/focus_frames")
    ap.add_argument("--vla", type=Path, required=True)
    ap.add_argument("--vlm", type=Path, required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    args = ap.parse_args()

    vla = list(csv.DictReader(args.vla.open(encoding="utf-8", newline="")))
    vlm = {r["uid"]: r["question_vlm"] for r in csv.DictReader(args.vlm.open(encoding="utf-8", newline=""))}
    out, missing, mismatch = [], 0, 0
    for cfg in args.configs:
        fr = {}
        for r in csv.DictReader((args.frames_root / cfg / "frames.csv").open(encoding="utf-8", newline="")):
            fr[(r["uid_base"], r["order"])] = r
        for r in vla:
            base, order = r["uid"].rsplit("_", 2)[0], r["uid"].rsplit("_", 1)[1]
            f = fr.get((base, order))
            if f is None:
                missing += 1
                continue
            if f["left_image"] != r["left_image"] or f["right_image"] != r["right_image"]:
                mismatch += 1
                continue
            if not (args.frames_root / f["frame"]).exists():
                missing += 1
                continue
            out.append({"uid": r["uid"], "config": cfg, "frame": f["frame"],
                        "question_vla": r["question_vla"], "question_vlm": vlm[r["uid"]],
                        "left_image": r["left_image"], "right_image": r["right_image"],
                        "occupation": r["occupation"], "left_group": r["left_group"],
                        "right_group": r["right_group"], "attribute": r["attribute"],
                        "order": order, "board_xy_scale": f["board_xy_scale"], "tile_y": f["tile_y"]})
    p = args.frames_root / "manifest.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    stats = {"rows": len(out), "configs": args.configs, "missing": missing, "mismatch": mismatch,
             "expected": len(vla) * len(args.configs)}
    (args.frames_root / "manifest_stats.json").write_text(json.dumps(stats, indent=1))
    print(stats, "->", p)


if __name__ == "__main__":
    main()
