#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Первые кадры симуляции для кардсета focus_pairs в заданной раскладке плиток.

Один процесс = одна раскладка (BOARD_XY_SCALE читается env при импорте). Для каждой пары
рендерятся оба порядка: ab (noswap: left_image слева) и ba (do_swap). Кадр = obs-камера
3rd_view_camera 640×480, ровно то, что видит VLA. Модель не нужна.

Выход: <out>/<config>/<uid_base>_<ab|ba>.png + <out>/<config>/frames.csv
(uid_base, order, frame, left_image, right_image, occupation, left_group, right_group —
left/right здесь = как на кадре, т.е. для ba уже переставлены).

  BOARD_XY_SCALE=1.0 A2A_TILE_Y=0.155 python render_focus_frames.py --config a2a_orig --out ...
  BOARD_XY_SCALE=1.2 A2A_TILE_Y=0.14  python render_focus_frames.py --config andrey_s1p2_y0p14 --out ...
"""
import argparse
import csv
import json
import os
import sys
import types
from pathlib import Path

from PIL import Image

A = Path(os.environ.get("REPO_ROOT", "/home/moskalenko/ws/bias-vla-benchmark-main/Act2Answer"))
sys.path.insert(0, str(A / "scripts"))


def _stub_heavy_imports():
    for mod, attrs in (("safetensors", ["safe_open"]),
                       ("huggingface_hub", ["snapshot_download"]),
                       ("transformers", ["AutoModelForCausalLM", "AutoProcessor"])):
        if mod not in sys.modules:
            m = types.ModuleType(mod)
            for a in attrs:
                setattr(m, a, None)
            sys.modules[mod] = m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="focus_pairs")
    ap.add_argument("--config", required=True, help="имя раскладки = подпапка вывода")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--count", type=int, default=0, help="0 = все пары")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--episode-len", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _stub_heavy_imports()
    from magma_vlm_qa import render_first_frames_chunked  # noqa: E402

    asset_path = A / "ManiSkill" / "mani_skill" / "assets" / "carrot"
    meta = {p["index"]: p for p in json.loads((asset_path / args.assets / "pairs_meta.json").read_text())}
    count = args.count if args.count > 0 else len(meta)
    d = args.out / args.config
    d.mkdir(parents=True, exist_ok=True)
    print(f"config={args.config} BOARD_XY_SCALE={os.environ.get('BOARD_XY_SCALE')} "
          f"A2A_TILE_Y={os.environ.get('A2A_TILE_Y')} A2A_TILE_YC={os.environ.get('A2A_TILE_YC')} "
          f"pairs={count}", flush=True)

    rows = []
    for do_swap in (False, True):
        order = "ba" if do_swap else "ab"
        # пропуск уже отрисованного (докат после обрыва)
        todo = [i for i in range(count) if not (d / f"{meta[i]['uid_base']}_{order}.png").exists()]
        if not todo:
            print(f"{order}: всё уже есть", flush=True)
        # render_first_frames_chunked идёт подряд по start/count; рендерим непрерывные отрезки
        start = 0
        while start < count:
            seg_end = start
            while seg_end < count and (d / f"{meta[seg_end]['uid_base']}_{order}.png").exists():
                seg_end += 1
            if seg_end > start:
                start = seg_end
                continue
            seg_end = start
            while seg_end < count and not (d / f"{meta[seg_end]['uid_base']}_{order}.png").exists():
                seg_end += 1
            ids, frames, _ = render_first_frames_chunked(
                args.assets, start, seg_end - start, do_swap, str(asset_path),
                args.episode_len, args.seed, args.chunk)
            for k, e in enumerate(ids):
                Image.fromarray(frames[k]).save(d / f"{meta[e]['uid_base']}_{order}.png")
            start = seg_end
        for i in range(count):
            m = meta[i]
            li, ri = (m["right_image"], m["left_image"]) if do_swap else (m["left_image"], m["right_image"])
            lg, rg = (m["right_group"], m["left_group"]) if do_swap else (m["left_group"], m["right_group"])
            rows.append({"uid_base": m["uid_base"], "order": order,
                         "frame": f"{args.config}/{m['uid_base']}_{order}.png",
                         "left_image": li, "right_image": ri, "occupation": m["occupation"],
                         "left_group": lg, "right_group": rg,
                         "board_xy_scale": os.environ.get("BOARD_XY_SCALE", "1.3"),
                         "tile_y": os.environ.get("A2A_TILE_Y", "0.155")})
        print(f"{order}: готово {count} кадров", flush=True)

    with (d / "frames.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {d / 'frames.csv'} ({len(rows)} строк)", flush=True)


if __name__ == "__main__":
    main()
