#!/usr/bin/env python3
"""Рендер первых кадров single-card кардсета (проверка одиночного режима).

Ставит A2A_SINGLE_TILE=1 и рендерит первые кадры выбранных эпизодов в обеих
раскладках (noswap = карточка в левом слоте, swap = в правом). Запускать с
BOARD_XY_SCALE=1.0 — масштаб 1.3 уже вшит в model_db кардсета.

  BOARD_XY_SCALE=1.0 python3 render_single_card_frames.py \
      --assets pairs_single_pilot --ids 0 250 --out-dir ../outputs/single_card_preview
"""
import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

os.environ["A2A_SINGLE_TILE"] = "1"  # до импорта/создания env

sys.path.insert(0, str(Path(__file__).resolve().parent))
from magma_vlm_qa import render_first_frames_chunked  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="pairs_single_pilot")
    ap.add_argument("--asset-path", default=None)
    ap.add_argument("--ids", type=int, nargs="+", default=None,
                    help="конкретные индексы эпизодов (по умолчанию первые --count)")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--episode-len", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render-chunk", type=int, default=48)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    if args.asset_path is None:
        repo = Path(os.environ["REPO_ROOT"])
        args.asset_path = str(repo / "ManiSkill" / "mani_skill" / "assets" / "carrot")

    pairs = json.loads((Path(args.asset_path) / args.assets / "pairs.json").read_text())
    by_index = {p["index"]: p for p in pairs}

    manifest = []
    for do_swap in (False, True):
        slot = "swap" if do_swap else "noswap"
        d = args.out_dir / slot
        d.mkdir(parents=True, exist_ok=True)
        if args.ids is not None:
            # рендерим по одному: render_first_frames идёт подряд от start_id
            for eid in args.ids:
                ids, frames, instr = render_first_frames_chunked(
                    args.assets, eid, 1, do_swap, args.asset_path,
                    args.episode_len, args.seed, args.render_chunk)
                Image.fromarray(frames[0]).save(d / f"ep{ids[0]}.png")
                manifest.append({"index": ids[0], "slot": slot,
                                 "instruction": instr[0], **{
                                     k: by_index[ids[0]].get(k) for k in
                                     ("card", "qkey", "polarity", "scene", "race", "gender")}})
        else:
            ids, frames, instr = render_first_frames_chunked(
                args.assets, 0, args.count, do_swap, args.asset_path,
                args.episode_len, args.seed, args.render_chunk)
            for k, e in enumerate(ids):
                Image.fromarray(frames[k]).save(d / f"ep{e}.png")
                manifest.append({"index": e, "slot": slot,
                                 "instruction": instr[k], **{
                                     kk: by_index[e].get(kk) for kk in
                                     ("card", "qkey", "polarity", "scene", "race", "gender")}})
        print(f"saved -> {d}", flush=True)

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"manifest: {len(manifest)} кадров -> {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
