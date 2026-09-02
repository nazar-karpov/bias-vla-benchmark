#!/usr/bin/env python3
"""Рендер ПЕРВЫХ КАДРОВ симуляции для safety-кардсета sohas96x2 (обе раскладки).

То же, что render_sim_choice_frames.py для pairs_choice, но манифест строится по
полям sohas-кардсета (weapon_class / distractor_class), а не по именам PAIRS-плиток.

  cand_a = left  = плитка с оружием (knife|pistol)
  cand_b = right = безобидный предмет (monedero|smartphone|tarjeta|billete)
  axis   = distractor_class,  scene = "<weapon_class>/<distractor_class>"

Блок 0..95 = pos-вопрос, 96..191 = neg; порядок плиток в блоках одинаков, поэтому
кадры рендерим только для первых 96 индексов и переиспользуем для обеих полярностей.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from magma_vlm_qa import render_first_frames_chunked  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="sohas96x2_13")
    ap.add_argument("--asset-path", default=None)
    ap.add_argument("--count", type=int, default=96)
    ap.add_argument("--episode-len", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render-chunk", type=int, default=48)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    if args.asset_path is None:
        repo = Path(os.environ["REPO_ROOT"])
        args.asset_path = str(repo / "ManiSkill" / "mani_skill" / "assets" / "carrot")

    pairs = json.loads((Path(args.asset_path) / args.assets / "pairs.json").read_text())
    count = args.count if args.count > 0 else len(pairs)

    for do_swap in (False, True):
        d = args.out_dir / ("swap" if do_swap else "noswap")
        d.mkdir(parents=True, exist_ok=True)
        ids, frames, _ = render_first_frames_chunked(
            args.assets, 0, count, do_swap, args.asset_path,
            args.episode_len, args.seed, args.render_chunk)
        for k, e in enumerate(ids):
            Image.fromarray(frames[k]).save(d / f"ep{e}.png")
        print(f"saved {len(ids)} frames -> {d}", flush=True)

    manifest = []
    for p in pairs[:count]:
        w, dis = p.get("weapon_class", "?"), p.get("distractor_class", "?")
        manifest.append({"index": p["index"], "scene": f"{w}/{dis}", "axis": dis,
                         "cand_a": p["left"], "cand_b": p["right"]})
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {len(manifest)} эпизодов -> {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
