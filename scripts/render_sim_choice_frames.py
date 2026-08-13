#!/usr/bin/env python3
"""Рендер ПЕРВЫХ КАДРОВ симуляции для кардсета pairs_choice — обе раскладки.

Кадры считаются один раз и сохраняются PNG: QA-скриптам (vlm_sim_choice.py) потом
не нужен ни SimplerEnv, ни GPU-симулятор — только модель. noswap = порядок ab
(cand_a слева), swap = ba (симулятор физически меняет плитки местами).

Рядом пишется manifest.json: index -> scene/axis/cand_a/cand_b, чтобы QA не парсил
имена плиток заново.

Запуск на сервере (env magma_act2answer, из папки SimplerEnv — нужен относительный
overlay-фон ./bridge_real_eval_1.png):
  source ~/bias_benchmark/miniconda3/etc/profile.d/conda.sh
  conda activate ~/bias_benchmark/miniconda3/envs/magma_act2answer
  export REPO_ROOT=~/bias_benchmark/nazar_folder/Act2Answer
  export PYTHONPATH=$REPO_ROOT/SimplerEnv:$REPO_ROOT/ManiSkill
  cd $REPO_ROOT/SimplerEnv
  CUDA_VISIBLE_DEVICES=1 python -u $REPO_ROOT/../scripts/render_sim_choice_frames.py \
    --out-dir $REPO_ROOT/outputs/simframes_choice < /dev/null
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
    ap.add_argument("--assets", default="pairs_choice")
    ap.add_argument("--asset-path", default=None)
    ap.add_argument("--count", type=int, default=0, help="0 = все эпизоды")
    ap.add_argument("--episode-len", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render-chunk", type=int, default=50,
                    help="SAPIEN не тянет 200 GPU-камер сразу; 50-55 проверено")
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
        proto = "__".join(p["left"].split("__")[:2])
        da = p["left"].split("__", 2)[2]
        db = p["right"].split("__", 2)[2]
        axis = "gender" if da.split("_")[0] == db.split("_")[0] else "race"
        manifest.append({"index": p["index"], "scene": proto.replace("__", "/"),
                         "axis": axis, "cand_a": da, "cand_b": db})
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {len(manifest)} эпизодов -> {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
