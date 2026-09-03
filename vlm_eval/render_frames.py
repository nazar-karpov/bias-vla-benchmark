#!/usr/bin/env python3
"""Рендер первых кадров симуляции для кардсета — обе раскладки, порциями.

Логика взята из Act2Answer/scripts/magma_vlm_qa.py (render_first_frames), но без
импорта transformers: окружение с симулятором и окружение с VLM у нас разные.

Кадры пишутся PNG + manifest.json, дальше VLM работает только с картинками.

Запуск (окружение с sapien/mani_skill, из папки SimplerEnv):
  MS_ASSET_DIR=/workspace/moskalenko/maniskill_assets \
  CUDA_VISIBLE_DEVICES=0 python render_frames.py --assets weapon200 \
      --repo /workspace/moskalenko/roma/Act2Answer --out-dir .../frames
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def render_chunk(Args, SimplerWrapper, assets, asset_path, ids, total,
                 do_swap, episode_len, seed):
    a = Args()
    a.env_id = "Act2AnswerV4-v1"
    a.seed = seed
    a.name = f"render-{assets}-{'swap' if do_swap else 'noswap'}"
    a.obj_set = "test"
    a.episode_len = episode_len
    a.assets = assets
    a.do_swap = bool(do_swap)
    a.asset_path = str(asset_path)
    a.ids = list(ids)
    a.total_envs = total
    a.shard_start = ids[0]
    a.shard_end = ids[-1] + 1
    a.num_envs = len(ids)
    a.init_grasp_steps = 10
    a.hold_cube_steps = 15
    env = SimplerWrapper(a)
    obs, instruction, _ = env.reset(a.obj_set)
    frames = [obs[i].cpu().numpy().astype(np.uint8) for i in range(len(ids))]
    try:
        env.env.close()
    except Exception:
        pass
    del env
    import torch
    torch.cuda.empty_cache()
    return frames, list(instruction)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path, help="папка Act2Answer")
    ap.add_argument("--assets", required=True)
    ap.add_argument("--asset-path", type=Path, default=None)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--chunk", type=int, default=50,
                    help="SAPIEN не тянет 200 GPU-камер сразу; 50 проверено")
    ap.add_argument("--episode-len", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("pair", "single"), default=None,
                    help="по умолчанию берётся из pairs_meta.json кардсета")
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo / "SimplerEnv"))
    sys.path.insert(0, str(args.repo / "ManiSkill"))
    from simpler_env.env.simpler_wrapper_v4 import SimplerWrapper
    from simpler_env.run import Args

    asset_path = args.asset_path or (args.repo / "ManiSkill" / "mani_skill" / "assets" / "carrot")
    pairs = json.loads((asset_path / args.assets / "pairs.json").read_text())
    total = len(pairs)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Режим берём из кардсета, если не задан явно.
    mode = args.mode
    if mode is None:
        meta_path = asset_path / args.assets / "pairs_meta.json"
        mode = (json.loads(meta_path.read_text()).get("mode", "pair")
                if meta_path.exists() else "pair")
    if mode == "single":
        # сцена ставит одну плитку по центру, вторую не выбирает; раскладка одна,
        # менять местами нечего
        os.environ["A2A_SINGLE_TILE"] = "1"
        layouts = [(False, "center")]
    else:
        os.environ.pop("A2A_SINGLE_TILE", None)
        layouts = [(False, "noswap"), (True, "swap")]
    print(f"режим: {mode}, раскладки: {[t for _, t in layouts]}", flush=True)

    manifest = {"assets": args.assets, "mode": mode, "n_pairs": total, "frames": []}
    chunk = args.chunk
    for do_swap, tag in layouts:
        (args.out_dir / tag).mkdir(exist_ok=True)
        s = 0
        while s < total:
            # узел общий: сколько камер влезет, зависит от соседей. Не влезло —
            # уполовиниваем чанк и пробуем снова, вплоть до одного эпизода.
            while True:
                ids = list(range(s, min(s + chunk, total)))
                try:
                    frames, instr = render_chunk(Args, SimplerWrapper, args.assets, asset_path,
                                                 ids, total, do_swap, args.episode_len, args.seed)
                    break
                except RuntimeError as e:
                    if chunk <= 1:
                        raise
                    chunk = max(1, chunk // 2)
                    print(f"  !! {type(e).__name__}: уменьшаю чанк до {chunk}", flush=True)
                    import torch
                    torch.cuda.empty_cache()
            for i, idx in enumerate(ids):
                p = args.out_dir / tag / f"ep{idx:04d}.png"
                Image.fromarray(frames[i]).save(p)
                manifest["frames"].append({"index": idx, "layout": tag,
                                           "path": str(p), "instruction": instr[i]})
            print(f"  {tag}: {ids[0]}..{ids[-1]} -> {len(frames)} кадров", flush=True)
            s += len(ids)

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"ГОТОВО: {len(manifest['frames'])} кадров -> {args.out_dir}")


if __name__ == "__main__":
    main()
