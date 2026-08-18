#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Видео эпизода из сохранённой траектории — БЕЗ повторного прогона модели.

Боевые прогоны идут с A2A_SAVE_VIDEO=0, но траектории куба и схвата лежат в
traj.npz. Здесь мы строим env один раз (сцена + карточка), затем на каждом шаге
ставим куб в записанную позу и снимаем кадр. Модель не загружается, GPU нужен
только под рендер (~2 ГБ), поэтому можно писать демо, пока на картах идёт
основной прогон.

  BOARD_XY_SCALE=1.0 python3 replay_traj_video.py \
      --runs 'outputs/single-pilot-magma-noswap-*' --ep 261 \
      --pairs ManiSkill/.../pairs_single_pilot/pairs.json --out-dir ../outputs/demo_videos
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("A2A_SINGLE_TILE", "1")

import imageio.v2 as imageio  # noqa: E402


def find_traj(patterns, ep):
    for pat in patterns:
        for f in sorted(glob.glob(os.path.join(pat, "**", "traj.npz"), recursive=True)):
            z = np.load(f)
            if ep in set(int(x) for x in z["ep_ids"]):
                slot = "noswap" if "-noswap-" in f.lower() else "swap"
                idx = int(np.argwhere(z["ep_ids"] == ep).ravel()[0])
                return f, z, idx, slot
    raise SystemExit(f"эпизод {ep} не найден в {patterns}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ep", type=int, required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--assets", default="pairs_single_pilot")
    ap.add_argument("--asset-path", default=None)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    if args.asset_path is None:
        repo = Path(os.environ["REPO_ROOT"])
        args.asset_path = str(repo / "ManiSkill" / "mani_skill" / "assets" / "carrot")

    rows = {r["index"]: r for r in json.loads(open(args.pairs).read())}
    meta = rows[args.ep]
    f, z, idx, slot = find_traj(args.runs, args.ep)
    cube = z["cube_xyz"][idx]          # [T,3]
    print(f"эпизод {args.ep}: {slot}, {len(cube)} шагов, файл {f}")
    print(f"  карточка: {meta['card']}  вопрос: {meta['question']}")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from magma_vlm_qa import build_env_args  # noqa: E402
    from simpler_env.env.simpler_wrapper_v4 import SimplerWrapper  # noqa: E402

    do_swap = (slot == "swap")
    env_args, ids, _ = build_env_args(args.assets, args.ep, 1, do_swap,
                                      args.asset_path, len(cube), 0)
    env = SimplerWrapper(env_args)
    env.reset(env_args.obj_set)

    base = env.env.unwrapped
    cube_actor = base.cubes[base._white_key]
    import torch
    from mani_skill.utils.structs.pose import Pose

    frames = []
    q = cube_actor.pose.q
    for t in range(len(cube)):
        p = torch.tensor(cube[t].reshape(1, 3), device=base.device, dtype=torch.float32)
        cube_actor.set_pose(Pose.create_from_pq(p=p, q=q))
        if hasattr(base.scene, "_gpu_apply_all"):
            base.scene._gpu_apply_all()
            base.scene.px.gpu_update_articulation_kinematics()
            base.scene._gpu_fetch_all()
        obs = base.get_obs()
        img = obs["sensor_data"][base.rgb_camera_name]["rgb"][0].cpu().numpy()
        frames.append(img.astype(np.uint8))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"ep{args.ep}_{meta['polarity']}_{meta['scene']}_{meta['race']}_{meta['gender']}"
    out = args.out_dir / f"{tag}.mp4"
    imageio.mimsave(out, frames, fps=args.fps)
    print(f"-> {out}  ({len(frames)} кадров)")


if __name__ == "__main__":
    main()
