#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke без VLA: крутим env случайными действиями и сохраняем traj.npz тем же
способом, что run.py. Цель — убедиться, что env.traj_log реально накапливается
и что integral_pull.py читает получившийся файл.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

A = Path("/home/moskalenko/ws/bias-vla-benchmark-main/Act2Answer")
sys.path.insert(0, str(A / "scripts"))
# magma_vlm_qa тянет transformers/safetensors (в env их нет и они тут не нужны) —
# берём только build_env_args, подсунув заглушки тяжёлых импортов.
import types  # noqa: E402
for _m, _attrs in (("safetensors", ["safe_open"]),
                   ("huggingface_hub", ["snapshot_download"]),
                   ("transformers", ["AutoModelForCausalLM", "AutoProcessor"])):
    if _m not in sys.modules:
        _mod = types.ModuleType(_m)
        for _a in _attrs:
            setattr(_mod, _a, None)
        sys.modules[_m] = _mod
from magma_vlm_qa import build_env_args  # noqa: E402

ASSETS = "pairs_choice_vla_confirm"
COUNT = int(os.environ.get("SMOKE_N", "4"))
STEPS = int(os.environ.get("SMOKE_STEPS", "12"))
OUT = Path(os.environ.get("SMOKE_OUT", "/home/moskalenko/ws/_smoke_traj/run-noswap"))


def main():
    from simpler_env.env.simpler_wrapper_v4 import SimplerWrapper

    asset_path = str(A / "ManiSkill" / "mani_skill" / "assets" / "carrot")
    args, ids, _ = build_env_args(ASSETS, 0, COUNT, False, asset_path, 80, 0)
    env = SimplerWrapper(args)
    env.reset(args.obj_set)

    base = env.env.unwrapped
    print("traj_log присутствует:", hasattr(base, "traj_log"), flush=True)

    for t in range(STEPS):
        act = torch.zeros((len(ids), 7), dtype=torch.float32)
        act[:, 1] = 0.05 * (1 if t % 2 == 0 else -1)  # шевелим по оси плиток
        env.step(act)

    tl = getattr(base, "traj_log", None)
    assert tl and tl["cube_xyz"], "traj_log пуст — лог не накапливается!"
    print(f"накоплено шагов: {len(tl['cube_xyz'])}, envs: {tl['cube_xyz'][0].shape[0]}")

    OUT.mkdir(parents=True, exist_ok=True)
    st = base.episode_stats
    np.savez_compressed(
        OUT / "traj.npz",
        cube_xyz=torch.stack(tl["cube_xyz"]).permute(1, 0, 2).numpy(),
        tcp_xyz=torch.stack(tl["tcp_xyz"]).permute(1, 0, 2).numpy(),
        grasped=torch.stack(tl["grasped"]).permute(1, 0).numpy(),
        boardL_y=st["boardL_y"].detach().cpu().numpy(),
        boardR_y=st["boardR_y"].detach().cpu().numpy(),
        ep_ids=np.array(ids, dtype=np.int64),
    )
    z = np.load(OUT / "traj.npz")
    print("СОХРАНЁН traj.npz:", {k: z[k].shape for k in z.files})
    print("cube y перв/посл шаг:", z["cube_xyz"][0, 0, 1], z["cube_xyz"][0, -1, 1])
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
