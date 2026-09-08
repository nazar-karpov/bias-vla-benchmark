#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Главный рычаг Magma: base_img_size процессора -> число визуальных токенов -> длина префилла.

512 (дефолт) даёт сетку ConvNeXt 16x16 = 256 визуальных токенов. 384 -> 12x12 = 144,
256 -> 8x8 = 64. Префилл линейен по длине, вижн-башня квадратична по стороне.

ВНИМАНИЕ: меняет то, что видит модель. Скрипт меряет ТОЛЬКО скорость; пригодность
для замеров надо проверять отдельным прогоном с сравнением метрик.
"""
import os
import statistics
import sys
import time
from pathlib import Path

A = Path(os.environ.get("REPO_ROOT", "/home/moskalenko/ws/bias-vla-benchmark-main/Act2Answer"))
sys.path.insert(0, str(A / "SimplerEnv"))
sys.path.insert(0, str(A / "ManiSkill"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from simpler_env import eval as E  # noqa: E402
from simpler_env.policies.magma.magma_model import MagmaInference  # noqa: E402

B = int(os.environ.get("B", "24"))
sys.argv = ["p", "--vla", "magma", "--assets", "pairs_choice_vla_confirm",
            "--count", "6", "--buffer-minibatch", "1"]
args = E.build_runner_args(E.parse_args(), E.VLA_CONFIGS["magma"])
pol = MagmaInference(0, args, "microsoft/Magma-8B")
ip = pol.processor.image_processor

dec = pol.vla.get_decoder()
odec, seq = dec.forward, []


def dec_fwd(*a, **kw):
    ie, ids = kw.get("inputs_embeds"), kw.get("input_ids")
    seq.append(ie.shape[1] if ie is not None else (ids.shape[1] if ids is not None else -1))
    return odec(*a, **kw)


dec.forward = dec_fwd

rng = np.random.default_rng(0)
imgs = torch.from_numpy(rng.integers(0, 255, (B, 480, 640, 3), dtype=np.uint8))
obs = {"image": imgs, "task_description": ["Put cube on the boss"] * B}

print(f"батч {B} | дефолт base_img_size={ip.base_img_size}\n")
print(f"{'base_img':>9} {'префилл':>9} {'время шага':>12} {'на среду':>10} {'ускорение':>10} {'действие[0]':>14}")
print("-" * 72)
base_t, base_a = None, None
for px in (512, 384, 256, 128):
    try:
        ip.base_img_size = px
        a0 = pol.get_action(obs)  # прогрев + образец действия
        seq.clear()
        ts = []
        for _ in range(3):
            torch.cuda.synchronize()
            t = time.perf_counter()
            pol.get_action(obs)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        med, plen = statistics.median(ts), max(seq[:8])
        if base_t is None:
            base_t = med
        v = a0[0][:3] if hasattr(a0, "__getitem__") else None
        vs = ", ".join(f"{float(x):+.3f}" for x in v) if v is not None else "?"
        print(f"{px:>9} {plen:>9} {1000*med:>9.0f} мс {1000*med/B:>7.1f} мс "
              f"{base_t/med:>9.2f}x  {vs}")
    except Exception as e:
        print(f"{px:>9}  ОШИБКА: {str(e)[:60]}")
print("-" * 72)
print("IMGSIZE_DONE")
