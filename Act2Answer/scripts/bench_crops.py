#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сколько стоит anyres-нарезка Magma: num_crops = 4 (дефолт) против 2 и 1.

Меньше кропов -> меньше визуальных токенов -> короче префилл -> меньше FLOPs.
ВНИМАНИЕ: меняет то, что видит модель. Скрипт меряет только скорость и длину префилла;
эквивалентность действий надо проверять отдельно.
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

print(f"батч {B} | дефолт num_crops={getattr(ip, 'num_crops', '?')}\n")
print(f"{'num_crops':>10} {'префилл, ток':>13} {'время шага':>12} {'на среду':>10} {'ускорение':>10}")
print("-" * 60)
base = None
for nc in (4, 2, 1):
    try:
        ip.num_crops = nc
        pol.get_action(obs)  # прогрев
        seq.clear()
        ts = []
        for _ in range(3):
            torch.cuda.synchronize()
            t = time.perf_counter()
            pol.get_action(obs)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        med = statistics.median(ts)
        plen = max(seq[:8])
        base = base or med
        print(f"{nc:>10} {plen:>13} {1000*med:>9.0f} мс {1000*med/B:>7.1f} мс {base/med:>9.2f}x")
    except Exception as e:
        print(f"{nc:>10}  ОШИБКА: {str(e)[:60]}")
print("-" * 60)
print("CROPS_DONE")
