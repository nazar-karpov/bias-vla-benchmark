#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Профиль одного шага Magma: препроцессинг vs вижн-башня vs LLM-префилл vs декод.

Нужен, чтобы понять, что резать. Печатает также размер картинки, который реально требует
процессор, число визуальных токенов и оценку MFU (доля пиковой fp16-производительности 4090).
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
PEAK_FP16 = 165e12  # RTX 4090, dense fp16 через тензорные ядра

sys.argv = ["p", "--vla", "magma", "--assets", "pairs_choice_vla_confirm",
            "--count", "6", "--buffer-minibatch", "1"]
args = E.build_runner_args(E.parse_args(), E.VLA_CONFIGS["magma"])
pol = MagmaInference(0, args, "microsoft/Magma-8B")

ip = getattr(pol.processor, "image_processor", None)
print("процессор:", type(ip).__name__ if ip else "?",
      "| size:", getattr(ip, "size", None), "| crop:", getattr(ip, "crop_size", None),
      "| anyres:", getattr(ip, "anyres_strategy", None), getattr(ip, "num_crops", None))
print("параметров в модели: %.2f млрд" % (sum(p.numel() for p in pol.vla.parameters()) / 1e9))

T = {}


def mark(name, fn, *a, **kw):
    torch.cuda.synchronize()
    t = time.perf_counter()
    r = fn(*a, **kw)
    torch.cuda.synchronize()
    T.setdefault(name, []).append(time.perf_counter() - t)
    return r


# --- хук на вижн-башню
vt = pol.vla.model.vision_tower if hasattr(pol.vla, "model") else None
if vt is not None:
    ovt = vt.forward

    def vt_fwd(*a, **kw):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = ovt(*a, **kw)
        torch.cuda.synchronize()
        T.setdefault("vision_tower", []).append(time.perf_counter() - t)
        return r
    vt.forward = vt_fwd

# --- хук на LLM: длина последовательности и число вызовов (префилл + декоды)
dec = pol.vla.get_decoder()
odec = dec.forward
seq = {}


def dec_fwd(*a, **kw):
    ie = kw.get("inputs_embeds")
    ids = kw.get("input_ids")
    n = ie.shape[1] if ie is not None else (ids.shape[1] if ids is not None else -1)
    seq.setdefault("lens", []).append(n)
    return odec(*a, **kw)


dec.forward = dec_fwd

rng = np.random.default_rng(0)
imgs = torch.from_numpy(rng.integers(0, 255, (B, 480, 640, 3), dtype=np.uint8))
obs = {"image": imgs, "task_description": ["Put cube on the boss"] * B}

pol.get_action(obs)  # прогрев
seq["lens"].clear()
for k in T:
    T[k].clear()

REPS = 3
for _ in range(REPS):
    mark("get_action_total", pol.get_action, obs)

n_calls = len(seq["lens"]) // REPS
prefill_len = max(seq["lens"][:n_calls])
print(f"\nвызовов LLM за шаг: {n_calls} (1 префилл + {n_calls-1} декодов)")
print(f"длина префилла: {prefill_len} токенов")

tot = statistics.median(T["get_action_total"])
vis = sum(T.get("vision_tower", [0])) / REPS if "vision_tower" in T else 0.0
print("\n" + "=" * 66)
print(f"батч {B}: шаг целиком {1000*tot:.0f} мс  ({1000*tot/B:.1f} мс на среду)")
if vis:
    print(f"  вижн-башня      : {1000*vis:>7.0f} мс ({100*vis/tot:.0f}%)")
print(f"  остальное       : {1000*(tot-vis):>7.0f} мс ({100*(tot-vis)/tot:.0f}%)")

P = sum(p.numel() for p in pol.vla.parameters())
flops_prefill = 2 * P * prefill_len * B
print("-" * 66)
print(f"FLOP префилла (2·P·L·B) : {flops_prefill/1e12:.1f} ТFLOP")
print(f"нижняя граница на 4090  : {1000*flops_prefill/PEAK_FP16:.0f} мс при 100% пике")
print(f"MFU по шагу целиком     : {100*flops_prefill/PEAK_FP16/tot:.0f}%")
print("=" * 66)
