#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Чистая стоимость модели без симулятора: префилл vs декодирование, размер картинки, батч.

Симулятор не запускается — подаём синтетические кадры того же формата. Отвечает на вопросы:
сколько стоит префилл (кодирование картинки) против 8 шагов декодирования, сколько визуальных
токенов, и что даст уменьшение картинки.
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
from PIL import Image  # noqa: E402
from simpler_env.policies.magma.magma_model import MagmaInference  # noqa: E402

from simpler_env import eval as E  # noqa: E402

sys.argv = ["bench", "--vla", "magma", "--assets", "pairs_choice_vla_confirm",
            "--count", "6", "--buffer-minibatch", "1"]
_args = E.build_runner_args(E.parse_args(), E.VLA_CONFIGS["magma"])
pol = MagmaInference(0, _args, "microsoft/Magma-8B")
if hasattr(pol, "prep_rollout"):
    pol.prep_rollout()

# длина последовательности после вклейки картинки — сколько токенов реально жуёт LLM
seq_seen = {}
lm = pol.vla.get_decoder() if hasattr(pol.vla, "get_decoder") else None
if lm is not None:
    orig_fwd = lm.forward

    def probe_fwd(*a, **kw):
        ie = kw.get("inputs_embeds")
        if ie is not None and "len" not in seq_seen:
            seq_seen["len"] = int(ie.shape[1])
        return orig_fwd(*a, **kw)

    lm.forward = probe_fwd


def run(batch, img_px, n_tok, reps=3):
    rng = np.random.default_rng(0)
    imgs = torch.from_numpy(rng.integers(0, 255, (batch, 480, 640, 3), dtype=np.uint8))
    obs = {"image": imgs, "task_description": ["Put cube on the boss"] * batch}
    # подмена целевого размера ресайза
    orig_resize = Image.Image.resize

    def patched(self, size, *a, **kw):
        return orig_resize(self, (img_px, img_px), *a, **kw)

    orig_gen = pol.vla.generate

    def capped(**kw):
        kw["max_new_tokens"] = n_tok
        return orig_gen(**kw)

    Image.Image.resize = patched
    pol.vla.generate = capped
    try:
        pol.get_action(obs)  # прогрев
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            pol.get_action(obs)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
    except Exception as e:
        return None, str(e)[:80]
    finally:
        Image.Image.resize = orig_resize
        pol.vla.generate = orig_gen
    return statistics.median(ts), None


print("\n" + "=" * 78)
print(f"{'картинка':>10} {'батч':>5} {'префилл(1ток)':>14} {'полный(8ток)':>13} "
      f"{'декод 7ток':>11} {'на эпизод-шаг':>14}")
print("=" * 78)
for px in (256, 192, 128):
    for b in (6, 24, 48):
        t1, e1 = run(b, px, 1)
        t8, e8 = run(b, px, 8)
        if e1 or e8:
            print(f"{px:>7}px {b:>5}  ОШИБКА {e1 or e8}")
            continue
        print(f"{px:>7}px {b:>5} {1000*t1:>11.0f} мс {1000*t8:>10.0f} мс "
              f"{1000*(t8-t1):>8.0f} мс {1000*t8/b:>11.1f} мс")
print("=" * 78)
print("токенов в последовательности (с картинкой):", seq_seen.get("len", "?"))
