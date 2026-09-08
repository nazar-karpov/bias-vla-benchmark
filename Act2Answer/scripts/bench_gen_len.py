#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сколько токенов Magma генерирует на один шаг и сколько это стоит.

Оборачивает vla.generate: считает длину сгенерированной части и время. Нужен, чтобы понять,
что именно ест время в модели — префилл картинки или авторегрессивный хвост.
"""
import os
import statistics
import sys
import time
from pathlib import Path

A = Path(os.environ.get("REPO_ROOT", "/home/moskalenko/ws/bias-vla-benchmark-main/Act2Answer"))
sys.path.insert(0, str(A / "SimplerEnv"))
sys.path.insert(0, str(A / "ManiSkill"))
os.environ.setdefault("A2A_SAVE_VIDEO", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from simpler_env import eval as E  # noqa: E402
from simpler_env.run import Runner  # noqa: E402

N = int(os.environ.get("N", "6"))
L = int(os.environ.get("L", "5"))
sys.argv = ["bench", "--vla", "magma", "--assets", "pairs_choice_vla_confirm",
            "--count", str(N), "--buffer-inferbatch", str(N), "--episode-len", str(L),
            "--buffer-minibatch", "1",  # иначе assert num_envs*L % minibatch == 0
            "--name", f"genlen-n{N}"]
ns = E.parse_args()
args = E.build_runner_args(ns, E.VLA_CONFIGS["magma"])
runner = Runner(args)

pol = runner.policy
print("sample (do_sample):", getattr(pol, "sample", "?"))
gen_lens, gen_times, in_lens = [], [], []
orig = pol.vla.generate


def timed_generate(**kw):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = orig(**kw)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    in_len = kw["input_ids"].shape[1]
    gen = out.shape[1] - in_len
    in_lens.append(in_len)
    gen_lens.append(gen)
    gen_times.append(dt)
    return out


pol.vla.generate = timed_generate
runner.render(epoch=0, obj_set=args.obj_set)

print("\n" + "=" * 64)
print(f"вызовов generate: {len(gen_lens)} | батч {N}")
print(f"длина промпта (input_ids) : медиана {statistics.median(in_lens):.0f} токенов")
print(f"сгенерировано за вызов    : медиана {statistics.median(gen_lens):.0f}, "
      f"мин {min(gen_lens)}, макс {max(gen_lens)}")
print(f"время generate            : медиана {1000*statistics.median(gen_times):.0f} мс")
per_tok = statistics.median(gen_times) / max(statistics.median(gen_lens), 1)
print(f"≈ на токен декода         : {1000*per_tok:.1f} мс (весь батч)")
print("GENLEN_RESULT gen_median=%.0f in_median=%.0f t_median_ms=%.0f" %
      (statistics.median(gen_lens), statistics.median(in_lens), 1000 * statistics.median(gen_times)))
