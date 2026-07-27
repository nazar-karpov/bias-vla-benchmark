#!/usr/bin/env python3
"""Параллельный оркестратор прогонов Magma через joblib: один воркер = одна GPU.

Каждый воркер запускает свои прогоны как отдельные subprocess `simpler_env.eval`
(модель грузится один раз на процесс, держит CUDA-контекст своей карты через
CUDA_VISIBLE_DEVICES). joblib.Parallel(n_jobs=len(JOBS), backend="loky") разводит
воркеры по процессам, так что карты работают параллельно.

Замер памяти: Magma fp16 ~28.7 ГБ из 32 → на одну V100 влезает ТОЛЬКО один процесс.
Поэтому распараллеливаем по РАЗНЫМ картам (по одному процессу на GPU), а не два на одну.
"""
import os
import subprocess
import sys
import time
from joblib import Parallel, delayed

A = os.path.expanduser("~/bias_benchmark/nazar_folder/Act2Answer")
LOGDIR = f"{A}/logs/night"
os.makedirs(LOGDIR, exist_ok=True)

# (gpu, asset, start_id, count, tag)  — по одному воркеру на GPU
JOBS = [
    (0, "safeeditbench", 25, 25, "fp16b"),  # следующие 25 сэмплов (первые 25 уже есть)
    (1, "pairs_bias",     0, 50, "fp16"),   # 50 эпизодов
]


def run_pass(gpu, asset, start_id, count, swap, tag):
    name = f"{tag}-magma-{asset}-s{start_id}-{swap}"
    log = f"{LOGDIR}/{name}.log"
    env = dict(os.environ)
    env.update(
        REPO_ROOT=A,
        PYTHONPATH=f"{A}/SimplerEnv:{A}/ManiSkill",
        TOKENIZERS_PARALLELISM="false",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        CUDA_VISIBLE_DEVICES=str(gpu),
    )
    cmd = [
        sys.executable, "-u", "-m", "simpler_env.eval",
        "--vla", "magma", "--start-id", str(start_id), "--count", str(count),
        "--assets", asset, "--obj-set", "test", "--episode-len", "80",
        "--buffer-inferbatch", "5", "--buffer-minibatch", "-1",
        "--name", name,
    ]
    if swap == "swap":
        cmd.append("--do-swap")
    with open(log, "w") as f:
        rc = subprocess.call(cmd, cwd=f"{A}/SimplerEnv", env=env, stdout=f, stderr=subprocess.STDOUT)
    final = ""
    for line in open(log, errors="ignore"):
        if "FINAL_STATS" in line:
            final = line.strip()
    prog = f"{LOGDIR}/_progress_joblib.log"
    with open(prog, "a") as p:
        p.write(f"[GPU{gpu}] DONE {name} rc={rc} {final} {time.strftime('%H:%M:%S')}\n")
    return name, rc, final


def worker(gpu, asset, start_id, count, tag):
    """Один воркер = одна GPU: гоняет noswap затем swap последовательно на своей карте."""
    out = []
    for swap in ("noswap", "swap"):
        out.append(run_pass(gpu, asset, start_id, count, swap, tag))
    return out


if __name__ == "__main__":
    prog = f"{LOGDIR}/_progress_joblib.log"
    open(prog, "w").write(f"JOBLIB_START {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    results = Parallel(n_jobs=len(JOBS), backend="loky")(
        delayed(worker)(*job) for job in JOBS
    )
    with open(prog, "a") as p:
        p.write(f"JOBLIB_ALL_DONE {time.strftime('%H:%M:%S')}\n")
        for r in results:
            for name, rc, final in r:
                p.write(f"  {name} rc={rc} {final}\n")
    print("done")
