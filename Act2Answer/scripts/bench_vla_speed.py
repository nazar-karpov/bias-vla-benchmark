#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Замер скорости VLA-прогона: сколько времени ест модель, сколько симулятор.

Эпизоды идут параллельно (num_envs = --count), на каждом из episode_len шагов делается
ceil(num_envs / buffer_inferbatch) вызовов модели и один шаг симулятора на все среды.
Скрипт оборачивает policy.get_action и env.step таймерами (с cuda.synchronize) и печатает
разбивку + пропускную способность в эпизодах/час.

  python bench_vla_speed.py --vla magma --assets pairs_choice_vla_confirm --count 6
  python bench_vla_speed.py --vla magma --count 24 --inferbatch 24 --episode-len 20
"""
import argparse
import os
import statistics
import sys
import time
from pathlib import Path

A = Path(os.environ.get("REPO_ROOT", "/home/moskalenko/ws/bias-vla-benchmark-main/Act2Answer"))
sys.path.insert(0, str(A / "SimplerEnv"))
sys.path.insert(0, str(A / "ManiSkill"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vla", default="magma")
    ap.add_argument("--assets", default="pairs_choice_vla_confirm")
    ap.add_argument("--count", type=int, default=6, help="сколько эпизодов параллельно (num_envs)")
    ap.add_argument("--inferbatch", type=int, default=0, help="0 = как count")
    ap.add_argument("--episode-len", type=int, default=80)
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--tag", default="")
    args_cli = ap.parse_args()

    os.environ.setdefault("A2A_SAVE_VIDEO", "0")   # видео не пишем: это отдельная статья расходов
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from simpler_env import eval as E
    from simpler_env.run import Runner

    ib = args_cli.inferbatch or args_cli.count
    argv = ["--vla", args_cli.vla, "--assets", args_cli.assets,
            "--count", str(args_cli.count), "--start-id", str(args_cli.start_id),
            "--buffer-inferbatch", str(ib), "--episode-len", str(args_cli.episode_len),
            "--name", f"bench-{args_cli.vla}-n{args_cli.count}-b{ib}{args_cli.tag}"]
    old_argv, sys.argv = sys.argv, ["bench"] + argv
    ns = E.parse_args()
    sys.argv = old_argv
    cfg = E.VLA_CONFIGS[args_cli.vla]
    args = E.build_runner_args(ns, cfg)

    t0 = time.monotonic()
    runner = Runner(args)
    t_load = time.monotonic() - t0

    # A2A_BASE_IMG: сторона, к которой процессор приводит кадр (Magma: 512 по умолчанию ->
    # 256 визуальных токенов). Меньше -> короче префилл, но модель видит грубее.
    px = int(os.environ.get("A2A_BASE_IMG", "0"))
    if px:
        ip = getattr(getattr(runner.policy, "processor", None), "image_processor", None)
        if ip is not None and hasattr(ip, "base_img_size"):
            ip.base_img_size = px
            print(f"BASE_IMG_SIZE установлен в {px}", flush=True)
    # A2A_LMHEAD_LAST=1: считать lm_head только по последней позиции. modeling_magma.py:795
    # гоняет голову по всем позициям префилла, хотя generate() берёт только logits[:, -1].
    # Математически эквивалентно для генерации; экономит время и ~2 ГБ на логиты.
    if os.environ.get("A2A_LMHEAD_LAST") == "1":
        lm = runner.policy.vla.language_model
        head = lm.lm_head
        import torch.nn as nn

        class LastPosHead(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                if x.dim() == 3 and x.shape[1] > 1:
                    x = x[:, -1:, :]
                return self.inner(x)

        lm.lm_head = LastPosHead(head)
        print("LM_HEAD только по последней позиции", flush=True)

    # Magma по умолчанию сэмплирует (do_sample=True, T=0.7) даже при deterministic=True,
    # поэтому для сравнения конфигураций нужен жадный режим.
    if os.environ.get("A2A_GREEDY") == "1" and hasattr(runner.policy, "sample"):
        runner.policy.sample = False
        print("GREEDY включён (do_sample=False)", flush=True)

    tm, te = [], []
    orig_action = runner.policy.get_action
    orig_step = runner.env.step

    # A2A_ACTION_REPEAT=k: спрашивать модель раз в k шагов, между опросами повторять действие.
    # Режет число вызовов модели в k раз. МЕНЯЕТ управление — требует проверки метрик.
    rep = int(os.environ.get("A2A_ACTION_REPEAT", "1"))
    if rep > 1:
        state = {"i": 0, "last": None}

        def repeated(obs, deterministic=False):
            if state["i"] % rep == 0 or state["last"] is None:
                state["last"] = orig_action(obs, deterministic)
            state["i"] += 1
            return state["last"]

        orig_action_raw, orig_action = orig_action, repeated
        print(f"ACTION_REPEAT={rep} (модель раз в {rep} шага)", flush=True)

    def timed_action(obs, deterministic=False):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = orig_action(obs, deterministic)
        torch.cuda.synchronize()
        tm.append(time.perf_counter() - t)
        return r

    def timed_step(action):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = orig_step(action)
        torch.cuda.synchronize()
        te.append(time.perf_counter() - t)
        return r

    runner.policy.get_action = timed_action
    runner.env.step = timed_step

    t1 = time.monotonic()
    stats = runner.render(epoch=0, obj_set=args.obj_set)
    t_roll = time.monotonic() - t1
    keys = [k for k in ("is_answered", "is_answered_soft", "chosen_side", "success",
                        "grasped", "consecutive_grasp") if k in stats]
    if keys:
        print("STATS " + " ".join(f"{k}={float(stats[k]):.4f}" for k in keys), flush=True)

    n_env, L = args.num_envs, args.episode_len
    calls_per_step = (n_env + ib - 1) // ib
    model_s, env_s = sum(tm), sum(te)
    other = t_roll - model_s - env_s
    mem = torch.cuda.max_memory_allocated() / 2**30

    def ms(x):
        return f"{1000 * x:.0f} мс"

    print("\n" + "=" * 72)
    print(f"BENCH {args_cli.vla} | сред {n_env} | inferbatch {ib} | шагов {L} | "
          f"вызовов модели/шаг {calls_per_step}")
    print("=" * 72)
    print(f"загрузка модели+сцены : {t_load:6.1f} с (разово)")
    print(f"прогон эпизода        : {t_roll:6.1f} с на {n_env} эпизодов")
    print(f"  модель              : {model_s:6.1f} с ({100*model_s/t_roll:4.1f}%) "
          f"| {len(tm)} вызовов, медиана {ms(statistics.median(tm))}")
    print(f"  симулятор (step)    : {env_s:6.1f} с ({100*env_s/t_roll:4.1f}%) "
          f"| {len(te)} шагов, медиана {ms(statistics.median(te))}")
    print(f"  прочее (reset/логи) : {other:6.1f} с ({100*other/t_roll:4.1f}%)")
    print(f"пик VRAM              : {mem:.1f} ГБ")
    print("-" * 72)
    per_ep = t_roll / n_env
    print(f"на эпизод             : {per_ep:.2f} с | {3600/per_ep:.0f} эпизодов/час на карту")
    print(f"BENCH_RESULT vla={args_cli.vla} n={n_env} ib={ib} L={L} "
          f"roll={t_roll:.1f} model={model_s:.1f} env={env_s:.1f} per_ep={per_ep:.2f} vram={mem:.1f}")


if __name__ == "__main__":
    main()
