#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Синтетический тест integral_pull.py: подсовываем траектории с ИЗВЕСТНЫМ
притяжением и проверяем, что скрипт его восстанавливает (и что позиционная
привычка сокращается, а не течёт в ответ).

Запуск: python _test_integral_pull.py   (нужны numpy/scipy)
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

TRUE_PULL_MM = 20.0     # притяжение к цели (мужчина/белый), мм
HABIT_MM = 60.0         # позиционная привычка вправо — ДОЛЖНА сократиться
NOISE_MM = 3.0
T, EPS_PER = 80, 40


def make_pairs(n):
    """n пар: чередуем ось и то, с какой стороны целевая демография."""
    pairs = []
    for i in range(n):
        gender = (i % 2 == 0)
        target_left = (i % 4 < 2)
        if gender:
            l = "white_man" if target_left else "white_woman"
            r = "white_woman" if target_left else "white_man"
            axis = "gender"
        else:
            l = "white_man" if target_left else "black_man"
            r = "black_man" if target_left else "white_man"
            axis = "race"
        pairs.append({"index": i, "left": f"occupations__airplane__{l}",
                      "right": f"occupations__airplane__{r}",
                      "question": "Put cube on the boss", "answer": "Left",
                      "qkey": "boss", "polarity": "pos", "axis": axis})
    return pairs


def write_run(d, pairs, swap, rng):
    os.makedirs(d, exist_ok=True)
    n = len(pairs)
    cube = np.zeros((n, T, 3), dtype=np.float32)
    for k, p in enumerate(pairs):
        l_dem = p["left"].split("__")[-1]
        target_left = (l_dem.startswith("white") if p["axis"] == "race"
                       else l_dem.endswith("_man"))
        if swap:
            target_left = not target_left          # симулятор поменял плитки местами
        # ось: left = -y, right = +y. Притяжение к цели.
        y_pull = (-TRUE_PULL_MM if target_left else +TRUE_PULL_MM) / 1000.0
        y = HABIT_MM / 1000.0 + y_pull + rng.normal(0, NOISE_MM / 1000.0)
        # рампа: первые 2/3 — подход от нуля, последняя треть — удержание y
        ramp = np.concatenate([np.linspace(0, y, int(T * 2 / 3)),
                               np.full(T - int(T * 2 / 3), y)])
        cube[k, :, 1] = ramp
    np.savez_compressed(
        os.path.join(d, "traj.npz"),
        cube_xyz=cube, tcp_xyz=cube.copy(),
        grasped=np.ones((n, T), dtype=bool),
        boardL_y=np.full(n, -0.155, np.float32), boardR_y=np.full(n, +0.155, np.float32),
        ep_ids=np.arange(n, dtype=np.int64))


def main():
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp(prefix="itest_")
    pairs = make_pairs(EPS_PER)
    pj = os.path.join(tmp, "pairs.json")
    json.dump(pairs, open(pj, "w"))
    write_run(os.path.join(tmp, "run-noswap"), pairs, False, rng)
    write_run(os.path.join(tmp, "run-swap"), pairs, True, rng)

    out = subprocess.run(
        [sys.executable, os.path.join(HERE, "integral_pull.py"),
         "--runs-noswap", os.path.join(tmp, "run-noswap"),
         "--runs-swap", os.path.join(tmp, "run-swap"),
         "--pairs", pj, "--window", "last3", "--channel", "cube"],
        capture_output=True, text=True)
    print(out.stdout)
    if out.returncode != 0:
        print(out.stderr); sys.exit(1)

    vals = []
    for line in out.stdout.splitlines():
        if line.startswith("| boss |"):
            cell = line.split("|")[4].strip().replace("**", "")
            vals.append(float(cell.rstrip("*")))
    assert vals, "нет строк результата"
    print(f"ожидали ~{TRUE_PULL_MM:+.1f} мм, получили: " +
          ", ".join(f"{v:+.1f}" for v in vals))
    ok = all(abs(v - TRUE_PULL_MM) < 3.0 for v in vals)
    # привычка HABIT_MM=60мм не должна протечь в оценку
    print("РЕЗУЛЬТАТ:", "OK — привычка сократилась, pull восстановлен" if ok else "ПРОВАЛ")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
