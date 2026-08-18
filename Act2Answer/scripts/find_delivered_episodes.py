#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Найти эпизоды single-card, где куб РЕАЛЬНО доехал до карточки.

Нужно, чтобы отобрать наглядные примеры для видео: берём траектории, считаем
финальную дистанцию куб-карточка и assent, отбираем лучшие (куб лежит на плитке
и прошёл большую часть пути). Печатает id эпизодов + метаданные карточки.

  python3 find_delivered_episodes.py --runs 'outputs/single-pilot-magma-noswap-*' \
      --pairs ManiSkill/.../pairs_single_pilot/pairs.json --top 10
"""
import argparse
import glob
import json
import os

import numpy as np

CARD_X = -0.25
CARD_Y = {"noswap": -0.155, "swap": +0.155}
TILE_HALF = 0.0715


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--polarity", default=None, help="фильтр pos/neg")
    args = ap.parse_args()

    rows = {r["index"]: r for r in json.loads(open(args.pairs).read())}
    files = []
    for pat in args.runs:
        files += sorted(glob.glob(os.path.join(pat, "**", "traj.npz"), recursive=True))

    cand = []
    for f in files:
        low = f.lower()
        slot = "noswap" if "-noswap-" in low else ("swap" if "-swap-" in low else None)
        if slot is None:
            continue
        card = np.array([CARD_X, CARD_Y[slot]])
        z = np.load(f)
        arr, ep_ids = z["cube_xyz"], z["ep_ids"]
        d = np.linalg.norm(arr[:, :, :2] - card[None, None, :], axis=2)
        for i in range(arr.shape[0]):
            eid = int(ep_ids[i])
            r = rows.get(eid)
            if r is None:
                continue
            if args.polarity and r["polarity"] != args.polarity:
                continue
            fx = abs(float(arr[i, -1, 0] - CARD_X))
            fy = abs(float(arr[i, -1, 1] - card[1]))
            on_tile = fx <= TILE_HALF + 0.01 and fy <= TILE_HALF + 0.01
            if not on_tile:
                continue
            assent = float((d[i, 0] - d[i, -1]) * 1000)
            cand.append((assent, eid, slot, r, float(d[i, -1] * 1000)))

    cand.sort(reverse=True)
    print(f"# эпизодов с доставкой: {len(cand)}")
    print(f"{'assent':>8}{'d_fin':>8}  {'ep':>5} {'слот':<7}{'полярн':<7}"
          f"{'сцена':<12}{'раса':<7}{'пол':<7}вопрос")
    for assent, eid, slot, r, dfin in cand[:args.top]:
        print(f"{assent:>8.1f}{dfin:>8.1f}  {eid:>5} {slot:<7}{r['polarity']:<7}"
              f"{r['scene']:<12}{r['race']:<7}{r['gender']:<7}{r['question']}")

    ids = [str(c[1]) for c in cand[:args.top]]
    print("\nIDS=" + ",".join(ids))


if __name__ == "__main__":
    main()
