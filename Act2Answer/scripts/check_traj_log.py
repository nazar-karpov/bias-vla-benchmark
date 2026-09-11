#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Валидатор данных прогона: целы ли stats.yaml и traj.npz, есть ли все поля.

Запускать на ПЕРВОМ шарде каждой модели перед тем, как оставлять прогон на ночь,
и на всём прогоне после — метрики можно поправить потом, а недописанные поля
в traj.npz восстановить нельзя.

  python check_traj_log.py --run q10-visbias_q10 --sep s          # весь прогон
  python check_traj_log.py --run q10-visbias_q10 --sep s --last 3 # только 3 свежих шарда
  REPO_ROOT=/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer python check_traj_log.py ...
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import yaml

A = os.environ.get("REPO_ROOT", "/home/moskalenko/ws/bias-vla-benchmark-main/Act2Answer")
OUT = os.path.join(A, "outputs")

# что обязано быть в traj.npz после 11.09.2026; старые прогоны — только первые 6
REQUIRED = ["cube_xyz", "tcp_xyz", "grasped", "ep_ids", "boardL_y", "boardR_y"]
EXTENDED = ["boardL_xy", "boardR_xy", "gripper_q", "qvel", "action"]
STATS_FIELDS = ["chosen_side", "chosen_side_soft", "first_touch_side", "is_answered",
                "cube_fx", "cube_fy", "cube_fz", "tcp_fx", "tcp_fy", "tcp_fz",
                "boardL_y", "boardR_y"]


def check_shard(d, sep):
    m = re.search(rf"-{sep}(\d+)$", d)
    start = int(m.group(1)) if m else None
    base = os.path.join(d, "glob", "vis_0_test")
    rep = dict(dir=os.path.basename(d), start=start, problems=[], missing_ext=[])
    sf, tf = os.path.join(base, "stats.yaml"), os.path.join(base, "traj.npz")
    if not os.path.exists(sf):
        # шард ещё пишется: папка свежая (< 2 ч) и stats.yaml пока нет
        import time
        age_min = (time.time() - os.path.getmtime(d)) / 60
        rep["problems"].append("нет stats.yaml" + (f" (папке {age_min:.0f} мин — скорее всего в работе)"
                                                   if age_min < 120 else " (старая папка — шард сорвался)"))
        return rep
    try:
        st = yaml.safe_load(open(sf))
    except Exception as e:
        rep["problems"].append(f"stats.yaml не читается: {e}"); return rep
    li = st.get("last_info") or {}
    rep["n_eps"] = len(li)
    if li:
        e0 = li[next(iter(li))]
        miss = [k for k in STATS_FIELDS if k not in e0]
        if miss:
            rep["problems"].append(f"в last_info нет полей {miss}")
    if not os.path.exists(tf):
        rep["problems"].append("нет traj.npz"); return rep
    try:
        z = np.load(tf, allow_pickle=True)
        keys = set(z.files)
    except Exception as e:
        rep["problems"].append(f"traj.npz не читается: {e}"); return rep
    miss = [k for k in REQUIRED if k not in keys]
    if miss:
        rep["problems"].append(f"в traj.npz нет {miss}"); return rep
    rep["missing_ext"] = [k for k in EXTENDED if k not in keys]
    c, t, g, ids = z["cube_xyz"], z["tcp_xyz"], z["grasped"], z["ep_ids"]
    b, T = c.shape[0], c.shape[1]
    rep["b"], rep["T"] = b, T
    if t.shape != c.shape or g.shape != (b, T):
        rep["problems"].append(f"формы не сходятся: cube {c.shape} tcp {t.shape} grasped {g.shape}")
    for k in EXTENDED:
        if k == "action":
            # действий на эпизод episode_len (80), а кадров T = action_t0 + 80:
            # первые action_t0 кадров делает reset (скриптовый захват куба)
            if k in keys:
                t0 = int(z["action_t0"]) if "action_t0" in keys else T - z[k].shape[1]
                if z[k].shape[0] != b or z[k].shape[1] + t0 != T:
                    rep["problems"].append(f"action формы {z[k].shape} при T={T}, action_t0={t0}: не сходится")
            continue
        if k in keys and z[k].shape[:2] != (b, T):
            rep["problems"].append(f"{k} формы {z[k].shape}, ждём ({b},{T},…)")
    if len(ids) != b:
        rep["problems"].append(f"ep_ids {len(ids)} != эпизодов {b}")
    if start is not None and not np.array_equal(ids, np.arange(start, start + len(ids))):
        rep["problems"].append(f"ep_ids не start+i (первые {ids[:3].tolist()}, start {start})")
    if li and len(li) != b:
        rep["problems"].append(f"last_info {len(li)} эп. vs traj {b}")
    if not np.isfinite(c).all() or not np.isfinite(t).all():
        rep["problems"].append("NaN/inf в cube_xyz или tcp_xyz")
    if T < 10:
        rep["problems"].append(f"эпизод из {T} шагов — обрезан?")
    # согласованность: финал куба в traj == cube_fy в stats
    if li:
        try:
            dy = max(abs(float(li[i]["cube_fy"]) - float(c[k, -1, 1]))
                     for k, i in enumerate(sorted(li, key=int)))
            if dy > 1e-3:
                rep["problems"].append(f"cube_fy в stats vs traj расходятся на {dy*1000:.1f} мм")
        except Exception as e:
            rep["problems"].append(f"не сверить cube_fy: {e}")
    rep["grasp_ever"] = float(g.any(axis=1).mean())
    rep["release_frac"] = float((~g[:, -1]).mean())
    if rep["grasp_ever"] < 0.5:
        rep["problems"].append(f"куб схвачен лишь в {rep['grasp_ever']*100:.0f}% эпизодов")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sep", default="s")
    ap.add_argument("--orders", nargs="+", default=["noswap", "swap"])
    ap.add_argument("--last", type=int, default=0, help="проверить только N самых свежих шардов")
    args = ap.parse_args()
    bad = 0
    for order in args.orders:
        dirs = glob.glob(os.path.join(OUT, f"{args.run}-{order}-{args.sep}*"))
        dirs = [d for d in dirs if re.search(rf"-{args.sep}(\d+)$", d)]
        if args.last:
            dirs = sorted(dirs, key=os.path.getmtime)[-args.last:]
        dirs = sorted(dirs, key=lambda d: int(re.search(rf"-{args.sep}(\d+)$", d).group(1)))
        if not dirs:
            print(f"{args.run}-{order}: шардов нет"); continue
        reps = [check_shard(d, args.sep) for d in dirs]
        n_eps = sum(r.get("n_eps", 0) for r in reps)
        ext_missing = {}
        for r in reps:
            for k in r["missing_ext"]:
                ext_missing[k] = ext_missing.get(k, 0) + 1
        starts = [r["start"] for r in reps if r["start"] is not None]
        gaps = []
        # дыры в нумерации сообщаем только для ЗАКОНЧЕННОГО прогона (свежему шарду > 2 ч):
        # пока идут параллельные диапазоны, недосчитанные шарды — не ошибка
        import time
        newest = max(os.path.getmtime(d) for d in dirs)
        if len(starts) > 1 and time.time() - newest > 7200:
            step = int(np.median(np.diff(starts)))
            gaps = [s for s in range(starts[0], starts[-1], step) if s not in set(starts)]
        print(f"\n== {args.run}-{order}: шардов {len(reps)}, эпизодов {n_eps}, "
              f"диапазон {starts[0] if starts else '?'}..{starts[-1] if starts else '?'}"
              f"{', ДЫРЫ в нумерации: ' + str(gaps[:10]) if gaps else ''}")
        Ts = sorted({r.get("T") for r in reps if r.get("T")})
        ge = [r["grasp_ever"] for r in reps if "grasp_ever" in r]
        print(f"   шагов в эпизоде: {Ts}; куб схвачен хоть раз: {np.mean(ge)*100 if ge else float('nan'):.0f}%; "
              f"отпущен к финалу: {np.mean([r['release_frac'] for r in reps if 'release_frac' in r])*100 if ge else float('nan'):.0f}%")
        if ext_missing:
            print(f"   ⚠ расширенных полей нет в шардах: " +
                  ", ".join(f"{k} ({v}/{len(reps)})" for k, v in ext_missing.items()) +
                  "  ← из этих прогонов vigor/дрейф плиток/точный момент разжатия не восстановить")
        else:
            print("   ✅ все расширенные поля (boardL_xy/boardR_xy/gripper_q/qvel/action) на месте")
        probs = [r for r in reps if r["problems"]]
        bad += len(probs)
        for r in probs[:15]:
            print(f"   ❌ {r['dir']}: " + "; ".join(r["problems"]))
        if len(probs) > 15:
            print(f"   ... ещё {len(probs)-15} шардов с проблемами")
        if not probs:
            print("   ✅ структура шардов цела")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
