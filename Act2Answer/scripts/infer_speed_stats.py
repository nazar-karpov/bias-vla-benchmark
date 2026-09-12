#!/usr/bin/env python3
"""Скорость инференса моделей по уже лежащим на диске прогонам.

Для каждого шарда: старт = mtime glob/config.yaml (пишется при запуске), конец = mtime
glob/vis_0_test/traj.npz (пишется по завершении), эпизодов = число инструкций в stats.yaml.
Группировка по модели (имя прогона `<prog>-[<vla>-]<assets>-<order>-s<i>`; без vla = magma).

  python infer_speed_stats.py [--outputs DIR] [--prefix g10-] [--csv metrics/infer_speed.csv]

Колонки: shards, episodes, sec_per_ep (медиана по шардам, один поток на карте),
eps_per_h_stream, wall_h (от первого старта до последнего финиша), eps_per_h_wall
(с учётом всех параллельных потоков), streams_est = сумма секунд шардов / wall.
"""
import argparse, os, re, statistics as st, sys, csv, time
import yaml

KNOWN_VLA = ("magma", "internvla", "spatialvla", "gr00t", "xiaomi", "rldx", "rldx2", "xvla")

def parse(name):
    m = re.match(r"^(?P<prog>[a-z0-9]+)-(?P<rest>.+)-(?P<order>noswap|swap)-s(?P<i>\d+)$", name)
    if not m: return None
    rest = m["rest"].split("-")
    vla = rest[0] if rest[0] in KNOWN_VLA else "magma"
    assets = "-".join(rest[1:]) if rest[0] in KNOWN_VLA else m["rest"]
    return m["prog"], vla, assets, m["order"], int(m["i"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=os.path.join(os.path.dirname(__file__), "..", "outputs"))
    ap.add_argument("--prefix", default="")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--by-assets", action="store_true", help="разбивать ещё и по датасету")
    a = ap.parse_args()
    rows = {}
    for n in sorted(os.listdir(a.outputs)):
        if not n.startswith(a.prefix): continue
        p = parse(n)
        if not p: continue
        prog, vla, assets, order, i = p
        cfg = os.path.join(a.outputs, n, "glob", "config.yaml")
        trj = os.path.join(a.outputs, n, "glob", "vis_0_test", "traj.npz")
        sty = os.path.join(a.outputs, n, "glob", "vis_0_test", "stats.yaml")
        if not (os.path.exists(cfg) and os.path.exists(trj) and os.path.exists(sty)): continue
        t0, t1 = os.path.getmtime(cfg), os.path.getmtime(trj)
        try:
            neps = len(yaml.safe_load(open(sty))["instruction"])
        except Exception:
            continue
        if t1 <= t0 or neps == 0: continue
        key = (prog, vla, assets) if a.by_assets else (prog, vla, "*")
        rows.setdefault(key, []).append((t0, t1, neps))
    out = []
    for key, lst in sorted(rows.items()):
        secs = [t1 - t0 for t0, t1, _ in lst]
        neps = sum(n for _, _, n in lst)
        spe = [ (t1 - t0) / n for t0, t1, n in lst ]
        wall = max(t1 for _, t1, _ in lst) - min(t0 for t0, _, _ in lst)
        out.append(dict(prog=key[0], vla=key[1], assets=key[2], shards=len(lst), episodes=neps,
                        sec_per_ep=round(st.median(spe), 1), shard_min=round(st.median(secs) / 60, 1),
                        eps_per_h_stream=round(3600 / st.median(spe)),
                        wall_h=round(wall / 3600, 2), eps_per_h_wall=round(neps / (wall / 3600)) if wall > 0 else None,
                        streams_est=round(sum(secs) / wall, 1) if wall > 0 else None,
                        first=time.strftime("%m-%d %H:%M", time.localtime(min(t0 for t0, _, _ in lst))),
                        last=time.strftime("%m-%d %H:%M", time.localtime(max(t1 for _, t1, _ in lst)))))
    if not out:
        print("нет шардов"); return
    cols = list(out[0])
    w = {c: max(len(c), *(len(str(r[c])) for r in out)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    for r in out: print("  ".join(str(r[c]).ljust(w[c]) for c in cols))
    if a.csv:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        with open(a.csv, "w", newline="") as f:
            cw = csv.DictWriter(f, fieldnames=cols); cw.writeheader(); cw.writerows(out)
        print("->", a.csv)

if __name__ == "__main__":
    main()
