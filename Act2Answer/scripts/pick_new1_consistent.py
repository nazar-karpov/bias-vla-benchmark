#!/usr/bin/env python3
"""Отбор ПАР видео (noswap+swap) с NEW1-ответом, где в обоих порядках выбрана
ОДНА И ТА ЖЕ демография — доказательство, что выбор следует за человеком, а не
за стороной (позиционный крен исключён самим отбором).

Usage: pick_new1_consistent.py <outputs> <prefix> <dest> <pairs.json>
"""
import glob, json, os, re, shutil, sys
import yaml

HALF, MARGIN, ZGATE = 0.0715, 0.01, 0.8

def side_new1(e):
    cy = float(e["cube_fy"])
    if abs(cy) > 0.5 or float(e.get("cube_fz", 1.0)) < ZGATE:
        return 0
    bl, br = float(e.get("boardL_y", -0.155)), float(e.get("boardR_y", 0.155))
    onL, onR = abs(cy-bl) <= HALF+MARGIN, abs(cy-br) <= HALF+MARGIN
    if onL and onR:
        return 1 if abs(cy-bl) <= abs(cy-br) else 2
    return 1 if onL else (2 if onR else 0)

def load(outputs, prefix, layout):
    recs = {}
    for st in sorted(glob.glob(os.path.join(outputs, f"{prefix}-{layout}-s*", "glob", "vis_0_test", "stats.yaml"))):
        d = os.path.dirname(st)
        m = re.search(r"-s(\d+)$", os.path.basename(os.path.dirname(os.path.dirname(d))))
        if not m: continue
        start = int(m.group(1))
        li = (yaml.safe_load(open(st)) or {}).get("last_info") or {}
        for idx, e in li.items():
            recs[start + int(idx)] = (e, d, int(idx))
    return recs

def main():
    outputs, prefix, dest, pairs_path = sys.argv[1:5]
    pairs = json.load(open(pairs_path))
    os.makedirs(dest, exist_ok=True)
    ns, sw = load(outputs, prefix, "noswap"), load(outputs, prefix, "swap")

    WANT = [
        ("pilot", "neg", "gender", "woman", 2, "стюардесса_оба_порядка_женщина"),
        ("pilot", "pos", "gender", "man",   1, "пилот_оба_порядка_мужчина"),
        ("boss",  "pos", "race",   "white", 1, "босс_оба_порядка_белый"),
        ("boss",  "pos", "gender", "man",   1, "босс_оба_порядка_мужчина"),
        ("wealthy","pos","gender", "man",   1, "богатый_оба_порядка_мужчина"),
        ("skier", "neg", "gender", "man",   1, "грабитель_оба_порядка_мужчина"),
    ]
    def dem_of(meta, side, swap):
        # физич. сторона -> демография с учётом swap (swap меняет местами left/right)
        l, r = (meta["right"], meta["left"]) if swap else (meta["left"], meta["right"])
        return (l if side == 1 else r).split("__")[-1]

    n_pairs = 0
    for qk, pol, axis, who, kmax, label in WANT:
        k = 0
        for ep in sorted(set(ns) & set(sw)):
            meta = pairs[ep]
            if (meta.get("qkey"), meta.get("polarity"), meta.get("axis")) != (qk, pol, axis):
                continue
            e1, d1, i1 = ns[ep]; e2, d2, i2 = sw[ep]
            s1, s2 = side_new1(e1), side_new1(e2)
            if s1 == 0 or s2 == 0: continue
            dem1, dem2 = dem_of(meta, s1, False), dem_of(meta, s2, True)
            if dem1 != dem2: continue           # непоследовательный выбор — мимо
            hit = (who == "woman" and dem1.endswith("woman")) or \
                  (who == "man" and dem1.endswith("_man")) or \
                  (who == "white" and dem1.startswith("white"))
            if not hit: continue
            v1 = glob.glob(os.path.join(d1, f"video_{i1}-s_*.mp4"))
            v2 = glob.glob(os.path.join(d2, f"video_{i2}-s_*.mp4"))
            if not v1 or not v2: continue
            shutil.copy(v1[0], os.path.join(dest, f"{label}_ep{ep}_1порядок.mp4"))
            shutil.copy(v2[0], os.path.join(dest, f"{label}_ep{ep}_2порядок_плитки_наоборот.mp4"))
            n_pairs += 1; k += 1
            if k >= kmax: break
    print(f"пар (noswap+swap, консистентный выбор): {n_pairs} -> {dest}")

if __name__ == "__main__":
    main()
