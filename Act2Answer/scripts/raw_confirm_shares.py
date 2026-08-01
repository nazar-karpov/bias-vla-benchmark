#!/usr/bin/env python3
"""Сырые доли P(выбрал A) по каждой ячейке вопрос×ось×полярность (не S!),
плюс примеры эпизодов с путями к видео. Логика чтения = watch_confirm_incremental.
Usage: raw_confirm_shares.py <pairs.json> <outputs> --noswap-prefix P1 --swap-prefix P2 [--examples]
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict
import yaml

LEVELS = [("hard", "chosen_side"), ("soft", "chosen_side_soft"), ("touch", "first_touch_side")]

def load_shards(outputs, prefix):
    out = {}
    for st in sorted(glob.glob(os.path.join(outputs, f"{prefix}-s*", "glob", "vis_0_test", "stats.yaml"))):
        shard_dir = os.path.dirname(os.path.dirname(os.path.dirname(st)))
        m = re.search(r"-s(\d+)$", os.path.basename(shard_dir))
        if not m:
            continue
        start = int(m.group(1))
        try:
            li = (yaml.safe_load(open(st)) or {}).get("last_info") or {}
        except Exception:
            continue
        for idx, info in li.items():
            out[start + int(idx)] = (info, shard_dir, int(idx))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs"); ap.add_argument("outputs")
    ap.add_argument("--noswap-prefix", required=True)
    ap.add_argument("--swap-prefix", required=True)
    ap.add_argument("--examples", action="store_true")
    a = ap.parse_args()
    pairs = json.load(open(a.pairs))
    ns = load_shards(a.outputs, a.noswap_prefix)
    sw = load_shards(a.outputs, a.swap_prefix)

    # (qkey, axis, lvl, polarity) -> [chose_a...]
    g = defaultdict(list)
    for runs, swap in ((ns, False), (sw, True)):
        for ep, (info, _, _) in runs.items():
            meta = pairs[ep]
            if "qkey" not in meta:
                continue
            for lvl, field in LEVELS:
                side = int(info.get(field, 0) or 0)
                if side == 0:
                    continue
                chose_a = 1 if ((side == 1) != swap) else 0
                g[(meta["qkey"], meta["axis"], lvl, meta["polarity"])].append(chose_a)

    # вопрос из pairs (текст инструкции по полярности)
    qtext = {}
    for p in pairs:
        if "qkey" in p:
            qtext[(p["qkey"], p["polarity"])] = p.get("question", "")

    qkeys, seen = [], set()
    for p in pairs:
        k = p.get("qkey")
        if k and k not in seen:
            seen.add(k); qkeys.append(k)

    print(f"{'вопрос':<9}{'ось':<7}{'поляр.':<7}{'инструкция':<38}" +
          "".join(f"{l+' P(A)% (n)':>17}" for l, _ in LEVELS))
    for qk in qkeys:
        for axis in ("gender", "race"):
            for pol in ("pos", "neg"):
                cells = []
                for lvl, _ in LEVELS:
                    v = g.get((qk, axis, lvl, pol))
                    if not v:
                        cells.append(f"{'—':>17}"); continue
                    cells.append(f"{100*sum(v)/len(v):>8.1f} (n{len(v):<4d})")
                print(f"{qk:<9}{axis:<7}{pol:<7}{qtext.get((qk,pol),'')[:36]:<38}" + "".join(cells))
        print()

    if a.examples:
        print("=== ПРИМЕРЫ ЭПИЗОДОВ (noswap, hard-ответ) ===")
        want = [("pilot","gender","pos",1), ("pilot","gender","neg",2),
                ("boss","race","pos",1), ("wealthy","gender","pos",1),
                ("skier","gender","pos",1), ("boss","gender","pos",2)]
        for qk, axis, pol, side_want in want:
            cnt = 0
            for ep, (info, shard_dir, idx) in sorted(ns.items()):
                meta = pairs[ep]
                if meta.get("qkey")!=qk or meta.get("axis")!=axis or meta.get("polarity")!=pol:
                    continue
                if int(info.get("chosen_side",0) or 0) != side_want:
                    continue
                vids = glob.glob(os.path.join(shard_dir,"glob","vis_0_test",f"video_{idx}-s_*.mp4"))
                if vids:
                    label = "choseA" if side_want==1 else "choseB"
                    print(f"{qk}/{axis}/{pol} ep{ep} {label}: {vids[0]}")
                    cnt += 1
                if cnt >= 2:
                    break
        # пример «не доехал» (hard=0, touch!=0)
        cnt=0
        for ep,(info,shard_dir,idx) in sorted(ns.items()):
            if int(info.get("chosen_side",0) or 0)==0 and int(info.get("first_touch_side",0) or 0)!=0:
                vids = glob.glob(os.path.join(shard_dir,"glob","vis_0_test",f"video_{idx}-s_*.mp4"))
                if vids:
                    print(f"NO-ANSWER ep{ep} ({pairs[ep].get('qkey')}/{pairs[ep].get('polarity')}): {vids[0]}")
                    cnt+=1
                if cnt>=2: break

if __name__ == "__main__":
    main()
