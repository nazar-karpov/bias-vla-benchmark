#!/usr/bin/env python3
"""Pick the clearest demo episodes for the 3 headline effects and map to video files.

For each category we want episodes where the model MOTORICALLY put the cube on the
biased demographic (hard chosen_side), and — the strongest demo — where it chose the
SAME demographic tile in BOTH layouts (noswap+swap), i.e. content-driven not positional.
"""
import glob, json, os, re
from collections import defaultdict
import yaml

A = "/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer"
OUT = A + "/outputs"
CARD = A + "/ManiSkill/mani_skill/assets/carrot/pairs_choice_vla_confirm/pairs.json"
pairs = {r["index"]: r for r in json.load(open(CARD))}

def demo(tile):
    race, gen = tile.split("__")[-1].split("_")
    return race, gen

# gather: (ep, swap) -> (chosen_side, first_touch, dir, video_path, is_answered)
rec = {}
for d in glob.glob(OUT + "/confirm-mid-magma-w*-*-s*"):
    name = os.path.basename(d)
    parts = name.split("-")
    swap = parts[-2] == "swap"
    start = int(parts[-1][1:])
    st = os.path.join(d, "glob", "vis_0_test", "stats.yaml")
    if not os.path.exists(st): continue
    li = yaml.safe_load(open(st))["last_info"]
    vis = os.path.dirname(st)
    vids = {}
    for f in os.listdir(vis):
        m = re.match(r"video_(\d+)-s_(\d+)\.mp4", f)
        if m: vids[int(m.group(1))] = os.path.join(vis, f)
    for idx, info in li.items():
        ep = start + int(idx)
        rec[(ep, swap)] = dict(
            cs=int(info.get("chosen_side", 0) or 0),
            ft=int(info.get("first_touch_side", 0) or 0),
            ans=bool(info.get("is_answered", False)),
            vid=vids.get(int(idx)),
        )

# which demographic tile did the model pick? side 1=left,2=right in the shown frame;
# swap flips physical tiles. picked tile token:
def picked_token(ep, swap, side):
    if side == 0: return None
    p = pairs[ep]
    left = p["right"] if swap else p["left"]
    right = p["left"] if swap else p["right"]
    tile = left if side == 1 else right
    return demo(tile)  # (race, gen)

# Categories: (qkey, axis, which demographic value is the "biased" pick, label)
CATS = [
    ("pilot",   "gender", ("man",),   "pilot→men"),
    ("skier",   "gender", ("woman",), "skier→women"),
    ("wealthy", "race",   ("white",), "wealthy→white"),
]

for qk, axis, target, label in CATS:
    print(f"\n===== {label}  (qkey={qk} axis={axis} target={target[0]}) =====")
    # candidates: episodes of this qkey/axis where BOTH layouts answered (hard) and
    # BOTH picked the target demographic -> content-stable, cleanest demo.
    cand_both = []
    cand_one = []
    for ep, p in pairs.items():
        if p.get("qkey") != qk or p.get("axis") != axis: continue
        ns = rec.get((ep, False)); sw = rec.get((ep, True))
        if not ns or not sw: continue
        gn = picked_token(ep, False, ns["cs"]); gs = picked_token(ep, True, sw["cs"])
        # target index: gender->[1], race->[0]
        ti = 1 if axis == "gender" else 0
        pick_ns = gn[ti] if gn else None
        pick_sw = gs[ti] if gs else None
        proto = "__".join(p["left"].split("__")[:2])
        if pick_ns in target and pick_sw in target and ns["vid"] and sw["vid"]:
            cand_both.append((ep, p["polarity"], proto, ns["vid"], sw["vid"]))
        elif (pick_ns in target and ns["vid"]) or (pick_sw in target and sw["vid"]):
            v = ns["vid"] if pick_ns in target else sw["vid"]
            cand_one.append((ep, p["polarity"], proto, v))
    print(f"  content-stable (both layouts picked {target[0]}): {len(cand_both)}")
    for ep, pol, proto, v1, v2 in cand_both[:6]:
        print(f"    ep{ep} pol={pol} {proto}")
        print(f"      NOSWAP {v1}")
        print(f"      SWAP   {v2}")
    print(f"  one-layout picks: {len(cand_one)}")
