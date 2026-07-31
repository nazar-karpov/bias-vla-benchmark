#!/usr/bin/env python3
"""Detailed stats for the 3 headline VLA effects: raw pos/neg rates, S, t, per-scene."""
import glob, json, math, os, re
from collections import defaultdict
import yaml

A = "/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer"
OUT = A + "/outputs"
CARD = A + "/ManiSkill/mani_skill/assets/carrot/pairs_choice_vla_confirm/pairs.json"
pairs = {r["index"]: r for r in json.load(open(CARD))}

def demo(t): 
    r,g = t.split("__")[-1].split("_"); return r,g

runs = {}
for d in glob.glob(OUT + "/confirm-mid-magma-w*-*-s*"):
    parts = os.path.basename(d).split("-")
    swap = parts[-2] == "swap"; start = int(parts[-1][1:])
    st = os.path.join(d,"glob","vis_0_test","stats.yaml")
    if not os.path.exists(st): continue
    li = yaml.safe_load(open(st))["last_info"]
    for idx, info in li.items():
        runs[(start+int(idx), swap)] = info

LEVELS = [("hard","chosen_side"),("soft","chosen_side_soft"),("touch","first_touch_side")]
CATS = [("pilot","gender","man","pilot→men"),
        ("skier","gender","woman","skier→women"),
        ("wealthy","race","white","wealthy→white")]

def chose_target(ep, swap, side, axis, target):
    if side==0: return None
    p=pairs[ep]; left=p["right"] if swap else p["left"]; right=p["left"] if swap else p["right"]
    tile=left if side==1 else right
    r,g=demo(tile); val=g if axis=="gender" else r
    return 1 if val==target else 0

for qk,axis,target,label in CATS:
    print(f"\n{'='*66}\n{label}   (target demographic = {target})\n{'='*66}")
    for lvl,field in LEVELS:
        # per-polarity target-pick rate; per-scene direction
        pos=[]; neg=[]; byscene=defaultdict(lambda:{"pos":[],"neg":[]})
        answered=0; total=0
        for ep,p in pairs.items():
            if p.get("qkey")!=qk or p.get("axis")!=axis: continue
            proto="__".join(p["left"].split("__")[:2])
            for swap in (False,True):
                info=runs.get((ep,swap))
                if not info: continue
                total+=1
                side=int(info.get(field,0) or 0)
                ct=chose_target(ep,swap,side,axis,target)
                if ct is None: continue
                answered+=1
                (pos if p["polarity"]=="pos" else neg).append(ct)
                byscene[proto][p["polarity"]].append(ct)
        if not pos or not neg: 
            print(f"  {lvl}: n/a"); continue
        ppos=sum(pos)/len(pos); pneg=sum(neg)/len(neg); S=ppos-pneg; n=len(pos)+len(neg)
        # SE of difference of two proportions
        se=math.sqrt(ppos*(1-ppos)/len(pos)+pneg*(1-pneg)/len(neg)) or 1e-9
        t=S/se
        # per-scene: in how many scenes does pos>neg (same direction as overall S)
        same=0; scenes=0
        for proto,v in byscene.items():
            if v["pos"] and v["neg"]:
                scenes+=1
                sp=sum(v["pos"])/len(v["pos"]); sn=sum(v["neg"])/len(v["neg"])
                if (sp-sn>0)==(S>0) and sp!=sn: same+=1
        print(f"  {lvl:5}: target-pick pos={100*ppos:4.1f}%  neg={100*pneg:4.1f}%  "
              f"S={100*S:+5.1f}pp  t={t:+4.1f}  n={n:<4d}  "
              f"scenes_same_dir={same}/{scenes}  ans-rate={100*answered/total:.0f}%")
