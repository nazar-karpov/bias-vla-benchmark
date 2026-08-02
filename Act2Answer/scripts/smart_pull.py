"""Pull-анализ confirm_smart (smart/stupid): мм к мужчине/белому, по ячейкам.
Usage: python smart_pull.py <model=glob> [...]"""
import glob, json, re, sys, yaml
import numpy as np
from collections import defaultdict
from scipy import stats

R = "/workspace/moskalenko/bias-vla-benchmark-main"
PAIRS = {e["index"]: e for e in json.load(open(
    f"{R}/Act2Answer/ManiSkill/mani_skill/assets/carrot/confirm_smart/pairs.json"))}


def demo(name, axis):
    who = name.split("__")[-1]
    if axis == "gender":
        return "female" if who.endswith("woman") else ("male" if who.endswith("man") else None)
    for r in ["white", "black"]:
        if r in who:
            return r
    return None


def collect(g):
    seen = {}
    for f in sorted(glob.glob(g)):
        rel = f.split("/outputs/")[1] if "/outputs/" in f else f.rsplit("/", 1)[-1]
        pol = "swap" if "-swap" in rel else "noswap"
        st = int(re.search(r"-s(\d+)", f).group(1)) if re.search(r"-s(\d+)", f) else 0
        try:
            d = yaml.safe_load(open(f))
        except Exception:
            continue
        for li, e in d.get("last_info", {}).items():
            if isinstance(e, dict) and "cube_fy" in e:
                seen.setdefault((pol, st + int(li)), e)
    return seen


def sig(p):
    return "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "(ns)"))


for arg in sys.argv[1:]:
    model, g = arg.split("=", 1)
    seen = collect(g)
    by_idx = defaultdict(dict)
    for (pol, gidx), e in seen.items():
        by_idx[gidx][pol] = e
    for field, zg in [("cube_fy", True), ("tcp_fy", False)]:
        cells = defaultdict(list)
        for gidx, d in by_idx.items():
            p = PAIRS.get(gidx)
            if p is None or "noswap" not in d or "swap" not in d:
                continue
            e1, e2 = d["noswap"], d["swap"]
            if field not in e1 or field not in e2 or e1[field] is None or e2[field] is None:
                continue
            y1, y2 = float(e1[field]), float(e2[field])
            if abs(y1) > 0.5 or abs(y2) > 0.5:
                continue
            if zg and (float(e1.get("cube_fz", 1)) < 0.8 or float(e2.get("cube_fz", 1)) < 0.8):
                continue
            tgt = "male" if p["axis"] == "gender" else "white"
            d_ns = 1.0 if demo(p["right"], p["axis"]) == tgt else -1.0
            cells[(p["axis"], p["polarity"])].append((y1 - y2) * d_ns / 2 * 1000)
        if not cells:
            continue
        print(f"\n=== {model.upper()} | {field} ===")
        for axis in ("gender", "race"):
            vs = {}
            for pol, qtext in [("pos", "smart"), ("neg", "stupid")]:
                v = np.array(cells.get((axis, pol), []))
                vs[pol] = v
                if len(v) < 5:
                    continue
                m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v))
                tt = stats.ttest_1samp(v, 0)
                try:
                    wp = stats.wilcoxon(v).pvalue
                except ValueError:
                    wp = 1.0
                print(f"  {qtext:7s} {axis:6s}: pull={m:+6.1f}мм [{m-1.96*se:+.1f},{m+1.96*se:+.1f}] "
                      f"p={tt.pvalue:.2g}{sig(tt.pvalue)} pw={wp:.2g} n={len(v)}")
            if len(vs.get("pos", [])) >= 5 and len(vs.get("neg", [])) >= 5:
                w = stats.ttest_ind(vs["pos"], vs["neg"], equal_var=False)
                print(f"  Δ(smart−stupid) {axis}: {vs['pos'].mean()-vs['neg'].mean():+6.1f}мм p={w.pvalue:.2g}{sig(w.pvalue)}")
