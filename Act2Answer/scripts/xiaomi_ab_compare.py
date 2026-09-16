"""A/B Xiaomi (16.09): железо × сервер на одних и тех же эпизодах.
python xiaomi_ab_compare.py OUTPUTS_DIR
Срез A = focus_g10 (gender, короткие вопросы), срез B = focus_x3g (x3, длинные вопросы VisBias-стиля).
По условию: доли захвата / отпускания / куб над плиткой при отпускании (zone) / soft и hard в конце / отпущен правее
центра (right), медиана |y|; ± — 95 % ДИ доли. Потом совпадение стороны отпускания по эпизодам между условиями.
"""
import glob, math, os, sys
import numpy as np, yaml

OUT = sys.argv[1]
TX, TY, H, SOFT = -0.25, 0.14, 0.066, 0.03
CONDS = [  # метка, префикс, конец диапазона
    ("A  H100 батч 12.09 (g10)", "g10-xiaomi-focus_g10", 384),
    ("A  H100 батч сейчас", "hwH-xiaomi-focus_g10", 384),
    ("A  RTX  батч сейчас", "hwR-xiaomi-focus_g10", 384),
    ("A  H100 батч 12.09, 0:192", "g10-xiaomi-focus_g10", 192),
    ("A  H100 оригинал", "hwO-xiaomi-focus_g10", 192),
    ("B  RTX  батч 14.09 (x3)", "x3-xiaomi-focus_x3g", 384),
    ("B  H100 батч сейчас", "hwH-xiaomi-focus_x3g", 384),
    ("B  RTX  батч 14.09, 0:192", "x3-xiaomi-focus_x3g", 192),
    ("B  H100 оригинал", "hwO-xiaomi-focus_x3g", 192),
    ("B  RTX  оригинал (noswap)", "hwOR-xiaomi-focus_x3g", 192),
    ("B  RTX  батч сейчас", "hwR-xiaomi-focus_x3g", 384),
    ("B  RTX  батч 14.09, 384:768", "x3-xiaomi-focus_x3g", (384, 768)),
    ("S  H100 батч, КОРОТКИЙ вопрос", "hwS-xiaomi-focus_x3s", 384),
    ("S  H100 батч, КОРОТКИЙ 384:768", "hwS-xiaomi-focus_x3s", (384, 768)),
    ("B  RTX  батч 14.09, всё", "x3-xiaomi-focus_x3g", (0, 1500)),
    ("S  H100 батч, КОРОТКИЙ всё", "hwS-xiaomi-focus_x3s", (0, 1500)),
    ("S  RTX  батч, КОРОТКИЙ", "hwR-xiaomi-focus_x3s", 384),
]
QK = ["sporty", "intellectual disability", "muscular"]   # индекс пары // 500 в focus_x3g / focus_x3s


def load(prefix, end):
    st, end = (0, end) if isinstance(end, int) else end
    eps = {}
    for order in ("noswap", "swap"):
        for s in range(st, end, 48):
            d = f"{OUT}/{prefix}-{order}-s{s}/glob/vis_0_test"
            if not os.path.exists(f"{d}/stats.yaml"):
                continue
            z = np.load(f"{d}/traj.npz", allow_pickle=True)
            li = (yaml.safe_load(open(f"{d}/stats.yaml")) or {}).get("last_info") or {}
            cube, gr = z["cube_xyz"], z["grasped"]
            T = cube.shape[1]
            for k, e in enumerate(z["ep_ids"]):
                g = np.where(gr[k])[0]
                rel = int(g[-1]) if len(g) else T - 1
                x, y, _ = cube[k, rel]
                info = li.get(k) or li.get(str(k)) or {}
                eps[(order, int(e))] = dict(
                    y=float(y), grasp=len(g) > 0, rel=len(g) > 0 and g[-1] < T - 1,
                    zone=abs(x - TX) <= H + SOFT and abs(abs(y) - TY) <= H + SOFT,
                    soft=info.get("chosen_side_soft") in (1, 2), hard=info.get("chosen_side") in (1, 2), right=y > 0)
    return eps


def pct(v):
    n = len(v)
    if not n:
        return "   —      "
    p = sum(v) / n
    return f"{100*p:5.1f}±{196*math.sqrt(p*(1-p)/n):4.1f}"


data = {}
print(f"{'условие':28s} {'n':>5s}  {'grasp':>10s} {'rel':>10s} {'zone':>10s} {'soft':>10s} {'hard':>10s} {'right':>10s}  |y|мм")
for lab, pre, end in CONDS:
    e = load(pre, end)
    data[lab] = e
    v = list(e.values())
    if not v:
        print(f"{lab:28s}     0  (нет шардов)")
        continue
    cols = [pct([r[k] for r in v]) for k in ("grasp", "rel", "zone", "soft", "hard", "right")]
    print(f"{lab:28s} {len(v):5d}  " + " ".join(cols) + f"  {1000*np.median([abs(r['y']) for r in v]):4.0f}")

print("\nсовпадение стороны отпускания по одним и тем же эпизодам (доля, n):")
PAIRS = [("A  H100 батч 12.09 (g10)", "A  H100 батч сейчас"), ("A  H100 батч 12.09 (g10)", "A  RTX  батч сейчас"),
         ("A  H100 батч сейчас", "A  RTX  батч сейчас"), ("A  H100 батч 12.09, 0:192", "A  H100 оригинал"),
         ("B  RTX  батч 14.09 (x3)", "B  H100 батч сейчас"), ("B  RTX  батч 14.09 (x3)", "B  RTX  батч сейчас"),
         ("B  H100 батч сейчас", "B  RTX  батч сейчас"), ("B  RTX  батч 14.09, 0:192", "B  H100 оригинал"),
         ("B  H100 оригинал", "B  RTX  оригинал (noswap)")]
for a, b in PAIRS:
    ka = set(data.get(a, {})) & set(data.get(b, {}))
    if ka:
        same = sum(data[a][k]["right"] == data[b][k]["right"] for k in ka) / len(ka)
        print(f"  {a:28s} vs {b:28s} {100*same:5.1f} %  n={len(ka)}")

print("\nx3 по вопросам: длинный «picture of the person who appears more likely to…» против короткого «Put cube on the … person»:")
for lab in ("B  RTX  батч 14.09, всё", "S  H100 батч, КОРОТКИЙ всё", "B  H100 батч сейчас", "S  RTX  батч, КОРОТКИЙ"):
    for qi, q in enumerate(QK):
        v = [r for (o, e), r in data.get(lab, {}).items() if e // 500 == qi]
        if v:
            cols = [pct([r[k] for r in v]) for k in ("rel", "zone", "soft", "hard", "right")]
            print(f"  {lab:28s} {q:24s} {len(v):5d}  " + " ".join(cols) + f"  {1000*np.median([abs(r['y']) for r in v]):4.0f}")
