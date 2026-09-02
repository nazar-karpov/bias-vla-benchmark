"""Позиционный крен каждой VLA: доля выбора ЛЕВОЙ плитки среди ответивших эпизодов,
по всем confirm-прогонам (объединяя шарды каждой модели, noswap+swap вместе — крен не
зависит от порядка). chosen_side: 1=left, 2=right. 50% = крена нет."""
import glob as g, os, re, yaml, collections

R = "/workspace/moskalenko/bias-vla-benchmark-main"

def model_of(run):
    for m in ["gr00t", "internvla", "magma", "rldx2", "rldx", "xiaomi", "xvla"]:
        if m in run.lower():
            return {"rldx2": "RLDX2", "rldx": "RLDX1", "internvla": "InternVLA",
                    "gr00t": "GR00T", "magma": "Magma", "xiaomi": "Xiaomi", "xvla": "xVLA"}[m]
    return None

# на каждую модель считаем только ОДИН канонический набор шардов, чтобы не задваивать
# берём ALL/FULL, иначе A/B/partB. Ключ = (model), собираем множество (run,ep) уникальных.
agg = collections.defaultdict(lambda: [0, 0, 0])  # model -> [left, right, noanswer]
seen = collections.defaultdict(set)

pref = {}  # model -> предпочтительный маркер набора
for f in sorted(g.glob(f"{R}/Act2Answer/outputs/confirm-*/glob/vis_0_test/stats.yaml")):
    run = f.split("/outputs/")[1].split("/")[0]
    mdl = model_of(run)
    if not mdl:
        continue
    tag = "ALL" if ("-ALL-" in run or "-FULL-" in run or "FULL" in run) else "other"
    pref.setdefault(mdl, tag)
    if pref[mdl] == "ALL" and tag != "ALL":
        continue
    if pref[mdl] != "ALL" and tag == "ALL":
        agg[mdl] = [0, 0, 0]; seen[mdl] = set(); pref[mdl] = "ALL"
    st = int(re.search(r"-s(\d+)$", run).group(1)) if re.search(r"-s(\d+)$", run) else 0
    pol = "swap" if "-swap" in run else "noswap"
    try:
        li = yaml.safe_load(open(f))["last_info"]
    except Exception:
        continue
    for i, e in li.items():
        if not isinstance(e, dict):
            continue
        key = (pol, st + int(i))
        if key in seen[mdl]:
            continue
        seen[mdl].add(key)
        ans = int(float(e.get("is_answered", 0)))
        ch = int(float(e.get("chosen_side", 0)))
        if not ans or ch == 0:
            agg[mdl][2] += 1
        elif ch == 1:
            agg[mdl][0] += 1
        else:
            agg[mdl][1] += 1

print(f"{'model':10s} {'left%':>7s} {'n_ans':>7s} {'ans_rate':>9s}  крен")
lefts = []
for mdl in sorted(agg):
    L, Rr, N = agg[mdl]
    tot = L + Rr + N
    na = L + Rr
    lp = 100 * L / na if na else 0
    lefts.append((mdl, lp, na))
    side = "ЛЕВО" if lp > 55 else ("ПРАВО" if lp < 45 else "нет")
    print(f"{mdl:10s} {lp:6.1f}% {na:7d} {100*na/tot:8.1f}%  {abs(lp-50):.0f}пп {side}")

# средний крен по модулю
import statistics
dev = [abs(lp - 50) for _, lp, _ in lefts]
print(f"\nсредний |крен| по 7 моделям: {statistics.mean(dev):.1f}пп")
print(f"диапазон left%: {min(l for _,l,_ in lefts):.0f}% .. {max(l for _,l,_ in lefts):.0f}%")
