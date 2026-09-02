"""Поиск ФАНТОМНЫХ эпизодов на РАСОВОЙ оси с МУЖСКОЙ парой:
кубик НЕ схвачен (остался в центре, |cube_fy|<~0.05, cube_fz~стол) НО рука уехала к
плитке (|tcp_fy| большой). Мужская пара = обе плитки white_man/black_man.
Печатает кандидатов, отсортированных по «дальности доноса руки»."""
import glob as g, json, os, re, yaml

R = "/workspace/moskalenko/bias-vla-benchmark-main"
PAIRS = {e["index"]: e for e in json.load(open(
    f"{R}/Act2Answer/ManiSkill/mani_skill/assets/carrot/pairs_choice_vla_confirm/pairs.json"))}

# какие индексы — race + мужская пара
male_race = {}
for idx, p in PAIRS.items():
    if p.get("axis") != "race":
        continue
    who = (p["left"].split("__")[-1], p["right"].split("__")[-1])
    if all(w.endswith("_man") for w in who):
        male_race[idx] = p

cands = []
for f in sorted(g.glob(f"{R}/Act2Answer/outputs/confirm-*/glob/vis_0_test/stats.yaml")):
    run = f.split("/outputs/")[1].split("/")[0]
    st = int(re.search(r"-s(\d+)$", run).group(1)) if re.search(r"-s(\d+)$", run) else 0
    d = os.path.dirname(f)
    try:
        li = yaml.safe_load(open(f))["last_info"]
    except Exception:
        continue
    for i, e in li.items():
        if not isinstance(e, dict) or "cube_fy" not in e:
            continue
        ep = st + int(i)
        p = male_race.get(ep)
        if not p:
            continue
        cy = float(e["cube_fy"]); cx = float(e.get("cube_fx", 9.0)); cz = float(e.get("cube_fz", 1.0))
        ty = e.get("tcp_fy"); tx = e.get("tcp_fx")
        if ty is None:
            continue
        ty = float(ty)
        # кубик остался в центре (не схвачен): близко к 0 по y и на столе
        cube_central = abs(cy) < 0.06 and cz > 0.75 and abs(cx) < 0.35
        # рука уехала к плитке
        bR = float(e.get("boardR_y", 0.155)); bL = float(e.get("boardL_y", -0.155))
        arm_reach = abs(ty) > 0.10
        if cube_central and arm_reach:
            pol = "swap" if "-swap" in run else "noswap"
            side = "R" if ty > 0 else "L"
            who = (p["left"] if (side == "L") ^ (pol == "swap") else p["right"]).split("__")[-1]
            vids = g.glob(os.path.join(d, f"video_{int(i)}-s_*.mp4"))
            cands.append((abs(ty), run, i, ep, pol, side, who, p["question"],
                          p["left"].split("__")[1], round(cy, 3), round(ty, 3), bool(vids)))

cands.sort(reverse=True)
for c in cands[:40]:
    reach, run, i, ep, pol, side, who, q, scene, cy, ty, vid = c
    print(f"reach={reach:.2f} {run:40s} i={i} ep{ep} {pol:6s} arm->{side} {who:9s} "
          f"{q.replace('Put cube on ','')[:22]:24s} {scene:14s} cube_y={cy} tcp_y={ty} vid={vid}")
print(f"\n{len(cands)} кандидатов; мужских race-эпизодов всего: {len(male_race)}")
