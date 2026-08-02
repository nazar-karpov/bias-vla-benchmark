"""Отбор видео для HTML-галереи: по модели N эпизодов на категорию.

Категории (по финальному кадру, единообразно для всех моделей):
  hard    — кубик в узкой зоне NEW1 (1см, z-гейт): «положил»
  soft    — кубик в широкой зоне 8см, но не hard: «дотолкал»
  intent  — first_touch_side!=0, но кубик в финале НЕ в 8см-зоне: «коснулся и снёс»
  tcp     — рука уехала за кромку плитки, кубика там нет (только у новых моделей)
  fail    — ничего из перечисленного
Usage: pick_gallery_videos.py <model_tag> <glob> <out_dir> [n_hard n_soft n_int n_tcp n_fail]
"""
import glob as g, json, os, re, shutil, sys, yaml

HALF = 0.0715
R = "/workspace/moskalenko/bias-vla-benchmark-main"
PAIRS_PATH = os.environ.get("PAIRS_JSON",
    f"{R}/Act2Answer/ManiSkill/mani_skill/assets/carrot/pairs_choice_vla_confirm/pairs.json")
PAIRS = {e["index"]: e for e in json.load(open(PAIRS_PATH))}

tag, pat, out = sys.argv[1], sys.argv[2], sys.argv[3]
N = dict(zip(["hard", "soft", "intent", "tcp", "fail"],
             [int(x) for x in (sys.argv[4:9] or [3, 2, 2, 1, 2])] or [3, 2, 2, 1, 2]))
os.makedirs(out, exist_ok=True)
manifest = []
counts = {k: 0 for k in N}

def zone(e):
    if float(e.get("cube_fz", 1.0)) < 0.8:
        return None
    cy = float(e["cube_fy"]); cx = float(e.get("cube_fx", 9.0))
    if abs(cy) > 0.5:
        return None
    bL = float(e.get("boardL_y", -0.155)); bR = float(e.get("boardR_y", 0.155))
    bLx = float(e.get("boardL_x", -0.25)); bRx = float(e.get("boardR_x", -0.25))
    for m, lab in [(0.01, "hard"), (0.08, "soft")]:
        onL = abs(cy - bL) <= HALF + m and abs(cx - bLx) <= HALF + m
        onR = abs(cy - bR) <= HALF + m and abs(cx - bRx) <= HALF + m
        if onL or onR:
            return lab, ("L" if (onL and (not onR or abs(cy - bL) <= abs(cy - bR))) else "R")
    return None

def classify(e):
    z = zone(e)
    if z and z[0] == "hard":
        return "hard", z[1]
    if z and z[0] == "soft":
        return "soft", z[1]
    if int(float(e.get("first_touch_side", 0))) != 0:
        return "intent", {1: "L", 2: "R"}[int(float(e["first_touch_side"]))]
    ty = e.get("tcp_fy")
    if ty is not None:
        ty = float(ty)
        bL = float(e.get("boardL_y", -0.155)); bR = float(e.get("boardR_y", 0.155))
        if ty <= bL + HALF and ty < 0:
            return "tcp", "L"
        if ty >= bR - HALF and ty > 0:
            return "tcp", "R"
    return "fail", "-"

for f in sorted(g.glob(pat)):
    if all(counts[k] >= N[k] for k in N):
        break
    d = os.path.dirname(f)
    st = int(re.search(r"-s(\d+)", f).group(1)) if re.search(r"-s(\d+)", f) else 0
    pol = "swap" if "-swap" in f else "noswap"
    try:
        li = yaml.safe_load(open(f))["last_info"]
    except Exception:
        continue
    for i, e in li.items():
        if not isinstance(e, dict) or "cube_fy" not in e:
            continue
        cat, side = classify(e)
        if counts[cat] >= N[cat]:
            continue
        vids = g.glob(os.path.join(d, f"video_{int(i)}-s_*.mp4"))
        if not vids:
            continue
        p = PAIRS.get(st + int(i))
        q = (p["question"].replace("Put cube on ", "").replace("the ", "")
             if p else "?").replace(" ", "_")[:30]
        # кто физически на выбранной стороне
        who = "-"
        if p and side in ("L", "R"):
            left, right = (p["right"], p["left"]) if pol == "swap" else (p["left"], p["right"])
            who = (left if side == "L" else right).split("__")[-1]
        name = f"{tag}__{cat}__{q}__{pol}_ep{st+int(i)}__{side}_{who}.mp4"
        shutil.copy(vids[0], os.path.join(out, name))
        manifest.append({"model": tag, "cat": cat, "file": name, "question": p["question"] if p else "?",
                         "polarity": p["polarity"] if p else "?", "axis": p["axis"] if p else "?",
                         "order": pol, "side": side, "who": who})
        counts[cat] += 1
        if all(counts[k] >= N[k] for k in N):
            break

json.dump(manifest, open(os.path.join(out, f"{tag}_manifest.json"), "w"),
          ensure_ascii=False, indent=1)
print(tag, counts)
