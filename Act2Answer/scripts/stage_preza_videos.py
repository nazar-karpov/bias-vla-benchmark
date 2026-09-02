"""Стейджинг видео для презы:
1) magma confirm noswap: на каждый (вопрос x ось) один hard-эпизод (кубик идеально на
   плитке, узкая зона), все РАЗНЫЕ сцены; + 2 фейла (кубик упал/не донёс).
2) ep703 (money, жен-пара, раса) у всех моделей h100 — для сравнительной сетки.
Копирует в ~/preza_stage с говорящими именами."""
import glob as g, json, os, re, shutil, yaml

HALF = 0.0715
R = "/workspace/moskalenko/bias-vla-benchmark-main"
PAIRS = {e["index"]: e for e in json.load(open(
    f"{R}/Act2Answer/ManiSkill/mani_skill/assets/carrot/pairs_choice_vla_confirm/pairs.json"))}
OUT = os.path.expanduser("~/preza_stage")
os.makedirs(OUT, exist_ok=True)

def coords(e):
    if "cube_fy" not in e or float(e.get("cube_fz", 1.0)) < 0.8:
        return None
    cy = float(e["cube_fy"]); cx = float(e.get("cube_fx", 9.0))
    if abs(cy) > 0.5:
        return None
    bL = float(e.get("boardL_y", -0.155)); bR = float(e.get("boardR_y", 0.155))
    bLx = float(e.get("boardL_x", -0.25)); bRx = float(e.get("boardR_x", -0.25))
    dL = ((cy - bL) ** 2 + (cx - bLx) ** 2) ** 0.5
    dR = ((cy - bR) ** 2 + (cx - bRx) ** 2) ** 0.5
    side, dist, ty, tx = ("L", dL, bL, bLx) if dL <= dR else ("R", dR, bR, bRx)
    hard = abs(cy - ty) <= HALF + 0.01 and abs(cx - tx) <= HALF + 0.01
    return side, dist, hard

# --- 1. magma: hard на каждый вопрос x ось, разные сцены ---
best = {}   # (question, axis) -> (dist, path, name)
fails = []
for f in sorted(g.glob(f"{R}/Act2Answer/outputs/confirm-mid-magma-ALL-noswap-*/glob/vis_0_test/stats.yaml")):
    st = int(re.search(r"-s(\d+)/", f).group(1))
    d = os.path.dirname(f)
    try:
        li = yaml.safe_load(open(f))["last_info"]
    except Exception:
        continue
    for i, e in li.items():
        if not isinstance(e, dict):
            continue
        p = PAIRS.get(st + int(i))
        if not p:
            continue
        c = coords(e)
        vids = g.glob(os.path.join(d, f"video_{int(i)}-s_*.mp4"))
        if not vids:
            continue
        q = p["question"].replace("Put cube on ", "").replace("the ", "").strip().replace(" ", "_")
        scene = p["left"].split("__")[1]
        if c is None:
            if len(fails) < 3 and float(e.get("cube_fz", 1.0)) < 0.8:
                fails.append((vids[0], f"magma_FAIL_drop__{q}__{scene}__ep{st+int(i)}.mp4"))
            continue
        side, dist, hard = c
        if not hard:
            continue
        who = (p["left"] if side == "L" else p["right"]).split("__")[-1]
        key = (q, p["axis"])
        used_scenes = {v[3] for k, v in best.items() if k != key}
        if key not in best or (dist < best[key][0] and scene not in used_scenes):
            if scene in used_scenes:
                continue
            best[key] = (dist, vids[0],
                         f"magma_{q}__{p['axis']}__{scene}__{who}__d{dist*1000:.0f}mm__ep{st+int(i)}.mp4",
                         scene)

for k, (dist, path, name, scene) in sorted(best.items()):
    shutil.copy(path, os.path.join(OUT, name))
    print("HARD", k, name)
for path, name in fails:
    shutil.copy(path, os.path.join(OUT, name))
    print("FAIL", name)

# --- 2. ep703 у всех моделей (noswap) ---
RUNS = {
    "gr00t": "confirm-gr00t-noswap-s700",
    "magma": "confirm-mid-magma-ALL-noswap-s700",
    "rldx1": "confirm-rldx-FULL-noswap-s680",
    "rldx2": "confirm-rldx2-ALL-noswap-s700",
    "xiaomi": "confirm-xiaomi-ALL-noswap-s700",
    "xvla": "confirm-xvla-ALL-noswap-s700",
}
for tag, run in RUNS.items():
    st = int(re.search(r"-s(\d+)$", run).group(1))
    i = 703 - st
    vids = g.glob(f"{R}/Act2Answer/outputs/{run}/glob/vis_0_test/video_{i}-s_*.mp4")
    if vids:
        shutil.copy(vids[0], os.path.join(OUT, f"cmp703_{tag}.mp4"))
        print("CMP", tag)
    else:
        print("CMP-MISS", tag)
print("done ->", OUT)
