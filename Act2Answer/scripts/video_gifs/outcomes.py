"""Исходы перепрогона демо-кардсетов (record_gif_demos.sh) — по каждому эпизоду и по парам порядков.

Ответ модели — как в статье: куб в момент отпускания (последний шаг «в схвате») лежит в soft-зоне
плитки = половина плитки по model_db (0.066 м) + 3 см = ±0.096 м от её центра по x и y. Плюс strict-
сторона в конце эпизода (chosen_side из stats.yaml), дрейф плиток до отпускания и кадр отпускания в видео.

  python outcomes.py [--outputs <Act2Answer/outputs>]
Пишет outcomes.csv (эпизоды) и outcomes_pairs.csv (пара noswap/swap: одна ли картинка выбрана).
"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SOFT = 0.096
# прогон -> (модель, файл кандидатов); индексы кардсета gifdemo_* = порядок строк в кандидатах этой модели
RUNS = {"magma": ("magma", "candidates_w1.json"), "internvla": ("internvla", "candidates_w1.json"),
        "magma_w2": ("magma", "candidates_w2.json"), "magma_w3": ("magma", "candidates_w3.json"),
        "magma_w4": ("magma", "candidates_w4.json"), "internvla_w5": ("internvla", "candidates_w5.json")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=os.path.join(HERE, "..", "..", "outputs"))
    a = ap.parse_args()
    rows = []
    for run, (vla, cand) in RUNS.items():
        sp = [r for r in json.load(open(os.path.join(HERE, cand), encoding="utf-8")) if r["vla"] == vla]
        for order in ("noswap", "swap"):
            for d in sorted(glob.glob(os.path.join(a.outputs, f"gif-{run}-{order}-s*"))):
                base = os.path.join(d, "glob", "vis_0_test")
                if not os.path.exists(os.path.join(base, "stats.yaml")):
                    continue
                z = np.load(os.path.join(base, "traj.npz"))
                st = yaml.safe_load(open(os.path.join(base, "stats.yaml")))
                t0 = int(z["action_t0"])
                for k, eid in enumerate(z["ep_ids"]):
                    eid = int(eid); s = sp[eid]
                    c, g = z["cube_xyz"][k], z["grasped"][k]
                    bl, br = z["boardL_xy"][k], z["boardR_xy"][k]
                    idx = np.where(g)[0]; rel = int(idx[-1]) if len(idx) else -1
                    cx, cy = (float(c[rel, 0]), float(c[rel, 1])) if rel >= 0 else (np.nan, np.nan)
                    inL = rel >= 0 and abs(cx - bl[rel, 0]) <= SOFT and abs(cy - bl[rel, 1]) <= SOFT
                    inR = rel >= 0 and abs(cx - br[rel, 0]) <= SOFT and abs(cy - br[rel, 1]) <= SOFT
                    side = 1 if inL else (2 if inR else 0)
                    # noswap: слева картинка 1; swap: слева картинка 2
                    pick = 0 if side == 0 else ((side if order == "noswap" else 3 - side))
                    rr = min(len(bl) - 1, rel + 4) if rel >= 0 else len(bl) - 1
                    drift_rel = float(max(np.abs(bl[:rr + 1] - bl[0]).max(), np.abs(br[:rr + 1] - br[0]).max()))
                    vids = glob.glob(os.path.join(base, f"video_{k}-s_*.mp4"))
                    rows.append(dict(run=run, vla=vla, order=order, idx=eid, key=s["key"], target=s["target"],
                                     a1=s["a1"], a2=s["a2"], img1=s["img1"], img2=s["img2"], cs=s["cs"], src_i=s["i"],
                                     rel_frame=rel - t0 + 1 if rel >= 0 else -1, x_rel=cx, y_rel=cy, side_rel=side,
                                     pick_img=pick, pick_grp=(s["a1"] if pick == 1 else s["a2"]) if pick else "",
                                     strict=int(st["last_info"][k].get("chosen_side", 0)),
                                     drift_rel_mm=drift_rel * 1000, video=os.path.abspath(vids[0]) if vids else ""))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "outcomes.csv"), index=False)
    w = df.pivot_table(index=["run", "vla", "key", "idx", "target"], columns="order",
                       values=["pick_grp", "strict", "rel_frame", "drift_rel_mm"], aggfunc="first")
    w.columns = [f"{x}_{y}" for x, y in w.columns]
    w = w.reset_index()
    w["both_target"] = (w.pick_grp_noswap == w.target) & (w.pick_grp_swap == w.target)
    w["both_strict"] = (w.strict_noswap > 0) & (w.strict_swap > 0)
    w.to_csv(os.path.join(HERE, "outcomes_pairs.csv"), index=False)
    print(w.groupby(["run", "key"]).agg(pairs=("idx", "size"), both_target=("both_target", "sum"),
                                        clean=("both_strict", lambda s: int((s & w.loc[s.index, "both_target"]).sum()))))


if __name__ == "__main__":
    main()
