#!/usr/bin/env python3
"""Logit lens по слоям Magma-8B на confirm-кадрах.

Для интересных ячеек (стюардесса, pilot, boss, mugger) берём N эпизодов (noswap),
hidden state последней позиции на КАЖДОМ слое -> final_norm -> lm_head -> топ-токены.
Смотрим: (1) на каком слое кристаллизуется Left/Right; (2) вероятности гендерных
слов по слоям; (3) проекция вектора гендер-пробы в словарь.

Usage: python vlm_logit_lens.py --n-per-cell 8 [--no-question]
"""
import argparse, glob, json, os, re, sys
import numpy as np
import torch
import cv2
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from magma_vlm_qa import load_magma, magma_build_inputs

A2A = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = os.path.join(A2A, "ManiSkill/mani_skill/assets/carrot/pairs_choice_vla_confirm/pairs.json")
OUT_DIR = os.path.join(A2A, "outputs")

CELLS = [("pilot", "neg"), ("pilot", "pos"), ("boss", "pos"), ("skier", "neg")]
GENDER_WORDS = ["Left", "Right", " woman", " man", " she", " he", " female", " male",
                " Woman", " Man", "woman", "man"]


def first_frame(path):
    cap = cv2.VideoCapture(path); ok, fr = cap.read(); cap.release()
    return Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)) if ok else None


def find_videos():
    out = {}
    for st in glob.glob(os.path.join(OUT_DIR, "confirm-internvla-FULL-noswap-s*",
                                     "glob", "vis_0_test", "stats.yaml")):
        d = os.path.dirname(st)
        m = re.search(r"-s(\d+)$", os.path.dirname(os.path.dirname(d)))
        if not m: continue
        start = int(m.group(1))
        li = (yaml.safe_load(open(st)) or {}).get("last_info") or {}
        for idx in li:
            v = glob.glob(os.path.join(d, f"video_{int(idx)}-s_*.mp4"))
            if v: out[start + int(idx)] = v[0]
    return out


def get_head(model):
    """final norm + lm_head (Magma: model.language_model.{model.norm, lm_head})."""
    lm = getattr(model, "language_model", None)
    if lm is not None:
        return lm.model.norm, lm.lm_head
    lm_head = getattr(model, "lm_head", None)
    norm = getattr(getattr(model, "model", model), "norm", None)
    assert lm_head is not None and norm is not None, "не нашёл norm/lm_head"
    return norm, lm_head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cell", type=int, default=8)
    ap.add_argument("--no-question", action="store_true",
                    help="только инструкция VLA, без хвоста про Left/Right")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    pairs = json.load(open(PAIRS))
    vids = find_videos()
    model, processor = load_magma(a.device)
    norm, lm_head = get_head(model)
    tok = processor.tokenizer
    gw_ids = {}
    for w in GENDER_WORDS:
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            gw_ids[w] = ids[0]
    print("отслеживаемые токены:", list(gw_ids))

    # подбор эпизодов по ячейкам (gender-ось)
    chosen = {c: [] for c in CELLS}
    for ep, meta in enumerate(pairs):
        key = (meta.get("qkey"), meta.get("polarity"))
        if key in chosen and meta.get("axis") == "gender" and ep in vids \
                and len(chosen[key]) < a.n_per_cell:
            chosen[key].append(ep)

    per_cell = {}
    for cell, eps in chosen.items():
        rows = []
        for ep in eps:
            meta = pairs[ep]
            img = first_frame(vids[ep])
            if img is None: continue
            q = meta["question"]
            if not a.no_question:
                q += " Which tile is correct, left or right? Answer with one word: Left or Right."
            inputs = magma_build_inputs(model, processor, img, q, a.device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            layer_info = []
            for L, h in enumerate(out.hidden_states):
                v = h[0, -1, :]
                logits = lm_head(norm(v.unsqueeze(0))).float().softmax(-1)[0]
                topv, topi = logits.topk(3)
                top = [(tok.decode([i]).strip(), round(float(p), 3))
                       for i, p in zip(topi.tolist(), topv.tolist())]
                gw = {w: float(logits[i]) for w, i in gw_ids.items()}
                layer_info.append((top, gw))
            rows.append((ep, meta["left"].split("__")[-1], meta["right"].split("__")[-1], layer_info))
        per_cell[cell] = rows

    n_layers = len(next(iter(per_cell.values()))[0][3])
    print(f"\nслоёв (incl. emb): {n_layers}")

    # 1) качественно: один пример стюардессы по слоям
    cell = ("pilot", "neg")
    ep, l_dem, r_dem, li = per_cell[cell][0]
    print(f"\n=== ПРИМЕР {cell} ep{ep} (слева {l_dem}, справа {r_dem}): топ-3 по слоям ===")
    for L in range(0, n_layers, 2):
        print(f"L{L:>2}: {li[L][0]}")

    # 2b) семантическое решение: P('left')-P('right') (строчные) по слоям,
    #     в знаке "к женщине" (+1 = сигнал в сторону женской плитки)
    low_ids = {}
    for w in ("left", " left", "Left", " Left"):
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1: low_ids.setdefault("L", []).append(ids[0])
    for w in ("right", " right", "Right", " Right"):
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1: low_ids.setdefault("R", []).append(ids[0])
    print("\n=== СЕМАНТИЧЕСКИЙ СИГНАЛ ПО СЛОЯМ: (P(left)-P(right)) со знаком 'к женщине' ===")
    for cell, rows in per_cell.items():
        curves = []
        for ep, ld, rd, li in rows:
            woman_left = 1 if ("woman" in ld) else -1   # +1: женщина слева
            c = []
            for L in range(n_layers):
                # пересчёт: полные софтмаксы уже не сохранены, но gw имеет только
                # словарь gw_ids; поэтому используем top-подход: из top-3 берём
                # массу left/right если попали. Грубее, но показательно.
                top = li[L][0]
                pl = sum(p for t, p in top if t.lower() == "left")
                pr = sum(p for t, p in top if t.lower() == "right")
                c.append(woman_left * (pl - pr))
            curves.append(c)
        m = np.mean(curves, axis=0)
        peaks = sorted(range(n_layers), key=lambda L: -abs(m[L]))[:3]
        print(f"{cell}: L22={m[22]:+.3f} L24={m[24]:+.3f} L26={m[26]:+.3f} "
              f"L28={m[28]:+.3f} L30={m[30]:+.3f} L32={m[n_layers-1]:+.3f} | пики: {[(L, round(m[L],3)) for L in peaks]}")

    # 2) агрегаты: слой кристаллизации Left/Right и P(gender-слова)
    print("\n=== АГРЕГАТЫ по ячейкам ===")
    for cell, rows in per_cell.items():
        first_lr = []
        gw_curves = {w: np.zeros(n_layers) for w in gw_ids}
        for ep, ld, rd, li in rows:
            f = next((L for L in range(n_layers)
                      if li[L][0][0][0] in ("Left", "Right")), None)
            if f is not None: first_lr.append(f)
            for L in range(n_layers):
                for w in gw_ids: gw_curves[w][L] += li[L][1][w]
        for w in gw_curves: gw_curves[w] /= max(len(rows), 1)
        print(f"\n--- {cell} (n={len(rows)}) ---")
        print(f"слой, где топ-1 впервые Left/Right: {first_lr}")
        interesting = [w for w in gw_curves if gw_curves[w].max() > 1e-3]
        for w in interesting:
            c = gw_curves[w]
            pk = int(np.argmax(c))
            print(f"  P('{w}'): пик {c[pk]:.3f} на L{pk}; L16={c[16]:.4f} L24={c[24]:.4f} L32={c[min(32, n_layers-1)]:.4f}")

    # 3) проекция гендер-пробы в словарь
    print("\n=== ГЕНДЕР-ПРОБА -> СЛОВАРЬ (что значит эта ось) ===")
    z = np.load(os.path.join(OUT_DIR, "probe_magma_full.npz"), allow_pickle=True)
    feats = z["feats"].astype(np.float32); meta_all = json.loads(str(z["meta"]))
    layers_saved = json.loads(str(z["layers"]))
    ax = np.array([m["axis"] for m in meta_all]); al = np.array([m["a_left"] for m in meta_all])
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    for li_idx, Lname in enumerate(layers_saved):
        X = feats[ax == "gender", li_idx, :]
        y = al[ax == "gender"]
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
        w_raw = torch.tensor(clf.coef_[0] / sc.scale_, dtype=torch.float16, device=a.device)
        logits = lm_head(norm(w_raw.unsqueeze(0))).float()[0]
        topi = logits.topk(8).indices.tolist()
        boti = (-logits).topk(8).indices.tolist()
        print(f"L{Lname}: +ось(муж слева): {[tok.decode([i]).strip() for i in topi]}")
        print(f"      −ось(жен слева): {[tok.decode([i]).strip() for i in boti]}")


if __name__ == "__main__":
    main()
