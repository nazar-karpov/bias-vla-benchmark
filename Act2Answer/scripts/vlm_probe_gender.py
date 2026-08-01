#!/usr/bin/env python3
"""Гендер-пробинг по ячейкам вопрос×полярность.

Гендер-проба («мужчина слева?») обучается на gender-эпизодах ДРУГИХ вопросов
(и с непересекающимися сценами — без утечки), затем её направление применяется
к решениям модели на целевой ячейке: AUC>0.5 — решение тянется к мужчине,
AUC<0.5 — к женщине. Отдельно для текстового выбора VLM и действия робота (VLA).

Usage: python vlm_probe_gender.py outputs/probe_magma_full.npz [--layer-idx 2]
"""
import argparse, json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer-idx", type=int, default=-1, help="-1 = все слои")
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    feats = z["feats"].astype(np.float32)
    meta = json.loads(str(z["meta"]))
    layers = json.loads(str(z["layers"]))

    axis = np.array([m["axis"] for m in meta])
    qkey = np.array([m["qkey"] for m in meta])
    pol = np.array([m["polarity"] for m in meta])
    scene = np.array([m["scene"] for m in meta])
    a_left = np.array([m["a_left"] for m in meta])       # 1: A (мужчина на gender-оси) слева
    vlm = np.array([m["vlm_side"] for m in meta])
    vla = np.array([m["vla_side"] for m in meta])

    g = axis == "gender"
    qs = ["boss", "wealthy", "skier", "pilot"]
    li_list = range(len(layers)) if a.layer_idx < 0 else [a.layer_idx]

    for li in li_list:
        X = feats[:, li, :]
        print(f"\n========== слой L{layers[li]} ==========")
        print(f"{'ячейка':<16}{'n':>5}{'AUC гендер-декод':>18}{'AUC->VLM выбор':>16}{'AUC->VLA выбор':>16}"
              f"  (AUC<0.5 = тянет к женщине)")
        for q in qs:
            for p in ("pos", "neg"):
                m_t = g & (qkey == q) & (pol == p)
                if m_t.sum() < 40:
                    continue
                # фолды по сценам: train = другие вопросы × train-сцены,
                # eval = целевая ячейка × eval-сцены (сцены общие для всех вопросов)
                uniq = np.unique(scene[m_t])
                rng = np.random.RandomState(0)
                order = rng.permutation(len(uniq))
                folds = np.array_split(order, 5)
                y_dec_all, s_dec_all = [], []
                per_arr = {0: ([], []), 1: ([], [])}   # vlm, vla: (y, score)
                for f in folds:
                    ev_scenes = uniq[f]
                    m_ev = m_t & np.isin(scene, ev_scenes)
                    m_tr = g & (qkey != q) & ~np.isin(scene, ev_scenes)
                    if m_ev.sum() == 0 or m_tr.sum() < 100:
                        continue
                    sc_ = StandardScaler().fit(X[m_tr])
                    clf = LogisticRegression(max_iter=2000, C=1.0).fit(
                        sc_.transform(X[m_tr]), a_left[m_tr])
                    s_ev = clf.decision_function(sc_.transform(X[m_ev]))
                    y_dec_all.extend(a_left[m_ev]); s_dec_all.extend(s_ev)
                    for k, arr in enumerate((vlm, vla)):
                        mm = m_ev & (arr > 0)
                        if mm.sum() == 0 or len(set(arr[mm])) < 2:
                            continue
                        s2 = clf.decision_function(sc_.transform(X[mm]))
                        per_arr[k][0].extend((arr[mm] == 1).astype(int))
                        per_arr[k][1].extend(s2)
                try:
                    auc_dec = roc_auc_score(y_dec_all, s_dec_all)
                except ValueError:
                    auc_dec = float("nan")
                out = [auc_dec]
                for k in (0, 1):
                    y_, s_ = per_arr[k]
                    if len(y_) < 30 or len(set(y_)) < 2:
                        out.append(float("nan")); continue
                    out.append(roc_auc_score(y_, s_))
                print(f"{q+'/'+p:<16}{m_t.sum():>5}{out[0]:>18.3f}{out[1]:>16.3f}{out[2]:>16.3f}")

        # поведение по ячейкам для сверки
        print(f"\n{'ячейка':<16}{'P(A) VLM':>10}{'P(A) VLA':>10}   (доля выбора мужчины)")
        for q in qs:
            for p in ("pos", "neg"):
                m_t = g & (qkey == q) & (pol == p)
                row = [f"{q+'/'+p:<16}"]
                for arr in (vlm, vla):
                    mm = m_t & (arr > 0)
                    if mm.sum() == 0:
                        row.append(f"{'—':>10}"); continue
                    pa = ((arr[mm] == 1) == (a_left[mm] == 1)).mean()
                    row.append(f"{pa:>10.3f}")
                print("".join(row))


if __name__ == "__main__":
    main()
