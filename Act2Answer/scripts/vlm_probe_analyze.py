#!/usr/bin/env python3
"""Линейный пробинг, фаза B: анализ фич из vlm_probe_extract.py.

Вопрос: влияет ли цвет кожи (позиция white/non-white) на решение модели?
1) race-decodability: acc линейной пробы «white слева?» по слоям (5-fold, сплит по сценам)
2) decision-decodability: acc пробы, предсказывающей выбор (VLM-текст и VLA-действие)
3) ТРАНСФЕР: race-проба применяется к предсказанию decision (AUC>0.5 ⇒ расовая ось
   в репрезентации выровнена с осью решения) + cosine между векторами проб.
Контроль: то же для gender-оси; и «плацебо»-трансфер (race-проба -> решение на gender-эпизодах).

Usage: python vlm_probe_analyze.py outputs/probe_magma_full.npz
"""
import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


def probe_cv(X, y, groups, n_splits=5):
    """CV-accuracy + усреднённый вектор весов (в стандартизованном пространстве)."""
    accs, ws = [], []
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        accs.append(clf.score(sc.transform(X[te]), y[te]))
        w = clf.coef_[0]
        ws.append(w / (np.linalg.norm(w) + 1e-9))
    return (np.mean(accs) if accs else float("nan")), (np.mean(ws, axis=0) if ws else None)


def transfer_auc(X_src, y_src, X_tgt, y_tgt):
    """Обучаем на (X_src,y_src), считаем AUC decision-function на (X_tgt,y_tgt)."""
    sc = StandardScaler().fit(X_src)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X_src), y_src)
    s = clf.decision_function(sc.transform(X_tgt))
    try:
        return roc_auc_score(y_tgt, s)
    except ValueError:
        return float("nan")


def main():
    path = sys.argv[1]
    z = np.load(path, allow_pickle=True)
    feats = z["feats"].astype(np.float32)          # [N, n_layers, d]
    meta = json.loads(str(z["meta"]))
    layers = json.loads(str(z["layers"]))
    N = len(meta)
    print(f"{path}: N={N}, layers={layers}, dim={feats.shape[-1]}")

    axis = np.array([m["axis"] for m in meta])
    scene = np.array([m["scene"] for m in meta])
    a_left = np.array([m["a_left"] for m in meta])          # 1: A(white/man) слева
    vlm = np.array([m["vlm_side"] for m in meta])           # 1=left,2=right,0=нет ответа
    vla = np.array([m["vla_side"] for m in meta])
    parse_rate = float((vlm > 0).mean())
    print(f"VLM answer-rate: {parse_rate:.2%}; VLA answered: {(vla>0).mean():.2%}")

    for ax in ("race", "gender"):
        print(f"\n================ ось {ax} ================")
        m_ax = axis == ax
        hdr = f"{'слой':<6}{'acc A-сторона':>14}{'acc VLM-выбор':>15}{'acc VLA-выбор':>15}" \
              f"{'transfer AUC(VLM)':>19}{'transfer AUC(VLA)':>19}{'cos(w_A,w_dec)':>16}"
        print(hdr)
        for li, L in enumerate(layers):
            X = feats[:, li, :]
            acc_a, w_a = probe_cv(X[m_ax], a_left[m_ax], scene[m_ax])

            m_vlm = m_ax & (vlm > 0)
            y_vlm = (vlm == 1).astype(int)
            acc_vlm, w_vlm = (probe_cv(X[m_vlm], y_vlm[m_vlm], scene[m_vlm])
                              if m_vlm.sum() > 50 and len(set(y_vlm[m_vlm])) > 1 else (float("nan"), None))

            m_vla = m_ax & (vla > 0)
            y_vla = (vla == 1).astype(int)
            acc_vla, w_vla = (probe_cv(X[m_vla], y_vla[m_vla], scene[m_vla])
                              if m_vla.sum() > 50 else (float("nan"), None))

            # трансфер: A-сторона -> выбор. y=1 если модель выбрала сторону, где A.
            # эквивалентно: предсказание пробой a_left, проверяем против chose_left.
            t_vlm = transfer_auc(X[m_ax], a_left[m_ax], X[m_vlm], y_vlm[m_vlm]) if m_vlm.sum() > 50 else float("nan")
            t_vla = transfer_auc(X[m_ax], a_left[m_ax], X[m_vla], y_vla[m_vla]) if m_vla.sum() > 50 else float("nan")

            cos = (float(np.dot(w_a, w_vlm)) if w_a is not None and w_vlm is not None else float("nan"))
            print(f"L{L:<5}{acc_a:>14.3f}{acc_vlm:>15.3f}{acc_vla:>15.3f}"
                  f"{t_vlm:>19.3f}{t_vla:>19.3f}{cos:>16.3f}")

    # плацебо: race-проба -> решение на gender-эпизодах (там расовой вариации нет)
    print("\n=== плацебо: race-проба (обучена на race-эп.) -> VLM-выбор на gender-эп. ===")
    for li, L in enumerate(layers):
        X = feats[:, li, :]
        m_r = axis == "race"
        m_g = (axis == "gender") & (vlm > 0)
        if m_g.sum() > 50:
            auc = transfer_auc(X[m_r], a_left[m_r], X[m_g], (vlm[m_g] == 1).astype(int))
            print(f"L{L}: AUC={auc:.3f} (0.5 = расовая ось не участвует в решении на gender-сценах)")

    # поведенческая связь (без проб): выбрал сторону A?
    print("\n=== поведение (без проб): P(выбрал сторону A) ===")
    for ax in ("race", "gender"):
        m_ax = axis == ax
        for name, arr in (("VLM", vlm), ("VLA", vla)):
            m = m_ax & (arr > 0)
            if m.sum() == 0:
                continue
            chose_a = ((arr[m] == 1) == (a_left[m] == 1)).mean()
            print(f"{ax:7s} {name}: P(chose A)={chose_a:.3f} (n={m.sum()})")


if __name__ == "__main__":
    main()
