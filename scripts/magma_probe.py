#!/usr/bin/env python3
"""Линейный пробинг активаций Magma (из scripts/magma_extract_acts.py).

Для каждого слоя L обучаем логистическую регрессию (L2) предсказывать:
  - race   (white/black)
  - gender (man/woman)
  - trait  (профессия/статус — многоклассовая; отвечает: понимает ли модель, ЧТО за
            человек, из фото)
Метрика — accuracy со стратифицированной 5-fold CV (честный train/test split).
Строим кривую accuracy vs слой + chance-уровень.

Ключевой вопрос стереотипа: если И раса, И trait декодируются на средних слоях —
модель формирует представление о профессии из лица, и это можно сцепить с расой.
Здесь считаем сам факт декодируемости; направление стереотипа (сцепку) — отдельным
cross-probe при желании.

Использование:  python magma_probe.py outputs/magma_acts.npz [--out probe_report.json]
Зависимости: numpy, scikit-learn.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def probe_task(X_by_layer, y, n_layers, task, min_per_class=5):
    """Возвращает список (layer, acc, chance) по всем слоям для одной задачи."""
    mask = np.array([v is not None and v != "None" and v != "" for v in y])
    y = np.array(y)[mask]
    classes, counts = np.unique(y, return_counts=True)
    # выкидываем классы с < min_per_class примеров (для CV)
    keep = set(c for c, n in zip(classes, counts) if n >= min_per_class)
    keep_mask = np.array([v in keep for v in y])
    y = y[keep_mask]
    classes, counts = np.unique(y, return_counts=True)
    chance = counts.max() / counts.sum()  # majority-class baseline
    n_splits = min(5, counts.min())
    if n_splits < 2:
        return [], chance, dict(zip(classes.tolist(), counts.tolist()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    rows = []
    for L in range(n_layers):
        X = X_by_layer[L][mask][keep_mask]
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        )
        acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean()
        rows.append((L, float(acc)))
    return rows, float(chance), dict(zip(classes.tolist(), counts.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rep", choices=["mean", "last"], default="mean",
                    help="представление: mean-pool по последовательности или last-token")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    n_layers = int(d["n_layers"])
    has_last = f"last_L0" in d.files
    prefix = "last_L" if (args.rep == "last" and has_last) else "L"
    X_by_layer = [d[f"{prefix}{L}"] for L in range(n_layers)]
    labels = {k: d[k] for k in ("race", "gender", "trait", "category")}
    N = X_by_layer[0].shape[0]
    print(f"N={N} images, {n_layers} layers (emb + {n_layers-1} LLM layers), "
          f"hidden={X_by_layer[0].shape[1]}, rep={args.rep}\n")

    # trait-пробинг НЕВАЛИДЕН на этих данных: в occupations каждый trait = 4 картинки
    # (4 демографии одной сцены), т.е. "предскажи профессию" == "предскажи, какая это
    # фотка" — probe заучивает identity картинки, а не выводит профессию из лица (leak).
    # status и вовсе схлопнут в один label. Поэтому зондируем только race/gender —
    # они сбалансированы (100/100) и НЕ конфаундятся identity (демография проходит через
    # все сцены). Оставляем 'trait' закомментированным, чтобы не публиковать ложные 100%.
    report = {}
    tasks = ("race", "gender")
    for task in tasks:
        rows, chance, dist = probe_task(X_by_layer, labels[task], n_layers, task)
        report[task] = {"chance": chance, "class_dist": dist, "acc_by_layer": rows}
        if not rows:
            print(f"### {task}: недостаточно данных {dist}\n")
            continue
        accs = [a for _, a in rows]
        best_L = int(np.argmax(accs))
        print(f"### {task}  (chance={chance:.2f}, classes={dist})")
        print(f"    best: слой {best_L} acc={accs[best_L]:.3f}  |  emb(L0)={accs[0]:.3f}  last(L{n_layers-1})={accs[-1]:.3f}")
        # компактная кривая: каждые 4 слоя
        sparse = "  ".join(f"L{L}:{a:.2f}" for L, a in rows if L % 4 == 0 or L == n_layers - 1)
        print(f"    {sparse}\n")

    # интерпретация одной строкой
    def peak(t):
        r = report[t]["acc_by_layer"]
        return (max(a for _, a in r) if r else 0.0, report[t]["chance"])
    (ra, rc), (ga, gc) = peak("race"), peak("gender")
    print("=" * 60)
    print("ИТОГ (peak accuracy vs chance):")
    print(f"  race:   {ra:.2f} vs chance {rc:.2f}  -> {'декодируется' if ra-rc>0.15 else 'слабо/нет'}")
    print(f"  gender: {ga:.2f} vs chance {gc:.2f}  -> {'декодируется' if ga-gc>0.15 else 'слабо/нет'}")
    print("  trait:  ПРОПУЩЕН (невалиден: identity-leak, см. коммент в коде)")

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
