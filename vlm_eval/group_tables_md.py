#!/usr/bin/env python3
"""Таблицы «доля выбора группы» в разрезе VLA-семейств, готовые для вставки.

Строки — как в основных таблицах бенчмарка: VLA-семейство и конкретная модель,
отдельно вариант 1 (исходная VLM) и вариант 2 (сам VLA-чекпойнт). Колонки —
десять демографических групп плюс размах и хи-квадрат.

Модели, которые не прогонялись или для которых прогон невозможен, остаются
с прочерками: строка в таблице всё равно нужна, иначе не видно, чего не хватает.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from group_rates import binom_p, chi2_sf, collect, stars  # noqa: E402

# порядок колонок: азиаты, чёрные, латино, белые, ближневосточные; в каждой паре м/ж
COLS = [("asian", "m"), ("asian", "w"), ("black", "m"), ("black", "w"),
        ("latino", "m"), ("latino", "w"), ("white", "m"), ("white", "w"),
        ("ME", "m"), ("ME", "w")]
HEAD = ["As_m", "As_w", "Bl_m", "Bl_w", "La_m", "La_w", "Wh_m", "Wh_w", "ME_m", "ME_w"]

# как называются группы в двух датасетах
ALIAS = {
    ("asian", "m"): ["Asian_man", "asian_male"], ("asian", "w"): ["Asian_woman", "asian_female"],
    ("black", "m"): ["Black_man", "black_male"], ("black", "w"): ["Black_woman", "black_female"],
    ("latino", "m"): ["Latino_man", "latino_male"], ("latino", "w"): ["Latino_woman", "latino_female"],
    ("white", "m"): ["White_man", "white_male"], ("white", "w"): ["White_woman", "white_female"],
    ("ME", "m"): ["Middle_Eastern_man", "ME_male"], ("ME", "w"): ["Middle_Eastern_woman", "ME_female"],
}

VARIANT1 = [
    ("Magma", "отдельной базы нет", None),
    ("InternVLA-M1", "Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"),
    ("Xiaomi-Robotics-0", "Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-4B-Instruct"),
    ("GR00T-N1.7", "Cosmos-Reason2-2B", "nvidia/Cosmos-Reason2-2B"),
    ("RLDX-1", "RLDX-1-VLM", "RLWRLD/RLDX-1-VLM"),
    ("MolmoAct2", "Molmo2-ER", "allenai/Molmo2-ER"),
    ("SpatialVLA", "paligemma2-3b-pt-224", "google/paligemma2-3b-pt-224"),
    ("SpatialVLA", "paligemma2-3b-mix-224 *(прокси)*", "google/paligemma2-3b-mix-224"),
    ("pi0", "paligemma-3b-pt-224", "google/paligemma-3b-pt-224"),
    ("pi0", "paligemma-3b-mix-224 *(прокси)*", "google/paligemma-3b-mix-224"),
    ("pi0.5", "paligemma-3b-pt-224", "google/paligemma-3b-pt-224"),
    ("pi0.5", "paligemma-3b-mix-224 *(прокси)*", "google/paligemma-3b-mix-224"),
    ("OpenVLA", "prism-dinosiglip-224px+7b", None),
    ("X-VLA", "Florence-2-large", "microsoft/Florence-2-large"),
    ("X-VLA", "Florence-2-large-ft *(сиблинг)*", "microsoft/Florence-2-large-ft"),
]

VARIANT2 = [
    ("Magma", "Magma-8B", "microsoft/Magma-8B"),
    ("InternVLA-M1", "InternVLA-M1-Pretrain-RT-1-Bridge", "vla:internvla"),
    ("Xiaomi-Robotics-0", "Xiaomi-Robotics-0-Pretrain", "vla:xiaomi_pretrain"),
    ("Xiaomi-Robotics-0", "Xiaomi-Robotics-0-SimplerEnv-WidowX", None),
    ("GR00T-N1.7", "GR00T-N1.7-SimplerEnv-Bridge", None),
    ("RLDX-1", "RLDX-1-FT-SIMPLER-WIDOWX", None),
    ("MolmoAct2", "MolmoAct2-SO100_101", "allenai/MolmoAct2-SO100_101"),
    ("SpatialVLA", "spatialvla-4b-224-pt", "IPEC-COMMUNITY/spatialvla-4b-224-pt"),
    ("pi0", "INTACT-pi0-finetune-bridge", None),
    ("pi0.5", "pi05_widowx", None),
    ("OpenVLA", "openvla-7b", "openvla/openvla-7b"),
    ("X-VLA", "X-VLA-WidowX", None),
]


def rates_for(results_dir, dataset, cond, model, attribute=None):
    tag = model.replace("/", "_")
    path = f"{results_dir}/{tag}__{dataset}__{cond}.json"
    if not os.path.exists(path):
        return None
    acc = collect(path, attribute)
    if not acc:
        return None
    out = {}
    for key in COLS:
        for name in ALIAS[key]:
            if name in acc:
                s, n, w = acc[name]
                out[key] = (s / n, binom_p(w, n), n, w)
                break
    return out


def render(results_dir, dataset, cond, attribute, title, rows):
    print(f"\n### {title}\n")
    print("| VLA-семейство | модель | " + " | ".join(HEAD) + " | размах | хи² p |")
    print("|---|---|" + "---|" * (len(HEAD) + 2))
    for family, label, model in rows:
        if model is None:
            print(f"| {family} | {label} | " + " | ".join(["—"] * len(HEAD)) + " | — | — |")
            continue
        r = rates_for(results_dir, dataset, cond, model, attribute)
        if not r:
            print(f"| {family} | {label} | " + " | ".join(["—"] * len(HEAD)) + " | — | — |")
            continue
        cells, vals, chi = [], [], 0.0
        for key in COLS:
            if key not in r:
                cells.append("—")
                continue
            rate, p, n, w = r[key]
            vals.append(rate)
            mark = "**" if p < .05 else ""
            cells.append(f"{mark}{rate * 100:.0f}{mark}")
            # хи-квадрат считаем по бинаризованным победам, как и биномиальный тест
            # по группе; на массе вероятности он был бы другой статистикой
            chi += (w - n / 2) ** 2 / (n / 2) + ((n - w) - n / 2) ** 2 / (n / 2)
        spread = (max(vals) - min(vals)) * 100 if len(vals) > 1 else 0
        p = chi2_sf(chi, max(1, len(vals) - 1))
        print(f"| {family} | {label} | " + " | ".join(cells) +
              f" | {spread:.1f} | {p:.1e}{stars(p).strip()} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_manifests"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cond", default="pair_ab")
    ap.add_argument("--attribute", default=None)
    a = ap.parse_args()
    suf = f" / {a.attribute}" if a.attribute else ""
    render(a.results, a.dataset, a.cond, a.attribute,
           f"{a.dataset}{suf} — вариант 1, исходная VLM", VARIANT1)
    render(a.results, a.dataset, a.cond, a.attribute,
           f"{a.dataset}{suf} — вариант 2, сам VLA-чекпойнт", VARIANT2)


if __name__ == "__main__":
    main()
