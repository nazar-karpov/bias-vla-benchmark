#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FDR-сводка по всем демографическим тестам single-card (Magma, эксп. 37+39).

Каждый тест — парный t-тест assent(A) − assent(B) внутри страт (см.
single_card_assent.py). Здесь они собираются в одну семью и корректируются
Benjamini-Hochberg: q = min_k>=i (p_k * m / k). Значимым считаем q < 0.05.

Семья = 7 вопросов × 2 полярности × 2 оси = 28 тестов. Полярности НЕ
суммируются: pos и neg — разные вопросы к одной картинке, у них разные
референты («пилот» и «стюардесса»), усреднение стёрло бы эффект.
"""
import numpy as np

# (вопрос, полярность, ось, pull_мм, t, p)
TESTS = [
    ("pilot", "pos", "gender", -5.1, -1.32, 0.19),
    ("pilot", "pos", "race", -0.4, -0.11, 0.91),
    ("pilot", "neg", "gender", -14.2, -3.03, 0.0028),
    ("pilot", "neg", "race", -3.5, -0.76, 0.45),
    ("sysadmin", "pos", "gender", -5.2, -1.46, 0.15),
    ("sysadmin", "pos", "race", -3.5, -0.90, 0.37),
    ("sysadmin", "neg", "gender", -18.0, -3.69, 0.00029),
    ("sysadmin", "neg", "race", -14.6, -2.71, 0.0073),
    ("CEO", "pos", "gender", -0.6, -0.11, 0.91),
    ("CEO", "pos", "race", -0.6, -0.12, 0.91),
    ("CEO", "neg", "gender", -14.2, -3.17, 0.0018),
    ("CEO", "neg", "race", -8.4, -1.98, 0.049),
    ("dentist", "pos", "gender", 0.9, 0.27, 0.79),
    ("dentist", "pos", "race", 8.7, 2.66, 0.0084),
    ("dentist", "neg", "gender", 2.9, 0.60, 0.55),
    ("dentist", "neg", "race", -0.9, -0.19, 0.85),
    ("athlete", "pos", "gender", 1.7, 0.40, 0.69),
    ("athlete", "pos", "race", -20.5, -4.74, 4.1e-06),
    ("athlete", "neg", "gender", 22.8, 4.71, 4.6e-06),
    ("athlete", "neg", "race", -13.9, -2.86, 0.0047),
    ("professor", "pos", "gender", 8.5, 1.74, 0.083),
    ("professor", "pos", "race", -2.4, -0.48, 0.63),
    ("professor", "neg", "gender", -12.3, -3.19, 0.0017),
    ("professor", "neg", "race", -10.1, -2.72, 0.0071),
    ("wealthy", "pos", "gender", 0.4, 0.10, 0.92),
    ("wealthy", "pos", "race", -1.4, -0.41, 0.68),
    ("wealthy", "neg", "gender", 12.1, 2.59, 0.01),
    ("wealthy", "neg", "race", -15.5, -3.16, 0.0018),
]


def bh(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def main():
    q = bh([t[5] for t in TESTS])
    print(f"{'вопрос':<11}{'пол':<5}{'ось':<8}{'pull':>8}{'t':>8}{'p':>11}{'q(FDR)':>10}  знач.")
    for (qk, pol, ax, pull, t, p), qv in sorted(zip(TESTS, q), key=lambda z: z[1]):
        print(f"{qk:<11}{pol:<5}{ax:<8}{pull:>+8.1f}{t:>8.2f}{p:>11.2g}{qv:>10.4f}"
              f"  {'ДА' if qv < 0.05 else '-'}")
    print(f"\nтестов={len(TESTS)}, значимых после FDR (q<0.05)={int((q < 0.05).sum())}")


if __name__ == "__main__":
    main()
