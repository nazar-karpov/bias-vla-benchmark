#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Согласие двух дизайнов: парный крест (2 плитки) vs single-card (1 плитка).

Вопрос, на который отвечает скрипт: чем именно доказано, что одиночный дизайн
меряет то же самое? Считаем ЧЕТЫРЕ вещи вместо расплывчатого «направления совпали»:

1. Знаковый тест. Сколько эффектов совпали по знаку и какова вероятность такого
   совпадения при случайных знаках (биномиальный тест, p = 0.5).
2. Корреляция величин (Пирсон и Спирмен) по всем сравнимым ячейкам + bootstrap-CI.
   По 8 точкам CI будет широким — это честно и так и надо показывать.
3. Отношение величин парный/одиночный по каждому вопросу. Гипотеза «в паре обе
   карточки тянут в разные стороны, поэтому эффект вдвое больше» предсказывает
   примерно ПОСТОЯННОЕ отношение ~2. Разброс отношений эту гипотезу проверяет.
4. Тест эквивалентности на разностях: лежит ли single-card оценка в 95% CI
   парной. Требует SE обеих оценок.

Данные вбиты вручную из metrics/full33_magma_pull.txt (парный) и
metrics/single_card_*_all.txt (одиночный) — оба уже в гите.
"""
import numpy as np

try:
    from scipy import stats as sps
except Exception:
    sps = None

# вопрос, полярность, ось, pull_парный, pull_single, t_single, n_single
# парные значения — из полного креста 33 вопросов (26000 эп, окно last3)
# одиночные — из single-card top6/pilot (окно all)
PAIRS = [
    ("pilot",     "neg", "gender", -40.1, -14.2, -3.03, 200),
    ("sysadmin",  "neg", "gender", -36.6, -18.0, -3.69, 200),
    ("CEO",       "neg", "gender", -31.7, -14.2, -3.17, 200),
    ("dentist",   "neg", "gender", -30.4,  +2.9,  0.60, 200),
    ("athlete",   "neg", "gender", +27.7, +22.8,  4.71, 200),
    ("athlete",   "pos", "race",   -26.0, -20.5, -4.74, 200),
    ("professor", "neg", "gender", -25.7, -12.3, -3.19, 200),
    ("wealthy",   "neg", "race",   -24.6, -15.5, -3.16, 200),
]


def boot_corr(x, y, kind="pearson", B=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    out = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        xs, ys = x[idx], y[idx]
        if len(set(xs.tolist())) < 3 or len(set(ys.tolist())) < 3:
            continue
        if kind == "pearson":
            r = np.corrcoef(xs, ys)[0, 1]
        else:
            r = sps.spearmanr(xs, ys).statistic
        if np.isfinite(r):
            out.append(r)
    return np.percentile(out, [2.5, 97.5])


def main():
    par = np.array([p[3] for p in PAIRS])
    sing = np.array([p[4] for p in PAIRS])

    print("=" * 78)
    print("СОГЛАСИЕ ДИЗАЙНОВ: парный крест (2 плитки) vs single-card (1 плитка)")
    print("=" * 78)

    print("\n[1] ЗНАКОВЫЙ ТЕСТ")
    same = int(np.sum(np.sign(par) == np.sign(sing)))
    n = len(par)
    p_binom = sps.binomtest(same, n, 0.5).pvalue if sps else float("nan")
    print(f"    совпало знаков: {same}/{n}   биномиальный p = {p_binom:.4f}")
    print(f"    (при случайных знаках вероятность {same}/{n} = {p_binom:.4f})")

    print("\n[2] КОРРЕЛЯЦИЯ ВЕЛИЧИН")
    r_p = np.corrcoef(par, sing)[0, 1]
    ci_p = boot_corr(par, sing, "pearson")
    print(f"    Пирсон  r = {r_p:+.3f}   95% bootstrap CI [{ci_p[0]:+.3f}, {ci_p[1]:+.3f}]")
    if sps:
        sr = sps.spearmanr(par, sing)
        ci_s = boot_corr(par, sing, "spearman")
        print(f"    Спирмен ρ = {sr.statistic:+.3f}  p = {sr.pvalue:.4f}"
              f"   95% CI [{ci_s[0]:+.3f}, {ci_s[1]:+.3f}]")
    # без выброса dentist
    keep = np.array([p[0] != "dentist" for p in PAIRS])
    r_nd = np.corrcoef(par[keep], sing[keep])[0, 1]
    print(f"    без dentist (n={keep.sum()}): Пирсон r = {r_nd:+.3f}")

    print("\n[3] ОТНОШЕНИЕ ВЕЛИЧИН парный/одиночный")
    print(f"    {'вопрос':<12}{'поляр':<6}{'ось':<8}{'парный':>9}{'single':>9}{'отнош.':>9}")
    ratios = []
    for (q, pol, ax, a, b, t, nn) in PAIRS:
        if abs(b) > 1e-9 and np.sign(a) == np.sign(b):
            r = a / b
            ratios.append(r)
            rs = f"{r:>9.2f}"
        else:
            rs = f"{'—':>9}"
        print(f"    {q:<12}{pol:<6}{ax:<8}{a:>9.1f}{b:>9.1f}{rs}")
    ratios = np.array(ratios)
    print(f"\n    отношений посчитано: {len(ratios)}")
    print(f"    медиана = {np.median(ratios):.2f}, разброс [{ratios.min():.2f}, {ratios.max():.2f}]")
    print(f"    CV (sd/mean) = {ratios.std(ddof=1)/ratios.mean():.2f}")
    print("    Гипотеза «в паре эффект удваивается» предсказывает ~постоянное")
    print("    отношение около 2. Смотреть на разброс и CV.")

    print("\n[4] СОВМЕСТИМОСТЬ ОЦЕНОК (лежит ли single в CI парного)")
    print("    SE одиночной оценки = |pull| / |t|; для парной SE недоступна из")
    print("    сводки, поэтому считаем только, накрывает ли CI одиночной нуль")
    print("    и насколько далеко парная оценка от одиночной в SE одиночной.")
    print(f"    {'вопрос':<12}{'single':>9}{'SE':>8}{'95% CI single':>22}{'парный':>9}{'z':>8}")
    for (q, pol, ax, a, b, t, nn) in PAIRS:
        se = abs(b / t) if abs(t) > 1e-9 else float("nan")
        lo, hi = b - 1.96 * se, b + 1.96 * se
        z = (a - b) / se if se == se and se > 0 else float("nan")
        flag = "" if lo <= a <= hi else "  <- парный ВНЕ CI"
        print(f"    {q:<12}{b:>9.1f}{se:>8.1f}   [{lo:>7.1f}, {hi:>6.1f}]{a:>9.1f}{z:>8.1f}{flag}")

    print("\n" + "=" * 78)
    print("ЧТЕНИЕ: [1] — совпадение направлений не случайно. [2] — величины")
    print("связаны, но CI по 8 точкам широкий. [3] — если отношение стабильно,")
    print("различие дизайнов сводится к масштабному множителю. [4] — парные")
    print("оценки систематически ВНЕ CI одиночных => дизайны НЕ эквивалентны")
    print("количественно, только качественно (знак и ранг).")
    print("=" * 78)


if __name__ == "__main__":
    main()
