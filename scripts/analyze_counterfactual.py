#!/usr/bin/env python3
"""Анализ counterfactual P(yes) (из magma_counterfactual.py): чистый causal bias.

Для каждого признака считаем PAIRED-разницу внутри сцены (фон/поза одинаковы,
меняется только раса или только пол):
  - race:   в каждой сцене берём P(yes|white) − P(yes|black) при фикс. поле, усредняем;
  - gender: P(yes|man) − P(yes|woman) при фикс. расе.
Paired = вычитаем внутри одной сцены → фон сокращается, остаётся чистый эффект
признака. Плюс t-тест по парам (значим ли сдвиг).

Использование: python analyze_counterfactual.py outputs/magma_counterfactual.json
"""
import json
import sys
from collections import defaultdict

import numpy as np


def main():
    rows = json.load(open(sys.argv[1]))
    # index: (trait, scene, race, gender) -> p_yes
    idx = {}
    traits = []
    for r in rows:
        idx[(r["trait"], r["scene"], r["race"], r["gender"])] = r["p_yes"]
        if r["trait"] not in traits:
            traits.append(r["trait"])
    scenes = sorted({r["scene"] for r in rows})

    print(f"{len(scenes)} сцен, {len(traits)} признаков\n")
    print(f"{'trait':<14} {'RACE white−black':>18} {'p':>7}   {'GENDER man−woman':>18} {'p':>7}")
    print("-" * 70)

    def paired(trait, axis):
        """axis='race': (white−black) при фикс. поле; 'gender': (man−woman) при фикс. расе."""
        diffs = []
        for sc in scenes:
            if axis == "race":
                for g in ("man", "woman"):
                    a = idx.get((trait, sc, "white", g)); b = idx.get((trait, sc, "black", g))
                    if a is not None and b is not None:
                        diffs.append(a - b)
            else:
                for rc in ("white", "black"):
                    a = idx.get((trait, sc, rc, "man")); b = idx.get((trait, sc, rc, "woman"))
                    if a is not None and b is not None:
                        diffs.append(a - b)
        d = np.array(diffs)
        if len(d) == 0:
            return 0.0, 1.0, 0
        mean = d.mean()
        # one-sample t против 0
        if d.std(ddof=1) > 0:
            t = mean / (d.std(ddof=1) / np.sqrt(len(d)))
            # двусторонний p через нормальную аппроксимацию (n достаточно велик)
            from math import erf, sqrt
            p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
        else:
            p = 1.0
        return mean, p, len(d)

    summary = {}
    for tr in traits:
        rm, rp, rn = paired(tr, "race")
        gm, gp, gn = paired(tr, "gender")
        star_r = "***" if rp < 0.001 else "**" if rp < 0.01 else "*" if rp < 0.05 else ""
        star_g = "***" if gp < 0.001 else "**" if gp < 0.01 else "*" if gp < 0.05 else ""
        print(f"{tr:<14} {rm:>+17.3f}{star_r:<3} {rp:>6.3f}   {gm:>+17.3f}{star_g:<3} {gp:>6.3f}")
        summary[tr] = {"race_white_minus_black": rm, "race_p": rp, "race_n": rn,
                       "gender_man_minus_woman": gm, "gender_p": gp, "gender_n": gn}

    print("\nЧитать: + по расе = признак чаще приписан WHITE; + по полу = чаще MAN.")
    print("* p<0.05  ** p<0.01  *** p<0.001 (paired, фон сокращён)")
    print("Контроль 'a person' должен быть ~0 (санити). 'a criminal' — негативный признак.")
    json.dump(summary, open(sys.argv[1].replace('.json', '_summary.json'), 'w'), indent=2)


if __name__ == "__main__":
    main()
