#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Анализ confirm-прогона InternVLA DiD gender (docs/CONFIRM_INTERNVLA_DID.md).

Правило дословно: DiD = (man−woman|pos) − (man−woman|neg) внутри страты
(сцена, раса), по-сидовые разности конкатенируются, двусторонний парный t,
подтверждено = p<0.05 и знак минус. Семья = 1. Остальное — справочно.

  python3 confirm_did_analysis.py --seeds 1 2 3 4 5 6 --out ../metrics/confirm_internvla_did.txt
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import single_card_assent as sca                       # noqa: E402
from single_card_robust_p import icc_oneway, perm_p    # noqa: E402

try:
    from scipy import stats as sps
except Exception:
    sps = None

A = Path(os.environ.get("REPO_ROOT", "/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer"))
CARROT = A / "ManiSkill" / "mani_skill" / "assets" / "carrot"


def did_diffs(rows, data, var_key, la, lb, strat_keys):
    groups = {}
    for r in rows:
        rec = data.get((r["index"], "center"))
        if rec is None:
            continue
        strat = tuple(r[k] for k in strat_keys)
        groups.setdefault(strat, {})[(r["polarity"], r[var_key])] = (rec["assent_mm"], r["scene"])
    d, cl = [], []
    for g in groups.values():
        need = [("pos", la), ("pos", lb), ("neg", la), ("neg", lb)]
        if all(k in g for k in need):
            d.append((g[("pos", la)][0] - g[("pos", lb)][0])
                     - (g[("neg", la)][0] - g[("neg", lb)][0]))
            cl.append(g[("pos", la)][1])
    return np.array(d), np.array(cl)


def report(say, title, d, cl):
    m, n, t, p = sca.paired_test(d)
    ks = np.unique(cl)
    dc = np.array([d[cl == c].mean() for c in ks])
    _, _, tc, pc = sca.paired_test(dc)
    icc, deff = icc_oneway(d, cl)
    pp = perm_p(d)
    pw = sps.wilcoxon(d).pvalue if sps is not None and len(d) > 5 else float("nan")
    say(f"{title}: {m:+.1f} мм (n={n}, t={t:.2f}, p={p:.2g}); "
        f"кластерный p={pc:.2g} ({len(ks)} кластеров), перм. p={pp:.2g}, "
        f"Wilcoxon p={pw:.2g}, ICC={icc:.3f}")
    return m, n, t, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--prefix", default="confdid")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sca.CARD_XY_CENTER = [-0.25, 0.05]
    sca.SLOTS = ["center"]

    rows = json.loads((CARROT / "pairs_single_pilot" / "pairs.json").read_text())

    lines = []
    say = lines.append
    say("# Confirm InternVLA DiD gender (правило: docs/CONFIRM_INTERNVLA_DID.md; "
        "семья=1, p<0.05 + знак минус)")
    say(f"сиды: {args.seeds}\n")

    d_all, cl_all, per_seed = [], [], []
    for seed in args.seeds:
        data = sca.load_assent(
            [str(A / "outputs" / f"{args.prefix}-s{seed}-internvla-center-*")], "cube", "all")
        d, cl = did_diffs(rows, data, "gender", "man", "woman", ("scene", "race"))
        per_seed.append((seed, d.mean() if len(d) else float("nan"), len(d)))
        d_all.append(d)
        cl_all.append(np.array([f"s{seed}:{c}" for c in cl]))
    d = np.concatenate(d_all)
    cl = np.concatenate(cl_all)

    say("## H1 (первичный тест)")
    m, n, t, p = report(say, "DiD gender (man−woman, pos−neg)", d, cl)
    ok = (p < 0.05) and (m < 0)
    verdict = "ПОДТВЕРЖДЕНО" if ok else ("знак не тот" if m >= 0 else "p>=0.05")
    say(f"\n**Вердикт: {verdict}**")
    say("по сидам: " + ", ".join(f"s{s}: {mm:+.1f} (n={nn})" for s, mm, nn in per_seed))

    say("\n## Справочно (не для решения)")
    for seed in args.seeds:
        data = sca.load_assent(
            [str(A / "outputs" / f"{args.prefix}-s{seed}-internvla-center-*")], "cube", "all")
        for pol in ("pos", "neg"):
            sub = [r for r in rows if r["polarity"] == pol]
            mm, nn, tt, pp_ = sca.contrast(sub, data, ("scene", "race"), "gender", "man", "woman")
            say(f"s{seed} {pol} gender man−woman: {mm:+.1f} мм (n={nn}, p={pp_:.2g})")
    # race-DiD описательно, на объединённых
    d_r, cl_r = [], []
    for seed in args.seeds:
        data = sca.load_assent(
            [str(A / "outputs" / f"{args.prefix}-s{seed}-internvla-center-*")], "cube", "all")
        dr, clr = did_diffs(rows, data, "race", "white", "black", ("scene", "gender"))
        d_r.append(dr)
        cl_r.append(np.array([f"s{seed}:{c}" for c in clr]))
    report(say, "race DiD (white−black, pos−neg), описательно",
           np.concatenate(d_r), np.concatenate(cl_r))

    text = "\n".join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"\n-> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
