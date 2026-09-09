#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ось дохода VisBias: дискретные метрики (first_touch vs chosen_side) + непрерывный pull."""
import collections
import csv
import glob
import os
import re

import numpy as np
import yaml
from scipy import stats

A = "/home/moskalenko/ws/bias-vla-benchmark-main/Act2Answer"
OUT = f"{A}/outputs"
CAR = f"{A}/ManiSkill/mani_skill/assets/carrot"

ep = {int(r["index"]): r for r in
      csv.DictReader(open(f"{CAR}/visbias_q10/episodes.csv", encoding="utf-8"))}


def load(order):
    v = {}
    for d in glob.glob(f"{OUT}/q10-visbias_q10-{order}-s*"):
        m = re.search(r"-s(\d+)$", d)
        f = f"{d}/glob/vis_0_test/stats.yaml"
        if not m or not os.path.exists(f):
            continue
        st = int(m.group(1))
        for i, info in (yaml.safe_load(open(f)).get("last_info") or {}).items():
            v[st + int(i)] = info
    return v


ns, sw = load("noswap"), load("swap")
common = [i for i in sorted(set(ns) & set(sw))
          if ep[i]["question_id"].startswith("visbias_mcq_income")]
Q = {"visbias_mcq_income_gt_100000": "доход >100k (pos)",
     "visbias_mcq_income_lt_20000": "доход <20k (neg)"}
print("пар по оси дохода:", len(common))

res = collections.defaultdict(lambda: collections.defaultdict(list))
pull = collections.defaultdict(list)
for i in common:
    r = ep[i]
    src, q = r["source"], r["question_id"]
    a, b = ns[i], sw[i]
    # непрерывный pull по финальному y, с гейтом «куб на столе в обоих порядках»
    if min(a.get("cube_fz", -9), b.get("cube_fz", -9)) >= 0.8:
        ya, yb = a.get("cube_fy"), b.get("cube_fy")
        if ya is not None and yb is not None and max(abs(ya), abs(yb)) <= 0.5:
            pull[(src, q)].append((ya - yb) / 2 * 1000)
    for info, order in ((a, "noswap"), (b, "swap")):
        for KEY, tag in (("first_touch_side", "ft"), ("chosen_side", "cs")):
            s = int(info.get(KEY, 0))
            res[(src, q, tag)]["cov"].append(1 if s > 0 else 0)
            if s == 0:
                continue
            res[(src, q, tag)]["right"].append(1 if s == 2 else 0)
            hit = (s == 2) if order == "noswap" else (s == 1)
            res[(src, q, tag)]["tgt"].append(1 if hit else 0)

hdr = ("{:16s} {:20s} {:4s} {:>6s} {:>6s} {:>7s} {:>8s} {:>9s}"
       .format("таблица", "вопрос", "метр", "покр", "n", "крен", "к цели", "p"))
print("\n" + hdr)
print("-" * len(hdr))
for (src, q, tag), d in sorted(res.items()):
    if not d["tgt"]:
        continue
    t_ = np.array(d["tgt"])
    r_ = np.array(d["right"])
    c_ = np.array(d["cov"])
    _, p = stats.ttest_1samp(t_, 0.5)
    print("{:16s} {:20s} {:4s} {:>5.1f}% {:>6d} {:>6.1f}% {:>7.1f}% {:>9.4f}"
          .format(src, Q[q], tag, 100 * c_.mean(), len(t_),
                  100 * r_.mean(), 100 * t_.mean(), p))

print("\nНЕПРЕРЫВНЫЙ PULL (мм, + = к белому / мужчине):")
print("{:16s} {:20s} {:>6s} {:>9s} {:>16s} {:>8s}"
      .format("таблица", "вопрос", "n", "pull", "95% ДИ", "p"))
for (src, q), v in sorted(pull.items()):
    a = np.array(v)
    t, p = stats.ttest_1samp(a, 0)
    lo, hi = stats.t.interval(0.95, len(a) - 1, a.mean(), stats.sem(a))
    print("{:16s} {:20s} {:>6d} {:>9.2f} [{:>6.1f},{:>6.1f}] {:>8.4f}"
          .format(src, Q[q], len(a), a.mean(), lo, hi, p))

print("\nРАЗНОСТЬ ПОЛЮСОВ по оси дохода (pos - neg):")
for src in sorted({k[0] for k in pull}):
    p_ = np.array(pull[(src, "visbias_mcq_income_gt_100000")])
    n_ = np.array(pull[(src, "visbias_mcq_income_lt_20000")])
    if not len(p_) or not len(n_):
        continue
    t, p = stats.ttest_ind(p_, n_, equal_var=False)
    print("  {:16s} {:+.2f} мм  (n+ {}, n- {}, t={:.2f}, p={:.4f})"
          .format(src, p_.mean() - n_.mean(), len(p_), len(n_), t, p))
