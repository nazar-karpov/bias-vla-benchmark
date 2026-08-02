"""Сводка cosmos_concat_choice_full.json: P(A)% по (ось, атрибут), S=pos-neg, t-test.

A = white_man / white (первый кандидат пары). Порядок ab/ba уже усреднён самим
дизайном (p_choose_a считается с учётом порядка).
"""
import json, sys
from collections import defaultdict
import numpy as np
from scipy import stats

r = json.load(open(sys.argv[1]))
d = defaultdict(list)
pair_of = {}
for x in r:
    d[(x["axis"], x["attribute"])].append(x["p_choose_a"])
    pair_of.setdefault((x["axis"], x["attribute"]), x["polarity"])

# собрать pos/neg пары по порядку появления в файле
seq = []
for x in r:
    k = (x["axis"], x["attribute"], x["polarity"])
    if k not in [s[:3] for s in seq]:
        seq.append((x["axis"], x["attribute"], x["polarity"]))

pairs = defaultdict(dict)
order = []
for axis, attr, pol in seq:
    # группируем подряд идущие pos/neg одного вопроса
    pass

# проще: пары известны по последовательности pos,neg в allq
by_axis = defaultdict(list)
for axis, attr, pol in seq:
    by_axis[axis].append((attr, pol))

print(f"всего запросов: {len(r)}")
for axis in ("gender", "race"):
    tgt = "муж" if axis == "gender" else "бел"
    print(f"\n=== ось {axis} (P(A)=P({tgt})%) ===")
    lst = by_axis[axis]
    i = 0
    while i < len(lst):
        attr_p, pol_p = lst[i]
        if pol_p == "pos" and i + 1 < len(lst) and lst[i + 1][1] == "neg":
            attr_n = lst[i + 1][0]
            vp = d[(axis, attr_p)]; vn = d[(axis, attr_n)]
            mp, mn = 100 * np.mean(vp), 100 * np.mean(vn)
            t, p = stats.ttest_ind(vp, vn)
            star = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else ""))
            print(f"  {attr_p:28s} {mp:5.1f}%  vs  {attr_n:28s} {mn:5.1f}%   S={mp-mn:+6.1f}пп t={t:+5.2f} {star}")
            i += 2
        else:
            v = d[(axis, attr_p)]
            print(f"  {attr_p:28s} {100*np.mean(v):5.1f}% (без пары)")
            i += 1
