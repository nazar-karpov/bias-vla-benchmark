#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FDR по InternVLA top8 (28 ячеек, из metrics/internvla_top8_final.txt)."""
import re
import numpy as np

TESTS = []
for line in open("/workspace/moskalenko/bias-vla-benchmark-main/metrics/internvla_top8_final.txt"):
    m = re.match(
        r"\|\s*(\w+)\s*\|\s*(pos|neg)\s*\|\s*(gender|race)\s*\|\s*\*\*(\*?)"
        r"([+-][\d.]+)(\*?)\*\*\s*\|.*?\|\s*([\d.]+|nan)\s*\|", line)
    if m:
        q, pol, ax, _, pull, _, p = m.groups()
        TESTS.append((q, pol, ax, float(pull), float(p)))

p = np.array([t[4] for t in TESTS])
m = len(p)
order = np.argsort(p)
q = p[order] * m / np.arange(1, m + 1)
q = np.minimum.accumulate(q[::-1])[::-1]
Q = np.empty(m)
Q[order] = np.minimum(q, 1.0)

print(f"{'вопрос':<11}{'пол':<5}{'ось':<8}{'pull':>8}{'p':>10}{'q(FDR)':>10}  знач.")
for (qk, pol, ax, pull, pv), qv in sorted(zip(TESTS, Q), key=lambda z: z[1]):
    print(f"{qk:<11}{pol:<5}{ax:<8}{pull:>+8.1f}{pv:>10.3g}{qv:>10.4f}  {'ДА' if qv<0.05 else '-'}")
print(f"\nтестов={m}, значимых после FDR={int((Q<0.05).sum())}")
