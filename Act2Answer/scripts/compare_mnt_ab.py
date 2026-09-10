#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Попарное сравнение двух плечей А/Б лимита генерации Magma на одних и тех же 48 эпизодах.

Плечо A = mnt1000-doctorate: боевое поведение (лимит 1000, при отсутствии EOS — последние 7 токенов).
Плечо B = mnt8-doctorate:    фикс (лимит 8, при отсутствии EOS — первые 7 токенов).
Оба в жадном режиме на одних сценах, поэтому среда без «убегающей» генерации должна пройти
одинаково в обоих плечах; траектории расходятся с шага первого убегания.
"""
import os
import sys

import numpy as np
import yaml

OUT = "/home/moskalenko/ws/bias-vla-benchmark-main/Act2Answer/outputs"
A_NAME, B_NAME = sys.argv[1:3] if len(sys.argv) >= 3 else ("mnt1000-doctorate", "mnt8-doctorate")


def load(name):
    d = os.path.join(OUT, name, "glob", "vis_0_test")
    li = yaml.safe_load(open(os.path.join(d, "stats.yaml")))["last_info"]
    tr = np.load(os.path.join(d, "traj.npz"))
    return li, tr


la, ta = load(A_NAME)
lb, tb = load(B_NAME)
idx = sorted(set(la) & set(lb))
n = len(idx)
print(f"эпизодов в обоих плечах: {n}   (A={A_NAME}, B={B_NAME})\n")


def col(li, k):
    return np.array([float(li[i].get(k, np.nan)) for i in idx])


for k, title in (("first_touch_side", "первое касание"), ("chosen_side", "финальный ответ")):
    a, b = col(la, k), col(lb, k)
    print(f"{title:18s}: совпадает в {100*np.mean(a == b):5.1f}% | покрытие A {100*np.mean(a > 0):5.1f}%, "
          f"B {100*np.mean(b > 0):5.1f}% | вправо A {100*np.mean(a[a > 0] == 2) if (a > 0).any() else 0:5.1f}%, "
          f"B {100*np.mean(b[b > 0] == 2) if (b > 0).any() else 0:5.1f}%")

ya, yb = col(la, "cube_fy"), col(lb, "cube_fy")
za, zb = col(la, "cube_fz"), col(lb, "cube_fz")
dy = np.abs(ya - yb) * 1000
print(f"\nфинальный y куба: |разница| медиана {np.median(dy):.1f} мм, "
      f"совпадает (<1 мм) в {100*np.mean(dy < 1):.1f}% эпизодов")
print(f"куб на столе: A {100*np.mean(za >= 0.8):.1f}%, B {100*np.mean(zb >= 0.8):.1f}%")

ca, cb = ta["cube_xyz"], tb["cube_xyz"]
steps = min(ca.shape[1], cb.shape[1])
div = []
for e in range(min(ca.shape[0], cb.shape[0])):
    d = np.abs(ca[e, :steps] - cb[e, :steps]).max(axis=1)
    w = np.where(d > 1e-3)[0]
    div.append(int(w[0]) if len(w) else -1)
div = np.array(div)
same = div < 0
print(f"\nтраектории куба: полностью совпали в {same.sum()} из {len(div)} эпизодов")
if (~same).any():
    print(f"  разошлись в {(~same).sum()}: шаг расхождения медиана {np.median(div[~same]):.0f}, "
          f"мин {div[~same].min()}, макс {div[~same].max()}")


def tele(c):
    return np.array([np.abs(np.diff(c[e, :, 1])).max() > 0.2 for e in range(c.shape[0])])


print(f"\nтелепорты (>0.2 м за шаг): A {100*tele(ca).mean():.1f}%, B {100*tele(cb).mean():.1f}%")
