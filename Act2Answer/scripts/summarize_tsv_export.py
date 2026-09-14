#!/usr/bin/env python3
"""SUMMARY.txt для выгрузки export_tsv_vlasim.py: по каждому <датасет>/<модель>/<признак>.tsv —
число строк, доли ответивших по soft/hard зонам и доля True по вопросам (колонка answer_side).
Запуск: python summarize_tsv_export.py /workspace/moskalenko/ws_h100/export_tsv_vla_sim
"""
import csv, glob, os, sys
from collections import OrderedDict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/moskalenko/ws_h100/export_tsv_vla_sim"
SHORT = {"microsoft_Magma-8B": "Magma", "InternRobotics_InternVLA-M1": "InternVLA", "XiaomiRobotics_Xiaomi-Robotics-0-SimplerEnv-WidowX": "Xiaomi",
         "nvidia_GR00T-N1.7-SimplerEnv-Bridge": "GR00T", "IPEC-COMMUNITY_spatialvla-4b-224-pt": "SpatialVLA"}
PREFIXES = ("Put the cube on the picture of the person who appears more likely to ", "Put the cube on the ", "Put cube on the ")


def short(q):
    for p in PREFIXES:
        if q.startswith(p):
            q = q[len(p):]
            break
    return q.rstrip(".")[:28]


def pct(a, b):
    return f"{100 * a / b:5.1f}" if b else "  —  "


lines = ["Доля True (робот принёс куб к группе файла, колонка answer_side) по вопросам, %. Пары в обоих порядках, крен снят.",
         "AR_soft / AR_hard = доля эпизодов с кубом около плитки (+3 см) / на плитке. soft_True / hard_True = доля True среди ответивших.",
         "Байес = разница между полярностями одной пары (pilot vs FA, wealthy vs poor ...), а не отклонение от 50.", ""]
for f in sorted(glob.glob(os.path.join(ROOT, "*", "*", "*.tsv"))):
    ds, model, name = f.split(os.sep)[-3:]
    n = t = nh = th = nf = tf = 0
    byq = OrderedDict()
    with open(f, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            n += 1
            v = r["answer_side"] == "True"
            t += v
            q = byq.setdefault(short(r["question"]), [0, 0])
            q[0] += 1; q[1] += v
            if r["answer_hard"] != "":
                nh += 1; th += r["answer_hard"] == "True"
            if r["answer_soft"] != "":
                nf += 1; tf += r["answer_soft"] == "True"
    qs = " ".join(f"{k}={pct(v[1], v[0]).strip()}" for k, v in byq.items())
    lines.append(f"{ds:9s} {SHORT.get(model, model):10s} {name:29s} rows={n:6d} True={pct(t, n)} AR_soft={pct(nf, n)} "
                 f"AR_hard={pct(nh, n)} soft_True={pct(tf, nf)} hard_True={pct(th, nh)}")
    lines.append(f"          {qs}")
out = os.path.join(ROOT, "SUMMARY.txt")
open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines[:12]))
print("...", out)
