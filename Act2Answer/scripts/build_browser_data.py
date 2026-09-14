#!/usr/bin/env python3
"""Компактные JS-данные для браузера эпизодов (артефакт): cardsets.js + ep_<vla>.js.
cardsets: на кардсет — список картинок (интернирован), вопросы, строки [left_img, right_img, qid, pol, axis, a1, a2].
ep_<vla>: на кардсет и порядок — y_release (×10, мм, int) и strict chosen_side (0/1/2), по индексу эпизода; null = нет.
"""
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_tsv_vlasim import load_run, run_prefix, ASSETS, MODELS, RUNS

OUTD = sys.argv[1] if len(sys.argv) > 1 else "/workspace/moskalenko/ws_h100/browser_data"
os.makedirs(OUTD, exist_ok=True)
ATTR = {"tsv:gender": "gender", "tsv:ethnicity": "ethnicity", "tsv:skin_color": "skin_color", "tsv:pairs": "safety"}

cs_out = {}
for prog, cs, ds in RUNS:
    eps = list(csv.DictReader(open(os.path.join(ASSETS, cs, "episodes.csv"), encoding="utf-8")))
    pairs = json.load(open(os.path.join(ASSETS, cs, "pairs.json"), encoding="utf-8"))
    imgs, im_idx, qs, q_idx, ax, ax_idx, rows = [], {}, [], {}, [], {}, []
    def intern(v, lst, idx):
        if v not in idx:
            idx[v] = len(lst); lst.append(v)
        return idx[v]
    attr = ATTR[eps[0]["source"]]
    for r in eps:
        i = int(r["index"])
        if attr == "safety":
            a1, a2 = r["attr_risk_1"], r["attr_risk_2"]
        else:
            a1, a2 = r[f"attr_{attr}_1"], r[f"attr_{attr}_2"]
        rows.append([intern(r["left_image"], imgs, im_idx), intern(r["right_image"], imgs, im_idx),
                     intern(r["question_id"] + "|" + pairs[i]["question"], qs, q_idx),
                     r["polarity"], intern(r["axis"], ax, ax_idx), a1, a2])
    cs_out[cs] = {"prog": prog, "ds": ds, "attr": attr, "imgs": imgs, "qs": qs, "axes": ax, "rows": rows}
    print(cs, len(rows), "rows,", len(imgs), "imgs")
with open(os.path.join(OUTD, "cardsets.js"), "w", encoding="utf-8") as f:
    f.write("window.CS=" + json.dumps(cs_out, ensure_ascii=False, separators=(",", ":")) + ";")

for vla in MODELS:
    out = {}
    for prog, cs, ds in RUNS:
        prefix = run_prefix(prog, vla, cs)
        n = len(cs_out[cs]["rows"])
        d = {}
        for order, key in (("noswap", "ns"), ("swap", "sw")):
            data = load_run(prefix, order)
            if not data:
                continue
            y = [None] * n; s = [None] * n
            for i, (yy, bl, br, chosen) in data.items():
                if i < n:
                    y[i] = int(round(yy * 10000)); s[i] = 0 if chosen is None else int(chosen)
            d[key] = {"y": y, "s": s, "n": sum(v is not None for v in y)}
        if d:
            out[cs] = d
        print(vla, cs, {k: v["n"] for k, v in d.items()})
    with open(os.path.join(OUTD, f"ep_{vla}.js"), "w", encoding="utf-8") as f:
        f.write(f"window.EP=window.EP||{{}};window.EP[{json.dumps(vla)}]=" + json.dumps(out, separators=(",", ":")) + ";")
print("done", OUTD)
