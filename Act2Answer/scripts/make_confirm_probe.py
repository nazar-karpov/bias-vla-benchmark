#!/usr/bin/env python3
"""Пробная подвыборка confirm-кардсета: первые PER эп. каждого блока
(вопрос x ось x полярность). Блоки в исходнике по 100 эп. каждый (16 блоков),
берём первые PER из каждого, переиндексируем 0..N-1, пересобираем blocks.json.

model_db.json индексируется по ИМЕНИ плитки, не по index эпизода -> копируем
как есть. shapes -> симлинк на исходный (та же геометрия).

Usage: python make_confirm_probe.py <src_cardset> <dst_cardset> [--per 40]
"""
import argparse, json, os, shutil
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--per", type=int, default=40)
    args = ap.parse_args()

    pairs = json.load(open(os.path.join(args.src, "pairs.json")))
    # блок = (qkey, axis, polarity); внутри блока берём первые PER по index
    blocks = defaultdict(list)
    for p in pairs:
        blocks[(p["qkey"], p["axis"], p["polarity"])].append(p)
    for k in blocks:
        blocks[k].sort(key=lambda x: x["index"])

    # порядок вывода: сохраняем крупные диапазоны как в оригинале
    # (qkey-порядок по первому появлению), внутри — pos перед neg, gender перед race
    order = []
    seen = []
    for p in pairs:
        q = p["qkey"]
        if q not in seen:
            seen.append(q)
    for q in seen:
        for pol in ("pos", "neg"):
            for ax in ("gender", "race"):
                key = (q, ax, pol)
                if key in blocks:
                    order.append(key)

    out = []
    new_blocks = []
    idx = 0
    for key in order:
        chosen = blocks[key][: args.per]
        b_start = idx
        for p in chosen:
            np_ = dict(p)
            np_["index"] = idx
            out.append(np_)
            idx += 1
        q, ax, pol = key
        new_blocks.append({"start_id": b_start, "end_id": idx - 1,
                           "qkey": q, "axis": ax, "polarity": pol,
                           "question_phrase": chosen[0]["question"]})

    os.makedirs(args.dst, exist_ok=True)
    json.dump(out, open(os.path.join(args.dst, "pairs.json"), "w"), indent=1)
    json.dump(new_blocks, open(os.path.join(args.dst, "blocks.json"), "w"), indent=2)
    shutil.copy(os.path.join(args.src, "model_db.json"),
                os.path.join(args.dst, "model_db.json"))
    # shapes: симлинк на исходный target (та же геометрия)
    src_shapes = os.path.join(args.src, "shapes")
    dst_shapes = os.path.join(args.dst, "shapes")
    if os.path.islink(dst_shapes) or os.path.exists(dst_shapes):
        os.remove(dst_shapes)
    os.symlink(os.path.realpath(src_shapes), dst_shapes)

    print(f"OK: {len(out)} эп ({args.per}/блок), {len(new_blocks)} блоков -> {args.dst}")
    for b in new_blocks:
        print(f"  {b['qkey']:<8} {b['axis']:<6} {b['polarity']:<3} "
              f"idx {b['start_id']}..{b['end_id']}")


if __name__ == "__main__":
    main()
