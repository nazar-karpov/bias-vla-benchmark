#!/usr/bin/env python3
"""КОНТРОЛЬ к safety-прогону: те же 96 пар и те же 2 вопроса, но БЕЗ симулятора —
модель видит склейку двух ИСХОДНЫХ картинок (левая|правая), а не кадр робота.

Зачем: на кадрах симулятора все модели легли на 50%. Это либо «VLM не отличает
оружие», либо «плитка в кадре слишком мелкая, объект не читается». Разделяет эти две
версии только этот контроль: если на склейке модель уверенно различает, а на кадре нет —
виновата подача, а не знание.

Всё остальное 1:1 с vlm_confirm_choice.py: логит-скоринг left/right, оба порядка,
pos/neg, p_choose_a с поправкой на порядок.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_confirm_choice import (  # noqa: E402
    QPAIRS_SAFETY, QKEY, make_q, load_backend, score_lr, generate_answer,
)


def concat_lr(a, b, gap=8):
    h = max(a.height, b.height)
    c = Image.new("RGB", (a.width + gap + b.width, h), (255, 255, 255))
    c.paste(a, (0, (h - a.height) // 2))
    c.paste(b, (a.width + gap, (h - b.height) // 2))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True,
                    choices=("magma", "paligemma", "qwen", "prismatic"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frames-dir", required=True, type=Path, help="нужен только manifest.json")
    ap.add_argument("--images-dir", required=True, type=Path, help="исходные jpg плиток")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    ap.add_argument("--gen-check", type=int, default=8)
    ap.add_argument("--limit-episodes", type=int, default=0)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    manifest = json.loads((args.frames_dir / "manifest.json").read_text())
    if args.limit_episodes:
        manifest = manifest[:args.limit_episodes]
    total = len(manifest) * 2 * 2
    print(f"эпизодов: {len(manifest)} -> {total} запросов (контроль на склейке)", flush=True)

    model, processor, build, gen_fn = load_backend(args.backend, args.model, args.device, dtype)
    print("loaded.", flush=True)

    def img_of(stem):
        for ext in (".jpg", ".png", ".jpeg"):
            p = args.images_dir / f"{stem}{ext}"
            if p.exists():
                return Image.open(p).convert("RGB")
        raise FileNotFoundError(f"нет картинки для {stem} в {args.images_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results, done = [], 0
    for ei, ep in enumerate(manifest):
        la, lb = img_of(ep["cand_a"]), img_of(ep["cand_b"])
        for order in ("ab", "ba"):
            img = concat_lr(la, lb) if order == "ab" else concat_lr(lb, la)
            for pos, neg in QPAIRS_SAFETY:
                for attr, pol in ((pos, "pos"), (neg, "neg")):
                    q = make_q(attr, "safety")
                    p_left = score_lr(model, processor, img, q, args.device, build)
                    rec = {"model": args.tag, "index": ep["index"], "scene": ep["scene"],
                           "axis": ep["axis"], "cand_a": ep["cand_a"], "cand_b": ep["cand_b"],
                           "qkey": QKEY[attr], "attribute": attr, "polarity": pol,
                           "order": order, "question": q, "p_left": p_left,
                           "p_choose_a": p_left if order == "ab" else 1 - p_left}
                    if ei < args.gen_check:
                        rec["gen_raw"], rec["gen_side"] = generate_answer(
                            model, processor, img, q, build, args.device, gen_fn)
                    results.append(rec)
                    done += 1
        if (ei + 1) % 20 == 0 or ei == len(manifest) - 1:
            args.out.write_text(json.dumps(results, ensure_ascii=False))
            print(f"  [{done}/{total}] ep{ep['index']} {ep['scene']}", flush=True)

    args.out.write_text(json.dumps(results, ensure_ascii=False))
    print(f"\nDone: {len(results)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
