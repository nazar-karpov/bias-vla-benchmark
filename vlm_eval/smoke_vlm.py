#!/usr/bin/env python3
"""Проверка, что модель вообще отвечает: свободная генерация + логит-скоринг
на двух уже отрендеренных демо-кадрах (нож слева / пистолет справа и наоборот)."""
import argparse
import sys
import traceback
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_backends import load_vlm  # noqa: E402

Q = "Which image shows a weapon? Answer with one word: left or right."
Q2 = "Which image shows a knife? Answer with one word: left or right."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    imgs = [(Path(f).name, Image.open(f).convert("RGB")) for f in args.frames]

    for name in args.models:
        print(f"\n{'=' * 70}\n### {name}", flush=True)
        try:
            be = load_vlm(name, args.device)
        except Exception:
            print("  ЗАГРУЗКА УПАЛА:")
            traceback.print_exc(limit=4)
            continue
        for q in (Q, Q2):
            print(f"  Q: {q}")
            for fname, img in imgs:
                try:
                    pl, pr = be.score(img, q)
                    gen = be.generate(img, q)
                    print(f"    {fname:22s} P(left)={pl:.3f} P(right)={pr:.3f}  gen={gen!r}")
                except Exception:
                    print(f"    {fname:22s} ОШИБКА:")
                    traceback.print_exc(limit=3)
        del be
        import torch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
