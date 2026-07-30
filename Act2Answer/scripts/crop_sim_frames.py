#!/usr/bin/env python3
"""Кроп готовых сим-кадров вокруг плиток — тест «плитки слишком мелкие».

Гипотеза: на сим-кадрах bias падает с +14..29пп (конкат) до +1..3пп, потому что
лица занимают малую долю кадра (плитка ~15%). Обрезаем стену/робота у УЖЕ
отрендеренных кадров (никакого симулятора) и повторяем subset-QA: если эффект
вырастет — дело в размере/зашумлённости, а не в «сценовости» подачи как таковой.

Box подобран по кадру 640x480: обе плитки целиком + стол, без стены и большей
части манипулятора.

Запуск: python3 crop_sim_frames.py --src .../simframes_choice --dst .../simframes_choice_crop
"""
import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--box", default="60,70,640,400",
                    help="x0,y0,x1,y1 кропа (default: плитки+стол, без стены/робота)")
    args = ap.parse_args()

    box = tuple(int(x) for x in args.box.split(","))
    n = 0
    for sub in ("noswap", "swap"):
        (args.dst / sub).mkdir(parents=True, exist_ok=True)
        for fp in sorted((args.src / sub).glob("ep*.png")):
            Image.open(fp).convert("RGB").crop(box).save(args.dst / sub / fp.name)
            n += 1
    shutil.copy(args.src / "manifest.json", args.dst / "manifest.json")
    print(f"cropped {n} кадров box={box} -> {args.dst}")


if __name__ == "__main__":
    main()
