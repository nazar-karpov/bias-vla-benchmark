#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Квадратные 512×512 версии картинок датасета под плитку Act2Answer (обобщение
focus_square_crops.py на датасеты без структуры «сцена + варианты»).

Режимы (--mode):
  face   — квадрат со стороной min(W,H), центр по крупнейшему лицу (Haar), иначе центр кадра;
           фото людей (VisBias).
  pad    — вписать целиком в квадрат, поля цветом среднего края (letterbox); сцены, где важен
           весь кадр (VERI-Emergency: опасность может быть с краю).
  center — центральный квадрат.
  none   — только ресайз (уже квадратные, PAIRS 256×256).

Пути сохраняются относительно --src (как в манифестах). Файлы с расширением .jpg, но
внутри WEBP/AVIF (VisBias) открываются PIL по содержимому; недекодируемые — в skipped.csv.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXT = {".jpg", ".jpeg", ".png", ".webp"}


def face_center(path, casc):
    import cv2
    im = np.asarray(Image.open(path).convert("RGB"))
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    det = casc.detectMultiScale(g, scaleFactor=1.1, minNeighbors=5,
                                minSize=(max(24, g.shape[0] // 12),) * 2)
    if len(det) == 0:
        return None
    x, y, w, h = max(det, key=lambda d: d[2] * d[3])
    return x + w / 2, y + h / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["face", "pad", "center", "none"], required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--subdir", default="", help="обрабатывать только этот подкаталог src")
    args = ap.parse_args()
    casc = None
    if args.mode == "face":
        import cv2
        casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    root = args.src / args.subdir if args.subdir else args.src
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXT)
    log, skipped = [], []
    for p in files:
        try:
            im = Image.open(p)
            im.load()
            im = im.convert("RGB")
        except Exception as e:
            skipped.append({"path": str(p.relative_to(args.src)), "reason": repr(e)[:120]})
            continue
        W, H = im.size
        src = args.mode
        if args.mode == "none":
            out = im
        elif args.mode == "pad":
            side = max(W, H)
            edge = np.concatenate([np.asarray(im)[0], np.asarray(im)[-1],
                                   np.asarray(im)[:, 0], np.asarray(im)[:, -1]])
            fill = tuple(int(v) for v in edge.mean(axis=0))
            out = ImageOps.pad(im, (side, side), color=fill, centering=(0.5, 0.5))
        else:
            side = min(W, H)
            cx, cy = W / 2, H / 2
            if args.mode == "face":
                c = face_center(p, casc)
                if c:
                    cx, cy = c
                else:
                    src = "center(no-face)"
            x0 = int(round(min(max(cx - side / 2, 0), W - side)))
            y0 = int(round(min(max(cy - side / 2, 0), H - side)))
            out = im.crop((x0, y0, x0 + side, y0 + side))
        out = out.resize((args.size, args.size), Image.LANCZOS)
        o = args.out / p.relative_to(args.src)
        o = o.with_suffix(".jpg")
        o.parent.mkdir(parents=True, exist_ok=True)
        out.save(o, quality=95)
        log.append({"path": str(p.relative_to(args.src)), "W": W, "H": H, "mode": src})
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "crops.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "W", "H", "mode"]); w.writeheader(); w.writerows(log)
    with (args.out / "skipped.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "reason"]); w.writeheader(); w.writerows(skipped)
    modes = {}
    for l in log:
        modes[l["mode"]] = modes.get(l["mode"], 0) + 1
    print(f"ok {len(log)}, skipped {len(skipped)}, modes {modes}")


if __name__ == "__main__":
    main()
