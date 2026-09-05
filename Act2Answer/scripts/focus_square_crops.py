#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FOCUS → квадратные 512×512 кропы под плитку Act2Answer, ОДИН кроп-бокс на сцену.

Фото FOCUS прямоугольные (1264×848 и др.), а плитка квадратная; make_cardset просто
сжимает в 512×512 и искажает лица. Вместо этого режем квадрат со стороной min(W,H),
центрируя его на лице. Лицо ищется Haar-каскадом на base.jpg сцены (и на всех вариантах —
берётся медиана центров, если base не дал детекции); при полном провале — центр кадра.
Бокс один на всю сцену (base + 10 демографий), чтобы контрфактические пары оставались
попиксельно параллельными вне лица.

Выход: <out>/original/focus/<occ>/<scene>/<name>.jpg (та же структура, что у манифестов)
+ <out>/crops.csv (сцена, источник бокса, бокс).
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def faces(img_bgr, casc):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    det = casc.detectMultiScale(g, scaleFactor=1.1, minNeighbors=5,
                                minSize=(max(24, g.shape[0] // 12),) * 2)
    return [(x + w / 2, y + h / 2, w * h) for (x, y, w, h) in det]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="…/focus_reflect (содержит original/focus)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    root = args.src / "original" / "focus"
    log = []
    n_img = 0
    for scene in sorted(p for p in root.glob("*/*") if p.is_dir()):
        imgs = sorted(q for q in scene.glob("*.jpg") if q.name != "base.jpg")  # base мельче (540×360), в парах не участвует
        if not imgs:
            continue
        W, H = Image.open(imgs[0]).size
        side = min(W, H)
        src = "center"
        cx, cy = W / 2, H / 2
        base = scene / "base.jpg"
        cands = []
        if base.exists():
            bimg = cv2.imread(str(base))
            f = faces(bimg, casc)
            if f:
                bx, by, _ = max(f, key=lambda t: t[2])
                cx, cy = bx * W / bimg.shape[1], by * H / bimg.shape[0]; src = "base"
        if src == "center":
            for p in imgs:
                f = faces(cv2.imread(str(p)), casc)
                if f:
                    cands.append(max(f, key=lambda t: t[2])[:2])
            if cands:
                cx, cy = np.median(np.array(cands), axis=0); src = f"median{len(cands)}"
        x0 = int(round(min(max(cx - side / 2, 0), W - side)))
        y0 = int(round(min(max(cy - side / 2, 0), H - side)))
        for p in imgs:
            im = Image.open(p).convert("RGB")
            assert im.size == (W, H), (p, im.size)
            im = im.crop((x0, y0, x0 + side, y0 + side)).resize((args.size, args.size), Image.LANCZOS)
            o = args.out / p.relative_to(args.src)
            o.parent.mkdir(parents=True, exist_ok=True)
            im.save(o, quality=95)
            n_img += 1
        log.append({"scene": str(scene.relative_to(root)), "W": W, "H": H, "side": side,
                    "x0": x0, "y0": y0, "source": src, "n": len(imgs)})
        print(f"{scene.relative_to(root)}: {W}x{H} box=({x0},{y0},{side}) {src}", flush=True)
    with (args.out / "crops.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys())); w.writeheader(); w.writerows(log)
    srcs = [l["source"] for l in log]
    print(f"сцен {len(log)}, картинок {n_img}; бокс по base: {srcs.count('base')}, "
          f"по медиане: {sum(s.startswith('median') for s in srcs)}, центр: {srcs.count('center')}")


if __name__ == "__main__":
    main()
