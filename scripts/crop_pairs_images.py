#!/usr/bin/env python3
"""Center-crop PAIRS tile images so the person/subject fills more of the frame.

The PAIRS portraits place the person roughly centered with scene/background
padding. For the "cropped" cardset we take a centered square crop covering
``--frac`` of the shorter side (default 0.65), biased slightly upward so faces
are kept (people are usually in the upper-center of a portrait), then resize to
512x512 -- the exact texture size make_cardset.py expects. No aspect distortion.

Input:  --in  DIR   flat folder of tile PNG/JPG (make_cardset naming, stem = tile)
Output: --out DIR   same filenames, cropped + resized to 512x512

Reuse the SAME questions.csv as the uncropped set -> identical episodes/answers,
only the tile textures change (isolates the "does cropping help the model see
the key content" question).
"""
import argparse
from pathlib import Path

from PIL import Image

TEX_SIZE = 512
EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def center_crop_square(img: Image.Image, frac: float, y_bias: float) -> Image.Image:
    """Square crop of side ``frac * min(W,H)``, centered in X, biased up in Y.

    ``y_bias`` in [0,1] is the vertical position of the crop *center* as a
    fraction of the range where the crop still fits (0 = as high as possible,
    0.5 = geometric center, 1 = as low as possible). Portraits keep faces with a
    value < 0.5.
    """
    w, h = img.size
    side = int(round(frac * min(w, h)))
    side = max(1, min(side, w, h))

    x0 = (w - side) // 2                      # centered horizontally
    y0 = int(round((h - side) * y_bias))      # biased vertically
    y0 = max(0, min(y0, h - side))
    return img.crop((x0, y0, x0 + side, y0 + side))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, type=Path, help="flat dir of tile images")
    ap.add_argument("--out", required=True, type=Path, help="destination dir (created)")
    ap.add_argument("--frac", type=float, default=0.65, help="crop side as fraction of min(W,H)")
    ap.add_argument("--y-bias", type=float, default=0.35, help="vertical crop center (0=top,0.5=mid); <0.5 keeps faces")
    args = ap.parse_args()

    imgs = [p for p in sorted(args.inp.iterdir()) if p.suffix.lower() in EXTS]
    if not imgs:
        raise SystemExit(f"No images in {args.inp}")
    args.out.mkdir(parents=True, exist_ok=True)

    for p in imgs:
        img = Image.open(p).convert("RGB")
        cropped = center_crop_square(img, args.frac, args.y_bias)
        cropped = cropped.resize((TEX_SIZE, TEX_SIZE), Image.LANCZOS)
        cropped.save(args.out / f"{p.stem}.png")

    print(f"Done: cropped {len(imgs)} images (frac={args.frac}, y_bias={args.y_bias}) -> {args.out}")


if __name__ == "__main__":
    main()
