"""Контактные листы кандидатов для отбора глазами: по строке на пару, 4 кадра — noswap (отпускание+2,
последний кадр) и swap (то же). Подпись: кадр отпускания, сторона при отпускании, strict в конце,
дрейф плиток до отпускания.

  python sheets.py --run magma_w2 --key C2_poor_magma [--all]   -> sheets/<run>__<key>.png
"""
import argparse, os
import pandas as pd, imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--key", required=True)
    ap.add_argument("--all", action="store_true", help="все пары, а не только «оба порядка -> цель»")
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(HERE, "outcomes.csv")); w = pd.read_csv(os.path.join(HERE, "outcomes_pairs.csv"))
    sub = w[(w.run == a.run) & (w.key == a.key)]
    if not a.all:
        sub = sub[sub.both_target]
    try:
        F = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        F = ImageFont.load_default()
    rows = []
    for r in sub.itertuples():
        row = Image.new("RGB", (1280, 262), (30, 30, 30)); d = ImageDraw.Draw(row)
        e0 = df[(df.run == a.run) & (df.idx == r.idx) & (df.order == "noswap")].iloc[0]
        d.text((4, 3), f"idx {r.idx} | -> {r.target} | {e0.cs}:{e0.src_i} | {e0.img1.split('original/')[-1]}  vs  {e0.img2.split('original/')[-1]}",
               fill=(255, 255, 255), font=F)
        for k, order in enumerate(("noswap", "swap")):
            e = df[(df.run == a.run) & (df.idx == r.idx) & (df.order == order)].iloc[0]
            v = iio.imread(e.video)
            for j, fi in enumerate((int(e.rel_frame) + 2, len(v) - 1)):
                im = Image.fromarray(v[min(max(fi, 0), len(v) - 1)]).resize((320, 240))
                if j == 0:
                    ImageDraw.Draw(im).text((4, 4), f"{order} rel@{e.rel_frame} side{e.side_rel} pick={e.pick_grp or '-'} strict{e.strict} drift{e.drift_rel_mm:.0f}",
                                            fill=(255, 255, 0), font=F)
                row.paste(im, ((2 * k + j) * 320, 22))
        rows.append(row)
    os.makedirs(os.path.join(HERE, "sheets"), exist_ok=True)
    out = os.path.join(HERE, "sheets", f"{a.run}__{a.key}.png")
    if rows:
        S = Image.new("RGB", (1280, 262 * len(rows)))
        for k, t in enumerate(rows):
            S.paste(t, (0, k * 262))
        S.save(out)
    print(out, len(rows))


if __name__ == "__main__":
    main()
