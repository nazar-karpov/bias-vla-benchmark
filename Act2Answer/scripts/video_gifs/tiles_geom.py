"""Где плитки в кадре камеры (640×480) для раскладки программ g10/e10/x3: BOARD_XY_SCALE=1.2, центры
плиток (x=-0.25, y=±0.14). Нужна композитору, чтобы обвести выбранную плитку.

Плитки неподвижны, а картинки на них в разных эпизодах разные, поэтому пиксели плиток = места, где
первые кадры разных эпизодов сильнее всего отличаются. Правая плитка видна целиком (4 угла), у левой
дальний внутренний угол закрыт кубом (3 угла). По 7 соответствиям «плоскость стола -> пиксель»
строим гомографию и проецируем точные четырёхугольники.

⚠ Реальный размер плитки = меш 0.1452 м × 1.2 = 0.174 м. model_db даёт bbox 0.11 м, и по нему считаются
зоны ответа (hard ±0.066, soft ±0.096 от центра). С размером 0.132 гомография не сходится (невязка до
30 px), с 0.174 — ≤3 px. То есть soft-зона шире видимой плитки лишь на ~1 см, а hard-зона на 2 см уже её.

  python tiles_geom.py --outputs <Act2Answer/outputs>   -> tile_geom.json (+ tile_geom_check.png)
"""
import argparse, glob, json, os
import numpy as np, imageio.v3 as iio
from scipy import ndimage
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
X0, Y0 = -0.25, 0.14
TILE = 0.1452 * 1.2                       # меш textured.glb/collision.obj × BOARD_XY_SCALE
SOFT, HARD = 0.096, 0.066                 # зоны ответа из кода среды (model_db bbox 0.11 × 1.2 / 2 [+ 0.03])


def visvalingam(poly, k):
    pts = [tuple(p) for p in poly]
    while len(pts) > k:
        areas = [abs(np.cross(np.subtract(pts[i], pts[i - 1]), np.subtract(pts[(i + 1) % len(pts)], pts[i - 1]))) / 2
                 for i in range(len(pts))]
        pts.pop(int(np.argmin(areas)))
    return np.array(pts, float)


def fit_h(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y, -u], [0, 0, 0, x, y, 1, -v * x, -v * y, -v]]
    Hm = np.linalg.svd(np.array(A))[2][-1].reshape(3, 3)
    return Hm / Hm[2, 2]


def proj(Hm, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ Hm.T
    return p[:, :2] / p[:, 2:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=os.path.join(HERE, "..", "..", "outputs"))
    a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(a.outputs, "gif-*", "glob", "vis_0_test", "video_*.mp4")))[:200]
    F0 = np.stack([iio.imread(f, index=0) for f in fs]).astype(np.float32)
    sd = F0.std(axis=0).mean(axis=2)
    lab, n = ndimage.label(ndimage.binary_opening(sd > 18, iterations=2))
    sizes = ndimage.sum(lab > 0, lab, range(1, n + 1))
    hulls = []
    for c in np.argsort(sizes)[::-1][:2] + 1:
        ys, xs = np.where(lab == c); pts = np.c_[xs, ys].astype(float)
        hulls.append((xs.mean(), pts[ConvexHull(pts).vertices]))
    (_, hl), (_, hr) = sorted(hulls, key=lambda t: t[0])
    R = visvalingam(hr, 4)
    c = R.mean(0); R = R[np.argsort(np.arctan2(R[:, 1] - c[1], R[:, 0] - c[0]))]   # TL, TR, BR, BL на экране
    L5 = visvalingam(hl, 5)                                                          # внешн.-дальний ... внешн.-ближний
    h = TILE / 2
    xf, xn, yi, yo = X0 - h, X0 + h, Y0 - h, Y0 + h
    world = [(xf, yi), (xf, yo), (xn, yo), (xn, yi), (xf, -yo), (xn, -yo), (xn, -yi)]
    img = [R[0], R[1], R[2], R[3], L5[0], L5[4], L5[3]]
    Hm = fit_h(world, img)
    res = np.linalg.norm(proj(Hm, world) - np.array(img), axis=1)
    zone = lambda yc, half: proj(Hm, [(X0 - half, yc - half), (X0 - half, yc + half), (X0 + half, yc + half), (X0 + half, yc - half)]).tolist()
    geo = dict(H=Hm.tolist(), tile_size_m=TILE, residual_px=res.round(2).tolist(),
               quadL=proj(Hm, [(xf, -yo), (xf, -yi), (xn, -yi), (xn, -yo)]).tolist(),
               quadR=proj(Hm, [(xf, yi), (xf, yo), (xn, yo), (xn, yi)]).tolist(),
               softL=zone(-Y0, SOFT), softR=zone(Y0, SOFT), hardL=zone(-Y0, HARD), hardR=zone(Y0, HARD),
               note="мир (x,y) плоскости стола -> пиксель кадра 640x480; ориентация x условна (четырёхугольники от неё не зависят)")
    json.dump(geo, open(os.path.join(HERE, "tile_geom.json"), "w"), indent=1)
    im = Image.fromarray(F0[0].astype(np.uint8)); d = ImageDraw.Draw(im)
    for k, col in (("quadL", "lime"), ("quadR", "lime"), ("softL", "yellow"), ("softR", "yellow"), ("hardL", "red"), ("hardR", "red")):
        d.polygon([tuple(p) for p in geo[k]], outline=col)
    im.save(os.path.join(HERE, "tile_geom_check.png"))
    print("эпизодов", len(fs), "невязка, px:", res.round(2))


if __name__ == "__main__":
    main()
