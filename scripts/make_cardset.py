#!/usr/bin/env python3
"""Generate an Act2Answer card set from your own images.

An Act2Answer "card" is a flat textured tile (~14.5x14.5x0.6 cm box) whose top
(+Z) face carries an image. The robot answers a question by placing the pre-grasped
cube on the tile it believes is correct. This script builds a drop-in asset set:

    carrot/<name>/
      pairs.json        # one entry per episode: {index,left,right,question,answer}
      model_db.json     # per-tile geometry (bbox / density / scale)
      shapes/<tile>/
        textured.glb    # box mesh, top face UV-mapped to your PNG (PBR baseColorTexture)
        collision.obj   # convex box, identical extents

Geometry is copied *exactly* from the reference tile in test_colors:
  - box extents      = 0.1452 x 0.1452 x 0.006 m  (half = 0.0726 / 0.0726 / 0.003)
  - top face (+Z) UV : u along +X, v along +Y, full 0..1 (no mirroring)
  - model_db bbox    : +-0.055 / +-0.055 / +-0.003  (scoring zone, intentionally
                       smaller than the mesh; matches the reference set)
  - density 6368, scale 1.0

Input:
  --images DIR    folder of PNG/JPG cards (filename stem = tile name used in CSV)
  --questions CSV columns: left,right,question,answer   (answer = Left|Right)
                          left/right are image filename stems (without extension)
  --out    DIR    destination carrot/<name> directory (created)

Example CSV:
  left,right,question,answer
  cat,dog,Put cube on the animal that barks,Right
  apple,car,Put cube on the food,Left
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

# --- exact reference geometry (measured from test_colors/tile_0) ---------------
HALF_XY = 0.0726      # mesh half-extent in X and Y (m)
HALF_Z = 0.003        # mesh half-thickness (m)
BBOX_HALF_XY = 0.055  # model_db scoring half-extent (smaller than mesh, per reference)
BBOX_HALF_Z = 0.003
DENSITY = 6368
SCALE = 1.0
TEX_SIZE = 512        # reference textures are 512x512


def build_box_mesh(image_path: Path) -> trimesh.Trimesh:
    """A 12-triangle box with the top (+Z) face UV-mapped to `image_path`.

    trimesh.creation.box gives a unit box with per-face vertices already suitable
    for texturing; we scale it to the reference extents, then assign UVs so the top
    face samples the whole image with u along +X and v along +Y (matching the
    reference tile exactly). Other faces sample corners of the image; only the top
    face is ever seen by the camera.
    """
    mesh = trimesh.creation.box(extents=(2 * HALF_XY, 2 * HALF_XY, 2 * HALF_Z))
    V = mesh.vertices
    zmax = V[:, 2].max()
    # Map every vertex: u = (x+HALF)/(2*HALF), v = (y+HALF)/(2*HALF).
    # For the top face this yields the exact reference mapping; side/bottom faces
    # are never visible to the eval camera so their mapping is irrelevant.
    u = (V[:, 0] + HALF_XY) / (2 * HALF_XY)
    v = (V[:, 1] + HALF_XY) / (2 * HALF_XY)
    uv = np.column_stack([u, v])

    img = Image.open(image_path).convert("RGB")
    if img.size != (TEX_SIZE, TEX_SIZE):
        img = img.resize((TEX_SIZE, TEX_SIZE), Image.LANCZOS)

    material = trimesh.visual.material.PBRMaterial(baseColorTexture=img)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    assert abs(zmax - HALF_Z) < 1e-9, "unexpected top-face Z"
    return mesh


def build_collision_obj(path: Path) -> None:
    """Write the exact convex box collision mesh from the reference set."""
    text = """# Act2Answer card collision box (convex)
o card_collision
v -0.07260000 0.07260000 -0.00300000
v -0.07260000 0.07260000 0.00300000
v 0.07260000 0.07260000 -0.00300000
v -0.07260000 -0.07260000 -0.00300000
v -0.07260000 -0.07260000 0.00300000
v 0.07260000 0.07260000 0.00300000
v 0.07260000 -0.07260000 -0.00300000
v 0.07260000 -0.07260000 0.00300000
f 1 2 6
f 1 6 3
f 1 3 7
f 1 7 4
f 1 4 5
f 1 5 2
f 2 5 8
f 2 8 6
f 3 6 8
f 3 8 7
f 4 7 8
f 4 8 5
"""
    path.write_text(text)


def model_db_entry(name: str) -> dict:
    return {
        "name": name,
        "sign": name,
        "bbox": {
            "min": [-BBOX_HALF_XY, -BBOX_HALF_XY, -BBOX_HALF_Z],
            "max": [BBOX_HALF_XY, BBOX_HALF_XY, BBOX_HALF_Z],
        },
        "scales": [SCALE],
        "density": DENSITY,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path, help="folder of card images (PNG/JPG)")
    ap.add_argument("--questions", required=True, type=Path, help="CSV: left,right,question,answer")
    ap.add_argument("--out", required=True, type=Path, help="destination carrot/<name> dir")
    args = ap.parse_args()

    # index images by filename stem
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    imgs = {p.stem: p for p in sorted(args.images.iterdir()) if p.suffix.lower() in exts}
    if not imgs:
        raise SystemExit(f"No images found in {args.images}")

    with args.questions.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {args.questions}")

    # collect the set of tiles actually referenced
    used = []
    for r in rows:
        for side in ("left", "right"):
            stem = r[side].strip()
            if stem not in imgs:
                raise SystemExit(f"CSV references '{stem}' but no matching image in {args.images}")
            if stem not in used:
                used.append(stem)

    shapes_dir = args.out / "shapes"
    shapes_dir.mkdir(parents=True, exist_ok=True)

    # build one tile per referenced image
    model_db = {}
    for stem in used:
        tdir = shapes_dir / stem
        tdir.mkdir(exist_ok=True)
        mesh = build_box_mesh(imgs[stem])
        mesh.export(tdir / "textured.glb")
        build_collision_obj(tdir / "collision.obj")
        model_db[stem] = model_db_entry(stem)
        print(f"[tile] {stem}: {imgs[stem].name} -> {tdir}")

    (args.out / "model_db.json").write_text(json.dumps(model_db, indent=2))

    # build pairs.json
    pairs = []
    for i, r in enumerate(rows):
        ans = r["answer"].strip().capitalize()
        if ans not in ("Left", "Right"):
            raise SystemExit(f"row {i}: answer must be Left/Right, got {r['answer']!r}")
        pairs.append({
            "index": i,
            "left": r["left"].strip(),
            "right": r["right"].strip(),
            "question": r["question"].strip(),
            "answer": ans,
        })
    (args.out / "pairs.json").write_text(json.dumps(pairs, indent=2))

    print(f"\nDone: {len(pairs)} episodes, {len(used)} tiles -> {args.out}")
    print(f"Run with:  --assets {args.out.name}  (place under ManiSkill/mani_skill/assets/carrot/)")


if __name__ == "__main__":
    main()
