#!/usr/bin/env python3
"""Convert datasets/PAIRS/pairs.json (one image + positive/negative statements)
into a left,right,question,answer CSV usable by Act2Answer's make_cardset.py.

Each source subfolder (e.g. potential_crime/courtroom) has exactly 4 images:
white_man, white_woman, black_man, black_woman (some occupations folders use
"white_man1" as the stem -> normalized to "white_man"). For every subfolder +
question-pair found in pairs.json, build same-axis 2AFC pairs:
  - gender axis: white_man vs white_woman, black_man vs black_woman
  - race axis:   white_man vs black_man, white_woman vs black_woman
using the "positive" statement as the question text. `answer` alternates
Left/Right (seeded) since these are subjective bias probes with no ground
truth; the eval's swap pass measures positional bias independently.

Tile names in the output CSV are made globally unique by prefixing with the
subfolder path (slashes -> double-underscore), since e.g. "black_man" repeats
across many subfolders with different photos.
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

AXES = [
    ("white_man", "white_woman"),
    ("black_man", "black_woman"),
    ("white_man", "black_man"),
    ("white_woman", "black_woman"),
]


def normalize_stem(stem: str) -> str:
    if stem == "white_man1":
        return "white_man"
    return stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-json", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path, help="PAIRS/data root (for resolving image files)")
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--out-images", required=True, type=Path, help="flat dir to copy/symlink renamed tiles into")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    lines = args.pairs_json.read_text(encoding="utf-8").splitlines()
    entries = [json.loads(l) for l in lines if l.strip()]

    # group by (subfolder, positive, negative) -> {stem: image_rel_path}
    groups = defaultdict(dict)
    for e in entries:
        rel = Path(e["image"])
        subfolder = str(rel.parent)
        stem = normalize_stem(rel.stem)
        key = (subfolder, e["positive"], e["negative"])
        groups[key][stem] = rel

    args.out_images.mkdir(parents=True, exist_ok=True)
    rows = []
    used_tiles = {}  # tile_name -> source path

    for (subfolder, positive, negative), stems in sorted(groups.items()):
        missing = [s for s in ("white_man", "white_woman", "black_man", "black_woman") if s not in stems]
        if missing:
            print(f"skip {subfolder} ({positive[:30]}...): missing {missing}")
            continue
        tag = subfolder.replace("/", "__")
        for a, b in AXES:
            tile_a = f"{tag}__{a}"
            tile_b = f"{tag}__{b}"
            used_tiles[tile_a] = stems[a]
            used_tiles[tile_b] = stems[b]
            answer = "Left" if rng.random() < 0.5 else "Right"
            rows.append({"left": tile_a, "right": tile_b, "question": positive, "answer": answer})

    # materialize flat image dir (copy)
    import shutil
    for tile_name, rel_path in used_tiles.items():
        src = args.data_root / rel_path
        dst = args.out_images / f"{tile_name}{src.suffix}"
        if not dst.exists():
            shutil.copy(src, dst)

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["left", "right", "question", "answer"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone: {len(rows)} episodes, {len(used_tiles)} tiles")
    print(f"CSV: {args.out_csv}")
    print(f"Images: {args.out_images}")


if __name__ == "__main__":
    main()
