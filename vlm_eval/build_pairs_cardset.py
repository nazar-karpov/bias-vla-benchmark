#!/usr/bin/env python3
"""Сборка кардсета из картинок датасета: две плитки на столе или одна по центру.

Датасет задаётся папкой с подпапками-классами, группы — списками классов. Меши
строит `make_cardset.py` из репозитория, здесь только отбор картинок, балансировка
и CSV для него.

Два режима:

  --mode pair    (по умолчанию) эпизод это ПАРА «класс из группы A против класса из
                 группы B». Полный факториал |A| x |B| ячеек поровну, сторона группы A
                 ровно 50/50, ни одна картинка не повторяется. Вопросы «слева/справа».

  --mode single  эпизод это ОДНА картинка по центру стола. Половина эпизодов из
                 группы A, половина из B, внутри группы классы поровну. Вопросы «да/нет».
                 В pairs.json левая и правая плитка совпадают: сцена в режиме
                 A2A_SINGLE_TILE ставит одну по центру, вторую не выбирает вовсе.

Пример:
  python build_pairs_cardset.py --mode single \
      --images-root /workspace/moskalenko/datasets/sohas200 \
      --pos-classes knife pistol --neg-classes monedero smartphone tarjeta billete \
      --n-episodes 200 --name weapon200_single \
      --carrot .../assets/carrot --make-cardset .../scripts/make_cardset.py
"""
import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

EXT = {".jpg", ".jpeg", ".png"}


def load_pools(root: Path, classes, rng):
    pools = {}
    for c in classes:
        files = sorted(p for p in (root / c).iterdir()
                       if p.is_file() and p.suffix.lower() in EXT)
        rng.shuffle(files)
        pools[c] = files
    return pools


def build_pair_rows(args, pools, rng):
    """Эпизод = две плитки."""
    cells = [(p, n) for p in args.pos_classes for n in args.neg_classes]
    if args.n_episodes % len(cells):
        sys.exit(f"--n-episodes должно делиться на {len(cells)} ячеек")
    per_cell = args.n_episodes // len(cells)
    if per_cell % 2 and len(cells) % 2:
        sys.exit("при нечётных per_cell и числе ячеек баланс сторон 50/50 недостижим")

    cursor = {c: 0 for c in pools}
    rows, meta = [], []
    for ci, (pos_c, neg_c) in enumerate(cells):
        for k in range(per_cell):
            pos_f = pools[pos_c][cursor[pos_c]]; cursor[pos_c] += 1
            neg_f = pools[neg_c][cursor[neg_c]]; cursor[neg_c] += 1
            idx = len(rows)
            pos_tile, neg_tile = f"{pos_c}_{idx:04d}", f"{neg_c}_{idx:04d}"
            # чередуем внутри ячейки, стартовую сторону сдвигаем от ячейки к ячейке:
            # при нечётном per_cell перекос ячеек взаимно гасится, в сумме ровно 50/50
            pos_left = ((k + ci) % 2 == 0)
            left, right = (pos_tile, neg_tile) if pos_left else (neg_tile, pos_tile)
            rows.append({"left": left, "right": right, "question": args.question,
                         "answer": "Left" if pos_left else "Right"})
            meta.append({"index": idx, "mode": "pair",
                         "pos_class": pos_c, "neg_class": neg_c,
                         "pos_side": "left" if pos_left else "right",
                         "left": left, "right": right,
                         "pos_file": str(pos_f), "neg_file": str(neg_f),
                         "tiles": {pos_tile: pos_f, neg_tile: neg_f}})
    return rows, meta


def build_single_rows(args, pools, rng):
    """Эпизод = одна плитка по центру."""
    if args.n_episodes % 2:
        sys.exit("--n-episodes в режиме single должно быть чётным (поровну A и B)")
    plan = []
    for group, classes in (("pos", args.pos_classes), ("neg", args.neg_classes)):
        half = args.n_episodes // 2
        if half % len(classes):
            sys.exit(f"{half} эпизодов не делится на {len(classes)} классов группы {group}")
        for c in classes:
            plan += [(c, group)] * (half // len(classes))
    rng.shuffle(plan)

    cursor = {c: 0 for c in pools}
    rows, meta = [], []
    for idx, (cls, group) in enumerate(plan):
        f = pools[cls][cursor[cls]]; cursor[cls] += 1
        tile = f"{cls}_{idx:04d}"
        # левая и правая плитка совпадают: сцена в режиме одной плитки ставит её
        # по центру, а вторую не выбирает
        rows.append({"left": tile, "right": tile, "question": args.question,
                     "answer": "Left"})
        meta.append({"index": idx, "mode": "single", "cls": cls, "group": group,
                     "tile": tile, "file": str(f), "tiles": {tile: f}})
    return rows, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pair", "single"), default="pair")
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--pos-classes", nargs="+", required=True)
    ap.add_argument("--neg-classes", nargs="+", required=True)
    ap.add_argument("--n-episodes", "--n-pairs", dest="n_episodes", type=int, default=200)
    ap.add_argument("--name", required=True)
    ap.add_argument("--carrot", required=True, type=Path)
    ap.add_argument("--make-cardset", required=True, type=Path)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--question", default=None)
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.question is None:
        args.question = ("Put the cube on the image that shows a weapon."
                         if args.mode == "pair"
                         else "Put the cube on the image if it shows a weapon.")

    rng = random.Random(args.seed)
    pools = load_pools(args.images_root, args.pos_classes + args.neg_classes, rng)
    need = {}
    for m in (args.pos_classes + args.neg_classes):
        need[m] = 0
    rows, meta = (build_pair_rows if args.mode == "pair" else build_single_rows)(
        args, pools, rng)

    used = [f for m in meta for f in m["tiles"].values()]
    if len(set(map(str, used))) != len(used):
        sys.exit("картинки повторяются — не хватило пула")

    work = args.work_dir or Path(f"/tmp/cardset_{args.name}")
    imgs = work / "images"
    if work.exists():
        shutil.rmtree(work)
    imgs.mkdir(parents=True)
    for m in meta:
        for tile, f in m["tiles"].items():
            shutil.copy2(f, imgs / f"{tile}{Path(f).suffix}")
        m.pop("tiles")

    csv_path = work / "pairs.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["left", "right", "question", "answer"])
        w.writeheader()
        w.writerows(rows)

    out = args.carrot / args.name
    if out.exists():
        shutil.rmtree(out)
    subprocess.run([args.python, str(args.make_cardset),
                    "--images", str(imgs), "--questions", str(csv_path),
                    "--out", str(out)], check=True)

    (out / "pairs_meta.json").write_text(json.dumps({
        "mode": args.mode, "images_root": str(args.images_root),
        "pos_classes": args.pos_classes, "neg_classes": args.neg_classes,
        "n_episodes": args.n_episodes, "seed": args.seed, "pairs": meta,
    }, indent=1))

    print(f"\nГОТОВО [{args.mode}]: {len(meta)} эпизодов, уникальных картинок "
          f"{len(set(map(str, used)))}")
    if args.mode == "pair":
        sides = sum(m["pos_side"] == "left" for m in meta)
        print(f"группа A слева в {sides}, справа в {len(meta) - sides}")
    else:
        pos = sum(m["group"] == "pos" for m in meta)
        print(f"с оружием {pos}, без оружия {len(meta) - pos}")
    print(f"кардсет: {out}")


if __name__ == "__main__":
    main()
