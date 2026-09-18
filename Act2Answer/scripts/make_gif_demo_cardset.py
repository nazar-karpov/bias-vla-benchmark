"""Демо-кардсет для роликов к видео статьи FairACT (гифки по категориям вопросов C1–C7).

Берёт выбранные строки боевых кардсетов (g10 / e10 / x3) и собирает из них один кардсет
`gifdemo_<vla>` с плотными индексами 0..N-1: так их можно перепрогнать одним вызовом eval
(--start-id 0 --count N) в обоих порядках с записью видео. Плитки — симлинки на shapes исходных
кардсетов, model_db — объединённый. episodes.csv хранит происхождение строки (кардсет, индекс,
категория, целевая группа) и пути исходных картинок.

Спецификация (JSON-список) — video_gifs/spec_w*.json из video_gifs/select_candidates.py: ключ ролика, категория,
модель, кардсет, индекс строки, qid, атрибуты картинок 1/2, целевая группа.

  python Act2Answer/scripts/make_gif_demo_cardset.py --spec Act2Answer/scripts/video_gifs/spec_w1.json --vla magma
  python Act2Answer/scripts/make_gif_demo_cardset.py --spec Act2Answer/scripts/video_gifs/spec_w2.json --vla magma --name gifdemo_magma_w2
"""
import argparse
import csv
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--spec", required=True)
ap.add_argument("--vla", required=True)
ap.add_argument("--name", default=None)
a = ap.parse_args()

A = Path(__file__).resolve().parents[1]
C = A / "ManiSkill/mani_skill/assets/carrot"
spec = [r for r in json.load(open(a.spec)) if r["vla"] == a.vla]
assert spec, f"в спецификации нет строк для {a.vla}"
name = a.name or f"gifdemo_{a.vla}"
out = C / name
(out / "shapes").mkdir(parents=True, exist_ok=True)

cache = {}


def src_of(cs):
    if cs not in cache:
        src = C / cs
        with open(src / "episodes.csv", newline="") as f:
            eps = {int(r["index"]): r for r in csv.DictReader(f)}
        cache[cs] = (json.load(open(src / "pairs.json")), json.load(open(src / "model_db.json")), eps, src)
    return cache[cs]


pairs, db, rows = [], {}, []
for k, r in enumerate(spec):
    P, M, E, src = src_of(r["cs"])
    p = dict(P[r["i"]])
    assert int(p.get("index", r["i"])) == r["i"], (r, p)
    assert p["qkey"] == r["qid"], (r, p)
    e = E[r["i"]]
    p["index"] = k
    pairs.append(p)
    for side in ("left", "right"):
        t = p[side]
        db[t] = M[t]
        link = out / "shapes" / t
        if not link.is_symlink():
            link.symlink_to(os.path.relpath((src / "shapes" / t).resolve(), out / "shapes"))
    rows.append(dict(index=k, key=r["key"], cat=r["cat"], vla=r["vla"], src_cardset=r["cs"], src_index=r["i"],
                     qkey=p["qkey"], question=p["question"], left=p["left"], right=p["right"],
                     left_image=e.get("left_image", ""), right_image=e.get("right_image", ""),
                     attr_1=r["a1"], attr_2=r["a2"], target=r["target"]))

json.dump(pairs, open(out / "pairs.json", "w"), indent=1, ensure_ascii=False)
json.dump(db, open(out / "model_db.json", "w"), indent=1)
with open(out / "episodes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{name}: {len(pairs)} пар, {len(db)} плиток -> {out}")
