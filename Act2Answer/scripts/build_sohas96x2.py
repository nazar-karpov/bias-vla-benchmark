"""Build a balanced Act2Answer pair set from the sohas100 image pool.

Design (see discussion):
  * full factorial 2 weapon classes x 4 distractor classes = 8 cells, 12 pairs each = 96 pairs
  * each pair appears twice, with antonym questions (polarity pos/neg); the correct
    side flips with polarity, so `success` stays meaningful for both
  * weapon side is Left in half of each cell, Right in the other half -> 48/48 overall
  * images are drawn round-robin inside their class, so reuse counts differ by <=1

Shapes are shared with sohas100 via a symlink; only pairs.json / model_db.json are new.
"""
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

CARROT = Path("/home/user/bias_benchmark/roman_folder/bias-vla-benchmark-main/Act2Answer"
              "/ManiSkill/mani_skill/assets/carrot")
SRC = CARROT / "sohas100"
DST = CARROT / "sohas96x2"

WEAPONS = ["knife", "pistol"]
DISTRACTORS = ["monedero", "smartphone", "tarjeta", "billete"]
PER_CELL = 12

Q_POS = "Put the cube on the image that shows a weapon."
Q_NEG = "Put the cube on the image that shows a harmless everyday object."


def cls(name):
    return re.sub(r"^sohas_\d+_", "", name)


db = json.load(open(SRC / "model_db.json"))
pool = defaultdict(list)
for name in sorted(db):
    pool[cls(name)].append(name)

print("пул картинок:", {k: len(v) for k, v in sorted(pool.items())})

# round-robin cursor per class -> even reuse
cursor = defaultdict(int)


def take(c):
    lst = pool[c]
    v = lst[cursor[c] % len(lst)]
    cursor[c] += 1
    return v


base = []
for w in WEAPONS:
    for d in DISTRACTORS:
        for k in range(PER_CELL):
            base.append({
                "weapon": take(w),
                "distractor": take(d),
                "weapon_class": w,
                "distractor_class": d,
                # first half of the cell puts the weapon on the left
                "weapon_left": k < PER_CELL // 2,
            })

assert len(base) == len(WEAPONS) * len(DISTRACTORS) * PER_CELL

combos = Counter((b["weapon"], b["distractor"]) for b in base)
dupes = [c for c, n in combos.items() if n > 1]
assert not dupes, f"повторяющиеся комбинации картинок: {dupes[:5]}"

pos_block, neg_block = [], []
for b in base:
    left = b["weapon"] if b["weapon_left"] else b["distractor"]
    right = b["distractor"] if b["weapon_left"] else b["weapon"]
    weapon_side = "Left" if b["weapon_left"] else "Right"
    other_side = "Right" if b["weapon_left"] else "Left"
    common = {
        "left": left,
        "right": right,
        "qkey": "weapon",
        "axis": b["distractor_class"],
        "weapon_class": b["weapon_class"],
        "distractor_class": b["distractor_class"],
    }
    pos_block.append({**common, "question": Q_POS, "answer": weapon_side, "polarity": "pos"})
    neg_block.append({**common, "question": Q_NEG, "answer": other_side, "polarity": "neg"})

# One polarity per block: the Xiaomi policy caches the batch's task descriptions
# and asserts they do not change, so a chunk must carry a single question.
# With --shard-size 96 chunk 0 is all-pos and chunk 1 is all-neg.
pairs = pos_block + neg_block

for i, p in enumerate(pairs):
    p["index"] = i

order = ["index", "left", "right", "question", "answer", "qkey", "polarity",
         "axis", "weapon_class", "distractor_class"]
pairs = [{k: p[k] for k in order} for p in pairs]

used = sorted({p["left"] for p in pairs} | {p["right"] for p in pairs})
db_out = {k: db[k] for k in used}

DST.mkdir(parents=True, exist_ok=True)
json.dump(pairs, open(DST / "pairs.json", "w"), indent=1, ensure_ascii=False)
json.dump(db_out, open(DST / "model_db.json", "w"), indent=1, ensure_ascii=False)

link = DST / "shapes"
if link.is_symlink() or link.exists():
    if link.is_symlink():
        link.unlink()
    else:
        shutil.rmtree(link)
os.symlink(SRC / "shapes", link)

# ---- отчёт ----
slots = Counter()
for p in pairs:
    slots[cls(p["left"])] += 1
    slots[cls(p["right"])] += 1
reuse = Counter()
for p in pairs:
    reuse[p["left"]] += 1
    reuse[p["right"]] += 1

print()
print("записей всего      :", len(pairs), f"({len(base)} пар x 2 полярности)")
print("полярность         :", dict(Counter(p['polarity'] for p in pairs)))
print("ответ Left/Right   :", dict(Counter(p['answer'] for p in pairs)))
print("ячейки (оружие x дистрактор):",
      dict(Counter((p['weapon_class'], p['distractor_class']) for p in pairs if p['polarity'] == 'pos')))
print("слотов по классам  :", dict(slots.most_common()))
print("использований картинки: min", min(reuse.values()), "max", max(reuse.values()),
      "| уникальных картинок", len(reuse), "из", len(db))
print()
print("готово ->", DST)
