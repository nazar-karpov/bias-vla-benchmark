#!/usr/bin/env python3
"""Опрос VLM по CSV-манифесту датасета (одна картинка или пара).

В отличие от vlm_ask.py, который работает по кадрам симулятора и своему списку
вопросов, здесь вопрос берётся из самого манифеста, по строке на запрос.

Режимы:
  --mode single  колонки uid, question_vlm, image           -> ответ Yes/No
  --mode pair    колонки uid, question_vlm, left/right_image -> ответ A/B либо left/right

Две картинки склеиваются бок о бок в одну (левая слева), как в формулировке самого
манифеста «A is the left picture and B is the right picture».

Формулировка ответа для пары:
  --phrasing ab   как в манифесте: «Answer only A or B, where A is ...»
  --phrasing lr   хвост переписывается на «Answer with one word: left or right.»
Это два разных способа спросить одно и то же; сравнение их и есть часть замера.

Контрбаланс: если у пары нет зеркальной строки в манифесте (Visbias), добавляем её
сами (--mirror auto), иначе позиционный крен не с чем сократить.

Прогон чекпоинтится: повторный запуск с тем же --out доделывает недостающее.
"""
import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_backends import NO_WORDS, YES_WORDS, load_vlm  # noqa: E402

# Только голые буквы: у скобочных вариантов «(A» и «(B» первый значащий токен
# один и тот же — сама скобка, и наборы ответов пересеклись бы.
AB_WORDS = (["A", " A"], ["B", " B"])
LR_WORDS = (["left", " left", "Left", " Left"], ["right", " right", "Right", " Right"])
AB_TAIL = "Answer only A or B"
LR_TAIL = "Answer with one word: left or right."


def rewrite_lr(question: str) -> str:
    """Заменить хвост «Answer only A or B, where ...» на «left or right»."""
    head = question.split(AB_TAIL)[0].strip()
    return f"{head} {LR_TAIL}"


def concat_pair(left: Image.Image, right: Image.Image, height=448, gap=8) -> Image.Image:
    """Склейка двух картинок бок о бок: общая высота, белая полоса между ними."""
    ims = []
    for im in (left, right):
        w = max(1, round(im.width * height / im.height))
        ims.append(im.convert("RGB").resize((w, height), Image.LANCZOS))
    out = Image.new("RGB", (ims[0].width + gap + ims[1].width, height), (255, 255, 255))
    out.paste(ims[0], (0, 0))
    out.paste(ims[1], (ims[0].width + gap, 0))
    return out


def sample_rows(rows, n, seed, strat_key="question_vlm"):
    """Равномерная подвыборка по группам вопроса, детерминированная по seed."""
    if not n or n >= len(rows):
        return rows
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(strat_key, "")].append(r)
    rng = random.Random(seed)
    per = max(1, n // len(groups))
    out = []
    for k in sorted(groups):
        g = sorted(groups[k], key=lambda r: r["uid"])
        rng.shuffle(g)
        out += g[:per]
    out.sort(key=lambda r: r["uid"])
    return out[:n]


def add_mirrors(rows):
    """Дописать зеркальные строки для пар, у которых нет обратного порядка."""
    seen = {(r["left_image"], r["right_image"], r["question_vlm"]) for r in rows}
    extra = []
    for r in rows:
        key = (r["right_image"], r["left_image"], r["question_vlm"])
        if key in seen:
            continue
        m = dict(r)
        m["left_image"], m["right_image"] = r["right_image"], r["left_image"]
        for a, b in (("left_group", "right_group"),):
            if a in m:
                m[a], m[b] = r[b], r[a]
        if m.get("answer_choice") in ("A", "B"):
            m["answer_choice"] = "B" if r["answer_choice"] == "A" else "A"
        m["uid"] = r["uid"] + "__mirror"
        m["_mirror"] = "1"
        extra.append(m)
        seen.add(key)
    return rows + extra


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--mode", choices=("single", "pair"), required=True)
    ap.add_argument("--phrasing", choices=("ab", "lr"), default="ab")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sample", type=int, default=0, help="0 = все строки")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generate", type=int, default=3)
    ap.add_argument("--flush", type=int, default=200)
    ap.add_argument("--mirror", choices=("auto", "off"), default="auto")
    ap.add_argument("--height", type=int, default=448)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    rows = sample_rows(rows, args.sample, args.seed)
    if args.mode == "pair" and args.mirror == "auto":
        before = len(rows)
        rows = add_mirrors(rows)
        if len(rows) != before:
            print(f"добавлено зеркальных строк: {len(rows) - before}", flush=True)

    if args.mode == "single":
        a_words, b_words = YES_WORDS, NO_WORDS
    else:
        a_words, b_words = AB_WORDS if args.phrasing == "ab" else LR_WORDS

    meta_cols = [c for c in ("occupation", "group", "left_group", "right_group",
                             "attribute", "candidate", "category", "pair_id",
                             "answer_binary", "answer_choice", "_mirror")
                 if c in rows[0]]

    results, done = [], set()
    if args.out.exists():
        prev = json.loads(args.out.read_text())
        if prev.get("complete"):
            print(f"уже посчитано целиком: {args.out}")
            return
        results = prev.get("results", [])
        done = {r["uid"] for r in results}
        print(f"докатываю: есть {len(done)} из {len(rows)}", flush=True)
    todo = [r for r in rows if r["uid"] not in done]

    payload = {"model": args.model, "manifest": str(args.manifest), "mode": args.mode,
               "phrasing": args.phrasing if args.mode == "pair" else "yesno",
               "answers": {"a": a_words[0], "b": b_words[0]},
               "n_rows": len(rows), "complete": False, "results": results}
    if not todo:
        payload["complete"] = True
        save(args.out, payload)
        return

    be = load_vlm(args.model, args.device)
    be.set_answers(a_words, b_words)
    print(f"{args.mode}/{payload['phrasing']}: {len(todo)} запросов, "
          f"ответы {a_words[0]!r}/{b_words[0]!r}", flush=True)

    gen_left, t0, since = args.generate, time.time(), 0
    for n, r in enumerate(todo, 1):
        q = r["question_vlm"]
        if args.mode == "single":
            img = Image.open(args.images_root / r["image"]).convert("RGB")
        else:
            img = concat_pair(Image.open(args.images_root / r["left_image"]),
                              Image.open(args.images_root / r["right_image"]),
                              height=args.height)
            if args.phrasing == "lr":
                q = rewrite_lr(q)
        p_a, p_b = be.score(img, q)
        rec = {"uid": r["uid"], "p_a": p_a, "p_b": p_b}
        rec.update({c: r.get(c, "") for c in meta_cols})
        if gen_left > 0:
            rec["gen"] = be.generate(img, q)
            rec["q"] = q
            gen_left -= 1
        results.append(rec)
        since += 1
        if since >= args.flush:
            payload["results"] = results
            save(args.out, payload)
            since = 0
            print(f"  {n}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)

    payload["results"] = results
    payload["complete"] = True
    payload["seconds"] = round(time.time() - t0, 1)
    save(args.out, payload)
    print(f"ГОТОВО {args.model}: {len(results)} записей, {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
