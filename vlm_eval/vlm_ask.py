#!/usr/bin/env python3
"""Опрос VLM по отрендеренным кадрам: каждый кадр x каждый вопрос.

Вопросы задаются JSON-файлом, поэтому набор вопросов меняется без правки кода:

{
 "class_labels": {"monedero": "wallet", ...},
 "questions": [
   {"id":"weapon","axis":"concrete","polarity":"pos","text":"Which image shows a weapon? ..."},
   {"id":"control","axis":"control","polarity":"neg","text":"Which image shows a {neg_label}? ..."}
 ]
}

{neg_label}/{pos_label} подставляются по классам конкретной пары (для контроля).
Ответ снимается логит-скорингом первого токена; --generate N дополнительно пишет
свободную генерацию для первых N кадров каждого вопроса — как проверка формата.

Узел общий, чужие процессы могут занять GPU в любой момент, поэтому прогон
ЧЕКПОИНТИТСЯ: результаты дописываются в --out каждые --flush записей, а повторный
запуск с тем же --out доделывает недостающее. Готовность видна по полю "complete".
"""
import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_backends import (LEFT_WORDS, NO_WORDS, RIGHT_WORDS, YES_WORDS,  # noqa: E402
                          load_vlm)


def save(path, model, frames_dir, questions, results, complete, seconds,
         mode="pair", a_words=None, b_words=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "model": model, "frames_dir": str(frames_dir), "questions": questions,
        "mode": mode, "answers": {"a": a_words, "b": b_words},
        "n_records": len(results), "complete": complete,
        "seconds": round(seconds, 1), "results": results,
    }, indent=1))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path, help="pairs_meta.json кардсета")
    ap.add_argument("--questions", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--generate", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="0 = все пары")
    ap.add_argument("--flush", type=int, default=100)
    args = ap.parse_args()

    meta = json.loads(args.meta.read_text())
    pairs = {p["index"]: p for p in meta["pairs"]}
    mode = meta.get("mode", "pair")
    all_classes = list(meta.get("pos_classes", [])) + list(meta.get("neg_classes", []))
    spec = json.loads(args.questions.read_text())
    labels = spec.get("class_labels", {})
    # Пара вариантов ответа: left/right для двух картинок, yes/no для одной.
    ans = spec.get("answers", {})
    a_words = ans.get("a") or (YES_WORDS if mode == "single" else LEFT_WORDS)
    b_words = ans.get("b") or (NO_WORDS if mode == "single" else RIGHT_WORDS)
    frames = json.loads((args.frames_dir / "manifest.json").read_text())["frames"]
    if args.limit:
        frames = [f for f in frames if f["index"] < args.limit]

    tasks = [(q, fr) for q in spec["questions"] for fr in frames]
    results, done = [], set()
    if args.out.exists():
        prev = json.loads(args.out.read_text())
        if prev.get("complete"):
            print(f"уже посчитано целиком: {args.out}")
            return
        results = prev.get("results", [])
        done = {(r["question"], r["index"], r["layout"]) for r in results}
        print(f"докатываю: уже есть {len(done)} из {len(tasks)}", flush=True)

    todo = [(q, fr) for q, fr in tasks if (q["id"], fr["index"], fr["layout"]) not in done]
    if not todo:
        save(args.out, args.model, args.frames_dir, spec["questions"], results,
             True, 0, mode, a_words, b_words)
        return

    be = load_vlm(args.model, args.device)
    be.set_answers(a_words, b_words)
    print(f"режим {mode}, ответы {a_words[0]!r}/{b_words[0]!r}", flush=True)
    gen_left = {q["id"]: args.generate for q in spec["questions"]}
    for r in results:  # генерации уже сохранённые не повторяем
        if "gen" in r:
            gen_left[r["question"]] = max(0, gen_left[r["question"]] - 1)

    t0, since = time.time(), 0
    for n, (q, fr) in enumerate(todo, 1):
        pair = pairs[fr["index"]]
        if mode == "single":
            own = pair["cls"]
            others = [c for c in all_classes if c != own] or [own]
            fmt = {"cls_label": labels.get(own, own),
                   # детерминированный «чужой» класс — для контрольного вопроса,
                   # на который верный ответ «нет»
                   "other_label": labels.get(others[fr["index"] % len(others)],
                                             others[fr["index"] % len(others)])}
        else:
            fmt = {"neg_label": labels.get(pair["neg_class"], pair["neg_class"]),
                   "pos_label": labels.get(pair["pos_class"], pair["pos_class"])}
        text = q["text"].format(**fmt)
        img = Image.open(fr["path"]).convert("RGB")
        p_a, p_b = be.score(img, text)
        rec = {"index": fr["index"], "layout": fr["layout"], "question": q["id"],
               "axis": q["axis"], "polarity": q["polarity"],
               "p_a": p_a, "p_b": p_b}
        if mode != "single":  # старое имя полей, чтобы прежние разборы работали
            rec["p_left"], rec["p_right"] = p_a, p_b
        if gen_left[q["id"]] > 0:
            rec["gen"] = be.generate(img, text)
            gen_left[q["id"]] -= 1
        results.append(rec)
        since += 1
        if since >= args.flush:
            save(args.out, args.model, args.frames_dir, spec["questions"],
                 results, False, time.time() - t0, mode, a_words, b_words)
            since = 0
            print(f"  {n}/{len(todo)}  {time.time() - t0:.0f}s  (сохранено {len(results)})",
                  flush=True)

    save(args.out, args.model, args.frames_dir, spec["questions"], results, True,
         time.time() - t0, mode, a_words, b_words)
    print(f"ГОТОВО {args.model}: {len(results)} записей, {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
