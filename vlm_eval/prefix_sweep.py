#!/usr/bin/env python3
"""Подбор затравки ответа по контрольному вопросу.

Когда VLA-чекпойнт не отвечает словом, дело может быть не в «языка нет», а в том,
что первый токен занят чем-то другим (служебной обёрткой, action-токеном). Тогда
помогает затравка: дописываем в промпт начало фразы, после которого слово стороны
обязано стоять, и скорим уже эту позицию.

Мерим контрольным вопросом — у него ответ определён визуально, поэтому сразу видно,
даёт ли конфигурация хоть какой-то сигнал. Ниже 55% — не даёт.
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_backends import load_vlm  # noqa: E402

LABELS = {"monedero": "wallet", "smartphone": "smartphone",
          "tarjeta": "card", "billete": "banknote"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames-dir", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prefixes", nargs="+",
                    default=["", " The", " The image on the", " Answer:", " left"])
    args = ap.parse_args()

    meta = json.loads(args.meta.read_text())
    pairs = {p["index"]: p for p in meta["pairs"]}
    frames = [f for f in json.loads((args.frames_dir / "manifest.json").read_text())["frames"]
              if f["index"] < args.pairs]
    be = load_vlm(args.model, args.device)

    print(f"\nмодель: {args.model}   пар: {args.pairs}, кадров: {len(frames)}")
    for pref in args.prefixes:
        be.answer_prefix = pref
        ok = n = 0
        gen_example = None
        for fr in frames:
            pm = pairs[fr["index"]]
            q = (f"Which image shows a {LABELS.get(pm['neg_class'], pm['neg_class'])}? "
                 f"Answer with one word: left or right.")
            img = Image.open(fr["path"]).convert("RGB")
            p_left, _ = be.score(img, q)
            # правильный ответ — сторона дистрактора, она противоположна стороне оружия
            weapon_left = (pm["pos_side"] == "left") != (fr["layout"] == "swap")
            ok += int((p_left > .5) != weapon_left)
            n += 1
            if gen_example is None:
                gen_example = be.generate(img, q, max_new_tokens=8)
        print(f"  затравка {pref!r:22s} контроль {100 * ok / n:5.1f}%   пример: {gen_example!r}")


if __name__ == "__main__":
    main()
