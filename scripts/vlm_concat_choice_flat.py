#!/usr/bin/env python3
"""Парный конкат-выбор на ПЛОСКОЙ папке картинок (стемы make_cardset:
occupations__airplane__white_man.png) — для КРОПНУТЫХ фото PAIRS.

Тот же дизайн, что vlm_concat_choice.py (простой вопрос + порядок ab/ba +
полярность), но картинки берутся не из PAIRS/data/<scene>/<demo>.png, а из
одной папки со стемами. Нужен для теста «усиливает ли кроп фото эффект в
сильном (конкат) дизайне»: baseline = choice-all-*.json на некропнутых фото.
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_single_yesno import load_backend  # noqa: E402
from vlm_posneg_yesno import pairs_map  # noqa: E402
from vlm_concat_choice import concat_lr, make_q, score_lr, PAIR_TYPES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("magma", "paligemma", "qwenbase"), required=True)
    ap.add_argument("--imgs", type=Path, required=True, help="плоская папка стемов")
    ap.add_argument("--pairs-root", type=Path,
                    default=Path.home() / "bias_benchmark/datasets/PAIRS")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only-question", default=None, help="подстроки через запятую")
    ap.add_argument("--gap", type=int, default=8)
    args = ap.parse_args()

    pm = pairs_map(args.pairs_root / "pairs.json")
    allq = []
    for v in pm.values():
        for qp in v:
            if qp not in allq:
                allq.append(qp)
    if args.only_question:
        subs = [s.strip().lower() for s in args.only_question.split(",") if s.strip()]
        allq = [(p, n) for p, n in allq
                if any(s in p.lower() or s in n.lower() for s in subs)]
        print("вопросы после фильтра:", allq, flush=True)
    print(f"сцен: {len(pm)}, вопросов: {len(allq)} -> "
          f"{len(allq)*len(pm)*len(PAIR_TYPES)*4} запросов", flush=True)

    model, processor, build_inputs = load_backend(args.model, args.device)

    results = []
    for scene in sorted(pm):
        stem = scene.replace("/", "__")
        for axis, da, db in PAIR_TYPES:
            pa, pb = args.imgs / f"{stem}__{da}.png", args.imgs / f"{stem}__{db}.png"
            if not pa.exists() or not pb.exists():
                print(f"  skip {scene} {da}|{db}", flush=True)
                continue
            ia, ib = Image.open(pa).convert("RGB"), Image.open(pb).convert("RGB")
            for pos, neg in allq:
                for attr, pol in ((pos, "pos"), (neg, "neg")):
                    q = make_q(attr)
                    for order in ("ab", "ba"):
                        img = concat_lr(ia, ib, args.gap) if order == "ab" \
                            else concat_lr(ib, ia, args.gap)
                        p_left = score_lr(model, processor, img, q, args.device,
                                          build_inputs)
                        p_a = p_left if order == "ab" else 1 - p_left
                        results.append({
                            "scene": scene, "axis": axis, "cand_a": da, "cand_b": db,
                            "attribute": attr, "polarity": pol, "order": order,
                            "question": q, "p_left": p_left, "p_choose_a": p_a,
                        })
        print(f"  {scene} готово ({len(results)})", flush=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nDone: {len(results)} -> {args.out}")


if __name__ == "__main__":
    main()
