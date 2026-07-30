#!/usr/bin/env python3
"""Парный выбор НА КАДРАХ СИМУЛЯЦИИ — тот же дизайн, что vlm_concat_choice.py.

Единственное отличие от concat-дизайна — ПОДАЧА: вместо PIL-склейки двух фото модель
видит первый кадр эпизода Act2Answer, где те же два изображения лежат плитками на
столе робота (рендер render_sim_choice_frames.py). Промпт НАМЕРЕННО идентичен
concat-версии — сравнение sim vs concat изолирует эффект сцены.

Контроли те же:
  - порядок ab/ba: ab = noswap-кадр (cand_a слева), ba = swap-кадр (плитки
    физически переставлены симулятором);
  - полярность pos/neg (boss/employee) — снимает остаточный крен.
Метрика: S = P(выбрал A | поз) − P(выбрал A | нег), позиция сокращается.

Объём: 200 эпизодов × 2 раскладки × 33 вопроса × 2 полярности = 26400 запросов.

Запуск на сервере (env под модель, симулятор НЕ нужен):
  python -u scripts/vlm_sim_choice.py --model magma \
    --frames-dir $REPO_ROOT/outputs/simframes_choice \
    --out $REPO_ROOT/outputs/simchoice-all-magma.json < /dev/null
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_single_yesno import load_backend  # noqa: E402
from vlm_posneg_yesno import pairs_map  # noqa: E402
from vlm_concat_choice import make_q, score_lr  # noqa: E402


def generate_answer(model, processor, image, question, build_inputs, max_new_tokens=6):
    """Кросс-проверка логит-скоринга ЖИВОЙ генерацией: тот же промпт, жадный decode.
    Возвращает (raw_text, side): side распарсен по словам left/right, None если модель
    ответила чем-то третьим — это и есть то, что логит-скоринг мог бы прятать."""
    inputs = build_inputs(image, question)
    tok = processor.tokenizer
    if getattr(model.generation_config, "pad_token_id", None) is None \
            and tok.pad_token_id is not None:
        model.generation_config.pad_token_id = tok.pad_token_id
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, num_beams=1,
                             max_new_tokens=max_new_tokens, use_cache=True)
    raw = tok.decode(out[0, inputs["input_ids"].shape[1]:],
                     skip_special_tokens=True).strip()
    t = raw.lower()
    has_l = bool(re.search(r"\bleft\b", t))
    has_r = bool(re.search(r"\bright\b", t))
    side = "left" if has_l and not has_r else ("right" if has_r and not has_l else None)
    return raw, side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("magma", "paligemma", "qwenbase"), required=True)
    ap.add_argument("--frames-dir", type=Path, required=True,
                    help="папка с manifest.json + noswap/ + swap/")
    ap.add_argument("--pairs-root", type=Path,
                    default=Path.home() / "bias_benchmark/datasets/PAIRS")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only-question", default=None,
                    help="подстроки через запятую: 'boss,wealthy,suburbs' оставит "
                         "все пары вопросов, где встречается любая из них")
    ap.add_argument("--gen-check", action="store_true",
                    help="дополнительно к логит-скорингу записать ЖИВОЙ жадный ответ "
                         "модели (gen_raw/gen_side) — проверка, что логиты не врут")
    ap.add_argument("--prompt-style", choices=("plain", "tiles"), default="plain",
                    help="plain: вопрос 1:1 как в конкате. tiles: с явной привязкой "
                         "к плиткам на столе («Two pictures of people lie on the "
                         "table...») — тест, помогает ли модели указание, куда смотреть")
    ap.add_argument("--limit-episodes", type=int, default=0, help="отладка")
    args = ap.parse_args()

    manifest = json.loads((args.frames_dir / "manifest.json").read_text())
    if args.limit_episodes:
        manifest = manifest[:args.limit_episodes]

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
    print(f"эпизодов: {len(manifest)}, вопросов: {len(allq)} -> "
          f"{len(manifest)*2*len(allq)*2} запросов", flush=True)

    model, processor, build_inputs = load_backend(args.model, args.device)

    results = []
    for ep in manifest:
        for order, sub in (("ab", "noswap"), ("ba", "swap")):
            fp = args.frames_dir / sub / f"ep{ep['index']}.png"
            img = Image.open(fp).convert("RGB")
            for pos, neg in allq:
                for attr, pol in ((pos, "pos"), (neg, "neg")):
                    q = make_q(attr)
                    if args.prompt_style == "tiles":
                        q = ("Two pictures of people lie on the table: one on the "
                             "left and one on the right. " + q)
                    p_left = score_lr(model, processor, img, q, args.device, build_inputs)
                    p_a = p_left if order == "ab" else 1 - p_left
                    rec = {
                        "index": ep["index"], "scene": ep["scene"], "axis": ep["axis"],
                        "cand_a": ep["cand_a"], "cand_b": ep["cand_b"],
                        "attribute": attr, "polarity": pol, "order": order,
                        "question": q, "p_left": p_left, "p_choose_a": p_a,
                    }
                    if args.gen_check:
                        rec["gen_raw"], rec["gen_side"] = generate_answer(
                            model, processor, img, q, build_inputs)
                    results.append(rec)
        print(f"  ep{ep['index']} {ep['scene']} {ep['cand_a']}|{ep['cand_b']} "
              f"готово ({len(results)})", flush=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nDone: {len(results)} -> {args.out}")


if __name__ == "__main__":
    main()
