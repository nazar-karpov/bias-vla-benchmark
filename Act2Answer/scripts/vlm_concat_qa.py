#!/usr/bin/env python3
"""VLM-опрос на СКЛЕЕННЫХ исходных фото (без симулятора: ни стола, ни робота, ни
перспективы). Берём две исходные картинки людей из pairs_bias/imgs, конкатим бок о бок
и спрашиваем Magma, кто из двоих <профессия>. Каждую пару показываем в ОБОИХ порядках
(AB и BA) — позиция вычитается усреднением, остаётся чистый демографический сигнал.
Гоняем все 3 формата промпта (legacy/verbal/mcq), как на симуляторных кадрах, чтобы
сравнить: сохраняется ли формат-крен на чистой конкат-картинке.

Зачем: на симуляторном кадре правая плитка была физически КРУПНЕЕ/БЛИЖЕ (перспектива
стола) — возможный источник правого крена Magma. Тут обе фото РАВНОГО размера (256x256),
рядом, симметрично. Если крен остаётся — он точно про формат/данные, а не про сцену.

Позиция/сторона: в конкате LEFT-половина = первая картинка, RIGHT = вторая.
order='ab': left_tile=pair.left, right_tile=pair.right; 'ba' — наоборот (аналог swap).
parsed_side (Left/Right) — ФИЗИЧЕСКАЯ половина конката. Демография: сверяем, какой
КОНТЕНТ (pair.left/right) оказался на выбранной половине.

Запуск (env magma_act2answer; симулятор НЕ нужен, PYTHONPATH только для импорта скрипта):
  source ~/bias_benchmark/miniconda3/etc/profile.d/conda.sh
  conda activate ~/bias_benchmark/miniconda3/envs/magma_act2answer
  export REPO_ROOT=~/bias_benchmark/nazar_folder/Act2Answer
  CUDA_VISIBLE_DEVICES=0 python -u $REPO_ROOT/scripts/vlm_concat_qa.py \
    --imgs ~/bias_benchmark/nazar_folder/pairs_bias/imgs \
    --pairs $REPO_ROOT/ManiSkill/mani_skill/assets/carrot/pairs_bias/pairs.json \
    --prompt-format mcq --count 520 \
    --out $REPO_ROOT/outputs/vlm_mcq_ab/concat-mcq-magma-pairs_bias.json < /dev/null
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from magma_vlm_qa import (  # noqa: E402
    load_magma, magma_build_inputs, ask as magma_ask,
    make_vqa_question, make_verbal_question, make_legacy_verbal_question,
    letter_of_left_for, side_from_letter, score_ab, parse_side,
)
# PaliGemma импортируем ЛЕНИВО (внутри main при --model paligemma): в env
# magma_act2answer старый transformers без PaliGemmaForConditionalGeneration, и
# импорт на верхнем уровне ронял бы даже Magma-путь.


def pali_ask(model, processor, paligemma_build_inputs, image, question, device, max_new_tokens=20):
    inputs = paligemma_build_inputs(processor, image, question, device)
    in_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()


def concat_lr(img_left: Image.Image, img_right: Image.Image, gap: int = 8) -> Image.Image:
    """Склеить две картинки бок о бок с небольшим белым зазором (чтобы модель видела
    их как ДВЕ раздельные, а не одну панораму)."""
    h = max(img_left.height, img_right.height)
    w = img_left.width + gap + img_right.width
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(img_left, (0, (h - img_left.height) // 2))
    canvas.paste(img_right, (img_left.width + gap, (h - img_right.height) // 2))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True, type=Path, help="папка с исходными PNG")
    ap.add_argument("--pairs", required=True, type=Path, help="pairs.json")
    ap.add_argument("--count", type=int, default=520)
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--prompt-format", choices=("mcq", "verbal", "legacy"), default="mcq")
    ap.add_argument("--verbal-first", choices=("Left", "Right"), default="Left")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", choices=("magma", "paligemma"), default="magma")
    ap.add_argument("--gap", type=int, default=8)
    ap.add_argument("--img-ext", default="png",
                    help="расширение картинок, если в манифесте нет orig_file/edited_file")
    ap.add_argument("--save-frames", type=Path, default=None)
    args = ap.parse_args()

    pairs = json.loads(args.pairs.read_text())
    end = len(pairs) if args.count <= 0 else min(args.start_id + args.count, len(pairs))
    ids = list(range(args.start_id, end))

    if args.model == "paligemma":
        # ленивый импорт — только когда реально нужна PaliGemma (см. коммент выше)
        from paligemma_vlm_qa import (  # noqa: E402
            load_paligemma, paligemma_build_inputs, MODEL_NAME as PALI_MODEL,
        )
        model, processor = load_paligemma(args.device, PALI_MODEL)
        def build_inputs(image, question):
            return paligemma_build_inputs(processor, image, question, args.device)
        def ask_fn(model, processor, image, question, device, max_new_tokens=20):
            return pali_ask(model, processor, paligemma_build_inputs,
                            image, question, device, max_new_tokens)
    else:
        model, processor = load_magma(args.device)
        ask_fn = magma_ask
        def build_inputs(image, question):
            return magma_build_inputs(model, processor, image, question, args.device)

    text_mode = args.prompt_format in ("verbal", "legacy")
    print(f"prompt-format={args.prompt_format}  concat gap={args.gap}  n={len(ids)}", flush=True)

    def img_path(pair, which):
        # приоритет явного имени файла из манифеста (safeedit: orig_file/edited_file),
        # иначе <left|right>.<ext> (pairs_bias: png). which='left'|'right'.
        fkey = "orig_file" if which == "left" else "edited_file"
        if fkey in pair:
            return args.imgs / pair[fkey]
        return args.imgs / f"{pair[which]}.{args.img_ext}"

    results = []
    for k, e in enumerate(ids):
        pair = pairs[e]
        lp = img_path(pair, "left")
        rp = img_path(pair, "right")
        if not (lp.exists() and rp.exists()):
            print(f"  skip ep{e}: missing {lp.name if not lp.exists() else rp.name}", flush=True)
            continue
        img_l = Image.open(lp).convert("RGB")
        img_r = Image.open(rp).convert("RGB")
        lol = letter_of_left_for(e)
        if args.prompt_format == "verbal":
            vqa_q = make_verbal_question(pair["question"], args.verbal_first)
        elif args.prompt_format == "legacy":
            vqa_q = make_legacy_verbal_question(pair["question"])
        else:
            vqa_q = make_vqa_question(pair["question"], lol)

        # order 'ab' = как в паре; 'ba' = свап половин (позиция вычитается)
        for order in ("ab", "ba"):
            if order == "ab":
                left_key, right_key = pair["left"], pair["right"]
                frame = concat_lr(img_l, img_r, args.gap)
            else:
                left_key, right_key = pair["right"], pair["left"]
                frame = concat_lr(img_r, img_l, args.gap)

            if args.save_frames and k < 4:
                args.save_frames.mkdir(parents=True, exist_ok=True)
                frame.save(args.save_frames / f"ep{e}_{order}.png")

            if text_mode:
                raw = ask_fn(model, processor, frame, vqa_q, args.device)
                side = parse_side(raw)
                rec = {"index": e, "order": order, "question": vqa_q,
                       "left": left_key, "right": right_key,
                       "raw_answer": raw, "parsed_side": side}
                info = f"{side!s:5} raw={raw[:24]!r}"
            else:
                letter, la, lb = score_ab(model, processor, frame, vqa_q, args.device, build_inputs)
                side = side_from_letter(letter, lol)
                rec = {"index": e, "order": order, "question": vqa_q,
                       "left": left_key, "right": right_key,
                       "letter_of_left": lol, "chosen_letter": letter,
                       "logit_a": la, "logit_b": lb,
                       "raw_answer": letter, "parsed_side": side}
                info = f"side={side!s:5} letter={letter} (Lp{la:.2f}/{lb:.2f} left={lol})"
            results.append(rec)
            print(f"[{k+1}/{len(ids)}] ep{e} {order}: {info}", flush=True)
        if (k + 1) % 10 == 0:
            args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    n_parsed = sum(1 for r in results if r["parsed_side"])
    print(f"\nDone: {len(results)} answers, parsed={n_parsed}/{len(results)}")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
