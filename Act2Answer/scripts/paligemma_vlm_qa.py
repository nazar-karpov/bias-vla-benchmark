#!/usr/bin/env python3
"""VLM-опрос базовой PaliGemma2 на тех же кадрах pairs_bias, что и magma_vlm_qa.py.

Зачем: SpatialVLA (robotics-SFT из PaliGemma) НЕ умеет отвечать текстом — на любой
вопрос выдаёт только action-токены (проверено). Чтобы всё же сравнить "мозг"
SpatialVLA-семейства с Magma в текстовом VLM-режиме, берём БАЗОВУЮ PaliGemma2-3b-mix
(та же основа Gemma2+SigLIP, из которой выросла SpatialVLA, но до robotics-SFT и с
сохранённым VQA-режимом). Это же даёт «до/после»: PaliGemma (до) vs SpatialVLA-VLA
(после robotics-обучения).

Кадры и разбор ответов — те же, что в magma_vlm_qa.py (импортируем оттуда:
render_first_frames_chunked, make_vqa_question, parse_side). Отличается только
модель: PaliGemma processor(text=<prompt>, images=<img>) + .generate().

PaliGemma-mix отвечает на вопрос напрямую (instruction-tuned под VQA); префикс —
просто сам вопрос (без <image>-плейсхолдера, процессор добавляет image-токены сам).

Запуск (env spatialvla_act2answer — там transformers 4.47 с PaliGemma2; нужен HF_TOKEN,
модель gated):
  source ~/bias_benchmark/miniconda3/etc/profile.d/conda.sh
  conda activate ~/bias_benchmark/miniconda3/envs/spatialvla_act2answer
  export REPO_ROOT=~/bias_benchmark/nazar_folder/Act2Answer HF_TOKEN=...
  export PYTHONPATH=$REPO_ROOT/SimplerEnv:$REPO_ROOT/ManiSkill
  cd $REPO_ROOT/SimplerEnv
  CUDA_VISIBLE_DEVICES=1 python -u $REPO_ROOT/scripts/paligemma_vlm_qa.py \
      --assets pairs_bias --count 55 \
      --out $REPO_ROOT/outputs/paligemma_vlm_qa_pairs_bias.json < /dev/null

Выход: тот же формат JSON, что magma_vlm_qa (index, swap, question, left, right,
raw_answer, parsed_side) — чтобы напрямую скормить scripts/compare_vlm_vla.py и
per-trait анализу.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# переиспользуем рендер кадров и разбор из соседнего скрипта
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from magma_vlm_qa import (  # noqa: E402
    render_first_frames_chunked, make_vqa_question,
    letter_of_left_for, side_from_letter, score_ab,
)

MODEL_NAME = "google/paligemma2-3b-mix-224"


def load_paligemma(device: str, model_name: str):
    print(f"Loading {model_name} ...", flush=True)
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    processor = AutoProcessor.from_pretrained(model_name, token=tok)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, token=tok,
    ).eval().to(device)
    print("PaliGemma loaded.", flush=True)
    return model, processor


def paligemma_build_inputs(processor, image, question, device):
    # PaliGemma-mix: prefix = вопрос; процессор сам ставит image-токены и <bos>.
    inputs = processor(text=question, images=image, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    return inputs


def ask(model, processor, image: Image.Image, question: str, device: str, max_new_tokens=20) -> str:
    inputs = paligemma_build_inputs(processor, image, question, device)
    in_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="pairs_bias")
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--count", type=int, default=55)
    ap.add_argument("--episode-len", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--asset-path", default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--render-chunk", type=int, default=55)
    ap.add_argument("--save-frames", type=Path, default=None)
    args = ap.parse_args()

    if args.asset_path is None:
        repo = Path(os.environ.get("REPO_ROOT",
                    Path(__file__).resolve().parents[1] / "nazar_folder" / "Act2Answer"))
        args.asset_path = str(repo / "ManiSkill" / "mani_skill" / "assets" / "carrot")

    pairs = json.loads((Path(args.asset_path) / args.assets / "pairs.json").read_text())

    # 1) кадры для обеих раскладок ДО загрузки модели
    frames_by_swap = {}
    ids_ref = None
    for do_swap in (False, True):
        ids, frames, _ = render_first_frames_chunked(
            args.assets, args.start_id, args.count, do_swap,
            args.asset_path, args.episode_len, args.seed, args.render_chunk,
        )
        frames_by_swap[do_swap] = frames
        ids_ref = ids
        print(f"rendered {len(frames)} first-frames (swap={do_swap})", flush=True)
        if args.save_frames:
            d = args.save_frames / ("swap" if do_swap else "noswap")
            d.mkdir(parents=True, exist_ok=True)
            for k, e in enumerate(ids):
                Image.fromarray(frames[k]).save(d / f"ep{e}.png")

    # 2) PaliGemma
    model, processor = load_paligemma(args.device, args.model)

    def build_inputs(image, question):
        return paligemma_build_inputs(processor, image, question, args.device)

    results = []
    for k, e in enumerate(ids_ref):
        pair = pairs[e]
        lol = letter_of_left_for(e)
        vqa_q = make_vqa_question(pair["question"], lol)
        for do_swap in (False, True):
            left_key = pair["right"] if do_swap else pair["left"]
            right_key = pair["left"] if do_swap else pair["right"]
            frame = Image.fromarray(frames_by_swap[do_swap][k]).convert("RGB")
            letter, la, lb = score_ab(model, processor, frame, vqa_q, args.device,
                                      build_inputs)
            side = side_from_letter(letter, lol)
            results.append({
                "index": e, "swap": do_swap, "question": vqa_q,
                "left": left_key, "right": right_key,
                "letter_of_left": lol, "chosen_letter": letter,
                "logit_a": la, "logit_b": lb,
                "raw_answer": letter, "parsed_side": side,
            })
            tag = "swap  " if do_swap else "noswap"
            print(f"[{k+1}/{len(ids_ref)}] ep{e} {tag}: side={side!s:5} "
                  f"letter={letter} (Lp{la:.2f}/{lb:.2f} left={lol})", flush=True)
        if (k + 1) % 5 == 0:
            args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    n_parsed = sum(1 for r in results if r["parsed_side"])
    print(f"\nDone: {len(results)} answers ({len(ids_ref)} eps x2), parsed={n_parsed}/{len(results)}")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
