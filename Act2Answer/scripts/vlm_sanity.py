#!/usr/bin/env python3
"""VLM sanity-тест «в вакууме»: PaliGemma на N эпизодах pairs_bias × {noswap,swap},
но вместо спорного «кто пилот» задаём НАБОР ПРОСТЫХ вопросов с ОБЪЕКТИВНО верным
ответом — чтобы понять, тупит ли модель на самом восприятии сцены или только на
формате/сложной формулировке bias-вопроса.

Сцена (проверено глазами на кадре): деревянный стол, робо-схват по центру держит
БЕЛЫЙ кубик, и ДВЕ плитки-картинки лежат слева и справа (правая крупнее/ближе к
камере — перспективная асимметрия). После swap плитки МЕНЯЮТСЯ содержимым, но
геометрия (где какая, размер) та же.

Вопросы и эталон:
  count   "How many people are in the image?"            -> 2
  cube    "What color is the cube? Answer one word."     -> white
  cubepos "Where is the white cube: left, center, or right?" -> center
  pilotL/R как в основном бенче — для контекста (нет объективной истины).

Ответы берём СВОБОДНОЙ генерацией (видно сырой текст модели), плюс печатаем.
Цель — глазами увидеть, что модель реально «понимает» на кадре.

Запуск (env spatialvla_act2answer, нужен HF_TOKEN):
  source ~/bias_benchmark/miniconda3/etc/profile.d/conda.sh
  conda activate ~/bias_benchmark/miniconda3/envs/spatialvla_act2answer
  export REPO_ROOT=~/bias_benchmark/nazar_folder/Act2Answer HF_TOKEN=...
  export PYTHONPATH=$REPO_ROOT/SimplerEnv:$REPO_ROOT/ManiSkill
  cd $REPO_ROOT/SimplerEnv
  CUDA_VISIBLE_DEVICES=1 python -u $REPO_ROOT/scripts/vlm_sanity.py \
      --count 25 --out $REPO_ROOT/outputs/vlm_sanity_pali.json < /dev/null
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from magma_vlm_qa import render_first_frames_chunked  # noqa: E402
from paligemma_vlm_qa import load_paligemma, paligemma_build_inputs, MODEL_NAME  # noqa: E402

# Простые вопросы с объективной истиной (истину печатаем рядом для глазной проверки).
SANITY_Q = [
    ("count",   "How many people are in the image?"),
    ("cube",    "What color is the cube? Answer in one word."),
    ("cubepos", "Where is the white cube: left, center or right?"),
    ("faces",   "How many faces do you see?"),
    ("desc",    "Describe the image in one short sentence."),
]


def gen(model, processor, image, question, device, max_new_tokens=25):
    inputs = paligemma_build_inputs(processor, image, question, device)
    in_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="pairs_bias")
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--count", type=int, default=25)
    ap.add_argument("--episode-len", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--asset-path", default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--render-chunk", type=int, default=25)
    args = ap.parse_args()

    if args.asset_path is None:
        repo = Path(os.environ.get("REPO_ROOT",
                    Path(__file__).resolve().parents[1] / "nazar_folder" / "Act2Answer"))
        args.asset_path = str(repo / "ManiSkill" / "mani_skill" / "assets" / "carrot")

    pairs = json.loads((Path(args.asset_path) / args.assets / "pairs.json").read_text())

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

    model, processor = load_paligemma(args.device, args.model)

    results = []
    for k, e in enumerate(ids_ref):
        pair = pairs[e]
        for do_swap in (False, True):
            frame = Image.fromarray(frames_by_swap[do_swap][k]).convert("RGB")
            answers = {}
            for key, q in SANITY_Q:
                answers[key] = gen(model, processor, frame, q, args.device)
            rec = {"index": e, "swap": do_swap, "answers": answers,
                   "left": pair["right"] if do_swap else pair["left"],
                   "right": pair["left"] if do_swap else pair["right"]}
            results.append(rec)
            tag = "swap  " if do_swap else "noswap"
            print(f"[{k+1}/{len(ids_ref)}] ep{e} {tag}: "
                  f"count={answers['count']!r} cube={answers['cube']!r} "
                  f"pos={answers['cubepos']!r} faces={answers['faces']!r}", flush=True)
            print(f"        desc={answers['desc']!r}", flush=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nDone: {len(results)} records. Saved -> {args.out}")


if __name__ == "__main__":
    main()
