#!/usr/bin/env python3
"""Logit lens по Magma-8B на кадрах симуляции: «текстовое представление рассуждения»
на каждом промежуточном слое перед ответом Left/Right.

Идея: hidden state слоя L прогоняем через ФИНАЛЬНЫЙ norm + lm_head (unembedding) →
распределение по словарю = «какие токены модель бы выдала, реши она отвечать прямо
на слое L». Это показывает поток мысли по глубине: когда рождается позиционный
токен Left/Right и всплывает ли где-то по пути признак (white/black/man/woman/
профессия). Иллюстрация к counterfactual-выводу: позиция появляется рано и держится,
признак в топ-токенах отсутствует → раса в «рассуждение» не входит.

Обязательная поправка logit lens: перед lm_head применяем финальный RMSNorm модели
(model.language_model.model.norm), иначе ранние слои дают мусор (масштаб не тот).

Вход и промпт — 1:1 как в magma_vlm_qa.py (тот же кадр симулятора, тот же VQA-вопрос,
Left/Right). Решающая позиция = последний токен промпта (там готовится 1-й токен ответа).

Запуск (env magma_act2answer, из SimplerEnv):
  CUDA_VISIBLE_DEVICES=0 python scripts/magma_logit_lens.py \
      --assets pairs_bias --ids 7 15 18 27 --swap 0 \
      --out outputs/magma_logit_lens.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

# переиспользуем весь пайплайн VLM-QA (загрузка, рендер кадров, промпт, парсинг)
from magma_vlm_qa import (
    load_magma, render_first_frames_chunked, make_vqa_question, parse_side,
    build_env_args,  # noqa: F401
)
from PIL import Image


def get_norm_and_head(model):
    """Финальный RMSNorm и lm_head (unembed) в LLaMA-стеке Magma."""
    lm_head = model.get_output_embeddings()  # надёжно возвращает lm_head Linear
    # финальный norm: пройти по обёрткам до LlamaModel.norm
    m = model
    for attr in ("language_model", "model"):
        if hasattr(m, attr):
            m = getattr(m, attr)
    norm = None
    for cand in (m, getattr(m, "model", None)):
        if cand is not None and hasattr(cand, "norm"):
            norm = cand.norm
            break
    if norm is None:
        raise RuntimeError("не нашёл финальный norm")
    return norm, lm_head


def tok_ids(processor, words):
    """id одного токена для слова (с ведущим пробелом — как в середине текста)."""
    out = {}
    for w in words:
        for variant in (" " + w, w, " " + w.capitalize(), w.capitalize()):
            ids = processor.tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                out[w] = ids[0]
                break
        else:
            # многотокенное слово — берём первый токен как прокси
            out[w] = processor.tokenizer.encode(" " + w, add_special_tokens=False)[0]
    return out


def build_inputs(model, processor, image, question, device):
    convs = [
        {"role": "system", "content": "You are agent that can see, talk and act."},
        {"role": "user", "content": f"<image>\n{question}"},
    ]
    prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
    if getattr(model.config, "mm_use_image_start_end", False):
        prompt = prompt.replace("<image>", "<image_start><image><image_end>")
    inputs = processor(images=image, texts=prompt, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)
    inputs["image_sizes"] = inputs["image_sizes"].unsqueeze(0)
    return inputs.to(device).to(torch.float16)


@torch.inference_mode()
def logit_lens_one(model, processor, norm, lm_head, image, question, device,
                   track_ids, topk=8):
    inputs = build_inputs(model, processor, image, question, device)
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple (n_layers+1) each (1, seq, hidden)
    per_layer = []
    track_curve = {w: [] for w in track_ids}
    for L, h in enumerate(hs):
        h_last = h[0, -1]                       # решающая позиция (последний токен промпта)
        logits = lm_head(norm(h_last.unsqueeze(0)))[0].float()  # (vocab,)
        probs = torch.softmax(logits, -1)
        top = torch.topk(probs, topk)
        toks = [processor.tokenizer.decode([int(i)]).strip() for i in top.indices]
        per_layer.append({
            "layer": L,
            "top_tokens": toks,
            "top_probs": [round(float(p), 4) for p in top.values],
        })
        for w, tid in track_ids.items():
            track_curve[w].append(round(float(probs[tid]), 5))
    # финальный ответ реально сгенерим (сверить с логит-линзой последнего слоя)
    return per_layer, track_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="pairs_bias")
    ap.add_argument("--ids", type=int, nargs="+", required=True,
                    help="индексы эпизодов (как в bias_detail), крупным планом")
    ap.add_argument("--swap", type=int, default=0, help="0=noswap раскладка, 1=swap")
    ap.add_argument("--asset-path", default=None)
    ap.add_argument("--episode-len", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.asset_path is None:
        import os
        repo = Path(os.environ.get("REPO_ROOT",
                    Path(__file__).resolve().parents[1] / "nazar_folder" / "Act2Answer"))
        args.asset_path = str(repo / "ManiSkill" / "mani_skill" / "assets" / "carrot")

    start_id = min(args.ids)
    count = max(args.ids) - start_id + 1
    do_swap = bool(args.swap)

    model, processor = load_magma(args.device)
    norm, lm_head = get_norm_and_head(model)
    track = tok_ids(processor, ["Left", "Right", "white", "black", "man", "woman"])
    print("tracked token ids:", track, flush=True)

    # рендерим кадры того же эпизода-диапазона (как magma_vlm_qa)
    ids, frames, instr = render_first_frames_chunked(
        args.assets, start_id, count, do_swap, args.asset_path,
        args.episode_len, args.seed, args.chunk,
    )
    id2frame = {i: f for i, f in zip(ids, frames)}
    id2instr = {i: q for i, q in zip(ids, instr)}

    results = []
    for ep in args.ids:
        if ep not in id2frame:
            print(f"ep {ep}: no frame (out of rendered range)", flush=True)
            continue
        img = Image.fromarray(id2frame[ep])
        q = make_vqa_question(id2instr[ep])
        per_layer, curve = logit_lens_one(model, processor, norm, lm_head,
                                          img, q, args.device, track, args.topk)
        final_tok = per_layer[-1]["top_tokens"][0]
        side = parse_side(final_tok) or parse_side(" ".join(per_layer[-1]["top_tokens"][:3]))
        results.append({
            "ep": ep, "swap": do_swap, "instruction": id2instr[ep],
            "final_top_token": final_tok, "parsed_side": side,
            "per_layer": per_layer, "track_curve": curve,
        })
        # компактный принт: слой -> top-3 + где Left/Right перегоняет признак
        print(f"\n===== ep {ep}  swap={do_swap}  final='{final_tok}' ({side}) =====", flush=True)
        print(f"  instruction: {id2instr[ep]}", flush=True)
        for pl in per_layer:
            if pl["layer"] % 4 == 0 or pl["layer"] >= len(per_layer) - 3:
                t3 = ", ".join(f"{tk}({p:.2f})" for tk, p in
                               zip(pl["top_tokens"][:4], pl["top_probs"][:4]))
                print(f"  L{pl['layer']:2}: {t3}", flush=True)
        # трек признака vs позиции
        lr_max = max(max(curve["Left"]), max(curve["Right"]))
        feat_max = max(max(curve["white"]), max(curve["black"]),
                       max(curve["man"]), max(curve["woman"]))
        print(f"  peak P(Left/Right)={lr_max:.3f}  vs  peak P(race/gender token)={feat_max:.4f}",
              flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} episodes -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
