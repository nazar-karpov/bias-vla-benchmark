#!/usr/bin/env python3
"""Извлечь послойные активации Magma-8B на одиночных картинках pairs_bias —
вход для линейного пробинга (scripts/magma_probe.py).

Идея: поведенческий bias (Left/Right) забит позиционным prior, но информация о
расе/поле/профессии может быть ЛИНЕЙНО декодируема во внутренних активациях, даже
если модель не выдаёт её в ответе. Пробинг покажет «о чём Magma думает», а не «что
говорит».

Что делаем: для каждой из 200 размеченных картинок (по одному человеку, метки
race/gender/trait из имени файла + pairs.json) прогоняем Magma с НЕЙТРАЛЬНЫМ
промптом ("Describe this person." — не left/right-вопрос, чтобы активации отражали
восприятие человека, а не задачу выбора). Снимаем hidden_states всех 33 точек
(эмбеддинги + 32 слоя LLaMA). Представление картинки на слое L = среднее hidden
state по позициям ИЗОБРАЖЕНЧЕСКИХ токенов (image-плейсхолдеры в последовательности).
Вектор 4096 на слой на картинку.

Выход: acts.npz с X[L] shape (N, 4096) для L=0..32 + метки race/gender/trait/category
и список имён файлов. Дальше scripts/magma_probe.py обучает логрег по каждому слою.

Запуск (env magma_act2answer, из SimplerEnv):
  CUDA_VISIBLE_DEVICES=0 python scripts/magma_extract_acts.py \
      --imgs ~/bias_benchmark/nazar_folder/pairs_bias/imgs \
      --out  outputs/magma_acts.npz
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_NAME = "microsoft/Magma-8B"
NEUTRAL_PROMPT = "Describe this person."


def fix_vision_layerscale(model):
    """Обязательный remap LayerScale (см. magma_vlm_qa.py) — иначе vision-башня
    коллапсирует и активации бессмысленны."""
    snap = snapshot_download(MODEL_NAME, local_files_only=True)
    model_keys = set(model.state_dict().keys())
    remap = {}
    for fp in glob.glob(os.path.join(snap, "*.safetensors")):
        with safe_open(fp, framework="pt") as st:
            for k in st.keys():
                if "vision_tower" not in k:
                    continue
                nk = k[:-6] + "gamma" if re.search(r"blocks\.\d+\.weight$", k) else k
                if nk in model_keys:
                    remap[nk] = st.get_tensor(k)
    if remap:
        model.load_state_dict(remap, strict=False)
    print(f"LayerScale remap: {len(remap)} vision weights", flush=True)


def label_from_name(stem, trait_by_stem):
    race = "white" if "white" in stem else ("black" if "black" in stem else None)
    gender = "woman" if "woman" in stem else ("man" if "man" in stem else None)
    category = stem.split("__")[0]
    trait = trait_by_stem.get(stem)
    return race, gender, category, trait


def build_trait_map(pairs_path):
    """stem -> trait (профессия/статус), из pairs.json. Один stem может встречаться
    в нескольких парах с ОДНИМ trait (same subfolder) — берём первый."""
    pairs = json.loads(Path(pairs_path).read_text())
    m = {}
    for p in pairs:
        t = re.search(r"probably (?:a |an )?(.+)$", p["question"])
        trait = t.group(1) if t else p["question"]
        for side in ("left", "right"):
            m.setdefault(p[side], trait)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True, type=Path)
    ap.add_argument("--pairs", default=None, help="pairs.json для меток trait")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prompt", default=NEUTRAL_PROMPT)
    args = ap.parse_args()

    if args.pairs is None:
        repo = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[1] / "nazar_folder" / "Act2Answer"))
        args.pairs = repo / "ManiSkill" / "mani_skill" / "assets" / "carrot" / "pairs_bias" / "pairs.json"

    trait_by_stem = build_trait_map(args.pairs)

    exts = {".png", ".jpg", ".jpeg"}
    files = sorted([p for p in args.imgs.iterdir() if p.suffix.lower() in exts])
    print(f"{len(files)} images", flush=True)

    print(f"Loading {MODEL_NAME} ...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map=args.device, low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2", torch_dtype=torch.float16,
        trust_remote_code=True,
    ).eval()
    fix_vision_layerscale(model)
    img_token_idx = model.config.image_token_index
    n_layers = model.config.text_config.num_hidden_layers if hasattr(model.config, "text_config") else 32
    print(f"image_token_index={img_token_idx}, layers={n_layers}", flush=True)

    # prompt (одинаковый для всех, картинка-плейсхолдер как в magma_vlm_qa)
    convs = [
        {"role": "system", "content": "You are agent that can see, talk and act."},
        {"role": "user", "content": f"<image>\n{args.prompt}"},
    ]
    base_prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
    if getattr(model.config, "mm_use_image_start_end", False):
        base_prompt = base_prompt.replace("<image>", "<image_start><image><image_end>")

    N = len(files)
    n_pts = n_layers + 1  # embeddings + each layer
    hidden = model.config.hidden_size if hasattr(model.config, "hidden_size") else 4096
    X = [np.zeros((N, hidden), dtype=np.float32) for _ in range(n_pts)]       # mean-pool
    Xlast = [np.zeros((N, hidden), dtype=np.float32) for _ in range(n_pts)]   # last-token
    races, genders, cats, traits, names = [], [], [], [], []

    for i, f in enumerate(files):
        img = Image.open(f).convert("RGB")
        inputs = processor(images=img, texts=base_prompt, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)
        inputs["image_sizes"] = inputs["image_sizes"].unsqueeze(0)
        inputs = inputs.to(args.device).to(torch.float16)
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple len n_layers+1, each (1, seq_expanded, hidden)
        # ВАЖНО: Magma РАЗВОРАЧИВАЕТ один <image>-плейсхолдер в ~256 визуальных токенов
        # внутри forward, поэтому seq у hidden_states длиннее input_ids и маска по
        # input_ids не сходится по форме. Берём представление = среднее по ВСЕЙ
        # последовательности (whole-sequence mean-pool) — робастно к форме и захватывает
        # и визуальные, и текстовые токены. Дополнительно кладём last-token (сводка перед
        # генерацией) как второй набор — пробинг обучаем на обоих, берём лучший.
        for L, h in enumerate(hs):
            h0 = h[0]  # (seq_expanded, hidden)
            X[L][i] = h0.mean(0).float().cpu().numpy()
            Xlast[L][i] = h0[-1].float().cpu().numpy()
        r, g, c, t = label_from_name(f.stem, trait_by_stem)
        races.append(r); genders.append(g); cats.append(c); traits.append(t); names.append(f.stem)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{N}", flush=True)

    np.savez_compressed(
        args.out,
        **{f"L{L}": X[L] for L in range(n_pts)},
        **{f"last_L{L}": Xlast[L] for L in range(n_pts)},
        race=np.array(races), gender=np.array(genders),
        category=np.array(cats), trait=np.array(traits), name=np.array(names),
        n_layers=n_pts,
    )
    print(f"Saved {n_pts} layers x {N} images -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
