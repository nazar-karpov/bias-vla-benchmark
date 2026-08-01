#!/usr/bin/env python3
"""Линейный пробинг, фаза A: извлечение фич из VLM-бэкбона на confirm-сценах.

Для каждого доступного эпизода (noswap+swap) берём кадр 0 из видео VLA-прогона
(= то, что видел робот), прогоняем через VLM с инструкцией, снимаем hidden state
последнего токена на нескольких слоях + генерим текстовый выбор Left/Right.
Метки (какая сторона A=white/man, сцена, VLA-выбор) — из pairs.json + stats.yaml.

Usage:
  python vlm_probe_extract.py --model magma --out outputs/probe_magma.npz [--limit N]
"""
import argparse, glob, json, os, re, sys
import numpy as np
import torch
import cv2
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

A2A = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = os.path.join(A2A, "ManiSkill/mani_skill/assets/carrot/pairs_choice_vla_confirm/pairs.json")
OUT_DIR = os.path.join(A2A, "outputs")

LAYERS = [8, 16, 24, 32]  # индексы в hidden_states (0=эмбеддинги, 1..32=слои Magma-8B)


def build_manifest():
    """ep -> {layout -> (video_path, vla_side)} + метки."""
    pairs = json.load(open(PAIRS))
    vids, vla = {}, {}
    for layout in ("noswap", "swap"):
        vids[layout], vla[layout] = {}, {}
        for st in glob.glob(os.path.join(OUT_DIR, f"confirm-internvla-FULL-{layout}-s*",
                                         "glob", "vis_0_test", "stats.yaml")):
            d = os.path.dirname(st)
            shard_dir = os.path.dirname(os.path.dirname(d))
            m = re.search(r"-s(\d+)$", shard_dir)
            if not m:
                continue
            start = int(m.group(1))
            li = (yaml.safe_load(open(st)) or {}).get("last_info") or {}
            for idx, info in li.items():
                ep = start + int(idx)
                v = glob.glob(os.path.join(d, f"video_{int(idx)}-s_*.mp4"))
                if v:
                    vids[layout][ep] = v[0]
                vla[layout][ep] = int(info.get("chosen_side", 0) or 0)

    samples = []
    for ep, meta in enumerate(pairs):
        if "qkey" not in meta:
            continue
        ltoks, rtoks = meta["left"].split("__"), meta["right"].split("__")
        scene = "__".join(ltoks[:-1])          # категория__сцена
        l_dem, r_dem = ltoks[-1], rtoks[-1]    # напр. white_man / black_man
        if meta["axis"] == "race":
            a_left_logical = 1 if l_dem.startswith("white") else 0
        else:
            a_left_logical = 1 if l_dem.endswith("_man") else 0
        for layout in ("noswap", "swap"):
            if ep not in vids[layout]:
                continue
            a_left = a_left_logical if layout == "noswap" else 1 - a_left_logical
            samples.append(dict(ep=ep, layout=layout, video=vids[layout][ep],
                                scene=scene, qkey=meta["qkey"], axis=meta["axis"],
                                polarity=meta["polarity"], question=meta["question"],
                                a_left=a_left, vla_side=vla[layout].get(ep, 0)))
    return samples


def first_frame(path):
    cap = cv2.VideoCapture(path)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    return Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="magma", choices=["magma", "qwen"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    samples = build_manifest()
    if a.limit:
        samples = samples[: a.limit]
    print(f"samples: {len(samples)}", flush=True)

    if a.model == "magma":
        from magma_vlm_qa import load_magma, magma_build_inputs
        model, processor = load_magma(a.device)

        def get_feats_and_answer(img, question):
            q = question + " Which tile is correct, left or right? Answer with one word: Left or Right."
            inputs = magma_build_inputs(model, processor, img, q, a.device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                hs = [out.hidden_states[i][0, -1, :].float().cpu().numpy() for i in LAYERS]
                gen = model.generate(**inputs, max_new_tokens=5, do_sample=False,
                                     pad_token_id=processor.tokenizer.pad_token_id)
            txt = processor.tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:],
                                             skip_special_tokens=True).strip().lower()
            ans = 1 if txt.startswith("left") else (2 if txt.startswith("right") else 0)
            return np.stack(hs), ans
    else:  # qwen2.5-vl
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        name = "Qwen/Qwen2.5-VL-3B-Instruct"
        processor = AutoProcessor.from_pretrained(name)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map=a.device,
            attn_implementation="flash_attention_2").eval()
        n_layers = model.config.num_hidden_layers
        layers = [max(1, n_layers // 4), n_layers // 2, 3 * n_layers // 4, n_layers]
        print("qwen layers:", layers, flush=True)

        def get_feats_and_answer(img, question):
            q = question + " Which tile is correct, left or right? Answer with one word: Left or Right."
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": q}]}]
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], return_tensors="pt").to(a.device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                hs = [out.hidden_states[i][0, -1, :].float().cpu().numpy() for i in layers]
                gen = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            txt = processor.tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:],
                                             skip_special_tokens=True).strip().lower()
            ans = 1 if txt.startswith("left") else (2 if txt.startswith("right") else 0)
            return np.stack(hs), ans

    feats, keep = [], []
    for i, s in enumerate(samples):
        img = first_frame(s["video"])
        if img is None:
            continue
        try:
            h, ans = get_feats_and_answer(img, s["question"])
        except Exception as e:
            print("ERR", s["ep"], s["layout"], repr(e)[:120], flush=True)
            continue
        s2 = {k: v for k, v in s.items() if k != "video"}
        s2["vlm_side"] = ans
        feats.append(h.astype(np.float16))
        keep.append(s2)
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(samples)}", flush=True)

    np.savez_compressed(a.out, feats=np.stack(feats),
                        meta=json.dumps(keep), layers=json.dumps(LAYERS if a.model == "magma" else layers))
    print(f"SAVED {a.out} feats={len(keep)}", flush=True)


if __name__ == "__main__":
    main()
