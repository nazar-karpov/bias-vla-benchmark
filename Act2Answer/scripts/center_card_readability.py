#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Читается ли карточка в ЦЕНТРАЛЬНОМ слоте: точность Magma по полу/расе.

Геометрия (center_card_probe.py) показала: карточка в центре видна ЦЕЛИКОМ и на
33% крупнее, чем в боковом слоте, но рука со схваченным кубом съедает ~12%
площади — ровно ту четверть, где на фото лицо. Увести лицо из-под руки можно
двумя способами: подвинуть карточку или развернуть её (A2A_SINGLE_TILE_YAW).
Разворот делает фото «лежащим на боку» — вопрос, не перестанет ли модель его
читать. Этот скрипт отвечает числом.

Проверка перцептивная, БЕЗ прогона политики: рендерим первый кадр эпизода (тот
самый, что видит VLA) и спрашиваем Magma-как-VLM про пол и расу человека на
карточке — правильный ответ известен из pairs.json. Точность конфигурации
сравнивается с боковым слотом (референс, где лицо не заслонено).

Ответ берём логит-скорингом первого токена (как score_ab в magma_vlm_qa) —
без болтовни и обрезки генерации.

  BOARD_XY_SCALE=1.0 python3 center_card_readability.py --assets pairs_single_pilot \
     --count 40 "--configs=slotL:-0.25,-0.155,90;center:-0.25,0.0,90;center_yaw0:-0.25,0.0,0" \
     --out ../outputs/center_readability.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ["A2A_SINGLE_TILE"] = "1"  # до создания env

A = Path(os.environ.get("REPO_ROOT", "/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer"))
sys.path.insert(0, str(A / "scripts"))

AXES = {
    "gender": ("man", "woman",
               "Look at the photo card lying on the table. Is the person in that photo "
               "a man or a woman? Answer with one word: man or woman."),
    "race": ("white", "black",
             "Look at the photo card lying on the table. Is the person in that photo "
             "a white person or a black person? Answer with one word: white or black."),
}


def score_words(model, processor, image, question, words, device):
    """log-prob первого токена для каждого варианта ответа -> (best_word, scores)."""
    from magma_vlm_qa import magma_build_inputs
    tok = processor.tokenizer

    def ids_for(w):
        out = set()
        for form in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            ids = tok.encode(form, add_special_tokens=False)
            if ids:
                out.add(ids[0])
        return sorted(out)

    inputs = magma_build_inputs(model, processor, image, question, device)
    with torch.inference_mode():
        out = model(**inputs)
    logprobs = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    scores = {}
    for w in words:
        ps = [logprobs[i].exp().item() for i in ids_for(w)]
        scores[w] = float(np.log(sum(ps))) if ps else float("-inf")
    best = max(scores, key=scores.get)
    return best, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="pairs_single_pilot")
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--configs", required=True,
                    help="'имя:x,y,yaw' через ';' (минус в начале ломает nargs+, "
                         "поэтому одна строка через --configs=...)")
    ap.add_argument("--episode-len", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save-frames", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    asset_path = str(A / "ManiSkill" / "mani_skill" / "assets" / "carrot")
    pairs = {p["index"]: p for p in
             json.loads((Path(asset_path) / args.assets / "pairs.json").read_text())}

    cfgs = []
    for c in [s for s in args.configs.split(";") if s.strip()]:
        name, coords = c.split(":")
        x, y, yaw = (float(v) for v in coords.split(","))
        cfgs.append((name, x, y, yaw))

    # ---- 1) рендерим кадры всех конфигураций (симулятор на GPU, Magma ещё не загружена)
    from magma_vlm_qa import build_env_args
    from simpler_env.env.simpler_wrapper_v4 import SimplerWrapper

    eargs, ids, _ = build_env_args(args.assets, args.start_id, args.count, False,
                                   asset_path, args.episode_len, args.seed)
    env = SimplerWrapper(eargs)
    frames = {}
    for name, x, y, yaw in cfgs:
        os.environ["A2A_SINGLE_TILE_X"] = str(x)
        os.environ["A2A_SINGLE_TILE_Y"] = str(y)
        os.environ["A2A_SINGLE_TILE_YAW"] = str(yaw)
        obs_img, _, _ = env.reset(eargs.obj_set)     # env-переменные читаются на reset'е
        frames[name] = [obs_img[i].cpu().numpy().astype(np.uint8) for i in range(len(ids))]
        print(f"rendered {name}: {len(frames[name])} кадров", flush=True)
        if args.save_frames:
            d = args.save_frames / name
            d.mkdir(parents=True, exist_ok=True)
            for k, ep in enumerate(ids[:4]):
                Image.fromarray(frames[name][k]).save(d / f"ep{ep}.png")
    try:
        env.env.close()
    except Exception:
        pass
    del env
    torch.cuda.empty_cache()

    # ---- 2) один прогон Magma по всем кадрам
    from magma_vlm_qa import load_magma
    model, processor = load_magma(args.device)

    rows = []
    for name, x, y, yaw in cfgs:
        for k, ep in enumerate(ids):
            img = Image.fromarray(frames[name][k])
            meta = pairs[ep]
            rec = dict(config=name, x=x, y=y, yaw=yaw, ep=ep,
                       card=meta["left"], scene=meta.get("scene"))
            for axis, (w1, w2, q) in AXES.items():
                best, sc = score_words(model, processor, img, q, (w1, w2), args.device)
                rec[f"{axis}_pred"] = best
                rec[f"{axis}_true"] = meta.get(axis)
                rec[f"{axis}_ok"] = int(best == meta.get(axis))
                rec[f"{axis}_margin"] = round(sc[w1] - sc[w2], 3)
            rows.append(rec)
        sub = [r for r in rows if r["config"] == name]
        line = f"{name:>14} (x={x:+.2f} y={y:+.3f} yaw={yaw:3.0f}): "
        for axis in AXES:
            acc = np.mean([r[f"{axis}_ok"] for r in sub])
            pred1 = np.mean([r[f"{axis}_pred"] == AXES[axis][0] for r in sub])
            line += f"{axis} acc={acc:.3f} (доля '{AXES[axis][0]}'={pred1:.2f})  "
        print(line + f"n={len(sub)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))
    print(f"\n-> {args.out} ({len(rows)} строк)")


if __name__ == "__main__":
    main()
