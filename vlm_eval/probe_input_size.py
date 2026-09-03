#!/usr/bin/env python3
"""Во что процессор модели превращает наш кадр 640x480.

Считаем не по документации, а фактически: гоняем один и тот же PNG через
процессор и смотрим форму pixel_values. Для Qwen-семейства разрешение плавающее,
поэтому берём его из image_grid_thw (патч 14, слияние 2x2).
"""
import sys

from PIL import Image
from transformers import AutoProcessor

FRAME = "/workspace/moskalenko/exp_weapon/frames_weapon200/noswap/ep0000.png"


def describe(name, trust=False):
    img = Image.open(FRAME).convert("RGB")
    try:
        proc = AutoProcessor.from_pretrained(name, trust_remote_code=trust)
    except Exception as e:
        return f"{name}: процессор не загрузился ({type(e).__name__})"
    for kwargs in ({"text": "test", "images": img}, {"images": img}, {"images": [img]}):
        try:
            out = proc(return_tensors="pt", **kwargs)
            break
        except Exception:
            out = None
    if out is None:
        try:
            out = proc("test", img, return_tensors="pt")
        except Exception as e:
            return f"{name}: не удалось обработать ({type(e).__name__})"

    parts = []
    pv = out.get("pixel_values", None)
    if pv is not None:
        parts.append(f"pixel_values{tuple(pv.shape)}")
    grid = out.get("image_grid_thw", None)
    if grid is not None:
        t, h, w = [int(x) for x in grid[0]]
        parts.append(f"grid {h}x{w} патчей -> {w * 14}x{h * 14} px, "
                     f"визуальных токенов {h * w // 4}")
    elif pv is not None and pv.dim() >= 3:
        s = pv.shape
        parts.append(f"-> {s[-1]}x{s[-2]} px" + (f", кропов {s[0] * (s[1] if pv.dim() > 4 else 1)}"
                                                 if pv.dim() > 3 else ""))
    ids = out.get("input_ids", None)
    if ids is not None:
        parts.append(f"input_ids {tuple(ids.shape)}")
    return f"{name}: " + "; ".join(parts)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        name, _, flag = arg.partition("#")
        print(describe(name, trust=(flag == "trust")), flush=True)
