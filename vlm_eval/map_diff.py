#!/usr/bin/env python3
"""Какие тензоры базовой модели НЕ покрыты VLA-чекпойнтом.

Если непокрытые есть, «вариант 2» получается химерой: часть весов из VLA, часть
из базы. Прежде чем считать такую модель, надо знать, какая именно часть осталась
базовой — вычислительный блок или служебная мелочь.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vla_vlm_loader import SPECS, _iter_ckpt_tensors  # noqa: E402


def group(key):
    """Сворачиваем номера слоёв, чтобы увидеть структуру, а не 500 строк."""
    return re.sub(r"\.\d+\.", ".N.", key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, choices=sorted(SPECS))
    a = ap.parse_args()
    spec = SPECS[a.key]

    from transformers import AutoConfig, AutoModelForImageTextToText
    cfg = AutoConfig.from_pretrained(spec["base"])
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(cfg)
    base_keys = set(model.state_dict())

    ckpt_keys = set()
    for name, _ in _iter_ckpt_tensors(spec["ckpt"]):
        if name.startswith(spec["prefix"]):
            ckpt_keys.add(name[len(spec["prefix"]):])

    missing = base_keys - ckpt_keys   # есть в базе, нет в VLA -> останется базовым
    extra = ckpt_keys - base_keys     # есть в VLA, некуда класть

    print(f"### {a.key}: база {len(base_keys)} тензоров, чекпойнт {len(ckpt_keys)}")
    print(f"    НЕ покрыто базовых: {len(missing)}   лишних в чекпойнте: {len(extra)}")
    for title, s in (("не покрыто (останется от базы)", missing), ("лишние в чекпойнте", extra)):
        if not s:
            continue
        print(f"  -- {title}:")
        for pat, n in Counter(group(k) for k in s).most_common(12):
            print(f"     {n:4d} x {pat}")
        nums = sorted({int(m.group(1)) for k in s
                       for m in [re.search(r"layers\.(\d+)\.", k)] if m})
        if nums:
            print(f"     номера слоёв: {nums[:6]} … {nums[-6:]} (всего {len(nums)})")


if __name__ == "__main__":
    main()
