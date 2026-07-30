#!/usr/bin/env python3
"""Сравнить прямой VLM-опрос Magma (scripts/magma_vlm_qa.py) с VLA-прогоном
`bias-magma-pairs_bias-{noswap,swap}` на тех же 55 эпизодах pairs_bias.

Мотивация: в VLA-прогоне 23/55 (noswap) и 20/55 (swap) эпизодов — no-answer
(кубик физически не доехал). Вопрос: это провал МАНИПУЛЯЦИИ или модель и не знала
ответа? VLM-режим отвечает на КАЖДЫЙ эпизод (нет "не доехал") → прямой тест.

Что считаем (по аналогии с scripts/bias_detail.py):
1. answer-rate: VLA (is_answered) vs VLM (parsed_side) — сколько эпизодов получили ответ.
2. Для эпизодов, где VLA no-answer: что VLM ответил (был ли устойчивый признак).
3. Чистый признаковый сигнал: выбор ОДНОГО И ТОГО ЖЕ пола/расы в обеих раскладках
   (noswap+swap) → реальная ассоциация, а не позиционный крен. Сравниваем VLM vs VLA.

Использование (на сервере или локально, нужен PyYAML):
  python compare_vlm_vla.py <vlm_full.json> <vla_noswap.yaml> <vla_swap.yaml> <pairs.json>
"""
import json
import re
import sys
from collections import Counter

import yaml


def parse_tile(name):
    race = "white" if "white" in name else ("black" if "black" in name else None)
    sex = "woman" if "woman" in name else ("man" if "man" in name else None)
    return race, sex


def axis(pair):
    lr, ls = parse_tile(pair["left"])
    rr, rs = parse_tile(pair["right"])
    if ls != rs and lr == rr:
        return "gender"
    if lr != rr and ls == rs:
        return "race"
    return "other"


def prof(q):
    m = re.search(r"probably (?:a |an )?(.+?)[:?]", q) or re.search(r"probably (?:a |an )?(.+)$", q)
    return m.group(1).strip() if m else q


# --- VLA: chosen image from chosen_side + swap layout -------------------------
def vla_chosen(pair, side, swap):
    if side == 0:
        return None
    if not swap:
        return pair["left"] if side == 1 else pair["right"]
    return pair["right"] if side == 1 else pair["left"]


def load_vla(pairs, path, swap):
    st = yaml.safe_load(open(path))
    out = {}
    for idx, info in st["last_info"].items():
        e = int(idx)
        side = int(info.get("chosen_side", 0))
        ans = bool(info.get("is_answered", False))
        out[e] = vla_chosen(pairs[e], side, swap) if (ans and side) else None
    return out


# --- VLM: chosen image from parsed_side ("Left"/"Right") + that row's left/right
def load_vlm(rows):
    """rows: list из magma_vlm_qa.py. Ключи: index, swap, left, right, parsed_side.
    left/right в строке УЖЕ учитывают раскладку (для swap они переставлены)."""
    ns, sw = {}, {}
    for r in rows:
        e = r["index"]
        side = r["parsed_side"]  # "Left"/"Right"/None
        img = None
        if side == "Left":
            img = r["left"]
        elif side == "Right":
            img = r["right"]
        (sw if r["swap"] else ns)[e] = img
    return ns, sw


def clean_signal(pairs, ns, sw, AX, gi):
    """Возвращает (clean Counter, noisy, unans, rows)."""
    clean = Counter()
    noisy = unans = 0
    rows = []
    for e in sorted(ns):
        if axis(pairs[e]) != AX:
            continue
        a = parse_tile(ns[e])[gi] if ns.get(e) else None
        b = parse_tile(sw[e])[gi] if sw.get(e) else None
        p = prof(pairs[e]["question"])
        if a is None or b is None:
            unans += 1
            tag = "no-answer"
        elif a == b:
            clean[a] += 1
            tag = f"STABLE->{a.upper()}"
        else:
            noisy += 1
            tag = f"flip({a}/{b})=позиция"
        rows.append((e, p, tag))
    return clean, noisy, unans, rows


def answer_rate(d):
    return sum(1 for v in d.values() if v), len(d)


def main():
    vlm_path, vla_ns_path, vla_sw_path, pairs_path = sys.argv[1:5]
    pairs = json.load(open(pairs_path))
    vlm_rows = json.load(open(vlm_path))

    vla_ns = load_vla(pairs, vla_ns_path, swap=False)
    vla_sw = load_vla(pairs, vla_sw_path, swap=True)
    vlm_ns, vlm_sw = load_vlm(vlm_rows)

    print("=" * 70)
    print("ANSWER-RATE (сколько эпизодов получили ответ)")
    print("=" * 70)
    for tag, d in [("VLA noswap", vla_ns), ("VLA swap", vla_sw),
                   ("VLM noswap", vlm_ns), ("VLM swap", vlm_sw)]:
        a, n = answer_rate(d)
        print(f"  {tag:<12} {a:>2}/{n}  ({a/n:.0%})")

    # эпизоды, где VLA не доехал — а VLM ответил
    print("\n" + "=" * 70)
    print("ЭПИЗОДЫ, ГДЕ VLA = no-answer (кубик не доехал) — что сказал VLM")
    print("=" * 70)
    rescued = 0
    for e in sorted(vla_ns):
        va, vs = vla_ns.get(e), vla_sw.get(e)
        if va is not None and vs is not None:
            continue  # VLA ответил в обеих — не интересно тут
        ma, ms = vlm_ns.get(e), vlm_sw.get(e)
        vla_state = f"ns={'—' if va is None else parse_tile(va)}, sw={'—' if vs is None else parse_tile(vs)}"
        vlm_state = f"ns={parse_tile(ma) if ma else '—'}, sw={parse_tile(ms) if ms else '—'}"
        if (va is None or vs is None) and ma and ms:
            rescued += 1
        print(f"  ep{e:>2} {prof(pairs[e]['question']):<22} VLA[{vla_state}]  ->  VLM[{vlm_state}]")
    print(f"\n  VLM дал ответ там, где VLA хотя бы раз не доехал: способность к задаче не была блокером")

    # чистый признаковый сигнал
    for AX, gi, name in [("gender", 1, "ПОЛ (man/woman)"), ("race", 0, "РАСА (white/black)")]:
        print("\n" + "#" * 70)
        print(f"# ЧИСТЫЙ СИГНАЛ — ось {name}")
        print("#" * 70)
        for who, ns, sw in [("VLA", vla_ns, vla_sw), ("VLM", vlm_ns, vlm_sw)]:
            clean, noisy, unans, rows = clean_signal(pairs, ns, sw, AX, gi)
            tot = sum(clean.values())
            print(f"\n  --- {who} ---")
            print(f"    чистый (устойчив в обеих раскладках): {dict(clean)}")
            print(f"    позиционный флип: {noisy} | без ответа хотя бы раз: {unans}")
            if tot:
                for k, v in clean.items():
                    print(f"      {k}: {v}/{tot} = {v/tot:.0%}")


if __name__ == "__main__":
    main()
