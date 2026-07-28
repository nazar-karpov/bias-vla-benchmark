#!/usr/bin/env python3
"""Bias-детализация с разделением ПОЗИЦИОННОГО шума и ЧИСТОГО признакового bias.

Для каждой пары смотрим выбор модели в ОБЕИХ раскладках (noswap+swap):
- если модель выбрала один и тот же ПРИЗНАК (пол/раса) в обоих проходах -> ЧИСТЫЙ сигнал
  (позиция ни при чём, это реальная ассоциация);
- если выбор перевернулся вместе со стороной -> ПОЗИЦИОННЫЙ шум (не bias по признаку).
"""
import yaml, json, re, sys

A = "/home/User17/bias_benchmark/nazar_folder/Act2Answer"
P = json.load(open(f"{A}/ManiSkill/mani_skill/assets/carrot/pairs_bias/pairs.json"))
model = sys.argv[1] if len(sys.argv) > 1 else "magma"
O = f"{A}/outputs"

def parse(n):
    return ("white" if "white" in n else "black" if "black" in n else None,
            "woman" if "woman" in n else "man" if "man" in n else None)

def axis(e):
    lr, ls = parse(e["left"]); rr, rs = parse(e["right"])
    if ls != rs and lr == rr: return "gender"
    if lr != rr and ls == rs: return "race"
    return "other"

def chosen(pair, side, swap):
    if side == 0: return None
    if not swap: return pair["left"] if side == 1 else pair["right"]
    return pair["right"] if side == 1 else pair["left"]

def prof(q):
    m = re.search(r"probably (?:a|an) (.+)", q)
    return m.group(1) if m else q

def load(pass_name, swap):
    st = yaml.safe_load(open(f"{O}/bias-{model}-pairs_bias-{pass_name}/glob/vis_0_test/stats.yaml"))
    r = {}
    for idx, info in st["last_info"].items():
        e = int(idx)
        side = int(info.get("chosen_side", 0))
        r[e] = parse(chosen(P[e], side, swap)) if (info.get("is_answered") and side) else (None, None)
    return r

ns = load("noswap", False)
sw = load("swap", True)

for AX, gi in [("gender", 1), ("race", 0)]:
    label = "ПОЛ (man/woman)" if AX == "gender" else "РАСА (white/black)"
    print(f"\n########## {model.upper()} — ось {label} ##########")
    clean = {}   # признак -> счётчик устойчивых
    noisy = 0; unans = 0
    rows = []
    for e in sorted(ns):
        if axis(P[e]) != AX: continue
        a = ns[e][gi]; b = sw.get(e, (None, None))[gi]
        prf = prof(P[e]["question"])
        if a is None or b is None:
            unans += 1; tag = "no-answer"
        elif a == b:
            clean[a] = clean.get(a, 0) + 1; tag = f"STABLE -> {a.upper()}"
        else:
            noisy += 1; tag = f"flip ({a}/{b}) = позиция"
        rows.append((e, prf, tag))
    for e, prf, tag in rows:
        print(f"  ep {e:>2}  {prf:<20} {tag}")
    print(f"\n  ЧИСТЫЙ сигнал (устойчивы в обеих раскладках): {clean}")
    print(f"  позиционный шум (флип): {noisy} | без ответа: {unans}")
    tot = sum(clean.values())
    if tot:
        for k, v in clean.items():
            print(f"    {k}: {v}/{tot} = {v/tot:.0%}")
