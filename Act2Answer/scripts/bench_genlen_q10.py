#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сколько токенов Magma генерирует на каждом из 10 вопросов q10 и что будет при обрезке.

Без симулятора: подаём уже отрендеренные первые кадры VisBias в той же раскладке, что и прогон
(andrey_s1p2_y0p14 = BOARD_XY_SCALE 1.2, A2A_TILE_Y 0.14). Для каждого вопроса:
  - длина генерации и позиция EOS (сэмплирование, как в боевом прогоне);
  - доля сгенерированных токенов из диапазона действий (верхние 256 id словаря) против текста;
  - пример текстовой части, если она есть;
  - время вызова;
  - в жадном режиме: совпадает ли действие, извлечённое из полной генерации, с тем, что даст
    обрезка на MAX_NEW_TOKENS (8 по умолчанию). При жадном декодировании префикс не зависит
    от лимита, поэтому расхождение возможно ТОЛЬКО если EOS стоит дальше 8-й позиции.

  N=48 python bench_genlen_q10.py
"""
import csv
import os
import random
import statistics
import sys
import time
from pathlib import Path

A = Path(os.environ.get("REPO_ROOT", "/home/moskalenko/ws/bias-vla-benchmark-main/Act2Answer"))
sys.path.insert(0, str(A / "SimplerEnv"))
sys.path.insert(0, str(A / "ManiSkill"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from simpler_env import eval as E  # noqa: E402
from simpler_env.policies.magma.magma_model import MagmaInference  # noqa: E402

N = int(os.environ.get("N", "48"))
CUT = int(os.environ.get("CUT", "8"))
FR = A / "outputs" / "visbias_frames"
CFG = "andrey_s1p2_y0p14"
QIDS = ["visbias_mcq_income_gt_100000", "visbias_mcq_income_lt_20000",
        "visbias_mcq_education_doctorate", "visbias_form_education_no_education",
        "visbias_form_criminal_record_no", "visbias_form_criminal_record_yes",
        "visbias_form_disability_status_none", "visbias_form_disability_status_mental_disorder",
        "visbias_form_occupation_doctor", "visbias_form_occupation_unemployed"]

qtext = {r["question_id"]: r["question_vla"] for r in
         csv.DictReader(open(FR / "questions.tsv", encoding="utf-8-sig", newline=""), delimiter="\t")}
frames = [r["frame"] for r in csv.DictReader(open(FR / "manifest.csv", encoding="utf-8"))
          if r["config"] == CFG and r["order"] == "ab" and "profession" not in r["source"]]
random.Random(0).shuffle(frames)
frames = frames[:N]
imgs = torch.from_numpy(np.stack([np.array(Image.open(FR / f).convert("RGB")) for f in frames]))
print(f"кадров: {len(frames)} ({CFG}), форма {tuple(imgs.shape)}", flush=True)

sys.argv = ["g", "--vla", "magma", "--assets", "pairs_choice_vla_confirm",
            "--count", "6", "--buffer-minibatch", "1"]
args = E.build_runner_args(E.parse_args(), E.VLA_CONFIGS["magma"])
pol = MagmaInference(0, args, "microsoft/Magma-8B")
tok = pol.processor.tokenizer
V, EOS = pol.vocab_size, tok.eos_token_id
ACT_LO = V - 256  # токены действия: id в [V-256, V)

cap = {}
orig_gen = pol.vla.generate


def spy(**kw):
    if "max_new_tokens" in cap:
        kw["max_new_tokens"] = cap["max_new_tokens"]
    out = orig_gen(**kw)
    cap["in_len"] = kw["input_ids"].shape[1]
    cap["out"] = out.detach().cpu()
    return out


pol.vla.generate = spy


def run(q, sample, limit=None):
    pol.sample = sample
    cap.clear()
    if limit:
        cap["max_new_tokens"] = limit
    obs = {"image": imgs, "task_description": [q] * len(frames)}
    torch.cuda.synchronize()
    t = time.perf_counter()
    act = pol.get_action(obs)
    torch.cuda.synchronize()
    return act, cap["out"][:, cap["in_len"]:], time.perf_counter() - t


def eos_pos(row):
    p = (row == EOS).nonzero()
    return int(p[0]) if len(p) else len(row)


def extract(row):
    e = eos_pos(row)
    return tuple(row[max(0, e - 7):e].tolist())


print("\n{:44s} {:>6s} {:>7s} {:>7s} {:>8s} {:>9s} {:>10s}".format(
    "вопрос", "время", "дл.мед", "дл.макс", "EOS>8,%", "текст,%", "совп.@" + str(CUT)))
print("-" * 100)
examples = {}
for qid in QIDS:
    q = qtext[qid]
    run(q, sample=True)                              # прогрев
    _, gen_s, dt = run(q, sample=True)               # как в боевом прогоне
    L = [eos_pos(r) for r in gen_s]
    txt_frac = []
    for r in gen_s:
        body = r[:eos_pos(r)]
        if len(body):
            txt_frac.append(float((body < ACT_LO).float().mean()))
    # жадно: полная генерация против обрезки
    _, gen_full, _ = run(q, sample=False)
    _, gen_cut, _ = run(q, sample=False, limit=CUT)
    same = [extract(a) == extract(b) for a, b in zip(gen_full, gen_cut)]
    print("{:44s} {:>5.2f}с {:>7.0f} {:>7d} {:>8.0f} {:>9.0f} {:>9.0f}%".format(
        qid[:44], dt, statistics.median(L), max(L), 100 * np.mean([x > 8 for x in L]),
        100 * (np.mean(txt_frac) if txt_frac else 0), 100 * np.mean(same)), flush=True)
    longest = max(range(len(L)), key=lambda i: L[i])
    if L[longest] > 8:
        body = gen_s[longest][:eos_pos(gen_s[longest])]
        text_ids = [t for t in body.tolist() if t < ACT_LO]
        examples[qid] = (L[longest], tok.decode(text_ids)[:220])

print("\nПримеры текста, который модель пишет перед действием (самая длинная генерация):")
for qid, (n, s) in examples.items():
    print(f"  [{qid}] {n} токенов: {s!r}")
if not examples:
    print("  нет — везде ровно действие + EOS")
