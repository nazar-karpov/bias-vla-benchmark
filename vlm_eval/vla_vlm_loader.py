#!/usr/bin/env python3
"""Достать VLM-часть из VLA-чекпойнта (вариант 2).

Аудит показал, что у Xiaomi, GR00T и InternVLA-M1 внутри лежит стоковая Qwen-VLM,
просто под именным префиксом: `vlm.`, `backbone.model.`, `qwen_vl_interface.model.`.
Значит отдельное окружение и репозиторий модели не нужны — достаточно переклеить
имена тензоров в обычный Qwen*ForConditionalGeneration.

Схема: грузим БАЗОВУЮ модель целиком, поверх накатываем веса из VLA-чекпойнта и
считаем, сколько тензоров реально изменилось. Это даёт три вещи сразу:
  * рабочую модель со всей обвязкой (процессор, генерация, chat-template);
  * жёсткую проверку, что мэппинг полный — непокрытые тензоры не проглатываются;
  * ответ на вопрос «заморожен ли бэкбон» — если изменилось 0 тензоров, VLM-часть
    VLA идентична базе и отдельный прогон варианта 2 бессмыслен.

ВАЖНО: кривой мэппинг не падает, а даёт слепую модель. Поэтому после загрузки
обязателен контрольный вопрос с визуально определённым ответом.
"""
from __future__ import annotations

import glob
import json
import os

import torch

SPECS = {
    "xiaomi": {
        "ckpt": "XiaomiRobotics/Xiaomi-Robotics-0-SimplerEnv-WidowX",
        "base": "Qwen/Qwen3-VL-4B-Instruct",
        "prefix": "vlm.",
        # Официальный шаблон инференса заканчивает инструкцию на `/no_cot`, а ход
        # ассистента открывает `<cot></cot>` — то есть пустой блок это штатный
        # маркер «без рассуждения», а не сбой. Оставляем затравку пустой, чтобы
        # модель сама решала, писать ли рассуждение.
        "answer_prefix": "",
        # Своя предобработка картинки: бюджет 90k пикселей против 16.7M у базовой.
        "processor": "XiaomiRobotics/Xiaomi-Robotics-0-SimplerEnv-WidowX",
        "note": "база Xiaomi-Robotics-0; голова связана с embed_tokens",
    },
    "xiaomi_pretrain": {
        # Тот же VLA до узкого файнтюна под SimplerEnv-WidowX. В отличие от него
        # текстом отвечает и контроль проходит — значит языковой канал закрывается
        # именно на задачном дообучении, а не заложен закрытым в архитектуре.
        "ckpt": "XiaomiRobotics/Xiaomi-Robotics-0-Pretrain",
        "base": "Qwen/Qwen3-VL-4B-Instruct",
        "prefix": "vlm.",
        "processor": "XiaomiRobotics/Xiaomi-Robotics-0-Pretrain",
        "note": "Xiaomi-Robotics-0 pretrain (предок WidowX-чекпойнта)",
    },
    "groot": {
        "ckpt": "nvidia/GR00T-N1.7-SimplerEnv-Bridge",
        "base": "nvidia/Cosmos-Reason2-2B",
        "prefix": "backbone.model.",
        # В чекпойнте лежат слои 0-15, слоёв 16-27 нет: GR00T несёт только нижнюю
        # половину языковой модели. Догружать верх из базы нельзя — это была бы
        # химера «низ от VLA, верх от базы». Собираем ровно то, что есть, вместе
        # с сохранившейся в чекпойнте lm_head.
        "truncate_layers": 16,
        "note": "база GR00T-N1.7 (Cosmos-Reason2-2B = Qwen3-VL-2B), усечена до 16 слоёв",
    },
    "pi05": {
        "ckpt": "qownscks/pi05_widowx",
        "base": "google/paligemma-3b-pt-224",
        # Порт LeRobot: VLM и action-эксперт лежат двумя половинами одного модуля.
        # Берём половину paligemma — у неё в чекпойнте своя lm_head (у эксперта своя).
        "prefix": "model.paligemma_with_expert.paligemma.",
        "backend": "paligemma",
        "note": "pi0.5 widowx; 603 тензора в половине paligemma",
    },
    "internvla": {
        "ckpt": "/workspace/moskalenko/bias-vla-benchmark-main/internvla_ckpt/"
                "InternVLA-M1-Pretrain-RT-1-Bridge/checkpoints/steps_50000_pytorch_model.pt",
        "base": "Qwen/Qwen2.5-VL-3B-Instruct",
        "prefix": "qwen_vl_interface.model.",
        "note": "база InternVLA-M1",
    },
}


def _iter_ckpt_tensors(path_or_repo: str):
    """Отдаёт (имя, тензор) из чекпойнта: локальный .pt, локальная папка или репо HF."""
    if path_or_repo.endswith((".pt", ".pth", ".bin")):
        sd = torch.load(path_or_repo, map_location="cpu", mmap=True, weights_only=True)
        while isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        for k, v in sd.items():
            yield k, v
        return

    from safetensors import safe_open
    if os.path.isdir(path_or_repo):
        files = sorted(glob.glob(os.path.join(path_or_repo, "*.safetensors")))
    else:
        from huggingface_hub import hf_hub_download, list_repo_files
        names = [f for f in list_repo_files(path_or_repo) if f.endswith(".safetensors")]
        files = [hf_hub_download(path_or_repo, n) for n in sorted(names)]
    for f in files:
        with safe_open(f, framework="pt") as fh:
            for k in fh.keys():
                yield k, fh.get_tensor(k)


def load_vla_vlm(key: str, dtype=torch.bfloat16, strict: bool = True):
    """Возвращает (model, processor, отчёт). Модель уже в eval, но ещё на CPU."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    spec = SPECS[key]
    print(f"[vla] {key}: база {spec['base']}, чекпойнт {spec['ckpt']}", flush=True)

    import transformers
    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    # Процессор по умолчанию базовый, но у некоторых VLA своя предобработка картинки.
    # У Xiaomi бюджет 90k пикселей против 16.7M у базовой Qwen3-VL: подав базовый
    # ресайз, мы дали бы модели вчетверо больше визуальных токенов, чем она видела.
    proc_src = spec.get("processor", spec["base"])
    processor = AutoProcessor.from_pretrained(
        proc_src, trust_remote_code=(proc_src != spec["base"]))
    if proc_src != spec["base"]:
        print(f"[vla] процессор из чекпойнта: {proc_src}", flush=True)
    if spec.get("truncate_layers"):
        # Веса базы не берём вообще: чекпойнт покрывает усечённую модель целиком,
        # поэтому случайная инициализация до наложения роли не играет.
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(spec["base"])
        tcfg = getattr(cfg, "text_config", cfg)
        print(f"[vla] усечение: {tcfg.num_hidden_layers} -> {spec['truncate_layers']} слоёв",
              flush=True)
        tcfg.num_hidden_layers = spec["truncate_layers"]
        model = AutoModelForImageTextToText.from_config(cfg).to(dtype)
    else:
        model = AutoModelForImageTextToText.from_pretrained(spec["base"], **{dtype_kw: dtype})
    model.eval()

    sd = model.state_dict()
    prefix = spec["prefix"]
    rep = {"ckpt_under_prefix": 0, "mapped": 0, "changed": 0,
           "unmapped": [], "shape_mismatch": []}
    mapped_keys = set()

    with torch.no_grad():
        for name, tensor in _iter_ckpt_tensors(spec["ckpt"]):
            if not name.startswith(prefix):
                continue
            rep["ckpt_under_prefix"] += 1
            short = name[len(prefix):]
            if short not in sd:
                rep["unmapped"].append(short)
                continue
            dst = sd[short]
            src = tensor.to(dst.dtype)
            if dst.shape != src.shape:
                # Единственное расхождение, которое разрешаем: словарь базы длиннее
                # за счёт паддинга (у PaliGemma 257216 против 257152 у pi0.5).
                # Переносим все настоящие строки, хвост оставляем базовым.
                if dst.shape[1:] == src.shape[1:] and dst.shape[0] > src.shape[0]:
                    n = src.shape[0]
                    if not torch.equal(dst[:n], src):
                        rep["changed"] += 1
                    dst[:n].copy_(src)
                    rep["mapped"] += 1
                    mapped_keys.add(short)
                    rep.setdefault("padded", []).append(
                        (short, dst.shape[0], src.shape[0]))
                    continue
                rep["shape_mismatch"].append((short, tuple(dst.shape), tuple(src.shape)))
                continue
            if not torch.equal(dst, src):
                rep["changed"] += 1
            dst.copy_(src)
            rep["mapped"] += 1
            mapped_keys.add(short)

    # Если голова связана с эмбеддингами, в чекпойнте её нет (Xiaomi) — пересвязываем,
    # иначе останется головой БАЗОВОЙ модели при эмбеддингах из VLA.
    tied = bool(getattr(getattr(model.config, "text_config", model.config),
                        "tie_word_embeddings", False))
    if tied:
        head_keys = {k for k in sd if k.endswith("lm_head.weight")}
        emb_keys = {k for k in sd if k.endswith("embed_tokens.weight")}
        head_loaded = bool(head_keys & mapped_keys)
        emb_loaded = bool(emb_keys & mapped_keys)
        inp, out = model.get_input_embeddings(), model.get_output_embeddings()
        with torch.no_grad():
            if head_loaded and not emb_loaded and out is not None:
                # В чекпойнте лежит голова, а таблица эмбеддингов опущена (pi0.5).
                # tie_weights() копирует ВХОД в выход и затёр бы загруженную голову,
                # поэтому переносим руками в обратную сторону.
                inp.weight.copy_(out.weight)
                print("[vla] embed_tokens восстановлены из lm_head чекпойнта", flush=True)
                rep["mapped"] += len(emb_keys)
            elif emb_loaded and not head_loaded:
                model.tie_weights()
                print("[vla] lm_head пересвязана с embed_tokens", flush=True)
                rep["mapped"] += len(head_keys)
        ok = out is None or torch.equal(inp.weight, out.weight)
        print(f"[vla] tie_word_embeddings=True, вход и выход совпадают: {ok}", flush=True)
        rep["retied"] = ok

    rep["base_tensors"] = len(sd)
    rep["base_not_covered"] = rep["base_tensors"] - rep["mapped"]
    print(f"[vla] тензоров под префиксом: {rep['ckpt_under_prefix']}, "
          f"наложено: {rep['mapped']}/{rep['base_tensors']} базовых, "
          f"ОТЛИЧАЮТСЯ от базы: {rep['changed']}", flush=True)
    if rep["unmapped"]:
        print(f"[vla] НЕ НАШЛОСЬ в базе ({len(rep['unmapped'])}): {rep['unmapped'][:5]}", flush=True)
    if rep["shape_mismatch"]:
        print(f"[vla] РАЗНЫЕ ФОРМЫ ({len(rep['shape_mismatch'])}): {rep['shape_mismatch'][:3]}", flush=True)
    if rep["changed"] == 0:
        print("[vla] ВНИМАНИЕ: ни один тензор не отличается от базы — "
              "VLM-часть заморожена, вариант 2 == вариант 1", flush=True)
    if strict and (rep["unmapped"] or rep["shape_mismatch"] or rep["base_not_covered"]):
        raise RuntimeError(
            f"мэппинг неполный: unmapped={len(rep['unmapped'])}, "
            f"mismatch={len(rep['shape_mismatch'])}, "
            f"базовых тензоров без замены={rep['base_not_covered']}")
    return model, processor, rep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, choices=sorted(SPECS))
    ap.add_argument("--report", default=None)
    ap.add_argument("--no-strict", action="store_true")
    a = ap.parse_args()
    _, _, rep = load_vla_vlm(a.key, strict=not a.no_strict)
    if a.report:
        open(a.report, "w").write(json.dumps(
            {k: v for k, v in rep.items() if k not in ("unmapped", "shape_mismatch")}
            | {"unmapped": rep["unmapped"][:20],
               "shape_mismatch": [list(x) for x in rep["shape_mismatch"][:20]]}, indent=1))
