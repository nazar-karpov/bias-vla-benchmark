#!/usr/bin/env python3
"""Единый интерфейс к VLM для парного вопроса «left / right».

Каждый бэкенд возвращает объект с двумя методами:
    score(image, question) -> (p_left, p_right)   логит-скоринг первого токена ответа
    generate(image, question) -> str              свободная генерация (для проверки)

Добавить модель = добавить ветку в load_vlm(). Всё остальное (рендер, вопросы,
анализ) от списка моделей не зависит.
"""
from __future__ import annotations

import torch

# Варианты написания ответа — логиты суммируются по всем, что дают разный первый
# токен. Набор задаётся снаружи: для двух картинок это left/right, для одной —
# yes/no. По умолчанию left/right, чтобы старые вызовы работали как прежде.
LEFT_WORDS = ["left", " left", "Left", " Left", "LEFT"]
RIGHT_WORDS = ["right", " right", "Right", " Right", "RIGHT"]
YES_WORDS = ["yes", " yes", "Yes", " Yes", "YES"]
NO_WORDS = ["no", " no", "No", " No", "NO"]


def _first_token_ids(tokenizer, words):
    """Первые токены всех написаний ответа.

    Осторожно с SentencePiece: у Llama-2 `" left"` кодируется как ['▁','▁left'],
    то есть первым токеном идёт голый пробел. Он одинаково попадает и в левый,
    и в правый набор, а общий член в обеих суммах механически стягивает ответ
    к 0.5 — именно так у OpenVLA контроль выходил ровно 50.0%. Пустые и
    пробельные токены отбрасываем.
    """
    ids = set()
    for w in words:
        enc = tokenizer.encode(w, add_special_tokens=False)
        for tid in enc:
            piece = tokenizer.convert_ids_to_tokens([tid])[0]
            if piece.replace("▁", "").replace("Ġ", "").strip():
                ids.add(tid)
                break  # берём первый ЗНАЧАЩИЙ токен написания
    return sorted(ids)


class _Backend:
    """Общая логика скоринга; специфика — в build_inputs() наследника."""

    # Некоторые модели начинают ответ со служебной обёртки (Xiaomi-Robotics-0 —
    # с `<cot></cot>`). Тогда первый токен не ответ, и логит-скоринг мерит не то.
    # Затравка дописывается в конец промпта, чтобы скорилась уже позиция слова.
    answer_prefix = ""

    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        tok = getattr(processor, "tokenizer", processor)
        self.tok = tok
        self.set_answers(LEFT_WORDS, RIGHT_WORDS)

    def set_answers(self, a_words, b_words):
        """Задать пару вариантов ответа: left/right для двух картинок, yes/no для одной."""
        tok = self.tok
        self.a_words, self.b_words = list(a_words), list(b_words)
        self.a_ids = _first_token_ids(tok, self.a_words)
        self.b_ids = _first_token_ids(tok, self.b_words)
        if not self.a_ids or not self.b_ids:
            raise RuntimeError(f"не удалось закодировать {a_words} / {b_words}")
        shared = set(self.a_ids) & set(self.b_ids)
        if shared:  # общий токен в обоих наборах стягивает ответ к 0.5 — так молча
            raise RuntimeError(  # терялся сигнал у OpenVLA, больше не допускаем
                f"наборы ответов пересекаются по токенам {sorted(shared)}: "
                f"{tok.convert_ids_to_tokens(sorted(shared))}")

    def build_inputs(self, image, question):
        raise NotImplementedError

    def _place(self, inputs):
        """На устройство + привести картинку к dtype модели: процессоры отдают
        pixel_values в float32, а модель обычно в bf16."""
        dtype = next(self.model.parameters()).dtype
        out = {}
        for k, v in dict(inputs).items():
            if hasattr(v, "to"):
                v = v.to(self.device)
                if v.is_floating_point():
                    v = v.to(dtype)
            out[k] = v
        return out

    @torch.no_grad()
    def score(self, image, question):
        inputs = self.build_inputs(image, question)
        logits = self.model(**inputs).logits[0, -1].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        l = torch.logsumexp(logprobs[self.a_ids], dim=0)
        r = torch.logsumexp(logprobs[self.b_ids], dim=0)
        both = torch.logsumexp(torch.stack([l, r]), dim=0)
        return float(torch.exp(l - both)), float(torch.exp(r - both))

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=6):
        inputs = self.build_inputs(image, question)
        gen_cfg = getattr(self.model, "generation_config", None)
        if gen_cfg is not None and getattr(gen_cfg, "pad_token_id", None) is None:
            gen_cfg.pad_token_id = self.tok.pad_token_id or self.tok.eos_token_id
        out = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                  max_new_tokens=max_new_tokens)
        new = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True).strip()


class QwenVLBackend(_Backend):
    """Qwen2.5-VL / Qwen3-VL — обычный chat-template."""

    def build_inputs(self, image, question):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True) + self.answer_prefix
        return self.processor(text=[text], images=[image],
                              return_tensors="pt").to(self.device)


class PaliGemmaBackend(_Backend):
    """PaliGemma / PaliGemma2 — без chat-template, вопрос идёт префиксом."""

    def build_inputs(self, image, question):
        return self._place(self.processor(text=question + self.answer_prefix,
                                          images=image, return_tensors="pt"))


class MolmoBackend(_Backend):
    """Molmo2 / MolmoAct2 — remote-code модели со своим chat-шаблоном.

    Документированный путь инференса у MolmoAct2 — `predict_action`, но модель
    зарегистрирована как AutoModelForImageTextToText и шаблон несёт, поэтому
    текстовый канал пробуем штатно. Способ подачи определяется один раз при первом
    вызове: сначала пробуем современный путь (картинки внутри apply_chat_template),
    иначе откатываемся на «шаблон отдельно, процессор отдельно».
    """

    _mode = None
    # {id процессора: id модели} для спецтокенов картинки. Нужен, когда процессор
    # взят от базы, а у модели расширенный словарь: у MolmoAct2 image_end это 154625,
    # а базовая Molmo2 ставит 151937, и forward падает на подсчёте картинок.
    token_remap: dict[int, int] | None = None

    def _apply_remap(self, inputs):
        if not self.token_remap or "input_ids" not in inputs:
            return inputs
        ids = inputs["input_ids"]
        for src, dst in self.token_remap.items():
            ids[ids == src] = dst
        return inputs

    def build_inputs(self, image, question):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": question}]}]
        if self._mode != "split":
            try:
                inputs = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt")
                self._mode = "template"
                return self._apply_remap(self._place(inputs))
            except Exception:
                self._mode = "split"
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True) + self.answer_prefix
        return self._apply_remap(self._place(self.processor(
            text=[text], images=[image], return_tensors="pt")))


class Florence2Backend(_Backend):
    """Florence-2 — encoder-decoder, а не чат-модель.

    Ответ рождается на ПЕРВОЙ позиции декодера, поэтому в inputs кладём стартовый
    токен декодера, и общий скоринг (последняя позиция логитов) попадает куда надо.
    Chat-шаблона у неё нет: вопрос идёт как есть, текстовым промптом.
    """

    def build_inputs(self, image, question):
        inputs = self._place(self.processor(text=question + self.answer_prefix,
                                            images=image, return_tensors="pt"))
        cfg = self.model.config
        start = getattr(cfg, "decoder_start_token_id", None)
        if start is None:
            start = getattr(cfg, "bos_token_id", None) or self.tok.bos_token_id or 2
        inputs["decoder_input_ids"] = torch.tensor(
            [[start]], device=inputs["input_ids"].device)
        return inputs

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=6):
        inputs = self.build_inputs(image, question)
        inputs.pop("decoder_input_ids", None)  # generate заведёт его сам
        out = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                  max_new_tokens=max_new_tokens)
        # у seq2seq generate возвращает только выход декодера, срезать нечего
        return self.tok.decode(out[0], skip_special_tokens=True).strip()


class OpenVLABackend(_Backend):
    """OpenVLA/Prismatic — промпт в их формате `In: … Out:`, без chat-template."""

    def build_inputs(self, image, question):
        return self._place(self.processor(f"In: {question}\nOut:{self.answer_prefix}",
                                          image, return_tensors="pt"))


class MagmaBackend(_Backend):
    """Magma-8B — свой формат промпта, плюс обязательная починка ConvNeXt."""

    def build_inputs(self, image, question):
        convs = [
            {"role": "system", "content": "You are an agent that can see, talk and act."},
            {"role": "user", "content": f"<image_start><image><image_end>\n{question}"},
        ]
        prompt = self.processor.tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(images=[image], texts=prompt, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)
        inputs["image_sizes"] = inputs["image_sizes"].unsqueeze(0)
        dtype = next(self.model.parameters()).dtype
        out = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                v = v.to(self.device)
                if v.is_floating_point():  # картинка приходит float32, модель в bf16
                    v = v.to(dtype)
            out[k] = v
        return out


def auto_max_memory(reserve_mb=900, cpu_gb=200):
    """Узел общий: раскладываем модель по тому, что реально свободно на картах.
    Возвращает (device_map, max_memory) для from_pretrained."""
    import subprocess
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    mm = {}
    for line in out.strip().splitlines():
        i, total, used = (int(x) for x in line.split(", "))
        free = max(0, total - used - reserve_mb)
        if free > 500:
            mm[i] = f"{free}MiB"
    mm["cpu"] = f"{cpu_gb}GiB"
    return "auto", mm


def load_vlm(name: str, device: str = "cuda:0", dtype=torch.bfloat16):
    """name — id на HF или локальный путь. device="auto" — разложить по всем
    картам с учётом занятости (узел общий). Возвращает готовый бэкенд."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    # `модель::процессор` — когда в репозитории модели нет своего токенизатора.
    # У MolmoAct2 лежат только веса, картиночный процессор и chat-шаблон, поэтому
    # процессор берётся от базовой Molmo2, а шаблон подменяется на её собственный.
    proc_name = None
    if "::" in name:
        name, proc_name = name.split("::", 1)

    low = name.lower()
    if device == "cpu":
        dtype = torch.float32  # на CPU bf16-ядра есть не везде, float32 надёжнее
        torch.set_num_threads(int(__import__("os").environ.get("TORCH_THREADS", "64")))
    # transformers <5 ждёт torch_dtype, >=5 — dtype
    import transformers
    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    kw = {dtype_kw: dtype}
    if device == "auto":
        device_map, max_memory = auto_max_memory()
        kw.update(device_map=device_map, max_memory=max_memory)
        print(f"[load] device_map=auto, max_memory={max_memory}", flush=True)

    def place(model):
        if device == "auto":
            return model.eval(), str(getattr(model, "device", "cuda:0"))
        return model.eval().to(device), device

    if name.startswith("vla:"):
        # вариант 2: VLM-часть, вынутая из VLA-чекпойнта переклейкой имён тензоров
        from vla_vlm_loader import SPECS, load_vla_vlm
        model, processor, _ = load_vla_vlm(name[4:], dtype=dtype)
        model, dev = place(model)
        spec = SPECS[name[4:]]
        cls = PaliGemmaBackend if spec.get("backend") == "paligemma" else QwenVLBackend
        be = cls(model, processor, dev)
        be.answer_prefix = spec.get("answer_prefix", "")
        if be.answer_prefix:
            print(f"[vla] затравка ответа: {be.answer_prefix!r}", flush=True)
        return be

    if "paligemma" in low:
        from transformers import PaliGemmaForConditionalGeneration
        processor = AutoProcessor.from_pretrained(name)
        model, dev = place(PaliGemmaForConditionalGeneration.from_pretrained(name, **kw))
        return PaliGemmaBackend(model, processor, dev)

    if "florence" in low:
        from transformers import AutoModelForCausalLM as _AutoCLM
        # remote-code Florence-2 безусловно требует flash_attn, которого в окружении
        # нет и который ей на самом деле не нужен: убираем его из списка импортов.
        import transformers.dynamic_module_utils as dmu
        orig = dmu.get_imports
        dmu.get_imports = lambda f: [i for i in orig(f) if i != "flash_attn"]
        try:
            processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
            model, dev = place(_AutoCLM.from_pretrained(
                name, trust_remote_code=True, attn_implementation="eager", **kw))
        finally:
            dmu.get_imports = orig
        return Florence2Backend(model, processor, dev)

    if "molmo" in low:
        from transformers import AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(proc_name or name, trust_remote_code=True)
        if proc_name:
            # Шаблон берём от процессора, а не от модели: у MolmoAct2 он ставит
            # свои image-токены, которых нет в токенизаторе базы, и forward падает
            # на «Could not infer image counts». Семейство одно, шаблон совместим.
            print(f"[molmo] процессор и chat-шаблон от {proc_name}", flush=True)
        # Конфиг MolmoAct2 написан под transformers 5.x и не несёт use_cache,
        # который 4.5x спрашивает. Скорингу кэш не нужен: дописываем поле в конфиг
        # ДО создания модели — конструктор такой аргумент не принимает.
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        for c in (cfg, getattr(cfg, "text_config", None), getattr(cfg, "llm_config", None)):
            if c is not None and not hasattr(c, "use_cache"):
                c.use_cache = False
        model, dev = place(AutoModelForImageTextToText.from_pretrained(
            name, config=cfg, trust_remote_code=True, **kw))
        be = MolmoBackend(model, processor, dev)
        if proc_name:
            pcfg = AutoConfig.from_pretrained(proc_name, trust_remote_code=True)
            remap = {getattr(pcfg, k): getattr(cfg, k)
                     for k in vars(cfg) if k.endswith("_token_id")
                     and isinstance(getattr(cfg, k, None), int)
                     and isinstance(getattr(pcfg, k, None), int)
                     and getattr(pcfg, k) != getattr(cfg, k)}
            if remap:
                be.token_remap = remap
                print(f"[molmo] перенос спецтокенов из словаря {proc_name}: {remap}",
                      flush=True)
        return be

    if "spatialvla" in low:
        # PaliGemma2 + пространственные action-токены в расширенном словаре.
        # Свободной генерацией не отвечает, но голова цела — только логит-скоринг.
        from transformers import AutoModel
        processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        model, dev = place(AutoModel.from_pretrained(name, trust_remote_code=True, **kw))
        return PaliGemmaBackend(model, processor, dev)

    if "openvla" in low or "prismatic" in low:
        # Prismatic (DINOv2+SigLIP -> Llama-2), действия занимают 256 редких токенов
        # исходного словаря, поэтому голова цела и скоринг left/right осмыслен.
        from transformers import AutoModelForVision2Seq
        processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        model, dev = place(AutoModelForVision2Seq.from_pretrained(
            name, trust_remote_code=True, low_cpu_mem_usage=True, **kw))
        return OpenVLABackend(model, processor, dev)

    if "qwen" in low or "cosmos" in low or "rldx" in low:
        from transformers import AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(name)
        model, dev = place(AutoModelForImageTextToText.from_pretrained(name, **kw))
        return QwenVLBackend(model, processor, dev)

    if "magma" in low:
        processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        model, dev = place(AutoModelForCausalLM.from_pretrained(
            name, trust_remote_code=True, **kw))
        _fix_magma_convnext(model, name)
        return MagmaBackend(model, processor, dev)

    raise ValueError(f"неизвестная модель: {name}")


def _fix_magma_convnext(model, name):
    """ОБЯЗАТЕЛЬНО: чекпойнт Magma-8B хранит веса ConvNeXt под именами, которые
    from_pretrained не сопоставляет, и молча оставляет их на инициализации ~1e-5.
    Зрение при этом умирает без единой ошибки. Логика повторяет magma_vlm_qa.py."""
    import glob
    import os

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = name if os.path.isdir(name) else snapshot_download(name)
    tensors = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="pt") as fh:
            for k in fh.keys():
                if "vision_tower" in k or "convnext" in k.lower():
                    tensors[k] = fh.get_tensor(k)
    if not tensors:
        print("[magma] предупреждение: веса vision_tower в чекпойнте не найдены")
        return
    sd = model.state_dict()
    fixed = 0
    for k, v in tensors.items():
        for cand in (k, k.replace("model.", ""), "model." + k):
            if cand in sd and sd[cand].shape == v.shape:
                sd[cand].copy_(v.to(sd[cand].dtype))
                fixed += 1
                break
    print(f"[magma] восстановлено тензоров vision_tower: {fixed}/{len(tensors)}")
