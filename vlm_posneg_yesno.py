#!/usr/bin/env python3
"""YES/NO с ПОЗИТИВНЫМ и НЕГАТИВНЫМ вопросом — контроль салиентности картинки.

ЗАЧЕМ: в PAIRS у каждой сцены есть пара утверждений:
    positive: "who is more probably the boss"
    negative: "who is more probably the employee"
Наш build_pairs_csv.py брал только positive, и half дизайна датасета терялась.

Негатив снимает главный конфаунд (сцены в PAIRS перерисовываются целиком: фон
различается на 69% от того, насколько различается человек — измерено):
  - если персонажа просто нарисовали «солиднее/заметнее», вырастет P(yes) И на boss,
    И на employee (общая склонность отвечать «да» на этой картинке);
  - если это СЕМАНТИЧЕСКИЙ байес, boss вырастет, а employee УПАДЁТ.
Метрика: S = P(yes|positive) − P(yes|negative) — вычитает всё, что относится к картинке
как таковой. Байес = разница S между демографиями.

Источник вопросов: datasets/PAIRS/pairs.json — это JSONL (по объекту на строку!),
поля image/positive/negative.

Запуск: см. vlm_single_yesno.py (те же env/бэкенды).
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_single_yesno import make_question, score_yes_no, load_backend  # noqa: E402


def attr_of(statement: str):
    m = re.search(r"more probably (?:a |an )?(.+)", statement.strip())
    return m.group(1).strip().rstrip("?.") if m else None


def pairs_map(jsonl: Path):
    """scene ('status/city') -> СПИСОК уникальных (positive_attr, negative_attr).

    ВАЖНО: в PAIRS 520 строк = 200 картинок × вопрос. У status-картинок ПЯТЬ разных
    вопросов (boss/employee, wealthy/poor, ...), у occupations и crime — один.
    Раньше здесь был dict со scene-ключом, и пять вопросов схлопывались в последний
    (терялся в т.ч. boss). Теперь копим все.
    """
    m = defaultdict(list)
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        img = r.get("image", "")
        scene = "/".join(img.split("/")[:2])
        p, n = attr_of(r.get("positive", "")), attr_of(r.get("negative", ""))
        if scene and p and n and (p, n) not in m[scene]:
            m[scene].append((p, n))
    return dict(m)


def find_image(scene_dir: Path, demo: str):
    """Картинка демографии в сцене. В датасете есть ОДИН нестандартный файл —
    status/phone/white_man1.png — поэтому не жёсткое имя, а glob по префиксу."""
    exact = scene_dir / f"{demo}.png"
    if exact.exists():
        return exact
    cand = sorted(scene_dir.glob(f"{demo}*.png"))
    return cand[0] if cand else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("magma", "paligemma", "qwenbase"), required=True)
    ap.add_argument("--pairs-root", type=Path,
                    default=Path.home() / "bias_benchmark/datasets/PAIRS")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--all-questions", action="store_true",
                    help="задавать КАЖДЫЙ из 33 уникальных вопросов на КАЖДОЙ сцене "
                         "(n=50 сцен на вопрос вместо 1-20). 33*200*2=13200 запросов.")
    ap.add_argument("--only-question", default=None,
                    help="подстрока: оставить только вопросы, где она встречается "
                         "(напр. 'boss' -> быстрый прогон 400 запросов).")
    args = ap.parse_args()

    jsonl = args.pairs_root / "pairs.json"
    data_root = args.pairs_root / "data"
    pm = pairs_map(jsonl)
    if args.all_questions:
        # ВСЕ уникальные пары вопросов на КАЖДОЙ сцене: у каждого вопроса n=50 сцен.
        # Чинит перекос — «the boss» имел n=20 сцен, а «a pilot» всего n=1 (airplane).
        allq = []
        for v in pm.values():
            for qp in v:
                if qp not in allq:
                    allq.append(qp)
        pm = {scene: allq for scene in pm}
        print(f"ALL-QUESTIONS: {len(allq)} уникальных пар × {len(pm)} сцен", flush=True)
    if args.only_question:
        s = args.only_question.lower()
        pm = {sc: [(p, n) for p, n in v if s in p.lower() or s in n.lower()]
              for sc, v in pm.items()}
        pm = {sc: v for sc, v in pm.items() if v}
        print(f"ONLY-QUESTION {args.only_question!r}: осталось сцен {len(pm)}", flush=True)
    nq = sum(len(v) for v in pm.values())
    print(f"сцен: {len(pm)}, пар вопросов всего: {nq} "
          f"(2 запроса × 4 демографии на каждую -> {nq*8} запросов)", flush=True)

    model, processor, build_inputs = load_backend(args.model, args.device)

    results = []
    for scene in sorted(pm):
        for demo in ("white_man", "white_woman", "black_man", "black_woman"):
            path = find_image(data_root / scene, demo)
            if path is None:
                print(f"  skip {scene}/{demo}: нет файла", flush=True)
                continue
            image = Image.open(path).convert("RGB")
            race = "white" if "white" in demo else "black"
            sex = "woman" if "woman" in demo else "man"
            tasks = [(p, "pos") for p, n in pm[scene]] + [(n, "neg") for p, n in pm[scene]]
            for attr, poln in tasks:
                q = make_question(attr)
                py, ly, ln = score_yes_no(model, processor, image, q, args.device, build_inputs)
                results.append({
                    "scene": scene, "prototype": scene.replace("/", "__"),
                    "race": race, "sex": sex, "demo": demo,
                    # реальный путь к файлу: в датасете есть нестандартные имена
                    # (status/phone/white_man1.png), нужен для прослеживаемости
                    "image_path": str(path.relative_to(data_root)),
                    "polarity": poln, "attribute": attr, "question": q,
                    "p_yes": py, "logit_yes": ly, "logit_no": ln,
                })
        print(f"  {scene}: " + "; ".join(f"{p!r}/{n!r}" for p, n in pm[scene]), flush=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nDone: {len(results)} queries -> {args.out}")


if __name__ == "__main__":
    main()
