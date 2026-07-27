# bias-vla-benchmark

Бенчмарк для измерения **социальных смещений (bias)** у Vision-Language-Action (VLA)
и Vision-Language (VLM) моделей. Построен поверх
[Act2Answer](https://github.com/CognitiveAISystems/Act2Answer): модель отвечает на
вопрос **действием** — ставит захваченный кубик на ту из двух карточек-плиток,
которую считает верным ответом. Мы генерируем контрфактические пары карточек
(меняем только пол/расу/атрибут человека на изображении) и смотрим, различается ли
поведение модели.

> **Что в этом репозитории:** только **код** — форк Act2Answer с нашими правками,
> скрипты генерации карточек, прогонов и анализа bias. Данные (меши карточек,
> изображения, датасеты), веса моделей и логи **намеренно не хранятся** в git
> (см. [`.gitignore`](.gitignore)) — они генерируются/скачиваются по шагам ниже.

---

## Структура

```
Act2Answer/                 форк https://github.com/CognitiveAISystems/Act2Answer
  ManiSkill/                вендорится (симулятор сцен, SAPIEN/Vulkan)
  SimplerEnv/               вендорится (обёртка политик + eval-петля)
  openvla/                  вендорится (prismatic / OpenVLA)
  scripts/                  <-- НАШ КОД
    setup/                  установка окружений и клон внешних репо
    make_cardset.py         генерация карточек Act2Answer из своих картинок
    eval_<model>.sh         прогон VLA-модели в симуляторе
    run_vlmqa.sh            прямой VLM-опрос (Magma) на тех же эпизодах
    magma_*.py              probing / logit-lens / steering / counterfactual
    analyze_bias.py         сводка bias по результатам прогона
    ...
  ManiSkill/mani_skill/assets/carrot/<cardset>/   карточки (генерируются, не в git)
build_pairs_csv.py          сборка CSV пар (left,right,question,answer) для карточек
```

**Наши правки к апстриму Act2Answer** (три файла — можно посмотреть `git log`/`git diff`
внутри форка, если история подключена):
- `SimplerEnv/simpler_env/policies/magma/magma_model.py` — Magma как политика / VLM-QA
- `SimplerEnv/simpler_env/run.py`
- `ManiSkill/.../bridge_dataset_eval/put_on_in_scene_multi_v4.py` — сцена с двумя карточками

---

## Требования

- Linux, NVIDIA GPU (тестировалось на V100, CUDA 12.1), рабочий **Vulkan** (для SAPIEN/ManiSkill).
- `conda` (miniconda/mamba). Каждая модель ставится в **отдельное окружение**.
- Интернет для скачивания весов моделей с HuggingFace.

---

## Воспроизведение — по шагам

Все скрипты сами подтягивают конфиг из [`Act2Answer/scripts/env.sh`](Act2Answer/scripts/env.sh)
(пути, `PYTHONPATH`, каталоги логов/выходов). Переменные можно переопределять через
окружение (`ASSETS=...`, `COUNT=...`, `EVAL_GPU=...` и т.д.).

### 1. Получить код

```bash
git clone https://github.com/nazar-karpov/bias-vla-benchmark.git
cd bias-vla-benchmark
```

`ManiSkill`, `SimplerEnv`, `openvla` уже лежат внутри `Act2Answer/` (вендорены).

### 2. Клонировать внешние репо (нужны только для части моделей)

```bash
cd Act2Answer
bash scripts/setup/clone_external_repos.sh
```
Клонирует рядом с `Act2Answer/`:
[RL4VLA](https://github.com/gen-robot/RL4VLA) (pi0),
[Xiaomi-Robotics-0](https://github.com/XiaomiRobotics/Xiaomi-Robotics-0),
[InternVLA-M1](https://github.com/InternRobotics/InternVLA-M1),
[molmoact2](https://github.com/allenai/molmoact2).
Путь можно сменить через `A2A_EXTERNAL_DIR`.

> Для нашего основного bias-пайплайна (Magma) внешние репо не обязательны.

### 3. Поставить окружение модели

Один скрипт на модель, создаёт conda-env и ставит зависимости:

```bash
bash scripts/setup/setup_magma_env.sh       # Magma (основная модель bias-анализа)
# либо: setup_openvla_env.sh, setup_spatialvla_env.sh, setup_pi0_env.sh, ...
```

### 4. Сгенерировать набор карточек

Карточка = плоская текстурированная плитка с картинкой на верхней грани.
Из папки картинок + CSV пар получаем drop-in ассет:

```bash
python scripts/make_cardset.py \
  --images    <папка с PNG/JPG> \
  --questions <pairs.csv> \
  --out       ManiSkill/mani_skill/assets/carrot/<cardset>
```
`pairs.csv` — колонки `left,right,question,answer` (`answer` = `Left|Right`;
`left`/`right` = имена файлов картинок без расширения). Собрать такой CSV из
готового датасета помогает [`build_pairs_csv.py`](build_pairs_csv.py).
Наши наборы: `pairs_bias` (PAIRS: occupations/status/crime), `safeeditbench`.

### 5. Прогнать модель в симуляторе

```bash
ASSETS=pairs_bias COUNT=100 EVAL_GPU=0 bash scripts/eval_magma.sh
```
Прогоняет эпизоды дважды — `noswap` и `swap` (карточки местами), чтобы отделить
bias от позиционного предпочтения. Результаты (`*_stats.yaml`, логи) пишутся в
`Act2Answer/outputs/` и `Act2Answer/logs/` (не коммитятся).

### 6. (опц.) Прямой VLM-опрос — «не знает» vs «не доехал»

```bash
GPU=0 bash scripts/run_vlmqa.sh 100 run1
```
Спрашивает Magma-как-VLM тот же вопрос текстом на тех же кадрах — чтобы отделить
незнание ответа от неудачи манипуляции. Пишет `outputs/magma_vlm_qa_pairs_bias_run1.json`.

### 7. Посчитать bias

```bash
python scripts/analyze_bias.py \
  ManiSkill/mani_skill/assets/carrot/pairs_bias/pairs.json \
  outputs/<noswap>_stats.yaml \
  outputs/<swap>_stats.yaml \
  magma
```
Даёт разбивку по **полу** (man vs woman, раса фиксирована) и **расе**
(white vs black, пол фиксирован). Родственные скрипты:
`analyze_counterfactual.py`, `bias_detail.py`, `compare_vlm_vla.py`.

---

## Механистическая интерпретация (Magma)

Скрипты `magma_*` работают на тех же эпизодах:
`magma_extract_acts.py` (активации) → `magma_probe.py` (линейные пробы) /
`magma_logit_lens.py` (logit lens) / `magma_steering.py` (стиринг) /
`magma_counterfactual.py`. Запуск-обёртки: `run_lens.sh`, `run_steering.sh`, `run_pg.sh`.

---

## Заметки для агентов

- Точка входа в код — **`Act2Answer/scripts/`**. Всё остальное в `Act2Answer/`
  (`ManiSkill`, `SimplerEnv`, `openvla`) — вендоренный апстрим, править осторожно.
- Скрипты **обязательно** сначала `source scripts/env.sh` — оттуда берутся пути и `PYTHONPATH`.
- Ассеты карточек кладутся строго в `Act2Answer/ManiSkill/mani_skill/assets/carrot/<name>/`
  и передаются флагом `--assets <name>`.
- Данных/весов/логов в репозитории нет — их надо сгенерировать (шаги 4–6) или скачать.

## Благодарности

Форк [Act2Answer](https://github.com/CognitiveAISystems/Act2Answer)
(*"Does VLA Even Know the Basics?"*, [arXiv:2606.19297](https://arxiv.org/abs/2606.19297)).
Симуляция — [ManiSkill](https://github.com/haosulab/ManiSkill) /
[SimplerEnv](https://github.com/simpler-env/SimplerEnv).
