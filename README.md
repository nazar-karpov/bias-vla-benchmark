# bias-vla-benchmark

Бенчмарк **демографического bias у VLA-моделей** (Vision-Language-Action) и VLM.
Вопрос простой: если поставить роботу задачу вроде «положи кубик на **босса**», а на
столе две плитки-портрета (например мужчина и женщина), — **предвзято ли** робот
выбирает плитку по полу/расе? И отличается ли это от того, что та же модель отвечает
как чистый VLM (просто по картинке, без моторики)?

Данные — из датасета **PAIRS** (occupations / status / crime). Симуляция — на
**Act2Answer + ManiSkill/SimplerEnv** (робот реально двигает кубик). Замеряем сдвиг
выбора в процентных пунктах с обязательными контролями (полярность, порядок, валидность).

> Полная методика и грабли — в [`BIAS_EXPERIMENTS.md`](BIAS_EXPERIMENTS.md),
> результаты по моделям — в [`docs/RESULTS.md`](docs/RESULTS.md),
> повествовательный журнал — в [`docs/JOURNAL.md`](docs/JOURNAL.md),
> таблица всех прогонов — в [`EXPERIMENTS.md`](EXPERIMENTS.md).

---

## Кроп — ключевая фича

На полных портретах PAIRS человек занимает малую долю кадра (много фона), и в
симуляции плитка ещё мельче — bias «тонет». Поэтому есть **кропнутые** кардсеты:
плитка обрезается по центру на человека и ресайзится в 512×512 (тот размер текстуры,
что ждёт `make_cardset.py`), без искажения пропорций. Вопросы/эпизоды те же — меняются
**только текстуры**, что изолирует эффект «даём модели лучше разглядеть содержимое».

| скрипт | что делает |
|---|---|
| [`scripts/crop_pairs_images.py`](scripts/crop_pairs_images.py) | центр-кроп плиток PAIRS → 512×512 (доля кадра `--frac`, смещение вверх под лица) |
| [`scripts/run_cropped_benchmark.sh`](scripts/run_cropped_benchmark.sh) | **resumable** драйвер: собрать кроп-кардсет → поставить модели → прогнать magma/spatialvla/internvla/rldx (noswap+swap). Маркеры состояния, skip-on-fail |
| [`Act2Answer/scripts/crop_sim_frames.py`](Act2Answer/scripts/crop_sim_frames.py) | кроп уже отрендеренных сим-кадров вокруг плиток (без симулятора) — проверка «плитки слишком мелкие» |

Кропнутые кардсеты: `pairs_bias_crop`, `pairs_choice_crop`, `pairs_choice_vla_confirm`
(последний — весь на кропе, проверено по хешам мешей).

> ⚠️ Сами меши плиток (`*.glb`) **не в git** (тяжёлые, генерируются скриптами).
> В репозитории лежит **структура** кардсета (`pairs.json`, `model_db.json`,
> `blocks.json`) и **скрипты генерации**. Чтобы получить меши — прогнать
> `crop_pairs_images.py` + `make_cardset.py` (см. шаг 4).

---

## Структура репозитория

```
.
├── README.md                 ← этот файл
├── BIAS_EXPERIMENTS.md       методика замера bias + обязательные контроли
├── EXPERIMENTS.md            таблица всех прогонов (номер, флаги, датасет, результат)
├── CLAUDE.md                 правила работы в репо (журнал, метрики, контроли)
├── metrics/                  сводные CSV: metrics_{yesno,choice,comparison}.csv
├── docs/
│   ├── RESULTS.md            результаты по моделям (главная сводка)
│   ├── JOURNAL.md            повествовательный журнал (свежее сверху)
│   ├── INFRA.md              инфраструктура (серверы, окружения)
│   ├── SIMPLER_SETUP.md      установка SimplerEnv/симулятора
│   └── bias_benchmark_results.xlsx
│
├── scripts/                  ← «тонкая» линия: VLM-probing, кроп, анализ
│   ├── crop_pairs_images.py, run_cropped_benchmark.sh   (кроп)
│   ├── magma_vlm_qa.py, paligemma_vlm_qa.py             (VLM-опрос напрямую)
│   ├── magma_probe.py, magma_steering.py, magma_logit_lens.py,
│   │   magma_extract_acts.py, magma_counterfactual.py   (мех. интерпретация)
│   ├── analyze_bias.py, bias_detail.py, compare_vlm_vla.py, final_summary_crop.py
│   ├── make_cardset.py                                  (сборка кардсета из плиток)
│   └── {internvla,rldx}_server.sh, setup_rldx.sh, fetch_internvla_ckpt.sh
│
├── outputs_local/cropped_run/   результаты кроп-прогона (RESULTS_cropped_pairs.md, json)
│
└── Act2Answer/               ← «толстая» линия: симулятор + прогоны
    ├── scripts/              launch_*.sh (core6/fastvla/night/simchoice),
    │                         confirm_v100.sh, make_sim_choice_cardset.py,
    │                         render_sim_choice_frames.py, vla_fast_summary.py,
    │                         sim_variant_summary.py, build_metrics_table.py, ...
    ├── ManiSkill/            движок симуляции + assets/carrot/<кардсеты> (JSON, без .glb)
    ├── SimplerEnv/           обёртка eval (simpler_env.eval), политики моделей
    └── ...                   (InternVLA-M1, RLDX-1, RL4VLA — клоны upstream, см. .gitignore)
```

**Почему две папки со скриптами.** Репозиторий сведён из двух линий разработки:
`scripts/` (в корне) — прямой VLM-опрос, кроп-эксперименты и механистическая
интерпретация Magma; `Act2Answer/scripts/` — оркестрация VLA-прогонов в симуляторе.
Обе сохранены намеренно.

---

## Требования

- Python 3.10/3.11, CUDA-GPU (V100/H100 — на чём гоняли).
- Симулятор: SimplerEnv + ManiSkill (SAPIEN/Vulkan). Установка — [`docs/SIMPLER_SETUP.md`](docs/SIMPLER_SETUP.md).
- У каждой модели своё окружение (transformers-версии конфликтуют) — ставится
  отдельно (`setup_*` / `*_server.sh`).
- Апстрим-репы моделей (InternVLA-M1, RLDX-1, RL4VLA) клонируются отдельно —
  они в `.gitignore`.

---

## Воспроизведение — по шагам

### 1. Код
```bash
git clone <this-repo> bias-vla-benchmark && cd bias-vla-benchmark
```

### 2. Внешние репо (нужны только для части моделей)
```bash
# SimplerEnv/ManiSkill — см. docs/SIMPLER_SETUP.md
# InternVLA-M1, RLDX-1 — клонировать в Act2Answer/ (в .gitignore)
```

### 3. Окружение модели
```bash
scripts/setup_rldx.sh            # пример; у каждой модели своё
scripts/internvla_server.sh      # zmq-сервер для server-based моделей
```

### 4. Сгенерировать кардсет (в т.ч. кропнутый)
```bash
# полный набор плиток PAIRS -> плоская папка
# кроп на человека:
python scripts/crop_pairs_images.py --in tiles/ --out tiles_crop/ --frac 0.65
# собрать кардсет (создаёт textured.glb — их нет в git):
python scripts/make_cardset.py  --tiles tiles_crop/ --name pairs_bias_crop
```

### 5. Прогнать модель в симуляторе
```bash
cd Act2Answer/SimplerEnv
python -m simpler_env.eval --vla magma \
  --assets pairs_choice_vla_confirm --obj-set test \
  --start-id 0 --count 1600 --shard-size 50 --buffer-inferbatch 8 [--do-swap]
# или одним resumable-драйвером сразу по всем моделям:
bash scripts/run_cropped_benchmark.sh
```
Контрбаланс порядка — двумя прогонами `--do-swap` / без него.

### 6. (опц.) Прямой VLM-опрос — «не знает» vs «не доехал»
```bash
python scripts/magma_vlm_qa.py       --assets pairs_bias_crop
python scripts/paligemma_vlm_qa.py   --assets pairs_bias_crop
```

### 7. Посчитать bias и пересобрать метрики
```bash
python Act2Answer/scripts/build_metrics_table.py --out-dir metrics/
python Act2Answer/scripts/compare_designs.py
python Act2Answer/scripts/vla_fast_summary.py --model magma --prefix confirm \
  --assets pairs_choice_vla_confirm
```
→ `metrics/metrics_{yesno,choice,comparison}.csv`.

---

## Обязательные контроли (иначе ложные результаты — проверено)

- **Негативная полярность** (напр. `boss` + `employee`) — снимает эффект «заметности» картинки.
- **Контрбаланс порядка** (ab/ba, `--do-swap`) — снимает позиционный крен (у PaliGemma до 20пп).
- **Проверка валидности** (чужой атрибут, `--controls`) — если ответ одинаков на свой и
  чужой атрибут, замер бессмыслен.

Подробнее — в [`BIAS_EXPERIMENTS.md`](BIAS_EXPERIMENTS.md) и [`CLAUDE.md`](CLAUDE.md).

---

## Механистическая интерпретация (Magma)

Отдельная линия — заглянуть внутрь Magma: `magma_probe.py` (линейные пробы),
`magma_logit_lens.py` (logit lens), `magma_steering.py` (стиринг активаций),
`magma_extract_acts.py`, `magma_counterfactual.py`.

---

## Заметки

- `answer` в `pairs_bias/pairs.json` **фиктивный** (проставлен чередованием) — у вопроса
  «кто вероятнее босс» правильного ответа нет; не считать по нему success.
- PAIRS **не параллелен**: смена демографии меняет ~35% пикселей и фон.
- Данные/меши/веса/медиа — **не в git** (`.gitignore`); в репо только код и структуры.
- Правила ведения журнала и метрик для будущей работы — в [`CLAUDE.md`](CLAUDE.md).

## Благодарности

PAIRS (демографические пары), Act2Answer, ManiSkill / SimplerEnv (SAPIEN).
