# FOCUS/REFLECT × Act2Answer: первые кадры симуляции для трёх раскладок плиток

Кадры obs-камеры (`3rd_view_camera`, 640×480 PNG) Bridge-сцены WidowX в первый шаг эпизода,
модель не запускалась. Две плитки с фотографиями FOCUS (face-only контрфактуалы), куб в схвате
над центром стола.

**Картинки:** 480 фото REFLECT/FOCUS → квадрат по лицу (Haar-каскад на `base.jpg` сцены),
один бокс на сцену, чтобы пары оставались попиксельно параллельными вне лица (см. `crops.csv`).
**Пары:** актуальные таблицы команды `focus_reflect/pairs/`: gender.tsv (500) + ethnicity.tsv
(2500) + profession.tsv (3000) = **6000 пар**.

> **Правка 08.09.2026 (эксп. 52).** Раньше кардсет строился из deprecated-манифеста
> `focus_two_image_selection.csv` и содержал 2160 пар, из которых лишь 1200 есть в актуальных
> таблицах, а 4800 актуальных пар не рендерились вовсе. Пересобрано с `pairs/*.tsv`.
> Прежние кадры лежат на Bohr в `outputs/_deprecated_removed_backup/focus_frames_from_deprecated/`.

## Конфиги
| подпапка | масштаб плитки | сторона | центры плиток, y | env |
|---|---|---|---|---|
| `a2a_default_s1p0_y0p155` | 1.0 (исходный Act2Answer) | 14.5 см | ±0.155 м | `BOARD_XY_SCALE=1.0 A2A_TILE_Y=0.155` |
| `andrey_s1p2_y0p14` | 1.2 (выбор А. Москаленко) | 17.4 см | ±0.140 м | `BOARD_XY_SCALE=1.2 A2A_TILE_Y=0.14` |
| `confirm_s1p3_y0p155` | 1.3 (как в confirm-кардсетах VLA-прогонов) | 18.9 см | ±0.155 м | `BOARD_XY_SCALE=1.3 A2A_TILE_Y=0.155` |

## Файлы
- `manifest.csv` — 36000 строк = 6000 пар × ab/ba × 3 конфига: `pair_id` (= `pair_id` таблицы
  команды), `source` (`tsv:<таблица>`), `config`, `order`, `frame`, `left_image`/`right_image`
  (что ЛЕЖИТ слева/справа на кадре; для `ba` уже переставлено), `board_xy_scale`, `tile_y`,
  `attr_*` (все колонки таблиц пар: gender_1/2, ethnicity_1/2, profession_1/2, identity_1/2,
  same_identity и т.д.; пусто там, где колонки нет в исходной таблице).
- `questions.tsv` — 155 вопросов (150 из общей таблицы команды + 5 из veri_emergency).
  Задаются ВСЕ вопросы ко ВСЕМ парам: `source_dataset`/`source_format` — только происхождение
  вопроса, не фильтр; все уже в парном формате («Put the cube…» / «Which image, A or B…»).
  В манифест вопросы НЕ развёрнуты — кадр от вопроса не зависит, полный крест это 5.6 млн строк.
- `<config>/<pair_id>_<ab|ba>.png`, `<config>/frames.csv`.
- `crops.csv` — как получены квадраты.

## Как читать кадр
Плитка слева на кадре = `left_image`, справа = `right_image`. У каждой пары есть оба порядка
(`ab`/`ba`) — контроль позиционного крена.

Скрипты (репо, `Act2Answer/scripts/`): `focus_square_crops.py` → `gen_pairs_cardset.py` →
`render_focus_frames.py` → `build_pair_frames_manifest.py`; раннер `scripts/setup/bohr/rerender_focus_tsv.sh`.
