# VisBias × Act2Answer: первые кадры симуляции для двух раскладок плиток

Кадры obs-камеры (`3rd_view_camera`, 640×480 PNG) Bridge-сцены WidowX в первый шаг эпизода,
модель не запускалась. Две плитки с картинками датасета, куб в схвате над центром стола.

**Картинки:** 699 реальных фото (WEBP под .jpg; один AVIF исключён самой командой) → квадрат по лицу (Haar), 167 без детекции → центр.
**Пары:** pairs/gender.tsv (500) + ethnicity.tsv (2500) + profession.tsv (3500) = 6500 пар.
Deprecated-манифест (`visbias_two_image_selection.csv`) НЕ подмешивается — только актуальные tsv-таблицы
(08.09: убрал 65 пар/390 строк, которые раньше добавлялись из deprecated, см. `_deprecated_removed_backup/`
на Bohr).

## Конфиги
| подпапка | масштаб плитки | сторона | центры плиток, y | env |
|---|---|---|---|---|
| `a2a_default_s1p0_y0p155` | 1.0 (исходный Act2Answer) | 14.5 см | ±0.155 м | `BOARD_XY_SCALE=1.0 A2A_TILE_Y=0.155` |
| `andrey_s1p2_y0p14` | 1.2 (выбор А. Москаленко) | 17.4 см | ±0.140 м | `BOARD_XY_SCALE=1.2 A2A_TILE_Y=0.14` |
| `confirm_s1p3_y0p155` | 1.3 (как в confirm-кардсетах VLA-прогонов) | 18.9 см | ±0.155 м | `BOARD_XY_SCALE=1.3 A2A_TILE_Y=0.155` |

## Файлы
- `manifest.csv` — 39000 строк = 6500 пар × ab/ba × 3 конфига (пропусков 0):
  `pair_id`, `source` (всегда `tsv:<таблица>`), `config`, `order`, `frame`, `left_image`/`right_image`
  (что ЛЕЖИТ слева/справа на кадре; для ba уже переставлено), `board_xy_scale`, `tile_y`, `attr_*`
  (все колонки таблицы пар). Вопросы к парам — `questions.tsv` команды (по `source_dataset`).
- `<config>/<pair_id>_<ab|ba>.png`, `<config>/frames.csv`.
- `crops.csv` — как получены квадраты.

Скрипты (репо, `Act2Answer/scripts/`): `square_images.py` → `gen_pairs_cardset.py` →
`render_focus_frames.py` → `build_pair_frames_manifest.py`; конвейер `scripts/setup/bohr/run_dataset_frames.sh`.


**Дополнение (третий конфиг):** `confirm_s1p3_y0p155` — раскладка confirm-прогонов VLA (плитка 1.3, слоты ±0.155; правая плитка режется краем на ~2% площади, как и в тех прогонах). manifest.csv пересобран на 3 конфига.

**Правка 08.09:** из manifest.csv убраны 65 пар (390 строк), которые раньше подмешивались из
deprecated-манифеста `visbias_two_image_selection.csv` — они там не должны были быть (это не
актуальные пары команды). Итог: 39000 строк = 6500 пар × ab/ba × 3 конфига, только источник
`tsv:*`. `manifest_deprecated.csv` удалён как отдельный файл (был построен из того же
deprecated-источника). PNG-кадры этих 65 пар убраны из рабочих папок конфигов на Bohr/vlm8
в `_deprecated_removed_backup/`.
