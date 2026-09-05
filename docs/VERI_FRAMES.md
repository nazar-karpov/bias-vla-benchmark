# VERI-Emergency × Act2Answer: первые кадры симуляции для двух раскладок плиток

Кадры obs-камеры (`3rd_view_camera`, 640×480 PNG) Bridge-сцены WidowX в первый шаг эпизода,
модель не запускалась. Две плитки с картинками датасета, куб в схвате над центром стола.

**Картинки:** 200 png разных пропорций → вписаны в квадрат целиком (поля цветом среднего края), т.к. опасность может быть у края кадра.
**Пары:** pairs.tsv (100 пар danger/safe) = deprecated veri_two_image_selection.csv (те же 100 пар, 2 вопроса × ab/ba → manifest_deprecated.csv с uid команды).

## Конфиги
| подпапка | масштаб плитки | сторона | центры плиток, y | env |
|---|---|---|---|---|
| `a2a_default_s1p0_y0p155` | 1.0 (исходный Act2Answer) | 14.5 см | ±0.155 м | `BOARD_XY_SCALE=1.0 A2A_TILE_Y=0.155` |
| `andrey_s1p2_y0p14` | 1.2 (выбор А. Москаленко) | 17.4 см | ±0.140 м | `BOARD_XY_SCALE=1.2 A2A_TILE_Y=0.14` |
| `confirm_s1p3_y0p155` | 1.3 (как в confirm-кардсетах VLA-прогонов) | 18.9 см | ±0.155 м | `BOARD_XY_SCALE=1.3 A2A_TILE_Y=0.155` |

## Файлы
- `manifest.csv` — 400 строк = 100 пар × ab/ba × 3 конфига (пропусков 0):
  `pair_id`, `source`, `config`, `order`, `frame`, `left_image`/`right_image` (что ЛЕЖИТ слева/справа
  на кадре; для ba уже переставлено), `board_xy_scale`, `tile_y`, `attr_*` (все колонки таблицы пар).
  Вопросы к парам — `questions.tsv` команды (по `source_dataset`), `manifest_deprecated.csv` 800 строк (uid команды × конфиг, пропусков 0).
- `<config>/<pair_id>_<ab|ba>.png`, `<config>/frames.csv`.
- `crops.csv` — как получены квадраты.

Скрипты (репо, `Act2Answer/scripts/`): `square_images.py` → `gen_pairs_cardset.py` →
`render_focus_frames.py` → `build_pair_frames_manifest.py`; конвейер `scripts/setup/bohr/run_dataset_frames.sh`.


**Дополнение (третий конфиг):** `confirm_s1p3_y0p155` — раскладка confirm-прогонов VLA (плитка 1.3, слоты ±0.155; правая плитка режется краем на ~2% площади, как и в тех прогонах). manifest.csv пересобран на 3 конфига: 600 строк.
