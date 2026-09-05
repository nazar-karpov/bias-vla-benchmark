# FOCUS × Act2Answer: первые кадры симуляции для двух раскладок плиток

Кадры obs-камеры WidowX/Bridge-сцены (`3rd_view_camera`, 640×480 PNG) — ровно то, что видит
VLA в первый шаг эпизода; модель не запускалась. На столе две плитки с фотографиями FOCUS
(REFLECT, face-only counterfactuals), куб в схвате над центром стола.

## Конфиги (подпапки)

| подпапка | масштаб плитки | сторона плитки | центры плиток, y | просвет | env |
|---|---|---|---|---|---|
| `a2a_default_s1p0_y0p155` | 1.0 (исходный Act2Answer) | 14.5 см | ±0.155 м | 16.5 см | `BOARD_XY_SCALE=1.0 A2A_TILE_Y=0.155` |
| `andrey_s1p2_y0p14` | 1.2 (выбор А. Москаленко, 03.09) | 17.4 см | ±0.140 м | 10.6 см | `BOARD_XY_SCALE=1.2 A2A_TILE_Y=0.14` |

Оба конфига: обе плитки целиком в кадре (запас правой до края +11 px и +8 px), одна и та же
камера, один сид (0), одинаковая поза робота.

## Файлы

- `manifest.csv` — строка на (uid × конфиг), 25 920 строк. Колонки: `uid` (как в
  `focus_two_image_selection.csv` / `focus_vlm_parallel_two_image_selection.csv` команды),
  `config`, `frame` (путь относительно этой папки), `question_vla`, `question_vlm`,
  `left_image`, `right_image` (что ЛЕЖИТ слева/справа на кадре), `occupation`, `left_group`,
  `right_group`, `attribute`, `order` (ab/ba), `board_xy_scale`, `tile_y`.
  Один кадр обслуживает 3 вопроса (income/education/safety): uid'ы с разным атрибутом
  ссылаются на один и тот же PNG.
- `<config>/<uid_base>_<ab|ba>.png` — 4 320 кадров на конфиг (2 160 пар × 2 порядка).
- `<config>/frames.csv` — то же без вопросов.
- `crops.csv` — как резались квадраты из прямоугольных фото FOCUS: один бокс на сцену
  (min(W,H) со стороной, центр — по лицу Haar-каскадом на base.jpg; 44/48 сцен по base,
  3 по медиане детекций вариантов, 1 центр кадра), потом 512×512. Пары остаются
  попиксельно параллельными вне лица.

## Как читать кадр

Плитка слева на кадре = `left_image`, справа = `right_image`. Для порядка `ba` картинки
манифеста уже переставлены (симулятор физически меняет плитки местами). Контроль
позиционного крена: у каждой пары есть оба порядка, `uid` отличается суффиксом.

Скрипты (репо bias-vla-benchmark, `Act2Answer/scripts/`): `focus_square_crops.py` →
`gen_focus_cardset.py` → `render_focus_frames.py` → `build_focus_frames_manifest.py`.
