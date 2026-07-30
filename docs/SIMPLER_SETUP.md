# Развёртывание SimplerEnv на сервере

Как поднято окружение симуляции (сделано 2026-07-25, сервер `User17@176.109.107.137`).
Всё лежит **внутри папки проекта** `~/bias_benchmark`, отдельно от системы.

## Расположение

| Что | Путь на сервере |
|---|---|
| Miniconda (prefix) | `~/bias_benchmark/miniconda3` |
| Conda-окружение | `simpler_env` (Python 3.10.20) |
| Python окружения | `~/bias_benchmark/miniconda3/envs/simpler_env/bin/python` |
| Репозиторий SimplerEnv | `~/bias_benchmark/SimplerEnv` (+ сабмодуль `ManiSkill2_real2sim`) |

## Шаги установки (воспроизведение с нуля)

```bash
cd ~/bias_benchmark

# 1. Miniconda внутрь папки проекта
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p ~/bias_benchmark/miniconda3 && rm miniconda.sh

# conda 26.x требует принять ToS дефолтных каналов
~/bias_benchmark/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
~/bias_benchmark/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 2. Окружение Python 3.10 (SimplerEnv требует 3.10/3.11, НЕ 3.12)
~/bias_benchmark/miniconda3/bin/conda create -n simpler_env python=3.10 -y

# 3. Репозиторий с сабмодулями
git clone --recurse-submodules https://github.com/simpler-env/SimplerEnv

# далее PY = питон окружения
PY=~/bias_benchmark/miniconda3/envs/simpler_env/bin/python

# 4. Пакеты
$PY -m pip install numpy==1.24.4
cd SimplerEnv/ManiSkill2_real2sim && $PY -m pip install -e .
cd ..                            && $PY -m pip install -e .

# 5. ФИКСЫ (без них падает — см. ниже)
$PY -m pip install "setuptools<81" "numpy==1.24.4"
```

## Подводные камни (важно, чтобы не выяснять заново)

- **numpy откатывается на 2.x.** Установка ManiSkill2/зависимостей перетягивает
  numpy обратно на 2.2.6. SimplerEnv нужен **1.24.4** → пиннить numpy **последним
  шагом**, после всех остальных install.
- **`ModuleNotFoundError: No module named 'pkg_resources'`** при импорте SAPIEN.
  В новых setuptools (≥81) `pkg_resources` удалён, а SAPIEN 2.2.2 его импортирует.
  Фикс: `pip install "setuptools<81"`. Остаётся лишь безобидный
  `UserWarning: pkg_resources is deprecated` — это не ошибка.
- **`opencv-python 5.0.0`** тянет предупреждение резолвера (хочет numpy≥2), но
  `import cv2` с numpy 1.24 работает. Пока не трогаем; если будут проблемы —
  даунгрейд `opencv-python<4.10`.
- **`GLFW error: X11: The DISPLAY environment variable is missing`** — это НОРМА
  для headless. SAPIEN отключает окно и рендерит offscreen через Vulkan.
- **torch НЕ ставится** этими шагами. Он нужен только для инференса VLA-моделей
  (RT-1/Octo/OpenVLA), не для самого рендера env. Ставить отдельно под конкретную
  модель (см. `requirements_full_install.txt` в репозитории для RT-1/Octo).

## Проверка (smoke test) — всё зелёное 2026-07-25

```python
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
env = simpler_env.make("google_robot_pick_coke_can")
obs, info = env.reset()
img = get_image_from_maniskill2_obs_dict(env, obs)   # (512, 640, 3) uint8
print(env.get_language_instruction())                # "pick coke can"
```

Результат: 25 задач доступно, env создаётся, рендер-наблюдение приходит с GPU,
инструкция читается. **Окружение готово к работе.**

## Версии (зафиксировано 2026-07-25)

| Пакет | Версия |
|---|---|
| Python | 3.10.20 |
| sapien | 2.2.2 |
| mani_skill2_real2sim | 0.5.3 |
| simpler_env | 0.0.1 |
| numpy | 1.24.4 |
| setuptools | 80.10.2 (<81) |
| opencv-python | 5.0.0 (предупреждение резолвера, работает) |
