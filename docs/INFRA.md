# Инфраструктура

## ⭐ Bohr — Selectel, рабочий сервер с 05.09.2026

`ssh moskalenko@176.114.85.176` (пароль у Назара; ключи Назара/агента в authorized_keys).
Ubuntu 24.04, 96 CPU, 251 GB RAM, 2× RTX 4090 48 GB (driver 580.173), диск 1.8 TB.
**sudo нет** → без apt/nvcc//workspace; всё в `/home/moskalenko/ws`:

| что | где |
|---|---|
| репа | `ws/bias-vla-benchmark-main` (push через deploy-ключ `~/.ssh/github_deploy`, `core.sshCommand` в конфиге) |
| веса HF | `ws/hf_cache` (HF_HOME) — перенесены с cloud.ru целиком |
| ассеты ManiSkill | `ws/maniskill_assets` (MS_ASSET_DIR) |
| conda | `ws/conda` (Miniconda), env `magma_act2answer`: py3.10, torch 2.4.0+cu121, sapien 3.0.0b1, mani_skill/simpler_env editable |
| экспорты env с cloud.ru | `ws/env_exports/*.yml, *.pip.txt` (internvla ещё не собран) |
| старые раннеры cloud.ru | `ws/setup/*.sh` (пути `/workspace/moskalenko` → менять на `~/ws`) |

Запуск: `source ~/ws/env_bohr.sh` → REPO_ROOT/HF_HOME/MS_ASSET_DIR/PYTHONPATH, env, cd SimplerEnv.
Приёмка 05.09: `scripts/setup/bohr/test_sapien_gpu.py` (GPU-Vulkan рендер ок, 0.9 с),
`smoke_traj_bohr.sh` (4 эпизода симулятора без модели, SMOKE_OK), `test_magma_bohr.sh`
(Magma-8B + рендер, 2 эпизода). Сборка env: `bohr_env_stage1.sh` → `bohr_env_stage2.sh`.


## Сервер для экспериментов

Прогоны VLA-моделей выполняются на удалённом сервере по SSH.

- **IP:** `176.109.107.137`
- **Пользователь (из `~/.ssh/config`):** `User17`
- **Пароль:** хранится локально в `.env` (`SERVER_PASSWORD`), в git **не** коммитится.

- **Рабочая папка на сервере:** `~/bias_benchmark` (с подчёркиванием; пустая на 2026-07-25).

### Подключение

```bash
ssh User17@176.109.107.137
cd ~/bias_benchmark
```

С Windows-машины неинтерактивно (в `~/.ssh/config` для этого IP — только User17):
```bash
plink -batch -hostkey "SHA256:mPAyR5qRJyB/LisWN0dohNx8eQXYkVwFlmdTha3/lXs" \
      -pw "<SERVER_PASSWORD из .env>" User17@176.109.107.137 '<команда>'
```

Пароль подставлять из `.env`. Не вставлять пароль в код, коммиты или логи.

> ⚠️ Секреты (`SERVER_PASSWORD` и т.п.) — только в `.env`, который в `.gitignore`.
> При добавлении новых доступов дописывать сюда, а сам секрет — в `.env`.

## Окружение сервера (проверено 2026-07-25)

| Параметр | Значение |
|---|---|
| Хостнейм | `school17` |
| ОС | Ubuntu 24.04.4 LTS |
| GPU | **4× Tesla V100-SXM3-32GB** (все свободны) |
| NVIDIA driver | 580.159.03 (nvidia-smi: CUDA 13.0) |
| CUDA toolkit (`nvcc`) | не установлен (для инференса не обязателен — хватает pip-колёс) |
| Системный Python | 3.12.3, **без pip и без conda** — нужно ставить окружение |
| sudo | работает с паролем из `.env` (passwordless — нет) |
| Интернет | есть (github/pip доступны) |
| Диск | `/` = 2.0 TB, свободно ~1.9 TB |

### Vulkan — ✅ рабочий (критично для SIMPLER/SAPIEN)

SIMPLER рендерит сцены через SAPIEN, которому нужен **Vulkan на GPU**. Проверено:

- Vulkan loader: `libvulkan.so.1`, инстанс 1.3.275.
- NVIDIA ICD зарегистрирован: `/usr/share/vulkan/icd.d/nvidia_icd.json`
  (`libGLX_nvidia.so.0`, api 1.4.312).
- `vulkaninfo --summary` видит **все 4 Tesla V100** как
  `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`, driver `NVIDIA_PROPRIETARY` 580.159.03.
- Есть также Mesa llvmpipe (CPU softrender) — игнорируем, нужен GPU.
- Утилита `vulkaninfo` доустановлена (`apt install vulkan-tools`).

**Вывод:** железо готово к SIMPLER. GPU для рендера можно выбирать через
переменные окружения (напр. `CUDA_VISIBLE_DEVICES`), т.к. карт четыре.

Развёртывание SimplerEnv и smoke-test — см. [SIMPLER_SETUP.md](SIMPLER_SETUP.md).

### Копирование файлов сервер↔локально (Windows)

```bash
# скачать с сервера:
"/c/Program Files/PuTTY/pscp" -batch -hostkey "SHA256:mPAyR5qRJyB/LisWN0dohNx8eQXYkVwFlmdTha3/lXs" \
    -pw "<SERVER_PASSWORD>" User17@176.109.107.137:/путь/на/сервере ./локально
```
