# Инфраструктура

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
