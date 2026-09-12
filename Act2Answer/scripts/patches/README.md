# Локальные патчи внешних репозиториев (не отслеживаются основной репой)

- `isaac_gr00t_x86_cosmos_local.patch` — Isaac-GR00T (12.09.2026): (1) `pyproject.toml`: только
  x86_64 (aarch64-колесо torchcodec в репе — LFS-заглушка, ломает `uv sync`); (2) бэкбон
  Cosmos-Reason2-2B гейтится на HF — без токена путь берётся из `GR00T_COSMOS_PATH`
  (локальный снапшот кеша). Применять: `cd Isaac-GR00T && git apply ../Act2Answer/scripts/patches/...`,
  затем `rm uv.lock && uv sync --python 3.12`. Сервер Xiaomi (`deploy/server.py`) живёт в этом же venv
  с `XR0_ATTN=flash_attention_2` (MiBoTForActionGeneration не проходит sdpa-проверку tf 4.57.3).
