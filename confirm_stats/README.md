# confirm_stats — сырые stats.yaml канонических confirm-прогонов

По одному yaml на шард (last_info: финальные метрики + СЫРЫЕ координаты
cube_f{x,y,z}, board{L,R}_{x,y} каждого эпизода). Глобальный ep = s<start> + local_idx.
Позволяют пересчитать chosen_side с ЛЮБЫМИ зонами зачёта (см.
scripts/recompute_demographic_bias.py, bias_stats_ci.py, recompute_bbox_sweep.py)
без доступа к эфемерным cloud.ru-нодам.

- confirm-mid-magma-ALL-* — Magma, mid-тайлы (64 шарда по 50)
- confirm-internvla-FULL-* — InternVLA-M1 (34 шарда по 100)
- confirm-rldx-FULL-* — RLDX-1 (34 шарда по 100)
- confirm-svla-ALL-* — SpatialVLA (64 шарда по 50)

Каждый прогон = 1600 эп × {noswap, swap}. Источник истины для всех S-метрик.
