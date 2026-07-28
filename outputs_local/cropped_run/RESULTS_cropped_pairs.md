# Cropped-PAIRS multimodel benchmark — results

Dataset: **pairs_bias_crop** (PAIRS bias cardset, center-cropped onto the person so
the subject fills the tile — 520 episodes, 55 per layout). Each model answers by
placing the cube on the tile it picks; run in two layouts, **noswap** and **swap**
(tiles L/R exchanged). The noswap↔swap gap isolates positional bias from content
choice (a content-driven model scores similarly in both; a position-driven one flips).

`success` = Act2Answer `is_success` mean over 55 episodes.

| Model        | noswap | swap  | \|gap\| |
|--------------|-------:|------:|------:|
| magma        | 0.268  | 0.307 | 0.040 |
| spatialvla   | 0.365  | 0.317 | 0.048 |
| internvla    | 0.364  | 0.527 | 0.164 |
| rldx         | 0.098  | 0.122 | 0.025 |

Notes
- All four models completed both layouts; no model was skipped/FAILED.
- **magma, spatialvla** — small noswap↔swap gap: choice is largely content-consistent
  across layouts (low positional bias), success ~0.3.
- **internvla** — large gap (0.36 → 0.53): its answers shift substantially with tile
  side, i.e. strong positional dependence on this cropped set.
- **rldx** — low absolute success (~0.1) via the Act2Answer client adapter; it answers
  every episode (is_answered 1.0) but the manipulation/action mapping lands the cube
  correctly less often. Gap is small. (RLDX runs server-side with sdpa on V100 and a
  per-step client round-trip, so it is slow but produces valid results.)

Raw per-layout logs: `~/bias_benchmark/nazar_folder/cropped_run/logs/` on the server.
Machine-readable: `results_summary.json` (this dir).
