# Frozen 200-question probe rerun

This version recalculates Magma, SpatialVLA and Xiaomi Robotics-0 probes using
the exact 200 questions supplied in `questions_200.md`. That file is a byte-for-byte
copy of the requested inventory. Its introductory selection instructions are
archived prose; this runner does not read scores, rerank questions or select a
different subset. `selection.json` maps every retained stable QID to exactly one
original canonical question ID by exact action text. `questions_216_reference.md`
preserves the original stable-ID inventory.

The selected manifest retains 80,000 original binary lines in their original
order: 200 questions, 400 frames and 200 pairs from `andrey_s1p2_y0p14`.
`prepare_selection.py --manifest ORIGINAL_MANIFEST --check` verifies the frozen
mapping and checksums. Its optional `--filtered-manifest NEW_PATH` writes the
selected lines without changing their UIDs, text, metadata or line endings.

`subset_cache.py` reads a complete, verified 216-question cache and physically
copies the selected rows into a separate 200-question cache. Copies preserve
native dtype and every stored bit, including BF16 bit encoding. Feature hashes,
source row indices, source cache identity and selection checksums record this
derivation. Original activation files, full caches, manifests and 216-question
results remain unchanged. The new caches and results use separate directories;
all existing 216-question history remains available for comparison.

Every training scaler and classifier is fitted again on the selected inputs.
No 216-question scaler or classifier is reused. Each feature has independent
male, Black and White membership fits, with all 200 questions pooled. The target
is the demographic membership of the right-hand person. Each slice retains
100 pairs and 40,000 records. The seed-42 split assigns 50 pairs to training and
50 to testing, keeping both layouts and all questions of a pair together.
Black and White fits use the same pair split and complementary targets.

| Model | Prefix features | Decode/flow features | Action feature width | Features / independent fits |
|---|---:|---:|---:|---:|
| Magma | 32 layers, width 4096 | 6 steps × 32 layers, width 4096 | 7 token IDs | 225 / 675 |
| SpatialVLA | 26 layers, width 2304 | 12 steps × 26 layers, width 2304 | 28 coordinates | 339 / 1017 |
| Xiaomi Robotics-0 | 36 VLM layers, width 2560 | 5 DiT flow steps × 16 layers, width 1024 | 960 coordinates | 117 / 351 |

The baseline uses StandardScaler and L2 logistic regression with LBFGS, C=1,
tolerance 1e-4 and a 2,000-iteration ceiling. Strict refinement independently
refits every feature/slice from zero with gtol=1e-10, ftol=0 and at most 50,000
iterations. Each strict scaler must exactly match its new 200-question baseline
scaler. A fit must have successful optimizer termination and an independently
checked infinity-norm gradient at most 1e-7. Selective repair may increase only
the failed strict fits' line-search allowance; it preserves the objective,
data, split, scaler and stopping thresholds. CPU stages use two workers with
one thread per worker.

Held-out predictions produce a layout-balanced named-group selection rate
`p_named` separately for each question, with `p_other = 1 - p_named`.
Signed SPD is `p_named - p_other`, absolute SPD is its absolute value, and the
symmetric DPR is `min(p_named, p_other) / max(p_named, p_other)`. Summaries take
the equal-weight mean across the exact 200 questions. Under these demographic
membership targets, these metrics summarize demographic recognition/location
decodability; they do not establish action-choice preference.

A completed baseline may contain nonconverged fits and proceed to strict
refinement. Final results require full feature/fit coverage, an empty
nonconvergence list, and passed independent metric and cache-backed optimizer
audits bound to the same result and frozen selection. Each feature retains
coefficients, scaler parameters, held-out predictions and per-question metrics.
`summary.csv` and `per_question.csv.gz` cover every layer, every decode/flow step
and the action feature; verified figures show absolute SPD and DPR by layer.

The server work directory is `/workspace/moskalenko/VLA_LINPROBING/q200`, with
this version deployed under `code/`. Separate `cache`, `results_base`,
`results_strict`, `results_repaired` and `exports` subdirectories contain a
directory for each model. The three selected caches require about 279 GiB.
Exports include `per_layer.csv` (an exact copy of `summary.csv`),
`per_question.csv.gz`, the run/split records, both audits, acceptance, selection
and plots. Full feature coefficient/scaler/prediction NPZ files remain in the
accepted result directories on the server.

After deployment validation passes, launch from the work directory with
`python code/pipeline.py --config code/server_config.json --detach`.
`status.py` reads progress without starting work. Every stage has a dedicated
process group monitored every five seconds. Admission requires 128 GiB effective
RAM headroom and 100 GiB free disk; a stage stops below 64 GiB RAM headroom or
50 GiB free disk, above 64 GiB group RSS, or when inspection fails. Source,
configuration and selection hashes are checked again before each stage.
