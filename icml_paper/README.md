# ICML LaTeX-отчёт — Bias Benchmark для VLA

LaTeX-версия статьи по шаблону ICML 2025 (русский текст, xelatex) плюс все исходные
данные, на основе которых она написана.

## Как собрать PDF

Русский текст → нужен **xelatex** (не pdflatex) + шрифт Times New Roman.

```bash
xelatex paper.tex
xelatex paper.tex   # второй проход — для ссылок на рисунки (\ref)
```

Требуются пакеты: `polyglossia`, `fontspec`, `microtype`, `booktabs`, `mathtools`,
`hyperref`, `forloop`, `cyrillic` (ставятся через `tlmgr install ...`). Стили ICML
(`icml2025.sty`, `.bst`, `fancyhdr`, `algorithm*`) лежат в этой же папке.

## Файлы

| файл | что это |
|---|---|
| `paper.tex` | **исходник статьи** (ICML-вёрстка, русский) |
| `paper.pdf` | собранный PDF (5 страниц) |
| `icml2025.sty` · `icml2025.bst` · `fancyhdr.sty` · `algorithm*.sty` | стили шаблона ICML 2025 |
| `example_paper.tex` · `example_paper.bib` | образец из шаблона (для справки) |
| `figs/` | 7 графиков статьи (Figure 1–7) |
| `sources/` | **данные-основа** — см. ниже |

## sources/ — на основе чего написан пейпер

| файл | содержит |
|---|---|
| `PAPER_source.md` | исходный markdown-текст статьи (полная версия до LaTeX) |
| `METRICS_ALL_MODELS.md` | демографические метрики: S по всем VLA × 4 канала, VLM-кросс |
| `CONTINUOUS_PULL_REPORT.md` | непрерывное притяжение (мм), по-вопросные тесты t+Wilcoxon |
| `SAFETY_SUMMARY.md` | **safety-часть целиком**: VLM-канал, выбор SOHAS, cross-pairs, баг метрики, таблицы S по 8 VLA и 8 VLM, 8 методических выводов |
| `metrics_csv/` | первичные CSV: `metrics_choice` (парный выбор), `metrics_yesno` (одна картинка), `metrics_attr_choice` (по каждому вопросу) |
| `safety_data/` | barplot (Рис. 7), 4 примера SOHAS (knife/pistol/monedero/smartphone), 3 видео VLA (internvla_success, openvla_false, xiaomi_no_answer) |

## Соответствие рисунков

| Figure | файл | раздел |
|---|---|---|
| 1 | `fig1_three_effects.png` | 3 надёжных эффекта × 4 канала |
| 2 | `fig4_pull_mm.png` | притяжение в мм |
| 3 | `fig6_posneg.png` | механика P(pos)/P(neg) |
| 4 | `fig3_vlm_gender.png` | VLM-кросс по гендеру |
| 5 | `fig5_vlm33.png` | топ по 33 вопросам |
| 6 | `fig2_decomposition.png` | проницаемость обвязки |
| 7 | `fig7_safety_bars.png` | safety: answer-rate vs success-rate |
