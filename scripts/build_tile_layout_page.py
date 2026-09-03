#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Страница-сравнение расстановок плиток (артефакт) из выходов tile_layout_probe.py.

Читает tile_layout_preview/probe_*.json + PNG-кадры, вшивает кадры как JPEG data-URI
и пишет один HTML. Запуск: python scripts/build_tile_layout_page.py --src tile_layout_preview
--out <путь>.html
"""
import argparse
import base64
import html
import io
import json
from pathlib import Path

from PIL import Image


def jpeg_uri(path: Path, q: int = 82) -> str:
    im = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def status(margin_r: float, margin_l: float, gap_cm: float):
    m = min(margin_r, margin_l)
    if m < 0:
        return "cut", "режется"
    if gap_cm < 4:
        return "tight", "просвет меньше куба"
    if m < 15:
        return "tight", "впритык"
    return "ok", "влезает"


def recipe(row):
    parts = []
    if abs(row["scale"] - 1.3) > 1e-6:
        parts.append(f"BOARD_XY_SCALE={row['scale'] / 1.3:.4f}")
    else:
        parts.append("BOARD_XY_SCALE=1.0")
    if abs(row["y_half"] - 0.155) > 1e-9:
        parts.append(f"A2A_TILE_Y={row['y_half']:g}")
    if abs(row["yc"]) > 1e-9:
        parts.append(f"A2A_TILE_YC={row['yc']:g}")
    if abs(row["x"] + 0.25) > 1e-9:
        parts.append(f"A2A_TILE_X={row['x']:g}")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--recommend", default="s1p30_y0p135_yc-0p020_x-0p25")
    args = ap.parse_args()

    rows = []
    for j in sorted(args.src.glob("probe_*.json")):
        rows += json.loads(j.read_text(encoding="utf-8"))
    rows = [r for r in rows if not r.get("swap")]
    rows.sort(key=lambda r: (-r["scale"], -r["y_half"], -r["yc"]))

    cards = []
    for r in rows:
        tag = r["tag"]
        raw = args.src / f"{tag}_ep{r['ep']}.png"
        ov = args.src / f"{tag}_ep{r['ep']}_overlay.png"
        if not raw.exists():
            continue
        st, st_label = status(r["margin_R"], r["margin_L"], r["gap_cm"])
        is_cur = abs(r["scale"] - 1.3) < 1e-6 and abs(r["y_half"] - 0.155) < 1e-9 and abs(r["yc"]) < 1e-9
        is_base = abs(r["scale"] - 1.0) < 1e-6 and abs(r["y_half"] - 0.155) < 1e-9
        note = ""
        if is_cur:
            note = "текущий боевой сетап (все confirm-прогоны)"
        elif is_base:
            note = "исходный Act2Answer (масштаб 1.0)"
        rec = tag == args.recommend
        ratio = r["seg_px_R"] / max(r["seg_px_L"], 1)
        title = f"масштаб {r['scale']:.2f} · слоты ±{r['y_half']:.3f}"
        if abs(r["yc"]) > 1e-9:
            title += f" · сдвиг пары {r['yc']:+.2f} м"
        cards.append(f"""
<article class="card st-{st}{' rec' if rec else ''}{' cur' if is_cur else ''}" id="{html.escape(tag)}">
  <header class="card-head">
    <div class="card-title">
      <h3>{html.escape(title)}</h3>
      {f'<p class="note">{html.escape(note)}</p>' if note else ''}
      {'<p class="note rec-note">рекомендуемый вариант</p>' if rec else ''}
    </div>
    <span class="pill pill-{st}">{st_label}</span>
  </header>
  <figure class="frame">
    <img class="img-raw" src="{jpeg_uri(raw)}" alt="первый кадр obs-камеры, {html.escape(title)}" width="640" height="480">
    <img class="img-ov" src="{jpeg_uri(ov)}" alt="то же с контурами плиток" width="640" height="480" hidden>
  </figure>
  <dl class="metrics">
    <div><dt>запас правой до края</dt><dd class="{ 'neg' if r['margin_R'] < 0 else ''}">{r['margin_R']:+.0f} px</dd></div>
    <div><dt>запас левой</dt><dd>{r['margin_L']:+.0f} px</dd></div>
    <div><dt>в кадре, правая</dt><dd>{min(r['clip_R'], 1.0) * 100:.1f}%</dd></div>
    <div><dt>сторона плитки</dt><dd>{r['tile_side_cm']:.1f} см</dd></div>
    <div><dt>просвет между плитками</dt><dd class="{ 'neg' if r['gap_cm'] < 4 else ''}">{r['gap_cm']:.1f} см</dd></div>
    <div><dt>пикселей R / L</dt><dd>{r['seg_px_R']:,} / {r['seg_px_L']:,} <span class="dim">(×{ratio:.2f})</span></dd></div>
  </dl>
  <pre class="recipe"><code>{html.escape(recipe(r))}</code></pre>
</article>""")

    n_rows = len(cards)
    page = f"""<title>Раскладка плиток</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --bg: #f2f3ef; --surface: #fbfbf9; --ink: #1c211d; --muted: #5f675f; --rule: #d3d8d0;
  --accent: #1f5f8b; --ok: #2c7a47; --ok-bg: #e2f1e6; --tight: #9a6a08; --tight-bg: #f7ecd0;
  --cut: #b8371a; --cut-bg: #f8e1da; --code-bg: #e9ebe5; --rec: #1f5f8b;
  --shadow: 0 1px 2px rgba(20,30,20,.08), 0 6px 20px rgba(20,30,20,.06);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #15181a; --surface: #1d2124; --ink: #e6e8e3; --muted: #9aa39b; --rule: #30363a;
    --accent: #7fb3d8; --ok: #7fd09a; --ok-bg: #1d3325; --tight: #e2b95c; --tight-bg: #3a2f12;
    --cut: #f08a6e; --cut-bg: #44211a; --code-bg: #24292d; --rec: #7fb3d8;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 6px 20px rgba(0,0,0,.35);
  }}
}}
:root[data-theme="dark"] {{
  --bg: #15181a; --surface: #1d2124; --ink: #e6e8e3; --muted: #9aa39b; --rule: #30363a;
  --accent: #7fb3d8; --ok: #7fd09a; --ok-bg: #1d3325; --tight: #e2b95c; --tight-bg: #3a2f12;
  --cut: #f08a6e; --cut-bg: #44211a; --code-bg: #24292d; --rec: #7fb3d8;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 6px 20px rgba(0,0,0,.35);
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 "IBM Plex Sans", system-ui, sans-serif; }}
main {{ max-width: 1240px; margin: 0 auto; padding: 32px 24px 64px; }}
h1, h2, h3 {{ font-family: "IBM Plex Sans Condensed", "IBM Plex Sans", system-ui, sans-serif; font-weight: 600; text-wrap: balance; margin: 0; }}
h1 {{ font-size: 34px; line-height: 1.15; }}
h2 {{ font-size: 22px; margin: 0 0 12px; }}
h3 {{ font-size: 18px; line-height: 1.25; }}
p {{ margin: 0; }}
.lede {{ max-width: 68ch; margin-top: 10px; color: var(--muted); }}
.lede strong {{ color: var(--ink); font-weight: 500; }}
.eyebrow {{ font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); font-weight: 500; }}
section {{ margin-top: 40px; }}
.verdict {{ display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 14px; margin-top: 18px; }}
.verdict > div {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 16px; }}
.verdict .big {{ font-family: "IBM Plex Sans Condensed", sans-serif; font-size: 26px; font-weight: 600; font-variant-numeric: tabular-nums; }}
.verdict .lab {{ color: var(--muted); font-size: 13px; }}
.controls {{ display: flex; gap: 18px; align-items: center; flex-wrap: wrap; margin: 14px 0 18px; color: var(--muted); font-size: 14px; }}
.controls label {{ display: inline-flex; gap: 8px; align-items: center; cursor: pointer; color: var(--ink); }}
.legend {{ display: inline-flex; gap: 10px; align-items: center; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 20px; }}
.card {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; display: flex; flex-direction: column; }}
.card.rec {{ border-color: var(--rec); box-shadow: var(--shadow); }}
.card-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; padding: 14px 16px 10px; }}
.note {{ color: var(--muted); font-size: 13px; margin-top: 2px; }}
.rec-note {{ color: var(--rec); font-weight: 500; }}
.pill {{ flex: none; font-size: 12px; font-weight: 500; letter-spacing: .04em; padding: 3px 10px; border-radius: 999px; white-space: nowrap; }}
.pill-ok {{ color: var(--ok); background: var(--ok-bg); }}
.pill-tight {{ color: var(--tight); background: var(--tight-bg); }}
.pill-cut {{ color: var(--cut); background: var(--cut-bg); }}
.frame {{ margin: 0; position: relative; background: #000; aspect-ratio: 4 / 3; }}
.frame img {{ display: block; width: 100%; height: auto; }}
.frame::after {{ content: ""; position: absolute; inset: 0; pointer-events: none; border: 2px dashed transparent; }}
.st-cut .frame::after {{ border-right-color: var(--cut); }}
.metrics {{ display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 10px 14px; margin: 0; padding: 12px 16px 6px; }}
.metrics dt {{ font-size: 12px; color: var(--muted); }}
.metrics dd {{ margin: 0; font-variant-numeric: tabular-nums; font-weight: 500; }}
.metrics dd.neg {{ color: var(--cut); }}
.dim {{ color: var(--muted); font-weight: 400; }}
.recipe {{ margin: 8px 16px 16px; padding: 8px 10px; background: var(--code-bg); border-radius: 4px; font: 13px/1.5 "IBM Plex Mono", ui-monospace, monospace; overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
.tbl {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; }}
th, td {{ padding: 8px 12px; text-align: right; border-bottom: 1px solid var(--rule); white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
thead th {{ font-size: 12px; letter-spacing: .05em; text-transform: uppercase; color: var(--muted); font-weight: 500; }}
tbody tr:last-child td {{ border-bottom: 0; }}
td.hi {{ color: var(--accent); font-weight: 500; }}
.two {{ display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr); gap: 24px; align-items: start; }}
.prose {{ max-width: 66ch; }}
.prose p + p {{ margin-top: 10px; }}
ul {{ margin: 10px 0 0; padding-left: 20px; }}
li + li {{ margin-top: 6px; }}
code {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: .92em; background: var(--code-bg); padding: 1px 5px; border-radius: 3px; }}
.recipe code {{ background: none; padding: 0; }}
@media (max-width: 900px) {{ .two, .verdict {{ grid-template-columns: 1fr; }} }}
@media (max-width: 520px) {{ .grid {{ grid-template-columns: 1fr; }} .metrics {{ grid-template-columns: 1fr 1fr; }} }}
input[type=checkbox]:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
</style>

<main>
  <p class="eyebrow">Bridge-сцена · obs-камера 640×480 · первый кадр эпизода 0 (pilot, pairs_choice_vla_confirm)</p>
  <h1>Раскладка плиток без подрезки краем кадра</h1>
  <p class="lede">Боевой сетап — масштаб 1.3 и слоты ±0.155 м — режет правую плитку правым краем.
  Ниже реальные кадры симулятора для {n_rows} вариантов: тот же масштаб со сближенными плитками,
  сдвиг пары влево и уменьшенные масштабы. Кадры отрисованы без GPU (physx_cpu + SwiftShader),
  геометрия та же, что в боевых прогонах.</p>

  <div class="verdict">
    <div><div class="lab">текущий сетап, правая плитка</div><div class="big" style="color:var(--cut)">−20 px</div><div class="lab">за правым краем кадра; в кадре 98.2% площади</div></div>
    <div><div class="lab">рекомендация</div><div class="big">±0.135 м, сдвиг −0.02</div><div class="lab">масштаб 1.3 сохранён, запас справа +28 px, просвет 8.1 см</div></div>
    <div><div class="lab">цена возврата к масштабу 1.0 (Magma)</div><div class="big">60% → 84%</div><div class="lab">чистый позиционный крен влево, neutral-кардсет, n=800</div></div>
  </div>

  <section>
    <h2>Варианты</h2>
    <div class="controls">
      <label><input type="checkbox" id="ov"> показать контуры плиток (проекция углов меша)</label>
      <span class="legend"><span class="pill pill-cut">режется</span><span class="pill pill-tight">впритык, &lt;15 px</span><span class="pill pill-ok">влезает</span></span>
    </div>
    <div class="grid">{''.join(cards)}
    </div>
  </section>

  <section class="two">
    <div>
      <h2>Прав ли ты про масштаб: да, для Magma</h2>
      <div class="prose">
        <p>Сетка масштабов 15.08 (metrics/scale_sweep_full.md). Точность чтения плитки у Magma ≈1.0 на любом масштабе, различаются
        <strong>позиционный крен</strong> и <strong>доезжаемость</strong>. На нейтральном кардсете (плитки без смысла, крен чистый)
        1.3 — единственный минимум; 1.0 и 1.15 отбрасывают к 84–91% влево. У InternVLA масштаб ничего не лечит (80–90% везде).</p>
        <p>Поэтому уменьшать плитку до 1.15, чтобы она влезла, невыгодно: подрезки ещё −5 px, а крен уже как на 1.0.
        Сближение плиток при масштабе 1.3 не трогает размер, который и даёт эффект.</p>
        <p>Оговорка: любая новая раскладка — новая базовая линия. Все confirm-прогоны шли на ±0.155; сравнивать величины
        напрямую нельзя, нужен свой нейтральный замер крена на выбранной раскладке (200–400 эп.).</p>
      </div>
    </div>
    <div class="tbl">
      <table>
        <thead><tr><th>масштаб</th><th>Magma neutral, left%</th><th>AR</th><th>Magma ceiling, left%</th><th>AR</th><th>InternVLA neutral, left%</th></tr></thead>
        <tbody>
          <tr><td>0.8</td><td>94.4</td><td>49%</td><td>87.2</td><td>35%</td><td>87.3</td></tr>
          <tr><td>1.0</td><td>83.9</td><td>41%</td><td>91.3</td><td>41%</td><td>80.1</td></tr>
          <tr><td>1.15</td><td>—</td><td>—</td><td>83.3</td><td>43%</td><td>—</td></tr>
          <tr><td>1.3</td><td class="hi">59.8</td><td>38%</td><td class="hi">63.6</td><td>49%</td><td>86.4</td></tr>
          <tr><td>1.5</td><td>69.7</td><td>39%</td><td>52.9</td><td>46%</td><td>90.0</td></tr>
          <tr><td>1.7</td><td>61.6</td><td>30%</td><td>50.0</td><td>50%</td><td>89.5</td></tr>
          <tr><td>1.9</td><td>37.4</td><td>24%</td><td>46.6</td><td>65%</td><td>81.6</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Как это применить</h2>
    <div class="prose">
      <p>В env добавлены три переменные с прежними значениями по умолчанию, старые прогоны не меняются:</p>
      <ul>
        <li><code>A2A_TILE_Y</code> — полурасстояние между центрами плиток (было зашито 0.155).</li>
        <li><code>A2A_TILE_YC</code> — сдвиг центра пары по y, отрицательный = обе плитки левее.</li>
        <li><code>A2A_TILE_X</code> — x обеих плиток (−0.25).</li>
      </ul>
      <p>Масштаб по-прежнему = 1.3 из model_db × <code>BOARD_XY_SCALE</code>. Строка под каждым кадром — готовый набор переменных
      для раннера. Просвет между плитками должен оставаться заметно больше куба (3 см): при ±0.105 он 2.1 см, и куб, брошенный
      между плитками, попадает в обе зоны сразу.</p>
      <p>Проба: <code>Act2Answer/scripts/tile_layout_probe.py</code>, раннер для CPU-ноды <code>run_tile_layout_cpu.sh</code>.
      Зоны зачёта делятся по знаку y, поэтому сдвиг пары допустим, пока обе плитки остаются по разные стороны от нуля.</p>
    </div>
  </section>
</main>
<script>
(function () {{
  var box = document.getElementById('ov');
  function apply() {{
    var on = box.checked;
    document.querySelectorAll('.img-raw').forEach(function (el) {{ el.hidden = on; }});
    document.querySelectorAll('.img-ov').forEach(function (el) {{ el.hidden = !on; }});
    try {{ localStorage.setItem('tile_layout_overlay', on ? '1' : '0'); }} catch (e) {{}}
  }}
  try {{ box.checked = localStorage.getItem('tile_layout_overlay') === '1'; }} catch (e) {{}}
  box.addEventListener('change', apply);
  apply();
}})();
</script>
"""
    args.out.write_text(page, encoding="utf-8")
    print(f"{args.out}: {len(cards)} карточек, {args.out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
