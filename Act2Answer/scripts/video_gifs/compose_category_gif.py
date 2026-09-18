"""Ролик-гифка одной категории вопросов FairACT (C1–C7) для видео к статье.

Две синхронные панели — один вопрос и одна пара картинок в исходном и в зеркальном порядке (как
в протоколе статьи). Сверху — номер и название категории в цветах статьи (Fig. 1/2) и инструкция
ровно в том виде, в каком её получила модель, с выделенным атрибутом. Когда модель отпускает куб,
выбранная плитка обводится цветом категории: ответ считается по положению куба в момент
отпускания, как в статье. Внизу — модель, источник картинок и итог «одна и та же картинка в обоих
порядках».

Вход: перепрогон демо-кардсетов (Act2Answer/scripts/record_gif_demos.sh) и outcomes.csv (outcomes.py).
Выход в --out: gif/ (1280×720), gif_light/ (960×540), mp4/ (1920×1080, 24 fps), poster/ (последний кадр).
Вариант --variant clean — только панели и подсветка, без шапки и подвала (под свои титры в монтаже).

  python compose_category_gif.py --select selection.json --out ~/ws/video_gifs
  python compose_category_gif.py --key C4_sahp_magma --run magma --idx 33 --out /tmp/try
Шрифт Inter (OFL) ищется в $FONT_DIR (по умолчанию ~/ws/fonts/Inter), иначе DejaVu Sans.
"""
import argparse, glob, json, os, shutil, subprocess, tempfile
import numpy as np, pandas as pd, imageio.v3 as iio, imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.environ.get("FONT_DIR", os.path.expanduser("~/ws/fonts/Inter"))
CATS = {
    "C1": ("Demographics and Appearance", "#287da8"),
    "C2": ("Education and Socioeconomic Status", "#259fad"),
    "C3": ("Occupations, Employment and Commercial Roles", "#3e9676"),
    "C4": ("Family, Culture and Civic Context", "#d69c35"),
    "C5": ("Health, Disability and Substance Use", "#cc735b"),
    "C6": ("Personality and Interests", "#8971b3"),
    "C7": ("Trust and Legal Status", "#b65f8f"),
}
MODELS = {"magma": ("Magma-8B", "#1f5c86"), "internvla": ("InternVLA-M1", "#c9822a"),
          "xiaomi": ("Xiaomi-Robotics-0", "#2e7d5b"), "gr00t": ("GR00T-N1.7", "#44525e"),
          "spatialvla": ("SpatialVLA-4B", "#a8497c")}
DATASET = {"pairs": "PAIRS", "focus": "FOCUS", "visbias": "VisBias"}
HL = {"x3_muscular": "be muscular", "pairs_poor": "poor person", "pairs_wealthy": "wealthy person",
      "pairs_flight_attendant": "flight attendant", "pairs_pilot": "pilot", "pairs_janitor": "janitor",
      "pairs_stay_at_home_parent": "stay-at-home parent",
      "x3_intellectual_disability": "have an intellectual disability", "x3_sports": "be interested in sports",
      "visbias_form_criminal_record_no": "not to have a criminal record",
      "visbias_form_criminal_record_yes": "have a criminal record"}
NAVY, GRAY, TEAL, BG = "#172b3a", "#52616d", "#167d8d", "#f3f6f8"

W, H = 1920, 1080
CROP = (16, 0, 640, 442)                 # область кадра камеры 640×480, где идёт всё действие
PW = 868
S = PW / (CROP[2] - CROP[0])
PH = int(round((CROP[3] - CROP[1]) * S))
GAP = 64
PX = [(W - 2 * PW - GAP) // 2, (W - 2 * PW - GAP) // 2 + PW + GAP]
PY = 318
RAD = 22


def hexrgb(h, a=255):
    h = h.lstrip("#"); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (a,)


_FONTS = {}


def font(w, size):
    k = (w, size)
    if k not in _FONTS:
        p = os.path.join(FONT_DIR, f"Inter-{w}.otf")
        if not os.path.exists(p):
            p = "DejaVuSans-Bold.ttf" if w in ("Bold", "SemiBold", "ExtraBold") else "DejaVuSans.ttf"
        _FONTS[k] = ImageFont.truetype(p, size)
    return _FONTS[k]


def rrect_mask(w, h, r, ss=4):
    m = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), r * ss, fill=255)
    return m.resize((w, h), Image.LANCZOS)


def rich_width(line, hl, size):
    i = line.find(hl) if hl else -1
    fr, fb = font("Medium", size), font("Bold", size)
    if i < 0:
        return fr.getlength(line)
    return fr.getlength(line[:i]) + fb.getlength(hl) + fr.getlength(line[i + len(hl):])


def layout_instruction(q, hl, max_w):
    """Кегль (и при необходимости перенос на две строки), чтобы инструкция влезла в max_w."""
    qq = f"“{q}”"
    for size in (50, 46, 42, 38, 36, 34):
        if rich_width(qq, hl, size) <= max_w:
            return size, [qq]
    size = 40
    words = qq.split(" ")
    for k in range(len(words) // 2, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        if rich_width(a, hl, size) <= max_w and rich_width(b, hl, size) <= max_w:
            return size, [a, b]
    return 34, [qq]


def draw_rich(d, xy, line, hl, size, color_main, color_hl):
    x, y = xy
    fr, fb = font("Medium", size), font("Bold", size)
    i = line.find(hl) if hl else -1
    parts = [(line, fr, color_main)] if i < 0 else [(line[:i], fr, color_main), (hl, fb, color_hl), (line[i + len(hl):], fr, color_main)]
    for s, f, c in parts:
        if s:
            d.text((x, y), s, font=f, fill=c)
            x += f.getlength(s)


def base_canvas(cat, question, qid, model, dataset_label, variant="full"):
    cname, ccol = CATS[cat]
    img = Image.new("RGBA", (W, H), hexrgb(BG))
    d = ImageDraw.Draw(img)
    if variant == "full":
        f_chip = font("Bold", 40)
        cw = int(f_chip.getlength(cat)) + 44; ch = 66
        x0, y0 = PX[0], 52
        d.rounded_rectangle((x0, y0, x0 + cw, y0 + ch), 16, fill=hexrgb(ccol))
        d.text((x0 + cw / 2, y0 + ch / 2 + 1), cat, font=f_chip, fill="white", anchor="mm")
        d.text((x0 + cw + 24, y0 + ch / 2 + 1), cname, font=font("SemiBold", 40), fill=hexrgb(ccol), anchor="lm")
        d.text((PX[1] + PW, y0 + ch / 2 + 1), "FairACT", font=font("Bold", 34), fill=hexrgb(NAVY, 200), anchor="rm")
        d.text((x0, 150), "INSTRUCTION TO THE ROBOT", font=font("Bold", 21), fill=hexrgb(TEAL))
        size, lines = layout_instruction(question, HL.get(qid, ""), PX[1] + PW - x0)
        y = 184 if len(lines) == 1 else 178
        for ln in lines:
            draw_rich(d, (x0, y), ln, HL.get(qid, ""), size, hexrgb(NAVY), hexrgb(ccol))
            y += int(size * 1.22)
    for k, lab in enumerate(("Original order", "Swapped order")):
        d.text((PX[k], PY - 14), lab, font=font("SemiBold", 27), fill=hexrgb(GRAY), anchor="ls")
    d.text((PX[1] + PW, PY - 14), "left ↔ right", font=font("Medium", 23), fill=hexrgb(GRAY, 170), anchor="rs")
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0)); sd = ImageDraw.Draw(sh)
    for px in PX:
        sd.rounded_rectangle((px, PY + 8, px + PW, PY + PH + 8), RAD, fill=(23, 43, 58, 60))
    img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(14)))
    d = ImageDraw.Draw(img)
    if variant == "full":
        mname, mcol = MODELS[model]
        fy = PY + PH + 58
        d.text((PX[0], fy), "Model", font=font("Medium", 24), fill=hexrgb(GRAY), anchor="lm")
        mx = PX[0] + font("Medium", 24).getlength("Model") + 16
        mw = int(font("Bold", 30).getlength(mname)) + 56
        d.rounded_rectangle((mx, fy - 25, mx + mw, fy + 25), 25, fill=hexrgb("#ffffff"), outline=hexrgb(mcol), width=3)
        d.ellipse((mx + 16, fy - 7, mx + 30, fy + 7), fill=hexrgb(mcol))
        d.text((mx + 40, fy + 1), mname, font=font("Bold", 30), fill=hexrgb(mcol), anchor="lm")
        d.text((PX[1] + PW, fy), f"Images: {dataset_label}", font=font("Medium", 24), fill=hexrgb(GRAY), anchor="rm")
    return img


def quad_to_panel(q, k):
    return [((x - CROP[0]) * S + PX[k], (y - CROP[1]) * S + PY) for x, y in np.asarray(q, float)]


_HL = {}


def highlight_layer(quad, color, t):
    """Обводка выбранной плитки; t ∈ (0,1] — фаза появления. Фаз всего несколько — слои кэшируются."""
    key = (tuple(map(tuple, np.round(quad, 2))), color, round(float(t), 3))
    if key in _HL:
        return _HL[key]
    ss = 2
    q = [(x * ss, y * ss) for x, y in quad]
    col = hexrgb(color)
    lay = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    glow = Image.new("RGBA", lay.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).line(q + [q[0]], fill=col[:3] + (int(200 * t),), width=26 * ss, joint="curve")
    lay = Image.alpha_composite(lay, glow.filter(ImageFilter.GaussianBlur(10 * ss)))
    fill = Image.new("RGBA", lay.size, (0, 0, 0, 0))
    ImageDraw.Draw(fill).polygon(q, fill=col[:3] + (int(34 * t),))
    lay = Image.alpha_composite(lay, fill)
    ImageDraw.Draw(lay).line(q + [q[0]], fill=col[:3] + (int(255 * t),), width=int((7 + 8 * (1 - t)) * ss), joint="curve")
    lay = lay.resize((W, H), Image.LANCZOS)
    if t > 0.5:
        d = ImageDraw.Draw(lay)
        (x1, y1), (x2, y2) = sorted(quad, key=lambda p: p[1])[:2]       # дальняя кромка плитки
        cx, top_y = (x1 + x2) / 2, min(y1, y2)
        f = font("Bold", 26); txt = "✓  selected"
        tw = f.getlength(txt) + 34
        k = 0 if cx < PX[1] else 1
        x0 = min(max(cx - tw / 2, PX[k] + 12), PX[k] + PW - tw - 12)
        y0 = max(top_y - 60, PY + 12)
        a = int(255 * min(1, (t - 0.5) * 2))
        d.rounded_rectangle((x0, y0, x0 + tw, y0 + 44), 22, fill=col[:3] + (a,))
        d.text((x0 + tw / 2, y0 + 22), txt, font=f, fill=(255, 255, 255, a), anchor="mm")
    _HL[key] = lay
    return lay


def load_tables():
    oc = pd.read_csv(os.path.join(HERE, "outcomes.csv"))
    cand = {}
    for p in sorted(glob.glob(os.path.join(HERE, "candidates_w*.json"))):
            for r in json.load(open(p, encoding="utf-8")):
                cand.setdefault(r["key"], r)
    questions = json.load(open(os.path.join(HERE, "questions.json"), encoding="utf-8"))
    geo = json.load(open(os.path.join(HERE, "tile_geom.json")))
    return oc, cand, questions, geo


def compose(key, run, idx, outdir, variant="full", play_fps=8, intro_s=1.4, outro_s=2.4, tail=10, name=None):
    oc, cand, questions, geo = load_tables()
    e = {o: oc[(oc.key == key) & (oc.run == run) & (oc.idx == idx) & (oc.order == o)].iloc[0] for o in ("noswap", "swap")}
    r0 = e["noswap"]
    cat = key.split("_")[0]; ccol = CATS[cat][1]
    base = base_canvas(cat, questions[key], cand[key]["qid"], r0.vla, DATASET[r0.cs.split("_")[0]], variant)
    vids = {o: iio.imread(e[o].video) for o in e}
    rel = {o: int(e[o].rel_frame) for o in e}
    side = {o: int(e[o].side_rel) for o in e}
    same = e["noswap"].pick_img == e["swap"].pick_img and e["noswap"].pick_img > 0
    end = min(80, max(rel.values()) + tail)
    mask = rrect_mask(PW, PH, RAD)
    tmp = tempfile.mkdtemp(prefix="gif_")
    n = 0

    def emit(img):
        nonlocal n
        img.convert("RGB").save(os.path.join(tmp, f"f_{n:04d}.png"), compress_level=1)
        n += 1

    def frame(i):
        img = base.copy()
        for k, o in enumerate(("noswap", "swap")):
            fr = vids[o][min(i, len(vids[o]) - 1)]
            img.paste(Image.fromarray(fr).crop(CROP).resize((PW, PH), Image.LANCZOS), (PX[k], PY), mask)
        for k, o in enumerate(("noswap", "swap")):
            s = rel[o] + 2                      # куб отпущен и лёг
            t = 0.0 if i < s else min(1.0, (i - s + 1) / 4)
            if t > 0 and side[o] in (1, 2):
                img = Image.alpha_composite(img, highlight_layer(quad_to_panel(geo["quadL" if side[o] == 1 else "quadR"], k), ccol, t))
        return img

    first = frame(0)
    for _ in range(int(round(intro_s * play_fps))):
        emit(first)
    last = first
    for i in range(1, end + 1):
        last = frame(i); emit(last)
    for j in range(int(round(outro_s * play_fps))):
        img = last.copy()
        if variant == "full" and same:
            a = min(1, (j + 1) / 4)
            ImageDraw.Draw(img).text(((PX[0] + PX[1] + PW) / 2 + 120, PY + PH + 58), "✓  Same picture selected in both orders",
                                     font=font("Bold", 31), fill=hexrgb(ccol, int(255 * a)), anchor="mm")
        emit(img)
    name = name or key
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    src = ["-framerate", str(play_fps), "-i", os.path.join(tmp, "f_%04d.png")]
    for sub in ("mp4", "gif", "gif_light", "poster"):
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)
    subprocess.run([ff, "-nostdin", "-y", "-v", "error", *src, "-vf", "fps=24,format=yuv420p", "-c:v", "libx264", "-crf", "14",
                    "-preset", "slow", "-movflags", "+faststart", os.path.join(outdir, "mp4", f"{name}.mp4")], check=True)
    for sub, width, colors, dither in (("gif", 1280, 256, "sierra2_4a"), ("gif_light", 960, 192, "bayer:bayer_scale=4")):
        vf = (f"scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors={colors}:stats_mode=full[p];"
              f"[b][p]paletteuse=dither={dither}:diff_mode=rectangle")
        subprocess.run([ff, "-nostdin", "-y", "-v", "error", *src, "-filter_complex", vf, "-loop", "0",
                        os.path.join(outdir, sub, f"{name}.gif")], check=True)
    shutil.copy(os.path.join(tmp, f"f_{n - 1:04d}.png"), os.path.join(outdir, "poster", f"{name}.png"))
    shutil.rmtree(tmp, ignore_errors=True)
    return dict(name=name, key=key, run=run, idx=idx, model=r0.vla, cs=r0.cs, src_index=int(r0.src_i),
                img1=r0.img1, img2=r0.img2, target=r0.target, frames=n, seconds=round(n / play_fps, 2),
                same_picture=bool(same), release_frame=rel, side_at_release=side,
                strict_end={o: int(e[o].strict) for o in e})


def _run_task(arg):
    """((job, variant), out, fps) -> запись манифеста; отдельная функция — чтобы работал пул процессов."""
    (j, variant), out_root, fps = arg
    out = out_root if variant == "full" else os.path.join(out_root, "clean")
    r = compose(j["key"], j["run"], j["idx"], out, variant=variant, play_fps=fps, name=j.get("name"))
    r.update(variant=variant, note=j.get("note", ""))
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", help="JSON-список {name,key,run,idx} — собрать все")
    ap.add_argument("--key"); ap.add_argument("--run"); ap.add_argument("--idx", type=int)
    ap.add_argument("--name"); ap.add_argument("--variant", default="full", choices=("full", "clean"))
    ap.add_argument("--out", default=os.path.expanduser("~/ws/video_gifs"))
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--procs", type=int, default=8, help="параллельных процессов в режиме --select")
    a = ap.parse_args()
    jobs = json.load(open(a.select)) if a.select else [dict(name=a.name or a.key, key=a.key, run=a.run, idx=a.idx)]
    tasks = [((j, v), a.out, a.fps) for j in jobs for v in ((a.variant,) if not a.select else ("full", "clean"))]
    if a.select and a.procs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.procs) as ex:
            log = list(ex.map(_run_task, tasks))
    else:
        log = [_run_task(t) for t in tasks]
    for r in log:
        print(json.dumps(r, ensure_ascii=False))
    if a.select:
        json.dump(log, open(os.path.join(a.out, "manifest.json"), "w"), ensure_ascii=False, indent=1)
