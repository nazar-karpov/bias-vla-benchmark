"""Фон обложки презентации: мозаика 6 × 4 из роликов симуляции (каждый порядок отдельным клипом, с обводкой
выбранной плитки, как raw_selected), клипы стартуют со сдвигом, чтобы не двигались синхронно; поверх на слайде
лежит полупрозрачная карточка с названием. Клипы берутся из raw_selected/mp4 (16 штук) и дособираются
compose_raw'ом для дополнительных пар из EXTRA.

  python make_cover_mosaic.py --out ~/ws/video_gifs/deck
Пишет cover_mosaic.mp4 (1920×1080), cover_mosaic.gif (облегчённая) и cover_mosaic_still.png.
"""
import argparse, glob, os, random, shutil, subprocess, tempfile
import imageio.v3 as iio, imageio_ffmpeg
from PIL import Image
import compose_category_gif as C

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRA = [("C1_muscular_internvla", "internvla", 2), ("C6_sports_internvla", "internvla", 26),
         ("C7_nocrim_internvla", "internvla", 45),
         ("C3_pilot_internvla", "internvla", 12)]
COLS, ROWS, TW, TH, GAP = 6, 4, 300, 225, 12
W, H, FPS, SEC = 1920, 1080, 8, 8
BG = "#f3f6f8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/ws/video_gifs/deck"))
    a = ap.parse_args()
    raw_dir = os.path.join(os.path.dirname(a.out.rstrip("/")), "raw_selected")
    extra_dir = os.path.join(a.out, "_mosaic_extra")
    if not os.path.isdir(extra_dir):
        for key, run, idx in EXTRA:
            C.compose_raw(key, run, idx, extra_dir, play_fps=FPS, name=f"{key}_i{idx}")
    clips = sorted(glob.glob(os.path.join(raw_dir, "mp4", "*.mp4"))) + sorted(glob.glob(os.path.join(extra_dir, "mp4", "*.mp4")))
    assert len(clips) >= COLS * ROWS, len(clips)
    rnd = random.Random(7)
    rnd.shuffle(clips)
    clips = clips[:COLS * ROWS]
    frames = []
    for p in clips:
        v = [Image.fromarray(f).resize((TW, TH), Image.LANCZOS) for f in iio.imiter(p)][::3]   # mp4 24 fps -> 8 fps
        frames.append((v, rnd.randrange(len(v))))
    mask = C.rrect_mask(TW, TH, 12)
    x0 = (W - (COLS * TW + (COLS - 1) * GAP)) // 2
    y0 = (H - (ROWS * TH + (ROWS - 1) * GAP)) // 2
    tmp = tempfile.mkdtemp(prefix="mosaic_")
    n = FPS * SEC
    for t in range(n):
        img = Image.new("RGB", (W, H), C.hexrgb(BG)[:3])
        for k, (v, off) in enumerate(frames):
            fr = v[(off + t) % len(v)]
            img.paste(fr, (x0 + (k % COLS) * (TW + GAP), y0 + (k // COLS) * (TH + GAP)), mask)
        img.save(os.path.join(tmp, f"f_{t:04d}.png"), compress_level=1)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    src = ["-framerate", str(FPS), "-i", os.path.join(tmp, "f_%04d.png")]
    subprocess.run([ff, "-nostdin", "-y", "-v", "error", *src, "-vf", "fps=24,format=yuv420p", "-c:v", "libx264", "-crf", "18",
                    "-preset", "slow", "-movflags", "+faststart", os.path.join(a.out, "cover_mosaic.mp4")], check=True)
    vf = ("fps=5,scale=1280:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160:stats_mode=full[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    subprocess.run([ff, "-nostdin", "-y", "-v", "error", *src, "-filter_complex", vf, "-loop", "0",
                    os.path.join(a.out, "cover_mosaic.gif")], check=True)
    shutil.copy(os.path.join(tmp, "f_0000.png"), os.path.join(a.out, "cover_mosaic_still.png"))
    shutil.rmtree(tmp, ignore_errors=True)
    for f in ("cover_mosaic.mp4", "cover_mosaic.gif", "cover_mosaic_still.png"):
        print(f, round(os.path.getsize(os.path.join(a.out, f)) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
