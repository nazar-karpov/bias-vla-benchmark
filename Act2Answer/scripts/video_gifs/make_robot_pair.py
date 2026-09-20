"""Парный клип «прямая / зеркальная раскладка» из записей реального робота (MolmoAct2, SO-101; recordings.zip Андрея,
на Bohr ~/ws/lerobot_recordings/recordings/<id>/cam1.mp4 — общий вид, cam0 — запястье) для слайда презентации.

Кадр поворачивается на 180° (камера смотрит «вверх ногами»), эпизод обрезается по времени укладки куба и ускоряется так,
чтобы обе панели шли одинаково долго; в конце — стоп-кадр. Стиль подписей — как у пар из симулятора.

  python make_robot_pair.py --direct <id>:<t_end> --swap <id>:<t_end> --name pilot --out ~/ws/video_gifs/deck
"""
import argparse, os, shutil, subprocess, tempfile
import imageio.v3 as iio, imageio_ffmpeg
from PIL import Image, ImageDraw
import compose_category_gif as C

REC = os.path.expanduser("~/ws/lerobot_recordings/recordings")
BG = "#f3f6f8"
PW, PH, GAP, TOP = 800, 600, 64, 56
W, H = 2 * PW + GAP, TOP + PH + 8


def pick_frames(sid, t_end, n, fps_src=29.4):
    last = int(t_end * fps_src)
    want = sorted({round(k * last / (n - 1)) for k in range(n)})
    got = {}
    for i, fr in enumerate(iio.imiter(os.path.join(REC, sid, "cam1.mp4"))):
        if i in want:
            got[i] = Image.fromarray(fr).rotate(180).resize((PW, PH), Image.LANCZOS)
        if i >= last:
            break
    seq = [got[i] for i in want if i in got]
    return seq + [seq[-1]] * (n - len(seq))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", required=True); ap.add_argument("--swap", required=True)
    ap.add_argument("--name", required=True); ap.add_argument("--out", default=os.path.expanduser("~/ws/video_gifs/deck"))
    ap.add_argument("--fps", type=float, default=8); ap.add_argument("--seconds", type=float, default=7.0)
    a = ap.parse_args()
    n = int(a.seconds * a.fps)
    panels = []
    for spec, label in ((a.direct, "Direct layout"), (a.swap, "Swapped layout")):
        sid, t_end = spec.split(":"); t_end = float(t_end)
        panels.append((label, f"×{t_end / a.seconds:.0f} speed", pick_frames(sid, t_end, n)))
    mask = C.rrect_mask(PW, PH, 20)
    tmp = tempfile.mkdtemp(prefix="robot_"); k = 0
    order = [0] * int(1.0 * a.fps) + list(range(n)) + [n - 1] * int(2.5 * a.fps)
    for i in order:
        img = Image.new("RGB", (W, H), C.hexrgb(BG)[:3]); d = ImageDraw.Draw(img)
        for j, (label, speed, seq) in enumerate(panels):
            x = j * (PW + GAP)
            d.text((x, TOP - 14), label, font=C.font("SemiBold", 27), fill=C.hexrgb(C.GRAY)[:3], anchor="ls")
            d.text((x + PW, TOP - 14), speed, font=C.font("Medium", 23), fill=C.hexrgb(C.GRAY)[:3], anchor="rs")
            img.paste(seq[i], (x, TOP), mask)
        img.save(os.path.join(tmp, f"f_{k:04d}.png"), compress_level=1); k += 1
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    src = ["-framerate", str(a.fps), "-i", os.path.join(tmp, "f_%04d.png")]
    os.makedirs(a.out, exist_ok=True)
    mp4 = os.path.join(a.out, f"robot_pair_{a.name}.mp4")
    subprocess.run([ff, "-nostdin", "-y", "-v", "error", *src, "-vf", "fps=24,format=yuv420p", "-c:v", "libx264", "-crf", "17",
                    "-preset", "slow", "-movflags", "+faststart", mp4], check=True)
    vf = ("hqdn3d=4:4:8:8,scale=1280:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    gif = os.path.join(a.out, f"robot_pair_{a.name}.gif")
    subprocess.run([ff, "-nostdin", "-y", "-v", "error", *src, "-filter_complex", vf, "-loop", "0", gif], check=True)
    shutil.copy(os.path.join(tmp, f"f_{k - 1:04d}.png"), os.path.join(a.out, f"robot_pair_{a.name}_still.png"))
    shutil.rmtree(tmp, ignore_errors=True)
    for f in (mp4, gif):
        print(os.path.basename(f), round(os.path.getsize(f) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
