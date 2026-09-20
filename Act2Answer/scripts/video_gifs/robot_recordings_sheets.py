"""Запускать из ~/ws/lerobot_recordings (рядом recordings/ и index.csv: id, task, dur, end, status из metadata.json).
Контактные листы записей реального робота (MolmoAct2, SO-101): строка на сессию — общий вид cam1 в начале, середине
и конце + кадр с запястья cam0 в конце; подпись: номер, id, инструкция, длительность, причина остановки."""
import csv, os, subprocess, tempfile
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

FF = imageio_ffmpeg.get_ffmpeg_exe()
rows = list(csv.DictReader(open("index.csv")))
F = ImageFont.truetype("DejaVuSans.ttf", 15)
PER = 22
os.makedirs("sheets", exist_ok=True)
tmp = tempfile.mkdtemp()


def grab(path, t, from_end=False):
    out = os.path.join(tmp, "f.png")
    if os.path.exists(out):
        os.remove(out)
    pos = ["-sseof", f"-{t}"] if from_end else ["-ss", f"{t}"]
    subprocess.run([FF, "-nostdin", "-y", "-v", "error", *pos, "-i", path, "-frames:v", "1", out], check=False)
    return Image.open(out).convert("RGB").resize((240, 180)) if os.path.exists(out) else None


for s in range(0, len(rows), PER):
    part = rows[s:s + PER]
    sheet = Image.new("RGB", (4 * 240 + 520, len(part) * 184), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    for k, r in enumerate(part):
        y = k * 184
        base = os.path.join("recordings", r["id"])
        dur = float(r["dur"])
        shots = [grab(f"{base}/cam1.mp4", 0.2), grab(f"{base}/cam1.mp4", dur / 2), grab(f"{base}/cam1.mp4", 0.4, True), grab(f"{base}/cam0.mp4", 0.4, True)]
        for j, im in enumerate(shots):
            if im is not None:
                sheet.paste(im, (j * 240, y + 2))
        d.text((970, y + 10), f"#{s + k}  {r['id'][:16]}…{r['id'][-12:]}", fill=(255, 255, 0), font=F)
        d.text((970, y + 40), r["task"][:60], fill=(255, 255, 255), font=F)
        d.text((970, y + 70), f"{r['dur']} s · end={r['end']} · {r['status']}", fill=(170, 170, 170), font=F)
    sheet.save(f"sheets/sheet_{s // PER:02d}.jpg", quality=80)
    print("sheet", s // PER, len(part), flush=True)
