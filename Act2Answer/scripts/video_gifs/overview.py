"""Сводная картинка: последние кадры всех роликов из manifest.json сеткой 2 × N (для README и быстрого просмотра).

  python overview.py --dir ~/ws/video_gifs
"""
import argparse, json, os
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.expanduser("~/ws/video_gifs"))
a = ap.parse_args()
items = [m for m in json.load(open(os.path.join(a.dir, "manifest.json"))) if m["variant"] == "full"]
w, h, cols = 960, 540, 2
rows = (len(items) + cols - 1) // cols
sheet = Image.new("RGB", (cols * w, rows * h), (243, 246, 248))
for k, m in enumerate(items):
    im = Image.open(os.path.join(a.dir, "poster", f"{m['name']}.png")).convert("RGB").resize((w, h), Image.LANCZOS)
    sheet.paste(im, ((k % cols) * w, (k // cols) * h))
sheet.save(os.path.join(a.dir, "overview.jpg"), quality=88)
print(os.path.join(a.dir, "overview.jpg"), len(items))
