#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проба ЦЕНТРАЛЬНОГО слота single-card дизайна: куда и как повернуть карточку.

Зачем: в слотах ±0.155 позиционный крен даёт R−L ≈ −111 мм (metrics/
single_card_pilotA_all.txt) — на порядок больше любого демографического
эффекта, и половина эпизодов уходит на контрбаланс слотов. Карточка по центру
(y=0) убирает лево-правую асимметрию совсем. Проблема ровно одна: рука со
схваченным кубом висит над центром стола и закрывает ДАЛЬНЮЮ половину плитки —
а лицо на фото лежит именно там. Отсюда второй параметр — разворот карточки
(A2A_SINGLE_TILE_YAW), уводящий лицо из-под руки.

Меряем по кадру ИМЕННО той камеры, что идёт модели (3rd_view_camera 640×480):

  quad_px  — проекция плитки, если бы ничего не мешало (геометрия, растеризация);
  clip_px  — сколько из неё внутри кадра (обрезка краем);
  seg_px   — сколько РЕАЛЬНО видно по маске сегментации (край + рука + куб);
  crop     = clip/quad   (1.0 = плитка целиком в кадре);
  occl     = 1 − seg/clip (доля, съеденная рукой и кубом);
  vis      = seg/quad;
  q_pp/q_pm/q_mp/q_mm — видимость по ЧЕТВЕРТЯМ плитки в её собственных осях
             (знаки = локальные +x/−x, +y/−y ДО разворота): по ним видно, цела
             ли именно та четверть, где лицо.

Полуразмер плитки берётся из реальной геометрии меша (model_db.json врёт:
объявленный bbox 0.11 против фактических ~0.146 у glb, т.е. при scale 1.3
сторона плитки 0.189 м, а не 0.143 — на это опирались старые оценки зон).

⚠ Разворот: числа по четвертям верны только для yaw, КРАТНЫХ 90°. Меш отдаёт
bounds уже повёрнутыми, и на промежуточных углах (45/60/75) полуразмер берётся
от AABB — площадь завышается ~вдвое, occl/четверти становятся мусором.

  BOARD_XY_SCALE=1.0 python3 center_card_probe.py --assets pairs_single_pilot \
      --ids 0 1 2 3 "--places=-0.25,0.0;-0.30,0.0" --yaws 90 180 270 \
      --out-dir ../outputs/center_probe
"""
import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

os.environ["A2A_SINGLE_TILE"] = "1"  # до создания env

A = Path(os.environ.get("REPO_ROOT", "/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer"))
sys.path.insert(0, str(A / "scripts"))


def _stub_heavy_imports():
    """magma_vlm_qa тянет transformers/safetensors ради самой модели — нам нужен
    только build_env_args. Заглушаем (как в tile_visibility.py)."""
    for mod, attrs in (("safetensors", ["safe_open"]),
                       ("huggingface_hub", ["snapshot_download"]),
                       ("transformers", ["AutoModelForCausalLM", "AutoProcessor"])):
        if mod not in sys.modules:
            m = types.ModuleType(mod)
            for a in attrs:
                setattr(m, a, None)
            sys.modules[mod] = m


def obs_camera(base):
    """Камера, чей кадр реально идёт модели: 3rd_view_camera 640×480.
    human_render_cameras отдаёт render_camera 512×512 — по ней метрики фиктивны."""
    for holder in (getattr(base.scene, "sensors", {}) or {},
                   base.scene.human_render_cameras or {}):
        if "3rd_view_camera" in holder:
            return holder["3rd_view_camera"]
    raise SystemExit("не нашёл obs-камеру 3rd_view_camera")


def tile_half_size(actor, fallback):
    """Реальный полуразмер плитки в мире (меш, а не запись в model_db)."""
    try:
        m = actor.get_first_collision_mesh()
        b = np.asarray(m.bounds, dtype=float)
        return float((b[1, 0] - b[0, 0]) / 2), float((b[1, 1] - b[0, 1]) / 2)
    except Exception as e:
        print(f"  [warn] меш не прочитался ({e}), беру bbox из model_db", flush=True)
        return fallback


def project(pts_xyz, K, E):
    cc = (E[:3, :3] @ np.asarray(pts_xyz, dtype=float).T).T + E[:3, 3]
    uv = (K @ cc.T).T
    return uv[:, :2] / uv[:, 2:3]


def raster(uv, W, H):
    """Маска полигона в кадре (обрезка краем учтена растеризацией)."""
    img = Image.new("1", (W, H), 0)
    ImageDraw.Draw(img).polygon([tuple(map(float, q)) for q in uv], fill=1)
    return np.asarray(img, dtype=bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="pairs_single_pilot")
    ap.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3],
                    help="эпизоды (карточки), обычно 4 демографии одной сцены")
    ap.add_argument("--places", required=True,
                    help="точки 'x,y' через ';' (argparse не берёт nargs+ с минусом "
                         "в начале — писать --places=-0.25,0.0;-0.30,0.0). "
                         "y=±0.155 воспроизводит старые слоты")
    ap.add_argument("--yaws", type=float, nargs="+", default=[90.0],
                    help="развороты карточки в градусах (90 = как во всех прежних прогонах)")
    ap.add_argument("--episode-len", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    _stub_heavy_imports()
    from magma_vlm_qa import build_env_args               # noqa: E402
    from simpler_env.env.simpler_wrapper_v4 import SimplerWrapper  # noqa: E402

    asset_path = str(A / "ManiSkill" / "mani_skill" / "assets" / "carrot")
    pairs = {p["index"]: p for p in
             json.loads((Path(asset_path) / args.assets / "pairs.json").read_text())}

    eargs, _, _ = build_env_args(args.assets, min(args.ids), 1, False,
                                 asset_path, args.episode_len, args.seed)
    ids = list(args.ids)
    eargs.ids = ids
    eargs.num_envs = len(ids)
    eargs.shard_start, eargs.shard_end = min(ids), max(ids) + 1

    env = SimplerWrapper(eargs)
    base = env.env.unwrapped
    cam = obs_camera(base)
    K = cam.camera.get_intrinsic_matrix()
    K = np.asarray(K[0].cpu() if hasattr(K, "cpu") else K, dtype=float)
    W, H = int(cam.config.width), int(cam.config.height)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    QUAD = {"q_pp": (0, 1, 0, 1), "q_pm": (0, 1, -1, 0),
            "q_mp": (-1, 0, 0, 1), "q_mm": (-1, 0, -1, 0)}
    for place in [s for s in args.places.split(";") if s.strip()]:
        x, y = (float(v) for v in place.split(","))
        for yaw in args.yaws:
            os.environ["A2A_SINGLE_TILE_X"] = str(x)
            os.environ["A2A_SINGLE_TILE_Y"] = str(y)
            os.environ["A2A_SINGLE_TILE_YAW"] = str(yaw)
            env.reset(eargs.obj_set)      # env-переменные читаются на reset'е

            obs = base.get_obs()
            sd = obs["sensor_data"]["3rd_view_camera"]
            rgb = sd["rgb"].to("cpu").numpy().astype(np.uint8)          # [B,H,W,3]
            seg = sd["segmentation"].to("cpu").numpy()                  # [B,H,W,1]
            E = cam.camera.get_extrinsic_matrix()   # камера на base_link — после reset
            E = np.asarray(E[0].cpu() if hasattr(E, "cpu") else E, dtype=float)

            tag = f"x{x:+.2f}_y{y:+.3f}_yaw{int(yaw):03d}".replace(".", "p")
            d = args.out_dir / tag
            d.mkdir(exist_ok=True)
            for i, ep in enumerate(ids):
                name = base._current_left_names[i]
                actor = base.objs_board[name]
                p = actor.pose.p[i].detach().cpu().numpy()
                size = base.model_bbox_sizes[name].detach().cpu().numpy()
                hx, hy = tile_half_size(actor, (size[0] / 2, size[1] / 2))
                zt = p[2] + size[2] / 2
                c, s = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
                R = np.array([[c, -s], [s, c]])

                def world(sx0, sx1, sy0, sy1):
                    loc = np.array([[sx0 * hx, sy0 * hy], [sx1 * hx, sy0 * hy],
                                    [sx1 * hx, sy1 * hy], [sx0 * hx, sy1 * hy]])
                    w = (R @ loc.T).T + p[:2]
                    return np.column_stack([w, np.full(4, zt)])

                sid = actor._objs[i].per_scene_id
                vis_mask = (seg[i, :, :, 0] == sid)
                seg_px = int(vis_mask.sum())

                uv = project(world(-1, 1, -1, 1), K, E)
                clip_mask = raster(uv, W, H)
                clip_px = int(clip_mask.sum())
                quad_px = float(abs(np.dot(uv[:, 0], np.roll(uv[:, 1], 1))
                                    - np.dot(uv[:, 1], np.roll(uv[:, 0], 1))) / 2)
                margin = float(min(uv[:, 0].min(), W - uv[:, 0].max(),
                                   uv[:, 1].min(), H - uv[:, 1].max()))

                qvis = {}
                for qname, (a0, a1, b0, b1) in QUAD.items():
                    qm = raster(project(world(a0, a1, b0, b1), K, E), W, H)
                    tot = int(qm.sum())
                    qvis[qname] = round(int((qm & vis_mask).sum()) / max(tot, 1), 3)

                Image.fromarray(rgb[i]).save(d / f"ep{ep}_full.png")
                if seg_px > 0:                   # увеличенный кроп плитки — смотреть лицо
                    vs, us = np.nonzero(vis_mask)
                    pad = 10
                    u0, u1 = max(0, us.min() - pad), min(W, us.max() + pad)
                    v0, v1 = max(0, vs.min() - pad), min(H, vs.max() + pad)
                    crop = Image.fromarray(rgb[i][v0:v1, u0:u1])
                    crop = crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS)
                    crop.save(d / f"ep{ep}_tile.png")

                meta = pairs.get(ep, {})
                rows.append(dict(
                    place=place, x=x, y=y, yaw=yaw, ep=ep, card=name,
                    race=meta.get("race"), gender=meta.get("gender"),
                    half=[round(hx, 4), round(hy, 4)],
                    quad_px=round(quad_px, 1), clip_px=clip_px, seg_px=seg_px,
                    crop=round(clip_px / max(quad_px, 1e-9), 4),
                    occl=round(1 - seg_px / max(clip_px, 1), 4),
                    vis=round(seg_px / max(quad_px, 1e-9), 4),
                    margin_px=round(margin, 1), **qvis))
            agg = [r for r in rows if r["place"] == place and r["yaw"] == yaw]
            q = {k: np.mean([r[k] for r in agg]) for k in QUAD}
            print(f"{place:>13} yaw={int(yaw):3d}: quad={np.mean([r['quad_px'] for r in agg]):6.0f}px "
                  f"crop={np.mean([r['crop'] for r in agg]):.2f} "
                  f"occl={np.mean([r['occl'] for r in agg]):.2f} "
                  f"margin={np.mean([r['margin_px'] for r in agg]):+5.0f}px | "
                  + " ".join(f"{k}={v:.2f}" for k, v in q.items()) + f"  -> {d.name}",
                  flush=True)

    (args.out_dir / "probe.json").write_text(json.dumps(rows, indent=1))
    try:
        env.env.close()
    except Exception:
        pass
    print(f"\n-> {args.out_dir}/probe.json ({len(rows)} строк)")


if __name__ == "__main__":
    main()
