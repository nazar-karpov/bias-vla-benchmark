#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проба РАССТАНОВКИ двух плиток: масштаб × расстояние между плитками × сдвиг пары.

Зачем: при рабочем масштабе 1.3 правая плитка режется правым краем кадра
(камера Bridge смотрит несимметрично: правая плитка ×1.8 пикселей левой).
Возврат к масштабу 1.0 нежелателен — у Magma крен и answer-rate зависят от
размера плитки (metrics/scale_sweep.md). Альтернатива — сблизить плитки
(A2A_TILE_Y < 0.155) и/или сдвинуть пару влево (A2A_TILE_YC < 0).

На каждый вариант: реальный первый кадр obs-камеры (3rd_view_camera 640×480,
тот, что идёт модели) + метрики по маске сегментации:
  seg_px_L/R  — реально видимых пикселей плитки (край + рука + куб);
  clip_L/R    — доля площади плитки внутри кадра (по проекции углов меша);
  margin_L/R  — запас до ближайшего края кадра в px (<0 = режется);
  gap_cm      — просвет между плитками на столе (по реальному размеру меша).

Масштаб = model_db (1.3) × BOARD_XY_SCALE; BOARD_XY_SCALE читается при импорте
env, поэтому один процесс = один масштаб; расстановки перебираются reset'ами.

  BOARD_XY_SCALE=1.0 python tile_layout_probe.py --ids 0 \
      "--layouts=0.155,0;0.13,0;0.155,-0.03" --out-dir ../outputs/tile_layout
"""
import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

A = Path(os.environ.get("REPO_ROOT", "/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer"))
sys.path.insert(0, str(A / "scripts"))


def _stub_heavy_imports():
    for mod, attrs in (("safetensors", ["safe_open"]),
                       ("huggingface_hub", ["snapshot_download"]),
                       ("transformers", ["AutoModelForCausalLM", "AutoProcessor"])):
        if mod not in sys.modules:
            m = types.ModuleType(mod)
            for a in attrs:
                setattr(m, a, None)
            sys.modules[mod] = m


def obs_camera(base):
    for holder in (getattr(base.scene, "sensors", {}) or {},
                   base.scene.human_render_cameras or {}):
        if "3rd_view_camera" in holder:
            return holder["3rd_view_camera"]
    raise SystemExit("не нашёл obs-камеру 3rd_view_camera")


def tile_half_size(actor, fallback):
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
    img = Image.new("1", (W, H), 0)
    ImageDraw.Draw(img).polygon([tuple(map(float, q)) for q in uv], fill=1)
    return np.asarray(img, dtype=bool)


def quad_area(uv):
    return float(abs(np.dot(uv[:, 0], np.roll(uv[:, 1], 1))
                     - np.dot(uv[:, 1], np.roll(uv[:, 0], 1))) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="pairs_choice_vla_confirm")
    ap.add_argument("--ids", type=int, nargs="+", default=[0])
    ap.add_argument("--layouts", required=True,
                    help="'y_half,yc[,x]' через ';' (писать --layouts=... из-за минусов)")
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--episode-len", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true",
                    help="CPU-нода: physx_cpu + программный Vulkan (lavapipe)")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    _stub_heavy_imports()
    import gymnasium as gym
    from magma_vlm_qa import build_env_args               # noqa: E402
    import simpler_env.env.simpler_wrapper_v4 as swmod    # noqa: E402
    from simpler_env.env.simpler_wrapper_v4 import SimplerWrapper  # noqa: E402

    if args.cpu:
        # sapien.Device("cpu") не находит SwiftShader/llvmpipe как физ. устройство;
        # RenderSystem(None) берёт устройство по умолчанию — единственное доступное.
        import torch as _torch
        _torch.cuda.synchronize = lambda *a, **k: None   # sapien_env зовёт безусловно
        import mani_skill.envs.sapien_env as _se
        _orig_parse = _se.parse_sim_and_render_backend

        def _parse_cpu(sim_backend, render_backend):
            bi = _orig_parse(sim_backend, render_backend)
            bi.render_device = None
            return bi
        _se.parse_sim_and_render_backend = _parse_cpu
        _orig_make = gym.make

        def _make_cpu(**kw):
            kw["sim_backend"] = "cpu"
            kw["render_backend"] = "cpu"
            return _orig_make(**kw)
        swmod.gym.make = _make_cpu

    scale_env = float(os.environ.get("BOARD_XY_SCALE", "1.3"))
    asset_path = str(A / "ManiSkill" / "mani_skill" / "assets" / "carrot")
    pairs = {p["index"]: p for p in
             json.loads((Path(asset_path) / args.assets / "pairs.json").read_text())}

    eargs, _, _ = build_env_args(args.assets, min(args.ids), 1, args.swap,
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
    db_scale = float(next(iter(base.model_db_board.values())).get("scales", [1.0])[0])
    eff_scale = db_scale * scale_env
    print(f"камера {W}x{H}; model_db scale={db_scale} × BOARD_XY_SCALE={scale_env} "
          f"= эффективный {eff_scale:.3f}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for lay in [s for s in args.layouts.split(";") if s.strip()]:
        vals = [float(v) for v in lay.split(",")]
        y_half, yc = vals[0], vals[1]
        x = vals[2] if len(vals) > 2 else -0.25
        os.environ["A2A_TILE_Y"] = str(y_half)
        os.environ["A2A_TILE_YC"] = str(yc)
        os.environ["A2A_TILE_X"] = str(x)
        env.reset(eargs.obj_set)

        obs = base.get_obs()
        sd = obs["sensor_data"]["3rd_view_camera"]
        rgb = sd["rgb"].to("cpu").numpy().astype(np.uint8)
        seg = sd["segmentation"].to("cpu").numpy()
        E = cam.camera.get_extrinsic_matrix()
        E = np.asarray(E[0].cpu() if hasattr(E, "cpu") else E, dtype=float)

        tag = (f"s{eff_scale:.2f}_y{y_half:.3f}_yc{yc:+.3f}_x{x:+.2f}"
               + ("_swap" if args.swap else "")).replace(".", "p")
        for i, ep in enumerate(ids):
            row = {"tag": tag, "ep": ep, "scale": round(eff_scale, 3),
                   "y_half": y_half, "yc": yc, "x": x, "swap": args.swap}
            half = None
            for side, names in (("L", base._current_left_names),
                                ("R", base._current_right_names)):
                actor = base.objs_board[names[i]]
                p = actor.pose.p[i].detach().cpu().numpy()
                size = base.model_bbox_sizes[names[i]].detach().cpu().numpy()
                hx, hy = tile_half_size(actor, (size[0] / 2, size[1] / 2))
                half = (hx, hy)
                zt = p[2] + size[2] / 2
                corners = np.array([[p[0]-hx, p[1]-hy, zt], [p[0]+hx, p[1]-hy, zt],
                                    [p[0]+hx, p[1]+hy, zt], [p[0]-hx, p[1]+hy, zt]])
                uv = project(corners, K, E)
                clip_mask = raster(uv, W, H)
                clip_px = int(clip_mask.sum())
                quad_px = quad_area(uv)
                margin = float(min(uv[:, 0].min(), W - uv[:, 0].max(),
                                   uv[:, 1].min(), H - uv[:, 1].max()))
                sid = actor._objs[i].per_scene_id
                seg_px = int((seg[i, :, :, 0] == sid).sum())
                row[f"seg_px_{side}"] = seg_px
                row[f"quad_px_{side}"] = round(quad_px)
                row[f"clip_{side}"] = round(clip_px / max(quad_px, 1), 4)
                row[f"margin_{side}"] = round(margin, 1)
                row[f"uv_{side}"] = np.round(uv, 1).tolist()
                row[f"name_{side}"] = names[i]
            # просвет между плитками на столе (реальный размер меша, yaw=90 → квадрат)
            row["tile_side_cm"] = round(2 * half[1] * 100, 1)
            row["gap_cm"] = round((2 * y_half - 2 * half[1]) * 100, 1)
            rows.append(row)
            img = Image.fromarray(rgb[i])
            img.save(args.out_dir / f"{tag}_ep{ep}.png")
            ov = img.copy()
            d = ImageDraw.Draw(ov)
            for side, col in (("L", (255, 60, 60)), ("R", (60, 255, 60))):
                d.polygon([tuple(q) for q in row[f"uv_{side}"]], outline=col, width=2)
            ov.save(args.out_dir / f"{tag}_ep{ep}_overlay.png")
            print(f"{tag} ep{ep}: side={row['tile_side_cm']}см gap={row['gap_cm']}см | "
                  f"L clip={row['clip_L']} margin={row['margin_L']} seg={row['seg_px_L']} | "
                  f"R clip={row['clip_R']} margin={row['margin_R']} seg={row['seg_px_R']}",
                  flush=True)

    out = args.out_dir / f"probe_s{eff_scale:.2f}{'_swap' if args.swap else ''}.json".replace(".", "p", 1)
    out = args.out_dir / (f"probe_s{eff_scale:.2f}".replace(".", "p") + ("_swap" if args.swap else "") + ".json")
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print("->", out, flush=True)
    try:
        env.env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
