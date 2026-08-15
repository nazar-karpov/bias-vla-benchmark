#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Подрезка плиток краем кадра — объективный критерий выбора масштаба (v2).

v1 давал vis_R≈0.5 на всех масштабах — ошибка клиппера. Здесь площадь видимой
части считается РАСТЕРИЗАЦИЕЙ маски полигона (без хитрой геометрии): рисуем
плитку в маску размера кадра и в маску «бесконечного» холста и делим площади.

Метрики на масштаб:
  vis_L / vis_R  — доля площади плитки, попавшая в кадр (1.0 = вся видна);
  asym_vis       — |vis_L − vis_R| — АСИММЕТРИЯ подрезки (позиционный артефакт);
  px_L / px_R    — видимая площадь в пикселях (сколько семантики доступно);
  px_ratio       — px_R/px_L: перекос «сколько видно» между сторонами;
  margin_px      — мин. расстояние от плитки до края кадра (запас до подрезки).
"""
import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

A = Path(os.environ.get("REPO_ROOT", "/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer"))
sys.path.insert(0, str(A / "scripts"))
for _m, _attrs in (("safetensors", ["safe_open"]),
                   ("huggingface_hub", ["snapshot_download"]),
                   ("transformers", ["AutoModelForCausalLM", "AutoProcessor"])):
    if _m not in sys.modules:
        _mod = types.ModuleType(_m)
        for _a in _attrs:
            setattr(_mod, _a, None)
        sys.modules[_m] = _mod


def _shoelace(pts):
    if len(pts) < 3:
        return 0.0
    p = np.asarray(pts, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _clip_half(poly, axis, val, keep_greater):
    """Sutherland-Hodgman: обрезка выпуклого полигона полуплоскостью."""
    if len(poly) == 0:
        return poly
    out = []
    n = len(poly)
    for i in range(n):
        cur, prv = poly[i], poly[i - 1]
        ci = (cur[axis] >= val) if keep_greater else (cur[axis] <= val)
        pi = (prv[axis] >= val) if keep_greater else (prv[axis] <= val)
        if ci != pi:                       # пересечение ребра с границей
            t = (val - prv[axis]) / (cur[axis] - prv[axis] + 1e-12)
            out.append(prv + t * (cur - prv))
        if ci:
            out.append(cur)
    return np.array(out) if out else np.zeros((0, 2))


def poly_px_area(uv, W, H, pad=None):
    """(видимая в кадре площадь, полная площадь) — точной геометрией.

    v2 растеризовал полигон на холсте с PAD=2000 и ломался на координатах PIL;
    отсюда фиктивные vis_R≈0.5 даже когда плитка целиком в кадре.
    """
    full = _shoelace(uv)
    poly = np.asarray(uv, dtype=float)
    for axis, val, gt in ((0, 0.0, True), (0, float(W), False),
                          (1, 0.0, True), (1, float(H), False)):
        poly = _clip_half(poly, axis, val, gt)
    return _shoelace(poly), full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--assets", default="pairs_choice_vla_confirm")
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from magma_vlm_qa import build_env_args
    from simpler_env.env.simpler_wrapper_v4 import SimplerWrapper

    asset_path = str(A / "ManiSkill" / "mani_skill" / "assets" / "carrot")
    eargs, ids, _ = build_env_args(args.assets, 0, args.count, False, asset_path, 80, 0)
    env = SimplerWrapper(eargs)
    env.reset(eargs.obj_set)
    base = env.env.unwrapped

    # ВАЖНО: мерить надо по камере, чей кадр реально идёт модели —
    # 3rd_view_camera 640x480 (obs). human_render_cameras здесь содержит только
    # render_camera 512x512 (для видео) — по ней получались фиктивные vis≈0.49.
    cam = None
    for holder in (getattr(base.scene, "sensors", {}) or {},
                   base.scene.human_render_cameras or {}):
        if "3rd_view_camera" in holder:
            cam = holder["3rd_view_camera"]
            break
    if cam is None:  # запасной путь: любая камера 640x480
        for holder in (getattr(base.scene, "sensors", {}) or {},
                       base.scene.human_render_cameras or {}):
            for c in holder.values():
                if int(c.config.width) == 640 and int(c.config.height) == 480:
                    cam = c
                    break
    if cam is None:
        raise SystemExit("не нашёл obs-камеру 3rd_view_camera (640x480)")
    K = cam.camera.get_intrinsic_matrix()
    Ext = cam.camera.get_extrinsic_matrix()
    K = np.asarray(K[0].cpu() if hasattr(K, "cpu") else K, dtype=float)
    Ext = np.asarray(Ext[0].cpu() if hasattr(Ext, "cpu") else Ext, dtype=float)
    H, W = int(cam.config.height), int(cam.config.width)
    print(f"  камера: {W}x{H}", flush=True)

    rows = []
    for i in range(len(ids)):
        for side, names in (("L", base._current_left_names), ("R", base._current_right_names)):
            actor = base.objs_board[names[i]]
            p = actor.pose.p[i].detach().cpu().numpy()
            size = base.model_bbox_sizes[names[i]].detach().cpu().numpy()
            hx, hy = size[0] / 2, size[1] / 2
            z = p[2] + size[2] / 2
            corners = np.array([[p[0]-hx, p[1]-hy, z], [p[0]+hx, p[1]-hy, z],
                                [p[0]+hx, p[1]+hy, z], [p[0]-hx, p[1]+hy, z]])
            cc = (Ext[:3, :3] @ corners.T).T + Ext[:3, 3]
            uv = (K @ cc.T).T
            uv = uv[:, :2] / uv[:, 2:3]
            vis, full = poly_px_area(uv, W, H)
            # запас до края кадра (отрицательный = вылезает)
            margin = min(uv[:, 0].min() - 0, W - uv[:, 0].max(),
                         uv[:, 1].min() - 0, H - uv[:, 1].max())
            rows.append({"side": side, "vis": vis / max(full, 1), "px": vis,
                         "margin": float(margin)})

    try:
        env.env.close()
    except Exception:
        pass

    def col(side, key):
        return np.array([r[key] for r in rows if r["side"] == side], dtype=float)

    res = {
        "vis_L": float(col("L", "vis").mean()), "vis_R": float(col("R", "vis").mean()),
        "px_L": float(col("L", "px").mean()), "px_R": float(col("R", "px").mean()),
        "margin_L": float(col("L", "margin").mean()), "margin_R": float(col("R", "margin").mean()),
        "n": len(rows) // 2,
    }
    res["asym_vis"] = abs(res["vis_L"] - res["vis_R"])
    res["px_ratio_R_L"] = res["px_R"] / max(res["px_L"], 1e-9)
    print(f"scale={args.scale}: vis_L={res['vis_L']:.4f} vis_R={res['vis_R']:.4f} "
          f"asym={res['asym_vis']:.4f} px_L={res['px_L']:.0f} px_R={res['px_R']:.0f} "
          f"px_R/L={res['px_ratio_R_L']:.3f} margin_L={res['margin_L']:.0f} "
          f"margin_R={res['margin_R']:.0f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({str(args.scale): res}, indent=1))


if __name__ == "__main__":
    main()
