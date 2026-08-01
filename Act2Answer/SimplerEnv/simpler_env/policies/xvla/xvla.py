"""Политика X-VLA (2toinf/X-VLA) для Act2Answer.

Схема как у xiaomi: тяжёлая модель живёт в отдельном процессе (их FastAPI-сервер
`deploy.py`), политика — тонкий клиент.

    POST http://host:port/act
        {"proprio": <json_numpy>, "language_instruction": str,
         "image0": <json_numpy>, "domain_id": 0, "steps": N}
    -> {"action": [[x, y, z, r6(6), gripper], ...]}   # чанк из N шагов

ГЛАВНОЕ ОТЛИЧИЕ ОТ ИХ ОРИГИНАЛЬНОГО КЛИЕНТА.
X-VLA предсказывает АБСОЛЮТНУЮ позу схвата, и их форк SimplerEnv переведён на
абсолютное управление. Наша среда работает контроллером
`arm_pd_ee_target_delta_pose_align2`, то есть принимает ДЕЛЬТЫ к накопленной
целевой позе. Поэтому здесь ведётся собственная целевая поза: на каждом шаге
выдаётся разница между новым предсказанием и предыдущим. Постоянный сдвиг систем
координат (мир против базы робота) при таком вычитании сокращается, поэтому
начальную позу достаточно взять из proprio среды.

Формулы поворотов (6D <-> euler) скопированы из их клиента без изменений, включая
поправку +[0, pi/2, 0] — она часть их соглашения о системе координат.
"""
import collections
import math
import os

import json_numpy
import numpy as np
import requests
import torch
from scipy.spatial.transform import Rotation as R


def rotate6D_to_euler_xyz(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return R.from_matrix(np.stack((b1, b2, b3), axis=-1)).as_euler("xyz")


def wrap_pi(angles: np.ndarray) -> np.ndarray:
    """Разница углов в (-pi, pi], иначе на переходе через pi будет скачок."""
    return (angles + np.pi) % (2 * np.pi) - np.pi


class XVLAPolicy:
    """Клиент X-VLA с интерфейсом политик Act2Answer (prep_rollout/reset/get_action)."""

    def __init__(self, host=None, port=None, chunk=10, timeout=60, action_scale=1.0):
        self.host = host or os.environ.get("XVLA_HOST", "localhost")
        self.port = int(port or os.environ.get("XVLA_PORT", 8010))
        self.url = f"http://{self.host}:{self.port}/act"
        self.chunk = int(os.environ.get("XVLA_CHUNK", chunk))
        self.timeout = timeout
        self.action_scale = float(os.environ.get("XVLA_ACTION_SCALE", action_scale))
        self.task_descriptions = []
        self._debug_left = int(os.environ.get("XVLA_DEBUG_STEPS", 3))

    def prep_rollout(self):
        pass

    def reset(self, task_descriptions):
        n = len(task_descriptions)
        self.task_descriptions = list(task_descriptions)
        self.plans = [collections.deque() for _ in range(n)]
        self.proprio = [None] * n          # 20-мерный вектор состояния для модели
        self.target_pos = [None] * n       # накопленная целевая позиция
        self.target_euler = [None] * n     # накопленная целевая ориентация
        self.first_pred = [True] * n       # первое предсказание задаёт начало отсчёта

    # --- служебное -------------------------------------------------------
    def _init_state(self, i, eef):
        """eef: [x, y, z, qw, qx, qy, qz, gripper] из среды."""
        pos = np.asarray(eef[:3], dtype=np.float32)
        quat_wxyz = np.asarray(eef[3:7], dtype=np.float32)
        # proprio как в их клиенте: позиция + фиксированный поворот-заглушка, затем нули
        base = np.concatenate([pos, np.array([1, 0, 0, 1, 0, 0, 0], dtype=np.float32)])
        self.proprio[i] = np.concatenate([base, np.zeros_like(base)]).astype(np.float32)
        # цель НЕ задаём здесь: её задаст первое предсказание модели

    def _request(self, i, image):
        payload = {
            "proprio": json_numpy.dumps(self.proprio[i]),
            "language_instruction": self.task_descriptions[i],
            "image0": json_numpy.dumps(image),
            "domain_id": 0,
            "steps": self.chunk,
        }
        resp = requests.post(self.url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        seq = np.array(resp.json()["action"], dtype=np.float32)
        self.plans[i].extend(seq.tolist())

    # --- интерфейс политики ---------------------------------------------
    def get_action(self, obs, _deterministic=True):
        return self.step(obs["image"], obs["task_description"], obs["proprio"])

    def step(self, images, task_descriptions, proprio, *args, **kwargs):
        if list(task_descriptions) != self.task_descriptions:
            self.reset(list(task_descriptions))

        actions = []
        for i in range(len(images)):
            image = images[i]
            image = image.cpu().numpy() if hasattr(image, "cpu") else np.asarray(image)
            image = image.astype(np.uint8)

            eef = proprio[i]["agent"]["eef_pos"]
            eef = eef.cpu().numpy() if hasattr(eef, "cpu") else np.asarray(eef)
            if self.proprio[i] is None:
                self._init_state(i, eef)

            if not self.plans[i]:
                self._request(i, image)

            pred = np.array(self.plans[i].popleft(), dtype=np.float32)
            self.proprio[i][:10] = pred[:10]

            abs_pos = pred[:3]
            abs_euler = rotate6D_to_euler_xyz(pred[3:9]) + np.array([0.0, math.pi / 2, 0.0])
            gripper = 1.0 if pred[9] < 0.95 else -1.0

            # система координат модели своя, поэтому «нулём» считаем её первое
            # предсказание: иначе первый шаг даёт скачок на всю разницу систем
            if self.target_pos[i] is None or self.first_pred[i]:
                self.target_pos[i] = abs_pos.copy()
                self.target_euler[i] = abs_euler.copy()
                self.first_pred[i] = False

            # абсолютная поза -> дельта к накопленной цели (среда ждёт дельты)
            d_pos = (abs_pos - self.target_pos[i]) * self.action_scale
            d_euler = wrap_pi(abs_euler - self.target_euler[i]) * self.action_scale
            self.target_pos[i] = abs_pos
            self.target_euler[i] = abs_euler

            if self._debug_left > 0 and i == 0:
                print(f"[xvla] abs_pos={np.round(abs_pos, 4)} d_pos={np.round(d_pos, 4)} "
                      f"d_euler={np.round(d_euler, 4)} grip={gripper}", flush=True)
                self._debug_left -= 1

            actions.append(np.concatenate([d_pos, d_euler, [gripper]]).astype(np.float32))

        return torch.tensor(np.stack(actions))
