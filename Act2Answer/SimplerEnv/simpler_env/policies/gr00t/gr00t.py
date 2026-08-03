"""GR00T N1.7 (SimplerEnv-Bridge) client policy.

Сервер: Isaac-GR00T gr00t/eval/run_gr00t_server.py (zmq REQ/REP + msgpack_numpy,
порт GR00T_PORT, дефолт 5555), embodiment simpler_env_widowx.
Маппинг obs/action скопирован 1:1 с их gr00t/eval/sim/SimplerEnv/simpler_env.py
WidowXBridgeEnv (image 256x256, state = eef xyz + euler(bridge default_rot) +
pad + gripper, gripper 2*(g>0.5)-1).
"""
import os
from collections import deque

import cv2
import msgpack
import msgpack_numpy as mnp
import numpy as np
import torch
import zmq
from transforms3d import euler as te
from transforms3d import quaternions as tq

_DEFAULT_ROT = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
_IMG = 256


def _pack(data):
    return msgpack.packb(data, default=mnp.encode)


def _unpack(b):
    return msgpack.unpackb(b, object_hook=mnp.decode, raw=False)


class GR00TPolicy:
    def __init__(self, host=None, port=None, replan_steps=None):
        self.host = host or os.environ.get("GR00T_HOST", "127.0.0.1")
        self.port = int(port or os.environ.get("GR00T_PORT", 5555))
        self.replan_steps = int(replan_steps or os.environ.get("GR00T_REPLAN", 8))
        self.ctx = zmq.Context()
        self._connect()
        self.action_plans = []
        self.task_descriptions = []

    def _connect(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, 60000)
        self.sock.setsockopt(zmq.SNDTIMEO, 60000)
        self.sock.connect(f"tcp://{self.host}:{self.port}")

    def _call(self, obs_payload):
        req = {"endpoint": "get_action", "data": {"observation": obs_payload}}
        self.sock.send(_pack(req))
        msg = self.sock.recv()
        if msg == b"ERROR":
            raise RuntimeError("GR00T server error")
        resp = _unpack(msg)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"GR00T server error: {resp['error']}")
        return resp

    # --- интерфейс Act2Answer (как xiaomi/rldx) ---
    def prep_rollout(self):
        pass

    def reset(self, task_descriptions):
        self.action_plans = [deque() for _ in task_descriptions]
        self.task_descriptions = list(task_descriptions)

    def get_action(self, obs, _deterministic=True):
        return self.step(obs["image"], obs["task_description"], obs["proprio"])

    def _build_payload(self, image, instruction, pr):
        img = image.cpu().numpy() if hasattr(image, "cpu") else np.asarray(image)
        img = cv2.resize(img, (_IMG, _IMG)).astype(np.uint8)
        dump = os.environ.get("GR00T_DUMP_IMG")
        if dump and not os.path.exists(dump):
            cv2.imwrite(dump, img[:, :, ::-1])  # cv2 пишет BGR; если img RGB — файл верный

        eef = pr["agent"]["eef_pos"]
        eef = eef.cpu().numpy() if hasattr(eef, "cpu") else np.asarray(eef)
        rm = tq.quat2mat(eef[3:7])
        rpy = te.mat2euler(rm @ _DEFAULT_ROT.T)
        def _st(v):  # (B=1, T=1, D=1) float32 — формат Gr00tSimPolicyWrapper
            return np.asarray([[[float(v)]]], dtype=np.float32)

        return {
            "video.image_0": img[None, None],  # (1, 1, H, W, 3) uint8
            "state.x": _st(eef[0]),
            "state.y": _st(eef[1]),
            "state.z": _st(eef[2]),
            "state.roll": _st(rpy[0]),
            "state.pitch": _st(rpy[1]),
            "state.yaw": _st(rpy[2]),
            "state.pad": _st(0.0),
            "state.gripper": _st(eef[7]),
            "annotation.human.action.task_description": [str(instruction)],
        }

    def _compute_plan(self, images, task_descriptions, proprio):
        for i, (image, instruction, pr) in enumerate(zip(images, task_descriptions, proprio)):
            resp = self._call(self._build_payload(image, instruction, pr))
            if isinstance(resp, (list, tuple)):
                resp = resp[0]                      # (action, info) -> action
            cols = [np.asarray(resp[k]).reshape(-1) for k in (
                "action.x", "action.y", "action.z",
                "action.roll", "action.pitch", "action.yaw", "action.gripper")]
            h = min(self.replan_steps, len(cols[0]))
            self.action_plans[i] = deque(np.stack(cols, axis=1)[:h])

    def step(self, images, task_descriptions, proprio, *args, **kwargs):
        tds = list(task_descriptions)
        if not self.action_plans or tds != self.task_descriptions or \
                len(self.action_plans) != len(tds):
            self.reset(tds)
        if any(len(p) == 0 for p in self.action_plans):
            self._compute_plan(images, tds, proprio)

        actions = []
        for i in range(len(images)):
            a = self.action_plans[i].popleft()
            # как в их WidowXBridgeEnv.step: xyz + roll/pitch/yaw НАПРЯМУЮ (без
            # euler2axangle) + грипер 2*(g>0.5)-1
            act = np.concatenate([a[0:3], a[3:6], [2.0 * (float(a[6]) > 0.5) - 1.0]])
            if os.environ.get("GR00T_FLIP_Y"):  # диагностика зеркала по y
                act[1] = -act[1]
            if os.environ.get("GR00T_FLIP_XY"):  # диагностика 180°-поворота фрейма
                act[0] = -act[0]
                act[1] = -act[1]
            actions.append(act)
        return torch.tensor(np.stack(actions), dtype=torch.float32)
