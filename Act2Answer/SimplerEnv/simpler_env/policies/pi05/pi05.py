"""Act2Answer policy client for lerobot pi0.5 (pi05), checkpoint qownscks/pi05_widowx.

pi05 runs as a separate zmq server (scripts/pi05_server.py) in its own conda env
(lerobot + patched transformers). This is a *thin client* living in the eval env
(magma_act2answer); it never imports lerobot. Same REQ/REP + msgpack wire protocol
as the RLDX client (see policies/rldx/rldx.py).

SimplerEnv contract (mirrors internvla.py / rldx.py):
  get_action(obs, deterministic) -> torch.FloatTensor (B, 7)
    obs["image"]            : torch uint8 (B, H, W, 3)
    obs["task_description"] : list[str] length B
    return per env: [world_vector(3), rot_axangle(3), gripper(1)]   # widowx bridge

The server returns the raw pi05 WidowX action per env: [dx, dy, dz, d_roll,
d_pitch, d_yaw, gripper]. We convert the euler-delta rotation to an axis-angle
vector (as InternVLA/RLDX do) and apply the widowx sticky-gripper postprocess so
the env's gripper timing matches the other bridge policies.

pi05 keeps a 50-step action chunk queue *on the server*; a per-episode reset()
clears it. run.py calls prep_rollout() at each episode start -> we send reset.
"""
from __future__ import annotations

import io

import msgpack
import numpy as np
import torch
from transforms3d.euler import euler2axangle


# ------------------------------------------------------------------- wire ---
def _encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj):
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def _pack(d):
    return msgpack.packb(d, default=_encode)


def _unpack(b):
    return msgpack.unpackb(b, object_hook=_decode)


class _Pi05Client:
    def __init__(self, host="127.0.0.1", port=20005, timeout_ms=180000):
        import zmq
        self._zmq = zmq
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self.ctx = zmq.Context.instance()
        self._connect()

    def _connect(self):
        self.sock = self.ctx.socket(self._zmq.REQ)
        self.sock.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(self._zmq.LINGER, 0)
        self.sock.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint, data=None):
        req = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        self.sock.send(_pack(req))
        resp = _unpack(self.sock.recv())
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"pi05 server error: {resp['error']}")
        return resp


class Pi05Inference:
    """lerobot pi05 client policy with the SimplerEnv get_action(obs) contract."""

    def __init__(self, host="127.0.0.1", port=20005, policy_setup="widowx_bridge"):
        assert policy_setup == "widowx_bridge", "pi05 client is wired for widowx (bridge)"
        import os
        self.policy_setup = policy_setup
        host = os.environ.get("PI05_HOST", host)
        port = int(os.environ.get("PI05_PORT", port))
        self.client = _Pi05Client(host, port)
        self.client.call("ping")
        self.client.call("reset")
        self._first = True
        # action_scale: qownscks/pi05_widowx is a real-robot finetune; its delta
        # magnitudes overshoot the SimplerEnv workspace and fling the cube. Scale
        # pos+rot deltas down (tunable via PI05_ACTION_SCALE, default 1.0).
        self.action_scale = float(os.environ.get("PI05_ACTION_SCALE", "1.0"))
        # gripper polarity: flip via PI05_GRIP_SIGN=-1 if grasp/release inverted.
        self.grip_sign = float(os.environ.get("PI05_GRIP_SIGN", "1.0"))
        # sticky gripper (widowx), mirrors rldx.py / bridge policies
        self._sticky_on: dict[int, bool] = {}
        self._sticky_val: dict[int, float] = {}
        self._sticky_rep: dict[int, int] = {}
        self.sticky_num_repeat = 1

    # ---- required by run.py's render loop -----------------------------------
    def prep_rollout(self):
        self.client.call("reset")
        self._first = True
        self._sticky_on.clear(); self._sticky_val.clear(); self._sticky_rep.clear()

    def reset(self, task_description=None):
        self.prep_rollout()

    def _postprocess_gripper(self, env_i: int, g: float) -> float:
        # pi05 widowx gripper channel -> widowx [-1,1] sticky (same shape as rldx.py).
        rel = float(np.clip(g, -1.0, 1.0))
        if abs(rel) > 0.5 and not self._sticky_on.get(env_i, False):
            self._sticky_on[env_i] = True
            self._sticky_val[env_i] = rel
        if self._sticky_on.get(env_i, False):
            self._sticky_rep[env_i] = self._sticky_rep.get(env_i, 0) + 1
            rel = self._sticky_val[env_i]
        if self._sticky_rep.get(env_i, 0) == self.sticky_num_repeat:
            self._sticky_on[env_i] = False
            self._sticky_rep[env_i] = 0
            self._sticky_val[env_i] = 0.0
        return rel

    @torch.no_grad()
    def get_action(self, obs, deterministic: bool = True) -> torch.Tensor:
        images = obs["image"]
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        images = np.asarray(images, dtype=np.uint8)  # (B, H, W, 3)
        instructions = obs["task_description"]
        if isinstance(instructions, str):
            instructions = [instructions] * images.shape[0]
        B = images.shape[0]

        # WidowX proprio: the pi05 prompt discretizes a padded state; for this bench
        # we feed a neutral zero state, matching rldx.py (state not the bias signal).
        states = np.zeros((B, 7), dtype=np.float32)
        resets = [self._first] * B
        self._first = False

        resp = self.client.call("get_action", {
            "images": images,
            "state": states,
            "tasks": list(instructions),
            "reset": resets,
        })
        actions = np.asarray(resp["action"], dtype=np.float64)  # (B, 7)

        vecs = []
        for i in range(B):
            a = actions[i]
            pos = a[:3]
            euler = a[3:6]
            axes, ang = euler2axangle(euler[0], euler[1], euler[2])
            rot_axangle = axes * ang
            grip = self._postprocess_gripper(i, float(a[6]))
            vecs.append(np.concatenate([pos, rot_axangle, [grip]]).astype(np.float32))
        return torch.from_numpy(np.stack(vecs)).float().cuda()
