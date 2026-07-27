"""Act2Answer policy client for RLDX-1 (RLWRLD/RLDX-1).

RLDX-1 runs as a separate zmq policy server (rldx.policy.server_client.PolicyServer,
REQ/REP + msgpack). This module is a *thin client* that lives in the Act2Answer
conda env and speaks the RLDX wire protocol directly, so we never import the heavy
`rldx` package (conflicting deps) into the eval process.

Contract mirrors SimplerEnv's other policies (see internvla.py / magma_model.py):
  get_action(obs, deterministic) -> torch.FloatTensor (B, 7)
    obs["image"]            : torch uint8 (B, H, W, 3)
    obs["task_description"] : list[str] length B
    return per env: [world_vector(3), rot_axangle(3), gripper(1)]

RLDX returns an *action chunk* (several timesteps) per query; Act2Answer calls
get_action every sim step, so we keep a per-env chunk buffer and re-query only when
it drains. Obs/action packing follows RLDX's own widowx SimplerEnv wrapper.

Drop into: SimplerEnv/simpler_env/policies/rldx/rldx.py  (+ empty __init__.py)
and wire `--vla rldx` in run.py to `RLDXInference`.
"""
from __future__ import annotations

import io
from collections import deque

import cv2
import msgpack
import numpy as np
import torch
import zmq
from transforms3d.euler import euler2axangle


# --------------------------------------------------------------------- wire ---
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


def _to_bytes(data) -> bytes:
    return msgpack.packb(data, default=_encode)


def _from_bytes(data: bytes):
    return msgpack.unpackb(data, object_hook=_decode)


class _RLDXClient:
    """Minimal zmq REQ client for the RLDX PolicyServer (matches PolicyClient)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 20000, timeout_ms: int = 120000):
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self.ctx = zmq.Context.instance()
        self._connect()

    def _connect(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint: str, data: dict | None = None, requires_input: bool = True):
        req = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        self.sock.send(_to_bytes(req))
        msg = self.sock.recv()
        if msg == b"ERROR":
            raise RuntimeError("RLDX server error (wrong policy server?)")
        resp = _from_bytes(msg)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"RLDX server error: {resp['error']}")
        return resp

    def get_action(self, observation: dict, options: dict | None = None):
        return self.call("get_action", {"observation": observation, "options": options})

    def reset(self, options: dict | None = None):
        return self.call("reset", {"options": options})

    def ping(self) -> bool:
        try:
            self.call("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self.sock.close(); self._connect()
            return False


# ------------------------------------------------------------------- policy ---
_IMG_H, _IMG_W = 256, 320   # RLDX widowx image_size (rows, cols)
_ACTION_KEYS = ("action.x", "action.y", "action.z",
                "action.roll", "action.pitch", "action.yaw", "action.gripper")


class RLDXInference:
    """RLDX-1 client policy with the SimplerEnv get_action(obs) contract."""

    def __init__(self, host: str = "127.0.0.1", port: int = 20000, policy_setup: str = "widowx_bridge"):
        assert policy_setup == "widowx_bridge", "RLDX Act2Answer client is wired for widowx only"
        self.policy_setup = policy_setup
        self.client = _RLDXClient(host, port)
        self.client.reset()
        # per-env chunk buffers + sticky-gripper state (widowx: repeat 1)
        self._chunks: dict[int, deque] = {}
        self._sticky_on: dict[int, bool] = {}
        self._sticky_val: dict[int, float] = {}
        self._sticky_rep: dict[int, int] = {}
        self.sticky_num_repeat = 1

    # ---- required by run.py's render loop -----------------------------------
    def prep_rollout(self):
        self.client.reset()
        self._chunks.clear()
        self._sticky_on.clear(); self._sticky_val.clear(); self._sticky_rep.clear()

    def reset(self, task_description=None):
        self.prep_rollout()

    # ---- obs packing (mirrors RLDX widowx SimplerEnv wrapper) ----------------
    @staticmethod
    def _pack_obs(image_hw3_uint8: np.ndarray, instruction: str) -> dict:
        img = cv2.resize(image_hw3_uint8, (_IMG_W, _IMG_H))  # (cols, rows)
        # proprio is unknown from Act2Answer's obs at this layer; RLDX-SIMPLER
        # widowx uses eef state, but the released policy tolerates a neutral
        # state (identity quat, open gripper) -- the visual+language stream
        # drives the choice, which is what this bias probe measures.
        return {
            "video.image": img.astype(np.uint8),
            "state.x": [0.0], "state.y": [0.0], "state.z": [0.0],
            "state.rx": [0.0], "state.ry": [0.0], "state.rz": [0.0], "state.rw": [1.0],
            "state.gripper": [0.0],
            "annotation.human.action.task_description": instruction,
        }

    def _postprocess_gripper(self, env_i: int, g01: float) -> float:
        # [0,1] -> [-1,1], relative, sticky (identical to RLDX widowx wrapper)
        cur = (g01 * 2.0) - 1.0
        rel = -cur
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

    def _next_action_vec(self, env_i: int, image: np.ndarray, instruction: str) -> np.ndarray:
        """Return one 7-vec [wx,wy,wz, ax,ay,az, gripper] for env_i."""
        buf = self._chunks.get(env_i)
        if not buf:
            action, _info = self.client.get_action(self._pack_obs(image, instruction))
            # action[key] is a 1-D array over the chunk horizon
            horizon = len(np.atleast_1d(action[_ACTION_KEYS[0]]))
            buf = deque()
            for t in range(horizon):
                buf.append({k: float(np.atleast_1d(action[k])[t]) for k in _ACTION_KEYS})
            self._chunks[env_i] = buf
        a = buf.popleft()

        world = np.array([a["action.x"], a["action.y"], a["action.z"]], dtype=np.float64)
        axes, ang = euler2axangle(a["action.roll"], a["action.pitch"], a["action.yaw"])
        rot_axangle = axes * ang
        grip = self._postprocess_gripper(env_i, a["action.gripper"])
        return np.concatenate([world, rot_axangle, [grip]]).astype(np.float32)

    @torch.no_grad()
    def get_action(self, obs, deterministic: bool = True) -> torch.Tensor:
        images = obs["image"]
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        images = np.asarray(images, dtype=np.uint8)          # (B, H, W, 3)
        instructions = obs["task_description"]
        if isinstance(instructions, str):
            instructions = [instructions] * images.shape[0]

        vecs = [self._next_action_vec(i, images[i], instructions[i]) for i in range(images.shape[0])]
        return torch.from_numpy(np.stack(vecs)).float().cuda()
