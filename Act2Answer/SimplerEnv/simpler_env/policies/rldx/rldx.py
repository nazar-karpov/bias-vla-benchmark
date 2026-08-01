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

RLDX server obs (embodiment `bridge_orig`, from processor_config.json) is a NESTED,
BATCHED, TEMPORAL dict:
  video.image_0 : uint8 (B, T=4, H, W, 3)  at delta_indices [-6,-4,-2,0]
  state.{end_effector_position(3), end_effector_rotation(3 euler), gripper_position(1)}
                : float32 (B, T=1, D)
  language.annotation.human.action.task_description : list[list[str]] (B, 1)
Server returns (action_dict, info); action keys (delta, chunk of 16):
  end_effector_position(3), end_effector_rotation(3 euler delta), gripper_close(1).

We keep a per-env frame ring buffer (last 7 frames) to satisfy the [-6,-4,-2,0]
video offsets, and a per-env action-chunk buffer (query once, pop one step at a
time). euler delta -> axangle; gripper_close -> widowx [-1,1] sticky.

Installed on the server as: SimplerEnv/.../policies/rldx/rldx.py (+ empty __init__).
"""
from __future__ import annotations

import io
from collections import deque

import cv2
import msgpack
import numpy as np
import torch
from transforms3d.euler import euler2axangle, mat2euler
from transforms3d.quaternions import quat2mat


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

    def __init__(self, host: str = "127.0.0.1", port: int = 20000, timeout_ms: int = 180000):
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


# ------------------------------------------------------------------- policy ---
_IMG_H, _IMG_W = 256, 256           # RLDX widowx image_size (rows, cols); official
                                    # SimplerEnv wrapper uses (256,256) square, not 320
# bridge ee-rotation frame (top-down), matches official WidowXBridgeEnv wrapper
_BRIDGE_DEFAULT_ROT = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
_VIDEO_KEY = "image_0"
_VIDEO_DELTAS = [-6, -4, -2, 0]     # bridge_orig video delta_indices (T=4)
_RING = 7                           # need frames back to t-6
_ACT_POS, _ACT_ROT, _ACT_GRIP = "end_effector_position", "end_effector_rotation", "gripper_close"


class RLDXInference:
    """RLDX-1 client policy with the SimplerEnv get_action(obs) contract."""

    def __init__(self, host: str = "127.0.0.1", port: int = 20000, policy_setup: str = "widowx_bridge"):
        assert policy_setup == "widowx_bridge", "RLDX Act2Answer client is wired for widowx (bridge_orig)"
        self.policy_setup = policy_setup
        self.client = _RLDXClient(host, port)
        self.client.reset()
        self._first_step: dict[int, bool] = {}   # per-env: is this the first query?
        self._frames: dict[int, deque] = {}     # per-env ring buffer of resized frames
        self._chunks: dict[int, deque] = {}      # per-env action-chunk buffer

    # ---- required by run.py's render loop -----------------------------------
    def prep_rollout(self):
        self.client.reset()
        self._first_step.clear()
        self._frames.clear(); self._chunks.clear()

    @staticmethod
    def _sid(env_i: int) -> str:
        return f"a2a-env-{env_i}"

    def reset(self, task_description=None):
        self.prep_rollout()

    # ---- obs packing (nested/batched/temporal, bridge_orig) ------------------
    def _video_stack(self, env_i: int, frame_hw3: np.ndarray) -> np.ndarray:
        """Push newest frame; return (T=4, H, W, 3) at deltas [-6,-4,-2,0]."""
        ring = self._frames.get(env_i)
        if ring is None:
            ring = deque(maxlen=_RING)
            self._frames[env_i] = ring
        ring.append(frame_hw3)
        # index from the newest: offset 0 = last, -2 = 2 steps back, etc.
        out = []
        for d in _VIDEO_DELTAS:            # d in {-6,-4,-2,0}
            idx = -1 + d                   # position from the end
            if -idx <= len(ring):
                out.append(ring[idx])
            else:
                out.append(ring[0])        # pad with the oldest available frame
        return np.stack(out).astype(np.uint8)

    @staticmethod
    def _state_from_proprio(proprio):
        """Build (eef_pos xyz, eef_rot euler top-down, gripper) from SimplerEnv
        proprio = agent eef_pos [x,y,z, quat_wxyz(4), gripper(1)], matching the
        official RLDX WidowXBridgeEnv wrapper (_process_observation)."""
        if proprio is None:
            return (np.zeros(3, np.float32), np.zeros(3, np.float32), 0.0)
        p = np.asarray(proprio, dtype=np.float64).reshape(-1)
        pos = p[0:3].astype(np.float32)
        rm = quat2mat(p[3:7])
        euler = np.asarray(mat2euler(rm @ _BRIDGE_DEFAULT_ROT.T), dtype=np.float32)
        gripper = float(p[7]) if p.shape[0] > 7 else 0.0
        return (pos, euler, gripper)

    def _pack_obs(self, env_i: int, image_hw3_uint8: np.ndarray, instruction: str,
                  proprio=None) -> dict:
        frame = cv2.resize(image_hw3_uint8, (_IMG_W, _IMG_H)).astype(np.uint8)  # (cols,rows)
        vid = self._video_stack(env_i, frame)[None]        # (B=1, T=4, H, W, 3)
        pos, euler, gripper = self._state_from_proprio(proprio)
        return {
            "video": {_VIDEO_KEY: vid},
            "state": {
                "end_effector_position": pos.reshape(1, 1, 3).astype(np.float32),
                "end_effector_rotation": euler.reshape(1, 1, 3).astype(np.float32),
                "gripper_position": np.array([[[gripper]]], dtype=np.float32),
            },
            "language": {"annotation.human.action.task_description": [[instruction]]},
        }

    def _postprocess_gripper(self, env_i: int, g_close: float) -> float:
        # RLDX model outputs `gripper_close` in [0,1]: 1=close(grasp), 0=open. It is
        # an ABSOLUTE gripper command, not a relative one, so the SimplerEnv-style
        # sticky machine is WRONG here — it latched a stale value for 15 steps and
        # inverted the phase (gripper opened exactly when the model wanted to hold
        # the grasp), so the cube slipped out on transport. Map directly to widowx
        # [-1,1] with a threshold (mirrors RLDX CALVIN eval: gripper_close>0 -> close).
        # widowx convention: +1 = close/grasp, -1 = open.
        return 1.0 if g_close > 0.5 else -1.0

    def _next_action_vec(self, env_i: int, image: np.ndarray, instruction: str,
                         proprio=None) -> np.ndarray:
        buf = self._chunks.get(env_i)
        if not buf:
            is_first = self._first_step.get(env_i, True)
            self._first_step[env_i] = False
            options = {"reset_memory": [is_first], "session_ids": [self._sid(env_i)]}
            action, _info = self.client.get_action(
                self._pack_obs(env_i, image, instruction, proprio), options=options
            )
            # server returns batched actions (B=1, chunk, D); drop the batch axis.
            pos = np.asarray(action[_ACT_POS], dtype=np.float64)
            rot = np.asarray(action[_ACT_ROT], dtype=np.float64)
            grip = np.asarray(action[_ACT_GRIP], dtype=np.float64)
            if pos.ndim == 3:   # (B, chunk, 3) -> (chunk, 3)
                pos, rot = pos[0], rot[0]
                grip = grip[0]
            pos = np.atleast_2d(pos)          # (chunk, 3)
            rot = np.atleast_2d(rot)          # (chunk, 3)
            grip = grip.reshape(-1)           # (chunk,)
            horizon = pos.shape[0]
            buf = deque()
            for t in range(horizon):
                buf.append((pos[t], rot[t], float(grip[t]) if t < len(grip) else float(grip[-1])))
            self._chunks[env_i] = buf
        else:
            # still advance the frame buffer even when replaying a cached chunk
            self._video_stack(env_i, cv2.resize(image, (_IMG_W, _IMG_H)).astype(np.uint8))
        pos, rot_euler, g = buf.popleft()

        axes, ang = euler2axangle(rot_euler[0], rot_euler[1], rot_euler[2])
        rot_axangle = axes * ang
        grip = self._postprocess_gripper(env_i, g)
        return np.concatenate([pos, rot_axangle, [grip]]).astype(np.float32)

    @torch.no_grad()
    def get_action(self, obs, deterministic: bool = True) -> torch.Tensor:
        images = obs["image"]
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        images = np.asarray(images, dtype=np.uint8)          # (B, H, W, 3)
        instructions = obs["task_description"]
        if isinstance(instructions, str):
            instructions = [instructions] * images.shape[0]

        # real proprio (agent eef_pos) per env, as the official RLDX wrapper uses;
        # feeding zeros here was the answer-rate killer. run.py passes obs['proprio'].
        proprio = obs.get("proprio")

        def _proprio_i(i):
            if proprio is None:
                return None
            pr = proprio[i]
            if isinstance(pr, dict):
                pr = pr.get("agent", pr)
                if isinstance(pr, dict):
                    pr = pr.get("eef_pos")
            return pr.cpu().numpy() if hasattr(pr, "cpu") else pr

        vecs = [self._next_action_vec(i, images[i], instructions[i], _proprio_i(i))
                for i in range(images.shape[0])]
        return torch.from_numpy(np.stack(vecs)).float().cuda()
